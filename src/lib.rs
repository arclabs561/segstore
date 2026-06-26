//! Generic durable segmented store.
//!
//! `segstore` provides the LSM-style lifecycle shared by updatable indexes,
//! immutable segments plus tombstone deletes plus a write-ahead log plus
//! checkpoint and compaction, on top of [`durability`]. It is generic over the
//! segment payload: a consumer implements [`Store`] to say how a batch of items
//! becomes a segment and how segments merge during compaction, and `segstore`
//! owns the durability and the orchestration.
//!
//! The split mirrors the rest of the retrieval stack: a storage substrate
//! (`postings`, a graph index) keeps its own representation while delegating the
//! mutable-and-persistent machinery here, the same way it already delegates the
//! raw WAL/checkpoint primitives to [`durability`].
//!
//! # What it owns vs what the consumer owns
//!
//! - `segstore` owns: the append-only WAL of operations, the in-memory buffer of
//!   not-yet-flushed adds, the ordered list of immutable segments, the tombstone
//!   set, checkpoint snapshots, and crash recovery (replay the WAL past the last
//!   checkpoint).
//! - The consumer owns: the segment representation (an opaque `Serialize` type),
//!   how a batch of live items becomes one segment ([`Store::build_segment`]),
//!   how segments merge while dropping tombstoned ids ([`Store::merge_segments`]),
//!   and querying (iterate [`SegmentedStore::segments`] + [`SegmentedStore::buffer`],
//!   filtering with [`SegmentedStore::is_live`]).
//!
//! # Example
//!
//! ```
//! use segstore::{SegmentedStore, Store};
//! use durability::MemoryDirectory;
//!
//! // A trivial key-value store: a segment is just a batch of (id, item) pairs.
//! struct Kv;
//! impl Store for Kv {
//!     type Id = u32;
//!     type Item = String;
//!     type Segment = Vec<(u32, String)>;
//!     fn build_segment(&self, batch: &[(u32, String)]) -> Vec<(u32, String)> {
//!         batch.to_vec()
//!     }
//!     fn merge_segments(
//!         &self,
//!         segs: &[Vec<(u32, String)>],
//!         live: &dyn Fn(&u32) -> bool,
//!     ) -> Vec<(u32, String)> {
//!         segs.iter().flatten().filter(|(id, _)| live(id)).cloned().collect()
//!     }
//! }
//!
//! let dir = MemoryDirectory::arc();
//! let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
//! s.add(1, "a".into()).unwrap();
//! s.add(2, "b".into()).unwrap();
//! s.delete(2).unwrap();
//! assert!(s.is_live(&1) && !s.is_live(&2));
//! ```
//!
//! # Durability
//!
//! The WAL is rotated by epoch: a checkpoint snapshots the segments + tombstones
//! durably, then a fresh WAL generation (`segstore.wal.<epoch>`) is started and
//! the old one deleted, so the log never grows past one checkpoint interval. The
//! checkpoint records the epoch it covers; recovery replays only that epoch's
//! WAL, so a stale WAL left by a crash mid-rotation is simply never read (no
//! duplicates, no loss). Checkpoints publish atomically (via
//! `Directory::atomic_write`, CRC-checked), and on a filesystem backend also
//! pass a power-loss barrier (fsync of the file and parent dir).
//!
//! [`SyncPolicy`] chooses the WAL write durability: the default [`SyncPolicy::Flush`]
//! is a visibility boundary (userspace -> OS), while [`SyncPolicy::Fsync`] syncs
//! every record to stable storage (filesystem backend only).
//!
//! On-disk format note: segstore 0.2 changed the WAL layout (epoch-suffixed
//! files, with the epoch recorded in the checkpoint) from 0.1's single
//! unsuffixed `segstore.wal`. A 0.1 store is detected and rejected with a clear
//! error rather than misread.

use std::collections::HashSet;
use std::hash::Hash;
use std::sync::Arc;

use durability::checkpoint::CheckpointFile;
use durability::recordlog::{RecordLogReadMode, RecordLogReader, RecordLogWriter};
use durability::{Directory, PersistenceError, PersistenceResult};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// Prefix for epoch-suffixed WAL files (`segstore.wal.<epoch>`).
const WAL_PREFIX: &str = "segstore.wal.";
/// The single unsuffixed WAL of the 0.1 on-disk format; its presence flags a
/// store written by segstore 0.1, which 0.2 cannot read.
const LEGACY_WAL_PATH: &str = "segstore.wal";
const CKPT_PATH: &str = "segstore.ckpt";

/// The WAL file path for a given epoch.
fn wal_path(epoch: u64) -> String {
    format!("{WAL_PREFIX}{epoch}")
}

/// How durable each WAL write is before [`SegmentedStore::add`] / [`SegmentedStore::delete`]
/// returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPolicy {
    /// Flush each record to the underlying writer (userspace -> OS). The default;
    /// works on every [`Directory`] backend.
    Flush,
    /// fsync each record to stable storage. Stronger (survives power loss) and
    /// slower; requires a filesystem-backed [`Directory`] (one that provides
    /// `file_path()`), else [`SegmentedStore::open_with_sync`] errors.
    Fsync,
}

/// The consumer-defined payload model: how items batch into a segment and how
/// segments merge during compaction.
///
/// `segstore` never inspects a segment's contents; it only stores, snapshots,
/// and hands segments back to these two methods, so the representation is
/// entirely the consumer's (posting lists, a graph delta, a vector block, ...).
pub trait Store {
    /// Stable identity of an item, used for tombstones and WAL replay.
    type Id: Clone + Eq + Hash + Serialize + DeserializeOwned;
    /// The per-item payload carried in the WAL until it is flushed to a segment.
    type Item: Clone + Serialize + DeserializeOwned;
    /// An immutable batch of items. Opaque to `segstore`.
    type Segment: Clone + Serialize + DeserializeOwned;

    /// Build one immutable segment from a batch of live `(id, item)` pairs.
    fn build_segment(&self, batch: &[(Self::Id, Self::Item)]) -> Self::Segment;

    /// Merge `segments` into one during compaction, keeping only ids for which
    /// `live(id)` is true (i.e. dropping tombstoned ids).
    fn merge_segments(
        &self,
        segments: &[Self::Segment],
        live: &dyn Fn(&Self::Id) -> bool,
    ) -> Self::Segment;
}

/// One write-ahead-log operation.
#[derive(Serialize, Deserialize)]
enum Op<Id, Item> {
    Add(Id, Item),
    Delete(Id),
}

/// The persisted checkpoint snapshot (segments + tombstones; the buffer is
/// always flushed before a checkpoint, so it is never part of the snapshot).
#[derive(Serialize, Deserialize)]
struct Snapshot<Id, Seg> {
    segments: Vec<Seg>,
    tombstones: Vec<Id>,
}

/// A generic, durable, segmented mutable store.
pub struct SegmentedStore<S: Store> {
    store: S,
    dir: Arc<dyn Directory>,
    /// Live adds not yet flushed into a segment.
    buffer: Vec<(S::Id, S::Item)>,
    /// Immutable segments, oldest first.
    segments: Vec<S::Segment>,
    /// Logically-deleted ids.
    tombstones: HashSet<S::Id>,
    wal: RecordLogWriter,
    /// Current WAL generation; the live WAL is `segstore.wal.<epoch>` and the
    /// checkpoint records the epoch it covers.
    epoch: u64,
    /// Per-write WAL durability.
    sync: SyncPolicy,
    /// Buffer size that triggers a flush into a new segment.
    flush_threshold: usize,
}

impl<S: Store> SegmentedStore<S> {
    /// Open (or create) a store backed by `dir`, recovering any prior state from
    /// the checkpoint plus the write-ahead log.
    ///
    /// `flush_threshold` is the buffered-add count that triggers a new segment.
    pub fn open(
        dir: Arc<dyn Directory>,
        store: S,
        flush_threshold: usize,
    ) -> PersistenceResult<Self> {
        Self::open_with_sync(dir, store, flush_threshold, SyncPolicy::Flush)
    }

    /// Like [`Self::open`], but choosing the per-write WAL durability.
    ///
    /// [`SyncPolicy::Fsync`] requires a filesystem-backed `Directory`; opening
    /// with it on a backend that lacks `file_path()` (e.g. an in-memory
    /// directory) returns an error rather than silently degrading to a flush.
    pub fn open_with_sync(
        dir: Arc<dyn Directory>,
        store: S,
        flush_threshold: usize,
        sync: SyncPolicy,
    ) -> PersistenceResult<Self> {
        if sync == SyncPolicy::Fsync && dir.file_path(CKPT_PATH).is_none() {
            return Err(PersistenceError::InvalidConfig(
                "SyncPolicy::Fsync requires a filesystem-backed Directory".into(),
            ));
        }
        // A 0.1 store wrote a single unsuffixed WAL; 0.2 changed the format.
        // Reject it explicitly rather than misread its ops as an epoch.
        if dir.exists(LEGACY_WAL_PATH) {
            return Err(PersistenceError::InvalidConfig(
                "segstore 0.1 on-disk store detected (unsuffixed segstore.wal); the 0.2 \
                 on-disk format is epoch-suffixed and cannot read it"
                    .into(),
            ));
        }

        // Load the checkpoint snapshot if one exists. The checkpoint records the
        // WAL epoch it covers; recovery replays only that epoch's WAL.
        let (epoch, segments, mut tombstones): (u64, Vec<S::Segment>, HashSet<S::Id>) =
            if dir.exists(CKPT_PATH) {
                let ckpt = CheckpointFile::new(dir.clone());
                let (epoch, snap): (u64, Snapshot<S::Id, S::Segment>) =
                    ckpt.read_postcard(CKPT_PATH)?;
                (epoch, snap.segments, snap.tombstones.into_iter().collect())
            } else {
                (0, Vec::new(), HashSet::new())
            };

        // Replay the current epoch's WAL in full. It holds exactly the ops since
        // the checkpoint, so every record is applied (no skip offset).
        let mut buffer: Vec<(S::Id, S::Item)> = Vec::new();
        let live_wal = wal_path(epoch);
        if dir.exists(&live_wal) {
            let reader = RecordLogReader::new(dir.clone(), live_wal.clone());
            let ops: Vec<Op<S::Id, S::Item>> =
                reader.read_all_postcard(RecordLogReadMode::BestEffort)?;
            for op in ops {
                apply(&mut buffer, &mut tombstones, op);
            }
        }

        // Best-effort GC of stale WAL generations a crash may have left behind
        // (any `segstore.wal.<k>` with k != epoch is superseded by the checkpoint).
        if let Ok(names) = dir.list_dir("") {
            for name in names {
                if let Some(k) = name
                    .strip_prefix(WAL_PREFIX)
                    .and_then(|s| s.parse::<u64>().ok())
                {
                    if k != epoch {
                        let _ = dir.delete(&name);
                    }
                }
            }
        }

        let wal = RecordLogWriter::new(dir.clone(), live_wal);
        Ok(Self {
            store,
            dir,
            buffer,
            segments,
            tombstones,
            wal,
            epoch,
            sync,
            flush_threshold,
        })
    }

    /// Add (or re-add) an item. Durably logged before it becomes visible.
    pub fn add(&mut self, id: S::Id, item: S::Item) -> PersistenceResult<()> {
        self.wal
            .append_postcard(&Op::Add(id.clone(), item.clone()))?;
        self.sync_wal()?;
        apply(&mut self.buffer, &mut self.tombstones, Op::Add(id, item));
        if self.buffer.len() >= self.flush_threshold {
            self.flush_buffer();
        }
        Ok(())
    }

    /// Tombstone an item. Durably logged before it takes effect.
    pub fn delete(&mut self, id: S::Id) -> PersistenceResult<()> {
        self.wal
            .append_postcard::<Op<S::Id, S::Item>>(&Op::Delete(id.clone()))?;
        self.sync_wal()?;
        apply(&mut self.buffer, &mut self.tombstones, Op::Delete(id));
        Ok(())
    }

    /// Make the just-appended WAL record durable per the [`SyncPolicy`].
    fn sync_wal(&mut self) -> PersistenceResult<()> {
        match self.sync {
            SyncPolicy::Flush => self.wal.flush(),
            SyncPolicy::Fsync => self.wal.flush_and_sync(),
        }
    }

    /// Drain the buffer into a fresh immutable segment (no-op when empty).
    fn flush_buffer(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        let seg = self.store.build_segment(&self.buffer);
        self.segments.push(seg);
        self.buffer.clear();
    }

    /// Merge all segments into one, dropping tombstoned ids, then checkpoint.
    pub fn compact(&mut self) -> PersistenceResult<()> {
        self.flush_buffer();
        if !self.segments.is_empty() {
            let tombstones = std::mem::take(&mut self.tombstones);
            let merged = self
                .store
                .merge_segments(&self.segments, &|id| !tombstones.contains(id));
            self.segments = vec![merged];
            // Tombstones are now physically gone from the merged segment.
        }
        self.checkpoint()
    }

    /// Snapshot the current segments + tombstones durably, then rotate the WAL:
    /// start a fresh epoch and delete the old log so it cannot grow unbounded.
    ///
    /// The checkpoint is the commit point. It records the *new* epoch, so once it
    /// is durable, recovery replays only the new (initially empty) WAL; the old
    /// WAL is superseded. A crash after the checkpoint but before the old WAL is
    /// deleted is safe: recovery never reads the old epoch, and the stale file is
    /// GC'd on the next open.
    pub fn checkpoint(&mut self) -> PersistenceResult<()> {
        self.flush_buffer();
        let new_epoch = self.epoch + 1;
        let snap = Snapshot {
            segments: self.segments.clone(),
            tombstones: self.tombstones.iter().cloned().collect::<Vec<_>>(),
        };
        let ckpt = CheckpointFile::new(self.dir.clone());
        // The checkpoint publishes atomically (atomic_write + CRC). On a
        // filesystem backend, also pass a power-loss barrier; in-memory backends
        // have no such barrier, so fall back to the atomic-only write.
        if self.dir.file_path(CKPT_PATH).is_some() {
            ckpt.write_postcard_durable(CKPT_PATH, new_epoch, &snap)?;
        } else {
            ckpt.write_postcard(CKPT_PATH, new_epoch, &snap)?;
        }

        // Past the commit point: start the new WAL generation and drop the old.
        let old_epoch = self.epoch;
        self.wal = RecordLogWriter::new(self.dir.clone(), wal_path(new_epoch));
        self.epoch = new_epoch;
        let _ = self.dir.delete(&wal_path(old_epoch));
        Ok(())
    }

    /// The immutable segments, oldest first. Query these plus [`Self::buffer`],
    /// filtering with [`Self::is_live`].
    pub fn segments(&self) -> &[S::Segment] {
        &self.segments
    }

    /// Live adds not yet flushed into a segment.
    pub fn buffer(&self) -> &[(S::Id, S::Item)] {
        &self.buffer
    }

    /// Whether `id` is not tombstoned.
    pub fn is_live(&self, id: &S::Id) -> bool {
        !self.tombstones.contains(id)
    }

    /// Number of immutable segments.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Number of tombstoned ids.
    pub fn tombstone_count(&self) -> usize {
        self.tombstones.len()
    }
}

/// Apply one operation to the in-memory buffer + tombstone set. Shared by live
/// writes and WAL replay so the two paths cannot diverge.
fn apply<Id: Clone + Eq + Hash, Item>(
    buffer: &mut Vec<(Id, Item)>,
    tombstones: &mut HashSet<Id>,
    op: Op<Id, Item>,
) {
    match op {
        Op::Add(id, item) => {
            // A re-add revives a previously-deleted id.
            tombstones.remove(&id);
            buffer.push((id, item));
        }
        Op::Delete(id) => {
            buffer.retain(|(bid, _)| bid != &id);
            tombstones.insert(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use durability::MemoryDirectory;

    struct Kv;
    impl Store for Kv {
        type Id = u32;
        type Item = String;
        type Segment = Vec<(u32, String)>;
        fn build_segment(&self, batch: &[(u32, String)]) -> Vec<(u32, String)> {
            batch.to_vec()
        }
        fn merge_segments(
            &self,
            segs: &[Vec<(u32, String)>],
            live: &dyn Fn(&u32) -> bool,
        ) -> Vec<(u32, String)> {
            segs.iter()
                .flatten()
                .filter(|(id, _)| live(id))
                .cloned()
                .collect()
        }
    }

    /// Collect the live `(id, item)` set across segments + buffer.
    fn live_set(s: &SegmentedStore<Kv>) -> Vec<(u32, String)> {
        let mut out: Vec<(u32, String)> = Vec::new();
        for seg in s.segments() {
            for (id, it) in seg {
                if s.is_live(id) {
                    out.push((*id, it.clone()));
                }
            }
        }
        for (id, it) in s.buffer() {
            if s.is_live(id) {
                out.push((*id, it.clone()));
            }
        }
        out.sort();
        out
    }

    #[test]
    fn buffer_flushes_into_a_segment_at_threshold() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
        s.add(1, "a".into()).unwrap();
        assert_eq!(
            s.segment_count(),
            0,
            "one add stays buffered below threshold"
        );
        s.add(2, "b".into()).unwrap();
        assert_eq!(
            s.segment_count(),
            1,
            "second add hits threshold 2 and flushes"
        );
        assert!(s.buffer().is_empty());
    }

    #[test]
    fn delete_tombstones_and_hides_from_live_set() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
        s.add(1, "a".into()).unwrap();
        s.add(2, "b".into()).unwrap();
        s.delete(2).unwrap();
        assert!(s.is_live(&1) && !s.is_live(&2));
        assert_eq!(live_set(&s), vec![(1, "a".into())]);
    }

    #[test]
    fn compaction_physically_drops_tombstoned_ids() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
        s.add(1, "a".into()).unwrap();
        s.add(2, "b".into()).unwrap();
        s.add(3, "c".into()).unwrap();
        s.delete(2).unwrap();
        s.compact().unwrap();
        assert_eq!(s.segment_count(), 1, "compaction merges into one segment");
        assert_eq!(s.tombstone_count(), 0, "compaction purges tombstones");
        // The merged segment no longer carries id 2 at all.
        let flat: Vec<u32> = s.segments()[0].iter().map(|(id, _)| *id).collect();
        assert!(!flat.contains(&2));
        assert_eq!(live_set(&s), vec![(1, "a".into()), (3, "c".into())]);
    }

    #[test]
    fn recovers_from_checkpoint_plus_wal_replay() {
        let dir = MemoryDirectory::arc();
        {
            let mut s = SegmentedStore::open(dir.clone(), Kv, 2).unwrap();
            s.add(1, "a".into()).unwrap();
            s.add(2, "b".into()).unwrap(); // flushes a segment
            s.checkpoint().unwrap(); // checkpoint covers ids 1,2
            s.add(3, "c".into()).unwrap(); // post-checkpoint, only in the WAL
            s.delete(2).unwrap(); // post-checkpoint tombstone
        } // drop without a second checkpoint: 3 and delete(2) live only in the WAL

        // Reopen: checkpoint gives {1,2}; WAL replay applies add(3) + delete(2).
        let s2 = SegmentedStore::open(dir, Kv, 2).unwrap();
        assert_eq!(live_set(&s2), vec![(1, "a".into()), (3, "c".into())]);
        assert!(!s2.is_live(&2));
    }

    /// A unique scratch directory under the OS temp dir (no tempfile dep).
    fn temp_root(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("segstore-test-{}-{}", std::process::id(), tag));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn recovers_on_real_disk_via_fs_directory() {
        let root = temp_root("fs-recover");
        {
            let dir = durability::FsDirectory::arc(&root).unwrap();
            let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
            s.add(1, "a".into()).unwrap();
            s.add(2, "b".into()).unwrap(); // flushes a segment
            s.checkpoint().unwrap();
            s.add(3, "c".into()).unwrap(); // post-checkpoint, WAL only
            s.delete(2).unwrap();
        } // process-style drop: nothing in memory, only the on-disk WAL + checkpoint

        // A fresh FsDirectory on the same root must reconstruct the live set.
        let dir = durability::FsDirectory::arc(&root).unwrap();
        let s2 = SegmentedStore::open(dir, Kv, 2).unwrap();
        assert_eq!(live_set(&s2), vec![(1, "a".into()), (3, "c".into())]);
        assert!(!s2.is_live(&2));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn torn_wal_tail_is_dropped_not_fatal() {
        let root = temp_root("torn-wal");
        // Threshold high enough that every op stays in the WAL (no checkpoint,
        // no segment files), so the WAL is the single source of truth.
        {
            let dir = durability::FsDirectory::arc(&root).unwrap();
            let mut s = SegmentedStore::open(dir, Kv, 100).unwrap();
            s.add(1, "aaaa".into()).unwrap();
            s.add(2, "bbbb".into()).unwrap();
            s.add(3, "cccc".into()).unwrap();
            s.add(4, "dddd".into()).unwrap(); // this record's tail gets corrupted
        }

        // Corrupt the final WAL record by chopping bytes off the end of the file.
        // postcard records are length-prefixed, so a truncated final payload makes
        // the BestEffort reader stop before it rather than decode garbage.
        let wal = root.join(wal_path(0));
        let bytes = std::fs::read(&wal).unwrap();
        assert!(bytes.len() > 3, "WAL should hold four records");
        std::fs::write(&wal, &bytes[..bytes.len() - 3]).unwrap();

        // Recovery must succeed (no panic, no Err) and recover the intact prefix.
        let dir = durability::FsDirectory::arc(&root).unwrap();
        let s2 = SegmentedStore::open(dir, Kv, 100).unwrap();
        // id 4 never made it into the store: the torn final record was dropped,
        // so the live set is exactly the intact prefix. (`is_live` only reports
        // tombstone state, so it is not the right probe for "never added".)
        assert_eq!(
            live_set(&s2),
            vec![(1, "aaaa".into()), (2, "bbbb".into()), (3, "cccc".into())],
            "the torn final record is dropped; the prefix survives"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// Each checkpoint rotates the WAL, so at most one generation file exists.
    #[test]
    fn wal_is_bounded_to_one_generation_across_checkpoints() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir.clone(), Kv, 100).unwrap();
        for i in 0..10u32 {
            s.add(i, format!("v{i}")).unwrap();
            s.checkpoint().unwrap();
        }
        let wals: Vec<String> = dir
            .list_dir("")
            .unwrap()
            .into_iter()
            .filter(|n| n.starts_with(WAL_PREFIX))
            .collect();
        assert!(
            wals.len() <= 1,
            "rotation bounds the WAL to the current generation, got {wals:?}"
        );
    }

    /// A stale WAL from an earlier epoch (the crash-mid-rotation leftover) is
    /// never replayed and is garbage-collected on open.
    #[test]
    fn stale_wal_generation_is_ignored_and_gced() {
        let dir = MemoryDirectory::arc();
        {
            let mut s = SegmentedStore::open(dir.clone(), Kv, 100).unwrap();
            s.add(1, "a".into()).unwrap();
            s.add(2, "b".into()).unwrap();
            s.add(3, "c".into()).unwrap();
            s.checkpoint().unwrap(); // epoch 0 -> 1; snapshot holds {1,2,3}
        }
        // Forge a stale wal.0 (the generation a clean rotation deleted) carrying a
        // ghost op, as a crash mid-rotation would leave behind.
        {
            let mut ghost = RecordLogWriter::new(dir.clone(), wal_path(0));
            ghost
                .append_postcard(&Op::Add(99u32, "ghost".to_string()))
                .unwrap();
            ghost.flush().unwrap();
        }
        assert!(dir.exists(&wal_path(0)), "the stale WAL exists pre-open");

        let s2 = SegmentedStore::open(dir.clone(), Kv, 100).unwrap();
        assert_eq!(
            live_set(&s2),
            vec![(1, "a".into()), (2, "b".into()), (3, "c".into())],
            "the ghost op in the stale epoch is never replayed"
        );
        assert!(
            !dir.exists(&wal_path(0)),
            "the stale WAL generation is GC'd on open"
        );
    }

    #[test]
    fn fsync_policy_round_trips_on_filesystem() {
        let root = temp_root("fsync");
        {
            let dir = durability::FsDirectory::arc(&root).unwrap();
            let mut s = SegmentedStore::open_with_sync(dir, Kv, 2, SyncPolicy::Fsync).unwrap();
            s.add(1, "a".into()).unwrap();
            s.add(2, "b".into()).unwrap(); // flush
            s.checkpoint().unwrap();
            s.add(3, "c".into()).unwrap();
            s.delete(2).unwrap();
        }
        let dir = durability::FsDirectory::arc(&root).unwrap();
        let s2 = SegmentedStore::open_with_sync(dir, Kv, 2, SyncPolicy::Fsync).unwrap();
        assert_eq!(live_set(&s2), vec![(1, "a".into()), (3, "c".into())]);
        assert!(!s2.is_live(&2));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fsync_policy_rejected_without_filesystem_backend() {
        let dir = MemoryDirectory::arc();
        let err = SegmentedStore::open_with_sync(dir, Kv, 2, SyncPolicy::Fsync);
        assert!(
            matches!(err, Err(PersistenceError::InvalidConfig(_))),
            "Fsync on an in-memory directory is a config error"
        );
    }

    #[test]
    fn legacy_v01_unsuffixed_wal_is_rejected() {
        let dir = MemoryDirectory::arc();
        // Simulate a 0.1 store: a single unsuffixed WAL file.
        {
            let mut legacy = RecordLogWriter::new(dir.clone(), LEGACY_WAL_PATH);
            legacy
                .append_postcard(&Op::Add(1u32, "x".to_string()))
                .unwrap();
            legacy.flush().unwrap();
        }
        let err = SegmentedStore::open(dir, Kv, 2);
        assert!(
            matches!(err, Err(PersistenceError::InvalidConfig(_))),
            "a 0.1 on-disk store is rejected, not misread"
        );
    }
}
