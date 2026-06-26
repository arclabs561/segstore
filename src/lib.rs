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
//! # v0 limitations
//!
//! - The WAL is append-only and never truncated, so it grows until a process
//!   restart re-bases it onto a fresh checkpoint. WAL rotation (a two-phase
//!   checkpoint that bounds the log) is a planned follow-up.
//! - Every `add`/`delete` flushes the WAL record to the directory. Per-op
//!   `fsync` hardening (on a real filesystem) is left to the `durability` writer
//!   flush policy; group-commit batching is a planned follow-up.

use std::collections::HashSet;
use std::hash::Hash;
use std::sync::Arc;

use durability::checkpoint::CheckpointFile;
use durability::recordlog::{RecordLogReadMode, RecordLogReader, RecordLogWriter};
use durability::{Directory, PersistenceResult};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

const WAL_PATH: &str = "segstore.wal";
const CKPT_PATH: &str = "segstore.ckpt";

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
    /// Total ops appended to the WAL.
    applied: u64,
    /// Ops covered by the last checkpoint (the WAL replay offset).
    ckpt_applied: u64,
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
        // Load the checkpoint snapshot if one exists.
        let (ckpt_applied, segments, mut tombstones): (u64, Vec<S::Segment>, HashSet<S::Id>) =
            if dir.exists(CKPT_PATH) {
                let ckpt = CheckpointFile::new(dir.clone());
                let (applied, snap): (u64, Snapshot<S::Id, S::Segment>) =
                    ckpt.read_postcard(CKPT_PATH)?;
                (
                    applied,
                    snap.segments,
                    snap.tombstones.into_iter().collect(),
                )
            } else {
                (0, Vec::new(), HashSet::new())
            };

        // Replay the WAL operations recorded after the checkpoint.
        let mut applied = ckpt_applied;
        let mut buffer: Vec<(S::Id, S::Item)> = Vec::new();
        if dir.exists(WAL_PATH) {
            let reader = RecordLogReader::new(dir.clone(), WAL_PATH);
            let ops: Vec<Op<S::Id, S::Item>> =
                reader.read_all_postcard(RecordLogReadMode::BestEffort)?;
            applied = ops.len() as u64;
            for op in ops.into_iter().skip(ckpt_applied as usize) {
                apply(&mut buffer, &mut tombstones, op);
            }
        }

        let wal = RecordLogWriter::new(dir.clone(), WAL_PATH);
        Ok(Self {
            store,
            dir,
            buffer,
            segments,
            tombstones,
            wal,
            applied,
            ckpt_applied,
            flush_threshold,
        })
    }

    /// Add (or re-add) an item. Durably logged before it becomes visible.
    pub fn add(&mut self, id: S::Id, item: S::Item) -> PersistenceResult<()> {
        self.wal
            .append_postcard(&Op::Add(id.clone(), item.clone()))?;
        self.wal.flush()?;
        self.applied += 1;
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
        self.wal.flush()?;
        self.applied += 1;
        apply(&mut self.buffer, &mut self.tombstones, Op::Delete(id));
        Ok(())
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

    /// Snapshot the current segments + tombstones, advancing the WAL replay
    /// offset so recovery starts from here.
    pub fn checkpoint(&mut self) -> PersistenceResult<()> {
        self.flush_buffer();
        let snap = Snapshot {
            segments: self.segments.clone(),
            tombstones: self.tombstones.iter().cloned().collect::<Vec<_>>(),
        };
        let ckpt = CheckpointFile::new(self.dir.clone());
        ckpt.write_postcard(CKPT_PATH, self.applied, &snap)?;
        self.ckpt_applied = self.applied;
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
        let wal = root.join(WAL_PATH);
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
}
