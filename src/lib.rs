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
//!     fn segment_len(&self, seg: &Vec<(u32, String)>) -> usize { seg.len() }
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
//! Recovery is *point-in-time*: the WAL is read best-effort, so a torn final record
//! (the expected damage from a crash mid-write) is truncated and the consistent
//! prefix recovered. This is the same default as RocksDB's `kPointInTimeRecovery`.
//! It does NOT detect corruption *beyond* the torn tail (a bit-flip of an
//! already-committed interior record): each record is CRC-checked, so such a record
//! is never decoded as garbage, but best-effort read stops at it and drops the rest.
//! Media-rot detection (a strict "fail on any corruption" mode) is out of scope for
//! this crash-consistency WAL; a consumer needing it should use checksummed storage.
//!
//! On-disk format note: segstore 0.2 changed the WAL layout (epoch-suffixed
//! files, with the epoch recorded in the checkpoint) from 0.1's single
//! unsuffixed `segstore.wal`. A 0.1 store is detected and rejected with a clear
//! error rather than misread.

use std::cmp::Ordering;
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
    /// `file_path()`), else [`SegmentedStore::open_with_options`] errors.
    Fsync,
}

/// Size-tiered compaction tuning. Defaults follow Cassandra's SizeTieredCompactionStrategy
/// (`min_threshold = 4`, `max_threshold = 32`, bucket band `0.5x .. 1.5x`) and Lucene's
/// `TieredMergePolicy` (a max merged-segment cap, above half of which a segment is frozen
/// out of further tier merges).
///
/// The size metric is item count ([`Store::segment_len`]); these are item counts, not
/// bytes. Because a query scans every segment, the per-tier segment count is paid on every
/// read, so `min_merge` is kept small.
#[derive(Debug, Clone, Copy)]
pub struct TierConfig {
    /// Minimum segments in a size bucket before it is eligible to merge (Cassandra
    /// `min_threshold`). 2 degenerates tiering into leveling, so 4 is the practical floor.
    pub min_merge: usize,
    /// Maximum segments merged in one job (Cassandra `max_threshold`), bounding per-merge
    /// cost and pause.
    pub max_merge: usize,
    /// Lower size-band multiplier for bucket membership (Cassandra `bucket_low`).
    pub bucket_low: f64,
    /// Upper size-band multiplier for bucket membership (Cassandra `bucket_high`).
    pub bucket_high: f64,
    /// Cap on a merged segment's item count. A segment larger than half of this is frozen
    /// out of tier merges (Lucene's rule), so the largest segment is never rewritten by
    /// tiering, only by a full [`SegmentedStore::compact`]. Without this, top-tier merges
    /// rewrite the whole dataset (O(N^2) write amplification).
    pub max_merged_len: usize,
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            min_merge: 4,
            max_merge: 32,
            bucket_low: 0.5,
            bucket_high: 1.5,
            max_merged_len: 10_000_000,
        }
    }
}

/// What a compaction did, for the consumer to track merge cost vs corpus size (the signal
/// that a multi-segment store is outgrowing the search-all-segments model).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactionStats {
    /// Number of merge operations performed.
    pub merges: usize,
    /// Segment count before the compaction.
    pub segments_before: usize,
    /// Segment count after the compaction.
    pub segments_after: usize,
    /// Total items written into the merged output segment(s).
    pub items_merged: usize,
}

/// Options for opening a [`SegmentedStore`].
#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// Buffered-add count that seals a new segment.
    pub flush_threshold: usize,
    /// Per-write WAL durability.
    pub sync: SyncPolicy,
    /// Size-tiered compaction tuning (used by [`SegmentedStore::compact_tiers`]).
    pub tiering: TierConfig,
    /// When true, [`SegmentedStore::add`] runs `compact_tiers` after sealing a segment if a
    /// bucket is eligible, on the calling thread. Default false: the consumer drives
    /// `compact_tiers` when convenient (e.g. on a background thread), so the merge latency
    /// never lands on the ingest hot path.
    pub auto_compact: bool,
}

impl Options {
    /// Options with the given flush threshold and defaults for the rest (`Flush` sync,
    /// default tiering, no auto-compaction).
    pub fn new(flush_threshold: usize) -> Self {
        Self {
            flush_threshold,
            sync: SyncPolicy::Flush,
            tiering: TierConfig::default(),
            auto_compact: false,
        }
    }
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

    /// The number of items in `segment`. This is the size metric size-tiered
    /// compaction groups segments by, so a count of *live* items (after any
    /// tombstone drop a merge applied) is the right answer. For the common
    /// `Vec`-backed segment this is just `segment.len()`.
    fn segment_len(&self, segment: &Self::Segment) -> usize;

    /// The number of items in `segment` for which `live` is true (i.e. excluding
    /// tombstoned ids), if the consumer can compute it cheaply. Returning `Some`
    /// enables space-amplification reporting ([`SegmentedStore::space_amplification`])
    /// and tombstone-reclaiming compaction ([`SegmentedStore::reclaim_tombstones`]).
    /// The default `None` disables both; the size-tiered path does not need it.
    /// For a `Vec`-backed segment: `Some(segment.iter().filter(|(id, _)| live(id)).count())`.
    fn live_len(
        &self,
        _segment: &Self::Segment,
        _live: &dyn Fn(&Self::Id) -> bool,
    ) -> Option<usize> {
        None
    }
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

/// The reader-visible published state: the segments and tombstones as of the last
/// mutation, behind `Arc`s so a [`View`] is a cheap, consistent snapshot.
struct PubState<S: Store> {
    segments: Arc<Vec<Arc<S::Segment>>>,
    tombstones: Arc<HashSet<S::Id>>,
}

/// A consistent, point-in-time read view of the store, independent of the writer.
///
/// A [`Reader`] hands these out; holding one keeps its segments alive for the whole
/// query even as the writer adds, deletes, or compacts concurrently (single-writer,
/// many-readers, like Lucene's `SearcherManager` / Tantivy's `Searcher`). The view
/// reflects state as of the last *sealed* segment + tombstones; adds still buffered
/// in the writer become visible after the next flush.
pub struct View<S: Store> {
    segments: Arc<Vec<Arc<S::Segment>>>,
    tombstones: Arc<HashSet<S::Id>>,
}

impl<S: Store> View<S> {
    /// The snapshot's immutable segments, oldest first. Query these (each derefs to
    /// `S::Segment`), filtering with [`Self::is_live`].
    pub fn segments(&self) -> &[Arc<S::Segment>] {
        &self.segments
    }

    /// Whether `id` is not tombstoned in this snapshot.
    pub fn is_live(&self, id: &S::Id) -> bool {
        !self.tombstones.contains(id)
    }

    /// Number of segments in this snapshot.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }
}

/// A cloneable, thread-safe handle for concurrent snapshot reads while the writer
/// mutates. Obtain via [`SegmentedStore::reader`]; clone freely across threads.
pub struct Reader<S: Store> {
    published: Arc<std::sync::RwLock<Arc<PubState<S>>>>,
}

impl<S: Store> Clone for Reader<S> {
    fn clone(&self) -> Self {
        Reader {
            published: self.published.clone(),
        }
    }
}

impl<S: Store> Reader<S> {
    /// Take a consistent [`View`] of the store as of the last published mutation.
    /// Cheap (a couple of `Arc` clones under a brief read lock); the view is then
    /// held lock-free for the query's duration.
    pub fn view(&self) -> View<S> {
        let state = self.published.read().unwrap().clone();
        View {
            segments: state.segments.clone(),
            tombstones: state.tombstones.clone(),
        }
    }
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
    /// Published snapshot for concurrent readers (rebuilt on each mutation).
    published: Arc<std::sync::RwLock<Arc<PubState<S>>>>,
    wal: RecordLogWriter,
    /// Current WAL generation; the live WAL is `segstore.wal.<epoch>` and the
    /// checkpoint records the epoch it covers.
    epoch: u64,
    /// Per-write WAL durability.
    sync: SyncPolicy,
    /// Buffer size that triggers a flush into a new segment.
    flush_threshold: usize,
    /// Size-tiered compaction tuning.
    tiering: TierConfig,
    /// Whether `add` auto-runs `compact_tiers` when a bucket is eligible.
    auto_compact: bool,
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
        Self::open_with_options(dir, store, Options::new(flush_threshold))
    }

    /// Like [`Self::open`], but with full [`Options`] (per-write durability,
    /// size-tiered compaction tuning, auto-compaction).
    ///
    /// [`SyncPolicy::Fsync`] requires a filesystem-backed `Directory`; opening
    /// with it on a backend that lacks `file_path()` (e.g. an in-memory
    /// directory) returns an error rather than silently degrading to a flush.
    pub fn open_with_options(
        dir: Arc<dyn Directory>,
        store: S,
        opts: Options,
    ) -> PersistenceResult<Self> {
        let Options {
            flush_threshold,
            sync,
            tiering,
            auto_compact,
        } = opts;
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
        let published = Arc::new(std::sync::RwLock::new(Arc::new(PubState {
            segments: Arc::new(segments.iter().map(|s| Arc::new(s.clone())).collect()),
            tombstones: Arc::new(tombstones.clone()),
        })));
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
            tiering,
            auto_compact,
            published,
        })
    }

    /// A cloneable handle for concurrent snapshot reads while this writer mutates.
    /// Clone it across threads; each [`Reader::view`] is a consistent snapshot.
    pub fn reader(&self) -> Reader<S> {
        Reader {
            published: self.published.clone(),
        }
    }

    /// Rebuild the published snapshot from the current segments + tombstones. Called
    /// after any mutation that changes what a reader should see (a sealed segment, a
    /// delete, a compaction). v1 clones the segment data into the new published view;
    /// holding segments as `Arc` internally to make this a refcount-only publish is a
    /// future optimization if publish cost shows up under profiling.
    fn republish(&self) {
        let segments: Vec<Arc<S::Segment>> =
            self.segments.iter().map(|s| Arc::new(s.clone())).collect();
        let state = Arc::new(PubState {
            segments: Arc::new(segments),
            tombstones: Arc::new(self.tombstones.clone()),
        });
        *self.published.write().unwrap() = state;
    }

    /// Add (or re-add) an item. Durably logged before it becomes visible.
    pub fn add(&mut self, id: S::Id, item: S::Item) -> PersistenceResult<()> {
        self.wal
            .append_postcard(&Op::Add(id.clone(), item.clone()))?;
        self.sync_wal()?;
        apply(&mut self.buffer, &mut self.tombstones, Op::Add(id, item));
        if self.buffer.len() >= self.flush_threshold {
            self.flush_buffer();
            if self.auto_compact && self.has_eligible_tier() {
                self.compact_tiers()?;
            }
        }
        Ok(())
    }

    /// Add (or re-add) many items, syncing the WAL once for the whole batch
    /// instead of per item. This is the bulk-ingest path (the build phase of an
    /// index, e.g. lexir/sporse loading a corpus): per-item flush is the dominant
    /// cost there, and one sync per batch is several times faster on a real disk.
    ///
    /// The durability boundary is the batch's final sync, so a crash mid-batch may
    /// leave a *prefix* of the batch durable (each item is an independently
    /// CRC-checked WAL record); recovery yields a consistent prefix, never a
    /// half-written item. Auto-compaction (if enabled) still runs when a segment
    /// seals during the batch.
    pub fn extend(
        &mut self,
        items: impl IntoIterator<Item = (S::Id, S::Item)>,
    ) -> PersistenceResult<()> {
        let mut any = false;
        for (id, item) in items {
            self.wal
                .append_postcard(&Op::Add(id.clone(), item.clone()))?;
            apply(&mut self.buffer, &mut self.tombstones, Op::Add(id, item));
            any = true;
            if self.buffer.len() >= self.flush_threshold {
                self.flush_buffer();
                if self.auto_compact && self.has_eligible_tier() {
                    self.sync_wal()?; // make the sealed batch durable before a merge rewrites it
                    self.compact_tiers()?;
                }
            }
        }
        if any {
            self.sync_wal()?;
        }
        Ok(())
    }

    /// Tombstone an item. Durably logged before it takes effect.
    pub fn delete(&mut self, id: S::Id) -> PersistenceResult<()> {
        self.wal
            .append_postcard::<Op<S::Id, S::Item>>(&Op::Delete(id.clone()))?;
        self.sync_wal()?;
        apply(&mut self.buffer, &mut self.tombstones, Op::Delete(id));
        self.republish();
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
        self.republish();
    }

    /// Merge ALL segments into one, dropping tombstoned ids and purging the
    /// tombstone set, then checkpoint. This is the full compaction; it is the only
    /// path that touches frozen (over-cap) segments and the only one that purges
    /// tombstones (a partial tier merge cannot, since a tombstoned id may live in a
    /// segment it did not merge).
    pub fn compact(&mut self) -> PersistenceResult<CompactionStats> {
        self.flush_buffer();
        let before = self.segments.len();
        let mut stats = CompactionStats {
            segments_before: before,
            ..Default::default()
        };
        // Nothing to do for 0/1 segments with no tombstones to purge.
        if before > 1 || (before == 1 && !self.tombstones.is_empty()) {
            let tombstones = std::mem::take(&mut self.tombstones);
            let merged = self
                .store
                .merge_segments(&self.segments, &|id| !tombstones.contains(id));
            stats.merges = 1;
            stats.items_merged = self.store.segment_len(&merged);
            // Drop a fully-tombstoned merge result rather than keep an empty segment.
            self.segments = if stats.items_merged > 0 {
                vec![merged]
            } else {
                vec![]
            };
        }
        // After a full compaction no segment references a tombstoned id, so the set
        // is purged even if there was nothing to merge (stale tombstones for ids that
        // were only ever buffered).
        self.tombstones.clear();
        self.checkpoint()?;
        stats.segments_after = self.segments.len();
        Ok(stats)
    }

    /// Run all currently-eligible size-tier merges (size-tiered compaction) per the
    /// [`TierConfig`], persisting the result if anything merged. Segments are grouped
    /// into size buckets (a Cassandra-style `bucket_low .. bucket_high` band around a
    /// running average, with a [`Options::flush_threshold`] floor); a bucket with at
    /// least `min_merge` segments is merged (smallest first, up to `max_merge` at once,
    /// never exceeding `max_merged_len` items), dropping tombstoned ids from the merged
    /// output. Segments above `max_merged_len / 2` are frozen out, so the largest segment
    /// is never rewritten here. The global tombstone set is NOT purged (only [`Self::compact`]
    /// can); a tombstoned id surviving in a frozen segment stays filtered by [`Self::is_live`].
    ///
    /// Scheduling is the consumer's: call this when convenient (e.g. on a background
    /// thread) so merge latency stays off the ingest path, or set [`Options::auto_compact`].
    pub fn compact_tiers(&mut self) -> PersistenceResult<CompactionStats> {
        self.flush_buffer();
        let mut stats = CompactionStats {
            segments_before: self.segments.len(),
            ..Default::default()
        };
        while let Some(group) = self.next_merge_group() {
            stats.items_merged += self.merge_group(group);
            stats.merges += 1;
        }
        if stats.merges > 0 {
            self.checkpoint()?;
        }
        stats.segments_after = self.segments.len();
        Ok(stats)
    }

    /// Consolidate on demand: merge segments until at most `max_segments` remain,
    /// ignoring the size band, the per-bucket minimum, and the [`TierConfig`] cap
    /// (this is explicit user intent, e.g. before a read-heavy phase, so it may
    /// produce a segment larger than `max_merged_len`). Merging to a single segment
    /// (`max_segments <= 1`) also purges the tombstone set, exactly like
    /// [`Self::compact`]; merging to more keeps the set (an id may live in a segment
    /// the call did not merge). Smallest segments are merged first to keep the work
    /// down. A no-op when already at or below `max_segments`.
    pub fn force_merge_to(&mut self, max_segments: usize) -> PersistenceResult<CompactionStats> {
        if max_segments <= 1 {
            return self.compact();
        }
        self.flush_buffer();
        let mut stats = CompactionStats {
            segments_before: self.segments.len(),
            ..Default::default()
        };
        while self.segments.len() > max_segments {
            let mut idx: Vec<usize> = (0..self.segments.len()).collect();
            idx.sort_by_key(|&i| self.store.segment_len(&self.segments[i]));
            // Merging k segments drops the count by k-1; to remove `over` segments we
            // merge `over + 1`, bounded by max_merge and what's available.
            let over = self.segments.len() - max_segments;
            let k = (over + 1)
                .min(self.tiering.max_merge.max(2))
                .min(self.segments.len());
            let group: Vec<usize> = idx.into_iter().take(k).collect();
            stats.items_merged += self.merge_group(group);
            stats.merges += 1;
        }
        if stats.merges > 0 {
            self.checkpoint()?;
        }
        stats.segments_after = self.segments.len();
        Ok(stats)
    }

    /// Merge the segments at `indices` into one (dropping tombstoned ids), replacing
    /// them in place with the single result. Returns the merged item count.
    fn merge_group(&mut self, indices: Vec<usize>) -> usize {
        let segs: Vec<S::Segment> = indices.iter().map(|&i| self.segments[i].clone()).collect();
        let merged = {
            let tombstones = &self.tombstones;
            self.store
                .merge_segments(&segs, &|id| !tombstones.contains(id))
        };
        let n = self.store.segment_len(&merged);
        // Rebuild the segment list in one O(n) pass (filtering the merged indices)
        // instead of k O(n) `Vec::remove` calls, then append the result unless it is
        // empty (a fully-tombstoned group leaves no segment behind).
        let mut merged_idx = indices;
        merged_idx.sort_unstable();
        let old = std::mem::take(&mut self.segments);
        self.segments = old
            .into_iter()
            .enumerate()
            .filter(|(i, _)| merged_idx.binary_search(i).is_err())
            .map(|(_, seg)| seg)
            .collect();
        if n > 0 {
            self.segments.push(merged);
        }
        n
    }

    /// Whether at least one size bucket is eligible to merge right now.
    pub fn has_eligible_tier(&self) -> bool {
        self.next_merge_group().is_some()
    }

    /// Choose the next group of segment indices to merge (size-tiered selection), or
    /// `None` if no bucket is eligible. Smallest-average bucket first.
    fn next_merge_group(&self) -> Option<Vec<usize>> {
        let cfg = self.tiering;
        // A group of 1 would merge a segment into itself (no progress), so the loop
        // in `compact_tiers` would not terminate. 2 is the floor regardless of config.
        let min_merge = cfg.min_merge.max(2);
        let floor = self.flush_threshold.max(1);
        let cap_half = cfg.max_merged_len / 2;
        // (raw size, floored size, index) for segments not frozen by the cap.
        let mut elig: Vec<(usize, f64, usize)> = self
            .segments
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let raw = self.store.segment_len(s);
                (raw, raw.max(floor) as f64, i)
            })
            .filter(|&(raw, _, _)| raw <= cap_half)
            .collect();
        if elig.len() < min_merge {
            return None;
        }
        elig.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Cassandra-style bucketing: join the first bucket whose running average is
        // within the size band, else open a new bucket.
        let mut buckets: Vec<(f64, Vec<(usize, usize)>)> = Vec::new();
        'item: for &(raw, sz, idx) in &elig {
            for b in buckets.iter_mut() {
                if sz > b.0 * cfg.bucket_low && sz < b.0 * cfg.bucket_high {
                    let n = b.1.len() as f64;
                    b.0 = (b.0 * n + sz) / (n + 1.0);
                    b.1.push((raw, idx));
                    continue 'item;
                }
            }
            buckets.push((sz, vec![(raw, idx)]));
        }
        buckets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        for (_, mut members) in buckets {
            if members.len() < min_merge {
                continue;
            }
            members.sort_by_key(|&(raw, _)| raw); // smallest first
            let mut chosen = Vec::new();
            let mut total = 0usize;
            for &(raw, idx) in &members {
                if chosen.len() >= cfg.max_merge {
                    break;
                }
                if !chosen.is_empty() && total + raw > cfg.max_merged_len {
                    break;
                }
                total += raw;
                chosen.push(idx);
            }
            if chosen.len() >= min_merge {
                return Some(chosen);
            }
        }
        None
    }

    /// Total items physically stored across segments and the buffer. This counts
    /// tombstoned-but-not-yet-purged items (segstore cannot see inside an opaque
    /// segment to subtract them); pair with [`Self::tombstone_count`] to gauge
    /// space amplification.
    pub fn stored_len(&self) -> usize {
        let in_segments: usize = self
            .segments
            .iter()
            .map(|s| self.store.segment_len(s))
            .sum();
        in_segments + self.buffer.len()
    }

    /// The current WAL generation.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Space amplification = stored items / live items, or `None` if any segment's
    /// [`Store::live_len`] returns `None` (the default). A value near 1.0 means
    /// little dead data; a higher value means tombstoned/obsolete items are
    /// accumulating, reclaimable with [`Self::reclaim_tombstones`] or [`Self::compact`].
    pub fn space_amplification(&self) -> Option<f64> {
        let mut stored = 0usize;
        let mut live = 0usize;
        for seg in &self.segments {
            let l = self
                .store
                .live_len(seg, &|id| !self.tombstones.contains(id))?;
            stored += self.store.segment_len(seg);
            live += l;
        }
        // Buffer items are always live (a delete removes from the buffer).
        stored += self.buffer.len();
        live += self.buffer.len();
        if live == 0 {
            return Some(if stored == 0 { 1.0 } else { f64::INFINITY });
        }
        Some(stored as f64 / live as f64)
    }

    /// Merge segments whose live ratio (`live_len / segment_len`) is below
    /// `min_live_ratio`, reclaiming their dead (tombstoned) entries into one fresh
    /// segment. The cheap alternative to [`Self::compact`] when only a few segments
    /// are tombstone-heavy. Requires [`Store::live_len`]; a no-op if it returns
    /// `None` or nothing qualifies. Keeps the tombstone set (a reclaimed id may
    /// still live in a segment this did not touch); only `compact` purges it.
    pub fn reclaim_tombstones(
        &mut self,
        min_live_ratio: f64,
    ) -> PersistenceResult<CompactionStats> {
        self.flush_buffer();
        let mut stats = CompactionStats {
            segments_before: self.segments.len(),
            ..Default::default()
        };
        let mut targets = Vec::new();
        for (i, seg) in self.segments.iter().enumerate() {
            let total = self.store.segment_len(seg);
            if total == 0 {
                continue;
            }
            let live = match self
                .store
                .live_len(seg, &|id| !self.tombstones.contains(id))
            {
                Some(l) => l,
                None => {
                    // Consumer can't report live counts; reclaim is unavailable.
                    stats.segments_after = self.segments.len();
                    return Ok(stats);
                }
            };
            if (live as f64) < min_live_ratio * total as f64 {
                targets.push(i);
            }
        }
        if targets.is_empty() {
            stats.segments_after = self.segments.len();
            return Ok(stats);
        }
        // Even a single tombstone-heavy segment is worth rewriting to drop dead data.
        stats.items_merged += self.merge_group(targets);
        stats.merges = 1;
        self.checkpoint()?;
        stats.segments_after = self.segments.len();
        Ok(stats)
    }

    /// Per-segment item counts, oldest segment first. The size distribution the
    /// consumer needs to watch the crossover signal (segment count + per-query
    /// fan-out vs corpus size) that says a multi-segment store is outgrowing the
    /// search-all-segments model.
    pub fn segment_sizes(&self) -> Vec<usize> {
        self.segments
            .iter()
            .map(|s| self.store.segment_len(s))
            .collect()
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
        // Publish the post-compaction segment set to readers (the commit point).
        self.republish();
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
        fn segment_len(&self, seg: &Vec<(u32, String)>) -> usize {
            seg.len()
        }
        fn live_len(&self, seg: &Vec<(u32, String)>, live: &dyn Fn(&u32) -> bool) -> Option<usize> {
            Some(seg.iter().filter(|(id, _)| live(id)).count())
        }
    }

    /// A store that does NOT implement `live_len` (uses the default `None`).
    struct OpaqueKv;
    impl Store for OpaqueKv {
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
        fn segment_len(&self, seg: &Vec<(u32, String)>) -> usize {
            seg.len()
        }
    }

    /// A `Directory` that fails its Nth mutating operation (and every one after),
    /// to test IO-error propagation and crash atomicity. Reads always succeed.
    struct FaultDir {
        inner: Arc<dyn Directory>,
        countdown: std::sync::atomic::AtomicUsize,
    }
    impl FaultDir {
        fn arc(inner: Arc<dyn Directory>, fail_after: usize) -> Arc<dyn Directory> {
            Arc::new(FaultDir {
                inner,
                countdown: std::sync::atomic::AtomicUsize::new(fail_after),
            })
        }
        fn gate(&self) -> PersistenceResult<()> {
            use std::sync::atomic::Ordering::Relaxed;
            let n = self.countdown.load(Relaxed);
            if n == 0 {
                return Err(PersistenceError::InvalidState("injected IO fault".into()));
            }
            self.countdown.store(n - 1, Relaxed);
            Ok(())
        }
    }
    impl Directory for FaultDir {
        fn create_file(&self, p: &str) -> PersistenceResult<Box<dyn std::io::Write + Send>> {
            self.gate()?;
            self.inner.create_file(p)
        }
        fn open_file(&self, p: &str) -> PersistenceResult<Box<dyn std::io::Read + Send>> {
            self.inner.open_file(p)
        }
        fn exists(&self, p: &str) -> bool {
            self.inner.exists(p)
        }
        fn delete(&self, p: &str) -> PersistenceResult<()> {
            self.gate()?;
            self.inner.delete(p)
        }
        fn atomic_rename(&self, a: &str, b: &str) -> PersistenceResult<()> {
            self.gate()?;
            self.inner.atomic_rename(a, b)
        }
        fn create_dir_all(&self, p: &str) -> PersistenceResult<()> {
            self.inner.create_dir_all(p)
        }
        fn list_dir(&self, p: &str) -> PersistenceResult<Vec<String>> {
            self.inner.list_dir(p)
        }
        fn append_file(&self, p: &str) -> PersistenceResult<Box<dyn std::io::Write + Send>> {
            self.gate()?;
            self.inner.append_file(p)
        }
        fn atomic_write(&self, p: &str, d: &[u8]) -> PersistenceResult<()> {
            self.gate()?;
            self.inner.atomic_write(p, d)
        }
        fn file_path(&self, p: &str) -> Option<std::path::PathBuf> {
            self.inner.file_path(p)
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
            let opts = Options {
                sync: SyncPolicy::Fsync,
                ..Options::new(2)
            };
            let mut s = SegmentedStore::open_with_options(dir, Kv, opts).unwrap();
            s.add(1, "a".into()).unwrap();
            s.add(2, "b".into()).unwrap(); // flush
            s.checkpoint().unwrap();
            s.add(3, "c".into()).unwrap();
            s.delete(2).unwrap();
        }
        let dir = durability::FsDirectory::arc(&root).unwrap();
        let opts = Options {
            sync: SyncPolicy::Fsync,
            ..Options::new(2)
        };
        let s2 = SegmentedStore::open_with_options(dir, Kv, opts).unwrap();
        assert_eq!(live_set(&s2), vec![(1, "a".into()), (3, "c".into())]);
        assert!(!s2.is_live(&2));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fsync_policy_rejected_without_filesystem_backend() {
        let dir = MemoryDirectory::arc();
        let opts = Options {
            sync: SyncPolicy::Fsync,
            ..Options::new(2)
        };
        let err = SegmentedStore::open_with_options(dir, Kv, opts);
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

    // ---- size-tiered compaction ----

    /// The live `(id, item)` multiset, sorted, for invariant comparisons.
    fn live_multiset(s: &SegmentedStore<Kv>) -> Vec<(u32, String)> {
        live_set(s)
    }

    fn tier_opts(flush: usize, cfg: TierConfig, auto: bool) -> Options {
        Options {
            tiering: cfg,
            auto_compact: auto,
            ..Options::new(flush)
        }
    }

    #[test]
    fn compact_tiers_merges_a_full_bucket_and_is_idempotent() {
        let dir = MemoryDirectory::arc();
        let cfg = TierConfig {
            min_merge: 4,
            ..Default::default()
        };
        let mut s = SegmentedStore::open_with_options(dir, Kv, tier_opts(2, cfg, false)).unwrap();
        // 8 adds at flush=2 -> 4 segments of size 2: one full bucket.
        for i in 0..8u32 {
            s.add(i, format!("v{i}")).unwrap();
        }
        assert_eq!(
            s.segment_count(),
            4,
            "four size-2 segments before compaction"
        );
        let stats = s.compact_tiers().unwrap();
        assert_eq!(stats.merges, 1, "the full bucket merges once");
        assert_eq!(s.segment_count(), 1, "into a single segment");
        assert_eq!(stats.items_merged, 8);
        // Idempotent: nothing eligible now.
        assert!(!s.has_eligible_tier());
        let again = s.compact_tiers().unwrap();
        assert_eq!(again.merges, 0, "second call is a no-op");
    }

    #[test]
    fn tier_merge_respects_max_merged_len_cap() {
        let dir = MemoryDirectory::arc();
        // cap = 8 -> segments above 4 are frozen; 4 size-2 segments merge to exactly 8.
        let cfg = TierConfig {
            min_merge: 4,
            max_merged_len: 8,
            ..Default::default()
        };
        let mut s = SegmentedStore::open_with_options(dir, Kv, tier_opts(2, cfg, true)).unwrap();
        for i in 0..64u32 {
            s.add(i, format!("v{i}")).unwrap();
        }
        s.compact_tiers().unwrap();
        for seg in s.segments() {
            assert!(
                seg.len() <= cfg.max_merged_len,
                "no segment exceeds the cap; got {}",
                seg.len()
            );
        }
    }

    #[test]
    fn auto_compact_bounds_segment_count() {
        let dir = MemoryDirectory::arc();
        let cfg = TierConfig::default(); // min_merge 4, large cap
        let mut s = SegmentedStore::open_with_options(dir, Kv, tier_opts(2, cfg, true)).unwrap();
        // 200 adds at flush=2 would be 100 segments without compaction.
        for i in 0..200u32 {
            s.add(i, format!("v{i}")).unwrap();
        }
        s.compact_tiers().unwrap();
        assert!(
            s.segment_count() <= 20,
            "tiering bounds segment count well below the uncompacted 100; got {}",
            s.segment_count()
        );
        // No data lost.
        assert_eq!(live_multiset(&s).len(), 200);
    }

    #[test]
    fn tier_merge_preserves_live_set_and_keeps_tombstones() {
        let dir = MemoryDirectory::arc();
        let cfg = TierConfig {
            min_merge: 4,
            max_merged_len: 8, // force some frozen segments
            ..Default::default()
        };
        let mut s = SegmentedStore::open_with_options(dir, Kv, tier_opts(2, cfg, false)).unwrap();
        let mut expect: std::collections::BTreeMap<u32, String> = Default::default();
        for i in 0..40u32 {
            s.add(i, format!("v{i}")).unwrap();
            expect.insert(i, format!("v{i}"));
        }
        // Delete a spread of ids (some land in soon-to-be-frozen segments).
        for i in (0..40u32).step_by(5) {
            s.delete(i).unwrap();
            expect.remove(&i);
        }
        s.compact_tiers().unwrap();
        let want: Vec<(u32, String)> = expect.into_iter().collect();
        assert_eq!(live_multiset(&s), want, "tier merge preserves the live set");
        assert!(
            s.tombstone_count() > 0,
            "partial tier merge keeps the tombstone set (ids may live in frozen segments)"
        );
        // Full compaction purges the tombstones.
        s.compact().unwrap();
        assert_eq!(s.tombstone_count(), 0, "full compact purges tombstones");
        assert_eq!(live_multiset(&s), want, "and preserves the live set");
    }

    #[test]
    fn min_merge_below_two_terminates() {
        let dir = MemoryDirectory::arc();
        let cfg = TierConfig {
            min_merge: 1, // would group-of-1 forever without the clamp
            ..Default::default()
        };
        let mut s = SegmentedStore::open_with_options(dir, Kv, tier_opts(1, cfg, false)).unwrap();
        for i in 0..8u32 {
            s.add(i, format!("v{i}")).unwrap();
        }
        // Must terminate (clamp forces min_merge >= 2).
        let stats = s.compact_tiers().unwrap();
        assert!(stats.segments_after <= stats.segments_before);
        assert_eq!(live_multiset(&s).len(), 8);
    }

    #[test]
    fn large_scale_random_simulation_holds_invariants() {
        let dir = MemoryDirectory::arc();
        let cfg = TierConfig {
            min_merge: 4,
            max_merge: 8,
            max_merged_len: 64,
            ..Default::default()
        };
        let mut s = SegmentedStore::open_with_options(dir, Kv, tier_opts(3, cfg, true)).unwrap();
        // Unique ids per add (segstore makes no dedup promise; identity is the
        // Store impl's job, and the toy Kv appends duplicates on re-add). A
        // monotonic id keeps the reference model exact while still exercising
        // insert/delete/merge invariants.
        let mut expect: std::collections::BTreeMap<u32, String> = Default::default();
        let mut live_ids: Vec<u32> = Vec::new();
        let mut next_id = 0u32;
        // Deterministic LCG (no rand dep, no Math.random).
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for _ in 0..2000 {
            if next() % 4 == 0 && !live_ids.is_empty() {
                let pos = (next() as usize) % live_ids.len();
                let id = live_ids.swap_remove(pos);
                s.delete(id).unwrap();
                expect.remove(&id);
            } else {
                let id = next_id;
                next_id += 1;
                let v = format!("v{id}");
                s.add(id, v.clone()).unwrap();
                expect.insert(id, v);
                live_ids.push(id);
            }
            if next() % 50 == 0 {
                s.compact_tiers().unwrap();
            }
            // Invariant: every segment is within the cap.
            for seg in s.segments() {
                assert!(
                    seg.len() <= cfg.max_merged_len,
                    "seg size {} exceeds cap {}",
                    seg.len(),
                    cfg.max_merged_len
                );
            }
        }
        s.compact().unwrap();
        let want: Vec<(u32, String)> = expect.into_iter().collect();
        assert_eq!(
            live_multiset(&s),
            want,
            "after a long random run, the live set matches the reference"
        );
    }

    #[test]
    fn segment_sizes_reports_per_segment_counts() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
        for i in 0..6u32 {
            s.add(i, format!("v{i}")).unwrap(); // 3 segments of size 2
        }
        assert_eq!(s.segment_sizes(), vec![2, 2, 2]);
        assert_eq!(s.stored_len(), 6);
    }

    /// The store must be `Send` so the consumer can drive `compact_tiers` on a
    /// background thread (the whole point of consumer-driven scheduling).
    #[test]
    fn store_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<SegmentedStore<Kv>>();
    }

    // ---- checkpoint integrity (crash injection) ----

    #[test]
    fn corrupt_checkpoint_is_rejected_not_misread() {
        let root = temp_root("corrupt-ckpt");
        {
            let dir = durability::FsDirectory::arc(&root).unwrap();
            let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
            s.add(1, "a".into()).unwrap();
            s.add(2, "b".into()).unwrap();
            s.checkpoint().unwrap();
        }
        // Flip a byte in the middle of the checkpoint payload (past the header).
        let ckpt = root.join(CKPT_PATH);
        let mut bytes = std::fs::read(&ckpt).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        std::fs::write(&ckpt, &bytes).unwrap();

        let dir = durability::FsDirectory::arc(&root).unwrap();
        let res = SegmentedStore::open(dir, Kv, 2);
        assert!(
            res.is_err(),
            "a CRC-corrupt checkpoint is an error, not silently misread"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn stale_checkpoint_tmp_is_ignored() {
        let root = temp_root("stale-tmp");
        {
            let dir = durability::FsDirectory::arc(&root).unwrap();
            let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
            s.add(1, "a".into()).unwrap();
            s.add(2, "b".into()).unwrap();
            s.checkpoint().unwrap();
        }
        // A crash mid atomic_write can leave a `<ckpt>.tmp` next to the real file.
        std::fs::write(root.join(format!("{CKPT_PATH}.tmp")), b"garbage").unwrap();

        let dir = durability::FsDirectory::arc(&root).unwrap();
        let s2 = SegmentedStore::open(dir, Kv, 2).unwrap();
        assert_eq!(live_set(&s2), vec![(1, "a".into()), (2, "b".into())]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn recovers_consistently_from_io_fault_at_every_step() {
        // FoundationDB-style fault injection: fail the Nth IO op, then reopen on
        // the underlying (un-faulted) directory and assert the recovered live set
        // is EXACTLY the durably-acked adds -- every Ok(add) survives, no failed add
        // appears, recovery never panics and never sees corruption. Sweeps the
        // fault point across the whole op sequence.
        for fail_after in 0..30usize {
            let mem = MemoryDirectory::arc();
            let dir = FaultDir::arc(mem.clone(), fail_after);
            let mut acked: Vec<u32> = Vec::new();
            if let Ok(mut s) = SegmentedStore::open(dir, Kv, 2) {
                for i in 0..10u32 {
                    if s.add(i, format!("v{i}")).is_err() {
                        break;
                    }
                    acked.push(i);
                    if i % 3 == 0 && s.compact_tiers().is_err() {
                        break;
                    }
                }
            }
            // Reopen on the clean underlying directory (the partial on-disk state).
            let s2 = SegmentedStore::open(mem, Kv, 2)
                .expect("recovery after an injected IO fault must not fail");
            let mut live: Vec<u32> = live_set(&s2).into_iter().map(|(id, _)| id).collect();
            live.sort_unstable();
            assert_eq!(
                live, acked,
                "fail_after={fail_after}: recovered set must be exactly the durably-acked adds"
            );
        }
    }

    #[test]
    fn no_space_leak_across_crash_recovery_cycles() {
        // After many crash+recover cycles, deleting everything and a full compaction
        // returns the store to baseline: no leaked segments, items, tombstones, or WAL
        // generations. This is redb's canonical invariant and the bug class targeted
        // crash tests structurally miss (a slow leak across recovery cycles).
        let mem = MemoryDirectory::arc();
        for cycle in 0..20u32 {
            // "Crash": run a few ops under an IO fault and drop the store mid-flight.
            let dir = FaultDir::arc(mem.clone(), (cycle as usize % 5) + 1);
            if let Ok(mut s) = SegmentedStore::open(dir, Kv, 2) {
                for j in 0..4u32 {
                    let id = cycle * 100 + j;
                    let _ = s.add(id, format!("v{id}"));
                    let _ = s.compact_tiers();
                }
            }
            // Recover on the clean directory between cycles.
            let _ = SegmentedStore::open(mem.clone(), Kv, 2).unwrap();
        }
        // Delete every surviving id, then full-compact.
        let mut s = SegmentedStore::open(mem.clone(), Kv, 2).unwrap();
        let live: Vec<u32> = live_set(&s).into_iter().map(|(id, _)| id).collect();
        for id in live {
            s.delete(id).unwrap();
        }
        s.compact().unwrap();
        assert_eq!(
            s.segment_count(),
            0,
            "no leaked segments after delete-all + compact"
        );
        assert_eq!(s.stored_len(), 0, "no leaked stored items");
        assert_eq!(s.tombstone_count(), 0, "compact purged the tombstone set");
        let wals: Vec<String> = mem
            .list_dir("")
            .unwrap()
            .into_iter()
            .filter(|n| n.starts_with(WAL_PREFIX))
            .collect();
        assert!(wals.len() <= 1, "no leaked WAL generations: {wals:?}");
    }
    // ---- tombstone reclamation (optional live_len) ----

    #[test]
    fn space_amplification_tracks_dead_data() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
        for i in 0..10u32 {
            s.add(i, format!("v{i}")).unwrap();
        }
        assert_eq!(
            s.space_amplification(),
            Some(1.0),
            "no deletes, no dead data"
        );
        for i in 0..5u32 {
            s.delete(i).unwrap();
        }
        // 10 stored, 5 live -> amplification 2.0.
        assert_eq!(s.space_amplification(), Some(2.0));
        s.compact().unwrap();
        assert_eq!(s.space_amplification(), Some(1.0), "compaction reclaims it");
    }

    #[test]
    fn space_amplification_is_none_without_live_len() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir, OpaqueKv, 2).unwrap();
        s.add(1, "a".into()).unwrap();
        s.add(2, "b".into()).unwrap();
        assert_eq!(
            s.space_amplification(),
            None,
            "a store without live_len cannot report it"
        );
    }

    #[test]
    fn reclaim_tombstones_rewrites_only_heavy_segments() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
        // 5 segments of size 2 (ids 0..10).
        for i in 0..10u32 {
            s.add(i, format!("v{i}")).unwrap();
        }
        // Tombstone both items in the first two segments (ids 0,1,2,3) -> those
        // segments are fully dead; the rest are fully live.
        for i in 0..4u32 {
            s.delete(i).unwrap();
        }
        let before_stored = s.stored_len();
        let stats = s.reclaim_tombstones(0.5).unwrap();
        assert!(stats.merges > 0, "the dead segments were reclaimed");
        assert!(
            s.stored_len() < before_stored,
            "dead data dropped: {} < {}",
            s.stored_len(),
            before_stored
        );
        // Live set unchanged; tombstones kept (no full purge).
        let live: Vec<u32> = live_set(&s).into_iter().map(|(id, _)| id).collect();
        assert_eq!(live, (4..10).collect::<Vec<_>>());
    }

    #[test]
    fn reclaim_tombstones_is_noop_without_live_len() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir, OpaqueKv, 2).unwrap();
        for i in 0..6u32 {
            s.add(i, format!("v{i}")).unwrap();
        }
        s.delete(0).unwrap();
        let before = s.segment_count();
        let stats = s.reclaim_tombstones(0.9).unwrap();
        assert_eq!(stats.merges, 0, "no live_len -> reclaim is a no-op");
        assert_eq!(s.segment_count(), before);
    }

    #[test]
    fn reclaim_drops_a_fully_dead_segment() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
        for i in 0..4u32 {
            s.add(i, format!("v{i}")).unwrap(); // segment A=[0,1], B=[2,3]
        }
        assert_eq!(s.segment_count(), 2);
        s.delete(0).unwrap();
        s.delete(1).unwrap(); // segment A is now 0% live
        s.reclaim_tombstones(0.5).unwrap();
        // A's merge produces an empty segment, which is dropped (not kept as empty).
        assert_eq!(s.segment_count(), 1, "the fully-dead segment is dropped");
        assert_eq!(s.stored_len(), 2, "only the live segment remains");
    }

    // ---- force-merge / on-demand consolidation ----

    #[test]
    fn force_merge_to_consolidates_without_data_loss() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
        for i in 0..20u32 {
            s.add(i, format!("v{i}")).unwrap(); // 10 segments of size 2
        }
        assert_eq!(s.segment_count(), 10);
        let stats = s.force_merge_to(3).unwrap();
        // Minimum work: consolidates to exactly 3 in a single merge (k = over+1 = 8
        // segments at once), like Lucene's least-number-of-merges force-merge.
        assert_eq!(s.segment_count(), 3, "consolidated to exactly 3");
        assert_eq!(stats.merges, 1, "one merge suffices (minimum work)");
        assert!(stats.items_merged > 0, "merged work is reported");
        assert_eq!(live_multiset(&s).len(), 20, "no data lost");
    }

    #[test]
    fn force_merge_to_one_purges_tombstones() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
        for i in 0..10u32 {
            s.add(i, format!("v{i}")).unwrap();
        }
        for i in 0..5u32 {
            s.delete(i).unwrap();
        }
        s.force_merge_to(1).unwrap();
        assert_eq!(s.segment_count(), 1);
        assert_eq!(s.tombstone_count(), 0, "merge to one purges tombstones");
        let live: Vec<u32> = live_set(&s).into_iter().map(|(id, _)| id).collect();
        assert_eq!(live, (5..10).collect::<Vec<_>>());
    }

    #[test]
    fn force_merge_to_is_noop_when_already_small() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
        for i in 0..4u32 {
            s.add(i, format!("v{i}")).unwrap(); // 2 segments
        }
        let e = s.epoch();
        let stats = s.force_merge_to(5).unwrap();
        assert_eq!(stats.merges, 0, "already at/below target, nothing to merge");
        assert_eq!(s.segment_count(), 2);
        assert_eq!(s.epoch(), e, "no merge -> no checkpoint");
    }

    #[test]
    fn force_merge_persists_across_reopen() {
        let dir = MemoryDirectory::arc();
        {
            let mut s = SegmentedStore::open(dir.clone(), Kv, 2).unwrap();
            for i in 0..20u32 {
                s.add(i, format!("v{i}")).unwrap(); // 10 segments
            }
            s.force_merge_to(2).unwrap();
            assert_eq!(s.segment_count(), 2);
        }
        let s = SegmentedStore::open(dir, Kv, 2).unwrap();
        assert_eq!(s.segment_count(), 2, "force-merge result was checkpointed");
        assert_eq!(live_multiset(&s).len(), 20);
    }

    /// Tiering rewrites each item O(log) times, so the total merged work over N
    /// inserts is O(N log N) -- far below the O(N^2) of rebuilding everything on
    /// each compaction, which is the cliff the max_merged_len cap exists to avoid.
    #[test]
    fn tiered_write_amplification_is_subquadratic() {
        let dir = MemoryDirectory::arc();
        let cfg = TierConfig::default(); // min_merge 4, large cap (no freezing here)
        let mut s = SegmentedStore::open_with_options(dir, Kv, tier_opts(4, cfg, false)).unwrap();
        let n: u32 = 4000;
        let mut total_merged = 0usize;
        for i in 0..n {
            s.add(i, format!("v{i}")).unwrap();
            if i % 16 == 0 {
                total_merged += s.compact_tiers().unwrap().items_merged;
            }
        }
        total_merged += s.compact_tiers().unwrap().items_merged;
        let n = n as usize;
        assert!(total_merged > 0, "compaction actually ran");
        assert!(
            total_merged < 20 * n,
            "tiered merge work {total_merged} is ~O(N log N) (< 20*N = {}), not O(N^2={})",
            20 * n,
            n * n
        );
        assert_eq!(live_multiset(&s).len(), n, "no data lost");
    }

    // ---- mutation-driven: pin behavior the live-set tests left unobserved ----

    #[test]
    fn epoch_advances_per_checkpoint() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
        assert_eq!(s.epoch(), 0);
        s.add(1, "a".into()).unwrap();
        s.checkpoint().unwrap();
        assert_eq!(s.epoch(), 1);
        s.checkpoint().unwrap();
        assert_eq!(s.epoch(), 2);
    }

    #[test]
    fn stored_len_and_space_amp_count_the_buffer() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir, Kv, 100).unwrap(); // stays buffered
        s.add(1, "a".into()).unwrap();
        s.add(2, "b".into()).unwrap();
        assert_eq!(s.segment_count(), 0, "all buffered");
        assert_eq!(s.stored_len(), 2, "stored_len counts the buffer");
        assert_eq!(
            s.space_amplification(),
            Some(1.0),
            "buffered items are live"
        );
    }

    #[test]
    fn space_amp_of_empty_store_is_one() {
        let dir = MemoryDirectory::arc();
        let s = SegmentedStore::open(dir, Kv, 2).unwrap();
        assert_eq!(s.space_amplification(), Some(1.0));
    }

    #[test]
    fn space_amp_is_infinite_when_all_dead() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
        s.add(1, "a".into()).unwrap();
        s.add(2, "b".into()).unwrap(); // 1 segment, size 2
        s.delete(1).unwrap();
        s.delete(2).unwrap(); // stored 2, live 0
        assert!(
            s.space_amplification().unwrap().is_infinite(),
            "an all-dead segment is infinite amplification"
        );
    }

    #[test]
    fn has_eligible_tier_is_true_when_a_bucket_is_full() {
        let dir = MemoryDirectory::arc();
        let cfg = TierConfig {
            min_merge: 4,
            ..Default::default()
        };
        let mut s = SegmentedStore::open_with_options(dir, Kv, tier_opts(2, cfg, false)).unwrap();
        for i in 0..8u32 {
            s.add(i, format!("v{i}")).unwrap(); // 4 size-2 segments = one full bucket
        }
        assert!(s.has_eligible_tier());
    }

    #[test]
    fn compact_merges_only_when_there_is_work() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
        s.add(1, "a".into()).unwrap();
        s.add(2, "b".into()).unwrap(); // 1 segment, no tombstones
        assert_eq!(
            s.compact().unwrap().merges,
            0,
            "a single clean segment: nothing to merge"
        );
        s.delete(1).unwrap();
        assert_eq!(
            s.compact().unwrap().merges,
            1,
            "a tombstone present: merge to purge it"
        );
    }

    #[test]
    fn reclaim_respects_the_live_ratio_threshold() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir, Kv, 4).unwrap();
        for i in 0..12u32 {
            s.add(i, format!("v{i}")).unwrap(); // 3 segments: [0-3] [4-7] [8-11]
        }
        // A: delete 0,1,2 -> 1/4 = 0.25 live. B: delete 4 -> 3/4 = 0.75. C: delete 8,9 -> 2/4 = 0.5.
        for i in [0u32, 1, 2, 4, 8, 9] {
            s.delete(i).unwrap();
        }
        let stats = s.reclaim_tombstones(0.5).unwrap();
        // Only A (0.25 < 0.5) qualifies; C (0.5) is at the threshold, not below; B (0.75) is above.
        assert_eq!(
            stats.merges, 1,
            "only the sub-threshold segment is reclaimed"
        );
        assert_eq!(stats.items_merged, 1, "A's single live item is rewritten");
        assert_eq!(
            s.segment_count(),
            3,
            "A rewritten in place; B and C untouched"
        );
        assert_eq!(s.stored_len(), 1 + 4 + 4, "only A's dead entries dropped");
    }

    #[test]
    fn compact_tiers_persists_across_reopen() {
        let dir = MemoryDirectory::arc();
        let cfg = TierConfig {
            min_merge: 4,
            ..Default::default()
        };
        {
            let mut s =
                SegmentedStore::open_with_options(dir.clone(), Kv, tier_opts(2, cfg, false))
                    .unwrap();
            for i in 0..8u32 {
                s.add(i, format!("v{i}")).unwrap();
            }
            assert_eq!(s.compact_tiers().unwrap().merges, 1);
            assert_eq!(s.segment_count(), 1);
        }
        let s = SegmentedStore::open_with_options(dir, Kv, tier_opts(2, cfg, false)).unwrap();
        assert_eq!(s.segment_count(), 1, "the tier merge was checkpointed");
        assert_eq!(live_multiset(&s).len(), 8);
    }

    #[test]
    fn compact_tiers_without_eligible_bucket_does_not_checkpoint() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir, Kv, 100).unwrap();
        s.add(1, "a".into()).unwrap();
        let e = s.epoch();
        assert_eq!(s.compact_tiers().unwrap().merges, 0);
        assert_eq!(s.epoch(), e, "no merge -> no checkpoint -> epoch unchanged");
    }

    // ---- research failure modes: the cap is the O(N^2) write-amp preventer ----

    #[test]
    fn cap_freezes_large_segments_from_tier_merges() {
        let dir = MemoryDirectory::arc();
        // cap = 8 -> a segment above cap/2 = 4 is frozen out of tier merges, so the
        // biggest segment is never rewritten by tiering (the O(N^2) cliff guard).
        let cfg = TierConfig {
            min_merge: 4,
            max_merged_len: 8,
            ..Default::default()
        };
        let mut s = SegmentedStore::open_with_options(dir, Kv, tier_opts(2, cfg, false)).unwrap();
        for i in 0..8u32 {
            s.add(i, format!("v{i}")).unwrap();
        }
        s.force_merge_to(1).unwrap(); // one size-8 segment, now frozen (> cap/2)
        assert_eq!(s.segment_sizes(), vec![8]);
        for i in 8..20u32 {
            s.add(i, format!("v{i}")).unwrap(); // 6 more size-2 segments
        }
        s.compact_tiers().unwrap();
        assert!(
            s.segment_sizes().contains(&8),
            "the frozen segment is never selected for a tier merge"
        );
        for sz in s.segment_sizes() {
            assert!(sz <= cfg.max_merged_len, "no segment exceeds the cap");
        }
        assert_eq!(live_multiset(&s).len(), 20, "no data lost");
    }

    #[test]
    fn without_compaction_fan_out_grows_then_collapses() {
        // The failure mode tiering exists to prevent: with no compaction every
        // flush adds a segment, and a query scans all segments, so fan-out grows
        // linearly with the corpus. Compaction collapses it.
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
        for i in 0..40u32 {
            s.add(i, format!("v{i}")).unwrap();
        }
        assert_eq!(
            s.segment_count(),
            20,
            "no compaction: 40 adds / flush 2 = 20 segments to scan per query"
        );
        s.compact_tiers().unwrap();
        assert!(
            s.segment_count() < 20,
            "tiered compaction collapses the fan-out, got {}",
            s.segment_count()
        );
        assert_eq!(live_multiset(&s).len(), 40, "without losing data");
    }

    // ---- concurrent snapshot reads ----

    #[test]
    fn view_reflects_segments_minus_tombstones() {
        let mut s = SegmentedStore::open(MemoryDirectory::arc(), Kv, 2).unwrap();
        for i in 0..6u32 {
            s.add(i, format!("v{i}")).unwrap();
        }
        s.delete(2).unwrap();
        let v = s.reader().view();
        let mut live: Vec<u32> = Vec::new();
        for seg in v.segments() {
            for (id, _) in seg.iter() {
                if v.is_live(id) {
                    live.push(*id);
                }
            }
        }
        live.sort_unstable();
        assert_eq!(
            live,
            vec![0, 1, 3, 4, 5],
            "view = sealed segments minus tombstones"
        );
    }

    #[test]
    fn view_is_a_stable_snapshot_across_writes() {
        let mut s = SegmentedStore::open(MemoryDirectory::arc(), Kv, 2).unwrap();
        for i in 0..6u32 {
            s.add(i, format!("v{i}")).unwrap(); // 3 segments
        }
        let reader = s.reader();
        let v1 = reader.view();
        let before = v1.segment_count();
        assert_eq!(before, 3);
        // Mutate heavily while v1 is held.
        for i in 6..30u32 {
            s.add(i, format!("v{i}")).unwrap();
        }
        s.compact().unwrap();
        assert_eq!(v1.segment_count(), before, "the held view never changes");
        assert_eq!(
            reader.view().segment_count(),
            1,
            "a fresh view sees the compacted state"
        );
    }

    #[test]
    fn concurrent_reader_during_writes_stays_consistent() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let mut s = SegmentedStore::open_with_options(
            MemoryDirectory::arc(),
            Kv,
            Options {
                tiering: TierConfig {
                    min_merge: 4,
                    ..Default::default()
                },
                ..Options::new(4)
            },
        )
        .unwrap();
        for i in 0..40u32 {
            s.add(i, format!("v{i}")).unwrap();
        }
        let reader = s.reader();
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        // Reader thread: take views and fully scan them while the writer mutates.
        // Each view is an internally-consistent snapshot, so scanning never panics
        // and never sees a torn segment list.
        let handle = std::thread::spawn(move || {
            let mut iters = 0u64;
            while !stop2.load(Ordering::Relaxed) {
                let v = reader.view();
                let mut count = 0usize;
                for seg in v.segments() {
                    for (id, _) in seg.iter() {
                        if v.is_live(id) {
                            count += 1;
                        }
                    }
                }
                std::hint::black_box(count);
                iters += 1;
            }
            iters
        });
        // Writer: add + compact concurrently with the reader.
        for i in 40..400u32 {
            s.add(i, format!("v{i}")).unwrap();
            if i % 20 == 0 {
                s.compact_tiers().unwrap();
            }
        }
        s.compact().unwrap();
        stop.store(true, Ordering::Relaxed);
        handle.join().unwrap();
        // Final snapshot sees every live item.
        let v = s.reader().view();
        let mut live = 0usize;
        for seg in v.segments() {
            for (id, _) in seg.iter() {
                if v.is_live(id) {
                    live += 1;
                }
            }
        }
        assert_eq!(live, 400, "all 400 items live after the concurrent run");
    }

    // ---- bulk ingest (extend) + harder corruption ----

    #[test]
    fn extend_matches_individual_adds() {
        let mut s1 = SegmentedStore::open(MemoryDirectory::arc(), Kv, 2).unwrap();
        let mut s2 = SegmentedStore::open(MemoryDirectory::arc(), Kv, 2).unwrap();
        for i in 0..20u32 {
            s1.add(i, format!("v{i}")).unwrap();
        }
        s2.extend((0..20u32).map(|i| (i, format!("v{i}")))).unwrap();
        assert_eq!(live_set(&s1), live_set(&s2), "extend == individual adds");
        assert_eq!(s1.segment_count(), s2.segment_count());
    }

    #[test]
    fn extend_is_durable_across_reopen() {
        let dir = MemoryDirectory::arc();
        {
            let mut s = SegmentedStore::open(dir.clone(), Kv, 4).unwrap();
            s.extend((0..20u32).map(|i| (i, format!("v{i}")))).unwrap();
        }
        let s = SegmentedStore::open(dir, Kv, 4).unwrap();
        assert_eq!(
            live_set(&s).len(),
            20,
            "the whole batch is durable after sync"
        );
    }

    #[test]
    fn garbled_wal_record_is_caught_by_crc() {
        // Harder than truncation: flip a byte inside a committed WAL record. Each
        // record carries a CRC, so recovery detects the bad record and (BestEffort)
        // stops there -- yielding a consistent PREFIX of the adds, never a panic or
        // a decoded-garbage record. This also documents the interior-corruption
        // behavior: records after the corruption are dropped (treated as a torn tail).
        let root = temp_root("garble-wal");
        {
            let dir = durability::FsDirectory::arc(&root).unwrap();
            let mut s = SegmentedStore::open(dir, Kv, 100).unwrap(); // all in one WAL
            for i in 1..=6u32 {
                s.add(i, format!("v{i}")).unwrap();
            }
        }
        let wal = root.join(wal_path(0));
        let mut bytes = std::fs::read(&wal).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF; // corrupt an interior record
        std::fs::write(&wal, &bytes).unwrap();

        // Recovery is SAFE either way: it returns a hard error (a corrupted length
        // field is not a clean torn tail) OR a consistent contiguous prefix of the
        // adds -- but NEVER a panic, a decoded-garbage id, or the corrupted record.
        // (Which of the two depends on where in the record the byte landed; the
        // safe-behavior invariant is what matters and is what we assert.)
        let dir = durability::FsDirectory::arc(&root).unwrap();
        match SegmentedStore::open(dir, Kv, 100) {
            Ok(s2) => {
                let mut live: Vec<u32> = live_set(&s2).into_iter().map(|(id, _)| id).collect();
                live.sort_unstable();
                assert!(
                    live.len() < 6,
                    "the corrupted record + tail are dropped: {live:?}"
                );
                assert_eq!(
                    live,
                    (1..=live.len() as u32).collect::<Vec<_>>(),
                    "survivors are the contiguous prefix before the corruption: {live:?}"
                );
            }
            Err(_) => { /* hard fault on corruption: acceptable, never garbage */ }
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    // ---- property-based: random op sequences vs a reference model ----

    use proptest::prelude::*;

    /// An op applied to both the store and a reference model. `Add` uses a fresh
    /// unique id (segstore makes no dedup promise, and compaction reorders
    /// segments, so re-adds of one id have no last-write-wins guarantee); `Delete`
    /// targets the k-th currently-live id.
    #[derive(Debug, Clone)]
    enum SimOp {
        Add,
        Delete(usize),
        Compact,
        CompactTiers,
        Reopen,
    }

    fn op_strategy() -> impl Strategy<Value = SimOp> {
        prop_oneof![
            3 => Just(SimOp::Add),
            2 => (0usize..100).prop_map(SimOp::Delete),
            1 => Just(SimOp::Compact),
            1 => Just(SimOp::CompactTiers),
            1 => Just(SimOp::Reopen),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]
        /// For any sequence of add/delete/compact/compact_tiers/reopen, the live set
        /// equals the reference model and every segment stays within the cap.
        #[test]
        fn live_set_matches_reference_under_random_ops(
            ops in proptest::collection::vec(op_strategy(), 0..200)
        ) {
            let dir = MemoryDirectory::arc();
            let cfg = TierConfig {
                min_merge: 4,
                max_merge: 8,
                max_merged_len: 64,
                ..Default::default()
            };
            let mk = || Options { tiering: cfg, ..Options::new(3) };
            let mut s = SegmentedStore::open_with_options(dir.clone(), Kv, mk()).unwrap();
            let mut model: std::collections::BTreeMap<u32, String> = Default::default();
            let mut live_ids: Vec<u32> = Vec::new();
            let mut next_id = 0u32;
            for op in ops {
                match op {
                    SimOp::Add => {
                        let id = next_id;
                        next_id += 1;
                        let v = format!("v{id}");
                        s.add(id, v.clone()).unwrap();
                        model.insert(id, v);
                        live_ids.push(id);
                    }
                    SimOp::Delete(k) => {
                        if !live_ids.is_empty() {
                            let id = live_ids.swap_remove(k % live_ids.len());
                            s.delete(id).unwrap();
                            model.remove(&id);
                        }
                    }
                    SimOp::Compact => {
                        s.compact().unwrap();
                    }
                    SimOp::CompactTiers => {
                        s.compact_tiers().unwrap();
                    }
                    SimOp::Reopen => {
                        s = SegmentedStore::open_with_options(dir.clone(), Kv, mk()).unwrap();
                    }
                }
                for sz in s.segment_sizes() {
                    prop_assert!(sz <= cfg.max_merged_len);
                }
            }
            let want: Vec<(u32, String)> = model.into_iter().collect();
            prop_assert_eq!(live_set(&s), want);
        }
    }
}
