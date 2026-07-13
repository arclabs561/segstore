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
//! use segstore::{DefaultStore, SegmentedStore};
//! use durability::MemoryDirectory;
//!
//! let dir = MemoryDirectory::arc();
//! let mut s = SegmentedStore::open(dir, DefaultStore::<u32, String>::new(), 2).unwrap();
//! s.add(1, "a".into()).unwrap();
//! s.add(2, "b".into()).unwrap();
//! s.delete(2).unwrap();
//! assert!(s.is_live(&1) && !s.is_live(&2));
//! ```
//!
//! For custom segment layouts, implement [`Store`] directly.
//!
//! # Durability
//!
//! A checkpoint is the commit point. It persists each new segment to its own
//! `segstore.seg.<id>` file (written once, never rewritten), then atomically
//! writes a small manifest (`segstore.manifest`, CRC-checked) naming the current
//! segment files + tombstones, then rotates the WAL: a fresh generation
//! (`segstore.wal.<epoch>`) is started and the old one deleted, so the log never
//! grows past one checkpoint interval. Because only *new* segments are written, a
//! checkpoint is O(new data), not O(total): the Lucene `segments_N` / RocksDB
//! MANIFEST model. The manifest records the epoch it covers; recovery loads the
//! named segment files and replays only that epoch's WAL.
//!
//! The write order makes every crash window safe: new segment files are durable
//! before the manifest names them, and superseded segment files are GC'd only
//! after the new manifest is durable. A crash before the manifest leaves orphan
//! segment files the next open never reads and garbage-collects; a crash after it
//! leaves the old files as the orphans, GC'd the same way. The WAL, not the
//! segment files, is the durability backbone: an `add`/`delete` is durable once
//! its WAL record is, so a partial checkpoint never loses an acknowledged write
//! (recovery replays the WAL onto the last good manifest). On a filesystem backend
//! each segment file and the manifest pass a power-loss barrier (fsync of the file
//! and parent dir); in-memory backends publish atomically without one.
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
//! A CRC-corrupt manifest or segment file is a hard error on open, never misread.
//!
//! On-disk format note: 0.3 replaced 0.2's single monolithic `segstore.ckpt`
//! checkpoint blob with the manifest + per-segment-file layout above. A 0.1
//! unsuffixed `segstore.wal` and a 0.2 `segstore.ckpt` (with no manifest) are each
//! detected and rejected with a clear error rather than misread.
//!
//! # Memory model
//!
//! The on-disk layout is segment-per-file, but the current read API is still an
//! in-memory segment model: [`SegmentedStore::open`] deserializes every manifest
//! segment into `Arc<Store::Segment>`, and [`View::segments`] exposes those loaded
//! segments. This is not yet an out-of-core reader for corpora larger than RAM.
//! Such consumers need a future reader shape that exposes stable segment ids and
//! segment file handles (or mmap-backed consumer readers) while preserving the
//! same manifest, checkpoint, and garbage-collection invariants.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::hash::Hash;
use std::io::Read;
use std::marker::PhantomData;
use std::sync::Arc;

use durability::checkpoint::{CheckpointFile, MAX_CHECKPOINT_PAYLOAD_BYTES};
use durability::recordlog::{RecordLogReadMode, RecordLogReader, RecordLogWriter};
use durability::{Directory, PersistenceError, PersistenceResult};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// Prefix for epoch-suffixed WAL files (`segstore.wal.<epoch>`).
const WAL_PREFIX: &str = "segstore.wal.";
/// The single unsuffixed WAL of the 0.1 on-disk format; its presence flags a
/// store written by segstore 0.1, which later formats cannot read.
const LEGACY_WAL_PATH: &str = "segstore.wal";
/// The monolithic checkpoint blob of the 0.2 on-disk format; its presence with no
/// manifest flags a 0.2 store, which the 0.3 manifest format cannot read.
const LEGACY_CKPT_PATH: &str = "segstore.ckpt";
/// The 0.3 checkpoint commit point: a small manifest naming the current segment
/// files + tombstones (see the module-level Durability docs).
const MANIFEST_PATH: &str = "segstore.manifest";
/// Prefix for per-segment files (`segstore.seg.<id>`); each holds one immutable
/// segment, written once and never rewritten.
const SEG_PREFIX: &str = "segstore.seg.";
const CHECKPOINT_MAGIC: [u8; 4] = *b"VCKP";
const CHECKPOINT_FORMAT_VERSION_V1: u32 = 1;
const CHECKPOINT_FORMAT_VERSION_V2: u32 = 2;
const CHECKPOINT_HEADER_BYTES: u64 = 4 + 4 + 8 + 8 + 4;
const SIDECAR_ENVELOPE_HEADER_BYTES: usize = 8 + 4 + 8 + 4 + 4;

/// The WAL file path for a given epoch.
fn wal_path(epoch: u64) -> String {
    format!("{WAL_PREFIX}{epoch}")
}

/// The file path for the segment with the given id.
fn seg_path(id: u64) -> String {
    format!("{SEG_PREFIX}{id}")
}

/// Each segment (and the manifest) is written as one checkpoint blob, so it is
/// bounded by [`MAX_CHECKPOINT_PAYLOAD_BYTES`] (256 MiB). That byte cap and the
/// item-count cap [`TierConfig::max_merged_len`] are in different units and do
/// not know about each other, so a compaction can target an item count whose
/// serialized bytes exceed the blob cap and fail here. This rewraps
/// durability's generic "payload too large" into an actionable message naming
/// the artifact and the levers; unrelated errors pass through unchanged.
fn size_cap_context(e: PersistenceError, what: &str) -> PersistenceError {
    match &e {
        PersistenceError::Format(msg) if msg.contains("payload too large") => {
            PersistenceError::Format(format!(
                "{what} does not fit the {MAX_CHECKPOINT_PAYLOAD_BYTES}-byte per-blob limit \
                 (durability::MAX_CHECKPOINT_PAYLOAD_BYTES): {msg}. Each segment and the \
                 manifest are written as a single checkpoint blob. Keep them under the cap: \
                 lower TierConfig::max_merged_len (it counts items, not bytes) or the open() \
                 flush_threshold so segments stay smaller, and run compact() to drop \
                 tombstones from the manifest."
            ))
        }
        _ => e,
    }
}

fn reject_legacy_formats(dir: &dyn Directory) -> PersistenceResult<()> {
    // A 0.1 store wrote a single unsuffixed WAL; later formats are epoch-suffixed.
    // Reject it explicitly rather than misread its ops as an epoch.
    if dir.exists(LEGACY_WAL_PATH) {
        return Err(PersistenceError::InvalidConfig(
            "segstore 0.1 on-disk store detected (unsuffixed segstore.wal); the \
             on-disk format is epoch-suffixed and cannot read it"
                .into(),
        ));
    }
    // A 0.2 store wrote one monolithic segstore.ckpt blob; 0.3 uses a manifest
    // + per-segment files. A ckpt with no manifest is a 0.2 store; reject it
    // rather than ignore the data it holds.
    if dir.exists(LEGACY_CKPT_PATH) && !dir.exists(MANIFEST_PATH) {
        return Err(PersistenceError::InvalidConfig(
            "segstore 0.2 on-disk store detected (monolithic segstore.ckpt); the 0.3 \
             on-disk format is a manifest + per-segment files and cannot read it"
                .into(),
        ));
    }
    Ok(())
}

/// Reserved prefix for consumer-written per-segment index sidecar files
/// (`segstore.idx.<id>.<kind>`). A consumer persists a built per-segment index
/// (an HNSW graph, posting lists, ...) here keyed by a stable segment id; segstore
/// garbage-collects any file in this namespace whose segment id is not in the
/// current manifest, on the same crash-safe schedule as the segment files. The
/// consumer owns the file's encoding and full compatibility checks (segstore
/// never reads it). Those checks should include the expected segment id when a
/// copied or misnamed sidecar would otherwise be unsafe to accept. `kind` is
/// only a short namespace/coexistence tag; new names are validated by
/// [`try_index_name`], while GC accepts older non-empty kind strings so upgrades
/// do not strand legacy sidecars.
const IDX_PREFIX: &str = "segstore.idx.";

/// The segment id named by an index sidecar file `segstore.idx.<id>.<kind>`, or
/// `None` if `name` is not in the index namespace.
fn idx_seg_id(name: &str) -> Option<u64> {
    let rest = name.strip_prefix(IDX_PREFIX)?;
    let (id, kind) = rest.split_once('.')?;
    if kind.is_empty() {
        return None;
    }
    id.parse::<u64>().ok()
}

fn validate_index_kind(kind: &str) -> PersistenceResult<()> {
    if kind.is_empty() {
        return Err(PersistenceError::InvalidConfig(
            "index sidecar kind must be non-empty".into(),
        ));
    }
    if !kind
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
    {
        return Err(PersistenceError::InvalidConfig(format!(
            "index sidecar kind must contain only ASCII letters, digits, '_' or '-': {kind:?}"
        )));
    }
    Ok(())
}

fn index_sidecar_name(id: u64, kind: &str) -> PersistenceResult<String> {
    validate_index_kind(kind)?;
    Ok(format!("{IDX_PREFIX}{id}.{kind}"))
}

fn delete_files_best_effort(dir: &dyn Directory, names: Vec<String>, durable: bool) {
    let mut deleted = false;
    for name in names {
        if dir.exists(&name) && dir.delete(&name).is_ok() {
            deleted = true;
        }
    }
    if durable && deleted {
        let _ = dir.durable_sync_parent_dir(MANIFEST_PATH);
    }
}

/// Location and integrity metadata for the serialized payload inside a segment file.
///
/// `segstore.seg.<id>` files are checkpoint envelopes: a fixed header followed by
/// the serialized `Store::Segment` payload. This metadata lets filesystem-backed
/// consumers map or range-read only the payload bytes while keeping header parsing
/// in `segstore`. The CRC is the expected checksum of the payload; callers that
/// bypass [`SegmentCatalog::read_segment_payload`] or
/// [`SegmentCatalog::read_segment_payload_into`] must verify it before trusting
/// bytes from untrusted storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentPayloadInfo {
    /// Byte offset in the segment file where the serialized payload starts.
    pub payload_offset: u64,
    /// Length of the serialized payload in bytes.
    pub payload_len: usize,
    /// CRC32 expected for the serialized payload bytes.
    pub payload_crc32: u32,
}

impl SegmentPayloadInfo {
    /// Verify bytes read from [`Self::payload_offset`] match this payload record.
    ///
    /// This is the check mmap and range-read consumers should run before
    /// trusting bytes obtained without [`SegmentCatalog::read_segment_payload`]
    /// or [`SegmentCatalog::read_segment_payload_into`].
    pub fn verify_payload(&self, payload: &[u8]) -> PersistenceResult<()> {
        if payload.len() != self.payload_len {
            return Err(PersistenceError::Format(format!(
                "segment payload length mismatch: got {} bytes, expected {}",
                payload.len(),
                self.payload_len
            )));
        }
        let actual = crc32fast::hash(payload);
        if actual != self.payload_crc32 {
            return Err(PersistenceError::CrcMismatch {
                expected: self.payload_crc32,
                actual,
            });
        }
        Ok(())
    }
}

fn read_checkpoint_payload_header<R: Read + ?Sized>(
    f: &mut R,
) -> PersistenceResult<CheckpointPayloadHeader> {
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    f.read_exact(&mut buf4)?;
    if buf4 != CHECKPOINT_MAGIC {
        return Err(PersistenceError::Format("invalid checkpoint magic".into()));
    }

    f.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if !matches!(
        version,
        CHECKPOINT_FORMAT_VERSION_V1 | CHECKPOINT_FORMAT_VERSION_V2
    ) {
        return Err(PersistenceError::Format(format!(
            "checkpoint version mismatch (got {version})"
        )));
    }

    f.read_exact(&mut buf8)?;
    let last_applied_id = u64::from_le_bytes(buf8);
    f.read_exact(&mut buf8)?;
    let payload_len_u64 = u64::from_le_bytes(buf8);
    let payload_len = usize::try_from(payload_len_u64)
        .map_err(|_| PersistenceError::Format("payload_len overflow".into()))?;
    if payload_len > MAX_CHECKPOINT_PAYLOAD_BYTES {
        return Err(PersistenceError::Format(format!(
            "checkpoint payload too large: {} bytes (max {})",
            payload_len, MAX_CHECKPOINT_PAYLOAD_BYTES
        )));
    }

    f.read_exact(&mut buf4)?;
    let checksum = u32::from_le_bytes(buf4);
    Ok(CheckpointPayloadHeader {
        version,
        last_applied_id,
        payload_len_u64,
        payload_len,
        checksum,
    })
}

#[derive(Debug, Clone, Copy)]
struct CheckpointPayloadHeader {
    version: u32,
    last_applied_id: u64,
    payload_len_u64: u64,
    payload_len: usize,
    checksum: u32,
}

impl CheckpointPayloadHeader {
    fn verify_payload(&self, payload: &[u8]) -> PersistenceResult<()> {
        if payload.len() != self.payload_len {
            return Err(PersistenceError::Format(format!(
                "checkpoint payload truncated: got {} of {} bytes",
                payload.len(),
                self.payload_len
            )));
        }
        let actual = match self.version {
            CHECKPOINT_FORMAT_VERSION_V1 => crc32fast::hash(payload),
            CHECKPOINT_FORMAT_VERSION_V2 => {
                let mut hasher = crc32fast::Hasher::new();
                hasher.update(&CHECKPOINT_MAGIC);
                hasher.update(&self.version.to_le_bytes());
                hasher.update(&self.last_applied_id.to_le_bytes());
                hasher.update(&self.payload_len_u64.to_le_bytes());
                hasher.update(payload);
                hasher.finalize()
            }
            _ => unreachable!("checkpoint header version is validated at parse time"),
        };
        if actual != self.checksum {
            return Err(PersistenceError::CrcMismatch {
                expected: self.checksum,
                actual,
            });
        }
        Ok(())
    }

    fn payload_info(&self, payload: &[u8]) -> SegmentPayloadInfo {
        SegmentPayloadInfo {
            payload_offset: CHECKPOINT_HEADER_BYTES,
            payload_len: self.payload_len,
            payload_crc32: crc32fast::hash(payload),
        }
    }
}

fn read_checkpoint_payload_info(
    dir: &dyn Directory,
    path: &str,
) -> PersistenceResult<SegmentPayloadInfo> {
    let mut f = dir.open_file(path)?;
    let header = read_checkpoint_payload_header(&mut *f)?;
    let mut payload = Vec::new();
    read_checkpoint_payload_into_reader(&mut *f, &header, &mut payload)?;
    Ok(header.payload_info(&payload))
}

fn read_checkpoint_payload_into(
    dir: &dyn Directory,
    path: &str,
    payload: &mut Vec<u8>,
) -> PersistenceResult<()> {
    let mut f = dir.open_file(path)?;
    let header = read_checkpoint_payload_header(&mut *f)?;
    match read_checkpoint_payload_into_reader(&mut *f, &header, payload) {
        Ok(()) => Ok(()),
        Err(err) => {
            payload.clear();
            Err(err)
        }
    }
}

fn read_checkpoint_payload_into_reader<R: Read + ?Sized>(
    f: &mut R,
    header: &CheckpointPayloadHeader,
    payload: &mut Vec<u8>,
) -> PersistenceResult<()> {
    // Don't trust `len` for the upfront allocation: a corrupt length prefix
    // under the cap would otherwise force a zeroed allocation up to 256 MiB
    // before the read fails. Grow with the bytes actually present instead
    // (mirrors durability's read_exact_bounded).
    payload.clear();
    let reserve = header.payload_len.min(1 << 20);
    if payload.capacity() < reserve {
        payload.reserve(reserve - payload.capacity());
    }
    let n = std::io::Read::take(&mut *f, header.payload_len as u64).read_to_end(payload)?;
    if n < header.payload_len {
        return Err(PersistenceError::Format(format!(
            "checkpoint payload truncated: got {n} of {} bytes",
            header.payload_len
        )));
    }
    header.verify_payload(payload)?;
    Ok(())
}

fn read_checkpoint_payload(dir: &dyn Directory, path: &str) -> PersistenceResult<Vec<u8>> {
    let mut payload = Vec::new();
    read_checkpoint_payload_into(dir, path, &mut payload)?;
    Ok(payload)
}

/// The reserved sidecar file name for a consumer's per-segment built index.
///
/// Use this with [`View::segment_ids`] when a reader/searcher needs to load a
/// persisted sidecar without holding the writer's [`SegmentedStore`] handle.
/// `kind` must be non-empty ASCII alphanumeric plus `_` or `-`. It is a short
/// namespace tag, not a compatibility proof; put full algorithm/config/input
/// fingerprints in the sidecar bytes, include the expected segment id when
/// accepting a copied sidecar would be unsafe, and rebuild from the raw segment
/// when they mismatch.
pub fn try_index_name(id: u64, kind: &str) -> PersistenceResult<String> {
    index_sidecar_name(id, kind)
}

/// Infallible convenience wrapper around [`try_index_name`].
///
/// Panics if `kind` violates the documented sidecar-kind grammar.
pub fn index_name(id: u64, kind: &str) -> String {
    try_index_name(id, kind).unwrap_or_else(|e| panic!("invalid segstore index kind: {e}"))
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
    ///
    /// This is an item count, but each segment is persisted as one checkpoint blob
    /// bounded by [`MAX_CHECKPOINT_PAYLOAD_BYTES`] (256 MiB). The two caps are in
    /// different units: keep `max_merged_len * (your serialized bytes per item)` under
    /// that byte cap, or a large merge will fail the checkpoint write. There is no
    /// byte-aware merge planning yet (see the size-cap discussion in the crate docs).
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
    ///
    /// Segments arrive by reference (`&[&Self::Segment]`) so `segstore` can pass
    /// its `Arc`-held segments straight through without cloning the payloads.
    fn merge_segments(
        &self,
        segments: &[&Self::Segment],
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

/// Default vec-backed segment model for append-only source batches.
///
/// This is the common consumer shape where each immutable segment is just
/// `Vec<(Id, Item)>`, `build_segment` copies the sealed buffer, and compaction
/// concatenates live entries from the merged segments. It deliberately does not
/// implement last-write-wins or deduplication: ids are live or tombstoned exactly
/// as in [`SegmentedStore::add`] and [`SegmentedStore::delete`]. Consumers that
/// need sorted segments, replacement semantics, or custom merge logic should
/// keep their own [`Store`] implementation.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultStore<Id, Item> {
    _marker: PhantomData<fn() -> (Id, Item)>,
}

impl<Id, Item> DefaultStore<Id, Item> {
    /// Create a default vec-backed store model.
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<Id, Item> Store for DefaultStore<Id, Item>
where
    Id: Clone + Eq + Hash + Serialize + DeserializeOwned,
    Item: Clone + Serialize + DeserializeOwned,
{
    type Id = Id;
    type Item = Item;
    type Segment = Vec<(Id, Item)>;

    fn build_segment(&self, batch: &[(Id, Item)]) -> Self::Segment {
        batch.to_vec()
    }

    fn merge_segments(
        &self,
        segments: &[&Self::Segment],
        live: &dyn Fn(&Id) -> bool,
    ) -> Self::Segment {
        segments
            .iter()
            .flat_map(|segment| segment.iter())
            .filter(|(id, _)| live(id))
            .cloned()
            .collect()
    }

    fn segment_len(&self, segment: &Self::Segment) -> usize {
        segment.len()
    }

    fn live_len(&self, segment: &Self::Segment, live: &dyn Fn(&Id) -> bool) -> Option<usize> {
        Some(segment.iter().filter(|(id, _)| live(id)).count())
    }
}

/// Consumer-owned sidecar envelope with checked shared mechanics.
///
/// The envelope covers the repeated cross-consumer header pattern:
/// `magic[8] | version:u32 | segment_id:u64 | recipe_len:u32 | crc32:u32 |
/// recipe | payload`. The CRC covers every byte except the CRC field itself,
/// so corrupt fixed header fields, recipe bytes, or payload bytes are rejected
/// before the payload is trusted.
///
/// This type does not define a sidecar format for consumers. `magic`, `version`,
/// `recipe`, and the payload codec remain consumer-owned compatibility choices;
/// this helper only centralizes the bounds checks and CRC framing.
pub struct SidecarEnvelope;

/// Payload range described by a sidecar envelope header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SidecarPayloadInfo {
    payload_offset: u64,
    payload_len: u64,
    crc32: u32,
}

impl SidecarPayloadInfo {
    /// Byte offset of the payload inside the containing sidecar file.
    pub fn payload_offset(self) -> u64 {
        self.payload_offset
    }

    /// Byte length of the payload inside the containing sidecar file.
    pub fn payload_len(self) -> u64 {
        self.payload_len
    }

    /// Stored envelope CRC32.
    ///
    /// [`SidecarEnvelope::payload_info`] returns this value without verifying
    /// it, so lazy readers can decide whether to verify the full envelope or
    /// rely on payload-format checksums.
    pub fn crc32(self) -> u32 {
        self.crc32
    }
}

impl SidecarEnvelope {
    /// Encode a payload with the common sidecar envelope.
    pub fn encode(
        magic: &[u8; 8],
        version: u32,
        segment_id: u64,
        recipe: &[u8],
        payload: &[u8],
    ) -> PersistenceResult<Vec<u8>> {
        let recipe_len = u32::try_from(recipe.len()).map_err(|_| {
            PersistenceError::InvalidConfig(format!(
                "sidecar recipe too large: {} bytes (max {})",
                recipe.len(),
                u32::MAX
            ))
        })?;
        let mut bytes =
            Vec::with_capacity(SIDECAR_ENVELOPE_HEADER_BYTES + recipe.len() + payload.len());
        bytes.extend_from_slice(magic);
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&segment_id.to_le_bytes());
        bytes.extend_from_slice(&recipe_len.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(recipe);
        bytes.extend_from_slice(payload);
        let crc = sidecar_envelope_crc(&bytes);
        bytes[24..28].copy_from_slice(&crc.to_le_bytes());
        Ok(bytes)
    }

    /// Decode and validate a common sidecar envelope, returning the payload.
    pub fn decode<'a>(
        expected_magic: &[u8; 8],
        expected_version: u32,
        expected_segment_id: u64,
        expected_recipe: &[u8],
        bytes: &'a [u8],
    ) -> PersistenceResult<&'a [u8]> {
        if bytes.len() < SIDECAR_ENVELOPE_HEADER_BYTES {
            return Err(PersistenceError::Format(
                "sidecar envelope header is truncated".into(),
            ));
        }
        if &bytes[..8] != expected_magic {
            return Err(PersistenceError::Format(
                "sidecar envelope magic mismatch".into(),
            ));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != expected_version {
            return Err(PersistenceError::Format(format!(
                "sidecar envelope version mismatch: got {version}, expected {expected_version}"
            )));
        }
        let segment_id = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
        if segment_id != expected_segment_id {
            return Err(PersistenceError::Format(format!(
                "sidecar envelope segment id mismatch: got {segment_id}, expected {expected_segment_id}"
            )));
        }
        let recipe_len = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
        let recipe_start = SIDECAR_ENVELOPE_HEADER_BYTES;
        let recipe_end = recipe_start
            .checked_add(recipe_len)
            .ok_or_else(|| PersistenceError::Format("sidecar recipe length overflow".into()))?;
        if bytes.len() < recipe_end {
            return Err(PersistenceError::Format(
                "sidecar envelope recipe is truncated".into(),
            ));
        }

        let expected_crc = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
        let actual_crc = sidecar_envelope_crc(bytes);
        if actual_crc != expected_crc {
            return Err(PersistenceError::CrcMismatch {
                expected: expected_crc,
                actual: actual_crc,
            });
        }

        if &bytes[recipe_start..recipe_end] != expected_recipe {
            return Err(PersistenceError::Format(
                "sidecar envelope recipe mismatch".into(),
            ));
        }
        Ok(&bytes[recipe_end..])
    }

    /// Read and validate the sidecar header/recipe, returning the payload range.
    ///
    /// This reads only the fixed header and recipe from `reader`, which must be
    /// positioned at the start of the sidecar. It does not verify the stored
    /// envelope CRC over the payload; use [`Self::decode`] when the full sidecar
    /// is already resident and must be fully verified before use. This helper is
    /// for payload formats that have their own lazy checksums and need to keep
    /// the payload file-backed.
    pub fn payload_info<R: Read + ?Sized>(
        expected_magic: &[u8; 8],
        expected_version: u32,
        expected_segment_id: u64,
        expected_recipe: &[u8],
        reader: &mut R,
        sidecar_len: u64,
    ) -> PersistenceResult<SidecarPayloadInfo> {
        if sidecar_len < SIDECAR_ENVELOPE_HEADER_BYTES as u64 {
            return Err(PersistenceError::Format(
                "sidecar envelope header is truncated".into(),
            ));
        }

        let mut header = [0u8; SIDECAR_ENVELOPE_HEADER_BYTES];
        reader.read_exact(&mut header)?;
        if &header[..8] != expected_magic {
            return Err(PersistenceError::Format(
                "sidecar envelope magic mismatch".into(),
            ));
        }
        let version = u32::from_le_bytes(header[8..12].try_into().unwrap());
        if version != expected_version {
            return Err(PersistenceError::Format(format!(
                "sidecar envelope version mismatch: got {version}, expected {expected_version}"
            )));
        }
        let segment_id = u64::from_le_bytes(header[12..20].try_into().unwrap());
        if segment_id != expected_segment_id {
            return Err(PersistenceError::Format(format!(
                "sidecar envelope segment id mismatch: got {segment_id}, expected {expected_segment_id}"
            )));
        }
        let recipe_len = u32::from_le_bytes(header[20..24].try_into().unwrap()) as u64;
        let payload_offset = (SIDECAR_ENVELOPE_HEADER_BYTES as u64)
            .checked_add(recipe_len)
            .ok_or_else(|| PersistenceError::Format("sidecar recipe length overflow".into()))?;
        if sidecar_len < payload_offset {
            return Err(PersistenceError::Format(
                "sidecar envelope recipe is truncated".into(),
            ));
        }
        if recipe_len != expected_recipe.len() as u64 {
            return Err(PersistenceError::Format(
                "sidecar envelope recipe mismatch".into(),
            ));
        }

        let mut recipe = vec![0; expected_recipe.len()];
        reader.read_exact(&mut recipe)?;
        if recipe != expected_recipe {
            return Err(PersistenceError::Format(
                "sidecar envelope recipe mismatch".into(),
            ));
        }

        Ok(SidecarPayloadInfo {
            payload_offset,
            payload_len: sidecar_len - payload_offset,
            crc32: u32::from_le_bytes(header[24..28].try_into().unwrap()),
        })
    }
}

fn sidecar_envelope_crc(bytes: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&bytes[..24]);
    hasher.update(&bytes[28..]);
    hasher.finalize()
}

/// One write-ahead-log operation.
#[derive(Serialize, Deserialize)]
enum Op<Id, Item> {
    Add(Id, Item),
    Delete(Id),
}

/// The checkpoint manifest: the commit point of the on-disk state. It names the
/// current segment files by id and carries the tombstone set; the segments
/// themselves live in `segstore.seg.<id>` files written once and never rewritten,
/// so a checkpoint writes only the *new* segments plus this small manifest
/// (O(delta), not O(total)). The epoch lives in the [`CheckpointFile`] header,
/// not the body. This owned form is the *read* side (recovery decodes into it).
#[derive(Deserialize)]
struct Manifest<Id> {
    next_seg_id: u64,
    segment_ids: Vec<u64>,
    tombstones: Vec<Id>,
}

/// Borrowing view of the manifest for *writing*: serializes the ids + tombstones
/// in place. Wire-identical to [`Manifest`] (postcard encodes a struct as its
/// fields in order, and a `&[&Id]` as the same sequence as a `Vec<Id>`).
#[derive(Serialize)]
struct ManifestRef<'a, Id> {
    next_seg_id: u64,
    segment_ids: &'a [u64],
    tombstones: &'a [&'a Id],
}

/// The reader-visible published state: the segments and tombstones as of the last
/// checkpoint, behind `Arc`s so a [`View`] is a cheap, consistent snapshot.
struct PubState<S: Store> {
    segments: Arc<Vec<Arc<S::Segment>>>,
    segment_ids: Arc<Vec<u64>>,
    tombstones: Arc<HashSet<S::Id>>,
}

/// Checkpointed segment metadata without deserializing segment payloads.
///
/// `SegmentCatalog` is a manifest reader, not an out-of-core query engine. It
/// exposes the segment ids and tombstones named by the last checkpoint so a
/// consumer can find segment files or sidecars without opening a
/// [`SegmentedStore`] writer, but the current `segstore.seg.<id>` payloads are
/// still checkpoint-wrapped encodings of `Store::Segment`.
pub struct SegmentCatalog<Id> {
    dir: Arc<dyn Directory>,
    epoch: u64,
    next_segment_id: u64,
    segment_ids: Vec<u64>,
    tombstones: HashSet<Id>,
}

impl<Id> SegmentCatalog<Id>
where
    Id: Eq + Hash + DeserializeOwned,
{
    /// Open the last checkpoint manifest without reading any segment files.
    ///
    /// Visibility matches [`Reader::view`]: this reflects only the most recent
    /// checkpoint. WAL suffix records after that checkpoint are not exposed
    /// because they have not been sealed into immutable segment files.
    pub fn open(dir: Arc<dyn Directory>) -> PersistenceResult<Self> {
        reject_legacy_formats(&*dir)?;

        let mut epoch = 0u64;
        let mut next_segment_id = 0u64;
        let mut segment_ids = Vec::new();
        let mut tombstones = HashSet::new();
        if dir.exists(MANIFEST_PATH) {
            let ckpt = CheckpointFile::new(dir.clone());
            let (e, manifest): (u64, Manifest<Id>) = ckpt.read_postcard(MANIFEST_PATH)?;
            epoch = e;
            next_segment_id = manifest.next_seg_id;
            segment_ids = manifest.segment_ids;
            tombstones = manifest.tombstones.into_iter().collect();
        }

        Ok(Self {
            dir,
            epoch,
            next_segment_id,
            segment_ids,
            tombstones,
        })
    }

    /// WAL epoch covered by the checkpoint manifest.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Next segment id recorded in the manifest.
    pub fn next_segment_id(&self) -> u64 {
        self.next_segment_id
    }

    /// Stable ids of the checkpointed immutable segments, oldest first.
    pub fn segment_ids(&self) -> &[u64] {
        &self.segment_ids
    }

    /// Number of checkpointed immutable segments.
    pub fn segment_count(&self) -> usize {
        self.segment_ids.len()
    }

    /// Number of tombstoned ids in the checkpoint.
    pub fn tombstone_count(&self) -> usize {
        self.tombstones.len()
    }

    /// Whether `id` is not tombstoned in this checkpoint.
    pub fn is_live(&self, id: &Id) -> bool {
        !self.tombstones.contains(id)
    }

    /// The directory backing this catalog.
    pub fn dir(&self) -> &Arc<dyn Directory> {
        &self.dir
    }

    /// Segment file name for `id`, if the checkpoint names that segment.
    pub fn segment_name(&self, id: u64) -> PersistenceResult<String> {
        self.ensure_segment_id(id)?;
        Ok(seg_path(id))
    }

    /// Filesystem path for `id`, if this directory exposes paths.
    pub fn segment_file_path(&self, id: u64) -> PersistenceResult<Option<std::path::PathBuf>> {
        Ok(self.dir.file_path(&self.segment_name(id)?))
    }

    /// Open the raw segment file for `id`.
    ///
    /// The returned file is the current segstore segment file, including its
    /// durability checkpoint envelope. This method intentionally does not decode
    /// `Store::Segment`; consumers that need byte-native segments need the raw
    /// segment API described in `docs/design/out-of-core-segment-reader.md`.
    pub fn open_segment_file(&self, id: u64) -> PersistenceResult<Box<dyn std::io::Read + Send>> {
        self.dir.open_file(&self.segment_name(id)?)
    }

    /// Read one segment file's header and return where its serialized payload lives.
    ///
    /// This validates the checkpoint envelope and reads the payload bytes once
    /// to compute the public payload CRC. Use the returned offset and length
    /// with [`Self::segment_file_path`] for mmap/range-read sidecar rebuilders,
    /// then verify [`SegmentPayloadInfo::payload_crc32`] over the mapped bytes
    /// before trusting them. If you want segstore to return the payload bytes
    /// directly, use [`Self::read_segment_payload`] or
    /// [`Self::read_segment_payload_into`].
    pub fn segment_payload_info(&self, id: u64) -> PersistenceResult<SegmentPayloadInfo> {
        read_checkpoint_payload_info(&*self.dir, &self.segment_name(id)?)
    }

    /// Read one checkpointed segment's CRC-validated serialized payload bytes.
    ///
    /// This avoids opening a full [`SegmentedStore`] and avoids deserializing
    /// `Store::Segment`, but it is still a whole-payload read of the postcard
    /// bytes that segstore wrote for that segment. It is a loader/build helper,
    /// not a streaming or mmap-backed query reader. Tombstones are NOT applied:
    /// the payload is the segment as written, including records later deleted;
    /// filter with [`Self::is_live`] before treating contents as live data.
    pub fn read_segment_payload(&self, id: u64) -> PersistenceResult<Vec<u8>> {
        read_checkpoint_payload(&*self.dir, &self.segment_name(id)?)
    }

    /// Read one checkpointed segment's CRC-validated serialized payload bytes into
    /// `payload`.
    ///
    /// This is the reusable-buffer form of [`Self::read_segment_payload`] for
    /// sidecar builders that scan many segments. `payload` is cleared before the
    /// read; on error it is left empty.
    pub fn read_segment_payload_into(
        &self,
        id: u64,
        payload: &mut Vec<u8>,
    ) -> PersistenceResult<()> {
        read_checkpoint_payload_into(&*self.dir, &self.segment_name(id)?, payload)
    }

    /// Decode one checkpointed segment by id.
    ///
    /// This is a bounded restart/build helper: it decodes only the requested
    /// segment file instead of opening a full [`SegmentedStore`] and decoding
    /// every segment in the manifest. It is still a whole-segment postcard decode,
    /// not a byte-native or mmap-backed query reader. Tombstones are NOT applied:
    /// the decoded segment is as written, including records later deleted;
    /// filter with [`Self::is_live`] before treating contents as live data.
    pub fn read_segment<Segment>(&self, id: u64) -> PersistenceResult<Segment>
    where
        Segment: DeserializeOwned,
    {
        let ckpt = CheckpointFile::new(self.dir.clone());
        let (_, segment) = ckpt.read_postcard(&self.segment_name(id)?)?;
        Ok(segment)
    }

    /// Reserved sidecar file name for a per-segment index of the given `kind`.
    pub fn try_index_name(&self, id: u64, kind: &str) -> PersistenceResult<String> {
        self.ensure_segment_id(id)?;
        try_index_name(id, kind)
    }

    /// Infallible convenience wrapper around [`Self::try_index_name`].
    ///
    /// Panics if `id` is not in this catalog or `kind` violates the documented
    /// sidecar-kind grammar.
    pub fn index_name(&self, id: u64, kind: &str) -> String {
        self.try_index_name(id, kind)
            .unwrap_or_else(|e| panic!("invalid segstore catalog index name: {e}"))
    }

    fn ensure_segment_id(&self, id: u64) -> PersistenceResult<()> {
        // Segment ids are allocated monotonically and kept oldest-first, so the
        // manifest vector is sorted without an extra lookup structure.
        if self.segment_ids.binary_search(&id).is_ok() {
            Ok(())
        } else {
            Err(PersistenceError::NotFound(format!(
                "segment id {id} is not in the checkpoint manifest"
            )))
        }
    }
}

/// A consistent, point-in-time read view of the store, independent of the writer.
///
/// A [`Reader`] hands these out; holding one keeps its segments alive for the whole
/// query even as the writer adds, deletes, or compacts concurrently (single-writer,
/// many-readers, like Lucene's `SearcherManager` / Tantivy's `Searcher`). Visibility
/// is *commit-style*: a view reflects the state as of the last [`SegmentedStore::checkpoint`]
/// (which compaction also performs), so adds and deletes since then become visible
/// after the next checkpoint. Publishing only at the checkpoint keeps it off the
/// ingest hot path (republishing per write made bulk ingest quadratic).
pub struct View<S: Store> {
    segments: Arc<Vec<Arc<S::Segment>>>,
    segment_ids: Arc<Vec<u64>>,
    tombstones: Arc<HashSet<S::Id>>,
}

impl<S: Store> View<S> {
    /// The snapshot's immutable segments, oldest first. Query these (each derefs to
    /// `S::Segment`), filtering with [`Self::is_live`].
    pub fn segments(&self) -> &[Arc<S::Segment>] {
        &self.segments
    }

    /// The stable persistence id of each segment in [`Self::segments`], same order.
    /// Unlike the segment's `Arc` pointer, this id is stable across checkpoints AND
    /// restarts (it names the segment's `segstore.seg.<id>` file), so a consumer can
    /// key a *persisted* per-segment built-index cache on it (via
    /// [`try_index_name`]) and reload across a process restart instead of rebuilding
    /// from the raw payload.
    pub fn segment_ids(&self) -> &[u64] {
        &self.segment_ids
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
            segment_ids: state.segment_ids.clone(),
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
    segments: Vec<Arc<S::Segment>>,
    /// Logically-deleted ids.
    tombstones: HashSet<S::Id>,
    /// Published snapshot for concurrent readers (rebuilt on each checkpoint).
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
    /// The persisted id of each in-memory segment, parallel to `segments` and kept
    /// in lockstep through every segment mutation. The id names the segment's
    /// `segstore.seg.<id>` file.
    segment_ids: Vec<u64>,
    /// Monotonic source of new segment ids, persisted in the manifest so an id is
    /// never reused after its `seg.<id>` file is GC'd.
    next_seg_id: u64,
    /// Ids whose `segstore.seg.<id>` file is durably on disk, so a checkpoint can
    /// skip rewriting an unchanged segment.
    persisted_ids: HashSet<u64>,
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
        if sync == SyncPolicy::Fsync && dir.file_path(MANIFEST_PATH).is_none() {
            return Err(PersistenceError::InvalidConfig(
                "SyncPolicy::Fsync requires a filesystem-backed Directory".into(),
            ));
        }
        reject_legacy_formats(&*dir)?;

        // Load the manifest if one exists. It records the WAL epoch it covers (in
        // the CheckpointFile header) and names the current segment files; recovery
        // loads each segment file, then replays only that epoch's WAL.
        let mut segments: Vec<Arc<S::Segment>> = Vec::new();
        let mut segment_ids: Vec<u64> = Vec::new();
        let mut tombstones: HashSet<S::Id> = HashSet::new();
        let mut epoch = 0u64;
        let mut next_seg_id = 0u64;
        if dir.exists(MANIFEST_PATH) {
            let ckpt = CheckpointFile::new(dir.clone());
            let (e, manifest): (u64, Manifest<S::Id>) = ckpt.read_postcard(MANIFEST_PATH)?;
            epoch = e;
            next_seg_id = manifest.next_seg_id;
            tombstones = manifest.tombstones.into_iter().collect();
            for id in manifest.segment_ids {
                let (_, seg): (u64, S::Segment) = ckpt.read_postcard(&seg_path(id))?;
                segments.push(Arc::new(seg));
                segment_ids.push(id);
            }
        }
        // Reader visibility is checkpoint-only, so retain the manifest state before
        // WAL replay reconstructs the writer's newer, uncommitted state.
        let published_segments = segments.clone();
        let published_segment_ids = segment_ids.clone();
        let published_tombstones = tombstones.clone();
        let persisted_ids: HashSet<u64> = segment_ids.iter().copied().collect();
        let durable = dir.file_path(MANIFEST_PATH).is_some();

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
                // Reconstruct the same immutable segment boundaries as live writes.
                // Otherwise a long WAL epoch becomes one oversized buffer on reopen.
                if !buffer.is_empty() && buffer.len() >= flush_threshold {
                    let seg = store.build_segment(&buffer);
                    segments.push(Arc::new(seg));
                    segment_ids.push(next_seg_id);
                    next_seg_id += 1;
                    buffer.clear();
                }
            }
        }

        // Best-effort GC of crash leftovers: stale WAL generations (any
        // `segstore.wal.<k>` with k != epoch is superseded by the manifest) and
        // orphan segment files (any `segstore.seg.<k>` the manifest does not name,
        // left by a crash between a segment write and the manifest write).
        if let Ok(names) = dir.list_dir("") {
            let mut garbage = Vec::new();
            for name in names {
                if let Some(k) = name
                    .strip_prefix(WAL_PREFIX)
                    .and_then(|s| s.parse::<u64>().ok())
                {
                    if k != epoch {
                        garbage.push(name);
                    }
                } else if let Some(k) = name
                    .strip_prefix(SEG_PREFIX)
                    .and_then(|s| s.parse::<u64>().ok())
                {
                    if !persisted_ids.contains(&k) {
                        garbage.push(name);
                    }
                } else if let Some(k) = idx_seg_id(&name) {
                    // Consumer index sidecar whose segment is gone: orphan, GC it.
                    if !persisted_ids.contains(&k) {
                        garbage.push(name);
                    }
                }
            }
            delete_files_best_effort(&*dir, garbage, durable);
        }

        let wal = RecordLogWriter::new(dir.clone(), live_wal);
        let published = Arc::new(std::sync::RwLock::new(Arc::new(PubState {
            segments: Arc::new(published_segments),
            segment_ids: Arc::new(published_segment_ids),
            tombstones: Arc::new(published_tombstones),
        })));
        Ok(Self {
            store,
            dir,
            buffer,
            segments,
            segment_ids,
            tombstones,
            wal,
            epoch,
            next_seg_id,
            persisted_ids,
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
    /// only at the checkpoint (commit point). Segments are held as `Arc`, so this
    /// shares them by refcount (no data clone) and an *unchanged* segment keeps its
    /// `Arc` identity across checkpoints -- which lets a consumer cache per-segment
    /// state (a built index) keyed by that identity and rebuild only new segments.
    fn republish(&self) {
        let state = Arc::new(PubState {
            segments: Arc::new(self.segments.clone()),
            segment_ids: Arc::new(self.segment_ids.clone()),
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
        let id = self.alloc_id();
        self.segments.push(Arc::new(seg));
        self.segment_ids.push(id);
        self.buffer.clear();
    }

    /// Allocate a fresh, never-reused segment id.
    fn alloc_id(&mut self) -> u64 {
        let id = self.next_seg_id;
        self.next_seg_id += 1;
        id
    }

    /// Delete every `segstore.seg.<k>` file whose id is not in `keep`, and forget
    /// those ids from `persisted_ids`. Called only after a manifest is durable, so
    /// a crash mid-GC just leaves orphans that the next open re-GCs.
    fn gc_orphan_segments(&mut self, keep: &HashSet<u64>, durable: bool, mut garbage: Vec<String>) {
        self.persisted_ids.retain(|id| keep.contains(id));
        if let Ok(names) = self.dir.list_dir("") {
            for name in names {
                if let Some(k) = name
                    .strip_prefix(SEG_PREFIX)
                    .and_then(|s| s.parse::<u64>().ok())
                {
                    if !keep.contains(&k) {
                        garbage.push(name);
                    }
                } else if let Some(k) = idx_seg_id(&name) {
                    // A consumer's index sidecar for a compacted-away segment.
                    if !keep.contains(&k) {
                        garbage.push(name);
                    }
                }
            }
        }
        delete_files_best_effort(&*self.dir, garbage, durable);
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
            // Borrow the Arc-held segments rather than cloning their payloads to
            // satisfy merge_segments (&[&Segment]); the merge reads them in place.
            let refs: Vec<&S::Segment> = self.segments.iter().map(|a| &**a).collect();
            let merged = self
                .store
                .merge_segments(&refs, &|id| !tombstones.contains(id));
            stats.merges = 1;
            stats.items_merged = self.store.segment_len(&merged);
            // Drop a fully-tombstoned merge result rather than keep an empty segment.
            if stats.items_merged > 0 {
                let id = self.alloc_id();
                self.segments = vec![Arc::new(merged)];
                self.segment_ids = vec![id];
            } else {
                self.segments = vec![];
                self.segment_ids = vec![];
            }
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
        let segs: Vec<&S::Segment> = indices.iter().map(|&i| &*self.segments[i]).collect();
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
        // Keep segment_ids aligned with segments: drop the same merged indices.
        let old_ids = std::mem::take(&mut self.segment_ids);
        self.segment_ids = old_ids
            .into_iter()
            .enumerate()
            .filter(|(i, _)| merged_idx.binary_search(i).is_err())
            .map(|(_, id)| id)
            .collect();
        if n > 0 {
            let id = self.alloc_id();
            self.segments.push(Arc::new(merged));
            self.segment_ids.push(id);
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
        debug_assert_eq!(self.segments.len(), self.segment_ids.len());
        let new_epoch = self.epoch + 1;
        // On a filesystem backend, pass a power-loss barrier (fsync); in-memory
        // backends have no barrier and fall back to the atomic-only write.
        let durable = self.dir.file_path(MANIFEST_PATH).is_some();
        let ckpt = CheckpointFile::new(self.dir.clone());

        // 1. Persist every not-yet-written segment to its own file, durably, BEFORE
        //    the manifest names it. A crash here leaves an orphan seg file the
        //    manifest never references (GC'd on open); the old manifest still points
        //    at the old set, so nothing committed is lost.
        let to_write: Vec<(usize, u64)> = self
            .segment_ids
            .iter()
            .enumerate()
            .filter(|(_, id)| !self.persisted_ids.contains(id))
            .map(|(i, &id)| (i, id))
            .collect();
        for (idx, id) in to_write {
            let seg = &*self.segments[idx];
            let seg_write = if durable {
                ckpt.write_postcard_durable(&seg_path(id), 0, seg)
            } else {
                ckpt.write_postcard(&seg_path(id), 0, seg)
            };
            seg_write.map_err(|e| size_cap_context(e, &format!("segment {id}")))?;
            self.persisted_ids.insert(id);
        }

        // 2. Write the manifest: the commit point. It records the new epoch and
        //    names the current segment files, so once it is durable, recovery
        //    replays only the new (initially empty) WAL; the old WAL is superseded.
        let tomb_refs: Vec<&S::Id> = self.tombstones.iter().collect();
        let manifest = ManifestRef {
            next_seg_id: self.next_seg_id,
            segment_ids: &self.segment_ids,
            tombstones: &tomb_refs,
        };
        let manifest_write = if durable {
            ckpt.write_postcard_durable(MANIFEST_PATH, new_epoch, &manifest)
        } else {
            ckpt.write_postcard(MANIFEST_PATH, new_epoch, &manifest)
        };
        manifest_write.map_err(|e| size_cap_context(e, "manifest (segment ids + tombstones)"))?;

        // 3. Past the commit point: start the new WAL generation and drop the old.
        let old_epoch = self.epoch;
        self.wal = RecordLogWriter::new(self.dir.clone(), wal_path(new_epoch));
        self.epoch = new_epoch;

        // 4. GC the segment files the new manifest no longer names (a merge's
        //    inputs). Safe only now that the manifest is durable.
        let keep: HashSet<u64> = self.segment_ids.iter().copied().collect();
        self.gc_orphan_segments(&keep, durable, vec![wal_path(old_epoch)]);

        // 5. Publish the post-checkpoint segment set to readers (the commit point).
        self.republish();
        Ok(())
    }

    /// The immutable segments, oldest first (each derefs to `S::Segment`). Query
    /// these plus [`Self::buffer`], filtering with [`Self::is_live`]. Segments are
    /// `Arc`-shared: an unchanged segment keeps its identity across mutations, so a
    /// consumer can cache per-segment state keyed by `Arc::as_ptr` and rebuild only
    /// new segments. For a consistent view while another thread mutates, use a
    /// [`Reader`] / [`View`] instead. For a cache key that *also* survives a restart,
    /// use [`Self::segment_ids`].
    pub fn segments(&self) -> &[Arc<S::Segment>] {
        &self.segments
    }

    /// The stable persistence id of each segment in [`Self::segments`], same order.
    /// Unlike the segment's `Arc` pointer (stable only within one process), this id
    /// names the segment's `segstore.seg.<id>` file and is stable across checkpoints
    /// *and* restarts, so a consumer can key a *persisted* per-segment built-index
    /// cache on it and reload across a process restart instead of rebuilding from the
    /// raw payload. An id is never reused after its segment is compacted away.
    pub fn segment_ids(&self) -> &[u64] {
        &self.segment_ids
    }

    /// The directory this store is backed by, so a consumer can persist a built
    /// per-segment index alongside the segments. Write the index into the reserved
    /// sidecar name from [`Self::index_name`], keyed by a stable [`Self::segment_ids`]
    /// id; segstore garbage-collects it when its segment is compacted away (on the
    /// same crash-safe schedule as the segment files). A missing or stale sidecar
    /// means "rebuild from the raw segment." This is the persist-the-built-index
    /// hook: recovery loads the sidecar instead of rebuilding every consumer index.
    pub fn dir(&self) -> &Arc<dyn Directory> {
        &self.dir
    }

    /// The reserved sidecar file name for a per-segment index of the given `kind`
    /// (e.g. `"hnsw"`), keyed by a stable [`Self::segment_ids`] id. `kind` must be
    /// non-empty ASCII alphanumeric plus `_` or `-`. It is a short namespace tag,
    /// not a compatibility proof; put full algorithm/config/input fingerprints in
    /// the sidecar bytes and rebuild from the raw segment when they mismatch. Use
    /// with [`Self::dir`] to persist/load a built index that segstore GCs in
    /// lockstep with the segment. Reader/searcher code that only has a [`View`]
    /// can use the module-level [`try_index_name`] helper with [`View::segment_ids`].
    pub fn try_index_name(&self, id: u64, kind: &str) -> PersistenceResult<String> {
        try_index_name(id, kind)
    }

    /// Infallible convenience wrapper around [`Self::try_index_name`].
    ///
    /// Panics if `kind` violates the documented sidecar-kind grammar.
    pub fn index_name(&self, id: u64, kind: &str) -> String {
        self.try_index_name(id, kind)
            .unwrap_or_else(|e| panic!("invalid segstore index kind: {e}"))
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

    #[test]
    fn size_cap_context_makes_the_blob_error_actionable() {
        // The raw durability error is opaque about what overflowed and how to
        // fix it; the wrapper names the artifact and the levers.
        let raw = PersistenceError::Format(
            "checkpoint payload too large: 300000000 bytes (max 268435456)".into(),
        );
        let wrapped = size_cap_context(raw, "segment 7");
        let msg = wrapped.to_string();
        assert!(msg.contains("segment 7"), "names the artifact: {msg}");
        assert!(
            msg.contains("max_merged_len"),
            "names the item-count lever: {msg}"
        );
        assert!(
            msg.contains("checkpoint payload too large"),
            "preserves the original detail: {msg}"
        );
    }

    #[test]
    fn size_cap_context_passes_unrelated_errors_through() {
        let before = PersistenceError::InvalidState("something else".into()).to_string();
        let after = size_cap_context(
            PersistenceError::InvalidState("something else".into()),
            "segment 1",
        )
        .to_string();
        assert_eq!(after, before, "non-cap errors are unchanged");
    }

    #[test]
    fn catalog_payload_reader_rejects_truncated_length_claim() {
        use std::io::Read;

        let dir = MemoryDirectory::arc();
        let seg_id = {
            let mut s = SegmentedStore::open(dir.clone(), Kv, 1).unwrap();
            s.add(1, "a".into()).unwrap();
            s.checkpoint().unwrap();
            s.segment_ids()[0]
        };

        let name = seg_path(seg_id);
        let mut file = dir.open_file(&name).unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        // Checkpoint envelope layout: magic(4), version(4), last_applied_id(8),
        // payload_len(8), crc32(4), then payload bytes.
        let len_offset = 4 + 4 + 8;
        let claimed = 4_u64 * 1024 * 1024;
        bytes[len_offset..len_offset + 8].copy_from_slice(&claimed.to_le_bytes());
        dir.atomic_write(&name, &bytes).unwrap();

        let catalog = SegmentCatalog::<u32>::open(dir).unwrap();
        let err = catalog
            .read_segment_payload(seg_id)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("truncated") || err.to_lowercase().contains("eof"),
            "expected truncation error, got: {err}"
        );
    }

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
            segs: &[&Vec<(u32, String)>],
            live: &dyn Fn(&u32) -> bool,
        ) -> Vec<(u32, String)> {
            segs.iter()
                .flat_map(|s| s.iter())
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
            segs: &[&Vec<(u32, String)>],
            live: &dyn Fn(&u32) -> bool,
        ) -> Vec<(u32, String)> {
            segs.iter()
                .flat_map(|s| s.iter())
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
            for (id, it) in seg.iter() {
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
    fn default_store_implements_the_vec_segment_pattern() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir, DefaultStore::<u32, String>::new(), 2).unwrap();
        s.add(1, "a".into()).unwrap();
        s.add(2, "b".into()).unwrap();
        assert_eq!(s.segment_count(), 1);
        assert_eq!(
            s.segments()[0].as_ref(),
            &vec![(1, "a".into()), (2, "b".into())]
        );

        s.delete(2).unwrap();
        assert_eq!(
            s.space_amplification(),
            Some(2.0),
            "DefaultStore exposes live_len for tombstone policy"
        );

        s.compact().unwrap();
        assert_eq!(s.tombstone_count(), 0);
        assert_eq!(s.segments()[0].as_ref(), &vec![(1, "a".into())]);
    }

    #[test]
    fn sidecar_envelope_round_trips_payload() {
        let magic = b"SIDETST1";
        let recipe = b"demo-v1";
        let payload = b"payload bytes";

        let bytes = SidecarEnvelope::encode(magic, 2, 7, recipe, payload).unwrap();
        let decoded = SidecarEnvelope::decode(magic, 2, 7, recipe, &bytes).unwrap();

        assert_eq!(decoded, payload);
    }

    #[test]
    fn sidecar_envelope_reports_payload_info_without_full_decode() {
        let magic = b"SIDETST1";
        let recipe = b"demo-v1";
        let payload = b"payload bytes";
        let bytes = SidecarEnvelope::encode(magic, 2, 7, recipe, payload).unwrap();
        let mut reader = &bytes[..];

        let info =
            SidecarEnvelope::payload_info(magic, 2, 7, recipe, &mut reader, bytes.len() as u64)
                .unwrap();

        let expected_offset = (SIDECAR_ENVELOPE_HEADER_BYTES + recipe.len()) as u64;
        assert_eq!(info.payload_offset(), expected_offset);
        assert_eq!(info.payload_len(), payload.len() as u64);
        assert_eq!(
            info.crc32(),
            u32::from_le_bytes(bytes[24..28].try_into().unwrap())
        );
        assert_eq!(&bytes[info.payload_offset() as usize..], payload);
    }

    #[test]
    fn sidecar_envelope_rejects_corrupt_payload_by_crc() {
        let magic = b"SIDETST1";
        let recipe = b"demo-v1";
        let mut bytes = SidecarEnvelope::encode(magic, 2, 7, recipe, b"payload").unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;

        let err = SidecarEnvelope::decode(magic, 2, 7, recipe, &bytes).unwrap_err();
        assert!(
            matches!(err, PersistenceError::CrcMismatch { .. }),
            "corrupt sidecar bytes should fail CRC, got: {err}"
        );
    }

    #[test]
    fn sidecar_envelope_distinguishes_stale_recipe_from_corruption() {
        let magic = b"SIDETST1";
        let bytes = SidecarEnvelope::encode(magic, 2, 7, b"demo-v1", b"payload").unwrap();

        let err = SidecarEnvelope::decode(magic, 2, 7, b"demo-v2", &bytes).unwrap_err();
        assert!(
            err.to_string().contains("recipe mismatch"),
            "valid but stale sidecars should be rebuildable mismatches, got: {err}"
        );
    }

    #[test]
    fn sidecar_envelope_payload_info_distinguishes_stale_recipe() {
        let magic = b"SIDETST1";
        let bytes = SidecarEnvelope::encode(magic, 2, 7, b"demo-v1", b"payload").unwrap();
        let mut reader = &bytes[..];

        let err =
            SidecarEnvelope::payload_info(magic, 2, 7, b"demo-v2", &mut reader, bytes.len() as u64)
                .unwrap_err();
        assert!(
            err.to_string().contains("recipe mismatch"),
            "valid but stale sidecars should be rebuildable mismatches, got: {err}"
        );
    }

    #[test]
    fn sidecar_envelope_payload_info_rejects_corrupt_recipe_length() {
        let magic = b"SIDETST1";
        let mut bytes = SidecarEnvelope::encode(magic, 2, 7, b"demo-v1", b"payload").unwrap();
        bytes[20..24].copy_from_slice(&u32::MAX.to_le_bytes());
        let mut reader = &bytes[..];

        let err =
            SidecarEnvelope::payload_info(magic, 2, 7, b"demo-v1", &mut reader, bytes.len() as u64)
                .unwrap_err();
        assert!(
            err.to_string().contains("recipe is truncated"),
            "corrupt recipe lengths should fail before reading payload bytes, got: {err}"
        );
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
    fn wal_replay_preserves_flush_threshold() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir.clone(), Kv, 3).unwrap();
        for id in 0..65 {
            s.add(id, format!("v{id}")).unwrap();
        }
        drop(s);

        let mut reopened = SegmentedStore::open(dir, Kv, 3).unwrap();
        assert_eq!(reopened.segment_sizes(), vec![3; 21]);
        assert_eq!(reopened.buffer().len(), 2);

        reopened.add(65, "v65".into()).unwrap();
        assert_eq!(reopened.segment_sizes(), vec![3; 22]);
        assert!(reopened.buffer().is_empty());
        assert_eq!(live_set(&reopened).len(), 66);
    }

    #[test]
    fn wal_replay_preserves_checkpoint_visibility() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir.clone(), Kv, 2).unwrap();
        s.add(1, "a".into()).unwrap();
        s.add(2, "b".into()).unwrap();
        s.checkpoint().unwrap();

        s.add(3, "c".into()).unwrap();
        s.add(4, "d".into()).unwrap();
        s.delete(1).unwrap();
        drop(s);

        let mut reopened = SegmentedStore::open(dir, Kv, 2).unwrap();
        assert_eq!(
            live_set(&reopened),
            vec![(2, "b".into()), (3, "c".into()), (4, "d".into())]
        );

        let before_checkpoint = reopened.reader().view();
        assert_eq!(before_checkpoint.segment_count(), 1);
        assert!(before_checkpoint.is_live(&1));
        assert_eq!(before_checkpoint.segment_ids(), &[0]);

        reopened.checkpoint().unwrap();
        let after_checkpoint = reopened.reader().view();
        assert_eq!(after_checkpoint.segment_count(), 2);
        assert!(!after_checkpoint.is_live(&1));
        assert_eq!(after_checkpoint.segment_ids(), &[0, 1]);
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
    fn re_add_of_a_live_id_double_lives_there_is_no_replace() {
        // Liveness is a per-id boolean, and `add` revives an id (clears its
        // tombstone), so a re-add of an already-live id does NOT supersede the old
        // copy: both the old and new values stay live. There is no last-write-wins
        // and no `replace`; a consumer must use unique ids, or compact away the old
        // copy before re-adding.
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
        s.add(1, "a".into()).unwrap();
        s.add(2, "x".into()).unwrap(); // flush=2 seals [(1,"a"),(2,"x")]
        s.add(1, "b".into()).unwrap(); // re-add 1: buffered; segment's (1,"a") survives
        let n_live_1 = live_set(&s).iter().filter(|(id, _)| *id == 1).count();
        assert_eq!(
            n_live_1, 2,
            "a re-add double-lives the id; no last-write-wins"
        );

        // delete-then-add does NOT supersede either: `add` revives (un-tombstones)
        // the old copy, so it comes back alongside the new value.
        s.delete(1).unwrap();
        s.add(1, "c".into()).unwrap();
        let live_1: Vec<String> = live_set(&s)
            .into_iter()
            .filter(|(id, _)| *id == 1)
            .map(|(_, v)| v)
            .collect();
        assert_eq!(
            live_1,
            vec!["a".to_string(), "c".to_string()],
            "delete-then-add revived the old value: no replace in the per-id model"
        );
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
        // Flip a byte in the middle of the manifest payload (past the header).
        let manifest = root.join(MANIFEST_PATH);
        let mut bytes = std::fs::read(&manifest).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        std::fs::write(&manifest, &bytes).unwrap();

        let dir = durability::FsDirectory::arc(&root).unwrap();
        let res = SegmentedStore::open(dir, Kv, 2);
        assert!(
            res.is_err(),
            "a CRC-corrupt manifest is an error, not silently misread"
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
        // A crash mid atomic_write can leave a `<manifest>.tmp` next to the real file.
        std::fs::write(root.join(format!("{MANIFEST_PATH}.tmp")), b"garbage").unwrap();

        let dir = durability::FsDirectory::arc(&root).unwrap();
        let s2 = SegmentedStore::open(dir, Kv, 2).unwrap();
        assert_eq!(live_set(&s2), vec![(1, "a".into()), (2, "b".into())]);
        let _ = std::fs::remove_dir_all(&root);
    }

    // ---- manifest format: incremental write, GC, version rejection ----

    /// A `Directory` that records the path of every mutating write (`atomic_write`
    /// + `create_file`), to assert which files a checkpoint actually rewrites.
    struct CountDir {
        inner: Arc<dyn Directory>,
        writes: Arc<std::sync::Mutex<Vec<String>>>,
    }
    impl CountDir {
        fn wrap(
            inner: Arc<dyn Directory>,
        ) -> (Arc<dyn Directory>, Arc<std::sync::Mutex<Vec<String>>>) {
            let writes = Arc::new(std::sync::Mutex::new(Vec::new()));
            let dir: Arc<dyn Directory> = Arc::new(CountDir {
                inner,
                writes: writes.clone(),
            });
            (dir, writes)
        }
    }
    impl Directory for CountDir {
        fn create_file(&self, p: &str) -> PersistenceResult<Box<dyn std::io::Write + Send>> {
            self.writes.lock().unwrap().push(p.to_string());
            self.inner.create_file(p)
        }
        fn atomic_write(&self, p: &str, d: &[u8]) -> PersistenceResult<()> {
            self.writes.lock().unwrap().push(p.to_string());
            self.inner.atomic_write(p, d)
        }
        fn open_file(&self, p: &str) -> PersistenceResult<Box<dyn std::io::Read + Send>> {
            self.inner.open_file(p)
        }
        fn exists(&self, p: &str) -> bool {
            self.inner.exists(p)
        }
        fn delete(&self, p: &str) -> PersistenceResult<()> {
            self.inner.delete(p)
        }
        fn atomic_rename(&self, a: &str, b: &str) -> PersistenceResult<()> {
            self.inner.atomic_rename(a, b)
        }
        fn create_dir_all(&self, p: &str) -> PersistenceResult<()> {
            self.inner.create_dir_all(p)
        }
        fn list_dir(&self, p: &str) -> PersistenceResult<Vec<String>> {
            self.inner.list_dir(p)
        }
        fn append_file(&self, p: &str) -> PersistenceResult<Box<dyn std::io::Write + Send>> {
            self.inner.append_file(p)
        }
        fn file_path(&self, p: &str) -> Option<std::path::PathBuf> {
            self.inner.file_path(p)
        }
    }

    /// Count the `segstore.seg.*` files currently on disk.
    fn seg_file_count(dir: &Arc<dyn Directory>) -> usize {
        dir.list_dir("")
            .unwrap()
            .into_iter()
            .filter(|n| n.starts_with(SEG_PREFIX))
            .count()
    }

    #[test]
    fn checkpoint_writes_only_new_segment_files() {
        // The point of the manifest format: a checkpoint rewrites only the segments
        // sealed since the last one, not the whole corpus (O(delta), not O(total)).
        let (dir, writes) = CountDir::wrap(MemoryDirectory::arc());
        let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
        for i in 0..4u32 {
            s.add(i, format!("v{i}")).unwrap(); // two sealed segments (ids 0, 1)
        }
        s.checkpoint().unwrap(); // writes seg.0, seg.1, manifest
        writes.lock().unwrap().clear();

        // One more segment, then checkpoint again.
        s.add(4, "e".into()).unwrap();
        s.add(5, "f".into()).unwrap(); // seals a third segment (id 2)
        s.checkpoint().unwrap();

        let w = writes.lock().unwrap();
        let seg_writes: Vec<&String> = w.iter().filter(|p| p.starts_with(SEG_PREFIX)).collect();
        assert_eq!(
            seg_writes.len(),
            1,
            "only the one new segment file is rewritten, not the unchanged two: {:?}",
            *w
        );
        assert!(
            seg_writes[0].ends_with(".2"),
            "and it is the new segment's file: {:?}",
            seg_writes
        );
    }

    #[test]
    fn compaction_gcs_superseded_segment_files() {
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir.clone(), Kv, 2).unwrap();
        for i in 0..8u32 {
            s.add(i, format!("v{i}")).unwrap(); // four segments
        }
        s.checkpoint().unwrap();
        assert_eq!(seg_file_count(&dir), 4, "one file per segment");
        s.compact().unwrap(); // merge to one; the four inputs are superseded
        assert_eq!(s.segment_count(), 1);
        assert_eq!(
            seg_file_count(&dir),
            1,
            "the merged-away segment files are GC'd; only the result remains"
        );
    }

    #[test]
    fn best_effort_delete_batch_syncs_parent_once() {
        struct SyncCountDir {
            inner: Arc<dyn Directory>,
            syncs: std::sync::atomic::AtomicUsize,
        }
        impl Directory for SyncCountDir {
            fn create_file(&self, p: &str) -> PersistenceResult<Box<dyn std::io::Write + Send>> {
                self.inner.create_file(p)
            }
            fn open_file(&self, p: &str) -> PersistenceResult<Box<dyn std::io::Read + Send>> {
                self.inner.open_file(p)
            }
            fn exists(&self, p: &str) -> bool {
                self.inner.exists(p)
            }
            fn delete(&self, p: &str) -> PersistenceResult<()> {
                self.inner.delete(p)
            }
            fn atomic_rename(&self, a: &str, b: &str) -> PersistenceResult<()> {
                self.inner.atomic_rename(a, b)
            }
            fn create_dir_all(&self, p: &str) -> PersistenceResult<()> {
                self.inner.create_dir_all(p)
            }
            fn list_dir(&self, p: &str) -> PersistenceResult<Vec<String>> {
                self.inner.list_dir(p)
            }
            fn append_file(&self, p: &str) -> PersistenceResult<Box<dyn std::io::Write + Send>> {
                self.inner.append_file(p)
            }
            fn atomic_write(&self, p: &str, d: &[u8]) -> PersistenceResult<()> {
                self.inner.atomic_write(p, d)
            }
            fn file_path(&self, p: &str) -> Option<std::path::PathBuf> {
                Some(std::path::PathBuf::from("/tmp/segstore-sync-count").join(p))
            }
            fn durable_sync_parent_dir(&self, _path: &str) -> PersistenceResult<()> {
                self.syncs
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
        }

        let dir = SyncCountDir {
            inner: MemoryDirectory::arc(),
            syncs: std::sync::atomic::AtomicUsize::new(0),
        };
        dir.atomic_write("a", b"a").unwrap();
        dir.atomic_write("b", b"b").unwrap();

        delete_files_best_effort(&dir, vec!["a".into(), "b".into(), "missing".into()], true);
        assert!(!dir.exists("a"));
        assert!(!dir.exists("b"));
        assert_eq!(
            dir.syncs.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "multiple GC deletes share one parent-directory sync"
        );

        delete_files_best_effort(&dir, vec!["missing".into()], true);
        assert_eq!(
            dir.syncs.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "missing files do not force a parent-directory sync"
        );

        dir.atomic_write("c", b"c").unwrap();
        delete_files_best_effort(&dir, vec!["c".into()], false);
        assert_eq!(
            dir.syncs.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "non-durable cleanup does not sync"
        );
    }

    #[test]
    fn index_sidecars_are_gced_with_their_segments() {
        // The persist-index hook: a consumer writes a per-segment index sidecar keyed
        // by a stable seg id; segstore GCs it when the segment is compacted away, and
        // sweeps orphans on open, on the same schedule as the segment files.
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir.clone(), Kv, 2).unwrap();
        for i in 0..8u32 {
            s.add(i, format!("v{i}")).unwrap(); // four segments
        }
        s.checkpoint().unwrap();
        // Simulate the consumer persisting one index sidecar per segment.
        for &id in s.segment_ids() {
            dir.atomic_write(&s.index_name(id, "toy"), b"built-index")
                .unwrap();
        }
        let idx_count = |d: &Arc<dyn Directory>| {
            d.list_dir("")
                .unwrap()
                .into_iter()
                .filter(|n| idx_seg_id(n).is_some())
                .count()
        };
        assert_eq!(idx_count(&dir), 4, "one sidecar per segment");

        // Compaction merges to one segment; the four inputs' sidecars are orphaned.
        s.compact().unwrap();
        assert_eq!(s.segment_count(), 1);
        assert_eq!(
            idx_count(&dir),
            0,
            "sidecars of compacted-away segments are GC'd"
        );

        // A sidecar for a segment that no longer exists is swept on open.
        dir.atomic_write(&s.index_name(9999, "toy"), b"orphan")
            .unwrap();
        drop(s);
        let _ = SegmentedStore::open(dir.clone(), Kv, 2).unwrap();
        assert_eq!(idx_count(&dir), 0, "orphan sidecar GC'd on open");
    }

    #[test]
    fn legacy_sidecar_names_are_still_gced() {
        // `try_index_name` rejects dotted kinds for new writes, but pre-grammar
        // releases allowed callers to write names such as `hnsw.v1`. Treat them
        // as sidecars for GC so upgrades do not strand stale derived artifacts.
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir.clone(), Kv, 2).unwrap();
        for i in 0..4u32 {
            s.add(i, format!("v{i}")).unwrap();
        }
        s.checkpoint().unwrap();
        let stale_id = s.segment_ids()[0];
        dir.atomic_write(&format!("{IDX_PREFIX}{stale_id}.hnsw.v1"), b"old-index")
            .unwrap();

        s.compact().unwrap();

        assert!(
            !dir.exists(&format!("{IDX_PREFIX}{stale_id}.hnsw.v1")),
            "legacy dotted-kind sidecar is removed when its segment is compacted away"
        );
    }

    #[test]
    fn index_sidecar_kind_is_checked() {
        let dir = MemoryDirectory::arc();
        let s = SegmentedStore::open(dir, Kv, 2).unwrap();

        assert_eq!(
            try_index_name(7, "hnsw_v1-fast").unwrap(),
            "segstore.idx.7.hnsw_v1-fast"
        );
        assert_eq!(
            s.try_index_name(7, "hnsw_v1-fast").unwrap(),
            try_index_name(7, "hnsw_v1-fast").unwrap()
        );
        assert_eq!(
            s.index_name(7, "hnsw_v1-fast"),
            index_name(7, "hnsw_v1-fast")
        );

        for kind in ["", "hnsw.v1", "hnsw/tmp", "../hnsw", ".tmp", "hnsw tmp"] {
            assert!(
                matches!(
                    try_index_name(7, kind),
                    Err(PersistenceError::InvalidConfig(_))
                ),
                "kind {kind:?} should be rejected by the free helper"
            );
            assert!(
                matches!(
                    s.try_index_name(7, kind),
                    Err(PersistenceError::InvalidConfig(_))
                ),
                "kind {kind:?} should be rejected"
            );
        }
    }

    #[test]
    #[should_panic(expected = "invalid segstore index kind")]
    fn index_name_panics_on_invalid_kind() {
        let _ = index_name(7, "hnsw.v1");
    }

    #[test]
    #[should_panic(expected = "invalid segstore index kind")]
    fn store_index_name_panics_on_invalid_kind() {
        let dir = MemoryDirectory::arc();
        let s = SegmentedStore::open(dir, Kv, 2).unwrap();
        let _ = s.index_name(7, "hnsw.v1");
    }

    #[test]
    fn index_sidecar_parser_recognizes_legacy_names_for_gc() {
        assert_eq!(idx_seg_id("segstore.idx.7.hnsw_v1-fast"), Some(7));
        assert_eq!(
            idx_seg_id("segstore.idx.7.hnsw.v1"),
            Some(7),
            "legacy dotted kind is still classified for cleanup"
        );

        for name in [
            "segstore.idx.7",
            "segstore.idx.7.",
            "segstore.idx.not-a-number.hnsw",
            "segstore.idx..hnsw",
            "not-segstore.idx.7.hnsw",
        ] {
            assert_eq!(idx_seg_id(name), None, "{name:?} should not parse");
        }
    }

    #[test]
    fn v02_ckpt_format_is_rejected() {
        let dir = MemoryDirectory::arc();
        // Simulate a 0.2 store: a monolithic checkpoint blob, no manifest.
        dir.atomic_write(LEGACY_CKPT_PATH, b"a 0.2 checkpoint blob")
            .unwrap();
        let err = SegmentedStore::open(dir, Kv, 2);
        assert!(
            matches!(err, Err(PersistenceError::InvalidConfig(_))),
            "a 0.2 on-disk store is rejected, not misread"
        );
    }

    #[test]
    fn corrupt_segment_file_is_rejected() {
        let root = temp_root("corrupt-seg");
        {
            let dir = durability::FsDirectory::arc(&root).unwrap();
            let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
            s.add(1, "aaaaaaaa".into()).unwrap();
            s.add(2, "bbbbbbbb".into()).unwrap(); // seals segment id 0
            s.checkpoint().unwrap();
        }
        // Corrupt the one segment file (seg.0); recovery must reject, not misread.
        let seg = root.join(seg_path(0));
        let mut bytes = std::fs::read(&seg).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        std::fs::write(&seg, &bytes).unwrap();

        let dir = durability::FsDirectory::arc(&root).unwrap();
        let res = SegmentedStore::open(dir, Kv, 2);
        assert!(
            res.is_err(),
            "a CRC-corrupt segment file is an error, not silently misread"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn segment_catalog_reads_manifest_without_decoding_segments() {
        let root = temp_root("segment-catalog");
        {
            let dir = durability::FsDirectory::arc(&root).unwrap();
            let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
            s.add(1, "a".into()).unwrap();
            s.add(2, "b".into()).unwrap();
            s.add(3, "c".into()).unwrap();
            s.add(4, "d".into()).unwrap();
            s.delete(2).unwrap();
            s.checkpoint().unwrap();
        }

        // Corrupt one segment file. Opening the full store must reject it, but
        // opening the catalog should still succeed because it reads only the
        // manifest. This is the narrow diagnostic/catalog step, not a query reader.
        let seg = root.join(seg_path(0));
        let mut bytes = std::fs::read(&seg).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        std::fs::write(&seg, &bytes).unwrap();

        let dir = durability::FsDirectory::arc(&root).unwrap();
        let res = SegmentedStore::open(dir.clone(), Kv, 2);
        assert!(
            res.is_err(),
            "full store open decodes segment payloads and rejects corruption"
        );

        let catalog = SegmentCatalog::<u32>::open(dir.clone()).unwrap();
        assert_eq!(catalog.epoch(), 1);
        assert_eq!(catalog.segment_ids(), &[0, 1]);
        assert_eq!(catalog.segment_count(), 2);
        assert_eq!(catalog.tombstone_count(), 1);
        assert!(catalog.is_live(&1));
        assert!(!catalog.is_live(&2));
        assert_eq!(catalog.segment_name(1).unwrap(), seg_path(1));
        assert_eq!(
            catalog.try_index_name(1, "toy").unwrap(),
            "segstore.idx.1.toy"
        );
        assert!(matches!(
            catalog.segment_name(99),
            Err(PersistenceError::NotFound(_))
        ));
        assert!(catalog.open_segment_file(1).is_ok());
        let payload = catalog.read_segment_payload(1).unwrap();
        assert!(
            !payload.is_empty(),
            "payload bytes are available without decoding"
        );
        let seg1: Vec<(u32, String)> = catalog.read_segment(1).unwrap();
        assert_eq!(seg1, vec![(3, "c".into()), (4, "d".into())]);
        assert!(
            catalog.read_segment_payload(0).is_err(),
            "payload reads still validate the segment checkpoint CRC"
        );
        assert!(
            catalog.read_segment::<Vec<(u32, String)>>(0).is_err(),
            "reading the corrupt segment should fail when that segment is requested"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn segment_catalog_payload_info_points_to_validated_payload_bytes() {
        use std::io::{Read, Seek};

        let root = temp_root("segment-payload-info");
        {
            let dir = durability::FsDirectory::arc(&root).unwrap();
            let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
            s.add(1, "a".into()).unwrap();
            s.add(2, "b".into()).unwrap();
            s.checkpoint().unwrap();
        }

        let dir = durability::FsDirectory::arc(&root).unwrap();
        let catalog = SegmentCatalog::<u32>::open(dir).unwrap();
        let seg_id = catalog.segment_ids()[0];
        let info = catalog.segment_payload_info(seg_id).unwrap();
        let payload = catalog.read_segment_payload(seg_id).unwrap();

        assert_eq!(info.payload_len, payload.len());
        assert_eq!(info.payload_crc32, crc32fast::hash(&payload));
        info.verify_payload(&payload).unwrap();

        let truncated = &payload[..payload.len() - 1];
        assert!(matches!(
            info.verify_payload(truncated),
            Err(PersistenceError::Format(_))
        ));

        let mut corrupt = payload.clone();
        corrupt[0] ^= 0xFF;
        assert!(matches!(
            info.verify_payload(&corrupt),
            Err(PersistenceError::CrcMismatch { .. })
        ));

        let path = catalog.segment_file_path(seg_id).unwrap().unwrap();
        let mut file = std::fs::File::open(path).unwrap();
        file.seek(std::io::SeekFrom::Start(info.payload_offset))
            .unwrap();
        let mut from_file = vec![0u8; info.payload_len];
        file.read_exact(&mut from_file).unwrap();
        assert_eq!(from_file, payload);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn segment_catalog_reads_source_payloads_and_exposes_liveness_separately() {
        let dir = MemoryDirectory::arc();
        {
            let mut s = SegmentedStore::open(dir.clone(), Kv, 2).unwrap();
            s.add(1, "a".into()).unwrap();
            s.add(2, "b".into()).unwrap();
            s.delete(2).unwrap();
            s.checkpoint().unwrap();
        }

        let catalog = SegmentCatalog::<u32>::open(dir).unwrap();
        let segment_id = catalog.segment_ids()[0];
        let segment: Vec<(u32, String)> = catalog.read_segment(segment_id).unwrap();

        assert_eq!(segment, vec![(1, "a".into()), (2, "b".into())]);
        assert!(catalog.is_live(&1));
        assert!(!catalog.is_live(&2));
    }

    #[test]
    fn segment_catalog_payload_into_reuses_buffer_and_clears_on_error() {
        use std::io::Read;

        let dir = MemoryDirectory::arc();
        let seg_id = {
            let mut s = SegmentedStore::open(dir.clone(), Kv, 1).unwrap();
            s.add(1, "a".into()).unwrap();
            s.checkpoint().unwrap();
            s.segment_ids()[0]
        };
        let catalog = SegmentCatalog::<u32>::open(dir.clone()).unwrap();

        let mut scratch = Vec::with_capacity(1024);
        let ptr = scratch.as_ptr();
        catalog
            .read_segment_payload_into(seg_id, &mut scratch)
            .unwrap();
        assert_eq!(
            scratch.as_ptr(),
            ptr,
            "a sufficiently large scratch buffer is reused"
        );
        let good_payload = scratch.clone();
        scratch.extend_from_slice(b"stale suffix");
        catalog
            .read_segment_payload_into(seg_id, &mut scratch)
            .unwrap();
        assert_eq!(scratch, good_payload, "the destination is cleared first");

        let name = seg_path(seg_id);
        let mut bytes = Vec::new();
        dir.open_file(&name)
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        dir.atomic_write(&name, &bytes).unwrap();

        let err = catalog
            .read_segment_payload_into(seg_id, &mut scratch)
            .unwrap_err();
        assert!(
            matches!(err, PersistenceError::CrcMismatch { .. }),
            "payload corruption is still detected: {err}"
        );
        assert!(
            scratch.is_empty(),
            "partial/corrupt reads do not leak bytes"
        );
    }

    #[test]
    fn segment_catalog_rejects_legacy_formats() {
        let dir = MemoryDirectory::arc();
        dir.atomic_write(LEGACY_CKPT_PATH, b"a 0.2 checkpoint blob")
            .unwrap();
        let err = SegmentCatalog::<u32>::open(dir);
        assert!(
            matches!(err, Err(PersistenceError::InvalidConfig(_))),
            "legacy monolithic checkpoints are rejected consistently"
        );
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
        assert_eq!(
            seg_file_count(&mem),
            0,
            "no leaked segment files after delete-all + compact"
        );
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
        s.checkpoint().unwrap(); // commit-visibility: publish to readers
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
            "view = checkpointed segments minus tombstones"
        );
    }

    #[test]
    fn view_is_a_stable_snapshot_across_writes() {
        let mut s = SegmentedStore::open(MemoryDirectory::arc(), Kv, 2).unwrap();
        for i in 0..6u32 {
            s.add(i, format!("v{i}")).unwrap(); // 3 segments
        }
        s.checkpoint().unwrap(); // publish the 3 segments to readers
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
    fn unchanged_segments_keep_arc_identity_across_checkpoints() {
        // The point of Arc-internal segments: an unchanged segment keeps its Arc
        // identity across mutations, so a consumer can cache per-segment state keyed
        // by it and rebuild only new segments.
        let mut s = SegmentedStore::open(MemoryDirectory::arc(), Kv, 2).unwrap();
        for i in 0..6u32 {
            s.add(i, format!("v{i}")).unwrap(); // 3 segments
        }
        s.checkpoint().unwrap();
        let r = s.reader();
        let ptrs1: Vec<*const _> = r.view().segments().iter().map(Arc::as_ptr).collect();
        assert_eq!(ptrs1.len(), 3);

        // A no-op checkpoint must keep every segment's identity.
        s.checkpoint().unwrap();
        let ptrs2: Vec<*const _> = r.view().segments().iter().map(Arc::as_ptr).collect();
        assert_eq!(
            ptrs1, ptrs2,
            "a no-op checkpoint keeps all segment identities"
        );

        // Sealing a new segment leaves the originals' identities intact.
        s.add(10, "x".into()).unwrap();
        s.add(11, "y".into()).unwrap(); // seals a 4th segment
        s.checkpoint().unwrap();
        let ptrs3: Vec<*const _> = r.view().segments().iter().map(Arc::as_ptr).collect();
        assert_eq!(ptrs3.len(), 4);
        assert_eq!(
            &ptrs3[..3],
            &ptrs1[..],
            "the original segments keep identity; only the new one is fresh"
        );
    }

    #[test]
    fn segment_ids_align_and_are_stable_across_checkpoint_and_reopen() {
        // The stable-id contract the persist-index hook rests on: ids align 1:1 with
        // segments, an unchanged segment keeps its id (so a persisted-index cache key
        // does not move), and ids survive a restart (unlike the Arc pointer).
        let dir = MemoryDirectory::arc();
        let mut s = SegmentedStore::open(dir.clone(), Kv, 2).unwrap();
        for i in 0..6u32 {
            s.add(i, format!("v{i}")).unwrap(); // 3 segments
        }
        s.checkpoint().unwrap();
        assert_eq!(s.segment_ids().len(), s.segments().len());
        assert_eq!(s.segment_ids().len(), 3);
        let ids0: Vec<u64> = s.segment_ids().to_vec();
        // The concurrent View exposes the same ids as the writer.
        assert_eq!(s.reader().view().segment_ids(), &ids0[..]);

        // A no-op checkpoint keeps every id (the cache key must not move on no change).
        s.checkpoint().unwrap();
        assert_eq!(s.segment_ids(), &ids0[..], "no-op checkpoint keeps ids");

        // Sealing a new segment keeps the originals' ids and appends a fresh one.
        s.add(10, "x".into()).unwrap();
        s.add(11, "y".into()).unwrap(); // seals a 4th segment
        s.checkpoint().unwrap();
        assert_eq!(&s.segment_ids()[..3], &ids0[..], "old ids unchanged");
        assert_eq!(s.segment_ids().len(), 4);
        assert!(
            !ids0.contains(&s.segment_ids()[3]),
            "the new segment got a fresh, never-reused id"
        );

        // Ids survive a restart: a consumer keying a persisted index cache on these
        // reloads it instead of rebuilding from the raw payload.
        let before: Vec<u64> = s.segment_ids().to_vec();
        drop(s);
        let s2 = SegmentedStore::open(dir, Kv, 2).unwrap();
        assert_eq!(
            s2.segment_ids(),
            &before[..],
            "segment ids are stable across a restart (persisted in the manifest)"
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

    #[derive(Debug, Clone)]
    enum CatalogOp {
        Add,
        Delete(usize),
        Checkpoint,
        Compact,
        CompactTiers,
        Reopen,
    }

    fn catalog_op_strategy() -> impl Strategy<Value = CatalogOp> {
        prop_oneof![
            3 => Just(CatalogOp::Add),
            2 => (0usize..100).prop_map(CatalogOp::Delete),
            2 => Just(CatalogOp::Checkpoint),
            1 => Just(CatalogOp::Compact),
            1 => Just(CatalogOp::CompactTiers),
            1 => Just(CatalogOp::Reopen),
        ]
    }

    fn catalog_expectation(
        s: &SegmentedStore<Kv>,
    ) -> (u64, u64, Vec<u64>, std::collections::HashSet<u32>) {
        (
            s.epoch,
            s.next_seg_id,
            s.segment_ids.clone(),
            s.tombstones.clone(),
        )
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
                // segment_ids must stay aligned 1:1 with segments under every op,
                // including across Reopen (the persist-index cache key invariant).
                prop_assert_eq!(s.segment_ids().len(), s.segment_count());
            }
            let want: Vec<(u32, String)> = model.into_iter().collect();
            prop_assert_eq!(live_set(&s), want);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]
        /// The catalog is a manifest reader: it tracks the last checkpointed
        /// segment ids/tombstones exactly, while ignoring WAL-only suffix records.
        #[test]
        fn segment_catalog_matches_last_checkpoint_under_random_ops(
            ops in proptest::collection::vec(catalog_op_strategy(), 0..160)
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
            let mut checkpointed = catalog_expectation(&s);
            let mut live_ids: Vec<u32> = Vec::new();
            let mut next_id = 0u32;

            for op in ops {
                match op {
                    CatalogOp::Add => {
                        let id = next_id;
                        next_id += 1;
                        s.add(id, format!("v{id}")).unwrap();
                        live_ids.push(id);
                    }
                    CatalogOp::Delete(k) => {
                        if !live_ids.is_empty() {
                            let id = live_ids.swap_remove(k % live_ids.len());
                            s.delete(id).unwrap();
                        }
                    }
                    CatalogOp::Checkpoint => {
                        s.checkpoint().unwrap();
                        checkpointed = catalog_expectation(&s);
                    }
                    CatalogOp::Compact => {
                        s.compact().unwrap();
                        checkpointed = catalog_expectation(&s);
                    }
                    CatalogOp::CompactTiers => {
                        if s.compact_tiers().unwrap().merges > 0 {
                            checkpointed = catalog_expectation(&s);
                        }
                    }
                    CatalogOp::Reopen => {
                        s = SegmentedStore::open_with_options(dir.clone(), Kv, mk()).unwrap();
                    }
                }

                let catalog = SegmentCatalog::<u32>::open(dir.clone()).unwrap();
                prop_assert_eq!(catalog.epoch(), checkpointed.0);
                prop_assert_eq!(catalog.next_segment_id(), checkpointed.1);
                prop_assert_eq!(catalog.segment_ids(), &checkpointed.2[..]);
                prop_assert_eq!(catalog.segment_count(), checkpointed.2.len());
                prop_assert_eq!(catalog.tombstone_count(), checkpointed.3.len());
                for id in 0..next_id {
                    prop_assert_eq!(catalog.is_live(&id), !checkpointed.3.contains(&id));
                }
                for &seg_id in catalog.segment_ids() {
                    prop_assert_eq!(catalog.segment_name(seg_id).unwrap(), seg_path(seg_id));
                    prop_assert!(catalog.open_segment_file(seg_id).is_ok());
                    let _: Vec<(u32, String)> = catalog.read_segment(seg_id).unwrap();
                }
                prop_assert!(matches!(
                    catalog.segment_name(u64::MAX),
                    Err(PersistenceError::NotFound(_))
                ));
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]
        /// Any single-byte manifest corruption must be detected before the
        /// catalog exposes segment ids or tombstones.
        #[test]
        fn segment_catalog_rejects_manifest_corruption_at_any_byte(offset in 0usize..256) {
            use std::io::Read;

            let dir = MemoryDirectory::arc();
            {
                let mut s = SegmentedStore::open(dir.clone(), Kv, 2).unwrap();
                s.add(1, "a".into()).unwrap();
                s.add(2, "b".into()).unwrap();
                s.delete(1).unwrap();
                s.checkpoint().unwrap();
            }

            let mut file = dir.open_file(MANIFEST_PATH).unwrap();
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes).unwrap();
            let header_len = durability::checkpoint::CheckpointHeader::SIZE;
            prop_assume!(bytes.len() > header_len);
            let i = header_len + (offset % (bytes.len() - header_len));
            bytes[i] ^= 0x80;
            dir.atomic_write(MANIFEST_PATH, &bytes).unwrap();

            prop_assert!(
                SegmentCatalog::<u32>::open(dir).is_err(),
                "corrupt manifest byte {i} was accepted"
            );
        }
    }
}
