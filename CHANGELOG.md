# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The 0.x series is
unstable: minor bumps may break the public API and the on-disk format.

## [Unreleased]

### Changed

- A segment or manifest that serializes past the 256 MiB per-blob checkpoint
  cap now fails the checkpoint write with an actionable error naming the
  artifact and the levers (`TierConfig::max_merged_len`, the `open()` flush
  threshold, `compact()` to drop tombstones) instead of durability's opaque
  "checkpoint payload too large". `TierConfig::max_merged_len` docs now note it
  counts items while the blob cap counts bytes, and that there is no byte-aware
  merge planning yet. This does not raise the cap or change behavior on success.
- `SegmentCatalog` now validates segment ids with binary search instead of a
  linear scan. In the many-segment catalog benchmark, naming every segment moved
  from `[979.29 us 983.18 us 988.99 us]` to
  `[296.10 us 298.27 us 300.57 us]` for 4,000 segments, while catalog-open time
  stayed flat.
- Batched best-effort GC of stale WAL, segment, and index-sidecar files so
  filesystem-backed cleanup syncs the parent directory once per batch instead of
  leaving deletion durability to a later operation.
- Documented the current memory boundary: segstore persists immutable segments
  per file, but `SegmentedStore::open` still loads manifest segments into
  `Arc<Segment>` memory. Larger-than-memory readers need a future file/mmap-backed
  segment-reader API rather than the current in-memory `View::segments` model.
- Refined the out-of-core reader design around backend capabilities (memory,
  local filesystem/mmap, range/vectored reads, and object-store publish) and
  mapped the raw-segment path across postings, lexir, sporse, vicinity, precinct,
  sketchir, gramdex, and artifact/generation-store consumers.
- Added an I/O-classification matrix to the out-of-core reader design, tying
  memory, local SSD, slow/network filesystem, object-store/range, and tiered
  search backends to their expected access patterns and evidence gates.
- Extended the out-of-core design with operation classes and store-family
  boundaries, separating local durability, segment lifecycle, raw segment
  readers, materialized logs, generation/artifact snapshots, and rebuildable
  caches.

### Added

- Filesystem sidecar-GC compaction benchmark to track the real directory-sync
  cost of deleting stale segment and index files after checkpoint commits.
- `SegmentCatalog<Id>` for reading the checkpoint manifest without decoding
  segment payload files. It exposes stable segment ids, tombstone checks, segment
  file names/paths, and sidecar names for diagnostic and restart-time loader
  code. This is a catalog helper, not yet a byte-native out-of-core query API.
- `SegmentCatalog::read_segment(id)` for decoding one checkpointed segment by
  stable id, so restart-time sidecar builders can load only the segment they
  need instead of opening the full in-memory store.
- `SegmentCatalog::read_segment_payload(id)` for reading one segment's
  CRC-validated serialized payload bytes without deserializing `Store::Segment`.
- Module-level `try_index_name(id, kind)` and `index_name(id, kind)` helpers so
  reader/searcher code can load sidecars from `View::segment_ids()` without
  holding a writer `SegmentedStore`.

## [0.4.0] - 2026-06-29

### Added

- Persist-the-built-index hook (so recovery loads a consumer's per-segment index
  instead of rebuilding it from the raw payload on every restart):
  - `segment_ids()` (on `SegmentedStore` and `View`): the stable per-segment id,
    parallel to `segments()`, that names the `segstore.seg.<id>` file and survives a
    restart (unlike the `Arc` pointer). A consumer keys a persisted index cache on it.
  - `dir()` and `index_name(id, kind)`: write a built per-segment index into the
    reserved `segstore.idx.<id>.<kind>` namespace via the store's `Directory`.
    segstore garbage-collects each sidecar when its segment is compacted away, and
    sweeps orphans on open, on the same crash-safe schedule as the segment files.
    The consumer owns the sidecar's encoding (segstore never reads it). No `Store`
    trait change.

## [0.3.0] - 2026-06-28

### Changed

- Incremental checkpoints. A checkpoint now persists each new segment to its own
  `segstore.seg.<id>` file and atomically publishes a `segstore.manifest` naming the
  current segment files + tombstones, instead of re-serializing the whole corpus into
  one blob on every checkpoint. Only newly-sealed segments are written, so a checkpoint
  is O(new data), not O(total): the Lucene `segments_N` / RocksDB MANIFEST model.
  Segment files a merge supersedes are garbage-collected after the manifest is durable.

### Breaking

- On-disk format: 0.3 replaces 0.2's single monolithic `segstore.ckpt` checkpoint blob
  with the `segstore.manifest` + per-segment `segstore.seg.<id>` layout above. A 0.2
  store (a `segstore.ckpt` with no manifest) is detected and rejected with a clear error
  rather than misread.
- `Store::merge_segments` now takes `segments: &[&Self::Segment]` (was `&[Self::Segment]`),
  so `segstore` passes its `Arc`-held segments to the consumer's merge without cloning the
  payloads (it previously deep-cloned every merged segment on each compaction to satisfy
  the owned-slice signature). Migration: a `segs.iter().flatten()...` body becomes
  `segs.iter().flat_map(|s| s.iter())...`; the yielded item type and merge semantics are
  unchanged.

## [0.2.0] - 2026-06-26

### Added

- Concurrent snapshot reads: `reader()` returns a cloneable, thread-safe `Reader`;
  `Reader::view()` takes a consistent point-in-time `View` (Arc-shared segments +
  tombstones) that a query holds lock-free while the writer adds/deletes/compacts
  concurrently (single-writer, many-readers). Commit-style visibility: a view
  reflects state as of the last `checkpoint()`.
- Size-tiered compaction: `compact_tiers()` runs Cassandra/Lucene-style size-tiered
  merges (size-banded buckets, `min_merge`/`max_merge`, a `max_merged_len` cap that
  freezes the largest segment out of tiering), tuned by `TierConfig`. Scheduling is
  consumer-driven, or set `Options::auto_compact` to merge inline after a flush.
- `extend(items)`: bulk-ingest that syncs the WAL once per batch instead of per item
  (the index build phase); a crash mid-batch leaves a consistent prefix.
- `force_merge_to(n)`: on-demand consolidation to at most `n` segments.
- Tombstone reclamation: an optional `Store::live_len` (default `None`) enables
  `space_amplification()` and `reclaim_tombstones(min_live_ratio)`, which rewrites
  only tombstone-heavy segments.
- `SyncPolicy::Fsync` (via `open_with_options`): fsync every WAL record to stable
  storage on a filesystem backend. The default `Flush` is unchanged.
- Instrumentation: `compact()`/`compact_tiers()`/`force_merge_to()` return
  `CompactionStats`; `segment_sizes()`, `stored_len()`, `epoch()`.

### Changed

- The WAL is rotated per checkpoint: a checkpoint starts a fresh epoch-suffixed log
  and deletes the old one, so the log no longer grows unbounded mid-process. Recovery
  replays only the current epoch's WAL; a stale WAL left by a crash mid-rotation is
  ignored and garbage-collected on open.
- Checkpoints publish atomically (CRC-checked) and, on a filesystem backend, pass an
  fsync barrier on the checkpoint file and its parent directory.
- `compact()` now returns `CompactionStats`.

### Fixed

- A full compaction of an all-deleted store no longer leaves an empty segment, and
  `compact()` purges the tombstone set unconditionally (no stale tombstones for ids
  that were only ever buffered).

### Removed

- `open_with_sync` (folded into `open_with_options`, which takes an `Options` struct).

### Breaking

- `Store` now requires `segment_len(&Segment) -> usize` (the size metric size-tiered
  compaction groups by; `seg.len()` for a `Vec`-backed segment).
- `segments()` now returns `&[Arc<Segment>]` (was `&[Segment]`): segments are `Arc`-shared
  so an unchanged segment keeps a stable identity across mutations, letting a consumer
  cache per-segment state keyed by `Arc::as_ptr` and rebuild only new segments.
- On-disk format changed (epoch-suffixed WAL files; the checkpoint records the epoch
  rather than an op count). A 0.1.0 store is detected and rejected with a clear error
  rather than misread.

## [0.1.0] - 2026-06-26

Initial release: the `Store` trait, `SegmentedStore` with a write-ahead log,
immutable segments, tombstone deletes, checkpoint snapshots, full compaction, and
crash recovery, on top of `durability`.
