# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The 0.x series is
unstable: minor bumps may break the public API and the on-disk format.

## [Unreleased]

### Added

- Concurrent snapshot reads: `reader()` returns a cloneable, thread-safe `Reader`;
  `Reader::view()` takes a consistent point-in-time `View` (Arc-shared segments +
  tombstones) that a query holds lock-free while the writer adds/deletes/compacts
  concurrently (single-writer, many-readers). Commit-style visibility: a view sees
  sealed segments + tombstones.
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
- On-disk format changed (epoch-suffixed WAL files; the checkpoint records the epoch
  rather than an op count). A 0.1.0 store is detected and rejected with a clear error
  rather than misread.

## [0.1.0] - 2026-06-26

Initial release: the `Store` trait, `SegmentedStore` with a write-ahead log,
immutable segments, tombstone deletes, checkpoint snapshots, full compaction, and
crash recovery, on top of `durability`.
