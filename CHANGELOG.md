# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The 0.x series is
unstable: minor bumps may break the public API and the on-disk format.

## [Unreleased]

### Added

- Size-tiered compaction: `compact_tiers()` runs Cassandra/Lucene-style size-tiered
  merges (size-banded buckets, `min_merge`/`max_merge`, a `max_merged_len` cap that
  freezes the largest segment out of tiering), tuned by `TierConfig`. Scheduling is
  consumer-driven, or set `Options::auto_compact` to merge inline after a flush.
- `SyncPolicy::Fsync` (via `open_with_options`): fsync every WAL record to stable
  storage on a filesystem backend. The default `Flush` is unchanged.
- Instrumentation: `compact()` and `compact_tiers()` return `CompactionStats`;
  `segment_sizes()`, `stored_len()`, and `epoch()` expose the segment-count and
  merge-cost signal to watch as the corpus grows.

### Changed

- The WAL is rotated per checkpoint: a checkpoint starts a fresh epoch-suffixed log
  and deletes the old one, so the log no longer grows unbounded mid-process. Recovery
  replays only the current epoch's WAL; a stale WAL left by a crash mid-rotation is
  ignored and garbage-collected on open.
- Checkpoints publish atomically (CRC-checked) and, on a filesystem backend, pass an
  fsync barrier on the checkpoint file and its parent directory.
- `compact()` now returns `CompactionStats`.

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
