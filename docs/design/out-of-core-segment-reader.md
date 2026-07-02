---
status: proposed
date: 2026-07-01
scope: segstore out-of-core reads, raw segment formats, consumer storage
---

# Design: Out-Of-Core Segment Reader

## Problem

`segstore` now writes one immutable file per segment, but its public model is
still in-memory: `SegmentedStore::open` deserializes every `Store::Segment` into
an `Arc`, and `Reader::view` snapshots those loaded values. That is good for
small embedded indexes and restart-stable sidecar caches, but it does not let
`lexir`, `postings`, `sporse`, `vicinity`, or `precinct` search a corpus larger
than memory.

The tempting fix is to expose `segstore.seg.<id>` paths. That is insufficient.
Current segment files are `durability::CheckpointFile` envelopes containing a
postcard-encoded `Store::Segment`. A consumer that needs mmap, byte ranges,
block-max postings, disk graph pages, or quantized vector codes needs to own the
segment byte format directly. A path to a postcard blob only moves the full
decode somewhere else.

## Context

Existing local contracts:

- `durability::Directory` exposes whole-file reads and optional filesystem paths.
  With its `mmap` feature, `durability::mmap::MappedFile` can map a file and
  apply advisory access hints. It does not expose byte-range reads today.
- `segstore` owns the mutable lifecycle: WAL epoch, manifest commit, stable
  segment ids, tombstones, compaction, reader snapshots, and sidecar garbage
  collection.
- Consumer sidecars are rebuildable caches. They may be mmap-friendly, but they
  are not the raw segment source of truth.
- `Store::Segment: Serialize + DeserializeOwned` is the current in-memory
  contract. It is the wrong place to hide disk-page layout.

External evidence points the same way:

- Tantivy searchers hold snapshots of immutable segment readers, and its on-disk
  data structures are read through segment readers rather than by loading a whole
  segment into anonymous memory. See Tantivy's architecture notes:
  <https://github.com/quickwit-oss/tantivy/blob/main/ARCHITECTURE.md>.
- Rust `object_store` exposes backend capabilities such as conditional writes,
  range reads, and vectored reads. Those are backend-access primitives, not a
  search-index data model:
  <https://github.com/apache/arrow-rs-object-store/blob/main/_autodocs/api-reference/objectstore-core.md>.
- Disk ANN work such as SPANN keeps centroids in memory and large posting lists
  on disk, while DiskANN-family work makes page and graph layout part of the
  algorithm. This argues for consumer-owned bytes and metadata, not a generic
  `DeserializeOwned` segment.

## Non-Goals

- Do not make the existing `Store` trait pretend to be out-of-core by exposing
  segment file paths. Its segment format is still postcard-owned.
- Do not add `object_store` or another backend dependency to `segstore` v1.
  Local filesystem and `durability::Directory` are enough for the first cut.
- Do not make `segstore` parse postings blocks, HNSW pages, IVF lists, codebooks,
  or positional payloads.
- Do not manifest-track rebuildable sidecars. That remains a separate gate for
  non-rebuildable or cross-engine derived artifacts.
- Do not solve out-of-core compaction for every algorithm in the first API. The
  read substrate and file-pinning contract come first.

## Options Considered

### Expose segment paths from the current `SegmentedStore`

Rejected as the main answer. It would help consumers find files, but the file
payload is still a checkpoint-wrapped postcard `Store::Segment`. Consumers that
need mmap-friendly postings or disk graph pages would either decode the whole
payload or reverse-engineer a serialization envelope that was not designed as
their query format.

This can still be useful as a small catalog helper for diagnostics and sidecar
lookup, but it should not be named or documented as the out-of-core reader.

### Extend `Store` with byte hooks

Rejected for now. Adding methods like `build_segment_bytes`,
`open_segment_reader`, and `merge_segment_files` to `Store` would couple two
different contracts: the existing in-memory `Segment` API and a byte-format API.
The default implementations would be either fake or full-decode fallbacks, which
would hide the performance cliff the API is supposed to remove.

### Add a parallel raw-segment API

Chosen. Keep `SegmentedStore<S: Store>` as the in-memory API. Add a separate
raw-segment path whose source of truth is a consumer-owned byte payload plus
manifest metadata. The consumer owns encoding and query readers. `segstore`
owns ids, tombstones, manifests, WAL replay, compaction scheduling, snapshot
pinning, and garbage collection.

## Chosen Approach

Add a new raw-segment layer in phases.

### Phase 1: Manifest/catalog reader

Expose the checkpointed segment set without deserializing segment payloads:

- `SegmentCatalog<Id>`: loaded from `segstore.manifest`, containing epoch,
  segment ids, tombstones, and helper methods for segment and sidecar names.
- Visibility is the same as today's `Reader::view`: last checkpoint only. WAL
  suffix records are buffered adds/deletes that have not become immutable
  segment files yet.
- This catalog is read-only and diagnostic unless paired with a raw-segment
  format. It should be documented as a catalog, not an out-of-core search API.

This phase is useful because consumers can build restart-time sidecar loaders
without opening a writer, and tests can exercise manifest compatibility without
forcing segment deserialization. It does not by itself make existing postcard
segments mmap-friendly.

### Phase 2: Raw segment trait

Introduce a separate trait for consumers whose segment files are byte-native:

```rust
pub trait RawSegmentStore {
    type Id;
    type Item;
    type SegmentMeta;

    fn build_segment(&self, batch: &[(Self::Id, Self::Item)])
        -> PersistenceResult<(Self::SegmentMeta, Vec<u8>)>;

    fn segment_len(&self, meta: &Self::SegmentMeta) -> usize;

    fn open_reader<'a>(
        &self,
        segment: RawSegmentRef<'a, Self::SegmentMeta>,
    ) -> PersistenceResult<Box<dyn RawSegmentReader<Id = Self::Id> + 'a>>;
}
```

The exact names can change during implementation. The load-bearing properties
should not:

- the manifest stores `SegmentMeta` beside each segment id;
- the segment payload is written as consumer-owned bytes, not postcard
  `Store::Segment`;
- readers get a stable id, metadata, and byte access;
- compaction is a later extension that can require a consumer merge method over
  raw readers.

`SegmentMeta` is where `postings` records item count, doc-id coding, block size,
field/schema ids, and statistics offsets; `vicinity` records dimensions, metric,
vector count, and layout version; `sporse` records sparse dimensionality and
nonnegative-weight assumptions; `precinct` records region and lift metadata.

### Phase 3: Pinned file views

Out-of-core snapshots need a replacement for the safety `Arc<Segment>` gives
today. A file-backed view must keep the files it references alive while a query
is running.

The portable contract should be explicit pinning, not "the OS keeps deleted
open files alive":

- `RawReader::view()` returns a pinned segment set with ids, metadata, tombstone
  snapshot, and byte/file access.
- `segstore`'s GC skips segment and sidecar files whose ids are currently pinned.
- Dropping the view releases the pin; a later checkpoint or GC pass may delete
  unpinned obsolete files.

The implementation can still open files or mmaps under the hood, but the public
invariant is pin-based. This matters for Windows and for consumers that open
sidecars lazily after creating a view.

## Tradeoffs

This keeps the existing API honest: users with in-memory segments keep the small
`Store` contract, while larger consumers opt into a byte-format API that makes
layout decisions explicit. The cost is a second public surface and likely a
manifest-format bump when raw segment metadata is added.

The raw path also shifts more work to consumers. That is the point. A postings
reader, an HNSW disk graph reader, and an IVF/PQ reader have different page
shapes, warmup patterns, metadata, and query-time pruning. A generic
deserialization API would flatten those differences exactly where performance
depends on them.

## Implementation Plan

1. Add `SegmentCatalog<Id>` over the existing manifest format. Tests should prove
   it opens a manifest without reading or decoding any `segstore.seg.<id>` file.
2. Add a low-level mapped/checkpoint-payload helper only if a consumer can use it
   without full decode. Do not expose it as a general search reader.
3. Design the raw-segment manifest extension: segment id plus `SegmentMeta`, with
   compatibility rejection for old manifests that lack metadata.
4. Implement one consumer first, most likely `postings`, because block metadata,
   positional payloads, and top-k pruning give measurable correctness and
   performance gates.
5. Add pinned raw views before any live writer exposes file-backed concurrent
   readers.
6. Only after one raw consumer works, consider object-store-style range reads or
   vectored reads as a backend capability.

## Decision Gates

- If no consumer can show a benchmark where full segment decode dominates, stop
  at the catalog helper.
- If `postings` cannot express its block metadata and positional payloads through
  `SegmentMeta` without awkward escape hatches, redesign the raw trait before a
  second consumer adopts it.
- If a live writer needs out-of-core readers before raw compaction exists, ship
  pinning first and leave compaction manual.
- If two consumers need remote/object storage, design a backend capability trait
  around create/update/range/vectored reads. Do not add it speculatively.
- If a consumer creates a non-rebuildable derived index, route it to the
  generation/artifact-store design instead of segstore sidecars.

## Open Questions

- Should raw segment payloads keep the `CheckpointFile` CRC envelope, or should
  the raw path use a smaller header that validates metadata plus payload in one
  mmap-friendly parse?
- Should `SegmentMeta` live in the main manifest or in a per-segment descriptor
  file named by the manifest?
- Should raw compaction be mandatory before the first release of the raw API, or
  can the first consumer use force-merge/build-time generation only?
- Which consumer should be the first implementation gate: `postings` for exact
  lexical/sparse evidence, or `vicinity` for ANN restart and disk-graph pressure?
