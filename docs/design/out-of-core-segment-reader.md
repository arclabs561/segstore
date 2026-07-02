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

The actual requirement is broader than restart-time persistence. These crates
need bounded memory across every stage: indexing, updating, deleting, searching,
sidecar rebuild, compaction, and long-term maintenance. A design that only keeps
startup fast by loading persisted sidecars still fails once the sidecars or raw
segments themselves exceed RAM.

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

The backend matters. Some `durability` and `segstore` ideas only make sense for
particular I/O classes:

- Memory directory: useful for tests, fuzzing, examples, and deterministic crash
  modeling. It has no fsync, mmap, page cache, true deletion durability, or
  range I/O cost model. Performance numbers from it do not predict filesystem or
  object-store behavior.
- Local filesystem on SSD/NVMe: useful for WAL fsync, atomic rename, parent-dir
  fsync, mmap, page-cache reuse, sequential compaction, and small random reads.
  This is the first real out-of-core target.
- Slow disks or network filesystems: similar API surface, but random I/O, sync
  costs, and sometimes rename/fsync semantics differ from local SSDs. Treat these
  as explicit backend targets. Search readers need larger blocks, prefetch,
  batching, and fewer metadata round trips.
- Object store: useful for immutable blobs, conditional publish, bulk/range
  reads, vectored reads, multipart writes, and generation manifests. It does not
  behave like a seekable file with cheap per-query random reads. Search over
  object storage needs large coalesced ranges and a different cache policy.

That argues for capability-driven APIs, not a single `Directory` performance
story. `segstore` should keep the local crash-consistency contract small while
making it possible to add range/vectored/object-store capabilities later.

The useful classification is not just slow versus fast storage. It is which
access pattern is cheap enough to design around:

| Backend class | Cheap path | Expensive path | Algorithmic response | Evidence required |
| --- | --- | --- | --- | --- |
| Memory-only test backend | Whole-object reads, deterministic failure modeling | None of fsync, mmap, page cache, deletion durability, or I/O latency is real | Use for correctness, fuzzing, and crash-model tests only | Unit/property tests; never performance claims |
| Local SSD/NVMe filesystem | mmap, page-cache reuse, sequential compaction, bounded random reads | Excessive small-file metadata churn, unbounded random page faults, per-record fsync | Page/block layouts, hot dictionaries, advisory access hints, background compaction | Criterion benchmarks on filesystem `Directory`; mmap/page-fault-sensitive tests when relevant |
| HDD or network filesystem | Large sequential reads and writes | Fine-grained random reads, frequent syncs, metadata round trips | Larger blocks, prefetch, merge scans, fewer files, fewer directory operations | Backend-named benchmarks; no extrapolation from SSD |
| Object store / HTTP range backend | Immutable blobs, multipart writes, conditional publish, coarse range/vectored reads | Seek-like small random reads, directory semantics, rename/fsync assumptions | Coalesced range planning, local cache, generation manifests, conditional publication | Range/vectored benchmarks; publish-race tests |
| Tiered vector/search system | Hot routers/centroids/dictionaries in RAM or SSD cache; cold vectors/postings in large blocks | Fetch-to-discard reranking, graph traversal that reads full vectors too early | Consumer-owned layout: SPANN-style hot centroids plus cold lists, graph topology separated from heavy vectors, postings block summaries | Recall/latency/memory benchmarks with a named tier mix |

External evidence points the same way:

- Tantivy searchers hold snapshots of immutable segment readers, and its on-disk
  data structures are read through segment readers rather than by loading a whole
  segment into anonymous memory. See Tantivy's architecture notes:
  <https://github.com/quickwit-oss/tantivy/blob/main/ARCHITECTURE.md>.
- Rust `object_store` exposes backend capabilities such as conditional writes,
  range reads, and vectored reads. Those are backend-access primitives, not a
  search-index data model:
  <https://docs.rs/object_store>.
- Apache OpenDAL makes the same capability distinction from another direction:
  its design guidance is to describe service capabilities and use native backend
  features when possible:
  <https://opendal.apache.org/docs/vision/>.
- A 2026 vector-search storage survey frames the field as memory-resident,
  static memory-SSD, and elastic memory-SSD-object-store architectures. That is
  a useful mental model for `vicinity` and `precinct`: storage tiering changes
  the ANN algorithm, not just its persistence backend:
  <https://arxiv.org/html/2601.01937v1>.
- Disk ANN work such as SPANN keeps centroids in memory and large posting lists
  on disk, while DiskANN-family work makes page and graph layout part of the
  algorithm. This argues for consumer-owned bytes and metadata, not a generic
  `DeserializeOwned` segment.
- Learned-sparse retrieval work keeps the inverted index central but adds
  Block-Max pruning, query-term pruning, and high-document-frequency term
  handling. That belongs in `postings`/`sporse` byte layouts, not in `segstore`:
  <https://arxiv.org/html/2405.01117v1>.
- RISE shows current Rust inverted-index prior art with compressed postings,
  DAAT, WAND, MaxScore, and Block-Max variants. This raises the bar for
  `postings`: a raw segment format should carry block metadata and compression
  choices explicitly, not through a generic store wrapper:
  <https://arxiv.org/html/2606.07187v1>.

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

### Phase 4: Backend capabilities

Only after a filesystem raw reader works should `segstore` grow backend
capabilities. The likely split is:

- whole-file access: current `Directory`, enough for manifests, WALs, raw
  segment writes, sidecars, and small local tests;
- filesystem path access: enables mmap and page-cache-driven readers;
- range access: reads byte ranges without decoding the full segment;
- vectored range access: coalesces many small posting/page reads into fewer I/O
  operations;
- conditional publish: needed for object-store generation pointers and
  multi-writer-safe artifact publication, not for the current single-writer
  local `SegmentedStore`.

Do not fake these capabilities on backends where their cost model is different.
For example, an in-memory range read can test correctness but says nothing about
whether query-time random reads are acceptable on SSD or object storage.

## Per-Consumer Consequences

- `postings`: first raw-segment target. It needs term dictionaries, compressed
  doc-id blocks, frequencies, positions, skip/block-max metadata, and optional
  impact-score upper bounds in byte-native files. Massive lexical search should
  keep dictionaries and small metadata hot, then page posting and position
  blocks as queries demand them.
- `lexir`: should sit above `postings`, not grow a second Lucene-like storage
  engine. Its materialized-log CLI path is useful for operation replay and
  maintenance commands, but large lexical search should use postings segment
  readers with corpus statistics and scoring policy layered on top.
- `sporse`: current WAND sidecars are useful for medium in-memory segments, but
  massive learned-sparse search should converge toward postings-like disk blocks
  for sparse dimensions. Keep the WAND semantics and learned-sparse adapters; do
  not assume a full `SporseIndex` per segment can stay memory resident.
- `vicinity`: per-segment HNSW sidecars help restart, but massive vector search
  needs an ANN-specific disk layout: hot routing/entry structures, cold vector or
  graph pages, and a cache/prefetch policy. SPANN-style hot centroids plus cold
  posting lists is a plausible separate path from pure HNSW.
- `precinct`: inherits the vector problem and adds region geometry. Its raw
  segment metadata needs region family, dimension, metric/lift parameters, and
  which sub-indexes are hot versus paged.
- `sketchir` and `gramdex`: MinHash/LSH-style workloads can often keep signatures
  or bucket directories hot while storing bucket payloads cold. Segment metadata
  should expose hash family, banding, seed/config, and bucket offsets. Do not
  load every bucket payload to answer a narrow query.
- `tranz`, `subsume`, `flowmatch`, `hopfield`, and `symproj`: these are usually
  artifact/generation-store consumers, not `segstore` consumers. Their large
  outputs are model checkpoints, codebooks, datasets, metrics, and generated
  samples that need digest, provenance, and generation manifests more than
  tombstone compaction.

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
2. Add an explicit I/O capability layer on paper before code: whole-file,
   filesystem path/mmap, range, vectored range, conditional publish. Keep
   `MemoryDirectory` as a correctness backend, not a performance proxy.
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
- If a proposed optimization only improves `MemoryDirectory` benchmarks, do not
  treat it as evidence for filesystem or object-store behavior.
- If a proposed backend feature claims performance or crash semantics, its test
  or benchmark must name the backend class it proves.
- If a consumer's sidecars must all be loaded to search, cap the claim at
  restart-time persistence. It is not yet an out-of-core search path.

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
- Should backend capability traits live in `durability` because they are I/O
  primitives, or in `segstore` because pinning/GC visibility is segment-lifecycle
  state?
