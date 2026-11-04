# segstore

Generic durable segmented store with write-ahead log and compaction.

## Overview

A consumer implements the `Store` trait to say how a batch of items becomes a
segment and how segments merge during compaction; segstore owns the durability
(built on [`durability`](https://crates.io/crates/durability)) and the LSM-style
lifecycle. The segment representation is opaque, so the same machinery backs an
inverted index (posting-list segments), a graph index (graph-delta segments), or
any other updatable structure.

## Install

```toml
[dependencies]
segstore = "0.8"
durability = "0.9"
```

## Example

```rust
use segstore::{DefaultStore, SegmentedStore};
use durability::MemoryDirectory;

let dir = MemoryDirectory::arc();
let mut s = SegmentedStore::open(dir, DefaultStore::<u32, String>::new(), 2).unwrap();
s.add(1, "a".into()).unwrap();
s.add(2, "b".into()).unwrap();
s.delete(2).unwrap();
assert!(s.is_live(&1) && !s.is_live(&2));
```

Use `DefaultStore<Id, Item>` when a segment is just `Vec<(Id, Item)>`. Implement
`Store` directly when a consumer needs sorted segments or a custom merge.

A re-add replaces the id's previous value. A copy already sealed in a segment is
marked superseded rather than rewritten, so filter each segment's items with
`is_live_in(segment_id, &id)` (on the writer, a `View`, or a `SegmentCatalog`);
`is_live(&id)` only says the id is not deleted. Custom stores get this by
implementing `Store::item_ids`.

## Durability

In-memory or on-disk via `durability`. Each `add`/`delete` is logged to a
write-ahead log before it takes effect. A checkpoint writes each new segment to
its own file, then atomically publishes a CRC-checked manifest naming the current
segments and tombstones. The filesystem backend also applies an fsync barrier.
The checkpoint starts a new WAL epoch and deletes the old log, so the WAL covers
at most one checkpoint interval. Only new segment payloads are written, but each
checkpoint also serializes the current segment IDs, tombstones, and
superseded-copy metadata.

Recovery loads the manifest's segment files and replays only the current epoch's
WAL. `SyncPolicy::Fsync` (via `open_with_options`) syncs every WAL record to
stable storage; the default flushes to the OS without a per-op fsync.

## Memory model

The on-disk layout is segment-per-file, but the current `SegmentedStore::open`
API loads every manifest segment into `Arc<Segment>` memory and `View::segments`
returns those in-memory segments. That is the right shape for fast embedded
indexes whose active segment set fits RAM, and for caching expensive per-segment
sidecars across restarts.

There is no out-of-core reader yet for corpora larger than memory; see
[docs/out-of-core-segment-reader.md](docs/out-of-core-segment-reader.md) for the
proposed design.

`SegmentCatalog` reads the checkpoint manifest without opening segment files.
From there it can report one segment's file offset, length, and CRC, read its
validated payload bytes, or decode it for a sidecar rebuild. It is a catalog
helper for loaders and diagnostics, not a byte-native query reader.
The catalog boundary is precise: segment bytes are the source payload as written,
and liveness is exposed separately through `is_live_in`. Consumers that decode or
map segment bytes must apply liveness and their own query metadata; segstore does
not convert source payloads into a live query view. Consumers that mmap or
range-read bytes from `SegmentPayloadInfo` should call
`SegmentPayloadInfo::verify_payload` before trusting those bytes.

For byte-native query paths, use consumer sidecars. `segstore` reserves and
garbage-collects `segstore.idx.<segment-id>.<kind>` next to the source segment,
but the consumer owns the sidecar format and compatibility checks. A postings
crate can store a raw postings block there; an ANN crate can store graph pages
there. `SidecarEnvelope` is available for the repeated magic/version/segment-id/
recipe/CRC byte envelope, but recipe meaning and payload decoding stay in the
consumer. `segstore.seg.<id>` remains the durable source payload.

## Compaction

`compact()` merges all segments into one and purges tombstones.
`compact_tiers()` runs size-tiered compaction (Cassandra/Lucene-style): segments
are grouped into size buckets and a full bucket is merged, smallest first, never
exceeding `max_merged_len` items, with larger segments frozen out so the biggest
one is never rewritten by tiering. Scheduling is the consumer's: call
`compact_tiers()` when convenient (e.g. a background thread), or set
`Options::auto_compact` to run it inline after a flush. `segment_sizes()` and the
`CompactionStats` returned by both expose the segment-count and merge-cost signal
to watch as the corpus grows.

For bulk ingest (an index build phase), `extend(items)` syncs the WAL once per
batch instead of per item.

## Concurrent reads

Single writer, many readers. `reader()` returns a cloneable, thread-safe `Reader`;
`reader.view()` takes a consistent point-in-time `View` of the segments and
tombstones that a query holds (lock-free) for its whole duration, even while the
writer adds, deletes, or compacts on another thread. Visibility is commit-style:
a view reflects the state as of the last `checkpoint()` (which compaction also
performs), so writes since then become visible after the next checkpoint. This is
the Lucene `SearcherManager` / Tantivy `Searcher` model, made light by segstore's
in-memory segments (a `View` is `Arc` clones; an old segment's memory frees when
the last view holding it drops). Publishing only at the checkpoint keeps the
snapshot off the ingest hot path; republishing per write would make bulk ingest
quadratic. `View::segment_ids()` plus `try_index_name(id, kind)` lets readers load
consumer sidecars without holding the writer.

## Examples

```bash
cargo run --release --example inverted_index
```

```text
indexed 4 docs; df(rust)=3
after delete(1): df(rust)=2
  [PASS] segstore backs an updatable inverted index: build, delete, compact, recover
```

[examples/README.md](examples/README.md) covers the other examples (vector
search, sidecars) with captured output.

## Status

0.x; the API and on-disk format may change between minor versions.

## License

Dual-licensed under MIT or Apache-2.0.
