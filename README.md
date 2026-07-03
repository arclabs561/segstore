# segstore

Generic durable segmented store: write-ahead log, immutable segments, tombstone
deletes, checkpoint, and compaction.

A consumer implements the `Store` trait to say how a batch of items becomes a
segment and how segments merge during compaction; segstore owns the durability
(built on [`durability`](https://crates.io/crates/durability)) and the LSM-style
lifecycle. The segment representation is opaque, so the same machinery backs an
inverted index (posting-list segments), a graph index (graph-delta segments), or
any other updatable structure.

## Example

```rust
use segstore::{SegmentedStore, Store};
use durability::MemoryDirectory;

// A segment is just a batch of (id, item) pairs.
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
        segs.iter().flat_map(|s| s.iter()).filter(|(id, _)| live(id)).cloned().collect()
    }
    fn segment_len(&self, seg: &Vec<(u32, String)>) -> usize { seg.len() }
}

let dir = MemoryDirectory::arc();
let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
s.add(1, "a".into()).unwrap();
s.add(2, "b".into()).unwrap();
s.delete(2).unwrap();
assert!(s.is_live(&1) && !s.is_live(&2));
```

## Durability

In-memory or on-disk via `durability`. Each `add`/`delete` is logged to a
write-ahead log before it takes effect. A checkpoint writes each new segment to
its own file and atomically publishes a manifest (CRC-checked, with an fsync
barrier on a filesystem backend) naming the current segments and tombstones, then
rotates the WAL (a new epoch-suffixed log is started and the old one deleted) so
the log never grows past one checkpoint interval. Because only new segments are
written, a checkpoint is O(new data), not O(total): the Lucene `segments_N` /
RocksDB MANIFEST model. Recovery loads the manifest's segment files and replays
only the current epoch's WAL. `SyncPolicy::Fsync` (via `open_with_options`) syncs
every WAL record to stable storage; the default flushes to the OS without a
per-op fsync.

## Memory model

The on-disk layout is segment-per-file, but the current `SegmentedStore::open`
API loads every manifest segment into `Arc<Segment>` memory and `View::segments`
returns those in-memory segments. That is the right shape for fast embedded
indexes whose active segment set fits RAM, and for caching expensive per-segment
sidecars across restarts.

For corpora larger than memory, segstore is not yet a complete out-of-core
reader. The next layer needs a reader/open path that can expose stable segment
ids and segment file handles (or mmap-backed consumer readers) without
deserializing each payload. `durability` already provides filesystem paths and an
optional mmap helper; the missing piece is a segstore API that keeps the
manifest/GC/checkpoint guarantees while letting consumers stream or map their own
segment formats.

`SegmentCatalog` can inspect the checkpoint manifest without opening segment
payload files, read one segment's validated serialized payload bytes, or decode
one requested segment for sidecar rebuilds. It is still a catalog helper for
loaders and diagnostics, not a byte-native query reader.

For byte-native query paths, use consumer sidecars. `segstore` reserves and
garbage-collects `segstore.idx.<segment-id>.<kind>` next to the source segment,
but the consumer owns the sidecar format and compatibility checks. A postings
crate can store a raw postings block there; an ANN crate can store graph pages
there. `segstore.seg.<id>` remains the durable source payload.

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
snapshot off the ingest hot path -- republishing per write would make bulk ingest
quadratic. `View::segment_ids()` plus `try_index_name(id, kind)` lets readers load
consumer sidecars without holding the writer.

## Examples

See [examples/README.md](examples/README.md) for runnable examples with
captured output.

## Status

0.x; the API and on-disk format may change between minor versions.

Not to be confused with `seglog`, an append-only *event log* for event sourcing:
segstore is a mutable index-backing store with deletes and compaction, the layer
*above* a write-ahead log rather than the log itself.

## License

Dual-licensed under MIT or Apache-2.0.
