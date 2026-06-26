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
        segs: &[Vec<(u32, String)>],
        live: &dyn Fn(&u32) -> bool,
    ) -> Vec<(u32, String)> {
        segs.iter().flatten().filter(|(id, _)| live(id)).cloned().collect()
    }
}

let dir = MemoryDirectory::arc();
let mut s = SegmentedStore::open(dir, Kv, 2).unwrap();
s.add(1, "a".into()).unwrap();
s.add(2, "b".into()).unwrap();
s.delete(2).unwrap();
assert!(s.is_live(&1) && !s.is_live(&2));
```

## Status

v0. In-memory and on-disk via `durability`, with crash recovery by replaying the
write-ahead log past the last checkpoint. Known v0 limitations: the WAL is not
yet rotated (it grows until a fresh checkpoint re-bases it), and writes flush but
do not `fsync` per operation.

Not to be confused with `seglog`, an append-only *event log* for event sourcing:
segstore is a mutable index-backing store with deletes and compaction, the layer
*above* a write-ahead log rather than the log itself.

## License

Dual-licensed under MIT or Apache-2.0.
