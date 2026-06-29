//! Validation + canonical pattern for the persist-the-built-index hook (0.4.0).
//!
//! Every segstore consumer caches a built per-segment index (an HNSW graph,
//! posting lists, ...) and, without persistence, rebuilds it from the raw segment
//! on every restart. This shows the hook that fixes it: persist the built index
//! into the reserved `segstore.idx.<id>.<kind>` sidecar keyed by the stable
//! segment id (`segment_ids()`), then on reopen LOAD it instead of rebuilding.
//! segstore garbage-collects the sidecar in lockstep with its segment.
//!
//! The sidecar here is raw little-endian bytes (a POD layout, no serde), the
//! zero-copy-friendly format the consumer owns; segstore never reads it.
//!
//! Run: `cargo run --example persist_index`

use std::io::Read;
use std::time::Instant;

use durability::FsDirectory;
use segstore::{SegmentedStore, Store};

/// A toy payload: a batch of `(id, value)`. The "built index" is the values sorted
/// with their ids -- cheap here, but it stands in for an HNSW graph or posting
/// lists whose *build* is the dominant restart cost we want to pay once and persist.
struct Kv;
impl Store for Kv {
    type Id = u32;
    type Item = u64;
    type Segment = Vec<(u32, u64)>;
    fn build_segment(&self, batch: &[(u32, u64)]) -> Vec<(u32, u64)> {
        batch.to_vec()
    }
    fn merge_segments(
        &self,
        segs: &[&Vec<(u32, u64)>],
        live: &dyn Fn(&u32) -> bool,
    ) -> Vec<(u32, u64)> {
        segs.iter()
            .flat_map(|s| s.iter())
            .filter(|(id, _)| live(id))
            .cloned()
            .collect()
    }
    fn segment_len(&self, seg: &Vec<(u32, u64)>) -> usize {
        seg.len()
    }
}

/// "Build" the per-segment index. Like a real graph/posting index, *construction*
/// is the dominant restart cost: each entry is inserted into a sorted vec by an
/// O(n) shift (so the build is O(n^2)), a stand-in for the per-node neighbour
/// search an HNSW build pays. This is exactly the cost the persisted sidecar lets a
/// restart skip (loading is O(n)).
fn build_index(seg: &[(u32, u64)]) -> Vec<(u64, u32)> {
    let mut idx: Vec<(u64, u32)> = Vec::with_capacity(seg.len());
    for &(id, val) in seg {
        let pos = idx.partition_point(|&(v, _)| v < val);
        idx.insert(pos, (val, id));
    }
    idx
}

/// Persist the built index as raw little-endian bytes (12 per entry: u64 + u32).
/// This is the consumer-owned, POD/zero-copy-friendly sidecar format.
fn encode(index: &[(u64, u32)]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(index.len() * 12);
    for &(v, id) in index {
        bytes.extend_from_slice(&v.to_le_bytes());
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    bytes
}

fn decode(bytes: &[u8]) -> Vec<(u64, u32)> {
    bytes
        .chunks_exact(12)
        .map(|c| {
            let v = u64::from_le_bytes(c[..8].try_into().unwrap());
            let id = u32::from_le_bytes(c[8..12].try_into().unwrap());
            (v, id)
        })
        .collect()
}

fn main() {
    let mut root = std::env::temp_dir();
    root.push(format!("segstore-persist-index-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);

    // Build a corpus, then persist a per-segment index sidecar for each segment.
    {
        let dir = FsDirectory::arc(&root).unwrap();
        let mut s = SegmentedStore::open(dir.clone(), Kv, 2_000).unwrap();
        for i in 0..20_000u32 {
            s.add(i, (i as u64).wrapping_mul(2_654_435_761)).unwrap();
        }
        s.checkpoint().unwrap();
        for (idx, &seg_id) in s.segment_ids().iter().enumerate() {
            let bytes = encode(&build_index(&s.segments()[idx]));
            dir.atomic_write(&s.index_name(seg_id, "demo"), &bytes)
                .unwrap();
        }
        println!(
            "built + persisted {} per-segment index sidecars",
            s.segment_ids().len()
        );
    } // drop: simulate a process restart

    // Reopen. Time loading the sidecars vs rebuilding from the raw segments.
    let dir = FsDirectory::arc(&root).unwrap();
    let s = SegmentedStore::open(dir.clone(), Kv, 2_000).unwrap();

    let t = Instant::now();
    let rebuilt: Vec<Vec<(u64, u32)>> = s.segments().iter().map(|seg| build_index(seg)).collect();
    let rebuild = t.elapsed();

    let t = Instant::now();
    let loaded: Vec<Vec<(u64, u32)>> = s
        .segment_ids()
        .iter()
        .map(|&id| {
            let mut bytes = Vec::new();
            dir.open_file(&s.index_name(id, "demo"))
                .unwrap()
                .read_to_end(&mut bytes)
                .unwrap();
            decode(&bytes)
        })
        .collect();
    let load = t.elapsed();

    assert_eq!(
        rebuilt, loaded,
        "the loaded index must equal a fresh rebuild"
    );
    println!(
        "reopen over {} segments: rebuild-all {rebuild:?} vs load-all {load:?}",
        rebuilt.len()
    );
    println!("  [PASS] persisted indices load on reopen instead of rebuilding");
    println!("  (the win scales with build cost; a real HNSW build dwarfs this toy build)");

    let _ = std::fs::remove_dir_all(&root);
}
