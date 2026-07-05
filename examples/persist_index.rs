//! Validation + canonical pattern for the persist-the-built-index hook (0.4.0).
//!
//! Every segstore consumer caches a built per-segment index (an HNSW graph,
//! posting lists, ...) and, without persistence, rebuilds it from the raw segment
//! on every restart. This shows the hook that fixes it: persist the built index
//! into the reserved `segstore.idx.<id>.<kind>` sidecar keyed by the stable
//! segment id (`segment_ids()`), then on reopen LOAD it instead of rebuilding.
//! segstore garbage-collects the sidecar in lockstep with its segment.
//!
//! The sidecar here is raw little-endian bytes with a consumer-owned
//! magic/version/segment-id/recipe/count header. segstore never reads it;
//! missing or invalid sidecars are rebuilt from that segment through
//! `SegmentCatalog::read_segment`.
//!
//! Run: `cargo run --example persist_index`

use std::io::Read;
use std::time::Instant;

use durability::FsDirectory;
use segstore::{SegmentCatalog, SegmentedStore, Store};

const SIDE_MAGIC: &[u8; 8] = b"SIDXDEMO";
const SIDE_VERSION: u32 = 1;
const SIDE_RECIPE: &[u8] = b"demo-sorted-u64-u32-v1";

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

/// Persist the built index as raw little-endian bytes.
///
/// The header is deliberately consumer-owned. The recipe identifies the
/// algorithm/config/input assumptions that make this sidecar valid. The segment
/// id rejects copied or misnamed sidecars before their payload is trusted.
/// segstore only reserves and GCs the filename namespace.
fn encode(seg_id: u64, index: &[(u64, u32)]) -> Vec<u8> {
    let count = u32::try_from(index.len()).expect("demo sidecar count fits in u32");
    let recipe_len = u32::try_from(SIDE_RECIPE.len()).expect("demo recipe fits in u32");
    let mut bytes = Vec::with_capacity(28 + SIDE_RECIPE.len() + index.len() * 12);
    bytes.extend_from_slice(SIDE_MAGIC);
    bytes.extend_from_slice(&SIDE_VERSION.to_le_bytes());
    bytes.extend_from_slice(&seg_id.to_le_bytes());
    bytes.extend_from_slice(&recipe_len.to_le_bytes());
    bytes.extend_from_slice(SIDE_RECIPE);
    bytes.extend_from_slice(&count.to_le_bytes());
    for &(v, id) in index {
        bytes.extend_from_slice(&v.to_le_bytes());
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    bytes
}

fn decode(expected_seg_id: u64, bytes: &[u8]) -> Result<Vec<(u64, u32)>, String> {
    if bytes.len() < 24 {
        return Err("sidecar header is truncated".into());
    }
    if &bytes[..8] != SIDE_MAGIC {
        return Err("sidecar magic mismatch".into());
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != SIDE_VERSION {
        return Err(format!("unsupported sidecar version {version}"));
    }
    let seg_id = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
    if seg_id != expected_seg_id {
        return Err(format!(
            "sidecar segment id mismatch: got {seg_id}, expected {expected_seg_id}"
        ));
    }
    let recipe_len = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
    let recipe_start = 24usize;
    let recipe_end = recipe_start
        .checked_add(recipe_len)
        .ok_or("sidecar recipe length overflow")?;
    if bytes.len() < recipe_end + 4 {
        return Err("sidecar recipe/count header is truncated".into());
    }
    if &bytes[recipe_start..recipe_end] != SIDE_RECIPE {
        return Err("sidecar recipe mismatch".into());
    }
    let count = u32::from_le_bytes(bytes[recipe_end..recipe_end + 4].try_into().unwrap()) as usize;
    let payload_bytes = count
        .checked_mul(12)
        .ok_or("sidecar entry count overflow")?;
    let expected = recipe_end
        .checked_add(4)
        .and_then(|n| n.checked_add(payload_bytes))
        .ok_or("sidecar length overflow")?;
    if bytes.len() != expected {
        return Err(format!(
            "sidecar length mismatch: got {}, expected {expected}",
            bytes.len()
        ));
    }
    Ok(bytes[recipe_end + 4..]
        .chunks_exact(12)
        .map(|c| {
            let v = u64::from_le_bytes(c[..8].try_into().unwrap());
            let id = u32::from_le_bytes(c[8..12].try_into().unwrap());
            (v, id)
        })
        .collect())
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
            let bytes = encode(seg_id, &build_index(&s.segments()[idx]));
            dir.atomic_write(&s.index_name(seg_id, "demo"), &bytes)
                .unwrap();
        }
        println!(
            "built + persisted {} per-segment index sidecars",
            s.segment_ids().len()
        );
    } // drop: simulate a process restart

    // Reopen the full store only for the comparison baseline: rebuilding every
    // index still has to decode every segment.
    let dir = FsDirectory::arc(&root).unwrap();
    let s = SegmentedStore::open(dir.clone(), Kv, 2_000).unwrap();

    let t = Instant::now();
    let rebuilt: Vec<Vec<(u64, u32)>> = s.segments().iter().map(|seg| build_index(seg)).collect();
    let rebuild = t.elapsed();
    drop(s);

    // The sidecar loader path does not open `SegmentedStore`, so it does not
    // decode all segment payloads up front. A missing/stale sidecar pays only for
    // the segment it must rebuild.
    let catalog = SegmentCatalog::<u32>::open(dir.clone()).unwrap();
    let t = Instant::now();
    let mut rebuilt_sidecars = 0usize;
    let loaded: Vec<Vec<(u64, u32)>> = catalog
        .segment_ids()
        .iter()
        .map(|&id| {
            let name = catalog.index_name(id, "demo");
            let mut bytes = Vec::new();
            let loaded = dir
                .open_file(&name)
                .and_then(|mut f| {
                    f.read_to_end(&mut bytes)?;
                    Ok(())
                })
                .ok()
                .and_then(|_| decode(id, &bytes).ok());
            loaded.unwrap_or_else(|| {
                rebuilt_sidecars += 1;
                let segment: Vec<(u32, u64)> = catalog.read_segment(id).unwrap();
                build_index(&segment)
            })
        })
        .collect();
    let load = t.elapsed();

    assert_eq!(
        rebuilt, loaded,
        "the loaded index must equal a fresh rebuild"
    );
    println!(
        "catalog reopen over {} segments: rebuild-all {rebuild:?} vs load-all {load:?}",
        rebuilt.len()
    );
    println!("  rebuilt invalid/missing sidecars: {rebuilt_sidecars}");
    println!("  [PASS] persisted indices load on reopen instead of rebuilding");
    println!("  (the win scales with build cost; a real HNSW build dwarfs this toy build)");

    let _ = std::fs::remove_dir_all(&root);
}
