//! Validation: back a vector / nearest-neighbor index with segstore.
//!
//! Third rule-of-three consumer, after the toy KV and the inverted index, and
//! the one that surfaces segstore's defining architectural property. A segment
//! is a batch of `(id, vector)`; k-NN searches every live segment and merges the
//! top-k.
//!
//! The finding: segstore is **Lucene-style** — many immutable segments, search
//! all and merge results. That fits an inverted index natively (postings already
//! is exactly this) and fits a vector index via search-all-merge. It is NOT the
//! single-evolving-graph + in-place consolidation model of FreshDiskANN. So a
//! segstore-backed ANN is a *multi-segment* index (a per-segment graph, merged
//! across segments at query time), a legitimate design with different
//! recall/latency characteristics than a single consolidated graph. Choosing
//! segstore for an ANN index is therefore choosing that model on purpose.
//!
//! (This example brute-force-scans each segment for clarity; a real consumer
//! would build an HNSW inside each segment and merge the per-segment top-k.)
//!
//! Run: `cargo run --example vector_search`

use std::cmp::Ordering;

use durability::MemoryDirectory;
use segstore::{SegmentedStore, Store};

/// A flat vector index: a segment is a batch of `(id, vector)`.
struct VectorIndex;

impl Store for VectorIndex {
    type Id = u32;
    type Item = Vec<f32>;
    type Segment = Vec<(u32, Vec<f32>)>;

    fn build_segment(&self, batch: &[(u32, Vec<f32>)]) -> Vec<(u32, Vec<f32>)> {
        batch.to_vec()
    }

    fn merge_segments(
        &self,
        segs: &[Vec<(u32, Vec<f32>)>],
        live: &dyn Fn(&u32) -> bool,
    ) -> Vec<(u32, Vec<f32>)> {
        segs.iter()
            .flatten()
            .filter(|(id, _)| live(id))
            .cloned()
            .collect()
    }
}

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// k-NN by the search-all-segments-then-merge model: scan every live segment +
/// the live buffer, then keep the global top-k.
fn knn(index: &SegmentedStore<VectorIndex>, query: &[f32], k: usize) -> Vec<u32> {
    let mut cand: Vec<(u32, f32)> = Vec::new();
    for seg in index.segments() {
        for (id, v) in seg {
            if index.is_live(id) {
                cand.push((*id, dist2(query, v)));
            }
        }
    }
    for (id, v) in index.buffer() {
        if index.is_live(id) {
            cand.push((*id, dist2(query, v)));
        }
    }
    cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    cand.into_iter().take(k).map(|(id, _)| id).collect()
}

fn main() {
    let dir = MemoryDirectory::arc();
    let mut index = SegmentedStore::open(dir.clone(), VectorIndex, 2).unwrap();

    // Two clusters: near the origin and near (10, 10). flush_threshold = 2 makes
    // two segments, so the query genuinely searches across segments.
    index.add(0, vec![0.0, 0.0]).unwrap();
    index.add(1, vec![10.0, 10.0]).unwrap(); // flush -> segment 1
    index.add(2, vec![0.1, 0.1]).unwrap();
    index.add(3, vec![9.9, 9.9]).unwrap(); // flush -> segment 2

    assert_eq!(knn(&index, &[0.0, 0.0], 2), vec![0, 2], "origin cluster");
    assert_eq!(knn(&index, &[10.0, 10.0], 2), vec![1, 3], "far cluster");
    println!("indexed 4 vectors across 2 segments");

    // Delete the exact origin point; the next-nearest must take over.
    index.delete(0).unwrap();
    assert_eq!(
        knn(&index, &[0.0, 0.0], 1),
        vec![2],
        "deleted 0, nearest is now 2"
    );

    // Compaction merges segments and physically drops the tombstone.
    index.compact().unwrap();
    assert_eq!(index.segment_count(), 1);
    assert_eq!(index.tombstone_count(), 0);
    assert_eq!(
        knn(&index, &[0.0, 0.0], 1),
        vec![2],
        "compaction preserves search"
    );

    // Crash recovery.
    drop(index);
    let recovered = SegmentedStore::open(dir, VectorIndex, 2).unwrap();
    assert_eq!(
        knn(&recovered, &[0.0, 0.0], 1),
        vec![2],
        "recovery preserves search"
    );
    assert_eq!(knn(&recovered, &[10.0, 10.0], 1), vec![1]);

    println!(
        "  [PASS] segstore backs a multi-segment vector index (search-all-merge): delete, compact, recover"
    );
}
