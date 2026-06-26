#![no_main]
//! Coverage-guided differential fuzz: a random sequence of store operations is
//! applied to a `SegmentedStore` and to a reference model in lockstep, asserting
//! the live set always matches. Mirrors the in-crate proptest but lets libFuzzer
//! drive the op sequence by coverage.
//!
//! Run: `cargo +nightly fuzz run ops`.

use std::collections::BTreeMap;

use arbitrary::Arbitrary;
use durability::MemoryDirectory;
use libfuzzer_sys::fuzz_target;
use segstore::{Options, SegmentedStore, Store, TierConfig};

/// A trivial Vec-backed store, exercising the public API (segment_len + live_len).
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
        segs.iter()
            .flatten()
            .filter(|(id, _)| live(id))
            .cloned()
            .collect()
    }

    fn segment_len(&self, seg: &Vec<(u32, String)>) -> usize {
        seg.len()
    }

    fn live_len(&self, seg: &Vec<(u32, String)>, live: &dyn Fn(&u32) -> bool) -> Option<usize> {
        Some(seg.iter().filter(|(id, _)| live(id)).count())
    }
}

#[derive(Arbitrary, Debug)]
enum Op {
    Add,
    Delete(u8),
    Compact,
    CompactTiers,
    ForceMerge(u8),
    Reclaim(u8),
    Reopen,
}

fn live_set(s: &SegmentedStore<Kv>) -> Vec<(u32, String)> {
    let mut out: Vec<(u32, String)> = Vec::new();
    for seg in s.segments() {
        for (id, it) in seg {
            if s.is_live(id) {
                out.push((*id, it.clone()));
            }
        }
    }
    for (id, it) in s.buffer() {
        if s.is_live(id) {
            out.push((*id, it.clone()));
        }
    }
    out.sort();
    out
}

fuzz_target!(|ops: Vec<Op>| {
    let dir = MemoryDirectory::arc();
    let cfg = TierConfig {
        min_merge: 4,
        max_merge: 8,
        max_merged_len: 64,
        ..Default::default()
    };
    let mk = || Options {
        tiering: cfg,
        ..Options::new(3)
    };
    let mut s = SegmentedStore::open_with_options(dir.clone(), Kv, mk()).unwrap();
    // Reference model: last-write-wins live id -> item. Add uses unique ids
    // (segstore makes no dedup promise), so the model stays exact.
    let mut model: BTreeMap<u32, String> = BTreeMap::new();
    let mut live_ids: Vec<u32> = Vec::new();
    let mut next_id = 0u32;

    for op in ops {
        match op {
            Op::Add => {
                let id = next_id;
                next_id += 1;
                let v = format!("v{id}");
                s.add(id, v.clone()).unwrap();
                model.insert(id, v);
                live_ids.push(id);
            }
            Op::Delete(k) => {
                if !live_ids.is_empty() {
                    let id = live_ids.swap_remove(k as usize % live_ids.len());
                    s.delete(id).unwrap();
                    model.remove(&id);
                }
            }
            Op::Compact => {
                s.compact().unwrap();
            }
            Op::CompactTiers => {
                s.compact_tiers().unwrap();
            }
            Op::ForceMerge(n) => {
                s.force_merge_to(n as usize % 8).unwrap();
            }
            Op::Reclaim(pct) => {
                s.reclaim_tombstones(pct as f64 / 255.0).unwrap();
            }
            Op::Reopen => {
                s = SegmentedStore::open_with_options(dir.clone(), Kv, mk()).unwrap();
            }
        }
    }

    let want: Vec<(u32, String)> = model.into_iter().collect();
    assert_eq!(live_set(&s), want, "live set diverged from the reference model");
});
