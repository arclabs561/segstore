#![no_main]
//! Manifest-format crash fuzz: build a store, checkpoint it (producing a manifest
//! + per-segment files), then corrupt a fuzz-chosen on-disk file at a fuzz-chosen
//! offset, and reopen. The invariant matches `recover`: recovery NEVER panics and
//! NEVER invents an id -- it returns either a hard error or a store whose live ids
//! are a subset of what was added. This target complements `recover` (which only
//! corrupts the WAL) by reaching the 0.3 checkpoint surface: the `segstore.manifest`
//! and the per-segment `segstore.seg.<id>` files.
//!
//! Run: `cargo +nightly fuzz run recover_manifest`.

use std::io::{Read, Write};

use arbitrary::Arbitrary;
use durability::MemoryDirectory;
use libfuzzer_sys::fuzz_target;
use segstore::{SegmentedStore, Store};

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
        segs.iter()
            .flat_map(|s| s.iter())
            .filter(|(id, _)| live(id))
            .cloned()
            .collect()
    }
    fn segment_len(&self, seg: &Vec<(u32, String)>) -> usize {
        seg.len()
    }
}

#[derive(Arbitrary, Debug)]
struct Scenario {
    n_adds: u8,
    /// Which on-disk file to corrupt (indexes into the directory listing).
    target: u8,
    corrupt_at: u16,
    corrupt_kind: u8,
}

fuzz_target!(|sc: Scenario| {
    let dir = MemoryDirectory::arc();
    let n = (sc.n_adds as u32) % 40;
    // flush_threshold 2 seals several segments, then a checkpoint writes the
    // manifest + one seg file per segment.
    {
        let mut s = SegmentedStore::open(dir.clone(), Kv, 2).unwrap();
        for i in 0..n {
            s.add(i, format!("v{i}")).unwrap();
        }
        let _ = s.checkpoint();
    }

    // Pick one on-disk file (manifest, seg.*, or wal.*) and corrupt it in place.
    let files: Vec<String> = match dir.list_dir("") {
        Ok(f) if !f.is_empty() => f,
        _ => return,
    };
    let path = files[(sc.target as usize) % files.len()].clone();
    let mut bytes = Vec::new();
    if let Ok(mut r) = dir.open_file(&path) {
        let _ = r.read_to_end(&mut bytes);
    }
    if bytes.is_empty() {
        return;
    }
    let off = (sc.corrupt_at as usize) % bytes.len();
    match sc.corrupt_kind % 3 {
        0 => bytes[off] ^= 0xFF,
        1 => bytes.truncate(off),
        _ => bytes[off] = 0,
    }
    if let Ok(mut w) = dir.create_file(&path) {
        let _ = w.write_all(&bytes);
        let _ = w.flush();
    }

    // Reopen. A hard error is acceptable; a successful open must yield only ids
    // that were actually added (never a panic, never an invented id).
    if let Ok(s) = SegmentedStore::open(dir, Kv, 2) {
        for seg in s.segments() {
            for (id, _) in seg.iter() {
                if s.is_live(id) {
                    assert!(*id < n, "recovery surfaced an id never added: {id} (n={n})");
                }
            }
        }
    }
});
