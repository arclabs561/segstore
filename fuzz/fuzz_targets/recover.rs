#![no_main]
//! Corruption-recovery fuzz: write a random number of adds, then corrupt the WAL
//! at a fuzz-chosen offset (bit-flip, truncate, or zero a byte), and reopen. The
//! invariant: recovery NEVER panics and NEVER invents an id -- it returns either a
//! hard error or a store whose live ids are a subset of what was added (a valid
//! recovered prefix). This is the per-byte sector-garble model (SQLite/FDB/ALICE),
//! beyond the differential `ops` target which only exercises clean op sequences.
//!
//! Run: `cargo +nightly fuzz run recover`.

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
    corrupt_at: u16,
    corrupt_kind: u8,
}

const WAL: &str = "segstore.wal.0";

fuzz_target!(|sc: Scenario| {
    let dir = MemoryDirectory::arc();
    let n = (sc.n_adds as u32) % 60;
    // High flush threshold: keep every add in epoch 0's WAL (no checkpoint), so the
    // corruption lands in the single record log.
    {
        let mut s = SegmentedStore::open(dir.clone(), Kv, 1000).unwrap();
        for i in 0..n {
            s.add(i, format!("v{i}")).unwrap();
        }
    }

    // Corrupt the WAL bytes in place.
    let mut bytes = Vec::new();
    if let Ok(mut r) = dir.open_file(WAL) {
        let _ = r.read_to_end(&mut bytes);
    }
    if !bytes.is_empty() {
        let off = (sc.corrupt_at as usize) % bytes.len();
        match sc.corrupt_kind % 3 {
            0 => bytes[off] ^= 0xFF,
            1 => bytes.truncate(off),
            _ => bytes[off] = 0,
        }
        if let Ok(mut w) = dir.create_file(WAL) {
            let _ = w.write_all(&bytes);
            let _ = w.flush();
        }
    }

    // Reopen. A hard error is acceptable; a successful open must yield only ids that
    // were actually added (never a panic, never an invented/garbage id).
    if let Ok(s) = SegmentedStore::open(dir, Kv, 1000) {
        for seg in s.segments() {
            for (id, _) in seg.iter() {
                if s.is_live(id) {
                    assert!(*id < n, "recovery surfaced an id never added: {id} (n={n})");
                }
            }
        }
    }
});
