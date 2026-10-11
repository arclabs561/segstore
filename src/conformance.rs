//! Shared behaviour tests for indexes built on [`SegmentedStore`](crate::SegmentedStore).
//!
//! Several crates wrap a `SegmentedStore` in their own updatable index (a
//! k-gram index, a MinHash index, a sparse-vector index). The wrappers repeat
//! the same lifecycle (add, re-add, delete, seal, compact, checkpoint, reopen)
//! and have drifted apart before: one kept serving a re-added id's old
//! document, one missed a writer-lock fix. This module states that lifecycle
//! once. A store crate implements [`ConformanceStore`] for its index in a test
//! and calls [`run_all`]; a wrapper that diverges then fails its own suite.
//!
//! Enabled by the `conformance` feature, meant for `[dev-dependencies]`.
//!
//! The central check is model-based: a seeded random sequence of operations
//! runs against the store and against a plain map, and after every step each
//! live id must be found by its current document while no replaced or deleted
//! document may still return its id. Reopening drops the store and opens the
//! same directory again, without a checkpoint, so acknowledged writes must
//! survive through the log alone.

use std::collections::BTreeMap;
use std::sync::Arc;

use durability::{Directory, FsDirectory, MemoryDirectory, PersistenceResult};

/// An updatable index under test. Implemented by the store crate, in a test.
pub trait ConformanceStore: Sized {
    /// The document type the index stores per id.
    type Doc: Clone + std::fmt::Debug;

    /// Open (or recover) the index in `dir`. Use a small flush threshold so
    /// the suite seals several segments.
    fn open(dir: Arc<dyn Directory>) -> PersistenceResult<Self>;
    /// Add, or replace, the document for `id`.
    fn add(&mut self, id: u32, doc: Self::Doc) -> PersistenceResult<()>;
    /// Delete `id`.
    fn delete(&mut self, id: u32) -> PersistenceResult<()>;
    /// Merge all segments.
    fn compact(&mut self) -> PersistenceResult<()>;
    /// Merge by size tier.
    fn compact_tiers(&mut self) -> PersistenceResult<()>;
    /// Rewrite segments whose live ratio is below `min_live_ratio`.
    fn reclaim(&mut self, min_live_ratio: f64) -> PersistenceResult<()>;
    /// Persist a checkpoint.
    fn checkpoint(&mut self) -> PersistenceResult<()>;
    /// A document for `seed`. Documents for different seeds must not match
    /// each other under [`ConformanceStore::matching`].
    fn doc(seed: u64) -> Self::Doc;
    /// Ids whose current document matches `doc`. The suite only needs the
    /// document's own id back, and never an id whose document was replaced.
    fn matching(&self, doc: &Self::Doc) -> Vec<u32>;
}

/// Run every check. Panics with the failing step on a violation.
pub fn run_all<S: ConformanceStore>() {
    re_add_replaces_old_document::<S>();
    delete_hides_document::<S>();
    reopen_without_checkpoint_keeps_writes::<S>();
    for seed in 1..=8 {
        random_operations_match_model::<S>(seed, 160);
    }
    second_writer_is_rejected::<S>();
}

/// Re-adding an id replaces its document in every later read, across
/// sealing, compaction and reopen.
pub fn re_add_replaces_old_document<S: ConformanceStore>() {
    let dir = MemoryDirectory::arc();
    let mut store = S::open(dir.clone()).expect("open");
    store.add(7, S::doc(1)).expect("add");
    store.checkpoint().expect("checkpoint");
    store.add(7, S::doc(2)).expect("re-add");
    assert_current(&store, 7, 2, "after re-add");
    assert_retired(&store, 7, 1, "after re-add");
    store.compact().expect("compact");
    assert_retired(&store, 7, 1, "after compact");
    drop(store);
    let store = S::open(dir).expect("reopen");
    assert_current(&store, 7, 2, "after reopen");
    assert_retired(&store, 7, 1, "after reopen");
}

/// A deleted id is never returned again, including after reopen.
pub fn delete_hides_document<S: ConformanceStore>() {
    let dir = MemoryDirectory::arc();
    let mut store = S::open(dir.clone()).expect("open");
    store.add(3, S::doc(10)).expect("add");
    store.add(4, S::doc(11)).expect("add");
    store.checkpoint().expect("checkpoint");
    store.delete(3).expect("delete");
    assert_retired(&store, 3, 10, "after delete");
    assert_current(&store, 4, 11, "neighbour after delete");
    drop(store);
    let store = S::open(dir).expect("reopen");
    assert_retired(&store, 3, 10, "after reopen");
}

/// Writes acknowledged before a drop survive reopen with no checkpoint.
pub fn reopen_without_checkpoint_keeps_writes<S: ConformanceStore>() {
    let dir = MemoryDirectory::arc();
    let mut store = S::open(dir.clone()).expect("open");
    for id in 0..5 {
        store.add(id, S::doc(100 + u64::from(id))).expect("add");
    }
    drop(store);
    let store = S::open(dir).expect("reopen");
    for id in 0..5 {
        assert_current(&store, id, 100 + u64::from(id), "after reopen");
    }
}

/// A random add/re-add/delete/seal/compact/reopen sequence agrees with a map.
pub fn random_operations_match_model<S: ConformanceStore>(seed: u64, steps: usize) {
    let dir = MemoryDirectory::arc();
    let mut store = Some(S::open(dir.clone()).expect("open"));
    let mut rng = XorShift(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut live: BTreeMap<u32, u64> = BTreeMap::new();
    let mut retired: Vec<(u32, u64)> = Vec::new();
    let mut next_doc = seed * 1_000_000;
    let mut log: Vec<String> = Vec::new();

    for step in 0..steps {
        let s = store.as_mut().expect("store is open");
        let id = (rng.next() % 12) as u32;
        let op = match rng.next() % 16 {
            0..=6 => {
                next_doc += 1;
                s.add(id, S::doc(next_doc)).expect("add");
                if let Some(old) = live.insert(id, next_doc) {
                    retired.push((id, old));
                }
                format!("add({id}, doc {next_doc})")
            }
            7..=9 => {
                s.delete(id).expect("delete");
                if let Some(old) = live.remove(&id) {
                    retired.push((id, old));
                }
                format!("delete({id})")
            }
            10 => {
                s.checkpoint().expect("checkpoint");
                "checkpoint".into()
            }
            11 => {
                s.compact().expect("compact");
                "compact".into()
            }
            12 => {
                s.compact_tiers().expect("compact_tiers");
                "compact_tiers".into()
            }
            13 => {
                s.reclaim(0.5).expect("reclaim");
                "reclaim(0.5)".into()
            }
            _ => {
                drop(store.take());
                store = Some(S::open(dir.clone()).expect("reopen"));
                "reopen".into()
            }
        };
        log.push(op);
        let s = store.as_ref().expect("store is open");
        let context = || format!("seed {seed}, step {step}; ops: {}", log.join(", "));
        for (&id, &doc) in &live {
            assert!(
                s.matching(&S::doc(doc)).contains(&id),
                "live id {id} not found by its document {doc} ({})",
                context()
            );
        }
        for &(id, doc) in &retired {
            assert!(
                !s.matching(&S::doc(doc)).contains(&id),
                "id {id} still returned for its replaced or deleted document {doc} ({})",
                context()
            );
        }
    }
}

/// Two writers on one directory: the second open fails while the first holds
/// the store, and succeeds after the first is dropped.
pub fn second_writer_is_rejected<S: ConformanceStore>() {
    let root = std::env::temp_dir().join(format!(
        "segstore-conformance-{}-{}",
        std::process::id(),
        std::any::type_name::<S>().len()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("create temp dir");
    let dir = FsDirectory::arc(&root).expect("fs directory");
    let first = S::open(dir.clone()).expect("first open");
    assert!(
        S::open(dir.clone()).is_err(),
        "a second writer opened a store that is already open"
    );
    drop(first);
    S::open(dir).expect("open after the first writer is dropped");
    let _ = std::fs::remove_dir_all(&root);
}

fn assert_current<S: ConformanceStore>(store: &S, id: u32, doc: u64, when: &str) {
    assert!(
        store.matching(&S::doc(doc)).contains(&id),
        "id {id} not found by its current document {doc} {when}"
    );
}

fn assert_retired<S: ConformanceStore>(store: &S, id: u32, doc: u64, when: &str) {
    assert!(
        !store.matching(&S::doc(doc)).contains(&id),
        "id {id} still returned for its old document {doc} {when}"
    );
}

/// Small deterministic generator, so failures replay from the printed seed.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}
