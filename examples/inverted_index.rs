//! Validation: back a real inverted index with segstore.
//!
//! This is the rule-of-three test for the [`segstore::Store`] abstraction. A toy
//! key-value store proves the lifecycle; this proves the abstraction fits the
//! shape of an actual updatable inverted index (postings' domain): a segment is
//! a term -> posting-list map, deletes are tombstones, and compaction merges
//! posting lists while dropping tombstoned docs.
//!
//! The one design difference from an incremental index like `postings`: document
//! frequency is computed from the live segments at query time
//! (`query(term).len()`) rather than tracked incrementally and adjusted on
//! delete. segstore's opaque-segment model does not know about terms, so global
//! per-term stats are the consumer's concern, computed from `segments()` +
//! `is_live()`. That fits cleanly; it is a maintenance-strategy choice, not a
//! wall.
//!
//! Run: `cargo run --example inverted_index`

use std::collections::BTreeMap;

use durability::MemoryDirectory;
use segstore::{SegmentedStore, Store};

/// An inverted index: a document is a set of terms; a segment is the term ->
/// doc-id postings for one batch of documents.
struct InvertedIndex;

impl Store for InvertedIndex {
    type Id = u32;
    type Item = Vec<String>;
    type Segment = BTreeMap<String, Vec<u32>>;

    fn build_segment(&self, batch: &[(u32, Vec<String>)]) -> BTreeMap<String, Vec<u32>> {
        let mut seg: BTreeMap<String, Vec<u32>> = BTreeMap::new();
        for (id, terms) in batch {
            for t in terms {
                seg.entry(t.clone()).or_default().push(*id);
            }
        }
        for ids in seg.values_mut() {
            ids.sort_unstable();
            ids.dedup();
        }
        seg
    }

    fn merge_segments(
        &self,
        segs: &[BTreeMap<String, Vec<u32>>],
        live: &dyn Fn(&u32) -> bool,
    ) -> BTreeMap<String, Vec<u32>> {
        let mut out: BTreeMap<String, Vec<u32>> = BTreeMap::new();
        for seg in segs {
            for (term, ids) in seg {
                let postings = out.entry(term.clone()).or_default();
                postings.extend(ids.iter().copied().filter(|id| live(id)));
            }
        }
        for ids in out.values_mut() {
            ids.sort_unstable();
            ids.dedup();
        }
        // Drop terms whose postings were entirely tombstoned away.
        out.retain(|_, ids| !ids.is_empty());
        out
    }

    fn segment_len(&self, seg: &BTreeMap<String, Vec<u32>>) -> usize {
        // The size metric is distinct documents (the unit of `Item`), not terms.
        let mut docs = std::collections::HashSet::new();
        for ids in seg.values() {
            docs.extend(ids.iter().copied());
        }
        docs.len()
    }
}

/// Retrieve the live document ids for `term` across segments + the live buffer.
fn query(index: &SegmentedStore<InvertedIndex>, term: &str) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    for seg in index.segments() {
        if let Some(ids) = seg.get(term) {
            out.extend(ids.iter().copied().filter(|id| index.is_live(id)));
        }
    }
    for (id, terms) in index.buffer() {
        if index.is_live(id) && terms.iter().any(|t| t == term) {
            out.push(*id);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn doc(terms: &[&str]) -> Vec<String> {
    terms.iter().map(|t| t.to_string()).collect()
}

fn main() {
    let dir = MemoryDirectory::arc();
    let mut index = SegmentedStore::open(dir.clone(), InvertedIndex, 2).unwrap();

    // Index four documents. With flush_threshold = 2, this produces two flushed
    // segments plus nothing buffered.
    index.add(0, doc(&["rust", "search", "index"])).unwrap();
    index.add(1, doc(&["rust", "graph"])).unwrap(); // flush -> segment 1
    index.add(2, doc(&["search", "ranking"])).unwrap();
    index.add(3, doc(&["rust", "ranking"])).unwrap(); // flush -> segment 2

    assert_eq!(query(&index, "rust"), vec![0, 1, 3]);
    assert_eq!(query(&index, "ranking"), vec![2, 3]);
    println!("indexed 4 docs; df(rust)={}", query(&index, "rust").len());

    // Delete doc 1: it must vanish from every term it appeared under.
    index.delete(1).unwrap();
    assert_eq!(
        query(&index, "rust"),
        vec![0, 3],
        "deleted doc 1 must leave rust"
    );
    assert_eq!(
        query(&index, "graph"),
        Vec::<u32>::new(),
        "graph had only doc 1"
    );
    println!("after delete(1): df(rust)={}", query(&index, "rust").len());

    // Compact: the tombstone is physically purged; queries are unchanged.
    index.compact().unwrap();
    assert_eq!(index.segment_count(), 1, "compaction merges to one segment");
    assert_eq!(
        index.tombstone_count(),
        0,
        "compaction purges the tombstone"
    );
    assert_eq!(
        query(&index, "rust"),
        vec![0, 3],
        "compaction preserves results"
    );
    assert!(
        query(&index, "graph").is_empty(),
        "doc 1 is physically gone"
    );

    // Crash recovery: reopen from the on-disk checkpoint + WAL and re-query.
    drop(index);
    let recovered = SegmentedStore::open(dir, InvertedIndex, 2).unwrap();
    assert_eq!(
        query(&recovered, "rust"),
        vec![0, 3],
        "recovery preserves results"
    );
    assert_eq!(query(&recovered, "ranking"), vec![2, 3]);

    println!(
        "  [PASS] segstore backs an updatable inverted index: build, delete, compact, recover"
    );
}
