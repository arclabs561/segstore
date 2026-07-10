//! Persist a `postings::raw` sidecar for each segstore segment.
//!
//! The `segstore` segment remains the durable source payload. The consumer builds
//! a byte-native postings segment for each stable segment id, stores it under
//! `segstore.idx.<id>.postings_raw`, and loads those sidecars after restart.
//!
//! Run: `cargo run --example postings_sidecar`

use std::collections::BTreeMap;
use std::fs::File;

use durability::{Directory, FsDirectory};
use postings::raw::{write_u64_u32_segment_from_term_postings, RawSegmentFile, RawTermPostingList};
use segstore::{SegmentCatalog, SegmentedStore, SidecarEnvelope, Store};
use serde::{Deserialize, Serialize};

const SIDECAR_KIND: &str = "postings_raw";
const SIDECAR_MAGIC: &[u8; 8] = b"PSTRSC01";
const SIDECAR_VERSION: u32 = 1;
const SIDECAR_RECIPE: &[u8] = b"u64-u32-v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Document {
    terms: Vec<(u64, u32)>,
}

/// Store raw document term vectors as opaque segstore segments.
///
/// The postings sidecar is derived data, not the segment payload. That keeps the
/// lifecycle split explicit: segstore owns WAL/checkpoint/GC, while the consumer
/// owns term mapping and postings bytes.
struct SearchSegments;

impl Store for SearchSegments {
    type Id = u32;
    type Item = Document;
    type Segment = Vec<(u32, Document)>;

    fn build_segment(&self, batch: &[(u32, Document)]) -> Self::Segment {
        let mut segment = batch.to_vec();
        segment.sort_unstable_by_key(|(doc_id, _)| *doc_id);
        segment
    }

    fn merge_segments(
        &self,
        segments: &[&Self::Segment],
        live: &dyn Fn(&u32) -> bool,
    ) -> Self::Segment {
        let mut merged: Self::Segment = segments
            .iter()
            .flat_map(|segment| segment.iter())
            .filter(|(doc_id, _)| live(doc_id))
            .cloned()
            .collect();
        merged.sort_unstable_by_key(|(doc_id, _)| *doc_id);
        merged
    }

    fn segment_len(&self, segment: &Self::Segment) -> usize {
        segment.len()
    }
}

fn doc(terms: &[(u64, u32)]) -> Document {
    Document {
        terms: terms.to_vec(),
    }
}

fn build_raw_postings(segment: &[(u32, Document)]) -> Vec<u8> {
    let mut document_lengths = Vec::with_capacity(segment.len());
    let mut postings_by_term: BTreeMap<u64, Vec<(u32, u32)>> = BTreeMap::new();

    for (doc_id, document) in segment {
        let doc_len = document
            .terms
            .iter()
            .map(|(_, weight)| *weight)
            .sum::<u32>();
        document_lengths.push((*doc_id, doc_len.max(1)));

        for &(term_id, weight) in &document.terms {
            if weight != 0 {
                postings_by_term
                    .entry(term_id)
                    .or_default()
                    .push((*doc_id, weight));
            }
        }
    }

    let term_postings: Vec<_> = postings_by_term
        .iter()
        .map(|(&term_id, postings)| RawTermPostingList::new(term_id, postings))
        .collect();

    write_u64_u32_segment_from_term_postings(&document_lengths, &term_postings)
        .expect("example builds valid raw postings")
}

fn persist_postings_sidecars(store: &SegmentedStore<SearchSegments>, dir: &dyn Directory) {
    for (idx, &segment_id) in store.segment_ids().iter().enumerate() {
        let bytes = build_raw_postings(&store.segments()[idx]);
        let sidecar = SidecarEnvelope::encode(
            SIDECAR_MAGIC,
            SIDECAR_VERSION,
            segment_id,
            SIDECAR_RECIPE,
            &bytes,
        )
        .expect("postings sidecar envelope is valid");
        dir.atomic_write(&store.index_name(segment_id, SIDECAR_KIND), &sidecar)
            .expect("write postings sidecar");
    }
}

fn query_sidecars(catalog: &SegmentCatalog<u32>, term_id: u64) -> Vec<u32> {
    let mut out = Vec::new();

    for &segment_id in catalog.segment_ids() {
        let sidecar = catalog.index_name(segment_id, SIDECAR_KIND);
        let path = catalog
            .dir()
            .file_path(&sidecar)
            .expect("example uses FsDirectory");
        let mut envelope = File::open(&path).expect("postings sidecar exists");
        let sidecar_len = envelope.metadata().expect("sidecar metadata").len();
        let payload = SidecarEnvelope::payload_info(
            SIDECAR_MAGIC,
            SIDECAR_VERSION,
            segment_id,
            SIDECAR_RECIPE,
            &mut envelope,
            sidecar_len,
        )
        .expect("open postings sidecar envelope");
        let mut raw = RawSegmentFile::from_file_range(
            File::open(&path).expect("postings sidecar exists"),
            payload.payload_offset(),
            payload.payload_len(),
        )
        .expect("open raw postings payload");
        let docs = raw
            .candidates_any_terms(&[term_id])
            .expect("query postings sidecar");
        out.extend(docs.into_iter().filter(|doc_id| catalog.is_live(doc_id)));
    }

    out.sort_unstable();
    out.dedup();
    out
}

fn main() {
    let mut root = std::env::temp_dir();
    root.push(format!("segstore-postings-sidecar-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);

    let dir = FsDirectory::arc(&root).unwrap();

    {
        let mut store = SegmentedStore::open(dir.clone(), SearchSegments, 2).unwrap();
        store.add(1, doc(&[(10, 2), (20, 1)])).unwrap();
        store.add(2, doc(&[(10, 1), (30, 1)])).unwrap();
        store.add(3, doc(&[(20, 3)])).unwrap();
        store.add(4, doc(&[(10, 4), (40, 1)])).unwrap();
        store.delete(2).unwrap();
        store.checkpoint().unwrap();

        persist_postings_sidecars(&store, &*dir);
        assert_eq!(
            query_sidecars(&SegmentCatalog::open(dir.clone()).unwrap(), 10),
            vec![1, 4]
        );
    }

    let catalog = SegmentCatalog::<u32>::open(dir.clone()).unwrap();
    assert_eq!(query_sidecars(&catalog, 10), vec![1, 4]);
    assert_eq!(query_sidecars(&catalog, 20), vec![1, 3]);
    assert!(query_sidecars(&catalog, 30).is_empty());

    println!("  [PASS] segstore segment ids can own postings::raw sidecars across restart");

    let _ = std::fs::remove_dir_all(&root);
}
