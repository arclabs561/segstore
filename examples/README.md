# segstore examples

Each example is runnable from the repo root. Output excerpts below are real,
captured from release runs.

## Which example should I run?

| I want to... | Example |
|---|---|
| Back an inverted index with segmented storage | `inverted_index` |
| Back a vector index with search-all-merge segments | `vector_search` |
| Persist a built per-segment index sidecar | `persist_index` |

## Index Shapes

### `inverted_index`: can `Store` model an updatable inverted index?

Uses terms as postings lists inside immutable segments. Deletes are tombstones;
compaction merges postings while dropping deleted documents.

```bash
cargo run --release --example inverted_index
```

```text
indexed 4 docs; df(rust)=3
after delete(1): df(rust)=2
  [PASS] segstore backs an updatable inverted index: build, delete, compact, recover
```

This is the native Lucene-like shape for segstore: many immutable segments,
query across live segments, then compact when convenient.

### `vector_search`: can vector search use the same segment lifecycle?

Stores `(id, vector)` batches as immutable segments and runs k-NN by searching
all live segments plus the live buffer, then merging the top-k.

```bash
cargo run --release --example vector_search
```

```text
indexed 4 vectors across 2 segments
  [PASS] segstore backs a multi-segment vector index (search-all-merge): delete, compact, recover
```

The example brute-force scans each segment for clarity. A real consumer would
put an ANN graph inside each segment and merge per-segment top-k results.

## Persistence Hook

### `persist_index`: can a consumer cache built per-segment indices?

Builds a toy sorted sidecar for each segment, writes it under segstore's
reserved index-sidecar name, opens the manifest catalog, then loads the sidecars
instead of rebuilding them. A missing or stale sidecar decodes only that segment.

```bash
cargo run --release --example persist_index
```

```text
built + persisted 10 per-segment index sidecars
catalog reopen over 10 segments: rebuild-all 164.542µs vs load-all 438.458µs
  rebuilt invalid/missing sidecars: 0
  [PASS] persisted indices load on reopen instead of rebuilding
  (the win scales with build cost; a real HNSW build dwarfs this toy build)
```

The toy build is cheap, so the timing is not a benchmark. The point is the
sidecar contract: a consumer-owned magic/version/recipe header plus segment-id
keying lets expensive built indices survive restarts.
