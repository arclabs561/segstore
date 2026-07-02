# Segstore, Durability, and Sidecars

Status: accepted, initial hardening implemented on 2026-06-30.

## Problem

Segstore now has stable segment ids and a reserved per-segment sidecar namespace
for persisted built indexes. The design needs to keep three layers separate:
durability's raw crash primitives, segstore's segment lifecycle, and each
consumer's query/index encoding.

## Decision

Keep the layer split:

- `durability` owns raw byte durability primitives and optional postcard typed
  helpers.
- A missing `materialized log` layer should own the generic checkpoint plus
  operation-log pattern now implemented manually by `lexir`: store a materialized
  state checkpoint, record how many log entries it includes, replay only the
  suffix, and validate or compact the log when needed.
- `segstore` owns segment lifecycle: WAL epoch, manifest, segment ids,
  tombstones, compaction, snapshots, and sidecar garbage collection.
- Consumers own sidecar encoding and validation. A sidecar is a rebuildable
  cache keyed by stable segment id, not part of segstore's manifest commit.

Harden the existing contracts before adding broader APIs:

1. `durability::recordlog::RecordLogWriter` poisons after write, flush, or sync
   failure, matching `walog`'s error posture.
2. Segstore exposes checked sidecar naming via `try_index_name`; the existing
   `index_name` convenience method panics on invalid sidecar kinds.
3. Segstore accepts legacy sidecar names with now-invalid kinds for garbage
   collection, so upgrading does not strand old derived artifacts.
4. Consumer sidecars must carry their own validity data, such as format version
   and algorithm/config assumptions. Missing, corrupt, or mismatched sidecars
   must be rebuilt from the raw segment.

## Per-Consumer Sidecar Recipes

Consumers do not share one validation algorithm. The common invariant is only
that a sidecar is attached to a stable immutable segment id. Whether the sidecar
is reusable depends on the consumer's current index recipe.

`kind` is therefore only a short filesystem namespace and coexistence key. It is
not the full validity proof. The sidecar payload must include a consumer-owned
recipe descriptor or fingerprint covering every choice that changes the built
structure or its results:

- sidecar codec and format version
- algorithm family and parameters that affect build output
- input transform, tokenizer, schema, dimensionality, and distance metric
- persisted feature flags that change layout or interpretation
- source assumptions, such as live ids covered by the segment

Consumers vary enough that centralizing this descriptor in segstore would be the
wrong layer:

- `sporse`: sparse-vector schema, adapter version into `postings`, learned-sparse
  weighting assumptions, and any nonnegative-weight contract. Reusable
  posting-list codec, WAND, MaxScore, block-max, and BMP-style metadata belong
  in `postings`; a `sporse` sidecar records which `postings` recipe it used.
  Query-time `k` is not part of the recipe.
- `sketchir`: `BlockingConfig` (`num_bands`, `num_hashes_per_band`,
  `ngram_size`, `char_ngrams`), shingling/tokenizer version, MinHash hash
  family and seed, and id-map layout.
- `vicinity` HNSW: dimension, metric, normalization policy, `m`, `m_max`,
  `m_l`, `ef_construction`, seed selection, diversification policy, deterministic
  seed when present, and sidecar codec or layout flags. Query-time `ef` is not
  part of the recipe.
- `precinct`: region type, dimension, region-to-vector lift transform version,
  HNSW parameters for center and lift indexes, metric, and sidecar codec version.
  Query-time `ef` and over-retrieval are not part of the recipe.
- Future `vicinity` variants such as Vamana, IVF, scalar quantization, or RaBitQ
  are separate recipes. They may share the same segment id, but they do not share
  compatibility rules.

If multiple variants need to coexist for one segment, put a short stable slug or
hash in `kind` (for example `hnsw-a1b2c3d4`) and the full descriptor in the
payload. If only one active variant exists, use a coarse `kind` such as `hnsw`
and let recipe mismatch trigger rebuild. In both cases, segstore should only
reserve the name and garbage-collect it with the segment.

For compatibility, sidecar garbage collection is more tolerant than sidecar name
generation: it recognizes old files such as `segstore.idx.<id>.hnsw.v1` so
compaction can remove them, even though `try_index_name` rejects dotted `kind`
values for new writes.

## Algorithm-Specific Optimization Shape

The optimization target is different for each consumer. A useful sidecar is not
"whatever serde can dump"; it is the smallest loadable artifact that preserves
the expensive build work and can reject stale assumptions before search.

- `postings` should own finalized posting-list structures: sorted doc ids,
  weights, compressed blocks, per-block maxima, list maxima, and any WAND,
  MaxScore, or BMP-style metadata. `sporse` can persist or reference those
  structures for learned sparse vectors, but its recipe descriptor should cover
  sparse-vector assumptions and the `postings` recipe version rather than fork
  the posting-list engine. Query-time `k` and any current heap threshold stay
  outside the sidecar. A later search optimization can share a global top-k
  threshold across segments to let Block-Max WAND prune harder; that is a query
  executor concern, not a sidecar-format concern.
- `sketchir` should persist the MinHash signatures, band buckets, and insertion
  order id map for a segment. The text source remains in the segment. The recipe
  must cover shingling/tokenization and deterministic hash seeds because changing
  either makes bucket collisions incomparable. Query text should still be
  shingled and min-hashed once per query, then reused across all segment blocks.
- `vicinity` HNSW should persist the graph envelope plus vectors/doc ids, guarded
  by dimension, metric, normalization policy, HNSW build parameters, seed and
  diversification choices, codec version, and relevant compile-time layout flags.
  Query-time `ef` is deliberately excluded. This matches hnswlib's distinction:
  `save_index` persists the graph while `ef` is a runtime setting that must be set
  after load.
- `precinct` should persist a region sidecar, not a generic "two HNSWs" blob:
  region type, original regions, id map, center graph, lifted graph, lift transform
  version, and the segment-level normalization constant used by the lift. The
  center and lift graphs share some HNSW parameters but have different dimensions
  and query semantics.
- Quantized or partitioned `vicinity` families need different sidecar shapes:
  IVF-PQ/IVF-AVQ/RaBitQ persist trained centroids, codebooks or rotations, encoded
  vectors, inverted lists, and rerank source data when needed. Vamana and DiskANN
  persist flat graph topology plus entry/medoid and any disk/cache layout. These
  recipes should not be forced through an HNSW envelope.

Do not introduce a shared `SidecarCodec` trait in segstore yet. The repeated
pattern is real, but the variation is still semantically important. A shared
trait becomes useful only after at least two consumers converge on the same
header mechanics while keeping their recipe descriptors separate.

## Segstore Fit by Algorithm Family

Segstore is right when the consumer can search immutable per-segment indexes and
merge their local results without violating the algorithm's intended quality
model. It is not automatically right for every ANN method.

Good fits:

- Sparse WAND/posting-list search. Segments are natural inverted-list shards;
  per-segment top-k is exact and the global merge is exact when each segment
  returns its local top-k.
- MinHash/LSH blocking. Segment-local bucket hits can be unioned and deduped.
  The query signature is shared across segments, so fan-out is mostly bucket
  lookup plus id-map translation.
- HNSW over modest or churn-heavy corpora. Per-segment graphs give durable
  incremental adds/deletes without mutating one large graph. Recall is approximate
  either way, but many tiny graphs can hurt recall/latency, so compaction and
  segment-count metrics matter.
- Region indexes like `precinct` when per-segment candidate generation plus
  rerank is acceptable. The region lift and rerank keep semantics local to the
  consumer.

Conditional fits:

- ScaNN-style IVF-AVQ. The expensive state is trained centroids, residual
  codebooks, anisotropic quantizer state, encoded residuals, and rerank source
  vectors. Per-segment training is easy to persist but may reduce global recall
  versus one corpus-trained partitioner. A good design likely needs either
  bounded segment counts, background global rebuild, or a two-level plan where
  segstore owns raw durability and the consumer owns a global trained index
  generation.
- IVF-PQ/OPQ/RaBitQ. Same concern: sidecars can persist centroids, codebooks,
  rotations, quantized codes, and inverted lists, but the quality of per-segment
  training must be measured against a global training baseline.
- Vamana/DiskANN. Segment-local flat graphs are easy to persist, but the algorithms
  are often chosen for large global graph quality and disk-aware access layout.
  Segstore helps update durability; it may fight the point of a single optimized
  SSD graph unless a consolidation layer builds larger read-optimized generations.

Poor fits unless the consumer adds another layer:

- Algorithms whose correctness or quality depends on one global trained model and
  cannot tolerate per-segment models.
- Algorithms whose main feature is in-place online graph repair. Segstore's model
  is immutable segment plus rebuild/merge, so in-place mutation belongs in the
  consumer's own structure or a separate fresh-graph layer.
- Indexes whose persisted artifact is non-rebuildable from segment payloads. Those
  need manifest-tracked index generations, not best-effort sidecars.

## Lexir and Materialized Logs

`lexir` is the counterexample that keeps this design honest. It is a lexical
ranking layer over `postings`, and its CLI already has a checkpointed operation
log: `Add` and `Delete` records are appended to a record log, a whole
`InvertedIndex` snapshot is written as `index.bin`, and a sidecar meta file tracks
how many log records the checkpoint includes. The CLI can diagnose missing meta,
validate checkpoint-plus-suffix replay against full log replay, compact the log,
and prune history.

That is a real storage shape, but it is not segstore. It is an event-sourced
materialized view:

- `State`: a materialized index such as `lexir::bm25::InvertedIndex`.
- `Op`: deterministic updates such as add-document and delete-document.
- `Checkpoint`: state plus `last_applied_record`.
- `Recovery`: load checkpoint, replay suffix, reject ambiguous checkpoint/log
  combinations instead of guessing.
- `Maintenance`: validate, compact, prune, and repair metadata.

This belongs above `durability` and beside `segstore`. It is useful when the
materialized state is cheap enough to checkpoint whole and replay from an op log.
It is the wrong fit when the expensive object is a large trained or graph index
whose rebuild cost dominates; those want segstore sidecars or published
generations.

For `lexir`, there are two plausible futures:

- Keep the current exact lexical index as a materialized-log store. This preserves
  simple state-machine semantics and strong replay validation for small and
  medium corpora.
- Move the postings payload itself toward segstore-style segments when
  whole-index checkpoints become too expensive. Then global BM25 statistics
  (`N`, `df`, `avgdl`) must be handled explicitly so segment-local candidate
  generation does not corrupt global scoring.

## Crate-Local Artifact Producers

Two consumers now produce durable descriptors, but they are still different
enough that a shared "artifact store" crate would be premature:

- `tranz` writes an embedding export manifest. Its durable unit is a set of
  artifact files plus metadata: schema, model, score order, artifact bytes and
  SHA-256 digests, embedding dimensions and row counts, dataset split counts,
  training config, final loss, and optional eval metrics.
- `flowmatch` writes a USGS pipeline run report when `FLOWMATCH_REPORT_OUT` is
  set. Its durable unit is a structured report: schema, dataset identity,
  training/evaluation configs, seeds, metric pairs, and timing breakdowns. It
  does not yet reference separate artifact files or content digests.

This is enough to standardize vocabulary in future designs: schema id, producer,
inputs, config, metrics, timings, artifacts, and digests. It is not enough to
standardize storage mechanics. `tranz` needs a file-artifact manifest;
`flowmatch` needs a run-report record; sidecar consumers need segment-attached
rebuildable caches; `lexir` needs a materialized operation log.

Use current crate names when describing these relationships. `vicinity` is the
ANN/HNSW crate formerly referred to as `jin`. `tranz` is the point-embedding KGE
crate for TransE/RotatE/ComplEx/DistMult. `sheaf` is still the current facade for
the cluster, Leiden, distribution-distance, and kNN graph evaluation APIs used
by `flowmatch`.

## Segstore Feature Use

What consumers already get from segstore:

- durable incremental adds/deletes through the WAL
- stable immutable segment ids for restart-stable sidecar names
- `Arc` segment identity for in-process cache reuse
- checkpoints that write only new segments
- tombstones plus compaction/reclaim to remove dead data
- sidecar garbage collection when segments disappear
- snapshot views for concurrent readers

Features we are not using enough yet:

- `Reader`/`View` in consumer query APIs. Current wrappers query through the writer
  object; read-heavy consumers should expose snapshot readers so long searches do
  not care about concurrent mutations.
- `segment_sizes`, `space_amplification`, and `tombstone_count` as policy signals.
  These should drive when to compact, force-merge, or rebuild a global trained
  generation.
- `force_merge_to` before read-heavy phases. Some algorithms should explicitly
  collapse segment fan-out before measuring query latency or recall.
- sidecar hit/miss/rebuild counters. The substrate exposes the hooks, but
  consumers need metrics to know whether sidecars are actually buying restart
  time.

Features that are intentionally irrelevant at the segstore layer:

- algorithm parameter tuning (`ef`, `nprobe`, `num_reorder`, rescoring, WAND heap
  thresholds). These are query-time executor concerns.
- algorithm-specific codec choices. Postcard is fine for HNSW graphs today; sparse
  postings, quantized codes, and disk graphs may need custom binary layouts.
- global training policy for ScaNN/IVF/quantized families. Segstore can preserve
  raw segments and sidecars; it should not decide whether per-segment training is
  good enough.

## Research Basis

Lucene and Tantivy use immutable segments plus an atomically published segment
metadata file. Searchers hold immutable segment-reader snapshots while commits,
merges, and garbage collection proceed separately. This supports segstore's
manifest plus immutable segment files model.

RocksDB separates WAL data from MANIFEST state edits and exposes both strict and
point-in-time WAL recovery modes. Segstore's current point-in-time WAL recovery
is the right crash-recovery default; strict media-corruption detection can be a
future option if consumers need it.

Cassandra STCS groups similarly-sized immutable tables for compaction, which
matches segstore's current size-tiered compaction shape. More compaction policy
should come from measured segment fan-out or tombstone density, not from adding
adjacent-system knobs preemptively.

sled and redb are useful mainly for crash-model discipline. Their tests model
durable versus maybe-durable states and inject failures below high-level storage
calls. Segstore already had directory-level fault tests; the missing local piece
was writer-level write/flush/sync failure coverage in `recordlog`.

Lance and Weaviate both support persisted derived index artifacts. They also
show the boundary: index metadata is appropriate when the storage layer owns
fields, schema, fragment coverage, index type, and index version. Segstore does
not own those semantics, so its sidecars remain consumer-owned caches. Weaviate's
HNSW persistence also shows why filename classification matters: commit logs,
snapshots, compacted files, and temp files need non-overlapping names.

Faiss documents the build-time/runtime split directly: indexes have parameters
that must be chosen when the index is built and separate runtime parameters such
as IVF `nprobe` or HNSW `efSearch` that can be tuned later. hnswlib makes the
same split concrete: `save_index`/`load_index` persist the graph, while `ef` is a
query-time tradeoff that is not saved by `save_index`. Weaviate's named vectors
show the per-consumer/per-vector version of the same idea: each vector space can
choose its own index, compression, vectorizer, and distance metric. Qdrant's
quantization docs likewise separate storage/build choices such as scalar/product
quantization and `always_ram` from query-time choices such as rescoring.

## Options Rejected

Moving manifest logic into `durability` was rejected. The manifest names segment
ids, tombstones, and `next_seg_id`; those are segstore concepts. Durability
should provide atomic files, WALs, checkpoints, CRC framing, and sync helpers.

Switching segstore from `recordlog` to `walog` was deferred. `walog` has stronger
lifecycle machinery, but segstore only needs one epoch-suffixed log between
checkpoints. The missing poison-after-failure posture belongs in `recordlog`.

Manifest-tracking sidecars was rejected for now. Sidecars should remain
consumer-owned caches until a consumer proves it needs non-rebuildable index
commits.

Strict recovery mode was deferred. The current point-in-time behavior is
documented and safe for crash recovery: recover a CRC-validated prefix, never
garbage. Add strict mode only when a consumer needs media-corruption detection
semantics.

## Non-Goals

- Do not turn segstore into a vector database or query engine.
- Do not make segstore own consumer sidecar encoding.
- Do not add a second manifest format for sidecars without a non-rebuildable
  index use case.
- Do not add delete-ratio auto-compaction without consumer metrics.
- Do not add group commit or a concurrent writer redesign without benchmarks.

## Decision Gates

- If two consumers need strict interior-corruption detection, add a segstore
  `RecoveryPolicy`.
- If a consumer stores an index that cannot be rebuilt from segment payloads,
  design manifest-tracked index metadata before shipping that consumer.
- If segment fan-out, tombstone density, or sidecar rebuild misses dominate a
  consumer benchmark, revisit compaction policy with that benchmark as input.
