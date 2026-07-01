---
status: proposed
date: 2026-06-30
scope: segstore, durability, materialized logs, generation and artifact stores
---

# Design: Storage Ecosystem And Generation Store

## Problem

The workspace now has several crates with persistence pressure, but they do not
all want the same store. `segstore` is a good fit for mutable segmented search
indexes. `lexir` already has a checkpoint plus operation-log store built into its
CLI. Training crates such as `tranz`, `subsume`, and `flowmatch` mostly produce
model and experiment artifacts. Treating all of these as "segstore consumers"
would blur different failure models and access patterns.

The goal is to keep the homegrown crates small while using the de facto
community vocabulary at the boundaries: content-addressed artifacts, manifests,
snapshots or generations, conditional publish, reader pins, provenance, and
typed descriptors.

## Context

Existing local decisions matter:

- `durability` owns raw crash-consistent primitives: directories, atomic files,
  checkpoints, record logs, WALs, CRC framing, and sync policy. Postcard is an
  optional typed convenience, not a durability primitive.
- `segstore` owns mutable segment lifecycle: immutable segment files,
  manifest commit point, WAL replay, tombstones, compaction, snapshots, and
  per-segment sidecar garbage collection.
- `segstore` sidecars are rebuildable consumer caches keyed by stable segment
  ids. They are not manifest-committed correctness artifacts.
- `lexir` already proves a separate materialized-log shape: state checkpoint,
  `applied_records` metadata, operation-log suffix replay, validation, compact,
  prune, and doctor commands.

External research points in the same direction:

- BatchWeave-style object stores publish immutable payloads through versioned
  manifests and conditional writes. Payload existence alone is not visibility:
  a batch or generation becomes visible only when a committed manifest names it.
  Watermark-based GC belongs to the manifest/generation layer, not the blob
  layer.
- Iceberg Puffin-style index sidecars bind typed derived artifacts to a snapshot
  while letting engines ignore unknown blob types. The useful fields are blob
  type, covered field ids or subjects, offset/length, compression, and a
  property map carrying algorithm parameters such as dimensions, metric, and
  snapshot id.
- A deeper pass over the Puffin vector-index paper and Iceberg specs sharpens
  the boundary: Puffin statistics files are optional information that readers
  may ignore, but a snapshot can bind a derived artifact to exactly the data
  generation it was computed from. That is the right model for non-rebuildable
  or cross-engine derived indexes, not for segstore's rebuildable per-segment
  caches.
- OxyMake-style artifact keys include declared input digests, rule code,
  parameters, environment, shell, and platform. Correctness only covers declared
  inputs unless a sandbox prevents hidden inputs.
- Atompack-style immutable datasets optimize for mmap and whole-record shuffled
  reads. That is a different shape from a mutable segment store.
- OCI, SRI, and in-toto provide useful exterior vocabulary: media type or
  artifact type, digest algorithm plus digest bytes, subject, predicate type,
  and provenance predicate. Their strongest lesson for v1 is tolerant
  evolution: readers should ignore unknown artifact types or provenance
  predicates unless they explicitly require them.
- Recent reproducible-ML infrastructure work describes the artifact graph as
  datasets, features, workflows, executions, assets, and controlled vocabulary,
  with executions recording inputs, outputs, configuration, and context. That is
  closer to a generation/artifact store than to a segment store.
- Rust prior art separates backend access from data-model lifecycle.
  `object_store` is the strongest fit when we need object-store semantics:
  atomic writes, conditional operations, ranged/vectored reads, and S3/GCS/Azure
  backends. OpenDAL has broader service coverage and layering, but it is still a
  portability layer, not a generation-manifest model. `redb`, `fjall`, and
  other embedded KV engines are useful substrates for specific local indexes,
  but they do not provide artifact descriptors, provenance, or publish
  semantics by themselves.

Modern retrieval and embedding research does not point to one shared index
format. It points to narrower per-consumer work:

- Learned sparse retrieval: Block-Max Pruning for learned sparse indexes, GPU
  scatter-add scoring over GPU-resident inverted indexes, causal/LLM sparse
  encoders, Latent Terms, and sparse late interaction all keep the inverted
  index central but change block layout, query processing, and scorer shape.
- Late interaction: ColBERT-family models, LEMUR, Col-Bandit, FastLane, SLIM,
  SPLATE, ConstBERT, and sparse-coding variants need MaxSim or MaxSim-like
  candidate/rerank APIs. They should not be forced through a plain postings
  list API.
- Vector ANN: LSM-VEC, Quake, QuIVer, RaBitQ, DiskANN3, adaptive beam search,
  and in-place graph update work all argue for algorithm-owned sidecars and
  workload-specific maintenance. Segment lifecycle helps, but it is not the
  algorithm.
- Region and ontology models: RegD, octagons, TaxoBell, BoxLitE, SectorE, and
  description-logic box variants need geometry descriptors and query semantics
  more than generic vector storage.
- Model-output crates: `tranz`, `subsume`, `flowmatch`, `hopfield`, and
  `symproj` need artifact descriptors for checkpoints, codebooks, generated
  samples, evaluation results, and training configuration. Their storage need is
  reproducibility and comparability, not segment compaction.

## Chosen Approach

Keep four store shapes distinct.

1. `durability`: raw local crash primitives.
2. Materialized log: deterministic state plus operation log plus checkpoint
   metadata. This should be extracted from `lexir` only after a second consumer
   or a cleanup pass proves the generic shape.
3. `segstore`: mutable segmented indexes whose raw segment payloads are the
   source of truth. Built indexes remain sidecars with consumer-owned recipes.
4. Generation and artifact store: immutable artifact blobs plus typed
   descriptors and generation manifests. This is for model checkpoints,
   datasets, trained codebooks, benchmark outputs, and non-rebuildable derived
   products.

The generation/artifact split should be conceptual first and crate-level only
when implementation pressure appears:

- Artifact store: content-addressed bytes, descriptor metadata, optional
  provenance records, and local atomic writes. Descriptor fields should include:
  `artifact_type` or `media_type`, digest algorithm, digest bytes, byte length,
  codec/schema version, properties, optional subject links, and optional
  provenance statement references.
- Generation store: a durably published manifest naming a set of artifact
  descriptors, declared input descriptors, and optional execution/provenance
  records. A generation is visible only after the manifest is published.
  Readers can pin a generation. Garbage collection uses reader pins and
  watermarks.

The refined generation-store contract is:

- `ArtifactDescriptor`: media-type-like kind, digest string, byte length,
  codec/schema version, optional properties, optional subject links, and optional
  provenance references.
- `GenerationManifest`: immutable set of output artifact descriptors, declared
  input descriptors, execution/config metadata, and optional parent/base
  generation id.
- `GenerationPointer`: the small durable commit point naming the current
  manifest. Payload bytes are written first; the pointer is published last.
- `ReaderPin`: an in-process or persisted lease saying a reader may still use a
  generation. Garbage collection must keep the manifest and transitive artifacts
  while any pin or retention watermark protects them.
- `Validation`: consumers verify size and digest before trusting bytes, then
  verify their own schema/recipe properties before using the artifact.
- `Evolution`: unknown artifact kinds, codecs, and provenance predicates are
  skipped unless the caller explicitly requires them.

This is useful when at least one of these is true:

- The derived state is expensive or impossible to rebuild from local segment
  payloads.
- Readers need stable snapshot semantics across process restarts or concurrent
  writers.
- A model, dataset, codebook, benchmark result, or trained global index needs
  reproducibility metadata.
- A corpus-level trained artifact, such as IVF/PQ codebooks or a global ANN
  generation, must be published atomically against a specific input snapshot.

It is not useful for ordinary segstore sidecars while those sidecars are
rebuildable from raw segment files.

Use standard-friendly identifiers without adopting a full external system in v1:

- Support algorithm-tagged digests. Prefer `sha256` in exported descriptors
  because OCI, SRI, and in-toto ecosystems understand it. Allow `blake3` for
  local fast cache keys when explicitly marked. If both are present, treat
  `sha256` as the portability digest and `blake3` as a local acceleration key.
- Use media-type-like strings for artifact kinds when exporting. Internally, a
  small typed enum or validated slug is enough.
- Model provenance after in-toto's shape: subject digest, predicate type, and
  predicate bytes or structured data. Do not implement signing or policy
  verification in v1.
- Keep manifests append-friendly and tolerant of unknown records. Readers should
  be able to load artifacts they understand and skip descriptors whose
  `artifact_type`, codec, or predicate type is unknown.

## Options Considered

### Put generation support into segstore

Rejected. `segstore` is about mutable item streams, tombstones, compaction, and
segment-local queries. Generation manifests are about publishing immutable sets
of artifacts. The two can compose, but forcing them into one API would couple
search-index lifecycle to experiment/model artifact lifecycle.

### Make segstore sidecars manifest-tracked now

Rejected by the existing sidecar design. Most current sidecars are rebuildable
from raw segments, and segstore does not know the consumer's schema, tokenizer,
metric, HNSW parameters, MinHash seeds, block-max layout, or quantizer recipe.
If a consumer creates a non-rebuildable sidecar, that is a gate to design
manifest-tracked index generations.

### Adopt Iceberg, Delta, OCI, or a registry as the native format

Deferred. Their terms are useful, and compatibility adapters may be useful, but
the local need is smaller: local filesystem durability, typed descriptors,
conditional publish where available, and explicit provenance. A registry or
table format would add network and governance assumptions the current crates do
not need.

### Use `redb`, `fjall`, `object_store`, or `opendal` as the abstraction

Deferred. `redb` and `fjall` are good embedded storage engines but they do not
define this lifecycle. `object_store` and `opendal` expose backend capability
questions we should copy: conditional create, conditional update, atomic rename,
multipart caveats, and copy semantics. They become useful when a non-local
backend is required.

## Per-Crate Fit

- `lexir`: best first candidate for a materialized-log helper. Its CLI now has
  one internal recovery helper for checkpoint plus unapplied log suffix and one
  internal rewrite helper for compact/prune. Extraction is still not justified
  until another crate needs the same pattern.
- `postings`: should grow query/index features, not storage first. The next
  storage-relevant artifact is a sparse postings sidecar with block maxima and
  codec/schema version, likely through `segstore` if postings becomes segmented.
- `vicinity`, `precinct`, `sporse`, `sketchir`: keep using `segstore` sidecars,
  each with algorithm-specific recipe validation. Avoid one shared sidecar
  codec until two consumers converge on the same header mechanics.
- `tranz`: needs artifact export more than segstore. It now writes a local
  `manifest.json` beside `entities.tsv` and `relations.tsv`, with model family,
  score convention, split counts, SHA-256 digests, byte sizes, training config,
  final loss, and optional aggregate metrics. Dataset digests and binary matrix
  formats such as `.npy` or safetensors can come later without turning this into
  a shared store yet.
- `subsume`: same artifact need, but with geometry-specific descriptors. Its
  current JSON save/load for `TrainedElModel` is a model artifact, not a store.
  A descriptor should record region family, dimension, containment direction,
  beta/temperature schedule, dataset digest, ontology vocabulary, checkpoint
  format, and metric/eval summary.
- `flowmatch`: generation manifests fit experiment outputs: dataset digest,
  coupling method, ODE solver, training config, seed, model parameters, metrics,
  and generated sample artifacts. The current USGS report validates the report
  vocabulary but does not yet reference artifact files or content digests.
- `symproj`: no store yet, but trained or imported codebooks are natural
  artifacts once they become external files. A descriptor should record tokenizer
  identity, vocabulary size, dimension, pooling/normalization policy, source
  model or corpus, and matrix digest.
- `hopfield`: no store yet. If memory banks become durable data products, they
  fit artifact descriptors: pattern count, dimension, normalization, energy
  function, separation map, and source dataset digest. They do not need segstore
  unless updates become segmented and queryable.
- `heyting`: no immediate store extraction. It mostly adapts scorer APIs; it
  should consume `tranz` or `subsume` descriptors when external scorers become
  loadable assets, not define its own persistence model first.

## Implementation Status

The sidecar path is now implemented in four consumers, and the variation is
real enough to keep the consumer-owned recipe boundary.

- `vicinity 0.10.5`: released graph persistence under `persistence`, plus
  per-segment HNSW sidecars guarded by dimension, metric, normalization, HNSW
  build parameters, and codec recipe.
- `precinct`: per-segment region sidecars persist the original regions, id map,
  center HNSW graph, and lifted power-distance HNSW graph. Bench evidence over
  8k 64d regions: cold restart with sidecars measured `[10.150 ms 10.350 ms
  10.553 ms]`; missing sidecars rebuilt in `[1.7605 s 1.7678 s 1.7750 s]`.
- `sporse`: per-segment WAND sidecars persist finalized posting lists and
  block-max metadata plus the live id set. Bench evidence over 20k sparse docs:
  cold restart with sidecars measured `[31.390 ms 31.937 ms 32.538 ms]`;
  missing sidecars rebuilt in `[105.88 ms 107.20 ms 108.85 ms]`.
- `sketchir`: per-segment MinHash sidecars persist the built LSH block and
  insertion-order id map, guarded by the `BlockingConfig` recipe. Bench evidence
  over 20k documents: cold restart with sidecars measured `[42.303 ms 42.631 ms
  42.942 ms]`; missing sidecars rebuilt in `[478.41 ms 480.19 ms 482.13 ms]`.

The shared envelope shape is visible, but it should still stay local for now:
the recipe contents and payload validation differ by algorithm, while segstore
only needs to provide stable sidecar names and garbage collection.

`lexir` now has an internal materialized-log cleanup: `log-add`, `log-delete`,
`log-search`, `log-checkpoint`, and `log-validate` share the same
checkpoint-plus-suffix recovery path, and `log-compact`/`log-prune` share the
self-contained rewrite path. This reduces the risk of diverging meta semantics,
but it is still one consumer, so it does not cross the crate-extraction gate.

`tranz` is the first concrete artifact producer, but only at the crate-local
descriptor layer. Its training CLI writes `manifest.json` for exported embedding
TSVs and records per-file SHA-256, byte length, model/training config, split
counts, final loss, and optional aggregate metrics. That validates the
descriptor shape without committing to `genstore`, `digest-store`, or a shared
artifact lifecycle crate.

## Non-Goals

- Do not turn `durability` into a KV store, table format, or artifact registry.
- Do not make `segstore` own consumer index recipes or query semantics.
- Do not require object-store dependencies for the local filesystem v1.
- Do not implement signing, admission policy, or SLSA verification in the first
  artifact store.
- Do not use content addressing as a substitute for declared-input tracking.
  Undeclared inputs still make a recipe cache unsound.

## Implementation Plan

1. Finish algorithm-local wins before new crate work. Done for the immediate
   batch: `postings::top_k_weighted` has a dense accumulator fast path, and the
   four segstore-backed consumers above have restart sidecars.
2. Add Block-Max or Block-Max-MaxScore style learned-sparse primitives to
   `postings` only if benchmarks show the exact weighted scorer is now the
   bottleneck. Keep `sporse`'s WAND path independent unless a shared primitive
   removes real duplication.
3. Release the `vicinity` sidecar path before moving `precinct`. Done:
   `vicinity 0.10.5` exposes graph postcard persistence under `persistence`, and
   `precinct` uses a region-aware sidecar format rather than a vector-only one.
4. Keep the `lexir` materialized-log cleanup internal. Do not extract until a
   second consumer needs checkpoint plus operation-log replay.
5. Design an artifact descriptor type on paper before code. Include digest,
   size, kind, codec/schema version, properties, subject links, and provenance
   hooks.
6. Keep `tranz`'s first descriptor local. Add shared descriptor code only after
   a second producer needs the same field names and validation rules. `flowmatch`
   is close, but its current report is still a run record without artifact-file
   descriptors.
7. Add generation manifests after artifact descriptors exist. Start
   single-writer local. Add conditional publish and object-store adapters only
   behind an explicit backend capability design.
8. Keep `segstore` sidecar adoption separate and consumer-by-consumer. The first
   adoption wave confirms this: HNSW, region HNSW, WAND, and MinHash all share
   lifecycle mechanics but not payload recipes.

## Naming Shortlist

Names should be plain and searchable. Current `cargo search` checks found no
exact hits for `matlog`, `viewlog`, `genstore`, `digest-store`, or `casstore`;
`blobstore` is already taken, and underscore searches were too broad to treat as
availability evidence.

- Materialized log: `matlog` is shortest and matches the abstraction. `viewlog`
  is clearer if the crate centers "materialized view plus operation log" rather
  than the checkpoint mechanics.
- Artifact store: `digest-store` says content-addressed bytes and avoids the
  generic `blobstore` name. `artifactstore` is plain but long and easy to
  confuse with workflow-engine artifact APIs.
- Generation store: `genstore` is the best placeholder if the crate publishes
  generation manifests. `manifeststore` is more explicit if the artifact store
  remains separate and the crate mainly publishes manifest sets.

Provisional preference: `matlog`, `digest-store`, and `genstore`.

## Decision Gates

- If two crates independently implement checkpoint plus operation-log replay,
  extract the materialized-log helper.
- If a consumer needs to publish immutable experiment/model outputs with
  reproducibility metadata, implement the artifact descriptor/store before
  adding more ad hoc save/load paths.
- If readers need to pin multiple visible artifact sets while writers publish
  replacements, promote from artifact descriptors to a generation manifest plus
  reader-pin and retention rules.
- If any artifact store needs S3/GCS/Azure, design a backend capability matrix
  before choosing `object_store`, `opendal`, or a custom trait. Default toward
  `object_store` if conditional object writes and range reads are the central
  need; default toward OpenDAL only if broad backend coverage is itself the
  requirement.
- If a segstore sidecar becomes non-rebuildable from raw segment data, stop and
  design manifest-tracked index generations.
- If external interoperability becomes a user requirement, add OCI or
  in-toto-style export adapters rather than changing the internal v1 format.

## Open Questions

- Crate names are not chosen. `artifactstore`, `genstore`, and `matlog` describe
  the bounded contexts, but naming should happen when implementation starts.
- Digest policy needs a final default. Current leaning: `sha256` is required
  for exported descriptors; `blake3` is optional for local cache keys.
- The second concrete artifact producer is still open. `flowmatch` has the
  clearest generation/provenance gap, but its current report lacks artifact
  descriptors and digests. `subsume` model checkpoints are the other likely
  forcing function.

## Primary Source Read Depth

- Puffin-backed vector indexes, arXiv:2606.04196: read abstract, sections 2.1,
  3.2, 4, 5, 7, 10, 11, and conclusion from the PDF text. I did not read every
  cited ANN reference or the companion wire-format document.
- Apache Iceberg table spec and Puffin spec: checked snapshot, manifest-list,
  table-statistics, and BlobMetadata fields. The important spec point is that
  statistics files are informational and readers may ignore them, while blob
  metadata carries type, field ids, snapshot id, offset, length, compression, and
  properties.
- OCI descriptor spec: checked descriptor fields and digest rules. The useful
  portable minimum is still media type, digest string, and byte size, with
  SHA-256 required for compliant descriptor verification.

## Research Sources

- BatchWeave: versioned manifests, conditional object writes, manifest-gated
  visibility, and watermark-based reclamation for training batches.
  <https://arxiv.org/abs/2605.09994>
- OxyMake: content-addressed workflow keys over declared inputs, rule source,
  parameters, environment, shell, and platform, with the undeclared-input caveat.
  <https://arxiv.org/abs/2606.20989>
- Puffin-backed vector indexes: typed sidecar blobs bound to Iceberg snapshots
  with unknown-blob tolerance and properties for algorithm validation.
  <https://arxiv.org/abs/2606.04196>
- Apache Iceberg table and Puffin specs: snapshots, manifest lists,
  informational statistics files, and BlobMetadata fields.
  <https://iceberg.apache.org/spec/> and
  <https://iceberg.apache.org/puffin-spec/>
- Reproducible ML infrastructure: Dataset, Feature, Workflow, Execution, Asset,
  and Controlled Vocabulary as first-class artifact types.
  <https://arxiv.org/abs/2506.16051>
- Rust backend prior art: `object_store` and OpenDAL are backend abstraction
  layers, not generation-store models. <https://docs.rs/object_store> and
  <https://opendal.apache.org/>
- Descriptor vocabulary: OCI descriptors, W3C SRI, and in-toto/SLSA provenance.
  <https://github.com/opencontainers/image-spec/blob/main/descriptor.md>,
  <https://www.w3.org/TR/sri-2/>, and
  <https://slsa.dev/blog/2023/05/in-toto-and-slsa>
- Dynamic and learned-index pressure: IVF-TQ highlights codebook drift in
  streaming vector search; recent SPLADE variants and Latent Terms reinforce
  postings as a first-class learned-sparse substrate rather than a storage-layer
  concern. <https://arxiv.org/abs/2605.17415>,
  <https://arxiv.org/abs/2505.15070>, and
  <https://arxiv.org/abs/2605.29384>

---
Decided: 2026-06-30 | Session: Codex handoff from Claude 03edba07
