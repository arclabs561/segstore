//! Hot-path benchmarks: ingest (WAL append + flush per op), size-tiered
//! compaction drain, full compaction, and recovery (WAL replay on open).
//!
//! Run: `cargo bench`. Backend is in-memory, so these isolate segstore's own
//! overhead (op orchestration, bucketing, segment-list manipulation) from real
//! disk IO -- the relative costs and any O(n^2) surprises are what matter here.

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use durability::MemoryDirectory;
use segstore::{Options, SegmentCatalog, SegmentedStore, Store};

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

fn fresh(flush: usize) -> SegmentedStore<Kv> {
    SegmentedStore::open_with_options(MemoryDirectory::arc(), Kv, Options::new(flush)).unwrap()
}

fn fill(s: &mut SegmentedStore<Kv>, n: u32) {
    for i in 0..n {
        s.add(i, format!("v{i}")).unwrap();
    }
}

fn bench_add(c: &mut Criterion) {
    let mut g = c.benchmark_group("add");
    for &n in &[1_000u32, 4_000, 16_000] {
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(|| fresh(64), |mut s| fill(&mut s, n), BatchSize::SmallInput);
        });
    }
    g.finish();
}

fn bench_compact_tiers(c: &mut Criterion) {
    let mut g = c.benchmark_group("compact_tiers_drain");
    for &n in &[1_000u32, 4_000, 16_000] {
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut s = fresh(16);
                    fill(&mut s, n);
                    s
                },
                |mut s| {
                    s.compact_tiers().unwrap();
                },
                BatchSize::SmallInput,
            );
        });
    }
    g.finish();
}

fn bench_full_compact(c: &mut Criterion) {
    c.bench_function("compact_full_4k", |b| {
        b.iter_batched(
            || {
                let mut s = fresh(16);
                fill(&mut s, 4_000);
                s
            },
            |mut s| {
                s.compact().unwrap();
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_recovery(c: &mut Criterion) {
    let mut g = c.benchmark_group("recover");
    for &n in &[1_000u32, 4_000] {
        let dir = MemoryDirectory::arc();
        {
            let mut s =
                SegmentedStore::open_with_options(dir.clone(), Kv, Options::new(64)).unwrap();
            fill(&mut s, n);
        }
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let s =
                    SegmentedStore::open_with_options(dir.clone(), Kv, Options::new(64)).unwrap();
                criterion::black_box(s.segment_count());
            });
        });
    }
    g.finish();
}

/// Bulk ingest on a real filesystem: per-item add (one WAL flush per item) vs
/// extend (one flush for the batch). In-memory hides this (flush is free); on disk
/// the flush is the cost extend amortizes.
fn bench_ingest_fs(c: &mut Criterion) {
    use durability::FsDirectory;
    let mut g = c.benchmark_group("ingest_fs_2k");
    let n = 2_000u32;
    let mk = |tag: &str| {
        let mut p = std::env::temp_dir();
        p.push(format!("segbench-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    };
    g.bench_function("add", |b| {
        b.iter_batched(
            || mk("add"),
            |p| {
                let mut s = SegmentedStore::open_with_options(
                    FsDirectory::arc(&p).unwrap(),
                    Kv,
                    Options::new(256),
                )
                .unwrap();
                for i in 0..n {
                    s.add(i, format!("v{i}")).unwrap();
                }
                let _ = std::fs::remove_dir_all(&p);
            },
            BatchSize::PerIteration,
        );
    });
    g.bench_function("extend", |b| {
        b.iter_batched(
            || mk("extend"),
            |p| {
                let mut s = SegmentedStore::open_with_options(
                    FsDirectory::arc(&p).unwrap(),
                    Kv,
                    Options::new(256),
                )
                .unwrap();
                s.extend((0..n).map(|i| (i, format!("v{i}")))).unwrap();
                let _ = std::fs::remove_dir_all(&p);
            },
            BatchSize::PerIteration,
        );
    });
    g.finish();
}

// ---- realistic payload ----
//
// The Kv benches above carry a toy (u32, "vN") payload, which understates the
// cost the borrow-not-clone change removed: with a tiny payload, materializing
// a segment is dominated by per-String allocation, not bytes copied. A real
// consumer (sporse: SparseVec, vicinity: Vec<f32>) carries hundreds of bytes
// per item, where the eliminated full-segment clone is the dominant term.
// `Blob` models that with a 512-byte value per item.

const BLOB_BYTES: usize = 512;

struct Blob;
impl Store for Blob {
    type Id = u32;
    type Item = Vec<u8>;
    type Segment = Vec<(u32, Vec<u8>)>;
    fn build_segment(&self, batch: &[(u32, Vec<u8>)]) -> Vec<(u32, Vec<u8>)> {
        batch.to_vec()
    }
    fn merge_segments(
        &self,
        segs: &[&Vec<(u32, Vec<u8>)>],
        live: &dyn Fn(&u32) -> bool,
    ) -> Vec<(u32, Vec<u8>)> {
        segs.iter()
            .flat_map(|s| s.iter())
            .filter(|(id, _)| live(id))
            .cloned()
            .collect()
    }
    fn segment_len(&self, seg: &Vec<(u32, Vec<u8>)>) -> usize {
        seg.len()
    }
}

/// Direct A/B of the materialization the two perf commits changed, on a
/// consumer-realistic segment set (8 segments x 1000 items x 512 B ~= 4 MB).
/// `clone` is what `compact()` / `merge_group()` / `checkpoint()` did before
/// (build a `Vec<Segment>` by deep-cloning every payload); `borrow` is what they
/// do now (a `Vec<&Segment>` of pointers). Both arms run under identical machine
/// load, so the *ratio* isolates the eliminated overhead even when the absolute
/// compaction benches are contention-noisy.
fn bench_merge_input_materialization(c: &mut Criterion) {
    use std::sync::Arc;
    type Seg = Vec<(u32, Vec<u8>)>;
    let segs: Vec<Arc<Seg>> = (0..8u32)
        .map(|s| {
            Arc::new(
                (0..1000u32)
                    .map(|i| (s * 1000 + i, vec![0u8; BLOB_BYTES]))
                    .collect(),
            )
        })
        .collect();
    let mut g = c.benchmark_group("merge_input_materialization");
    g.bench_function("clone", |b| {
        b.iter(|| {
            let owned: Vec<Seg> = segs.iter().map(|a| (**a).clone()).collect();
            criterion::black_box(&owned);
        });
    });
    g.bench_function("borrow", |b| {
        b.iter(|| {
            let refs: Vec<&Seg> = segs.iter().map(|a| &**a).collect();
            criterion::black_box(&refs);
        });
    });
    g.finish();
}

/// Full compaction on the 512-byte payload (8 segments of ~250 items), where the
/// borrow-not-clone change has its largest relative effect on the whole op.
fn bench_compact_realistic(c: &mut Criterion) {
    c.bench_function("compact_full_blob_512B", |b| {
        b.iter_batched(
            || {
                let mut s = SegmentedStore::open_with_options(
                    MemoryDirectory::arc(),
                    Blob,
                    Options::new(256),
                )
                .unwrap();
                for i in 0..2_000u32 {
                    s.add(i, vec![0u8; BLOB_BYTES]).unwrap();
                }
                s
            },
            |mut s| {
                s.compact().unwrap();
            },
            BatchSize::SmallInput,
        );
    });
}

/// Per-op WAL durability cost: the default Flush (userspace -> OS, no fsync) vs
/// Fsync (fdatasync the WAL file AND its parent dir on every record). On a real
/// filesystem this exposes the fsync-barrier cost the strong policy pays. The
/// parent-dir fsync per append is redundant after file creation; this bench is what
/// shows the win when that redundancy is removed in the durability layer (the gap
/// should narrow toward `flush`).
fn bench_sync_policy(c: &mut Criterion) {
    use durability::FsDirectory;
    use segstore::SyncPolicy;
    let mut g = c.benchmark_group("sync_policy_fs_100");
    let n = 100u32; // Fsync per-op is slow on real disk; keep the count small.
    let mk = |tag: &str| {
        let mut p = std::env::temp_dir();
        p.push(format!("segbench-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    };
    for (label, sync) in [("flush", SyncPolicy::Flush), ("fsync", SyncPolicy::Fsync)] {
        g.bench_function(label, |b| {
            b.iter_batched(
                || mk(label),
                |p| {
                    let opts = Options {
                        sync,
                        ..Options::new(256)
                    };
                    let mut s =
                        SegmentedStore::open_with_options(FsDirectory::arc(&p).unwrap(), Kv, opts)
                            .unwrap();
                    for i in 0..n {
                        s.add(i, format!("v{i}")).unwrap();
                    }
                    let _ = std::fs::remove_dir_all(&p);
                },
                BatchSize::PerIteration,
            );
        });
    }
    g.finish();
}

/// Compaction with one consumer sidecar per input segment. This isolates the
/// sidecar-GC tax added by persisted built-index caches: the compact itself is
/// identical, but the `with_sidecars` arm also deletes stale `segstore.idx.*`
/// files for the segments that disappeared.
fn bench_sidecar_gc_on_compact(c: &mut Criterion) {
    let mut g = c.benchmark_group("sidecar_gc_compact_4k");
    for sidecars in [false, true] {
        let label = if sidecars {
            "with_sidecars"
        } else {
            "without_sidecars"
        };
        g.bench_function(label, |b| {
            b.iter_batched(
                || {
                    let dir = MemoryDirectory::arc();
                    let mut s =
                        SegmentedStore::open_with_options(dir.clone(), Kv, Options::new(64))
                            .unwrap();
                    fill(&mut s, 4_000);
                    s.checkpoint().unwrap();
                    if sidecars {
                        for &id in s.segment_ids() {
                            dir.atomic_write(&s.index_name(id, "bench"), b"built-index")
                                .unwrap();
                        }
                    }
                    s
                },
                |mut s| {
                    s.compact().unwrap();
                },
                BatchSize::SmallInput,
            );
        });
    }
    g.finish();
}

/// Same sidecar-GC shape on a real filesystem. The in-memory bench above
/// isolates orchestration cost; this one tracks the directory-sync cost of
/// deleting stale segment/index files after the manifest commit.
fn bench_sidecar_gc_on_compact_fs(c: &mut Criterion) {
    use durability::FsDirectory;
    let mut g = c.benchmark_group("sidecar_gc_compact_fs_512");
    let mk = |tag: &str| {
        let mut p = std::env::temp_dir();
        p.push(format!("segbench-sidecar-gc-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    };
    for sidecars in [false, true] {
        let label = if sidecars {
            "with_sidecars"
        } else {
            "without_sidecars"
        };
        g.bench_function(label, |b| {
            b.iter_batched(
                || {
                    let p = mk(label);
                    let dir = FsDirectory::arc(&p).unwrap();
                    let mut s =
                        SegmentedStore::open_with_options(dir.clone(), Kv, Options::new(64))
                            .unwrap();
                    fill(&mut s, 512);
                    s.checkpoint().unwrap();
                    if sidecars {
                        for &id in s.segment_ids() {
                            dir.atomic_write(&s.index_name(id, "bench"), b"built-index")
                                .unwrap();
                        }
                    }
                    (p, s)
                },
                |(p, mut s)| {
                    s.compact().unwrap();
                    let _ = std::fs::remove_dir_all(&p);
                },
                BatchSize::PerIteration,
            );
        });
    }
    g.finish();
}

/// Recovery (reopen) cost on the 512-byte payload, the dimension where zero-copy
/// would matter: `open` postcard-DECODES every segment file (O(total payload)),
/// while queries are in-memory and never decode. Contrast the toy `recover` group
/// (tiny strings) to see decode scale with payload size, not just item count.
fn bench_recovery_blob(c: &mut Criterion) {
    let mut g = c.benchmark_group("recover_blob_512B");
    for &n in &[1_000u32, 4_000] {
        let dir = MemoryDirectory::arc();
        {
            let mut s =
                SegmentedStore::open_with_options(dir.clone(), Blob, Options::new(256)).unwrap();
            for i in 0..n {
                s.add(i, vec![0u8; BLOB_BYTES]).unwrap();
            }
            s.checkpoint().unwrap();
        }
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let s = SegmentedStore::open_with_options(dir.clone(), Blob, Options::new(256))
                    .unwrap();
                criterion::black_box(s.segment_count());
            });
        });
    }
    g.finish();
}

/// Open only the checkpoint manifest catalog vs fully opening the store and
/// decoding every segment. This is the narrow performance claim behind
/// `SegmentCatalog`: it is useful for diagnostics and loader planning, but it is
/// not yet a byte-native query reader.
fn bench_segment_catalog_open(c: &mut Criterion) {
    let mut g = c.benchmark_group("open_catalog_vs_full_blob_512B");
    for &n in &[1_000u32, 4_000] {
        let dir = MemoryDirectory::arc();
        {
            let mut s =
                SegmentedStore::open_with_options(dir.clone(), Blob, Options::new(256)).unwrap();
            for i in 0..n {
                s.add(i, vec![0u8; BLOB_BYTES]).unwrap();
            }
            s.checkpoint().unwrap();
        }
        g.bench_with_input(BenchmarkId::new("full_open", n), &n, |b, _| {
            b.iter(|| {
                let s = SegmentedStore::open_with_options(dir.clone(), Blob, Options::new(256))
                    .unwrap();
                criterion::black_box(s.segment_count());
            });
        });
        g.bench_with_input(BenchmarkId::new("catalog_open", n), &n, |b, _| {
            b.iter(|| {
                let catalog = SegmentCatalog::<u32>::open(dir.clone()).unwrap();
                criterion::black_box(catalog.segment_count());
            });
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_add,
    bench_compact_tiers,
    bench_full_compact,
    bench_recovery,
    bench_ingest_fs,
    bench_merge_input_materialization,
    bench_compact_realistic,
    bench_sync_policy,
    bench_sidecar_gc_on_compact,
    bench_sidecar_gc_on_compact_fs,
    bench_recovery_blob,
    bench_segment_catalog_open
);
criterion_main!(benches);
