//! Hot-path benchmarks: ingest (WAL append + flush per op), size-tiered
//! compaction drain, full compaction, and recovery (WAL replay on open).
//!
//! Run: `cargo bench`. Backend is in-memory, so these isolate segstore's own
//! overhead (op orchestration, bucketing, segment-list manipulation) from real
//! disk IO -- the relative costs and any O(n^2) surprises are what matter here.

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use durability::MemoryDirectory;
use segstore::{Options, SegmentedStore, Store};

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

criterion_group!(
    benches,
    bench_add,
    bench_compact_tiers,
    bench_full_compact,
    bench_recovery
);
criterion_main!(benches);
