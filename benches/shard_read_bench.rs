use criterion::{criterion_group, criterion_main, Criterion};

fn shard_read_benchmark(_c: &mut Criterion) {
    // Benchmark placeholder — will be implemented with real shard reading benchmarks
}

criterion_group!(benches, shard_read_benchmark);
criterion_main!(benches);
