use criterion::{criterion_group, criterion_main, Criterion};

fn quantize_benchmark(_c: &mut Criterion) {
    // Benchmark placeholder — will be implemented with real quantization benchmarks
}

criterion_group!(benches, quantize_benchmark);
criterion_main!(benches);
