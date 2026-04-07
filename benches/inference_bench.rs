//! Inference benchmark harness for hf2q.
//!
//! Measures harness overhead and validates the benchmark report JSON schema.
//! For real model benchmarks, use `scripts/benchmark.sh` which invokes the
//! `hf2q` binary directly.
//!
//! ## Usage
//!
//! Synthetic mode (harness overhead):
//!   cargo bench --bench inference_bench
//!
//! Real model benchmarks:
//!   scripts/benchmark.sh --model /path/to/model

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use serde::Serialize;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Output schema (JSON) -- must match scripts/benchmark.sh output
// ---------------------------------------------------------------------------

/// Full benchmark report.
#[derive(Debug, Serialize)]
struct BenchmarkReport {
    tool: String,
    timestamp: String,
    model: String,
    synthetic: bool,
    model_load_time_secs: Option<f64>,
    results: Vec<PromptLengthResult>,
    methodology: MethodologyNotes,
}

/// Results for a single prompt length.
#[derive(Debug, Serialize)]
struct PromptLengthResult {
    prompt_length: usize,
    actual_prompt_tokens: usize,
    decode_tok_per_sec: MedianStat,
    prefill_tok_per_sec: MedianStat,
    ttft_ms: MedianStat,
    peak_memory_bytes: Option<usize>,
    generated_tokens: MedianStat,
    prompt_cache_active: bool,
    cached_tokens_last_run: usize,
}

/// Median statistic with min/max range.
#[derive(Debug, Serialize)]
struct MedianStat {
    median: f64,
    min: f64,
    max: f64,
}

impl MedianStat {
    fn from_values(mut values: Vec<f64>) -> Self {
        assert!(!values.is_empty());
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let len = values.len();
        let median = if len % 2 == 0 {
            (values[len / 2 - 1] + values[len / 2]) / 2.0
        } else {
            values[len / 2]
        };
        Self {
            median,
            min: values[0],
            max: values[len - 1],
        }
    }
}

/// Methodology notes embedded in the report.
#[derive(Debug, Serialize)]
struct MethodologyNotes {
    warm_up_runs: usize,
    measurement_runs: usize,
    statistic: String,
    max_tokens_per_run: usize,
    prompt_generation: String,
    notes: Vec<String>,
}

// ---------------------------------------------------------------------------
// Synthetic prompt generation
// ---------------------------------------------------------------------------

/// Generate a deterministic synthetic prompt of approximately `target_tokens`.
fn generate_synthetic_prompt(target_tokens: usize) -> String {
    let words = [
        "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy",
        "dog", "and", "then", "runs", "through", "the", "forest", "where",
        "many", "trees", "grow", "tall", "under", "the", "bright", "sun",
        "that", "shines", "down", "upon", "the", "green", "meadow", "below",
    ];

    let word_count = target_tokens + target_tokens / 10;
    let mut prompt = String::with_capacity(word_count * 6);
    for i in 0..word_count {
        if i > 0 {
            prompt.push(' ');
        }
        prompt.push_str(words[i % words.len()]);
    }
    prompt
}

// ---------------------------------------------------------------------------
// Prefix matching benchmark (measures prompt cache overhead)
// ---------------------------------------------------------------------------

/// Simulates the prefix matching algorithm from prompt_cache.rs.
/// This benchmarks the raw comparison cost.
fn find_prefix_match(cached: &[u32], new: &[u32]) -> usize {
    let limit = cached.len().min(new.len());
    let mut i = 0;
    while i < limit && cached[i] == new[i] {
        i += 1;
    }
    i
}

fn bench_prefix_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_matching");

    for &size in &[100, 1000, 4000, 8000] {
        // Full match (worst case for comparison)
        let cached: Vec<u32> = (0..size).collect();
        let new_tokens: Vec<u32> = (0..size + 50).collect();

        group.bench_with_input(
            BenchmarkId::new("full_match", size),
            &(cached.clone(), new_tokens.clone()),
            |b, (cached, new)| {
                b.iter(|| find_prefix_match(cached, new));
            },
        );

        // No match (best case)
        let cached_no_match: Vec<u32> = (0..size).collect();
        let new_no_match: Vec<u32> = (size as u32..size as u32 * 2).collect();

        group.bench_with_input(
            BenchmarkId::new("no_match", size),
            &(cached_no_match, new_no_match),
            |b, (cached, new)| {
                b.iter(|| find_prefix_match(cached, new));
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Synthetic prompt generation benchmark
// ---------------------------------------------------------------------------

fn bench_prompt_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("prompt_generation");

    for &size in &[20, 256, 1024, 4096] {
        group.bench_with_input(
            BenchmarkId::new("generate", size),
            &size,
            |b, &size| {
                b.iter(|| generate_synthetic_prompt(size));
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// JSON report schema validation
// ---------------------------------------------------------------------------

fn bench_report_serialization(c: &mut Criterion) {
    let report = BenchmarkReport {
        tool: "hf2q".to_string(),
        timestamp: "test".to_string(),
        model: "synthetic".to_string(),
        synthetic: true,
        model_load_time_secs: None,
        results: vec![
            PromptLengthResult {
                prompt_length: 20,
                actual_prompt_tokens: 22,
                decode_tok_per_sec: MedianStat::from_values(vec![100.0, 110.0, 105.0]),
                prefill_tok_per_sec: MedianStat::from_values(vec![500.0, 520.0, 510.0]),
                ttft_ms: MedianStat::from_values(vec![5.0, 4.5, 4.8]),
                peak_memory_bytes: None,
                generated_tokens: MedianStat::from_values(vec![64.0, 64.0, 64.0]),
                prompt_cache_active: false,
                cached_tokens_last_run: 0,
            },
            PromptLengthResult {
                prompt_length: 256,
                actual_prompt_tokens: 280,
                decode_tok_per_sec: MedianStat::from_values(vec![95.0, 100.0, 98.0]),
                prefill_tok_per_sec: MedianStat::from_values(vec![450.0, 460.0, 455.0]),
                ttft_ms: MedianStat::from_values(vec![50.0, 48.0, 52.0]),
                peak_memory_bytes: Some(1024 * 1024 * 500),
                generated_tokens: MedianStat::from_values(vec![128.0, 128.0, 128.0]),
                prompt_cache_active: true,
                cached_tokens_last_run: 200,
            },
        ],
        methodology: MethodologyNotes {
            warm_up_runs: 2,
            measurement_runs: 5,
            statistic: "median of N runs with min/max range".to_string(),
            max_tokens_per_run: 128,
            prompt_generation: "Deterministic synthetic prompts".to_string(),
            notes: vec!["Test note".to_string()],
        },
    };

    c.bench_function("report_serialize", |b| {
        b.iter(|| serde_json::to_string_pretty(&report).unwrap());
    });
}

criterion_group!(
    benches,
    bench_prefix_matching,
    bench_prompt_generation,
    bench_report_serialization
);
criterion_main!(benches);
