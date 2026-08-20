use std::cmp::Ordering;
use std::time::Instant;

use mlx_native::MlxBuffer;

use super::cache::Deepseek4Cache;
use super::model::Deepseek4Model;
use super::real_artifact_tests::official_artifact;
use super::verifier_forward::MAX_COOPERATIVE_PREFILL_ROWS;

// The protected agent anchor is 6,676 tokens: mod-128 = 20 and mod-4 = 0.
// A 148-token prefix reproduces both compressor remainders while keeping the
// hardware parity test's cache footprint compact.
const PREFIX_ROWS: usize = 148;
const COHORT_SHAPES: &[(usize, usize)] = &[(2, 1_024), (3, 640), (4, 512)];
const BENCHMARK_OVERRIDE_ENV: &[&str] = &[
    "HF2Q_DEEPSEEK_COMPRESSED_STAGE_PROFILE",
    "HF2Q_DEEPSEEK_DUMP_ATTENTION_DIR",
    "HF2Q_DEEPSEEK_DUMP_LAYER_DIR",
    "HF2Q_DEEPSEEK_ENCODER_STAGES",
    "HF2Q_DEEPSEEK_GRAPH_DIAG",
    "HF2Q_DEEPSEEK_GRAPH_LAYERS_PER_CB",
    "HF2Q_DEEPSEEK_GRAPH_REORDER",
    "HF2Q_DEEPSEEK_LAYERS_PER_CB",
    "HF2Q_DEEPSEEK_MMAP_WEIGHTS",
    "HF2Q_DEEPSEEK_PREFILL_TIMING",
    "HF2Q_DEEPSEEK_PREFILL_WINDOWS",
    "HF2Q_DEEPSEEK_STAGE_PROFILE",
    "HF2Q_MM_ID_ROUTING_THRESHOLD",
    "MLX_PROFILE_CB",
    "MLX_UNRETAINED_REFS",
];

struct LaneReceipt {
    prefill_state: Vec<f32>,
    logits: Vec<f32>,
    greedy_token: u32,
    decode_state: Vec<f32>,
    decode_greedy_token: u32,
    final_position: usize,
}

fn token_batch(sequence: usize, phase: usize, rows: usize) -> Vec<u32> {
    (0..rows)
        .map(|row| ((row * 97 + sequence * 193 + phase * 389 + 11) % 120_000) as u32)
        .collect()
}

fn read_f32(buffer: &MlxBuffer, label: &str) -> Vec<f32> {
    buffer
        .as_logical_slice::<f32>()
        .unwrap_or_else(|error| panic!("read {label}: {error}"))
        .to_vec()
}

fn receipt_after_prefill(
    model: &mut Deepseek4Model,
    cache: &mut Deepseek4Cache,
    state: MlxBuffer,
) -> LaneReceipt {
    let prefill_state = read_f32(&state, "prefill state");
    let last = model
        .last_token_state(&state)
        .expect("view last prefill row");
    let logits_buffer = model.forward_logits(&last).expect("project prefill logits");
    let logits = read_f32(&logits_buffer, "prefill logits");
    let greedy_token = model
        .greedy_token(&logits_buffer)
        .expect("select prefill greedy token");
    let decode = model
        .forward_verifier_one(greedy_token, cache)
        .expect("decode from prefilled cache");
    let decode_state = read_f32(&decode, "post-prefill decode state");
    let decode_logits = model
        .forward_logits(&decode)
        .expect("project post-prefill decode logits");
    let decode_greedy_token = model
        .greedy_token(&decode_logits)
        .expect("select post-prefill greedy token");
    LaneReceipt {
        prefill_state,
        logits,
        greedy_token,
        decode_state,
        decode_greedy_token,
        final_position: cache.position(),
    }
}

fn assert_exact_f32(label: &str, expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len(), "{label} length differs");
    let mismatch = expected
        .iter()
        .zip(actual)
        .enumerate()
        .find(|(_, (expected, actual))| expected.to_bits() != actual.to_bits())
        .map(|(index, (expected, actual))| (index, *expected, *actual));
    assert!(mismatch.is_none(), "{label} differs: {mismatch:?}");
}

fn assert_lane_receipt_exact(sequence: usize, serial: &LaneReceipt, cohort: &LaneReceipt) {
    assert_exact_f32(
        &format!("sequence {sequence} prefill state"),
        &serial.prefill_state,
        &cohort.prefill_state,
    );
    assert_exact_f32(
        &format!("sequence {sequence} prefill logits"),
        &serial.logits,
        &cohort.logits,
    );
    assert_eq!(
        serial.greedy_token, cohort.greedy_token,
        "sequence {sequence} prefill greedy token differs"
    );
    assert_exact_f32(
        &format!("sequence {sequence} post-prefill decode state"),
        &serial.decode_state,
        &cohort.decode_state,
    );
    assert_eq!(
        serial.decode_greedy_token, cohort.decode_greedy_token,
        "sequence {sequence} post-prefill greedy token differs"
    );
    assert_eq!(
        serial.final_position, cohort.final_position,
        "sequence {sequence} final cache cursor differs"
    );
}

fn prepare_warm_caches(
    model: &mut Deepseek4Model,
    caches: &mut [Deepseek4Cache],
    prefixes: &[Vec<u32>],
) {
    for (cache, prefix) in caches.iter_mut().zip(prefixes) {
        cache.reset().expect("reset benchmark cache");
        model
            .forward_verifier_prefill(prefix, cache)
            .expect("install benchmark warm prefix");
        assert_eq!(cache.position(), PREFIX_ROWS);
    }
}

fn timed_serial(
    model: &mut Deepseek4Model,
    caches: &mut [Deepseek4Cache],
    suffixes: &[Vec<u32>],
) -> f64 {
    let started = Instant::now();
    for (cache, suffix) in caches.iter_mut().zip(suffixes) {
        let _ = model
            .forward_verifier_prefill(suffix, cache)
            .expect("timed serial warm prefill");
    }
    started.elapsed().as_secs_f64() * 1_000.0
}

fn timed_cohort(
    model: &mut Deepseek4Model,
    caches: &mut [Deepseek4Cache],
    suffixes: &[Vec<u32>],
) -> f64 {
    let suffix_refs = suffixes.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let mut cache_refs = caches.iter_mut().collect::<Vec<_>>();
    let started = Instant::now();
    let _ = model
        .forward_verifier_prefill_cohort(&suffix_refs, &mut cache_refs)
        .expect("timed cooperative warm prefill");
    started.elapsed().as_secs_f64() * 1_000.0
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    values[values.len() / 2]
}

#[cfg(target_os = "macos")]
fn process_peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: getrusage initializes the supplied rusage structure for the
    // current process. The return code is checked before assume_init.
    let code = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    assert_eq!(code, 0, "getrusage failed with errno {code}");
    // SAFETY: a zero return from getrusage guarantees initialization.
    let usage = unsafe { usage.assume_init() };
    u64::try_from(usage.ru_maxrss).expect("macOS peak RSS must be nonnegative")
}

#[cfg(not(target_os = "macos"))]
fn process_peak_rss_bytes() -> u64 {
    0
}

#[test]
#[ignore = "loads the release checkpoint and proves production cooperative warm prefill"]
fn official_artifact_cooperative_warm_prefill_is_exact_and_faster() {
    let allowed_benchmark_variables = [
        "HF2Q_DEEPSEEK4_COHORT_BENCH_PAIRS",
        "HF2Q_DEEPSEEK4_COHORT_RECEIPT",
        "HF2Q_DEEPSEEK4_GGUF",
    ];
    let unexpected_benchmark_variables = std::env::vars_os()
        .filter_map(|(name, _)| name.into_string().ok())
        .filter(|name| {
            (name.starts_with("HF2Q_")
                || name.starts_with("MLX_")
                || name.starts_with("METAL_")
                || name.starts_with("MTL_"))
                && !allowed_benchmark_variables.contains(&name.as_str())
        })
        .collect::<Vec<_>>();
    assert!(
        unexpected_benchmark_variables.is_empty(),
        "release benchmark requires a clean hf2q/MLX/Metal environment; unexpected variables: {unexpected_benchmark_variables:?}"
    );
    for name in BENCHMARK_OVERRIDE_ENV {
        assert!(
            std::env::var_os(name).is_none(),
            "release benchmark requires the canonical default environment; unset {name}"
        );
    }
    let pairs = std::env::var("HF2Q_DEEPSEEK4_COHORT_BENCH_PAIRS")
        .ok()
        .map(|value| value.parse::<usize>().expect("parse benchmark pair count"))
        .unwrap_or(5);
    assert_eq!(pairs, 5, "release benchmark requires exactly five pairs");

    let (path, gguf) = official_artifact();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf)
        .unwrap_or_else(|error| panic!("load official artifact {}: {error:#}", path.display()));

    for &(sequences, rows) in COHORT_SHAPES {
        assert!(
            sequences * rows <= MAX_COOPERATIVE_PREFILL_ROWS,
            "parity shape exceeds the production aggregate-row bound"
        );
        let prefixes = (0..sequences)
            .map(|sequence| token_batch(sequence, 0, PREFIX_ROWS))
            .collect::<Vec<_>>();
        let suffixes = (0..sequences)
            .map(|sequence| token_batch(sequence, 1, rows))
            .collect::<Vec<_>>();
        let capacity = PREFIX_ROWS + rows + 1;

        let serial = prefixes
            .iter()
            .zip(&suffixes)
            .map(|(prefix, suffix)| {
                let mut cache = model
                    .allocate_cache(capacity)
                    .expect("allocate serial cache");
                model
                    .forward_verifier_prefill(prefix, &mut cache)
                    .expect("install serial warm prefix");
                assert_eq!(cache.position(), PREFIX_ROWS);
                let state = model
                    .forward_verifier_prefill(suffix, &mut cache)
                    .expect("serial warm suffix");
                receipt_after_prefill(&mut model, &mut cache, state)
            })
            .collect::<Vec<_>>();

        let mut cohort_caches = (0..sequences)
            .map(|_| {
                model
                    .allocate_cache(capacity)
                    .expect("allocate cohort cache")
            })
            .collect::<Vec<_>>();
        prepare_warm_caches(&mut model, &mut cohort_caches, &prefixes);
        let suffix_refs = suffixes.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let mut cache_refs = cohort_caches.iter_mut().collect::<Vec<_>>();
        let states = model
            .forward_verifier_prefill_cohort(&suffix_refs, &mut cache_refs)
            .expect("cooperative warm suffix");
        let cohort = states
            .into_iter()
            .zip(cohort_caches.iter_mut())
            .map(|(state, cache)| receipt_after_prefill(&mut model, cache, state))
            .collect::<Vec<_>>();

        for (sequence, (serial, cohort)) in serial.iter().zip(&cohort).enumerate() {
            assert_lane_receipt_exact(sequence, serial, cohort);
            assert_eq!(serial.final_position, PREFIX_ROWS + rows + 1);
        }
        eprintln!(
            "DeepSeek-V4 cooperative parity: sequences={sequences} prefix={PREFIX_ROWS} rows={rows} aggregate_rows={} layers=43 exact_state_logits_decode=true",
            sequences * rows
        );
    }

    let (sequences, rows) = COHORT_SHAPES[COHORT_SHAPES.len() - 1];
    let prefixes = (0..sequences)
        .map(|sequence| token_batch(sequence, 0, PREFIX_ROWS))
        .collect::<Vec<_>>();
    let suffixes = (0..sequences)
        .map(|sequence| token_batch(sequence, 1, rows))
        .collect::<Vec<_>>();
    let capacity = PREFIX_ROWS + rows + 1;
    let mut serial_caches = (0..sequences)
        .map(|_| {
            model
                .allocate_cache(capacity)
                .expect("allocate serial benchmark cache")
        })
        .collect::<Vec<_>>();
    let mut cohort_caches = (0..sequences)
        .map(|_| {
            model
                .allocate_cache(capacity)
                .expect("allocate cohort benchmark cache")
        })
        .collect::<Vec<_>>();

    prepare_warm_caches(&mut model, &mut serial_caches, &prefixes);
    let _ = timed_serial(&mut model, &mut serial_caches, &suffixes);
    prepare_warm_caches(&mut model, &mut cohort_caches, &prefixes);
    let _ = timed_cohort(&mut model, &mut cohort_caches, &suffixes);

    let mut serial_ms = Vec::with_capacity(pairs);
    let mut cohort_ms = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        if pair % 2 == 0 {
            prepare_warm_caches(&mut model, &mut serial_caches, &prefixes);
            serial_ms.push(timed_serial(&mut model, &mut serial_caches, &suffixes));
            prepare_warm_caches(&mut model, &mut cohort_caches, &prefixes);
            cohort_ms.push(timed_cohort(&mut model, &mut cohort_caches, &suffixes));
        } else {
            prepare_warm_caches(&mut model, &mut cohort_caches, &prefixes);
            cohort_ms.push(timed_cohort(&mut model, &mut cohort_caches, &suffixes));
            prepare_warm_caches(&mut model, &mut serial_caches, &prefixes);
            serial_ms.push(timed_serial(&mut model, &mut serial_caches, &suffixes));
        }
    }
    let serial_median = median(serial_ms.clone());
    let cohort_median = median(cohort_ms.clone());
    let speedup = serial_median / cohort_median;
    let peak_rss_bytes = process_peak_rss_bytes();
    eprintln!(
        "DeepSeek-V4 cooperative benchmark: sequences={sequences} prefix={PREFIX_ROWS} rows={rows} aggregate_rows={} pairs={pairs} order=alternating serial_ms={serial_ms:?} cohort_ms={cohort_ms:?} serial_median_ms={serial_median:.3} cohort_median_ms={cohort_median:.3} speedup={speedup:.4}x process_peak_rss_bytes={peak_rss_bytes}",
        sequences * rows
    );
    assert!(
        cohort_median < serial_median,
        "cooperative median must beat serial: serial={serial_median:.3}ms cohort={cohort_median:.3}ms"
    );
    if let Some(receipt_path) = std::env::var_os("HF2Q_DEEPSEEK4_COHORT_RECEIPT") {
        let receipt = serde_json::json!({
            "schema_version": 1,
            "status": "pass",
            "artifact_bytes": std::fs::metadata(&path)
                .expect("stat official artifact")
                .len(),
            "layers": 43,
            "prefix_rows": PREFIX_ROWS,
            "prefix_mod_128": PREFIX_ROWS % 128,
            "prefix_mod_4": PREFIX_ROWS % 4,
            "parity_shapes": COHORT_SHAPES
                .iter()
                .map(|(sequences, rows)| serde_json::json!({
                    "sequences": sequences,
                    "rows_per_lane": rows,
                    "aggregate_rows": sequences * rows,
                    "exact_state_logits_decode": true
                }))
                .collect::<Vec<_>>(),
            "benchmark": {
                "sequences": sequences,
                "rows_per_lane": rows,
                "aggregate_rows": sequences * rows,
                "pairs": pairs,
                "order": "alternating",
                "serial_ms": serial_ms,
                "cohort_ms": cohort_ms,
                "serial_median_ms": serial_median,
                "cohort_median_ms": cohort_median,
                "speedup": speedup,
                "process_lifetime_peak_rss_bytes": peak_rss_bytes
            },
            "benchmark_environment": {
                "profile": "clean-hf2q-mlx-metal-v1",
                "override_variables_absent": true,
                "unexpected_override_variables": unexpected_benchmark_variables,
                "pairs": pairs
            }
        });
        std::fs::write(
            receipt_path,
            serde_json::to_vec_pretty(&receipt).expect("serialize cooperative receipt"),
        )
        .expect("write cooperative receipt");
    }
}
