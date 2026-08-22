use std::cmp::Ordering;
use std::path::PathBuf;
use std::time::Instant;

use mlx_native::{
    barrier_count, cmd_buf_count, dispatch_count, reset_counters, sync_count, MlxBuffer,
};

use super::cache::{Deepseek4Cache, Deepseek4CacheSnapshot};
use super::model::Deepseek4Model;
use super::real_artifact_tests::official_artifact;

const LANES: usize = 4;
const PREFIX_ROWS: usize = 148;
const DECODE_STEPS: usize = 132;
const PARITY_CAPACITY: usize = PREFIX_ROWS + DECODE_STEPS;
const BENCHMARK_POSITION: usize = 6_676;
const BENCHMARK_LOGICAL_CAPACITY: usize = 131_072;
// Gives the hardware runner enough room to require 30 continuously nominal
// seconds after the model and production-capacity caches are resident.
const LOADED_IDLE_SECONDS: u64 = 45;
const BENCHMARK_PAIRS: usize = 10;
const RECEIPT_ENV: &str = "HF2Q_DEEPSEEK4_DECODE_COHORT_RECEIPT";
const PHYSICAL_TO_LOGICAL: [usize; LANES] = [2, 0, 3, 1];
const BENCHMARK_OVERRIDE_ENV: &[&str] = &[
    "HF2Q_DEEPSEEK_COMPRESSED_STAGE_PROFILE",
    "HF2Q_DEEPSEEK_DUMP_ATTENTION_DIR",
    "HF2Q_DEEPSEEK_DUMP_LAYER_DIR",
    "HF2Q_DEEPSEEK_ENCODER_STAGES",
    "HF2Q_DEEPSEEK_GRAPH_DIAG",
    "HF2Q_DEEPSEEK_LAYERS_PER_CB",
    "HF2Q_DEEPSEEK_STAGE_PROFILE",
    "HF2Q_MM_ID_ROUTING_THRESHOLD",
    "MLX_PROFILE_CB",
    "MLX_PROFILE_DISPATCH",
    "MLX_UNRETAINED_REFS",
];

#[derive(Clone, Copy, Debug)]
struct TrialCounters {
    command_buffers: u64,
    synchronizations: u64,
    dispatches: u64,
    barriers: u64,
}

fn prefix_tokens(logical_lane: usize) -> Vec<u32> {
    (0..PREFIX_ROWS)
        .map(|row| ((row * 97 + logical_lane * 193 + 11) % 120_000) as u32)
        .collect()
}

fn supplied_tokens(step: usize) -> [u32; LANES] {
    std::array::from_fn(|logical_lane| ((step * 389 + logical_lane * 7_919 + 17) % 120_000) as u32)
}

fn benchmark_extension(logical_lane: usize) -> Vec<u32> {
    (PARITY_CAPACITY..BENCHMARK_POSITION)
        .map(|position| ((position * 521 + logical_lane * 7_919 + 29) % 120_000) as u32)
        .collect()
}

fn grow_and_extend_caches(
    model: &mut Deepseek4Model,
    caches: Vec<Deepseek4Cache>,
    physical_to_logical: [usize; LANES],
) -> Vec<Deepseek4Cache> {
    caches
        .into_iter()
        .enumerate()
        .map(|(physical_lane, source)| {
            let logical_lane = physical_to_logical[physical_lane];
            let mut grown = model
                .allocate_logical_cache(BENCHMARK_LOGICAL_CAPACITY)
                .expect("allocate production-capacity benchmark cache");
            grown
                .migrate_from(&source, None)
                .expect("grow parity cache into production logical capacity");
            model
                .forward_verifier_prompt(&benchmark_extension(logical_lane), &mut grown)
                .expect("extend benchmark cache to production anchor");
            assert_eq!(grown.position(), BENCHMARK_POSITION);
            assert_eq!(grown.capacity(), BENCHMARK_LOGICAL_CAPACITY);
            grown
        })
        .collect()
}

fn read_f32(buffer: &MlxBuffer, label: &str) -> Vec<f32> {
    buffer
        .as_logical_slice::<f32>()
        .unwrap_or_else(|error| panic!("read {label}: {error}"))
        .to_vec()
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

fn valid_row_bytes(buffer: &MlxBuffer, rows: usize, label: &str) -> usize {
    let row_elements = buffer.shape()[1..]
        .iter()
        .try_fold(1_usize, |total, &dimension| total.checked_mul(dimension))
        .unwrap_or_else(|| panic!("{label} row size overflow"));
    rows.checked_mul(row_elements)
        .and_then(|elements| elements.checked_mul(buffer.dtype().size_of()))
        .unwrap_or_else(|| panic!("{label} byte size overflow"))
}

fn assert_buffer_prefix_exact(
    label: &str,
    expected: &MlxBuffer,
    actual: &MlxBuffer,
    valid_bytes: usize,
) {
    assert_eq!(expected.dtype(), actual.dtype(), "{label} dtype differs");
    assert_eq!(expected.shape(), actual.shape(), "{label} shape differs");
    assert!(
        valid_bytes <= expected.data_byte_len() && valid_bytes <= actual.data_byte_len(),
        "{label} valid byte bound exceeds allocation"
    );
    let expected = &expected
        .as_logical_slice::<u8>()
        .unwrap_or_else(|error| panic!("read expected {label}: {error}"))[..valid_bytes];
    let actual = &actual
        .as_logical_slice::<u8>()
        .unwrap_or_else(|error| panic!("read actual {label}: {error}"))[..valid_bytes];
    let mismatch = expected
        .iter()
        .zip(actual)
        .position(|(expected, actual)| expected != actual);
    assert!(mismatch.is_none(), "{label} differs at byte {mismatch:?}");
}

fn assert_optional_buffer_exact(
    label: &str,
    expected: Option<&MlxBuffer>,
    actual: Option<&MlxBuffer>,
    valid_bytes: usize,
) {
    match (expected, actual) {
        (Some(expected), Some(actual)) => {
            assert_buffer_prefix_exact(label, expected, actual, valid_bytes)
        }
        (None, None) => assert_eq!(valid_bytes, 0, "{label} unexpectedly has valid bytes"),
        _ => panic!("{label} optional-buffer presence differs"),
    }
}

fn assert_cache_exact(label: &str, expected: &Deepseek4Cache, actual: &Deepseek4Cache) {
    assert_eq!(
        expected.position(),
        actual.position(),
        "{label} cursor differs"
    );
    assert_eq!(
        expected.capacity(),
        actual.capacity(),
        "{label} capacity differs"
    );
    assert_eq!(expected.plan, actual.plan, "{label} plan differs");
    assert_eq!(
        expected.is_poisoned(),
        actual.is_poisoned(),
        "{label} poison state differs"
    );

    let position = expected.position();
    for (layer_index, ((layer_plan, expected), actual)) in expected
        .plan
        .layers
        .iter()
        .zip(expected.layers())
        .zip(actual.layers())
        .enumerate()
    {
        let window_rows = position.min(expected.window_kv.shape()[0]);
        assert_buffer_prefix_exact(
            &format!("{label} layer {layer_index} window"),
            &expected.window_kv,
            &actual.window_kv,
            valid_row_bytes(&expected.window_kv, window_rows, "window"),
        );
        let compressed_rows = if layer_plan.compress_ratio == 0 {
            0
        } else {
            position / layer_plan.compress_ratio as usize
        };
        let compressed_bytes = expected.compressed_kv.as_ref().map_or(0, |buffer| {
            valid_row_bytes(buffer, compressed_rows, "compressed KV")
        });
        assert_optional_buffer_exact(
            &format!("{label} layer {layer_index} compressed"),
            expected.compressed_kv.as_ref(),
            actual.compressed_kv.as_ref(),
            compressed_bytes,
        );
        let indexer_rows = if layer_plan.compress_ratio == 4 {
            compressed_rows
        } else {
            0
        };
        let indexer_bytes = expected.indexer_kv.as_ref().map_or(0, |buffer| {
            valid_row_bytes(buffer, indexer_rows, "indexer KV")
        });
        assert_optional_buffer_exact(
            &format!("{label} layer {layer_index} indexer"),
            expected.indexer_kv.as_ref(),
            actual.indexer_kv.as_ref(),
            indexer_bytes,
        );
        for (kind, expected, actual) in [
            (
                "main KV state",
                expected.main_kv_state.as_ref(),
                actual.main_kv_state.as_ref(),
            ),
            (
                "main score state",
                expected.main_score_state.as_ref(),
                actual.main_score_state.as_ref(),
            ),
            (
                "indexer KV state",
                expected.indexer_kv_state.as_ref(),
                actual.indexer_kv_state.as_ref(),
            ),
            (
                "indexer score state",
                expected.indexer_score_state.as_ref(),
                actual.indexer_score_state.as_ref(),
            ),
        ] {
            let valid_bytes = expected.map_or(0, MlxBuffer::data_byte_len);
            assert_optional_buffer_exact(
                &format!("{label} layer {layer_index} {kind}"),
                expected,
                actual,
                valid_bytes,
            );
        }
    }
}

fn counter_snapshot() -> TrialCounters {
    TrialCounters {
        command_buffers: cmd_buf_count(),
        synchronizations: sync_count(),
        dispatches: dispatch_count(),
        barriers: barrier_count(),
    }
}

fn timed_serial(
    model: &mut Deepseek4Model,
    caches: &mut [Deepseek4Cache],
    tokens: [u32; LANES],
) -> (f64, TrialCounters) {
    reset_counters();
    let started = Instant::now();
    for logical_lane in 0..LANES {
        let state = model
            .forward_verifier_one(tokens[logical_lane], &mut caches[logical_lane])
            .expect("timed serial decode body");
        let _ = model
            .forward_logits(&state)
            .expect("timed serial output head");
    }
    (
        started.elapsed().as_secs_f64() * 1_000.0,
        counter_snapshot(),
    )
}

fn timed_cohort(
    model: &mut Deepseek4Model,
    caches: &mut [Deepseek4Cache],
    logical_tokens: [u32; LANES],
) -> (f64, TrialCounters) {
    let physical_tokens = PHYSICAL_TO_LOGICAL.map(|logical_lane| logical_tokens[logical_lane]);
    let [lane0, lane1, lane2, lane3] = caches else {
        panic!("B=4 benchmark cache count drift")
    };
    let mut cache_refs = [lane0, lane1, lane2, lane3];
    reset_counters();
    let started = Instant::now();
    let state = model
        .forward_verifier_decode_cohort(physical_tokens, &mut cache_refs)
        .expect("timed B=4 decode body");
    let _ = model.forward_logits(&state).expect("timed B=4 output head");
    (
        started.elapsed().as_secs_f64() * 1_000.0,
        counter_snapshot(),
    )
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    if values.len() % 2 == 0 {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
    } else {
        values[values.len() / 2]
    }
}

fn order_stratum(values: &[f64], parity: usize) -> Vec<f64> {
    values
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| (index % 2 == parity).then_some(value))
        .collect()
}

fn restore_caches(
    caches: &mut [Deepseek4Cache],
    snapshots: &[Deepseek4CacheSnapshot],
    label: &str,
) {
    for (cache, snapshot) in caches.iter_mut().zip(snapshots) {
        cache
            .restore(snapshot)
            .unwrap_or_else(|error| panic!("restore {label} cache: {error:#}"));
    }
}

fn timed_serial_conditioned(
    model: &mut Deepseek4Model,
    caches: &mut [Deepseek4Cache],
    snapshots: &[Deepseek4CacheSnapshot],
    tokens: [u32; LANES],
) -> (TrialCounters, f64, TrialCounters) {
    restore_caches(caches, snapshots, "conditioned serial prime");
    let (_, prime_counters) = timed_serial(model, caches, tokens);
    restore_caches(caches, snapshots, "conditioned serial timing");
    let (elapsed, measured_counters) = timed_serial(model, caches, tokens);
    (prime_counters, elapsed, measured_counters)
}

fn timed_cohort_conditioned(
    model: &mut Deepseek4Model,
    caches: &mut [Deepseek4Cache],
    snapshots: &[Deepseek4CacheSnapshot],
    tokens: [u32; LANES],
) -> (TrialCounters, f64, TrialCounters) {
    restore_caches(caches, snapshots, "conditioned cohort prime");
    let (_, prime_counters) = timed_cohort(model, caches, tokens);
    restore_caches(caches, snapshots, "conditioned cohort timing");
    let (elapsed, measured_counters) = timed_cohort(model, caches, tokens);
    (prime_counters, elapsed, measured_counters)
}

#[test]
#[ignore = "loads the official checkpoint and proves the production B=4 decode transaction"]
fn official_artifact_b4_decode_body_is_exact_and_measured() {
    for name in BENCHMARK_OVERRIDE_ENV {
        assert!(
            std::env::var_os(name).is_none(),
            "B=4 decode proof requires the canonical default environment; unset {name}"
        );
    }
    assert_eq!(
        DECODE_STEPS, 132,
        "proof must retain the reviewed 132-step span"
    );
    assert!(DECODE_STEPS >= 130, "proof must execute at least 130 steps");
    let mut sorted_permutation = PHYSICAL_TO_LOGICAL;
    sorted_permutation.sort_unstable();
    assert_eq!(sorted_permutation, [0, 1, 2, 3]);

    let (path, gguf) = official_artifact();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf)
        .unwrap_or_else(|error| panic!("load official artifact {}: {error:#}", path.display()));
    assert!(model.cfg.compress_ratios.contains(&4));
    assert!(model.cfg.compress_ratios.contains(&128));
    assert_eq!(model.cfg.sliding_window, 128);

    let prefixes = (0..LANES).map(prefix_tokens).collect::<Vec<_>>();
    let mut serial_caches = (0..LANES)
        .map(|logical_lane| {
            let mut cache = model
                .allocate_cache(PARITY_CAPACITY)
                .expect("allocate serial cache");
            model
                .forward_verifier_prefill(&prefixes[logical_lane], &mut cache)
                .expect("install serial prefix");
            cache
        })
        .collect::<Vec<_>>();
    let mut cohort_caches = PHYSICAL_TO_LOGICAL
        .iter()
        .map(|&logical_lane| {
            let mut cache = model
                .allocate_cache(PARITY_CAPACITY)
                .expect("allocate cohort cache");
            model
                .forward_verifier_prefill(&prefixes[logical_lane], &mut cache)
                .expect("install cohort prefix");
            cache
        })
        .collect::<Vec<_>>();
    for step in 0..DECODE_STEPS {
        let logical_tokens = supplied_tokens(step);
        let mut distinct = logical_tokens;
        distinct.sort_unstable();
        assert!(
            distinct.windows(2).all(|pair| pair[0] != pair[1]),
            "decode step {step} must use distinct lane tokens"
        );

        let mut serial_states = Vec::with_capacity(LANES);
        let mut serial_logits = Vec::with_capacity(LANES);
        for logical_lane in 0..LANES {
            let state = model
                .forward_verifier_one(
                    logical_tokens[logical_lane],
                    &mut serial_caches[logical_lane],
                )
                .unwrap_or_else(|error| {
                    panic!("serial step {step} lane {logical_lane} body: {error:#}")
                });
            let logits = model.forward_logits(&state).unwrap_or_else(|error| {
                panic!("serial step {step} lane {logical_lane} head: {error:#}")
            });
            serial_states.push(read_f32(&state, "serial decode state"));
            serial_logits.push(read_f32(&logits, "serial logits"));
        }

        let physical_tokens = PHYSICAL_TO_LOGICAL.map(|logical_lane| logical_tokens[logical_lane]);
        let [lane0, lane1, lane2, lane3] = cohort_caches.as_mut_slice() else {
            panic!("B=4 parity cache count drift")
        };
        let mut cache_refs = [lane0, lane1, lane2, lane3];
        let cohort_state = model
            .forward_verifier_decode_cohort(physical_tokens, &mut cache_refs)
            .unwrap_or_else(|error| panic!("B=4 step {step} body: {error:#}"));
        let cohort_logits = model
            .forward_logits(&cohort_state)
            .unwrap_or_else(|error| panic!("B=4 step {step} head: {error:#}"));
        assert_eq!(cohort_state.shape(), [LANES, 4, 4_096]);
        assert_eq!(
            cohort_logits.shape(),
            [LANES, model.cfg.vocab_size as usize]
        );
        let cohort_state = read_f32(&cohort_state, "B=4 decode state");
        let cohort_logits = read_f32(&cohort_logits, "B=4 logits");
        let state_row = 4 * 4_096;
        let logit_row = model.cfg.vocab_size as usize;
        for (physical_lane, &logical_lane) in PHYSICAL_TO_LOGICAL.iter().enumerate() {
            assert_exact_f32(
                &format!("step {step} logical lane {logical_lane} state"),
                &serial_states[logical_lane],
                &cohort_state[physical_lane * state_row..(physical_lane + 1) * state_row],
            );
            assert_exact_f32(
                &format!("step {step} logical lane {logical_lane} logits"),
                &serial_logits[logical_lane],
                &cohort_logits[physical_lane * logit_row..(physical_lane + 1) * logit_row],
            );
            assert_cache_exact(
                &format!("step {step} logical lane {logical_lane} cache"),
                &serial_caches[logical_lane],
                &cohort_caches[physical_lane],
            );
        }
    }
    assert_eq!(serial_caches[0].position(), PARITY_CAPACITY);
    assert_eq!(cohort_caches[0].position(), PARITY_CAPACITY);
    assert_eq!(
        PARITY_CAPACITY % 128,
        24,
        "proof must cross a ratio-128 boundary"
    );
    assert_eq!(
        PARITY_CAPACITY % 4,
        0,
        "proof must finish on a ratio-4 boundary"
    );

    serial_caches = grow_and_extend_caches(&mut model, serial_caches, [0, 1, 2, 3]);
    cohort_caches = grow_and_extend_caches(&mut model, cohort_caches, PHYSICAL_TO_LOGICAL);
    for (physical_lane, &logical_lane) in PHYSICAL_TO_LOGICAL.iter().enumerate() {
        assert_cache_exact(
            &format!("benchmark anchor logical lane {logical_lane} cache"),
            &serial_caches[logical_lane],
            &cohort_caches[physical_lane],
        );
    }
    let serial_snapshots = serial_caches
        .iter()
        .map(|cache| cache.snapshot().expect("snapshot serial benchmark anchor"))
        .collect::<Vec<_>>();
    let cohort_snapshots = cohort_caches
        .iter()
        .map(|cache| cache.snapshot().expect("snapshot cohort benchmark anchor"))
        .collect::<Vec<_>>();
    eprintln!(
        "DeepSeek-V4 B=4 benchmark loaded-idle settle: position={BENCHMARK_POSITION} logical_capacity={BENCHMARK_LOGICAL_CAPACITY} seconds={LOADED_IDLE_SECONDS}"
    );
    std::thread::sleep(std::time::Duration::from_secs(LOADED_IDLE_SECONDS));

    for (cache, snapshot) in serial_caches.iter_mut().zip(&serial_snapshots) {
        cache
            .restore(snapshot)
            .expect("restore serial warmup cache");
    }
    let warm_tokens = supplied_tokens(0);
    let _ = timed_serial(&mut model, &mut serial_caches, warm_tokens);
    for (cache, snapshot) in cohort_caches.iter_mut().zip(&cohort_snapshots) {
        cache
            .restore(snapshot)
            .expect("restore cohort warmup cache");
    }
    let _ = timed_cohort(&mut model, &mut cohort_caches, warm_tokens);
    let mut serial_ms = Vec::with_capacity(BENCHMARK_PAIRS);
    let mut cohort_ms = Vec::with_capacity(BENCHMARK_PAIRS);
    let mut serial_counters = Vec::with_capacity(BENCHMARK_PAIRS);
    let mut cohort_counters = Vec::with_capacity(BENCHMARK_PAIRS);
    for pair in 0..BENCHMARK_PAIRS {
        let tokens = supplied_tokens(pair + 1);
        if pair % 2 == 0 {
            for (cache, snapshot) in serial_caches.iter_mut().zip(&serial_snapshots) {
                cache
                    .restore(snapshot)
                    .expect("restore serial timing cache");
            }
            let (elapsed, counters) = timed_serial(&mut model, &mut serial_caches, tokens);
            serial_ms.push(elapsed);
            serial_counters.push(counters);
            for (cache, snapshot) in cohort_caches.iter_mut().zip(&cohort_snapshots) {
                cache
                    .restore(snapshot)
                    .expect("restore cohort timing cache");
            }
            let (elapsed, counters) = timed_cohort(&mut model, &mut cohort_caches, tokens);
            cohort_ms.push(elapsed);
            cohort_counters.push(counters);
        } else {
            for (cache, snapshot) in cohort_caches.iter_mut().zip(&cohort_snapshots) {
                cache
                    .restore(snapshot)
                    .expect("restore cohort timing cache");
            }
            let (elapsed, counters) = timed_cohort(&mut model, &mut cohort_caches, tokens);
            cohort_ms.push(elapsed);
            cohort_counters.push(counters);
            for (cache, snapshot) in serial_caches.iter_mut().zip(&serial_snapshots) {
                cache
                    .restore(snapshot)
                    .expect("restore serial timing cache");
            }
            let (elapsed, counters) = timed_serial(&mut model, &mut serial_caches, tokens);
            serial_ms.push(elapsed);
            serial_counters.push(counters);
        }
    }

    // Diagnostic control for topology-conditioned transient lifetime work.
    // Every measured arm follows an untimed execution of that same arm from
    // the exact benchmark snapshot. The second restore keeps the measured
    // cache/token work identical while preventing a serial trial from
    // inheriting cohort pool geometry (and vice versa).
    let mut conditioned_serial_ms = Vec::with_capacity(BENCHMARK_PAIRS);
    let mut conditioned_cohort_ms = Vec::with_capacity(BENCHMARK_PAIRS);
    let mut conditioned_serial_prime_counters = Vec::with_capacity(BENCHMARK_PAIRS);
    let mut conditioned_cohort_prime_counters = Vec::with_capacity(BENCHMARK_PAIRS);
    let mut conditioned_serial_counters = Vec::with_capacity(BENCHMARK_PAIRS);
    let mut conditioned_cohort_counters = Vec::with_capacity(BENCHMARK_PAIRS);
    for pair in 0..BENCHMARK_PAIRS {
        let tokens = supplied_tokens(pair + 1);
        if pair % 2 == 0 {
            let (prime_counters, elapsed, counters) =
                timed_serial_conditioned(&mut model, &mut serial_caches, &serial_snapshots, tokens);
            conditioned_serial_prime_counters.push(prime_counters);
            conditioned_serial_ms.push(elapsed);
            conditioned_serial_counters.push(counters);
            let (prime_counters, elapsed, counters) =
                timed_cohort_conditioned(&mut model, &mut cohort_caches, &cohort_snapshots, tokens);
            conditioned_cohort_prime_counters.push(prime_counters);
            conditioned_cohort_ms.push(elapsed);
            conditioned_cohort_counters.push(counters);
        } else {
            let (prime_counters, elapsed, counters) =
                timed_cohort_conditioned(&mut model, &mut cohort_caches, &cohort_snapshots, tokens);
            conditioned_cohort_prime_counters.push(prime_counters);
            conditioned_cohort_ms.push(elapsed);
            conditioned_cohort_counters.push(counters);
            let (prime_counters, elapsed, counters) =
                timed_serial_conditioned(&mut model, &mut serial_caches, &serial_snapshots, tokens);
            conditioned_serial_prime_counters.push(prime_counters);
            conditioned_serial_ms.push(elapsed);
            conditioned_serial_counters.push(counters);
        }
    }

    assert_eq!(serial_ms.len(), BENCHMARK_PAIRS);
    assert_eq!(cohort_ms.len(), BENCHMARK_PAIRS);
    assert_eq!(conditioned_serial_ms.len(), BENCHMARK_PAIRS);
    assert_eq!(conditioned_cohort_ms.len(), BENCHMARK_PAIRS);
    let body_command_buffers = 43_u64.div_ceil(2);
    let expected_serial_command_buffers = LANES as u64 * (body_command_buffers + 1);
    let expected_cohort_command_buffers = body_command_buffers + 1;
    for pair in 0..BENCHMARK_PAIRS {
        assert_eq!(
            serial_counters[pair].command_buffers, expected_serial_command_buffers,
            "pair {pair} serial command-buffer topology drift: {:?}",
            serial_counters[pair]
        );
        assert_eq!(
            cohort_counters[pair].command_buffers, expected_cohort_command_buffers,
            "pair {pair} B=4 command-buffer topology drift: {:?}",
            cohort_counters[pair]
        );
        assert_eq!(serial_counters[pair].synchronizations, LANES as u64);
        assert_eq!(cohort_counters[pair].synchronizations, 1);
        assert!(serial_counters[pair].dispatches > 0);
        assert!(serial_counters[pair].barriers > 0);
        assert!(cohort_counters[pair].dispatches > 0);
        assert!(cohort_counters[pair].barriers > 0);
        assert_eq!(
            conditioned_serial_counters[pair].command_buffers, expected_serial_command_buffers,
            "pair {pair} conditioned serial command-buffer topology drift: {:?}",
            conditioned_serial_counters[pair]
        );
        assert_eq!(
            conditioned_cohort_counters[pair].command_buffers, expected_cohort_command_buffers,
            "pair {pair} conditioned B=4 command-buffer topology drift: {:?}",
            conditioned_cohort_counters[pair]
        );
        assert_eq!(
            conditioned_serial_counters[pair].synchronizations,
            LANES as u64
        );
        assert_eq!(conditioned_cohort_counters[pair].synchronizations, 1);
        assert!(conditioned_serial_counters[pair].dispatches > 0);
        assert!(conditioned_serial_counters[pair].barriers > 0);
        assert!(conditioned_cohort_counters[pair].dispatches > 0);
        assert!(conditioned_cohort_counters[pair].barriers > 0);
        assert_eq!(
            conditioned_serial_prime_counters[pair].command_buffers,
            expected_serial_command_buffers
        );
        assert_eq!(
            conditioned_serial_prime_counters[pair].synchronizations,
            LANES as u64
        );
        assert_eq!(
            conditioned_cohort_prime_counters[pair].command_buffers,
            expected_cohort_command_buffers
        );
        assert_eq!(conditioned_cohort_prime_counters[pair].synchronizations, 1);
    }
    let serial_median = median(serial_ms.clone());
    let cohort_median = median(cohort_ms.clone());
    let speedup = serial_median / cohort_median;
    let conditioned_serial_median = median(conditioned_serial_ms.clone());
    let conditioned_cohort_median = median(conditioned_cohort_ms.clone());
    let conditioned_serial_even_median = median(order_stratum(&conditioned_serial_ms, 0));
    let conditioned_cohort_even_median = median(order_stratum(&conditioned_cohort_ms, 0));
    let conditioned_serial_odd_median = median(order_stratum(&conditioned_serial_ms, 1));
    let conditioned_cohort_odd_median = median(order_stratum(&conditioned_cohort_ms, 1));
    let conditioned_speedup = conditioned_serial_median / conditioned_cohort_median;
    let conditioned_even_speedup = conditioned_serial_even_median / conditioned_cohort_even_median;
    let conditioned_odd_speedup = conditioned_serial_odd_median / conditioned_cohort_odd_median;
    let unconditioned_deltas = serial_ms
        .iter()
        .zip(&cohort_ms)
        .map(|(&serial, &cohort)| serial - cohort)
        .collect::<Vec<_>>();
    let unconditioned_even_delta_median = median(order_stratum(&unconditioned_deltas, 0));
    let unconditioned_odd_delta_median = median(order_stratum(&unconditioned_deltas, 1));
    let unconditioned_order_signature =
        unconditioned_even_delta_median < 0.0 && unconditioned_odd_delta_median > 0.0;
    eprintln!(
        "DeepSeek-V4 B=4 decode spike: artifact={} parity_prefix={} parity_steps={} benchmark_position={} benchmark_logical_capacity={} loaded_idle_seconds={} permutation={:?} pairs={} order=alternating exact_state_logits_cache_recurrent=true serial_ms={:?} cohort_ms={:?} serial_median_ms={:.3} cohort_median_ms={:.3} speedup={:.4}x unconditioned_even_delta_median_ms={:.3} unconditioned_odd_delta_median_ms={:.3} unconditioned_order_signature={} conditioned_serial_ms={:?} conditioned_cohort_ms={:?} conditioned_serial_median_ms={:.3} conditioned_cohort_median_ms={:.3} conditioned_speedup={:.4}x conditioned_even_speedup={:.4}x conditioned_odd_speedup={:.4}x serial_counters={:?} cohort_counters={:?} conditioned_serial_prime_counters={:?} conditioned_cohort_prime_counters={:?} conditioned_serial_counters={:?} conditioned_cohort_counters={:?}",
        path.display(),
        PREFIX_ROWS,
        DECODE_STEPS,
        BENCHMARK_POSITION,
        BENCHMARK_LOGICAL_CAPACITY,
        LOADED_IDLE_SECONDS,
        PHYSICAL_TO_LOGICAL,
        BENCHMARK_PAIRS,
        serial_ms,
        cohort_ms,
        serial_median,
        cohort_median,
        speedup,
        unconditioned_even_delta_median,
        unconditioned_odd_delta_median,
        unconditioned_order_signature,
        conditioned_serial_ms,
        conditioned_cohort_ms,
        conditioned_serial_median,
        conditioned_cohort_median,
        conditioned_speedup,
        conditioned_even_speedup,
        conditioned_odd_speedup,
        serial_counters,
        cohort_counters,
        conditioned_serial_prime_counters,
        conditioned_cohort_prime_counters,
        conditioned_serial_counters,
        conditioned_cohort_counters,
    );
    let conditioned_pass = unconditioned_order_signature
        && conditioned_serial_median.is_finite()
        && conditioned_cohort_median.is_finite()
        && conditioned_serial_median > conditioned_cohort_median
        && conditioned_cohort_median > 0.0
        && conditioned_even_speedup > 1.0
        && conditioned_odd_speedup > 1.0;

    if let Some(receipt_path) = std::env::var_os(RECEIPT_ENV).map(PathBuf::from) {
        let artifact_bytes = std::fs::metadata(&path)
            .unwrap_or_else(|error| panic!("stat official artifact {}: {error}", path.display()))
            .len();
        let counters_json = |counters: &[TrialCounters]| {
            counters
                .iter()
                .map(|counter| {
                    serde_json::json!({
                        "command_buffers": counter.command_buffers,
                        "synchronizations": counter.synchronizations,
                        "dispatches": counter.dispatches,
                        "barriers": counter.barriers,
                    })
                })
                .collect::<Vec<_>>()
        };
        let receipt = serde_json::json!({
            "schema_version": 1,
            "status": if conditioned_pass { "pass" } else { "fail" },
            "artifact_bytes": artifact_bytes,
            "layers": model.cfg.num_hidden_layers,
            "lanes": LANES,
            "parity": {
                "prefix_rows": PREFIX_ROWS,
                "steps": DECODE_STEPS,
                "final_position": PARITY_CAPACITY,
                "final_mod_4": PARITY_CAPACITY % 4,
                "final_mod_128": PARITY_CAPACITY % 128,
                "physical_to_logical": PHYSICAL_TO_LOGICAL,
                "exact_state_logits_cache_recurrent": true,
            },
            "benchmark": {
                "position": BENCHMARK_POSITION,
                "logical_capacity": BENCHMARK_LOGICAL_CAPACITY,
                "loaded_idle_seconds": LOADED_IDLE_SECONDS,
                "pairs": BENCHMARK_PAIRS,
                "order": "alternating",
                "serial_ms": serial_ms,
                "cohort_ms": cohort_ms,
                "serial_median_ms": serial_median,
                "cohort_median_ms": cohort_median,
                "speedup": speedup,
                "unconditioned_order_signature": {
                    "expected": "even_delta_negative_odd_delta_positive",
                    "even_delta_median_ms": unconditioned_even_delta_median,
                    "odd_delta_median_ms": unconditioned_odd_delta_median,
                    "confirmed": unconditioned_order_signature,
                },
                "conditioned": {
                    "protocol": "same-topology-prime-restore-measure",
                    "primes_per_measurement": 1,
                    "serial_ms": conditioned_serial_ms,
                    "cohort_ms": conditioned_cohort_ms,
                    "serial_median_ms": conditioned_serial_median,
                    "cohort_median_ms": conditioned_cohort_median,
                    "speedup": conditioned_speedup,
                    "even_order_speedup": conditioned_even_speedup,
                    "odd_order_speedup": conditioned_odd_speedup,
                    "serial_prime_counters": counters_json(&conditioned_serial_prime_counters),
                    "cohort_prime_counters": counters_json(&conditioned_cohort_prime_counters),
                    "serial_counters": counters_json(&conditioned_serial_counters),
                    "cohort_counters": counters_json(&conditioned_cohort_counters),
                },
                "serial_counters": counters_json(&serial_counters),
                "cohort_counters": counters_json(&cohort_counters),
                "serial_command_buffers_per_pair": expected_serial_command_buffers,
                "cohort_command_buffers_per_pair": expected_cohort_command_buffers,
                "serial_synchronizations_per_pair": LANES,
                "cohort_synchronizations_per_pair": 1,
            },
            "benchmark_environment": {
                "profile": "clean-hf2q-mlx-metal-v1",
                "override_variables_absent": true,
                "unexpected_override_variables": [],
            },
        });
        let parent = receipt_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| std::path::Path::new("."));
        std::fs::create_dir_all(parent).unwrap_or_else(|error| {
            panic!(
                "create B=4 decode receipt directory {}: {error}",
                parent.display()
            )
        });
        let temporary = receipt_path.with_extension("json.tmp");
        std::fs::write(
            &temporary,
            serde_json::to_vec_pretty(&receipt).expect("serialize B=4 decode receipt"),
        )
        .unwrap_or_else(|error| {
            panic!("write B=4 decode receipt {}: {error}", temporary.display())
        });
        std::fs::rename(&temporary, &receipt_path).unwrap_or_else(|error| {
            panic!(
                "publish B=4 decode receipt {} -> {}: {error}",
                temporary.display(),
                receipt_path.display()
            )
        });
    }
    assert!(
        conditioned_pass,
        "exact B=4 decode lifecycle hypothesis and self-conditioned medians must pass: unconditioned_even_delta={unconditioned_even_delta_median:.3}ms unconditioned_odd_delta={unconditioned_odd_delta_median:.3}ms serial={conditioned_serial_median:.3}ms cohort={conditioned_cohort_median:.3}ms even={conditioned_even_speedup:.4}x odd={conditioned_odd_speedup:.4}x"
    );
}
