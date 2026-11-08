use std::cmp::Ordering;
use std::time::Instant;

use mlx_native::MlxBuffer;

use super::cache::Deepseek4Cache;
use super::chunk_split_invariance_tests::{cache_bytes, CacheBytes};
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
    "HF2Q_MM_ID_SCHEDULE",
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

#[test]
#[ignore = "diagnostic common-prefix cache-copy spike; loads the release checkpoint"]
fn diagnostic_common_prefix_cache_copy_is_exact_and_faster() {
    const LANES: usize = 4;
    const COMMON_PREFIX_ROWS: usize = 319;
    const LOGICAL_CAPACITY: usize = 131_072;
    const PAIRS: usize = 3;

    let (path, gguf) = official_artifact();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf)
        .unwrap_or_else(|error| panic!("load official artifact {}: {error:#}", path.display()));
    let prefix = token_batch(0, 0, COMMON_PREFIX_ROWS);
    let prefix_batches = std::array::from_fn::<_, LANES, _>(|_| prefix.as_slice());

    let mut baseline_caches = (0..LANES)
        .map(|_| {
            model
                .allocate_logical_cache(LOGICAL_CAPACITY)
                .expect("allocate baseline logical cache")
        })
        .collect::<Vec<_>>();
    let mut source_cache = model
        .allocate_cache(COMMON_PREFIX_ROWS + 1)
        .expect("allocate compact prefix source cache");
    let mut copied_caches = (0..LANES - 1)
        .map(|_| {
            model
                .allocate_logical_cache(LOGICAL_CAPACITY)
                .expect("allocate copied logical cache")
        })
        .collect::<Vec<_>>();

    let run_baseline = |model: &mut Deepseek4Model, caches: &mut [Deepseek4Cache]| {
        for cache in caches.iter_mut() {
            cache.reset().expect("reset baseline cache");
        }
        let mut cache_refs = caches.iter_mut().collect::<Vec<_>>();
        let started = Instant::now();
        let states = model
            .forward_verifier_prefill_cohort(&prefix_batches, &mut cache_refs)
            .expect("compute four identical prefixes cooperatively");
        (started.elapsed().as_secs_f64() * 1_000.0, states)
    };
    let run_copied = |
        model: &mut Deepseek4Model,
        source: &mut Deepseek4Cache,
        destinations: &mut [Deepseek4Cache],
    | {
        source.reset().expect("reset common-prefix source cache");
        for cache in destinations.iter_mut() {
            cache.reset().expect("reset copied-prefix destination cache");
        }
        let started = Instant::now();
        let state = model
            .forward_verifier_prefill(&prefix, source)
            .expect("compute common prefix once");
        for cache in destinations.iter_mut() {
            cache
                .migrate_from(source, None)
                .expect("copy common prefix into independent cache");
        }
        (started.elapsed().as_secs_f64() * 1_000.0, state)
    };

    let _ = run_baseline(&mut model, &mut baseline_caches);
    let _ = run_copied(&mut model, &mut source_cache, &mut copied_caches);

    let (baseline_parity_ms, baseline_states) =
        run_baseline(&mut model, &mut baseline_caches);
    let (copied_parity_ms, copied_state) =
        run_copied(&mut model, &mut source_cache, &mut copied_caches);
    let copied_state = read_f32(&copied_state, "single common-prefix state");
    for (lane, state) in baseline_states.iter().enumerate() {
        assert_exact_f32(
            &format!("common-prefix lane {lane} final state"),
            &copied_state,
            &read_f32(state, "cooperative common-prefix state"),
        );
    }

    let next_token = 42_424;
    let baseline_next = baseline_caches
        .iter_mut()
        .map(|cache| {
            model
                .forward_verifier_one(next_token, cache)
                .expect("continue baseline common prefix")
        })
        .collect::<Vec<_>>();
    let copied_source_next = model
        .forward_verifier_one(next_token, &mut source_cache)
        .expect("continue source common prefix");
    let mut copied_next = vec![read_f32(&copied_source_next, "source continuation")];
    copied_next.extend(copied_caches.iter_mut().map(|cache| {
        read_f32(
            &model
                .forward_verifier_one(next_token, cache)
                .expect("continue copied common prefix"),
            "copied continuation",
        )
    }));
    for lane in 0..LANES {
        assert_exact_f32(
            &format!("common-prefix lane {lane} continuation state"),
            &read_f32(&baseline_next[lane], "baseline continuation"),
            &copied_next[lane],
        );
        assert_eq!(baseline_caches[lane].position(), COMMON_PREFIX_ROWS + 1);
    }
    assert_eq!(source_cache.position(), COMMON_PREFIX_ROWS + 1);
    assert!(
        copied_caches
            .iter()
            .all(|cache| cache.position() == COMMON_PREFIX_ROWS + 1)
    );

    let mut baseline_ms = Vec::with_capacity(PAIRS);
    let mut copied_ms = Vec::with_capacity(PAIRS);
    for pair in 0..PAIRS {
        if pair % 2 == 0 {
            baseline_ms.push(run_baseline(&mut model, &mut baseline_caches).0);
            copied_ms.push(run_copied(&mut model, &mut source_cache, &mut copied_caches).0);
        } else {
            copied_ms.push(run_copied(&mut model, &mut source_cache, &mut copied_caches).0);
            baseline_ms.push(run_baseline(&mut model, &mut baseline_caches).0);
        }
    }
    let baseline_median_ms = median(baseline_ms.clone());
    let copied_median_ms = median(copied_ms.clone());
    eprintln!(
        "DeepSeek-V4 common-prefix cache-copy spike: artifact={} lanes={LANES} common_prefix_rows={COMMON_PREFIX_ROWS} logical_capacity={LOGICAL_CAPACITY} exact_state_and_next_token=true parity_baseline_ms={baseline_parity_ms:.3} parity_copy_ms={copied_parity_ms:.3} pairs={PAIRS} order=alternating baseline_ms={baseline_ms:?} copied_ms={copied_ms:?} baseline_median_ms={baseline_median_ms:.3} copied_median_ms={copied_median_ms:.3} speedup={:.4}x saved_ms={:.3} peak_rss_bytes={}",
        path.display(),
        baseline_median_ms / copied_median_ms,
        baseline_median_ms - copied_median_ms,
        process_peak_rss_bytes(),
    );
}

/// How a cohort route's companion lanes relate to the lane under test.
#[derive(Clone, Copy, Debug)]
enum Companions {
    /// Same warm prefix length and chunk lengths as the lane under test,
    /// which packs first (the pre-S3 equal-plan cohort).
    Equal,
    /// S3 unequal-plan cohorts: companions sit at other cursors (warm
    /// prefixes of `PREFIX_ROWS + UNEQUAL_PREFIX_OFFSETS[lane]`) and ingest
    /// their own planner chunks under the same window cap, so starts and
    /// lengths differ; the lane under test packs at `target_offset`.
    Unequal { target_offset: usize },
}

/// Extra warm-prefix rows of each unequal companion.
const UNEQUAL_PREFIX_OFFSETS: [usize; 3] = [211, 523, 1_337];

/// One route's observable result for the #726 route-boundary gate.
struct RouteReceipt {
    chunks: Vec<String>,
    suffix_rows: Vec<f32>,
    cache: Vec<CacheBytes>,
    logits: Vec<f32>,
    greedy_token: u32,
    decode_state: Vec<f32>,
}

/// Ingest `prompt` from a warm `PREFIX_ROWS` cache with the serving planner's
/// chunk sequence under one route: `windows` is the route's window cap
/// (solo 16, Mixed 2, cohort lanes 8/5/4) and `width > 1` runs every
/// pre-anchor matrix chunk as a cooperative cohort with `width - 1` real
/// companion lanes. Token-wise chunks replay one row at a time, as the engine
/// does. `decoding_peers > 0` adds that many warm peer caches that run one
/// joint decode step (the serving joint path) before every target chunk, as
/// peers decode between a Mixed lane's prefill slices.
fn route_receipt(
    model: &mut Deepseek4Model,
    prompt: &[u32],
    windows: usize,
    width: usize,
    decoding_peers: usize,
    relation: Companions,
) -> RouteReceipt {
    use crate::serve::api::engine_deepseek4::{
        plan_resumable_prefill_chunk, ResumablePrefillChunk, RECOVERY_TAIL_TOKENS,
    };
    let recovery = prompt.len() - RECOVERY_TAIL_TOKENS;
    let extra = |lane: usize| match relation {
        Companions::Equal => 0,
        Companions::Unequal { .. } => UNEQUAL_PREFIX_OFFSETS[lane - 1],
    };
    // Unequal companions get prompts long enough to always plan a full
    // matrix chunk of their own; the anchor of each stays far ahead.
    let companion_len = |lane: usize| match relation {
        Companions::Equal => prompt.len(),
        Companions::Unequal { .. } => prompt.len() + extra(lane) + 4_096,
    };
    let capacity = (1..width).map(companion_len).fold(prompt.len(), usize::max) + 1;
    let companion_prompts: Vec<Vec<u32>> = (1..width)
        .map(|lane| {
            let prefix = PREFIX_ROWS + extra(lane);
            let mut tokens = token_batch(lane, 0, prefix);
            tokens.extend(token_batch(lane, 2, companion_len(lane) - prefix));
            tokens
        })
        .collect();
    let mut cache = model.allocate_cache(capacity).expect("allocate route cache");
    let mut companions: Vec<Deepseek4Cache> = (1..width)
        .map(|_| model.allocate_cache(capacity).expect("allocate companion cache"))
        .collect();
    model
        .forward_verifier_prefill(&prompt[..PREFIX_ROWS], &mut cache)
        .expect("install warm prefix");
    let mut companion_cursors = Vec::with_capacity(width.saturating_sub(1));
    for (lane, (companion, tokens)) in companions.iter_mut().zip(&companion_prompts).enumerate() {
        let prefix = PREFIX_ROWS + extra(lane + 1);
        model
            .forward_verifier_prefill(&tokens[..prefix], companion)
            .expect("install companion prefix");
        companion_cursors.push(prefix);
    }
    let mut peers: Vec<Deepseek4Cache> = (0..decoding_peers)
        .map(|peer| {
            let mut peer_cache = model.allocate_cache(capacity).expect("allocate peer cache");
            model
                .forward_verifier_prefill(&token_batch(10 + peer, 0, PREFIX_ROWS), &mut peer_cache)
                .expect("install decoding peer prefix");
            peer_cache
        })
        .collect();
    let mut peer_step = 0;
    let mut cursor = PREFIX_ROWS;
    let mut chunks = Vec::new();
    let mut suffix_rows = Vec::new();
    let mut last = None;
    while cursor < prompt.len() {
        if !peers.is_empty() {
            let tokens: Vec<u32> = (0..peers.len())
                .map(|peer| token_batch(10 + peer, 3, peer_step + 1)[peer_step])
                .collect();
            let mut peer_refs: Vec<&mut Deepseek4Cache> = peers.iter_mut().collect();
            model
                .forward_verifier_decode_joint_with_commit_gate(&tokens, &mut peer_refs, || Ok(()))
                .expect("decoding peers' joint step");
            peer_step += 1;
        }
        let segment_end = if cursor < recovery { recovery } else { prompt.len() };
        let slice = plan_resumable_prefill_chunk(cursor, segment_end - cursor, 128, windows, recovery)
            .expect("plan route chunk");
        match slice.chunk {
            ResumablePrefillChunk::Matrix(rows) => {
                let tokens = &prompt[cursor..cursor + rows];
                let state = if width > 1 && cursor + rows <= recovery {
                    // Each companion's chunk: the target's span (equal) or its
                    // own planner chunk at its own cursor (unequal).
                    let spans: Vec<(usize, usize)> = companion_cursors
                        .iter()
                        .zip(&companion_prompts)
                        .map(|(&at, tokens)| match relation {
                            Companions::Equal => (cursor, rows),
                            Companions::Unequal { .. } => {
                                let anchor = tokens.len() - RECOVERY_TAIL_TOKENS;
                                let slice = plan_resumable_prefill_chunk(at, anchor - at, 128, windows, anchor)
                                    .expect("plan companion chunk");
                                let ResumablePrefillChunk::Matrix(own) = slice.chunk else {
                                    panic!("unequal companion planned token-wise work at {at}");
                                };
                                (at, own)
                            }
                        })
                        .collect();
                    chunks.push(format!(
                        "cohort{width}x{rows}+{:?}",
                        spans.iter().map(|(_, own)| *own).collect::<Vec<_>>()
                    ));
                    let target_offset = match relation {
                        Companions::Equal => 0,
                        Companions::Unequal { target_offset } => target_offset,
                    };
                    let mut batches: Vec<&[u32]> = companion_prompts
                        .iter()
                        .zip(&spans)
                        .map(|(lane, &(at, own))| &lane[at..at + own])
                        .collect();
                    batches.insert(target_offset, tokens);
                    let mut caches: Vec<&mut Deepseek4Cache> = companions.iter_mut().collect();
                    caches.insert(target_offset, &mut cache);
                    let state = model
                        .forward_verifier_prefill_cohort(&batches, &mut caches)
                        .expect("cohort route chunk")
                        .swap_remove(target_offset);
                    for (at, (_, own)) in companion_cursors.iter_mut().zip(&spans) {
                        *at += own;
                    }
                    state
                } else {
                    chunks.push(format!("matrix{rows}"));
                    model
                        .forward_verifier_prefill(tokens, &mut cache)
                        .expect("matrix route chunk")
                };
                suffix_rows.extend(read_f32(&state, "route chunk state"));
                cursor += rows;
                last = Some(state);
            }
            ResumablePrefillChunk::Incremental(rows) => {
                chunks.push(format!("token-wise{rows}"));
                for &token in &prompt[cursor..cursor + rows] {
                    let state = model
                        .forward_verifier_one(token, &mut cache)
                        .expect("token-wise route row");
                    suffix_rows.extend(read_f32(&state, "token-wise state"));
                    last = Some(state);
                }
                cursor += rows;
            }
        }
    }
    let last = model
        .last_token_state(&last.expect("route ingested rows"))
        .expect("view last prompt row");
    let logits_buffer = model.forward_logits(&last).expect("route prompt logits");
    let logits = read_f32(&logits_buffer, "route logits");
    let greedy_token = model.greedy_token(&logits_buffer).expect("route greedy token");
    let cache_snapshot = cache_bytes(&cache);
    let decode = model
        .forward_verifier_one(greedy_token, &mut cache)
        .expect("route first decode step");
    RouteReceipt {
        chunks,
        suffix_rows,
        cache: cache_snapshot,
        logits,
        greedy_token,
        decode_state: read_f32(&decode, "route decode state"),
    }
}

/// #726 model gate: the same warm long suffix ingested under solo
/// (2,048-row chunks), Mixed (256-row chunks, alone and with two or three
/// peers running a joint decode step before every slice) and cohort widths
/// 2/3/4 (1,024/640/512-row lanes with real companions, and width 2 with
/// decoding peers). The serving planner keeps every pre-anchor row on the
/// matrix route and the anchor's last eight rows token-wise under every
/// route; chunk boundaries still differ by route, and equivalence rests on
/// matrix-chunk split invariance. So every suffix row's final state, every
/// physical cache byte, the prompt logits and the first decode step must be
/// bitwise equal to solo. The 2,312-row pre-anchor segment is nine 256-row
/// Mixed windows plus 8 rows: before #726 Mixed replayed those 8 rows
/// token-wise. S3 unequal-plan cohorts add routes whose companions sit at
/// other cursors with other chunk lengths (target first, middle or last in
/// the pack, with and without decoding peers, and the two-lane Mixed slice):
/// they must equal solo too.
#[test]
#[ignore = "loads the release checkpoint; #726 route-boundary model gate"]
fn official_artifact_prefill_route_boundary_is_route_independent() {
    use crate::serve::api::engine_deepseek4::RECOVERY_TAIL_TOKENS;
    const PRE_ANCHOR_ROWS: usize = 9 * 256 + 8;
    let (path, gguf) = official_artifact();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf)
        .unwrap_or_else(|error| panic!("load official artifact {}: {error:#}", path.display()));
    let mut prompt = token_batch(0, 0, PREFIX_ROWS);
    prompt.extend(token_batch(0, 2, PRE_ANCHOR_ROWS + RECOVERY_TAIL_TOKENS));

    // (route, window cap, cohort width, decoding peers, companions). The
    // S3 unequal-plan routes pack the lane under test with companions at
    // other cursors and chunk lengths, first and last in the pack, with and
    // without decoding peers, and as the Mixed cohort slice (two lanes, one
    // window each).
    use Companions::{Equal, Unequal};
    let routes: [(&str, usize, usize, usize, Companions); 14] = [
        ("solo", 16, 1, 0, Equal),
        ("mixed", 2, 1, 0, Equal),
        ("mixed+2-decoding-peers", 2, 1, 2, Equal),
        ("mixed+3-decoding-peers", 2, 1, 3, Equal),
        ("cohort-2", 8, 2, 0, Equal),
        ("cohort-2+2-decoding-peers", 8, 2, 2, Equal),
        ("cohort-3", 5, 3, 0, Equal),
        ("cohort-4", 4, 4, 0, Equal),
        ("unequal-cohort-2-first", 8, 2, 0, Unequal { target_offset: 0 }),
        ("unequal-cohort-4-last", 4, 4, 0, Unequal { target_offset: 3 }),
        ("unequal-cohort-3-middle+2-decoding-peers", 5, 3, 2, Unequal { target_offset: 1 }),
        ("unequal-cohort-4-first+3-decoding-peers", 4, 4, 3, Unequal { target_offset: 0 }),
        ("mixed-cohort-2-first+2-decoding-peers", 1, 2, 2, Unequal { target_offset: 0 }),
        ("mixed-cohort-2-last+3-decoding-peers", 1, 2, 3, Unequal { target_offset: 1 }),
    ];
    let (_, solo_windows, solo_width, solo_peers, solo_relation) = routes[0];
    let reference =
        route_receipt(&mut model, &prompt, solo_windows, solo_width, solo_peers, solo_relation);
    eprintln!("DS4_ROUTE_BOUNDARY solo chunks={:?}", reference.chunks);
    for &(route, windows, width, decoding_peers, relation) in &routes[1..] {
        let receipt = route_receipt(&mut model, &prompt, windows, width, decoding_peers, relation);
        eprintln!("DS4_ROUTE_BOUNDARY {route} chunks={:?}", receipt.chunks);
        assert_exact_f32(
            &format!("{route} suffix final states"),
            &reference.suffix_rows,
            &receipt.suffix_rows,
        );
        for (expected, actual) in reference.cache.iter().zip(&receipt.cache) {
            assert_eq!(expected.label, actual.label);
            let row = expected
                .bytes
                .iter()
                .zip(&actual.bytes)
                .position(|(a, b)| a != b)
                .map(|byte| byte / expected.row_bytes);
            assert!(row.is_none(), "{route} {} differs from row {row:?}", expected.label);
        }
        assert_exact_f32(&format!("{route} prompt logits"), &reference.logits, &receipt.logits);
        assert_eq!(reference.greedy_token, receipt.greedy_token, "{route} greedy token");
        assert_exact_f32(
            &format!("{route} first decode state"),
            &reference.decode_state,
            &receipt.decode_state,
        );
        eprintln!("DS4_ROUTE_BOUNDARY {route} bitwise_equal=true");
    }
}
