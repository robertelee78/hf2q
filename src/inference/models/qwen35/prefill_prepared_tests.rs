//! Metal gates for the prepared prompt-prefill route
//! (`HF2Q_QWEN_PREFILL_PREPARED`, [`super::prefill_prepared`]).
//!
//! The route is not bitwise equal to the stock GGML matmul (Q1 Class B), so
//! these gates check what it must guarantee instead:
//! - every product stays within a K-scaled bound of an F64 reference;
//! - a run is repeatable bit for bit, also on recycled pool storage;
//! - a prefilled row does not depend on the rows after it, on the chunk's
//!   row count, or on what another slot holds.
//!
//! The model-level gates run the synthetic native K-quant model of the joint
//! verify tests (no artifact). `HF2Q_QWEN38_GGUF` runs the ignored
//! real-artifact gate at the end of this file.

use mlx_native::gguf::test_only_dequantize;
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{KernelRegistry, MlxBuffer, MlxDevice};

use super::gpu_full_attn::{download_f32, upload_f32};
use super::joint_decode_test_support::{random_f32, synthetic_blocks};
use super::joint_verify_tests::{padded_policy, synthetic_model};
use super::kv_cache::HybridKvCache;
use super::model::Qwen35Model;
use super::prefill_prepared::PrefillScope;
use super::prepared_projection::{self, Planes};
use super::spec_decode::positions_for_range;
use crate::serve::multi_seq_kv::SlotId;

/// Relative rounding of a BF16 activation (8 significant bits, to nearest).
const BF16_UNIT: f64 = 1.0 / 256.0;
/// Relative rounding of an F16 weight or product (11 significant bits).
const F16_UNIT: f64 = 1.0 / 2048.0;
/// Relative rounding of one F32 accumulation step.
const F32_UNIT: f64 = 1.0 / 16_777_216.0;

/// `|y - Σ w·x| <= gamma(K) · Σ |w·x|`: one BF16 rounding of the activation,
/// F16 roundings of the weight and of the product, and K F32 accumulation
/// steps.
fn gamma(k: usize) -> f64 {
    BF16_UNIT + 2.0 * F16_UNIT + k as f64 * F32_UNIT
}

fn dequantize(weight: &MlxBuffer, ggml_type: GgmlType, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * k];
    test_only_dequantize(
        weight.as_slice::<u8>().expect("weight bytes"),
        ggml_type,
        &mut out,
    )
    .expect("dequantize reference weight");
    out
}

/// One product on the prefill route's exact helper sequence: BF16 copy of
/// the activation, barrier, prepared row-tile MM.
#[allow(clippy::too_many_arguments)]
fn prepared_product(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    weight: &MlxBuffer,
    ggml_type: GgmlType,
    planes: &Planes,
    x: &[f32],
    rows: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let input = upload_f32(x, device)
        .expect("upload activation")
        .with_shape(vec![rows, k])
        .expect("shape activation");
    let mut enc = device.command_encoder().expect("encoder");
    let x_bf16 =
        prepared_projection::bf16_input(&mut enc, registry, device, &input, rows as u32, k as u32)
            .expect("BF16 activation");
    enc.memory_barrier();
    let output = prepared_projection::project(
        &mut enc,
        registry,
        device,
        &x_bf16,
        weight,
        ggml_type,
        planes,
        rows as u32,
        k as u32,
        n as u32,
    )
    .expect("prepared product");
    enc.commit_and_wait_labeled("test.qwen.prefill_prepared.product")
        .expect("complete prepared product");
    download_f32(&output).expect("download product")[..rows * n].to_vec()
}

#[test]
fn prepared_products_stay_within_the_k_scaled_bound_of_an_f64_reference() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    let mut registry = KernelRegistry::new();
    let mut seed = 0x9E37_79B9_u32;
    let mut worst = 0.0f64;
    for (ggml_type, n, k) in [
        (GgmlType::Q4_K, 128usize, 512usize),
        (GgmlType::Q6_K, 128, 512),
        (GgmlType::Q4_K, 64, 1024),
        (GgmlType::Q6_K, 192, 768),
    ] {
        let weight = synthetic_blocks(&device, ggml_type, n as u32, k as u32, &mut seed);
        let w = dequantize(&weight, ggml_type, n, k);
        let planes = Planes::new();
        // One partial tile, the full tile, one row past it, several tiles
        // with a partial last one.
        for rows in [2usize, 5, 31, 32, 33, 157] {
            let x = random_f32(&mut seed, rows * k);
            let y = prepared_product(
                &device,
                &mut registry,
                &weight,
                ggml_type,
                &planes,
                &x,
                rows,
                k,
                n,
            );
            for row in 0..rows {
                let xr = &x[row * k..(row + 1) * k];
                for col in 0..n {
                    let (exact, magnitude) = w[col * k..(col + 1) * k].iter().zip(xr).fold(
                        (0.0f64, 0.0f64),
                        |(exact, magnitude), (&w, &x)| {
                            let product = f64::from(w) * f64::from(x);
                            (exact + product, magnitude + product.abs())
                        },
                    );
                    let got = f64::from(y[row * n + col]);
                    let error = (got - exact).abs();
                    let bound = gamma(k) * magnitude + 1e-30;
                    assert!(
                        error <= bound,
                        "{ggml_type:?} N={n} K={k} rows={rows}: row {row} col {col}: \
                         got {got}, F64 {exact}, |error| {error} > bound {bound}"
                    );
                    worst = worst.max(error / (gamma(k) * magnitude + 1e-30));
                }
            }
        }
    }
    eprintln!("prepared prefill products: worst |error| / bound = {worst:.4}");
    assert!(
        worst > 0.0,
        "the reference comparison saw no rounding at all"
    );
}

const SLOTS: u32 = 2;
const CAPACITY: u32 = 256;
/// Chunk sizes of a fresh slot: one partial 32-row tile, one row either side
/// of a tile boundary, whole tiles, several tiles with a partial last one.
const FRESH_ROWS: [usize; 5] = [17, 31, 33, 64, 97];

fn token(slot: u32, position: usize) -> u32 {
    (slot * 131 + position as u32 * 37 + 11) % 512
}

fn tokens(slot: u32, start: usize, len: usize) -> Vec<u32> {
    (start..start + len)
        .map(|position| token(slot, position))
        .collect()
}

fn cache(model: &Qwen35Model) -> HybridKvCache {
    model
        .with_gpu_cache_mut(|device, _registry| {
            HybridKvCache::new_with_options(&model.cfg, device, CAPACITY, SLOTS, true)
        })
        .expect("allocate TQ cache")
}

/// Every row's logits of one chunk on `slot`, as bits.
fn chunk_rows(
    model: &Qwen35Model,
    cache: &mut HybridKvCache,
    slot: u32,
    start: usize,
    len: usize,
    prepared: bool,
) -> Vec<u32> {
    let _scope = PrefillScope::forced(prepared);
    let logits = model
        .forward_gpu(
            &tokens(slot, start, len),
            &positions_for_range(start as i32, len),
            cache,
            SlotId(slot),
        )
        .unwrap_or_else(|error| panic!("slot {slot} chunk start={start} n={len}: {error:#}"));
    assert_eq!(logits.len(), len * model.cfg.vocab_size as usize);
    assert!(
        logits.iter().all(|value| value.is_finite()),
        "slot {slot} chunk start={start} n={len}: non-finite logits"
    );
    logits.iter().map(|value| value.to_bits()).collect()
}

fn with_synthetic_model(run: impl FnOnce(&Qwen35Model)) {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    let model = synthetic_model(&device, padded_policy(), false);
    // `synthetic_model` primed the thread's GPU cache for the model's
    // address before the move out of it.
    model
        .ensure_gpu_cache_primed()
        .expect("prime the GPU cache at the model's final address");
    run(&model);
}

#[test]
fn prepared_prefill_rows_do_not_depend_on_later_rows_or_the_chunk_size() {
    with_synthetic_model(|model| {
        let vocab = model.cfg.vocab_size as usize;
        let longest = *FRESH_ROWS.iter().max().expect("sizes");
        let full = chunk_rows(model, &mut cache(model), 0, 0, longest, true);
        for rows in FRESH_ROWS {
            let prefix = chunk_rows(model, &mut cache(model), 0, 0, rows, true);
            for row in 0..rows {
                assert!(
                    prefix[row * vocab..(row + 1) * vocab] == full[row * vocab..(row + 1) * vocab],
                    "row {row} of a {rows}-row chunk differs from the same row of a \
                     {longest}-row chunk"
                );
            }
        }
    });
}

#[test]
fn prepared_prefill_is_repeatable_on_recycled_storage_and_across_resumed_chunks() {
    with_synthetic_model(|model| {
        let run = |model: &Qwen35Model| {
            let mut cache = cache(model);
            let mut rows = chunk_rows(model, &mut cache, 0, 0, 97, true);
            // Resumed chunks: the direct TQ route (at most 32 queries) and
            // the dequantized-history route.
            rows.extend(chunk_rows(model, &mut cache, 0, 97, 5, true));
            rows.extend(chunk_rows(model, &mut cache, 0, 102, 40, true));
            rows
        };
        let first = run(model);
        // Different rows through the same pooled scratch and arenas.
        chunk_rows(model, &mut cache(model), 1, 0, 64, true);
        chunk_rows(model, &mut cache(model), 1, 0, 33, false);
        let second = run(model);
        assert!(first == second, "a repeated prepared prefill changed bits");
    });
}

#[test]
fn prepared_prefill_of_a_slot_does_not_depend_on_another_slot() {
    with_synthetic_model(|model| {
        let alone = chunk_rows(model, &mut cache(model), 1, 0, 33, true);
        let mut shared = cache(model);
        chunk_rows(model, &mut shared, 0, 0, 64, true);
        chunk_rows(model, &mut shared, 0, 64, 20, true);
        let beside = chunk_rows(model, &mut shared, 1, 0, 33, true);
        assert!(
            alone == beside,
            "slot 1's prefill depends on what slot 0 holds"
        );
    });
}

#[test]
fn the_scope_selects_the_route_and_a_closed_scope_keeps_the_stock_kernels() {
    with_synthetic_model(|model| {
        let stock = chunk_rows(model, &mut cache(model), 0, 0, 33, false);
        let unscoped = {
            let logits = model
                .forward_gpu(
                    &tokens(0, 0, 33),
                    &positions_for_range(0, 33),
                    &mut cache(model),
                    SlotId(0),
                )
                .expect("unscoped forward");
            logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert!(
            stock == unscoped,
            "a scope with the route off changed the stock forward"
        );
        let prepared = chunk_rows(model, &mut cache(model), 0, 0, 33, true);
        assert!(
            prepared != stock,
            "the routed scope did not change any product"
        );
    });
}

/// Real-artifact gate, run explicitly:
/// `HF2Q_QWEN38_GGUF=… <test binary> qwen38_prepared_prefill_real_model_gate --ignored --nocapture --test-threads=1`
///
/// On the served cache shape it checks, for the chunk sizes of the
/// four-agent workload, that the prepared route is repeatable bit for bit
/// (fresh and resumed chunks), and reports how far its last-row logits sit
/// from the stock route's, whether the greedy token agrees, and the wall
/// time of each chunk on both routes (second pass of each, so weights,
/// pipelines and prepared planes are warm).
#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF (a Qwen3.8-named path) and Apple Metal; run alone"]
fn qwen38_prepared_prefill_real_model_gate() {
    use super::real_artifact_joint_decode_tests::{argmax, load, GateRouting};
    const PATTERN: [u32; 7] = [9707, 374, 279, 15, 1_782, 2_610, 13];
    /// Fresh content chunk, cue, then resumed content chunks and cues.
    const SCHEDULE: [usize; 7] = [720, 5, 157, 5, 31, 200, 5];
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let model = load("HF2Q_QWEN38_GGUF", GateRouting::Serve);
    let run = |prepared: bool| -> Vec<(Vec<f32>, f64)> {
        model
            .ensure_gpu_cache_primed()
            .expect("prime the GPU cache at the model's final address");
        let mut cache = model
            .with_gpu_cache_mut(|device, _registry| {
                HybridKvCache::new_with_options(&model.cfg, device, 4096, 1, true)
            })
            .expect("allocate TQ cache");
        let _scope = PrefillScope::forced(prepared);
        let mut start = 0usize;
        SCHEDULE
            .iter()
            .map(|&len| {
                let chunk: Vec<u32> = (start..start + len)
                    .map(|index| PATTERN[index % PATTERN.len()])
                    .collect();
                let started = std::time::Instant::now();
                let logits = model
                    .forward_gpu_last_logits(
                        &chunk,
                        &positions_for_range(start as i32, len),
                        &mut cache,
                        SlotId(0),
                    )
                    .unwrap_or_else(|error| panic!("chunk start={start} n={len}: {error:#}"));
                let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
                start += len;
                (logits, elapsed_ms)
            })
            .collect()
    };
    let _cold_stock = run(false);
    let stock = run(false);
    let prepared = run(true);
    let repeated = run(true);
    for (index, &len) in SCHEDULE.iter().enumerate() {
        let (stock_logits, stock_ms) = &stock[index];
        let (prepared_logits, _) = &prepared[index];
        let (repeated_logits, prepared_ms) = &repeated[index];
        assert!(
            prepared_logits
                .iter()
                .zip(repeated_logits)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "chunk {index} (n={len}): the prepared route is not repeatable"
        );
        assert!(
            prepared_logits.iter().all(|value| value.is_finite()),
            "chunk {index} (n={len}): non-finite prepared logits"
        );
        let scale = stock_logits
            .iter()
            .fold(0.0f32, |scale, value| scale.max(value.abs()));
        let worst = stock_logits
            .iter()
            .zip(prepared_logits)
            .fold(0.0f32, |worst, (a, b)| worst.max((a - b).abs()));
        eprintln!(
            "[PREPARED_PREFILL] chunk {index} n={len}: stock {stock_ms:.1} ms, prepared \
             {prepared_ms:.1} ms; max |prepared - stock| = {worst:.4} (stock max |logit| \
             {scale:.2}), greedy token stock {} prepared {}",
            argmax(stock_logits),
            argmax(prepared_logits)
        );
    }
}
