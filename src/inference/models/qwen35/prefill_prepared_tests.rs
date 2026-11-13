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

/// Statistical companion of [`gamma`]. The activation's BF16 rounding and the
/// weight's and product's F16 roundings are independent per term, so over
/// many outputs the squared error is at most a third of the squared unit
/// roundoff times `Σ (w·x)²` (a relative error bounded by `u` has variance at
/// most `u²/3`), plus `K` F32 accumulation steps on the running sum. Where
/// [`gamma`] grows with `Σ |w·x|` (about `K` terms), this grows with their
/// L2 norm (about `√K`), so it sits within a small factor of the error the
/// kernel really makes: a tenfold larger error fails it.
struct RmsCheck {
    squared_error: f64,
    squared_bound: f64,
}

impl RmsCheck {
    fn new() -> Self {
        Self {
            squared_error: 0.0,
            squared_bound: 0.0,
        }
    }

    fn add(&mut self, error: f64, squared_terms: f64, magnitude: f64, k: usize) {
        self.squared_error += error * error;
        self.squared_bound += (BF16_UNIT * BF16_UNIT + 2.0 * F16_UNIT * F16_UNIT) / 3.0
            * squared_terms
            + k as f64 * F32_UNIT * F32_UNIT / 3.0 * magnitude * magnitude;
    }

    /// RMS error over RMS bound.
    fn ratio(&self) -> f64 {
        (self.squared_error / self.squared_bound).sqrt()
    }

    /// The RMS error must stay below the modelled bound, and the kernel must
    /// round at all like the model says (otherwise the check has no power).
    fn assert_within(&self, what: &str) {
        let ratio = self.ratio();
        assert!(
            ratio <= RMS_SLACK,
            "{what}: RMS error is {ratio:.3} of the modelled RMS bound (limit {RMS_SLACK})"
        );
        assert!(
            ratio >= RMS_SLACK / 10.0,
            "{what}: RMS error is only {ratio:.4} of the modelled RMS bound; a tenfold \
             regression would pass this gate"
        );
    }
}

/// Headroom of the RMS check over the modelled bound: the uniform-rounding
/// model overstates a BF16 rounding's variance by about two, and a case
/// averages a few hundred outputs.
const RMS_SLACK: f64 = 1.5;

/// `Σ w·x`, `Σ |w·x|` and `Σ (w·x)²` in F64.
fn exact_product(w: &[f32], x: &[f32]) -> (f64, f64, f64) {
    w.iter().zip(x).fold(
        (0.0f64, 0.0f64, 0.0f64),
        |(exact, magnitude, squares), (&w, &x)| {
            let product = f64::from(w) * f64::from(x);
            (
                exact + product,
                magnitude + product.abs(),
                squares + product * product,
            )
        },
    )
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
    // Small shapes at every tiling, then the served inner dimensions: the
    // Qwen3.8 FFN gate (K = 5120) and FFN down (K = 17408) at their real
    // output widths, whose columns are sampled for the reference.
    for (ggml_type, n, k, row_counts) in [
        (
            GgmlType::Q4_K,
            128usize,
            512usize,
            &[2usize, 5, 31, 32, 33, 157][..],
        ),
        (GgmlType::Q6_K, 128, 512, &[2, 5, 31, 32, 33, 157]),
        (GgmlType::Q4_K, 64, 1024, &[2, 5, 31, 32, 33, 157]),
        (GgmlType::Q6_K, 192, 768, &[2, 5, 31, 32, 33, 157]),
        (GgmlType::Q4_K, 17408, 5120, &[5, 33, 157]),
        (GgmlType::Q6_K, 5120, 17408, &[5, 33, 157]),
    ] {
        let weight = synthetic_blocks(&device, ggml_type, n as u32, k as u32, &mut seed);
        let w = dequantize(&weight, ggml_type, n, k);
        let planes = Planes::new();
        let columns: Vec<usize> = (0..n).step_by((n / 128).max(1)).collect();
        for &rows in row_counts {
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
            let mut rms = RmsCheck::new();
            for row in 0..rows {
                let xr = &x[row * k..(row + 1) * k];
                for &col in &columns {
                    let (exact, magnitude, squares) = exact_product(&w[col * k..(col + 1) * k], xr);
                    let got = f64::from(y[row * n + col]);
                    let error = (got - exact).abs();
                    let bound = gamma(k) * magnitude + 1e-30;
                    assert!(
                        error <= bound,
                        "{ggml_type:?} N={n} K={k} rows={rows}: row {row} col {col}: \
                         got {got}, F64 {exact}, |error| {error} > bound {bound}"
                    );
                    worst = worst.max(error / bound);
                    rms.add(error, squares, magnitude, k);
                }
            }
            eprintln!(
                "prepared prefill product {ggml_type:?} N={n} K={k} rows={rows}: RMS error / RMS bound = {:.3}",
                rms.ratio()
            );
            rms.assert_within(&format!("{ggml_type:?} N={n} K={k} rows={rows}"));
        }
    }
    eprintln!("prepared prefill products: worst |error| / bound = {worst:.4}");
    assert!(
        worst > 0.0,
        "the reference comparison saw no rounding at all"
    );
}

/// The fused alpha/beta product and its split against the F64 reference of
/// the two separate projections, at synthetic and at the served head counts
/// and inner dimensions (Qwen3.8: 48 heads, K = 5120; Qwen3.6: 32, 2048).
#[test]
fn fused_alpha_beta_stays_within_the_bounds_of_an_f64_reference() {
    use super::prepared_projection::DeltaNetPlanes;
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    let mut registry = KernelRegistry::new();
    let mut seed = 0x8230_0004_u32;
    for (ggml_type, heads, k) in [
        (GgmlType::Q4_K, 4usize, 256usize),
        (GgmlType::Q4_K, 48, 5120),
        (GgmlType::Q4_K, 32, 2048),
        (GgmlType::Q6_K, 16, 2048),
    ] {
        let alpha = synthetic_blocks(&device, ggml_type, heads as u32, k as u32, &mut seed);
        let beta = synthetic_blocks(&device, ggml_type, heads as u32, k as u32, &mut seed);
        let wa = dequantize(&alpha, ggml_type, heads, k);
        let wb = dequantize(&beta, ggml_type, heads, k);
        let planes = DeltaNetPlanes::default();
        let mut rms = RmsCheck::new();
        for rows in [2usize, 5, 33, 157] {
            let x = random_f32(&mut seed, rows * k);
            let input = upload_f32(&x, &device)
                .expect("upload activation")
                .with_shape(vec![rows, k])
                .expect("shape activation");
            let mut enc = device.command_encoder().expect("encoder");
            let x_bf16 = prepared_projection::bf16_input(
                &mut enc,
                &mut registry,
                &device,
                &input,
                rows as u32,
                k as u32,
            )
            .expect("BF16 activation");
            enc.memory_barrier();
            let fused = prepared_projection::project_alpha_beta(
                &mut enc,
                &mut registry,
                &device,
                &x_bf16,
                &alpha,
                &beta,
                ggml_type,
                &planes,
                rows as u32,
                k as u32,
                heads as u32,
            )
            .expect("fused alpha/beta");
            enc.memory_barrier();
            let (a, b) = prepared_projection::split_alpha_beta(
                &mut enc,
                &mut registry,
                &device,
                &fused,
                rows as u32,
                heads as u32,
            )
            .expect("split alpha/beta");
            enc.commit_and_wait_labeled("test.qwen.prefill_prepared.alpha_beta")
                .expect("complete fused alpha/beta");
            let got_a = download_f32(&a).expect("alpha logits");
            let got_b = download_f32(&b).expect("beta logits");
            for (got, w, name) in [(&got_a, &wa, "alpha"), (&got_b, &wb, "beta")] {
                for row in 0..rows {
                    let xr = &x[row * k..(row + 1) * k];
                    for head in 0..heads {
                        let (exact, magnitude, squares) =
                            exact_product(&w[head * k..(head + 1) * k], xr);
                        let error = (f64::from(got[row * heads + head]) - exact).abs();
                        assert!(
                            error <= gamma(k) * magnitude + 1e-30,
                            "{ggml_type:?} heads={heads} K={k} rows={rows}: {name} row {row} \
                             head {head}: got {}, F64 {exact}",
                            got[row * heads + head]
                        );
                        rms.add(error, squares, magnitude, k);
                    }
                }
            }
        }
        eprintln!(
            "prepared fused alpha/beta {ggml_type:?} heads={heads} K={k}: RMS error / RMS bound = {:.3}",
            rms.ratio()
        );
        rms.assert_within(&format!(
            "fused alpha/beta {ggml_type:?} heads={heads} K={k}"
        ));
    }
}

const SLOTS: u32 = 2;
const CAPACITY: u32 = 1024;
/// Chunk sizes of a fresh slot: the served cue and content sizes (5, 31,
/// 157, 720), one row either side of a 32-row tile boundary, whole tiles,
/// several tiles with a partial last one.
const FRESH_ROWS: [usize; 8] = [5, 17, 31, 33, 64, 97, 157, 720];

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

/// Scale fields of the fixture's K-quant blocks. The joint-verify fixture's
/// own scale (0x211F) drives the residual stream to about 6e6 after the
/// first layer, where no attention product and no later layer can move a
/// logit; at 1/64 of it every routed product of every layer does
/// (`the_fixture_observes_every_routed_product_in_every_layer`).
pub(super) const OBSERVABLE_SCALE: u16 = 0x091F;

/// `(block bytes, quant bytes, scale field offsets)` of a K-quant block.
fn block_layout(ty: GgmlType) -> (usize, std::ops::Range<usize>, &'static [usize]) {
    match ty {
        GgmlType::Q4_K => (144, 16..144, &[0, 2]),
        GgmlType::Q6_K => (210, 0..128, &[208]),
        other => panic!("no synthetic layout for {other:?}"),
    }
}

/// Flip quant bits of every block in place; twice restores the weight.
pub(super) fn xor_quants(buffer: &MlxBuffer, ty: GgmlType) {
    let (block, quants, _) = block_layout(ty);
    let mut handle = buffer.clone();
    for chunk in handle
        .as_mut_slice::<u8>()
        .expect("map weight")
        .chunks_exact_mut(block)
    {
        for byte in &mut chunk[quants.clone()] {
            *byte ^= 0x5A;
        }
    }
}

pub(super) fn set_scale(buffer: &MlxBuffer, ty: GgmlType, scale: u16) {
    let (block, _, offsets) = block_layout(ty);
    let mut handle = buffer.clone();
    for chunk in handle
        .as_mut_slice::<u8>()
        .expect("map weight")
        .chunks_exact_mut(block)
    {
        for &offset in offsets {
            chunk[offset..offset + 2].copy_from_slice(&scale.to_le_bytes());
        }
    }
}

/// Every weight the prepared prefill route serves: `(layer, product, handle,
/// type)`.
fn routed_weights(model: &Qwen35Model) -> Vec<(usize, &'static str, MlxBuffer, GgmlType)> {
    use super::gpu_full_attn::FullAttnQGateWeightsGpu;
    use super::model::{Qwen35FfnWeights, Qwen35LayerWeights};
    let mut out = Vec::new();
    for (index, layer) in model.layers.iter().enumerate() {
        let ffn = match layer {
            Qwen35LayerWeights::NativeLinearAttn { attn, ffn } => {
                out.push((
                    index,
                    "gdn.qkv",
                    attn.attn_qkv.clone(),
                    attn.attn_qkv_ggml_type,
                ));
                out.push((
                    index,
                    "gdn.gate",
                    attn.attn_gate.clone(),
                    attn.attn_gate_ggml_type,
                ));
                out.push((
                    index,
                    "gdn.alpha",
                    attn.ssm_alpha.clone(),
                    attn.ssm_alpha_ggml_type,
                ));
                out.push((
                    index,
                    "gdn.beta",
                    attn.ssm_beta.clone(),
                    attn.ssm_beta_ggml_type,
                ));
                out.push((
                    index,
                    "gdn.out",
                    attn.ssm_out.clone(),
                    attn.ssm_out_ggml_type,
                ));
                ffn
            }
            Qwen35LayerWeights::NativeFullAttn { attn, ffn } => {
                if let FullAttnQGateWeightsGpu::Fused { weight, ggml_type } = &attn.q_gate {
                    out.push((index, "attn.q_gate", weight.clone(), *ggml_type));
                }
                out.push((index, "attn.k", attn.wk.clone(), attn.wk_ggml_type));
                out.push((index, "attn.v", attn.wv.clone(), attn.wv_ggml_type));
                out.push((index, "attn.out", attn.wo.clone(), attn.wo_ggml_type));
                ffn
            }
            _ => panic!("the synthetic model holds a float layer"),
        };
        if let Qwen35FfnWeights::DenseQ(weights) = ffn {
            out.push((
                index,
                "ffn.gate",
                weights.gate_q.clone(),
                weights.ggml_type_gate_up,
            ));
            out.push((
                index,
                "ffn.up",
                weights.up_q.clone(),
                weights.ggml_type_gate_up,
            ));
            out.push((
                index,
                "ffn.down",
                weights.down_q.clone(),
                weights.ggml_type_down,
            ));
        }
    }
    out
}

/// Rescale every weight the prepared prefill route serves to
/// [`OBSERVABLE_SCALE`].
pub(super) fn rescale_routed_weights(model: &Qwen35Model) {
    for (_, _, buffer, ty) in &routed_weights(model) {
        set_scale(buffer, *ty, OBSERVABLE_SCALE);
    }
}

/// The synthetic model at [`OBSERVABLE_SCALE`]. `synthetic_model` primed
/// the thread's GPU cache for an address the model no longer has: the
/// caller primes it again where the model finally lives.
pub(crate) fn observable_synthetic_model(device: &MlxDevice) -> Qwen35Model {
    let model = synthetic_model(device, padded_policy(), false);
    rescale_routed_weights(&model);
    model
}

/// [`observable_synthetic_model`] with its GPU cache primed, mutable.
pub(super) fn with_synthetic_model_mut(run: impl FnOnce(&mut Qwen35Model)) {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    let mut model = observable_synthetic_model(&device);
    model
        .ensure_gpu_cache_primed()
        .expect("prime the GPU cache at the model's final address");
    run(&mut model);
}

/// [`observable_synthetic_model`] with its GPU cache primed.
pub(super) fn with_synthetic_model(run: impl FnOnce(&Qwen35Model)) {
    with_synthetic_model_mut(|model| run(model));
}

/// A schedule that reaches every attention route: a fresh chunk, a short
/// resumed chunk (direct TQ attention), a long resumed chunk (dequantized
/// history) and a one-row forward. Every row's logits per chunk, as bits.
const OBSERVED_CHUNKS: [usize; 4] = [33, 5, 40, 1];

fn observed_schedule(model: &Qwen35Model, prepared: bool) -> Vec<Vec<u32>> {
    let mut cache = cache(model);
    let mut start = 0usize;
    OBSERVED_CHUNKS
        .iter()
        .map(|&rows| {
            let logits = chunk_rows(model, &mut cache, 0, start, rows, prepared);
            start += rows;
            logits
        })
        .collect()
}

/// The gates below are only as strong as the fixture: a product whose bits
/// cannot move a logit is not tested by them. Flipping the quant bits of any
/// one routed weight, in any layer, must change the schedule's logits. (The
/// stock route is mutated: the prepared planes are prepared once per model.)
#[test]
fn the_fixture_observes_every_routed_product_in_every_layer() {
    with_synthetic_model(|model| {
        let baseline = observed_schedule(model, false);
        let weights = routed_weights(model);
        let mut products = std::collections::BTreeSet::new();
        for (layer, product, buffer, ty) in &weights {
            xor_quants(buffer, *ty);
            let mutated = observed_schedule(model, false);
            xor_quants(buffer, *ty);
            assert!(
                mutated != baseline,
                "layer {layer} {product}: no logit of the schedule depends on this weight"
            );
            products.insert(*product);
        }
        assert!(
            observed_schedule(model, false) == baseline,
            "the weights were not restored"
        );
        let expected: std::collections::BTreeSet<&str> = [
            "attn.k",
            "attn.out",
            "attn.q_gate",
            "attn.v",
            "ffn.down",
            "ffn.gate",
            "ffn.up",
            "gdn.alpha",
            "gdn.beta",
            "gdn.gate",
            "gdn.out",
            "gdn.qkv",
        ]
        .into_iter()
        .collect();
        assert_eq!(products, expected, "routed products of the fixture");
        // The prepared route computes the same function to within rounding
        // and differs in bits in every chunk of the schedule.
        let prepared = observed_schedule(model, true);
        for (chunk, (stock, prepared)) in baseline.iter().zip(&prepared).enumerate() {
            assert!(
                stock != prepared,
                "chunk {chunk} ({} rows): the prepared route left the stock bits",
                OBSERVED_CHUNKS[chunk]
            );
        }
    });
}

/// Every layer's output rows of a fresh prepared chunk, as bits
/// (`[layer][row][hidden]`).
fn fresh_layer_rows(model: &Qwen35Model, rows: usize) -> Vec<u32> {
    use crate::inference::spec_decode::dflash::hidden_capture::DFlashCaptureSession;
    let _scope = PrefillScope::forced(true);
    let hidden = model.cfg.hidden_size as usize;
    let layers = model.layers.len();
    let mut session = DFlashCaptureSession::new((0..layers).collect(), rows, hidden, false);
    model
        .forward_gpu_last_logits_with_hidden_dflash(
            &tokens(0, 0, rows),
            &positions_for_range(0, rows),
            &mut cache(model),
            Some(&mut session),
            SlotId(0),
        )
        .unwrap_or_else(|error| panic!("fresh {rows}-row chunk: {error:#}"));
    session.hidden_output.iter().map(|v| v.to_bits()).collect()
}

/// A prefilled row's residual stream after every layer does not depend on
/// the rows after it or on the chunk's size. The logits are compared too
/// where the chunks share the output head's kernel (the head is not a
/// routed product: the stock matmul serves it and picks its kernel by row
/// count, at most eight rows on the mat-vec family).
#[test]
fn prepared_prefill_rows_do_not_depend_on_later_rows_or_the_chunk_size() {
    with_synthetic_model(|model| {
        let vocab = model.cfg.vocab_size as usize;
        let hidden = model.cfg.hidden_size as usize;
        let layers = model.layers.len();
        let longest = *FRESH_ROWS.iter().max().expect("sizes");
        let full_layers = fresh_layer_rows(model, longest);
        let full = chunk_rows(model, &mut cache(model), 0, 0, longest, true);
        for rows in FRESH_ROWS {
            let part = fresh_layer_rows(model, rows);
            for layer in 0..layers {
                for row in 0..rows {
                    let at = |total: usize| {
                        let start = (layer * total + row) * hidden;
                        start..start + hidden
                    };
                    assert!(
                        part[at(rows)] == full_layers[at(longest)],
                        "layer {layer} row {row} of a {rows}-row chunk differs from the same \
                         row of a {longest}-row chunk"
                    );
                }
            }
            if rows <= 8 {
                continue;
            }
            let prefix = chunk_rows(model, &mut cache(model), 0, 0, rows, true);
            for row in 0..rows {
                assert!(
                    prefix[row * vocab..(row + 1) * vocab] == full[row * vocab..(row + 1) * vocab],
                    "logits row {row} of a {rows}-row chunk differ from the same row of a \
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

/// What decode reads after a prepared prefill (full-attention K/V, DeltaNet
/// state) is deterministic, and the decoded logits stay close to those after
/// a stock prefill of the same rows: three serial steps and one four-row
/// step. The greedy token must agree wherever the stock margin exceeds twice
/// the difference.
#[test]
fn decode_after_a_prepared_prefill_is_repeatable_and_close_to_the_stock_prefill() {
    /// Largest |prepared - stock| decode logit over the largest |logit|.
    /// The fixture measures about 5e-4; ten times that fails.
    const LIMIT: f32 = 2.5e-3;
    const HISTORY: usize = 40;
    with_synthetic_model(|model| {
        let vocab = model.cfg.vocab_size as usize;
        let decode = |prepared: bool| -> Vec<f32> {
            let mut cache = cache(model);
            {
                let _scope = PrefillScope::forced(prepared);
                model
                    .forward_gpu_last_logits(
                        &tokens(0, 0, HISTORY),
                        &positions_for_range(0, HISTORY),
                        &mut cache,
                        SlotId(0),
                    )
                    .expect("prefill");
            }
            let mut out = Vec::new();
            for (start, rows) in [
                (HISTORY, 1),
                (HISTORY + 1, 1),
                (HISTORY + 2, 1),
                (HISTORY + 3, 4),
            ] {
                out.extend(
                    model
                        .forward_gpu(
                            &tokens(0, start, rows),
                            &positions_for_range(start as i32, rows),
                            &mut cache,
                            SlotId(0),
                        )
                        .expect("decode step"),
                );
            }
            out
        };
        let first = decode(true);
        let second = decode(true);
        let stock = decode(false);
        assert!(first.iter().all(|value| value.is_finite()));
        assert!(
            first
                .iter()
                .zip(&second)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "decode after a prepared prefill is not repeatable"
        );
        assert!(
            first
                .iter()
                .zip(&stock)
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "decode does not see which route prefilled its history"
        );
        let scale = stock
            .iter()
            .fold(0.0f32, |scale, value| scale.max(value.abs()));
        let worst = first
            .iter()
            .zip(&stock)
            .fold(0.0f32, |worst, (a, b)| worst.max((a - b).abs()));
        eprintln!(
            "decode after prefill: max |prepared - stock| = {worst:.5} of max |logit| {scale:.3} ({:.2e})",
            worst / scale
        );
        assert!(
            worst <= LIMIT * scale,
            "decode after a prepared prefill differs from the stock prefill by {worst} of {scale}"
        );
        for row in 0..7 {
            let top = |logits: &[f32]| {
                let mut order: Vec<usize> = (0..vocab).collect();
                order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
                (order[0], logits[order[0]] - logits[order[1]])
            };
            let (stock_token, margin) = top(&stock[row * vocab..(row + 1) * vocab]);
            let (token, _) = top(&first[row * vocab..(row + 1) * vocab]);
            assert!(
                token == stock_token || margin <= 2.0 * worst,
                "decode row {row}: greedy token {token} against {stock_token} at margin {margin}"
            );
        }
    });
}

/// The DFlash target-layer capture of a prepared chunk does not depend on
/// where the host waits: short chunks at the default eight-layer batch
/// against a wait after every layer, fresh and resumed, and the captured
/// layers are distinct residual streams (so an earlier layer's rows cannot
/// pass for a later one's).
#[test]
fn a_prepared_chunk_capture_equals_the_wait_every_layer_capture() {
    use crate::inference::spec_decode::dflash::hidden_capture::DFlashCaptureSession;
    const K_BATCH_ENV: &str = "HF2Q_FFN_TERMINAL_K_BATCH";
    struct WaitEveryLayer;
    impl WaitEveryLayer {
        fn set() -> Self {
            std::env::set_var(K_BATCH_ENV, "1");
            Self
        }
    }
    impl Drop for WaitEveryLayer {
        fn drop(&mut self) {
            std::env::remove_var(K_BATCH_ENV);
        }
    }
    with_synthetic_model(|model| {
        assert!(
            std::env::var_os(K_BATCH_ENV).is_none(),
            "{K_BATCH_ENV} must be unset: this gate compares the default batch against 1"
        );
        let hidden = model.cfg.hidden_size as usize;
        let capture = |history: usize, rows: usize| -> (Vec<u32>, Vec<u32>) {
            let _scope = PrefillScope::forced(true);
            let mut cache = cache(model);
            if history > 0 {
                model
                    .forward_gpu_last_logits(
                        &tokens(0, 0, history),
                        &positions_for_range(0, history),
                        &mut cache,
                        SlotId(0),
                    )
                    .expect("history prefill");
            }
            let mut session = DFlashCaptureSession::new(vec![1, 4, 6, 7], rows, hidden, false);
            let (logits, _hidden) = model
                .forward_gpu_last_logits_with_hidden_dflash(
                    &tokens(0, history, rows),
                    &positions_for_range(history as i32, rows),
                    &mut cache,
                    Some(&mut session),
                    SlotId(0),
                )
                .expect("captured chunk");
            (
                session.hidden_output.iter().map(|v| v.to_bits()).collect(),
                logits.iter().map(|v| v.to_bits()).collect(),
            )
        };
        for (history, rows) in [
            (0usize, 20usize),
            (20, 5),
            (20, 31),
            (40, 2),
            (17, 32),
            (0, 97),
            (97, 40),
        ] {
            let batched = capture(history, rows);
            let waited = {
                let _wait = WaitEveryLayer::set();
                capture(history, rows)
            };
            assert!(
                batched.1 == waited.1,
                "history {history}, {rows} rows: logits differ"
            );
            assert!(
                batched.0 == waited.0,
                "history {history}, {rows} rows: the capture differs from the wait-every-layer capture"
            );
            let layers: Vec<&[u32]> = batched.0.chunks(rows * hidden).collect();
            assert!(
                layers.windows(2).all(|pair| pair[0] != pair[1]),
                "history {history}, {rows} rows: captured layers are not distinct"
            );
        }
    });
}

/// `HF2Q_QWEN_PREFILL_WAIT_LAYERS` changes when the host waits and when a
/// layer's command buffer is submitted, not what the GPU computes: a large
/// fresh chunk and resumed ones, with target layers the host copies on and
/// off a wait boundary, give the same logits, the same captured rows and the
/// same slot at every setting.
#[test]
fn waiting_every_few_layers_leaves_the_bits_of_waiting_every_layer() {
    use super::prefill_prepared::ForcedWaitLayers;
    use crate::inference::spec_decode::dflash::hidden_capture::DFlashCaptureSession;
    with_synthetic_model(|model| {
        let hidden = model.cfg.hidden_size as usize;
        let run = |wait_layers: usize, captured_layers: &[usize]| {
            let _cadence = ForcedWaitLayers::set(wait_layers);
            let _scope = PrefillScope::forced(true);
            assert_eq!(super::prefill_prepared::wait_layers(), wait_layers);
            let mut cache = cache(model);
            let mut bits = Vec::new();
            let mut start = 0usize;
            for len in [97usize, 40, 33, 157, 5] {
                let mut session =
                    DFlashCaptureSession::new(captured_layers.to_vec(), len, hidden, false);
                let (logits, _hidden) = model
                    .forward_gpu_last_logits_with_hidden_dflash(
                        &tokens(0, start, len),
                        &positions_for_range(start as i32, len),
                        &mut cache,
                        Some(&mut session),
                        SlotId(0),
                    )
                    .unwrap_or_else(|error| panic!("chunk start={start} n={len}: {error:#}"));
                bits.extend(logits.iter().map(|value| value.to_bits()));
                bits.extend(session.hidden_output.iter().map(|value| value.to_bits()));
                start += len;
            }
            // Every row of a chunk without a capture reads the slot back.
            let read_back = model
                .forward_gpu(
                    &tokens(0, start, 41),
                    &positions_for_range(start as i32, 41),
                    &mut cache,
                    SlotId(0),
                )
                .expect("read-back chunk");
            bits.extend(read_back.iter().map(|value| value.to_bits()));
            bits
        };
        for captured_layers in [&[1usize, 4, 6, 7][..], &[0, 3][..], &[2][..]] {
            let every_layer = run(1, captured_layers);
            for wait_layers in [2, 3, 4, 8] {
                assert!(
                    run(wait_layers, captured_layers) == every_layer,
                    "waiting every {wait_layers} layers changed bits (captured layers \
                     {captured_layers:?})"
                );
            }
        }
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

fn rms(values: &[f32]) -> f64 {
    (values
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

fn rms_diff(a: &[f32], b: &[f32]) -> f64 {
    (a.iter()
        .zip(b)
        .map(|(a, b)| {
            let d = f64::from(*a) - f64::from(*b);
            d * d
        })
        .sum::<f64>()
        / a.len() as f64)
        .sqrt()
}

/// Real-artifact measurement of the Class B perturbation, run explicitly on
/// one served model:
/// `HF2Q_QWEN38_GGUF=… HF2Q_PREFILL_PROMPTS=prompts.json <test binary> qwen_prepared_vs_stock_on_served_prompts --ignored --nocapture --test-threads=1`
/// (or `HF2Q_QWEN36_GGUF=…`).
///
/// `HF2Q_PREFILL_PROMPTS` names a JSON list of `{"agent", "tokens", "spans"}`
/// objects: a conversation's final prompt and the row spans `[start, end,
/// prefill]` the server ran, `prefill = false` for rows it decoded. Each
/// conversation is replayed twice on its own cache, once with the stock and
/// once with the prepared prefill route (decoded rows run the stock prefill
/// kernels in both replays, outside the scope). For every prefilled span it
/// prints, per layer, the RMS difference of the two residual streams over
/// the stock stream's RMS (whole chunk and worst row), and for the span's
/// last row the largest logit difference, the stock top-1 margin and whether
/// the greedy token agrees. Nothing is asserted beyond finiteness: this
/// reports the distribution a quality decision needs.
#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF or HF2Q_QWEN36_GGUF, HF2Q_PREFILL_PROMPTS and Apple Metal; run alone"]
fn qwen_prepared_vs_stock_on_served_prompts() {
    use crate::inference::spec_decode::dflash::hidden_capture::DFlashCaptureSession;
    use crate::serve::header::LoadProgress;
    use mlx_native::gguf::GgufFile;

    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let path = match (
        std::env::var_os("HF2Q_QWEN38_GGUF"),
        std::env::var_os("HF2Q_QWEN36_GGUF"),
    ) {
        (Some(path), None) | (None, Some(path)) => std::path::PathBuf::from(path),
        _ => panic!("set exactly one of HF2Q_QWEN38_GGUF and HF2Q_QWEN36_GGUF"),
    };
    let prompts: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var_os("HF2Q_PREFILL_PROMPTS").expect("HF2Q_PREFILL_PROMPTS"))
            .expect("read the prompt file"),
    )
    .expect("parse the prompt file");
    let policy = super::routing_policy::resolve_qwen_ggml_routing_policy(&path);
    let gguf = GgufFile::open(&path).expect("open GGUF");
    let mut progress = LoadProgress::new(false, 1, 0);
    let mut model = Qwen35Model::load_from_gguf(&gguf, &mut progress).expect("load Qwen model");
    model.bind_ggml_routing_policy(policy);
    model.ensure_gpu_cache_primed().expect("prime GPU cache");
    let hidden = model.cfg.hidden_size as usize;
    let layers = model.layers.len();
    let list = |values: &[f64]| {
        values
            .iter()
            .map(|value| format!("{value:.3e}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    for prompt in prompts.as_array().expect("a list of prompts") {
        let agent = prompt["agent"].as_u64().expect("agent");
        let prompt_tokens: Vec<u32> = prompt["tokens"]
            .as_array()
            .expect("tokens")
            .iter()
            .map(|token| token.as_u64().expect("token id") as u32)
            .collect();
        let capacity = ((prompt_tokens.len() + 64).next_multiple_of(256)) as u32;
        let new_cache = || {
            model
                .with_gpu_cache_mut(|device, _registry| {
                    HybridKvCache::new_with_options(&model.cfg, device, capacity, 1, true)
                })
                .expect("allocate TQ cache")
        };
        // [stock replay, prepared replay]
        let mut caches = [new_cache(), new_cache()];
        for span in prompt["spans"].as_array().expect("spans") {
            let start = span[0].as_u64().expect("span start") as usize;
            let end = span[1].as_u64().expect("span end") as usize;
            let prefilled = span[2].as_bool().expect("span kind");
            let rows = end - start;
            let chunk = &prompt_tokens[start..end];
            let positions = positions_for_range(start as i32, rows);
            if !prefilled {
                for cache in &mut caches {
                    model
                        .forward_gpu_last_logits(chunk, &positions, cache, SlotId(0))
                        .expect("decoded rows");
                }
                continue;
            }
            let mut runs = Vec::with_capacity(2);
            for (arm, cache) in caches.iter_mut().enumerate() {
                let _scope = PrefillScope::forced(arm == 1);
                let mut session =
                    DFlashCaptureSession::new((0..layers).collect(), rows, hidden, false);
                let (logits, _hidden) = model
                    .forward_gpu_last_logits_with_hidden_dflash(
                        chunk,
                        &positions,
                        cache,
                        Some(&mut session),
                        SlotId(0),
                    )
                    .expect("prefilled span");
                assert!(logits.iter().all(|value| value.is_finite()));
                runs.push((logits, session.hidden_output));
            }
            let (stock, prepared) = (&runs[0], &runs[1]);
            let mut layer_rel = Vec::with_capacity(layers);
            let mut layer_worst_row = Vec::with_capacity(layers);
            for layer in 0..layers {
                let range = layer * rows * hidden..(layer + 1) * rows * hidden;
                let (s, p) = (&stock.1[range.clone()], &prepared.1[range]);
                layer_rel.push(rms_diff(s, p) / rms(s));
                layer_worst_row.push((0..rows).fold(0.0f64, |worst, row| {
                    let r = row * hidden..(row + 1) * hidden;
                    worst.max(rms_diff(&s[r.clone()], &p[r.clone()]) / rms(&s[r]))
                }));
            }
            let top = |logits: &[f32]| {
                let (mut best, mut second) = ((0usize, f32::NEG_INFINITY), f32::NEG_INFINITY);
                for (index, &value) in logits.iter().enumerate() {
                    if value > best.1 {
                        second = best.1;
                        best = (index, value);
                    } else if value > second {
                        second = value;
                    }
                }
                (best.0, best.1 - second)
            };
            let (stock_token, margin) = top(&stock.0);
            let (prepared_token, _) = top(&prepared.0);
            let logit_diff = stock
                .0
                .iter()
                .zip(&prepared.0)
                .fold(0.0f32, |worst, (a, b)| worst.max((a - b).abs()));
            let logit_scale = stock.0.iter().fold(0.0f32, |scale, v| scale.max(v.abs()));
            eprintln!(
                "[PREP_VS_STOCK] agent={agent} start={start} rows={rows} logit_max_abs_diff={logit_diff:.5} \
                 logit_scale={logit_scale:.3} stock_margin={margin:.4} greedy_agrees={} layer_rel={} layer_worst_row_rel={}",
                stock_token == prepared_token,
                list(&layer_rel),
                list(&layer_worst_row)
            );
        }
    }
}
