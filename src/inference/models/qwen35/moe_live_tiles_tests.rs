//! Model-free Metal gates for the live-tile MoE layer
//! (`HF2Q_QWEN_MOE_LIVE_TILES=1`, [`super::moe_live_tiles`]) at the
//! Qwen3.6-35B-A3B layer shape, on the random layers of
//! [`super::moe_batched_tests`].
//!
//! * row independence: every row of a width-8, -16, -24 or -32 step, in any
//!   order, beside any other rows and under a DFlash2 verify-row mask, is
//!   bitwise the row this route computes alone, including its expert ids and
//!   routing weights;
//! * masked rows: a row the mask removed is never read (its input is NaN)
//!   and the live rows keep their bits;
//! * determinism: two encodings of one step agree bitwise;
//! * F64 reference of the whole layer with a K-scaled bound: BF16 expert
//!   inputs and SwiGLU, half expert weights, FP32 sums, for the routed
//!   experts and the shared expert.
//!
//! Run: `cargo test --release --bin hf2q moe_live_tiles_tests -- --test-threads=1 --nocapture`

use mlx_native::ops::dflash2_attention_pack::DFlash2MoeRouteMaskParams;
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::decode_pool;
use super::gpu_ffn::MoeFfnWeightsGpuQ;
use super::moe_batched::{
    build_moe_ffn_layer_gpu_batched_route_into, DFlash2MoeRouteMask, MoeExpertRoute,
};
use super::moe_batched_tests::{
    bits, dequant_rows, packed, router_logits_f64, rows, shape, silu, top_k_f64, Layer, RowResult,
    H, MI, MSH, NE, TOP_K,
};

const LANE_ROWS: usize = 8;

/// Native-storage weights of `layer` with the live-tile planes prepared.
fn live_tile_weights(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    layer: &Layer,
) -> MoeFfnWeightsGpuQ {
    let mut weights = layer.gpu(device, true, &layer.gpu_experts(device));
    let mut enc = device.command_encoder().expect("prepare encoder");
    weights
        .prepare_grouped_m8(&mut enc, registry, device, H as u32, MI as u32, MSH as u32)
        .expect("prepare expert planes");
    // The process switch is unset in tests: prepare the live-tile state directly.
    let live_tiles = weights
        .prepare_live_tiles(&mut enc, registry, device, H as u32, MSH as u32)
        .expect("prepare shared expert planes");
    enc.commit_and_wait_labeled("test.qwen.moe_live_tiles.prepare")
        .expect("commit preparation");
    weights.prepared_m8.as_mut().expect("prepared").live_tiles = Some(live_tiles);
    weights
}

/// One live-tile layer over the rows `select` (in that order). `verify_rows`
/// masks each lane of eight rows to its first `verify_rows[lane]` rows.
fn run_live_tiles(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    weights: &MoeFfnWeightsGpuQ,
    x: &[Vec<f32>],
    residual: &[Vec<f32>],
    select: &[usize],
    verify_rows: Option<&[u32]>,
) -> Vec<RowResult> {
    decode_pool::checked_reset_decode_pool().expect("reset decode pool");
    let r = select.len();
    let xin = packed(x, select, device);
    let res = packed(residual, select, device);
    let mut out = device
        .alloc_buffer(r * H * 4, DType::F32, vec![r, H])
        .expect("alloc out");
    let verify_buffer: Option<MlxBuffer> = verify_rows.map(|rows| {
        let mut buffer = device
            .alloc_buffer(rows.len() * 4, DType::U32, vec![rows.len()])
            .expect("verify rows");
        buffer
            .as_mut_slice::<u32>()
            .expect("verify slice")
            .copy_from_slice(rows);
        buffer
    });
    let route_mask = verify_buffer
        .as_ref()
        .map(|verify_rows| DFlash2MoeRouteMask {
            verify_rows,
            params: DFlash2MoeRouteMaskParams {
                physical_lane: [0, 1, 2, 3],
                lanes: (r / LANE_ROWS) as u32,
                rows_per_lane: LANE_ROWS as u32,
                top_k: TOP_K as u32,
                n_experts: NE as u32,
                width: r as u32,
            },
        });
    let mut enc = device.command_encoder().expect("encoder");
    let routing = build_moe_ffn_layer_gpu_batched_route_into(
        &mut enc,
        device,
        registry,
        &xin,
        weights,
        shape(),
        Some(&res),
        &mut out,
        r as u32,
        0,
        route_mask,
        MoeExpertRoute::LiveTiles,
    )
    .expect("live-tile MoE layer");
    enc.commit_and_wait_labeled("test.qwen.moe_live_tiles")
        .expect("commit");
    let out = out.as_slice::<f32>().expect("read out");
    let ids = routing.ids.as_slice::<u32>().expect("read ids");
    let w = routing.weights.as_slice::<f32>().expect("read weights");
    (0..r)
        .map(|i| RowResult {
            out: out[i * H..(i + 1) * H].to_vec(),
            ids: ids[i * TOP_K..(i + 1) * TOP_K].to_vec(),
            weights: w[i * TOP_K..(i + 1) * TOP_K].to_vec(),
        })
        .collect()
}

fn assert_row(actual: &RowResult, expected: &RowResult, label: &str) {
    assert_eq!(actual.ids, expected.ids, "{label}: expert ids");
    assert_eq!(
        bits(&actual.weights),
        bits(&expected.weights),
        "{label}: routing weights"
    );
    assert_eq!(bits(&actual.out), bits(&expected.out), "{label}: output");
}

fn rows_are_independent(down_t: GgmlType, seed: u64) {
    const ROWS: usize = 32;
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let layer = Layer::random(down_t, seed);
    let weights = live_tile_weights(&device, &mut registry, &layer);
    let (x, residual) = rows(ROWS, seed ^ 0x711e);

    let single: Vec<RowResult> = (0..ROWS)
        .map(|row| {
            run_live_tiles(
                &device,
                &mut registry,
                &weights,
                &x,
                &residual,
                &[row],
                None,
            )
            .remove(0)
        })
        .collect();
    assert!(
        single.iter().all(|r| r.out.iter().all(|v| v.is_finite())),
        "non-finite output"
    );

    // Determinism and every width, order and repeat.
    let all: Vec<usize> = (0..ROWS).collect();
    let mut steps: Vec<Vec<usize>> = vec![
        all.clone(),
        all.clone(),
        all.iter().rev().copied().collect(),
        (0..24).collect(),
        (8..24).collect(),
        (24..32).collect(),
        vec![3, 1, 0, 2],
        vec![6, 4, 6],
        (0..17).collect(),
    ];
    // Rows that share experts: the same row in every lane.
    steps.push((0..ROWS).map(|row| row % LANE_ROWS).collect());
    for select in &steps {
        let step = run_live_tiles(
            &device,
            &mut registry,
            &weights,
            &x,
            &residual,
            select,
            None,
        );
        for (place, &row) in select.iter().enumerate() {
            assert_row(
                &step[place],
                &single[row],
                &format!("{down_t:?} rows {select:?}: row {row}"),
            );
        }
    }

    // Verify-row masks: live rows keep their bits while the masked rows'
    // inputs are NaN, so no live result read a masked row.
    for verify_rows in [[5u32, 8, 1, 3], [8, 8, 8, 8], [1, 1, 1, 1], [2, 7, 4, 6]] {
        let live = |row: usize| (row % LANE_ROWS) < verify_rows[row / LANE_ROWS] as usize;
        let mut masked_x = x.clone();
        for row in (0..ROWS).filter(|row| !live(*row)) {
            masked_x[row].fill(f32::NAN);
        }
        let step = run_live_tiles(
            &device,
            &mut registry,
            &weights,
            &masked_x,
            &residual,
            &all,
            Some(&verify_rows),
        );
        for row in 0..ROWS {
            if live(row) {
                assert_row(
                    &step[row],
                    &single[row],
                    &format!("{down_t:?} mask {verify_rows:?}: row {row}"),
                );
            } else {
                assert!(
                    step[row].ids.iter().all(|&id| id == NE as u32)
                        && step[row].weights.iter().all(|&w| w == 0.0),
                    "{down_t:?} mask {verify_rows:?}: row {row} kept a route"
                );
            }
        }
    }
    eprintln!(
        "moe_live_tiles {down_t:?}: {} unmasked steps and 4 masked width-32 steps bitwise equal to rows = 1 \
         (register path {})",
        steps.len(),
        weights
            .prepared_m8
            .as_ref()
            .and_then(|prepared| prepared.live_tiles.as_ref())
            .expect("prepared")
            .column_tile
    );
}

#[test]
fn live_tile_rows_equal_single_row_route_bitwise_down_q6_k() {
    rows_are_independent(GgmlType::Q6_K, 0x711e_06);
}

#[test]
fn live_tile_rows_equal_single_row_route_bitwise_down_q4_k() {
    rows_are_independent(GgmlType::Q4_K, 0x711e_04);
}

/// Relative rounding error of BF16.
const BF16_ULP: f64 = 1.0 / 256.0;
/// A half below this magnitude is subnormal: its rounding error is absolute.
const HALF_MIN_NORMAL: f64 = 1.0 / 16384.0;
const F32_ULP: f64 = 1.0 / 16_777_216.0;

/// `(sum w x, sum |x| (|w| + HALF_MIN_NORMAL))` for each of `rows` rows of `w`.
fn dot_rows(w: &[f32], x: &[f64], rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
    (0..rows)
        .map(|row| {
            w[row * cols..(row + 1) * cols].iter().zip(x).fold(
                (0.0, 0.0),
                |(sum, abs), (&w, &x)| {
                    let w = f64::from(w);
                    (sum + w * x, abs + x.abs() * (w.abs() + HALF_MIN_NORMAL))
                },
            )
        })
        .unzip()
}

/// Worst-case relative error of a live-tile dot product of `k` terms: BF16
/// inputs when `rounded_input`, half weights, an FP32 sum of `k` products.
fn tile_epsilon(k: usize, rounded_input: bool) -> f64 {
    let input = if rounded_input { BF16_ULP } else { 0.0 };
    (1.0 + input) * (1.0 + 2.0f64.powi(-11) + 2.0 * F32_ULP) * (1.0 + (k as f64 + 2.0) * F32_ULP)
        - 1.0
}

/// One expert's F64 output for the row `x` and the worst-case error of its
/// live tiles: `(y, bound)` per output element.
fn expert_reference(
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    x: &[f64],
    intermediate: usize,
) -> (Vec<f64>, Vec<f64>) {
    let epsilon = tile_epsilon(H, true);
    let (gate, gate_abs) = dot_rows(gate_w, x, intermediate, H);
    let (up, up_abs) = dot_rows(up_w, x, intermediate, H);
    let hidden: Vec<f64> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
    // |SiLU'| <= 1.1; the FP32 SwiGLU rounds to BF16 once.
    let hidden_bound: Vec<f64> = (0..intermediate)
        .map(|j| {
            let (gate_bound, up_bound) = (epsilon * gate_abs[j], epsilon * up_abs[j]);
            let exact =
                1.1 * gate_bound * (up[j].abs() + up_bound) + silu(gate[j]).abs() * up_bound;
            exact + (BF16_ULP + 2.0f64.powi(-16)) * (hidden[j].abs() + exact)
        })
        .collect();
    let epsilon = tile_epsilon(intermediate, false);
    (0..H)
        .map(|i| {
            let w = &down_w[i * intermediate..(i + 1) * intermediate];
            let (mut y, mut bound) = (0.0f64, 0.0f64);
            for j in 0..intermediate {
                let w = f64::from(w[j]);
                y += w * hidden[j];
                bound += w.abs() * hidden_bound[j]
                    + epsilon * (w.abs() + HALF_MIN_NORMAL) * (hidden[j].abs() + hidden_bound[j]);
            }
            (y, bound)
        })
        .unzip()
}

fn matches_f64_reference(down_t: GgmlType, seed: u64) {
    const ROWS: usize = 32;
    /// Rows whose F64 logit gap among the top 9 is below this may select
    /// other experts under FP32 rounding.
    const TIE_GAP: f64 = 1e-4;
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let layer = Layer::random(down_t, seed);
    let weights = live_tile_weights(&device, &mut registry, &layer);
    let (x, residual) = rows(ROWS, seed ^ 0xf64);
    let all: Vec<usize> = (0..ROWS).collect();
    let gpu = run_live_tiles(&device, &mut registry, &weights, &x, &residual, &all, None);
    let shared = (
        dequant_rows(&layer.shared_gate, GgmlType::Q4_K, 0, MSH, H),
        dequant_rows(&layer.shared_up, GgmlType::Q4_K, 0, MSH, H),
        dequant_rows(&layer.shared_down, down_t, 0, H, MSH),
    );
    // FP32 stages (routing weights, shared gate, reduce): the batched
    // route's envelope per unit of accumulated magnitude.
    let dense = 8.0 * (H + MSH) as f64 * F32_ULP;
    let (mut worst, mut squared_error, mut squared_norm, mut near_ties) =
        (0.0f64, 0.0f64, 0.0f64, 0);
    for row in 0..ROWS {
        let xf: Vec<f64> = x[row].iter().map(|&v| f64::from(v)).collect();
        let (ids, route_weights, gap) = top_k_f64(&router_logits_f64(&layer.router, &x[row]));
        if gap < TIE_GAP {
            near_ties += 1;
            continue;
        }
        let gpu_ids: Vec<usize> = gpu[row].ids.iter().map(|&id| id as usize).collect();
        assert_eq!(
            gpu_ids, ids,
            "{down_t:?} row {row}: expert ids differ from F64"
        );
        let mut out: Vec<f64> = residual[row].iter().map(|&v| f64::from(v)).collect();
        let mut bound: Vec<f64> = out.iter().map(|v| dense * v.abs()).collect();
        let mut add = |weight: f64, (y, y_bound): (Vec<f64>, Vec<f64>)| {
            for i in 0..H {
                out[i] += weight * y[i];
                bound[i] += weight * (y_bound[i] + dense * (y[i].abs() + y_bound[i]));
            }
        };
        for (&expert, &weight) in ids.iter().zip(&route_weights) {
            let gate_w = dequant_rows(&layer.gate_exps, GgmlType::Q4_K, expert * MI, MI, H);
            let up_w = dequant_rows(&layer.up_exps, GgmlType::Q4_K, expert * MI, MI, H);
            let down_w = dequant_rows(&layer.down_exps, down_t, expert * H, H, MI);
            add(weight, expert_reference(&gate_w, &up_w, &down_w, &xf, MI));
        }
        let logit: f64 = layer
            .gate_inp
            .iter()
            .zip(&xf)
            .map(|(&w, &v)| f64::from(w) * v)
            .sum();
        add(
            1.0 / (1.0 + (-logit).exp()),
            expert_reference(&shared.0, &shared.1, &shared.2, &xf, MSH),
        );
        for i in 0..H {
            let error = (f64::from(gpu[row].out[i]) - out[i]).abs();
            assert!(
                error <= bound[i],
                "{down_t:?} row {row} [{i}]: |{} - {}| = {error:e} > {:e}",
                gpu[row].out[i],
                out[i],
                bound[i]
            );
            worst = worst.max(error / bound[i]);
            squared_error += error * error;
            squared_norm += (out[i] - f64::from(residual[row][i])).powi(2);
        }
    }
    assert!(near_ties < ROWS, "every row is a near tie");
    eprintln!(
        "moe_live_tiles {down_t:?} vs F64: {ROWS} rows ({near_ties} near-tie rows skipped), ids identical, \
         relative RMS error of the FFN term {:.3e}, worst error/bound {worst:.3}",
        (squared_error / squared_norm).sqrt()
    );
}

#[test]
fn live_tile_route_matches_f64_reference_down_q6_k() {
    matches_f64_reference(GgmlType::Q6_K, 0x64_16);
}

#[test]
fn live_tile_route_matches_f64_reference_down_q4_k() {
    matches_f64_reference(GgmlType::Q4_K, 0x64_14);
}
