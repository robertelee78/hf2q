//! Model-free Metal gates for the batched MoE decode route
//! (`HF2Q_QWEN_MOE_BATCHED=1`, [`super::moe_batched`]) at the Qwen3.6-35B-A3B
//! layer shape: hidden 2048, 256 experts top-8, expert intermediate 512,
//! shared intermediate 512. Weights are random bytes in the exact GGML block
//! layouts (expert and shared gate/up Q4_K; down Q6_K or Q4_K; router and
//! shared gate logit F32), with finite f16 scales.
//!
//! * batched == serial: every row of a `rows = 2..8` decode layer, in any row
//!   order and next to any other rows, is bitwise the `rows = 1` layer of that
//!   row, including expert ids and routing weights; the serial decode builder
//!   (`build_moe_ffn_layer_gpu_q_into` at one row) runs exactly that route;
//! * one width-32 DFlash2 target layer stays within the F64 reference envelope.
//! * F64 reference of the whole layer (routing on F64 router logits, ties to
//!   the lower expert): identical expert ids, output within
//!   `8 * (hidden + moe_i) * 2^-24 * S` where `S` is the F64 magnitude of the
//!   final accumulation.
//! * old route (BF16 router/shared expert, per row) vs new route: identical
//!   expert ids wherever BF16 rounding of the router does not change the F64
//!   selection; selection-change frequency and output max/mean |diff| are
//!   printed.
//!
//! Run: `cargo test --release --bin hf2q moe_batched_tests -- --test-threads=1 --nocapture`

use std::collections::HashMap;

use mlx_native::gguf::test_only_dequantize;
use mlx_native::ops::dense_gemv_bf16::dense_gemv_bf16_f32;
use mlx_native::ops::dense_mm_bf16::DenseMmBf16F32Params;
use mlx_native::ops::moe_softmax_topk::dispatch_moe_softmax_topk;
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::decode_pool;
use super::ffn::MoeFfnShape;
use super::gpu_ffn::{build_moe_ffn_layer_gpu_q_into, MoeDenseStorage, MoeFfnWeightsGpuQ};
use super::gpu_full_attn::{upload_bf16_from_f32, upload_f32};
use super::moe_batched::build_moe_ffn_layer_gpu_batched_into;

const H: usize = 2048;
const NE: usize = 256;
const TOP_K: usize = 8;
const MI: usize = 512;
const MSH: usize = 512;

fn shape() -> MoeFfnShape {
    MoeFfnShape {
        hidden_size: H as u32,
        num_experts: NE as u32,
        num_experts_per_tok: TOP_K as u32,
        moe_intermediate_size: MI as u32,
        shared_intermediate_size: MSH as u32,
    }
}

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    /// Uniform in [-1, 1).
    fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn in_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.uniform() * 0.5 + 0.5) * (hi - lo)
    }
}

fn f16_le(value: f32) -> [u8; 2] {
    half::f16::from_f32(value).to_bits().to_le_bytes()
}

/// Random GGML blocks for a `[rows, cols]` matrix with finite, normal f16
/// scales sized so weights stay in about ±0.1.
fn random_blocks(t: GgmlType, rows: usize, cols: usize, rng: &mut Rng) -> Vec<u8> {
    let block_bytes = t.block_bytes() as usize;
    let blocks = rows * cols / t.block_values() as usize;
    let mut bytes = Vec::with_capacity(blocks * block_bytes);
    for _ in 0..blocks * block_bytes / 8 {
        bytes.extend_from_slice(&rng.next_u64().to_le_bytes());
    }
    bytes.resize(blocks * block_bytes, 0);
    for block in bytes.chunks_exact_mut(block_bytes) {
        match t {
            GgmlType::Q4_K => {
                block[0..2].copy_from_slice(&f16_le(rng.in_range(0.8e-4, 1.6e-4)));
                block[2..4].copy_from_slice(&f16_le(rng.in_range(0.8e-4, 1.6e-4)));
            }
            GgmlType::Q6_K => {
                for scale in &mut block[192..208] {
                    *scale = ((rng.next_u64() % 33) as i8 - 16) as u8;
                }
                block[208..210].copy_from_slice(&f16_le(rng.in_range(0.8e-4, 1.6e-4)));
            }
            other => panic!("no random layout for {other:?}"),
        }
    }
    bytes
}

fn dequant_rows(bytes: &[u8], t: GgmlType, row0: usize, rows: usize, cols: usize) -> Vec<f32> {
    let row_bytes = cols / t.block_values() as usize * t.block_bytes() as usize;
    let mut out = vec![0.0f32; rows * cols];
    test_only_dequantize(&bytes[row0 * row_bytes..(row0 + rows) * row_bytes], t, &mut out)
        .expect("dequantize");
    out
}

fn upload_u8(device: &MlxDevice, bytes: &[u8]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(bytes.len(), DType::U8, vec![bytes.len()])
        .expect("alloc u8");
    buffer.as_mut_slice::<u8>().expect("map u8").copy_from_slice(bytes);
    buffer
}

fn uniform_vec(rng: &mut Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| rng.uniform() * scale).collect()
}

/// Host copy of one random layer.
struct Layer {
    down_t: GgmlType,
    router: Vec<f32>,
    gate_inp: Vec<f32>,
    shared_gate: Vec<u8>,
    shared_up: Vec<u8>,
    shared_down: Vec<u8>,
    gate_exps: Vec<u8>,
    up_exps: Vec<u8>,
    down_exps: Vec<u8>,
}

impl Layer {
    fn random(down_t: GgmlType, seed: u64) -> Self {
        let mut rng = Rng(seed);
        Self {
            down_t,
            router: uniform_vec(&mut rng, NE * H, 0.05),
            gate_inp: uniform_vec(&mut rng, H, 0.05),
            shared_gate: random_blocks(GgmlType::Q4_K, MSH, H, &mut rng),
            shared_up: random_blocks(GgmlType::Q4_K, MSH, H, &mut rng),
            shared_down: random_blocks(down_t, H, MSH, &mut rng),
            gate_exps: random_blocks(GgmlType::Q4_K, NE * MI, H, &mut rng),
            up_exps: random_blocks(GgmlType::Q4_K, NE * MI, H, &mut rng),
            down_exps: random_blocks(down_t, NE * H, MI, &mut rng),
        }
    }

    /// Expert stacks on the GPU (shared by both storages of a test).
    fn gpu_experts(&self, device: &MlxDevice) -> [MlxBuffer; 3] {
        [
            upload_u8(device, &self.gate_exps),
            upload_u8(device, &self.up_exps),
            upload_u8(device, &self.down_exps),
        ]
    }

    /// GPU weights; `native` selects the batched route's storage, otherwise
    /// the BF16 router/shared copies of the per-row route.
    fn gpu(&self, device: &MlxDevice, native: bool, experts: &[MlxBuffer; 3]) -> MoeFfnWeightsGpuQ {
        let stride = |t: GgmlType, rows: usize, cols: usize| {
            (rows * cols / t.block_values() as usize * t.block_bytes() as usize) as u64
        };
        let f32_matrix = |values: &[f32], shape: Vec<usize>| {
            upload_f32(values, device).expect("upload f32").with_shape(shape).expect("shape")
        };
        let (router, shared_gate_inp, shared_gate, shared_up, shared_down, dense_storage) = if native {
            (
                f32_matrix(&self.router, vec![NE, H]),
                f32_matrix(&self.gate_inp, vec![H]),
                upload_u8(device, &self.shared_gate),
                upload_u8(device, &self.shared_up),
                upload_u8(device, &self.shared_down),
                MoeDenseStorage::Native {
                    shared_gate_up: GgmlType::Q4_K,
                    shared_down: self.down_t,
                },
            )
        } else {
            let bf16 = |values: &[f32]| upload_bf16_from_f32(values, device).expect("upload bf16");
            (
                bf16(&self.router),
                bf16(&self.gate_inp),
                bf16(&dequant_rows(&self.shared_gate, GgmlType::Q4_K, 0, MSH, H)),
                bf16(&dequant_rows(&self.shared_up, GgmlType::Q4_K, 0, MSH, H)),
                bf16(&dequant_rows(&self.shared_down, self.down_t, 0, H, MSH)),
                MoeDenseStorage::Bf16,
            )
        };
        MoeFfnWeightsGpuQ {
            router,
            expert_gate_q: experts[0].clone(),
            expert_up_q: experts[1].clone(),
            expert_down_q: experts[2].clone(),
            ggml_type_gate_up: GgmlType::Q4_K,
            ggml_type_down: self.down_t,
            expert_gate_stride: stride(GgmlType::Q4_K, MI, H),
            expert_up_stride: stride(GgmlType::Q4_K, MI, H),
            expert_down_stride: stride(self.down_t, H, MI),
            num_experts: NE as u32,
            shared_gate_inp,
            shared_gate,
            shared_up,
            shared_down,
            dense_storage,
            expert_gate_affine: None,
            expert_up_affine: None,
            expert_down_affine: None,
        }
    }
}

/// Inputs and residuals for `n` rows.
fn rows(n: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = Rng(seed);
    let x = (0..n).map(|_| uniform_vec(&mut rng, H, 1.5)).collect();
    let residual = (0..n).map(|_| uniform_vec(&mut rng, H, 1.0)).collect();
    (x, residual)
}

struct RowResult {
    out: Vec<f32>,
    ids: Vec<u32>,
    weights: Vec<f32>,
}

fn packed(values: &[Vec<f32>], select: &[usize], device: &MlxDevice) -> MlxBuffer {
    let mut flat = Vec::with_capacity(select.len() * H);
    for &row in select {
        flat.extend_from_slice(&values[row]);
    }
    upload_f32(&flat, device)
        .expect("upload rows")
        .with_shape(vec![select.len(), H])
        .expect("shape rows")
}

/// One batched layer over the rows `select` (in that order); per-row results.
fn run_batched(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    weights: &MoeFfnWeightsGpuQ,
    x: &[Vec<f32>],
    residual: &[Vec<f32>],
    select: &[usize],
) -> Vec<RowResult> {
    decode_pool::checked_reset_decode_pool().expect("reset decode pool");
    let r = select.len();
    let xin = packed(x, select, device);
    let res = packed(residual, select, device);
    let mut out = device.alloc_buffer(r * H * 4, DType::F32, vec![r, H]).expect("alloc out");
    let mut enc = device.command_encoder().expect("encoder");
    let routing = build_moe_ffn_layer_gpu_batched_into(
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
        None,
    )
    .expect("batched MoE layer");
    enc.commit_and_wait_labeled("test.qwen.moe_batched").expect("commit");
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

/// The serial decode builder at one row (what `forward_gpu_greedy` calls).
fn run_serial_builder(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    weights: &MoeFfnWeightsGpuQ,
    x: &[f32],
    residual: &[f32],
) -> Vec<f32> {
    decode_pool::checked_reset_decode_pool().expect("reset decode pool");
    let xin = upload_f32(x, device).expect("x").with_shape(vec![1, H]).expect("shape");
    let res = upload_f32(residual, device).expect("res").with_shape(vec![1, H]).expect("shape");
    let mut enc = device.command_encoder().expect("encoder");
    let out = build_moe_ffn_layer_gpu_q_into(&mut enc, device, registry, &xin, weights, shape(), Some(&res), 0)
        .expect("serial MoE builder");
    enc.commit_and_wait_labeled("test.qwen.moe_serial").expect("commit");
    out.as_slice::<f32>().expect("read")[..H].to_vec()
}

/// The old route's routing: BF16 router gemv + the shipped top-k kernel.
fn old_route_ids(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    weights: &MoeFfnWeightsGpuQ,
    x: &[f32],
) -> Vec<u32> {
    let xin = upload_f32(x, device).expect("x").with_shape(vec![1, H]).expect("shape");
    let mut logits = device.alloc_buffer(NE * 4, DType::F32, vec![1, NE]).expect("logits");
    let ids = device.alloc_buffer(TOP_K * 4, DType::U32, vec![TOP_K]).expect("ids");
    let w = device.alloc_buffer(TOP_K * 4, DType::F32, vec![TOP_K]).expect("w");
    let mut enc = device.command_encoder().expect("encoder");
    let params = DenseMmBf16F32Params { m: 1, n: NE as u32, k: H as u32, src0_batch: 1, src1_batch: 1 };
    dense_gemv_bf16_f32(&mut enc, registry, device, &weights.router, &xin, &mut logits, &params)
        .expect("bf16 router");
    enc.memory_barrier();
    dispatch_moe_softmax_topk(&mut enc, registry, device, &logits, &ids, &w, 1, NE as u32, TOP_K as u32)
        .expect("topk");
    enc.commit_and_wait_labeled("test.qwen.moe_old_routing").expect("commit");
    ids.as_slice::<u32>().expect("ids").to_vec()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn silu(v: f64) -> f64 {
    v / (1.0 + (-v).exp())
}

/// F64 top-k by (probability desc, expert asc) of `logits`, with the smallest
/// logit gap among the top `TOP_K + 1` (how close the selection is to a tie).
fn top_k_f64(logits: &[f64]) -> (Vec<usize>, Vec<f64>, f64) {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f64 = e.iter().sum();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap().then(a.cmp(&b)));
    let chosen = order[..TOP_K].to_vec();
    let total: f64 = chosen.iter().map(|&i| e[i] / sum).sum();
    let weights = chosen.iter().map(|&i| e[i] / sum / total).collect();
    let gap = order[..=TOP_K]
        .windows(2)
        .map(|pair| logits[pair[0]] - logits[pair[1]])
        .fold(f64::INFINITY, f64::min);
    (chosen, weights, gap)
}

fn router_logits_f64(router: &[f32], x: &[f32]) -> Vec<f64> {
    (0..NE)
        .map(|e| {
            router[e * H..(e + 1) * H]
                .iter()
                .zip(x)
                .map(|(&w, &v)| w as f64 * v as f64)
                .sum()
        })
        .collect()
}

fn matvec_f64(w: &[f32], x: &[f64], rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
    let mut y = vec![0.0; rows];
    let mut magnitude = vec![0.0; rows];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        for c in 0..cols {
            let p = row[c] as f64 * x[c];
            y[r] += p;
            magnitude[r] += p.abs();
        }
    }
    (y, magnitude)
}

/// F64 reference of one row: (expert ids, output, magnitude S per element).
struct Reference {
    ids: Vec<usize>,
    gap: f64,
    out: Vec<f64>,
    magnitude: Vec<f64>,
}

struct ExpertCache<'a> {
    layer: &'a Layer,
    experts: HashMap<usize, (Vec<f32>, Vec<f32>, Vec<f32>)>,
    shared: (Vec<f32>, Vec<f32>, Vec<f32>),
}

impl<'a> ExpertCache<'a> {
    fn new(layer: &'a Layer) -> Self {
        Self {
            layer,
            experts: HashMap::new(),
            shared: (
                dequant_rows(&layer.shared_gate, GgmlType::Q4_K, 0, MSH, H),
                dequant_rows(&layer.shared_up, GgmlType::Q4_K, 0, MSH, H),
                dequant_rows(&layer.shared_down, layer.down_t, 0, H, MSH),
            ),
        }
    }

    fn reference(&mut self, x: &[f32], residual: &[f32]) -> Reference {
        let layer = self.layer;
        let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let (ids, weights, gap) = top_k_f64(&router_logits_f64(&layer.router, x));
        let mut out: Vec<f64> = residual.iter().map(|&v| v as f64).collect();
        let mut magnitude: Vec<f64> = residual.iter().map(|&v| (v as f64).abs()).collect();
        for (&e, &w) in ids.iter().zip(&weights) {
            let (gate, up, down) = self.experts.entry(e).or_insert_with(|| {
                (
                    dequant_rows(&layer.gate_exps, GgmlType::Q4_K, e * MI, MI, H),
                    dequant_rows(&layer.up_exps, GgmlType::Q4_K, e * MI, MI, H),
                    dequant_rows(&layer.down_exps, layer.down_t, e * H, H, MI),
                )
            });
            let (g, _) = matvec_f64(gate, &xf, MI, H);
            let (u, _) = matvec_f64(up, &xf, MI, H);
            let hidden: Vec<f64> = g.iter().zip(&u).map(|(&g, &u)| silu(g) * u).collect();
            let (y, m) = matvec_f64(down, &hidden, H, MI);
            for i in 0..H {
                out[i] += w * y[i];
                magnitude[i] += w * m[i];
            }
        }
        let (sg, su, sd) = &self.shared;
        let (a, _) = matvec_f64(sg, &xf, MSH, H);
        let (b, _) = matvec_f64(su, &xf, MSH, H);
        let hs: Vec<f64> = a.iter().zip(&b).map(|(&a, &b)| silu(a) * b).collect();
        let (ys, ms) = matvec_f64(sd, &hs, H, MSH);
        let logit: f64 = layer.gate_inp.iter().zip(&xf).map(|(&w, &v)| w as f64 * v).sum();
        let gate = 1.0 / (1.0 + (-logit).exp());
        for i in 0..H {
            out[i] += gate * ys[i];
            magnitude[i] += gate * ms[i];
        }
        Reference { ids, gap, out, magnitude }
    }
}

fn batched_equals_serial(down_t: GgmlType, seed: u64) {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let layer = Layer::random(down_t, seed);
    let weights = layer.gpu(&device, true, &layer.gpu_experts(&device));
    let (x, residual) = rows(8, seed ^ 0x5eed);

    let single: Vec<RowResult> = (0..8)
        .map(|row| run_batched(&device, &mut registry, &weights, &x, &residual, &[row]).remove(0))
        .collect();
    assert!(single.iter().all(|r| r.out.iter().all(|v| v.is_finite())), "non-finite output");

    // The serial decode builder runs the same route at one row.
    for row in [0usize, 5] {
        let serial = run_serial_builder(&device, &mut registry, &weights, &x[row], &residual[row]);
        assert_eq!(
            bits(&serial),
            bits(&single[row].out),
            "{down_t:?}: serial builder row {row} differs from the batched route at rows = 1"
        );
    }

    let mut batches: Vec<Vec<usize>> = (2..=8).map(|w| (0..w).collect()).collect();
    batches.extend([
        vec![3, 1, 0, 2],
        vec![2, 0],
        vec![1, 3, 2],
        vec![7, 0, 5, 2],
        vec![6, 4, 6],
        vec![5, 6, 7, 4],
    ]);
    for select in &batches {
        let batched = run_batched(&device, &mut registry, &weights, &x, &residual, select);
        for (slot, &row) in select.iter().enumerate() {
            let (b, s) = (&batched[slot], &single[row]);
            assert_eq!(b.ids, s.ids, "{down_t:?} rows {select:?}: row {row} expert ids");
            assert_eq!(
                bits(&b.weights),
                bits(&s.weights),
                "{down_t:?} rows {select:?}: row {row} routing weights"
            );
            assert_eq!(
                bits(&b.out),
                bits(&s.out),
                "{down_t:?} rows {select:?}: row {row} output differs from rows = 1"
            );
        }
    }
    eprintln!(
        "moe_batched {down_t:?}: {} batches (widths 2..8, permutations, repeats) bitwise equal to rows = 1; serial builder == rows = 1",
        batches.len()
    );
}

#[test]
fn batched_rows_equal_single_row_route_bitwise_down_q6_k() {
    batched_equals_serial(GgmlType::Q6_K, 0x36_06);
}

#[test]
fn batched_rows_equal_single_row_route_bitwise_down_q4_k() {
    batched_equals_serial(GgmlType::Q4_K, 0x36_04);
}

#[test]
fn batched_selected_rows_ignore_larger_output_tile() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let layer = Layer::random(GgmlType::Q6_K, 0x36_17);
    let weights = layer.gpu(&device, true, &layer.gpu_experts(&device));
    let (x, residual) = rows(17, 0x17_32);
    let selected = (0..17).collect::<Vec<_>>();
    let expected = run_batched(&device, &mut registry, &weights, &x, &residual, &selected);

    decode_pool::checked_reset_decode_pool().expect("reset decode pool");
    let xin = packed(&x, &selected, &device);
    let res = packed(&residual, &selected, &device);
    let mut out = device.alloc_buffer(32 * H * 4, DType::F32, vec![32, H]).expect("alloc out");
    let mut enc = device.command_encoder().expect("encoder");
    let routing = build_moe_ffn_layer_gpu_batched_into(
        &mut enc,
        &device,
        &mut registry,
        &xin,
        &weights,
        shape(),
        Some(&res),
        &mut out,
        17,
        0,
        None,
    )
    .expect("batched MoE layer");
    enc.commit_and_wait_labeled("test.qwen.moe_batched.padded_output").expect("commit");
    let out = out.as_slice::<f32>().expect("read output");
    let ids = routing.ids.as_slice::<u32>().expect("read ids");
    let routing_weights = routing.weights.as_slice::<f32>().expect("read weights");
    for (row, expected) in expected.iter().enumerate() {
        assert_eq!(&out[row * H..(row + 1) * H], expected.out, "row {row} output");
        assert_eq!(&ids[row * TOP_K..(row + 1) * TOP_K], expected.ids, "row {row} ids");
        assert_eq!(
            &routing_weights[row * TOP_K..(row + 1) * TOP_K],
            expected.weights,
            "row {row} weights"
        );
    }
}

const REFERENCE_ROWS: usize = 32;
/// Rows whose F64 logit gap among the top 9 is below this are near a tie:
/// their selection or order may legitimately differ under F32 rounding.
const TIE_GAP: f64 = 1e-4;

fn matches_f64_reference(down_t: GgmlType, seed: u64) {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let layer = Layer::random(down_t, seed);
    let weights = layer.gpu(&device, true, &layer.gpu_experts(&device));
    let (x, residual) = rows(REFERENCE_ROWS, seed ^ 0xf64);
    let mut cache = ExpertCache::new(&layer);
    let unit = 2f64.powi(-24);
    let factor = 8.0 * (H + MI) as f64 * unit;
    let (mut worst_ratio, mut max_abs, mut sum_abs, mut near_ties) = (0.0f64, 0.0f64, 0.0f64, 0);
    let all_rows = (0..REFERENCE_ROWS).collect::<Vec<_>>();
    let gpu = run_batched(&device, &mut registry, &weights, &x, &residual, &all_rows);
    for row in 0..REFERENCE_ROWS {
        let reference = cache.reference(&x[row], &residual[row]);
        if reference.gap < TIE_GAP {
            near_ties += 1;
        } else {
            let ids: Vec<usize> = gpu[row].ids.iter().map(|&id| id as usize).collect();
            assert_eq!(ids, reference.ids, "{down_t:?} row {row}: expert ids differ from F64");
        }
        for i in 0..H {
            let err = (gpu[row].out[i] as f64 - reference.out[i]).abs();
            let bound = factor * reference.magnitude[i];
            assert!(
                err <= bound,
                "{down_t:?} row {row} [{i}]: |{} - {}| = {err:e} > {bound:e}",
                gpu[row].out[i],
                reference.out[i]
            );
            worst_ratio = worst_ratio.max(err / reference.magnitude[i]);
            max_abs = max_abs.max(err);
            sum_abs += err;
        }
    }
    eprintln!(
        "moe_batched {down_t:?} vs F64: {REFERENCE_ROWS} rows, ids identical ({near_ties} near-tie rows skipped), \
         max |err| {max_abs:.3e}, mean |err| {:.3e}, worst |err|/S {worst_ratio:.3e} (bound {factor:.3e})",
        sum_abs / (REFERENCE_ROWS * H) as f64
    );
}

#[test]
fn batched_route_matches_f64_reference_down_q6_k() {
    matches_f64_reference(GgmlType::Q6_K, 0x64_06);
}

#[test]
fn batched_route_matches_f64_reference_down_q4_k() {
    matches_f64_reference(GgmlType::Q4_K, 0x64_04);
}

fn bf16_round(v: f32) -> f32 {
    half::bf16::from_f32(v).to_f32()
}

fn old_route_vs_new(down_t: GgmlType, seed: u64) {
    const ROWS: usize = 64;
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let layer = Layer::random(down_t, seed);
    let experts = layer.gpu_experts(&device);
    let new_weights = layer.gpu(&device, true, &experts);
    let old_weights = layer.gpu(&device, false, &experts);
    let router_bf16: Vec<f32> = layer.router.iter().map(|&v| bf16_round(v)).collect();
    let (x, residual) = rows(ROWS, seed ^ 0x01d);
    let (mut selection_changed, mut set_differs, mut order_differs) = (0, 0, 0);
    let (mut max_same, mut sum_same, mut n_same) = (0.0f64, 0.0f64, 0usize);
    let (mut max_all, mut sum_all) = (0.0f64, 0.0f64);
    let mut max_out = 0.0f64;
    for chunk in (0..ROWS).collect::<Vec<_>>().chunks(4) {
        let new = run_batched(&device, &mut registry, &new_weights, &x, &residual, chunk);
        for (slot, &row) in chunk.iter().enumerate() {
            let old_out = run_serial_builder(&device, &mut registry, &old_weights, &x[row], &residual[row]);
            let old_ids = old_route_ids(&device, &mut registry, &old_weights, &x[row]);
            let (f32_sel, _, f32_gap) = top_k_f64(&router_logits_f64(&layer.router, &x[row]));
            let (bf16_sel, _, bf16_gap) = top_k_f64(&router_logits_f64(&router_bf16, &x[row]));
            let mut f32_set = f32_sel.clone();
            f32_set.sort_unstable();
            let mut bf16_set = bf16_sel.clone();
            bf16_set.sort_unstable();
            let bf16_changes = f32_set != bf16_set;
            selection_changed += usize::from(bf16_changes);
            let mut old_set = old_ids.clone();
            old_set.sort_unstable();
            let mut new_set = new[slot].ids.clone();
            new_set.sort_unstable();
            set_differs += usize::from(old_set != new_set);
            order_differs += usize::from(old_ids != new[slot].ids);
            if !bf16_changes && f32_gap >= TIE_GAP && bf16_gap >= TIE_GAP {
                assert_eq!(
                    old_set, new_set,
                    "{down_t:?} row {row}: old and new routes select different experts \
                     although BF16 rounding of the router does not change the F64 selection"
                );
            }
            let diff = old_out
                .iter()
                .zip(&new[slot].out)
                .map(|(&a, &b)| (a as f64 - b as f64).abs());
            let (row_max, row_sum) = diff.fold((0.0f64, 0.0f64), |(m, s), d| (m.max(d), s + d));
            max_out = max_out.max(new[slot].out.iter().map(|v| v.abs() as f64).fold(0.0, f64::max));
            max_all = max_all.max(row_max);
            sum_all += row_sum;
            if old_set == new_set {
                max_same = max_same.max(row_max);
                sum_same += row_sum;
                n_same += 1;
            }
        }
    }
    eprintln!(
        "moe_batched {down_t:?} old (BF16 router/shared, per row) vs new: {ROWS} rows; \
         F64 selection changed by BF16 router rounding in {selection_changed} rows; \
         GPU expert sets differ in {set_differs} rows, ordered ids differ in {order_differs} rows; \
         same-set rows {n_same}: max |diff| {max_same:.3e}, mean |diff| {:.3e}; \
         all rows: max |diff| {max_all:.3e}, mean |diff| {:.3e}; max |out| {max_out:.3e}",
        sum_same / (n_same.max(1) * H) as f64,
        sum_all / (ROWS * H) as f64
    );
    assert!(n_same > 0, "no row kept its expert set");
    // BF16 rounding of the shared expert and router (2^-9 relative) bounds the
    // difference of a row that keeps its experts well below 1% of the output.
    assert!(
        max_same <= 1e-2 * max_out,
        "{down_t:?}: same-expert rows differ by {max_same:e} (max |out| {max_out:e})"
    );
}

#[test]
fn old_bf16_route_vs_batched_route_down_q6_k() {
    old_route_vs_new(GgmlType::Q6_K, 0x01d_06);
}

#[test]
fn old_bf16_route_vs_batched_route_down_q4_k() {
    old_route_vs_new(GgmlType::Q4_K, 0x01d_04);
}
