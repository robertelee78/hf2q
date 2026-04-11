//! Fused MoE expert matmul — ports llama.cpp `kernel_mul_mv_id_*` dispatch.
//!
//! # What this module does
//!
//! Provides a single Rust wrapper `call_quantized_matmul_mv_id_t` that
//! dispatches a **batched, index-driven** quantized vector×matrix multiply
//! across all (token, expert-slot) pairs in one Metal encoder call. The
//! underlying Metal kernel already lives inside candle's pre-compiled
//! `Source::Quantized` library (`kernel_mul_mv_id_q6_K_f32`,
//! `kernel_mul_mv_id_q8_0_f32`, etc.) — we do NOT fork candle. We only load
//! the pipeline by symbol name and bind buffers in the order candle's
//! vendored `kernel_mul_mv_id` template expects.
//!
//! # Reference citations (ADR-005 Walk discipline)
//!
//! - **llama.cpp `kernel_mul_mv_id` template** (older version, the one candle
//!   vendored): `/opt/candle/candle-metal-kernels/src/metal_src/quantized.metal:7544-7618`.
//!   Argument layout and dispatch semantics are ported from this exact file.
//! - **llama.cpp host caller** (for the kargs struct layout that motivates
//!   `nei0`/`nei1`/`nbi1`): `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2393-2414`.
//! - **llama.cpp kargs struct**: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h:512-533`
//!   (`ggml_metal_kargs_mul_mv_id`).
//! - **candle non-id wrapper we mirror** (Rust-side dispatch pattern): the
//!   existing `call_quantized_matmul_mv_t` at
//!   `/opt/candle/candle-metal-kernels/src/kernels/quantized.rs:25-176`. Our
//!   threadgroup dimensions (`nth0`, `nth1`, `align`) are copied verbatim from
//!   that function's `match dtype { ... }` arms.
//! - **mlx-lm cross-reference** (shape of the dispatch we expect):
//!   `/Users/robert/.pyenv/versions/3.13.12/lib/python3.13/site-packages/mlx_lm/models/switch_layers.py:76-90`
//!   (`mx.gather_qmm`).
//!
//! # Kernel binding order (from candle's vendored template)
//!
//! ```text
//! kernel void kernel_mul_mv_id(
//!     device const char * src0s,    // buffer 0 — 3D weight, contiguous [num_experts, n, k]
//!     device const char * src1,     // buffer 1 — flat input [n_tokens, k] (f32)
//!     device       float* dst,      // buffer 2 — output  [n_tokens, n_expert_used, n] (f32)
//!     device const char * ids,      // buffer 3 — [n_tokens, n_expert_used] (i32)
//!     constant int64_t  & nei0,     // n_expert_used
//!     constant int64_t  & nei1,     // n_tokens
//!     constant uint64_t & nbi1,     // row stride of ids, in bytes = n_expert_used * 4
//!     constant int64_t  & ne00,     // k (inner dim)
//!     constant int64_t  & ne01,     // n (output dim of one expert)
//!     constant int64_t  & ne02,     // num_experts
//!     constant uint64_t & nb00,     // unused by q6_K/q8_0 impl — pass 0
//!     constant uint64_t & nb01,     // unused by q6_K/q8_0 impl — pass 0
//!     constant uint64_t & nb02,     // bytes per expert weight = (n*k/block_size)*type_size
//!     constant int64_t  & ne10,     // k (input inner dim) — equal to ne00
//!     constant int64_t  & ne11,     // 1 (single-row per token slot)
//!     constant int64_t  & ne12,     // 1
//!     constant int64_t  & ne13,     // 1
//!     constant uint64_t & nb10,     // sizeof(f32) = 4
//!     constant uint64_t & nb11,     // unused — pass 0 (ne11=1 ⇒ i11=0)
//!     constant uint64_t & nb12,     // bytes per token row in input = k * 4
//!     constant int64_t  & ne0,      // n (output feature dim)
//!     constant int64_t  & ne1,      // 1
//!     constant uint64_t & nb1,      // unused by impl — pass n*4 for documentation
//!     threadgroup int8_t * shared_values,  // unused by q6_K/q8_0 impls
//!     ...);
//! ```
//!
//! # Output layout
//!
//! The kernel computes, for each `(iid1 = token_idx, idx = expert_slot)`:
//!   `expert_id = ids[iid1 * n_expert_used + idx]   (i32)`
//!   `W_e      = src0s + expert_id * nb02`         (quantized weight rows)
//!   `x_tok    = src1  + iid1 * nb12`              (input row, f32)
//!   `y_out    = dst + iid1 * n_expert_used * n + idx * n`
//!   `y_out[row] = <W_e[row], x_tok>`              (vector-row dot product)
//!
//! i.e. **dst is `[n_tokens, n_expert_used, n]` row-major in f32**.
//!
//! # What this does NOT do
//!
//! - No per-expert scale multiply (the MoeBlock layer adds that as a
//!   scalar broadcast_mul after the dispatch).
//! - No router weight multiply (same — layer-side operation).
//! - No GELU / SwiGLU (same — layer-side operation).
//! - No softmax / top-k (those stayed on the GPU in Phase 1 via
//!   `arg_sort_last_dim`; this file does not touch them).
//!
//! Everything above is intentional: keep the kernel wrapper as a minimal
//! faithful port of the llama.cpp primitive and build the MoE semantics
//! in `gemma4.rs` on top of it using candle Tensor ops.

use candle_metal_kernels::metal::{Buffer, ComputeCommandEncoder, Device};
use candle_metal_kernels::source::Source;
use candle_metal_kernels::utils::EncoderProvider;
use candle_metal_kernels::{set_params, GgmlDType, Kernels, MetalKernelError};
use objc2_metal::{MTLResourceUsage, MTLSize};

/// Shape describing the 3D quantized weight and the 2D input that feed a
/// fused mul_mv_id dispatch. Named fields because seven-tuple call sites
/// are a foot-gun.
///
/// `dead_code` is allowed pending Phase B (wiring into `gemma4.rs`); the
/// struct is fully exercised by the Phase A unit tests under `#[cfg(test)]`.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct MoeDispatchShape {
    /// Number of experts in the full weight tensor (src0 is [num_experts, n, k]).
    pub num_experts: usize,
    /// Output feature dim per expert (rows of a single expert's 2D weight).
    pub n: usize,
    /// Input feature dim (columns of a single expert's 2D weight).
    pub k: usize,
    /// Number of tokens in the input (rows of src1).
    pub n_tokens: usize,
    /// Number of experts each token is routed to (top-k).
    pub n_expert_used: usize,
}

/// Per-quant-type bytes-per-expert-weight. Computed from candle's
/// `GgmlDType::type_size() / block_size()` but without depending on those
/// private helpers — we hardcode the two types we actually ship with
/// Gemma 4 (`Q6_K` for gate_up, `Q8_0` for down). Adding a new quant type
/// here is a 1-line change, keyed by Walk discipline.
#[allow(dead_code)]
fn bytes_per_expert(dtype: GgmlDType, n: usize, k: usize) -> usize {
    // row_elems * (type_size / block_size); multiply by n rows.
    // Values cross-checked against candle-core's k_quants.rs compile-time
    // asserts (lines 63-170) and llama.cpp's ggml-common.h — the two
    // references agree by construction since candle vendored from ggml.
    let (block_size, type_size) = match dtype {
        // k_quants.rs:63 — `sizeof(BlockQ4_0) == 18` (QK=32)
        GgmlDType::Q4_0 => (32, 18),
        // k_quants.rs:72 — `sizeof(BlockQ4_1) == 20`
        GgmlDType::Q4_1 => (32, 20),
        // k_quants.rs:81 — `sizeof(BlockQ5_0) == 22`
        GgmlDType::Q5_0 => (32, 22),
        // k_quants.rs:91 — `sizeof(BlockQ5_1) == 24`
        GgmlDType::Q5_1 => (32, 24),
        // k_quants.rs:99 — `sizeof(BlockQ8_0) == 34`
        GgmlDType::Q8_0 => (32, 34),
        // k_quants.rs:108 — `sizeof(BlockQ8_1) == 36`
        GgmlDType::Q8_1 => (32, 36),
        // k_quants.rs:118 — Q2K = QK_K/16 + QK_K/4 + 4  (QK_K=256)
        GgmlDType::Q2K => (256, 84),
        // k_quants.rs:128 — Q3K = QK_K/8 + QK_K/4 + 12 + 2
        GgmlDType::Q3K => (256, 110),
        // k_quants.rs:139 — Q4K = QK_K/2 + K_SCALE_SIZE(12) + 4
        GgmlDType::Q4K => (256, 144),
        // k_quants.rs:151 — Q5K = QK_K/8 + QK_K/2 + 4 + K_SCALE_SIZE(12)
        GgmlDType::Q5K => (256, 176),
        // k_quants.rs:161 — Q6K = 3*QK_K/4 + QK_K/16 + 2
        GgmlDType::Q6K => (256, 210),
        // k_quants.rs:170 — Q8K = 4 + QK_K + 2*QK_K/16
        GgmlDType::Q8K => (256, 292),
        other => {
            // No kernel_mul_mv_id_* symbol exists for F16/BF16/F32 in
            // candle's vendored quantized.metal (only the quant block
            // variants are templated). Per Anti-Goal #7 (no stubs),
            // refuse rather than guess.
            panic!(
                "moe_kernel::bytes_per_expert: dtype {other:?} not supported. \
                 Add the (block_size, type_size) pair with a k_quants.rs citation \
                 before dispatching."
            );
        }
    };
    let total_elems = n * k;
    debug_assert!(
        total_elems % block_size == 0,
        "MoE weight shape {n}x{k} not divisible by block size {block_size} for {dtype:?}"
    );
    (total_elems / block_size) * type_size
}

/// Pick the threadgroup dimensions for a given quant type. Copied verbatim
/// from candle's `call_quantized_matmul_mv_t` (`quantized.rs:61-112`) — the
/// *id* kernel wraps the same `kernel_mul_mv_q*_f32_impl` functions as the
/// non-id path, so the thread layout is identical.
#[allow(dead_code)]
fn dispatch_dims(dtype: GgmlDType) -> (usize, usize, usize) {
    match dtype {
        GgmlDType::Q4_0
        | GgmlDType::Q4_1
        | GgmlDType::Q5_0
        | GgmlDType::Q5_1
        | GgmlDType::Q8_0
        | GgmlDType::Q8_1 => (8, 8, 8),
        GgmlDType::Q2K => (2, 32, 4),
        GgmlDType::Q4K => (4, 8, 4),
        GgmlDType::Q3K | GgmlDType::Q5K => (2, 32, 4),
        GgmlDType::Q6K => (2, 32, 2),
        // The id template currently supports float and k-quant variants; the
        // f16/bf16/f32 id kernels use (32, 1, 8) in candle's non-id path.
        GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::Q8K | GgmlDType::F32 => (32, 1, 8),
    }
}

/// Map a quant type to its compiled `kernel_mul_mv_id_*_f32` Metal symbol
/// name. The symbols all live in candle's `Source::Quantized` pre-compiled
/// library (verified via grep for `host_name("kernel_mul_mv_id_*)` in
/// `candle-metal-kernels/src/metal_src/quantized.metal:7622-7642`).
#[allow(dead_code)]
fn kernel_name(dtype: GgmlDType) -> &'static str {
    match dtype {
        GgmlDType::Q4_0 => "kernel_mul_mv_id_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mv_id_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mv_id_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mv_id_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mv_id_q8_0_f32",
        GgmlDType::Q2K => "kernel_mul_mv_id_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mv_id_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mv_id_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mv_id_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mv_id_q6_K_f32",
        GgmlDType::F16 => "kernel_mul_mv_id_f16_f32",
        GgmlDType::F32 => "kernel_mul_mv_id_f32_f32",
        GgmlDType::Q8_1 | GgmlDType::Q8K | GgmlDType::BF16 => panic!(
            "moe_kernel: dtype {dtype:?} does not have a kernel_mul_mv_id_* symbol in \
             candle's quantized.metal library; reject rather than dispatch a missing kernel"
        ),
    }
}

#[allow(dead_code)]
fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Dispatch a single fused `kernel_mul_mv_id_*_f32` over all (token, expert-slot)
/// pairs described by `shape`.
///
/// # Arguments
/// - `device`, `ep`, `kernels`: the candle-metal plumbing, same as the
///   non-id variant in `candle-metal-kernels`.
/// - `dtype`: the GGML quant type of the weight buffer.
/// - `shape`: see [`MoeDispatchShape`].
/// - `src0s`: the 3D weight buffer, byte-contiguous per expert.
/// - `src1`:  the flat input buffer `[n_tokens, k]` in f32.
/// - `src1_offset`: byte offset into `src1` (usually 0).
/// - `ids`:   the per-token top-k expert index buffer. **Row-major
///   `[n_tokens, n_expert_used]`**, element dtype `i32` (or bitwise-equivalent
///   `u32` — the kernel reads 32 bits unsigned values as signed, which is safe
///   for `expert_id ∈ [0, num_experts)`). `nbi1` is hardcoded to
///   `n_expert_used * 4`, matching candle's `arg_sort_last_dim` U32 output.
/// - `dst`:   the output buffer `[n_tokens, n_expert_used, n]` in f32.
/// - `dst_offset`: byte offset into `dst` (usually 0).
///
/// # Errors
/// Returns `MetalKernelError::LoadPipelineError` if the kernel symbol is not
/// present in `Source::Quantized` (should never happen for supported dtypes
/// since candle 0.10.2 compiles all `kernel_mul_mv_id_*_f32` templates).
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn call_quantized_matmul_mv_id_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    shape: MoeDispatchShape,
    src0s: &Buffer,
    src1: &Buffer,
    src1_offset: usize,
    ids: &Buffer,
    dst: &Buffer,
    dst_offset: usize,
) -> Result<(), MetalKernelError> {
    let MoeDispatchShape {
        num_experts,
        n,
        k,
        n_tokens,
        n_expert_used,
    } = shape;

    // --- Kargs for kernel_mul_mv_id (see citation at top of file) -----------
    let nei0: i64 = n_expert_used as i64;
    let nei1: i64 = n_tokens as i64;
    let nbi1: u64 = (n_expert_used as u64) * 4; // i32/u32 row stride in bytes

    let ne00: i64 = k as i64;
    let ne01: i64 = n as i64;
    let ne02: i64 = num_experts as i64;
    let nb00: u64 = 0;
    let nb01: u64 = 0;
    let nb02: u64 = bytes_per_expert(dtype, n, k) as u64;

    let ne10: i64 = k as i64;
    let ne11: i64 = 1;
    let ne12: i64 = 1;
    let ne13: i64 = 1;
    let nb10: u64 = 4; // f32 element stride
    let nb11: u64 = 0;
    let nb12: u64 = (k as u64) * 4; // token row stride in bytes

    let ne0: i64 = n as i64;
    // ne1 is used by the *outer* `kernel_mul_mv_id` template to index the
    // destination buffer: `dst_cur = dst + i1*ne0 + i2*ne1*ne0` where
    // `i1 = slot` and `i2 = token`. For a `[n_tokens, n_expert_used, n]`
    // row-major output we need `ne1 = n_expert_used` so that successive
    // tokens are `n_expert_used * n` elements apart. The `impl_fn` inside
    // the template separately hardcodes its own local `ne1 = 1` (see
    // quantized.metal:7609), so the inner per-row write is unaffected.
    // llama.cpp's kargs struct matches: `ne1 = ggml_tensor->ne[1]` for
    // `mul_mat_id` which is `n_expert_used` (ggml-metal-ops.cpp:2365).
    let ne1: i64 = n_expert_used as i64;
    let nb1: u64 = (n as u64) * 4;

    // --- Pipeline + grid/threadgroup shape ----------------------------------
    let pipeline = kernels.load_pipeline(device, Source::Quantized, kernel_name(dtype))?;
    let (nth0, nth1, align) = dispatch_dims(dtype);

    // Grid: sweep `ne01` output rows in groups of `align`, single-row per
    // token slot (ne11=1), depth = n_tokens * n_expert_used (one per pair).
    let grid = MTLSize {
        width: ceil_div(n, align),
        height: 1,
        depth: (n_tokens * n_expert_used),
    };
    let tg = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Parameter binding order matches candle's vendored kernel_mul_mv_id
    // template signature (quantized.metal:7545-7573). Buffers are bound at
    // positions 0/1/2/3 via set_params!'s (buf) / (buf, offset) patterns.
    set_params!(
        encoder,
        (
            src0s,
            (src1, src1_offset),
            (dst, dst_offset),
            ids,
            nei0,
            nei1,
            nbi1,
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            ne13,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            nb1
        )
    );

    encoder.use_resource(src0s, MTLResourceUsage::Read);
    encoder.use_resource(src1, MTLResourceUsage::Read);
    encoder.use_resource(ids, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);

    // kernel_mul_mv_id declares `threadgroup int8_t * shared_values
    // [[threadgroup(0)]]`, but the Q6_K / Q8_0 `impl_fn`s we wrap pass
    // `nullptr` / never dereference it. Metal tolerates a zero-size
    // threadgroup binding in that case (the same is true in the non-id
    // `call_quantized_matmul_mv_t`, which also does not call
    // `set_threadgroup_memory_length`).
    encoder.dispatch_thread_groups(grid, tg);
    Ok(())
}

// ---------------------------------------------------------------------------
// Phase A unit test — numerical parity vs the loop baseline
// ---------------------------------------------------------------------------
//
// Builds a small synthetic 3D Q6_K weight on the Metal device, dispatches
// the fused kernel for (n_tokens=4, n_expert_used=2), then compares the
// f32 output element-by-element against the reference computed by calling
// the existing `QMatMul::forward` eight times (one per routed expert slot).
//
// The test is skipped when no Metal device is available (e.g. CI runners
// without GPU access). Build with `--features metal` and run with
// `cargo test --features metal --release -p hf2q -- moe_kernel::tests --nocapture`.
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::op::BackpropOp;
    use candle_core::quantized::{GgmlDType as CoreGgml, QMatMul, QStorage, QTensor};
    use candle_core::{DType, Device as CoreDevice, Module, Tensor};
    use std::sync::Arc;

    /// Deterministic pseudo-random f32 tensor — avoids a test dep on `rand`.
    /// Uses a splitmix64-style mixer; the exact values don't matter as long
    /// as they are non-trivial and reproducible.
    fn make_f32_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..len)
            .map(|_| {
                s = s.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^= z >> 31;
                // map to a well-behaved range so Q6_K quantization doesn't
                // saturate on outliers
                ((z as i64 as f64) * (1.0 / (i64::MAX as f64)) * 0.25) as f32
            })
            .collect()
    }

    /// Shared Phase-A numerical parity harness. Builds a synthetic 3D
    /// quantized weight of the given `dtype`, routes `n_tokens × top_k`
    /// expert slots through the fused kernel, and compares against the
    /// reference built from `top_k*n_tokens` separate `QMatMul::forward`
    /// calls. Returns the elementwise L_inf delta.
    fn run_parity(dtype_pair: (CoreGgml, GgmlDType), shape_override: Option<(usize, usize)>) {
        let (core_dtype, kernel_dtype) = dtype_pair;

        let device = match CoreDevice::new_metal(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip: metal device unavailable ({e})");
                return;
            }
        };

        // Small but realistic shape. Use multiples of QK_K=256 so k-quants
        // can quantize without padding. The `n` must be a multiple of the
        // per-dtype `align` (2 for Q6_K, 8 for Q8_0) so the dispatch grid
        // is exact. Use `shape_override` to exercise a non-default path.
        let num_experts: usize = 4;
        let (n, k): (usize, usize) = shape_override.unwrap_or((512, 256));
        let n_tokens: usize = 4;
        let top_k: usize = 2;

        // --- Build the 3D quantized weight from f32 and store as QTensor -
        // Quantize each expert separately with candle's public API, then
        // concatenate the raw quantized bytes into a single 3D buffer, the
        // exact layout the fused kernel expects.
        let per_expert_bytes = bytes_per_expert(kernel_dtype, n, k);
        let mut combined_bytes: Vec<u8> = Vec::with_capacity(num_experts * per_expert_bytes);

        // Keep both a reference Vec<Arc<QTensor>> of 2D per-expert tensors
        // (for the loop baseline) and the flat 3D byte buffer (for the
        // fused dispatch).
        let mut expert_qmatmuls: Vec<QMatMul> = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let f32_expert = make_f32_vec(n * k, 0xCAFE_F00D + (e as u64) * 0xDEAD_BEEF);
            let t = Tensor::from_vec(f32_expert, (n, k), &device).unwrap();
            let qt = QTensor::quantize(&t, core_dtype).unwrap();
            let bytes = qt.data().unwrap();
            assert_eq!(bytes.len(), per_expert_bytes);
            combined_bytes.extend_from_slice(&bytes);
            expert_qmatmuls.push(QMatMul::from_arc(Arc::new(qt)).unwrap());
        }
        assert_eq!(combined_bytes.len(), num_experts * per_expert_bytes);

        // Build a 3D QStorage::Metal holding the concatenated bytes.
        let qstorage_3d = QStorage::from_data(
            std::borrow::Cow::Borrowed(&combined_bytes),
            &device,
            core_dtype,
        )
        .unwrap();
        let QStorage::Metal(metal_weight) = &qstorage_3d else {
            panic!("metal device must yield a metal QStorage");
        };

        // --- Input tokens and ids -----------------------------------------
        let x_f32 = make_f32_vec(n_tokens * k, 0x1234_5678);
        let x_tensor = Tensor::from_vec(x_f32.clone(), (n_tokens, k), &device).unwrap();

        // ids: [n_tokens, top_k] as U32 (byte-identical to i32 for small
        // positive values), deterministically chosen so every (token, slot)
        // picks a different expert to exercise expert pointer math.
        let ids_vec: Vec<u32> = (0..n_tokens)
            .flat_map(|t| (0..top_k).map(move |k| ((t + k) % num_experts) as u32))
            .collect();
        let ids_tensor =
            Tensor::from_vec(ids_vec.clone(), (n_tokens, top_k), &device).unwrap();

        // --- Reference: 8 separate QMatMul::forward calls -----------------
        // For each (token, slot), take the per-expert 2D QMatMul, feed it
        // the single-row f32 input, and collect the N-element row into a
        // reference buffer of size [n_tokens, top_k, n].
        let mut reference = vec![0.0f32; n_tokens * top_k * n];
        for t in 0..n_tokens {
            let token_row = x_tensor.narrow(0, t, 1).unwrap(); // [1, k]
            for s in 0..top_k {
                let expert_id = ids_vec[t * top_k + s] as usize;
                let out = expert_qmatmuls[expert_id].forward(&token_row).unwrap();
                let out = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                assert_eq!(out.len(), n);
                let base = (t * top_k + s) * n;
                reference[base..base + n].copy_from_slice(&out);
            }
        }

        // --- Fused dispatch ------------------------------------------------
        // Allocate an output buffer of the right size, grab the Metal device
        // plumbing, and invoke `call_quantized_matmul_mv_id_t`. Afterwards
        // flush the encoder so we can read the buffer back CPU-side.
        let CoreDevice::Metal(metal_device) = &device else {
            panic!("device must be Metal");
        };

        // Pull the underlying Metal buffers for input and ids via
        // storage_and_layout (public candle API, no patching required).
        let (x_storage, _x_layout) = x_tensor.storage_and_layout();
        let candle_core::Storage::Metal(x_metal) = &*x_storage else {
            panic!("x_tensor must live on Metal")
        };
        let src1_buf = x_metal.buffer().clone();

        let ids_tensor_contig = ids_tensor.contiguous().unwrap();
        let (ids_storage, _ids_layout) = ids_tensor_contig.storage_and_layout();
        let candle_core::Storage::Metal(ids_metal) = &*ids_storage else {
            panic!("ids tensor must live on Metal")
        };
        let ids_buf = ids_metal.buffer().clone();

        let out_elems = n_tokens * top_k * n;
        let dst_buf = metal_device
            .new_buffer(out_elems, DType::F32, "moe_kernel_test_dst")
            .unwrap();

        {
            let encoder = metal_device.command_encoder().unwrap();
            call_quantized_matmul_mv_id_t(
                metal_device.device(),
                &encoder,
                metal_device.kernels(),
                kernel_dtype,
                MoeDispatchShape {
                    num_experts,
                    n,
                    k,
                    n_tokens,
                    n_expert_used: top_k,
                },
                metal_weight.buffer(),
                &src1_buf,
                0,
                &ids_buf,
                &dst_buf,
                0,
            )
            .unwrap();
            // Encoder is scoped inside this block so its `Drop` impl fires
            // `endEncoding` at the closing brace. That releases the encoder
            // lock on the command buffer, allowing the subsequent readback
            // blit inside `to_vec1::<f32>()` to acquire its own encoder
            // without deadlocking. We must NOT call `encoder.end_encoding()`
            // explicitly here — Metal asserts if `endEncoding` is called
            // twice, and Drop always fires.
        }

        // Wrap the destination buffer in a candle MetalStorage/Tensor via
        // the public `Tensor::from_storage` hatch so we can use the existing
        // async-GPU-to-CPU path; this also forces the command buffer to
        // flush before we read.
        let out_storage = candle_core::MetalStorage::new(
            dst_buf,
            metal_device.clone(),
            out_elems,
            DType::F32,
        );
        let out_tensor = Tensor::from_storage(
            candle_core::Storage::Metal(out_storage),
            (n_tokens, top_k, n),
            BackpropOp::none(),
            false,
        );
        let fused = out_tensor
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(fused.len(), reference.len());

        // --- Compare elementwise at ε=1e-5 --------------------------------
        let mut max_abs: f32 = 0.0;
        let mut max_idx: usize = 0;
        for (i, (a, b)) in fused.iter().zip(reference.iter()).enumerate() {
            let d: f32 = (*a - *b).abs();
            if d > max_abs {
                max_abs = d;
                max_idx = i;
            }
        }
        eprintln!(
            "fused_mv_id {kernel_dtype:?}: shape=({num_experts}x{n}x{k}), \
             n_tokens={n_tokens}, top_k={top_k}, \
             max|Δ|={max_abs:.3e} at idx {max_idx} \
             (fused={}, ref={})",
            fused[max_idx], reference[max_idx]
        );
        assert!(
            max_abs < 1e-5,
            "fused {kernel_dtype:?} kernel disagrees with QMatMul loop at ε=1e-5; max|Δ|={max_abs}"
        );
    }

    #[test]
    fn fused_mv_id_matches_loop_baseline_q6k() {
        run_parity((CoreGgml::Q6K, GgmlDType::Q6K), None);
    }

    #[test]
    fn fused_mv_id_matches_loop_baseline_q8_0() {
        // Q8_0 uses `align=8` so `n` must be a multiple of 8. 512 works.
        run_parity((CoreGgml::Q8_0, GgmlDType::Q8_0), None);
    }

    /// Exercise the non-decode shape — prefill-size n_tokens — to satisfy
    /// Anti-Goal #6 (no benchmark tuning): the same code path must handle
    /// arbitrary `n_tokens`. Uses the larger Q6_K shape Gemma 4 experts
    /// actually run on (`[num_experts, moe_intermediate*2, hidden_size]`)
    /// is too large for a unit test, so we scale down proportionally
    /// (n*2 for the gate_up double-output) while keeping n divisible by
    /// QK_K=256 to avoid quantizer padding artifacts.
    #[test]
    fn fused_mv_id_handles_multi_token_prefill_shape_q6k() {
        run_parity(
            (CoreGgml::Q6K, GgmlDType::Q6K),
            Some((1024, 512)), // mimics a bigger expert; still small enough to quantize quickly
        );
    }
}
