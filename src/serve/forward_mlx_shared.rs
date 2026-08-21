//! Shared quantized-weight primitives for all architecture forward passes.
//!
//! ADR-038 Step 1: extracted from `src/serve/forward_mlx.rs` to make
//! shared types available to both Gemma 4 and Qwen 3.5 paths without
//! coupling either to the other's module tree.
//!
//! External callers continue to resolve `crate::serve::forward_mlx_shared::X`
//! via the `pub use` shim in `forward_mlx.rs`; remove that shim after
//! ADR-038 Step 3 retires `forward_mlx.rs`.

use crate::core::mlx_safetensors_loader::MlxAffineLinear;
use crate::quantize::imatrix::{intercept_qmatmul_with_hint, ImatrixHint};
use crate::serve::gpu::QuantWeightInfo;
use anyhow::Result;
use mlx_native::ops::dense_gemm::DenseGemmF16Params;
use mlx_native::{GgmlQuantizedMatmulParams, GraphSession, MlxBuffer, MlxDevice};

// ---------------------------------------------------------------------------
// Cluster 1 — Quantized weight types
// ---------------------------------------------------------------------------

/// ADR-020 AC#5 Iter B — extra metadata + buffers for an mlx-affine
/// (DWQ) packed-U32 weight.  When this is `Some`, the parent
/// [`MlxQWeight`]'s `buffer` is interpreted as packed-U32 affine-quant
/// codes (shape `[N, K/pack_factor]`, dtype `U32`) instead of GGML
/// block-format bytes; `info.ggml_dtype` is unused on the affine path.
///
/// `scales` and `biases` are the per-group `(s, b)` pairs from the
/// DWQ-trained safetensors.  Held as F32 buffers (cast at load time
/// from BF16/F32 depending on the on-disk safetensors dtype) so the
/// kernel can read them without an inline cast.
pub struct MlxAffineExtra {
    /// Per-group scales, F32, shape `[N, K/group_size]`.
    pub scales: MlxBuffer,
    /// Per-group biases (zero-points), F32, shape `[N, K/group_size]`.
    pub biases: MlxBuffer,
    /// Quantization bit-width (currently only 4 supported).
    pub bits: u32,
    /// Per-group axis length (currently only 32 supported via `simd4_b4`).
    pub group_size: u32,
}

/// Pre-loaded quantized weight buffer paired with its GGML metadata
/// (or its mlx-affine metadata, when `affine` is `Some`).
pub struct MlxQWeight {
    pub buffer: MlxBuffer,
    pub info: QuantWeightInfo,
    /// `Some(...)` when this weight was loaded from a DWQ-trained mlx
    /// safetensors overlay.  `None` for the default GGML-block-loaded
    /// path.  Routing in `dispatch_qmatmul` checks `affine.is_some()`
    /// FIRST and skips both the F32 and GGML branches when set.
    pub affine: Option<MlxAffineExtra>,
    /// ADR-029 iter-28 H29 — F16 pre-dequantized shadow.  When `Some`,
    /// `dispatch_qmatmul` at m > MM_ROUTING_THRESHOLD routes through
    /// `kernel_mul_mm_f16_f32_*` (peer's gemma4 pattern) instead of
    /// per-call dequant inside `kernel_mul_mm_<qtype>_tensor_f32`.
    ///
    /// Materialized at load via `dispatch_dequant_to_f16` when the
    /// `HF2Q_F16_SHADOW=1` env gate is set and the weight is a quantized
    /// type the dequant kernel supports (Q4_0/Q8_0/Q5_1/IQ4_NL/Q4_K/
    /// Q5_K/Q6_K).  ~1 GB extra resident on gemma4-26B; M5 Max's 128 GB
    /// unified memory accommodates this without pressure.
    ///
    /// Default OFF until coherence + multi-regime bench parity proven.
    pub f16_shadow: Option<MlxBuffer>,
    /// ADR-029 iter-175 Step 1d — pre-baked dispatch record for the
    /// Q6_K NR2 decode-m=1 mat-vec hot path.  Lazy-init on the first
    /// `dispatch_qmatmul` call (held in a `OnceLock` so the bake cost
    /// is amortized across the model's lifetime).
    ///
    /// `OnceLock<Option<DispatchRecord>>` encodes three states:
    ///   - `OnceLock` empty: not yet attempted (will try on first call)
    ///   - `Some(None)`: bake attempted but not applicable (wrong dtype,
    ///     env-flag off, etc.); caller falls back to the unbaked path
    ///   - `Some(Some(rec))`: bake succeeded; caller uses the fast path
    ///
    /// Gated by [`mlx_native::ops::quantized_matmul_ggml::build_q6k_nr2_m1_record`]
    /// — only Q6_K with `HF2Q_Q6K_MV_NR2` truthy is bakeable through
    /// this slot.  Other weight types (Q8_0, Q5_K, Q4_K, F32, affine,
    /// F16 shadow) keep using the legacy dispatch path until additional
    /// bake helpers land in later substeps.
    pub decode_record_q6k_m1: std::sync::OnceLock<Option<mlx_native::DispatchRecord>>,
}

impl MlxQWeight {
    /// Build `GgmlQuantizedMatmulParams` for a mat-vec dispatch.
    ///
    /// `m` is the number of input tokens (1 for decode).
    pub fn matmul_params(&self, m: u32) -> Result<GgmlQuantizedMatmulParams> {
        Ok(GgmlQuantizedMatmulParams {
            m,
            n: self.info.rows as u32,
            k: self.info.cols as u32,
            ggml_type: self.info.ggml_dtype,
        })
    }

    /// AC#5 Iter B — construct an affine-mode `MlxQWeight` from a
    /// loaded `MlxAffineLinear` (the safetensors loader's runtime
    /// representation).  Uploads the packed-U32 weight + F32 scales +
    /// F32 biases to GPU buffers.
    ///
    /// The `info.ggml_dtype` is stamped to `GgmlType::F32` as a sentinel
    /// (unused on the affine path; routing gates on `affine.is_some()`
    /// before the F32 check).  `info.rows = N`, `info.cols = K`.
    pub fn from_mlx_affine_linear(device: &MlxDevice, linear: &MlxAffineLinear) -> Result<Self> {
        if linear.bits != 4 {
            anyhow::bail!(
                "MlxQWeight::from_mlx_affine_linear: only bits=4 supported in AC#5 Iter B; got {}",
                linear.bits
            );
        }
        if linear.group_size != 32 {
            anyhow::bail!(
                "MlxQWeight::from_mlx_affine_linear: only group_size=32 supported in AC#5 Iter B; got {}",
                linear.group_size
            );
        }
        let n = linear.n;
        let k = linear.k;
        let pack_factor = (32 / linear.bits) as usize;
        if k % pack_factor != 0 {
            anyhow::bail!(
                "MlxQWeight::from_mlx_affine_linear: K ({k}) must be divisible by pack_factor ({pack_factor})"
            );
        }
        let k_packed = k / pack_factor;
        let groups_per_row = k / linear.group_size;

        // Pack the unpacked u8 codes back to U32 mlx-on-disk layout
        // (low nibble at slot 0).  Mirrors mlx/ops.cpp:4762-4772.
        let mut packed = vec![0u32; n * k_packed];
        for row in 0..n {
            for kp in 0..k_packed {
                let mut word: u32 = 0;
                for j in 0..pack_factor {
                    let code = linear.q_int[row * k + kp * pack_factor + j] as u32;
                    debug_assert!(code <= 0xF);
                    word |= (code & 0xF) << (j * 4);
                }
                packed[row * k_packed + kp] = word;
            }
        }

        let mut weight_buf = device
            .alloc_buffer(
                n * k_packed * std::mem::size_of::<u32>(),
                mlx_native::DType::U32,
                vec![n, k_packed],
            )
            .map_err(|e| anyhow::anyhow!("affine weight alloc: {e}"))?;
        weight_buf
            .as_mut_slice::<u32>()
            .map_err(|e| anyhow::anyhow!("affine weight slice: {e}"))?
            .copy_from_slice(&packed);

        let mut scales_buf = device
            .alloc_buffer(
                n * groups_per_row * std::mem::size_of::<f32>(),
                mlx_native::DType::F32,
                vec![n, groups_per_row],
            )
            .map_err(|e| anyhow::anyhow!("affine scales alloc: {e}"))?;
        scales_buf
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("affine scales slice: {e}"))?
            .copy_from_slice(&linear.scales);

        let mut biases_buf = device
            .alloc_buffer(
                n * groups_per_row * std::mem::size_of::<f32>(),
                mlx_native::DType::F32,
                vec![n, groups_per_row],
            )
            .map_err(|e| anyhow::anyhow!("affine biases alloc: {e}"))?;
        biases_buf
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("affine biases slice: {e}"))?
            .copy_from_slice(&linear.biases);

        Ok(Self {
            buffer: weight_buf,
            info: QuantWeightInfo {
                ggml_dtype: mlx_native::GgmlType::F32, // sentinel; affine path bypasses it
                rows: n,
                cols: k,
            },
            affine: Some(MlxAffineExtra {
                scales: scales_buf,
                biases: biases_buf,
                bits: linear.bits,
                group_size: linear.group_size as u32,
            }),
            f16_shadow: None,
            decode_record_q6k_m1: std::sync::OnceLock::new(),
        })
    }
}

/// ADR-020 AC#5 Iter C2.2/C2.3 — stacked mlx-affine MoE expert weights
/// for one (role, layer) tuple.  Each buffer holds the full expert
/// stack in row-major order: `weight[e, n, k_packed]` U32,
/// `scales[e, n, k/group_size]` BF16, `biases[e, n, k/group_size]` BF16.
///
/// Consumed by the MoE-id dispatch path (Iter C2.3) via
/// `mlx_native::quantized_matmul_id_into` (same packed-U32 kernel
/// that mlx-lm uses).  BF16 scales/biases are the kernel's native
/// dtype — F32 input from `MlxAffineLinear::scales`/`biases` is cast
/// at upload time inside `MlxAffineMoeStack::from_per_expert_linears`.
#[derive(Clone)]
pub struct MlxAffineMoeStack {
    /// Packed-U32 weight stack `[n_experts, N, K/pack_factor]`.
    pub weight: MlxBuffer,
    /// BF16 scales stack `[n_experts, N, K/group_size]`.
    pub scales: MlxBuffer,
    /// BF16 biases stack `[n_experts, N, K/group_size]`.
    pub biases: MlxBuffer,
    /// Output dim per expert.
    pub n: usize,
    /// Input dim per expert.
    pub k: usize,
    /// Quant bit-width (4 in Iter C2.x).
    pub bits: u32,
    /// Per-group axis length (32 in Iter C2.x).
    pub group_size: u32,
    /// Number of experts.
    pub num_experts: usize,
}

/// Helper: load a GGUF tensor as raw quantized bytes into an MlxQWeight.
///
/// The tensor name is looked up in the GGUF file, its raw GGML block data
/// is copied into a new MlxBuffer, and the `QuantWeightInfo` is derived
/// from the tensor's metadata (shape and GGML type).
pub(crate) fn load_gguf_qweight(
    gguf: &mlx_native::gguf::GgufFile,
    name: &str,
    device: &MlxDevice,
) -> Result<MlxQWeight> {
    let full_name = if name.ends_with(".weight") {
        name.to_string()
    } else {
        format!("{name}.weight")
    };
    let info = gguf
        .tensor_info(&full_name)
        .ok_or_else(|| crate::serve::load_diagnostic::MissingGgufTensor::new(full_name.clone()))?;
    let buffer = gguf
        .load_tensor(&full_name, device)
        .map_err(|e| anyhow::anyhow!("load {}: {e}", full_name))?;

    // Shape: [rows, cols] for 2D weight matrices.
    let rows = info.shape.first().copied().unwrap_or(1);
    let cols = if info.shape.len() > 1 {
        info.shape[1]
    } else {
        1
    };

    Ok(MlxQWeight {
        buffer,
        info: QuantWeightInfo {
            ggml_dtype: info.ggml_type,
            rows,
            cols,
        },
        affine: None,
        f16_shadow: None,
        decode_record_q6k_m1: std::sync::OnceLock::new(),
    })
}

/// ADR-029 iter-28 H29 — populate the F16 pre-dequantized shadow for a
/// quantized weight.  Returns Ok(()) silently if:
///   * HF2Q_F16_SHADOW is unset / falsy (default), OR
///   * the weight is not a quantized type that dequant_to_f16 supports
///     (F32 / F16 / I16 / affine).
///
/// When ENABLED and the weight type is supported, dispatches the dequant
/// kernel at load time, allocates a 2-bytes/elem F16 buffer, and stores
/// it as the `f16_shadow` field.  Subsequent `dispatch_qmatmul` at m > 8
/// will fast-path through the F16-input matmul kernel (kernel_mul_mm_f16
/// _f32_tensor) which avoids per-call dequant overhead.
///
/// Memory cost: ~1 GB extra resident for gemma4-26B attn weights when
/// applied to attn_q/k/v/output + ffn_gate/up.  On the M5 Max target
/// (128 GB unified), this is well within budget; matches peer's strategy.
pub(crate) fn populate_f16_shadow_if_enabled(
    qweight: &mut MlxQWeight,
    device: &MlxDevice,
    registry: &mut mlx_native::KernelRegistry,
    tensor_name: &str,
) -> Result<()> {
    // iter-31 default-flip: H29 default-true; opt-out via =0/false/off.
    let enabled = std::env::var("HF2Q_F16_SHADOW")
        .ok()
        .map(|v| !matches!(v.as_str(), "0" | "false" | "off"))
        .unwrap_or(true);
    if !enabled {
        return Ok(());
    }

    // Skip when affine — that path doesn't go through dispatch_qmatmul's
    // quantized branch.
    if qweight.affine.is_some() {
        return Ok(());
    }

    // Skip types the dequant kernel doesn't handle.
    use mlx_native::GgmlType;
    match qweight.info.ggml_dtype {
        GgmlType::Q4_0
        | GgmlType::Q8_0
        | GgmlType::Q5_1
        | GgmlType::IQ4_NL
        | GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q6_K => {}
        _ => return Ok(()),
    }

    let n_rows = qweight.info.rows as u32;
    let n_cols = qweight.info.cols as u32;

    let f16 = mlx_native::ops::dequant_to_f16::materialize_f16_shadow(
        device,
        registry,
        &qweight.buffer,
        n_rows,
        n_cols,
        qweight.info.ggml_dtype,
    )
    .map_err(|e| anyhow::anyhow!("F16 shadow materialize for '{}': {e}", tensor_name))?;

    qweight.f16_shadow = Some(f16);
    Ok(())
}

/// ADR-022 P1.9 — APEX-format Gemma4 GGUFs preserve `ffn_gate_inp.weight`
/// (router projection) as F32 for accuracy. mlx-native's
/// `quantized_matmul_ggml` correctly refuses F32 because the GGML block
/// kernels require block-format input; this wrapper routes F32 to the
/// dense F32 matmul kernel that mlx-native already ships.
///
/// Takes `registry` and `device` separately to avoid borrow conflicts on
/// `GpuContext` (registry is `&mut`, device is `&`).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_qmatmul(
    session: &mut GraphSession<'_>,
    registry: &mut mlx_native::KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxQWeight,
    output: &MlxBuffer,
    m: u32,
    imatrix_hint: ImatrixHint<'_>,
) -> Result<()> {
    // ADR-033 §Pi Stage 2 — imatrix collector intercept (inline-hint API).
    //
    // When a Phase-B driver has installed an `ImatrixCollector`, capture
    // the input activations BEFORE the matmul fires. The hint is a
    // cheap inline enum (no thread-local, no allocation) that the
    // intercept formats lazily only when a collector is present.
    //
    // In the production decode path (no collector installed), the
    // intercept adds one `is_active()` load + branch and falls through —
    // the materialize closure (which would `commit_and_wait` and read
    // the input buffer to host) is never invoked.
    //
    // Per ADR-033 §Risk 2 amendment (2026-05-19): Metal-native
    // accumulation is acceptable; we read the input buffer post-sync on
    // this same Metal-backed GraphSession instead of re-running on CPU.
    intercept_qmatmul_with_hint(imatrix_hint, m as usize, weight.info.cols, || {
        // Materialize the input buffer. We must drain any pending GPU
        // work that wrote to `input` before reading it on the host,
        // AND rotate to a fresh CB so subsequent dispatches don't
        // hit Metal's `MTLCommandBufferStatusCommitted` assertion.
        // `commit_wait_and_rotate` is a hard sync — only acceptable
        // because this closure is gated on collector-present.
        if let Err(e) = session.encoder_mut().commit_wait_and_rotate() {
            eprintln!("[hf2q imatrix intercept] commit_wait_and_rotate failed: {e}");
            return None;
        }
        match input.as_slice::<f32>() {
            Ok(slice) => Some(slice.to_vec()),
            Err(e) => {
                eprintln!("[hf2q imatrix intercept] input.as_slice::<f32>() failed: {e}");
                None
            }
        }
    })
    .map_err(|e| anyhow::anyhow!("imatrix intercept: {e}"))?;

    // ADR-029 iter-40 H40 — annotate the next-captured dispatch with this
    // call's exact reads/writes so the graph_opt reorder pass (graph.rs
    // `ComputeGraph::reorder`) can detect non-conflicts between adjacent
    // dispatches (e.g. Q/K/V which write to disjoint outputs from the
    // same input).  Without per-dispatch annotation, the prior
    // `barrier_between` carries the union of write targets and the reorder
    // pass treats the dispatches as conflicting.  The annotation only
    // takes effect in `begin_recorded` mode (HF2Q_GRAPH_OPT_PREFILL=1).
    //
    // ADR-029 iter-175 Step 1c — gate on `is_capturing()`.  Previously
    // this block unconditionally heap-allocated two `Vec<MemRange>` per
    // `dispatch_qmatmul` call (~91/token: QKV proj × 3 × 30 layers +
    // lm_head). `CommandEncoder::set_pending_buffer_ranges` doc:
    // "Only meaningful in capture mode — has no effect on direct-dispatch."
    // In production decode (capture None), the allocated vectors were
    // taken + dropped on the next encode call — ~182 heap allocs/token
    // of pure waste. First substep of multi-week DispatchRecord refactor.
    //
    // Weights are read-only (never mutated post-load), so we annotate
    // reads=[input] only — weight RANGES would force conservative
    // RAW-conflict checks against the load-time dequant pass that ran
    // hours ago (already complete, no live dependency).
    {
        let encoder = session.encoder_mut();
        if encoder.is_capturing() {
            let input_range = {
                let start = input.contents_ptr() as usize;
                (start, start + input.byte_len())
            };
            let output_range = {
                let start = output.contents_ptr() as usize;
                (start, start + output.byte_len())
            };
            encoder.set_pending_buffer_ranges(vec![input_range], vec![output_range]);
        }
    }

    // ADR-020 AC#5 Iter C — affine route MUST be checked first (before
    // F32/GGML).  When `weight.affine` is `Some`, the buffer holds
    // packed-U32 mlx-affine codes and `info.ggml_dtype` is a sentinel.
    if let Some(extra) = weight.affine.as_ref() {
        if extra.bits != 4 || extra.group_size != 32 {
            return Err(anyhow::anyhow!(
                "dispatch_qmatmul affine: only bits=4, group_size=32 supported in AC#5 Iter C; got bits={} gs={}",
                extra.bits,
                extra.group_size,
            ));
        }
        let n = weight.info.rows as u32;
        let k = weight.info.cols as u32;
        // Per-call meta buffer [M, N, K, group_size] — M varies per
        // dispatch (decode m=1 vs prefill m≥2), so it can't be cached
        // on the weight.
        let mut meta = device
            .alloc_buffer(16, mlx_native::DType::U32, vec![4])
            .map_err(|e| anyhow::anyhow!("affine meta alloc: {e}"))?;
        meta.as_mut_slice::<u32>()
            .map_err(|e| anyhow::anyhow!("affine meta slice: {e}"))?
            .copy_from_slice(&[m, n, k, extra.group_size]);
        return mlx_native::ops::qmm_affine::dispatch_qmm_affine_t_packed_simd4_b4(
            session.encoder_mut(),
            registry,
            device.metal_device(),
            input,
            &weight.buffer,
            &extra.scales,
            &extra.biases,
            output,
            &meta,
            m,
            n,
            k,
            extra.group_size,
            extra.bits,
        )
        .map_err(|e| anyhow::anyhow!("qmm_affine_t_packed_simd4_b4 failed: {e}"));
    }

    // ADR-029 iter-28 H29 / iter-30 H29-speed — F16 pre-dequant fast path.
    //
    // When a quantized weight has been pre-dequantized to F16 at load
    // (via populate_f16_shadow_if_enabled under HF2Q_F16_SHADOW=1),
    // route m > MM_ROUTING_THRESHOLD (= 8, prefill) through the V2-tile
    // F16-weight × F32-input mat-mat kernel (`hf2q_mul_mm_tensor_v2_f16`).
    // This is the F16-input analog of the V2 quantized kernel — same
    // 64×128 tile + direct-device B-read — without the per-call dequant
    // work inside the matmul.  Peer's gemma4 strategy.
    //
    // Decode m=1 still routes through the quantized branch below (the
    // dequant cost is amortized over fewer dispatches, and the m=1 mv
    // kernel is bandwidth-bound — no clear win for F16 there).
    if m > mlx_native::ops::quantized_matmul_ggml::MM_ROUTING_THRESHOLD {
        if let Some(ref f16w) = weight.f16_shadow {
            let n = weight.info.rows as u32;
            let k = weight.info.cols as u32;
            return mlx_native::ops::quantized_matmul_ggml::dispatch_mm_v2_f16(
                session.encoder_mut(),
                registry,
                device,
                f16w,
                input,
                output,
                m,
                n,
                k,
            )
            .map_err(|e| anyhow::anyhow!("dispatch_qmatmul F16-shadow V2 path failed: {e}"));
        }
    }

    if weight.info.ggml_dtype == mlx_native::GgmlType::F32 {
        // F32 dense path.  Weight buffer holds [n_rows, k_cols] f32 row-major.
        //
        // ADR-029 iter-18 (H26): for m=1 (decode), the matrix-MATRIX tile
        // kernel `dense_matmul_f32_f32_tensor` wastes 87.5% of its 8x8 SIMD-
        // group-matrix tile (uses 1 row of input out of 8 loaded).  Route
        // m=1 dispatches through `dispatch_dense_matvec_f32` (mat-VECTOR
        // kernel, MTLSize threads=32x2, 4 rows/dst x 2 SGs) which matches
        // peer's `kernel_mul_mv_f32_f32` pattern.  Gemma4 router_proj
        // (ffn_gate_inp F32 [2816,128]) measured ~73 µs/layer via the
        // mat-mat kernel under HF2Q_FFN_SPLIT bisect; the mat-vec kernel
        // is bandwidth-bound at ~7-10 µs/layer = ~63 µs/layer x 30 layers
        // = ~1.9 ms/tok savings if H26 holds.  Opt-out via
        // HF2Q_F32_MATVEC=0 (legacy mat-mat path); default ON for m=1.
        let n = weight.info.rows as u32;
        let k = weight.info.cols as u32;
        let f32_matvec_default = std::env::var("HF2Q_F32_MATVEC")
            .ok()
            .map(|v| !matches!(v.as_str(), "0" | "false" | "off"))
            .unwrap_or(true);
        if m == 1 && f32_matvec_default {
            let params = DenseGemmF16Params { m, n, k };
            return mlx_native::ops::dense_gemm::dispatch_dense_matvec_f32(
                session.encoder_mut(),
                registry,
                device.metal_device(),
                input,
                &weight.buffer,
                output,
                &params,
            )
            .map_err(|e| anyhow::anyhow!("dispatch_dense_matvec_f32 failed: {e}"));
        }
        // m>=2 (prefill) or opt-out: fall back to mat-mat tile kernel.
        let params = mlx_native::DenseMmF32F32Params {
            m,
            n,
            k,
            src0_batch: 1,
            src1_batch: 1,
        };
        return mlx_native::dense_matmul_f32_f32_tensor(
            session.encoder_mut(),
            registry,
            device,
            &weight.buffer,
            input,
            output,
            &params,
        )
        .map_err(|e| anyhow::anyhow!("dense_matmul_f32_f32_tensor failed: {e}"));
    }

    // Native F16 weight tensor from GGUF (NOT the Q*→F16 shadow path above).
    //
    // Some convert pipelines, including hf2q's own (`should_emit_f16_for_kquant`
    // in `src/quantize/layer_mix.rs`), emit raw F16 for tensors whose row length
    // isn't a multiple of the K-quant super-block size (256) — e.g. Gemma 4
    // 26B-A4B's `ffn_down.weight` (intermediate=2112) and `ffn_down_exps.weight`
    // (moe_intermediate=704).  Community quants (bartowski, unsloth) instead
    // fall back to 32-aligned legacy quants (Q5_1 / IQ4_NL / Q8_0); both
    // patterns are valid per the GGUF spec and the peer accepts either.
    //
    // `quantized_matmul_ggml` below only handles Q-family block types, so
    // F16 weights would error there ("does not support F16").  Route them to
    // the matching F16 kernels — F16 weight × F32 input → F32 output — at
    // both prefill (m > 1, `dispatch_mm_v2_f16`) and decode (m = 1,
    // `dispatch_dense_matvec_f16w_f32io`).  These are the same kernels used
    // by the F16-shadow prefill path above, plus the decode-m=1 matvec
    // already used by lm_head's F16 weight when present.
    if weight.info.ggml_dtype == mlx_native::GgmlType::F16 {
        let n = weight.info.rows as u32;
        let k = weight.info.cols as u32;
        if m == 1 {
            let params = DenseGemmF16Params { m, n, k };
            return mlx_native::ops::dense_gemm::dispatch_dense_matvec_f16w_f32io(
                session.encoder_mut(),
                registry,
                device.metal_device(),
                input,
                &weight.buffer,
                output,
                &params,
            )
            .map_err(|e| {
                anyhow::anyhow!("dispatch_dense_matvec_f16w_f32io (native F16) failed: {e}")
            });
        }
        return mlx_native::ops::quantized_matmul_ggml::dispatch_mm_v2_f16(
            session.encoder_mut(),
            registry,
            device,
            &weight.buffer,
            input,
            output,
            m,
            n,
            k,
        )
        .map_err(|e| anyhow::anyhow!("dispatch_mm_v2_f16 (native F16) failed: {e}"));
    }

    // ADR-029 iter-175 Step 1d — Q6_K NR2 decode-m=1 fast path via
    // pre-baked DispatchRecord.  Eliminates per-call HashMap pipeline
    // lookup, MTLSize::new, ggml_type match arms, and
    // GgmlMatvecGpuParams struct construction.
    //
    // Falls through to the legacy path when:
    //   - bake returns None (HF2Q_Q6K_MV_NR2 disabled)
    //   - first call ever fails to compile the PSO
    // The legacy path is what handles all other dtypes (Q8_0, Q4_K,
    // Q5_K, etc.) — those gain fast paths in subsequent substeps.
    //
    // Pre-conditions established by earlier branches in this function:
    //   - weight.affine.is_none() (would have returned at affine block)
    //   - weight.f16_shadow not applicable at m=1 (only consulted at
    //     m > MM_ROUTING_THRESHOLD)
    //   - weight.info.ggml_dtype != F32 (would have returned at F32 block)
    //   - weight.info.ggml_dtype != F16 (would have returned at F16 block)
    if m == 1 && weight.info.ggml_dtype == mlx_native::GgmlType::Q6_K {
        let n = weight.info.rows as u32;
        let k = weight.info.cols as u32;
        let record_opt = weight.decode_record_q6k_m1.get_or_init(|| {
            mlx_native::ops::quantized_matmul_ggml::build_q6k_nr2_m1_record(
                registry,
                device.metal_device(),
                n,
                k,
            )
            .ok()
            .flatten()
        });
        if let Some(record) = record_opt {
            session
                .encoder_mut()
                .dispatch_record(record, &[&weight.buffer, input, output]);
            return Ok(());
        }
    }

    let params = weight.matmul_params(m)?;
    session
        .quantized_matmul_ggml(registry, device, input, &weight.buffer, output, &params)
        .map_err(|e| anyhow::anyhow!("quantized_matmul_ggml failed: {e}"))
}

// ---------------------------------------------------------------------------
// Cluster 2 — DWQ overlay parsing
// ---------------------------------------------------------------------------

/// ADR-020 AC#5 Iter D — classification of `blk.{i}.<role>` stems
/// emitted by `hf2q dwq-train`.  Drives slot-routing in
/// [`MlxModelWeights::apply_dwq_overlay`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DwqOverlayRole {
    AttnQ,
    AttnK,
    AttnV,
    AttnOutput,
    FfnGate,
    FfnUp,
    FfnDown,
    /// Per-expert MoE tensor (`ffn_gate.{e}`, `ffn_up.{e}`,
    /// `ffn_down.{e}`); skipped in Iter D, handled in Iter C2.
    MoeExpert,
    /// Stem doesn't match any known DWQ role.
    Unknown,
}

/// ADR-020 AC#5 Iter D — parse the DWQ safetensors metadata header,
/// extracting `(bits, group_size)`.  Defaults `(4, 32)` if the
/// metadata is absent (legacy DWQ output).  Returns an error if a
/// `format` field is present but doesn't match `mlx-affine-dwq-v1`.
pub fn parse_dwq_overlay_metadata(
    metadata: Option<&std::collections::HashMap<String, String>>,
) -> Result<(u32, usize)> {
    match metadata {
        Some(meta) => {
            if let Some(format_str) = meta.get("format") {
                if format_str != "mlx-affine-dwq-v1" {
                    anyhow::bail!(
                        "DWQ overlay: unsupported format '{}' (expected 'mlx-affine-dwq-v1')",
                        format_str
                    );
                }
            }
            let bits = meta
                .get("bits")
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(4u32);
            let group_size = meta
                .get("group_size")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(32usize);
            Ok((bits, group_size))
        }
        None => Ok((4u32, 32usize)),
    }
}

/// Classify a DWQ stem's role token (the part after `blk.{i}.`).
pub fn parse_dwq_overlay_role(role: &str) -> DwqOverlayRole {
    match role {
        "attn_q" => DwqOverlayRole::AttnQ,
        "attn_k" => DwqOverlayRole::AttnK,
        "attn_v" => DwqOverlayRole::AttnV,
        "attn_output" => DwqOverlayRole::AttnOutput,
        "ffn_gate" => DwqOverlayRole::FfnGate,
        "ffn_up" => DwqOverlayRole::FfnUp,
        "ffn_down" => DwqOverlayRole::FfnDown,
        r if r.starts_with("ffn_gate_up.")
            || r.starts_with("ffn_gate.")
            || r.starts_with("ffn_up.")
            || r.starts_with("ffn_down.") =>
        {
            DwqOverlayRole::MoeExpert
        }
        _ => DwqOverlayRole::Unknown,
    }
}

/// ADR-020 AC#5 Iter C2.2 — base role buckets for stacked MoE expert
/// loading.  An expert stem `ffn_gate_up.{N}` maps to `GateUp`,
/// `ffn_down.{N}` → `Down`, `ffn_gate.{N}` → `Gate`, `ffn_up.{N}` →
/// `Up`.  GateUp is the FUSED case (qwen3.5 GGUF `ffn_gate_up_exps`);
/// Gate + Up separately is the unfused case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoeBaseRole {
    GateUp,
    Gate,
    Up,
    Down,
}

/// Parse a per-expert MoE stem's role suffix (e.g. `ffn_gate_up.13`)
/// into `(MoeBaseRole, expert_idx)`.  Returns `None` if the role does
/// not match a known per-expert pattern.
pub fn parse_dwq_moe_expert_role(role: &str) -> Option<(MoeBaseRole, usize)> {
    // `ffn_gate_up.` must be checked before `ffn_gate.` to avoid the
    // longer prefix being consumed as `Gate.up.{e}` (which would be
    // "Gate" with a stray `up.` suffix).
    let (base, rest) = if let Some(rest) = role.strip_prefix("ffn_gate_up.") {
        (MoeBaseRole::GateUp, rest)
    } else if let Some(rest) = role.strip_prefix("ffn_gate.") {
        (MoeBaseRole::Gate, rest)
    } else if let Some(rest) = role.strip_prefix("ffn_up.") {
        (MoeBaseRole::Up, rest)
    } else if let Some(rest) = role.strip_prefix("ffn_down.") {
        (MoeBaseRole::Down, rest)
    } else {
        return None;
    };
    rest.parse::<usize>().ok().map(|e| (base, e))
}

// ---------------------------------------------------------------------------
// Cluster 3 — Norm dispatch helpers
// ---------------------------------------------------------------------------

/// Buffers + shape for a single per-head RMS-norm dispatch.
///
/// Grouped out of `dispatch_rms_norm_unit_perhead` (was 8 positional
/// args). The I/O buffers and the `(rows, dim)` shape describe *what*
/// the kernel operates on; `encoder`/`registry`/`device` describe *where*
/// it runs. Separating the two groups makes call sites scannable.
pub struct RmsNormPerHeadArgs<'a> {
    /// F32 `[rows, dim]` input tensor.
    pub input: &'a MlxBuffer,
    /// F32 `[rows, dim]` output tensor (separate buffer; kernel does not
    /// support in-place).
    pub output: &'a MlxBuffer,
    /// Constant params buffer (`dim`, `eps`) — pre-populated at load time.
    pub params_buf: &'a MlxBuffer,
    /// Number of rows (per-layer: `num_kv_heads`, or `seq_len * num_kv_heads`
    /// in the batched prefill path).
    pub rows: u32,
    /// Per-row element count (per-layer `head_dim`).
    pub dim: u32,
}

/// Dispatch per-head RMS norm without learned scale (unit norm, f32).
///
/// Same as `dispatch_rms_norm_perhead` but uses `rms_norm_no_scale_f32`
/// (no weight buffer — just unit normalization).
///
/// ADR-028 iter-338 — env-gated V2 (float4 + simd_sum) variant when
/// `dim % 4 == 0`, mirroring mlx-native ops/rms_norm.rs:662-683 pattern
/// that the iter-310 V2 dispatcher uses.  This was a V2-bypass site:
/// the function was authored before the V2 dispatcher existed and
/// directly gets the v1 pipeline by name.  Default-ON via
/// `HF2Q_RMS_NORM_V2`; opt-out via `=0`/`false`/`off`.
pub fn dispatch_rms_norm_unit_perhead(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut mlx_native::KernelRegistry,
    device: &mlx_native::metal::DeviceRef,
    args: &RmsNormPerHeadArgs<'_>,
) -> Result<()> {
    // Inline V2 env-gate (mirrors iter-326 inline pattern; preserves
    // debug→serve module boundary).
    let v2_env_off = matches!(
        std::env::var("HF2Q_RMS_NORM_V2").ok().as_deref(),
        Some(v) if v.eq_ignore_ascii_case("0")
            || v.eq_ignore_ascii_case("false")
            || v.eq_ignore_ascii_case("off")
    );
    let use_v2 = (args.dim % 4 == 0) && !v2_env_off;
    let kernel_name = if use_v2 {
        "rms_norm_no_scale_f32_v2"
    } else {
        "rms_norm_no_scale_f32"
    };
    let pipeline = registry
        .get_pipeline(kernel_name, device)
        .map_err(|e| anyhow::anyhow!("{kernel_name} pipeline: {e}"))?;
    let mut tg_size = std::cmp::min(256, args.dim.next_power_of_two()) as u64;
    if use_v2 && tg_size < 32 {
        tg_size = 32;
    }
    let shared_mem_bytes = if use_v2 {
        // V2 only needs n_sg = tg_size/32 floats; allocate at least 32
        // so partial-warp tg_sizes are safe (matches rms_norm.rs:679-681).
        (tg_size / 32).max(1) * 4
    } else {
        tg_size * 4
    };
    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[(0, args.input), (1, args.output), (2, args.params_buf)],
        &[(0, shared_mem_bytes)],
        mlx_native::MTLSize::new(args.rows as u64, 1, 1),
        mlx_native::MTLSize::new(tg_size, 1, 1),
    );
    Ok(())
}

/// Dual-output variant: writes both f32 (for KV cache copy) AND bf16
/// (for the bf16 attention island) — ADR-011 Phase 3 Wave P3b-tensor.3.
///
/// Used by batched prefill V-norm to fuse the f32→bf16 cast that was
/// previously a separate `cast_f32_to_bf16` dispatch.  Same compute,
/// one extra device write per element (effectively free on Apple
/// unified memory since the f32 result is in registers).
#[allow(clippy::too_many_arguments)]
/// Fused per-head V-norm + permute (Wave P4.16).
///
/// Same compute as `dispatch_rms_norm_unit_perhead_dual` but writes the
/// bf16 output at the permuted [n_heads, seq_len, head_dim] layout
/// instead of the natural [seq_len, n_heads, head_dim] layout.  Saves
/// the post-norm `permute_021_bf16` dispatch on V (~30 dispatches/
/// prefill on Gemma 4) and ~10 MB of intermediate-buffer traffic at
/// pp2455.
///
/// The f32 output stays at natural layout — KV cache copy reads it.
pub fn dispatch_rms_norm_unit_perhead_dual_perm(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut mlx_native::KernelRegistry,
    device: &mlx_native::metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    output_bf16_perm: &MlxBuffer,
    params_buf: &MlxBuffer,
    n_heads: u32,
    seq_len: u32,
    dim: u32,
) -> Result<()> {
    use mlx_native::ops::encode_helpers::{encode_threadgroups_with_args_and_shared, KernelArg};
    let pipeline = registry
        .get_pipeline("rms_norm_no_scale_f32_dual_perm", device)
        .map_err(|e| anyhow::anyhow!("rms_norm_no_scale_f32_dual_perm pipeline: {e}"))?;
    let rows = (n_heads as u64) * (seq_len as u64);
    let tg_size = std::cmp::min(256, dim.next_power_of_two()) as u64;
    let shared_mem_bytes = tg_size * 4;
    let aux_bytes: [u32; 2] = [n_heads, seq_len];
    let aux_bytes_b: &[u8] = unsafe {
        std::slice::from_raw_parts(
            aux_bytes.as_ptr() as *const u8,
            std::mem::size_of_val(&aux_bytes),
        )
    };
    encode_threadgroups_with_args_and_shared(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Buffer(params_buf)),
            (3, KernelArg::Buffer(output_bf16_perm)),
            (4, KernelArg::Bytes(aux_bytes_b)),
        ],
        &[(0, shared_mem_bytes)],
        mlx_native::MTLSize::new(rows, 1, 1),
        mlx_native::MTLSize::new(tg_size, 1, 1),
    );
    Ok(())
}

/// ADR-029 iter-175 Step 1f — cached fast-path wrapper for hidden-size
/// F32 `rms_norm` decode dispatches.
///
/// On first call, lazily bakes a `DispatchRecord` via
/// `build_rms_norm_decode_record(F32, 1, hs)` into the shared `cache`
/// slot.  On subsequent calls (the steady state), the encoder fires
/// `dispatch_record` directly — skipping the `KernelRegistry::get_pipeline`
/// HashMap lookup, `MTLSize::new` × 2, `dim.next_power_of_two`, shmem
/// calculation, and `set_op_kind` indirection (~50 ns total per call).
///
/// Caller contract: input/weight/output dtype must all be F32, rows=1,
/// dim=hs.  These match the four production `rms_norm` call sites in
/// `encode_one_layer` (pre-FF norm, pre-FF norm 2, router norm,
/// post-FF norm 1) which hit ~120 dispatches/decode-tok on gemma4 APEX-Q5_K_M.
///
/// Falls through to unbaked `session.rms_norm` when the bake returns
/// None (unsupported dtype, pipeline-lookup failure).
///
/// `#[inline(always)]` per ADR-029 Step 1f2 — the call-site inlining
/// is the load-bearing optimization (without it, Step 1f bench was
/// neutral despite ~120 dispatches/tok coverage).  Step 1d's win at
/// the inline `dispatch_qmatmul` fast-path branch motivated this.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn rms_norm_f32_hs_cached(
    cache: &std::sync::OnceLock<Option<mlx_native::DispatchRecord>>,
    session: &mut mlx_native::graph::GraphSession<'_>,
    reg: &mut mlx_native::KernelRegistry,
    metal_dev: &mlx_native::metal::DeviceRef,
    input: &mlx_native::MlxBuffer,
    weight: &mlx_native::MlxBuffer,
    output: &mlx_native::MlxBuffer,
    params: &mlx_native::MlxBuffer,
    hs: u32,
) -> Result<()> {
    let rec = cache.get_or_init(|| {
        mlx_native::ops::rms_norm::build_rms_norm_decode_record(
            reg,
            metal_dev,
            mlx_native::DType::F32,
            1,
            hs,
        )
        .ok()
        .flatten()
    });
    if let Some(r) = rec {
        session
            .encoder_mut()
            .dispatch_record(r, &[input, weight, output, params]);
        return Ok(());
    }
    session
        .rms_norm(reg, metal_dev, input, weight, output, params, 1, hs)
        .map_err(|e| anyhow::anyhow!("rms_norm cached fallback: {e}"))
}

// ---------------------------------------------------------------------------
// Cluster 4 — cosine_pairwise_f32 (standalone)
// ---------------------------------------------------------------------------

/// dumps of the same SDPA-output shape.
///
/// (Wired by iter-108b's release-check.sh Gate 5 harness; the audit
/// binaries iter23/24_audit.rs will switch from `python3 cosine_sim.py`
/// to a direct call into this function in iter-108b.  No in-tree
/// caller as of iter-108a — the surface is designed for the iter-108b
/// release-check entry point + the unit tests below.)
#[allow(dead_code)]
pub fn cosine_pairwise_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine vectors must match length");
    let n = a.len().min(b.len());
    let mut dot: f64 = 0.0;
    let mut na2: f64 = 0.0;
    let mut nb2: f64 = 0.0;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na2 += x * x;
        nb2 += y * y;
    }
    let na = na2.sqrt();
    let nb = nb2.sqrt();
    if na == 0.0 || nb == 0.0 {
        f32::NAN
    } else {
        (dot / (na * nb)) as f32
    }
}

// ---------------------------------------------------------------------------
// Tests (relocated from forward_mlx.rs)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod cosine_tests {
    use super::cosine_pairwise_f32;

    #[test]
    fn identity_is_one() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let a = vec![1.0_f32, 2.0, 3.0, -4.5, 0.25];
        let s = cosine_pairwise_f32(&a, &a);
        assert!((s - 1.0).abs() < 1e-6, "identity cosine = {s}");
    }

    #[test]
    fn antiparallel_is_negative_one() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let a: Vec<f32> = (0..128).map(|i| (i as f32) - 64.0).collect();
        let neg: Vec<f32> = a.iter().map(|x| -x).collect();
        let s = cosine_pairwise_f32(&a, &neg);
        assert!((s + 1.0).abs() < 1e-6, "antiparallel cosine = {s}");
    }

    #[test]
    fn zero_norm_is_nan() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let a = vec![0.0_f32; 32];
        let b: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let s = cosine_pairwise_f32(&a, &b);
        assert!(s.is_nan(), "zero-norm cosine should be NaN, got {s}");
        // And symmetric: nonzero on the left, zero on the right.
        let s2 = cosine_pairwise_f32(&b, &a);
        assert!(
            s2.is_nan(),
            "zero-norm cosine (rhs) should be NaN, got {s2}"
        );
        // Both zero → NaN.
        let z = vec![0.0_f32; 32];
        let s3 = cosine_pairwise_f32(&z, &z);
        assert!(s3.is_nan(), "both-zero cosine should be NaN, got {s3}");
    }

    #[test]
    fn orthogonal_is_zero() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // [1,0,0,...] vs [0,1,0,...] → dot=0, both norms=1 → cosine=0.
        let mut a = vec![0.0_f32; 16];
        let mut b = vec![0.0_f32; 16];
        a[0] = 1.0;
        b[1] = 1.0;
        let s = cosine_pairwise_f32(&a, &b);
        assert!(s.abs() < 1e-6, "orthogonal cosine = {s}");
    }

    #[test]
    fn matches_python_reference_within_tolerance() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Reference shape: same kernel as iter24_audit.rs:752-761 in F64.
        // We compare against the explicit numpy-style formula to make sure
        // our F64 accumulation tracks the audit-binary numbers.
        let a: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.013).sin()).collect();
        let b: Vec<f32> = (0..512)
            .map(|i| ((i as f32) * 0.013).sin() + 1e-3)
            .collect();
        let s = cosine_pairwise_f32(&a, &b);
        let py_dot: f64 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (*x as f64) * (*y as f64))
            .sum();
        let py_na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let py_nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let py = (py_dot / (py_na * py_nb)) as f32;
        assert!(
            (s - py).abs() < 1e-6,
            "rust cosine={s} vs python-equiv={py}"
        );
    }
}

/// ADR-022 P1.9 iter-15 — `dispatch_qmatmul` F32 routing test.
///
/// Hypothesis: an `MlxQWeight` with `ggml_dtype = F32` (router weight from
/// APEX-format Gemma4 GGUF) routes through `dense_matmul_f32_f32_tensor`
/// and produces `output[m, n] = sum_k input[m, k] * weight[n, k]`
/// matching a CPU reference.
///
/// Falsifier: pre-iter-15 hf2q dispatched F32 weights through
/// `quantized_matmul_ggml`, which (correctly) returned an error because
/// the GGML block kernels require block-format input. With this routing
/// fix the same call must succeed and produce the expected matmul.
#[cfg(test)]
mod dispatch_qmatmul_f32_router_test {
    use super::*;

    #[test]
    fn f32_router_weight_routes_to_dense_matmul() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match mlx_native::MlxDevice::new() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping f32_router_weight_routes_to_dense_matmul: no MlxDevice");
                return;
            }
        };
        let mut registry = mlx_native::KernelRegistry::new();

        // [n=4 output, k=64 inner] — k≥32 required by dense_mm_f32_f32 kernel.
        let n: usize = 4;
        let k: usize = 64;
        let m: usize = 2;

        // Deterministic pseudo-random fixtures.
        let mut state: u64 = 0xDEAD_BEEF_F00D_F00D;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let weight: Vec<f32> = (0..(n * k)).map(|_| next()).collect();
        let input: Vec<f32> = (0..(m * k)).map(|_| next()).collect();

        // CPU reference: out[m, n] = sum_k input[m, k] * weight[n, k].
        let mut expected = vec![0.0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0.0f64;
                for ki in 0..k {
                    acc += (input[mi * k + ki] as f64) * (weight[ni * k + ki] as f64);
                }
                expected[mi * n + ni] = acc as f32;
            }
        }

        // GPU buffers.
        let f32_sz = std::mem::size_of::<f32>();
        let mut weight_buf = device
            .alloc_buffer(n * k * f32_sz, mlx_native::DType::F32, vec![n, k])
            .expect("alloc weight");
        weight_buf
            .as_mut_slice::<f32>()
            .expect("weight write")
            .copy_from_slice(&weight);

        let mut input_buf = device
            .alloc_buffer(m * k * f32_sz, mlx_native::DType::F32, vec![m, k])
            .expect("alloc input");
        input_buf
            .as_mut_slice::<f32>()
            .expect("input write")
            .copy_from_slice(&input);

        let mut output_buf = device
            .alloc_buffer(m * n * f32_sz, mlx_native::DType::F32, vec![m, n])
            .expect("alloc output");

        let qweight = MlxQWeight {
            buffer: weight_buf,
            info: crate::serve::gpu::QuantWeightInfo {
                ggml_dtype: mlx_native::GgmlType::F32,
                rows: n,
                cols: k,
            },
            affine: None,
            f16_shadow: None,
            decode_record_q6k_m1: std::sync::OnceLock::new(),
        };

        // Run through GraphSession (mirrors production dispatch path).
        let executor = mlx_native::GraphExecutor::new(device.clone());
        let mut session = executor.begin().expect("begin session");
        dispatch_qmatmul(
            &mut session,
            &mut registry,
            &device,
            &input_buf,
            &qweight,
            &mut output_buf,
            m as u32,
            crate::quantize::imatrix::ImatrixHint::None,
        )
        .expect("dispatch_qmatmul F32 path");
        session.finish().expect("session finish");

        // Validate output.
        let got: &[f32] = output_buf.as_slice().expect("read output");
        let mut max_abs_diff = 0.0f32;
        for i in 0..(m * n) {
            let d = (got[i] - expected[i]).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        assert!(
            max_abs_diff < 1e-4,
            "F32 dispatch_qmatmul mismatch: max|diff|={max_abs_diff}, got={:?}, expected={:?}",
            got,
            expected
        );
    }
}

/// ADR-020 AC#5 Iter B — `MlxQWeight::from_mlx_affine_linear` round-trip
/// + GPU dispatch parity test.
///
/// Hypothesis: an MlxAffineLinear constructed in-process (skipping the
/// safetensors disk round-trip), uploaded via `from_mlx_affine_linear`,
/// and dispatched via the new `qmm_affine_t_packed_simd4_b4` kernel
/// produces output matching a CPU oracle that does
/// `y = x @ (q_int * scales + biases)^T`.
///
/// Falsifier: any divergence in the packed-U32 emission inside
/// `from_mlx_affine_linear` (e.g. wrong slot ordering, wrong nibble
/// position) would surface as a measurable error vs the CPU oracle.
#[cfg(test)]
mod ac5_iter_b_affine_qweight_roundtrip {
    use super::*;
    use crate::core::mlx_safetensors_loader::MlxAffineLinear;
    use mlx_native::ops::qmm_affine::dispatch_qmm_affine_t_packed_simd4_b4;

    #[test]
    fn from_mlx_affine_linear_roundtrips_through_packed_kernel() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match mlx_native::MlxDevice::new() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping ac5_iter_b: no MlxDevice");
                return;
            }
        };
        let mut registry = mlx_native::KernelRegistry::new();

        let m = 16usize;
        let n = 64usize;
        let k = 96usize;
        let group_size = 32usize;
        let bits = 4u32;
        let pack_factor = (32 / bits) as usize;
        let groups_per_row = k / group_size;

        // Synthetic deterministic linear: q_int in [0, 16), scales/biases F32.
        let q_int: Vec<u8> = (0..(n * k)).map(|i| ((i * 11 + 5) % 16) as u8).collect();
        let scales: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| 0.05 + (i as f32) * 0.0017)
            .collect();
        let biases: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| -0.13 + (i as f32) * 0.0023)
            .collect();
        let linear = MlxAffineLinear {
            n,
            k,
            group_size,
            bits,
            q_int: q_int.clone(),
            scales: scales.clone(),
            biases: biases.clone(),
        };

        // Hf2q: pack into MlxQWeight via the AC#5 Iter B constructor.
        let qweight =
            MlxQWeight::from_mlx_affine_linear(&device, &linear).expect("from_mlx_affine_linear");
        assert_eq!(qweight.info.rows, n);
        assert_eq!(qweight.info.cols, k);
        let extra = qweight.affine.as_ref().expect("affine extra");
        assert_eq!(extra.bits, bits);
        assert_eq!(extra.group_size, group_size as u32);
        assert_eq!(qweight.buffer.element_count(), n * (k / pack_factor));
        assert_eq!(extra.scales.element_count(), n * groups_per_row);
        assert_eq!(extra.biases.element_count(), n * groups_per_row);

        // Upload x.
        let x: Vec<f32> = (0..(m * k))
            .map(|i| ((i as f32) * 0.013 - 0.4).sin() * 0.6)
            .collect();
        let mut x_buf = device
            .alloc_buffer(m * k * 4, mlx_native::DType::F32, vec![m, k])
            .expect("x");
        x_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&x);

        let y_buf = device
            .alloc_buffer(m * n * 4, mlx_native::DType::F32, vec![m, n])
            .expect("y");

        // meta: [M, N, K, group_size]
        let mut meta = device
            .alloc_buffer(16, mlx_native::DType::U32, vec![4])
            .unwrap();
        meta.as_mut_slice::<u32>().unwrap().copy_from_slice(&[
            m as u32,
            n as u32,
            k as u32,
            group_size as u32,
        ]);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_qmm_affine_t_packed_simd4_b4(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &x_buf,
            &qweight.buffer,
            &extra.scales,
            &extra.biases,
            &y_buf,
            &meta,
            m as u32,
            n as u32,
            k as u32,
            group_size as u32,
            bits,
        )
        .expect("dispatch packed simd4");
        encoder.commit_and_wait().unwrap();

        // CPU oracle: y[r, col] = sum_k x[r, k] * (q_int[col, k] * scales[col, g] + biases[col, g]).
        let mut expected = vec![0.0f32; m * n];
        for r in 0..m {
            for col in 0..n {
                let mut acc = 0.0f64;
                for g in 0..groups_per_row {
                    let s = scales[col * groups_per_row + g] as f64;
                    let b = biases[col * groups_per_row + g] as f64;
                    for i in 0..group_size {
                        let kk = g * group_size + i;
                        let q = q_int[col * k + kk] as f64;
                        acc += (x[r * k + kk] as f64) * (q * s + b);
                    }
                }
                expected[r * n + col] = acc as f32;
            }
        }

        let got = y_buf.as_slice::<f32>().unwrap();
        let mut max_abs = 0.0f32;
        for i in 0..(m * n) {
            let d = (got[i] - expected[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        assert!(
            max_abs < 1e-3,
            "max|y - oracle| = {max_abs} (m={m}, n={n}, k={k})"
        );
    }

    /// AC#5 Iter C — `dispatch_qmatmul` routes MlxQWeight with
    /// `affine.is_some()` through the new packed kernel.  This is the
    /// production entry point; correctness here = AC #5 dense closure
    /// at the dispatch boundary.
    #[test]
    fn dispatch_qmatmul_routes_affine_weight_to_packed_kernel() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match mlx_native::MlxDevice::new() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping: no MlxDevice");
                return;
            }
        };
        let mut registry = mlx_native::KernelRegistry::new();

        let m = 8usize;
        let n = 32usize;
        let k = 64usize;
        let gs = 32usize;
        let bits = 4u32;
        let groups_per_row = k / gs;

        let q_int: Vec<u8> = (0..(n * k)).map(|i| ((i * 7 + 3) % 16) as u8).collect();
        let scales: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| 0.07 + (i as f32) * 0.0011)
            .collect();
        let biases: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| -0.09 + (i as f32) * 0.0027)
            .collect();
        let linear = MlxAffineLinear {
            n,
            k,
            group_size: gs,
            bits,
            q_int: q_int.clone(),
            scales: scales.clone(),
            biases: biases.clone(),
        };
        let qweight =
            MlxQWeight::from_mlx_affine_linear(&device, &linear).expect("from_mlx_affine_linear");

        let x: Vec<f32> = (0..(m * k))
            .map(|i| ((i as f32) * 0.011 - 0.3).cos() * 0.5)
            .collect();
        let mut x_buf = device
            .alloc_buffer(m * k * 4, mlx_native::DType::F32, vec![m, k])
            .expect("x");
        x_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&x);

        let mut y_buf = device
            .alloc_buffer(m * n * 4, mlx_native::DType::F32, vec![m, n])
            .expect("y");

        // Drive through GraphSession exactly like production.
        let executor = mlx_native::GraphExecutor::new(device.clone());
        let mut session = executor.begin().expect("begin session");
        dispatch_qmatmul(
            &mut session,
            &mut registry,
            &device,
            &x_buf,
            &qweight,
            &mut y_buf,
            m as u32,
            crate::quantize::imatrix::ImatrixHint::None,
        )
        .expect("dispatch_qmatmul affine route");
        session.finish().expect("finish");

        // CPU oracle — same formula as Iter B test.
        let mut expected = vec![0.0f32; m * n];
        for r in 0..m {
            for col in 0..n {
                let mut acc = 0.0f64;
                for g in 0..groups_per_row {
                    let s = scales[col * groups_per_row + g] as f64;
                    let b = biases[col * groups_per_row + g] as f64;
                    for i in 0..gs {
                        let kk = g * gs + i;
                        let q = q_int[col * k + kk] as f64;
                        acc += (x[r * k + kk] as f64) * (q * s + b);
                    }
                }
                expected[r * n + col] = acc as f32;
            }
        }

        let got = y_buf.as_slice::<f32>().unwrap();
        let mut max_abs = 0.0f32;
        for i in 0..(m * n) {
            let d = (got[i] - expected[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        assert!(
            max_abs < 1e-3,
            "dispatch_qmatmul affine route: max|y - oracle| = {max_abs}"
        );
    }

    /// AC#5 Iter C — affine route is byte-identical to direct kernel
    /// dispatch.  Confirms `dispatch_qmatmul` doesn't introduce any
    /// per-call drift (e.g. wrong meta values, batch-dim confusion).
    #[test]
    fn dispatch_qmatmul_affine_equals_direct_kernel() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match mlx_native::MlxDevice::new() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping: no MlxDevice");
                return;
            }
        };
        let mut registry = mlx_native::KernelRegistry::new();

        let m = 4usize;
        let n = 32usize;
        let k = 32usize;
        let gs = 32usize;
        let bits = 4u32;
        let groups_per_row = k / gs;

        let q_int: Vec<u8> = (0..(n * k)).map(|i| ((i * 5 + 1) % 16) as u8).collect();
        let scales: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| 0.05 + (i as f32) * 0.001)
            .collect();
        let biases: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| -0.1 + (i as f32) * 0.002)
            .collect();
        let linear = MlxAffineLinear {
            n,
            k,
            group_size: gs,
            bits,
            q_int,
            scales,
            biases,
        };
        let qweight =
            MlxQWeight::from_mlx_affine_linear(&device, &linear).expect("from_mlx_affine_linear");
        let extra = qweight.affine.as_ref().unwrap();

        let x: Vec<f32> = (0..(m * k))
            .map(|i| ((i as f32) * 0.017 + 0.2).sin() * 0.4)
            .collect();
        let mut x_buf = device
            .alloc_buffer(m * k * 4, mlx_native::DType::F32, vec![m, k])
            .expect("x");
        x_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&x);

        let mut y_via_dispatch = device
            .alloc_buffer(m * n * 4, mlx_native::DType::F32, vec![m, n])
            .expect("y_d");
        let y_direct = device
            .alloc_buffer(m * n * 4, mlx_native::DType::F32, vec![m, n])
            .expect("y_k");
        let mut meta = device
            .alloc_buffer(16, mlx_native::DType::U32, vec![4])
            .unwrap();
        meta.as_mut_slice::<u32>()
            .unwrap()
            .copy_from_slice(&[m as u32, n as u32, k as u32, gs as u32]);

        // Direct kernel call.
        let mut encoder = device.command_encoder().unwrap();
        mlx_native::ops::qmm_affine::dispatch_qmm_affine_t_packed_simd4_b4(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &x_buf,
            &qweight.buffer,
            &extra.scales,
            &extra.biases,
            &y_direct,
            &meta,
            m as u32,
            n as u32,
            k as u32,
            gs as u32,
            bits,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();

        // dispatch_qmatmul path.
        let executor = mlx_native::GraphExecutor::new(device.clone());
        let mut session = executor.begin().expect("begin session");
        dispatch_qmatmul(
            &mut session,
            &mut registry,
            &device,
            &x_buf,
            &qweight,
            &mut y_via_dispatch,
            m as u32,
            crate::quantize::imatrix::ImatrixHint::None,
        )
        .expect("dispatch_qmatmul");
        session.finish().expect("finish");

        let direct = y_direct.as_slice::<f32>().unwrap();
        let dispatch = y_via_dispatch.as_slice::<f32>().unwrap();
        for i in 0..(m * n) {
            assert_eq!(
                dispatch[i].to_bits(),
                direct[i].to_bits(),
                "y[{i}] (m={m} n={n}): dispatch={} direct={}",
                dispatch[i],
                direct[i],
            );
        }
    }

    /// AC#5 Iter C2.2 — `parse_dwq_moe_expert_role` covers all 4 MoE
    /// base roles + handles invalid suffixes correctly.
    #[test]
    fn parse_dwq_moe_expert_role_covers_all_bases() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        use super::{parse_dwq_moe_expert_role, MoeBaseRole};
        // Fused gate+up case (qwen3.5 GGUF).
        assert_eq!(
            parse_dwq_moe_expert_role("ffn_gate_up.0"),
            Some((MoeBaseRole::GateUp, 0))
        );
        assert_eq!(
            parse_dwq_moe_expert_role("ffn_gate_up.127"),
            Some((MoeBaseRole::GateUp, 127))
        );
        // Separate gate / up (uncommon; future archs).
        assert_eq!(
            parse_dwq_moe_expert_role("ffn_gate.5"),
            Some((MoeBaseRole::Gate, 5))
        );
        assert_eq!(
            parse_dwq_moe_expert_role("ffn_up.7"),
            Some((MoeBaseRole::Up, 7))
        );
        // Down expert.
        assert_eq!(
            parse_dwq_moe_expert_role("ffn_down.42"),
            Some((MoeBaseRole::Down, 42))
        );
        // Critical: `ffn_gate_up.X` must not match the `ffn_gate.` prefix.
        assert_ne!(
            parse_dwq_moe_expert_role("ffn_gate_up.3"),
            Some((MoeBaseRole::Gate, 3))
        );
        // Invalid suffixes.
        assert_eq!(parse_dwq_moe_expert_role("ffn_gate_up.abc"), None);
        assert_eq!(parse_dwq_moe_expert_role("ffn_gate"), None);
        assert_eq!(parse_dwq_moe_expert_role("attn_q.0"), None);
        assert_eq!(parse_dwq_moe_expert_role(""), None);
    }

    /// AC#5 Iter D — `parse_dwq_overlay_role` covers all production
    /// stems plus rejects unknowns.  Pure CPU; no MlxDevice required.
    #[test]
    fn parse_dwq_overlay_role_covers_all_dense_stems() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        use super::{parse_dwq_overlay_role, DwqOverlayRole};
        // Dense Linears.
        assert_eq!(parse_dwq_overlay_role("attn_q"), DwqOverlayRole::AttnQ);
        assert_eq!(parse_dwq_overlay_role("attn_k"), DwqOverlayRole::AttnK);
        assert_eq!(parse_dwq_overlay_role("attn_v"), DwqOverlayRole::AttnV);
        assert_eq!(
            parse_dwq_overlay_role("attn_output"),
            DwqOverlayRole::AttnOutput
        );
        assert_eq!(parse_dwq_overlay_role("ffn_gate"), DwqOverlayRole::FfnGate);
        assert_eq!(parse_dwq_overlay_role("ffn_up"), DwqOverlayRole::FfnUp);
        assert_eq!(parse_dwq_overlay_role("ffn_down"), DwqOverlayRole::FfnDown);
        // MoE per-expert (Iter C2 territory).
        assert_eq!(
            parse_dwq_overlay_role("ffn_gate.0"),
            DwqOverlayRole::MoeExpert
        );
        assert_eq!(
            parse_dwq_overlay_role("ffn_up.255"),
            DwqOverlayRole::MoeExpert
        );
        assert_eq!(
            parse_dwq_overlay_role("ffn_down.42"),
            DwqOverlayRole::MoeExpert
        );
        // Unknown roles.
        assert_eq!(
            parse_dwq_overlay_role("token_embd"),
            DwqOverlayRole::Unknown
        );
        assert_eq!(parse_dwq_overlay_role(""), DwqOverlayRole::Unknown);
        assert_eq!(parse_dwq_overlay_role("output"), DwqOverlayRole::Unknown);
    }

    /// AC#5 Iter D — `parse_dwq_overlay_metadata` honors metadata when
    /// present, defaults sanely when absent, rejects mismatched format.
    #[test]
    fn parse_dwq_overlay_metadata_handles_all_cases() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        use super::parse_dwq_overlay_metadata;
        use std::collections::HashMap;

        // Absent metadata → defaults (4, 32).
        let (bits, gs) = parse_dwq_overlay_metadata(None).unwrap();
        assert_eq!(bits, 4);
        assert_eq!(gs, 32);

        // Format mismatch → error.
        let mut bad_format = HashMap::new();
        bad_format.insert("format".to_string(), "wrong-format".to_string());
        assert!(parse_dwq_overlay_metadata(Some(&bad_format)).is_err());

        // Correct format + custom bits/gs.
        let mut meta = HashMap::new();
        meta.insert("format".to_string(), "mlx-affine-dwq-v1".to_string());
        meta.insert("bits".to_string(), "8".to_string());
        meta.insert("group_size".to_string(), "64".to_string());
        let (bits, gs) = parse_dwq_overlay_metadata(Some(&meta)).unwrap();
        assert_eq!(bits, 8);
        assert_eq!(gs, 64);

        // Format absent + valid bits/gs → values respected.
        let mut nofmt = HashMap::new();
        nofmt.insert("bits".to_string(), "4".to_string());
        nofmt.insert("group_size".to_string(), "32".to_string());
        let (bits, gs) = parse_dwq_overlay_metadata(Some(&nofmt)).unwrap();
        assert_eq!(bits, 4);
        assert_eq!(gs, 32);

        // Garbage bits → falls back to default 4.
        let mut garbage = HashMap::new();
        garbage.insert("format".to_string(), "mlx-affine-dwq-v1".to_string());
        garbage.insert("bits".to_string(), "not-a-number".to_string());
        let (bits, gs) = parse_dwq_overlay_metadata(Some(&garbage)).unwrap();
        assert_eq!(bits, 4);
        assert_eq!(gs, 32);
    }

    /// AC#5 Iter D — full DWQ-format safetensors round-trip:
    /// MlxAffineLinear → safetensors-on-disk-with-metadata → re-read +
    /// rebuild MlxAffineLinear → byte-identical contents.  Confirms the
    /// metadata embed (Iter D step 1) + the safetensors loader contract
    /// hold without MlxModelWeights involvement.
    #[test]
    fn dwq_safetensors_metadata_roundtrip() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        use crate::core::mlx_safetensors_loader::{MlxAffineLinear, MlxAffineLinearBytes};
        use safetensors::tensor::{serialize, Dtype};
        use std::collections::HashMap;

        let n = 32usize;
        let k = 64usize;
        let group_size = 32usize;
        let bits = 4u32;
        let groups_per_row = k / group_size;
        let q_int: Vec<u8> = (0..(n * k)).map(|i| ((i * 3 + 7) % 16) as u8).collect();
        let scales: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| 0.05 + (i as f32) * 0.001)
            .collect();
        let biases: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| -0.1 + (i as f32) * 0.002)
            .collect();

        let linear = MlxAffineLinear {
            n,
            k,
            group_size,
            bits,
            q_int: q_int.clone(),
            scales: scales.clone(),
            biases: biases.clone(),
        };
        let stem = "blk.0.attn_q";
        let bytes_owned: MlxAffineLinearBytes = linear.to_safetensors_bytes(Dtype::F32).unwrap();
        let (w, s, b) = bytes_owned.to_safetensors_views().unwrap();
        let pairs: Vec<(String, _)> = vec![
            (format!("{stem}.weight"), w),
            (format!("{stem}.scales"), s),
            (format!("{stem}.biases"), b),
        ];
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "mlx-affine-dwq-v1".to_string());
        metadata.insert("bits".to_string(), bits.to_string());
        metadata.insert("group_size".to_string(), group_size.to_string());

        let serialized =
            serialize(pairs.iter().map(|(k, v)| (k.as_str(), v)), Some(metadata)).unwrap();

        // Verify metadata round-trips via read_metadata.
        let (_n, md) = safetensors::SafeTensors::read_metadata(&serialized).unwrap();
        let meta_map = md.metadata().as_ref().expect("metadata present");
        assert_eq!(meta_map.get("format").unwrap(), "mlx-affine-dwq-v1");
        assert_eq!(meta_map.get("bits").unwrap(), "4");
        assert_eq!(meta_map.get("group_size").unwrap(), "32");

        let (parsed_bits, parsed_gs) = super::parse_dwq_overlay_metadata(Some(meta_map)).unwrap();
        assert_eq!(parsed_bits, bits);
        assert_eq!(parsed_gs, group_size);

        // Rebuild the MlxAffineLinear via from_safetensors and compare.
        let st = safetensors::SafeTensors::deserialize(&serialized).unwrap();
        let stems: Vec<&str> = st
            .names()
            .iter()
            .filter_map(|n| n.strip_suffix(".weight"))
            .collect();
        assert_eq!(stems.len(), 1);
        assert_eq!(stems[0], stem);

        let rebuilt = MlxAffineLinear::from_safetensors(&st, stem, parsed_bits, parsed_gs).unwrap();
        assert_eq!(rebuilt.n, n);
        assert_eq!(rebuilt.k, k);
        assert_eq!(rebuilt.bits, bits);
        assert_eq!(rebuilt.group_size, group_size);
        assert_eq!(rebuilt.q_int, q_int);
        assert_eq!(rebuilt.scales, scales);
        assert_eq!(rebuilt.biases, biases);
    }

    #[test]
    fn from_mlx_affine_linear_rejects_unsupported_bits() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match mlx_native::MlxDevice::new() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping: no MlxDevice");
                return;
            }
        };
        // bits=8 is not supported by the simd4_b4 kernel — constructor
        // must reject so we surface the gap at load time, not dispatch time.
        let linear = MlxAffineLinear {
            n: 32,
            k: 32,
            group_size: 32,
            bits: 8,
            q_int: vec![0u8; 32 * 32],
            scales: vec![0.1f32; 32],
            biases: vec![0.0f32; 32],
        };
        let res = MlxQWeight::from_mlx_affine_linear(&device, &linear);
        assert!(res.is_err(), "should reject bits=8");
    }
}
