//! ADR-037 Phase E4b — EAGLE-3 drafter forward primitives.
//!
//! Mirrors `src/inference/spec_decode/dflash/forward.rs` pattern but
//! adapted for the EAGLE-3 1-layer drafter. Each function dispatches
//! one stage of the forward pass; the orchestrator (Phase E4b.6,
//! future iter) chains them.
//!
//! ## Current sub-phase coverage
//!
//! - **E4b.1** `tensors.rs`: GPU upload pipeline (SHIPPED).
//! - **E4b.2** `dispatch_eagle3_fc` (this file): projects
//!   `[seq, num_aux * H]` → `[seq, H]` via the BF16 `fc.weight`.
//! - **E4b.3** `dispatch_eagle3_input_layernorm` +
//!   `dispatch_eagle3_hidden_norm` + `dispatch_eagle3_concat_2x_hidden`
//!   (this file): per vLLM `llama_eagle3.py:102-106` first-layer pre-attn:
//!   normalize embeds + hidden states separately, then concat along the
//!   last dim to produce the `[seq, 2*H]` input to Q/K/V projections.
//! - **E4b.4+** TODO: self-attn (via Phase E1 `tree_attention`
//!   kernel), MLP, final norm, lm_head, top-K extraction.

use super::config::Eagle3DrafterConfig;
use super::tensors::Eagle3DrafterTensors;
use crate::inference::models::qwen35::gpu_full_attn::{
    apply_imrope, apply_linear_projection_f32,
};
use mlx_native::ops::tree_attention::{
    self as tree_attn_ops, TreeAttentionParams,
};
use anyhow::{anyhow, Context, Result};
use mlx_native::ops::add_bias_row_2d::dispatch_add_bias_row_2d_f32;
use mlx_native::ops::elementwise::elementwise_add;
use mlx_native::ops::feature_concat::dispatch_feature_concat_f32;
use mlx_native::ops::rms_norm::dispatch_rms_norm;
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

/// EAGLE-3 FC projection.
///
/// Mirrors vLLM `Eagle3LlamaForCausalLM::combine_hidden_states` last
/// step (`llama_eagle3.py:407` `return self.model.fc(hidden_states)`).
/// Projects the concatenated multi-aux hidden tensor
/// `[seq, num_aux * target_hidden_size]` (which `Eagle3HiddenCollector`
/// produces) down to the drafter's hidden_size:
///
/// ```text
///   output[s, h] = sum over k of input[s, k] * fc_weight[h, k]
/// ```
///
/// Where `fc_weight: [hidden_size, fc_input_size]` BF16 and
/// `fc_input_size = num_aux * target_hidden_size`.
///
/// # Arguments
/// * `concat_hidden_gpu` — F32 buffer of shape
///   `[seq_len, fc_input_size]`. Caller uploads `Eagle3HiddenCollector::
///   concatenated_hidden()` to GPU before this call.
/// * `tensors` — uploaded drafter weights from Phase E4b.1.
/// * `cfg` — drafter config (provides expected dims for validation).
/// * `seq_len` — number of token positions in `concat_hidden_gpu`.
///
/// Returns the output buffer `[seq_len, hidden_size]` F32. Caller
/// commits the encoder.
pub fn dispatch_eagle3_fc(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    concat_hidden_gpu: &MlxBuffer,
    tensors: &Eagle3DrafterTensors,
    cfg: &Eagle3DrafterConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    // Codex /cfa E4b.2 Critical (2026-05-22): input dtype must be
    // F32. apply_linear_projection_f32 only debug-asserts this;
    // a release build with BF16/U8 input would mis-stride reads.
    // Fail-fast here at the wrapper.
    if concat_hidden_gpu.dtype() != DType::F32 {
        return Err(anyhow!(
            "dispatch_eagle3_fc: concat_hidden dtype must be F32, got {:?}",
            concat_hidden_gpu.dtype()
        ));
    }
    // Codex /cfa E4b.2 Major (2026-05-22): use try_from instead of
    // `as u32` to catch silent truncation on adversarial config.
    let fc_in_usize = cfg.fc_input_size();
    let hidden_usize = cfg.hidden_size;
    let fc_in: u32 = u32::try_from(fc_in_usize).map_err(|_| {
        anyhow!(
            "dispatch_eagle3_fc: fc_input_size ({}) exceeds u32::MAX",
            fc_in_usize
        )
    })?;
    let hidden: u32 = u32::try_from(hidden_usize).map_err(|_| {
        anyhow!(
            "dispatch_eagle3_fc: hidden_size ({}) exceeds u32::MAX",
            hidden_usize
        )
    })?;
    // Codex /cfa E4b.2 Major (2026-05-22): checked multiply for the
    // expected element count. Otherwise large seq_len could wrap on
    // release and let an undersized buffer pass validation.
    let expected_input_elems = (seq_len as usize)
        .checked_mul(fc_in_usize)
        .ok_or_else(|| {
            anyhow!(
                "dispatch_eagle3_fc: seq_len ({}) * fc_input_size ({}) overflows usize",
                seq_len,
                fc_in_usize
            )
        })?;
    let actual_elems = concat_hidden_gpu.element_count();
    if actual_elems != expected_input_elems {
        return Err(anyhow!(
            "dispatch_eagle3_fc: concat_hidden has {} elements, expected {} (seq_len={} * fc_input_size={})",
            actual_elems, expected_input_elems, seq_len, fc_in
        ));
    }
    apply_linear_projection_f32(
        encoder, registry, device,
        concat_hidden_gpu, &tensors.fc,
        seq_len, fc_in, hidden,
    )
    .context("dispatch_eagle3_fc")
}

// ----------------------------------------------------------------
// Phase E4b.3 — first-layer pre-attention norms + concat
// ----------------------------------------------------------------
//
// Per vLLM `llama_eagle3.py:102-106`:
//
//     embeds = self.input_layernorm(embeds)
//     hidden_states, residual = self._residual_norm(hidden_states)
//     hidden_states = torch.cat([embeds, hidden_states], dim=-1)
//
// `_residual_norm` is `hidden_norm` per `llama_eagle3.py:69-75`.
// The concat is along the last dim (feature axis) yielding the
// `[seq, 2 * hidden_size]` input to Q/K/V projections.

/// Maximum `dim` that survives `as f32` without precision loss.
/// Above 2^24, f32 mantissa rounds to even — params[1] would lose
/// integer fidelity. Codex /cfa E4b.3 Minor (2026-05-22).
const RMS_NORM_DIM_F32_EXACT_MAX: u32 = 1 << 24;

/// Build the `[eps, dim]` F32 params buffer required by
/// `dispatch_rms_norm`. Allocates a fresh tiny buffer per call.
fn alloc_rms_norm_params_eagle3(
    device: &MlxDevice,
    eps: f32,
    dim: u32,
) -> Result<MlxBuffer> {
    if dim > RMS_NORM_DIM_F32_EXACT_MAX {
        return Err(anyhow!(
            "alloc_rms_norm_params_eagle3: dim {} exceeds 2^24 — `as f32` would round-to-even",
            dim
        ));
    }
    let mut params = device
        .alloc_buffer(8, DType::F32, vec![2])
        .map_err(|e| anyhow!("alloc eagle3 rms_norm params: {e}"))?;
    let slice = params
        .as_mut_slice::<f32>()
        .map_err(|e| anyhow!("eagle3 rms_norm params slice: {e}"))?;
    slice[0] = eps;
    slice[1] = dim as f32;
    Ok(params)
}

/// Shared internal helper: RMSNorm an `[seq, hidden_size]` F32
/// tensor against the given F32 RMSNorm weight. Returns a freshly
/// allocated F32 output buffer. Used by both `input_layernorm` and
/// `hidden_norm` since they have the same shape contract and
/// dtype rules.
///
/// Codex /cfa E4b.3 (preemptive applies E4b.2 patterns):
/// - dtype check at boundary (input + weight must be F32)
/// - u32::try_from for hidden_size
/// - checked_mul for element count
fn dispatch_eagle3_rms_norm_seq_x_hidden(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    norm_weight: &MlxBuffer,
    cfg: &Eagle3DrafterConfig,
    seq_len: u32,
    label: &str,
) -> Result<MlxBuffer> {
    if input.dtype() != DType::F32 {
        return Err(anyhow!(
            "dispatch_eagle3_rms_norm ({}): input dtype must be F32, got {:?}",
            label,
            input.dtype()
        ));
    }
    if norm_weight.dtype() != DType::F32 {
        return Err(anyhow!(
            "dispatch_eagle3_rms_norm ({}): norm weight dtype must be F32 (RMSNorm \
             weights cast BF16→F32 at upload per ADR-030 iter-106), got {:?}",
            label,
            norm_weight.dtype()
        ));
    }
    // Codex /cfa E4b.3 Major (2026-05-22): reject zero seq_len. A
    // zero-element dispatch is structurally meaningless and existing
    // kernels (e.g. mlx-native::ops::rms_norm) reject zero rows.
    if seq_len == 0 {
        return Err(anyhow!(
            "dispatch_eagle3_rms_norm ({}): seq_len must be > 0",
            label
        ));
    }
    let hidden_usize = cfg.hidden_size;
    if hidden_usize == 0 {
        return Err(anyhow!(
            "dispatch_eagle3_rms_norm ({}): hidden_size must be > 0",
            label
        ));
    }
    let hidden: u32 = u32::try_from(hidden_usize).map_err(|_| {
        anyhow!(
            "dispatch_eagle3_rms_norm ({}): hidden_size ({}) exceeds u32::MAX",
            label,
            hidden_usize
        )
    })?;
    // Codex /cfa E4b.3 Critical 3 (2026-05-22): validate weight
    // element count. Wrappers receive `&tensors.input_layernorm`
    // which is upload-time-validated, but the internal helper's
    // contract is "F32 RMSNorm weight" — adding the length check
    // here surfaces shape mismatches before the kernel reads past
    // the gain buffer.
    if norm_weight.element_count() != hidden_usize {
        return Err(anyhow!(
            "dispatch_eagle3_rms_norm ({}): weight has {} elements, expected hidden_size {}",
            label,
            norm_weight.element_count(),
            hidden_usize
        ));
    }
    let expected_elems = (seq_len as usize)
        .checked_mul(hidden_usize)
        .ok_or_else(|| {
            anyhow!(
                "dispatch_eagle3_rms_norm ({}): seq_len ({}) * hidden_size ({}) overflows usize",
                label,
                seq_len,
                hidden_usize
            )
        })?;
    let actual = input.element_count();
    if actual != expected_elems {
        return Err(anyhow!(
            "dispatch_eagle3_rms_norm ({}): input has {} elements, expected {} (seq_len={} * hidden_size={})",
            label, actual, expected_elems, seq_len, hidden
        ));
    }
    // Codex /cfa E4b.3 Critical 1 (2026-05-22): checked byte multiply.
    // Without this, `expected_elems * 4` could overflow usize on
    // release, causing alloc_buffer to allocate too small.
    let out_bytes = expected_elems
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            anyhow!(
                "dispatch_eagle3_rms_norm ({}): expected_elems ({}) * 4 overflows usize",
                label,
                expected_elems
            )
        })?;
    let out = device
        .alloc_buffer(
            out_bytes,
            DType::F32,
            vec![seq_len as usize, hidden_usize],
        )
        .map_err(|e| anyhow!("alloc {label} output: {e}"))?;
    let params = alloc_rms_norm_params_eagle3(device, cfg.rms_norm_eps, hidden)?;
    dispatch_rms_norm(
        encoder,
        registry,
        device.metal_device(),
        input,
        norm_weight,
        &out,
        &params,
        seq_len,
        hidden,
    )
    .with_context(|| format!("dispatch_rms_norm {label}"))?;
    Ok(out)
}

/// Dispatch `input_layernorm` of the embedded input tokens.
///
/// Mirrors vLLM `llama_eagle3.py:104` `embeds =
/// self.input_layernorm(embeds)`. Input: `[seq, hidden_size]` F32
/// embedding lookup result. Output: `[seq, hidden_size]` F32
/// normalized embeds, ready for concat with the hidden-states branch.
pub fn dispatch_eagle3_input_layernorm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    embeds_gpu: &MlxBuffer,
    tensors: &Eagle3DrafterTensors,
    cfg: &Eagle3DrafterConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    dispatch_eagle3_rms_norm_seq_x_hidden(
        encoder,
        registry,
        device,
        embeds_gpu,
        &tensors.input_layernorm,
        cfg,
        seq_len,
        "input_layernorm",
    )
}

/// Dispatch `hidden_norm` of the FC-projected target hidden state.
///
/// Mirrors vLLM `llama_eagle3.py:69` (`self.hidden_norm =
/// RMSNorm(config.hidden_size, ...)`) + line 84-86 / 91-93
/// (`_residual_norm` applies `hidden_norm`). Input: `[seq, hidden_size]`
/// F32 — typically the output of `dispatch_eagle3_fc`. Output:
/// `[seq, hidden_size]` F32 normalized, ready for concat.
pub fn dispatch_eagle3_hidden_norm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    fc_output_gpu: &MlxBuffer,
    tensors: &Eagle3DrafterTensors,
    cfg: &Eagle3DrafterConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    dispatch_eagle3_rms_norm_seq_x_hidden(
        encoder,
        registry,
        device,
        fc_output_gpu,
        &tensors.hidden_norm,
        cfg,
        seq_len,
        "hidden_norm",
    )
}

/// Concatenate `[seq, hidden_size]` embeds (column-left) with
/// `[seq, hidden_size]` hidden-states (column-right) into a fresh
/// `[seq, 2 * hidden_size]` F32 buffer.
///
/// Per vLLM `llama_eagle3.py:106`: `hidden_states = torch.cat(
/// [embeds, hidden_states], dim=-1)`. The output is the input to
/// the layer-0 Q/K/V projections (which have input width
/// `2 * hidden_size` — see `Eagle3DrafterConfig::qkv_input_width`).
///
/// Uses mlx-native's `dispatch_feature_concat_f32` primitive (a
/// strided memcpy — no FP arithmetic, byte-identical to the
/// reference torch.cat for any F32 input).
pub fn dispatch_eagle3_concat_2x_hidden(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    embeds_normed: &MlxBuffer,
    hidden_normed: &MlxBuffer,
    cfg: &Eagle3DrafterConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    if embeds_normed.dtype() != DType::F32 {
        return Err(anyhow!(
            "dispatch_eagle3_concat_2x_hidden: embeds_normed dtype must be F32, got {:?}",
            embeds_normed.dtype()
        ));
    }
    if hidden_normed.dtype() != DType::F32 {
        return Err(anyhow!(
            "dispatch_eagle3_concat_2x_hidden: hidden_normed dtype must be F32, got {:?}",
            hidden_normed.dtype()
        ));
    }
    // Codex /cfa E4b.3 Major (2026-05-22): reject zero seq_len.
    if seq_len == 0 {
        return Err(anyhow!(
            "dispatch_eagle3_concat_2x_hidden: seq_len must be > 0"
        ));
    }
    let hidden_usize = cfg.hidden_size;
    if hidden_usize == 0 {
        return Err(anyhow!(
            "dispatch_eagle3_concat_2x_hidden: hidden_size must be > 0"
        ));
    }
    let hidden: u32 = u32::try_from(hidden_usize).map_err(|_| {
        anyhow!(
            "dispatch_eagle3_concat_2x_hidden: hidden_size ({}) exceeds u32::MAX",
            hidden_usize
        )
    })?;
    let dst_stride: u32 = hidden.checked_mul(2).ok_or_else(|| {
        anyhow!(
            "dispatch_eagle3_concat_2x_hidden: 2 * hidden_size ({}) exceeds u32::MAX",
            hidden_usize
        )
    })?;
    let per_branch_elems = (seq_len as usize)
        .checked_mul(hidden_usize)
        .ok_or_else(|| {
            anyhow!(
                "dispatch_eagle3_concat_2x_hidden: seq_len ({}) * hidden_size ({}) overflows usize",
                seq_len,
                hidden_usize
            )
        })?;
    let total_elems = per_branch_elems.checked_mul(2).ok_or_else(|| {
        anyhow!(
            "dispatch_eagle3_concat_2x_hidden: dst total elements (2 * {}) overflows usize",
            per_branch_elems
        )
    })?;
    if embeds_normed.element_count() != per_branch_elems {
        return Err(anyhow!(
            "dispatch_eagle3_concat_2x_hidden: embeds_normed has {} elements, expected {} (seq_len={} * hidden_size={})",
            embeds_normed.element_count(),
            per_branch_elems,
            seq_len,
            hidden
        ));
    }
    if hidden_normed.element_count() != per_branch_elems {
        return Err(anyhow!(
            "dispatch_eagle3_concat_2x_hidden: hidden_normed has {} elements, expected {} (seq_len={} * hidden_size={})",
            hidden_normed.element_count(),
            per_branch_elems,
            seq_len,
            hidden
        ));
    }
    // Codex /cfa E4b.3 Critical 2 (2026-05-22): checked byte multiply
    // for dst allocation.
    let total_bytes = total_elems
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            anyhow!(
                "dispatch_eagle3_concat_2x_hidden: total_elems ({}) * 4 overflows usize",
                total_elems
            )
        })?;
    let dst = device
        .alloc_buffer(
            total_bytes,
            DType::F32,
            vec![seq_len as usize, 2 * hidden_usize],
        )
        .map_err(|e| anyhow!("alloc concat output: {e}"))?;
    // Marker — concat dispatch is appended below.
    // Copy embeds → dst columns [0, hidden).
    dispatch_feature_concat_f32(
        encoder,
        registry,
        device.metal_device(),
        embeds_normed,
        &dst,
        seq_len,
        hidden,
        0,
        dst_stride,
    )
    .context("dispatch_feature_concat_f32 embeds branch")?;
    // Copy hidden → dst columns [hidden, 2*hidden).
    dispatch_feature_concat_f32(
        encoder,
        registry,
        device.metal_device(),
        hidden_normed,
        &dst,
        seq_len,
        hidden,
        hidden,
        dst_stride,
    )
    .context("dispatch_feature_concat_f32 hidden branch")?;
    Ok(dst)
}

// ----------------------------------------------------------------
// Phase E4b.4 — Q/K/V projections from the concat input
// ----------------------------------------------------------------
//
// Per vLLM `llama_eagle3.py:111-115`: after the concat step produces
// `[seq, 2 * hidden_size]`, the layer-0 Q/K/V projections map it to:
//   - Q: `[seq, num_q_heads * head_dim]`
//   - K: `[seq, num_kv_heads * head_dim]`
//   - V: `[seq, num_kv_heads * head_dim]`
//
// Each projection is a BF16 GEMM (or GEMV for seq=1) wrapping
// `apply_linear_projection_f32`. Optional per-projection bias-add
// (configurable via `attention_bias`) is applied as a fused
// post-step via `dispatch_add_bias_row_2d_f32`.

/// Internal helper: dispatch one projection (`weight @ input^T`) +
/// optional bias-add. Returns the F32 output buffer.
///
/// Codex /cfa patterns applied:
/// - input dtype check at boundary (must be F32)
/// - u32::try_from for usize→u32 conversions
/// - checked_mul for both element-count + byte-multiply
/// - bias.element_count() validated against `out_features`
/// - zero-seq rejection
#[allow(clippy::too_many_arguments)]
fn dispatch_eagle3_projection_with_optional_bias(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    bias: Option<&MlxBuffer>,
    seq_len: u32,
    in_features_usize: usize,
    out_features_usize: usize,
    label: &str,
) -> Result<MlxBuffer> {
    if input.dtype() != DType::F32 {
        return Err(anyhow!(
            "dispatch_eagle3_projection ({}): input dtype must be F32, got {:?}",
            label,
            input.dtype()
        ));
    }
    if seq_len == 0 {
        return Err(anyhow!(
            "dispatch_eagle3_projection ({}): seq_len must be > 0",
            label
        ));
    }
    if in_features_usize == 0 || out_features_usize == 0 {
        return Err(anyhow!(
            "dispatch_eagle3_projection ({}): in_features ({}) and out_features ({}) must be > 0",
            label,
            in_features_usize,
            out_features_usize
        ));
    }
    let in_features: u32 = u32::try_from(in_features_usize).map_err(|_| {
        anyhow!(
            "dispatch_eagle3_projection ({}): in_features ({}) exceeds u32::MAX",
            label,
            in_features_usize
        )
    })?;
    let out_features: u32 = u32::try_from(out_features_usize).map_err(|_| {
        anyhow!(
            "dispatch_eagle3_projection ({}): out_features ({}) exceeds u32::MAX",
            label,
            out_features_usize
        )
    })?;
    // Validate input element count.
    let expected_in_elems = (seq_len as usize)
        .checked_mul(in_features_usize)
        .ok_or_else(|| {
            anyhow!(
                "dispatch_eagle3_projection ({}): seq_len ({}) * in_features ({}) overflows usize",
                label,
                seq_len,
                in_features_usize
            )
        })?;
    if input.element_count() != expected_in_elems {
        return Err(anyhow!(
            "dispatch_eagle3_projection ({}): input has {} elements, expected {} (seq_len={} * in_features={})",
            label,
            input.element_count(),
            expected_in_elems,
            seq_len,
            in_features
        ));
    }
    // Validate weight dtype + element count.
    if weight.dtype() != DType::BF16 {
        return Err(anyhow!(
            "dispatch_eagle3_projection ({}): weight dtype must be BF16, got {:?}",
            label,
            weight.dtype()
        ));
    }
    let expected_w_elems = out_features_usize
        .checked_mul(in_features_usize)
        .ok_or_else(|| {
            anyhow!(
                "dispatch_eagle3_projection ({}): out * in overflows usize",
                label
            )
        })?;
    if weight.element_count() != expected_w_elems {
        return Err(anyhow!(
            "dispatch_eagle3_projection ({}): weight has {} elements, expected {} (out * in = {} * {})",
            label,
            weight.element_count(),
            expected_w_elems,
            out_features,
            in_features
        ));
    }
    // Validate bias if present.
    if let Some(b) = bias {
        if b.dtype() != DType::F32 {
            return Err(anyhow!(
                "dispatch_eagle3_projection ({}): bias dtype must be F32 (cast from BF16 at upload), got {:?}",
                label,
                b.dtype()
            ));
        }
        if b.element_count() != out_features_usize {
            return Err(anyhow!(
                "dispatch_eagle3_projection ({}): bias has {} elements, expected out_features {}",
                label,
                b.element_count(),
                out_features
            ));
        }
    }
    // Codex /cfa E4b.4 Major (2026-05-22): bound the OUTPUT product
    // before the callee allocates `(seq_len * out_features) * 4` bytes
    // (where the multiply is `u32` and can wrap). Also relevant for
    // the bias-add kernel which computes `params.m * params.n` as
    // uint internally — products above u32::MAX corrupt the grid.
    let out_elems_usize = (seq_len as usize)
        .checked_mul(out_features_usize)
        .ok_or_else(|| {
            anyhow!(
                "dispatch_eagle3_projection ({}): seq_len ({}) * out_features ({}) overflows usize",
                label,
                seq_len,
                out_features_usize
            )
        })?;
    if out_elems_usize > (u32::MAX as usize) {
        return Err(anyhow!(
            "dispatch_eagle3_projection ({}): output elements ({}) exceeds u32::MAX (downstream kernels use u32 grid)",
            label,
            out_elems_usize
        ));
    }
    // Dispatch the matmul.
    let out = apply_linear_projection_f32(
        encoder, registry, device,
        input, weight,
        seq_len, in_features, out_features,
    )
    .with_context(|| format!("apply_linear_projection_f32 {label}"))?;
    // Optional bias add — `add_bias_row_2d_f32` broadcasts `[N]` bias
    // across rows of the `[M, N]` output, in-place.
    if let Some(b) = bias {
        // Memory barrier between the matmul (writes `out`) and the
        // bias-add (reads+writes `out` in-place). Metal command
        // encoders don't insert implicit barriers between sequential
        // compute dispatches with R/W dependency on the same buffer
        // (mirrors qwen35/gpu_full_attn.rs:901 pattern).
        encoder.memory_barrier();
        dispatch_add_bias_row_2d_f32(
            encoder,
            registry,
            device.metal_device(),
            &out,
            b,
            &out,
            seq_len,
            out_features,
        )
        .with_context(|| format!("dispatch_add_bias_row_2d_f32 {label}"))?;
    }
    Ok(out)
}

/// Dispatch the Q projection from the layer-0 concat input.
///
/// Input: `[seq, 2 * hidden_size]` F32 (output of
/// `dispatch_eagle3_concat_2x_hidden`).
/// Weight: `[num_q_heads * head_dim, 2 * hidden_size]` BF16.
/// Optional bias: `[num_q_heads * head_dim]` F32.
/// Output: `[seq, num_q_heads * head_dim]` F32.
pub fn dispatch_eagle3_q_proj(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    concat_input: &MlxBuffer,
    tensors: &Eagle3DrafterTensors,
    cfg: &Eagle3DrafterConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    dispatch_eagle3_projection_with_optional_bias(
        encoder, registry, device,
        concat_input,
        &tensors.q_proj,
        tensors.q_bias.as_ref(),
        seq_len,
        cfg.qkv_input_width(),
        cfg.q_proj_out(),
        "q_proj",
    )
}

/// Dispatch the K projection from the layer-0 concat input.
///
/// Input: `[seq, 2 * hidden_size]` F32. Weight: BF16. Optional bias.
/// Output: `[seq, num_kv_heads * head_dim]` F32.
pub fn dispatch_eagle3_k_proj(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    concat_input: &MlxBuffer,
    tensors: &Eagle3DrafterTensors,
    cfg: &Eagle3DrafterConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    dispatch_eagle3_projection_with_optional_bias(
        encoder, registry, device,
        concat_input,
        &tensors.k_proj,
        tensors.k_bias.as_ref(),
        seq_len,
        cfg.qkv_input_width(),
        cfg.kv_proj_out(),
        "k_proj",
    )
}

/// Dispatch the V projection from the layer-0 concat input.
///
/// Input: `[seq, 2 * hidden_size]` F32. Weight: BF16. Optional bias.
/// Output: `[seq, num_kv_heads * head_dim]` F32.
pub fn dispatch_eagle3_v_proj(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    concat_input: &MlxBuffer,
    tensors: &Eagle3DrafterTensors,
    cfg: &Eagle3DrafterConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    dispatch_eagle3_projection_with_optional_bias(
        encoder, registry, device,
        concat_input,
        &tensors.v_proj,
        tensors.v_bias.as_ref(),
        seq_len,
        cfg.qkv_input_width(),
        cfg.kv_proj_out(),
        "v_proj",
    )
}

// ----------------------------------------------------------------
// Phase E4b.5a — Q/K per-head RMSNorm (Qwen-style)
// ----------------------------------------------------------------
//
// Per-head normalization: input `[seq, num_heads, head_dim]` flat
// gets RMSNorm applied along the `head_dim` axis using the F32
// `q_norm.weight` / `k_norm.weight` of shape `[head_dim]`.
//
// The flat input layout `[seq * num_heads * head_dim]` lets us
// dispatch `dispatch_rms_norm` with rows = seq * num_heads,
// dim = head_dim (same trick as
// `dflash::forward::dispatch_dflash_head_norm`).
//
// Gated by `cfg.use_qk_norm` (Qwen-3 style). Llama-style targets
// have this off and the dispatch is skipped at the orchestrator.

/// Internal helper for both Q and K head-norms. Returns a freshly
/// allocated F32 output of the same shape as `proj`.
#[allow(clippy::too_many_arguments)]
fn dispatch_eagle3_head_norm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    proj: &MlxBuffer,
    norm_weight: &MlxBuffer,
    cfg: &Eagle3DrafterConfig,
    seq_len: u32,
    num_heads: u32,
    label: &str,
) -> Result<MlxBuffer> {
    if proj.dtype() != DType::F32 {
        return Err(anyhow!(
            "dispatch_eagle3_head_norm ({}): proj dtype must be F32, got {:?}",
            label,
            proj.dtype()
        ));
    }
    if norm_weight.dtype() != DType::F32 {
        return Err(anyhow!(
            "dispatch_eagle3_head_norm ({}): norm weight dtype must be F32 (BF16→F32 at upload), got {:?}",
            label,
            norm_weight.dtype()
        ));
    }
    if seq_len == 0 || num_heads == 0 {
        return Err(anyhow!(
            "dispatch_eagle3_head_norm ({}): seq_len ({}) and num_heads ({}) must both be > 0",
            label, seq_len, num_heads
        ));
    }
    let head_dim_usize = cfg.head_dim;
    if head_dim_usize == 0 {
        return Err(anyhow!(
            "dispatch_eagle3_head_norm ({}): head_dim must be > 0",
            label
        ));
    }
    let head_dim: u32 = u32::try_from(head_dim_usize).map_err(|_| {
        anyhow!(
            "dispatch_eagle3_head_norm ({}): head_dim ({}) exceeds u32::MAX",
            label,
            head_dim_usize
        )
    })?;
    if head_dim > RMS_NORM_DIM_F32_EXACT_MAX {
        return Err(anyhow!(
            "dispatch_eagle3_head_norm ({}): head_dim ({}) exceeds 2^24 — params[1] would lose F32 precision",
            label,
            head_dim
        ));
    }
    if norm_weight.element_count() != head_dim_usize {
        return Err(anyhow!(
            "dispatch_eagle3_head_norm ({}): weight has {} elements, expected head_dim {}",
            label,
            norm_weight.element_count(),
            head_dim
        ));
    }
    // rows = seq * num_heads. checked + bounded by u32::MAX for the
    // kernel's grid (mirrors E4b.4 output-product bound).
    let rows_usize = (seq_len as usize)
        .checked_mul(num_heads as usize)
        .ok_or_else(|| {
            anyhow!(
                "dispatch_eagle3_head_norm ({}): seq_len ({}) * num_heads ({}) overflows usize",
                label, seq_len, num_heads
            )
        })?;
    if rows_usize > (u32::MAX as usize) {
        return Err(anyhow!(
            "dispatch_eagle3_head_norm ({}): rows ({}) exceeds u32::MAX",
            label,
            rows_usize
        ));
    }
    let rows: u32 = rows_usize as u32;
    // expected element count for proj = rows * head_dim
    let expected_elems = rows_usize
        .checked_mul(head_dim_usize)
        .ok_or_else(|| {
            anyhow!(
                "dispatch_eagle3_head_norm ({}): rows ({}) * head_dim ({}) overflows usize",
                label, rows_usize, head_dim_usize
            )
        })?;
    if proj.element_count() != expected_elems {
        return Err(anyhow!(
            "dispatch_eagle3_head_norm ({}): proj has {} elements, expected {} (rows={} * head_dim={})",
            label, proj.element_count(), expected_elems, rows, head_dim
        ));
    }
    let out_bytes = expected_elems
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            anyhow!(
                "dispatch_eagle3_head_norm ({}): expected_elems ({}) * 4 overflows usize",
                label, expected_elems
            )
        })?;
    let out = device
        .alloc_buffer(
            out_bytes,
            DType::F32,
            vec![seq_len as usize, num_heads as usize, head_dim_usize],
        )
        .map_err(|e| anyhow!("alloc {label} output: {e}"))?;
    let params = alloc_rms_norm_params_eagle3(device, cfg.rms_norm_eps, head_dim)?;
    // Codex /cfa E4b.5a Major (2026-05-22): RAW barrier before
    // dispatch_rms_norm reads `proj`. Q/K head-norm is chained
    // after q_proj/k_proj in the orchestrator (E4b.6+), and Metal
    // serial command queues don't insert implicit barriers between
    // sequential compute dispatches with R/W dependency on the
    // same buffer (see E4b.4 commit message — same class of bug
    // caused diffs 0.156-0.794 there). Insert at the wrapper so
    // callers don't have to remember.
    encoder.memory_barrier();
    dispatch_rms_norm(
        encoder,
        registry,
        device.metal_device(),
        proj,
        norm_weight,
        &out,
        &params,
        rows,
        head_dim,
    )
    .with_context(|| format!("dispatch_rms_norm {label}"))?;
    Ok(out)
}

/// Per-head RMSNorm on the Q projection output.
///
/// Input: `[seq, num_q_heads * head_dim]` F32 (q_proj output).
/// Output: `[seq, num_q_heads, head_dim]` F32 normalized.
/// Returns an error if `cfg.use_qk_norm = false` (the orchestrator
/// must check the gate; calling this in non-QK-norm mode is a bug).
pub fn dispatch_eagle3_q_head_norm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q_proj_out: &MlxBuffer,
    tensors: &Eagle3DrafterTensors,
    cfg: &Eagle3DrafterConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    // Codex /cfa E4b.5a Major (2026-05-22): enforce config gate at
    // the wrapper. Without this, an inconsistent (cfg=false, tensor=Some)
    // bundle would silently apply Q-norm in non-QK mode, feeding wrong
    // values into RoPE.
    if !cfg.use_qk_norm {
        return Err(anyhow!(
            "dispatch_eagle3_q_head_norm: cfg.use_qk_norm is false — orchestrator must check the gate before calling"
        ));
    }
    let q_norm = tensors.q_norm.as_ref().ok_or_else(|| {
        anyhow!(
            "dispatch_eagle3_q_head_norm: q_norm absent (use_qk_norm = {})",
            cfg.use_qk_norm
        )
    })?;
    let num_q_heads: u32 = u32::try_from(cfg.num_q_heads).map_err(|_| {
        anyhow!(
            "dispatch_eagle3_q_head_norm: num_q_heads ({}) exceeds u32::MAX",
            cfg.num_q_heads
        )
    })?;
    dispatch_eagle3_head_norm(
        encoder, registry, device,
        q_proj_out, q_norm, cfg, seq_len, num_q_heads,
        "q_norm",
    )
}

/// Per-head RMSNorm on the K projection output.
///
/// Input: `[seq, num_kv_heads * head_dim]` F32 (k_proj output).
/// Output: `[seq, num_kv_heads, head_dim]` F32 normalized.
pub fn dispatch_eagle3_k_head_norm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    k_proj_out: &MlxBuffer,
    tensors: &Eagle3DrafterTensors,
    cfg: &Eagle3DrafterConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    if !cfg.use_qk_norm {
        return Err(anyhow!(
            "dispatch_eagle3_k_head_norm: cfg.use_qk_norm is false — orchestrator must check the gate before calling"
        ));
    }
    let k_norm = tensors.k_norm.as_ref().ok_or_else(|| {
        anyhow!(
            "dispatch_eagle3_k_head_norm: k_norm absent (use_qk_norm = {})",
            cfg.use_qk_norm
        )
    })?;
    let num_kv_heads: u32 = u32::try_from(cfg.num_kv_heads).map_err(|_| {
        anyhow!(
            "dispatch_eagle3_k_head_norm: num_kv_heads ({}) exceeds u32::MAX",
            cfg.num_kv_heads
        )
    })?;
    dispatch_eagle3_head_norm(
        encoder, registry, device,
        k_proj_out, k_norm, cfg, seq_len, num_kv_heads,
        "k_norm",
    )
}

// ----------------------------------------------------------------
// Phase E4b.5b — RoPE on Q/K with tree-position support
// ----------------------------------------------------------------
//
// Per vLLM `llama_eagle3.py:112-115` the self-attn block is called
// with positions:
//
//     hidden_states = self.self_attn(positions=positions,
//                                    hidden_states=hidden_states)
//
// For dynamic tree decoding (Phase E4a), each tree node has a
// distinct absolute position. The caller computes these once per
// expansion step and passes them as a slice of `[u32]`. Linear-chain
// callers can pass `None` to get `base_pos..base_pos+seq_len`.
//
// We use the same NeoX-style RoPE primitive (`apply_imrope`) DFlash
// uses, with the plain-NeoX section layout `[rope_dim/2, 0, 0, 0]`.

/// Build the position buffer required by `apply_imrope`. The imrope
/// kernel expects 4 axes × `seq_len` int32 positions (one per
/// mrope-section), but plain NeoX RoPE puts all rotation in axis 0
/// so the other 3 axes are functionally ignored. We replicate the
/// same value to all 4 axes (matches DFlash's `build_dflash_pos_buf`).
fn build_eagle3_pos_buf(
    device: &MlxDevice,
    seq_len: u32,
    positions_override: Option<&[u32]>,
    base_pos: u32,
) -> Result<MlxBuffer> {
    let l = seq_len as usize;
    if let Some(p) = positions_override {
        if p.len() != l {
            return Err(anyhow!(
                "build_eagle3_pos_buf: positions_override len {} != seq_len {}",
                p.len(),
                seq_len
            ));
        }
    }
    let n_pos = 4 * l;
    let mut buf = device
        .alloc_buffer(n_pos * 4, DType::I32, vec![n_pos])
        .map_err(|e| anyhow!("alloc eagle3 rope pos_buf: {e}"))?;
    let slice = buf
        .as_mut_slice::<i32>()
        .map_err(|e| anyhow!("eagle3 rope pos_buf slice: {e}"))?;
    // Codex /cfa E4b.5b Major (2026-05-22): reject positions above
    // i32::MAX instead of silently saturating (saturation produces
    // wrong RoPE angles).
    let pos_values: Vec<i32> = if let Some(p) = positions_override {
        let mut out = Vec::with_capacity(p.len());
        for (i, &v) in p.iter().enumerate() {
            let iv = i32::try_from(v).map_err(|_| {
                anyhow!(
                    "build_eagle3_pos_buf: positions_override[{}] = {} exceeds i32::MAX (kernel uses signed i32 positions)",
                    i,
                    v
                )
            })?;
            out.push(iv);
        }
        out
    } else {
        let mut out = Vec::with_capacity(l);
        for i in 0..l {
            let v = (base_pos as i64).checked_add(i as i64).ok_or_else(|| {
                anyhow!(
                    "build_eagle3_pos_buf: base_pos ({}) + {} overflows i64",
                    base_pos,
                    i
                )
            })?;
            if v > (i32::MAX as i64) {
                return Err(anyhow!(
                    "build_eagle3_pos_buf: linear position {} (base_pos {} + offset {}) exceeds i32::MAX",
                    v,
                    base_pos,
                    i
                ));
            }
            out.push(v as i32);
        }
        out
    };
    for axis in 0..4 {
        let dst = &mut slice[axis * l..(axis + 1) * l];
        dst.copy_from_slice(&pos_values);
    }
    Ok(buf)
}

/// Apply NeoX-style RoPE to a per-head-normalized (or raw) Q or K
/// buffer.
///
/// # Arguments
///
/// * `qk_in`: `[seq_len * num_heads, head_dim]` F32 — post-head-norm
///   when `cfg.use_qk_norm`, post-projection-reshape otherwise.
/// * `seq_len`: number of token (or tree-node) positions in the input.
/// * `num_heads`: per-head count for this Q or K call.
/// * `positions_override`: when `Some`, per-position `u32` index used
///   for RoPE rotation — typically `base_pos + tree_depths[i]` for
///   dynamic-tree decoding. When `None`, falls back to linear
///   `base_pos..base_pos + seq_len`.
/// * `base_pos`: linear-chain base offset (used when
///   `positions_override = None`, ignored otherwise).
///
/// Returns `[seq_len * num_heads, head_dim]` F32 rotated buffer.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_eagle3_rope(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    qk_in: &MlxBuffer,
    cfg: &Eagle3DrafterConfig,
    seq_len: u32,
    num_heads: u32,
    positions_override: Option<&[u32]>,
    base_pos: u32,
    label: &str,
) -> Result<MlxBuffer> {
    // Codex-validated boundary checks.
    if qk_in.dtype() != DType::F32 {
        return Err(anyhow!(
            "dispatch_eagle3_rope ({}): qk_in dtype must be F32, got {:?}",
            label,
            qk_in.dtype()
        ));
    }
    if seq_len == 0 || num_heads == 0 {
        return Err(anyhow!(
            "dispatch_eagle3_rope ({}): seq_len ({}) and num_heads ({}) must be > 0",
            label,
            seq_len,
            num_heads
        ));
    }
    let head_dim_usize = cfg.head_dim;
    let rope_dim_usize = cfg.rope_dim;
    let head_dim: u32 = u32::try_from(head_dim_usize).map_err(|_| {
        anyhow!(
            "dispatch_eagle3_rope ({}): head_dim ({}) exceeds u32::MAX",
            label,
            head_dim_usize
        )
    })?;
    let rope_dim: u32 = u32::try_from(rope_dim_usize).map_err(|_| {
        anyhow!(
            "dispatch_eagle3_rope ({}): rope_dim ({}) exceeds u32::MAX",
            label,
            rope_dim_usize
        )
    })?;
    // Validate qk_in element count.
    let expected_elems = (seq_len as usize)
        .checked_mul(num_heads as usize)
        .and_then(|v| v.checked_mul(head_dim_usize))
        .ok_or_else(|| {
            anyhow!(
                "dispatch_eagle3_rope ({}): seq * heads * head_dim overflows usize",
                label
            )
        })?;
    if expected_elems > (u32::MAX as usize) {
        return Err(anyhow!(
            "dispatch_eagle3_rope ({}): expected_elems ({}) exceeds u32::MAX",
            label,
            expected_elems
        ));
    }
    if qk_in.element_count() != expected_elems {
        return Err(anyhow!(
            "dispatch_eagle3_rope ({}): qk_in has {} elements, expected {} (seq={} heads={} head_dim={})",
            label,
            qk_in.element_count(),
            expected_elems,
            seq_len,
            num_heads,
            head_dim
        ));
    }
    // Build positions buffer. Plain NeoX = all pairs in axis 0:
    // mrope_section[0] = rope_dim / 2; others = 0.
    let positions = build_eagle3_pos_buf(device, seq_len, positions_override, base_pos)?;
    let sections = [rope_dim / 2, 0, 0, 0];
    // Memory barrier so apply_imrope reads the just-written positions
    // and any upstream write to qk_in (e.g. head-norm output).
    encoder.memory_barrier();
    apply_imrope(
        encoder,
        registry,
        device,
        qk_in,
        &positions,
        seq_len,
        num_heads,
        head_dim,
        rope_dim,
        cfg.rope_theta,
        sections,
    )
    .with_context(|| format!("apply_imrope {label}"))
}

// ----------------------------------------------------------------
// Phase E4b.6 — tree attention dispatch (Phase E1 kernel integration)
// ----------------------------------------------------------------
//
// Wraps mlx-native's `tree_attention` (Phase E1) with Eagle3 config
// validation and buffer-shape contracts. The caller supplies Q/K/V
// already in head-outer layout (`[num_heads, q_seq_len, head_dim]` for
// Q; `[num_kv_heads, kv_capacity, head_dim]` for K/V) and a tree mask
// from `ExpandedTree::build_tree_mask` (Phase E4a).
//
// Layout note: Q/K/V projections in earlier sub-phases produce
// `[seq, num_heads * head_dim]` (seq-outer). A permute_021_f32 step
// is needed to feed this dispatch; the orchestrator (E4b.9 future)
// will own that step so this wrapper stays focused on the kernel
// surface.

/// Sentinel value for attended positions in the tree mask. Matches
/// `mlx_native::ops::tree_attention::TREE_MASK_ATTENDED` (0.0) so
/// callers don't need to depend on mlx-native to construct masks.
pub const EAGLE3_TREE_MASK_ATTENDED: f32 = 0.0;

/// Sentinel value for masked positions (-65504.0 = -MAXHALF; matches
/// the flash_attn_vec implicit-causal sentinel that tree_attention
/// chains through for early-exit on all-masked chunks).
pub const EAGLE3_TREE_MASK_MASKED: f32 = -65504.0;

/// Dispatch tree-aware self-attention.
///
/// # Arguments
/// * `q` — F32 `[num_q_heads, q_seq_len, head_dim]` (post-RoPE,
///   head-outer; caller must permute from `[seq, n_q, hd]` if upstream
///   primitives produced seq-outer).
/// * `k` — F32 or F16 `[num_kv_heads, kv_capacity, head_dim]`
///   (post-RoPE K cache; only first `q_seq_len` positions valid for
///   tree decoding).
/// * `v` — same dtype + shape as `k` (post-projection V cache).
/// * `tree_mask` — F32 `[q_seq_len, mask_stride]` from
///   `ExpandedTree::build_tree_mask` (Phase E4a). Cell `(i, j)`
///   is `EAGLE3_TREE_MASK_ATTENDED` (0.0) if tree-node `i` can
///   attend to KV position `j`, `EAGLE3_TREE_MASK_MASKED`
///   (-65504.0) otherwise.
/// * `kv_seq_len` — number of valid KV positions in `k` and `v`
///   (typically `prefix_len + q_seq_len` for tree decoding).
/// * `kv_capacity` — allocated capacity (stride between KV heads
///   in `k` / `v`; must be `>= kv_seq_len`).
/// * `mask_stride` — stride between tree mask rows (must be
///   `>= kv_seq_len`).
/// * `q_seq_len` — number of tree-node queries (tree.len()).
/// * `scale` — typically `1.0 / sqrt(head_dim)`.
///
/// Returns a freshly allocated F32 output buffer with layout
/// `[q_seq_len, num_q_heads, head_dim]` (query-outer, head-inner,
/// dim-innermost — see Phase E1 layout contract).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_eagle3_tree_attention(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    tree_mask: &MlxBuffer,
    cfg: &Eagle3DrafterConfig,
    q_seq_len: u32,
    kv_seq_len: u32,
    kv_capacity: u32,
    mask_stride: u32,
    scale: f32,
) -> Result<MlxBuffer> {
    // Codex-style validation patterns (preemptively applied).
    if q_seq_len == 0 {
        return Err(anyhow!(
            "dispatch_eagle3_tree_attention: q_seq_len must be > 0"
        ));
    }
    if kv_seq_len == 0 {
        return Err(anyhow!(
            "dispatch_eagle3_tree_attention: kv_seq_len must be > 0"
        ));
    }
    if kv_capacity < kv_seq_len {
        return Err(anyhow!(
            "dispatch_eagle3_tree_attention: kv_capacity ({}) must be >= kv_seq_len ({})",
            kv_capacity,
            kv_seq_len
        ));
    }
    if mask_stride < kv_seq_len {
        return Err(anyhow!(
            "dispatch_eagle3_tree_attention: mask_stride ({}) must be >= kv_seq_len ({})",
            mask_stride,
            kv_seq_len
        ));
    }
    if !scale.is_finite() {
        return Err(anyhow!(
            "dispatch_eagle3_tree_attention: scale ({}) must be finite",
            scale
        ));
    }

    let num_q_heads: u32 = u32::try_from(cfg.num_q_heads).map_err(|_| {
        anyhow!(
            "dispatch_eagle3_tree_attention: num_q_heads ({}) exceeds u32::MAX",
            cfg.num_q_heads
        )
    })?;
    let num_kv_heads: u32 = u32::try_from(cfg.num_kv_heads).map_err(|_| {
        anyhow!(
            "dispatch_eagle3_tree_attention: num_kv_heads ({}) exceeds u32::MAX",
            cfg.num_kv_heads
        )
    })?;
    let head_dim: u32 = u32::try_from(cfg.head_dim).map_err(|_| {
        anyhow!(
            "dispatch_eagle3_tree_attention: head_dim ({}) exceeds u32::MAX",
            cfg.head_dim
        )
    })?;

    // Output buffer allocation: [q_seq_len, num_q_heads, head_dim] F32.
    let out_elems = (q_seq_len as usize)
        .checked_mul(num_q_heads as usize)
        .and_then(|v| v.checked_mul(cfg.head_dim))
        .ok_or_else(|| {
            anyhow!(
                "dispatch_eagle3_tree_attention: out elements (q={} * n_q={} * hd={}) overflows usize",
                q_seq_len,
                num_q_heads,
                cfg.head_dim
            )
        })?;
    if out_elems > (u32::MAX as usize) {
        return Err(anyhow!(
            "dispatch_eagle3_tree_attention: out elements ({}) exceeds u32::MAX",
            out_elems
        ));
    }
    let out_bytes = out_elems
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            anyhow!(
                "dispatch_eagle3_tree_attention: out bytes ({} * 4) overflows usize",
                out_elems
            )
        })?;
    let output = device
        .alloc_buffer(
            out_bytes,
            DType::F32,
            vec![q_seq_len as usize, num_q_heads as usize, cfg.head_dim],
        )
        .map_err(|e| anyhow!("alloc tree_attention output: {e}"))?;

    // tmp buffer for the reduce pass: sized by mlx-native helper.
    let tmp_bytes =
        tree_attn_ops::tmp_buffer_bytes(num_q_heads, head_dim, q_seq_len);
    let tmp = device
        .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
        .map_err(|e| anyhow!("alloc tree_attention tmp: {e}"))?;

    let params = TreeAttentionParams {
        num_heads: num_q_heads,
        num_kv_heads,
        head_dim,
        kv_seq_len,
        kv_capacity,
        scale,
        q_seq_len,
        mask_stride,
    };

    // Memory barrier: tree_attention reads Q, K, V, tree_mask that
    // were written by upstream dispatches (RoPE, projections, mask
    // upload) in the same encoder.
    encoder.memory_barrier();

    tree_attn_ops::tree_attention(
        encoder,
        registry,
        device,
        q,
        k,
        v,
        tree_mask,
        &output,
        &tmp,
        &params,
    )
    .context("tree_attention")?;

    Ok(output)
}

// ----------------------------------------------------------------
// Phase E4b.7 — O projection + residual add
// ----------------------------------------------------------------
//
// After tree_attention produces `[q_seq, num_q_heads, head_dim]`
// F32, the O projection maps it back to `[q_seq, hidden_size]` via
// the BF16 o_proj weight `[hidden_size, num_q_heads * head_dim]`.
// Per vLLM `llama_eagle3.py:117`, the residual stream then adds
// the pre-attention input back.
//
// Layout reuse: tree_attention's `[q_seq, num_q_heads, head_dim]`
// row-major flat IS equivalent to `[q_seq, num_q_heads * head_dim]`
// row-major flat (trailing dims contiguous) — no permute step needed
// to feed into apply_linear_projection_f32.

/// Dispatch the O projection from tree_attention output.
///
/// Input: `[q_seq, num_q_heads, head_dim]` F32 (tree_attention output;
/// also valid as `[q_seq, num_q_heads * head_dim]` since the trailing
/// dims are contiguous).
/// Weight: `[hidden_size, num_q_heads * head_dim]` BF16.
/// Optional bias: `[hidden_size]` F32 (gated by `attention_bias`).
/// Output: `[q_seq, hidden_size]` F32.
pub fn dispatch_eagle3_o_proj(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    attn_out: &MlxBuffer,
    tensors: &Eagle3DrafterTensors,
    cfg: &Eagle3DrafterConfig,
    q_seq_len: u32,
) -> Result<MlxBuffer> {
    // Codex /cfa E4b.7 Major (2026-05-22): RAW barrier before reading
    // attn_out. In the chained orchestrator path, attn_out is
    // tree_attention's output (written by the prior kernel dispatch in
    // the same encoder). Without this barrier, the o_proj matmul could
    // read partially-written data. Mirrors the pattern from E4b.5a
    // (head_norm) and E4b.6 (tree_attention).
    encoder.memory_barrier();
    dispatch_eagle3_projection_with_optional_bias(
        encoder, registry, device,
        attn_out,
        &tensors.o_proj,
        tensors.o_bias.as_ref(),
        q_seq_len,
        cfg.q_proj_out(), // input width = num_q_heads * head_dim
        cfg.hidden_size,  // output width = hidden_size
        "o_proj",
    )
}

/// Element-wise residual add: `out = a + b`. Both inputs must be F32
/// `[seq, hidden_size]` (same shape).
///
/// Allocates a fresh output buffer (vLLM keeps both `hidden_states`
/// and `residual` live for the next layer; allocating fresh avoids
/// the in-place R/W barrier needed for elementwise+matmul chains).
pub fn dispatch_eagle3_residual_add(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    a: &MlxBuffer,
    b: &MlxBuffer,
    cfg: &Eagle3DrafterConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    if a.dtype() != DType::F32 || b.dtype() != DType::F32 {
        return Err(anyhow!(
            "dispatch_eagle3_residual_add: inputs must be F32, got a={:?} b={:?}",
            a.dtype(),
            b.dtype()
        ));
    }
    if seq_len == 0 {
        return Err(anyhow!(
            "dispatch_eagle3_residual_add: seq_len must be > 0"
        ));
    }
    let hidden_usize = cfg.hidden_size;
    if hidden_usize == 0 {
        return Err(anyhow!(
            "dispatch_eagle3_residual_add: hidden_size must be > 0"
        ));
    }
    let n_elements = (seq_len as usize)
        .checked_mul(hidden_usize)
        .ok_or_else(|| {
            anyhow!(
                "dispatch_eagle3_residual_add: seq_len ({}) * hidden_size ({}) overflows usize",
                seq_len,
                hidden_usize
            )
        })?;
    if n_elements > (u32::MAX as usize) {
        return Err(anyhow!(
            "dispatch_eagle3_residual_add: n_elements ({}) exceeds u32::MAX",
            n_elements
        ));
    }
    if a.element_count() != n_elements {
        return Err(anyhow!(
            "dispatch_eagle3_residual_add: a has {} elements, expected {} (seq={} * hidden={})",
            a.element_count(),
            n_elements,
            seq_len,
            hidden_usize
        ));
    }
    if b.element_count() != n_elements {
        return Err(anyhow!(
            "dispatch_eagle3_residual_add: b has {} elements, expected {} (seq={} * hidden={})",
            b.element_count(),
            n_elements,
            seq_len,
            hidden_usize
        ));
    }
    let out_bytes = n_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            anyhow!(
                "dispatch_eagle3_residual_add: n_elements * 4 overflows usize"
            )
        })?;
    let out = device
        .alloc_buffer(out_bytes, DType::F32, vec![seq_len as usize, hidden_usize])
        .map_err(|e| anyhow!("alloc residual_add output: {e}"))?;
    // Memory barrier: inputs may have been written by upstream
    // dispatches in the same encoder (o_proj writes a; pre-attn
    // residual may have been computed earlier).
    encoder.memory_barrier();
    elementwise_add(
        encoder,
        registry,
        device.metal_device(),
        a,
        b,
        &out,
        n_elements,
        DType::F32,
    )
    .map_err(|e| anyhow!("elementwise_add residual: {e}"))?;
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::inference::spec_decode::eagle3::weights::{
        expected_manifest, Eagle3Weights, ExpectedTensor,
    };
    use mlx_native::DType;
    use safetensors::tensor::{Dtype as SafeDtype, TensorView};
    use std::collections::BTreeMap;

    /// Small config so synthetic-blob tests run fast.
    fn tiny_cfg() -> Eagle3DrafterConfig {
        Eagle3DrafterConfig {
            hidden_size: 256,
            intermediate_size: 512,
            head_dim: 32,
            num_q_heads: 8,
            num_kv_heads: 4,
            vocab_size: 1000,
            draft_vocab_size: 1000,
            target_hidden_size: 256,
            num_aux_hidden_states: 3,
            rms_norm_eps: 1e-6,
            norm_before_fc: false,
            fc_norm: false, // simpler manifest for parity test
            use_qk_norm: false,
            attention_bias: false,
            tie_lm_head: true, // simpler
            include_draft_id_mapping: false,
            has_own_embed_tokens: false,
            rope_theta: 1_000_000.0,
            rope_dim: 32, // matches head_dim in tiny_cfg
        }
    }

    /// Truncate F32 → BF16 (top 16 bits) and serialize as little-endian.
    fn f32_to_bf16_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for v in values {
            let bf16_bits = (v.to_bits() >> 16) as u16;
            out.push((bf16_bits & 0xff) as u8);
            out.push(((bf16_bits >> 8) & 0xff) as u8);
        }
        out
    }

    /// Round F32 to its BF16-representable value (truncate low 16 bits
    /// of the F32 bit pattern, then promote back to F32). Used in the
    /// CPU reference so the parity comparison accounts for the BF16
    /// quantization of the on-disk weight.
    fn bf16_quantize_f32(v: f32) -> f32 {
        let bits = v.to_bits() & 0xFFFF0000;
        f32::from_bits(bits)
    }

    /// CPU reference matmul: output[s, h] = sum_k input[s, k] * weight[h, k].
    /// Uses f64 accumulator for precision.
    fn cpu_fc_reference(
        input: &[f32],         // [seq, in_features]
        weight_bf16_q: &[f32], // [out_features, in_features], pre-BF16-quantized
        seq_len: usize,
        in_features: usize,
        out_features: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; seq_len * out_features];
        for s in 0..seq_len {
            for h in 0..out_features {
                let mut acc = 0.0f64;
                for k in 0..in_features {
                    acc += (input[s * in_features + k] as f64)
                        * (weight_bf16_q[h * in_features + k] as f64);
                }
                out[s * out_features + h] = acc as f32;
            }
        }
        out
    }

    /// Deterministic pseudo-random F32 in [-1, 1).
    fn pseudo_random(seed: u64) -> f32 {
        let x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = ((x >> 33) as u32) & 0x7FFFFF;
        (bits as f32 / 0x7FFFFF as f32) * 2.0 - 1.0
    }

    fn fill_random(buf: &mut [f32], seed: u64) {
        for (i, v) in buf.iter_mut().enumerate() {
            *v = pseudo_random(seed.wrapping_add(i as u64));
        }
    }

    /// Build a safetensors blob with custom `fc.weight` bytes. All
    /// other manifest tensors get zero bytes (sufficient for upload —
    /// we only consume `fc` in this test).
    fn build_blob_with_fc_weight(
        manifest: &[ExpectedTensor],
        fc_bytes: &[u8],
    ) -> Vec<u8> {
        let mut storage: Vec<Vec<u8>> = Vec::with_capacity(manifest.len());
        for exp in manifest {
            let elem_bytes = match exp.dtype {
                SafeDtype::BF16 => 2,
                SafeDtype::I64 => 8,
                _ => panic!("unexpected dtype in test"),
            };
            let nelem: usize = exp.shape.iter().product();
            if exp.name == "fc.weight" {
                assert_eq!(fc_bytes.len(), nelem * elem_bytes);
                storage.push(fc_bytes.to_vec());
            } else {
                storage.push(vec![0u8; nelem * elem_bytes]);
            }
        }
        let mut tensors: BTreeMap<String, TensorView> = BTreeMap::new();
        for (i, exp) in manifest.iter().enumerate() {
            let view =
                TensorView::new(exp.dtype, exp.shape.clone(), storage[i].as_slice())
                    .expect("synthetic view");
            tensors.insert(exp.name.clone(), view);
        }
        safetensors::serialize(
            &tensors,
            None::<std::collections::HashMap<String, String>>,
        )
        .expect("serialize")
    }

    fn upload_f32_to_gpu(
        device: &MlxDevice,
        data: &[f32],
        shape: Vec<usize>,
    ) -> MlxBuffer {
        let bytes = data.len() * 4;
        let mut buf = device
            .alloc_buffer(bytes, DType::F32, shape)
            .expect("alloc input");
        buf.as_mut_slice::<f32>()
            .expect("input slice")
            .copy_from_slice(data);
        buf
    }

    #[test]
    fn adr_037_e4b2_fc_cpu_parity_seq_4_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);

        let seq_len: u32 = 4;
        let fc_in = cfg.fc_input_size();
        let hidden = cfg.hidden_size;

        // Synthesize input (F32) and weight (F32 reference + BF16 truncation
        // for both the GPU buffer and the CPU reference matmul).
        let mut input_data = vec![0.0f32; (seq_len as usize) * fc_in];
        fill_random(&mut input_data, 0xA00);

        let mut weight_f32 = vec![0.0f32; hidden * fc_in];
        fill_random(&mut weight_f32, 0xB00);
        let weight_bf16_bytes = f32_to_bf16_bytes(&weight_f32);
        // BF16-quantized F32 (what the GPU kernel effectively uses).
        let weight_bf16_q: Vec<f32> = weight_f32.iter().map(|&v| bf16_quantize_f32(v)).collect();

        // CPU reference using BF16-quantized weights → matches what GPU computes.
        let cpu_out =
            cpu_fc_reference(&input_data, &weight_bf16_q, seq_len as usize, fc_in, hidden);

        // GPU path: build synthetic safetensors blob, load + upload, dispatch.
        let blob = build_blob_with_fc_weight(&manifest, &weight_bf16_bytes);
        let weights = Eagle3Weights::load(&blob, &cfg).expect("weights load");
        let tensors = Eagle3DrafterTensors::upload(&device, &cfg, &weights)
            .expect("upload tensors");
        let input_gpu = upload_f32_to_gpu(
            &device,
            &input_data,
            vec![seq_len as usize, fc_in],
        );

        let mut enc = device.command_encoder().expect("encoder");
        let out_buf =
            dispatch_eagle3_fc(&mut enc, &mut registry, &device, &input_gpu, &tensors, &cfg, seq_len)
                .expect("dispatch_eagle3_fc");
        enc.commit_and_wait().expect("commit");

        let gpu_out: &[f32] = out_buf.as_slice::<f32>().expect("output slice");
        assert_eq!(
            gpu_out.len(),
            (seq_len as usize) * hidden,
            "output shape"
        );

        // Compare GPU vs CPU within tolerance. BF16 quantized weights ×
        // F32 input × seq=4 accumulation gives relative error ~1e-3
        // on random inputs; absolute error scales with input/weight
        // magnitudes (both in [-1,1)) and the inner-dim accumulation
        // size (768). Tolerance 5e-2 is conservative.
        let mut max_diff = 0.0f32;
        for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            let d = (g - c).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(
                d < 5e-2,
                "FC parity violated at idx {i}: gpu={g} cpu={c} diff={d}"
            );
        }
        eprintln!("fc parity seq=4 max_diff={max_diff:.6e}");
    }

    #[test]
    fn adr_037_e4b2_fc_cpu_parity_seq_1_decode_2026_05_22() {
        // seq=1 exercises the GEMV path (dense_gemv_bf16_f32) instead
        // of the tiled GEMM path.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);

        let _seq_len: u32 = 1; // documents intent; cpu_fc_reference uses literal 1 below
        let fc_in = cfg.fc_input_size();
        let hidden = cfg.hidden_size;

        let mut input_data = vec![0.0f32; fc_in];
        fill_random(&mut input_data, 0xC00);
        let mut weight_f32 = vec![0.0f32; hidden * fc_in];
        fill_random(&mut weight_f32, 0xD00);
        let weight_bf16_bytes = f32_to_bf16_bytes(&weight_f32);
        let weight_bf16_q: Vec<f32> = weight_f32.iter().map(|&v| bf16_quantize_f32(v)).collect();

        let cpu_out = cpu_fc_reference(&input_data, &weight_bf16_q, 1, fc_in, hidden);

        let blob = build_blob_with_fc_weight(&manifest, &weight_bf16_bytes);
        let weights = Eagle3Weights::load(&blob, &cfg).expect("weights load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");
        let input_gpu = upload_f32_to_gpu(&device, &input_data, vec![1, fc_in]);

        let mut enc = device.command_encoder().expect("encoder");
        let out_buf =
            dispatch_eagle3_fc(&mut enc, &mut registry, &device, &input_gpu, &tensors, &cfg, 1)
                .expect("dispatch_eagle3_fc (seq=1)");
        enc.commit_and_wait().expect("commit");

        let gpu_out: &[f32] = out_buf.as_slice::<f32>().expect("output slice");
        assert_eq!(gpu_out.len(), hidden);
        let mut max_diff = 0.0f32;
        for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
            let d = (g - c).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(d < 5e-2, "seq=1 GEMV parity: diff={d} > 5e-2");
        }
        eprintln!("fc parity seq=1 (GEMV) max_diff={max_diff:.6e}");
    }

    #[test]
    fn adr_037_e4b2_fc_rejects_wrong_input_shape_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);
        let blob = build_blob_with_fc_weight(
            &manifest,
            &vec![0u8; cfg.hidden_size * cfg.fc_input_size() * 2],
        );
        let weights = Eagle3Weights::load(&blob, &cfg).expect("weights load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");

        // Allocate input with wrong size — should be seq_len * fc_in
        // = 4 * 768 = 3072 floats. Allocate 100 instead.
        let bad_data = vec![0.0f32; 100];
        let bad_input = upload_f32_to_gpu(&device, &bad_data, vec![100]);

        let mut enc = device.command_encoder().expect("encoder");
        let err =
            dispatch_eagle3_fc(&mut enc, &mut registry, &device, &bad_input, &tensors, &cfg, 4)
                .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("concat_hidden has"),
            "expected shape error, got: {msg}"
        );
    }

    #[test]
    fn adr_037_e4b2_gate_fc_rejects_non_f32_input_dtype_2026_05_22() {
        // Codex /cfa E4b.2 Critical fix (2026-05-22): wrapper must
        // reject non-F32 input even though apply_linear_projection_f32
        // only debug-asserts this. Validates the hard check.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);
        let blob = build_blob_with_fc_weight(
            &manifest,
            &vec![0u8; cfg.hidden_size * cfg.fc_input_size() * 2],
        );
        let weights = Eagle3Weights::load(&blob, &cfg).expect("weights load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");

        // Allocate a BF16 input with the correct element count. Wrapper
        // should reject due to wrong dtype, not pass through.
        let seq_len = 2_u32;
        let elem_count = (seq_len as usize) * cfg.fc_input_size();
        let bad_input = device
            .alloc_buffer(
                elem_count * 2, // BF16 size
                DType::BF16,
                vec![seq_len as usize, cfg.fc_input_size()],
            )
            .expect("alloc bad input");

        let mut enc = device.command_encoder().expect("encoder");
        let err = dispatch_eagle3_fc(
            &mut enc, &mut registry, &device, &bad_input, &tensors, &cfg, seq_len,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("dtype must be F32"),
            "expected F32-dtype error, got: {msg}"
        );
    }

    // ----------------------------------------------------------------
    // Phase E4b.3 tests — input_layernorm + hidden_norm + concat
    // ----------------------------------------------------------------

    /// CPU reference RMSNorm: out[s, d] = x[s, d] / sqrt(mean(x^2) + eps) * w[d].
    fn cpu_rms_norm_f32(
        input: &[f32],   // [seq, dim]
        weight: &[f32],  // [dim]
        seq_len: usize,
        dim: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; seq_len * dim];
        for s in 0..seq_len {
            let row = &input[s * dim..(s + 1) * dim];
            // Use f64 accumulator (matches mlx-native's rms_norm_f32_v2 scheme).
            let mean_sq: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()
                / (dim as f64);
            let inv_rms = 1.0 / ((mean_sq as f32 + eps).sqrt());
            for d in 0..dim {
                out[s * dim + d] = row[d] * inv_rms * weight[d];
            }
        }
        out
    }

    /// Build a synthetic safetensors blob with custom bytes for one
    /// or more tensors keyed by name. Other manifest tensors get
    /// zero bytes.
    fn build_blob_with_overrides(
        manifest: &[ExpectedTensor],
        overrides: &std::collections::HashMap<String, Vec<u8>>,
    ) -> Vec<u8> {
        let mut storage: Vec<Vec<u8>> = Vec::with_capacity(manifest.len());
        for exp in manifest {
            let elem_bytes = match exp.dtype {
                SafeDtype::BF16 => 2,
                SafeDtype::I64 => 8,
                _ => panic!("unexpected dtype in test"),
            };
            let nelem: usize = exp.shape.iter().product();
            if let Some(bytes) = overrides.get(&exp.name) {
                assert_eq!(bytes.len(), nelem * elem_bytes, "override bytes for {}", exp.name);
                storage.push(bytes.clone());
            } else {
                storage.push(vec![0u8; nelem * elem_bytes]);
            }
        }
        let mut tensors: BTreeMap<String, TensorView> = BTreeMap::new();
        for (i, exp) in manifest.iter().enumerate() {
            let view =
                TensorView::new(exp.dtype, exp.shape.clone(), storage[i].as_slice())
                    .expect("synthetic view");
            tensors.insert(exp.name.clone(), view);
        }
        safetensors::serialize(
            &tensors,
            None::<std::collections::HashMap<String, String>>,
        )
        .expect("serialize")
    }

    #[test]
    fn adr_037_e4b3_input_layernorm_cpu_parity_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);

        let seq_len: u32 = 4;
        let hidden = cfg.hidden_size;

        // Random F32 input + BF16-quantized weight (RMSNorm weights
        // are uploaded BF16 → F32 cast; CPU ref uses the BF16-quantized
        // F32 values so parity matches what kernel computes).
        let mut input_data = vec![0.0f32; (seq_len as usize) * hidden];
        fill_random(&mut input_data, 0xE10);
        let mut weight_f32 = vec![0.0f32; hidden];
        fill_random(&mut weight_f32, 0xE11);
        let weight_bf16_bytes = f32_to_bf16_bytes(&weight_f32);
        let weight_bf16_q: Vec<f32> = weight_f32.iter().map(|&v| bf16_quantize_f32(v)).collect();

        let cpu_out = cpu_rms_norm_f32(
            &input_data, &weight_bf16_q,
            seq_len as usize, hidden, cfg.rms_norm_eps,
        );

        let mut overrides = std::collections::HashMap::new();
        overrides.insert(
            "layers.0.input_layernorm.weight".to_string(),
            weight_bf16_bytes,
        );
        let blob = build_blob_with_overrides(&manifest, &overrides);
        let weights = Eagle3Weights::load(&blob, &cfg).expect("weights load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");
        let input_gpu = upload_f32_to_gpu(
            &device, &input_data, vec![seq_len as usize, hidden],
        );

        let mut enc = device.command_encoder().expect("encoder");
        let out_buf = dispatch_eagle3_input_layernorm(
            &mut enc, &mut registry, &device, &input_gpu, &tensors, &cfg, seq_len,
        )
        .expect("dispatch_eagle3_input_layernorm");
        enc.commit_and_wait().expect("commit");

        let gpu_out: &[f32] = out_buf.as_slice::<f32>().expect("output slice");
        let mut max_diff = 0.0f32;
        for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
            let d = (g - c).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(d < 1e-3, "input_layernorm parity: diff={d} > 1e-3");
        }
        eprintln!("input_layernorm parity max_diff={max_diff:.6e}");
    }

    #[test]
    fn adr_037_e4b3_hidden_norm_cpu_parity_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);

        let seq_len: u32 = 4;
        let hidden = cfg.hidden_size;

        let mut input_data = vec![0.0f32; (seq_len as usize) * hidden];
        fill_random(&mut input_data, 0xE20);
        let mut weight_f32 = vec![0.0f32; hidden];
        fill_random(&mut weight_f32, 0xE21);
        let weight_bf16_bytes = f32_to_bf16_bytes(&weight_f32);
        let weight_bf16_q: Vec<f32> = weight_f32.iter().map(|&v| bf16_quantize_f32(v)).collect();

        let cpu_out = cpu_rms_norm_f32(
            &input_data, &weight_bf16_q,
            seq_len as usize, hidden, cfg.rms_norm_eps,
        );

        let mut overrides = std::collections::HashMap::new();
        overrides.insert(
            "layers.0.hidden_norm.weight".to_string(),
            weight_bf16_bytes,
        );
        let blob = build_blob_with_overrides(&manifest, &overrides);
        let weights = Eagle3Weights::load(&blob, &cfg).expect("weights load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");
        let input_gpu = upload_f32_to_gpu(
            &device, &input_data, vec![seq_len as usize, hidden],
        );

        let mut enc = device.command_encoder().expect("encoder");
        let out_buf = dispatch_eagle3_hidden_norm(
            &mut enc, &mut registry, &device, &input_gpu, &tensors, &cfg, seq_len,
        )
        .expect("dispatch_eagle3_hidden_norm");
        enc.commit_and_wait().expect("commit");

        let gpu_out: &[f32] = out_buf.as_slice::<f32>().expect("output slice");
        let mut max_diff = 0.0f32;
        for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
            let d = (g - c).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(d < 1e-3, "hidden_norm parity: diff={d} > 1e-3");
        }
        eprintln!("hidden_norm parity max_diff={max_diff:.6e}");
    }

    #[test]
    fn adr_037_e4b3_concat_2x_hidden_layout_matches_vllm_2026_05_22() {
        // Verify the concat produces [seq, 2*H] with embeds-left,
        // hidden-right column ordering per vLLM torch.cat([embeds,
        // hidden_states], dim=-1).
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);
        let blob = build_blob_with_overrides(&manifest, &std::collections::HashMap::new());
        let weights = Eagle3Weights::load(&blob, &cfg).expect("weights load");
        let _tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");

        let seq_len: u32 = 3;
        let hidden = cfg.hidden_size;
        // Sentinel inputs: embeds = positive ramp (1.0..), hidden_states
        // = negative ramp (-1.0..). The concat result must place each
        // row's positive values FIRST (cols 0..hidden) then negative
        // (cols hidden..2*hidden).
        let mut embeds_data = vec![0.0f32; (seq_len as usize) * hidden];
        let mut hidden_data = vec![0.0f32; (seq_len as usize) * hidden];
        for s in 0..(seq_len as usize) {
            for d in 0..hidden {
                embeds_data[s * hidden + d] = (s * 1000 + d + 1) as f32;
                hidden_data[s * hidden + d] = -((s * 1000 + d + 1) as f32);
            }
        }
        let embeds_gpu = upload_f32_to_gpu(
            &device, &embeds_data, vec![seq_len as usize, hidden],
        );
        let hidden_gpu = upload_f32_to_gpu(
            &device, &hidden_data, vec![seq_len as usize, hidden],
        );

        let mut enc = device.command_encoder().expect("encoder");
        let out_buf = dispatch_eagle3_concat_2x_hidden(
            &mut enc, &mut registry, &device,
            &embeds_gpu, &hidden_gpu, &cfg, seq_len,
        )
        .expect("concat dispatch");
        enc.commit_and_wait().expect("commit");

        let gpu_out: &[f32] = out_buf.as_slice::<f32>().expect("output slice");
        assert_eq!(gpu_out.len(), (seq_len as usize) * 2 * hidden);

        // Row-major [seq, 2*hidden]: row s, col c = gpu_out[s*2H + c].
        for s in 0..(seq_len as usize) {
            for d in 0..hidden {
                let left = gpu_out[s * 2 * hidden + d];
                let right = gpu_out[s * 2 * hidden + hidden + d];
                let expected_pos = (s * 1000 + d + 1) as f32;
                assert_eq!(left, expected_pos, "embeds (s={s} d={d})");
                assert_eq!(right, -expected_pos, "hidden_states (s={s} d={d})");
            }
        }
    }

    #[test]
    fn adr_037_e4b3_input_layernorm_rejects_non_f32_input_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);
        let blob = build_blob_with_overrides(&manifest, &std::collections::HashMap::new());
        let weights = Eagle3Weights::load(&blob, &cfg).expect("weights load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");

        let seq_len = 2_u32;
        let bf16_input = device
            .alloc_buffer(
                (seq_len as usize) * cfg.hidden_size * 2,
                DType::BF16,
                vec![seq_len as usize, cfg.hidden_size],
            )
            .expect("alloc bad input");

        let mut enc = device.command_encoder().expect("encoder");
        let err = dispatch_eagle3_input_layernorm(
            &mut enc, &mut registry, &device, &bf16_input, &tensors, &cfg, seq_len,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("dtype must be F32"),
            "expected F32-dtype error, got: {err}"
        );
    }

    #[test]
    fn adr_037_e4b3_gate_input_layernorm_rejects_zero_seq_len_2026_05_22() {
        // Codex /cfa E4b.3 Major fix (2026-05-22): zero seq_len was
        // structurally meaningless but previously slipped through.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);
        let blob = build_blob_with_overrides(&manifest, &std::collections::HashMap::new());
        let weights = Eagle3Weights::load(&blob, &cfg).expect("weights load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");
        let empty_input = device
            .alloc_buffer(4, DType::F32, vec![1])
            .expect("alloc empty");
        let mut enc = device.command_encoder().expect("encoder");
        let err = dispatch_eagle3_input_layernorm(
            &mut enc, &mut registry, &device, &empty_input, &tensors, &cfg, 0,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("seq_len must be > 0"),
            "expected seq_len-zero error, got: {err}"
        );
    }

    #[test]
    fn adr_037_e4b3_gate_concat_rejects_zero_seq_len_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);
        let blob = build_blob_with_overrides(&manifest, &std::collections::HashMap::new());
        let weights = Eagle3Weights::load(&blob, &cfg).expect("weights load");
        let _tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");
        let empty_a = device
            .alloc_buffer(4, DType::F32, vec![1])
            .expect("alloc a");
        let empty_b = device
            .alloc_buffer(4, DType::F32, vec![1])
            .expect("alloc b");
        let mut enc = device.command_encoder().expect("encoder");
        let err = dispatch_eagle3_concat_2x_hidden(
            &mut enc, &mut registry, &device, &empty_a, &empty_b, &cfg, 0,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("seq_len must be > 0"),
            "expected seq_len-zero error, got: {err}"
        );
    }

    // ----------------------------------------------------------------
    // Phase E4b.4 tests — Q/K/V projections
    // ----------------------------------------------------------------

    /// Run Q projection through GPU + CPU reference, return both outputs.
    fn run_q_proj_with_overrides(
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        cfg: &Eagle3DrafterConfig,
        seq_len: u32,
        q_weight_f32: &[f32],   // [q_out, qkv_in]
        q_bias_f32: Option<&[f32]>, // [q_out] when attention_bias=true
        input_data: &[f32],     // [seq, qkv_in]
    ) -> (Vec<f32>, Vec<f32>) {
        let manifest = expected_manifest(cfg);
        let mut overrides = std::collections::HashMap::new();
        overrides.insert(
            "layers.0.self_attn.q_proj.weight".to_string(),
            f32_to_bf16_bytes(q_weight_f32),
        );
        if let Some(b) = q_bias_f32 {
            overrides.insert(
                "layers.0.self_attn.q_proj.bias".to_string(),
                f32_to_bf16_bytes(b),
            );
        }
        let blob = build_blob_with_overrides(&manifest, &overrides);
        let weights = Eagle3Weights::load(&blob, cfg).expect("weights load");
        let tensors =
            Eagle3DrafterTensors::upload(device, cfg, &weights).expect("upload");
        let input_gpu = upload_f32_to_gpu(
            device,
            input_data,
            vec![seq_len as usize, cfg.qkv_input_width()],
        );

        let mut enc = device.command_encoder().expect("encoder");
        let out_buf = dispatch_eagle3_q_proj(
            &mut enc, registry, device, &input_gpu, &tensors, cfg, seq_len,
        )
        .expect("dispatch_eagle3_q_proj");
        enc.commit_and_wait().expect("commit");

        let gpu_out: Vec<f32> = out_buf
            .as_slice::<f32>()
            .expect("output slice")
            .to_vec();

        // CPU reference uses BF16-quantized weights + bias.
        let weight_bf16_q: Vec<f32> =
            q_weight_f32.iter().map(|&v| bf16_quantize_f32(v)).collect();
        let mut cpu_out = cpu_fc_reference(
            input_data,
            &weight_bf16_q,
            seq_len as usize,
            cfg.qkv_input_width(),
            cfg.q_proj_out(),
        );
        if let Some(b) = q_bias_f32 {
            let b_q: Vec<f32> = b.iter().map(|&v| bf16_quantize_f32(v)).collect();
            for s in 0..(seq_len as usize) {
                for d in 0..cfg.q_proj_out() {
                    cpu_out[s * cfg.q_proj_out() + d] += b_q[d];
                }
            }
        }
        (gpu_out, cpu_out)
    }

    #[test]
    fn adr_037_e4b4_q_proj_cpu_parity_no_bias_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg(); // attention_bias = false
        let seq_len: u32 = 4;
        let qkv_in = cfg.qkv_input_width();
        let q_out = cfg.q_proj_out();

        let mut input_data = vec![0.0f32; (seq_len as usize) * qkv_in];
        fill_random(&mut input_data, 0xF40);
        let mut q_weight = vec![0.0f32; q_out * qkv_in];
        fill_random(&mut q_weight, 0xF41);

        let (gpu_out, cpu_out) = run_q_proj_with_overrides(
            &device,
            &mut registry,
            &cfg,
            seq_len,
            &q_weight,
            None,
            &input_data,
        );
        let mut max_diff = 0.0f32;
        for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
            let d = (g - c).abs();
            if d > max_diff {
                max_diff = d;
            }
            // 2*hidden inner-dim (=512) gives more accumulation than
            // E4b.2's 3*hidden (=768) but the input/weight ranges
            // are similar; 5e-2 stays adequate.
            assert!(d < 5e-2, "q_proj parity: diff={d} > 5e-2");
        }
        eprintln!("q_proj parity max_diff={max_diff:.6e}");
    }

    fn cfg_with_bias() -> Eagle3DrafterConfig {
        let mut c = tiny_cfg();
        c.attention_bias = true;
        c
    }

    #[test]
    fn adr_037_e4b4_q_proj_cpu_parity_with_bias_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = cfg_with_bias();
        let seq_len: u32 = 4;
        let qkv_in = cfg.qkv_input_width();
        let q_out = cfg.q_proj_out();

        let mut input_data = vec![0.0f32; (seq_len as usize) * qkv_in];
        fill_random(&mut input_data, 0xF50);
        let mut q_weight = vec![0.0f32; q_out * qkv_in];
        fill_random(&mut q_weight, 0xF51);
        let mut q_bias = vec![0.0f32; q_out];
        fill_random(&mut q_bias, 0xF52);

        let (gpu_out, cpu_out) = run_q_proj_with_overrides(
            &device,
            &mut registry,
            &cfg,
            seq_len,
            &q_weight,
            Some(&q_bias),
            &input_data,
        );
        let mut max_diff = 0.0f32;
        for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
            let d = (g - c).abs();
            if d > max_diff {
                max_diff = d;
            }
            // Same tolerance as no-bias path; the memory_barrier
            // between matmul + bias-add ensures bias reads the
            // committed matmul output. Without the barrier, we
            // observed diffs of 0.156-0.794 from racy bias reads
            // of partially-written matmul output (debugged during
            // E4b.4 implementation 2026-05-22).
            assert!(d < 5e-2, "q_proj+bias parity: diff={d} > 5e-2");
        }
        eprintln!("q_proj+bias parity max_diff={max_diff:.6e}");
    }

    #[test]
    fn adr_037_e4b4_k_proj_output_shape_matches_kv_proj_out_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);
        let blob = build_blob_with_overrides(&manifest, &std::collections::HashMap::new());
        let weights = Eagle3Weights::load(&blob, &cfg).expect("load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");

        let seq_len = 3_u32;
        let input_data = vec![0.0f32; (seq_len as usize) * cfg.qkv_input_width()];
        let input_gpu = upload_f32_to_gpu(
            &device,
            &input_data,
            vec![seq_len as usize, cfg.qkv_input_width()],
        );

        let mut enc = device.command_encoder().expect("encoder");
        let k_out = dispatch_eagle3_k_proj(
            &mut enc, &mut registry, &device, &input_gpu, &tensors, &cfg, seq_len,
        )
        .expect("k_proj");
        enc.commit_and_wait().expect("commit");
        // K output shape: [seq, num_kv_heads * head_dim].
        assert_eq!(k_out.dtype(), DType::F32);
        assert_eq!(
            k_out.element_count(),
            (seq_len as usize) * cfg.kv_proj_out()
        );
    }

    #[test]
    fn adr_037_e4b4_v_proj_output_shape_matches_kv_proj_out_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);
        let blob = build_blob_with_overrides(&manifest, &std::collections::HashMap::new());
        let weights = Eagle3Weights::load(&blob, &cfg).expect("load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");

        let seq_len = 3_u32;
        let input_data = vec![0.0f32; (seq_len as usize) * cfg.qkv_input_width()];
        let input_gpu = upload_f32_to_gpu(
            &device,
            &input_data,
            vec![seq_len as usize, cfg.qkv_input_width()],
        );

        let mut enc = device.command_encoder().expect("encoder");
        let v_out = dispatch_eagle3_v_proj(
            &mut enc, &mut registry, &device, &input_gpu, &tensors, &cfg, seq_len,
        )
        .expect("v_proj");
        enc.commit_and_wait().expect("commit");
        assert_eq!(v_out.dtype(), DType::F32);
        assert_eq!(
            v_out.element_count(),
            (seq_len as usize) * cfg.kv_proj_out()
        );
    }

    #[test]
    fn adr_037_e4b4_gate_q_proj_rejects_non_f32_input_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);
        let blob = build_blob_with_overrides(&manifest, &std::collections::HashMap::new());
        let weights = Eagle3Weights::load(&blob, &cfg).expect("load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");

        let seq_len = 2_u32;
        let bad_input = device
            .alloc_buffer(
                (seq_len as usize) * cfg.qkv_input_width() * 2,
                DType::BF16,
                vec![seq_len as usize, cfg.qkv_input_width()],
            )
            .expect("alloc bad");

        let mut enc = device.command_encoder().expect("encoder");
        let err = dispatch_eagle3_q_proj(
            &mut enc, &mut registry, &device, &bad_input, &tensors, &cfg, seq_len,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("dtype must be F32"),
            "got: {err}"
        );
    }

    #[test]
    fn adr_037_e4b4_gate_q_proj_rejects_zero_seq_len_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);
        let blob = build_blob_with_overrides(&manifest, &std::collections::HashMap::new());
        let weights = Eagle3Weights::load(&blob, &cfg).expect("load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");

        let empty = device
            .alloc_buffer(4, DType::F32, vec![1])
            .expect("alloc empty");
        let mut enc = device.command_encoder().expect("encoder");
        let err = dispatch_eagle3_q_proj(
            &mut enc, &mut registry, &device, &empty, &tensors, &cfg, 0,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("seq_len must be > 0"),
            "got: {err}"
        );
    }

    // ----------------------------------------------------------------
    // Phase E4b.5a tests — Q/K per-head norm
    // ----------------------------------------------------------------

    fn cfg_qk_norm_tiny() -> Eagle3DrafterConfig {
        let mut c = tiny_cfg();
        c.use_qk_norm = true;
        c
    }

    #[test]
    fn adr_037_e4b5a_q_head_norm_cpu_parity_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = cfg_qk_norm_tiny();
        let seq_len: u32 = 4;
        let n_heads = cfg.num_q_heads;
        let head_dim = cfg.head_dim;
        let total_elems = (seq_len as usize) * n_heads * head_dim;

        // Random F32 input (treated as flat [seq*n_heads, head_dim]).
        let mut proj_data = vec![0.0f32; total_elems];
        fill_random(&mut proj_data, 0xA51);
        // Random F32 q_norm weight, BF16-truncated for upload.
        let mut weight_f32 = vec![0.0f32; head_dim];
        fill_random(&mut weight_f32, 0xA52);

        // CPU reference per row (seq * head): RMSNorm over head_dim.
        let weight_bf16_q: Vec<f32> =
            weight_f32.iter().map(|&v| bf16_quantize_f32(v)).collect();
        let cpu_out = cpu_rms_norm_f32(
            &proj_data,
            &weight_bf16_q,
            (seq_len as usize) * n_heads, // rows
            head_dim,
            cfg.rms_norm_eps,
        );

        // GPU: build synthetic blob with custom q_norm weight,
        // upload, dispatch.
        let manifest = expected_manifest(&cfg);
        let mut overrides = std::collections::HashMap::new();
        overrides.insert(
            "layers.0.self_attn.q_norm.weight".to_string(),
            f32_to_bf16_bytes(&weight_f32),
        );
        let blob = build_blob_with_overrides(&manifest, &overrides);
        let weights = Eagle3Weights::load(&blob, &cfg).expect("load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");
        let proj_gpu = upload_f32_to_gpu(
            &device,
            &proj_data,
            vec![seq_len as usize, n_heads * head_dim],
        );

        let mut enc = device.command_encoder().expect("encoder");
        let out_buf = dispatch_eagle3_q_head_norm(
            &mut enc, &mut registry, &device, &proj_gpu, &tensors, &cfg, seq_len,
        )
        .expect("q_head_norm");
        enc.commit_and_wait().expect("commit");

        let gpu_out: &[f32] = out_buf.as_slice::<f32>().expect("output slice");
        let mut max_diff = 0.0f32;
        for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
            let d = (g - c).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(d < 1e-3, "q_head_norm parity: diff={d} > 1e-3");
        }
        eprintln!("q_head_norm parity max_diff={max_diff:.6e}");
    }

    #[test]
    fn adr_037_e4b5a_k_head_norm_cpu_parity_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = cfg_qk_norm_tiny();
        let seq_len: u32 = 4;
        let n_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let total_elems = (seq_len as usize) * n_heads * head_dim;

        let mut proj_data = vec![0.0f32; total_elems];
        fill_random(&mut proj_data, 0xA61);
        let mut weight_f32 = vec![0.0f32; head_dim];
        fill_random(&mut weight_f32, 0xA62);

        let weight_bf16_q: Vec<f32> =
            weight_f32.iter().map(|&v| bf16_quantize_f32(v)).collect();
        let cpu_out = cpu_rms_norm_f32(
            &proj_data,
            &weight_bf16_q,
            (seq_len as usize) * n_heads,
            head_dim,
            cfg.rms_norm_eps,
        );

        let manifest = expected_manifest(&cfg);
        let mut overrides = std::collections::HashMap::new();
        overrides.insert(
            "layers.0.self_attn.k_norm.weight".to_string(),
            f32_to_bf16_bytes(&weight_f32),
        );
        let blob = build_blob_with_overrides(&manifest, &overrides);
        let weights = Eagle3Weights::load(&blob, &cfg).expect("load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");
        let proj_gpu = upload_f32_to_gpu(
            &device,
            &proj_data,
            vec![seq_len as usize, n_heads * head_dim],
        );

        let mut enc = device.command_encoder().expect("encoder");
        let out_buf = dispatch_eagle3_k_head_norm(
            &mut enc, &mut registry, &device, &proj_gpu, &tensors, &cfg, seq_len,
        )
        .expect("k_head_norm");
        enc.commit_and_wait().expect("commit");

        let gpu_out: &[f32] = out_buf.as_slice::<f32>().expect("output slice");
        let mut max_diff = 0.0f32;
        for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
            let d = (g - c).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(d < 1e-3, "k_head_norm parity: diff={d} > 1e-3");
        }
        eprintln!("k_head_norm parity max_diff={max_diff:.6e}");
    }

    #[test]
    fn adr_037_e4b5a_q_head_norm_errors_when_use_qk_norm_false_2026_05_22() {
        // Calling q_head_norm when cfg.use_qk_norm=false should
        // fail-fast: q_norm tensor is None.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg(); // use_qk_norm = false
        let manifest = expected_manifest(&cfg);
        let blob = build_blob_with_overrides(&manifest, &std::collections::HashMap::new());
        let weights = Eagle3Weights::load(&blob, &cfg).expect("load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");
        let seq_len = 2_u32;
        let n = (seq_len as usize) * cfg.num_q_heads * cfg.head_dim;
        let proj_data = vec![0.0f32; n];
        let proj_gpu = upload_f32_to_gpu(
            &device, &proj_data,
            vec![seq_len as usize, cfg.num_q_heads * cfg.head_dim],
        );

        let mut enc = device.command_encoder().expect("encoder");
        let err = dispatch_eagle3_q_head_norm(
            &mut enc, &mut registry, &device, &proj_gpu, &tensors, &cfg, seq_len,
        )
        .unwrap_err();
        // Now codex /cfa Major fix gates cfg.use_qk_norm BEFORE
        // tensor-absent check; either gate-error OR absent-error
        // is acceptable here (gate fires first).
        let msg = err.to_string();
        assert!(
            msg.contains("use_qk_norm is false") || msg.contains("q_norm absent"),
            "expected gate or absent-tensor error, got: {err}"
        );
    }

    #[test]
    fn adr_037_e4b5a_gate_q_head_norm_rejects_when_cfg_off_with_tensor_present_2026_05_22() {
        // Codex /cfa E4b.5a Major fix (2026-05-22): cfg gate must be
        // enforced even when the tensor happens to be present (an
        // inconsistent config/tensor bundle is a bug, not silent fallback).
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        // Load tensors with use_qk_norm=true (so q_norm is present)
        let load_cfg = cfg_qk_norm_tiny();
        let manifest = expected_manifest(&load_cfg);
        let blob = build_blob_with_overrides(&manifest, &std::collections::HashMap::new());
        let weights = Eagle3Weights::load(&blob, &load_cfg).expect("load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &load_cfg, &weights).expect("upload");
        // But call with use_qk_norm=false (forces gate rejection)
        let mut dispatch_cfg = load_cfg.clone();
        dispatch_cfg.use_qk_norm = false;

        let seq_len = 2_u32;
        let n = (seq_len as usize) * dispatch_cfg.num_q_heads * dispatch_cfg.head_dim;
        let proj_data = vec![0.0f32; n];
        let proj_gpu = upload_f32_to_gpu(
            &device, &proj_data,
            vec![seq_len as usize, dispatch_cfg.num_q_heads * dispatch_cfg.head_dim],
        );

        let mut enc = device.command_encoder().expect("encoder");
        let err = dispatch_eagle3_q_head_norm(
            &mut enc, &mut registry, &device, &proj_gpu, &tensors, &dispatch_cfg, seq_len,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("cfg.use_qk_norm is false"),
            "expected gate error, got: {err}"
        );
    }

    #[test]
    fn adr_037_e4b5a_gate_q_head_norm_rejects_non_f32_input_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = cfg_qk_norm_tiny();
        let manifest = expected_manifest(&cfg);
        let blob = build_blob_with_overrides(&manifest, &std::collections::HashMap::new());
        let weights = Eagle3Weights::load(&blob, &cfg).expect("load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");

        let seq_len = 2_u32;
        let n = (seq_len as usize) * cfg.num_q_heads * cfg.head_dim;
        let bad = device
            .alloc_buffer(
                n * 2,
                DType::BF16,
                vec![seq_len as usize, cfg.num_q_heads * cfg.head_dim],
            )
            .expect("alloc bad");
        let mut enc = device.command_encoder().expect("encoder");
        let err = dispatch_eagle3_q_head_norm(
            &mut enc, &mut registry, &device, &bad, &tensors, &cfg, seq_len,
        )
        .unwrap_err();
        assert!(err.to_string().contains("dtype must be F32"), "got: {err}");
    }

    // ----------------------------------------------------------------
    // Phase E4b.5b tests — RoPE on Q/K (NeoX-style, tree-position-aware)
    // ----------------------------------------------------------------

    /// CPU reference NeoX RoPE: rotates pairs (d, d + rope_dim/2)
    /// for d in 0..rope_dim/2. Input layout
    /// `[seq * num_heads, head_dim]`. Position used: `positions[s]`
    /// (broadcast across heads). Untouched dims: `[rope_dim, head_dim)`.
    fn cpu_neox_rope(
        input: &[f32],         // [seq * num_heads, head_dim]
        positions: &[u32],     // [seq]
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        rope_dim: usize,
        freq_base: f32,
    ) -> Vec<f32> {
        let half = rope_dim / 2;
        let mut out = vec![0.0f32; seq_len * num_heads * head_dim];
        for s in 0..seq_len {
            let pos = positions[s] as f32;
            for h in 0..num_heads {
                let row_base = (s * num_heads + h) * head_dim;
                // Copy any dims beyond rope_dim untouched.
                for d in rope_dim..head_dim {
                    out[row_base + d] = input[row_base + d];
                }
                // Rotate pairs (d, d + half) for d in 0..half.
                for d in 0..half {
                    let inv_freq =
                        (freq_base as f64).powf(-(2.0 * d as f64) / (rope_dim as f64));
                    let theta = (pos as f64) * inv_freq;
                    let (s_t, c_t) = (theta.sin() as f32, theta.cos() as f32);
                    let x = input[row_base + d];
                    let y = input[row_base + d + half];
                    out[row_base + d] = x * c_t - y * s_t;
                    out[row_base + d + half] = x * s_t + y * c_t;
                }
            }
        }
        out
    }

    #[test]
    fn adr_037_e4b5b_rope_cpu_parity_linear_positions_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let seq_len: u32 = 4;
        let num_heads = cfg.num_q_heads;
        let head_dim = cfg.head_dim;
        let rope_dim = cfg.rope_dim;
        let total = (seq_len as usize) * num_heads * head_dim;

        // Random F32 input (representing Q post-head-norm or raw projection).
        let mut input_data = vec![0.0f32; total];
        fill_random(&mut input_data, 0xB10);
        let input_gpu = upload_f32_to_gpu(
            &device, &input_data,
            vec![seq_len as usize * num_heads, head_dim],
        );

        let base_pos: u32 = 17;
        let positions: Vec<u32> = (0..seq_len).map(|i| base_pos + i).collect();
        let cpu_out = cpu_neox_rope(
            &input_data,
            &positions,
            seq_len as usize,
            num_heads,
            head_dim,
            rope_dim,
            cfg.rope_theta,
        );

        let mut enc = device.command_encoder().expect("encoder");
        let out_buf = dispatch_eagle3_rope(
            &mut enc,
            &mut registry,
            &device,
            &input_gpu,
            &cfg,
            seq_len,
            num_heads as u32,
            None, // linear positions
            base_pos,
            "rope_test_linear",
        )
        .expect("rope dispatch");
        enc.commit_and_wait().expect("commit");

        let gpu_out: &[f32] = out_buf.as_slice::<f32>().expect("output slice");
        assert_eq!(gpu_out.len(), total);
        let mut max_diff = 0.0f32;
        for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
            let d = (g - c).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(d < 1e-4, "rope linear parity: diff={d} > 1e-4");
        }
        eprintln!("rope linear parity max_diff={max_diff:.6e}");
    }

    #[test]
    fn adr_037_e4b5b_rope_cpu_parity_tree_positions_2026_05_22() {
        // Tree positions: simulate ExpandedTree where tree-node i has
        // depth d_i. Position[i] = base_pos + d_i. This is the path
        // EAGLE-3 uses for dynamic tree decoding.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let seq_len: u32 = 5;
        let num_heads = cfg.num_q_heads;
        let head_dim = cfg.head_dim;
        let rope_dim = cfg.rope_dim;
        let total = (seq_len as usize) * num_heads * head_dim;

        let mut input_data = vec![0.0f32; total];
        fill_random(&mut input_data, 0xB20);
        let input_gpu = upload_f32_to_gpu(
            &device, &input_data,
            vec![seq_len as usize * num_heads, head_dim],
        );

        // Asymmetric tree shape: depths = [0, 1, 1, 2, 2]
        // (root + 2 children + 2 grandchildren)
        let depths: [u32; 5] = [0, 1, 1, 2, 2];
        let base_pos: u32 = 42;
        let positions: Vec<u32> = depths.iter().map(|d| base_pos + d).collect();
        let cpu_out = cpu_neox_rope(
            &input_data,
            &positions,
            seq_len as usize,
            num_heads,
            head_dim,
            rope_dim,
            cfg.rope_theta,
        );

        let mut enc = device.command_encoder().expect("encoder");
        let out_buf = dispatch_eagle3_rope(
            &mut enc,
            &mut registry,
            &device,
            &input_gpu,
            &cfg,
            seq_len,
            num_heads as u32,
            Some(&positions),
            base_pos, // unused when positions_override = Some
            "rope_test_tree",
        )
        .expect("rope tree dispatch");
        enc.commit_and_wait().expect("commit");

        let gpu_out: &[f32] = out_buf.as_slice::<f32>().expect("output slice");
        let mut max_diff = 0.0f32;
        for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
            let d = (g - c).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(d < 1e-4, "rope tree parity: diff={d} > 1e-4");
        }
        // Sentinel: same-depth nodes get IDENTICAL rotations →
        // verify rows 1 (depth=1) and 2 (depth=1) produce equal
        // outputs IF given equal input.
        eprintln!("rope tree parity max_diff={max_diff:.6e}");
    }

    #[test]
    fn adr_037_e4b5b_rope_rejects_positions_len_mismatch_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let seq_len = 4_u32;
        let num_heads = cfg.num_q_heads as u32;
        let total = (seq_len as usize) * (num_heads as usize) * cfg.head_dim;
        let input = upload_f32_to_gpu(
            &device,
            &vec![0.0f32; total],
            vec![seq_len as usize * num_heads as usize, cfg.head_dim],
        );
        let bad_positions = vec![0u32, 1u32, 2u32]; // len != seq_len
        let mut enc = device.command_encoder().expect("encoder");
        let err = dispatch_eagle3_rope(
            &mut enc, &mut registry, &device,
            &input, &cfg, seq_len, num_heads,
            Some(&bad_positions), 0, "rope_bad_pos",
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("positions_override len"),
            "expected positions-len error, got: {err}"
        );
    }

    #[test]
    fn adr_037_e4b5b_gate_rope_rejects_position_above_i32_max_2026_05_22() {
        // Codex /cfa E4b.5b Major fix (2026-05-22): positions above
        // i32::MAX must be rejected, not silently saturated.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let seq_len = 2_u32;
        let num_heads = cfg.num_q_heads as u32;
        let total = (seq_len as usize) * (num_heads as usize) * cfg.head_dim;
        let input = upload_f32_to_gpu(
            &device,
            &vec![0.0f32; total],
            vec![seq_len as usize * num_heads as usize, cfg.head_dim],
        );
        // u32 value above i32::MAX
        let bad_positions = vec![0u32, (i32::MAX as u32) + 1];
        let mut enc = device.command_encoder().expect("encoder");
        let err = dispatch_eagle3_rope(
            &mut enc, &mut registry, &device,
            &input, &cfg, seq_len, num_heads,
            Some(&bad_positions), 0, "rope_overflow",
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("exceeds i32::MAX"),
            "expected i32::MAX rejection, got: {err}"
        );
    }

    #[test]
    fn adr_037_e4b5b_gate_rope_rejects_non_f32_input_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let seq_len = 2_u32;
        let num_heads = cfg.num_q_heads as u32;
        let total = (seq_len as usize) * (num_heads as usize) * cfg.head_dim;
        let bad = device
            .alloc_buffer(
                total * 2,
                DType::BF16,
                vec![seq_len as usize * num_heads as usize, cfg.head_dim],
            )
            .expect("alloc bad");
        let mut enc = device.command_encoder().expect("encoder");
        let err = dispatch_eagle3_rope(
            &mut enc, &mut registry, &device,
            &bad, &cfg, seq_len, num_heads,
            None, 0, "rope_bad_dtype",
        )
        .unwrap_err();
        assert!(err.to_string().contains("dtype must be F32"), "got: {err}");
    }

    // ----------------------------------------------------------------
    // Phase E4b.6 tests — tree_attention dispatch via Phase E1 kernel
    // ----------------------------------------------------------------

    fn cfg_for_attention_dk128() -> Eagle3DrafterConfig {
        // head_dim=128 to exercise the new mlx-native dk128 kernel
        // template. num_q_heads × head_dim = 4 × 128 = 512 hidden;
        // keeps test fast while exercising production-shaped head_dim.
        Eagle3DrafterConfig {
            hidden_size: 512,
            intermediate_size: 1024,
            head_dim: 128,
            num_q_heads: 4,
            num_kv_heads: 2,
            vocab_size: 1000,
            draft_vocab_size: 1000,
            target_hidden_size: 512,
            num_aux_hidden_states: 3,
            rms_norm_eps: 1e-6,
            norm_before_fc: false,
            fc_norm: false,
            use_qk_norm: false,
            attention_bias: false,
            tie_lm_head: true,
            include_draft_id_mapping: false,
            has_own_embed_tokens: false,
            rope_theta: 1_000_000.0,
            rope_dim: 128,
        }
    }

    /// CPU reference for tree attention: matches the mlx-native
    /// `cpu_tree_sdpa` semantics used in Phase E1.3 tests, adapted
    /// for our Q layout `[num_heads, q_seq, head_dim]` (head-outer).
    #[allow(clippy::too_many_arguments)]
    fn cpu_tree_attention_reference(
        q: &[f32],         // [num_q_heads, q_seq_len, head_dim]
        k: &[f32],         // [num_kv_heads, kv_capacity, head_dim]
        v: &[f32],         // [num_kv_heads, kv_capacity, head_dim]
        mask: &[f32],      // [q_seq_len, mask_stride]
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        q_seq_len: usize,
        kv_seq_len: usize,
        kv_capacity: usize,
        mask_stride: usize,
        scale: f32,
    ) -> Vec<f32> {
        let heads_per_kv = num_q_heads / num_kv_heads;
        // Output layout: [q_seq_len, num_q_heads, head_dim].
        let mut out = vec![0.0f32; q_seq_len * num_q_heads * head_dim];
        for h in 0..num_q_heads {
            let kv_h = h / heads_per_kv;
            for iq1 in 0..q_seq_len {
                let q_off = h * q_seq_len * head_dim + iq1 * head_dim;
                let mask_row = iq1 * mask_stride;
                let mut scores = Vec::<(usize, f32)>::new();
                for k_pos in 0..kv_seq_len {
                    if mask[mask_row + k_pos] == EAGLE3_TREE_MASK_MASKED {
                        continue;
                    }
                    let k_off = kv_h * kv_capacity * head_dim + k_pos * head_dim;
                    let mut dot = 0.0f64;
                    for d in 0..head_dim {
                        dot += q[q_off + d] as f64 * k[k_off + d] as f64;
                    }
                    scores.push((k_pos, dot as f32 * scale));
                }
                if scores.is_empty() {
                    continue;
                }
                let max_s = scores
                    .iter()
                    .map(|(_, s)| *s)
                    .fold(f32::NEG_INFINITY, f32::max);
                let exp_s: Vec<f32> =
                    scores.iter().map(|(_, s)| (*s - max_s).exp()).collect();
                let sum_e: f32 = exp_s.iter().sum();
                let inv = if sum_e == 0.0 { 0.0 } else { 1.0 / sum_e };
                // Output: [q_seq, n_q, hd] layout.
                let o_off = iq1 * num_q_heads * head_dim + h * head_dim;
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for ((k_pos, _), &es) in scores.iter().zip(exp_s.iter()) {
                        let weight = es * inv;
                        let v_off = kv_h * kv_capacity * head_dim + k_pos * head_dim;
                        acc += weight * v[v_off + d];
                    }
                    out[o_off + d] = acc;
                }
            }
        }
        out
    }

    #[test]
    fn adr_037_e4b6_tree_attention_cpu_parity_dk128_fixed_square_2026_05_22() {
        // Use head_dim=128 (Qwen 3.6 27B production shape) +
        // fixed-square tree (root + 4 leaves; depth=2) over a
        // prefix of 27 tokens. Exercises both the new dk128
        // kernel template AND non-trivial tree masking.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = cfg_for_attention_dk128();

        let num_q_heads = cfg.num_q_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let q_seq_len = 5_u32; // 1 root + 4 leaves
        let prefix_len = 27_usize;
        let kv_seq_len = (prefix_len + q_seq_len as usize) as u32; // 32
        let kv_capacity = 64_u32;
        let mask_stride = kv_seq_len;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        // Random Q [n_q, q_seq, hd]
        let q_elems = num_q_heads * (q_seq_len as usize) * head_dim;
        let mut q_data = vec![0.0f32; q_elems];
        fill_random(&mut q_data, 0xC60);
        // Random K/V [n_kv, kv_capacity, hd] — initialize past
        // kv_seq_len with junk (mask should mask them).
        let kv_elems = num_kv_heads * (kv_capacity as usize) * head_dim;
        let mut k_data = vec![0.0f32; kv_elems];
        fill_random(&mut k_data, 0xC61);
        let mut v_data = vec![0.0f32; kv_elems];
        fill_random(&mut v_data, 0xC62);

        // Build fixed-square mask: tree-nodes occupy positions
        // [prefix_len, prefix_len + q_seq_len). Root at 27, leaves at
        // 28..32. Each leaf attends prefix + root + self.
        let mask_elems = (q_seq_len as usize) * (mask_stride as usize);
        let mut mask_data = vec![EAGLE3_TREE_MASK_MASKED; mask_elems];
        for iq1 in 0..(q_seq_len as usize) {
            let row_base = iq1 * (mask_stride as usize);
            // Prefix always attended
            for k in 0..prefix_len {
                mask_data[row_base + k] = EAGLE3_TREE_MASK_ATTENDED;
            }
            // Self
            mask_data[row_base + prefix_len + iq1] = EAGLE3_TREE_MASK_ATTENDED;
            // Leaves (iq1 > 0) attend root (iq1=0 at prefix_len)
            if iq1 > 0 {
                mask_data[row_base + prefix_len] = EAGLE3_TREE_MASK_ATTENDED;
            }
        }

        // CPU reference
        let cpu_out = cpu_tree_attention_reference(
            &q_data, &k_data, &v_data, &mask_data,
            num_q_heads, num_kv_heads, head_dim,
            q_seq_len as usize, kv_seq_len as usize, kv_capacity as usize,
            mask_stride as usize, scale,
        );

        // GPU
        let q_gpu = upload_f32_to_gpu(&device, &q_data,
            vec![num_q_heads, q_seq_len as usize, head_dim]);
        let k_gpu = upload_f32_to_gpu(&device, &k_data,
            vec![num_kv_heads, kv_capacity as usize, head_dim]);
        let v_gpu = upload_f32_to_gpu(&device, &v_data,
            vec![num_kv_heads, kv_capacity as usize, head_dim]);
        let mask_gpu = upload_f32_to_gpu(&device, &mask_data,
            vec![q_seq_len as usize, mask_stride as usize]);

        let mut enc = device.command_encoder().expect("encoder");
        let out_buf = dispatch_eagle3_tree_attention(
            &mut enc, &mut registry, &device,
            &q_gpu, &k_gpu, &v_gpu, &mask_gpu,
            &cfg, q_seq_len, kv_seq_len, kv_capacity, mask_stride, scale,
        )
        .expect("tree attn dispatch");
        enc.commit_and_wait().expect("commit");

        let gpu_out: &[f32] = out_buf.as_slice::<f32>().expect("output slice");
        assert_eq!(gpu_out.len(), cpu_out.len());
        let mut max_diff = 0.0f32;
        for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
            let d = (g - c).abs();
            if d > max_diff {
                max_diff = d;
            }
            // Softmax + GEMM in F32 input × F32 KV gives ~1e-3
            // relative; absolute bound at scale ~ 1/sqrt(128) ≈ 0.088
            // and Q/K random in [-1,1) is well within 5e-3.
            assert!(d < 5e-3, "tree_attention parity: diff={d} > 5e-3");
        }
        eprintln!("tree_attention dk128 fixed-square parity max_diff={max_diff:.6e}");
    }

    #[test]
    fn adr_037_e4b6_gate_tree_attention_rejects_zero_q_seq_len_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = cfg_for_attention_dk128();
        // Dummy buffers — won't actually be read because validation
        // fails before kernel dispatch.
        let dummy = device
            .alloc_buffer(4, DType::F32, vec![1])
            .expect("alloc dummy");
        let mut enc = device.command_encoder().expect("encoder");
        let err = dispatch_eagle3_tree_attention(
            &mut enc, &mut registry, &device,
            &dummy, &dummy, &dummy, &dummy,
            &cfg, 0, 1, 1, 1, 1.0,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("q_seq_len must be > 0"),
            "got: {err}"
        );
    }

    #[test]
    fn adr_037_e4b6_gate_tree_attention_rejects_kv_capacity_less_than_seq_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = cfg_for_attention_dk128();
        let dummy = device.alloc_buffer(4, DType::F32, vec![1]).expect("alloc");
        let mut enc = device.command_encoder().expect("encoder");
        let err = dispatch_eagle3_tree_attention(
            &mut enc, &mut registry, &device,
            &dummy, &dummy, &dummy, &dummy,
            &cfg, 1, 10, 5, 10, 1.0, // kv_capacity 5 < kv_seq_len 10
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("kv_capacity"),
            "got: {err}"
        );
    }

    // ----------------------------------------------------------------
    // Phase E4b.7 tests — O projection + residual add
    // ----------------------------------------------------------------

    #[test]
    fn adr_037_e4b7_o_proj_cpu_parity_no_bias_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg(); // attention_bias=false
        let seq_len: u32 = 4;
        // Input to o_proj is the attention output: [seq, num_q*head_dim].
        let in_features = cfg.q_proj_out();
        let out_features = cfg.hidden_size;

        let mut input_data = vec![0.0f32; (seq_len as usize) * in_features];
        fill_random(&mut input_data, 0xD70);
        let mut weight_f32 = vec![0.0f32; out_features * in_features];
        fill_random(&mut weight_f32, 0xD71);

        // CPU reference: BF16-quantized weight matmul.
        let weight_bf16_q: Vec<f32> =
            weight_f32.iter().map(|&v| bf16_quantize_f32(v)).collect();
        let cpu_out = cpu_fc_reference(
            &input_data,
            &weight_bf16_q,
            seq_len as usize,
            in_features,
            out_features,
        );

        // GPU: synthetic safetensors blob with custom o_proj.weight.
        let manifest = expected_manifest(&cfg);
        let mut overrides = std::collections::HashMap::new();
        overrides.insert(
            "layers.0.self_attn.o_proj.weight".to_string(),
            f32_to_bf16_bytes(&weight_f32),
        );
        let blob = build_blob_with_overrides(&manifest, &overrides);
        let weights = Eagle3Weights::load(&blob, &cfg).expect("load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");
        let input_gpu = upload_f32_to_gpu(
            &device,
            &input_data,
            vec![seq_len as usize, in_features],
        );

        let mut enc = device.command_encoder().expect("encoder");
        let out_buf = dispatch_eagle3_o_proj(
            &mut enc, &mut registry, &device, &input_gpu, &tensors, &cfg, seq_len,
        )
        .expect("o_proj dispatch");
        enc.commit_and_wait().expect("commit");

        let gpu_out: &[f32] = out_buf.as_slice::<f32>().expect("output slice");
        let mut max_diff = 0.0f32;
        for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
            let d = (g - c).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(d < 5e-2, "o_proj parity: diff={d} > 5e-2");
        }
        eprintln!("o_proj parity max_diff={max_diff:.6e}");
    }

    #[test]
    fn adr_037_e4b7_residual_add_cpu_parity_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let seq_len: u32 = 4;
        let hidden = cfg.hidden_size;
        let n = (seq_len as usize) * hidden;

        let mut a_data = vec![0.0f32; n];
        let mut b_data = vec![0.0f32; n];
        fill_random(&mut a_data, 0xE70);
        fill_random(&mut b_data, 0xE71);
        let cpu_out: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(a, b)| a + b).collect();

        let a_gpu = upload_f32_to_gpu(&device, &a_data, vec![seq_len as usize, hidden]);
        let b_gpu = upload_f32_to_gpu(&device, &b_data, vec![seq_len as usize, hidden]);

        let mut enc = device.command_encoder().expect("encoder");
        let out_buf = dispatch_eagle3_residual_add(
            &mut enc, &mut registry, &device, &a_gpu, &b_gpu, &cfg, seq_len,
        )
        .expect("residual_add");
        enc.commit_and_wait().expect("commit");

        let gpu_out: &[f32] = out_buf.as_slice::<f32>().expect("output slice");
        // F32 add — should be bit-exact (no precision loss).
        for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
            assert_eq!(g.to_bits(), c.to_bits(), "residual_add bit-equal expected");
        }
    }

    #[test]
    fn adr_037_e4b7_gate_residual_add_rejects_non_f32_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let seq_len = 2_u32;
        let n = (seq_len as usize) * cfg.hidden_size;
        let bf16 = device
            .alloc_buffer(n * 2, DType::BF16, vec![seq_len as usize, cfg.hidden_size])
            .expect("alloc bf16");
        let f32_buf = device
            .alloc_buffer(n * 4, DType::F32, vec![seq_len as usize, cfg.hidden_size])
            .expect("alloc f32");
        let mut enc = device.command_encoder().expect("encoder");
        let err = dispatch_eagle3_residual_add(
            &mut enc, &mut registry, &device, &bf16, &f32_buf, &cfg, seq_len,
        )
        .unwrap_err();
        assert!(err.to_string().contains("must be F32"), "got: {err}");
    }

    #[test]
    fn adr_037_e4b7_gate_residual_add_rejects_shape_mismatch_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let seq_len = 4_u32;
        let n = (seq_len as usize) * cfg.hidden_size;
        let good = upload_f32_to_gpu(&device, &vec![0.0f32; n], vec![seq_len as usize, cfg.hidden_size]);
        let bad = upload_f32_to_gpu(&device, &vec![0.0f32; 10], vec![10]);
        let mut enc = device.command_encoder().expect("encoder");
        let err = dispatch_eagle3_residual_add(
            &mut enc, &mut registry, &device, &good, &bad, &cfg, seq_len,
        )
        .unwrap_err();
        assert!(err.to_string().contains("b has"), "got: {err}");
    }

    #[test]
    fn adr_037_e4b3_concat_rejects_wrong_input_elements_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);
        let blob = build_blob_with_overrides(&manifest, &std::collections::HashMap::new());
        let weights = Eagle3Weights::load(&blob, &cfg).expect("weights load");
        let _tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");

        let seq_len = 2_u32;
        let good_data = vec![0.0f32; (seq_len as usize) * cfg.hidden_size];
        let good_gpu = upload_f32_to_gpu(
            &device, &good_data, vec![seq_len as usize, cfg.hidden_size],
        );
        // hidden branch undersized
        let bad_data = vec![0.0f32; 10];
        let bad_gpu = upload_f32_to_gpu(&device, &bad_data, vec![10]);

        let mut enc = device.command_encoder().expect("encoder");
        let err = dispatch_eagle3_concat_2x_hidden(
            &mut enc, &mut registry, &device, &good_gpu, &bad_gpu, &cfg, seq_len,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("hidden_normed has"),
            "expected element-count error, got: {err}"
        );
    }

    #[test]
    fn adr_037_e4b2_fc_output_shape_correct_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);
        let blob = build_blob_with_fc_weight(
            &manifest,
            &vec![0u8; cfg.hidden_size * cfg.fc_input_size() * 2],
        );
        let weights = Eagle3Weights::load(&blob, &cfg).expect("weights load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");

        let seq_len = 8_u32;
        let input_data = vec![0.0f32; (seq_len as usize) * cfg.fc_input_size()];
        let input_gpu = upload_f32_to_gpu(
            &device,
            &input_data,
            vec![seq_len as usize, cfg.fc_input_size()],
        );

        let mut enc = device.command_encoder().expect("encoder");
        let out_buf = dispatch_eagle3_fc(
            &mut enc, &mut registry, &device, &input_gpu, &tensors, &cfg, seq_len,
        )
        .expect("dispatch");
        enc.commit_and_wait().expect("commit");
        assert_eq!(out_buf.dtype(), DType::F32);
        assert_eq!(
            out_buf.element_count(),
            (seq_len as usize) * cfg.hidden_size
        );
    }
}
