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
use crate::inference::models::qwen35::gpu_full_attn::apply_linear_projection_f32;
use anyhow::{anyhow, Context, Result};
use mlx_native::ops::add_bias_row_2d::dispatch_add_bias_row_2d_f32;
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
        assert!(
            err.to_string().contains("q_norm absent"),
            "expected absent-tensor error, got: {err}"
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
