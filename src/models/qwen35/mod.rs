//! Qwen3.5-family conversion module — shared (Dense + MoE) logic.
//!
//! # Naming boundary
//! This module (`src/models/qwen35/`) is the **conversion-side** module owned by
//! ADR-012.  The **inference-side** module at `src/inference/models/qwen35/` is
//! owned by ADR-013 and is developed by a separate parallel session.  These two
//! modules must never cross-import.
//!
//! # Architecture
//!
//! ## Enum + context
//! `Qwen35Arch` distinguishes between the two Qwen3.5-family shapes:
//! - `Dense`: standard dense FFN (Qwen3.5-27B-Instruct, etc.)
//! - `Moe`:   Mixture-of-Experts FFN (Qwen3.5-MoE-35B-A3.9B, etc.)
//!
//! `Qwen35ConvertContext` is derived from `ModelMetadata` (P1 output).  It holds
//! pre-extracted hparams so downstream code never has to unwrap Options repeatedly.
//!
//! ## V-head grouped → tiled reorder
//!
//! Port of the six-case reorder in:
//!   `/opt/llama.cpp/convert_hf_to_gguf.py:5375-5424`
//!   (`_LinearAttentionVReorderBase.modify_tensors`)
//!
//! ### Why the reorder is needed
//! Linear-attention layers in Qwen3.5 can have `num_v_heads > num_k_heads`.
//! HF stores V heads **grouped by K head**: `[G0_v0..v{r-1}, G1_v0..v{r-1}, …]`
//! ggml binary ops use **tiled broadcast**: `[K0, K1, …, K0, K1, …]`
//! i.e. the slower-varying index must be `v_within_k_group`, faster-varying `k_head`.
//!
//! The reorder swaps the `num_k_heads` and `num_v_per_k` axes along the
//! relevant tensor dimension.  See `reorder_v_heads` for the index math.
//!
//! ### Helper signature
//! ```text
//! fn reorder_v_heads(
//!     data:        &[u8],          // raw bytes of the tensor slice to reorder
//!     elem_size:   usize,          // bytes per element (dtype-dependent)
//!     num_k_heads: u32,            // number of K heads
//!     num_v_per_k: u32,            // number of V heads per K head  (= num_v_heads / num_k_heads)
//!     head_dim:    u32,            // elements per head in the reordered slice
//! ) -> Vec<u8>
//! ```
//! The input slice is interpreted as `[num_k_heads, num_v_per_k, head_dim]` elements
//! (C-contiguous, outer-first).  The output is the same shape with axes 0 and 1
//! transposed: `[num_v_per_k, num_k_heads, head_dim]`, then flattened back.
//!
//! This matches the Python one-liner (py:5273-5282):
//! ```python
//! new_shape = [num_k_heads, num_v_per_k, head_dim]
//! tensor = tensor.reshape(*new_shape)
//! perm = [1, 0, 2]           # swap dim 0 and dim 1
//! tensor.permute(*perm).contiguous().reshape(*original_shape)
//! ```
//!
//! ### The 6 cases (py:5384-5422)
//! | # | Tensor-name substring | What is reordered | dim | head_dim arg |
//! |---|---|---|---|---|
//! | 1 | `.in_proj_qkv.` | V rows only (rows after q_dim+k_dim) | row (0) | head_v_dim |
//! | 2 | `.in_proj_z.`   | All rows | row (0) | head_v_dim |
//! | 3 | `.in_proj_b.` / `.in_proj_a.` | All rows | row (0) | 1 |
//! | 4 | `.A_log` / `.dt_bias` / `.dt_proj` | Last dim (may be 1-D) | col (last) | 1 |
//! | 5 | `.conv1d` | V channel portion (channels after `head_k_dim * num_k_heads * 2`) | row (0) | head_v_dim |
//! | 6 | `.out_proj.` | Columns (input dim) | col (1) | head_v_dim |
//!
//! ## Stub labelling
//! Stubs for P3 (A_log negation body, conv1d squeeze, in_proj_qkvz reorder) and
//! P4/P5 (metadata emission, expert merge) return `ConvertError::PhaseStub` with
//! the phase label.  **No `unimplemented!()`, no `todo!()`, no empty bodies.**

pub mod dense;
pub mod moe;

use crate::input::config_parser::{
    validate_required_qwen35_fields, validate_required_qwen35moe_fields, ConfigParseError,
};
use crate::ir::{DType, ModelMetadata, TensorRef};
use thiserror::Error;

// ---------------------------------------------------------------------------
// ConvertError
// ---------------------------------------------------------------------------

/// Errors from qwen35 conversion operations.
#[derive(Error, Debug)]
pub enum ConvertError {
    /// A feature that is fully specified here but implemented in a later phase.
    /// No `unimplemented!()` — this error documents the contract.
    #[error("Phase {phase} stub: '{what}' is not yet implemented (scheduled for {phase})")]
    PhaseStub { phase: &'static str, what: &'static str },

    /// Required hparam field is absent from the parsed metadata.
    #[error("Qwen3.5 conversion: missing required hparam '{field}'")]
    MissingHparam { field: &'static str },

    /// Configuration validation failed (wraps ConfigParseError).
    #[error("Config validation: {0}")]
    ConfigValidation(#[from] ConfigParseError),

    /// Tensor data length is inconsistent with declared shape/dtype.
    #[error("Tensor '{name}' data length {actual} does not match expected {expected} bytes")]
    TensorLengthMismatch { name: String, expected: usize, actual: usize },

    /// A reorder invariant was violated (e.g. total element count changed).
    #[error("Reorder invariant violated for tensor '{name}': {reason}")]
    ReorderInvariantViolated { name: String, reason: String },

    /// `merge_expert_tensors` was called from a dense-arch context.
    /// Dense FFN tensors are emitted as singletons — expert merge is MoE-only.
    #[error("merge_expert_tensors called from a Dense arch context; expert merge is MoE-only")]
    DenseContextMergeCall,

    /// `merge_expert_tensors` was called with an empty expert slice.
    #[error("merge_expert_tensors: expert_tensors slice is empty")]
    ExpertMergeEmpty,

    /// A tensor's shape or dtype does not match expert 0.
    #[error("merge_expert_tensors: expert {expert_idx} mismatch — {reason}")]
    ExpertMergeShapeMismatch { expert_idx: usize, reason: String },
}

// ---------------------------------------------------------------------------
// Arch enum
// ---------------------------------------------------------------------------

/// Which Qwen3.5-family shape this model has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35Arch {
    /// Standard dense FFN (e.g. Qwen3.5-27B-Instruct).
    Dense,
    /// Mixture-of-Experts FFN (e.g. Qwen3.5-MoE-35B-A3.9B).
    Moe,
}

// ---------------------------------------------------------------------------
// Convert context
// ---------------------------------------------------------------------------

/// Pre-extracted hparams derived from `ModelMetadata` for Qwen3.5-family conversion.
///
/// Built once at arch-dispatch time; downstream code never unwraps `Option`s.
/// All fields are the result of calling `validate_required_qwen35moe_fields` first.
#[derive(Debug, Clone)]
pub struct Qwen35ConvertContext {
    /// Dense or MoE variant.
    pub arch: Qwen35Arch,

    /// Resolved per-layer type list (from `ModelMetadata::resolved_layer_types()`).
    /// Length == `num_layers`.
    pub layer_types: Vec<String>,

    /// Total number of transformer layers.
    pub num_layers: u32,

    // --- full-attention hparams ---

    /// Number of Q/K/V heads in full-attention layers.
    pub num_attention_heads: u32,
    /// Number of KV heads (GQA) in full-attention layers.
    pub num_kv_heads: u32,
    /// Head dimension in full-attention layers (explicit from config — never derived).
    pub head_dim: u32,

    // --- linear-attention (Gated DeltaNet) hparams ---

    /// Convolution kernel width (linear_conv_kernel_dim).
    pub linear_conv_kernel_dim: u32,
    /// Key head dimension in linear-attention layers.
    pub linear_key_head_dim: u32,
    /// Number of key heads in linear-attention layers.
    pub linear_num_key_heads: u32,
    /// Value head dimension in linear-attention layers.
    pub linear_value_head_dim: u32,
    /// Number of value heads in linear-attention layers.
    pub linear_num_value_heads: u32,

    // --- derived convenience ---

    /// `linear_num_value_heads / linear_num_key_heads`.
    /// This is the `num_v_per_k` used throughout the reorder logic.
    pub linear_num_v_per_k: u32,

    // --- MoE sizing (Some only when arch == Moe) ---

    /// Expert FFN intermediate size (moe_intermediate_size).
    pub moe_intermediate_size: Option<u32>,
    /// Shared-expert intermediate size (shared_expert_intermediate_size).
    pub shared_expert_intermediate_size: Option<u32>,
    /// Number of experts (num_experts).
    pub num_experts: Option<u32>,
    /// Top-k experts per token (top_k_experts).
    pub top_k_experts: Option<u32>,

    // --- dense FFN sizing (Some only when arch == Dense) ---

    /// Dense FFN intermediate size (intermediate_size).
    pub intermediate_size: Option<u64>,
}

impl Qwen35ConvertContext {
    /// Build a `Qwen35ConvertContext` from a `ModelMetadata` produced by P1 parsing.
    ///
    /// Calls `validate_required_qwen35moe_fields` first so that the caller gets
    /// a clear error if any required field is absent.
    ///
    /// The `arch` variant is inferred from `metadata.is_moe()`.
    pub fn from_metadata(metadata: &ModelMetadata) -> Result<Self, ConvertError> {
        // Arch-specific validators: dense doesn't need moe_intermediate_size
        // or shared_expert_intermediate_size. Calling the MoE validator on a
        // dense model produced false-positive "missing field" errors pre-
        // 2026-04-24.
        let arch = if metadata.is_moe() {
            validate_required_qwen35moe_fields(metadata)?;
            Qwen35Arch::Moe
        } else {
            validate_required_qwen35_fields(metadata)?;
            Qwen35Arch::Dense
        };

        let layer_types = metadata.resolved_layer_types();

        // Unwrap fields already validated above.
        let head_dim = metadata
            .head_dim
            .ok_or(ConvertError::MissingHparam { field: "head_dim" })?;
        let linear_conv_kernel_dim = metadata
            .linear_conv_kernel_dim
            .ok_or(ConvertError::MissingHparam { field: "linear_conv_kernel_dim" })?;
        let linear_key_head_dim = metadata
            .linear_key_head_dim
            .ok_or(ConvertError::MissingHparam { field: "linear_key_head_dim" })?;
        let linear_num_key_heads = metadata
            .linear_num_key_heads
            .ok_or(ConvertError::MissingHparam { field: "linear_num_key_heads" })?;
        let linear_value_head_dim = metadata
            .linear_value_head_dim
            .ok_or(ConvertError::MissingHparam { field: "linear_value_head_dim" })?;
        let linear_num_value_heads = metadata
            .linear_num_value_heads
            .ok_or(ConvertError::MissingHparam { field: "linear_num_value_heads" })?;

        // Guard: num_v must be a multiple of num_k.
        if linear_num_key_heads == 0 || linear_num_value_heads % linear_num_key_heads != 0 {
            return Err(ConvertError::MissingHparam {
                field: "linear_num_value_heads (not a multiple of linear_num_key_heads)",
            });
        }
        let linear_num_v_per_k = linear_num_value_heads / linear_num_key_heads;

        let num_kv_heads = metadata.num_kv_heads.unwrap_or(metadata.num_attention_heads);

        Ok(Self {
            arch,
            layer_types,
            num_layers: metadata.num_layers,
            num_attention_heads: metadata.num_attention_heads,
            num_kv_heads,
            head_dim,
            linear_conv_kernel_dim,
            linear_key_head_dim,
            linear_num_key_heads,
            linear_value_head_dim,
            linear_num_value_heads,
            linear_num_v_per_k,
            moe_intermediate_size: metadata.moe_intermediate_size,
            shared_expert_intermediate_size: metadata.shared_expert_intermediate_size,
            num_experts: metadata.num_experts,
            top_k_experts: metadata.top_k_experts,
            intermediate_size: metadata.intermediate_size,
        })
    }
}

// ---------------------------------------------------------------------------
// Layer-type predicates
// ---------------------------------------------------------------------------

/// Returns `true` if `layer_idx` (0-based) is a linear-attention layer.
///
/// Uses `ctx.layer_types` which was built from
/// `ModelMetadata::resolved_layer_types()` (P1 contract).
pub fn is_linear_attention_layer(layer_idx: usize, ctx: &Qwen35ConvertContext) -> bool {
    ctx.layer_types
        .get(layer_idx)
        .map(|t| t == "linear_attention")
        .unwrap_or(false)
}

/// Returns `true` if `layer_idx` (0-based) is a full-attention layer.
pub fn is_full_attention_layer(layer_idx: usize, ctx: &Qwen35ConvertContext) -> bool {
    ctx.layer_types
        .get(layer_idx)
        .map(|t| t == "full_attention")
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// V-head grouped → tiled reorder — core helper
// ---------------------------------------------------------------------------

/// Reorder V heads from HF grouped layout to ggml tiled layout.
///
/// # Layout semantics
///
/// HF stores V heads **grouped by K head** (outer dimension = k_group):
/// ```text
/// [G0_v0, G0_v1, …, G0_v{r-1},   G1_v0, …, G1_v{r-1},   …, G{K-1}_v{r-1}]
/// ```
/// where `r = num_v_per_k` and each Gx_vy occupies `head_dim` consecutive elements.
///
/// ggml binary ops use **tiled broadcast** (outer dimension = v_within_group):
/// ```text
/// [G0_v0, G1_v0, …, G{K-1}_v0,   G0_v1, …, G{K-1}_v1,   …, G{K-1}_v{r-1}]
/// ```
///
/// This is a transpose of the `[num_k_heads, num_v_per_k]` axes when the
/// slice is viewed as `[num_k_heads, num_v_per_k, head_dim]`.
///
/// # Python reference
/// `_LinearAttentionVReorderBase._reorder_v_heads` at
/// `/opt/llama.cpp/convert_hf_to_gguf.py:5272-5282`:
/// ```python
/// new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim+1:]
/// tensor = tensor.reshape(*new_shape)
/// perm = list(range(len(new_shape)))
/// perm[dim], perm[dim+1] = perm[dim+1], perm[dim]   # swap dim and dim+1
/// tensor.permute(*perm).contiguous().reshape(*shape)
/// ```
/// Here we only handle the common 1-D slice case (the slice passed in is
/// already the relevant rows or columns extracted by the caller).
///
/// # Arguments
/// - `data`: raw bytes of the slice; must be `num_k_heads * num_v_per_k * head_dim`
///   elements long (each element is `elem_size` bytes).
/// - `elem_size`: bytes per scalar element (from `DType::element_size()`).
/// - `num_k_heads`: K axis (outer in HF layout).
/// - `num_v_per_k`: V-per-K axis (inner in HF layout, outer in ggml layout).
/// - `head_dim`: number of elements per individual head.
///
/// # Returns
/// A new `Vec<u8>` with the same length as `data` but in tiled order.
///
/// # Invariant
/// Applying this function twice (with the same params) returns the original data,
/// because swapping two axes is a self-inverse permutation.
pub fn reorder_v_heads(
    data: &[u8],
    elem_size: usize,
    num_k_heads: u32,
    num_v_per_k: u32,
    head_dim: u32,
) -> Result<Vec<u8>, ConvertError> {
    let nk = num_k_heads as usize;
    let nv = num_v_per_k as usize;
    let hd = head_dim as usize;

    let expected_bytes = nk * nv * hd * elem_size;
    if data.len() != expected_bytes {
        return Err(ConvertError::ReorderInvariantViolated {
            name: String::from("<slice>"),
            reason: format!(
                "expected {} bytes ({}*{}*{}*{}), got {}",
                expected_bytes, nk, nv, hd, elem_size, data.len()
            ),
        });
    }

    // Source layout: [nk, nv, hd] (C-order, row-major)
    //   src[k, v, d] is at element offset: k * (nv * hd) + v * hd + d
    //
    // Target layout: [nv, nk, hd]
    //   dst[v, k, d] is at element offset: v * (nk * hd) + k * hd + d
    //
    // For each (k, v, d) in source, copy head-sized block from
    //   src_elem = k * (nv * hd) + v * hd + d
    // to
    //   dst_elem = v * (nk * hd) + k * hd + d
    //
    // Optimised: copy entire head_dim block per (k, v) pair.

    let mut out = vec![0u8; expected_bytes];

    for k in 0..nk {
        for v in 0..nv {
            let src_elem_off = (k * nv * hd + v * hd) * elem_size;
            let dst_elem_off = (v * nk * hd + k * hd) * elem_size;
            let block = hd * elem_size;
            out[dst_elem_off..dst_elem_off + block]
                .copy_from_slice(&data[src_elem_off..src_elem_off + block]);
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// transform_linear_attn_tensor — dispatcher for the 6 cases
// ---------------------------------------------------------------------------

/// Transform a linear-attention tensor from HF layout to ggml layout.
///
/// Dispatches to the appropriate case based on the HF tensor name suffix,
/// matching the six branches in:
/// `/opt/llama.cpp/convert_hf_to_gguf.py:5384-5422`
///
/// # Case summary
///
/// | Case | Trigger (contains) | Action |
/// |------|--------------------|--------|
/// | 1 | `.in_proj_qkv.` | V rows reorder (rows after q_dim+k_dim) |
/// | 2 | `.in_proj_z.`   | All rows reorder (head_v_dim) |
/// | 3 | `.in_proj_b.` / `.in_proj_a.` | All rows reorder (head_dim=1) |
/// | 4 | `.A_log` / `.dt_bias` / `.dt_proj` | 1-D or last-dim reorder (head_dim=1) |
/// | 5 | `.conv1d` | V channel portion reorder |
/// | 6 | `.out_proj.` | Column (input-dim) reorder |
///
/// If the tensor name does not match any case the tensor is returned unchanged.
///
/// A_log negation, conv1d squeeze, and in_proj_qkvz reorder bodies are scheduled
/// for P3 and return `ConvertError::PhaseStub` from this dispatcher.
pub fn transform_linear_attn_tensor(
    tensor: TensorRef,
    ctx: &Qwen35ConvertContext,
) -> Result<TensorRef, ConvertError> {
    let name = &tensor.name;
    let nk = ctx.linear_num_key_heads;
    let nv_per_k = ctx.linear_num_v_per_k;
    let head_k_dim = ctx.linear_key_head_dim;
    let head_v_dim = ctx.linear_value_head_dim;
    let elem_size = tensor.dtype.element_size();

    // Only reorder when GQA mismatch exists (py:5379)
    // "if num_k_heads > 0 and num_v_heads > 0 and num_k_heads != num_v_heads"
    if nv_per_k == 1 {
        // num_k_heads == num_v_heads: no reorder needed
        return Ok(tensor);
    }

    // Case 1: in_proj_qkv — reorder only the V rows (py:5384-5392)
    if name.contains(".in_proj_qkv.") {
        return transform_case1_in_proj_qkv(tensor, nk, nv_per_k, head_k_dim, head_v_dim, elem_size);
    }

    // Case 2: in_proj_z — reorder all rows (py:5394-5396)
    if name.contains(".in_proj_z.") {
        return transform_case2_in_proj_z(tensor, nk, nv_per_k, head_v_dim, elem_size);
    }

    // Case 3: in_proj_b / in_proj_a — reorder all rows, head_dim=1 (py:5398-5400)
    if name.contains(".in_proj_b.") || name.contains(".in_proj_a.") {
        return transform_case3_in_proj_ab(tensor, nk, nv_per_k, elem_size);
    }

    // Case 4: A_log / dt_bias / dt_proj — 1-D or last-dim reorder (py:5402-5409)
    //
    // Python reference (convert_hf_to_gguf.py:5402-5409):
    //   elif ".A_log" in name or ".dt_bias" in name or ".dt_proj" in name:
    //     if data_torch.ndim == 1:
    //       data_torch = _reorder_v_heads(data_torch.unsqueeze(-1), 0, nk, nv_per_k, 1).squeeze(-1)
    //     else:
    //       data_torch = _reorder_v_heads(data_torch, -1, nk, nv_per_k, 1)
    //
    // A_log negation is applied by Qwen3NextModel.modify_tensors (convert_hf_to_gguf.py:4788-4789)
    // BEFORE the V-head reorder in _LinearAttentionVReorderBase.modify_tensors.
    // We apply negation + reorder in one pass here for efficiency; the result is identical.
    //
    // dt_bias rename (.dt_bias → .dt_proj.bias) is applied by Qwen3NextModel.modify_tensors
    // (convert_hf_to_gguf.py:4790-4791). The name rename is handled in the tensor-name mapping
    // layer (dense.rs / moe.rs), NOT here; this function only transforms data + returns tensor.
    // (The GGUF name map in dense.rs:118 and moe.rs:162 emits "lin_attn_dt_bias" already;
    //  P4 name-mapping will apply the dt_proj.bias convention at that layer.)
    if name.ends_with(".A_log") || name.contains(".dt_bias") || name.contains(".dt_proj") {
        return transform_case4_linear_attn_scalar(tensor, nk, nv_per_k, elem_size);
    }

    // Case 5: conv1d — squeeze dim 1, then reorder V channel portion (py:5411-5418)
    //
    // Python reference (convert_hf_to_gguf.py:5411-5418):
    //   elif ".conv1d" in name:
    //     data = data_torch.squeeze()          # [k, 1, d] → [k, d]
    //     qk_channels = head_k_dim * num_k_heads * 2
    //     qk_part = data[:qk_channels]
    //     v_part  = data[qk_channels:]
    //     v_part  = _reorder_v_heads(v_part, 0, nk, nv_per_k, head_v_dim)
    //     data_torch = torch.cat([qk_part, v_part], dim=0)
    //
    // The squeeze is unconditional in Python (any singleton dims removed).
    // We enforce typed shape [k, 1, d] and error clearly if wrong — no silent reshape.
    if name.contains(".conv1d") {
        return transform_case5_conv1d(tensor, nk, nv_per_k, head_k_dim, head_v_dim, elem_size);
    }

    // Case 6: out_proj — reorder columns (py:5420-5422)
    if name.contains(".out_proj.") {
        return transform_case6_out_proj(tensor, nk, nv_per_k, head_v_dim, elem_size);
    }

    // No reorder applies — return unchanged.
    Ok(tensor)
}

// ---------------------------------------------------------------------------
// Individual case implementations
// ---------------------------------------------------------------------------

/// Case 1: `.in_proj_qkv.weight` — reorder V rows only.
///
/// Python (py:5384-5392):
/// ```python
/// q_dim = head_k_dim * num_k_heads
/// k_dim = head_k_dim * num_k_heads
/// q = data_torch[:q_dim]; k = data_torch[q_dim:q_dim+k_dim]; v = data_torch[q_dim+k_dim:]
/// v = self._reorder_v_heads(v, 0, num_k_heads, num_v_per_k, head_v_dim)
/// data_torch = torch.cat([q, k, v], dim=0)
/// ```
/// Rows are the first dimension of the weight matrix.  Each "row" here is an
/// entire `[hidden_size]` vector.  We split rows into Q/K/V slabs, reorder the
/// V slab's `num_k_heads × num_v_per_k` row groups along dim 0, and concat.
fn transform_case1_in_proj_qkv(
    mut tensor: TensorRef,
    nk: u32,
    nv_per_k: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    elem_size: usize,
) -> Result<TensorRef, ConvertError> {
    // Shape: [total_rows, cols] where total_rows = q_rows + k_rows + v_rows
    // q_rows = k_rows = head_k_dim * nk
    // v_rows = head_v_dim * nk * nv_per_k
    let q_rows = (head_k_dim * nk) as usize;
    let k_rows = q_rows;

    // cols = number of elements per row (last dim)
    let total_rows = tensor.shape.first().copied().unwrap_or(0);
    let cols = if tensor.shape.len() >= 2 { tensor.shape[1] } else { 1 };

    let q_bytes = q_rows * cols * elem_size;
    let k_bytes = k_rows * cols * elem_size;
    let v_bytes = tensor.data.len() - q_bytes - k_bytes;

    if tensor.data.len() < q_bytes + k_bytes {
        return Err(ConvertError::TensorLengthMismatch {
            name: tensor.name.clone(),
            expected: q_bytes + k_bytes,
            actual: tensor.data.len(),
        });
    }

    let _ = total_rows; // used in shape checks above via q_rows/k_rows
    let v_data = tensor.data[q_bytes + k_bytes..].to_vec();

    // Each "element" for reorder_v_heads here is cols*elem_size (a full row).
    // We treat each row as a single opaque blob by setting elem_size = cols*elem_size
    // and head_dim elements = 1 row = cols scalars.
    // But reorder_v_heads works on flat bytes with elem_size per scalar,
    // and head_dim is in scalars.  So head_dim = head_v_dim * cols.
    //
    // v layout: [nk, nv_per_k, head_v_dim * cols] scalars
    let reordered_v = reorder_v_heads(&v_data, elem_size, nk, nv_per_k, head_v_dim * cols as u32)?;

    let mut new_data = Vec::with_capacity(tensor.data.len());
    new_data.extend_from_slice(&tensor.data[..q_bytes + k_bytes]);
    new_data.extend_from_slice(&reordered_v);

    if new_data.len() != tensor.data.len() {
        return Err(ConvertError::ReorderInvariantViolated {
            name: tensor.name.clone(),
            reason: format!(
                "case 1: length changed {} -> {}",
                tensor.data.len(),
                new_data.len()
            ),
        });
    }

    // v_bytes unused after the reorder; suppress the warning.
    let _ = v_bytes;

    tensor.data = new_data;
    Ok(tensor)
}

/// Case 2: `.in_proj_z.weight` — reorder all rows.
///
/// Python (py:5394-5396):
/// ```python
/// data_torch = self._reorder_v_heads(data_torch, 0, num_k_heads, num_v_per_k, head_v_dim)
/// ```
/// The entire row-dimension is `[nk * nv_per_k, head_v_dim, cols]`.  Reorder
/// the outer `[nk, nv_per_k]` axes, keeping the inner head_v_dim*cols block together.
fn transform_case2_in_proj_z(
    mut tensor: TensorRef,
    nk: u32,
    nv_per_k: u32,
    head_v_dim: u32,
    elem_size: usize,
) -> Result<TensorRef, ConvertError> {
    let cols = if tensor.shape.len() >= 2 { tensor.shape[1] as u32 } else { 1 };
    let head_dim_total = head_v_dim * cols; // elements per "head" in the reorder

    let reordered = reorder_v_heads(&tensor.data, elem_size, nk, nv_per_k, head_dim_total)?;
    tensor.data = reordered;
    Ok(tensor)
}

/// Case 3: `.in_proj_a.weight` / `.in_proj_b.weight` — reorder all rows, head_dim=1.
///
/// Python (py:5398-5400):
/// ```python
/// data_torch = self._reorder_v_heads(data_torch, 0, num_k_heads, num_v_per_k, 1)
/// ```
/// Each head occupies exactly 1 row (times cols).  head_dim=1 means each "head block"
/// is `1 * cols` scalars.
fn transform_case3_in_proj_ab(
    mut tensor: TensorRef,
    nk: u32,
    nv_per_k: u32,
    elem_size: usize,
) -> Result<TensorRef, ConvertError> {
    let cols = if tensor.shape.len() >= 2 { tensor.shape[1] as u32 } else { 1 };
    // head_dim = 1 row = cols scalars
    let reordered = reorder_v_heads(&tensor.data, elem_size, nk, nv_per_k, cols)?;
    tensor.data = reordered;
    Ok(tensor)
}

/// Case 6: `.out_proj.weight` — reorder columns (input dimension).
///
/// Python (py:5420-5422):
/// ```python
/// data_torch = self._reorder_v_heads(data_torch, 1, num_k_heads, num_v_per_k, head_v_dim)
/// ```
/// `dim=1` means reorder along the column axis.  The weight matrix is
/// `[out_features, in_features]` where `in_features = nk * nv_per_k * head_v_dim`.
/// We operate row-by-row: for each output row, extract the column slice,
/// reorder it, and put it back.
fn transform_case6_out_proj(
    mut tensor: TensorRef,
    nk: u32,
    nv_per_k: u32,
    head_v_dim: u32,
    elem_size: usize,
) -> Result<TensorRef, ConvertError> {
    if tensor.shape.len() < 2 {
        return Err(ConvertError::ReorderInvariantViolated {
            name: tensor.name.clone(),
            reason: "out_proj expected 2-D tensor".to_string(),
        });
    }
    let rows = tensor.shape[0];
    let cols = tensor.shape[1];
    let row_bytes = cols * elem_size;

    let mut new_data = Vec::with_capacity(tensor.data.len());

    for r in 0..rows {
        let row_start = r * row_bytes;
        let row_slice = &tensor.data[row_start..row_start + row_bytes];
        // Reorder the column dimension: [nk, nv_per_k, head_v_dim] scalars
        let reordered_row = reorder_v_heads(row_slice, elem_size, nk, nv_per_k, head_v_dim)?;
        new_data.extend_from_slice(&reordered_row);
    }

    if new_data.len() != tensor.data.len() {
        return Err(ConvertError::ReorderInvariantViolated {
            name: tensor.name.clone(),
            reason: format!(
                "case 6: length changed {} -> {}",
                tensor.data.len(),
                new_data.len()
            ),
        });
    }

    tensor.data = new_data;
    Ok(tensor)
}

// ---------------------------------------------------------------------------
// P3 implementations — Case 4 and Case 5
// ---------------------------------------------------------------------------

/// Case 4: `.A_log` / `.dt_bias` / `.dt_proj` — elementwise negation (A_log only) + last-dim reorder.
///
/// # Python reference
/// `_LinearAttentionVReorderBase.modify_tensors` at `/opt/llama.cpp/convert_hf_to_gguf.py:5402-5409`
/// ```python
/// if data_torch.ndim == 1:
///     data_torch = _reorder_v_heads(data_torch.unsqueeze(-1), 0, nk, nv_per_k, 1).squeeze(-1)
/// else:
///     data_torch = _reorder_v_heads(data_torch, -1, nk, nv_per_k, 1)
/// ```
///
/// A_log negation (`-exp(input)`) is applied by `Qwen3NextModel.modify_tensors`
/// at `/opt/llama.cpp/convert_hf_to_gguf.py:4788-4789` (runs before the reorder in the
/// Python call-chain via `super().modify_tensors`).  We apply negation in the same pass here.
///
/// # Overflow handling
/// `exp(x)` on large positive f32 produces `+inf`; negated gives `-inf`.
/// This is the correct behaviour: it matches llama.cpp exactly, which makes no attempt to
/// clamp.  NaN inputs remain NaN; the conversion tool is not responsible for invalid weights.
/// We do NOT clamp, saturate, or substitute any sentinel value.
///
/// # dt_bias rename
/// The `.dt_bias` → `.dt_proj.bias` name rename (py:4790-4791) is the responsibility of
/// the tensor-name-to-GGUF mapping layer (`dense.rs` / `moe.rs`), not this transform.
/// This function only transforms data; the caller must map the name separately.
fn transform_case4_linear_attn_scalar(
    mut tensor: TensorRef,
    nk: u32,
    nv_per_k: u32,
    elem_size: usize,
) -> Result<TensorRef, ConvertError> {
    let is_a_log = tensor.name.ends_with(".A_log");

    // If A_log: negate elementwise — output[i] = -exp(input[i]) (py:4788-4789).
    // Overflow: exp(large x) → +inf, negated → -inf. This matches llama.cpp exactly.
    if is_a_log {
        if elem_size != 4 {
            return Err(ConvertError::ReorderInvariantViolated {
                name: tensor.name.clone(),
                reason: format!(
                    "A_log negation requires F32 data (elem_size=4), got elem_size={}",
                    elem_size
                ),
            });
        }
        let n_elems = tensor.data.len() / 4;
        for i in 0..n_elems {
            let off = i * 4;
            let val = f32::from_le_bytes([
                tensor.data[off],
                tensor.data[off + 1],
                tensor.data[off + 2],
                tensor.data[off + 3],
            ]);
            // -exp(val): overflow (val >> 0) gives -inf; underflow (val << 0) gives -1.0.
            // No clamping — match llama.cpp behaviour exactly.
            let negexp = -val.exp();
            let bytes = negexp.to_le_bytes();
            tensor.data[off..off + 4].copy_from_slice(&bytes);
        }
    }

    // V-head reorder along last dim with head_dim=1 (py:5402-5409).
    // 1D tensor: unsqueeze(-1) → [nk*nv_per_k, 1] → reorder row-dim → squeeze(-1).
    // The "unsqueeze then squeeze" pattern means we treat each scalar element as a
    // length-1 head, and reorder the [nk, nv_per_k] axes.
    //
    // For 1-D: reorder_v_heads(&data, elem_size, nk, nv_per_k, head_dim=1) directly,
    //   since head_dim=1 means one scalar per "block".
    // For 2-D (ndim > 1): reorder the last dimension; head_dim=1, so each scalar in
    //   the last dim is its own head.  We reorder per-row along the column axis.
    let ndim = tensor.shape.len();

    let reordered_data = if ndim <= 1 {
        // 1-D (or 0-D scalar, though that would be unusual)
        reorder_v_heads(&tensor.data, elem_size, nk, nv_per_k, 1)?
    } else {
        // Multi-dimensional: reorder the last dimension, row by row.
        // Each row's last-dim slice has nk*nv_per_k elements; head_dim=1.
        let last_dim = *tensor.shape.last().unwrap();
        let n_rows: usize = tensor.shape[..tensor.shape.len() - 1].iter().product();
        let row_bytes = last_dim * elem_size;

        let mut out = Vec::with_capacity(tensor.data.len());
        for r in 0..n_rows {
            let start = r * row_bytes;
            let row_slice = &tensor.data[start..start + row_bytes];
            let reordered_row = reorder_v_heads(row_slice, elem_size, nk, nv_per_k, 1)?;
            out.extend_from_slice(&reordered_row);
        }
        out
    };

    if reordered_data.len() != tensor.data.len() {
        return Err(ConvertError::ReorderInvariantViolated {
            name: tensor.name.clone(),
            reason: format!(
                "case 4: length changed {} -> {}",
                tensor.data.len(),
                reordered_data.len()
            ),
        });
    }

    tensor.data = reordered_data;
    Ok(tensor)
}

/// Case 5: `.conv1d` — squeeze singleton dim, then reorder only the V-channel portion.
///
/// # Python reference
/// `_LinearAttentionVReorderBase.modify_tensors` at `/opt/llama.cpp/convert_hf_to_gguf.py:5411-5418`
/// ```python
/// elif ".conv1d" in name:
///     data = data_torch.squeeze()          # [k, 1, d] → [k, d]
///     qk_channels = head_k_dim * num_k_heads * 2
///     qk_part = data[:qk_channels]
///     v_part  = data[qk_channels:]
///     v_part  = _reorder_v_heads(v_part, 0, nk, nv_per_k, head_v_dim)
///     data_torch = torch.cat([qk_part, v_part], dim=0)
/// ```
///
/// The squeeze is unconditional in Python (removes all singleton dims from any position).
/// We enforce the specific expected shape `[k, 1, d]` and return a typed error for any
/// other shape — no silent reshape.
///
/// # Error
/// Returns `ConvertError::ReorderInvariantViolated` if input shape is not `[k, 1, d]`.
fn transform_case5_conv1d(
    mut tensor: TensorRef,
    nk: u32,
    nv_per_k: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    elem_size: usize,
) -> Result<TensorRef, ConvertError> {
    // Enforce shape [k, 1, d].
    if tensor.shape.len() != 3 || tensor.shape[1] != 1 {
        return Err(ConvertError::ReorderInvariantViolated {
            name: tensor.name.clone(),
            reason: format!(
                "conv1d expected shape [k, 1, d] but got {:?}",
                tensor.shape
            ),
        });
    }

    let k = tensor.shape[0];
    let d = tensor.shape[2];

    // Squeeze: drop dim 1 (the singleton 1).  Data bytes are identical — just reshape.
    tensor.shape = vec![k, d];

    // Split at QK boundary (py:5414-5415).
    // qk_channels = head_k_dim * num_k_heads * 2  (Q channels + K channels)
    // v_channels  = total_channels - qk_channels
    let qk_channels = (head_k_dim * nk * 2) as usize;
    if k < qk_channels {
        return Err(ConvertError::TensorLengthMismatch {
            name: tensor.name.clone(),
            expected: qk_channels,
            actual: k,
        });
    }

    let qk_bytes = qk_channels * d * elem_size;
    let v_data = tensor.data[qk_bytes..].to_vec();

    // Reorder V channel portion: each "head" is head_v_dim channels × d scalars.
    // The V rows are [nk * nv_per_k, head_v_dim, d] → reorder the [nk, nv_per_k] axes.
    // In reorder_v_heads terms: data=[nk, nv_per_k, head_v_dim*d], elem_size=elem_size,
    // head_dim = head_v_dim * d.
    let reordered_v = reorder_v_heads(&v_data, elem_size, nk, nv_per_k, head_v_dim * d as u32)?;

    let mut new_data = Vec::with_capacity(tensor.data.len());
    new_data.extend_from_slice(&tensor.data[..qk_bytes]);
    new_data.extend_from_slice(&reordered_v);

    if new_data.len() != tensor.data.len() {
        return Err(ConvertError::ReorderInvariantViolated {
            name: tensor.name.clone(),
            reason: format!(
                "case 5: length changed {} -> {}",
                tensor.data.len(),
                new_data.len()
            ),
        });
    }

    tensor.data = new_data;
    Ok(tensor)
}

// ---------------------------------------------------------------------------
// P3 implementations — in_proj_qkvz split + RMS norm +1
// ---------------------------------------------------------------------------

/// Split `.linear_attn.in_proj_qkvz.weight` (fused Q/K/V/Z variant) into separate tensors.
///
/// # Python reference
/// `Qwen3NextModel.modify_tensors` at `/opt/llama.cpp/convert_hf_to_gguf.py:4797-4825`
///
/// The fused tensor has HF layout `[total_head_features, hidden_size]` where
/// `total_head_features = num_k_heads * (head_k_dim + head_k_dim + nv_per_k*head_v_dim + nv_per_k*head_v_dim)`.
///
/// Python logic (py:4811-4825):
/// ```python
/// # view as (hidden_size, num_k_heads, [q+k+v+z])
/// data = data.permute(1, 0)           # [hidden_size, total_head_features]
/// data = data.view(-1, num_k_heads, q+k+v+z_per_head)
/// q, k, v, z = torch.split(data, split_arg_list_qkvz, dim=-1)
/// q = q.view(hidden_size, -1).permute(1, 0)   # [q_rows, hidden_size]
/// k = k.view(hidden_size, -1).permute(1, 0)   # [k_rows, hidden_size]
/// v = v.view(hidden_size, -1).permute(1, 0)   # [v_rows, hidden_size]
/// z = z.view(hidden_size, -1).permute(1, 0)   # [z_rows, hidden_size]
/// yield ATTN_QKV = cat([q, k, v], dim=0)
/// yield ATTN_GATE = z
/// ```
///
/// Per-head partition sizes (py:4805-4809):
/// - q: `head_k_dim` elements
/// - k: `head_k_dim` elements
/// - v: `nv_per_k * head_v_dim` elements
/// - z: `nv_per_k * head_v_dim` elements
///
/// # Returns
/// Two tensors: `(qkv_tensor, z_tensor)`.
///   - `qkv_tensor`: Q, K, V rows concatenated — analogous to `in_proj_qkv.weight`
///   - `z_tensor`: Z rows — the gate projection
///
/// Both tensors inherit the original dtype and elem_size; shape is updated.
///
/// # V-head reorder
/// The V rows in the output `qkv_tensor` are in the same grouped layout as `in_proj_qkv.weight`.
/// The caller must subsequently pass `qkv_tensor` through `transform_linear_attn_tensor`
/// (case 1) to apply the V-head reorder, just as for a non-fused `in_proj_qkv.weight`.
///
/// # Error
/// Returns `Err` if the tensor name does not contain `in_proj_qkvz.weight`, if shape
/// is inconsistent, or if required hparams yield a zero partition.
pub fn transform_in_proj_qkvz(
    tensor: TensorRef,
    ctx: &Qwen35ConvertContext,
) -> Result<(TensorRef, TensorRef), ConvertError> {
    if !tensor.name.contains("in_proj_qkvz.weight") {
        return Err(ConvertError::ReorderInvariantViolated {
            name: tensor.name.clone(),
            reason: "transform_in_proj_qkvz called on non-qkvz tensor".to_string(),
        });
    }

    // Shape is [total_head_features, hidden_size] (row-major weight matrix).
    if tensor.shape.len() != 2 {
        return Err(ConvertError::ReorderInvariantViolated {
            name: tensor.name.clone(),
            reason: format!(
                "in_proj_qkvz.weight expected 2-D shape, got {:?}",
                tensor.shape
            ),
        });
    }

    let nk = ctx.linear_num_key_heads as usize;
    let nv_per_k = ctx.linear_num_v_per_k as usize;
    let head_k_dim = ctx.linear_key_head_dim as usize;
    let head_v_dim = ctx.linear_value_head_dim as usize;
    let elem_size = tensor.dtype.element_size();

    // Per-head partition sizes (py:4805-4809):
    let q_per_head = head_k_dim;
    let k_per_head = head_k_dim;
    let v_per_head = nv_per_k * head_v_dim;   // num_v_per_k * head_v_dim
    let z_per_head = nv_per_k * head_v_dim;
    let total_per_head = q_per_head + k_per_head + v_per_head + z_per_head;

    let total_rows = tensor.shape[0]; // total_head_features = nk * total_per_head
    let hidden_size = tensor.shape[1];

    // Validate that total_rows matches expected
    let expected_rows = nk * total_per_head;
    if total_rows != expected_rows {
        return Err(ConvertError::TensorLengthMismatch {
            name: tensor.name.clone(),
            expected: expected_rows * hidden_size * elem_size,
            actual: tensor.data.len(),
        });
    }

    // Python does: data.permute(1,0) → [hidden_size, nk * total_per_head]
    // then view(-1, nk, total_per_head) → [hidden_size, nk, total_per_head]
    // We replicate this in-memory.
    //
    // Source tensor layout: [nk * total_per_head, hidden_size] (row-major).
    //   Row r = head_idx * total_per_head + intra_head_offset.
    //   src[r, c] at byte offset: (r * hidden_size + c) * elem_size
    //
    // After permute(1,0): [hidden_size, nk * total_per_head]
    //   src_perm[c, r] — but we don't actually materialise this, we address it by swapping coords.
    //
    // After view(-1, nk, total_per_head): [hidden_size, nk, total_per_head]
    //   src_view[c, h, f] where f in [0, total_per_head), h in [0, nk), c in [0, hidden_size)
    //   src_perm[c, h * total_per_head + f] = src[(h * total_per_head + f), c]
    //   byte offset in original tensor: ((h * total_per_head + f) * hidden_size + c) * elem_size
    //
    // Split along last dim (f axis) into q[0..q_per_head], k[q..q+k_per_head], etc.
    //
    // Then: q.view(hidden_size, -1) = [hidden_size, nk * q_per_head]
    //       q.permute(1,0)          = [nk * q_per_head, hidden_size]
    // i.e. output row (h * q_per_head + fi, c) = src_view[c, h, fi]
    //                                           = original[(h*total_per_head + fi)*hidden_size + c]
    //
    // We build Q, K, V, Z output matrices directly.

    // Output row counts
    let q_rows = nk * q_per_head;
    let k_rows = nk * k_per_head;
    let v_rows = nk * v_per_head;
    let z_rows = nk * z_per_head;

    let mut q_data = vec![0u8; q_rows * hidden_size * elem_size];
    let mut k_data = vec![0u8; k_rows * hidden_size * elem_size];
    let mut v_data = vec![0u8; v_rows * hidden_size * elem_size];
    let mut z_data = vec![0u8; z_rows * hidden_size * elem_size];

    // Offsets within per-head block for each partition
    let q_off = 0;
    let k_off = q_per_head;
    let v_off = k_off + k_per_head;
    let z_off = v_off + v_per_head;

    for h in 0..nk {
        for c in 0..hidden_size {
            // Q: head h, features fi = 0..q_per_head
            for fi in 0..q_per_head {
                let src_row = h * total_per_head + q_off + fi;
                let src_byte = (src_row * hidden_size + c) * elem_size;
                let dst_row = h * q_per_head + fi;
                let dst_byte = (dst_row * hidden_size + c) * elem_size;
                q_data[dst_byte..dst_byte + elem_size]
                    .copy_from_slice(&tensor.data[src_byte..src_byte + elem_size]);
            }
            // K
            for fi in 0..k_per_head {
                let src_row = h * total_per_head + k_off + fi;
                let src_byte = (src_row * hidden_size + c) * elem_size;
                let dst_row = h * k_per_head + fi;
                let dst_byte = (dst_row * hidden_size + c) * elem_size;
                k_data[dst_byte..dst_byte + elem_size]
                    .copy_from_slice(&tensor.data[src_byte..src_byte + elem_size]);
            }
            // V
            for fi in 0..v_per_head {
                let src_row = h * total_per_head + v_off + fi;
                let src_byte = (src_row * hidden_size + c) * elem_size;
                let dst_row = h * v_per_head + fi;
                let dst_byte = (dst_row * hidden_size + c) * elem_size;
                v_data[dst_byte..dst_byte + elem_size]
                    .copy_from_slice(&tensor.data[src_byte..src_byte + elem_size]);
            }
            // Z
            for fi in 0..z_per_head {
                let src_row = h * total_per_head + z_off + fi;
                let src_byte = (src_row * hidden_size + c) * elem_size;
                let dst_row = h * z_per_head + fi;
                let dst_byte = (dst_row * hidden_size + c) * elem_size;
                z_data[dst_byte..dst_byte + elem_size]
                    .copy_from_slice(&tensor.data[src_byte..src_byte + elem_size]);
            }
        }
    }

    // Concatenate Q, K, V → qkv (py:4822: cat([q,k,v], dim=0) then permute(1,0))
    // At this point q/k/v_data are already in [rows, hidden_size] layout — just concatenate.
    let mut qkv_data = Vec::with_capacity(q_data.len() + k_data.len() + v_data.len());
    qkv_data.extend_from_slice(&q_data);
    qkv_data.extend_from_slice(&k_data);
    qkv_data.extend_from_slice(&v_data);

    let qkv_rows = q_rows + k_rows + v_rows;
    let base_name = tensor.name.rsplitn(2, '.').last().unwrap_or(&tensor.name);
    // The qkv tensor name: replace "in_proj_qkvz.weight" with "in_proj_qkv.weight"
    let qkv_name = tensor.name.replace("in_proj_qkvz.weight", "in_proj_qkv.weight");
    let z_name   = tensor.name.replace("in_proj_qkvz.weight", "in_proj_z.weight");

    // Suppress unused variable warning
    let _ = base_name;

    let qkv_tensor = TensorRef {
        name: qkv_name,
        shape: vec![qkv_rows, hidden_size],
        dtype: tensor.dtype,
        data: qkv_data,
    };

    let z_tensor = TensorRef {
        name: z_name,
        shape: vec![z_rows, hidden_size],
        dtype: tensor.dtype,
        data: z_data,
    };

    Ok((qkv_tensor, z_tensor))
}

/// Apply the RMS norm +1 weight bias (Qwen3.5 `gamma + 1` convention).
///
/// # Audit outcome: Qwen3.5 DOES use `gamma + 1`
///
/// Citation: `/opt/llama.cpp/convert_hf_to_gguf.py:4794-4795`
/// in `Qwen3NextModel.modify_tensors` (the base class for all Qwen3.5 variants):
/// ```python
/// elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
///     data_torch = data_torch + 1
/// ```
/// `Qwen3_5TextModel` and `Qwen3_5MoeTextModel` both inherit from
/// `_LinearAttentionVReorderBase` which inherits from `Qwen3NextModel`
/// (py:5259, 5427-5434). The `modify_tensors` call chain reaches
/// `Qwen3NextModel.modify_tensors` via `super().modify_tensors` at py:5424.
///
/// The `build_norm` implementation in `llama-graph.cpp:1028-1055` calls
/// `ggml_mul(ctx0, cur, mw)` — plain multiply, no +1.  The +1 is therefore
/// baked into the stored weight at conversion time, not applied at inference time.
///
/// # Exclusion
/// `linear_attn.norm.weight` is explicitly excluded (py:4794: `not name.endswith(...)`).
/// This norm is the post-linear-attention sub-layer norm which is stored and applied
/// without the +1 bias.
///
/// # Scope
/// Applies to both Dense and MoE variants identically (same base class).
/// Applies to: `input_layernorm.weight`, `post_attention_layernorm.weight`,
/// `attn_q_norm.weight`, `attn_k_norm.weight`, `output_norm.weight`, etc.
/// Does NOT apply to: `linear_attn.norm.weight`.
///
/// # Implementation
/// For F32 tensors: add 1.0_f32 to each element in-place.
/// For BF16 tensors: decode, add 1.0, re-encode.
/// For F16 tensors: decode, add 1.0, re-encode.
/// Other dtypes (norm weights should always be a float type): return typed error.
/// Apply `apply_rms_norm_plus_one` to every qualifying norm tensor in
/// a pre-quantization `TensorMap`. Arch-gated: no-op for non-Qwen3.5
/// arches (Gemma4, LLaMA, etc. do not use the `gamma + 1` convention).
///
/// Qualifying: tensor name ends with `norm.weight` AND does NOT end
/// with `linear_attn.norm.weight` — matches convert_hf_to_gguf.py:4794-4795
/// exactly.
///
/// # Silent wire-up gap (2026-04-24)
///
/// P3 shipped `apply_rms_norm_plus_one` in commit `73a96e4` but never
/// wired it into the convert pipeline. Before this function, the
/// per-tensor transform only ran inside its own unit tests; real
/// qwen35/qwen35moe convert output shipped RMS norm weights WITHOUT
/// the +1 bias. llama.cpp's forward pass assumes `gamma + 1` baked
/// in at convert time (`build_norm` at `llama-graph.cpp:1028-1055`
/// does plain multiply, no +1), so the missing bias produces ~1.0x
/// norm multiplier shift through every layer — compounding silent
/// logit skew.
///
/// This walker is the fix. Called from `src/main.rs` Phase 1.5
/// alongside the MoE expert merge.
/// Apply the full Qwen3.5 linear-attention transform suite to every
/// qualifying tensor in a pre-quantization `TensorMap`.
///
/// Matches `convert_hf_to_gguf.py` `_LinearAttentionVReorderBase.modify_tensors`
/// (py:5367-5424) + `Qwen3NextModel.modify_tensors` preprocessing
/// (py:4786-4830): A_log negation, conv1d squeeze, in_proj_qkvz split,
/// then the 6-case V-head reorder for the remaining linear-attn tensors.
///
/// # Silent wire-up gap (2026-04-24)
///
/// P2/P3 shipped `transform_linear_attn_tensor` + `transform_in_proj_qkvz`
/// in commits `1a849e1` + `73a96e4` but neither was ever called from the
/// convert pipeline. Result: every Qwen3.5 GGUF produced before this
/// fix shipped HF-grouped V-head layout (not the ggml-tiled layout the
/// loader expects) — THE named silent-corruption R2 failure mode ADR-012
/// called out, producing "plausible-looking nonsense" at inference.
///
/// This walker is the wire-up. Called from `src/main.rs` Phase 1.7
/// alongside the MoE merge + RMS norm +1 bias.
///
/// # Arch gate
///
/// No-op for non-Qwen3.5 arches. Linear-attention transforms are specific
/// to the Qwen3.5 family's Gated DeltaNet; Gemma4 and LLaMA have no
/// linear_attn tensors.
///
/// # Short-circuit when num_k_heads == num_v_heads
///
/// Per py:5379, the V-head reorder only fires when
/// `num_k_heads > 0 and num_v_heads > 0 and num_k_heads != num_v_heads`.
/// `transform_linear_attn_tensor` short-circuits itself when
/// `ctx.linear_num_v_per_k == 1` — we rely on that short-circuit rather
/// than duplicating the check here.
pub fn apply_qwen35_linear_attn_transforms_in_tensor_map(
    tensor_map: &mut crate::ir::TensorMap,
    metadata: &crate::ir::ModelMetadata,
) -> Result<(), ConvertError> {
    let arch_str = metadata.architecture.as_str();
    let model_type = metadata.model_type.as_str();
    let is_qwen35 = arch_str == "Qwen3_5ForCausalLM"
        || arch_str == "Qwen3_5ForConditionalGeneration"
        || arch_str == "Qwen3_5MoeForCausalLM"
        || arch_str == "Qwen3_5MoeForConditionalGeneration"
        || model_type == "qwen3_5"
        || model_type == "qwen3_5_moe_text";
    if !is_qwen35 {
        return Ok(());
    }

    let ctx = Qwen35ConvertContext::from_metadata(metadata)?;

    // Step 1: handle in_proj_qkvz tensors first (split into qkv + z),
    // then transform the split qkv through case 1 (V-head reorder).
    // The qkvz form is used by Qwen3Next/3.5 fused projections.
    let qkvz_keys: Vec<String> = tensor_map
        .tensors
        .keys()
        .filter(|n| n.contains("in_proj_qkvz.weight"))
        .cloned()
        .collect();

    for key in qkvz_keys {
        let Some(qkvz_tensor) = tensor_map.tensors.remove(&key) else {
            continue;
        };
        let (qkv_tensor, z_tensor) = transform_in_proj_qkvz(qkvz_tensor, &ctx)?;
        // Apply V-head reorder to the qkv tensor (case 1).
        let qkv_reordered = transform_linear_attn_tensor(qkv_tensor, &ctx)?;
        // z_tensor follows the same V-head reorder path as in_proj_z (case 2).
        let z_reordered = transform_linear_attn_tensor(z_tensor, &ctx)?;
        tensor_map.tensors.insert(qkv_reordered.name.clone(), qkv_reordered);
        tensor_map.tensors.insert(z_reordered.name.clone(), z_reordered);
    }

    // Step 2: walk all linear-attn tensors that didn't come from qkvz
    // split (independent in_proj_qkv, in_proj_z, in_proj_a/b, A_log,
    // dt_bias, dt_proj, conv1d, out_proj). transform_linear_attn_tensor
    // dispatches to the right case or no-ops.
    let linear_attn_keys: Vec<String> = tensor_map
        .tensors
        .keys()
        .filter(|n| n.contains("linear_attn."))
        .cloned()
        .collect();

    for key in linear_attn_keys {
        let Some(tensor) = tensor_map.tensors.remove(&key) else {
            continue;
        };
        let transformed = transform_linear_attn_tensor(tensor, &ctx)?;
        tensor_map
            .tensors
            .insert(transformed.name.clone(), transformed);
    }

    Ok(())
}

pub fn apply_rms_norm_plus_one_in_tensor_map(
    tensor_map: &mut crate::ir::TensorMap,
    metadata: &crate::ir::ModelMetadata,
) -> Result<(), ConvertError> {
    // Arch gate — Qwen3.5 family only. convert_hf_to_gguf.py:4794 is
    // scoped to Qwen3NextModel which Qwen3_5TextModel + Qwen3_5MoeTextModel
    // both inherit from (py:5259, 5427-5434).
    let arch_str = metadata.architecture.as_str();
    let model_type = metadata.model_type.as_str();
    let is_qwen35 = arch_str == "Qwen3_5ForCausalLM"
        || arch_str == "Qwen3_5ForConditionalGeneration"
        || arch_str == "Qwen3_5MoeForCausalLM"
        || arch_str == "Qwen3_5MoeForConditionalGeneration"
        || model_type == "qwen3_5"
        || model_type == "qwen3_5_moe_text";
    if !is_qwen35 {
        return Ok(());
    }

    // Collect the names first (cannot mutate the map while iterating).
    let qualifying: Vec<String> = tensor_map
        .tensors
        .keys()
        .filter(|n| {
            n.ends_with("norm.weight")
                && !n.ends_with("linear_attn.norm.weight")
        })
        .cloned()
        .collect();

    for name in qualifying {
        let Some(tensor) = tensor_map.tensors.remove(&name) else {
            continue;
        };
        let transformed = apply_rms_norm_plus_one(tensor)?;
        tensor_map.tensors.insert(transformed.name.clone(), transformed);
    }

    Ok(())
}

pub fn apply_rms_norm_plus_one(
    mut tensor: TensorRef,
) -> Result<TensorRef, ConvertError> {
    // Exclusion: linear_attn.norm.weight does NOT get +1 (py:4794).
    if tensor.name.ends_with("linear_attn.norm.weight") {
        return Ok(tensor);
    }

    match tensor.dtype {
        DType::F32 => {
            let n = tensor.data.len() / 4;
            for i in 0..n {
                let off = i * 4;
                let val = f32::from_le_bytes([
                    tensor.data[off],
                    tensor.data[off + 1],
                    tensor.data[off + 2],
                    tensor.data[off + 3],
                ]);
                let result = (val + 1.0_f32).to_le_bytes();
                tensor.data[off..off + 4].copy_from_slice(&result);
            }
        }
        DType::BF16 => {
            let n = tensor.data.len() / 2;
            for i in 0..n {
                let off = i * 2;
                let bits = u16::from_le_bytes([tensor.data[off], tensor.data[off + 1]]);
                let val = half::bf16::from_bits(bits).to_f32();
                let result = half::bf16::from_f32(val + 1.0_f32).to_bits().to_le_bytes();
                tensor.data[off..off + 2].copy_from_slice(&result);
            }
        }
        DType::F16 => {
            let n = tensor.data.len() / 2;
            for i in 0..n {
                let off = i * 2;
                let bits = u16::from_le_bytes([tensor.data[off], tensor.data[off + 1]]);
                let val = half::f16::from_bits(bits).to_f32();
                let result = half::f16::from_f32(val + 1.0_f32).to_bits().to_le_bytes();
                tensor.data[off..off + 2].copy_from_slice(&result);
            }
        }
        other => {
            return Err(ConvertError::ReorderInvariantViolated {
                name: tensor.name.clone(),
                reason: format!("rms_norm_plus_one: unexpected dtype {other}"),
            });
        }
    }

    Ok(tensor)
}

/// Inverse of `reorder_v_heads`: convert ggml tiled layout back to HF grouped layout.
///
/// # Proof that this is the correct inverse
///
/// `reorder_v_heads(data, e, nk, nv, hd)` maps:
///   src index `k * (nv * hd) + v * hd + d`  →  dst index `v * (nk * hd) + k * hd + d`
///
/// This is a transpose of the `[nk, nv]` axes when the slice is viewed as
/// `[nk, nv, hd]`.  The inverse transpose is to swap `nk` and `nv` as arguments:
///   `reorder_v_heads(data, e, nv, nk, hd)` maps:
///   src index `v * (nk * hd) + k * hd + d`  →  dst index `k * (nv * hd) + v * hd + d`
/// which is exactly the forward map undone.
///
/// # When is reorder_v_heads self-inverse?
///
/// Only when `nk == nv` (the permutation is symmetric).  In that case
/// `reorder_v_heads(reorder_v_heads(x, e, n, n, hd), e, n, n, hd) == x`.
///
/// For all other cases (e.g. Qwen3.5-MoE with nk=16, nv=32 → nv_per_k=2),
/// the inverse requires swapping the params.
pub fn reorder_v_heads_inverse(
    data: &[u8],
    elem_size: usize,
    num_k_heads: u32,
    num_v_per_k: u32,
    head_dim: u32,
) -> Result<Vec<u8>, ConvertError> {
    // Inverse: swap nk and nv_per_k.
    reorder_v_heads(data, elem_size, num_v_per_k, num_k_heads, head_dim)
}

// ---------------------------------------------------------------------------
// DType helper — available for dense.rs / moe.rs
// ---------------------------------------------------------------------------

/// Map a `DType` to the number of bytes per element.
///
/// Thin re-export so dense.rs / moe.rs can call it without importing ir directly.
#[allow(dead_code)]
pub(crate) fn dtype_elem_size(dtype: DType) -> usize {
    dtype.element_size()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::DType;

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /// Build a `TensorRef` with u32 marker values 0,1,2,… (each 4-byte LE).
    fn make_u32_tensor(name: &str, shape: Vec<usize>) -> TensorRef {
        let n: usize = shape.iter().product();
        let mut data = Vec::with_capacity(n * 4);
        for i in 0u32..(n as u32) {
            data.extend_from_slice(&i.to_le_bytes());
        }
        TensorRef { name: name.to_string(), shape, dtype: DType::F32, data }
    }

    /// Read element `i` of a flat u32 buffer.
    fn read_u32(data: &[u8], i: usize) -> u32 {
        let off = i * 4;
        u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
    }

    /// Build a minimal `Qwen35ConvertContext` with apex-like params for dispatcher tests.
    ///
    /// Apex config: 40 layers, full_attention_interval=4,
    ///   linear_num_key_heads=16, linear_num_value_heads=32 (nv_per_k=2),
    ///   linear_key_head_dim=128, linear_value_head_dim=128.
    fn apex_ctx() -> Qwen35ConvertContext {
        // Build a layer_types vec matching apex: full_attention at indices 3,7,…,39
        // (every 4th, 1-indexed, so 0-indexed: 3, 7, 11, …, 39)
        let layer_types: Vec<String> = (0..40_usize)
            .map(|i| {
                if (i + 1) % 4 == 0 {
                    "full_attention".to_string()
                } else {
                    "linear_attention".to_string()
                }
            })
            .collect();

        Qwen35ConvertContext {
            arch: Qwen35Arch::Moe,
            layer_types,
            num_layers: 40,
            num_attention_heads: 40,
            num_kv_heads: 8,
            head_dim: 256,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 128,
            linear_num_key_heads: 16,
            linear_value_head_dim: 128,
            linear_num_value_heads: 32,
            linear_num_v_per_k: 2,
            moe_intermediate_size: Some(2048),
            shared_expert_intermediate_size: Some(512),
            num_experts: Some(256),
            top_k_experts: Some(8),
            intermediate_size: None,
        }
    }

    // =========================================================================
    // Specification-driven test
    // =========================================================================

    /// Spec-driven test: hand-author the complete expected permutation map.
    ///
    /// Parameters: num_k_heads=2, num_v_per_k=2, head_dim=4, elem_size=4 (F32).
    ///
    /// # Derivation of EXPECTED_PERM
    ///
    /// Source (grouped, HF order) — 16 elements (2*2*4):
    ///   [G0_v0_0, G0_v0_1, G0_v0_2, G0_v0_3,  <- k=0, v=0, d=0..3
    ///    G0_v1_0, G0_v1_1, G0_v1_2, G0_v1_3,  <- k=0, v=1, d=0..3
    ///    G1_v0_0, G1_v0_1, G1_v0_2, G1_v0_3,  <- k=1, v=0, d=0..3
    ///    G1_v1_0, G1_v1_1, G1_v1_2, G1_v1_3]  <- k=1, v=1, d=0..3
    ///
    /// Source linear element index: k * (nv*hd) + v * hd + d = k*8 + v*4 + d
    ///   G0_v0: [0,1,2,3]; G0_v1: [4,5,6,7]; G1_v0: [8,9,10,11]; G1_v1: [12,13,14,15]
    ///
    /// Target (tiled, ggml order): [nv, nk, hd] = [2, 2, 4]
    ///   Outer index = v, inner = k, then d.
    ///   Target position v * (nk*hd) + k * hd + d = v*8 + k*4 + d
    ///
    /// Mapping dst_pos -> src_value:
    ///   dst[0]=v=0,k=0,d=0 <- src[k=0,v=0,d=0] = 0
    ///   dst[1]=v=0,k=0,d=1 <- src[k=0,v=0,d=1] = 1
    ///   dst[2]=v=0,k=0,d=2 <- src[k=0,v=0,d=2] = 2
    ///   dst[3]=v=0,k=0,d=3 <- src[k=0,v=0,d=3] = 3
    ///   dst[4]=v=0,k=1,d=0 <- src[k=1,v=0,d=0] = 8
    ///   dst[5]=v=0,k=1,d=1 <- src[k=1,v=0,d=1] = 9
    ///   dst[6]=v=0,k=1,d=2 <- src[k=1,v=0,d=2] = 10
    ///   dst[7]=v=0,k=1,d=3 <- src[k=1,v=0,d=3] = 11
    ///   dst[8]=v=1,k=0,d=0 <- src[k=0,v=1,d=0] = 4
    ///   dst[9]=v=1,k=0,d=1 <- src[k=0,v=1,d=1] = 5
    ///   dst[10]=v=1,k=0,d=2 <- src[k=0,v=1,d=2] = 6
    ///   dst[11]=v=1,k=0,d=3 <- src[k=0,v=1,d=3] = 7
    ///   dst[12]=v=1,k=1,d=0 <- src[k=1,v=1,d=0] = 12
    ///   dst[13]=v=1,k=1,d=1 <- src[k=1,v=1,d=1] = 13
    ///   dst[14]=v=1,k=1,d=2 <- src[k=1,v=1,d=2] = 14
    ///   dst[15]=v=1,k=1,d=3 <- src[k=1,v=1,d=3] = 15
    ///
    /// So: EXPECTED_PERM = [0,1,2,3, 8,9,10,11, 4,5,6,7, 12,13,14,15]
    #[test]
    fn spec_driven_permutation_map() {
        const NP: usize = 16;
        // The expected values at each output position (derived above).
        const EXPECTED_PERM: [u32; NP] = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15];

        // Build source: element i has value i.
        let mut src_bytes = Vec::with_capacity(NP * 4);
        for i in 0u32..NP as u32 {
            src_bytes.extend_from_slice(&i.to_le_bytes());
        }

        let result = reorder_v_heads(&src_bytes, 4, 2, 2, 4).unwrap();
        assert_eq!(result.len(), NP * 4);

        for (pos, &expected) in EXPECTED_PERM.iter().enumerate() {
            let got = read_u32(&result, pos);
            assert_eq!(
                got, expected,
                "spec-driven: dst[{}] = {} but expected {} \
                 (ggml tiled: outer=v, inner=k; HF grouped: outer=k, inner=v)",
                pos, got, expected
            );
        }
    }

    // =========================================================================
    // Round-trip tests
    // =========================================================================

    /// Round-trip via explicit inverse: forward then inverse = identity.
    ///
    /// # Why reorder_v_heads is NOT generally self-inverse
    ///
    /// `reorder_v_heads(data, e, nk, nv, hd)` transposes `[nk, nv]` axes.
    /// Applying it a second time with the SAME nk/nv re-interprets the tiled
    /// layout `[nv, nk, hd]` as if it were grouped `[nk, nv, hd]`, which only
    /// produces the identity when `nk == nv` (symmetric permutation).
    ///
    /// For asymmetric cases (e.g. nk=2, nv=3) the inverse must swap the params:
    ///   `reorder_v_heads_inverse(data, e, nk, nv, hd)` ≡
    ///   `reorder_v_heads(data, e, nv, nk, hd)`
    ///
    /// This is `reorder_v_heads_inverse`.  See its doc comment for the proof.
    ///
    /// This test verifies: forward(inverse(x)) == x AND inverse(forward(x)) == x.
    #[test]
    fn round_trip_via_explicit_inverse() {
        let nk = 2u32;
        let nv = 3u32;  // asymmetric: nk != nv
        let hd = 4u32;
        let n = (nk * nv * hd) as usize;

        let mut original = Vec::with_capacity(n * 4);
        for i in 0u32..n as u32 {
            original.extend_from_slice(&i.to_le_bytes());
        }

        // forward then inverse
        let fwd = reorder_v_heads(&original, 4, nk, nv, hd).unwrap();
        let inv = reorder_v_heads_inverse(&fwd, 4, nk, nv, hd).unwrap();
        assert_eq!(inv, original, "forward then inverse != identity");

        // inverse then forward
        let inv2 = reorder_v_heads_inverse(&original, 4, nk, nv, hd).unwrap();
        let fwd2 = reorder_v_heads(&inv2, 4, nk, nv, hd).unwrap();
        assert_eq!(fwd2, original, "inverse then forward != identity");
    }

    /// Self-inverse property: when nk == nv, applying reorder_v_heads twice = identity.
    ///
    /// For nk == nv the permutation is symmetric (the axes have the same size),
    /// so swapping them twice returns to the original.
    #[test]
    fn round_trip_self_inverse_symmetric() {
        let n = 3u32;  // nk == nv → self-inverse
        let hd = 4u32;
        let total = (n * n * hd) as usize;

        let mut original = Vec::with_capacity(total * 4);
        for i in 0u32..total as u32 {
            original.extend_from_slice(&i.to_le_bytes());
        }

        let pass1 = reorder_v_heads(&original, 4, n, n, hd).unwrap();
        let pass2 = reorder_v_heads(&pass1, 4, n, n, hd).unwrap();
        assert_eq!(pass2, original, "reorder_v_heads(nk==nv) is not self-inverse");
    }

    // =========================================================================
    // Per-case unit tests
    // =========================================================================

    /// Case 2: in_proj_z — full row reorder.
    ///
    /// Parameters: nk=2, nv_per_k=2, head_v_dim=2, cols=3.
    /// Tensor shape [8, 3]: 8 rows, each row is 3 F32 elements.
    ///
    /// Marker: row i has values [i*3, i*3+1, i*3+2].
    ///
    /// Expected reordering of rows (nk=2, nv_per_k=2, head_dim=2*3=6 scalars):
    ///   src row groups: k=0 → rows [0,1,2,3] (head_v_dim=2 rows per group)
    ///                   k=1 → rows [4,5,6,7]
    ///   After reorder (swap k↔v axes): each v_group spans all k:
    ///   Actually head_dim_total = head_v_dim * cols = 2*3 = 6 elements.
    ///   So nk*nv_per_k*head_dim_total = 2*2*6 = 24 elements = 8 rows * 3 cols. Correct.
    ///   src[k=0,v=0] = rows 0..1 (elements 0..5)
    ///   src[k=0,v=1] = rows 2..3 (elements 6..11)
    ///   src[k=1,v=0] = rows 4..5 (elements 12..17)
    ///   src[k=1,v=1] = rows 6..7 (elements 18..23)
    ///   dst[v=0,k=0] = elements 0..5  -> src[k=0,v=0] = 0..5
    ///   dst[v=0,k=1] = elements 6..11 -> src[k=1,v=0] = 12..17
    ///   dst[v=1,k=0] = elements 12..17 -> src[k=0,v=1] = 6..11
    ///   dst[v=1,k=1] = elements 18..23 -> src[k=1,v=1] = 18..23
    #[test]
    fn case2_in_proj_z_reorder() {
        let nk = 2u32;
        let nv_per_k = 2u32;
        let head_v_dim = 2u32;
        let cols = 3u32;
        let head_dim_total = head_v_dim * cols; // 6 scalars
        let total = (nk * nv_per_k * head_dim_total) as usize; // 24

        let mut src = Vec::with_capacity(total * 4);
        for i in 0u32..total as u32 {
            src.extend_from_slice(&i.to_le_bytes());
        }

        let out = reorder_v_heads(&src, 4, nk, nv_per_k, head_dim_total).unwrap();

        // dst[0..5] = src[0..5] (k=0,v=0 block)
        for i in 0..6 {
            assert_eq!(read_u32(&out, i), i as u32, "case2 v=0,k=0 pos {i}");
        }
        // dst[6..11] = src[12..17] (v=0,k=1 block)
        for i in 0..6 {
            assert_eq!(read_u32(&out, 6 + i), 12 + i as u32, "case2 v=0,k=1 pos {i}");
        }
        // dst[12..17] = src[6..11] (v=1,k=0 block)
        for i in 0..6 {
            assert_eq!(read_u32(&out, 12 + i), 6 + i as u32, "case2 v=1,k=0 pos {i}");
        }
        // dst[18..23] = src[18..23] (v=1,k=1 block)
        for i in 0..6 {
            assert_eq!(read_u32(&out, 18 + i), 18 + i as u32, "case2 v=1,k=1 pos {i}");
        }
    }

    /// Case 3: in_proj_a/b — row reorder with head_dim=1 per head (times cols).
    ///
    /// nk=2, nv_per_k=2, cols=5 → head_dim_for_reorder = 1*5 = 5 scalars.
    /// total = 2*2*5 = 20 scalars.
    /// src: element i = value i.
    /// Expected: same as spec-driven with head_dim=5.
    ///   dst[v=0,k=0] = src[k=0,v=0] = [0..4]
    ///   dst[v=0,k=1] = src[k=1,v=0] = [10..14]
    ///   dst[v=1,k=0] = src[k=0,v=1] = [5..9]
    ///   dst[v=1,k=1] = src[k=1,v=1] = [15..19]
    #[test]
    fn case3_in_proj_ab_reorder() {
        let nk = 2u32;
        let nv_per_k = 2u32;
        let cols = 5u32; // head_dim = 1 * cols
        let total = (nk * nv_per_k * cols) as usize;

        let mut src = Vec::with_capacity(total * 4);
        for i in 0u32..total as u32 {
            src.extend_from_slice(&i.to_le_bytes());
        }

        let out = reorder_v_heads(&src, 4, nk, nv_per_k, cols).unwrap();

        // v=0, k=0 → src[0..4]
        for i in 0..5 { assert_eq!(read_u32(&out, i), i as u32, "case3 v0k0 {i}"); }
        // v=0, k=1 → src[10..14]
        for i in 0..5 { assert_eq!(read_u32(&out, 5 + i), 10 + i as u32, "case3 v0k1 {i}"); }
        // v=1, k=0 → src[5..9]
        for i in 0..5 { assert_eq!(read_u32(&out, 10 + i), 5 + i as u32, "case3 v1k0 {i}"); }
        // v=1, k=1 → src[15..19]
        for i in 0..5 { assert_eq!(read_u32(&out, 15 + i), 15 + i as u32, "case3 v1k1 {i}"); }
    }

    /// Case 1: in_proj_qkv — only V rows reordered; Q and K pass through.
    ///
    /// nk=2, nv_per_k=2, head_k_dim=2, head_v_dim=2, cols=3.
    /// q_rows = k_rows = 2*2 = 4.
    /// v_rows = 2*2*2 = 8.
    /// total rows = 16, shape=[16,3].
    #[test]
    fn case1_in_proj_qkv_v_reorder() {
        let nk = 2u32;
        let nv_per_k = 2u32;
        let head_k_dim = 2u32;
        let head_v_dim = 2u32;
        let cols = 3u32;
        let q_rows = (head_k_dim * nk) as usize; // 4
        let k_rows = q_rows;
        let v_rows = (head_v_dim * nk * nv_per_k) as usize; // 8
        let total_rows = q_rows + k_rows + v_rows; // 16
        let total_elems = total_rows * cols as usize;

        let tensor = make_u32_tensor(
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            vec![total_rows, cols as usize],
        );
        assert_eq!(tensor.data.len(), total_elems * 4);

        let ctx = Qwen35ConvertContext {
            arch: Qwen35Arch::Moe,
            layer_types: vec!["linear_attention".to_string()],
            num_layers: 1,
            num_attention_heads: 4,
            num_kv_heads: 4,
            head_dim: 4,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: head_k_dim,
            linear_num_key_heads: nk,
            linear_value_head_dim: head_v_dim,
            linear_num_value_heads: nk * nv_per_k,
            linear_num_v_per_k: nv_per_k,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: None,
        };

        let result = transform_linear_attn_tensor(tensor.clone(), &ctx).unwrap();

        // Q rows unchanged: elements 0..q_rows*cols
        let q_elems = q_rows * cols as usize;
        for i in 0..q_elems {
            assert_eq!(read_u32(&result.data, i), i as u32, "Q elem {i} changed");
        }

        // K rows unchanged
        let k_start = q_elems;
        let k_elems = k_rows * cols as usize;
        for i in 0..k_elems {
            assert_eq!(
                read_u32(&result.data, k_start + i),
                (k_start + i) as u32,
                "K elem {i} changed"
            );
        }

        // V rows reordered: values should differ from original in non-identity positions
        // Just verify length preserved and at least one value moved.
        assert_eq!(result.data.len(), tensor.data.len(), "case1: length changed");
        let v_start_elem = (q_rows + k_rows) * cols as usize;
        // Element at v_start should be from k=0,v=0,d=0 (same as original)
        assert_eq!(
            read_u32(&result.data, v_start_elem),
            v_start_elem as u32,
            "V first element should be same (k=0,v=0 maps to dst v=0,k=0)"
        );
    }

    /// Case 6: out_proj — column reorder.
    ///
    /// Shape [3, 16]: 3 output rows, 16 input cols.
    /// nk=2, nv_per_k=2, head_v_dim=4 → 2*2*4=16 cols. Correct.
    ///
    /// Each row is independently reordered.  For row r with marker values
    /// [r*16, r*16+1, …, r*16+15], the column order changes per spec-driven map:
    ///   dst_col positions: [0,1,2,3, 8,9,10,11, 4,5,6,7, 12,13,14,15]
    #[test]
    fn case6_out_proj_column_reorder() {
        let nk = 2u32;
        let nv_per_k = 2u32;
        let head_v_dim = 4u32;
        let rows = 3usize;
        let cols = (nk * nv_per_k * head_v_dim) as usize; // 16

        let tensor = make_u32_tensor(
            "model.layers.0.linear_attn.out_proj.weight",
            vec![rows, cols],
        );

        let ctx = Qwen35ConvertContext {
            arch: Qwen35Arch::Dense,
            layer_types: vec!["linear_attention".to_string()],
            num_layers: 1,
            num_attention_heads: 4,
            num_kv_heads: 4,
            head_dim: 4,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 4,
            linear_num_key_heads: nk,
            linear_value_head_dim: head_v_dim,
            linear_num_value_heads: nk * nv_per_k,
            linear_num_v_per_k: nv_per_k,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: None,
        };

        let result = transform_linear_attn_tensor(tensor.clone(), &ctx).unwrap();
        assert_eq!(result.data.len(), tensor.data.len(), "case6: length changed");

        // For each row, check the column permutation matches the spec-driven map.
        // nk=2, nv_per_k=2, head_v_dim=4 matches spec-driven test params.
        const PERM: [usize; 16] = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15];
        for r in 0..rows {
            let row_base = r * cols;
            for (dst_col, &src_col) in PERM.iter().enumerate() {
                let expected = (row_base + src_col) as u32;
                let got = read_u32(&result.data, row_base + dst_col);
                assert_eq!(got, expected, "case6 row={r} dst_col={dst_col}");
            }
        }
    }

    // =========================================================================
    // P3 tests — Case 4: A_log negation + V-head reorder
    // =========================================================================

    /// Case 4a: A_log negation — output[i] = -exp(input[i]), tolerance < 1e-6.
    ///
    /// Python ref: convert_hf_to_gguf.py:4788-4789  (`data_torch = -torch.exp(data_torch)`)
    ///
    /// Test: input = [0.0, 1.0, -1.0, 0.5].  Expected = [-1.0, -e, -exp(-1), -exp(0.5)].
    /// With nk=2, nv_per_k=2, head_dim=1 → 4 elements → valid reorder size.
    /// After negation the V-head reorder permutes: [nk=2, nv_per_k=2, head_dim=1]
    /// → reorder_v_heads swaps nk/nv axes → output element order: [0→0, 2→1, 1→2, 3→3].
    ///
    /// So final output = [-exp(input[0]), -exp(input[2]), -exp(input[1]), -exp(input[3])].
    #[test]
    fn case4_a_log_negation_and_reorder() {
        let inputs: [f32; 4] = [0.0, 1.0, -1.0, 0.5];
        let mut data = Vec::with_capacity(16);
        for &v in &inputs {
            data.extend_from_slice(&v.to_le_bytes());
        }

        // nk=2, nv_per_k=2 (apex-like): total 4 elements = nk*nv_per_k*head_dim(=1)
        let ctx = Qwen35ConvertContext {
            arch: Qwen35Arch::Moe,
            layer_types: vec!["linear_attention".to_string()],
            num_layers: 1,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 4,
            linear_num_key_heads: 2,
            linear_value_head_dim: 4,
            linear_num_value_heads: 4,
            linear_num_v_per_k: 2,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: None,
        };

        let tensor = TensorRef {
            name: "model.layers.0.linear_attn.A_log".to_string(),
            shape: vec![4],
            dtype: DType::F32,
            data,
        };

        let result = transform_linear_attn_tensor(tensor, &ctx).unwrap();
        assert_eq!(result.data.len(), 16);

        // After negation: negated[i] = -exp(inputs[i])
        // After reorder (nk=2, nv_per_k=2, head_dim=1, 1-D → per-element reorder):
        //   src: [k=0,v=0]=negated[0], [k=0,v=1]=negated[1], [k=1,v=0]=negated[2], [k=1,v=1]=negated[3]
        //   dst: [v=0,k=0]=negated[0], [v=0,k=1]=negated[2], [v=1,k=0]=negated[1], [v=1,k=1]=negated[3]
        //   Output positions: [neg[0], neg[2], neg[1], neg[3]]
        let expected_src_indices = [0usize, 2, 1, 3];
        for (out_pos, &src_idx) in expected_src_indices.iter().enumerate() {
            let off = out_pos * 4;
            let got = f32::from_le_bytes([
                result.data[off],
                result.data[off + 1],
                result.data[off + 2],
                result.data[off + 3],
            ]);
            let expected = -inputs[src_idx].exp();
            assert!(
                (got - expected).abs() < 1e-6,
                "A_log out[{out_pos}] = {got} but expected -exp(inputs[{src_idx}]={}) = {expected}",
                inputs[src_idx]
            );
        }
    }

    /// Case 4b: A_log overflow — input = 200.0 → output = -inf (no clamping, matches llama.cpp).
    #[test]
    fn case4_a_log_overflow_is_neg_inf() {
        let ctx = Qwen35ConvertContext {
            arch: Qwen35Arch::Moe,
            layer_types: vec!["linear_attention".to_string()],
            num_layers: 1,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 4,
            linear_num_key_heads: 1,
            linear_value_head_dim: 4,
            linear_num_value_heads: 1,
            linear_num_v_per_k: 1,  // nv_per_k=1 → no reorder, just negation check
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: None,
        };

        // Use nv_per_k=1 so the transform short-circuits before reorder —
        // but A_log negation is applied in case 4, which is AFTER the nv_per_k guard.
        // Actually the guard (nv_per_k==1 → return unchanged) fires first.
        // So test directly using transform_case4 via a context that has nv_per_k=2.
        let mut data = Vec::new();
        data.extend_from_slice(&200.0f32.to_le_bytes()); // will overflow exp
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&200.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());

        let tensor = TensorRef {
            name: "model.layers.0.linear_attn.A_log".to_string(),
            shape: vec![4],
            dtype: DType::F32,
            data,
        };

        let ctx2 = Qwen35ConvertContext {
            linear_num_key_heads: 2,
            linear_num_value_heads: 4,
            linear_num_v_per_k: 2,
            ..ctx.clone()
        };

        let result = transform_linear_attn_tensor(tensor, &ctx2).unwrap();
        // After negation + reorder(nk=2,nv_per_k=2,hd=1):
        // negated = [-inf, -1.0, -inf, -1.0]
        // reorder: [v=0,k=0]=neg[0]=-inf, [v=0,k=1]=neg[2]=-inf, [v=1,k=0]=neg[1]=-1.0, [v=1,k=1]=neg[3]=-1.0
        let off0 = 0;
        let v0 = f32::from_le_bytes([result.data[off0], result.data[1], result.data[2], result.data[3]]);
        assert!(v0.is_infinite() && v0 < 0.0, "expected -inf for large A_log input, got {v0}");
    }

    /// Case 4c: dt_bias — V-head reorder applies (no negation), data layout matches reorder.
    ///
    /// dt_bias is 1-D with num_v_heads elements.  nk=2, nv_per_k=2 → 4 elements.
    /// Expected reorder (nk=2, nv_per_k=2, hd=1):
    ///   src layout: [k=0,v=0]=0, [k=0,v=1]=1, [k=1,v=0]=2, [k=1,v=1]=3
    ///   dst: [v=0,k=0]=0, [v=0,k=1]=2, [v=1,k=0]=1, [v=1,k=1]=3
    #[test]
    fn case4_dt_bias_reorder_no_negation() {
        let inputs: [f32; 4] = [10.0, 20.0, 30.0, 40.0];
        let mut data = Vec::with_capacity(16);
        for &v in &inputs {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let ctx = Qwen35ConvertContext {
            arch: Qwen35Arch::Moe,
            layer_types: vec!["linear_attention".to_string()],
            num_layers: 1,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 4,
            linear_num_key_heads: 2,
            linear_value_head_dim: 4,
            linear_num_value_heads: 4,
            linear_num_v_per_k: 2,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: None,
        };

        let tensor = TensorRef {
            name: "model.layers.0.linear_attn.dt_bias".to_string(),
            shape: vec![4],
            dtype: DType::F32,
            data,
        };

        let result = transform_linear_attn_tensor(tensor, &ctx).unwrap();
        assert_eq!(result.data.len(), 16);

        // No negation (not A_log).  Reorder: [0,2,1,3] → inputs [10,30,20,40]
        let expected = [10.0f32, 30.0, 20.0, 40.0];
        for (i, &exp) in expected.iter().enumerate() {
            let off = i * 4;
            let got = f32::from_le_bytes([
                result.data[off], result.data[off+1], result.data[off+2], result.data[off+3],
            ]);
            assert!((got - exp).abs() < 1e-7, "dt_bias out[{i}] = {got}, expected {exp}");
        }
    }

    // =========================================================================
    // P3 tests — Case 5: conv1d squeeze + V-channel reorder
    // =========================================================================

    /// Case 5: conv1d squeeze [k,1,d] → [k,d] then V-channel reorder.
    ///
    /// # Parameters
    /// nk=2, nv_per_k=2, head_k_dim=1, head_v_dim=1.
    /// qk_channels = head_k_dim * nk * 2 = 1 * 2 * 2 = 4.
    /// Total channels k = qk_channels + nk * nv_per_k * head_v_dim = 4 + 2*2*1 = 8.
    /// d = 3 (conv kernel width).  Shape = [8, 1, 3].
    ///
    /// # Expected output shape
    /// [8, 3] (squeezed).
    ///
    /// # Expected reorder
    /// QK channels 0..3 pass through unchanged (rows 0,1,2,3).
    /// V channels 4..7 (4 channels = nk*nv_per_k) are reordered:
    ///   src v_slice: [k=0,v=0]=row4, [k=0,v=1]=row5, [k=1,v=0]=row6, [k=1,v=1]=row7
    ///   dst: [v=0,k=0]=row4, [v=0,k=1]=row6, [v=1,k=0]=row5, [v=1,k=1]=row7
    ///   → output row order in v region: [row4, row6, row5, row7]
    #[test]
    fn case5_conv1d_squeeze_and_v_reorder() {
        let nk = 2u32;
        let nv_per_k = 2u32;
        let head_k_dim = 1u32;
        let head_v_dim = 1u32;
        let d = 3usize;
        let qk_channels = (head_k_dim * nk * 2) as usize; // 4
        let v_channels  = (nk * nv_per_k * head_v_dim) as usize; // 4
        let total_k     = qk_channels + v_channels; // 8

        // Shape [8, 1, 3]: row i has values [i*3, i*3+1, i*3+2].
        let n = total_k * 1 * d;
        let mut data = Vec::with_capacity(n * 4);
        for i in 0u32..(n as u32) {
            data.extend_from_slice(&i.to_le_bytes());
        }

        let ctx = Qwen35ConvertContext {
            arch: Qwen35Arch::Moe,
            layer_types: vec!["linear_attention".to_string()],
            num_layers: 1,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            linear_conv_kernel_dim: 3,
            linear_key_head_dim: head_k_dim,
            linear_num_key_heads: nk,
            linear_value_head_dim: head_v_dim,
            linear_num_value_heads: nk * nv_per_k,
            linear_num_v_per_k: nv_per_k,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: None,
        };

        let tensor = TensorRef {
            name: "model.layers.0.linear_attn.conv1d.weight".to_string(),
            shape: vec![total_k, 1, d],
            dtype: DType::F32,
            data,
        };

        let result = transform_linear_attn_tensor(tensor, &ctx).unwrap();

        // Shape should be squeezed to [8, 3].
        assert_eq!(result.shape, vec![total_k, d], "shape not squeezed correctly");
        assert_eq!(result.data.len(), total_k * d * 4, "length mismatch after squeeze");

        // QK channels 0..3 unchanged (rows 0-3, each 3 elements).
        for row in 0..qk_channels {
            for col in 0..d {
                let elem_idx = row * d + col;
                let expected = elem_idx as u32;
                let got = read_u32(&result.data, elem_idx);
                assert_eq!(got, expected, "QK row={row} col={col}: expected {expected} got {got}");
            }
        }

        // V-channel reorder: src rows [4,5,6,7] → dst order [4,6,5,7]
        // (nk=2, nv_per_k=2, head_v_dim=1 → head_dim_for_reorder = head_v_dim*d = 1*3 = 3)
        // After reorder: v_dst[v=0,k=0]=v_src[k=0,v=0]=row4, v_dst[v=0,k=1]=v_src[k=1,v=0]=row6,
        //                v_dst[v=1,k=0]=v_src[k=0,v=1]=row5, v_dst[v=1,k=1]=v_src[k=1,v=1]=row7
        let v_src_row_order = [4usize, 6, 5, 7]; // output row i → original input row
        for (v_out_idx, &src_row) in v_src_row_order.iter().enumerate() {
            let dst_row = qk_channels + v_out_idx;
            for col in 0..d {
                let src_elem = src_row * d + col;
                let dst_elem = dst_row * d + col;
                let expected = src_elem as u32;
                let got = read_u32(&result.data, dst_elem);
                assert_eq!(
                    got, expected,
                    "V conv1d: dst_row={dst_row} col={col}: expected elem from src_row={src_row} ({expected}), got {got}"
                );
            }
        }
    }

    /// Case 5: conv1d wrong shape → typed error, no silent reshape.
    #[test]
    fn case5_conv1d_wrong_shape_returns_error() {
        let ctx = apex_ctx();
        // Wrong: shape [64, 4] (2-D, no singleton dim).
        let tensor = TensorRef {
            name: "model.layers.0.linear_attn.conv1d.weight".to_string(),
            shape: vec![64, 4],
            dtype: DType::F32,
            data: vec![0u8; 64 * 4 * 4],
        };
        let err = transform_linear_attn_tensor(tensor, &ctx)
            .expect_err("conv1d with wrong shape should return error");
        assert!(
            matches!(err, ConvertError::ReorderInvariantViolated { .. }),
            "expected ReorderInvariantViolated, got: {err}"
        );
    }

    // =========================================================================
    // P3 tests — in_proj_qkvz split
    // =========================================================================

    /// in_proj_qkvz split: verify that Q/K/V/Z are correctly extracted.
    ///
    /// # Parameters
    /// nk=2, nv_per_k=2, head_k_dim=2, head_v_dim=2, hidden_size=4.
    ///
    /// per-head sizes: q=2, k=2, v=nv_per_k*head_v_dim=4, z=4.  total_per_head=12.
    /// total_rows = nk * total_per_head = 2 * 12 = 24.
    /// Shape = [24, 4].
    ///
    /// # Layout verification
    /// We label each source row value `r * hidden_size + c` and check the output
    /// Q/K rows match those source elements after the permute/view/split/permute back.
    #[test]
    fn in_proj_qkvz_split_correct_partition() {
        let nk = 2usize;
        let nv_per_k = 2usize;
        let head_k_dim = 2usize;
        let head_v_dim = 2usize;
        let hidden_size = 4usize;

        let q_per_head = head_k_dim;
        let k_per_head = head_k_dim;
        let v_per_head = nv_per_k * head_v_dim;
        let z_per_head = nv_per_k * head_v_dim;
        let total_per_head = q_per_head + k_per_head + v_per_head + z_per_head; // 12
        let total_rows = nk * total_per_head; // 24

        // Source tensor: row i, col c → value = i * hidden_size + c (as u32 LE)
        let n_elems = total_rows * hidden_size;
        let mut data = Vec::with_capacity(n_elems * 4);
        for elem_idx in 0u32..(n_elems as u32) {
            data.extend_from_slice(&elem_idx.to_le_bytes());
        }

        let ctx = Qwen35ConvertContext {
            arch: Qwen35Arch::Dense,
            layer_types: vec!["linear_attention".to_string()],
            num_layers: 1,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: head_k_dim as u32,
            linear_num_key_heads: nk as u32,
            linear_value_head_dim: head_v_dim as u32,
            linear_num_value_heads: (nk * nv_per_k) as u32,
            linear_num_v_per_k: nv_per_k as u32,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: None,
        };

        let tensor = TensorRef {
            name: "model.layers.0.linear_attn.in_proj_qkvz.weight".to_string(),
            shape: vec![total_rows, hidden_size],
            dtype: DType::F32,
            data,
        };

        let (qkv, z) = transform_in_proj_qkvz(tensor.clone(), &ctx).unwrap();

        // Check shape: qkv = [q_rows + k_rows + v_rows, hidden_size]
        let q_rows = nk * q_per_head;
        let k_rows = nk * k_per_head;
        let v_rows = nk * v_per_head;
        let z_rows = nk * z_per_head;
        assert_eq!(qkv.shape, vec![q_rows + k_rows + v_rows, hidden_size], "qkv shape mismatch");
        assert_eq!(z.shape, vec![z_rows, hidden_size], "z shape mismatch");

        // Check Q output: qkv row (h * q_per_head + fi, c) should equal
        // source element at row = h * total_per_head + 0 + fi, col = c.
        // In our flat encoding: source_elem_idx = (h*total_per_head + fi)*hidden_size + c
        let q_off_in_head = 0;
        for h in 0..nk {
            for fi in 0..q_per_head {
                for c in 0..hidden_size {
                    let dst_row = h * q_per_head + fi;
                    let dst_idx = dst_row * hidden_size + c;
                    let src_row = h * total_per_head + q_off_in_head + fi;
                    let expected = (src_row * hidden_size + c) as u32;
                    let got = read_u32(&qkv.data, dst_idx);
                    assert_eq!(got, expected, "Q h={h} fi={fi} c={c}: expected {expected} got {got}");
                }
            }
        }

        // Check Z output: z row (h * z_per_head + fi, c) should equal
        // source element at row = h * total_per_head + z_off + fi, col = c.
        let z_off_in_head = q_per_head + k_per_head + v_per_head;
        for h in 0..nk {
            for fi in 0..z_per_head {
                for c in 0..hidden_size {
                    let dst_row = h * z_per_head + fi;
                    let dst_idx = dst_row * hidden_size + c;
                    let src_row = h * total_per_head + z_off_in_head + fi;
                    let expected = (src_row * hidden_size + c) as u32;
                    let got = read_u32(&z.data, dst_idx);
                    assert_eq!(got, expected, "Z h={h} fi={fi} c={c}: expected {expected} got {got}");
                }
            }
        }

        // Check names
        assert!(qkv.name.contains("in_proj_qkv.weight"), "qkv name: {}", qkv.name);
        assert!(z.name.contains("in_proj_z.weight"), "z name: {}", z.name);
    }

    // =========================================================================
    // P3 tests — RMS norm +1
    // =========================================================================

    /// RMS norm +1: F32 input tensor — output = input + 1.0.
    ///
    /// # Citation
    /// `/opt/llama.cpp/convert_hf_to_gguf.py:4794-4795` in `Qwen3NextModel.modify_tensors`:
    /// `elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):`
    /// `    data_torch = data_torch + 1`
    #[test]
    fn rms_norm_plus_one_f32() {
        let inputs = [0.0f32, 0.5, -0.5, 1.0, -1.0, 2.0];
        let mut data = Vec::with_capacity(inputs.len() * 4);
        for &v in &inputs {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let tensor = TensorRef {
            name: "model.layers.0.input_layernorm.weight".to_string(),
            shape: vec![inputs.len()],
            dtype: DType::F32,
            data,
        };

        let result = apply_rms_norm_plus_one(tensor).unwrap();

        for (i, &inp) in inputs.iter().enumerate() {
            let off = i * 4;
            let got = f32::from_le_bytes([
                result.data[off], result.data[off+1],
                result.data[off+2], result.data[off+3],
            ]);
            let expected = inp + 1.0;
            assert!(
                (got - expected).abs() < 1e-7,
                "rms_norm_plus_one F32: out[{i}] = {got}, expected {expected}"
            );
        }
    }

    /// RMS norm +1: `linear_attn.norm.weight` is EXCLUDED (no +1 applied).
    ///
    /// Citation: `/opt/llama.cpp/convert_hf_to_gguf.py:4794`:
    /// `not name.endswith("linear_attn.norm.weight")` — the linear-attention sub-norm
    /// is stored as-is, without the +1 bias.
    #[test]
    fn rms_norm_plus_one_excluded_for_linear_attn_norm() {
        let inputs = [0.0f32, 0.5, -0.5, 1.0];
        let mut data = Vec::with_capacity(inputs.len() * 4);
        for &v in &inputs {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let original = data.clone();

        let tensor = TensorRef {
            name: "model.layers.0.linear_attn.norm.weight".to_string(),
            shape: vec![inputs.len()],
            dtype: DType::F32,
            data,
        };

        let result = apply_rms_norm_plus_one(tensor).unwrap();
        assert_eq!(result.data, original, "linear_attn.norm.weight should NOT be +1'd");
    }

    /// RMS norm +1: BF16 tensor — roundtrip through BF16 preserves +1 to within BF16 precision.
    #[test]
    fn rms_norm_plus_one_bf16() {
        let inputs = [0.0f32, 0.5, 1.0, -1.0];
        let mut data = Vec::with_capacity(inputs.len() * 2);
        for &v in &inputs {
            data.extend_from_slice(&half::bf16::from_f32(v).to_bits().to_le_bytes());
        }

        let tensor = TensorRef {
            name: "model.layers.0.post_attention_layernorm.weight".to_string(),
            shape: vec![inputs.len()],
            dtype: DType::BF16,
            data,
        };

        let result = apply_rms_norm_plus_one(tensor).unwrap();
        assert_eq!(result.dtype, DType::BF16);

        for (i, &inp) in inputs.iter().enumerate() {
            let off = i * 2;
            let bits = u16::from_le_bytes([result.data[off], result.data[off + 1]]);
            let got = half::bf16::from_bits(bits).to_f32();
            let expected = inp + 1.0;
            // BF16 has ~2 decimal digits of precision; allow 0.01 tolerance.
            assert!(
                (got - expected).abs() < 0.01,
                "rms_norm_plus_one BF16: out[{i}] = {got}, expected {expected}"
            );
        }
    }

    // =========================================================================
    // Layer-type dispatcher tests
    // =========================================================================

    /// is_linear_attention_layer: index 0 is linear_attention in apex config.
    #[test]
    fn dispatcher_layer0_is_linear_attention() {
        let ctx = apex_ctx();
        assert!(is_linear_attention_layer(0, &ctx), "layer 0 should be linear_attention");
        assert!(!is_full_attention_layer(0, &ctx), "layer 0 should not be full_attention");
    }

    /// is_full_attention_layer: index 3 is full_attention in apex config
    /// (every 4th layer, 1-indexed: 4, 8, …, i.e. 0-indexed: 3, 7, …).
    #[test]
    fn dispatcher_layer3_is_full_attention() {
        let ctx = apex_ctx();
        assert!(is_full_attention_layer(3, &ctx), "layer 3 should be full_attention");
        assert!(!is_linear_attention_layer(3, &ctx), "layer 3 should not be linear_attention");
    }

    /// All 10 full-attention layers in apex are at indices 3,7,11,15,19,23,27,31,35,39.
    #[test]
    fn dispatcher_all_full_attention_indices_apex() {
        let ctx = apex_ctx();
        let expected_full: Vec<usize> = (0..40).filter(|i| (i + 1) % 4 == 0).collect();
        for i in 0..40 {
            let is_full = is_full_attention_layer(i, &ctx);
            let is_linear = is_linear_attention_layer(i, &ctx);
            let expected = expected_full.contains(&i);
            assert_eq!(is_full, expected, "layer {i}: is_full_attention mismatch");
            assert_eq!(is_linear, !expected, "layer {i}: is_linear_attention mismatch");
        }
    }

    /// No reorder when num_v_per_k == 1 (identity case).
    #[test]
    fn no_reorder_when_num_v_per_k_is_1() {
        let tensor = make_u32_tensor(
            "model.layers.0.linear_attn.in_proj_z.weight",
            vec![16, 4],
        );
        let original_data = tensor.data.clone();

        let ctx = Qwen35ConvertContext {
            arch: Qwen35Arch::Dense,
            layer_types: vec!["linear_attention".to_string()],
            num_layers: 1,
            num_attention_heads: 16,
            num_kv_heads: 16,
            head_dim: 4,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 4,
            linear_num_key_heads: 16,
            linear_value_head_dim: 4,
            linear_num_value_heads: 16, // num_v_per_k = 1
            linear_num_v_per_k: 1,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: None,
        };

        let result = transform_linear_attn_tensor(tensor, &ctx).unwrap();
        assert_eq!(result.data, original_data, "should be identity when nv_per_k=1");
    }

    // =========================================================================
    // Round-trip per-case tests (using explicit inverse)
    // =========================================================================

    /// Case 2 round-trip via explicit inverse.
    ///
    /// The forward maps `[nk, nv, hd]` → `[nv, nk, hd]`.
    /// The inverse maps `[nv, nk, hd]` → `[nk, nv, hd]` by swapping nk/nv.
    /// Double-application with same params is NOT the identity for nk != nv.
    #[test]
    fn case2_round_trip() {
        let nk = 3u32;
        let nv = 2u32;
        let hd = 4u32;
        let cols = 5u32;
        let head_dim_total = hd * cols;
        let total = (nk * nv * head_dim_total) as usize;

        let mut src = Vec::with_capacity(total * 4);
        for i in 0u32..total as u32 { src.extend_from_slice(&i.to_le_bytes()); }

        let fwd = reorder_v_heads(&src, 4, nk, nv, head_dim_total).unwrap();
        // Inverse: swap nk and nv
        let inv = reorder_v_heads_inverse(&fwd, 4, nk, nv, head_dim_total).unwrap();
        assert_eq!(inv, src, "case2 round-trip (fwd then inv) failed");

        // Also check inv then fwd
        let inv2 = reorder_v_heads_inverse(&src, 4, nk, nv, head_dim_total).unwrap();
        let fwd2 = reorder_v_heads(&inv2, 4, nk, nv, head_dim_total).unwrap();
        assert_eq!(fwd2, src, "case2 round-trip (inv then fwd) failed");
    }

    /// Case 6 round-trip: apply forward transform then its inverse, expect identity.
    ///
    /// The inverse of case 6 (column reorder with nk/nv params) requires swapping
    /// nk and nv_per_k — i.e. calling `transform_linear_attn_tensor` with a context
    /// where `linear_num_key_heads` and `linear_num_v_per_k` are swapped.
    #[test]
    fn case6_round_trip() {
        let nk = 4u32;
        let nv = 3u32;
        let head_v_dim = 8u32;
        let rows = 5usize;
        let cols = (nk * nv * head_v_dim) as usize;

        let tensor = make_u32_tensor(
            "model.layers.0.linear_attn.out_proj.weight",
            vec![rows, cols],
        );
        let original_data = tensor.data.clone();

        // Forward context: nk=4, nv_per_k=3
        let ctx_fwd = Qwen35ConvertContext {
            arch: Qwen35Arch::Dense,
            layer_types: vec!["linear_attention".to_string()],
            num_layers: 1,
            num_attention_heads: 4,
            num_kv_heads: 4,
            head_dim: 8,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 8,
            linear_num_key_heads: nk,
            linear_value_head_dim: head_v_dim,
            linear_num_value_heads: nk * nv,
            linear_num_v_per_k: nv,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: None,
        };

        // Inverse context: swap nk and nv (since the tiled output has nv groups of nk heads)
        let ctx_inv = Qwen35ConvertContext {
            linear_num_key_heads: nv,
            linear_num_value_heads: nv * nk,
            linear_num_v_per_k: nk,
            ..ctx_fwd.clone()
        };

        let fwd = transform_linear_attn_tensor(tensor.clone(), &ctx_fwd).unwrap();
        let inv = transform_linear_attn_tensor(fwd, &ctx_inv).unwrap();
        assert_eq!(inv.data, original_data, "case6 round-trip (fwd then inv) failed");
    }
}
