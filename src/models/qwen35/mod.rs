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

use crate::input::config_parser::{validate_required_qwen35moe_fields, ConfigParseError};
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
        validate_required_qwen35moe_fields(metadata)?;

        let arch = if metadata.is_moe() {
            Qwen35Arch::Moe
        } else {
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
    if name.contains(".A_log") || name.contains(".dt_bias") || name.contains(".dt_proj") {
        // A_log negation (negate then reorder) body is P3.
        // For now, only perform the V-head reorder; negation deferred.
        //
        // NOTE: this stub returns an error so P3 knows exactly what to fill in.
        return Err(ConvertError::PhaseStub {
            phase: "P3",
            what: "A_log / dt_bias / dt_proj negation + reorder body",
        });
    }

    // Case 5: conv1d — reorder V channel portion (py:5411-5418)
    if name.contains(".conv1d") {
        // conv1d squeeze + channel-split body is P3.
        return Err(ConvertError::PhaseStub {
            phase: "P3",
            what: "conv1d channel-split reorder body",
        });
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

    /// Case 4: A_log — PhaseStub (P3 not yet implemented).
    #[test]
    fn case4_a_log_returns_phase_stub() {
        let tensor = make_u32_tensor("model.layers.0.linear_attn.A_log", vec![32]);
        let ctx = apex_ctx();
        let err = transform_linear_attn_tensor(tensor, &ctx)
            .expect_err("case4 A_log should return PhaseStub");
        assert!(
            matches!(err, ConvertError::PhaseStub { phase: "P3", .. }),
            "expected PhaseStub(P3), got: {err}"
        );
    }

    /// Case 5: conv1d — PhaseStub (P3 not yet implemented).
    #[test]
    fn case5_conv1d_returns_phase_stub() {
        let tensor = make_u32_tensor(
            "model.layers.0.linear_attn.conv1d.weight",
            vec![64, 1, 4],
        );
        let ctx = apex_ctx();
        let err = transform_linear_attn_tensor(tensor, &ctx)
            .expect_err("case5 conv1d should return PhaseStub");
        assert!(
            matches!(err, ConvertError::PhaseStub { phase: "P3", .. }),
            "expected PhaseStub(P3), got: {err}"
        );
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
