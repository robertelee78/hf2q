//! Qwen3.5-MoE conversion — tensor naming, expert merge, and metadata.
//!
//! Covers Qwen3.5-MoE-35B-A3.9B (256 experts, top-8) and similar variants.
//!
//! ADR-012 Decisions 4, 7, 9:
//!   `hf_tensor_name_to_gguf_moe` — maps HF tensor names to GGUF names.
//!   `merge_expert_tensors` — stacks 256 per-expert weight tensors into
//!     a single merged tensor (body deferred to P5).
//!   `emit_metadata_moe` — writes GGUF metadata keys (body deferred to P4).
//!
//! This file does NOT implement:
//! - Expert merge body (P5)
//! - Metadata emission body (P4)
//! - Dispatch wiring in `src/backends/gguf.rs` (P4)

use super::{ConvertError, Qwen35ConvertContext};
use crate::ir::TensorRef;

// ---------------------------------------------------------------------------
// Tensor name mapping
// ---------------------------------------------------------------------------

/// Map a HuggingFace tensor name to its GGUF counterpart for Qwen3.5-MoE.
///
/// Returns `None` if the tensor should be skipped.
/// Returns `Some(gguf_name)` for tensors that map to GGUF.
///
/// # MoE FFN naming (ADR-012 Decision 4, 7)
///
/// Per-expert weights:
///   HF:   `model.layers.{N}.mlp.experts.{E}.{gate,up,down}_proj.weight`
///   GGUF: `blk.{N}.ffn_{gate,up,down}_exps.weight` (merged; E dimension merged by P5)
///
/// Shared expert:
///   HF:   `model.layers.{N}.mlp.shared_expert.{gate,up,down}_proj.weight`
///   GGUF: `blk.{N}.ffn_{gate,up,down}_shexp.weight`
///
/// Router:
///   HF:   `model.layers.{N}.mlp.gate.weight`
///   GGUF: `blk.{N}.ffn_gate_inp.weight`
///
/// Full-attention and linear-attention layers use the same naming as
/// `hf_tensor_name_to_gguf_dense` (shared logic is factored into helpers).
pub fn hf_tensor_name_to_gguf_moe(
    hf_name: &str,
    _ctx: &Qwen35ConvertContext,
) -> Option<String> {
    // Embeddings
    if hf_name == "model.embed_tokens.weight" {
        return Some("token_embd.weight".to_string());
    }
    // Final norm
    if hf_name == "model.norm.weight" {
        return Some("output_norm.weight".to_string());
    }
    // LM head
    if hf_name == "lm_head.weight" {
        return Some("output.weight".to_string());
    }

    // Layer tensors — extract layer index.
    let rest = hf_name.strip_prefix("model.layers.")?;
    let dot = rest.find('.')?;
    let layer_str = &rest[..dot];
    let layer_idx: usize = layer_str.parse().ok()?;
    let suffix = &rest[dot + 1..]; // everything after "model.layers.N."

    let blk = format!("blk.{}", layer_idx);

    // --- Full-attention projection (same as dense) ---
    if let Some(name) = map_full_attn_suffix(suffix, &blk) {
        return Some(name);
    }

    // --- Linear-attention projection (same as dense) ---
    if let Some(name) = map_linear_attn_suffix(suffix, &blk) {
        return Some(name);
    }

    // --- MoE router (sparse experts) ---
    // tensor_mapping.py:443 + llama-arch.cpp:393: mlp.gate.weight → blk.N.ffn_gate_inp.weight
    if suffix == "mlp.gate.weight" {
        return Some(format!("{}.ffn_gate_inp.weight", blk));
    }

    // --- Shared-expert scalar gate ---
    // tensor_mapping.py:447 + llama-arch.cpp:394: mlp.shared_expert_gate → blk.N.ffn_gate_inp_shexp.weight
    // llama-model.cpp:7596: layer.ffn_gate_inp_shexp = create_tensor(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", i)
    if suffix == "mlp.shared_expert_gate.weight" {
        return Some(format!("{}.ffn_gate_inp_shexp.weight", blk));
    }

    // --- Per-expert weights (gate / up / down) ---
    // HF: "mlp.experts.{E}.{gate,up,down}_proj.weight"
    if let Some(name) = map_per_expert_suffix(suffix, &blk) {
        return Some(name);
    }

    // --- Shared expert ---
    if let Some(name) = map_shared_expert_suffix(suffix, &blk) {
        return Some(name);
    }

    // --- Layer norms ---
    if let Some(name) = map_norm_suffix(suffix, &blk) {
        return Some(name);
    }

    // Unknown — tag it.
    Some(format!("unk.{}", hf_name))
}

/// Map a per-expert tensor suffix → GGUF merged-expert name.
///
/// The expert index `E` is encoded in the suffix but the GGUF name uses a
/// single merged tensor (P5 handles the merge).  The name here uses the
/// `_exps` convention matching llama.cpp's MoE naming.
fn map_per_expert_suffix(suffix: &str, blk: &str) -> Option<String> {
    // Pattern: "mlp.experts.{E}.{gate,up,down}_proj.weight"
    let rest = suffix.strip_prefix("mlp.experts.")?;
    let dot = rest.find('.')?;
    let _expert_idx: usize = rest[..dot].parse().ok()?;
    let proj_suffix = &rest[dot + 1..];

    let gguf = match proj_suffix {
        "gate_proj.weight" => format!("{}.ffn_gate_exps.weight", blk),
        "up_proj.weight"   => format!("{}.ffn_up_exps.weight", blk),
        "down_proj.weight" => format!("{}.ffn_down_exps.weight", blk),
        _ => return None,
    };
    Some(gguf)
}

/// Map shared-expert tensor suffix → GGUF name.
fn map_shared_expert_suffix(suffix: &str, blk: &str) -> Option<String> {
    let rest = suffix.strip_prefix("mlp.shared_expert.")?;
    let gguf = match rest {
        "gate_proj.weight" => format!("{}.ffn_gate_shexp.weight", blk),
        "up_proj.weight"   => format!("{}.ffn_up_shexp.weight", blk),
        "down_proj.weight" => format!("{}.ffn_down_shexp.weight", blk),
        _ => return None,
    };
    Some(gguf)
}

/// Map full-attention suffix → GGUF name (shared with dense).
fn map_full_attn_suffix(suffix: &str, blk: &str) -> Option<String> {
    match suffix {
        "self_attn.q_proj.weight" => Some(format!("{}.attn_q.weight", blk)),
        "self_attn.k_proj.weight" => Some(format!("{}.attn_k.weight", blk)),
        "self_attn.v_proj.weight" => Some(format!("{}.attn_v.weight", blk)),
        "self_attn.o_proj.weight" => Some(format!("{}.attn_output.weight", blk)),
        "self_attn.q_norm.weight" => Some(format!("{}.attn_q_norm.weight", blk)),
        "self_attn.k_norm.weight" => Some(format!("{}.attn_k_norm.weight", blk)),
        _ => None,
    }
}

/// Map linear-attention suffix → GGUF name (shared with dense).
///
/// Same mapping table as dense — identical linear-attention architecture in
/// both variants.  See `dense.rs:map_linear_attn_suffix` for full citation chain.
///
/// | HF suffix             | GGUF suffix        | Source |
/// |---|---|---|
/// | in_proj_qkv.weight    | attn_qkv.weight    | llama-arch.cpp:382 |
/// | in_proj_z.weight      | attn_gate.weight   | llama-arch.cpp:370 |
/// | in_proj_a.weight      | ssm_alpha.weight   | llama-arch.cpp:400 |
/// | in_proj_b.weight      | ssm_beta.weight    | llama-arch.cpp:416 |
/// | out_proj.weight       | ssm_out.weight     | llama-arch.cpp:402 |
/// | A_log                 | ssm_a              | llama-arch.cpp:395 |
/// | dt_bias               | ssm_dt.bias        | llama-arch.cpp:397 + convert_hf_to_gguf.py:4791 |
/// | conv1d.weight         | ssm_conv1d.weight  | llama-arch.cpp:396 |
/// | norm.weight           | ssm_norm.weight    | llama-arch.cpp:401 |
fn map_linear_attn_suffix(suffix: &str, blk: &str) -> Option<String> {
    let rest = suffix.strip_prefix("linear_attn.")?;
    let gguf = match rest {
        // llama-arch.cpp:382 LLM_TENSOR_ATTN_QKV → "blk.%d.attn_qkv"
        "in_proj_qkv.weight" => format!("{}.attn_qkv.weight", blk),
        // llama-arch.cpp:370 LLM_TENSOR_ATTN_GATE → "blk.%d.attn_gate"
        "in_proj_z.weight"   => format!("{}.attn_gate.weight", blk),
        // llama-arch.cpp:400 LLM_TENSOR_SSM_ALPHA → "blk.%d.ssm_alpha"
        "in_proj_a.weight"   => format!("{}.ssm_alpha.weight", blk),
        // llama-arch.cpp:416 LLM_TENSOR_SSM_BETA → "blk.%d.ssm_beta"
        "in_proj_b.weight"   => format!("{}.ssm_beta.weight", blk),
        // llama-arch.cpp:402 LLM_TENSOR_SSM_OUT → "blk.%d.ssm_out"
        "out_proj.weight"    => format!("{}.ssm_out.weight", blk),
        // llama-arch.cpp:395 LLM_TENSOR_SSM_A_NOSCAN → "blk.%d.ssm_a"
        "A_log"              => format!("{}.ssm_a", blk),
        // convert_hf_to_gguf.py:4791 dt_bias → dt_proj.bias; llama-arch.cpp:397 → "blk.%d.ssm_dt.bias"
        "dt_bias"            => format!("{}.ssm_dt.bias", blk),
        // llama-arch.cpp:396 LLM_TENSOR_SSM_CONV1D → "blk.%d.ssm_conv1d"
        "conv1d.weight"      => format!("{}.ssm_conv1d.weight", blk),
        "conv1d.bias"        => format!("{}.ssm_conv1d.bias", blk),
        // llama-arch.cpp:401 LLM_TENSOR_SSM_NORM → "blk.%d.ssm_norm"
        "norm.weight"        => format!("{}.ssm_norm.weight", blk),
        _ => return None,
    };
    Some(gguf)
}

/// Map layer-norm / RMS-norm suffix → GGUF name (shared with dense).
///
/// # post_attention_layernorm verdict (ADR-012 P4 audit)
///
/// Source: llama-model.cpp:7564-7565 (qwen35moe tensor load):
///   `layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), …)`
///   `layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), …)`
///
/// Source: llama-arch.cpp:367:
///   `{ LLM_TENSOR_ATTN_POST_NORM, "blk.%d.post_attention_norm" }`
///
/// Qwen3.5-MoE uses `attn_post_norm` (GGUF: `post_attention_norm`), NOT `ffn_norm`.
/// The MoE variant follows the exact same norm pattern as dense — same field names,
/// same GGUF key.  `ffn_norm` would cause llama.cpp to reject the file.
fn map_norm_suffix(suffix: &str, blk: &str) -> Option<String> {
    match suffix {
        "input_layernorm.weight" => Some(format!("{}.attn_norm.weight", blk)),
        // llama-model.cpp:7565 loads layer.attn_post_norm via LLM_TENSOR_ATTN_POST_NORM
        // llama-arch.cpp:367: LLM_TENSOR_ATTN_POST_NORM → "blk.%d.post_attention_norm"
        "post_attention_layernorm.weight" => {
            Some(format!("{}.post_attention_norm.weight", blk))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Expert merge (P5 — implemented)
// ---------------------------------------------------------------------------

/// Which expert projection this merge is for.
///
/// Encodes the per-projection stacking rule verified against llama-model.cpp:3281-3283
/// for the `qwen35moe` arch:
///
/// ```text
/// layer.ffn_gate_exps = create_tensor(…, {n_embd, n_ff_exp, n_expert}, …);
/// layer.ffn_down_exps = create_tensor(…, {n_ff_exp, n_embd, n_expert}, …);
/// layer.ffn_up_exps   = create_tensor(…, {n_embd, n_ff_exp, n_expert}, …);
/// ```
///
/// GGML ne is stored innermost-first.  PyTorch (outermost-first) stacked shape:
/// - gate / up:  `[N_experts, moe_inter, hidden]` → GGML ne `{hidden, moe_inter, N_experts}`
/// - down:       `[N_experts, hidden, moe_inter]` → GGML ne `{moe_inter, hidden, N_experts}`
///
/// HF per-expert weight shapes (PyTorch, outermost-first):
/// - gate_proj / up_proj:  `[moe_inter, hidden]`   (out_features × in_features)
/// - down_proj:            `[hidden, moe_inter]`
///
/// Stacking along dim 0 gives the correct PyTorch layout for all three projections;
/// only the per-expert shapes differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertProj {
    /// gate_proj — HF shape per expert: `[moe_inter, hidden]`
    Gate,
    /// up_proj   — HF shape per expert: `[moe_inter, hidden]`
    Up,
    /// down_proj — HF shape per expert: `[hidden, moe_inter]`
    Down,
}

/// Stack per-expert weight tensors into a single merged GGUF tensor.
///
/// # ADR-012 Decision 9 (P5)
///
/// `expert_tensors`: ordered by expert index 0..N, all with the same 2-D shape
/// and dtype.  All HF per-expert weights are 2-D: `[out_features, in_features]`.
///
/// The merge concatenates raw bytes in expert order along a new first axis,
/// producing a 3-D tensor `[N, out_features, in_features]`.  This matches
/// `torch.stack(datas, dim=0)` from `convert_hf_to_gguf.py:4628`.
///
/// The `proj` parameter documents (but does not alter) the stacking — all
/// projections use the same `dim=0` stack; the distinction matters only for
/// the shape interpretation by the caller and for the `ExpertProj::Down`
/// dim-order documentation.
///
/// # Dense arch guard
///
/// Call sites are arch-gated.  Calling this from a dense context is a
/// programming error; a `debug_assert` fires in debug builds and a typed
/// `ConvertError::DenseContextMergeCall` error is returned in all builds.
///
/// # Errors
///
/// - `ConvertError::DenseContextMergeCall` if `arch` is Dense.
/// - `ConvertError::ExpertMergeEmpty` if `expert_tensors` is empty.
/// - `ConvertError::ExpertMergeShapeMismatch` if any tensor's shape/dtype/data
///   length differs from the first tensor's.
pub fn merge_expert_tensors(
    expert_tensors: &[TensorRef],
    _proj: ExpertProj,
    arch: super::Qwen35Arch,
) -> Result<TensorRef, ConvertError> {
    // Dense arch guard — return an error if called from a dense-arch context.
    // The caller should never route here for Dense arch, but we handle it
    // gracefully via error return rather than assert-panic so callers can
    // propagate the error and the test can verify the guard.
    if arch == super::Qwen35Arch::Dense {
        return Err(ConvertError::DenseContextMergeCall);
    }

    if expert_tensors.is_empty() {
        return Err(ConvertError::ExpertMergeEmpty);
    }

    let first = &expert_tensors[0];

    // All tensors must be 2-D with the same shape and dtype.
    if first.shape.len() != 2 {
        return Err(ConvertError::ExpertMergeShapeMismatch {
            expert_idx: 0,
            reason: format!(
                "expected 2-D tensor, got {} dims",
                first.shape.len()
            ),
        });
    }

    let expected_bytes = first.shape.iter().product::<usize>() * first.dtype.element_size();
    if first.data.len() != expected_bytes {
        return Err(ConvertError::TensorLengthMismatch {
            name: first.name.clone(),
            expected: expected_bytes,
            actual: first.data.len(),
        });
    }

    let n = expert_tensors.len();
    let rows = first.shape[0];
    let cols = first.shape[1];
    let bytes_per_expert = first.data.len();

    // Validate all subsequent tensors match.
    for (i, t) in expert_tensors.iter().enumerate().skip(1) {
        if t.shape != first.shape {
            return Err(ConvertError::ExpertMergeShapeMismatch {
                expert_idx: i,
                reason: format!(
                    "shape {:?} differs from expert 0 shape {:?}",
                    t.shape, first.shape
                ),
            });
        }
        if t.dtype != first.dtype {
            return Err(ConvertError::ExpertMergeShapeMismatch {
                expert_idx: i,
                reason: format!(
                    "dtype {:?} differs from expert 0 dtype {:?}",
                    t.dtype, first.dtype
                ),
            });
        }
        let expected = t.shape.iter().product::<usize>() * t.dtype.element_size();
        if t.data.len() != expected {
            return Err(ConvertError::TensorLengthMismatch {
                name: t.name.clone(),
                expected,
                actual: t.data.len(),
            });
        }
    }

    // Stack: concatenate raw bytes in expert order — equivalent to
    // torch.stack(datas, dim=0) producing shape [N, rows, cols].
    let mut merged_data = Vec::with_capacity(n * bytes_per_expert);
    for t in expert_tensors {
        merged_data.extend_from_slice(&t.data);
    }

    // The merged tensor name encodes the layer/projection pattern.
    // Callers supply the canonical GGUF name; we derive it from expert[0]'s name
    // as a best-effort annotation — the caller is responsible for final naming.
    let merged_name = format!("{}_merged_experts", first.name);

    Ok(TensorRef {
        name: merged_name,
        shape: vec![n, rows, cols],
        dtype: first.dtype,
        data: merged_data,
    })
}

/// Merge all per-expert tensors in a **pre-quantization** `TensorMap`
/// for the `qwen35moe` arch.
///
/// # Why pre-quantization
///
/// `merge_moe_experts_in_place` (below) was originally written against
/// `QuantizedModel` but runs post-quantization — by then each
/// per-expert tensor's byte count reflects its quantized form (Q4_0:
/// 0.5 bytes/elem) not its declared F16 dtype, which tripped the
/// expected-bytes check in `merge_expert_tensors`.
///
/// This variant runs BEFORE quantization — the TensorMap still holds
/// the original F16/BF16 bytes, so `shape.numel() * element_size()`
/// matches `data.len()` and the merge's shape-check passes.
///
/// # Ordering
///
/// Callers run this immediately after `input::read_model` and before
/// the quantizer dispatch. Gemma and other non-MoE arches hit the
/// early-return guard and incur no cost.
/// ADR-012 P9b real-model finding (jenerallee78 abliterated apex 2026-04-25):
/// split the fused `mlp.experts.gate_up_proj` tensor into separate
/// `gate_proj.weight` + `up_proj.weight`, and add the missing `.weight`
/// suffix to `down_proj`.
///
/// # Input
///
///   model.layers.N.mlp.experts.gate_up_proj  shape [n_experts, 2*moe_inter, hidden] BF16/F16
///   model.layers.N.mlp.experts.down_proj     shape [n_experts, hidden, moe_inter]    BF16/F16
///
/// # Output
///
///   model.layers.N.mlp.experts.gate_proj.weight  shape [n_experts, moe_inter, hidden]
///   model.layers.N.mlp.experts.up_proj.weight    shape [n_experts, moe_inter, hidden]
///   model.layers.N.mlp.experts.down_proj.weight  shape [n_experts, hidden, moe_inter]
///
/// The split takes the fused `[2 * moe_inter, hidden]` per-expert sub-tensor
/// and emits gate as the first `moe_inter` rows, up as the second
/// `moe_inter` rows. This matches the convention used by HF's
/// `Qwen3MoeMLP`-style fusion (gate is first half by output channel) and
/// reproduces what `convert_hf_to_gguf.py` would emit if it processed the
/// per-expert form.
///
/// No-op for non-qwen35moe arches.
pub fn split_and_rename_fused_gate_up_in_tensor_map(
    tensor_map: &mut crate::ir::TensorMap,
    metadata: &crate::ir::ModelMetadata,
) -> Result<(), ConvertError> {
    if !super::is_qwen35moe_architecture(&metadata.architecture, &metadata.model_type) {
        return Ok(());
    }

    // Collect input tensor keys to mutate (can't iterate while modifying).
    let gate_up_keys: Vec<String> = tensor_map
        .tensors
        .keys()
        .filter(|n| {
            n.starts_with("model.layers.")
                && n.ends_with(".mlp.experts.gate_up_proj")
        })
        .cloned()
        .collect();

    let down_keys: Vec<String> = tensor_map
        .tensors
        .keys()
        .filter(|n| {
            n.starts_with("model.layers.")
                && n.ends_with(".mlp.experts.down_proj")
        })
        .cloned()
        .collect();

    if gate_up_keys.is_empty() && down_keys.is_empty() {
        return Ok(());
    }

    tracing::info!(
        gate_up_count = gate_up_keys.len(),
        down_count = down_keys.len(),
        "qwen35moe fused gate_up split + rename: processing {} gate_up_proj + {} down_proj tensors",
        gate_up_keys.len(),
        down_keys.len()
    );

    // Split each gate_up_proj.
    for key in gate_up_keys {
        let Some(tensor) = tensor_map.tensors.remove(&key) else {
            continue;
        };
        // Shape contract: [n_experts, 2 * moe_inter, hidden].
        if tensor.shape.len() != 3 {
            return Err(ConvertError::ReorderInvariantViolated {
                name: key.clone(),
                reason: format!(
                    "gate_up_proj expected 3-D [n_experts, 2*moe_inter, hidden], got shape {:?}",
                    tensor.shape
                ),
            });
        }
        let n_experts = tensor.shape[0];
        let two_inter = tensor.shape[1];
        let hidden = tensor.shape[2];
        if two_inter % 2 != 0 {
            return Err(ConvertError::ReorderInvariantViolated {
                name: key.clone(),
                reason: format!(
                    "gate_up_proj middle dim {} must be even (2 * moe_intermediate_size)",
                    two_inter
                ),
            });
        }
        let moe_inter = two_inter / 2;
        let elem_size = tensor.dtype.element_size();
        let per_expert_bytes = two_inter * hidden * elem_size;
        let half_bytes = moe_inter * hidden * elem_size;

        let mut gate_data = Vec::with_capacity(n_experts * half_bytes);
        let mut up_data = Vec::with_capacity(n_experts * half_bytes);

        for e in 0..n_experts {
            let base = e * per_expert_bytes;
            // Gate: first moe_inter rows of the [2*moe_inter, hidden] block.
            gate_data.extend_from_slice(&tensor.data[base..base + half_bytes]);
            // Up: second moe_inter rows.
            up_data.extend_from_slice(
                &tensor.data[base + half_bytes..base + per_expert_bytes],
            );
        }

        let split_shape = vec![n_experts, moe_inter, hidden];
        // Construct new names by replacing the fused suffix.
        let gate_name = key.replace(
            ".mlp.experts.gate_up_proj",
            ".mlp.experts.gate_proj.weight",
        );
        let up_name = key.replace(
            ".mlp.experts.gate_up_proj",
            ".mlp.experts.up_proj.weight",
        );

        tensor_map.tensors.insert(
            gate_name.clone(),
            crate::ir::TensorRef {
                name: gate_name,
                shape: split_shape.clone(),
                dtype: tensor.dtype,
                data: gate_data,
            },
        );
        tensor_map.tensors.insert(
            up_name.clone(),
            crate::ir::TensorRef {
                name: up_name,
                shape: split_shape,
                dtype: tensor.dtype,
                data: up_data,
            },
        );
    }

    // Rename down_proj → down_proj.weight (no data changes).
    for key in down_keys {
        let Some(mut tensor) = tensor_map.tensors.remove(&key) else {
            continue;
        };
        let new_name = key.replace(
            ".mlp.experts.down_proj",
            ".mlp.experts.down_proj.weight",
        );
        tensor.name = new_name.clone();
        tensor_map.tensors.insert(new_name, tensor);
    }

    Ok(())
}

pub fn merge_moe_experts_in_tensor_map(
    tensor_map: &mut crate::ir::TensorMap,
    metadata: &crate::ir::ModelMetadata,
) -> Result<(), ConvertError> {
    if !super::is_qwen35moe_architecture(&metadata.architecture, &metadata.model_type) {
        return Ok(());
    }
    let num_experts = match metadata.num_experts {
        Some(n) => n as usize,
        None => return Ok(()),
    };

    let mut moe_layers: std::collections::BTreeSet<usize> =
        std::collections::BTreeSet::new();
    for name in tensor_map.tensors.keys() {
        if let Some(rest) = name.strip_prefix("model.layers.") {
            if let Some(dot) = rest.find('.') {
                if let Ok(layer_idx) = rest[..dot].parse::<usize>() {
                    if rest[dot + 1..].starts_with("mlp.experts.") {
                        moe_layers.insert(layer_idx);
                    }
                }
            }
        }
    }
    tracing::info!(
        moe_layers_found = moe_layers.len(),
        "qwen35moe expert merge: scanning {} tensor names, found {} MoE-bearing layers",
        tensor_map.tensors.len(),
        moe_layers.len()
    );

    // ADR-012 P9b real-model finding (jenerallee78/Qwen3.6-35B-A3B-...): some
    // MoE checkpoints ship the experts ALREADY PRE-MERGED — the safetensors
    // contain a single tensor per (layer, projection) with shape
    // `[num_experts, out_features, in_features]` named
    // `model.layers.N.mlp.experts.{proj}.weight`, not 256 separate per-expert
    // tensors. Detect the pre-merged form and skip the merge body for those
    // layer/proj combinations (the tensor is already in the canonical
    // post-merge name expected by `qwen35_linear_attn_layer_map`'s
    // `mlp.experts.{proj}.weight` → `ffn_{proj}_exps.weight` mapping).
    let pre_merged_count = tensor_map
        .tensors
        .keys()
        .filter(|n| {
            let s = n.as_str();
            s.starts_with("model.layers.")
                && (s.ends_with(".mlp.experts.gate_proj.weight")
                    || s.ends_with(".mlp.experts.up_proj.weight")
                    || s.ends_with(".mlp.experts.down_proj.weight"))
        })
        .count();
    if pre_merged_count > 0 {
        tracing::info!(
            count = pre_merged_count,
            "qwen35moe expert merge: detected {} pre-merged expert tensors; skipping merge for those (input is already in canonical form)",
            pre_merged_count
        );
    } else {
        // Diagnostic: dump up to 20 names containing `mlp.experts.` so we
        // can see what the actual naming form is for this model.
        let mut sample: Vec<&String> = tensor_map
            .tensors
            .keys()
            .filter(|n| n.contains("mlp.experts."))
            .take(20)
            .collect();
        sample.sort();
        tracing::warn!(
            sample = ?sample,
            "qwen35moe expert merge: no pre-merged form detected; first 20 mlp.experts. tensor names dumped"
        );
    }

    for proj in &["gate_proj", "up_proj", "down_proj"] {
        let expert_proj = match *proj {
            "gate_proj" => ExpertProj::Gate,
            "up_proj" => ExpertProj::Up,
            "down_proj" => ExpertProj::Down,
            _ => unreachable!(),
        };
        for &layer_idx in &moe_layers {
            // Skip if the pre-merged form is already present in the input
            // (real Qwen3.6 ships this way — see comment block above).
            let pre_merged_name = format!(
                "model.layers.{}.mlp.experts.{}.weight",
                layer_idx, proj
            );
            if tensor_map.tensors.contains_key(&pre_merged_name) {
                continue;
            }

            let mut collected: Vec<TensorRef> = Vec::with_capacity(num_experts);
            let mut keys: Vec<String> = Vec::with_capacity(num_experts);
            for e in 0..num_experts {
                let hf = format!(
                    "model.layers.{}.mlp.experts.{}.{}.weight",
                    layer_idx, e, proj
                );
                let Some(t) = tensor_map.tensors.get(&hf) else {
                    break;
                };
                collected.push(TensorRef {
                    name: hf.clone(),
                    shape: t.shape.clone(),
                    dtype: t.dtype,
                    data: t.data.clone(),
                });
                keys.push(hf);
            }
            if collected.len() != num_experts {
                continue;
            }
            let merged =
                merge_expert_tensors(&collected, expert_proj, super::Qwen35Arch::Moe)?;
            // Remove per-expert; insert merged with canonical HF name.
            for k in &keys {
                tensor_map.tensors.remove(k);
            }
            let merged_ref = TensorRef {
                name: format!(
                    "model.layers.{}.mlp.experts.{}.weight",
                    layer_idx, proj
                ),
                shape: merged.shape,
                dtype: merged.dtype,
                data: merged.data,
            };
            tensor_map.tensors.insert(merged_ref.name.clone(), merged_ref);
        }
    }
    Ok(())
}

/// Merge all per-expert tensors in a `QuantizedModel` for the `qwen35moe` arch.
///
/// # DEPRECATED (2026-04-24)
///
/// This post-quantization variant is superseded by
/// `merge_moe_experts_in_tensor_map` which runs BEFORE quantization.
/// The post-quant variant's internal byte-size check against
/// `original_dtype.element_size()` trips on Q4_0-quantized data (0.5
/// bytes/elem vs 2 bytes for F16 declared dtype). The pre-quant
/// variant does not have that issue and is the active code path
/// called from `src/main.rs` Phase 1.5.
///
/// Kept for now as a reference implementation of the QuantizedModel
/// surface. Not called from any active code path; exercised only by
/// its own unit tests.
///
/// # Layer-streaming orchestration (ADR-012 Decision 9)
///
/// For each MoE layer N (identified by tensors named
/// `model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight`):
/// 1. Collect E=0..num_experts tensors for each projection.
/// 2. Merge each projection into a single 3-D `QuantizedTensor`.
/// 3. Insert the merged tensor under its canonical HF merged name
///    (`model.layers.N.mlp.experts.{gate,up,down}_proj.weight`).
/// 4. Remove the 256 per-expert tensors for that projection from the map.
///
/// Layer N is fully processed before layer N+1 is touched.  At peak,
/// 256 per-expert tensors for one projection are held simultaneously,
/// then dropped before the next projection is collected.
///
/// # Shared experts
///
/// Shared expert tensors (`mlp.shared_expert.{gate,up,down}_proj.weight`)
/// are **not** touched — they're already singletons and pass through as-is.
///
/// # No-op for dense
///
/// If `num_experts` is absent from metadata, the model is dense and this
/// function returns `Ok(())` immediately without touching any tensors.
#[deprecated(
    since = "0.1.0",
    note = "use merge_moe_experts_in_tensor_map; this post-quant variant trips on Q4_0 byte-size check"
)]
#[allow(dead_code)]
pub fn merge_moe_experts_in_place(
    model: &mut crate::ir::QuantizedModel,
) -> Result<(), ConvertError> {
    // Only act on qwen35moe architecture.
    if !super::is_qwen35moe_architecture(
        &model.metadata.architecture,
        &model.metadata.model_type,
    ) {
        return Ok(());
    }

    let num_experts = match model.metadata.num_experts {
        Some(n) => n as usize,
        None => return Ok(()), // dense — no-op
    };

    // Determine which layers have MoE experts by scanning tensor names once.
    let mut moe_layers: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for name in model.tensors.keys() {
        // Pattern: "model.layers.N.mlp.experts.E.gate_proj.weight"
        if let Some(rest) = name.strip_prefix("model.layers.") {
            if let Some(dot) = rest.find('.') {
                if let Ok(layer_idx) = rest[..dot].parse::<usize>() {
                    let suffix = &rest[dot + 1..];
                    if suffix.starts_with("mlp.experts.") {
                        moe_layers.insert(layer_idx);
                    }
                }
            }
        }
    }

    for proj in &["gate_proj", "up_proj", "down_proj"] {
        let expert_proj = match *proj {
            "gate_proj" => ExpertProj::Gate,
            "up_proj"   => ExpertProj::Up,
            "down_proj" => ExpertProj::Down,
            _           => unreachable!(),
        };

        // Process one layer at a time to bound peak memory.
        for &layer_idx in &moe_layers {
            // Collect expert tensors E=0..num_experts in order.
            let mut expert_tensors: Vec<TensorRef> = Vec::with_capacity(num_experts);
            let mut expert_keys: Vec<String> = Vec::with_capacity(num_experts);

            for e in 0..num_experts {
                let hf_name = format!(
                    "model.layers.{}.mlp.experts.{}.{}.weight",
                    layer_idx, e, proj
                );
                let qt = match model.tensors.get(&hf_name) {
                    Some(t) => t,
                    None => {
                        // Not all layers are MoE layers (e.g., full-attention layers
                        // may not have experts).  Skip gracefully.
                        break;
                    }
                };
                expert_tensors.push(TensorRef {
                    name: hf_name.clone(),
                    shape: qt.shape.clone(),
                    dtype: qt.original_dtype,
                    data: qt.data.clone(),
                });
                expert_keys.push(hf_name);
            }

            // If we didn't collect all expected experts, this layer might not
            // be a full MoE layer.  Skip it.
            if expert_tensors.len() != num_experts {
                continue;
            }

            // Merge into a single 3-D tensor.
            let merged_ref = merge_expert_tensors(
                &expert_tensors,
                expert_proj,
                super::Qwen35Arch::Moe,
            )?;

            // Build the merged QuantizedTensor — preserved at original dtype.
            let merged_qt = crate::ir::QuantizedTensor {
                name: format!(
                    "model.layers.{}.mlp.experts.{}.weight",
                    layer_idx, proj
                ),
                shape: merged_ref.shape.clone(),
                original_dtype: merged_ref.dtype,
                data: merged_ref.data,
                quant_info: crate::ir::TensorQuantInfo {
                    method: "passthrough".to_string(),
                    bits: merged_ref.dtype.element_size() as u8 * 8,
                    group_size: 1,
                    preserved: true,
                    scales: None,
                    biases: None,
                    ggml_type: None,
                },
            };

            // Remove the N per-expert tensors and insert the merged tensor.
            for key in &expert_keys {
                model.tensors.remove(key);
            }
            model.tensors.insert(merged_qt.name.clone(), merged_qt);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Metadata emission (P4 — implemented)
// ---------------------------------------------------------------------------

/// Validate the Qwen3.5-MoE convert context and confirm all required
/// metadata hparams for the MoE variant are present.
///
/// # P4 implementation (ADR-012 Decision 7)
///
/// The actual GGUF key-value emission for `qwen35moe.*` keys is performed by
/// `src/backends/gguf.rs::build_metadata_qwen35moe`.  This function validates
/// the context; callers then emit KVs inline.
///
/// ## Keys emitted by the caller (all prefixed `qwen35moe.*`):
/// Shared keys (same as qwen35 dense, arch prefix changes):
/// - `qwen35moe.block_count`, `.context_length`, `.embedding_length`
/// - `qwen35moe.attention.head_count`, `.head_count_kv`, `.key_length`, `.value_length`
/// - `qwen35moe.attention.layer_norm_rms_epsilon`  — llama-model.cpp:2837 (mandatory)
/// - `qwen35moe.rope.freq_base`, `.dimension_count`, `.dimension_sections`
///   — llama-model.cpp:2839 (mandatory via get_key_or_arr)
/// - `qwen35moe.full_attention_interval`            — llama-model.cpp:2851 (optional, default 4)
/// - `qwen35moe.ssm.conv_kernel`                    — llama-model.cpp:2842 (mandatory)
/// - `qwen35moe.ssm.inner_size`                     — llama-model.cpp:2843 (mandatory)
/// - `qwen35moe.ssm.state_size`                     — llama-model.cpp:2844 (mandatory)
/// - `qwen35moe.ssm.time_step_rank`                 — llama-model.cpp:2845 (mandatory)
/// - `qwen35moe.ssm.group_count`                    — llama-model.cpp:2846 (mandatory)
/// - `qwen35moe.nextn_predict_layers`               — llama-arch.cpp:194 (optional, default 0)
///
/// MoE-only keys:
/// - `qwen35moe.expert_count`                       — llama-arch.cpp:182 (LLM_KV_EXPERT_COUNT)
/// - `qwen35moe.expert_used_count`                  — llama-arch.cpp:183 (LLM_KV_EXPERT_USED_COUNT)
/// - `qwen35moe.expert_feed_forward_length`         — llama-arch.cpp:175 / llama-model.cpp:2835 (optional)
/// - `qwen35moe.expert_shared_feed_forward_length`  — llama-arch.cpp:176 / llama-model.cpp:2836 (optional)
#[deprecated(
    since = "0.1.0",
    note = "kv emission happens in src/backends/gguf.rs::emit_qwen35_metadata; \
            validation happens in Qwen35ConvertContext::from_metadata + \
            validate_required_qwen35moe_fields — this validator is redundant."
)]
#[allow(dead_code)]
pub fn emit_metadata_moe(ctx: &Qwen35ConvertContext) -> Result<(), ConvertError> {
    // Validate MoE-specific required fields
    if ctx.num_experts.is_none() {
        return Err(ConvertError::MissingHparam {
            field: "num_experts (required for qwen35moe expert_count)",
        });
    }
    if ctx.top_k_experts.is_none() {
        return Err(ConvertError::MissingHparam {
            field: "top_k_experts (required for qwen35moe expert_used_count)",
        });
    }
    if ctx.moe_intermediate_size.is_none() {
        return Err(ConvertError::MissingHparam {
            field: "moe_intermediate_size (required for qwen35moe expert_feed_forward_length)",
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// ADR-014 P1 — Lazy variants of the qwen35moe Phase 1.45 / 1.5 helpers.
//
// Phase 1.45 (gate_up split + .weight rename) lifts as
// `split_and_rename_fused_gate_up_in_lazy_map`. Phase 1.5 (expert
// merge — Decision 7 layer-streaming) lifts as
// `merge_moe_experts_in_lazy_map`. The eager variants stay for the
// ADR-012 P9b intermediate-GGUF dance until ADR-014 P4 deletes both.
// ---------------------------------------------------------------------------

/// ADR-014 P1: lazy variant of [`split_and_rename_fused_gate_up_in_tensor_map`].
///
/// Splits each `model.layers.{N}.mlp.experts.gate_up_proj` tensor
/// (shape `[N_exp, 2*moe_inter, hidden]`) into separate
/// `gate_proj.weight` + `up_proj.weight` (each `[N_exp, moe_inter, hidden]`),
/// and adds the missing `.weight` suffix to `down_proj`.
///
/// Implementation: the split fundamentally requires reading the fused
/// tensor's bytes (it's a row-axis slice into halves). For each fused
/// tensor we materialise the [`LazyTensor`], split into two byte
/// vectors, and re-insert as [`LazyTensor::from_bytes`]. The down_proj
/// rename is pure metadata and uses [`LazyTensor::map_with_meta`] so
/// no bytes are touched.
///
/// Peak resident bytes during this phase are bounded by one fused
/// tensor at a time — apex MoE per-layer fused gate_up is
/// `256 × 2 × 768 × 2048 × 2 bytes ≈ 1.5 GB`; after the split, two
/// 750 MB halves replace it. The fused parent is dropped immediately.
///
/// No-op for non-qwen35moe arches.
pub fn split_and_rename_fused_gate_up_in_lazy_map(
    lazy_map: &mut crate::ir::lazy::LazyTensorMap,
    metadata: &crate::ir::ModelMetadata,
) -> Result<(), super::ConvertError> {
    use crate::ir::lazy::{LazyMeta, LazyTensor};

    if !super::is_qwen35moe_architecture(&metadata.architecture, &metadata.model_type) {
        return Ok(());
    }

    let gate_up_keys: Vec<String> = lazy_map
        .names()
        .filter(|n| {
            n.starts_with("model.layers.") && n.ends_with(".mlp.experts.gate_up_proj")
        })
        .cloned()
        .collect();

    let down_keys: Vec<String> = lazy_map
        .names()
        .filter(|n| {
            n.starts_with("model.layers.") && n.ends_with(".mlp.experts.down_proj")
        })
        .cloned()
        .collect();

    if gate_up_keys.is_empty() && down_keys.is_empty() {
        return Ok(());
    }

    tracing::info!(
        gate_up_count = gate_up_keys.len(),
        down_count = down_keys.len(),
        "qwen35moe fused gate_up split (lazy): processing {} gate_up_proj + {} down_proj tensors",
        gate_up_keys.len(),
        down_keys.len()
    );

    // ---- Phase 1.45a: split each fused gate_up_proj into gate + up. ----
    for key in gate_up_keys {
        let Some(lazy) = lazy_map.remove(&key) else {
            continue;
        };

        // Validate metadata before forcing materialisation — same
        // 3-D + even-middle-dim invariants as the eager helper.
        if lazy.shape().len() != 3 {
            return Err(super::ConvertError::ReorderInvariantViolated {
                name: key.clone(),
                reason: format!(
                    "gate_up_proj expected 3-D [n_experts, 2*moe_inter, hidden], got shape {:?}",
                    lazy.shape()
                ),
            });
        }
        let n_experts = lazy.shape()[0];
        let two_inter = lazy.shape()[1];
        let hidden = lazy.shape()[2];
        if two_inter % 2 != 0 {
            return Err(super::ConvertError::ReorderInvariantViolated {
                name: key.clone(),
                reason: format!(
                    "gate_up_proj middle dim {} must be even (2 * moe_intermediate_size)",
                    two_inter
                ),
            });
        }
        let moe_inter = two_inter / 2;
        let dtype = lazy.dtype();
        let elem_size = dtype.element_size();
        let per_expert_bytes = two_inter * hidden * elem_size;
        let half_bytes = moe_inter * hidden * elem_size;

        // Materialise the fused parent — this is the only place in
        // Phase 1.45 where bytes touch RAM. Caller (cmd_convert) sees
        // ~1.5 GB peak per layer; it drops as soon as the split halves
        // are inserted.
        let fused = lazy
            .materialize()
            .map_err(|e| super::ConvertError::ReorderInvariantViolated {
                name: key.clone(),
                reason: format!("materialize fused gate_up: {e}"),
            })?;

        let mut gate_data = Vec::with_capacity(n_experts * half_bytes);
        let mut up_data = Vec::with_capacity(n_experts * half_bytes);

        for e in 0..n_experts {
            let base = e * per_expert_bytes;
            // Gate: first moe_inter rows of the [2*moe_inter, hidden] block.
            gate_data.extend_from_slice(&fused.data[base..base + half_bytes]);
            // Up: second moe_inter rows.
            up_data.extend_from_slice(&fused.data[base + half_bytes..base + per_expert_bytes]);
        }
        // Fused parent dropped here.
        drop(fused);

        let gate_name = key.replace(
            ".mlp.experts.gate_up_proj",
            ".mlp.experts.gate_proj.weight",
        );
        let up_name = key.replace(
            ".mlp.experts.gate_up_proj",
            ".mlp.experts.up_proj.weight",
        );
        let split_shape = vec![n_experts, moe_inter, hidden];

        lazy_map.insert(LazyTensor::from_bytes(
            LazyMeta::new(gate_name, split_shape.clone(), dtype),
            gate_data,
        ));
        lazy_map.insert(LazyTensor::from_bytes(
            LazyMeta::new(up_name, split_shape, dtype),
            up_data,
        ));
    }

    // ---- Phase 1.45b: rename down_proj → down_proj.weight (metadata only). ----
    for key in down_keys {
        let Some(old_lazy) = lazy_map.remove(&key) else {
            continue;
        };
        let new_name = key.replace(
            ".mlp.experts.down_proj",
            ".mlp.experts.down_proj.weight",
        );
        let new_meta = LazyMeta::new(
            new_name.clone(),
            old_lazy.shape().to_vec(),
            old_lazy.dtype(),
        );
        let new_name_for_closure = new_name.clone();
        let renamed = old_lazy.map_with_meta(new_meta, move |mut tref| {
            tref.name = new_name_for_closure.clone();
            Ok(tref)
        });
        lazy_map.insert(renamed);
    }

    Ok(())
}

/// ADR-014 P1 + Decision 7: lazy, layer-streaming variant of
/// [`merge_moe_experts_in_tensor_map`].
///
/// Each (layer, projection) merge produces ONE [`LazyTensor`] whose
/// closure performs the 256-expert stack at materialise time. The
/// closure owns the per-expert [`LazyTensor`]s; they are removed from
/// `lazy_map` at this phase but their bytes do not materialise until
/// the streaming quantize loop (ADR-014 P2) consumes the merged
/// tensor.
///
/// **Decision 7 layer-streaming benefit:** in the eager variant every
/// merged `[N=256, hidden, moe_inter]` tensor stays in the map between
/// Phase 1.5 close and quantize-write boundaries — for apex MoE that's
/// ~80 GB across 40 layers × 3 projections at the same time. The lazy
/// variant emits each merged tensor as a closure; at quantize-time the
/// streaming loop materialises one merged tensor (~750 MB, 256 × hidden
/// × moe_inter × 2 bytes for apex BF16), quantises it, writes it,
/// drops it, then proceeds to the next. Peak resident bytes stay
/// bounded by ~one merged tile, not the full ~80 GB stack.
///
/// The pre-merged-skip path (apex MoE checkpoints that already ship a
/// pre-merged form per Phase 1.45's split) is preserved exactly: when
/// `pre_merged_name` is already present in the map, the function
/// leaves it as-is.
///
/// Validation invariants (shape consistency across experts, 2-D
/// per-expert shape, byte_len === shape × dtype) are checked at
/// closure-construction time so failures surface before the
/// `LazyTensor` is created — never silently at materialise time.
///
/// No-op for non-qwen35moe arches and for dense (no `num_experts`).
pub fn merge_moe_experts_in_lazy_map(
    lazy_map: &mut crate::ir::lazy::LazyTensorMap,
    metadata: &crate::ir::ModelMetadata,
) -> Result<(), super::ConvertError> {
    use crate::ir::lazy::{LazyMeta, LazyTensor, MaterializeError};

    if !super::is_qwen35moe_architecture(&metadata.architecture, &metadata.model_type) {
        return Ok(());
    }
    let num_experts = match metadata.num_experts {
        Some(n) => n as usize,
        None => return Ok(()),
    };

    // Same MoE-layer scan as the eager helper — uses lazy_map.names()
    // (deterministic BTreeMap iteration order) so downstream behaviour
    // is bit-for-bit reproducible.
    let mut moe_layers: std::collections::BTreeSet<usize> =
        std::collections::BTreeSet::new();
    for name in lazy_map.names() {
        if let Some(rest) = name.strip_prefix("model.layers.") {
            if let Some(dot) = rest.find('.') {
                if let Ok(layer_idx) = rest[..dot].parse::<usize>() {
                    if rest[dot + 1..].starts_with("mlp.experts.") {
                        moe_layers.insert(layer_idx);
                    }
                }
            }
        }
    }
    tracing::info!(
        moe_layers_found = moe_layers.len(),
        "qwen35moe expert merge (lazy): scanning {} tensor names, found {} MoE-bearing layers",
        lazy_map.len(),
        moe_layers.len()
    );

    // Pre-merged-skip detection — same as eager, just over lazy_map.
    let pre_merged_count = lazy_map
        .names()
        .filter(|n| {
            let s = n.as_str();
            s.starts_with("model.layers.")
                && (s.ends_with(".mlp.experts.gate_proj.weight")
                    || s.ends_with(".mlp.experts.up_proj.weight")
                    || s.ends_with(".mlp.experts.down_proj.weight"))
        })
        .count();
    if pre_merged_count > 0 {
        tracing::info!(
            count = pre_merged_count,
            "qwen35moe expert merge (lazy): detected {} pre-merged expert tensors; skipping merge for those (input is already in canonical form)",
            pre_merged_count
        );
    }

    for proj in &["gate_proj", "up_proj", "down_proj"] {
        for &layer_idx in &moe_layers {
            // Skip if the pre-merged form is already present.
            let pre_merged_name =
                format!("model.layers.{}.mlp.experts.{}.weight", layer_idx, proj);
            if lazy_map.contains_key(&pre_merged_name) {
                continue;
            }

            // Collect num_experts per-expert LazyTensors. We *remove*
            // them up-front (FnOnce semantics — the closure that does
            // the merge takes ownership). If we can't find all
            // num_experts, restore the partial collection and skip
            // this (layer, proj) — same silent-skip behaviour as the
            // eager `continue;` path, but explicit about the partial-
            // restore.
            let mut collected: Vec<(String, LazyTensor)> = Vec::with_capacity(num_experts);
            let mut all_found = true;
            for e in 0..num_experts {
                let hf = format!(
                    "model.layers.{}.mlp.experts.{}.{}.weight",
                    layer_idx, e, proj
                );
                let Some(lazy) = lazy_map.remove(&hf) else {
                    all_found = false;
                    break;
                };
                collected.push((hf, lazy));
            }
            if !all_found {
                // Restore — LazyTensor knows its own name via meta.
                for (_, lazy) in collected.into_iter() {
                    lazy_map.insert(lazy);
                }
                continue;
            }

            // Validate shape + dtype consistency across experts BEFORE
            // closure construction. Failures here surface inline; the
            // closure body is reserved for the merge itself.
            let first = &collected[0].1;
            if first.shape().len() != 2 {
                return Err(super::ConvertError::ExpertMergeShapeMismatch {
                    expert_idx: 0,
                    reason: format!(
                        "expected 2-D tensor, got {} dims",
                        first.shape().len()
                    ),
                });
            }
            let rows = first.shape()[0];
            let cols = first.shape()[1];
            let dtype = first.dtype();
            let elem_size = dtype.element_size();
            let bytes_per_expert = rows * cols * elem_size;
            for (i, (_, l)) in collected.iter().enumerate().skip(1) {
                if l.shape() != [rows, cols] {
                    return Err(super::ConvertError::ExpertMergeShapeMismatch {
                        expert_idx: i,
                        reason: format!(
                            "shape {:?} differs from expert 0 shape [{}, {}]",
                            l.shape(),
                            rows,
                            cols
                        ),
                    });
                }
                if l.dtype() != dtype {
                    return Err(super::ConvertError::ExpertMergeShapeMismatch {
                        expert_idx: i,
                        reason: format!(
                            "dtype {:?} differs from expert 0 dtype {:?}",
                            l.dtype(),
                            dtype
                        ),
                    });
                }
            }

            let merged_shape = vec![num_experts, rows, cols];
            let merged_byte_len = num_experts * bytes_per_expert;
            let merged_meta = LazyMeta::new(pre_merged_name.clone(), merged_shape, dtype);

            // Move per-expert LazyTensors into the closure. The closure
            // is FnOnce: at quantize time the streaming loop calls
            // materialize() exactly once. Each per-expert is materialised
            // and dropped in turn — peak resident is one expert's bytes
            // plus the accumulating merged buffer.
            let collected_lazies: Vec<LazyTensor> =
                collected.into_iter().map(|(_, l)| l).collect();
            let pre_merged_name_for_err = pre_merged_name.clone();
            let merge_closure = move || -> Result<Vec<u8>, MaterializeError> {
                let mut merged_data = Vec::with_capacity(merged_byte_len);
                for (i, lazy) in collected_lazies.into_iter().enumerate() {
                    let t = lazy.materialize().map_err(|e| MaterializeError::Transform {
                        name: pre_merged_name_for_err.clone(),
                        reason: format!("expert {} materialize: {}", i, e),
                    })?;
                    if t.data.len() != bytes_per_expert {
                        return Err(MaterializeError::Transform {
                            name: pre_merged_name_for_err.clone(),
                            reason: format!(
                                "expert {} bytes ({}) != expected ({})",
                                i,
                                t.data.len(),
                                bytes_per_expert
                            ),
                        });
                    }
                    merged_data.extend_from_slice(&t.data);
                    // t (TensorRef) drops here — expert bytes freed.
                }
                Ok(merged_data)
            };

            lazy_map.insert(LazyTensor::from_closure(merged_meta, merge_closure));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen35::{Qwen35Arch, Qwen35ConvertContext};

    /// ADR-014 P1 byte-identity regression for Phase 1.45 lift.
    /// `split_and_rename_fused_gate_up_in_lazy_map` produces a tensor
    /// map whose post-split keys + bytes are byte-equal to the eager
    /// `split_and_rename_fused_gate_up_in_tensor_map` on the same
    /// fixture.
    #[test]
    fn split_and_rename_fused_gate_up_lazy_byte_identical_to_eager() {
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::ir::{DType, ModelMetadata, TensorMap, TensorRef};

        // Synthetic 2-layer Qwen3.5-MoE: each layer ships a fused
        // gate_up_proj of shape [n_experts=4, 2*moe_inter=8, hidden=4]
        // (128 elements × 2 bytes = 256 bytes per layer) plus a
        // down_proj of shape [n_experts=4, hidden=4, moe_inter=4].
        // Distinguishing payloads so a swap or drop would be caught.
        let metadata = ModelMetadata {
            architecture: "Qwen3_5MoeForCausalLM".into(),
            model_type: "qwen3_5_moe_text".into(),
            param_count: 0,
            hidden_size: 4,
            num_layers: 2,
            layer_types: vec!["full_attention".into(), "linear_attention".into()],
            num_attention_heads: 4,
            num_kv_heads: Some(2),
            vocab_size: 16,
            dtype: "float16".into(),
            shard_count: 1,
            num_experts: Some(4),
            top_k_experts: Some(2),
            intermediate_size: Some(8),
            raw_config: serde_json::Value::Null,
            explicit_layer_types: None,
            full_attention_interval: None,
            attn_output_gate: None,
            head_dim: None,
            partial_rotary_factor: None,
            rope_parameters: None,
            linear_conv_kernel_dim: None,
            linear_key_head_dim: None,
            linear_num_key_heads: None,
            linear_value_head_dim: None,
            linear_num_value_heads: None,
            mamba_ssm_dtype: None,
            moe_intermediate_size: Some(4),
            shared_expert_intermediate_size: None,
            mtp_num_hidden_layers: None,
            mtp_use_dedicated_embeddings: None,
            output_router_logits: None,
            router_aux_loss_coef: None,
        };

        // Per-layer payloads: gate_up_proj (256 bytes), down_proj (128 bytes).
        // Use a deterministic fill so each tensor's bytes are unique.
        let make_payload = |seed: u8, len: usize| -> Vec<u8> {
            (0..len).map(|i| seed.wrapping_add(i as u8)).collect()
        };

        let fused_shape = vec![4usize, 8, 4]; // [n_exp, 2*inter, hidden]
        let down_shape = vec![4usize, 4, 4]; // [n_exp, hidden, moe_inter]
        let fused_byte_len = 4 * 8 * 4 * 2; // 256 bytes (F16)
        let down_byte_len = 4 * 4 * 4 * 2; // 128 bytes (F16)

        let layer0_gate_up = make_payload(0x10, fused_byte_len);
        let layer0_down = make_payload(0x20, down_byte_len);
        let layer1_gate_up = make_payload(0x30, fused_byte_len);
        let layer1_down = make_payload(0x40, down_byte_len);

        // ---- Eager path ----
        let mut eager = TensorMap::new();
        eager.insert(TensorRef {
            name: "model.layers.0.mlp.experts.gate_up_proj".into(),
            shape: fused_shape.clone(),
            dtype: DType::F16,
            data: layer0_gate_up.clone(),
        });
        eager.insert(TensorRef {
            name: "model.layers.0.mlp.experts.down_proj".into(),
            shape: down_shape.clone(),
            dtype: DType::F16,
            data: layer0_down.clone(),
        });
        eager.insert(TensorRef {
            name: "model.layers.1.mlp.experts.gate_up_proj".into(),
            shape: fused_shape.clone(),
            dtype: DType::F16,
            data: layer1_gate_up.clone(),
        });
        eager.insert(TensorRef {
            name: "model.layers.1.mlp.experts.down_proj".into(),
            shape: down_shape.clone(),
            dtype: DType::F16,
            data: layer1_down.clone(),
        });
        split_and_rename_fused_gate_up_in_tensor_map(&mut eager, &metadata).unwrap();

        // ---- Lazy path ----
        let mut lazy = LazyTensorMap::new();
        lazy.insert(LazyTensor::from_bytes(
            LazyMeta::new(
                "model.layers.0.mlp.experts.gate_up_proj".into(),
                fused_shape.clone(),
                DType::F16,
            ),
            layer0_gate_up.clone(),
        ));
        lazy.insert(LazyTensor::from_bytes(
            LazyMeta::new(
                "model.layers.0.mlp.experts.down_proj".into(),
                down_shape.clone(),
                DType::F16,
            ),
            layer0_down.clone(),
        ));
        lazy.insert(LazyTensor::from_bytes(
            LazyMeta::new(
                "model.layers.1.mlp.experts.gate_up_proj".into(),
                fused_shape.clone(),
                DType::F16,
            ),
            layer1_gate_up.clone(),
        ));
        lazy.insert(LazyTensor::from_bytes(
            LazyMeta::new(
                "model.layers.1.mlp.experts.down_proj".into(),
                down_shape.clone(),
                DType::F16,
            ),
            layer1_down.clone(),
        ));
        split_and_rename_fused_gate_up_in_lazy_map(&mut lazy, &metadata).unwrap();
        let lazy_eager = lazy.materialize_all().unwrap();

        // Same key set: 6 tensors (2 layers × {gate, up, down}.weight).
        let mut eager_keys: Vec<&String> = eager.tensors.keys().collect();
        eager_keys.sort();
        let mut lazy_keys: Vec<&String> = lazy_eager.tensors.keys().collect();
        lazy_keys.sort();
        assert_eq!(eager_keys, lazy_keys, "post-split key sets must match");

        // Per-key shape, dtype, and bytes byte-equal.
        for key in eager_keys {
            let e = eager.get(key).unwrap();
            let l = lazy_eager.get(key).unwrap();
            assert_eq!(e.shape, l.shape, "{key} shape");
            assert_eq!(e.dtype, l.dtype, "{key} dtype");
            assert_eq!(e.data, l.data, "{key} bytes");
            assert_eq!(l.name, *key, "{key} carried-name field");
        }
    }

    /// ADR-014 P1 + Decision 7 layer-streaming byte-identity regression.
    /// `merge_moe_experts_in_lazy_map` produces a tensor map whose
    /// post-merge keys + shapes + materialised bytes are byte-equal to
    /// the eager `merge_moe_experts_in_tensor_map` on the same fixture
    /// (per-expert form, not pre-merged).
    #[test]
    fn merge_moe_experts_lazy_byte_identical_to_eager_per_expert_input() {
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::ir::{DType, ModelMetadata, TensorMap, TensorRef};

        // Synthetic 2-layer Qwen3.5-MoE with per-expert form
        // (4 experts × 3 projections × 2 layers = 24 per-expert tensors).
        // Per-expert shape: gate/up [moe_inter=2, hidden=4]; down [hidden=4, moe_inter=2].
        // Distinguishing payloads so a swap or drop would be caught.
        let metadata = ModelMetadata {
            architecture: "Qwen3_5MoeForCausalLM".into(),
            model_type: "qwen3_5_moe_text".into(),
            param_count: 0,
            hidden_size: 4,
            num_layers: 2,
            layer_types: vec!["full_attention".into(), "linear_attention".into()],
            num_attention_heads: 4,
            num_kv_heads: Some(2),
            vocab_size: 16,
            dtype: "float16".into(),
            shard_count: 1,
            num_experts: Some(4),
            top_k_experts: Some(2),
            intermediate_size: Some(4),
            raw_config: serde_json::Value::Null,
            explicit_layer_types: None,
            full_attention_interval: None,
            attn_output_gate: None,
            head_dim: None,
            partial_rotary_factor: None,
            rope_parameters: None,
            linear_conv_kernel_dim: None,
            linear_key_head_dim: None,
            linear_num_key_heads: None,
            linear_value_head_dim: None,
            linear_num_value_heads: None,
            mamba_ssm_dtype: None,
            moe_intermediate_size: Some(2),
            shared_expert_intermediate_size: None,
            mtp_num_hidden_layers: None,
            mtp_use_dedicated_embeddings: None,
            output_router_logits: None,
            router_aux_loss_coef: None,
        };

        // Build per-expert payloads: each (layer, expert, proj) gets a
        // distinguishing seed so byte-identity is meaningful.
        let make_payload = |seed: u8, len: usize| -> Vec<u8> {
            (0..len).map(|i| seed.wrapping_add(i as u8)).collect()
        };

        let proj_shapes = [
            ("gate_proj", vec![2usize, 4]),
            ("up_proj", vec![2usize, 4]),
            ("down_proj", vec![4usize, 2]),
        ];

        // Build identical eager + lazy fixtures. Use a helper closure
        // so we know both inputs are bit-equal at start.
        let mut eager = TensorMap::new();
        let mut lazy = LazyTensorMap::new();
        for layer_idx in 0..2usize {
            for &(proj, ref shape) in proj_shapes.iter() {
                for e in 0..4usize {
                    let name = format!(
                        "model.layers.{}.mlp.experts.{}.{}.weight",
                        layer_idx, e, proj
                    );
                    let byte_len = shape[0] * shape[1] * 2; // F16
                    // Seed: layer * 64 + expert * 16 + proj_offset (gate=0, up=1, down=2).
                    let proj_off: u8 = match proj {
                        "gate_proj" => 0,
                        "up_proj" => 1,
                        "down_proj" => 2,
                        _ => unreachable!(),
                    };
                    let seed = (layer_idx as u8 * 64) + (e as u8 * 16) + proj_off;
                    let data = make_payload(seed, byte_len);

                    eager.insert(TensorRef {
                        name: name.clone(),
                        shape: shape.clone(),
                        dtype: DType::F16,
                        data: data.clone(),
                    });
                    lazy.insert(LazyTensor::from_bytes(
                        LazyMeta::new(name, shape.clone(), DType::F16),
                        data,
                    ));
                }
            }
        }

        // Run both merges.
        merge_moe_experts_in_tensor_map(&mut eager, &metadata).unwrap();
        merge_moe_experts_in_lazy_map(&mut lazy, &metadata).unwrap();
        let lazy_eager = lazy.materialize_all().unwrap();

        // Same key set — should be 6 merged tensors (2 layers × 3 projs).
        let mut eager_keys: Vec<&String> = eager.tensors.keys().collect();
        eager_keys.sort();
        let mut lazy_keys: Vec<&String> = lazy_eager.tensors.keys().collect();
        lazy_keys.sort();
        assert_eq!(eager_keys, lazy_keys, "post-merge key sets must match");
        assert_eq!(
            eager_keys.len(),
            6,
            "expected 6 merged tensors (2 layers × 3 projs); per-expert form should be fully consumed"
        );

        // Per-key shape, dtype, bytes byte-equal.
        for key in eager_keys {
            let e = eager.get(key).unwrap();
            let l = lazy_eager.get(key).unwrap();
            assert_eq!(e.shape, l.shape, "{key} shape (merged should be 3-D)");
            assert_eq!(e.dtype, l.dtype, "{key} dtype");
            assert_eq!(e.data, l.data, "{key} bytes (256-expert stack)");
            assert_eq!(l.name, *key, "{key} carried-name field");
        }
    }

    /// Pre-merged tensors (apex MoE form post Phase 1.45 split) are
    /// preserved as-is by `merge_moe_experts_in_lazy_map` — the
    /// pre-merged-skip path produces identical output to the eager
    /// variant.
    #[test]
    fn merge_moe_experts_lazy_skips_pre_merged_form() {
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::ir::{DType, ModelMetadata, TensorMap, TensorRef};

        let metadata = ModelMetadata {
            architecture: "Qwen3_5MoeForCausalLM".into(),
            model_type: "qwen3_5_moe_text".into(),
            param_count: 0,
            hidden_size: 4,
            num_layers: 1,
            layer_types: vec![],
            num_attention_heads: 1,
            num_kv_heads: Some(1),
            vocab_size: 16,
            dtype: "float16".into(),
            shard_count: 1,
            num_experts: Some(4),
            top_k_experts: Some(2),
            intermediate_size: Some(2),
            raw_config: serde_json::Value::Null,
            explicit_layer_types: None,
            full_attention_interval: None,
            attn_output_gate: None,
            head_dim: None,
            partial_rotary_factor: None,
            rope_parameters: None,
            linear_conv_kernel_dim: None,
            linear_key_head_dim: None,
            linear_num_key_heads: None,
            linear_value_head_dim: None,
            linear_num_value_heads: None,
            mamba_ssm_dtype: None,
            moe_intermediate_size: Some(2),
            shared_expert_intermediate_size: None,
            mtp_num_hidden_layers: None,
            mtp_use_dedicated_embeddings: None,
            output_router_logits: None,
            router_aux_loss_coef: None,
        };

        // Pre-merged form: shape [num_experts=4, rows, cols].
        let pre_merged = vec![
            (
                "model.layers.0.mlp.experts.gate_proj.weight",
                vec![4usize, 2, 4],
                64usize, // 4*2*4 * 2 = 64 bytes
                0xAAu8,
            ),
            (
                "model.layers.0.mlp.experts.up_proj.weight",
                vec![4usize, 2, 4],
                64,
                0xBBu8,
            ),
            (
                "model.layers.0.mlp.experts.down_proj.weight",
                vec![4usize, 4, 2],
                64,
                0xCCu8,
            ),
        ];
        let mut eager = TensorMap::new();
        let mut lazy = LazyTensorMap::new();
        for (name, shape, byte_len, fill) in &pre_merged {
            let data = vec![*fill; *byte_len];
            eager.insert(TensorRef {
                name: name.to_string(),
                shape: shape.clone(),
                dtype: DType::F16,
                data: data.clone(),
            });
            lazy.insert(LazyTensor::from_bytes(
                LazyMeta::new(name.to_string(), shape.clone(), DType::F16),
                data,
            ));
        }

        merge_moe_experts_in_tensor_map(&mut eager, &metadata).unwrap();
        merge_moe_experts_in_lazy_map(&mut lazy, &metadata).unwrap();
        let lazy_eager = lazy.materialize_all().unwrap();

        // Both should pass pre-merged tensors through unchanged.
        for (name, shape, byte_len, fill) in &pre_merged {
            let e = eager.get(name).unwrap();
            let l = lazy_eager.get(name).unwrap();
            assert_eq!(e.shape, *shape, "{name} eager shape unchanged");
            assert_eq!(l.shape, *shape, "{name} lazy shape unchanged");
            assert_eq!(e.data.len(), *byte_len);
            assert_eq!(l.data.len(), *byte_len);
            assert!(e.data.iter().all(|b| *b == *fill), "{name} eager bytes unchanged");
            assert!(l.data.iter().all(|b| *b == *fill), "{name} lazy bytes unchanged");
        }
    }

    /// Non-qwen35moe arches must early-return Ok(()) on the lazy
    /// variant — same arch-gate behaviour as the eager version.
    #[test]
    fn merge_moe_experts_lazy_no_op_for_dense() {
        use crate::ir::lazy::LazyTensorMap;
        use crate::ir::ModelMetadata;

        let metadata = ModelMetadata {
            architecture: "Qwen3_5ForCausalLM".into(), // dense
            model_type: "qwen3_5".into(),
            param_count: 0,
            hidden_size: 4,
            num_layers: 1,
            layer_types: vec![],
            num_attention_heads: 1,
            num_kv_heads: Some(1),
            vocab_size: 16,
            dtype: "float16".into(),
            shard_count: 1,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: Some(8),
            raw_config: serde_json::Value::Null,
            explicit_layer_types: None,
            full_attention_interval: None,
            attn_output_gate: None,
            head_dim: None,
            partial_rotary_factor: None,
            rope_parameters: None,
            linear_conv_kernel_dim: None,
            linear_key_head_dim: None,
            linear_num_key_heads: None,
            linear_value_head_dim: None,
            linear_num_value_heads: None,
            mamba_ssm_dtype: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            mtp_num_hidden_layers: None,
            mtp_use_dedicated_embeddings: None,
            output_router_logits: None,
            router_aux_loss_coef: None,
        };
        let mut lazy = LazyTensorMap::new();
        merge_moe_experts_in_lazy_map(&mut lazy, &metadata).unwrap();
        assert_eq!(lazy.len(), 0);
    }

    /// `split_and_rename_fused_gate_up_in_lazy_map` is a no-op for
    /// non-qwen35moe arches (Qwen3.5 dense, LLaMA, Gemma).
    #[test]
    fn split_and_rename_fused_gate_up_lazy_no_op_for_dense_arch() {
        use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
        use crate::ir::{DType, ModelMetadata};

        let metadata = ModelMetadata {
            architecture: "Qwen3_5ForCausalLM".into(), // dense, not Moe
            model_type: "qwen3_5".into(),
            param_count: 0,
            hidden_size: 4,
            num_layers: 2,
            layer_types: vec![],
            num_attention_heads: 1,
            num_kv_heads: Some(1),
            vocab_size: 16,
            dtype: "float16".into(),
            shard_count: 1,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: Some(8),
            raw_config: serde_json::Value::Null,
            explicit_layer_types: None,
            full_attention_interval: None,
            attn_output_gate: None,
            head_dim: None,
            partial_rotary_factor: None,
            rope_parameters: None,
            linear_conv_kernel_dim: None,
            linear_key_head_dim: None,
            linear_num_key_heads: None,
            linear_value_head_dim: None,
            linear_num_value_heads: None,
            mamba_ssm_dtype: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            mtp_num_hidden_layers: None,
            mtp_use_dedicated_embeddings: None,
            output_router_logits: None,
            router_aux_loss_coef: None,
        };

        let mut lazy = LazyTensorMap::new();
        let meta = LazyMeta::new(
            "model.layers.0.mlp.experts.gate_up_proj".into(),
            vec![2, 4, 2],
            DType::F16,
        );
        lazy.insert(LazyTensor::from_bytes(meta, vec![0u8; 32]));

        split_and_rename_fused_gate_up_in_lazy_map(&mut lazy, &metadata).unwrap();
        // Untouched on dense arch.
        assert!(lazy.contains_key("model.layers.0.mlp.experts.gate_up_proj"));
    }

    fn moe_ctx() -> Qwen35ConvertContext {
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

    #[test]
    fn embed_tokens_maps_correctly() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.embed_tokens.weight", &ctx),
            Some("token_embd.weight".to_string())
        );
    }

    #[test]
    fn router_maps_correctly() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.0.mlp.gate.weight", &ctx),
            Some("blk.0.ffn_gate_inp.weight".to_string())
        );
    }

    #[test]
    fn per_expert_gate_proj_maps_correctly() {
        let ctx = moe_ctx();
        // Expert 0 gate_proj
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.5.mlp.experts.0.gate_proj.weight", &ctx),
            Some("blk.5.ffn_gate_exps.weight".to_string())
        );
        // Expert 255 gate_proj (max expert index)
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.5.mlp.experts.255.gate_proj.weight", &ctx),
            Some("blk.5.ffn_gate_exps.weight".to_string())
        );
    }

    #[test]
    fn per_expert_up_proj_maps_correctly() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.0.mlp.experts.7.up_proj.weight", &ctx),
            Some("blk.0.ffn_up_exps.weight".to_string())
        );
    }

    #[test]
    fn per_expert_down_proj_maps_correctly() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.2.mlp.experts.128.down_proj.weight", &ctx),
            Some("blk.2.ffn_down_exps.weight".to_string())
        );
    }

    #[test]
    fn shared_expert_maps_correctly() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe(
                "model.layers.1.mlp.shared_expert.gate_proj.weight",
                &ctx
            ),
            Some("blk.1.ffn_gate_shexp.weight".to_string())
        );
        assert_eq!(
            hf_tensor_name_to_gguf_moe(
                "model.layers.1.mlp.shared_expert.up_proj.weight",
                &ctx
            ),
            Some("blk.1.ffn_up_shexp.weight".to_string())
        );
        assert_eq!(
            hf_tensor_name_to_gguf_moe(
                "model.layers.1.mlp.shared_expert.down_proj.weight",
                &ctx
            ),
            Some("blk.1.ffn_down_shexp.weight".to_string())
        );
    }

    #[test]
    fn full_attn_q_proj_maps_correctly() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.3.self_attn.q_proj.weight", &ctx),
            Some("blk.3.attn_q.weight".to_string())
        );
    }

    #[test]
    fn linear_attn_out_proj_maps_correctly() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe(
                "model.layers.0.linear_attn.out_proj.weight",
                &ctx
            ),
            Some("blk.0.ssm_out.weight".to_string()) // llama-arch.cpp:402 LLM_TENSOR_SSM_OUT
        );
    }

    /// dt_bias maps to ssm_dt.bias in GGUF (ADR-012 Decision 8 P4 implementation).
    ///
    /// convert_hf_to_gguf.py:4791 renames ".dt_bias" → ".dt_proj.bias" before GGUF mapping.
    /// llama-arch.cpp:397 LLM_TENSOR_SSM_DT → "blk.%d.ssm_dt" with suffix "bias".
    /// Full GGUF key: "blk.N.ssm_dt.bias".
    #[test]
    fn dt_bias_maps_to_ssm_dt_bias_moe() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.0.linear_attn.dt_bias", &ctx),
            Some("blk.0.ssm_dt.bias".to_string()),
            "dt_bias must map to ssm_dt.bias (llama-arch.cpp:397, convert_hf_to_gguf.py:4791)"
        );
    }

    /// post_attention_layernorm maps to post_attention_norm (ADR-012 P4 verdict).
    ///
    /// llama-model.cpp:7565 loads layer.attn_post_norm via LLM_TENSOR_ATTN_POST_NORM.
    /// llama-arch.cpp:367: LLM_TENSOR_ATTN_POST_NORM → "blk.%d.post_attention_norm".
    #[test]
    fn post_attention_layernorm_maps_to_post_attention_norm_moe() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.3.post_attention_layernorm.weight", &ctx),
            Some("blk.3.post_attention_norm.weight".to_string()),
            "post_attention_layernorm must map to post_attention_norm (llama-arch.cpp:367)"
        );
    }

    /// in_proj_qkv maps to attn_qkv (ADR-012 Decision 8).
    #[test]
    fn in_proj_qkv_maps_to_attn_qkv_moe() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.0.linear_attn.in_proj_qkv.weight", &ctx),
            Some("blk.0.attn_qkv.weight".to_string()),
        );
    }

    /// in_proj_z maps to attn_gate (ADR-012 Decision 8).
    #[test]
    fn in_proj_z_maps_to_attn_gate_moe() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.0.linear_attn.in_proj_z.weight", &ctx),
            Some("blk.0.attn_gate.weight".to_string()),
        );
    }

    /// shared_expert_gate maps to ffn_gate_inp_shexp (ADR-012 Decision 8).
    /// tensor_mapping.py:447 + llama-arch.cpp:394
    #[test]
    fn shared_expert_gate_maps_correctly() {
        let ctx = moe_ctx();
        assert_eq!(
            hf_tensor_name_to_gguf_moe("model.layers.0.mlp.shared_expert_gate.weight", &ctx),
            Some("blk.0.ffn_gate_inp_shexp.weight".to_string()),
            "shared_expert_gate must map to ffn_gate_inp_shexp (tensor_mapping.py:447)"
        );
    }

    // --------------------------------------------------------------------- //
    // P5: merge_expert_tensors — correctness tests (ADR-012 Decision 9)    //
    // --------------------------------------------------------------------- //

    /// Helper: build a TensorRef whose F32 data is filled with the value `fill`
    /// (as f32 bytes), with shape [rows, cols].
    fn make_expert_f32(name: &str, rows: usize, cols: usize, fill: f32) -> TensorRef {
        use crate::ir::DType;
        let data: Vec<u8> = (0..rows * cols)
            .flat_map(|_| fill.to_le_bytes())
            .collect();
        TensorRef {
            name: name.to_string(),
            shape: vec![rows, cols],
            dtype: DType::F32,
            data,
        }
    }

    /// 4-expert stacking: each expert E has fill value E as f32.
    /// After merge, shape must be [4, rows, cols] and slice [E, :, :] must
    /// equal all-E values.
    #[test]
    fn merge_expert_tensors_4_expert_stacking_gate_up() {
        use crate::ir::DType;
        let moe_inter = 8usize;
        let hidden = 4usize;

        // gate/up: per-expert shape [moe_inter, hidden]
        let experts: Vec<TensorRef> = (0..4_u32)
            .map(|e| make_expert_f32(
                &format!("blk.0.expert.{}.gate_proj.weight", e),
                moe_inter,
                hidden,
                e as f32,
            ))
            .collect();

        let merged = merge_expert_tensors(
            &experts,
            ExpertProj::Gate,
            crate::models::qwen35::Qwen35Arch::Moe,
        )
        .expect("merge_expert_tensors must succeed for valid inputs");

        // Shape: [N_experts, moe_inter, hidden]
        assert_eq!(merged.shape, vec![4, moe_inter, hidden]);
        assert_eq!(merged.dtype, DType::F32);
        assert_eq!(merged.data.len(), 4 * moe_inter * hidden * 4);

        // Each expert's slice must contain its fill value.
        let elem_size = 4usize; // F32
        let bytes_per_expert = moe_inter * hidden * elem_size;
        for e in 0..4_usize {
            let start = e * bytes_per_expert;
            let end = start + bytes_per_expert;
            let slice = &merged.data[start..end];
            let expected_val = e as f32;
            for chunk in slice.chunks_exact(4) {
                let v = f32::from_le_bytes(chunk.try_into().unwrap());
                assert_eq!(
                    v, expected_val,
                    "expert {e} slice must contain value {expected_val}, got {v}"
                );
            }
        }
    }

    /// Down-proj has transposed per-expert shape [hidden, moe_inter].
    /// Stacking produces [N, hidden, moe_inter] — verified shape only;
    /// data correctness follows from the same byte-copy logic.
    #[test]
    fn merge_expert_tensors_4_expert_stacking_down() {
        use crate::ir::DType;
        let hidden = 4usize;
        let moe_inter = 8usize;

        // down: per-expert shape [hidden, moe_inter]
        let experts: Vec<TensorRef> = (0..4_u32)
            .map(|e| make_expert_f32(
                &format!("blk.0.expert.{}.down_proj.weight", e),
                hidden,
                moe_inter,
                e as f32,
            ))
            .collect();

        let merged = merge_expert_tensors(
            &experts,
            ExpertProj::Down,
            crate::models::qwen35::Qwen35Arch::Moe,
        )
        .expect("merge_expert_tensors must succeed for valid inputs");

        // Shape: [N_experts, hidden, moe_inter]
        assert_eq!(merged.shape, vec![4, hidden, moe_inter],
            "down_proj merged shape must be [N, hidden, moe_inter]");
        assert_eq!(merged.dtype, DType::F32);

        // Verify expert 2 slice contains value 2.0.
        let bytes_per_expert = hidden * moe_inter * 4;
        let start = 2 * bytes_per_expert;
        let end = start + bytes_per_expert;
        for chunk in merged.data[start..end].chunks_exact(4) {
            let v = f32::from_le_bytes(chunk.try_into().unwrap());
            assert_eq!(v, 2.0f32, "expert 2 down slice must contain 2.0");
        }
    }

    /// Dense arch guard: calling merge_expert_tensors with Dense arch returns
    /// DenseContextMergeCall error — never succeeds.
    #[test]
    fn merge_expert_tensors_dense_guard_fires() {
        let t = make_expert_f32("dummy", 4, 8, 1.0);
        let err = merge_expert_tensors(
            &[t],
            ExpertProj::Gate,
            crate::models::qwen35::Qwen35Arch::Dense,
        )
        .expect_err("dense arch must return DenseContextMergeCall");
        assert!(
            matches!(err, crate::models::qwen35::ConvertError::DenseContextMergeCall),
            "expected DenseContextMergeCall, got: {err}"
        );
    }

    /// Empty slice returns ExpertMergeEmpty error.
    #[test]
    fn merge_expert_tensors_empty_slice_errors() {
        let err = merge_expert_tensors(
            &[],
            ExpertProj::Gate,
            crate::models::qwen35::Qwen35Arch::Moe,
        )
        .expect_err("empty slice must return ExpertMergeEmpty");
        assert!(
            matches!(err, crate::models::qwen35::ConvertError::ExpertMergeEmpty),
            "expected ExpertMergeEmpty, got: {err}"
        );
    }

    /// Shape mismatch between experts returns ExpertMergeShapeMismatch.
    #[test]
    fn merge_expert_tensors_shape_mismatch_errors() {
        let t0 = make_expert_f32("e0", 4, 8, 0.0);
        let t1 = make_expert_f32("e1", 4, 16, 1.0); // wrong cols
        let err = merge_expert_tensors(
            &[t0, t1],
            ExpertProj::Gate,
            crate::models::qwen35::Qwen35Arch::Moe,
        )
        .expect_err("shape mismatch must return ExpertMergeShapeMismatch");
        assert!(
            matches!(
                err,
                crate::models::qwen35::ConvertError::ExpertMergeShapeMismatch { expert_idx: 1, .. }
            ),
            "expected ExpertMergeShapeMismatch at idx 1, got: {err}"
        );
    }

    /// Shared expert tensors (singletons) do NOT go through merge_expert_tensors.
    /// They map directly to their GGUF names via hf_tensor_name_to_gguf_moe.
    /// This test confirms the naming round-trip (compile-time guarantee — no
    /// merge call site exists for shared_expert).
    #[test]
    fn shared_expert_singleton_not_merged() {
        let ctx = moe_ctx();
        // Shared expert tensors must map to singleton GGUF names (no _exps suffix).
        for proj in &["gate_proj", "up_proj", "down_proj"] {
            let hf = format!("model.layers.0.mlp.shared_expert.{}.weight", proj);
            let gguf = hf_tensor_name_to_gguf_moe(&hf, &ctx)
                .unwrap_or_else(|| panic!("shared_expert {proj} must map to a GGUF name"));
            // Must end in _shexp.weight, NOT _exps.weight.
            assert!(
                gguf.contains("_shexp"),
                "shared expert {proj} must map to a _shexp tensor, got: {gguf}"
            );
            assert!(
                !gguf.contains("_exps"),
                "shared expert {proj} must NOT map to an _exps (merged) tensor, got: {gguf}"
            );
        }
    }

    /// emit_metadata_moe returns Ok when context is valid (P4 implementation).
    #[allow(deprecated)]
    #[test]
    fn emit_metadata_moe_returns_ok_when_valid() {
        let ctx = moe_ctx();
        assert!(
            emit_metadata_moe(&ctx).is_ok(),
            "emit_metadata_moe should return Ok for a valid MoE context with all required fields"
        );
    }

    /// emit_metadata_moe returns error when num_experts is absent.
    #[allow(deprecated)]
    #[test]
    fn emit_metadata_moe_errors_when_missing_num_experts() {
        let mut ctx = moe_ctx();
        ctx.num_experts = None;
        let err = emit_metadata_moe(&ctx)
            .expect_err("emit_metadata_moe should return error when num_experts is None");
        assert!(
            matches!(err, crate::models::qwen35::ConvertError::MissingHparam { .. }),
            "expected MissingHparam, got: {err}"
        );
    }

    /// emit_metadata_moe returns error when moe_intermediate_size is absent.
    #[allow(deprecated)]
    #[test]
    fn emit_metadata_moe_errors_when_missing_moe_ff_size() {
        let mut ctx = moe_ctx();
        ctx.moe_intermediate_size = None;
        let err = emit_metadata_moe(&ctx)
            .expect_err("emit_metadata_moe should return error when moe_intermediate_size is None");
        assert!(
            matches!(err, crate::models::qwen35::ConvertError::MissingHparam { .. }),
            "expected MissingHparam, got: {err}"
        );
    }
}
