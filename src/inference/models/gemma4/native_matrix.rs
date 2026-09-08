//! Native GGUF matrix storage for Gemma 4.
//!
//! The artifact's declared representation is an execution contract.  This
//! module admits that representation before Metal allocation, maps the bytes
//! read-only, and routes embedding lookup without constructing a whole-table
//! dense or re-quantized shadow.

use anyhow::{bail, Context, Result};
use mlx_native::{
    ggml_capability, GgmlCapabilityRequest, GgmlExpertInputLayout, GgmlExpertShape, GgmlInvocation,
    GgmlRoutingPolicy, GgmlType, GgmlWorkloadClass, GGML_CAPABILITY_SCHEMA_VERSION,
};

use crate::serve::config::Gemma4Config;
use crate::serve::forward_mlx_shared::MlxQWeight;

/// Allocation-free description of the embedding and (optional) untied head.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeIoPlan {
    pub embedding: NativeMatrixSpec,
    pub output: Option<NativeMatrixSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeMatrixSpec {
    pub name: String,
    pub ggml_type: GgmlType,
    pub rows: usize,
    pub cols: usize,
    pub byte_len: usize,
}

/// Resolve tied storage without manufacturing a second value.  Kept generic
/// so the ownership rule is testable without allocating a Metal resource.
#[inline]
pub fn resolve_tied_or_explicit<'a, T>(embedding: &'a T, output: Option<&'a T>) -> &'a T {
    output.unwrap_or(embedding)
}

fn dimensions(info: &mlx_native::gguf::TensorInfo, name: &str) -> Result<(usize, usize)> {
    let [rows, cols] = info.shape.as_slice() else {
        bail!(
            "Gemma native matrix '{name}' must be rank 2 [rows, cols], got {:?}",
            info.shape
        );
    };
    if *rows == 0 || *cols == 0 {
        bail!("Gemma native matrix '{name}' has a zero dimension");
    }
    Ok((*rows, *cols))
}

fn capability(
    ggml_type: GgmlType,
    invocation: GgmlInvocation,
    workload: GgmlWorkloadClass,
) -> mlx_native::GgmlCapability {
    ggml_capability(GgmlCapabilityRequest {
        schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
        invocation,
        ggml_type,
        workload,
        routing: GgmlRoutingPolicy::default(),
    })
}

/// Whether the stored O-projection can consume head-major BF16 flash output
/// directly. Unsupported stored types take the explicit activation-only
/// permute into the ordinary native projection route; weights never change.
pub fn supports_native_perm021(weight: &MlxQWeight, m: u32, head_dim: u32) -> bool {
    crate::serve::forward_mlx_shared::supports_native_perm021(weight, m, head_dim)
}

/// Admit the exact stored embedding representation for both gather and tied
/// output-head execution.  No buffer is allocated by this function.
pub fn admit_embedding(
    name: &str,
    info: &mlx_native::gguf::TensorInfo,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<NativeMatrixSpec> {
    let (rows, cols) = dimensions(info, name)?;
    if rows != expected_rows || cols != expected_cols {
        bail!(
            "Gemma native matrix '{name}' shape [{rows}, {cols}] does not match expected [{expected_rows}, {expected_cols}]"
        );
    }
    let vocab = u32::try_from(rows).context("embedding row count exceeds u32")?;
    let width = u32::try_from(cols).context("embedding width exceeds u32")?;
    let gather = capability(
        info.ggml_type,
        GgmlInvocation::EmbeddingGather {
            n_tokens: 1,
            vocab_size: vocab,
            embed_dim: width,
        },
        GgmlWorkloadClass::Embedding,
    );
    if !gather.executable {
        bail!(
            "Gemma native embedding '{name}' rejects stored {:?}: {}",
            info.ggml_type,
            gather.diagnostic
        );
    }
    admit_projection(name, info, rows, cols)?;
    Ok(NativeMatrixSpec {
        name: name.to_owned(),
        ggml_type: info.ggml_type,
        rows,
        cols,
        byte_len: info.byte_len,
    })
}

/// Admit one exact native projection for decode and prompt execution.
pub fn admit_projection(
    name: &str,
    info: &mlx_native::gguf::TensorInfo,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<NativeMatrixSpec> {
    let (rows, cols) = dimensions(info, name)?;
    if rows != expected_rows || cols != expected_cols {
        bail!(
            "Gemma native projection '{name}' shape [{rows}, {cols}] does not match expected [{expected_rows}, {expected_cols}]"
        );
    }
    let n = u32::try_from(rows).context("projection row count exceeds u32")?;
    let k = u32::try_from(cols).context("projection column count exceeds u32")?;
    // m=1, every speculative/multi-slot width, and the first prompt width
    // exercise all production dense-routing regimes. Larger prompt widths
    // retain the same route (IQ4_XS intentionally remains on matvec).
    for m in 1..=9 {
        let workload = match m {
            1 => GgmlWorkloadClass::DecodeSingle,
            2..=8 => GgmlWorkloadClass::ContinuousWidth,
            _ => GgmlWorkloadClass::Prompt,
        };
        let decision = capability(
            info.ggml_type,
            GgmlInvocation::DenseAuto { m, n, k },
            workload,
        );
        if !decision.executable {
            bail!(
                "Gemma native projection '{name}' rejects stored {:?} for {workload:?}: {}",
                info.ggml_type,
                decision.diagnostic
            );
        }
        if u64::try_from(info.byte_len).unwrap_or(u64::MAX) < decision.minimum_weight_buffer_bytes {
            bail!(
                "Gemma native projection '{name}' payload is {} bytes; stored {:?} requires at least {}",
                info.byte_len,
                info.ggml_type,
                decision.minimum_weight_buffer_bytes
            );
        }
    }
    Ok(NativeMatrixSpec {
        name: name.to_owned(),
        ggml_type: info.ggml_type,
        rows,
        cols,
        byte_len: info.byte_len,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpertMatrixRole {
    GateUp,
    Down,
}

#[derive(Debug, Clone, Copy)]
enum ExpertEntrypoint {
    Auto,
    ForcedMv,
    Pooled,
}

#[allow(clippy::too_many_arguments)]
fn admit_expert_invocation(
    name: &str,
    info: &mlx_native::gguf::TensorInfo,
    n_experts: u32,
    n: u32,
    k: u32,
    expert_stride_bytes: u64,
    n_tokens: u32,
    top_k: u32,
    workload: GgmlWorkloadClass,
    entrypoint: ExpertEntrypoint,
) -> Result<()> {
    let scalar_dtype = match info.ggml_type {
        GgmlType::F32 => Some(mlx_native::DType::F32),
        GgmlType::F16 => Some(mlx_native::DType::F16),
        GgmlType::BF16 => Some(mlx_native::DType::BF16),
        _ => None,
    };
    if let Some(dtype) = scalar_dtype {
        let admitted = mlx_native::dense_matmul_id_capability(
            dtype,
            &mlx_native::DenseMatmulIdParams {
                m: n_tokens,
                n,
                k,
                top_k,
                n_experts,
                expert_stride_bytes,
                input_layout: mlx_native::DenseMatmulIdInputLayout::SharedPerToken,
                id_multiplicity: mlx_native::DenseMatmulIdMultiplicity::MayRepeat,
                route: mlx_native::DenseMatmulIdRoute::Direct,
            },
        )
        .with_context(|| format!("native scalar expert {name}"))?;
        anyhow::ensure!(
            info.byte_len >= admitted.required_weight_bytes,
            "native scalar expert {name} has insufficient stored bytes"
        );
        return Ok(());
    }
    let shape = GgmlExpertShape {
        n_tokens,
        n,
        k,
        top_k,
        n_experts,
        expert_stride_bytes,
        ids_are_distinct_per_token: true,
        ids_within_expert_range: true,
    };
    let invocation = match entrypoint {
        ExpertEntrypoint::Auto => GgmlInvocation::ExpertAutoAllocated { shape },
        ExpertEntrypoint::ForcedMv => GgmlInvocation::ExpertForceMv { shape },
        ExpertEntrypoint::Pooled => GgmlInvocation::ExpertPooled {
            shape,
            input_layout: GgmlExpertInputLayout::SharedPerToken,
        },
    };
    let decision = capability(info.ggml_type, invocation, workload);
    if !decision.executable {
        bail!(
            "Gemma native expert matrix '{name}' rejects stored {:?} for {workload:?} {entrypoint:?} (n_tokens={n_tokens}, top_k={top_k}): {}",
            info.ggml_type,
            decision.diagnostic
        );
    }
    if u64::try_from(info.byte_len).unwrap_or(u64::MAX) < decision.minimum_weight_buffer_bytes {
        bail!(
            "Gemma native expert matrix '{name}' payload is {} bytes; stored {:?} requires at least {}",
            info.byte_len,
            info.ggml_type,
            decision.minimum_weight_buffer_bytes
        );
    }
    Ok(())
}

/// Admit one native expert stack across the exact decode, multi-slot, and
/// pooled-prompt entry points that consume it. The shape is logical
/// `[experts, rows, cols]`; every expert owns one exact contiguous matrix.
fn admit_expert_stack(
    name: &str,
    info: &mlx_native::gguf::TensorInfo,
    n_experts: usize,
    rows: usize,
    cols: usize,
    top_k: usize,
    role: ExpertMatrixRole,
) -> Result<()> {
    if info.shape != [n_experts, rows, cols] {
        bail!(
            "Gemma native expert matrix '{name}' shape {:?} does not match expected [{n_experts}, {rows}, {cols}]",
            info.shape
        );
    }
    if n_experts == 0 || info.byte_len % n_experts != 0 {
        bail!(
            "Gemma native expert matrix '{name}' payload {} is not evenly divisible across {n_experts} experts",
            info.byte_len
        );
    }
    let expert_stride_bytes =
        u64::try_from(info.byte_len / n_experts).context("expert stride exceeds u64")?;
    let n_experts = u32::try_from(n_experts).context("expert count exceeds u32")?;
    let n = u32::try_from(rows).context("expert row count exceeds u32")?;
    let k = u32::try_from(cols).context("expert column count exceeds u32")?;
    let top_k = u32::try_from(top_k).context("expert top-k exceeds u32")?;
    if top_k == 0 || top_k > n_experts {
        bail!(
            "Gemma native expert matrix '{name}' has invalid top-k {top_k} for {n_experts} experts"
        );
    }

    let routed_rows = |tokens: u32| -> Result<u32> {
        match role {
            ExpertMatrixRole::GateUp => Ok(tokens),
            ExpertMatrixRole::Down => tokens
                .checked_mul(top_k)
                .context("expert routed-row count overflow"),
        }
    };
    let routed_top_k = match role {
        ExpertMatrixRole::GateUp => top_k,
        ExpertMatrixRole::Down => 1,
    };
    let workload_for_rows = |rows: u32| {
        if rows == 1 {
            GgmlWorkloadClass::DecodeSingle
        } else if rows <= 8 {
            GgmlWorkloadClass::ContinuousWidth
        } else {
            GgmlWorkloadClass::Prompt
        }
    };

    // Serial decode uses the auto-allocating entry point.
    let serial_rows = routed_rows(1)?;
    admit_expert_invocation(
        name,
        info,
        n_experts,
        n,
        k,
        expert_stride_bytes,
        serial_rows,
        routed_top_k,
        workload_for_rows(serial_rows),
        ExpertEntrypoint::Auto,
    )?;

    // Speculative verification and multi-slot decode deliberately force the
    // row-identical matvec entry point for widths 2..=8.
    for slots in 2..=8 {
        let rows = routed_rows(slots)?;
        admit_expert_invocation(
            name,
            info,
            n_experts,
            n,
            k,
            expert_stride_bytes,
            rows,
            routed_top_k,
            workload_for_rows(rows),
            ExpertEntrypoint::ForcedMv,
        )?;
    }

    // Pooled prefill routes below and above the grouped-MM threshold. Checking
    // both prevents admission based only on the cheap short-prompt path.
    for prompt_tokens in [9, 33] {
        let rows = routed_rows(prompt_tokens)?;
        admit_expert_invocation(
            name,
            info,
            n_experts,
            n,
            k,
            expert_stride_bytes,
            rows,
            routed_top_k,
            GgmlWorkloadClass::Prompt,
            ExpertEntrypoint::Pooled,
        )?;
    }
    Ok(())
}

/// Preflight the IO matrices before the loader creates any Metal resource.
pub fn preflight_io(
    gguf: &mlx_native::gguf::GgufFile,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<NativeIoPlan> {
    let embedding_name = "token_embd.weight";
    let embedding_info = gguf
        .tensor_info(embedding_name)
        .ok_or_else(|| anyhow::anyhow!("missing required tensor '{embedding_name}'"))?;
    let embedding = admit_embedding(embedding_name, embedding_info, vocab_size, hidden_size)?;
    let output = match gguf.tensor_info("output.weight") {
        Some(info) => Some(admit_projection(
            "output.weight",
            info,
            vocab_size,
            hidden_size,
        )?),
        None => None,
    };
    Ok(NativeIoPlan { embedding, output })
}

/// Validate every rank-2 projection consumed by the Gemma graph before the
/// loader maps or allocates model storage.  Norm vectors and 3-D expert stacks
/// have separate execution contracts and are intentionally excluded.
pub fn preflight_projections(gguf: &mlx_native::gguf::GgufFile, cfg: &Gemma4Config) -> Result<()> {
    let require = |name: &str, rows: usize, cols: usize| -> Result<()> {
        let info = gguf.tensor_info(name).ok_or_else(|| {
            anyhow::anyhow!("missing required tensor '{name}' (native projection)")
        })?;
        admit_projection(name, info, rows, cols)?;
        Ok(())
    };
    for layer in 0..cfg.num_hidden_layers {
        let head_width = cfg
            .num_attention_heads
            .checked_mul(cfg.head_dim_for_layer(layer))
            .context("Gemma query projection width overflow")?;
        let kv_width = cfg
            .num_kv_heads_for_layer(layer)
            .checked_mul(cfg.head_dim_for_layer(layer))
            .context("Gemma key/value projection width overflow")?;
        require(
            &format!("blk.{layer}.attn_q.weight"),
            head_width,
            cfg.hidden_size,
        )?;
        require(
            &format!("blk.{layer}.attn_k.weight"),
            kv_width,
            cfg.hidden_size,
        )?;
        if !(cfg.is_full_attention(layer) && cfg.attention_k_eq_v) {
            require(
                &format!("blk.{layer}.attn_v.weight"),
                kv_width,
                cfg.hidden_size,
            )?;
        }
        require(
            &format!("blk.{layer}.attn_output.weight"),
            cfg.hidden_size,
            head_width,
        )?;
        require(
            &format!("blk.{layer}.ffn_gate.weight"),
            cfg.intermediate_size,
            cfg.hidden_size,
        )?;
        require(
            &format!("blk.{layer}.ffn_up.weight"),
            cfg.intermediate_size,
            cfg.hidden_size,
        )?;
        require(
            &format!("blk.{layer}.ffn_down.weight"),
            cfg.hidden_size,
            cfg.intermediate_size,
        )?;

        let gate_up_name = format!("blk.{layer}.ffn_gate_up_exps.weight");
        let down_name = format!("blk.{layer}.ffn_down_exps.weight");
        match (
            gguf.tensor_info(&gate_up_name),
            gguf.tensor_info(&down_name),
        ) {
            (None, None) => {}
            (Some(_), None) | (None, Some(_)) => bail!(
                "Gemma layer {layer} must declare both expert matrices or neither ({gate_up_name}, {down_name})"
            ),
            (Some(gate_up), Some(down)) => {
                let gate_up_rows = cfg
                    .moe_intermediate_size
                    .checked_mul(2)
                    .context("Gemma expert gate/up width overflow")?;
                require(
                    &format!("blk.{layer}.ffn_gate_inp.weight"),
                    cfg.num_experts,
                    cfg.hidden_size,
                )?;
                admit_expert_stack(
                    &gate_up_name,
                    gate_up,
                    cfg.num_experts,
                    gate_up_rows,
                    cfg.hidden_size,
                    cfg.top_k_experts,
                    ExpertMatrixRole::GateUp,
                )?;
                admit_expert_stack(
                    &down_name,
                    down,
                    cfg.num_experts,
                    cfg.hidden_size,
                    cfg.moe_intermediate_size,
                    cfg.top_k_experts,
                    ExpertMatrixRole::Down,
                )?;
            }
        }
    }
    Ok(())
}

pub fn load_mapped_projection(
    gguf: &mlx_native::gguf::GgufFile,
    mapped: &mlx_native::gguf::GgufMappedTensorSet<'_>,
    name: &str,
) -> Result<MlxQWeight> {
    let info = gguf
        .tensor_info(name)
        .ok_or_else(|| anyhow::anyhow!("missing required tensor '{name}' (native projection)"))?;
    let (rows, cols) = dimensions(info, name)?;
    admit_projection(name, info, rows, cols)?;
    MlxQWeight::from_mapped_gguf_tensor(mapped, info)
}

pub use super::native_embedding::{encode_embedding, encode_embedding_rows};

#[cfg(test)]
#[path = "native_matrix_tests.rs"]
mod tests;
