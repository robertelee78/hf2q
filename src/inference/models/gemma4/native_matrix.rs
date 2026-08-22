//! Native GGUF matrix storage for Gemma 4.
//!
//! The artifact's declared representation is an execution contract.  This
//! module admits that representation before Metal allocation, maps the bytes
//! read-only, and routes embedding lookup without constructing a whole-table
//! dense or re-quantized shadow.

use anyhow::{bail, Context, Result};
use mlx_native::{
    ggml_capability, GgmlCapabilityRequest, GgmlExpertInputLayout, GgmlExpertShape, GgmlInvocation,
    GgmlRoutingPolicy, GgmlType, GgmlWorkloadClass, MlxBuffer, MlxDevice,
    GGML_CAPABILITY_SCHEMA_VERSION,
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
        let info = gguf
            .tensor_info(name)
            .ok_or_else(|| anyhow::anyhow!("missing required projection '{name}'"))?;
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
        .ok_or_else(|| anyhow::anyhow!("missing required projection '{name}'"))?;
    let (rows, cols) = dimensions(info, name)?;
    admit_projection(name, info, rows, cols)?;
    MlxQWeight::from_mapped_gguf_tensor(mapped, info)
}

/// Gather stored embedding rows and apply Gemma's sqrt(hidden) scale.
#[allow(clippy::too_many_arguments)]
pub fn encode_embedding_rows(
    session: &mut mlx_native::GraphSession<'_>,
    registry: &mut mlx_native::KernelRegistry,
    device: &MlxDevice,
    weight: &MlxQWeight,
    token_ids: &MlxBuffer,
    output: &MlxBuffer,
    n_tokens: usize,
) -> Result<()> {
    let params = (weight.info.rows, weight.info.cols, n_tokens);
    match weight.info.ggml_dtype {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
            mlx_native::ops::embedding_dense::embedding_gather_dense(
                session.encoder_mut(),
                registry,
                device,
                &weight.buffer,
                token_ids,
                output,
                &mlx_native::ops::embedding_dense::EmbeddingDenseParams {
                    vocab_size: params.0,
                    embed_dim: params.1,
                    n_tokens: params.2,
                },
            )?
        }
        GgmlType::Q4_0 => session.embedding_gather_q4_0(
            registry,
            device,
            &weight.buffer,
            token_ids,
            output,
            &mlx_native::ops::embedding_q4_0::EmbeddingQ4_0Params {
                vocab_size: params.0,
                embed_dim: params.1,
                n_tokens: params.2,
            },
        )?,
        GgmlType::Q2_K => session.embedding_gather_q2_k(
            registry,
            device,
            &weight.buffer,
            token_ids,
            output,
            &mlx_native::ops::embedding_q2_k::EmbeddingQ2KParams {
                vocab_size: params.0,
                embed_dim: params.1,
                n_tokens: params.2,
            },
        )?,
        GgmlType::Q4_K => session.embedding_gather_q4_k(
            registry,
            device,
            &weight.buffer,
            token_ids,
            output,
            &mlx_native::ops::embedding_q4_k::EmbeddingQ4KParams {
                vocab_size: params.0,
                embed_dim: params.1,
                n_tokens: params.2,
            },
        )?,
        GgmlType::Q5_K => session.embedding_gather_q5_k(
            registry,
            device,
            &weight.buffer,
            token_ids,
            output,
            &mlx_native::ops::embedding_kquant::EmbeddingQ5KParams {
                vocab_size: params.0,
                embed_dim: params.1,
                n_tokens: params.2,
            },
        )?,
        GgmlType::Q6_K => session.embedding_gather_q6_k(
            registry,
            device,
            &weight.buffer,
            token_ids,
            output,
            &mlx_native::ops::embedding_kquant::EmbeddingQ6KParams {
                vocab_size: params.0,
                embed_dim: params.1,
                n_tokens: params.2,
            },
        )?,
        GgmlType::Q8_0 => session.embedding_gather_q8_0(
            registry,
            device,
            &weight.buffer,
            token_ids,
            output,
            &mlx_native::ops::embedding_q8_0::EmbeddingQ8_0Params {
                vocab_size: params.0,
                embed_dim: params.1,
                n_tokens: params.2,
            },
        )?,
        other => bail!("Gemma embedding dispatch reached unadmitted type {other:?}"),
    }
    session.track_dispatch(&[&weight.buffer, token_ids], &[output]);
    Ok(())
}

/// Gather stored embedding rows and apply Gemma's model-input scale.
#[allow(clippy::too_many_arguments)]
pub fn encode_embedding(
    session: &mut mlx_native::GraphSession<'_>,
    registry: &mut mlx_native::KernelRegistry,
    device: &MlxDevice,
    weight: &MlxQWeight,
    token_ids: &MlxBuffer,
    output: &MlxBuffer,
    n_tokens: usize,
) -> Result<()> {
    encode_embedding_rows(
        session, registry, device, weight, token_ids, output, n_tokens,
    )?;
    session.barrier_between(&[output], &[output]);
    mlx_native::ops::elementwise::scalar_mul_f32(
        session.encoder_mut(),
        registry,
        device.metal_device(),
        output,
        output,
        n_tokens * weight.info.cols,
        (weight.info.cols as f32).sqrt(),
    )?;
    session.track_dispatch(&[output], &[output]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored_bytes(dtype: GgmlType, rows: usize, cols: usize) -> usize {
        match dtype {
            GgmlType::F32 => rows * cols * std::mem::size_of::<f32>(),
            GgmlType::F16 | GgmlType::BF16 => rows * cols * std::mem::size_of::<u16>(),
            _ => mlx_native::ggml_matrix_bytes(dtype, rows as u32, cols as u32).unwrap() as usize,
        }
    }

    fn info(
        ggml_type: GgmlType,
        rows: usize,
        cols: usize,
        byte_len: usize,
    ) -> mlx_native::gguf::TensorInfo {
        mlx_native::gguf::TensorInfo {
            name: "token_embd.weight".into(),
            shape: vec![rows, cols],
            ggml_type,
            offset: 0,
            byte_len,
        }
    }

    #[test]
    fn admitted_embedding_types_are_exact_native_intersection() {
        for dtype in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::BF16,
            GgmlType::Q4_0,
            GgmlType::Q2_K,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q8_0,
        ] {
            let bytes = stored_bytes(dtype, 4, 256);
            let admitted =
                admit_embedding("token_embd.weight", &info(dtype, 4, 256, bytes), 4, 256)
                    .unwrap_or_else(|error| panic!("{dtype:?} should be native: {error}"));
            assert_eq!(admitted.ggml_type, dtype);
            assert_eq!(admitted.byte_len, bytes);
        }
    }

    #[test]
    fn native_matrix_embedding_dispatches_every_admitted_codec() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let Some(device) = MlxDevice::new().ok() else {
            eprintln!("[skip] Metal device unavailable");
            return;
        };
        let rows = 4usize;
        let cols = 256usize;
        let n_tokens = 2usize;
        let mut ids = device
            .alloc_buffer(
                n_tokens * std::mem::size_of::<u32>(),
                mlx_native::DType::U32,
                vec![n_tokens],
            )
            .unwrap();
        ids.as_mut_slice::<u32>().unwrap().copy_from_slice(&[0, 3]);

        for dtype in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::BF16,
            GgmlType::Q4_0,
            GgmlType::Q2_K,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q8_0,
        ] {
            let byte_len = stored_bytes(dtype, rows, cols);
            let storage_dtype = match dtype {
                GgmlType::F32 => mlx_native::DType::F32,
                GgmlType::F16 => mlx_native::DType::F16,
                GgmlType::BF16 => mlx_native::DType::BF16,
                _ => mlx_native::DType::U8,
            };
            let storage_shape = if storage_dtype == mlx_native::DType::U8 {
                vec![byte_len]
            } else {
                vec![rows, cols]
            };
            let mut stored = device
                .alloc_buffer(byte_len, storage_dtype, storage_shape)
                .unwrap();
            match storage_dtype {
                mlx_native::DType::F32 => stored.as_mut_slice::<f32>().unwrap().fill(0.0),
                mlx_native::DType::F16 | mlx_native::DType::BF16 => {
                    stored.as_mut_slice::<u16>().unwrap().fill(0)
                }
                mlx_native::DType::U8 => stored.as_mut_slice::<u8>().unwrap().fill(0),
                other => panic!("unexpected embedding storage dtype {other:?}"),
            }
            let weight = MlxQWeight {
                buffer: stored,
                info: crate::serve::gpu::QuantWeightInfo {
                    ggml_dtype: dtype,
                    rows,
                    cols,
                },
                affine: None,
                decode_record_q6k_m1: std::sync::OnceLock::new(),
            };
            let output = device
                .alloc_buffer(
                    n_tokens * cols * std::mem::size_of::<f32>(),
                    mlx_native::DType::F32,
                    vec![n_tokens, cols],
                )
                .unwrap();
            let mut registry = mlx_native::KernelRegistry::new();
            let executor = mlx_native::GraphExecutor::new(device.clone());
            let mut session = executor.begin().unwrap();
            encode_embedding_rows(
                &mut session,
                &mut registry,
                &device,
                &weight,
                &ids,
                &output,
                n_tokens,
            )
            .unwrap_or_else(|error| panic!("{dtype:?} embedding dispatch failed: {error}"));
            session.finish().unwrap();
            assert!(
                output
                    .as_slice::<f32>()
                    .unwrap()
                    .iter()
                    .all(|value| *value == 0.0),
                "{dtype:?} zero embedding did not produce numeric zeros"
            );
        }
    }

    #[test]
    fn unsupported_and_malformed_embeddings_fail_before_loading() {
        let unsupported = info(GgmlType::Q3_K, 4, 256, 440);
        assert!(admit_embedding("token_embd.weight", &unsupported, 4, 256)
            .unwrap_err()
            .to_string()
            .contains("rejects stored Q3_K"));

        let mut malformed = info(GgmlType::Q6_K, 4, 256, 1);
        assert!(admit_embedding("token_embd.weight", &malformed, 4, 256)
            .unwrap_err()
            .to_string()
            .contains("payload is 1 bytes"));
        malformed.shape = vec![4, 2, 128];
        assert!(admit_embedding("token_embd.weight", &malformed, 4, 256)
            .unwrap_err()
            .to_string()
            .contains("must be rank 2"));

        let wrong_projection_shape =
            info(GgmlType::Q6_K, 8, 256, stored_bytes(GgmlType::Q6_K, 8, 256));
        assert!(
            admit_projection("blk.0.attn_q.weight", &wrong_projection_shape, 4, 256)
                .unwrap_err()
                .to_string()
                .contains("does not match expected [4, 256]")
        );
    }

    #[test]
    fn projections_admit_every_native_dense_codec_without_transforming_it() {
        for dtype in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::BF16,
            GgmlType::Q4_0,
            GgmlType::Q8_0,
            GgmlType::Q2_K,
            GgmlType::Q3_K,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q5_1,
            GgmlType::IQ4_NL,
            GgmlType::IQ4_XS,
        ] {
            let bytes = stored_bytes(dtype, 4, 256);
            let admitted = admit_projection("output.weight", &info(dtype, 4, 256, bytes), 4, 256)
                .unwrap_or_else(|error| panic!("{dtype:?} should be native: {error}"));
            assert_eq!(admitted.ggml_type, dtype);
            assert_eq!(admitted.byte_len, bytes);
        }
    }

    #[test]
    fn native_matrix_projection_dispatches_every_admitted_codec_and_width() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let Some(device) = MlxDevice::new().ok() else {
            eprintln!("[skip] Metal device unavailable");
            return;
        };
        let rows = 4usize;
        let cols = 256usize;
        for dtype in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::BF16,
            GgmlType::Q4_0,
            GgmlType::Q8_0,
            GgmlType::Q2_K,
            GgmlType::Q3_K,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q5_1,
            GgmlType::IQ4_NL,
            GgmlType::IQ4_XS,
        ] {
            let byte_len = stored_bytes(dtype, rows, cols);
            let storage_dtype = match dtype {
                GgmlType::F32 => mlx_native::DType::F32,
                GgmlType::F16 => mlx_native::DType::F16,
                GgmlType::BF16 => mlx_native::DType::BF16,
                _ => mlx_native::DType::U8,
            };
            let storage_shape = if storage_dtype == mlx_native::DType::U8 {
                vec![byte_len]
            } else {
                vec![rows, cols]
            };
            let mut stored = device
                .alloc_buffer(byte_len, storage_dtype, storage_shape)
                .unwrap();
            match storage_dtype {
                mlx_native::DType::F32 => stored.as_mut_slice::<f32>().unwrap().fill(0.0),
                mlx_native::DType::F16 | mlx_native::DType::BF16 => {
                    stored.as_mut_slice::<u16>().unwrap().fill(0)
                }
                mlx_native::DType::U8 => stored.as_mut_slice::<u8>().unwrap().fill(0),
                other => panic!("unexpected projection storage dtype {other:?}"),
            }
            let weight = MlxQWeight {
                buffer: stored,
                info: crate::serve::gpu::QuantWeightInfo {
                    ggml_dtype: dtype,
                    rows,
                    cols,
                },
                affine: None,
                decode_record_q6k_m1: std::sync::OnceLock::new(),
            };
            assert_eq!(
                supports_native_perm021(&weight, 9, 256),
                matches!(dtype, GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::Q6_K),
                "{dtype:?} perm021 admission drifted from the runtime fallback contract"
            );
            for m in [1u32, 2, 9] {
                let mut input = device
                    .alloc_buffer(
                        m as usize * cols * std::mem::size_of::<f32>(),
                        mlx_native::DType::F32,
                        vec![m as usize, cols],
                    )
                    .unwrap();
                input.as_mut_slice::<f32>().unwrap().fill(0.0);
                let output = device
                    .alloc_buffer(
                        m as usize * rows * std::mem::size_of::<f32>(),
                        mlx_native::DType::F32,
                        vec![m as usize, rows],
                    )
                    .unwrap();
                let mut registry = mlx_native::KernelRegistry::new();
                let executor = mlx_native::GraphExecutor::new(device.clone());
                let mut session = executor.begin().unwrap();
                crate::serve::forward_mlx_shared::dispatch_qmatmul(
                    &mut session,
                    &mut registry,
                    &device,
                    &input,
                    &weight,
                    &output,
                    m,
                    crate::quantize::imatrix::ImatrixHint::Global("output.weight"),
                )
                .unwrap_or_else(|error| {
                    panic!("{dtype:?} projection dispatch failed at m={m}: {error}")
                });
                session.finish().unwrap();
                assert!(
                    output
                        .as_slice::<f32>()
                        .unwrap()
                        .iter()
                        .all(|value| *value == 0.0),
                    "{dtype:?} zero projection at m={m} did not produce numeric zeros"
                );
            }

            // Exercise the exact production O-projection helper, not just its
            // capability predicate: direct codecs consume head-major BF16;
            // every other admitted codec takes the activation-only fallback.
            let m = 9u32;
            let mut head_major = device
                .alloc_buffer(
                    m as usize * cols * std::mem::size_of::<u16>(),
                    mlx_native::DType::BF16,
                    vec![1, m as usize, cols],
                )
                .unwrap();
            head_major.as_mut_slice::<u16>().unwrap().fill(0);
            let scratch = device
                .alloc_buffer(
                    m as usize * cols * std::mem::size_of::<f32>(),
                    mlx_native::DType::F32,
                    vec![m as usize, cols],
                )
                .unwrap();
            let output = device
                .alloc_buffer(
                    m as usize * rows * std::mem::size_of::<f32>(),
                    mlx_native::DType::F32,
                    vec![m as usize, rows],
                )
                .unwrap();
            let mut registry = mlx_native::KernelRegistry::new();
            let executor = mlx_native::GraphExecutor::new(device.clone());
            let mut session = executor.begin().unwrap();
            let route = crate::serve::forward_mlx_shared::dispatch_qmatmul_head_major_bf16(
                &mut session,
                &mut registry,
                &device,
                &head_major,
                &scratch,
                &weight,
                &output,
                m,
                1,
                cols,
                crate::quantize::imatrix::ImatrixHint::Global("output.weight"),
            )
            .unwrap_or_else(|error| {
                panic!("{dtype:?} production O-projection route failed: {error}")
            });
            session.finish().unwrap();
            let expected_route =
                if matches!(dtype, GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::Q6_K) {
                    crate::serve::forward_mlx_shared::HeadMajorQmatmulRoute::DirectPerm021
                } else {
                    crate::serve::forward_mlx_shared::HeadMajorQmatmulRoute::ActivationPermute
                };
            assert_eq!(route, expected_route, "{dtype:?} O-projection route drift");
            assert!(
                output
                    .as_slice::<f32>()
                    .unwrap()
                    .iter()
                    .all(|value| *value == 0.0),
                "{dtype:?} production O-projection did not produce numeric zeros"
            );
        }
    }

    #[test]
    fn tied_resolution_reuses_the_embedding_object() {
        let embedding = String::from("artifact-native embedding");
        let explicit = String::from("artifact-native output");
        assert!(std::ptr::eq(
            resolve_tied_or_explicit(&embedding, None),
            &embedding
        ));
        assert!(std::ptr::eq(
            resolve_tied_or_explicit(&embedding, Some(&explicit)),
            &explicit
        ));
    }

    #[test]
    fn expert_stacks_admit_native_id_codecs_and_reject_dense_shadow_storage() {
        let experts = 8;
        let rows = 512;
        let cols = 256;
        for dtype in [
            GgmlType::Q4_0,
            GgmlType::Q8_0,
            GgmlType::Q2_K,
            GgmlType::Q3_K,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q5_1,
            GgmlType::IQ4_NL,
            GgmlType::IQ4_XS,
        ] {
            let mut tensor = info(dtype, rows, cols, experts * stored_bytes(dtype, rows, cols));
            tensor.name = "blk.0.ffn_gate_up_exps.weight".into();
            tensor.shape = vec![experts, rows, cols];
            for role in [ExpertMatrixRole::GateUp, ExpertMatrixRole::Down] {
                admit_expert_stack(&tensor.name, &tensor, experts, rows, cols, 8, role)
                    .unwrap_or_else(|error| {
                        panic!("{dtype:?} {role:?} expert stack should be native: {error}")
                    });
            }
        }

        let mut dense = info(
            GgmlType::F16,
            rows,
            cols,
            experts * stored_bytes(GgmlType::F16, rows, cols),
        );
        dense.name = "blk.0.ffn_gate_up_exps.weight".into();
        dense.shape = vec![experts, rows, cols];
        assert!(admit_expert_stack(
            &dense.name,
            &dense,
            experts,
            rows,
            cols,
            8,
            ExpertMatrixRole::GateUp,
        )
        .unwrap_err()
        .to_string()
        .contains("rejects stored F16"));
    }

    #[test]
    fn mapped_weight_owns_file_pages_across_an_unlinked_fixture() {
        use crate::backends::gguf::writer::GgufWriter;
        use crate::quantize::ggml_quants::GgmlType as WriterGgmlType;

        let Some(device) = MlxDevice::new().ok() else {
            eprintln!("[skip] Metal device unavailable");
            return;
        };
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("native-matrix.gguf");
        let rows = 4usize;
        let cols = 256usize;
        let payload: Vec<f32> = (0..rows * cols).map(|value| value as f32).collect();
        let payload_bytes: &[u8] = bytemuck::cast_slice(&payload);
        {
            let file = std::fs::File::create(&path).unwrap();
            let mut writer = GgufWriter::new(file);
            writer.write_header(1, 0).unwrap();
            let tensor = writer
                .reserve_tensor_info(
                    "token_embd.weight",
                    &[cols as u64, rows as u64],
                    WriterGgmlType::F32,
                )
                .unwrap();
            writer.pad_to_alignment().unwrap();
            writer.stream_tensor_payload(tensor, payload_bytes).unwrap();
            writer.finalize().unwrap();
        }

        let gguf = mlx_native::gguf::GgufFile::open(&path).unwrap();
        let spec = admit_embedding(
            "token_embd.weight",
            gguf.tensor_info("token_embd.weight").unwrap(),
            rows,
            cols,
        )
        .unwrap();
        let mapped = gguf.map_tensor_data(&device).unwrap();
        let weight =
            MlxQWeight::from_mapped_gguf_tensor(&mapped, gguf.tensor_info(&spec.name).unwrap())
                .unwrap();
        assert!(weight.buffer.is_file_backed());
        assert_eq!(weight.buffer.data_byte_len(), payload_bytes.len());

        drop(mapped);
        drop(gguf);
        std::fs::remove_file(&path).unwrap();
        let retained = weight.buffer.as_slice::<f32>().unwrap();
        assert_eq!(retained[0], 0.0);
        assert_eq!(retained[513], 513.0);
        assert_eq!(retained[rows * cols - 1], (rows * cols - 1) as f32);
    }
}
