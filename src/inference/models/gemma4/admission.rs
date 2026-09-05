//! Header-only validation of the tensors consumed by the Gemma runtime graph.
//! Shapes belong to hf2q; execution/storage capabilities belong to mlx-native.

use anyhow::{ensure, Context, Result};
use mlx_native::gguf::{GgufFile, MetadataValue};
use mlx_native::{
    DType, DenseMatmulIdInputLayout, DenseMatmulIdMultiplicity, DenseMatmulIdParams,
    DenseMatmulIdRoute, GgmlCapabilityRequest, GgmlExpertShape, GgmlInvocation, GgmlRoutingPolicy,
    GgmlType, GgmlWorkloadClass, GGML_CAPABILITY_SCHEMA_VERSION,
};

use crate::serve::config::Gemma4Config;

pub(crate) fn validate(gguf: &GgufFile) -> Result<()> {
    let count = gguf.metadata_u32("gemma4.block_count").unwrap_or(0);
    ensure!(
        count > 0 && count <= 4096 && count as usize <= gguf.tensor_count(),
        "invalid Gemma block count {count} for bounded tensor directory"
    );
    let cfg = Gemma4Config::from_gguf(gguf)?;
    validate_tensors(gguf, &cfg)?;
    super::tokenizer::build_tokenizer_from_gguf(gguf)
        .context("Gemma embedded tokenizer admission")?;
    let template = gguf
        .metadata_string("tokenizer.chat_template")
        .unwrap_or(crate::serve::FALLBACK_GEMMA4_API_CHAT_TEMPLATE);
    crate::core::chat_templates::validate_tool_chat_template("gemma4", template)
        .map_err(|error| anyhow::anyhow!("Gemma chat-template admission: {error}"))?;
    Ok(())
}

fn tensor<'a>(gguf: &'a GgufFile, name: &str, shape: &[usize]) -> Result<GgmlType> {
    let info = gguf
        .tensor_info(name)
        .ok_or_else(|| anyhow::anyhow!("missing required tensor {name:?}"))?;
    ensure!(
        info.shape == shape,
        "tensor {name:?} shape {:?} != {shape:?}",
        info.shape
    );
    Ok(info.ggml_type)
}

fn scalar(gguf: &GgufFile, name: &str, shape: &[usize]) -> Result<()> {
    let dtype = tensor(gguf, name, shape)?;
    ensure!(
        matches!(dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16),
        "tensor {name:?} requires native floating-point scalar storage, got {dtype:?}"
    );
    Ok(())
}

fn dense(gguf: &GgufFile, name: &str, n: usize, k: usize) -> Result<()> {
    let dtype = tensor(gguf, name, &[n, k])?;
    for (m, workload) in [
        (1, GgmlWorkloadClass::DecodeSingle),
        (8, GgmlWorkloadClass::ContinuousWidth),
        (32, GgmlWorkloadClass::Prompt),
    ] {
        let capability = mlx_native::ggml_capability(GgmlCapabilityRequest {
            schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
            invocation: GgmlInvocation::DenseAuto {
                m,
                n: n.try_into()?,
                k: k.try_into()?,
            },
            ggml_type: dtype,
            workload,
            routing: GgmlRoutingPolicy::default(),
        });
        ensure!(
            capability.executable,
            "tensor {name:?}: {}",
            capability.diagnostic
        );
    }
    Ok(())
}

fn experts(
    gguf: &GgufFile,
    name: &str,
    cfg: &Gemma4Config,
    n: usize,
    k: usize,
    down: bool,
) -> Result<()> {
    let dtype = tensor(gguf, name, &[cfg.num_experts, n, k])?;
    let info = gguf
        .tensor_info(name)
        .context("validated expert tensor disappeared")?;
    let stride = (info.byte_len / cfg.num_experts) as u64;
    let scalar_dtype = match dtype {
        GgmlType::F32 => Some(DType::F32),
        GgmlType::F16 => Some(DType::F16),
        GgmlType::BF16 => Some(DType::BF16),
        _ => None,
    };
    for tokens in [1_usize, 8, 32] {
        let (m, top_k) = if down {
            (
                tokens
                    .checked_mul(cfg.top_k_experts)
                    .context("expert row overflow")?
                    .try_into()?,
                1,
            )
        } else {
            (tokens.try_into()?, cfg.top_k_experts.try_into()?)
        };
        if let Some(weight_dtype) = scalar_dtype {
            mlx_native::dense_matmul_id_capability(
                weight_dtype,
                &DenseMatmulIdParams {
                    m,
                    n: n.try_into()?,
                    k: k.try_into()?,
                    top_k,
                    n_experts: cfg.num_experts.try_into()?,
                    expert_stride_bytes: stride,
                    input_layout: DenseMatmulIdInputLayout::SharedPerToken,
                    id_multiplicity: DenseMatmulIdMultiplicity::MayRepeat,
                    route: DenseMatmulIdRoute::Direct,
                },
            )
            .with_context(|| format!("tensor {name:?} native scalar expert capability"))?;
        } else {
            let shape = GgmlExpertShape {
                n_tokens: m,
                n: n.try_into()?,
                k: k.try_into()?,
                top_k,
                n_experts: cfg.num_experts.try_into()?,
                expert_stride_bytes: stride,
                ids_are_distinct_per_token: true,
                ids_within_expert_range: true,
            };
            let capability = mlx_native::ggml_capability(GgmlCapabilityRequest {
                schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
                invocation: GgmlInvocation::ExpertAutoAllocated { shape },
                ggml_type: dtype,
                workload: if m == 1 {
                    GgmlWorkloadClass::DecodeSingle
                } else {
                    GgmlWorkloadClass::Prompt
                },
                routing: GgmlRoutingPolicy::default(),
            });
            ensure!(
                capability.executable,
                "tensor {name:?}: {}",
                capability.diagnostic
            );
        }
    }
    Ok(())
}

pub(crate) fn validate_tensors(gguf: &GgufFile, cfg: &Gemma4Config) -> Result<()> {
    for (name, value) in [
        ("hidden size", cfg.hidden_size),
        ("vocabulary", cfg.vocab_size),
        ("feed-forward size", cfg.intermediate_size),
        ("layers", cfg.num_hidden_layers),
        ("query heads", cfg.num_attention_heads),
        ("KV heads", cfg.num_key_value_heads),
        ("global KV heads", cfg.num_global_key_value_heads),
        ("head dimension", cfg.head_dim),
        ("global head dimension", cfg.global_head_dim),
        ("context", cfg.max_position_embeddings),
    ] {
        ensure!(
            value > 0 && value <= u32::MAX as usize,
            "invalid {name}: {value}"
        );
    }
    ensure!(
        cfg.num_hidden_layers <= 4096 && cfg.layer_types.len() == cfg.num_hidden_layers,
        "Gemma layer count/pattern is outside the bounded runtime contract"
    );
    ensure!(
        cfg.num_attention_heads % cfg.num_key_value_heads == 0
            && cfg.num_attention_heads % cfg.num_global_key_value_heads == 0,
        "query heads must be divisible by KV heads"
    );
    ensure!(
        cfg.head_dim % 2 == 0 && cfg.global_head_dim % 2 == 0,
        "rotary head dimensions must be even"
    );
    for value in [
        cfg.rms_norm_eps,
        cfg.rope_theta_sliding,
        cfg.rope_theta_global,
    ] {
        ensure!(
            value.is_finite() && value > 0.0,
            "non-executable normalization/rotary scalar"
        );
    }
    ensure!(
        cfg.final_logit_softcapping
            .is_none_or(|v| v.is_finite() && v > 0.0),
        "invalid logit soft cap"
    );
    ensure!(
        cfg.num_experts == 0
            || (cfg.moe_intermediate_size > 0
                && cfg.top_k_experts > 0
                && cfg.top_k_experts <= cfg.num_experts),
        "invalid expert geometry"
    );
    // These metadata features require additional graph operations. A familiar
    // architecture string must not silently activate an incomplete graph.
    for key in [
        "gemma4.attention.shared_kv_layers",
        "gemma4.embedding_length_per_layer_input",
    ] {
        ensure!(
            gguf.metadata_u32(key).unwrap_or(0) == 0,
            "runtime does not implement nonzero {key}"
        );
    }
    let h = cfg.hidden_size;
    tensor(gguf, "token_embd.weight", &[cfg.vocab_size, h])?;
    let token_count = match gguf.metadata("tokenizer.ggml.tokens") {
        Some(MetadataValue::Array(tokens)) => tokens.len(),
        _ => 0,
    };
    ensure!(
        token_count > 0 && token_count <= cfg.vocab_size,
        "tokenizer vocabulary does not fit embedding rows"
    );
    scalar(gguf, "output_norm.weight", &[h])?;
    ensure!(gguf.tensor_info("output.weight").is_none(),
        "Gemma runtime requires a tied embedding/output head; distinct output.weight is not implemented");
    for layer in 0..cfg.num_hidden_layers {
        let p = format!("blk.{layer}");
        let full = cfg.is_full_attention(layer);
        let d = if full {
            cfg.global_head_dim
        } else {
            cfg.head_dim
        };
        let kv = if full {
            cfg.num_global_key_value_heads
        } else {
            cfg.num_key_value_heads
        };
        let q_width = cfg
            .num_attention_heads
            .checked_mul(d)
            .context("Q shape overflow")?;
        let kv_width = kv.checked_mul(d).context("KV shape overflow")?;
        dense(gguf, &format!("{p}.attn_q.weight"), q_width, h)?;
        dense(gguf, &format!("{p}.attn_k.weight"), kv_width, h)?;
        if !(full && cfg.attention_k_eq_v) {
            dense(gguf, &format!("{p}.attn_v.weight"), kv_width, h)?;
        }
        dense(gguf, &format!("{p}.attn_output.weight"), h, q_width)?;
        scalar(gguf, &format!("{p}.attn_q_norm.weight"), &[d])?;
        scalar(gguf, &format!("{p}.attn_k_norm.weight"), &[d])?;
        for suffix in [
            "attn_norm",
            "post_attention_norm",
            "ffn_norm",
            "post_ffw_norm",
        ] {
            scalar(gguf, &format!("{p}.{suffix}.weight"), &[h])?;
        }
        for suffix in ["pre_ffw_norm_2", "post_ffw_norm_1", "post_ffw_norm_2"] {
            let name = format!("{p}.{suffix}.weight");
            if gguf.tensor_info(&name).is_some() {
                scalar(gguf, &name, &[h])?;
            }
        }
        scalar(gguf, &format!("{p}.layer_output_scale.weight"), &[1])?;
        for suffix in ["ffn_gate", "ffn_up"] {
            dense(
                gguf,
                &format!("{p}.{suffix}.weight"),
                cfg.intermediate_size,
                h,
            )?;
        }
        dense(
            gguf,
            &format!("{p}.ffn_down.weight"),
            h,
            cfg.intermediate_size,
        )?;
        let gate = format!("{p}.ffn_gate_up_exps.weight");
        let down = format!("{p}.ffn_down_exps.weight");
        let has_gate = gguf.tensor_info(&gate).is_some();
        let has_down = gguf.tensor_info(&down).is_some();
        ensure!(has_gate == has_down, "incomplete expert pair at {p}");
        if has_gate {
            ensure!(
                cfg.num_experts > 0,
                "expert tensors without expert configuration"
            );
            experts(
                gguf,
                &gate,
                cfg,
                cfg.moe_intermediate_size
                    .checked_mul(2)
                    .context("expert width overflow")?,
                h,
                false,
            )?;
            experts(gguf, &down, cfg, h, cfg.moe_intermediate_size, true)?;
            dense(
                gguf,
                &format!("{p}.ffn_gate_inp.weight"),
                cfg.num_experts,
                h,
            )?;
            scalar(gguf, &format!("{p}.ffn_gate_inp.scale"), &[h])?;
            scalar(
                gguf,
                &format!("{p}.ffn_down_exps.scale"),
                &[cfg.num_experts],
            )?;
        }
    }
    Ok(())
}
