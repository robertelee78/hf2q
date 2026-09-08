//! Runtime-owned metadata and topology admission; shared with hosted preflight.

pub(crate) fn validate_qwen_runtime_admission(
    gguf: &mlx_native::gguf::GgufFile,
) -> Result<(), String> {
    let key_prefix = match gguf.metadata_string("general.architecture").unwrap_or("") {
        "qwen35" => "qwen35",
        "qwen35moe" => "qwen35moe",
        _ => return Ok(()),
    };
    for name in gguf.tensor_names() {
        let info = gguf.tensor_info(name).expect("tensor directory entry");
        if let Some(reason) =
            super::tensor_admission::tensor_incompatibility(key_prefix, name, info.ggml_type)
        {
            return Err(reason);
        }
    }
    let block_count = gguf
        .metadata_u32(&format!("{key_prefix}.block_count"))
        .ok_or_else(|| format!("GGUF is missing required {key_prefix}.block_count metadata"))?;
    let tensor_count = gguf.tensor_names().len() as u64;
    if block_count == 0 || block_count > 4096 || u64::from(block_count) > tensor_count {
        return Err(format!(
            "GGUF declares invalid {key_prefix}.block_count={block_count} for {tensor_count} tensors"
        ));
    }
    let cfg = crate::inference::models::qwen35::Qwen35Config::from_gguf(gguf)
        .map_err(|error| format!("Qwen runtime metadata admission failed: {error}"))?;
    crate::inference::models::qwen35::tokenizer::build_tokenizer_from_gguf(gguf)
        .map_err(|error| format!("Qwen tokenizer metadata admission failed: {error}"))?;
    validate_qwen_operational_config(&cfg)
        .map_err(|reason| format!("Qwen operational config admission failed: {reason}"))?;
    validate_qwen_hosted_topology(gguf, &cfg)
        .map_err(|reason| format!("Qwen tensor topology admission failed: {reason}"))?;
    crate::inference::models::qwen35::mtp_weights_load::validate_mtp_tensor_topology(gguf, &cfg)
        .map_err(|error| format!("Qwen MTP topology admission failed: {error}"))
}

pub(crate) fn validate_qwen_operational_config(
    cfg: &crate::inference::models::qwen35::Qwen35Config,
) -> Result<(), String> {
    if cfg.hidden_size == 0
        || cfg.num_hidden_layers == 0
        || cfg.num_attention_heads == 0
        || cfg.num_key_value_heads == 0
        || cfg.head_dim == 0
    {
        return Err("hidden, layer, Q-head, KV-head, and head dimensions must be nonzero".into());
    }
    if cfg.num_attention_heads % cfg.num_key_value_heads != 0 {
        return Err(format!(
            "Q head count {} is not divisible by KV head count {}",
            cfg.num_attention_heads, cfg.num_key_value_heads
        ));
    }
    if cfg.linear_num_key_heads == 0
        || cfg.linear_num_value_heads == 0
        || cfg.linear_num_value_heads % cfg.linear_num_key_heads != 0
        || cfg.linear_key_head_dim == 0
        || cfg.linear_value_head_dim == 0
        || cfg.linear_conv_kernel_dim == 0
    {
        return Err("linear-attention head counts/dimensions/kernel are not executable".into());
    }
    let mrope_sum = cfg.mrope_section.iter().try_fold(0_u32, |sum, value| {
        sum.checked_add(*value)
            .ok_or_else(|| "mRoPE section sum overflow".to_owned())
    })?;
    if cfg.rotary_dim == 0
        || cfg.rotary_dim > cfg.head_dim
        || cfg.rotary_dim % 2 != 0
        || mrope_sum != cfg.rotary_dim / 2
    {
        return Err(format!(
            "rotary/mRoPE dimensions are incoherent: rotary_dim={}, head_dim={}, sections={:?}",
            cfg.rotary_dim, cfg.head_dim, cfg.mrope_section
        ));
    }
    if !cfg.rope_theta.is_finite()
        || cfg.rope_theta <= 0.0
        || !cfg.rms_norm_eps.is_finite()
        || cfg.rms_norm_eps <= 0.0
        || cfg.max_position_embeddings == 0
        || cfg.vocab_size == 0
    {
        return Err(
            "rope, norm, context, and vocabulary scalars must be finite and positive".into(),
        );
    }
    match cfg.variant {
        crate::inference::models::qwen35::Qwen35Variant::Dense => {
            if cfg.intermediate_size.is_none_or(|value| value == 0) {
                return Err("dense feed-forward length must be nonzero".into());
            }
        }
        crate::inference::models::qwen35::Qwen35Variant::Moe => {
            let moe = cfg
                .moe
                .as_ref()
                .ok_or_else(|| "MoE configuration is absent".to_owned())?;
            if moe.num_experts == 0
                || moe.num_experts_per_tok == 0
                || moe.num_experts_per_tok > moe.num_experts
                || moe.moe_intermediate_size == 0
                || moe.shared_expert_intermediate_size == 0
            {
                return Err(format!(
                    "MoE expert routing is not executable: experts={}, top_k={}, expert_ffn={}, shared_ffn={}",
                    moe.num_experts,
                    moe.num_experts_per_tok,
                    moe.moe_intermediate_size,
                    moe.shared_expert_intermediate_size
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_qwen_hosted_topology(
    gguf: &mlx_native::gguf::GgufFile,
    cfg: &crate::inference::models::qwen35::Qwen35Config,
) -> Result<(), String> {
    use crate::inference::models::qwen35::{Qwen35LayerKind, Qwen35Variant};

    fn checked_product(values: &[u32], label: &str) -> Result<usize, String> {
        values.iter().try_fold(1_usize, |product, value| {
            product
                .checked_mul(*value as usize)
                .ok_or_else(|| format!("{label} dimension product overflow"))
        })
    }

    fn require_shape(
        gguf: &mlx_native::gguf::GgufFile,
        name: &str,
        expected: &[usize],
    ) -> Result<(), String> {
        let info = gguf
            .tensor_info(name)
            .ok_or_else(|| format!("missing required tensor `{name}`"))?;
        if info.shape != expected {
            return Err(format!(
                "tensor `{name}` shape {:?} != expected {expected:?}",
                info.shape
            ));
        }
        Ok(())
    }

    let h = cfg.hidden_size as usize;
    if h == 0 || cfg.num_hidden_layers == 0 {
        return Err("hidden size and normal block count must be nonzero".into());
    }
    let token_count = match gguf.metadata("tokenizer.ggml.tokens") {
        Some(mlx_native::gguf::MetadataValue::Array(tokens)) if !tokens.is_empty() => tokens.len(),
        _ => return Err("missing nonempty tokenizer.ggml.tokens array".into()),
    };
    let token = gguf
        .tensor_info("token_embd.weight")
        .ok_or_else(|| "missing required tensor `token_embd.weight`".to_owned())?;
    if token.shape.len() != 2 || token.shape[1] != h || token.shape[0] < token_count {
        return Err(format!(
            "token_embd.weight shape {:?} cannot cover {token_count} tokenizer rows at hidden size {h}",
            token.shape
        ));
    }
    require_shape(gguf, "output_norm.weight", &[h])?;
    let output_rows = gguf
        .tensor_info("output.weight")
        .map(|output| output.shape.first().copied())
        .flatten()
        .unwrap_or(token.shape[0]);
    if let Some(output) = gguf.tensor_info("output.weight") {
        if output.shape.len() != 2 || output.shape[1] != h {
            return Err(format!(
                "output.weight shape {:?} is not [vocab,{h}]",
                output.shape
            ));
        }
    }
    if token.shape[0] < output_rows {
        return Err(format!(
            "token_embd.weight rows {} cannot cover resolved output-head rows {output_rows}",
            token.shape[0]
        ));
    }

    let q_rows = checked_product(&[cfg.num_attention_heads, cfg.head_dim], "full-attention Q")?;
    let kv_rows = checked_product(
        &[cfg.num_key_value_heads, cfg.head_dim],
        "full-attention KV",
    )?;
    let nk_d = checked_product(
        &[cfg.linear_num_key_heads, cfg.linear_key_head_dim],
        "linear-attention K",
    )?;
    let nv_d = checked_product(
        &[cfg.linear_num_value_heads, cfg.linear_value_head_dim],
        "linear-attention V",
    )?;
    let qkv_rows = nk_d
        .checked_mul(2)
        .and_then(|value| value.checked_add(nv_d))
        .ok_or_else(|| "linear-attention QKV dimension overflow".to_owned())?;
    let q_projection_rows = q_rows
        .checked_mul(2)
        .ok_or_else(|| "full-attention gated Q projection dimension overflow".to_owned())?;

    if cfg.layer_types.len() != cfg.num_hidden_layers as usize {
        return Err("runtime layer-kind topology length differs from block count".into());
    }
    for (layer, kind) in cfg.layer_types.iter().copied().enumerate() {
        let p = format!("blk.{layer}");
        require_shape(gguf, &format!("{p}.attn_norm.weight"), &[h])?;
        require_shape(gguf, &format!("{p}.post_attention_norm.weight"), &[h])?;
        match kind {
            Qwen35LayerKind::FullAttention => {
                require_shape(gguf, &format!("{p}.attn_q.weight"), &[q_projection_rows, h])?;
                require_shape(gguf, &format!("{p}.attn_k.weight"), &[kv_rows, h])?;
                require_shape(gguf, &format!("{p}.attn_v.weight"), &[kv_rows, h])?;
                require_shape(
                    gguf,
                    &format!("{p}.attn_q_norm.weight"),
                    &[cfg.head_dim as usize],
                )?;
                require_shape(
                    gguf,
                    &format!("{p}.attn_k_norm.weight"),
                    &[cfg.head_dim as usize],
                )?;
                require_shape(gguf, &format!("{p}.attn_output.weight"), &[h, q_rows])?;
            }
            Qwen35LayerKind::LinearAttention => {
                require_shape(gguf, &format!("{p}.attn_qkv.weight"), &[qkv_rows, h])?;
                require_shape(gguf, &format!("{p}.attn_gate.weight"), &[nv_d, h])?;
                require_shape(
                    gguf,
                    &format!("{p}.ssm_conv1d.weight"),
                    &[qkv_rows, cfg.linear_conv_kernel_dim as usize],
                )?;
                require_shape(
                    gguf,
                    &format!("{p}.ssm_alpha.weight"),
                    &[cfg.linear_num_value_heads as usize, h],
                )?;
                require_shape(
                    gguf,
                    &format!("{p}.ssm_beta.weight"),
                    &[cfg.linear_num_value_heads as usize, h],
                )?;
                require_shape(
                    gguf,
                    &format!("{p}.ssm_dt.bias"),
                    &[cfg.linear_num_value_heads as usize],
                )?;
                require_shape(
                    gguf,
                    &format!("{p}.ssm_a"),
                    &[cfg.linear_num_value_heads as usize],
                )?;
                require_shape(
                    gguf,
                    &format!("{p}.ssm_norm.weight"),
                    &[cfg.linear_value_head_dim as usize],
                )?;
                require_shape(gguf, &format!("{p}.ssm_out.weight"), &[h, nv_d])?;
            }
        }

        match cfg.variant {
            Qwen35Variant::Dense => {
                let intermediate = cfg
                    .intermediate_size
                    .ok_or_else(|| "dense Qwen config has no intermediate size".to_owned())?
                    as usize;
                require_shape(gguf, &format!("{p}.ffn_gate.weight"), &[intermediate, h])?;
                require_shape(gguf, &format!("{p}.ffn_up.weight"), &[intermediate, h])?;
                require_shape(gguf, &format!("{p}.ffn_down.weight"), &[h, intermediate])?;
                let tensor_type = |role: &str| {
                    gguf.tensor_info(&format!("{p}.ffn_{role}.weight"))
                        .map(|info| info.ggml_type)
                        .ok_or_else(|| format!("{p}.ffn_{role}.weight is missing"))
                };
                crate::inference::models::qwen35::weight_loader::validate_qwen35_dense_ffn_storage(
                    layer as u32,
                    tensor_type("gate")?,
                    tensor_type("up")?,
                    tensor_type("down")?,
                )
                .map_err(|error| error.to_string())?;
            }
            Qwen35Variant::Moe => {
                let moe = cfg
                    .moe
                    .as_ref()
                    .ok_or_else(|| "MoE Qwen config has no expert topology".to_owned())?;
                let experts = moe.num_experts as usize;
                let expert_intermediate = moe.moe_intermediate_size as usize;
                let shared_intermediate = moe.shared_expert_intermediate_size as usize;
                require_shape(gguf, &format!("{p}.ffn_gate_inp.weight"), &[experts, h])?;
                require_shape(
                    gguf,
                    &format!("{p}.ffn_gate_exps.weight"),
                    &[experts, expert_intermediate, h],
                )?;
                require_shape(
                    gguf,
                    &format!("{p}.ffn_up_exps.weight"),
                    &[experts, expert_intermediate, h],
                )?;
                require_shape(
                    gguf,
                    &format!("{p}.ffn_down_exps.weight"),
                    &[experts, h, expert_intermediate],
                )?;
                // hf2q's Qwen3.5/3.6 converter deliberately squeezes the HF
                // `[1, hidden]` shared-expert gate to canonical GGUF `[hidden]`
                // so the scalar router stays F32. Runtime consumes it as one
                // length-hidden dot-product vector.
                require_shape(
                    gguf,
                    &format!("{p}.ffn_gate_inp_shexp.weight"),
                    &crate::inference::models::qwen35::shared_expert_gate_shape(h),
                )?;
                require_shape(
                    gguf,
                    &format!("{p}.ffn_gate_shexp.weight"),
                    &[shared_intermediate, h],
                )?;
                require_shape(
                    gguf,
                    &format!("{p}.ffn_up_shexp.weight"),
                    &[shared_intermediate, h],
                )?;
                require_shape(
                    gguf,
                    &format!("{p}.ffn_down_shexp.weight"),
                    &[h, shared_intermediate],
                )?;
                let gate = gguf
                    .tensor_info(&format!("{p}.ffn_gate_exps.weight"))
                    .expect("shape validated");
                let up = gguf
                    .tensor_info(&format!("{p}.ffn_up_exps.weight"))
                    .expect("shape validated");
                if gate.ggml_type != up.ggml_type {
                    return Err(format!(
                        "{p} expert gate/up GGML types differ ({:?} vs {:?})",
                        gate.ggml_type, up.ggml_type
                    ));
                }
            }
        }
    }
    Ok(())
}
