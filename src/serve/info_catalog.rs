//! Header-only family/context and tensor-directory validation shared by
//! `hf2q info` and the pre-load serve path.

use std::path::Path;

use anyhow::{Context, Result};
use mlx_native::gguf::GgufFile;

use crate::inference::models::qwen35::{Qwen35Config, Qwen35LayerKind};

use super::config::Gemma4Config;
use super::operator_settings::ResolvedContext;

/// Enforce architecture-specific logical-context floors while the caller has
/// only mapped the GGUF header. This keeps `serve` and `info` aligned and
/// prevents a huge model upload that can only fail after its cache geometry is
/// examined.
pub(crate) fn validate_family_context_floor(
    gguf: &GgufFile,
    context: ResolvedContext,
) -> Result<(), String> {
    if gguf.metadata_string("general.architecture") == Some("deepseek4") {
        let config = crate::inference::models::deepseek4::Deepseek4Model::load_config_only(gguf)
            .map_err(|error| format!("DeepSeek-V4 metadata/config validation failed: {error:#}"))?;
        if context.effective_tokens < config.sliding_window {
            return Err(format!(
                "DeepSeek-V4 serving context {} must be at least its {}-token native window",
                context.effective_tokens, config.sliding_window
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_tensor_headers(gguf: &GgufFile, path: &Path) -> Result<u64> {
    anyhow::ensure!(gguf.tensor_count() > 0, "GGUF tensor directory is empty");
    let file_bytes = std::fs::metadata(path)
        .with_context(|| format!("read GGUF size for {}", path.display()))?
        .len();
    for name in gguf.tensor_names() {
        let info = gguf
            .tensor_info(name)
            .ok_or_else(|| anyhow::anyhow!("tensor directory entry {name:?} cannot be read"))?;
        anyhow::ensure!(
            !info.shape.is_empty() && info.shape.iter().all(|dimension| *dimension > 0),
            "tensor {name:?} has invalid shape {:?}",
            info.shape
        );
        anyhow::ensure!(info.byte_len > 0, "tensor {name:?} has zero encoded bytes");
        let encoded_bytes = u64::try_from(info.byte_len)
            .map_err(|_| anyhow::anyhow!("tensor {name:?} byte length exceeds u64"))?;
        let absolute_end = gguf
            .tensor_data_offset()
            .checked_add(info.offset)
            .and_then(|start| start.checked_add(encoded_bytes))
            .ok_or_else(|| anyhow::anyhow!("tensor {name:?} file range overflows u64"))?;
        anyhow::ensure!(
            absolute_end <= file_bytes,
            "tensor {name:?} ends at byte {absolute_end}, beyond the {}-byte GGUF",
            file_bytes
        );
        let _parsed_type = info.ggml_type;
    }
    Ok(file_bytes)
}

fn require_tensor(gguf: &GgufFile, name: &str) -> Result<()> {
    anyhow::ensure!(
        gguf.tensor_info(name).is_some(),
        "required tensor {name:?} is missing"
    );
    Ok(())
}

fn require_shape(gguf: &GgufFile, name: &str, expected: &[usize]) -> Result<()> {
    let info = gguf
        .tensor_info(name)
        .ok_or_else(|| anyhow::anyhow!("required tensor {name:?} is missing"))?;
    anyhow::ensure!(
        info.shape == expected,
        "tensor {name:?} shape {:?} does not match expected {:?}",
        info.shape,
        expected
    );
    Ok(())
}

pub(super) fn validate_gemma_tensors(gguf: &GgufFile, cfg: &Gemma4Config) -> Result<()> {
    require_shape(
        gguf,
        "token_embd.weight",
        &[cfg.vocab_size, cfg.hidden_size],
    )?;
    require_shape(gguf, "output_norm.weight", &[cfg.hidden_size])?;
    for layer in 0..cfg.num_hidden_layers {
        for suffix in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_output.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
            "attn_norm.weight",
            "post_attention_norm.weight",
            "ffn_norm.weight",
            "post_ffw_norm.weight",
            "layer_output_scale.weight",
        ] {
            require_tensor(gguf, &format!("blk.{layer}.{suffix}"))?;
        }
        if !(cfg.is_full_attention(layer) && cfg.attention_k_eq_v) {
            require_tensor(gguf, &format!("blk.{layer}.attn_v.weight"))?;
        }
        let gate_up = format!("blk.{layer}.ffn_gate_up_exps.weight");
        let down = format!("blk.{layer}.ffn_down_exps.weight");
        match (
            gguf.tensor_info(&gate_up).is_some(),
            gguf.tensor_info(&down).is_some(),
        ) {
            (true, true) => {
                for suffix in [
                    "ffn_gate_inp.weight",
                    "ffn_gate_inp.scale",
                    "ffn_down_exps.scale",
                ] {
                    require_tensor(gguf, &format!("blk.{layer}.{suffix}"))?;
                }
            }
            (false, false) => {}
            _ => anyhow::bail!(
                "Gemma layer {layer} has an incomplete MoE expert pair; both {gate_up:?} and {down:?} are required together"
            ),
        }
    }
    Ok(())
}

pub(super) fn validate_qwen35_tensors(gguf: &GgufFile, cfg: &Qwen35Config) -> Result<()> {
    let token_embedding = gguf
        .tensor_info("token_embd.weight")
        .ok_or_else(|| anyhow::anyhow!("required tensor \"token_embd.weight\" is missing"))?;
    anyhow::ensure!(
        token_embedding.shape.len() == 2 && token_embedding.shape[1] == cfg.hidden_size as usize,
        "tensor \"token_embd.weight\" shape {:?} is not [rows, hidden_size={}]",
        token_embedding.shape,
        cfg.hidden_size
    );
    require_shape(gguf, "output_norm.weight", &[cfg.hidden_size as usize])?;
    if let Some(output) = gguf.tensor_info("output.weight") {
        anyhow::ensure!(
            output.shape.len() == 2 && output.shape[1] == cfg.hidden_size as usize,
            "tensor \"output.weight\" shape {:?} is not [rows, hidden_size={}]",
            output.shape,
            cfg.hidden_size
        );
        anyhow::ensure!(
            token_embedding.shape[0] >= output.shape[0],
            "token_embd.weight rows {} are fewer than output.weight rows {}",
            token_embedding.shape[0],
            output.shape[0]
        );
    }
    for (layer, kind) in cfg.layer_types.iter().enumerate() {
        let prefix = format!("blk.{layer}");
        for suffix in ["attn_norm.weight", "post_attention_norm.weight"] {
            require_tensor(gguf, &format!("{prefix}.{suffix}"))?;
        }
        match kind {
            Qwen35LayerKind::FullAttention => {
                for suffix in [
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "attn_output.weight",
                    "attn_q_norm.weight",
                    "attn_k_norm.weight",
                ] {
                    require_tensor(gguf, &format!("{prefix}.{suffix}"))?;
                }
            }
            Qwen35LayerKind::LinearAttention => {
                for suffix in [
                    "attn_qkv.weight",
                    "attn_gate.weight",
                    "ssm_alpha.weight",
                    "ssm_beta.weight",
                    "ssm_out.weight",
                    "ssm_conv1d.weight",
                    "ssm_dt.bias",
                    "ssm_a",
                    "ssm_norm.weight",
                ] {
                    require_tensor(gguf, &format!("{prefix}.{suffix}"))?;
                }
            }
        }
        let ffn_suffixes: &[&str] = if cfg.moe.is_some() {
            &[
                "ffn_gate_inp.weight",
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
                "ffn_gate_inp_shexp.weight",
                "ffn_gate_shexp.weight",
                "ffn_up_shexp.weight",
                "ffn_down_shexp.weight",
            ]
        } else {
            &["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]
        };
        for suffix in ffn_suffixes {
            require_tensor(gguf, &format!("{prefix}.{suffix}"))?;
        }
        if cfg.moe.is_some() {
            use mlx_native::ops::quantized_matmul_ggml::GgmlType;
            for suffix in [
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
            ] {
                let name = format!("{prefix}.{suffix}");
                let info = gguf
                    .tensor_info(&name)
                    .ok_or_else(|| anyhow::anyhow!("required tensor {name:?} is missing"))?;
                anyhow::ensure!(
                    !matches!(info.ggml_type, GgmlType::F16 | GgmlType::F32),
                    "MoE expert tensor {name:?} has unsupported {:?} storage; native GGML block quantization is required",
                    info.ggml_type
                );
            }
        }
    }
    validate_qwen35_mtp_tensors(gguf, cfg)
}

fn validate_qwen35_mtp_tensors(gguf: &GgufFile, cfg: &Qwen35Config) -> Result<()> {
    if cfg.mtp_num_hidden_layers == 0 {
        return Ok(());
    }
    anyhow::ensure!(
        cfg.mtp_num_hidden_layers == 1,
        "Qwen MTP serving supports exactly one next-token layer, GGUF declares {}",
        cfg.mtp_num_hidden_layers
    );
    let layer = cfg.num_hidden_layers;
    let prefix = format!("blk.{layer}");
    for suffix in [
        "nextn.enorm.weight",
        "nextn.hnorm.weight",
        "nextn.eh_proj.weight",
        "nextn.shared_head_norm.weight",
        "attn_norm.weight",
        "post_attention_norm.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
    ] {
        require_tensor(gguf, &format!("{prefix}.{suffix}"))?;
    }
    let dedicated_embedding = format!("{prefix}.nextn.embed_tokens.weight");
    if cfg.mtp_use_dedicated_embeddings {
        require_tensor(gguf, &dedicated_embedding)?;
    } else {
        anyhow::ensure!(
            gguf.tensor_info(&dedicated_embedding).is_none(),
            "Qwen MTP metadata selects shared embeddings but {dedicated_embedding:?} is present"
        );
    }
    let dedicated_head = format!("{prefix}.nextn.shared_head_head.weight");
    anyhow::ensure!(
        !cfg.mtp_use_dedicated_embeddings || gguf.tensor_info(&dedicated_head).is_some(),
        "Qwen MTP metadata selects dedicated embeddings but {dedicated_head:?} is missing"
    );

    let dense = gguf
        .tensor_info(&format!("{prefix}.ffn_gate.weight"))
        .is_some();
    let moe = gguf
        .tensor_info(&format!("{prefix}.ffn_gate_exps.weight"))
        .is_some();
    anyhow::ensure!(
        dense ^ moe,
        "Qwen MTP block {layer} must contain exactly one dense or MoE FFN schema"
    );
    let suffixes: &[&str] = if dense {
        &["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]
    } else {
        &[
            "ffn_gate_inp.weight",
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
            "ffn_gate_inp_shexp.weight",
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
        ]
    };
    for suffix in suffixes {
        require_tensor(gguf, &format!("{prefix}.{suffix}"))?;
    }
    Ok(())
}
