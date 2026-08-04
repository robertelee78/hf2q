//! Strict DeepSeek-V4 GGUF architecture contract.

use anyhow::{anyhow, bail, Result};
use mlx_native::gguf::{GgufFile, MetadataValue};

use super::ARCH_DEEPSEEK4;

#[derive(Clone, Debug, PartialEq)]
pub struct Deepseek4Config {
    pub hidden_size: u32,
    pub hidden_size_out: u32,
    pub num_hidden_layers: u32,
    pub max_position_embeddings: u32,
    pub vocab_size: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub rope_head_dim: u32,
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub original_context_length: u32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub q_lora_rank: u32,
    pub o_lora_rank: u32,
    pub output_groups: u32,
    pub sliding_window: u32,
    pub compress_ratios: Vec<u32>,
    pub compress_rope_theta: f32,
    pub index_num_heads: u32,
    pub index_head_dim: u32,
    pub index_top_k: u32,
    pub rms_norm_eps: f32,
    pub num_experts: u32,
    pub num_experts_per_tok: u32,
    pub num_shared_experts: u32,
    pub expert_intermediate_size: u32,
    pub route_scale: f32,
    pub normalize_topk: bool,
    pub swiglu_clamp_experts: Vec<f32>,
    pub swiglu_clamp_shared: Vec<f32>,
    pub hyper_connection_count: u32,
    pub hyper_connection_sinkhorn_iterations: u32,
    pub hyper_connection_epsilon: f32,
    pub hash_layer_count: u32,
}

fn required_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
    gguf.metadata_u32(key)
        .ok_or_else(|| anyhow!("DeepSeek-V4 config: required u32 key {key:?} missing"))
}

fn required_f32(gguf: &GgufFile, key: &str) -> Result<f32> {
    let value = gguf
        .metadata_f32(key)
        .ok_or_else(|| anyhow!("DeepSeek-V4 config: required f32 key {key:?} missing"))?;
    if !value.is_finite() {
        bail!("DeepSeek-V4 config: key {key:?} must be finite");
    }
    Ok(value)
}

fn required_bool(gguf: &GgufFile, key: &str) -> Result<bool> {
    match gguf.metadata(key) {
        Some(MetadataValue::Bool(value)) => Ok(*value),
        _ => bail!("DeepSeek-V4 config: required bool key {key:?} missing"),
    }
}

fn required_u32_array(gguf: &GgufFile, key: &str) -> Result<Vec<u32>> {
    let Some(MetadataValue::Array(values)) = gguf.metadata(key) else {
        bail!("DeepSeek-V4 config: required u32 array key {key:?} missing");
    };
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value.as_u32().ok_or_else(|| {
                anyhow!("DeepSeek-V4 config: {key:?}[{index}] is not a non-negative u32")
            })
        })
        .collect()
}

fn required_f32_array(gguf: &GgufFile, key: &str) -> Result<Vec<f32>> {
    let Some(MetadataValue::Array(values)) = gguf.metadata(key) else {
        bail!("DeepSeek-V4 config: required f32 array key {key:?} missing");
    };
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let value = value
                .as_f32()
                .ok_or_else(|| anyhow!("DeepSeek-V4 config: {key:?}[{index}] is not f32"))?;
            if !value.is_finite() {
                bail!("DeepSeek-V4 config: {key:?}[{index}] must be finite");
            }
            Ok(value)
        })
        .collect()
}

impl Deepseek4Config {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch = gguf
            .metadata_string("general.architecture")
            .ok_or_else(|| anyhow!("GGUF missing required key 'general.architecture'"))?;
        if arch != ARCH_DEEPSEEK4 {
            bail!("general.architecture={arch:?} is not DeepSeek-V4 (expected {ARCH_DEEPSEEK4:?})");
        }
        let p = ARCH_DEEPSEEK4;
        let rope_scaling_type = gguf
            .metadata_string(&format!("{p}.rope.scaling.type"))
            .ok_or_else(|| anyhow!("DeepSeek-V4 config: required YaRN scaling type missing"))?;
        if rope_scaling_type != "yarn" {
            bail!("DeepSeek-V4 config: rope scaling must be yarn, got {rope_scaling_type:?}");
        }
        let yarn_ext_factor = required_f32(gguf, &format!("{p}.rope.scaling.yarn_ext_factor"))?;
        let yarn_attn_factor = required_f32(gguf, &format!("{p}.rope.scaling.yarn_attn_factor"))?;
        if yarn_ext_factor != -1.0 || yarn_attn_factor != 1.0 {
            bail!(
                "DeepSeek-V4 config: unsupported YaRN ext/attention factors ({yarn_ext_factor}, {yarn_attn_factor})"
            );
        }
        let config = Self {
            hidden_size: required_u32(gguf, &format!("{p}.embedding_length"))?,
            hidden_size_out: required_u32(gguf, &format!("{p}.embedding_length_out"))?,
            num_hidden_layers: required_u32(gguf, &format!("{p}.block_count"))?,
            max_position_embeddings: required_u32(gguf, &format!("{p}.context_length"))?,
            vocab_size: required_u32(gguf, &format!("{p}.vocab_size"))?,
            num_attention_heads: required_u32(gguf, &format!("{p}.attention.head_count"))?,
            num_key_value_heads: required_u32(gguf, &format!("{p}.attention.head_count_kv"))?,
            head_dim: required_u32(gguf, &format!("{p}.attention.key_length"))?,
            rope_head_dim: required_u32(gguf, &format!("{p}.rope.dimension_count"))?,
            rope_theta: required_f32(gguf, &format!("{p}.rope.freq_base"))?,
            rope_factor: required_f32(gguf, &format!("{p}.rope.scaling.factor"))?,
            original_context_length: required_u32(
                gguf,
                &format!("{p}.rope.scaling.original_context_length"),
            )?,
            yarn_beta_fast: required_f32(gguf, &format!("{p}.rope.scaling.yarn_beta_fast"))?,
            yarn_beta_slow: required_f32(gguf, &format!("{p}.rope.scaling.yarn_beta_slow"))?,
            q_lora_rank: required_u32(gguf, &format!("{p}.attention.q_lora_rank"))?,
            o_lora_rank: required_u32(gguf, &format!("{p}.attention.output_lora_rank"))?,
            output_groups: required_u32(gguf, &format!("{p}.attention.output_group_count"))?,
            sliding_window: required_u32(gguf, &format!("{p}.attention.sliding_window"))?,
            compress_ratios: required_u32_array(gguf, &format!("{p}.attention.compress_ratios"))?,
            compress_rope_theta: required_f32(
                gguf,
                &format!("{p}.attention.compress_rope_freq_base"),
            )?,
            index_num_heads: required_u32(gguf, &format!("{p}.attention.indexer.head_count"))?,
            index_head_dim: required_u32(gguf, &format!("{p}.attention.indexer.key_length"))?,
            index_top_k: required_u32(gguf, &format!("{p}.attention.indexer.top_k"))?,
            rms_norm_eps: required_f32(gguf, &format!("{p}.attention.layer_norm_rms_epsilon"))?,
            num_experts: required_u32(gguf, &format!("{p}.expert_count"))?,
            num_experts_per_tok: required_u32(gguf, &format!("{p}.expert_used_count"))?,
            num_shared_experts: required_u32(gguf, &format!("{p}.expert_shared_count"))?,
            expert_intermediate_size: required_u32(
                gguf,
                &format!("{p}.expert_feed_forward_length"),
            )?,
            route_scale: required_f32(gguf, &format!("{p}.expert_weights_scale"))?,
            normalize_topk: required_bool(gguf, &format!("{p}.expert_weights_norm"))?,
            swiglu_clamp_experts: required_f32_array(gguf, &format!("{p}.swiglu_clamp_exp"))?,
            swiglu_clamp_shared: required_f32_array(gguf, &format!("{p}.swiglu_clamp_shexp"))?,
            hyper_connection_count: required_u32(gguf, &format!("{p}.hyper_connection.count"))?,
            hyper_connection_sinkhorn_iterations: required_u32(
                gguf,
                &format!("{p}.hyper_connection.sinkhorn_iterations"),
            )?,
            hyper_connection_epsilon: required_f32(gguf, &format!("{p}.hyper_connection.epsilon"))?,
            hash_layer_count: required_u32(gguf, &format!("{p}.hash_layer_count"))?,
        };
        let value_dim = required_u32(gguf, &format!("{p}.attention.value_length"))?;
        if value_dim != config.head_dim {
            bail!(
                "DeepSeek-V4 config: attention key/value lengths differ ({} vs {value_dim})",
                config.head_dim
            );
        }
        let gating = required_u32(gguf, &format!("{p}.expert_gating_func"))?;
        if gating != 4 {
            bail!("DeepSeek-V4 config: expert_gating_func must be sqrtsoftplus (4), got {gating}");
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 || self.num_hidden_layers == 0 || self.vocab_size == 0 {
            bail!("DeepSeek-V4 config: hidden, layer, and vocabulary sizes must be nonzero");
        }
        if self.hyper_connection_count != 4 {
            bail!(
                "DeepSeek-V4 config: this runtime requires four Hyper-Connection lanes, got {}",
                self.hyper_connection_count
            );
        }
        if self.hidden_size_out != self.hidden_size * self.hyper_connection_count {
            bail!("DeepSeek-V4 config: embedding_length_out must equal embedding_length * hyper_connection.count");
        }
        if self.num_attention_heads == 0
            || self.num_key_value_heads != 1
            || self.head_dim == 0
            || self.rope_head_dim == 0
            || self.rope_head_dim > self.head_dim
            || self.rope_head_dim % 2 != 0
        {
            bail!("DeepSeek-V4 config: invalid MLA head dimensions");
        }
        if self.output_groups == 0 || self.num_attention_heads % self.output_groups != 0 {
            bail!("DeepSeek-V4 config: attention heads must divide evenly into output groups");
        }
        if self.sliding_window == 0
            || self.q_lora_rank == 0
            || self.o_lora_rank == 0
            || self.index_num_heads == 0
            || self.index_head_dim == 0
            || self.index_top_k == 0
        {
            bail!("DeepSeek-V4 config: attention and indexer dimensions must be nonzero");
        }
        if self.compress_ratios.len() != self.num_hidden_layers as usize
            || self
                .compress_ratios
                .iter()
                .any(|ratio| !matches!(ratio, 0 | 4 | 128))
        {
            bail!("DeepSeek-V4 config: compress_ratios must contain one 0/4/128 value per layer");
        }
        if self.num_experts == 0
            || self.num_experts_per_tok == 0
            || self.num_experts_per_tok > self.num_experts
            || self.num_shared_experts != 1
            || self.expert_intermediate_size == 0
        {
            bail!("DeepSeek-V4 config: invalid MoE expert dimensions");
        }
        if self.swiglu_clamp_experts.len() != self.num_hidden_layers as usize
            || self.swiglu_clamp_shared.len() != self.num_hidden_layers as usize
            || self.swiglu_clamp_experts.iter().any(|value| *value <= 0.0)
            || self.swiglu_clamp_shared.iter().any(|value| *value <= 0.0)
        {
            bail!("DeepSeek-V4 config: invalid per-layer SwiGLU clamps");
        }
        if !self.normalize_topk
            || self.route_scale <= 0.0
            || self.rms_norm_eps <= 0.0
            || self.hyper_connection_epsilon <= 0.0
            || self.hyper_connection_sinkhorn_iterations == 0
            || self.hash_layer_count > self.num_hidden_layers
            || self.rope_factor <= 0.0
            || self.original_context_length == 0
            || self.yarn_beta_fast <= 0.0
            || self.yarn_beta_slow <= 0.0
        {
            bail!("DeepSeek-V4 config: invalid normalization, routing, Hyper-Connection, or YaRN values");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::gguf::types::MetaValue;
    use crate::backends::gguf::writer::GgufWriter;
    use std::fs;

    fn official_metadata() -> Vec<(String, MetaValue)> {
        let mut kv = vec![
            (
                "general.architecture",
                MetaValue::String("deepseek4".into()),
            ),
            ("deepseek4.block_count", MetaValue::U32(43)),
            ("deepseek4.context_length", MetaValue::U32(1_048_576)),
            ("deepseek4.embedding_length", MetaValue::U32(4096)),
            ("deepseek4.embedding_length_out", MetaValue::U32(16384)),
            ("deepseek4.vocab_size", MetaValue::U32(129280)),
            ("deepseek4.attention.head_count", MetaValue::U32(64)),
            ("deepseek4.attention.head_count_kv", MetaValue::U32(1)),
            ("deepseek4.attention.key_length", MetaValue::U32(512)),
            ("deepseek4.attention.value_length", MetaValue::U32(512)),
            ("deepseek4.rope.dimension_count", MetaValue::U32(64)),
            ("deepseek4.rope.freq_base", MetaValue::F32(10000.0)),
            ("deepseek4.rope.scaling.factor", MetaValue::F32(16.0)),
            (
                "deepseek4.rope.scaling.type",
                MetaValue::String("yarn".into()),
            ),
            (
                "deepseek4.rope.scaling.yarn_ext_factor",
                MetaValue::F32(-1.0),
            ),
            (
                "deepseek4.rope.scaling.yarn_attn_factor",
                MetaValue::F32(1.0),
            ),
            (
                "deepseek4.rope.scaling.original_context_length",
                MetaValue::U32(65536),
            ),
            (
                "deepseek4.rope.scaling.yarn_beta_fast",
                MetaValue::F32(32.0),
            ),
            ("deepseek4.rope.scaling.yarn_beta_slow", MetaValue::F32(1.0)),
            ("deepseek4.attention.q_lora_rank", MetaValue::U32(1024)),
            ("deepseek4.attention.output_lora_rank", MetaValue::U32(1024)),
            ("deepseek4.attention.output_group_count", MetaValue::U32(8)),
            ("deepseek4.attention.sliding_window", MetaValue::U32(128)),
            (
                "deepseek4.attention.compress_rope_freq_base",
                MetaValue::F32(160000.0),
            ),
            ("deepseek4.attention.indexer.head_count", MetaValue::U32(64)),
            (
                "deepseek4.attention.indexer.key_length",
                MetaValue::U32(128),
            ),
            ("deepseek4.attention.indexer.top_k", MetaValue::U32(512)),
            (
                "deepseek4.attention.layer_norm_rms_epsilon",
                MetaValue::F32(1e-6),
            ),
            ("deepseek4.expert_count", MetaValue::U32(256)),
            ("deepseek4.expert_used_count", MetaValue::U32(6)),
            ("deepseek4.expert_shared_count", MetaValue::U32(1)),
            ("deepseek4.expert_feed_forward_length", MetaValue::U32(2048)),
            ("deepseek4.expert_weights_scale", MetaValue::F32(1.5)),
            ("deepseek4.expert_weights_norm", MetaValue::Bool(true)),
            ("deepseek4.expert_gating_func", MetaValue::U32(4)),
            ("deepseek4.hyper_connection.count", MetaValue::U32(4)),
            (
                "deepseek4.hyper_connection.sinkhorn_iterations",
                MetaValue::U32(20),
            ),
            ("deepseek4.hyper_connection.epsilon", MetaValue::F32(1e-6)),
            ("deepseek4.hash_layer_count", MetaValue::U32(3)),
        ]
        .into_iter()
        .map(|(key, value)| (key.into(), value))
        .collect::<Vec<_>>();
        kv.push((
            "deepseek4.attention.compress_ratios".into(),
            MetaValue::ArrayU32(
                (0..43)
                    .map(|layer| {
                        if layer < 2 {
                            0
                        } else if layer % 2 == 0 {
                            4
                        } else {
                            128
                        }
                    })
                    .collect(),
            ),
        ));
        kv.push((
            "deepseek4.swiglu_clamp_exp".into(),
            MetaValue::ArrayF32(vec![10.0; 43]),
        ));
        kv.push((
            "deepseek4.swiglu_clamp_shexp".into(),
            MetaValue::ArrayF32(vec![10.0; 43]),
        ));
        kv
    }

    fn open_fixture(kv: &[(String, MetaValue)]) -> (tempfile::TempDir, GgufFile) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.gguf");
        let file = fs::File::create(&path).unwrap();
        let mut writer = GgufWriter::new(file);
        writer.write_header(0, kv.len() as u64).unwrap();
        for (key, value) in kv {
            writer.write_metadata_kv(key, value).unwrap();
        }
        writer.pad_to_alignment().unwrap();
        writer.finalize().unwrap();
        (dir, GgufFile::open(&path).unwrap())
    }

    #[test]
    fn parses_official_0731_contract() {
        let (_dir, gguf) = open_fixture(&official_metadata());
        let config = Deepseek4Config::from_gguf(&gguf).unwrap();
        assert_eq!(config.num_hidden_layers, 43);
        assert_eq!(config.head_dim, 512);
        assert_eq!(config.compress_ratios[2], 4);
        assert_eq!(config.num_experts_per_tok, 6);
        assert_eq!(config.hyper_connection_sinkhorn_iterations, 20);
    }

    #[test]
    fn rejects_missing_metadata_and_invalid_compression_schedule() {
        let mut kv = official_metadata();
        kv.retain(|(key, _)| key != "deepseek4.attention.key_length");
        let (_dir, gguf) = open_fixture(&kv);
        assert!(Deepseek4Config::from_gguf(&gguf)
            .unwrap_err()
            .to_string()
            .contains("attention.key_length"));

        let mut kv = official_metadata();
        let ratio = kv
            .iter_mut()
            .find(|(key, _)| key == "deepseek4.attention.compress_ratios")
            .unwrap();
        ratio.1 = MetaValue::ArrayU32(vec![4; 42]);
        let (_dir, gguf) = open_fixture(&kv);
        assert!(Deepseek4Config::from_gguf(&gguf)
            .unwrap_err()
            .to_string()
            .contains("compress_ratios"));
    }
}
