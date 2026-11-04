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

fn required_vocab_size(gguf: &GgufFile, key: &str) -> Result<u32> {
    if let Some(value) = gguf.metadata_u32(key) {
        return Ok(value);
    }
    let vocab = gguf
        .tensor_info("token_embd.weight")
        .and_then(|tensor| tensor.shape.first().copied())
        .ok_or_else(|| {
            anyhow!(
                "DeepSeek-V4 config: {key:?} is absent and token_embd.weight has no vocabulary dimension"
            )
        })?;
    u32::try_from(vocab)
        .map_err(|_| anyhow!("DeepSeek-V4 config: token embedding vocabulary exceeds u32"))
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

fn optional_f32(gguf: &GgufFile, key: &str, default: f32) -> Result<f32> {
    let Some(value) = gguf.metadata(key) else {
        return Ok(default);
    };
    let value = value
        .as_f32()
        .ok_or_else(|| anyhow!("DeepSeek-V4 config: optional f32 key {key:?} has wrong type"))?;
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

/// `swiglu_clamp_shexp` falls back to `swiglu_clamp_exp` when the key is
/// absent, as the peer does (ds4's 0731 file omits it; ADR-063
/// Q4-DS4-RACE L3). A present key is parsed strictly.
fn shared_swiglu_clamps(gguf: &GgufFile, key: &str, experts: &[f32]) -> Result<Vec<f32>> {
    if gguf.metadata(key).is_none() {
        return Ok(experts.to_vec());
    }
    required_f32_array(gguf, key)
}

/// `embedding_length_out` defaults to `embedding_length ×
/// hyper_connection.count` when absent (L3); `validate` keeps the equality
/// check for a present key.
fn hidden_size_out(gguf: &GgufFile, key: &str, hidden: u32, lanes: u32) -> Result<u32> {
    if gguf.metadata(key).is_none() {
        return hidden.checked_mul(lanes).ok_or_else(|| {
            anyhow!("DeepSeek-V4 config: embedding_length * hyper_connection.count overflows u32")
        });
    }
    required_u32(gguf, key)
}

/// Accept a `compress_ratios` array longer than `block_count` (ds4's file
/// carries a trailing entry for its unshipped next-token-prediction layer,
/// L4) only when every entry past `block_count` is 0, and keep the first
/// `block_count` entries. Shorter arrays reach `validate`'s length check.
fn layer_compress_ratios(mut ratios: Vec<u32>, block_count: u32) -> Result<Vec<u32>> {
    let layers = block_count as usize;
    if ratios.len() > layers {
        if let Some(index) = ratios[layers..].iter().position(|ratio| *ratio != 0) {
            bail!(
                "DeepSeek-V4 config: compress_ratios[{}] past block_count {block_count} must be 0, got {}",
                layers + index,
                ratios[layers + index]
            );
        }
        ratios.truncate(layers);
    }
    Ok(ratios)
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
        // The official 0731 config and canonical converter omit both optional
        // YaRN knobs. The reference model's custom frequency interpolation
        // therefore uses no attention multiplier; preserve those defaults
        // while still rejecting explicit incompatible values.
        let yarn_ext_factor =
            optional_f32(gguf, &format!("{p}.rope.scaling.yarn_ext_factor"), -1.0)?;
        let yarn_attn_factor =
            optional_f32(gguf, &format!("{p}.rope.scaling.yarn_attn_factor"), 1.0)?;
        if yarn_ext_factor != -1.0 || yarn_attn_factor != 1.0 {
            bail!(
                "DeepSeek-V4 config: unsupported YaRN ext/attention factors ({yarn_ext_factor}, {yarn_attn_factor})"
            );
        }
        let hidden_size = required_u32(gguf, &format!("{p}.embedding_length"))?;
        let hyper_connection_count = required_u32(gguf, &format!("{p}.hyper_connection.count"))?;
        let num_hidden_layers = required_u32(gguf, &format!("{p}.block_count"))?;
        let swiglu_clamp_experts = required_f32_array(gguf, &format!("{p}.swiglu_clamp_exp"))?;
        let config = Self {
            hidden_size,
            hidden_size_out: hidden_size_out(
                gguf,
                &format!("{p}.embedding_length_out"),
                hidden_size,
                hyper_connection_count,
            )?,
            num_hidden_layers,
            max_position_embeddings: required_u32(gguf, &format!("{p}.context_length"))?,
            // Canonical DeepSeek-V4 GGUFs derive vocabulary size from the
            // embedding tensor instead of emitting an architecture KV.
            vocab_size: required_vocab_size(gguf, &format!("{p}.vocab_size"))?,
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
            compress_ratios: layer_compress_ratios(
                required_u32_array(gguf, &format!("{p}.attention.compress_ratios"))?,
                num_hidden_layers,
            )?,
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
            swiglu_clamp_shared: shared_swiglu_clamps(
                gguf,
                &format!("{p}.swiglu_clamp_shexp"),
                &swiglu_clamp_experts,
            )?,
            swiglu_clamp_experts,
            hyper_connection_count,
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
        if self.hyper_connection_sinkhorn_iterations != 20
            || self.hyper_connection_epsilon.to_bits() != 1e-6f32.to_bits()
        {
            bail!(
                "DeepSeek-V4 config: native Hyper-Connection kernel requires exactly 20 Sinkhorn iterations and epsilon 1e-6, got {} and {}",
                self.hyper_connection_sinkhorn_iterations,
                self.hyper_connection_epsilon
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
pub(super) mod tests {
    use super::*;
    use crate::backends::gguf::types::MetaValue;
    use crate::backends::gguf::writer::GgufWriter;
    use std::fs;

    pub(crate) fn official_metadata() -> Vec<(String, MetaValue)> {
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
    fn accepts_officially_omitted_optional_yarn_factors() {
        let mut kv = official_metadata();
        kv.retain(|(key, _)| {
            key != "deepseek4.rope.scaling.yarn_ext_factor"
                && key != "deepseek4.rope.scaling.yarn_attn_factor"
        });
        let (_dir, gguf) = open_fixture(&kv);
        Deepseek4Config::from_gguf(&gguf).expect("official omitted YaRN defaults");
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

    #[test]
    fn rejects_hyper_connection_constants_that_drift_from_native_kernel() {
        for (key, value) in [
            (
                "deepseek4.hyper_connection.sinkhorn_iterations",
                MetaValue::U32(19),
            ),
            ("deepseek4.hyper_connection.epsilon", MetaValue::F32(2e-6)),
        ] {
            let mut kv = official_metadata();
            kv.iter_mut()
                .find(|(candidate, _)| candidate == key)
                .expect("official HC key")
                .1 = value;
            let (_dir, gguf) = open_fixture(&kv);
            let error = Deepseek4Config::from_gguf(&gguf)
                .expect_err("drifted fixed HC parameter must fail closed")
                .to_string();
            assert!(error.contains("requires exactly 20 Sinkhorn iterations and epsilon 1e-6"));
        }
    }

    fn set_ratios(kv: &mut [(String, MetaValue)], ratios: Vec<u32>) {
        kv.iter_mut()
            .find(|(key, _)| key == "deepseek4.attention.compress_ratios")
            .unwrap()
            .1 = MetaValue::ArrayU32(ratios);
    }

    fn official_ratios() -> Vec<u32> {
        let kv = official_metadata();
        let MetaValue::ArrayU32(ratios) = &kv
            .iter()
            .find(|(key, _)| key == "deepseek4.attention.compress_ratios")
            .unwrap()
            .1
        else {
            unreachable!("fixture ratios are u32")
        };
        ratios.clone()
    }

    /// ds4's 0731 file: no `embedding_length_out`, no `swiglu_clamp_shexp`
    /// and a 44-entry `compress_ratios` with a trailing 0 (Q4-DS4-RACE L3/L4).
    fn ds4_file_metadata() -> Vec<(String, MetaValue)> {
        let mut kv = official_metadata();
        kv.retain(|(key, _)| {
            key != "deepseek4.embedding_length_out" && key != "deepseek4.swiglu_clamp_shexp"
        });
        let mut ratios = official_ratios();
        ratios.push(0);
        set_ratios(&mut kv, ratios);
        kv
    }

    #[test]
    fn ds4_file_config_defaults_equal_the_hf2q_artifact_config() {
        let (_dir, hf2q) = open_fixture(&official_metadata());
        let (_dir2, ds4) = open_fixture(&ds4_file_metadata());
        let hf2q = Deepseek4Config::from_gguf(&hf2q).unwrap();
        let ds4 = Deepseek4Config::from_gguf(&ds4).expect("ds4 file metadata parses");
        assert_eq!(ds4.hidden_size_out, 16384);
        assert_eq!(ds4.compress_ratios.len(), 43);
        assert_eq!(ds4.swiglu_clamp_shared, ds4.swiglu_clamp_experts);
        assert_eq!(
            ds4, hf2q,
            "defaults must reproduce the 0731 artifact config"
        );
    }

    #[test]
    fn present_config_keys_keep_their_strict_checks() {
        let mut kv = official_metadata();
        kv.iter_mut()
            .find(|(key, _)| key == "deepseek4.embedding_length_out")
            .unwrap()
            .1 = MetaValue::U32(8192);
        let (_dir, gguf) = open_fixture(&kv);
        assert!(Deepseek4Config::from_gguf(&gguf)
            .unwrap_err()
            .to_string()
            .contains("embedding_length_out"));

        let mut kv = official_metadata();
        kv.iter_mut()
            .find(|(key, _)| key == "deepseek4.swiglu_clamp_shexp")
            .unwrap()
            .1 = MetaValue::ArrayF32(vec![10.0; 42]);
        let (_dir, gguf) = open_fixture(&kv);
        assert!(Deepseek4Config::from_gguf(&gguf)
            .unwrap_err()
            .to_string()
            .contains("SwiGLU clamps"));
    }

    #[test]
    fn compress_ratios_tail_must_be_zero_and_short_arrays_fail() {
        let mut kv = ds4_file_metadata();
        let mut ratios = official_ratios();
        ratios.push(4);
        set_ratios(&mut kv, ratios);
        let (_dir, gguf) = open_fixture(&kv);
        let error = Deepseek4Config::from_gguf(&gguf).unwrap_err().to_string();
        assert!(error.contains("compress_ratios[43]"), "{error}");

        let mut kv = ds4_file_metadata();
        let mut ratios = official_ratios();
        ratios.pop();
        set_ratios(&mut kv, ratios);
        let (_dir, gguf) = open_fixture(&kv);
        assert!(Deepseek4Config::from_gguf(&gguf)
            .unwrap_err()
            .to_string()
            .contains("compress_ratios must contain one 0/4/128 value per layer"));
    }
}
