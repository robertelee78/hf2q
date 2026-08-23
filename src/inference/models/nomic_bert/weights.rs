//! Native GGUF storage for nomic-bert embedding inference.
//!
//! Fused QKV and every other matrix retain their exact file-backed GGUF
//! representation. Only declared normalization and bias vectors expand to
//! F32. Header-only validation completes before mapping or model allocation.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use mlx_native::gguf::GgufFile;
use mlx_native::metal::foreign_types::ForeignType;
use mlx_native::{MlxBuffer, MlxDevice};

use crate::inference::models::bert::native_storage::{
    mapped_qweight, preflight_embedding, preflight_linear, preflight_state, MatrixPlan,
    NativeStorageStats, StatePlan,
};
use crate::serve::forward_mlx_shared::MlxQWeight;

use super::config::{
    nomic_bert_layer_tensor, NomicBertConfig, NOMIC_BERT_TENSOR_EMBED_NORM_BIAS,
    NOMIC_BERT_TENSOR_EMBED_NORM_WEIGHT, NOMIC_BERT_TENSOR_TOKEN_EMBD,
    NOMIC_BERT_TENSOR_TOKEN_TYPES,
};

pub const NOMIC_BERT_BLOCK_REQUIRED_SUFFIXES: &[&str] = &[
    "attn_qkv.weight",
    "attn_output.weight",
    "attn_output_norm.weight",
    "attn_output_norm.bias",
    "ffn_up.weight",
    "ffn_gate.weight",
    "ffn_down.weight",
    "layer_output_norm.weight",
    "layer_output_norm.bias",
];

pub const NOMIC_BERT_BLOCK_OPTIONAL_SUFFIXES: &[&str] = &[
    "attn_qkv.bias",
    "attn_output.bias",
    "ffn_up.bias",
    "ffn_gate.bias",
    "ffn_down.bias",
];

pub fn validate_tensor_set(gguf: &GgufFile, cfg: &NomicBertConfig) -> Result<()> {
    let names: std::collections::HashSet<&str> = gguf.tensor_names().into_iter().collect();
    let mut missing = Vec::new();
    for name in [
        NOMIC_BERT_TENSOR_TOKEN_EMBD,
        NOMIC_BERT_TENSOR_EMBED_NORM_WEIGHT,
        NOMIC_BERT_TENSOR_EMBED_NORM_BIAS,
    ] {
        if !names.contains(name) {
            missing.push(name.to_string());
        }
    }
    for layer in 0..cfg.num_hidden_layers {
        for suffix in NOMIC_BERT_BLOCK_REQUIRED_SUFFIXES {
            let name = nomic_bert_layer_tensor(layer, suffix);
            if !names.contains(name.as_str()) {
                missing.push(name);
            }
        }
    }
    if !missing.is_empty() {
        missing.sort();
        return Err(anyhow!(
            "nomic-bert GGUF missing {} tensor(s): {}",
            missing.len(),
            missing.join(", ")
        ));
    }
    Ok(())
}

pub struct LoadedNomicBertWeights {
    matrices: HashMap<String, MlxQWeight>,
    states: HashMap<String, MlxBuffer>,
    stats: NativeStorageStats,
    _device: MlxDevice,
}

impl std::fmt::Debug for LoadedNomicBertWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedNomicBertWeights")
            .field("matrix_count", &self.matrices.len())
            .field("state_count", &self.states.len())
            .field("storage", &self.stats)
            .finish()
    }
}

impl LoadedNomicBertWeights {
    pub fn load(gguf: &GgufFile, cfg: &NomicBertConfig, device: MlxDevice) -> Result<Self> {
        validate_tensor_set(gguf, cfg)?;
        let (matrix_plans, state_plans) = preflight_plans(gguf, cfg)?;

        let mapped = gguf
            .map_tensor_data(&device)
            .map_err(|e| anyhow!("map nomic-bert GGUF tensor payload: {e}"))?;
        let mut matrices = HashMap::with_capacity(matrix_plans.len());
        let mut mapped_resources = std::collections::HashSet::new();
        let mut file_backed_bytes = 0u64;
        for plan in &matrix_plans {
            let info = gguf
                .tensor_info(&plan.name)
                .ok_or_else(|| anyhow!("nomic-bert native tensor '{}' disappeared", plan.name))?;
            let weight = mapped_qweight(plan, &mapped, info)
                .map_err(|e| anyhow!("map nomic-bert native tensor '{}': {e}", plan.name))?;
            mapped_resources.insert(weight.buffer.metal_buffer().as_ptr() as usize);
            file_backed_bytes = file_backed_bytes
                .checked_add(plan.byte_len as u64)
                .ok_or_else(|| anyhow!("nomic-bert file-backed byte accounting overflow"))?;
            matrices.insert(plan.name.clone(), weight);
        }

        let mut states = HashMap::with_capacity(state_plans.len());
        let mut anonymous_state_bytes = 0u64;
        for plan in &state_plans {
            let buffer = gguf
                .load_tensor_f32(&plan.name, &device)
                .map_err(|e| anyhow!("load nomic-bert F32 state '{}': {e}", plan.name))?;
            anonymous_state_bytes = anonymous_state_bytes
                .checked_add((plan.elements as u64) * 4)
                .ok_or_else(|| anyhow!("nomic-bert state byte accounting overflow"))?;
            states.insert(plan.name.clone(), buffer);
        }
        let resident_bytes = file_backed_bytes
            .checked_add(anonymous_state_bytes)
            .ok_or_else(|| anyhow!("nomic-bert resident byte accounting overflow"))?;
        Ok(Self {
            matrices,
            states,
            stats: NativeStorageStats {
                resident_bytes,
                file_backed_bytes,
                anonymous_state_bytes,
                mapped_segment_count: mapped_resources.len(),
            },
            _device: device,
        })
    }

    pub fn load_from_path(path: &Path, cfg: &NomicBertConfig) -> Result<Self> {
        let gguf = GgufFile::open(path)
            .map_err(|e| anyhow!("open nomic-bert GGUF {}: {e}", path.display()))?;
        validate_tensor_set(&gguf, cfg)?;
        let device =
            MlxDevice::new().map_err(|e| anyhow!("create MlxDevice for nomic-bert: {e}"))?;
        Self::load(&gguf, cfg, device)
    }

    pub fn empty(device: MlxDevice) -> Self {
        Self {
            matrices: HashMap::new(),
            states: HashMap::new(),
            stats: NativeStorageStats::default(),
            _device: device,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_tensors_for_test(
        tensors: HashMap<String, MlxBuffer>,
        cfg: &NomicBertConfig,
        device: MlxDevice,
    ) -> Self {
        let mut matrices = HashMap::new();
        let mut states = HashMap::new();
        for (name, buffer) in tensors {
            let matrix_shape = test_matrix_shape(&name, cfg);
            if let Some((rows, cols)) = matrix_shape {
                matrices.insert(
                    name,
                    crate::inference::models::bert::native_storage::f32_qweight_for_test(
                        buffer, rows, cols,
                    ),
                );
            } else {
                states.insert(name, buffer);
            }
        }
        Self {
            matrices,
            states,
            stats: NativeStorageStats::default(),
            _device: device,
        }
    }

    pub fn len(&self) -> usize {
        self.matrices.len() + self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.matrices.is_empty() && self.states.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&MlxBuffer> {
        self.states
            .get(name)
            .or_else(|| self.matrices.get(name).map(|weight| &weight.buffer))
    }

    pub fn storage_stats(&self) -> NativeStorageStats {
        self.stats
    }

    pub fn token_embd_weight(&self) -> Result<&MlxQWeight> {
        self.matrices
            .get(NOMIC_BERT_TENSOR_TOKEN_EMBD)
            .ok_or_else(|| anyhow!("nomic-bert missing '{}'", NOMIC_BERT_TENSOR_TOKEN_EMBD))
    }

    pub fn token_types_weight(&self) -> Option<&MlxQWeight> {
        self.matrices.get(NOMIC_BERT_TENSOR_TOKEN_TYPES)
    }

    pub fn embed_norm_weight(&self) -> Result<&MlxBuffer> {
        self.states
            .get(NOMIC_BERT_TENSOR_EMBED_NORM_WEIGHT)
            .ok_or_else(|| {
                anyhow!(
                    "nomic-bert missing '{}'",
                    NOMIC_BERT_TENSOR_EMBED_NORM_WEIGHT
                )
            })
    }

    pub fn embed_norm_bias(&self) -> Result<&MlxBuffer> {
        self.states
            .get(NOMIC_BERT_TENSOR_EMBED_NORM_BIAS)
            .ok_or_else(|| anyhow!("nomic-bert missing '{}'", NOMIC_BERT_TENSOR_EMBED_NORM_BIAS))
    }

    pub fn block_required(&self, layer: usize, suffix: &str) -> Result<&MlxBuffer> {
        let name = nomic_bert_layer_tensor(layer, suffix);
        self.states
            .get(&name)
            .ok_or_else(|| anyhow!("nomic-bert missing '{}'", name))
    }

    pub fn block_optional(&self, layer: usize, suffix: &str) -> Option<&MlxBuffer> {
        self.states.get(&nomic_bert_layer_tensor(layer, suffix))
    }

    pub fn block_matrix(&self, layer: usize, suffix: &str) -> Result<&MlxQWeight> {
        let name = nomic_bert_layer_tensor(layer, suffix);
        self.matrices
            .get(&name)
            .ok_or_else(|| anyhow!("nomic-bert missing native matrix '{}'", name))
    }
}

#[cfg(test)]
fn test_matrix_shape(name: &str, cfg: &NomicBertConfig) -> Option<(usize, usize)> {
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    if name == NOMIC_BERT_TENSOR_TOKEN_EMBD {
        return Some((cfg.vocab_size, h));
    }
    if name == NOMIC_BERT_TENSOR_TOKEN_TYPES {
        return Some((cfg.type_vocab_size, h));
    }
    for (suffix, shape) in [
        ("attn_qkv.weight", (3 * h, h)),
        ("attn_output.weight", (h, h)),
        ("ffn_up.weight", (i, h)),
        ("ffn_gate.weight", (i, h)),
        ("ffn_down.weight", (h, i)),
    ] {
        if name.starts_with("blk.") && name.ends_with(suffix) {
            return Some(shape);
        }
    }
    None
}

fn preflight_plans(
    gguf: &GgufFile,
    cfg: &NomicBertConfig,
) -> Result<(Vec<MatrixPlan>, Vec<StatePlan>)> {
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    let mut matrices = vec![preflight_embedding(
        gguf,
        NOMIC_BERT_TENSOR_TOKEN_EMBD,
        cfg.vocab_size,
        h,
        cfg.max_position_embeddings,
    )?];
    if gguf.tensor_info(NOMIC_BERT_TENSOR_TOKEN_TYPES).is_some() {
        matrices.push(preflight_embedding(
            gguf,
            NOMIC_BERT_TENSOR_TOKEN_TYPES,
            cfg.type_vocab_size,
            h,
            cfg.max_position_embeddings,
        )?);
    }
    let mut states = vec![
        preflight_state(gguf, NOMIC_BERT_TENSOR_EMBED_NORM_WEIGHT, h)?,
        preflight_state(gguf, NOMIC_BERT_TENSOR_EMBED_NORM_BIAS, h)?,
    ];
    for layer in 0..cfg.num_hidden_layers {
        matrices.push(preflight_linear(
            gguf,
            nomic_bert_layer_tensor(layer, "attn_qkv.weight"),
            3 * h,
            h,
            cfg.max_position_embeddings,
        )?);
        matrices.push(preflight_linear(
            gguf,
            nomic_bert_layer_tensor(layer, "attn_output.weight"),
            h,
            h,
            cfg.max_position_embeddings,
        )?);
        for suffix in ["ffn_up.weight", "ffn_gate.weight"] {
            matrices.push(preflight_linear(
                gguf,
                nomic_bert_layer_tensor(layer, suffix),
                i,
                h,
                cfg.max_position_embeddings,
            )?);
        }
        matrices.push(preflight_linear(
            gguf,
            nomic_bert_layer_tensor(layer, "ffn_down.weight"),
            h,
            i,
            cfg.max_position_embeddings,
        )?);
        for suffix in [
            "attn_output_norm.weight",
            "attn_output_norm.bias",
            "layer_output_norm.weight",
            "layer_output_norm.bias",
        ] {
            states.push(preflight_state(
                gguf,
                nomic_bert_layer_tensor(layer, suffix),
                h,
            )?);
        }
        for (suffix, elements) in [
            ("attn_qkv.bias", 3 * h),
            ("attn_output.bias", h),
            ("ffn_up.bias", i),
            ("ffn_gate.bias", i),
            ("ffn_down.bias", h),
        ] {
            let name = nomic_bert_layer_tensor(layer, suffix);
            if gguf.tensor_info(&name).is_some() {
                states.push(preflight_state(gguf, name, elements)?);
            }
        }
    }
    Ok((matrices, states))
}

#[cfg(test)]
mod tests {
    use super::super::super::bert::config::PoolingType;
    use super::*;
    use crate::inference::models::bert::native_storage::test_support::{
        bert_tensors, write_fixture,
    };

    fn synthetic_cfg(layers: usize) -> NomicBertConfig {
        NomicBertConfig {
            hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: layers,
            intermediate_size: 3072,
            max_position_embeddings: 2048,
            vocab_size: 30522,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            pooling_type: PoolingType::Mean,
            rope_freq_base: 1000.0,
            causal_attention: false,
        }
    }

    #[test]
    fn manifest_preserves_fused_qkv_and_optional_biases() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        assert!(NOMIC_BERT_BLOCK_REQUIRED_SUFFIXES.contains(&"attn_qkv.weight"));
        assert!(!NOMIC_BERT_BLOCK_REQUIRED_SUFFIXES.contains(&"attn_q.weight"));
        assert!(NOMIC_BERT_BLOCK_OPTIONAL_SUFFIXES
            .iter()
            .all(|suffix| suffix.ends_with(".bias")));
    }

    #[test]
    fn empty_loaded_weights_fails_closed() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let weights = LoadedNomicBertWeights::empty(MlxDevice::new().expect("device"));
        assert_eq!(weights.len(), 0);
        assert!(weights.is_empty());
        assert!(weights.token_embd_weight().is_err());
        assert!(weights.block_matrix(0, "attn_qkv.weight").is_err());
        assert_eq!(weights.storage_stats(), NativeStorageStats::default());
    }

    #[test]
    fn required_name_count_tracks_layers() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let cfg = synthetic_cfg(2);
        assert_eq!(
            3 + cfg.num_hidden_layers * NOMIC_BERT_BLOCK_REQUIRED_SUFFIXES.len(),
            21
        );
    }

    #[test]
    fn bert_manifest_fails_closed_before_nomic_payload_mapping() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("bert-named-as-nomic.gguf");
        write_fixture(&path, &bert_tensors(4, 0x88, 2), false);
        let gguf = GgufFile::open(&path).expect("header-only BERT fixture");
        let error = validate_tensor_set(&gguf, &synthetic_cfg(1))
            .expect_err("a separate-QKV BERT manifest must not enter NomicBert");
        let message = error.to_string();
        assert!(message.contains("attn_qkv.weight"), "{message}");
        assert!(
            !message.contains("payload") && !message.contains("map"),
            "{message}"
        );
    }
}
