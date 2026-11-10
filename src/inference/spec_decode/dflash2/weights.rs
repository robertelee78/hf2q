//! Strict DFlash2 safetensors schema validation (ADR-067 D0).

use super::config::{DFlash2Config, DFLASH2_CONVOLUTION_STAGES};
use super::schema::{DFlash2CheckpointDescriptor, DFlash2SchemaError};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use std::collections::HashSet;
use std::path::Path;

pub const DFLASH2_WEIGHT_DTYPE: Dtype = Dtype::BF16;
pub const OPTIONAL_MASK_EMBEDDING: &str = "mask_embedding";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DFlash2ExpectedTensor {
    pub name: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, thiserror::Error)]
pub enum DFlash2WeightsError {
    #[error("DFlash2 weights IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("DFlash2 safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("DFlash2 tensor missing: {0}")]
    Missing(String),
    #[error("DFlash2 tensor not in the checkpoint schema: {0}")]
    Extra(String),
    #[error("DFlash2 tensor {name} has dtype {actual:?}, expected {expected:?}")]
    Dtype {
        name: String,
        actual: Dtype,
        expected: Dtype,
    },
    #[error("DFlash2 tensor {name} has shape {actual:?}, expected {expected:?}")]
    Shape {
        name: String,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },
    #[error(transparent)]
    Schema(#[from] DFlash2SchemaError),
    #[error("DFlash2 checkpoint has {actual} tensors, expected {expected}")]
    TensorCount { actual: usize, expected: usize },
}

pub fn expected_manifest(cfg: &DFlash2Config) -> Vec<DFlash2ExpectedTensor> {
    let hidden = cfg.hidden_size;
    let intermediate = cfg.intermediate_size;
    let query = cfg.num_attention_heads * cfg.head_dim;
    let kv = cfg.num_key_value_heads * cfg.head_dim;
    let dynamic = cfg.dynamic_projection_width();
    let mut manifest = Vec::with_capacity(6 + cfg.num_hidden_layers * 15);

    manifest.push(DFlash2ExpectedTensor {
        name: "candidate_selector.hidden_projection.weight".into(),
        shape: vec![cfg.selector_rank, hidden],
    });
    manifest.push(DFlash2ExpectedTensor {
        name: "candidate_selector.predecessor_codebook".into(),
        shape: vec![cfg.vocab_size, cfg.selector_rank],
    });
    manifest.push(DFlash2ExpectedTensor {
        name: "candidate_selector.successor_codebook".into(),
        shape: vec![cfg.vocab_size, cfg.selector_rank],
    });
    manifest.push(DFlash2ExpectedTensor {
        name: "fc.weight".into(),
        shape: vec![hidden, cfg.fc_input_dim()],
    });
    manifest.push(DFlash2ExpectedTensor {
        name: "hidden_norm.weight".into(),
        shape: vec![hidden],
    });

    for layer in 0..cfg.num_hidden_layers {
        let prefix = format!("layers.{layer}");
        for stem in ["attention_conv", "mlp_conv"] {
            manifest.push(DFlash2ExpectedTensor {
                name: format!("{prefix}.{stem}.base_kernel"),
                shape: vec![DFLASH2_CONVOLUTION_STAGES, cfg.conv_kernel_size, hidden],
            });
            manifest.push(DFlash2ExpectedTensor {
                name: format!("{prefix}.{stem}.kernel_projection.weight"),
                shape: vec![dynamic, hidden],
            });
        }
        manifest.push(DFlash2ExpectedTensor {
            name: format!("{prefix}.input_layernorm.weight"),
            shape: vec![hidden],
        });
        manifest.push(DFlash2ExpectedTensor {
            name: format!("{prefix}.mlp.down_proj.weight"),
            shape: vec![hidden, intermediate],
        });
        manifest.push(DFlash2ExpectedTensor {
            name: format!("{prefix}.mlp.gate_proj.weight"),
            shape: vec![intermediate, hidden],
        });
        manifest.push(DFlash2ExpectedTensor {
            name: format!("{prefix}.mlp.up_proj.weight"),
            shape: vec![intermediate, hidden],
        });
        manifest.push(DFlash2ExpectedTensor {
            name: format!("{prefix}.post_attention_layernorm.weight"),
            shape: vec![hidden],
        });
        manifest.push(DFlash2ExpectedTensor {
            name: format!("{prefix}.self_attn.k_norm.weight"),
            shape: vec![cfg.head_dim],
        });
        manifest.push(DFlash2ExpectedTensor {
            name: format!("{prefix}.self_attn.k_proj.weight"),
            shape: vec![kv, hidden],
        });
        manifest.push(DFlash2ExpectedTensor {
            name: format!("{prefix}.self_attn.o_proj.weight"),
            shape: vec![hidden, query],
        });
        manifest.push(DFlash2ExpectedTensor {
            name: format!("{prefix}.self_attn.q_norm.weight"),
            shape: vec![cfg.head_dim],
        });
        manifest.push(DFlash2ExpectedTensor {
            name: format!("{prefix}.self_attn.q_proj.weight"),
            shape: vec![query, hidden],
        });
        manifest.push(DFlash2ExpectedTensor {
            name: format!("{prefix}.self_attn.v_proj.weight"),
            shape: vec![kv, hidden],
        });
    }

    manifest.push(DFlash2ExpectedTensor {
        name: "norm.weight".into(),
        shape: vec![hidden],
    });
    manifest
}

/// Read-only mapped checkpoint owner. Views returned by [`Self::load`] cannot
/// outlive the mapping.
pub struct DFlash2WeightsFile {
    mmap: memmap2::Mmap,
}

impl DFlash2WeightsFile {
    pub fn open_verified(
        path: impl AsRef<Path>,
        descriptor: &DFlash2CheckpointDescriptor,
    ) -> Result<Self, DFlash2WeightsError> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        descriptor.verify_weights_bytes(&mmap)?;
        Ok(Self { mmap })
    }

    pub fn load_verified<'a>(
        &'a self,
        cfg: &DFlash2Config,
        descriptor: &DFlash2CheckpointDescriptor,
    ) -> Result<DFlash2Weights<'a>, DFlash2WeightsError> {
        let weights = DFlash2Weights::load(&self.mmap, cfg)?;
        if weights.manifest.len() != descriptor.tensor_count {
            return Err(DFlash2WeightsError::TensorCount {
                actual: weights.manifest.len(),
                expected: descriptor.tensor_count,
            });
        }
        Ok(weights)
    }
}

#[derive(Debug)]
pub struct DFlash2Weights<'data> {
    pub manifest: Vec<DFlash2ExpectedTensor>,
    pub tensors: Vec<TensorView<'data>>,
}

impl<'data> DFlash2Weights<'data> {
    fn load(bytes: &'data [u8], cfg: &DFlash2Config) -> Result<Self, DFlash2WeightsError> {
        let tensors = SafeTensors::deserialize(bytes)?;
        let mut manifest = expected_manifest(cfg);
        let mut expected_names: HashSet<&str> =
            manifest.iter().map(|tensor| tensor.name.as_str()).collect();
        expected_names.insert(OPTIONAL_MASK_EMBEDDING);
        for name in tensors.names() {
            if !expected_names.contains(name) {
                return Err(DFlash2WeightsError::Extra(name.to_string()));
            }
        }
        if tensors.tensor(OPTIONAL_MASK_EMBEDDING).is_ok() {
            manifest.push(DFlash2ExpectedTensor {
                name: OPTIONAL_MASK_EMBEDDING.into(),
                shape: vec![cfg.hidden_size],
            });
        }

        let mut views = Vec::with_capacity(manifest.len());
        for expected in &manifest {
            let view = tensors
                .tensor(&expected.name)
                .map_err(|error| match error {
                    safetensors::SafeTensorError::TensorNotFound(_) => {
                        DFlash2WeightsError::Missing(expected.name.clone())
                    }
                    other => DFlash2WeightsError::Safetensors(other),
                })?;
            if view.dtype() != DFLASH2_WEIGHT_DTYPE {
                return Err(DFlash2WeightsError::Dtype {
                    name: expected.name.clone(),
                    actual: view.dtype(),
                    expected: DFLASH2_WEIGHT_DTYPE,
                });
            }
            if view.shape() != expected.shape.as_slice() {
                return Err(DFlash2WeightsError::Shape {
                    name: expected.name.clone(),
                    actual: view.shape().to_vec(),
                    expected: expected.shape.clone(),
                });
            }
            views.push(view);
        }
        Ok(Self {
            manifest,
            tensors: views,
        })
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorView<'data>> {
        self.manifest
            .iter()
            .position(|tensor| tensor.name == name)
            .map(|index| &self.tensors[index])
    }

    pub fn total_data_bytes(&self) -> usize {
        self.tensors.iter().map(|tensor| tensor.data().len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> DFlash2Config {
        DFlash2Config::from_json_str(
            r#"{
              "architectures":["DFlash2DraftModel"], "attention_bias":false,
              "is_causal":false, "dtype":"bfloat16", "head_dim":8,
              "hidden_act":"silu", "hidden_size":16, "intermediate_size":32,
              "layer_types":["sliding_attention"], "max_position_embeddings":128,
              "max_window_layers":1, "model_type":"test", "num_attention_heads":2,
              "num_hidden_layers":1, "num_key_value_heads":1, "num_target_layers":2,
              "rms_norm_eps":0.000001,
              "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"},
              "sliding_window":64, "tie_word_embeddings":false,
              "use_sliding_window":true, "vocab_size":32,
              "dflash_config":{"block_size":8,"conv_group_size":16,
                "conv_kernel_size":2,"mask_token_id":31,"selector_rank":256,
                "selector_top_k":16,"target_layer_ids":[0]}
            }"#,
        )
        .unwrap()
    }

    fn checkpoint_bytes(cfg: &DFlash2Config, omit: Option<&str>) -> Vec<u8> {
        let manifest = expected_manifest(cfg);
        let storage: Vec<Vec<u8>> = manifest
            .iter()
            .filter(|tensor| Some(tensor.name.as_str()) != omit)
            .map(|tensor| vec![0; tensor.shape.iter().product::<usize>() * 2])
            .collect();
        let filtered: Vec<&DFlash2ExpectedTensor> = manifest
            .iter()
            .filter(|tensor| Some(tensor.name.as_str()) != omit)
            .collect();
        let views: Vec<TensorView<'_>> = filtered
            .iter()
            .zip(&storage)
            .map(|(tensor, bytes)| {
                TensorView::new(Dtype::BF16, tensor.shape.clone(), bytes).unwrap()
            })
            .collect();
        let entries: Vec<(String, &TensorView<'_>)> = filtered
            .iter()
            .zip(&views)
            .map(|(tensor, view)| (tensor.name.clone(), view))
            .collect();
        safetensors::tensor::serialize(entries, None).unwrap()
    }

    #[test]
    fn qwen38_manifest_has_all_81_tensors() {
        let cfg = DFlash2Config::from_json_str(super::super::config::tests::QWEN38_CONFIG).unwrap();
        let manifest = expected_manifest(&cfg);
        assert_eq!(manifest.len(), 81);
        assert_eq!(
            manifest
                .iter()
                .find(|tensor| tensor.name == "layers.0.attention_conv.base_kernel")
                .unwrap()
                .shape,
            [2, 2, 5_120]
        );
        assert_eq!(
            manifest
                .iter()
                .find(|tensor| {
                    tensor.name == "layers.0.attention_conv.kernel_projection.weight"
                })
                .unwrap()
                .shape,
            [1_280, 5_120]
        );
    }

    #[test]
    fn strict_loader_accepts_complete_checkpoint() {
        let cfg = small_config();
        let bytes = checkpoint_bytes(&cfg, None);
        let weights = DFlash2Weights::load(&bytes, &cfg).unwrap();
        assert_eq!(weights.manifest.len(), 21);
        assert_eq!(weights.tensors.len(), 21);
        assert!(weights.total_data_bytes() > 0);
    }

    #[test]
    fn strict_loader_rejects_missing_tensor() {
        let cfg = small_config();
        let bytes = checkpoint_bytes(&cfg, Some("norm.weight"));
        let error = DFlash2Weights::load(&bytes, &cfg).unwrap_err();
        assert!(matches!(error, DFlash2WeightsError::Missing(name) if name == "norm.weight"));
    }

    #[test]
    #[ignore = "requires pinned 3.85 GB local DFlash2 checkpoint"]
    fn loads_pinned_qwen38_checkpoint() {
        let root = std::path::Path::new(env!("HOME")).join(
            ".cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
        );
        let descriptor = &super::super::schema::QWEN38_DFLASH2;
        let cfg = descriptor
            .parse_config(&std::fs::read(root.join("config.json")).unwrap())
            .unwrap();
        let file =
            DFlash2WeightsFile::open_verified(root.join("model.safetensors"), descriptor).unwrap();
        let weights = file.load_verified(&cfg, descriptor).unwrap();
        assert_eq!(weights.manifest.len(), 81);
        assert_eq!(weights.total_data_bytes(), 3_848_808_960);
    }

    #[test]
    #[ignore = "requires pinned 1.05 GB local Qwen3.6 DFlash2 checkpoint"]
    fn loads_pinned_qwen36_checkpoint() {
        let root = std::path::Path::new(env!("HOME")).join(
            ".cache/huggingface/hub/models--incoai--Qwen3.6-35B-A3B-DFlash2/snapshots/51ef7b6923ad6c14cb1bb41c37a9041446496ab6",
        );
        let descriptor = &super::super::schema::QWEN36_DFLASH2;
        let cfg = descriptor
            .parse_config(&std::fs::read(root.join("config.json")).unwrap())
            .unwrap();
        let file =
            DFlash2WeightsFile::open_verified(root.join("model.safetensors"), descriptor).unwrap();
        let weights = file.load_verified(&cfg, descriptor).unwrap();
        assert_eq!(weights.manifest.len(), 96);
    }
}
