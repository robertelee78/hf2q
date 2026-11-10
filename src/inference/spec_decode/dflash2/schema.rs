//! Pinned DFlash2 artifact identities and family bindings (ADR-067 D0).

use super::config::{DFlash2Config, DFlash2ConfigError, DFlash2TargetShape};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DFlash2FamilyAdapter {
    QwenDense,
    QwenMoe,
    Gemma4,
    DeepSeek4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DFlash2ConfigSchema {
    FlatTransformersV1,
    NestedSpeculatorsV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DFlash2CheckpointDescriptor {
    pub id: &'static str,
    pub family_adapter: DFlash2FamilyAdapter,
    pub config_schema: DFlash2ConfigSchema,
    pub repository: &'static str,
    pub revision: &'static str,
    pub config_sha256: &'static str,
    pub weights_sha256: &'static str,
    pub weights_bytes: u64,
    pub tensor_count: usize,
    pub target_architecture: &'static str,
    pub target_model: &'static str,
    /// Canonical hf2q source-bundle identity of the exact target weights.
    /// `None` keeps production activation unavailable for this descriptor.
    pub target_source_sha256: Option<&'static str>,
    pub target_hidden_size: usize,
    pub target_vocab_size: usize,
    pub target_layers: usize,
    pub reference_repository: &'static str,
    pub reference_revision: &'static str,
}

pub const QWEN38_DFLASH2: DFlash2CheckpointDescriptor = DFlash2CheckpointDescriptor {
    id: "qwen38-27b-dflash2-50307d4c",
    family_adapter: DFlash2FamilyAdapter::QwenDense,
    config_schema: DFlash2ConfigSchema::FlatTransformersV1,
    repository: "z-lab/Qwen3.8-27B-DFlash2",
    revision: "50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
    config_sha256: "873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980",
    weights_sha256: "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c",
    weights_bytes: 3_848_817_896,
    tensor_count: 81,
    target_architecture: "qwen35",
    target_model: "Qwen/Qwen3.8-27B",
    target_source_sha256: Some("73ded708c49c2d0a47c790ce1d6181e848ac7591dab741de83dbb57218cc6873"),
    target_hidden_size: 5_120,
    target_vocab_size: 248_320,
    target_layers: 64,
    reference_repository: "splash",
    reference_revision: "031a25a6cc18eed83a99a9883de91dcaa272f160",
};

pub const QWEN36_DFLASH2: DFlash2CheckpointDescriptor = DFlash2CheckpointDescriptor {
    id: "qwen36-35b-a3b-dflash2-51ef7b69",
    family_adapter: DFlash2FamilyAdapter::QwenMoe,
    config_schema: DFlash2ConfigSchema::FlatTransformersV1,
    repository: "incoai/Qwen3.6-35B-A3B-DFlash2",
    revision: "51ef7b6923ad6c14cb1bb41c37a9041446496ab6",
    config_sha256: "86d2e8e57e7c70bb867240708d00c0535ae9ed3c828dbdcb21584471c7bbce59",
    weights_sha256: "d1f16c0ff85afbe6166fabfae998af3ec16fa11da87da4391b5e0e0a7c796a4d",
    weights_bytes: 1_052_513_456,
    tensor_count: 96,
    target_architecture: "qwen35moe",
    target_model: "Qwen/Qwen3.6-35B-A3B",
    target_source_sha256: Some("851811f43867ae177636f0c536bb98c33731d992ce357831809a1eda5eb974ca"),
    target_hidden_size: 2_048,
    target_vocab_size: 248_320,
    target_layers: 40,
    reference_repository: "splash",
    reference_revision: "031a25a6cc18eed83a99a9883de91dcaa272f160",
};

pub const GEMMA4_E2B_DFLASH2: DFlash2CheckpointDescriptor = DFlash2CheckpointDescriptor {
    id: "gemma4-e2b-dflash2-6eb3ade2",
    family_adapter: DFlash2FamilyAdapter::Gemma4,
    config_schema: DFlash2ConfigSchema::NestedSpeculatorsV1,
    repository: "rasyosef/gemma-4-E2B-it-dflash2",
    revision: "6eb3ade2e4ae841974d692e9edc177dcaa7e20ff",
    config_sha256: "44a97c599f09cf1b32859841e1b0ba4a5b8124b250b236db0e1335c545306f3e",
    weights_sha256: "842a33bc4f38cc054dc5df0784ee332b49b2a824956d60795d8e94f269024fee",
    weights_bytes: 646_093_904,
    tensor_count: 81,
    target_architecture: "gemma4",
    target_model: "google/gemma-4-E2B-it",
    target_source_sha256: None,
    target_hidden_size: 1_536,
    target_vocab_size: 262_144,
    target_layers: 35,
    reference_repository: "rasyosef/train-dspark-draft-models",
    reference_revision: "ca13206b7399cc54ee1fecf7a4c233741121289d",
};

pub const GEMMA4_31B_DFLASH2: DFlash2CheckpointDescriptor = DFlash2CheckpointDescriptor {
    id: "gemma4-31b-dflash2-b0cd01b4",
    family_adapter: DFlash2FamilyAdapter::Gemma4,
    config_schema: DFlash2ConfigSchema::NestedSpeculatorsV1,
    repository: "RedHatAI/gemma-4-31B-it-speculator.dflash2",
    revision: "b0cd01b4918aae6b6388272950c4f9851734674b",
    config_sha256: "2eb1b6cad1c0736b7735a1b3f12f398d7e835c00e2f0c97c1e40ed9eb90135ba",
    weights_sha256: "3e82ec115f31020a81a69c6145aee727996ea9d83664f6285728e6bff0dd6b0f",
    weights_bytes: 5_494_652_680,
    tensor_count: 81,
    target_architecture: "gemma4",
    target_model: "google/gemma-4-31B-it",
    target_source_sha256: None,
    target_hidden_size: 5_376,
    target_vocab_size: 262_144,
    target_layers: 60,
    reference_repository: "vllm-project/speculators",
    reference_revision: "442ce1bf4e2f70dd0b45f36e63bd5849fbb35912",
};

pub const KNOWN_DFLASH2_CHECKPOINTS: &[DFlash2CheckpointDescriptor] = &[
    QWEN38_DFLASH2,
    QWEN36_DFLASH2,
    GEMMA4_E2B_DFLASH2,
    GEMMA4_31B_DFLASH2,
];

#[derive(Debug, thiserror::Error)]
pub enum DFlash2SchemaError {
    #[error("unknown DFlash2 checkpoint {repository}@{revision}")]
    Unknown {
        repository: String,
        revision: String,
    },
    #[error("DFlash2 artifact IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("DFlash2 artifact {artifact} has {actual}, expected {expected}")]
    Identity {
        artifact: &'static str,
        actual: String,
        expected: String,
    },
    #[error(transparent)]
    Config(#[from] DFlash2ConfigError),
}

pub fn descriptor(
    repository: &str,
    revision: &str,
) -> Result<&'static DFlash2CheckpointDescriptor, DFlash2SchemaError> {
    KNOWN_DFLASH2_CHECKPOINTS
        .iter()
        .find(|entry| entry.repository == repository && entry.revision == revision)
        .ok_or_else(|| DFlash2SchemaError::Unknown {
            repository: repository.into(),
            revision: revision.into(),
        })
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

impl DFlash2CheckpointDescriptor {
    pub fn parse_config(&self, bytes: &[u8]) -> Result<DFlash2Config, DFlash2SchemaError> {
        let actual = sha256(bytes);
        if actual != self.config_sha256 {
            return Err(DFlash2SchemaError::Identity {
                artifact: "config.json SHA256",
                actual,
                expected: self.config_sha256.into(),
            });
        }
        let config =
            DFlash2Config::from_json_str(std::str::from_utf8(bytes).map_err(|error| {
                DFlash2ConfigError::Invalid(format!("config.json must be UTF-8: {error}"))
            })?)?;
        config.validate_target(DFlash2TargetShape {
            architecture: self.target_architecture,
            hidden_size: self.target_hidden_size,
            vocab_size: self.target_vocab_size,
            num_hidden_layers: self.target_layers,
        })?;
        Ok(config)
    }

    pub fn verify_weights_bytes(&self, bytes: &[u8]) -> Result<(), DFlash2SchemaError> {
        if bytes.len() as u64 != self.weights_bytes {
            return Err(DFlash2SchemaError::Identity {
                artifact: "model.safetensors byte length",
                actual: bytes.len().to_string(),
                expected: self.weights_bytes.to_string(),
            });
        }
        let actual = sha256(bytes);
        if actual != self.weights_sha256 {
            return Err(DFlash2SchemaError::Identity {
                artifact: "model.safetensors SHA256",
                actual,
                expected: self.weights_sha256.into(),
            });
        }
        Ok(())
    }

    pub fn validate_target_identity(
        &self,
        architecture: &str,
        source_model: &str,
    ) -> Result<(), DFlash2SchemaError> {
        if architecture != self.target_architecture {
            return Err(DFlash2SchemaError::Identity {
                artifact: "target architecture",
                actual: architecture.into(),
                expected: self.target_architecture.into(),
            });
        }
        if source_model != self.target_model {
            return Err(DFlash2SchemaError::Identity {
                artifact: "target source model",
                actual: source_model.into(),
                expected: self.target_model.into(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_revision_is_fail_closed() {
        let error = descriptor(QWEN38_DFLASH2.repository, "main").unwrap_err();
        assert!(matches!(error, DFlash2SchemaError::Unknown { .. }));
    }

    #[test]
    fn cross_family_target_is_rejected() {
        let error = GEMMA4_E2B_DFLASH2
            .validate_target_identity("qwen35", GEMMA4_E2B_DFLASH2.target_model)
            .unwrap_err();
        assert!(matches!(
            error,
            DFlash2SchemaError::Identity {
                artifact: "target architecture",
                ..
            }
        ));
    }
}
