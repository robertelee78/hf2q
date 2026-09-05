//! Runtime architecture registrations and their allocation-bounded admission.
//!
//! The Hub resolver consumes these callbacks; it does not maintain its own
//! architecture allowlist. A filename or publisher label grants no capability.

use anyhow::{bail, Result};
use mlx_native::gguf::GgufFile;

type Admission = fn(&GgufFile) -> Result<()>;

pub(crate) struct RuntimeContract {
    pub(crate) architecture: &'static str,
    admission: Option<Admission>,
}

const CONTRACTS: &[RuntimeContract] = &[
    RuntimeContract {
        architecture: "gemma4",
        admission: Some(super::models::gemma4::admission::validate),
    },
    RuntimeContract {
        architecture: "qwen35",
        admission: Some(validate_qwen),
    },
    RuntimeContract {
        architecture: "qwen35moe",
        admission: Some(validate_qwen),
    },
    RuntimeContract {
        architecture: "deepseek4",
        // The existing loader owns this forward graph. Its complete bounded
        // hosted contract has not been established; do not imply otherwise.
        admission: None,
    },
];

pub(crate) fn runtime_contract(architecture: &str) -> Option<&'static RuntimeContract> {
    CONTRACTS
        .iter()
        .find(|entry| entry.architecture == architecture)
}

impl RuntimeContract {
    pub(crate) fn supports_hosted(&self) -> bool {
        self.admission.is_some()
    }

    pub(crate) fn validate_hosted(&self, gguf: &GgufFile) -> Result<()> {
        match self.admission {
            Some(validate) => validate(gguf),
            None => bail!(
                "architecture {:?} has no complete bounded runtime admission contract",
                self.architecture
            ),
        }
    }

    pub(crate) fn validate_before_load(&self, gguf: &GgufFile) -> Result<()> {
        if let Some(validate) = self.admission {
            validate(gguf)?;
        }
        Ok(())
    }
}

fn validate_qwen(gguf: &GgufFile) -> Result<()> {
    super::models::qwen35::admission::validate_qwen_runtime_admission(gguf)
        .map_err(anyhow::Error::msg)
}

pub(crate) fn validate_hosted(gguf: &GgufFile) -> Result<()> {
    let architecture = gguf.metadata_string("general.architecture").unwrap_or("");
    let contract = runtime_contract(architecture).ok_or_else(|| {
        anyhow::anyhow!("architecture {architecture:?} has no registered primary serving runtime")
    })?;
    contract.validate_hosted(gguf)
}

#[cfg(test)]
mod tests {
    #[test]
    fn hosted_resolution_kata_unknown_architecture_is_never_approximated() {
        assert!(super::runtime_contract("gemma4-assistant").is_none());
        assert!(super::runtime_contract("anything-else").is_none());
        assert!(super::runtime_contract("gemma4").is_some());
    }
}
