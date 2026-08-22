//! Canonical runtime configuration for evidence-bearing dense Qwen execution.
//!
//! This is intentionally a semantic profile plus the exact native routing
//! policy, not a dump of every process environment variable. Concrete
//! prefill/decode routes are observed in mlx-native dispatch traces; benchmark
//! receipts bind scheduling and diagnostic state separately.

use anyhow::{bail, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::{Qwen35Config, Qwen35Variant};

pub(crate) const QWEN35_EXECUTION_CONFIGURATION_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Qwen35ExecutionProfile {
    /// Dense autoregressive text with copied GGUF loads, the current prefill
    /// arena topology, no MoE/vision/MTP/speculative execution, no fused QKVG,
    /// no F16 shadow weights, and no DWQ/MLX-affine overlay.
    DenseTextNoDwqCopiedGgmlV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Qwen35GateUpPolicy {
    /// Use a native fused gate/up/SILU entrypoint when the actual codec and
    /// path support it; the prefill arena may still use separate projections.
    PreferFusedWhenSupported,
    Separate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct Qwen35ExecutionConfiguration {
    schema_version: u32,
    profile: Qwen35ExecutionProfile,
    ggml_routing_policy: mlx_native::GgmlRoutingPolicy,
    dense_ffn_gate_up: Qwen35GateUpPolicy,
    routing_policy_sha256: String,
    graph_configuration_sha256: String,
}

#[derive(Serialize)]
struct ConfigurationHashView {
    schema_version: u32,
    profile: Qwen35ExecutionProfile,
    dense_ffn_gate_up: Qwen35GateUpPolicy,
}

fn sha256_json(value: &impl Serialize) -> Result<String> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(value)?)))
}

fn env_exact_one(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

impl Qwen35ExecutionConfiguration {
    /// Resolve every admitted tensor-route choice once at model load.
    ///
    /// The returned typed policy must be passed unchanged to capability
    /// requests and every explicit-policy native dispatch. Hot paths in the
    /// evidence profile may not reread these environment variables.
    pub(crate) fn resolve_no_dwq_from_environment<T: ?Sized>(
        model: &Qwen35Config,
        dwq_overlay: Option<&T>,
    ) -> Result<Self> {
        if model.variant != Qwen35Variant::Dense
            || model.moe.is_some()
            || model.intermediate_size.is_none()
        {
            bail!("evidence profile v1 admits only dense Qwen text models");
        }
        if dwq_overlay.is_some() {
            bail!("the evidence-bearing Qwen execution profile does not admit DWQ overlays");
        }
        if env_exact_one("HF2Q_FUSED_QKVG") {
            bail!("evidence profile v1 does not admit fused QKVG");
        }
        if std::env::var("HF2Q_DENSE_Q_ARENA_RESET").as_deref() == Ok("0") {
            bail!("evidence profile v1 requires the production prefill arena topology");
        }
        let dense_ffn_gate_up = if matches!(
            std::env::var("HF2Q_FUSED_GATE_UP_SILU").as_deref(),
            Ok("0") | Ok("false") | Ok("off")
        ) {
            Qwen35GateUpPolicy::Separate
        } else {
            Qwen35GateUpPolicy::PreferFusedWhenSupported
        };
        Self::from_resolved(
            mlx_native::ggml_routing_policy_from_environment(),
            dense_ffn_gate_up,
        )
    }

    pub(super) fn from_resolved(
        ggml_routing_policy: mlx_native::GgmlRoutingPolicy,
        dense_ffn_gate_up: Qwen35GateUpPolicy,
    ) -> Result<Self> {
        let routing_policy_sha256 = sha256_json(&ggml_routing_policy)?;
        let mut configuration = Self {
            schema_version: QWEN35_EXECUTION_CONFIGURATION_SCHEMA_VERSION,
            profile: Qwen35ExecutionProfile::DenseTextNoDwqCopiedGgmlV1,
            ggml_routing_policy,
            dense_ffn_gate_up,
            routing_policy_sha256,
            graph_configuration_sha256: String::new(),
        };
        configuration.graph_configuration_sha256 = configuration.recomputed_hash()?;
        Ok(configuration)
    }

    fn recomputed_hash(&self) -> Result<String> {
        sha256_json(&ConfigurationHashView {
            schema_version: self.schema_version,
            profile: self.profile,
            dense_ffn_gate_up: self.dense_ffn_gate_up,
        })
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema_version != QWEN35_EXECUTION_CONFIGURATION_SCHEMA_VERSION
            || self.profile != Qwen35ExecutionProfile::DenseTextNoDwqCopiedGgmlV1
            || self.routing_policy_sha256 != sha256_json(&self.ggml_routing_policy)?
            || self.graph_configuration_sha256 != self.recomputed_hash()?
        {
            bail!("Qwen execution configuration does not reproduce canonically");
        }
        Ok(())
    }

    pub(crate) fn ggml_routing_policy(&self) -> &mlx_native::GgmlRoutingPolicy {
        &self.ggml_routing_policy
    }

    pub(crate) fn dense_ffn_gate_up(&self) -> Qwen35GateUpPolicy {
        self.dense_ffn_gate_up
    }

    pub(crate) fn routing_policy_sha256(&self) -> &str {
        &self.routing_policy_sha256
    }

    pub(crate) fn graph_configuration_sha256(&self) -> &str {
        &self.graph_configuration_sha256
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_hashes_full_policy_and_fails_closed() {
        let configuration = Qwen35ExecutionConfiguration::from_resolved(
            mlx_native::GgmlRoutingPolicy::default(),
            Qwen35GateUpPolicy::PreferFusedWhenSupported,
        )
        .unwrap();
        configuration.validate().unwrap();
        assert_eq!(configuration.routing_policy_sha256().len(), 64);
        assert_eq!(configuration.graph_configuration_sha256().len(), 64);
        let mut changed = configuration.clone();
        changed.ggml_routing_policy.dense_decode_mvn =
            !changed.ggml_routing_policy.dense_decode_mvn;
        assert!(changed.validate().is_err());
    }

    #[test]
    fn gate_up_policy_changes_graph_identity() {
        let fused = Qwen35ExecutionConfiguration::from_resolved(
            mlx_native::GgmlRoutingPolicy::default(),
            Qwen35GateUpPolicy::PreferFusedWhenSupported,
        )
        .unwrap();
        let separate = Qwen35ExecutionConfiguration::from_resolved(
            mlx_native::GgmlRoutingPolicy::default(),
            Qwen35GateUpPolicy::Separate,
        )
        .unwrap();
        assert_ne!(
            fused.graph_configuration_sha256(),
            separate.graph_configuration_sha256()
        );
    }
}
