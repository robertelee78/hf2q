//! Typed, path-free identity for policy that survives model replacement.
//!
//! Model-local sidecars and artifact paths belong to `EngineConfigIdentity`.
//! This smaller identity contains only process-wide serving and native GGML
//! routing choices, so the model-swap gate can prove one policy stayed active
//! while families and quantization formats were replaced.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::api::engine::EngineMode;
use super::multi_model::EngineConfig;

pub(crate) const PROCESS_POLICY_SCHEMA_VERSION: u32 = 1;
pub(crate) const Q5_CANONICAL_ROUTE: &str = "dense_q5k_canonical_q4x4";
const Q5_CANONICAL_PIPELINE_PREFIX: &str = "kernel_mul_mv_ext_q5_K_f32_r1_";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode")]
pub(crate) enum ProcessSchedulerPolicy {
    SerialFifo,
    SlotAware { max_slots: u32 },
}

impl From<EngineMode> for ProcessSchedulerPolicy {
    fn from(mode: EngineMode) -> Self {
        match mode {
            EngineMode::SerialFifo => Self::SerialFifo,
            EngineMode::SlotAware { max_slots } => Self::SlotAware { max_slots },
        }
    }
}

/// Complete process-wide policy included in a model-swap execution receipt.
///
/// The native routing policy is resolved by mlx-native itself. Direct
/// `hf2q serve` therefore retains the dependency's typed defaults; canonical
/// launchers may pass explicit, validated environment values before startup.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ServingProcessPolicy {
    pub schema_version: u32,
    pub scheduler: ProcessSchedulerPolicy,
    pub requested_context_tokens: Option<u32>,
    pub kv_cache_budget_bytes: Option<u64>,
    pub queue_capacity: usize,
    pub warmup_synchronously: bool,
    pub kv_metrics_sink: bool,
    pub kv_persist_enabled: bool,
    pub kv_persist_budget_bytes: u64,
    pub ggml_routing: mlx_native::GgmlRoutingPolicy,
}

impl ServingProcessPolicy {
    pub(crate) fn resolve(config: &EngineConfig) -> Self {
        Self::with_routing(config, mlx_native::ggml_routing_policy_from_environment())
    }

    fn with_routing(config: &EngineConfig, ggml_routing: mlx_native::GgmlRoutingPolicy) -> Self {
        Self {
            schema_version: PROCESS_POLICY_SCHEMA_VERSION,
            scheduler: config.engine_mode.into(),
            requested_context_tokens: config.requested_context.map(|request| request.tokens),
            kv_cache_budget_bytes: config.kv_cache_budget_bytes,
            queue_capacity: config.queue_capacity,
            warmup_synchronously: config.warmup_synchronously,
            kv_metrics_sink: config.kv_metrics_sink.is_some(),
            kv_persist_enabled: config.kv_persist_dir.is_some(),
            kv_persist_budget_bytes: config.kv_persist_budget_bytes,
            ggml_routing,
        }
    }

    pub(crate) fn sha256(&self) -> Result<String, serde_json::Error> {
        // Hash the same JSON value representation emitted by the runtime
        // endpoint, not Rust declaration order.
        let encoded = serde_json::to_vec(&serde_json::to_value(self)?)?;
        Ok(hex::encode(Sha256::digest(encoded)))
    }
}

/// Number of concrete canonical Q5_K projection dispatches encoded so far.
/// The count is available only when the gate starts the process with
/// `MLX_DISP_BUCKET=1`; an empty/disabled bucket set correctly yields zero.
pub(crate) fn q5_canonical_route_dispatches() -> u64 {
    mlx_native::pipeline_dispatch_buckets()
        .into_iter()
        .filter(|(label, _)| label.starts_with(Q5_CANONICAL_PIPELINE_PREFIX))
        .map(|(_, count)| count)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_policy_hash_changes_with_q5_route_and_not_model_sidecars() {
        let mut config = EngineConfig {
            queue_capacity: 17,
            warmup_synchronously: true,
            engine_mode: EngineMode::SlotAware { max_slots: 4 },
            kv_cache_budget_bytes: Some(4096),
            kv_persist_budget_bytes: 8192,
            ..EngineConfig::default()
        };
        let mut routing = mlx_native::GgmlRoutingPolicy::default();
        routing.dense_q5k_canonical_q4x4 = false;
        let off = ServingProcessPolicy::with_routing(&config, routing);
        routing.dense_q5k_canonical_q4x4 = true;
        let on = ServingProcessPolicy::with_routing(&config, routing);
        assert_ne!(off, on);
        assert_ne!(off.sha256().unwrap(), on.sha256().unwrap());
        let round_trip: ServingProcessPolicy =
            serde_json::from_value(serde_json::to_value(&on).unwrap()).unwrap();
        assert_eq!(round_trip, on);

        config.tokenizer_path = Some("/model-local/tokenizer.json".into());
        config.config_path = Some("/model-local/config.json".into());
        config.dwq_overlay_path = Some("/model-local/overlay.safetensors".into());
        assert_eq!(
            on,
            ServingProcessPolicy::with_routing(&config, routing),
            "artifact-local sidecars must not change process policy"
        );
    }

    #[test]
    fn canonical_route_counter_ignores_adjacent_q5_and_width_mn_labels() {
        assert!("kernel_mul_mv_ext_q5_K_f32_r1_4|600:i2|601:i8"
            .starts_with(Q5_CANONICAL_PIPELINE_PREFIX));
        assert!(!"kernel_mul_mv_q5_K_f32_mN_r1_4".starts_with(Q5_CANONICAL_PIPELINE_PREFIX));
        assert!(!"kernel_mul_mv_ext_q5_0_f32_r1_4".starts_with(Q5_CANONICAL_PIPELINE_PREFIX));
    }
}
