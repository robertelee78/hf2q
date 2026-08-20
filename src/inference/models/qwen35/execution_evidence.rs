//! Fail-closed entry to dense Qwen loaded/executed evidence production.
//!
//! This slice authenticates the source-derived execution configuration,
//! consumes the exact D2b GGUF inode, rehashes loaded/cache weight buffers,
//! records bounded typed host-encoding traces, and forbids DWQ by construction.
//! It does not prove Metal completion, numerical correctness, timing, quality,
//! or a cross-process hardware identity and therefore intentionally exposes no
//! runtime-cost manifest or Dynamic solver authority.

use anyhow::{bail, Context, Result};
use mlx_native::GgmlWorkloadClass;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use crate::convert::cli_driver::{RetainedQwenArtifactLoad, VerifiedStoredQwenArtifact};
use crate::serve::header::LoadProgress;
use crate::serve::multi_seq_kv::SlotId;

use super::execution_config::Qwen35ExecutionConfiguration;
use super::execution_dispatch::{
    with_execution_configuration, with_execution_trace_capture, Qwen35EncodedDispatchObservation,
    Qwen35TraceWeightSlot,
};
use super::execution_observation::{
    capture_loaded_tensor_catalog, VerifiedExecutedTensorCatalog, VerifiedLoadedTensorCatalog,
};
use super::execution_trace::{verify_encoded_dispatch_catalog, VerifiedEncodedDispatchCatalog};
use super::kv_cache::HybridKvCache;
use super::model::Qwen35Model;
use super::source_config::qwen35_config_from_authenticated_source;
use super::Qwen35Config;

const MAX_SESSION_ENCODED_DISPATCHES: usize = 8_192;

/// Opaque no-DWQ model candidate loaded through the exact D2b artifact inode.
///
/// Only immutable model access is exposed. Loaded/executed byte observations
/// and cache-epoch sealing are required before a later type may call itself
/// runtime-verified.
pub(crate) struct NoDwqQwen35LoadedCandidate {
    loaded: RetainedQwenArtifactLoad<LoadedQwenCandidateState>,
    execution: Qwen35ExecutionConfiguration,
}

struct LoadedQwenCandidateState {
    model: Qwen35Model,
    catalog: Arc<VerifiedLoadedTensorCatalog>,
}

/// Narrow autoregressive text session for the no-DWQ copied-load candidate.
///
/// The raw model and buffers remain inaccessible. Every forward call enters
/// the canonical execution scope, so the stored native policy and graph
/// switches are consumed rather than reread from mutable process state.
pub(crate) struct NoDwqQwen35TextSession<'a> {
    candidate: &'a NoDwqQwen35LoadedCandidate,
    kv_cache: HybridKvCache,
    executed_catalog: Arc<VerifiedExecutedTensorCatalog>,
    trace_weight_slots: BTreeMap<usize, Qwen35TraceWeightSlot>,
    dispatches: Vec<Qwen35EncodedDispatchObservation>,
    captured_prompt: bool,
    captured_decode: bool,
    poisoned: bool,
}

/// Unforgeable handoff from the exact retained-artifact load join to the
/// model's cache-invalidation state. Its fields and constructor remain private
/// to this producer module; it is not runtime or solver authority.
pub(super) struct ReconciledLoadedCandidateIdentity {
    configuration: Qwen35ExecutionConfiguration,
    conversion_receipt_sha256: String,
    loaded_catalog: Arc<VerifiedLoadedTensorCatalog>,
}

impl ReconciledLoadedCandidateIdentity {
    fn new(
        configuration: Qwen35ExecutionConfiguration,
        conversion_receipt_sha256: String,
        loaded_catalog: Arc<VerifiedLoadedTensorCatalog>,
    ) -> Self {
        Self {
            configuration,
            conversion_receipt_sha256,
            loaded_catalog,
        }
    }

    pub(super) fn into_parts(
        self,
    ) -> (
        Qwen35ExecutionConfiguration,
        String,
        Arc<VerifiedLoadedTensorCatalog>,
    ) {
        (
            self.configuration,
            self.conversion_receipt_sha256,
            self.loaded_catalog,
        )
    }
}

impl NoDwqQwen35LoadedCandidate {
    pub(crate) fn config(&self) -> &Qwen35Config {
        &self.loaded.value().model.cfg
    }

    pub(crate) fn execution(&self) -> &Qwen35ExecutionConfiguration {
        &self.execution
    }

    pub(crate) fn conversion_receipt_sha256(&self) -> &str {
        &self.loaded.conversion().receipt().receipt_sha256
    }

    pub(crate) fn loaded_catalog_sha256(&self) -> &str {
        self.loaded.value().catalog.catalog_sha256()
    }

    pub(crate) fn loaded_catalog_conversion_receipt_sha256(&self) -> &str {
        self.loaded.value().catalog.conversion_receipt_sha256()
    }

    pub(crate) fn loaded_tensor_count(&self) -> usize {
        self.loaded.value().catalog.observations().len()
    }

    pub(crate) fn start_text_session(
        &self,
        max_seq_len: u32,
    ) -> Result<NoDwqQwen35TextSession<'_>> {
        let model = &self.loaded.value().model;
        let kv_cache = with_execution_configuration(&self.execution, || {
            model.ensure_gpu_cache_primed()?;
            model.with_gpu_cache_mut(|device, _registry| {
                HybridKvCache::new(&model.cfg, device, max_seq_len, 1)
            })
        })?;
        let executed_catalog = model
            .loaded_candidate_executed_catalog()?
            .context("evidence-bearing Qwen cache omitted executed tensor observations")?;
        let trace_weight_slots = model
            .loaded_candidate_trace_slots()?
            .context("evidence-bearing Qwen cache omitted trace weight slots")?;
        Ok(NoDwqQwen35TextSession {
            candidate: self,
            kv_cache,
            executed_catalog,
            trace_weight_slots,
            dispatches: Vec::new(),
            captured_prompt: false,
            captured_decode: false,
            poisoned: false,
        })
    }
}

impl NoDwqQwen35TextSession<'_> {
    fn preflight_capture(&self) -> Result<()> {
        let maximum = self
            .trace_weight_slots
            .len()
            .checked_mul(2)
            .context("encoded dispatch observation bound overflow")?;
        if maximum > MAX_SESSION_ENCODED_DISPATCHES {
            bail!(
                "encoded dispatch topology exceeds bounded session limit {}",
                MAX_SESSION_ENCODED_DISPATCHES
            );
        }
        Ok(())
    }

    pub(crate) fn executed_catalog_sha256(&self) -> &str {
        self.executed_catalog.catalog_sha256()
    }

    pub(crate) fn executed_tensor_count(&self) -> usize {
        self.executed_catalog.observations().len()
    }

    pub(crate) fn executed_catalog_loaded_parent_sha256(&self) -> &str {
        self.executed_catalog.loaded_catalog_sha256()
    }

    pub(crate) fn encoded_dispatches(&self) -> &[Qwen35EncodedDispatchObservation] {
        &self.dispatches
    }

    pub(crate) fn seal_encoded_dispatches(self) -> Result<VerifiedEncodedDispatchCatalog> {
        verify_encoded_dispatch_catalog(
            &self.candidate.execution,
            &self.executed_catalog,
            self.dispatches,
        )
    }

    #[cfg(test)]
    pub(crate) fn duplicate_observation_fails_sealing(&self) -> bool {
        let mut observations = self.dispatches.clone();
        let Some(first) = observations.first().cloned() else {
            return false;
        };
        observations.push(first);
        verify_encoded_dispatch_catalog(
            &self.candidate.execution,
            &self.executed_catalog,
            observations,
        )
        .is_err()
    }

    pub(crate) fn forward(&mut self, tokens: &[u32], positions_flat: &[i32]) -> Result<Vec<f32>> {
        if self.poisoned {
            bail!("bounded evidence session is poisoned by an earlier failed execution");
        }
        if tokens.len() <= 8 {
            bail!(
                "bounded prompt evidence requires more than eight tokens; width-N and decode are separate regimes"
            );
        }
        if positions_flat.len() != tokens.len().saturating_mul(4) {
            bail!("bounded prompt evidence requires four MROPE positions per token");
        }
        if self.captured_decode {
            bail!("bounded prompt evidence cannot be captured after decode");
        }
        if self.captured_prompt {
            bail!("the bounded evidence session already captured its prompt execution");
        }
        self.preflight_capture()?;
        self.poisoned = true;
        let model = &self.candidate.loaded.value().model;
        let capture = with_execution_trace_capture(
            &self.candidate.execution,
            self.trace_weight_slots.clone(),
            GgmlWorkloadClass::Prompt,
            || model.forward_gpu(tokens, positions_flat, &mut self.kv_cache, SlotId(0)),
        );
        let (logits, dispatches) = capture?;
        self.captured_prompt = true;
        self.poisoned = false;
        self.dispatches.extend(dispatches);
        Ok(logits)
    }

    pub(crate) fn forward_greedy(&mut self, token: u32, positions_flat: [i32; 4]) -> Result<u32> {
        if self.poisoned {
            bail!("bounded evidence session is poisoned by an earlier failed execution");
        }
        if !self.captured_prompt {
            bail!("bounded decode evidence requires a successfully captured prompt first");
        }
        if self.captured_decode {
            bail!("the bounded evidence session already captured its decode execution");
        }
        self.preflight_capture()?;
        self.poisoned = true;
        let model = &self.candidate.loaded.value().model;
        let capture = with_execution_trace_capture(
            &self.candidate.execution,
            self.trace_weight_slots.clone(),
            GgmlWorkloadClass::DecodeSingle,
            || model.forward_gpu_greedy(&[token], &positions_flat, &mut self.kv_cache, SlotId(0)),
        );
        let (token, dispatches) = capture?;
        self.captured_decode = true;
        self.poisoned = false;
        self.dispatches.extend(dispatches);
        Ok(token)
    }
}

pub(crate) fn load_no_dwq_qwen35_candidate(
    artifact: VerifiedStoredQwenArtifact,
    dwq_overlay_path: Option<&Path>,
    progress: &mut LoadProgress,
) -> Result<NoDwqQwen35LoadedCandidate> {
    let loaded = artifact
        .load_and_reconcile(|gguf, source_config, conversion| {
            let actual = Qwen35Model::load_config_only(gguf)?;
            let expected = qwen35_config_from_authenticated_source(source_config)?;
            if actual != expected {
                bail!("GGUF execution configuration differs from authenticated source config");
            }
            let execution = Qwen35ExecutionConfiguration::resolve_no_dwq_from_environment(
                &actual,
                dwq_overlay_path,
            )?;
            execution.validate()?;
            let ((model, execution), catalog) = capture_loaded_tensor_catalog(conversion, || {
                let model = Qwen35Model::load_from_gguf(gguf, progress)
                    .context("load copied no-DWQ Qwen weights from retained GGUF")?;
                if model.cfg != expected {
                    bail!("loaded Qwen configuration drifted from authenticated source geometry");
                }
                validate_loaded_global_geometry(&model, &expected)?;
                Ok((model, execution))
            })?;
            Ok((model, execution, catalog))
        })
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let execution = loaded.value().1.clone();
    let loaded = loaded.try_map(|(mut model, _, catalog), conversion| {
        let catalog = Arc::new(catalog);
        model.bind_loaded_candidate_identity(ReconciledLoadedCandidateIdentity::new(
            execution.clone(),
            conversion.receipt().receipt_sha256.clone(),
            Arc::clone(&catalog),
        ))?;
        Ok::<_, anyhow::Error>(LoadedQwenCandidateState { model, catalog })
    })?;
    Ok(NoDwqQwen35LoadedCandidate { loaded, execution })
}

fn validate_loaded_global_geometry(model: &Qwen35Model, expected: &Qwen35Config) -> Result<()> {
    let expected_embedding_values = usize::try_from(expected.vocab_size)?
        .checked_mul(usize::try_from(expected.hidden_size)?)
        .context("authenticated Qwen embedding geometry overflows usize")?;
    if model.token_embd.len() != expected_embedding_values
        || model.output_weight.len() != expected_embedding_values
        || model.output_norm.len() != usize::try_from(expected.hidden_size)?
    {
        bail!(
            "loaded Qwen globals differ from authenticated source geometry; synthetic vocab extension is not admitted"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn loaded_candidate_identity_is_atomic_and_synthetic_vocab_is_rejected() {
        let config = json!({
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "intermediate_size": 32,
            "max_position_embeddings": 128,
            "vocab_size": 4
        });
        let projected = qwen35_config_from_authenticated_source(&config).unwrap();
        let resolved = Qwen35ExecutionConfiguration::from_resolved(
            mlx_native::GgmlRoutingPolicy::default(),
            super::super::execution_config::Qwen35GateUpPolicy::Separate,
        )
        .unwrap();
        let mut model = Qwen35Model::empty_from_cfg(projected.clone());
        let catalog = Arc::new(
            super::super::execution_observation::VerifiedLoadedTensorCatalog::for_test_empty(
                "a".repeat(64),
            ),
        );
        model
            .bind_loaded_candidate_identity(ReconciledLoadedCandidateIdentity::new(
                resolved.clone(),
                "a".repeat(64),
                Arc::clone(&catalog),
            ))
            .unwrap();
        assert_eq!(
            model.loaded_candidate_cache_identity(),
            Some((
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                resolved.graph_configuration_sha256(),
                resolved.routing_policy_sha256(),
            ))
        );
        assert!(model
            .bind_loaded_candidate_identity(ReconciledLoadedCandidateIdentity::new(
                resolved,
                "b".repeat(64),
                catalog,
            ))
            .is_err());

        let mut synthesized = Qwen35Model::empty_from_cfg(projected.clone());
        synthesized.token_embd.extend([0.0; 32]);
        assert!(validate_loaded_global_geometry(&synthesized, &projected).is_err());
    }
}
