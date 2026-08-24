//! Authoritative dense-Qwen source-to-stored conversion and persisted replay.
//!
//! This is deliberately a child of `cli_driver`: it reuses the exact private
//! mapper/plan/writer path without widening that implementation as a public
//! API, while keeping evidence policy and replay out of the legacy driver.

use std::fs::File;
use std::io::Read;

use super::*;
use crate::convert::tensor_lineage::{
    tensor_conversion_receipt_path, validate_stored_tensor_conversion_receipt,
    StoredTensorConversionReceipt, MAX_STORED_TENSOR_CONVERSION_RECEIPT_BYTES,
};

/// Non-cloneable capability over the exact GGUF inode admitted by D2b.
///
/// The conversion receipt alone is not runtime authority: a pathname can be
/// replaced after replay. D2c loaders must consume this retained GGUF handle,
/// then rehash `artifact` after loading before they may mint loaded/executed
/// evidence. Non-tensor GGUF metadata remains outside the D2b claim and must
/// be cross-checked against authenticated source configuration by that loader.
#[allow(dead_code)] // consumed by the loaded/executed Qwen producer slice
pub(crate) struct VerifiedStoredQwenArtifact {
    conversion: VerifiedSourceToStoredConversion,
    artifact: File,
    gguf: mlx_native::GgufFile,
    source_config: serde_json::Value,
}

/// Result of one load through the retained GGUF parser followed by automatic
/// same-inode reconciliation. This proves artifact continuity only; D2c must
/// separately authenticate the loaded value and its executed buffers.
#[allow(dead_code)] // consumed by the loaded/executed Qwen producer slice
pub(crate) struct RetainedQwenArtifactLoad<T> {
    value: T,
    conversion: VerifiedSourceToStoredConversion,
}

#[allow(dead_code)] // consumed by the loaded/executed Qwen producer slice
impl<T> RetainedQwenArtifactLoad<T> {
    pub(crate) fn value(&self) -> &T {
        &self.value
    }

    pub(crate) fn conversion(&self) -> &VerifiedSourceToStoredConversion {
        &self.conversion
    }

    pub(crate) fn try_map<U, E>(
        self,
        map: impl FnOnce(T, &VerifiedSourceToStoredConversion) -> Result<U, E>,
    ) -> Result<RetainedQwenArtifactLoad<U>, E> {
        let value = map(self.value, &self.conversion)?;
        Ok(RetainedQwenArtifactLoad {
            value,
            conversion: self.conversion,
        })
    }
}

#[allow(dead_code)] // consumed by the loaded/executed Qwen producer slice
impl VerifiedStoredQwenArtifact {
    /// Consume this capability, load through its exact verified parser, then
    /// rehash the retained inode. The mandatory final hash detects persistent
    /// mutation; independently verifying the loaded values remains necessary
    /// to reject a transient mutate-and-restore race.
    pub(crate) fn load_and_reconcile<T>(
        self,
        load: impl FnOnce(
            &mlx_native::GgufFile,
            &serde_json::Value,
            &VerifiedSourceToStoredConversion,
        ) -> anyhow::Result<T>,
    ) -> Result<RetainedQwenArtifactLoad<T>, ConvertError> {
        let value = load(&self.gguf, &self.source_config, &self.conversion).map_err(|error| {
            ConvertError::Source(SourceError::Safetensors(format!(
                "load retained Qwen GGUF: {error}"
            )))
        })?;
        verify_open_artifact_identity(&self.artifact, &self.conversion.receipt().gguf_container)
            .map_err(|error| {
                ConvertError::Source(SourceError::Safetensors(format!(
                    "verified stored artifact changed after replay: {error}"
                )))
            })?;
        Ok(RetainedQwenArtifactLoad {
            value,
            conversion: self.conversion,
        })
    }
}

/// Run the same production conversion loop while producing authoritative
/// source-to-stored evidence. This remains crate-private until the canonical
/// tensor-conversion receipt and CLI/autoquant admission are complete.
#[allow(dead_code)] // consumed by the next Dynamic materializer/admission slice
pub(crate) fn run_convert_with_stored_evidence(
    args: ConvertArgs,
    verified_source: VerifiedSourceManifest,
    context: VerifiedConversionEvidenceContext,
) -> Result<VerifiedSourceToStoredConversion, ConvertError> {
    if args.dry_run {
        return Err(ConvertError::Source(SourceError::Safetensors(
            "stored conversion evidence cannot be produced by a dry run".into(),
        )));
    }
    if !matches!(args.mode, super::ConvertMode::TextOnly) {
        return Err(ConvertError::UnsupportedArch {
            arch_name: "stored conversion evidence is text-only; paired/projector conversion is not admitted".into(),
        });
    }
    if args.imatrix.is_some()
        || args.imatrix_corpus.is_some()
        || args.imatrix_out.is_some()
        || args.imatrix_n_ctx.is_some()
    {
        return Err(ConvertError::Source(SourceError::Safetensors(
            "stored conversion evidence v1 does not admit imatrix calibration".into(),
        )));
    }
    if !matches!(
        args.selector,
        QuantSelector::Standard(GgufFtype::MostlyQ4_K_M | GgufFtype::MostlyQ8_0)
    ) {
        return Err(ConvertError::Source(SourceError::Safetensors(
            "stored conversion evidence v1 admits only q4_k_m and q8_0 standard selectors".into(),
        )));
    }
    let remote = args.remote_source.as_ref().ok_or_else(|| {
        ConvertError::Source(SourceError::Safetensors(
            "stored conversion evidence requires an exact verified remote source".into(),
        ))
    })?;
    if remote.reference().repo_id() != context.source().model_id
        || remote.reference().revision() != context.source().revision
        || remote.reference().filename().is_some()
        || remote.source_sha256() != context.source().tensor_bundle_sha256
    {
        return Err(ConvertError::Source(SourceError::Safetensors(
            "remote conversion source differs from the D1 source identity".into(),
        )));
    }
    let source_artifacts = verify_source_weight_artifacts(&args.hf_dir, &verified_source, &context)
        .map_err(|error| {
            ConvertError::Source(SourceError::Safetensors(format!(
                "stored evidence source verification failed: {error}"
            )))
        })?;
    let converter_git_commit = require_converter_git_commit()?;
    run_convert_internal(
        args,
        Some(StoredEvidenceRequest {
            verified_source,
            context,
            source_artifacts,
            converter_git_commit,
        }),
        PairBinding::default(),
        true,
    )?
    .verified_conversion
    .ok_or_else(|| {
        ConvertError::Source(SourceError::Safetensors(
            "stored-evidence conversion did not produce a verified catalog".into(),
        ))
    })
}

/// Independently reconstruct a persisted source-to-stored receipt from the
/// authenticated source bytes, the production Qwen mapping/bake path, and the
/// exact promoted GGUF inode. A self-consistent JSON sidecar is never enough
/// to recover the opaque authority used by Dynamic admission.
#[allow(dead_code)] // consumed by the next Dynamic materializer/admission slice
#[cfg(test)]
pub(crate) fn verify_persisted_stored_evidence(
    model_dir: &Path,
    output: &Path,
    verified_source: &VerifiedSourceManifest,
    context: &VerifiedConversionEvidenceContext,
) -> Result<VerifiedSourceToStoredConversion, ConvertError> {
    verify_persisted_stored_artifact(model_dir, output, verified_source, context)
        .map(|artifact| artifact.conversion)
}

/// Replay D2b and retain the exact authenticated GGUF inode for D2c loading.
#[allow(dead_code)] // consumed by the loaded/executed Qwen producer slice
pub(crate) fn verify_persisted_stored_artifact(
    model_dir: &Path,
    output: &Path,
    verified_source: &VerifiedSourceManifest,
    context: &VerifiedConversionEvidenceContext,
) -> Result<VerifiedStoredQwenArtifact, ConvertError> {
    let receipt_path = tensor_conversion_receipt_path(output);
    let persisted_bytes = read_bounded_stored_conversion_receipt(&receipt_path)?;
    let persisted: StoredTensorConversionReceipt = serde_json::from_slice(&persisted_bytes)
        .map_err(|error| {
            ConvertError::Source(SourceError::Safetensors(format!(
                "parse stored conversion receipt: {error}"
            )))
        })?;
    validate_stored_tensor_conversion_receipt(&persisted).map_err(|error| {
        ConvertError::Source(SourceError::Safetensors(format!(
            "validate stored conversion receipt: {error}"
        )))
    })?;
    if &persisted.source != context.source()
        || persisted.verified_source_manifest_sha256 != context.verified_source_manifest_sha256()
        || persisted.source_inventory_manifest_sha256 != context.source_inventory_manifest_sha256()
        || persisted.tensor_partition_manifest_sha256 != context.tensor_partition_manifest_sha256()
    {
        return Err(ConvertError::Source(SourceError::Safetensors(
            "stored conversion receipt differs from the D1 evidence authority".into(),
        )));
    }
    let current_converter_git_commit = require_converter_git_commit()?;
    if persisted.producer.git_commit != current_converter_git_commit {
        return Err(ConvertError::Source(SourceError::Safetensors(
            "stored conversion receipt was not produced by this exact hf2q revision".into(),
        )));
    }

    let initial_source_artifacts =
        verify_source_weight_artifacts(model_dir, verified_source, context).map_err(|error| {
            ConvertError::Source(SourceError::Safetensors(format!(
                "persisted receipt source verification failed: {error}"
            )))
        })?;
    if initial_source_artifacts != persisted.source_artifacts {
        return Err(ConvertError::Source(SourceError::Safetensors(
            "persisted receipt source artifacts differ from current authenticated bytes".into(),
        )));
    }
    let source = VerifiedEvidenceSource::open(
        model_dir,
        verified_source,
        &context.source().model_id,
        &context.source().revision,
        &context.source().config_sha256,
    )?;
    let arch = detect_arch(&source.config)?;
    if arch != ArchName::Qwen35 {
        return Err(ConvertError::UnsupportedArch {
            arch_name: format!(
                "stored conversion evidence v1 requires dense Qwen3.8, detected {arch:?}"
            ),
        });
    }
    let plan = build_convert_plan(arch, &source.config, source.tensor_metas(), &[])?;
    let mapped_source_names: Result<Vec<&str>, ConvertError> = plan
        .steps
        .iter()
        .map(|step| match step {
            PlanStep::Direct { hf_name, .. } => Ok(hf_name.as_str()),
            _ => Err(ConvertError::Source(SourceError::Safetensors(
                "persisted D2b evidence contains a non-direct Qwen mapping".into(),
            ))),
        })
        .collect();
    context
        .validate_dense_qwen_source_coverage(mapped_source_names?, &plan.dropped_source_names)
        .map_err(|error| {
            ConvertError::Source(SourceError::Safetensors(format!(
                "persisted receipt source coverage failed: {error}"
            )))
        })?;

    let replay_ftype = GgufFtype::from_name(&persisted.policy.selector).ok_or_else(|| {
        ConvertError::Source(SourceError::Safetensors(
            "persisted receipt selector is not a standard GGML policy".into(),
        ))
    })?;
    if !matches!(
        replay_ftype,
        GgufFtype::MostlyQ4_K_M | GgufFtype::MostlyQ8_0
    ) {
        return Err(ConvertError::Source(SourceError::Safetensors(
            "persisted receipt selector is outside stored-evidence v1".into(),
        )));
    }
    let mut replay_orchestrator =
        ConvertOrchestrator::new(replay_ftype, arch, build_hparams(&source.config)?);
    replay_orchestrator.plan_tensors(plan.steps.iter().map(PlanStep::plan_entry).collect())?;

    let artifact = File::open(output)?;
    let gguf = mlx_native::GgufFile::from_file(artifact.try_clone()?).map_err(|error| {
        ConvertError::Source(SourceError::Safetensors(format!(
            "parse persisted GGUF evidence artifact: {error}"
        )))
    })?;
    let mut materialized = Vec::with_capacity(plan.steps.len());
    let mut writes = Vec::with_capacity(plan.steps.len());
    for (plan_index, step) in plan.steps.iter().enumerate() {
        let evidence = step.materialize_with_evidence(plan_index, &source, context)?;
        let relative_payload_offset = gguf
            .tensor_info(&evidence.record.gguf_tensor_name)
            .ok_or_else(|| {
                ConvertError::Source(SourceError::Safetensors(format!(
                    "persisted GGUF is missing planned tensor {}",
                    evidence.record.gguf_tensor_name
                )))
            })?
            .offset;
        writes.push(replay_orchestrator.reproduce_tensor_write_evidence(
            plan_index,
            &evidence.data,
            relative_payload_offset,
        )?);
        materialized.push(evidence.record);
        drop(evidence.data);
    }

    let stored =
        verify_written_tensor_evidence_open(&artifact, &gguf, &writes).map_err(|error| {
            ConvertError::Source(SourceError::Safetensors(format!(
                "verify persisted GGUF bytes: {error}"
            )))
        })?;
    verify_source_to_stored_continuity(&materialized, &writes, &stored).map_err(|error| {
        ConvertError::Source(SourceError::Safetensors(format!(
            "verify persisted source-to-stored continuity: {error}"
        )))
    })?;

    let final_source_artifacts =
        verify_source_weight_artifacts(model_dir, verified_source, context).map_err(|error| {
            ConvertError::Source(SourceError::Safetensors(format!(
                "persisted receipt source changed during verification: {error}"
            )))
        })?;
    if final_source_artifacts != initial_source_artifacts {
        return Err(ConvertError::Source(SourceError::Safetensors(
            "source artifacts changed during persisted receipt verification".into(),
        )));
    }
    verify_complete_conversion_source(model_dir, verified_source, context).map_err(|error| {
        ConvertError::Source(SourceError::Safetensors(format!(
            "conversion metadata source changed during persisted receipt verification: {error}"
        )))
    })?;
    let verified = build_verified_source_to_stored_conversion(
        context,
        &final_source_artifacts,
        &materialized,
        &writes,
        &stored,
        &current_converter_git_commit,
        &persisted.policy.selector,
    )
    .map_err(|error| {
        ConvertError::Source(SourceError::Safetensors(format!(
            "rebuild persisted stored conversion receipt: {error}"
        )))
    })?;
    if verified.receipt() != &persisted {
        return Err(ConvertError::Source(SourceError::Safetensors(
            "persisted stored conversion receipt does not reproduce from source and GGUF bytes"
                .into(),
        )));
    }
    verify_open_artifact_identity(&artifact, &verified.receipt().gguf_container).map_err(
        |error| {
            ConvertError::Source(SourceError::Safetensors(format!(
                "final persisted GGUF identity verification failed: {error}"
            )))
        },
    )?;
    Ok(VerifiedStoredQwenArtifact {
        conversion: verified,
        artifact,
        gguf,
        source_config: source.config.clone(),
    })
}

fn read_bounded_stored_conversion_receipt(path: &Path) -> Result<Vec<u8>, ConvertError> {
    let file = File::open(path).map_err(ConvertError::Io)?;
    let capacity = file
        .metadata()
        .map_err(ConvertError::Io)?
        .len()
        .min(MAX_STORED_TENSOR_CONVERSION_RECEIPT_BYTES) as usize;
    let mut bytes = Vec::with_capacity(capacity);
    file.take(
        MAX_STORED_TENSOR_CONVERSION_RECEIPT_BYTES
            .checked_add(1)
            .expect("receipt byte bound is below u64::MAX"),
    )
    .read_to_end(&mut bytes)
    .map_err(ConvertError::Io)?;
    if bytes.is_empty()
        || u64::try_from(bytes.len()).unwrap_or(u64::MAX)
            > MAX_STORED_TENSOR_CONVERSION_RECEIPT_BYTES
    {
        return Err(ConvertError::Source(SourceError::Safetensors(format!(
            "stored conversion receipt size {} is outside the v1 bound",
            bytes.len()
        ))));
    }
    Ok(bytes)
}
