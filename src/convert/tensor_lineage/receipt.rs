use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::convert::orchestrator::TensorWriteEvidence;
use crate::core::provenance::tensor_execution::ArtifactEvidence;
use crate::intelligence::measured_auto_quant::SourceIdentity;

use super::{
    ConversionSourceDisposition, ExcludedSourceTensorEvidence, MaterializedDirectTensorRecord,
    MaterializedSourceTensorEvidence, Qwen35DenseBakeStepEvidence,
    STORED_TENSOR_CONVERSION_RECEIPT_SCHEMA_VERSION, VerifiedConversionEvidenceContext,
    VerifiedStoredTensorCatalog, verify_source_to_stored_continuity,
};

pub(crate) const STORED_TENSOR_CONVERSION_SCOPE: &str =
    "qwen3_5_dense_autoregressive_text_source_to_stored_v1";
pub(crate) const STORED_TENSOR_GGUF_METADATA_SCOPE: &str =
    "container_identity_only_metadata_provenance_not_authenticated_v1";
pub(crate) const MAX_STORED_TENSOR_CONVERSION_RECEIPT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_STORED_TENSOR_RECEIPT_ITEMS: usize = 4_096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StoredConversionProducerIdentity {
    pub package: String,
    pub version: String,
    pub git_commit: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StoredConversionPolicyIdentity {
    pub selector: String,
    pub quantizer: String,
    pub calibration_manifest_sha256: Option<String>,
    pub imatrix_receipt_sha256: Option<String>,
    pub dwq_overlay_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StoredTensorPhysicalReceipt {
    pub tensor_name: String,
    pub shape_outermost_first: Vec<u64>,
    pub ggml_wire_type_id: u32,
    pub ggml_type_name: String,
    pub absolute_payload_offset: u64,
    pub payload_bytes: u64,
    pub payload_sha256: String,
    pub converted_f32_bytes_sha256: String,
    pub converted_logical_f32_sha256: String,
    pub f16_roundtrip_f32_bytes_sha256: Option<String>,
    pub f16_roundtrip_logical_f32_sha256: Option<String>,
    pub stored_f32_bytes_sha256: String,
    pub stored_logical_f32_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StoredTensorLineageReceipt {
    pub plan_index: usize,
    pub hf_tensor_name: String,
    pub gguf_tensor_name: String,
    pub disposition: ConversionSourceDisposition,
    pub source: MaterializedSourceTensorEvidence,
    pub bake_steps: Vec<Qwen35DenseBakeStepEvidence>,
    pub stored: StoredTensorPhysicalReceipt,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StoredTensorConversionReceipt {
    pub schema_version: u32,
    pub source: SourceIdentity,
    pub verified_source_manifest_sha256: String,
    pub source_inventory_manifest_sha256: String,
    pub tensor_partition_manifest_sha256: String,
    pub scope: String,
    pub producer: StoredConversionProducerIdentity,
    pub policy: StoredConversionPolicyIdentity,
    pub source_artifacts: Vec<ArtifactEvidence>,
    /// Exact container identity used to locate tensor regions. D2b v1 does
    /// not claim that non-tensor GGUF metadata was derived from authenticated
    /// tokenizer/model-card bytes; that is a separate serving admission gate.
    pub gguf_container: ArtifactEvidence,
    pub gguf_metadata_scope: String,
    pub tensor_lineages: Vec<StoredTensorLineageReceipt>,
    pub excluded_sources: Vec<ExcludedSourceTensorEvidence>,
    pub stored_tensor_count: usize,
    pub stored_payload_bytes: u64,
    pub receipt_sha256: String,
}

/// Opaque authority returned only after source, bake, writer, reopened-GGUF,
/// and receipt continuity have all been independently checked.
#[derive(Debug, Clone)]
pub(crate) struct VerifiedSourceToStoredConversion {
    receipt: StoredTensorConversionReceipt,
}

impl VerifiedSourceToStoredConversion {
    pub(crate) fn receipt(&self) -> &StoredTensorConversionReceipt {
        &self.receipt
    }

    #[cfg(test)]
    pub(crate) fn artifact_bytes(&self) -> u64 {
        self.receipt.gguf_container.byte_len
    }

    #[cfg(test)]
    pub(crate) fn artifact_sha256(&self) -> &str {
        &self.receipt.gguf_container.sha256
    }
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn is_lower_git_sha(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn normalized_receipt(receipt: &StoredTensorConversionReceipt) -> StoredTensorConversionReceipt {
    let mut normalized = receipt.clone();
    normalized
        .source_artifacts
        .sort_by(|left, right| left.artifact_id.cmp(&right.artifact_id));
    normalized.tensor_lineages.sort_by(|left, right| {
        left.plan_index
            .cmp(&right.plan_index)
            .then_with(|| left.hf_tensor_name.cmp(&right.hf_tensor_name))
    });
    normalized
        .excluded_sources
        .sort_by(|left, right| left.tensor_name.cmp(&right.tensor_name));
    normalized.receipt_sha256.clear();
    normalized
}

pub(crate) fn stored_tensor_conversion_receipt_sha256(
    receipt: &StoredTensorConversionReceipt,
) -> Result<String> {
    let bytes = serde_json::to_vec(&normalized_receipt(receipt))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_verified_source_to_stored_conversion(
    context: &VerifiedConversionEvidenceContext,
    source_artifacts: &[ArtifactEvidence],
    materialized: &[MaterializedDirectTensorRecord],
    writes: &[TensorWriteEvidence],
    stored: &VerifiedStoredTensorCatalog,
    converter_git_commit: &str,
    selector: &str,
) -> Result<VerifiedSourceToStoredConversion> {
    verify_source_to_stored_continuity(materialized, writes, stored)?;
    if !is_lower_git_sha(converter_git_commit) {
        bail!("stored conversion receipt requires a lowercase exact converter commit");
    }
    if !matches!(selector, "q4_k_m" | "q8_0") {
        bail!("stored conversion receipt selector is outside the v1 evidence scope");
    }
    if source_artifacts.is_empty() {
        bail!("stored conversion receipt has no source artifacts");
    }
    let mut source_artifact_ids = BTreeSet::new();
    for artifact in source_artifacts {
        if artifact.artifact_id.is_empty()
            || artifact.role != "source_safetensors_shard"
            || artifact.byte_len == 0
            || !is_lower_sha256(&artifact.sha256)
            || !source_artifact_ids.insert(artifact.artifact_id.as_str())
        {
            bail!("invalid or duplicate source artifact evidence");
        }
    }

    let mut tensor_lineages = Vec::with_capacity(materialized.len());
    let mut stored_payload_bytes = 0_u64;
    let mut represented_sources = BTreeSet::new();
    for (record, tensor) in materialized.iter().zip(stored.tensors()) {
        if !represented_sources.insert(record.source.tensor_name.as_str()) {
            bail!("stored conversion receipt has duplicate source lineages");
        }
        let inventory = context
            .records
            .get(&record.source.tensor_name)
            .ok_or_else(|| anyhow::anyhow!("stored lineage source is absent from D1 inventory"))?;
        if inventory.source_shape
            != record
                .source
                .shape_outermost_first
                .iter()
                .map(|dimension| usize::try_from(*dimension))
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("stored lineage source dimension is not representable")?
            || inventory.source_dtype != record.source.source_dtype
            || inventory.source_byte_len != record.source.payload_bytes
            || inventory.source_tensor_sha256 != record.source.payload_sha256
            || context.dispositions.get(&record.source.tensor_name)
                != Some(&record.source.disposition)
            || record.source.disposition == ConversionSourceDisposition::Excluded
        {
            bail!("stored lineage source differs from the D1 authority");
        }
        stored_payload_bytes = stored_payload_bytes
            .checked_add(tensor.payload_bytes)
            .context("stored conversion payload byte total overflow")?;
        tensor_lineages.push(StoredTensorLineageReceipt {
            plan_index: record.plan_index,
            hf_tensor_name: record.source.tensor_name.clone(),
            gguf_tensor_name: record.gguf_tensor_name.clone(),
            disposition: record.source.disposition,
            source: record.source.clone(),
            bake_steps: record.bake_steps.clone(),
            stored: StoredTensorPhysicalReceipt {
                tensor_name: tensor.tensor_name.clone(),
                shape_outermost_first: tensor.shape_outermost_first.clone(),
                ggml_wire_type_id: tensor.ggml_wire_type_id,
                ggml_type_name: tensor.ggml_type_name.clone(),
                absolute_payload_offset: tensor.absolute_payload_offset,
                payload_bytes: tensor.payload_bytes,
                payload_sha256: tensor.payload_sha256.clone(),
                converted_f32_bytes_sha256: tensor.converted_f32_bytes_sha256.clone(),
                converted_logical_f32_sha256: tensor.converted_logical_f32_sha256.clone(),
                f16_roundtrip_f32_bytes_sha256: tensor.f16_roundtrip_f32_bytes_sha256.clone(),
                f16_roundtrip_logical_f32_sha256: tensor.f16_roundtrip_logical_f32_sha256.clone(),
                stored_f32_bytes_sha256: tensor.stored_f32_bytes_sha256.clone(),
                stored_logical_f32_sha256: tensor.stored_logical_f32_sha256.clone(),
            },
        });
    }
    let expected_stored_sources: BTreeSet<_> = context
        .dispositions
        .iter()
        .filter_map(|(name, disposition)| {
            (*disposition != ConversionSourceDisposition::Excluded).then_some(name.as_str())
        })
        .collect();
    if represented_sources != expected_stored_sources {
        bail!("stored conversion receipt does not cover every non-excluded D1 source exactly");
    }

    let mut receipt = StoredTensorConversionReceipt {
        schema_version: STORED_TENSOR_CONVERSION_RECEIPT_SCHEMA_VERSION,
        source: context.source().clone(),
        verified_source_manifest_sha256: context.verified_source_manifest_sha256().into(),
        source_inventory_manifest_sha256: context.source_inventory_manifest_sha256().into(),
        tensor_partition_manifest_sha256: context.tensor_partition_manifest_sha256().into(),
        scope: STORED_TENSOR_CONVERSION_SCOPE.into(),
        producer: StoredConversionProducerIdentity {
            package: env!("CARGO_PKG_NAME").into(),
            version: env!("CARGO_PKG_VERSION").into(),
            git_commit: converter_git_commit.into(),
        },
        policy: StoredConversionPolicyIdentity {
            selector: selector.into(),
            quantizer: "hf2q-rust-ggml-round-to-nearest-v1".into(),
            calibration_manifest_sha256: None,
            imatrix_receipt_sha256: None,
            dwq_overlay_sha256: None,
        },
        source_artifacts: source_artifacts.to_vec(),
        gguf_container: ArtifactEvidence {
            artifact_id: "converted_gguf".into(),
            role: "stored_weight_container".into(),
            byte_len: stored.artifact_bytes(),
            sha256: stored.artifact_sha256().into(),
        },
        gguf_metadata_scope: STORED_TENSOR_GGUF_METADATA_SCOPE.into(),
        tensor_lineages,
        excluded_sources: context.excluded_sources().to_vec(),
        stored_tensor_count: stored.tensors().len(),
        stored_payload_bytes,
        receipt_sha256: String::new(),
    };
    receipt = normalized_receipt(&receipt);
    receipt.receipt_sha256 = stored_tensor_conversion_receipt_sha256(&receipt)?;
    validate_stored_tensor_conversion_receipt(&receipt)?;
    Ok(VerifiedSourceToStoredConversion { receipt })
}

pub(crate) fn validate_stored_tensor_conversion_receipt(
    receipt: &StoredTensorConversionReceipt,
) -> Result<()> {
    if receipt.schema_version != STORED_TENSOR_CONVERSION_RECEIPT_SCHEMA_VERSION
        || receipt.scope != STORED_TENSOR_CONVERSION_SCOPE
        || !is_lower_sha256(&receipt.verified_source_manifest_sha256)
        || !is_lower_sha256(&receipt.source_inventory_manifest_sha256)
        || !is_lower_sha256(&receipt.tensor_partition_manifest_sha256)
        || !is_lower_sha256(&receipt.gguf_container.sha256)
        || !is_lower_sha256(&receipt.receipt_sha256)
        || stored_tensor_conversion_receipt_sha256(receipt)? != receipt.receipt_sha256
    {
        bail!("stored conversion receipt identity or self-hash is invalid");
    }
    if receipt.gguf_container.artifact_id != "converted_gguf"
        || receipt.gguf_container.role != "stored_weight_container"
        || receipt.gguf_container.byte_len == 0
        || receipt.gguf_metadata_scope != STORED_TENSOR_GGUF_METADATA_SCOPE
    {
        bail!("stored conversion receipt output artifact is invalid");
    }
    if receipt.producer.package != env!("CARGO_PKG_NAME")
        || receipt.producer.version != env!("CARGO_PKG_VERSION")
        || !is_lower_git_sha(&receipt.producer.git_commit)
        || !matches!(receipt.policy.selector.as_str(), "q4_k_m" | "q8_0")
        || receipt.policy.quantizer != "hf2q-rust-ggml-round-to-nearest-v1"
    {
        bail!("stored conversion receipt producer or policy identity is invalid");
    }
    if receipt.source_artifacts.len() > MAX_STORED_TENSOR_RECEIPT_ITEMS
        || receipt.tensor_lineages.len() > MAX_STORED_TENSOR_RECEIPT_ITEMS
        || receipt.excluded_sources.len() > MAX_STORED_TENSOR_RECEIPT_ITEMS
    {
        bail!("stored conversion receipt collection count exceeds the v1 bound");
    }
    let mut artifact_ids = BTreeSet::new();
    if receipt.source_artifacts.is_empty()
        || receipt.source_artifacts.iter().any(|artifact| {
            artifact.artifact_id.is_empty()
                || artifact.role != "source_safetensors_shard"
                || artifact.byte_len == 0
                || !is_lower_sha256(&artifact.sha256)
                || !artifact_ids.insert(artifact.artifact_id.as_str())
        })
    {
        bail!("stored conversion receipt source artifacts are invalid");
    }
    if receipt.policy.calibration_manifest_sha256.is_some()
        || receipt.policy.imatrix_receipt_sha256.is_some()
        || receipt.policy.dwq_overlay_sha256.is_some()
    {
        bail!("stored conversion receipt v1 does not admit calibrated or DWQ state");
    }
    if receipt.stored_tensor_count == 0
        || receipt.stored_tensor_count != receipt.tensor_lineages.len()
    {
        bail!("stored conversion receipt tensor count is invalid");
    }
    let mut plan_indices = BTreeSet::new();
    let mut source_names = BTreeSet::new();
    let mut gguf_names = BTreeSet::new();
    let mut payload_bytes = 0_u64;
    for (canonical_index, lineage) in receipt.tensor_lineages.iter().enumerate() {
        if lineage.plan_index != canonical_index
            || !plan_indices.insert(lineage.plan_index)
            || !source_names.insert(lineage.hf_tensor_name.as_str())
            || !gguf_names.insert(lineage.gguf_tensor_name.as_str())
            || lineage.hf_tensor_name != lineage.source.tensor_name
            || lineage.gguf_tensor_name != lineage.stored.tensor_name
            || lineage.disposition != lineage.source.disposition
        {
            bail!("stored conversion receipt lineage identity is invalid");
        }
        if !artifact_ids.contains(lineage.source.artifact_id.as_str()) {
            bail!("stored conversion source tensor refers to an unknown artifact");
        }
        let source_hashes = [
            lineage.source.payload_sha256.as_str(),
            lineage.source.decoded_f32_bytes_sha256.as_str(),
            lineage.source.decoded_logical_f32_sha256.as_str(),
        ];
        let stored_hashes = [
            lineage.stored.payload_sha256.as_str(),
            lineage.stored.converted_f32_bytes_sha256.as_str(),
            lineage.stored.converted_logical_f32_sha256.as_str(),
            lineage.stored.stored_f32_bytes_sha256.as_str(),
            lineage.stored.stored_logical_f32_sha256.as_str(),
        ];
        if lineage.source.shape_outermost_first.is_empty()
            || lineage.stored.shape_outermost_first.is_empty()
            || lineage.source.payload_bytes == 0
            || lineage.stored.payload_bytes == 0
            || source_hashes.into_iter().any(|hash| !is_lower_sha256(hash))
            || stored_hashes.into_iter().any(|hash| !is_lower_sha256(hash))
        {
            bail!("stored conversion receipt lineage hashes or geometry are invalid");
        }
        let f16_pair = (
            lineage.stored.f16_roundtrip_f32_bytes_sha256.as_deref(),
            lineage.stored.f16_roundtrip_logical_f32_sha256.as_deref(),
        );
        let has_f16_roundtrip = match f16_pair {
            (None, None) => false,
            (Some(raw), Some(logical)) if is_lower_sha256(raw) && is_lower_sha256(logical) => true,
            _ => bail!("stored conversion receipt has an incomplete F16 roundtrip"),
        };
        let quantized = !matches!(
            lineage.stored.ggml_type_name.as_str(),
            "f32" | "f16" | "bf16" | "i16" | "i32"
        );
        if quantized != has_f16_roundtrip {
            bail!("stored conversion receipt F16 roundtrip does not match its wire type");
        }
        if lineage.bake_steps.iter().any(|step| {
            !is_lower_sha256(&step.input_f32_bytes_sha256)
                || !is_lower_sha256(&step.input_logical_f32_sha256)
                || !is_lower_sha256(&step.output_f32_bytes_sha256)
                || !is_lower_sha256(&step.output_logical_f32_sha256)
                || !is_lower_sha256(&step.step_sha256)
        }) {
            bail!("stored conversion receipt bake evidence is invalid");
        }
        payload_bytes = payload_bytes
            .checked_add(lineage.stored.payload_bytes)
            .context("stored conversion receipt payload total overflow")?;
    }
    if payload_bytes != receipt.stored_payload_bytes {
        bail!("stored conversion receipt payload total does not reproduce");
    }
    let excluded_names: BTreeSet<_> = receipt
        .excluded_sources
        .iter()
        .map(|source| source.tensor_name.as_str())
        .collect();
    if excluded_names.len() != receipt.excluded_sources.len()
        || excluded_names
            .iter()
            .any(|name| source_names.contains(name))
    {
        bail!("stored conversion receipt exclusions are duplicate or also stored");
    }
    Ok(())
}

pub(crate) fn tensor_conversion_receipt_path(output: &Path) -> PathBuf {
    let mut name = output.as_os_str().to_os_string();
    name.push(".tensor-conversion.json");
    PathBuf::from(name)
}

pub(crate) struct PreparedTensorConversionReceipt {
    temporary: tempfile::NamedTempFile,
    path: PathBuf,
}

pub(crate) fn prepare_tensor_conversion_receipt(
    output: &Path,
    verified: &VerifiedSourceToStoredConversion,
) -> Result<PreparedTensorConversionReceipt> {
    validate_stored_tensor_conversion_receipt(verified.receipt())?;
    let path = tensor_conversion_receipt_path(output);
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut temporary, verified.receipt())?;
    temporary.write_all(b"\n")?;
    temporary.as_file().sync_all()?;
    let receipt_bytes = temporary.as_file().metadata()?.len();
    if receipt_bytes == 0 || receipt_bytes > MAX_STORED_TENSOR_CONVERSION_RECEIPT_BYTES {
        bail!("stored conversion receipt serialized size exceeds the v1 bound");
    }
    Ok(PreparedTensorConversionReceipt { temporary, path })
}

pub(crate) fn promote_tensor_conversion_receipt(
    prepared: PreparedTensorConversionReceipt,
) -> Result<PathBuf> {
    prepared
        .temporary
        .persist(&prepared.path)
        .map_err(|error| error.error)?;
    Ok(prepared.path)
}

pub(crate) fn clear_stale_tensor_conversion_receipt(output: &Path) -> Result<()> {
    match std::fs::remove_file(tensor_conversion_receipt_path(output)) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}
