//! Independent verification of tensor bytes emitted by the streaming GGUF
//! converter.
//!
//! The streaming orchestrator records hashes while it owns the authoritative
//! post-bake F32 values and exact payload bytes. This module reopens the
//! finalized artifact and proves that its tensor directory and payload regions
//! reproduce those observations before any lineage receipt can refer to them.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, bail};
use mlx_native::gguf::GgufFile;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::core::provenance::tensor_execution::{ArtifactEvidence, logical_f32_sha256};
use crate::input::integrity::{VerifiedSourceManifest, verify_conversion_manifest};
use crate::intelligence::dynamic_allocator::TensorAllocationUnit;
use crate::intelligence::dynamic_allocator::producer::{
    NonVariableDisposition, TensorPartitionManifest, VerifiedSourceTensorInventory,
    validate_tensor_partition,
};
use crate::intelligence::measured_auto_quant::SourceIdentity;

use super::orchestrator::TensorWriteEvidence;
use super::source_reader::HfTensor;
use super::{
    arch::bake::{self, BakeOp},
    source_reader::SourceError,
};

mod receipt;
#[cfg(test)]
pub(crate) use receipt::stored_tensor_conversion_receipt_sha256;
pub(crate) use receipt::{
    MAX_STORED_TENSOR_CONVERSION_RECEIPT_BYTES, StoredTensorConversionReceipt,
    VerifiedSourceToStoredConversion, build_verified_source_to_stored_conversion,
    clear_stale_tensor_conversion_receipt, prepare_tensor_conversion_receipt,
    promote_tensor_conversion_receipt, tensor_conversion_receipt_path,
    validate_stored_tensor_conversion_receipt,
};

pub const STORED_TENSOR_CONVERSION_RECEIPT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversionSourceDisposition {
    Variable,
    Fixed,
    Protected,
    Excluded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MaterializedSourceTensorEvidence {
    pub(crate) tensor_name: String,
    pub(crate) shape_outermost_first: Vec<u64>,
    pub(crate) source_dtype: String,
    pub(crate) artifact_id: String,
    pub(crate) absolute_payload_offset: u64,
    pub(crate) payload_bytes: u64,
    pub(crate) payload_sha256: String,
    pub(crate) decoded_f32_bytes_sha256: String,
    pub(crate) decoded_logical_f32_sha256: String,
    pub(crate) disposition: ConversionSourceDisposition,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExcludedSourceTensorEvidence {
    pub tensor_name: String,
    pub source_tensor_sha256: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Qwen35DenseBakeOperation {
    AddOne,
    NegExp,
    ReorderVHeads {
        num_k_heads: u64,
        num_v_per_k: u64,
        head_dim: u64,
        slice: Option<(u64, u64)>,
    },
    ReorderVHeadsPerRow {
        row_count: u64,
        num_k_heads: u64,
        num_v_per_k: u64,
        head_dim_in_row: u64,
    },
    Squeeze {
        axis: u32,
    },
}

/// One actual intermediate produced by the dense-Qwen conversion bake path.
/// Sequence operations are flattened into their exact left-to-right steps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Qwen35DenseBakeStepEvidence {
    pub step_index: u32,
    pub operation: Qwen35DenseBakeOperation,
    pub input_shape_outermost_first: Vec<u64>,
    pub output_shape_outermost_first: Vec<u64>,
    pub input_f32_bytes_sha256: String,
    pub input_logical_f32_sha256: String,
    pub output_f32_bytes_sha256: String,
    pub output_logical_f32_sha256: String,
    pub step_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Qwen35DenseBakeResult {
    pub data: Vec<f32>,
    pub output_shape_outermost_first: Vec<u64>,
    pub steps: Vec<Qwen35DenseBakeStepEvidence>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MaterializedDirectTensorEvidence {
    pub(crate) data: Vec<f32>,
    pub(crate) record: MaterializedDirectTensorRecord,
}

/// Hash-only materialization record retained after the per-tensor F32 buffer
/// has been streamed and dropped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MaterializedDirectTensorRecord {
    pub(crate) plan_index: usize,
    pub(crate) gguf_tensor_name: String,
    pub(crate) output_shape_outermost_first: Vec<u64>,
    pub(crate) source: MaterializedSourceTensorEvidence,
    pub(crate) bake_steps: Vec<Qwen35DenseBakeStepEvidence>,
}

#[derive(Serialize)]
struct Qwen35DenseBakeStepHashView<'a> {
    step_index: u32,
    operation: &'a Qwen35DenseBakeOperation,
    input_shape_outermost_first: &'a [u64],
    output_shape_outermost_first: &'a [u64],
    input_f32_bytes_sha256: &'a str,
    input_logical_f32_sha256: &'a str,
    output_f32_bytes_sha256: &'a str,
    output_logical_f32_sha256: &'a str,
}

fn f32_bytes_sha256(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    hex::encode(hasher.finalize())
}

fn bake_step_hash(step: &Qwen35DenseBakeStepEvidence) -> Result<String> {
    let view = Qwen35DenseBakeStepHashView {
        step_index: step.step_index,
        operation: &step.operation,
        input_shape_outermost_first: &step.input_shape_outermost_first,
        output_shape_outermost_first: &step.output_shape_outermost_first,
        input_f32_bytes_sha256: &step.input_f32_bytes_sha256,
        input_logical_f32_sha256: &step.input_logical_f32_sha256,
        output_f32_bytes_sha256: &step.output_f32_bytes_sha256,
        output_logical_f32_sha256: &step.output_logical_f32_sha256,
    };
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(&view)?)))
}

fn checked_u64(value: usize, label: &str) -> Result<u64> {
    u64::try_from(value).with_context(|| format!("{label} is not representable"))
}

fn trace_operation(op: &BakeOp, shape: &[u64]) -> Result<(Qwen35DenseBakeOperation, Vec<u64>)> {
    let same_shape = || shape.to_vec();
    Ok(match op {
        BakeOp::AddOne => (Qwen35DenseBakeOperation::AddOne, same_shape()),
        BakeOp::NegExp => (Qwen35DenseBakeOperation::NegExp, same_shape()),
        BakeOp::ReorderVHeads {
            num_k_heads,
            num_v_per_k,
            head_dim,
            slice,
        } => (
            Qwen35DenseBakeOperation::ReorderVHeads {
                num_k_heads: checked_u64(*num_k_heads, "num_k_heads")?,
                num_v_per_k: checked_u64(*num_v_per_k, "num_v_per_k")?,
                head_dim: checked_u64(*head_dim, "head_dim")?,
                slice: slice
                    .as_ref()
                    .map(|range| -> Result<(u64, u64)> {
                        Ok((
                            checked_u64(range.start, "slice start")?,
                            checked_u64(range.end, "slice end")?,
                        ))
                    })
                    .transpose()?,
            },
            same_shape(),
        ),
        BakeOp::ReorderVHeadsPerRow {
            row_count,
            num_k_heads,
            num_v_per_k,
            head_dim_in_row,
        } => (
            Qwen35DenseBakeOperation::ReorderVHeadsPerRow {
                row_count: checked_u64(*row_count, "row_count")?,
                num_k_heads: checked_u64(*num_k_heads, "num_k_heads")?,
                num_v_per_k: checked_u64(*num_v_per_k, "num_v_per_k")?,
                head_dim_in_row: checked_u64(*head_dim_in_row, "head_dim_in_row")?,
            },
            same_shape(),
        ),
        BakeOp::Squeeze => {
            let singleton_axes: Vec<_> = shape
                .iter()
                .enumerate()
                .filter_map(|(axis, dimension)| (*dimension == 1).then_some(axis))
                .collect();
            if singleton_axes.len() != 1 {
                bail!(
                    "dense Qwen squeeze requires exactly one singleton axis, found {:?}",
                    singleton_axes
                );
            }
            let axis = singleton_axes[0];
            let mut output_shape = shape.to_vec();
            output_shape.remove(axis);
            (
                Qwen35DenseBakeOperation::Squeeze {
                    axis: u32::try_from(axis).context("squeeze axis is not representable")?,
                },
                output_shape,
            )
        }
        BakeOp::Sequence(_) => bail!("nested Sequence must be flattened before tracing"),
        unsupported => bail!("bake operation {unsupported} is outside dense Qwen3.8 D2b scope"),
    })
}

fn flatten_bake_ops<'a>(op: &'a BakeOp, output: &mut Vec<&'a BakeOp>) {
    match op {
        BakeOp::Sequence(steps) => {
            for step in steps {
                flatten_bake_ops(step, output);
            }
        }
        step => output.push(step),
    }
}

/// Execute the production bake implementation while retaining every exact
/// intermediate that the source-to-stored receipt must bind. This entrypoint
/// is deliberately limited to the dense Qwen3.8 vocabulary accepted by D2b.
pub(crate) fn apply_qwen35_dense_bake_with_evidence(
    mut data: Vec<f32>,
    input_shape_outermost_first: &[u64],
    expected_output_shape_outermost_first: &[u64],
    op: &BakeOp,
) -> Result<Qwen35DenseBakeResult> {
    let mut shape = input_shape_outermost_first.to_vec();
    let mut operations = Vec::new();
    flatten_bake_ops(op, &mut operations);
    if operations.is_empty() {
        bail!("dense Qwen bake sequence is empty");
    }
    let mut steps = Vec::with_capacity(operations.len());
    for (step_index, operation) in operations.into_iter().enumerate() {
        let input = data;
        let input_shape = shape;
        let (typed_operation, output_shape) = trace_operation(operation, &input_shape)?;
        let input_f32_bytes_sha256 = f32_bytes_sha256(&input);
        let input_logical_f32_sha256 = logical_f32_sha256(&input_shape, &input)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let output = bake::apply_bake_op(input, operation).map_err(|error| {
            anyhow::anyhow!(SourceError::Safetensors(error.to_string()).to_string())
        })?;
        let mut evidence = Qwen35DenseBakeStepEvidence {
            step_index: u32::try_from(step_index)
                .context("bake step index is not representable")?,
            operation: typed_operation,
            input_shape_outermost_first: input_shape.clone(),
            output_shape_outermost_first: output_shape.clone(),
            input_f32_bytes_sha256,
            input_logical_f32_sha256,
            output_f32_bytes_sha256: f32_bytes_sha256(&output),
            output_logical_f32_sha256: logical_f32_sha256(&output_shape, &output)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?,
            step_sha256: String::new(),
        };
        evidence.step_sha256 = bake_step_hash(&evidence)?;
        data = output;
        shape = output_shape;
        steps.push(evidence);
    }
    if shape != expected_output_shape_outermost_first {
        bail!(
            "dense Qwen bake output shape {:?} != planned {:?}",
            shape,
            expected_output_shape_outermost_first
        );
    }
    Ok(Qwen35DenseBakeResult {
        data,
        output_shape_outermost_first: shape,
        steps,
    })
}

/// Source/partition authority for the explicit evidence-producing conversion
/// path. It can only be constructed from the opaque verified source inventory.
#[derive(Debug, Clone)]
pub struct VerifiedConversionEvidenceContext {
    source: SourceIdentity,
    verified_source_manifest_sha256: String,
    source_inventory_manifest_sha256: String,
    tensor_partition_manifest_sha256: String,
    records: BTreeMap<String, crate::intelligence::dynamic_allocator::producer::SourceTensorRecord>,
    dispositions: BTreeMap<String, ConversionSourceDisposition>,
    excluded: Vec<ExcludedSourceTensorEvidence>,
}

impl VerifiedConversionEvidenceContext {
    pub fn new(
        inventory: &VerifiedSourceTensorInventory,
        partition: &TensorPartitionManifest,
        units: &[TensorAllocationUnit],
    ) -> Result<Self> {
        validate_tensor_partition(partition, inventory, units)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let records: BTreeMap<_, _> = inventory
            .manifest()
            .tensors
            .iter()
            .cloned()
            .map(|record| (record.name.clone(), record))
            .collect();
        let mut dispositions = BTreeMap::new();
        let mut excluded = Vec::new();
        for unit in &partition.variable_units {
            for member in &unit.members {
                if dispositions
                    .insert(member.name.clone(), ConversionSourceDisposition::Variable)
                    .is_some()
                {
                    bail!("duplicate variable source tensor {}", member.name);
                }
            }
        }
        for tensor in &partition.non_variable_tensors {
            let disposition = match tensor.disposition {
                NonVariableDisposition::Fixed => ConversionSourceDisposition::Fixed,
                NonVariableDisposition::Protected => ConversionSourceDisposition::Protected,
                NonVariableDisposition::Excluded => ConversionSourceDisposition::Excluded,
            };
            if dispositions
                .insert(tensor.source.name.clone(), disposition)
                .is_some()
            {
                bail!(
                    "duplicate non-variable source tensor {}",
                    tensor.source.name
                );
            }
            if tensor.disposition == NonVariableDisposition::Excluded {
                excluded.push(ExcludedSourceTensorEvidence {
                    tensor_name: tensor.source.name.clone(),
                    source_tensor_sha256: tensor.source.source_tensor_sha256.clone(),
                    reason: tensor.reason.clone(),
                });
            }
        }
        if records.len() != dispositions.len() || records.keys().ne(dispositions.keys()) {
            bail!("source inventory and tensor partition do not have exact name coverage");
        }
        excluded.sort_by(|left, right| left.tensor_name.cmp(&right.tensor_name));
        Ok(Self {
            source: inventory.manifest().source.clone(),
            verified_source_manifest_sha256: inventory
                .manifest()
                .verified_source_manifest_sha256
                .clone(),
            source_inventory_manifest_sha256: inventory.manifest().manifest_sha256.clone(),
            tensor_partition_manifest_sha256: partition.manifest_sha256.clone(),
            records,
            dispositions,
            excluded,
        })
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        source: SourceIdentity,
        verified_source_manifest_sha256: String,
        records: Vec<crate::intelligence::dynamic_allocator::producer::SourceTensorRecord>,
        dispositions: Vec<(String, ConversionSourceDisposition)>,
    ) -> Self {
        Self {
            source,
            verified_source_manifest_sha256,
            source_inventory_manifest_sha256: "a".repeat(64),
            tensor_partition_manifest_sha256: "b".repeat(64),
            records: records
                .into_iter()
                .map(|record| (record.name.clone(), record))
                .collect(),
            dispositions: dispositions.into_iter().collect(),
            excluded: Vec::new(),
        }
    }

    pub fn source(&self) -> &SourceIdentity {
        &self.source
    }

    pub fn verified_source_manifest_sha256(&self) -> &str {
        &self.verified_source_manifest_sha256
    }

    pub fn source_inventory_manifest_sha256(&self) -> &str {
        &self.source_inventory_manifest_sha256
    }

    pub fn tensor_partition_manifest_sha256(&self) -> &str {
        &self.tensor_partition_manifest_sha256
    }

    pub fn excluded_sources(&self) -> &[ExcludedSourceTensorEvidence] {
        &self.excluded
    }

    /// Reconcile the architecture mapper with the authoritative D1 source
    /// partition. Dense Qwen evidence admits one stored lineage per
    /// non-excluded source tensor and requires every excluded source to be an
    /// explicit mapper drop. MTP tensors therefore remain stored as
    /// fixed/protected evidence when the partition says so; they never vanish
    /// under a broad "non-base" rule.
    pub(crate) fn validate_dense_qwen_source_coverage<'a>(
        &self,
        mapped_source_names: impl IntoIterator<Item = &'a str>,
        dropped_source_names: &[String],
    ) -> Result<()> {
        let mut mapped_counts = BTreeMap::<&str, usize>::new();
        for name in mapped_source_names {
            let count = mapped_counts.entry(name).or_default();
            *count = count
                .checked_add(1)
                .context("mapped source occurrence count overflow")?;
        }
        let dropped: BTreeSet<_> = dropped_source_names.iter().map(String::as_str).collect();
        if dropped.len() != dropped_source_names.len() {
            bail!("dense Qwen mapper drop list contains duplicates");
        }
        for (name, disposition) in &self.dispositions {
            let mapped_count = mapped_counts.remove(name.as_str()).unwrap_or(0);
            let was_dropped = dropped.contains(name.as_str());
            if name.starts_with("mtp.")
                && !matches!(
                    disposition,
                    ConversionSourceDisposition::Fixed | ConversionSourceDisposition::Protected
                )
            {
                bail!("MTP source tensor {name} must be fixed or protected in D2b");
            }
            match disposition {
                ConversionSourceDisposition::Excluded => {
                    if mapped_count != 0 || !was_dropped {
                        bail!(
                            "excluded source tensor {name} must be dropped exactly once by the mapper"
                        );
                    }
                }
                ConversionSourceDisposition::Variable
                | ConversionSourceDisposition::Fixed
                | ConversionSourceDisposition::Protected => {
                    if mapped_count != 1 || was_dropped {
                        bail!(
                            "source tensor {name} must map to exactly one dense-Qwen stored lineage"
                        );
                    }
                }
            }
        }
        if let Some(name) = mapped_counts.keys().next() {
            bail!("mapped source tensor {name} is absent from the D1 partition");
        }
        if let Some(name) = dropped
            .iter()
            .find(|name| !self.dispositions.contains_key(**name))
        {
            bail!("dropped source tensor {name} is absent from the D1 partition");
        }
        Ok(())
    }

    pub(crate) fn verify_materialized_source(
        &self,
        tensor: &HfTensor,
    ) -> Result<MaterializedSourceTensorEvidence> {
        let record = self.records.get(&tensor.name).ok_or_else(|| {
            anyhow::anyhow!(
                "source tensor {} is not in the verified inventory",
                tensor.name
            )
        })?;
        let raw = tensor.raw_source.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "source tensor {} was not materialized with evidence",
                tensor.name
            )
        })?;
        let source_dtype = format!("{:?}", tensor.source_dtype);
        if !matches!(source_dtype.as_str(), "F16" | "BF16" | "F32") {
            bail!(
                "source tensor {} dtype {} is outside dense Qwen3.8 D2b scope",
                tensor.name,
                source_dtype
            );
        }
        if tensor.shape != record.source_shape
            || source_dtype != record.source_dtype
            || raw.byte_len != record.source_byte_len
            || raw.sha256 != record.source_tensor_sha256
        {
            bail!(
                "source tensor {} changed from the verified inventory",
                tensor.name
            );
        }
        let shape_outermost_first: Vec<u64> = tensor
            .shape
            .iter()
            .map(|dimension| u64::try_from(*dimension))
            .collect::<std::result::Result<_, _>>()
            .context("source tensor dimension is not representable")?;
        let mut raw_f32 = Sha256::new();
        for value in &tensor.data {
            raw_f32.update(value.to_bits().to_le_bytes());
        }
        Ok(MaterializedSourceTensorEvidence {
            tensor_name: tensor.name.clone(),
            shape_outermost_first: shape_outermost_first.clone(),
            source_dtype,
            artifact_id: raw.artifact_id.clone(),
            absolute_payload_offset: raw.absolute_byte_offset,
            payload_bytes: raw.byte_len,
            payload_sha256: raw.sha256.clone(),
            decoded_f32_bytes_sha256: hex::encode(raw_f32.finalize()),
            decoded_logical_f32_sha256: logical_f32_sha256(&shape_outermost_first, &tensor.data)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?,
            disposition: *self.dispositions.get(&tensor.name).ok_or_else(|| {
                anyhow::anyhow!("source tensor {} has no disposition", tensor.name)
            })?,
        })
    }
}

/// Rehash the exact safetensors artifacts selected by the opaque verified
/// source manifest. This closes the receipt's artifact-region authority over
/// the raw tensor slices captured by [`HfModelSource`].
pub(crate) fn verify_source_weight_artifacts(
    model_dir: &Path,
    verified_source: &VerifiedSourceManifest,
    context: &VerifiedConversionEvidenceContext,
) -> Result<Vec<ArtifactEvidence>> {
    if verified_source.repo() != context.source.model_id
        || verified_source.revision() != context.source.revision
    {
        bail!("verified source repository/revision differs from conversion context");
    }
    let manifest_sha256 = hex::encode(Sha256::digest(serde_json::to_vec(verified_source)?));
    if manifest_sha256 != context.verified_source_manifest_sha256 {
        bail!("verified source manifest differs from source inventory authority");
    }
    let records: BTreeMap<_, _> = verified_source
        .records()
        .iter()
        .map(|record| (record.filename.as_str(), record))
        .collect();
    let mut artifacts = Vec::with_capacity(verified_source.required_weight_shards().len());
    for shard_name in verified_source.required_weight_shards() {
        let record = records
            .get(shard_name.as_str())
            .ok_or_else(|| anyhow::anyhow!("verified source is missing shard {shard_name}"))?;
        let path = model_dir.join(shard_name);
        let byte_len = std::fs::metadata(&path)?.len();
        if byte_len != record.bytes {
            bail!("source shard {shard_name} byte length changed after verification");
        }
        let sha256 = sha256_reader(File::open(&path)?)?;
        if record.sha256.as_deref() != Some(sha256.as_str()) {
            bail!("source shard {shard_name} hash changed after verification");
        }
        artifacts.push(ArtifactEvidence {
            artifact_id: shard_name.clone(),
            role: "source_safetensors_shard".into(),
            byte_len,
            sha256,
        });
    }
    artifacts.sort_by(|left, right| left.artifact_id.cmp(&right.artifact_id));
    Ok(artifacts)
}

/// Reverify every input recorded by the opaque source manifest after
/// conversion. D2b still scopes non-tensor GGUF metadata as unauthenticated;
/// this check catches ordinary mutation of recorded config/tokenizer/template
/// files without claiming an immutable metadata snapshot.
pub(crate) fn verify_complete_conversion_source(
    model_dir: &Path,
    verified_source: &VerifiedSourceManifest,
    context: &VerifiedConversionEvidenceContext,
) -> Result<()> {
    if verified_source.repo() != context.source.model_id
        || verified_source.revision() != context.source.revision
    {
        bail!("verified source repository/revision differs from conversion context");
    }
    let reverified = verify_conversion_manifest(
        verified_source.repo(),
        verified_source.revision(),
        model_dir,
        verified_source.records().to_vec(),
    )?;
    let expected_sha256 = hex::encode(Sha256::digest(serde_json::to_vec(verified_source)?));
    let actual_sha256 = hex::encode(Sha256::digest(serde_json::to_vec(&reverified)?));
    if expected_sha256 != context.verified_source_manifest_sha256
        || actual_sha256 != expected_sha256
    {
        bail!("conversion-relevant source manifest changed during evidence production");
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VerifiedStoredTensor {
    pub(crate) plan_index: usize,
    pub(crate) tensor_name: String,
    pub(crate) shape_outermost_first: Vec<u64>,
    pub(crate) ggml_wire_type_id: u32,
    pub(crate) ggml_type_name: String,
    pub(crate) absolute_payload_offset: u64,
    pub(crate) payload_bytes: u64,
    pub(crate) payload_sha256: String,
    pub(crate) converted_f32_bytes_sha256: String,
    pub(crate) converted_logical_f32_sha256: String,
    pub(crate) f16_roundtrip_f32_bytes_sha256: Option<String>,
    pub(crate) f16_roundtrip_logical_f32_sha256: Option<String>,
    pub(crate) stored_f32_bytes_sha256: String,
    pub(crate) stored_logical_f32_sha256: String,
}

/// Opaque result of independently reopening and hashing the finalized GGUF.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VerifiedStoredTensorCatalog {
    artifact_bytes: u64,
    artifact_sha256: String,
    tensors: Vec<VerifiedStoredTensor>,
}

impl VerifiedStoredTensorCatalog {
    pub fn artifact_bytes(&self) -> u64 {
        self.artifact_bytes
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    pub fn tensors(&self) -> &[VerifiedStoredTensor] {
        &self.tensors
    }
}

fn sha256_reader(mut reader: impl Read) -> Result<String> {
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn sha256_file_handle(file: &File) -> Result<String> {
    let mut reader = file.try_clone()?;
    reader.seek(SeekFrom::Start(0))?;
    sha256_reader(reader)
}

/// Rehash one already-open artifact identity against an evidence record.
///
/// Runtime evidence keeps this file identity open so a later pathname swap
/// cannot redirect the loader to bytes other than those admitted by D2b.
pub(crate) fn verify_open_artifact_identity(
    file: &File,
    expected: &ArtifactEvidence,
) -> Result<()> {
    let byte_len = file.metadata()?.len();
    if byte_len != expected.byte_len || sha256_file_handle(file)? != expected.sha256 {
        bail!("open artifact identity differs from its evidence record");
    }
    Ok(())
}

/// Reopen `artifact` and compare every physical tensor directory entry and
/// byte range with the evidence captured during conversion.
#[cfg(test)]
pub(crate) fn verify_written_tensor_evidence(
    artifact: &Path,
    claimed: &[TensorWriteEvidence],
) -> Result<VerifiedStoredTensorCatalog> {
    let file = File::open(artifact)?;
    verify_written_tensor_evidence_file(&file, claimed)
}

/// Verify an artifact through one already-open file identity. The GGUF
/// directory, exact payload bytes, decoded logical values, and final whole
/// artifact hash are all read from this inode; no mutable pathname is reopened.
pub(crate) fn verify_written_tensor_evidence_file(
    artifact: &File,
    claimed: &[TensorWriteEvidence],
) -> Result<VerifiedStoredTensorCatalog> {
    let gguf = GgufFile::from_file(artifact.try_clone()?)
        .map_err(|error| anyhow::anyhow!("parse finalized GGUF: {error}"))?;
    verify_written_tensor_evidence_open(artifact, &gguf, claimed)
}

/// Verify through the exact parser instance that a later runtime loader will
/// retain. This avoids authenticating one parse while returning cached
/// directory metadata from another.
pub(crate) fn verify_written_tensor_evidence_open(
    artifact: &File,
    gguf: &GgufFile,
    claimed: &[TensorWriteEvidence],
) -> Result<VerifiedStoredTensorCatalog> {
    if claimed.is_empty() {
        bail!("tensor write evidence is empty");
    }
    if gguf.tensor_count() != claimed.len() {
        bail!(
            "finalized GGUF tensor count {} != write evidence count {}",
            gguf.tensor_count(),
            claimed.len()
        );
    }

    let artifact_bytes = artifact.metadata()?.len();
    let mut seen = BTreeSet::new();
    let mut tensors = Vec::with_capacity(claimed.len());
    let mut previous_relative_end = 0_u64;

    for (expected_plan_index, evidence) in claimed.iter().enumerate() {
        if evidence.plan_index != expected_plan_index {
            bail!(
                "tensor write evidence plan index {} != canonical position {}",
                evidence.plan_index,
                expected_plan_index
            );
        }
        if !seen.insert(evidence.tensor_name.as_str()) {
            bail!(
                "duplicate tensor write evidence for {}",
                evidence.tensor_name
            );
        }
        let info = gguf.tensor_info(&evidence.tensor_name).ok_or_else(|| {
            anyhow::anyhow!("finalized GGUF is missing tensor {}", evidence.tensor_name)
        })?;
        let mut expected_shape = evidence.dims_gguf_innermost_first.clone();
        expected_shape.reverse();
        let actual_shape: Vec<u64> = info
            .shape
            .iter()
            .map(|dimension| u64::try_from(*dimension))
            .collect::<std::result::Result<_, _>>()
            .context("GGUF tensor dimension is not representable")?;
        if actual_shape != expected_shape {
            bail!(
                "tensor {} shape {:?} != write evidence {:?}",
                evidence.tensor_name,
                actual_shape,
                expected_shape
            );
        }
        let actual_type_name = format!("{:?}", info.ggml_type).to_ascii_lowercase();
        if actual_type_name != evidence.ggml_type.name() {
            bail!(
                "tensor {} type {} != write evidence {}",
                evidence.tensor_name,
                actual_type_name,
                evidence.ggml_type.name()
            );
        }
        let info_bytes = u64::try_from(info.byte_len).context("GGUF tensor size overflow")?;
        if info_bytes != evidence.payload_bytes || info.offset != evidence.relative_payload_offset {
            bail!(
                "tensor {} payload region ({}, {}) != write evidence ({}, {})",
                evidence.tensor_name,
                info.offset,
                info_bytes,
                evidence.relative_payload_offset,
                evidence.payload_bytes
            );
        }
        if info.offset < previous_relative_end {
            bail!(
                "tensor {} payload overlaps or precedes the prior plan entry",
                evidence.tensor_name
            );
        }
        previous_relative_end = info
            .offset
            .checked_add(info_bytes)
            .context("GGUF tensor relative end offset overflow")?;
        let absolute_payload_offset = gguf
            .tensor_data_offset()
            .checked_add(info.offset)
            .context("GGUF tensor absolute offset overflow")?;
        let end = absolute_payload_offset
            .checked_add(info_bytes)
            .context("GGUF tensor end offset overflow")?;
        if end > artifact_bytes {
            bail!(
                "tensor {} payload ends beyond the artifact",
                evidence.tensor_name
            );
        }
        let payload = gguf
            .read_tensor_bytes_host(&evidence.tensor_name)
            .map_err(|error| anyhow::anyhow!("read finalized GGUF payload: {error}"))?;
        let payload_sha256 = hex::encode(Sha256::digest(&payload));
        if payload_sha256 != evidence.payload_sha256 {
            bail!(
                "tensor {} payload hash does not reproduce from the finalized GGUF",
                evidence.tensor_name
            );
        }
        let stored_f32 = gguf
            .read_tensor_f32_host(&evidence.tensor_name)
            .map_err(|error| anyhow::anyhow!("decode finalized GGUF tensor: {error}"))?;
        let stored_f32_bytes_sha256 = f32_bytes_sha256(&stored_f32);
        let stored_logical_f32_sha256 = logical_f32_sha256(&actual_shape, &stored_f32)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        tensors.push(VerifiedStoredTensor {
            plan_index: evidence.plan_index,
            tensor_name: evidence.tensor_name.clone(),
            shape_outermost_first: actual_shape,
            ggml_wire_type_id: evidence.ggml_type as u32,
            ggml_type_name: actual_type_name,
            absolute_payload_offset,
            payload_bytes: info_bytes,
            payload_sha256,
            converted_f32_bytes_sha256: evidence.converted_f32_bytes_sha256.clone(),
            converted_logical_f32_sha256: evidence.converted_logical_f32_sha256.clone(),
            f16_roundtrip_f32_bytes_sha256: evidence.f16_roundtrip_f32_bytes_sha256.clone(),
            f16_roundtrip_logical_f32_sha256: evidence.f16_roundtrip_logical_f32_sha256.clone(),
            stored_f32_bytes_sha256,
            stored_logical_f32_sha256,
        });
    }

    let final_artifact_bytes = artifact.metadata()?.len();
    if final_artifact_bytes != artifact_bytes {
        bail!("finalized GGUF length changed while its tensor evidence was verified");
    }
    let artifact_sha256 = sha256_file_handle(artifact)?;

    Ok(VerifiedStoredTensorCatalog {
        artifact_bytes,
        artifact_sha256,
        tensors,
    })
}

pub(crate) fn verify_promoted_stored_catalog(
    artifact: &Path,
    catalog: &VerifiedStoredTensorCatalog,
) -> Result<()> {
    if std::fs::metadata(artifact)?.len() != catalog.artifact_bytes
        || sha256_reader(File::open(artifact)?)? != catalog.artifact_sha256
    {
        bail!("promoted GGUF does not match its verified temporary artifact");
    }
    Ok(())
}

/// Join the three independently captured authorities: source/bake
/// materialization, the streaming writer, and the reopened GGUF directory.
/// No receipt may be assembled unless every hash and shape is continuous.
pub(crate) fn verify_source_to_stored_continuity(
    materialized: &[MaterializedDirectTensorRecord],
    writes: &[TensorWriteEvidence],
    stored: &VerifiedStoredTensorCatalog,
) -> Result<()> {
    if materialized.len() != writes.len() || writes.len() != stored.tensors.len() {
        bail!(
            "source/write/stored evidence counts differ ({}/{}/{})",
            materialized.len(),
            writes.len(),
            stored.tensors.len()
        );
    }
    for (index, ((materialized, write), stored)) in materialized
        .iter()
        .zip(writes)
        .zip(&stored.tensors)
        .enumerate()
    {
        if materialized.plan_index != index
            || write.plan_index != index
            || stored.plan_index != index
        {
            bail!("tensor evidence plan index is discontinuous at {index}");
        }
        if materialized.gguf_tensor_name != write.tensor_name
            || write.tensor_name != stored.tensor_name
            || materialized.output_shape_outermost_first != stored.shape_outermost_first
        {
            bail!("tensor identity/shape is discontinuous at plan index {index}");
        }
        let (final_bytes, final_logical) = if let Some(first) = materialized.bake_steps.first() {
            if first.input_shape_outermost_first != materialized.source.shape_outermost_first
                || first.input_f32_bytes_sha256 != materialized.source.decoded_f32_bytes_sha256
                || first.input_logical_f32_sha256 != materialized.source.decoded_logical_f32_sha256
            {
                bail!(
                    "tensor {} source does not feed its first bake step",
                    materialized.source.tensor_name
                );
            }
            for pair in materialized.bake_steps.windows(2) {
                if pair[0].output_shape_outermost_first != pair[1].input_shape_outermost_first
                    || pair[0].output_f32_bytes_sha256 != pair[1].input_f32_bytes_sha256
                    || pair[0].output_logical_f32_sha256 != pair[1].input_logical_f32_sha256
                {
                    bail!(
                        "tensor {} bake steps are discontinuous",
                        materialized.source.tensor_name
                    );
                }
            }
            let last = materialized
                .bake_steps
                .last()
                .expect("first bake step exists");
            if last.output_shape_outermost_first != materialized.output_shape_outermost_first {
                bail!(
                    "tensor {} final bake shape is discontinuous",
                    materialized.source.tensor_name
                );
            }
            (
                last.output_f32_bytes_sha256.as_str(),
                last.output_logical_f32_sha256.as_str(),
            )
        } else {
            if materialized.source.shape_outermost_first
                != materialized.output_shape_outermost_first
            {
                bail!(
                    "tensor {} direct shape is discontinuous",
                    materialized.source.tensor_name
                );
            }
            (
                materialized.source.decoded_f32_bytes_sha256.as_str(),
                materialized.source.decoded_logical_f32_sha256.as_str(),
            )
        };
        if final_bytes != write.converted_f32_bytes_sha256
            || final_logical != write.converted_logical_f32_sha256
            || write.converted_f32_bytes_sha256 != stored.converted_f32_bytes_sha256
            || write.converted_logical_f32_sha256 != stored.converted_logical_f32_sha256
        {
            bail!(
                "tensor {} converted values do not feed the writer",
                materialized.source.tensor_name
            );
        }
        if write.f16_roundtrip_f32_bytes_sha256 != stored.f16_roundtrip_f32_bytes_sha256
            || write.f16_roundtrip_logical_f32_sha256 != stored.f16_roundtrip_logical_f32_sha256
            || write.payload_sha256 != stored.payload_sha256
            || write.payload_bytes != stored.payload_bytes
        {
            bail!(
                "tensor {} writer/stored evidence diverges",
                write.tensor_name
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::{Seek, SeekFrom, Write};

    use crate::backends::gguf::writer::GgufWriter;
    use crate::core::integrity::ShardIntegrity;
    use crate::quantize::ggml_quants::GgmlType;

    use super::*;

    fn write_fixture() -> (tempfile::NamedTempFile, TensorWriteEvidence) {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let payload: Vec<u8> = (0_u8..18).collect();
        let relative_payload_offset;
        {
            let mut writer = GgufWriter::new(file.as_file_mut());
            writer.write_header(1, 0).unwrap();
            writer
                .reserve_tensor_info("weight", &[32, 1], GgmlType::Q4_0)
                .unwrap();
            writer.pad_to_alignment().unwrap();
            writer.stream_tensor_payload(0, &payload).unwrap();
            relative_payload_offset = writer.tensor_offsets()[0].unwrap();
            writer.finalize().unwrap();
        }
        file.as_file_mut().sync_all().unwrap();
        (
            file,
            TensorWriteEvidence {
                plan_index: 0,
                tensor_name: "weight".into(),
                dims_gguf_innermost_first: vec![32, 1],
                ggml_type: GgmlType::Q4_0,
                converted_f32_bytes_sha256: hex::encode(Sha256::digest(b"converted")),
                converted_logical_f32_sha256: hex::encode(Sha256::digest(b"logical")),
                f16_roundtrip_f32_bytes_sha256: Some(hex::encode(Sha256::digest(b"roundtrip"))),
                f16_roundtrip_logical_f32_sha256: Some(hex::encode(Sha256::digest(
                    b"roundtrip-logical",
                ))),
                payload_sha256: hex::encode(Sha256::digest(&payload)),
                payload_bytes: payload.len() as u64,
                relative_payload_offset,
            },
        )
    }

    #[test]
    fn finalized_payload_is_independently_reopened_and_hashed() {
        let (file, evidence) = write_fixture();
        let verified = verify_written_tensor_evidence(file.path(), &[evidence.clone()]).unwrap();
        assert_eq!(verified.tensors().len(), 1);
        assert_eq!(
            verified.tensors()[0].payload_sha256,
            evidence.payload_sha256
        );
        assert_eq!(verified.tensors()[0].shape_outermost_first, vec![1, 32]);
        assert_eq!(verified.tensors()[0].ggml_wire_type_id, 2);
    }

    #[test]
    fn finalized_payload_mutation_fails_closed() {
        let (mut file, evidence) = write_fixture();
        let verified = verify_written_tensor_evidence(file.path(), &[evidence.clone()]).unwrap();
        file.as_file_mut()
            .seek(SeekFrom::Start(
                verified.tensors()[0].absolute_payload_offset,
            ))
            .unwrap();
        file.as_file_mut().write_all(&[0xff]).unwrap();
        file.as_file_mut().sync_all().unwrap();
        let error = verify_written_tensor_evidence(file.path(), &[evidence]).unwrap_err();
        assert!(error.to_string().contains("payload hash"));
    }

    #[test]
    fn dense_qwen_sequence_records_each_exact_intermediate() {
        let input: Vec<f32> = (0..8).map(|value| value as f32).collect();
        let result = apply_qwen35_dense_bake_with_evidence(
            input,
            &[2, 1, 4],
            &[2, 4],
            &BakeOp::Sequence(vec![
                BakeOp::Squeeze,
                BakeOp::ReorderVHeads {
                    num_k_heads: 2,
                    num_v_per_k: 2,
                    head_dim: 2,
                    slice: None,
                },
                BakeOp::NegExp,
            ]),
        )
        .unwrap();
        assert_eq!(result.output_shape_outermost_first, vec![2, 4]);
        assert_eq!(result.steps.len(), 3);
        assert_eq!(
            result.steps[0].operation,
            Qwen35DenseBakeOperation::Squeeze { axis: 1 }
        );
        assert_eq!(
            result.steps[0].output_f32_bytes_sha256,
            result.steps[1].input_f32_bytes_sha256
        );
        assert_eq!(
            result.steps[1].output_logical_f32_sha256,
            result.steps[2].input_logical_f32_sha256
        );
        assert!(result.steps.iter().all(|step| step.step_sha256.len() == 64));
    }

    #[test]
    fn dense_qwen_trace_rejects_ambiguous_or_out_of_scope_bakes() {
        let ambiguous =
            apply_qwen35_dense_bake_with_evidence(vec![1.0], &[1, 1], &[1], &BakeOp::Squeeze)
                .unwrap_err();
        assert!(ambiguous.to_string().contains("exactly one singleton"));

        let unsupported =
            apply_qwen35_dense_bake_with_evidence(vec![1.0, 2.0], &[2], &[1], &BakeOp::Slice(0..1))
                .unwrap_err();
        assert!(unsupported.to_string().contains("outside dense Qwen3.8"));
    }

    #[test]
    fn source_artifacts_are_rehashed_against_exact_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let shard = dir.path().join("model.safetensors");
        std::fs::write(&shard, b"exact-weight-bytes").unwrap();
        let shard_sha = hex::encode(Sha256::digest(b"exact-weight-bytes"));
        let verified_source = VerifiedSourceManifest::for_test_bound(
            "org/model",
            "a".repeat(40),
            vec![ShardIntegrity {
                filename: "model.safetensors".into(),
                bytes: 18,
                sha256: Some(shard_sha.clone()),
                hf_etag: shard_sha,
                is_lfs: true,
            }],
        );
        let context = VerifiedConversionEvidenceContext {
            source: SourceIdentity {
                model_id: "org/model".into(),
                revision: "a".repeat(40),
                config_sha256: "b".repeat(64),
                tensor_bundle_sha256: "c".repeat(64),
                tokenizer_bundle_sha256: "d".repeat(64),
                chat_template_sha256: "e".repeat(64),
            },
            verified_source_manifest_sha256: hex::encode(Sha256::digest(
                serde_json::to_vec(&verified_source).unwrap(),
            )),
            source_inventory_manifest_sha256: "f".repeat(64),
            tensor_partition_manifest_sha256: "1".repeat(64),
            records: BTreeMap::new(),
            dispositions: BTreeMap::new(),
            excluded: Vec::new(),
        };
        let artifacts =
            verify_source_weight_artifacts(dir.path(), &verified_source, &context).unwrap();
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].artifact_id, "model.safetensors");

        std::fs::write(&shard, b"changed-weight-byte").unwrap();
        assert!(verify_source_weight_artifacts(dir.path(), &verified_source, &context).is_err());
    }

    #[test]
    fn dense_qwen_mapper_coverage_keeps_fixed_mtp_and_drops_only_excluded() {
        let dispositions = BTreeMap::from([
            ("base.weight".into(), ConversionSourceDisposition::Variable),
            ("mtp.weight".into(), ConversionSourceDisposition::Fixed),
            (
                "model.visual.weight".into(),
                ConversionSourceDisposition::Excluded,
            ),
        ]);
        let context = VerifiedConversionEvidenceContext {
            source: SourceIdentity {
                model_id: "org/model".into(),
                revision: "a".repeat(40),
                config_sha256: "b".repeat(64),
                tensor_bundle_sha256: "c".repeat(64),
                tokenizer_bundle_sha256: "d".repeat(64),
                chat_template_sha256: "e".repeat(64),
            },
            verified_source_manifest_sha256: "f".repeat(64),
            source_inventory_manifest_sha256: "1".repeat(64),
            tensor_partition_manifest_sha256: "2".repeat(64),
            records: BTreeMap::new(),
            dispositions,
            excluded: Vec::new(),
        };
        context
            .validate_dense_qwen_source_coverage(
                ["base.weight", "mtp.weight"],
                &["model.visual.weight".into()],
            )
            .unwrap();
        let mut variable_mtp = context.clone();
        variable_mtp
            .dispositions
            .insert("mtp.weight".into(), ConversionSourceDisposition::Variable);
        assert!(
            variable_mtp
                .validate_dense_qwen_source_coverage(
                    ["base.weight", "mtp.weight"],
                    &["model.visual.weight".into()],
                )
                .unwrap_err()
                .to_string()
                .contains("must be fixed or protected")
        );
        assert!(
            context
                .validate_dense_qwen_source_coverage(
                    ["base.weight"],
                    &["mtp.weight".into(), "model.visual.weight".into()],
                )
                .unwrap_err()
                .to_string()
                .contains("mtp.weight")
        );
        assert!(
            context
                .validate_dense_qwen_source_coverage(
                    ["base.weight", "mtp.weight", "model.visual.weight"],
                    &[],
                )
                .is_err()
        );
    }
}
