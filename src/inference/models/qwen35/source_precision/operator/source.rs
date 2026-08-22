use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use std::os::unix::fs::MetadataExt;

use anyhow::{ensure, Context, Result};
use mlx_native::MlxDevice;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::core::provenance::{compute_source_bundle_sha256, SourceShard};
use crate::input::integrity::{verify_conversion_manifest, VerifiedSourceManifest};
use crate::intelligence::dynamic_allocator::producer::{
    build_tensor_partition, derive_source_tensor_inventory, NonVariableDisposition,
    NonVariableTensor, TensorPartitionManifest, VerifiedSourceTensorInventory,
};
use crate::intelligence::dynamic_allocator::TensorAllocationUnit;
use crate::intelligence::exact_teacher::{
    build_exact_teacher_reference_input, ExactTeacherReferenceInputV1, ExactTeacherTargetReceipt,
};
use crate::intelligence::measured_auto_quant::SourceIdentity;

use super::super::types::QwenSourceSnapshotLimits;
use super::super::{
    admit_qwen35_bf16_topology, open_verified_qwen_source_snapshot,
    preflight_qwen35_source_teacher_execution, preflight_qwen35_source_teacher_run_inputs_capacity,
    prepare_qwen35_source_teacher_run_inputs, run_qwen35_source_teacher,
    Qwen35SourceTeacherCapacityPreflightV1, Qwen35SourceTeacherPreparationPolicyV1,
    QwenSourceMetalUploadLimits, StructurallyBoundQwen35SourceTeacherWorkV1,
    VerifiedQwen35Bf16TopologyV1,
};
use super::corpus::build_official_prediction_plan;
use super::profile::{official_profile, OfficialEvidenceProfileV1, PROFILE_SHA256};
use super::source_manifest::{official_source_manifest, OfficialSourceManifestV1};
use super::OfficialQwen38EvaluationSplitV1;

mod execute;

pub(crate) use execute::{
    preflight_official_qwen38_source_teacher, run_official_qwen38_source_teacher,
};

const OFFICIAL_TENSOR_COUNT: usize = 1_199;
const OFFICIAL_WEIGHT_SHARD_COUNT: usize = 18;
const OFFICIAL_OUTPUT_TENSOR_COUNT: usize = 867;
const OFFICIAL_BF16_OUTPUT_TENSOR_COUNT: usize = 514;
const OFFICIAL_F32_OUTPUT_TENSOR_COUNT: usize = 353;
const OFFICIAL_PLANNED_WEIGHT_BYTES: u64 = 53_797_287_936;

/// Inputs that select bytes and destinations, never evidence semantics.
#[derive(Debug, Clone)]
pub(crate) struct OfficialQwen38SourceTeacherRequestV1 {
    pub(crate) model_dir: PathBuf,
    pub(crate) output: PathBuf,
    pub(crate) evaluation_split: OfficialQwen38EvaluationSplitV1,
}

/// Sanitized operational summary. Content/work hashes are stable; device,
/// capacity, and timing observations are deliberately process-local.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct OfficialQwen38SourceTeacherSummaryV1 {
    pub(crate) profile: String,
    pub(crate) evidence_profile_sha256: String,
    pub(crate) hf2q_version: String,
    pub(crate) hf2q_git_commit: String,
    pub(crate) mlx_native_version: &'static str,
    pub(crate) host_os: String,
    pub(crate) host_arch: &'static str,
    pub(crate) source_manifest_id: String,
    pub(crate) source_manifest_sha256: String,
    pub(crate) source: SourceIdentity,
    pub(crate) verified_source_manifest_sha256: String,
    pub(crate) source_inventory_sha256: String,
    pub(crate) source_partition_sha256: String,
    pub(crate) topology_sha256: String,
    pub(crate) dataset_partition_sha256: String,
    pub(crate) evaluation_split: crate::intelligence::calibration::DatasetSplit,
    pub(crate) calibration_corpus_sha256: String,
    pub(crate) policy_validation_corpus_sha256: String,
    pub(crate) acceptance_holdout_corpus_sha256: String,
    pub(crate) threshold_profile_sha256: Option<String>,
    pub(crate) prediction_plan_sha256: String,
    pub(crate) work_plan_sha256: String,
    pub(crate) source_tensor_count: usize,
    pub(crate) weight_shard_count: usize,
    pub(crate) output_tensor_count: usize,
    pub(crate) bf16_output_tensor_count: usize,
    pub(crate) f32_output_tensor_count: usize,
    pub(crate) example_count: usize,
    pub(crate) prediction_row_count: usize,
    pub(crate) forward_call_count: u64,
    pub(crate) input_tokens_processed: u64,
    pub(crate) output_head_evaluation_count: u64,
    pub(crate) max_cache_tokens: usize,
    pub(crate) target_artifact_bytes: u64,
    pub(crate) reference_input: ExactTeacherReferenceInputV1,
    pub(crate) metal_device: Option<OfficialMetalDeviceSummaryV1>,
    pub(crate) capacity_preflight: Option<Qwen35SourceTeacherCapacityPreflightV1>,
    pub(crate) timings: OfficialSourceTeacherTimingsV1,
    pub(crate) target_path: PathBuf,
    pub(crate) executed: bool,
    pub(crate) target_artifact_sha256: Option<String>,
    pub(crate) completion_receipt_sha256: Option<String>,
    pub(crate) structural_target_receipt: Option<ExactTeacherTargetReceipt>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub(crate) struct OfficialSourceTeacherTimingsV1 {
    pub(crate) source_authentication_ms: u64,
    pub(crate) corpus_and_prediction_plan_ms: u64,
    pub(crate) topology_and_work_preflight_ms: u64,
    pub(crate) capacity_preflight_ms: u64,
    pub(crate) prepare_weights_and_cache_ms: Option<u64>,
    pub(crate) execute_and_publish_ms: Option<u64>,
    pub(crate) total_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct OfficialMetalDeviceSummaryV1 {
    pub(crate) name: String,
    pub(crate) registry_id: u64,
    pub(crate) residency_sets_enabled: bool,
}

pub(super) struct OfficialSourceV1 {
    _materialized_source: tempfile::TempDir,
    pub(super) model_dir: PathBuf,
    source_manifest_id: String,
    source_manifest_sha256: String,
    pub(super) source: SourceIdentity,
    pub(super) verified_source: VerifiedSourceManifest,
    inventory: VerifiedSourceTensorInventory,
    units: Vec<TensorAllocationUnit>,
    partition: TensorPartitionManifest,
}

struct OfficialWorkV1 {
    summary: OfficialQwen38SourceTeacherSummaryV1,
    work: StructurallyBoundQwen35SourceTeacherWorkV1,
    upload_limits: QwenSourceMetalUploadLimits,
    preparation_policy: Qwen35SourceTeacherPreparationPolicyV1,
}

fn build_official_work(
    model_dir: &Path,
    output: &Path,
    profile: &OfficialEvidenceProfileV1,
    evaluation: OfficialQwen38EvaluationSplitV1,
) -> Result<OfficialWorkV1> {
    let source_started = Instant::now();
    let source = authenticate_official_source(model_dir, profile)?;
    let source_authentication_ms = elapsed_ms(source_started.elapsed());
    let corpus_started = Instant::now();
    let prediction = build_official_prediction_plan(&source, profile, evaluation)?;
    let corpus_and_prediction_plan_ms = elapsed_ms(corpus_started.elapsed());
    let source_inventory_sha256 = source.inventory.manifest().manifest_sha256.clone();
    let source_partition_sha256 = source.partition.manifest_sha256.clone();
    let verified_source_manifest_sha256 =
        sha256_json(&source.verified_source).context("hash verified source manifest")?;
    let source_identity = source.source.clone();
    let source_manifest_id = source.source_manifest_id.clone();
    let source_manifest_sha256 = source.source_manifest_sha256.clone();
    let source_tensor_count = source.inventory.manifest().tensors.len();
    let weight_shard_count = source.verified_source.required_weight_shards().len();
    let prediction_plan_sha256 = prediction.plan.manifest().manifest_sha256.clone();
    let reference_input = build_exact_teacher_reference_input(
        &prediction.plan,
        profile.target_limits().max_vocabulary_size,
        profile.target_limits(),
    )?;
    let topology_started = Instant::now();
    let topology = source.into_topology()?;
    let topology_sha256 = topology.topology_sha256().to_owned();
    let output_tensor_count = topology.future_tensor_count();
    let bf16_output_tensor_count = topology.future_bf16_tensor_count();
    let f32_output_tensor_count = topology.future_f32_tensor_count();
    ensure!(
        output_tensor_count == OFFICIAL_OUTPUT_TENSOR_COUNT
            && bf16_output_tensor_count == OFFICIAL_BF16_OUTPUT_TENSOR_COUNT
            && f32_output_tensor_count == OFFICIAL_F32_OUTPUT_TENSOR_COUNT
            && topology.planned_output_bytes()? == OFFICIAL_PLANNED_WEIGHT_BYTES,
        "official Qwen3.8 topology differs from the accepted output profile"
    );
    let work = preflight_qwen35_source_teacher_execution(
        topology,
        prediction.plan,
        profile.target_limits(),
        profile.run_limits(),
    )?;
    let topology_and_work_preflight_ms = elapsed_ms(topology_started.elapsed());
    let summary = OfficialQwen38SourceTeacherSummaryV1 {
        profile: profile.profile.clone(),
        evidence_profile_sha256: PROFILE_SHA256.into(),
        hf2q_version: env!("CARGO_PKG_VERSION").into(),
        hf2q_git_commit: crate::convert::receipt::require_converter_git_commit()
            .context("official source-teacher build lacks an exact git commit")?,
        mlx_native_version: "0.11.2",
        host_os: sysinfo::System::long_os_version().unwrap_or_else(|| std::env::consts::OS.into()),
        host_arch: std::env::consts::ARCH,
        source_manifest_id,
        source_manifest_sha256,
        source: source_identity,
        verified_source_manifest_sha256,
        source_inventory_sha256,
        source_partition_sha256,
        topology_sha256,
        dataset_partition_sha256: prediction.dataset_partition_sha256,
        evaluation_split: prediction.evaluation_split,
        calibration_corpus_sha256: prediction.calibration_corpus_sha256,
        policy_validation_corpus_sha256: prediction.policy_validation_corpus_sha256,
        acceptance_holdout_corpus_sha256: prediction.acceptance_holdout_corpus_sha256,
        threshold_profile_sha256: prediction.threshold_profile_sha256,
        prediction_plan_sha256,
        work_plan_sha256: work.work_plan_sha256().to_owned(),
        source_tensor_count,
        weight_shard_count,
        output_tensor_count,
        bf16_output_tensor_count,
        f32_output_tensor_count,
        example_count: work.example_count(),
        prediction_row_count: work.prediction_row_count(),
        forward_call_count: work.forward_call_count(),
        input_tokens_processed: work.input_tokens_processed(),
        output_head_evaluation_count: work.output_head_evaluation_count(),
        max_cache_tokens: work.max_cache_tokens(),
        target_artifact_bytes: work.target_artifact_bytes(),
        reference_input,
        metal_device: None,
        capacity_preflight: None,
        timings: OfficialSourceTeacherTimingsV1 {
            source_authentication_ms,
            corpus_and_prediction_plan_ms,
            topology_and_work_preflight_ms,
            ..OfficialSourceTeacherTimingsV1::default()
        },
        target_path: output.to_path_buf(),
        executed: false,
        target_artifact_sha256: None,
        completion_receipt_sha256: None,
        structural_target_receipt: None,
    };
    Ok(OfficialWorkV1 {
        summary,
        work,
        upload_limits: profile.upload_limits(),
        preparation_policy: profile.preparation_policy(),
    })
}

fn elapsed_ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn device_summary(device: &MlxDevice) -> OfficialMetalDeviceSummaryV1 {
    OfficialMetalDeviceSummaryV1 {
        name: device.name(),
        registry_id: device.registry_id(),
        residency_sets_enabled: device.residency_sets_enabled(),
    }
}

impl OfficialSourceV1 {
    pub(super) fn into_topology(self) -> Result<VerifiedQwen35Bf16TopologyV1> {
        let snapshot = open_verified_qwen_source_snapshot(
            &self.model_dir,
            &self.verified_source,
            &self.inventory,
            &self.partition,
            &self.units,
            snapshot_limits(&self.verified_source)?,
        )?;
        admit_qwen35_bf16_topology(snapshot)
    }
}

pub(super) fn authenticate_official_source(
    model_dir: &Path,
    profile: &OfficialEvidenceProfileV1,
) -> Result<OfficialSourceV1> {
    let manifest = official_source_manifest()?;
    ensure!(
        manifest.manifest_id() == profile.source.manifest_id
            && manifest.manifest_sha256() == profile.source.manifest_sha256
            && manifest.repository_id() == profile.source.repository_id
            && manifest.revision() == profile.source.revision
            && manifest.bundle_sha256() == profile.source.bundle_sha256,
        "embedded source-teacher profile differs from the source manifest"
    );
    let materialized_source = materialize_manifest_source(model_dir, &manifest)?;
    let model_dir = materialized_source.path().to_path_buf();
    let records = manifest.records();
    let verified_source = verify_conversion_manifest(
        manifest.repository_id(),
        manifest.revision(),
        &model_dir,
        records,
    )
    .context("authenticate official Qwen3.8 source files")?;
    manifest
        .verify_source(&model_dir, &verified_source)
        .context("join official source to embedded source manifest")?;
    let source = source_identity(&model_dir, &verified_source)?;
    ensure!(
        source.tensor_bundle_sha256 == manifest.bundle_sha256(),
        "official source bundle differs from the embedded source manifest"
    );
    let inventory = derive_source_tensor_inventory(&model_dir, source.clone(), &verified_source)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    ensure!(
        inventory.manifest().tensors.len() == OFFICIAL_TENSOR_COUNT,
        "official Qwen3.8 source tensor count differs from the accepted recipe"
    );
    let (units, partition) = source_teacher_partition(&inventory)?;
    Ok(OfficialSourceV1 {
        _materialized_source: materialized_source,
        model_dir,
        source_manifest_id: manifest.manifest_id().to_owned(),
        source_manifest_sha256: manifest.manifest_sha256().to_owned(),
        source,
        verified_source,
        inventory,
        units,
        partition,
    })
}

/// Convert an hf-hub symlink snapshot into a private flat regular-file view
/// without copying payload bytes. Every link is re-authenticated below before
/// B1 retains its descriptors, so pathname races can only fail closed.
fn materialize_manifest_source(
    model_dir: &Path,
    manifest: &OfficialSourceManifestV1,
) -> Result<tempfile::TempDir> {
    let parent = model_dir
        .parent()
        .context("official source directory has no parent for hard-link staging")?;
    let staging = tempfile::Builder::new()
        .prefix(".hf2q-source-teacher-")
        .tempdir_in(parent)
        .context("create private source-teacher hard-link view")?;
    for file in manifest.files() {
        hard_link_source_leaf(model_dir, staging.path(), file.path(), file.size())?;
    }
    Ok(staging)
}

fn hard_link_source_leaf(
    model_dir: &Path,
    staging: &Path,
    filename: &str,
    expected_bytes: u64,
) -> Result<()> {
    super::super::retained_io::require_safe_leaf(filename)?;
    let resolved = std::fs::canonicalize(model_dir.join(filename))
        .with_context(|| format!("resolve official source file {filename}"))?;
    let before = std::fs::metadata(&resolved)
        .with_context(|| format!("inspect official source file {filename}"))?;
    ensure!(
        before.file_type().is_file() && before.len() == expected_bytes,
        "official source file {filename} is not the expected regular payload"
    );
    let destination = staging.join(filename);
    std::fs::hard_link(&resolved, &destination)
        .with_context(|| format!("hard-link official source file {filename}"))?;
    let after = std::fs::symlink_metadata(&destination)?;
    ensure!(
        after.file_type().is_file()
            && after.len() == expected_bytes
            && (after.dev(), after.ino()) == (before.dev(), before.ino()),
        "official source hard-link {filename} changed identity"
    );
    Ok(())
}

#[cfg(test)]
pub(super) fn hard_link_source_leaf_for_test(
    model_dir: &Path,
    staging: &Path,
    filename: &str,
    expected_bytes: u64,
) -> Result<()> {
    hard_link_source_leaf(model_dir, staging, filename, expected_bytes)
}

fn source_identity(
    model_dir: &Path,
    verified_source: &VerifiedSourceManifest,
) -> Result<SourceIdentity> {
    let config = std::fs::read(model_dir.join("config.json"))?;
    let tokenizer = std::fs::read(model_dir.join("tokenizer.json"))?;
    let render = crate::core::chat_template_resolver::resolve_chat_render_inputs(
        model_dir, &tokenizer, "qwen35",
    )?;
    let template = render
        .template
        .context("official Qwen3.8 source has no admitted chat template")?;
    let tensor_bundle_sha256 = compute_source_bundle_sha256(
        &verified_source
            .records()
            .iter()
            .map(SourceShard::from_integrity)
            .collect::<Vec<_>>(),
    )
    .context("official source manifest contains no strong tensor identity")?;
    Ok(SourceIdentity {
        model_id: verified_source.repo().to_owned(),
        revision: verified_source.revision().to_owned(),
        config_sha256: hex::encode(Sha256::digest(config)),
        tensor_bundle_sha256,
        tokenizer_bundle_sha256: render.tokenizer_bundle_sha256,
        chat_template_sha256: template.sha256,
    })
}

/// Teacher-only source closure. Every text tensor is protected; the later
/// one-option-at-a-time sensitivity partition is a distinct authority.
fn source_teacher_partition(
    inventory: &VerifiedSourceTensorInventory,
) -> Result<(Vec<TensorAllocationUnit>, TensorPartitionManifest)> {
    let non_variable = inventory
        .manifest()
        .tensors
        .iter()
        .cloned()
        .map(|source| {
            let vision = crate::convert::arch::qwen35_dense::is_qwen35_dense_vision_source_tensor(
                &source.name,
            );
            NonVariableTensor {
                source,
                disposition: if vision {
                    NonVariableDisposition::Excluded
                } else {
                    NonVariableDisposition::Protected
                },
                reason: if vision {
                    "official Qwen source teacher excludes the vision tower"
                } else {
                    "official Qwen source teacher protects exact source text weights"
                }
                .into(),
            }
        })
        .collect();
    let units = Vec::new();
    let partition = build_tensor_partition(inventory, &units, non_variable)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok((units, partition))
}

fn snapshot_limits(verified: &VerifiedSourceManifest) -> Result<QwenSourceSnapshotLimits> {
    let config_bytes = verified
        .records()
        .iter()
        .find(|record| record.filename == "config.json")
        .context("official source manifest is missing config.json")?
        .bytes;
    let max_total_source_bytes = verified
        .required_weight_shards()
        .iter()
        .try_fold(config_bytes, |total, name| {
            verified
                .records()
                .iter()
                .find(|record| record.filename == *name)
                .and_then(|record| total.checked_add(record.bytes))
        })
        .context("official source byte count overflow")?;
    ensure!(
        verified.required_weight_shards().len() == OFFICIAL_WEIGHT_SHARD_COUNT,
        "official Qwen3.8 weight-shard count differs from the accepted recipe"
    );
    Ok(QwenSourceSnapshotLimits {
        max_shards: OFFICIAL_WEIGHT_SHARD_COUNT,
        max_tensors: OFFICIAL_TENSOR_COUNT,
        max_header_bytes_per_shard: 64 * 1024 * 1024,
        max_total_header_bytes: 256 * 1024 * 1024,
        max_config_bytes: config_bytes,
        max_total_source_bytes,
    })
}

fn sha256_json(value: &impl Serialize) -> Result<String> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(value)?)))
}
