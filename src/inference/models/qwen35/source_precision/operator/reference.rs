//! Pinned, cross-process source-reference comparison operator.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use rustix::fs::{self, FileType, Mode, OFlags};
use serde::{Deserialize, Serialize};

use crate::intelligence::exact_teacher::{
    compare_exact_teacher_reference_targets, validate_exact_teacher_reference_input,
    ExactTeacherExternalReferenceEvidenceV1, ExactTeacherReferenceComparisonReceiptV1,
    ExactTeacherReferenceInputV1, ExactTeacherTargetReceipt,
};

use super::acceptance::{
    evaluate_official_acceptance_comparison, official_acceptance_thresholds,
    validate_official_acceptance_gate_receipt_artifact, OfficialQwen38AcceptanceGateReceiptV1,
};
use super::corpus::{
    build_official_acceptance_prediction_plan, build_official_prediction_plan,
    OfficialPredictionPlanV1,
};
use super::profile::official_profile;
use super::source::authenticate_official_source;
use super::OfficialQwen38EvaluationSplitV1;

const MAX_EVIDENCE_JSON_BYTES: u64 = 16 * 1024 * 1024;
const EVIDENCE_OPEN_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::NOFOLLOW)
    .union(OFlags::NONBLOCK)
    .union(OFlags::CLOEXEC);

#[derive(Debug, Clone)]
pub(crate) struct OfficialQwen38SourceReferenceRequestV1 {
    pub(crate) model_dir: PathBuf,
    pub(crate) native_summary: PathBuf,
    pub(crate) native_target: PathBuf,
    pub(crate) external_evidence: PathBuf,
    pub(crate) external_target: PathBuf,
}

#[derive(Debug, Clone)]
pub(crate) struct OfficialQwen38AcceptanceReferenceRequestV1 {
    pub(crate) model_dir: PathBuf,
    pub(crate) native_summary: PathBuf,
    pub(crate) native_target: PathBuf,
    pub(crate) external_evidence: PathBuf,
    pub(crate) external_target: PathBuf,
    pub(crate) raw_comparison_output: PathBuf,
    pub(crate) quality_gate_output: PathBuf,
}

#[derive(Deserialize)]
struct NativeSummaryReferenceViewV1 {
    profile: String,
    hf2q_git_commit: String,
    evaluation_split: crate::intelligence::calibration::DatasetSplit,
    threshold_profile_sha256: Option<String>,
    prediction_plan_sha256: String,
    executed: bool,
    target_artifact_sha256: Option<String>,
    completion_receipt_sha256: Option<String>,
    reference_input: ExactTeacherReferenceInputV1,
    structural_target_receipt: Option<ExactTeacherTargetReceipt>,
}

pub(crate) fn compare_official_qwen38_source_reference(
    request: &OfficialQwen38SourceReferenceRequestV1,
) -> Result<ExactTeacherReferenceComparisonReceiptV1> {
    let profile = official_profile()?;
    let (_, comparison) = compare_reference(request, &profile, ReferenceModeV1::Characterization)?;
    Ok(comparison)
}

pub(crate) fn compare_official_qwen38_acceptance_reference(
    request: &OfficialQwen38AcceptanceReferenceRequestV1,
) -> Result<OfficialQwen38AcceptanceGateReceiptV1> {
    ensure!(
        evidence_destination_identity(&request.raw_comparison_output)?
            != evidence_destination_identity(&request.quality_gate_output)?,
        "raw comparison and quality-gate destinations must differ"
    );
    preflight_new_evidence_path(&request.raw_comparison_output)?;
    preflight_new_evidence_path(&request.quality_gate_output)?;
    let profile = official_profile()?;
    let thresholds_for_plan = official_acceptance_thresholds(&profile)?;
    let base = OfficialQwen38SourceReferenceRequestV1 {
        model_dir: request.model_dir.clone(),
        native_summary: request.native_summary.clone(),
        native_target: request.native_target.clone(),
        external_evidence: request.external_evidence.clone(),
        external_target: request.external_target.clone(),
    };
    let (prediction, comparison) = compare_reference(
        &base,
        &profile,
        ReferenceModeV1::Acceptance(thresholds_for_plan),
    )?;
    write_json_new(&request.raw_comparison_output, &comparison)
        .context("publish raw AcceptanceHoldout comparison before gating")?;
    let thresholds_for_gate = official_acceptance_thresholds(&profile)?;
    let receipt = evaluate_official_acceptance_comparison(
        &thresholds_for_gate,
        &prediction.plan,
        comparison,
    )?;
    write_json_new(&request.quality_gate_output, &receipt)
        .context("publish successful AcceptanceHoldout quality receipt")?;
    let published = read_bounded_bytes(&request.quality_gate_output)
        .context("reopen published AcceptanceHoldout quality receipt")?;
    let thresholds_for_verification = official_acceptance_thresholds(&profile)?;
    let verified = validate_official_acceptance_gate_receipt_artifact(
        &thresholds_for_verification,
        &published,
    )?;
    ensure!(
        verified == receipt,
        "published AcceptanceHoldout quality receipt changed after creation"
    );
    Ok(verified)
}

enum ReferenceModeV1 {
    Characterization,
    Acceptance(super::acceptance::VerifiedQwen38AcceptanceThresholdsV1),
}

fn compare_reference(
    request: &OfficialQwen38SourceReferenceRequestV1,
    profile: &super::profile::OfficialEvidenceProfileV1,
    mode: ReferenceModeV1,
) -> Result<(
    OfficialPredictionPlanV1,
    ExactTeacherReferenceComparisonReceiptV1,
)> {
    let native: NativeSummaryReferenceViewV1 = read_json(&request.native_summary)
        .context("read native source-teacher evidence summary")?;
    ensure!(
        native.profile == "qwen38_source_teacher_canary_v1"
            && native.executed
            && native.prediction_plan_sha256
                == native.reference_input.prediction_plan.manifest_sha256,
        "native source-teacher summary is not a completed reference input"
    );
    validate_exact_teacher_reference_input(&native.reference_input)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let native_receipt = native
        .structural_target_receipt
        .context("native summary lacks its structural target receipt")?;
    let native_completion = native
        .completion_receipt_sha256
        .context("native summary lacks its family completion receipt hash")?;
    ensure!(
        native.target_artifact_sha256.as_deref()
            == Some(native_receipt.target_artifact.sha256.as_str()),
        "native summary target hash differs from its structural receipt"
    );
    let external: ExactTeacherExternalReferenceEvidenceV1 =
        read_json(&request.external_evidence).context("read external reference evidence")?;
    let comparator_git_commit = crate::convert::receipt::require_converter_git_commit()
        .context("resolve exact hf2q comparator Git commit")?;

    let source = authenticate_official_source(&request.model_dir, profile)?;
    let prediction = match (mode, native.evaluation_split) {
        (
            ReferenceModeV1::Characterization,
            crate::intelligence::calibration::DatasetSplit::Calibration,
        ) => {
            ensure!(
                native.threshold_profile_sha256.is_none(),
                "characterization summary unexpectedly binds acceptance thresholds"
            );
            build_official_prediction_plan(
                &source,
                profile,
                OfficialQwen38EvaluationSplitV1::Calibration,
            )?
        }
        (
            ReferenceModeV1::Characterization,
            crate::intelligence::calibration::DatasetSplit::PolicyValidation,
        ) => {
            ensure!(
                native.threshold_profile_sha256.is_none(),
                "characterization summary unexpectedly binds acceptance thresholds"
            );
            build_official_prediction_plan(
                &source,
                profile,
                OfficialQwen38EvaluationSplitV1::PolicyValidation,
            )?
        }
        (
            ReferenceModeV1::Acceptance(thresholds),
            crate::intelligence::calibration::DatasetSplit::AcceptanceHoldout,
        ) => {
            ensure!(
                native.threshold_profile_sha256.as_deref()
                    == Some(thresholds.plan_authority.threshold_profile_sha256())
                    && native.hf2q_git_commit == comparator_git_commit
                    && external.implementation == thresholds.external_implementation,
                "holdout hf2q lineage or external implementation differs from its predeclaration"
            );
            build_official_acceptance_prediction_plan(&source, profile, thresholds)?
        }
        _ => anyhow::bail!("source-reference command does not match the native evaluation split"),
    };
    ensure!(
        prediction.evaluation_split == native.evaluation_split
            && prediction.plan.manifest().manifest_sha256 == native.prediction_plan_sha256,
        "freshly authenticated prediction plan differs from native evidence"
    );
    let comparison = compare_exact_teacher_reference_targets(
        &prediction.plan,
        &native.reference_input,
        &request.native_target,
        native_receipt,
        native_completion,
        comparator_git_commit,
        &request.external_target,
        &external,
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok((prediction, comparison))
}

fn write_json_new(path: &Path, value: &impl Serialize) -> Result<()> {
    let descriptor = fs::open(
        path,
        OFlags::WRONLY
            .union(OFlags::CREATE)
            .union(OFlags::EXCL)
            .union(OFlags::NOFOLLOW)
            .union(OFlags::CLOEXEC),
        Mode::from_raw_mode(0o600),
    )
    .map_err(std::io::Error::from)
    .with_context(|| format!("create fresh evidence file {}", path.display()))?;
    let mut file = File::from(descriptor);
    serde_json::to_writer(&mut file, value)
        .with_context(|| format!("serialize evidence file {}", path.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("finish evidence file {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("sync evidence file {}", path.display()))?;
    sync_parent_directory(path)?;
    Ok(())
}

fn preflight_new_evidence_path(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => {
            Err(error).with_context(|| format!("inspect evidence destination {}", path.display()))
        }
        Ok(_) => anyhow::bail!("evidence destination already exists: {}", path.display()),
    }
}

fn evidence_destination_identity(path: &Path) -> Result<PathBuf> {
    let file_name = path
        .file_name()
        .context("evidence destination lacks a file name")?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let canonical_parent = parent.canonicalize().with_context(|| {
        format!(
            "canonicalize evidence parent directory {}",
            parent.display()
        )
    })?;
    Ok(canonical_parent.join(file_name))
}

fn sync_parent_directory(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let descriptor = fs::open(
        parent,
        OFlags::RDONLY
            .union(OFlags::DIRECTORY)
            .union(OFlags::NOFOLLOW)
            .union(OFlags::CLOEXEC),
        Mode::empty(),
    )
    .map_err(std::io::Error::from)
    .with_context(|| format!("open evidence parent directory {}", parent.display()))?;
    File::from(descriptor)
        .sync_all()
        .with_context(|| format!("sync evidence parent directory {}", parent.display()))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = read_bounded_bytes(path)?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parse evidence file {}", path.display()))
}

fn read_bounded_bytes(path: &Path) -> Result<Vec<u8>> {
    let descriptor = fs::open(path, EVIDENCE_OPEN_FLAGS, Mode::empty())
        .map_err(std::io::Error::from)
        .with_context(|| format!("open evidence file {}", path.display()))?;
    let mut file = File::from(descriptor);
    let stat = fs::fstat(&file)
        .map_err(std::io::Error::from)
        .with_context(|| format!("inspect evidence file {}", path.display()))?;
    ensure!(
        FileType::from_raw_mode(stat.st_mode) == FileType::RegularFile
            && stat.st_size > 0
            && u64::try_from(stat.st_size).is_ok_and(|size| size <= MAX_EVIDENCE_JSON_BYTES),
        "evidence input {} is not a bounded regular file",
        path.display()
    );
    let mut bytes = Vec::with_capacity(usize::try_from(stat.st_size).unwrap_or(0));
    file.read_to_end(&mut bytes)
        .with_context(|| format!("read evidence file {}", path.display()))?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    #[test]
    fn evidence_publication_is_create_new_and_no_clobber() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("receipt.json");
        let alias = temp.path().join(".").join("receipt.json");
        assert_eq!(
            super::evidence_destination_identity(&output).unwrap(),
            super::evidence_destination_identity(&alias).unwrap()
        );
        super::preflight_new_evidence_path(&output).unwrap();
        super::write_json_new(&output, &serde_json::json!({"receipt": 1})).unwrap();
        assert_eq!(
            std::fs::read_to_string(&output).unwrap(),
            "{\"receipt\":1}\n"
        );
        assert!(super::write_json_new(&output, &serde_json::json!({"receipt": 2})).is_err());
        assert!(super::preflight_new_evidence_path(&output).is_err());
        assert_eq!(
            std::fs::read_to_string(&output).unwrap(),
            "{\"receipt\":1}\n"
        );
    }
}
