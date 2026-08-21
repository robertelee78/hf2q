//! Pinned, cross-process source-reference comparison operator.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use rustix::fs::{self, FileType, Mode, OFlags};
use serde::Deserialize;

use crate::intelligence::exact_teacher::{
    compare_exact_teacher_reference_targets, validate_exact_teacher_reference_input,
    ExactTeacherExternalReferenceEvidenceV1, ExactTeacherReferenceComparisonReceiptV1,
    ExactTeacherReferenceInputV1, ExactTeacherTargetReceipt,
};

use super::corpus::build_official_prediction_plan;
use super::profile::official_profile;
use super::source::authenticate_official_source;

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

#[derive(Deserialize)]
struct NativeSummaryReferenceViewV1 {
    profile: String,
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

    let profile = official_profile()?;
    let source = authenticate_official_source(&request.model_dir, &profile)?;
    let prediction = build_official_prediction_plan(&source, &profile)?;
    ensure!(
        prediction.plan.manifest().manifest_sha256 == native.prediction_plan_sha256,
        "freshly authenticated prediction plan differs from native evidence"
    );
    compare_exact_teacher_reference_targets(
        &prediction.plan,
        &native.reference_input,
        &request.native_target,
        native_receipt,
        native_completion,
        &request.external_target,
        &external,
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
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
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parse evidence file {}", path.display()))
}
