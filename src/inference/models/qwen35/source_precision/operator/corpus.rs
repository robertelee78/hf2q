use std::path::PathBuf;

use anyhow::{ensure, Context, Result};

use crate::intelligence::calibration::{
    build_teacher_prediction_plan, render_and_tokenize_split, render_and_tokenize_verified_split,
    verify_dataset_partition, verify_embedded_calibration_corpus_artifact,
    CalibrationCorpusArtifactLimits, DatasetSplit, RenderDatasetRequest,
    VerifiedCalibrationPredictionPlan, VerifyCalibrationCorpusRequest,
};

use super::profile::OfficialEvidenceProfileV1;
use super::source::OfficialSourceV1;

const CALIBRATION: &[u8] = include_bytes!(
    "../../../../../../data/calibration/qwen38-source-teacher-canary-v1/calibration.json"
);
const POLICY_VALIDATION: &[u8] = include_bytes!(
    "../../../../../../data/calibration/qwen38-source-teacher-canary-v1/policy-validation.json"
);
const ACCEPTANCE_HOLDOUT: &[u8] = include_bytes!(
    "../../../../../../data/calibration/qwen38-source-teacher-canary-v1/acceptance-holdout.json"
);

pub(super) struct OfficialPredictionPlanV1 {
    pub(super) plan: VerifiedCalibrationPredictionPlan,
    pub(super) dataset_partition_sha256: String,
    pub(super) calibration_corpus_sha256: String,
    pub(super) policy_validation_corpus_sha256: String,
    pub(super) acceptance_holdout_corpus_sha256: String,
}

pub(super) fn build_official_prediction_plan(
    source: &OfficialSourceV1,
    profile: &OfficialEvidenceProfileV1,
) -> Result<OfficialPredictionPlanV1> {
    let limits = profile.prediction_limits();
    let calibration = verified_corpus(
        CALIBRATION,
        &profile.dataset.calibration_sha256,
        DatasetSplit::Calibration,
        profile,
    )?;
    let validation = verified_corpus(
        POLICY_VALIDATION,
        &profile.dataset.policy_validation_sha256,
        DatasetSplit::PolicyValidation,
        profile,
    )?;
    let holdout = verified_corpus(
        ACCEPTANCE_HOLDOUT,
        &profile.dataset.acceptance_holdout_sha256,
        DatasetSplit::AcceptanceHoldout,
        profile,
    )?;
    let request = RenderDatasetRequest {
        model_dir: source.model_dir.clone(),
        arch: "qwen35".into(),
        source: source.source.clone(),
        verified_source: source.verified_source.clone(),
        renderer_revision: profile.render.renderer_revision.clone(),
        max_tokens_per_example: profile.render.max_tokens_per_example,
        token_window_size: profile.render.token_window_size,
    };
    let rendered_calibration = render_and_tokenize_verified_split(&calibration, &request, limits)
        .context("render official Calibration corpus")?;
    let rendered_validation = render_and_tokenize_split(validation.manifest(), &request)
        .context("render official PolicyValidation corpus")?;
    let rendered_holdout = render_and_tokenize_split(holdout.manifest(), &request)
        .context("render official AcceptanceHoldout corpus")?;
    let partition = verify_dataset_partition(
        &rendered_calibration,
        &rendered_validation,
        &rendered_holdout,
    )
    .context("verify official three-way corpus partition")?;
    let plan = build_teacher_prediction_plan(
        &partition,
        &calibration,
        &rendered_calibration,
        &rendered_validation,
        &rendered_holdout,
        limits,
    )
    .context("build official source-teacher prediction plan")?;
    Ok(OfficialPredictionPlanV1 {
        plan,
        dataset_partition_sha256: partition.manifest_sha256,
        calibration_corpus_sha256: calibration.artifact().sha256.clone(),
        policy_validation_corpus_sha256: validation.artifact().sha256.clone(),
        acceptance_holdout_corpus_sha256: holdout.artifact().sha256.clone(),
    })
}

fn verified_corpus(
    bytes: &[u8],
    sha256: &str,
    split: DatasetSplit,
    profile: &OfficialEvidenceProfileV1,
) -> Result<crate::intelligence::calibration::VerifiedCalibrationCorpus> {
    let corpus = verify_embedded_calibration_corpus_artifact(
        bytes,
        &VerifyCalibrationCorpusRequest {
            path: PathBuf::from(format!("<embedded:{split:?}>")),
            expected_sha256: sha256.into(),
            expected_dataset_id: profile.dataset.dataset_id.clone(),
            expected_revision: profile.dataset.revision.clone(),
            expected_declared_license: profile.dataset.license.clone(),
            expected_split: split,
            limits: CalibrationCorpusArtifactLimits {
                max_artifact_bytes: 16 * 1024,
                max_examples: 2,
                max_messages: 8,
                max_tools: 2,
            },
        },
    )?;
    ensure!(
        corpus.manifest().seed == profile.dataset.seed,
        "embedded corpus seed differs from the evidence profile"
    );
    Ok(corpus)
}

#[cfg(test)]
pub(super) fn embedded_corpus_bytes_for_test() -> [(&'static [u8], &'static str, DatasetSplit); 3] {
    [
        (
            CALIBRATION,
            "962ad640b9dab5beabcec535fca068b114dccace74d7735d2547fe4facd474e0",
            DatasetSplit::Calibration,
        ),
        (
            POLICY_VALIDATION,
            "b5bd050535a9d8effa5deb0552b77d7fb8f4a3f55dd3189771121ca2953fe662",
            DatasetSplit::PolicyValidation,
        ),
        (
            ACCEPTANCE_HOLDOUT,
            "29c619efbee46623dea3fdc0c60ce82f2dd7d4fc01cedeb199ad5f774a3ad018",
            DatasetSplit::AcceptanceHoldout,
        ),
    ]
}
