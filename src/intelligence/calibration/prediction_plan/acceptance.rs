//! Sealed, capability-consuming AcceptanceHoldout plan construction.

use super::*;

/// The only prediction-plan constructor that may retain AcceptanceHoldout.
/// Its opaque threshold capability must bind the freshly rendered source and
/// exact embedded holdout corpus before any plan or Metal work can proceed.
pub(crate) fn build_teacher_acceptance_holdout_plan(
    thresholds: VerifiedTeacherAcceptanceThresholdsV1,
    expected_partition: &DatasetPartitionManifest,
    evaluation_corpus: &VerifiedCalibrationCorpus,
    evaluation: &RenderedDataset,
    calibration: &RenderedDataset,
    policy_validation: &RenderedDataset,
    acceptance_holdout: &RenderedDataset,
    limits: TeacherPredictionPlanLimits,
) -> Result<VerifiedTeacherAcceptanceHoldoutPlanV1, CalibrationInputError> {
    if evaluation.manifest.split != DatasetSplit::AcceptanceHoldout
        || thresholds.source != evaluation.manifest.source
        || thresholds.verified_source_manifest_sha256
            != evaluation.manifest.verified_source_manifest_sha256
        || thresholds.acceptance_holdout_corpus_sha256 != evaluation_corpus.artifact.sha256
    {
        return Err(CalibrationInputError::InvalidDataset(
            "AcceptanceHoldout differs from its predeclared threshold binding".into(),
        ));
    }
    if evaluation.structured.examples.len() != 1
        || evaluation
            .structured
            .examples
            .iter()
            .any(|example| example.render_mode != RenderMode::GenerationPrompt)
    {
        return Err(CalibrationInputError::InvalidDataset(
            "AcceptanceHoldout must contain exactly one generation prompt".into(),
        ));
    }
    let plan = super::build_teacher_prediction_plan(
        expected_partition,
        DatasetSplit::AcceptanceHoldout,
        evaluation_corpus,
        evaluation,
        calibration,
        policy_validation,
        acceptance_holdout,
        limits,
        true,
    )?;
    if plan.manifest.examples.len() != 1
        || plan.manifest.prediction_points.len() != 1
        || plan.manifest.greedy_prompts.len() != 1
        || !matches!(
            plan.manifest.prediction_points[0].kind,
            TeacherPredictionPointKind::GenerationNext
        )
    {
        return Err(CalibrationInputError::InvalidDataset(
            "AcceptanceHoldout prediction plan violates its sealed shape".into(),
        ));
    }
    Ok(VerifiedTeacherAcceptanceHoldoutPlanV1 {
        plan,
        threshold_profile_sha256: thresholds.threshold_profile_sha256,
    })
}

/// Seal a family-verified, predeclared threshold binding into the capability
/// consumed by the holdout-only constructor. Numeric threshold verification
/// remains with the family-owned exact-reference operator.
pub(crate) fn bind_teacher_acceptance_thresholds(
    threshold_profile_sha256: String,
    calibration_comparison_receipt_sha256: String,
    policy_validation_comparison_receipt_sha256: String,
    source: crate::intelligence::measured_auto_quant::SourceIdentity,
    verified_source_manifest_sha256: String,
    acceptance_holdout_corpus_sha256: String,
) -> Result<VerifiedTeacherAcceptanceThresholdsV1, CalibrationInputError> {
    let is_sha256 = |value: &str| {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    };
    if !super::super::render::source_valid(&source)
        || !is_sha256(&threshold_profile_sha256)
        || !is_sha256(&calibration_comparison_receipt_sha256)
        || !is_sha256(&policy_validation_comparison_receipt_sha256)
        || calibration_comparison_receipt_sha256 == policy_validation_comparison_receipt_sha256
        || !is_sha256(&verified_source_manifest_sha256)
        || !is_sha256(&acceptance_holdout_corpus_sha256)
    {
        return Err(CalibrationInputError::InvalidDataset(
            "predeclared acceptance-threshold binding is invalid".into(),
        ));
    }
    Ok(VerifiedTeacherAcceptanceThresholdsV1 {
        threshold_profile_sha256,
        calibration_comparison_receipt_sha256,
        policy_validation_comparison_receipt_sha256,
        source,
        verified_source_manifest_sha256,
        acceptance_holdout_corpus_sha256,
    })
}
