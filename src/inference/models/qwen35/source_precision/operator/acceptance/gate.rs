use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

use crate::intelligence::calibration::{
    DatasetSplit, RenderMode, TeacherPredictionPointKind, VerifiedTeacherPredictionPlan,
};
use crate::intelligence::exact_teacher::{
    validate_canonical_exact_teacher_reference_comparison_receipt,
    ExactTeacherReferenceComparisonReceiptV1, ExactTeacherReferenceTrajectoryComparisonV1,
    EXACT_TEACHER_GREEDY_TOKEN_COUNT as GREEDY_TOKEN_COUNT,
};

use super::{sha256_json, Qwen38SourceReferenceThresholdsV1, VerifiedQwen38AcceptanceThresholdsV1};

const ACCEPTANCE_GATE_SCHEMA_VERSION: u32 = 1;
const ACCEPTANCE_GATE_PROFILE: &str = "qwen38_source_bf16_acceptance_gate_v1";
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Qwen38AcceptanceMetricEvaluationV1 {
    row_count: usize,
    trajectory_count: usize,
    trajectory_token_count: usize,
    row_count_passed: bool,
    trajectory_count_passed: bool,
    max_abs_passed: bool,
    row_kl_passed: bool,
    top1_match_passed: bool,
    trajectory_passed: bool,
    greedy_matching_prefix_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OfficialQwen38AcceptanceGateReceiptV1 {
    schema_version: u32,
    profile: String,
    threshold_profile_sha256: String,
    calibration_comparison_receipt_sha256: String,
    policy_validation_comparison_receipt_sha256: String,
    acceptance_prediction_plan_sha256: String,
    thresholds: Qwen38SourceReferenceThresholdsV1,
    evaluation: Qwen38AcceptanceMetricEvaluationV1,
    comparison: ExactTeacherReferenceComparisonReceiptV1,
    thresholds_predeclared: bool,
    quality_gate_authority: bool,
    source_teacher_authority: bool,
    sensitivity_authority: bool,
    allocator_authority: bool,
    selector_authority: bool,
    autoquant_authority: bool,
    runtime_dependency: bool,
    dwq: bool,
    acceptance_gate_receipt_sha256: String,
}

#[derive(Serialize)]
struct AcceptanceGateHashView<'a> {
    schema_version: u32,
    profile: &'a str,
    threshold_profile_sha256: &'a str,
    calibration_comparison_receipt_sha256: &'a str,
    policy_validation_comparison_receipt_sha256: &'a str,
    acceptance_prediction_plan_sha256: &'a str,
    thresholds: Qwen38SourceReferenceThresholdsV1,
    evaluation: &'a Qwen38AcceptanceMetricEvaluationV1,
    comparison: &'a ExactTeacherReferenceComparisonReceiptV1,
    thresholds_predeclared: bool,
    quality_gate_authority: bool,
    source_teacher_authority: bool,
    sensitivity_authority: bool,
    allocator_authority: bool,
    selector_authority: bool,
    autoquant_authority: bool,
    runtime_dependency: bool,
    dwq: bool,
}

pub(crate) fn evaluate_official_acceptance_comparison(
    authority: &VerifiedQwen38AcceptanceThresholdsV1,
    acceptance_plan: &VerifiedTeacherPredictionPlan,
    comparison: ExactTeacherReferenceComparisonReceiptV1,
) -> Result<OfficialQwen38AcceptanceGateReceiptV1> {
    validate_canonical_exact_teacher_reference_comparison_receipt(&comparison)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    validate_acceptance_plan_shape(acceptance_plan, authority.thresholds)?;
    ensure!(
        acceptance_plan.manifest().evaluation_split == DatasetSplit::AcceptanceHoldout
            && comparison.prediction_plan_sha256 == acceptance_plan.manifest().manifest_sha256,
        "holdout comparison differs from its freshly authenticated plan"
    );
    let evaluation = comparison_passes_thresholds(&comparison, authority.thresholds)?;
    let mut receipt = OfficialQwen38AcceptanceGateReceiptV1 {
        schema_version: ACCEPTANCE_GATE_SCHEMA_VERSION,
        profile: ACCEPTANCE_GATE_PROFILE.into(),
        threshold_profile_sha256: authority.plan_authority.threshold_profile_sha256().into(),
        calibration_comparison_receipt_sha256: authority
            .plan_authority
            .calibration_comparison_receipt_sha256()
            .into(),
        policy_validation_comparison_receipt_sha256: authority
            .plan_authority
            .policy_validation_comparison_receipt_sha256()
            .into(),
        acceptance_prediction_plan_sha256: acceptance_plan.manifest().manifest_sha256.clone(),
        thresholds: authority.thresholds,
        evaluation,
        comparison,
        thresholds_predeclared: true,
        quality_gate_authority: true,
        source_teacher_authority: false,
        sensitivity_authority: false,
        allocator_authority: false,
        selector_authority: false,
        autoquant_authority: false,
        runtime_dependency: false,
        dwq: false,
        acceptance_gate_receipt_sha256: String::new(),
    };
    receipt.acceptance_gate_receipt_sha256 = acceptance_gate_sha256(&receipt)?;
    validate_official_acceptance_gate_receipt(authority, &receipt)?;
    Ok(receipt)
}

pub(crate) fn validate_official_acceptance_gate_receipt_artifact(
    authority: &VerifiedQwen38AcceptanceThresholdsV1,
    bytes: &[u8],
) -> Result<OfficialQwen38AcceptanceGateReceiptV1> {
    ensure!(
        !bytes.is_empty() && bytes.len() <= 16 * 1024 * 1024,
        "acceptance quality receipt is empty or too large"
    );
    let receipt: OfficialQwen38AcceptanceGateReceiptV1 =
        serde_json::from_slice(bytes).context("parse acceptance quality receipt")?;
    validate_official_acceptance_gate_receipt(authority, &receipt)?;
    Ok(receipt)
}

fn validate_official_acceptance_gate_receipt(
    authority: &VerifiedQwen38AcceptanceThresholdsV1,
    receipt: &OfficialQwen38AcceptanceGateReceiptV1,
) -> Result<()> {
    validate_canonical_exact_teacher_reference_comparison_receipt(&receipt.comparison)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let reproduced = comparison_passes_thresholds(&receipt.comparison, authority.thresholds)?;
    ensure!(
        receipt.schema_version == ACCEPTANCE_GATE_SCHEMA_VERSION
            && receipt.profile == ACCEPTANCE_GATE_PROFILE
            && receipt.threshold_profile_sha256
                == authority.plan_authority.threshold_profile_sha256()
            && receipt.calibration_comparison_receipt_sha256
                == authority
                    .plan_authority
                    .calibration_comparison_receipt_sha256()
            && receipt.policy_validation_comparison_receipt_sha256
                == authority
                    .plan_authority
                    .policy_validation_comparison_receipt_sha256()
            && receipt.acceptance_prediction_plan_sha256
                == receipt.comparison.prediction_plan_sha256
            && receipt.thresholds == authority.thresholds
            && receipt.comparison.external_implementation == authority.external_implementation
            && receipt.evaluation == reproduced
            && receipt.comparison.rows[0].point_ordinal == 0
            && receipt.comparison.rows[0].stable_id == "holdout-cache-001"
            && receipt.comparison.trajectories[0].stable_id == "holdout-cache-001"
            && receipt.thresholds_predeclared
            && receipt.quality_gate_authority
            && !receipt.source_teacher_authority
            && !receipt.sensitivity_authority
            && !receipt.allocator_authority
            && !receipt.selector_authority
            && !receipt.autoquant_authority
            && !receipt.runtime_dependency
            && !receipt.dwq
            && acceptance_gate_sha256(receipt)? == receipt.acceptance_gate_receipt_sha256,
        "acceptance quality receipt identity, evidence, or authority scope is invalid"
    );
    Ok(())
}

pub(super) fn comparison_passes_thresholds(
    comparison: &ExactTeacherReferenceComparisonReceiptV1,
    thresholds: Qwen38SourceReferenceThresholdsV1,
) -> Result<Qwen38AcceptanceMetricEvaluationV1> {
    let greedy_matching_prefix_tokens = trajectory_matching_prefix(&comparison.trajectories)?;
    let rows_pass_max_abs = comparison
        .rows
        .iter()
        .all(|row| row.max_abs <= thresholds.max_abs);
    let rows_pass_kl = comparison
        .rows
        .iter()
        .all(|row| row.kl_reference_to_native <= thresholds.max_row_kl_reference_to_native);
    let rows_pass_top1 =
        !thresholds.require_top1_match || comparison.rows.iter().all(|row| row.top1_match);
    let evaluation = Qwen38AcceptanceMetricEvaluationV1 {
        row_count: comparison.rows.len(),
        trajectory_count: comparison.trajectories.len(),
        trajectory_token_count: GREEDY_TOKEN_COUNT,
        row_count_passed: comparison.rows.len() == thresholds.expected_row_count,
        trajectory_count_passed: comparison.trajectories.len()
            == thresholds.required_trajectory_count,
        max_abs_passed: rows_pass_max_abs,
        row_kl_passed: rows_pass_kl,
        top1_match_passed: rows_pass_top1,
        trajectory_passed: greedy_matching_prefix_tokens >= thresholds.min_first_divergence_index,
        greedy_matching_prefix_tokens,
    };
    ensure!(
        evaluation.row_count_passed
            && evaluation.trajectory_count_passed
            && evaluation.max_abs_passed
            && evaluation.row_kl_passed
            && evaluation.top1_match_passed
            && evaluation.trajectory_passed,
        "AcceptanceHoldout failed its predeclared source-reference thresholds"
    );
    Ok(evaluation)
}

fn validate_acceptance_plan_shape(
    plan: &VerifiedTeacherPredictionPlan,
    thresholds: Qwen38SourceReferenceThresholdsV1,
) -> Result<()> {
    let manifest = plan.manifest();
    let generation_examples = manifest
        .examples
        .iter()
        .filter(|example| example.render_mode == RenderMode::GenerationPrompt)
        .count();
    ensure!(
        manifest.evaluation_split == DatasetSplit::AcceptanceHoldout
            && manifest.examples.len() == 1
            && generation_examples == thresholds.expected_generation_prompt_count
            && manifest
                .prediction_points
                .iter()
                .all(|point| matches!(point.kind, TeacherPredictionPointKind::GenerationNext))
            && manifest.prediction_points.len() == thresholds.expected_row_count
            && manifest.greedy_prompts.len() == thresholds.required_trajectory_count,
        "authorized AcceptanceHoldout plan violates its sealed shape"
    );
    Ok(())
}

pub(super) fn trajectory_matching_prefix(
    trajectories: &[ExactTeacherReferenceTrajectoryComparisonV1],
) -> Result<usize> {
    ensure!(
        trajectories.len() == 1,
        "acceptance comparison must contain exactly one greedy trajectory"
    );
    let trajectory = &trajectories[0];
    let hashes_match = trajectory.native_token_ids_sha256 == trajectory.reference_token_ids_sha256;
    ensure!(
        if trajectory.exact_match {
            trajectory.first_divergence_index.is_none() && hashes_match
        } else {
            trajectory.first_divergence_index.is_some() && !hashes_match
        },
        "holdout trajectory hashes or divergence are inconsistent"
    );
    if trajectory.exact_match {
        return Ok(GREEDY_TOKEN_COUNT);
    }
    let first_divergence_index = trajectory
        .first_divergence_index
        .context("non-matching trajectory lacks its first divergence")?;
    ensure!(
        first_divergence_index < GREEDY_TOKEN_COUNT,
        "holdout trajectory divergence exceeds its fixed token count"
    );
    Ok(first_divergence_index)
}

fn acceptance_gate_sha256(receipt: &OfficialQwen38AcceptanceGateReceiptV1) -> Result<String> {
    sha256_json(&AcceptanceGateHashView {
        schema_version: receipt.schema_version,
        profile: &receipt.profile,
        threshold_profile_sha256: &receipt.threshold_profile_sha256,
        calibration_comparison_receipt_sha256: &receipt.calibration_comparison_receipt_sha256,
        policy_validation_comparison_receipt_sha256: &receipt
            .policy_validation_comparison_receipt_sha256,
        acceptance_prediction_plan_sha256: &receipt.acceptance_prediction_plan_sha256,
        thresholds: receipt.thresholds,
        evaluation: &receipt.evaluation,
        comparison: &receipt.comparison,
        thresholds_predeclared: receipt.thresholds_predeclared,
        quality_gate_authority: receipt.quality_gate_authority,
        source_teacher_authority: receipt.source_teacher_authority,
        sensitivity_authority: receipt.sensitivity_authority,
        allocator_authority: receipt.allocator_authority,
        selector_authority: receipt.selector_authority,
        autoquant_authority: receipt.autoquant_authority,
        runtime_dependency: receipt.runtime_dependency,
        dwq: receipt.dwq,
    })
}

#[cfg(test)]
pub(crate) fn comparison_passes_thresholds_for_test(
    comparison: &ExactTeacherReferenceComparisonReceiptV1,
    thresholds: Qwen38SourceReferenceThresholdsV1,
) -> Result<()> {
    comparison_passes_thresholds(comparison, thresholds).map(|_| ())
}
