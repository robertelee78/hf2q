//! Predeclared Qwen3.8 source-reference acceptance authority.
//!
//! The checked-in profile is derived only from the first Calibration and
//! PolicyValidation receipts. Holdout can be planned only after this module
//! verifies those exact bytes, their self-hashes, and the deterministic
//! headroom rule. A raw comparison remains non-authoritative.

use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::intelligence::calibration::{
    bind_teacher_acceptance_thresholds, VerifiedTeacherAcceptanceThresholdsV1,
};
use crate::intelligence::exact_teacher::{
    validate_exact_teacher_reference_comparison_artifact, ExactTeacherReferenceComparisonReceiptV1,
    ExternalReferenceImplementationV1,
};
use crate::intelligence::measured_auto_quant::SourceIdentity;

use super::profile::{OfficialEvidenceProfileV1, PROFILE_SHA256};
use super::source_manifest::MANIFEST_SHA256;

mod gate;

#[cfg(test)]
pub(super) use gate::comparison_passes_thresholds_for_test;
use gate::trajectory_matching_prefix;
pub(super) use gate::OfficialQwen38AcceptanceGateReceiptV1;

const THRESHOLD_PROFILE_BYTES: &[u8] = include_bytes!(
    "../../../../../../data/calibration/qwen38-source-teacher-canary-v1/acceptance-thresholds.json"
);
const THRESHOLD_PROFILE_ARTIFACT_SHA256: &str =
    "6a3d36c3006355315820b331aaaeb75bc04ef58b04b81c2be31692b7f99ababb";
const CALIBRATION_COMPARISON_BYTES: &[u8] = include_bytes!(
    "../../../../../../data/calibration/qwen38-source-teacher-canary-v1/calibration-reference-comparison.json"
);
const POLICY_VALIDATION_COMPARISON_BYTES: &[u8] = include_bytes!(
    "../../../../../../data/calibration/qwen38-source-teacher-canary-v1/policy-validation-reference-comparison.json"
);
const ACCEPTANCE_COMPARISON_BYTES: &[u8] = include_bytes!(
    "../../../../../../data/calibration/qwen38-source-teacher-canary-v1/acceptance-reference-comparison.json"
);
const ACCEPTANCE_COMPARISON_ARTIFACT_SHA256: &str =
    "e2f3cbd3bd1cce9e3964053a52409e36bf590679dea993881a066654a6e3ff01";
const ACCEPTANCE_GATE_BYTES: &[u8] = include_bytes!(
    "../../../../../../data/calibration/qwen38-source-teacher-canary-v1/acceptance-quality-gate.json"
);
const ACCEPTANCE_GATE_ARTIFACT_SHA256: &str =
    "9a7836e5ca1ed848dd6cc2bd64c4c9bcc97a418db346e5f182d0639720f8df2d";
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Qwen38SourceReferenceThresholdsV1 {
    pub(crate) expected_row_count: usize,
    pub(crate) expected_generation_prompt_count: usize,
    pub(crate) required_trajectory_count: usize,
    pub(crate) max_abs: f64,
    pub(crate) max_row_kl_reference_to_native: f64,
    pub(crate) require_top1_match: bool,
    pub(crate) min_first_divergence_index: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ThresholdProfileV1 {
    schema_version: u32,
    profile: String,
    source: SourceIdentity,
    verified_source_manifest_sha256: String,
    source_manifest_sha256: String,
    evidence_profile_sha256: String,
    acceptance_holdout_corpus_sha256: String,
    characterization: CharacterizationBindingV1,
    derivation: ThresholdDerivationV1,
    thresholds: Qwen38SourceReferenceThresholdsV1,
    scope: ThresholdScopeV1,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CharacterizationBindingV1 {
    calibration: ComparisonBindingV1,
    policy_validation: ComparisonBindingV1,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ComparisonBindingV1 {
    artifact_sha256: String,
    comparison_receipt_sha256: String,
    prediction_plan_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ThresholdDerivationV1 {
    max_abs_rule: String,
    row_kl_rule: String,
    top1_rule: String,
    trajectory_rule: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ThresholdScopeV1 {
    canary_only: bool,
    source_teacher_authority: bool,
    sensitivity_authority: bool,
    allocator_authority: bool,
    selector_authority: bool,
    autoquant_authority: bool,
    runtime_dependency: bool,
    dwq: bool,
}

pub(super) struct VerifiedQwen38AcceptanceThresholdsV1 {
    pub(super) plan_authority: VerifiedTeacherAcceptanceThresholdsV1,
    pub(super) thresholds: Qwen38SourceReferenceThresholdsV1,
    pub(super) external_implementation: ExternalReferenceImplementationV1,
}

pub(super) fn official_acceptance_thresholds(
    evidence_profile: &OfficialEvidenceProfileV1,
) -> Result<VerifiedQwen38AcceptanceThresholdsV1> {
    verify_threshold_bundle(
        THRESHOLD_PROFILE_BYTES,
        CALIBRATION_COMPARISON_BYTES,
        POLICY_VALIDATION_COMPARISON_BYTES,
        THRESHOLD_PROFILE_ARTIFACT_SHA256,
        evidence_profile,
    )
}

pub(crate) fn verify_official_qwen38_acceptance_evidence(
    model_dir: &Path,
) -> Result<OfficialQwen38AcceptanceGateReceiptV1> {
    let profile = super::profile::official_profile()?;
    let authority = official_acceptance_thresholds(&profile)?;
    let thresholds = authority.thresholds;
    let (comparison, receipt) = verify_closed_acceptance_receipts(
        &authority,
        ACCEPTANCE_COMPARISON_BYTES,
        ACCEPTANCE_GATE_BYTES,
    )?;
    let source = super::source::authenticate_official_source(model_dir, &profile)?;
    let prediction =
        super::corpus::build_official_acceptance_prediction_plan(&source, &profile, authority)?;
    gate::validate_closed_acceptance_plan_binding(
        &prediction.plan,
        thresholds,
        &comparison,
        &receipt,
    )?;
    Ok(receipt)
}

fn verify_closed_acceptance_receipts(
    authority: &VerifiedQwen38AcceptanceThresholdsV1,
    comparison_bytes: &[u8],
    gate_bytes: &[u8],
) -> Result<(
    ExactTeacherReferenceComparisonReceiptV1,
    OfficialQwen38AcceptanceGateReceiptV1,
)> {
    ensure!(
        sha256(comparison_bytes) == ACCEPTANCE_COMPARISON_ARTIFACT_SHA256
            && sha256(gate_bytes) == ACCEPTANCE_GATE_ARTIFACT_SHA256,
        "embedded closed AcceptanceHoldout receipt bytes changed"
    );
    gate::validate_closed_acceptance_receipts(authority, comparison_bytes, gate_bytes)
}

fn verify_threshold_bundle(
    profile_bytes: &[u8],
    calibration_bytes: &[u8],
    policy_validation_bytes: &[u8],
    expected_profile_artifact_sha256: &str,
    evidence_profile: &OfficialEvidenceProfileV1,
) -> Result<VerifiedQwen38AcceptanceThresholdsV1> {
    ensure!(
        sha256(profile_bytes) == expected_profile_artifact_sha256,
        "embedded acceptance-threshold profile bytes changed"
    );
    let profile: ThresholdProfileV1 =
        serde_json::from_slice(profile_bytes).context("parse acceptance-threshold profile")?;
    let calibration = validate_exact_teacher_reference_comparison_artifact(calibration_bytes)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let policy = validate_exact_teacher_reference_comparison_artifact(policy_validation_bytes)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;

    let calibration_binding = &profile.characterization.calibration;
    let policy_binding = &profile.characterization.policy_validation;
    ensure!(
        profile.schema_version == 1
            && profile.profile == "qwen38_source_bf16_acceptance_thresholds_v1"
            && profile.source.model_id == evidence_profile.source.repository_id
            && profile.source.revision == evidence_profile.source.revision
            && profile.source.tensor_bundle_sha256 == evidence_profile.source.bundle_sha256
            && profile.source_manifest_sha256 == MANIFEST_SHA256
            && profile.source_manifest_sha256 == evidence_profile.source.manifest_sha256
            && profile.evidence_profile_sha256 == PROFILE_SHA256
            && profile.acceptance_holdout_corpus_sha256
                == evidence_profile.dataset.acceptance_holdout_sha256
            && sha256(calibration_bytes) == calibration_binding.artifact_sha256
            && sha256(policy_validation_bytes) == policy_binding.artifact_sha256
            && calibration.comparison_receipt_sha256
                == calibration_binding.comparison_receipt_sha256
            && policy.comparison_receipt_sha256 == policy_binding.comparison_receipt_sha256
            && calibration.prediction_plan_sha256 == calibration_binding.prediction_plan_sha256
            && policy.prediction_plan_sha256 == policy_binding.prediction_plan_sha256
            && calibration.external_implementation == policy.external_implementation
            && calibration.aggregate.row_count == 22
            && policy.aggregate.row_count == 33
            && calibration.trajectories.len() == 1
            && policy.trajectories.is_empty(),
        "acceptance thresholds differ from their source or characterization receipts"
    );
    validate_threshold_derivation(&profile, &calibration, &policy)?;
    ensure!(
        profile.scope.canary_only
            && !profile.scope.source_teacher_authority
            && !profile.scope.sensitivity_authority
            && !profile.scope.allocator_authority
            && !profile.scope.selector_authority
            && !profile.scope.autoquant_authority
            && !profile.scope.runtime_dependency
            && !profile.scope.dwq,
        "acceptance-threshold scope grants unsupported authority"
    );

    let external_implementation = calibration.external_implementation.clone();
    let plan_authority = bind_teacher_acceptance_thresholds(
        expected_profile_artifact_sha256.into(),
        calibration.comparison_receipt_sha256,
        policy.comparison_receipt_sha256,
        profile.source,
        profile.verified_source_manifest_sha256,
        profile.acceptance_holdout_corpus_sha256,
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(VerifiedQwen38AcceptanceThresholdsV1 {
        plan_authority,
        thresholds: profile.thresholds,
        external_implementation,
    })
}

fn validate_threshold_derivation(
    profile: &ThresholdProfileV1,
    calibration: &ExactTeacherReferenceComparisonReceiptV1,
    policy: &ExactTeacherReferenceComparisonReceiptV1,
) -> Result<()> {
    let derivation = &profile.derivation;
    ensure!(
        derivation.max_abs_rule == "ceil_worst_characterized_max_abs_to_one_decimal"
            && derivation.row_kl_rule == "ceil_worst_characterized_max_kl_to_two_decimals"
            && derivation.top1_rule == "single_holdout_row_must_match"
            && derivation.trajectory_rule == "retain_observed_calibration_first_divergence_index",
        "unsupported acceptance-threshold derivation rule"
    );
    let aggregates = [&calibration.aggregate, &policy.aggregate];
    let worse_max_abs = aggregates
        .iter()
        .map(|value| value.max_abs)
        .fold(0.0, f64::max);
    let worse_max_kl = aggregates
        .iter()
        .map(|value| value.max_kl_reference_to_native)
        .fold(0.0, f64::max);
    let calibration_prefix = trajectory_matching_prefix(&calibration.trajectories)?;
    let expected = Qwen38SourceReferenceThresholdsV1 {
        expected_row_count: 1,
        expected_generation_prompt_count: 1,
        required_trajectory_count: 1,
        max_abs: round_up(worse_max_abs, 0.1),
        max_row_kl_reference_to_native: round_up(worse_max_kl, 0.01),
        require_top1_match: true,
        min_first_divergence_index: calibration_prefix,
    };
    ensure!(
        profile.thresholds == expected
            && characterization_is_within_numeric_ceilings(calibration, profile.thresholds)
            && characterization_is_within_numeric_ceilings(policy, profile.thresholds),
        "acceptance thresholds do not reproduce from both characterization receipts"
    );
    Ok(())
}

fn characterization_is_within_numeric_ceilings(
    comparison: &ExactTeacherReferenceComparisonReceiptV1,
    thresholds: Qwen38SourceReferenceThresholdsV1,
) -> bool {
    comparison.aggregate.max_abs <= thresholds.max_abs
        && comparison.aggregate.max_kl_reference_to_native
            <= thresholds.max_row_kl_reference_to_native
}

fn round_up(value: f64, increment: f64) -> f64 {
    (value / increment).ceil() * increment
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn sha256_json(value: &impl Serialize) -> Result<String> {
    serde_json::to_vec(value)
        .map(|bytes| sha256(&bytes))
        .context("serialize acceptance evidence hash view")
}

#[cfg(test)]
pub(super) fn threshold_evidence_for_test() -> (&'static [u8], &'static [u8], &'static [u8]) {
    (
        THRESHOLD_PROFILE_BYTES,
        CALIBRATION_COMPARISON_BYTES,
        POLICY_VALIDATION_COMPARISON_BYTES,
    )
}

#[cfg(test)]
pub(super) fn verify_threshold_bundle_for_test(
    profile_bytes: &[u8],
    calibration_bytes: &[u8],
    policy_validation_bytes: &[u8],
    evidence_profile: &OfficialEvidenceProfileV1,
) -> Result<VerifiedQwen38AcceptanceThresholdsV1> {
    verify_threshold_bundle(
        profile_bytes,
        calibration_bytes,
        policy_validation_bytes,
        THRESHOLD_PROFILE_ARTIFACT_SHA256,
        evidence_profile,
    )
}

#[cfg(test)]
pub(super) fn closed_acceptance_evidence_for_test() -> (&'static [u8], &'static [u8]) {
    (ACCEPTANCE_COMPARISON_BYTES, ACCEPTANCE_GATE_BYTES)
}

#[cfg(test)]
pub(super) fn verify_closed_acceptance_receipts_for_test(
    authority: &VerifiedQwen38AcceptanceThresholdsV1,
    comparison_bytes: &[u8],
    gate_bytes: &[u8],
) -> Result<()> {
    verify_closed_acceptance_receipts(authority, comparison_bytes, gate_bytes).map(|_| ())
}
