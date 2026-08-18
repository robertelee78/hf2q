//! Evidence-backed auto-quantization selection.
//!
//! This module is the fail-closed selection core from ADR-046. It deliberately
//! does not estimate throughput from bit width or nominal memory bandwidth.
//! A candidate is eligible only when its receipt matches the exact source,
//! converter/runtime, host, and workload contract and passes every required
//! quality gate. Selection then ranks eligible candidates by measured inference
//! performance for the contract's primary regime.

use std::collections::BTreeSet;

mod performance;
mod types;

use performance::{check_performance, compare_candidates};
pub use types::*;

/// Select the fastest coherent artifact for an exact workload contract.
///
/// Quality and service-level constraints are hard gates. Among eligible
/// candidates, the primary regime's measured token rate wins; semantic TTFT,
/// artifact size, then candidate id provide deterministic tie-breaks.
pub fn select_fastest_coherent(
    contract: &SelectionContract,
    candidates: &[CandidateReceipt],
) -> Result<SelectionDecision, MeasuredAutoQuantError> {
    validate_contract(contract)?;

    let mut seen = BTreeSet::new();
    for candidate in candidates {
        if !seen.insert(candidate.recipe.candidate_id.clone()) {
            return Err(MeasuredAutoQuantError::DuplicateCandidateId(
                candidate.recipe.candidate_id.clone(),
            ));
        }
    }

    let mut eligible = Vec::new();
    let mut rejected = Vec::new();
    for candidate in candidates {
        let reasons = rejection_reasons(contract, candidate);
        if reasons.is_empty() {
            eligible.push(candidate);
        } else {
            rejected.push(CandidateRejection {
                candidate_id: candidate.recipe.candidate_id.clone(),
                reasons,
            });
        }
    }

    if eligible.is_empty() {
        return Err(MeasuredAutoQuantError::NoEligibleCandidate {
            rejections: rejected,
        });
    }

    eligible.sort_by(|left, right| compare_candidates(contract, left, right));
    let selected = eligible[0];
    let selected_candidate_id = selected.recipe.candidate_id.clone();
    let eligible_candidate_ids = eligible
        .iter()
        .map(|candidate| candidate.recipe.candidate_id.clone())
        .collect();

    Ok(SelectionDecision {
        schema_version: RECEIPT_SCHEMA_VERSION,
        selection_profile_sha256: contract.selection_profile_sha256.clone(),
        source: contract.source.clone(),
        execution: contract.execution.clone(),
        selected_candidate_id,
        selected_artifact_sha256: selected.artifact_sha256.clone(),
        eligible_candidate_ids,
        rejected_candidates: rejected,
    })
}

fn validate_contract(contract: &SelectionContract) -> Result<(), MeasuredAutoQuantError> {
    if !valid_sha256(&contract.selection_profile_sha256)
        || !valid_source_identity(&contract.source)
        || contract.execution.hf2q_revision.is_empty()
        || contract.execution.mlx_native_version.is_empty()
        || contract.execution.hardware_id.is_empty()
        || contract.execution.os_build.is_empty()
    {
        return Err(MeasuredAutoQuantError::InvalidContract(
            "source or execution identity is incomplete".into(),
        ));
    }
    if contract.required_regimes.is_empty() {
        return Err(MeasuredAutoQuantError::InvalidContract(
            "at least one workload regime is required".into(),
        ));
    }
    if !contract
        .required_regimes
        .iter()
        .any(|requirement| requirement.regime == contract.primary_regime)
    {
        return Err(MeasuredAutoQuantError::InvalidContract(
            "primary regime is not a required regime".into(),
        ));
    }

    let mut regimes = BTreeSet::new();
    for requirement in &contract.required_regimes {
        if !regimes.insert(requirement.regime) {
            return Err(MeasuredAutoQuantError::InvalidContract(format!(
                "duplicate workload requirement: {:?}",
                requirement.regime
            )));
        }
        if !valid_sha256(&requirement.workload_sha256)
            || !valid_nonnegative(requirement.min_tokens_per_second)
            || requirement
                .max_semantic_ttft_ms
                .is_some_and(|value| !valid_nonnegative(value))
            || requirement.min_measured_runs == 0
            || requirement.min_tokens_per_run == 0
        {
            return Err(MeasuredAutoQuantError::InvalidContract(format!(
                "invalid threshold for {:?}",
                requirement.regime
            )));
        }
    }

    if !valid_sha256(&contract.quality.quality_suite_sha256)
        || !valid_nonnegative(contract.quality.max_teacher_kl_divergence)
        || !valid_unit_interval(contract.quality.min_top1_token_agreement)
        || !valid_unit_interval(contract.quality.min_activation_cosine_similarity)
        || !valid_positive(contract.quality.max_perplexity_ratio)
        || contract.max_peak_mlx_bytes == 0
    {
        return Err(MeasuredAutoQuantError::InvalidContract(
            "quality or memory thresholds are invalid".into(),
        ));
    }
    if contract
        .quality
        .required_behavioral_regressions
        .iter()
        .any(String::is_empty)
    {
        return Err(MeasuredAutoQuantError::InvalidContract(
            "behavioral regression suite names must not be empty".into(),
        ));
    }
    Ok(())
}

fn rejection_reasons(
    contract: &SelectionContract,
    candidate: &CandidateReceipt,
) -> Vec<CandidateRejectionReason> {
    let mut reasons = Vec::new();
    if candidate.schema_version != RECEIPT_SCHEMA_VERSION {
        reject(
            &mut reasons,
            CandidateRejectionCode::ReceiptSchemaMismatch,
            format!(
                "receipt schema {} does not match {}",
                candidate.schema_version, RECEIPT_SCHEMA_VERSION
            ),
        );
    }
    if candidate.source != contract.source {
        reject(
            &mut reasons,
            CandidateRejectionCode::SourceIdentityMismatch,
            "source identity mismatch",
        );
    }
    if candidate.execution != contract.execution {
        reject(
            &mut reasons,
            CandidateRejectionCode::ExecutionIdentityMismatch,
            "execution identity mismatch",
        );
    }
    if candidate.recipe.candidate_id.is_empty()
        || !valid_sha256(&candidate.artifact_sha256)
        || candidate.artifact_bytes == 0
    {
        reject(
            &mut reasons,
            CandidateRejectionCode::ArtifactIdentityInvalid,
            "artifact identity is incomplete or malformed",
        );
    }
    match &candidate.recipe.encoding {
        WeightEncoding::Gguf { quant_type } if quant_type.is_empty() => {
            reject(
                &mut reasons,
                CandidateRejectionCode::WeightEncodingInvalid,
                "GGUF encoding type is empty",
            );
        }
        WeightEncoding::MlxAffine { bits, group_size }
            if *bits == 0 || *bits > 8 || *group_size == 0 =>
        {
            reject(
                &mut reasons,
                CandidateRejectionCode::WeightEncodingInvalid,
                "MLX affine encoding parameters are invalid",
            );
        }
        _ => {}
    }
    if !valid_sha256(&candidate.recipe.policy_sha256)
        || !valid_sha256(&candidate.recipe.kernel_profile_sha256)
        || !valid_sha256(&candidate.recipe.server_config_sha256)
    {
        reject(
            &mut reasons,
            CandidateRejectionCode::RecipeIdentityInvalid,
            "candidate recipe identity is incomplete or malformed",
        );
    }
    let calibrated = !matches!(
        candidate.recipe.algorithm,
        CalibrationAlgorithm::RoundToNearest
    );
    if calibrated
        && candidate
            .recipe
            .calibration_corpus_sha256
            .as_deref()
            .is_none_or(|hash| !valid_sha256(hash))
    {
        reject(
            &mut reasons,
            CandidateRejectionCode::CalibrationCorpusIdentityInvalid,
            "calibrated candidate is missing its corpus identity",
        );
    }
    if candidate
        .recipe
        .calibration_corpus_sha256
        .as_deref()
        .is_some_and(|hash| !valid_sha256(hash))
    {
        reject(
            &mut reasons,
            CandidateRejectionCode::CalibrationCorpusIdentityInvalid,
            "candidate calibration-corpus identity is malformed",
        );
    }
    if candidate.recipe.algorithm == CalibrationAlgorithm::Dwq
        && candidate
            .recipe
            .teacher_targets_sha256
            .as_deref()
            .is_none_or(|hash| !valid_sha256(hash))
    {
        reject(
            &mut reasons,
            CandidateRejectionCode::TeacherTargetIdentityInvalid,
            "DWQ candidate is missing its teacher-target identity",
        );
    }
    if candidate
        .recipe
        .teacher_targets_sha256
        .as_deref()
        .is_some_and(|hash| !valid_sha256(hash))
    {
        reject(
            &mut reasons,
            CandidateRejectionCode::TeacherTargetIdentityInvalid,
            "candidate teacher-target identity is malformed",
        );
    }

    check_servability(candidate, &mut reasons);
    check_quality(contract, candidate, &mut reasons);
    check_performance(contract, candidate, &mut reasons);
    reasons
}

fn check_servability(candidate: &CandidateReceipt, reasons: &mut Vec<CandidateRejectionReason>) {
    let evidence = &candidate.servability;
    if !evidence.conversion_completed {
        reject(
            reasons,
            CandidateRejectionCode::ConversionGateFailed,
            "conversion completion gate failed",
        );
    }
    if !evidence.artifact_catalog_passed {
        reject(
            reasons,
            CandidateRejectionCode::ArtifactCatalogGateFailed,
            "artifact tensor-catalog gate failed",
        );
    }
    if !evidence.runtime_load_cases.is_complete() {
        reject(
            reasons,
            CandidateRejectionCode::RuntimeLoadGateFailed,
            "runtime-load gate failed or is empty",
        );
    }
    if !evidence.kernel_contract_cases.is_complete() {
        reject(
            reasons,
            CandidateRejectionCode::KernelContractGateFailed,
            "kernel-contract gate failed or is empty",
        );
    }
}

fn check_quality(
    contract: &SelectionContract,
    candidate: &CandidateReceipt,
    reasons: &mut Vec<CandidateRejectionReason>,
) {
    let evidence = &candidate.quality;
    let quality = &contract.quality;

    if evidence.quality_suite_sha256 != quality.quality_suite_sha256
        || !valid_sha256(&evidence.quality_suite_sha256)
    {
        reject(
            reasons,
            CandidateRejectionCode::QualitySuiteIdentityMismatch,
            "quality-suite identity mismatch or malformed digest",
        );
    }
    if !evidence.source_integrity_passed {
        reject(
            reasons,
            CandidateRejectionCode::SourceIntegrityGateFailed,
            "source integrity gate failed",
        );
    }
    if !valid_nonnegative(evidence.teacher_kl_divergence)
        || evidence.teacher_kl_divergence > quality.max_teacher_kl_divergence
    {
        reject(
            reasons,
            CandidateRejectionCode::TeacherKlGateFailed,
            "teacher KL gate failed",
        );
    }
    if !valid_unit_interval(evidence.top1_token_agreement)
        || evidence.top1_token_agreement < quality.min_top1_token_agreement
    {
        reject(
            reasons,
            CandidateRejectionCode::Top1AgreementGateFailed,
            "top-1 token agreement gate failed",
        );
    }
    if !valid_unit_interval(evidence.activation_cosine_similarity)
        || evidence.activation_cosine_similarity < quality.min_activation_cosine_similarity
    {
        reject(
            reasons,
            CandidateRejectionCode::ActivationCosineGateFailed,
            "activation cosine-similarity gate failed",
        );
    }
    if !valid_positive(evidence.perplexity_ratio)
        || evidence.perplexity_ratio > quality.max_perplexity_ratio
    {
        reject(
            reasons,
            CandidateRejectionCode::PerplexityGateFailed,
            "perplexity ratio gate failed",
        );
    }

    check_cases(
        quality.require_tool_calls,
        evidence.tool_call_cases,
        "tool-call",
        reasons,
    );
    check_cases(
        quality.require_context,
        evidence.context_cases,
        "context",
        reasons,
    );
    check_cases(
        quality.require_cache,
        evidence.cache_cases,
        "cache",
        reasons,
    );
    if quality.require_multimodal
        && !evidence
            .multimodal_cases
            .is_some_and(CaseResult::is_complete)
    {
        reject(
            reasons,
            CandidateRejectionCode::MultimodalGateFailed,
            "multimodal gate failed or is missing",
        );
    }

    for name in &quality.required_behavioral_regressions {
        if !evidence
            .behavioral_regressions
            .get(name)
            .copied()
            .is_some_and(CaseResult::is_complete)
        {
            reject(
                reasons,
                CandidateRejectionCode::BehavioralRegressionGateFailed,
                format!("behavioral regression gate failed or is missing: {name}"),
            );
        }
    }
}

fn check_cases(
    required: bool,
    result: CaseResult,
    name: &str,
    reasons: &mut Vec<CandidateRejectionReason>,
) {
    if required && !result.is_complete() {
        reject(
            reasons,
            CandidateRejectionCode::RequiredCaseGateFailed,
            format!("{name} gate failed or is empty"),
        );
    }
}

fn reject(
    reasons: &mut Vec<CandidateRejectionReason>,
    code: CandidateRejectionCode,
    message: impl Into<String>,
) {
    reasons.push(CandidateRejectionReason {
        code,
        message: message.into(),
    });
}

fn valid_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn valid_unit_interval(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn valid_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn valid_source_identity(source: &SourceIdentity) -> bool {
    !source.model_id.is_empty()
        && !source.revision.is_empty()
        && valid_sha256(&source.config_sha256)
        && valid_sha256(&source.tensor_bundle_sha256)
        && valid_sha256(&source.tokenizer_bundle_sha256)
        && valid_sha256(&source.chat_template_sha256)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests;
