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
        || !valid_sha256(&contract.quality.evaluation_manifest_sha256)
        || !valid_sha256(&contract.quality.deduplication_policy_sha256)
        || !valid_nonnegative(contract.quality.max_teacher_kl_mean)
        || !valid_nonnegative(contract.quality.max_teacher_kl_p95)
        || !valid_nonnegative(contract.quality.max_teacher_kl_max)
        || contract.quality.max_teacher_kl_mean > contract.quality.max_teacher_kl_max
        || contract.quality.max_teacher_kl_p95 > contract.quality.max_teacher_kl_max
        || contract.quality.min_teacher_kl_prompts == 0
        || contract.quality.min_teacher_kl_tokens == 0
        || !valid_unit_interval(contract.quality.min_top1_token_agreement)
        || !valid_unit_interval(contract.quality.min_activation_cosine_similarity)
        || !valid_positive(contract.quality.max_perplexity_ratio)
        || contract.quality.min_greedy_trajectory_prompts == 0
        || contract.quality.min_greedy_trajectory_tokens_per_prompt == 0
        || !valid_unit_interval(contract.quality.min_greedy_trajectory_exact_match_rate)
        || !valid_unit_interval(
            contract
                .quality
                .min_greedy_trajectory_mean_common_prefix_ratio,
        )
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
    if !valid_sha256(&candidate.recipe.precision_policy_manifest_sha256)
        || !valid_sha256(&candidate.recipe.kernel_profile_sha256)
        || !valid_sha256(&candidate.recipe.server_config_sha256)
    {
        reject(
            &mut reasons,
            CandidateRejectionCode::RecipeIdentityInvalid,
            "candidate recipe identity is incomplete or malformed",
        );
    }
    let pipeline = &candidate.recipe.calibration_pipeline;
    let pipeline_has_duplicate = pipeline
        .iter()
        .enumerate()
        .any(|(index, stage)| pipeline[..index].contains(stage));
    let round_to_nearest_control = pipeline.as_slice() == [CalibrationAlgorithm::RoundToNearest];
    let pipeline_valid = !pipeline.is_empty()
        && !pipeline_has_duplicate
        && (round_to_nearest_control || !pipeline.contains(&CalibrationAlgorithm::RoundToNearest));
    if !pipeline_valid {
        reject(
            &mut reasons,
            CandidateRejectionCode::RecipeIdentityInvalid,
            "calibration pipeline is empty, duplicated, or mixes round-to-nearest with calibrated stages",
        );
    }
    let calibrated = pipeline_valid && !round_to_nearest_control;
    if !calibrated
        && (candidate.recipe.calibration_corpus_sha256.is_some()
            || candidate.recipe.calibration_rendered_text_sha256.is_some()
            || candidate.recipe.calibration_token_ids_sha256.is_some()
            || candidate.recipe.calibration_manifest_sha256.is_some()
            || candidate.recipe.teacher_targets_sha256.is_some())
    {
        reject(
            &mut reasons,
            CandidateRejectionCode::RecipeIdentityInvalid,
            "round-to-nearest control carries unexpected calibration identity",
        );
    }
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
    if calibrated
        && candidate
            .recipe
            .calibration_rendered_text_sha256
            .as_deref()
            .is_none_or(|hash| !valid_sha256(hash))
    {
        reject(
            &mut reasons,
            CandidateRejectionCode::CalibrationRenderingIdentityInvalid,
            "calibrated candidate is missing its rendered calibration identity",
        );
    }
    if calibrated
        && candidate
            .recipe
            .calibration_token_ids_sha256
            .as_deref()
            .is_none_or(|hash| !valid_sha256(hash))
    {
        reject(
            &mut reasons,
            CandidateRejectionCode::CalibrationRenderingIdentityInvalid,
            "calibrated candidate is missing its tokenized calibration identity",
        );
    }
    if calibrated
        && candidate
            .recipe
            .calibration_manifest_sha256
            .as_deref()
            .is_none_or(|hash| !valid_sha256(hash))
    {
        reject(
            &mut reasons,
            CandidateRejectionCode::CalibrationManifestIdentityInvalid,
            "calibrated candidate is missing its calibration-manifest identity",
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
    if candidate
        .recipe
        .calibration_rendered_text_sha256
        .as_deref()
        .is_some_and(|hash| !valid_sha256(hash))
    {
        reject(
            &mut reasons,
            CandidateRejectionCode::CalibrationRenderingIdentityInvalid,
            "candidate rendered calibration identity is malformed",
        );
    }
    if candidate
        .recipe
        .calibration_token_ids_sha256
        .as_deref()
        .is_some_and(|hash| !valid_sha256(hash))
    {
        reject(
            &mut reasons,
            CandidateRejectionCode::CalibrationRenderingIdentityInvalid,
            "candidate tokenized calibration identity is malformed",
        );
    }
    if candidate
        .recipe
        .calibration_manifest_sha256
        .as_deref()
        .is_some_and(|hash| !valid_sha256(hash))
    {
        reject(
            &mut reasons,
            CandidateRejectionCode::CalibrationManifestIdentityInvalid,
            "candidate calibration-manifest identity is malformed",
        );
    }
    if pipeline.contains(&CalibrationAlgorithm::Dwq)
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
    let calibrated =
        candidate.recipe.calibration_pipeline.as_slice() != [CalibrationAlgorithm::RoundToNearest];
    let separation = &evidence.dataset_separation;
    let calibration_identity_matches = if calibrated {
        separation.calibration_manifest_sha256 == candidate.recipe.calibration_manifest_sha256
            && separation
                .calibration_manifest_sha256
                .as_deref()
                .is_some_and(valid_sha256)
    } else {
        separation.calibration_manifest_sha256.is_none()
    };
    if !calibration_identity_matches
        || separation.evaluation_manifest_sha256 != quality.evaluation_manifest_sha256
        || separation.deduplication_policy_sha256 != quality.deduplication_policy_sha256
        || !valid_sha256(&separation.evaluation_manifest_sha256)
        || !valid_sha256(&separation.deduplication_policy_sha256)
        || !valid_sha256(&separation.receipt_sha256)
    {
        reject(
            reasons,
            CandidateRejectionCode::CalibrationEvaluationIdentityMismatch,
            "calibration/evaluation split identity mismatch or malformed digest",
        );
    }
    if !evidence.source_integrity_passed {
        reject(
            reasons,
            CandidateRejectionCode::SourceIntegrityGateFailed,
            "source integrity gate failed",
        );
    }
    let teacher_kl = &evidence.teacher_kl;
    if !valid_nonnegative(teacher_kl.mean)
        || !valid_nonnegative(teacher_kl.p95)
        || !valid_nonnegative(teacher_kl.max)
        || teacher_kl.mean > teacher_kl.max
        || teacher_kl.p95 > teacher_kl.max
        || teacher_kl.mean > quality.max_teacher_kl_mean
        || teacher_kl.p95 > quality.max_teacher_kl_p95
        || teacher_kl.max > quality.max_teacher_kl_max
        || teacher_kl.prompt_count < quality.min_teacher_kl_prompts
        || teacher_kl.token_count < quality.min_teacher_kl_tokens
        || !valid_sha256(&teacher_kl.receipt_sha256)
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

    let trajectory = &evidence.greedy_trajectory;
    let max_prefix_tokens =
        u64::from(trajectory.prompt_count).checked_mul(u64::from(trajectory.tokens_per_prompt));
    let minimum_prefix_tokens_from_exact_matches = u64::from(trajectory.exact_match_prompts)
        .checked_mul(u64::from(trajectory.tokens_per_prompt));
    let trajectory_shape_valid = trajectory.prompt_count > 0
        && trajectory.tokens_per_prompt > 0
        && trajectory.exact_match_prompts <= trajectory.prompt_count
        && max_prefix_tokens.is_some_and(|max| trajectory.total_common_prefix_tokens <= max)
        && minimum_prefix_tokens_from_exact_matches
            .is_some_and(|min| trajectory.total_common_prefix_tokens >= min);
    let exact_match_rate = if trajectory.prompt_count == 0 {
        f64::NAN
    } else {
        f64::from(trajectory.exact_match_prompts) / f64::from(trajectory.prompt_count)
    };
    let mean_common_prefix_ratio = match max_prefix_tokens {
        Some(0) | None => f64::NAN,
        Some(max) => trajectory.total_common_prefix_tokens as f64 / max as f64,
    };
    if !trajectory_shape_valid
        || trajectory.prompt_count < quality.min_greedy_trajectory_prompts
        || trajectory.tokens_per_prompt < quality.min_greedy_trajectory_tokens_per_prompt
        || !valid_unit_interval(exact_match_rate)
        || exact_match_rate < quality.min_greedy_trajectory_exact_match_rate
        || !valid_unit_interval(mean_common_prefix_ratio)
        || mean_common_prefix_ratio < quality.min_greedy_trajectory_mean_common_prefix_ratio
        || !valid_sha256(&trajectory.receipt_sha256)
    {
        reject(
            reasons,
            CandidateRejectionCode::GreedyTrajectoryGateFailed,
            "fixed-horizon greedy trajectory gate failed or is malformed",
        );
    }
    if quality.require_calibration_evaluation_disjoint && separation.overlap_count != 0 {
        reject(
            reasons,
            CandidateRejectionCode::CalibrationEvaluationLeakageGateFailed,
            "calibration/evaluation disjointness gate failed",
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
