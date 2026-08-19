use std::collections::{BTreeMap, BTreeSet};

use super::*;

fn digest(nibble: char) -> String {
    nibble.to_string().repeat(64)
}

fn source(hash: &str) -> SourceIdentity {
    SourceIdentity {
        model_id: "Qwen/Qwen3.8-27B".into(),
        revision: "source-revision".into(),
        config_sha256: digest('a'),
        tensor_bundle_sha256: hash.into(),
        tokenizer_bundle_sha256: digest('b'),
        chat_template_sha256: digest('c'),
    }
}

fn execution() -> ExecutionIdentity {
    ExecutionIdentity {
        hf2q_revision: "hf2q-revision".into(),
        mlx_native_version: "0.10.10".into(),
        hardware_id: "apple-m5-max-128g".into(),
        os_build: "macos-build".into(),
    }
}

fn contract() -> SelectionContract {
    SelectionContract {
        selection_profile_sha256: digest('0'),
        source: source(&digest('d')),
        execution: execution(),
        quality: QualityContract {
            quality_suite_sha256: digest('9'),
            max_teacher_kl_divergence: 0.05,
            min_top1_token_agreement: 0.90,
            min_activation_cosine_similarity: 0.98,
            max_perplexity_ratio: 1.03,
            require_tool_calls: true,
            require_context: true,
            require_cache: true,
            require_multimodal: false,
            required_behavioral_regressions: BTreeSet::from(["owner.source_behavior.v1".into()]),
        },
        required_regimes: vec![
            RegimeRequirement {
                regime: InferenceRegime::TextPrefill,
                workload_sha256: digest('e'),
                min_tokens_per_second: 100.0,
                max_semantic_ttft_ms: Some(1_000.0),
                min_warmup_runs: 1,
                min_measured_runs: 5,
                min_tokens_per_run: 256,
            },
            RegimeRequirement {
                regime: InferenceRegime::TextDecodeM1,
                workload_sha256: digest('f'),
                min_tokens_per_second: 20.0,
                max_semantic_ttft_ms: Some(1_000.0),
                min_warmup_runs: 1,
                min_measured_runs: 5,
                min_tokens_per_run: 256,
            },
        ],
        primary_regime: InferenceRegime::TextDecodeM1,
        max_peak_mlx_bytes: 32 * 1024 * 1024 * 1024,
    }
}

fn candidate(id: &str, algorithm: CalibrationAlgorithm, decode_tps: f64) -> CandidateReceipt {
    CandidateReceipt {
        schema_version: RECEIPT_SCHEMA_VERSION,
        source: source(&digest('d')),
        execution: execution(),
        recipe: CandidateRecipe {
            candidate_id: id.into(),
            algorithm,
            encoding: WeightEncoding::MlxAffine {
                bits: 4,
                group_size: 64,
            },
            policy_sha256: digest('1'),
            calibration_corpus_sha256: Some(digest('2')),
            teacher_targets_sha256: Some(digest('3')),
            kernel_profile_sha256: digest('4'),
            server_config_sha256: digest('5'),
        },
        artifact_sha256: digest('6'),
        artifact_bytes: 16_000_000_000,
        servability: ServabilityEvidence {
            conversion_completed: true,
            artifact_catalog_passed: true,
            runtime_load_cases: CaseResult {
                passed: 2,
                total: 2,
            },
            kernel_contract_cases: CaseResult {
                passed: 5,
                total: 5,
            },
        },
        quality: QualityEvidence {
            quality_suite_sha256: digest('9'),
            source_integrity_passed: true,
            teacher_kl_divergence: 0.02,
            top1_token_agreement: 0.95,
            activation_cosine_similarity: 0.99,
            perplexity_ratio: 1.01,
            tool_call_cases: CaseResult {
                passed: 8,
                total: 8,
            },
            context_cases: CaseResult {
                passed: 4,
                total: 4,
            },
            cache_cases: CaseResult {
                passed: 3,
                total: 3,
            },
            multimodal_cases: None,
            behavioral_regressions: BTreeMap::from([(
                "owner.source_behavior.v1".into(),
                CaseResult {
                    passed: 100,
                    total: 100,
                },
            )]),
        },
        performance: vec![
            PerformanceMeasurement {
                regime: InferenceRegime::TextPrefill,
                workload_sha256: digest('e'),
                median_tokens_per_second: 500.0,
                median_semantic_ttft_ms: 800.0,
                peak_mlx_bytes: 24_000_000_000,
                warmup_runs: 2,
                measured_runs: 7,
                tokens_per_run: 4_096,
                output_quality_cases: CaseResult {
                    passed: 2,
                    total: 2,
                },
            },
            PerformanceMeasurement {
                regime: InferenceRegime::TextDecodeM1,
                workload_sha256: digest('f'),
                median_tokens_per_second: decode_tps,
                median_semantic_ttft_ms: 850.0,
                peak_mlx_bytes: 24_000_000_000,
                warmup_runs: 2,
                measured_runs: 7,
                tokens_per_run: 512,
                output_quality_cases: CaseResult {
                    passed: 2,
                    total: 2,
                },
            },
        ],
    }
}

#[test]
fn fastest_candidate_wins_only_after_all_gates_pass() {
    let rtn = candidate("rtn-q4", CalibrationAlgorithm::RoundToNearest, 29.0);
    let dwq = candidate("dwq-q4", CalibrationAlgorithm::Dwq, 33.0);

    let decision = select_fastest_coherent(&contract(), &[rtn, dwq]).unwrap();

    assert_eq!(decision.selected_candidate_id, "dwq-q4");
    assert_eq!(decision.selected_artifact_sha256, digest('6'));
    assert_eq!(decision.selection_profile_sha256, digest('0'));
    assert_eq!(decision.source, contract().source);
    assert_eq!(decision.eligible_candidate_ids, ["dwq-q4", "rtn-q4"]);
}

#[test]
fn faster_candidate_is_rejected_when_source_behavior_drifts() {
    let good = candidate("coherent", CalibrationAlgorithm::RoundToNearest, 29.0);
    let mut drifted = candidate("drifted", CalibrationAlgorithm::Dwq, 40.0);
    drifted.quality.behavioral_regressions.clear();

    let decision = select_fastest_coherent(&contract(), &[good, drifted]).unwrap();

    assert_eq!(decision.selected_candidate_id, "coherent");
    assert_eq!(decision.rejected_candidates[0].candidate_id, "drifted");
    assert_eq!(
        decision.rejected_candidates[0].reasons[0].code,
        CandidateRejectionCode::BehavioralRegressionGateFailed
    );
}

#[test]
fn converter_output_that_the_runtime_cannot_load_is_ineligible() {
    let good = candidate("servable", CalibrationAlgorithm::RoundToNearest, 29.0);
    let mut unservable = candidate("unservable", CalibrationAlgorithm::Dwq, 100.0);
    unservable.servability.runtime_load_cases = CaseResult {
        passed: 0,
        total: 1,
    };

    let decision = select_fastest_coherent(&contract(), &[good, unservable]).unwrap();

    assert_eq!(decision.selected_candidate_id, "servable");
    assert_eq!(
        decision.rejected_candidates[0].reasons[0].code,
        CandidateRejectionCode::RuntimeLoadGateFailed
    );
}

#[test]
fn modified_and_vanilla_sources_use_the_same_exact_hash_rule() {
    let good = candidate("exact", CalibrationAlgorithm::Dwq, 33.0);
    let mut wrong_source = candidate("other-source", CalibrationAlgorithm::Dwq, 50.0);
    wrong_source.source = source(&digest('7'));

    let decision = select_fastest_coherent(&contract(), &[good, wrong_source]).unwrap();

    assert_eq!(decision.selected_candidate_id, "exact");
    assert_eq!(decision.rejected_candidates[0].reasons.len(), 1);
    assert_eq!(
        decision.rejected_candidates[0].reasons[0].code,
        CandidateRejectionCode::SourceIdentityMismatch
    );
}

#[test]
fn incomplete_execution_regime_evidence_fails_closed() {
    let mut incomplete = candidate("incomplete", CalibrationAlgorithm::Dwq, 33.0);
    incomplete
        .performance
        .retain(|measurement| measurement.regime != InferenceRegime::TextPrefill);

    let error = select_fastest_coherent(&contract(), &[incomplete]).unwrap_err();

    let MeasuredAutoQuantError::NoEligibleCandidate { rejections } = error else {
        panic!("expected a fail-closed no-candidate error");
    };
    assert_eq!(
        rejections[0].reasons[0].code,
        CandidateRejectionCode::MissingPerformanceRegime
    );
}

#[test]
fn cached_evidence_is_bound_to_runtime_and_hardware() {
    let mut stale = candidate("stale", CalibrationAlgorithm::Dwq, 100.0);
    stale.execution.mlx_native_version = "0.10.9".into();

    let error = select_fastest_coherent(&contract(), &[stale]).unwrap_err();

    let MeasuredAutoQuantError::NoEligibleCandidate { rejections } = error else {
        panic!("expected a fail-closed no-candidate error");
    };
    assert_eq!(rejections[0].reasons.len(), 1);
    assert_eq!(
        rejections[0].reasons[0].code,
        CandidateRejectionCode::ExecutionIdentityMismatch
    );
}

#[test]
fn benchmark_evidence_is_bound_to_the_exact_workload() {
    let mut mismatched = candidate("mismatched", CalibrationAlgorithm::Dwq, 100.0);
    mismatched.performance[1].workload_sha256 = digest('8');

    let error = select_fastest_coherent(&contract(), &[mismatched]).unwrap_err();

    let MeasuredAutoQuantError::NoEligibleCandidate { rejections } = error else {
        panic!("expected a fail-closed no-candidate error");
    };
    assert_eq!(
        rejections[0].reasons[0].code,
        CandidateRejectionCode::WorkloadIdentityMismatch
    );
}

#[test]
fn receipt_schema_round_trips_without_losing_identity() {
    let original = candidate("round-trip", CalibrationAlgorithm::Dwq, 33.0);

    let json = serde_json::to_string(&original).unwrap();
    let restored: CandidateReceipt = serde_json::from_str(&json).unwrap();

    assert_eq!(restored, original);
}

#[test]
fn non_finite_measurement_fails_closed() {
    let mut invalid = candidate("nan", CalibrationAlgorithm::Dwq, f64::NAN);
    invalid.performance[1].median_semantic_ttft_ms = f64::INFINITY;

    let error = select_fastest_coherent(&contract(), &[invalid]).unwrap_err();

    let MeasuredAutoQuantError::NoEligibleCandidate { rejections } = error else {
        panic!("expected a fail-closed no-candidate error");
    };
    assert!(rejections[0]
        .reasons
        .iter()
        .any(|reason| reason.code == CandidateRejectionCode::InvalidPerformanceMeasurement));
}

#[test]
fn missing_peak_memory_measurement_fails_closed() {
    let mut invalid = candidate("zero-memory", CalibrationAlgorithm::Dwq, 33.0);
    invalid.performance[1].peak_mlx_bytes = 0;

    let error = select_fastest_coherent(&contract(), &[invalid]).unwrap_err();

    let MeasuredAutoQuantError::NoEligibleCandidate { rejections } = error else {
        panic!("expected a fail-closed no-candidate error");
    };
    assert!(rejections[0]
        .reasons
        .iter()
        .any(|reason| reason.code == CandidateRejectionCode::InvalidPerformanceMeasurement));
}

#[test]
fn quality_evidence_is_bound_to_exact_suite() {
    let mut invalid = candidate("wrong-quality-suite", CalibrationAlgorithm::Dwq, 33.0);
    invalid.quality.quality_suite_sha256 = digest('a');

    let error = select_fastest_coherent(&contract(), &[invalid]).unwrap_err();

    let MeasuredAutoQuantError::NoEligibleCandidate { rejections } = error else {
        panic!("expected a fail-closed no-candidate error");
    };
    assert!(rejections[0]
        .reasons
        .iter()
        .any(|reason| reason.code == CandidateRejectionCode::QualitySuiteIdentityMismatch));
}

#[test]
fn invalid_affine_encoding_fails_closed() {
    let mut invalid = candidate("invalid-affine", CalibrationAlgorithm::Dwq, 33.0);
    invalid.recipe.encoding = WeightEncoding::MlxAffine {
        bits: 0,
        group_size: 0,
    };

    let error = select_fastest_coherent(&contract(), &[invalid]).unwrap_err();

    let MeasuredAutoQuantError::NoEligibleCandidate { rejections } = error else {
        panic!("expected a fail-closed no-candidate error");
    };
    assert!(rejections[0]
        .reasons
        .iter()
        .any(|reason| reason.code == CandidateRejectionCode::WeightEncodingInvalid));
}

#[test]
fn output_quality_and_warmup_are_required_evidence() {
    let mut invalid = candidate("shallow-benchmark", CalibrationAlgorithm::Dwq, 33.0);
    invalid.performance[1].warmup_runs = 0;
    invalid.performance[1].output_quality_cases = CaseResult {
        passed: 0,
        total: 0,
    };

    let error = select_fastest_coherent(&contract(), &[invalid]).unwrap_err();

    let MeasuredAutoQuantError::NoEligibleCandidate { rejections } = error else {
        panic!("expected a fail-closed no-candidate error");
    };
    assert!(rejections[0]
        .reasons
        .iter()
        .any(|reason| reason.code == CandidateRejectionCode::InvalidPerformanceMeasurement));
    assert!(rejections[0]
        .reasons
        .iter()
        .any(|reason| reason.code == CandidateRejectionCode::InsufficientBenchmarkEvidence));
}

#[test]
fn ttft_sla_can_be_explicitly_omitted() {
    let mut contract = contract();
    for requirement in &mut contract.required_regimes {
        requirement.max_semantic_ttft_ms = None;
    }
    let mut candidate = candidate("no-ttft-sla", CalibrationAlgorithm::Dwq, 33.0);
    for measurement in &mut candidate.performance {
        measurement.median_semantic_ttft_ms = 1_000_000.0;
    }

    let decision = select_fastest_coherent(&contract, &[candidate]).unwrap();

    assert_eq!(decision.selected_candidate_id, "no-ttft-sla");
}

#[test]
fn primary_regime_must_be_part_of_the_workload_contract() {
    let mut invalid = contract();
    invalid.primary_regime = InferenceRegime::TextDecodeWidthN;

    assert!(matches!(
        select_fastest_coherent(&invalid, &[]),
        Err(MeasuredAutoQuantError::InvalidContract(_))
    ));
}

#[test]
fn malformed_digest_is_not_accepted_as_exact_identity() {
    let mut invalid = candidate("bad-hash", CalibrationAlgorithm::Dwq, 33.0);
    invalid.recipe.kernel_profile_sha256 = "not-a-sha256".into();

    let error = select_fastest_coherent(&contract(), &[invalid]).unwrap_err();

    let MeasuredAutoQuantError::NoEligibleCandidate { rejections } = error else {
        panic!("expected a fail-closed no-candidate error");
    };
    assert!(rejections[0]
        .reasons
        .iter()
        .any(|reason| reason.code == CandidateRejectionCode::RecipeIdentityInvalid));
}
