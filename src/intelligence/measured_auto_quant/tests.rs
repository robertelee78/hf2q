use std::collections::{BTreeMap, BTreeSet};

use super::*;

fn digest(label: &str) -> String {
    use sha2::{Digest, Sha256};

    format!("{:x}", Sha256::digest(label.as_bytes()))
}

fn source(hash: &str) -> SourceIdentity {
    SourceIdentity {
        model_id: "Qwen/Qwen3.8-27B".into(),
        revision: "source-revision".into(),
        config_sha256: digest("source-config"),
        tensor_bundle_sha256: hash.into(),
        tokenizer_bundle_sha256: digest("source-tokenizer"),
        chat_template_sha256: digest("source-chat-template"),
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
        selection_profile_sha256: digest("selection-profile"),
        source: source(&digest("source-tensors")),
        execution: execution(),
        quality: QualityContract {
            quality_suite_sha256: digest("quality-suite"),
            evaluation_manifest_sha256: digest("evaluation-manifest"),
            deduplication_policy_sha256: digest("deduplication-policy"),
            max_teacher_kl_mean: 0.05,
            max_teacher_kl_p95: 0.10,
            max_teacher_kl_max: 0.20,
            min_teacher_kl_prompts: 300,
            min_teacher_kl_tokens: 9_600,
            min_top1_token_agreement: 0.90,
            min_activation_cosine_similarity: 0.98,
            max_perplexity_ratio: 1.03,
            min_greedy_trajectory_prompts: 300,
            min_greedy_trajectory_tokens_per_prompt: 32,
            min_greedy_trajectory_exact_match_rate: 0.70,
            min_greedy_trajectory_mean_common_prefix_ratio: 0.90,
            require_calibration_evaluation_disjoint: true,
            require_tool_calls: true,
            require_context: true,
            require_cache: true,
            require_multimodal: false,
            required_behavioral_regressions: BTreeSet::from(["owner.source_behavior.v1".into()]),
        },
        required_regimes: vec![
            RegimeRequirement {
                regime: InferenceRegime::TextPrefill,
                workload_sha256: digest("prefill-workload"),
                min_tokens_per_second: 100.0,
                max_semantic_ttft_ms: Some(1_000.0),
                min_warmup_runs: 1,
                min_measured_runs: 5,
                min_tokens_per_run: 256,
            },
            RegimeRequirement {
                regime: InferenceRegime::TextDecodeM1,
                workload_sha256: digest("decode-workload"),
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
    let calibrated = !matches!(algorithm, CalibrationAlgorithm::RoundToNearest);
    CandidateReceipt {
        schema_version: RECEIPT_SCHEMA_VERSION,
        source: source(&digest("source-tensors")),
        execution: execution(),
        recipe: CandidateRecipe {
            candidate_id: id.into(),
            calibration_pipeline: vec![algorithm],
            encoding: WeightEncoding::MlxAffine {
                bits: 4,
                group_size: 64,
            },
            precision_policy_manifest_sha256: digest("precision-policy-manifest"),
            calibration_corpus_sha256: calibrated.then(|| digest("calibration-corpus")),
            calibration_rendered_text_sha256: calibrated
                .then(|| digest("calibration-rendered-text")),
            calibration_token_ids_sha256: calibrated.then(|| digest("calibration-token-ids")),
            calibration_manifest_sha256: calibrated.then(|| digest("calibration-manifest")),
            teacher_targets_sha256: calibrated.then(|| digest("teacher-targets")),
            kernel_profile_sha256: digest("kernel-profile"),
            server_config_sha256: digest("server-config"),
        },
        artifact_sha256: digest("artifact"),
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
            quality_suite_sha256: digest("quality-suite"),
            source_integrity_passed: true,
            teacher_kl: TeacherKlEvidence {
                mean: 0.02,
                p95: 0.04,
                max: 0.08,
                prompt_count: 300,
                token_count: 9_600,
                receipt_sha256: digest("teacher-kl-receipt"),
            },
            top1_token_agreement: 0.95,
            activation_cosine_similarity: 0.99,
            perplexity_ratio: 1.01,
            greedy_trajectory: GreedyTrajectoryEvidence {
                prompt_count: 300,
                tokens_per_prompt: 32,
                exact_match_prompts: 240,
                total_common_prefix_tokens: 9_000,
                receipt_sha256: digest("greedy-trajectory-receipt"),
            },
            dataset_separation: DatasetSeparationEvidence {
                calibration_manifest_sha256: calibrated.then(|| digest("calibration-manifest")),
                evaluation_manifest_sha256: digest("evaluation-manifest"),
                deduplication_policy_sha256: digest("deduplication-policy"),
                overlap_count: 0,
                receipt_sha256: digest("dataset-separation-receipt"),
            },
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
                workload_sha256: digest("prefill-workload"),
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
                workload_sha256: digest("decode-workload"),
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
    assert_eq!(decision.selected_artifact_sha256, digest("artifact"));
    assert_eq!(
        decision.selection_profile_sha256,
        digest("selection-profile")
    );
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
    wrong_source.source = source(&digest("wrong-source-tensors"));

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
    mismatched.performance[1].workload_sha256 = digest("wrong-workload");

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
    invalid.quality.quality_suite_sha256 = digest("wrong-quality-suite");

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
fn multi_token_trajectory_drift_is_not_hidden_by_one_step_quality() {
    let mut drifted = candidate("trajectory-drift", CalibrationAlgorithm::Dwq, 100.0);
    drifted.quality.greedy_trajectory.exact_match_prompts = 10;
    drifted.quality.greedy_trajectory.total_common_prefix_tokens = 1_000;

    let error = select_fastest_coherent(&contract(), &[drifted]).unwrap_err();

    let MeasuredAutoQuantError::NoEligibleCandidate { rejections } = error else {
        panic!("expected a fail-closed no-candidate error");
    };
    assert!(rejections[0]
        .reasons
        .iter()
        .any(|reason| reason.code == CandidateRejectionCode::GreedyTrajectoryGateFailed));
}

#[test]
fn calibration_and_evaluation_overlap_fails_closed() {
    let mut leaked = candidate("leaked-eval", CalibrationAlgorithm::ImportanceMatrix, 100.0);
    leaked.quality.dataset_separation.overlap_count = 1;

    let error = select_fastest_coherent(&contract(), &[leaked]).unwrap_err();

    let MeasuredAutoQuantError::NoEligibleCandidate { rejections } = error else {
        panic!("expected a fail-closed no-candidate error");
    };
    assert!(rejections[0].reasons.iter().any(|reason| {
        reason.code == CandidateRejectionCode::CalibrationEvaluationLeakageGateFailed
    }));
}

#[test]
fn dataset_separation_is_bound_to_the_candidate_calibration_manifest() {
    let mut mismatched = candidate(
        "mismatched-calibration-manifest",
        CalibrationAlgorithm::ImportanceMatrix,
        100.0,
    );
    mismatched
        .quality
        .dataset_separation
        .calibration_manifest_sha256 = Some(digest("wrong-calibration-manifest"));

    let error = select_fastest_coherent(&contract(), &[mismatched]).unwrap_err();

    let MeasuredAutoQuantError::NoEligibleCandidate { rejections } = error else {
        panic!("expected a fail-closed no-candidate error");
    };
    assert!(rejections[0].reasons.iter().any(|reason| {
        reason.code == CandidateRejectionCode::CalibrationEvaluationIdentityMismatch
    }));
}

#[test]
fn evaluation_and_deduplication_identities_are_checked_independently() {
    let mut wrong_evaluation = candidate(
        "wrong-evaluation-manifest",
        CalibrationAlgorithm::ImportanceMatrix,
        100.0,
    );
    wrong_evaluation
        .quality
        .dataset_separation
        .evaluation_manifest_sha256 = digest("wrong-evaluation-manifest");
    let mut wrong_dedup = candidate(
        "wrong-deduplication-policy",
        CalibrationAlgorithm::ImportanceMatrix,
        100.0,
    );
    wrong_dedup
        .quality
        .dataset_separation
        .deduplication_policy_sha256 = digest("wrong-deduplication-policy");

    let error = select_fastest_coherent(&contract(), &[wrong_evaluation, wrong_dedup]).unwrap_err();
    let MeasuredAutoQuantError::NoEligibleCandidate { rejections } = error else {
        panic!("expected a fail-closed no-candidate error");
    };
    assert_eq!(rejections.len(), 2);
    assert!(rejections
        .iter()
        .all(|rejection| rejection
            .reasons
            .iter()
            .any(|reason| reason.code
                == CandidateRejectionCode::CalibrationEvaluationIdentityMismatch)));
}

#[test]
fn evidence_receipt_identities_are_required() {
    let mut bad_kl = candidate("bad-kl-receipt", CalibrationAlgorithm::Dwq, 100.0);
    bad_kl.quality.teacher_kl.receipt_sha256 = "bad".into();
    let mut bad_trajectory = candidate("bad-trajectory-receipt", CalibrationAlgorithm::Dwq, 100.0);
    bad_trajectory.quality.greedy_trajectory.receipt_sha256 = "bad".into();
    let mut bad_separation = candidate("bad-separation-receipt", CalibrationAlgorithm::Dwq, 100.0);
    bad_separation.quality.dataset_separation.receipt_sha256 = "bad".into();

    let error = select_fastest_coherent(&contract(), &[bad_kl, bad_trajectory, bad_separation])
        .unwrap_err();
    let MeasuredAutoQuantError::NoEligibleCandidate { rejections } = error else {
        panic!("expected a fail-closed no-candidate error");
    };
    assert!(rejections[0]
        .reasons
        .iter()
        .any(|reason| reason.code == CandidateRejectionCode::TeacherKlGateFailed));
    assert!(rejections[1]
        .reasons
        .iter()
        .any(|reason| reason.code == CandidateRejectionCode::GreedyTrajectoryGateFailed));
    assert!(rejections[2].reasons.iter().any(|reason| {
        reason.code == CandidateRejectionCode::CalibrationEvaluationIdentityMismatch
    }));
}

#[test]
fn kl_tail_and_evidence_depth_are_hard_gates() {
    let mut invalid = candidate("bad-kl-tail", CalibrationAlgorithm::Dwq, 100.0);
    invalid.quality.teacher_kl.p95 = 0.11;
    invalid.quality.teacher_kl.max = 0.21;
    invalid.quality.teacher_kl.prompt_count = 299;

    let error = select_fastest_coherent(&contract(), &[invalid]).unwrap_err();

    let MeasuredAutoQuantError::NoEligibleCandidate { rejections } = error else {
        panic!("expected a fail-closed no-candidate error");
    };
    assert!(rejections[0]
        .reasons
        .iter()
        .any(|reason| reason.code == CandidateRejectionCode::TeacherKlGateFailed));
}

#[test]
fn every_non_finite_kl_statistic_fails_closed() {
    for (name, field, value) in [
        ("nan-mean", 0, f64::NAN),
        ("infinite-p95", 1, f64::INFINITY),
        ("nan-max", 2, f64::NAN),
    ] {
        let mut invalid = candidate(name, CalibrationAlgorithm::Dwq, 100.0);
        match field {
            0 => invalid.quality.teacher_kl.mean = value,
            1 => invalid.quality.teacher_kl.p95 = value,
            2 => invalid.quality.teacher_kl.max = value,
            _ => unreachable!(),
        }
        let error = select_fastest_coherent(&contract(), &[invalid]).unwrap_err();
        let MeasuredAutoQuantError::NoEligibleCandidate { rejections } = error else {
            panic!("expected a fail-closed no-candidate error");
        };
        assert!(rejections[0]
            .reasons
            .iter()
            .any(|reason| reason.code == CandidateRejectionCode::TeacherKlGateFailed));
    }
}

#[test]
fn calibrated_candidate_binds_rendered_token_stream() {
    let mut unbound = candidate("unbound-rendering", CalibrationAlgorithm::Dwq, 100.0);
    unbound.recipe.calibration_rendered_text_sha256 = None;

    let error = select_fastest_coherent(&contract(), &[unbound]).unwrap_err();

    let MeasuredAutoQuantError::NoEligibleCandidate { rejections } = error else {
        panic!("expected a fail-closed no-candidate error");
    };
    assert!(rejections[0].reasons.iter().any(|reason| {
        reason.code == CandidateRejectionCode::CalibrationRenderingIdentityInvalid
    }));
}

#[test]
fn calibrated_candidate_requires_distinct_text_and_token_identities() {
    let mut missing_tokens = candidate("missing-token-ids", CalibrationAlgorithm::Dwq, 100.0);
    missing_tokens.recipe.calibration_token_ids_sha256 = None;
    let mut missing_manifest = candidate("missing-manifest", CalibrationAlgorithm::Dwq, 100.0);
    missing_manifest.recipe.calibration_manifest_sha256 = None;
    let mut malformed_manifest = candidate("malformed-manifest", CalibrationAlgorithm::Dwq, 100.0);
    malformed_manifest.recipe.calibration_manifest_sha256 = Some("bad".into());

    let error = select_fastest_coherent(
        &contract(),
        &[missing_tokens, missing_manifest, malformed_manifest],
    )
    .unwrap_err();
    let MeasuredAutoQuantError::NoEligibleCandidate { rejections } = error else {
        panic!("expected a fail-closed no-candidate error");
    };
    assert!(rejections[0].reasons.iter().any(|reason| {
        reason.code == CandidateRejectionCode::CalibrationRenderingIdentityInvalid
    }));
    assert!(rejections[1..]
        .iter()
        .all(|rejection| rejection.reasons.iter().any(
            |reason| reason.code == CandidateRejectionCode::CalibrationManifestIdentityInvalid
        )));
}

#[test]
fn uncalibrated_control_does_not_require_a_calibration_rendering() {
    let mut control = candidate("rtn-control", CalibrationAlgorithm::RoundToNearest, 33.0);
    control.recipe.calibration_corpus_sha256 = None;
    control.recipe.calibration_rendered_text_sha256 = None;
    control.recipe.calibration_token_ids_sha256 = None;
    control.recipe.calibration_manifest_sha256 = None;
    control
        .quality
        .dataset_separation
        .calibration_manifest_sha256 = None;

    let decision = select_fastest_coherent(&contract(), &[control]).unwrap();

    assert_eq!(decision.selected_candidate_id, "rtn-control");
}

#[test]
fn round_to_nearest_control_rejects_hidden_calibration_inputs() {
    let mut hidden_corpus = candidate(
        "rtn-hidden-corpus",
        CalibrationAlgorithm::RoundToNearest,
        33.0,
    );
    hidden_corpus.recipe.calibration_corpus_sha256 = Some(digest("hidden-corpus"));
    let mut hidden_text = candidate(
        "rtn-hidden-text",
        CalibrationAlgorithm::RoundToNearest,
        33.0,
    );
    hidden_text.recipe.calibration_rendered_text_sha256 = Some(digest("hidden-text"));
    let mut hidden_tokens = candidate(
        "rtn-hidden-tokens",
        CalibrationAlgorithm::RoundToNearest,
        33.0,
    );
    hidden_tokens.recipe.calibration_token_ids_sha256 = Some(digest("hidden-tokens"));
    let mut hidden_manifest = candidate(
        "rtn-hidden-manifest",
        CalibrationAlgorithm::RoundToNearest,
        33.0,
    );
    hidden_manifest.recipe.calibration_manifest_sha256 = Some(digest("hidden-manifest"));
    let mut hidden_teacher = candidate(
        "rtn-hidden-teacher",
        CalibrationAlgorithm::RoundToNearest,
        33.0,
    );
    hidden_teacher.recipe.teacher_targets_sha256 = Some(digest("hidden-teacher"));

    let error = select_fastest_coherent(
        &contract(),
        &[
            hidden_corpus,
            hidden_text,
            hidden_tokens,
            hidden_manifest,
            hidden_teacher,
        ],
    )
    .unwrap_err();

    let MeasuredAutoQuantError::NoEligibleCandidate { rejections } = error else {
        panic!("expected a fail-closed no-candidate error");
    };
    assert_eq!(rejections.len(), 5);
    assert!(rejections.iter().all(|rejection| rejection
        .reasons
        .iter()
        .any(|reason| reason.code == CandidateRejectionCode::RecipeIdentityInvalid)));
}

#[test]
fn ordered_calibration_pipelines_are_explicit_and_fail_closed() {
    let mut cascade = candidate(
        "dynamic-awq-dwq",
        CalibrationAlgorithm::DynamicMixedPrecision,
        100.0,
    );
    cascade.recipe.calibration_pipeline = vec![
        CalibrationAlgorithm::DynamicMixedPrecision,
        CalibrationAlgorithm::Awq,
        CalibrationAlgorithm::Dwq,
    ];
    assert!(select_fastest_coherent(&contract(), &[cascade]).is_ok());

    let mut mixed_rtn = candidate("mixed-rtn", CalibrationAlgorithm::Dwq, 100.0);
    mixed_rtn.recipe.calibration_pipeline = vec![
        CalibrationAlgorithm::RoundToNearest,
        CalibrationAlgorithm::Dwq,
    ];
    let mut duplicated = candidate("duplicated", CalibrationAlgorithm::Dwq, 100.0);
    duplicated.recipe.calibration_pipeline =
        vec![CalibrationAlgorithm::Dwq, CalibrationAlgorithm::Dwq];
    let mut empty = candidate("empty", CalibrationAlgorithm::Dwq, 100.0);
    empty.recipe.calibration_pipeline.clear();

    let error = select_fastest_coherent(&contract(), &[mixed_rtn, duplicated, empty]).unwrap_err();
    let MeasuredAutoQuantError::NoEligibleCandidate { rejections } = error else {
        panic!("expected a fail-closed no-candidate error");
    };
    assert!(rejections.iter().all(|rejection| rejection
        .reasons
        .iter()
        .any(|reason| reason.code == CandidateRejectionCode::RecipeIdentityInvalid)));
}

#[test]
fn receipt_schema_v2_exposes_new_fields_and_rejects_deprecated_names() {
    let value = serde_json::to_value(candidate(
        "schema-v2",
        CalibrationAlgorithm::DynamicMixedPrecision,
        100.0,
    ))
    .unwrap();
    let recipe = value["recipe"].as_object().unwrap();
    assert!(recipe.contains_key("calibration_pipeline"));
    assert!(recipe.contains_key("precision_policy_manifest_sha256"));
    assert!(recipe.contains_key("calibration_rendered_text_sha256"));
    assert!(recipe.contains_key("calibration_token_ids_sha256"));
    assert!(!recipe.contains_key("algorithm"));
    assert!(!recipe.contains_key("policy_sha256"));
    assert!(value["quality"].get("teacher_kl").is_some());
    assert!(value["quality"].get("teacher_kl_divergence").is_none());

    for (container, field) in [
        ("recipe", "algorithm"),
        ("recipe", "policy_sha256"),
        ("quality", "teacher_kl_divergence"),
    ] {
        let mut stale = value.clone();
        stale[container]
            .as_object_mut()
            .unwrap()
            .insert(field.to_string(), serde_json::json!("deprecated"));
        assert!(serde_json::from_value::<CandidateReceipt>(stale).is_err());
    }
}

#[test]
fn contradictory_trajectory_counts_fail_closed() {
    let mut invalid = candidate("contradictory-trajectory", CalibrationAlgorithm::Dwq, 100.0);
    invalid.quality.greedy_trajectory.exact_match_prompts = 240;
    invalid.quality.greedy_trajectory.total_common_prefix_tokens = 7_000;

    let error = select_fastest_coherent(&contract(), &[invalid]).unwrap_err();

    let MeasuredAutoQuantError::NoEligibleCandidate { rejections } = error else {
        panic!("expected a fail-closed no-candidate error");
    };
    assert!(rejections[0]
        .reasons
        .iter()
        .any(|reason| reason.code == CandidateRejectionCode::GreedyTrajectoryGateFailed));
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
