use safetensors::Dtype;

use super::teacher_execution_plan::{
    preflight_qwen35_source_teacher_execution, Qwen35SourceTeacherRunLimitsV1,
};
use super::topology::admit_qwen35_bf16_topology;
use super::topology_tests::{fixture, open};
use crate::intelligence::calibration::{
    policy_prediction_plan_for_test_bound, prediction_plan_for_test,
    prediction_plan_for_test_bound, prediction_plan_for_test_bound_with_first_prefix,
};
use crate::intelligence::exact_teacher::TeacherTargetArtifactLimits;

fn target_limits() -> TeacherTargetArtifactLimits {
    TeacherTargetArtifactLimits {
        max_vocabulary_size: 8,
        max_prediction_rows: 3,
        max_target_bytes: 1024 * 1024,
        top_k: 4,
    }
}

#[test]
fn policy_validation_work_preflight_has_rows_without_greedy_work() -> anyhow::Result<()> {
    let source_fixture = fixture(Dtype::BF16, |_, _| {});
    let topology = admit_qwen35_bf16_topology(open(&source_fixture)?)?;
    let plan = policy_prediction_plan_for_test_bound(
        topology.source().clone(),
        topology.verified_source_manifest_sha256().into(),
    );
    let verified =
        preflight_qwen35_source_teacher_execution(topology, plan, target_limits(), run_limits())?;
    let (_, retained_plan, _, _, expected, _) = verified.into_parts();

    assert_eq!(
        retained_plan.manifest().evaluation_split,
        crate::intelligence::calibration::DatasetSplit::PolicyValidation
    );
    assert_eq!(expected.example_count, 1);
    assert_eq!(expected.completed_transcript_count, 1);
    assert_eq!(expected.generation_prompt_count, 0);
    assert_eq!(expected.prediction_row_count, 2);
    assert_eq!(expected.forward_call_count, 2);
    assert_eq!(expected.input_tokens_processed, 17);
    assert_eq!(expected.output_head_evaluation_count, 2);
    assert_eq!(expected.max_cache_tokens, 17);
    Ok(())
}

fn run_limits() -> Qwen35SourceTeacherRunLimitsV1 {
    Qwen35SourceTeacherRunLimitsV1 {
        max_examples: 2,
        max_forward_calls: 34,
        max_input_tokens_processed: 64,
        max_output_head_evaluations: 34,
        max_cache_tokens: 47,
    }
}

fn bound_inputs() -> anyhow::Result<(
    super::topology::VerifiedQwen35Bf16TopologyV1,
    crate::intelligence::calibration::VerifiedTeacherPredictionPlan,
)> {
    let source_fixture = fixture(Dtype::BF16, |_, _| {});
    let topology = admit_qwen35_bf16_topology(open(&source_fixture)?)?;
    let plan = prediction_plan_for_test_bound(
        topology.source().clone(),
        topology.verified_source_manifest_sha256().into(),
    );
    Ok((topology, plan))
}

#[test]
fn source_bound_work_preflight_reproduces_exact_prefix_and_greedy_counts() -> anyhow::Result<()> {
    let (topology, plan) = bound_inputs()?;
    let expected_plan_sha256 = plan.manifest().manifest_sha256.clone();
    let verified =
        preflight_qwen35_source_teacher_execution(topology, plan, target_limits(), run_limits())?;
    assert_eq!(verified.forward_call_count(), 34);
    assert_eq!(verified.max_cache_tokens(), 47);
    assert_eq!(verified.work_plan_sha256().len(), 64);
    let work_plan_sha256 = verified.work_plan_sha256().to_owned();
    let (_, retained_plan, retained_target_limits, retained_run_limits, expected, retained_hash) =
        verified.into_parts();
    assert_eq!(
        retained_plan.manifest().manifest_sha256,
        expected_plan_sha256
    );
    assert_eq!(retained_target_limits, target_limits());
    assert_eq!(retained_run_limits, run_limits());
    assert_eq!(expected.example_count, 2);
    assert_eq!(expected.completed_transcript_count, 1);
    assert_eq!(expected.generation_prompt_count, 1);
    assert_eq!(expected.prediction_row_count, 3);
    assert_eq!(expected.forward_call_count, 34);
    assert_eq!(expected.input_tokens_processed, 64);
    assert_eq!(expected.output_head_evaluation_count, 34);
    assert_eq!(expected.max_cache_tokens, 47);
    assert_eq!(expected.greedy_token_count_per_prompt, 32);
    assert!(expected.target_artifact_bytes > 0);
    assert_eq!(retained_hash, work_plan_sha256);

    let (topology, plan) = bound_inputs()?;
    let repeated =
        preflight_qwen35_source_teacher_execution(topology, plan, target_limits(), run_limits())?;
    assert_eq!(work_plan_sha256, repeated.work_plan_sha256());
    Ok(())
}

#[test]
fn work_preflight_rejects_source_substitution_and_each_tight_limit() -> anyhow::Result<()> {
    let source_fixture = fixture(Dtype::BF16, |_, _| {});
    let topology = admit_qwen35_bf16_topology(open(&source_fixture)?)?;
    let error = preflight_qwen35_source_teacher_execution(
        topology,
        prediction_plan_for_test(),
        target_limits(),
        run_limits(),
    )
    .err()
    .expect("a different source identity must reject");
    assert!(format!("{error:#}").contains("does not match the retained source topology"));

    let source_fixture = fixture(Dtype::BF16, |_, _| {});
    let topology = admit_qwen35_bf16_topology(open(&source_fixture)?)?;
    let same_source_wrong_manifest =
        prediction_plan_for_test_bound(topology.source().clone(), "f".repeat(64));
    let error = preflight_qwen35_source_teacher_execution(
        topology,
        same_source_wrong_manifest,
        target_limits(),
        run_limits(),
    )
    .err()
    .expect("a different verified-source manifest must reject");
    assert!(format!("{error:#}").contains("does not match the retained source topology"));

    for tighten in 0..5 {
        let (topology, plan) = bound_inputs()?;
        let mut limits = run_limits();
        match tighten {
            0 => limits.max_examples -= 1,
            1 => limits.max_forward_calls -= 1,
            2 => limits.max_input_tokens_processed -= 1,
            3 => limits.max_output_head_evaluations -= 1,
            4 => limits.max_cache_tokens -= 1,
            _ => unreachable!(),
        }
        assert!(
            preflight_qwen35_source_teacher_execution(topology, plan, target_limits(), limits,)
                .is_err()
        );
    }

    let (topology, plan) = bound_inputs()?;
    let mut too_few_target_rows = target_limits();
    too_few_target_rows.max_prediction_rows -= 1;
    let error = preflight_qwen35_source_teacher_execution(
        topology,
        plan,
        too_few_target_rows,
        run_limits(),
    )
    .err()
    .expect("target dimensions must reject inside the pre-upload join");
    assert!(format!("{error:#}").contains("target preflight failed"));

    let (topology, plan) = bound_inputs()?;
    let mut too_small_vocabulary = target_limits();
    too_small_vocabulary.max_vocabulary_size -= 1;
    let error = preflight_qwen35_source_teacher_execution(
        topology,
        plan,
        too_small_vocabulary,
        run_limits(),
    )
    .err()
    .expect("the authenticated vocabulary must fit the target envelope");
    assert!(format!("{error:#}").contains("target preflight failed"));
    Ok(())
}

#[test]
fn work_preflight_rejects_short_fresh_prefix_before_target_or_metal_work() -> anyhow::Result<()> {
    let source_fixture = fixture(Dtype::BF16, |_, _| {});
    let topology = admit_qwen35_bf16_topology(open(&source_fixture)?)?;
    let plan = prediction_plan_for_test_bound_with_first_prefix(
        topology.source().clone(),
        topology.verified_source_manifest_sha256().into(),
        15,
    );
    let error =
        preflight_qwen35_source_teacher_execution(topology, plan, target_limits(), run_limits())
            .err()
            .expect("fresh prefixes shorter than 16 must reject");
    assert!(format!("{error:#}").contains("at least 16 tokens"));
    Ok(())
}

#[test]
fn work_preflight_rejects_work_beyond_authenticated_model_context() -> anyhow::Result<()> {
    let source_fixture = fixture(Dtype::BF16, |config, _| {
        config["text_config"]["max_position_embeddings"] = serde_json::json!(32);
    });
    let topology = admit_qwen35_bf16_topology(open(&source_fixture)?)?;
    let plan = prediction_plan_for_test_bound(
        topology.source().clone(),
        topology.verified_source_manifest_sha256().into(),
    );
    let error =
        preflight_qwen35_source_teacher_execution(topology, plan, target_limits(), run_limits())
            .err()
            .expect("cache work beyond the authenticated context must reject");
    assert!(format!("{error:#}").contains("authenticated model context"));
    Ok(())
}
