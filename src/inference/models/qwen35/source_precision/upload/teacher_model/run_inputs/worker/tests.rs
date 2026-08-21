//! End-to-end source-teacher worker falsifiers.

use std::fs::File;
use std::os::unix::fs::FileExt;

use anyhow::{ensure, Result};
use mlx_native::MlxDevice;

use super::super::super::runner::{cpu_model, h256_fixture, last_cpu_logits};
use super::*;
use crate::inference::models::qwen35::source_precision::teacher_execution_plan::{
    preflight_qwen35_source_teacher_execution, Qwen35SourceTeacherRunLimitsV1,
    StructurallyBoundQwen35SourceTeacherWorkV1,
};
use crate::inference::models::qwen35::source_precision::topology::admit_qwen35_bf16_topology;
use crate::inference::models::qwen35::source_precision::topology_tests::open;
use crate::inference::models::qwen35::source_precision::upload::teacher_model::{
    prepare_qwen35_source_teacher_run_inputs, Qwen35SourceTeacherPreparationPolicyV1,
};
use crate::inference::models::qwen35::source_precision::upload_plan::QwenSourceMetalUploadLimits;
use crate::intelligence::calibration::{
    policy_prediction_plan_for_test_bound, prediction_plan_for_test_bound,
    prediction_plan_for_test_bound_with_gap,
};
use crate::intelligence::exact_teacher::{canonical_teacher_argmax, TeacherTargetArtifactLimits};

fn target_limits() -> TeacherTargetArtifactLimits {
    TeacherTargetArtifactLimits {
        max_vocabulary_size: 32,
        max_prediction_rows: 3,
        max_target_bytes: 1024 * 1024,
        top_k: 4,
    }
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

fn preparation_policy() -> Qwen35SourceTeacherPreparationPolicyV1 {
    Qwen35SourceTeacherPreparationPolicyV1 {
        max_cpu_control_mirror_bytes: 1024 * 1024,
        unmeasured_runtime_reserve_bytes: 512 * 1024 * 1024,
    }
}

fn h256_work() -> Result<(
    crate::inference::models::qwen35::source_precision::topology_tests::TopologyFixture,
    StructurallyBoundQwen35SourceTeacherWorkV1,
)> {
    let fixture = h256_fixture();
    let topology = admit_qwen35_bf16_topology(open(&fixture)?)?;
    let prediction_plan = prediction_plan_for_test_bound(
        topology.source().clone(),
        topology.verified_source_manifest_sha256().into(),
    );
    let work = preflight_qwen35_source_teacher_execution(
        topology,
        prediction_plan,
        target_limits(),
        run_limits(),
    )?;
    Ok((fixture, work))
}

fn h256_gap_work() -> Result<(
    crate::inference::models::qwen35::source_precision::topology_tests::TopologyFixture,
    StructurallyBoundQwen35SourceTeacherWorkV1,
)> {
    let fixture = h256_fixture();
    let topology = admit_qwen35_bf16_topology(open(&fixture)?)?;
    let prediction_plan = prediction_plan_for_test_bound_with_gap(
        topology.source().clone(),
        topology.verified_source_manifest_sha256().into(),
    );
    let work = preflight_qwen35_source_teacher_execution(
        topology,
        prediction_plan,
        target_limits(),
        Qwen35SourceTeacherRunLimitsV1 {
            max_examples: 2,
            max_forward_calls: 35,
            max_input_tokens_processed: 65,
            max_output_head_evaluations: 34,
            max_cache_tokens: 47,
        },
    )?;
    Ok((fixture, work))
}

fn h256_policy_work() -> Result<(
    crate::inference::models::qwen35::source_precision::topology_tests::TopologyFixture,
    StructurallyBoundQwen35SourceTeacherWorkV1,
)> {
    let fixture = h256_fixture();
    let topology = admit_qwen35_bf16_topology(open(&fixture)?)?;
    let prediction_plan = policy_prediction_plan_for_test_bound(
        topology.source().clone(),
        topology.verified_source_manifest_sha256().into(),
    );
    let work = preflight_qwen35_source_teacher_execution(
        topology,
        prediction_plan,
        target_limits(),
        run_limits(),
    )?;
    Ok((fixture, work))
}

fn prepare_h256_inputs(
    output: &std::path::Path,
) -> Result<(
    crate::inference::models::qwen35::source_precision::topology_tests::TopologyFixture,
    PreparedQwen35SourceTeacherRunInputsV1,
)> {
    let (fixture, work) = h256_work()?;
    let device = MlxDevice::new()?;
    let inputs = prepare_qwen35_source_teacher_run_inputs(
        work,
        output,
        &device,
        QwenSourceMetalUploadLimits::default(),
        preparation_policy(),
    )?;
    Ok((fixture, inputs))
}

fn expected_cpu_outputs(
    model: &crate::inference::models::qwen35::model::Qwen35Model,
    plan: &VerifiedTeacherPredictionPlan,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<u32>>)> {
    let mut rows = Vec::new();
    let mut trajectories = Vec::new();
    plan.visit_examples(|example, tokens, points, greedy| -> Result<()> {
        match example.render_mode {
            RenderMode::CompletedAssistantTranscript => {
                ensure!(greedy.is_none());
                for point in points {
                    rows.push(last_cpu_logits(model, &tokens[..point.prefix_token_count]));
                }
            }
            RenderMode::GenerationPrompt => {
                ensure!(points.len() == 1 && greedy.is_some());
                let mut prefix = tokens.to_vec();
                let mut trajectory = Vec::with_capacity(EXACT_TEACHER_GREEDY_TOKEN_COUNT);
                for step in 0..EXACT_TEACHER_GREEDY_TOKEN_COUNT {
                    let logits = last_cpu_logits(model, &prefix);
                    if step == 0 {
                        rows.push(logits.clone());
                    }
                    let token = canonical_teacher_argmax(&logits)?;
                    trajectory.push(token);
                    if step + 1 < EXACT_TEACHER_GREEDY_TOKEN_COUNT {
                        prefix.push(token);
                    }
                }
                trajectories.push(trajectory);
            }
        }
        Ok(())
    })?;
    Ok((rows, trajectories))
}

fn read_target_row(
    file: &File,
    row: &crate::intelligence::exact_teacher::TeacherTargetRowReceipt,
) -> Result<Vec<f32>> {
    let mut bytes = vec![0_u8; usize::try_from(row.payload_bytes)?];
    file.read_exact_at(&mut bytes, row.payload_offset)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
        .collect())
}

fn assert_row_parity(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} width");
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "{label} finite"
    );
    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    let actual_top = canonical_teacher_argmax(actual).unwrap();
    let expected_top = canonical_teacher_argmax(expected).unwrap();
    assert_eq!(actual_top, expected_top, "{label} top-1");
    assert!(max_abs <= 5.0e-3, "{label} max_abs={max_abs}");
}

#[test]
fn production_worker_completes_exact_plan_and_publishes_retained_target() -> Result<()> {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let output_dir = tempfile::tempdir()?;
    let destination = output_dir.path().join("completed-target.bin");
    let (fixture, inputs) = prepare_h256_inputs(&destination)?;
    let model = cpu_model(&inputs._teacher);
    let retained_plan = inputs._prediction_plan.clone();
    let (expected_rows, expected_trajectories) = expected_cpu_outputs(&model, &retained_plan)?;
    drop(fixture);

    let completed = run_qwen35_source_teacher(inputs)?;
    assert_eq!(completed.path(), destination.canonicalize()?);
    assert!(destination.is_file());
    assert_eq!(completed.target_artifact_sha256().len(), 64);
    assert_eq!(completed.completion_receipt_sha256().len(), 64);
    super::receipt::validate_completion_receipt(&completed.receipt)?;
    let mut corrupt_policy = completed.receipt.clone();
    corrupt_policy.corrupt_graph_policy_for_test();
    assert!(super::receipt::validate_completion_receipt(&corrupt_policy).is_err());
    let mut short_completion = completed.receipt.clone();
    short_completion.decrement_completion_for_test();
    assert!(super::receipt::validate_completion_receipt(&short_completion).is_err());
    let (expected_work, observed_work) = completed.receipt.work_for_test();
    assert_eq!(observed_work, observed_from_expected(expected_work));
    assert_eq!(observed_work.forward_call_count, 34);
    assert_eq!(observed_work.input_tokens_processed, 64);
    assert_eq!(observed_work.output_head_evaluation_count, 34);
    assert_eq!(observed_work.prediction_row_count, 3);
    assert_eq!(observed_work.max_cache_tokens, 47);

    let structural = completed.target.receipt();
    assert_eq!(structural.rows.len(), expected_rows.len());
    assert_eq!(
        structural.greedy_trajectories.len(),
        expected_trajectories.len()
    );
    let file = File::open(completed.path())?;
    for (index, (row, expected)) in structural.rows.iter().zip(&expected_rows).enumerate() {
        let actual = read_target_row(&file, row)?;
        assert_eq!(row.argmax_token_id, canonical_teacher_argmax(&actual)?);
        assert_row_parity(&actual, expected, &format!("target row {index}"));
    }
    for (actual, expected) in structural
        .greedy_trajectories
        .iter()
        .zip(&expected_trajectories)
    {
        assert_eq!(&actual.token_ids, expected);
        assert_eq!(actual.token_ids.len(), EXACT_TEACHER_GREEDY_TOKEN_COUNT);
    }
    Ok(())
}

#[test]
fn production_worker_completes_policy_rows_without_a_trajectory() -> Result<()> {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let output_dir = tempfile::tempdir()?;
    let destination = output_dir.path().join("policy-target.bin");
    let (fixture, work) = h256_policy_work()?;
    let device = MlxDevice::new()?;
    let inputs = prepare_qwen35_source_teacher_run_inputs(
        work,
        &destination,
        &device,
        QwenSourceMetalUploadLimits::default(),
        preparation_policy(),
    )?;
    let model = cpu_model(&inputs._teacher);
    let retained_plan = inputs._prediction_plan.clone();
    let (expected_rows, expected_trajectories) = expected_cpu_outputs(&model, &retained_plan)?;
    drop(fixture);

    let completed = run_qwen35_source_teacher(inputs)?;
    let structural = completed.target.receipt();
    assert_eq!(structural.rows.len(), 2);
    assert_eq!(structural.rows.len(), expected_rows.len());
    assert!(expected_trajectories.is_empty());
    assert_eq!(structural.generation_prompt_count, 0);
    assert!(structural.greedy_trajectories.is_empty());
    let file = File::open(completed.path())?;
    for (index, (row, expected)) in structural.rows.iter().zip(&expected_rows).enumerate() {
        let actual = read_target_row(&file, row)?;
        assert_row_parity(&actual, expected, &format!("policy target row {index}"));
    }
    Ok(())
}

#[test]
fn retained_source_mutation_rejects_before_execution_and_publication() -> Result<()> {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let output_dir = tempfile::tempdir()?;
    let destination = output_dir.path().join("mutated-source.bin");
    let (fixture, inputs) = prepare_h256_inputs(&destination)?;
    let shard = fixture.temp.path().join("model.safetensors");
    let mut bytes = std::fs::read(&shard)?;
    let last = bytes.last_mut().unwrap();
    *last ^= 0x1;
    std::fs::write(&shard, bytes)?;

    let error = run_qwen35_source_teacher(inputs)
        .err()
        .expect("mutated source must reject");
    let rendered = format!("{error:#}");
    assert!(
        rendered.contains("retained source shard") && rendered.contains("changed"),
        "{rendered}"
    );
    assert!(!destination.exists());
    assert_eq!(std::fs::read_dir(output_dir.path())?.count(), 0);
    Ok(())
}

#[test]
fn worker_panic_after_real_call_drains_and_leaves_target_private() -> Result<()> {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let output_dir = tempfile::tempdir()?;
    let destination = output_dir.path().join("panic.bin");
    let (_fixture, inputs) = prepare_h256_inputs(&destination)?;
    let error = run_qwen35_source_teacher_with_behavior_for_test(
        inputs,
        WorkerBehavior {
            panic_after_completed_call: Some(1),
            error_drain_directory: Some(output_dir.path().to_path_buf()),
        },
        || Ok(()),
    )
    .err()
    .expect("injected worker panic must reject");
    assert!(format!("{error:#}").contains("execution panicked"));
    assert!(!destination.exists());
    assert_eq!(std::fs::read_dir(output_dir.path())?.count(), 0);
    Ok(())
}

#[test]
fn collision_after_worker_join_preserves_competitor_and_mints_no_authority() -> Result<()> {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let output_dir = tempfile::tempdir()?;
    let destination = output_dir.path().join("late-collision.bin");
    let (_fixture, inputs) = prepare_h256_inputs(&destination)?;
    let error =
        run_qwen35_source_teacher_with_behavior_for_test(inputs, WorkerBehavior::default(), || {
            std::fs::write(&destination, b"competitor")?;
            Ok(())
        })
        .err()
        .expect("late destination collision must reject");
    assert!(format!("{error:#}").contains("destination"));
    assert_eq!(std::fs::read(&destination)?, b"competitor");
    assert_eq!(std::fs::read_dir(output_dir.path())?.count(), 1);
    Ok(())
}

#[test]
fn completed_transcript_gap_advances_cache_without_emitting_an_extra_row() -> Result<()> {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let output_dir = tempfile::tempdir()?;
    let destination = output_dir.path().join("gapped-target.bin");
    let (_fixture, work) = h256_gap_work()?;
    let device = MlxDevice::new()?;
    let inputs = prepare_qwen35_source_teacher_run_inputs(
        work,
        &destination,
        &device,
        QwenSourceMetalUploadLimits::default(),
        preparation_policy(),
    )?;
    let completed = run_qwen35_source_teacher(inputs)?;
    let (_, observed) = completed.receipt.work_for_test();
    assert_eq!(observed.forward_call_count, 35);
    assert_eq!(observed.input_tokens_processed, 65);
    assert_eq!(observed.output_head_evaluation_count, 34);
    assert_eq!(observed.prediction_row_count, 3);
    assert_eq!(completed.target.receipt().rows.len(), 3);
    Ok(())
}
