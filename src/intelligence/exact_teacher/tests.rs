use std::io::{Seek, SeekFrom, Write};

use super::*;
use crate::inference::models::qwen35::{
    forward_cpu::text_positions, model::Qwen35Model, Qwen35Config, Qwen35Variant,
};
use crate::intelligence::calibration::prediction_plan_for_test;

fn limits() -> TeacherTargetArtifactLimits {
    TeacherTargetArtifactLimits {
        max_vocabulary_size: 8,
        max_prediction_rows: 3,
        max_target_bytes: 4_096,
        top_k: 3,
    }
}

#[test]
fn full_logit_rows_are_framed_reread_and_summarized_deterministically() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("teacher-targets.bin");
    let plan = prediction_plan_for_test();
    let mut artifact = write_structural_teacher_target_artifact(
        &plan,
        &output,
        4,
        limits(),
        |request| {
            assert!(request.point.point_ordinal < plan.prediction_point_count());
            Ok(vec![-0.0, 0.0, -1.0, -2.0])
        },
        |_request| {
            Ok((0..EXACT_TEACHER_GREEDY_TOKEN_COUNT as u32)
                .map(|token| token % 4)
                .collect())
        },
    )
    .unwrap();

    assert_eq!(artifact.receipt().rows.len(), 3);
    assert_eq!(artifact.receipt().rows[0].argmax_token_id, 0);
    assert_eq!(
        artifact.receipt().rows[0]
            .top_k
            .iter()
            .map(|entry| entry.token_id)
            .collect::<Vec<_>>(),
        vec![0, 1, 2],
        "numerically equal signed zero logits must break ties by ascending token id"
    );
    assert_eq!(
        artifact.receipt().greedy_trajectories[0].token_ids.len(),
        32
    );
    assert_eq!(artifact.path(), output);
    target_artifact::verify_for_test(&mut artifact).unwrap();
}

#[test]
fn target_writer_rejects_nonfinite_wrong_shape_and_preflight_overflow() {
    let temp = tempfile::tempdir().unwrap();
    let plan = prediction_plan_for_test();
    let wrong_shape = write_structural_teacher_target_artifact(
        &plan,
        &temp.path().join("wrong.bin"),
        4,
        limits(),
        |_request| Ok(vec![0.0; 3]),
        |_request| Ok(vec![0; 32]),
    );
    assert!(matches!(
        wrong_shape,
        Err(ExactTeacherTargetError::Invalid(_))
    ));

    let nonfinite = write_structural_teacher_target_artifact(
        &plan,
        &temp.path().join("nan.bin"),
        4,
        limits(),
        |_request| Ok(vec![0.0, f32::NAN, 1.0, 2.0]),
        |_request| Ok(vec![0; 32]),
    );
    assert!(matches!(
        nonfinite,
        Err(ExactTeacherTargetError::Invalid(_))
    ));

    let mut tiny = limits();
    tiny.max_target_bytes = 1;
    let oversized = write_structural_teacher_target_artifact(
        &plan,
        &temp.path().join("oversized.bin"),
        4,
        tiny,
        |_request| panic!("preflight must reject before calling the producer"),
        |_request| panic!("preflight must reject before calling the producer"),
    );
    assert!(matches!(
        oversized,
        Err(ExactTeacherTargetError::Invalid(_))
    ));
}

#[test]
fn retained_artifact_rejects_payload_mutation_and_wrong_greedy_horizon() {
    let temp = tempfile::tempdir().unwrap();
    let plan = prediction_plan_for_test();
    let short_greedy = write_structural_teacher_target_artifact(
        &plan,
        &temp.path().join("short.bin"),
        4,
        limits(),
        |_request| Ok(vec![0.0, 1.0, 2.0, 3.0]),
        |_request| Ok(vec![0; 31]),
    );
    assert!(matches!(
        short_greedy,
        Err(ExactTeacherTargetError::Invalid(_))
    ));

    let out_of_vocabulary = write_structural_teacher_target_artifact(
        &plan,
        &temp.path().join("bad-token.bin"),
        4,
        limits(),
        |_request| Ok(vec![0.0, 1.0, 2.0, 3.0]),
        |_request| Ok(vec![4; 32]),
    );
    assert!(matches!(
        out_of_vocabulary,
        Err(ExactTeacherTargetError::Invalid(_))
    ));

    let mut one_token_limits = limits();
    one_token_limits.top_k = 1;
    let out_of_vocabulary_target = write_structural_teacher_target_artifact(
        &plan,
        &temp.path().join("bad-target-token.bin"),
        1,
        one_token_limits,
        |_request| panic!("target-token preflight must run before the logit producer"),
        |_request| panic!("target-token preflight must run before the greedy producer"),
    );
    assert!(matches!(
        out_of_vocabulary_target,
        Err(ExactTeacherTargetError::Invalid(_))
    ));

    let output = temp.path().join("mutated.bin");
    let mut artifact = write_structural_teacher_target_artifact(
        &plan,
        &output,
        4,
        limits(),
        |_request| Ok(vec![0.0, 1.0, 2.0, 3.0]),
        |_request| Ok(vec![0; 32]),
    )
    .unwrap();
    let offset = artifact.receipt().rows[0].payload_offset;
    let mut writable = std::fs::OpenOptions::new()
        .write(true)
        .open(&output)
        .unwrap();
    writable.seek(SeekFrom::Start(offset)).unwrap();
    writable.write_all(&1.0f32.to_le_bytes()).unwrap();
    writable.sync_all().unwrap();
    assert!(target_artifact::verify_for_test(&mut artifact).is_err());
}

#[test]
fn streaming_writer_matches_callback_writer_and_rejects_incomplete_or_reordered_work() {
    let temp = tempfile::tempdir().unwrap();
    let plan = prediction_plan_for_test();
    let streaming_path = temp.path().join("streaming.bin");
    let preflight =
        target_artifact::preflight_structural_teacher_target(&plan, 4, limits()).unwrap();
    assert!(
        !streaming_path.exists(),
        "preflight must not publish or allocate the target file"
    );
    let mut stream = preflight.begin(&streaming_path).unwrap();
    for point in &plan.manifest().prediction_points {
        stream.write_row(point, &[-0.0, 0.0, -1.0, -2.0]).unwrap();
    }
    let trajectory = (0..EXACT_TEACHER_GREEDY_TOKEN_COUNT as u32)
        .map(|token| token % 4)
        .collect::<Vec<_>>();
    for prompt in &plan.manifest().greedy_prompts {
        stream.write_trajectory(prompt, &trajectory).unwrap();
    }
    let streaming = stream.finish().unwrap();

    let callback_path = temp.path().join("callback.bin");
    let callback = write_structural_teacher_target_artifact(
        &plan,
        &callback_path,
        4,
        limits(),
        |_request| Ok(vec![-0.0, 0.0, -1.0, -2.0]),
        |_request| Ok(trajectory.clone()),
    )
    .unwrap();
    assert_eq!(streaming.receipt(), callback.receipt());
    assert_eq!(
        std::fs::read(&streaming_path).unwrap(),
        std::fs::read(&callback_path).unwrap()
    );

    let mut reordered = target_artifact::preflight_structural_teacher_target(&plan, 4, limits())
        .unwrap()
        .begin(&temp.path().join("reordered.bin"))
        .unwrap();
    assert!(reordered
        .write_row(&plan.manifest().prediction_points[1], &[0.0, 1.0, 2.0, 3.0],)
        .is_err());

    let incomplete = target_artifact::preflight_structural_teacher_target(&plan, 4, limits())
        .unwrap()
        .begin(&temp.path().join("incomplete.bin"))
        .unwrap();
    assert!(matches!(
        incomplete.finish(),
        Err(ExactTeacherTargetError::Invalid(_))
    ));
}

fn tiny_f32_oracle() -> Qwen35Model {
    let config = Qwen35Config {
        variant: Qwen35Variant::Dense,
        hidden_size: 4,
        num_hidden_layers: 0,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 4,
        linear_num_key_heads: 1,
        linear_num_value_heads: 1,
        linear_key_head_dim: 4,
        linear_value_head_dim: 4,
        linear_conv_kernel_dim: 2,
        full_attention_interval: 1,
        layer_types: Vec::new(),
        partial_rotary_factor: 0.5,
        rope_theta: 10_000.0,
        rotary_dim: 2,
        mrope_section: [1, 0, 0, 0],
        mrope_interleaved: true,
        rms_norm_eps: 1e-6,
        max_position_embeddings: 128,
        vocab_size: 4,
        attn_output_gate: true,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: true,
        intermediate_size: Some(4),
        moe: None,
    };
    Qwen35Model::empty_from_cfg(config)
}

#[test]
fn tiny_qwen_f32_cpu_oracle_streams_completed_full_vocab_rows() {
    let temp = tempfile::tempdir().unwrap();
    let plan = prediction_plan_for_test();
    let model = tiny_f32_oracle();
    let artifact = write_structural_teacher_target_artifact(
        &plan,
        &temp.path().join("tiny-qwen.bin"),
        model.cfg.vocab_size as usize,
        limits(),
        |request| {
            let positions = text_positions(request.prefix_token_ids.len() as u32);
            let logits = model
                .forward_cpu(request.prefix_token_ids, &positions)
                .map_err(|error| ExactTeacherTargetError::Producer(error.to_string()))?;
            let vocabulary = model.cfg.vocab_size as usize;
            Ok(logits[logits.len() - vocabulary..].to_vec())
        },
        |request| {
            assert_eq!(
                request.prompt.prefix_token_ids_sha256,
                crate::intelligence::calibration::prediction_plan_for_test()
                    .manifest()
                    .greedy_prompts[0]
                    .prefix_token_ids_sha256
            );
            let mut tokens = request.prompt_token_ids.to_vec();
            let mut generated = Vec::with_capacity(EXACT_TEACHER_GREEDY_TOKEN_COUNT);
            for _ in 0..EXACT_TEACHER_GREEDY_TOKEN_COUNT {
                let positions = text_positions(tokens.len() as u32);
                let logits = model
                    .forward_cpu(&tokens, &positions)
                    .map_err(|error| ExactTeacherTargetError::Producer(error.to_string()))?;
                let vocabulary = model.cfg.vocab_size as usize;
                let row = &logits[logits.len() - vocabulary..];
                let next = row
                    .iter()
                    .enumerate()
                    .max_by(|(left_id, left), (right_id, right)| {
                        left.total_cmp(right).then_with(|| right_id.cmp(left_id))
                    })
                    .map(|(token_id, _)| token_id as u32)
                    .unwrap();
                tokens.push(next);
                generated.push(next);
            }
            Ok(generated)
        },
    )
    .unwrap();
    assert_eq!(artifact.receipt().rows.len(), plan.prediction_point_count());
    assert!(artifact
        .receipt()
        .rows
        .iter()
        .all(|row| row.argmax_token_id == 0));
}
