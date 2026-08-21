use std::io::{Seek, SeekFrom, Write};
use std::os::unix::fs::{symlink, MetadataExt};

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
    assert_eq!(canonical_teacher_argmax(&[-0.0, 0.0, -1.0]).unwrap(), 0);
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
    assert_eq!(
        artifact.path(),
        std::fs::canonicalize(output.parent().unwrap())
            .unwrap()
            .join(output.file_name().unwrap())
    );
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
    let unpublished = stream.finish_unpublished().unwrap();
    assert_eq!(unpublished.receipt().prediction_point_count, 3);
    assert!(
        !streaming_path.exists(),
        "a sealed target must remain unpublished until the consuming transition"
    );
    let streaming = unpublished.publish_noclobber().unwrap();
    assert!(streaming_path.exists());
    let named_metadata = std::fs::metadata(&streaming_path).unwrap();
    let retained_metadata = streaming._file.metadata().unwrap();
    assert_eq!(named_metadata.dev(), retained_metadata.dev());
    assert_eq!(named_metadata.ino(), retained_metadata.ino());

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

    let unpublished_path = temp.path().join("drop-unpublished.bin");
    let mut unpublished = target_artifact::preflight_structural_teacher_target(&plan, 4, limits())
        .unwrap()
        .begin(&unpublished_path)
        .unwrap();
    for point in &plan.manifest().prediction_points {
        unpublished
            .write_row(point, &[-0.0, 0.0, -1.0, -2.0])
            .unwrap();
    }
    for prompt in &plan.manifest().greedy_prompts {
        unpublished.write_trajectory(prompt, &trajectory).unwrap();
    }
    drop(unpublished.finish_unpublished().unwrap());
    assert!(!unpublished_path.exists());

    let tampered_path = temp.path().join("tampered-before-publish.bin");
    let mut tampered = target_artifact::preflight_structural_teacher_target(&plan, 4, limits())
        .unwrap()
        .begin(&tampered_path)
        .unwrap();
    for point in &plan.manifest().prediction_points {
        tampered.write_row(point, &[-0.0, 0.0, -1.0, -2.0]).unwrap();
    }
    for prompt in &plan.manifest().greedy_prompts {
        tampered.write_trajectory(prompt, &trajectory).unwrap();
    }
    let mut tampered = tampered.finish_unpublished().unwrap();
    let payload_offset = tampered.receipt().rows[0].payload_offset;
    tampered
        .retained_file_for_test()
        .seek(SeekFrom::Start(payload_offset))
        .unwrap();
    tampered
        .retained_file_for_test()
        .write_all(&1.0f32.to_le_bytes())
        .unwrap();
    tampered.retained_file_for_test().sync_all().unwrap();
    assert!(tampered.publish_noclobber().is_err());
    assert!(!tampered_path.exists());

    let collision_path = temp.path().join("collision.bin");
    std::fs::write(&collision_path, b"existing").unwrap();
    assert!(
        target_artifact::preflight_structural_teacher_target(&plan, 4, limits())
            .unwrap()
            .begin(&collision_path)
            .is_err()
    );
    assert_eq!(std::fs::read(&collision_path).unwrap(), b"existing");

    let late_collision_path = temp.path().join("late-collision.bin");
    let mut collision = target_artifact::preflight_structural_teacher_target(&plan, 4, limits())
        .unwrap()
        .begin(&late_collision_path)
        .unwrap();
    for point in &plan.manifest().prediction_points {
        collision
            .write_row(point, &[-0.0, 0.0, -1.0, -2.0])
            .unwrap();
    }
    for prompt in &plan.manifest().greedy_prompts {
        collision.write_trajectory(prompt, &trajectory).unwrap();
    }
    std::fs::write(&late_collision_path, b"late-existing").unwrap();
    assert!(collision
        .finish_unpublished()
        .unwrap()
        .publish_noclobber()
        .is_err());
    assert_eq!(
        std::fs::read(late_collision_path).unwrap(),
        b"late-existing"
    );

    let original_parent = temp.path().join("parent");
    std::fs::create_dir(&original_parent).unwrap();
    let parent_swap_path = original_parent.join("parent-swap.bin");
    let mut parent_swap = target_artifact::preflight_structural_teacher_target(&plan, 4, limits())
        .unwrap()
        .begin(&parent_swap_path)
        .unwrap();
    for point in &plan.manifest().prediction_points {
        parent_swap
            .write_row(point, &[-0.0, 0.0, -1.0, -2.0])
            .unwrap();
    }
    for prompt in &plan.manifest().greedy_prompts {
        parent_swap.write_trajectory(prompt, &trajectory).unwrap();
    }
    let parent_swap = parent_swap.finish_unpublished().unwrap();
    let moved_parent = temp.path().join("moved-parent");
    std::fs::rename(&original_parent, &moved_parent).unwrap();
    std::fs::create_dir(&original_parent).unwrap();
    assert!(parent_swap.publish_noclobber().is_err());
    assert!(!parent_swap_path.exists());
    assert_eq!(std::fs::read_dir(moved_parent).unwrap().count(), 0);

    let alias_a = temp.path().join("alias-a");
    let alias_b = temp.path().join("alias-b");
    let alias = temp.path().join("alias");
    std::fs::create_dir(&alias_a).unwrap();
    std::fs::create_dir(&alias_b).unwrap();
    symlink(&alias_a, &alias).unwrap();
    let alias_output = alias.join("canonical-target.bin");
    let canonical_output = std::fs::canonicalize(&alias_a)
        .unwrap()
        .join("canonical-target.bin");
    let mut alias_stream = target_artifact::preflight_structural_teacher_target(&plan, 4, limits())
        .unwrap()
        .begin(&alias_output)
        .unwrap();
    for point in &plan.manifest().prediction_points {
        alias_stream
            .write_row(point, &[-0.0, 0.0, -1.0, -2.0])
            .unwrap();
    }
    for prompt in &plan.manifest().greedy_prompts {
        alias_stream.write_trajectory(prompt, &trajectory).unwrap();
    }
    let alias_stream = alias_stream.finish_unpublished().unwrap();
    std::fs::remove_file(&alias).unwrap();
    symlink(&alias_b, &alias).unwrap();
    let alias_artifact = alias_stream.publish_noclobber().unwrap();
    assert_eq!(alias_artifact.path(), canonical_output);
    assert!(canonical_output.exists());
    assert!(!alias_output.exists());
}

#[test]
fn retained_target_file_is_not_redirected_by_final_path_replacement() {
    let temp = tempfile::tempdir().unwrap();
    let plan = prediction_plan_for_test();
    let output = temp.path().join("retained.bin");
    let mut artifact = write_structural_teacher_target_artifact(
        &plan,
        &output,
        4,
        limits(),
        |_request| Ok(vec![0.0, 1.0, 2.0, 3.0]),
        |_request| Ok(vec![0; 32]),
    )
    .unwrap();
    let retained_metadata = artifact._file.metadata().unwrap();
    std::fs::rename(&output, temp.path().join("moved.bin")).unwrap();
    std::fs::write(&output, b"replacement").unwrap();
    let replacement_metadata = std::fs::metadata(&output).unwrap();
    assert_ne!(retained_metadata.ino(), replacement_metadata.ino());
    target_artifact::verify_for_test(&mut artifact).unwrap();
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

#[test]
fn matched_reference_input_and_targets_are_rebuilt_and_compared() {
    let temp = tempfile::tempdir().unwrap();
    let plan = prediction_plan_for_test();
    let input = build_exact_teacher_reference_input(&plan, 4, limits()).unwrap();
    validate_exact_teacher_reference_input(&input).unwrap();

    let native_path = temp.path().join("native.bin");
    let native = write_structural_teacher_target_artifact(
        &plan,
        &native_path,
        4,
        limits(),
        |_request| Ok(vec![0.0, 1.0, 2.0, 3.0]),
        |_request| Ok(vec![3; EXACT_TEACHER_GREEDY_TOKEN_COUNT]),
    )
    .unwrap();
    let reference_path = temp.path().join("reference.bin");
    let reference = write_structural_teacher_target_artifact(
        &plan,
        &reference_path,
        4,
        limits(),
        |_request| Ok(vec![0.0, 1.125, 2.0, 3.0]),
        |_request| Ok(vec![3; EXACT_TEACHER_GREEDY_TOKEN_COUNT]),
    )
    .unwrap();
    let mut external = reference::ExactTeacherExternalReferenceEvidenceV1 {
        schema_version: 1,
        profile: "external_exact_teacher_reference_target_v1".into(),
        reference_input_sha256: input.reference_input_sha256.clone(),
        prediction_plan_sha256: input.prediction_plan.manifest_sha256.clone(),
        target_artifact: crate::core::provenance::tensor_execution::ArtifactEvidence {
            artifact_id: "external_exact_teacher_logits".into(),
            role: "external_full_vocabulary_f32_target_rows".into(),
            byte_len: reference.receipt().target_artifact.byte_len,
            sha256: reference.receipt().target_artifact.sha256.clone(),
        },
        greedy_trajectories: reference.receipt().greedy_trajectories.clone(),
        implementation: reference::ExternalReferenceImplementationV1 {
            name: "fixture".into(),
            repository_url: "https://example.invalid/reference".into(),
            repository_commit: "1".repeat(64),
            package_version: "1.0.0".into(),
            dependency_lock_sha256: "2".repeat(64),
            python_version: "3.12".into(),
            framework_version: "test".into(),
            device: "cpu".into(),
            source_dtype: "f32".into(),
            logit_dtype: "f32_le".into(),
            attention_implementation: "reference".into(),
            cache_enabled: true,
        },
        external_reference: true,
        runtime_dependency: false,
        source_teacher_authority: false,
        sensitivity_authority: false,
        allocator_authority: false,
        selector_authority: false,
        autoquant_authority: false,
        dwq: false,
        evidence_sha256: String::new(),
    };
    external.evidence_sha256 = reference::external_evidence_sha256(&external).unwrap();
    let external: reference::ExactTeacherExternalReferenceEvidenceV1 =
        serde_json::from_slice(&serde_json::to_vec(&external).unwrap()).unwrap();

    let receipt = compare_exact_teacher_reference_targets(
        &plan,
        &input,
        &native_path,
        native.receipt().clone(),
        "3".repeat(64),
        &reference_path,
        &external,
    )
    .unwrap();
    assert_eq!(receipt.rows.len(), 3);
    assert_eq!(receipt.aggregate.top1_match_count, 3);
    assert_eq!(receipt.aggregate.max_abs, 0.125);
    assert!(receipt.aggregate.mean_kl_reference_to_native > 0.0);
    assert!(receipt.trajectories[0].exact_match);
    assert!(!receipt.thresholds_predeclared);
    assert!(!receipt.quality_gate_authority);

    let mut tampered = input.clone();
    tampered.examples[0].token_ids[0] ^= 1;
    assert!(validate_exact_teacher_reference_input(&tampered).is_err());

    use std::os::unix::fs::FileExt;
    std::fs::OpenOptions::new()
        .write(true)
        .open(&reference_path)
        .unwrap()
        .write_all_at(&[1_u8], reference.receipt().rows[0].payload_offset)
        .unwrap();
    assert!(compare_exact_teacher_reference_targets(
        &plan,
        &input,
        &native_path,
        native.receipt().clone(),
        "3".repeat(64),
        &reference_path,
        &external,
    )
    .is_err());
}
