use super::*;
use crate::intelligence::calibration::{prediction_plan_for_test, prediction_plan_for_test_bound};
use crate::intelligence::measured_auto_quant::SourceIdentity;
use std::fs::OpenOptions;
use std::os::unix::fs::FileExt;

fn limits() -> TeacherTargetArtifactLimits {
    TeacherTargetArtifactLimits {
        max_vocabulary_size: 8,
        max_prediction_rows: 3,
        max_target_bytes: 4_096,
        top_k: 3,
    }
}

fn private_entries(path: &std::path::Path) -> usize {
    std::fs::read_dir(path).unwrap().count()
}

fn private_target(path: &std::path::Path) -> std::path::PathBuf {
    std::fs::read_dir(path)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|entry| {
            entry
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(".hf2q-exact-teacher-")
        })
        .unwrap()
}

#[test]
fn reservation_is_private_hash_bound_and_drop_cleaned() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("reserved.bin");
    let plan = prediction_plan_for_test();
    let reservation = preflight_structural_teacher_target(&plan, 4, limits())
        .unwrap()
        .reserve(&output)
        .unwrap();

    assert!(!output.exists());
    assert_eq!(private_entries(temp.path()), 1);
    assert_eq!(
        reservation.receipt().prediction_plan_sha256(),
        plan.manifest().manifest_sha256
    );
    assert_eq!(reservation.receipt().vocabulary_size(), 4);
    assert_eq!(reservation.receipt().prediction_point_count(), 3);
    assert_eq!(reservation.receipt().generation_prompt_count(), 1);
    assert!(reservation.receipt().final_artifact_bytes() > 0);
    assert_eq!(reservation.receipt().contract_sha256().len(), 64);

    drop(reservation);
    assert_eq!(private_entries(temp.path()), 0);
    assert!(!output.exists());
}

#[test]
fn reservation_rebinds_only_the_exact_plan_and_preserves_collisions() {
    let temp = tempfile::tempdir().unwrap();
    let wrong_output = temp.path().join("wrong-plan.bin");
    let plan = prediction_plan_for_test();
    let reservation = preflight_structural_teacher_target(&plan, 4, limits())
        .unwrap()
        .reserve(&wrong_output)
        .unwrap();
    let wrong = prediction_plan_for_test_bound(
        SourceIdentity {
            model_id: "Qwen/Qwen3.8-27B".into(),
            revision: "different-revision".into(),
            config_sha256: "1".repeat(64),
            tensor_bundle_sha256: "2".repeat(64),
            tokenizer_bundle_sha256: "3".repeat(64),
            chat_template_sha256: "4".repeat(64),
        },
        "6".repeat(64),
    );
    assert!(reservation.begin(&wrong).is_err());
    assert!(!wrong_output.exists());

    let collision = temp.path().join("collision.bin");
    std::fs::write(&collision, b"competitor").unwrap();
    let rejected = preflight_structural_teacher_target(&plan, 4, limits())
        .unwrap()
        .reserve(&collision);
    assert!(rejected.is_err());
    assert_eq!(std::fs::read(&collision).unwrap(), b"competitor");
}

#[test]
fn reservation_rebind_rechecks_magic_and_late_destination_absence() {
    let temp = tempfile::tempdir().unwrap();
    let plan = prediction_plan_for_test();
    let mutated_output = temp.path().join("mutated.bin");
    let reservation = preflight_structural_teacher_target(&plan, 4, limits())
        .unwrap()
        .reserve(&mutated_output)
        .unwrap();
    OpenOptions::new()
        .write(true)
        .open(private_target(temp.path()))
        .unwrap()
        .write_all_at(b"X", 0)
        .unwrap();
    assert!(reservation.begin(&plan).is_err());
    assert_eq!(private_entries(temp.path()), 0);

    let collision = temp.path().join("late-collision.bin");
    let reservation = preflight_structural_teacher_target(&plan, 4, limits())
        .unwrap()
        .reserve(&collision)
        .unwrap();
    std::fs::write(&collision, b"competitor").unwrap();
    assert!(reservation.begin(&plan).is_err());
    assert_eq!(std::fs::read(&collision).unwrap(), b"competitor");
    assert_eq!(private_entries(temp.path()), 1);
}

#[test]
fn reservation_rebinds_to_the_existing_stream_without_publication() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("stream.bin");
    let plan = prediction_plan_for_test();
    let reservation = preflight_structural_teacher_target(&plan, 4, limits())
        .unwrap()
        .reserve(&output)
        .unwrap();
    let mut stream = reservation.begin(&plan).unwrap();
    for point in &plan.manifest().prediction_points {
        stream.write_row(point, &[0.0, 1.0, 2.0, 3.0]).unwrap();
    }
    let trajectory = (0..EXACT_TEACHER_GREEDY_TOKEN_COUNT as u32)
        .map(|token| token % 4)
        .collect::<Vec<_>>();
    for prompt in &plan.manifest().greedy_prompts {
        stream.write_trajectory(prompt, &trajectory).unwrap();
    }
    let unpublished = stream.finish_unpublished().unwrap();
    assert!(!output.exists());
    assert_eq!(unpublished.receipt().prediction_point_count, 3);
    drop(unpublished);
    assert!(!output.exists());
    assert_eq!(private_entries(temp.path()), 0);
}
