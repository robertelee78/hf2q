use sha2::{Digest, Sha256};
use std::os::unix::fs::{symlink, MetadataExt};

#[test]
fn embedded_qwen38_evidence_profile_is_exact_and_canary_scoped() {
    let profile = super::profile::official_profile().unwrap();
    assert_eq!(
        hex::encode(Sha256::digest(super::profile::profile_bytes_for_test())),
        super::profile::PROFILE_SHA256
    );
    assert_eq!(profile.schema_version, 1);
    assert_eq!(profile.dataset.seed, 380_027);
    assert!(profile.scope.canary_only);
    assert!(!profile.scope.dynamic_calibration_sufficient);
    assert!(!profile.scope.dwq);
}

#[test]
fn embedded_qwen38_source_manifest_projects_exact_offline_inventory() {
    let manifest = super::source_manifest::official_source_manifest().unwrap();
    assert_eq!(
        hex::encode(Sha256::digest(
            super::source_manifest::manifest_bytes_for_test()
        )),
        super::source_manifest::MANIFEST_SHA256
    );
    let records = manifest.records();
    assert_eq!(records.len(), 29);
    assert_eq!(records.iter().filter(|record| record.is_lfs).count(), 19);
    assert_eq!(
        records
            .iter()
            .filter(|record| record.filename.ends_with(".safetensors"))
            .count(),
        18
    );
    assert!(records.iter().all(|record| {
        record.bytes > 0
            && record.is_lfs == record.sha256.is_some()
            && (!record.is_lfs || record.sha256.as_deref() == Some(record.hf_etag.as_str()))
    }));
}

#[test]
fn hf_cache_symlink_is_materialized_as_the_same_regular_inode() {
    let root = tempfile::tempdir().unwrap();
    let model_dir = root.path().join("snapshot");
    let staging = root.path().join("staging");
    std::fs::create_dir(&model_dir).unwrap();
    std::fs::create_dir(&staging).unwrap();
    let blob = root.path().join("blob");
    std::fs::write(&blob, b"exact-source").unwrap();
    symlink(&blob, model_dir.join("config.json")).unwrap();

    super::source::hard_link_source_leaf_for_test(&model_dir, &staging, "config.json", 12).unwrap();
    let linked = staging.join("config.json");
    let blob_metadata = std::fs::metadata(&blob).unwrap();
    let linked_metadata = std::fs::symlink_metadata(&linked).unwrap();
    assert!(linked_metadata.file_type().is_file());
    assert_eq!(
        (linked_metadata.dev(), linked_metadata.ino()),
        (blob_metadata.dev(), blob_metadata.ino())
    );
    std::fs::remove_file(model_dir.join("config.json")).unwrap();
    assert_eq!(std::fs::read(linked).unwrap(), b"exact-source");
    assert!(
        super::source::hard_link_source_leaf_for_test(&model_dir, &staging, "../blob", 12,)
            .is_err()
    );
}

#[test]
fn embedded_qwen38_corpora_are_exact_owned_disjoint_splits() {
    let profile = super::profile::official_profile().unwrap();
    let mut seen = std::collections::BTreeSet::new();
    for (bytes, expected_sha256, split) in super::corpus::embedded_corpus_bytes_for_test() {
        assert_eq!(hex::encode(Sha256::digest(bytes)), expected_sha256);
        let corpus = crate::intelligence::calibration::verify_embedded_calibration_corpus_artifact(
            bytes,
            &crate::intelligence::calibration::VerifyCalibrationCorpusRequest {
                path: format!("<embedded:{split:?}>").into(),
                expected_sha256: expected_sha256.into(),
                expected_dataset_id: profile.dataset.dataset_id.clone(),
                expected_revision: profile.dataset.revision.clone(),
                expected_declared_license: profile.dataset.license.clone(),
                expected_split: split,
                limits: crate::intelligence::calibration::CalibrationCorpusArtifactLimits {
                    max_artifact_bytes: 16 * 1024,
                    max_examples: 2,
                    max_messages: 8,
                    max_tools: 2,
                },
            },
        )
        .unwrap();
        assert_eq!(corpus.manifest().seed, profile.dataset.seed);
        for example in &corpus.manifest().examples {
            assert!(seen.insert(example.stable_id.clone()));
            assert!(!example.enable_thinking);
        }
    }
    assert_eq!(seen.len(), 4);
}

#[test]
fn acceptance_threshold_bundle_is_exact_predeclared_and_substitution_closed() {
    let profile = super::profile::official_profile().unwrap();
    let (threshold_bytes, calibration_bytes, policy_bytes) =
        super::acceptance::threshold_evidence_for_test();
    let thresholds = super::acceptance::verify_threshold_bundle_for_test(
        threshold_bytes,
        calibration_bytes,
        policy_bytes,
        &profile,
    )
    .unwrap();
    assert_eq!(
        thresholds.plan_authority.threshold_profile_sha256(),
        "6a3d36c3006355315820b331aaaeb75bc04ef58b04b81c2be31692b7f99ababb"
    );
    assert_eq!(
        thresholds
            .plan_authority
            .calibration_comparison_receipt_sha256(),
        "41fdf58a53bca32c255951bcc8e9193843afb176c89fdbaee057afadea8bc77d"
    );
    assert_eq!(
        thresholds
            .plan_authority
            .policy_validation_comparison_receipt_sha256(),
        "ed24074db26dde69ccafb6ac797dd77a999000993a26f8eb661b4ac91f1fb919"
    );
    assert_eq!(
        thresholds.external_implementation.repository_commit,
        "945dac9117cb54196888c0e6c08035792a98c485"
    );
    assert_eq!(thresholds.external_implementation.source_dtype, "bfloat16");
    assert_eq!(thresholds.external_implementation.logit_dtype, "f32_le");

    let mut mutated_profile = threshold_bytes.to_vec();
    let threshold_offset = mutated_profile
        .windows(b"\"max_abs\":5.0".len())
        .position(|window| window == b"\"max_abs\":5.0")
        .unwrap();
    mutated_profile[threshold_offset + b"\"max_abs\":5.".len()] = b'1';
    assert!(super::acceptance::verify_threshold_bundle_for_test(
        &mutated_profile,
        calibration_bytes,
        policy_bytes,
        &profile,
    )
    .is_err());

    let mut substituted_calibration = calibration_bytes.to_vec();
    let commit_offset = substituted_calibration
        .windows(b"9b314ce4".len())
        .position(|window| window == b"9b314ce4")
        .unwrap();
    substituted_calibration[commit_offset] = b'8';
    assert!(super::acceptance::verify_threshold_bundle_for_test(
        threshold_bytes,
        &substituted_calibration,
        policy_bytes,
        &profile,
    )
    .is_err());
    assert!(super::acceptance::verify_threshold_bundle_for_test(
        threshold_bytes,
        policy_bytes,
        calibration_bytes,
        &profile,
    )
    .is_err());

    crate::intelligence::exact_teacher::validate_exact_teacher_reference_comparison_artifact(
        calibration_bytes,
    )
    .unwrap();
    let mut rehashed_metric = calibration_bytes.to_vec();
    let metric_offset = rehashed_metric
        .windows(b"4.955787658691406".len())
        .position(|window| window == b"4.955787658691406")
        .unwrap();
    rehashed_metric[metric_offset] = b'3';
    assert!(
        crate::intelligence::exact_teacher::validate_exact_teacher_reference_comparison_artifact(
            &rehashed_metric,
        )
        .is_err()
    );
}

#[test]
fn acceptance_metrics_fail_at_every_predeclared_boundary() {
    let (_, calibration_bytes, policy_bytes) = super::acceptance::threshold_evidence_for_test();
    let mut holdout: crate::intelligence::exact_teacher::ExactTeacherReferenceComparisonReceiptV1 =
        serde_json::from_slice(calibration_bytes).unwrap();
    let policy: crate::intelligence::exact_teacher::ExactTeacherReferenceComparisonReceiptV1 =
        serde_json::from_slice(policy_bytes).unwrap();
    holdout.rows.truncate(1);
    holdout.rows[0].max_abs = 5.0;
    holdout.rows[0].kl_reference_to_native = 0.12;
    holdout.rows[0].top1_match = true;
    let passing = super::acceptance::Qwen38SourceReferenceThresholdsV1 {
        expected_row_count: 1,
        expected_generation_prompt_count: 1,
        required_trajectory_count: 1,
        max_abs: 5.0,
        max_row_kl_reference_to_native: 0.12,
        require_top1_match: true,
        min_first_divergence_index: 10,
    };
    super::acceptance::comparison_passes_thresholds_for_test(&holdout, passing).unwrap();

    let mut above_max_abs = holdout.clone();
    above_max_abs.rows[0].max_abs = f64::from_bits(5.0_f64.to_bits() + 1);
    assert!(
        super::acceptance::comparison_passes_thresholds_for_test(&above_max_abs, passing).is_err()
    );
    let mut above_kl = holdout.clone();
    above_kl.rows[0].kl_reference_to_native = f64::from_bits(0.12_f64.to_bits() + 1);
    assert!(super::acceptance::comparison_passes_thresholds_for_test(&above_kl, passing).is_err());

    let mut failing = passing;
    failing.expected_row_count = 2;
    assert!(super::acceptance::comparison_passes_thresholds_for_test(&holdout, failing).is_err());
    let mut failing = passing;
    failing.required_trajectory_count = 2;
    assert!(super::acceptance::comparison_passes_thresholds_for_test(&holdout, failing).is_err());
    let mut top1_mismatch = holdout.clone();
    top1_mismatch.rows[0].top1_match = false;
    assert!(
        super::acceptance::comparison_passes_thresholds_for_test(&top1_mismatch, passing).is_err()
    );
    let mut failing = passing;
    failing.min_first_divergence_index = 11;
    assert!(super::acceptance::comparison_passes_thresholds_for_test(&holdout, failing).is_err());
    let mut early_divergence = holdout;
    early_divergence.trajectories[0].first_divergence_index = Some(9);
    assert!(
        super::acceptance::comparison_passes_thresholds_for_test(&early_divergence, passing)
            .is_err()
    );
    let mut missing_divergence = early_divergence;
    missing_divergence.trajectories[0].first_divergence_index = None;
    assert!(
        super::acceptance::comparison_passes_thresholds_for_test(&missing_divergence, passing)
            .is_err()
    );
    let mut out_of_range_divergence = missing_divergence;
    out_of_range_divergence.trajectories[0].first_divergence_index = Some(32);
    assert!(super::acceptance::comparison_passes_thresholds_for_test(
        &out_of_range_divergence,
        passing,
    )
    .is_err());
    assert!(
        super::acceptance::comparison_passes_thresholds_for_test(&policy, passing).is_err(),
        "a zero-trajectory split cannot masquerade as AcceptanceHoldout"
    );
}
