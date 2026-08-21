use super::source::recipe_records;

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
fn embedded_qwen38_recipe_projects_exact_offline_manifest() {
    let recipe = crate::input::model_recipe::embedded_qwen38_recipe().unwrap();
    let records = recipe_records(&recipe);
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
