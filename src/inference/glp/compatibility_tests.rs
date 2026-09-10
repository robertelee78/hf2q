use super::*;
use std::collections::BTreeMap;

fn checkpoint() -> CheckpointIdentity {
    CheckpointIdentity {
        name: Some("Example-Model".into()),
        organization: Some("Example".into()),
        repository: Some("example/Example-Model".into()),
        revision: Some("a".repeat(40)),
    }
}

fn metadata(values: &[(&str, &str)]) -> BTreeMap<String, MetadataValue> {
    values
        .iter()
        .map(|(key, value)| (key.to_string(), MetadataValue::String(value.to_string())))
        .collect()
}

#[test]
fn matching_checkpoint_is_independent_of_weight_quantization() {
    // No model-byte hash, dtype or quant label enters this contract. The
    // same source checkpoint may produce several differently sized GGUFs.
    assert_eq!(
        checkpoint().validate_for(&checkpoint()).unwrap(),
        Compatibility::Checkpoint
    );
}

#[test]
fn same_family_and_shape_cannot_hide_revision_mismatch() {
    let mut model = checkpoint();
    model.revision = Some("b".repeat(40));
    assert!(checkpoint()
        .validate_for(&model)
        .unwrap_err()
        .to_string()
        .contains("checkpoint mismatch"));
}

#[test]
fn declared_commit_requires_model_revision_evidence() {
    let mut model = checkpoint();
    model.revision = None;
    assert!(checkpoint()
        .validate_for(&model)
        .unwrap_err()
        .to_string()
        .contains("no verifiable checkpoint"));
}

#[test]
fn namespace_cannot_be_discarded_for_same_slug() {
    let mut model = checkpoint();
    model.repository = Some("different-owner/Example-Model".into());
    assert!(checkpoint()
        .validate_for(&model)
        .unwrap_err()
        .to_string()
        .contains("repository mismatch"));
    model.repository = None;
    assert!(checkpoint().validate_for(&model).is_err());
}

#[test]
fn name_comparison_is_not_a_substring_match() {
    let vector = CheckpointIdentity {
        name: Some("Example-Model".into()),
        ..Default::default()
    };
    let model = CheckpointIdentity {
        name: Some("Example-Model-Unrelated-Finetune".into()),
        ..Default::default()
    };
    assert!(vector.validate_for(&model).is_err());
    let model = CheckpointIdentity {
        name: Some("Example Model".into()),
        ..Default::default()
    };
    assert_eq!(vector.validate_for(&model).unwrap(), Compatibility::Named);
}

#[test]
fn legacy_vectors_remain_explicitly_unverified() {
    assert_eq!(
        CheckpointIdentity::default()
            .validate_for(&checkpoint())
            .unwrap(),
        Compatibility::Undeclared
    );
    let mut vector = checkpoint();
    vector.revision = None;
    assert_eq!(
        vector.validate_for(&checkpoint()).unwrap(),
        Compatibility::Named
    );
}

#[test]
fn malformed_glp_commit_cannot_be_a_branch_or_tensor_hash() {
    for value in ["main".to_owned(), "a".repeat(64), "z".repeat(40)] {
        let map = metadata(&[("general.base_model.0.version", &value)]);
        assert!(CheckpointIdentity::from_metadata(|key| map.get(key), true).is_err());
    }
}

#[test]
fn human_model_version_is_not_checkpoint_evidence() {
    let map = metadata(&[("general.base_model.0.version", "v1")]);
    let model = CheckpointIdentity::from_metadata(|key| map.get(key), false).unwrap();
    assert!(model.revision.is_none());
}

#[test]
fn repo_url_and_commit_survive_metadata_read() {
    let revision = "A".repeat(40);
    let map = metadata(&[
        ("general.base_model.0.name", "Example Model"),
        (
            "general.base_model.0.repo_url",
            "https://huggingface.co/example/Example-Model",
        ),
        ("general.base_model.0.version", &revision),
    ]);
    let identity = CheckpointIdentity::from_metadata(|key| map.get(key), true).unwrap();
    assert_eq!(
        identity.repository.as_deref(),
        Some("example/Example-Model")
    );
    assert_eq!(identity.revision, Some(revision.to_lowercase()));
}

#[test]
fn inconsistent_identity_or_wrong_metadata_type_is_rejected() {
    let mut map = metadata(&[
        ("general.base_model.0.name", "other/Example-Model"),
        (
            "general.base_model.0.repo_url",
            "https://huggingface.co/example/Example-Model",
        ),
    ]);
    assert!(CheckpointIdentity::from_metadata(|key| map.get(key), true).is_err());
    map.clear();
    map.insert(
        "general.base_model.0.version".into(),
        MetadataValue::Uint32(1),
    );
    assert!(CheckpointIdentity::from_metadata(|key| map.get(key), true).is_err());
    map.clear();
    map.insert("general.base_model.count".into(), MetadataValue::Uint32(2));
    assert!(CheckpointIdentity::from_metadata(|key| map.get(key), true).is_err());
}

#[test]
fn tensor_content_hash_is_not_used_as_checkpoint_identity() {
    let map = metadata(&[("glp.content_sha256", &"a".repeat(64))]);
    assert_eq!(
        CheckpointIdentity::from_metadata(|key| map.get(key), true).unwrap(),
        CheckpointIdentity::default()
    );
}
