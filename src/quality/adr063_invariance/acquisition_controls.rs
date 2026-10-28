//! Controls invoke the same digest, FD and comparison-acquisition boundaries.
use super::*;
fn work() -> Workload {
    Workload {
        family: Family::Qwen38,
        vocab_size: 1000,
        lanes: (0..4)
            .map(|lane| (0..144).map(|i| i + lane * 144).collect())
            .collect(),
    }
}
fn identity() -> Identity {
    Identity {
        family: Family::Qwen38,
        source_commit: "a".repeat(40),
        binary_sha256: "b".repeat(64),
        native_version: env!("HF2Q_MLX_NATIVE_VERSION").into(),
        native_checksum: "c".repeat(64),
        model_sha256: "d".repeat(64),
        tokenizer_sha256: "e".repeat(64),
        template_sha256: "f".repeat(64),
        config_sha256: "1".repeat(64),
        token_vectors_sha256: "2".repeat(64),
    }
}
#[test]
fn q1_acquisition_le_tokens_and_actual_config_changes_cannot_relabel() {
    let original = work();
    let digest = tokens_digest(&original).unwrap();
    let mut changed = original.clone();
    changed.lanes.swap(0, 1);
    assert_ne!(digest, tokens_digest(&changed).unwrap());
    let mut changed = original;
    changed.lanes[0][17] += 1;
    assert_ne!(digest, tokens_digest(&changed).unwrap());
    let mut recipe = ConfigRecipe {
        family: Family::Qwen38,
        capacity: 262144,
        slots: 1,
        output_allowance: 0,
        cache_mode: "QwenTQ8",
        route: "endpoint",
        actual_model_config_sha256: hash(b"actual cfg"),
    };
    let digest = config_digest(&recipe).unwrap();
    recipe.capacity = 144;
    assert_ne!(digest, config_digest(&recipe).unwrap());
    recipe.capacity = 262144;
    recipe.cache_mode = "QwenF32";
    assert_ne!(digest, config_digest(&recipe).unwrap());
}
#[test]
fn q1_acquisition_fd_not_path_or_equal_payload_is_the_binding() {
    use std::{fs::OpenOptions, io::Write, os::unix::fs::OpenOptionsExt};
    let root = std::env::temp_dir().join(format!("q1-fd-control-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let original = root.join("model");
    let replacement = root.join("replacement");
    let mut first = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .mode(0o600)
        .open(&original)
        .unwrap();
    first.write_all(b"same bytes").unwrap();
    let before = FileIdentity::observed(&first).unwrap();
    let mut second = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .mode(0o600)
        .open(&replacement)
        .unwrap();
    second.write_all(b"same bytes").unwrap();
    std::fs::rename(&replacement, &original).unwrap();
    let reopened = File::open(&original).unwrap();
    let old_fd = FileIdentity::observed(&first).unwrap();
    let new_fd = FileIdentity::observed(&reopened).unwrap();
    // Unlink/rename can change ctime of the retained descriptor too: either
    // outcome must fail closed, never substitute equal contents/path.
    assert!(same_opened_file(&before, &new_fd).is_err());
    assert!(same_opened_file(&old_fd, &new_fd).is_err());
    drop((first, second, reopened));
    std::fs::remove_file(&original).unwrap();
    std::fs::remove_dir(&root).unwrap();
}
#[test]
fn q1_acquisition_cross_run_excludes_process_local_device_identity() {
    let i = identity();
    let key = ComparisonKey::from(&i);
    assert_eq!(key, ComparisonKey::from(&i));
    // Device/queue/model addresses live exclusively in Acquisition, never
    // caller-supplied immutable Identity or the cross-process key.
    let mut a = synthetic(Path::new("/one"), &i);
    let before = key;
    a.device_registry_id = 7;
    a.queue_identity = 99;
    a.model_instance = 123;
    assert_eq!(before, ComparisonKey::from(&i));
}

fn trace(root: &Path) -> Trace {
    let identity = identity();
    Trace {
        schema: "hf2q.adr063.invariance.v2".into(),
        comparison_key: (&identity).into(),
        acquisition: Some(synthetic(root, &identity)),
        identity,
        coverage: Coverage {
            expected: vec![],
            captured: vec![],
            errors: vec![],
            missing_native_capabilities: vec![],
            trajectory: "fixture".into(),
        },
    }
}
#[test]
fn q1_acquisition_missing_same_nonce_or_unreviewed_provenance_refuses() {
    let left = trace(Path::new("/left"));
    let right = trace(Path::new("/right"));
    assert_eq!(
        validate_pair_for(
            Path::new("/left"),
            &left,
            Path::new("/right"),
            &right,
            false
        )
        .unwrap(),
        false,
        "synthetic is never model attestation"
    );
    let mut changed = right.clone();
    changed.acquisition = None;
    assert!(validate_pair_for(
        Path::new("/left"),
        &left,
        Path::new("/right"),
        &changed,
        false
    )
    .is_err());
    let mut changed = right.clone();
    changed.acquisition.as_mut().unwrap().nonce = left.acquisition.as_ref().unwrap().nonce.clone();
    assert!(validate_pair_for(
        Path::new("/left"),
        &left,
        Path::new("/right"),
        &changed,
        false
    )
    .is_err());
    let mut changed = right;
    changed.acquisition.as_mut().unwrap().provenance = "IndependentRootReviewedComposition".into();
    assert!(
        validate_pair_for(
            Path::new("/left"),
            &left,
            Path::new("/right"),
            &changed,
            false
        )
        .is_err(),
        "label alone without actual bound evidence cannot qualify"
    );
}

#[test]
fn q1_acquisition_historical_receipt_must_name_exact_full_content_digest() {
    let digest = "a".repeat(64);
    let bytes = serde_json::to_vec(&serde_json::json!({"artifacts":[{"sha256":digest}]})).unwrap();
    assert!(full_content_receipt_has_model(&bytes, &"a".repeat(64)).is_ok());
    assert!(full_content_receipt_has_model(&bytes, &"b".repeat(64)).is_err());
    assert!(full_content_receipt_has_model(b"not JSON", &"a".repeat(64)).is_err());
}
