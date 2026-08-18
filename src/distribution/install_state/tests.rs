use std::fs;
use std::os::unix::fs::symlink;
use std::path::Path;
use std::sync::{mpsc, Arc, Barrier};

use serde_json::json;
use sha2::{Digest, Sha256};

use super::test_fixture::{chmod, copy_directory, Fixture};
use super::*;

fn canonical_marker(value: serde_json::Value) -> Vec<u8> {
    schema::InstalledVersionMarkerV2::parse_and_validate(
        &serde_json::to_vec(&value).expect("marker JSON"),
    )
    .expect("valid marker")
    .to_deterministic_json()
    .expect("canonical marker")
}

fn canonical_receipt(value: serde_json::Value) -> Vec<u8> {
    schema::InstallReceiptV1::parse_and_validate(&serde_json::to_vec(&value).expect("receipt JSON"))
        .expect("valid receipt")
        .to_deterministic_json()
        .expect("canonical receipt")
}

#[test]
fn first_activation_commits_exact_state_and_retry_is_idempotent() {
    let fixture = Fixture::new();
    let FirstActivationPreparation::Ready(prepared) = fixture.prepare().expect("prepare") else {
        panic!("new fixture cannot already be committed");
    };
    assert_eq!(prepared.commit().expect("commit").sequence, 1);
    assert_eq!(
        fs::read_link(fixture.root.join("current")).expect("current link"),
        Path::new(CURRENT_TARGET)
    );
    assert!(!fixture.root.join(PENDING_CURRENT).exists());
    assert!(!fixture
        .root
        .join("activations")
        .join(PENDING_ACTIVATION)
        .exists());

    assert!(matches!(
        fixture.prepare().expect("idempotent prepare"),
        FirstActivationPreparation::AlreadyCommitted { sequence: 1 }
    ));
}

#[test]
fn committed_activation_rejects_pending_current_transaction_cruft() {
    let fixture = Fixture::new();
    let FirstActivationPreparation::Ready(prepared) = fixture.prepare().expect("prepare") else {
        panic!("new fixture cannot already be committed");
    };
    prepared.commit().expect("commit");
    symlink(CURRENT_TARGET, fixture.root.join(PENDING_CURRENT)).expect("add pending current cruft");

    assert!(matches!(
        fixture.prepare(),
        Err(InstallStateError::CommittedDurabilityUnknown { sequence: 1, .. })
    ));
    assert_eq!(
        fs::read_link(fixture.root.join(PENDING_CURRENT)).expect("cruft retained for inspection"),
        Path::new(CURRENT_TARGET)
    );
}

#[test]
fn concurrent_first_activation_is_rejected_without_waiting() {
    let fixture = Fixture::new();
    let FirstActivationPreparation::Ready(_held) = fixture.prepare().expect("first prepare") else {
        panic!("new fixture cannot already be committed");
    };
    assert!(matches!(fixture.prepare(), Err(InstallStateError::Busy)));
    assert!(!fixture.root.join("current").exists());
}

#[test]
fn simultaneous_fresh_bootstrap_yields_one_ready_and_one_busy() {
    let fixture = Fixture::new();
    let start = Arc::new(Barrier::new(3));
    let release_ready = Arc::new(Barrier::new(2));
    let (sender, receiver) = mpsc::channel();
    let mut workers = Vec::new();
    for _ in 0..2 {
        let root = fixture.root.clone();
        let receipt = fixture.receipt_bytes.clone();
        let start = Arc::clone(&start);
        let release_ready = Arc::clone(&release_ready);
        let sender = sender.clone();
        workers.push(std::thread::spawn(move || {
            start.wait();
            let result = prepare_first_activation(
                ExplicitRootAuthorization::new(&root).expect("root authorization"),
                AuthenticatedPreparedVersion::for_test_only(receipt),
            );
            match result {
                Ok(FirstActivationPreparation::Ready(_held)) => {
                    sender.send("ready").expect("send ready");
                    release_ready.wait();
                }
                Err(InstallStateError::Busy) => sender.send("busy").expect("send busy"),
                other => panic!("unexpected bootstrap outcome: {other:?}"),
            }
        }));
    }
    start.wait();
    let mut outcomes = [
        receiver.recv().expect("first outcome"),
        receiver.recv().expect("second outcome"),
    ];
    outcomes.sort_unstable();
    assert_eq!(outcomes, ["busy", "ready"]);
    release_ready.wait();
    for worker in workers {
        worker.join().expect("bootstrap worker");
    }
}

#[test]
fn exact_root_authority_does_not_create_missing_ancestors() {
    let fixture = Fixture::new();
    let parent = fixture
        .root
        .parent()
        .expect("fixture root parent")
        .to_owned();
    fs::remove_dir_all(&fixture.root).expect("remove prepared root");
    fs::remove_dir(&parent).expect("remove now-empty ancestor");

    assert!(matches!(
        fixture.prepare(),
        Err(InstallStateError::Missing("explicit root ancestor"))
    ));
    assert!(!parent.exists());
}

#[test]
fn confirmed_migration_requires_a_separate_future_authorization() {
    let fixture = Fixture::new();
    let mut value: serde_json::Value =
        serde_json::from_slice(&fixture.receipt_bytes).expect("receipt value");
    let transition = value
        .get_mut("last_successful_transition")
        .and_then(serde_json::Value::as_object_mut)
        .expect("transition object");
    transition.insert("type".into(), json!("confirmed-migration"));
    transition.insert(
        "from".into(),
        json!({
            "owner_family": "unknown/manual",
            "release": {
                "version": "0.2.0",
                "target": "aarch64-apple-darwin"
            }
        }),
    );
    let parsed = schema::InstallReceiptV1::parse_and_validate(
        &serde_json::to_vec(&value).expect("migration receipt JSON"),
    )
    .expect("schema-valid migration receipt");
    let migration = parsed
        .to_deterministic_json()
        .expect("canonical migration receipt");

    assert!(matches!(
        prepare_first_activation(
            ExplicitRootAuthorization::new(&fixture.root).expect("root authorization"),
            AuthenticatedPreparedVersion::for_test_only(migration),
        ),
        Err(InstallStateError::InvalidLayout(
            "receipt is not a standalone sequence-one activation"
        ))
    ));
}

#[test]
fn receipt_evidence_and_completion_time_must_be_derived_from_marker() {
    for field in [
        "root_version",
        "timestamp_version",
        "snapshot_version",
        "targets_version",
        "completed_at_unix_seconds",
    ] {
        let fixture = Fixture::new();
        let mut value: serde_json::Value =
            serde_json::from_slice(&fixture.receipt_bytes).expect("receipt value");
        if field == "completed_at_unix_seconds" {
            value["last_successful_transition"][field] = json!(1_787_011_201_u64);
        } else {
            value["last_successful_transition"]["authority"][field] = json!(2);
        }
        let receipt_bytes = canonical_receipt(value);

        assert!(
            matches!(
                prepare_first_activation(
                    ExplicitRootAuthorization::new(&fixture.root).expect("root authorization"),
                    AuthenticatedPreparedVersion::for_test_only(receipt_bytes),
                ),
                Err(InstallStateError::InvalidLayout(
                    "install receipt is not the exact record derived from the installed marker"
                ))
            ),
            "accepted receipt with mismatched {field}"
        );
        assert!(!fixture.root.join("current").exists());
    }
}

#[test]
fn noncanonical_receipt_bytes_cannot_activate() {
    let fixture = Fixture::new();
    let value: serde_json::Value =
        serde_json::from_slice(&fixture.receipt_bytes).expect("receipt value");
    let noncanonical = serde_json::to_vec_pretty(&value).expect("pretty receipt");

    assert!(matches!(
        prepare_first_activation(
            ExplicitRootAuthorization::new(&fixture.root).expect("root authorization"),
            AuthenticatedPreparedVersion::for_test_only(noncanonical),
        ),
        Err(InstallStateError::InvalidLayout(
            "install receipt is not in canonical byte encoding"
        ))
    ));
    assert!(!fixture.root.join("current").exists());
}

#[test]
fn changed_marker_cannot_reuse_stale_receipt_evidence() {
    let fixture = Fixture::new();
    let marker_path = fixture.version.join("version-installation.json");
    let mut marker: serde_json::Value =
        serde_json::from_slice(&fs::read(&marker_path).expect("marker bytes"))
            .expect("marker value");
    marker["prepared_from"]["root_version"] = json!(2);
    marker["installed_at_unix_seconds"] = json!(1_787_011_201_u64);
    let marker_bytes = canonical_marker(marker);
    fs::write(&marker_path, &marker_bytes).expect("replace marker");
    chmod(&marker_path, 0o600);
    let marker_digest = hex::encode(Sha256::digest(&marker_bytes));

    let mut receipt: serde_json::Value =
        serde_json::from_slice(&fixture.receipt_bytes).expect("receipt value");
    receipt["active"]["bundle"]["installed_version_marker_sha256"] = json!(marker_digest.clone());
    receipt["last_successful_transition"]["to"]["release"]["bundle"]
        ["installed_version_marker_sha256"] = json!(marker_digest);
    let receipt_bytes = canonical_receipt(receipt);

    assert!(matches!(
        prepare_first_activation(
            ExplicitRootAuthorization::new(&fixture.root).expect("root authorization"),
            AuthenticatedPreparedVersion::for_test_only(receipt_bytes),
        ),
        Err(InstallStateError::InvalidLayout(
            "install receipt is not the exact record derived from the installed marker"
        ))
    ));
    assert!(!fixture.root.join("current").exists());
}

#[test]
fn unexpected_version_entry_fails_before_current_exists() {
    let fixture = Fixture::new();
    fs::write(fixture.version.join("surprise"), b"not signed").expect("write extra entry");
    chmod(&fixture.version.join("surprise"), 0o644);
    assert!(matches!(
        fixture.prepare(),
        Err(InstallStateError::InvalidLayout(
            "installed version entry inventory is not exact"
        ))
    ));
    assert!(!fixture.root.join("current").exists());
}

#[test]
fn hardlinked_payload_fails_closed() {
    let fixture = Fixture::new();
    fs::hard_link(
        fixture.version.join("bin/hf2q"),
        fixture.root.join("payload-alias"),
    )
    .expect("create hostile hard link");
    assert!(fixture.prepare().is_err());
    assert!(!fixture.root.join("current").exists());
}

#[test]
fn symlinked_payload_fails_closed() {
    let fixture = Fixture::new();
    fs::remove_file(fixture.version.join("bin/hf2q")).expect("remove fixture payload");
    symlink(
        "../share/doc/hf2q/README.md",
        fixture.version.join("bin/hf2q"),
    )
    .expect("create hostile payload symlink");
    assert!(fixture.prepare().is_err());
    assert!(!fixture.root.join("current").exists());
}

#[test]
fn same_size_payload_change_after_prepare_is_reverified() {
    let fixture = Fixture::new();
    let FirstActivationPreparation::Ready(prepared) = fixture.prepare().expect("prepare") else {
        panic!("new fixture cannot already be committed");
    };
    let binary = fixture.version.join("bin/hf2q");
    let original = fs::read(&binary).expect("read fixture binary");
    fs::write(&binary, vec![b'x'; original.len()]).expect("replace payload in place");
    chmod(&binary, 0o755);

    assert!(prepared.commit().is_err());
    assert!(!fixture.root.join("current").exists());
}

#[test]
fn conflicting_current_link_is_never_adopted() {
    let fixture = Fixture::new();
    symlink(
        "activations/99999999999999999999",
        fixture.root.join("current"),
    )
    .expect("create conflicting current");
    assert!(matches!(
        fixture.prepare(),
        Err(InstallStateError::CommittedDurabilityUnknown { sequence: 1, .. })
    ));
}

#[test]
fn exact_precommit_orphans_are_reverified_and_adopted() {
    let fixture = Fixture::new();
    let FirstActivationPreparation::Ready(prepared) = fixture.prepare().expect("prepare") else {
        panic!("new fixture cannot already be committed");
    };
    let (activations, version) = prepared
        .reopen_verified_namespace()
        .expect("reopen namespace");
    prepared
        .publish_or_adopt_activation(&activations, &version)
        .expect("publish activation");
    prepared.stage_current_link().expect("stage current link");
    drop(prepared);

    let FirstActivationPreparation::Ready(recovered) = fixture.prepare().expect("recover") else {
        panic!("precommit state is not committed");
    };
    recovered.commit().expect("adopt exact orphan");
    assert_eq!(
        fs::read_link(fixture.root.join("current")).expect("current link"),
        Path::new(CURRENT_TARGET)
    );
}

#[test]
fn conflicting_partial_activation_is_not_deleted_or_adopted() {
    let fixture = Fixture::new();
    let pending = fixture.root.join("activations").join(PENDING_ACTIVATION);
    fs::create_dir_all(&pending).expect("create pending activation");
    chmod(&fixture.root.join("activations"), 0o700);
    chmod(&pending, 0o700);
    fs::write(pending.join("install-receipt.json"), b"{}\n").expect("write conflicting receipt");
    chmod(&pending.join("install-receipt.json"), 0o600);

    let FirstActivationPreparation::Ready(prepared) = fixture.prepare().expect("prepare") else {
        panic!("new fixture cannot already be committed");
    };
    assert!(prepared.commit().is_err());
    assert_eq!(
        fs::read(pending.join("install-receipt.json")).expect("conflict remains for inspection"),
        b"{}\n"
    );
    assert!(!fixture.root.join("current").exists());
}

#[test]
fn exact_receipt_prefix_is_resumed_and_published() {
    let fixture = Fixture::new();
    let pending = fixture.root.join("activations").join(PENDING_ACTIVATION);
    fs::create_dir_all(&pending).expect("create pending activation");
    chmod(&fixture.root.join("activations"), 0o700);
    chmod(&pending, 0o700);
    let split = fixture.receipt_bytes.len() / 2;
    fs::write(
        pending.join(".install-receipt.json.partial"),
        &fixture.receipt_bytes[..split],
    )
    .expect("write exact receipt prefix");
    chmod(&pending.join(".install-receipt.json.partial"), 0o600);

    let FirstActivationPreparation::Ready(prepared) = fixture.prepare().expect("prepare") else {
        panic!("prefix state is not committed");
    };
    prepared.commit().expect("resume exact prefix");
    assert_eq!(
        fs::read(
            fixture
                .root
                .join("activations")
                .join(FIRST_GENERATION)
                .join("install-receipt.json")
        )
        .expect("committed receipt"),
        fixture.receipt_bytes
    );
    assert!(!fixture
        .root
        .join("activations")
        .join(FIRST_GENERATION)
        .join(".install-receipt.json.partial")
        .exists());
}

#[test]
fn replaced_named_version_is_rejected_before_commit() {
    let fixture = Fixture::new();
    let replacement = fixture.root.join("replacement-version");
    copy_directory(&fixture.version, &replacement);
    let FirstActivationPreparation::Ready(prepared) = fixture.prepare().expect("prepare") else {
        panic!("new fixture cannot already be committed");
    };
    fs::rename(&fixture.version, fixture.root.join("original-version"))
        .expect("move verified version away");
    fs::rename(&replacement, &fixture.version).expect("replace named version");

    assert!(matches!(
        prepared.commit(),
        Err(InstallStateError::InvalidLayout(
            "named prepared version changed after preparation"
        ))
    ));
    assert!(!fixture.root.join("current").exists());
}

#[test]
fn replaced_activations_directory_is_rejected_before_commit() {
    let fixture = Fixture::new();
    let FirstActivationPreparation::Ready(prepared) = fixture.prepare().expect("prepare") else {
        panic!("new fixture cannot already be committed");
    };
    fs::rename(
        fixture.root.join("activations"),
        fixture.root.join("original-activations"),
    )
    .expect("move activations directory");
    fs::create_dir(fixture.root.join("activations")).expect("replace activations directory");
    chmod(&fixture.root.join("activations"), 0o700);

    assert!(matches!(
        prepared.commit(),
        Err(InstallStateError::InvalidLayout(
            "named activations directory changed after preparation"
        ))
    ));
    assert!(!fixture.root.join("current").exists());
}

#[test]
fn swapped_staged_current_link_is_not_committed() {
    let fixture = Fixture::new();
    let FirstActivationPreparation::Ready(prepared) = fixture.prepare().expect("prepare") else {
        panic!("new fixture cannot already be committed");
    };
    let (activations, version) = prepared
        .reopen_verified_namespace()
        .expect("reopen namespace");
    prepared
        .publish_or_adopt_activation(&activations, &version)
        .expect("publish activation");
    prepared.stage_current_link().expect("stage current");
    fs::remove_file(fixture.root.join(PENDING_CURRENT)).expect("remove staged current");
    symlink(
        "activations/99999999999999999999",
        fixture.root.join(PENDING_CURRENT),
    )
    .expect("swap staged current");

    assert!(prepared.commit().is_err());
    assert!(!fixture.root.join("current").exists());
}

#[test]
fn replaced_lock_name_blocks_commit_before_current() {
    let fixture = Fixture::new();
    let FirstActivationPreparation::Ready(prepared) = fixture.prepare().expect("prepare") else {
        panic!("new fixture cannot already be committed");
    };
    let lock = fixture.root.join("update/install.lock");
    fs::rename(&lock, fixture.root.join("update/replaced.lock")).expect("replace lock name");
    fs::write(&lock, b"").expect("create replacement lock");
    chmod(&lock, 0o600);

    assert!(matches!(
        prepared.commit(),
        Err(InstallStateError::InvalidLayout(
            "named entry changed after verification"
        ))
    ));
    assert!(!fixture.root.join("current").exists());
}

#[test]
fn replaced_update_directory_blocks_commit_before_current() {
    let fixture = Fixture::new();
    let FirstActivationPreparation::Ready(prepared) = fixture.prepare().expect("prepare") else {
        panic!("new fixture cannot already be committed");
    };
    fs::rename(
        fixture.root.join("update"),
        fixture.root.join("original-update"),
    )
    .expect("move verified update directory");
    fs::create_dir(fixture.root.join("update")).expect("replace update directory");
    chmod(&fixture.root.join("update"), 0o700);

    assert!(matches!(
        prepared.commit(),
        Err(InstallStateError::InvalidLayout(
            "named update directory changed after preparation"
        ))
    ));
    assert!(!fixture.root.join("current").exists());
}

#[test]
fn symlinked_lock_file_is_never_followed() {
    let fixture = Fixture::new();
    let update = fixture.root.join("update");
    fs::create_dir_all(&update).expect("create update directory");
    chmod(&update, 0o700);
    symlink("../versions", update.join("install.lock")).expect("create hostile lock symlink");
    assert!(fixture.prepare().is_err());
    assert!(!fixture.root.join("current").exists());
}

#[test]
fn postcommit_failure_is_distinct_and_reopen_recovers() {
    let fixture = Fixture::new();
    let FirstActivationPreparation::Ready(prepared) = fixture.prepare().expect("prepare") else {
        panic!("new fixture cannot already be committed");
    };
    FAIL_AFTER_CURRENT_COMMIT.set(true);
    assert!(matches!(
        prepared.commit(),
        Err(InstallStateError::CommittedDurabilityUnknown { sequence: 1, .. })
    ));
    assert_eq!(
        fs::read_link(fixture.root.join("current")).expect("possibly committed current"),
        Path::new(CURRENT_TARGET)
    );
    assert!(matches!(
        fixture.prepare().expect("reopen committed state"),
        FirstActivationPreparation::AlreadyCommitted { sequence: 1 }
    ));
}

#[test]
fn every_recovery_barrier_failure_preserves_committed_classification() {
    let fixture = Fixture::new();
    let FirstActivationPreparation::Ready(prepared) = fixture.prepare().expect("prepare") else {
        panic!("new fixture cannot already be committed");
    };
    prepared.commit().expect("commit");

    for barrier in [
        RecoveryBarrier::ActivationDirectory,
        RecoveryBarrier::ActivationsParent,
        RecoveryBarrier::RootDirectory,
        RecoveryBarrier::ReceiptFullSync,
    ] {
        FAIL_RECOVERY_BARRIER.with(|selected| selected.set(Some(barrier)));
        assert!(matches!(
            fixture.prepare(),
            Err(InstallStateError::CommittedDurabilityUnknown { sequence: 1, .. })
        ));
    }
    assert!(matches!(
        fixture.prepare().expect("repair after injected failures"),
        FirstActivationPreparation::AlreadyCommitted { sequence: 1 }
    ));
}
