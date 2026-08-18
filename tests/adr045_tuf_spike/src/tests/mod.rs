//! Adversarial acceptance corpus for the spike.

use url::Url;

use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::os::unix::fs::PermissionsExt;

use crate::application_binding::verify_application_binding;
use crate::candidates::{
    attempt_from_fixture, committed_from_fixture, verify_sigstore_core, verify_tough_attempt,
    AttemptConfig, UntrustedRole,
};
use crate::capture_transport::{CapturingTransport, FetchOutcome, ScriptedResponse};
use crate::generation_journal::{
    commit_candidate, read_selected_sequence, CommitOutcome, FaultPlan,
};
use crate::model::{SpikeError, MAX_TIMESTAMP_BYTES};
use crate::test_repository::build_expired_lower_repository;
use crate::test_repository::{
    build_case_collision_manifest_repository, build_delegated_repository,
    build_duplicate_signature_repository, build_embedded_manifest_mismatch_repository,
    build_insufficient_new_root_threshold_repository,
    build_insufficient_old_root_threshold_repository, build_insufficient_threshold_repository,
    build_multi_key_root_rotation_repository, build_multi_root_rotation_repository,
    build_new_only_root_rotation_repository, build_non_ascii_manifest_repository,
    build_old_only_root_rotation_repository, build_pointer_manifest_mismatch_repository,
    build_repository, build_rotated_repository, build_skipped_root_rotation_repository,
    build_special_archive_mode_repository, build_threshold_repository,
    build_wrong_role_repository_pair,
};

fn transport(responses: std::collections::HashMap<String, ScriptedResponse>) -> CapturingTransport {
    CapturingTransport::new(config().metadata_base().clone(), 1, responses)
}

fn transport_from_root(
    root_version: u64,
    responses: std::collections::HashMap<String, ScriptedResponse>,
) -> CapturingTransport {
    CapturingTransport::new(config().metadata_base().clone(), root_version, responses)
}

fn config() -> AttemptConfig {
    config_for("stable")
}

fn config_for(channel: &str) -> AttemptConfig {
    AttemptConfig::new(
        Url::parse(&format!("https://updates.invalid/{channel}/metadata/"))
            .expect("fixed URL is valid"),
        Url::parse(&format!("https://updates.invalid/{channel}/targets/"))
            .expect("fixed URL is valid"),
        channel,
    )
    .expect("fixed attempt config is valid")
}

#[tokio::test]
async fn both_candidates_accept_same_valid_top_level_repository() {
    let fixture = build_repository(1).await.expect("fixture builds");
    let committed = committed_from_fixture(&fixture, 1);
    let evidence = verify_tough_attempt(&config(), &committed, transport(fixture.responses()))
        .await
        .expect("tough comparator accepts valid repository");
    assert!(evidence.verifier_sample() >= evidence.update_start());
    let sigstore = verify_sigstore_core(&config(), &committed, &evidence.as_untrusted_attempt())
        .expect("sigstore transport-free core accepts same repository");
    assert!(evidence.exact_metadata_eq(&sigstore));
    let binding = verify_application_binding(
        &evidence,
        &fixture.pointer,
        &fixture.manifest,
        &mut Cursor::new(&fixture.archive),
    )
    .expect("three direct application targets cross-bind");
    assert_eq!(
        binding.identity(),
        ("0.2.0", "aarch64-apple-darwin", "stable")
    );
    assert_eq!(binding.archive_descriptor().0, fixture.archive.len() as u64);
}

#[tokio::test]
async fn verified_evidence_becomes_authority_only_at_journal_commit() {
    let fixture = build_repository(1).await.expect("fixture builds");
    let committed = committed_from_fixture(&fixture, 1);
    let evidence = verify_tough_attempt(&config(), &committed, transport(fixture.responses()))
        .await
        .expect("candidate verifies");
    let root = tempfile::TempDir::new().expect("journal root");
    std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o700))
        .expect("private journal root");
    let canonical = root.path().canonicalize().expect("canonical journal root");
    assert_eq!(read_selected_sequence(&canonical).unwrap(), None);
    assert_eq!(
        commit_candidate(&canonical, &evidence, FaultPlan::default()).unwrap(),
        CommitOutcome::Committed { sequence: 1 }
    );
    assert_eq!(read_selected_sequence(&canonical).unwrap(), Some(1));

    let sigstore = verify_sigstore_core(&config(), &committed, &attempt_from_fixture(&fixture))
        .expect("sigstore evidence verifies independently");
    let second_root = tempfile::TempDir::new().expect("second journal root");
    std::fs::set_permissions(second_root.path(), std::fs::Permissions::from_mode(0o700))
        .expect("private second journal root");
    let second_canonical = second_root
        .path()
        .canonicalize()
        .expect("canonical second root");
    assert_eq!(
        commit_candidate(&second_canonical, &sigstore, FaultPlan::default()).unwrap(),
        CommitOutcome::Committed { sequence: 1 }
    );
    assert_eq!(read_selected_sequence(&second_canonical).unwrap(), Some(1));
}

#[tokio::test]
async fn tough_non_not_found_root_probe_is_fatal() {
    let fixture = build_repository(1).await.expect("fixture builds");
    let committed = committed_from_fixture(&fixture, 1);
    let mut responses = fixture.responses();
    responses.insert("2.root.json".to_string(), ScriptedResponse::Other);
    assert!(matches!(
        verify_tough_attempt(&config(), &committed, transport(responses)).await,
        Err(SpikeError::CandidateRejected | SpikeError::TransportPolicy)
    ));
}

#[tokio::test]
async fn equal_version_different_bytes_is_replay() {
    let fixture = build_repository(1).await.expect("fixture builds");
    let committed = committed_from_fixture(&fixture, 1);
    let mut responses = fixture.responses();
    let mut changed = fixture.timestamp.clone();
    changed.push(b'\n');
    responses.insert(
        "timestamp.json".to_string(),
        ScriptedResponse::Bytes(changed.clone()),
    );
    assert!(matches!(
        verify_tough_attempt(&config(), &committed, transport(responses)).await,
        Err(SpikeError::EqualVersionChangedBytes)
    ));
    let mut attempt = attempt_from_fixture(&fixture);
    attempt.timestamp.raw = changed;
    assert!(matches!(
        verify_sigstore_core(&config(), &committed, &attempt),
        Err(SpikeError::EqualVersionChangedBytes)
    ));
}

#[tokio::test]
async fn both_candidates_accept_monotonic_lower_role_update() {
    let old = build_repository(1).await.expect("old fixture builds");
    let new = build_repository(2).await.expect("new fixture builds");
    let committed = committed_from_fixture(&old, 1);
    let evidence = verify_tough_attempt(&config(), &committed, transport(new.responses()))
        .await
        .expect("tough accepts monotonic metadata versions");
    let sigstore = verify_sigstore_core(&config(), &committed, &evidence.as_untrusted_attempt())
        .expect("sigstore core accepts the same monotonic update");
    assert!(evidence.exact_metadata_eq(&sigstore));
    assert_eq!(evidence.timestamp().version, 2);
    assert_eq!(evidence.snapshot().version, 2);
    assert_eq!(evidence.targets().version, 2);
}

#[tokio::test]
async fn retained_normalization_fixture_preserves_exact_verified_envelope() {
    const ROOT: &[u8] = include_bytes!("../../testdata/root-v1.json");
    const NORMALIZED: &[u8] = include_bytes!("../../testdata/timestamp-v2-normalized.json");
    const SNAPSHOT: &[u8] = include_bytes!("../../testdata/snapshot-v2.json");
    const TARGETS: &[u8] = include_bytes!("../../testdata/targets-v2.json");
    for (bytes, expected) in [
        (
            ROOT,
            "5401ed31f3943848b78c5e2c2998026b65bb7430cd41c9f587ee2c1700ca57c4",
        ),
        (
            NORMALIZED,
            "774d9fef78ecd45e84ca5316ab87f8fa39cbf07fb6f60d80d940e6aa0a091dfd",
        ),
        (
            SNAPSHOT,
            "585d0e5f57e76acedbe01069d11a799c24f503c4b6064dc2ba077f00d94ed193",
        ),
        (
            TARGETS,
            "e1a2a22d67c7b9af5b291a0950bb26e5262ed85317ac59259cfc27448f2dc88e",
        ),
    ] {
        assert_eq!(hex::encode(crate::model::sha256(bytes)), expected);
    }
    let baseline = build_repository(1).await.expect("baseline fixture builds");
    let normalized: tough::schema::Signed<tough::schema::Timestamp> =
        serde_json::from_slice(NORMALIZED).expect("retained timestamp parses");
    assert_ne!(serde_json::to_vec(&normalized).unwrap(), NORMALIZED);

    let mut committed = committed_from_fixture(&baseline, 1);
    committed.raw_root = ROOT.to_vec();
    committed.root = crate::model::RoleFloor::from_bytes(1, ROOT);
    let responses = std::collections::HashMap::from([
        ("2.root.json".to_string(), ScriptedResponse::NotFound),
        (
            "timestamp.json".to_string(),
            ScriptedResponse::Bytes(NORMALIZED.to_vec()),
        ),
        (
            "snapshot.json".to_string(),
            ScriptedResponse::Bytes(SNAPSHOT.to_vec()),
        ),
        (
            "targets.json".to_string(),
            ScriptedResponse::Bytes(TARGETS.to_vec()),
        ),
    ]);
    let tough = verify_tough_attempt(&config(), &committed, transport(responses))
        .await
        .expect("tough accepts semantically identical normalized envelope");
    assert_eq!(tough.timestamp().raw, NORMALIZED);
    let sigstore = verify_sigstore_core(&config(), &committed, &tough.as_untrusted_attempt())
        .expect("sigstore accepts the same normalized envelope");
    assert!(tough.exact_metadata_eq(&sigstore));
    assert_eq!(sigstore.timestamp().raw, NORMALIZED);
}

#[tokio::test]
async fn both_candidates_enforce_multi_key_threshold_and_role_authorization() {
    let baseline = build_threshold_repository(1)
        .await
        .expect("threshold baseline builds");
    let update = build_threshold_repository(2)
        .await
        .expect("threshold update builds");
    let committed = committed_from_fixture(&baseline, 1);
    let tough = verify_tough_attempt(&config(), &committed, transport(update.responses()))
        .await
        .expect("two-of-two signatures pass tough");
    let sigstore = verify_sigstore_core(&config(), &committed, &tough.as_untrusted_attempt())
        .expect("two-of-two signatures pass sigstore");
    assert!(tough.exact_metadata_eq(&sigstore));

    for rejected in [
        build_insufficient_threshold_repository(2)
            .await
            .expect("insufficient-threshold fixture builds"),
        build_duplicate_signature_repository(2)
            .await
            .expect("duplicate-signature fixture builds"),
    ] {
        assert!(
            verify_tough_attempt(&config(), &committed, transport(rejected.responses()))
                .await
                .is_err()
        );
        let attempt = attempt_from_fixture(&rejected);
        assert!(verify_sigstore_core(&config(), &committed, &attempt).is_err());
    }

    let (valid_wrong_role_baseline, wrong_role) = build_wrong_role_repository_pair(2)
        .await
        .expect("wrong-role pair builds");
    let wrong_role_committed = committed_from_fixture(&valid_wrong_role_baseline, 1);
    assert!(verify_tough_attempt(
        &config(),
        &wrong_role_committed,
        transport(wrong_role.responses()),
    )
    .await
    .is_err());
    let wrong_role_attempt = attempt_from_fixture(&wrong_role);
    assert!(verify_sigstore_core(&config(), &wrong_role_committed, &wrong_role_attempt,).is_err());
}

#[tokio::test]
async fn gapless_dual_signed_root_rotation_passes() {
    let old = build_repository(1).await.expect("old fixture builds");
    let rotated = build_rotated_repository()
        .await
        .expect("rotated fixture builds");
    let committed = committed_from_fixture(&old, 1);
    let mut responses = rotated.responses();
    responses.insert(
        "2.root.json".to_string(),
        ScriptedResponse::Bytes(rotated.root.clone()),
    );
    responses.insert("3.root.json".to_string(), ScriptedResponse::NotFound);
    let evidence = verify_tough_attempt(&config(), &committed, transport(responses))
        .await
        .expect("tough accepts exact dual-signed N+1 root");
    assert_eq!(evidence.root_chain().len(), 1);
    assert_eq!(evidence.root().version, 2);
    let sigstore = verify_sigstore_core(&config(), &committed, &evidence.as_untrusted_attempt())
        .expect("sigstore core accepts the same dual-signed root rotation");
    assert!(evidence.exact_metadata_eq(&sigstore));
    let root = tempfile::TempDir::new().expect("rotation journal root");
    std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let root = root.path().canonicalize().unwrap();
    assert_eq!(
        commit_candidate(&root, &evidence, FaultPlan::default()).unwrap(),
        CommitOutcome::Committed { sequence: 1 }
    );
    assert_eq!(read_selected_sequence(&root).unwrap(), Some(1));
}

#[tokio::test]
async fn both_candidates_reject_skipped_and_single_sided_root_rotations() {
    let baseline = build_repository(1).await.expect("baseline fixture builds");
    let committed = committed_from_fixture(&baseline, 1);
    for rejected in [
        build_old_only_root_rotation_repository()
            .await
            .expect("old-only root fixture builds"),
        build_new_only_root_rotation_repository()
            .await
            .expect("new-only root fixture builds"),
        build_skipped_root_rotation_repository()
            .await
            .expect("skipped root fixture builds"),
    ] {
        let mut responses = rejected.responses();
        responses.insert(
            "2.root.json".to_string(),
            ScriptedResponse::Bytes(rejected.root.clone()),
        );
        assert!(
            verify_tough_attempt(&config(), &committed, transport(responses))
                .await
                .is_err()
        );
        let mut attempt = attempt_from_fixture(&rejected);
        attempt.root_chain = vec![UntrustedRole {
            request_name: "2.root.json".to_string(),
            raw: rejected.root,
        }];
        assert!(verify_sigstore_core(&config(), &committed, &attempt).is_err());
    }
}

#[tokio::test]
async fn root_rotation_enforces_both_old_and_new_multi_key_thresholds() {
    let baseline = build_threshold_repository(1)
        .await
        .expect("threshold baseline builds");
    let committed = committed_from_fixture(&baseline, 1);
    let rotated = build_multi_key_root_rotation_repository()
        .await
        .expect("multi-key rotated repository builds");
    let mut responses = rotated.responses();
    responses.insert(
        "2.root.json".to_string(),
        ScriptedResponse::Bytes(rotated.root.clone()),
    );
    responses.insert("3.root.json".to_string(), ScriptedResponse::NotFound);
    let tough = verify_tough_attempt(&config(), &committed, transport(responses))
        .await
        .expect("both old and new two-of-two root thresholds verify");
    let sigstore = verify_sigstore_core(&config(), &committed, &tough.as_untrusted_attempt())
        .expect("sigstore agrees on both root thresholds");
    assert!(tough.exact_metadata_eq(&sigstore));

    for rejected in [
        build_insufficient_old_root_threshold_repository()
            .await
            .expect("insufficient-old fixture builds"),
        build_insufficient_new_root_threshold_repository()
            .await
            .expect("insufficient-new fixture builds"),
    ] {
        let mut responses = rejected.responses();
        responses.insert(
            "2.root.json".to_string(),
            ScriptedResponse::Bytes(rejected.root.clone()),
        );
        assert!(
            verify_tough_attempt(&config(), &committed, transport(responses))
                .await
                .is_err()
        );
        let mut attempt = attempt_from_fixture(&rejected);
        attempt.root_chain = vec![UntrustedRole {
            request_name: "2.root.json".to_string(),
            raw: rejected.root,
        }];
        assert!(verify_sigstore_core(&config(), &committed, &attempt).is_err());
    }
}

#[tokio::test]
async fn one_attempt_accepts_and_durably_recovers_a_gapless_multi_root_chain() {
    let baseline = build_repository(1).await.expect("baseline builds");
    let (rotated, intermediate_root) = build_multi_root_rotation_repository()
        .await
        .expect("multi-root repository builds");
    let committed = committed_from_fixture(&baseline, 1);
    let mut responses = rotated.responses();
    responses.insert(
        "2.root.json".to_string(),
        ScriptedResponse::Bytes(intermediate_root.clone()),
    );
    responses.insert(
        "3.root.json".to_string(),
        ScriptedResponse::Bytes(rotated.root.clone()),
    );
    responses.insert("4.root.json".to_string(), ScriptedResponse::NotFound);
    let tough = verify_tough_attempt(&config(), &committed, transport(responses))
        .await
        .expect("tough accepts the exact N+1 and N+2 chain");
    assert_eq!(tough.root_chain().len(), 2);
    let sigstore = verify_sigstore_core(&config(), &committed, &tough.as_untrusted_attempt())
        .expect("sigstore accepts the same gapless chain");
    assert!(tough.exact_metadata_eq(&sigstore));

    let root = tempfile::TempDir::new().expect("journal root");
    std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let root = root.path().canonicalize().unwrap();
    assert!(commit_candidate(
        &root,
        &tough,
        FaultPlan {
            barrier: Some(crate::generation_journal::Barrier::RootChainFiles),
            action: Some(crate::generation_journal::FaultAction::ReturnError),
        },
    )
    .is_err());
    assert_eq!(
        commit_candidate(&root, &tough, FaultPlan::default()).unwrap(),
        CommitOutcome::Committed { sequence: 1 }
    );
    assert_eq!(read_selected_sequence(&root).unwrap(), Some(1));
}

#[tokio::test]
async fn restart_from_rotated_root_uses_actual_root_identity() {
    let rotated = build_rotated_repository()
        .await
        .expect("rotated fixture builds");
    let committed = committed_from_fixture(&rotated, 2);
    let evidence = verify_tough_attempt(
        &config(),
        &committed,
        transport_from_root(2, rotated.responses()),
    )
    .await
    .expect("restart from root v2 verifies without another rotation");
    assert!(evidence.root_chain().is_empty());
    assert_eq!(evidence.root().request_name, "2.root.json");
    let sigstore = verify_sigstore_core(&config(), &committed, &evidence.as_untrusted_attempt())
        .expect("sigstore agrees after root-v2 restart");
    assert!(evidence.exact_metadata_eq(&sigstore));
}

#[tokio::test]
async fn persistent_role_floors_reject_restart_rollback() {
    let committed_fixture = build_expired_lower_repository(2)
        .await
        .expect("expired committed fixture builds");
    let rollback = build_repository(1).await.expect("rollback fixture builds");
    let committed = committed_from_fixture(&committed_fixture, 2);
    assert!(
        verify_tough_attempt(&config(), &committed, transport(rollback.responses()))
            .await
            .is_err()
    );
    let attempt = attempt_from_fixture(&rollback);
    assert!(verify_sigstore_core(&config(), &committed, &attempt).is_err());
}

#[tokio::test]
async fn expired_candidate_metadata_rejects_freeze() {
    let committed_fixture = build_repository(1).await.expect("committed fixture builds");
    let expired = build_expired_lower_repository(2)
        .await
        .expect("expired fixture builds");
    let committed = committed_from_fixture(&committed_fixture, 1);
    assert!(matches!(
        verify_tough_attempt(&config(), &committed, transport(expired.responses())).await,
        Err(SpikeError::CandidateRejected | SpikeError::ExpiredAtWrapperTime)
    ));
    let attempt = attempt_from_fixture(&expired);
    assert!(verify_sigstore_core(&config(), &committed, &attempt).is_err());
}

#[tokio::test]
async fn mixed_snapshot_and_targets_reject() {
    let old = build_repository(1).await.expect("old fixture builds");
    let new = build_repository(2).await.expect("new fixture builds");
    let committed = committed_from_fixture(&old, 1);
    let mut responses = new.responses();
    responses.insert(
        "targets.json".to_string(),
        ScriptedResponse::Bytes(old.targets.clone()),
    );
    assert!(matches!(
        verify_tough_attempt(&config(), &committed, transport(responses)).await,
        Err(SpikeError::CandidateRejected)
    ));
    let mut attempt = attempt_from_fixture(&new);
    attempt.targets.raw = old.targets;
    assert!(verify_sigstore_core(&config(), &committed, &attempt).is_err());
}

#[tokio::test]
async fn duplicate_and_oversized_metadata_reject_before_candidate_parse() {
    let fixture = build_repository(1).await.expect("fixture builds");
    let committed = committed_from_fixture(&fixture, 1);

    let mut duplicate = fixture.timestamp.clone();
    duplicate.splice(1..1, br#""signed":null,"#.iter().copied());
    let mut duplicate_responses = fixture.responses();
    duplicate_responses.insert(
        "timestamp.json".to_string(),
        ScriptedResponse::Bytes(duplicate.clone()),
    );
    let duplicate_transport = transport(duplicate_responses);
    assert!(
        verify_tough_attempt(&config(), &committed, duplicate_transport.clone())
            .await
            .is_err()
    );
    assert!(duplicate_transport
        .records()
        .iter()
        .any(|record| record.outcome == FetchOutcome::Rejected && record.bytes.is_empty()));
    let mut duplicate_attempt = attempt_from_fixture(&fixture);
    duplicate_attempt.timestamp.raw = duplicate;
    assert!(verify_sigstore_core(&config(), &committed, &duplicate_attempt).is_err());

    let mut oversized_responses = fixture.responses();
    oversized_responses.insert(
        "timestamp.json".to_string(),
        ScriptedResponse::Bytes(vec![b' '; MAX_TIMESTAMP_BYTES + 1]),
    );
    let oversized_transport = transport(oversized_responses);
    assert!(
        verify_tough_attempt(&config(), &committed, oversized_transport.clone())
            .await
            .is_err()
    );
    assert!(oversized_transport
        .records()
        .iter()
        .any(|record| record.outcome == FetchOutcome::Rejected && record.bytes.is_empty()));
    let mut oversized_attempt = attempt_from_fixture(&fixture);
    oversized_attempt.timestamp.raw = vec![b' '; MAX_TIMESTAMP_BYTES + 1];
    assert!(verify_sigstore_core(&config(), &committed, &oversized_attempt).is_err());
}

#[tokio::test]
async fn sigstore_core_bounds_root_count_and_binds_versioned_request_names() {
    let fixture = build_repository(1).await.expect("fixture builds");
    let committed = committed_from_fixture(&fixture, 1);

    let mut too_many_roots = attempt_from_fixture(&fixture);
    too_many_roots.root_chain = vec![
        UntrustedRole {
            request_name: "2.root.json".to_string(),
            raw: fixture.root.clone(),
        };
        33
    ];
    assert!(matches!(
        verify_sigstore_core(&config(), &committed, &too_many_roots),
        Err(SpikeError::MetadataTooLarge)
    ));

    let mut wrong_snapshot_name = attempt_from_fixture(&fixture);
    wrong_snapshot_name.snapshot.request_name = "999.snapshot.json".to_string();
    assert!(matches!(
        verify_sigstore_core(&config(), &committed, &wrong_snapshot_name),
        Err(SpikeError::TransportPolicy)
    ));

    let mut wrong_targets_name = attempt_from_fixture(&fixture);
    wrong_targets_name.targets.request_name = "999.targets.json".to_string();
    assert!(matches!(
        verify_sigstore_core(&config(), &committed, &wrong_targets_name),
        Err(SpikeError::TransportPolicy)
    ));
}

#[tokio::test]
async fn delegations_reject_before_target_resolution() {
    let baseline = build_repository(1).await.expect("baseline fixture builds");
    let delegated = build_delegated_repository(2)
        .await
        .expect("delegated fixture builds");
    let committed = committed_from_fixture(&baseline, 1);
    assert!(matches!(
        verify_tough_attempt(&config(), &committed, transport(delegated.responses())).await,
        Err(SpikeError::DelegationsForbidden)
    ));
    let attempt = attempt_from_fixture(&delegated);
    assert!(matches!(
        verify_sigstore_core(&config(), &committed, &attempt),
        Err(SpikeError::DelegationsForbidden)
    ));
}

#[test]
fn pointer_repeats_exact_manifest_and_archive_descriptors() {
    let runtime = tokio::runtime::Runtime::new().expect("runtime builds");
    let fixture = runtime
        .block_on(build_repository(1))
        .expect("fixture builds");
    let committed = committed_from_fixture(&fixture, 1);
    let evidence = runtime
        .block_on(verify_tough_attempt(
            &config(),
            &committed,
            transport(fixture.responses()),
        ))
        .expect("fixture metadata verifies");
    let mut changed_archive = fixture.archive.clone();
    changed_archive.push(0);
    assert!(matches!(
        verify_application_binding(
            &evidence,
            &fixture.pointer,
            &fixture.manifest,
            &mut Cursor::new(&changed_archive),
        ),
        Err(SpikeError::ApplicationBinding)
    ));

    let mut truncated = fixture.archive.clone();
    truncated.pop();
    assert!(matches!(
        verify_application_binding(
            &evidence,
            &fixture.pointer,
            &fixture.manifest,
            &mut Cursor::new(&truncated),
        ),
        Err(SpikeError::ApplicationBinding)
    ));

    let mut changed_manifest = fixture.manifest.clone();
    changed_manifest.push(b' ');
    assert!(matches!(
        verify_application_binding(
            &evidence,
            &fixture.pointer,
            &changed_manifest,
            &mut Cursor::new(&fixture.archive),
        ),
        Err(SpikeError::ApplicationBinding)
    ));

    let mut failing = FailingArchive::new(&fixture.archive, fixture.archive.len() / 2);
    assert!(matches!(
        verify_application_binding(&evidence, &fixture.pointer, &fixture.manifest, &mut failing,),
        Err(SpikeError::ApplicationBinding)
    ));
}

#[tokio::test]
async fn signed_pointer_and_embedded_manifest_mismatches_fail_closed() {
    for fixture in [
        build_pointer_manifest_mismatch_repository(1)
            .await
            .expect("pointer mismatch fixture builds"),
        build_embedded_manifest_mismatch_repository(1)
            .await
            .expect("embedded mismatch fixture builds"),
    ] {
        let committed = committed_from_fixture(&fixture, 1);
        let evidence = verify_tough_attempt(&config(), &committed, transport(fixture.responses()))
            .await
            .expect("TUF metadata and exact target descriptors verify");
        assert!(matches!(
            verify_application_binding(
                &evidence,
                &fixture.pointer,
                &fixture.manifest,
                &mut Cursor::new(&fixture.archive),
            ),
            Err(SpikeError::ApplicationBinding)
        ));
    }
}

#[tokio::test]
async fn authenticated_channel_must_match_the_signed_pointer_channel() {
    let fixture = build_repository(1).await.expect("fixture builds");
    let committed = committed_from_fixture(&fixture, 1);
    let beta = config_for("beta");
    let beta_transport =
        CapturingTransport::new(beta.metadata_base().clone(), 1, fixture.responses());
    let evidence = verify_tough_attempt(&beta, &committed, beta_transport)
        .await
        .expect("the repository is valid under its configured beta route");
    assert!(matches!(
        verify_application_binding(
            &evidence,
            &fixture.pointer,
            &fixture.manifest,
            &mut Cursor::new(&fixture.archive),
        ),
        Err(SpikeError::ApplicationBinding)
    ));
}

#[tokio::test]
async fn signed_archives_reject_special_file_types_and_undeclared_mode_bits() {
    for mode in [0o010644, 0o104755] {
        let fixture = build_special_archive_mode_repository(1, mode)
            .await
            .expect("special-mode fixture builds");
        let committed = committed_from_fixture(&fixture, 1);
        let evidence = verify_tough_attempt(&config(), &committed, transport(fixture.responses()))
            .await
            .expect("TUF authenticates the deliberately hostile archive bytes");
        assert!(matches!(
            verify_application_binding(
                &evidence,
                &fixture.pointer,
                &fixture.manifest,
                &mut Cursor::new(&fixture.archive),
            ),
            Err(SpikeError::ApplicationBinding)
        ));
    }
}

#[tokio::test]
async fn production_manifest_semantics_reject_case_collisions_and_non_ascii_paths() {
    for fixture in [
        build_case_collision_manifest_repository(1)
            .await
            .expect("case-collision fixture builds"),
        build_non_ascii_manifest_repository(1)
            .await
            .expect("non-ASCII fixture builds"),
    ] {
        let committed = committed_from_fixture(&fixture, 1);
        let evidence = verify_tough_attempt(&config(), &committed, transport(fixture.responses()))
            .await
            .expect("TUF authenticates the deliberately hostile manifest bytes");
        assert!(matches!(
            verify_application_binding(
                &evidence,
                &fixture.pointer,
                &fixture.manifest,
                &mut Cursor::new(&fixture.archive),
            ),
            Err(SpikeError::ApplicationBinding)
        ));
    }
}

struct FailingArchive<'a> {
    inner: Cursor<&'a [u8]>,
    fail_after: usize,
    seen: usize,
}

impl<'a> FailingArchive<'a> {
    fn new(bytes: &'a [u8], fail_after: usize) -> Self {
        Self {
            inner: Cursor::new(bytes),
            fail_after,
            seen: 0,
        }
    }
}

impl Read for FailingArchive<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.seen >= self.fail_after {
            return Err(io::Error::other("injected read failure"));
        }
        let allowed = buffer.len().min(self.fail_after - self.seen);
        let count = self.inner.read(&mut buffer[..allowed])?;
        self.seen += count;
        Ok(count)
    }
}

impl Seek for FailingArchive<'_> {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        let offset = self.inner.seek(position)?;
        self.seen = usize::try_from(offset).unwrap_or(usize::MAX);
        Ok(offset)
    }
}

#[tokio::test]
#[ignore = "explicit fixture regeneration utility"]
async fn regenerate_static_normalization_corpus() {
    let output = std::env::var_os("HF2Q_TUF_FIXTURE_OUT")
        .expect("HF2Q_TUF_FIXTURE_OUT must name an explicit output directory");
    let output = std::path::PathBuf::from(output);
    std::fs::create_dir_all(&output).expect("create fixture output directory");
    let [root, timestamp, snapshot, targets] =
        crate::test_repository::build_static_normalization_corpus()
            .await
            .expect("deterministic corpus builds");
    for (name, bytes) in [
        ("root-v1.json", root.as_slice()),
        ("timestamp-v2-normalized.json", timestamp.as_slice()),
        ("snapshot-v2.json", snapshot.as_slice()),
        ("targets-v2.json", targets.as_slice()),
    ] {
        std::fs::write(output.join(name), bytes).expect("write retained fixture");
    }
}
