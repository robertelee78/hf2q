use std::collections::BTreeSet;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::Path;

use super::*;
use crate::distribution::install_state::{PENDING_ACTIVATION, PENDING_CURRENT};

const TEST_ID: &str = "7c907c7a-3125-4a40-a8b3-1c125080e46a";

fn test_root(parent: &tempfile::TempDir) -> std::path::PathBuf {
    parent
        .path()
        .canonicalize()
        .expect("canonical temp path")
        .join("state")
}

fn authorization(root: &Path) -> ExplicitRootAuthorization {
    ExplicitRootAuthorization::new(root).expect("explicit root authorization")
}

fn expected_bytes(root: &Path) -> Vec<u8> {
    InstallationIdentityV1::new(
        InstallationId::parse(TEST_ID.to_owned()).expect("test UUID"),
        AbsoluteInstallPath::parse("state_root", root.to_str().expect("UTF-8 root").to_owned())
            .expect("state root"),
    )
    .to_deterministic_json()
    .expect("identity bytes")
}

fn seed_intent(root: &Path, prefix: &[u8]) {
    let locked = LockedInstallation::acquire(root).expect("bootstrap lock");
    drop(locked);
    let path = root.join("update").join(intent_name(
        &InstallationId::parse(TEST_ID.to_owned()).expect("test UUID"),
    ));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true).mode(0o600);
    let mut file = options.open(path).expect("create identity intent");
    std::io::Write::write_all(&mut file, prefix).expect("write identity prefix");
    file.sync_all().expect("sync identity prefix");
}

fn directory_inventory(path: &Path) -> BTreeSet<std::ffi::OsString> {
    std::fs::read_dir(path)
        .expect("directory inventory")
        .map(|entry| entry.expect("entry").file_name())
        .collect()
}

fn update_inventory(root: &Path) -> BTreeSet<std::ffi::OsString> {
    directory_inventory(&root.join("update"))
}

#[test]
fn bootstrap_commits_one_immutable_root_bound_identity() {
    let parent = tempfile::tempdir().expect("tempdir");
    let root = test_root(&parent);
    let created = bootstrap_installation_identity_for_test(
        authorization(&root),
        TEST_ID,
        IdentityFaultPlan::default(),
    )
    .expect("bootstrap identity");
    assert!(matches!(created, InstallationIdentityBootstrap::Created(_)));
    let identity = created.into_identity();
    assert_eq!(identity.installation_id().as_str(), TEST_ID);
    assert_eq!(identity.state_root().as_str(), root.to_str().unwrap());
    assert_eq!(
        std::fs::read(root.join("update").join(IDENTITY_FILE)).expect("identity bytes"),
        expected_bytes(&root)
    );

    let reopened = open_existing_installation_identity(authorization(&root))
        .expect("open identity")
        .expect("identity exists");
    assert_eq!(reopened.installation_id().as_str(), TEST_ID);
    let repeated = bootstrap_installation_identity_for_test(
        authorization(&root),
        TEST_ID,
        IdentityFaultPlan::default(),
    )
    .expect("repeat identity bootstrap");
    assert!(matches!(
        repeated,
        InstallationIdentityBootstrap::AlreadyCreated(_)
    ));
    assert_eq!(repeated.into_identity().installation_id().as_str(), TEST_ID);
}

#[test]
fn every_exact_intent_prefix_recovers_the_filename_uuid() {
    let sample_parent = tempfile::tempdir().expect("sample tempdir");
    let sample_root = test_root(&sample_parent);
    let length = expected_bytes(&sample_root).len();
    for prefix_len in 0..=length {
        let parent = tempfile::tempdir().expect("tempdir");
        let root = test_root(&parent);
        let expected = expected_bytes(&root);
        assert_eq!(expected.len(), length, "temporary root width changed");
        seed_intent(&root, &expected[..prefix_len]);
        let identity = bootstrap_installation_identity(authorization(&root))
            .expect("resume exact prefix")
            .into_identity();
        assert_eq!(identity.installation_id().as_str(), TEST_ID);
        assert_eq!(
            std::fs::read(root.join("update").join(IDENTITY_FILE)).expect("final identity"),
            expected
        );
    }
}

#[test]
fn every_durability_barrier_is_exactly_retryable() {
    for barrier in IdentityBarrier::ALL {
        let parent = tempfile::tempdir().expect("tempdir");
        let root = test_root(&parent);
        let error = bootstrap_installation_identity_for_test(
            authorization(&root),
            TEST_ID,
            IdentityFaultPlan::once(barrier),
        )
        .expect_err("barrier must fail once");
        let committed = matches!(
            barrier,
            IdentityBarrier::IdentityRename
                | IdentityBarrier::FinalReopen
                | IdentityBarrier::FinalFullSync
                | IdentityBarrier::UpdateSync
                | IdentityBarrier::RootSync
                | IdentityBarrier::LockFullSync
        );
        assert_eq!(
            matches!(
                error,
                InstallStateError::IdentityCommittedDurabilityUnknown { .. }
            ),
            committed,
            "wrong error phase for {barrier:?}"
        );
        let repaired = bootstrap_installation_identity_for_test(
            authorization(&root),
            TEST_ID,
            IdentityFaultPlan::default(),
        )
        .expect("exact retry repairs identity")
        .into_identity();
        assert_eq!(repaired.installation_id().as_str(), TEST_ID);
        assert_eq!(
            std::fs::read(root.join("update").join(IDENTITY_FILE)).expect("final identity"),
            expected_bytes(&root)
        );
    }
}

#[test]
fn conflicting_or_ambiguous_identity_residue_is_preserved() {
    for hostile in ["conflicting-prefix", "malformed-name", "final-plus-intent"] {
        let parent = tempfile::tempdir().expect("tempdir");
        let root = test_root(&parent);
        let expected = expected_bytes(&root);
        seed_intent(
            &root,
            if hostile == "conflicting-prefix" {
                b"X"
            } else {
                &expected
            },
        );
        let update = root.join("update");
        if hostile == "malformed-name" {
            std::fs::rename(
                update.join(intent_name(
                    &InstallationId::parse(TEST_ID.to_owned()).expect("test UUID"),
                )),
                update.join(".installation-identity-v1-not-a-uuid.partial"),
            )
            .expect("rename hostile intent");
        } else if hostile == "final-plus-intent" {
            std::fs::write(update.join(IDENTITY_FILE), &expected).expect("hostile final");
            std::fs::set_permissions(
                update.join(IDENTITY_FILE),
                std::fs::Permissions::from_mode(0o600),
            )
            .expect("identity mode");
        }
        let before = update_inventory(&root);
        assert!(bootstrap_installation_identity(authorization(&root)).is_err());
        let after = update_inventory(&root);
        assert_eq!(before, after, "hostile evidence was mutated for {hostile}");
    }
}

#[test]
fn hostile_intent_attributes_and_bounded_inventory_are_preserved() {
    for hostile in [
        "symlink",
        "hardlink",
        "wrong-mode",
        "oversized",
        "multiple",
        "over-cap",
    ] {
        let parent = tempfile::tempdir().expect("tempdir");
        let root = test_root(&parent);
        let expected = expected_bytes(&root);
        seed_intent(&root, &expected);
        let update = root.join("update");
        let intent = update.join(intent_name(
            &InstallationId::parse(TEST_ID.to_owned()).expect("test UUID"),
        ));
        match hostile {
            "symlink" => {
                std::fs::rename(&intent, root.join("intent-target")).expect("move intent target");
                std::os::unix::fs::symlink("../intent-target", &intent)
                    .expect("hostile intent symlink");
            }
            "hardlink" => {
                std::fs::hard_link(&intent, root.join("hostile-intent-link"))
                    .expect("hostile intent hardlink");
                assert_eq!(
                    std::fs::metadata(&intent).expect("intent metadata").nlink(),
                    2
                );
            }
            "wrong-mode" => {
                std::fs::set_permissions(&intent, std::fs::Permissions::from_mode(0o644))
                    .expect("hostile intent mode")
            }
            "oversized" => {
                use std::io::Write as _;
                std::fs::OpenOptions::new()
                    .append(true)
                    .open(&intent)
                    .expect("open intent")
                    .write_all(b"X")
                    .expect("extend intent");
            }
            "multiple" => {
                let other =
                    InstallationId::parse("a70ee078-5f20-45f6-bf42-bfcd1a992382".to_owned())
                        .expect("second UUID");
                std::fs::write(update.join(intent_name(&other)), b"")
                    .expect("second identity intent");
                std::fs::set_permissions(
                    update.join(intent_name(&other)),
                    std::fs::Permissions::from_mode(0o600),
                )
                .expect("second intent mode");
            }
            "over-cap" => {
                for index in 0..MAX_UPDATE_INVENTORY {
                    std::fs::create_dir(update.join(format!("hostile-{index}")))
                        .expect("hostile inventory entry");
                }
            }
            _ => unreachable!(),
        }
        let before = update_inventory(&root);
        assert!(open_existing_installation_identity(authorization(&root)).is_err());
        assert!(bootstrap_installation_identity(authorization(&root)).is_err());
        assert_eq!(
            update_inventory(&root),
            before,
            "evidence changed for {hostile}"
        );
    }
}

#[test]
fn hostile_final_identity_and_missing_scaffold_are_preserved_fail_closed() {
    for hostile in [
        "symlink",
        "hardlink",
        "wrong-mode",
        "corrupt-bytes",
        "oversized",
        "missing-scaffold",
    ] {
        let parent = tempfile::tempdir().expect("tempdir");
        let root = test_root(&parent);
        bootstrap_installation_identity_for_test(
            authorization(&root),
            TEST_ID,
            IdentityFaultPlan::default(),
        )
        .expect("identity");
        let update = root.join("update");
        let identity = update.join(IDENTITY_FILE);
        match hostile {
            "symlink" => {
                std::fs::rename(&identity, root.join("identity-target"))
                    .expect("move identity target");
                std::os::unix::fs::symlink("../identity-target", &identity)
                    .expect("hostile identity symlink");
            }
            "hardlink" => {
                std::fs::hard_link(&identity, root.join("hostile-identity-link"))
                    .expect("hostile identity hardlink");
                assert_eq!(
                    std::fs::metadata(&identity)
                        .expect("identity metadata")
                        .nlink(),
                    2
                );
            }
            "wrong-mode" => {
                std::fs::set_permissions(&identity, std::fs::Permissions::from_mode(0o644))
                    .expect("hostile identity mode");
            }
            "corrupt-bytes" => {
                std::fs::write(&identity, b"not canonical identity\n")
                    .expect("corrupt identity bytes");
            }
            "oversized" => {
                std::fs::write(&identity, vec![b'x'; MAX_INSTALLATION_IDENTITY_BYTES + 1])
                    .expect("oversized identity bytes");
            }
            "missing-scaffold" => {
                std::fs::remove_file(update.join(".noreplace-source"))
                    .expect("remove bootstrap scaffold");
            }
            _ => unreachable!(),
        }
        let root_before = directory_inventory(&root);
        let update_before = update_inventory(&root);
        assert!(open_existing_installation_identity(authorization(&root)).is_err());
        assert!(bootstrap_installation_identity(authorization(&root)).is_err());
        assert_eq!(
            directory_inventory(&root),
            root_before,
            "root changed for {hostile}"
        );
        assert_eq!(
            update_inventory(&root),
            update_before,
            "update changed for {hostile}"
        );
    }
}

#[test]
fn pre_identity_reads_are_nonmutating_but_dependent_state_fails_closed() {
    let parent = tempfile::tempdir().expect("tempdir");
    let root = test_root(&parent);
    assert!(open_existing_installation_identity(authorization(&root))
        .expect("absent identity")
        .is_none());
    assert!(!root.exists(), "read-only identity lookup created the root");

    std::fs::create_dir(&root).expect("explicit root");
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).expect("root mode");
    std::fs::create_dir(root.join("versions")).expect("dependent versions");
    std::fs::set_permissions(
        root.join("versions"),
        std::fs::Permissions::from_mode(0o700),
    )
    .expect("versions mode");
    assert!(open_existing_installation_identity(authorization(&root)).is_err());
    assert!(bootstrap_installation_identity(authorization(&root)).is_err());
    assert!(root.join("versions").is_dir());
    assert!(
        !root.join("update").exists(),
        "invalid dependent state was mutated before rejection"
    );
}

#[test]
fn every_reserved_dependent_namespace_requires_the_final_identity() {
    for (scope, name) in [
        ("root", "versions"),
        ("root", "activations"),
        ("root", "current"),
        ("root", PENDING_ACTIVATION),
        ("root", PENDING_CURRENT),
        ("root", "uninstall"),
        ("update", "metadata"),
        ("update", "downloads"),
        ("update", "extractions"),
        ("update", "prepared"),
    ] {
        let parent = tempfile::tempdir().expect("tempdir");
        let root = test_root(&parent);
        std::fs::create_dir(&root).expect("explicit root");
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).expect("root mode");
        let dependent_parent = if scope == "root" {
            root.clone()
        } else {
            let update = root.join("update");
            std::fs::create_dir(&update).expect("update directory");
            std::fs::set_permissions(&update, std::fs::Permissions::from_mode(0o700))
                .expect("update mode");
            update
        };
        std::fs::create_dir(dependent_parent.join(name)).expect("dependent namespace");
        std::fs::set_permissions(
            dependent_parent.join(name),
            std::fs::Permissions::from_mode(0o700),
        )
        .expect("dependent namespace mode");
        let root_before = directory_inventory(&root);
        let update_before = root
            .join("update")
            .is_dir()
            .then(|| update_inventory(&root));

        assert!(open_existing_installation_identity(authorization(&root)).is_err());
        assert!(bootstrap_installation_identity(authorization(&root)).is_err());
        assert_eq!(
            directory_inventory(&root),
            root_before,
            "root changed for {name}"
        );
        assert_eq!(
            root.join("update")
                .is_dir()
                .then(|| update_inventory(&root)),
            update_before,
            "update changed for {name}"
        );
    }
}

#[test]
fn final_update_inventory_accepts_its_exact_cap_and_rejects_cap_plus_one() {
    let parent = tempfile::tempdir().expect("tempdir");
    let root = test_root(&parent);
    bootstrap_installation_identity_for_test(
        authorization(&root),
        TEST_ID,
        IdentityFaultPlan::default(),
    )
    .expect("identity");
    for name in ["metadata", "downloads", "extractions", "prepared"] {
        std::fs::create_dir(root.join("update").join(name)).expect("owned update namespace");
        std::fs::set_permissions(
            root.join("update").join(name),
            std::fs::Permissions::from_mode(0o700),
        )
        .expect("owned update namespace mode");
    }
    assert_eq!(update_inventory(&root).len(), MAX_UPDATE_INVENTORY);
    open_existing_installation_identity(authorization(&root))
        .expect("exact-cap inventory")
        .expect("identity");

    std::fs::create_dir(root.join("update/unexpected-ninth-entry")).expect("cap-plus-one entry");
    let before = update_inventory(&root);
    assert!(open_existing_installation_identity(authorization(&root)).is_err());
    assert!(bootstrap_installation_identity(authorization(&root)).is_err());
    assert_eq!(update_inventory(&root), before);
}

#[test]
fn unrelated_preserved_root_state_does_not_block_bootstrap() {
    let parent = tempfile::tempdir().expect("tempdir");
    let root = test_root(&parent);
    let locked = LockedInstallation::acquire(&root).expect("bootstrap root");
    unix::ensure_private_directory(locked.root(), "ruvector").expect("preserved state");
    drop(locked);
    bootstrap_installation_identity_for_test(
        authorization(&root),
        TEST_ID,
        IdentityFaultPlan::default(),
    )
    .expect("bootstrap beside preserved state");
    assert!(root.join("ruvector").is_dir());
}

#[test]
fn copied_or_replaced_identity_never_authorizes_another_root_or_inode() {
    let first_parent = tempfile::tempdir().expect("first tempdir");
    let first_root = test_root(&first_parent);
    let identity = bootstrap_installation_identity_for_test(
        authorization(&first_root),
        TEST_ID,
        IdentityFaultPlan::default(),
    )
    .expect("first identity")
    .into_identity();

    let second_parent = tempfile::tempdir().expect("second tempdir");
    let second_root = test_root(&second_parent);
    let second_lock = LockedInstallation::acquire(&second_root).expect("second root");
    drop(second_lock);
    let copied = second_root.join("update").join(IDENTITY_FILE);
    std::fs::copy(first_root.join("update").join(IDENTITY_FILE), &copied).expect("copy identity");
    std::fs::set_permissions(&copied, std::fs::Permissions::from_mode(0o600)).expect("copied mode");
    assert!(open_existing_installation_identity(authorization(&second_root)).is_err());

    let final_path = first_root.join("update").join(IDENTITY_FILE);
    let displaced = first_root.join("update").join("displaced-identity");
    std::fs::rename(&final_path, &displaced).expect("displace identity");
    std::fs::write(&final_path, expected_bytes(&first_root)).expect("replace identity bytes");
    std::fs::set_permissions(&final_path, std::fs::Permissions::from_mode(0o600))
        .expect("replacement mode");
    assert!(
        identity.lock().is_err(),
        "same bytes with a new inode authorized"
    );
}

#[test]
fn same_byte_identity_swap_between_read_snapshots_is_rejected() {
    let parent = tempfile::tempdir().expect("tempdir");
    let root = test_root(&parent);
    bootstrap_installation_identity_for_test(
        authorization(&root),
        TEST_ID,
        IdentityFaultPlan::default(),
    )
    .expect("identity");
    let final_path = root.join("update").join(IDENTITY_FILE);
    let expected = expected_bytes(&root);
    assert!(
        open_existing_installation_identity_with_hook(authorization(&root), || {
            std::fs::rename(&final_path, root.join("displaced-identity"))
                .expect("displace first identity inode");
            std::fs::write(&final_path, &expected).expect("replace with identical bytes");
            std::fs::set_permissions(&final_path, std::fs::Permissions::from_mode(0o600))
                .expect("replacement identity mode");
        })
        .is_err()
    );
}

#[test]
fn replacing_the_lock_inode_cannot_split_a_live_identity_capability() {
    let parent = tempfile::tempdir().expect("tempdir");
    let root = test_root(&parent);
    let identity = bootstrap_installation_identity_for_test(
        authorization(&root),
        TEST_ID,
        IdentityFaultPlan::default(),
    )
    .expect("identity")
    .into_identity();
    let lock = root.join("update").join(LOCK_FILE);
    std::fs::rename(&lock, root.join("displaced-install-lock")).expect("displace lock inode");
    std::fs::write(&lock, b"").expect("replacement lock");
    std::fs::set_permissions(&lock, std::fs::Permissions::from_mode(0o600))
        .expect("replacement lock mode");
    assert!(identity.lock().is_err());
}

#[test]
fn capability_debug_never_exposes_a_file_descriptor_or_root_path() {
    let parent = tempfile::tempdir().expect("tempdir");
    let root = test_root(&parent);
    let identity = bootstrap_installation_identity_for_test(
        authorization(&root),
        TEST_ID,
        IdentityFaultPlan::default(),
    )
    .expect("identity")
    .into_identity();
    let rendered = format!("{identity:?}");
    assert!(!rendered.contains("fd:"));
    assert!(!rendered.contains(root.to_str().unwrap()));
    assert!(!rendered.contains(TEST_ID));
}

#[path = "identity_tests/process_cases.rs"]
mod process_cases;
