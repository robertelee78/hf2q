use std::os::unix::fs::{symlink, MetadataExt, PermissionsExt};
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Barrier};

use super::*;

const FINAL: &str = "profile.json";
const PARTIAL: &str = ".profile.json.partial";
const BYTES: &[u8] = b"{\"exact\":true}\n";

fn publish(
    parent: &Path,
    bytes: &[u8],
    cap: usize,
) -> Result<(), ModelPreparationPublicationError> {
    let identity = directory_identity(parent)?;
    publish_exact_private_file(parent, identity, FINAL, PARTIAL, bytes, cap)
}

#[test]
fn exact_private_publication_is_idempotent_and_leaves_one_final_inode() {
    let root = tempfile::tempdir().unwrap();
    let parent = root.path().canonicalize().unwrap();
    publish(&parent, BYTES, 1024).unwrap();
    publish(&parent, BYTES, 1024).unwrap();

    let final_path = parent.join(FINAL);
    assert_eq!(std::fs::read(&final_path).unwrap(), BYTES);
    assert_eq!(
        std::fs::symlink_metadata(&final_path).unwrap().mode() & 0o7777,
        0o600
    );
    assert_eq!(std::fs::symlink_metadata(&final_path).unwrap().nlink(), 1);
    assert!(!parent.join(PARTIAL).exists());
}

#[test]
fn exact_prefix_and_published_hardlink_crash_states_are_resumed() {
    for crash_after_link in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let parent = root.path().canonicalize().unwrap();
        let partial = parent.join(PARTIAL);
        std::fs::write(&partial, &BYTES[..5]).unwrap();
        std::fs::set_permissions(&partial, std::fs::Permissions::from_mode(0o600)).unwrap();
        if crash_after_link {
            std::fs::write(&partial, BYTES).unwrap();
            std::fs::hard_link(&partial, parent.join(FINAL)).unwrap();
        }

        publish(&parent, BYTES, 1024).unwrap();
        assert_eq!(std::fs::read(parent.join(FINAL)).unwrap(), BYTES);
        assert!(!partial.exists());
        assert_eq!(
            std::fs::symlink_metadata(parent.join(FINAL))
                .unwrap()
                .nlink(),
            1
        );
    }
}

#[test]
fn hostile_partial_final_and_symlink_evidence_is_retained() {
    let cases: Vec<Box<dyn Fn(&Path)>> = vec![
        Box::new(|root| {
            std::fs::write(root.join(PARTIAL), b"wrong").unwrap();
            std::fs::set_permissions(root.join(PARTIAL), std::fs::Permissions::from_mode(0o600))
                .unwrap();
        }),
        Box::new(|root| {
            std::fs::write(root.join(FINAL), b"wrong").unwrap();
            std::fs::set_permissions(root.join(FINAL), std::fs::Permissions::from_mode(0o600))
                .unwrap();
        }),
        Box::new(|root| {
            symlink(root.join("missing"), root.join(PARTIAL)).unwrap();
        }),
        Box::new(|root| {
            symlink(root.join("missing"), root.join(FINAL)).unwrap();
        }),
        Box::new(|root| {
            std::fs::write(root.join(PARTIAL), BYTES).unwrap();
            std::fs::write(root.join(FINAL), BYTES).unwrap();
            for name in [PARTIAL, FINAL] {
                std::fs::set_permissions(root.join(name), std::fs::Permissions::from_mode(0o600))
                    .unwrap();
            }
        }),
    ];
    for prepare in cases {
        let root = tempfile::tempdir().unwrap();
        let parent = root.path().canonicalize().unwrap();
        prepare(&parent);
        assert!(publish(&parent, BYTES, 1024).is_err());
        assert!(std::fs::symlink_metadata(parent.join(PARTIAL))
            .or_else(|_| std::fs::symlink_metadata(parent.join(FINAL)))
            .is_ok());
    }
}

#[test]
fn cap_empty_mode_and_unrelated_hardlink_are_rejected_without_cleanup() {
    let root = tempfile::tempdir().unwrap();
    let parent = root.path().canonicalize().unwrap();
    assert!(publish(&parent, b"", 1024).is_err());
    assert!(publish(&parent, BYTES, 2).is_err());

    std::fs::write(parent.join(PARTIAL), BYTES).unwrap();
    std::fs::set_permissions(parent.join(PARTIAL), std::fs::Permissions::from_mode(0o644)).unwrap();
    assert!(publish(&parent, BYTES, 1024).is_err());
    assert!(parent.join(PARTIAL).exists());

    std::fs::set_permissions(parent.join(PARTIAL), std::fs::Permissions::from_mode(0o600)).unwrap();
    std::fs::hard_link(parent.join(PARTIAL), parent.join("outside-evidence")).unwrap();
    assert!(publish(&parent, BYTES, 1024).is_err());
    assert!(parent.join("outside-evidence").exists());
}

#[test]
fn symlinked_parent_is_rejected_before_any_record_is_created() {
    let root = tempfile::tempdir().unwrap();
    let base = root.path().canonicalize().unwrap();
    let real = base.join("real");
    std::fs::create_dir(&real).unwrap();
    let alias = base.join("alias");
    symlink(&real, &alias).unwrap();
    assert!(publish(&alias, BYTES, 1024).is_err());
    assert!(!real.join(FINAL).exists());
    assert!(!real.join(PARTIAL).exists());
}

#[test]
fn every_publication_barrier_is_exactly_retryable() {
    let new_record_barriers = [
        PublicationBarrier::ParentOpened,
        PublicationBarrier::PartialCreated,
        PublicationBarrier::PartialWritten,
        PublicationBarrier::PartialFullSynced,
        PublicationBarrier::PartialDirectorySynced,
        PublicationBarrier::FinalLinked,
        PublicationBarrier::LinkDirectorySynced,
        PublicationBarrier::PartialUnlinked,
        PublicationBarrier::UnlinkDirectorySynced,
        PublicationBarrier::FinalFullSynced,
        PublicationBarrier::ParentRebound,
    ];
    for fault in new_record_barriers {
        let root = tempfile::tempdir().unwrap();
        let parent = root.path().canonicalize().unwrap();
        let identity = directory_identity(&parent).unwrap();
        let mut fired = false;
        let result = publish_exact_private_file_with(
            &parent,
            identity,
            FINAL,
            PARTIAL,
            BYTES,
            1024,
            |barrier| {
                if !fired && barrier == fault {
                    fired = true;
                    return Err(publication_error("injected publication barrier").into());
                }
                Ok(())
            },
        );
        assert!(result.is_err(), "barrier {fault:?} did not fire");
        assert!(fired);
        publish(&parent, BYTES, 1024).unwrap();
        assert_eq!(std::fs::read(parent.join(FINAL)).unwrap(), BYTES);
        assert!(!parent.join(PARTIAL).exists());
    }

    let root = tempfile::tempdir().unwrap();
    let parent = root.path().canonicalize().unwrap();
    std::fs::write(parent.join(PARTIAL), &BYTES[..5]).unwrap();
    std::fs::set_permissions(parent.join(PARTIAL), std::fs::Permissions::from_mode(0o600)).unwrap();
    let identity = directory_identity(&parent).unwrap();
    let mut fired = false;
    assert!(publish_exact_private_file_with(
        &parent,
        identity,
        FINAL,
        PARTIAL,
        BYTES,
        1024,
        |barrier| {
            if !fired && barrier == PublicationBarrier::PartialPrefixVerified {
                fired = true;
                return Err(publication_error("injected prefix barrier").into());
            }
            Ok(())
        },
    )
    .is_err());
    assert!(fired);
    publish(&parent, BYTES, 1024).unwrap();

    for fault in [
        PublicationBarrier::FinalFullSynced,
        PublicationBarrier::AdoptedParentSynced,
        PublicationBarrier::ParentRebound,
    ] {
        let root = tempfile::tempdir().unwrap();
        let parent = root.path().canonicalize().unwrap();
        publish(&parent, BYTES, 1024).unwrap();
        let identity = directory_identity(&parent).unwrap();
        let mut fired = false;
        assert!(publish_exact_private_file_with(
            &parent,
            identity,
            FINAL,
            PARTIAL,
            BYTES,
            1024,
            |barrier| {
                if !fired && barrier == fault {
                    fired = true;
                    return Err(publication_error("injected adoption barrier").into());
                }
                Ok(())
            },
        )
        .is_err());
        assert!(fired, "adoption barrier {fault:?} did not fire");
        publish(&parent, BYTES, 1024).unwrap();
    }
}

#[test]
fn concurrent_publishers_cannot_append_after_a_measured_partial_prefix() {
    let root = tempfile::tempdir().unwrap();
    let parent = root.path().canonicalize().unwrap();
    std::fs::write(parent.join(PARTIAL), &BYTES[..5]).unwrap();
    std::fs::set_permissions(parent.join(PARTIAL), std::fs::Permissions::from_mode(0o600)).unwrap();
    let identity = directory_identity(&parent).unwrap();
    let both_measured = Arc::new(Barrier::new(2));
    let first_write_complete = Arc::new(Barrier::new(2));

    let first_parent = parent.clone();
    let first_measured = Arc::clone(&both_measured);
    let first_write = Arc::clone(&first_write_complete);
    let first = std::thread::spawn(move || {
        publish_exact_private_file_with(
            &first_parent,
            identity,
            FINAL,
            PARTIAL,
            BYTES,
            1024,
            |barrier| {
                if barrier == PublicationBarrier::PartialPrefixVerified {
                    first_measured.wait();
                }
                if barrier == PublicationBarrier::PartialWritten {
                    first_write.wait();
                }
                Ok(())
            },
        )
    });

    let second_parent = parent.clone();
    let second_measured = Arc::clone(&both_measured);
    let second_write = Arc::clone(&first_write_complete);
    let second = std::thread::spawn(move || {
        publish_exact_private_file_with(
            &second_parent,
            identity,
            FINAL,
            PARTIAL,
            BYTES,
            1024,
            |barrier| {
                if barrier == PublicationBarrier::PartialPrefixVerified {
                    second_measured.wait();
                    second_write.wait();
                }
                Ok(())
            },
        )
    });

    assert!(first.join().unwrap().is_ok());
    assert!(second.join().unwrap().is_err());
    publish(&parent, BYTES, 1024).unwrap();
    assert_eq!(std::fs::read(parent.join(FINAL)).unwrap(), BYTES);
    assert!(!parent.join(PARTIAL).exists());
}

#[test]
fn exact_read_rejects_leaf_replacement_after_the_open_fd_was_read() {
    let root = tempfile::tempdir().unwrap();
    let parent_path = root.path().canonicalize().unwrap();
    publish(&parent_path, BYTES, 1024).unwrap();
    let parent = unix::open_exact_directory(&parent_path, None).unwrap();
    let identity = unix::named_file_identity(&parent, FINAL, 1, Some(BYTES.len() as u64)).unwrap();
    let detached = parent_path.join("detached-final-evidence");

    assert!(
        unix::verify_named_file_with(&parent, FINAL, identity, BYTES, 1, || {
            std::fs::rename(parent_path.join(FINAL), &detached)?;
            std::fs::write(parent_path.join(FINAL), BYTES)?;
            std::fs::set_permissions(
                parent_path.join(FINAL),
                std::fs::Permissions::from_mode(0o600),
            )?;
            Ok(())
        })
        .is_err()
    );
    assert_eq!(std::fs::read(detached).unwrap(), BYTES);
    assert_eq!(std::fs::read(parent_path.join(FINAL)).unwrap(), BYTES);
}

#[test]
fn retained_parent_descriptor_prevents_namespace_redirection() {
    let root = tempfile::tempdir().unwrap();
    let base = root.path().canonicalize().unwrap();
    let parent = base.join("records");
    let detached = base.join("detached-records");
    let replacement = base.join("replacement");
    std::fs::create_dir(&parent).unwrap();
    std::fs::create_dir(&replacement).unwrap();
    let identity = directory_identity(&parent).unwrap();
    let mut swapped = false;
    let result = publish_exact_private_file_with(
        &parent,
        identity,
        FINAL,
        PARTIAL,
        BYTES,
        1024,
        |barrier| {
            if !swapped && barrier == PublicationBarrier::ParentOpened {
                swapped = true;
                std::fs::rename(&parent, &detached).unwrap();
                symlink(&replacement, &parent).unwrap();
            }
            Ok(())
        },
    );
    assert!(result.is_err());
    assert!(swapped);
    assert!(!replacement.join(FINAL).exists());
    assert!(!replacement.join(PARTIAL).exists());
    assert_eq!(std::fs::read(detached.join(FINAL)).unwrap(), BYTES);
}

#[test]
fn restrictive_umask_cannot_strand_an_unusable_partial() {
    const CHILD_ENV: &str = "HF2Q_TEST_PREPARED_PROFILE_UMASK_CHILD";
    const ROOT_ENV: &str = "HF2Q_TEST_PREPARED_PROFILE_UMASK_ROOT";
    if std::env::var_os(CHILD_ENV).is_some() {
        let parent = PathBuf::from(std::env::var_os(ROOT_ENV).unwrap());
        // SAFETY: this executes in a dedicated child process before it starts
        // any worker thread, and the process exits immediately after the test.
        unsafe { libc::umask(0o777) };
        publish(&parent, BYTES, 1024).unwrap();
        return;
    }

    let root = tempfile::tempdir().unwrap();
    let parent = root.path().canonicalize().unwrap();
    let status = Command::new(std::env::current_exe().unwrap())
        .arg("input::hf_download::resolution::payload::source::conversion::publication::file::tests::restrictive_umask_cannot_strand_an_unusable_partial")
        .arg("--exact")
        .arg("--test-threads=1")
        .env(CHILD_ENV, "1")
        .env(ROOT_ENV, &parent)
        .status()
        .unwrap();
    assert!(status.success());
    assert_eq!(
        std::fs::symlink_metadata(parent.join(FINAL))
            .unwrap()
            .mode()
            & 0o7777,
        0o600
    );
    assert!(!parent.join(PARTIAL).exists());
}
