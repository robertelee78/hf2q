use super::*;
use crate::distribution::install_state::ExplicitRootAuthorization;
use crate::distribution::schema::Sha256Digest;
use tempfile::TempDir;

fn authorization(root: &std::path::Path) -> MetadataStateAuthorization {
    MetadataStateAuthorization::for_test(
        ExplicitRootAuthorization::new(root).expect("root authorization"),
        "11111111-1111-4111-8111-111111111111",
    )
}

#[test]
fn staged_archive_is_unlinked_streamed_and_rehashed() {
    let temp = TempDir::new().expect("tempdir");
    let temp_path = temp.path().canonicalize().expect("canonical tempdir");
    std::fs::create_dir(temp_path.join("state")).expect("state root");
    std::fs::set_permissions(
        temp_path.join("state"),
        std::os::unix::fs::PermissionsExt::from_mode(0o700),
    )
    .expect("state permissions");
    let auth = authorization(&temp_path.join("state"));
    let bytes = b"authenticated archive bytes";
    let digest = Sha256Digest::parse("sha256", hex::encode(Sha256::digest(bytes))).expect("digest");
    let mut stage =
        create_ephemeral_artifact_stage(&auth, bytes.len() as u64).expect("create stage");
    let staged_debug = format!("{stage:?}");
    for secret in ["fd", "path", ".artifact-", "read: ", "write: "] {
        assert!(!staged_debug.contains(secret));
    }
    stage.write_chunk(&bytes[..7]).expect("first chunk");
    stage.write_chunk(&bytes[7..]).expect("second chunk");
    let mut verified = stage.finish(&digest).expect("finish");
    let verified_debug = format!("{verified:?}");
    for secret in ["fd", "path", ".artifact-", "read: ", "write: "] {
        assert!(!verified_debug.contains(secret));
    }
    verified.revalidate().expect("revalidate");
    assert_eq!(verified.length(), bytes.len() as u64);
    assert_eq!(verified.sha256(), &digest);
    let mut reread = Vec::new();
    verified.read_to_end(&mut reread).expect("read verified");
    assert_eq!(reread, bytes);
    assert!(std::fs::read_dir(temp_path.join("state/update/downloads"))
        .expect("downloads")
        .next()
        .is_none());
}

#[test]
fn overrun_and_digest_mismatch_never_finish() {
    let temp = TempDir::new().expect("tempdir");
    let temp_path = temp.path().canonicalize().expect("canonical tempdir");
    std::fs::create_dir(temp_path.join("state")).expect("state root");
    std::fs::set_permissions(
        temp_path.join("state"),
        std::os::unix::fs::PermissionsExt::from_mode(0o700),
    )
    .expect("state permissions");
    let auth = authorization(&temp_path.join("state"));
    let mut stage = create_ephemeral_artifact_stage(&auth, 3).expect("stage");
    assert!(matches!(
        stage.write_chunk(b"four"),
        Err(ArtifactStageError::Integrity)
    ));

    let mut stage = create_ephemeral_artifact_stage(&auth, 3).expect("stage");
    stage.write_chunk(b"abc").expect("write");
    let wrong = Sha256Digest::parse("sha256", "0".repeat(64)).expect("digest");
    assert!(matches!(
        stage.finish(&wrong),
        Err(ArtifactStageError::Integrity)
    ));

    let expected =
        Sha256Digest::parse("sha256", hex::encode(Sha256::digest(b"abc"))).expect("digest");
    let mut stage = create_ephemeral_artifact_stage(&auth, 3).expect("stage");
    stage.write_chunk(b"abc").expect("write");
    let mut verified = stage.finish(&expected).expect("verified");
    verified.file.write_all(b"x").expect("test-only mutation");
    assert!(matches!(
        verified.revalidate(),
        Err(ArtifactStageError::Integrity)
    ));
}

#[test]
fn next_stage_removes_only_exact_private_crash_residue() {
    let temp = TempDir::new().expect("tempdir");
    let temp_path = temp.path().canonicalize().expect("canonical tempdir");
    let state = temp_path.join("state");
    std::fs::create_dir(&state).expect("state root");
    std::fs::set_permissions(&state, std::os::unix::fs::PermissionsExt::from_mode(0o700))
        .expect("state permissions");
    let auth = authorization(&state);
    drop(create_ephemeral_artifact_stage(&auth, 1).expect("bootstrap downloads"));
    let downloads = state.join("update/downloads");
    let residue = downloads.join(format!("{STAGE_PREFIX}00000000000040008000000000000000"));
    std::fs::File::create(&residue).expect("empty v4 residue");
    std::fs::set_permissions(
        &residue,
        std::os::unix::fs::PermissionsExt::from_mode(0o600),
    )
    .expect("residue permissions");
    drop(create_ephemeral_artifact_stage(&auth, 1).expect("recover residue"));
    assert!(!residue.exists());

    let nonempty = downloads.join(format!("{STAGE_PREFIX}00000000000040008000000000000001"));
    std::fs::write(&nonempty, b"hostile").expect("nonempty residue");
    std::fs::set_permissions(
        &nonempty,
        std::os::unix::fs::PermissionsExt::from_mode(0o600),
    )
    .expect("nonempty permissions");
    assert!(matches!(
        create_ephemeral_artifact_stage(&auth, 1),
        Err(ArtifactStageError::InstallState(
            InstallStateError::InvalidLayout(_)
        ))
    ));
    assert!(nonempty.exists());
}

#[test]
fn residue_inventory_and_archive_length_are_bounded_before_mutation() {
    let temp = TempDir::new().expect("tempdir");
    let state = temp.path().canonicalize().expect("canonical").join("state");
    std::fs::create_dir(&state).expect("state root");
    std::fs::set_permissions(&state, std::os::unix::fs::PermissionsExt::from_mode(0o700))
        .expect("state permissions");
    let auth = authorization(&state);
    drop(create_ephemeral_artifact_stage(&auth, 1).expect("bootstrap downloads"));
    let downloads = state.join("update/downloads");
    let first = downloads.join(format!("{STAGE_PREFIX}00000000000040008000000000000000"));
    let second = downloads.join(format!("{STAGE_PREFIX}00000000000040008000000000000001"));
    for entry in [&first, &second] {
        std::fs::File::create(entry).expect("empty residue");
        std::fs::set_permissions(entry, std::os::unix::fs::PermissionsExt::from_mode(0o600))
            .expect("residue permissions");
    }
    assert!(matches!(
        create_ephemeral_artifact_stage(&auth, 1),
        Err(ArtifactStageError::InstallState(
            InstallStateError::InvalidLayout(_)
        ))
    ));
    assert!(first.exists() && second.exists());

    assert!(matches!(
        create_ephemeral_artifact_stage(
            &auth,
            crate::distribution::schema::MAX_RELEASE_ARCHIVE_BYTES + 1
        ),
        Err(ArtifactStageError::Integrity)
    ));
}

#[test]
fn disk_full_errors_have_a_distinct_classification() {
    for code in [libc::ENOSPC, libc::EDQUOT] {
        assert!(matches!(
            ArtifactStageError::io(std::io::Error::from_raw_os_error(code)),
            ArtifactStageError::StorageFull
        ));
        assert!(matches!(
            ArtifactStageError::from(InstallStateError::Io {
                operation: "test directory operation",
                source: std::io::Error::from_raw_os_error(code),
            }),
            ArtifactStageError::StorageFull
        ));
    }
}
