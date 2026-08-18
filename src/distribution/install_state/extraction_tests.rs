use std::io::{Cursor, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::os::unix::process::ExitStatusExt;

use serde_json::json;
use sha2::{Digest, Sha256};

use super::*;
use crate::distribution::install_state::locked::LockedInstallation;
use crate::distribution::schema::ReleaseManifestV1;

const PAYLOADS: [(&str, &[u8], &str); 3] = [
    ("bin/hf2q", b"signed binary\n", "0755"),
    ("share/doc/hf2q/README.md", b"documentation\n", "0644"),
    ("share/licenses/hf2q/LICENSE", b"license\n", "0644"),
];

fn manifest() -> (ReleaseManifestV1, Vec<u8>) {
    let files: Vec<_> = PAYLOADS
        .iter()
        .map(|(path, bytes, mode)| {
            json!({
                "path": path,
                "type": "regular",
                "size": bytes.len(),
                "mode": mode,
                "sha256": hex::encode(Sha256::digest(bytes)),
            })
        })
        .collect();
    let raw = serde_json::to_vec(&json!({
        "kind": "hf2q.release-manifest",
        "schema_version": 1,
        "package": "hf2q",
        "version": "0.2.0",
        "target": "aarch64-apple-darwin",
        "minimum_macos": "14.0",
        "source_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "channel": "stable",
        "code_signing": {
            "team_id": "A1B2C3D4E5",
            "identifier": "us.hf2q.cli",
            "certificate_common_name": "Developer ID Application: hf2q (A1B2C3D4E5)"
        },
        "compatibility": {
            "minimum_installer_protocol": 1,
            "minimum_updater_protocol": 1,
            "launcher_registry_schema": 1
        },
        "files": files,
        "non_system_dynamic_dependencies": []
    }))
    .expect("manifest JSON");
    let parsed = ReleaseManifestV1::parse_and_validate(&raw).expect("valid manifest");
    let deterministic = parsed
        .to_deterministic_json()
        .expect("deterministic manifest");
    (parsed, deterministic)
}

fn root() -> (tempfile::TempDir, std::path::PathBuf) {
    let temp = tempfile::tempdir().expect("temporary state parent");
    let root = temp
        .path()
        .canonicalize()
        .expect("canonical temporary parent")
        .join("state");
    (temp, root)
}

fn authorization() -> ExtractionStageAuthorization {
    ExtractionStageAuthorization::for_test("0.2.0", b"archive identity")
}

fn resume_all(
    stage: &mut ReleaseExtractionStage<'_>,
    manifest: &ReleaseManifestV1,
    manifest_bytes: &[u8],
) -> Result<(), ExtractionError> {
    stage.resume_manifest(&mut Cursor::new(manifest_bytes))?;
    for (file, (_, bytes, _)) in manifest.files().iter().zip(PAYLOADS) {
        stage.resume_payload(file, &mut Cursor::new(bytes))?;
    }
    Ok(())
}

#[test]
fn exact_authenticated_replay_repairs_the_last_private_file_without_deleting_it() {
    let (_temp, root) = root();
    let locked = LockedInstallation::acquire(&root).expect("installation lock");
    let (manifest, manifest_bytes) = manifest();
    let stage_name = authorization().stage_name();

    let mut stage = open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
        .expect("new extraction");
    resume_all(&mut stage, &manifest, &manifest_bytes).expect("initial extraction");
    drop(stage.finish().expect("complete extraction"));

    let last = root
        .join("update/extractions")
        .join(&stage_name)
        .join(PAYLOADS.last().expect("last payload").0);
    let before = std::fs::metadata(&last).expect("last payload metadata");
    let mut corrupt = PAYLOADS.last().expect("last payload").1.to_vec();
    corrupt.fill(b'x');
    std::fs::OpenOptions::new()
        .write(true)
        .open(&last)
        .expect("open retained private file")
        .write_all(&corrupt)
        .expect("simulate storage-crash data loss");

    let mut resumed = open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
        .expect("safe residue is repairable");
    resume_all(&mut resumed, &manifest, &manifest_bytes).expect("exact replay repairs");
    drop(resumed.finish().expect("repaired extraction"));

    let after = std::fs::metadata(&last).expect("repaired payload metadata");
    assert_eq!(
        before.ino(),
        after.ino(),
        "repair must not replace the file"
    );
    assert_eq!(
        std::fs::read(&last).expect("repaired bytes"),
        PAYLOADS.last().expect("last payload").1
    );
}

#[test]
fn unexpected_tree_entries_fail_closed_and_are_never_deleted() {
    let (_temp, root) = root();
    let locked = LockedInstallation::acquire(&root).expect("installation lock");
    let (manifest, manifest_bytes) = manifest();
    let stage_name = authorization().stage_name();

    let stage = open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
        .expect("new extraction");
    drop(stage);
    let unexpected = root
        .join("update/extractions")
        .join(stage_name)
        .join("unexpected");
    std::fs::write(&unexpected, b"preserve hostile evidence").expect("unexpected file");
    std::fs::set_permissions(&unexpected, std::fs::Permissions::from_mode(0o600))
        .expect("private mode");

    assert!(matches!(
        open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest,),
        Err(ExtractionError::InstallState(_)) | Err(ExtractionError::Integrity)
    ));
    assert_eq!(
        std::fs::read(&unexpected).expect("unexpected evidence retained"),
        b"preserve hostile evidence"
    );
}

#[test]
fn hostile_expected_nodes_and_nonprefix_trees_fail_closed() {
    enum Hostile {
        Symlink,
        Hardlink,
        Fifo,
        WrongMode,
        Oversized,
        CompleteAfterAbsent,
    }

    for hostile in [
        Hostile::Symlink,
        Hostile::Hardlink,
        Hostile::Fifo,
        Hostile::WrongMode,
        Hostile::Oversized,
        Hostile::CompleteAfterAbsent,
    ] {
        let (_temp, root) = root();
        let locked = LockedInstallation::acquire(&root).expect("installation lock");
        let (manifest, manifest_bytes) = manifest();
        let stage_name = authorization().stage_name();
        let stage = open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
            .expect("new extraction");
        drop(stage);
        let stage = root.join("update/extractions").join(stage_name);
        let manifest_path = stage.join(MANIFEST_NAME);
        match hostile {
            Hostile::Symlink => {
                std::os::unix::fs::symlink("elsewhere", &manifest_path).expect("hostile symlink");
            }
            Hostile::Hardlink => {
                std::fs::write(&manifest_path, &manifest_bytes).expect("manifest file");
                std::fs::set_permissions(&manifest_path, std::fs::Permissions::from_mode(0o600))
                    .expect("private mode");
                std::fs::hard_link(&manifest_path, root.join("hostile-hardlink-evidence"))
                    .expect("hostile hardlink");
                assert_eq!(
                    std::fs::metadata(&manifest_path)
                        .expect("manifest metadata")
                        .nlink(),
                    2,
                    "fixture must exercise the single-link invariant"
                );
            }
            Hostile::Fifo => {
                let path = std::ffi::CString::new(manifest_path.as_os_str().as_bytes())
                    .expect("FIFO path");
                // SAFETY: `path` is a live NUL-terminated pathname and the
                // return value is checked before the scanner opens it.
                assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
            }
            Hostile::WrongMode => {
                std::fs::write(&manifest_path, &manifest_bytes).expect("manifest file");
                std::fs::set_permissions(&manifest_path, std::fs::Permissions::from_mode(0o644))
                    .expect("wrong mode");
            }
            Hostile::Oversized => {
                let mut bytes = manifest_bytes.clone();
                bytes.push(b'!');
                std::fs::write(&manifest_path, bytes).expect("oversized manifest");
                std::fs::set_permissions(&manifest_path, std::fs::Permissions::from_mode(0o600))
                    .expect("private mode");
            }
            Hostile::CompleteAfterAbsent => {
                let binary = stage.join("bin/hf2q");
                std::fs::create_dir(stage.join("bin")).expect("derived directory");
                std::fs::set_permissions(stage.join("bin"), std::fs::Permissions::from_mode(0o700))
                    .expect("private directory mode");
                std::fs::write(&binary, PAYLOADS[0].1).expect("out-of-order complete payload");
                std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o600))
                    .expect("private file mode");
            }
        }

        assert!(
            open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest,).is_err()
        );
        assert!(stage.exists(), "hostile evidence must be retained");
    }
}

#[test]
fn detached_extractions_namespace_cannot_finish() {
    let (_temp, root) = root();
    let locked = LockedInstallation::acquire(&root).expect("installation lock");
    let (manifest, manifest_bytes) = manifest();
    let mut stage = open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
        .expect("new extraction");
    resume_all(&mut stage, &manifest, &manifest_bytes).expect("complete staged bytes");

    let update = root.join("update");
    std::fs::rename(
        update.join("extractions"),
        update.join("extractions-detached"),
    )
    .expect("detach extraction namespace");
    std::fs::create_dir(update.join("extractions")).expect("replacement namespace");
    std::fs::set_permissions(
        update.join("extractions"),
        std::fs::Permissions::from_mode(0o700),
    )
    .expect("replacement mode");

    assert!(stage.finish().is_err());
    assert!(
        update
            .join("extractions-detached")
            .join(authorization().stage_name())
            .join(MANIFEST_NAME)
            .exists(),
        "detached evidence must not be deleted"
    );
}

#[test]
fn retained_stage_cap_allows_current_resume_but_rejects_a_ninth_stage() {
    let (_temp, root) = root();
    let locked = LockedInstallation::acquire(&root).expect("installation lock");
    let extractions = unix::ensure_private_directory(locked.update(), EXTRACTIONS)
        .expect("extractions directory");
    let mut names = Vec::new();
    for value in 0..MAX_RETAINED_EXTRACTIONS {
        let name = format!(".extract-v0.2.{value}-{}", format!("{value:064x}"));
        unix::ensure_private_directory(&extractions, &name).expect("retained stage");
        names.push(name);
    }

    validate_retained_stages(&extractions, &names[0]).expect("current stage resumes at cap");
    let ninth = format!(".extract-v9.9.9-{}", "f".repeat(64));
    assert!(matches!(
        validate_retained_stages(&extractions, &ninth),
        Err(ExtractionError::Integrity)
    ));
    assert_eq!(
        unix::list_names_bounded(&extractions, MAX_RETAINED_EXTRACTIONS)
            .expect("unchanged retained inventory")
            .len(),
        MAX_RETAINED_EXTRACTIONS
    );
}

#[test]
fn malformed_retained_siblings_and_incomplete_finish_are_preserved_and_rejected() {
    let (_temp, root) = root();
    let locked = LockedInstallation::acquire(&root).expect("installation lock");
    let (manifest, manifest_bytes) = manifest();
    let mut stage = open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
        .expect("new extraction");
    stage
        .resume_manifest(&mut Cursor::new(&manifest_bytes))
        .expect("only manifest completes");
    assert!(matches!(stage.finish(), Err(ExtractionError::Integrity)));

    let extractions = root.join("update/extractions");
    let malformed = extractions.join("not-an-authenticated-stage");
    std::fs::create_dir(&malformed).expect("malformed sibling");
    std::fs::set_permissions(&malformed, std::fs::Permissions::from_mode(0o700))
        .expect("private sibling mode");
    assert!(
        open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest,).is_err()
    );
    assert!(
        malformed.is_dir(),
        "malformed sibling is retained as evidence"
    );
}

#[test]
fn storage_quota_mapping_applies_only_to_destination_writes() {
    for code in [libc::ENOSPC, libc::EDQUOT] {
        assert!(matches!(
            ExtractionError::write_io(std::io::Error::from_raw_os_error(code)),
            ExtractionError::StorageFull
        ));
        assert!(matches!(
            ExtractionError::read_io(std::io::Error::from_raw_os_error(code)),
            ExtractionError::Io(_)
        ));
    }
}

#[test]
fn process_abort_mid_file_leaves_only_resumable_inert_state() {
    const CHILD: &str = "HF2Q_EXTRACTION_ABORT_CHILD";
    const ROOT: &str = "HF2Q_EXTRACTION_ABORT_ROOT";

    if std::env::var_os(CHILD).is_some() {
        let root = std::path::PathBuf::from(std::env::var_os(ROOT).expect("child root"));
        let locked = LockedInstallation::acquire(&root).expect("child installation lock");
        let (manifest, manifest_bytes) = manifest();
        let mut stage =
            open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
                .expect("child extraction");
        abort_after_next_extraction_write();
        let _ = stage.resume_manifest(&mut Cursor::new(&manifest_bytes));
        panic!("the child did not abort at the extraction write barrier");
    }

    let (_temp, root) = root();
    let status = std::process::Command::new(std::env::current_exe().expect("test executable"))
        .arg("--exact")
        .arg(
            "distribution::install_state::extraction::tests::process_abort_mid_file_leaves_only_resumable_inert_state",
        )
        .arg("--nocapture")
        .env(CHILD, "1")
        .env(ROOT, &root)
        .status()
        .expect("run abort child");
    assert_eq!(
        status.signal(),
        Some(libc::SIGABRT),
        "child must stop specifically at the extraction SIGABRT barrier"
    );

    let locked = LockedInstallation::acquire(&root).expect("fresh-process installation lock");
    let (manifest, manifest_bytes) = manifest();
    let mut stage = open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
        .expect("fresh process reopens exact residue");
    resume_all(&mut stage, &manifest, &manifest_bytes).expect("fresh process resumes");
    drop(stage.finish().expect("fresh process finishes inert tree"));
    assert!(!root.join("versions").exists());
    assert!(!root.join("activations").exists());
    assert!(!root.join("current").exists());
}
