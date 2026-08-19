use std::io::{Cursor, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::os::unix::process::ExitStatusExt;

use serde_json::json;
use sha2::{Digest, Sha256};

use super::*;
use crate::distribution::install_state::{
    bootstrap_installation_identity_for_test, ExplicitRootAuthorization, IdentityFaultPlan,
    LockedInstallationIdentity,
};
use crate::distribution::schema::ReleaseManifestV1;

const INSTALLATION_ID: &str = "550e8400-e29b-41d4-a716-446655440000";

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

fn locked_identity(root: &std::path::Path) -> LockedInstallationIdentity {
    bootstrap_installation_identity_for_test(
        ExplicitRootAuthorization::new(root).expect("root authorization"),
        INSTALLATION_ID,
        IdentityFaultPlan::default(),
    )
    .expect("installation identity")
    .into_identity()
    .lock()
    .expect("installation lock")
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
    let locked = locked_identity(&root);
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
fn exact_final_mode_prefix_is_resumable_after_a_normalization_crash() {
    let (_temp, root) = root();
    let locked = locked_identity(&root);
    let (manifest, manifest_bytes) = manifest();
    let stage_name = authorization().stage_name();

    let mut stage = open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
        .expect("new extraction");
    resume_all(&mut stage, &manifest, &manifest_bytes).expect("initial extraction");
    drop(stage.finish().expect("complete private extraction"));

    let staged_manifest = root
        .join("update/extractions")
        .join(stage_name)
        .join(MANIFEST_NAME);
    std::fs::set_permissions(&staged_manifest, std::fs::Permissions::from_mode(0o644))
        .expect("simulate the first completed normalization step");

    let mut resumed = open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
        .expect("final-mode prefix is valid restart state");
    resume_all(&mut resumed, &manifest, &manifest_bytes).expect("exact replay resumes");
    drop(
        resumed
            .finish()
            .expect("resumed extraction remains complete"),
    );
    assert_eq!(
        std::fs::metadata(staged_manifest)
            .expect("normalized manifest metadata")
            .permissions()
            .mode()
            & 0o777,
        0o644
    );
}

#[test]
fn finish_rejects_an_out_of_order_final_file_created_after_the_last_write() {
    let (_temp, root) = root();
    let locked = locked_identity(&root);
    let (manifest, manifest_bytes) = manifest();
    let stage_name = authorization().stage_name();
    let mut stage = open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
        .expect("new extraction");
    resume_all(&mut stage, &manifest, &manifest_bytes).expect("complete extraction writes");

    let binary = root
        .join("update/extractions")
        .join(stage_name)
        .join("bin/hf2q");
    std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755))
        .expect("simulate an out-of-order final mode after the last write");
    assert!(matches!(stage.finish(), Err(ExtractionError::Integrity)));
}

#[cfg(target_os = "macos")]
#[test]
fn developer_id_proof_for_one_tree_cannot_normalize_another_tree() {
    let (_temp_a, root_a) = root();
    let locked_a = locked_identity(&root_a);
    let (manifest_a, manifest_bytes_a) = manifest();
    let mut stage_a =
        open_release_extraction(&locked_a, authorization(), &manifest_bytes_a, &manifest_a)
            .expect("first extraction");
    resume_all(&mut stage_a, &manifest_a, &manifest_bytes_a).expect("first exact extraction");
    let tree_a = stage_a.finish().expect("first inert tree");
    let proof_a = modes::with_extracted_executable(
        &locked_a,
        &tree_a,
        &manifest_bytes_a,
        &manifest_a,
        |_path, _file, binding| {
            Ok::<_, ExtractionError>(
                crate::distribution::prepared_release::DeveloperIdVerification::for_test(binding),
            )
        },
    )
    .expect("first exact tree proof");

    let (_temp_b, root_b) = root();
    let locked_b = locked_identity(&root_b);
    let (manifest_b, manifest_bytes_b) = manifest();
    let mut stage_b =
        open_release_extraction(&locked_b, authorization(), &manifest_bytes_b, &manifest_b)
            .expect("second extraction");
    resume_all(&mut stage_b, &manifest_b, &manifest_bytes_b).expect("second exact extraction");
    let tree_b = stage_b.finish().expect("second inert tree");

    assert!(matches!(
        modes::normalize_developer_id_verified_release(
            &locked_b,
            proof_a,
            tree_b,
            &manifest_bytes_b,
            &manifest_b,
        ),
        Err(ExtractionError::Integrity)
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn native_callback_cannot_replace_the_executable_inode_inside_one_bracket() {
    let (_temp, root) = root();
    let locked = locked_identity(&root);
    let (manifest, manifest_bytes) = manifest();
    let mut stage = open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
        .expect("new extraction");
    resume_all(&mut stage, &manifest, &manifest_bytes).expect("exact extraction");
    let tree = stage.finish().expect("inert tree");

    let result = modes::with_extracted_executable(
        &locked,
        &tree,
        &manifest_bytes,
        &manifest,
        |path, _file, _binding| {
            let displaced = path.with_extension("displaced");
            std::fs::rename(path, &displaced).expect("displace verified executable name");
            std::fs::write(path, PAYLOADS[0].1).expect("install same-byte replacement");
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
                .expect("replacement private mode");
            Ok::<_, ExtractionError>(())
        },
    );
    assert!(matches!(result, Err(ExtractionError::Integrity)));
}

#[test]
fn completed_inert_tree_normalizes_only_to_manifest_modes() {
    let (_temp, root) = root();
    let locked = locked_identity(&root);
    let (manifest, manifest_bytes) = manifest();
    let stage_name = authorization().stage_name();

    let mut stage = open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
        .expect("new extraction");
    resume_all(&mut stage, &manifest, &manifest_bytes).expect("initial extraction");
    let tree = stage.finish().expect("complete private extraction");
    let normalized = normalize_release_tree(&locked, tree, &manifest_bytes, &manifest)
        .expect("durable signed-mode normalization");
    drop(normalized);

    let stage = root.join("update/extractions").join(stage_name);
    assert_eq!(
        std::fs::metadata(stage.join(MANIFEST_NAME))
            .expect("manifest metadata")
            .permissions()
            .mode()
            & 0o777,
        0o644
    );
    for (path, _, mode) in PAYLOADS {
        assert_eq!(
            std::fs::metadata(stage.join(path))
                .expect("payload metadata")
                .permissions()
                .mode()
                & 0o777,
            u32::from_str_radix(mode, 8).expect("octal mode")
        );
    }
    for directory in manifest.derived_directories() {
        assert_eq!(
            std::fs::metadata(stage.join(directory))
                .expect("derived directory metadata")
                .permissions()
                .mode()
                & 0o777,
            0o755
        );
    }
    assert_eq!(
        std::fs::metadata(stage)
            .expect("stage metadata")
            .permissions()
            .mode()
            & 0o777,
        0o700,
        "the deterministic stage root remains private"
    );
    assert!(!root.join("versions").exists());
    assert!(!root.join("activations").exists());
    assert!(!root.join("current").exists());
}

#[test]
fn normalization_is_idempotent_and_a_final_directory_prefix_resumes() {
    let (_temp, root) = root();
    let locked = locked_identity(&root);
    let (manifest, manifest_bytes) = manifest();
    let stage_name = authorization().stage_name();

    let mut stage = open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
        .expect("new extraction");
    resume_all(&mut stage, &manifest, &manifest_bytes).expect("initial extraction");
    let tree = stage.finish().expect("complete private extraction");
    drop(
        normalize_release_tree(&locked, tree, &manifest_bytes, &manifest)
            .expect("first normalization"),
    );

    let stage_path = root.join("update/extractions").join(&stage_name);
    let ordered = directory_normalization_order(manifest.derived_directories());
    assert!(ordered.len() > 2, "fixture needs a directory prefix");
    for path in ordered.iter().skip(2) {
        std::fs::set_permissions(
            stage_path.join(path),
            std::fs::Permissions::from_mode(0o700),
        )
        .expect("simulate a crash after a final directory prefix");
    }

    let mut resumed = open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
        .expect("canonical final directory prefix resumes");
    resume_all(&mut resumed, &manifest, &manifest_bytes).expect("replay exact final files");
    let resumed_tree = resumed.finish().expect("finish resumed tree");
    drop(
        normalize_release_tree(&locked, resumed_tree, &manifest_bytes, &manifest)
            .expect("idempotent normalization completes the suffix"),
    );
    for path in manifest.derived_directories() {
        assert_eq!(
            std::fs::metadata(stage_path.join(path))
                .expect("normalized directory metadata")
                .permissions()
                .mode()
                & 0o777,
            0o755
        );
    }
    assert!(!root.join("versions").exists());
    assert!(!root.join("activations").exists());
    assert!(!root.join("current").exists());
}

#[test]
fn out_of_order_final_file_and_directory_modes_fail_closed() {
    {
        let (_temp, root) = root();
        let locked = locked_identity(&root);
        let (manifest, manifest_bytes) = manifest();
        let stage_name = authorization().stage_name();
        let mut stage =
            open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
                .expect("new extraction");
        resume_all(&mut stage, &manifest, &manifest_bytes).expect("initial extraction");
        drop(stage.finish().expect("complete private extraction"));

        let binary = root
            .join("update/extractions")
            .join(stage_name)
            .join("bin/hf2q");
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755))
            .expect("out-of-order final file mode");
        assert!(
            open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest).is_err(),
            "a final file after a private predecessor must fail closed"
        );
        assert_eq!(
            std::fs::metadata(binary)
                .expect("hostile file retained")
                .permissions()
                .mode()
                & 0o777,
            0o755
        );
    }

    {
        let (_temp, root) = root();
        let locked = locked_identity(&root);
        let (manifest, manifest_bytes) = manifest();
        let stage_name = authorization().stage_name();
        let mut stage =
            open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
                .expect("new extraction");
        resume_all(&mut stage, &manifest, &manifest_bytes).expect("initial extraction");
        let tree = stage.finish().expect("complete private extraction");
        drop(
            normalize_release_tree(&locked, tree, &manifest_bytes, &manifest)
                .expect("complete normalization"),
        );

        let stage_path = root.join("update/extractions").join(stage_name);
        let ordered = directory_normalization_order(manifest.derived_directories());
        assert!(ordered.len() > 1, "fixture needs multiple directories");
        std::fs::set_permissions(
            stage_path.join(ordered[0]),
            std::fs::Permissions::from_mode(0o700),
        )
        .expect("private directory before a final successor");
        assert!(
            open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest).is_err(),
            "an out-of-order final directory must fail closed"
        );
        assert_eq!(
            std::fs::metadata(stage_path.join(ordered[0]))
                .expect("hostile directory retained")
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
    }
}

#[test]
fn corrupted_final_mode_file_is_preserved_and_never_repaired() {
    let (_temp, root) = root();
    let locked = locked_identity(&root);
    let (manifest, manifest_bytes) = manifest();
    let stage_name = authorization().stage_name();
    let mut stage = open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
        .expect("new extraction");
    resume_all(&mut stage, &manifest, &manifest_bytes).expect("initial extraction");
    let tree = stage.finish().expect("complete private extraction");
    drop(
        normalize_release_tree(&locked, tree, &manifest_bytes, &manifest)
            .expect("complete normalization"),
    );

    let binary = root
        .join("update/extractions")
        .join(stage_name)
        .join("bin/hf2q");
    let before = std::fs::metadata(&binary).expect("binary metadata");
    let corrupt = vec![b'x'; PAYLOADS[0].1.len()];
    std::fs::OpenOptions::new()
        .write(true)
        .open(&binary)
        .expect("open final-mode binary")
        .write_all(&corrupt)
        .expect("simulate post-normalization corruption");

    assert!(
        open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest).is_err(),
        "final-mode corruption must never enter the reconstruction path"
    );
    let after = std::fs::metadata(&binary).expect("retained corrupt binary metadata");
    assert_eq!(before.ino(), after.ino());
    assert_eq!(
        std::fs::read(binary).expect("retained corrupt bytes"),
        corrupt
    );
}

#[test]
fn unexpected_tree_entries_fail_closed_and_are_never_deleted() {
    let (_temp, root) = root();
    let locked = locked_identity(&root);
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
        let locked = locked_identity(&root);
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
                std::fs::set_permissions(&manifest_path, std::fs::Permissions::from_mode(0o666))
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
    let locked = locked_identity(&root);
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
fn replaced_identity_inode_cannot_finish_extraction() {
    let (_temp, root) = root();
    let locked = locked_identity(&root);
    let (manifest, manifest_bytes) = manifest();
    let mut stage = open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
        .expect("new extraction");
    resume_all(&mut stage, &manifest, &manifest_bytes).expect("complete staged bytes");

    let identity = root.join("update/installation-identity.json");
    let bytes = std::fs::read(&identity).expect("identity bytes");
    std::fs::rename(&identity, root.join("detached-installation-identity.json"))
        .expect("detach identity inode");
    std::fs::write(&identity, bytes).expect("same-byte identity replacement");
    std::fs::set_permissions(&identity, std::fs::Permissions::from_mode(0o600))
        .expect("private identity replacement");

    assert!(stage.finish().is_err());
    assert!(!root.join("versions").exists());
    assert!(!root.join("activations").exists());
    assert!(!root.join("current").exists());
}

#[test]
fn retained_stage_cap_allows_current_resume_but_rejects_a_ninth_stage() {
    let (_temp, root) = root();
    let locked = locked_identity(&root);
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
    let locked = locked_identity(&root);
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
        let locked = locked_identity(&root);
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

    let locked = locked_identity(&root);
    let (manifest, manifest_bytes) = manifest();
    let mut stage = open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
        .expect("fresh process reopens exact residue");
    resume_all(&mut stage, &manifest, &manifest_bytes).expect("fresh process resumes");
    drop(stage.finish().expect("fresh process finishes inert tree"));
    assert!(!root.join("versions").exists());
    assert!(!root.join("activations").exists());
    assert!(!root.join("current").exists());
}

#[test]
fn every_normalization_barrier_is_exactly_restartable_after_sigabrt() {
    const CHILD: &str = "HF2Q_NORMALIZATION_ABORT_CHILD";
    const ROOT: &str = "HF2Q_NORMALIZATION_ABORT_ROOT";
    const BARRIER: &str = "HF2Q_NORMALIZATION_ABORT_BARRIER";

    if std::env::var_os(CHILD).is_some() {
        let root = std::path::PathBuf::from(std::env::var_os(ROOT).expect("child root"));
        let barrier: usize = std::env::var(BARRIER)
            .expect("child barrier")
            .parse()
            .expect("numeric child barrier");
        let locked = locked_identity(&root);
        let (manifest, manifest_bytes) = manifest();
        let mut stage =
            open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
                .expect("child extraction");
        resume_all(&mut stage, &manifest, &manifest_bytes).expect("child exact extraction");
        let tree = stage.finish().expect("child inert tree");
        modes::abort_after_normalization_barrier(barrier);
        let _ = normalize_release_tree(&locked, tree, &manifest_bytes, &manifest);
        panic!("the child did not abort at normalization barrier {barrier}");
    }

    let (fixture_manifest, _) = manifest();
    let barrier_count = modes::normalization_barrier_count(&fixture_manifest);
    assert!(
        barrier_count > 10,
        "fixture must exercise the full barrier matrix"
    );
    for barrier in 1..=barrier_count {
        let (_temp, root) = root();
        let status = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .arg("--exact")
            .arg(
                "distribution::install_state::extraction::tests::every_normalization_barrier_is_exactly_restartable_after_sigabrt",
            )
            .arg("--nocapture")
            .env(CHILD, "1")
            .env(ROOT, &root)
            .env(BARRIER, barrier.to_string())
            .status()
            .expect("run normalization abort child");
        assert_eq!(
            status.signal(),
            Some(libc::SIGABRT),
            "barrier {barrier} must terminate specifically through SIGABRT"
        );

        let locked = locked_identity(&root);
        let (manifest, manifest_bytes) = manifest();
        let mut stage =
            open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
                .unwrap_or_else(|error| panic!("barrier {barrier} residue must reopen: {error}"));
        resume_all(&mut stage, &manifest, &manifest_bytes)
            .unwrap_or_else(|error| panic!("barrier {barrier} must replay: {error}"));
        let tree = stage
            .finish()
            .unwrap_or_else(|error| panic!("barrier {barrier} must finish: {error}"));
        drop(
            normalize_release_tree(&locked, tree, &manifest_bytes, &manifest)
                .unwrap_or_else(|error| panic!("barrier {barrier} must normalize: {error}")),
        );
        assert!(!root.join("versions").exists());
        assert!(!root.join("activations").exists());
        assert!(!root.join("current").exists());
    }
}

#[test]
fn every_normalization_barrier_is_exactly_retryable_after_returned_error() {
    let (fixture_manifest, _) = manifest();
    let barrier_count = modes::normalization_barrier_count(&fixture_manifest);
    for barrier in 1..=barrier_count {
        let (_temp, root) = root();
        let locked = locked_identity(&root);
        let (manifest, manifest_bytes) = manifest();
        let mut stage =
            open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
                .expect("new extraction");
        resume_all(&mut stage, &manifest, &manifest_bytes).expect("exact extraction");
        let tree = stage.finish().expect("complete inert tree");
        modes::fail_after_normalization_barrier(barrier);
        assert!(matches!(
            normalize_release_tree(&locked, tree, &manifest_bytes, &manifest),
            Err(ExtractionError::Integrity)
        ));

        let mut retry =
            open_release_extraction(&locked, authorization(), &manifest_bytes, &manifest)
                .unwrap_or_else(|error| panic!("barrier {barrier} residue must reopen: {error}"));
        resume_all(&mut retry, &manifest, &manifest_bytes)
            .unwrap_or_else(|error| panic!("barrier {barrier} must replay: {error}"));
        let tree = retry
            .finish()
            .unwrap_or_else(|error| panic!("barrier {barrier} must finish: {error}"));
        drop(
            normalize_release_tree(&locked, tree, &manifest_bytes, &manifest)
                .unwrap_or_else(|error| panic!("barrier {barrier} must normalize: {error}")),
        );
        assert!(!root.join("versions").exists());
        assert!(!root.join("activations").exists());
        assert!(!root.join("current").exists());
    }
}
