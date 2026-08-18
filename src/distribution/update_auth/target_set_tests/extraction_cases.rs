use std::io::{self, Cursor, Read, Write};
use std::os::unix::fs::PermissionsExt;

use sha2::{Digest, Sha256};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

use super::*;
use crate::distribution::install_state::metadata::MetadataJournalError;
use crate::distribution::prepared_release::{bind_archive, extract_release, PreparedReleaseError};
use crate::distribution::schema::ReleaseManifestV1;
use crate::distribution::update_auth::test_repository::{
    stable_release_repository_for_artifacts, stable_release_repository_for_artifacts_with_expiry,
    stable_release_successor_for_artifacts,
};

const PAYLOADS: [(&str, &[u8], &str); 3] = [
    ("bin/hf2q", b"signed binary\n", "0755"),
    ("share/doc/hf2q/README.md", b"documentation\n", "0644"),
    ("share/licenses/hf2q/LICENSE", b"license\n", "0644"),
];

fn manifest() -> (ReleaseManifestV1, Vec<u8>) {
    let files: Vec<_> = PAYLOADS
        .iter()
        .map(|(path, bytes, mode)| {
            serde_json::json!({
                "path": path,
                "type": "regular",
                "size": bytes.len(),
                "mode": mode,
                "sha256": hex::encode(Sha256::digest(bytes)),
            })
        })
        .collect();
    let raw = serde_json::to_vec(&serde_json::json!({
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
    let parsed = ReleaseManifestV1::parse_and_validate(&raw).expect("extraction manifest");
    let bytes = parsed
        .to_deterministic_json()
        .expect("deterministic manifest");
    (parsed, bytes)
}

fn archive(manifest_bytes: &[u8], compression: CompressionMethod) -> Vec<u8> {
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    write_entry(
        &mut writer,
        "release-manifest.json",
        manifest_bytes,
        0o644,
        compression,
    );
    for (path, bytes, mode) in PAYLOADS {
        write_entry(
            &mut writer,
            path,
            bytes,
            u32::from_str_radix(mode, 8).expect("octal fixture mode"),
            compression,
        );
    }
    writer.finish().expect("finish extraction ZIP").into_inner()
}

fn write_entry(
    writer: &mut ZipWriter<Cursor<Vec<u8>>>,
    name: &str,
    bytes: &[u8],
    mode: u32,
    compression: CompressionMethod,
) {
    let options = SimpleFileOptions::default()
        .compression_method(compression)
        .unix_permissions(mode);
    writer.start_file(name, options).expect("start ZIP entry");
    writer.write_all(bytes).expect("write ZIP entry");
}

fn bundle<'a>(
    authorization: &'a MetadataStateAuthorization,
    anchor: &'a EmbeddedTrustRoot,
    pointer: &[u8],
    manifest: ReleaseManifestV1,
    manifest_bytes: Vec<u8>,
    archive_bytes: &[u8],
) -> crate::distribution::update_transport::VerifiedReleaseBundle<'a> {
    let fetch = begin_artifact_fetch_for_test(
        authorization,
        anchor,
        [
            instant("2026-08-18T09:01:00Z"),
            instant("2026-08-18T09:01:01Z"),
        ],
    )
    .expect("artifact fetch authority");
    let mut bound = fetch.bind_pointer(pointer).expect("bound pointer");
    let archive_sha256 = bound.archive().sha256().clone();
    let mut stage = bound
        .create_archive_stage_for_test([
            instant("2026-08-18T09:01:02Z"),
            instant("2026-08-18T09:01:03Z"),
        ])
        .expect("anonymous archive stage");
    for chunk in archive_bytes.chunks(17) {
        stage.write_chunk(chunk).expect("stream archive chunk");
    }
    let archive = stage.finish(&archive_sha256).expect("verified archive");
    let final_authorization = bound
        .finalize_for_test([
            instant("2026-08-18T09:01:04Z"),
            instant("2026-08-18T09:01:05Z"),
        ])
        .expect("final artifact authorization");
    crate::distribution::update_transport::VerifiedReleaseBundle::from_test_parts(
        final_authorization,
        manifest_bytes.into_boxed_slice(),
        manifest,
        archive,
    )
}

struct FailingPrefix {
    bytes: Cursor<Vec<u8>>,
    remaining: usize,
}

impl Read for FailingPrefix {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Err(io::Error::new(
                io::ErrorKind::ConnectionReset,
                "scripted extraction interruption",
            ));
        }
        let limit = output.len().min(self.remaining);
        let count = self.bytes.read(&mut output[..limit])?;
        self.remaining -= count;
        Ok(count)
    }
}

fn resume(
    stage: &mut crate::distribution::install_state::ReleaseExtractionStage<'_>,
    manifest: &ReleaseManifestV1,
    manifest_bytes: &[u8],
) -> Result<(), PreparedReleaseError> {
    stage.resume_manifest(&mut Cursor::new(manifest_bytes))?;
    for (file, (_, bytes, _)) in manifest.files().iter().zip(PAYLOADS) {
        stage.resume_payload(file, &mut Cursor::new(bytes))?;
    }
    Ok(())
}

#[test]
fn authenticated_extraction_resumes_an_exact_prefix_and_seals_post_io_replay() {
    let (manifest, manifest_bytes) = manifest();
    let archive_bytes = b"authenticated archive identity";
    let fixture = stable_release_repository_for_artifacts(&manifest_bytes, archive_bytes);
    let (temp, authorization) = make_authorization();
    let root = temp
        .path()
        .canonicalize()
        .expect("canonical parent")
        .join("state");
    let anchor = leaked_anchor(&fixture.repository.anchor);
    commit_fixture(&authorization, &anchor, &fixture.repository);

    let first = finalized_artifact_authorization(&authorization, &anchor, &fixture.pointer);
    let failure = first.with_locked_extraction_for_test::<PreparedReleaseError>(
        &manifest_bytes,
        &manifest,
        [
            instant("2026-08-18T09:01:06Z"),
            instant("2026-08-18T09:01:07Z"),
        ],
        [
            instant("2026-08-18T09:01:08Z"),
            instant("2026-08-18T09:01:09Z"),
        ],
        |stage| {
            stage.resume_manifest(&mut FailingPrefix {
                bytes: Cursor::new(manifest_bytes.clone()),
                remaining: 31,
            })?;
            Ok(())
        },
    );
    assert!(matches!(failure, Err(PreparedReleaseError::Extraction(_))));

    let second = finalized_artifact_authorization(&authorization, &anchor, &fixture.pointer);
    let extracted = second
        .with_locked_extraction_for_test::<PreparedReleaseError>(
            &manifest_bytes,
            &manifest,
            [
                instant("2026-08-18T09:01:10Z"),
                instant("2026-08-18T09:01:11Z"),
            ],
            [
                instant("2026-08-18T09:01:12Z"),
                instant("2026-08-18T09:01:13Z"),
            ],
            |stage| resume(stage, &manifest, &manifest_bytes),
        )
        .expect("exact prefix resumes");
    assert_eq!(
        format!("{extracted:?}"),
        "PostLocalIoReleaseAuthorization { .. }"
    );

    let stage = root.join("update/extractions").join(format!(
        ".extract-v0.2.0-{}",
        hex::encode(Sha256::digest(archive_bytes))
    ));
    assert_eq!(
        std::fs::metadata(&stage).unwrap().permissions().mode() & 0o7777,
        0o700
    );
    assert_eq!(
        std::fs::read(stage.join("release-manifest.json")).unwrap(),
        manifest_bytes
    );
    for (path, bytes, _) in PAYLOADS {
        let staged = stage.join(path);
        assert_eq!(std::fs::read(&staged).unwrap(), bytes);
        assert_eq!(
            std::fs::metadata(staged).unwrap().permissions().mode() & 0o7777,
            0o600
        );
    }
    drop(extracted);

    let rollback = finalized_artifact_authorization(&authorization, &anchor, &fixture.pointer)
        .with_locked_extraction_for_test::<PreparedReleaseError>(
        &manifest_bytes,
        &manifest,
        [
            instant("2026-08-18T09:02:00Z"),
            instant("2026-08-18T09:02:01Z"),
        ],
        [
            instant("2026-08-18T09:01:59Z"),
            instant("2026-08-18T09:02:00Z"),
        ],
        |stage| resume(stage, &manifest, &manifest_bytes),
    );
    assert!(matches!(
        rollback,
        Err(PreparedReleaseError::Authentication(
            ArtifactFetchAuthorizationError::Authentication(TufVerifierError::ClockRollback)
        ))
    ));
}

#[test]
fn stored_and_deflated_bundles_cross_the_complete_extraction_boundary() {
    for compression in [CompressionMethod::Stored, CompressionMethod::Deflated] {
        let (manifest, manifest_bytes) = manifest();
        let archive_bytes = archive(&manifest_bytes, compression);
        let fixture = stable_release_repository_for_artifacts(&manifest_bytes, &archive_bytes);
        let (temp, authorization) = make_authorization();
        let root = temp
            .path()
            .canonicalize()
            .expect("canonical parent")
            .join("state");
        let anchor = leaked_anchor(&fixture.repository.anchor);
        commit_fixture(&authorization, &anchor, &fixture.repository);

        let release = bundle(
            &authorization,
            &anchor,
            &fixture.pointer,
            manifest,
            manifest_bytes.clone(),
            &archive_bytes,
        );
        let extracted = extract_release(bind_archive(release).expect("archive profile binds"))
            .expect("profile-bound extraction succeeds");
        assert_eq!(format!("{extracted:?}"), "ExtractedRelease { .. }");
        let stage = root.join("update/extractions").join(format!(
            ".extract-v0.2.0-{}",
            hex::encode(Sha256::digest(&archive_bytes))
        ));
        assert_eq!(
            std::fs::read(stage.join("release-manifest.json")).unwrap(),
            manifest_bytes
        );
        for (path, bytes, _) in PAYLOADS {
            assert_eq!(std::fs::read(stage.join(path)).unwrap(), bytes);
        }
        assert!(!root.join("versions").exists());
        assert!(!root.join("activations").exists());
        assert!(!root.join("current").exists());
        assert!(matches!(
            crate::distribution::install_state::metadata::lock_metadata_state(&authorization),
            Err(MetadataJournalError::InstallState(
                crate::distribution::install_state::InstallStateError::Busy
            ))
        ));
        drop(extracted);
        drop(
            crate::distribution::install_state::metadata::lock_metadata_state(&authorization)
                .expect("dropping extracted proof releases the shared lock"),
        );
    }
}

#[test]
fn extraction_that_reaches_expiry_returns_no_post_io_authority() {
    let (manifest, manifest_bytes) = manifest();
    let archive_bytes = archive(&manifest_bytes, CompressionMethod::Stored);
    let fixture = stable_release_repository_for_artifacts_with_expiry(
        &manifest_bytes,
        &archive_bytes,
        "2026-08-18T09:01:12Z",
    );
    let (_temp, authorization) = make_authorization();
    let anchor = leaked_anchor(&fixture.repository.anchor);
    commit_fixture(&authorization, &anchor, &fixture.repository);
    let error = finalized_artifact_authorization(&authorization, &anchor, &fixture.pointer)
        .with_locked_extraction_for_test::<PreparedReleaseError>(
            &manifest_bytes,
            &manifest,
            [
                instant("2026-08-18T09:01:06Z"),
                instant("2026-08-18T09:01:07Z"),
            ],
            [
                instant("2026-08-18T09:01:12Z"),
                instant("2026-08-18T09:01:13Z"),
            ],
            |stage| resume(stage, &manifest, &manifest_bytes),
        )
        .expect_err("expiry consumes the authority");
    assert!(matches!(
        error,
        PreparedReleaseError::Authentication(ArtifactFetchAuthorizationError::Authentication(
            TufVerifierError::ExpiredMetadata
        ))
    ));
}

#[test]
fn metadata_generation_drift_after_download_blocks_extraction_before_mutation() {
    let (manifest, manifest_bytes) = manifest();
    let archive_bytes = archive(&manifest_bytes, CompressionMethod::Stored);
    let initial = stable_release_repository_for_artifacts(&manifest_bytes, &archive_bytes);
    let successor = stable_release_successor_for_artifacts(&manifest_bytes, &archive_bytes);
    let (_temp, authorization) = make_authorization();
    let anchor = leaked_anchor(&initial.repository.anchor);
    commit_fixture(&authorization, &anchor, &initial.repository);
    let final_authorization =
        finalized_artifact_authorization(&authorization, &anchor, &initial.pointer);
    let candidate = successor_candidate(&authorization, &anchor, &successor.repository);
    let completed = candidate.verification_completed_at();
    commit_and_reopen_for_test(&authorization, &anchor, candidate, [completed, completed])
        .expect("successor commits");

    let error = final_authorization
        .with_locked_extraction_for_test::<PreparedReleaseError>(
            &manifest_bytes,
            &manifest,
            [
                instant("2026-08-18T09:02:04Z"),
                instant("2026-08-18T09:02:05Z"),
            ],
            [
                instant("2026-08-18T09:02:06Z"),
                instant("2026-08-18T09:02:07Z"),
            ],
            |_stage| panic!("drift must fail before staging"),
        )
        .expect_err("generation-bound download is stale");
    assert!(matches!(
        error,
        PreparedReleaseError::Authentication(ArtifactFetchAuthorizationError::Authentication(
            TufVerifierError::DurableCommitMismatch | TufVerifierError::TargetBinding
        ))
    ));
}

#[test]
fn self_consistent_but_unauthenticated_manifest_cannot_open_the_stage() {
    let (manifest, manifest_bytes) = manifest();
    let archive_bytes = archive(&manifest_bytes, CompressionMethod::Stored);
    let fixture = stable_release_repository_for_artifacts(&manifest_bytes, &archive_bytes);
    let (_temp, authorization) = make_authorization();
    let anchor = leaked_anchor(&fixture.repository.anchor);
    commit_fixture(&authorization, &anchor, &fixture.repository);

    let mut value: serde_json::Value =
        serde_json::from_slice(&manifest_bytes).expect("manifest JSON");
    value["source_commit"] = serde_json::json!("b".repeat(40));
    let mutated =
        ReleaseManifestV1::parse_and_validate(&serde_json::to_vec(&value).expect("mutated JSON"))
            .expect("self-consistent different manifest");
    let mutated_bytes = mutated
        .to_deterministic_json()
        .expect("deterministic mutation");
    assert_ne!(mutated_bytes, manifest_bytes);

    let error = finalized_artifact_authorization(&authorization, &anchor, &fixture.pointer)
        .with_locked_extraction_for_test::<PreparedReleaseError>(
            &mutated_bytes,
            &mutated,
            [
                instant("2026-08-18T09:02:04Z"),
                instant("2026-08-18T09:02:05Z"),
            ],
            [
                instant("2026-08-18T09:02:06Z"),
                instant("2026-08-18T09:02:07Z"),
            ],
            |_stage| panic!("untrusted manifest must fail before staging"),
        )
        .expect_err("TUF manifest descriptor is the authority");
    assert!(matches!(error, PreparedReleaseError::Extraction(_)));
    assert_eq!(manifest.version().as_str(), mutated.version().as_str());
}

#[test]
fn post_extraction_archive_revalidation_failure_returns_no_capability_and_is_retryable() {
    let (manifest, manifest_bytes) = manifest();
    let archive_bytes = archive(&manifest_bytes, CompressionMethod::Deflated);
    let fixture = stable_release_repository_for_artifacts(&manifest_bytes, &archive_bytes);
    let (_temp, authorization) = make_authorization();
    let anchor = leaked_anchor(&fixture.repository.anchor);
    commit_fixture(&authorization, &anchor, &fixture.repository);

    let release = bundle(
        &authorization,
        &anchor,
        &fixture.pointer,
        manifest.clone(),
        manifest_bytes.clone(),
        &archive_bytes,
    );
    let mut bound = bind_archive(release).expect("archive profile binds");
    bound.fail_archive_revalidation_after_for_test(2);
    assert!(matches!(
        extract_release(bound),
        Err(PreparedReleaseError::ArchiveIntegrity(
            crate::distribution::install_state::ArtifactStageError::Integrity
        ))
    ));

    let retry = bundle(
        &authorization,
        &anchor,
        &fixture.pointer,
        manifest,
        manifest_bytes,
        &archive_bytes,
    );
    drop(extract_release(bind_archive(retry).unwrap()).expect("exact retry"));
}
