use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::*;

const VERSION: &str = "0.2.0";
const ARCHIVE_DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

pub(super) struct Fixture {
    pub(super) _temp: tempfile::TempDir,
    pub(super) root: PathBuf,
    pub(super) version: PathBuf,
    pub(super) receipt_bytes: Vec<u8>,
}

impl Fixture {
    pub(super) fn new() -> Self {
        let temp = tempfile::tempdir().expect("temporary directory");
        // macOS exposes /var as a symlink. The production traversal correctly
        // refuses it, so tests use the descriptor's canonical private path.
        let temp_path = temp.path().canonicalize().expect("canonical temp path");
        let root = temp_path.join("state");
        bootstrap_installation_identity_for_test(
            ExplicitRootAuthorization::new(&root).expect("root authorization"),
            "550e8400-e29b-41d4-a716-446655440000",
            IdentityFaultPlan::default(),
        )
        .expect("bootstrap fixture installation identity");
        let version = root.join("versions").join(VERSION);
        for directory in [root.join("versions"), version.clone()] {
            fs::create_dir_all(&directory).expect("create private fixture directory");
            chmod(&directory, 0o700);
        }

        let payloads = [
            ("bin/hf2q", b"hf2q-binary\n".as_slice(), 0o755),
            (
                "libexec/serve_qwen38_opencode.sh",
                b"#!/bin/sh\nexec hf2q serve Qwen/Qwen3.8-27B\n".as_slice(),
                0o755,
            ),
            (
                "share/doc/hf2q/README.md",
                b"hf2q packaged documentation\n".as_slice(),
                0o644,
            ),
            (
                "share/licenses/hf2q/LICENSE-APACHE",
                b"Apache-2.0\n".as_slice(),
                0o644,
            ),
        ];
        for (relative, bytes, mode) in payloads {
            let path = version.join(relative);
            fs::create_dir_all(path.parent().expect("payload parent"))
                .expect("create payload parents");
            fs::write(&path, bytes).expect("write payload");
            chmod(&path, mode);
        }
        for directory in [
            version.join("bin"),
            version.join("libexec"),
            version.join("share"),
            version.join("share/doc"),
            version.join("share/doc/hf2q"),
            version.join("share/licenses"),
            version.join("share/licenses/hf2q"),
        ] {
            chmod(&directory, 0o755);
        }

        let files: Vec<Value> = payloads
            .iter()
            .map(|(path, bytes, mode)| {
                json!({
                    "path": path,
                    "type": "regular",
                    "size": bytes.len(),
                    "mode": if *mode == 0o755 { "0755" } else { "0644" },
                    "sha256": digest(bytes),
                })
            })
            .collect();
        let manifest_bytes = deterministic_manifest(json!({
            "kind": "hf2q.release-manifest",
            "schema_version": 1,
            "package": "hf2q",
            "version": VERSION,
            "target": "aarch64-apple-darwin",
            "minimum_macos": "14.0",
            "source_commit": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
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
        }));
        fs::write(version.join("release-manifest.json"), &manifest_bytes).expect("write manifest");
        chmod(&version.join("release-manifest.json"), 0o644);
        let manifest_digest = digest(&manifest_bytes);

        let root_text = root.to_str().expect("UTF-8 fixture root");
        let marker_bytes = deterministic_marker(json!({
            "kind": "hf2q.installed-version",
            "schema_version": 2,
            "package": "hf2q",
            "installation_layout_schema": 1,
            "installation_id": "550e8400-e29b-41d4-a716-446655440000",
            "installation_root": root_text,
            "release": {
                "version": VERSION,
                "target": "aarch64-apple-darwin",
                "release_manifest_sha256": manifest_digest,
                "archive_sha256": ARCHIVE_DIGEST
            },
            "prepared_from": {
                "kind": "verified-update-metadata",
                "root_version": 1,
                "timestamp_version": 1,
                "snapshot_version": 1,
                "targets_version": 1
            },
            "installation_sequence": 1,
            "installed_at_unix_seconds": 1787011200_u64
        }));
        fs::write(version.join("version-installation.json"), &marker_bytes).expect("write marker");
        chmod(&version.join("version-installation.json"), 0o600);
        let marker_digest = digest(&marker_bytes);

        let release = json!({
            "version": VERSION,
            "target": "aarch64-apple-darwin",
            "bundle": {
                "release_manifest_sha256": manifest_digest,
                "archive_sha256": ARCHIVE_DIGEST,
                "installed_version_marker_sha256": marker_digest,
                "installation_sequence": 1
            }
        });
        let receipt_bytes = deterministic_receipt(json!({
            "kind": "hf2q.install-receipt",
            "schema_version": 1,
            "package": "hf2q",
            "state_layout_schema": 1,
            "installation_layout_schema": 1,
            "installation_id": "550e8400-e29b-41d4-a716-446655440000",
            "state_root": root_text,
            "installation_root": root_text,
            "owner_family": "standalone",
            "update_route": "standalone",
            "active": release.clone(),
            "retained": [],
            "last_successful_transition": {
                "sequence": 1,
                "type": "install",
                "to": { "owner_family": "standalone", "release": release },
                "authority": {
                    "kind": "verified-update-metadata",
                    "root_version": 1,
                    "timestamp_version": 1,
                    "snapshot_version": 1,
                    "targets_version": 1
                },
                "completed_at_unix_seconds": 1787011200_u64
            }
        }));
        Self {
            _temp: temp,
            root,
            version,
            receipt_bytes,
        }
    }

    pub(super) fn prepare(&self) -> Result<FirstActivationPreparation, InstallStateError> {
        prepare_first_activation(
            open_existing_installation_identity(ExplicitRootAuthorization::new(&self.root)?)?
                .ok_or(InstallStateError::Missing("installation identity"))?,
            AuthenticatedPreparedVersion::for_test_only(self.receipt_bytes.clone()),
        )
    }
}

pub(super) fn chmod(path: &Path, mode: u32) {
    fs::set_permissions(path, fs::Permissions::from_mode(mode)).expect("set fixture mode");
}

pub(super) fn copy_directory(source: &Path, target: &Path) {
    fs::create_dir(target).expect("create copied directory");
    chmod(
        target,
        fs::metadata(source)
            .expect("source metadata")
            .permissions()
            .mode()
            & 0o7777,
    );
    for entry in fs::read_dir(source).expect("read copied directory") {
        let entry = entry.expect("directory entry");
        let source_path = entry.path();
        let target_path = target.join(entry.file_name());
        let metadata = entry.metadata().expect("entry metadata");
        if metadata.is_dir() {
            copy_directory(&source_path, &target_path);
        } else {
            fs::copy(&source_path, &target_path).expect("copy fixture file");
            chmod(&target_path, metadata.permissions().mode() & 0o7777);
        }
    }
}

fn deterministic_manifest(raw: Value) -> Vec<u8> {
    let bytes = serde_json::to_vec(&raw).expect("manifest JSON");
    schema::ReleaseManifestV1::parse_and_validate(&bytes)
        .expect("valid fixture manifest")
        .to_deterministic_json()
        .expect("deterministic manifest")
}

fn deterministic_marker(raw: Value) -> Vec<u8> {
    let bytes = serde_json::to_vec(&raw).expect("marker JSON");
    schema::InstalledVersionMarkerV2::parse_and_validate(&bytes)
        .expect("valid fixture marker")
        .to_deterministic_json()
        .expect("deterministic marker")
}

fn deterministic_receipt(raw: Value) -> Vec<u8> {
    let bytes = serde_json::to_vec(&raw).expect("receipt JSON");
    schema::InstallReceiptV1::parse_and_validate(&bytes)
        .expect("valid fixture receipt")
        .to_deterministic_json()
        .expect("deterministic receipt")
}

fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
