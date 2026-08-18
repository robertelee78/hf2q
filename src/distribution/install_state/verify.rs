use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;

use sha2::{Digest, Sha256};

use super::file;
use super::host;
use super::schema::{
    AbsoluteInstallPath, BundleEntryType, FirstStandaloneInstallRecord, InstallReceiptV1,
    ReleaseManifestV1, MAX_INSTALLED_VERSION_MARKER_BYTES, MAX_INSTALL_RECEIPT_BYTES,
    MAX_RELEASE_MANIFEST_BYTES,
};
use super::unix::{self, Directory};
use super::{validate_receipt_shape, InstallStateError, CURRENT_TARGET, FIRST_GENERATION};

const MANIFEST_NAME: &str = "release-manifest.json";
const MARKER_NAME: &str = "version-installation.json";
const RECEIPT_NAME: &str = "install-receipt.json";
const RECEIPT_PARTIAL_NAME: &str = ".install-receipt.json.partial";
const VERSION_LINK_NAME: &str = "version";

#[derive(Debug)]
pub(super) struct VerifiedPreparedVersion {
    pub(super) directory: Directory,
    pub(super) link_target: String,
}

#[derive(Debug)]
pub(super) struct VerifiedActivation {
    pub(super) directory: Directory,
    pub(super) receipt_file: File,
}

pub(super) fn validate_first_receipt(
    bytes: &[u8],
    authorized_root: &AbsoluteInstallPath,
) -> Result<InstallReceiptV1, InstallStateError> {
    let receipt = InstallReceiptV1::parse_and_validate(bytes)?;
    if receipt.to_deterministic_json()? != bytes {
        return Err(InstallStateError::InvalidLayout(
            "install receipt is not in canonical byte encoding",
        ));
    }
    if receipt.state_root() != authorized_root || receipt.installation_root() != authorized_root {
        return Err(InstallStateError::InvalidLayout(
            "receipt roots do not match the explicitly authorized root",
        ));
    }
    validate_receipt_shape(&receipt)?;
    Ok(receipt)
}

pub(super) fn verify_prepared_version(
    versions: &Directory,
    receipt: &InstallReceiptV1,
) -> Result<VerifiedPreparedVersion, InstallStateError> {
    let release = receipt.active();
    let bundle = release.bundle().ok_or(InstallStateError::InvalidLayout(
        "active release lacks bundle identity",
    ))?;
    let marker_digest =
        bundle
            .installed_version_marker_sha256()
            .ok_or(InstallStateError::InvalidLayout(
                "active release lacks installed-marker identity",
            ))?;
    if bundle.installation_sequence() != Some(1) {
        return Err(InstallStateError::InvalidLayout(
            "prepared version is not installation sequence one",
        ));
    }

    let name = release.version().as_str().to_owned();
    if unix::list_names(versions)? != BTreeSet::from([name.clone()]) {
        return Err(InstallStateError::InvalidLayout(
            "first installation versions inventory is not exact",
        ));
    }
    let directory = unix::open_directory_at(versions, &name, Some(0o700), true)?;
    let (manifest_file, manifest_bytes, manifest_identity) =
        file::read_regular_file(&directory, MANIFEST_NAME, 0o644, MAX_RELEASE_MANIFEST_BYTES)?;
    if manifest_identity.size != manifest_bytes.len() as u64
        || digest(&manifest_bytes) != bundle.release_manifest_sha256().as_str()
    {
        return Err(InstallStateError::InvalidLayout(
            "release manifest bytes do not match the receipt",
        ));
    }
    let manifest = ReleaseManifestV1::parse_and_validate(&manifest_bytes)?;
    if manifest.version() != release.version() || manifest.target() != release.target() {
        return Err(InstallStateError::InvalidLayout(
            "release manifest identity does not match the receipt",
        ));
    }
    host::require_compatible_host(manifest.minimum_macos())?;
    unix::full_sync_file(&manifest_file)?;

    let (marker_file, marker_bytes, marker_identity) = file::read_regular_file(
        &directory,
        MARKER_NAME,
        0o600,
        MAX_INSTALLED_VERSION_MARKER_BYTES,
    )?;
    if marker_identity.size != marker_bytes.len() as u64 {
        return Err(InstallStateError::InvalidLayout(
            "installed marker changed while verifying",
        ));
    }
    let record = FirstStandaloneInstallRecord::reconstruct_from_exact_marker(&marker_bytes)?;
    if record.marker_sha256() != marker_digest || record.receipt() != receipt {
        return Err(InstallStateError::InvalidLayout(
            "install receipt is not the exact record derived from the installed marker",
        ));
    }
    let marker = record.marker();
    if marker.installation_id() != receipt.installation_id()
        || marker.installation_root() != receipt.installation_root()
        || marker.release().version() != release.version()
        || marker.release().target() != release.target()
        || marker.release().release_manifest_sha256() != bundle.release_manifest_sha256()
        || marker.release().archive_sha256() != bundle.archive_sha256()
        || marker.installation_sequence() != 1
    {
        return Err(InstallStateError::InvalidLayout(
            "installed marker does not cross-bind the receipt and manifest",
        ));
    }
    unix::full_sync_file(&marker_file)?;

    verify_exact_version_inventory(&directory, &manifest)?;
    unix::sync_directory(&directory)?;
    unix::sync_directory(versions)?;
    Ok(VerifiedPreparedVersion {
        directory,
        link_target: format!("../../versions/{name}"),
    })
}

pub(super) fn resume_activation_prefix(
    activation: &Directory,
    receipt: &InstallReceiptV1,
    receipt_bytes: &[u8],
    version: &VerifiedPreparedVersion,
) -> Result<(), InstallStateError> {
    let names = unix::list_names(activation)?;
    if !names.is_subset(&BTreeSet::from([
        RECEIPT_NAME.to_owned(),
        RECEIPT_PARTIAL_NAME.to_owned(),
        VERSION_LINK_NAME.to_owned(),
    ])) {
        return Err(InstallStateError::InvalidLayout(
            "pending activation contains an unexpected entry",
        ));
    }
    if names.contains(RECEIPT_NAME) && names.contains(RECEIPT_PARTIAL_NAME) {
        return Err(InstallStateError::InvalidLayout(
            "complete and partial activation receipts coexist",
        ));
    }
    if names.contains(RECEIPT_NAME) {
        verify_receipt_file(activation, receipt, receipt_bytes)?;
    } else {
        file::write_or_resume_private_file(activation, RECEIPT_PARTIAL_NAME, receipt_bytes)?;
        unix::rename_noreplace(activation, RECEIPT_PARTIAL_NAME, activation, RECEIPT_NAME)?;
        unix::sync_directory(activation)?;
        verify_receipt_file(activation, receipt, receipt_bytes)?;
    }
    if names.contains(VERSION_LINK_NAME) {
        if unix::read_symlink(activation, VERSION_LINK_NAME)? != version.link_target {
            return Err(InstallStateError::InvalidLayout(
                "pending activation version link conflicts with the receipt",
            ));
        }
    } else {
        unix::create_symlink(activation, VERSION_LINK_NAME, &version.link_target)?;
        unix::sync_directory(activation)?;
    }
    Ok(())
}

pub(super) fn verify_activation(
    activations: &Directory,
    generation: &str,
    receipt: &InstallReceiptV1,
    receipt_bytes: &[u8],
    version: &VerifiedPreparedVersion,
) -> Result<VerifiedActivation, InstallStateError> {
    if generation != FIRST_GENERATION {
        return Err(InstallStateError::InvalidLayout(
            "first activation generation is not canonical",
        ));
    }
    let activation = unix::open_directory_at(activations, generation, Some(0o700), true)?;
    let receipt_file = verify_activation_directory(&activation, receipt, receipt_bytes, version)?;
    Ok(VerifiedActivation {
        directory: activation,
        receipt_file,
    })
}

pub(super) fn verify_activation_directory(
    activation: &Directory,
    receipt: &InstallReceiptV1,
    receipt_bytes: &[u8],
    version: &VerifiedPreparedVersion,
) -> Result<File, InstallStateError> {
    let expected = BTreeSet::from([RECEIPT_NAME.to_owned(), VERSION_LINK_NAME.to_owned()]);
    if unix::list_names(activation)? != expected {
        return Err(InstallStateError::InvalidLayout(
            "activation entry inventory is not exact",
        ));
    }
    let receipt_file = verify_receipt_file(activation, receipt, receipt_bytes)?;
    if unix::read_symlink(activation, VERSION_LINK_NAME)? != version.link_target {
        return Err(InstallStateError::InvalidLayout(
            "activation version link does not match its receipt",
        ));
    }
    Ok(receipt_file)
}

pub(super) fn verify_committed_first_activation(
    root: &Directory,
    activations: &Directory,
    receipt: &InstallReceiptV1,
    receipt_bytes: &[u8],
    version: &VerifiedPreparedVersion,
) -> Result<VerifiedActivation, InstallStateError> {
    if unix::read_symlink(root, "current")? != CURRENT_TARGET {
        return Err(InstallStateError::InvalidLayout(
            "current does not select the canonical first activation",
        ));
    }
    verify_activation(
        activations,
        FIRST_GENERATION,
        receipt,
        receipt_bytes,
        version,
    )
}

fn verify_receipt_file(
    activation: &Directory,
    expected_receipt: &InstallReceiptV1,
    expected_bytes: &[u8],
) -> Result<File, InstallStateError> {
    let (file, bytes, identity) =
        file::read_regular_file(activation, RECEIPT_NAME, 0o600, MAX_INSTALL_RECEIPT_BYTES)?;
    if identity.size != expected_bytes.len() as u64 || bytes != expected_bytes {
        return Err(InstallStateError::InvalidLayout(
            "activation receipt bytes do not match the authenticated input",
        ));
    }
    let parsed = InstallReceiptV1::parse_and_validate(&bytes)?;
    if parsed != *expected_receipt || parsed.to_deterministic_json()? != bytes {
        return Err(InstallStateError::InvalidLayout(
            "activation receipt is not the exact canonical receipt",
        ));
    }
    Ok(file)
}

fn verify_exact_version_inventory(
    version: &Directory,
    manifest: &ReleaseManifestV1,
) -> Result<(), InstallStateError> {
    let mut expected: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    expected
        .entry(String::new())
        .or_default()
        .extend([MANIFEST_NAME.to_owned(), MARKER_NAME.to_owned()]);
    for file in manifest.files() {
        if file.file_type() != BundleEntryType::Regular {
            return Err(InstallStateError::InvalidLayout(
                "release inventory contains a non-regular payload",
            ));
        }
        add_expected_path(&mut expected, file.path().as_str());
    }

    for (relative, names) in &expected {
        let directory = open_relative_directory(version, relative)?;
        if unix::list_names(&directory)? != *names {
            return Err(InstallStateError::InvalidLayout(
                "installed version entry inventory is not exact",
            ));
        }
    }

    for expected_file in manifest.files() {
        let (parent_path, name) = split_parent(expected_file.path().as_str());
        let parent = open_relative_directory(version, parent_path)?;
        let (payload_file, actual_digest) = file::hash_regular_file(
            &parent,
            name,
            expected_file.mode().as_octal(),
            expected_file.size(),
        )?;
        if actual_digest != expected_file.sha256().as_str() {
            return Err(InstallStateError::InvalidLayout(
                "installed payload does not match the release manifest",
            ));
        }
        unix::full_sync_file(&payload_file)?;
    }

    let mut directories: Vec<_> = expected.keys().map(String::as_str).collect();
    directories.sort_by_key(|path| {
        std::cmp::Reverse(if path.is_empty() {
            0
        } else {
            path.split('/').count()
        })
    });
    for relative in directories {
        unix::sync_directory(&open_relative_directory(version, relative)?)?;
    }
    Ok(())
}

fn add_expected_path(tree: &mut BTreeMap<String, BTreeSet<String>>, path: &str) {
    let parts: Vec<_> = path.split('/').collect();
    let mut parent = String::new();
    for (index, part) in parts.iter().enumerate() {
        tree.entry(parent.clone())
            .or_default()
            .insert((*part).to_owned());
        if index + 1 < parts.len() {
            parent = if parent.is_empty() {
                (*part).to_owned()
            } else {
                format!("{parent}/{part}")
            };
            tree.entry(parent.clone()).or_default();
        }
    }
}

fn open_relative_directory(
    version: &Directory,
    relative: &str,
) -> Result<Directory, InstallStateError> {
    if relative.is_empty() {
        return unix::duplicate_directory(version);
    }
    let mut directory = unix::duplicate_directory(version)?;
    for component in relative.split('/') {
        directory = unix::open_directory_at(&directory, component, Some(0o755), true)?;
    }
    Ok(directory)
}

fn split_parent(path: &str) -> (&str, &str) {
    path.rsplit_once('/').unwrap_or(("", path))
}

fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
