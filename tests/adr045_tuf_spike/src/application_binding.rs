//! Disposable application-level target binding used only by the spike.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, SeekFrom};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tough::schema::{Signed, Target, Targets};
use tough::TargetName;

use crate::candidates::VerifiedMetadataEvidence;
use crate::model::{sha256, SpikeError};
use crate::production_release_manifest::ReleaseManifestV1;
use crate::strict_json;
use crate::test_repository::{RELEASE_TARGET, RELEASE_VERSION};

const MAX_POINTER_BYTES: usize = 16 * 1024;
const MAX_MANIFEST_BYTES: usize = 1024 * 1024;
const MAX_BUNDLE_FILES: usize = 4096;
const MAX_BUNDLE_PAYLOAD_BYTES: u64 = 4 * 1024 * 1024 * 1024;

pub(crate) fn pointer_name() -> &'static str {
    "channels/stable/aarch64-apple-darwin.json"
}

pub(crate) fn manifest_name() -> &'static str {
    "releases/v0.2.0/aarch64-apple-darwin/release-manifest.json"
}

pub(crate) fn archive_name() -> &'static str {
    "releases/v0.2.0/aarch64-apple-darwin/hf2q-v0.2.0-aarch64-apple-darwin.zip"
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExperimentalPointerV0 {
    schema_version: u64,
    channel: String,
    version: String,
    target: String,
    manifest: ExperimentalDescriptorV0,
    archive: ExperimentalDescriptorV0,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExperimentalDescriptorV0 {
    name: String,
    length: u64,
    sha256: String,
}

/// Exact v1 wire shape. The production parser owns semantic validation; this
/// spike repeats the complete shape so target cross-binding is tested against
/// a representative manifest rather than a reduced identity surrogate.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExperimentalReleaseManifestV1 {
    kind: String,
    schema_version: u64,
    package: String,
    version: String,
    target: String,
    minimum_macos: String,
    source_commit: String,
    channel: String,
    code_signing: ExperimentalCodeSigningV1,
    compatibility: ExperimentalCompatibilityV1,
    files: Vec<ExperimentalBundleFileV1>,
    non_system_dynamic_dependencies: Vec<ExperimentalDynamicDependencyV1>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExperimentalCodeSigningV1 {
    team_id: String,
    identifier: String,
    certificate_common_name: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExperimentalCompatibilityV1 {
    minimum_installer_protocol: u64,
    minimum_updater_protocol: u64,
    launcher_registry_schema: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExperimentalBundleFileV1 {
    path: String,
    #[serde(rename = "type")]
    file_type: String,
    size: u64,
    mode: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExperimentalDynamicDependencyV1 {
    consumer: String,
    install_name: String,
}

/// Opaque evidence that authenticated top-level targets, the channel pointer,
/// external manifest, embedded manifest, and streamed archive all agree.
#[derive(Debug)]
pub(crate) struct VerifiedReleaseBinding {
    version: String,
    target: String,
    channel: String,
    archive_length: u64,
    archive_sha256: [u8; 32],
}

impl VerifiedReleaseBinding {
    pub(crate) fn identity(&self) -> (&str, &str, &str) {
        (&self.version, &self.target, &self.channel)
    }

    pub(crate) fn archive_descriptor(&self) -> (u64, [u8; 32]) {
        (self.archive_length, self.archive_sha256)
    }
}

pub(crate) fn pointer_bytes(manifest: &[u8], archive: &[u8]) -> Vec<u8> {
    serde_json::to_vec(&ExperimentalPointerV0 {
        schema_version: 0,
        channel: "stable".to_string(),
        version: RELEASE_VERSION.to_string(),
        target: RELEASE_TARGET.to_string(),
        manifest: ExperimentalDescriptorV0 {
            name: manifest_name().to_string(),
            length: manifest.len() as u64,
            sha256: hex::encode(sha256(manifest)),
        },
        archive: ExperimentalDescriptorV0 {
            name: archive_name().to_string(),
            length: archive.len() as u64,
            sha256: hex::encode(sha256(archive)),
        },
    })
    .expect("fixture pointer serializes")
}

pub(crate) fn verify_application_binding<R: Read + Seek>(
    metadata: &VerifiedMetadataEvidence,
    pointer_bytes: &[u8],
    manifest_bytes: &[u8],
    archive: &mut R,
) -> Result<VerifiedReleaseBinding, SpikeError> {
    strict_json::validate(pointer_bytes, MAX_POINTER_BYTES)?;
    strict_json::validate(manifest_bytes, MAX_MANIFEST_BYTES)?;
    let production_manifest = ReleaseManifestV1::parse_and_validate(manifest_bytes)
        .map_err(|_| SpikeError::ApplicationBinding)?;
    let pointer: ExperimentalPointerV0 =
        serde_json::from_slice(pointer_bytes).map_err(|_| SpikeError::ApplicationBinding)?;
    let manifest: ExperimentalReleaseManifestV1 =
        serde_json::from_slice(manifest_bytes).map_err(|_| SpikeError::ApplicationBinding)?;
    validate_manifest_identity(&pointer, &manifest)?;
    if metadata.channel() != pointer.channel
        || production_manifest.version().as_str() != pointer.version
        || production_manifest.target().as_str() != pointer.target
        || !matches!(
            production_manifest.channel(),
            crate::common::UpdateChannel::Stable
        )
    {
        return Err(SpikeError::ApplicationBinding);
    }
    descriptor_matches_bytes(&pointer.manifest, manifest_bytes)?;

    let targets = authenticated_targets(metadata)?;
    let pointer_target = required_target(&targets, pointer_name())?;
    let manifest_target = required_target(&targets, manifest_name())?;
    let archive_target = required_target(&targets, archive_name())?;
    tuf_descriptor_matches(pointer_target, pointer_bytes)?;
    tuf_descriptor_matches(manifest_target, manifest_bytes)?;
    descriptor_matches_target(&pointer.archive, archive_target)?;
    let archive_sha256 = stream_descriptor_matches(archive, &pointer.archive)?;
    verify_archive_inventory(archive, manifest_bytes, &manifest)?;

    Ok(VerifiedReleaseBinding {
        version: pointer.version,
        target: pointer.target,
        channel: pointer.channel,
        archive_length: pointer.archive.length,
        archive_sha256,
    })
}

fn validate_manifest_identity(
    pointer: &ExperimentalPointerV0,
    manifest: &ExperimentalReleaseManifestV1,
) -> Result<(), SpikeError> {
    let code_signing_is_shaped = manifest.code_signing.team_id.len() == 10
        && !manifest.code_signing.identifier.is_empty()
        && !manifest.code_signing.certificate_common_name.is_empty();
    let compatibility_is_v1 = manifest.compatibility.minimum_installer_protocol == 1
        && manifest.compatibility.minimum_updater_protocol == 1
        && manifest.compatibility.launcher_registry_schema == 1;
    let dependencies_are_shaped = manifest
        .non_system_dynamic_dependencies
        .iter()
        .all(|item| !item.consumer.is_empty() && !item.install_name.is_empty());
    if pointer.schema_version != 0
        || manifest.kind != "hf2q.release-manifest"
        || manifest.schema_version != 1
        || manifest.package != "hf2q"
        || pointer.channel != "stable"
        || pointer.version != RELEASE_VERSION
        || pointer.target != RELEASE_TARGET
        || manifest.version != pointer.version
        || manifest.target != pointer.target
        || manifest.channel != pointer.channel
        || manifest.minimum_macos.is_empty()
        || manifest.source_commit.len() != 40
        || !code_signing_is_shaped
        || !compatibility_is_v1
        || !dependencies_are_shaped
        || pointer.manifest.name != manifest_name()
        || pointer.archive.name != archive_name()
        || pointer.manifest.name == pointer.archive.name
        || !valid_digest(&pointer.manifest.sha256)
        || !valid_digest(&pointer.archive.sha256)
    {
        return Err(SpikeError::ApplicationBinding);
    }
    Ok(())
}

fn authenticated_targets(
    metadata: &VerifiedMetadataEvidence,
) -> Result<Signed<Targets>, SpikeError> {
    strict_json::validate(metadata.targets_raw(), crate::model::MAX_TARGETS_BYTES)?;
    let targets: Signed<Targets> = serde_json::from_slice(metadata.targets_raw())
        .map_err(|_| SpikeError::ApplicationBinding)?;
    if targets.signed.delegations.is_some() {
        return Err(SpikeError::DelegationsForbidden);
    }
    Ok(targets)
}

fn required_target<'a>(targets: &'a Signed<Targets>, name: &str) -> Result<&'a Target, SpikeError> {
    let name = TargetName::new(name).map_err(|_| SpikeError::ApplicationBinding)?;
    targets
        .signed
        .targets
        .get(&name)
        .ok_or(SpikeError::ApplicationBinding)
}

fn descriptor_matches_bytes(
    descriptor: &ExperimentalDescriptorV0,
    bytes: &[u8],
) -> Result<(), SpikeError> {
    let expected = decode_digest(&descriptor.sha256)?;
    if descriptor.length != bytes.len() as u64 || expected != sha256(bytes) {
        return Err(SpikeError::ApplicationBinding);
    }
    Ok(())
}

fn descriptor_matches_target(
    descriptor: &ExperimentalDescriptorV0,
    target: &Target,
) -> Result<(), SpikeError> {
    if descriptor.length != target.length
        || decode_digest(&descriptor.sha256)?.as_slice() != target.hashes.sha256.as_ref()
    {
        return Err(SpikeError::ApplicationBinding);
    }
    Ok(())
}

fn stream_descriptor_matches<R: Read + Seek>(
    archive: &mut R,
    descriptor: &ExperimentalDescriptorV0,
) -> Result<[u8; 32], SpikeError> {
    archive
        .seek(SeekFrom::Start(0))
        .map_err(|_| SpikeError::ApplicationBinding)?;
    let expected = decode_digest(&descriptor.sha256)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut seen = 0_u64;
    loop {
        let count = archive
            .read(&mut buffer)
            .map_err(|_| SpikeError::ApplicationBinding)?;
        if count == 0 {
            break;
        }
        seen = seen
            .checked_add(count as u64)
            .ok_or(SpikeError::ApplicationBinding)?;
        if seen > descriptor.length {
            return Err(SpikeError::ApplicationBinding);
        }
        hasher.update(&buffer[..count]);
    }
    let actual: [u8; 32] = hasher.finalize().into();
    if seen != descriptor.length || actual != expected {
        return Err(SpikeError::ApplicationBinding);
    }
    archive
        .seek(SeekFrom::Start(0))
        .map_err(|_| SpikeError::ApplicationBinding)?;
    Ok(actual)
}

fn verify_archive_inventory<R: Read + Seek>(
    archive: &mut R,
    expected_manifest: &[u8],
    manifest: &ExperimentalReleaseManifestV1,
) -> Result<(), SpikeError> {
    if manifest.files.is_empty() || manifest.files.len() > MAX_BUNDLE_FILES {
        return Err(SpikeError::ApplicationBinding);
    }
    let mut expected = BTreeMap::new();
    let mut total = 0_u64;
    let mut previous: Option<&str> = None;
    for file in &manifest.files {
        if !valid_bundle_path(&file.path)
            || file.path == "release-manifest.json"
            || file.file_type != "regular"
            || !matches!(file.mode.as_str(), "0644" | "0755")
            || !valid_digest(&file.sha256)
            || previous.is_some_and(|prior| prior >= file.path.as_str())
            || expected.insert(file.path.as_str(), file).is_some()
        {
            return Err(SpikeError::ApplicationBinding);
        }
        total = total
            .checked_add(file.size)
            .ok_or(SpikeError::ApplicationBinding)?;
        previous = Some(&file.path);
    }
    if total > MAX_BUNDLE_PAYLOAD_BYTES {
        return Err(SpikeError::ApplicationBinding);
    }

    archive
        .seek(SeekFrom::Start(0))
        .map_err(|_| SpikeError::ApplicationBinding)?;
    let mut zip = zip::ZipArchive::new(archive).map_err(|_| SpikeError::ApplicationBinding)?;
    if zip.len() != expected.len() + 1 {
        return Err(SpikeError::ApplicationBinding);
    }
    let mut seen = BTreeSet::new();
    let mut saw_manifest = false;
    for index in 0..zip.len() {
        let mut entry = zip
            .by_index(index)
            .map_err(|_| SpikeError::ApplicationBinding)?;
        let name = entry.name().to_string();
        if !seen.insert(name.clone()) || !entry.is_file() {
            return Err(SpikeError::ApplicationBinding);
        }
        if name == "release-manifest.json" {
            if saw_manifest
                || entry.size() != expected_manifest.len() as u64
                || !has_exact_regular_mode(entry.unix_mode(), 0o644)
                || !stream_entry_matches(
                    &mut entry,
                    expected_manifest.len() as u64,
                    sha256(expected_manifest),
                )?
            {
                return Err(SpikeError::ApplicationBinding);
            }
            saw_manifest = true;
            continue;
        }
        let file = expected
            .get(name.as_str())
            .ok_or(SpikeError::ApplicationBinding)?;
        let expected_mode =
            u32::from_str_radix(&file.mode, 8).map_err(|_| SpikeError::ApplicationBinding)?;
        if entry.size() != file.size
            || !has_exact_regular_mode(entry.unix_mode(), expected_mode)
            || !stream_entry_matches(&mut entry, file.size, decode_digest(&file.sha256)?)?
        {
            return Err(SpikeError::ApplicationBinding);
        }
    }
    if !saw_manifest || seen.len() != expected.len() + 1 {
        return Err(SpikeError::ApplicationBinding);
    }
    Ok(())
}

fn has_exact_regular_mode(mode: Option<u32>, expected_permissions: u32) -> bool {
    mode.is_some_and(|mode| mode & 0o170000 == 0o100000 && mode & 0o7777 == expected_permissions)
}

fn stream_entry_matches<R: Read>(
    entry: &mut R,
    expected_length: u64,
    expected_sha256: [u8; 32],
) -> Result<bool, SpikeError> {
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut seen = 0_u64;
    loop {
        let count = entry
            .read(&mut buffer)
            .map_err(|_| SpikeError::ApplicationBinding)?;
        if count == 0 {
            break;
        }
        seen = seen
            .checked_add(count as u64)
            .ok_or(SpikeError::ApplicationBinding)?;
        if seen > expected_length {
            return Ok(false);
        }
        hasher.update(&buffer[..count]);
    }
    Ok(seen == expected_length && <[u8; 32]>::from(hasher.finalize()) == expected_sha256)
}

fn tuf_descriptor_matches(target: &Target, bytes: &[u8]) -> Result<(), SpikeError> {
    if target.length != bytes.len() as u64 || target.hashes.sha256.as_ref() != sha256(bytes) {
        return Err(SpikeError::ApplicationBinding);
    }
    Ok(())
}

fn valid_bundle_path(path: &str) -> bool {
    !path.is_empty()
        && path.len() <= 512
        && !path.starts_with('/')
        && !path.contains('\\')
        && path
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != ".." && part.len() <= 255)
        && path.bytes().all(|byte| byte.is_ascii_graphic())
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value == value.to_ascii_lowercase()
        && hex::decode(value).is_ok_and(|bytes| bytes.len() == 32)
}

fn decode_digest(value: &str) -> Result<[u8; 32], SpikeError> {
    let bytes = hex::decode(value).map_err(|_| SpikeError::ApplicationBinding)?;
    bytes.try_into().map_err(|_| SpikeError::ApplicationBinding)
}
