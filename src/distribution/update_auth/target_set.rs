use std::collections::BTreeMap;

use jiff::Timestamp;
use sha2::{Digest, Sha256};
use sigstore_tuf::metadata::TargetFile;

use super::model::EmbeddedTrustRoot;
use super::replay::replay_selected;
use super::verifier::ClockSource;
use super::{profile, TufVerifierError};
use crate::distribution::install_state::metadata::schema::MetadataGenerationReceiptV2;
use crate::distribution::install_state::metadata::{
    read_selected, MetadataStateAuthorization, StoredMetadataGeneration,
};
use crate::distribution::schema::{
    ChannelPointerV1, ConsistentSnapshotTargetName, LogicalTargetKind, LogicalTargetName,
    ReleaseVersion, Sha256Digest, TargetTriple, UpdateChannel, MAX_CHANNEL_POINTER_BYTES,
    MAX_RELEASE_ARCHIVE_BYTES, MAX_RELEASE_MANIFEST_BYTES,
};

#[derive(Debug, PartialEq, Eq)]
struct SelectedMetadataIdentity {
    installation_id: String,
    state_root: String,
    sequence: u64,
    generation_sha256: [u8; 32],
}

#[derive(Debug, PartialEq, Eq)]
struct MetadataVersions {
    root: u64,
    timestamp: u64,
    snapshot: u64,
    targets: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub(in crate::distribution) struct AuthenticatedTargetDescriptor {
    logical_name: LogicalTargetName,
    physical_name: ConsistentSnapshotTargetName,
    length: u64,
    sha256: Sha256Digest,
}

impl AuthenticatedTargetDescriptor {
    pub(in crate::distribution) fn logical_name(&self) -> &str {
        self.logical_name.as_str()
    }

    pub(in crate::distribution) fn physical_name(&self) -> &ConsistentSnapshotTargetName {
        &self.physical_name
    }

    pub(in crate::distribution) fn length(&self) -> u64 {
        self.length
    }

    pub(in crate::distribution) fn sha256(&self) -> &Sha256Digest {
        &self.sha256
    }

    pub(super) fn matches_bytes(&self, bytes: &[u8]) -> bool {
        self.length == bytes.len() as u64
            && self.sha256.as_str() == hex::encode(Sha256::digest(bytes))
    }

    #[cfg(test)]
    pub(crate) fn for_test(logical_name: LogicalTargetName, bytes: &[u8]) -> Self {
        let sha256 = Sha256Digest::parse("sha256", hex::encode(Sha256::digest(bytes)))
            .expect("test target digest");
        let physical_name = logical_name.consistent_snapshot_name(&sha256);
        Self {
            logical_name,
            physical_name,
            length: bytes.len() as u64,
            sha256,
        }
    }
}

#[derive(Debug)]
struct AuthenticatedReleasePair {
    version: ReleaseVersion,
    manifest: AuthenticatedTargetDescriptor,
    archive: AuthenticatedTargetDescriptor,
}

/// Fresh, generation-bound authentication of the stable target inventory.
///
/// This exposes no generic target lookup. Older release pairs are retained as
/// inert authenticated history only so the stable pointer can select one pair.
#[derive(Debug)]
pub(in crate::distribution) struct AuthenticatedTargetSet {
    selected: SelectedMetadataIdentity,
    versions: MetadataVersions,
    authenticated_at: Timestamp,
    earliest_expiry: Timestamp,
    pointer: AuthenticatedTargetDescriptor,
    releases: Vec<AuthenticatedReleasePair>,
}

/// Exact TUF and pointer agreement for one stable release.
///
/// This is not download, downgrade, install, extraction, or activation
/// authority. A later transition must reacquire the installation lock and
/// prove the same selected metadata identity before mutating state.
#[derive(Debug)]
pub(in crate::distribution) struct AuthenticatedReleaseTargets {
    selected: SelectedMetadataIdentity,
    versions: MetadataVersions,
    authenticated_at: Timestamp,
    earliest_expiry: Timestamp,
    version: ReleaseVersion,
    target: TargetTriple,
    pointer: AuthenticatedTargetDescriptor,
    manifest: AuthenticatedTargetDescriptor,
    archive: AuthenticatedTargetDescriptor,
    exact_pointer_bytes: Box<[u8]>,
}

impl AuthenticatedReleaseTargets {
    pub(in crate::distribution) fn selected_sequence(&self) -> u64 {
        self.selected.sequence
    }

    pub(in crate::distribution) fn selected_generation_sha256(&self) -> [u8; 32] {
        self.selected.generation_sha256
    }

    pub(in crate::distribution) fn version(&self) -> &ReleaseVersion {
        &self.version
    }

    pub(in crate::distribution) fn target(&self) -> TargetTriple {
        self.target
    }

    pub(in crate::distribution) fn pointer(&self) -> &AuthenticatedTargetDescriptor {
        &self.pointer
    }

    pub(in crate::distribution) fn manifest(&self) -> &AuthenticatedTargetDescriptor {
        &self.manifest
    }

    pub(in crate::distribution) fn archive(&self) -> &AuthenticatedTargetDescriptor {
        &self.archive
    }

    pub(in crate::distribution) fn authenticated_at(&self) -> Timestamp {
        self.authenticated_at
    }

    pub(in crate::distribution) fn earliest_expiry(&self) -> Timestamp {
        self.earliest_expiry
    }

    pub(in crate::distribution) fn metadata_versions(&self) -> [u64; 4] {
        [
            self.versions.root,
            self.versions.timestamp,
            self.versions.snapshot,
            self.versions.targets,
        ]
    }

    pub(in crate::distribution) fn installation_id(&self) -> &str {
        &self.selected.installation_id
    }

    pub(in crate::distribution) fn state_root(&self) -> &str {
        &self.selected.state_root
    }

    pub(in crate::distribution) fn exact_pointer_bytes(&self) -> &[u8] {
        &self.exact_pointer_bytes
    }

    pub(super) fn exactly_matches_bound_release(&self, other: &Self) -> bool {
        self.selected == other.selected
            && self.versions == other.versions
            && self.earliest_expiry == other.earliest_expiry
            && self.version == other.version
            && self.target == other.target
            && self.pointer == other.pointer
            && self.manifest == other.manifest
            && self.archive == other.archive
            && self.exact_pointer_bytes == other.exact_pointer_bytes
    }
}

impl AuthenticatedTargetSet {
    pub(in crate::distribution) fn pointer(&self) -> &AuthenticatedTargetDescriptor {
        &self.pointer
    }

    pub(super) fn authenticated_at(&self) -> Timestamp {
        self.authenticated_at
    }

    pub(in crate::distribution) fn bind_channel_pointer(
        self,
        exact_pointer_bytes: &[u8],
    ) -> Result<AuthenticatedReleaseTargets, TufVerifierError> {
        if !self.pointer.matches_bytes(exact_pointer_bytes) {
            return Err(TufVerifierError::TargetBinding);
        }
        let pointer = ChannelPointerV1::parse_and_validate(exact_pointer_bytes)?;
        if pointer.channel() != UpdateChannel::Stable
            || pointer.target() != TargetTriple::Aarch64AppleDarwin
        {
            return Err(TufVerifierError::TargetBinding);
        }
        let index = self
            .releases
            .binary_search_by(|pair| pair.version.cmp(pointer.version()))
            .map_err(|_| TufVerifierError::TargetBinding)?;
        let pair = self
            .releases
            .into_iter()
            .nth(index)
            .ok_or(TufVerifierError::TargetBinding)?;
        if !descriptor_matches(pointer.manifest(), &pair.manifest)
            || !descriptor_matches(pointer.archive(), &pair.archive)
        {
            return Err(TufVerifierError::TargetBinding);
        }
        Ok(AuthenticatedReleaseTargets {
            selected: self.selected,
            versions: self.versions,
            authenticated_at: self.authenticated_at,
            earliest_expiry: self.earliest_expiry,
            version: pair.version,
            target: pointer.target(),
            pointer: self.pointer,
            manifest: pair.manifest,
            archive: pair.archive,
            exact_pointer_bytes: exact_pointer_bytes.into(),
        })
    }
}

/// Deliberately dormant production entry point.
///
/// It remains unreachable until ADR-045 lands the real compiled stable root
/// and the key-custody/recovery runbook. Even then it yields only a fresh
/// descriptor plan, never transport or installation authority.
pub(super) fn authenticate_selected_targets(
    authorization: &MetadataStateAuthorization,
    anchor: &EmbeddedTrustRoot,
) -> Result<AuthenticatedTargetSet, TufVerifierError> {
    authenticate_selected_targets_with_clock(authorization, anchor, ClockSource::System)
}

fn authenticate_selected_targets_with_clock(
    authorization: &MetadataStateAuthorization,
    anchor: &EmbeddedTrustRoot,
    mut clock: ClockSource,
) -> Result<AuthenticatedTargetSet, TufVerifierError> {
    let stored = read_selected(authorization)?.ok_or(TufVerifierError::NoSelectedMetadata)?;
    authenticate_stored_targets(authorization, anchor, stored, &mut clock)
}

pub(super) fn authenticate_stored_targets(
    authorization: &MetadataStateAuthorization,
    anchor: &EmbeddedTrustRoot,
    stored: StoredMetadataGeneration,
    clock: &mut ClockSource,
) -> Result<AuthenticatedTargetSet, TufVerifierError> {
    let sequence = stored.sequence();
    let generation_sha256: [u8; 32] = Sha256::digest(stored.generation_receipt()).into();
    let receipt = MetadataGenerationReceiptV2::parse(stored.generation_receipt())?;
    let parsed_targets = profile::targets(stored.targets())?;
    let started_at = clock.sample()?;
    if started_at < receipt.verification_completed_at()? {
        return Err(TufVerifierError::ClockRollback);
    }
    let state = replay_selected(anchor, &receipt, stored, started_at)?;
    let authenticated_at = clock.sample()?;
    if authenticated_at < started_at {
        return Err(TufVerifierError::ClockRollback);
    }
    require_all_fresh(&state.trusted, authenticated_at)?;
    if state.trusted.targets() != Some(&parsed_targets.signed) {
        return Err(TufVerifierError::AuthenticationFailed);
    }
    if !state.trusted.root().consistent_snapshot {
        return Err(TufVerifierError::UnsupportedTargetProfile);
    }

    let versions = MetadataVersions {
        root: state.trusted.root().version,
        timestamp: state
            .trusted
            .timestamp()
            .ok_or(TufVerifierError::AuthenticationFailed)?
            .version,
        snapshot: state
            .trusted
            .snapshot()
            .ok_or(TufVerifierError::AuthenticationFailed)?
            .version,
        targets: parsed_targets.signed.version,
    };
    let earliest_expiry = earliest_expiry(&state.trusted)?;
    let (pointer, releases) = validate_inventory(&parsed_targets.signed.targets)?;
    Ok(AuthenticatedTargetSet {
        selected: SelectedMetadataIdentity {
            installation_id: authorization.installation_id().to_owned(),
            state_root: authorization.state_root().to_owned(),
            sequence,
            generation_sha256,
        },
        versions,
        authenticated_at,
        earliest_expiry,
        pointer,
        releases,
    })
}

fn validate_inventory(
    targets: &BTreeMap<String, TargetFile>,
) -> Result<(AuthenticatedTargetDescriptor, Vec<AuthenticatedReleasePair>), TufVerifierError> {
    let mut pointer = None;
    let mut releases: BTreeMap<ReleaseVersion, PartialReleasePair> = BTreeMap::new();
    for (raw_name, target) in targets {
        let name = LogicalTargetName::parse("targets.name", raw_name.clone())
            .map_err(|_| TufVerifierError::TargetInventory)?;
        let descriptor = authenticated_descriptor(name, target)?;
        match descriptor.logical_name.kind() {
            LogicalTargetKind::ChannelPointer => {
                if pointer.replace(descriptor).is_some() {
                    return Err(TufVerifierError::TargetInventory);
                }
            }
            LogicalTargetKind::ReleaseManifest => {
                let version = descriptor
                    .logical_name
                    .version()
                    .ok_or(TufVerifierError::TargetInventory)?
                    .clone();
                if releases
                    .entry(version)
                    .or_default()
                    .manifest
                    .replace(descriptor)
                    .is_some()
                {
                    return Err(TufVerifierError::TargetInventory);
                }
            }
            LogicalTargetKind::ReleaseArchive => {
                let version = descriptor
                    .logical_name
                    .version()
                    .ok_or(TufVerifierError::TargetInventory)?
                    .clone();
                if releases
                    .entry(version)
                    .or_default()
                    .archive
                    .replace(descriptor)
                    .is_some()
                {
                    return Err(TufVerifierError::TargetInventory);
                }
            }
        }
    }
    let pointer = pointer.ok_or(TufVerifierError::TargetInventory)?;
    if releases.is_empty() {
        return Err(TufVerifierError::TargetInventory);
    }
    let releases = releases
        .into_iter()
        .map(|(version, pair)| {
            Ok(AuthenticatedReleasePair {
                version,
                manifest: pair.manifest.ok_or(TufVerifierError::TargetInventory)?,
                archive: pair.archive.ok_or(TufVerifierError::TargetInventory)?,
            })
        })
        .collect::<Result<Vec<_>, TufVerifierError>>()?;
    Ok((pointer, releases))
}

/// Once a selected targets role has entered the stable application profile,
/// its versioned release pairs are an append-only semantic floor. The stable
/// pointer may move and new complete pairs may be appended, but a later
/// correctly signed targets role may not rewrite or remove an existing pair.
///
/// The lock-held coordinator calls this only after replay has authenticated
/// both the selected predecessor and the candidate. Pairwise enforcement is
/// transitive, so predecessor cleanup after selector commit loses no floor.
pub(super) fn require_retained_release_floor(
    prior_targets: &[u8],
    candidate_targets: &[u8],
) -> Result<(), TufVerifierError> {
    let prior = profile::targets(prior_targets)?;
    let Ok((_, prior_releases)) = validate_inventory(&prior.signed.targets) else {
        // Generic TUF fixtures and any pre-profile generation establish no
        // application release floor. The first valid stable inventory does.
        return Ok(());
    };
    let candidate = profile::targets(candidate_targets)?;
    let (_, candidate_releases) = validate_inventory(&candidate.signed.targets)
        .map_err(|_| TufVerifierError::RetainedReleaseMutation)?;

    for prior_pair in prior_releases {
        let index = candidate_releases
            .binary_search_by(|pair| pair.version.cmp(&prior_pair.version))
            .map_err(|_| TufVerifierError::RetainedReleaseMutation)?;
        let candidate_pair = candidate_releases
            .get(index)
            .ok_or(TufVerifierError::RetainedReleaseMutation)?;
        if !same_authenticated_descriptor(&prior_pair.manifest, &candidate_pair.manifest)
            || !same_authenticated_descriptor(&prior_pair.archive, &candidate_pair.archive)
        {
            return Err(TufVerifierError::RetainedReleaseMutation);
        }
    }
    Ok(())
}

#[derive(Default)]
struct PartialReleasePair {
    manifest: Option<AuthenticatedTargetDescriptor>,
    archive: Option<AuthenticatedTargetDescriptor>,
}

fn authenticated_descriptor(
    logical_name: LogicalTargetName,
    target: &TargetFile,
) -> Result<AuthenticatedTargetDescriptor, TufVerifierError> {
    if target.length == 0 || target.custom.is_some() || !target.extra.is_empty() {
        return Err(TufVerifierError::TargetInventory);
    }
    let maximum = match logical_name.kind() {
        LogicalTargetKind::ChannelPointer => MAX_CHANNEL_POINTER_BYTES as u64,
        LogicalTargetKind::ReleaseManifest => MAX_RELEASE_MANIFEST_BYTES as u64,
        LogicalTargetKind::ReleaseArchive => MAX_RELEASE_ARCHIVE_BYTES,
    };
    if target.length > maximum || target.hashes.len() != 1 {
        return Err(TufVerifierError::TargetInventory);
    }
    let sha256 = target
        .hashes
        .get("sha256")
        .ok_or(TufVerifierError::TargetInventory)?;
    let sha256 = Sha256Digest::parse("targets.sha256", sha256.clone())
        .map_err(|_| TufVerifierError::TargetInventory)?;
    let physical_name = logical_name.consistent_snapshot_name(&sha256);
    Ok(AuthenticatedTargetDescriptor {
        logical_name,
        physical_name,
        length: target.length,
        sha256,
    })
}

fn require_all_fresh(
    trusted: &sigstore_tuf::trusted::TrustedMetadataSet,
    reference: Timestamp,
) -> Result<(), TufVerifierError> {
    profile::require_fresh(trusted.root(), reference)?;
    profile::require_fresh(
        trusted
            .timestamp()
            .ok_or(TufVerifierError::AuthenticationFailed)?,
        reference,
    )?;
    profile::require_fresh(
        trusted
            .snapshot()
            .ok_or(TufVerifierError::AuthenticationFailed)?,
        reference,
    )?;
    profile::require_fresh(
        trusted
            .targets()
            .ok_or(TufVerifierError::AuthenticationFailed)?,
        reference,
    )
}

fn earliest_expiry(
    trusted: &sigstore_tuf::trusted::TrustedMetadataSet,
) -> Result<Timestamp, TufVerifierError> {
    [
        profile::expiry(trusted.root())?,
        profile::expiry(
            trusted
                .timestamp()
                .ok_or(TufVerifierError::AuthenticationFailed)?,
        )?,
        profile::expiry(
            trusted
                .snapshot()
                .ok_or(TufVerifierError::AuthenticationFailed)?,
        )?,
        profile::expiry(
            trusted
                .targets()
                .ok_or(TufVerifierError::AuthenticationFailed)?,
        )?,
    ]
    .into_iter()
    .min()
    .ok_or(TufVerifierError::AuthenticationFailed)
}

fn descriptor_matches(
    pointer: &crate::distribution::schema::ReleaseTargetDescriptorV1,
    target: &AuthenticatedTargetDescriptor,
) -> bool {
    pointer.name() == target.logical_name()
        && pointer.length() == target.length()
        && pointer.sha256() == target.sha256()
}

fn same_authenticated_descriptor(
    left: &AuthenticatedTargetDescriptor,
    right: &AuthenticatedTargetDescriptor,
) -> bool {
    left.logical_name == right.logical_name
        && left.length == right.length
        && left.sha256 == right.sha256
}

#[cfg(test)]
pub(super) fn authenticate_selected_targets_for_test(
    authorization: &MetadataStateAuthorization,
    anchor: &EmbeddedTrustRoot,
    samples: impl IntoIterator<Item = Timestamp>,
) -> Result<AuthenticatedTargetSet, TufVerifierError> {
    authenticate_selected_targets_with_clock(authorization, anchor, ClockSource::scripted(samples))
}

#[cfg(test)]
#[path = "target_set_tests.rs"]
mod tests;
