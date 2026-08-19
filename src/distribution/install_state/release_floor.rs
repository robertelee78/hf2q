use sha2::{Digest, Sha256};

use super::file;
use super::identity::LockedInstallationIdentity;
use super::unix::{self, Directory, EntryIdentity};
use super::verify;
use super::{InstallStateError, CURRENT_TARGET, FIRST_GENERATION, FIRST_SEQUENCE};
use crate::distribution::schema::{
    ReleaseVersion, Sha256Digest, TargetTriple, MAX_INSTALL_RECEIPT_BYTES,
};

const ACTIVATIONS: &str = "activations";
const CURRENT: &str = "current";
const RECEIPT: &str = "install-receipt.json";
const VERSIONS: &str = "versions";

/// Descriptor-verified state of the active standalone release.
///
/// This is not constructed from caller-supplied receipt or marker bytes. The
/// shared-lock reader verifies the exact named activation, receipt, version,
/// marker, manifest, and payload inventory before this value can exist.
#[derive(Debug, PartialEq, Eq)]
pub(in crate::distribution) struct ActiveInstalledReleaseFloor {
    version: ReleaseVersion,
    target: TargetTriple,
    manifest_sha256: Sha256Digest,
    archive_sha256: Sha256Digest,
    installation_sequence: u64,
    activation_sequence: u64,
    receipt_sha256: [u8; 32],
}

/// Live automatic-update floor read under the shared installation lock.
///
/// `Absent` means that no `current` activation exists at either read
/// snapshot. It does not authorize adoption of staged or unselected state.
#[derive(Debug, PartialEq, Eq)]
pub(in crate::distribution) enum LiveInstalledReleaseFloor {
    Absent,
    Active(ActiveInstalledReleaseFloor),
}

impl ActiveInstalledReleaseFloor {
    pub(in crate::distribution) fn version(&self) -> &ReleaseVersion {
        &self.version
    }

    pub(in crate::distribution) fn target(&self) -> TargetTriple {
        self.target
    }

    pub(in crate::distribution) fn manifest_sha256(&self) -> &Sha256Digest {
        &self.manifest_sha256
    }

    pub(in crate::distribution) fn archive_sha256(&self) -> &Sha256Digest {
        &self.archive_sha256
    }

    pub(in crate::distribution) fn installation_sequence(&self) -> u64 {
        self.installation_sequence
    }

    pub(in crate::distribution) fn activation_sequence(&self) -> u64 {
        self.activation_sequence
    }

    pub(in crate::distribution) fn receipt_sha256(&self) -> [u8; 32] {
        self.receipt_sha256
    }

    #[cfg(test)]
    pub(in crate::distribution) fn for_test(
        version: &str,
        manifest_sha256: &str,
        archive_sha256: &str,
    ) -> Self {
        Self {
            version: ReleaseVersion::parse_stable("test.version", version.to_owned())
                .expect("test release version"),
            target: TargetTriple::Aarch64AppleDarwin,
            manifest_sha256: Sha256Digest::parse(
                "test.manifest_sha256",
                manifest_sha256.to_owned(),
            )
            .expect("test manifest digest"),
            archive_sha256: Sha256Digest::parse("test.archive_sha256", archive_sha256.to_owned())
                .expect("test archive digest"),
            installation_sequence: FIRST_SEQUENCE,
            activation_sequence: FIRST_SEQUENCE,
            receipt_sha256: [0x5a; 32],
        }
    }
}

struct ActiveSnapshot {
    floor: ActiveInstalledReleaseFloor,
    versions: Directory,
    version: Directory,
    activations: Directory,
    activation: Directory,
    receipt_identity: EntryIdentity,
}

pub(super) fn read_live_installed_release_floor(
    locked: &LockedInstallationIdentity,
) -> Result<LiveInstalledReleaseFloor, InstallStateError> {
    let first = read_snapshot(locked)?;
    let second = read_snapshot(locked)?;
    match (first, second) {
        (None, None) => Ok(LiveInstalledReleaseFloor::Absent),
        (Some(first), Some(second))
            if first.floor == second.floor
                && first.versions.same_object(&second.versions)
                && first.version.same_object(&second.version)
                && first.activations.same_object(&second.activations)
                && first.activation.same_object(&second.activation)
                && first.receipt_identity == second.receipt_identity =>
        {
            Ok(LiveInstalledReleaseFloor::Active(second.floor))
        }
        _ => Err(InstallStateError::InvalidLayout(
            "active standalone release changed between floor snapshots",
        )),
    }
}

fn read_snapshot(
    locked: &LockedInstallationIdentity,
) -> Result<Option<ActiveSnapshot>, InstallStateError> {
    let live = locked.reopen()?;
    if unix::entry_identity(&live.root, CURRENT)?.is_none() {
        return Ok(None);
    }
    if unix::read_symlink(&live.root, CURRENT)? != CURRENT_TARGET {
        return Err(InstallStateError::InvalidLayout(
            "current does not select the canonical active release",
        ));
    }

    let versions = unix::open_directory_at(&live.root, VERSIONS, Some(0o700), true)?;
    let activations = unix::open_directory_at(&live.root, ACTIVATIONS, Some(0o700), true)?;
    if unix::list_names_bounded(&activations, 2)?
        != std::collections::BTreeSet::from([FIRST_GENERATION.to_owned()])
    {
        return Err(InstallStateError::InvalidLayout(
            "active activation inventory is not exact",
        ));
    }
    let activation = unix::open_directory_at(&activations, FIRST_GENERATION, Some(0o700), true)?;
    let (receipt_file, receipt_bytes, receipt_identity) =
        file::read_regular_file(&activation, RECEIPT, 0o600, MAX_INSTALL_RECEIPT_BYTES)?;
    let receipt = verify::validate_first_receipt(&receipt_bytes, locked.state_root())?;
    if receipt.installation_id() != locked.installation_id() {
        return Err(InstallStateError::InvalidLayout(
            "active receipt belongs to a different installation identity",
        ));
    }
    let prepared = verify::verify_prepared_version(&versions, &receipt)?;
    let verified_activation = verify::verify_committed_first_activation(
        &live.root,
        &activations,
        &receipt,
        &receipt_bytes,
        &prepared,
    )?;
    if unix::regular_file_identity(&receipt_file, activation.device())? != receipt_identity
        || unix::regular_file_identity(&verified_activation.receipt_file, activation.device())?
            != receipt_identity
    {
        return Err(InstallStateError::InvalidLayout(
            "active receipt identity changed while reading its floor",
        ));
    }

    let release = receipt.active();
    let bundle = release.bundle().ok_or(InstallStateError::InvalidLayout(
        "active receipt lacks its bundle identity",
    ))?;
    let installation_sequence =
        bundle
            .installation_sequence()
            .ok_or(InstallStateError::InvalidLayout(
                "active receipt lacks its installation sequence",
            ))?;
    if installation_sequence != FIRST_SEQUENCE {
        return Err(InstallStateError::InvalidLayout(
            "active release is outside the implemented sequence-one floor",
        ));
    }

    Ok(Some(ActiveSnapshot {
        floor: ActiveInstalledReleaseFloor {
            version: release.version().clone(),
            target: release.target(),
            manifest_sha256: bundle.release_manifest_sha256().clone(),
            archive_sha256: bundle.archive_sha256().clone(),
            installation_sequence,
            activation_sequence: FIRST_SEQUENCE,
            receipt_sha256: Sha256::digest(&receipt_bytes).into(),
        },
        versions,
        version: prepared.directory,
        activations,
        activation: verified_activation.directory,
        receipt_identity,
    }))
}
