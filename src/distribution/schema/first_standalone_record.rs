use sha2::{Digest, Sha256};

use super::common::{ReleaseVersion, Sha256Digest, TargetTriple};
use super::install_receipt::{
    build_first_install_receipt, AbsoluteInstallPath, InstallReceiptError, InstallReceiptV1,
    InstallationId,
};
use super::installed_version_marker::{
    InstalledVersionMarkerV2, RecordedPreparationEvidenceV2, MAX_INSTALLED_VERSION_MARKER_BYTES,
};

/// Canonical marker and receipt bytes for the first standalone installation.
///
/// This record is structural output only. Its private constructor is intended
/// for the future lock-held prepared-version coordinator after it has
/// authenticated every supplied value.
#[derive(Debug)]
pub(in crate::distribution) struct FirstStandaloneInstallRecord {
    marker: InstalledVersionMarkerV2,
    marker_bytes: Vec<u8>,
    marker_sha256: Sha256Digest,
    receipt: InstallReceiptV1,
    receipt_bytes: Vec<u8>,
}

impl FirstStandaloneInstallRecord {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::distribution) fn build(
        installation_id: InstallationId,
        installation_root: AbsoluteInstallPath,
        version: ReleaseVersion,
        target: TargetTriple,
        release_manifest_sha256: Sha256Digest,
        archive_sha256: Sha256Digest,
        prepared_from: RecordedPreparationEvidenceV2,
        installed_at_unix_seconds: u64,
    ) -> Result<Self, InstallReceiptError> {
        let marker = InstalledVersionMarkerV2::first_install(
            installation_id,
            installation_root,
            version,
            target,
            release_manifest_sha256,
            archive_sha256,
            prepared_from,
            installed_at_unix_seconds,
        );
        let marker_bytes = marker.to_deterministic_json()?;
        Self::reconstruct_from_exact_marker(&marker_bytes)
    }

    /// Reconstructs the exact first-activation receipt from a durable marker.
    ///
    /// This is the crash-recovery path after prepared-version publication.
    /// Canonical marker bytes are mandatory because their exact digest is part
    /// of the reconstructed receipt.
    pub(in crate::distribution) fn reconstruct_from_exact_marker(
        marker_bytes: &[u8],
    ) -> Result<Self, InstallReceiptError> {
        if marker_bytes.len() > MAX_INSTALLED_VERSION_MARKER_BYTES {
            return Err(InstallReceiptError::InputTooLarge {
                document: "installed-version marker",
                limit: MAX_INSTALLED_VERSION_MARKER_BYTES,
                actual: marker_bytes.len(),
            });
        }
        let marker_sha256 = Sha256Digest::parse(
            "installed_version_marker_sha256",
            hex::encode(Sha256::digest(marker_bytes)),
        )?;
        let marker =
            InstalledVersionMarkerV2::parse_and_validate_exact(marker_bytes, &marker_sha256)?;
        if marker.installation_sequence() != 1 {
            return Err(InstallReceiptError::invalid(
                "installation_sequence",
                "first standalone installation record requires sequence one",
            ));
        }
        let (receipt, receipt_bytes) = build_first_install_receipt(&marker, &marker_sha256)?;

        Ok(Self {
            marker,
            marker_bytes: marker_bytes.to_vec(),
            marker_sha256,
            receipt,
            receipt_bytes,
        })
    }

    pub(in crate::distribution) fn marker(&self) -> &InstalledVersionMarkerV2 {
        &self.marker
    }

    pub(in crate::distribution) fn marker_bytes(&self) -> &[u8] {
        &self.marker_bytes
    }

    pub(in crate::distribution) fn marker_sha256(&self) -> &Sha256Digest {
        &self.marker_sha256
    }

    pub(in crate::distribution) fn receipt(&self) -> &InstallReceiptV1 {
        &self.receipt
    }

    pub(in crate::distribution) fn receipt_bytes(&self) -> &[u8] {
        &self.receipt_bytes
    }
}
