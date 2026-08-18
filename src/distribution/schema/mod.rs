//! Versioned wire schemas used by hf2q distribution artifacts.

mod common;
mod install_receipt;
mod release_manifest;

pub use common::{
    BundleEntryType, BundlePath, FileMode, GitCommit, MacOsVersion, ReleaseVersion, Sha256Digest,
    TargetTriple, UpdateChannel,
};
pub use install_receipt::{
    AbsoluteInstallPath, InstallReceiptError, InstallReceiptV1, InstallationId, InstalledReleaseV1,
    InstalledVersionMarkerV1, MarkerReleaseV1, OwnerFamily, RecordedBundleIdentityV1,
    RecordedTransitionEvidenceV1, SuccessfulTransitionV1, TransitionEndpointV1, TransitionKind,
    UpdateRoute, INSTALLATION_LAYOUT_SCHEMA_V1, INSTALLED_VERSION_MARKER_KIND,
    INSTALLED_VERSION_MARKER_SCHEMA_VERSION, INSTALL_RECEIPT_KIND, INSTALL_RECEIPT_SCHEMA_VERSION,
    MAX_INSTALLED_VERSION_MARKER_BYTES, MAX_INSTALL_RECEIPT_BYTES, STATE_LAYOUT_SCHEMA_V1,
};
pub use release_manifest::{
    BundleFileV1, CodeSigningIdentityV1, CompatibilityV1, DynamicDependencyV1,
    ReleaseManifestError, ReleaseManifestV1, RELEASE_MANIFEST_KIND,
    RELEASE_MANIFEST_SCHEMA_VERSION,
};
