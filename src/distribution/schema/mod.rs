//! Versioned wire schemas used by hf2q distribution artifacts.

mod channel_pointer;
mod common;
// The next prepared-version slice consumes this private structural builder.
#[allow(dead_code)]
mod first_standalone_record;
mod install_receipt;
mod installation_identity;
mod installed_version_marker;
mod release_manifest;
mod target_name;

pub use channel_pointer::{
    ChannelPointerError, ChannelPointerV1, ReleaseTargetDescriptorV1, CHANNEL_POINTER_KIND,
    CHANNEL_POINTER_SCHEMA_VERSION, MAX_CHANNEL_POINTER_BYTES, MAX_RELEASE_ARCHIVE_BYTES,
};

pub use common::{
    BundleEntryType, BundlePath, FileMode, GitCommit, MacOsVersion, ReleaseVersion, Sha256Digest,
    TargetTriple, UpdateChannel,
};
#[allow(unused_imports)]
pub(in crate::distribution) use first_standalone_record::FirstStandaloneInstallRecord;
pub use install_receipt::{
    AbsoluteInstallPath, InstallReceiptError, InstallReceiptV1, InstallationId, InstalledReleaseV1,
    OwnerFamily, RecordedBundleIdentityV1, RecordedTransitionEvidenceV1, SuccessfulTransitionV1,
    TransitionEndpointV1, TransitionKind, UpdateRoute, INSTALLATION_LAYOUT_SCHEMA_V1,
    INSTALL_RECEIPT_KIND, INSTALL_RECEIPT_SCHEMA_VERSION, MAX_INSTALL_RECEIPT_BYTES,
    STATE_LAYOUT_SCHEMA_V1,
};
pub(in crate::distribution) use installation_identity::{
    InstallationIdentityError, InstallationIdentityV1, MAX_INSTALLATION_IDENTITY_BYTES,
};
pub use installed_version_marker::{
    InstalledVersionMarkerV2, MarkerReleaseV1, RecordedPreparationEvidenceV2,
    INSTALLED_VERSION_MARKER_KIND, INSTALLED_VERSION_MARKER_SCHEMA_VERSION,
    MAX_INSTALLED_VERSION_MARKER_BYTES,
};
pub use release_manifest::{
    BundleFileV1, CodeSigningIdentityV1, CompatibilityV1, DynamicDependencyV1,
    ReleaseManifestError, ReleaseManifestV1, MAX_RELEASE_MANIFEST_BYTES, RELEASE_MANIFEST_KIND,
    RELEASE_MANIFEST_SCHEMA_VERSION,
};
pub(crate) use release_manifest::{MAX_BUNDLE_DIRECTORIES, MAX_BUNDLE_FILES};
pub(crate) use target_name::{ConsistentSnapshotTargetName, LogicalTargetKind, LogicalTargetName};
