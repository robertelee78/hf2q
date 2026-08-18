//! Versioned wire schemas used by hf2q distribution artifacts.

mod common;
mod release_manifest;

pub use common::{
    BundleEntryType, BundlePath, FileMode, GitCommit, MacOsVersion, ReleaseVersion, Sha256Digest,
    TargetTriple, UpdateChannel,
};
pub use release_manifest::{
    BundleFileV1, CodeSigningIdentityV1, CompatibilityV1, DynamicDependencyV1,
    ReleaseManifestError, ReleaseManifestV1, RELEASE_MANIFEST_KIND,
    RELEASE_MANIFEST_SCHEMA_VERSION,
};
