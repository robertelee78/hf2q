use std::collections::BTreeMap;
use std::path::{Component, Path};

use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::core::integrity::ShardIntegrity;
use crate::core::provenance::{compute_source_bundle_sha256, SourceShard};
use crate::core::sha256::compute_file_sha256;
use crate::input::integrity::VerifiedSourceManifest;

const MANIFEST_BYTES: &[u8] = include_bytes!(
    "../../../../../../data/calibration/qwen38-source-teacher-canary-v1/source-manifest.json"
);
pub(super) const MANIFEST_SHA256: &str =
    "e87e1b2637f929288fded02e3acbb72818fe18d4fd29d21558e6007385eee09d";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OfficialSourceManifestV1 {
    kind: String,
    schema_version: u32,
    manifest_id: String,
    source: OfficialSourceIdentityV1,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OfficialSourceIdentityV1 {
    repository_id: String,
    repository_type: String,
    canonical_url: String,
    revision: String,
    bundle_sha256: String,
    files: Vec<OfficialSourceFileV1>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OfficialSourceFileV1 {
    path: String,
    size: u64,
    sha256: String,
    hub_etag: String,
    #[serde(default)]
    hf_lfs_sha256: Option<String>,
}

impl OfficialSourceManifestV1 {
    pub(super) fn manifest_id(&self) -> &str {
        &self.manifest_id
    }

    pub(super) fn manifest_sha256(&self) -> &'static str {
        MANIFEST_SHA256
    }

    pub(super) fn repository_id(&self) -> &str {
        &self.source.repository_id
    }

    pub(super) fn revision(&self) -> &str {
        &self.source.revision
    }

    pub(super) fn bundle_sha256(&self) -> &str {
        &self.source.bundle_sha256
    }

    pub(super) fn files(&self) -> &[OfficialSourceFileV1] {
        &self.source.files
    }

    pub(super) fn records(&self) -> Vec<ShardIntegrity> {
        self.source
            .files
            .iter()
            .map(|file| ShardIntegrity {
                filename: file.path.clone(),
                bytes: file.size,
                sha256: file.hf_lfs_sha256.clone(),
                hf_etag: file.hub_etag.clone(),
                is_lfs: file.hf_lfs_sha256.is_some(),
            })
            .collect()
    }

    pub(super) fn verify_source(
        &self,
        local_dir: &Path,
        verified: &VerifiedSourceManifest,
    ) -> Result<()> {
        ensure!(
            verified.repo() == self.repository_id()
                && verified.revision() == self.revision()
                && verified.records().len() == self.source.files.len(),
            "authenticated source identity differs from the source-teacher manifest"
        );
        let records = verified
            .records()
            .iter()
            .map(|record| (record.filename.as_str(), record))
            .collect::<BTreeMap<_, _>>();
        for expected in &self.source.files {
            let actual = records
                .get(expected.path.as_str())
                .with_context(|| format!("authenticated source is missing `{}`", expected.path))?;
            ensure!(
                actual.bytes == expected.size
                    && actual.hf_etag.eq_ignore_ascii_case(&expected.hub_etag)
                    && actual.is_lfs == expected.hf_lfs_sha256.is_some()
                    && actual.sha256.as_deref() == expected.hf_lfs_sha256.as_deref(),
                "authenticated source record `{}` differs from the source-teacher manifest",
                expected.path
            );
            if expected.hf_lfs_sha256.is_none() {
                ensure!(
                    compute_file_sha256(&local_dir.join(&expected.path))? == expected.sha256,
                    "source file `{}` differs from the source-teacher manifest",
                    expected.path
                );
            }
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.kind == "hf2q.qwen38-source-teacher-manifest"
                && self.schema_version == 1
                && self.manifest_id == "qwen38-source-teacher-v1",
            "unsupported source-teacher manifest identity"
        );
        ensure!(
            self.source.repository_id == "Qwen/Qwen3.8-27B"
                && self.source.repository_type == "model"
                && self.source.canonical_url == "https://huggingface.co/Qwen/Qwen3.8-27B"
                && self.source.revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
                && self.source.files.len() == 29,
            "source-teacher manifest does not describe the accepted Qwen3.8 source"
        );
        require_lower_hex(&self.source.bundle_sha256, 64, "source bundle SHA-256")?;
        let mut previous = None;
        for file in &self.source.files {
            file.validate()?;
            if let Some(previous) = previous {
                ensure!(
                    previous < file.path.as_str(),
                    "source files are not unique and sorted"
                );
            }
            previous = Some(file.path.as_str());
        }
        let shards = self
            .source
            .files
            .iter()
            .map(|file| SourceShard {
                filename: file.path.clone(),
                bytes: file.size,
                sha256: file.hf_lfs_sha256.clone(),
                hf_etag: file.hub_etag.clone(),
                is_lfs: file.hf_lfs_sha256.is_some(),
                verified_at_secs: 0,
            })
            .collect::<Vec<_>>();
        ensure!(
            compute_source_bundle_sha256(&shards).as_deref()
                == Some(self.source.bundle_sha256.as_str()),
            "source bundle SHA-256 differs from the file inventory"
        );
        Ok(())
    }
}

impl OfficialSourceFileV1 {
    pub(super) fn path(&self) -> &str {
        &self.path
    }

    pub(super) fn size(&self) -> u64 {
        self.size
    }

    fn validate(&self) -> Result<()> {
        let path = Path::new(&self.path);
        ensure!(
            !self.path.is_empty()
                && self.path.len() <= crate::input::hf_reference::MAX_HF_FILENAME_BYTES
                && self.path.is_ascii()
                && !self.path.contains('\\')
                && path.components().count() == 1
                && path
                    .components()
                    .all(|component| matches!(component, Component::Normal(_))),
            "unsafe source-teacher filename"
        );
        ensure!(self.size > 0, "source-teacher file is empty");
        require_lower_hex(&self.sha256, 64, "source file SHA-256")?;
        if let Some(lfs) = &self.hf_lfs_sha256 {
            require_lower_hex(lfs, 64, "source file LFS SHA-256")?;
            ensure!(
                lfs == &self.sha256 && lfs == &self.hub_etag,
                "source file LFS identities differ"
            );
        } else {
            require_lower_hex(&self.hub_etag, 40, "source file Git blob SHA-1")?;
        }
        ensure!(
            !self.path.ends_with(".safetensors") || self.hf_lfs_sha256.is_some(),
            "safetensors source lacks a strong LFS identity"
        );
        Ok(())
    }
}

pub(super) fn official_source_manifest() -> Result<OfficialSourceManifestV1> {
    ensure!(
        hex::encode(Sha256::digest(MANIFEST_BYTES)) == MANIFEST_SHA256,
        "embedded source-teacher manifest bytes changed"
    );
    let manifest: OfficialSourceManifestV1 =
        serde_json::from_slice(MANIFEST_BYTES).context("parse embedded source-teacher manifest")?;
    manifest.validate()?;
    Ok(manifest)
}

fn require_lower_hex(value: &str, length: usize, field: &str) -> Result<()> {
    ensure!(
        value.len() == length
            && value
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        "invalid {field}"
    );
    Ok(())
}

#[cfg(test)]
pub(super) fn manifest_bytes_for_test() -> &'static [u8] {
    MANIFEST_BYTES
}
