//! Exact, one-hash text-artifact identities shared by release launchers and
//! the server.  The legacy shell receipt remains a gate ledger; only this
//! versioned Rust schema carries the nanosecond filesystem identity needed to
//! skip a second in-process scan without weakening replacement detection.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use super::multi_model::{ArtifactFileStamp, TextArtifactIdentity};

pub const MODEL_VERIFICATION_RECEIPT_ENV: &str = "HF2Q_MODEL_VERIFICATION_RECEIPT";
pub const MODEL_VERIFICATION_RECEIPT_DIR_ENV: &str = "HF2Q_MODEL_VERIFICATION_RECEIPT_DIR";
pub const MODEL_VERIFICATION_RECEIPT_SCHEMA_V2: u32 = 2;
const MAX_RECEIPT_BYTES: u64 = 64 * 1024;
const MAX_RECEIPTS: usize = 512;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ModelVerificationReceiptV2 {
    pub schema_version: u32,
    pub path: PathBuf,
    pub sha256: String,
    /// The portable stamp the shell harness checks before reusing the v2
    /// content-hash evidence.  `file_stamp` remains the stricter runtime
    /// authority, including nanoseconds.
    pub file_snapshot: String,
    pub file_stamp: ArtifactFileStamp,
    pub content_hash_verified: bool,
}

fn read_receipt_value(receipt_path: &Path) -> Result<serde_json::Value> {
    let metadata = fs::symlink_metadata(receipt_path).with_context(|| {
        format!(
            "inspect model verification receipt {}",
            receipt_path.display()
        )
    })?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() > MAX_RECEIPT_BYTES
    {
        bail!("model verification receipt must be a bounded non-symlink regular file");
    }
    let bytes = fs::read(receipt_path)
        .with_context(|| format!("read model verification receipt {}", receipt_path.display()))?;
    serde_json::from_slice(&bytes).context("parse model verification receipt")
}

/// Load a bounded operator-owned directory of exact identities for model-swap
/// candidates. Every entry must be schema v2 and current; one stale or legacy
/// receipt rejects the whole authority rather than silently making one switch
/// hash again. The returned canonical paths are suitable for AppState's exact
/// per-artifact policy registry.
pub fn verified_entries_from_directory(
    receipt_dir: &Path,
) -> Result<Vec<(PathBuf, TextArtifactIdentity)>> {
    if !receipt_dir.is_absolute() {
        bail!("model verification receipt directory must be absolute");
    }
    let metadata = fs::symlink_metadata(receipt_dir).with_context(|| {
        format!(
            "inspect model verification receipt directory {}",
            receipt_dir.display()
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!("model verification receipt directory must be a non-symlink directory");
    }
    let mut paths = fs::read_dir(receipt_dir)
        .with_context(|| {
            format!(
                "read model verification receipt directory {}",
                receipt_dir.display()
            )
        })?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.sort();
    if paths.is_empty() {
        bail!("model verification receipt directory must not be empty");
    }
    if paths.len() > MAX_RECEIPTS {
        bail!("model verification receipt directory exceeds bounded entry limit");
    }

    let mut seen = BTreeSet::new();
    let mut entries = Vec::with_capacity(paths.len());
    for receipt_path in paths {
        if receipt_path.extension().and_then(|value| value.to_str()) != Some("json") {
            bail!("model verification receipt directory contains a non-JSON entry");
        }
        let value = read_receipt_value(&receipt_path)?;
        let receipt: ModelVerificationReceiptV2 =
            serde_json::from_value(value).context("parse schema-v2 model verification receipt")?;
        let expected_path = receipt.path.clone();
        let canonical = expected_path
            .canonicalize()
            .context("canonicalize directory receipt text artifact")?;
        let identity = receipt.into_identity_for(&canonical)?;
        if !seen.insert(canonical.clone()) {
            bail!("model verification receipt directory contains a duplicate artifact");
        }
        entries.push((canonical, identity));
    }
    Ok(entries)
}

pub fn verified_entries_from_directory_env() -> Result<Vec<(PathBuf, TextArtifactIdentity)>> {
    let Some(receipt_dir) = std::env::var_os(MODEL_VERIFICATION_RECEIPT_DIR_ENV) else {
        return Ok(Vec::new());
    };
    verified_entries_from_directory(&PathBuf::from(receipt_dir))
}

impl ModelVerificationReceiptV2 {
    fn into_identity_for(self, expected_path: &Path) -> Result<TextArtifactIdentity> {
        if self.schema_version != MODEL_VERIFICATION_RECEIPT_SCHEMA_V2
            || !self.content_hash_verified
        {
            bail!("model verification receipt is not runtime schema v2");
        }
        let expected = expected_path
            .canonicalize()
            .context("canonicalize requested text artifact")?;
        let recorded = self
            .path
            .canonicalize()
            .context("canonicalize receipt text artifact")?;
        if recorded != expected {
            bail!("model verification receipt names a different text artifact");
        }
        let identity = TextArtifactIdentity {
            sha256: self.sha256,
            stamp: self.file_stamp,
        };
        identity.validate()?;
        if self.file_snapshot != identity.stamp.shell_snapshot() {
            bail!("model verification receipt file snapshot disagrees with its runtime stamp");
        }
        if !identity.stamp.matches_path(&expected) {
            bail!("text artifact changed after model verification receipt");
        }
        Ok(identity)
    }
}

/// Hash one stable local artifact and emit the only receipt schema admissible
/// for runtime hash reuse.  It is deliberately process-local / operator-owned
/// evidence, not an API request surface.
pub fn record_model_verification(
    path: &Path,
    expected_sha256: &str,
) -> Result<ModelVerificationReceiptV2> {
    let canonical = path.canonicalize().context("canonicalize model artifact")?;
    let identity = TextArtifactIdentity::inspect(&canonical)?;
    identity.validate()?;
    if identity.sha256 != expected_sha256 {
        bail!("model artifact SHA-256 mismatch");
    }
    Ok(ModelVerificationReceiptV2 {
        schema_version: MODEL_VERIFICATION_RECEIPT_SCHEMA_V2,
        path: canonical,
        sha256: identity.sha256,
        file_snapshot: identity.stamp.shell_snapshot(),
        file_stamp: identity.stamp,
        content_hash_verified: true,
    })
}

/// Resolve a supplied v2 receipt for a startup model.  Absence intentionally
/// preserves the ordinary in-process hash.  A supplied legacy/malformed
/// receipt is an operator error rather than a quiet downgrade to different
/// bytes.
pub fn identity_from_env(expected_path: &Path) -> Result<Option<TextArtifactIdentity>> {
    let Some(receipt_path) = std::env::var_os(MODEL_VERIFICATION_RECEIPT_ENV) else {
        return Ok(None);
    };
    let receipt_path = PathBuf::from(receipt_path);
    let value = read_receipt_value(&receipt_path)?;
    // The shell's v1 receipt intentionally remains valid gate evidence.  It
    // lacks nanosecond fields, however, so it cannot authorize server-side
    // hash reuse; preserve the ordinary loader scan instead.
    if value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        == Some(1)
    {
        return Ok(None);
    }
    let receipt: ModelVerificationReceiptV2 =
        serde_json::from_value(value).context("parse model verification receipt")?;
    Ok(Some(receipt.into_identity_for(expected_path)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v2_receipt_rejects_a_same_path_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model.gguf");
        let replacement = dir.path().join("replacement.gguf");
        fs::write(&model, b"model-a").unwrap();
        let digest = crate::core::sha256::compute_file_sha256(&model).unwrap();
        let receipt = record_model_verification(&model, &digest).unwrap();
        fs::write(&replacement, b"model-b").unwrap();
        fs::rename(&replacement, &model).unwrap();
        assert!(receipt.into_identity_for(&model).is_err());
    }

    #[test]
    fn v2_receipt_rejects_legacy_schema_before_runtime_reuse() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model.gguf");
        fs::write(&model, b"model-a").unwrap();
        let digest = crate::core::sha256::compute_file_sha256(&model).unwrap();
        let mut receipt = record_model_verification(&model, &digest).unwrap();
        receipt.schema_version = 1;
        assert!(receipt.into_identity_for(&model).is_err());
    }

    #[test]
    fn v2_receipt_rejects_a_shell_snapshot_that_disagrees_with_runtime_stamp() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model.gguf");
        fs::write(&model, b"model-a").unwrap();
        let digest = crate::core::sha256::compute_file_sha256(&model).unwrap();
        let mut receipt = record_model_verification(&model, &digest).unwrap();
        receipt.file_snapshot = "not-the-artifact-stamp".into();
        assert!(receipt.into_identity_for(&model).is_err());
    }

    #[test]
    fn receipt_directory_returns_all_current_exact_identities() {
        let dir = tempfile::tempdir().unwrap();
        let receipts = dir.path().join("receipts");
        fs::create_dir(&receipts).unwrap();
        let model_a = dir.path().join("a.gguf");
        let model_b = dir.path().join("b.gguf");
        fs::write(&model_a, b"model-a").unwrap();
        fs::write(&model_b, b"model-b").unwrap();
        for (name, model) in [("a.json", &model_a), ("b.json", &model_b)] {
            let digest = crate::core::sha256::compute_file_sha256(model).unwrap();
            let receipt = record_model_verification(model, &digest).unwrap();
            fs::write(receipts.join(name), serde_json::to_vec(&receipt).unwrap()).unwrap();
        }
        let entries = verified_entries_from_directory(&receipts).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, model_a.canonicalize().unwrap());
        assert_eq!(entries[1].0, model_b.canonicalize().unwrap());
    }

    #[test]
    fn receipt_directory_rejects_one_stale_member() {
        let dir = tempfile::tempdir().unwrap();
        let receipts = dir.path().join("receipts");
        fs::create_dir(&receipts).unwrap();
        let model = dir.path().join("model.gguf");
        fs::write(&model, b"model-a").unwrap();
        let digest = crate::core::sha256::compute_file_sha256(&model).unwrap();
        let receipt = record_model_verification(&model, &digest).unwrap();
        fs::write(
            receipts.join("model.json"),
            serde_json::to_vec(&receipt).unwrap(),
        )
        .unwrap();
        fs::write(&model, b"model-b").unwrap();
        assert!(verified_entries_from_directory(&receipts).is_err());
    }

    #[test]
    fn receipt_directory_rejects_empty_authority() {
        let dir = tempfile::tempdir().unwrap();
        assert!(verified_entries_from_directory(dir.path()).is_err());
    }
}
