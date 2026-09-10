//! Actual converted checkpoint identity, distinct from model-card ancestry.
//!
//! Reuse hf2q's adjacent conversion receipt and stable-output verification.
//! Never dereference receipt.output.path or invent a GGUF source-revision key.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use mlx_native::gguf::GgufFile;

use super::compatibility::CheckpointIdentity;
use crate::convert::receipt::{receipt_path, ConversionReceipt, CONVERSION_RECEIPT_SCHEMA_VERSION};
use crate::core::bounded_file::{
    read_bounded_regular_nofollow, StableFileIdentity, StableRegularFile,
};
use crate::input::hf_reference::HfModelReference;

const MAX_RECEIPT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_VERIFIED_OUTPUTS: usize = 16;

// Resolver and family loader check the same artifact in one process. Reuse a
// computed output digest only while its retained-file identity is unchanged.
// This cache stores no source claims; each call re-reads/validates the receipt.
static VERIFIED_OUTPUTS: Mutex<Vec<(StableFileIdentity, String)>> = Mutex::new(Vec::new());

pub(super) fn from_conversion_receipt(
    model_path: &Path,
    model: &GgufFile,
) -> Result<Option<CheckpointIdentity>> {
    let bytes = std::fs::metadata(model_path)
        .context("inspect GLP target model")?
        .len();
    let mut opened = StableRegularFile::open_operator_path_exact(model_path, bytes)?
        .context("GLP target is not one stable regular file")?;
    let canonical = opened
        .canonical_path_for_identity()?
        .context("cannot identify the retained GLP target model")?;
    let sidecar = receipt_path(&canonical);
    if !sidecar.try_exists()? {
        return Ok(None);
    }
    let bytes = read_bounded_regular_nofollow(&sidecar, MAX_RECEIPT_BYTES)?
        .context("GLP target conversion receipt is not one stable bounded regular file")?;
    let receipt: ConversionReceipt =
        serde_json::from_slice(&bytes).context("parse GLP target conversion receipt")?;
    if receipt.schema_version != CONVERSION_RECEIPT_SCHEMA_VERSION {
        bail!("GLP target conversion receipt has an unsupported schema version");
    }
    crate::serve::api::local_artifacts::validate_receipt_identity(&receipt)?;
    let original = HfModelReference::parse(&receipt.source.original_reference, None)?;
    if original.repo_id() != receipt.source.repository_id
        || original.filename() != receipt.source.filename.as_deref()
    {
        bail!("conversion receipt original reference does not identify its source repository/file");
    }
    let reference = HfModelReference::parse(&receipt.source.canonical_url, None)?;
    if reference.repo_id() != receipt.source.repository_id
        || reference.requested_revision().is_some()
        || reference.filename().is_some()
    {
        bail!("conversion receipt canonical URL does not identify its source repository");
    }
    let reference = reference.resolve(&receipt.source.revision)?;
    if receipt.output.size != std::fs::metadata(model_path)?.len() {
        bail!("GLP target model size does not match its conversion receipt");
    }
    if let Some(bundle) = model.metadata_string(crate::core::provenance::KEY_SOURCE_SHA256) {
        if !bundle.eq_ignore_ascii_case(&receipt.source.bundle_sha256) {
            bail!("GLP target source-bundle metadata contradicts its conversion receipt");
        }
    }
    verify_output(&mut opened, &receipt.output.sha256)?;
    Ok(Some(CheckpointIdentity {
        name: reference.repo_id().rsplit('/').next().map(str::to_owned),
        // GGUF organization strings may be descriptive labels, so an HF
        // account handle is not silently repurposed as one.
        organization: None,
        repository: Some(reference.repo_id().to_owned()),
        revision: Some(reference.revision().to_owned()),
    }))
}

fn verify_output(opened: &mut StableRegularFile, expected: &str) -> Result<()> {
    let identity = opened.identity();
    let cached = VERIFIED_OUTPUTS
        .lock()
        .map_err(|_| anyhow::anyhow!("GLP identity cache poisoned"))?
        .iter()
        .any(|(file, digest)| *file == identity && digest.eq_ignore_ascii_case(expected));
    if !cached {
        tracing::info!("verifying converted model SHA-256 for GLP checkpoint binding");
        let actual = opened
            .sha256_with_progress(|_| {})?
            .context("GLP target model changed during receipt verification")?;
        if !actual.eq_ignore_ascii_case(expected) {
            bail!("GLP target model SHA-256 does not match its conversion receipt");
        }
        let mut verified = VERIFIED_OUTPUTS
            .lock()
            .map_err(|_| anyhow::anyhow!("GLP identity cache poisoned"))?;
        if verified.len() >= MAX_VERIFIED_OUTPUTS {
            verified.remove(0);
        }
        verified.push((identity, actual));
    }
    if !opened.is_stable()? {
        bail!("GLP target model changed after receipt verification");
    }
    Ok(())
}

#[cfg(test)]
#[path = "source_identity_tests.rs"]
mod tests;
