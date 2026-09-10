//! Declared checkpoint compatibility, independent of device allocation.
//!
//! Weightless spec/GLP.md defines base_model.0.version as the HF commit.
//! glp.content_sha256 identifies direction tensors, never model weights.

use std::path::Path;

use anyhow::{bail, Context, Result};
use mlx_native::gguf::{GgufFile, MetadataValue};

use crate::input::hf_reference::HfModelReference;

/// Checkpoint identity declared in a GGUF. Metadata is a compatibility
/// assertion, not a cryptographic attestation of the model's weight bytes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CheckpointIdentity {
    pub name: Option<String>,
    pub organization: Option<String>,
    pub repository: Option<String>,
    pub revision: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compatibility {
    /// Legacy control vector with no checkpoint identity declaration.
    Undeclared,
    /// Names agree, but the vector does not declare an immutable commit.
    Named,
    /// Repository/name and the declared immutable commit agree.
    Checkpoint,
}

impl CheckpointIdentity {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        Self::from_metadata(|key| gguf.metadata(key), true)
    }

    fn from_metadata<'a>(
        get: impl Fn(&str) -> Option<&'a MetadataValue>,
        glp: bool,
    ) -> Result<Self> {
        if let Some(count) = get("general.base_model.count") {
            if count.as_u32() != Some(1) {
                bail!("GLP checkpoint binding requires exactly one declared base model");
            }
        }
        let mut identity = Self {
            name: string_metadata(&get, "general.base_model.0.name")?,
            organization: string_metadata(&get, "general.base_model.0.organization")?,
            repository: string_metadata(&get, "general.base_model.0.repo_url")?
                .map(|url| {
                    let reference = HfModelReference::parse(&url, None)
                        .context("invalid base_model.0.repo_url")?;
                    if reference.filename().is_some() || reference.requested_revision().is_some() {
                        bail!("base_model.0.repo_url must identify a repository; use .0.version for its HF commit");
                    }
                    Ok(reference.repo_id().to_owned())
                })
                .transpose()?,
            revision: string_metadata(&get, "general.base_model.0.version")?,
        };
        if glp && get("general.base_model.0.version").is_some() && identity.revision.is_none() {
            bail!("declared GLP base_model.0.version must name an exact HF commit");
        }
        if let Some(revision) = identity.revision.as_mut() {
            if revision.len() != 40 || !revision.bytes().all(|b| b.is_ascii_hexdigit()) {
                if glp {
                    bail!("base_model.0.version must be an exact 40-hex HF commit, not a branch or model-byte hash");
                }
                // Ordinary model GGUFs also use human version labels. They
                // cannot satisfy a GLP checkpoint pin, but are not malformed.
                identity.revision = None;
            } else {
                revision.make_ascii_lowercase();
            }
        }
        // A repo-qualified name is itself an explicit namespace declaration.
        if let Some(name) = identity.name.as_deref().filter(|name| name.contains('/')) {
            let reference = HfModelReference::parse(name, None)
                .context("invalid repository-qualified base model name")?;
            if let Some(repository) = identity.repository.as_deref() {
                if !repository.eq_ignore_ascii_case(reference.repo_id()) {
                    bail!("base model name and repo_url declare different repositories");
                }
            }
            identity.repository = Some(reference.repo_id().to_owned());
        }
        if let (Some(name), Some(repository)) = (&identity.name, &identity.repository) {
            if normalize_name(name.rsplit('/').next().unwrap_or(name))
                != normalize_name(repository.rsplit('/').next().unwrap_or(repository))
            {
                bail!("base model name and repo_url declare different model names");
            }
        }
        Ok(identity)
    }

    /// Model display names are a fallback for name-only legacy vectors.
    /// They never supply an organization, repository, or checkpoint revision.
    pub fn from_model_gguf(gguf: &GgufFile) -> Result<Self> {
        let mut identity = Self::from_metadata(|key| gguf.metadata(key), false)?;
        if identity.name.is_none() && identity.repository.is_none() {
            identity.name = gguf.metadata_string("general.name").map(str::to_owned);
        }
        Ok(identity)
    }

    pub fn slug(&self) -> Option<&str> {
        self.repository
            .as_deref()
            .or(self.name.as_deref())
            .and_then(|name| name.rsplit('/').next())
    }

    pub fn validate_for(&self, model: &Self) -> Result<Compatibility> {
        if let Some(repository) = self.repository.as_deref() {
            let served = model.repository.as_deref()
                .context("GLP declares a base repository but the model has no verifiable repository metadata")?;
            if !repository.eq_ignore_ascii_case(served) {
                bail!("GLP base repository mismatch: vector={repository}, model={served}");
            }
        }
        if let Some(name) = self.name.as_deref() {
            let expected = name.rsplit('/').next().unwrap_or(name);
            let served = model
                .slug()
                .context("GLP declares a base name but the model has no base name metadata")?;
            if normalize_name(expected) != normalize_name(served) {
                bail!("GLP base model mismatch: vector={name}, model={served}");
            }
        }
        // Organization is descriptive GGUF provenance, not necessarily an HF
        // account handle. Compare it when supplied by both sides, and never
        // manufacture a repository ID from it.
        if let (Some(expected), Some(served)) = (&self.organization, &model.organization) {
            if normalize_name(expected) != normalize_name(served) {
                bail!("GLP base organization mismatch: vector={expected}, model={served}");
            }
        }
        if let Some(revision) = self.revision.as_deref() {
            if self.name.is_none() && self.repository.is_none() {
                bail!("GLP checkpoint revision has no accompanying base model identity");
            }
            let served = model.revision.as_deref()
                .context("GLP declares an HF commit but the model has no verifiable checkpoint revision metadata")?;
            if !revision.eq_ignore_ascii_case(served) {
                bail!("GLP base checkpoint mismatch: vector={revision}, model={served}");
            }
            return Ok(Compatibility::Checkpoint);
        }
        Ok(if self.name.is_some() || self.repository.is_some() {
            Compatibility::Named
        } else {
            Compatibility::Undeclared
        })
    }
}

fn normalize_name(name: &str) -> String {
    name.chars()
        .filter(|c| !c.is_whitespace() && *c != '-' && *c != '_')
        .flat_map(char::to_lowercase)
        .collect()
}

fn string_metadata<'a>(
    get: &impl Fn(&str) -> Option<&'a MetadataValue>,
    key: &str,
) -> Result<Option<String>> {
    match get(key) {
        None => Ok(None),
        Some(MetadataValue::String(value)) if value == "unknown" || value.trim().is_empty() => {
            Ok(None)
        }
        Some(MetadataValue::String(value)) => Ok(Some(value.clone())),
        Some(_) => bail!("{key} must be a string when supplied"),
    }
}

/// Validate local and downloaded vectors through the same loader boundary.
pub fn validate_glp_for_model(glp_path: &Path, model: &GgufFile) -> Result<Compatibility> {
    let vector = GgufFile::open(glp_path).context("open GLP checkpoint metadata")?;
    let status = CheckpointIdentity::from_gguf(&vector)?
        .validate_for(&CheckpointIdentity::from_model_gguf(model)?)?;
    if status != Compatibility::Checkpoint {
        tracing::warn!(
            ?status,
            "GLP does not declare a verified checkpoint revision; behavior requires validation"
        );
    }
    Ok(status)
}

#[cfg(test)]
#[path = "compatibility_tests.rs"]
mod tests;
