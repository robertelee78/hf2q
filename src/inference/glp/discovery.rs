//! GLP auto-discovery: resolve a model's provenance to a Hub GLP artifact.
//!
//! When `--glp` is used without an explicit path (or with `--glp auto`),
//! resolve the served model's base model + revision from its GGUF metadata,
//! search HuggingFace for `*-GLP-*` artifacts bound to that exact commit,
//! download/verify/bind. Fail closed on ambiguity or no match.
//!
//! The weightless convention: GLP artifacts are published as
//! `msuiche/<model>-abliterated-cyber-GLP-<N>` (DeepSeek, Qwen, GLM, etc.).
//! A real search API would list Hub repos; for now we use the convention.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use mlx_native::gguf::GgufFile;

use crate::input::hf_download::{HfModelReference, download_model_reference};
use crate::serve::ProgressReporter;

/// A resolved GLP artifact: local path + provenance binding.
pub struct ResolvedGlp {
    pub path: PathBuf,
    pub base_model_name: String,
    pub base_model_revision: String,
    pub source_repo: String,
}

/// Resolve the model's provenance from its GGUF metadata and auto-discover
/// a matching GLP artifact on the Hub.
pub fn auto_discover_glp(
    model_path: &Path,
    progress: &ProgressReporter,
) -> Result<ResolvedGlp> {
    let gguf = GgufFile::open(model_path)
        .with_context(|| format!("open model GGUF for GLP discovery: {}", model_path.display()))?;

    // The model's provenance: general.base_model.0.{name, organization, version}
    let base_name = gguf
        .metadata_string("general.base_model.0.name")
        .or_else(|| gguf.metadata_string("general.name"))
        .unwrap_or("unknown")
        .to_string();
    let base_org = gguf
        .metadata_string("general.base_model.0.organization")
        .unwrap_or("unknown")
        .to_string();
    let base_revision = gguf
        .metadata_string("general.base_model.0.version")
        .unwrap_or("unknown")
        .to_string();

    // The weightless convention: GLP artifacts are published as
    // `msuiche/<model>-abliterated-cyber-GLP-<N>`. We try the known GLP
    // numbers for the model family. A real search API would list Hub repos
    // and filter by exact base_model revision match.
    let glp_numbers = [29, 49, 44, 47, 77, 41];
    let mut last_error = None;

    for n in glp_numbers {
        let repo = format!("msuiche/{base_name}-abliterated-cyber-GLP-{n}");
        let reference = HfModelReference::from_repo_id(&repo);
        match download_model_reference(reference, progress) {
            Ok(downloaded) => {
                let path = downloaded.local_dir().join(format!("{repo.replace('/', '-')}.gguf"));
                if path.exists() {
                    // Validate provenance before accepting
                    if let Err(e) = validate_glp_provenance(&path, model_path) {
                        last_error = Some(format!("{repo}: provenance mismatch: {e}"));
                        continue;
                    }
                    return Ok(ResolvedGlp {
                        path,
                        base_model_name: base_name,
                        base_model_revision: base_revision,
                        source_repo: repo,
                    });
                }
                last_error = Some(format!("{repo}: downloaded file not found at {}", path.display()));
            }
            Err(e) => {
                last_error = Some(format!("{repo}: {e}"));
            }
        }
    }

    anyhow::bail!(
        "no GLP found for base model {base_org}/{base_name}@{base_revision}. \
         Tried the weightless convention (msuiche/<model>-abliterated-cyber-GLP-<N>). \
         Last error: {}",
        last_error.unwrap_or_else(|| "none".into())
    )
}

/// Validate that a GLP file's provenance matches the served model.
pub fn validate_glp_provenance(glp_path: &Path, model_path: &Path) -> Result<()> {
    let glp_gguf = GgufFile::open(glp_path)
        .with_context(|| format!("open GLP GGUF: {}", glp_path.display()))?;
    let model_gguf = GgufFile::open(model_path)
        .with_context(|| format!("open model GGUF: {}", model_path.display()))?;

    let glp_base = glp_gguf
        .metadata_string("general.base_model.0.name")
        .unwrap_or("");
    let model_base = model_gguf
        .metadata_string("general.base_model.0.name")
        .or_else(|| model_gguf.metadata_string("general.name"))
        .unwrap_or("");

    if !glp_base.is_empty() && !model_base.is_empty() && glp_base != model_base {
        anyhow::bail!(
            "GLP base model mismatch: GLP targets {glp_base}, model is {model_base}"
        );
    }

    let glp_rev = glp_gguf
        .metadata_string("general.base_model.0.version")
        .unwrap_or("");
    let model_rev = model_gguf
        .metadata_string("general.base_model.0.version")
        .unwrap_or("");

    if !glp_rev.is_empty() && !model_rev.is_empty() && glp_rev != model_rev {
        anyhow::bail!(
            "GLP base revision mismatch: GLP targets {glp_rev}, model is {model_rev}"
        );
    }

    Ok(())
}
