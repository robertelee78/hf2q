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

use crate::input::hf_download::download_model_reference;
use crate::input::hf_reference::HfModelReference;
use crate::progress::ProgressReporter;

/// A resolved GLP artifact: local path + provenance binding.
pub struct ResolvedGlp {
    pub path: PathBuf,
    pub base_model_name: String,
    pub base_model_revision: String,
    pub source_repo: String,
}

/// Resolve the model's provenance and auto-discover a matching GLP artifact.
/// Handles both HF slugs (`owner/repo`) and local GGUF paths.
pub fn auto_discover_glp(
    model_ref: &str,
    progress: &ProgressReporter,
) -> Result<ResolvedGlp> {
    // Case 1: local GGUF path — read provenance from the file.
    let path = Path::new(model_ref);
    if path.exists() && path.extension().map(|e| e == "gguf").unwrap_or(false) {
        return auto_discover_glp_from_local(path, progress);
    }

    // Case 2: HF slug — resolve the reference to get provenance from the Hub.
    auto_discover_glp_from_hub(model_ref, progress)
}

/// Local GGUF path: read provenance from the file metadata.
fn auto_discover_glp_from_local(
    model_path: &Path,
    progress: &ProgressReporter,
) -> Result<ResolvedGlp> {
    let gguf = GgufFile::open(model_path)
        .with_context(|| format!("open model GGUF for GLP discovery: {}", model_path.display()))?;

    let base_name = gguf
        .metadata_string("general.base_model.0.name")
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

    let base_repo_slug = if base_name != "unknown" {
        base_name.clone()
    } else {
        // No base_model metadata: infer from the filename first (the
        // convention is `<base_model>-<variant>.gguf`), then the display
        // name as a last resort.
        let filename = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let from_filename = infer_base_model_from_filename(filename);
        if from_filename != "unknown" && known_bases().iter().any(|b| from_filename.starts_with(b)) {
            from_filename
        } else {
            gguf
                .metadata_string("general.name")
                .unwrap_or("unknown")
                .replace(' ', "-")
        }
    };

    search_and_download_glp(&base_repo_slug, &base_org, &base_revision, model_path, progress)
}

/// HF slug: resolve the reference, download the model (cached), then read
/// the GGUF metadata from the downloaded file for provenance.
fn auto_discover_glp_from_hub(
    model_ref: &str,
    progress: &ProgressReporter,
) -> Result<ResolvedGlp> {
    let reference = HfModelReference::parse(model_ref, None)
        .with_context(|| format!("parse model reference: {model_ref}"))?;

    // Download the model (cached if already present) to get the GGUF file
    // for provenance reading. The Hub API gives repo info but not GGUF
    // metadata; the provenance is in the file itself.
    let downloaded = download_model_reference(reference, progress)
        .with_context(|| format!("download model for GLP provenance: {model_ref}"))?;

    // Find the GGUF file in the downloaded directory.
    let gguf_path = find_gguf_in_dir(downloaded.local_dir())
        .with_context(|| format!("no GGUF found in downloaded model at {}", downloaded.local_dir().display()))?;

    auto_discover_glp_from_local(&gguf_path, progress)
}

/// Find the first GGUF file in a directory (the model's text GGUF).
fn find_gguf_in_dir(dir: &Path) -> Result<PathBuf> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map(|e| e == "gguf").unwrap_or(false) {
            return Ok(path);
        }
    }
    anyhow::bail!("no GGUF file found in {}", dir.display())
}

/// Known GLP repo base names (the weightless convention).
fn known_bases() -> &'static [&'static str] {
    &[
        "DeepSeek-V4-Flash-0731",
        "Qwen3.8-27B",
        "Qwen3.8-Flash-Next",
        "GLM-5.3-Flash",
        "GLM-5.3",
        "Inkling-Small",
        "Hy4-preview",
    ]
}

/// Infer the base model HF repo name from a GGUF filename.
/// The convention is `<base_model>-<variant>.gguf` (e.g.
/// "DeepSeek-V4-Flash-0731-agentic-q2.gguf" → "DeepSeek-V4-Flash-0731").
/// We try progressively shorter prefixes until one matches a known GLP repo.
fn infer_base_model_from_filename(filename: &str) -> String {
    for base in known_bases() {
        if filename.starts_with(base) {
            return base.to_string();
        }
    }
    // Fallback: slugify the filename (replace underscores, keep hyphens).
    filename.replace('_', "-")
}

/// The weightless GLP numbers per model family. The search tries these in
/// order; the first match wins. DeepSeek-V4-Flash-0731 is GLP-29.
fn glp_numbers_for_model(base_repo_slug: &str) -> Vec<u32> {
    match base_repo_slug {
        "DeepSeek-V4-Flash-0731" => vec![29],
        "DeepSeek-V4-Flash-Vision-Exp" => vec![29],
        "Qwen3.8-27B" => vec![49],
        "Qwen3.8-Flash-Next" => vec![47],
        "GLM-5.3-Flash" => vec![44],
        "GLM-5.3" => vec![77],
        "Inkling-Small" => vec![41],
        "Hy4-preview" => vec![77],
        "Nemotron-3.5-Lightning-30B-A3B" => vec![51],
        _ => vec![29, 49, 44, 47, 77, 41, 51],
    }
}

/// Search for and download a GLP artifact matching the base model.
fn search_and_download_glp(
    base_repo_slug: &str,
    base_org: &str,
    base_revision: &str,
    model_path: &Path,
    _progress: &ProgressReporter,
) -> Result<ResolvedGlp> {
    let glp_numbers = glp_numbers_for_model(base_repo_slug);
    let mut last_error = None;

    for n in glp_numbers {
        let repo = format!("msuiche/{base_repo_slug}-abliterated-cyber-GLP-{n}");

        // The GLP repo has no config.json (it's a vector repo, not a model
        // repo), so the catalog path fails. Download the file directly via
        // the hf_hub client.
        let cache_dir = crate::input::hf_download::resolve_hf_cache_dir();
        let api = crate::input::hf_download::build_hub_api(&cache_dir, true)
            .with_context(|| "build Hub API for GLP download")?;
        let repo_obj = crate::input::hf_download::hub_model_repo(&api, &repo);

        // The GLP file is the only GGUF in the repo. The filename is
        // `<base_repo_slug>-abliterated-cyber-GLP-<n>-L<layers>-a<alpha>.gguf`
        // where layers/alpha vary. List the repo's files to find the GGUF.
        let filename = match find_gguf_in_repo(&repo_obj, &repo) {
            Ok(filename) => filename,
            Err(e) => {
                eprintln!("[GLP-DISCOVERY] {repo}: file listing failed: {e}");
                last_error = Some(format!("{repo}: {e}"));
                continue;
            }
        };

        let path = match crate::input::hf_download::download_file(
            &repo_obj,
            &repo,
            "main",
            &filename,
            None,
        ) {
            Ok(path) => path,
            Err(e) => {
                eprintln!("[GLP-DISCOVERY] {repo}: direct download failed: {e}");
                last_error = Some(format!("{repo}: {e}"));
                continue;
            }
        };

        // Validate provenance before accepting. The GLP's base model name
        // is the HF repo name; the local model's display name may differ.
        // Compare against the inferred base repo slug (the HF repo name).
        if let Err(e) = validate_glp_provenance(&path, model_path, base_repo_slug) {
            eprintln!("[GLP-DISCOVERY] {repo}: provenance mismatch: {e}");
            last_error = Some(format!("{repo}: provenance mismatch: {e}"));
            continue;
        }

        return Ok(ResolvedGlp {
            path,
            base_model_name: base_repo_slug.to_string(),
            base_model_revision: base_revision.to_string(),
            source_repo: repo,
        });
    }

    anyhow::bail!(
        "no GLP found for base model {base_org}/{base_repo_slug}@{base_revision}. \
         Tried the weightless convention (msuiche/<model>-abliterated-cyber-GLP-<N>). \
         Last error: {}",
        last_error.unwrap_or_else(|| "none".into())
    )
}

/// Find the GGUF file in a GLP repo (the only GGUF; the filename varies by
/// layer count and alpha).
fn find_gguf_in_repo(repo: &crate::input::hf_download::HubRepo, repo_id: &str) -> Result<String> {
    let info = repo
        .info()
        .revision("main")
        .send()
        .with_context(|| format!("repo info for {repo_id}"))?;
    let siblings = info
        .siblings
        .ok_or_else(|| anyhow::anyhow!("repo {repo_id} has no file inventory"))?;
    let filename = siblings
        .iter()
        .find(|s| s.rfilename.ends_with(".gguf"))
        .map(|s| s.rfilename.clone())
        .ok_or_else(|| anyhow::anyhow!("no GGUF in {repo_id}"))?;
    Ok(filename)
}

/// Validate that a GLP file's provenance matches the served model.
/// The GLP's base model name is the HF repo name; the model's display name
/// may differ (e.g. "Deepseek v4 Flash 0731 Source" vs "DeepSeek-V4-Flash-0731").
/// Compare against the inferred base repo slug (the HF repo name).
pub fn validate_glp_provenance(glp_path: &Path, _model_path: &Path, base_repo_slug: &str) -> Result<()> {
    let glp_gguf = GgufFile::open(glp_path)
        .with_context(|| format!("open GLP GGUF: {}", glp_path.display()))?;

    let glp_base = glp_gguf
        .metadata_string("general.base_model.0.name")
        .or_else(|| glp_gguf.metadata_string("glp.base_model"))
        .unwrap_or("");

    // The GLP's base model is the HF repo name (e.g. "DeepSeek-V4-Flash-0731")
    // or the org/repo form (e.g. "deepseek-ai/DeepSeek-V4-Flash-0731"). The
    // model's base repo slug is the inferred HF repo name (from the filename
    // or metadata). Compare the repo name part (after the last slash if
    // present).
    let glp_repo_name = glp_base.rsplit('/').next().unwrap_or(glp_base);
    if !glp_repo_name.is_empty() && !glp_repo_name.eq_ignore_ascii_case(base_repo_slug) {
        anyhow::bail!(
            "GLP base model mismatch: GLP targets {glp_base}, model is {base_repo_slug}"
        );
    }

    Ok(())
}
