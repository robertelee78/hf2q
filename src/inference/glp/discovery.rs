//! Resolve GLP modifiers without resolving or downloading the model again.
//! Explicit local paths and canonical Hub references share loader validation.
//! Automatic discovery is bounded to the public Weightless naming convention;
//! ambiguous repositories/files and incomplete searches are startup errors.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use mlx_native::gguf::GgufFile;

use super::compatibility::{validate_glp_for_model, CheckpointIdentity};
use crate::input::hf_download::{self, HubApi};
use crate::input::hf_reference::{HfModelReference, ResolvedHfModelReference};
use crate::progress::ProgressReporter;

const DISCOVERY_AUTHOR: &str = "msuiche";
const MAX_DISCOVERY_REPOS: usize = 100;

#[derive(Debug)]
pub struct ResolvedGlp {
    pub path: PathBuf,
    /// Hub identity is absent for an explicitly selected local file.
    pub source_repo: Option<String>,
    pub source_revision: Option<String>,
    pub source_filename: Option<String>,
}

/// The model path is the file already selected by the serving resolver.
/// Repository IDs and Hugging Face tree/blob/resolve URLs use the canonical
/// parser, including embedded revisions and nested filenames.
pub fn resolve_glp(
    reference: &Path,
    model_path: &Path,
    _progress: &ProgressReporter,
) -> Result<ResolvedGlp> {
    // Resolve local files before constructing the Hub client: local modifiers
    // must continue working offline, without consulting Hub credentials.
    let mut hub = LiveHub { api: None };
    let resolved = resolve_with(reference, model_path, &mut hub)?;
    let model = GgufFile::open(model_path).context("open resolved model for GLP binding")?;
    validate_glp_for_model(&resolved.path, &model)?;
    tracing::info!(
        repository = ?resolved.source_repo,
        revision = ?resolved.source_revision,
        filename = ?resolved.source_filename,
        path = %resolved.path.display(),
        "resolved GLP artifact"
    );
    Ok(resolved)
}

struct RepositoryInventory {
    revision: String,
    filenames: Vec<String>,
}

/// Isolate network I/O so adversarial inventories and mutable branch responses
/// are exercised without live Hub access or a model/device allocation.
trait GlpHub {
    fn search(&mut self, author: &str, search: &str, limit: usize) -> Result<Vec<String>>;
    fn inventory(&mut self, reference: &HfModelReference) -> Result<RepositoryInventory>;
    fn download(&mut self, reference: &ResolvedHfModelReference, filename: &str)
        -> Result<PathBuf>;
}

struct LiveHub {
    api: Option<HubApi>,
}

impl LiveHub {
    fn api(&mut self) -> Result<&HubApi> {
        if self.api.is_none() {
            self.api = Some(hf_download::build_hub_api(
                &hf_download::resolve_hf_cache_dir(),
                true,
            )?);
        }
        Ok(self.api.as_ref().expect("initialized Hub API"))
    }
}

impl GlpHub for LiveHub {
    fn search(&mut self, author: &str, search: &str, limit: usize) -> Result<Vec<String>> {
        Ok(hf_download::search_hub_model_repos(
            self.api()?,
            author,
            search,
            limit,
        )?)
    }

    fn inventory(&mut self, reference: &HfModelReference) -> Result<RepositoryInventory> {
        let repo = hf_download::hub_model_repo(self.api()?, reference.repo_id());
        let info = repo
            .info()
            .revision(reference.requested_revision().unwrap_or("main"))
            .send()
            .with_context(|| format!("GLP repository inventory for {}", reference.repo_id()))?;
        Ok(RepositoryInventory {
            revision: info
                .sha
                .context("GLP repository inventory omitted its exact commit")?,
            filenames: info
                .siblings
                .context("GLP repository omitted its file inventory")?
                .into_iter()
                .map(|entry| entry.rfilename)
                .collect(),
        })
    }

    fn download(
        &mut self,
        reference: &ResolvedHfModelReference,
        filename: &str,
    ) -> Result<PathBuf> {
        let repo = hf_download::hub_model_repo(self.api()?, reference.repo_id());
        Ok(hf_download::download_file(
            &repo,
            reference.repo_id(),
            reference.revision(),
            filename,
            None,
        )?)
    }
}

fn resolve_with(reference: &Path, model_path: &Path, hub: &mut impl GlpHub) -> Result<ResolvedGlp> {
    if reference.is_file() {
        return Ok(ResolvedGlp {
            path: reference.to_owned(),
            source_repo: None,
            source_revision: None,
            source_filename: None,
        });
    }
    let value = reference
        .to_str()
        .context("GLP reference must be a UTF-8 path or Hub reference")?;
    if reference.exists()
        || reference.is_absolute()
        || value.starts_with('.')
        || (value.ends_with(".gguf") && !value.starts_with("https://"))
    {
        bail!(
            "GLP local file does not exist or is not a regular file: {}",
            reference.display()
        );
    }
    let parsed = if value == "auto" {
        let model = GgufFile::open(model_path).context("open resolved model for GLP discovery")?;
        let identity = CheckpointIdentity::from_model_gguf(&model)?;
        let slug = identity.slug().context("GLP auto-discovery needs a declared model identity; select a local file or Hub reference explicitly")?;
        let search = format!("{}-abliterated-cyber-GLP-", slug.replace(' ', "-"));
        let found = hub.search(DISCOVERY_AUTHOR, &search, MAX_DISCOVERY_REPOS + 1)?;
        let repo = unique_discovery_repo(&search, found)?;
        HfModelReference::parse(&repo, None)?
    } else {
        HfModelReference::parse(value, None)
            .context("--glp expects a local file, owner/repo, or https://huggingface.co/owner/repo/resolve/revision/file.gguf")?
    };
    resolve_hub(parsed, hub)
}

fn unique_discovery_repo(search: &str, found: Vec<String>) -> Result<String> {
    if found.len() > MAX_DISCOVERY_REPOS {
        bail!("GLP discovery exceeded its repository limit; select an explicit Hub reference");
    }
    let prefix = format!("{DISCOVERY_AUTHOR}/{search}");
    let mut candidates = std::collections::BTreeSet::new();
    for repo in found {
        // A search result is not selection authority. Keep the exact author,
        // base slug and published coverage/layer/dose convention; validate
        // the ID independently before treating it as a candidate.
        HfModelReference::parse(&repo, None)?;
        if repo
            .strip_prefix(&prefix)
            .is_some_and(weightless_artifact_suffix)
        {
            candidates.insert(repo);
        }
    }
    match candidates.len() {
        0 => bail!(
            "no GLP repository matches {prefix}<N>; select an explicit local file or Hub reference"
        ),
        1 => Ok(candidates.into_iter().next().expect("one candidate")),
        _ => bail!(
            "ambiguous GLP repositories: {}; select an explicit Hub reference",
            candidates.into_iter().collect::<Vec<_>>().join(", ")
        ),
    }
}

/// Weightless repositories use GLP-<coverage>, optionally followed by the
/// published -L<first>-<last>-a<dose> suffix. Coverage is a count, and the
/// layer/dose suffix is identity text, not authorization to override metadata.
fn weightless_artifact_suffix(tail: &str) -> bool {
    fn integer(value: &str) -> Option<u32> {
        (!value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()))
            .then(|| value.parse().ok())
            .flatten()
    }
    let (coverage, suffix) = match tail.split_once("-L") {
        Some((coverage, suffix)) => (coverage, Some(suffix)),
        None => (tail, None),
    };
    if !integer(coverage).is_some_and(|coverage| coverage > 0) {
        return false;
    }
    let Some(suffix) = suffix else {
        return true;
    };
    let Some((layers, dose)) = suffix.split_once("-a") else {
        return false;
    };
    let Some((first, last)) = layers.split_once('-') else {
        return false;
    };
    let (Some(first), Some(last)) = (integer(first), integer(last)) else {
        return false;
    };
    if first == 0 || last < first {
        return false;
    }
    match dose.split_once('.') {
        Some((whole, fraction)) => integer(whole).is_some() && integer(fraction).is_some(),
        None => integer(dose).is_some(),
    }
}

fn resolve_hub(reference: HfModelReference, hub: &mut impl GlpHub) -> Result<ResolvedGlp> {
    let inventory = hub.inventory(&reference)?;
    let files =
        hf_download::validate_repo_inventory(inventory.filenames.iter().map(String::as_str))?;
    if let Some(requested) = reference.requested_revision() {
        if requested.len() == 40
            && requested.bytes().all(|b| b.is_ascii_hexdigit())
            && !requested.eq_ignore_ascii_case(&inventory.revision)
        {
            bail!(
                "GLP Hub lookup returned a different commit from the explicitly requested revision"
            );
        }
    }
    // Seal the returned commit before any download. Branch movement between
    // inventory and transfer cannot silently select different tensor bytes.
    let resolved = reference.resolve(&inventory.revision)?;
    let candidates: Vec<_> = files
        .into_iter()
        .filter(|name| name.to_ascii_lowercase().ends_with(".gguf"))
        .collect();
    let filename = if let Some(filename) = resolved.filename() {
        if !candidates.iter().any(|candidate| candidate == filename) {
            bail!("explicit GLP filename {filename} is not a GGUF in the resolved repository inventory");
        }
        filename.to_owned()
    } else {
        match candidates.as_slice() {
            [filename] => filename.clone(),
            [] => bail!("no GLP GGUF in {}", resolved.repo_id()),
            _ => bail!(
                "ambiguous GLP files in {}: {}; use a full Hub file URL",
                resolved.repo_id(),
                candidates.join(", ")
            ),
        }
    };
    let path = hub.download(&resolved, &filename)?;
    Ok(ResolvedGlp {
        path,
        source_repo: Some(resolved.repo_id().to_owned()),
        source_revision: Some(resolved.revision().to_owned()),
        source_filename: Some(filename),
    })
}

#[cfg(test)]
#[path = "discovery_tests.rs"]
mod tests;
