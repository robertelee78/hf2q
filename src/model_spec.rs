//! Shared operator-facing model reference syntax and managed-model paths.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};

use crate::serve::auto_pipeline::looks_like_hf_repo_id;
use crate::serve::cache::slug_repo_id;
use crate::serve::quant_select::QuantType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RepositoryModelSpec {
    pub(crate) repository: String,
    pub(crate) quant: Option<QuantType>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ModelSpec {
    List,
    Path(PathBuf),
    Repository(RepositoryModelSpec),
}

/// Parse the common `serve` / `chat` model operand.
pub(crate) fn parse_model_spec(raw: &str) -> Result<ModelSpec> {
    if raw == "list" {
        return Ok(ModelSpec::List);
    }
    let path = Path::new(raw);
    if path.exists() || is_explicit_path(path) {
        return Ok(ModelSpec::Path(path.to_path_buf()));
    }
    parse_repository_spec(raw).map(ModelSpec::Repository)
}

/// Split an optional final quant suffix from a simple Hub repository id.
/// Canonical Hugging Face URLs are left intact for the existing URL parser.
pub(crate) fn split_repository_quant_suffix(raw: &str) -> (&str, Option<&str>) {
    match raw.rsplit_once(':') {
        Some((repository, suffix)) if looks_like_hf_repo_id(repository) => {
            (repository, Some(suffix))
        }
        _ => (raw, None),
    }
}

pub(crate) fn parse_repository_spec(raw: &str) -> Result<RepositoryModelSpec> {
    let (repository, suffix) = split_repository_quant_suffix(raw);
    if !looks_like_hf_repo_id(repository) {
        bail!(
            "model {raw:?} is neither an existing/explicit path nor a Hugging Face repository (expected owner/repository[:QUANT])"
        );
    }
    let quant = suffix
        .map(|value| QuantType::from_canonical_str(value).map_err(|error| anyhow!(error)))
        .transpose()?;
    Ok(RepositoryModelSpec {
        repository: repository.to_owned(),
        quant,
    })
}

pub(crate) fn is_explicit_path(path: &Path) -> bool {
    if path.is_absolute() {
        return true;
    }
    let rendered = path.as_os_str().to_string_lossy();
    matches!(rendered.as_ref(), "." | "..")
        || rendered.starts_with("./")
        || rendered.starts_with("../")
        || rendered.starts_with('~')
}

/// `${XDG_DATA_HOME:-$HOME/.local/share}/hf2q/models`.
pub(crate) fn managed_model_root() -> Result<PathBuf> {
    managed_model_root_from(
        std::env::var_os("XDG_DATA_HOME").map(PathBuf::from),
        std::env::var_os("HOME").map(PathBuf::from),
    )
}

fn managed_model_root_from(
    xdg_data_home: Option<PathBuf>,
    home: Option<PathBuf>,
) -> Result<PathBuf> {
    let data_home = xdg_data_home
        .filter(|path| path.is_absolute())
        .or_else(|| {
            home.filter(|path| path.is_absolute())
                .map(|path| path.join(".local/share"))
        })
        .context("cannot resolve managed model root: set absolute XDG_DATA_HOME or HOME")?;
    Ok(data_home.join("hf2q/models"))
}

pub(crate) fn managed_revision_dir(
    root: &Path,
    repository: &str,
    revision: &str,
) -> Result<PathBuf> {
    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("managed model revision must be an exact 40-hex commit, got {revision:?}");
    }
    Ok(root.join(slug_repo_id(repository)?).join(revision))
}

pub(crate) fn default_convert_output(
    root: &Path,
    repository: &str,
    revision: &str,
    quant_name: &str,
) -> Result<PathBuf> {
    let model_name = repository
        .rsplit_once('/')
        .map(|(_, name)| name)
        .filter(|name| !name.is_empty())
        .context("repository has no model name")?;
    Ok(
        managed_revision_dir(root, repository, revision)?.join(format!(
            "{model_name}-hf2q-{}.gguf",
            quant_name.to_ascii_lowercase()
        )),
    )
}

/// Existing directories are destinations; every other explicit path remains
/// an exact filename for backward compatibility.
pub(crate) fn resolve_output_path(
    explicit: Option<&Path>,
    default_path: PathBuf,
) -> Result<PathBuf> {
    let Some(explicit) = explicit else {
        return Ok(default_path);
    };
    if explicit.exists() && explicit.is_dir() {
        let filename = default_path
            .file_name()
            .context("default model output has no filename")?;
        Ok(explicit.join(filename))
    } else {
        Ok(explicit.to_path_buf())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

    #[test]
    fn repository_quant_suffix_is_case_insensitive_and_exact() {
        let parsed = parse_repository_spec("owner/model:Q8_0").unwrap();
        assert_eq!(parsed.repository, "owner/model");
        assert_eq!(parsed.quant, Some(QuantType::Q8_0));
        assert_eq!(
            parse_repository_spec("owner/model:q4_k_m").unwrap().quant,
            Some(QuantType::Q4_K_M)
        );
        assert!(parse_repository_spec("owner/model:maybe").is_err());
    }

    #[test]
    fn paths_and_list_are_not_reinterpreted_as_repositories() {
        assert_eq!(parse_model_spec("list").unwrap(), ModelSpec::List);
        assert_eq!(
            parse_model_spec("./missing.gguf").unwrap(),
            ModelSpec::Path(PathBuf::from("./missing.gguf"))
        );
        assert!(parse_model_spec("missing.gguf").is_err());
    }

    #[test]
    fn managed_root_honors_xdg_then_home() {
        assert_eq!(
            managed_model_root_from(Some(PathBuf::from("/data")), Some(PathBuf::from("/home/u")))
                .unwrap(),
            PathBuf::from("/data/hf2q/models")
        );
        assert_eq!(
            managed_model_root_from(None, Some(PathBuf::from("/home/u"))).unwrap(),
            PathBuf::from("/home/u/.local/share/hf2q/models")
        );
        assert!(managed_model_root_from(None, None).is_err());
    }

    #[test]
    fn default_conversion_path_is_revision_bound() {
        let root = tempfile::tempdir().unwrap();
        let default =
            default_convert_output(root.path(), "owner/My-Model", REVISION, "Q4_K_M").unwrap();
        assert_eq!(
            default,
            root.path().join(format!(
                "v2-6f776e65722f4d792d4d6f64656c/{REVISION}/My-Model-hf2q-q4_k_m.gguf"
            ))
        );
        let destination = tempfile::tempdir().unwrap();
        assert_eq!(
            resolve_output_path(Some(destination.path()), default.clone()).unwrap(),
            destination.path().join("My-Model-hf2q-q4_k_m.gguf")
        );
        assert_eq!(
            resolve_output_path(Some(Path::new("/tmp/exact.bin")), default).unwrap(),
            PathBuf::from("/tmp/exact.bin")
        );
    }
}
