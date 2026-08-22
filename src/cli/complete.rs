//! Side-effect-free completion grammar and semantic candidates.
//!
//! Completion must stay a bounded parser-only path: no network, cache creation,
//! model loading, Metal initialization, or diagnostics are allowed here.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use clap::CommandFactory as _;
use clap_complete::CompletionCandidate;

use super::Cli;

const MAX_MODEL_PATH_CANDIDATES: usize = 256;

#[derive(Clone, Copy)]
enum ModelPathKind {
    Decoder,
    Mmproj,
}

/// Build the public completion grammar. Clap's `hide = true` is a help policy,
/// not a reliable completion boundary: AOT generators traverse hidden nodes,
/// and the dynamic engine can retain a hidden-only prefix match. Recursively
/// omitting hidden nodes makes the boundary structural.
pub(crate) fn public_completion_command() -> clap::Command {
    project_public(&Cli::command())
}

fn project_public(source: &clap::Command) -> clap::Command {
    let mut out = clap::Command::new(source.get_name().to_owned())
        .display_order(source.get_display_order())
        .disable_help_flag(source.is_disable_help_flag_set())
        .disable_help_subcommand(source.is_disable_help_subcommand_set())
        .disable_version_flag(source.is_disable_version_flag_set());

    if let Some(value) = source.get_version() {
        out = out.version(value.to_owned());
    }
    if let Some(value) = source.get_long_version() {
        out = out.long_version(value.to_owned());
    }
    if let Some(value) = source.get_about() {
        out = out.about(value.clone());
    }
    if let Some(value) = source.get_long_about() {
        out = out.long_about(value.clone());
    }
    if let Some(value) = source.get_subcommand_help_heading() {
        out = out.subcommand_help_heading(value.to_owned());
    }
    if let Some(value) = source.get_subcommand_value_name() {
        out = out.subcommand_value_name(value.to_owned());
    }
    if let Some(value) = source.get_short_flag() {
        out = out.short_flag(value);
    }
    if let Some(value) = source.get_long_flag() {
        out = out.long_flag(value.to_owned());
    }

    out = out.visible_aliases(source.get_visible_aliases().map(str::to_owned));
    out = out.visible_short_flag_aliases(source.get_visible_short_flag_aliases());
    out = out.visible_long_flag_aliases(source.get_visible_long_flag_aliases().map(str::to_owned));
    out = out.args(
        source
            .get_arguments()
            .filter(|argument| !argument.is_hide_set())
            .cloned(),
    );
    out.subcommands(
        source
            .get_subcommands()
            .filter(|subcommand| !subcommand.is_hide_set())
            .map(project_public),
    )
}

pub(crate) fn quant_names(current: &OsStr) -> Vec<CompletionCandidate> {
    prefixed_candidates(
        current,
        crate::convert::quant_selector::QuantSelector::COMPLETION_NAMES
            .iter()
            .copied(),
    )
}

pub(crate) fn architecture_names(current: &OsStr) -> Vec<CompletionCandidate> {
    prefixed_candidates(
        current,
        crate::arch::ArchRegistry::global()
            .known_arches()
            .into_iter(),
    )
}

/// Complete `serve --model` from hf2q's managed model directory by default.
/// An explicit path (`/`, `./`, `../`, `~/`, or any value containing a path
/// separator) remains anchored to the caller's filesystem instead.
pub(crate) fn serve_model_paths(current: &OsStr) -> Vec<CompletionCandidate> {
    model_paths(current, ModelPathKind::Decoder)
}

/// Complete `serve --mmproj` with the same root policy as `--model`, while
/// limiting final file candidates to conventionally named projector GGUFs.
pub(crate) fn serve_mmproj_paths(current: &OsStr) -> Vec<CompletionCandidate> {
    model_paths(current, ModelPathKind::Mmproj)
}

fn model_paths(current: &OsStr, kind: ModelPathKind) -> Vec<CompletionCandidate> {
    if is_bare_name(current) {
        if let Some(root) = canonical_model_root() {
            let candidates = complete_from_root(current, &root, &root, kind);
            if !candidates.is_empty() {
                return candidates;
            }
        }
    }

    complete_explicit_path(current, kind)
}

fn canonical_model_root() -> Option<PathBuf> {
    let data_home = std::env::var_os("XDG_DATA_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| {
            std::env::var_os("HOME")
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
                .filter(|path| path.is_absolute())
                .map(|home| home.join(".local/share"))
        })?;
    Some(data_home.join("hf2q/models"))
}

fn is_bare_name(current: &OsStr) -> bool {
    current != OsStr::new("~")
        && Path::new(current).components().count() <= 1
        && !current
            .as_encoded_bytes()
            .iter()
            .any(|byte| std::path::is_separator(*byte as char))
}

fn complete_explicit_path(current: &OsStr, kind: ModelPathKind) -> Vec<CompletionCandidate> {
    let current_path = Path::new(current);
    let (typed_parent, file_prefix) = if current == OsStr::new("~") {
        (current_path, OsStr::new(""))
    } else {
        split_file_name(current_path)
    };

    let search_root = if typed_parent.is_absolute() {
        typed_parent.to_path_buf()
    } else if typed_parent.iter().next() == Some(OsStr::new("~")) {
        let Some(home) = std::env::var_os("HOME")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
        else {
            return Vec::new();
        };
        home.join(typed_parent.strip_prefix("~").unwrap_or(typed_parent))
    } else {
        let Some(cwd) = std::env::current_dir().ok() else {
            return Vec::new();
        };
        cwd.join(typed_parent)
    };

    complete_from_root(file_prefix, &search_root, typed_parent, kind)
}

fn complete_from_root(
    file_prefix: &OsStr,
    search_root: &Path,
    output_parent: &Path,
    kind: ModelPathKind,
) -> Vec<CompletionCandidate> {
    let Ok(entries) = std::fs::read_dir(search_root) else {
        return Vec::new();
    };

    let mut candidates = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name();
            if !name
                .as_encoded_bytes()
                .starts_with(file_prefix.as_encoded_bytes())
            {
                return None;
            }

            let path = entry.path();
            let mut suggestion = output_parent.join(&name);
            if path.is_dir() {
                suggestion.push("");
            } else if !kind.wants_file(&path) {
                return None;
            }

            Some(
                CompletionCandidate::new(suggestion.into_os_string())
                    .hide(name.as_encoded_bytes().starts_with(b".")),
            )
        })
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.truncate(MAX_MODEL_PATH_CANDIDATES);
    candidates
}

impl ModelPathKind {
    fn wants_file(self, path: &Path) -> bool {
        if !path.is_file()
            || !path
                .extension()
                .and_then(OsStr::to_str)
                .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        {
            return false;
        }

        let is_mmproj = path
            .file_name()
            .and_then(OsStr::to_str)
            .is_some_and(|name| name.to_ascii_lowercase().contains("mmproj"));
        match self {
            Self::Decoder => !is_mmproj,
            Self::Mmproj => is_mmproj,
        }
    }
}

fn split_file_name(path: &Path) -> (&Path, &OsStr) {
    if path_has_name(path) {
        (
            path.parent().unwrap_or_else(|| Path::new("")),
            path.file_name().expect("path with a name"),
        )
    } else {
        (path, OsStr::new(""))
    }
}

fn path_has_name(path: &Path) -> bool {
    path.as_os_str()
        .as_encoded_bytes()
        .last()
        .is_some_and(|byte| !std::path::is_separator(*byte as char))
        && path.file_name().is_some()
}

fn prefixed_candidates<'a>(
    current: &OsStr,
    values: impl IntoIterator<Item = &'a str>,
) -> Vec<CompletionCandidate> {
    let Some(prefix) = current.to_str() else {
        return Vec::new();
    };
    values
        .into_iter()
        .filter(|value| value.starts_with(prefix))
        .take(256)
        .map(CompletionCandidate::new)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_projection_excludes_every_hidden_surface() {
        let projected = public_completion_command();
        projected.clone().debug_assert();
        let rendered = format!("{projected:#?}");
        for hidden in [
            "__standalone-install",
            "__fetch-hub-gguf",
            "__catalog-hub-gguf",
            "__verify-local-gguf",
            "source-teacher",
            "source-teacher-reference",
            "source-teacher-acceptance-verify",
            "chat_parent_lifeline_fd",
        ] {
            assert!(
                !rendered.contains(hidden),
                "hidden surface leaked: {hidden}"
            );
        }
    }

    #[test]
    fn quant_candidates_are_exactly_parseable_shipped_names() {
        let values = quant_names(OsStr::new(""));
        assert_eq!(
            values.len(),
            crate::convert::quant_selector::QuantSelector::COMPLETION_NAMES.len()
        );
        for candidate in values {
            let value = candidate.get_value().to_str().expect("ASCII quant");
            crate::convert::quant_selector::QuantSelector::from_name(value)
                .unwrap_or_else(|error| panic!("completion advertised {value:?}: {error}"));
        }
        for reserved in ["dwq", "apex", "apex-custom", "tq1_0", "tq2_0"] {
            assert!(
                !crate::convert::quant_selector::QuantSelector::COMPLETION_NAMES
                    .contains(&reserved)
            );
        }
    }

    #[test]
    fn model_path_candidates_are_globally_sorted_before_truncation() {
        let root = tempfile::tempdir().expect("model root");
        for index in (0..1_100).rev() {
            std::fs::create_dir(root.path().join(format!("model-{index:04}")))
                .expect("model directory");
        }

        let candidates = complete_from_root(
            OsStr::new("model-"),
            root.path(),
            root.path(),
            ModelPathKind::Decoder,
        );
        assert_eq!(candidates.len(), MAX_MODEL_PATH_CANDIDATES);
        assert_eq!(
            candidates.first().unwrap().get_value(),
            root.path().join("model-0000/").as_os_str()
        );
        assert_eq!(
            candidates.last().unwrap().get_value(),
            root.path().join("model-0255/").as_os_str()
        );
    }
}
