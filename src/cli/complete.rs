//! Side-effect-free completion grammar and semantic candidates.
//!
//! Completion must stay a bounded parser-only path: no network, cache creation,
//! model loading, Metal initialization, or diagnostics are allowed here.

use std::ffi::OsStr;

use clap::CommandFactory as _;
use clap_complete::CompletionCandidate;

use super::Cli;

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
}
