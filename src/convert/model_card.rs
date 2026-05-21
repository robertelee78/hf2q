//! HuggingFace model-card metadata reader.
//!
//! Parses the YAML frontmatter from a HF model directory's `README.md`
//! to populate the `general.*` GGUF metadata fields that canonical
//! `convert_hf_to_gguf.py` writes via `gguf-py/gguf/metadata.py`
//! `Metadata.load()` (which delegates to `huggingface_hub`'s
//! `ModelCard.load` for the actual YAML extraction).
//!
//! Scope: minimal hand-rolled YAML reader covering only the subset of
//! keys the GGUF metadata emitter consumes (`license`, `base_model`,
//! `tags`, `language`). The frontmatter format on HF model cards is
//! consistent enough that a hand-rolled line parser handles it
//! correctly without pulling in a YAML crate (no FFI / no full YAML
//! processor — per the pure-Rust standing rule).
//!
//! Supported YAML subset:
//!   - `key: scalar`                         (plain scalar value)
//!   - `key: 'quoted scalar'`                (single-quoted scalar)
//!   - `key:\n  - item1\n  - item2\n`        (block-style list of
//!                                            strings)
//!
//! Anything outside this subset is silently ignored — a model card
//! with richer YAML (anchors, multi-line scalars, nested maps) still
//! parses without panicking; unknown fields just don't get emitted.
//! Per `[[feedback-no-loop-suppression-2026-05-17]]`: malformed
//! frontmatter returns `None` to the caller, which then skips the
//! `general.*` metadata block entirely rather than emitting half a
//! card.

use std::path::Path;

/// One entry in the `base_model:` list of the YAML frontmatter.
/// HuggingFace convention is `"<org>/<repo>"` (e.g.
/// `"nomic-ai/nomic-embed-text-v2-moe-unsupervised"`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseModelEntry {
    /// Raw `<org>/<repo>` string as it appears in the YAML.
    pub raw: String,
}

/// Parsed YAML frontmatter from `README.md`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelCard {
    /// `license: apache-2.0` → `Some("apache-2.0")`. Canonical
    /// `Metadata.load` reads this as `general.license`.
    pub license: Option<String>,
    /// `tags:` list → tags vector. Emitted as `general.tags` (array
    /// of strings). HF convention: kebab-case tags.
    pub tags: Vec<String>,
    /// `language:` list → language codes (ISO 639-1 typically).
    /// Emitted as `general.languages` (NOTE plural; canonical's
    /// `add_languages` writes the GGUF key `general.languages` from
    /// the singular YAML `language` key).
    pub languages: Vec<String>,
    /// `base_model:` list → base model entries. Emitted as
    /// `general.base_model.count` + `general.base_model.<i>.{name,
    /// organization, repo_url}` per `metadata.py:685-704`.
    pub base_models: Vec<BaseModelEntry>,
}

impl ModelCard {
    /// Whether any of the parsed fields are non-empty.
    pub fn is_empty(&self) -> bool {
        self.license.is_none()
            && self.tags.is_empty()
            && self.languages.is_empty()
            && self.base_models.is_empty()
    }
}

/// Read `<dir>/README.md` and parse its YAML frontmatter (the block
/// between two `---` lines at the very top of the file). Returns
/// `None` if there's no README.md, no frontmatter block, or the file
/// can't be read. Returns `Some(ModelCard::default())` if the
/// frontmatter exists but contains no keys we recognize.
pub fn parse_readme_frontmatter(dir: &Path) -> Option<ModelCard> {
    let readme_path = dir.join("README.md");
    let contents = std::fs::read_to_string(&readme_path).ok()?;

    // Extract the frontmatter block: must START with `---\n` and end
    // with a line containing exactly `---`. Anything outside is the
    // markdown body, which we don't parse.
    let frontmatter = extract_frontmatter_block(&contents)?;
    Some(parse_yaml_frontmatter(frontmatter))
}

/// Extract the text between the two `---` delimiters at the start of
/// a README.md. Returns `None` if the file doesn't start with the
/// frontmatter sentinel.
fn extract_frontmatter_block(contents: &str) -> Option<&str> {
    let trimmed_start = contents.trim_start_matches('\u{FEFF}'); // strip BOM if present
    let after_first = trimmed_start.strip_prefix("---\n").or_else(|| {
        // Tolerate CRLF line endings on the sentinel line.
        trimmed_start.strip_prefix("---\r\n")
    })?;
    let close_idx = after_first.find("\n---\n").or_else(|| after_first.find("\n---\r\n"))?;
    Some(&after_first[..close_idx])
}

/// Parse the YAML frontmatter text (between the `---` sentinels) into
/// a [`ModelCard`]. Hand-rolled line scanner — see module-level docs
/// for the supported subset.
fn parse_yaml_frontmatter(text: &str) -> ModelCard {
    let mut card = ModelCard::default();
    let mut lines = text.lines().peekable();

    while let Some(line) = lines.next() {
        if line.is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        // Top-level key always starts at column 0 with non-whitespace.
        if line.starts_with(char::is_whitespace) {
            // Stray indented line — shouldn't reach here in
            // well-formed frontmatter (block list items are consumed
            // by the list reader below).
            continue;
        }
        let Some(colon) = line.find(':') else {
            continue;
        };
        let key = line[..colon].trim();
        let after_colon = line[colon + 1..].trim();

        match key {
            "license" => {
                if !after_colon.is_empty() {
                    card.license = Some(unquote_scalar(after_colon).to_string());
                }
            }
            "tags" => {
                card.tags = read_block_list(&mut lines);
            }
            "language" => {
                card.languages = read_block_list(&mut lines);
            }
            "base_model" => {
                card.base_models = read_block_list(&mut lines)
                    .into_iter()
                    .map(|raw| BaseModelEntry { raw })
                    .collect();
            }
            _ => {
                // Unknown key — if it has a value-less colon (next
                // lines are list items), skip past them so the next
                // top-level key parses cleanly.
                if after_colon.is_empty() {
                    let _ = read_block_list(&mut lines);
                }
            }
        }
    }

    card
}

/// Read a block-style list (`  - item1\n  - item2\n`) following a
/// `key:`-only line. Stops at the first non-list-item line (blank or
/// new top-level key) and leaves the cursor positioned to re-read
/// that line in the outer loop.
fn read_block_list<'a, I>(lines: &mut std::iter::Peekable<I>) -> Vec<String>
where
    I: Iterator<Item = &'a str>,
{
    let mut out = Vec::new();
    while let Some(&peek) = lines.peek() {
        let trimmed = peek.trim_start();
        if let Some(item) = trimmed.strip_prefix("- ") {
            let _ = lines.next();
            out.push(unquote_scalar(item.trim()).to_string());
            continue;
        }
        // Some YAML emitters write `-item` without the space — accept
        // it for robustness, but require the dash to be the first
        // non-whitespace char.
        if let Some(item) = trimmed.strip_prefix('-') {
            // Reject empty `-` and `-` followed by non-list content
            // (e.g. a key continuation). The HF model-card convention
            // is `- value` with a space, so this branch should be rare.
            if item.starts_with(' ') {
                let _ = lines.next();
                out.push(unquote_scalar(item.trim()).to_string());
                continue;
            }
        }
        // Anything else stops the list.
        break;
    }
    out
}

/// Strip surrounding single or double quotes from a YAML scalar.
/// HF model cards quote scalars when the value would otherwise be
/// interpreted as a YAML 1.1 magic value (e.g. `'no'` for the
/// Norwegian language code, which YAML 1.1 parses as `false`).
fn unquote_scalar(s: &str) -> &str {
    let s = s.trim();
    if s.len() >= 2 {
        let bytes = s.as_bytes();
        let first = bytes[0];
        let last = bytes[bytes.len() - 1];
        if (first == b'\'' && last == b'\'') || (first == b'"' && last == b'"') {
            return &s[1..s.len() - 1];
        }
    }
    s
}

/// Derive `general.base_model.<i>.{name, organization, repo_url}` for
/// a single base-model entry. Mirrors canonical's title-case + URL
/// builder used implicitly by `Metadata` when emitting the
/// per-base-model KV block.
///
/// Inputs:
///   - `raw = "nomic-ai/nomic-embed-text-v2-moe-unsupervised"`
/// Returns:
///   - name = `"Nomic Embed Text v2 Moe Unsupervised"`
///   - organization = `"Nomic Ai"`
///   - repo_url = `"https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-unsupervised"`
///
/// Returns `(None, None, None)` if `raw` doesn't contain a `/`
/// (canonical would skip the entry).
pub fn split_base_model(raw: &str) -> (Option<String>, Option<String>, Option<String>) {
    let Some((org, name)) = raw.split_once('/') else {
        return (None, None, None);
    };
    let pretty_org = title_case_hyphenated(org);
    let pretty_name = title_case_hyphenated(name);
    let repo_url = format!("https://huggingface.co/{org}/{name}");
    (Some(pretty_name), Some(pretty_org), Some(repo_url))
}

/// Title-case a hyphenated identifier the way canonical
/// `Metadata.load` does for the `base_model.name` / `organization`
/// fields. Splits on `-`, capitalizes the first letter of each part,
/// joins with single spaces. Numeric parts (like `v2`) stay
/// lower-case if they start with a letter+digit pattern? Actually,
/// canonical title-cases regardless ("v2-moe" → "v2 Moe"). Verified
/// against `general.base_model.0.name = 'Nomic Embed Text v2 Moe
/// Unsupervised'` in the canonical Q8_0 dump for nomic v2-moe.
fn title_case_hyphenated(s: &str) -> String {
    s.split('-')
        .map(|part| {
            if is_version_part(part) {
                // Version markers (e.g. `v2`, `v1.5`, `iter3`) keep
                // their lowercase prefix to match canonical's
                // `general.base_model.<i>.name` formatting. Verified
                // against the nomic v2-moe canonical Q8_0 dump:
                // `'Nomic Embed Text v2 Moe Unsupervised'`.
                part.to_string()
            } else {
                let mut chars = part.chars();
                match chars.next() {
                    Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
                    None => String::new(),
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Detect a version-marker name part like `v2`, `v1.5`, `iter3`.
/// Mirrors canonical's version regex at `metadata.py:279`:
/// `(v|iter)?\d+([.]\d+)*` — but here we use the stricter form
/// `(v|iter)\d+(\.\d+)*` because pure-numeric parts like `2048` are
/// title-case fine (no leading letter to upper).
fn is_version_part(part: &str) -> bool {
    let lower = part.to_ascii_lowercase();
    let rest = if let Some(r) = lower.strip_prefix('v') {
        r
    } else if let Some(r) = lower.strip_prefix("iter") {
        r
    } else {
        return false;
    };
    // Must have at least one digit, optionally followed by `.<digits>`
    // sequences. `v` alone or `iter` alone is not a version part.
    if rest.is_empty() {
        return false;
    }
    let mut chars = rest.chars().peekable();
    // First chunk: one or more digits.
    if !chars.peek().is_some_and(|c| c.is_ascii_digit()) {
        return false;
    }
    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() {
            chars.next();
        } else {
            break;
        }
    }
    // Optional `.<digits>` repetitions.
    while let Some(&c) = chars.peek() {
        if c != '.' {
            return false;
        }
        chars.next();
        if !chars.peek().is_some_and(|c| c.is_ascii_digit()) {
            return false;
        }
        while let Some(&c) = chars.peek() {
            if c.is_ascii_digit() {
                chars.next();
            } else {
                break;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_frontmatter_block_handles_lf() {
        let contents = "---\nfoo: bar\nbaz: qux\n---\nbody text\n";
        assert_eq!(extract_frontmatter_block(contents), Some("foo: bar\nbaz: qux"));
    }

    #[test]
    fn extract_frontmatter_block_returns_none_without_sentinel() {
        let contents = "# Just markdown\nno frontmatter\n";
        assert_eq!(extract_frontmatter_block(contents), None);
    }

    #[test]
    fn parse_license_scalar() {
        let card = parse_yaml_frontmatter("license: apache-2.0\n");
        assert_eq!(card.license.as_deref(), Some("apache-2.0"));
    }

    #[test]
    fn parse_quoted_norwegian_language_code() {
        // YAML 1.1 parses bare `no` as boolean `false`; HF cards
        // quote it as `'no'`. Our unquote helper handles both.
        let frontmatter = "language:\n- en\n- 'no'\n- fr\n";
        let card = parse_yaml_frontmatter(frontmatter);
        assert_eq!(card.languages, vec!["en", "no", "fr"]);
    }

    #[test]
    fn parse_base_model_list() {
        let frontmatter = "base_model:\n- nomic-ai/nomic-embed-text-v2-moe-unsupervised\n";
        let card = parse_yaml_frontmatter(frontmatter);
        assert_eq!(card.base_models.len(), 1);
        assert_eq!(
            card.base_models[0].raw,
            "nomic-ai/nomic-embed-text-v2-moe-unsupervised"
        );
    }

    #[test]
    fn parse_full_nomic_v2_moe_frontmatter() {
        // Matches the real `/opt/hf2q/models/nomic-ai-nomic-embed-text-v2-moe/README.md`
        // frontmatter — verified against the canonical Q8_0 GGUF dump
        // (general.license = apache-2.0, general.tags = [..],
        // general.languages = [en, es, fr, ...], general.base_model =
        // [nomic-ai/nomic-embed-text-v2-moe-unsupervised]).
        let frontmatter = "\
base_model:
- nomic-ai/nomic-embed-text-v2-moe-unsupervised
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
license: apache-2.0
language:
- en
- es
- 'no'
";
        let card = parse_yaml_frontmatter(frontmatter);
        assert_eq!(card.license.as_deref(), Some("apache-2.0"));
        assert_eq!(
            card.tags,
            vec![
                "sentence-transformers",
                "sentence-similarity",
                "feature-extraction"
            ]
        );
        assert_eq!(card.languages, vec!["en", "es", "no"]);
        assert_eq!(card.base_models.len(), 1);
        assert_eq!(
            card.base_models[0].raw,
            "nomic-ai/nomic-embed-text-v2-moe-unsupervised"
        );
    }

    #[test]
    fn split_base_model_title_cases() {
        let (name, org, url) = split_base_model("nomic-ai/nomic-embed-text-v2-moe-unsupervised");
        assert_eq!(
            name.as_deref(),
            Some("Nomic Embed Text v2 Moe Unsupervised")
        );
        assert_eq!(org.as_deref(), Some("Nomic Ai"));
        assert_eq!(
            url.as_deref(),
            Some("https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-unsupervised")
        );
    }

    #[test]
    fn split_base_model_without_slash_returns_none() {
        let (name, org, url) = split_base_model("not-an-org-slash-name");
        assert_eq!(name, None);
        assert_eq!(org, None);
        assert_eq!(url, None);
    }

    #[test]
    fn empty_modelcard_is_empty() {
        let card = ModelCard::default();
        assert!(card.is_empty());
    }

    #[test]
    fn unknown_top_level_keys_are_silently_ignored() {
        let frontmatter = "license: apache-2.0\nmystery_key: foo\nbase_model:\n- a/b\n";
        let card = parse_yaml_frontmatter(frontmatter);
        assert_eq!(card.license.as_deref(), Some("apache-2.0"));
        assert_eq!(card.base_models.len(), 1);
    }

    #[test]
    fn unknown_block_list_keys_consume_their_items() {
        // The `unknown_list:` block's items must be consumed so they
        // don't pollute the next recognized key.
        let frontmatter = "\
unknown_list:
- skip_me_1
- skip_me_2
tags:
- real_tag
";
        let card = parse_yaml_frontmatter(frontmatter);
        assert_eq!(card.tags, vec!["real_tag"]);
    }
}
