//! ADR-033 §9 — mudler tensor-type-file parser.
//!
//! Lifts a vendored `configs/<model>_<tier>.txt` (whose content is
//! baked at compile time via [`super::fingerprint::VENDOR_CONFIGS`])
//! into a `HashMap<String, GgmlType>` keyed by GGUF tensor name.
//!
//! Mudler's file format is one assignment per line:
//!
//! ```text
//! blk.0.ffn_gate_exps=Q6_K
//! blk.0.attn_q=q5_K       # mixed-case is fine, GgmlType::from_name folds
//! # blank lines + leading-`#` comments are ignored
//! ```
//!
//! No regex, no glob, no escape handling — every line is a literal
//! `tensor_name=ggml_type_name` pair. The format matches stock
//! `llama-quantize`'s `--tensor-type-file` consumer.
//!
//! Per ADR Decision §9 + the no-silent-fallback rule: if a tensor name
//! reaches `ApexPolicy::target_for` but its entry is missing from the
//! resolved mudler config, we return a typed
//! [`ApexError::TensorNotInMudlerConfig`] — NEVER fall through to the
//! algorithmic generator. The whole point of the override is "the
//! vendored config's rules win over the algorithmic generator's
//! output" (ADR §9 line 102); a silent fall-through would mute that
//! invariant.

use std::collections::HashMap;
use std::sync::OnceLock;

use super::super::ggml_type::GgmlType;
use super::error::ApexError;
use super::fingerprint::{vendor_config_content, ApexConfigRef};

/// Parsed mudler config — a flat literal map.
///
/// Built once per (vendor_config_path) the first time it's queried;
/// the cache is keyed by the manifest path string (which is stable
/// because the manifest is compile-time-baked).
#[derive(Debug)]
pub struct MudlerConfig {
    /// Tensor name → target [`GgmlType`]. Insertion order is the line
    /// order in the source `.txt`, but lookup is via the hash map.
    pub map: HashMap<String, GgmlType>,
    /// Original manifest-relative path, for error messages.
    pub source_path: &'static str,
}

impl MudlerConfig {
    /// Parse one mudler `.txt` content into a tensor-name → GgmlType
    /// map.
    ///
    /// `source_path` is stored only for error messages — parsing is
    /// purely content-driven.
    ///
    /// Errors on:
    ///   - A non-blank, non-comment line missing the `=` separator.
    ///   - A `GgmlType` token that doesn't parse (e.g. `Q9_K`).
    ///   - A duplicate tensor name with conflicting types.
    pub fn parse(content: &str, source_path: &'static str) -> Result<Self, ApexError> {
        let mut map: HashMap<String, GgmlType> = HashMap::new();
        for (lineno, raw) in content.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (name, tyname) =
                line.split_once('=').ok_or_else(|| ApexError::MudlerConfigParse {
                    source_path: source_path.to_string(),
                    line_number: lineno + 1,
                    detail: format!("missing `=` separator in `{line}`"),
                })?;
            let name = name.trim().to_string();
            let tyname = tyname.trim();
            let ggml = GgmlType::from_name(tyname).ok_or_else(|| ApexError::MudlerConfigParse {
                source_path: source_path.to_string(),
                line_number: lineno + 1,
                detail: format!("unknown GgmlType token `{tyname}`"),
            })?;
            if let Some(prev) = map.insert(name.clone(), ggml) {
                if prev != ggml {
                    return Err(ApexError::MudlerConfigParse {
                        source_path: source_path.to_string(),
                        line_number: lineno + 1,
                        detail: format!(
                            "tensor `{name}` reassigned: {} → {}",
                            prev.name(),
                            ggml.name()
                        ),
                    });
                }
            }
        }
        Ok(Self {
            map,
            source_path,
        })
    }

    /// Look up one tensor name in the parsed map.
    ///
    /// Match policy mirrors stock `llama-quantize --tensor-type-file`
    /// semantics (`std::regex_search` at `llama-quant.cpp:680-687`):
    /// for the simple literal patterns mudler ships
    /// (`blk.0.ffn_gate_exps=Q6_K`), this reduces to: the manifest
    /// key is a prefix or whole-name substring of the GGUF tensor
    /// name. Real GGUF names typically carry a `.weight` suffix
    /// (`blk.0.ffn_gate_exps.weight`) that mudler omits. We match on
    /// (a) exact equality, (b) `tensor_name` starts with `key + '.'`
    /// to catch `.weight` / `.bias` / `.scales` / `.qweight`, and
    /// (c) `tensor_name == key` (no suffix). No regex engine is
    /// needed for v1's literal-only mudler surface; if a future
    /// mudler config introduces real regex syntax, this is the place
    /// to add it.
    ///
    /// Returns `Err(TensorNotInMudlerConfig)` if no key matches —
    /// per ADR §9 the per-model config is authoritative; a missing
    /// tensor is a typed error, NOT a silent algorithmic fall-through.
    pub fn target_for(&self, tensor_name: &str) -> Result<GgmlType, ApexError> {
        // First try exact match (fast path).
        if let Some(&t) = self.map.get(tensor_name) {
            return Ok(t);
        }
        // Then try prefix-with-dot match: `key + '.'` is a prefix of
        // `tensor_name`. Covers the `.weight` / `.bias` suffixes
        // GGUF appends. Linear scan is fine — `self.map` has ≤900
        // entries for the largest vendored config (qwen3.6 122B).
        for (key, &t) in &self.map {
            if tensor_name.len() > key.len() + 1
                && tensor_name.starts_with(key)
                && tensor_name.as_bytes().get(key.len()) == Some(&b'.')
            {
                return Ok(t);
            }
        }
        Err(ApexError::TensorNotInMudlerConfig {
            source_path: self.source_path.to_string(),
            tensor_name: tensor_name.to_string(),
        })
    }

    /// Whether the parsed config has ANY rule that would match this
    /// GGUF tensor name (exact, or `key.<suffix>` prefix). Used by
    /// `ApexPolicy::target_for` to decide between override and
    /// structural-fall-through.
    pub fn contains_match(&self, tensor_name: &str) -> bool {
        if self.map.contains_key(tensor_name) {
            return true;
        }
        for key in self.map.keys() {
            if tensor_name.len() > key.len() + 1
                && tensor_name.starts_with(key)
                && tensor_name.as_bytes().get(key.len()) == Some(&b'.')
            {
                return true;
            }
        }
        false
    }
}

/// Process-wide cache: per `mudler_config_path`, parse the baked
/// content exactly once. Cache key is the same string used in the
/// manifest, so &'static is the natural lifetime.
fn cache_slot(path: &'static str) -> &'static OnceLock<Result<MudlerConfig, ApexError>> {
    // Small static map keyed by &'static str. Built lazily on first
    // miss. We use a Mutex<Vec<...>> instead of a HashMap because the
    // total cardinality is small (≤21 v1 entries) and the key set is
    // closed at compile time.
    use std::sync::Mutex;
    static SLOTS: Mutex<Vec<(&'static str, &'static OnceLock<Result<MudlerConfig, ApexError>>)>> =
        Mutex::new(Vec::new());
    let mut slots = SLOTS.lock().unwrap();
    if let Some((_, slot)) = slots.iter().find(|(p, _)| *p == path) {
        return slot;
    }
    let slot: &'static OnceLock<Result<MudlerConfig, ApexError>> =
        Box::leak(Box::new(OnceLock::new()));
    slots.push((path, slot));
    slot
}

/// Resolve an [`ApexConfigRef`] from the manifest to its parsed
/// [`MudlerConfig`], parsing-and-caching on first access.
///
/// Errors:
///   - [`ApexError::FingerprintConfigMissing`] when the manifest's
///     `mudler_config_path` isn't in [`super::fingerprint::VENDOR_CONFIGS`].
///     This is the hard-error the task spec calls for ("if a
///     mudler_config path in the manifest doesn't exist on disk,
///     ApexPolicy must hard-error").
///   - [`ApexError::MudlerConfigParse`] on parse failure (corrupt
///     vendor content).
pub fn load_mudler_config(entry: &ApexConfigRef) -> Result<&'static MudlerConfig, ApexError> {
    // Find the &'static str matching the manifest path, so the cache
    // slot lives forever (a Box::leak'd OnceLock keyed by &'static).
    // The manifest's entry.mudler_config_path is a String (deserialized);
    // we look it up in VENDOR_CONFIGS to get the static &str.
    let static_path: &'static str = super::fingerprint::VENDOR_CONFIGS
        .iter()
        .find(|(p, _)| *p == entry.mudler_config_path)
        .map(|(p, _)| *p)
        .ok_or_else(|| ApexError::FingerprintConfigMissing {
            fingerprint: entry.fingerprint.clone(),
            mudler_config_path: entry.mudler_config_path.clone(),
        })?;

    let content = vendor_config_content(static_path).ok_or_else(|| {
        ApexError::FingerprintConfigMissing {
            fingerprint: entry.fingerprint.clone(),
            mudler_config_path: entry.mudler_config_path.clone(),
        }
    })?;

    let slot = cache_slot(static_path);
    let result = slot.get_or_init(|| MudlerConfig::parse(content, static_path));
    match result {
        Ok(cfg) => Ok(cfg),
        Err(e) => Err(e.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-check against `vendor/apex-quant/configs/gemma4_26b_balanced.txt`:
    /// at layer 5, `ffn_gate_exps=Q5_K`. Mirrors the analogous test in
    /// `policy.rs::apex_policy_routed_expert_*` but goes through the
    /// per-model override path instead of the algorithmic generator.
    /// Also exercises the `.weight` suffix prefix-match — real GGUF
    /// tensor names carry `.weight`, mudler omits it.
    #[test]
    fn parse_gemma4_balanced_layer_5_routed_expert() {
        let content =
            super::super::fingerprint::vendor_config_content(
                "vendor/apex-quant/configs/gemma4_26b_balanced.txt",
            )
            .expect("baked vendor content present");
        let cfg = MudlerConfig::parse(content, "test").unwrap();
        // Bare key (exact-match path).
        assert_eq!(
            cfg.target_for("blk.5.ffn_gate_exps").unwrap(),
            GgmlType::Q5_K
        );
        // With `.weight` suffix (prefix-match path — real GGUF).
        assert_eq!(
            cfg.target_for("blk.5.ffn_gate_exps.weight").unwrap(),
            GgmlType::Q5_K
        );
        assert_eq!(
            cfg.target_for("blk.0.attn_q.weight").unwrap(),
            GgmlType::Q6_K
        );
        assert_eq!(
            cfg.target_for("blk.0.ffn_gate_shexp.weight").unwrap(),
            GgmlType::Q8_0
        );
    }

    /// Carnice qwen3.6-MTP at layer 5: ffn_gate_exps=Q5_K (per the
    /// existing policy.rs::apex_policy_matches_carnice_qwen36_layer_5
    /// cross-check on the algorithmic generator). The per-model
    /// override MUST produce the same Q5_K at this layer/role.
    #[test]
    fn parse_carnice_mtp_quality_layer_5() {
        let content = super::super::fingerprint::vendor_config_content(
            "vendor/apex-quant/configs/carnice_qwen36_mtp_quality.txt",
        )
        .expect("baked vendor content present");
        let cfg = MudlerConfig::parse(content, "test").unwrap();
        assert_eq!(
            cfg.target_for("blk.5.ffn_gate_exps").unwrap(),
            GgmlType::Q5_K
        );
        assert_eq!(cfg.target_for("blk.5.attn_q").unwrap(), GgmlType::Q6_K);
        assert_eq!(
            cfg.target_for("blk.5.ffn_gate_shexp").unwrap(),
            GgmlType::Q8_0
        );
    }

    /// Mudler emits both lower- and upper-case GgmlType tokens; the
    /// parser MUST fold both via `GgmlType::from_name`.
    #[test]
    fn parse_handles_mixed_case_ggml_names() {
        let content = "blk.0.attn_q=Q5_K\nblk.0.attn_k=q5_K\nblk.0.attn_v=q5_k";
        let cfg = MudlerConfig::parse(content, "test").unwrap();
        assert_eq!(cfg.target_for("blk.0.attn_q").unwrap(), GgmlType::Q5_K);
        assert_eq!(cfg.target_for("blk.0.attn_k").unwrap(), GgmlType::Q5_K);
        assert_eq!(cfg.target_for("blk.0.attn_v").unwrap(), GgmlType::Q5_K);
    }

    /// Blank lines + `#` comments are skipped silently.
    #[test]
    fn parse_skips_blank_and_comment_lines() {
        let content = "
            # this is a comment

            blk.0.attn_q=Q5_K
            ## indented comment
            blk.0.attn_k=Q5_K
        ";
        let cfg = MudlerConfig::parse(content, "test").unwrap();
        assert_eq!(cfg.map.len(), 2);
    }

    /// Missing `=` → typed parse error with the line number.
    #[test]
    fn parse_errors_on_missing_separator() {
        let content = "blk.0.attn_q Q5_K";
        let err = MudlerConfig::parse(content, "test/path").unwrap_err();
        match err {
            ApexError::MudlerConfigParse {
                source_path,
                line_number,
                ..
            } => {
                assert_eq!(source_path, "test/path");
                assert_eq!(line_number, 1);
            }
            other => panic!("expected MudlerConfigParse, got {other:?}"),
        }
    }

    /// Unknown GgmlType token → typed parse error.
    #[test]
    fn parse_errors_on_unknown_ggml_type() {
        let content = "blk.0.attn_q=Q9_K";
        let err = MudlerConfig::parse(content, "test").unwrap_err();
        assert!(matches!(err, ApexError::MudlerConfigParse { .. }));
    }

    /// Tensor not in the parsed config → typed
    /// `TensorNotInMudlerConfig`. No silent fall-through.
    #[test]
    fn missing_tensor_returns_typed_error() {
        let cfg = MudlerConfig::parse("blk.0.attn_q=Q5_K", "test").unwrap();
        let err = cfg.target_for("blk.99.fake").unwrap_err();
        assert!(matches!(err, ApexError::TensorNotInMudlerConfig { .. }));
    }
}
