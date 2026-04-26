//! Arch registry (ADR-012 Decision 20).
//!
//! Singleton [`Registry`] keyed by the GGUF arch string (`qwen35`,
//! `qwen35moe`, …).  Lookup by [`Registry::get`].

use std::collections::HashMap;
use std::sync::OnceLock;

use thiserror::Error;

use super::catalog::TensorCatalog;

/// Quality thresholds enforced by Decision 17's PPL/KL eval.  Constants live
/// here (not in `dwq.rs`) so the smoke harness can read them without pulling
/// in the calibration crate.  ADR-012 party-mode-confirmed values (2026-04-24).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityThresholds {
    /// `dwq46` PPL must be ≤ this multiple of the F16 reference PPL.
    pub ppl_ratio_dwq46: f64,
    /// `dwq48` PPL must be ≤ this multiple of the F16 reference PPL.
    pub ppl_ratio_dwq48: f64,
    /// Median per-token KL divergence (in nats) must be < this value.
    pub max_median_kl: f64,
}

impl QualityThresholds {
    /// ADR-012 party-mode-confirmed defaults.
    pub const ADR_012_DEFAULTS: Self = Self {
        ppl_ratio_dwq46: 1.10,
        ppl_ratio_dwq48: 1.05,
        max_median_kl: 0.02,
    };
}

/// Description of the eval corpus consumed by Decision 17's PPL/KL helper.
#[derive(Debug, Clone, Copy)]
pub struct EvalCorpus {
    /// Path under `tests/fixtures/ppl-corpus/` that holds the tokenized eval
    /// stream.  Decision 17 ships `wikitext2.tokens` + a SHA-256 sidecar.
    pub fixture_path: &'static str,
    /// Number of tokens to evaluate from the start of the corpus.
    pub eval_tokens: usize,
}

impl EvalCorpus {
    pub const WIKITEXT2_512: Self = Self {
        fixture_path: "tests/fixtures/ppl-corpus/wikitext2.tokens",
        eval_tokens: 512,
    };
}

/// One entry in the arch registry.  Hand-populated per arch, no inference.
#[derive(Debug, Clone)]
pub struct ArchEntry {
    /// GGUF `general.architecture` string (e.g. `"qwen35moe"`).
    pub arch: &'static str,

    /// HuggingFace `config.json::architectures[0]` strings that resolve to
    /// this arch.  Used by the smoke harness to validate `--arch X` against
    /// what the source repo claims to be.
    pub hf_architectures: &'static [&'static str],

    /// Hand-transcribed tensor catalog.
    pub tensor_catalog: &'static TensorCatalog,

    /// Whether this arch carries an MTP (Multi-Token Prediction) block.
    pub has_mtp: bool,

    /// Whether this arch ships a paired vision tower (mmproj GGUF).
    pub has_vision: bool,

    /// Disk floor in GB required for a clean convert (download + intermediate
    /// + output).  Smoke preflight asserts free-space ≥ this + 10 GB buffer.
    pub disk_floor_gb: u64,

    /// Deterministic 8-token smoke prompts, executed with `--seed 42 --temp 0`.
    pub smoke_prompts: &'static [&'static str],

    /// HF repo IDs canonical for this arch (resolvable by `huggingface-cli
    /// repo info`).  Smoke preflight asserts at least one is reachable.
    pub hf_repos: &'static [&'static str],

    /// Eval corpus consumed by Decision 17's PPL/KL helper.
    pub ppl_corpus: EvalCorpus,

    /// Quality thresholds enforced by Decision 17.
    pub quality_thresholds: QualityThresholds,
}

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("unknown arch '{requested}'; known arches: {known}")]
    UnknownArch { requested: String, known: String },
}

/// Process-wide arch registry.
pub struct Registry {
    by_arch: HashMap<&'static str, &'static ArchEntry>,
}

static REGISTRY: OnceLock<Registry> = OnceLock::new();

impl Registry {
    pub fn global() -> &'static Self {
        REGISTRY.get_or_init(Self::build)
    }

    fn build() -> Self {
        let mut by_arch: HashMap<&'static str, &'static ArchEntry> = HashMap::new();

        // ADR-012 P8 ships exactly two entries.  Future arches add their own
        // file under `entries/` in their own ADR — no placeholders here.
        for entry in super::entries::all() {
            by_arch.insert(entry.arch, entry);
        }

        Self { by_arch }
    }

    /// Look up an arch by GGUF string.  Returns the canonical
    /// "unknown arch; known arches: …" error when missing — same surface for
    /// every unregistered key, no per-arch `todo!()` branches.
    pub fn get(&self, arch: &str) -> Result<&'static ArchEntry, RegistryError> {
        self.by_arch
            .get(arch)
            .copied()
            .ok_or_else(|| RegistryError::UnknownArch {
                requested: arch.to_string(),
                known: self.known().join(", "),
            })
    }

    /// Sorted list of registered arch names.
    pub fn known(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = self.by_arch.keys().copied().collect();
        names.sort_unstable();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_qwen35_and_qwen35moe() {
        let reg = Registry::global();
        assert!(reg.get("qwen35").is_ok());
        assert!(reg.get("qwen35moe").is_ok());
    }

    #[test]
    fn registry_unknown_arch_uniform_error() {
        let reg = Registry::global();
        // Every unregistered key (including names from future ADRs and
        // typos) returns the SAME error shape — proves the dispatch is
        // load-bearing per Decision 20.
        for name in &["bogus", "gemma4", "ministral", "deepseekv3", "llama3"] {
            let err = reg.get(name).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("unknown arch"), "got: {msg}");
            assert!(msg.contains(name), "got: {msg}");
            assert!(msg.contains("qwen35"), "got: {msg}");
            assert!(msg.contains("qwen35moe"), "got: {msg}");
        }
    }

    #[test]
    fn registry_known_is_sorted_and_exact() {
        let reg = Registry::global();
        let known = reg.known();
        assert_eq!(known, vec!["qwen35", "qwen35moe"]);
    }

    #[test]
    fn quality_thresholds_match_adr012_party_mode() {
        let q = QualityThresholds::ADR_012_DEFAULTS;
        // Locked at 2026-04-24; if these change the ADR ships an addendum.
        assert_eq!(q.ppl_ratio_dwq46, 1.10);
        assert_eq!(q.ppl_ratio_dwq48, 1.05);
        assert_eq!(q.max_median_kl, 0.02);
    }
}
