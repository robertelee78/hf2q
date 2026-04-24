//! ArchRegistry + ArchEntry — single source of truth per ADR-012
//! Decision 20.
//!
//! The registry is a `const` lookup keyed on the GGUF arch string
//! (`qwen35`, `qwen35moe`, ...). `ArchRegistry::get` is the only
//! dispatch surface. Unknown arches return `ArchError::UnknownArch`
//! with the list of known entries — never a `todo!()` branch.
//!
//! P8 ships exactly two entries: `qwen35` and `qwen35moe`. Gemma4,
//! Ministral (ADR-015), and DeepSeek-V3 (ADR-016) each add their
//! own entry file in their own ADR when opened. Per mantra:
//! populated-stub is still a stub — no placeholder entries.

use std::fmt;

use super::catalog::{CatalogExpansion, TensorCatalog};

/// Quality thresholds enforced by P9's PPL/KL eval.
///
/// Values are the party-mode-confirmed numbers from ADR-012 Decision 17
/// (2026-04-24). Any drift is caught by `tests/quality_thresholds.rs`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityThresholds {
    /// Max PPL ratio (DWQ / F16) for dwq-mixed-4-6.
    pub ppl_ratio_dwq46: f64,
    /// Max PPL ratio (DWQ / F16) for dwq-mixed-4-8.
    pub ppl_ratio_dwq48: f64,
    /// Max median per-token KL divergence, in nats.
    pub max_median_kl: f64,
}

impl QualityThresholds {
    /// Default for qwen35 / qwen35moe per ADR-012 Decision 17
    /// (party-mode 2026-04-24).
    pub const ADR_012_DEFAULT: QualityThresholds = QualityThresholds {
        ppl_ratio_dwq46: 1.10,
        ppl_ratio_dwq48: 1.05,
        max_median_kl: 0.02,
    };
}

/// Evaluation corpus pointer. The bytes live under
/// `tests/fixtures/ppl-corpus/{id}.tokens` with a `.sha256` sidecar.
#[derive(Debug, Clone, Copy)]
pub struct EvalCorpus {
    /// Stable identifier (e.g. `"wikitext2"`).
    pub id: &'static str,
    /// Token count asserted at load time.
    pub token_count: u32,
    /// Expected SHA-256 of the token-bytes file, lowercased hex.
    pub sha256_hex: &'static str,
}

/// One arch's conformance surface.
#[derive(Debug, Clone, Copy)]
pub struct ArchEntry {
    /// GGUF arch string (`"qwen35"`, `"qwen35moe"`).
    pub arch: &'static str,
    /// HF `config.json::architectures[0]` strings that map to this arch.
    pub hf_architectures: &'static [&'static str],
    /// Hand-transcribed tensor catalog (P4 tensor mapping rendered as templates).
    pub tensor_catalog: &'static TensorCatalog,
    /// Whether the arch emits `blk.{L}.nextn.*` MTP tensors.
    pub has_mtp: bool,
    /// Whether the arch has an `--emit-vision-tower` path.
    pub has_vision: bool,
    /// Smoke prompts — deterministic inputs for `hf2q smoke`.
    pub smoke_prompts: &'static [&'static str],
    /// PPL eval corpus (Decision 17 quality gate).
    pub ppl_corpus: EvalCorpus,
    /// Per-arch quality bounds.
    pub quality_thresholds: QualityThresholds,
    /// Disk floor in GB required to convert this arch; smoke preflight
    /// exit code 3 fires when free space is below `disk_floor_gb + 10`.
    pub disk_floor_gb: u32,
    /// HF repos this arch's smoke path expects to resolve.
    pub hf_repos: &'static [&'static str],
}

impl ArchEntry {
    /// Compute the expected loaded-tensor count for a concrete model
    /// matching this arch. Smoke harness asserts
    /// `llama_model_load: loaded tensor 0x%x == this value`.
    pub fn expected_tensor_count(&self, exp: CatalogExpansion) -> u64 {
        self.tensor_catalog.expected_tensor_count(exp)
    }
}

/// Errors raised by the registry dispatcher. Uniform for every
/// unregistered arch — no per-arch placeholder branches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArchError {
    /// The requested arch is not registered in this build.
    /// The `known` list is sorted alphabetically for stable output.
    UnknownArch {
        requested: String,
        known: Vec<&'static str>,
    },
}

impl fmt::Display for ArchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArchError::UnknownArch { requested, known } => {
                write!(
                    f,
                    "unknown arch: {:?}; known arches: {}",
                    requested,
                    known.join(", ")
                )
            }
        }
    }
}

impl std::error::Error for ArchError {}

/// The arch registry. `const`-constructed at compile time; lookups are
/// `O(N)` over a tiny slice — cheaper than hashing for N ≤ 16.
#[derive(Debug)]
pub struct ArchRegistry {
    entries: &'static [&'static ArchEntry],
}

impl ArchRegistry {
    /// Return the process-global registry.
    pub fn global() -> &'static ArchRegistry {
        &GLOBAL_REGISTRY
    }

    /// Lookup by GGUF arch string. Returns a uniform structured error
    /// for any unknown key (including `gemma4`, `ministral`,
    /// `deepseekv3`, `bogus` — no special cases, no `todo!()`).
    pub fn get(&self, arch: &str) -> Result<&'static ArchEntry, ArchError> {
        for entry in self.entries {
            if entry.arch == arch {
                return Ok(entry);
            }
        }
        Err(ArchError::UnknownArch {
            requested: arch.to_string(),
            known: self.known_arches(),
        })
    }

    /// Sorted alphabetically for stable stderr output.
    pub fn known_arches(&self) -> Vec<&'static str> {
        let mut v: Vec<&'static str> = self.entries.iter().map(|e| e.arch).collect();
        v.sort_unstable();
        v
    }

    /// Lookup by HF `architectures[0]` string (for config-driven dispatch).
    pub fn get_by_hf_architecture(
        &self,
        hf_arch: &str,
    ) -> Result<&'static ArchEntry, ArchError> {
        for entry in self.entries {
            if entry.hf_architectures.contains(&hf_arch) {
                return Ok(entry);
            }
        }
        Err(ArchError::UnknownArch {
            requested: hf_arch.to_string(),
            known: self.known_arches(),
        })
    }

    /// Iterate over all registered entries (used by `hf2q smoke --help`
    /// for listing and by conformance sweeps in CI).
    pub fn iter(&self) -> impl Iterator<Item = &&'static ArchEntry> {
        self.entries.iter()
    }
}

/// Process-global singleton. Exactly two entries — Gemma4, Ministral,
/// DeepSeek-V3 each register in their own ADR.
const GLOBAL_REGISTRY: ArchRegistry = ArchRegistry {
    entries: &[
        &super::entries::qwen35::ENTRY,
        &super::entries::qwen35moe::ENTRY,
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_registry_has_exactly_qwen35_and_qwen35moe() {
        let known = ArchRegistry::global().known_arches();
        assert_eq!(known, vec!["qwen35", "qwen35moe"]);
    }

    #[test]
    fn get_returns_entry_for_known_arch() {
        let e = ArchRegistry::global().get("qwen35").expect("known");
        assert_eq!(e.arch, "qwen35");
        let e = ArchRegistry::global().get("qwen35moe").expect("known");
        assert_eq!(e.arch, "qwen35moe");
    }

    #[test]
    fn unknown_arch_returns_uniform_structured_error() {
        // Every unregistered arch — gemma4, ministral, deepseekv3,
        // bogus — returns the SAME error variant with the list of
        // known arches. Proves the registry is load-bearing.
        for bogus in &["gemma4", "ministral", "deepseekv3", "bogus", ""] {
            let err = ArchRegistry::global().get(bogus).unwrap_err();
            match err {
                ArchError::UnknownArch { requested, known } => {
                    assert_eq!(requested, *bogus);
                    assert_eq!(known, vec!["qwen35", "qwen35moe"]);
                }
            }
        }
    }

    #[test]
    fn hf_architectures_dispatch_preserves_arch_routing() {
        // Decision 1 insight: `config.architectures[0]`, NOT `model_type`.
        let e = ArchRegistry::global()
            .get_by_hf_architecture("Qwen3_5ForCausalLM")
            .expect("Qwen3_5ForCausalLM → qwen35");
        assert_eq!(e.arch, "qwen35");
        let e = ArchRegistry::global()
            .get_by_hf_architecture("Qwen3_5MoeForCausalLM")
            .expect("Qwen3_5MoeForCausalLM → qwen35moe");
        assert_eq!(e.arch, "qwen35moe");
    }

    #[test]
    fn unknown_hf_architecture_returns_uniform_error() {
        let err = ArchRegistry::global()
            .get_by_hf_architecture("FakeNonexistentForCausalLM")
            .unwrap_err();
        match err {
            ArchError::UnknownArch { requested, known } => {
                assert_eq!(requested, "FakeNonexistentForCausalLM");
                assert_eq!(known, vec!["qwen35", "qwen35moe"]);
            }
        }
    }

    #[test]
    fn quality_thresholds_match_adr012_party_mode() {
        // ADR-012 Decision 17 (2026-04-24 party-mode refinement):
        //   DWQ46 PPL ≤ 1.10× F16, DWQ48 ≤ 1.05× F16, median KL < 0.02 nats
        // Values are literal `const` per Decision 17's "not TBD, not reasonable"
        // clause; this test catches silent drift.
        assert_eq!(QualityThresholds::ADR_012_DEFAULT.ppl_ratio_dwq46, 1.10);
        assert_eq!(QualityThresholds::ADR_012_DEFAULT.ppl_ratio_dwq48, 1.05);
        assert_eq!(QualityThresholds::ADR_012_DEFAULT.max_median_kl, 0.02);
    }

    #[test]
    fn arch_error_display_is_actionable() {
        let err = ArchError::UnknownArch {
            requested: "gemma4".to_string(),
            known: vec!["qwen35", "qwen35moe"],
        };
        let s = format!("{}", err);
        assert!(s.contains("unknown arch"));
        assert!(s.contains("\"gemma4\""));
        assert!(s.contains("qwen35"));
        assert!(s.contains("qwen35moe"));
    }

    /// Field-invariant sweep — every registered arch entry must satisfy
    /// load-bearing non-empty / non-zero invariants. Decision 20 §future-
    /// arch onboarding: a new entry with e.g. empty `hf_repos` or zero
    /// `disk_floor_gb` would silently break smoke preflight. This test
    /// is the gate — adding Ministral / DeepSeek-V3 with a nonsensical
    /// field fails CI before any runtime manifest.
    #[test]
    fn every_registered_entry_passes_field_invariants() {
        for entry in ArchRegistry::global().iter() {
            // Identity.
            assert!(!entry.arch.is_empty(), "arch string must be non-empty");
            assert!(
                !entry.hf_architectures.is_empty(),
                "{}: hf_architectures must list at least one HF arch",
                entry.arch
            );
            for hf in entry.hf_architectures {
                assert!(
                    !hf.is_empty(),
                    "{}: hf_architectures must not contain empty strings",
                    entry.arch
                );
            }

            // Preflight-load-bearing.
            assert!(
                entry.disk_floor_gb > 0,
                "{}: disk_floor_gb must be > 0 (else preflight exit 3 never fires)",
                entry.arch
            );
            assert!(
                !entry.hf_repos.is_empty(),
                "{}: hf_repos must list at least one resolvable repo",
                entry.arch
            );
            for repo in entry.hf_repos {
                assert!(
                    repo.contains('/'),
                    "{}: hf_repo {:?} must be `owner/name` form",
                    entry.arch,
                    repo
                );
            }

            // Smoke-load-bearing.
            assert!(
                !entry.smoke_prompts.is_empty(),
                "{}: smoke_prompts must have at least one prompt (llama-cli needs -p)",
                entry.arch
            );
            for prompt in entry.smoke_prompts {
                assert!(
                    !prompt.trim().is_empty(),
                    "{}: smoke_prompts must not contain whitespace-only prompts",
                    entry.arch
                );
            }

            // Tensor catalog non-empty — otherwise expected_tensor_count
            // returns 0 and the transcript assertion becomes vacuous.
            assert!(
                !entry.tensor_catalog.entries.is_empty(),
                "{}: tensor_catalog must have at least one entry",
                entry.arch
            );

            // Quality thresholds — ratios > 1.0 (DWQ must be at least
            // tie-or-worse with F16; a ratio of 1.0 means bit-identical)
            // and KL > 0 (0 would accept any divergence).
            assert!(
                entry.quality_thresholds.ppl_ratio_dwq46 >= 1.0,
                "{}: ppl_ratio_dwq46 must be >= 1.0",
                entry.arch
            );
            assert!(
                entry.quality_thresholds.ppl_ratio_dwq48 >= 1.0,
                "{}: ppl_ratio_dwq48 must be >= 1.0",
                entry.arch
            );
            assert!(
                entry.quality_thresholds.ppl_ratio_dwq48
                    <= entry.quality_thresholds.ppl_ratio_dwq46,
                "{}: dwq48 threshold must be tighter than dwq46 (8-bit keeps more fidelity)",
                entry.arch
            );
            assert!(
                entry.quality_thresholds.max_median_kl > 0.0,
                "{}: max_median_kl must be > 0",
                entry.arch
            );
        }
    }
}
