//! ADR-012 P9 — `QualityThresholds` drift gate.
//!
//! Decision 17 (party-mode refinement 2026-04-24) names the quality
//! thresholds as literal constants, NOT "reasonable" or "TBD". This
//! test pins them so a silent edit to `src/arch/registry.rs` cannot
//! drift the ppl/kl gates without tripping a test.
//!
//! Integration-style so the test lives next to the ADR artefacts in
//! `tests/` rather than inline; readers come to this file first when
//! investigating a "why do the DWQ numbers shift" question.

// The `arch` module lives inside the binary crate; integration tests
// can reach its unit tests' `const ADR_012_DEFAULT` only via the
// binary. Use `assert_cmd` — the `hf2q` CLI exposes the constants
// indirectly through `hf2q smoke --help` (the thresholds are referenced
// in the subcommand's `about` text when P9 lands) and through the
// arch-registry traversal. Pre-P9, we prove the constants exist in
// the source and carry the right values by parsing the source file
// directly — a stable drift gate that doesn't require a public API.

#[test]
fn dwq46_ppl_ratio_is_1_10_per_adr012_party_mode_2026_04_24() {
    // ADR-012 Decision 17 party-mode-confirmed (2026-04-24):
    //   "DWQ46 perplexity ≤ 1.10× F16 reference PPL"
    let registry_src = std::fs::read_to_string("src/arch/registry.rs")
        .expect("src/arch/registry.rs present");
    assert!(
        registry_src.contains("ppl_ratio_dwq46: 1.10"),
        "ADR-012 Decision 17 requires DWQ46 PPL ratio = 1.10; src/arch/registry.rs must carry that literal. \
         If you're changing this, open an ADR amendment first."
    );
}

#[test]
fn dwq48_ppl_ratio_is_1_05_per_adr012_party_mode_2026_04_24() {
    // ADR-012 Decision 17:
    //   "DWQ48 perplexity ≤ 1.05× F16 reference PPL"
    let registry_src = std::fs::read_to_string("src/arch/registry.rs")
        .expect("src/arch/registry.rs present");
    assert!(
        registry_src.contains("ppl_ratio_dwq48: 1.05"),
        "ADR-012 Decision 17 requires DWQ48 PPL ratio = 1.05."
    );
}

#[test]
fn median_kl_is_0_02_nats_per_adr012_party_mode_2026_04_24() {
    // ADR-012 Decision 17:
    //   "Median KL-divergence per token < 0.02 nats"
    let registry_src = std::fs::read_to_string("src/arch/registry.rs")
        .expect("src/arch/registry.rs present");
    assert!(
        registry_src.contains("max_median_kl: 0.02"),
        "ADR-012 Decision 17 requires median KL < 0.02 nats."
    );
}

#[test]
fn thresholds_appear_as_literal_constants_not_tbd() {
    // Decision 17 rejects "TBD" / "reasonable" values. The test
    // catches accidental refactors that migrate the constants to a
    // config file or environment lookup.
    let registry_src = std::fs::read_to_string("src/arch/registry.rs")
        .expect("src/arch/registry.rs present");
    assert!(
        registry_src.contains("pub const ADR_012_DEFAULT: QualityThresholds"),
        "ADR_012_DEFAULT must be a `const`, not a runtime lookup."
    );
    // Decision 17 rejects a TBD literal VALUE. The comment trail
    // inside registry.rs may reference the word "TBD" when quoting
    // the decision text ("not TBD, not reasonable"); that's fine.
    // What we reject is any `: TBD` value form or a `const` whose
    // literal value is the word TBD.
    assert!(
        !registry_src.contains(": TBD"),
        "no placeholder ': TBD' value allowed in QualityThresholds constants"
    );
    assert!(
        !registry_src.contains("= TBD"),
        "no '= TBD' placeholder allowed in QualityThresholds constants"
    );
}
