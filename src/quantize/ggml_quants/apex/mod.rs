//! ADR-033 Pa — `ApexPolicy` pure-Rust port of `mudler/apex-quant`.
//!
//! Vendored upstream reference at `/opt/hf2q/vendor/apex-quant/` @
//! pinned SHA `63c5048b7dc9ff230f2397d7bc445ca28894b769`. The
//! authoritative algorithmic source is
//! `vendor/apex-quant/scripts/generate_config.sh:69-143` (per-tier
//! rule table) and `:147-193` (per-tensor emission loop).
//!
//! v1 surface (per ADR-033 Decision §6):
//!   - 7 algorithmic tiers: `quality, i-quality, balanced, i-balanced,
//!     compact, i-compact, mini`. Mudler's `nano / i-nano / micro /
//!     i-micro` (experimental) are out of v1 scope; reachable only via
//!     `--quant apex-custom --tensor-type-file <file>`.
//!   - 3 supported MoE arches: `Qwen35Moe, Gemma4 (MoE only),
//!     MiniMaxM2`.
//!   - Per-model fingerprint override (Decision §9): handled at the
//!     CLI-driver layer, not here. This module is the **algorithmic
//!     generator** path only.
//!
//! Module layout:
//!   - `rules`   — per-tier 7-tuples + layer-region partitioning
//!   - `arches`  — MoE tensor-name classifier + supported-arch gate
//!   - `policy`  — `ApexPolicy::target_for` (the QuantPolicy entry point)
//!   - `error`   — `ApexError` typed error taxonomy
//!
//! Per [[feedback-no-loop-suppression-2026-05-17]]: every failure path
//! returns a typed `ApexError` — no silent F16 escape, no implicit
//! dense-policy fallback. Vision/audio tensor F16 emission is handled
//! UPSTREAM at the convert dispatcher; ApexPolicy is not even called
//! for those tensors.

pub mod arches;
pub mod error;
pub mod policy;
pub mod rules;

// Re-exports — the public Pa surface.
pub use arches::{
    classify_moe_tensor, is_apex_supported_arch, MoeTensorRole, SUPPORTED_APEX_ARCHES,
};
pub use error::ApexError;
pub use policy::ApexPolicy;
pub use rules::{
    attn_region, exp_region, shared_region, tier_rules, ApexTier, AttnRegion, ExpRegion,
    SharedRegion, TierRules, SUPPORTED_APEX_TIERS,
};
