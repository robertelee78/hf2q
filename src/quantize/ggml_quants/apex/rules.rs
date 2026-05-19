//! ApexPolicy per-tier rule tables — pure-Rust port of
//! `mudler/apex-quant/scripts/generate_config.sh` @ pinned SHA
//! `63c5048b7dc9ff230f2397d7bc445ca28894b769`.
//!
//! Source authoritative ref (verified line-for-line against the bash):
//! `/opt/hf2q/vendor/apex-quant/scripts/generate_config.sh:69-143`.
//!
//! v1 Apex ships **7 tiers** (per ADR-033 §"Decision §6"):
//!   - quality, i-quality
//!   - balanced, i-balanced
//!   - compact, i-compact
//!   - mini
//!
//! Dropped from v1 (mudler's experimental tiers): `nano, i-nano,
//! micro, i-micro`. Per-model configs for these still ship in
//! `data/apex-references/` and are usable via `--quant apex-custom`.
//!
//! Each tier resolves to a **7-tuple** of `GgmlType` per (layer-region,
//! tensor-role):
//!
//! | Tier             | edge_exp | near_exp | mid_exp | edge_shared | mid_shared | edge_attn | mid_attn |
//! | ---------------- | -------- | -------- | ------- | ----------- | ---------- | --------- | -------- |
//! | quality, i-quality | Q6_K   | Q5_K     | IQ4_XS  | Q8_0        | Q8_0       | Q6_K      | Q6_K     |
//! | balanced, i-balanced | Q6_K | Q5_K     | Q5_K    | Q8_0        | Q8_0       | Q6_K      | Q6_K     |
//! | compact, i-compact | Q4_K   | Q3_K     | Q3_K    | Q6_K        | Q6_K       | Q4_K      | Q4_K     |
//! | mini             | Q3_K     | Q3_K     | IQ2_S   | Q5_K        | Q4_K       | Q4_K      | Q3_K     |
//!
//! Layer-region partitions (per `generate_config.sh:62-66, 147-169`,
//! `L_LAST = n_layers - 1`):
//!   - **EXP/SHARED EDGE** = `[0..=4] ∪ [(L_LAST-4)..=L_LAST]` (5+5 layers)
//!   - **EXP NEAR**        = `[5..=9] ∪ [(L_LAST-9)..=(L_LAST-5)]` (5+5 layers)
//!   - **EXP MID**         = `[10..=(L_LAST-10)]` (everything else)
//!   - **SHARED MID**      = `[5..=(L_LAST-5)]` (no near band for shared)
//!   - **ATTN EDGE**       = `[0..=2] ∪ [(L_LAST-2)..=L_LAST]` (3+3 layers)
//!   - **ATTN MID**        = `[3..=(L_LAST-3)]` (everything else)
//!
//! The asymmetric ATTN band (3 vs 5) is faithful to mudler's bash and
//! is the reason ATTN_EDGE and EXP_EDGE differ at layers 3, 4, L-4,
//! L-5. Cross-checked against the vendored configs (e.g.
//! `configs/gemma4_26b_mini.txt:blk.3.attn_q=Q3_K` while
//! `blk.3.ffn_gate_exps=Q3_K` and `blk.2.attn_q=Q4_K`).

use super::super::ggml_type::GgmlType;

/// The 7 v1 Apex algorithmic tiers. Custom + experimental nano/micro
/// (with their I-variants) are out of v1 scope (handled via
/// `apex-custom --tensor-type-file`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ApexTier {
    /// `quality` — Q6_K edge / Q5_K near / IQ4_XS mid experts, Q8_0
    /// shared, Q6_K attention. Mudler's highest-quality tier.
    Quality,
    /// `i-quality` — identical type-table to `Quality`. The `I-` prefix
    /// flags "calibrate with imatrix at quantize time"; the
    /// destination `GgmlType` for each tensor is the same as
    /// non-I tier. (See mudler `generate_config.sh:70` — `quality`
    /// and `i-quality` share the same case branch.)
    IQuality,
    /// `balanced` — Q6_K edge / Q5_K near + mid experts, Q8_0 shared,
    /// Q6_K attention. The default mid-tier; no aggressive sub-Q5
    /// experts.
    Balanced,
    /// `i-balanced` — same table as `Balanced` + imatrix at quantize
    /// time.
    IBalanced,
    /// `compact` — Q4_K edge / Q3_K near + mid experts, Q6_K shared,
    /// Q4_K attention. The "small but coherent" tier.
    Compact,
    /// `i-compact` — same table as `Compact` + imatrix.
    ICompact,
    /// `mini` — Q3_K edge experts / Q3_K near experts / IQ2_S mid
    /// experts, Q5_K edge shared / Q4_K mid shared, Q4_K edge
    /// attention / Q3_K mid attention. Mudler notes: "benefits from
    /// imatrix" but works without it. (No matching `i-mini` in
    /// mudler's published surface.)
    Mini,
}

impl ApexTier {
    /// Canonical CLI suffix (`--quant apex-<name>`). Used for
    /// round-tripping and error messages.
    pub const fn cli_name(self) -> &'static str {
        match self {
            ApexTier::Quality => "quality",
            ApexTier::IQuality => "i-quality",
            ApexTier::Balanced => "balanced",
            ApexTier::IBalanced => "i-balanced",
            ApexTier::Compact => "compact",
            ApexTier::ICompact => "i-compact",
            ApexTier::Mini => "mini",
        }
    }

    /// Does this tier require an imatrix at quantize time?  True for
    /// the `i-*` variants only. (`mini` "benefits from" imatrix per
    /// mudler but isn't gated on it.)
    pub const fn requires_imatrix(self) -> bool {
        matches!(
            self,
            ApexTier::IQuality | ApexTier::IBalanced | ApexTier::ICompact
        )
    }

    /// Parse a CLI tier suffix (e.g. `"quality"` or `"i-balanced"`)
    /// into the variant. Returns `None` if the string isn't in the
    /// v1 supported set.
    pub fn from_cli_name(s: &str) -> Option<Self> {
        match s {
            "quality" => Some(ApexTier::Quality),
            "i-quality" => Some(ApexTier::IQuality),
            "balanced" => Some(ApexTier::Balanced),
            "i-balanced" => Some(ApexTier::IBalanced),
            "compact" => Some(ApexTier::Compact),
            "i-compact" => Some(ApexTier::ICompact),
            "mini" => Some(ApexTier::Mini),
            _ => None,
        }
    }
}

/// All v1 tier CLI names, in declaration order. For error messages.
pub const SUPPORTED_APEX_TIERS: &[&str] = &[
    "quality",
    "i-quality",
    "balanced",
    "i-balanced",
    "compact",
    "i-compact",
    "mini",
];

/// Per-tier 7-tuple of GgmlType assignments. Mirrors the {EDGE_EXP,
/// NEAR_EXP, MID_EXP, EDGE_SHARED, MID_SHARED, EDGE_ATTN, MID_ATTN}
/// shell variables in `generate_config.sh`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierRules {
    /// Routed expert tensors (`ffn_{gate,up,down}_exps`) in EDGE
    /// layers `[0..=4] ∪ [(L-5)..=(L-1)]`.
    pub edge_exp: GgmlType,
    /// Routed expert tensors in NEAR-edge layers `[5..=9] ∪ [(L-10)..=(L-6)]`.
    pub near_exp: GgmlType,
    /// Routed expert tensors in MID layers `[10..=(L-11)]`.
    pub mid_exp: GgmlType,
    /// Shared expert tensors (`ffn_{gate,up,down}_shexp`) in EDGE layers
    /// (`[0..=4] ∪ [(L-5)..=(L-1)]`). There is **no NEAR band for shared**
    /// in mudler's recipe.
    pub edge_shared: GgmlType,
    /// Shared expert tensors in MID layers `[5..=(L-6)]`.
    pub mid_shared: GgmlType,
    /// Attention tensors (attn_q / attn_k / attn_v / attn_output /
    /// attn_qkv / attn_gate) in EDGE-attn layers
    /// `[0..=2] ∪ [(L-3)..=(L-1)]`. **Note**: attention's EDGE band
    /// is 3-wide, not 5-wide like experts/shared. Asymmetry is
    /// faithful to mudler.
    pub edge_attn: GgmlType,
    /// Attention tensors in MID-attn layers `[3..=(L-4)]`.
    pub mid_attn: GgmlType,
}

/// Lookup the per-tier 7-tuple. Mirrors the `case "$PROFILE"` block
/// in `generate_config.sh:69-143`. The I-variants share the same
/// type-table as their non-I siblings; the I-prefix only flags
/// "use imatrix at quantize time".
pub const fn tier_rules(tier: ApexTier) -> TierRules {
    match tier {
        // generate_config.sh:70-78 — quality | i-quality
        ApexTier::Quality | ApexTier::IQuality => TierRules {
            edge_exp: GgmlType::Q6_K,
            near_exp: GgmlType::Q5_K,
            mid_exp: GgmlType::IQ4_XS,
            edge_shared: GgmlType::Q8_0,
            mid_shared: GgmlType::Q8_0,
            edge_attn: GgmlType::Q6_K,
            mid_attn: GgmlType::Q6_K,
        },
        // generate_config.sh:79-87 — balanced | i-balanced
        ApexTier::Balanced | ApexTier::IBalanced => TierRules {
            edge_exp: GgmlType::Q6_K,
            near_exp: GgmlType::Q5_K,
            mid_exp: GgmlType::Q5_K,
            edge_shared: GgmlType::Q8_0,
            mid_shared: GgmlType::Q8_0,
            edge_attn: GgmlType::Q6_K,
            mid_attn: GgmlType::Q6_K,
        },
        // generate_config.sh:88-96 — compact | i-compact
        ApexTier::Compact | ApexTier::ICompact => TierRules {
            edge_exp: GgmlType::Q4_K,
            near_exp: GgmlType::Q3_K,
            mid_exp: GgmlType::Q3_K,
            edge_shared: GgmlType::Q6_K,
            mid_shared: GgmlType::Q6_K,
            edge_attn: GgmlType::Q4_K,
            mid_attn: GgmlType::Q4_K,
        },
        // generate_config.sh:97-105 — mini
        ApexTier::Mini => TierRules {
            edge_exp: GgmlType::Q3_K,
            near_exp: GgmlType::Q3_K,
            mid_exp: GgmlType::IQ2_S,
            edge_shared: GgmlType::Q5_K,
            mid_shared: GgmlType::Q4_K,
            edge_attn: GgmlType::Q4_K,
            mid_attn: GgmlType::Q3_K,
        },
    }
}

/// Layer-region for routed-expert / shared-expert tensors (5-layer
/// edge band). Mirrors `generate_config.sh:149-155` and `:158-162`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpRegion {
    /// `i <= 4 || i >= L_LAST - 4`, i.e. first/last 5 layers.
    Edge,
    /// `i <= 9 || i >= L_LAST - 9` (and not Edge), i.e. next 5 layers
    /// from each side.
    Near,
    /// Everything else: `[10..=(L_LAST-10)]`.
    Mid,
}

/// Compute the EXP/SHARED layer-region. The match is `Edge` then
/// `Near` then `Mid` — mirroring the bash if/elif/else order at
/// `generate_config.sh:149-155`.
///
/// Edge cases (per `generate_config.sh`'s integer arithmetic):
/// - For `n_layers <= 10` every layer is `Edge` or `Near`; `Mid` is
///   empty. The bash handles this implicitly (`EDGE_LO = L - 5` can be
///   negative; `i <= 4 || i >= -ve` reduces to just `i <= 4`). We
///   reproduce that behavior via `i32` arithmetic and saturating
///   comparisons.
/// - Very small models (`n_layers <= 5`) classify all layers as `Edge`.
pub fn exp_region(layer: usize, n_layers: u32) -> ExpRegion {
    let l = layer as i64;
    let last = n_layers as i64 - 1; // L_LAST

    // generate_config.sh:63-64: EDGE_HI=4; EDGE_LO=LAYERS-5
    // bash condition (line 149): i <= EDGE_HI || i >= EDGE_LO
    let edge_hi: i64 = 4;
    let edge_lo: i64 = last - 4; // == LAYERS - 5

    if l <= edge_hi || l >= edge_lo {
        return ExpRegion::Edge;
    }

    // generate_config.sh:65-66: NEAR_HI=9; NEAR_LO=LAYERS-10
    // bash condition (line 151): i <= NEAR_HI || i >= NEAR_LO
    let near_hi: i64 = 9;
    let near_lo: i64 = last - 9; // == LAYERS - 10

    if l <= near_hi || l >= near_lo {
        return ExpRegion::Near;
    }

    ExpRegion::Mid
}

/// Compute the SHARED layer-region. There is no NEAR band for shared
/// in mudler's recipe; shared collapses NEAR into MID. Mirrors
/// `generate_config.sh:158-162`.
pub fn shared_region(layer: usize, n_layers: u32) -> SharedRegion {
    let l = layer as i64;
    let last = n_layers as i64 - 1;
    let edge_hi: i64 = 4;
    let edge_lo: i64 = last - 4;

    if l <= edge_hi || l >= edge_lo {
        SharedRegion::Edge
    } else {
        SharedRegion::Mid
    }
}

/// Layer-region for shared-expert tensors (no NEAR band).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedRegion {
    /// First/last 5 layers.
    Edge,
    /// Everything else.
    Mid,
}

/// Compute the ATTN layer-region. Attention uses a **3-wide** edge band,
/// not 5-wide. Mirrors `generate_config.sh:165-169`.
///
/// Bash condition: `i <= 2 || i >= LAYERS - 3`.
pub fn attn_region(layer: usize, n_layers: u32) -> AttnRegion {
    let l = layer as i64;
    // generate_config.sh:165 — i <= 2 || i >= LAYERS - 3
    let layers = n_layers as i64;
    if l <= 2 || l >= layers - 3 {
        AttnRegion::Edge
    } else {
        AttnRegion::Mid
    }
}

/// Layer-region for attention tensors (3-wide edge band).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnRegion {
    /// First/last 3 layers.
    Edge,
    /// Everything else.
    Mid,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At quality tier on a 40-layer model, verify the 7-tuple against
    /// mudler's verbatim values (`generate_config.sh:70-78`).
    #[test]
    fn tier_rules_quality_matches_mudler() {
        let r = tier_rules(ApexTier::Quality);
        assert_eq!(r.edge_exp, GgmlType::Q6_K);
        assert_eq!(r.near_exp, GgmlType::Q5_K);
        assert_eq!(r.mid_exp, GgmlType::IQ4_XS);
        assert_eq!(r.edge_shared, GgmlType::Q8_0);
        assert_eq!(r.mid_shared, GgmlType::Q8_0);
        assert_eq!(r.edge_attn, GgmlType::Q6_K);
        assert_eq!(r.mid_attn, GgmlType::Q6_K);

        // i-quality shares the table.
        assert_eq!(r, tier_rules(ApexTier::IQuality));
    }

    /// Mini tier has the most asymmetric table — exercises every
    /// 7-tuple slot independently.
    #[test]
    fn tier_rules_mini_matches_mudler() {
        let r = tier_rules(ApexTier::Mini);
        assert_eq!(r.edge_exp, GgmlType::Q3_K);
        assert_eq!(r.near_exp, GgmlType::Q3_K);
        assert_eq!(r.mid_exp, GgmlType::IQ2_S);
        assert_eq!(r.edge_shared, GgmlType::Q5_K);
        assert_eq!(r.mid_shared, GgmlType::Q4_K);
        assert_eq!(r.edge_attn, GgmlType::Q4_K);
        assert_eq!(r.mid_attn, GgmlType::Q3_K);
    }

    /// Validate the layer-region partition for a 40-layer model (the
    /// `gemma4-26b-A4B` / `qwen35moe` default). Boundary layers are:
    ///   EDGE_EXP = [0..=4] + [35..=39]
    ///   NEAR_EXP = [5..=9] + [30..=34]
    ///   MID_EXP  = [10..=29]
    #[test]
    fn exp_region_40_layers_boundaries() {
        // EDGE on left
        for i in 0..=4 {
            assert_eq!(exp_region(i, 40), ExpRegion::Edge, "layer {i}");
        }
        // NEAR on left
        for i in 5..=9 {
            assert_eq!(exp_region(i, 40), ExpRegion::Near, "layer {i}");
        }
        // MID
        for i in 10..=29 {
            assert_eq!(exp_region(i, 40), ExpRegion::Mid, "layer {i}");
        }
        // NEAR on right (35-10=25..=35-6=29 wait recompute: L_LAST=39, NEAR_LO=L_LAST-9=30)
        for i in 30..=34 {
            assert_eq!(exp_region(i, 40), ExpRegion::Near, "layer {i}");
        }
        // EDGE on right (EDGE_LO=L_LAST-4=35)
        for i in 35..=39 {
            assert_eq!(exp_region(i, 40), ExpRegion::Edge, "layer {i}");
        }
    }

    /// Attention edge band is 3-wide, not 5-wide. For 40 layers:
    ///   EDGE_ATTN = [0..=2] + [37..=39]
    ///   MID_ATTN  = [3..=36]
    #[test]
    fn attn_region_40_layers_3_wide_edge() {
        for i in 0..=2 {
            assert_eq!(attn_region(i, 40), AttnRegion::Edge, "layer {i}");
        }
        for i in 3..=36 {
            assert_eq!(attn_region(i, 40), AttnRegion::Mid, "layer {i}");
        }
        for i in 37..=39 {
            assert_eq!(attn_region(i, 40), AttnRegion::Edge, "layer {i}");
        }
    }

    /// Shared has no NEAR band (collapses into MID). For 40 layers:
    ///   EDGE_SHARED = [0..=4] + [35..=39]
    ///   MID_SHARED  = [5..=34]
    #[test]
    fn shared_region_no_near_band() {
        for i in 0..=4 {
            assert_eq!(shared_region(i, 40), SharedRegion::Edge, "layer {i}");
        }
        for i in 5..=34 {
            assert_eq!(shared_region(i, 40), SharedRegion::Mid, "layer {i}");
        }
        for i in 35..=39 {
            assert_eq!(shared_region(i, 40), SharedRegion::Edge, "layer {i}");
        }
    }

    /// Edge-case: 30-layer model (gemma4-26b). L_LAST=29.
    ///   EDGE_EXP = [0..=4] + [25..=29]
    ///   NEAR_EXP = [5..=9] + [20..=24]
    ///   MID_EXP  = [10..=19]
    ///   ATTN_EDGE = [0..=2] + [27..=29]
    #[test]
    fn exp_region_30_layers_gemma4() {
        assert_eq!(exp_region(0, 30), ExpRegion::Edge);
        assert_eq!(exp_region(4, 30), ExpRegion::Edge);
        assert_eq!(exp_region(5, 30), ExpRegion::Near);
        assert_eq!(exp_region(9, 30), ExpRegion::Near);
        assert_eq!(exp_region(10, 30), ExpRegion::Mid);
        assert_eq!(exp_region(19, 30), ExpRegion::Mid);
        assert_eq!(exp_region(20, 30), ExpRegion::Near);
        assert_eq!(exp_region(24, 30), ExpRegion::Near);
        assert_eq!(exp_region(25, 30), ExpRegion::Edge);
        assert_eq!(exp_region(29, 30), ExpRegion::Edge);

        // Vendored config `gemma4_26b_mini.txt` cross-check:
        // blk.2.attn_q=Q4_K (EDGE), blk.3.attn_q=Q3_K (MID).
        assert_eq!(attn_region(2, 30), AttnRegion::Edge);
        assert_eq!(attn_region(3, 30), AttnRegion::Mid);
    }
}
