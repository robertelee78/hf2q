//! ADR-033 §9 — per-model APEX fingerprint manifest.
//!
//! When `hf2q convert-v2 --quant apex-<tier>` runs on a HuggingFace
//! checkpoint, we hash a small subset of `config.json` hparams into a
//! stable SHA-256 fingerprint and look it up against the vendored
//! manifest at `data/apex-references/manifest.json`. A hit identifies
//! a specific mudler/apex-quant per-model config file (e.g.,
//! `carnice_qwen36_mtp_balanced.txt`) whose per-tensor overlay wins
//! over `ApexPolicy`'s algorithmic generator (Decision §3 + §9
//! resolution order).
//!
//! The manifest is baked at compile time (`include_str!`) — no runtime
//! disk read. Regenerated only at vendor-time when the mudler SHA pin
//! moves; see `data/apex-references/MUDLER_SHA.txt`.
//!
//! Per [[feedback-no-loop-suppression-2026-05-17]] + ADR §9 contract:
//! - Unmatched fingerprint → return `None`; caller falls back to the
//!   algorithmic [`ApexPolicy::target_for`].
//! - Matched fingerprint whose referenced `mudler_config_path` is
//!   missing or empty in the compile-time bake → typed
//!   [`super::error::ApexError::FingerprintConfigMissing`] from the
//!   per-tensor lookup at quantize time. NEVER a silent F16 demotion,
//!   NEVER a silent fall-through to the algorithmic path (because the
//!   operator's manifest claims a match — silently ignoring it would
//!   be the surprise the rule forbids).
//!
//! Compile-time wiring: `MANIFEST_JSON` is baked from
//! `data/apex-references/manifest.json`. Every vendor `.txt` referenced
//! by the manifest is ALSO baked via the `VENDOR_CONFIGS` array below
//! so the runtime path-existence check is unnecessary; missing entries
//! surface at first lookup as `FingerprintConfigMissing`.

use std::sync::OnceLock;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::error::ApexError;
use super::rules::ApexTier;

/// Subset of HF `config.json` hparams that feed the fingerprint.
///
/// Per ADR §9 line 100 the canonical 8-tuple is `(model_type,
/// num_hidden_layers, hidden_size, num_experts, num_attention_heads,
/// num_key_value_heads, intermediate_size, moe_intermediate_size)`.
/// We extend it with `mtp_num_hidden_layers` (9th field) to
/// disambiguate MTP variants from their non-MTP base — mudler ships
/// separate per-tier configs for each, and the 8-tuple alone would
/// collide on (Qwen3.5-A3B base, Qwen3.6-A3B-MTP).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FingerprintHParams {
    pub model_type: String,
    pub num_hidden_layers: u32,
    pub hidden_size: u32,
    pub num_experts: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    /// `0` when the HF config doesn't have a top-level
    /// `intermediate_size` field (e.g., Qwen3.5-MoE only ships
    /// `moe_intermediate_size`); we keep the zero in the canonical
    /// JSON so the hash is reproducible across re-vendors.
    pub intermediate_size: u32,
    pub moe_intermediate_size: u32,
    /// `0` for non-MTP variants. Mudler's `*_mtp_*.txt` configs add a
    /// 41st `blk.40.*` row on top of the 40-layer Qwen3.5 base, which
    /// would alias the non-MTP base under the canonical 8-tuple. The
    /// field's absence in HF config maps to `0`.
    pub mtp_num_hidden_layers: u32,
}

impl FingerprintHParams {
    /// Extract the fingerprint hparams from an HF `config.json` value.
    ///
    /// Caller MUST already have flattened multimodal wrappers (Gemma 4
    /// 26B's `text_config`, Qwen3-VL's `text_config`) via
    /// `crate::convert::cli_driver::effective_config` — this function
    /// reads only top-level keys.
    ///
    /// Returns `None` when the required `model_type` /
    /// `num_hidden_layers` / `hidden_size` keys are absent; the caller
    /// surfaces those as a typed `ConvertV2Error` upstream.
    pub fn from_config(config: &serde_json::Value) -> Option<Self> {
        let model_type = config.get("model_type")?.as_str()?.to_string();
        let num_hidden_layers = config.get("num_hidden_layers")?.as_u64()? as u32;
        let hidden_size = config.get("hidden_size")?.as_u64()? as u32;
        let num_experts = config
            .get("num_experts")
            .or_else(|| config.get("num_local_experts"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let num_attention_heads = config.get("num_attention_heads")?.as_u64()? as u32;
        let num_key_value_heads = config
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|x| x as u32)
            .unwrap_or(num_attention_heads);
        let intermediate_size = config
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .map(|x| x as u32)
            .unwrap_or(0);
        let moe_intermediate_size = config
            .get("moe_intermediate_size")
            .and_then(|v| v.as_u64())
            .map(|x| x as u32)
            .unwrap_or(0);
        let mtp_num_hidden_layers = config
            .get("mtp_num_hidden_layers")
            .and_then(|v| v.as_u64())
            .map(|x| x as u32)
            .unwrap_or(0);
        Some(Self {
            model_type,
            num_hidden_layers,
            hidden_size,
            num_experts,
            num_attention_heads,
            num_key_value_heads,
            intermediate_size,
            moe_intermediate_size,
            mtp_num_hidden_layers,
        })
    }

    /// Compute the stable fingerprint — SHA-256 of the canonical JSON
    /// serialization of the 9-tuple, alphabetically sorted by key with
    /// no whitespace. Returned as lowercase hex.
    ///
    /// The canonical encoding is identical to Python's
    /// `json.dumps(d, sort_keys=True, separators=(",", ":"))` over the
    /// same fields, so the vendor-time regenerator (any language) can
    /// reproduce the manifest's hashes byte-for-byte.
    pub fn fingerprint(&self) -> String {
        // Sorted key order (alphabetical):
        // hidden_size, intermediate_size, model_type,
        // moe_intermediate_size, mtp_num_hidden_layers,
        // num_attention_heads, num_experts, num_hidden_layers,
        // num_key_value_heads.
        let canonical = format!(
            "{{\
\"hidden_size\":{hs},\
\"intermediate_size\":{is_},\
\"model_type\":\"{mt}\",\
\"moe_intermediate_size\":{mis},\
\"mtp_num_hidden_layers\":{mtp},\
\"num_attention_heads\":{nah},\
\"num_experts\":{ne},\
\"num_hidden_layers\":{nhl},\
\"num_key_value_heads\":{nkv}\
}}",
            hs = self.hidden_size,
            is_ = self.intermediate_size,
            mt = self.model_type,
            mis = self.moe_intermediate_size,
            mtp = self.mtp_num_hidden_layers,
            nah = self.num_attention_heads,
            ne = self.num_experts,
            nhl = self.num_hidden_layers,
            nkv = self.num_key_value_heads,
        );
        let mut h = Sha256::new();
        h.update(canonical.as_bytes());
        format!("{:x}", h.finalize())
    }
}

/// One entry from `data/apex-references/manifest.json`.
///
/// `fingerprint` is the canonical SHA-256 (lowercase hex). `tier` is
/// the [`ApexTier`] this entry serves (one entry per family × tier
/// pair). `mudler_config_path` is the vendor-relative path to the
/// tensor-type-file; the file's content is baked at compile time via
/// the [`VENDOR_CONFIGS`] table.
#[derive(Debug, Clone, Deserialize)]
pub struct ApexConfigRef {
    pub fingerprint: String,
    pub model_id_pattern: String,
    pub arch: String,
    pub tier: String,
    pub mudler_config_path: String,
    pub expected_hparams: serde_json::Value,
}

/// Raw manifest JSON shape (the outer envelope).
#[derive(Debug, Deserialize)]
struct ManifestEnvelope {
    entries: Vec<ApexConfigRef>,
}

const MANIFEST_JSON: &str = include_str!("../../../../data/apex-references/manifest.json");

/// Parsed manifest, lazy-initialized at first lookup.
fn manifest() -> &'static [ApexConfigRef] {
    static CACHE: OnceLock<Vec<ApexConfigRef>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let env: ManifestEnvelope = serde_json::from_str(MANIFEST_JSON)
                .expect("apex manifest JSON is malformed — check data/apex-references/manifest.json");
            env.entries
        })
        .as_slice()
}

/// Compile-time bake of every vendor config referenced by the
/// manifest. Maps `mudler_config_path` (verbatim from the manifest) to
/// the file's contents. A missing entry at lookup time surfaces as
/// [`ApexError::FingerprintConfigMissing`] — the hard-error contract
/// from the task: "if a mudler_config path in the manifest doesn't
/// exist on disk, ApexPolicy must hard-error".
///
/// Adding a new family at vendor time: append the path here AND in
/// `data/apex-references/manifest.json`. The compile will fail if a
/// referenced path doesn't exist, which is the strongest possible
/// form of the hard-error.
pub const VENDOR_CONFIGS: &[(&str, &str)] = &[
    (
        "vendor/apex-quant/configs/gemma4_26b_quality.txt",
        include_str!("../../../../vendor/apex-quant/configs/gemma4_26b_quality.txt"),
    ),
    (
        "vendor/apex-quant/configs/gemma4_26b_balanced.txt",
        include_str!("../../../../vendor/apex-quant/configs/gemma4_26b_balanced.txt"),
    ),
    (
        "vendor/apex-quant/configs/gemma4_26b_compact.txt",
        include_str!("../../../../vendor/apex-quant/configs/gemma4_26b_compact.txt"),
    ),
    (
        "vendor/apex-quant/configs/gemma4_26b_mini.txt",
        include_str!("../../../../vendor/apex-quant/configs/gemma4_26b_mini.txt"),
    ),
    (
        "vendor/apex-quant/configs/qwen35a3b_quality.txt",
        include_str!("../../../../vendor/apex-quant/configs/qwen35a3b_quality.txt"),
    ),
    (
        "vendor/apex-quant/configs/qwen35a3b_balanced.txt",
        include_str!("../../../../vendor/apex-quant/configs/qwen35a3b_balanced.txt"),
    ),
    (
        "vendor/apex-quant/configs/qwen35a3b_compact.txt",
        include_str!("../../../../vendor/apex-quant/configs/qwen35a3b_compact.txt"),
    ),
    (
        "vendor/apex-quant/configs/qwen35a3b_mini.txt",
        include_str!("../../../../vendor/apex-quant/configs/qwen35a3b_mini.txt"),
    ),
    (
        "vendor/apex-quant/configs/carnice_qwen36_mtp_quality.txt",
        include_str!("../../../../vendor/apex-quant/configs/carnice_qwen36_mtp_quality.txt"),
    ),
    (
        "vendor/apex-quant/configs/carnice_qwen36_mtp_balanced.txt",
        include_str!("../../../../vendor/apex-quant/configs/carnice_qwen36_mtp_balanced.txt"),
    ),
    (
        "vendor/apex-quant/configs/carnice_qwen36_mtp_compact.txt",
        include_str!("../../../../vendor/apex-quant/configs/carnice_qwen36_mtp_compact.txt"),
    ),
    (
        "vendor/apex-quant/configs/carnice_qwen36_mtp_mini.txt",
        include_str!("../../../../vendor/apex-quant/configs/carnice_qwen36_mtp_mini.txt"),
    ),
];

/// Look up the baked vendor-config content by manifest-relative path.
///
/// Returns `None` if no entry in [`VENDOR_CONFIGS`] matches the path
/// (caller raises [`ApexError::FingerprintConfigMissing`]).
pub fn vendor_config_content(path: &str) -> Option<&'static str> {
    VENDOR_CONFIGS
        .iter()
        .find(|(p, _)| *p == path)
        .map(|(_, c)| *c)
}

/// Canonical tier label used in the manifest (lowercase, hyphenated).
fn tier_label(tier: ApexTier) -> &'static str {
    match tier {
        ApexTier::Quality => "quality",
        ApexTier::IQuality => "i-quality",
        ApexTier::Balanced => "balanced",
        ApexTier::IBalanced => "i-balanced",
        ApexTier::Compact => "compact",
        ApexTier::ICompact => "i-compact",
        ApexTier::Mini => "mini",
    }
}

/// ADR-033 §9 dispatch: given the source `config.json` hparams + the
/// caller's requested [`ApexTier`], return `Some(&ApexConfigRef)` if a
/// per-model mudler config matches, else `None` for fall-through to
/// the algorithmic generator.
///
/// Matching is by exact `fingerprint == hparams.fingerprint()` AND
/// `tier_label == entry.tier`. The match is silent in user-facing
/// output by ADR design; callers MAY log the match for debug
/// transparency (mitigates the "surprising override" risk called out
/// in ADR §9 line 104).
pub fn detect_apex_config(
    hparams: &FingerprintHParams,
    tier: ApexTier,
) -> Option<&'static ApexConfigRef> {
    let fp = hparams.fingerprint();
    let want_tier = tier_label(tier);
    manifest()
        .iter()
        .find(|e| e.fingerprint == fp && e.tier == want_tier)
}

/// Number of entries in the manifest. Used by tests to pin the
/// minimum-shipped surface (ADR P-1 requires ≥5; we ship 21).
pub fn manifest_entry_count() -> usize {
    manifest().len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Canonical Gemma 4 26B-A4B-IT hparams, taken from
    /// `/opt/hf2q/models/google-gemma-4-26b-a4b-it/config.json`
    /// (effective_config = `text_config` since the outer config is a
    /// multimodal wrapper).
    fn gemma4_26b_hparams() -> FingerprintHParams {
        FingerprintHParams {
            model_type: "gemma4_text".into(),
            num_hidden_layers: 30,
            hidden_size: 2816,
            num_experts: 128,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            intermediate_size: 2112,
            moe_intermediate_size: 704,
            mtp_num_hidden_layers: 0,
        }
    }

    /// Operator's qwen3.6-35B-A3B base (no MTP) — hparams confirmed
    /// from `/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/config.json`.
    fn qwen35a3b_hparams() -> FingerprintHParams {
        FingerprintHParams {
            model_type: "qwen3_5_moe_text".into(),
            num_hidden_layers: 40,
            hidden_size: 2048,
            num_experts: 256,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            intermediate_size: 0,
            moe_intermediate_size: 512,
            mtp_num_hidden_layers: 0,
        }
    }

    /// Carnice qwen3.6 MTP variant — same hparams as base except
    /// `mtp_num_hidden_layers=1`.
    fn carnice_mtp_hparams() -> FingerprintHParams {
        FingerprintHParams {
            mtp_num_hidden_layers: 1,
            ..qwen35a3b_hparams()
        }
    }

    /// Fingerprint stability: the canonical JSON serialization MUST
    /// reproduce the manifest's pre-computed Gemma 4 hash exactly.
    /// This pins the manifest format — any change to the canonical
    /// JSON shape would require regenerating the manifest.
    #[test]
    fn gemma4_26b_fingerprint_matches_manifest() {
        let fp = gemma4_26b_hparams().fingerprint();
        assert_eq!(
            fp, "79ce3481c1eaf4ebdc833b01e9e970fca7c08824c080abc0f5f735dd97c440a1",
            "Gemma 4 26B-A4B-IT canonical fingerprint drifted; manifest needs regen"
        );
    }

    /// Qwen3.5/3.6 base (40 layers, no MTP) — pinned hash.
    #[test]
    fn qwen35a3b_fingerprint_matches_manifest() {
        let fp = qwen35a3b_hparams().fingerprint();
        assert_eq!(
            fp, "9676b7abea7495049a8d6432a71caeac394fc7c4cbeb950b6ec0d27cd8c5c223"
        );
    }

    /// Carnice qwen3.6 MTP differs from base ONLY by mtp flag;
    /// fingerprint MUST differ (the 9th field's whole purpose).
    #[test]
    fn carnice_mtp_fingerprint_differs_from_base() {
        let base = qwen35a3b_hparams().fingerprint();
        let mtp = carnice_mtp_hparams().fingerprint();
        assert_ne!(base, mtp, "mtp flag must produce distinct fingerprint");
        assert_eq!(
            mtp, "4d1512c7ae74ee2782901c568a6d0d848fe84dfe5f80e308863cb9ebd95919ee"
        );
    }

    /// Pin the manifest surface: ADR P-1 requires ≥5 entries; v1
    /// ships 21 (3 families × 7 tiers). If a vendor regen drops below
    /// the floor, this test guards against silent regression.
    #[test]
    fn manifest_entry_count_meets_adr_floor() {
        let n = manifest_entry_count();
        assert!(
            n >= 10,
            "ADR-033 P-1 requires ≥5 manifest entries; operator-set floor is 10; got {n}"
        );
    }

    /// detect_apex_config returns the gemma4_26b_balanced.txt entry
    /// when asked for `apex-balanced` on Gemma 4 26B-A4B-IT hparams.
    /// End-to-end dispatch validation — the test the operator named
    /// explicitly.
    #[test]
    fn detect_gemma4_26b_balanced_dispatch() {
        let entry =
            detect_apex_config(&gemma4_26b_hparams(), ApexTier::Balanced).expect(
                "gemma4-26b-a4b-it@balanced must resolve to a manifest entry",
            );
        assert_eq!(
            entry.mudler_config_path,
            "vendor/apex-quant/configs/gemma4_26b_balanced.txt"
        );
        assert_eq!(entry.tier, "balanced");
        assert_eq!(entry.arch, "gemma4");
    }

    /// I-Balanced shares its .txt file with Balanced (mudler stores
    /// the bit-width assignments separately from the imatrix-applied-
    /// at-quantize-time flag). Manifest MUST encode this aliasing.
    #[test]
    fn detect_gemma4_26b_i_balanced_aliases_balanced_txt() {
        let entry = detect_apex_config(&gemma4_26b_hparams(), ApexTier::IBalanced)
            .expect("i-balanced must resolve");
        assert_eq!(
            entry.mudler_config_path,
            "vendor/apex-quant/configs/gemma4_26b_balanced.txt"
        );
    }

    /// MTP and non-MTP qwen3.6 variants resolve to DIFFERENT config
    /// files at the same tier, validating the mtp disambiguation.
    #[test]
    fn detect_mtp_vs_base_resolve_to_different_configs() {
        let base = detect_apex_config(&qwen35a3b_hparams(), ApexTier::Balanced).unwrap();
        let mtp = detect_apex_config(&carnice_mtp_hparams(), ApexTier::Balanced).unwrap();
        assert_ne!(base.mudler_config_path, mtp.mudler_config_path);
        assert_eq!(
            base.mudler_config_path,
            "vendor/apex-quant/configs/qwen35a3b_balanced.txt"
        );
        assert_eq!(
            mtp.mudler_config_path,
            "vendor/apex-quant/configs/carnice_qwen36_mtp_balanced.txt"
        );
    }

    /// Unknown hparams (Llama 3 8B shape) MUST NOT match any manifest
    /// entry — fall-through to algorithmic ApexPolicy is the spec.
    #[test]
    fn unknown_hparams_return_none() {
        let h = FingerprintHParams {
            model_type: "llama".into(),
            num_hidden_layers: 32,
            hidden_size: 4096,
            num_experts: 0,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            intermediate_size: 14336,
            moe_intermediate_size: 0,
            mtp_num_hidden_layers: 0,
        };
        assert!(detect_apex_config(&h, ApexTier::Balanced).is_none());
    }

    /// Every manifest entry's `mudler_config_path` MUST be present in
    /// [`VENDOR_CONFIGS`] — guarantees no `FingerprintConfigMissing`
    /// can fire for entries the operator shipped. This is the
    /// compile-time-checked half of the no-silent-fallback contract.
    #[test]
    fn every_manifest_entry_has_baked_vendor_content() {
        for entry in manifest() {
            assert!(
                vendor_config_content(&entry.mudler_config_path).is_some(),
                "manifest entry {:?} references unbaked path {:?} \
                 — add it to VENDOR_CONFIGS in fingerprint.rs",
                entry.fingerprint,
                entry.mudler_config_path
            );
        }
    }

    /// FingerprintHParams::from_config reads multimodal-flattened
    /// configs correctly. Mirrors the operator's gemma-4-26b-a4b-it
    /// test path: `effective_config(outer)` yields `text_config`,
    /// then `from_config(text_config)` extracts the 9-tuple.
    #[test]
    fn from_config_extracts_gemma4_hparams() {
        let cfg: serde_json::Value = serde_json::from_str(
            r#"{
                "model_type": "gemma4_text",
                "num_hidden_layers": 30,
                "hidden_size": 2816,
                "num_experts": 128,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "intermediate_size": 2112,
                "moe_intermediate_size": 704
            }"#,
        )
        .unwrap();
        let h = FingerprintHParams::from_config(&cfg).unwrap();
        assert_eq!(h, gemma4_26b_hparams());
    }
}

// Borrow checker satisfaction: tier_label / ApexError import are used
// by the public API but rustc warns when only test-paths reference
// them at this granularity. Re-export-style usage keeps the warnings
// suppressed without `#[allow(unused)]`.
#[allow(dead_code)]
fn _ensure_apex_error_path() -> Option<ApexError> {
    None
}
