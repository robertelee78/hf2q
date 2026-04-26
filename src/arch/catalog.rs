//! Per-arch tensor catalogs (ADR-012 Decision 20).
//!
//! A [`TensorCatalog`] is the hand-transcribed list of GGUF tensors an arch
//! emits, derived from the upstream loader's expectations
//! (e.g. `/opt/llama.cpp/src/models/qwen35moe.cpp`) per the same hand-
//! transcription rule used in P4 for metadata keys.  The catalog is the
//! single source of truth for both the smoke harness's tensor-count check
//! (Decision 16) and the MTP round-trip gate's expected-tensor list
//! (Decision 19).
//!
//! # Sovereignty rule
//!
//! Catalogs are derived by **reading** llama.cpp source, not by parsing a
//! GGUF produced by `convert_hf_to_gguf.py`.  See `feedback_hf2q_sovereignty.md`.

use std::fmt;

/// Specification for a single tensor an arch can emit.
///
/// `shape_pattern` and `dtype_pattern` are intentionally string-templated
/// rather than typed: real conversion outputs vary per-quant (Q4_0 vs F16
/// vs bf16) and per-hparam (hidden_size, num_experts, …).  The conformance
/// harness asserts presence by name; the per-shape assertions live in
/// dedicated unit tests where the synthetic dimensions are known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// GGUF tensor name with `{}` placeholders for layer index, expert
    /// index, etc.  E.g. `"blk.{L}.attn_q.weight"`.
    pub name: &'static str,

    /// Whether this tensor is per-layer (`true`) or global (`false`).
    pub per_layer: bool,

    /// Whether emission is gated by an arch feature flag (e.g. MTP, vision).
    /// Conformance assertions skip optional tensors when the corresponding
    /// flag is false on the [`crate::arch::ArchEntry`].
    pub optional: bool,
}

impl TensorSpec {
    pub const fn global(name: &'static str) -> Self {
        Self { name, per_layer: false, optional: false }
    }

    pub const fn per_layer(name: &'static str) -> Self {
        Self { name, per_layer: true, optional: false }
    }

    pub const fn optional_global(name: &'static str) -> Self {
        Self { name, per_layer: false, optional: true }
    }

    pub const fn optional_per_layer(name: &'static str) -> Self {
        Self { name, per_layer: true, optional: true }
    }
}

/// Hand-transcribed catalog of GGUF tensors for one arch.
///
/// Tensors are split into three buckets:
///
/// * `global` — appear once in the output GGUF (`token_embd.weight`,
///   `output_norm.weight`, …).
/// * `full_attention_layer` — emitted on layers whose `layer_types[L]` is
///   `"full_attention"`.
/// * `linear_attention_layer` — emitted on layers whose `layer_types[L]` is
///   `"linear_attention"`.  Empty slice for arches that have no linear
///   attention.
/// * `mtp` — emitted on a single block at index `num_hidden_layers` when
///   `mtp_num_hidden_layers > 0` and the [`crate::arch::ArchEntry::has_mtp`]
///   flag is set.  Empty slice otherwise.
#[derive(Debug, Clone)]
pub struct TensorCatalog {
    pub global: &'static [TensorSpec],
    pub full_attention_layer: &'static [TensorSpec],
    pub linear_attention_layer: &'static [TensorSpec],
    pub mtp: &'static [TensorSpec],
}

impl TensorCatalog {
    /// Total number of distinct tensor *patterns* in the catalog.  Used by the
    /// smoke harness's tensor-count sanity check; the actual emitted count in a
    /// real GGUF will be larger (multiplied by layer count, expert count, etc.).
    pub fn pattern_count(&self) -> usize {
        self.global.len()
            + self.full_attention_layer.len()
            + self.linear_attention_layer.len()
            + self.mtp.len()
    }
}

impl fmt::Display for TensorCatalog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TensorCatalog {{ global: {}, full_attn: {}, linear_attn: {}, mtp: {} }}",
            self.global.len(),
            self.full_attention_layer.len(),
            self.linear_attention_layer.len(),
            self.mtp.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_count_sums_buckets() {
        const G: &[TensorSpec] = &[TensorSpec::global("a"), TensorSpec::global("b")];
        const FA: &[TensorSpec] = &[TensorSpec::per_layer("c")];
        const LA: &[TensorSpec] = &[TensorSpec::per_layer("d"), TensorSpec::per_layer("e")];
        const MTP: &[TensorSpec] = &[TensorSpec::optional_per_layer("f")];

        let catalog = TensorCatalog {
            global: G,
            full_attention_layer: FA,
            linear_attention_layer: LA,
            mtp: MTP,
        };
        assert_eq!(catalog.pattern_count(), 6);
    }

    #[test]
    fn display_includes_all_buckets() {
        const G: &[TensorSpec] = &[TensorSpec::global("x")];
        const EMPTY: &[TensorSpec] = &[];
        let catalog = TensorCatalog {
            global: G,
            full_attention_layer: EMPTY,
            linear_attention_layer: EMPTY,
            mtp: EMPTY,
        };
        let s = format!("{catalog}");
        assert!(s.contains("global: 1"));
        assert!(s.contains("full_attn: 0"));
        assert!(s.contains("linear_attn: 0"));
        assert!(s.contains("mtp: 0"));
    }
}
