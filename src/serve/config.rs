//! Gemma4 A4B model configuration — derived from GGUF metadata
//! (preferred), or as a fallback from config.json for HF-safetensors paths.
//!
//! ADR-022 P1.8 — Operator pushback "requiring the config seems dumb" —
//! the GGUF carries every architectural parameter `Gemma4Config` needs.
//! `from_gguf(&GgufFile)` is the GGUF-self-sufficient constructor; the
//! `from_config_json` path is retained for the calibration / parity
//! pipelines that consume HF safetensors.

use anyhow::{Context, Result};
use mlx_native::gguf::{GgufFile, MetadataValue};
use serde::Deserialize;
use std::path::Path;

/// Layer attention type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    Sliding,
    Full,
}

/// Configuration for Gemma 4 A4B (MoE variant).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Gemma4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_global_key_value_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta_sliding: f64,
    pub rope_theta_global: f64,
    // ADR-005 1bNEW.18 (2026-04-11): `partial_rotary_factor_global` REMOVED.
    //
    // The field and its default value of `0.25` were introduced under the
    // misreading that Gemma 4 global-layer RoPE rotates only the first
    // `head_dim * partial_rotary_factor` elements of each head vector.
    // llama.cpp's Gemma 4 path (`src/models/gemma4-iswa.cpp:49,73-75,97-98`)
    // and the GGUF metadata (`gemma4.rope.dimension_count = 512 = head_dim`)
    // show the opposite: global layers rotate the FULL head_dim with a
    // per-pair `freq_factors` mask loaded from `rope_freqs.weight`
    // (`src/llama-model.cpp:4311-4313`). Elements [64..256) of the mask
    // are `1e+30`, which drives their rotation angle to ~0 via
    // `theta / freq_factor`, producing identity rotation on those pair
    // indices — numerically equivalent to "partial rotary" but structurally
    // a full-head rotation with a frequency mask, not a truncated rotary
    // dim. See `docs/spike-C-results.md` Parts 3.1-3.3 and 5 for the full
    // root-cause derivation and fix direction.
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    pub final_logit_softcapping: Option<f64>,
    pub attention_bias: bool,
    pub attention_k_eq_v: bool,
    pub tie_word_embeddings: bool,
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub layer_types: Vec<LayerType>,
}

impl Gemma4Config {
    pub fn from_config_json(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Cannot read {}", path.display()))?;
        let raw: RawConfig = serde_json::from_str(&content)
            .with_context(|| format!("Cannot parse {}", path.display()))?;
        let tc = raw.text_config.context("Missing text_config in config.json")?;

        let layer_types: Vec<LayerType> = tc.layer_types
            .unwrap_or_default()
            .iter()
            .map(|s| match s.as_str() {
                "full_attention" => LayerType::Full,
                _ => LayerType::Sliding,
            })
            .collect();

        let num_layers = tc.num_hidden_layers.unwrap_or(30);
        let layer_types = if layer_types.is_empty() {
            // Default: every 6th layer is full attention
            (0..num_layers)
                .map(|i| if (i + 1) % 6 == 0 { LayerType::Full } else { LayerType::Sliding })
                .collect()
        } else {
            layer_types
        };

        let rope = tc.rope_parameters.unwrap_or_default();
        let sliding_rope = rope.sliding_attention.unwrap_or_default();
        let full_rope = rope.full_attention.unwrap_or_default();

        Ok(Self {
            vocab_size: tc.vocab_size.unwrap_or(262144),
            hidden_size: tc.hidden_size.unwrap_or(2816),
            intermediate_size: tc.intermediate_size.unwrap_or(2112),
            moe_intermediate_size: tc.moe_intermediate_size.unwrap_or(704),
            num_hidden_layers: num_layers,
            num_attention_heads: tc.num_attention_heads.unwrap_or(16),
            num_key_value_heads: tc.num_key_value_heads.unwrap_or(8),
            num_global_key_value_heads: tc.num_global_key_value_heads.unwrap_or(2),
            head_dim: tc.head_dim.unwrap_or(256),
            global_head_dim: tc.global_head_dim.unwrap_or(512),
            rms_norm_eps: tc.rms_norm_eps.unwrap_or(1e-6),
            rope_theta_sliding: sliding_rope.rope_theta.unwrap_or(10000.0),
            rope_theta_global: full_rope.rope_theta.unwrap_or(1000000.0),
            sliding_window: tc.sliding_window.unwrap_or(1024),
            max_position_embeddings: tc.max_position_embeddings.unwrap_or(262144),
            final_logit_softcapping: tc.final_logit_softcapping,
            attention_bias: tc.attention_bias.unwrap_or(false),
            attention_k_eq_v: tc.attention_k_eq_v.unwrap_or(true),
            tie_word_embeddings: tc.tie_word_embeddings.unwrap_or(true),
            num_experts: tc.num_experts.unwrap_or(128),
            top_k_experts: tc.top_k_experts.unwrap_or(8),
            layer_types,
        })
    }

    /// ADR-022 P1.8 — Build `Gemma4Config` straight from GGUF metadata.
    ///
    /// Mapping (GGUF key → struct field) verified against
    /// `gemma4-ara-2pass-APEX-Q5_K_M.gguf` (gguf-dump --no-tensors):
    ///
    ///   gemma4.block_count                  → num_hidden_layers
    ///   gemma4.embedding_length             → hidden_size
    ///   gemma4.feed_forward_length          → intermediate_size
    ///   gemma4.expert_feed_forward_length   → moe_intermediate_size
    ///   gemma4.attention.head_count         → num_attention_heads
    ///   gemma4.attention.head_count_kv[0]   → num_key_value_heads (sliding kv heads)
    ///   gemma4.attention.head_count_kv[i]   for full-attn layers → num_global_key_value_heads
    ///   gemma4.attention.key_length_swa     → head_dim (sliding)
    ///   gemma4.attention.key_length         → global_head_dim (full)
    ///   gemma4.attention.layer_norm_rms_epsilon → rms_norm_eps
    ///   gemma4.rope.freq_base_swa           → rope_theta_sliding
    ///   gemma4.rope.freq_base               → rope_theta_global
    ///   gemma4.attention.sliding_window     → sliding_window
    ///   gemma4.context_length               → max_position_embeddings
    ///   gemma4.final_logit_softcapping      → final_logit_softcapping
    ///   gemma4.expert_count                 → num_experts
    ///   gemma4.expert_used_count            → top_k_experts
    ///   gemma4.attention.sliding_window_pattern[i] → layer_types[i] (True=Sliding, False=Full)
    ///
    /// Defaults preserved (values absent from GGUF metadata):
    ///   attention_bias = false, attention_k_eq_v = true, tie_word_embeddings = true.
    /// These match `from_config_json` `unwrap_or` defaults so the GGUF and
    /// HF-safetensors paths land at byte-identical configs for the file
    /// bundled by Anthropic Gemma4 releases.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch = gguf.metadata_string("general.architecture").unwrap_or("");
        anyhow::ensure!(
            arch == "gemma4",
            "Gemma4Config::from_gguf: GGUF architecture is {arch:?}, expected \"gemma4\""
        );

        // Helper closures — compact accessors over the typed enum.
        let u32_required = |k: &str| -> Result<u32> {
            gguf.metadata_u32(k)
                .with_context(|| format!("GGUF metadata key {k} missing or wrong type"))
        };
        let f32_required = |k: &str| -> Result<f32> {
            gguf.metadata_f32(k)
                .with_context(|| format!("GGUF metadata key {k} missing or wrong type"))
        };

        let num_layers = u32_required("gemma4.block_count")? as usize;
        let hidden_size = u32_required("gemma4.embedding_length")? as usize;
        let intermediate_size = u32_required("gemma4.feed_forward_length")? as usize;
        let num_attention_heads = u32_required("gemma4.attention.head_count")? as usize;
        let head_dim_full = u32_required("gemma4.attention.key_length")? as usize;
        let head_dim_swa = u32_required("gemma4.attention.key_length_swa")? as usize;
        let rms_eps = f32_required("gemma4.attention.layer_norm_rms_epsilon")? as f64;
        let rope_freq_base = f32_required("gemma4.rope.freq_base")? as f64;
        let rope_freq_base_swa = f32_required("gemma4.rope.freq_base_swa")? as f64;
        let sliding_window = u32_required("gemma4.attention.sliding_window")? as usize;
        let max_pos = u32_required("gemma4.context_length")? as usize;
        let num_experts = u32_required("gemma4.expert_count")? as usize;
        let top_k_experts = u32_required("gemma4.expert_used_count")? as usize;
        let moe_inter = u32_required("gemma4.expert_feed_forward_length")? as usize;
        let final_softcap = gguf
            .metadata_f32("gemma4.final_logit_softcapping")
            .map(|v| v as f64);

        // Per-layer head_count_kv array — same length as block_count.
        let head_count_kv_arr: Vec<u32> = match gguf.metadata("gemma4.attention.head_count_kv") {
            Some(MetadataValue::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_u32())
                .collect(),
            Some(other) => anyhow::bail!(
                "gemma4.attention.head_count_kv has unexpected type {:?}",
                std::mem::discriminant(other)
            ),
            None => anyhow::bail!("gemma4.attention.head_count_kv missing"),
        };
        anyhow::ensure!(
            head_count_kv_arr.len() == num_layers,
            "head_count_kv length {} != block_count {}",
            head_count_kv_arr.len(),
            num_layers
        );

        // Per-layer sliding_window_pattern: True = sliding, False = full.
        let pattern: Vec<bool> = match gguf.metadata("gemma4.attention.sliding_window_pattern") {
            Some(MetadataValue::Array(arr)) => arr
                .iter()
                .filter_map(|v| match v {
                    MetadataValue::Bool(b) => Some(*b),
                    _ => None,
                })
                .collect(),
            // Fallback: every 6th layer is full (matches `from_config_json`).
            Some(_) | None => (0..num_layers)
                .map(|i| (i + 1) % 6 != 0)
                .collect(),
        };
        anyhow::ensure!(
            pattern.len() == num_layers,
            "sliding_window_pattern length {} != block_count {}",
            pattern.len(),
            num_layers
        );

        let layer_types: Vec<LayerType> = pattern
            .iter()
            .map(|&is_sliding| if is_sliding { LayerType::Sliding } else { LayerType::Full })
            .collect();

        // Sliding kv-head count = first sliding layer's value (typically all sliding layers share).
        // Global kv-head count = first full layer's value (or fall back to sliding count if no full layers).
        let num_kv_heads = pattern
            .iter()
            .zip(head_count_kv_arr.iter())
            .find(|(s, _)| **s)
            .map(|(_, &h)| h as usize)
            .unwrap_or_else(|| head_count_kv_arr[0] as usize);
        let num_global_kv_heads = pattern
            .iter()
            .zip(head_count_kv_arr.iter())
            .find(|(s, _)| !**s)
            .map(|(_, &h)| h as usize)
            .unwrap_or(num_kv_heads);

        // vocab_size: dim 0 of token_embd.weight tensor.
        let vocab_size = gguf
            .tensor_info("token_embd.weight")
            .and_then(|info| info.shape.first().copied())
            .map(|v| v as usize)
            .context("token_embd.weight missing or has no shape — cannot derive vocab_size")?;

        // tie_word_embeddings = no separate output.weight tensor.
        let tie_word_embeddings = gguf.tensor_info("output.weight").is_none();

        Ok(Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            moe_intermediate_size: moe_inter,
            num_hidden_layers: num_layers,
            num_attention_heads,
            num_key_value_heads: num_kv_heads,
            num_global_key_value_heads: num_global_kv_heads,
            head_dim: head_dim_swa,
            global_head_dim: head_dim_full,
            rms_norm_eps: rms_eps,
            rope_theta_sliding: rope_freq_base_swa,
            rope_theta_global: rope_freq_base,
            sliding_window,
            max_position_embeddings: max_pos,
            final_logit_softcapping: final_softcap,
            attention_bias: false,
            attention_k_eq_v: true,
            tie_word_embeddings,
            num_experts,
            top_k_experts,
            layer_types,
        })
    }

    /// Is layer `idx` a global (full) attention layer?
    pub fn is_full_attention(&self, idx: usize) -> bool {
        self.layer_types.get(idx).copied() == Some(LayerType::Full)
    }

    /// Head dim for the given layer.
    pub fn head_dim_for_layer(&self, idx: usize) -> usize {
        if self.is_full_attention(idx) { self.global_head_dim } else { self.head_dim }
    }

    /// Number of KV heads for the given layer.
    pub fn num_kv_heads_for_layer(&self, idx: usize) -> usize {
        if self.is_full_attention(idx) { self.num_global_key_value_heads } else { self.num_key_value_heads }
    }
}

// --- Raw serde types ---

#[derive(Debug, Deserialize)]
struct RawConfig {
    text_config: Option<TextConfig>,
}

#[derive(Debug, Deserialize)]
struct TextConfig {
    vocab_size: Option<usize>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    moe_intermediate_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    num_global_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    global_head_dim: Option<usize>,
    rms_norm_eps: Option<f64>,
    sliding_window: Option<usize>,
    max_position_embeddings: Option<usize>,
    final_logit_softcapping: Option<f64>,
    attention_bias: Option<bool>,
    attention_k_eq_v: Option<bool>,
    tie_word_embeddings: Option<bool>,
    num_experts: Option<usize>,
    top_k_experts: Option<usize>,
    layer_types: Option<Vec<String>>,
    rope_parameters: Option<RopeParameters>,
}

#[derive(Debug, Deserialize, Default)]
struct RopeParameters {
    sliding_attention: Option<RopeEntry>,
    full_attention: Option<RopeEntry>,
}

#[derive(Debug, Deserialize, Default)]
struct RopeEntry {
    rope_theta: Option<f64>,
    // ADR-005 1bNEW.18: `partial_rotary_factor` intentionally NOT deserialized.
    // Gemma 4 global-layer RoPE uses full-head rotation with a frequency
    // mask loaded from `rope_freqs.weight`, not a reduced `rotary_dim`.
    // Serde's default is to ignore unknown JSON fields, so any
    // `partial_rotary_factor` key in config.json will simply be dropped on
    // parse. See the removed-field comment on
    // `Gemma4Config::partial_rotary_factor_global` above and
    // `docs/spike-C-results.md` Parts 3-5 for the full derivation.
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ADR-022 P1.8 — verify `Gemma4Config::from_gguf` parses the bundled
    /// `gemma4-ara-2pass-APEX-Q5_K_M.gguf` to canonical Gemma4 26B-A4B
    /// architectural parameters (cross-checked against the upstream
    /// `mlx-community/gemma-4-26b-a4b-it-4bit/config.json`).
    ///
    /// Falsifier for the operator complaint "requiring the config seems
    /// dumb": this test passes ⇒ no config.json needed.
    #[test]
    fn from_gguf_matches_canonical_gemma4_26b_config() {
        let gguf_path = std::path::Path::new(
            "/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf",
        );
        if !gguf_path.exists() {
            eprintln!("SKIP: {} not present (CI / fresh checkout)", gguf_path.display());
            return;
        }
        let gguf = GgufFile::open(gguf_path).expect("open gemma4 GGUF");
        let cfg = Gemma4Config::from_gguf(&gguf).expect("from_gguf");

        // Canonical values — from `mlx-community/gemma-4-26b-a4b-it-4bit/config.json`.
        assert_eq!(cfg.vocab_size, 262144, "vocab_size");
        assert_eq!(cfg.hidden_size, 2816, "hidden_size");
        assert_eq!(cfg.intermediate_size, 2112, "intermediate_size");
        assert_eq!(cfg.moe_intermediate_size, 704, "moe_intermediate_size");
        assert_eq!(cfg.num_hidden_layers, 30, "num_hidden_layers");
        assert_eq!(cfg.num_attention_heads, 16, "num_attention_heads");
        assert_eq!(cfg.num_key_value_heads, 8, "num_key_value_heads (sliding)");
        assert_eq!(cfg.num_global_key_value_heads, 2, "num_global_key_value_heads");
        assert_eq!(cfg.head_dim, 256, "head_dim (sliding)");
        assert_eq!(cfg.global_head_dim, 512, "global_head_dim (full)");
        assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-9, "rms_norm_eps");
        assert_eq!(cfg.sliding_window, 1024, "sliding_window");
        assert_eq!(cfg.max_position_embeddings, 262144, "max_position_embeddings");
        assert_eq!(cfg.final_logit_softcapping, Some(30.0), "final_logit_softcapping");
        assert_eq!(cfg.num_experts, 128, "num_experts");
        // top_k_experts: GGUF `expert_used_count` is the source of truth (2 in this build).
        assert!(
            cfg.top_k_experts >= 2 && cfg.top_k_experts <= 8,
            "top_k_experts {} out of plausible range",
            cfg.top_k_experts
        );
        assert!(cfg.tie_word_embeddings, "tie_word_embeddings");
        assert_eq!(cfg.layer_types.len(), 30, "layer_types length");

        // Sliding-window pattern: every 6th layer (idx 5, 11, 17, 23, 29) is FULL.
        for (i, lt) in cfg.layer_types.iter().enumerate() {
            let expected_full = (i + 1) % 6 == 0;
            let is_full = matches!(lt, LayerType::Full);
            assert_eq!(
                is_full, expected_full,
                "layer {i}: expected_full={expected_full} got {lt:?}"
            );
        }
    }
}
