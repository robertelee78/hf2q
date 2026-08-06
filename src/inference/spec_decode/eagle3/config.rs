//! ADR-037 Phase E3b — Configuration for the EAGLE-3 drafter.
//!
//! Mirrors the subset of vLLM's `Eagle3LlamaForCausalLM` config that
//! affects the weight schema (peer code:
//! `/opt/vllm/vllm/model_executor/models/llama_eagle3.py:300-460`).
//!
//! This config drives the tensor manifest in `weights.rs`. Every
//! config knob that gates a tensor's presence (e.g. `norm_before_fc`
//! → presence of `input_norm.weight`; `use_qk_norm` → presence of
//! per-head q_norm/k_norm) is documented at the field with a peer-code
//! line reference and an inline `weights.rs` cross-link.
//!
//! The target model is Qwen 3.6 27B (our primary E2 training target),
//! but the schema is architecturally generic enough to support Llama
//! targets as well — the Qwen-specific gates (`use_qk_norm`) default
//! `true` and Llama-style trainers would set them `false`.

use anyhow::{anyhow, ensure, Result};

/// EAGLE-3 drafter architectural config.
///
/// All fields are validated by [`Self::validate`]. Construct via
/// `from_json_str` (when published) or directly for tests.
///
/// ## Field grouping
///
/// 1. **Core shape**: hidden_size, intermediate_size, head_dim, etc.
/// 2. **Vocabulary**: vocab_size, draft_vocab_size (may be smaller than
///    target's for "fast vocab projection" optimization per EAGLE-3
///    paper Sec. 4).
/// 3. **Multi-layer hidden**: `num_aux_hidden_states` (default 3 per
///    vLLM line 181); `target_hidden_size` (default == hidden_size).
/// 4. **Optional gates** (drive manifest conditional tensors):
///    - `norm_before_fc` — RMSNorm before `fc` (input_norm.weight)
///    - `fc_norm` — per-aux RMSNorms (fc_norm.{i}.weight × num_aux)
///    - `use_qk_norm` — per-head q_norm/k_norm (Qwen-style)
///    - `attention_bias` — q_proj/k_proj/v_proj/o_proj biases
///    - `tie_lm_head` — share lm_head weight with embed_tokens
#[derive(Debug, Clone)]
pub struct Eagle3DrafterConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    /// Drafter LM head output size. Default = `vocab_size`. Smaller
    /// values trigger the "fast vocab projection" path (see
    /// `draft_id_to_target_id` mapping in vLLM line 332-335).
    pub draft_vocab_size: usize,
    /// Hidden size of the TARGET model (typically == drafter
    /// hidden_size). `fc_input_size = target_hidden_size * num_aux`.
    pub target_hidden_size: usize,
    /// Number of auxiliary hidden states from target model layers.
    /// vLLM default: 3 (peer line 181). Matches the `Eagle3HiddenCollector`'s
    /// `target_layer_ids.len()` at runtime.
    pub num_aux_hidden_states: usize,
    pub rms_norm_eps: f32,
    /// If true, apply RMSNorm to the concatenated `[seq, fc_input_size]`
    /// hidden BEFORE the `fc` projection. Adds `input_norm.weight` of
    /// shape `[fc_input_size]` to the manifest. Peer: vLLM line 188-194.
    pub norm_before_fc: bool,
    /// If true, apply per-aux RMSNorm to each `[seq, target_hidden_size]`
    /// chunk BEFORE concatenation+fc. Adds `fc_norm.{i}.weight` of
    /// shape `[target_hidden_size]` for `i in 0..num_aux`. Peer:
    /// vLLM line 196-205.
    pub fc_norm: bool,
    /// If true, per-head Q-norm / K-norm tensors are present. Standard
    /// for Qwen-3 family targets; absent on Llama-style targets. Each
    /// is shape `[head_dim]`.
    pub use_qk_norm: bool,
    /// If true, q/k/v/o projections carry bias vectors. Peer:
    /// vLLM line 55 `qkv_bias = getattr(config, "attention_bias", False)`.
    pub attention_bias: bool,
    /// If true, drafter shares its lm_head with embed_tokens (tied
    /// weights). If false, an explicit `lm_head.weight` tensor is
    /// present in the manifest.
    pub tie_lm_head: bool,
    /// If true, the manifest includes a `draft_id_to_target_id`
    /// integer mapping tensor of shape `[draft_vocab_size]`. Peer:
    /// vLLM line 332-335.
    pub include_draft_id_mapping: bool,
    /// If true, drafter ships its own `embed_tokens.weight` in the
    /// safetensors file. If false, the drafter shares the target's
    /// embedding table (no `embed_tokens.weight` in manifest). Per
    /// vLLM peer behavior at `llama_eagle3.py:449-450`: missing
    /// embed_tokens is valid (drafter borrows target's embeddings).
    /// Default true for backward-compat; published EAGLE-3 checkpoints
    /// usually omit duplicate embeddings to save disk.
    pub has_own_embed_tokens: bool,
    /// RoPE base frequency. Qwen 3.6: 1_000_000; Llama 3: 500_000;
    /// older Llama/Mistral: 10_000. Must be > 0.
    pub rope_theta: f32,
    /// RoPE rotation dimension. For full rotation: equals head_dim
    /// (Qwen/Llama default). Partial-rotation models use a smaller
    /// value. Must be even (RoPE rotates pairs) and `<= head_dim`.
    pub rope_dim: usize,
    /// When true, the residual stream passed into the attention O-projection
    /// add is the NORMED hidden (`hidden_norm(fc_out)`). When false, the
    /// residual is `fc_out` directly (pre-norm).
    ///
    /// Per `vllm-project/speculators/.../eagle3/model_definitions.py:72-85`:
    /// RedHatAI Gemma 4 31B-it checkpoint sets `norm_before_residual=true`.
    /// Qwen 3.6 checkpoints omit the key → default `false`.
    pub norm_before_residual: bool,
}

impl Eagle3DrafterConfig {
    /// `fc_input_size = target_hidden_size * num_aux_hidden_states`.
    /// Drives the `fc.weight` second-dim and `input_norm.weight` length.
    #[inline]
    pub fn fc_input_size(&self) -> usize {
        self.target_hidden_size * self.num_aux_hidden_states
    }

    /// `q_proj.weight` first-dim (output channels): `num_q_heads * head_dim`.
    #[inline]
    pub fn q_proj_out(&self) -> usize {
        self.num_q_heads * self.head_dim
    }

    /// `k_proj.weight` / `v_proj.weight` first-dim: `num_kv_heads * head_dim`.
    #[inline]
    pub fn kv_proj_out(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// First-layer Q/K/V input width.
    ///
    /// vLLM line 53: `qkv_input_size = 2 * self.hidden_size if layer_idx == 0
    /// else self.hidden_size`. Since EAGLE-3 has ONLY one decoder layer
    /// (layer_idx == 0), this is ALWAYS `2 * hidden_size`. The factor 2
    /// comes from the concat of `input_layernorm(embeds)` + `hidden_norm(
    /// hidden_states)` along the last dim (vLLM line 104-106).
    #[inline]
    pub fn qkv_input_width(&self) -> usize {
        2 * self.hidden_size
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.hidden_size > 0, "hidden_size must be > 0");
        ensure!(self.intermediate_size > 0, "intermediate_size must be > 0");
        ensure!(self.head_dim > 0, "head_dim must be > 0");
        ensure!(self.num_q_heads > 0, "num_q_heads must be > 0");
        ensure!(self.num_kv_heads > 0, "num_kv_heads must be > 0");
        ensure!(
            self.num_q_heads % self.num_kv_heads == 0,
            "num_q_heads ({}) must be divisible by num_kv_heads ({}) — GQA invariant",
            self.num_q_heads,
            self.num_kv_heads,
        );
        // Codex /cfa E3 Critical (2026-05-22): checked multiply
        // for `q_proj.weight` first-dim. Adversarial config (e.g.
        // num_q_heads=usize::MAX / 2 + 1, head_dim=2) would wrap on
        // raw multiply at manifest-build time.
        //
        // ADR-038 G4-CFA-5 (2026-05-23): RELAXED — `q_proj_out` is NOT
        // required to equal `hidden_size`. Llama-style EAGLE-3 drafters
        // (e.g. RedHatAI gemma-4-31B-it-speculator.eagle3) use
        // `num_attention_heads=32`, `head_dim=256` → `q_proj_out=8192`
        // while `hidden_size=5376`. The `o_proj` weight maps
        // `[hidden_size, q_proj_out]` (= `[5376, 8192]`) — the kernel
        // already supports independent dims (see
        // `forward.rs::dispatch_eagle3_o_proj` line ~1435 passing
        // `cfg.q_proj_out()` as input width and `cfg.hidden_size` as
        // output width). The historical Qwen35 default happens to have
        // `q_proj_out == hidden_size` but this is NOT a kernel
        // requirement. The original invariant was over-tight.
        let _q_out = self.num_q_heads.checked_mul(self.head_dim).ok_or_else(|| {
            anyhow!(
                "num_q_heads * head_dim overflows usize (num_q_heads={}, head_dim={})",
                self.num_q_heads,
                self.head_dim
            )
        })?;
        // kv_proj_out() reuse — checked here once so the helper can
        // stay simple at call sites.
        let _kv_out = self
            .num_kv_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| {
                anyhow!(
                    "num_kv_heads * head_dim overflows usize (num_kv_heads={}, head_dim={})",
                    self.num_kv_heads,
                    self.head_dim
                )
            })?;
        ensure!(self.vocab_size > 0, "vocab_size must be > 0");
        ensure!(self.draft_vocab_size > 0, "draft_vocab_size must be > 0");
        ensure!(
            self.draft_vocab_size <= self.vocab_size,
            "draft_vocab_size ({}) must be <= vocab_size ({})",
            self.draft_vocab_size,
            self.vocab_size,
        );
        ensure!(
            self.target_hidden_size > 0,
            "target_hidden_size must be > 0"
        );
        ensure!(
            self.num_aux_hidden_states > 0,
            "num_aux_hidden_states must be > 0"
        );
        // Match Eagle3HiddenCollector's u64 written_mask limit so the
        // two pieces of E3 stay congruent.
        ensure!(
            self.num_aux_hidden_states <= 64,
            "num_aux_hidden_states must be <= 64 (matches Eagle3HiddenCollector)"
        );
        ensure!(self.rms_norm_eps > 0.0, "rms_norm_eps must be > 0");
        // Codex /cfa E4b.5b Minor (2026-05-22): finite-check on rope_theta
        // (rejects +inf/NaN that would later fail in rope_multi).
        ensure!(
            self.rope_theta.is_finite() && self.rope_theta > 0.0,
            "rope_theta ({}) must be finite and > 0",
            self.rope_theta
        );
        ensure!(self.rope_dim > 0, "rope_dim must be > 0");
        // Codex /cfa E4b.5b Minor: kernel requires head_dim even
        // (NeoX pairing depends on head_dim/2).
        ensure!(
            self.head_dim % 2 == 0,
            "head_dim ({}) must be even (NeoX RoPE pairing requires head_dim/2)",
            self.head_dim
        );
        // Codex /cfa E4b.5b Major (2026-05-22): kernel pairs NeoX dims
        // as (p, p + head_dim/2) — partial rotation `rope_dim < head_dim`
        // would rotate the wrong second-half coordinates. Require full
        // rotation until a partial-NeoX kernel ships.
        ensure!(
            self.rope_dim == self.head_dim,
            "rope_dim ({}) must equal head_dim ({}) — partial rotation not supported by apply_imrope NeoX pairing",
            self.rope_dim,
            self.head_dim
        );
        ensure!(
            self.rope_dim % 2 == 0,
            "rope_dim ({}) must be even (RoPE rotates pairs)",
            self.rope_dim
        );
        // Overflow guards: fc_input_size + qkv_input_width fit in usize
        // at every realistic shape, but the multiply could overflow on
        // adversarial input.
        self.target_hidden_size
            .checked_mul(self.num_aux_hidden_states)
            .ok_or_else(|| anyhow!("fc_input_size overflow"))?;
        self.hidden_size
            .checked_mul(2)
            .ok_or_else(|| anyhow!("qkv_input_width overflow"))?;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
pub(crate) mod tests {
    use super::*;

    /// Default config for a hypothetical Qwen 3.6 27B-style EAGLE-3
    /// drafter. The published target is 27B params; the drafter is
    /// 1-layer ~600M per ADR-037 §3. Shapes here are illustrative for
    /// the manifest tests — actual values come from the trained
    /// checkpoint config (post-E2).
    pub fn qwen35_default() -> Eagle3DrafterConfig {
        Eagle3DrafterConfig {
            hidden_size: 5120,
            intermediate_size: 13824,
            head_dim: 128,
            num_q_heads: 40,
            num_kv_heads: 8,
            vocab_size: 152064,
            draft_vocab_size: 152064,
            target_hidden_size: 5120,
            num_aux_hidden_states: 3,
            rms_norm_eps: 1e-6,
            norm_before_fc: false,
            fc_norm: true,
            use_qk_norm: true,
            attention_bias: false,
            tie_lm_head: false,
            include_draft_id_mapping: true,
            has_own_embed_tokens: true,
            rope_theta: 1_000_000.0,
            rope_dim: 128,
            norm_before_residual: false,
        }
    }

    #[test]
    fn adr_037_e3b_qwen35_default_validates_2026_05_22() {
        qwen35_default()
            .validate()
            .expect("default should validate");
    }

    #[test]
    fn adr_037_e3b_fc_input_size_formula_2026_05_22() {
        let cfg = qwen35_default();
        assert_eq!(cfg.fc_input_size(), 5120 * 3);
    }

    #[test]
    fn adr_037_e3b_qkv_input_width_is_2x_hidden_2026_05_22() {
        let cfg = qwen35_default();
        assert_eq!(cfg.qkv_input_width(), 2 * 5120);
    }

    #[test]
    fn adr_037_e3b_gqa_invariant_enforced_2026_05_22() {
        let mut cfg = qwen35_default();
        cfg.num_kv_heads = 7; // 40 % 7 != 0
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("GQA invariant"), "got: {err}");
    }

    /// ADR-038 G4-CFA-5 (2026-05-23): the `num_q_heads * head_dim ==
    /// hidden_size` invariant from CFA-3 was over-tight for Llama-style
    /// EAGLE-3 drafters (e.g. RedHatAI gemma-4-31B-it-speculator.eagle3
    /// uses `num_q_heads=32, head_dim=256, hidden_size=5376` →
    /// `q_proj_out=8192 != 5376`; the `o_proj` weight `[5376, 8192]`
    /// reduces the projected Q-stream back to hidden_size). The kernel
    /// itself supports independent `q_proj_out` and `hidden_size`
    /// (`forward.rs::dispatch_eagle3_o_proj`); only `validate()` was
    /// rejecting valid configs. This test now pins the RELAXED behavior:
    /// a Llama-style shape (q_proj_out != hidden_size) MUST validate.
    #[test]
    fn adr_038_g4_cfa5_llama_style_q_proj_out_validates_2026_05_23() {
        let mut cfg = qwen35_default();
        // Llama-style: num_q_heads * head_dim != hidden_size. The
        // RedHatAI Gemma 4 31B drafter is this shape. We pin
        // GQA-divisible (32 % 8 == 0) so only the (formerly) tight
        // `q_proj_out == hidden_size` check would fire.
        cfg.num_q_heads = 32; // 32 * 128 = 4096 != hidden_size 5120
        cfg.validate()
            .expect("Llama-style q_proj_out != hidden_size must validate (ADR-038 G4-CFA-5)");
    }

    #[test]
    fn adr_037_e3b_draft_vocab_size_at_most_target_vocab_2026_05_22() {
        let mut cfg = qwen35_default();
        cfg.draft_vocab_size = cfg.vocab_size + 1;
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("must be <="), "got: {err}");
    }

    #[test]
    fn adr_037_e3b_num_aux_at_most_64_2026_05_22() {
        let mut cfg = qwen35_default();
        cfg.num_aux_hidden_states = 65;
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("<= 64"), "got: {err}");
    }

    /// AC-G4-CFA-4.1 — norm_before_residual=false validates (Qwen default).
    #[test]
    fn g4_cfa4_norm_before_residual_false_validates_2026_05_22() {
        let mut cfg = qwen35_default();
        cfg.norm_before_residual = false;
        cfg.validate()
            .expect("norm_before_residual=false must validate");
    }

    /// AC-G4-CFA-4.2 — norm_before_residual=true validates (Gemma4/RedHatAI).
    #[test]
    fn g4_cfa4_norm_before_residual_true_validates_2026_05_22() {
        let mut cfg = qwen35_default();
        cfg.norm_before_residual = true;
        cfg.validate()
            .expect("norm_before_residual=true must validate");
    }

    /// AC-G4-CFA-4.3 — default_gemma4_eagle3_config shape matches RedHatAI schema.
    ///
    /// Per ADR-038 §3.4.2: hidden_size=5376, head_dim=256, num_q_heads=32, etc.
    /// Tests the config values documented in the ADR without needing a real model.
    ///
    /// ADR-038 G4-CFA-5 (2026-05-23): updated to the published Llama-style
    /// shape (num_q_heads=32 / num_kv_heads=16 → q_proj_out=8192) after
    /// relaxing the over-tight `q_proj_out == hidden_size` invariant in
    /// `validate()`. Pre-fix this test used the work-around values 21/7
    /// (which satisfied the tight invariant but would have mismatched the
    /// real checkpoint's `q_proj=[8192, 10752]` shape at load time).
    #[test]
    fn g4_cfa4_default_gemma4_eagle3_config_shape_2026_05_22() {
        // RedHatAI checkpoint schema (verified via safetensors header):
        //   q_proj.weight = [8192, 10752]  → num_q_heads=32, head_dim=256
        //   k_proj.weight = [4096, 10752]  → num_kv_heads=16, head_dim=256
        //   o_proj.weight = [5376, 8192]   → hidden_size=5376
        let cfg = Eagle3DrafterConfig {
            hidden_size: 5376,
            intermediate_size: 21504,
            head_dim: 256,
            num_q_heads: 32,  // 32 * 256 = 8192 = q_proj_out (Llama-style)
            num_kv_heads: 16, // 32/16=2 (GQA ratio 2:1, divisible ✓)
            vocab_size: 262144,
            draft_vocab_size: 32000,
            target_hidden_size: 5376,
            num_aux_hidden_states: 3,
            rms_norm_eps: 1e-6,
            norm_before_fc: false,
            fc_norm: false,
            use_qk_norm: false,
            attention_bias: false,
            tie_lm_head: false,
            include_draft_id_mapping: true,
            has_own_embed_tokens: true,
            rope_theta: 10000.0,
            rope_dim: 256,
            norm_before_residual: true,
        };
        cfg.validate()
            .expect("Gemma4 RedHatAI config shape must validate");
        assert_eq!(
            cfg.fc_input_size(),
            5376 * 3,
            "fc_input_size = 3 aux * 5376"
        );
        assert_eq!(cfg.norm_before_residual, true);
        assert_eq!(
            cfg.q_proj_out(),
            8192,
            "q_proj_out = num_q_heads(32) * head_dim(256)"
        );
        assert_eq!(
            cfg.kv_proj_out(),
            4096,
            "kv_proj_out = num_kv_heads(16) * head_dim(256)"
        );
    }
}
