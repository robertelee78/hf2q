//! Speculative-decoding rejection sampler for Qwen3.5 MTP (ADR-013 P14).
//!
//! Wraps the verifier `Qwen35Model::forward_gpu_greedy` with an MTP draft
//! head + accept/reject loop:
//!
//! ```text
//!   step k:
//!     1. verifier.forward(token_k)         → hidden_k, logits_k
//!     2. token_{k+1} = argmax(logits_k)
//!     3. draft.forward_draft(hidden_k, embed(token_{k+1}))
//!                                          → draft_logits
//!     4. proposed_{k+2} = argmax(draft_logits)
//!     5. verifier.forward(token_{k+1})     → logits_{k+1}
//!     6. verified_{k+2} = argmax(logits_{k+1})
//!     7. if proposed_{k+2} == verified_{k+2}:
//!          ACCEPT → emit token_{k+1}, token_{k+2}; advance k by 2
//!        else:
//!          REJECT → emit only token_{k+1}; advance k by 1; resync state
//! ```
//!
//! At T=0 (greedy), acceptance reduces to argmax-equality, which is what
//! the ADR-013 P14 acceptance criterion pins:
//!
//!   "speculative decoding does not change output logits vs. greedy
//!    single-token decode at temperature 0. Throughput improvement
//!    measured on a small-fixture generate; must be positive (≥ 10%)
//!    on at least one prompt set."
//!
//! # Status (2026-04-26)
//!
//! This module ships the **integration scaffold** — public types, the
//! `SpecDecode::run` entry point with its argument shape, and the
//! presence-gated fall-through to the existing greedy decode path. The
//! `forward_draft` GPU implementation it depends on is not yet landed
//! (it requires Metal-encoded MTP block forward + MTP KV slot allocation
//! + a verified MTP-bearing GGUF on disk to parity-test against).
//!
//! When `model.mtp_full` is `None` (the apex / default-converted GGUF
//! case that ALL on-disk Qwen3.5/3.6 GGUFs hit today), `SpecDecode::run`
//! deterministically falls through to a clean error so callers know to
//! use the regular greedy path. When `mtp_full` is `Some(...)`, the
//! current scaffold returns an explicit `not-yet-implemented` error
//! pointing at this file — NOT a silent no-op or a degraded result.
//! Per feedback_no_shortcuts.md, "no stubs" means a stub that quietly
//! returns wrong output is worse than an explicit unimplemented error.

use anyhow::{anyhow, Result};

use super::kv_cache::HybridKvCache;
use super::model::Qwen35Model;

/// Speculative-decoding sampler over a verifier model with an MTP
/// draft head.
pub struct SpecDecode<'a> {
    pub verifier: &'a Qwen35Model,
}

impl<'a> SpecDecode<'a> {
    /// Construct a sampler. Does not validate that `verifier.mtp_full`
    /// is populated — callers can opt into spec-decode at run-time and
    /// fall through to greedy when MTP is absent.
    pub fn new(verifier: &'a Qwen35Model) -> Self {
        Self { verifier }
    }

    /// Returns `true` if the wrapped verifier has fully-loaded MTP
    /// weights (the `HF2Q_QWEN35_KEEP_MTP=1` convert path landed and
    /// the GGUF was loaded).
    pub fn has_mtp(&self) -> bool {
        self.verifier.mtp_full.is_some()
    }

    /// Greedy speculative-decoding loop over `prompt_tokens`, generating
    /// up to `max_new` new tokens.
    ///
    /// # Arguments
    ///
    /// * `prompt_tokens` — pre-tokenized prompt (caller has already
    ///   applied chat template).
    /// * `max_new` — max generated tokens (excludes prompt).
    /// * `kv_cache` — fresh `HybridKvCache` sized for at least
    ///   `prompt.len() + max_new + 1` slots (the +1 is for MTP K/V).
    /// * `eos` — stop token id (typically 151645 for Qwen3.5/3.6).
    ///
    /// # Errors
    ///
    /// Returns `Err(...)` and emits a clear pointer to this module when:
    ///
    /// * `verifier.mtp_full.is_none()` — caller should fall back to
    ///   `Qwen35Model::forward_gpu_greedy` directly.
    /// * `forward_draft` is not yet implemented — see the module-level
    ///   doc for the unblocking sequence.
    pub fn run(
        &self,
        prompt_tokens: &[u32],
        max_new: usize,
        kv_cache: &mut HybridKvCache,
        eos: u32,
    ) -> Result<Vec<u32>> {
        if self.verifier.mtp_full.is_none() {
            return Err(anyhow!(
                "SpecDecode::run: verifier has no fully-loaded MTP weights \
                 (mtp_full is None) — convert the GGUF with \
                 HF2Q_QWEN35_KEEP_MTP=1 to enable speculative decoding, or \
                 fall back to Qwen35Model::forward_gpu_greedy for the regular \
                 greedy path. See src/inference/models/qwen35/spec_decode.rs."
            ));
        }
        // Suppress unused warnings until forward_draft lands.
        let _ = (prompt_tokens, max_new, kv_cache, eos);
        Err(anyhow!(
            "SpecDecode::run: forward_draft GPU implementation not yet \
             landed (ADR-013 P14 follow-on). MtpFullWeights are loaded \
             into model.mtp_full but the Metal-encoded MTP block forward \
             + MTP KV slot allocation + parity test against a real \
             KEEP_MTP-emitted GGUF are pending. See \
             src/inference/models/qwen35/spec_decode.rs and \
             docs/ADR-013-qwen35-inference.md P14 for the unblocking \
             sequence. Caller should fall back to \
             Qwen35Model::forward_gpu_greedy."
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::qwen35::{
        default_layer_types, Qwen35Config, Qwen35Variant,
    };

    fn tiny_dense_cfg() -> Qwen35Config {
        Qwen35Config {
            variant: Qwen35Variant::Dense,
            hidden_size: 8,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            linear_num_key_heads: 2,
            linear_num_value_heads: 2,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            layer_types: default_layer_types(4, 4),
            partial_rotary_factor: 0.25,
            rope_theta: 1e7,
            rotary_dim: 1,
            mrope_section: [1, 1, 0, 0],
            mrope_interleaved: true,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 64,
            vocab_size: 32,
            attn_output_gate: true,
            mtp_num_hidden_layers: 1,
            intermediate_size: Some(16),
            moe: None,
        }
    }

    /// `has_mtp` returns false on an empty (mtp_full=None) model.
    #[test]
    fn has_mtp_false_on_empty_model() {
        let cfg = tiny_dense_cfg();
        let model = Qwen35Model::empty_from_cfg(cfg);
        let sd = SpecDecode::new(&model);
        assert!(!sd.has_mtp(), "empty model has no MTP weights");
    }

    /// `run` returns a clear error (not silent no-op) when MTP weights
    /// are absent — caller must explicitly fall back to greedy.
    #[test]
    fn run_errors_when_mtp_absent() {
        use mlx_native::MlxDevice;
        let cfg = tiny_dense_cfg();
        let model = Qwen35Model::empty_from_cfg(cfg.clone());
        let device = MlxDevice::new().expect("device");
        let mut kv = HybridKvCache::new(&cfg, &device, 16, 1).expect("kv");
        let sd = SpecDecode::new(&model);
        let result = sd.run(&[1, 2, 3], 8, &mut kv, 100);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("HF2Q_QWEN35_KEEP_MTP") && msg.contains("forward_gpu_greedy"),
            "error message must guide caller to the greedy fallback or convert flag; got: {msg}"
        );
    }
}
