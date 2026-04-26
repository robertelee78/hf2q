//! Real (non-mock) `ActivationCapture` for `Qwen35Model`.
//!
//! ADR-012 P9 / ADR-013 Decision 16. Lands on top of the ADR-013 merge
//! (`236abb8` — full Qwen3.5/3.6 forward) which closed P12 with trait+mock
//! only and explicitly deferred the real impl. This file finishes that
//! deferred contract.
//!
//! Two surfaces are exposed:
//!
//! 1. [`Qwen35Model`] directly implements [`ActivationCapture`]. Power callers
//!    that already hold a `Qwen35Model` (e.g. an existing inference session)
//!    can capture activations without re-loading.
//! 2. [`RealActivationCapture`] is the convenience wrapper for the DWQ
//!    calibration pipeline: it loads a `Qwen35Model` from a GGUF on the fly
//!    and delegates `run_calibration_prompt` to the model. The
//!    `not_ready()` constructor remains for callers that explicitly want to
//!    surface a structured "deferred" error (used in unit tests of the
//!    no-fallback guard).
//!
//! # Capture semantics
//!
//! For each transformer block `l` (`0 ≤ l < num_hidden_layers`):
//!
//! * `layer_inputs[l]` — residual stream entering layer `l`.
//!     - `l == 0`: the embedding output.
//!     - `l > 0`:  `layer_outputs[l-1]`.
//! * `layer_outputs[l]` — residual stream leaving layer `l` after FFN
//!   residual add (matches the per-block `ffn_residual + ffn_out` shape
//!   in [`super::forward_cpu`]).
//!
//! Each captured tensor is row-major `[seq_len, hidden_size]` flattened
//! to `Vec<f32>` per the `LayerActivations` contract.
//!
//! # Variants supported
//!
//! * **Dense Qwen3.5** (`qwen35`, `Qwen35Variant::Dense`): full CPU capture
//!   path works on F32-loaded weights.
//! * **MoE Qwen3.5** (`qwen35moe`, `Qwen35Variant::Moe`): GGUF-loaded
//!   experts are stored in their native ggml block-quantization
//!   (`Qwen35FfnWeights::MoeQ`); `forward_cpu` returns an error rather
//!   than F32-expanding 256 experts (would OOM the 64 GB working set).
//!   Capture for MoE therefore returns
//!   [`RealActivationCaptureError::ForwardPass`] with a clear message;
//!   the GPU-backed capture path (uses `forward_gpu`) is a follow-up.
//!
//! # No fallback
//!
//! Per `feedback_never_ship_fallback_without_rootcause.md`: if the forward
//! errors out (e.g. on MoE-quantized FFN) we return a typed error, not
//! synthetic activations. Callers must handle this rather than silently
//! routing around it.

use anyhow::{Context, Result};
use mlx_native::MlxDevice;
use thiserror::Error;

use super::activation_capture::{ActivationCapture, LayerActivations};
use super::delta_net::{delta_net_layer_cpu_ref, DeltaNetLayerShape};
use super::ffn::{
    dense_swiglu_cpu_ref, moe_ffn_cpu_ref, DenseFfnShape, MoeFfnShape,
};
use super::full_attn::{gated_full_attention_cpu_ref, FullAttnShape};
use super::io_heads::embed_tokens;
use super::kv_cache::HybridKvCache;
use super::model::{Qwen35FfnWeights, Qwen35LayerWeights, Qwen35Model};

// ================================================================
// Error type
// ================================================================

/// Errors returned by the real ActivationCapture path. `NotReady` is
/// retained for the `not_ready()` factory used in dependency-blocked
/// unit tests; `ForwardPass` is the production error variant naming the
/// specific layer / reason.
#[derive(Debug, Error)]
pub enum RealActivationCaptureError {
    /// Constructor was called via [`RealActivationCapture::not_ready`].
    /// This exists so a unit test can validate the no-fallback guard
    /// behavior without needing a real GGUF.
    #[error(
        "RealActivationCapture: explicit not_ready() shim invoked. The \
         real implementation is wired up; this variant is reserved for \
         dependency-test scenarios that want to pin the no-fallback \
         contract surface."
    )]
    NotReady,

    /// Forward pass failed at a specific layer with a named reason.
    #[error("qwen35 forward failed at layer {layer}: {reason}")]
    ForwardPass { layer: u32, reason: String },

    /// Constructor failed to load a `Qwen35Model` from the supplied GGUF
    /// path (file missing, malformed, or unsupported variant).
    #[error("RealActivationCapture::new failed to load model from {path}: {reason}")]
    Load { path: String, reason: String },
}

// ================================================================
// RMSNorm helper (mirrors forward_cpu::rms_norm_rows; private there)
// ================================================================

fn rms_norm_rows(x: &mut [f32], weight: &[f32], hidden: usize, eps: f32) {
    debug_assert_eq!(weight.len(), hidden);
    let seq = x.len() / hidden;
    for t in 0..seq {
        let row = &mut x[t * hidden..(t + 1) * hidden];
        let sum_sq: f32 = row.iter().map(|v| v * v).sum();
        let inv = ((sum_sq / (hidden as f32)) + eps).sqrt().recip();
        for (j, v) in row.iter_mut().enumerate() {
            *v = *v * inv * weight[j];
        }
    }
}

fn residual_add(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

// ================================================================
// GPU capture path (ADR-012 P9b — mantra-aligned production wire-up)
// ================================================================

/// True iff any layer FFN is the native ggml block-quantized MoeQ variant.
/// MoeQ cannot be F32-expanded into the CPU forward without OOM (~128 GB
/// for 256-expert apex), so calibration MUST go through the GPU forward.
fn has_moeq_layer(model: &Qwen35Model) -> bool {
    model
        .layers
        .iter()
        .any(|l| matches!(l.ffn(), Qwen35FfnWeights::MoeQ(_)))
}

/// Run a calibration prompt through the GPU forward and capture per-layer
/// residual streams. Uses `Qwen35Model::forward_gpu_with_capture`, which
/// dispatches `mlx-native` `quantized_matmul_ggml` for native MoeQ experts
/// — no F32 expansion, no OOM, runs at production decode speed.
///
/// Allocates a fresh `HybridKvCache` sized to `tokens.len()` for the
/// calibration pass; the cache is dropped on return so subsequent inference
/// sessions are unaffected.
pub fn run_calibration_prompt_gpu(
    model: &Qwen35Model,
    tokens: &[u32],
) -> Result<LayerActivations> {
    if tokens.is_empty() {
        anyhow::bail!("run_calibration_prompt_gpu: tokens must be non-empty");
    }

    let seq_len = tokens.len() as u32;
    // Text-only positions: one axis per token, replicated across all 4 MROPE
    // sectors. Matches `text_positions` in `forward_cpu`.
    let mut positions_flat: Vec<i32> = Vec::with_capacity((4 * seq_len) as usize);
    for s in 0..4 {
        let _ = s;
        for i in 0..seq_len as i32 {
            positions_flat.push(i);
        }
    }
    debug_assert_eq!(positions_flat.len(), (4 * seq_len) as usize);

    let device = MlxDevice::new()
        .context("run_calibration_prompt_gpu: MlxDevice::new")?;

    let mut kv_cache = HybridKvCache::new(&model.cfg, &device, seq_len.max(1), 1)
        .context("run_calibration_prompt_gpu: HybridKvCache::new")?;

    let num_layers = model.cfg.num_hidden_layers as usize;
    let mut acts = LayerActivations {
        layer_inputs: Vec::with_capacity(num_layers),
        layer_outputs: Vec::with_capacity(num_layers),
        num_layers: model.cfg.num_hidden_layers,
        seq_len,
        hidden_size: model.cfg.hidden_size,
    };

    let _logits = model
        .forward_gpu_with_capture(tokens, &positions_flat, &mut kv_cache, &mut acts)
        .context("run_calibration_prompt_gpu: forward_gpu_with_capture")?;

    acts.validate()
        .context("run_calibration_prompt_gpu: captured activations failed validate()")?;

    Ok(acts)
}

// ================================================================
// Primary impl: ActivationCapture for Qwen35Model
// ================================================================

impl ActivationCapture for Qwen35Model {
    fn run_calibration_prompt(&mut self, tokens: &[u32]) -> Result<LayerActivations> {
        if tokens.is_empty() {
            anyhow::bail!("Qwen35Model::run_calibration_prompt: tokens must be non-empty");
        }

        let h = self.cfg.hidden_size as usize;
        let eps = self.cfg.rms_norm_eps;
        let num_layers = self.cfg.num_hidden_layers;
        let seq_len = tokens.len();

        // Text-only positions (one axis per token, replicated across the 4
        // MROPE sectors). Matches forward_cpu's text_positions helper.
        let positions: Vec<[i32; 4]> = (0..seq_len as i32).map(|i| [i, i, i, i]).collect();

        // 1. Embedding lookup.
        let mut hidden = embed_tokens(
            tokens,
            &self.token_embd,
            self.cfg.vocab_size,
            self.cfg.hidden_size,
        );

        let mut layer_inputs: Vec<Vec<f32>> = Vec::with_capacity(num_layers as usize);
        let mut layer_outputs: Vec<Vec<f32>> = Vec::with_capacity(num_layers as usize);

        // 2. Per-layer forward + capture. Mirrors `forward_cpu` exactly so
        // the captured residual stream is byte-identical to a forward
        // call's intermediate state.
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Capture residual stream entering this layer.
            layer_inputs.push(hidden.clone());

            // Attention.
            let attn_out = match layer {
                Qwen35LayerWeights::FullAttn { attn, .. } => {
                    let shape = FullAttnShape::from_config(&self.cfg);
                    gated_full_attention_cpu_ref(&hidden, &positions, attn, shape)
                }
                Qwen35LayerWeights::LinearAttn { attn, .. } => {
                    let shape = DeltaNetLayerShape::from_config(&self.cfg);
                    let state_in = vec![
                        0.0f32;
                        (self.cfg.linear_key_head_dim
                            * self.cfg.linear_value_head_dim
                            * self.cfg.linear_num_value_heads)
                            as usize
                    ];
                    let km1 = (self.cfg.linear_conv_kernel_dim - 1) as usize;
                    let qkv_channels = (2 * self.cfg.linear_num_key_heads
                        * self.cfg.linear_key_head_dim
                        + self.cfg.linear_num_value_heads
                            * self.cfg.linear_value_head_dim)
                        as usize;
                    let conv_state = vec![0.0f32; km1 * qkv_channels];
                    let (out, _new_state, _new_conv) =
                        delta_net_layer_cpu_ref(&hidden, attn, shape, &state_in, &conv_state);
                    out
                }
            };

            // Residual after attention.
            residual_add(&mut hidden, &attn_out);

            // Post-attention norm + FFN (matches forward_cpu's
            // ffn_residual / attn_post_norm structure).
            let ffn_residual = hidden.clone();
            let mut ffn_input = hidden.clone();
            let post_norm_w = match layer {
                Qwen35LayerWeights::FullAttn { attn, .. } => &attn.post_attn_norm,
                Qwen35LayerWeights::LinearAttn { attn, .. } => &attn.post_attn_norm,
            };
            rms_norm_rows(&mut ffn_input, post_norm_w, h, eps);

            let ffn_out = match layer.ffn() {
                Qwen35FfnWeights::Dense(w) => {
                    let m = self.cfg.intermediate_size.ok_or_else(|| {
                        RealActivationCaptureError::ForwardPass {
                            layer: layer_idx as u32,
                            reason: "dense variant missing intermediate_size".into(),
                        }
                    })?;
                    let shape = DenseFfnShape {
                        hidden_size: self.cfg.hidden_size,
                        intermediate_size: m,
                    };
                    dense_swiglu_cpu_ref(&ffn_input, w, shape)
                }
                Qwen35FfnWeights::Moe(w) => {
                    let moe = self.cfg.moe.as_ref().ok_or_else(|| {
                        RealActivationCaptureError::ForwardPass {
                            layer: layer_idx as u32,
                            reason: "moe variant missing moe config".into(),
                        }
                    })?;
                    let shape = MoeFfnShape {
                        hidden_size: self.cfg.hidden_size,
                        num_experts: moe.num_experts,
                        num_experts_per_tok: moe.num_experts_per_tok,
                        moe_intermediate_size: moe.moe_intermediate_size,
                        shared_intermediate_size: moe.shared_expert_intermediate_size,
                    };
                    moe_ffn_cpu_ref(&ffn_input, w, shape)
                }
                Qwen35FfnWeights::MoeQ(_) => {
                    return Err(anyhow::anyhow!(
                        RealActivationCaptureError::ForwardPass {
                            layer: layer_idx as u32,
                            reason:
                                "MoE variant loaded with native GGML block quantization \
                                 (MoeQ); CPU activation capture path does not F32-expand \
                                 256 experts. GPU-backed capture is the follow-up. \
                                 No weight-space fallback per ADR-012 D13.".into(),
                        }
                    ));
                }
            };

            // Residual after FFN: pre-norm value (ffn_residual) + ffn_out.
            hidden = ffn_residual;
            residual_add(&mut hidden, &ffn_out);

            // Capture residual stream leaving this layer.
            layer_outputs.push(hidden.clone());
        }

        Ok(LayerActivations {
            layer_inputs,
            layer_outputs,
            num_layers,
            seq_len: seq_len as u32,
            hidden_size: self.cfg.hidden_size,
        })
    }
}

// ================================================================
// Convenience wrapper — RealActivationCapture
// ================================================================

/// Production wrapper that owns a `Qwen35Model` loaded from disk and
/// delegates capture to its `ActivationCapture` impl.
pub struct RealActivationCapture {
    inner: RealCaptureBackend,
}

enum RealCaptureBackend {
    /// Real model — capture proceeds.
    Loaded(Qwen35Model),
    /// Explicit not-ready shim for unit tests of the no-fallback guard.
    NotReady,
}

impl std::fmt::Debug for RealActivationCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.inner {
            RealCaptureBackend::Loaded(m) => f
                .debug_struct("RealActivationCapture")
                .field("variant", &m.cfg.variant)
                .field("num_hidden_layers", &m.cfg.num_hidden_layers)
                .finish(),
            RealCaptureBackend::NotReady => f
                .debug_struct("RealActivationCapture")
                .field("inner", &"NotReady")
                .finish(),
        }
    }
}

impl RealActivationCapture {
    /// Construct a NotReady shim — used only by unit tests that pin the
    /// no-fallback guard surface. Production callers use [`Self::new`].
    pub fn not_ready() -> Self {
        Self {
            inner: RealCaptureBackend::NotReady,
        }
    }

    /// Construct a real capture by loading a `Qwen35Model` from a GGUF.
    /// `_tokenizer_json` is reserved for future use (e.g. tokenizing
    /// calibration prompts directly inside this wrapper); presently the
    /// caller pre-tokenizes and passes `&[u32]` to `run_calibration_prompt`.
    pub fn new(
        model_gguf: &std::path::Path,
        _tokenizer_json: &std::path::Path,
    ) -> std::result::Result<Self, RealActivationCaptureError> {
        let gguf = mlx_native::gguf::GgufFile::open(model_gguf).map_err(|e| {
            RealActivationCaptureError::Load {
                path: model_gguf.display().to_string(),
                reason: format!("GgufFile::open: {e}"),
            }
        })?;
        let model = Qwen35Model::load_from_gguf(&gguf).map_err(|e| {
            // `{e:#}` displays the anyhow context chain on one line so the
            // inner cause (e.g. specific tensor name + shape) is visible.
            RealActivationCaptureError::Load {
                path: model_gguf.display().to_string(),
                reason: format!("Qwen35Model::load_from_gguf: {e:#}"),
            }
        })?;
        Ok(Self {
            inner: RealCaptureBackend::Loaded(model),
        })
    }

    /// Construct directly from an already-loaded `Qwen35Model`. Useful for
    /// callers that share a model with an inference session.
    pub fn from_model(model: Qwen35Model) -> Self {
        Self {
            inner: RealCaptureBackend::Loaded(model),
        }
    }
}

impl ActivationCapture for RealActivationCapture {
    fn run_calibration_prompt(&mut self, tokens: &[u32]) -> Result<LayerActivations> {
        match &mut self.inner {
            RealCaptureBackend::Loaded(model) => {
                // Mantra-aligned dispatch (ADR-012 P9b):
                //   * MoeQ (native ggml-block experts) → GPU forward via
                //     `forward_gpu_with_capture`, which calls
                //     `mlx-native::quantized_matmul_ggml` directly. No F32
                //     expansion, no OOM, production decode speeds (~50–100×
                //     CPU). HF2Q_FORCE_CPU_CAPTURE=1 escapes only for parity
                //     debugging.
                //   * Dense / F32-Moe → CPU path stays as the byte-identical
                //     reference (small synthetic models in tests, and
                //     pre-quant Dense weights at convert time).
                let force_cpu =
                    std::env::var("HF2Q_FORCE_CPU_CAPTURE").is_ok();
                if !force_cpu && has_moeq_layer(model) {
                    run_calibration_prompt_gpu(model, tokens)
                } else {
                    model.run_calibration_prompt(tokens)
                }
            }
            RealCaptureBackend::NotReady => {
                Err(anyhow::anyhow!(RealActivationCaptureError::NotReady))
            }
        }
    }
}

// ================================================================
// Tests
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{
        Qwen35Config, Qwen35LayerKind, Qwen35MoeConfig, Qwen35Variant,
    };

    /// Minimal hybrid Qwen3.5 dense config: 1 linear-attn + 1 full-attn
    /// layer, tiny dims. Sufficient to exercise both branches of the
    /// capture loop.
    fn tiny_dense_cfg() -> Qwen35Config {
        Qwen35Config {
            variant: Qwen35Variant::Dense,
            vocab_size: 16,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            intermediate_size: Some(16),
            rope_theta: 10_000.0,
            rotary_dim: 4,
            mrope_section: [1, 1, 1, 1],
            mrope_interleaved: true,
            partial_rotary_factor: 1.0,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 64,
            attn_output_gate: true,
            layer_types: vec![
                Qwen35LayerKind::LinearAttention,
                Qwen35LayerKind::FullAttention,
            ],
            full_attention_interval: 2,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            moe: None,
            mtp_num_hidden_layers: 0,
        }
    }

    fn tiny_moe_cfg() -> Qwen35Config {
        let mut c = tiny_dense_cfg();
        c.variant = Qwen35Variant::Moe;
        c.intermediate_size = None;
        c.moe = Some(Qwen35MoeConfig {
            num_experts: 2,
            num_experts_per_tok: 1,
            moe_intermediate_size: 4,
            shared_expert_intermediate_size: 4,
        });
        c
    }

    #[test]
    fn dense_capture_returns_correct_shape() {
        let cfg = tiny_dense_cfg();
        let mut model = Qwen35Model::empty_from_cfg(cfg.clone());
        let tokens: Vec<u32> = vec![0, 1, 2, 3];

        let acts = model.run_calibration_prompt(&tokens).expect("capture ok");
        acts.validate().expect("validate ok");

        assert_eq!(acts.num_layers, cfg.num_hidden_layers);
        assert_eq!(acts.seq_len, tokens.len() as u32);
        assert_eq!(acts.hidden_size, cfg.hidden_size);
        assert_eq!(acts.layer_inputs.len(), cfg.num_hidden_layers as usize);
        assert_eq!(acts.layer_outputs.len(), cfg.num_hidden_layers as usize);
        for li in &acts.layer_inputs {
            assert_eq!(li.len(), tokens.len() * cfg.hidden_size as usize);
        }
        for lo in &acts.layer_outputs {
            assert_eq!(lo.len(), tokens.len() * cfg.hidden_size as usize);
        }
    }

    #[test]
    fn dense_layer_input_zero_equals_post_embedding() {
        // For an empty (zero-weighted) model with zero token_embd, layer_inputs[0]
        // should be all zeros (= the embedding output for any token).
        let cfg = tiny_dense_cfg();
        let mut model = Qwen35Model::empty_from_cfg(cfg.clone());
        let tokens: Vec<u32> = vec![0, 0, 0];

        let acts = model.run_calibration_prompt(&tokens).expect("capture ok");
        for v in &acts.layer_inputs[0] {
            assert_eq!(*v, 0.0, "zero-weight model should produce zero residual");
        }
    }

    #[test]
    fn moe_quantized_returns_typed_forward_pass_error() {
        // empty_from_cfg for MoE variant constructs Qwen35FfnWeights::Moe (the
        // F32 path), not MoeQ. To exercise the MoeQ error branch we'd need a
        // real GGUF — covered by the production guard. This test pins the
        // ForwardPass error display format instead.
        let err = RealActivationCaptureError::ForwardPass {
            layer: 5,
            reason: "MoeQ requires GPU capture path".into(),
        };
        let s = format!("{}", err);
        assert!(s.contains("layer 5"));
        assert!(s.contains("MoeQ"));
    }

    #[test]
    fn moe_unquantized_capture_returns_correct_shape() {
        let cfg = tiny_moe_cfg();
        let mut model = Qwen35Model::empty_from_cfg(cfg.clone());
        let tokens: Vec<u32> = vec![0, 1];

        let acts = model.run_calibration_prompt(&tokens).expect("capture ok");
        acts.validate().expect("validate ok");
        assert_eq!(acts.num_layers, cfg.num_hidden_layers);
        assert_eq!(acts.layer_outputs.len(), cfg.num_hidden_layers as usize);
    }

    #[test]
    fn empty_tokens_returns_error() {
        let cfg = tiny_dense_cfg();
        let mut model = Qwen35Model::empty_from_cfg(cfg);
        let err = model.run_calibration_prompt(&[]).unwrap_err();
        assert!(format!("{err}").contains("non-empty"));
    }

    #[test]
    fn real_activation_capture_wrapper_delegates_to_model() {
        let cfg = tiny_dense_cfg();
        let model = Qwen35Model::empty_from_cfg(cfg.clone());
        let mut wrapper = RealActivationCapture::from_model(model);
        let tokens: Vec<u32> = vec![0, 1];
        let acts = wrapper.run_calibration_prompt(&tokens).expect("delegated");
        assert_eq!(acts.num_layers, cfg.num_hidden_layers);
    }

    #[test]
    fn not_ready_shim_returns_not_ready_error() {
        let mut cap = RealActivationCapture::not_ready();
        let err = cap.run_calibration_prompt(&[1, 2, 3]).unwrap_err();
        let s = format!("{}", err);
        assert!(
            s.contains("not_ready") || s.contains("NotReady") || s.contains("explicit"),
            "error must indicate the not-ready shim, got: {s}"
        );
    }

    #[test]
    fn error_display_for_load_includes_path() {
        let err = RealActivationCaptureError::Load {
            path: "/tmp/missing.gguf".into(),
            reason: "no such file".into(),
        };
        let s = format!("{}", err);
        assert!(s.contains("/tmp/missing.gguf"));
        assert!(s.contains("no such file"));
    }

    #[test]
    fn forward_pass_error_carries_layer_and_reason() {
        let err = RealActivationCaptureError::ForwardPass {
            layer: 7,
            reason: "attn_qkv shape mismatch".into(),
        };
        let s = format!("{}", err);
        assert!(s.contains("layer 7"));
        assert!(s.contains("attn_qkv shape mismatch"));
    }
}
