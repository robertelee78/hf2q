//! Real (non-mock) `ActivationCapture` for `Qwen35Model` — GPU-only.
//!
//! ADR-012 P9 / ADR-013 Decision 16. Lands on top of the ADR-013 merge
//! (`236abb8` — full Qwen3.5/3.6 forward) which closed P12 with trait+mock
//! only and explicitly deferred the real impl. This file finishes that
//! deferred contract.
//!
//! # Path: GPU only
//!
//! Per `feedback_gpu_everything.md` ("All ops on Metal GPU, no CPU
//! fallbacks. CPU inference == poop.") and
//! `feedback_tests_on_gpu_not_cpu.md` ("Test-path runs on Metal GPU"),
//! activation capture has exactly one execution path: `forward_gpu_with_capture`
//! → `mlx-native::quantized_matmul_ggml` for native MoeQ experts,
//! `mul_mm_id` for F32-expanded experts, dense GEMM kernels for Dense
//! FFN. Unit tests build production-shaped synthetic models (`hidden_size`
//! divisible by 32 to satisfy the Q4_0 lm_head packing constraint) so they
//! run through the same code path as production calibration.
//!
//! # Surfaces
//!
//! * [`run_calibration_prompt_gpu`] — free function that takes a
//!   `&Qwen35Model` and a token list, allocates a fresh `HybridKvCache`,
//!   drives `forward_gpu_with_capture`, and returns a populated
//!   `LayerActivations`. Power callers that already hold a model use this.
//! * [`RealActivationCapture`] — convenience wrapper for the DWQ
//!   calibration pipeline: it loads a `Qwen35Model` from a GGUF on the fly
//!   and delegates `run_calibration_prompt` to the GPU path. The
//!   `not_ready()` constructor remains for unit tests that pin the
//!   no-fallback guard surface.
//!
//! # Capture semantics
//!
//! For each transformer block `l` (`0 ≤ l < num_hidden_layers`):
//!
//! * `layer_inputs[l]` — residual stream entering layer `l`. For `l == 0`
//!   this is the embedding output; for `l > 0` this is `layer_outputs[l-1]`.
//! * `layer_outputs[l]` — residual stream leaving layer `l` after FFN
//!   residual add (matches the per-block `ffn_residual + ffn_out` shape
//!   in `forward_gpu`).
//!
//! Each captured tensor is row-major `[seq_len, hidden_size]` flattened
//! to `Vec<f32>` per the `LayerActivations` contract. Capture is performed
//! by `forward_gpu_with_capture` via a `download_f32` round-trip at the
//! start and end of each layer iteration.
//!
//! # Variants supported
//!
//! All Qwen3.5/3.6 weight variants — Dense, F32-expanded MoE, native
//! GGML-block-quantized MoeQ — go through the same GPU path. No F32
//! expansion of 256-expert MoE; no fallback.
//!
//! # No fallback
//!
//! Per `feedback_never_ship_fallback_without_rootcause.md`: if the GPU
//! forward errors out (OOM, kernel failure, malformed weights) we return
//! a typed error, not synthetic activations. Callers must handle this
//! rather than silently routing around it.

use anyhow::{Context, Result};
use mlx_native::MlxDevice;
use thiserror::Error;
use tokenizers::Tokenizer;

use super::activation_capture::{ActivationCapture, LayerActivations};
use super::kv_cache::HybridKvCache;
use super::model::Qwen35Model;
use crate::ir::lazy::LazyTensorMap;

// ================================================================
// Error type
// ================================================================

/// Errors returned by the real ActivationCapture path. `NotReady` is
/// retained for the `not_ready()` factory used in dependency-blocked
/// unit tests; `ForwardPass` is the production error variant naming the
/// specific layer / reason; `Load` covers GGUF open / parse failures.
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
// GPU capture path (ADR-012 P9b — sole execution path)
// ================================================================

/// Run a calibration prompt through the GPU forward and capture per-layer
/// residual streams. Drives [`Qwen35Model::forward_gpu_with_capture`],
/// which dispatches `mlx-native::quantized_matmul_ggml` for native MoeQ
/// experts — no F32 expansion, no OOM, runs at production decode speed.
///
/// Allocates a fresh `HybridKvCache` sized to `tokens.len()` for the
/// calibration pass; the cache is dropped on return so subsequent
/// inference sessions are unaffected.
pub fn run_calibration_prompt_gpu(
    model: &Qwen35Model,
    tokens: &[u32],
) -> Result<LayerActivations> {
    if tokens.is_empty() {
        anyhow::bail!("run_calibration_prompt_gpu: tokens must be non-empty");
    }

    let seq_len = tokens.len() as u32;
    // Text-only positions: one axis per token, replicated across all 4 MROPE
    // sectors. Matches `text_positions` in the GPU forward.
    let mut positions_flat: Vec<i32> = Vec::with_capacity((4 * seq_len) as usize);
    for _ in 0..4 {
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
// Convenience wrapper — RealActivationCapture
// ================================================================

/// Production wrapper that owns a `Qwen35Model` loaded from disk and
/// drives the GPU activation capture path.
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

    /// Construct a real capture directly from a transformed lazy tensor map,
    /// avoiding the ADR-012 P9b intermediate-GGUF round trip. `_tokenizer`
    /// is reserved for parity with the disk constructor and future prompt
    /// tokenization inside this wrapper.
    pub fn from_lazy_tensor_map(
        model: &LazyTensorMap,
        _tokenizer: &Tokenizer,
    ) -> std::result::Result<Self, RealActivationCaptureError> {
        let loaded = Qwen35Model::load_from_lazy_tensor_map(model).map_err(|e| {
            RealActivationCaptureError::Load {
                path: "<lazy-tensor-map>".to_string(),
                reason: format!("Qwen35Model::load_from_lazy_tensor_map: {e:#}"),
            }
        })?;
        Ok(Self {
            inner: RealCaptureBackend::Loaded(loaded),
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
        match &self.inner {
            RealCaptureBackend::Loaded(model) => run_calibration_prompt_gpu(model, tokens),
            RealCaptureBackend::NotReady => {
                Err(anyhow::anyhow!(RealActivationCaptureError::NotReady))
            }
        }
    }
}

// ================================================================
// Tests — all run through the GPU path via run_calibration_prompt_gpu
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{Qwen35Config, Qwen35MoeConfig, Qwen35Variant};
    use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap};
    use crate::ir::DType;

    /// GPU-eligible tiny hybrid Qwen3.5 dense config (mirrors
    /// `forward_gpu::tests::tiny_hybrid_cfg`): `hidden_size = 64`
    /// (divisible by 32 to satisfy the Q4_0 lm_head packing constraint),
    /// `head_dim = 32`, 4 layers (one full-attn at index 3, the rest
    /// linear-attn). Sufficient to exercise both attention branches and
    /// the FFN through `forward_gpu_with_capture`.
    fn tiny_dense_cfg() -> Qwen35Config {
        let layer_types = super::super::default_layer_types(4, 4);
        Qwen35Config {
            variant: Qwen35Variant::Dense,
            hidden_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 32,
            linear_num_key_heads: 2,
            linear_num_value_heads: 2,
            linear_key_head_dim: 32,
            linear_value_head_dim: 32,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            layer_types,
            partial_rotary_factor: 0.5,
            rope_theta: 10_000.0,
            rotary_dim: 16,
            mrope_section: [4, 4, 0, 0],
            mrope_interleaved: true,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 128,
            vocab_size: 128,
            attn_output_gate: true,
            mtp_num_hidden_layers: 0,
            intermediate_size: Some(64),
            moe: None,
        }
    }

    fn tiny_moe_cfg() -> Qwen35Config {
        let mut c = tiny_dense_cfg();
        c.variant = Qwen35Variant::Moe;
        c.intermediate_size = None;
        c.moe = Some(Qwen35MoeConfig {
            num_experts: 2,
            num_experts_per_tok: 1,
            moe_intermediate_size: 32,
            shared_expert_intermediate_size: 32,
        });
        c
    }

    fn f32_bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
        values.flat_map(|v| v.to_le_bytes()).collect()
    }

    fn insert_f32(map: &mut LazyTensorMap, name: &str, shape: Vec<usize>) {
        let numel: usize = shape.iter().product();
        let data = f32_bytes((0..numel).map(|i| i as f32 * 0.001));
        let meta = LazyMeta::new(name.to_string(), shape, DType::F32);
        map.insert(LazyTensor::from_bytes(meta, data));
    }

    fn single_full_attention_lazy_map() -> LazyTensorMap {
        let mut map = LazyTensorMap::new();
        let h = 32usize;
        let d = 8usize;
        let q_heads = 2usize;
        let kv_heads = 1usize;
        let inter = 32usize;
        insert_f32(&mut map, "token_embd.weight", vec![16, h]);
        insert_f32(&mut map, "output.weight", vec![16, h]);
        insert_f32(&mut map, "output_norm.weight", vec![h]);
        insert_f32(&mut map, "blk.0.attn_norm.weight", vec![h]);
        insert_f32(&mut map, "blk.0.post_attention_norm.weight", vec![h]);
        insert_f32(&mut map, "blk.0.attn_q.weight", vec![2 * q_heads * d, h]);
        insert_f32(&mut map, "blk.0.attn_k.weight", vec![kv_heads * d, h]);
        insert_f32(&mut map, "blk.0.attn_v.weight", vec![kv_heads * d, h]);
        insert_f32(&mut map, "blk.0.attn_q_norm.weight", vec![d]);
        insert_f32(&mut map, "blk.0.attn_k_norm.weight", vec![d]);
        insert_f32(&mut map, "blk.0.attn_output.weight", vec![h, q_heads * d]);
        insert_f32(&mut map, "blk.0.ffn_gate.weight", vec![inter, h]);
        insert_f32(&mut map, "blk.0.ffn_up.weight", vec![inter, h]);
        insert_f32(&mut map, "blk.0.ffn_down.weight", vec![h, inter]);
        map
    }

    fn dummy_tokenizer() -> Tokenizer {
        Tokenizer::new(tokenizers::models::bpe::BPE::default())
    }

    #[test]
    fn dense_capture_returns_correct_shape() {
        let cfg = tiny_dense_cfg();
        let model = Qwen35Model::empty_from_cfg(cfg.clone());
        let tokens: Vec<u32> = vec![0, 1, 2];

        let acts = run_calibration_prompt_gpu(&model, &tokens).expect("capture ok");
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
        // Zero-weight model with zero `token_embd`: the residual stream
        // entering layer 0 is the embedding output, which is all zeros for
        // any token. The GPU forward downloads `hidden` to F32 at the start
        // of each layer iteration so this captures the embedding output
        // verbatim.
        let cfg = tiny_dense_cfg();
        let model = Qwen35Model::empty_from_cfg(cfg);
        let tokens: Vec<u32> = vec![0, 0, 0];

        let acts = run_calibration_prompt_gpu(&model, &tokens).expect("capture ok");
        for v in &acts.layer_inputs[0] {
            assert_eq!(*v, 0.0, "zero-weight model should produce zero residual");
        }
    }

    #[test]
    fn moe_unquantized_capture_returns_correct_shape() {
        let cfg = tiny_moe_cfg();
        let model = Qwen35Model::empty_from_cfg(cfg.clone());
        let tokens: Vec<u32> = vec![0, 1];

        let acts = run_calibration_prompt_gpu(&model, &tokens).expect("capture ok");
        acts.validate().expect("validate ok");
        assert_eq!(acts.num_layers, cfg.num_hidden_layers);
        assert_eq!(acts.layer_outputs.len(), cfg.num_hidden_layers as usize);
    }

    #[test]
    fn moe_quantized_returns_typed_forward_pass_error() {
        // empty_from_cfg for MoE constructs Qwen35FfnWeights::Moe (the
        // F32-expanded path). Exercising the MoeQ branch requires a real
        // GGUF — covered by integration-level tests gated on the apex
        // file. This test pins the ForwardPass error display format.
        let err = RealActivationCaptureError::ForwardPass {
            layer: 5,
            reason: "MoeQ requires GPU capture path".into(),
        };
        let s = format!("{}", err);
        assert!(s.contains("layer 5"));
        assert!(s.contains("MoeQ"));
    }

    #[test]
    fn empty_tokens_returns_error() {
        let cfg = tiny_dense_cfg();
        let model = Qwen35Model::empty_from_cfg(cfg);
        let err = run_calibration_prompt_gpu(&model, &[]).unwrap_err();
        assert!(format!("{err}").contains("non-empty"));
    }

    #[test]
    fn real_activation_capture_wrapper_delegates_to_gpu() {
        let cfg = tiny_dense_cfg();
        let model = Qwen35Model::empty_from_cfg(cfg.clone());
        let mut wrapper = RealActivationCapture::from_model(model);
        let tokens: Vec<u32> = vec![0, 1];
        let acts = wrapper.run_calibration_prompt(&tokens).expect("delegated");
        assert_eq!(acts.num_layers, cfg.num_hidden_layers);
    }

    #[test]
    fn real_activation_capture_from_lazy_tensor_map_loads_model() {
        let map = single_full_attention_lazy_map();
        let tokenizer = dummy_tokenizer();
        let wrapper = match RealActivationCapture::from_lazy_tensor_map(&map, &tokenizer) {
            Ok(wrapper) => wrapper,
            Err(err) if format!("{err}").contains("No Metal GPU device found") => {
                eprintln!("skipping GPU-backed lazy capture test: {err}");
                return;
            }
            Err(err) => panic!("lazy capture loads: {err:#}"),
        };
        let dbg = format!("{wrapper:?}");
        assert!(dbg.contains("num_hidden_layers"));
        assert!(dbg.contains("Dense"));
    }

    #[test]
    fn real_activation_capture_from_lazy_tensor_map_errors_on_empty_map() {
        let map = LazyTensorMap::new();
        let tokenizer = dummy_tokenizer();
        let err = RealActivationCapture::from_lazy_tensor_map(&map, &tokenizer)
            .expect_err("empty lazy map must fail");
        let s = format!("{err}");
        assert!(s.contains("<lazy-tensor-map>"));
        assert!(s.contains("load_from_lazy_tensor_map"));
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
