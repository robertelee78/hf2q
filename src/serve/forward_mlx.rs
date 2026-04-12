//! mlx-native forward pass for Gemma 4 inference.
//!
//! ADR-006 Phase 5: routes the entire `Gemma4Model::forward` through
//! mlx-native's `GraphExecutor`, encoding all ops into a single Metal
//! command buffer with one `commit_and_wait` per token.
//!
//! # Architecture
//!
//! At model load time, all weights are copied from candle's Metal buffers
//! into mlx-native `MlxBuffer` instances via `gpu.rs` bridge functions.
//! At inference time, the forward pass uses mlx-native's `GraphSession`
//! to encode all ops (qmatmul, RmsNorm, RoPE, SDPA, etc.) into a single
//! command buffer.
//!
//! # Status
//!
//! Phase 5 step 1: quantized matmul (QLinear projections) wired.
//! Steps 2-8 (SDPA, RoPE, RmsNorm, MoE, etc.) are documented as TODOs.

use anyhow::Result;
use mlx_native::{
    GgmlQuantizedMatmulParams, GgmlType, GraphSession, MlxBuffer, MlxDevice,
};

use super::gpu::{self, GpuContext, QuantWeightInfo};

// ---------------------------------------------------------------------------
// Weight storage for the mlx-native forward path
// ---------------------------------------------------------------------------

/// Pre-loaded quantized weight buffer paired with its GGML metadata.
pub struct MlxQWeight {
    pub buffer: MlxBuffer,
    pub info: QuantWeightInfo,
}

impl MlxQWeight {
    /// Build `GgmlQuantizedMatmulParams` for a mat-vec dispatch.
    ///
    /// `m` is the number of input tokens (1 for decode).
    pub fn matmul_params(&self, m: u32) -> Result<GgmlQuantizedMatmulParams> {
        let ggml_type = gpu::candle_ggml_to_mlx(self.info.ggml_dtype)?;
        Ok(GgmlQuantizedMatmulParams {
            m,
            n: self.info.rows as u32,
            k: self.info.cols as u32,
            ggml_type,
        })
    }
}

/// Per-layer attention weights for the mlx-native forward path.
pub struct MlxAttentionWeights {
    pub q_proj: MlxQWeight,
    pub k_proj: MlxQWeight,
    pub v_proj: Option<MlxQWeight>, // None when k_eq_v
    pub o_proj: MlxQWeight,
    pub q_norm_weight: MlxBuffer,
    pub k_norm_weight: MlxBuffer,
}

/// Per-layer dense MLP weights for the mlx-native forward path.
pub struct MlxMlpWeights {
    pub gate_proj: MlxQWeight,
    pub up_proj: MlxQWeight,
    pub down_proj: MlxQWeight,
}

/// Per-layer norm weights (7 RmsNorm per layer).
pub struct MlxLayerNorms {
    pub input_layernorm: MlxBuffer,
    pub post_attention_layernorm: MlxBuffer,
    pub pre_feedforward_layernorm: MlxBuffer,
    pub post_feedforward_layernorm: MlxBuffer,
    pub pre_feedforward_layernorm_2: MlxBuffer,
    pub post_feedforward_layernorm_1: MlxBuffer,
    pub post_feedforward_layernorm_2: MlxBuffer,
}

/// All mlx-native weights for one decoder layer.
pub struct MlxDecoderLayerWeights {
    pub attn: MlxAttentionWeights,
    pub mlp: MlxMlpWeights,
    pub norms: MlxLayerNorms,
    pub layer_scalar: MlxBuffer,
    // MoE weights: TODO Phase 5 step 8
    // pub moe_router: MlxQWeight,
    // pub moe_expert_gate_up_3d: MlxBuffer,
    // pub moe_expert_down_3d: MlxBuffer,
}

/// All mlx-native weights for the full Gemma 4 model.
pub struct MlxModelWeights {
    pub embed_weight: MlxBuffer,
    pub layers: Vec<MlxDecoderLayerWeights>,
    pub final_norm: MlxBuffer,
    pub lm_head_f16: MlxBuffer,
    pub hidden_size: usize,
    pub rms_norm_eps: f32,
    pub final_logit_softcapping: Option<f32>,
}

impl MlxModelWeights {
    /// Load all model weights from candle's QMatMul/Tensor objects into
    /// mlx-native MlxBuffers.
    ///
    /// This is called once at model load time.  Each weight is copied from
    /// candle's Metal buffer into a fresh mlx-native Metal allocation.
    ///
    /// # Arguments
    ///
    /// * `model` - The loaded candle-based Gemma4Model (for weight access)
    /// * `mlx_device` - The mlx-native device for buffer allocation
    /// * `cfg` - Model configuration
    ///
    /// # Current limitations
    ///
    /// MoE expert weights are not yet loaded (Phase 5 step 8).
    pub fn load_from_candle(
        model: &super::gemma4::Gemma4Model,
        _cfg: &super::config::Gemma4Config,
        _mlx_device: &MlxDevice,
    ) -> Result<Self> {
        // Phase 5 step 1: weight loading infrastructure is ready.
        // The actual weight extraction requires exposing model internals
        // from Gemma4Model (currently private fields).
        //
        // This will be wired in a follow-up commit that adds pub(crate)
        // accessors to Gemma4Model's weight fields.
        let _ = model;
        anyhow::bail!(
            "MlxModelWeights::load_from_candle is not yet implemented. \
             Phase 5 weight loading requires pub(crate) accessors on \
             Gemma4Model fields (layers, embed_tokens, norm, etc.)."
        )
    }
}

// ---------------------------------------------------------------------------
// Forward pass dispatch
// ---------------------------------------------------------------------------

/// Run one quantized matmul through the GraphSession.
///
/// Dispatches `output = input @ weight.T` where weight is in GGML block format.
/// The output buffer is pre-allocated by the caller.
///
/// Takes `registry` and `device` separately to avoid borrow conflicts on
/// `GpuContext` (registry is `&mut`, device is `&`).
pub fn dispatch_qmatmul(
    session: &mut GraphSession<'_>,
    registry: &mut mlx_native::KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxQWeight,
    output: &mut MlxBuffer,
    m: u32,
) -> Result<()> {
    let params = weight.matmul_params(m)?;
    session.quantized_matmul_ggml(
        registry,
        device,
        input,
        &weight.buffer,
        output,
        &params,
    ).map_err(|e| anyhow::anyhow!("quantized_matmul_ggml failed: {e}"))
}

// ---------------------------------------------------------------------------
// Phase 5 forward pass TODOs
// ---------------------------------------------------------------------------
//
// The full forward pass (Section 3.2 of arch-target-inference-path.md) requires
// these ops to be wired through GraphSession, in order:
//
// DONE:
// - [x] quantized_matmul_ggml (Q/K/V/O projections, gate/up/down MLP, lm_head)
// - [x] Weight bridge: candle QTensor/Tensor -> MlxBuffer (gpu.rs)
// - [x] GpuContext creation and lifecycle (gpu.rs)
// - [x] --backend CLI flag (cli.rs)
// - [x] Backend selection in cmd_generate (mod.rs)
//
// TODO (wired in mlx-native's GraphSession API, need integration here):
// - [ ] rms_norm (7 per layer + 1 final + V unit norm)
// - [ ] rope (Q and K per layer)
// - [ ] sdpa (vector kernel for decode)
// - [ ] elementwise_add (residual connections, 3 per layer)
// - [ ] elementwise_mul (SwiGLU gate*up, layer_scalar)
// - [ ] gelu (dense MLP activation)
// - [ ] softmax (MoE router)
// - [ ] softcap (final logit capping)
// - [ ] embedding_gather (input embedding lookup)
// - [ ] kv_cache_copy (in-place KV cache append)
// - [ ] argsort + gather (MoE top-k routing)
// - [ ] quantized_matmul_id (fused MoE expert dispatch)
// - [ ] f16_gemm / dense_gemm (lm_head projection)
// - [ ] scalar_mul (embedding scale by sqrt(hidden_size))
// - [ ] transpose (Q/K/V reshape for attention)
// - [ ] narrow (last-token extraction before lm_head)
//
// The forward pass orchestrator will be added once Gemma4Model exposes
// pub(crate) accessors to its weight fields, or alternatively, the
// MlxModelWeights are loaded directly from the GGUF during model construction.
