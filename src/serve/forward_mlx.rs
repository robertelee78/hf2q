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
//! Phase 5 step 1: weight loading + quantized matmul dispatch wired.
//! Remaining ops documented as TODOs at the end of this file.

use anyhow::Result;
use mlx_native::{
    GgmlQuantizedMatmulParams, GraphSession, MlxBuffer, MlxDevice,
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
}

/// All mlx-native weights for the full Gemma 4 model.
pub struct MlxModelWeights {
    pub embed_weight: MlxBuffer,
    pub layers: Vec<MlxDecoderLayerWeights>,
    pub final_norm: MlxBuffer,
    pub lm_head_f16: Option<MlxBuffer>,
    pub lm_head_f32: MlxBuffer,
    pub hidden_size: usize,
    pub rms_norm_eps: f32,
    pub final_logit_softcapping: Option<f32>,
}

impl MlxModelWeights {
    /// Load all model weights from the candle-based Gemma4Model into
    /// mlx-native MlxBuffers.
    ///
    /// This is called once at model load time.  Each weight is copied from
    /// candle's Metal buffer into a fresh mlx-native Metal allocation.
    /// The copy cost is one-time and does not affect inference throughput.
    pub fn load_from_candle(
        model: &super::gemma4::Gemma4Model,
        cfg: &super::config::Gemma4Config,
        mlx_device: &MlxDevice,
    ) -> Result<Self> {
        eprintln!("  Loading mlx-native weights from candle model...");

        // Embedding weight (F32)
        let embed_weight = gpu::candle_tensor_to_mlx_buffer(
            model.embed_weight(),
            mlx_device,
        )?;

        // Final norm weight (F32)
        let final_norm = gpu::candle_tensor_to_mlx_buffer(
            model.final_norm_weight(),
            mlx_device,
        )?;

        // lm_head weights
        let lm_head_f32 = gpu::candle_tensor_to_mlx_buffer(
            model.lm_head_weight(),
            mlx_device,
        )?;
        let lm_head_f16 = match model.lm_head_f16() {
            Some(t) => Some(gpu::candle_tensor_f16_to_mlx_buffer(t, mlx_device)?),
            None => None,
        };

        // Per-layer weights
        let num_layers = model.num_layers();
        let mut layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            eprint!("\r  Loading mlx-native layer weights {}/{}...", i + 1, num_layers);
            let refs = model.layer_weights(i);

            // Attention QMatMul weights
            let q_proj = load_qweight(refs.q_proj, mlx_device, "q_proj")?;
            let k_proj = load_qweight(refs.k_proj, mlx_device, "k_proj")?;
            let v_proj = match refs.v_proj {
                Some(qm) => Some(load_qweight(qm, mlx_device, "v_proj")?),
                None => None,
            };
            let o_proj = load_qweight(refs.o_proj, mlx_device, "o_proj")?;

            // Attention norm weights (F32)
            let q_norm_weight = gpu::candle_tensor_to_mlx_buffer(refs.q_norm, mlx_device)?;
            let k_norm_weight = gpu::candle_tensor_to_mlx_buffer(refs.k_norm, mlx_device)?;

            let attn = MlxAttentionWeights {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm_weight,
                k_norm_weight,
            };

            // Dense MLP QMatMul weights
            let gate_proj = load_qweight(refs.gate_proj, mlx_device, "gate_proj")?;
            let up_proj = load_qweight(refs.up_proj, mlx_device, "up_proj")?;
            let down_proj = load_qweight(refs.down_proj, mlx_device, "down_proj")?;

            let mlp = MlxMlpWeights {
                gate_proj,
                up_proj,
                down_proj,
            };

            // Norm weights (F32)
            let norms = MlxLayerNorms {
                input_layernorm: gpu::candle_tensor_to_mlx_buffer(
                    refs.input_layernorm, mlx_device,
                )?,
                post_attention_layernorm: gpu::candle_tensor_to_mlx_buffer(
                    refs.post_attention_layernorm, mlx_device,
                )?,
                pre_feedforward_layernorm: gpu::candle_tensor_to_mlx_buffer(
                    refs.pre_feedforward_layernorm, mlx_device,
                )?,
                post_feedforward_layernorm: gpu::candle_tensor_to_mlx_buffer(
                    refs.post_feedforward_layernorm, mlx_device,
                )?,
                pre_feedforward_layernorm_2: gpu::candle_tensor_to_mlx_buffer(
                    refs.pre_feedforward_layernorm_2, mlx_device,
                )?,
                post_feedforward_layernorm_1: gpu::candle_tensor_to_mlx_buffer(
                    refs.post_feedforward_layernorm_1, mlx_device,
                )?,
                post_feedforward_layernorm_2: gpu::candle_tensor_to_mlx_buffer(
                    refs.post_feedforward_layernorm_2, mlx_device,
                )?,
            };

            // Layer scalar (F32)
            let layer_scalar = gpu::candle_tensor_to_mlx_buffer(
                refs.layer_scalar, mlx_device,
            )?;

            layers.push(MlxDecoderLayerWeights {
                attn,
                mlp,
                norms,
                layer_scalar,
            });
        }
        eprintln!(
            "\r  Loaded {}/{} mlx-native layer weights.    ",
            num_layers, num_layers
        );

        Ok(Self {
            embed_weight,
            layers,
            final_norm,
            lm_head_f16,
            lm_head_f32,
            hidden_size: model.hidden_size_val(),
            rms_norm_eps: cfg.rms_norm_eps as f32,
            final_logit_softcapping: model.softcapping().map(|v| v as f32),
        })
    }
}

/// Helper: load a candle QMatMul into an MlxQWeight.
fn load_qweight(
    qmatmul: &candle_core::quantized::QMatMul,
    mlx_device: &MlxDevice,
    name: &str,
) -> Result<MlxQWeight> {
    let (buffer, info) = gpu::load_qmatmul_to_mlx_buffer(qmatmul, mlx_device)
        .map_err(|e| anyhow::anyhow!("Failed to load {} into MlxBuffer: {}", name, e))?;
    Ok(MlxQWeight { buffer, info })
}

// ---------------------------------------------------------------------------
// Forward pass dispatch helpers
// ---------------------------------------------------------------------------

/// Run one quantized matmul through the GraphSession.
///
/// Dispatches `output = input @ weight.T` where weight is in GGML block format.
/// The output buffer is pre-allocated by the caller.
///
/// Takes `registry` and `device` separately to avoid borrow conflicts on
/// `GpuContext` (registry is `&mut`, device is `&`).
#[allow(dead_code)]
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
// - [x] Weight bridge: candle QTensor/Tensor -> MlxBuffer (gpu.rs)
// - [x] GpuContext creation and lifecycle (gpu.rs)
// - [x] --backend CLI flag (cli.rs)
// - [x] Backend selection in cmd_generate (mod.rs)
// - [x] Weight loading from candle model (MlxModelWeights::load_from_candle)
// - [x] Gemma4Model pub(crate) accessors for weight extraction
// - [x] quantized_matmul_ggml dispatch helper
//
// TODO (ops available in mlx-native's GraphSession, need forward orchestration):
// - [ ] Full forward pass orchestrator (forward_mlx_pass function)
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
// - [ ] MoE weight loading (3D expert gate_up/down buffers)
//
// BLOCKERS documented:
// 1. candle-metal-kernels vendors its own `metal::Buffer` type, preventing
//    zero-copy buffer sharing.  Resolved with load-time copy.
// 2. KV cache state management: the mlx-native path needs its own KV cache
//    buffers (MlxBuffer) separate from candle's.  Not yet implemented.
// 3. Activation buffer pool: the forward pass needs transient MlxBuffer
//    allocations from MlxBufferPool for intermediate results.
