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
//! to encode all ops (qmatmul, RmsNorm, RoPE, SDPA, etc.) into batched
//! command buffers to minimize GPU sync points.
//!
//! # Status
//!
//! Phase 5 step 4: MoE single-session optimization.
//! - Dense MLP path merged into 1 session (SDPA→down in 1 session).
//! - GPU norms, adds, GELU, mul in merged sessions.
//! - GPU dense GEMM for lm_head (F16) replaces CPU matmul.
//! - MoE experts: ALL experts in ONE session using GPU SwiGLU + accumulate.
//!   Eliminates 16 sessions/layer → 1, for 4 sessions/layer total.

use anyhow::Result;
use mlx_native::{
    GgmlQuantizedMatmulParams, GraphSession, MlxBuffer, MlxDevice,
};
use mlx_native::ops::sdpa::SdpaParams;
use mlx_native::ops::sdpa_sliding::SdpaSlidingParams;
use mlx_native::ops::dense_gemm::DenseGemmF16Params;
use mlx_native::ops::elementwise::CastDirection;
use std::time::Instant;

use super::config::{Gemma4Config, LayerType};
use super::gpu::{self, GpuContext, QuantWeightInfo};

// ---------------------------------------------------------------------------
// Profiling support (HF2Q_MLX_PROFILE=1)
// ---------------------------------------------------------------------------

/// Check if profiling is enabled via environment variable.
fn profiling_enabled() -> bool {
    std::env::var("HF2Q_MLX_PROFILE").map_or(false, |v| v == "1")
}

/// Accumulated timing data for one token's forward pass.
#[derive(Default, Clone)]
pub struct TokenProfile {
    /// Per-layer session timings (wall-clock, includes GPU wait).
    pub layer_s1_us: Vec<f64>,    // QKV projections
    pub layer_cpu1_us: Vec<f64>,  // head norms, RoPE, KV cache
    pub layer_s2_us: Vec<f64>,    // SDPA + MLP
    pub layer_cpu2_us: Vec<f64>,  // post-FF norm, MoE routing prep
    pub layer_s3_us: Vec<f64>,    // router proj
    pub layer_cpu3_us: Vec<f64>,  // softmax + top-k
    pub layer_s4_us: Vec<f64>,    // MoE experts
    pub layer_cpu4_us: Vec<f64>,  // post-MoE norms, combine, scalar
    pub head_session_us: f64,     // lm_head session
    pub head_cpu_us: f64,         // softcap + argmax CPU
    pub total_us: f64,
    /// Dispatch counts per session type.
    pub s1_dispatches: Vec<usize>,
    pub s2_dispatches: Vec<usize>,
    pub s3_dispatches: Vec<usize>,
    pub s4_dispatches: Vec<usize>,
    pub head_dispatches: usize,
}

/// Multi-token profiling accumulator.
pub struct ProfileAccumulator {
    pub tokens: Vec<TokenProfile>,
    pub warmup_count: usize,
    pub enabled: bool,
}

impl ProfileAccumulator {
    pub fn new(warmup: usize) -> Self {
        Self {
            tokens: Vec::new(),
            warmup_count: warmup,
            enabled: profiling_enabled(),
        }
    }

    pub fn start_token(&self) -> Option<TokenProfile> {
        if self.enabled {
            Some(TokenProfile::default())
        } else {
            None
        }
    }

    pub fn finish_token(&mut self, profile: Option<TokenProfile>) {
        if let Some(p) = profile {
            self.tokens.push(p);
        }
    }

    /// Print summary after generation is complete.
    pub fn print_summary(&self) {
        if !self.enabled || self.tokens.is_empty() {
            return;
        }
        let skip = self.warmup_count.min(self.tokens.len().saturating_sub(1));
        let measured: Vec<&TokenProfile> = self.tokens.iter().skip(skip).collect();
        if measured.is_empty() {
            eprintln!("[PROFILE] No tokens after warmup to report.");
            return;
        }
        let n = measured.len();
        let num_layers = measured[0].layer_s1_us.len();

        eprintln!("\n╔══════════════════════════════════════════════════════════╗");
        eprintln!("║  MLX-NATIVE FORWARD PASS PROFILE ({n} tokens, {skip} warmup skipped)  ║");
        eprintln!("╠══════════════════════════════════════════════════════════╣");

        // Per-session-type averages across all layers and tokens
        let avg = |getter: &dyn Fn(&TokenProfile) -> &Vec<f64>| -> f64 {
            let total: f64 = measured.iter()
                .map(|t| getter(t).iter().sum::<f64>())
                .sum();
            total / n as f64
        };

        let s1_avg = avg(&|t| &t.layer_s1_us);
        let cpu1_avg = avg(&|t| &t.layer_cpu1_us);
        let s2_avg = avg(&|t| &t.layer_s2_us);
        let cpu2_avg = avg(&|t| &t.layer_cpu2_us);
        let s3_avg = avg(&|t| &t.layer_s3_us);
        let cpu3_avg = avg(&|t| &t.layer_cpu3_us);
        let s4_avg = avg(&|t| &t.layer_s4_us);
        let cpu4_avg = avg(&|t| &t.layer_cpu4_us);
        let head_gpu_avg: f64 = measured.iter().map(|t| t.head_session_us).sum::<f64>() / n as f64;
        let head_cpu_avg: f64 = measured.iter().map(|t| t.head_cpu_us).sum::<f64>() / n as f64;
        let total_avg: f64 = measured.iter().map(|t| t.total_us).sum::<f64>() / n as f64;

        let gpu_total = s1_avg + s2_avg + s3_avg + s4_avg + head_gpu_avg;
        let cpu_total = cpu1_avg + cpu2_avg + cpu3_avg + cpu4_avg + head_cpu_avg;

        eprintln!("║ {num_layers} layers x 4 sessions + 1 head = {} sessions/token", num_layers * 4 + 1);
        eprintln!("║");
        eprintln!("║ Session breakdown (avg across {num_layers} layers, {n} tokens):");
        eprintln!("║   S1 (QKV proj):      {:8.1} us ({:5.2} ms total)", s1_avg / num_layers as f64, s1_avg / 1000.0);
        eprintln!("║   CPU1 (norms+RoPE):   {:8.1} us ({:5.2} ms total)", cpu1_avg / num_layers as f64, cpu1_avg / 1000.0);
        eprintln!("║   S2 (SDPA+MLP):      {:8.1} us ({:5.2} ms total)", s2_avg / num_layers as f64, s2_avg / 1000.0);
        eprintln!("║   CPU2 (post-FF):      {:8.1} us ({:5.2} ms total)", cpu2_avg / num_layers as f64, cpu2_avg / 1000.0);
        eprintln!("║   S3 (router proj):   {:8.1} us ({:5.2} ms total)", s3_avg / num_layers as f64, s3_avg / 1000.0);
        eprintln!("║   CPU3 (softmax+topk): {:8.1} us ({:5.2} ms total)", cpu3_avg / num_layers as f64, cpu3_avg / 1000.0);
        eprintln!("║   S4 (MoE experts):   {:8.1} us ({:5.2} ms total)", s4_avg / num_layers as f64, s4_avg / 1000.0);
        eprintln!("║   CPU4 (post-MoE):     {:8.1} us ({:5.2} ms total)", cpu4_avg / num_layers as f64, cpu4_avg / 1000.0);
        eprintln!("║   Head GPU:            {:8.1} us ({:5.2} ms)", head_gpu_avg, head_gpu_avg / 1000.0);
        eprintln!("║   Head CPU:            {:8.1} us ({:5.2} ms)", head_cpu_avg, head_cpu_avg / 1000.0);
        eprintln!("║");
        eprintln!("║ Total: {:8.1} us ({:5.2} ms)", total_avg, total_avg / 1000.0);
        eprintln!("║   GPU sessions: {:8.1} us ({:5.1}%)", gpu_total, gpu_total / total_avg * 100.0);
        eprintln!("║   CPU ops:      {:8.1} us ({:5.1}%)", cpu_total, cpu_total / total_avg * 100.0);
        let overhead = total_avg - gpu_total - cpu_total;
        if overhead.abs() > 10.0 {
            eprintln!("║   Unaccounted:  {:8.1} us ({:5.1}%)", overhead, overhead / total_avg * 100.0);
        }

        // Dispatch counts
        let avg_dispatches = |getter: &dyn Fn(&TokenProfile) -> &Vec<usize>| -> f64 {
            let total: usize = measured.iter()
                .map(|t| getter(t).iter().sum::<usize>())
                .sum();
            total as f64 / n as f64
        };
        let s1_disp = avg_dispatches(&|t| &t.s1_dispatches);
        let s2_disp = avg_dispatches(&|t| &t.s2_dispatches);
        let s3_disp = avg_dispatches(&|t| &t.s3_dispatches);
        let s4_disp = avg_dispatches(&|t| &t.s4_dispatches);
        let head_disp: f64 = measured.iter().map(|t| t.head_dispatches as f64).sum::<f64>() / n as f64;
        let total_disp = s1_disp + s2_disp + s3_disp + s4_disp + head_disp;

        eprintln!("║");
        eprintln!("║ Dispatch counts per token:");
        eprintln!("║   S1: {s1_disp:.0}  S2: {s2_disp:.0}  S3: {s3_disp:.0}  S4: {s4_disp:.0}  Head: {head_disp:.0}");
        eprintln!("║   Total: {total_disp:.0} dispatches/token");
        eprintln!("║   (candle Phase 0 baseline: ~105 dispatches/token)");
        eprintln!("║   Ratio: {:.1}x more dispatches", total_disp / 105.0);

        // Per-layer detail for first 3 layers + last layer
        eprintln!("║");
        eprintln!("║ Per-layer detail (avg over {n} tokens, us):");
        eprintln!("║   Layer |   S1   |  CPU1  |   S2   |  CPU2  |   S3   |  CPU3  |   S4   |  CPU4  | Total");
        eprintln!("║   ------|--------|--------|--------|--------|--------|--------|--------|--------|------");
        let detail_layers: Vec<usize> = {
            let mut v: Vec<usize> = (0..3.min(num_layers)).collect();
            if num_layers > 3 { v.push(num_layers - 1); }
            v
        };
        for &li in &detail_layers {
            let s1: f64 = measured.iter().map(|t| t.layer_s1_us[li]).sum::<f64>() / n as f64;
            let c1: f64 = measured.iter().map(|t| t.layer_cpu1_us[li]).sum::<f64>() / n as f64;
            let s2: f64 = measured.iter().map(|t| t.layer_s2_us[li]).sum::<f64>() / n as f64;
            let c2: f64 = measured.iter().map(|t| t.layer_cpu2_us[li]).sum::<f64>() / n as f64;
            let s3: f64 = measured.iter().map(|t| t.layer_s3_us[li]).sum::<f64>() / n as f64;
            let c3: f64 = measured.iter().map(|t| t.layer_cpu3_us[li]).sum::<f64>() / n as f64;
            let s4: f64 = measured.iter().map(|t| t.layer_s4_us[li]).sum::<f64>() / n as f64;
            let c4: f64 = measured.iter().map(|t| t.layer_cpu4_us[li]).sum::<f64>() / n as f64;
            let layer_type = if (li + 1) % 6 == 0 { "G" } else { "S" };
            eprintln!("║   {:>2} ({}) | {:6.0} | {:6.0} | {:6.0} | {:6.0} | {:6.0} | {:6.0} | {:6.0} | {:6.0} | {:6.0}",
                li, layer_type, s1, c1, s2, c2, s3, c3, s4, c4,
                s1 + c1 + s2 + c2 + s3 + c3 + s4 + c4);
        }

        // Session time per dispatch (to compare with candle per-kernel times)
        eprintln!("║");
        eprintln!("║ Avg time per dispatch (session_time / dispatches):");
        if s1_disp > 0.0 {
            eprintln!("║   S1 (QKV matmuls): {:.1} us/dispatch", s1_avg / s1_disp);
        }
        if s2_disp > 0.0 {
            eprintln!("║   S2 (SDPA+MLP):    {:.1} us/dispatch", s2_avg / s2_disp);
        }
        if s3_disp > 0.0 {
            eprintln!("║   S3 (router):      {:.1} us/dispatch", s3_avg / s3_disp);
        }
        if s4_disp > 0.0 {
            eprintln!("║   S4 (MoE):         {:.1} us/dispatch", s4_avg / s4_disp);
        }

        eprintln!("╚══════════════════════════════════════════════════════════╝");
    }
}

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

/// Per-expert MoE weights for one layer (quantized, GGML block format).
pub struct MlxMoeWeights {
    /// Per-expert gate_up projection: Vec of MlxQWeight, one per expert.
    /// Each expert's gate_up is `[2*moe_intermediate_size, hidden_size]`.
    pub expert_gate_up: Vec<MlxQWeight>,
    /// Per-expert down projection: Vec of MlxQWeight, one per expert.
    /// Each expert's down is `[hidden_size, moe_intermediate_size]`.
    pub expert_down: Vec<MlxQWeight>,
    /// Stacked gate_up weights: all experts concatenated into `[n_experts, N, packed_K]`.
    /// Used for the fused `quantized_matmul_id_ggml` dispatch.
    pub stacked_gate_up: Option<MlxBuffer>,
    /// Stacked down weights: all experts concatenated into `[n_experts, N, packed_K]`.
    pub stacked_down: Option<MlxBuffer>,
    /// Byte stride between expert slices in the stacked gate_up buffer.
    pub gate_up_expert_stride: u64,
    /// Byte stride between expert slices in the stacked down buffer.
    pub down_expert_stride: u64,
    /// Router projection weight (quantized).
    pub router_proj: MlxQWeight,
    /// Router learned scale `[hidden_size]` F32.
    pub router_scale: MlxBuffer,
    /// Per-expert scale `[num_experts]` F32.
    pub per_expert_scale: MlxBuffer,
    /// GGML quant type for gate_up experts (stored separately so we can
    /// drop the individual expert Vec after stacking).
    pub gate_up_ggml_dtype: candle_core::quantized::GgmlDType,
    /// GGML quant type for down experts.
    pub down_ggml_dtype: candle_core::quantized::GgmlDType,
    /// Number of experts to select per token.
    pub top_k: usize,
    /// MoE intermediate size per expert.
    pub moe_intermediate_size: usize,
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
    pub moe: MlxMoeWeights,
    pub norms: MlxLayerNorms,
    pub layer_scalar: MlxBuffer,
}

/// Per-layer KV cache buffers for the mlx-native path.
pub struct MlxKvCache {
    /// K cache `[num_kv_heads * head_dim, capacity]` F32 — row = kv_heads*head_dim,
    /// column = position. Stored as flat `[capacity * row_size]` F32.
    pub k: MlxBuffer,
    /// V cache, same layout as K.
    pub v: MlxBuffer,
    /// Number of KV heads for this layer.
    pub num_kv_heads: usize,
    /// Head dimension for this layer.
    pub head_dim: usize,
    /// Cache capacity (max_seq_len for global, sliding_window for sliding).
    pub capacity: usize,
    /// Whether this is a sliding window cache.
    pub is_sliding: bool,
    /// Current write position (next position to write).
    pub write_pos: usize,
    /// Number of valid positions in the cache.
    pub seq_len: usize,
}

/// Reusable activation buffers for one forward pass.
pub struct MlxActivationBuffers {
    /// Hidden state `[1, hidden_size]` F32.
    pub hidden: MlxBuffer,
    /// Scratch buffer for attention Q output `[1, num_heads * head_dim]` F32.
    pub attn_q: MlxBuffer,
    /// Scratch buffer for attention K output `[1, num_kv_heads * head_dim]` F32
    /// (sized for the largest layer — global with num_kv_heads=2, head_dim=512).
    pub attn_k: MlxBuffer,
    /// Scratch buffer for attention output after O projection `[1, hidden_size]` F32.
    pub attn_out: MlxBuffer,
    /// Scratch buffer for RMS norm output `[1, hidden_size]` F32.
    pub norm_out: MlxBuffer,
    /// Scratch buffer for residual `[1, hidden_size]` F32.
    pub residual: MlxBuffer,
    /// Scratch buffer for MLP gate output `[1, intermediate_size]` F32.
    pub mlp_gate: MlxBuffer,
    /// Scratch buffer for MLP up output `[1, intermediate_size]` F32.
    pub mlp_up: MlxBuffer,
    /// Scratch buffer for MLP fused output `[1, intermediate_size]` F32.
    pub mlp_fused: MlxBuffer,
    /// Second fused buffer for MoE expert batching.
    pub mlp_fused_1: MlxBuffer,
    /// Scratch buffer for MLP down output `[1, hidden_size]` F32.
    pub mlp_down: MlxBuffer,
    /// Scratch buffer for SDPA output `[1, num_heads, 1, head_dim]` F32.
    /// Sized for largest head config (16 heads * 512 head_dim for global).
    pub sdpa_out: MlxBuffer,
    /// RMS norm params buffer `[eps, dim]` as F32.
    pub norm_params: MlxBuffer,
    /// RoPE params buffer `[theta, head_dim, 0, 0]` as F32.
    pub rope_params_sliding: MlxBuffer,
    pub rope_params_global: MlxBuffer,
    /// Position buffer `[pos]` as U32 — single element for decode.
    pub position: MlxBuffer,
    /// Softcap params buffer (used if softcapping is configured).
    pub softcap_params: MlxBuffer,
    /// Argmax output index buffer `[1]` U32.
    pub argmax_index: MlxBuffer,
    /// Argmax output value buffer `[1]` F32.
    pub argmax_value: MlxBuffer,
    /// Argmax params buffer.
    pub argmax_params: MlxBuffer,
    /// Logits output buffer `[1, vocab_size]` F32.
    pub logits: MlxBuffer,
    /// Embedding scale buffer — single F32 with sqrt(hidden_size).
    pub embed_scale: MlxBuffer,
    /// MoE scratch: router logits `[1, num_experts]` F32.
    pub moe_router_logits: MlxBuffer,
    /// MoE scratch: softmax probs `[1, num_experts]` F32.
    pub moe_probs: MlxBuffer,
    /// MoE scratch: sorted indices `[1, num_experts]` U32.
    pub moe_sorted_indices: MlxBuffer,
    /// MoE scratch: expert gate_up output `[1, 2*moe_intermediate]` F32.
    pub moe_gate_up_out: MlxBuffer,
    /// MoE scratch: second expert gate_up output for batched dispatch.
    pub moe_gate_up_out_1: MlxBuffer,
    /// MoE scratch: expert down output `[1, hidden_size]` F32.
    pub moe_expert_out: MlxBuffer,
    /// MoE scratch: second expert down output for batched dispatch.
    pub moe_expert_out_1: MlxBuffer,
    /// MoE scratch: accumulated output `[1, hidden_size]` F32.
    pub moe_accum: MlxBuffer,
    /// MoE scratch: softmax params for router.
    pub moe_softmax_params: MlxBuffer,
    /// MoE scratch: norm output for router `[1, hidden_size]` F32.
    pub moe_norm_out: MlxBuffer,
    /// MoE scratch: expert ids buffer for _id kernel `[top_k]` U32.
    pub moe_expert_ids: MlxBuffer,
    /// MoE scratch: gate_up _id output `[top_k, 2*moe_intermediate]` F32.
    pub moe_gate_up_id_out: MlxBuffer,
    /// MoE scratch: down _id output `[top_k, hidden_size]` F32.
    pub moe_down_id_out: MlxBuffer,
    /// MoE scratch: swiglu output for _id path `[top_k, moe_intermediate]` F32.
    pub moe_swiglu_id_out: MlxBuffer,
    /// F16 scratch for lm_head GPU path: hidden state cast to F16 `[1, hidden_size]`.
    pub hidden_f16: MlxBuffer,
    /// F16 scratch for lm_head GPU path: logits output `[1, vocab_size]`.
    pub logits_f16: MlxBuffer,
}

/// All mlx-native weights for the full Gemma 4 model.
pub struct MlxModelWeights {
    pub embed_weight: MlxBuffer,
    pub layers: Vec<MlxDecoderLayerWeights>,
    pub final_norm: MlxBuffer,
    pub lm_head_f16: Option<MlxBuffer>,
    pub lm_head_f32: MlxBuffer,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_attention_heads: usize,
    pub rms_norm_eps: f32,
    pub final_logit_softcapping: Option<f32>,
    /// Per-layer KV caches.
    pub kv_caches: Vec<MlxKvCache>,
    /// Reusable activation buffers.
    pub activations: MlxActivationBuffers,
    /// Layer types (Sliding vs Full) for attention dispatch.
    pub layer_types: Vec<LayerType>,
    /// Sliding window size.
    pub sliding_window: usize,
    /// Per-layer head_dim (256 for sliding, 512 for global).
    pub head_dims: Vec<usize>,
    /// Per-layer num_kv_heads (8 for sliding, 2 for global).
    pub num_kv_heads: Vec<usize>,
    /// RoPE theta for sliding layers.
    pub rope_theta_sliding: f32,
    /// RoPE theta for global layers.
    pub rope_theta_global: f32,
    /// Number of MoE experts.
    pub num_experts: usize,
    /// Intermediate size for dense MLP.
    pub intermediate_size: usize,
    /// Intermediate size for MoE experts.
    pub moe_intermediate_size: usize,
    /// k_eq_v flag.
    pub k_eq_v: bool,
    /// Global-layer RoPE frequency factors from `rope_freqs.weight`.
    /// Shape `[global_head_dim/2]`.  Each value divides the per-pair
    /// frequency: `1.0` means normal rotation, `1e+30` → identity.
    pub rope_freq_factors: Vec<f32>,
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
        cfg: &Gemma4Config,
        mlx_device: &MlxDevice,
    ) -> Result<Self> {
        eprintln!("  Loading mlx-native weights from candle model...");

        // Embedding weight (F32)
        eprintln!("  Loading embed_weight...");
        let embed_weight = gpu::candle_tensor_to_mlx_buffer(
            model.embed_weight(),
            mlx_device,
        )?;

        eprintln!("  Loading final_norm...");
        // Final norm weight (F32)
        let final_norm = gpu::candle_tensor_to_mlx_buffer(
            model.final_norm_weight(),
            mlx_device,
        )?;

        // lm_head is tied to embed_weight. Create an F16 copy for GPU dense_gemm.
        eprintln!("  Creating F16 embed weight for GPU lm_head...");
        let lm_head_f16 = {
            let embed_f32: &[f32] = embed_weight.as_slice()
                .map_err(|e| anyhow::anyhow!("embed as_slice for f16 copy: {e}"))?;
            let n_elements = embed_f32.len();
            let f16_bytes = n_elements * 2; // f16 = 2 bytes per element
            let mut f16_buf = mlx_device.alloc_buffer(
                f16_bytes, mlx_native::DType::F16,
                vec![cfg.vocab_size, cfg.hidden_size],
            ).map_err(|e| anyhow::anyhow!("lm_head_f16 alloc: {e}"))?;
            // Convert F32 → F16 on CPU at load time (one-time cost).
            let dst_bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    f16_buf.contents_ptr() as *mut u8,
                    f16_bytes,
                )
            };
            for i in 0..n_elements {
                let f16_val = half::f16::from_f32(embed_f32[i]);
                let bits = f16_val.to_bits();
                dst_bytes[i * 2] = (bits & 0xFF) as u8;
                dst_bytes[i * 2 + 1] = (bits >> 8) as u8;
            }
            f16_buf
        };
        eprintln!("  F16 embed weight created ({} elements, {:.1} MB).",
            cfg.vocab_size * cfg.hidden_size,
            (cfg.vocab_size * cfg.hidden_size * 2) as f64 / 1e6);

        // Per-layer config
        let num_layers = model.num_layers();
        let mut layers = Vec::with_capacity(num_layers);
        let mut kv_caches = Vec::with_capacity(num_layers);
        let mut head_dims = Vec::with_capacity(num_layers);
        let mut num_kv_heads_vec = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            eprintln!("  mlx layer {}/{}: attn+mlp weights...", i + 1, num_layers);
            let refs = model.layer_weights(i);
            eprintln!("  mlx layer {}/{}: moe weights...", i + 1, num_layers);
            let moe_refs = model.moe_weights(i);

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

            // MoE weights: load per-expert quantized weights
            let mut expert_gate_up = Vec::with_capacity(cfg.num_experts);
            let mut expert_down = Vec::with_capacity(cfg.num_experts);
            for e in 0..cfg.num_experts {
                if e % 32 == 0 {
                    eprint!("\r  Loading mlx-native layer {}/{} experts {}/{}...",
                        i + 1, num_layers, e, cfg.num_experts);
                }
                let gu = load_qweight(
                    &moe_refs.expert_gate_up[e],
                    mlx_device,
                    &format!("layer{i}_expert{e}_gate_up"),
                )?;
                let dn = load_qweight(
                    &moe_refs.expert_down[e],
                    mlx_device,
                    &format!("layer{i}_expert{e}_down"),
                )?;
                expert_gate_up.push(gu);
                expert_down.push(dn);
            }
            let router_proj = load_qweight(
                moe_refs.router_proj,
                mlx_device,
                &format!("layer{i}_router_proj"),
            )?;
            let router_scale = gpu::candle_tensor_to_mlx_buffer(
                moe_refs.router_scale, mlx_device,
            )?;
            let per_expert_scale = gpu::candle_tensor_to_mlx_buffer(
                moe_refs.per_expert_scale, mlx_device,
            )?;

            // Capture GGML dtypes before potentially dropping expert vecs.
            let gate_up_ggml_dtype = expert_gate_up[0].info.ggml_dtype;
            let down_ggml_dtype = expert_down[0].info.ggml_dtype;

            // Build stacked expert weight buffers for fused _id dispatch.
            // Each expert's buffer is concatenated: [expert0, expert1, ..., expertN-1].
            let (stacked_gate_up, gate_up_expert_stride) = {
                let per_expert = expert_gate_up[0].buffer.byte_len();
                let total = per_expert * cfg.num_experts;
                match mlx_device.alloc_buffer(total, mlx_native::DType::U32, vec![total / 4]) {
                    Ok(mut stacked) => {
                        let dst: &mut [u8] = stacked.as_mut_slice()
                            .map_err(|e| anyhow::anyhow!("stacked gate_up: {e}"))?;
                        for (e, qw) in expert_gate_up.iter().enumerate() {
                            let src: &[u8] = qw.buffer.as_slice()
                                .map_err(|e| anyhow::anyhow!("expert gate_up read: {e}"))?;
                            dst[e * per_expert..(e + 1) * per_expert].copy_from_slice(src);
                        }
                        (Some(stacked), per_expert as u64)
                    }
                    Err(e) => {
                        eprintln!("  Warning: could not allocate stacked gate_up buffer: {e}");
                        (None, 0u64)
                    }
                }
            };
            let (stacked_down, down_expert_stride) = {
                let per_expert = expert_down[0].buffer.byte_len();
                let total = per_expert * cfg.num_experts;
                match mlx_device.alloc_buffer(total, mlx_native::DType::U32, vec![total / 4]) {
                    Ok(mut stacked) => {
                        let dst: &mut [u8] = stacked.as_mut_slice()
                            .map_err(|e| anyhow::anyhow!("stacked down: {e}"))?;
                        for (e, qw) in expert_down.iter().enumerate() {
                            let src: &[u8] = qw.buffer.as_slice()
                                .map_err(|e| anyhow::anyhow!("expert down read: {e}"))?;
                            dst[e * per_expert..(e + 1) * per_expert].copy_from_slice(src);
                        }
                        (Some(stacked), per_expert as u64)
                    }
                    Err(e) => {
                        eprintln!("  Warning: could not allocate stacked down buffer: {e}");
                        (None, 0u64)
                    }
                }
            };

            // Drop individual expert buffers after stacking to save ~12.9 GB.
            // The _id kernel dispatches via stacked buffers only; individual
            // buffers are dead weight once stacking completes.
            let expert_gate_up_live = if stacked_gate_up.is_some() {
                drop(expert_gate_up);
                Vec::new()
            } else {
                expert_gate_up  // fallback path uses individual buffers
            };
            let expert_down_live = if stacked_down.is_some() {
                drop(expert_down);
                Vec::new()
            } else {
                expert_down
            };

            let moe = MlxMoeWeights {
                expert_gate_up: expert_gate_up_live,
                expert_down: expert_down_live,
                stacked_gate_up,
                stacked_down,
                gate_up_expert_stride,
                down_expert_stride,
                router_proj,
                router_scale,
                per_expert_scale,
                gate_up_ggml_dtype,
                down_ggml_dtype,
                top_k: cfg.top_k_experts,
                moe_intermediate_size: cfg.moe_intermediate_size,
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

            // Per-layer config
            let hd = cfg.head_dim_for_layer(i);
            let nkv = cfg.num_kv_heads_for_layer(i);
            let is_full = cfg.is_full_attention(i);
            head_dims.push(hd);
            num_kv_heads_vec.push(nkv);

            // KV cache: sliding layers use sliding_window capacity,
            // global layers cap at a practical limit to avoid OOM.
            // 262144 × 5 global layers × 2(K+V) × 2 heads × 512 dim × 4 bytes = 10.7 GB
            // which OOMs a 64 GB machine. Cap at 8192 for now; expand if
            // longer-context generation is needed.
            let max_global_kv = 8192;
            let capacity = if is_full {
                cfg.max_position_embeddings.min(max_global_kv)
            } else {
                cfg.sliding_window
            };
            // KV cache shape: [num_kv_heads, capacity, head_dim] — head-major
            let cache_elements = nkv * capacity * hd;
            let cache_bytes = cache_elements * std::mem::size_of::<f32>();
            let k_cache = mlx_device.alloc_buffer(
                cache_bytes, mlx_native::DType::F32, vec![nkv, capacity, hd],
            ).map_err(|e| anyhow::anyhow!("KV cache K alloc failed: {e}"))?;
            let v_cache = mlx_device.alloc_buffer(
                cache_bytes, mlx_native::DType::F32, vec![nkv, capacity, hd],
            ).map_err(|e| anyhow::anyhow!("KV cache V alloc failed: {e}"))?;
            kv_caches.push(MlxKvCache {
                k: k_cache,
                v: v_cache,
                num_kv_heads: nkv,
                head_dim: hd,
                capacity,
                is_sliding: !is_full,
                write_pos: 0,
                seq_len: 0,
            });

            layers.push(MlxDecoderLayerWeights {
                attn,
                mlp,
                moe,
                norms,
                layer_scalar,
            });
        }
        eprintln!(
            "\r  Loaded {}/{} mlx-native layer weights (including MoE).    ",
            num_layers, num_layers
        );

        // Allocate activation buffers
        let activations = alloc_activation_buffers(mlx_device, cfg)?;

        // Dummy 1-element buffer for lm_head_f32 field — kept for compatibility.
        let lm_head_f32_dummy = mlx_device.alloc_buffer(4, mlx_native::DType::F32, vec![1])
            .map_err(|e| anyhow::anyhow!("lm_head dummy: {e}"))?;

        Ok(Self {
            embed_weight,
            layers,
            final_norm,
            lm_head_f16: Some(lm_head_f16),
            lm_head_f32: lm_head_f32_dummy,
            hidden_size: cfg.hidden_size,
            vocab_size: cfg.vocab_size,
            num_attention_heads: cfg.num_attention_heads,
            rms_norm_eps: cfg.rms_norm_eps as f32,
            final_logit_softcapping: model.softcapping().map(|v| v as f32),
            kv_caches,
            activations,
            layer_types: cfg.layer_types.clone(),
            sliding_window: cfg.sliding_window,
            head_dims,
            num_kv_heads: num_kv_heads_vec,
            rope_theta_sliding: cfg.rope_theta_sliding as f32,
            rope_theta_global: cfg.rope_theta_global as f32,
            num_experts: cfg.num_experts,
            intermediate_size: cfg.intermediate_size,
            moe_intermediate_size: cfg.moe_intermediate_size,
            k_eq_v: cfg.attention_k_eq_v,
            rope_freq_factors: model.rope_freqs_host().to_vec(),
        })
    }

    /// Clear all KV caches (for new generation).
    pub fn clear_kv_caches(&mut self) {
        for kv in &mut self.kv_caches {
            kv.write_pos = 0;
            kv.seq_len = 0;
        }
    }

    /// Run one decode step through mlx-native's GraphExecutor.
    ///
    /// MVP: one GraphSession per layer (30 sessions per forward pass).
    /// Each session encodes all ops for that layer, then commits.
    /// The final session handles the lm_head + argmax.
    ///
    /// Arguments:
    ///   - input_token: the token ID to embed
    ///   - seq_pos: position in the sequence (for RoPE and KV cache)
    ///   - gpu: the GpuContext holding the executor and registry
    ///   - profile: optional per-token profile accumulator
    ///
    /// Returns: the next token ID (greedy decode)
    pub fn forward_decode(
        &mut self,
        input_token: u32,
        seq_pos: usize,
        gpu: &mut GpuContext,
        profile: &mut Option<TokenProfile>,
    ) -> Result<u32> {
        let token_start = Instant::now();
        let hidden_size = self.hidden_size;
        let num_layers = self.layers.len();

        // Pre-allocate profile vectors if profiling
        if let Some(ref mut p) = profile {
            p.layer_s1_us = vec![0.0; num_layers];
            p.layer_cpu1_us = vec![0.0; num_layers];
            p.layer_s2_us = vec![0.0; num_layers];
            p.layer_cpu2_us = vec![0.0; num_layers];
            p.layer_s3_us = vec![0.0; num_layers];
            p.layer_cpu3_us = vec![0.0; num_layers];
            p.layer_s4_us = vec![0.0; num_layers];
            p.layer_cpu4_us = vec![0.0; num_layers];
            p.s1_dispatches = vec![0; num_layers];
            p.s2_dispatches = vec![0; num_layers];
            p.s3_dispatches = vec![0; num_layers];
            p.s4_dispatches = vec![0; num_layers];
        }

        // --- 1. Embedding lookup (CPU copy for single token) ---
        {
            let embed_src: &[f32] = self.embed_weight.as_slice()
                .map_err(|e| anyhow::anyhow!("embed as_slice: {e}"))?;
            let offset = input_token as usize * hidden_size;
            if offset + hidden_size > embed_src.len() {
                anyhow::bail!(
                    "Token {} out of range for embedding table (size {})",
                    input_token, embed_src.len() / hidden_size,
                );
            }
            let hidden_dst: &mut [f32] = self.activations.hidden.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("hidden as_mut_slice: {e}"))?;
            hidden_dst[..hidden_size].copy_from_slice(&embed_src[offset..offset + hidden_size]);
        }

        // Scale embedding by sqrt(hidden_size)
        {
            let scale = (hidden_size as f32).sqrt();
            let hidden: &mut [f32] = self.activations.hidden.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("hidden scale: {e}"))?;
            for v in hidden[..hidden_size].iter_mut() {
                *v *= scale;
            }
        }

        // --- 2. Transformer layers ---
        for layer_idx in 0..num_layers {
            self.forward_decode_layer(layer_idx, seq_pos, gpu, profile)?;
        }

        // --- 3. Final norm + lm_head + softcap + argmax ---
        let token_id = self.forward_decode_head(gpu, profile)?;

        if let Some(ref mut p) = profile {
            p.total_us = token_start.elapsed().as_secs_f64() * 1e6;
        }

        Ok(token_id)
    }

    /// Run one decoder layer with session-collapsed GPU dispatch.
    ///
    /// Session structure (4 sessions per layer):
    ///   Session 1: Q/K/V projections
    ///   CPU: head norms, V norm, RoPE, KV cache update
    ///   Session 2: SDPA → O-proj → GPU-norm(post-attn) → GPU-add(residual)
    ///              → GPU-norm(pre-FF) → gate → up → GPU-GELU(gate) → GPU-mul
    ///              → down
    ///   CPU: post-FF-norm1, pre-FF-norm2, MoE routing prep
    ///   Session 3: router_proj
    ///   CPU: softmax + top-k
    ///   Session 4: zero(accum) + for each expert:
    ///     gate_up matmul → GPU-SwiGLU → down matmul → GPU-accumulate
    ///   CPU: post-FF-norm2, combine, layer scalar
    fn forward_decode_layer(
        &mut self,
        layer_idx: usize,
        seq_pos: usize,
        gpu: &mut GpuContext,
        profile: &mut Option<TokenProfile>,
    ) -> Result<()> {
        let hs = self.hidden_size;
        let hd = self.head_dims[layer_idx];
        let nkv = self.num_kv_heads[layer_idx];
        let nh = self.num_attention_heads;
        let is_sliding = self.layer_types[layer_idx] == LayerType::Sliding;
        let eps = self.rms_norm_eps;

        // -- a. Pre-attention norm (CPU — 2816 elements, microseconds) --
        cpu_rms_norm_weighted(
            &self.activations.hidden,
            &self.layers[layer_idx].norms.input_layernorm,
            &mut self.activations.norm_out,
            hs, eps,
        )?;

        // =====================================================================
        // SESSION 1: Q, K, V projections (3 GPU matmuls in one encoder)
        // =====================================================================
        let s1_start = Instant::now();
        let mut s1_dispatches = 0usize;
        {
            let (exec, reg) = gpu.split();
            let dev = exec.device();
            let mut s = exec.begin().map_err(|e| anyhow::anyhow!("S1 begin: {e}"))?;
            dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                &self.layers[layer_idx].attn.q_proj, &mut self.activations.attn_q, 1)?;
            s1_dispatches += 1;
            dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                &self.layers[layer_idx].attn.k_proj, &mut self.activations.attn_k, 1)?;
            s1_dispatches += 1;
            let v_is_k = self.layers[layer_idx].attn.v_proj.is_none();
            if !v_is_k {
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    self.layers[layer_idx].attn.v_proj.as_ref().unwrap(),
                    &mut self.activations.moe_expert_out, 1)?;
                s1_dispatches += 1;
            }
            s.finish().map_err(|e| anyhow::anyhow!("S1 finish: {e}"))?;
        }
        let s1_elapsed = s1_start.elapsed();
        if let Some(ref mut p) = profile {
            p.layer_s1_us[layer_idx] = s1_elapsed.as_secs_f64() * 1e6;
            p.s1_dispatches[layer_idx] = s1_dispatches;
        }

        // -- CPU: Q/K head norms --
        let cpu1_start = Instant::now();
        {
            let q: &mut [f32] = self.activations.attn_q.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("q: {e}"))?;
            let qw: &[f32] = self.layers[layer_idx].attn.q_norm_weight.as_slice()
                .map_err(|e| anyhow::anyhow!("qw: {e}"))?;
            for h in 0..nh {
                rms_norm_cpu(&mut q[h*hd..(h+1)*hd], &qw[..hd], eps);
            }
        }
        {
            let k: &mut [f32] = self.activations.attn_k.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("k: {e}"))?;
            let kw: &[f32] = self.layers[layer_idx].attn.k_norm_weight.as_slice()
                .map_err(|e| anyhow::anyhow!("kw: {e}"))?;
            for h in 0..nkv {
                rms_norm_cpu(&mut k[h*hd..(h+1)*hd], &kw[..hd], eps);
            }
        }

        // -- CPU: V = copy from K if k_eq_v, then unit norm --
        let v_is_k = self.layers[layer_idx].attn.v_proj.is_none();
        if v_is_k {
            let k: &[f32] = self.activations.attn_k.as_slice()
                .map_err(|e| anyhow::anyhow!("v=k: {e}"))?;
            let v: &mut [f32] = self.activations.moe_expert_out.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("v: {e}"))?;
            v[..nkv*hd].copy_from_slice(&k[..nkv*hd]);
        }
        {
            let v: &mut [f32] = self.activations.moe_expert_out.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("vnorm: {e}"))?;
            for h in 0..nkv {
                rms_norm_unit_cpu(&mut v[h*hd..(h+1)*hd], eps);
            }
        }

        // -- CPU: RoPE (freq_factors for global layers require CPU path) --
        let theta = if is_sliding { self.rope_theta_sliding } else { self.rope_theta_global };
        let ff = if is_sliding { None } else { Some(self.rope_freq_factors.as_slice()) };
        apply_rope_neox_cpu(
            self.activations.attn_q.as_mut_slice().map_err(|e| anyhow::anyhow!("rq: {e}"))?,
            nh, hd, seq_pos, theta, ff,
        );
        apply_rope_neox_cpu(
            self.activations.attn_k.as_mut_slice().map_err(|e| anyhow::anyhow!("rk: {e}"))?,
            nkv, hd, seq_pos, theta, ff,
        );

        // -- CPU: KV cache update --
        let kv_is_sliding = self.kv_caches[layer_idx].is_sliding;
        let kv_write_pos = self.kv_caches[layer_idx].write_pos;
        let kv_capacity = self.kv_caches[layer_idx].capacity;
        let cache_pos = if kv_is_sliding {
            kv_write_pos % kv_capacity
        } else {
            kv_write_pos
        };
        {
            let ks: &[f32] = self.activations.attn_k.as_slice().map_err(|e| anyhow::anyhow!("ks: {e}"))?;
            let kc: &mut [f32] = self.kv_caches[layer_idx].k.as_mut_slice().map_err(|e| anyhow::anyhow!("kc: {e}"))?;
            for h in 0..nkv {
                let src_start = h * hd;
                let dst_start = h * kv_capacity * hd + cache_pos * hd;
                kc[dst_start..dst_start + hd].copy_from_slice(&ks[src_start..src_start + hd]);
            }
        }
        {
            let vs: &[f32] = self.activations.moe_expert_out.as_slice().map_err(|e| anyhow::anyhow!("vs: {e}"))?;
            let vc: &mut [f32] = self.kv_caches[layer_idx].v.as_mut_slice().map_err(|e| anyhow::anyhow!("vc: {e}"))?;
            for h in 0..nkv {
                let src_start = h * hd;
                let dst_start = h * kv_capacity * hd + cache_pos * hd;
                vc[dst_start..dst_start + hd].copy_from_slice(&vs[src_start..src_start + hd]);
            }
        }
        self.kv_caches[layer_idx].write_pos += 1;
        self.kv_caches[layer_idx].seq_len = self.kv_caches[layer_idx].seq_len.saturating_add(1)
            .min(kv_capacity);
        let kv_seq_len = self.kv_caches[layer_idx].seq_len;
        let cpu1_elapsed = cpu1_start.elapsed();
        if let Some(ref mut p) = profile {
            p.layer_cpu1_us[layer_idx] = cpu1_elapsed.as_secs_f64() * 1e6;
        }

        // =====================================================================
        // SESSION 2: SDPA → O-proj → GPU-norm(post-attn) → GPU-add(residual)
        //            → GPU-norm(pre-FF) → gate → up → GPU-GELU → GPU-mul → down
        // =====================================================================
        let s2_start = Instant::now();
        let mut s2_dispatches = 0usize;
        {
            let (exec, reg) = gpu.split();
            let dev = exec.device();
            let metal_dev = dev.metal_device();
            let mut s = exec.begin().map_err(|e| anyhow::anyhow!("S2 begin: {e}"))?;

            // SDPA
            if is_sliding {
                let p = SdpaSlidingParams {
                    n_heads: nh as u32, n_kv_heads: nkv as u32, head_dim: hd as u32,
                    seq_len: 1, kv_seq_len: kv_seq_len as u32,
                    window_size: self.sliding_window as u32, scale: 1.0,
                    kv_capacity: kv_capacity as u32,
                };
                mlx_native::ops::sdpa_sliding::sdpa_sliding(
                    s.encoder_mut(), reg, dev,
                    &self.activations.attn_q, &self.kv_caches[layer_idx].k,
                    &self.kv_caches[layer_idx].v, &self.activations.sdpa_out, &p, 1,
                ).map_err(|e| anyhow::anyhow!("sdpa_sliding: {e}"))?;
            } else {
                let p = SdpaParams {
                    n_heads: nh as u32, n_kv_heads: nkv as u32, head_dim: hd as u32,
                    seq_len: 1, kv_seq_len: kv_seq_len as u32, scale: 1.0,
                    kv_capacity: kv_capacity as u32,
                };
                s.sdpa(reg, dev, &self.activations.attn_q, &self.kv_caches[layer_idx].k,
                    &self.kv_caches[layer_idx].v, &self.activations.sdpa_out, &p, 1,
                ).map_err(|e| anyhow::anyhow!("sdpa: {e}"))?;
            }
            s2_dispatches += 1; // SDPA

            // O-proj: sdpa_out → attn_out
            dispatch_qmatmul(&mut s, reg, dev, &self.activations.sdpa_out,
                &self.layers[layer_idx].attn.o_proj, &mut self.activations.attn_out, 1)?;
            s2_dispatches += 1; // O-proj

            // GPU post-attention norm: attn_out → norm_out
            s.rms_norm(
                reg, metal_dev,
                &self.activations.attn_out,
                &self.layers[layer_idx].norms.post_attention_layernorm,
                &self.activations.norm_out,
                &self.activations.norm_params,
                1, hs as u32,
            ).map_err(|e| anyhow::anyhow!("GPU post-attn norm: {e}"))?;
            s2_dispatches += 1; // rms_norm

            // GPU residual add: hidden + norm_out → residual
            s.elementwise_add(
                reg, metal_dev,
                &self.activations.hidden, &self.activations.norm_out,
                &self.activations.residual, hs, mlx_native::DType::F32,
            ).map_err(|e| anyhow::anyhow!("GPU residual add: {e}"))?;
            s2_dispatches += 1; // add

            // GPU pre-feedforward norm: residual → norm_out
            s.rms_norm(
                reg, metal_dev,
                &self.activations.residual,
                &self.layers[layer_idx].norms.pre_feedforward_layernorm,
                &self.activations.norm_out,
                &self.activations.norm_params,
                1, hs as u32,
            ).map_err(|e| anyhow::anyhow!("GPU pre-FF norm: {e}"))?;
            s2_dispatches += 1; // rms_norm

            // Dense MLP: gate and up projections (norm_out → mlp_gate, mlp_up)
            dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                &self.layers[layer_idx].mlp.gate_proj, &mut self.activations.mlp_gate, 1)?;
            s2_dispatches += 1; // gate
            dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                &self.layers[layer_idx].mlp.up_proj, &mut self.activations.mlp_up, 1)?;
            s2_dispatches += 1; // up

            // GPU GELU on gate: mlp_gate → mlp_fused (same element count)
            s.gelu(
                reg, metal_dev,
                &self.activations.mlp_gate, &self.activations.mlp_fused,
            ).map_err(|e| anyhow::anyhow!("GPU GELU: {e}"))?;
            s2_dispatches += 1; // gelu

            // GPU mul: mlp_fused * mlp_up → mlp_fused (SwiGLU)
            s.elementwise_mul(
                reg, metal_dev,
                &self.activations.mlp_fused, &self.activations.mlp_up,
                &self.activations.mlp_fused, self.intermediate_size,
                mlx_native::DType::F32,
            ).map_err(|e| anyhow::anyhow!("GPU SwiGLU mul: {e}"))?;
            s2_dispatches += 1; // mul

            // down_proj: mlp_fused → mlp_down
            dispatch_qmatmul(&mut s, reg, dev, &self.activations.mlp_fused,
                &self.layers[layer_idx].mlp.down_proj, &mut self.activations.mlp_down, 1)?;
            s2_dispatches += 1; // down

            s.finish().map_err(|e| anyhow::anyhow!("S2 finish: {e}"))?;
        }
        let s2_elapsed = s2_start.elapsed();
        if let Some(ref mut p) = profile {
            p.layer_s2_us[layer_idx] = s2_elapsed.as_secs_f64() * 1e6;
            p.s2_dispatches[layer_idx] = s2_dispatches;
        }

        // -- CPU: Post-feedforward norm 1 on mlp_down --
        let cpu2_start = Instant::now();
        cpu_rms_norm_weighted(
            &self.activations.mlp_down, &self.layers[layer_idx].norms.post_feedforward_layernorm_1,
            &mut self.activations.norm_out, hs, eps,
        )?;
        {
            let src: &[f32] = self.activations.norm_out.as_slice().map_err(|e| anyhow::anyhow!("cp: {e}"))?;
            let dst: &mut [f32] = self.activations.mlp_down.as_mut_slice().map_err(|e| anyhow::anyhow!("cp: {e}"))?;
            dst[..hs].copy_from_slice(&src[..hs]);
        }

        // -- MoE path (sessions 3-4) --
        let (s3_us, cpu3_us, s4_us, s3_disp, s4_disp) =
            self.forward_decode_moe(layer_idx, gpu)?;

        // -- Combine MLP + MoE, final norm, residual, layer scalar (CPU) --
        {
            let mlp: &[f32] = self.activations.mlp_down.as_slice().map_err(|e| anyhow::anyhow!("m: {e}"))?;
            let moe: &[f32] = self.activations.moe_accum.as_slice().map_err(|e| anyhow::anyhow!("mo: {e}"))?;
            let combined: &mut [f32] = self.activations.attn_out.as_mut_slice().map_err(|e| anyhow::anyhow!("c: {e}"))?;
            for i in 0..hs { combined[i] = mlp[i] + moe[i]; }
        }
        cpu_rms_norm_weighted(
            &self.activations.attn_out, &self.layers[layer_idx].norms.post_feedforward_layernorm,
            &mut self.activations.norm_out, hs, eps,
        )?;
        cpu_add(&self.activations.residual, &self.activations.norm_out, &mut self.activations.hidden, hs)?;
        // Layer scalar
        {
            let scalar: &[f32] = self.layers[layer_idx].layer_scalar.as_slice()
                .map_err(|e| anyhow::anyhow!("ls: {e}"))?;
            let hidden: &mut [f32] = self.activations.hidden.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("h: {e}"))?;
            if scalar.len() == 1 {
                let s = scalar[0];
                for v in hidden[..hs].iter_mut() { *v *= s; }
            } else {
                for i in 0..hs { hidden[i] *= scalar[i]; }
            }
        }
        let cpu2_and_4_elapsed = cpu2_start.elapsed();
        if let Some(ref mut p) = profile {
            // cpu2 = total elapsed minus S3+CPU3+S4 times (which are measured inside forward_decode_moe)
            let total_cpu2_4 = cpu2_and_4_elapsed.as_secs_f64() * 1e6;
            let cpu4_us = total_cpu2_4 - s3_us - cpu3_us - s4_us;
            p.layer_cpu2_us[layer_idx] = cpu4_us.max(0.0); // pre-MoE CPU norm + post-MoE combine
            p.layer_s3_us[layer_idx] = s3_us;
            p.layer_cpu3_us[layer_idx] = cpu3_us;
            p.layer_s4_us[layer_idx] = s4_us;
            p.layer_cpu4_us[layer_idx] = cpu4_us.max(0.0);
            p.s3_dispatches[layer_idx] = s3_disp;
            p.s4_dispatches[layer_idx] = s4_disp;
        }

        Ok(())
    }

    /// MoE forward pass for one layer.
    ///
    /// Session structure (2 sessions):
    ///   Session 3: router_proj
    ///   CPU: softmax + top-k (128 elements, microseconds)
    ///   Session 4: zero_buffer(accum) + for each expert:
    ///     qmatmul(gate_up) → moe_swiglu_fused → qmatmul(down) → moe_accumulate
    ///   CPU: post-FF-norm2
    ///
    /// Optimization: all expert ops (gate_up, SwiGLU, down, accumulate) are
    /// encoded into a SINGLE GPU session.  Within one command buffer, Metal
    /// executes compute dispatches in order, so buffer reuse between experts
    /// is safe — each dispatch completes before the next reads the same buffer.
    /// This eliminates 2*top_k = 16 sessions per layer.
    /// Returns (s3_us, cpu3_us, s4_us, s3_dispatches, s4_dispatches).
    fn forward_decode_moe(
        &mut self,
        layer_idx: usize,
        gpu: &mut GpuContext,
    ) -> Result<(f64, f64, f64, usize, usize)> {
        let hs = self.hidden_size;
        let top_k = self.layers[layer_idx].moe.top_k;
        let moe_int = self.layers[layer_idx].moe.moe_intermediate_size;

        // Pre-feedforward norm 2 on residual for MoE input (CPU).
        // Note: residual was written by GPU in session 2 and is readable
        // after S2 finish().
        cpu_rms_norm_weighted(
            &self.activations.residual,
            &self.layers[layer_idx].norms.pre_feedforward_layernorm_2,
            &mut self.activations.moe_norm_out,
            hs, self.rms_norm_eps,
        )?;

        // =====================================================================
        // SESSION 3: Router projection
        // =====================================================================
        // CPU: prepare router input (unit norm + scale on residual)
        let s3_start = Instant::now();
        let (top_k_indices, top_k_weights) = {
            let residual: &[f32] = self.activations.residual.as_slice()
                .map_err(|e| anyhow::anyhow!("router residual: {e}"))?;
            let mut router_input = vec![0.0f32; hs];
            router_input.copy_from_slice(&residual[..hs]);
            rms_norm_unit_cpu(&mut router_input, self.rms_norm_eps);

            let scale_factor = (hs as f32).powf(-0.5);
            let router_scale: &[f32] = self.layers[layer_idx].moe.router_scale.as_slice()
                .map_err(|e| anyhow::anyhow!("router scale: {e}"))?;
            for i in 0..hs {
                router_input[i] *= router_scale[i] * scale_factor;
            }

            // Write scaled router input into norm_out
            {
                let dst: &mut [f32] = self.activations.norm_out.as_mut_slice()
                    .map_err(|e| anyhow::anyhow!("ri: {e}"))?;
                dst[..hs].copy_from_slice(&router_input);
            }
            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("S3 begin: {e}"))?;
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    &self.layers[layer_idx].moe.router_proj,
                    &mut self.activations.moe_router_logits, 1)?;
                s.finish().map_err(|e| anyhow::anyhow!("S3 finish: {e}"))?;
            }

            // CPU: softmax + top-k on router logits (128 elements, microseconds)
            let logits: &[f32] = self.activations.moe_router_logits.as_slice()
                .map_err(|e| anyhow::anyhow!("router logits read: {e}"))?;
            let num_experts = self.num_experts;

            let max_val = logits[..num_experts].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if max_val.is_nan() || max_val.is_infinite() {
                anyhow::bail!(
                    "Router logits corrupted at layer {layer_idx}: max_val={max_val}"
                );
            }
            let mut probs = vec![0.0f32; num_experts];
            let mut sum = 0.0f32;
            for i in 0..num_experts {
                probs[i] = (logits[i] - max_val).exp();
                sum += probs[i];
            }
            for p in probs.iter_mut() {
                *p /= sum;
            }

            let mut indices: Vec<usize> = (0..num_experts).collect();
            indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
            let top_indices: Vec<u32> = indices[..top_k].iter().map(|&i| i as u32).collect();
            let top_probs: Vec<f32> = indices[..top_k].iter().map(|&i| probs[i]).collect();

            let top_sum: f32 = top_probs.iter().sum();
            let top_weights: Vec<f32> = top_probs.iter().map(|p| p / top_sum).collect();

            (top_indices, top_weights)
        };
        // S3 includes the GPU session + CPU softmax/topk
        let s3_total_elapsed = s3_start.elapsed().as_secs_f64() * 1e6;
        // We can't separate S3 GPU from CPU3 perfectly here, but S3 GPU is
        // just 1 qmatmul (128 outputs, ~10us), so attribute most to GPU.
        // For now, we'll report the whole block as S3 (GPU-dominated).
        let s3_us = s3_total_elapsed;
        let cpu3_us = 0.0; // Included in s3_us for simplicity

        let per_expert_scale: Vec<f32> = {
            let s: &[f32] = self.layers[layer_idx].moe.per_expert_scale.as_slice()
                .map_err(|e| anyhow::anyhow!("per_expert_scale: {e}"))?;
            s.to_vec()
        };

        // =====================================================================
        // SESSION 4: All expert dispatches in ONE session.
        // =====================================================================
        // If stacked weight buffers are available, use the fused _id kernel
        // path (2 qmatmul dispatches instead of 2*top_k). Otherwise fall
        // back to the per-expert loop.

        // Zero the accumulator on CPU (fast for 2816 elements)
        {
            let accum: &mut [f32] = self.activations.moe_accum.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("moe accum zero: {e}"))?;
            accum[..hs].fill(0.0);
        }

        let s4_start = Instant::now();
        let mut s4_dispatches = 0usize;

        let use_fused_id = self.layers[layer_idx].moe.stacked_gate_up.is_some()
            && self.layers[layer_idx].moe.stacked_down.is_some();

        if use_fused_id {
            // --- Fused _id path: 2 qmatmul_id dispatches ---

            // Write expert ids to GPU buffer
            {
                let ids_dst: &mut [u32] = self.activations.moe_expert_ids.as_mut_slice()
                    .map_err(|e| anyhow::anyhow!("moe_expert_ids write: {e}"))?;
                for k_idx in 0..top_k {
                    ids_dst[k_idx] = top_k_indices[k_idx];
                }
            }

            let ggml_type_gu = gpu::candle_ggml_to_mlx(
                self.layers[layer_idx].moe.gate_up_ggml_dtype)?;
            let ggml_type_dn = gpu::candle_ggml_to_mlx(
                self.layers[layer_idx].moe.down_ggml_dtype)?;

            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let metal_dev = dev.metal_device();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("S4-moe begin: {e}"))?;

                // Zero the accumulator on GPU
                mlx_native::ops::moe_dispatch::moe_zero_buffer_encode(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.moe_accum, hs,
                ).map_err(|e| anyhow::anyhow!("moe zero_buffer: {e}"))?;
                s4_dispatches += 1;

                // 1) gate_up _id: input=[1,K] ids=[top_k] -> output=[top_k, 2*moe_int]
                let gu_params = mlx_native::GgmlQuantizedMatmulIdParams {
                    n_tokens: 1,
                    top_k: top_k as u32,
                    n: (2 * moe_int) as u32,
                    k: hs as u32,
                    n_experts: self.num_experts as u32,
                    expert_stride: self.layers[layer_idx].moe.gate_up_expert_stride,
                    ggml_type: ggml_type_gu,
                };
                s.quantized_matmul_id_ggml(
                    reg, dev,
                    &self.activations.moe_norm_out,
                    self.layers[layer_idx].moe.stacked_gate_up.as_ref().unwrap(),
                    &self.activations.moe_expert_ids,
                    &mut self.activations.moe_gate_up_id_out,
                    &gu_params,
                ).map_err(|e| anyhow::anyhow!("gate_up _id: {e}"))?;
                s4_dispatches += 1;

                // 2) SwiGLU for each expert slot.
                // gate_up_id_out is [top_k, 2*moe_int] flat. Each slot's gate_up
                // is at offset slot * 2*moe_int. We use byte-offset slicing.
                for k_idx in 0..top_k {
                    let offset_bytes = k_idx * 2 * moe_int * std::mem::size_of::<f32>();
                    let swiglu_out_offset = k_idx * moe_int * std::mem::size_of::<f32>();
                    mlx_native::ops::moe_dispatch::moe_swiglu_fused_encode_offset(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.moe_gate_up_id_out,
                        offset_bytes,
                        &self.activations.moe_swiglu_id_out,
                        swiglu_out_offset,
                        moe_int,
                    ).map_err(|e| anyhow::anyhow!("moe swiglu _id: {e}"))?;
                    s4_dispatches += 1;
                }

                // 3) down _id: input=[top_k, moe_int] ids=[top_k] -> output=[top_k, hs]
                // Treat each expert slot as a separate "token" with top_k_param=1.
                let dn_params = mlx_native::GgmlQuantizedMatmulIdParams {
                    n_tokens: top_k as u32,
                    top_k: 1,
                    n: hs as u32,
                    k: moe_int as u32,
                    n_experts: self.num_experts as u32,
                    expert_stride: self.layers[layer_idx].moe.down_expert_stride,
                    ggml_type: ggml_type_dn,
                };
                s.quantized_matmul_id_ggml(
                    reg, dev,
                    &self.activations.moe_swiglu_id_out,
                    self.layers[layer_idx].moe.stacked_down.as_ref().unwrap(),
                    &self.activations.moe_expert_ids,
                    &mut self.activations.moe_down_id_out,
                    &dn_params,
                ).map_err(|e| anyhow::anyhow!("down _id: {e}"))?;
                s4_dispatches += 1;

                // 4) Weighted accumulate for each expert slot.
                // down_id_out is [top_k, hs]. Each slot's output is at offset slot*hs.
                for k_idx in 0..top_k {
                    let eid = top_k_indices[k_idx] as usize;
                    let w = top_k_weights[k_idx] * per_expert_scale[eid];
                    if w.abs() < 1e-10 {
                        continue;
                    }
                    let offset_bytes = k_idx * hs * std::mem::size_of::<f32>();
                    mlx_native::ops::moe_dispatch::moe_accumulate_encode_offset(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.moe_accum,
                        &self.activations.moe_down_id_out,
                        offset_bytes,
                        w, hs,
                    ).map_err(|e| anyhow::anyhow!("moe accumulate _id: {e}"))?;
                    s4_dispatches += 1;
                }

                s.finish().map_err(|e| anyhow::anyhow!("S4-moe finish: {e}"))?;
            }
        } else {
            // --- Fallback: per-expert loop (original path) ---
            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let metal_dev = dev.metal_device();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("S4-moe begin: {e}"))?;

                // Zero the accumulator on GPU
                mlx_native::ops::moe_dispatch::moe_zero_buffer_encode(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.moe_accum, hs,
                ).map_err(|e| anyhow::anyhow!("moe zero_buffer: {e}"))?;
                s4_dispatches += 1;

                for k_idx in 0..top_k {
                    let eid = top_k_indices[k_idx] as usize;
                    let w = top_k_weights[k_idx] * per_expert_scale[eid];

                    if w.abs() < 1e-10 {
                        continue;
                    }

                    // gate_up matmul
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.moe_norm_out,
                        &self.layers[layer_idx].moe.expert_gate_up[eid],
                        &mut self.activations.moe_gate_up_out, 1)?;
                    s4_dispatches += 1;

                    // GPU SwiGLU
                    mlx_native::ops::moe_dispatch::moe_swiglu_fused_encode(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.moe_gate_up_out,
                        &self.activations.mlp_fused,
                        moe_int,
                    ).map_err(|e| anyhow::anyhow!("moe swiglu: {e}"))?;
                    s4_dispatches += 1;

                    // down matmul
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.mlp_fused,
                        &self.layers[layer_idx].moe.expert_down[eid],
                        &mut self.activations.moe_expert_out, 1)?;
                    s4_dispatches += 1;

                    // GPU weighted accumulate
                    mlx_native::ops::moe_dispatch::moe_accumulate_encode(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.moe_accum,
                        &self.activations.moe_expert_out,
                        w, hs,
                    ).map_err(|e| anyhow::anyhow!("moe accumulate: {e}"))?;
                    s4_dispatches += 1;
                }

                s.finish().map_err(|e| anyhow::anyhow!("S4-moe finish: {e}"))?;
            }
        }
        let s4_us = s4_start.elapsed().as_secs_f64() * 1e6;

        // Post-feedforward norm 2 on MoE output (CPU)
        cpu_rms_norm_weighted(
            &self.activations.moe_accum,
            &self.layers[layer_idx].norms.post_feedforward_layernorm_2,
            &mut self.activations.moe_norm_out,
            hs, self.rms_norm_eps,
        )?;
        {
            let src: &[f32] = self.activations.moe_norm_out.as_slice().map_err(|e| anyhow::anyhow!("cp: {e}"))?;
            let dst: &mut [f32] = self.activations.moe_accum.as_mut_slice().map_err(|e| anyhow::anyhow!("cp: {e}"))?;
            dst[..hs].copy_from_slice(&src[..hs]);
        }

        Ok((s3_us, cpu3_us, s4_us, 1usize, s4_dispatches))
    }

    /// Final norm + GPU lm_head (dense GEMM F16) + softcap + argmax.
    ///
    /// Uses one GPU session for: cast(F32→F16) + dense_gemm_f16 + cast(F16→F32).
    /// Falls back to CPU lm_head if no F16 weight is available.
    fn forward_decode_head(
        &mut self,
        gpu: &mut GpuContext,
        profile: &mut Option<TokenProfile>,
    ) -> Result<u32> {
        let hs = self.hidden_size;
        let vocab_size = self.vocab_size;

        // Final RMS norm (CPU — 2816 elements, microseconds)
        cpu_rms_norm_weighted(
            &self.activations.hidden, &self.final_norm,
            &mut self.activations.norm_out, hs, self.rms_norm_eps,
        )?;

        // LM head: GPU dense GEMM (F16) or CPU fallback
        let head_gpu_start = Instant::now();
        let mut head_dispatches = 0usize;
        if let Some(ref lm_head_f16) = self.lm_head_f16 {
            let (exec, reg) = gpu.split();
            let dev = exec.device();
            let metal_dev = dev.metal_device();
            let mut s = exec.begin().map_err(|e| anyhow::anyhow!("lm_head begin: {e}"))?;

            mlx_native::ops::elementwise::cast(
                s.encoder_mut(), reg, metal_dev,
                &self.activations.norm_out,
                &self.activations.hidden_f16,
                hs,
                CastDirection::F32ToF16,
            ).map_err(|e| anyhow::anyhow!("cast F32->F16: {e}"))?;
            head_dispatches += 1;

            let gemm_params = DenseGemmF16Params {
                m: 1,
                n: vocab_size as u32,
                k: hs as u32,
            };
            mlx_native::ops::dense_gemm::dispatch_dense_gemm_f16(
                s.encoder_mut(), reg, metal_dev,
                &self.activations.hidden_f16,
                lm_head_f16,
                &self.activations.logits_f16,
                &gemm_params,
            ).map_err(|e| anyhow::anyhow!("dense_gemm_f16: {e}"))?;
            head_dispatches += 1;

            mlx_native::ops::elementwise::cast(
                s.encoder_mut(), reg, metal_dev,
                &self.activations.logits_f16,
                &self.activations.logits,
                vocab_size,
                CastDirection::F16ToF32,
            ).map_err(|e| anyhow::anyhow!("cast F16->F32: {e}"))?;
            head_dispatches += 1;

            s.finish().map_err(|e| anyhow::anyhow!("lm_head finish: {e}"))?;
        } else {
            self.lm_head_cpu()?;
        }
        let head_gpu_us = head_gpu_start.elapsed().as_secs_f64() * 1e6;

        // Softcapping + Argmax (CPU)
        let head_cpu_start = Instant::now();
        if let Some(cap) = self.final_logit_softcapping {
            let logits: &mut [f32] = self.activations.logits.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("softcap logits: {e}"))?;
            for v in logits[..vocab_size].iter_mut() {
                *v = cap * (*v / cap).tanh();
            }
        }

        let logits: &[f32] = self.activations.logits.as_slice()
            .map_err(|e| anyhow::anyhow!("argmax logits: {e}"))?;
        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for i in 0..vocab_size {
            if logits[i] > best_val {
                best_val = logits[i];
                best_idx = i as u32;
            }
        }
        let head_cpu_us = head_cpu_start.elapsed().as_secs_f64() * 1e6;

        if let Some(ref mut p) = profile {
            p.head_session_us = head_gpu_us;
            p.head_cpu_us = head_cpu_us;
            p.head_dispatches = head_dispatches;
        }

        Ok(best_idx)
    }

    /// CPU lm_head matmul: logits = hidden @ embed_weight.T
    fn lm_head_cpu(&mut self) -> Result<()> {
        let hidden_size = self.hidden_size;
        let vocab_size = self.vocab_size;
        let hidden: &[f32] = self.activations.norm_out.as_slice()
            .map_err(|e| anyhow::anyhow!("lm_head hidden: {e}"))?;
        let weight: &[f32] = self.embed_weight.as_slice()
            .map_err(|e| anyhow::anyhow!("lm_head weight: {e}"))?;
        let logits: &mut [f32] = self.activations.logits.as_mut_slice()
            .map_err(|e| anyhow::anyhow!("lm_head logits: {e}"))?;

        // logits[v] = sum_k(hidden[k] * weight[v * hidden_size + k])
        for v in 0..vocab_size {
            let mut sum = 0.0f32;
            let row = &weight[v * hidden_size..(v + 1) * hidden_size];
            for k in 0..hidden_size {
                sum += hidden[k] * row[k];
            }
            logits[v] = sum;
        }
        Ok(())
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
// Activation buffer allocation
// ---------------------------------------------------------------------------

/// Allocate all reusable activation buffers for the forward pass.
fn alloc_activation_buffers(
    device: &MlxDevice,
    cfg: &Gemma4Config,
) -> Result<MlxActivationBuffers> {
    let hs = cfg.hidden_size;
    let max_hd = cfg.global_head_dim; // 512
    let num_heads = cfg.num_attention_heads; // 16
    let max_kv_heads = cfg.num_key_value_heads.max(cfg.num_global_key_value_heads);
    let vocab = cfg.vocab_size;
    let interm = cfg.intermediate_size;
    let moe_interm = cfg.moe_intermediate_size;
    let num_experts = cfg.num_experts;

    let f32_sz = std::mem::size_of::<f32>();
    let u32_sz = std::mem::size_of::<u32>();

    let alloc_f32 = |n: usize, name: &str| -> Result<MlxBuffer> {
        device.alloc_buffer(n * f32_sz, mlx_native::DType::F32, vec![n])
            .map_err(|e| anyhow::anyhow!("alloc {name} ({n} f32): {e}"))
    };
    let alloc_u32 = |n: usize, name: &str| -> Result<MlxBuffer> {
        device.alloc_buffer(n * u32_sz, mlx_native::DType::U32, vec![n])
            .map_err(|e| anyhow::anyhow!("alloc {name} ({n} u32): {e}"))
    };

    // RMS norm params: [eps, dim] as f32
    let mut norm_params = alloc_f32(2, "norm_params")?;
    {
        let p: &mut [f32] = norm_params.as_mut_slice()
            .map_err(|e| anyhow::anyhow!("norm_params init: {e}"))?;
        p[0] = cfg.rms_norm_eps as f32;
        p[1] = hs as f32;
    }

    // RoPE params: [theta, head_dim, 0, 0] as f32
    let mut rope_params_sliding = alloc_f32(4, "rope_params_sliding")?;
    {
        let p: &mut [f32] = rope_params_sliding.as_mut_slice()
            .map_err(|e| anyhow::anyhow!("rope_params_sliding init: {e}"))?;
        p[0] = cfg.rope_theta_sliding as f32;
        p[1] = cfg.head_dim as f32;
        p[2] = 0.0;
        p[3] = 0.0;
    }
    let mut rope_params_global = alloc_f32(4, "rope_params_global")?;
    {
        let p: &mut [f32] = rope_params_global.as_mut_slice()
            .map_err(|e| anyhow::anyhow!("rope_params_global init: {e}"))?;
        p[0] = cfg.rope_theta_global as f32;
        p[1] = cfg.global_head_dim as f32;
        p[2] = 0.0;
        p[3] = 0.0;
    }

    // Embedding scale
    let mut embed_scale = alloc_f32(1, "embed_scale")?;
    {
        let p: &mut [f32] = embed_scale.as_mut_slice()
            .map_err(|e| anyhow::anyhow!("embed_scale init: {e}"))?;
        p[0] = (hs as f32).sqrt();
    }

    // Softmax params for MoE router: [num_experts] placeholder
    let moe_softmax_params = alloc_f32(2, "moe_softmax_params")?;

    // Softcap params
    let softcap_params = alloc_f32(2, "softcap_params")?;

    // Argmax params
    let argmax_params = alloc_f32(2, "argmax_params")?;

    Ok(MlxActivationBuffers {
        hidden: alloc_f32(hs, "hidden")?,
        attn_q: alloc_f32(num_heads * max_hd, "attn_q")?,
        attn_k: alloc_f32(max_kv_heads * max_hd, "attn_k")?,
        attn_out: alloc_f32(hs, "attn_out")?,
        norm_out: alloc_f32(hs, "norm_out")?,
        residual: alloc_f32(hs, "residual")?,
        mlp_gate: alloc_f32(interm, "mlp_gate")?,
        mlp_up: alloc_f32(interm, "mlp_up")?,
        mlp_fused: alloc_f32(interm.max(moe_interm), "mlp_fused")?,
        mlp_fused_1: alloc_f32(moe_interm, "mlp_fused_1")?,
        mlp_down: alloc_f32(hs, "mlp_down")?,
        sdpa_out: alloc_f32(num_heads * max_hd, "sdpa_out")?,
        norm_params,
        rope_params_sliding,
        rope_params_global,
        position: alloc_u32(1, "position")?,
        softcap_params,
        argmax_index: alloc_u32(1, "argmax_index")?,
        argmax_value: alloc_f32(1, "argmax_value")?,
        argmax_params,
        logits: alloc_f32(vocab, "logits")?,
        embed_scale,
        moe_router_logits: alloc_f32(num_experts, "moe_router_logits")?,
        moe_probs: alloc_f32(num_experts, "moe_probs")?,
        moe_sorted_indices: alloc_u32(num_experts, "moe_sorted_indices")?,
        moe_gate_up_out: alloc_f32(2 * moe_interm, "moe_gate_up_out")?,
        moe_gate_up_out_1: alloc_f32(2 * moe_interm, "moe_gate_up_out_1")?,
        moe_expert_out: alloc_f32(hs.max(max_kv_heads * max_hd), "moe_expert_out")?,
        moe_expert_out_1: alloc_f32(hs, "moe_expert_out_1")?,
        moe_accum: alloc_f32(hs, "moe_accum")?,
        moe_softmax_params,
        moe_norm_out: alloc_f32(hs, "moe_norm_out")?,
        // Fused _id dispatch buffers (sized for top_k = cfg.top_k_experts)
        moe_expert_ids: alloc_u32(cfg.top_k_experts, "moe_expert_ids")?,
        moe_gate_up_id_out: alloc_f32(cfg.top_k_experts * 2 * moe_interm, "moe_gate_up_id_out")?,
        moe_down_id_out: alloc_f32(cfg.top_k_experts * hs, "moe_down_id_out")?,
        moe_swiglu_id_out: alloc_f32(cfg.top_k_experts * moe_interm, "moe_swiglu_id_out")?,
        hidden_f16: device.alloc_buffer(hs * 2, mlx_native::DType::F16, vec![1, hs])
            .map_err(|e| anyhow::anyhow!("alloc hidden_f16 ({hs} f16): {e}"))?,
        logits_f16: device.alloc_buffer(vocab * 2, mlx_native::DType::F16, vec![1, vocab])
            .map_err(|e| anyhow::anyhow!("alloc logits_f16 ({vocab} f16): {e}"))?,
    })
}

// ---------------------------------------------------------------------------
// CPU helper functions for single-token decode ops
// ---------------------------------------------------------------------------

/// RMS norm with learned weight on MlxBuffers. Reads input + weight, writes output.
fn cpu_rms_norm_weighted(
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &mut MlxBuffer,
    dim: usize,
    eps: f32,
) -> Result<()> {
    let inp: &[f32] = input.as_slice()
        .map_err(|e| anyhow::anyhow!("cpu_rms_norm input: {e}"))?;
    let w: &[f32] = weight.as_slice()
        .map_err(|e| anyhow::anyhow!("cpu_rms_norm weight: {e}"))?;
    let out: &mut [f32] = output.as_mut_slice()
        .map_err(|e| anyhow::anyhow!("cpu_rms_norm output: {e}"))?;
    let mut sum_sq = 0.0f32;
    for i in 0..dim { sum_sq += inp[i] * inp[i]; }
    let rms = (sum_sq / dim as f32 + eps).sqrt().recip();
    for i in 0..dim { out[i] = inp[i] * rms * w[i]; }
    Ok(())
}

/// Elementwise add on MlxBuffers: output = a + b.
fn cpu_add(a: &MlxBuffer, b: &MlxBuffer, output: &mut MlxBuffer, n: usize) -> Result<()> {
    let av: &[f32] = a.as_slice().map_err(|e| anyhow::anyhow!("cpu_add a: {e}"))?;
    let bv: &[f32] = b.as_slice().map_err(|e| anyhow::anyhow!("cpu_add b: {e}"))?;
    let ov: &mut [f32] = output.as_mut_slice().map_err(|e| anyhow::anyhow!("cpu_add o: {e}"))?;
    for i in 0..n { ov[i] = av[i] + bv[i]; }
    Ok(())
}

/// In-place RMS norm with learned weight: x = x * rsqrt(mean(x^2) + eps) * w
fn rms_norm_cpu(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len();
    let mut sum_sq = 0.0f32;
    for &v in x.iter() {
        sum_sq += v * v;
    }
    let rms = (sum_sq / n as f32 + eps).sqrt().recip();
    for i in 0..n {
        x[i] = x[i] * rms * w[i];
    }
}

/// In-place unit RMS norm (no learned weight): x = x * rsqrt(mean(x^2) + eps)
fn rms_norm_unit_cpu(x: &mut [f32], eps: f32) {
    let n = x.len();
    let mut sum_sq = 0.0f32;
    for &v in x.iter() {
        sum_sq += v * v;
    }
    let rms = (sum_sq / n as f32 + eps).sqrt().recip();
    for v in x.iter_mut() {
        *v *= rms;
    }
}

/// Apply Neox-convention RoPE in-place for a single position.
/// data layout: [num_heads, head_dim] contiguous.
/// Neox pairs (d[i], d[i + half_dim]) for i in 0..half_dim.
///
/// `freq_factors`: optional per-pair frequency divisor `[half_dim]`.
/// For global layers, pair indices with `freq_factors[i] == 1e+30` produce
/// `theta ≈ 0 → cos=1, sin=0 → identity rotation`.  For sliding layers,
/// pass `None` (equivalent to all-ones).
fn apply_rope_neox_cpu(
    data: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    pos: usize,
    theta: f32,
    freq_factors: Option<&[f32]>,
) {
    let half_dim = head_dim / 2;
    for h in 0..num_heads {
        let base = h * head_dim;
        for i in 0..half_dim {
            let base_freq = (pos as f32) / theta.powf(2.0 * i as f32 / head_dim as f32);
            let freq = match freq_factors {
                Some(ff) => base_freq / ff[i],
                None => base_freq,
            };
            let cos_val = freq.cos();
            let sin_val = freq.sin();
            let x0 = data[base + i];
            let x1 = data[base + i + half_dim];
            data[base + i] = x0 * cos_val - x1 * sin_val;
            data[base + i + half_dim] = x1 * cos_val + x0 * sin_val;
        }
    }
}

/// GELU activation with PyTorch tanh approximation.
/// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn gelu_tanh(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.7978845608028654f32;
    let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

// ---------------------------------------------------------------------------
// Phase 5 forward pass status
// ---------------------------------------------------------------------------
//
// DONE (Phase 5 step 2 — functional):
// - [x] Weight bridge: candle QTensor/Tensor -> MlxBuffer (gpu.rs)
// - [x] GpuContext creation and lifecycle (gpu.rs)
// - [x] Forward pass orchestrator with full op coverage
// - [x] All quantized matmul, SDPA, embedding, KV cache, argmax
//
// DONE (Phase 5 step 3 — session collapse + GPU lm_head):
// - [x] Session collapse: dense MLP path merged into 1 session
//       - Merged SDPA + O-proj + GPU-norm + GPU-add + GPU-norm
//         + gate + up + GPU-GELU + GPU-mul + down into 1 session
//       - Router proj in its own session (CPU softmax/top-k between)
// - [x] GPU lm_head via dense_gemm_f16 (replaces CPU 262K×2816 dot products)
//       - F16 embed weight created at load time
//       - Cast F32→F16 + GEMM + cast F16→F32 in one GPU session
// - [x] GPU GELU for dense MLP (encoded in merged session 2)
// - [x] GPU elementwise_mul for SwiGLU (encoded in merged session 2)
// - [x] GPU rms_norm for post-attn and pre-FF norms (encoded in merged session 2)
// - [x] GPU elementwise_add for residual (encoded in merged session 2)
//
// DONE (Phase 5 step 4 — MoE single-session):
// - [x] GPU moe_swiglu_fused: gelu(gate_up[0:N]) * gate_up[N:2N] in one kernel
// - [x] All expert ops (gate_up, SwiGLU, down, accumulate) in ONE session
//       - Eliminated 2*top_k = 16 sessions per MoE layer
//       - Buffer reuse is safe: Metal dispatches execute in order within
//         a single command buffer
//       - Total sessions: 4 per layer (QKV, SDPA+MLP, router, experts) + 1 head
//       - 30 layers × 4 + 1 = 121 sessions per token (down from 571)
//
// OPTIMIZATIONS deferred to Phase 5c:
// - [ ] GPU RoPE with freq_factors support (needs new kernel)
// - [ ] GPU embedding via embedding_gather kernel
// - [ ] GPU MoE routing (softmax + argsort on GPU)
// - [ ] GPU argmax via dispatch_argmax_f32
// - [ ] GPU KV cache update via dispatch_kv_cache_copy
