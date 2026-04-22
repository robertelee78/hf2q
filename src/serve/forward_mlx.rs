//! mlx-native forward pass for Gemma 4 inference.
//!
//! ADR-008: routes the entire forward pass through mlx-native's
//! `GraphExecutor`, encoding all ops into a single Metal command buffer
//! with one `commit_and_wait` per token.
//!
//! # Architecture
//!
//! At model load time, weights are loaded directly from a GGUF file into
//! mlx-native `MlxBuffer` instances via `load_from_gguf()`.  No candle
//! involvement.
//! At inference time, the forward pass uses mlx-native's `GraphSession`
//! to encode all ops (qmatmul, RmsNorm, RoPE, SDPA, etc.) into batched
//! command buffers to minimize GPU sync points.
//!
//! # Status
//!
//! Phase 5e: fused kernel dispatch reduction — 1082 → 842 dispatches/token.
//!
//! Six kernel fusions applied per layer (single session, all ops on GPU):
//! - Fused Q head-norm + RoPE (f32, with freq_factors): 2 dispatches → 1
//! - Fused K head-norm + RoPE (f32, with freq_factors): 2 dispatches → 1
//! - Fused post-attention norm + residual add: 2 dispatches → 1
//! - Fused post-FF norm 2 + MLP+MoE combine add: 2 dispatches → 1
//! - Fused end-of-layer norm + residual add + scalar mul: 3 dispatches → 1
//! - Fused MoE routing (softmax + argsort + gather_topk): 3 dispatches → 1
//!
//! Total: 8 dispatches/layer saved × 30 layers = 240 fewer dispatches.
//! Coherence preserved (byte-identical output to Phase 5d baseline).

use anyhow::Result;
use mlx_native::{
    GgmlQuantizedMatmulParams, GraphSession, MlxBuffer, MlxDevice,
};
use mlx_native::ops::flash_attn_vec_tq::FlashAttnVecTqParams;
// CODEBOOK_4BIT is now embedded directly in the Metal shader as a constant array.
// No Rust-side centroid table construction needed.
use mlx_native::ops::dense_gemm::DenseGemmF16Params;
use mlx_native::ops::elementwise::CastDirection;
use std::time::Instant;

use crate::debug::{dumps, INVESTIGATION_ENV};
use super::config::{Gemma4Config, LayerType};
use super::gpu::{GpuContext, QuantWeightInfo};

// ---------------------------------------------------------------------------
// Profiling support (HF2Q_MLX_PROFILE=1)
// ---------------------------------------------------------------------------

/// Check if profiling is enabled via environment variable.
fn profiling_enabled() -> bool {
    INVESTIGATION_ENV.mlx_profile
}

/// Accumulated per-kernel-type timing for one token.
#[derive(Default, Clone)]
pub struct KernelTypeProfile {
    /// Per-layer timings in microseconds, indexed by layer.
    pub qkv_matmuls_us: Vec<f64>,
    pub head_norms_rope_us: Vec<f64>,
    pub kv_cache_copy_us: Vec<f64>,
    pub sdpa_us: Vec<f64>,
    pub o_proj_us: Vec<f64>,
    pub mlp_matmuls_us: Vec<f64>,
    pub moe_us: Vec<f64>,
    pub norms_adds_us: Vec<f64>,
    /// Head session timings.
    pub lm_head_us: f64,
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

        // Count actual sessions used (non-zero timings indicate a session was used)
        let actual_sessions = if s2_avg + s3_avg + s4_avg + head_gpu_avg < 1.0 {
            1  // Single session for entire forward pass
        } else {
            num_layers * 2 + 1
        };
        eprintln!("║ {} session(s)/token (single-session mode)", actual_sessions);
        eprintln!("║");
        eprintln!("║ Session breakdown (avg across {num_layers} layers, {n} tokens):");
        eprintln!("║   S1 (QKV+attn+MLP):  {:8.1} us ({:5.2} ms total)", s1_avg / num_layers as f64, s1_avg / 1000.0);
        eprintln!("║   CPU1 (eliminated):   {:8.1} us ({:5.2} ms total)", cpu1_avg / num_layers as f64, cpu1_avg / 1000.0);
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
            eprintln!("║   S1 (QKV+attn+MLP): {:.1} us/dispatch", s1_avg / s1_disp);
        }
        if s2_disp > 0.0 {
            eprintln!("║   S2 (unused):     {:.1} us/dispatch", s2_avg / s2_disp);
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
        Ok(GgmlQuantizedMatmulParams {
            m,
            n: self.info.rows as u32,
            k: self.info.cols as u32,
            ggml_type: self.info.ggml_dtype,
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
    /// Per-expert scale `[num_experts]` F32.
    pub per_expert_scale: MlxBuffer,
    /// GGML quant type for gate_up experts (stored separately so we can
    /// drop the individual expert Vec after stacking).
    pub gate_up_ggml_dtype: mlx_native::GgmlType,
    /// GGML quant type for down experts.
    pub down_ggml_dtype: mlx_native::GgmlType,
    /// Number of experts to select per token.
    pub top_k: usize,
    /// MoE intermediate size per expert.
    pub moe_intermediate_size: usize,
    /// Pre-computed router combined weight: `router_scale[i] * (hidden_size^-0.5)`.
    /// Used by GPU `rms_norm` to compute the router input in one dispatch:
    ///   `output = unit_norm(residual) * router_combined_weight`
    /// This replaces the 3-step CPU sequence: unit_norm → scale → mul.
    pub router_combined_weight: MlxBuffer,
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

/// All mlx-native weights for one decoder layer, plus per-layer config
/// that used to live as parallel Vecs on `MlxModelWeights`.
pub struct MlxDecoderLayerWeights {
    pub attn: MlxAttentionWeights,
    pub mlp: MlxMlpWeights,
    pub moe: MlxMoeWeights,
    pub norms: MlxLayerNorms,
    pub layer_scalar: MlxBuffer,
    /// Head dim for this layer (Gemma-4: 256 for sliding, 512 for global).
    pub head_dim: usize,
    /// KV heads for this layer (Gemma-4: 8 for sliding, 2 for global).
    pub num_kv_heads: usize,
    /// Sliding vs Full attention — drives SDPA dispatch and KV cache layout.
    pub layer_type: LayerType,
}

/// Per-layer KV cache buffers for the mlx-native path (TurboQuant compressed).
///
/// ADR-007 Phase 1.2: KV cache is stored as 4-bit nibble-packed indices
/// with per-position F32 norms, replacing F16 dense buffers.  This halves
/// KV memory bandwidth during SDPA and enables 262K context.
pub struct MlxKvCache {
    /// K packed indices `[num_kv_heads, capacity, head_dim/2]` U8 (nibble-packed).
    pub k_packed: MlxBuffer,
    /// K per-position norms `[num_kv_heads, capacity]` F32.
    pub k_norms: MlxBuffer,
    /// V packed indices `[num_kv_heads, capacity, head_dim/2]` U8 (nibble-packed).
    pub v_packed: MlxBuffer,
    /// V per-position norms `[num_kv_heads, capacity]` F32.
    pub v_norms: MlxBuffer,
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
    /// Scratch buffer for MLP down output `[1, hidden_size]` F32.
    pub mlp_down: MlxBuffer,
    /// Scratch buffer for SDPA output `[1, num_heads, 1, head_dim]` F32.
    /// Sized for largest head config (16 heads * 512 head_dim for global).
    pub sdpa_out: MlxBuffer,
    /// Temporary buffer for SDPA NWG>1 partial results (reduce kernel input).
    pub sdpa_tmp: MlxBuffer,
    /// RMS norm params buffer `[eps, dim]` as F32.
    pub norm_params: MlxBuffer,
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
    /// MoE scratch: router logits `[1, num_experts]` F32.
    pub moe_router_logits: MlxBuffer,
    /// MoE scratch: expert down output `[1, hidden_size]` F32.
    pub moe_expert_out: MlxBuffer,
    /// MoE scratch: accumulated output `[1, hidden_size]` F32.
    pub moe_accum: MlxBuffer,
    /// MoE scratch: norm output for router `[1, hidden_size]` F32.
    pub moe_norm_out: MlxBuffer,
    /// Router norm output `[1, hidden_size]` F32 — separate from `norm_out` to
    /// allow router norm to run concurrent with pre-FF norm 1 (which writes norm_out).
    pub router_norm_out: MlxBuffer,
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
    // --- Session merge buffers (S1+S2 collapse) ---
    /// Per-head norm params for sliding layers: `[eps, sliding_head_dim]` F32.
    pub norm_params_sliding_hd: MlxBuffer,
    /// Per-head norm params for global layers: `[eps, global_head_dim]` F32.
    pub norm_params_global_hd: MlxBuffer,
    /// GPU buffer holding global-layer freq_factors `[global_head_dim/2]` F32.
    pub rope_freq_factors_gpu: MlxBuffer,
    /// Dedicated V projection output buffer `[max_kv_heads * max_hd]` F32.
    /// Separates V from moe_expert_out to avoid aliasing in merged session.
    pub attn_v: MlxBuffer,
    /// Scratch buffer for Q after per-head norm `[num_heads * max_hd]` F32.
    pub attn_q_normed: MlxBuffer,
    /// Scratch buffer for K after per-head norm `[max_kv_heads * max_hd]` F32.
    pub attn_k_normed: MlxBuffer,
    /// MoE scratch: pre-scaled routing weights for weighted_sum kernel `[top_k]` F32.
    pub moe_routing_weights_gpu: MlxBuffer,
}

/// All mlx-native weights for the full Gemma 4 model.
pub struct MlxModelWeights {
    pub embed_weight: MlxBuffer,
    pub layers: Vec<MlxDecoderLayerWeights>,
    pub final_norm: MlxBuffer,
    pub lm_head_f16: Option<MlxBuffer>,
    /// Optional Q8_0-quantized lm_head (gated on HF2Q_LMHEAD_Q8=1 at load).
    /// When present and the env var is still set at decode time, used instead
    /// of lm_head_f16 via dispatch_qmatmul. Halves weight memory traffic vs F16.
    pub lm_head_q8: Option<MlxQWeight>,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_attention_heads: usize,
    pub rms_norm_eps: f32,
    pub final_logit_softcapping: Option<f32>,
    /// Per-layer KV caches.
    pub kv_caches: Vec<MlxKvCache>,
    /// Reusable activation buffers.
    pub activations: MlxActivationBuffers,
    /// Sliding window size.
    pub sliding_window: usize,
    /// RoPE theta for sliding layers.
    pub rope_theta_sliding: f32,
    /// RoPE theta for global layers.
    pub rope_theta_global: f32,
    /// Number of MoE experts.
    pub num_experts: usize,
    /// Intermediate size for dense MLP.
    pub intermediate_size: usize,
    /// Dense F32 KV buffers per layer for decode (ADR-009 Track 3).
    ///
    /// When set (by `forward_prefill`), `forward_decode` uses dense SDPA
    /// instead of TQ-packed SDPA. Each layer has K and V in head-major
    /// layout `[nkv_heads, capacity, head_dim]`.
    ///
    /// Per-layer capacity: sliding layers use ring-buffer mode sized to
    /// `sliding_window` (writes wrap at `seq_pos % sliding_window`);
    /// global layers use a linear buffer sized to `seq_len + max_tokens`.
    /// Attention is permutation-invariant over cached K,V (RoPE is baked
    /// in before caching), so the ring's slot order doesn't matter for
    /// correctness — the kernel just attends to all populated slots.
    pub dense_kvs: Option<Vec<DenseKvBuffers>>,
    /// Tmp buffer for flash_attn_vec when using dense decode.
    pub dense_sdpa_tmp: Option<MlxBuffer>,
}

/// Per-layer dense F32 KV buffers for dense attention path (ADR-009).
pub struct DenseKvBuffers {
    pub k: MlxBuffer,
    pub v: MlxBuffer,
    /// Capacity (positions) in this layer's cache. Sliding layers use
    /// ring-buffer mode: capacity == sliding_window, writes wrap.
    /// Global layers use linear mode: capacity >= seq_len + max_tokens.
    pub capacity: usize,
    /// True if this is a sliding layer (ring-buffer semantics).
    pub is_sliding: bool,
}

impl MlxModelWeights {
    /// Load all model weights directly from a GGUF file into mlx-native
    /// MlxBuffers.
    ///
    /// `progress` drives the default-mode in-place `\r`-overwrite progress
    /// line on stderr; it is a no-op when stderr isn't a TTY or verbosity > 0
    /// (tracing debug events then cover per-layer detail).
    ///
    /// ADR-008 Phase 2: replaces `load_from_candle()` — weights go
    /// GGUF → MlxBuffer with zero candle involvement.
    pub fn load_from_gguf(
        gguf: &mlx_native::gguf::GgufFile,
        cfg: &Gemma4Config,
        gpu: &mut GpuContext,
        progress: &mut crate::serve::header::LoadProgress,
    ) -> Result<Self> {
        let mlx_device = gpu.device();
        tracing::debug!("Loading mlx-native weights directly from GGUF...");

        // --- Embedding weight (F32) ---
        tracing::debug!("Loading embed_weight");
        let embed_weight = gguf.load_tensor_f32("token_embd.weight", mlx_device)
            .map_err(|e| anyhow::anyhow!("embed: {e}"))?;

        // --- Final norm (F32) ---
        tracing::debug!("Loading final_norm");
        let final_norm = gguf.load_tensor_f32("output_norm.weight", mlx_device)
            .map_err(|e| anyhow::anyhow!("final_norm: {e}"))?;

        // --- lm_head: auto-pick Q8_0 vs F16 based on model size ---
        //
        // lm_head is memory-bandwidth-bound at batch=1; Q8_0 halves the
        // weight traffic vs F16 and recovers ~12% decode throughput on
        // Gemma-4-26B (1.47 GB F16 → 784 MB Q8). Raw Q8 occasionally flips
        // a near-tiebreak (pad-emit mode, see ADR-010) — the rerank path
        // (HF2Q_LMHEAD_RERANK; default on when Q8 is active) recovers the
        // exact F16 trajectory by reranking top candidates on CPU using
        // the F32 embed_weight already resident for the embedding gather.
        //
        // Heuristic: enable Q8_0 when F16 weight would exceed 256 MB and
        // hidden_size % 32 == 0. Smaller models skip Q8 because the head
        // time is already negligible.
        //
        // Env overrides:
        //   HF2Q_LMHEAD_Q8=1   force Q8 (errors if hidden_size % 32 != 0)
        //   HF2Q_LMHEAD_Q8=0   force F16 (escape hatch)
        //   HF2Q_LMHEAD_RERANK=0   disable rerank (raw Q8 argmax, unsafe)
        //   (unset)            auto-detect by size
        let lm_head_f16_bytes = cfg.vocab_size * cfg.hidden_size * 2;
        // HF2Q_LMHEAD_Q8 is a category-2 operator knob — documented in
        // docs/operator-env-vars.md and docs/shipping-contract.md. It is
        // intentionally read directly here (not via InvestigationEnv) because
        // it is part of the supported user-facing product surface.
        let q8_env = std::env::var("HF2Q_LMHEAD_Q8").ok();
        let compare_mode = INVESTIGATION_ENV.lmhead_compare;
        let use_q8 = match q8_env.as_deref() {
            Some("1") => true,
            Some("0") => false,
            _ => {
                // Auto: Q8 when F16 weight would exceed 256 MB and the
                // shape is Q8-compatible.
                lm_head_f16_bytes > 256 * 1024 * 1024 && cfg.hidden_size % 32 == 0
            }
        };

        // Decide which buffers to allocate. Compare mode always keeps F16
        // (needed as the oracle for A/B), and Q8 if requested.
        let need_q8 = use_q8;
        let need_f16 = !use_q8 || compare_mode;
        if use_q8 && cfg.hidden_size % 32 != 0 {
            anyhow::bail!(
                "HF2Q_LMHEAD_Q8=1 requires hidden_size % 32 == 0 (got {})",
                cfg.hidden_size);
        }

        let lm_head_q8: Option<MlxQWeight> = if need_q8 {
            let source = match q8_env.as_deref() {
                Some("1") => "forced",
                _ => "auto",
            };
            tracing::info!("Quantizing lm_head to Q8_0 ({} — F16 size {:.1} MB)",
                source, lm_head_f16_bytes as f64 / 1e6);
            let embed_f32: &[f32] = embed_weight.as_slice()
                .map_err(|e| anyhow::anyhow!("embed as_slice for q8 quantize: {e}"))?;
            let rows = cfg.vocab_size;
            let cols = cfg.hidden_size;
            let blocks_per_row = cols / 32;
            let block_bytes: usize = 34;
            let total_bytes = rows * blocks_per_row * block_bytes;
            let q_buf = mlx_device.alloc_buffer(
                total_bytes, mlx_native::DType::U8,
                vec![rows, blocks_per_row * block_bytes],
            ).map_err(|e| anyhow::anyhow!("lm_head_q8 alloc: {e}"))?;
            let dst_bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    q_buf.contents_ptr() as *mut u8,
                    total_bytes,
                )
            };
            for r in 0..rows {
                let row_src = &embed_f32[r * cols..(r + 1) * cols];
                let row_dst = &mut dst_bytes[r * blocks_per_row * block_bytes
                    ..(r + 1) * blocks_per_row * block_bytes];
                for b in 0..blocks_per_row {
                    let block_src = &row_src[b * 32..(b + 1) * 32];
                    let amax = block_src.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
                    let inv_d = if d != 0.0 { 1.0 / d } else { 0.0 };
                    let block_off = b * block_bytes;
                    let d_bits = half::f16::from_f32(d).to_bits();
                    row_dst[block_off] = (d_bits & 0xFF) as u8;
                    row_dst[block_off + 1] = (d_bits >> 8) as u8;
                    for (i, &v) in block_src.iter().enumerate() {
                        let q = (v * inv_d).round().clamp(-127.0, 127.0) as i8;
                        row_dst[block_off + 2 + i] = q as u8;
                    }
                }
            }
            tracing::info!("Q8_0 lm_head created ({:.1} MB, {:.2}× smaller than F16){}",
                total_bytes as f64 / 1e6,
                lm_head_f16_bytes as f64 / total_bytes as f64,
                if compare_mode { " [COMPARE MODE — F16 also resident]" } else { "" });
            Some(MlxQWeight {
                buffer: q_buf,
                info: QuantWeightInfo {
                    ggml_dtype: mlx_native::GgmlType::Q8_0,
                    rows,
                    cols,
                },
            })
        } else {
            None
        };

        let lm_head_f16: Option<MlxBuffer> = if need_f16 {
            let reason = if use_q8 && compare_mode {
                "COMPARE MODE oracle".to_string()
            } else if q8_env.as_deref() == Some("0") {
                "forced — HF2Q_LMHEAD_Q8=0".to_string()
            } else if cfg.hidden_size % 32 != 0 {
                format!("auto — hidden_size {} not divisible by 32", cfg.hidden_size)
            } else {
                format!("auto — F16 size {:.1} MB ≤ 256 MB threshold",
                    lm_head_f16_bytes as f64 / 1e6)
            };
            eprintln!("  Creating F16 embed weight for GPU lm_head ({})...", reason);
            let embed_f32: &[f32] = embed_weight.as_slice()
                .map_err(|e| anyhow::anyhow!("embed as_slice for f16 copy: {e}"))?;
            let n_elements = embed_f32.len();
            let f16_buf = mlx_device.alloc_buffer(
                lm_head_f16_bytes, mlx_native::DType::F16,
                vec![cfg.vocab_size, cfg.hidden_size],
            ).map_err(|e| anyhow::anyhow!("lm_head_f16 alloc: {e}"))?;
            let dst_bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    f16_buf.contents_ptr() as *mut u8,
                    lm_head_f16_bytes,
                )
            };
            for i in 0..n_elements {
                let f16_val = half::f16::from_f32(embed_f32[i]);
                let bits = f16_val.to_bits();
                dst_bytes[i * 2] = (bits & 0xFF) as u8;
                dst_bytes[i * 2 + 1] = (bits >> 8) as u8;
            }
            eprintln!("  F16 embed weight created ({} elements, {:.1} MB).",
                n_elements, lm_head_f16_bytes as f64 / 1e6);
            Some(f16_buf)
        } else {
            None
        };

        // --- Per-layer weights ---
        let num_layers = cfg.num_hidden_layers;
        let mut layers = Vec::with_capacity(num_layers);
        let mut kv_caches = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            tracing::debug!("GGUF layer {}/{}: loading weights", i + 1, num_layers);

            // -- Attention quantized weights --
            let q_proj = load_gguf_qweight(gguf, &format!("blk.{i}.attn_q.weight"), mlx_device)?;
            let k_proj = load_gguf_qweight(gguf, &format!("blk.{i}.attn_k.weight"), mlx_device)?;
            let v_proj = if cfg.is_full_attention(i) && cfg.attention_k_eq_v {
                None
            } else {
                Some(load_gguf_qweight(gguf, &format!("blk.{i}.attn_v.weight"), mlx_device)?)
            };
            let o_proj = load_gguf_qweight(gguf, &format!("blk.{i}.attn_output.weight"), mlx_device)?;

            // -- Attention head norms (F32) --
            let q_norm_weight = gguf.load_tensor_f32(
                &format!("blk.{i}.attn_q_norm.weight"), mlx_device,
            ).map_err(|e| anyhow::anyhow!("layer {i} q_norm: {e}"))?;
            let k_norm_weight = gguf.load_tensor_f32(
                &format!("blk.{i}.attn_k_norm.weight"), mlx_device,
            ).map_err(|e| anyhow::anyhow!("layer {i} k_norm: {e}"))?;

            let attn = MlxAttentionWeights {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm_weight,
                k_norm_weight,
            };

            // -- Dense MLP (quantized) --
            let gate_proj = load_gguf_qweight(gguf, &format!("blk.{i}.ffn_gate.weight"), mlx_device)?;
            let up_proj = load_gguf_qweight(gguf, &format!("blk.{i}.ffn_up.weight"), mlx_device)?;
            let down_proj = load_gguf_qweight(gguf, &format!("blk.{i}.ffn_down.weight"), mlx_device)?;

            let mlp = MlxMlpWeights {
                gate_proj,
                up_proj,
                down_proj,
            };

            // -- MoE expert weights (3D tensors, already stacked in GGUF) --
            let gu_name = format!("blk.{i}.ffn_gate_up_exps.weight");
            let gu_info = gguf.tensor_info(&gu_name)
                .ok_or_else(|| anyhow::anyhow!("missing {gu_name}"))?;
            let stacked_gate_up_buf = gguf.load_tensor(&gu_name, mlx_device)
                .map_err(|e| anyhow::anyhow!("load {gu_name}: {e}"))?;
            let gate_up_expert_stride = stacked_gate_up_buf.byte_len() / cfg.num_experts;
            let gate_up_ggml_dtype = gu_info.ggml_type;

            let dn_name = format!("blk.{i}.ffn_down_exps.weight");
            let dn_info = gguf.tensor_info(&dn_name)
                .ok_or_else(|| anyhow::anyhow!("missing {dn_name}"))?;
            let stacked_down_buf = gguf.load_tensor(&dn_name, mlx_device)
                .map_err(|e| anyhow::anyhow!("load {dn_name}: {e}"))?;
            let down_expert_stride = stacked_down_buf.byte_len() / cfg.num_experts;
            let down_ggml_dtype = dn_info.ggml_type;

            if (i + 1) % 5 == 0 || i == 0 {
                tracing::debug!("GGUF layer {}/{}: MoE experts loaded (stacked, {:.1} MB + {:.1} MB)",
                    i + 1, num_layers,
                    stacked_gate_up_buf.byte_len() as f64 / 1e6,
                    stacked_down_buf.byte_len() as f64 / 1e6);
            }

            // -- Router and scales (F32) --
            let router_proj = load_gguf_qweight(
                gguf, &format!("blk.{i}.ffn_gate_inp.weight"), mlx_device,
            )?;
            let router_scale = gguf.load_tensor_f32(
                &format!("blk.{i}.ffn_gate_inp.scale"), mlx_device,
            ).map_err(|e| anyhow::anyhow!("layer {i} router_scale: {e}"))?;
            let per_expert_scale = gguf.load_tensor_f32(
                &format!("blk.{i}.ffn_down_exps.scale"), mlx_device,
            ).map_err(|e| anyhow::anyhow!("layer {i} per_expert_scale: {e}"))?;

            // Pre-compute router combined weight:
            //   router_combined_weight[j] = router_scale[j] * (hidden_size ^ -0.5)
            let router_combined_weight = {
                let scale_factor = (cfg.hidden_size as f32).powf(-0.5);
                let rs: &[f32] = router_scale.as_slice()
                    .map_err(|e| anyhow::anyhow!("router_scale read for combined weight: {e}"))?;
                let mut combined = mlx_device.alloc_buffer(
                    cfg.hidden_size * std::mem::size_of::<f32>(),
                    mlx_native::DType::F32,
                    vec![cfg.hidden_size],
                ).map_err(|e| anyhow::anyhow!("router_combined_weight alloc: {e}"))?;
                let dst: &mut [f32] = combined.as_mut_slice()
                    .map_err(|e| anyhow::anyhow!("router_combined_weight write: {e}"))?;
                for j in 0..cfg.hidden_size {
                    dst[j] = rs[j] * scale_factor;
                }
                combined
            };

            let moe = MlxMoeWeights {
                stacked_gate_up: Some(stacked_gate_up_buf),
                stacked_down: Some(stacked_down_buf),
                gate_up_expert_stride: gate_up_expert_stride as u64,
                down_expert_stride: down_expert_stride as u64,
                router_proj,
                per_expert_scale,
                gate_up_ggml_dtype,
                down_ggml_dtype,
                top_k: cfg.top_k_experts,
                moe_intermediate_size: cfg.moe_intermediate_size,
                router_combined_weight,
            };

            // -- Norm weights (all F32) --
            let norms = MlxLayerNorms {
                input_layernorm: gguf.load_tensor_f32(
                    &format!("blk.{i}.attn_norm.weight"), mlx_device,
                ).map_err(|e| anyhow::anyhow!("layer {i} attn_norm: {e}"))?,
                post_attention_layernorm: gguf.load_tensor_f32(
                    &format!("blk.{i}.post_attention_norm.weight"), mlx_device,
                ).map_err(|e| anyhow::anyhow!("layer {i} post_attn_norm: {e}"))?,
                pre_feedforward_layernorm: gguf.load_tensor_f32(
                    &format!("blk.{i}.ffn_norm.weight"), mlx_device,
                ).map_err(|e| anyhow::anyhow!("layer {i} ffn_norm: {e}"))?,
                post_feedforward_layernorm: gguf.load_tensor_f32(
                    &format!("blk.{i}.post_ffw_norm.weight"), mlx_device,
                ).map_err(|e| anyhow::anyhow!("layer {i} post_ffw_norm: {e}"))?,
                pre_feedforward_layernorm_2: gguf.load_tensor_f32(
                    &format!("blk.{i}.pre_ffw_norm_2.weight"), mlx_device,
                ).map_err(|e| anyhow::anyhow!("layer {i} pre_ffw_norm_2: {e}"))?,
                post_feedforward_layernorm_1: gguf.load_tensor_f32(
                    &format!("blk.{i}.post_ffw_norm_1.weight"), mlx_device,
                ).map_err(|e| anyhow::anyhow!("layer {i} post_ffw_norm_1: {e}"))?,
                post_feedforward_layernorm_2: gguf.load_tensor_f32(
                    &format!("blk.{i}.post_ffw_norm_2.weight"), mlx_device,
                ).map_err(|e| anyhow::anyhow!("layer {i} post_ffw_norm_2: {e}"))?,
            };

            // -- Layer scalar (F32) --
            let layer_scalar = gguf.load_tensor_f32(
                &format!("blk.{i}.layer_output_scale.weight"), mlx_device,
            ).map_err(|e| anyhow::anyhow!("layer {i} layer_scalar: {e}"))?;

            // -- Per-layer config --
            let hd = cfg.head_dim_for_layer(i);
            let nkv = cfg.num_kv_heads_for_layer(i);
            let is_full = cfg.is_full_attention(i);
            let layer_type = if is_full { LayerType::Full } else { LayerType::Sliding };

            // -- KV cache allocation (identical to the old load_from_candle) --
            let capacity = if is_full {
                cfg.max_position_embeddings
            } else {
                cfg.sliding_window
            };
            // TurboQuant 4-bit nibble-packed indices + F32 norms (ADR-007 Phase 1.2).
            let packed_bytes = nkv * capacity * (hd / 2);
            let norms_bytes = nkv * capacity * 4;

            let k_packed = mlx_device.alloc_buffer(
                packed_bytes, mlx_native::DType::U8, vec![nkv, capacity, hd / 2],
            ).map_err(|e| anyhow::anyhow!("KV cache K packed alloc: {e}"))?;
            let k_norms = mlx_device.alloc_buffer(
                norms_bytes, mlx_native::DType::F32, vec![nkv, capacity],
            ).map_err(|e| anyhow::anyhow!("KV cache K norms alloc: {e}"))?;
            let v_packed = mlx_device.alloc_buffer(
                packed_bytes, mlx_native::DType::U8, vec![nkv, capacity, hd / 2],
            ).map_err(|e| anyhow::anyhow!("KV cache V packed alloc: {e}"))?;
            let v_norms = mlx_device.alloc_buffer(
                norms_bytes, mlx_native::DType::F32, vec![nkv, capacity],
            ).map_err(|e| anyhow::anyhow!("KV cache V norms alloc: {e}"))?;

            kv_caches.push(MlxKvCache {
                k_packed,
                k_norms,
                v_packed,
                v_norms,
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
                head_dim: hd,
                num_kv_heads: nkv,
                layer_type,
            });
            progress.on_layer(i + 1);
        }
        progress.finish();
        tracing::info!(
            "Loaded {}/{} mlx-native layer weights from GGUF (including MoE)",
            num_layers, num_layers
        );

        // -- Allocate activation buffers --
        let mut activations = alloc_activation_buffers(mlx_device, cfg)?;

        // -- RoPE freq_factors from GGUF --
        if let Some(_info) = gguf.tensor_info("rope_freqs.weight") {
            let ff_buf = gguf.load_tensor_f32("rope_freqs.weight", mlx_device)
                .map_err(|e| anyhow::anyhow!("rope_freqs: {e}"))?;
            activations.rope_freq_factors_gpu = ff_buf;
        }

        // -- Build result --
        let mut result = Ok(Self {
            embed_weight,
            layers,
            final_norm,
            lm_head_f16,
            lm_head_q8,
            hidden_size: cfg.hidden_size,
            vocab_size: cfg.vocab_size,
            num_attention_heads: cfg.num_attention_heads,
            rms_norm_eps: cfg.rms_norm_eps as f32,
            final_logit_softcapping: cfg.final_logit_softcapping.map(|v| v as f32),
            kv_caches,
            activations,
            sliding_window: cfg.sliding_window,
            rope_theta_sliding: cfg.rope_theta_sliding as f32,
            rope_theta_global: cfg.rope_theta_global as f32,
            num_experts: cfg.num_experts,
            intermediate_size: cfg.intermediate_size,
            dense_kvs: None,
            dense_sdpa_tmp: None,
        });

        // Pre-initialize constant param buffers so we never write them
        // inside the hot forward_decode path.
        if let Ok(ref mut w) = result {
            // Softcap params: [cap, n_elements_as_f32_bits]
            if let Some(cap) = w.final_logit_softcapping {
                let p: &mut [f32] = w.activations.softcap_params.as_mut_slice()
                    .map_err(|e| anyhow::anyhow!("softcap_params init: {e}"))?;
                p[0] = cap;
                p[1] = f32::from_bits(w.vocab_size as u32);
            }
            // Argmax params: [vocab_size]
            {
                let p: &mut [u32] = w.activations.argmax_params.as_mut_slice()
                    .map_err(|e| anyhow::anyhow!("argmax_params init: {e}"))?;
                p[0] = w.vocab_size as u32;
            }
        }

        result
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
        let hs = self.hidden_size;
        let num_layers = self.layers.len();
        let vocab_size = self.vocab_size;

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

        // ADR-009 Phase 3A: boundary dump at specific token position.
        // Temporary diagnostic — to be merged into parity capture workflow.
        let dump_pos: Option<usize> = INVESTIGATION_ENV.dump_boundary;
        let dump_layers: bool = INVESTIGATION_ENV.dump_layers == Some(seq_pos);

        // --- Pre-session CPU work ---
        // Write position buffer (same for all layers)
        {
            let pos_dst: &mut [u32] = self.activations.position.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("position write: {e}"))?;
            pos_dst[0] = seq_pos as u32;
        }

        // KV cache bookkeeping for all layers (CPU counters only, no GPU buffers)
        let mut kv_info: Vec<(bool, usize, usize, usize)> = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let is_sliding = self.kv_caches[layer_idx].is_sliding;
            let write_pos = self.kv_caches[layer_idx].write_pos;
            let capacity = self.kv_caches[layer_idx].capacity;
            self.kv_caches[layer_idx].write_pos += 1;
            self.kv_caches[layer_idx].seq_len = self.kv_caches[layer_idx].seq_len.saturating_add(1)
                .min(capacity);
            let seq_len = self.kv_caches[layer_idx].seq_len;
            kv_info.push((is_sliding, write_pos, capacity, seq_len));
        }

        // =====================================================================
        // SINGLE SESSION: Embedding + All 30 Layers + Head
        //
        // ONE begin() → all GPU dispatches → ONE finish().
        // Zero CPU readbacks.  All norms, adds, MoE routing, scalar multiplies,
        // softcap, and argmax run on GPU.
        // =====================================================================
        let session_start = Instant::now();
        let mut total_dispatches = 0usize;
        {
            let (exec, reg) = gpu.split();
            let dev = exec.device();
            let metal_dev = dev.metal_device();
            let use_graph_opt = INVESTIGATION_ENV.graph_opt;
            let mut s = if use_graph_opt {
                exec.begin_recorded().map_err(|e| anyhow::anyhow!("recorded session begin: {e}"))?
            } else {
                exec.begin().map_err(|e| anyhow::anyhow!("single session begin: {e}"))?
            };

            // --- 1. Embedding gather + scale (GPU) ---
            // Set pending buffer ranges for graph capture (Phase 4e.5): the
            // embedding dispatch reads embed_weight and writes hidden.
            if use_graph_opt {
                let read_ranges = vec![{
                    let s_ptr = self.embed_weight.contents_ptr() as usize;
                    (s_ptr, s_ptr + self.embed_weight.byte_len())
                }];
                let write_ranges = vec![{
                    let s_ptr = self.activations.hidden.contents_ptr() as usize;
                    (s_ptr, s_ptr + self.activations.hidden.byte_len())
                }];
                s.encoder_mut().set_pending_buffer_ranges(read_ranges, write_ranges);
            }
            mlx_native::ops::elementwise::embedding_gather_scale_f32(
                s.encoder_mut(), reg, metal_dev,
                &self.embed_weight,
                &self.activations.hidden,
                input_token,
                hs,
                (hs as f32).sqrt(),
            ).map_err(|e| anyhow::anyhow!("embedding_gather_scale: {e}"))?;
            total_dispatches += 1;
            s.track_dispatch(&[&self.embed_weight], &[&self.activations.hidden]);

            // --- Dual command buffer: split encoding at a layer boundary ---
            // Default: split after layer 3 (~10% of dispatches committed early so
            // GPU starts while CPU encodes the remaining 90%). Measured +4.4 tok/s
            // (94.3→98.7) with zero correctness impact (sourdough gate PASS).
            //
            // Override: HF2Q_DUAL_BUFFER=N (split after layer N, 0=disabled).
            let dual_buffer_split: Option<usize> =
                INVESTIGATION_ENV.dual_buffer_split(num_layers);

            // --- 2. Transformer layers ---
            // Phase 3A: sub-layer detail dump (which specific layer to break down)
            let dump_detail_layer: Option<usize> = INVESTIGATION_ENV.dump_layer_detail;

            for layer_idx in 0..num_layers {
                let hd = self.layers[layer_idx].head_dim;
                let nkv = self.layers[layer_idx].num_kv_heads;
                let nh = self.num_attention_heads;
                let is_sliding = self.layers[layer_idx].layer_type == LayerType::Sliding;
                let eps = self.rms_norm_eps;
                let (kv_is_sliding, kv_write_pos, kv_capacity, kv_seq_len) = kv_info[layer_idx];

                // -- Pre-attention norm (GPU) --
                s.barrier_between(
                    &[&self.activations.hidden, &self.layers[layer_idx].norms.input_layernorm],
                    &[&self.activations.norm_out],
                );
                s.rms_norm(
                    reg, metal_dev,
                    &self.activations.hidden,
                    &self.layers[layer_idx].norms.input_layernorm,
                    &self.activations.norm_out,
                    &self.activations.norm_params,
                    1, hs as u32,
                ).map_err(|e| anyhow::anyhow!("GPU pre-attn norm L{layer_idx}: {e}"))?;
                total_dispatches += 1;

                // -- QKV projections (CONCURRENT: all read norm_out, write separate buffers) --
                // ONE barrier after norm (which wrote norm_out), then all 3 projections
                // dispatch without barriers between them — they share reads and have disjoint writes.
                s.barrier_between(
                    &[&self.activations.norm_out],
                    &[&self.activations.attn_q, &self.activations.attn_k, &self.activations.attn_v],
                );
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    &self.layers[layer_idx].attn.q_proj, &mut self.activations.attn_q, 1)?;
                total_dispatches += 1;
                // Per-dispatch range annotation for the reorder pass. The
                // single barrier_between above only annotates the first
                // dispatch; concurrent K and V need their own ranges.
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    &self.layers[layer_idx].attn.k_proj, &mut self.activations.attn_k, 1)?;
                s.track_dispatch(&[&self.activations.norm_out], &[&self.activations.attn_k]);
                total_dispatches += 1;
                let v_is_k = self.layers[layer_idx].attn.v_proj.is_none();
                if !v_is_k {
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                        self.layers[layer_idx].attn.v_proj.as_ref().unwrap(),
                        &mut self.activations.attn_v, 1)?;
                    s.track_dispatch(&[&self.activations.norm_out], &[&self.activations.attn_v]);
                    total_dispatches += 1;
                }

                // -- Fused per-head RMS norm + RoPE on Q and K (CONCURRENT) --
                let ff_gpu = if is_sliding {
                    None
                } else {
                    Some(&self.activations.rope_freq_factors_gpu)
                };
                let theta = if is_sliding {
                    self.rope_theta_sliding
                } else {
                    self.rope_theta_global
                };
                let half_rope = (hd / 2) as u32;

                // Fused Q + K norm+RoPE (CONCURRENT: read attn_q/attn_k from QKV proj,
                // write to disjoint attn_q_normed/attn_k_normed). ONE barrier for both.
                s.barrier_between(
                    &[&self.activations.attn_q, &self.activations.attn_k],
                    &[&self.activations.attn_q_normed, &self.activations.attn_k_normed],
                );
                mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.attn_q,
                    &self.activations.attn_q_normed,
                    Some(&self.layers[layer_idx].attn.q_norm_weight),
                    &self.activations.position,
                    ff_gpu,
                    nh as u32, hd as u32, half_rope,
                    eps, theta,
                ).map_err(|e| anyhow::anyhow!("fused Q norm+RoPE L{layer_idx}: {e}"))?;
                total_dispatches += 1;
                mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.attn_k,
                    &self.activations.attn_k_normed,
                    Some(&self.layers[layer_idx].attn.k_norm_weight),
                    &self.activations.position,
                    ff_gpu,
                    nkv as u32, hd as u32, half_rope,
                    eps, theta,
                ).map_err(|e| anyhow::anyhow!("fused K norm+RoPE L{layer_idx}: {e}"))?;
                total_dispatches += 1;

                // GPU V norm
                let hd_norm_params = if is_sliding {
                    &self.activations.norm_params_sliding_hd
                } else {
                    &self.activations.norm_params_global_hd
                };
                if v_is_k {
                    s.barrier_between(
                        &[&self.activations.attn_k],
                        &[&self.activations.attn_v],
                    );
                    dispatch_rms_norm_unit_perhead(
                        s.encoder_mut(), reg, metal_dev,
                        &RmsNormPerHeadArgs {
                            input: &self.activations.attn_k,
                            output: &self.activations.attn_v,
                            params_buf: hd_norm_params,
                            rows: nkv as u32,
                            dim: hd as u32,
                        },
                    )?;
                    total_dispatches += 1;
                } else {
                    s.barrier_between(
                        &[&self.activations.attn_v],
                        &[&self.activations.moe_expert_out],
                    );
                    dispatch_rms_norm_unit_perhead(
                        s.encoder_mut(), reg, metal_dev,
                        &RmsNormPerHeadArgs {
                            input: &self.activations.attn_v,
                            output: &self.activations.moe_expert_out,
                            params_buf: hd_norm_params,
                            rows: nkv as u32,
                            dim: hd as u32,
                        },
                    )?;
                    total_dispatches += 1;
                }

                let v_src = if v_is_k {
                    &self.activations.attn_v
                } else {
                    &self.activations.moe_expert_out
                };

                // -- GPU KV cache update: Hadamard-quantize into TQ packed cache (ADR-007) --
                // HF2Q_SKIP_TQ_ENCODE=1: skip for timing bisection (output garbage).
                if !INVESTIGATION_ENV.skip_tq_encode {
                    let cache_pos_val = if kv_is_sliding {
                        (kv_write_pos % kv_capacity) as u32
                    } else {
                        kv_write_pos as u32
                    };
                    s.barrier_between(
                        &[&self.activations.attn_k_normed, v_src],
                        &[&self.kv_caches[layer_idx].k_packed, &self.kv_caches[layer_idx].k_norms,
                          &self.kv_caches[layer_idx].v_packed, &self.kv_caches[layer_idx].v_norms],
                    );
                    mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.attn_k_normed,
                        &self.kv_caches[layer_idx].k_packed,
                        &self.kv_caches[layer_idx].k_norms,
                        nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                        kv_is_sliding,
                    ).map_err(|e| anyhow::anyhow!("hadamard_quantize K L{layer_idx}: {e}"))?;
                    total_dispatches += 1;
                    mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                        s.encoder_mut(), reg, metal_dev,
                        v_src,
                        &self.kv_caches[layer_idx].v_packed,
                        &self.kv_caches[layer_idx].v_norms,
                        nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                        kv_is_sliding,
                    ).map_err(|e| anyhow::anyhow!("hadamard_quantize V L{layer_idx}: {e}"))?;
                    total_dispatches += 1;
                }

                // C-1-unlock: post-hadamard_quantize pre-SDPA dump (decode step 1, layer 0).
                // Gate: dump_tq_state && layer_idx == 0 && kv_seq_len == 23 (one decode token
                // has been written into slot 22 of the TQ ring buffer).
                // Dumps full-capacity packed K/V + norms + Q (pre-FWHT) to post_quant subdir.
                if INVESTIGATION_ENV.dump_tq_state && layer_idx == 0 && kv_seq_len == 23 {
                    s.finish()
                        .map_err(|e| anyhow::anyhow!("post_quant dump finish L{layer_idx}: {e}"))?;
                    let hd_half = hd / 2;
                    let dump_dir = &INVESTIGATION_ENV.dump_dir;
                    let post_quant_dir = format!("{dump_dir}/post_quant");
                    std::fs::create_dir_all(&post_quant_dir)
                        .map_err(|e| anyhow::anyhow!("post_quant mkdir: {e}"))?;

                    // k_packed_post_quant.u8.bin — full [nkv, kv_capacity, hd/2] u8
                    {
                        let k_raw: &[u8] = self.kv_caches[layer_idx].k_packed.as_slice()
                            .map_err(|e| anyhow::anyhow!("post_quant k_packed read: {e}"))?;
                        let n_bytes = nkv * kv_capacity * hd_half;
                        let kp = format!("{post_quant_dir}/k_packed_post_quant.u8.bin");
                        std::fs::write(&kp, &k_raw[..n_bytes])
                            .map_err(|e| anyhow::anyhow!("write {kp}: {e}"))?;
                        eprintln!("[POST_QUANT_DUMP] k_packed [{nkv},{kv_capacity},{hd_half}] u8 -> {kp}");
                    }

                    // v_packed_post_quant.u8.bin — full [nkv, kv_capacity, hd/2] u8
                    {
                        let v_raw: &[u8] = self.kv_caches[layer_idx].v_packed.as_slice()
                            .map_err(|e| anyhow::anyhow!("post_quant v_packed read: {e}"))?;
                        let n_bytes = nkv * kv_capacity * hd_half;
                        let vp = format!("{post_quant_dir}/v_packed_post_quant.u8.bin");
                        std::fs::write(&vp, &v_raw[..n_bytes])
                            .map_err(|e| anyhow::anyhow!("write {vp}: {e}"))?;
                        eprintln!("[POST_QUANT_DUMP] v_packed [{nkv},{kv_capacity},{hd_half}] u8 -> {vp}");
                    }

                    // k_norms_post_quant.f32.bin — full [nkv, kv_capacity] f32
                    {
                        let kn_raw: &[f32] = self.kv_caches[layer_idx].k_norms.as_slice()
                            .map_err(|e| anyhow::anyhow!("post_quant k_norms read: {e}"))?;
                        let n_elems = nkv * kv_capacity;
                        let kn_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                kn_raw.as_ptr() as *const u8,
                                n_elems * std::mem::size_of::<f32>(),
                            )
                        };
                        let kn = format!("{post_quant_dir}/k_norms_post_quant.f32.bin");
                        std::fs::write(&kn, kn_bytes)
                            .map_err(|e| anyhow::anyhow!("write {kn}: {e}"))?;
                        eprintln!("[POST_QUANT_DUMP] k_norms [{nkv},{kv_capacity}] f32 -> {kn}");
                    }

                    // v_norms_post_quant.f32.bin — full [nkv, kv_capacity] f32
                    {
                        let vn_raw: &[f32] = self.kv_caches[layer_idx].v_norms.as_slice()
                            .map_err(|e| anyhow::anyhow!("post_quant v_norms read: {e}"))?;
                        let n_elems = nkv * kv_capacity;
                        let vn_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                vn_raw.as_ptr() as *const u8,
                                n_elems * std::mem::size_of::<f32>(),
                            )
                        };
                        let vn = format!("{post_quant_dir}/v_norms_post_quant.f32.bin");
                        std::fs::write(&vn, vn_bytes)
                            .map_err(|e| anyhow::anyhow!("write {vn}: {e}"))?;
                        eprintln!("[POST_QUANT_DUMP] v_norms [{nkv},{kv_capacity}] f32 -> {vn}");
                    }

                    // q_natural.f32.bin — Q pre-FWHT, shape [nh, hd] f32
                    {
                        let q_raw: &[f32] = self.activations.attn_q_normed.as_slice()
                            .map_err(|e| anyhow::anyhow!("post_quant q_normed read: {e}"))?;
                        let n_elems = nh * hd;
                        let q_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                q_raw.as_ptr() as *const u8,
                                n_elems * std::mem::size_of::<f32>(),
                            )
                        };
                        let qp = format!("{post_quant_dir}/q_natural.f32.bin");
                        std::fs::write(&qp, q_bytes)
                            .map_err(|e| anyhow::anyhow!("write {qp}: {e}"))?;
                        eprintln!("[POST_QUANT_DUMP] q_natural [{nh},{hd}] f32 -> {qp}");
                    }

                    // meta_post_quant.json — production call-site params + provenance
                    {
                        let ring_start = if kv_is_sliding && kv_seq_len >= kv_capacity {
                            (kv_write_pos % kv_capacity) as u32
                        } else {
                            0u32
                        };
                        let commit_sha = option_env!("GIT_COMMIT_SHA")
                            .unwrap_or("03bea75071a8b0fd43a47f1101a832e23317e429");
                        let meta = serde_json::json!({
                            "site": "post_hadamard_quantize_pre_sdpa",
                            "layer_idx": layer_idx,
                            "seq_pos": seq_pos,
                            "kv_seq_len": kv_seq_len,
                            "kv_capacity": kv_capacity,
                            "kv_write_pos": kv_write_pos,
                            "nkv": nkv,
                            "nh": nh,
                            "hd": hd,
                            "hd_half": hd_half,
                            "kv_is_sliding": kv_is_sliding,
                            "mask_type": if is_sliding { 2u32 } else { 1u32 },
                            "sliding_window": if is_sliding { self.sliding_window as u32 } else { 0u32 },
                            "ring_start": ring_start,
                            "k_packed_shape": [nkv, kv_capacity, hd_half],
                            "v_packed_shape": [nkv, kv_capacity, hd_half],
                            "k_norms_shape": [nkv, kv_capacity],
                            "v_norms_shape": [nkv, kv_capacity],
                            "q_natural_shape": [nh, hd],
                            "commit_sha": commit_sha,
                        });
                        let meta_str = serde_json::to_string_pretty(&meta)
                            .map_err(|e| anyhow::anyhow!("post_quant meta json: {e}"))?;
                        let mp = format!("{post_quant_dir}/meta_post_quant.json");
                        std::fs::write(&mp, meta_str.as_bytes())
                            .map_err(|e| anyhow::anyhow!("write {mp}: {e}"))?;
                        eprintln!("[POST_QUANT_DUMP] meta -> {mp}");
                    }

                    s = exec.begin()
                        .map_err(|e| anyhow::anyhow!("post_quant dump re-begin: {e}"))?;
                }

                // ADR-009 Phase 3A: dump Q,K,V before SDPA for the detail layer,
                // or ALL layers when HF2Q_DUMP_ALL_CACHE=1
                let dump_all_cache = INVESTIGATION_ENV.dump_all_cache;
                if dump_layers && (dump_detail_layer == Some(layer_idx) || dump_all_cache) {
                    s.finish()
                        .map_err(|e| anyhow::anyhow!("dump QKV finish L{layer_idx}: {e}"))?;
                    dumps::dump_f32(&self.activations.attn_q_normed, nh * hd,
                        "q_normed", Some(layer_idx), seq_pos)?;
                    dumps::dump_f32(&self.activations.attn_k_normed, nkv * hd,
                        "k_normed", Some(layer_idx), seq_pos)?;
                    dumps::dump_f32(v_src, nkv * hd,
                        "v_normed", Some(layer_idx), seq_pos)?;
                    s = exec.begin()
                        .map_err(|e| anyhow::anyhow!("dump QKV re-begin L{layer_idx}: {e}"))?;
                }

                // -- SDPA: dense or TQ-packed (ADR-009 Track 3) --
                // When dense_kvs is available (set by forward_prefill), use
                // flash_attn_vec with F32 K,V for reference-parity attention.
                // Otherwise fall back to the original TQ SDPA path.
                let use_dense_sdpa = self.dense_kvs.is_some();

                if use_dense_sdpa {
                    // -- Dense decode SDPA (ADR-009 Track 3) --
                    // Copy this position's K,V into dense KV buffers.
                    // Uses F16 cast kernel when dense_kvs are F16, else F32 copy.
                    let dense_kvs = self.dense_kvs.as_ref().unwrap();
                    let dense_cap = dense_kvs[layer_idx].capacity;
                    let layer_is_ring = dense_kvs[layer_idx].is_sliding;
                    // Ring-buffer write for sliding layers; linear for global.
                    let write_slot = if layer_is_ring {
                        (seq_pos % dense_cap) as u32
                    } else {
                        seq_pos as u32
                    };
                    let kv_is_f16 = dense_kvs[layer_idx].k.dtype() == mlx_native::DType::F16;
                    s.barrier_between(
                        &[&self.activations.attn_k_normed, v_src],
                        &[&dense_kvs[layer_idx].k, &dense_kvs[layer_idx].v],
                    );
                    if kv_is_f16 {
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16(
                            s.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed,
                            &dense_kvs[layer_idx].k,
                            nkv as u32, hd as u32,
                            dense_cap as u32, write_slot,
                        ).map_err(|e| anyhow::anyhow!("decode F16 K copy L{layer_idx}: {e}"))?;
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16(
                            s.encoder_mut(), reg, metal_dev,
                            v_src,
                            &dense_kvs[layer_idx].v,
                            nkv as u32, hd as u32,
                            dense_cap as u32, write_slot,
                        ).map_err(|e| anyhow::anyhow!("decode F16 V copy L{layer_idx}: {e}"))?;
                        total_dispatches += 2;
                    } else {
                        // F32 batched: one dispatch per K, one per V (all heads at once).
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32(
                            s.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed,
                            &dense_kvs[layer_idx].k,
                            nkv as u32, hd as u32,
                            dense_cap as u32, write_slot,
                        ).map_err(|e| anyhow::anyhow!("decode F32 K batch copy L{layer_idx}: {e}"))?;
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32(
                            s.encoder_mut(), reg, metal_dev,
                            v_src,
                            &dense_kvs[layer_idx].v,
                            nkv as u32, hd as u32,
                            dense_cap as u32, write_slot,
                        ).map_err(|e| anyhow::anyhow!("decode F32 V batch copy L{layer_idx}: {e}"))?;
                        total_dispatches += 2;
                    }

                    // ADR-009 Phase 3A: dump full cached K/V for the detail layer,
                    // or ALL layers when HF2Q_DUMP_ALL_CACHE=1
                    let dump_all_cache = INVESTIGATION_ENV.dump_all_cache;
                    if dump_layers && (dump_detail_layer == Some(layer_idx) || dump_all_cache) {
                        s.finish()
                            .map_err(|e| anyhow::anyhow!("dump cache finish L{layer_idx}: {e}"))?;
                        let dump_dir = &INVESTIGATION_ENV.dump_dir;
                        // Pack [nkv, kv_seq_len, hd] into a tight F32 buffer for comparison
                        let valid_len = kv_seq_len;
                        let mut k_valid = vec![0.0f32; nkv * valid_len * hd];
                        let mut v_valid = vec![0.0f32; nkv * valid_len * hd];
                        if kv_is_f16 {
                            // Read F16 bits and convert to F32
                            let k_raw: &[u16] = dense_kvs[layer_idx].k.as_slice()
                                .map_err(|e| anyhow::anyhow!("dump cache K L{layer_idx}: {e}"))?;
                            let v_raw: &[u16] = dense_kvs[layer_idx].v.as_slice()
                                .map_err(|e| anyhow::anyhow!("dump cache V L{layer_idx}: {e}"))?;
                            for h in 0..nkv {
                                for p in 0..valid_len {
                                    let src = h * dense_cap * hd + p * hd;
                                    let dst = h * valid_len * hd + p * hd;
                                    for i in 0..hd {
                                        k_valid[dst+i] = half::f16::from_bits(k_raw[src+i]).to_f32();
                                        v_valid[dst+i] = half::f16::from_bits(v_raw[src+i]).to_f32();
                                    }
                                }
                            }
                        } else {
                            let k_data: &[f32] = dense_kvs[layer_idx].k.as_slice()
                                .map_err(|e| anyhow::anyhow!("dump cache K L{layer_idx}: {e}"))?;
                            let v_data: &[f32] = dense_kvs[layer_idx].v.as_slice()
                                .map_err(|e| anyhow::anyhow!("dump cache V L{layer_idx}: {e}"))?;
                            for h in 0..nkv {
                                for p in 0..valid_len {
                                    let src = h * dense_cap * hd + p * hd;
                                    let dst = h * valid_len * hd + p * hd;
                                    k_valid[dst..dst+hd].copy_from_slice(&k_data[src..src+hd]);
                                    v_valid[dst..dst+hd].copy_from_slice(&v_data[src..src+hd]);
                                }
                            }
                        }
                        let k_path = format!("{dump_dir}/hf2q_cache_k_layer{layer_idx:02}_pos{seq_pos}.bin");
                        let v_path = format!("{dump_dir}/hf2q_cache_v_layer{layer_idx:02}_pos{seq_pos}.bin");
                        let k_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(k_valid.as_ptr() as *const u8, k_valid.len() * 4)
                        };
                        let v_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(v_valid.as_ptr() as *const u8, v_valid.len() * 4)
                        };
                        std::fs::write(&k_path, k_bytes)
                            .map_err(|e| anyhow::anyhow!("write {k_path}: {e}"))?;
                        std::fs::write(&v_path, v_bytes)
                            .map_err(|e| anyhow::anyhow!("write {v_path}: {e}"))?;
                        let dtype_str = if kv_is_f16 { "F16→F32" } else { "F32" };
                        eprintln!("[DUMP] cache K layer {layer_idx:02} [{nkv},{valid_len},{hd}] {dtype_str} -> {k_path}");
                        eprintln!("[DUMP] cache V layer {layer_idx:02} [{nkv},{valid_len},{hd}] {dtype_str} -> {v_path}");
                        s = exec.begin()
                            .map_err(|e| anyhow::anyhow!("dump cache re-begin L{layer_idx}: {e}"))?;
                    }

                    // Dense flash_attn_vec
                    let dense_sdpa_tmp = self.dense_sdpa_tmp.as_ref().unwrap();
                    s.barrier_between(
                        &[&self.activations.attn_q_normed,
                          &dense_kvs[layer_idx].k, &dense_kvs[layer_idx].v],
                        &[&self.activations.sdpa_out],
                    );
                    // kv_seq_len for the dense cache:
                    //   - Sliding (ring): min(seq_pos+1, capacity). The ring holds
                    //     at most `capacity=sliding_window` entries — the causal
                    //     mask then attends to exactly the populated slots.
                    //     Attention is permutation-invariant over cached K,V
                    //     (RoPE is baked in pre-cache), so slot order doesn't
                    //     matter for correctness.
                    //   - Global (linear): seq_pos + 1.
                    // In ring mode we use mask_type=1 (causal) since the ring
                    // itself applies the sliding-window constraint — the
                    // kernel's sliding-window mask would incorrectly mask slots
                    // whose logical positions don't equal their slot index.
                    let dense_kv_seq_len = if layer_is_ring {
                        ((seq_pos + 1).min(dense_cap)) as u32
                    } else {
                        (seq_pos + 1) as u32
                    };
                    let p = mlx_native::ops::flash_attn_vec::FlashAttnVecParams {
                        num_heads: nh as u32,
                        num_kv_heads: nkv as u32,
                        head_dim: hd as u32,
                        kv_seq_len: dense_kv_seq_len,
                        kv_capacity: dense_cap as u32,
                        scale: 1.0,
                        mask_type: 1, // causal; ring applies the sliding window for us
                        sliding_window: 0,
                        softcap: 0.0,
                    };
                    mlx_native::ops::flash_attn_vec::flash_attn_vec(
                        s.encoder_mut(), reg, dev,
                        &self.activations.attn_q_normed,
                        &dense_kvs[layer_idx].k,
                        &dense_kvs[layer_idx].v,
                        &self.activations.sdpa_out,
                        dense_sdpa_tmp,
                        &p,
                    ).map_err(|e| anyhow::anyhow!("dense flash_attn_vec L{layer_idx}: {e}"))?;
                    total_dispatches += 2; // main + reduce
                } else if !INVESTIGATION_ENV.skip_tq_sdpa {
                    // -- TQ-packed SDPA (original path) --
                    // Pre-rotate Q via standalone FWHT (1× per head).
                    s.barrier_between(
                        &[&self.activations.attn_q_normed],
                        &[&self.activations.attn_q_normed],
                    );
                    mlx_native::ops::fwht_standalone::dispatch_fwht_f32(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.attn_q_normed,
                        nh as u32, hd as u32,
                    ).map_err(|e| anyhow::anyhow!("FWHT Q pre-rotate L{layer_idx}: {e}"))?;
                    total_dispatches += 1;

                    // TQ SDPA (pre-rotated Q → rotated-domain output)
                    s.barrier_between(
                        &[&self.activations.attn_q_normed,
                          &self.kv_caches[layer_idx].k_packed, &self.kv_caches[layer_idx].k_norms,
                          &self.kv_caches[layer_idx].v_packed, &self.kv_caches[layer_idx].v_norms],
                        &[&self.activations.sdpa_out],
                    );
                    let ring_start = if kv_is_sliding && kv_seq_len >= kv_capacity {
                        (kv_write_pos % kv_capacity) as u32
                    } else {
                        0
                    };
                    let p = FlashAttnVecTqParams {
                        num_heads: nh as u32,
                        num_kv_heads: nkv as u32,
                        head_dim: hd as u32,
                        kv_seq_len: kv_seq_len as u32,
                        kv_capacity: kv_capacity as u32,
                        scale: 1.0,
                        mask_type: if is_sliding { 2 } else { 1 },
                        sliding_window: if is_sliding { self.sliding_window as u32 } else { 0 },
                        softcap: 0.0,
                        ring_start,
                    };
                    mlx_native::ops::flash_attn_vec_tq::flash_attn_vec_tq(
                        s.encoder_mut(), reg, dev,
                        &self.activations.attn_q_normed,
                        &self.kv_caches[layer_idx].k_packed,
                        &self.kv_caches[layer_idx].k_norms,
                        &self.kv_caches[layer_idx].v_packed,
                        &self.kv_caches[layer_idx].v_norms,
                        &self.activations.sdpa_out,
                        &self.activations.sdpa_tmp,
                        &p,
                    ).map_err(|e| anyhow::anyhow!("flash_attn_vec_tq L{layer_idx}: {e}"))?;

                    // Inverse-rotate SDPA output (1× per head).
                    s.barrier_between(
                        &[&self.activations.sdpa_out],
                        &[&self.activations.sdpa_out],
                    );
                    mlx_native::ops::fwht_standalone::dispatch_fwht_f32(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.sdpa_out,
                        nh as u32, hd as u32,
                    ).map_err(|e| anyhow::anyhow!("FWHT inv-rotate L{layer_idx}: {e}"))?;
                    total_dispatches += 1;
                    total_dispatches += 2; // main + reduce
                }

                // ADR-009 Phase 3A: dump sdpa_out before O-proj for the detail layer,
                // or ALL layers when HF2Q_DUMP_ALL_CACHE=1
                if dump_layers && (dump_detail_layer == Some(layer_idx) || dump_all_cache) {
                    s.finish()
                        .map_err(|e| anyhow::anyhow!("dump sdpa_out finish L{layer_idx}: {e}"))?;
                    // [nh, 1, hd] flattened.
                    dumps::dump_f32(&self.activations.sdpa_out, nh * hd,
                        "sdpa_out", Some(layer_idx), seq_pos)?;
                    s = exec.begin()
                        .map_err(|e| anyhow::anyhow!("dump sdpa_out re-begin L{layer_idx}: {e}"))?;
                }

                // -- O-proj --
                s.barrier_between(
                    &[&self.activations.sdpa_out, &self.layers[layer_idx].attn.o_proj.buffer],
                    &[&self.activations.attn_out],
                );
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.sdpa_out,
                    &self.layers[layer_idx].attn.o_proj, &mut self.activations.attn_out, 1)?;
                total_dispatches += 1;

                // -- Fused post-attention norm + residual add --
                s.barrier_between(
                    &[&self.activations.hidden, &self.activations.attn_out],
                    &[&self.activations.residual],
                );
                mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.hidden,
                    &self.activations.attn_out,
                    &self.layers[layer_idx].norms.post_attention_layernorm,
                    &self.activations.residual,
                    hs as u32, 1, eps,
                ).map_err(|e| anyhow::anyhow!("fused post-attn norm+add L{layer_idx}: {e}"))?;

                // (dump_detail_layer already declared above for sdpa_out dump)
                if dump_layers && dump_detail_layer == Some(layer_idx) {
                    s.finish()
                        .map_err(|e| anyhow::anyhow!("dump post-attn finish L{layer_idx}: {e}"))?;
                    // Post-attention residual (= attn_out after norm+add).
                    dumps::dump_f32(&self.activations.residual, hs,
                        "attn_out", Some(layer_idx), seq_pos)?;
                    s = exec.begin()
                        .map_err(|e| anyhow::anyhow!("dump post-attn re-begin L{layer_idx}: {e}"))?;
                }
                total_dispatches += 1;

                // ============================================================
                // Dense MLP + MoE routing INTERLEAVED dispatch
                // (ADR-006 Phase 4e: matches llama.cpp's graph reorder pattern)
                //
                // Group B8:  pre-FF norm1 + pre-FF norm2 + router norm  [3 concurrent]
                // Group B9:  dense gate + dense up + router logits      [3 concurrent]
                // Group B10: fused_gelu_mul + fused_moe_routing          [2 concurrent]
                // Group B11: dense down + gate_up_id                     [2 concurrent]
                //   ... then sequential MoE chain + post-processing
                // ============================================================

                let num_experts = self.num_experts;
                let top_k = self.layers[layer_idx].moe.top_k;

                // -- B8: pre-FF norm1 + pre-FF norm2 + router norm [3 CONCURRENT] --
                // All three read `residual` (written by post-attn norm+add), write disjoint buffers.
                // ONE barrier, then all three dispatch without barriers between them.
                s.barrier_between(
                    &[&self.activations.residual],
                    &[&self.activations.norm_out, &self.activations.moe_norm_out,
                      &self.activations.router_norm_out],
                );
                s.rms_norm(
                    reg, metal_dev,
                    &self.activations.residual,
                    &self.layers[layer_idx].norms.pre_feedforward_layernorm,
                    &self.activations.norm_out,
                    &self.activations.norm_params,
                    1, hs as u32,
                ).map_err(|e| anyhow::anyhow!("pre-FF norm L{layer_idx}: {e}"))?;
                total_dispatches += 1;

                s.rms_norm(
                    reg, metal_dev,
                    &self.activations.residual,
                    &self.layers[layer_idx].norms.pre_feedforward_layernorm_2,
                    &self.activations.moe_norm_out,
                    &self.activations.norm_params,
                    1, hs as u32,
                ).map_err(|e| anyhow::anyhow!("pre-FF norm 2 L{layer_idx}: {e}"))?;
                total_dispatches += 1;

                s.rms_norm(
                    reg, metal_dev,
                    &self.activations.residual,
                    &self.layers[layer_idx].moe.router_combined_weight,
                    &self.activations.router_norm_out,
                    &self.activations.norm_params,
                    1, hs as u32,
                ).map_err(|e| anyhow::anyhow!("router norm L{layer_idx}: {e}"))?;
                total_dispatches += 1;

                // -- B9: dense gate + dense up + router logits [3 CONCURRENT] --
                // gate/up read norm_out (from B8 norm1); router reads router_norm_out (from B8 router norm).
                // All write disjoint buffers. ONE barrier after B8, then 3 dispatches without barriers.
                s.barrier_between(
                    &[&self.activations.norm_out, &self.activations.router_norm_out],
                    &[&self.activations.mlp_gate, &self.activations.mlp_up,
                      &self.activations.moe_router_logits],
                );
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    &self.layers[layer_idx].mlp.gate_proj, &mut self.activations.mlp_gate, 1)?;
                total_dispatches += 1;
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    &self.layers[layer_idx].mlp.up_proj, &mut self.activations.mlp_up, 1)?;
                total_dispatches += 1;
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.router_norm_out,
                    &self.layers[layer_idx].moe.router_proj,
                    &mut self.activations.moe_router_logits, 1)?;
                total_dispatches += 1;

                // -- B10: fused_gelu_mul + fused_moe_routing [2 CONCURRENT] --
                // gelu_mul reads mlp_gate+mlp_up (from B9 gate/up), writes mlp_fused.
                // moe_routing reads moe_router_logits (from B9 router), writes expert_ids+weights.
                // Disjoint reads and writes — ONE barrier after B9, then both dispatch.
                s.barrier_between(
                    &[&self.activations.mlp_gate, &self.activations.mlp_up,
                      &self.activations.moe_router_logits],
                    &[&self.activations.mlp_fused,
                      &self.activations.moe_expert_ids, &self.activations.moe_routing_weights_gpu],
                );
                {
                    use mlx_native::ops::encode_helpers::{encode_with_args, KernelArg};
                    let n_elements_bytes = (self.intermediate_size as u32).to_ne_bytes();
                    let pipeline = reg.get_pipeline("fused_gelu_mul", metal_dev)?;
                    encode_with_args(
                        s.encoder_mut(), pipeline,
                        &[
                            (0, KernelArg::Buffer(&self.activations.mlp_gate)),
                            (1, KernelArg::Buffer(&self.activations.mlp_up)),
                            (2, KernelArg::Buffer(&self.activations.mlp_fused)),
                            (3, KernelArg::Bytes(&n_elements_bytes)),
                        ],
                        mlx_native::MTLSize::new(self.intermediate_size as u64, 1, 1),
                        mlx_native::MTLSize::new(
                            std::cmp::min(256, self.intermediate_size as u64), 1, 1),
                    );
                }
                total_dispatches += 1;
                mlx_native::ops::fused_norm_add::dispatch_fused_moe_routing_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.moe_router_logits,
                    &self.activations.moe_expert_ids,
                    &self.activations.moe_routing_weights_gpu,
                    &self.layers[layer_idx].moe.per_expert_scale,
                    num_experts as u32, top_k as u32,
                ).map_err(|e| anyhow::anyhow!("fused MoE routing L{layer_idx}: {e}"))?;
                total_dispatches += 1;

                // ============================================================
                // MoE expert dispatches (was S4, now in same session)
                // ============================================================
                let moe_int = self.layers[layer_idx].moe.moe_intermediate_size;
                let use_fused_id = self.layers[layer_idx].moe.stacked_gate_up.is_some()
                    && self.layers[layer_idx].moe.stacked_down.is_some();

                if use_fused_id {
                    let _ggml_type_gu = self.layers[layer_idx].moe.gate_up_ggml_dtype;
                    let _ggml_type_dn = self.layers[layer_idx].moe.down_ggml_dtype;

                    // -- B11: dense down + gate_up_id [2 concurrent] --
                    // dense_down reads mlp_fused (from B10), gate_up_id reads moe_norm_out
                    // (from B8) + moe_expert_ids (from B10). Disjoint writes.
                    s.barrier_between(
                        &[&self.activations.mlp_fused, &self.layers[layer_idx].mlp.down_proj.buffer],
                        &[&self.activations.mlp_down],
                    );
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.mlp_fused,
                        &self.layers[layer_idx].mlp.down_proj, &mut self.activations.mlp_down, 1)?;
                    total_dispatches += 1;

                    let ggml_type_gu = self.layers[layer_idx].moe.gate_up_ggml_dtype;
                    s.barrier_between(
                        &[&self.activations.moe_norm_out, &self.activations.moe_expert_ids,
                          self.layers[layer_idx].moe.stacked_gate_up.as_ref().unwrap()],
                        &[&self.activations.moe_gate_up_id_out],
                    );
                    let gu_params = mlx_native::GgmlQuantizedMatmulIdParams {
                        n_tokens: 1,
                        top_k: top_k as u32,
                        n: (2 * moe_int) as u32,
                        k: hs as u32,
                        n_experts: num_experts as u32,
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
                    ).map_err(|e| anyhow::anyhow!("gate_up _id L{layer_idx}: {e}"))?;
                    total_dispatches += 1;

                    // -- B12: swiglu (singleton) --
                    s.barrier_between(
                        &[&self.activations.moe_gate_up_id_out],
                        &[&self.activations.moe_swiglu_id_out],
                    );
                    mlx_native::ops::moe_dispatch::moe_swiglu_batch_encode(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.moe_gate_up_id_out,
                        &self.activations.moe_swiglu_id_out,
                        moe_int, top_k,
                    ).map_err(|e| anyhow::anyhow!("swiglu batch L{layer_idx}: {e}"))?;
                    total_dispatches += 1;

                    // -- B13: down_id + post-FF norm1 [2 concurrent] --
                    // down_id reads moe_swiglu_id_out (from B12). post-FF norm1 reads
                    // mlp_down (from B11). Disjoint writes.
                    let ggml_type_dn = self.layers[layer_idx].moe.down_ggml_dtype;
                    s.barrier_between(
                        &[&self.activations.moe_swiglu_id_out, &self.activations.moe_expert_ids,
                          self.layers[layer_idx].moe.stacked_down.as_ref().unwrap()],
                        &[&self.activations.moe_down_id_out],
                    );
                    let dn_params = mlx_native::GgmlQuantizedMatmulIdParams {
                        n_tokens: top_k as u32,
                        top_k: 1,
                        n: hs as u32,
                        k: moe_int as u32,
                        n_experts: num_experts as u32,
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
                    ).map_err(|e| anyhow::anyhow!("down _id L{layer_idx}: {e}"))?;
                    total_dispatches += 1;

                    // post-FF norm1: mlp_down → attn_out (concurrent with down_id)
                    s.barrier_between(
                        &[&self.activations.mlp_down],
                        &[&self.activations.attn_out],
                    );
                    s.rms_norm(
                        reg, metal_dev,
                        &self.activations.mlp_down,
                        &self.layers[layer_idx].norms.post_feedforward_layernorm_1,
                        &self.activations.attn_out,
                        &self.activations.norm_params,
                        1, hs as u32,
                    ).map_err(|e| anyhow::anyhow!("post-FF norm 1 L{layer_idx}: {e}"))?;
                    total_dispatches += 1;

                    // -- B14: weighted_sum (singleton) --
                    s.barrier_between(
                        &[&self.activations.moe_down_id_out, &self.activations.moe_routing_weights_gpu],
                        &[&self.activations.moe_accum],
                    );
                    mlx_native::ops::moe_dispatch::moe_weighted_sum_encode(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.moe_down_id_out,
                        &self.activations.moe_routing_weights_gpu,
                        &self.activations.moe_accum,
                        hs, top_k,
                    ).map_err(|e| anyhow::anyhow!("weighted_sum L{layer_idx}: {e}"))?;
                    total_dispatches += 1;
                } else {
                    // Fallback: per-expert loop (all in same session)
                    mlx_native::ops::moe_dispatch::moe_zero_buffer_encode(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.moe_accum, hs,
                    ).map_err(|e| anyhow::anyhow!("zero_buffer L{layer_idx}: {e}"))?;

                    // Note: fallback path still needs CPU to read expert_ids.
                    // For now, this path is unused (all layers have stacked weights).
                    // If needed, we'd add a finish/begin here, but the fused _id path
                    // is always available for Gemma4.
                    anyhow::bail!(
                        "Single-session forward requires fused _id path (stacked weights). \
                         Layer {layer_idx} missing stacked weights."
                    );
                }

                // ============================================================
                // GPU post-MoE: norm, combine MLP+MoE, final norm, residual, scalar
                // ============================================================

                // -- Fused post-FF norm 2 + combine MLP+MoE --
                s.barrier_between(
                    &[&self.activations.attn_out, &self.activations.moe_accum],
                    &[&self.activations.mlp_down],
                );
                mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.attn_out,
                    &self.activations.moe_accum,
                    &self.layers[layer_idx].norms.post_feedforward_layernorm_2,
                    &self.activations.mlp_down,
                    hs as u32, 1, eps,
                ).map_err(|e| anyhow::anyhow!("fused post-FF norm2+combine L{layer_idx}: {e}"))?;
                total_dispatches += 1;

                // -- Fused end-of-layer: post-FF norm + residual add + scalar mul --
                let scalar_is_vector = self.layers[layer_idx].layer_scalar.element_count() > 1;
                s.barrier_between(
                    &[&self.activations.residual, &self.activations.mlp_down],
                    &[&self.activations.hidden],
                );
                mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_scalar_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.residual,
                    &self.activations.mlp_down,
                    &self.layers[layer_idx].norms.post_feedforward_layernorm,
                    &self.activations.hidden,
                    &self.layers[layer_idx].layer_scalar,
                    1, hs as u32, eps,
                    scalar_is_vector,
                ).map_err(|e| anyhow::anyhow!("fused end-of-layer L{layer_idx}: {e}"))?;
                total_dispatches += 1;

                if let Some(ref mut p) = profile {
                    // All layer ops in single session — attribute everything to S1
                    p.s1_dispatches[layer_idx] = total_dispatches;
                }

                // ADR-009 Phase 3A: per-layer hidden state dump.
                // Commits the session mid-forward to read hidden state, then re-starts.
                // Only active when HF2Q_DUMP_LAYERS=<seq_pos> matches.
                if dump_layers {
                    s.finish()
                        .map_err(|e| anyhow::anyhow!("dump layer finish L{layer_idx}: {e}"))?;
                    dumps::dump_f32(&self.activations.hidden, hs,
                        "l_out", Some(layer_idx), seq_pos)?;
                    // Re-start session for remaining layers
                    s = exec.begin()
                        .map_err(|e| anyhow::anyhow!("dump layer re-begin L{layer_idx}: {e}"))?;
                }

                // Dual command buffer: commit buf0 after N layers, start buf1.
                // GPU begins executing buf0 immediately. CPU continues encoding
                // buf1 on the main thread — the overlap is implicit because Metal
                // command buffer execution is asynchronous.
                //
                // Tested and falsified:
                // - Sequential wait BEFORE encode: -5.6 tok/s (serialized pipeline)
                // - Threaded wait DURING encode:   -43 tok/s (thread spawn + Metal
                //   cross-thread synchronization overhead on command queue)
                // The async overlap without any wait is the correct approach.
                if dual_buffer_split == Some(layer_idx + 1) {
                    let b0_barriers = s.barrier_count();
                    let _b0_encoder = s.commit(); // commit buf0 → GPU starts async
                    s = exec.begin().map_err(|e| anyhow::anyhow!("dual-buffer begin: {e}"))?;
                    s.track_dispatch(&[], &[&self.activations.hidden]);
                    if INVESTIGATION_ENV.mlx_timing {
                        eprintln!("  [DUAL_BUFFER] split at layer {} — buf0: {} dispatches, {} barriers",
                            layer_idx + 1, total_dispatches, b0_barriers);
                    }
                }
            }

            // --- Body/Head timing split (HF2Q_SPLIT_TIMING=1) ---
            // Inserts a commit_and_wait between layers and head to measure each
            // GPU section separately. Adds ~50μs sync overhead — measurement only.
            let body_dispatches = total_dispatches;
            let split_timing = INVESTIGATION_ENV.split_timing;
            if split_timing {
                let body_barriers = s.barrier_count();
                let (enc_ns, gpu_ns) = s.finish_with_timing(session_start)
                    .map_err(|e| anyhow::anyhow!("body finish: {e}"))?;
                eprintln!("  [SPLIT] BODY: encode={:.2}ms gpu={:.2}ms dispatches={} barriers={}",
                    enc_ns as f64 / 1e6, gpu_ns as f64 / 1e6, body_dispatches, body_barriers);
                // Start a new session for the head
                s = exec.begin().map_err(|e| anyhow::anyhow!("head session: {e}"))?;
            }

            // --- 3. Final norm + lm_head + softcap + argmax (all GPU) ---

            // GPU final RMS norm: hidden → norm_out
            s.barrier_between(
                &[&self.activations.hidden, &self.final_norm],
                &[&self.activations.norm_out],
            );
            s.rms_norm(
                reg, metal_dev,
                &self.activations.hidden,
                &self.final_norm,
                &self.activations.norm_out,
                &self.activations.norm_params,
                1, hs as u32,
            ).map_err(|e| anyhow::anyhow!("final norm: {e}"))?;
            total_dispatches += 1;

            // --- ADR-009 Phase 3A: boundary dump at specific token position ---
            if dump_pos == Some(seq_pos) {
                // Finish session to read GPU buffers.
                s.finish().map_err(|e| anyhow::anyhow!("dump boundary finish: {e}"))?;
                // Pre-lm_head = final_norm applied to hidden.
                dumps::dump_f32(&self.activations.norm_out, hs,
                    "pre_lmhead", None, seq_pos)?;
                // Re-begin session for lm_head + argmax.
                s = exec.begin().map_err(|e| anyhow::anyhow!("dump boundary re-begin: {e}"))?;
            }

            // GPU lm_head: whichever weight buffer was chosen at load time
            // (Q8_0 auto-enabled for large vocab × hidden models, F16 otherwise).
            if let Some(ref q8) = self.lm_head_q8 {
                s.barrier_between(
                    &[&self.activations.norm_out, &q8.buffer],
                    &[&self.activations.logits],
                );
                dispatch_qmatmul(
                    &mut s, reg, dev,
                    &self.activations.norm_out,
                    q8,
                    &mut self.activations.logits,
                    1,
                )?;
                total_dispatches += 1;
            } else if let Some(ref lm_head_f16) = self.lm_head_f16 {
                // Mixed-precision mat-vec (F32 input × F16 weights → F32 output).
                // Single dispatch replaces the old 3-dispatch path (cast + gemm + cast).
                s.barrier_between(
                    &[&self.activations.norm_out, lm_head_f16],
                    &[&self.activations.logits],
                );
                let gemm_params = DenseGemmF16Params {
                    m: 1,
                    n: vocab_size as u32,
                    k: hs as u32,
                };
                mlx_native::ops::dense_gemm::dispatch_dense_matvec_f16w_f32io(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.norm_out,  // F32 input (no cast needed)
                    lm_head_f16,                 // F16 weights
                    &self.activations.logits,    // F32 output (no cast needed)
                    &gemm_params,
                ).map_err(|e| anyhow::anyhow!("lm_head mixed-precision: {e}"))?;
                total_dispatches += 1;
            } else {
                anyhow::bail!("Single-session forward requires GPU lm_head (F16 weight)");
            }

            // GPU softcap (if configured)
            if let Some(cap) = self.final_logit_softcapping {
                s.barrier_between(
                    &[&self.activations.logits],
                    &[&self.activations.logits],  // in-place
                );
                mlx_native::ops::softcap::dispatch_softcap(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.logits,
                    &self.activations.logits,
                    &self.activations.softcap_params,
                    cap,
                ).map_err(|e| anyhow::anyhow!("GPU softcap: {e}"))?;
                total_dispatches += 1;
            }

            // GPU argmax
            s.barrier_between(
                &[&self.activations.logits],
                &[&self.activations.argmax_index, &self.activations.argmax_value],
            );
            mlx_native::ops::argmax::dispatch_argmax_f32(
                s.encoder_mut(), reg, metal_dev,
                &self.activations.logits,
                &self.activations.argmax_index,
                &self.activations.argmax_value,
                &self.activations.argmax_params,
                vocab_size as u32,
            ).map_err(|e| anyhow::anyhow!("GPU argmax: {e}"))?;
            total_dispatches += 1;


            // === ONE finish() for the entire forward pass ===
            let head_dispatches = total_dispatches - body_dispatches;
            let barrier_count = s.barrier_count();
            let is_recording = s.is_recording();
            if is_recording {
                let (enc_ns, gpu_ns, fusions, reordered, b0, b1) =
                    s.finish_optimized_with_timing(reg, metal_dev, session_start)
                        .map_err(|e| anyhow::anyhow!("optimized session finish: {e}"))?;
                if INVESTIGATION_ENV.mlx_timing {
                    eprintln!("  [TIMING] encode={:.2}ms gpu_wait={:.2}ms dispatches={} barriers={}",
                        enc_ns as f64 / 1e6, gpu_ns as f64 / 1e6, total_dispatches, barrier_count);
                    eprintln!("  [GRAPH_OPT] fusions={} reordered={} barriers={}+{}",
                        fusions, reordered, b0, b1);
                }
            } else {
                let head_barriers = barrier_count; // snapshot before finish consumes s
                let (enc_ns, gpu_ns) = s.finish_with_timing(session_start)
                    .map_err(|e| anyhow::anyhow!("single session finish: {e}"))?;
                if INVESTIGATION_ENV.mlx_timing {
                    if split_timing {
                        eprintln!("  [SPLIT] HEAD: encode={:.2}ms gpu={:.2}ms dispatches={} barriers={}",
                            enc_ns as f64 / 1e6, gpu_ns as f64 / 1e6, head_dispatches, head_barriers);
                    } else {
                        eprintln!("  [TIMING] encode={:.2}ms gpu_wait={:.2}ms dispatches={} barriers={}",
                            enc_ns as f64 / 1e6, gpu_ns as f64 / 1e6, total_dispatches, barrier_count);
                    }
                }
            }
        }
        let session_us = session_start.elapsed().as_secs_f64() * 1e6;

        // --- ADR-009 Phase 3A: dump post-lm_head logits at boundary position ---
        if dump_pos == Some(seq_pos) {
            dumps::dump_f32(&self.activations.logits, vocab_size,
                "logits", None, seq_pos)?;
            // Also dump top-10 logits for quick inspection.
            let logits_data: &[f32] = self.activations.logits.as_slice()
                .map_err(|e| anyhow::anyhow!("dump top-10 read: {e}"))?;
            let mut indexed: Vec<(usize, f32)> = logits_data[..vocab_size]
                .iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            eprintln!("[DUMP] top-10 logits at pos {seq_pos}:");
            for (tok_id, logit) in indexed.iter().take(10) {
                eprintln!("  tok={tok_id:>6} logit={logit:.6}");
            }
        }

        // Read the Q8 GPU argmax result (8 bytes: 1 u32 index + 1 f32 value).
        let gpu_top1: u32 = {
            let idx: &[u32] = self.activations.argmax_index.as_slice()
                .map_err(|e| anyhow::anyhow!("argmax read: {e}"))?;
            idx[0]
        };

        // Q8 coarse → F32 exact rerank.
        //
        // Data shows Q8_0 lm_head adds ~2.5–5e-3 logit noise. Any token within
        // that envelope of top-1 has non-trivial chance of flipping. The pad
        // case is the most visible symptom; the mechanism is symmetric.
        //
        // Fix: keep Q8 for coarse scoring, but for the small set of tokens
        // plausibly eligible to win, recompute exact F32 logits from the F32
        // `embed_weight` (already resident) and take argmax on those.
        //
        // Candidate set (O(~100) tokens):
        //   - top-K Q8 tokens (K=64)
        //   - all tokens within delta=0.01 of Q8 top-1
        //   - special tokens: 0 <pad>, 1 <eos>, 2 <bos>, 105 <|turn>, 106 <turn|>
        //
        // Rerank is skipped when lm_head is already F16 (no coarse noise to
        // correct) or when HF2Q_LMHEAD_RERANK=0.
        let rerank_active = self.lm_head_q8.is_some()
            && !INVESTIGATION_ENV.lmhead_rerank_disabled;
        let token_id: u32 = if rerank_active {
            // CPU candidate selection via threshold scan over the full Q8
            // logits. GPU top-K was explored but a single-threadgroup
            // top-K on vocab=262144 serializes phase 2 onto one thread
            // and costs ~5 ms/token — worse than the ~40 μs CPU scan.
            //
            // Algorithm: read the Q8 top-1 value (from the GPU argmax
            // output), then collect all tokens with logit ≥ top1 - delta,
            // plus specials. Delta is chosen larger than the observed
            // Q8 noise envelope (~5e-3) so the true winner is always in
            // the set.
            let top1_q8_val: f32 = {
                let v: &[f32] = self.activations.argmax_value.as_slice()
                    .map_err(|e| anyhow::anyhow!("argmax_value read: {e}"))?;
                v[0]
            };
            // Headroom for Q8 noise. Empirical Q8 noise envelope is ~5e-3
            // per logit, so delta=0.5 is a comfortable ~100× margin. The
            // candidate set remains small (~10–100 tokens typically) because
            // real top-K distributions fall off quickly below the winner.
            let delta: f32 = 0.5;
            let threshold = top1_q8_val - delta;

            let logits: &[f32] = self.activations.logits.as_slice()
                .map_err(|e| anyhow::anyhow!("rerank logits read: {e}"))?;
            let hidden: &[f32] = self.activations.norm_out.as_slice()
                .map_err(|e| anyhow::anyhow!("rerank norm_out read: {e}"))?;
            let embed_f32: &[f32] = self.embed_weight.as_slice()
                .map_err(|e| anyhow::anyhow!("rerank embed read: {e}"))?;

            let mut candidates: Vec<u32> = Vec::with_capacity(64);
            for (i, &v) in logits[..vocab_size].iter().enumerate() {
                if v >= threshold {
                    candidates.push(i as u32);
                }
            }
            // Specials always included.
            for sp in [0u32, 1, 2, 105, 106] {
                if (sp as usize) < vocab_size {
                    candidates.push(sp);
                }
            }
            candidates.sort_unstable();
            candidates.dedup();

            // Exact F32 rerank via hidden · embed_row. Softcap is monotonic
            // so skipping it doesn't change argmax order. F64 accumulator
            // for precision; the set is tiny so cost is negligible.
            let mut best_tok: u32 = gpu_top1;
            let mut best_logit: f32 = f32::NEG_INFINITY;
            for &tok in &candidates {
                let row_off = (tok as usize) * hs;
                if row_off + hs > embed_f32.len() { continue; }
                let row = &embed_f32[row_off..row_off + hs];
                let mut acc: f64 = 0.0;
                for i in 0..hs {
                    acc += (hidden[i] as f64) * (row[i] as f64);
                }
                let l = acc as f32;
                if l > best_logit {
                    best_logit = l;
                    best_tok = tok;
                }
            }
            best_tok
        } else {
            gpu_top1
        };

        // Diagnostic: when <pad> (id 0) still wins AFTER rerank (or when
        // rerank is off and <pad> wins raw), dump top-10 Q8 logits so we
        // see whether pad is near-tie or a genuine model preference.
        if token_id == 0 {
            let logits: &[f32] = self.activations.logits.as_slice()
                .map_err(|e| anyhow::anyhow!("pad diag logits read: {e}"))?;
            let vocab = logits.len().min(vocab_size);
            let mut indexed: Vec<(usize, f32)> = logits[..vocab]
                .iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            eprintln!("\n[PAD-DIAG] <pad> won at seq_pos={} (rerank={}). Top 10 Q8 logits:",
                seq_pos, if rerank_active { "on" } else { "off" });
            for (tok_id, logit) in indexed.iter().take(10) {
                eprintln!("  tok={tok_id:>6} logit={logit:>10.6}");
            }
            let pad_rank = indexed.iter().position(|&(i, _)| i == 0).unwrap_or(999);
            let pad_logit = logits[0];
            let rank1_logit = indexed[0].1;
            eprintln!("  <pad> rank={} logit={:.6}  vs top-1 logit={:.6}  gap={:.6e}",
                pad_rank, pad_logit, rank1_logit, (rank1_logit - pad_logit).abs());
        }

        if let Some(ref mut p) = profile {
            // Single session — report all time in S1, zero everything else
            let per_layer_us = session_us / num_layers as f64;
            for li in 0..num_layers {
                p.layer_s1_us[li] = per_layer_us;
                p.layer_cpu1_us[li] = 0.0;
                p.layer_s2_us[li] = 0.0;
                p.layer_cpu2_us[li] = 0.0;
                p.layer_s3_us[li] = 0.0;
                p.layer_cpu3_us[li] = 0.0;
                p.layer_s4_us[li] = 0.0;
                p.layer_cpu4_us[li] = 0.0;
                p.s2_dispatches[li] = 0;
                p.s3_dispatches[li] = 0;
                p.s4_dispatches[li] = 0;
            }
            p.head_session_us = 0.0; // included in single session
            p.head_cpu_us = 0.0;
            p.head_dispatches = total_dispatches;
            p.total_us = token_start.elapsed().as_secs_f64() * 1e6;
        }

        Ok(token_id)
    }

    // forward_prefill() is defined in forward_prefill.rs (ADR-009 Track 1).

    /// Per-kernel-type profiling forward pass.
    ///
    /// Breaks the single session into one session PER KERNEL TYPE PER LAYER,
    /// using `finish_with_timing` to measure GPU wait time for each group.
    ///
    /// This is intentionally slow (many sessions = many sync points) but gives
    /// us per-kernel-type GPU timing to compare against candle Phase 0 data.
    ///
    /// Gated by `HF2Q_MLX_KERNEL_PROFILE=1`.
    pub fn forward_decode_kernel_profile(
        &mut self,
        input_token: u32,
        seq_pos: usize,
        gpu: &mut GpuContext,
    ) -> Result<(u32, KernelTypeProfile)> {
        let hs = self.hidden_size;
        let num_layers = self.layers.len();
        let vocab_size = self.vocab_size;

        let mut kp = KernelTypeProfile {
            qkv_matmuls_us: vec![0.0; num_layers],
            head_norms_rope_us: vec![0.0; num_layers],
            kv_cache_copy_us: vec![0.0; num_layers],
            sdpa_us: vec![0.0; num_layers],
            o_proj_us: vec![0.0; num_layers],
            mlp_matmuls_us: vec![0.0; num_layers],
            moe_us: vec![0.0; num_layers],
            norms_adds_us: vec![0.0; num_layers],
            ..Default::default()
        };

        // Write position buffer
        {
            let pos_dst: &mut [u32] = self.activations.position.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("position write: {e}"))?;
            pos_dst[0] = seq_pos as u32;
        }

        // KV cache bookkeeping
        let mut kv_info: Vec<(bool, usize, usize, usize)> = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let is_sliding = self.kv_caches[layer_idx].is_sliding;
            let write_pos = self.kv_caches[layer_idx].write_pos;
            let capacity = self.kv_caches[layer_idx].capacity;
            self.kv_caches[layer_idx].write_pos += 1;
            self.kv_caches[layer_idx].seq_len = self.kv_caches[layer_idx].seq_len.saturating_add(1)
                .min(capacity);
            let seq_len = self.kv_caches[layer_idx].seq_len;
            kv_info.push((is_sliding, write_pos, capacity, seq_len));
        }

        // --- Embedding (tiny, single session) ---
        {
            let (exec, reg) = gpu.split();
            let dev = exec.device();
            let metal_dev = dev.metal_device();
            let mut s = exec.begin().map_err(|e| anyhow::anyhow!("embed begin: {e}"))?;
            mlx_native::ops::elementwise::embedding_gather_scale_f32(
                s.encoder_mut(), reg, metal_dev,
                &self.embed_weight, &self.activations.hidden,
                input_token, hs, (hs as f32).sqrt(),
            ).map_err(|e| anyhow::anyhow!("embedding: {e}"))?;
            s.finish().map_err(|e| anyhow::anyhow!("embed finish: {e}"))?;
            // Embedding time intentionally not reported: trivial cost relative
            // to the per-layer kernel sessions profiled below.
        }

        // --- Per-layer kernel-type sessions ---
        //
        // clippy::needless_range_loop stays off here: the body writes into 8
        // parallel `kp.*_us` profile vectors and indexes `kv_info`/`self.kv_caches`
        // by layer_idx. Zipping all of them into one iterator chain would be
        // much less readable than the index form. Migration note: the `layer`
        // binding covers the per-layer config (head_dim, num_kv_heads, layer_type)
        // and attn/moe/norms accesses.
        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..num_layers {
            let layer = &self.layers[layer_idx];
            let hd = layer.head_dim;
            let nkv = layer.num_kv_heads;
            let nh = self.num_attention_heads;
            let is_sliding = layer.layer_type == LayerType::Sliding;
            let eps = self.rms_norm_eps;
            let (kv_is_sliding, kv_write_pos, kv_capacity, kv_seq_len) = kv_info[layer_idx];
            let v_is_k = layer.attn.v_proj.is_none();

            // ============================================================
            // GROUP 1: QKV matmuls (pre-attn norm + Q + K + V projections)
            // ============================================================
            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let metal_dev = dev.metal_device();
                let t0 = Instant::now();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("qkv begin L{layer_idx}: {e}"))?;

                // pre-attn norm
                s.rms_norm(
                    reg, metal_dev,
                    &self.activations.hidden,
                    &self.layers[layer_idx].norms.input_layernorm,
                    &self.activations.norm_out,
                    &self.activations.norm_params,
                    1, hs as u32,
                ).map_err(|e| anyhow::anyhow!("pre-attn norm L{layer_idx}: {e}"))?;

                // Q proj
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    &self.layers[layer_idx].attn.q_proj, &mut self.activations.attn_q, 1)?;
                // K proj
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    &self.layers[layer_idx].attn.k_proj, &mut self.activations.attn_k, 1)?;
                // V proj (if not k_eq_v)
                if !v_is_k {
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                        self.layers[layer_idx].attn.v_proj.as_ref().unwrap(),
                        &mut self.activations.attn_v, 1)?;
                }

                let (_enc_ns, gpu_ns) = s.finish_with_timing(t0)
                    .map_err(|e| anyhow::anyhow!("qkv finish L{layer_idx}: {e}"))?;
                kp.qkv_matmuls_us[layer_idx] = gpu_ns as f64 / 1000.0;
            }

            // ============================================================
            // GROUP 2: Head norms + RoPE (fused Q norm+RoPE, fused K norm+RoPE, V norm)
            // ============================================================
            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let metal_dev = dev.metal_device();
                let t0 = Instant::now();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("norms begin L{layer_idx}: {e}"))?;

                let ff_gpu = if is_sliding { None } else { Some(&self.activations.rope_freq_factors_gpu) };
                let theta = if is_sliding { self.rope_theta_sliding } else { self.rope_theta_global };
                let half_rope = (hd / 2) as u32;

                // Fused Q norm+RoPE
                mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.attn_q, &self.activations.attn_q_normed,
                    Some(&self.layers[layer_idx].attn.q_norm_weight),
                    &self.activations.position, ff_gpu,
                    nh as u32, hd as u32, half_rope, eps, theta,
                ).map_err(|e| anyhow::anyhow!("fused Q L{layer_idx}: {e}"))?;

                // Fused K norm+RoPE
                mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.attn_k, &self.activations.attn_k_normed,
                    Some(&self.layers[layer_idx].attn.k_norm_weight),
                    &self.activations.position, ff_gpu,
                    nkv as u32, hd as u32, half_rope, eps, theta,
                ).map_err(|e| anyhow::anyhow!("fused K L{layer_idx}: {e}"))?;

                // V norm
                let hd_norm_params = if is_sliding {
                    &self.activations.norm_params_sliding_hd
                } else {
                    &self.activations.norm_params_global_hd
                };
                if v_is_k {
                    dispatch_rms_norm_unit_perhead(
                        s.encoder_mut(), reg, metal_dev,
                        &RmsNormPerHeadArgs {
                            input: &self.activations.attn_k,
                            output: &self.activations.attn_v,
                            params_buf: hd_norm_params,
                            rows: nkv as u32,
                            dim: hd as u32,
                        },
                    )?;
                } else {
                    dispatch_rms_norm_unit_perhead(
                        s.encoder_mut(), reg, metal_dev,
                        &RmsNormPerHeadArgs {
                            input: &self.activations.attn_v,
                            output: &self.activations.moe_expert_out,
                            params_buf: hd_norm_params,
                            rows: nkv as u32,
                            dim: hd as u32,
                        },
                    )?;
                }

                let (_enc_ns, gpu_ns) = s.finish_with_timing(t0)
                    .map_err(|e| anyhow::anyhow!("norms finish L{layer_idx}: {e}"))?;
                kp.head_norms_rope_us[layer_idx] = gpu_ns as f64 / 1000.0;
            }

            let v_src = if v_is_k { &self.activations.attn_v } else { &self.activations.moe_expert_out };

            // ============================================================
            // GROUP 3: KV cache Hadamard-quantize (2 dispatches, ADR-007)
            // ============================================================
            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let metal_dev = dev.metal_device();
                let t0 = Instant::now();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("kv begin L{layer_idx}: {e}"))?;

                let cache_pos_val = if kv_is_sliding {
                    (kv_write_pos % kv_capacity) as u32
                } else {
                    kv_write_pos as u32
                };
                mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.attn_k_normed,
                    &self.kv_caches[layer_idx].k_packed,
                    &self.kv_caches[layer_idx].k_norms,
                    nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                    kv_is_sliding,
                ).map_err(|e| anyhow::anyhow!("hadamard_quantize K L{layer_idx}: {e}"))?;
                mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                    s.encoder_mut(), reg, metal_dev,
                    v_src,
                    &self.kv_caches[layer_idx].v_packed,
                    &self.kv_caches[layer_idx].v_norms,
                    nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                    kv_is_sliding,
                ).map_err(|e| anyhow::anyhow!("hadamard_quantize V L{layer_idx}: {e}"))?;

                let (_enc_ns, gpu_ns) = s.finish_with_timing(t0)
                    .map_err(|e| anyhow::anyhow!("kv finish L{layer_idx}: {e}"))?;
                kp.kv_cache_copy_us[layer_idx] = gpu_ns as f64 / 1000.0;
            }

            // ============================================================
            // GROUP 4: SDPA TQ (FWHT fused — Q rotation + output inv-rotation in-kernel)
            // ============================================================
            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let t0 = Instant::now();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("sdpa begin L{layer_idx}: {e}"))?;

                {
                    // ADR-009 Track 2: ring_start for correct sliding-window
                    // chronology after wrap. Before wrap, ring_start = 0
                    // (physical == logical). After wrap, ring_start = write_pos
                    // % capacity (physical slot of the oldest cached entry).
                    let ring_start = if kv_is_sliding && kv_seq_len >= kv_capacity {
                        (kv_write_pos % kv_capacity) as u32
                    } else {
                        0
                    };
                    let p = FlashAttnVecTqParams {
                        num_heads: nh as u32,
                        num_kv_heads: nkv as u32,
                        head_dim: hd as u32,
                        kv_seq_len: kv_seq_len as u32,
                        kv_capacity: kv_capacity as u32,
                        scale: 1.0,
                        mask_type: if is_sliding { 2 } else { 1 },
                        sliding_window: if is_sliding { self.sliding_window as u32 } else { 0 },
                        softcap: 0.0,
                        ring_start,
                    };
                    mlx_native::ops::flash_attn_vec_tq::flash_attn_vec_tq(
                        s.encoder_mut(), reg, dev,
                        &self.activations.attn_q_normed,
                        &self.kv_caches[layer_idx].k_packed,
                        &self.kv_caches[layer_idx].k_norms,
                        &self.kv_caches[layer_idx].v_packed,
                        &self.kv_caches[layer_idx].v_norms,
                        &self.activations.sdpa_out,
                        &self.activations.sdpa_tmp,
                        &p,
                    ).map_err(|e| anyhow::anyhow!("flash_attn_vec_tq L{layer_idx}: {e}"))?;
                }

                let (_enc_ns, gpu_ns) = s.finish_with_timing(t0)
                    .map_err(|e| anyhow::anyhow!("sdpa finish L{layer_idx}: {e}"))?;
                kp.sdpa_us[layer_idx] = gpu_ns as f64 / 1000.0;
            }

            // ============================================================
            // GROUP 5: O-proj matmul (1 dispatch)
            // ============================================================
            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let t0 = Instant::now();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("oproj begin L{layer_idx}: {e}"))?;

                dispatch_qmatmul(&mut s, reg, dev, &self.activations.sdpa_out,
                    &self.layers[layer_idx].attn.o_proj, &mut self.activations.attn_out, 1)?;

                let (_enc_ns, gpu_ns) = s.finish_with_timing(t0)
                    .map_err(|e| anyhow::anyhow!("oproj finish L{layer_idx}: {e}"))?;
                kp.o_proj_us[layer_idx] = gpu_ns as f64 / 1000.0;
            }

            // ============================================================
            // GROUP 6: MLP matmuls (post-attn norm+add, pre-FF norm, gate, up, gelu_mul, down)
            // ============================================================
            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let metal_dev = dev.metal_device();
                let t0 = Instant::now();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("mlp begin L{layer_idx}: {e}"))?;

                // Fused post-attn norm+add (needed to produce residual for MLP)
                mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.hidden, &self.activations.attn_out,
                    &self.layers[layer_idx].norms.post_attention_layernorm,
                    &self.activations.residual,
                    hs as u32, 1, eps,
                ).map_err(|e| anyhow::anyhow!("post-attn norm+add L{layer_idx}: {e}"))?;

                // Pre-FF norm
                s.rms_norm(
                    reg, metal_dev,
                    &self.activations.residual,
                    &self.layers[layer_idx].norms.pre_feedforward_layernorm,
                    &self.activations.norm_out,
                    &self.activations.norm_params,
                    1, hs as u32,
                ).map_err(|e| anyhow::anyhow!("pre-FF norm L{layer_idx}: {e}"))?;

                // gate
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    &self.layers[layer_idx].mlp.gate_proj, &mut self.activations.mlp_gate, 1)?;
                // up
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    &self.layers[layer_idx].mlp.up_proj, &mut self.activations.mlp_up, 1)?;
                // fused gelu_mul
                {
                    use mlx_native::ops::encode_helpers::{encode_with_args, KernelArg};
                    let n_elements_bytes = (self.intermediate_size as u32).to_ne_bytes();
                    let pipeline = reg.get_pipeline("fused_gelu_mul", metal_dev)?;
                    encode_with_args(
                        s.encoder_mut(), pipeline,
                        &[
                            (0, KernelArg::Buffer(&self.activations.mlp_gate)),
                            (1, KernelArg::Buffer(&self.activations.mlp_up)),
                            (2, KernelArg::Buffer(&self.activations.mlp_fused)),
                            (3, KernelArg::Bytes(&n_elements_bytes)),
                        ],
                        mlx_native::MTLSize::new(self.intermediate_size as u64, 1, 1),
                        mlx_native::MTLSize::new(
                            std::cmp::min(256, self.intermediate_size as u64), 1, 1),
                    );
                }
                // down
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.mlp_fused,
                    &self.layers[layer_idx].mlp.down_proj, &mut self.activations.mlp_down, 1)?;

                let (_enc_ns, gpu_ns) = s.finish_with_timing(t0)
                    .map_err(|e| anyhow::anyhow!("mlp finish L{layer_idx}: {e}"))?;
                kp.mlp_matmuls_us[layer_idx] = gpu_ns as f64 / 1000.0;
            }

            // ============================================================
            // GROUP 7: MoE (routing + experts)
            // ============================================================
            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let metal_dev = dev.metal_device();
                let t0 = Instant::now();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("moe begin L{layer_idx}: {e}"))?;

                // Post-FF norm 1
                s.rms_norm(
                    reg, metal_dev,
                    &self.activations.mlp_down,
                    &self.layers[layer_idx].norms.post_feedforward_layernorm_1,
                    &self.activations.attn_out,
                    &self.activations.norm_params,
                    1, hs as u32,
                ).map_err(|e| anyhow::anyhow!("post-FF norm 1 L{layer_idx}: {e}"))?;

                // Pre-FF norm 2
                s.rms_norm(
                    reg, metal_dev,
                    &self.activations.residual,
                    &self.layers[layer_idx].norms.pre_feedforward_layernorm_2,
                    &self.activations.moe_norm_out,
                    &self.activations.norm_params,
                    1, hs as u32,
                ).map_err(|e| anyhow::anyhow!("pre-FF norm 2 L{layer_idx}: {e}"))?;

                // Router norm
                s.rms_norm(
                    reg, metal_dev,
                    &self.activations.residual,
                    &self.layers[layer_idx].moe.router_combined_weight,
                    &self.activations.norm_out,
                    &self.activations.norm_params,
                    1, hs as u32,
                ).map_err(|e| anyhow::anyhow!("router norm L{layer_idx}: {e}"))?;

                // Router proj
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    &self.layers[layer_idx].moe.router_proj,
                    &mut self.activations.moe_router_logits, 1)?;

                // Fused MoE routing
                let num_experts = self.num_experts;
                let top_k = self.layers[layer_idx].moe.top_k;
                mlx_native::ops::fused_norm_add::dispatch_fused_moe_routing_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.moe_router_logits,
                    &self.activations.moe_expert_ids,
                    &self.activations.moe_routing_weights_gpu,
                    &self.layers[layer_idx].moe.per_expert_scale,
                    num_experts as u32, top_k as u32,
                ).map_err(|e| anyhow::anyhow!("fused MoE routing L{layer_idx}: {e}"))?;

                // MoE experts (fused _id path)
                let moe_int = self.layers[layer_idx].moe.moe_intermediate_size;
                let use_fused_id = self.layers[layer_idx].moe.stacked_gate_up.is_some()
                    && self.layers[layer_idx].moe.stacked_down.is_some();
                if !use_fused_id {
                    anyhow::bail!("Kernel profile requires fused _id path (stacked weights). Layer {layer_idx} missing.");
                }

                let ggml_type_gu = self.layers[layer_idx].moe.gate_up_ggml_dtype;
                let ggml_type_dn = self.layers[layer_idx].moe.down_ggml_dtype;

                // gate_up _id
                let gu_params = mlx_native::GgmlQuantizedMatmulIdParams {
                    n_tokens: 1,
                    top_k: top_k as u32,
                    n: (2 * moe_int) as u32,
                    k: hs as u32,
                    n_experts: num_experts as u32,
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
                ).map_err(|e| anyhow::anyhow!("gate_up _id L{layer_idx}: {e}"))?;

                // Batched SwiGLU
                mlx_native::ops::moe_dispatch::moe_swiglu_batch_encode(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.moe_gate_up_id_out,
                    &self.activations.moe_swiglu_id_out,
                    moe_int, top_k,
                ).map_err(|e| anyhow::anyhow!("swiglu batch L{layer_idx}: {e}"))?;

                // down _id
                let dn_params = mlx_native::GgmlQuantizedMatmulIdParams {
                    n_tokens: top_k as u32,
                    top_k: 1,
                    n: hs as u32,
                    k: moe_int as u32,
                    n_experts: num_experts as u32,
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
                ).map_err(|e| anyhow::anyhow!("down _id L{layer_idx}: {e}"))?;

                // Weighted sum
                mlx_native::ops::moe_dispatch::moe_weighted_sum_encode(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.moe_down_id_out,
                    &self.activations.moe_routing_weights_gpu,
                    &self.activations.moe_accum,
                    hs, top_k,
                ).map_err(|e| anyhow::anyhow!("weighted_sum L{layer_idx}: {e}"))?;

                let (_enc_ns, gpu_ns) = s.finish_with_timing(t0)
                    .map_err(|e| anyhow::anyhow!("moe finish L{layer_idx}: {e}"))?;
                kp.moe_us[layer_idx] = gpu_ns as f64 / 1000.0;
            }

            // ============================================================
            // GROUP 8: Fused norms/adds/end-of-layer
            // ============================================================
            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let metal_dev = dev.metal_device();
                let t0 = Instant::now();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("norms_end begin L{layer_idx}: {e}"))?;

                // Fused post-FF norm2 + combine MLP+MoE
                mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.attn_out, &self.activations.moe_accum,
                    &self.layers[layer_idx].norms.post_feedforward_layernorm_2,
                    &self.activations.mlp_down,
                    hs as u32, 1, eps,
                ).map_err(|e| anyhow::anyhow!("fused post-FF norm2+combine L{layer_idx}: {e}"))?;

                // Fused end-of-layer: post-FF norm + residual add + scalar mul
                let scalar_is_vector = self.layers[layer_idx].layer_scalar.element_count() > 1;
                mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_scalar_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.residual, &self.activations.mlp_down,
                    &self.layers[layer_idx].norms.post_feedforward_layernorm,
                    &self.activations.hidden,
                    &self.layers[layer_idx].layer_scalar,
                    1, hs as u32, eps, scalar_is_vector,
                ).map_err(|e| anyhow::anyhow!("fused end-of-layer L{layer_idx}: {e}"))?;

                let (_enc_ns, gpu_ns) = s.finish_with_timing(t0)
                    .map_err(|e| anyhow::anyhow!("norms_end finish L{layer_idx}: {e}"))?;
                kp.norms_adds_us[layer_idx] = gpu_ns as f64 / 1000.0;
            }
        }

        // --- Head: final norm + lm_head + softcap + argmax ---
        {
            let (exec, reg) = gpu.split();
            let dev = exec.device();
            let metal_dev = dev.metal_device();
            let t0 = Instant::now();
            let mut s = exec.begin().map_err(|e| anyhow::anyhow!("head begin: {e}"))?;

            // Final RMS norm
            s.rms_norm(
                reg, metal_dev,
                &self.activations.hidden, &self.final_norm,
                &self.activations.norm_out, &self.activations.norm_params,
                1, hs as u32,
            ).map_err(|e| anyhow::anyhow!("final norm: {e}"))?;

            // lm_head: cast F32->F16, dense GEMM F16, cast F16->F32
            if let Some(ref lm_head_f16) = self.lm_head_f16 {
                mlx_native::ops::elementwise::cast(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.norm_out, &self.activations.hidden_f16,
                    hs, CastDirection::F32ToF16,
                ).map_err(|e| anyhow::anyhow!("cast F32->F16: {e}"))?;

                let gemm_params = DenseGemmF16Params {
                    m: 1, n: vocab_size as u32, k: hs as u32,
                };
                mlx_native::ops::dense_gemm::dispatch_dense_gemm_f16(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.hidden_f16, lm_head_f16,
                    &self.activations.logits_f16, &gemm_params,
                ).map_err(|e| anyhow::anyhow!("dense_gemm_f16: {e}"))?;

                mlx_native::ops::elementwise::cast(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.logits_f16, &self.activations.logits,
                    vocab_size, CastDirection::F16ToF32,
                ).map_err(|e| anyhow::anyhow!("cast F16->F32: {e}"))?;
            } else {
                anyhow::bail!("Kernel profile requires GPU lm_head (F16 weight)");
            }

            // Softcap (params pre-initialized at model load time)
            if let Some(cap) = self.final_logit_softcapping {
                mlx_native::ops::softcap::dispatch_softcap(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.logits, &self.activations.logits,
                    &self.activations.softcap_params, cap,
                ).map_err(|e| anyhow::anyhow!("GPU softcap: {e}"))?;
            }

            // Argmax (params pre-initialized at model load time)
            mlx_native::ops::argmax::dispatch_argmax_f32(
                s.encoder_mut(), reg, metal_dev,
                &self.activations.logits, &self.activations.argmax_index,
                &self.activations.argmax_value, &self.activations.argmax_params,
                vocab_size as u32,
            ).map_err(|e| anyhow::anyhow!("GPU argmax: {e}"))?;

            let (_enc_ns, gpu_ns) = s.finish_with_timing(t0)
                .map_err(|e| anyhow::anyhow!("head finish: {e}"))?;
            kp.lm_head_us = gpu_ns as f64 / 1000.0;
        }

        // Read argmax result
        let token_id: u32 = {
            let idx: &[u32] = self.activations.argmax_index.as_slice()
                .map_err(|e| anyhow::anyhow!("argmax read: {e}"))?;
            idx[0]
        };

        Ok((token_id, kp))
    }

    /// Print the kernel-type profiling report comparing mlx-native vs candle.
    ///
    /// Expects results from multiple tokens (skipping warmup).
    pub fn print_kernel_profile_report(profiles: &[KernelTypeProfile]) {
        if profiles.is_empty() {
            eprintln!("[KERNEL_PROFILE] No tokens to report.");
            return;
        }
        let n = profiles.len();
        let num_layers = profiles[0].qkv_matmuls_us.len();

        // Compute median per-layer averages across tokens
        let median_sum = |getter: &dyn Fn(&KernelTypeProfile) -> &Vec<f64>| -> f64 {
            let mut sums: Vec<f64> = profiles.iter()
                .map(|p| getter(p).iter().sum::<f64>())
                .collect();
            sums.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sums[sums.len() / 2]
        };

        let qkv_total = median_sum(&|p| &p.qkv_matmuls_us);
        let norms_rope_total = median_sum(&|p| &p.head_norms_rope_us);
        let kv_cache_total = median_sum(&|p| &p.kv_cache_copy_us);
        let sdpa_total = median_sum(&|p| &p.sdpa_us);
        let o_proj_total = median_sum(&|p| &p.o_proj_us);
        let mlp_total = median_sum(&|p| &p.mlp_matmuls_us);
        let moe_total = median_sum(&|p| &p.moe_us);
        let norms_adds_total = median_sum(&|p| &p.norms_adds_us);

        let mut head_vals: Vec<f64> = profiles.iter().map(|p| p.lm_head_us).collect();
        head_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let head_total = head_vals[head_vals.len() / 2];

        let gpu_total = qkv_total + norms_rope_total + kv_cache_total + sdpa_total
            + o_proj_total + mlp_total + moe_total + norms_adds_total + head_total;

        // Per-layer averages (divide total by num_layers, except head)
        let qkv_per_layer = qkv_total / num_layers as f64;
        let norms_rope_per_layer = norms_rope_total / num_layers as f64;
        let kv_cache_per_layer = kv_cache_total / num_layers as f64;
        let sdpa_per_layer = sdpa_total / num_layers as f64;
        let o_proj_per_layer = o_proj_total / num_layers as f64;
        let mlp_per_layer = mlp_total / num_layers as f64;
        let moe_per_layer = moe_total / num_layers as f64;
        let norms_adds_per_layer = norms_adds_total / num_layers as f64;

        // Candle Phase 0 reference values (us_per_call from phase0-candle-perkernel.json).
        //
        // Phase 0 data has sample buffer overflow (103702 overflows), so us_per_token
        // totals are undersampled. However, us_per_call_median is reliable since it's
        // computed per observed dispatch. We use per-call values and multiply by the
        // known dispatch count per layer.
        //
        // Gemma4 26B Q4_K_M architecture per layer (decode, seq_len=1):
        //   - QKV: 3 quantized mat-vec (Q4_0 or Q6_K depending on weight quant)
        //     + 1 RMS norm before QKV
        //   - Head norms + RoPE: separate kernels in candle (not fused)
        //     ~3 norm dispatches + 2 RoPE dispatches + 1 V norm
        //   - KV cache: 2 copy dispatches (K, V)
        //   - SDPA: 1 dispatch (sdpa_vector_float_256 for sliding, _512 for global)
        //   - O-proj: 1 quantized mat-vec
        //   - MLP: 3 quantized mat-vec (gate, up, down) + 2 elementwise (gelu, mul)
        //     + 1 RMS norm before MLP
        //   - MoE: 2 _id mat-vec (gate_up, down) + routing overhead
        //     (norms, router proj, softmax, argsort, gather, mul, add)
        //   - Norms/adds: ~5 norm/add dispatches (post-attn, pre-FF-2, post-FF, etc.)
        //
        // Key candle per-call medians from Phase 0:
        //   Q4_0 mat-vec:    10.88 us    Q6_K mat-vec:    14.50 us
        //   Q8_0 mat-vec:    18.62 us    Q4_0 _id:        38.12 us
        //   Q6_K _id:        54.21 us    Q8_0 _id:        35.58 us
        //   SDPA-256:        14.96 us    SDPA-512:        25.75 us
        //   lm_head GEMM:  3483.29 us (F16 dense)
        //   RMS norm:        ~4 us       elementwise:     ~3 us
        //   copy2d:          ~2 us       affine:          ~2 us
        //
        // Per-layer estimates (assuming average Q4_0 mat-vec at ~11 us/call):
        //   QKV: 1 norm(4) + 3 matvec(11) = ~37 us
        //   Head norms + RoPE: ~6 dispatches * ~4 us = ~24 us
        //   KV cache: 2 * ~2 us = ~4 us
        //   SDPA: 15 us (sliding) or 26 us (global); avg = 25*15+5*26 / 30 = ~17 us
        //   O-proj: 1 matvec(11) = ~11 us
        //   MLP: 1 norm(4) + 3 matvec(11) + 2 elem(3) = ~43 us
        //   MoE: 2 _id(38) + 1 matvec(11) + ~10 dispatches * 4 us = ~127 us
        //   Norms/adds: ~5 dispatches * 4 us = ~20 us
        //
        // Total per layer: ~283 us. Over 30 layers = ~8490 us.
        // Plus lm_head: ~3483 us (from Phase 0 data, but measured at ~185 us/token
        // because it's called 0.05x/token during mixed prefill+decode).
        // For decode-only, lm_head = 1 call/token = ~3483 us is too high (that
        // includes queue overhead in the counter). Known candle decode = ~11000 us.
        //
        // Candle total decode: ~11000 us/token (from task description baseline).
        // Scale factor = 11000 / (8490 + 185) = ~1.27x (buffer overflow correction).
        // Apply scale to per-group estimates.

        let candle_qkv_per_layer = 37.0; // norm + 3 mat-vec
        let candle_norms_rope_per_layer = 24.0; // head norms + RoPE + V norm
        let candle_kv_cache_per_layer = 4.0; // 2 copy dispatches
        let candle_sdpa_per_layer = 17.0; // avg of 15 (sliding) and 26 (global)
        let candle_o_proj_per_layer = 11.0; // 1 mat-vec
        let candle_mlp_per_layer = 43.0; // norm + 3 mat-vec + 2 elementwise
        let candle_moe_per_layer = 127.0; // _id matmuls + routing overhead
        let candle_norms_adds_per_layer = 20.0; // post-layer norms/adds
        let candle_lm_head = 185.0; // 1 F16 GEMM call

        let candle_per_layer_total = candle_qkv_per_layer + candle_norms_rope_per_layer
            + candle_kv_cache_per_layer + candle_sdpa_per_layer + candle_o_proj_per_layer
            + candle_mlp_per_layer + candle_moe_per_layer + candle_norms_adds_per_layer;
        let candle_layers_total = candle_per_layer_total * num_layers as f64;
        let candle_total_reconstructed = candle_layers_total + candle_lm_head;

        eprintln!("\n=== PER-KERNEL-TYPE PROFILING (median over {n} tokens) ===");
        eprintln!("Per layer ({num_layers} layers):");
        eprintln!("  QKV matmuls (norm+3 proj):       {:7.0} us  [candle: ~{:.0} us]  ratio: {:.1}x",
            qkv_per_layer, candle_qkv_per_layer, qkv_per_layer / candle_qkv_per_layer);
        eprintln!("  Head norms + RoPE (3 dispatches): {:7.0} us  [candle: ~{:.0} us]  ratio: {:.1}x",
            norms_rope_per_layer, candle_norms_rope_per_layer, norms_rope_per_layer / candle_norms_rope_per_layer);
        eprintln!("  KV cache copy (2 dispatches):    {:7.0} us  [candle: ~{:.0} us]  ratio: {:.1}x",
            kv_cache_per_layer, candle_kv_cache_per_layer, kv_cache_per_layer / candle_kv_cache_per_layer);
        eprintln!("  SDPA (1 dispatch):               {:7.0} us  [candle: ~{:.0} us]  ratio: {:.1}x",
            sdpa_per_layer, candle_sdpa_per_layer, sdpa_per_layer / candle_sdpa_per_layer);
        eprintln!("  O-proj matmul (1 dispatch):      {:7.0} us  [candle: ~{:.0} us]  ratio: {:.1}x",
            o_proj_per_layer, candle_o_proj_per_layer, o_proj_per_layer / candle_o_proj_per_layer);
        eprintln!("  MLP matmuls (norm+3proj+gelu):   {:7.0} us  [candle: ~{:.0} us]  ratio: {:.1}x",
            mlp_per_layer, candle_mlp_per_layer, mlp_per_layer / candle_mlp_per_layer);
        eprintln!("  MoE (routing+4 expert):          {:7.0} us  [candle: ~{:.0} us]  ratio: {:.1}x",
            moe_per_layer, candle_moe_per_layer, moe_per_layer / candle_moe_per_layer);
        eprintln!("  Fused norms/adds (2 dispatches): {:7.0} us  [candle: ~{:.0} us]  ratio: {:.1}x",
            norms_adds_per_layer, candle_norms_adds_per_layer, norms_adds_per_layer / candle_norms_adds_per_layer);
        eprintln!();
        eprintln!("Head:");
        eprintln!("  lm_head GEMM (F16):              {:7.0} us  [candle: ~{:.0} us]  ratio: {:.1}x",
            head_total, candle_lm_head, head_total / candle_lm_head);
        eprintln!();
        eprintln!("Total GPU per token:               {:7.0} us  [candle: ~{:.0} us]  ratio: {:.1}x",
            gpu_total, candle_total_reconstructed, gpu_total / candle_total_reconstructed);
        eprintln!("  Layers total:                    {:7.0} us  [candle: ~{:.0} us]",
            gpu_total - head_total, candle_layers_total);
        eprintln!("  Head total:                      {:7.0} us  [candle: ~{:.0} us]",
            head_total, candle_lm_head);

        // Per-layer detail for sliding vs global
        eprintln!();
        eprintln!("Per-layer detail (median token, us):");
        eprintln!("  Layer | Type |    QKV | Nrm+RoPE |  KV$ |  SDPA | O-proj |    MLP |    MoE | Norms | Total");
        eprintln!("  ------|------|--------|----------|------|-------|--------|--------|--------|-------|------");
        let mid = profiles.len() / 2;
        let median_p = &profiles[mid]; // approximate median token
        for li in 0..num_layers {
            let lt = if (li + 1) % 6 == 0 { "G" } else { "S" };
            let layer_total = median_p.qkv_matmuls_us[li] + median_p.head_norms_rope_us[li]
                + median_p.kv_cache_copy_us[li] + median_p.sdpa_us[li]
                + median_p.o_proj_us[li] + median_p.mlp_matmuls_us[li]
                + median_p.moe_us[li] + median_p.norms_adds_us[li];
            eprintln!("  {:>2}    |  {}   | {:6.0} |    {:5.0} | {:4.0} | {:5.0} |  {:5.0} |  {:5.0} |  {:5.0} | {:5.0} | {:5.0}",
                li, lt,
                median_p.qkv_matmuls_us[li], median_p.head_norms_rope_us[li],
                median_p.kv_cache_copy_us[li], median_p.sdpa_us[li],
                median_p.o_proj_us[li], median_p.mlp_matmuls_us[li],
                median_p.moe_us[li], median_p.norms_adds_us[li],
                layer_total);
        }

        // Find top 3 slowest kernel types (by ratio vs candle)
        let mut ratios = vec![
            ("QKV matmuls", qkv_per_layer, candle_qkv_per_layer, qkv_per_layer / candle_qkv_per_layer),
            ("Head norms + RoPE", norms_rope_per_layer, candle_norms_rope_per_layer, norms_rope_per_layer / candle_norms_rope_per_layer),
            ("KV cache copy", kv_cache_per_layer, candle_kv_cache_per_layer, kv_cache_per_layer / candle_kv_cache_per_layer),
            ("SDPA", sdpa_per_layer, candle_sdpa_per_layer, sdpa_per_layer / candle_sdpa_per_layer),
            ("O-proj matmul", o_proj_per_layer, candle_o_proj_per_layer, o_proj_per_layer / candle_o_proj_per_layer),
            ("MLP matmuls", mlp_per_layer, candle_mlp_per_layer, mlp_per_layer / candle_mlp_per_layer),
            ("MoE", moe_per_layer, candle_moe_per_layer, moe_per_layer / candle_moe_per_layer),
            ("Fused norms/adds", norms_adds_per_layer, candle_norms_adds_per_layer, norms_adds_per_layer / candle_norms_adds_per_layer),
            ("lm_head GEMM", head_total, candle_lm_head, head_total / candle_lm_head),
        ];
        ratios.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
        eprintln!();
        eprintln!("TOP 3 SLOWEST (highest mlx-native/candle ratio):");
        for (i, (name, mlx_us, candle_us, ratio)) in ratios.iter().take(3).enumerate() {
            let overhead_per_token = (mlx_us - candle_us) * if *name != "lm_head GEMM" { num_layers as f64 } else { 1.0 };
            eprintln!("  {}. {} — {:.1}x slower ({:.0} vs {:.0} us/layer) — {:.0} us/token overhead",
                i + 1, name, ratio, mlx_us, candle_us, overhead_per_token);
        }

        eprintln!();
        eprintln!("NOTE: Per-session overhead (~30-50 us/session) inflates all groups.");
        eprintln!("      The ratio shows relative slowness, not absolute kernel time.");
        eprintln!("      {} sessions/token vs 1 in production mode.", 8 * num_layers + 2);
    }

}

/// Helper: load a GGUF tensor as raw quantized bytes into an MlxQWeight.
///
/// The tensor name is looked up in the GGUF file, its raw GGML block data
/// is copied into a new MlxBuffer, and the `QuantWeightInfo` is derived
/// from the tensor's metadata (shape and GGML type).
fn load_gguf_qweight(
    gguf: &mlx_native::gguf::GgufFile,
    name: &str,
    device: &MlxDevice,
) -> Result<MlxQWeight> {
    let full_name = if name.ends_with(".weight") {
        name.to_string()
    } else {
        format!("{name}.weight")
    };
    let info = gguf.tensor_info(&full_name)
        .ok_or_else(|| anyhow::anyhow!("tensor '{}' not found in GGUF", full_name))?;
    let buffer = gguf.load_tensor(&full_name, device)
        .map_err(|e| anyhow::anyhow!("load {}: {e}", full_name))?;

    // Shape: [rows, cols] for 2D weight matrices.
    let rows = info.shape.first().copied().unwrap_or(1);
    let cols = if info.shape.len() > 1 { info.shape[1] } else { 1 };

    Ok(MlxQWeight {
        buffer,
        info: QuantWeightInfo {
            ggml_dtype: info.ggml_type,
            rows,
            cols,
        },
    })
}

// ---------------------------------------------------------------------------
// Forward pass dispatch helpers
// ---------------------------------------------------------------------------

/// Buffers + shape for a single per-head RMS-norm dispatch.
///
/// Grouped out of `dispatch_rms_norm_unit_perhead` (was 8 positional
/// args). The I/O buffers and the `(rows, dim)` shape describe *what*
/// the kernel operates on; `encoder`/`registry`/`device` describe *where*
/// it runs. Separating the two groups makes call sites scannable.
pub struct RmsNormPerHeadArgs<'a> {
    /// F32 `[rows, dim]` input tensor.
    pub input: &'a MlxBuffer,
    /// F32 `[rows, dim]` output tensor (separate buffer; kernel does not
    /// support in-place).
    pub output: &'a MlxBuffer,
    /// Constant params buffer (`dim`, `eps`) — pre-populated at load time.
    pub params_buf: &'a MlxBuffer,
    /// Number of rows (per-layer: `num_kv_heads`, or `seq_len * num_kv_heads`
    /// in the batched prefill path).
    pub rows: u32,
    /// Per-row element count (per-layer `head_dim`).
    pub dim: u32,
}

/// Dispatch per-head RMS norm without learned scale (unit norm, f32).
///
/// Same as `dispatch_rms_norm_perhead` but uses `rms_norm_no_scale_f32`
/// (no weight buffer — just unit normalization).
pub fn dispatch_rms_norm_unit_perhead(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut mlx_native::KernelRegistry,
    device: &mlx_native::metal::DeviceRef,
    args: &RmsNormPerHeadArgs<'_>,
) -> Result<()> {
    let pipeline = registry.get_pipeline("rms_norm_no_scale_f32", device)
        .map_err(|e| anyhow::anyhow!("rms_norm_no_scale_f32 pipeline: {e}"))?;
    let tg_size = std::cmp::min(256, args.dim.next_power_of_two()) as u64;
    let shared_mem_bytes = tg_size * 4;
    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[(0, args.input), (1, args.output), (2, args.params_buf)],
        &[(0, shared_mem_bytes)],
        mlx_native::MTLSize::new(args.rows as u64, 1, 1),
        mlx_native::MTLSize::new(tg_size, 1, 1),
    );
    Ok(())
}

/// Dual-output variant: writes both f32 (for KV cache copy) AND bf16
/// (for the bf16 attention island) — ADR-011 Phase 3 Wave P3b-tensor.3.
///
/// Used by batched prefill V-norm to fuse the f32→bf16 cast that was
/// previously a separate `cast_f32_to_bf16` dispatch.  Same compute,
/// one extra device write per element (effectively free on Apple
/// unified memory since the f32 result is in registers).
#[allow(clippy::too_many_arguments)]
/// Fused per-head V-norm + permute (Wave P4.16).
///
/// Same compute as `dispatch_rms_norm_unit_perhead_dual` but writes the
/// bf16 output at the permuted [n_heads, seq_len, head_dim] layout
/// instead of the natural [seq_len, n_heads, head_dim] layout.  Saves
/// the post-norm `permute_021_bf16` dispatch on V (~30 dispatches/
/// prefill on Gemma 4) and ~10 MB of intermediate-buffer traffic at
/// pp2455.
///
/// The f32 output stays at natural layout — KV cache copy reads it.
pub fn dispatch_rms_norm_unit_perhead_dual_perm(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut mlx_native::KernelRegistry,
    device: &mlx_native::metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    output_bf16_perm: &MlxBuffer,
    params_buf: &MlxBuffer,
    n_heads: u32,
    seq_len: u32,
    dim: u32,
) -> Result<()> {
    use mlx_native::ops::encode_helpers::{encode_threadgroups_with_args_and_shared, KernelArg};
    let pipeline = registry.get_pipeline("rms_norm_no_scale_f32_dual_perm", device)
        .map_err(|e| anyhow::anyhow!("rms_norm_no_scale_f32_dual_perm pipeline: {e}"))?;
    let rows = (n_heads as u64) * (seq_len as u64);
    let tg_size = std::cmp::min(256, dim.next_power_of_two()) as u64;
    let shared_mem_bytes = tg_size * 4;
    let aux_bytes: [u32; 2] = [n_heads, seq_len];
    let aux_bytes_b: &[u8] = unsafe {
        std::slice::from_raw_parts(aux_bytes.as_ptr() as *const u8, std::mem::size_of_val(&aux_bytes))
    };
    encode_threadgroups_with_args_and_shared(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Buffer(params_buf)),
            (3, KernelArg::Buffer(output_bf16_perm)),
            (4, KernelArg::Bytes(aux_bytes_b)),
        ],
        &[(0, shared_mem_bytes)],
        mlx_native::MTLSize::new(rows, 1, 1),
        mlx_native::MTLSize::new(tg_size, 1, 1),
    );
    Ok(())
}

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
        mlp_down: alloc_f32(hs, "mlp_down")?,
        sdpa_out: alloc_f32(num_heads * max_hd, "sdpa_out")?,
        sdpa_tmp: {
            let tmp_bytes = mlx_native::ops::flash_attn_vec_tq::tmp_buffer_bytes(
                num_heads as u32, max_hd as u32);
            device.alloc_buffer(tmp_bytes, mlx_native::DType::F32,
                vec![tmp_bytes / 4])
                .map_err(|e| anyhow::anyhow!("sdpa_tmp alloc: {e}"))?
        },
        norm_params,
        position: alloc_u32(1, "position")?,
        softcap_params,
        argmax_index: alloc_u32(1, "argmax_index")?,
        argmax_value: alloc_f32(1, "argmax_value")?,
        argmax_params,
        logits: alloc_f32(vocab, "logits")?,
        moe_router_logits: alloc_f32(num_experts, "moe_router_logits")?,
        moe_expert_out: alloc_f32(hs.max(max_kv_heads * max_hd), "moe_expert_out")?,
        moe_accum: alloc_f32(hs, "moe_accum")?,
        moe_norm_out: alloc_f32(hs, "moe_norm_out")?,
        router_norm_out: alloc_f32(hs, "router_norm_out")?,
        // Fused _id dispatch buffers (sized for top_k = cfg.top_k_experts)
        moe_expert_ids: alloc_u32(cfg.top_k_experts, "moe_expert_ids")?,
        moe_gate_up_id_out: alloc_f32(cfg.top_k_experts * 2 * moe_interm, "moe_gate_up_id_out")?,
        moe_down_id_out: alloc_f32(cfg.top_k_experts * hs, "moe_down_id_out")?,
        moe_swiglu_id_out: alloc_f32(cfg.top_k_experts * moe_interm, "moe_swiglu_id_out")?,
        hidden_f16: device.alloc_buffer(hs * 2, mlx_native::DType::F16, vec![1, hs])
            .map_err(|e| anyhow::anyhow!("alloc hidden_f16 ({hs} f16): {e}"))?,
        logits_f16: device.alloc_buffer(vocab * 2, mlx_native::DType::F16, vec![1, vocab])
            .map_err(|e| anyhow::anyhow!("alloc logits_f16 ({vocab} f16): {e}"))?,
        // --- Session merge buffers (S1+S2 collapse) ---
        norm_params_sliding_hd: {
            let sliding_hd = cfg.head_dim;
            let mut buf = alloc_f32(2, "norm_params_sliding_hd")?;
            let p: &mut [f32] = buf.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("norm_params_sliding_hd init: {e}"))?;
            p[0] = cfg.rms_norm_eps as f32;
            p[1] = sliding_hd as f32;
            buf
        },
        norm_params_global_hd: {
            let global_hd = cfg.global_head_dim;
            let mut buf = alloc_f32(2, "norm_params_global_hd")?;
            let p: &mut [f32] = buf.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("norm_params_global_hd init: {e}"))?;
            p[0] = cfg.rms_norm_eps as f32;
            p[1] = global_hd as f32;
            buf
        },
        rope_freq_factors_gpu: alloc_f32(1, "rope_freq_factors_gpu_placeholder")?,
        attn_v: alloc_f32(max_kv_heads * max_hd, "attn_v")?,
        attn_q_normed: alloc_f32(num_heads * max_hd, "attn_q_normed")?,
        attn_k_normed: alloc_f32(max_kv_heads * max_hd, "attn_k_normed")?,
        moe_routing_weights_gpu: alloc_f32(cfg.top_k_experts, "moe_routing_weights_gpu")?,
    })
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
// DONE (Phase 5c — S3 merge):
// - [x] Merged S3 (router_proj) into S2:
//       - GPU post-FF norm 1 (mlp_down → attn_out) replaces CPU norm
//       - GPU pre-FF norm 2 (residual → moe_norm_out) replaces CPU norm
//       - GPU router input prep via rms_norm with pre-computed combined weight
//       - Router matmul now dispatched at end of S2
//       - Eliminates 30 sessions/token (S3 was 1 dispatch per session per layer)
//
// DONE (Phase 5d — S1+S2 merge):
// - [x] Merged S1 (QKV) into S2 (SDPA+MLP+router):
//       - GPU per-head RMS norm on Q and K (rms_norm rows=nh/nkv, dim=hd)
//       - GPU per-head unit RMS norm on V (rms_norm_no_scale_f32)
//       - GPU RoPE neox f32 with freq_factors on Q and K
//       - GPU KV cache copy f32 per-head (new kv_cache_copy_f32 kernel)
//       - V output separated from moe_expert_out (dedicated attn_v buffer)
//       - Eliminates CPU head norms, RoPE, V norm, KV cache update
//       - Total sessions: 2 per layer (merged, experts) + 1 head
//       - 30 layers × 2 + 1 = 61 sessions per token (down from 91)
//
// OPTIMIZATIONS deferred:
// - [ ] Merge S1+S2 with MoE (S4): needs GPU-side accumulate with GPU weights
// - [ ] GPU embedding via embedding_gather kernel
// - [ ] GPU argmax via dispatch_argmax_f32
