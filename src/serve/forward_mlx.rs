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
use mlx_native::ops::flash_attn_vec::FlashAttnVecParams;
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

/// Check if per-kernel-type profiling is enabled (HF2Q_MLX_KERNEL_PROFILE=1).
fn kernel_profile_enabled() -> bool {
    std::env::var("HF2Q_MLX_KERNEL_PROFILE").map_or(false, |v| v == "1")
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
    /// Temporary buffer for flash_attn_vec workgroup partial results.
    /// Sized for max(sliding, global) head config.
    pub flash_attn_tmp: MlxBuffer,
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
    /// NeoX RoPE params for sliding layers: `[theta, head_dim, rope_dim, 0]` F32.
    pub rope_params_sliding_neox: MlxBuffer,
    /// NeoX RoPE params for global layers: `[theta, head_dim, rope_dim, 0]` F32.
    pub rope_params_global_neox: MlxBuffer,
    /// GPU buffer holding global-layer freq_factors `[global_head_dim/2]` F32.
    pub rope_freq_factors_gpu: MlxBuffer,
    /// Dedicated V projection output buffer `[max_kv_heads * max_hd]` F32.
    /// Separates V from moe_expert_out to avoid aliasing in merged session.
    pub attn_v: MlxBuffer,
    /// Scratch buffer for Q after per-head norm `[num_heads * max_hd]` F32.
    pub attn_q_normed: MlxBuffer,
    /// Scratch buffer for K after per-head norm `[max_kv_heads * max_hd]` F32.
    pub attn_k_normed: MlxBuffer,
    /// MoE softmax params for GPU dispatch: `[num_experts, 0]` F32.
    pub moe_softmax_params_gpu: MlxBuffer,
    /// MoE scratch: pre-scaled routing weights for weighted_sum kernel `[top_k]` F32.
    pub moe_routing_weights_gpu: MlxBuffer,
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

            // Pre-compute router combined weight for GPU router input prep:
            //   router_combined_weight[i] = router_scale[i] * (hidden_size ^ -0.5)
            // This allows a single GPU rms_norm dispatch to replace the 3-step
            // CPU sequence (unit_norm + scale + mul).
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
                router_combined_weight,
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
            // Phase 4a: F16 KV cache halves memory bandwidth for SDPA reads.
            // Reference: llama.cpp stores KV cache in F16 for bandwidth-bound decode.
            let cache_elements = nkv * capacity * hd;
            let cache_bytes = cache_elements * 2; // F16 = 2 bytes per element
            let k_cache = mlx_device.alloc_buffer(
                cache_bytes, mlx_native::DType::F16, vec![nkv, capacity, hd],
            ).map_err(|e| anyhow::anyhow!("KV cache K alloc failed: {e}"))?;
            let v_cache = mlx_device.alloc_buffer(
                cache_bytes, mlx_native::DType::F16, vec![nkv, capacity, hd],
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
        let mut activations = alloc_activation_buffers(mlx_device, cfg)?;

        // Populate freq_factors GPU buffer from model weights.
        {
            let ff_host = model.rope_freqs_host();
            if !ff_host.is_empty() {
                let n = ff_host.len();
                let mut buf = mlx_device.alloc_buffer(
                    n * std::mem::size_of::<f32>(),
                    mlx_native::DType::F32,
                    vec![n],
                ).map_err(|e| anyhow::anyhow!("rope_freq_factors_gpu alloc: {e}"))?;
                let dst: &mut [f32] = buf.as_mut_slice()
                    .map_err(|e| anyhow::anyhow!("rope_freq_factors_gpu write: {e}"))?;
                dst[..n].copy_from_slice(ff_host);
                activations.rope_freq_factors_gpu = buf;
            }
        }

        // Dummy 1-element buffer for lm_head_f32 field — kept for compatibility.
        let lm_head_f32_dummy = mlx_device.alloc_buffer(4, mlx_native::DType::F32, vec![1])
            .map_err(|e| anyhow::anyhow!("lm_head dummy: {e}"))?;

        let mut result = Ok(Self {
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
        });

        // Pre-initialize constant param buffers so we never write them
        // inside the hot forward_decode path.  Writing to a shared Metal
        // buffer mid-session can force CPU-GPU synchronization.
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
            let use_graph_opt = std::env::var("HF2Q_GRAPH_OPT").map_or(false, |v| v == "1");
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
            let dual_buffer_split: Option<usize> = match std::env::var("HF2Q_DUAL_BUFFER") {
                Ok(v) => v.parse::<usize>().ok().filter(|&n| n > 0 && n < num_layers),
                Err(_) => Some(3), // default: split after layer 3
            };

            // --- 2. Transformer layers ---
            for layer_idx in 0..num_layers {
                let hd = self.head_dims[layer_idx];
                let nkv = self.num_kv_heads[layer_idx];
                let nh = self.num_attention_heads;
                let is_sliding = self.layer_types[layer_idx] == LayerType::Sliding;
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
                s.barrier_between(
                    &[&self.activations.norm_out, &self.layers[layer_idx].attn.q_proj.buffer],
                    &[&self.activations.attn_q],
                );
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    &self.layers[layer_idx].attn.q_proj, &mut self.activations.attn_q, 1)?;
                total_dispatches += 1;
                s.barrier_between(
                    &[&self.activations.norm_out, &self.layers[layer_idx].attn.k_proj.buffer],
                    &[&self.activations.attn_k],
                );
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    &self.layers[layer_idx].attn.k_proj, &mut self.activations.attn_k, 1)?;
                total_dispatches += 1;
                let v_is_k = self.layers[layer_idx].attn.v_proj.is_none();
                if !v_is_k {
                    s.barrier_between(
                        &[&self.activations.norm_out, &self.layers[layer_idx].attn.v_proj.as_ref().unwrap().buffer],
                        &[&self.activations.attn_v],
                    );
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                        self.layers[layer_idx].attn.v_proj.as_ref().unwrap(),
                        &mut self.activations.attn_v, 1)?;
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

                // Fused Q: attn_q → attn_q_normed
                s.barrier_between(
                    &[&self.activations.attn_q],
                    &[&self.activations.attn_q_normed],
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

                // Fused K: attn_k → attn_k_normed
                s.barrier_between(
                    &[&self.activations.attn_k],
                    &[&self.activations.attn_k_normed],
                );
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
                        &self.activations.attn_k,
                        &self.activations.attn_v,
                        hd_norm_params,
                        nkv as u32, hd as u32,
                    )?;
                    total_dispatches += 1;
                } else {
                    s.barrier_between(
                        &[&self.activations.attn_v],
                        &[&self.activations.moe_expert_out],
                    );
                    dispatch_rms_norm_unit_perhead(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.attn_v,
                        &self.activations.moe_expert_out,
                        hd_norm_params,
                        nkv as u32, hd as u32,
                    )?;
                    total_dispatches += 1;
                }

                let v_src = if v_is_k {
                    &self.activations.attn_v
                } else {
                    &self.activations.moe_expert_out
                };

                // -- GPU KV cache update (CONCURRENT: K and V copies are independent) --
                {
                    let cache_pos_val = if kv_is_sliding {
                        (kv_write_pos % kv_capacity) as u32
                    } else {
                        kv_write_pos as u32
                    };
                    s.barrier_between(
                        &[&self.activations.attn_k_normed],
                        &[&self.kv_caches[layer_idx].k],
                    );
                    mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.attn_k_normed,
                        &self.kv_caches[layer_idx].k,
                        nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                    ).map_err(|e| anyhow::anyhow!("kv K L{layer_idx}: {e}"))?;
                    total_dispatches += 1;
                    s.barrier_between(
                        &[v_src],
                        &[&self.kv_caches[layer_idx].v],
                    );
                    mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16(
                        s.encoder_mut(), reg, metal_dev,
                        v_src,
                        &self.kv_caches[layer_idx].v,
                        nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                    ).map_err(|e| anyhow::anyhow!("kv V L{layer_idx}: {e}"))?;
                    total_dispatches += 1;
                }

                // -- SDPA (flash_attn_vec) --
                s.barrier_between(
                    &[&self.activations.attn_q_normed, &self.kv_caches[layer_idx].k, &self.kv_caches[layer_idx].v],
                    &[&self.activations.sdpa_out, &self.activations.flash_attn_tmp],
                );
                {
                    let p = FlashAttnVecParams {
                        num_heads: nh as u32,
                        num_kv_heads: nkv as u32,
                        head_dim: hd as u32,
                        kv_seq_len: kv_seq_len as u32,
                        kv_capacity: kv_capacity as u32,
                        scale: 1.0,
                        mask_type: if is_sliding { 2 } else { 1 },
                        sliding_window: if is_sliding { self.sliding_window as u32 } else { 0 },
                        softcap: 0.0,
                    };
                    s.flash_attn_vec(reg, dev,
                        &self.activations.attn_q_normed, &self.kv_caches[layer_idx].k,
                        &self.kv_caches[layer_idx].v, &self.activations.sdpa_out,
                        &self.activations.flash_attn_tmp, &p,
                    ).map_err(|e| anyhow::anyhow!("flash_attn_vec L{layer_idx}: {e}"))?;
                }
                total_dispatches += 2;

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

                // -- B8: pre-FF norm1 + pre-FF norm2 + router norm [3 concurrent] --
                // All three read `residual`, write to disjoint buffers.
                s.barrier_between(
                    &[&self.activations.residual],
                    &[&self.activations.norm_out],
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

                s.barrier_between(
                    &[&self.activations.residual],
                    &[&self.activations.moe_norm_out],
                );
                s.rms_norm(
                    reg, metal_dev,
                    &self.activations.residual,
                    &self.layers[layer_idx].norms.pre_feedforward_layernorm_2,
                    &self.activations.moe_norm_out,
                    &self.activations.norm_params,
                    1, hs as u32,
                ).map_err(|e| anyhow::anyhow!("pre-FF norm 2 L{layer_idx}: {e}"))?;
                total_dispatches += 1;

                s.barrier_between(
                    &[&self.activations.residual],
                    &[&self.activations.router_norm_out],
                );
                s.rms_norm(
                    reg, metal_dev,
                    &self.activations.residual,
                    &self.layers[layer_idx].moe.router_combined_weight,
                    &self.activations.router_norm_out,
                    &self.activations.norm_params,
                    1, hs as u32,
                ).map_err(|e| anyhow::anyhow!("router norm L{layer_idx}: {e}"))?;
                total_dispatches += 1;

                // -- B9: dense gate + dense up + router logits [3 concurrent] --
                // gate/up read norm_out; router logits reads router_norm_out. All disjoint writes.
                s.barrier_between(
                    &[&self.activations.norm_out, &self.layers[layer_idx].mlp.gate_proj.buffer],
                    &[&self.activations.mlp_gate],
                );
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    &self.layers[layer_idx].mlp.gate_proj, &mut self.activations.mlp_gate, 1)?;
                total_dispatches += 1;

                s.barrier_between(
                    &[&self.activations.norm_out, &self.layers[layer_idx].mlp.up_proj.buffer],
                    &[&self.activations.mlp_up],
                );
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    &self.layers[layer_idx].mlp.up_proj, &mut self.activations.mlp_up, 1)?;
                total_dispatches += 1;

                s.barrier_between(
                    &[&self.activations.router_norm_out, &self.layers[layer_idx].moe.router_proj.buffer],
                    &[&self.activations.moe_router_logits],
                );
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.router_norm_out,
                    &self.layers[layer_idx].moe.router_proj,
                    &mut self.activations.moe_router_logits, 1)?;
                total_dispatches += 1;

                // -- B10: fused_gelu_mul + fused_moe_routing [2 concurrent] --
                s.barrier_between(
                    &[&self.activations.mlp_gate, &self.activations.mlp_up],
                    &[&self.activations.mlp_fused],
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

                s.barrier_between(
                    &[&self.activations.moe_router_logits],
                    &[&self.activations.moe_expert_ids, &self.activations.moe_routing_weights_gpu],
                );
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
                    let ggml_type_gu = gpu::candle_ggml_to_mlx(
                        self.layers[layer_idx].moe.gate_up_ggml_dtype)?;
                    let ggml_type_dn = gpu::candle_ggml_to_mlx(
                        self.layers[layer_idx].moe.down_ggml_dtype)?;

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

                    let ggml_type_gu = gpu::candle_ggml_to_mlx(
                        self.layers[layer_idx].moe.gate_up_ggml_dtype)?;
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
                    let ggml_type_dn = gpu::candle_ggml_to_mlx(
                        self.layers[layer_idx].moe.down_ggml_dtype)?;
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
                    total_dispatches += 1;

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

                // Dual command buffer split: commit buf0 after N layers, start buf1.
                // GPU starts executing buf0 immediately while CPU encodes buf1.
                // Metal FIFO queue ordering guarantees buf1 executes after buf0.
                if dual_buffer_split == Some(layer_idx + 1) {
                    let b0_barriers = s.barrier_count();
                    let b0_encoder = s.commit(); // commit buf0 → GPU starts executing
                    // Create fresh session for buf1
                    s = exec.begin().map_err(|e| anyhow::anyhow!("dual-buffer begin: {e}"))?;
                    // Seed tracker: hidden was written by the last layer in buf0
                    s.track_dispatch(&[], &[&self.activations.hidden]);
                    if std::env::var("HF2Q_MLX_TIMING").is_ok() {
                        eprintln!("  [DUAL_BUFFER] split at layer {} — buf0: {} dispatches, {} barriers",
                            layer_idx + 1, total_dispatches, b0_barriers);
                    }
                    // Store buf0 encoder so we can verify it completed (Metal FIFO
                    // guarantees buf1's wait_until_completed implies buf0 is done,
                    // but keeping the reference is defensive).
                    drop(b0_encoder);
                }
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

            // GPU lm_head: mixed-precision mat-vec (F32 input × F16 weights → F32 output)
            // Single dispatch replaces the old 3-dispatch path (cast + gemm + cast).
            if let Some(ref lm_head_f16) = self.lm_head_f16 {
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
            let barrier_count = s.barrier_count();
            let is_recording = s.is_recording();
            if is_recording {
                let (enc_ns, gpu_ns, fusions, reordered, b0, b1) =
                    s.finish_optimized_with_timing(reg, metal_dev, session_start)
                        .map_err(|e| anyhow::anyhow!("optimized session finish: {e}"))?;
                if std::env::var("HF2Q_MLX_TIMING").is_ok() {
                    eprintln!("  [TIMING] encode={:.2}ms gpu_wait={:.2}ms dispatches={} barriers={}",
                        enc_ns as f64 / 1e6, gpu_ns as f64 / 1e6, total_dispatches, barrier_count);
                    eprintln!("  [GRAPH_OPT] fusions={} reordered={} barriers={}+{}",
                        fusions, reordered, b0, b1);
                }
            } else {
                let (enc_ns, gpu_ns) = s.finish_with_timing(session_start)
                    .map_err(|e| anyhow::anyhow!("single session finish: {e}"))?;
                if std::env::var("HF2Q_MLX_TIMING").is_ok() {
                    eprintln!("  [TIMING] encode={:.2}ms gpu_wait={:.2}ms dispatches={} barriers={}",
                        enc_ns as f64 / 1e6, gpu_ns as f64 / 1e6, total_dispatches, barrier_count);
                }
            }
        }
        let session_us = session_start.elapsed().as_secs_f64() * 1e6;

        // Read the argmax result (8 bytes: 1 u32 index + 1 f32 value)
        let token_id: u32 = {
            let idx: &[u32] = self.activations.argmax_index.as_slice()
                .map_err(|e| anyhow::anyhow!("argmax read: {e}"))?;
            idx[0]
        };

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

        let mut kp = KernelTypeProfile::default();
        kp.qkv_matmuls_us = vec![0.0; num_layers];
        kp.head_norms_rope_us = vec![0.0; num_layers];
        kp.kv_cache_copy_us = vec![0.0; num_layers];
        kp.sdpa_us = vec![0.0; num_layers];
        kp.o_proj_us = vec![0.0; num_layers];
        kp.mlp_matmuls_us = vec![0.0; num_layers];
        kp.moe_us = vec![0.0; num_layers];
        kp.norms_adds_us = vec![0.0; num_layers];

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
            let t0 = Instant::now();
            let mut s = exec.begin().map_err(|e| anyhow::anyhow!("embed begin: {e}"))?;
            mlx_native::ops::elementwise::embedding_gather_scale_f32(
                s.encoder_mut(), reg, metal_dev,
                &self.embed_weight, &self.activations.hidden,
                input_token, hs, (hs as f32).sqrt(),
            ).map_err(|e| anyhow::anyhow!("embedding: {e}"))?;
            s.finish().map_err(|e| anyhow::anyhow!("embed finish: {e}"))?;
            let _ = t0.elapsed(); // embedding is trivial, don't report
        }

        // --- Per-layer kernel-type sessions ---
        for layer_idx in 0..num_layers {
            let hd = self.head_dims[layer_idx];
            let nkv = self.num_kv_heads[layer_idx];
            let nh = self.num_attention_heads;
            let is_sliding = self.layer_types[layer_idx] == LayerType::Sliding;
            let eps = self.rms_norm_eps;
            let (kv_is_sliding, kv_write_pos, kv_capacity, kv_seq_len) = kv_info[layer_idx];
            let v_is_k = self.layers[layer_idx].attn.v_proj.is_none();

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
                        &self.activations.attn_k, &self.activations.attn_v,
                        hd_norm_params, nkv as u32, hd as u32,
                    )?;
                } else {
                    dispatch_rms_norm_unit_perhead(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.attn_v, &self.activations.moe_expert_out,
                        hd_norm_params, nkv as u32, hd as u32,
                    )?;
                }

                let (_enc_ns, gpu_ns) = s.finish_with_timing(t0)
                    .map_err(|e| anyhow::anyhow!("norms finish L{layer_idx}: {e}"))?;
                kp.head_norms_rope_us[layer_idx] = gpu_ns as f64 / 1000.0;
            }

            let v_src = if v_is_k { &self.activations.attn_v } else { &self.activations.moe_expert_out };

            // ============================================================
            // GROUP 3: KV cache copy (2 dispatches)
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
                mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.attn_k_normed, &self.kv_caches[layer_idx].k,
                    nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                ).map_err(|e| anyhow::anyhow!("kv K L{layer_idx}: {e}"))?;
                mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32(
                    s.encoder_mut(), reg, metal_dev,
                    v_src, &self.kv_caches[layer_idx].v,
                    nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                ).map_err(|e| anyhow::anyhow!("kv V L{layer_idx}: {e}"))?;

                let (_enc_ns, gpu_ns) = s.finish_with_timing(t0)
                    .map_err(|e| anyhow::anyhow!("kv finish L{layer_idx}: {e}"))?;
                kp.kv_cache_copy_us[layer_idx] = gpu_ns as f64 / 1000.0;
            }

            // ============================================================
            // GROUP 4: SDPA (1 dispatch)
            // ============================================================
            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let metal_dev = dev.metal_device();
                let _ = metal_dev; // suppress unused warning for sliding path
                let t0 = Instant::now();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("sdpa begin L{layer_idx}: {e}"))?;

                {
                    let p = FlashAttnVecParams {
                        num_heads: nh as u32,
                        num_kv_heads: nkv as u32,
                        head_dim: hd as u32,
                        kv_seq_len: kv_seq_len as u32,
                        kv_capacity: kv_capacity as u32,
                        scale: 1.0,
                        mask_type: if is_sliding { 2 } else { 1 },
                        sliding_window: if is_sliding { self.sliding_window as u32 } else { 0 },
                        softcap: 0.0,
                    };
                    s.flash_attn_vec(reg, dev,
                        &self.activations.attn_q_normed, &self.kv_caches[layer_idx].k,
                        &self.kv_caches[layer_idx].v, &self.activations.sdpa_out,
                        &self.activations.flash_attn_tmp, &p,
                    ).map_err(|e| anyhow::anyhow!("flash_attn_vec L{layer_idx}: {e}"))?;
                }

                let (_enc_ns, gpu_ns) = s.finish_with_timing(t0)
                    .map_err(|e| anyhow::anyhow!("flash_attn_vec finish L{layer_idx}: {e}"))?;
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

                let ggml_type_gu = gpu::candle_ggml_to_mlx(
                    self.layers[layer_idx].moe.gate_up_ggml_dtype)?;
                let ggml_type_dn = gpu::candle_ggml_to_mlx(
                    self.layers[layer_idx].moe.down_ggml_dtype)?;

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
                let _ = dev; // suppress unused
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

    /// Run one decoder layer with session-collapsed GPU dispatch.
    ///
    /// Session structure (2 sessions per layer — S1+S2 merged):
    ///   Session 1: QKV matmuls → GPU per-head norms → GPU RoPE (neox f32
    ///              with freq_factors) → GPU V norm → GPU KV cache copy
    ///              → SDPA → O-proj → GPU-norm(post-attn) → GPU-add(residual)
    ///              → GPU-norm(pre-FF) → gate → up → GPU-GELU → GPU-mul → down
    ///              → GPU-norm(post-FF-1) → GPU-norm(pre-FF-2)
    ///              → GPU-norm(router) → router_proj
    ///   CPU: softmax + top-k on router logits (128 elements, microseconds)
    ///   Session 2 (was S4): zero(accum) + expert _id dispatches + accumulate
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

        // -- b. Update position buffer for this token --
        {
            let pos_dst: &mut [u32] = self.activations.position.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("position write: {e}"))?;
            pos_dst[0] = seq_pos as u32;
        }

        // -- c. KV cache bookkeeping (compute positions before GPU session) --
        let kv_is_sliding = self.kv_caches[layer_idx].is_sliding;
        let kv_write_pos = self.kv_caches[layer_idx].write_pos;
        let kv_capacity = self.kv_caches[layer_idx].capacity;
        self.kv_caches[layer_idx].write_pos += 1;
        self.kv_caches[layer_idx].seq_len = self.kv_caches[layer_idx].seq_len.saturating_add(1)
            .min(kv_capacity);
        let kv_seq_len = self.kv_caches[layer_idx].seq_len;

        // =====================================================================
        // MERGED SESSION 1: QKV + head norms + RoPE + KV cache + SDPA + MLP
        //                    + router prep + router_proj
        //
        // Phase 5d: S1+S2 merged into a single GPU session. All ops from
        // QKV projections through router proj execute in one command buffer.
        // Only one commit_and_wait per layer (down from 2 in Phase 5c).
        //
        // Head norms:    GPU rms_norm per-head on Q, K
        // V norm:        GPU rms_norm_no_scale per-head (unit norm) on V
        // RoPE:          GPU rope_neox_f32 with freq_factors support
        // KV cache:      GPU kv_cache_copy_f32 per-head (strided cache layout)
        // =====================================================================
        let s1_start = Instant::now();
        let mut s1_dispatches = 0usize;
        {
            let (exec, reg) = gpu.split();
            let dev = exec.device();
            let metal_dev = dev.metal_device();
            let mut s = exec.begin().map_err(|e| anyhow::anyhow!("S1 begin: {e}"))?;

            // -- QKV projections --
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
                    &mut self.activations.attn_v, 1)?;
                s1_dispatches += 1;
            }

            // -- GPU per-head RMS norm on Q: attn_q → attn_q_normed --
            // Uses perhead helper to bypass element_count validation (buffer
            // is allocated for max head_dim but actual dim may be smaller).
            let hd_norm_params = if is_sliding {
                &self.activations.norm_params_sliding_hd
            } else {
                &self.activations.norm_params_global_hd
            };
            dispatch_rms_norm_perhead(
                s.encoder_mut(), reg, metal_dev,
                &self.activations.attn_q,
                &self.layers[layer_idx].attn.q_norm_weight,
                &self.activations.attn_q_normed,
                hd_norm_params,
                nh as u32, hd as u32,
            )?;
            s1_dispatches += 1;

            // -- GPU per-head RMS norm on K: attn_k → attn_k_normed --
            dispatch_rms_norm_perhead(
                s.encoder_mut(), reg, metal_dev,
                &self.activations.attn_k,
                &self.layers[layer_idx].attn.k_norm_weight,
                &self.activations.attn_k_normed,
                hd_norm_params,
                nkv as u32, hd as u32,
            )?;
            s1_dispatches += 1;

            // -- GPU V: copy from K if k_eq_v, then per-head unit norm --
            if v_is_k {
                // With k_eq_v: V_proj == K_proj, so attn_k holds the raw
                // matmul output. K was normed into attn_k_normed (not in-place),
                // so attn_k still has the raw values. Dispatch unit norm on
                // attn_k → attn_v.
                dispatch_rms_norm_unit_perhead(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.attn_k,
                    &self.activations.attn_v,
                    hd_norm_params,
                    nkv as u32, hd as u32,
                )?;
                s1_dispatches += 1;
            } else {
                // V was projected into attn_v; apply unit norm per head.
                // Output to moe_expert_out (scratch) since we can't do in-place.
                dispatch_rms_norm_unit_perhead(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.attn_v,
                    &self.activations.moe_expert_out,
                    hd_norm_params,
                    nkv as u32, hd as u32,
                )?;
                s1_dispatches += 1;
            }

            // The V source for KV cache: attn_v if k_eq_v (unit norm wrote there),
            // moe_expert_out if not k_eq_v (unit norm wrote there).
            let v_src = if v_is_k {
                &self.activations.attn_v
            } else {
                &self.activations.moe_expert_out
            };

            // -- GPU RoPE (neox f32 with freq_factors) on Q and K --
            let rope_params = if is_sliding {
                &self.activations.rope_params_sliding_neox
            } else {
                &self.activations.rope_params_global_neox
            };
            let ff_gpu = if is_sliding {
                None
            } else {
                Some(&self.activations.rope_freq_factors_gpu)
            };

            // RoPE on Q: attn_q_normed → attn_q (reuse for in-session output)
            dispatch_rope_neox_f32_unchecked(
                s.encoder_mut(), reg, metal_dev,
                &self.activations.attn_q_normed,
                &self.activations.attn_q,
                rope_params,
                &self.activations.position,
                ff_gpu,
                1, // seq_len = 1 for decode
                nh as u32,
                hd as u32,
                hd as u32, // rope_dim = head_dim (full rotation)
            )?;
            s1_dispatches += 1;

            // RoPE on K: attn_k_normed → attn_k (reuse for in-session output)
            dispatch_rope_neox_f32_unchecked(
                s.encoder_mut(), reg, metal_dev,
                &self.activations.attn_k_normed,
                &self.activations.attn_k,
                rope_params,
                &self.activations.position,
                ff_gpu,
                1, // seq_len = 1 for decode
                nkv as u32,
                hd as u32,
                hd as u32, // rope_dim = head_dim (full rotation)
            )?;
            s1_dispatches += 1;

            // -- GPU KV cache update (batched — 2 dispatches for all heads) --
            // Cache layout: [num_kv_heads, capacity, head_dim] (head-major).
            // K/V source layout: [num_kv_heads * head_dim] (flat, 1 token).
            // Single dispatch per K and V copies all heads at once.
            {
                let cache_pos_val = if kv_is_sliding {
                    (kv_write_pos % kv_capacity) as u32
                } else {
                    kv_write_pos as u32
                };

                // K cache copy (all heads, 1 dispatch)
                mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.attn_k,
                    &self.kv_caches[layer_idx].k,
                    nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                ).map_err(|e| anyhow::anyhow!("kv_cache_copy_batch K: {e}"))?;
                s1_dispatches += 1;

                // V cache copy (all heads, 1 dispatch)
                mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32(
                    s.encoder_mut(), reg, metal_dev,
                    v_src,
                    &self.kv_caches[layer_idx].v,
                    nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                ).map_err(|e| anyhow::anyhow!("kv_cache_copy_batch V: {e}"))?;
                s1_dispatches += 1;
            }

            // -- SDPA (flash_attn_vec, was start of old S2) --
            {
                let p = FlashAttnVecParams {
                    num_heads: nh as u32,
                    num_kv_heads: nkv as u32,
                    head_dim: hd as u32,
                    kv_seq_len: kv_seq_len as u32,
                    kv_capacity: kv_capacity as u32,
                    scale: 1.0,
                    mask_type: if is_sliding { 2 } else { 1 },
                    sliding_window: if is_sliding { self.sliding_window as u32 } else { 0 },
                    softcap: 0.0,
                };
                s.flash_attn_vec(reg, dev,
                    &self.activations.attn_q, &self.kv_caches[layer_idx].k,
                    &self.kv_caches[layer_idx].v, &self.activations.sdpa_out,
                    &self.activations.flash_attn_tmp, &p,
                ).map_err(|e| anyhow::anyhow!("flash_attn_vec: {e}"))?;
            }
            s1_dispatches += 2; // flash_attn_vec main + reduce

            // O-proj: sdpa_out → attn_out
            dispatch_qmatmul(&mut s, reg, dev, &self.activations.sdpa_out,
                &self.layers[layer_idx].attn.o_proj, &mut self.activations.attn_out, 1)?;
            s1_dispatches += 1; // O-proj

            // GPU post-attention norm: attn_out → norm_out
            s.rms_norm(
                reg, metal_dev,
                &self.activations.attn_out,
                &self.layers[layer_idx].norms.post_attention_layernorm,
                &self.activations.norm_out,
                &self.activations.norm_params,
                1, hs as u32,
            ).map_err(|e| anyhow::anyhow!("GPU post-attn norm: {e}"))?;
            s1_dispatches += 1; // rms_norm

            // GPU residual add: hidden + norm_out → residual
            s.elementwise_add(
                reg, metal_dev,
                &self.activations.hidden, &self.activations.norm_out,
                &self.activations.residual, hs, mlx_native::DType::F32,
            ).map_err(|e| anyhow::anyhow!("GPU residual add: {e}"))?;
            s1_dispatches += 1; // add

            // GPU pre-feedforward norm: residual → norm_out
            s.rms_norm(
                reg, metal_dev,
                &self.activations.residual,
                &self.layers[layer_idx].norms.pre_feedforward_layernorm,
                &self.activations.norm_out,
                &self.activations.norm_params,
                1, hs as u32,
            ).map_err(|e| anyhow::anyhow!("GPU pre-FF norm: {e}"))?;
            s1_dispatches += 1; // rms_norm

            // Dense MLP: gate and up projections (norm_out → mlp_gate, mlp_up)
            dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                &self.layers[layer_idx].mlp.gate_proj, &mut self.activations.mlp_gate, 1)?;
            s1_dispatches += 1; // gate
            dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                &self.layers[layer_idx].mlp.up_proj, &mut self.activations.mlp_up, 1)?;
            s1_dispatches += 1; // up

            // Fused SwiGLU: GELU(mlp_gate) * mlp_up → mlp_fused (1 dispatch)
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
            s1_dispatches += 1; // fused gelu*mul

            // down_proj: mlp_fused → mlp_down
            dispatch_qmatmul(&mut s, reg, dev, &self.activations.mlp_fused,
                &self.layers[layer_idx].mlp.down_proj, &mut self.activations.mlp_down, 1)?;
            s1_dispatches += 1; // down

            // ---- Merged S3 ops (router prep + router proj) ----
            // GPU post-feedforward norm 1: mlp_down → attn_out (reuse as scratch)
            s.rms_norm(
                reg, metal_dev,
                &self.activations.mlp_down,
                &self.layers[layer_idx].norms.post_feedforward_layernorm_1,
                &self.activations.attn_out,  // output to attn_out (scratch)
                &self.activations.norm_params,
                1, hs as u32,
            ).map_err(|e| anyhow::anyhow!("GPU post-FF norm 1: {e}"))?;
            s1_dispatches += 1; // post-FF norm 1

            // GPU pre-feedforward norm 2: residual → moe_norm_out (MoE expert input)
            s.rms_norm(
                reg, metal_dev,
                &self.activations.residual,
                &self.layers[layer_idx].norms.pre_feedforward_layernorm_2,
                &self.activations.moe_norm_out,
                &self.activations.norm_params,
                1, hs as u32,
            ).map_err(|e| anyhow::anyhow!("GPU pre-FF norm 2: {e}"))?;
            s1_dispatches += 1; // pre-FF norm 2

            // GPU router input prep: unit_norm(residual) * router_combined_weight → norm_out
            s.rms_norm(
                reg, metal_dev,
                &self.activations.residual,
                &self.layers[layer_idx].moe.router_combined_weight,
                &self.activations.norm_out,
                &self.activations.norm_params,
                1, hs as u32,
            ).map_err(|e| anyhow::anyhow!("GPU router input prep: {e}"))?;
            s1_dispatches += 1; // router norm

            // Router projection: norm_out → moe_router_logits
            dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                &self.layers[layer_idx].moe.router_proj,
                &mut self.activations.moe_router_logits, 1)?;
            s1_dispatches += 1; // router proj

            s.finish().map_err(|e| anyhow::anyhow!("S1 finish: {e}"))?;
        }
        let s1_elapsed = s1_start.elapsed();
        if let Some(ref mut p) = profile {
            // S1 now contains everything that was S1+CPU1+S2 (merged session).
            // Report S1 timing for the merged session, zero out S2/CPU1.
            p.layer_s1_us[layer_idx] = s1_elapsed.as_secs_f64() * 1e6;
            p.s1_dispatches[layer_idx] = s1_dispatches;
            p.layer_cpu1_us[layer_idx] = 0.0; // CPU1 eliminated (GPU norms+RoPE+KV)
            p.layer_s2_us[layer_idx] = 0.0;   // S2 merged into S1
            p.s2_dispatches[layer_idx] = 0;
        }

        // -- CPU: softmax + top-k on router logits --
        // Post-FF norm 1 result is now in attn_out (GPU wrote it in S1).
        // MoE norm input is now in moe_norm_out (GPU wrote it in S1).
        // Router logits are now in moe_router_logits (GPU wrote it in S1).
        let cpu2_start = Instant::now();

        // -- MoE path (session 2 — the only remaining session break per layer) --
        let (s4_us, s4_disp) =
            self.forward_decode_moe_no_router(layer_idx, gpu)?;

        // -- Combine MLP + MoE, final norm, residual, layer scalar (CPU) --
        // Normed mlp_down is now in attn_out (GPU wrote post-FF norm 1 there in S1).
        {
            let mlp: &[f32] = self.activations.attn_out.as_slice().map_err(|e| anyhow::anyhow!("m: {e}"))?;
            let moe: &[f32] = self.activations.moe_accum.as_slice().map_err(|e| anyhow::anyhow!("mo: {e}"))?;
            let combined: &mut [f32] = self.activations.mlp_down.as_mut_slice().map_err(|e| anyhow::anyhow!("c: {e}"))?;
            for i in 0..hs { combined[i] = mlp[i] + moe[i]; }
        }
        cpu_rms_norm_weighted(
            &self.activations.mlp_down, &self.layers[layer_idx].norms.post_feedforward_layernorm,
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
            // S1+S2 merged, S3 merged previously. Only S4 (MoE experts) remains.
            let total_cpu2_4 = cpu2_and_4_elapsed.as_secs_f64() * 1e6;
            let cpu4_us = total_cpu2_4 - s4_us;
            p.layer_cpu2_us[layer_idx] = cpu4_us.max(0.0);
            p.layer_s3_us[layer_idx] = 0.0;  // S3 merged into S1
            p.layer_cpu3_us[layer_idx] = 0.0;
            p.layer_s4_us[layer_idx] = s4_us;
            p.layer_cpu4_us[layer_idx] = cpu4_us.max(0.0);
            p.s3_dispatches[layer_idx] = 0;
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

            // Write pre-scaled routing weights to GPU buffer for weighted_sum
            {
                let w_dst: &mut [f32] = self.activations.moe_routing_weights_gpu.as_mut_slice()
                    .map_err(|e| anyhow::anyhow!("moe_routing_weights write: {e}"))?;
                for k_idx in 0..top_k {
                    let eid = top_k_indices[k_idx] as usize;
                    w_dst[k_idx] = top_k_weights[k_idx] * per_expert_scale[eid];
                }
            }

            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let metal_dev = dev.metal_device();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("S4-moe begin: {e}"))?;

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

                // 2) Batched SwiGLU for all expert slots (1 dispatch).
                mlx_native::ops::moe_dispatch::moe_swiglu_batch_encode(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.moe_gate_up_id_out,
                    &self.activations.moe_swiglu_id_out,
                    moe_int, top_k,
                ).map_err(|e| anyhow::anyhow!("moe swiglu batch: {e}"))?;
                s4_dispatches += 1;

                // 3) down _id: input=[top_k, moe_int] ids=[top_k] -> output=[top_k, hs]
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

                // 4) Weighted sum of all expert outputs (1 dispatch, replaces zero+accumulate).
                mlx_native::ops::moe_dispatch::moe_weighted_sum_encode(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.moe_down_id_out,
                    &self.activations.moe_routing_weights_gpu,
                    &self.activations.moe_accum,
                    hs, top_k,
                ).map_err(|e| anyhow::anyhow!("moe weighted_sum: {e}"))?;
                s4_dispatches += 1;

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

    /// MoE forward pass WITHOUT router projection (S3 merged into S2).
    ///
    /// Preconditions (set by S2):
    /// - `moe_router_logits` contains router logits (from GPU router proj in S2)
    /// - `moe_norm_out` contains pre-FF-norm-2 of residual (from GPU norm in S2)
    ///
    /// Session structure (1 session):
    ///   CPU: softmax + top-k on router logits
    ///   Session 4: zero_buffer(accum) + expert _id dispatches + accumulate
    ///   CPU: post-FF-norm2
    ///
    /// Returns (s4_us, s4_dispatches).
    fn forward_decode_moe_no_router(
        &mut self,
        layer_idx: usize,
        gpu: &mut GpuContext,
    ) -> Result<(f64, usize)> {
        let hs = self.hidden_size;
        let top_k = self.layers[layer_idx].moe.top_k;
        let moe_int = self.layers[layer_idx].moe.moe_intermediate_size;

        // CPU: softmax + top-k on router logits (already in moe_router_logits from S2)
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
        let top_k_indices: Vec<u32> = indices[..top_k].iter().map(|&i| i as u32).collect();
        let top_probs: Vec<f32> = indices[..top_k].iter().map(|&i| probs[i]).collect();

        let top_sum: f32 = top_probs.iter().sum();
        let top_k_weights: Vec<f32> = top_probs.iter().map(|p| p / top_sum).collect();

        let per_expert_scale: Vec<f32> = {
            let s: &[f32] = self.layers[layer_idx].moe.per_expert_scale.as_slice()
                .map_err(|e| anyhow::anyhow!("per_expert_scale: {e}"))?;
            s.to_vec()
        };

        // Zero the accumulator on CPU (fast for 2816 elements)
        {
            let accum: &mut [f32] = self.activations.moe_accum.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("moe accum zero: {e}"))?;
            accum[..hs].fill(0.0);
        }

        // =====================================================================
        // SESSION 4: All expert dispatches in ONE session.
        // =====================================================================
        let s4_start = Instant::now();
        let mut s4_dispatches = 0usize;

        let use_fused_id = self.layers[layer_idx].moe.stacked_gate_up.is_some()
            && self.layers[layer_idx].moe.stacked_down.is_some();

        if use_fused_id {
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

            // Write pre-scaled routing weights to GPU buffer for weighted_sum
            {
                let w_dst: &mut [f32] = self.activations.moe_routing_weights_gpu.as_mut_slice()
                    .map_err(|e| anyhow::anyhow!("moe_routing_weights write: {e}"))?;
                for k_idx in 0..top_k {
                    let eid = top_k_indices[k_idx] as usize;
                    w_dst[k_idx] = top_k_weights[k_idx] * per_expert_scale[eid];
                }
            }

            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let metal_dev = dev.metal_device();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("S4-moe begin: {e}"))?;

                // 1) gate_up _id
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

                // 2) Batched SwiGLU (1 dispatch)
                mlx_native::ops::moe_dispatch::moe_swiglu_batch_encode(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.moe_gate_up_id_out,
                    &self.activations.moe_swiglu_id_out,
                    moe_int, top_k,
                ).map_err(|e| anyhow::anyhow!("moe swiglu batch: {e}"))?;
                s4_dispatches += 1;

                // 3) down _id
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

                // 4) Weighted sum (1 dispatch, replaces zero+accumulate)
                mlx_native::ops::moe_dispatch::moe_weighted_sum_encode(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.moe_down_id_out,
                    &self.activations.moe_routing_weights_gpu,
                    &self.activations.moe_accum,
                    hs, top_k,
                ).map_err(|e| anyhow::anyhow!("moe weighted_sum: {e}"))?;
                s4_dispatches += 1;

                s.finish().map_err(|e| anyhow::anyhow!("S4-moe finish: {e}"))?;
            }
        } else {
            // --- Fallback: per-expert loop (no _id kernels) ---
            // Note: fallback path still uses per-expert loop (not batched)
            // since the individual expert weight buffers are separate.
            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let metal_dev = dev.metal_device();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("S4-moe begin: {e}"))?;

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

                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.moe_norm_out,
                        &self.layers[layer_idx].moe.expert_gate_up[eid],
                        &mut self.activations.moe_gate_up_out, 1)?;
                    s4_dispatches += 1;

                    mlx_native::ops::moe_dispatch::moe_swiglu_fused_encode(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.moe_gate_up_out,
                        &self.activations.mlp_fused,
                        moe_int,
                    ).map_err(|e| anyhow::anyhow!("moe swiglu: {e}"))?;
                    s4_dispatches += 1;

                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.mlp_fused,
                        &self.layers[layer_idx].moe.expert_down[eid],
                        &mut self.activations.moe_expert_out, 1)?;
                    s4_dispatches += 1;

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

        Ok((s4_us, s4_dispatches))
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

/// Dispatch per-head RMS norm (f32) without element count validation.
///
/// The standard `dispatch_rms_norm` validates that input.element_count() ==
/// rows * dim, which fails when using over-sized buffers (e.g. Q buffer
/// allocated for max head_dim but used with smaller sliding head_dim).
///
/// This helper dispatches the kernel directly, processing only the first
/// `rows * dim` elements.  Safe because: (1) the buffer is at least as
/// large as rows*dim, and (2) the kernel reads/writes exactly rows*dim
/// elements controlled by its grid size (one threadgroup per row).
fn dispatch_rms_norm_perhead(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut mlx_native::KernelRegistry,
    device: &mlx_native::metal::DeviceRef,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    rows: u32,
    dim: u32,
) -> Result<()> {
    let pipeline = registry.get_pipeline("rms_norm_f32", device)
        .map_err(|e| anyhow::anyhow!("rms_norm_f32 pipeline: {e}"))?;
    let tg_size = std::cmp::min(256, dim.next_power_of_two()) as u64;
    let shared_mem_bytes = tg_size * 4;
    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[(0, input), (1, weight), (2, output), (3, params_buf)],
        &[(0, shared_mem_bytes)],
        mlx_native::MTLSize::new(rows as u64, 1, 1),
        mlx_native::MTLSize::new(tg_size, 1, 1),
    );
    Ok(())
}

/// Dispatch per-head RMS norm without learned scale (unit norm, f32).
///
/// Same as `dispatch_rms_norm_perhead` but uses `rms_norm_no_scale_f32`
/// (no weight buffer — just unit normalization).
fn dispatch_rms_norm_unit_perhead(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut mlx_native::KernelRegistry,
    device: &mlx_native::metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    rows: u32,
    dim: u32,
) -> Result<()> {
    let pipeline = registry.get_pipeline("rms_norm_no_scale_f32", device)
        .map_err(|e| anyhow::anyhow!("rms_norm_no_scale_f32 pipeline: {e}"))?;
    let tg_size = std::cmp::min(256, dim.next_power_of_two()) as u64;
    let shared_mem_bytes = tg_size * 4;
    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[(0, input), (1, output), (2, params_buf)],
        &[(0, shared_mem_bytes)],
        mlx_native::MTLSize::new(rows as u64, 1, 1),
        mlx_native::MTLSize::new(tg_size, 1, 1),
    );
    Ok(())
}

/// Dispatch NeoX RoPE f32 with freq_factors, bypassing element count validation.
///
/// Same as `dispatch_rope_neox_f32` but tolerates over-sized buffers.
/// Safe because the kernel grid is sized to (rope_dim/2, seq_len*n_heads)
/// and only accesses the first seq_len*n_heads*head_dim elements.
#[allow(clippy::too_many_arguments)]
fn dispatch_rope_neox_f32_unchecked(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut mlx_native::KernelRegistry,
    device: &mlx_native::metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    positions_buf: &MlxBuffer,
    freq_factors: Option<&MlxBuffer>,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    rope_dim: u32,
) -> Result<()> {
    use mlx_native::ops::encode_helpers::{encode_with_args, KernelArg};

    let pipeline = registry.get_pipeline("rope_neox_f32", device)
        .map_err(|e| anyhow::anyhow!("rope_neox_f32 pipeline: {e}"))?;
    let half_rope = rope_dim / 2;
    let n_rows = (seq_len as usize) * (n_heads as usize);

    let has_ff: u32 = if freq_factors.is_some() { 1 } else { 0 };
    // Pack [n_heads, has_freq_factors] as 2 x u32 = 8 bytes
    let mut param_bytes = [0u8; 8];
    param_bytes[0..4].copy_from_slice(&n_heads.to_ne_bytes());
    param_bytes[4..8].copy_from_slice(&has_ff.to_ne_bytes());

    let ff_buf = freq_factors.unwrap_or(input);

    let tg_x = std::cmp::min(64, half_rope as u64);
    let tg_y = std::cmp::min(4, n_rows as u64);

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Buffer(params_buf)),
            (3, KernelArg::Buffer(positions_buf)),
            (4, KernelArg::Bytes(&param_bytes)),
            (5, KernelArg::Buffer(ff_buf)),
        ],
        mlx_native::MTLSize::new(half_rope as u64, n_rows as u64, 1),
        mlx_native::MTLSize::new(tg_x, tg_y, 1),
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
        flash_attn_tmp: {
            // Sized for the largest head config (global: 16 heads * dk512).
            let tmp_bytes = mlx_native::ops::flash_attn_vec::tmp_buffer_bytes(
                num_heads as u32, max_hd as u32,
            );
            let tmp_elems = tmp_bytes / std::mem::size_of::<f32>();
            alloc_f32(tmp_elems, "flash_attn_tmp")?
        },
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
        rope_params_sliding_neox: {
            let sliding_hd = cfg.head_dim;
            let mut buf = alloc_f32(4, "rope_params_sliding_neox")?;
            let p: &mut [f32] = buf.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("rope_params_sliding_neox init: {e}"))?;
            p[0] = cfg.rope_theta_sliding as f32;
            p[1] = sliding_hd as f32;
            p[2] = sliding_hd as f32; // rope_dim = head_dim (full rotation)
            p[3] = 0.0;
            buf
        },
        rope_params_global_neox: {
            let global_hd = cfg.global_head_dim;
            let mut buf = alloc_f32(4, "rope_params_global_neox")?;
            let p: &mut [f32] = buf.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("rope_params_global_neox init: {e}"))?;
            p[0] = cfg.rope_theta_global as f32;
            p[1] = global_hd as f32;
            p[2] = global_hd as f32; // rope_dim = head_dim (full rotation)
            p[3] = 0.0;
            buf
        },
        rope_freq_factors_gpu: alloc_f32(1, "rope_freq_factors_gpu_placeholder")?,
        attn_v: alloc_f32(max_kv_heads * max_hd, "attn_v")?,
        attn_q_normed: alloc_f32(num_heads * max_hd, "attn_q_normed")?,
        attn_k_normed: alloc_f32(max_kv_heads * max_hd, "attn_k_normed")?,
        moe_softmax_params_gpu: {
            let mut buf = alloc_f32(2, "moe_softmax_params_gpu")?;
            let p: &mut [f32] = buf.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("moe_softmax_params_gpu init: {e}"))?;
            p[0] = num_experts as f32; // cols
            p[1] = 0.0;
            buf
        },
        moe_routing_weights_gpu: alloc_f32(cfg.top_k_experts, "moe_routing_weights_gpu")?,
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
