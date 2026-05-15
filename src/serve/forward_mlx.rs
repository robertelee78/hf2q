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

// iter-222 (ADR-005 closure, 2026-05-01): the iter-34 `dense_sdpa_on_tq_kv_enabled`
// helper + `HF2Q_LEGACY_TQ_SDPA` / `HF2Q_FORCE_DENSE_SDPA_ON_TQ_KV` env vars +
// the dense-on-shadow Leg F decode branch + the `leg_f_kvs` shadow cache field
// were deleted entirely. Iter34 routed TQ-regime SDPA through the dense
// `flash_attn_vec` kernel on a TQ→F32 shadow cache to lock in iter33's
// +11.97pp single-regime perf gain, but iter-222's bisect proved that path
// breaks Gate H (TQ-active two-regime decode quality envelope, ADR-007
// §853-866) — the encode→F32-shadow→decode round-trip introduced
// quantization noise the inline-fused `flash_attn_vec_tq` / `flash_attn_vec_tq_hb`
// kernels do not have. Worker R's TurboQuant peer-impl research (TheTom
// llama.cpp Phase 4b, animehacker CUDA, ollama mverrilli, sharpner-MLX V2,
// vivekvar-dl turbokv) found every shipping production engine uses inline-fused
// dequant as the default; the dequant-then-dense path is universally treated
// as an ablation. Per the user's mantra ("Fallback is basically a swear word
// to me — it's giving up"; "claiming we do TQ but falling back to not TQ ==
// bullshit") the iter-34 path was a fallback in the mantra's sense and is
// removed. The inline-fused TQ-native kernels (`flash_attn_vec_tq` for
// `HF2Q_TQ_CODEBOOK_BITS=4`, `flash_attn_vec_tq_hb` for the default 5/6/8-bit
// HB path) are now the SOLE TQ production path.

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

        // Dispatch counts.
        //
        // ADR-028 iter-90 BUG FIX: prior code used `getter(t).iter().sum()`
        // which double-counted because each `s*_dispatches[layer_idx]` is
        // assigned `total_dispatches` (the CUMULATIVE counter at end of
        // that layer), not the per-layer delta. Sum-of-cumulatives across
        // 30 layers reports ~15× the real per-token dispatch count
        // (e.g. 15310 reported vs 990 actual on gemma-4-26b decode at
        // HEAD `06a8eb3`). Per-token dispatch count is whatever the LAST
        // layer captured. `head_dispatches` is also a cumulative total
        // (assigned `total_dispatches` after the head ops at line 4288),
        // so the FINAL total per token == `head_dispatches`. We therefore
        // report body == s1_dispatches[last_layer], head == head_dispatches
        // - s1_dispatches[last_layer], total == head_dispatches.
        let last_layer_dispatch_avg = |getter: &dyn Fn(&TokenProfile) -> &Vec<usize>| -> f64 {
            let total: usize = measured.iter()
                .map(|t| getter(t).last().copied().unwrap_or(0))
                .sum();
            total as f64 / n as f64
        };
        let s1_disp = last_layer_dispatch_avg(&|t| &t.s1_dispatches);
        let s2_disp = last_layer_dispatch_avg(&|t| &t.s2_dispatches);
        let s3_disp = last_layer_dispatch_avg(&|t| &t.s3_dispatches);
        let s4_disp = last_layer_dispatch_avg(&|t| &t.s4_dispatches);
        let total_token_disp: f64 = measured.iter()
            .map(|t| t.head_dispatches as f64).sum::<f64>() / n as f64;
        // Head-only count is the delta between final cumulative and the body cumulative.
        let body_cum = s1_disp + s2_disp + s3_disp + s4_disp;
        let head_disp = (total_token_disp - body_cum).max(0.0);
        let total_disp = total_token_disp;

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

/// ADR-020 AC#5 Iter B — extra metadata + buffers for an mlx-affine
/// (DWQ) packed-U32 weight.  When this is `Some`, the parent
/// [`MlxQWeight`]'s `buffer` is interpreted as packed-U32 affine-quant
/// codes (shape `[N, K/pack_factor]`, dtype `U32`) instead of GGML
/// block-format bytes; `info.ggml_dtype` is unused on the affine path.
///
/// `scales` and `biases` are the per-group `(s, b)` pairs from the
/// DWQ-trained safetensors.  Held as F32 buffers (cast at load time
/// from BF16/F32 depending on the on-disk safetensors dtype) so the
/// kernel can read them without an inline cast.
pub struct MlxAffineExtra {
    /// Per-group scales, F32, shape `[N, K/group_size]`.
    pub scales: MlxBuffer,
    /// Per-group biases (zero-points), F32, shape `[N, K/group_size]`.
    pub biases: MlxBuffer,
    /// Quantization bit-width (currently only 4 supported).
    pub bits: u32,
    /// Per-group axis length (currently only 32 supported via `simd4_b4`).
    pub group_size: u32,
}

/// Pre-loaded quantized weight buffer paired with its GGML metadata
/// (or its mlx-affine metadata, when `affine` is `Some`).
pub struct MlxQWeight {
    pub buffer: MlxBuffer,
    pub info: QuantWeightInfo,
    /// `Some(...)` when this weight was loaded from a DWQ-trained mlx
    /// safetensors overlay.  `None` for the default GGML-block-loaded
    /// path.  Routing in `dispatch_qmatmul` checks `affine.is_some()`
    /// FIRST and skips both the F32 and GGML branches when set.
    pub affine: Option<MlxAffineExtra>,
    /// ADR-029 iter-28 H29 — F16 pre-dequantized shadow.  When `Some`,
    /// `dispatch_qmatmul` at m > MM_ROUTING_THRESHOLD routes through
    /// `kernel_mul_mm_f16_f32_*` (peer's gemma4 pattern) instead of
    /// per-call dequant inside `kernel_mul_mm_<qtype>_tensor_f32`.
    ///
    /// Materialized at load via `dispatch_dequant_to_f16` when the
    /// `HF2Q_F16_SHADOW=1` env gate is set and the weight is a quantized
    /// type the dequant kernel supports (Q4_0/Q8_0/Q5_1/IQ4_NL/Q4_K/
    /// Q5_K/Q6_K).  ~1 GB extra resident on gemma4-26B; M5 Max's 128 GB
    /// unified memory accommodates this without pressure.
    ///
    /// Default OFF until coherence + multi-regime bench parity proven.
    pub f16_shadow: Option<MlxBuffer>,
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

    /// AC#5 Iter B — construct an affine-mode `MlxQWeight` from a
    /// loaded `MlxAffineLinear` (the safetensors loader's runtime
    /// representation).  Uploads the packed-U32 weight + F32 scales +
    /// F32 biases to GPU buffers.
    ///
    /// The `info.ggml_dtype` is stamped to `GgmlType::F32` as a sentinel
    /// (unused on the affine path; routing gates on `affine.is_some()`
    /// before the F32 check).  `info.rows = N`, `info.cols = K`.
    pub fn from_mlx_affine_linear(
        device: &MlxDevice,
        linear: &crate::calibrate::mlx_safetensors_loader::MlxAffineLinear,
    ) -> Result<Self> {
        if linear.bits != 4 {
            anyhow::bail!(
                "MlxQWeight::from_mlx_affine_linear: only bits=4 supported in AC#5 Iter B; got {}",
                linear.bits
            );
        }
        if linear.group_size != 32 {
            anyhow::bail!(
                "MlxQWeight::from_mlx_affine_linear: only group_size=32 supported in AC#5 Iter B; got {}",
                linear.group_size
            );
        }
        let n = linear.n;
        let k = linear.k;
        let pack_factor = (32 / linear.bits) as usize;
        if k % pack_factor != 0 {
            anyhow::bail!(
                "MlxQWeight::from_mlx_affine_linear: K ({k}) must be divisible by pack_factor ({pack_factor})"
            );
        }
        let k_packed = k / pack_factor;
        let groups_per_row = k / linear.group_size;

        // Pack the unpacked u8 codes back to U32 mlx-on-disk layout
        // (low nibble at slot 0).  Mirrors mlx/ops.cpp:4762-4772.
        let mut packed = vec![0u32; n * k_packed];
        for row in 0..n {
            for kp in 0..k_packed {
                let mut word: u32 = 0;
                for j in 0..pack_factor {
                    let code = linear.q_int[row * k + kp * pack_factor + j] as u32;
                    debug_assert!(code <= 0xF);
                    word |= (code & 0xF) << (j * 4);
                }
                packed[row * k_packed + kp] = word;
            }
        }

        let mut weight_buf = device
            .alloc_buffer(
                n * k_packed * std::mem::size_of::<u32>(),
                mlx_native::DType::U32,
                vec![n, k_packed],
            )
            .map_err(|e| anyhow::anyhow!("affine weight alloc: {e}"))?;
        weight_buf
            .as_mut_slice::<u32>()
            .map_err(|e| anyhow::anyhow!("affine weight slice: {e}"))?
            .copy_from_slice(&packed);

        let mut scales_buf = device
            .alloc_buffer(
                n * groups_per_row * std::mem::size_of::<f32>(),
                mlx_native::DType::F32,
                vec![n, groups_per_row],
            )
            .map_err(|e| anyhow::anyhow!("affine scales alloc: {e}"))?;
        scales_buf
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("affine scales slice: {e}"))?
            .copy_from_slice(&linear.scales);

        let mut biases_buf = device
            .alloc_buffer(
                n * groups_per_row * std::mem::size_of::<f32>(),
                mlx_native::DType::F32,
                vec![n, groups_per_row],
            )
            .map_err(|e| anyhow::anyhow!("affine biases alloc: {e}"))?;
        biases_buf
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("affine biases slice: {e}"))?
            .copy_from_slice(&linear.biases);

        Ok(Self {
            buffer: weight_buf,
            info: QuantWeightInfo {
                ggml_dtype: mlx_native::GgmlType::F32, // sentinel; affine path bypasses it
                rows: n,
                cols: k,
            },
            affine: Some(MlxAffineExtra {
                scales: scales_buf,
                biases: biases_buf,
                bits: linear.bits,
                group_size: linear.group_size as u32,
            }),
            f16_shadow: None,
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

/// 1-element placeholder allocator helper.
///
/// Wedge-4 / iter-227: produces a tiny MlxBuffer that dense-FFN layers can
/// stash into the `MlxMoeWeights` slot without paying for the GBs of MoE
/// expert tensors that do not exist on disk. Dtype + shape are arbitrary
/// (the dense forward path never reads them); F32 with shape `[1]` is the
/// cheapest valid combination.
fn alloc_one_f32_placeholder(
    mlx_device: &mlx_native::MlxDevice,
    label: &'static str,
) -> Result<MlxBuffer> {
    mlx_device
        .alloc_buffer(
            std::mem::size_of::<f32>(),
            mlx_native::DType::F32,
            vec![1],
        )
        .map_err(|e| anyhow::anyhow!("dense MoE placeholder alloc ({label}): {e}"))
}

/// ADR-020 AC#5 Iter C2.2/C2.3 — stacked mlx-affine MoE expert weights
/// for one (role, layer) tuple.  Each buffer holds the full expert
/// stack in row-major order: `weight[e, n, k_packed]` U32,
/// `scales[e, n, k/group_size]` BF16, `biases[e, n, k/group_size]` BF16.
///
/// Consumed by the MoE-id dispatch path (Iter C2.3) via
/// `mlx_native::quantized_matmul_id_into` (same packed-U32 kernel
/// that mlx-lm uses).  BF16 scales/biases are the kernel's native
/// dtype — F32 input from `MlxAffineLinear::scales`/`biases` is cast
/// at upload time inside `MlxAffineMoeStack::from_per_expert_linears`.
#[derive(Clone)]
pub struct MlxAffineMoeStack {
    /// Packed-U32 weight stack `[n_experts, N, K/pack_factor]`.
    pub weight: MlxBuffer,
    /// BF16 scales stack `[n_experts, N, K/group_size]`.
    pub scales: MlxBuffer,
    /// BF16 biases stack `[n_experts, N, K/group_size]`.
    pub biases: MlxBuffer,
    /// Output dim per expert.
    pub n: usize,
    /// Input dim per expert.
    pub k: usize,
    /// Quant bit-width (4 in Iter C2.x).
    pub bits: u32,
    /// Per-group axis length (32 in Iter C2.x).
    pub group_size: u32,
    /// Number of experts.
    pub num_experts: usize,
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
    /// ADR-020 AC#5 Iter C2.2 — optional DWQ-overlay-applied affine
    /// stacks, replacing `stacked_gate_up` + `stacked_down` for the
    /// qwen35moe MoE dispatch path (Iter C2.3 wires the routing).
    /// `gate_up_affine` covers the FUSED gate+up case (qwen3.5 GGUF
    /// `ffn_gate_up_exps`); `gate_affine` + `up_affine` cover the
    /// SEPARATE case (uncommon — added for completeness, not yet
    /// produced by hf2q dwq-train).
    pub gate_up_affine: Option<MlxAffineMoeStack>,
    pub down_affine: Option<MlxAffineMoeStack>,
}

impl MlxMoeWeights {
    /// Construct a placeholder MoE bundle for **dense** layers.
    ///
    /// Wedge-4 / iter-227 (2026-05-02): dense GGUFs (e.g.
    /// `qwen3-vl-2b-q4_0.gguf` from Wedge-4f convert) carry zero MoE
    /// expert tensors. To keep the per-layer struct (`MlxDecoderLayerWeights`)
    /// uniform across dense + MoE layers without rippling
    /// `Vec<Option<MlxMoeWeights>>` through the forward path, we expose
    /// this constructor: it returns a bundle with `stacked_gate_up: None`
    /// and `stacked_down: None` plus 1-element placeholder buffers for
    /// every required field. The dense forward dispatch
    /// (`MlxModelWeights::forward_decode` / `forward_prefill`) consumes
    /// `MlxMlpWeights`, never the MoE bundle, so these placeholders are
    /// inert. The fused-id MoE dispatch already gates on
    /// `stacked_gate_up.is_some() && stacked_down.is_some()` (see
    /// `forward_decode` lines ~2863 / ~3922), so a misrouted MoE call
    /// against a dense-placeholder layer would falsify the `is_some()`
    /// gate at runtime rather than silently consuming garbage.
    ///
    /// Allocation cost is ~16 bytes per layer (vs. GBs of real expert
    /// tensors), and `top_k` / `moe_intermediate_size` are zeroed so
    /// any accidental read of them is also visibly wrong.
    pub fn dense_placeholder(mlx_device: &mlx_native::MlxDevice) -> Result<Self> {
        let router_proj_buf = alloc_one_f32_placeholder(mlx_device, "router_proj_buf")?;
        Ok(MlxMoeWeights {
            stacked_gate_up: None,
            stacked_down: None,
            gate_up_expert_stride: 0,
            down_expert_stride: 0,
            router_proj: MlxQWeight {
                buffer: router_proj_buf,
                info: QuantWeightInfo {
                    ggml_dtype: mlx_native::GgmlType::F32,
                    rows: 1,
                    cols: 1,
                },
                affine: None,
                f16_shadow: None,
            },
            per_expert_scale: alloc_one_f32_placeholder(mlx_device, "per_expert_scale")?,
            gate_up_ggml_dtype: mlx_native::GgmlType::F32,
            down_ggml_dtype: mlx_native::GgmlType::F32,
            top_k: 0,
            moe_intermediate_size: 0,
            router_combined_weight: alloc_one_f32_placeholder(mlx_device, "router_combined_weight")?,
            gate_up_affine: None,
            down_affine: None,
        })
    }
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

impl MlxKvCache {
    /// ADR-028 iter-229 / ADR-028 §iter-227 work item D: ds4-style counter
    /// rollback for speculative decode infrastructure.
    ///
    /// Logically discards the most-recent `n_back` positions from the cache.
    /// Following ds4's pattern (`DS4_MTP_KEEP_ACCEPTED` macro, ds4.c:16246),
    /// no actual cache bytes are cleared — `seq_len` is decremented to make
    /// the trailing positions invisible to subsequent SDPA reads.  Future
    /// writes (via `write_pos`) will overwrite those slots naturally.
    ///
    /// Returns the new `seq_len` for caller assertion.
    ///
    /// # Semantics
    /// - **Linear cache** (`is_sliding=false`): `write_pos == seq_len`, both
    ///   decrement by `n_back`.  Subsequent writes resume at the new
    ///   `write_pos`.
    /// - **Sliding cache** (`is_sliding=true`): more complex — `write_pos`
    ///   wraps modulo `capacity`.  Implemented for the linear case here;
    ///   sliding-aware rollback requires position-tracking metadata and is
    ///   deferred until iter-227 work item C (SD state machine).
    ///
    /// # Errors
    /// Returns `Err` if `n_back > seq_len` or sliding cache (not yet supported).
    pub fn trim(&mut self, n_back: usize) -> Result<usize, &'static str> {
        if self.is_sliding {
            // Sliding cache rollback requires logical-position tracking
            // (the slot index ≠ logical position when wrapped).  For ds4-style
            // SD on gemma4, sliding layers may need a counter parallel to
            // `write_pos` that tracks logical end-position separately.
            // Deferred to iter-227 work item C.
            return Err("trim() not yet supported on sliding cache");
        }
        if n_back > self.seq_len {
            return Err("trim n_back exceeds seq_len");
        }
        // Linear cache: write_pos == seq_len. Both decrement.
        self.seq_len -= n_back;
        self.write_pos = self.seq_len;
        Ok(self.seq_len)
    }

    /// Returns the count of valid (visible) positions.  Equivalent to
    /// `seq_len` post-iter-229 but exposed as named API for the SD
    /// state machine (matches ds4's `s->graph.mtp_n_raw` semantic).
    #[inline]
    pub fn visible_len(&self) -> usize {
        self.seq_len
    }
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
    /// Optional Q6_K-native lm_head (gated on HF2Q_LMHEAD_Q6K=1 at load).
    /// ADR-028 iter-188: gemma4 ships token_embd.weight as Q6_K [2816, 262144]
    /// = 605 MB; current Q8_0 re-quant path stores 784 MB.  Loading the
    /// on-disk Q6_K storage directly saves ~0.33 ms/token in lm_head
    /// (= ~2% gemma4 throughput).  Embedding lookup at input still uses
    /// the F32 `embed_weight`, so this is purely additive at load.
    /// Preferred over `lm_head_q8` when both are present.
    pub lm_head_q6k: Option<MlxQWeight>,
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
    /// ADR-017 Phase E.a iter-2.5 (Strategy A): per-layer Arc-wrapped
    /// owned KV buffers. The Arc tier is structural — at iter-2.5 the
    /// worker thread is still the sole holder of every Arc (strong
    /// count == 1 for every entry), so `Arc::get_mut` always succeeds
    /// at the kv-restore mutation site (engine.rs ~3479). Iter-3 will
    /// hand out Arc-clones to the LcpRegistry; at that point the
    /// mutation discipline tightens (registry-cloned Arcs become
    /// read-only via Arc::deref auto-coercion, and any in-place rewrite
    /// must consume + re-store the Arc to bring strong_count back to 1).
    ///
    /// **Read-path consumers UNCHANGED.** `dense_kvs[i].k`,
    /// `dense_kvs[i].v`, `dense_kvs[i].capacity`, `dense_kvs[i].is_sliding`
    /// all auto-deref through `Arc::deref` so existing forward_mlx.rs
    /// reader sites at lines 2432-2588 + 2794 compile without per-site
    /// edits. Field-access syntax `(&Arc<T>).k` resolves through the
    /// auto-deref chain `&Arc<T>` → `&T` → `&T.k` at zero cost.
    ///
    /// See dossier `docs/research/adr017-phase-e-option-a-2026-05-05.md`
    /// §10.3 Strategy A for the full rationale (~25 LOC additive in
    /// this file vs Strategy B's outer-Arc shape, which conflates
    /// per-layer eviction with whole-Vec rebuilds).
    pub dense_kvs: Option<Vec<std::sync::Arc<DenseKvBuffers>>>,
    /// ADR-017 Phase E.a iter-3.5b — end-of-prefill snapshot for the
    /// LcpRegistry. Populated by `forward_prefill_with_soft_tokens_resume`
    /// AT THE END of the per-token prefill loop (after all prompt
    /// positions written, BEFORE the function returns). Consumed by
    /// `engine.rs::generate_*` at the post-decode LCP store site, then
    /// cleared. `None` when the iter-3 env-gates (`HF2Q_KV_LCP_RESUME=1`
    /// + `HF2Q_USE_DENSE=1`) are off (no snapshot needed; LCP path
    /// inactive).
    ///
    /// **Why a separate field, not a return value:** changing
    /// `forward_prefill_with_soft_tokens_resume`'s return type from
    /// `Result<u32>` to `Result<(u32, Option<Vec<Arc<DenseKvBuffers>>>)>`
    /// would touch every call site (warmup, generate, embed_last,
    /// generate_stream_once, ...). A side-channel field on
    /// `MlxModelWeights` minimizes the surface — only the post-decode
    /// LCP store site reads it.
    ///
    /// **Why a snapshot, not a live-Arc clone:** decode mutates
    /// `dense_kvs[*][slot=p%capacity]` for sliding layers. The
    /// LcpRegistry must hold a SNAPSHOT taken at end-of-prefill
    /// (decode hasn't run yet) so future LCP hits read pure
    /// prompt-prefix state, not decode-corrupted ring slots. Lifts
    /// the iter-3 v1 wrap-guard restriction (which previously
    /// skipped store when `prompt_len + decode_writes > sliding_window`)
    /// at the cost of one extra per-layer KV allocation + memcpy per
    /// resume-eligible request (~50ms on Gemma 4 26B).
    pub dense_kvs_snapshot_for_lcp: Option<Vec<std::sync::Arc<DenseKvBuffers>>>,
    /// Tmp buffer for flash_attn_vec when using dense decode.
    pub dense_sdpa_tmp: Option<MlxBuffer>,
    // iter-20 Leg F `leg_f_kvs` + `leg_f_sdpa_tmp` shadow-cache fields deleted
    // iter-222 (2026-05-01) along with the iter-34 dense-on-shadow Leg F decode
    // branch and `dense_sdpa_on_tq_kv_enabled()` helper. See the file-level
    // iter-222 closure note above the deleted helper site for the rationale
    // (Gate H regression + peer-impl research + "no fallback" mantra). The
    // inline-fused TQ-native kernels (`flash_attn_vec_tq` / `flash_attn_vec_tq_hb`)
    // read directly from the TQ-packed `kv_caches[layer].{k,v}_packed` and
    // `leg_hb_encoded` buffers respectively — no F32 shadow cache required.

    /// iter-21 Track B: byte-packed higher-bit (5/6/8-bit) KV encoded cache.
    ///
    /// When `HF2Q_TQ_CODEBOOK_BITS=5|6|8` (default 8), K/V are encoded to
    /// byte-packed 5/6/8-bit Lloyd-Max indices via `hadamard_quantize_kv_hb`,
    /// stored here, and consumed inline by the `flash_attn_vec_tq_hb` kernel
    /// (no shadow-cache dequant round-trip).
    ///
    /// Layout: `[nkv_heads, capacity, head_dim]` U8 (1 byte per element).
    /// Norms: same layout as 4-bit caches (D=256: 1 norm/pos, D=512: 2/pos).
    pub leg_hb_encoded: Option<Vec<HbKvBuffers>>,
    /// ADR-028 Phase 10 (iter-347): hybrid K storage, F16 K + TQ-HB-packed V.
    ///
    /// Mutually exclusive with `leg_hb_encoded` at allocation time — exactly
    /// one of the two is `Some(…)` for any given model instance, governed by
    /// the `HF2Q_HYBRID_KV` env-gate (parsed in `investigation_env.rs`,
    /// default OFF until Phase 10f parity + 10g coherence gates pass).
    ///
    /// Why an `Option` field rather than a wrapping enum: the existing
    /// SDPA-dispatch site (`forward_decode`, ~line 3567) keys on the variant
    /// of `leg_hb_encoded` today; making the hybrid path additive (a sibling
    /// `Option` checked first) keeps the legacy TQ-HB path bit-identical when
    /// the gate is OFF (regression-safety mantra).
    #[allow(dead_code)] // Read in Phase 10c K-encode skip + 10e SDPA dispatcher (next iters).
    pub hybrid_kv: Option<Vec<HybridKvBuffers>>,
    /// Per-instance decode-step counter for the Gate H stderr emit lines.
    ///
    /// Increments on every successful `forward_decode`.  The audit-binary
    /// contract (`iter25_audit.rs::parse_nll_values`) sorts by `step=`,
    /// so a monotonic 0-based counter is the right shape.  Reset to 0
    /// at construction; [`MlxModelWeights::set_decode_regime`] also resets
    /// it between regimes for Gate H two-regime-one-process runs.
    pub decode_step: u64,
    /// ADR-007 Gate H per-call regime override (W12 iter-108a blocker #3).
    ///
    /// Default value [`DecodeRegime::Default`] preserves today's env-var-only
    /// path bit-exactly — the SDPA-mode gate reads `HF2Q_USE_DENSE` and
    /// `HF2Q_LAYER_POLICY` exactly as it does on the iter-108a base
    /// commit.  Set via [`MlxModelWeights::set_decode_regime`] to flip
    /// between TQ-active and dense-active SDPA within a single process
    /// (Gate H two-regime run); the setter also resets [`Self::decode_step`]
    /// so each regime's stderr `[HF2Q_NLL]` / `[HF2Q_DECODE_EMIT]` lines
    /// start at `step=0`.
    pub decode_regime: DecodeRegime,
    /// Cached startup-time flag: true iff none of the iter-108a Gate H
    /// runtime hooks are active. When true, the decode hot path skips
    /// per-token NLL emit, decode-emit, decode-replay, the `decode_step`
    /// counter mutation, AND the per-layer `decode_regime` enum match —
    /// keeping pre-iter-108a per-token cost bit-for-bit (W14b 5.6%
    /// regression: 95.0 → 100.6 tok/s baseline, 2026-04-25).
    ///
    /// Computed at construction from `INVESTIGATION_ENV` (a `LazyLock`
    /// that is populated exactly once per process via `from_env`) and
    /// the `decode_regime` field.  Re-evaluated only inside
    /// [`MlxModelWeights::set_decode_regime`] (since a non-Default
    /// regime requires the per-layer SDPA-gate match path to run).
    /// Never read or written on the per-token hot path beyond the
    /// initial single load — LLVM hoists the bool check above the
    /// per-layer loop and the per-token tail, so the entire
    /// instrumentation block becomes dead code under `if !gate_h_inactive`.
    pub gate_h_inactive: bool,
    /// ADR-007 Gate H per-instance replay-token override (W21 iter-108b).
    ///
    /// When non-empty, takes precedence over
    /// [`InvestigationEnv::decode_input_tokens`] in the post-argmax tail
    /// of [`Self::forward_decode`].  This is the in-process replay surface
    /// used by `cmd_parity_check_tq_quality` / `cmd_parity_capture_tq_quality`:
    /// pass 1 (dense) records the picked tokens directly via
    /// `forward_decode`'s return value, then pass 2 (TQ) sets this field
    /// before the decode loop so each TQ-step's logits are scored against
    /// the same token sequence dense produced (the ADR-007 §853-866 PPL
    /// input shape).  See [`Self::set_replay_tokens`].
    ///
    /// `LazyLock` makes [`InvestigationEnv::decode_input_tokens`] frozen
    /// at first access, so the env-var path can't switch mid-process —
    /// hence the per-instance override.  Empty by default; emit/NLL/decode-
    /// step bookkeeping in `forward_decode` continues to gate on
    /// [`Self::gate_h_inactive`], which is set to `false` by
    /// [`Self::set_replay_tokens`] whenever the replay vector is non-empty.
    ///
    /// (Wired by iter-108b's `parity_quality::run_two_regime_decode`;
    /// no in-tree caller as of iter-108a.)
    pub replay_tokens: Vec<u32>,
    /// ADR-007 Gate H per-instance dump-config override (W39 iter-112b).
    ///
    /// `INVESTIGATION_ENV.dump_dir` and `.dump_all_cache` are populated from
    /// process-start env via a `LazyLock` triggered in `main.rs::main` *before*
    /// `Cli::parse`, so the env vars W21's `parity_quality::run_two_regime_decode`
    /// sets at run time (after the LazyLock has frozen) never reach the SDPA
    /// dump gate at `forward_decode` lines 1268-1271 nor the dump-path
    /// formatter inside `dumps::dump_f32`.  W39 splits the two readers so the
    /// in-process Gate H harness can supply per-instance values instead:
    ///   - `Some(dir)` overrides `INVESTIGATION_ENV.dump_dir` for SDPA-out
    ///     dump file paths (consulted by the call site in `forward_decode`,
    ///     which routes through a path that picks up the override).
    ///   - `Some(true)` forces `dump_all_cache=true` for SDPA-out gating.
    ///   - `None` falls back to `INVESTIGATION_ENV` (i.e. the default
    ///     env-var-only path is byte-identical to pre-iter-112b).
    ///
    /// `set_dump_overrides` exposes the setter and is the only mutator.
    /// Like `set_replay_tokens`, the gate-H instrumentation flag isn't
    /// affected — these knobs are purely diagnostic plumbing for the
    /// SDPA-out file gate, not for the per-token NLL/replay block.
    ///
    /// (Wired by iter-112b's `parity_quality::run_two_regime_decode`; no
    /// other in-tree caller.)
    pub dump_dir_override: Option<std::path::PathBuf>,
    pub dump_all_cache_override: Option<bool>,
    /// ADR-007 Gate H per-instance decode-step dump counter (W39 iter-112b).
    ///
    /// Replaces the process-static `AtomicUsize` previously declared inside
    /// `forward_decode` at lines 1262-1267.  That static accumulated across
    /// the dense and TQ passes of the same Gate H run — pass 1 left it at
    /// `tokens`, so pass 2's `decode_step_for_dump < max_pos` was false at
    /// every step.  Per-instance + reset-between-passes restores the
    /// per-pass `[0, max_pos)` window.
    ///
    /// Reset to 0 by `set_decode_regime` and `set_replay_tokens` (matching
    /// `decode_step` semantics) and by the explicit
    /// `reset_decode_step_dump_counter` for the rare caller that wants to
    /// reset without touching regime / replay state.
    pub decode_step_dump_counter: usize,

    /// ADR-030 Phase 4 — optional DFlash spec-decode hidden-state
    /// capture session. When `Some`, `forward_prefill_batched`
    /// populates `dflash_capture.hidden_output` at indices matching
    /// `dflash_capture.target_layer_ids` during the layer loop.
    /// Default `None` preserves byte-identical legacy behavior — no
    /// production-path caller installs this; only the spec-decode
    /// orchestrator's `install_dflash_capture`/`take_dflash_capture`
    /// pair touches it.
    pub dflash_capture: Option<crate::inference::spec_decode::dflash::hidden_capture::DFlashCaptureSession>,
}

// ADR-031 Phase B foundation — compile-time Send+Sync assertion.
//
// Phase B needs to share `&MlxModelWeights` across a main thread and a
// worker thread during parallel-encode (HF2Q_PARALLEL_ENCODE=1).  That
// requires Self: Sync.  This assertion fails the build at this site if a
// future field violates the contract, surfacing the regression long
// before runtime.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MlxModelWeights>();
};

/// Per-layer byte-packed higher-bit (5/6-bit) KV buffers (iter-21 Track B).
pub struct HbKvBuffers {
    /// Byte-packed K indices `[nkv_heads, capacity, head_dim]` U8.
    pub k_packed: MlxBuffer,
    /// K per-position norms (same layout as 4-bit: D=256 → 1/pos, D=512 → 2/pos).
    pub k_norms: MlxBuffer,
    /// Byte-packed V indices `[nkv_heads, capacity, head_dim]` U8.
    pub v_packed: MlxBuffer,
    /// V per-position norms.
    pub v_norms: MlxBuffer,
    /// Cache capacity in positions.
    pub capacity: usize,
    /// True if ring-buffer (sliding) semantics.
    pub is_sliding: bool,
    /// Norms per position (1 for D=256, 2 for D=512).
    #[allow(dead_code)]
    pub norms_per_pos: usize,
}

/// Per-layer dense F32/F16 KV buffers for dense attention path (ADR-009).
pub struct DenseKvBuffers {
    pub k: MlxBuffer,
    pub v: MlxBuffer,
    /// Capacity (positions) in this layer's cache. Sliding layers use
    /// ring-buffer mode: capacity == sliding_window, writes wrap.
    /// Global layers use linear mode: capacity >= seq_len + max_tokens.
    pub capacity: usize,
    /// True if this is a sliding layer (ring-buffer semantics).
    pub is_sliding: bool,
    /// ADR-017 Phase E.a iter-3.5a (Codex round-1 LOW #4) — KV element
    /// dtype invariant. Today every engine in a process uses one
    /// `HF2Q_F16_KV` setting (parsed at LazyLock init), so all
    /// `DenseKvBuffers` in the live LcpRegistry necessarily share
    /// dtype. Recording it explicitly:
    ///
    ///   * makes the invariant a load-bearing struct field (not a
    ///     process-implicit assumption that could regress under any
    ///     future hot-reload / multi-engine path),
    ///   * lets the engine's `take_prefix` capacity-check site assert
    ///     `cached.dtype == model.kv_dtype` per layer alongside the
    ///     existing capacity + is_sliding checks, closing a class of
    ///     silent-corruption bugs at the type level.
    ///
    /// Populated at every construction site: `forward_prefill.rs` per-
    /// layer alloc, `forward_prefill_batched.rs` per-layer alloc, and
    /// `engine.rs::kv_restore_gemma` per-layer alloc.
    pub dtype: mlx_native::DType,
}

impl crate::serve::kv_persist::lcp_registry::ByteSized for DenseKvBuffers {
    /// Exact byte count of the K + V buffers for this layer.
    /// Uses `MlxBuffer::byte_len()` — the same API used at forward_mlx.rs:1167+
    /// and 5158+. No estimation.
    fn byte_len(&self) -> u64 {
        (self.k.byte_len() + self.v.byte_len()) as u64
    }
}

/// ADR-028 Phase 10 (iter-347): hybrid K storage — F16 K alongside TQ-HB-packed V.
///
/// Motivation (ADR-028 §iter-346 audit):
///   * Pure TQ-HB (today's default): 504 MB raw F32 → 128 MB packed (3.94× saving).
///   * Pure F16 K + F16 V (peer): 504 MB → 252 MB (2× saving).
///   * Hybrid (F16 K + TQ-HB V): 504 MB → 158 MB (3.19× saving — 81% of TQ-HB).
///
/// The structural decode-side gap vs llama.cpp peer (1.81× per-dispatch wall on
/// our TQ-HB SDPA, formally measured iter-326..342) is owned by the K-side
/// scalar dequant loop inside `flash_attn_vec_tq_hb`: peer's K is F16 and consumed
/// by `simdgroup_matrix` matmul, ours is byte-packed and consumed by per-thread
/// scalar lookup against the codebook. Storing K as F16 (this struct) and using a
/// new `flash_attn_vec_hybrid_dk256` SDPA kernel (Phase 10d) brings the K-side
/// throughput up to peer-equivalent simdgroup math while the V-side stays in
/// 1-byte-per-element TQ-HB packing.
///
/// Field layout mirrors the union of `DenseKvBuffers` (K) and `HbKvBuffers` (V) so
/// existing snapshot / restore / KV-persist code that walks the K and V buffers
/// can pattern-match by field name without learning a new shape.
///
/// Allocation gate: env `HF2Q_HYBRID_KV` (parsed in `investigation_env.rs`).
/// Default OFF until parity + bench gates pass (Phase 10f/g). When ON, the
/// per-layer alloc site at `forward_decode` (currently building `HbKvBuffers`)
/// instead builds `HybridKvBuffers`, the K-encode dispatch is skipped, and the
/// SDPA dispatcher routes to the hybrid kernel.
pub struct HybridKvBuffers {
    /// Dense F16 K cache `[nkv_heads, capacity, head_dim]`. dtype is always
    /// `mlx_native::DType::F16` for this struct (the whole point — F16 K → peer
    /// SDPA-equivalent simdgroup matmul). No F32 variant: F32 K would erase the
    /// memory advantage of the hybrid design (158 MB → 284 MB at gemma4 32K
    /// context, defeating the purpose of mixing in TQ-HB on V at all).
    pub k: MlxBuffer,
    /// Byte-packed V indices `[nkv_heads, capacity, head_dim]` U8 — same layout
    /// and codec as `HbKvBuffers::v_packed`. The V-encode dispatch
    /// (`hadamard_quantize_kv_*`) writes here unchanged.
    pub v_packed: MlxBuffer,
    /// V per-position norms — same layout as `HbKvBuffers::v_norms`.
    /// (D=256 → 1/pos, D=512 → 2/pos.)
    pub v_norms: MlxBuffer,
    /// Cache capacity in positions (matches `DenseKvBuffers::capacity` and
    /// `HbKvBuffers::capacity` — populated identically at alloc site).
    pub capacity: usize,
    /// True if ring-buffer (sliding) semantics — same as the sibling structs.
    pub is_sliding: bool,
    /// Norms per position (1 for D=256, 2 for D=512). Mirrors
    /// `HbKvBuffers::norms_per_pos`.
    #[allow(dead_code)]
    pub norms_per_pos: usize,
    /// ADR-030 iter-96: BF16 K cache for DFlash spec-decode xlen verify
    /// path. Same layout `[nkv_heads, capacity, head_dim]` as `k` but BF16
    /// dtype. Populated from `pf_k_perm` BF16 (head_norm_rope's bf16 output)
    /// via `dispatch_kv_cache_copy_seq_bf16_to_bf16_head_major`. Used by
    /// xlen branch's SDPA to read BF16 K bit-identical to what Option C
    /// reads from pf_k_perm — avoids the F16-roundtrip precision drift
    /// root-caused at iter-92/93.  Lazy-alloc'd on first xlen-mode call.
    pub bf16_xlen_k: Option<MlxBuffer>,
    /// ADR-030 iter-96: BF16 V cache, same semantics as `bf16_xlen_k`.
    pub bf16_xlen_v: Option<MlxBuffer>,
}

impl crate::serve::kv_persist::lcp_registry::ByteSized for HybridKvBuffers {
    /// Exact byte count: F16 K + U8 V + F32 V-norms. Used by the LcpRegistry
    /// byte budget the same way `DenseKvBuffers::byte_len` is.
    fn byte_len(&self) -> u64 {
        (self.k.byte_len() + self.v_packed.byte_len() + self.v_norms.byte_len()) as u64
    }
}

/// ADR-028 Phase 10c (iter-348): per-layer F16-K + TQ-HB-V buffer allocator.
///
/// Single-source-of-truth for the hybrid allocation shape — called from
/// 3 sites (decode lazy-alloc, per-token prefill alloc, batched-prefill
/// alloc) so all three stay in lockstep.
///
/// F16 K: 2 bytes/elem, shape `[nkv, cap, hd]`.
/// V layout identical to legacy `HbKvBuffers` V-side (1 byte/elem packed +
/// per-pos F32 norms with `norms_per_pos = max(1, hd / 256)`).
pub(super) fn alloc_hybrid_kv_for_layer(
    dev: &mlx_native::MlxDevice,
    layer_idx: usize,
    nkv: usize,
    hd: usize,
    cap: usize,
    is_ring: bool,
) -> anyhow::Result<HybridKvBuffers> {
    let norms_per_pos = (hd / 256).max(1);
    let norms_n = nkv * cap * norms_per_pos;
    // F16 K: byte_count = elements * 2.
    let k = dev.alloc_buffer(nkv * cap * hd * 2, mlx_native::DType::F16,
        vec![nkv, cap, hd])
        .map_err(|e| anyhow::anyhow!("hybrid F16 K L{layer_idx}: {e}"))?;
    // ADR-029 iter-20 H27: when HF2Q_FULL_F16_KV is set, V is F16 (2 bytes/elem)
    // and v_norms is a small dummy buffer (kernel ignores it when v_is_f16=1).
    // Otherwise: legacy TQ-HB packed V (1 byte/elem) + per-position F32 norms.
    let full_f16_v = std::env::var("HF2Q_FULL_F16_KV")
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "on"))
        .unwrap_or(false);
    let (v_packed, v_norms) = if full_f16_v {
        let v_f16 = dev.alloc_buffer(nkv * cap * hd * 2, mlx_native::DType::F16,
            vec![nkv, cap, hd])
            .map_err(|e| anyhow::anyhow!("hybrid F16 V L{layer_idx}: {e}"))?;
        // Dummy norms buffer (unused but kept for ABI compat with hybrid SDPA
        // signature; kernel's v_is_f16 FC=1 skips the read).
        let v_norms_dummy = dev.alloc_buffer(4, mlx_native::DType::F32, vec![1])
            .map_err(|e| anyhow::anyhow!("hybrid V norms (dummy) L{layer_idx}: {e}"))?;
        (v_f16, v_norms_dummy)
    } else {
        let v_p = dev.alloc_buffer(nkv * cap * hd, mlx_native::DType::U8,
            vec![nkv, cap, hd])
            .map_err(|e| anyhow::anyhow!("hybrid V packed L{layer_idx}: {e}"))?;
        let v_n = dev.alloc_buffer(norms_n * 4, mlx_native::DType::F32,
            if norms_per_pos == 1 { vec![nkv, cap] } else { vec![nkv, cap, norms_per_pos] })
            .map_err(|e| anyhow::anyhow!("hybrid V norms L{layer_idx}: {e}"))?;
        (v_p, v_n)
    };
    // ADR-030 iter-96: lazy-alloc the BF16 xlen cache only when env opted-in.
    // Saves ~55MB at gemma-4 when xlen mode disabled.
    let xlen_mode = std::env::var("HF2Q_DFLASH_XLEN_SDPA").as_deref() == Ok("1");
    let (bf16_xlen_k, bf16_xlen_v) = if xlen_mode {
        let bk = dev.alloc_buffer(nkv * cap * hd * 2, mlx_native::DType::BF16,
            vec![nkv, cap, hd])
            .map_err(|e| anyhow::anyhow!("bf16 xlen K L{layer_idx}: {e}"))?;
        let bv = dev.alloc_buffer(nkv * cap * hd * 2, mlx_native::DType::BF16,
            vec![nkv, cap, hd])
            .map_err(|e| anyhow::anyhow!("bf16 xlen V L{layer_idx}: {e}"))?;
        (Some(bk), Some(bv))
    } else {
        (None, None)
    };
    Ok(HybridKvBuffers { k, v_packed, v_norms, capacity: cap, is_sliding: is_ring, norms_per_pos, bf16_xlen_k, bf16_xlen_v })
}

/// Per-call decode regime override for ADR-007 Gate H two-regime-one-process
/// runs (W12 iter-108a blocker #3).
///
/// Set on `MlxModelWeights` before each prefill+decode trajectory via
/// [`MlxModelWeights::set_decode_regime`].  Consulted at the SDPA-mode
/// gate inside `forward_decode` (the `use_dense_sdpa` check); the four
/// codebook-bits gates (`forward_mlx.rs:1100/1234`, `forward_prefill.rs:330`)
/// stay env-var-driven because the codebook width is a representation
/// choice that is consistent across both regimes (both regimes read the
/// same KV format).  The four-gate lockstep contract (W9's mapping) is
/// preserved: when `regime == Default`, every gate reads env exactly as
/// it did on the iter-108a base commit; when `regime != Default`, only
/// the SDPA-mode gate flips.
///
/// - `Default` (the zero value): preserve today's env-var behavior.
/// - `ForceTq`: ignore env, behave as if `HF2Q_USE_DENSE` were unset and
///   `HF2Q_LAYER_POLICY=tq_all` — TQ-active SDPA on every layer.
/// - `ForceDense`: ignore env, behave as if `HF2Q_USE_DENSE=1` were set —
///   dense-active SDPA on every layer.
///
/// Gate H uses one [`MlxModelWeights`] instance and runs (a) `set_decode_regime
/// (ForceDense)` -> fresh `forward_prefill` + decode loop -> capture tokens
/// + per-token NLL + SDPA-output dumps; then (b) `set_decode_regime
/// (ForceTq)` -> fresh `forward_prefill` + decode loop with the same prompt
/// -> cosine the SDPA outputs against (a)'s dump and PPL the NLLs.  The
/// per-instance step counter is reset by `set_decode_regime` so each
/// regime's `[HF2Q_NLL]` / `[HF2Q_DECODE_EMIT]` lines start at `step=0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodeRegime {
    /// Honor `HF2Q_USE_DENSE` / `HF2Q_LAYER_POLICY` env vars (today's path).
    #[default]
    Default,
    /// Force TQ-active SDPA regardless of env.
    /// (Wired by iter-108b's release-check.sh Gate 5 harness; no in-tree
    /// caller as of iter-108a.)
    #[allow(dead_code)]
    ForceTq,
    /// Force dense-active SDPA regardless of env.
    /// (Wired by iter-108b's release-check.sh Gate 5 harness.)
    #[allow(dead_code)]
    ForceDense,
}

/// ADR-017 B-tq.3 helper: convert a `&[f32]` slice into a `Vec<u8>` of
/// little-endian bytes via per-element `to_le_bytes`.  Used by the
/// `tq_v2_snapshot_block` capture path to feed the pure-byte v2 codec
/// without taking a dep on `bytemuck`.  Snapshot is amortised — fires
/// per block, not per token — so the per-element loop is fine.
///
/// `#[allow(dead_code)]` because the only call site is
/// `MlxModelWeights::tq_v2_snapshot_block`, which itself is gated on
/// the operator-controlled spill activation (factory + descriptor
/// closure).  The method is reachable; the binary's default config
/// just doesn't fire it yet.
#[allow(dead_code)]
fn f32_slice_to_le_bytes(src: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() * 4);
    for &x in src {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// ADR-020 AC#5 Iter D — classification of `blk.{i}.<role>` stems
/// emitted by `hf2q dwq-train`.  Drives slot-routing in
/// [`MlxModelWeights::apply_dwq_overlay`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DwqOverlayRole {
    AttnQ,
    AttnK,
    AttnV,
    AttnOutput,
    FfnGate,
    FfnUp,
    FfnDown,
    /// Per-expert MoE tensor (`ffn_gate.{e}`, `ffn_up.{e}`,
    /// `ffn_down.{e}`); skipped in Iter D, handled in Iter C2.
    MoeExpert,
    /// Stem doesn't match any known DWQ role.
    Unknown,
}

/// ADR-020 AC#5 Iter D — parse the DWQ safetensors metadata header,
/// extracting `(bits, group_size)`.  Defaults `(4, 32)` if the
/// metadata is absent (legacy DWQ output).  Returns an error if a
/// `format` field is present but doesn't match `mlx-affine-dwq-v1`.
pub fn parse_dwq_overlay_metadata(
    metadata: Option<&std::collections::HashMap<String, String>>,
) -> Result<(u32, usize)> {
    match metadata {
        Some(meta) => {
            if let Some(format_str) = meta.get("format") {
                if format_str != "mlx-affine-dwq-v1" {
                    anyhow::bail!(
                        "DWQ overlay: unsupported format '{}' (expected 'mlx-affine-dwq-v1')",
                        format_str
                    );
                }
            }
            let bits = meta
                .get("bits")
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(4u32);
            let group_size = meta
                .get("group_size")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(32usize);
            Ok((bits, group_size))
        }
        None => Ok((4u32, 32usize)),
    }
}

/// Classify a DWQ stem's role token (the part after `blk.{i}.`).
pub fn parse_dwq_overlay_role(role: &str) -> DwqOverlayRole {
    match role {
        "attn_q" => DwqOverlayRole::AttnQ,
        "attn_k" => DwqOverlayRole::AttnK,
        "attn_v" => DwqOverlayRole::AttnV,
        "attn_output" => DwqOverlayRole::AttnOutput,
        "ffn_gate" => DwqOverlayRole::FfnGate,
        "ffn_up" => DwqOverlayRole::FfnUp,
        "ffn_down" => DwqOverlayRole::FfnDown,
        r if r.starts_with("ffn_gate_up.")
            || r.starts_with("ffn_gate.")
            || r.starts_with("ffn_up.")
            || r.starts_with("ffn_down.") =>
        {
            DwqOverlayRole::MoeExpert
        }
        _ => DwqOverlayRole::Unknown,
    }
}

/// ADR-020 AC#5 Iter C2.2 — base role buckets for stacked MoE expert
/// loading.  An expert stem `ffn_gate_up.{N}` maps to `GateUp`,
/// `ffn_down.{N}` → `Down`, `ffn_gate.{N}` → `Gate`, `ffn_up.{N}` →
/// `Up`.  GateUp is the FUSED case (qwen3.5 GGUF `ffn_gate_up_exps`);
/// Gate + Up separately is the unfused case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoeBaseRole {
    GateUp,
    Gate,
    Up,
    Down,
}

/// Parse a per-expert MoE stem's role suffix (e.g. `ffn_gate_up.13`)
/// into `(MoeBaseRole, expert_idx)`.  Returns `None` if the role does
/// not match a known per-expert pattern.
pub fn parse_dwq_moe_expert_role(role: &str) -> Option<(MoeBaseRole, usize)> {
    // `ffn_gate_up.` must be checked before `ffn_gate.` to avoid the
    // longer prefix being consumed as `Gate.up.{e}` (which would be
    // "Gate" with a stray `up.` suffix).
    let (base, rest) = if let Some(rest) = role.strip_prefix("ffn_gate_up.") {
        (MoeBaseRole::GateUp, rest)
    } else if let Some(rest) = role.strip_prefix("ffn_gate.") {
        (MoeBaseRole::Gate, rest)
    } else if let Some(rest) = role.strip_prefix("ffn_up.") {
        (MoeBaseRole::Up, rest)
    } else if let Some(rest) = role.strip_prefix("ffn_down.") {
        (MoeBaseRole::Down, rest)
    } else {
        return None;
    };
    rest.parse::<usize>().ok().map(|e| (base, e))
}

impl MlxModelWeights {
    /// ADR-030 Phase 4 — install a DFlash hidden-state capture session.
    ///
    /// While installed, `forward_prefill_batched` will populate the
    /// session's `hidden_output` buffer with `pf_hidden` contents at
    /// layer indices matching `session.target_layer_ids`. Reset the
    /// session via `take_dflash_capture` after the forward returns.
    ///
    /// Default state (no install): byte-identical to legacy behavior.
    pub fn install_dflash_capture(
        &mut self,
        session: crate::inference::spec_decode::dflash::hidden_capture::DFlashCaptureSession,
    ) {
        self.dflash_capture = Some(session);
    }

    /// Take back the installed DFlash capture session, returning its
    /// populated buffers. Returns `None` if no session was installed.
    /// After this call, subsequent `forward_prefill_batched` calls
    /// revert to legacy non-capturing behavior.
    pub fn take_dflash_capture(
        &mut self,
    ) -> Option<crate::inference::spec_decode::dflash::hidden_capture::DFlashCaptureSession> {
        self.dflash_capture.take()
    }

    /// True if a DFlash capture session is currently installed.
    pub fn has_dflash_capture(&self) -> bool {
        self.dflash_capture.is_some()
    }

    /// ADR-030 Phase 4 — public embed_tokens lookup.
    ///
    /// Mirrors the gather+scale embedding inside `forward_prefill_batched`
    /// (lines 618-641): for each token in `tokens`, copy
    /// `embed_weight[token_id * hidden_size..]` into the output buffer,
    /// then scale by `sqrt(hidden_size)` (gemma's embed_scale convention).
    ///
    /// Returns a fresh MlxBuffer of shape `[tokens.len(), hidden_size]`
    /// F32, ready to feed into `dispatch_dflash_model_forward` as `h`.
    ///
    /// Used by the DFlash spec-decode orchestrator to embed the
    /// "block" `[last_committed_token, mask, mask, ..., mask]` before
    /// the drafter forward.
    pub fn embed_tokens(
        &self,
        tokens: &[u32],
        gpu: &mut crate::serve::gpu::GpuContext,
    ) -> anyhow::Result<MlxBuffer> {
        let hs = self.hidden_size;
        let n_tokens = tokens.len();
        if n_tokens == 0 {
            anyhow::bail!("embed_tokens: empty tokens");
        }
        let scale = (hs as f32).sqrt();
        let (exec, _reg) = gpu.split();
        let dev = exec.device();
        let mut out = dev
            .alloc_buffer(n_tokens * hs * 4, mlx_native::DType::F32, vec![n_tokens, hs])
            .map_err(|e| anyhow::anyhow!("alloc embed output: {e}"))?;
        let embed_f32: &[f32] = self
            .embed_weight
            .as_slice()
            .map_err(|e| anyhow::anyhow!("embed_weight slice: {e}"))?;
        // Validate vocab bound
        let vocab_in_buf = embed_f32.len() / hs;
        for &tok in tokens.iter() {
            if (tok as usize) >= vocab_in_buf {
                anyhow::bail!(
                    "embed_tokens: token id {} out of vocab range {}",
                    tok, vocab_in_buf
                );
            }
        }
        {
            let out_slice: &mut [f32] = out
                .as_mut_slice()
                .map_err(|e| anyhow::anyhow!("embed output slice: {e}"))?;
            for (i, &tok) in tokens.iter().enumerate() {
                let src = (tok as usize) * hs;
                let dst = i * hs;
                out_slice[dst..dst + hs].copy_from_slice(&embed_f32[src..src + hs]);
            }
            for v in out_slice.iter_mut() {
                *v *= scale;
            }
        }
        Ok(out)
    }

    /// ADR-030 Phase 4 — compute per-position argmaxes from a
    /// post-last-layer hidden state buffer.
    ///
    /// Convenience wrapper around `per_position_argmax_from_hidden_opt`
    /// with `apply_final_norm = true` (matches the target's tail).
    ///
    /// Takes `hidden` of shape `[seq_len, hidden_size]` F32 (the
    /// `pf_hidden` content after all decoder layers ran, captured via
    /// the DFlash capture hook with the FINAL layer index included
    /// in `target_layer_ids`). For each row, runs final_norm +
    /// lm_head + softcap + argmax. Returns `[seq_len]` u32.
    ///
    /// Uses the same model state as the existing last-row tail in
    /// `forward_prefill_batched`. Bit-exact same dispatch ordering —
    /// guarantees the LAST-position argmax matches first_token.
    ///
    /// **Not on the production hot path.** Spec-decode verify only.
    pub fn per_position_argmax_from_hidden(
        &mut self,
        hidden: &[f32],
        seq_len: u32,
        gpu: &mut crate::serve::gpu::GpuContext,
    ) -> anyhow::Result<Vec<u32>> {
        self.per_position_argmax_from_hidden_opt(hidden, seq_len, true, gpu)
    }

    /// ADR-030 Phase 4 — per-position argmax with optional final_norm.
    ///
    /// `apply_final_norm = true` mirrors target's tail (used for
    /// `forward_decode_verify_batched`).
    ///
    /// `apply_final_norm = false` is the drafter-side path: the drafter
    /// applies its own `norm` (drafter's final_norm) inside
    /// `dispatch_dflash_model_forward`; the orchestrator then takes
    /// the drafter's h_final and runs target's lm_head + softcap +
    /// argmax via THIS method with apply_final_norm=false. Mirrors
    /// Python `model_mlx.py:194` — `logits = self.lm_head(self.norm(h))`
    /// where `self.norm` is the drafter's, `self.lm_head` is target's
    /// (shared via `bind()`).
    pub fn per_position_argmax_from_hidden_opt(
        &mut self,
        hidden: &[f32],
        seq_len: u32,
        apply_final_norm: bool,
        gpu: &mut crate::serve::gpu::GpuContext,
    ) -> anyhow::Result<Vec<u32>> {
        let hs = self.hidden_size;
        let vocab_size = self.vocab_size;
        let expected = (seq_len as usize) * hs;
        if hidden.len() != expected {
            anyhow::bail!(
                "per_position_argmax_from_hidden: hidden len {} != seq_len({}) * hs({}) = {}",
                hidden.len(), seq_len, hs, expected
            );
        }
        // ADR-030 iter-70: HF2Q_DFLASH_BATCH_ARGMAX=1 (opt-in) routes to
        // the batched implementation that:
        // 1. CPU-uploads ALL hidden rows in one shot (no per-iter CPU writes)
        // 2. Runs all seq_len iterations in ONE command buffer (one finish())
        // 3. Reads all argmaxes from a seq_len-sized output buffer at end
        // Saves ~K * sync_overhead per call.  Profile data at N=16 shows
        // target_argmax = 51 ms/round → expected ~10 ms/round after batching.
        if std::env::var("HF2Q_DFLASH_BATCH_ARGMAX").as_deref() == Ok("1") {
            return self.per_position_argmax_from_hidden_batched_impl(
                hidden, seq_len, apply_final_norm, gpu,
            );
        }
        let mut argmaxes = Vec::with_capacity(seq_len as usize);

        let (exec, reg) = gpu.split();
        let dev = exec.device();
        let metal_dev = dev.metal_device();

        for pos in 0..(seq_len as usize) {
            // Copy hidden[pos] into activations.hidden via CPU→GPU upload.
            {
                let slice: &mut [f32] = self
                    .activations
                    .hidden
                    .as_mut_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("activations.hidden slice: {e}"))?;
                slice[..hs].copy_from_slice(&hidden[pos * hs..(pos + 1) * hs]);
            }

            // Open session and run final_norm + lm_head + softcap + argmax.
            let mut s = exec
                .begin()
                .map_err(|e| anyhow::anyhow!("per_pos session begin: {e}"))?;

            // norm_out source: either final_norm(hidden) when caller
            // requests target's final_norm, OR a direct copy of hidden
            // when caller has already applied the drafter's final_norm
            // externally.
            if apply_final_norm {
                s.barrier_between(
                    &[&self.activations.hidden, &self.final_norm],
                    &[&self.activations.norm_out],
                );
                s.rms_norm(
                    reg,
                    metal_dev,
                    &self.activations.hidden,
                    &self.final_norm,
                    &self.activations.norm_out,
                    &self.activations.norm_params,
                    1,
                    hs as u32,
                )
                .map_err(|e| anyhow::anyhow!("per_pos final_norm: {e}"))?;
            } else {
                // Copy hidden → norm_out (already pre-normed by drafter)
                s.barrier_between(
                    &[&self.activations.hidden],
                    &[&self.activations.norm_out],
                );
                mlx_native::ops::copy::dispatch_copy_f32(
                    s.encoder_mut(),
                    reg,
                    metal_dev,
                    &self.activations.hidden,
                    &self.activations.norm_out,
                    0,
                    0,
                    hs,
                )
                .map_err(|e| anyhow::anyhow!("per_pos pre-normed copy: {e}"))?;
            }

            if let Some(ref q6k) = self.lm_head_q6k {
                s.barrier_between(
                    &[&self.activations.norm_out, &q6k.buffer],
                    &[&self.activations.logits],
                );
                super::forward_mlx::dispatch_qmatmul(
                    &mut s,
                    reg,
                    dev,
                    &self.activations.norm_out,
                    q6k,
                    &mut self.activations.logits,
                    1,
                )
                .map_err(|e| anyhow::anyhow!("per_pos lm_head Q6_K: {e}"))?;
            } else if let Some(ref q8) = self.lm_head_q8 {
                s.barrier_between(
                    &[&self.activations.norm_out, &q8.buffer],
                    &[&self.activations.logits],
                );
                super::forward_mlx::dispatch_qmatmul(
                    &mut s,
                    reg,
                    dev,
                    &self.activations.norm_out,
                    q8,
                    &mut self.activations.logits,
                    1,
                )
                .map_err(|e| anyhow::anyhow!("per_pos lm_head Q8: {e}"))?;
            } else if let Some(ref lm_head_f16) = self.lm_head_f16 {
                s.barrier_between(
                    &[&self.activations.norm_out, lm_head_f16],
                    &[&self.activations.logits],
                );
                mlx_native::ops::dense_gemm::dispatch_dense_matvec_f16w_f32io(
                    s.encoder_mut(),
                    reg,
                    metal_dev,
                    &self.activations.norm_out,
                    lm_head_f16,
                    &self.activations.logits,
                    &mlx_native::ops::dense_gemm::DenseGemmF16Params {
                        m: 1,
                        n: vocab_size as u32,
                        k: hs as u32,
                    },
                )
                .map_err(|e| anyhow::anyhow!("per_pos lm_head f16: {e}"))?;
            } else {
                anyhow::bail!("per_position_argmax_from_hidden requires lm_head_q6k / q8 / f16");
            }

            if let Some(cap) = self.final_logit_softcapping {
                s.barrier_between(
                    &[&self.activations.logits],
                    &[&self.activations.logits],
                );
                mlx_native::ops::softcap::dispatch_softcap(
                    s.encoder_mut(),
                    reg,
                    metal_dev,
                    &self.activations.logits,
                    &self.activations.logits,
                    &self.activations.softcap_params,
                    cap,
                )
                .map_err(|e| anyhow::anyhow!("per_pos softcap: {e}"))?;
            }

            s.barrier_between(
                &[&self.activations.logits],
                &[&self.activations.argmax_index, &self.activations.argmax_value],
            );
            mlx_native::ops::argmax::dispatch_argmax_f32(
                s.encoder_mut(),
                reg,
                metal_dev,
                &self.activations.logits,
                &self.activations.argmax_index,
                &self.activations.argmax_value,
                &self.activations.argmax_params,
                vocab_size as u32,
            )
            .map_err(|e| anyhow::anyhow!("per_pos argmax: {e}"))?;

            s.finish()
                .map_err(|e| anyhow::anyhow!("per_pos session finish: {e}"))?;

            let argmax_val: u32 = {
                let idx: &[u32] = self
                    .activations
                    .argmax_index
                    .as_slice()
                    .map_err(|e| anyhow::anyhow!("per_pos argmax read: {e}"))?;
                idx[0]
            };
            argmaxes.push(argmax_val);
        }

        Ok(argmaxes)
    }

    /// ADR-030 iter-70 — batched per-position argmax.
    ///
    /// Equivalent semantically to [`per_position_argmax_from_hidden_opt`]
    /// but:
    /// 1. Uploads ALL hidden rows once (one bulk CPU→GPU copy).
    /// 2. Allocates seq_len-element argmax_index/value output buffers.
    /// 3. Runs all `seq_len` chains (copy → norm → lm_head → softcap →
    ///    argmax) inside ONE command buffer.  Shared scratch
    ///    (activations.hidden / norm_out / logits) is reused per
    ///    iteration with `barrier_between` ensuring iter i's reads of
    ///    a shared buffer complete before iter i+1's writes.
    /// 4. Single `finish()` at end → reads all argmaxes from the
    ///    per-position output buffer view.
    ///
    /// Eliminates `seq_len - 1` `commit_and_wait` syncs.  Profile data
    /// shows ~5-7 ms per sync, so for seq_len=8 we expect ~35-50 ms
    /// savings per call (validated by iter-71 bench).
    pub(crate) fn per_position_argmax_from_hidden_batched_impl(
        &mut self,
        hidden: &[f32],
        seq_len: u32,
        apply_final_norm: bool,
        gpu: &mut crate::serve::gpu::GpuContext,
    ) -> anyhow::Result<Vec<u32>> {
        let hs = self.hidden_size;
        let vocab_size = self.vocab_size;
        let n = seq_len as usize;
        let expected = n * hs;
        if hidden.len() != expected {
            anyhow::bail!(
                "per_position_argmax_batched: hidden len {} != seq_len({}) * hs({}) = {}",
                hidden.len(), seq_len, hs, expected
            );
        }
        let (exec, reg) = gpu.split();
        let dev = exec.device();
        let metal_dev = dev.metal_device();

        // (1) Bulk upload hidden → GPU.
        let mut gpu_hidden_all = dev
            .alloc_buffer(n * hs * 4, mlx_native::DType::F32, vec![n, hs])
            .map_err(|e| anyhow::anyhow!("alloc gpu_hidden_all: {e}"))?;
        gpu_hidden_all
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("gpu_hidden_all slice: {e}"))?
            .copy_from_slice(hidden);

        // (2) Per-position argmax output buffers.
        let argmax_index_all = dev
            .alloc_buffer(n * 4, mlx_native::DType::U32, vec![n])
            .map_err(|e| anyhow::anyhow!("alloc argmax_index_all: {e}"))?;
        let argmax_value_all = dev
            .alloc_buffer(n * 4, mlx_native::DType::F32, vec![n])
            .map_err(|e| anyhow::anyhow!("alloc argmax_value_all: {e}"))?;

        // iter-72: truly batched processing.  One rms_norm with rows=n,
        // one lm_head matmul with m=n, one softcap over n*vocab, then
        // n argmax dispatches with logits views per row.  Replaces the
        // iter-70 sequential loop on shared scratch.
        let mut norm_out_batched = dev
            .alloc_buffer(n * hs * 4, mlx_native::DType::F32, vec![n, hs])
            .map_err(|e| anyhow::anyhow!("alloc norm_out_batched: {e}"))?;
        let mut logits_batched = dev
            .alloc_buffer(
                n * (vocab_size as usize) * 4,
                mlx_native::DType::F32,
                vec![n, vocab_size as usize],
            )
            .map_err(|e| anyhow::anyhow!("alloc logits_batched: {e}"))?;

        let mut s = exec
            .begin()
            .map_err(|e| anyhow::anyhow!("batched argmax session begin: {e}"))?;

        // (a) ONE rms_norm or copy with rows=n
        if apply_final_norm {
            s.barrier_between(
                &[&gpu_hidden_all, &self.final_norm],
                &[&norm_out_batched],
            );
            s.rms_norm(
                reg,
                metal_dev,
                &gpu_hidden_all,
                &self.final_norm,
                &norm_out_batched,
                &self.activations.norm_params,
                n as u32,
                hs as u32,
            )
            .map_err(|e| anyhow::anyhow!("batched arg rms_norm: {e}"))?;
        } else {
            s.barrier_between(
                &[&gpu_hidden_all],
                &[&norm_out_batched],
            );
            mlx_native::ops::copy::dispatch_copy_f32(
                s.encoder_mut(),
                reg,
                metal_dev,
                &gpu_hidden_all,
                &norm_out_batched,
                0,
                0,
                n * hs,
            )
            .map_err(|e| anyhow::anyhow!("batched arg pre-norm copy: {e}"))?;
        }

        // (b) ONE lm_head dispatch with m=n.  dispatch_qmatmul routes
        //     m<=8 through mat-vec (multiple matvecs per dispatch) and
        //     m>8 through mat-mat (simdgroup MMA, tile kernel).
        if let Some(ref q6k) = self.lm_head_q6k {
            s.barrier_between(
                &[&norm_out_batched, &q6k.buffer],
                &[&logits_batched],
            );
            super::forward_mlx::dispatch_qmatmul(
                &mut s,
                reg,
                dev,
                &norm_out_batched,
                q6k,
                &mut logits_batched,
                n as u32,
            )
            .map_err(|e| anyhow::anyhow!("batched arg lm_head Q6_K: {e}"))?;
        } else if let Some(ref q8) = self.lm_head_q8 {
            s.barrier_between(
                &[&norm_out_batched, &q8.buffer],
                &[&logits_batched],
            );
            super::forward_mlx::dispatch_qmatmul(
                &mut s,
                reg,
                dev,
                &norm_out_batched,
                q8,
                &mut logits_batched,
                n as u32,
            )
            .map_err(|e| anyhow::anyhow!("batched arg lm_head Q8: {e}"))?;
        } else if let Some(ref lm_head_f16) = self.lm_head_f16 {
            s.barrier_between(
                &[&norm_out_batched, lm_head_f16],
                &[&logits_batched],
            );
            mlx_native::ops::dense_gemm::dispatch_dense_matvec_f16w_f32io(
                s.encoder_mut(),
                reg,
                metal_dev,
                &norm_out_batched,
                lm_head_f16,
                &logits_batched,
                &mlx_native::ops::dense_gemm::DenseGemmF16Params {
                    m: n as u32,
                    n: vocab_size as u32,
                    k: hs as u32,
                },
            )
            .map_err(|e| anyhow::anyhow!("batched arg lm_head F16: {e}"))?;
        } else {
            anyhow::bail!(
                "per_position_argmax_batched requires lm_head_q6k / q8 / f16"
            );
        }

        // (c) ONE softcap on the full n*vocab logits (element-wise).
        //     dispatch_softcap is element-wise so it works on any size.
        if let Some(cap) = self.final_logit_softcapping {
            s.barrier_between(
                &[&logits_batched],
                &[&logits_batched],
            );
            mlx_native::ops::softcap::dispatch_softcap(
                s.encoder_mut(),
                reg,
                metal_dev,
                &logits_batched,
                &logits_batched,
                &self.activations.softcap_params,
                cap,
            )
            .map_err(|e| anyhow::anyhow!("batched arg softcap: {e}"))?;
        }

        // (d) Per-row argmax (the kernel itself isn't batchable; this
        //     stage is cheap compared to lm_head).  We dispatch within
        //     the same session — n small dispatches, one finish().
        for pos in 0..n {
            let logits_row = logits_batched
                .slice_view((pos * (vocab_size as usize) * 4) as u64, vocab_size as usize);
            let argmax_idx_view = argmax_index_all.slice_view((pos * 4) as u64, 1);
            let argmax_val_view = argmax_value_all.slice_view((pos * 4) as u64, 1);
            s.barrier_between(
                &[&logits_batched],
                &[&argmax_idx_view, &argmax_val_view],
            );
            mlx_native::ops::argmax::dispatch_argmax_f32(
                s.encoder_mut(),
                reg,
                metal_dev,
                &logits_row,
                &argmax_idx_view,
                &argmax_val_view,
                &self.activations.argmax_params,
                vocab_size as u32,
            )
            .map_err(|e| anyhow::anyhow!("batched arg argmax L{pos}: {e}"))?;
        }

        // (4) Single commit + wait for ALL iterations.
        s.finish()
            .map_err(|e| anyhow::anyhow!("batched argmax session finish: {e}"))?;

        // (5) Read all argmaxes in one shot.
        let argmaxes: Vec<u32> = argmax_index_all
            .as_slice::<u32>()
            .map_err(|e| anyhow::anyhow!("batched argmax read: {e}"))?
            .iter()
            .take(n)
            .copied()
            .collect();
        Ok(argmaxes)
    }

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

        // ADR-028 iter-482: time pre-loop loads (embed_weight + final_norm)
        // separately to verify iter-481's embed_weight ~200ms estimate.
        let load_timing = std::env::var("HF2Q_LOAD_TIMING").as_deref() == Ok("1");
        let t_pre = std::time::Instant::now();

        // --- Embedding weight (F32) ---
        tracing::debug!("Loading embed_weight");
        let embed_weight = gguf.load_tensor_f32("token_embd.weight", mlx_device)
            .map_err(|e| anyhow::anyhow!("embed: {e}"))?;
        if load_timing {
            tracing::info!("[LOAD_TIMING] embed_weight_load={:.0}ms", t_pre.elapsed().as_secs_f64()*1000.0);
        }
        let t_fn = std::time::Instant::now();

        // --- Final norm (F32) ---
        tracing::debug!("Loading final_norm");
        let final_norm = gguf.load_tensor_f32("output_norm.weight", mlx_device)
            .map_err(|e| anyhow::anyhow!("final_norm: {e}"))?;
        if load_timing {
            tracing::info!("[LOAD_TIMING] final_norm_load={:.0}ms", t_fn.elapsed().as_secs_f64()*1000.0);
        }

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

        // ADR-028 iter-188: HF2Q_LMHEAD_Q6K — load token_embd.weight as
        // native on-disk Q6_K (no F32→Q8 re-quant). Saves 0.33 ms/token
        // (~2% decode).
        //
        // iter-326 default-flipped to ON; iter-344 reverted because of
        // batched-prefill conflict; iter-345 RESTORED to default-ON
        // because forward_prefill_batched.rs now has a Q6_K arm
        // dispatching via dispatch_qmatmul + kernel_mul_mv_q6_K_f32_nr2.
        // Q6_K lm_head + batched prefill COEXIST.  Opt-out via
        // HF2Q_LMHEAD_Q6K=0/false/off.
        let q6k_env_off = matches!(
            std::env::var("HF2Q_LMHEAD_Q6K").ok().as_deref(),
            Some(v) if v.eq_ignore_ascii_case("0")
                || v.eq_ignore_ascii_case("false")
                || v.eq_ignore_ascii_case("off")
        );
        let use_q6k = !q6k_env_off
            && {
                gguf.tensor_info("token_embd.weight")
                    .map(|t| t.ggml_type == mlx_native::GgmlType::Q6_K)
                    .unwrap_or(false)
            };

        let use_q8 = if use_q6k {
            false
        } else {
            match q8_env.as_deref() {
                Some("1") => true,
                Some("0") => false,
                _ => {
                    // Auto: Q8 when F16 weight would exceed 256 MB and the
                    // shape is Q8-compatible.
                    lm_head_f16_bytes > 256 * 1024 * 1024 && cfg.hidden_size % 32 == 0
                }
            }
        };

        // Decide which buffers to allocate. Compare mode always keeps F16
        // (needed as the oracle for A/B), and Q8 if requested.
        let need_q8 = use_q8;
        let need_f16 = !use_q8 && !use_q6k || compare_mode;
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
            let cols = cfg.hidden_size;
            // ADR-005 iter-214 follow-up — vocab-pad slice OOB fix.
            //
            // Derive `rows` from the actual embed tensor's element count,
            // NOT from `cfg.vocab_size`. ADR-012 Phase 1.8's vocab-pad
            // de-pad transform strips the trailing padded row from
            // `token_embd.weight` (e.g., Qwen3.6 27B carries the de-padded
            // tensor on-disk while `cfg.vocab_size` may still reflect the
            // padded count read from a different GGUF metadata key) — so
            // `cfg.vocab_size > embed_f32.len() / cols` by exactly the
            // pad-stride when de-pad fired upstream.
            //
            // The pre-fix code iterated `0..cfg.vocab_size` and hit a
            // slice-OOB panic at row `embed_rows` (one past the actual
            // tensor) on `forward_mlx.rs:861` — observed concretely on
            // /opt/hf2q/models/qwen3.6-27b-dwq46/...gguf where the loop
            // wanted row 248045 but the tensor had only 248044 rows.
            //
            // Per Engineering Mantra "code + test == truth": the tensor's
            // actual element count is the source of truth here, not the
            // metadata header. Surface a warn! when they differ so an
            // upstream cfg-vs-tensor inconsistency stays visible.
            if embed_f32.len() % cols != 0 {
                anyhow::bail!(
                    "embed_weight length {} is not divisible by hidden_size {} \
                     (cannot derive Q8_0 LMHEAD row count)",
                    embed_f32.len(), cols);
            }
            let rows = embed_f32.len() / cols;
            if rows != cfg.vocab_size {
                tracing::warn!(
                    "Q8_0 LMHEAD: embed_weight has {} rows but cfg.vocab_size={} \
                     (ADR-012 Phase 1.8 vocab-pad de-pad likely fired; using tensor's \
                     actual row count for Q8 quantize loop bound)",
                    rows, cfg.vocab_size);
            }
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
                affine: None,
                f16_shadow: None,
            })
        } else {
            None
        };

        // ADR-028 iter-188 — load token_embd.weight as Q6_K natively (no
        // F32→Q8_0 re-quant).  Same source tensor as `embed_weight` but
        // loaded directly from the GGUF Q6_K storage.  Used for lm_head
        // dispatch via dispatch_qmatmul (Q6_K mat-vec kernel).
        let lm_head_q6k: Option<MlxQWeight> = if use_q6k {
            tracing::info!(
                "Loading lm_head Q6_K natively (HF2Q_LMHEAD_Q6K=1, save \
                 ~179 MB vs Q8_0)"
            );
            Some(load_gguf_qweight(gguf, "token_embd.weight", mlx_device)
                .map_err(|e| anyhow::anyhow!("lm_head_q6k native load: {e}"))?)
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

        // ADR-028 iter-462: bucket timing inside layer loop, opt-in via
        // HF2Q_LOAD_TIMING=1.  Bisects mlx_weights_load (88% of startup
        // per iter-461) into attn/mlp/moe/misc.
        let load_timing = std::env::var("HF2Q_LOAD_TIMING").as_deref() == Ok("1");
        let mut cum_attn_ns = 0u128;
        let mut cum_mlp_ns = 0u128;
        let mut cum_moe_ns = 0u128;
        let mut cum_misc_ns = 0u128;
        // ADR-028 iter-463: MoE sub-buckets
        let mut cum_moe_gate_up_ns = 0u128;
        let mut cum_moe_down_ns = 0u128;
        let mut cum_moe_router_cpu_ns = 0u128;
        let mut cum_moe_other_ns = 0u128;

        for i in 0..num_layers {
            tracing::debug!("GGUF layer {}/{}: loading weights", i + 1, num_layers);

            // -- Attention quantized weights --
            let t_attn = std::time::Instant::now();
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
            cum_attn_ns += t_attn.elapsed().as_nanos();

            let attn = MlxAttentionWeights {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm_weight,
                k_norm_weight,
            };

            // -- Dense MLP (quantized) --
            let t_mlp = std::time::Instant::now();
            let gate_proj = load_gguf_qweight(gguf, &format!("blk.{i}.ffn_gate.weight"), mlx_device)?;
            let up_proj = load_gguf_qweight(gguf, &format!("blk.{i}.ffn_up.weight"), mlx_device)?;
            let down_proj = load_gguf_qweight(gguf, &format!("blk.{i}.ffn_down.weight"), mlx_device)?;

            let mlp = MlxMlpWeights {
                gate_proj,
                up_proj,
                down_proj,
            };
            cum_mlp_ns += t_mlp.elapsed().as_nanos();

            // -- MoE expert weights (3D tensors, already stacked in GGUF) --
            //
            // Wedge-4 / iter-227 (2026-05-02): make the MoE expert load
            // conditional on tensor presence. Pre-iter-227 the loader
            // unconditionally required `blk.{i}.ffn_gate_up_exps.weight`
            // and bailed with `missing blk.0.ffn_gate_up_exps.weight`
            // when handed a dense GGUF (e.g. the real Qwen3-VL-2B-Instruct
            // GGUF emitted by `scripts/wedge4_qwen3vl.sh` Step 3, which
            // is structurally dense — `general.architecture = "qwen3_vl"`
            // with `ffn_{gate,up,down}.weight` per layer and no expert
            // tensors).
            //
            // Detection rule (deterministic, GGUF-metadata-only — never
            // filename-based per the iter-227 correctness pin): a layer
            // is MoE iff BOTH `ffn_gate_up_exps.weight` AND
            // `ffn_down_exps.weight` tensors are present in the GGUF.
            // When either is absent we treat the layer as dense and
            // populate `MlxMoeWeights` with `stacked_{gate_up,down}: None`
            // plus 1-element placeholder buffers for the `router_*` /
            // `per_expert_scale` / `router_combined_weight` fields. The
            // forward dispatch (`forward_decode` lines 2863 / 3922)
            // already gates fused-id MoE on `stacked_gate_up.is_some()
            // && stacked_down.is_some()`; the dense MLP path consumes
            // `MlxMlpWeights` (loaded above unconditionally at lines
            // 962-971) and never reads the placeholder MoE fields.
            // Layer mixing (some layers MoE, some dense) is supported
            // structurally, mirroring llama.cpp's
            // `LLM_ARCH_QWEN3VLMOE` per-block decision.
            let gu_name = format!("blk.{i}.ffn_gate_up_exps.weight");
            let dn_name = format!("blk.{i}.ffn_down_exps.weight");
            let gu_info_opt = gguf.tensor_info(&gu_name);
            let dn_info_opt = gguf.tensor_info(&dn_name);
            let layer_has_moe_experts = gu_info_opt.is_some() && dn_info_opt.is_some();

            let t_moe = std::time::Instant::now();
            let moe = if layer_has_moe_experts {
                // MoE layer — preserve pre-iter-227 load behavior byte-
                // identically. The two-clone of `gguf.tensor_info` is
                // safe; we already established both are Some above.
                let t_gu = std::time::Instant::now();
                let gu_info = gu_info_opt.unwrap();
                let stacked_gate_up_buf = gguf.load_tensor(&gu_name, mlx_device)
                    .map_err(|e| anyhow::anyhow!("load {gu_name}: {e}"))?;
                let gate_up_expert_stride = stacked_gate_up_buf.byte_len() / cfg.num_experts;
                let gate_up_ggml_dtype = gu_info.ggml_type;
                cum_moe_gate_up_ns += t_gu.elapsed().as_nanos();

                let t_dn = std::time::Instant::now();
                let dn_info = dn_info_opt.unwrap();
                let stacked_down_buf = gguf.load_tensor(&dn_name, mlx_device)
                    .map_err(|e| anyhow::anyhow!("load {dn_name}: {e}"))?;
                let down_expert_stride = stacked_down_buf.byte_len() / cfg.num_experts;
                let down_ggml_dtype = dn_info.ggml_type;
                cum_moe_down_ns += t_dn.elapsed().as_nanos();

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
                let t_rcw = std::time::Instant::now();
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
                cum_moe_router_cpu_ns += t_rcw.elapsed().as_nanos();

                MlxMoeWeights {
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
                    gate_up_affine: None,
                    down_affine: None,
                }
            } else {
                // Dense layer — produce a placeholder MoE bundle so the
                // existing per-layer struct (`MlxDecoderLayerWeights`)
                // stays uniform without a Vec<Option<MoeWeights>>
                // ripple change. The dense forward path uses
                // `MlxMlpWeights` (loaded at lines 962-971) and never
                // reads these placeholder buffers; the fused-id MoE
                // dispatch already gates on `stacked_gate_up.is_some()
                // && stacked_down.is_some()` (forward_decode lines
                // 2863 / 3922) so the placeholders are consulted only
                // by metadata fields like `top_k` (read but unused on
                // the dense path). Buffer sizes are 1 element to
                // minimize wasted allocation; on a 28-layer Qwen3-VL-2B
                // dense load this adds 28 × ~16 bytes = ~448 bytes
                // overhead vs. the pre-iter-227 unconditional path
                // (which would have OOM-allocated GBs of expert
                // tensors that don't exist on disk).
                if i == 0 {
                    tracing::debug!(
                        "GGUF layer {}/{}: dense FFN detected (no {gu_name} / {dn_name}); \
                         skipping MoE expert load — using placeholder MoE bundle",
                        i + 1, num_layers,
                    );
                }
                MlxMoeWeights::dense_placeholder(mlx_device)
                    .map_err(|e| anyhow::anyhow!("layer {i} MoE placeholder alloc: {e}"))?
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
            // D=256: 1 norm per position (norms_per_pos=1).
            // D=512: 2 per-block norms per position (norms_per_pos=2),
            //   per AmesianX cpy-utils.cuh:241-269 (ADR-007 iter-15 per-block norm).
            let packed_bytes = nkv * capacity * (hd / 2);
            let norms_per_pos = (hd / 256).max(1);
            let norms_elements = nkv * capacity * norms_per_pos;
            let norms_bytes = norms_elements * 4; // f32 = 4 bytes

            let k_packed = mlx_device.alloc_buffer(
                packed_bytes, mlx_native::DType::U8, vec![nkv, capacity, hd / 2],
            ).map_err(|e| anyhow::anyhow!("KV cache K packed alloc: {e}"))?;
            let k_norms = mlx_device.alloc_buffer(
                norms_bytes, mlx_native::DType::F32,
                if norms_per_pos == 1 { vec![nkv, capacity] } else { vec![nkv, capacity, norms_per_pos] },
            ).map_err(|e| anyhow::anyhow!("KV cache K norms alloc: {e}"))?;
            let v_packed = mlx_device.alloc_buffer(
                packed_bytes, mlx_native::DType::U8, vec![nkv, capacity, hd / 2],
            ).map_err(|e| anyhow::anyhow!("KV cache V packed alloc: {e}"))?;
            let v_norms = mlx_device.alloc_buffer(
                norms_bytes, mlx_native::DType::F32,
                if norms_per_pos == 1 { vec![nkv, capacity] } else { vec![nkv, capacity, norms_per_pos] },
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

            cum_moe_ns += t_moe.elapsed().as_nanos();
            let t_misc = std::time::Instant::now();
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
            cum_misc_ns += t_misc.elapsed().as_nanos();
        }
        progress.finish();
        if load_timing {
            tracing::info!(
                "[LOAD_TIMING] layer_loop_buckets attn={:.0}ms mlp={:.0}ms moe={:.0}ms misc(norms+push+progress)={:.0}ms n_layers={}",
                cum_attn_ns as f64 / 1e6,
                cum_mlp_ns as f64 / 1e6,
                cum_moe_ns as f64 / 1e6,
                cum_misc_ns as f64 / 1e6,
                num_layers,
            );
            // ADR-028 iter-463: MoE sub-buckets
            cum_moe_other_ns = cum_moe_ns.saturating_sub(
                cum_moe_gate_up_ns + cum_moe_down_ns + cum_moe_router_cpu_ns
            );
            tracing::info!(
                "[LOAD_TIMING] moe_sub_buckets gate_up={:.0}ms down={:.0}ms router_cpu={:.0}ms other(router_proj+scales+placeholder)={:.0}ms",
                cum_moe_gate_up_ns as f64 / 1e6,
                cum_moe_down_ns as f64 / 1e6,
                cum_moe_router_cpu_ns as f64 / 1e6,
                cum_moe_other_ns as f64 / 1e6,
            );
        }
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
            lm_head_q6k,
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
            dense_kvs_snapshot_for_lcp: None,
            dense_sdpa_tmp: None,
            // iter-222 (2026-05-01): leg_f_kvs / leg_f_sdpa_tmp shadow-cache
            // fields deleted along with iter-34 dense-on-shadow Leg F branch.
            leg_hb_encoded: None,
            // ADR-028 Phase 10 (iter-347): hybrid F16-K + TQ-HB-V — Option
            // sibling, default `None` until lazy-allocated by the env-gated
            // path in `forward_decode` (Phase 10c).
            hybrid_kv: None,
            // ADR-007 Gate H release-check counter — increments per
            // forward_decode call, used by the `[HF2Q_NLL]` / `[HF2Q_DECODE_EMIT]`
            // stderr lines (W12 iter-108a blocker #1).
            decode_step: 0,
            // ADR-007 Gate H per-call regime override (W12 iter-108a blocker #3).
            // Default == today's env-var-only path; setter flips it for two-
            // regime-one-process release-check runs.
            decode_regime: DecodeRegime::Default,
            // iter-108a-fix (W15, 2026-04-25): cache the "Gate H inactive"
            // predicate so the decode hot path can elide the per-token
            // NLL/emit/replay block AND the per-layer regime-match site
            // when no Gate H hooks are armed. INVESTIGATION_ENV is a
            // process-lifetime LazyLock so this snapshot stays valid
            // until set_decode_regime is called (which refreshes it).
            gate_h_inactive: {
                let env = &*INVESTIGATION_ENV;
                !env.emit_nll
                    && !env.decode_emit_tokens
                    && env.decode_input_tokens.is_empty()
                // decode_regime is Default at construction, so the regime
                // arm is true here without an extra read.
                // replay_tokens is empty at construction (default Vec::new()),
                // so it does not flip gate_h_inactive here either.
            },
            // W21 iter-108b: per-instance replay vector for the in-process
            // two-regime Gate H run.  Empty by default → no behavior change
            // from iter-108a; populated only via [`set_replay_tokens`].
            replay_tokens: Vec::new(),
            // W39 iter-112b: per-instance dump-config overrides.  None by
            // default → SDPA-out dump gate falls back to INVESTIGATION_ENV,
            // bit-identical to pre-iter-112b.  parity_quality sets these
            // before each Gate H pass so the dumps land in the per-pass
            // dir even though INVESTIGATION_ENV's LazyLock is frozen.
            dump_dir_override: None,
            dump_all_cache_override: None,
            // W39 iter-112b: per-instance decode-step dump counter.  Replaces
            // the old process-static AtomicUsize so the SDPA dump gate's
            // [0, max_pos) window resets at the start of each Gate H pass.
            decode_step_dump_counter: 0,
            // ADR-030 Phase 4 — capture session NOT installed by default.
            // Spec-decode orchestrator installs via install_dflash_capture()
            // before calling forward_prefill_batched.
            dflash_capture: None,
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

            // ADR-029 iter-28 H29 / iter-31 — F16 shadow population pass.
            //
            // Materializes an F16 pre-dequantized buffer for every attn
            // and dense-MLP quantized weight in every layer, so that the
            // runtime dispatch_qmatmul fast-paths through the F16-input
            // matmul kernel (peer's gemma4 strategy).
            //
            // iter-31 default-flip: HF2Q_F16_SHADOW now default-true
            // (opt-out via =0/false/off).  Multi-regime bench (4 ctxs ×
            // 3 trials each, gemma4-APEX-Q5_K_M):
            //   2K prefill: +16.0%
            //   4K prefill: +7.3%
            //   8K prefill: +1.9%
            //   decode m=1: unaffected (V2 path gated on m > 8, H29 too)
            // Byte-identical first decode tokens at every context.
            // ~1 GB extra resident on gemma4-26B; 128 GB M5 Max budget
            // accommodates without pressure.  0.4s load-time pass.
            //
            // Doing this in a second pass (after weight load) avoids
            // borrow-checker conflicts between `gpu.device()` (read) and
            // `gpu.registry` (write) during the per-layer load loop.
            let f16_shadow_enabled = std::env::var("HF2Q_F16_SHADOW")
                .ok()
                .map(|v| !matches!(v.as_str(), "0" | "false" | "off"))
                .unwrap_or(true);
            if f16_shadow_enabled {
                let dev = gpu.executor.device();
                let n_layers = w.layers.len();
                eprintln!("[ADR-029 H29] Materializing F16 shadows for {} layers' attn + dense MLP weights...", n_layers);
                let t0 = std::time::Instant::now();
                for li in 0..n_layers {
                    populate_f16_shadow_if_enabled(
                        &mut w.layers[li].attn.q_proj, dev, &mut gpu.registry,
                        &format!("blk.{li}.attn_q"))?;
                    populate_f16_shadow_if_enabled(
                        &mut w.layers[li].attn.k_proj, dev, &mut gpu.registry,
                        &format!("blk.{li}.attn_k"))?;
                    if let Some(ref mut v) = w.layers[li].attn.v_proj {
                        populate_f16_shadow_if_enabled(
                            v, dev, &mut gpu.registry,
                            &format!("blk.{li}.attn_v"))?;
                    }
                    populate_f16_shadow_if_enabled(
                        &mut w.layers[li].attn.o_proj, dev, &mut gpu.registry,
                        &format!("blk.{li}.attn_output"))?;
                    populate_f16_shadow_if_enabled(
                        &mut w.layers[li].mlp.gate_proj, dev, &mut gpu.registry,
                        &format!("blk.{li}.ffn_gate"))?;
                    populate_f16_shadow_if_enabled(
                        &mut w.layers[li].mlp.up_proj, dev, &mut gpu.registry,
                        &format!("blk.{li}.ffn_up"))?;
                    populate_f16_shadow_if_enabled(
                        &mut w.layers[li].mlp.down_proj, dev, &mut gpu.registry,
                        &format!("blk.{li}.ffn_down"))?;
                }
                let elapsed = t0.elapsed();
                eprintln!("[ADR-029 H29] F16 shadow population done in {:.2}s", elapsed.as_secs_f64());
            }
        }

        result
    }

    /// Set the decode regime for the next prefill+decode trajectory.
    ///
    /// ADR-007 Gate H (W12 iter-108a blocker #3) — flips the SDPA-mode
    /// gate at `forward_mlx.rs::forward_decode` between TQ-active and
    /// dense-active without re-loading the model from GGUF.  The four
    /// codebook-bits gates (`forward_mlx.rs:1100/1234`, `forward_prefill
    /// .rs:330`) are *not* affected by this setter — the codebook width
    /// is a representation choice that stays consistent across both
    /// regimes (Gate H runs both regimes on the same KV format).  Only
    /// the SDPA-reader (the `use_dense_sdpa` gate) consults the override.
    ///
    /// Calling this also resets the per-instance step counter so each
    /// regime's stderr `[HF2Q_NLL]` / `[HF2Q_DECODE_EMIT]` lines start
    /// at `step=0`, matching the audit-binary contract (every audit
    /// invocation is a fresh process today).
    ///
    /// Default-mode behavior (i.e. `regime == DecodeRegime::Default`)
    /// is *byte-identical* to today's env-var-only path — see the
    /// `forward_mlx.rs::forward_decode` use_dense_sdpa gate for the
    /// invariant.  Non-Default regimes ignore `HF2Q_USE_DENSE` and
    /// `HF2Q_LAYER_POLICY` for the duration of the next decode loop.
    ///
    /// (Wired by iter-108b's release-check.sh Gate 5 harness; no in-tree
    /// caller as of iter-108a — the surface is designed for the
    /// iter-108b two-regime release-check entry point.)
    /// ADR-020 AC#5 Iter D — overlay a DWQ-trained mlx-affine safetensors
    /// file on top of an already-GGUF-loaded model.  For each Linear
    /// stem present in the safetensors file (`<stem>.weight`/.scales/.biases`
    /// triplet), the matching slot in `MlxModelWeights` is replaced by
    /// an affine-mode `MlxQWeight` (per Iter B + Iter C dispatch routing).
    ///
    /// The safetensors `bits` + `group_size` are read from the file's
    /// metadata (embedded by `train_all_linears_dwq` since AC#5 Iter D).
    /// Older DWQ safetensors without metadata fall back to bits=4,
    /// group_size=32 (the production default).
    ///
    /// Stem mapping (dense layers only — MoE expert tensors are skipped
    /// with a warning, tracked as Iter C2):
    ///
    /// | Stem | Slot |
    /// |---|---|
    /// | `blk.{i}.attn_q` | `layers[i].attn.q_proj` |
    /// | `blk.{i}.attn_k` | `layers[i].attn.k_proj` |
    /// | `blk.{i}.attn_v` | `layers[i].attn.v_proj` (if Some) |
    /// | `blk.{i}.attn_output` | `layers[i].attn.o_proj` |
    /// | `blk.{i}.ffn_gate` | `layers[i].mlp.gate_proj` |
    /// | `blk.{i}.ffn_up` | `layers[i].mlp.up_proj` |
    /// | `blk.{i}.ffn_down` | `layers[i].mlp.down_proj` |
    ///
    /// Returns the count of overridden Linears.  Logs each unmatched
    /// stem at `tracing::warn!` so operators can audit which trained
    /// tensors were ignored.
    pub fn apply_dwq_overlay(
        &mut self,
        device: &MlxDevice,
        path: &std::path::Path,
    ) -> Result<usize> {
        use crate::calibrate::mlx_safetensors_loader::MlxAffineLinear;
        use anyhow::Context;

        let bytes = std::fs::read(path)
            .with_context(|| format!("apply_dwq_overlay: read {}", path.display()))?;

        // Pull metadata via `read_metadata` (the deserialized SafeTensors
        // hides its `Metadata` field; `read_metadata` is the public path).
        let (_n, metadata_obj) = safetensors::SafeTensors::read_metadata(&bytes)
            .map_err(|e| anyhow::anyhow!("apply_dwq_overlay: read_metadata: {e:?}"))?;

        let (bits, group_size) = parse_dwq_overlay_metadata(metadata_obj.metadata().as_ref())
            .with_context(|| format!("apply_dwq_overlay: parse metadata of {}", path.display()))?;

        let st = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| anyhow::anyhow!("apply_dwq_overlay: deserialize safetensors: {e:?}"))?;

        // Walk all `<stem>.weight` keys; require `<stem>.scales` +
        // `<stem>.biases` to be present too.
        let mut stems: Vec<String> = Vec::new();
        for name in st.names() {
            if let Some(stem) = name.strip_suffix(".weight") {
                if st.tensor(&format!("{stem}.scales")).is_err() {
                    continue;
                }
                if st.tensor(&format!("{stem}.biases")).is_err() {
                    continue;
                }
                stems.push(stem.to_string());
            }
        }

        let mut overridden: usize = 0;
        let mut unknown_skipped: usize = 0;

        // ADR-020 AC#5 Iter C2.2 — stage MoE per-expert linears for
        // post-pass aggregation.  Key: `(layer_idx, MoeBaseRole)`,
        // value: Vec<(expert_idx, MlxAffineLinear)>.
        type MoeBucket = std::collections::HashMap<
            (usize, MoeBaseRole),
            Vec<(usize, MlxAffineLinear)>,
        >;
        let mut moe_buckets: MoeBucket = std::collections::HashMap::new();

        for stem in &stems {
            let linear = MlxAffineLinear::from_safetensors(&st, stem, bits, group_size)
                .with_context(|| format!("apply_dwq_overlay: parse {stem}"))?;

            // Match `blk.{i}.<role>` patterns.
            let after_blk = match stem.strip_prefix("blk.") {
                Some(s) => s,
                None => {
                    tracing::warn!(stem = %stem, "DWQ overlay: stem does not start with 'blk.'; skipping");
                    unknown_skipped += 1;
                    continue;
                }
            };
            let dot = match after_blk.find('.') {
                Some(d) => d,
                None => {
                    tracing::warn!(stem = %stem, "DWQ overlay: stem missing '.<role>'; skipping");
                    unknown_skipped += 1;
                    continue;
                }
            };
            let layer_idx: usize = match after_blk[..dot].parse() {
                Ok(v) => v,
                Err(_) => {
                    tracing::warn!(stem = %stem, "DWQ overlay: layer idx not numeric; skipping");
                    unknown_skipped += 1;
                    continue;
                }
            };
            if layer_idx >= self.layers.len() {
                tracing::warn!(stem = %stem, layer = layer_idx, "DWQ overlay: layer idx out of range; skipping");
                unknown_skipped += 1;
                continue;
            }
            let role = &after_blk[(dot + 1)..];
            match parse_dwq_overlay_role(role) {
                DwqOverlayRole::AttnQ => {
                    self.layers[layer_idx].attn.q_proj =
                        MlxQWeight::from_mlx_affine_linear(device, &linear)
                            .with_context(|| format!("apply_dwq_overlay: build qweight for {stem}"))?;
                }
                DwqOverlayRole::AttnK => {
                    self.layers[layer_idx].attn.k_proj =
                        MlxQWeight::from_mlx_affine_linear(device, &linear)
                            .with_context(|| format!("apply_dwq_overlay: build qweight for {stem}"))?;
                }
                DwqOverlayRole::AttnV => {
                    if self.layers[layer_idx].attn.v_proj.is_some() {
                        self.layers[layer_idx].attn.v_proj = Some(
                            MlxQWeight::from_mlx_affine_linear(device, &linear)
                                .with_context(|| format!("apply_dwq_overlay: build qweight for {stem}"))?,
                        );
                    } else {
                        tracing::warn!(stem = %stem, "DWQ overlay: attn_v but slot is None (k_eq_v); skipping");
                        unknown_skipped += 1;
                        continue;
                    }
                }
                DwqOverlayRole::AttnOutput => {
                    self.layers[layer_idx].attn.o_proj =
                        MlxQWeight::from_mlx_affine_linear(device, &linear)
                            .with_context(|| format!("apply_dwq_overlay: build qweight for {stem}"))?;
                }
                DwqOverlayRole::FfnGate => {
                    self.layers[layer_idx].mlp.gate_proj =
                        MlxQWeight::from_mlx_affine_linear(device, &linear)
                            .with_context(|| format!("apply_dwq_overlay: build qweight for {stem}"))?;
                }
                DwqOverlayRole::FfnUp => {
                    self.layers[layer_idx].mlp.up_proj =
                        MlxQWeight::from_mlx_affine_linear(device, &linear)
                            .with_context(|| format!("apply_dwq_overlay: build qweight for {stem}"))?;
                }
                DwqOverlayRole::FfnDown => {
                    self.layers[layer_idx].mlp.down_proj =
                        MlxQWeight::from_mlx_affine_linear(device, &linear)
                            .with_context(|| format!("apply_dwq_overlay: build qweight for {stem}"))?;
                }
                DwqOverlayRole::MoeExpert => {
                    if let Some((base, expert_idx)) = parse_dwq_moe_expert_role(role) {
                        moe_buckets
                            .entry((layer_idx, base))
                            .or_default()
                            .push((expert_idx, linear));
                    } else {
                        tracing::warn!(stem = %stem, role = %role, "DWQ overlay: malformed MoE expert stem; skipping");
                        unknown_skipped += 1;
                    }
                    continue;
                }
                DwqOverlayRole::Unknown => {
                    tracing::warn!(stem = %stem, role = %role, "DWQ overlay: unknown role; skipping");
                    unknown_skipped += 1;
                    continue;
                }
            }
            overridden += 1;
            tracing::debug!(stem = %stem, "DWQ overlay applied (dense)");
        }

        // ADR-020 AC#5 Iter C2.2 — second pass: aggregate per-expert
        // bucketed Linears into MlxAffineMoeStack and assign to the
        // matching MoE slot.  Verifies expert indices form a contiguous
        // 0..n_experts range with consistent shape across experts.
        let mut moe_stacked: usize = 0;
        for ((layer_idx, base), mut linears) in moe_buckets.into_iter() {
            // Sort by expert idx, dedup, validate contiguous 0..n_experts.
            linears.sort_by_key(|(e, _)| *e);
            let n_experts = linears.len();
            for (i, (e, _)) in linears.iter().enumerate() {
                if *e != i {
                    anyhow::bail!(
                        "DWQ overlay MoE bucket (layer={layer_idx}, base={:?}) has non-contiguous expert idx (got {} at slot {})",
                        base,
                        e,
                        i,
                    );
                }
            }
            // All experts share shape — validate from the first.
            let n = linears[0].1.n;
            let k = linears[0].1.k;
            let bits_per = linears[0].1.bits;
            let gs_per = linears[0].1.group_size;
            for (e, l) in &linears[1..] {
                if l.n != n || l.k != k || l.bits != bits_per || l.group_size != gs_per {
                    anyhow::bail!(
                        "DWQ overlay MoE bucket (layer={layer_idx}, base={:?}) expert {} shape ({},{},bits={},gs={}) ≠ expert 0 ({},{},bits={},gs={})",
                        base, e,
                        l.n, l.k, l.bits, l.group_size,
                        n, k, bits_per, gs_per,
                    );
                }
            }
            if bits_per != 4 || gs_per != 32 {
                anyhow::bail!(
                    "DWQ overlay MoE bucket (layer={layer_idx}, base={:?}): only bits=4 group_size=32 supported in Iter C2.2 (got bits={}, gs={})",
                    base, bits_per, gs_per,
                );
            }
            let pack_factor = 32 / bits_per as usize;
            let k_packed = k / pack_factor;
            let groups_per_row = k / (gs_per as usize);

            // Pack each expert's q_int → U32 and convert F32 → BF16
            // for scales/biases (the `quantized_matmul_id` kernel's
            // native dtype, mirroring mlx-lm's BF16 on-disk convention).
            let stack_words = n_experts * n * k_packed;
            let mut packed_stack: Vec<u32> = vec![0u32; stack_words];
            let mut scales_stack_bf16: Vec<u16> =
                vec![0u16; n_experts * n * groups_per_row];
            let mut biases_stack_bf16: Vec<u16> =
                vec![0u16; n_experts * n * groups_per_row];
            for (e, lin) in &linears {
                for row in 0..n {
                    for kp in 0..k_packed {
                        let mut word: u32 = 0;
                        for j in 0..pack_factor {
                            let code = lin.q_int[row * k + kp * pack_factor + j] as u32;
                            debug_assert!(code <= 0xF);
                            word |= (code & 0xF) << (j * 4);
                        }
                        packed_stack[((*e * n) + row) * k_packed + kp] = word;
                    }
                }
                let s_offset = e * n * groups_per_row;
                for (i, v) in lin.scales.iter().enumerate() {
                    scales_stack_bf16[s_offset + i] = half::bf16::from_f32(*v).to_bits();
                }
                for (i, v) in lin.biases.iter().enumerate() {
                    biases_stack_bf16[s_offset + i] = half::bf16::from_f32(*v).to_bits();
                }
            }

            // Allocate GPU buffers + upload.
            let mut weight_buf = device
                .alloc_buffer(
                    stack_words * std::mem::size_of::<u32>(),
                    mlx_native::DType::U32,
                    vec![n_experts, n, k_packed],
                )
                .map_err(|e| anyhow::anyhow!("MoE stack weight alloc: {e}"))?;
            weight_buf
                .as_mut_slice::<u32>()
                .map_err(|e| anyhow::anyhow!("MoE stack weight slice: {e}"))?
                .copy_from_slice(&packed_stack);

            let mut scales_buf = device
                .alloc_buffer(
                    scales_stack_bf16.len() * std::mem::size_of::<u16>(),
                    mlx_native::DType::BF16,
                    vec![n_experts, n, groups_per_row],
                )
                .map_err(|e| anyhow::anyhow!("MoE stack scales alloc: {e}"))?;
            scales_buf
                .as_mut_slice::<u16>()
                .map_err(|e| anyhow::anyhow!("MoE stack scales slice: {e}"))?
                .copy_from_slice(&scales_stack_bf16);

            let mut biases_buf = device
                .alloc_buffer(
                    biases_stack_bf16.len() * std::mem::size_of::<u16>(),
                    mlx_native::DType::BF16,
                    vec![n_experts, n, groups_per_row],
                )
                .map_err(|e| anyhow::anyhow!("MoE stack biases alloc: {e}"))?;
            biases_buf
                .as_mut_slice::<u16>()
                .map_err(|e| anyhow::anyhow!("MoE stack biases slice: {e}"))?
                .copy_from_slice(&biases_stack_bf16);

            let stack = MlxAffineMoeStack {
                weight: weight_buf,
                scales: scales_buf,
                biases: biases_buf,
                n,
                k,
                bits: bits_per,
                group_size: gs_per as u32,
                num_experts: n_experts,
            };
            let layer = &mut self.layers[layer_idx];
            match base {
                MoeBaseRole::GateUp => {
                    layer.moe.gate_up_affine = Some(stack);
                }
                MoeBaseRole::Down => {
                    layer.moe.down_affine = Some(stack);
                }
                MoeBaseRole::Gate | MoeBaseRole::Up => {
                    // Separate gate / up case — not yet wired into a
                    // dispatch path (Iter C2.3 only routes the FUSED
                    // gate_up case for qwen3.5).  Surface as warning;
                    // operator can revisit when a non-fused MoE GGUF
                    // arch shows up.
                    tracing::warn!(
                        layer_idx,
                        ?base,
                        n_experts,
                        "DWQ overlay: separate gate/up MoE case not wired to dispatch yet (qwen3.5 uses fused gate_up); stack constructed but unused"
                    );
                    let _ = stack;
                }
            }
            moe_stacked += 1;
            tracing::debug!(
                layer_idx,
                ?base,
                n_experts,
                n,
                k,
                "DWQ overlay applied (MoE stack)"
            );
        }

        tracing::info!(
            overridden,
            moe_stacked,
            unknown_skipped,
            bits,
            group_size,
            "DWQ overlay applied: {overridden} dense Linears + {moe_stacked} MoE stacks"
        );
        Ok(overridden + moe_stacked)
    }

    #[allow(dead_code)]
    pub fn set_decode_regime(&mut self, regime: DecodeRegime) {
        self.decode_regime = regime;
        self.decode_step = 0;
        // W39 iter-112b: also reset the per-instance SDPA-dump step
        // counter so each Gate H regime's [0, max_pos) dump window
        // restarts at 0 (the old process-static AtomicUsize accumulated
        // across passes and silently dropped pass 2's dumps).
        self.decode_step_dump_counter = 0;
        // iter-108a-fix (W15): re-evaluate the Gate H elision flag.
        // Non-Default regime forces the per-layer SDPA-gate match path,
        // so we can't elide it; the env-var hooks could still be off,
        // but we only treat the path as "inactive" when ALL Gate H
        // surfaces are quiet (env hooks unset AND regime is Default).
        let env = &*INVESTIGATION_ENV;
        self.gate_h_inactive = matches!(regime, DecodeRegime::Default)
            && !env.emit_nll
            && !env.decode_emit_tokens
            && env.decode_input_tokens.is_empty()
            && self.replay_tokens.is_empty();
    }

    /// Set the per-instance replay vector for the next decode trajectory
    /// (ADR-007 Gate H, W21 iter-108b two-regime in-process harness).
    ///
    /// When non-empty, the post-argmax tail of [`Self::forward_decode`]
    /// substitutes `replay[step]` for the model's argmax pick — same
    /// contract as `HF2Q_DECODE_INPUT_TOKENS` but bypassing the
    /// `INVESTIGATION_ENV` `LazyLock` (which is frozen at first access
    /// and so cannot be flipped between the dense and TQ passes of a
    /// single Gate H run).  After the replay buffer is exhausted, the
    /// loop falls through to the live argmax pick — identical fall-back
    /// to the env-var path.
    ///
    /// Pass an empty `Vec` to clear the override.  Also resets
    /// [`Self::decode_step`] (matching `set_decode_regime` semantics) so
    /// each replay run's `step` counter starts at 0, and refreshes
    /// [`Self::gate_h_inactive`] so the per-token instrumentation block
    /// runs whenever a replay is active even when env hooks are silent.
    ///
    /// (Wired by iter-108b's `parity_quality::run_two_regime_decode`;
    /// no other in-tree caller.)
    #[allow(dead_code)]
    pub fn set_replay_tokens(&mut self, replay: Vec<u32>) {
        self.replay_tokens = replay;
        self.decode_step = 0;
        // W39 iter-112b: see `set_decode_regime` — reset the SDPA-dump
        // step counter for the upcoming pass.
        self.decode_step_dump_counter = 0;
        let env = &*INVESTIGATION_ENV;
        self.gate_h_inactive = matches!(self.decode_regime, DecodeRegime::Default)
            && !env.emit_nll
            && !env.decode_emit_tokens
            && env.decode_input_tokens.is_empty()
            && self.replay_tokens.is_empty();
    }

    /// Set per-instance SDPA-dump overrides for the next decode trajectory
    /// (ADR-007 Gate H, W39 iter-112b two-regime in-process harness).
    ///
    /// `INVESTIGATION_ENV` is a `LazyLock` populated by
    /// `INVESTIGATION_ENV.activate()` at `main.rs::main` *before* `Cli::parse`.
    /// W21's `parity_quality::run_two_regime_decode` sets `HF2Q_DUMP_DIR` and
    /// `HF2Q_DUMP_ALL_CACHE` at run time, but those `set_var` calls reach
    /// `std::env` *after* the LazyLock has frozen, so the SDPA-out dump gate
    /// at `forward_decode` and the dump-path formatter inside `dumps::dump_f32`
    /// keep reading the pre-launch (default) values — `dump_all_cache=false`
    /// and `dump_dir=/tmp` — silently dropping every Gate H dump.
    ///
    /// This setter exposes the per-instance override surface.  Both
    /// arguments are `Option`: `Some(_)` overrides the corresponding
    /// `INVESTIGATION_ENV` field for SDPA-out gating + path formation;
    /// `None` falls back to `INVESTIGATION_ENV` (i.e. the default
    /// env-var-only path is bit-identical to pre-iter-112b).
    ///
    /// Also resets [`Self::decode_step_dump_counter`] so the upcoming pass's
    /// `[0, max_pos)` dump window starts at step 0.
    ///
    /// (Wired by iter-112b's `parity_quality::run_two_regime_decode`; no
    /// other in-tree caller.)
    #[allow(dead_code)]
    pub fn set_dump_overrides(
        &mut self,
        dir: Option<std::path::PathBuf>,
        all_cache: Option<bool>,
    ) {
        self.dump_dir_override = dir;
        self.dump_all_cache_override = all_cache;
        self.decode_step_dump_counter = 0;
    }

    /// Reset the per-instance SDPA-dump step counter without touching
    /// regime / replay / override state.  W39 iter-112b: most Gate H call
    /// sites reset via `set_decode_regime` / `set_replay_tokens` /
    /// `set_dump_overrides`; this is the explicit setter for callers that
    /// only want to roll the counter back (e.g. between sub-passes within
    /// the same regime).
    #[allow(dead_code)]
    pub fn reset_decode_step_dump_counter(&mut self) {
        self.decode_step_dump_counter = 0;
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
    /// ADR-028 iter-391: stub method for the upcoming layer-body extraction.
    /// Currently UNUSED (forward_decode keeps inline layer loop).  iter-392+
    /// will incrementally populate this method with code from the existing
    /// layer loop body, then switch forward_decode to call it instead.
    ///
    /// When complete, this method will be the unit of work that the
    /// EncoderWorker thread runs in parallel with the main thread for the
    /// second-half of layer encoding.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_one_layer<'sess>(
        &self,
        layer_idx: usize,
        ctx: &super::layer_ctx::LayerCtx<'_>,
        session: &mut mlx_native::graph::GraphSession<'sess>,
        exec: &'sess mlx_native::GraphExecutor,
        reg: &mut mlx_native::KernelRegistry,
        profile: &mut Option<TokenProfile>,
        per_layer_disp_log: &mut Vec<(usize, bool, u64)>,
        total_dispatches: &mut usize,
    ) -> Result<()> {
        let dev = exec.device();
        let metal_dev = dev.metal_device();
        let hs = ctx.hidden_size;
        let seq_pos = ctx.seq_pos;
        let dump_layers = ctx.dump_layers;
        let dump_detail_layer = ctx.dump_detail_layer;
        let dump_sliding_l0 = ctx.dump_sliding_l0;
        let dump_run_name = ctx.dump_run_name;
        let dual_buffer_splits = ctx.dual_buffer_splits;
        let per_layer_disp_enabled = ctx.per_layer_disp_enabled;
                let layer_disp_start = if per_layer_disp_enabled {
                    mlx_native::dispatch_count()
                } else { 0 };
                let hd = self.layers[layer_idx].head_dim;
                let nkv = self.layers[layer_idx].num_kv_heads;
                let nh = self.num_attention_heads;
                let is_sliding = self.layers[layer_idx].layer_type == LayerType::Sliding;
                let eps = self.rms_norm_eps;
                let (kv_is_sliding, kv_write_pos, kv_capacity, kv_seq_len) = ctx.kv_info[layer_idx];

                // -- Pre-attention norm (GPU) --
                session.barrier_between(
                    &[&self.activations.hidden, &self.layers[layer_idx].norms.input_layernorm],
                    &[&self.activations.norm_out],
                );
                session.rms_norm(
                    reg, metal_dev,
                    &self.activations.hidden,
                    &self.layers[layer_idx].norms.input_layernorm,
                    &self.activations.norm_out,
                    &self.activations.norm_params,
                    1, hs as u32,
                ).map_err(|e| anyhow::anyhow!("GPU pre-attn norm L{layer_idx}: {e}"))?;
                *total_dispatches += 1;

                // -- QKV projections (CONCURRENT: all read norm_out, write separate buffers) --
                // ONE barrier after norm (which wrote norm_out), then all 3 projections
                // dispatch without barriers between them — they share reads and have disjoint writes.
                session.barrier_between(
                    &[&self.activations.norm_out],
                    &[&self.activations.attn_q, &self.activations.attn_k, &self.activations.attn_v],
                );
                // ADR-028 iter-210: SKIP_ATTN_QKV bisect — skip Q/K/V
                // qmatmul dispatches.  Concurrent ops; their max time
                // is the sequential cost on critical path.  Garbage
                // attention output downstream.
                if !INVESTIGATION_ENV.skip_attn_qkv {
                    dispatch_qmatmul(session, reg, dev, &self.activations.norm_out,
                        &self.layers[layer_idx].attn.q_proj, &self.activations.attn_q, 1)?;
                    *total_dispatches += 1;
                    // Per-dispatch range annotation for the reorder pass. The
                    // single barrier_between above only annotates the first
                    // dispatch; concurrent K and V need their own ranges.
                    dispatch_qmatmul(session, reg, dev, &self.activations.norm_out,
                        &self.layers[layer_idx].attn.k_proj, &self.activations.attn_k, 1)?;
                    session.track_dispatch(&[&self.activations.norm_out], &[&self.activations.attn_k]);
                    *total_dispatches += 1;
                }
                let v_is_k = self.layers[layer_idx].attn.v_proj.is_none();
                if !v_is_k && !INVESTIGATION_ENV.skip_attn_qkv {
                    dispatch_qmatmul(session, reg, dev, &self.activations.norm_out,
                        self.layers[layer_idx].attn.v_proj.as_ref().unwrap(),
                        &self.activations.attn_v, 1)?;
                    session.track_dispatch(&[&self.activations.norm_out], &[&self.activations.attn_v]);
                    *total_dispatches += 1;
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
                session.barrier_between(
                    &[&self.activations.attn_q, &self.activations.attn_k],
                    &[&self.activations.attn_q_normed, &self.activations.attn_k_normed],
                );
                // ADR-028 iter-204: SKIP_HEAD_NORM_ROPE bisect — skip
                // both Q-norm-rope and K-norm-rope dispatches.  Produces
                // garbage SDPA (attn_q_normed/attn_k_normed stale).
                if !INVESTIGATION_ENV.skip_head_norm_rope {
                    mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32(
                        session.encoder_mut(), reg, metal_dev,
                        &self.activations.attn_q,
                        &self.activations.attn_q_normed,
                        Some(&self.layers[layer_idx].attn.q_norm_weight),
                        &self.activations.position,
                        ff_gpu,
                        nh as u32, hd as u32, half_rope,
                        eps, theta,
                    ).map_err(|e| anyhow::anyhow!("fused Q norm+RoPE L{layer_idx}: {e}"))?;
                    *total_dispatches += 1;
                    mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32(
                        session.encoder_mut(), reg, metal_dev,
                        &self.activations.attn_k,
                        &self.activations.attn_k_normed,
                        Some(&self.layers[layer_idx].attn.k_norm_weight),
                        &self.activations.position,
                        ff_gpu,
                        nkv as u32, hd as u32, half_rope,
                        eps, theta,
                    ).map_err(|e| anyhow::anyhow!("fused K norm+RoPE L{layer_idx}: {e}"))?;
                    *total_dispatches += 1;
                }

                // GPU V norm
                let hd_norm_params = if is_sliding {
                    &self.activations.norm_params_sliding_hd
                } else {
                    &self.activations.norm_params_global_hd
                };
                // ADR-028 iter-214: SKIP_V_NORM bisect.  V-norm output
                // is consumed passively by KV-copy + SDPA — no control
                // signal confound.
                if v_is_k && !INVESTIGATION_ENV.skip_v_norm {
                    session.barrier_between(
                        &[&self.activations.attn_k],
                        &[&self.activations.attn_v],
                    );
                    dispatch_rms_norm_unit_perhead(
                        session.encoder_mut(), reg, metal_dev,
                        &RmsNormPerHeadArgs {
                            input: &self.activations.attn_k,
                            output: &self.activations.attn_v,
                            params_buf: hd_norm_params,
                            rows: nkv as u32,
                            dim: hd as u32,
                        },
                    )?;
                    *total_dispatches += 1;
                } else if !v_is_k && !INVESTIGATION_ENV.skip_v_norm {
                    session.barrier_between(
                        &[&self.activations.attn_v],
                        &[&self.activations.moe_expert_out],
                    );
                    dispatch_rms_norm_unit_perhead(
                        session.encoder_mut(), reg, metal_dev,
                        &RmsNormPerHeadArgs {
                            input: &self.activations.attn_v,
                            output: &self.activations.moe_expert_out,
                            params_buf: hd_norm_params,
                            rows: nkv as u32,
                            dim: hd as u32,
                        },
                    )?;
                    *total_dispatches += 1;
                }

                let v_src = if v_is_k {
                    &self.activations.attn_v
                } else {
                    &self.activations.moe_expert_out
                };

                // ADR-007 C-2: pre-hadamard_quantize K/V dump (independent-floor oracle inputs).
                // Gate: dump_pre_quant && layer_idx == 0 && kv_seq_len == 23.
                // Fires BEFORE dispatch_hadamard_quantize_kv — captures raw F32 K (attn_k_normed)
                // and V (attn_v or moe_expert_out) at the exact moment before TQ encode.
                // Category-4 read-only diagnostic; no HF2Q_UNSAFE_EXPERIMENTS ack required.
                // Path C F-0.3 generalization: if HF2Q_DUMP_PRE_QUANT_LAYERS or
                // HF2Q_DUMP_PRE_QUANT_POSITIONS is set, fire at every matching
                // (layer, kv_seq_len) pair and write per-(layer, position) files
                // named L{layer:02}_p{pos:04}_{k,v}_pre_quant.f32.bin. Otherwise
                // preserve legacy single-file behavior at L0 / kv_seq_len=23.
                let pre_quant_layers_filter = &INVESTIGATION_ENV.dump_pre_quant_layers;
                let pre_quant_positions_filter = &INVESTIGATION_ENV.dump_pre_quant_positions;
                let pre_quant_extended = !pre_quant_layers_filter.is_empty()
                    || !pre_quant_positions_filter.is_empty();
                let layer_match = if pre_quant_extended {
                    pre_quant_layers_filter.is_empty()
                        || pre_quant_layers_filter.contains(&layer_idx)
                } else {
                    layer_idx == 0
                };
                let pos_match = if pre_quant_extended {
                    pre_quant_positions_filter.is_empty()
                        || pre_quant_positions_filter.contains(&kv_seq_len)
                } else {
                    kv_seq_len == 23
                };
                if INVESTIGATION_ENV.dump_pre_quant && layer_match && pos_match {
                    std::mem::replace(session, exec.begin()
            .map_err(|e| anyhow::anyhow!("pre_quant dump re-begin: {e}"))?).finish()
                        .map_err(|e| anyhow::anyhow!("pre_quant dump finish L{layer_idx}: {e}"))?;
                    let dump_dir = &INVESTIGATION_ENV.dump_dir;
                    let pre_quant_dir = format!("{dump_dir}/pre_quant");
                    std::fs::create_dir_all(&pre_quant_dir)
                        .map_err(|e| anyhow::anyhow!("pre_quant mkdir: {e}"))?;

                    // Filenames: legacy = `k_pre_quant.f32.bin`; extended =
                    // `L{layer:02}_p{kv_seq_len:04}_k_pre_quant.f32.bin`.
                    let (k_fname, v_fname, meta_fname) = if pre_quant_extended {
                        (
                            format!("L{:02}_p{:04}_k_pre_quant.f32.bin", layer_idx, kv_seq_len),
                            format!("L{:02}_p{:04}_v_pre_quant.f32.bin", layer_idx, kv_seq_len),
                            format!("L{:02}_p{:04}_meta.json", layer_idx, kv_seq_len),
                        )
                    } else {
                        (
                            "k_pre_quant.f32.bin".to_string(),
                            "v_pre_quant.f32.bin".to_string(),
                            "meta.json".to_string(),
                        )
                    };

                    // K pre-quant [nkv, hd] F32 little-endian
                    {
                        let k_raw: &[f32] = self.activations.attn_k_normed.as_slice()
                            .map_err(|e| anyhow::anyhow!("pre_quant k_normed read: {e}"))?;
                        let n_elems = nkv * hd;
                        let k_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                k_raw.as_ptr() as *const u8,
                                n_elems * std::mem::size_of::<f32>(),
                            )
                        };
                        let kp = format!("{pre_quant_dir}/{k_fname}");
                        std::fs::write(&kp, k_bytes)
                            .map_err(|e| anyhow::anyhow!("write {kp}: {e}"))?;
                        eprintln!("[PRE_QUANT_DUMP] L{layer_idx} p{kv_seq_len} k_pre_quant [{nkv},{hd}] f32 -> {kp}");
                    }

                    // V pre-quant [nkv, hd] F32 little-endian
                    {
                        let v_raw: &[f32] = v_src.as_slice()
                            .map_err(|e| anyhow::anyhow!("pre_quant v_src read: {e}"))?;
                        let n_elems = nkv * hd;
                        let v_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                v_raw.as_ptr() as *const u8,
                                n_elems * std::mem::size_of::<f32>(),
                            )
                        };
                        let vp = format!("{pre_quant_dir}/{v_fname}");
                        std::fs::write(&vp, v_bytes)
                            .map_err(|e| anyhow::anyhow!("write {vp}: {e}"))?;
                        eprintln!("[PRE_QUANT_DUMP] L{layer_idx} p{kv_seq_len} v_pre_quant [{nkv},{hd}] f32 -> {vp}");
                    }

                    // meta.json sidecar with provenance
                    {
                        let cache_pos_at_dump = if kv_is_sliding {
                            (kv_write_pos % kv_capacity) as u32
                        } else {
                            kv_write_pos as u32
                        };
                        let meta = serde_json::json!({
                            "site": "pre_hadamard_quantize_kv",
                            "layer_idx": layer_idx,
                            "kv_seq_len": kv_seq_len,
                            "cache_pos_val": cache_pos_at_dump,
                            "nkv": nkv,
                            "hd": hd,
                            "kv_is_sliding": kv_is_sliding,
                            "k_pre_quant_shape": [nkv, hd],
                            "v_pre_quant_shape": [nkv, hd],
                        });
                        let meta_str = serde_json::to_string_pretty(&meta)
                            .map_err(|e| anyhow::anyhow!("pre_quant meta json: {e}"))?;
                        let mp = format!("{pre_quant_dir}/{meta_fname}");
                        std::fs::write(&mp, meta_str.as_bytes())
                            .map_err(|e| anyhow::anyhow!("write {mp}: {e}"))?;
                        eprintln!("[PRE_QUANT_DUMP] meta -> {mp}");
                    }

                }

                // -- GPU KV cache update: Hadamard-quantize into TQ packed cache (ADR-007) --
                // HF2Q_SKIP_TQ_ENCODE=1: skip for timing bisection (output garbage).
                //
                // ADR-028 iter-485 (Phase 7d / H4): when HF2Q_TQ_FAST_FUSED_KV=1
                // collapse the two consecutive dispatches into one via the
                // Z-dim-split `dispatch_hadamard_quantize_kv_fast_dual`.
                // Byte-identical to the 2-dispatch reference; HF2Q_DEBUG_TQ_RMS
                // path forces the legacy split (probe is single-stream only).
                if !INVESTIGATION_ENV.skip_tq_encode {
                    let cache_pos_val = if kv_is_sliding {
                        (kv_write_pos % kv_capacity) as u32
                    } else {
                        kv_write_pos as u32
                    };
                    session.barrier_between(
                        &[&self.activations.attn_k_normed, v_src],
                        &[&self.kv_caches[layer_idx].k_packed, &self.kv_caches[layer_idx].k_norms,
                          &self.kv_caches[layer_idx].v_packed, &self.kv_caches[layer_idx].v_norms],
                    );
                    if INVESTIGATION_ENV.tq_fast_fused_kv && !INVESTIGATION_ENV.debug_tq_rms {
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_fast_dual(
                            session.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed,
                            v_src,
                            &self.kv_caches[layer_idx].k_packed,
                            &self.kv_caches[layer_idx].v_packed,
                            &self.kv_caches[layer_idx].k_norms,
                            &self.kv_caches[layer_idx].v_norms,
                            nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                            kv_is_sliding,
                            Some(ctx.tq_scale_factor_d512),
                        ).map_err(|e| anyhow::anyhow!("hadamard_quantize KV dual L{layer_idx}: {e}"))?;
                        *total_dispatches += 1;
                    } else {
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                            session.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed,
                            &self.kv_caches[layer_idx].k_packed,
                            &self.kv_caches[layer_idx].k_norms,
                            nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                            kv_is_sliding,
                            Some(ctx.tq_scale_factor_d512),
                            None, // rms_scratch: handled below by HF2Q_DEBUG_TQ_RMS path
                        ).map_err(|e| anyhow::anyhow!("hadamard_quantize K L{layer_idx}: {e}"))?;
                        *total_dispatches += 1;
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                            session.encoder_mut(), reg, metal_dev,
                            v_src,
                            &self.kv_caches[layer_idx].v_packed,
                            &self.kv_caches[layer_idx].v_norms,
                            nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                            kv_is_sliding,
                            Some(ctx.tq_scale_factor_d512),
                            None, // rms_scratch: probe not wired here
                        ).map_err(|e| anyhow::anyhow!("hadamard_quantize V L{layer_idx}: {e}"))?;
                        *total_dispatches += 1;
                    }
                }

                // iter-24: higher-bit (5/6/8-bit) KV encode into leg_hb_encoded.
                // When HF2Q_TQ_CODEBOOK_BITS=5|6|8, encode K/V to byte-packed HB format
                // for native HB SDPA dispatch via `flash_attn_vec_tq_hb` (which reads
                // `leg_hb_encoded` directly — no F32 shadow-cache round-trip).
                // iter-222 (2026-05-01): the `&& !force_dense_sdpa_on_tq_kv` gate
                // that suppressed this block under iter-34's dense-on-shadow
                // default was deleted along with the iter-34 Leg F branch.
                if ctx.use_native_hb_sdpa && !INVESTIGATION_ENV.skip_tq_encode {
                    // ADR-028 Phase 10c (iter-348): hybrid F16-K + TQ-HB-V
                    // encode path. K is written F32→F16 via the existing
                    // `kv_cache_copy_batch_f32_to_f16` (no Hadamard, no
                    // codebook lookup); V is encoded via the existing
                    // single-buffer `dispatch_hadamard_quantize_kv_hb` path
                    // (legacy 2-dispatch arm reused).
                    //
                    // 2 dispatches/layer/token vs 1 in the dual legacy path
                    // (+30 dispatches/decode-token at gemma4 30L).  Trade-off
                    // documented in ADR-028 §iter-348: the K-side SDPA
                    // throughput gain (Phase 10d) outweighs the encode
                    // overhead; if not, follow-up adds a fused
                    // `kv_copy_f16_quantize_v_dual` kernel.
                    if INVESTIGATION_ENV.hybrid_kv {
                        if let Some(ref hybrid_kv) = self.hybrid_kv {
                            let cache_pos_val = if kv_is_sliding {
                                (kv_write_pos % kv_capacity) as u32
                            } else {
                                kv_write_pos as u32
                            };
                            session.barrier_between(
                                &[&self.activations.attn_k_normed, v_src],
                                &[&hybrid_kv[layer_idx].k,
                                  &hybrid_kv[layer_idx].v_packed, &hybrid_kv[layer_idx].v_norms],
                            );
                            // ADR-029 iter-20 H27: when V is allocated as F16
                            // (HF2Q_FULL_F16_KV=1), write both K and V via a
                            // plain F32→F16 cast — no TQ-HB quantize, no FWHT.
                            // Detect via v_packed dtype (single source of truth
                            // matching the alloc-time selection).
                            if hybrid_kv[layer_idx].v_packed.dtype() == mlx_native::DType::F16 {
                                mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16_kv_dual(
                                    session.encoder_mut(), reg, metal_dev,
                                    &self.activations.attn_k_normed,
                                    v_src,
                                    &hybrid_kv[layer_idx].k,
                                    &hybrid_kv[layer_idx].v_packed,
                                    nkv as u32, hd as u32,
                                    hybrid_kv[layer_idx].capacity as u32,
                                    cache_pos_val,
                                ).map_err(|e| anyhow::anyhow!("full F16 KV write L{layer_idx}: {e}"))?;
                                *total_dispatches += 1;
                            } else {
                            // ADR-028 Phase 10c.5 (iter-354): fused F16-K-copy +
                            // V-no-FWHT-encode in a single dispatch (Z-dim splits
                            // K and V streams).  Byte-identical to the prior
                            // 2-dispatch sequence at identical params (verified
                            // via test_kv_copy_kf16_quantize_v_no_fwht_parity).
                            // Saves 30 KV-write dispatches/decode-token at gemma4
                            // 30L.
                            mlx_native::ops::hadamard_quantize_kv::dispatch_kv_copy_kf16_quantize_v_no_fwht(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.attn_k_normed,
                                v_src,
                                &hybrid_kv[layer_idx].k,
                                &hybrid_kv[layer_idx].v_packed,
                                &hybrid_kv[layer_idx].v_norms,
                                nkv as u32, hd as u32,
                                hybrid_kv[layer_idx].capacity as u32,
                                cache_pos_val,
                                hybrid_kv[layer_idx].is_sliding,
                                ctx.tq_scale_factor_d512,
                                ctx.tq_codebook_bits,
                            ).map_err(|e| anyhow::anyhow!("hybrid fused KV write L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                            } // closes else-block (legacy TQ-HB V path under hybrid)
                        }
                    } else if let Some(ref leg_hb_enc) = self.leg_hb_encoded {
                        let cache_pos_val = if kv_is_sliding {
                            (kv_write_pos % kv_capacity) as u32
                        } else {
                            kv_write_pos as u32
                        };
                        session.barrier_between(
                            &[&self.activations.attn_k_normed, v_src],
                            &[&leg_hb_enc[layer_idx].k_packed, &leg_hb_enc[layer_idx].k_norms,
                              &leg_hb_enc[layer_idx].v_packed, &leg_hb_enc[layer_idx].v_norms],
                        );
                        // ADR-028 iter-149: fused K+V HB encoder (default-on).
                        // HF2Q_HB_DUAL_LEGACY=1 forces 2-dispatch reference path
                        // for forensic A/B parity audit. Both paths byte-identical
                        // by mlx-native unit test
                        // (`test_hadamard_quantize_kv_hb_dual_byte_identity_d256`).
                        if INVESTIGATION_ENV.hb_dual_legacy {
                            mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.attn_k_normed,
                                &leg_hb_enc[layer_idx].k_packed,
                                &leg_hb_enc[layer_idx].k_norms,
                                nkv as u32, hd as u32,
                                leg_hb_enc[layer_idx].capacity as u32,
                                cache_pos_val,
                                leg_hb_enc[layer_idx].is_sliding,
                                ctx.tq_scale_factor_d512,
                                ctx.tq_codebook_bits,
                            ).map_err(|e| anyhow::anyhow!("hb_quantize K L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                            mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
                                session.encoder_mut(), reg, metal_dev,
                                v_src,
                                &leg_hb_enc[layer_idx].v_packed,
                                &leg_hb_enc[layer_idx].v_norms,
                                nkv as u32, hd as u32,
                                leg_hb_enc[layer_idx].capacity as u32,
                                cache_pos_val,
                                leg_hb_enc[layer_idx].is_sliding,
                                ctx.tq_scale_factor_d512,
                                ctx.tq_codebook_bits,
                            ).map_err(|e| anyhow::anyhow!("hb_quantize V L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                        } else {
                            mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb_dual(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.attn_k_normed, v_src,
                                &leg_hb_enc[layer_idx].k_packed, &leg_hb_enc[layer_idx].v_packed,
                                &leg_hb_enc[layer_idx].k_norms,  &leg_hb_enc[layer_idx].v_norms,
                                nkv as u32, hd as u32,
                                leg_hb_enc[layer_idx].capacity as u32,
                                cache_pos_val,
                                leg_hb_enc[layer_idx].is_sliding,
                                ctx.tq_scale_factor_d512,
                                ctx.tq_codebook_bits,
                            ).map_err(|e| anyhow::anyhow!("hb_quantize KV dual L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                        }
                    }
                }

                // iter-18 S2A: HF2Q_DEBUG_TQ_RMS — POST-SCALE RMS probe (Codex HIGH-1 fix).
                // Previous iter reported stored blk_norm (pre-scale ~0.06), which was WRONG.
                // This iter reports actual post-scale quantizer-input RMS by:
                //   1. Committing the encode command buffer.
                //   2. Reading back the stored norm from k_norms (blk_norm).
                //   3. Computing the post-scale RMS analytically:
                //      post_scale_rms = scale_factor * blk_norm / blk_norm = scale_factor
                //      (exact: after FWHT_norm, block RMS = blk_norm; scale = inv_blk_norm * sf;
                //       → post_scale_elem_rms = sqrt(mean(e^2)) = sf).
                //   4. Probing via scratch buffer (16 samples/block) for empirical verification.
                //
                // Reports both SLIDING (hd=256) and GLOBAL (hd=512) — spec AC-1 requires both.
                // ADR-005 wave-1 T1.2: read from INVESTIGATION_ENV LazyLock.
                if INVESTIGATION_ENV.debug_tq_rms {
                    // iter-19 A1: Fixed RMS probe (catalog #21 — write ALL EPT samples per lane).
                    // Previous iter-18 bug: scratch=[nkv, norms_per_pos, 16], only 8 values written
                    // for D=256 (EPT=8), rest zeros; host divided by 16 → RMS ≈ sqrt(0.5) * true_RMS.
                    // Fix: scratch=[1_head, head_dim] = 256 elements for D=256 (32 lanes × EPT=8).
                    //      For D=512: 512 elements (32 lanes × EPT=16); blk0=[0..255], blk1=[256..511].
                    //      Host divisor = 256 per block.
                    //
                    // iter-19 A2: RMS band LOCKED at [0.8, 1.2] (catalog #11).
                    // No expected*0.5/expected*2.0 arithmetic; constants are literal.
                    const RMS_BAND_LOW: f32 = 0.8;
                    const RMS_BAND_HIGH: f32 = 1.2;

                    // Commit the encode command buffer.
                    std::mem::replace(session, exec.begin()
            .map_err(|e| anyhow::anyhow!("HF2Q_DEBUG_TQ_RMS re-begin: {e}"))?).finish()
                        .map_err(|e| anyhow::anyhow!("HF2Q_DEBUG_TQ_RMS finish L{layer_idx}: {e}"))?;

                    let norms_per_pos = (hd / 256).max(1);
                    // Allocate scratch buffer: [1_head, head_dim] f32 = head_dim elements.
                    // All 32 lanes × EPT elements each = head_dim total samples per block (D=256)
                    // or head_dim total samples covering both blocks (D=512: blk0=[0..255], blk1=[256..511]).
                    let scratch_n = hd; // 256 for D=256, 512 for D=512
                    let mut scratch_buf = dev.alloc_buffer(
                        scratch_n * 4, mlx_native::DType::F32,
                        vec![1, hd],
                    ).map_err(|e| anyhow::anyhow!("HF2Q_DEBUG_TQ_RMS alloc scratch L{layer_idx}: {e}"))?;
                    // Zero-initialize scratch.
                    {
                        let scratch_slice: &mut [f32] = scratch_buf.as_mut_slice()
                            .map_err(|e| anyhow::anyhow!("HF2Q_DEBUG_TQ_RMS scratch zero L{layer_idx}: {e}"))?;
                        scratch_slice.iter_mut().for_each(|v| *v = 0.0);
                    }

                    // Compute actual write position for this token.
                    let actual_pos = if kv_is_sliding {
                        kv_write_pos % kv_capacity
                    } else {
                        kv_write_pos.min(kv_capacity - 1)
                    };
                    // Re-dispatch probe for head=0 only using a fresh command buffer.
                    let probe_kind = if kv_is_sliding { "sliding" } else { "global" };
                    let mut sp = exec.begin()
                        .map_err(|e| anyhow::anyhow!("HF2Q_DEBUG_TQ_RMS probe begin L{layer_idx}: {e}"))?;
                    mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                        sp.encoder_mut(), reg, metal_dev,
                        &self.activations.attn_k_normed,
                        &self.kv_caches[layer_idx].k_packed,
                        &self.kv_caches[layer_idx].k_norms,
                        1u32, // probe head=0 only
                        hd as u32, kv_capacity as u32, actual_pos as u32,
                        kv_is_sliding,
                        Some(ctx.tq_scale_factor_d512),
                        Some(&scratch_buf),
                    ).map_err(|e| anyhow::anyhow!("HF2Q_DEBUG_TQ_RMS probe dispatch L{layer_idx}: {e}"))?;
                    sp.finish()
                        .map_err(|e| anyhow::anyhow!("HF2Q_DEBUG_TQ_RMS probe finish L{layer_idx}: {e}"))?;

                    // Read back scratch: [1 head, head_dim] f32 — ALL samples written.
                    // D=256: 256 samples (1 block). D=512: 512 samples (2 blocks).
                    let scratch_raw: &[f32] = scratch_buf.as_slice()
                        .map_err(|e| anyhow::anyhow!("HF2Q_DEBUG_TQ_RMS scratch read L{layer_idx}: {e}"))?;

                    for blk in 0..norms_per_pos {
                        // Each block = 256 consecutive elements in scratch.
                        // D=256: blk=0, offset=0, 256 elements.
                        // D=512: blk=0 offset=0 (elements 0..255); blk=1 offset=256 (elements 256..511).
                        let blk_start = blk * 256;
                        let blk_end = (blk_start + 256).min(scratch_raw.len());
                        let samples: &[f32] = &scratch_raw[blk_start..blk_end];
                        // Compute RMS: divide by 256 (full block sample count).
                        let rms = if samples.len() == 256 {
                            let sum_sq: f32 = samples.iter().map(|v| v * v).sum();
                            (sum_sq / 256.0_f32).sqrt()
                        } else {
                            // Partial block (shouldn't happen, but guard):
                            let sum_sq: f32 = samples.iter().map(|v| v * v).sum();
                            if samples.is_empty() { 0.0 } else { (sum_sq / samples.len() as f32).sqrt() }
                        };
                        // iter-19 A2: band LOCKED at [0.8, 1.2] (catalog #11).
                        // This is the spec band for bare scale_factor=1.0 which is the iter-16 control.
                        // Only bare is valid (iter-16 result); sqrt256/sqrt512 are FALSIFIED (iter-16/18).
                        let status = if rms >= RMS_BAND_LOW && rms <= RMS_BAND_HIGH { "PASS" } else { "FAIL" };
                        eprintln!(
                            "[HF2Q_DEBUG_TQ_RMS] layer={layer_idx} kind={probe_kind} head=0 \
                             blk={blk} rms={rms:.4} band=[{RMS_BAND_LOW:.3},{RMS_BAND_HIGH:.3}] \
                             status={status} (divisor=256 samples)"
                        );
                    }

                }

                // C-1-unlock: post-hadamard_quantize pre-SDPA dump (decode step 1, layer 0).
                // Gate: dump_tq_state && layer_idx == 0 && kv_seq_len == 23 (one decode token
                // has been written into slot 22 of the TQ ring buffer).
                // Dumps full-capacity packed K/V + norms + Q (pre-FWHT) to post_quant subdir.
                if INVESTIGATION_ENV.dump_tq_state && layer_idx == 0 && kv_seq_len == 23 {
                    std::mem::replace(session, exec.begin()
            .map_err(|e| anyhow::anyhow!("post_quant dump re-begin: {e}"))?).finish()
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
                        // iter-25 Subtask B fix: use corrected ring_start formula (oldest slot).
                        let ring_start = if kv_is_sliding && kv_seq_len >= kv_capacity {
                            ((kv_write_pos + 1) % kv_capacity) as u32
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

                }

                // ADR-009 Phase 3A: dump Q,K,V before SDPA for the detail layer,
                // or ALL layers when HF2Q_DUMP_ALL_CACHE=1
                // W39 iter-112b: consult per-instance override first.
                let dump_all_cache = ctx.dump_all_cache_eff;
                if dump_layers && (dump_detail_layer == Some(layer_idx) || dump_all_cache) {
                    std::mem::replace(session, exec.begin()
            .map_err(|e| anyhow::anyhow!("dump QKV re-begin L{layer_idx}: {e}"))?).finish()
                        .map_err(|e| anyhow::anyhow!("dump QKV finish L{layer_idx}: {e}"))?;
                    let dir_override = self.dump_dir_override.as_deref();
                    dumps::dump_f32_to(&self.activations.attn_q_normed, nh * hd,
                        "q_normed", Some(layer_idx), seq_pos, dir_override)?;
                    dumps::dump_f32_to(&self.activations.attn_k_normed, nkv * hd,
                        "k_normed", Some(layer_idx), seq_pos, dir_override)?;
                    dumps::dump_f32_to(v_src, nkv * hd,
                        "v_normed", Some(layer_idx), seq_pos, dir_override)?;
                }

                // -- SDPA: TQ (default) or DENSE opt-out --
                // ADR-007 CLOSED 2026-04-24, post-close correction 2026-04-24:
                // TQ-8-bit is the DEFAULT decode path (2× memory savings vs F16;
                // Gate A cosine 0.9998 exceeds TurboQuant paper 0.999; Gate B argmax
                // divergence 0.8% exceeds <1%; Gate C PPL delta 1.24% / 0.017 absolute
                // meets KIVI + KVQuant + AmesianX + vLLM + TurboQuant shippability gates).
                // Rationale: "TQ should be default if it is better" — user feedback on
                // iter-28's overly-conservative flip to dense. Byte-exact vs llama.cpp is
                // still achievable via HF2Q_USE_DENSE=1 (sourdough_gate.sh sets this).
                //
                // HF2Q_LAYER_POLICY values:
                //   unset OR "tq_all"         = DEFAULT: full TQ decode (8-bit native HB SDPA)
                //   "dense_all"               = dense everywhere (byte-exact vs llama.cpp)
                //   "tq_slide_dense_global"   = TQ for sliding layers, dense for global
                //   "dense_slide_tq_global"   = dense for sliding, TQ for global
                //
                // HF2Q_USE_DENSE=1 forces dense_all (explicit opt-out for byte-exact gates).
                //
                // W12 iter-108a blocker #3 (ADR-007 Gate H): per-call regime override
                // consulted BEFORE the env vars.  When the regime is `DecodeRegime::Default`
                // (the default for every existing call site), this path is bit-identical
                // to today's env-var-only logic.  When the regime is `ForceDense` /
                // `ForceTq`, the env vars are skipped entirely so a single process
                // can run both regimes against the same prompt without subprocess
                // fork.  The four-gate lockstep contract (W9's mapping —
                // forward_mlx.rs:1100/1234/<this gate>, forward_prefill.rs:330) is
                // preserved: only this SDPA-reader gate is overridden; the codebook-
                // bits gates remain env-driven because the codebook width is a
                // representation choice consistent across both regimes.  See
                // `MlxModelWeights::set_decode_regime` for the contract.
                // iter-108a-fix (W15, 2026-04-25): when `gate_h_inactive` is true
                // (the default — no Gate H env hooks armed and regime is Default),
                // skip the regime-match arm entirely and use the pre-iter-108a
                // env-var-only path verbatim. This restores the byte-identical
                // hot-path branch sequence to the iter-108a base commit
                // (`1bcf172`). When Gate H is active, the regime override is
                // consulted as before. Per-layer = ~30× per token; the saved
                // enum-field load + match across the layer loop is the bulk of
                // the W14b 5.6% regression.
                // ADR-005 wave-1 T1.2: HF2Q_USE_DENSE and HF2Q_LAYER_POLICY read from
                // INVESTIGATION_ENV LazyLock (parsed once at process start) instead of
                // calling std::env::var per-token per-layer. Behavior is bit-identical:
                // `use_dense` mirrors `== Ok("1")`; `layer_policy.as_deref()` mirrors
                // `as_deref()` on the Result, with None mapping to the former Err(_) arm.
                // iter-222 (ADR-005 closure, 2026-05-01): the iter-50
                // `None if force_dense_sdpa_on_tq_kv => true` arms that routed
                // iter-34's default to Branch A (dense_kvs) were deleted — see
                // file-level iter-222 closure note. Default now flows through
                // the inline-fused TQ-native path below as in pre-iter-34.
                let use_dense_sdpa = if self.dense_kvs.is_none() {
                    false
                } else if self.gate_h_inactive {
                    // Pre-iter-108a path: LazyLock-cached env values. Bit-identical to base.
                    if INVESTIGATION_ENV.use_dense {
                        true
                    } else {
                        match INVESTIGATION_ENV.layer_policy.as_deref() {
                            Some("dense_all") => true,
                            Some("tq_all") | None => false,
                            Some("tq_slide_dense_global") => !kv_is_sliding,
                            Some("dense_slide_tq_global") => kv_is_sliding,
                            Some(other) => {
                                static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                                if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                                    eprintln!("[HF2Q_LAYER_POLICY] unknown value {:?}; defaulting to tq_all", other);
                                }
                                false
                            }
                        }
                    }
                } else {
                    match self.decode_regime {
                        DecodeRegime::ForceDense => true,
                        DecodeRegime::ForceTq => false,
                        DecodeRegime::Default => {
                            if INVESTIGATION_ENV.use_dense {
                                true
                            } else {
                                match INVESTIGATION_ENV.layer_policy.as_deref() {
                                    Some("dense_all") => true,
                                    Some("tq_all") | None => false,
                                    Some("tq_slide_dense_global") => !kv_is_sliding,
                                    Some("dense_slide_tq_global") => kv_is_sliding,
                                    Some(other) => {
                                        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                                        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                                            eprintln!("[HF2Q_LAYER_POLICY] unknown value {:?}; defaulting to tq_all", other);
                                        }
                                        false
                                    }
                                }
                            }
                        }
                    }
                };

                if use_dense_sdpa {
                    // -- Dense decode SDPA (ADR-009 Track 3) --
                    // Copy this position's K,V into dense KV buffers.
                    // Uses F16 cast kernel when dense_kvs are F16, else F32 copy.
                    let dense_kvs = self.dense_kvs.as_ref().unwrap();
                    let dense_cap = dense_kvs[layer_idx].capacity;
                    let layer_is_ring = dense_kvs[layer_idx].is_sliding;
                    // ADR-017 Phase E.a iter-3.6: when LONG_RESUME is on
                    // and layer is sliding, the buffer is LINEAR (no
                    // wrap); decode writes go to slot=seq_pos. When OFF
                    // (default), sliding wraps via slot=seq_pos%cap.
                    let kv_lcp_long_resume_for_write = INVESTIGATION_ENV.kv_lcp_long_resume
                        && INVESTIGATION_ENV.kv_lcp_resume
                        && INVESTIGATION_ENV.use_dense;
                    let write_slot = if layer_is_ring && !kv_lcp_long_resume_for_write {
                        (seq_pos % dense_cap) as u32
                    } else {
                        seq_pos as u32
                    };
                    let kv_is_f16 = dense_kvs[layer_idx].k.dtype() == mlx_native::DType::F16;
                    session.barrier_between(
                        &[&self.activations.attn_k_normed, v_src],
                        &[&dense_kvs[layer_idx].k, &dense_kvs[layer_idx].v],
                    );
                    // ADR-028 iter-146: fused K+V single-position copy (default-on).
                    // HF2Q_KV_DUAL_LEGACY=1 forces 2-dispatch reference path for
                    // forensic A/B parity audit; matches W-5b.10/14 sunset cadence.
                    let use_legacy_2dispatch = INVESTIGATION_ENV.kv_dual_legacy;
                    if kv_is_f16 {
                        if use_legacy_2dispatch {
                            mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.attn_k_normed,
                                &dense_kvs[layer_idx].k,
                                nkv as u32, hd as u32,
                                dense_cap as u32, write_slot,
                            ).map_err(|e| anyhow::anyhow!("decode F16 K copy L{layer_idx}: {e}"))?;
                            mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16(
                                session.encoder_mut(), reg, metal_dev,
                                v_src,
                                &dense_kvs[layer_idx].v,
                                nkv as u32, hd as u32,
                                dense_cap as u32, write_slot,
                            ).map_err(|e| anyhow::anyhow!("decode F16 V copy L{layer_idx}: {e}"))?;
                            *total_dispatches += 2;
                        } else {
                            mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16_kv_dual(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.attn_k_normed, v_src,
                                &dense_kvs[layer_idx].k, &dense_kvs[layer_idx].v,
                                nkv as u32, hd as u32,
                                dense_cap as u32, write_slot,
                            ).map_err(|e| anyhow::anyhow!("decode F16 KV dual copy L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                        }
                    } else if use_legacy_2dispatch {
                        // F32 batched: one dispatch per K, one per V (all heads at once).
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32(
                            session.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed,
                            &dense_kvs[layer_idx].k,
                            nkv as u32, hd as u32,
                            dense_cap as u32, write_slot,
                        ).map_err(|e| anyhow::anyhow!("decode F32 K batch copy L{layer_idx}: {e}"))?;
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32(
                            session.encoder_mut(), reg, metal_dev,
                            v_src,
                            &dense_kvs[layer_idx].v,
                            nkv as u32, hd as u32,
                            dense_cap as u32, write_slot,
                        ).map_err(|e| anyhow::anyhow!("decode F32 V batch copy L{layer_idx}: {e}"))?;
                        *total_dispatches += 2;
                    } else {
                        // ADR-028 iter-146: fused F32 K+V into single dispatch.
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_kv_dual(
                            session.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed, v_src,
                            &dense_kvs[layer_idx].k, &dense_kvs[layer_idx].v,
                            nkv as u32, hd as u32,
                            dense_cap as u32, write_slot,
                        ).map_err(|e| anyhow::anyhow!("decode F32 KV dual copy L{layer_idx}: {e}"))?;
                        *total_dispatches += 1;
                    }

                    // ADR-009 Phase 3A: dump full cached K/V for the detail layer,
                    // or ALL layers when HF2Q_DUMP_ALL_CACHE=1
                    // W39 iter-112b: consult per-instance override first.
                    let dump_all_cache = ctx.dump_all_cache_eff;
                    if dump_layers && (dump_detail_layer == Some(layer_idx) || dump_all_cache) {
                        std::mem::replace(session, exec.begin()
            .map_err(|e| anyhow::anyhow!("dump cache re-begin L{layer_idx}: {e}"))?).finish()
                            .map_err(|e| anyhow::anyhow!("dump cache finish L{layer_idx}: {e}"))?;
                        // W39 iter-112b: per-instance dump dir override; falls back
                        // to INVESTIGATION_ENV.dump_dir when unset.
                        let dump_dir_override = self
                            .dump_dir_override
                            .as_ref()
                            .map(|p| p.to_string_lossy().into_owned());
                        let dump_dir: &str = dump_dir_override
                            .as_deref()
                            .unwrap_or(&INVESTIGATION_ENV.dump_dir);
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
                    }

                    // Dense flash_attn_vec
                    let dense_sdpa_tmp = self.dense_sdpa_tmp.as_ref().unwrap();
                    session.barrier_between(
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
                    // ADR-017 Phase E.a iter-3.6: when LONG_RESUME is on
                    // and layer is sliding, the buffer is LINEAR (cap >
                    // sliding_window, slot index = logical position),
                    // and the kernel masks via mask_type=2 +
                    // sliding_window=sw. When OFF (default), behavior is
                    // byte-identical to pre-iter-3.6 (ring + mask_type=1).
                    let kv_lcp_long_resume = INVESTIGATION_ENV.kv_lcp_long_resume
                        && INVESTIGATION_ENV.kv_lcp_resume
                        && INVESTIGATION_ENV.use_dense;
                    let use_linear_sliding = layer_is_ring && kv_lcp_long_resume;
                    let dense_kv_seq_len = if layer_is_ring && !use_linear_sliding {
                        ((seq_pos + 1).min(dense_cap)) as u32
                    } else {
                        (seq_pos + 1) as u32
                    };
                    let (mask_type_val, sliding_window_val) = if use_linear_sliding {
                        let model_sw = self.sliding_window.max(1);
                        (2u32, model_sw as u32)
                    } else {
                        (1u32, 0u32)
                    };
                    let p = mlx_native::ops::flash_attn_vec::FlashAttnVecParams {
                        num_heads: nh as u32,
                        num_kv_heads: nkv as u32,
                        head_dim: hd as u32,
                        kv_seq_len: dense_kv_seq_len,
                        kv_capacity: dense_cap as u32,
                        scale: 1.0,
                        mask_type: mask_type_val,
                        sliding_window: sliding_window_val,
                        softcap: 0.0,
                    };
                    mlx_native::ops::flash_attn_vec::flash_attn_vec(
                        session.encoder_mut(), reg, dev,
                        &self.activations.attn_q_normed,
                        &dense_kvs[layer_idx].k,
                        &dense_kvs[layer_idx].v,
                        &self.activations.sdpa_out,
                        dense_sdpa_tmp,
                        &p,
                    ).map_err(|e| anyhow::anyhow!("dense flash_attn_vec L{layer_idx}: {e}"))?;
                    *total_dispatches += 2; // main + reduce
                // iter-222 (ADR-005 closure, 2026-05-01): the iter-20 Leg F /
                // iter-34 dense-on-shadow decode branch was deleted entirely
                // here (~170 LOC) — see file-level iter-222 closure note above
                // the (now-deleted) `dense_sdpa_on_tq_kv_enabled()` site for
                // rationale (Gate H regression + peer-impl research +
                // "no fallback" mantra). TQ-regime SDPA now flows through the
                // inline-fused `flash_attn_vec_tq_hb` (cb_bits>=5, default 8)
                // or `flash_attn_vec_tq` (cb_bits=4 legacy) branches below.
                } else if !INVESTIGATION_ENV.skip_tq_sdpa && ctx.use_native_hb_sdpa {
                    // ADR-028 Phase 10c (iter-348): hybrid path SDPA dispatcher
                    // not yet wired (Phase 10e).  When the user enables
                    // `HF2Q_HYBRID_KV=1` without 10e+10d (kernel) landed,
                    // hard-fail loud-not-silent rather than read stale F32
                    // SDPA-out from a previous decode token.  This is the
                    // intentional partial-stack failure mode signalled in
                    // Phase 10b's design (iter-347).
                    //
                    // ADR-028 Phase 10e (iter-350): live wiring lands here.
                    // K is stored F16 raw → Q stays raw (NO FWHT-pre dispatch),
                    // SDPA runs in raw domain (NO FWHT-undo dispatch), V comes
                    // from hybrid_kv[layer_idx].{v_packed, v_norms} (TQ-HB-encoded
                    // by the Phase 10c encode site at line ~3074).  Saves 60
                    // FWHT dispatches/decode-token at gemma4 30L on top of the
                    // K-side codebook elimination.
                    if INVESTIGATION_ENV.hybrid_kv {
                        // ADR-028 Phase 10e (iter-350): hybrid F16-K + TQ-HB-V SDPA.
                        //
                        // FWHT chain reasoning (Phase 10e initial wiring iter-350
                        // kept FWHT-undo because V was FWHT-rotated; Phase 10e.5
                        // iter-351 swapped V-encode to `kv_quantize_v_no_fwht`,
                        // which stores raw V — so output is now in raw domain):
                        //   * K stored RAW F16 → Q stays raw, NO fwht_sign_premult.
                        //   * V stored RAW (Phase 10e.5 V-encode dispatcher) → SDPA
                        //     output = softmax × V_raw → output IS raw → NO
                        //     fwht_sign_undo dispatch needed.
                        //
                        // Net dispatch saving: 60 dispatches/decode-token at gemma4
                        // 30L (the entire FWHT chain in attention is eliminated).
                        let hybrid_kv = self.hybrid_kv.as_ref().ok_or_else(|| anyhow::anyhow!(
                            "HF2Q_HYBRID_KV=1 but hybrid_kv buffers not allocated \
                             (gemma4 decode L{layer_idx}); should have been allocated \
                             by Phase 10c lazy-alloc gate. See ADR-028 §iter-350."
                        ))?;
                        let hb_cap = hybrid_kv[layer_idx].capacity;
                        let hb_is_ring = hybrid_kv[layer_idx].is_sliding;
                        let hb_kv_seq_len = if hb_is_ring {
                            ((kv_write_pos + 1).min(hb_cap)) as u32
                        } else {
                            (kv_write_pos + 1) as u32
                        };
                        let ring_start_hb = if hb_is_ring && hb_kv_seq_len as usize >= hb_cap {
                            ((kv_write_pos + 1) % hb_cap) as u32
                        } else {
                            0u32
                        };
                        session.barrier_between(
                            &[&self.activations.attn_q_normed,
                              &hybrid_kv[layer_idx].k,
                              &hybrid_kv[layer_idx].v_packed,
                              &hybrid_kv[layer_idx].v_norms],
                            &[&self.activations.sdpa_out],
                        );
                        let p_hyb = mlx_native::ops::flash_attn_vec_hybrid::FlashAttnVecTqHbParams {
                            num_heads: nh as u32,
                            num_kv_heads: nkv as u32,
                            head_dim: hd as u32,
                            kv_seq_len: hb_kv_seq_len,
                            kv_capacity: hb_cap as u32,
                            scale: 1.0,
                            mask_type: if is_sliding { 2 } else { 1 },
                            sliding_window: if is_sliding { self.sliding_window as u32 } else { 0 },
                            softcap: 0.0,
                            ring_start: ring_start_hb,
                            scale_factor_d512: ctx.tq_scale_factor_d512,
                            codebook_bits: ctx.tq_codebook_bits,
                            // Hybrid kernel: caller passes RAW Q (no rotation).
                            // fuse_fwht_pre=0 → kernel reads Q as-is.
                            fuse_fwht_pre: 0,
                            nsg: mlx_native::ops::flash_attn_vec_tq_hb::compute_nsg(hb_kv_seq_len),
                        };
                        // HF2Q_FA_PEER_PORT*: dispatch peer-port kernel variant instead of hybrid.
                        // Preconditions: head_dim==256, K dtype==F16, V dtype==F16.
                        //
                        // iter-137 — two variants:
                        //   HF2Q_FA_PEER_PORT       = NWG=1 verbatim port (iter-126).
                        //                             Falsified at tg5000 (-25%) because peer's
                        //                             actual runtime uses NWG=32 (iter-133 root
                        //                             cause). Kept for A/B + documentation;
                        //                             additionally gated on is_sliding so
                        //                             full-attn fallthrough to HYBRID.
                        //   HF2Q_FA_PEER_PORT_NWG32 = NWG=32 + reduce-kernel port (iters 134-137).
                        //                             Matches peer's actual runtime dispatch.
                        //                             Validated WIN +1.8-3.1pp at tg100/tg2000/tg5000
                        //                             vs HYBRID at PORT's f16-V regime (iter-138/140).
                        //                             Default-flipped ON iter-149 per operator
                        //                             approval: "best possible outcome for users —
                        //                             if coherent + TQ still enabled + marginally
                        //                             faster, of course default."
                        //                             Reuses existing sdpa_tmp buffer (identical
                        //                             size formula nrows*32*(dv+2)*4).
                        //
                        // PORT_NWG32 default ON; opt out via HF2Q_FA_PEER_PORT_NWG32=0.
                        // PORT (NWG=1, falsified) default OFF — explicit HF2Q_FA_PEER_PORT=1 only.
                        // The precondition `v_packed.dtype()==F16` means PORT_NWG32 ONLY fires when
                        // TQ-HB-V is bypassed (HF2Q_FULL_F16_KV=1 or otherwise F16-V regime).
                        // With default TQ-HB-V active, PORT_NWG32 gate falls through to hybrid —
                        // zero behavior change. With explicit F16-V request, PORT_NWG32 wins +2pp.
                        static FA_PEER_PORT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        let use_peer_port = *FA_PEER_PORT.get_or_init(|| {
                            std::env::var("HF2Q_FA_PEER_PORT").map(|v| v == "1").unwrap_or(false)
                        });
                        static FA_PEER_PORT_NWG32: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        let use_peer_port_nwg32 = *FA_PEER_PORT_NWG32.get_or_init(|| {
                            // env_default_true pattern (mirrors HF2Q_Q6K_MV_NR2 iter-326):
                            // unset → ON; "0"/"false"/"off" → OFF; "1"/"true"/"on" → ON.
                            match std::env::var("HF2Q_FA_PEER_PORT_NWG32").ok().as_deref() {
                                None => true,
                                Some(v) if v.eq_ignore_ascii_case("0")
                                    || v.eq_ignore_ascii_case("false")
                                    || v.eq_ignore_ascii_case("off") => false,
                                Some(_) => true,
                            }
                        });

                        if use_peer_port_nwg32
                            && hd == 256
                            && hybrid_kv[layer_idx].k.dtype() == mlx_native::DType::F16
                            && hybrid_kv[layer_idx].v_packed.dtype() == mlx_native::DType::F16
                        {
                            let p_peer = mlx_native::ops::flash_attn_vec_peer_port_f16::FlashAttnVecPeerPortParams {
                                num_heads: nh as u32,
                                num_kv_heads: nkv as u32,
                                head_dim: hd as u32,
                                kv_seq_len: hb_kv_seq_len,
                                kv_capacity: hb_cap as u32,
                                scale: 1.0,
                                mask_type: if is_sliding { 2 } else { 1 },
                                sliding_window: if is_sliding { self.sliding_window as u32 } else { 0 },
                                ring_start: ring_start_hb,
                            };
                            mlx_native::ops::flash_attn_vec_peer_port_f16::flash_attn_vec_peer_port_f16_nwg32(
                                session.encoder_mut(), reg, dev,
                                &self.activations.attn_q_normed,
                                &hybrid_kv[layer_idx].k,
                                &hybrid_kv[layer_idx].v_packed,
                                &self.activations.sdpa_tmp,
                                &self.activations.sdpa_out,
                                &p_peer,
                            ).map_err(|e| anyhow::anyhow!("flash_attn_vec_peer_port_f16_nwg32 L{layer_idx}: {e}"))?;
                            *total_dispatches += 2; // vec + reduce
                        } else if use_peer_port
                            && is_sliding
                            && hd == 256
                            && hybrid_kv[layer_idx].k.dtype() == mlx_native::DType::F16
                            && hybrid_kv[layer_idx].v_packed.dtype() == mlx_native::DType::F16
                        {
                            let p_peer = mlx_native::ops::flash_attn_vec_peer_port_f16::FlashAttnVecPeerPortParams {
                                num_heads: nh as u32,
                                num_kv_heads: nkv as u32,
                                head_dim: hd as u32,
                                kv_seq_len: hb_kv_seq_len,
                                kv_capacity: hb_cap as u32,
                                scale: 1.0,
                                mask_type: if is_sliding { 2 } else { 1 },
                                sliding_window: if is_sliding { self.sliding_window as u32 } else { 0 },
                                ring_start: ring_start_hb,
                            };
                            mlx_native::ops::flash_attn_vec_peer_port_f16::flash_attn_vec_peer_port_f16(
                                session.encoder_mut(), reg, dev,
                                &self.activations.attn_q_normed,
                                &hybrid_kv[layer_idx].k,
                                &hybrid_kv[layer_idx].v_packed,
                                &self.activations.sdpa_out,
                                &p_peer,
                            ).map_err(|e| anyhow::anyhow!("flash_attn_vec_peer_port_f16 L{layer_idx}: {e}"))?;
                            *total_dispatches += 1; // NWG=1: no reduce kernel
                        } else {
                            mlx_native::ops::flash_attn_vec_hybrid::flash_attn_vec_hybrid(
                                session.encoder_mut(), reg, dev,
                                &self.activations.attn_q_normed,
                                &hybrid_kv[layer_idx].k,
                                &hybrid_kv[layer_idx].v_packed,
                                &hybrid_kv[layer_idx].v_norms,
                                &self.activations.sdpa_out,
                                &self.activations.sdpa_tmp,
                                &p_hyb,
                            ).map_err(|e| anyhow::anyhow!("flash_attn_vec_hybrid L{layer_idx}: {e}"))?;
                            *total_dispatches += 2; // main + reduce (conservative)
                        }
                        // ADR-028 Phase 10e.5 (iter-351): NO fwht_sign_undo on output.
                        // V is raw (Lloyd-Max quantized but not FWHT-rotated), so
                        // SDPA output is already in the raw domain — feed directly
                        // to o_proj.
                        // Hybrid path complete; fall through to o_proj/MLP without
                        // entering the legacy `if let Some(ref leg_hb_enc) ...` block
                        // below (`leg_hb_encoded` is `None` under hybrid_kv per the
                        // Phase 10c lazy-alloc mutex, so the `if let` is a no-op).
                    } else
                    // -- iter-24: native HB SDPA (5/6/8-bit byte-packed K/V) --
                    //
                    // K/V have been HB-encoded into leg_hb_encoded above.
                    // We dispatch flash_attn_vec_tq_hb which reads byte-packed K/V
                    // and applies the appropriate codebook inline — no dequant step needed.
                    if let Some(ref leg_hb_enc) = &self.leg_hb_encoded {
                        let hb_cap = leg_hb_enc[layer_idx].capacity;
                        let hb_is_ring = leg_hb_enc[layer_idx].is_sliding;

                        // ADR-028 iter-108: env-gated FWHT-pre fusion.
                        // HF2Q_TQ_FUSE_FWHT_PRE=1 skips the standalone FWHT-pre
                        // dispatch + its forced WAR barrier, instead asking the
                        // FA-vec-tq-hb kernel to apply sign-premult+FWHT+normalize
                        // internally before the K-loop. Iter-107 byte-parity test
                        // confirmed bit-identical output (max_abs_diff=0).
                        // Saves 1 dispatch + 1 barrier per layer × 30 = ~9% decode.
                        let fuse_fwht_pre_env = std::env::var("HF2Q_TQ_FUSE_FWHT_PRE")
                            .map(|v| v == "1").unwrap_or(false);

                        if !fuse_fwht_pre_env {
                            // Pre-rotate Q via FWHT with D1 sign pre-mult (same as 4-bit path).
                            session.barrier_between(
                                &[&self.activations.attn_q_normed],
                                &[&self.activations.attn_q_normed],
                            );
                            mlx_native::ops::fwht_standalone::dispatch_fwht_sign_premult_f32(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.attn_q_normed,
                                nh as u32, hd as u32,
                            ).map_err(|e| anyhow::anyhow!("HB FWHT Q sign-premult L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                        }

                        // Native HB SDPA (pre-rotated Q → rotated-domain output).
                        let hb_kv_seq_len = if hb_is_ring {
                            ((kv_write_pos + 1).min(hb_cap)) as u32
                        } else {
                            (kv_write_pos + 1) as u32
                        };
                        let ring_start_hb = if hb_is_ring && hb_kv_seq_len as usize >= hb_cap {
                            ((kv_write_pos + 1) % hb_cap) as u32
                        } else {
                            0u32
                        };
                        session.barrier_between(
                            &[&self.activations.attn_q_normed,
                              &leg_hb_enc[layer_idx].k_packed, &leg_hb_enc[layer_idx].k_norms,
                              &leg_hb_enc[layer_idx].v_packed, &leg_hb_enc[layer_idx].v_norms],
                            &[&self.activations.sdpa_out],
                        );
                        let p_hb = mlx_native::ops::flash_attn_vec_tq_hb::FlashAttnVecTqHbParams {
                            num_heads: nh as u32,
                            num_kv_heads: nkv as u32,
                            head_dim: hd as u32,
                            kv_seq_len: hb_kv_seq_len,
                            kv_capacity: hb_cap as u32,
                            scale: 1.0,
                            mask_type: if is_sliding { 2 } else { 1 },
                            sliding_window: if is_sliding { self.sliding_window as u32 } else { 0 },
                            softcap: 0.0,
                            ring_start: ring_start_hb,
                            scale_factor_d512: ctx.tq_scale_factor_d512,
                            codebook_bits: ctx.tq_codebook_bits,
                            fuse_fwht_pre: if fuse_fwht_pre_env { 1 } else { 0 },
                            // ADR-028 iter-127a Path D: NSG axis. Default 1 in
                            // iter-127a (byte-identical scaffold); compute_nsg
                            // lifts based on kL once kernel logic supports NSG > 1.
                            nsg: mlx_native::ops::flash_attn_vec_tq_hb::compute_nsg(hb_kv_seq_len),
                        };
                        // ADR-028 §iter-485 (Phase 7d H3): env-gated fused
                        // reduce + FWHT-sign-undo path. Saves 1 dispatch + 1
                        // forced memory_barrier per layer per decode-token
                        // (~30 of each at gemma4 30 layers). Parity test
                        // `reduce_tq_hb_undo_fused_vs_unfused_parity` confirmed
                        // byte-identical output (max_abs_diff=0, max_rel=0).
                        let tq_hb_out_fused = std::env::var("HF2Q_TQ_HB_OUT_FUSED")
                            .map(|v| v == "1").unwrap_or(false);

                        if tq_hb_out_fused {
                            mlx_native::ops::flash_attn_vec_tq_hb::flash_attn_vec_tq_hb_with_fused_undo(
                                session.encoder_mut(), reg, dev,
                                &self.activations.attn_q_normed,
                                &leg_hb_enc[layer_idx].k_packed,
                                &leg_hb_enc[layer_idx].k_norms,
                                &leg_hb_enc[layer_idx].v_packed,
                                &leg_hb_enc[layer_idx].v_norms,
                                &self.activations.sdpa_out,
                                &self.activations.sdpa_tmp,
                                &p_hb,
                            ).map_err(|e| anyhow::anyhow!("flash_attn_vec_tq_hb_with_fused_undo L{layer_idx}: {e}"))?;
                            *total_dispatches += 2; // main + fused-reduce-undo
                            // Caller contract: no trailing fwht_sign_undo
                            // dispatch — the fused reduce already inverse-
                            // rotated the output.
                        } else {
                            mlx_native::ops::flash_attn_vec_tq_hb::flash_attn_vec_tq_hb(
                                session.encoder_mut(), reg, dev,
                                &self.activations.attn_q_normed,
                                &leg_hb_enc[layer_idx].k_packed,
                                &leg_hb_enc[layer_idx].k_norms,
                                &leg_hb_enc[layer_idx].v_packed,
                                &leg_hb_enc[layer_idx].v_norms,
                                &self.activations.sdpa_out,
                                &self.activations.sdpa_tmp,
                                &p_hb,
                            ).map_err(|e| anyhow::anyhow!("flash_attn_vec_tq_hb L{layer_idx}: {e}"))?;
                            *total_dispatches += 2; // main + reduce (conservative)

                            // Inverse-rotate SDPA output.
                            session.barrier_between(
                                &[&self.activations.sdpa_out],
                                &[&self.activations.sdpa_out],
                            );
                            mlx_native::ops::fwht_standalone::dispatch_fwht_sign_undo_f32(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.sdpa_out,
                                nh as u32, hd as u32,
                            ).map_err(|e| anyhow::anyhow!("HB FWHT sign-undo L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                        }
                    }
                } else if !INVESTIGATION_ENV.skip_tq_sdpa {
                    // -- TQ-packed SDPA (original path) --
                    // Pre-rotate Q via FWHT with D1 sign pre-mult (ADR-007 iter-14 SRHT).
                    // Applies sign_j * Q_j before FWHT so Q_rotated = FWHT(sign*Q)/sqrt(d).
                    // K was encoded as FWHT(sign*K)/sqrt(d); dot product = (sign*Q)·(sign*K) = Q·K.
                    // Sign tables verbatim from AmesianX cpy-utils.cuh:158-163/211-220.
                    session.barrier_between(
                        &[&self.activations.attn_q_normed],
                        &[&self.activations.attn_q_normed],
                    );
                    mlx_native::ops::fwht_standalone::dispatch_fwht_sign_premult_f32(
                        session.encoder_mut(), reg, metal_dev,
                        &self.activations.attn_q_normed,
                        nh as u32, hd as u32,
                    ).map_err(|e| anyhow::anyhow!("FWHT Q sign-premult pre-rotate L{layer_idx}: {e}"))?;
                    *total_dispatches += 1;

                    // TQ SDPA (pre-rotated Q → rotated-domain output)
                    session.barrier_between(
                        &[&self.activations.attn_q_normed,
                          &self.kv_caches[layer_idx].k_packed, &self.kv_caches[layer_idx].k_norms,
                          &self.kv_caches[layer_idx].v_packed, &self.kv_caches[layer_idx].v_norms],
                        &[&self.activations.sdpa_out],
                    );
                    // iter-25 Subtask B fix: ring_start must be the physical slot of the OLDEST
                    // entry (not newest). kv_write_pos is pre-increment (the slot just written
                    // this step). After wrap: oldest = (kv_write_pos + 1) % capacity.
                    let ring_start = if kv_is_sliding && kv_seq_len >= kv_capacity {
                        ((kv_write_pos + 1) % kv_capacity) as u32
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
                        scale_factor_d512: ctx.tq_scale_factor_d512,
                    };
                    mlx_native::ops::flash_attn_vec_tq::flash_attn_vec_tq(
                        session.encoder_mut(), reg, dev,
                        &self.activations.attn_q_normed,
                        &self.kv_caches[layer_idx].k_packed,
                        &self.kv_caches[layer_idx].k_norms,
                        &self.kv_caches[layer_idx].v_packed,
                        &self.kv_caches[layer_idx].v_norms,
                        &self.activations.sdpa_out,
                        &self.activations.sdpa_tmp,
                        &p,
                    ).map_err(|e| anyhow::anyhow!("flash_attn_vec_tq L{layer_idx}: {e}"))?;

                    // Inverse-rotate SDPA output with D1 sign undo (ADR-007 iter-14 SRHT).
                    // Applies FWHT (= IWHT for normalized H) → sign_j * elem_j.
                    // Output accumulated sign*V_weighted; sign undo recovers V_weighted.
                    session.barrier_between(
                        &[&self.activations.sdpa_out],
                        &[&self.activations.sdpa_out],
                    );
                    mlx_native::ops::fwht_standalone::dispatch_fwht_sign_undo_f32(
                        session.encoder_mut(), reg, metal_dev,
                        &self.activations.sdpa_out,
                        nh as u32, hd as u32,
                    ).map_err(|e| anyhow::anyhow!("FWHT sign-undo inv-rotate L{layer_idx}: {e}"))?;
                    *total_dispatches += 1;
                    *total_dispatches += 2; // main + reduce
                }

                // ADR-009 Phase 3A: dump sdpa_out before O-proj for the detail layer,
                // or ALL layers when HF2Q_DUMP_ALL_CACHE=1
                // W39 iter-112b: route through dump_f32_to with the
                // per-instance dir override so Gate H's in-process harness
                // can redirect dumps after INVESTIGATION_ENV's LazyLock froze.
                if dump_layers && (dump_detail_layer == Some(layer_idx) || dump_all_cache) {
                    std::mem::replace(session, exec.begin()
            .map_err(|e| anyhow::anyhow!("dump sdpa_out re-begin L{layer_idx}: {e}"))?).finish()
                        .map_err(|e| anyhow::anyhow!("dump sdpa_out finish L{layer_idx}: {e}"))?;
                    // [nh, 1, hd] flattened.
                    let dir_override = self.dump_dir_override.as_deref();
                    dumps::dump_f32_to(&self.activations.sdpa_out, nh * hd,
                        "sdpa_out", Some(layer_idx), seq_pos, dir_override)?;
                }

                // iter-18 S2C: first-divergence dump (layer=0, sliding, decode steps 1..=10).
                // kv_seq_len=23 = first decode step (prompt len 22 + 1), so steps 1..10 = seq_len 23..32.
                let s2c_step = if kv_seq_len >= 23 && kv_seq_len <= 32 { kv_seq_len - 22 } else { 0 };
                if dump_sliding_l0 && layer_idx == 0 && kv_is_sliding && s2c_step >= 1 {
                    if let Some(run_name) = dump_run_name {
                        std::mem::replace(session, exec.begin()
            .map_err(|e| anyhow::anyhow!("S2C re-begin step={s2c_step}: {e}"))?).finish()
                            .map_err(|e| anyhow::anyhow!("S2C dump finish step={s2c_step}: {e}"))?;
                        let dump_base = "/tmp/cfa-iter18/dumps";
                        std::fs::create_dir_all(dump_base)
                            .map_err(|e| anyhow::anyhow!("S2C mkdir: {e}"))?;
                        let p = s2c_step;
                        let run = run_name;
                        // Q (post-RoPE): [nh, hd] f32
                        {
                            let q_raw: &[f32] = self.activations.attn_q_normed.as_slice()
                                .map_err(|e| anyhow::anyhow!("S2C q read: {e}"))?;
                            let q_bytes: &[u8] = unsafe { std::slice::from_raw_parts(
                                q_raw.as_ptr() as *const u8, nh * hd * 4) };
                            std::fs::write(format!("{dump_base}/pos-{p}-layer-0-q-{run}.bin"), q_bytes)
                                .map_err(|e| anyhow::anyhow!("S2C write q: {e}"))?;
                        }
                        // K cache slot 0: dense path reads from dense_kvs; TQ path reads from k_norms.
                        // We dump K_norms (f32) for TQ and the cache K for dense.
                        if use_dense_sdpa {
                            if let Some(ref dkvs) = self.dense_kvs {
                                let k_raw: &[f32] = dkvs[layer_idx].k.as_slice()
                                    .map_err(|e| anyhow::anyhow!("S2C dense k read: {e}"))?;
                                let slot0_bytes = nkv * hd * 4;
                                let k_bytes: &[u8] = unsafe { std::slice::from_raw_parts(
                                    k_raw.as_ptr() as *const u8, slot0_bytes.min(k_raw.len() * 4)) };
                                std::fs::write(format!("{dump_base}/pos-{p}-layer-0-k-{run}.bin"), k_bytes)
                                    .map_err(|e| anyhow::anyhow!("S2C write dense k: {e}"))?;
                                let v_raw: &[f32] = dkvs[layer_idx].v.as_slice()
                                    .map_err(|e| anyhow::anyhow!("S2C dense v read: {e}"))?;
                                let v_bytes: &[u8] = unsafe { std::slice::from_raw_parts(
                                    v_raw.as_ptr() as *const u8, slot0_bytes.min(v_raw.len() * 4)) };
                                std::fs::write(format!("{dump_base}/pos-{p}-layer-0-v-{run}.bin"), v_bytes)
                                    .map_err(|e| anyhow::anyhow!("S2C write dense v: {e}"))?;
                            }
                        } else {
                            // TQ path: dump k_norms + k_packed (representative)
                            let npp = (hd / 256).max(1);
                            let k_norms_raw: &[f32] = self.kv_caches[layer_idx].k_norms.as_slice()
                                .map_err(|e| anyhow::anyhow!("S2C tq k_norms read: {e}"))?;
                            let k_norms_bytes: &[u8] = unsafe { std::slice::from_raw_parts(
                                k_norms_raw.as_ptr() as *const u8,
                                nkv * kv_capacity * npp * 4) };
                            std::fs::write(format!("{dump_base}/pos-{p}-layer-0-k-{run}.bin"), k_norms_bytes)
                                .map_err(|e| anyhow::anyhow!("S2C write tq k: {e}"))?;
                            let v_norms_raw: &[f32] = self.kv_caches[layer_idx].v_norms.as_slice()
                                .map_err(|e| anyhow::anyhow!("S2C tq v_norms read: {e}"))?;
                            let v_norms_bytes: &[u8] = unsafe { std::slice::from_raw_parts(
                                v_norms_raw.as_ptr() as *const u8,
                                nkv * kv_capacity * npp * 4) };
                            std::fs::write(format!("{dump_base}/pos-{p}-layer-0-v-{run}.bin"), v_norms_bytes)
                                .map_err(|e| anyhow::anyhow!("S2C write tq v: {e}"))?;
                        }
                        // SDPA output: [nh, hd] f32
                        {
                            let sdpa_raw: &[f32] = self.activations.sdpa_out.as_slice()
                                .map_err(|e| anyhow::anyhow!("S2C sdpa read: {e}"))?;
                            let sdpa_bytes: &[u8] = unsafe { std::slice::from_raw_parts(
                                sdpa_raw.as_ptr() as *const u8, nh * hd * 4) };
                            std::fs::write(format!("{dump_base}/pos-{p}-layer-0-sdpa-{run}.bin"), sdpa_bytes)
                                .map_err(|e| anyhow::anyhow!("S2C write sdpa: {e}"))?;
                        }
                        eprintln!("[HF2Q_S2C] pos={p} layer=0 dumped q/k/v/sdpa run={run}");
                    }
                }

                // -- O-proj --
                // ADR-028 iter-211: SKIP_O_PROJ bisect.  Sequential
                // single qmatmul on critical path after SDPA.
                if !INVESTIGATION_ENV.skip_o_proj {
                    session.barrier_between(
                        &[&self.activations.sdpa_out, &self.layers[layer_idx].attn.o_proj.buffer],
                        &[&self.activations.attn_out],
                    );
                    dispatch_qmatmul(session, reg, dev, &self.activations.sdpa_out,
                        &self.layers[layer_idx].attn.o_proj, &self.activations.attn_out, 1)?;
                    *total_dispatches += 1;
                }

                // ADR-029 iter-9 — phase split at attn/ffn boundary.
                // HF2Q_PER_LAYER_PHASE_GPU_TIME=1 commits the attn portion
                // and reports its GPU time, then begins a new session for ffn.
                if std::env::var("HF2Q_PER_LAYER_PHASE_GPU_TIME").as_deref() == Ok("1") {
                    let gpu_ns: u64 = std::mem::replace(session, exec.begin().map_err(|e| anyhow::anyhow!("phase-attn begin L{layer_idx}: {e}"))?).finish_with_gpu_time()
                        .map_err(|e| anyhow::anyhow!("phase-attn finish L{layer_idx}: {e}"))?;
                    eprintln!("    [PHASE_ATTN L{:02} {}] gpu={:>6.1}µs",
                        layer_idx,
                        if is_sliding { "S" } else { "G" },
                        gpu_ns as f64 / 1000.0);
                    session.track_dispatch(&[],
                        &[&self.activations.hidden, &self.activations.attn_out]);
                }

                let num_experts = self.num_experts;
                let top_k = self.layers[layer_idx].moe.top_k;

                let dump_after_post_attn = dump_layers && dump_detail_layer == Some(layer_idx);

                // ADR-028 iter-186 — opt-in fused 4→1 kernel that combines:
                //   (a) post-attn norm+add (hidden + norm(attn_out, post_attn_w) → residual)
                //   (b) B8's three concurrent rms_norms over `residual` with weights
                //       {pre_feedforward_layernorm, pre_feedforward_layernorm_2,
                //        router_combined_weight} → {norm_out, moe_norm_out, router_norm_out}
                // Saves 3 dispatches/layer × 30 layers = 90 dispatches/token on gemma4.
                // Kernel `fused_post_attn_triple_norm_f32` already exists in mlx-native
                // (used by batched prefill).  Default-OFF until decode coherence proven.
                //
                // Disabled when dump_layers requires reading `residual` between
                // (a) and (b) — would need a CB split that defeats the fusion.
                if INVESTIGATION_ENV.fused_triple_norm && !dump_after_post_attn {
                    session.barrier_between(
                        &[&self.activations.hidden, &self.activations.attn_out],
                        &[&self.activations.residual,
                          &self.activations.norm_out,
                          &self.activations.moe_norm_out,
                          &self.activations.router_norm_out],
                    );
                    mlx_native::ops::rms_norm::dispatch_fused_post_attn_triple_norm_f32(
                        session.encoder_mut(), reg, metal_dev,
                        &self.activations.hidden,
                        &self.activations.attn_out,
                        &self.layers[layer_idx].norms.post_attention_layernorm,
                        &self.layers[layer_idx].norms.pre_feedforward_layernorm,
                        &self.layers[layer_idx].norms.pre_feedforward_layernorm_2,
                        &self.layers[layer_idx].moe.router_combined_weight,
                        &self.activations.residual,
                        &self.activations.norm_out,
                        &self.activations.moe_norm_out,
                        &self.activations.router_norm_out,
                        eps, 1, hs as u32,
                    ).map_err(|e| anyhow::anyhow!("fused post-attn+triple-norm L{layer_idx}: {e}"))?;
                    *total_dispatches += 1;
                } else {
                    // -- Fused post-attention norm + residual add --
                    // ADR-028 iter-205: SKIP_POST_ATTN_NORM bisect — skip
                    // the fused_norm_add dispatch.  Sequential, 1 per layer.
                    // Produces garbage residual stream.
                    if !INVESTIGATION_ENV.skip_post_attn_norm {
                        // ADR-029 iter-107 H76 — env-gated SPLIT of the
                        // fused norm+add into 2 separate dispatches
                        // (rms_norm → norm_out; elementwise_add hidden+norm_out
                        // → residual). Tests the counter-fusion hypothesis
                        // (iter-105 confirmed: on Apple Metal scheduler, more
                        // smaller dispatches outperform fewer larger fused
                        // dispatches at decode shape).
                        let split_postattn = std::env::var("HF2Q_SPLIT_POSTATTN_NORM").as_deref() == Ok("1");
                        if split_postattn {
                            // Step 1: norm_out = rms_norm(attn_out, post_attn_weight)
                            session.barrier_between(
                                &[&self.activations.attn_out,
                                  &self.layers[layer_idx].norms.post_attention_layernorm],
                                &[&self.activations.norm_out],
                            );
                            session.rms_norm(
                                reg, metal_dev,
                                &self.activations.attn_out,
                                &self.layers[layer_idx].norms.post_attention_layernorm,
                                &self.activations.norm_out,
                                &self.activations.norm_params,
                                1, hs as u32,
                            ).map_err(|e| anyhow::anyhow!("split post-attn norm L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;

                            // Step 2: residual = hidden + norm_out
                            session.barrier_between(
                                &[&self.activations.hidden, &self.activations.norm_out],
                                &[&self.activations.residual],
                            );
                            mlx_native::ops::elementwise::elementwise_add(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.hidden,
                                &self.activations.norm_out,
                                &self.activations.residual,
                                hs,
                                mlx_native::DType::F32,
                            ).map_err(|e| anyhow::anyhow!("split post-attn add L{layer_idx}: {e}"))?;
                        } else {
                            session.barrier_between(
                                &[&self.activations.hidden, &self.activations.attn_out],
                                &[&self.activations.residual],
                            );
                            mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.hidden,
                                &self.activations.attn_out,
                                &self.layers[layer_idx].norms.post_attention_layernorm,
                                &self.activations.residual,
                                hs as u32, 1, eps,
                            ).map_err(|e| anyhow::anyhow!("fused post-attn norm+add L{layer_idx}: {e}"))?;
                        }
                    }

                    if dump_after_post_attn {
                        std::mem::replace(session, exec.begin()
            .map_err(|e| anyhow::anyhow!("dump post-attn re-begin L{layer_idx}: {e}"))?).finish()
                            .map_err(|e| anyhow::anyhow!("dump post-attn finish L{layer_idx}: {e}"))?;
                        dumps::dump_f32(&self.activations.residual, hs,
                            "attn_out", Some(layer_idx), seq_pos)?;
                    }
                    *total_dispatches += 1;

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

                    // -- B8: pre-FF norm1 + pre-FF norm2 + router norm [3 CONCURRENT] --
                    session.barrier_between(
                        &[&self.activations.residual],
                        &[&self.activations.norm_out, &self.activations.moe_norm_out,
                          &self.activations.router_norm_out],
                    );
                    session.rms_norm(
                        reg, metal_dev,
                        &self.activations.residual,
                        &self.layers[layer_idx].norms.pre_feedforward_layernorm,
                        &self.activations.norm_out,
                        &self.activations.norm_params,
                        1, hs as u32,
                    ).map_err(|e| anyhow::anyhow!("pre-FF norm L{layer_idx}: {e}"))?;
                    *total_dispatches += 1;

                    session.rms_norm(
                        reg, metal_dev,
                        &self.activations.residual,
                        &self.layers[layer_idx].norms.pre_feedforward_layernorm_2,
                        &self.activations.moe_norm_out,
                        &self.activations.norm_params,
                        1, hs as u32,
                    ).map_err(|e| anyhow::anyhow!("pre-FF norm 2 L{layer_idx}: {e}"))?;
                    *total_dispatches += 1;

                    session.rms_norm(
                        reg, metal_dev,
                        &self.activations.residual,
                        &self.layers[layer_idx].moe.router_combined_weight,
                        &self.activations.router_norm_out,
                        &self.activations.norm_params,
                        1, hs as u32,
                    ).map_err(|e| anyhow::anyhow!("router norm L{layer_idx}: {e}"))?;
                    *total_dispatches += 1;
                }

                // ADR-029 iter-14 — FFN sub-phase split (HF2Q_FFN_SPLIT=1).
                // Boundary 1: end of "FFN_NORMS" sub-phase (post-attn norm +
                // B8 3 pre-FF norms or the fused_triple_norm equivalent).
                // Commits the CB so the next session's GPU time reports just
                // the FFN body (B9-B13) under the FFN_BODY label.
                if std::env::var("HF2Q_FFN_SPLIT").as_deref() == Ok("1") {
                    let gpu_ns: u64 = std::mem::replace(session, exec.begin().map_err(|e| anyhow::anyhow!("ffn-norms begin L{layer_idx}: {e}"))?).finish_with_gpu_time()
                        .map_err(|e| anyhow::anyhow!("ffn-norms finish L{layer_idx}: {e}"))?;
                    eprintln!("    [FFN_NORMS L{:02} {}] gpu={:>6.1}µs",
                        layer_idx,
                        if is_sliding { "S" } else { "G" },
                        gpu_ns as f64 / 1000.0);
                    session.track_dispatch(&[],
                        &[&self.activations.residual,
                          &self.activations.norm_out,
                          &self.activations.moe_norm_out,
                          &self.activations.router_norm_out]);
                }

                // -- B9: dense gate + dense up + router logits [3 CONCURRENT] --
                // gate/up read norm_out (from B8 norm1); router reads router_norm_out (from B8 router norm).
                // All write disjoint buffers. ONE barrier after B8, then 3 dispatches without barriers.
                session.barrier_between(
                    &[&self.activations.norm_out, &self.activations.router_norm_out],
                    &[&self.activations.mlp_gate, &self.activations.mlp_up,
                      &self.activations.moe_router_logits],
                );
                // ADR-029 iter-15 (H17 probe): HF2Q_B9_FORCE_SEQUENTIAL=1
                // inserts memory_barrier()s between B9's 3 concurrent qmatmuls
                // to test the "peer's more smaller serial dispatches" lever
                // class. M5 Max scheduler may favor sequential issue at this
                // shape (Q5_K 2816→5760 × 2 + 2816→128). Tracks no math
                // change — barriers ONLY affect timing/scheduling.
                let b9_sequential = std::env::var("HF2Q_B9_FORCE_SEQUENTIAL").as_deref() == Ok("1");
                // ADR-028 iter-200: SKIP_DENSE_MLP bisect — skip mlp_gate +
                // mlp_up dispatches.  Router proj must run (MoE depends on it).
                if !INVESTIGATION_ENV.skip_dense_mlp {
                    dispatch_qmatmul(session, reg, dev, &self.activations.norm_out,
                        &self.layers[layer_idx].mlp.gate_proj, &self.activations.mlp_gate, 1)?;
                    *total_dispatches += 1;
                    if b9_sequential { session.encoder_mut().memory_barrier(); }
                    dispatch_qmatmul(session, reg, dev, &self.activations.norm_out,
                        &self.layers[layer_idx].mlp.up_proj, &self.activations.mlp_up, 1)?;
                    *total_dispatches += 1;
                    if b9_sequential { session.encoder_mut().memory_barrier(); }
                }
                // ADR-028 iter-213: SKIP_ROUTING bisect — skip router_proj qmatmul.
                if !INVESTIGATION_ENV.skip_routing {
                    dispatch_qmatmul(session, reg, dev, &self.activations.router_norm_out,
                        &self.layers[layer_idx].moe.router_proj,
                        &self.activations.moe_router_logits, 1)?;
                    *total_dispatches += 1;
                }

                // -- B10: fused_gelu_mul + fused_moe_routing [2 CONCURRENT] --
                // gelu_mul reads mlp_gate+mlp_up (from B9 gate/up), writes mlp_fused.
                // moe_routing reads moe_router_logits (from B9 router), writes expert_ids+weights.
                // Disjoint reads and writes — ONE barrier after B9, then both dispatch.
                session.barrier_between(
                    &[&self.activations.mlp_gate, &self.activations.mlp_up,
                      &self.activations.moe_router_logits],
                    &[&self.activations.mlp_fused,
                      &self.activations.moe_expert_ids, &self.activations.moe_routing_weights_gpu],
                );
                if !INVESTIGATION_ENV.skip_dense_mlp {
                    use mlx_native::ops::encode_helpers::{encode_with_args, KernelArg};
                    let n_elements_bytes = (self.intermediate_size as u32).to_ne_bytes();
                    let pipeline = reg.get_pipeline("fused_gelu_mul", metal_dev)?;
                    encode_with_args(
                        session.encoder_mut(), pipeline,
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
                    *total_dispatches += 1;
                }
                if !INVESTIGATION_ENV.skip_routing {
                    mlx_native::ops::fused_norm_add::dispatch_fused_moe_routing_f32(
                        session.encoder_mut(), reg, metal_dev,
                        &self.activations.moe_router_logits,
                        &self.activations.moe_expert_ids,
                        &self.activations.moe_routing_weights_gpu,
                        &self.layers[layer_idx].moe.per_expert_scale,
                        num_experts as u32, top_k as u32,
                    ).map_err(|e| anyhow::anyhow!("fused MoE routing L{layer_idx}: {e}"))?;
                    *total_dispatches += 1;
                }

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
                    if !INVESTIGATION_ENV.skip_dense_mlp {
                        session.barrier_between(
                            &[&self.activations.mlp_fused, &self.layers[layer_idx].mlp.down_proj.buffer],
                            &[&self.activations.mlp_down],
                        );
                        dispatch_qmatmul(session, reg, dev, &self.activations.mlp_fused,
                            &self.layers[layer_idx].mlp.down_proj, &self.activations.mlp_down, 1)?;
                        *total_dispatches += 1;
                    }

                    let ggml_type_gu = self.layers[layer_idx].moe.gate_up_ggml_dtype;
                    session.barrier_between(
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
                    // ADR-028 iter-201: SKIP_MOE_EXPERTS bisect — skip
                    // gate_up_id + swiglu + down_id dispatches.  Produces
                    // garbage moe_down_id_out (stale buffer).
                    if !INVESTIGATION_ENV.skip_moe_experts {
                        session.quantized_matmul_id_ggml(
                            reg, dev,
                            &self.activations.moe_norm_out,
                            self.layers[layer_idx].moe.stacked_gate_up.as_ref().unwrap(),
                            &self.activations.moe_expert_ids,
                            &self.activations.moe_gate_up_id_out,
                            &gu_params,
                        ).map_err(|e| anyhow::anyhow!("gate_up _id L{layer_idx}: {e}"))?;
                        *total_dispatches += 1;

                        // -- B12: swiglu (singleton) --
                        // ADR-028 iter-202: SKIP_MOE_SWIGLU isolates swiglu
                        // cost.  Skipping leaves moe_swiglu_id_out stale →
                        // down_id reads garbage.  Timing-only bisect.
                        if !INVESTIGATION_ENV.skip_moe_swiglu {
                            session.barrier_between(
                                &[&self.activations.moe_gate_up_id_out],
                                &[&self.activations.moe_swiglu_id_out],
                            );
                            mlx_native::ops::moe_dispatch::moe_swiglu_batch_encode(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.moe_gate_up_id_out,
                                &self.activations.moe_swiglu_id_out,
                                moe_int, top_k,
                            ).map_err(|e| anyhow::anyhow!("swiglu batch L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                        }
                    }

                    // -- B13: down_id + post-FF norm1 [2 concurrent] --
                    // down_id reads moe_swiglu_id_out (from B12). post-FF norm1 reads
                    // mlp_down (from B11). Disjoint writes.
                    let ggml_type_dn = self.layers[layer_idx].moe.down_ggml_dtype;
                    session.barrier_between(
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
                    if !INVESTIGATION_ENV.skip_moe_experts {
                        session.quantized_matmul_id_ggml(
                            reg, dev,
                            &self.activations.moe_swiglu_id_out,
                            self.layers[layer_idx].moe.stacked_down.as_ref().unwrap(),
                            &self.activations.moe_expert_ids,
                            &self.activations.moe_down_id_out,
                            &dn_params,
                        ).map_err(|e| anyhow::anyhow!("down _id L{layer_idx}: {e}"))?;
                        *total_dispatches += 1;
                    }

                    // post-FF norm1: mlp_down → attn_out (concurrent with down_id)
                    session.barrier_between(
                        &[&self.activations.mlp_down],
                        &[&self.activations.attn_out],
                    );
                    session.rms_norm(
                        reg, metal_dev,
                        &self.activations.mlp_down,
                        &self.layers[layer_idx].norms.post_feedforward_layernorm_1,
                        &self.activations.attn_out,
                        &self.activations.norm_params,
                        1, hs as u32,
                    ).map_err(|e| anyhow::anyhow!("post-FF norm 1 L{layer_idx}: {e}"))?;
                    *total_dispatches += 1;

                    // -- B14: weighted_sum (singleton) --
                    // ADR-028 iter-206: SKIP_WEIGHTED_SUM bisect.
                    // ADR-028 iter-367: fold moe_weighted_sum into the fused
                    // end-of-layer kernel (Path A only).  Default-ON.
                    let use_iter367_fusion = INVESTIGATION_ENV.fused_end_of_layer
                        && !INVESTIGATION_ENV.skip_end_of_layer
                        && INVESTIGATION_ENV.fused_moe_wsum_end_layer_v2
                        && (hs as u32) % 4 == 0;
                    if !INVESTIGATION_ENV.skip_weighted_sum && !use_iter367_fusion {
                        session.barrier_between(
                            &[&self.activations.moe_down_id_out, &self.activations.moe_routing_weights_gpu],
                            &[&self.activations.moe_accum],
                        );
                        mlx_native::ops::moe_dispatch::moe_weighted_sum_encode(
                            session.encoder_mut(), reg, metal_dev,
                            &self.activations.moe_down_id_out,
                            &self.activations.moe_routing_weights_gpu,
                            &self.activations.moe_accum,
                            hs, top_k,
                        ).map_err(|e| anyhow::anyhow!("weighted_sum L{layer_idx}: {e}"))?;
                        *total_dispatches += 1;
                    }
                } else {
                    // Fallback: per-expert loop (all in same session)
                    mlx_native::ops::moe_dispatch::moe_zero_buffer_encode(
                        session.encoder_mut(), reg, metal_dev,
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

                // ADR-029 iter-14 — FFN sub-phase split (HF2Q_FFN_SPLIT=1).
                // Boundary 2: end of "FFN_BODY" sub-phase (B9-B13: dense MLP +
                // MoE experts + interleaved post-FF norm 1).  Commits the CB
                // so the next session's GPU time reports just the end-of-layer
                // norm + add + scalar under the FFN_EOL label.
                if std::env::var("HF2Q_FFN_SPLIT").as_deref() == Ok("1") {
                    let gpu_ns: u64 = std::mem::replace(session, exec.begin().map_err(|e| anyhow::anyhow!("ffn-body begin L{layer_idx}: {e}"))?).finish_with_gpu_time()
                        .map_err(|e| anyhow::anyhow!("ffn-body finish L{layer_idx}: {e}"))?;
                    eprintln!("    [FFN_BODY  L{:02} {}] gpu={:>6.1}µs",
                        layer_idx,
                        if is_sliding { "S" } else { "G" },
                        gpu_ns as f64 / 1000.0);
                    session.track_dispatch(&[],
                        &[&self.activations.mlp_down, &self.activations.moe_accum,
                          &self.activations.attn_out, &self.activations.residual]);
                }

                // ============================================================
                // GPU post-MoE: norm, combine MLP+MoE, final norm, residual, scalar
                // ============================================================

                // ADR-028 iter-207: SKIP_END_OF_LAYER bisect — skip the
                // 2 sequential fused_norm_add dispatches at end-of-layer.
                if !INVESTIGATION_ENV.skip_end_of_layer {
                    let scalar_is_vector = self.layers[layer_idx].layer_scalar.element_count() > 1;

                    // ADR-028 iter-219: HF2Q_FUSED_END_OF_LAYER replaces
                    // the 2 sequential fused_norm_add dispatches with the
                    // single fused_post_ff_norm2_endlayer_f32 kernel.
                    // Bisect-confirmed +2.7% target (iter-208).  Parity test
                    // PASS (iter-218).  Default-OFF until production bench.
                    if INVESTIGATION_ENV.fused_end_of_layer {
                        // ADR-028 iter-367: HF2Q_FUSED_MOE_WSUM_END_LAYER_V2=1 fuses
                        // moe_weighted_sum INTO this end-of-layer kernel, eliminating
                        // 1 dispatch + moe_accum round-trip from gemma4 decode default.
                        let use_iter367_fusion = std::env::var("HF2Q_FUSED_MOE_WSUM_END_LAYER_V2")
                            .ok().as_deref() == Some("1")
                            && (hs as u32) % 4 == 0
                            && !INVESTIGATION_ENV.skip_weighted_sum;
                        if use_iter367_fusion {
                            // ADR-028 iter-371 (PROBE): explicit memory_barrier()
                            // forces a global Metal barrier even if the tracker
                            // doesn't detect a conflict.  Tests if iter-367's
                            // coherence regression under the iter-321 stack is
                            // caused by a missed barrier (tracker reset between
                            // distant write + read).
                            session.encoder_mut().memory_barrier();
                            session.barrier_between(
                                &[&self.activations.moe_down_id_out,
                                  &self.activations.moe_routing_weights_gpu,
                                  &self.activations.attn_out,
                                  &self.activations.residual,
                                  &self.layers[layer_idx].layer_scalar],
                                &[&self.activations.mlp_down, &self.activations.hidden],
                            );
                            mlx_native::ops::rms_norm::dispatch_fused_moe_wsum_post_ff_norm2_endlayer_f32_v2(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.moe_down_id_out,
                                &self.activations.moe_routing_weights_gpu,
                                &self.activations.attn_out,
                                &self.activations.residual,
                                &self.layers[layer_idx].norms.post_feedforward_layernorm_2,
                                &self.layers[layer_idx].norms.post_feedforward_layernorm,
                                &self.layers[layer_idx].layer_scalar,
                                &self.activations.mlp_down,
                                &self.activations.hidden,
                                eps, 1, hs as u32, top_k as u32,
                                scalar_is_vector,
                            ).map_err(|e| anyhow::anyhow!("iter-367 fused wsum+endlayer L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                        } else {
                        session.barrier_between(
                            &[&self.activations.attn_out, &self.activations.moe_accum,
                              &self.activations.residual, &self.layers[layer_idx].layer_scalar],
                            &[&self.activations.mlp_down, &self.activations.hidden],
                        );
                        mlx_native::ops::rms_norm::dispatch_fused_post_ff_norm2_endlayer_f32(
                            session.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_out,
                            &self.activations.moe_accum,
                            &self.activations.residual,
                            &self.layers[layer_idx].norms.post_feedforward_layernorm_2,
                            &self.layers[layer_idx].norms.post_feedforward_layernorm,
                            &self.layers[layer_idx].layer_scalar,
                            &self.activations.mlp_down,
                            &self.activations.hidden,
                            eps, 1, hs as u32,
                            scalar_is_vector,
                        ).map_err(|e| anyhow::anyhow!("fused end-of-layer L{layer_idx}: {e}"))?;
                        *total_dispatches += 1;
                        }
                    } else {
                        // -- Fused post-FF norm 2 + combine MLP+MoE --
                        // ADR-029 iter-108 H77: env-gated SPLIT into 2 dispatches
                        // (rms_norm + elementwise_add). Same counter-fusion test
                        // class as H76; tests if STACKING multiple de-fusions
                        // produces measurable wall improvement (individual
                        // de-fusions are below noise floor per iter-107).
                        let split_postff_normadd = std::env::var("HF2Q_SPLIT_POSTFF_NORMADD").as_deref() == Ok("1");
                        if split_postff_normadd {
                            // Step 1: mlp_down = rms_norm(moe_accum, post_ff_norm_2)
                            session.barrier_between(
                                &[&self.activations.moe_accum,
                                  &self.layers[layer_idx].norms.post_feedforward_layernorm_2],
                                &[&self.activations.mlp_down],
                            );
                            session.rms_norm(
                                reg, metal_dev,
                                &self.activations.moe_accum,
                                &self.layers[layer_idx].norms.post_feedforward_layernorm_2,
                                &self.activations.mlp_down,
                                &self.activations.norm_params,
                                1, hs as u32,
                            ).map_err(|e| anyhow::anyhow!("split post-FF norm2 L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;

                            // Step 2: mlp_down = attn_out + mlp_down (in-place add)
                            session.barrier_between(
                                &[&self.activations.attn_out, &self.activations.mlp_down],
                                &[&self.activations.mlp_down],
                            );
                            mlx_native::ops::elementwise::elementwise_add(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.attn_out,
                                &self.activations.mlp_down,
                                &self.activations.mlp_down,
                                hs,
                                mlx_native::DType::F32,
                            ).map_err(|e| anyhow::anyhow!("split post-FF add L{layer_idx}: {e}"))?;
                        } else {
                        session.barrier_between(
                            &[&self.activations.attn_out, &self.activations.moe_accum],
                            &[&self.activations.mlp_down],
                        );
                        mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
                            session.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_out,
                            &self.activations.moe_accum,
                            &self.layers[layer_idx].norms.post_feedforward_layernorm_2,
                            &self.activations.mlp_down,
                            hs as u32, 1, eps,
                        ).map_err(|e| anyhow::anyhow!("fused post-FF norm2+combine L{layer_idx}: {e}"))?;
                        }
                        *total_dispatches += 1;

                        // -- Fused end-of-layer: post-FF norm + residual add + scalar mul --
                        // ADR-028 iter-208 sub-bisect: SKIP_END_OF_LAYER_FINAL
                        // skips only this final dispatch (keeps post-FF norm 2).
                        if !INVESTIGATION_ENV.skip_end_of_layer_final {
                            // ADR-029 iter-108 H78: env-gated SPLIT of the
                            // 3-op fused end-of-layer into 3 separate dispatches
                            // (rms_norm + add + scalar_mul). Only enabled when
                            // scalar_is_vector (gemma4 default) — otherwise
                            // fall back to fused since no scalar_mul_f32 kernel
                            // for non-vector scalar exists.
                            let split_postff_normaddscalar = std::env::var("HF2Q_SPLIT_POSTFF_NORMADDSCALAR").as_deref() == Ok("1");
                            if split_postff_normaddscalar && scalar_is_vector {
                                // Step 1: norm_out = rms_norm(mlp_down, post_ff_norm)
                                session.barrier_between(
                                    &[&self.activations.mlp_down,
                                      &self.layers[layer_idx].norms.post_feedforward_layernorm],
                                    &[&self.activations.norm_out],
                                );
                                session.rms_norm(
                                    reg, metal_dev,
                                    &self.activations.mlp_down,
                                    &self.layers[layer_idx].norms.post_feedforward_layernorm,
                                    &self.activations.norm_out,
                                    &self.activations.norm_params,
                                    1, hs as u32,
                                ).map_err(|e| anyhow::anyhow!("split endlayer norm L{layer_idx}: {e}"))?;
                                *total_dispatches += 1;

                                // Step 2: norm_out = residual + norm_out
                                session.barrier_between(
                                    &[&self.activations.residual, &self.activations.norm_out],
                                    &[&self.activations.norm_out],
                                );
                                mlx_native::ops::elementwise::elementwise_add(
                                    session.encoder_mut(), reg, metal_dev,
                                    &self.activations.residual,
                                    &self.activations.norm_out,
                                    &self.activations.norm_out,
                                    hs,
                                    mlx_native::DType::F32,
                                ).map_err(|e| anyhow::anyhow!("split endlayer add L{layer_idx}: {e}"))?;
                                *total_dispatches += 1;

                                // Step 3: hidden = norm_out * layer_scalar (elementwise)
                                session.barrier_between(
                                    &[&self.activations.norm_out,
                                      &self.layers[layer_idx].layer_scalar],
                                    &[&self.activations.hidden],
                                );
                                mlx_native::ops::elementwise::elementwise_mul(
                                    session.encoder_mut(), reg, metal_dev,
                                    &self.activations.norm_out,
                                    &self.layers[layer_idx].layer_scalar,
                                    &self.activations.hidden,
                                    hs,
                                    mlx_native::DType::F32,
                                ).map_err(|e| anyhow::anyhow!("split endlayer scalar L{layer_idx}: {e}"))?;
                                // *total_dispatches += 1 happens via fall-through outside
                            } else {
                            session.barrier_between(
                                &[&self.activations.residual, &self.activations.mlp_down],
                                &[&self.activations.hidden],
                            );
                            mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_scalar_f32(
                                session.encoder_mut(), reg, metal_dev,
                                &self.activations.residual,
                                &self.activations.mlp_down,
                                &self.layers[layer_idx].norms.post_feedforward_layernorm,
                                &self.activations.hidden,
                                &self.layers[layer_idx].layer_scalar,
                                1, hs as u32, eps,
                                scalar_is_vector,
                            ).map_err(|e| anyhow::anyhow!("fused end-of-layer L{layer_idx}: {e}"))?;
                            *total_dispatches += 1;
                            }
                        }
                    }
                }

                if let Some(ref mut p) = profile {
                    // All layer ops in single session — attribute everything to S1
                    p.s1_dispatches[layer_idx] = *total_dispatches;
                }

                // ADR-029 iter-110 — CPU-encoding/GPU-execution overlap via
                // split CB. When HF2Q_DECODE_SPLIT_CB_AT_LAYER=N is set,
                // commit (non-blocking) the current session at end of layer
                // N-1 and start a new session for the remaining layers.
                // Mirrors peer's dispatch_apply overlap pattern at
                // /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:550
                // — peer encodes multi-CB in parallel during GPU execution.
                // We achieve the same overlap WITHOUT worker threads by
                // splitting into 2 CBs (non-blocking commit on CB1, encode
                // CB2 while GPU runs CB1, commit_and_wait CB2 at end).
                //
                // GraphSession::commit() returns the CommandEncoder which
                // we drop — Metal retains the committed CB and runs it to
                // completion. Cross-CB buffer dependencies (residual,
                // hidden) resolve via MTLCommandQueue's in-order execution.
                let split_at_layer: Option<usize> = {
                    static SPLIT_AT: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
                    *SPLIT_AT.get_or_init(|| {
                        std::env::var("HF2Q_DECODE_SPLIT_CB_AT_LAYER")
                            .ok()
                            .and_then(|v| v.parse::<usize>().ok())
                    })
                };
                if let Some(n) = split_at_layer {
                    if layer_idx + 1 == n {
                        // End current session via commit (non-blocking).
                        // The returned encoder drops; Metal owns the
                        // committed CB and runs it to completion.
                        let prev_session = std::mem::replace(
                            session,
                            exec.begin().map_err(|e| anyhow::anyhow!("split CB begin: {e}"))?,
                        );
                        let _committed_enc = prev_session.commit();
                        // GPU begins executing CB1 immediately; CPU now
                        // proceeds to encode CB2 (layers n..num_layers + head).
                    }
                }

                // ADR-028 iter-292: per-layer dispatch attribution.
                if per_layer_disp_enabled {
                    let layer_disp_end = mlx_native::dispatch_count();
                    per_layer_disp_log.push((
                        layer_idx,
                        is_sliding,
                        layer_disp_end - layer_disp_start,
                    ));
                }

                // ADR-029 iter-9 — per-layer GPU TIME ground truth.
                // HF2Q_PER_LAYER_GPU_TIME=1 commits the session per-layer
                // and records GPU wall-clock via finish_with_gpu_time.
                // HF2Q_PER_LAYER_PHASE_GPU_TIME=1 also commits at the
                // attn/ffn boundary, so this commit at end-of-layer
                // reports just the FFN+EOL phase.
                let phase_split = std::env::var("HF2Q_PER_LAYER_PHASE_GPU_TIME").as_deref() == Ok("1");
                let per_layer = std::env::var("HF2Q_PER_LAYER_GPU_TIME").as_deref() == Ok("1");
                let ffn_split = std::env::var("HF2Q_FFN_SPLIT").as_deref() == Ok("1");
                if per_layer || phase_split || ffn_split {
                    let gpu_ns: u64 = std::mem::replace(session, exec.begin().map_err(|e| anyhow::anyhow!("per-layer-gpu-time begin L{layer_idx}: {e}"))?).finish_with_gpu_time()
                        .map_err(|e| anyhow::anyhow!("per-layer finish L{layer_idx}: {e}"))?;
                    let label = if ffn_split { "FFN_EOL  " }
                                else if phase_split { "PHASE_FFN" }
                                else { "PER_LAYER_GPU" };
                    eprintln!("    [{label} L{:02} {}] gpu={:>6.1}µs",
                        layer_idx,
                        if is_sliding { "S" } else { "G" },
                        gpu_ns as f64 / 1000.0);
                    session.track_dispatch(&[], &[&self.activations.hidden]);
                }

                // ADR-009 Phase 3A: per-layer hidden state dump.
                // Commits the session mid-forward to read hidden state, then re-starts.
                // Only active when HF2Q_DUMP_LAYERS=<seq_pos> matches.
                if dump_layers {
                    std::mem::replace(session, exec.begin()
            .map_err(|e| anyhow::anyhow!("dump layer re-begin L{layer_idx}: {e}"))?).finish()
                        .map_err(|e| anyhow::anyhow!("dump layer finish L{layer_idx}: {e}"))?;
                    dumps::dump_f32(&self.activations.hidden, hs,
                        "l_out", Some(layer_idx), seq_pos)?;
                    // Re-start session for remaining layers
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
                // ADR-028 iter-374: multi-split — commit at any of the
                // configured split points, not just the first.
                if dual_buffer_splits.contains(&(layer_idx + 1)) {
                    let b0_barriers = session.barrier_count();
                    let _b0_encoder = std::mem::replace(session, exec.begin().map_err(|e| anyhow::anyhow!("dual-buffer begin: {e}"))?).commit(); // commit current buf → GPU starts async
                    session.track_dispatch(&[], &[&self.activations.hidden]);
                    if INVESTIGATION_ENV.mlx_timing {
                        eprintln!("  [DUAL_BUFFER] split at layer {} — buf0: {} dispatches, {} barriers",
                            layer_idx + 1, *total_dispatches, b0_barriers);
                    }
                }
        Ok(())
    }

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
        // iter-23: HF2Q_DUMP_SDPA_MAX_POS=N — when set along with HF2Q_DUMP_ALL_CACHE=1,
        // dump sdpa_out for all layers at every decode STEP < N (decode-step index, not
        // absolute seq_pos, so it is prompt-length independent).
        // The counter increments on each forward_decode call regardless of layer.
        // This enables the Gate A cosine-sim harness without requiring N separate hf2q runs.
        //
        // W39 iter-112b: HF2Q_DUMP_SDPA_MAX_POS still uses a process-static
        // OnceLock (read once, never overridden — Gate H always wants the
        // same `tokens` window across both passes), but the decode-step
        // counter moved from a process-static `AtomicUsize` to per-instance
        // `self.decode_step_dump_counter` so `set_decode_regime` /
        // `set_replay_tokens` / `set_dump_overrides` can reset it between
        // the dense and TQ passes of a single Gate H run.  The
        // `dump_all_cache` read also consults the per-instance override
        // first (W39 iter-112b: `INVESTIGATION_ENV` LazyLock is frozen at
        // `main.rs::main` before parity_quality's runtime `set_var` lands).
        let dump_sdpa_max_pos: Option<usize> = {
            static MAX_POS: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
            *MAX_POS.get_or_init(|| {
                std::env::var("HF2Q_DUMP_SDPA_MAX_POS")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
            })
        };
        let decode_step_for_dump: usize = self.decode_step_dump_counter;
        self.decode_step_dump_counter = self.decode_step_dump_counter.saturating_add(1);
        let dump_all_cache_eff: bool = self
            .dump_all_cache_override
            .unwrap_or(INVESTIGATION_ENV.dump_all_cache);
        let dump_layers: bool = INVESTIGATION_ENV.dump_layers == Some(seq_pos)
            || dump_sdpa_max_pos.map_or(false, |max| {
                decode_step_for_dump < max && dump_all_cache_eff
            });

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

        // iter-21 Track B: lazy allocation of leg_hb_encoded on first decode.
        // forward_prefill may have already allocated this; if not, do it here.
        // We read from the INVESTIGATION_ENV LazyLock (parsed once at process start).
        {
            // ADR-007 post-close correction 2026-04-24: TQ-8-bit is the default when
            // env is unset. Explicit HF2Q_TQ_CODEBOOK_BITS=4 selects the legacy 4-bit
            // native flash_attn_vec_tq path (127-byte sourdough ceiling, not shippable
            // as default). Explicit =5/=6 select intermediate HB-SDPA. This MUST match
            // the primary gate at tq_codebook_bits below.
            // ADR-005 wave-1 T1.2: read from INVESTIGATION_ENV LazyLock (parsed once at
            // process start) instead of calling std::env::var per forward_decode call.
            let cb_bits: u32 = INVESTIGATION_ENV.tq_codebook_bits;
            // ADR-028 Phase 10c (iter-348): if hybrid_kv gate is on, route to
            // HybridKvBuffers (F16 K + TQ-HB V) instead of legacy HbKvBuffers
            // (TQ-HB K + TQ-HB V). Mutually exclusive at alloc time per the
            // Phase 10b struct comment; only one of `hybrid_kv` /
            // `leg_hb_encoded` is `Some(_)` for a given model instance.
            if cb_bits >= 5 && INVESTIGATION_ENV.hybrid_kv && self.hybrid_kv.is_none() {
                let (exec, _reg) = gpu.split();
                let dev = exec.device();
                let mut hybrid_vec: Vec<HybridKvBuffers> = Vec::with_capacity(num_layers);
                for layer_idx in 0..num_layers {
                    let nkv = self.layers[layer_idx].num_kv_heads;
                    let hd = self.layers[layer_idx].head_dim;
                    let is_ring = self.kv_caches[layer_idx].is_sliding;
                    let cap = self.kv_caches[layer_idx].capacity;
                    hybrid_vec.push(alloc_hybrid_kv_for_layer(dev, layer_idx, nkv, hd, cap, is_ring)?);
                }
                eprintln!("[ADR-028 Phase 10c] Allocated hybrid_kv ({} layers, F16 K + TQ-HB V {}-bit)",
                    num_layers, cb_bits);
                self.hybrid_kv = Some(hybrid_vec);
            } else if cb_bits >= 5 && !INVESTIGATION_ENV.hybrid_kv && self.leg_hb_encoded.is_none() {
                let (exec, _reg) = gpu.split();
                let dev = exec.device();
                // Use kv_caches[0] write_pos - 1 to infer linear capacity. In practice
                // we use the same capacity as kv_caches per layer (the KV cache was sized
                // for the full sequence at init time).
                let mut leg_hb_vec: Vec<HbKvBuffers> = Vec::with_capacity(num_layers);
                for layer_idx in 0..num_layers {
                    let nkv = self.layers[layer_idx].num_kv_heads;
                    let hd = self.layers[layer_idx].head_dim;
                    let is_ring = self.kv_caches[layer_idx].is_sliding;
                    let cap = self.kv_caches[layer_idx].capacity;
                    let norms_per_pos = (hd / 256).max(1);
                    let norms_n = nkv * cap * norms_per_pos;
                    // byte-packed: 1 byte per element
                    let k_packed = dev.alloc_buffer(nkv * cap * hd, mlx_native::DType::U8,
                        vec![nkv, cap, hd])
                        .map_err(|e| anyhow::anyhow!("leg_hb K packed L{layer_idx}: {e}"))?;
                    let k_norms = dev.alloc_buffer(norms_n * 4, mlx_native::DType::F32,
                        if norms_per_pos == 1 { vec![nkv, cap] } else { vec![nkv, cap, norms_per_pos] })
                        .map_err(|e| anyhow::anyhow!("leg_hb K norms L{layer_idx}: {e}"))?;
                    let v_packed = dev.alloc_buffer(nkv * cap * hd, mlx_native::DType::U8,
                        vec![nkv, cap, hd])
                        .map_err(|e| anyhow::anyhow!("leg_hb V packed L{layer_idx}: {e}"))?;
                    let v_norms = dev.alloc_buffer(norms_n * 4, mlx_native::DType::F32,
                        if norms_per_pos == 1 { vec![nkv, cap] } else { vec![nkv, cap, norms_per_pos] })
                        .map_err(|e| anyhow::anyhow!("leg_hb V norms L{layer_idx}: {e}"))?;
                    leg_hb_vec.push(HbKvBuffers {
                        k_packed, k_norms, v_packed, v_norms,
                        capacity: cap, is_sliding: is_ring, norms_per_pos,
                    });
                }
                eprintln!("[iter-21 Track B] Allocated leg_hb_encoded ({} layers, {}-bit)", num_layers, cb_bits);
                self.leg_hb_encoded = Some(leg_hb_vec);
            }
            // iter-222 (2026-05-01): the lazy-allocate of `leg_f_kvs` shadow
            // cache that lived here was deleted along with the iter-34
            // dense-on-shadow Leg F decode branch — `flash_attn_vec_tq_hb`
            // consumes `leg_hb_encoded` directly with no F32 round-trip.
        }

        // =====================================================================
        // SINGLE SESSION: Embedding + All 30 Layers + Head
        //
        // ONE begin() → all GPU dispatches → ONE finish().
        // Zero CPU readbacks.  All norms, adds, MoE routing, scalar multiplies,
        // softcap, and argmax run on GPU.
        // =====================================================================
        // iter-18 S2B: D=512 per-block scale factor for encoder+decoder ablation.
        // HF2Q_SCALE_FORMULA: bare (1.0), sqrt256 (16.0), sqrt512 (≈22.627).
        // Read once per decode call; passed to dispatch_hadamard_quantize_kv + SDPA params.
        let tq_scale_factor_d512: f32 = {
            static SCALE_FACTOR: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
            *SCALE_FACTOR.get_or_init(|| {
                match std::env::var("HF2Q_SCALE_FORMULA").as_deref() {
                    Ok("sqrt256") => {
                        eprintln!("[HF2Q_SCALE_FORMULA] D=512 scale_factor = sqrt(256) = 16.0");
                        16.0_f32
                    }
                    Ok("sqrt512") => {
                        let v = 512.0_f32.sqrt();
                        eprintln!("[HF2Q_SCALE_FORMULA] D=512 scale_factor = sqrt(512) = {v:.4}");
                        v
                    }
                    Ok("bare") | Err(_) => {
                        // Default: bare (iter-16 control state)
                        1.0_f32
                    }
                    Ok(other) => {
                        eprintln!("[HF2Q_SCALE_FORMULA] unknown value {other:?}; using bare (1.0)");
                        1.0_f32
                    }
                }
            })
        };

        // iter-222 (ADR-005 closure, 2026-05-01): the iter-34 `force_dense_sdpa_on_tq_kv`
        // gate that lived here was deleted — see file-level iter-222 closure
        // note above the (now-deleted) `dense_sdpa_on_tq_kv_enabled()` site for
        // rationale. TQ-regime SDPA now flows through the inline-fused
        // `flash_attn_vec_tq` (cb_bits=4) / `flash_attn_vec_tq_hb` (cb_bits>=5)
        // kernels unconditionally — peer-correct production paths that read
        // TQ-packed K/V directly with no F32 shadow-cache round-trip.

        // iter-21 Track B + 2026-04-24 post-close default correction.
        // HF2Q_TQ_CODEBOOK_BITS selects the KV codebook width.
        //   unset  (DEFAULT) = 8-bit native HB SDPA (2× memory savings vs F16, 0.017 PPL
        //                      absolute / 1.24% delta, cosine 0.9998 — meets TurboQuant
        //                      paper + KIVI + KVQuant + AmesianX + vLLM published gates)
        //   "4"              = legacy 4-bit native flash_attn_vec_tq (iter-16 control;
        //                      127-byte sourdough ceiling — not shippable as default)
        //   "5" | "6"        = intermediate higher-bit HB SDPA (Lloyd-Max native)
        //   "8"              = explicit 8-bit (same as unset)
        // MUST stay in lockstep with the `cb_bits` lazy-alloc gate above.
        let tq_codebook_bits: u32 = {
            static CODEBOOK_BITS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
            *CODEBOOK_BITS.get_or_init(|| {
                match std::env::var("HF2Q_TQ_CODEBOOK_BITS").as_deref() {
                    Ok("4") => {
                        eprintln!("[HF2Q_TQ_CODEBOOK_BITS] 4-bit legacy TQ (opt-in; 127-byte sourdough ceiling)");
                        0u32
                    }
                    Ok("5") => {
                        eprintln!("[HF2Q_TQ_CODEBOOK_BITS] 5-bit Lloyd-Max native HB SDPA");
                        5u32
                    }
                    Ok("6") => {
                        eprintln!("[HF2Q_TQ_CODEBOOK_BITS] 6-bit Lloyd-Max native HB SDPA");
                        6u32
                    }
                    Ok("8") | Err(_) => {
                        eprintln!("[HF2Q_TQ_CODEBOOK_BITS] 8-bit Lloyd-Max native HB SDPA (default)");
                        8u32
                    }
                    Ok(other) => {
                        eprintln!("[HF2Q_TQ_CODEBOOK_BITS] unknown value {:?}; defaulting to 8-bit", other);
                        8u32
                    }
                }
            })
        };
        // iter-24: native HB SDPA via `flash_attn_vec_tq_hb` for cb_bits >= 5
        // (default 8). Reads TQ-packed K/V directly from `leg_hb_encoded` —
        // no F32 shadow-cache round-trip.
        let use_native_hb_sdpa = tq_codebook_bits >= 5;

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
            // ADR-028 iter-374: comma-separated list supported (e.g. "2,10,20").
            let dual_buffer_split: Option<usize> =
                INVESTIGATION_ENV.dual_buffer_split(num_layers);
            let dual_buffer_splits: Vec<usize> =
                INVESTIGATION_ENV.dual_buffer_splits(num_layers);
            let _ = dual_buffer_split;

            // --- 2. Transformer layers ---
            // Phase 3A: sub-layer detail dump (which specific layer to break down)
            let dump_detail_layer: Option<usize> = INVESTIGATION_ENV.dump_layer_detail;
            // iter-18 S2C: first-divergence dump for layer 0 (sliding, hd=256), decode positions 1..=10.
            // Gate: HF2Q_DUMP_SLIDING_LAYER_0=1 env var. Run name: HF2Q_DUMP_RUN_NAME (dense|tq).
            // ADR-005 wave-1 T1.2: read from INVESTIGATION_ENV LazyLock (parsed once at process start).
            let dump_sliding_l0: bool = INVESTIGATION_ENV.dump_sliding_layer_0;
            let dump_run_name: Option<&str> = INVESTIGATION_ENV.dump_run_name.as_deref();

            // ADR-028 iter-292: per-layer dispatch attribution
            // (HF2Q_PER_LAYER_DISP=1). Snapshot dispatch_count at layer start;
            // diff at layer end gives dispatches-per-layer-type.  Localizes
            // the iter-291 +72 mat-vec gap (sliding vs full-attn layers).
            let per_layer_disp_enabled = std::env::var("HF2Q_PER_LAYER_DISP").as_deref() == Ok("1");
            let mut per_layer_disp_log: Vec<(usize, bool, u64)> = Vec::new();
            let ctx = super::layer_ctx::LayerCtx {
                seq_pos,
                hidden_size: hs,
                kv_info: &kv_info,
                dump_layers,
                dump_detail_layer,
                dump_sliding_l0,
                dump_run_name,
                dual_buffer_splits: &dual_buffer_splits,
                per_layer_disp_enabled,
                tq_scale_factor_d512,
                tq_codebook_bits,
                use_native_hb_sdpa,
                dump_all_cache_eff,
            };
            for layer_idx in 0..num_layers {
                self.encode_one_layer(
                    layer_idx, &ctx, &mut s, exec, reg,
                    profile, &mut per_layer_disp_log, &mut total_dispatches,
                )?;
            }

            // ADR-028 iter-292: dump per-layer dispatch counts post-loop.
            if per_layer_disp_enabled && !per_layer_disp_log.is_empty() {
                let n_sliding = per_layer_disp_log.iter().filter(|(_, s, _)| *s).count();
                let n_full = per_layer_disp_log.len() - n_sliding;
                let total_sliding: u64 = per_layer_disp_log.iter()
                    .filter(|(_, s, _)| *s).map(|(_, _, n)| *n).sum();
                let total_full: u64 = per_layer_disp_log.iter()
                    .filter(|(_, s, _)| !*s).map(|(_, _, n)| *n).sum();
                let avg_sliding = if n_sliding > 0 { total_sliding / n_sliding as u64 } else { 0 };
                let avg_full = if n_full > 0 { total_full / n_full as u64 } else { 0 };
                eprintln!("[PER_LAYER_DISP] sliding_layers={} (avg {} disp/layer, total {})",
                    n_sliding, avg_sliding, total_sliding);
                eprintln!("[PER_LAYER_DISP] full_layers={} (avg {} disp/layer, total {})",
                    n_full, avg_full, total_full);
                for (idx, sliding, count) in &per_layer_disp_log {
                    eprintln!("[PER_LAYER_DISP]   L{:02} {} {} disp",
                        idx, if *sliding { "SLID" } else { "FULL" }, count);
                }
            }

            // --- Body/Head timing split (HF2Q_SPLIT_TIMING=1) ---
            // Inserts a commit_and_wait between layers and head to measure each
            // GPU section separately. Adds ~50μs sync overhead — measurement only.
            let body_dispatches = total_dispatches;
            let split_timing = INVESTIGATION_ENV.split_timing;
            // ADR-028 iter-312 — group-stats dump for barrier audit.
            let group_stats_enabled = std::env::var("HF2Q_GROUP_STATS")
                .ok()
                .as_deref()
                .map_or(false, |v| v == "1");
            if split_timing {
                let body_barriers = s.barrier_count();
                if group_stats_enabled {
                    s.dump_group_stats();
                }
                let (enc_ns, gpu_ns) = s.finish_with_timing(session_start)
                    .map_err(|e| anyhow::anyhow!("body finish: {e}"))?;
                eprintln!("  [SPLIT] BODY: encode={:.2}ms gpu={:.2}ms dispatches={} barriers={}",
                    enc_ns as f64 / 1e6, gpu_ns as f64 / 1e6, body_dispatches, body_barriers);
                // Start a new session for the head
                s = exec.begin().map_err(|e| anyhow::anyhow!("head session: {e}"))?;
            } else if group_stats_enabled {
                s.dump_group_stats();
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

            // GPU lm_head: prefer Q6_K-native (HF2Q_LMHEAD_Q6K=1, ADR-028
            // iter-188), then Q8_0 (HF2Q_LMHEAD_Q8 auto for large vocab),
            // then F16 dense.
            if let Some(ref q6k) = self.lm_head_q6k {
                s.barrier_between(
                    &[&self.activations.norm_out, &q6k.buffer],
                    &[&self.activations.logits],
                );
                dispatch_qmatmul(
                    &mut s, reg, dev,
                    &self.activations.norm_out,
                    q6k,
                    &self.activations.logits,
                    1,
                )?;
                total_dispatches += 1;
            } else if let Some(ref q8) = self.lm_head_q8 {
                s.barrier_between(
                    &[&self.activations.norm_out, &q8.buffer],
                    &[&self.activations.logits],
                );
                dispatch_qmatmul(
                    &mut s, reg, dev,
                    &self.activations.norm_out,
                    q8,
                    &self.activations.logits,
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
                    // ADR-028 iter-400: per-barrier wall measurement.
                    let bns = mlx_native::barrier_total_ns();
                    if bns > 0 && barrier_count > 0 {
                        eprintln!("  [BARRIER_PROFILE] total_ns={} per_barrier_ns={}",
                            bns, bns / barrier_count as u64);
                    }
                }
            } else {
                let head_barriers = barrier_count; // snapshot before finish consumes s
                let (enc_ns, gpu_ns) = s.finish_with_timing(session_start)
                    .map_err(|e| anyhow::anyhow!("single session finish: {e}"))?;
                if INVESTIGATION_ENV.mlx_timing {
                    // ADR-028 iter-400: per-barrier wall measurement.
                    let bns = mlx_native::barrier_total_ns();
                    if bns > 0 && barrier_count > 0 {
                        eprintln!("  [BARRIER_PROFILE] total_ns={} per_barrier_ns={}",
                            bns, bns / barrier_count as u64);
                    }
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
        // ADR-028 iter-188: Q6_K-direct lm_head also has quantization noise
        // and benefits from rerank against the F16 oracle (when in compare
        // mode).  Rerank fires for any quantized lm_head path.
        let rerank_active = (self.lm_head_q8.is_some() || self.lm_head_q6k.is_some())
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
            // Sort descending by logit.  NaN logits would crash
            // `partial_cmp().unwrap()` — sort them as the smallest
            // possible value (they land at the end of the descending
            // sort so the top-10 print stays useful) and surface a
            // separate NaN-count line below so the operator can see
            // the model is producing garbage logits without losing
            // the diagnostic itself.  Surfaced 2026-04-25 by the
            // ADR-005 iter-103 vision smoke test (mmproj load was
            // making warmup logits NaN; the diagnostic block crashed
            // before printing anything).
            indexed.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let n_nan = indexed.iter().filter(|(_, v)| v.is_nan()).count();
            if n_nan > 0 {
                eprintln!(
                    "[PAD-DIAG] WARNING: {} of {} logits are NaN — model produced garbage; \
                     pad-win is a downstream symptom",
                    n_nan, vocab
                );
            }
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

        // -----------------------------------------------------------------
        // ADR-007 Gate H release-check plumbing (W12 iter-108a blocker #1).
        //
        // Three coupled hooks, gated by env vars cached at process start:
        //
        //   HF2Q_DECODE_INPUT_TOKENS  → replay fixed tokens (override pick).
        //                               The argmax + Q8 rerank above already
        //                               ran, so cosine/NLL captures see live
        //                               logits — only the *picked* token is
        //                               replaced.  Falls through to the
        //                               sampler's pick once the replay is
        //                               exhausted.
        //   HF2Q_EMIT_NLL             → emit `[HF2Q_NLL] step=N token=X
        //                               nll=Y` per token (matches the audit
        //                               binaries' `parse_nll_values` regex).
        //                               The NLL is computed on the FINAL
        //                               picked token (post-replay), so a
        //                               TQ-active run replaying dense
        //                               tokens reports each replayed
        //                               token's NLL under the TQ logits —
        //                               this is the ADR-007 Gate C PPL
        //                               input shape.
        //   HF2Q_DECODE_EMIT_TOKENS   → emit `[HF2Q_DECODE_EMIT] step=N
        //                               token=X` per token (matches
        //                               `parse_emitted_tokens`).
        //
        // All three were previously honored only by the audit-binary
        // wrappers (`src/bin/iter2{3,4,5}_audit.rs`) shelling out to a
        // separate hf2q subprocess; per-token NLL/replay never reached the
        // production decode path.  iter-108b will replace the audit
        // binaries with a release-check.sh-driven Gate 5; for that to
        // work, the production binary itself must honor the contract.
        //
        // iter-108a-fix (W15, 2026-04-25): the entire block below is gated
        // behind `self.gate_h_inactive` so that pre-iter-108a per-token
        // cost is restored when no Gate H hooks are armed (W14b regression
        // 95.0 → 100.6 tok/s baseline). The flag is computed once at
        // construction (`from_gguf_with_options`) and refreshed only by
        // `set_decode_regime`; reading it here is a single field load + a
        // single branch, which LLVM/the M-series CPU hoists out of any
        // surrounding loop and skips entirely on the false branch.
        // -----------------------------------------------------------------
        if !self.gate_h_inactive {
            let env = &*INVESTIGATION_ENV;
            let step = self.decode_step;
            // Replay first — substitute picked token before NLL/emit so both
            // downstream observers see the SAME token id (otherwise a replay
            // run would emit replay tokens but NLL the original argmax pick).
            // W21 iter-108b: per-instance `replay_tokens` takes precedence
            // over the frozen env-var vector so the in-process two-regime
            // Gate H harness can switch replay sources between passes.
            let final_token = if !self.replay_tokens.is_empty()
                && (step as usize) < self.replay_tokens.len()
            {
                self.replay_tokens[step as usize]
            } else if !env.decode_input_tokens.is_empty()
                && (step as usize) < env.decode_input_tokens.len()
            {
                env.decode_input_tokens[step as usize]
            } else {
                token_id
            };
            if env.emit_nll {
                // token_nll_from_logits asserts token_id < vocab_size; replay
                // tokens come from the user, so guard against an out-of-vocab
                // entry rather than panicking the whole decode loop.
                if (final_token as usize) < self.vocab_size {
                    match self.token_nll_from_logits(final_token) {
                        Ok(nll) => eprintln!(
                            "[HF2Q_NLL] step={step} token={final_token} nll={nll:.6}"
                        ),
                        Err(e) => eprintln!(
                            "[HF2Q_NLL] step={step} token={final_token} error={e}"
                        ),
                    }
                } else {
                    eprintln!(
                        "[HF2Q_NLL] step={step} token={final_token} \
                         error=token_id_out_of_vocab vocab_size={}",
                        self.vocab_size
                    );
                }
            }
            if env.decode_emit_tokens {
                eprintln!("[HF2Q_DECODE_EMIT] step={step} token={final_token}");
            }
            // Only mutate decode_step when Gate H is active — pre-iter-108a
            // the field did not exist, and writing to it every token even
            // when no observer reads it is a per-token RMW that defeats
            // the rest of the elision.
            self.decode_step = self.decode_step.saturating_add(1);
            return Ok(final_token);
        }

        Ok(token_id)
    }

    // forward_prefill() is defined in forward_prefill.rs (ADR-009 Track 1).

    /// ADR-028 iter-123 / ADR-029 Phase 2 Shape S — serial spec-decode verify.
    ///
    /// Forwards each `tokens[i]` through the model at position `seq_pos + i`,
    /// collecting the model's argmax at each step. Returns `Vec<u32>` of
    /// argmaxes (length == `tokens.len()`).
    ///
    /// **Shape S contract**: each token is a full `forward_decode` (own
    /// `GraphSession` begin/finish + `commit_and_wait`). This runs at
    /// `K × default-decode-latency` — NO speedup vs default decode.
    ///
    /// Use case: byte-identity correctness gate for the `accept_prefix`
    /// wiring + `rollback_kv` helper. Shape B (batched single-pass) lands
    /// later for the actual speed lift.
    ///
    /// At greedy temperature, `forward_decode_verify_serial(&[t0, t1, t2])`
    /// produces argmaxes byte-identical to calling `forward_decode(t0)`
    /// then `forward_decode(t1)` then `forward_decode(t2)` independently.
    pub fn forward_decode_verify_serial(
        &mut self,
        tokens: &[u32],
        seq_pos: usize,
        gpu: &mut GpuContext,
    ) -> Result<Vec<u32>> {
        let mut argmaxes = Vec::with_capacity(tokens.len());
        for (i, &tok) in tokens.iter().enumerate() {
            let mut prof: Option<TokenProfile> = None;
            let argmax = self.forward_decode(tok, seq_pos + i, gpu, &mut prof)?;
            argmaxes.push(argmax);
        }
        Ok(argmaxes)
    }

    /// ADR-028 iter-123 / ADR-029 Phase 2 — KV-cache rollback after partial accept.
    ///
    /// Rolls back the last `trim` writes across all layers. Sliding-window
    /// caches wrap (write_pos modulo capacity); full-attention caches go
    /// monotonic. The math is delegated to
    /// [`crate::inference::spec_decode::verifier::rollback_kv_state`] —
    /// see its tests for invariants.
    ///
    /// After this call, the next `forward_decode`/`forward_decode_verify_serial`
    /// invocation resumes at `current_seq_pos - trim`. The K_packed/V_packed
    /// data past the new `seq_len` is left as garbage; this is safe because
    /// kernels only read `< seq_len` and writes always go to current
    /// `write_pos`.
    pub fn rollback_kv(&mut self, trim: usize) {
        for cache in &mut self.kv_caches {
            let (wp, sl) = crate::inference::spec_decode::verifier::rollback_kv_state(
                cache.write_pos,
                cache.seq_len,
                cache.capacity,
                cache.is_sliding,
                trim,
            );
            cache.write_pos = wp;
            cache.seq_len = sl;
        }
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

        // iter-18 S2B: scale factor for kernel profile path (same OnceLock as forward_decode).
        let tq_scale_factor_d512: f32 = {
            static SCALE_FACTOR_KP: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
            *SCALE_FACTOR_KP.get_or_init(|| match std::env::var("HF2Q_SCALE_FORMULA").as_deref() {
                Ok("sqrt256") => 16.0_f32,
                Ok("sqrt512") => 512.0_f32.sqrt(),
                _ => 1.0_f32,
            })
        };

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
                    &self.layers[layer_idx].attn.q_proj, &self.activations.attn_q, 1)?;
                // K proj
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    &self.layers[layer_idx].attn.k_proj, &self.activations.attn_k, 1)?;
                // V proj (if not k_eq_v)
                if !v_is_k {
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                        self.layers[layer_idx].attn.v_proj.as_ref().unwrap(),
                        &self.activations.attn_v, 1)?;
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
                    Some(tq_scale_factor_d512),
                    None,
                ).map_err(|e| anyhow::anyhow!("hadamard_quantize K L{layer_idx}: {e}"))?;
                mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                    s.encoder_mut(), reg, metal_dev,
                    v_src,
                    &self.kv_caches[layer_idx].v_packed,
                    &self.kv_caches[layer_idx].v_norms,
                    nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                    kv_is_sliding,
                    Some(tq_scale_factor_d512),
                    None,
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
                    // ADR-009 Track 2 + iter-25 Subtask B fix: ring_start must be the
                    // physical slot of the OLDEST entry (not newest).
                    // kv_write_pos is pre-increment (the slot just written this step).
                    // After wrap: oldest = (kv_write_pos + 1) % capacity.
                    // The kernel formula: logical_idx = (k_pos - ring_start + cap) % cap
                    // maps ring_start → logical 0 (oldest). Matches HB dispatch.
                    let ring_start = if kv_is_sliding && kv_seq_len >= kv_capacity {
                        ((kv_write_pos + 1) % kv_capacity) as u32
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
                        scale_factor_d512: tq_scale_factor_d512,
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
                    &self.layers[layer_idx].attn.o_proj, &self.activations.attn_out, 1)?;

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
                    &self.layers[layer_idx].mlp.gate_proj, &self.activations.mlp_gate, 1)?;
                // up
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    &self.layers[layer_idx].mlp.up_proj, &self.activations.mlp_up, 1)?;
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
                    &self.layers[layer_idx].mlp.down_proj, &self.activations.mlp_down, 1)?;

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
                    &self.activations.moe_router_logits, 1)?;

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
                    &self.activations.moe_gate_up_id_out,
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
                    &self.activations.moe_down_id_out,
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

            // ADR-029 §Decision item 3: kernel-profile lm_head — mirror
            // production single-session path at ~4818: prefer Q6_K-native
            // (HF2Q_LMHEAD_Q6K=1, ADR-028 iter-188/345, default-on when
            // token_embd.weight is Q6_K on-disk), then Q8_0 (auto for
            // big-vocab models like gemma4 262144), then F16 dense.
            //
            // Pre-ADR-029 this path only checked Q8_0/F16 and hard-failed
            // on gemma4-APEX-Q5_K_M (Q6_K token_embd) — blocking all
            // per-kernel-type buckets needed for MoE-1/-2/-3 audits.
            if let Some(ref q6k) = self.lm_head_q6k {
                s.barrier_between(
                    &[&self.activations.norm_out, &q6k.buffer],
                    &[&self.activations.logits],
                );
                dispatch_qmatmul(
                    &mut s, reg, dev,
                    &self.activations.norm_out,
                    q6k,
                    &self.activations.logits,
                    1,
                )?;
            } else if let Some(ref q8) = self.lm_head_q8 {
                s.barrier_between(
                    &[&self.activations.norm_out, &q8.buffer],
                    &[&self.activations.logits],
                );
                dispatch_qmatmul(
                    &mut s, reg, dev,
                    &self.activations.norm_out,
                    q8,
                    &self.activations.logits,
                    1,
                )?;
            } else if let Some(ref lm_head_f16) = self.lm_head_f16 {
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
                anyhow::bail!(
                    "Kernel profile requires GPU lm_head (Q6_K, Q8_0, or F16 weight)"
                );
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

    /// Read the `[vocab_size]` F32 logits buffer that was produced by the
    /// most recent `forward_decode` (or `forward_prefill`) call.
    ///
    /// Returns a borrowed slice into `self.activations.logits` (no copy).
    /// Caller holds the borrow until they drop the reference; no further
    /// `self.activations` reads invalidate this slice (the underlying
    /// `MlxBuffer` is reused across decode steps but writes only happen
    /// inside the next `forward_decode` invocation).
    ///
    /// **Phase 2a Task #5 / #7 hook (iter-94).**  This is the logits-side
    /// surface that the chat decode loop consults when any Tier 2/3/4
    /// sampling field (`temperature`, `top_p`, `top_k`, `repetition_penalty`,
    /// `logit_bias`) is non-default — it bypasses `forward_decode`'s on-GPU
    /// greedy argmax and runs the pure-Rust `sampler_pure::sample_token`
    /// over these logits instead.  When grammar lands (iter-95+), the same
    /// hook supplies the logits to the GBNF mask before sampling.
    ///
    /// The logits include any post-softcap that the kernel applied
    /// (`final_logit_softcapping` from the GGUF metadata) — sampling
    /// operates on the same logits the on-GPU argmax would have seen.
    ///
    /// # Errors
    /// Forwarded from `MlxBuffer::as_slice` — fails only if the buffer
    /// is in an unreadable state (typically: never written by any
    /// preceding forward call).
    pub fn logits_view(&self) -> Result<&[f32]> {
        let slice: &[f32] = self.activations.logits.as_slice()
            .map_err(|e| anyhow::anyhow!("logits_view read: {e}"))?;
        let v = self.vocab_size;
        anyhow::ensure!(
            slice.len() >= v,
            "logits_view: buffer length {} < vocab_size {}",
            slice.len(), v
        );
        Ok(&slice[..v])
    }

    /// Compute NLL (negative log-likelihood) for `token_id` from the logits buffer
    /// that was produced by the most recent `forward_decode` call.
    ///
    /// Uses log-sum-exp for numerical stability. The logits may include a soft-cap
    /// (tanh * 30) already applied by the kernel; we use them as-is since both
    /// dense and TQ paths apply the same cap, so the relative NLL is fair.
    ///
    /// Returns: -log P(token_id) under the softmax distribution.
    /// Call ONLY immediately after `forward_decode`; the logits buffer is live.
    ///
    /// Public surface for downstream eval/scoring crates; no internal caller.
    #[allow(dead_code)]
    pub fn token_nll_from_logits(&self, token_id: u32) -> Result<f32> {
        let logits: &[f32] = self.activations.logits.as_slice()
            .map_err(|e| anyhow::anyhow!("token_nll logits read: {e}"))?;
        let v = self.vocab_size;
        anyhow::ensure!(
            (token_id as usize) < v,
            "token_nll: token_id {token_id} >= vocab_size {v}"
        );
        let slice = &logits[..v];
        // Log-sum-exp with max subtraction for numerical stability.
        let max_logit = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f64 = slice.iter()
            .map(|&l| ((l - max_logit) as f64).exp())
            .sum();
        let log_sum_exp = max_logit as f64 + sum_exp.ln();
        let log_prob = (logits[token_id as usize] as f64) - log_sum_exp;
        Ok(-log_prob as f32)
    }

    // =========================================================================
    // ADR-017 Phase B-tq.3 — engine-side TQ-packed snapshot/restore hooks
    // =========================================================================
    //
    // These bridge the runtime `MlxKvCache` byte buffers to the
    // `tq_packed_v2` envelope codec at
    // `serve::kv_persist::families::tq_packed`.  They run AFTER
    // `dispatch_hadamard_quantize_kv` has committed (caller is responsible
    // for issuing `s.finish()` first — there's no implicit barrier here)
    // so the live K/V packed buffers carry the post-quantize Lloyd-Max
    // indices + per-token-per-head FWHT magnitudes.
    //
    // Snapshot path:
    //   `tq_v2_snapshot_block(layer, range, bits, flags, scale)`
    //     → reads `kv_caches[layer].{k_packed, k_norms, v_packed, v_norms}`
    //     → packs two `tq_packed_v2` envelopes (one for K, one for V)
    //     → returns `(k_payload, v_payload)` ready for
    //       `TqPackedSpill::insert_block(layer, range.start, ..)` × 2
    //       (the spiller stores K + V under different (layer, range)
    //       keys; convention: K at `range.start`, V at `range.start +
    //       0x80_00_00_00` — see callers).
    //
    // Restore path is the inverse: takes the two payloads + writes back
    // into the live MlxKvCache buffers at the correct (head, position,
    // hd_packed) offsets.
    //
    // BOTH operations require a prior `commit_and_wait` on the encoder
    // session if the caller has issued any GPU work targeting these
    // buffers — this method does NOT issue its own barrier (callers
    // already control session boundaries via `exec.begin/finish`).

    /// Capture (K, V) `tq_packed_v2` envelope payloads from a token range
    /// of `kv_caches[layer_rank]`.  See module-level B-tq.3 doc for the
    /// barrier preconditions.
    ///
    /// `bits_per_coord` MUST match the active codec at quantize time —
    /// production default is 4 (nibble-packed) per ADR-007 §3 default
    /// configuration.  `flags` should set `HADAMARD_ROTATED` whenever
    /// the runtime applied FWHT before quantizing (the production path
    /// always does).  `scale` is the per-block multiplicative scale
    /// (typically 1.0 since the magnitude lives in the per-token norms).
    ///
    /// `#[allow(dead_code)]` because activation lives behind the
    /// `TqPackedSpillFactory` registration in `cmd_serve` (operator-
    /// controlled, deferred per ADR-007 reopen Path C clearance).  The
    /// method's correctness is exercised via the byte-level helpers'
    /// unit tests at `serve::kv_persist::families::tq_packed::tests::
    /// tq_v2_capture_restore_byte_identity`.
    #[allow(dead_code)]
    pub fn tq_v2_snapshot_block(
        &self,
        layer_rank: usize,
        range: std::ops::Range<u32>,
        bits_per_coord: crate::serve::kv_persist::families::tq_packed::TqBitsPerCoord,
        flags: u32,
        scale: f64,
    ) -> Result<(Vec<u8>, Vec<u8>), crate::serve::multi_model::SpillErrorKind> {
        use crate::serve::kv_persist::families::tq_packed;
        use crate::serve::multi_model::SpillErrorKind;

        // **B-tq.7** — at bits >= 5 the runtime stores K/V in
        // `leg_hb_encoded[layer_rank]` (1 byte per coord, shape
        // `[nkv, capacity, head_dim]`); at bits == 4 it stores in
        // `kv_caches[layer_rank].k_packed` (nibble-packed, shape
        // `[nkv, capacity, head_dim/2]`).  The active SDPA reads
        // from the matching buffer; snapshot must do the same.
        //
        // Branch up-front so the seq_len gate, shape derivation,
        // and byte reads all use the same buffer.
        let use_hb = bits_per_coord.0 >= 5;

        let (k_packed_bytes, k_norms_f32, v_packed_bytes, v_norms_f32, capacity_runtime, hd_packed_runtime, n_kv_heads_runtime, seq_len_live):
            (&[u8], &[f32], &[u8], &[f32], usize, usize, usize, usize) = if use_hb {
            let hb = self
                .leg_hb_encoded
                .as_ref()
                .ok_or(SpillErrorKind::CodecErr)?;
            let lay = hb.get(layer_rank).ok_or(SpillErrorKind::CodecErr)?;
            let cache = self
                .kv_caches
                .get(layer_rank)
                .ok_or(SpillErrorKind::CodecErr)?;
            // HB shares the kv_caches' seq_len bookkeeping (same
            // forward_decode increments both); read from kv_caches.
            (
                lay.k_packed.as_slice::<u8>().map_err(|_| SpillErrorKind::IoErr)?,
                lay.k_norms.as_slice::<f32>().map_err(|_| SpillErrorKind::IoErr)?,
                lay.v_packed.as_slice::<u8>().map_err(|_| SpillErrorKind::IoErr)?,
                lay.v_norms.as_slice::<f32>().map_err(|_| SpillErrorKind::IoErr)?,
                lay.capacity,
                lay.k_packed.shape().get(2).copied().unwrap_or(0),
                lay.k_packed.shape().first().copied().unwrap_or(0),
                cache.seq_len,
            )
        } else {
            let cache = self
                .kv_caches
                .get(layer_rank)
                .ok_or(SpillErrorKind::CodecErr)?;
            (
                cache.k_packed.as_slice::<u8>().map_err(|_| SpillErrorKind::IoErr)?,
                cache.k_norms.as_slice::<f32>().map_err(|_| SpillErrorKind::IoErr)?,
                cache.v_packed.as_slice::<u8>().map_err(|_| SpillErrorKind::IoErr)?,
                cache.v_norms.as_slice::<f32>().map_err(|_| SpillErrorKind::IoErr)?,
                cache.capacity,
                cache.k_packed.shape().get(2).copied().unwrap_or(0),
                cache.k_packed.shape().first().copied().unwrap_or(0),
                cache.seq_len,
            )
        };

        // Gate snapshot on live state (B-tq.4 iter-5 fix; same logic
        // for both buffer paths).
        if (range.start as usize) >= seq_len_live {
            return Err(SpillErrorKind::CodecErr);
        }

        let n_kv_heads = n_kv_heads_runtime as u32;
        let capacity = capacity_runtime as u32;
        // head_dim derives from the runtime packed-buffer row stride.
        // 4-bit (kv_caches): `hd_packed = head_dim/2`, hd = hd_packed*8/4 = hd_packed*2.
        // 8-bit (leg_hb_encoded): `hd_packed = head_dim`, hd = hd_packed*8/8 = hd_packed.
        // Generalised: `hd = hd_packed * 8 / bits`.
        let head_dim_bits = (hd_packed_runtime as u64) * 8;
        if head_dim_bits % (bits_per_coord.0 as u64) != 0 {
            return Err(SpillErrorKind::CodecErr);
        }
        let head_dim = (head_dim_bits / (bits_per_coord.0 as u64)) as u32;
        // F32 → LE bytes via per-element `to_le_bytes` to avoid an extra
        // dep.  Hot path is amortised — snapshot fires per block, not
        // per token.
        let k_norms_le: Vec<u8> = f32_slice_to_le_bytes(k_norms_f32);
        let v_norms_le: Vec<u8> = f32_slice_to_le_bytes(v_norms_f32);

        let k_payload = tq_packed::capture_tq_v2_payload_from_buffers(
            k_packed_bytes,
            &k_norms_le,
            capacity,
            n_kv_heads,
            head_dim,
            bits_per_coord,
            range.clone(),
            flags,
            scale,
        )?;
        let v_payload = tq_packed::capture_tq_v2_payload_from_buffers(
            v_packed_bytes,
            &v_norms_le,
            capacity,
            n_kv_heads,
            head_dim,
            bits_per_coord,
            range,
            flags,
            scale,
        )?;
        Ok((k_payload, v_payload))
    }

    /// Restore (K, V) `tq_packed_v2` envelope payloads into a token
    /// range of `kv_caches[layer_rank]`.  Inverse of
    /// [`Self::tq_v2_snapshot_block`].  Writes through `as_mut_slice` —
    /// callers MUST hold exclusive access to the live KV cache (the
    /// engine's per-session mutex).
    ///
    /// `#[allow(dead_code)]` for the same reason as
    /// [`Self::tq_v2_snapshot_block`].
    #[allow(dead_code)]
    pub fn tq_v2_restore_block(
        &mut self,
        layer_rank: usize,
        range: std::ops::Range<u32>,
        bits_per_coord: crate::serve::kv_persist::families::tq_packed::TqBitsPerCoord,
        k_payload: &[u8],
        v_payload: &[u8],
    ) -> Result<(), crate::serve::multi_model::SpillErrorKind> {
        use crate::serve::kv_persist::families::tq_packed;
        use crate::serve::multi_model::SpillErrorKind;

        // **B-tq.7** — branch on bits like the snapshot path does.
        // Restore writes back to `leg_hb_encoded[layer_rank]` at
        // bits >= 5; otherwise to `kv_caches[layer_rank]`.
        let use_hb = bits_per_coord.0 >= 5;

        // Borrow the layer's K/V buffers from the appropriate field.
        // Branch separately for K then V to keep borrow lifetimes
        // tight (each `as_mut_slice` borrows the buffer).
        let (capacity, n_kv_heads, head_dim) = if use_hb {
            let hb = self
                .leg_hb_encoded
                .as_ref()
                .ok_or(SpillErrorKind::CodecErr)?;
            let lay = hb.get(layer_rank).ok_or(SpillErrorKind::CodecErr)?;
            let cap = lay.capacity as u32;
            let nkv = lay.k_packed.shape().first().copied().unwrap_or(0) as u32;
            let hd_packed = lay.k_packed.shape().get(2).copied().unwrap_or(0);
            let head_dim_bits = (hd_packed as u64) * 8;
            if head_dim_bits % (bits_per_coord.0 as u64) != 0 {
                return Err(SpillErrorKind::CodecErr);
            }
            let hd = (head_dim_bits / (bits_per_coord.0 as u64)) as u32;
            (cap, nkv, hd)
        } else {
            let cache = self
                .kv_caches
                .get(layer_rank)
                .ok_or(SpillErrorKind::CodecErr)?;
            let cap = cache.capacity as u32;
            let nkv = cache.k_packed.shape().first().copied().unwrap_or(0) as u32;
            let hd_packed = cache.k_packed.shape().get(2).copied().unwrap_or(0);
            let head_dim_bits = (hd_packed as u64) * 8;
            if head_dim_bits % (bits_per_coord.0 as u64) != 0 {
                return Err(SpillErrorKind::CodecErr);
            }
            let hd = (head_dim_bits / (bits_per_coord.0 as u64)) as u32;
            (cap, nkv, hd)
        };

        // Two passes (K, V) × two operations (packed indices, F32 norms),
        // each requiring a separate `&mut` borrow.  Inner closure
        // `with_layer` factors out the source-of-truth selection.

        macro_rules! borrow_k_packed {
            () => {{
                if use_hb {
                    self.leg_hb_encoded
                        .as_mut()
                        .ok_or(SpillErrorKind::CodecErr)?[layer_rank]
                        .k_packed
                        .as_mut_slice::<u8>()
                        .map_err(|_| SpillErrorKind::IoErr)?
                } else {
                    self.kv_caches[layer_rank]
                        .k_packed
                        .as_mut_slice::<u8>()
                        .map_err(|_| SpillErrorKind::IoErr)?
                }
            }};
        }
        macro_rules! borrow_k_norms {
            () => {{
                if use_hb {
                    self.leg_hb_encoded
                        .as_mut()
                        .ok_or(SpillErrorKind::CodecErr)?[layer_rank]
                        .k_norms
                        .as_mut_slice::<f32>()
                        .map_err(|_| SpillErrorKind::IoErr)?
                } else {
                    self.kv_caches[layer_rank]
                        .k_norms
                        .as_mut_slice::<f32>()
                        .map_err(|_| SpillErrorKind::IoErr)?
                }
            }};
        }
        macro_rules! borrow_v_packed {
            () => {{
                if use_hb {
                    self.leg_hb_encoded
                        .as_mut()
                        .ok_or(SpillErrorKind::CodecErr)?[layer_rank]
                        .v_packed
                        .as_mut_slice::<u8>()
                        .map_err(|_| SpillErrorKind::IoErr)?
                } else {
                    self.kv_caches[layer_rank]
                        .v_packed
                        .as_mut_slice::<u8>()
                        .map_err(|_| SpillErrorKind::IoErr)?
                }
            }};
        }
        macro_rules! borrow_v_norms {
            () => {{
                if use_hb {
                    self.leg_hb_encoded
                        .as_mut()
                        .ok_or(SpillErrorKind::CodecErr)?[layer_rank]
                        .v_norms
                        .as_mut_slice::<f32>()
                        .map_err(|_| SpillErrorKind::IoErr)?
                } else {
                    self.kv_caches[layer_rank]
                        .v_norms
                        .as_mut_slice::<f32>()
                        .map_err(|_| SpillErrorKind::IoErr)?
                }
            }};
        }

        // --- K ---
        {
            let k_packed_mut: &mut [u8] = borrow_k_packed!();
            let _ = restore_packed_only(
                k_packed_mut,
                capacity,
                n_kv_heads,
                head_dim,
                bits_per_coord,
                range.clone(),
                k_payload,
            )?;
        }
        {
            let k_norms_f32: &mut [f32] = borrow_k_norms!();
            let _ = restore_norms_only_f32(
                k_norms_f32,
                capacity,
                n_kv_heads,
                head_dim,
                range.clone(),
                k_payload,
            )?;
        }

        // --- V ---
        {
            let v_packed_mut: &mut [u8] = borrow_v_packed!();
            let _ = restore_packed_only(
                v_packed_mut,
                capacity,
                n_kv_heads,
                head_dim,
                bits_per_coord,
                range.clone(),
                v_payload,
            )?;
        }
        {
            let v_norms_f32: &mut [f32] = borrow_v_norms!();
            let _ = restore_norms_only_f32(
                v_norms_f32,
                capacity,
                n_kv_heads,
                head_dim,
                range.clone(),
                v_payload,
            )?;
        }

        // Helper closures defined here as fn items to avoid double-borrow.
        // (Defined as fn so they don't capture the surrounding scope.)
        fn restore_packed_only(
            packed_bytes_mut: &mut [u8],
            capacity: u32,
            n_kv_heads: u32,
            head_dim: u32,
            bits_per_coord: tq_packed::TqBitsPerCoord,
            range: std::ops::Range<u32>,
            payload: &[u8],
        ) -> Result<(), SpillErrorKind> {
            let (header, idx, _norms) =
                tq_packed::unpack_tq_v2_payload(payload).map_err(|_| SpillErrorKind::CodecErr)?;
            if header.bits_per_coord != bits_per_coord
                || header.head_dim != head_dim
                || header.n_kv_heads != n_kv_heads
                || header.n_tokens != (range.end - range.start)
            {
                return Err(SpillErrorKind::CodecErr);
            }
            let bits = bits_per_coord.0 as u64;
            if (head_dim as u64) * bits % 8 != 0 {
                return Err(SpillErrorKind::CodecErr);
            }
            let hd_packed = ((head_dim as u64) * bits / 8) as usize;
            let nkv_us = n_kv_heads as usize;
            let cap_us = capacity as usize;
            let n_tokens = (range.end - range.start) as usize;
            // **B-tq.7**: bounds-check write target.  Global layers
            // in Gemma 4 use dynamic capacity sizing — at server-B
            // post_admit time, the layer's buffer may be too small to
            // hold the snapshot's range (e.g. global layer cap=2 at
            // warmup vs snapshot range 0..256).  Writing OOB would
            // panic the worker thread.  Return CodecErr instead so
            // the spiller bails on this layer cleanly; the
            // prompt_cache replay path (R-P5) still gives the warm
            // benefit since it short-circuits prefill before the
            // cache needs to grow.
            let expected_buf_len = nkv_us
                .checked_mul(cap_us)
                .and_then(|v| v.checked_mul(hd_packed))
                .ok_or(SpillErrorKind::CodecErr)?;
            if packed_bytes_mut.len() != expected_buf_len {
                return Err(SpillErrorKind::CodecErr);
            }
            if (range.end as usize) > cap_us {
                return Err(SpillErrorKind::CodecErr);
            }
            for h in 0..nkv_us {
                let head_base = h * cap_us * hd_packed;
                let row_start = head_base + (range.start as usize) * hd_packed;
                let row_end = head_base + (range.end as usize) * hd_packed;
                let src_off = h * n_tokens * hd_packed;
                let src_end = src_off + n_tokens * hd_packed;
                packed_bytes_mut[row_start..row_end].copy_from_slice(&idx[src_off..src_end]);
            }
            Ok(())
        }
        fn restore_norms_only_f32(
            norms_f32_mut: &mut [f32],
            capacity: u32,
            n_kv_heads: u32,
            head_dim: u32,
            range: std::ops::Range<u32>,
            payload: &[u8],
        ) -> Result<(), SpillErrorKind> {
            let (header, _idx, norms) =
                tq_packed::unpack_tq_v2_payload(payload).map_err(|_| SpillErrorKind::CodecErr)?;
            if header.n_kv_heads != n_kv_heads
                || header.head_dim != head_dim
                || header.n_tokens != (range.end - range.start)
            {
                return Err(SpillErrorKind::CodecErr);
            }
            // **B-tq.7**: norms_per_pos derived from head_dim.  D=256
            // sliding layers → 1 norm/pos; D=512 global layers → 2.
            let norms_per_pos = ((head_dim as usize) / 256).max(1);
            let nkv_us = n_kv_heads as usize;
            let cap_us = capacity as usize;
            let n_tokens = (range.end - range.start) as usize;
            // **B-tq.7**: bounds-check write target (matches
            // restore_packed_only).
            let expected_norms_len = nkv_us
                .checked_mul(cap_us)
                .and_then(|v| v.checked_mul(norms_per_pos))
                .ok_or(SpillErrorKind::CodecErr)?;
            if norms_f32_mut.len() != expected_norms_len {
                return Err(SpillErrorKind::CodecErr);
            }
            if (range.end as usize) > cap_us {
                return Err(SpillErrorKind::CodecErr);
            }
            // norms F32 LE -> per-element decode into typed slice.
            // Layout: dst is `[nkv, capacity, norms_per_pos]` flat F32;
            // src is `[nkv, n_tokens, norms_per_pos]` packed F32 LE.
            for h in 0..nkv_us {
                let head_base = h * cap_us * norms_per_pos;
                for t in 0..n_tokens {
                    for k in 0..norms_per_pos {
                        let dst_idx = head_base + (range.start as usize + t) * norms_per_pos + k;
                        let src_off =
                            ((h * n_tokens + t) * norms_per_pos + k) * 4;
                        let bytes = [
                            norms[src_off],
                            norms[src_off + 1],
                            norms[src_off + 2],
                            norms[src_off + 3],
                        ];
                        norms_f32_mut[dst_idx] = f32::from_le_bytes(bytes);
                    }
                }
            }
            Ok(())
        }

        Ok(())
    }

}

/// Cosine similarity dot/(||a||·||b||) for two equal-length F32 vectors.
///
/// Returns NaN if either norm is zero; otherwise a value in [-1, 1].
/// The dot and per-vector norms are accumulated in F64 for numerical
/// stability against the long SDPA-output vectors Gate H feeds it
/// (the iter-23/24 dump pattern is `[num_heads * head_dim] = 8192 +`
/// elements per pair, where naive F32 accumulation can lose ~1 ULP per
/// multiplied pair → measurable cosine drift for vectors near 1.0).
/// The final ratio is cast back to F32 for downstream comparisons.
///
/// # Purpose
///
/// Pure-Rust port of `src/bin/iter24_audit.rs:752-789`'s numpy-cosine
/// kernel.  Clears blocker #2 in W12's iter-108a scope: the audit
/// binaries shell out to a `cosine_sim.py` script for the SDPA-output
/// cosine — that violates `feedback_hf2q_sovereignty.md` ("no Python
/// at runtime") and blocks Gate H from running release-check-side.
///
/// # ADR-007 Gate H lineage
///
/// Used as the inner kernel for the Gate H cosine-similarity check
/// (see ADR-007 close §853-866: "cosine similarity at SDPA outputs",
/// p1-percentile gate ≥ 0.998, mean ≥ 0.9998).  iter-108b will wire
/// it into release-check.sh's two-regime byte-validation harness;
/// for now the audit binaries can drop their Python detour by
/// calling this function directly.
///
/// # Errors
///
/// Debug-asserts on length mismatch; release builds silently use
/// the shorter length.  Inputs of unequal length are an upstream
/// contract violation — the caller is expected to be passing two
/// dumps of the same SDPA-output shape.
///
/// (Wired by iter-108b's release-check.sh Gate 5 harness; the audit
/// binaries iter23/24_audit.rs will switch from `python3 cosine_sim.py`
/// to a direct call into this function in iter-108b.  No in-tree
/// caller as of iter-108a — the surface is designed for the iter-108b
/// release-check entry point + the unit tests below.)
#[allow(dead_code)]
pub fn cosine_pairwise_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine vectors must match length");
    let n = a.len().min(b.len());
    let mut dot: f64 = 0.0;
    let mut na2: f64 = 0.0;
    let mut nb2: f64 = 0.0;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na2 += x * x;
        nb2 += y * y;
    }
    let na = na2.sqrt();
    let nb = nb2.sqrt();
    if na == 0.0 || nb == 0.0 {
        f32::NAN
    } else {
        (dot / (na * nb)) as f32
    }
}

#[cfg(test)]
mod cosine_tests {
    use super::cosine_pairwise_f32;

    #[test]
    fn identity_is_one() {
        let a = vec![1.0_f32, 2.0, 3.0, -4.5, 0.25];
        let s = cosine_pairwise_f32(&a, &a);
        assert!((s - 1.0).abs() < 1e-6, "identity cosine = {s}");
    }

    #[test]
    fn antiparallel_is_negative_one() {
        let a: Vec<f32> = (0..128).map(|i| (i as f32) - 64.0).collect();
        let neg: Vec<f32> = a.iter().map(|x| -x).collect();
        let s = cosine_pairwise_f32(&a, &neg);
        assert!((s + 1.0).abs() < 1e-6, "antiparallel cosine = {s}");
    }

    #[test]
    fn zero_norm_is_nan() {
        let a = vec![0.0_f32; 32];
        let b: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let s = cosine_pairwise_f32(&a, &b);
        assert!(s.is_nan(), "zero-norm cosine should be NaN, got {s}");
        // And symmetric: nonzero on the left, zero on the right.
        let s2 = cosine_pairwise_f32(&b, &a);
        assert!(s2.is_nan(), "zero-norm cosine (rhs) should be NaN, got {s2}");
        // Both zero → NaN.
        let z = vec![0.0_f32; 32];
        let s3 = cosine_pairwise_f32(&z, &z);
        assert!(s3.is_nan(), "both-zero cosine should be NaN, got {s3}");
    }

    #[test]
    fn orthogonal_is_zero() {
        // [1,0,0,...] vs [0,1,0,...] → dot=0, both norms=1 → cosine=0.
        let mut a = vec![0.0_f32; 16];
        let mut b = vec![0.0_f32; 16];
        a[0] = 1.0;
        b[1] = 1.0;
        let s = cosine_pairwise_f32(&a, &b);
        assert!(s.abs() < 1e-6, "orthogonal cosine = {s}");
    }

    #[test]
    fn matches_python_reference_within_tolerance() {
        // Reference shape: same kernel as iter24_audit.rs:752-761 in F64.
        // We compare against the explicit numpy-style formula to make sure
        // our F64 accumulation tracks the audit-binary numbers.
        let a: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.013).sin()).collect();
        let b: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.013).sin() + 1e-3).collect();
        let s = cosine_pairwise_f32(&a, &b);
        let py_dot: f64 = a.iter().zip(&b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
        let py_na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let py_nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let py = (py_dot / (py_na * py_nb)) as f32;
        assert!((s - py).abs() < 1e-6, "rust cosine={s} vs python-equiv={py}");
    }
}

/// Wedge-4 / iter-227 — `MlxMoeWeights::dense_placeholder` invariants.
///
/// These tests pin the placeholder constructor's contract at the Rust
/// type-system level so a future refactor cannot silently re-introduce
/// the iter-227 dispatch crash. They do NOT exercise GPU kernels — the
/// constructor is pure CPU + tiny MlxBuffer allocations.
///
/// Live-load coverage of the conditional MoE-expert load itself
/// (skipping `blk.0.ffn_gate_up_exps.weight` when the dense GGUF lacks
/// it) is covered by the `iter227_*` arch-dispatch tests in
/// `serve::tests` plus the operator-gated regression test referenced by
/// `scripts/wedge4_qwen3vl.sh` (the real Qwen3-VL-2B GGUF ships with
/// dense FFN tensors and surfaces the iter-227 actionable error from
/// `LoadedModel::load`, not from the per-layer MoE expert loader).
#[cfg(test)]
mod dense_placeholder_tests {
    use super::*;

    /// The dense placeholder bundle MUST report `stacked_gate_up: None`
    /// AND `stacked_down: None` so the fused-id MoE dispatch's
    /// `is_some() && is_some()` gate falsifies cleanly. If a future
    /// refactor accidentally allocates Some(empty_buffer) here, the
    /// MoE dispatch would walk into 1-element buffers and produce
    /// garbage logits without panicking — far worse than today's
    /// "missing tensor" load-time bail.
    #[test]
    fn iter227_dense_placeholder_has_no_stacked_expert_buffers() {
        let device = match mlx_native::MlxDevice::new() {
            Ok(d) => d,
            Err(_) => {
                // No Metal device available (e.g. CI without GPU);
                // skip — the live load path on M5 Max exercises this.
                eprintln!("skipping iter227_dense_placeholder_has_no_stacked_expert_buffers: no MlxDevice");
                return;
            }
        };
        let moe = MlxMoeWeights::dense_placeholder(&device)
            .expect("dense_placeholder allocation must succeed on Metal device");
        assert!(
            moe.stacked_gate_up.is_none(),
            "dense placeholder MUST have stacked_gate_up = None to falsify fused-id MoE gate"
        );
        assert!(
            moe.stacked_down.is_none(),
            "dense placeholder MUST have stacked_down = None to falsify fused-id MoE gate"
        );
    }

    /// Sentinel scalars (`top_k = 0`, `moe_intermediate_size = 0`,
    /// strides = 0) make any accidental read of the placeholder fields
    /// visibly wrong instead of producing plausible-looking garbage.
    #[test]
    fn iter227_dense_placeholder_zeros_scalar_metadata() {
        let device = match mlx_native::MlxDevice::new() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping iter227_dense_placeholder_zeros_scalar_metadata: no MlxDevice");
                return;
            }
        };
        let moe = MlxMoeWeights::dense_placeholder(&device)
            .expect("dense_placeholder allocation must succeed on Metal device");
        assert_eq!(moe.top_k, 0, "dense placeholder must zero top_k");
        assert_eq!(
            moe.moe_intermediate_size, 0,
            "dense placeholder must zero moe_intermediate_size"
        );
        assert_eq!(moe.gate_up_expert_stride, 0);
        assert_eq!(moe.down_expert_stride, 0);
    }

    /// Allocation cost regression guard: the placeholder bundle must
    /// stay tiny so a 28-layer Qwen3-VL-2B dense load adds <1 KB total
    /// MoE-bookkeeping overhead vs. the previous unconditional path
    /// (which would have OOM-allocated GBs of expert tensors that
    /// don't exist on disk).
    #[test]
    fn iter227_dense_placeholder_buffers_are_one_element_each() {
        let device = match mlx_native::MlxDevice::new() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping iter227_dense_placeholder_buffers_are_one_element_each: no MlxDevice");
                return;
            }
        };
        let moe = MlxMoeWeights::dense_placeholder(&device)
            .expect("dense_placeholder allocation must succeed on Metal device");
        // Each placeholder buffer is 1 F32 element = 4 bytes.
        assert_eq!(
            moe.per_expert_scale.byte_len(),
            std::mem::size_of::<f32>(),
            "per_expert_scale placeholder must be 1 F32 element"
        );
        assert_eq!(
            moe.router_combined_weight.byte_len(),
            std::mem::size_of::<f32>(),
            "router_combined_weight placeholder must be 1 F32 element"
        );
        assert_eq!(
            moe.router_proj.buffer.byte_len(),
            std::mem::size_of::<f32>(),
            "router_proj placeholder buffer must be 1 F32 element"
        );
    }
}

/// ADR-022 P1.9 iter-15 — `dispatch_qmatmul` F32 routing test.
///
/// Hypothesis: an `MlxQWeight` with `ggml_dtype = F32` (router weight from
/// APEX-format Gemma4 GGUF) routes through `dense_matmul_f32_f32_tensor`
/// and produces `output[m, n] = sum_k input[m, k] * weight[n, k]`
/// matching a CPU reference.
///
/// Falsifier: pre-iter-15 hf2q dispatched F32 weights through
/// `quantized_matmul_ggml`, which (correctly) returned an error because
/// the GGML block kernels require block-format input. With this routing
/// fix the same call must succeed and produce the expected matmul.
#[cfg(test)]
mod dispatch_qmatmul_f32_router_test {
    use super::*;

    #[test]
    fn f32_router_weight_routes_to_dense_matmul() {
        let device = match mlx_native::MlxDevice::new() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping f32_router_weight_routes_to_dense_matmul: no MlxDevice");
                return;
            }
        };
        let mut registry = mlx_native::KernelRegistry::new();

        // [n=4 output, k=64 inner] — k≥32 required by dense_mm_f32_f32 kernel.
        let n: usize = 4;
        let k: usize = 64;
        let m: usize = 2;

        // Deterministic pseudo-random fixtures.
        let mut state: u64 = 0xDEAD_BEEF_F00D_F00D;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let weight: Vec<f32> = (0..(n * k)).map(|_| next()).collect();
        let input: Vec<f32> = (0..(m * k)).map(|_| next()).collect();

        // CPU reference: out[m, n] = sum_k input[m, k] * weight[n, k].
        let mut expected = vec![0.0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0.0f64;
                for ki in 0..k {
                    acc += (input[mi * k + ki] as f64) * (weight[ni * k + ki] as f64);
                }
                expected[mi * n + ni] = acc as f32;
            }
        }

        // GPU buffers.
        let f32_sz = std::mem::size_of::<f32>();
        let mut weight_buf = device
            .alloc_buffer(n * k * f32_sz, mlx_native::DType::F32, vec![n, k])
            .expect("alloc weight");
        weight_buf
            .as_mut_slice::<f32>()
            .expect("weight write")
            .copy_from_slice(&weight);

        let mut input_buf = device
            .alloc_buffer(m * k * f32_sz, mlx_native::DType::F32, vec![m, k])
            .expect("alloc input");
        input_buf
            .as_mut_slice::<f32>()
            .expect("input write")
            .copy_from_slice(&input);

        let mut output_buf = device
            .alloc_buffer(m * n * f32_sz, mlx_native::DType::F32, vec![m, n])
            .expect("alloc output");

        let qweight = MlxQWeight {
            buffer: weight_buf,
            info: super::super::gpu::QuantWeightInfo {
                ggml_dtype: mlx_native::GgmlType::F32,
                rows: n,
                cols: k,
            },
            affine: None,
            f16_shadow: None,
        };

        // Run through GraphSession (mirrors production dispatch path).
        let executor = mlx_native::GraphExecutor::new(device.clone());
        let mut session = executor.begin().expect("begin session");
        dispatch_qmatmul(
            &mut session,
            &mut registry,
            &device,
            &input_buf,
            &qweight,
            &mut output_buf,
            m as u32,
        )
        .expect("dispatch_qmatmul F32 path");
        session.finish().expect("session finish");

        // Validate output.
        let got: &[f32] = output_buf.as_slice().expect("read output");
        let mut max_abs_diff = 0.0f32;
        for i in 0..(m * n) {
            let d = (got[i] - expected[i]).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        assert!(
            max_abs_diff < 1e-4,
            "F32 dispatch_qmatmul mismatch: max|diff|={max_abs_diff}, got={:?}, expected={:?}",
            got,
            expected
        );
    }
}

/// ADR-020 AC#5 Iter B — `MlxQWeight::from_mlx_affine_linear` round-trip
/// + GPU dispatch parity test.
///
/// Hypothesis: an MlxAffineLinear constructed in-process (skipping the
/// safetensors disk round-trip), uploaded via `from_mlx_affine_linear`,
/// and dispatched via the new `qmm_affine_t_packed_simd4_b4` kernel
/// produces output matching a CPU oracle that does
/// `y = x @ (q_int * scales + biases)^T`.
///
/// Falsifier: any divergence in the packed-U32 emission inside
/// `from_mlx_affine_linear` (e.g. wrong slot ordering, wrong nibble
/// position) would surface as a measurable error vs the CPU oracle.
#[cfg(test)]
mod ac5_iter_b_affine_qweight_roundtrip {
    use super::*;
    use crate::calibrate::mlx_safetensors_loader::MlxAffineLinear;
    use mlx_native::ops::qmm_affine::dispatch_qmm_affine_t_packed_simd4_b4;

    #[test]
    fn from_mlx_affine_linear_roundtrips_through_packed_kernel() {
        let device = match mlx_native::MlxDevice::new() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping ac5_iter_b: no MlxDevice");
                return;
            }
        };
        let mut registry = mlx_native::KernelRegistry::new();

        let m = 16usize;
        let n = 64usize;
        let k = 96usize;
        let group_size = 32usize;
        let bits = 4u32;
        let pack_factor = (32 / bits) as usize;
        let groups_per_row = k / group_size;

        // Synthetic deterministic linear: q_int in [0, 16), scales/biases F32.
        let q_int: Vec<u8> = (0..(n * k)).map(|i| ((i * 11 + 5) % 16) as u8).collect();
        let scales: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| 0.05 + (i as f32) * 0.0017)
            .collect();
        let biases: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| -0.13 + (i as f32) * 0.0023)
            .collect();
        let linear = MlxAffineLinear {
            n,
            k,
            group_size,
            bits,
            q_int: q_int.clone(),
            scales: scales.clone(),
            biases: biases.clone(),
        };

        // Hf2q: pack into MlxQWeight via the AC#5 Iter B constructor.
        let qweight =
            MlxQWeight::from_mlx_affine_linear(&device, &linear).expect("from_mlx_affine_linear");
        assert_eq!(qweight.info.rows, n);
        assert_eq!(qweight.info.cols, k);
        let extra = qweight.affine.as_ref().expect("affine extra");
        assert_eq!(extra.bits, bits);
        assert_eq!(extra.group_size, group_size as u32);
        assert_eq!(qweight.buffer.element_count(), n * (k / pack_factor));
        assert_eq!(extra.scales.element_count(), n * groups_per_row);
        assert_eq!(extra.biases.element_count(), n * groups_per_row);

        // Upload x.
        let x: Vec<f32> = (0..(m * k))
            .map(|i| ((i as f32) * 0.013 - 0.4).sin() * 0.6)
            .collect();
        let mut x_buf = device
            .alloc_buffer(m * k * 4, mlx_native::DType::F32, vec![m, k])
            .expect("x");
        x_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&x);

        let y_buf = device
            .alloc_buffer(m * n * 4, mlx_native::DType::F32, vec![m, n])
            .expect("y");

        // meta: [M, N, K, group_size]
        let mut meta = device
            .alloc_buffer(16, mlx_native::DType::U32, vec![4])
            .unwrap();
        meta.as_mut_slice::<u32>().unwrap().copy_from_slice(&[
            m as u32,
            n as u32,
            k as u32,
            group_size as u32,
        ]);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_qmm_affine_t_packed_simd4_b4(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &x_buf,
            &qweight.buffer,
            &extra.scales,
            &extra.biases,
            &y_buf,
            &meta,
            m as u32,
            n as u32,
            k as u32,
            group_size as u32,
            bits,
        )
        .expect("dispatch packed simd4");
        encoder.commit_and_wait().unwrap();

        // CPU oracle: y[r, col] = sum_k x[r, k] * (q_int[col, k] * scales[col, g] + biases[col, g]).
        let mut expected = vec![0.0f32; m * n];
        for r in 0..m {
            for col in 0..n {
                let mut acc = 0.0f64;
                for g in 0..groups_per_row {
                    let s = scales[col * groups_per_row + g] as f64;
                    let b = biases[col * groups_per_row + g] as f64;
                    for i in 0..group_size {
                        let kk = g * group_size + i;
                        let q = q_int[col * k + kk] as f64;
                        acc += (x[r * k + kk] as f64) * (q * s + b);
                    }
                }
                expected[r * n + col] = acc as f32;
            }
        }

        let got = y_buf.as_slice::<f32>().unwrap();
        let mut max_abs = 0.0f32;
        for i in 0..(m * n) {
            let d = (got[i] - expected[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        assert!(
            max_abs < 1e-3,
            "max|y - oracle| = {max_abs} (m={m}, n={n}, k={k})"
        );
    }

    /// AC#5 Iter C — `dispatch_qmatmul` routes MlxQWeight with
    /// `affine.is_some()` through the new packed kernel.  This is the
    /// production entry point; correctness here = AC #5 dense closure
    /// at the dispatch boundary.
    #[test]
    fn dispatch_qmatmul_routes_affine_weight_to_packed_kernel() {
        let device = match mlx_native::MlxDevice::new() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping: no MlxDevice");
                return;
            }
        };
        let mut registry = mlx_native::KernelRegistry::new();

        let m = 8usize;
        let n = 32usize;
        let k = 64usize;
        let gs = 32usize;
        let bits = 4u32;
        let groups_per_row = k / gs;

        let q_int: Vec<u8> = (0..(n * k)).map(|i| ((i * 7 + 3) % 16) as u8).collect();
        let scales: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| 0.07 + (i as f32) * 0.0011)
            .collect();
        let biases: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| -0.09 + (i as f32) * 0.0027)
            .collect();
        let linear = MlxAffineLinear {
            n,
            k,
            group_size: gs,
            bits,
            q_int: q_int.clone(),
            scales: scales.clone(),
            biases: biases.clone(),
        };
        let qweight =
            MlxQWeight::from_mlx_affine_linear(&device, &linear).expect("from_mlx_affine_linear");

        let x: Vec<f32> = (0..(m * k))
            .map(|i| ((i as f32) * 0.011 - 0.3).cos() * 0.5)
            .collect();
        let mut x_buf = device
            .alloc_buffer(m * k * 4, mlx_native::DType::F32, vec![m, k])
            .expect("x");
        x_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&x);

        let mut y_buf = device
            .alloc_buffer(m * n * 4, mlx_native::DType::F32, vec![m, n])
            .expect("y");

        // Drive through GraphSession exactly like production.
        let executor = mlx_native::GraphExecutor::new(device.clone());
        let mut session = executor.begin().expect("begin session");
        dispatch_qmatmul(
            &mut session,
            &mut registry,
            &device,
            &x_buf,
            &qweight,
            &mut y_buf,
            m as u32,
        )
        .expect("dispatch_qmatmul affine route");
        session.finish().expect("finish");

        // CPU oracle — same formula as Iter B test.
        let mut expected = vec![0.0f32; m * n];
        for r in 0..m {
            for col in 0..n {
                let mut acc = 0.0f64;
                for g in 0..groups_per_row {
                    let s = scales[col * groups_per_row + g] as f64;
                    let b = biases[col * groups_per_row + g] as f64;
                    for i in 0..gs {
                        let kk = g * gs + i;
                        let q = q_int[col * k + kk] as f64;
                        acc += (x[r * k + kk] as f64) * (q * s + b);
                    }
                }
                expected[r * n + col] = acc as f32;
            }
        }

        let got = y_buf.as_slice::<f32>().unwrap();
        let mut max_abs = 0.0f32;
        for i in 0..(m * n) {
            let d = (got[i] - expected[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        assert!(
            max_abs < 1e-3,
            "dispatch_qmatmul affine route: max|y - oracle| = {max_abs}"
        );
    }

    /// AC#5 Iter C — affine route is byte-identical to direct kernel
    /// dispatch.  Confirms `dispatch_qmatmul` doesn't introduce any
    /// per-call drift (e.g. wrong meta values, batch-dim confusion).
    #[test]
    fn dispatch_qmatmul_affine_equals_direct_kernel() {
        let device = match mlx_native::MlxDevice::new() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping: no MlxDevice");
                return;
            }
        };
        let mut registry = mlx_native::KernelRegistry::new();

        let m = 4usize;
        let n = 32usize;
        let k = 32usize;
        let gs = 32usize;
        let bits = 4u32;
        let groups_per_row = k / gs;

        let q_int: Vec<u8> = (0..(n * k)).map(|i| ((i * 5 + 1) % 16) as u8).collect();
        let scales: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| 0.05 + (i as f32) * 0.001)
            .collect();
        let biases: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| -0.1 + (i as f32) * 0.002)
            .collect();
        let linear = MlxAffineLinear {
            n,
            k,
            group_size: gs,
            bits,
            q_int,
            scales,
            biases,
        };
        let qweight =
            MlxQWeight::from_mlx_affine_linear(&device, &linear).expect("from_mlx_affine_linear");
        let extra = qweight.affine.as_ref().unwrap();

        let x: Vec<f32> = (0..(m * k))
            .map(|i| ((i as f32) * 0.017 + 0.2).sin() * 0.4)
            .collect();
        let mut x_buf = device
            .alloc_buffer(m * k * 4, mlx_native::DType::F32, vec![m, k])
            .expect("x");
        x_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&x);

        let mut y_via_dispatch = device
            .alloc_buffer(m * n * 4, mlx_native::DType::F32, vec![m, n])
            .expect("y_d");
        let y_direct = device
            .alloc_buffer(m * n * 4, mlx_native::DType::F32, vec![m, n])
            .expect("y_k");
        let mut meta = device
            .alloc_buffer(16, mlx_native::DType::U32, vec![4])
            .unwrap();
        meta.as_mut_slice::<u32>().unwrap().copy_from_slice(&[
            m as u32, n as u32, k as u32, gs as u32,
        ]);

        // Direct kernel call.
        let mut encoder = device.command_encoder().unwrap();
        mlx_native::ops::qmm_affine::dispatch_qmm_affine_t_packed_simd4_b4(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &x_buf,
            &qweight.buffer,
            &extra.scales,
            &extra.biases,
            &y_direct,
            &meta,
            m as u32, n as u32, k as u32, gs as u32, bits,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();

        // dispatch_qmatmul path.
        let executor = mlx_native::GraphExecutor::new(device.clone());
        let mut session = executor.begin().expect("begin session");
        dispatch_qmatmul(
            &mut session,
            &mut registry,
            &device,
            &x_buf,
            &qweight,
            &mut y_via_dispatch,
            m as u32,
        )
        .expect("dispatch_qmatmul");
        session.finish().expect("finish");

        let direct = y_direct.as_slice::<f32>().unwrap();
        let dispatch = y_via_dispatch.as_slice::<f32>().unwrap();
        for i in 0..(m * n) {
            assert_eq!(
                dispatch[i].to_bits(),
                direct[i].to_bits(),
                "y[{i}] (m={m} n={n}): dispatch={} direct={}",
                dispatch[i],
                direct[i],
            );
        }
    }

    /// AC#5 Iter C2.2 — `parse_dwq_moe_expert_role` covers all 4 MoE
    /// base roles + handles invalid suffixes correctly.
    #[test]
    fn parse_dwq_moe_expert_role_covers_all_bases() {
        use super::{parse_dwq_moe_expert_role, MoeBaseRole};
        // Fused gate+up case (qwen3.5 GGUF).
        assert_eq!(
            parse_dwq_moe_expert_role("ffn_gate_up.0"),
            Some((MoeBaseRole::GateUp, 0))
        );
        assert_eq!(
            parse_dwq_moe_expert_role("ffn_gate_up.127"),
            Some((MoeBaseRole::GateUp, 127))
        );
        // Separate gate / up (uncommon; future archs).
        assert_eq!(
            parse_dwq_moe_expert_role("ffn_gate.5"),
            Some((MoeBaseRole::Gate, 5))
        );
        assert_eq!(
            parse_dwq_moe_expert_role("ffn_up.7"),
            Some((MoeBaseRole::Up, 7))
        );
        // Down expert.
        assert_eq!(
            parse_dwq_moe_expert_role("ffn_down.42"),
            Some((MoeBaseRole::Down, 42))
        );
        // Critical: `ffn_gate_up.X` must not match the `ffn_gate.` prefix.
        assert_ne!(
            parse_dwq_moe_expert_role("ffn_gate_up.3"),
            Some((MoeBaseRole::Gate, 3))
        );
        // Invalid suffixes.
        assert_eq!(parse_dwq_moe_expert_role("ffn_gate_up.abc"), None);
        assert_eq!(parse_dwq_moe_expert_role("ffn_gate"), None);
        assert_eq!(parse_dwq_moe_expert_role("attn_q.0"), None);
        assert_eq!(parse_dwq_moe_expert_role(""), None);
    }

    /// AC#5 Iter D — `parse_dwq_overlay_role` covers all production
    /// stems plus rejects unknowns.  Pure CPU; no MlxDevice required.
    #[test]
    fn parse_dwq_overlay_role_covers_all_dense_stems() {
        use super::{parse_dwq_overlay_role, DwqOverlayRole};
        // Dense Linears.
        assert_eq!(parse_dwq_overlay_role("attn_q"), DwqOverlayRole::AttnQ);
        assert_eq!(parse_dwq_overlay_role("attn_k"), DwqOverlayRole::AttnK);
        assert_eq!(parse_dwq_overlay_role("attn_v"), DwqOverlayRole::AttnV);
        assert_eq!(
            parse_dwq_overlay_role("attn_output"),
            DwqOverlayRole::AttnOutput
        );
        assert_eq!(parse_dwq_overlay_role("ffn_gate"), DwqOverlayRole::FfnGate);
        assert_eq!(parse_dwq_overlay_role("ffn_up"), DwqOverlayRole::FfnUp);
        assert_eq!(parse_dwq_overlay_role("ffn_down"), DwqOverlayRole::FfnDown);
        // MoE per-expert (Iter C2 territory).
        assert_eq!(
            parse_dwq_overlay_role("ffn_gate.0"),
            DwqOverlayRole::MoeExpert
        );
        assert_eq!(
            parse_dwq_overlay_role("ffn_up.255"),
            DwqOverlayRole::MoeExpert
        );
        assert_eq!(
            parse_dwq_overlay_role("ffn_down.42"),
            DwqOverlayRole::MoeExpert
        );
        // Unknown roles.
        assert_eq!(
            parse_dwq_overlay_role("token_embd"),
            DwqOverlayRole::Unknown
        );
        assert_eq!(parse_dwq_overlay_role(""), DwqOverlayRole::Unknown);
        assert_eq!(parse_dwq_overlay_role("output"), DwqOverlayRole::Unknown);
    }

    /// AC#5 Iter D — `parse_dwq_overlay_metadata` honors metadata when
    /// present, defaults sanely when absent, rejects mismatched format.
    #[test]
    fn parse_dwq_overlay_metadata_handles_all_cases() {
        use super::parse_dwq_overlay_metadata;
        use std::collections::HashMap;

        // Absent metadata → defaults (4, 32).
        let (bits, gs) = parse_dwq_overlay_metadata(None).unwrap();
        assert_eq!(bits, 4);
        assert_eq!(gs, 32);

        // Format mismatch → error.
        let mut bad_format = HashMap::new();
        bad_format.insert("format".to_string(), "wrong-format".to_string());
        assert!(parse_dwq_overlay_metadata(Some(&bad_format)).is_err());

        // Correct format + custom bits/gs.
        let mut meta = HashMap::new();
        meta.insert("format".to_string(), "mlx-affine-dwq-v1".to_string());
        meta.insert("bits".to_string(), "8".to_string());
        meta.insert("group_size".to_string(), "64".to_string());
        let (bits, gs) = parse_dwq_overlay_metadata(Some(&meta)).unwrap();
        assert_eq!(bits, 8);
        assert_eq!(gs, 64);

        // Format absent + valid bits/gs → values respected.
        let mut nofmt = HashMap::new();
        nofmt.insert("bits".to_string(), "4".to_string());
        nofmt.insert("group_size".to_string(), "32".to_string());
        let (bits, gs) = parse_dwq_overlay_metadata(Some(&nofmt)).unwrap();
        assert_eq!(bits, 4);
        assert_eq!(gs, 32);

        // Garbage bits → falls back to default 4.
        let mut garbage = HashMap::new();
        garbage.insert("format".to_string(), "mlx-affine-dwq-v1".to_string());
        garbage.insert("bits".to_string(), "not-a-number".to_string());
        let (bits, gs) = parse_dwq_overlay_metadata(Some(&garbage)).unwrap();
        assert_eq!(bits, 4);
        assert_eq!(gs, 32);
    }

    /// AC#5 Iter D — full DWQ-format safetensors round-trip:
    /// MlxAffineLinear → safetensors-on-disk-with-metadata → re-read +
    /// rebuild MlxAffineLinear → byte-identical contents.  Confirms the
    /// metadata embed (Iter D step 1) + the safetensors loader contract
    /// hold without MlxModelWeights involvement.
    #[test]
    fn dwq_safetensors_metadata_roundtrip() {
        use crate::calibrate::mlx_safetensors_loader::{MlxAffineLinear, MlxAffineLinearBytes};
        use safetensors::tensor::{serialize, Dtype};
        use std::collections::HashMap;

        let n = 32usize;
        let k = 64usize;
        let group_size = 32usize;
        let bits = 4u32;
        let groups_per_row = k / group_size;
        let q_int: Vec<u8> = (0..(n * k)).map(|i| ((i * 3 + 7) % 16) as u8).collect();
        let scales: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| 0.05 + (i as f32) * 0.001)
            .collect();
        let biases: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| -0.1 + (i as f32) * 0.002)
            .collect();

        let linear = MlxAffineLinear {
            n,
            k,
            group_size,
            bits,
            q_int: q_int.clone(),
            scales: scales.clone(),
            biases: biases.clone(),
        };
        let stem = "blk.0.attn_q";
        let bytes_owned: MlxAffineLinearBytes = linear.to_safetensors_bytes(Dtype::F32).unwrap();
        let (w, s, b) = bytes_owned.to_safetensors_views().unwrap();
        let pairs: Vec<(String, _)> = vec![
            (format!("{stem}.weight"), w),
            (format!("{stem}.scales"), s),
            (format!("{stem}.biases"), b),
        ];
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "mlx-affine-dwq-v1".to_string());
        metadata.insert("bits".to_string(), bits.to_string());
        metadata.insert("group_size".to_string(), group_size.to_string());

        let serialized = serialize(
            pairs.iter().map(|(k, v)| (k.as_str(), v)),
            Some(metadata),
        )
        .unwrap();

        // Verify metadata round-trips via read_metadata.
        let (_n, md) = safetensors::SafeTensors::read_metadata(&serialized).unwrap();
        let meta_map = md.metadata().as_ref().expect("metadata present");
        assert_eq!(meta_map.get("format").unwrap(), "mlx-affine-dwq-v1");
        assert_eq!(meta_map.get("bits").unwrap(), "4");
        assert_eq!(meta_map.get("group_size").unwrap(), "32");

        let (parsed_bits, parsed_gs) =
            super::parse_dwq_overlay_metadata(Some(meta_map)).unwrap();
        assert_eq!(parsed_bits, bits);
        assert_eq!(parsed_gs, group_size);

        // Rebuild the MlxAffineLinear via from_safetensors and compare.
        let st = safetensors::SafeTensors::deserialize(&serialized).unwrap();
        let stems: Vec<&str> = st
            .names()
            .iter()
            .filter_map(|n| n.strip_suffix(".weight"))
            .collect();
        assert_eq!(stems.len(), 1);
        assert_eq!(stems[0], stem);

        let rebuilt = MlxAffineLinear::from_safetensors(&st, stem, parsed_bits, parsed_gs).unwrap();
        assert_eq!(rebuilt.n, n);
        assert_eq!(rebuilt.k, k);
        assert_eq!(rebuilt.bits, bits);
        assert_eq!(rebuilt.group_size, group_size);
        assert_eq!(rebuilt.q_int, q_int);
        assert_eq!(rebuilt.scales, scales);
        assert_eq!(rebuilt.biases, biases);
    }

    #[test]
    fn from_mlx_affine_linear_rejects_unsupported_bits() {
        let device = match mlx_native::MlxDevice::new() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping: no MlxDevice");
                return;
            }
        };
        // bits=8 is not supported by the simd4_b4 kernel — constructor
        // must reject so we surface the gap at load time, not dispatch time.
        let linear = MlxAffineLinear {
            n: 32,
            k: 32,
            group_size: 32,
            bits: 8,
            q_int: vec![0u8; 32 * 32],
            scales: vec![0.1f32; 32],
            biases: vec![0.0f32; 32],
        };
        let res = MlxQWeight::from_mlx_affine_linear(&device, &linear);
        assert!(res.is_err(), "should reject bits=8");
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
        affine: None,
        f16_shadow: None,
    })
}

/// ADR-029 iter-28 H29 — populate the F16 pre-dequantized shadow for a
/// quantized weight.  Returns Ok(()) silently if:
///   * HF2Q_F16_SHADOW is unset / falsy (default), OR
///   * the weight is not a quantized type that dequant_to_f16 supports
///     (F32 / F16 / I16 / affine).
///
/// When ENABLED and the weight type is supported, dispatches the dequant
/// kernel at load time, allocates a 2-bytes/elem F16 buffer, and stores
/// it as the `f16_shadow` field.  Subsequent `dispatch_qmatmul` at m > 8
/// will fast-path through the F16-input matmul kernel (kernel_mul_mm_f16
/// _f32_tensor) which avoids per-call dequant overhead.
///
/// Memory cost: ~1 GB extra resident for gemma4-26B attn weights when
/// applied to attn_q/k/v/output + ffn_gate/up.  On the M5 Max target
/// (128 GB unified), this is well within budget; matches peer's strategy.
fn populate_f16_shadow_if_enabled(
    qweight: &mut MlxQWeight,
    device: &MlxDevice,
    registry: &mut mlx_native::KernelRegistry,
    tensor_name: &str,
) -> Result<()> {
    // iter-31 default-flip: H29 default-true; opt-out via =0/false/off.
    let enabled = std::env::var("HF2Q_F16_SHADOW")
        .ok()
        .map(|v| !matches!(v.as_str(), "0" | "false" | "off"))
        .unwrap_or(true);
    if !enabled {
        return Ok(());
    }

    // Skip when affine — that path doesn't go through dispatch_qmatmul's
    // quantized branch.
    if qweight.affine.is_some() {
        return Ok(());
    }

    // Skip types the dequant kernel doesn't handle.
    use mlx_native::GgmlType;
    match qweight.info.ggml_dtype {
        GgmlType::Q4_0
        | GgmlType::Q8_0
        | GgmlType::Q5_1
        | GgmlType::IQ4_NL
        | GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q6_K => {}
        _ => return Ok(()),
    }

    let n_rows = qweight.info.rows as u32;
    let n_cols = qweight.info.cols as u32;

    let f16 = mlx_native::ops::dequant_to_f16::materialize_f16_shadow(
        device,
        registry,
        &qweight.buffer,
        n_rows,
        n_cols,
        qweight.info.ggml_dtype,
    )
    .map_err(|e| anyhow::anyhow!("F16 shadow materialize for '{}': {e}", tensor_name))?;

    qweight.f16_shadow = Some(f16);
    Ok(())
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
///
/// ADR-028 iter-338 — env-gated V2 (float4 + simd_sum) variant when
/// `dim % 4 == 0`, mirroring mlx-native ops/rms_norm.rs:662-683 pattern
/// that the iter-310 V2 dispatcher uses.  This was a V2-bypass site:
/// the function was authored before the V2 dispatcher existed and
/// directly gets the v1 pipeline by name.  Default-ON via
/// `HF2Q_RMS_NORM_V2`; opt-out via `=0`/`false`/`off`.
pub fn dispatch_rms_norm_unit_perhead(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut mlx_native::KernelRegistry,
    device: &mlx_native::metal::DeviceRef,
    args: &RmsNormPerHeadArgs<'_>,
) -> Result<()> {
    // Inline V2 env-gate (mirrors iter-326 inline pattern; preserves
    // debug→serve module boundary).
    let v2_env_off = matches!(
        std::env::var("HF2Q_RMS_NORM_V2").ok().as_deref(),
        Some(v) if v.eq_ignore_ascii_case("0")
            || v.eq_ignore_ascii_case("false")
            || v.eq_ignore_ascii_case("off")
    );
    let use_v2 = (args.dim % 4 == 0) && !v2_env_off;
    let kernel_name = if use_v2 {
        "rms_norm_no_scale_f32_v2"
    } else {
        "rms_norm_no_scale_f32"
    };
    let pipeline = registry.get_pipeline(kernel_name, device)
        .map_err(|e| anyhow::anyhow!("{kernel_name} pipeline: {e}"))?;
    let mut tg_size = std::cmp::min(256, args.dim.next_power_of_two()) as u64;
    if use_v2 && tg_size < 32 {
        tg_size = 32;
    }
    let shared_mem_bytes = if use_v2 {
        // V2 only needs n_sg = tg_size/32 floats; allocate at least 32
        // so partial-warp tg_sizes are safe (matches rms_norm.rs:679-681).
        (tg_size / 32).max(1) * 4
    } else {
        tg_size * 4
    };
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

/// Run one weight matmul through the GraphSession, routing by GGUF dtype.
///
/// Dispatches `output = input @ weight.T`. The output buffer is pre-allocated
/// by the caller.  This is the single GGUF-format dispatcher: the
/// `weight.info.ggml_dtype` field selects the underlying kernel, NOT a
/// fallback chain. F32 → `dense_matmul_f32_f32_tensor` (peer of llama.cpp's
/// `kernel_mul_mm_f32_f32`); Q* / IQ* → `quantized_matmul_ggml`.
///
/// ADR-022 P1.9 — APEX-format Gemma4 GGUFs preserve `ffn_gate_inp.weight`
/// (router projection) as F32 for accuracy. mlx-native's
/// `quantized_matmul_ggml` correctly refuses F32 because the GGML block
/// kernels require block-format input; this wrapper routes F32 to the
/// dense F32 matmul kernel that mlx-native already ships.
///
/// Takes `registry` and `device` separately to avoid borrow conflicts on
/// `GpuContext` (registry is `&mut`, device is `&`).
pub fn dispatch_qmatmul(
    session: &mut GraphSession<'_>,
    registry: &mut mlx_native::KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxQWeight,
    output: &MlxBuffer,
    m: u32,
) -> Result<()> {
    // ADR-029 iter-40 H40 — annotate the next-captured dispatch with this
    // call's exact reads/writes so the graph_opt reorder pass (graph.rs
    // `ComputeGraph::reorder`) can detect non-conflicts between adjacent
    // dispatches (e.g. Q/K/V which write to disjoint outputs from the
    // same input).  Without per-dispatch annotation, the prior
    // `barrier_between` carries the union of write targets and the reorder
    // pass treats the dispatches as conflicting.  The annotation only
    // takes effect in `begin_recorded` mode (HF2Q_GRAPH_OPT_PREFILL=1);
    // in direct-dispatch mode it's a no-op zero-cost ranges stash.
    //
    // Weights are read-only (never mutated post-load), so we annotate
    // reads=[input] only — weight RANGES would force conservative
    // RAW-conflict checks against the load-time dequant pass that ran
    // hours ago (already complete, no live dependency).
    {
        let input_range = {
            let start = input.contents_ptr() as usize;
            (start, start + input.byte_len())
        };
        let output_range = {
            let start = output.contents_ptr() as usize;
            (start, start + output.byte_len())
        };
        session.encoder_mut().set_pending_buffer_ranges(
            vec![input_range],
            vec![output_range],
        );
    }

    // ADR-020 AC#5 Iter C — affine route MUST be checked first (before
    // F32/GGML).  When `weight.affine` is `Some`, the buffer holds
    // packed-U32 mlx-affine codes and `info.ggml_dtype` is a sentinel.
    if let Some(extra) = weight.affine.as_ref() {
        if extra.bits != 4 || extra.group_size != 32 {
            return Err(anyhow::anyhow!(
                "dispatch_qmatmul affine: only bits=4, group_size=32 supported in AC#5 Iter C; got bits={} gs={}",
                extra.bits,
                extra.group_size,
            ));
        }
        let n = weight.info.rows as u32;
        let k = weight.info.cols as u32;
        // Per-call meta buffer [M, N, K, group_size] — M varies per
        // dispatch (decode m=1 vs prefill m≥2), so it can't be cached
        // on the weight.
        let mut meta = device
            .alloc_buffer(16, mlx_native::DType::U32, vec![4])
            .map_err(|e| anyhow::anyhow!("affine meta alloc: {e}"))?;
        meta.as_mut_slice::<u32>()
            .map_err(|e| anyhow::anyhow!("affine meta slice: {e}"))?
            .copy_from_slice(&[m, n, k, extra.group_size]);
        return mlx_native::ops::qmm_affine::dispatch_qmm_affine_t_packed_simd4_b4(
            session.encoder_mut(),
            registry,
            device.metal_device(),
            input,
            &weight.buffer,
            &extra.scales,
            &extra.biases,
            output,
            &meta,
            m,
            n,
            k,
            extra.group_size,
            extra.bits,
        )
        .map_err(|e| anyhow::anyhow!("qmm_affine_t_packed_simd4_b4 failed: {e}"));
    }

    // ADR-029 iter-28 H29 / iter-30 H29-speed — F16 pre-dequant fast path.
    //
    // When a quantized weight has been pre-dequantized to F16 at load
    // (via populate_f16_shadow_if_enabled under HF2Q_F16_SHADOW=1),
    // route m > MM_ROUTING_THRESHOLD (= 8, prefill) through the V2-tile
    // F16-weight × F32-input mat-mat kernel (`hf2q_mul_mm_tensor_v2_f16`).
    // This is the F16-input analog of the V2 quantized kernel — same
    // 64×128 tile + direct-device B-read — without the per-call dequant
    // work inside the matmul.  Peer's gemma4 strategy.
    //
    // Decode m=1 still routes through the quantized branch below (the
    // dequant cost is amortized over fewer dispatches, and the m=1 mv
    // kernel is bandwidth-bound — no clear win for F16 there).
    if m > mlx_native::ops::quantized_matmul_ggml::MM_ROUTING_THRESHOLD {
        if let Some(ref f16w) = weight.f16_shadow {
            let n = weight.info.rows as u32;
            let k = weight.info.cols as u32;
            return mlx_native::ops::quantized_matmul_ggml::dispatch_mm_v2_f16(
                session.encoder_mut(),
                registry,
                device,
                f16w,
                input,
                output,
                m,
                n,
                k,
            )
            .map_err(|e| anyhow::anyhow!("dispatch_qmatmul F16-shadow V2 path failed: {e}"));
        }
    }

    if weight.info.ggml_dtype == mlx_native::GgmlType::F32 {
        // F32 dense path.  Weight buffer holds [n_rows, k_cols] f32 row-major.
        //
        // ADR-029 iter-18 (H26): for m=1 (decode), the matrix-MATRIX tile
        // kernel `dense_matmul_f32_f32_tensor` wastes 87.5% of its 8x8 SIMD-
        // group-matrix tile (uses 1 row of input out of 8 loaded).  Route
        // m=1 dispatches through `dispatch_dense_matvec_f32` (mat-VECTOR
        // kernel, MTLSize threads=32x2, 4 rows/dst x 2 SGs) which matches
        // peer's `kernel_mul_mv_f32_f32` pattern.  Gemma4 router_proj
        // (ffn_gate_inp F32 [2816,128]) measured ~73 µs/layer via the
        // mat-mat kernel under HF2Q_FFN_SPLIT bisect; the mat-vec kernel
        // is bandwidth-bound at ~7-10 µs/layer = ~63 µs/layer x 30 layers
        // = ~1.9 ms/tok savings if H26 holds.  Opt-out via
        // HF2Q_F32_MATVEC=0 (legacy mat-mat path); default ON for m=1.
        let n = weight.info.rows as u32;
        let k = weight.info.cols as u32;
        let f32_matvec_default = std::env::var("HF2Q_F32_MATVEC")
            .ok()
            .map(|v| !matches!(v.as_str(), "0" | "false" | "off"))
            .unwrap_or(true);
        if m == 1 && f32_matvec_default {
            let params = mlx_native::ops::dense_gemm::DenseGemmF16Params {
                m, n, k,
            };
            return mlx_native::ops::dense_gemm::dispatch_dense_matvec_f32(
                session.encoder_mut(),
                registry,
                device.metal_device(),
                input,
                &weight.buffer,
                output,
                &params,
            )
            .map_err(|e| anyhow::anyhow!("dispatch_dense_matvec_f32 failed: {e}"));
        }
        // m>=2 (prefill) or opt-out: fall back to mat-mat tile kernel.
        let params = mlx_native::DenseMmF32F32Params {
            m,
            n,
            k,
            src0_batch: 1,
            src1_batch: 1,
        };
        return mlx_native::dense_matmul_f32_f32_tensor(
            session.encoder_mut(),
            registry,
            device,
            &weight.buffer,
            input,
            output,
            &params,
        )
        .map_err(|e| anyhow::anyhow!("dense_matmul_f32_f32_tensor failed: {e}"));
    }

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
