//! Kernel and token profiling for the Gemma 4 forward pass.
//!
//! Moved from `src/serve/forward_mlx.rs` by ADR-038 Step 3.

use super::model::MlxModelWeights;
use crate::debug::INVESTIGATION_ENV;

// ---------------------------------------------------------------------------
// Profiling support (HF2Q_MLX_PROFILE=1)
// ---------------------------------------------------------------------------

/// Check if profiling is enabled via environment variable.
pub(super) fn profiling_enabled() -> bool {
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
    pub layer_s1_us: Vec<f64>, // QKV projections
    pub layer_cpu1_us: Vec<f64>, // head norms, RoPE, KV cache
    pub layer_s2_us: Vec<f64>,   // SDPA + MLP
    pub layer_cpu2_us: Vec<f64>, // post-FF norm, MoE routing prep
    pub layer_s3_us: Vec<f64>,   // router proj
    pub layer_cpu3_us: Vec<f64>, // softmax + top-k
    pub layer_s4_us: Vec<f64>,   // MoE experts
    pub layer_cpu4_us: Vec<f64>, // post-MoE norms, combine, scalar
    pub head_session_us: f64,    // lm_head session
    pub head_cpu_us: f64,        // softcap + argmax CPU
    pub total_us: f64,
    /// Dispatch counts per session type.
    pub s1_dispatches: Vec<usize>,
    pub s2_dispatches: Vec<usize>,
    pub s3_dispatches: Vec<usize>,
    pub s4_dispatches: Vec<usize>,
    pub head_dispatches: usize,
}

/// Merge worker's `TokenProfile` into the main thread's profile (ADR-031 Phase B).
///
/// Per-layer Vec fields (layer_s1_us, s1_dispatches, …) are pre-allocated to
/// num_layers at forward_decode entry.  The worker writes only the range_a
/// indices (all others stay at 0.0/0); the main thread wrote the range_b
/// indices.  After this merge, every index is populated exactly once.
/// Scalar fields (head_session_us, head_cpu_us, total_us, head_dispatches) are
/// written by main (post-loop) and are not touched here.
pub(super) fn merge_profiles(main: &mut Option<TokenProfile>, worker: Option<TokenProfile>) {
    let (Some(m), Some(w)) = (main.as_mut(), worker) else {
        return; // profiling is disabled; nothing to merge.
    };
    // Per-layer f64 Vecs: element-wise addition.
    for (mv, wv) in [
        (&mut m.layer_s1_us, &w.layer_s1_us),
        (&mut m.layer_cpu1_us, &w.layer_cpu1_us),
        (&mut m.layer_s2_us, &w.layer_s2_us),
        (&mut m.layer_cpu2_us, &w.layer_cpu2_us),
        (&mut m.layer_s3_us, &w.layer_s3_us),
        (&mut m.layer_cpu3_us, &w.layer_cpu3_us),
        (&mut m.layer_s4_us, &w.layer_s4_us),
        (&mut m.layer_cpu4_us, &w.layer_cpu4_us),
    ] {
        for (mi, wi) in mv.iter_mut().zip(wv.iter()) {
            *mi += *wi;
        }
    }
    // Per-layer usize dispatch Vecs: element-wise addition.
    for (mv, wv) in [
        (&mut m.s1_dispatches, &w.s1_dispatches),
        (&mut m.s2_dispatches, &w.s2_dispatches),
        (&mut m.s3_dispatches, &w.s3_dispatches),
        (&mut m.s4_dispatches, &w.s4_dispatches),
    ] {
        for (mi, wi) in mv.iter_mut().zip(wv.iter()) {
            *mi += *wi;
        }
    }
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
            let total: f64 = measured.iter().map(|t| getter(t).iter().sum::<f64>()).sum();
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
            1 // Single session for entire forward pass
        } else {
            num_layers * 2 + 1
        };
        eprintln!(
            "║ {} session(s)/token (single-session mode)",
            actual_sessions
        );
        eprintln!("║");
        eprintln!("║ Session breakdown (avg across {num_layers} layers, {n} tokens):");
        eprintln!(
            "║   S1 (QKV+attn+MLP):  {:8.1} us ({:5.2} ms total)",
            s1_avg / num_layers as f64,
            s1_avg / 1000.0
        );
        eprintln!(
            "║   CPU1 (eliminated):   {:8.1} us ({:5.2} ms total)",
            cpu1_avg / num_layers as f64,
            cpu1_avg / 1000.0
        );
        eprintln!(
            "║   S2 (SDPA+MLP):      {:8.1} us ({:5.2} ms total)",
            s2_avg / num_layers as f64,
            s2_avg / 1000.0
        );
        eprintln!(
            "║   CPU2 (post-FF):      {:8.1} us ({:5.2} ms total)",
            cpu2_avg / num_layers as f64,
            cpu2_avg / 1000.0
        );
        eprintln!(
            "║   S3 (router proj):   {:8.1} us ({:5.2} ms total)",
            s3_avg / num_layers as f64,
            s3_avg / 1000.0
        );
        eprintln!(
            "║   CPU3 (softmax+topk): {:8.1} us ({:5.2} ms total)",
            cpu3_avg / num_layers as f64,
            cpu3_avg / 1000.0
        );
        eprintln!(
            "║   S4 (MoE experts):   {:8.1} us ({:5.2} ms total)",
            s4_avg / num_layers as f64,
            s4_avg / 1000.0
        );
        eprintln!(
            "║   CPU4 (post-MoE):     {:8.1} us ({:5.2} ms total)",
            cpu4_avg / num_layers as f64,
            cpu4_avg / 1000.0
        );
        eprintln!(
            "║   Head GPU:            {:8.1} us ({:5.2} ms)",
            head_gpu_avg,
            head_gpu_avg / 1000.0
        );
        eprintln!(
            "║   Head CPU:            {:8.1} us ({:5.2} ms)",
            head_cpu_avg,
            head_cpu_avg / 1000.0
        );
        eprintln!("║");
        eprintln!(
            "║ Total: {:8.1} us ({:5.2} ms)",
            total_avg,
            total_avg / 1000.0
        );
        eprintln!(
            "║   GPU sessions: {:8.1} us ({:5.1}%)",
            gpu_total,
            gpu_total / total_avg * 100.0
        );
        eprintln!(
            "║   CPU ops:      {:8.1} us ({:5.1}%)",
            cpu_total,
            cpu_total / total_avg * 100.0
        );
        let overhead = total_avg - gpu_total - cpu_total;
        if overhead.abs() > 10.0 {
            eprintln!(
                "║   Unaccounted:  {:8.1} us ({:5.1}%)",
                overhead,
                overhead / total_avg * 100.0
            );
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
            let total: usize = measured
                .iter()
                .map(|t| getter(t).last().copied().unwrap_or(0))
                .sum();
            total as f64 / n as f64
        };
        let s1_disp = last_layer_dispatch_avg(&|t| &t.s1_dispatches);
        let s2_disp = last_layer_dispatch_avg(&|t| &t.s2_dispatches);
        let s3_disp = last_layer_dispatch_avg(&|t| &t.s3_dispatches);
        let s4_disp = last_layer_dispatch_avg(&|t| &t.s4_dispatches);
        let total_token_disp: f64 = measured
            .iter()
            .map(|t| t.head_dispatches as f64)
            .sum::<f64>()
            / n as f64;
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
            if num_layers > 3 {
                v.push(num_layers - 1);
            }
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

impl MlxModelWeights {
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
            let mut sums: Vec<f64> = profiles
                .iter()
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

        let gpu_total = qkv_total
            + norms_rope_total
            + kv_cache_total
            + sdpa_total
            + o_proj_total
            + mlp_total
            + moe_total
            + norms_adds_total
            + head_total;

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

        let candle_per_layer_total = candle_qkv_per_layer
            + candle_norms_rope_per_layer
            + candle_kv_cache_per_layer
            + candle_sdpa_per_layer
            + candle_o_proj_per_layer
            + candle_mlp_per_layer
            + candle_moe_per_layer
            + candle_norms_adds_per_layer;
        let candle_layers_total = candle_per_layer_total * num_layers as f64;
        let candle_total_reconstructed = candle_layers_total + candle_lm_head;

        eprintln!("\n=== PER-KERNEL-TYPE PROFILING (median over {n} tokens) ===");
        eprintln!("Per layer ({num_layers} layers):");
        eprintln!(
            "  QKV matmuls (norm+3 proj):       {:7.0} us  [candle: ~{:.0} us]  ratio: {:.1}x",
            qkv_per_layer,
            candle_qkv_per_layer,
            qkv_per_layer / candle_qkv_per_layer
        );
        eprintln!(
            "  Head norms + RoPE (3 dispatches): {:7.0} us  [candle: ~{:.0} us]  ratio: {:.1}x",
            norms_rope_per_layer,
            candle_norms_rope_per_layer,
            norms_rope_per_layer / candle_norms_rope_per_layer
        );
        eprintln!(
            "  KV cache copy (2 dispatches):    {:7.0} us  [candle: ~{:.0} us]  ratio: {:.1}x",
            kv_cache_per_layer,
            candle_kv_cache_per_layer,
            kv_cache_per_layer / candle_kv_cache_per_layer
        );
        eprintln!(
            "  SDPA (1 dispatch):               {:7.0} us  [candle: ~{:.0} us]  ratio: {:.1}x",
            sdpa_per_layer,
            candle_sdpa_per_layer,
            sdpa_per_layer / candle_sdpa_per_layer
        );
        eprintln!(
            "  O-proj matmul (1 dispatch):      {:7.0} us  [candle: ~{:.0} us]  ratio: {:.1}x",
            o_proj_per_layer,
            candle_o_proj_per_layer,
            o_proj_per_layer / candle_o_proj_per_layer
        );
        eprintln!(
            "  MLP matmuls (norm+3proj+gelu):   {:7.0} us  [candle: ~{:.0} us]  ratio: {:.1}x",
            mlp_per_layer,
            candle_mlp_per_layer,
            mlp_per_layer / candle_mlp_per_layer
        );
        eprintln!(
            "  MoE (routing+4 expert):          {:7.0} us  [candle: ~{:.0} us]  ratio: {:.1}x",
            moe_per_layer,
            candle_moe_per_layer,
            moe_per_layer / candle_moe_per_layer
        );
        eprintln!(
            "  Fused norms/adds (2 dispatches): {:7.0} us  [candle: ~{:.0} us]  ratio: {:.1}x",
            norms_adds_per_layer,
            candle_norms_adds_per_layer,
            norms_adds_per_layer / candle_norms_adds_per_layer
        );
        eprintln!();
        eprintln!("Head:");
        eprintln!(
            "  lm_head GEMM (F16):              {:7.0} us  [candle: ~{:.0} us]  ratio: {:.1}x",
            head_total,
            candle_lm_head,
            head_total / candle_lm_head
        );
        eprintln!();
        eprintln!(
            "Total GPU per token:               {:7.0} us  [candle: ~{:.0} us]  ratio: {:.1}x",
            gpu_total,
            candle_total_reconstructed,
            gpu_total / candle_total_reconstructed
        );
        eprintln!(
            "  Layers total:                    {:7.0} us  [candle: ~{:.0} us]",
            gpu_total - head_total,
            candle_layers_total
        );
        eprintln!(
            "  Head total:                      {:7.0} us  [candle: ~{:.0} us]",
            head_total, candle_lm_head
        );

        // Per-layer detail for sliding vs global
        eprintln!();
        eprintln!("Per-layer detail (median token, us):");
        eprintln!("  Layer | Type |    QKV | Nrm+RoPE |  KV$ |  SDPA | O-proj |    MLP |    MoE | Norms | Total");
        eprintln!("  ------|------|--------|----------|------|-------|--------|--------|--------|-------|------");
        let mid = profiles.len() / 2;
        let median_p = &profiles[mid]; // approximate median token
        for li in 0..num_layers {
            let lt = if (li + 1) % 6 == 0 { "G" } else { "S" };
            let layer_total = median_p.qkv_matmuls_us[li]
                + median_p.head_norms_rope_us[li]
                + median_p.kv_cache_copy_us[li]
                + median_p.sdpa_us[li]
                + median_p.o_proj_us[li]
                + median_p.mlp_matmuls_us[li]
                + median_p.moe_us[li]
                + median_p.norms_adds_us[li];
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
            (
                "QKV matmuls",
                qkv_per_layer,
                candle_qkv_per_layer,
                qkv_per_layer / candle_qkv_per_layer,
            ),
            (
                "Head norms + RoPE",
                norms_rope_per_layer,
                candle_norms_rope_per_layer,
                norms_rope_per_layer / candle_norms_rope_per_layer,
            ),
            (
                "KV cache copy",
                kv_cache_per_layer,
                candle_kv_cache_per_layer,
                kv_cache_per_layer / candle_kv_cache_per_layer,
            ),
            (
                "SDPA",
                sdpa_per_layer,
                candle_sdpa_per_layer,
                sdpa_per_layer / candle_sdpa_per_layer,
            ),
            (
                "O-proj matmul",
                o_proj_per_layer,
                candle_o_proj_per_layer,
                o_proj_per_layer / candle_o_proj_per_layer,
            ),
            (
                "MLP matmuls",
                mlp_per_layer,
                candle_mlp_per_layer,
                mlp_per_layer / candle_mlp_per_layer,
            ),
            (
                "MoE",
                moe_per_layer,
                candle_moe_per_layer,
                moe_per_layer / candle_moe_per_layer,
            ),
            (
                "Fused norms/adds",
                norms_adds_per_layer,
                candle_norms_adds_per_layer,
                norms_adds_per_layer / candle_norms_adds_per_layer,
            ),
            (
                "lm_head GEMM",
                head_total,
                candle_lm_head,
                head_total / candle_lm_head,
            ),
        ];
        ratios.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
        eprintln!();
        eprintln!("TOP 3 SLOWEST (highest mlx-native/candle ratio):");
        for (i, (name, mlx_us, candle_us, ratio)) in ratios.iter().take(3).enumerate() {
            let overhead_per_token = (mlx_us - candle_us)
                * if *name != "lm_head GEMM" {
                    num_layers as f64
                } else {
                    1.0
                };
            eprintln!(
                "  {}. {} — {:.1}x slower ({:.0} vs {:.0} us/layer) — {:.0} us/token overhead",
                i + 1,
                name,
                ratio,
                mlx_us,
                candle_us,
                overhead_per_token
            );
        }

        eprintln!();
        eprintln!("NOTE: Per-session overhead (~30-50 us/session) inflates all groups.");
        eprintln!("      The ratio shows relative slowness, not absolute kernel time.");
        eprintln!(
            "      {} sessions/token vs 1 in production mode.",
            8 * num_layers + 2
        );
    }
}
