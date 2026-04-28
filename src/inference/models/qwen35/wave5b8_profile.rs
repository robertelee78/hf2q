//! Wave 5b.8 PP4096 measurement-spike instrumentation.
//!
//! Env gate: `HF2Q_PROFILE_W5B8=1` (default OFF — production codepath
//! unaffected when unset; matches the `HF2Q_DECODE_PROFILE` convention
//! at `forward_gpu.rs:698`).
//!
//! Purpose: capture per-section CPU wall-clock for the chunk-pipeline
//! prefill body so the W-5b.8 measurement spike (8.5× wall-clock gap
//! vs llama.cpp at pp4096) can be ranked by absolute ms contribution
//! instead of analytical guess. Zero kernel changes; no GPU
//! timestamping (per memory `project_m5max_no_dispatch_boundary_sampling`,
//! M5 Max only supports stage-boundary GPU counter sampling — CPU wall
//! captures encoder-split + commit_and_wait + memcpy + kernel-launch
//! overhead in a single number, which is what the gap is made of).
//!
//! Usage:
//! ```ignore
//! if w5b8_enabled() {
//!     let _t = Section::start(SectionKind::ChunkExpand);
//!     // ... work ...
//!     // _t drops -> recorded into thread-local accumulator
//! }
//! ```
//!
//! Print summary on demand via `w5b8_print_and_reset(label)` from
//! `forward_gpu_impl` after the per-layer loop completes.

use std::cell::RefCell;
use std::time::{Duration, Instant};

/// Sections instrumented in this measurement spike. Names map directly
/// to W-5b.8 task buckets — keep them stable so the docs section parses
/// the eprintln summary cleanly.
#[derive(Copy, Clone, Debug)]
pub enum SectionKind {
    /// One-time `upload_layer_weights_gpu` first-call cost.
    UploadWeights,
    /// `build_delta_net_layer` prefill ops1-3 encoder (pre_norm + qkv_proj
    /// + z_proj + ssm_conv + commit_and_wait). Per linear-attn layer.
    LayerOps1to3,
    /// CPU-side qkv_conv download + de-interleave + 3 GPU re-uploads
    /// (q_gpu, k_gpu, v_gpu). Per linear-attn layer.
    LayerQkvDeinterleave,
    /// Chunk-prep encoder (l2_norm q/k, alpha/beta proj, q_scale, g_beta
    /// + commit_and_wait). Per linear-attn layer, chunk path only.
    LayerChunkPrep,
    /// `apply_gated_delta_net_chunk` total wall (= sum of the four
    /// sub-buckets below + return-value memcpy). Per linear-attn layer.
    LayerChunkCall,
    /// Inside chunk wrapper: GQA F32 expansion CPU memcpy
    /// (`q_expanded`/`k_expanded` allocation + tiled fill loop, lines
    /// 880-933 of `gpu_delta_net.rs`).
    ChunkGqaExpand,
    /// Inside chunk wrapper: scratch BF16 + g_log_decay + final_state +
    /// output buffer allocations.
    ChunkAllocs,
    /// Inside chunk wrapper: wall around the single mega-encoder build
    /// (cast → sign-flip → `dispatch_chunk_gated_delta_rule_fwd` → cast
    /// back) — i.e. encoder build time without the commit.
    ChunkEncBuild,
    /// Inside chunk wrapper: the `enc.commit_and_wait()` at the very end
    /// of `apply_gated_delta_net_chunk` — measures GPU wait time for the
    /// 6-kernel chunk pipeline + 4 casts.
    ChunkCommitWait,
    /// Chunk-ops8-9 encoder (ssm_norm_gate + out_proj + commit_and_wait).
    /// Per linear-attn layer, chunk path only.
    LayerChunkOps8to9,
    /// Autoregressive ops5-9 encoder (l2_norm + alpha/beta proj +
    /// q_scale + g_beta + GDN + ssm_norm_gate + out_proj +
    /// commit_and_wait). Per linear-attn layer, autoreg path only.
    LayerAutoregOps5to9,
    /// Per linear-attn layer total wall (== sum of all per-layer buckets
    /// for that path; the residual after subtraction reveals
    /// non-instrumented overhead).
    LayerLinearTotal,
    /// Per full-attn layer total wall — includes the gated-attn
    /// build_gated_attn_layer call (no chunk pipeline involved).
    LayerFullTotal,
    // -- Wave 5b.9 sub-FA buckets (additive; reuses HF2Q_PROFILE_W5B8 gate). --
    // The W-5b.8 closure recommended a per-FA-layer breakdown to attribute
    // the 23.2 s `LayerFullTotal` bucket. The buckets below carve up
    // `build_gated_attn_layer` (`gpu_full_attn.rs:1119`) into:
    //   1. ops 1–4 session (pre-norm + 4 projections + per-head Q/K norm + IMROPE×2)
    //   2. ops 5 SDPA prefill else-branch in `apply_sdpa_with_kv_cache`,
    //      itself sub-divided into the 6 CPU↔GPU stages it performs
    //      around the actual `sdpa` kernel call
    //   3. ops 6–7 session (sigmoid-gate multiply + O-proj)
    /// FA op group 1 — pre-norm + Q/K/V/G linear projections + per-head Q/K
    /// RMS-norm + IMROPE on Q and K. Per FA layer (16 layers in Qwen3.6 27B).
    /// Implementation: `gpu_full_attn.rs:1156-1227` (single encoder, ends with
    /// `commit_and_wait` for prefill seq>1).
    FaOps1to4,
    /// FA op 5 — `apply_sdpa_with_kv_cache` total wall (prefill else-branch
    /// at `gpu_full_attn.rs:1027-1084`). This is the CPU↔GPU sandwich
    /// containing 3 GPU→CPU downloads, a CPU triple-loop KV cache write, a
    /// CPU permute, an upload, the SDPA kernel itself, an output download,
    /// an output CPU permute, and a final upload.
    FaSdpaTotal,
    /// Inside FA op 5: GPU→CPU downloads of K and V (lines 1030-1031)
    /// + CPU triple-nested loop writing them into the head-major KV cache
    /// (lines 1033-1046).
    FaSdpaKvDownloadCopy,
    /// Inside FA op 5: GPU→CPU download of Q + CPU permute to head-major +
    /// re-upload to GPU (lines 1052-1054).
    FaSdpaQDownloadPermuteUpload,
    /// Inside FA op 5: the actual `sdpa` kernel dispatch + commit_and_wait
    /// (lines 1065-1068). This is mlx-native's tiled SDPA kernel — the
    /// "real GPU work" inside the bucket.
    FaSdpaKernel,
    /// Inside FA op 5: output GPU→CPU download + CPU permute back to
    /// seq-major + final re-upload (lines 1071-1083).
    FaSdpaOutDownloadPermuteUpload,
    /// FA op group 6–7 — sigmoid gate × multiply + O-proj projection.
    /// Implementation: `gpu_full_attn.rs:1251-1276` (single encoder, ends
    /// with `commit_and_wait` for prefill seq>1).
    FaOps6to7,
    // -- Wave 5b.11 post-attention sub-buckets (additive; reuses
    // HF2Q_PROFILE_W5B8 gate). --
    //
    // The W-5b.8/9 closure showed that `layer.linear_total` (~402 ms/layer
    // chunk path × 48 DN layers ≈ 19.3 s) is only ~199 ms/layer accounted
    // for by the existing `layer.ops1_3 + qkv_deinterleave + chunk_prep
    // + chunk_call + chunk_ops8_9` buckets — leaving ~203 ms/layer
    // unprofiled. That residual lives **outside** the `apply_gated_delta_net_chunk`
    // wrapper, in the post-attention path: fused residual+norm encoder
    // (`forward_gpu.rs:906-927`), FFN dispatch (MoE-Q for Qwen3.6 27B
    // linear-attn layers; line 999-1001), and the post-FFN residual
    // (line 1013-1022; folded into FFN for MoeQ/Dense*). The buckets below
    // partition that residual.
    /// Fused residual + post-attention RMSNorm encoder. One encoder +
    /// `commit()` (no wait) per layer; runs `dispatch_fused_residual_norm_f32`
    /// (`forward_gpu.rs:906-927`). Counts both linear-attn and full-attn
    /// layers.
    LayerPostAttnFusedNorm,
    /// FFN dispatch wall. For Qwen3.6 27B: dense layers go through
    /// `build_dense_ffn_layer_gpu_q`, MoE layers go through
    /// `build_moe_ffn_layer_gpu_q` (`forward_gpu.rs:957-1001`). Residual
    /// is folded into the dispatch via `Some(&ffn_residual)`. One bucket
    /// across all FFN variants — the trial summary documents which variant
    /// fired by inspecting the model architecture (Qwen3.6 27B = MoeQ for
    /// every layer index that's a linear-attn layer).
    LayerFfnDispatch,
    /// Post-FFN residual fall-through wall. For MoeQ/DenseQ/Dense the
    /// residual is folded into the FFN dispatch above so this is a no-op
    /// match-arm pass-through (~ns). For F32-MoE this triggers the separate
    /// `residual_add_gpu` GPU encoder (`forward_gpu.rs:1020-1021`) — measured
    /// here so we know whether F32-MoE is engaged anywhere.
    LayerFfnPostResidual,
    // -- Wave 5b.17 DN-wrapper-overhead sub-buckets (additive; gated on a
    // SEPARATE `HF2Q_PROFILE_W5B17=1` env so default-off behaviour is
    // preserved even when `HF2Q_PROFILE_W5B8=1`). --
    //
    // The W-5b.16 closure measured `layer.linear_total` 7,287 ms with sub-
    // buckets `ops1_3 + qkv_deinterleave + chunk_prep + chunk_call +
    // chunk_ops8_9 = 3,962 ms` accounted for; the residual ~3,300 ms is
    // the FFN dispatch portion attributable to DN layers (already named via
    // `layer.ffn_dispatch`). Of the 3,962 ms DN-attention work,
    // `qkv_deinterleave` (826 ms) is wholly CPU GPU↔CPU round-trip (the
    // chunk wrapper's `chunk_call` already partitions into gqa_expand +
    // allocs + enc_build + commit_wait). These W-5b.17 buckets carve up
    // the unprofiled corners:
    //   - `dn.qkv_gpu_split` (W-5b.18 GPU `dispatch_qkv_split_f32` dispatch +
    //     `commit_and_wait`; replaced the legacy CPU round-trip). The
    //     `dn.qkv_download` / `dn.qkv_cpu_loop` / `dn.qkv_uploads` buckets
    //     were removed in W-5b.19 alongside the legacy gate they tracked.
    //   - `dn.state_pingpong_memcpy` (the unsafe copy_nonoverlapping at
    //     gpu_delta_net.rs:1625-1628 — n_state F32 per layer × 48 layers
    //     across the chunk-final-state → caller-state-out ping-pong slot).
    /// Chunk-pipeline final_state → caller state_out ping-pong copy at
    /// gpu_delta_net.rs:1625-1628 (`std::ptr::copy_nonoverlapping`,
    /// n_state = d_k × d_v × n_v_heads × 4 B per layer ≈ 4 MB × 48
    /// layers ≈ 200 MB CPU memcpy total at PP4096).
    DnStatePingpongMemcpy,
    /// W-5b.18 GPU `dispatch_qkv_split_f32` dispatch + `commit_and_wait`.
    /// Per linear-attn layer, prefill seq>1 only. The default and only
    /// path post-W-5b.19 (the legacy CPU round-trip and its
    /// `HF2Q_QKV_SPLIT_LEGACY=1` env gate were removed after a 30/30
    /// parity audit at PP4106).
    DnQkvGpuSplit,
    // -- Wave 5b.22 OUTER-per-DN-layer-choreography sub-buckets (additive;
    // gated on a SEPARATE `HF2Q_PROFILE_W5B22=1` env so default-off is
    // preserved even when `HF2Q_PROFILE_W5B8=1` and/or
    // `HF2Q_PROFILE_W5B17=1` are set). --
    //
    // Mission: attribute the W-5b.21 `layer.linear_total` 3,318 ms residual
    // (= 6,134 ms LayerLinearTotal − 2,816 ms named DN-attn sub-buckets).
    // The `_w5b8_layer_total` guard at `forward_gpu.rs:724-731` lexically
    // wraps the WHOLE per-layer iteration body (lines 724-1165): so the
    // residual ms include `LayerPostAttnFusedNorm` + `LayerFfnDispatch` +
    // `LayerFfnPostResidual` even though those have their own buckets —
    // those existing buckets are 64-layer aggregates (DN+FA), not DN-only,
    // so subtraction can't isolate the DN portion.
    //
    // Pre-implementation read-first per `feedback_structural_audit_before_kernel_work`:
    // The W-5b.21 doc speculated the residual is a "q/k/v/o projection
    // mat-mul wall," but reading `build_delta_net_layer`
    // (`gpu_delta_net.rs:1417-1715` prefill chunk path) shows DeltaNet has
    // NO separate q_proj/k_proj/v_proj/o_proj — instead it has:
    //   - `attn_qkv` (fused QKV+Z) inside `LayerOps1to3` already
    //   - `attn_gate` (Z proj) inside `LayerOps1to3` already
    //   - `ssm_alpha`/`ssm_beta` inside `LayerChunkPrep` already
    //   - `ssm_out` (out_proj) inside `LayerChunkOps8to9` already
    // Every DN projection is already inside a named sub-bucket. So the
    // 3,318 ms residual is NOT inside `build_delta_net_layer` — it's the
    // post-attn outer choreography. The buckets below partition that.
    //
    /// DN-only sister of `LayerPostAttnFusedNorm`. Same encoder, same
    /// timer span, but only fires for `LinearAttn` layer slots (48 of 64
    /// in Qwen3.6 27B). Lets us isolate the DN-layer portion of the
    /// 64-layer-aggregate `LayerPostAttnFusedNorm` bucket so the
    /// `LayerLinearTotal` residual subtraction is clean.
    DnOuterPostAttnNorm,
    /// DN-only sister of `LayerFfnDispatch`. Same dispatch, same timer
    /// span, but only fires for `LinearAttn` layer slots. The
    /// 64-layer-aggregate `LayerFfnDispatch` was 4,389 ms in W-5b.21;
    /// the DN portion is 48/64 of that ≈ 3,292 ms by simple arithmetic
    /// — this bucket measures it directly so the attribution is exact.
    DnOuterFfnDispatch,
    /// DN-only sister of `LayerFfnPostResidual`. For Qwen3.6 27B's
    /// MoeQ FFN this is a no-op match-arm pass (~ns); included for
    /// completeness so the residual subtraction has zero unaccounted
    /// terms.
    DnOuterPostFfnResidual,
    /// Sum of `DnOuter*` sub-buckets (post-attn outer choreography for
    /// DN layers). Should land within rounding-error of the
    /// `LayerLinearTotal` − DN-attn-buckets residual.
    DnOuterChoreographyTotal,
    // -- Wave 5b.26 FFN sub-buckets (additive; gated on a SEPARATE
    // `HF2Q_PROFILE_W5B26=1` env so default-off is preserved even when
    // `HF2Q_PROFILE_W5B8=1` and/or `HF2Q_PROFILE_W5B22=1` are set). --
    //
    // Mission: attribute the `DnOuterFfnDispatch` 3,229 ms post-W-5b.24
    // baseline.  Carves `build_moe_ffn_layer_gpu_q_into`
    // (`gpu_ffn.rs:1366-1626`) into the 11 dispatch sub-phases the audit
    // doc-comment lays out (`gpu_ffn.rs:1395-1407` Phase A→F sequence).
    // Sum of these sub-buckets should land within ±5 % of the parent
    // `DnOuterFfnDispatch` bucket; the residual fingers any
    // un-instrumented overhead (e.g. `pooled_alloc_buffer` lookups, 2
    // `silu_params_buf.as_mut_slice` writes, 6 `enc.memory_barrier()`
    // calls).
    /// Phase A — `proj_pooled(router)` (`gpu_ffn.rs:1471`).  One
    /// `dense_matmul_bf16_f32_tensor` dispatch per MoE layer (Phase A,
    /// concurrent with the 3 shared-expert projections below).
    DnFfnProjRouter,
    /// Phase A — `proj_pooled(shared_gate_inp)` (`gpu_ffn.rs:1472`).
    /// Single-output (out_features=1) projection that drives the shared
    /// expert sigmoid gate.
    DnFfnProjShLogit,
    /// Phase A — `proj_pooled(shared_gate)` (`gpu_ffn.rs:1473`).  Shared
    /// expert gate projection feeding the SwiGLU activation in Phase B.
    DnFfnProjAS,
    /// Phase A — `proj_pooled(shared_up)` (`gpu_ffn.rs:1474`).  Shared
    /// expert up projection feeding the SwiGLU activation in Phase B.
    DnFfnProjBS,
    /// Phase B — `dispatch_moe_softmax_topk` (`gpu_ffn.rs:1480-1484`).
    /// GPU softmax over router logits + top-k selection writing
    /// `(ids_buf, weights_buf)`.
    DnFfnSoftmaxTopk,
    /// Phase C — gate `quantized_matmul_id_ggml_pooled` for routed
    /// experts (`gpu_ffn.rs:1528-1536`).  Sparse mm_id matmul writing
    /// `gate_all_buf` over `ids_buf`.
    DnFfnGateMmId,
    /// Phase C — up `quantized_matmul_id_ggml_pooled` for routed
    /// experts (`gpu_ffn.rs:1537-1545`).  Concurrent with `DnFfnGateMmId`
    /// inside Phase C; same params except `expert_stride`.
    DnFfnUpMmId,
    /// Phase D — `dispatch_silu_mul(gate_all, up_all → h_all)`
    /// (`gpu_ffn.rs:1567-1570`).  SwiGLU activation across all routed
    /// expert tokens.
    DnFfnSiluMulPhase,
    /// Phase C — `proj_pooled(shared_down(h_s))` (`gpu_ffn.rs:1548`).
    /// Shared expert down projection.  Concurrent with gate/up mm_id
    /// inside Phase C.
    DnFfnSharedDown,
    /// Phase E — down `quantized_matmul_id_ggml_pooled` for routed
    /// experts (`gpu_ffn.rs:1592-1600`).  Sparse mm_id writing `y_all_buf`.
    DnFfnDownMmId,
    /// Phase F — `dispatch_moe_weighted_reduce`
    /// (`gpu_ffn.rs:1606-1618`).  Fused reduce-and-add: top-k weighted
    /// sum + sigmoid(sh_logit) × y_s + optional residual.
    DnFfnWeightedReduce,
}

impl SectionKind {
    fn idx(self) -> usize {
        self as usize
    }
    fn label(self) -> &'static str {
        match self {
            SectionKind::UploadWeights => "upload_weights",
            SectionKind::LayerOps1to3 => "layer.ops1_3",
            SectionKind::LayerQkvDeinterleave => "layer.qkv_deinterleave",
            SectionKind::LayerChunkPrep => "layer.chunk_prep",
            SectionKind::LayerChunkCall => "layer.chunk_call",
            SectionKind::ChunkGqaExpand => "chunk.gqa_expand",
            SectionKind::ChunkAllocs => "chunk.allocs",
            SectionKind::ChunkEncBuild => "chunk.enc_build",
            SectionKind::ChunkCommitWait => "chunk.commit_wait",
            SectionKind::LayerChunkOps8to9 => "layer.chunk_ops8_9",
            SectionKind::LayerAutoregOps5to9 => "layer.autoreg_ops5_9",
            SectionKind::LayerLinearTotal => "layer.linear_total",
            SectionKind::LayerFullTotal => "layer.full_total",
            SectionKind::FaOps1to4 => "fa.ops1_4",
            SectionKind::FaSdpaTotal => "fa.sdpa_total",
            SectionKind::FaSdpaKvDownloadCopy => "fa.sdpa.kv_dl_copy",
            SectionKind::FaSdpaQDownloadPermuteUpload => "fa.sdpa.q_dl_perm_ul",
            SectionKind::FaSdpaKernel => "fa.sdpa.kernel",
            SectionKind::FaSdpaOutDownloadPermuteUpload => "fa.sdpa.out_dl_perm_ul",
            SectionKind::FaOps6to7 => "fa.ops6_7",
            SectionKind::LayerPostAttnFusedNorm => "layer.post_attn_fused_norm",
            SectionKind::LayerFfnDispatch => "layer.ffn_dispatch",
            SectionKind::LayerFfnPostResidual => "layer.ffn_post_residual",
            SectionKind::DnStatePingpongMemcpy => "dn.state_pingpong_memcpy",
            SectionKind::DnQkvGpuSplit => "dn.qkv_gpu_split",
            SectionKind::DnOuterPostAttnNorm => "dn.outer_post_attn_norm",
            SectionKind::DnOuterFfnDispatch => "dn.outer_ffn_dispatch",
            SectionKind::DnOuterPostFfnResidual => "dn.outer_post_ffn_residual",
            SectionKind::DnOuterChoreographyTotal => "dn.outer_choreography_total",
            SectionKind::DnFfnProjRouter => "dn.ffn.proj_router",
            SectionKind::DnFfnProjShLogit => "dn.ffn.proj_sh_logit",
            SectionKind::DnFfnProjAS => "dn.ffn.proj_a_s",
            SectionKind::DnFfnProjBS => "dn.ffn.proj_b_s",
            SectionKind::DnFfnSoftmaxTopk => "dn.ffn.softmax_topk",
            SectionKind::DnFfnGateMmId => "dn.ffn.gate_mm_id",
            SectionKind::DnFfnUpMmId => "dn.ffn.up_mm_id",
            SectionKind::DnFfnSiluMulPhase => "dn.ffn.silu_mul_phase",
            SectionKind::DnFfnSharedDown => "dn.ffn.shared_down",
            SectionKind::DnFfnDownMmId => "dn.ffn.down_mm_id",
            SectionKind::DnFfnWeightedReduce => "dn.ffn.weighted_reduce",
        }
    }
    const COUNT: usize = 40;
}

#[derive(Default, Clone)]
struct Acc {
    samples: Vec<u128>, // microseconds per sample; small N (≤64 layers)
}

impl Acc {
    fn record(&mut self, dur: Duration) {
        self.samples.push(dur.as_micros());
    }
    fn count(&self) -> usize {
        self.samples.len()
    }
    fn sum_us(&self) -> u128 {
        self.samples.iter().sum()
    }
    fn min_us(&self) -> u128 {
        self.samples.iter().copied().min().unwrap_or(0)
    }
    fn max_us(&self) -> u128 {
        self.samples.iter().copied().max().unwrap_or(0)
    }
    fn mean_us(&self) -> u128 {
        if self.samples.is_empty() {
            0
        } else {
            self.sum_us() / self.samples.len() as u128
        }
    }
    fn percentile_us(&self, p: f64) -> u128 {
        if self.samples.is_empty() {
            return 0;
        }
        let mut s = self.samples.clone();
        s.sort_unstable();
        let idx = ((s.len() - 1) as f64 * p).round() as usize;
        s[idx]
    }
}

#[derive(Default, Clone)]
struct W5b8State {
    accs: Vec<Acc>,
}

impl W5b8State {
    fn new() -> Self {
        Self {
            accs: vec![Acc::default(); SectionKind::COUNT],
        }
    }
    fn record(&mut self, kind: SectionKind, dur: Duration) {
        if self.accs.is_empty() {
            self.accs = vec![Acc::default(); SectionKind::COUNT];
        }
        self.accs[kind.idx()].record(dur);
    }
}

thread_local! {
    static W5B8: RefCell<W5b8State> = RefCell::new(W5b8State::new());
}

/// True when `HF2Q_PROFILE_W5B8=1` is set in the environment.
#[inline]
pub fn w5b8_enabled() -> bool {
    // Cheap getenv every call is fine — measurement-spike code, not hot
    // path; matches the existing `HF2Q_DECODE_PROFILE` pattern at
    // `forward_gpu.rs:698`.
    std::env::var("HF2Q_PROFILE_W5B8").is_ok()
}

/// True when `HF2Q_PROFILE_W5B17=1` is set in the environment.
///
/// Separate gate from W-5b.8 so the four `DnQkv*` / `DnStatePingpongMemcpy`
/// sub-buckets only fire during W-5b.17 audits — W-5b.8/9/11/15/16 reruns
/// stay binary-identical when only `HF2Q_PROFILE_W5B8=1` is set.
#[inline]
pub fn w5b17_enabled() -> bool {
    std::env::var("HF2Q_PROFILE_W5B17").is_ok()
}

/// True when `HF2Q_PROFILE_W5B22=1` is set in the environment.
///
/// Separate gate from W-5b.8/W-5b.17 so the four `DnOuter*` sub-buckets
/// (audit of the `LayerLinearTotal` 3,318 ms residual via DN-only
/// outer-choreography sisters) only fire during W-5b.22 audits.
#[inline]
pub fn w5b22_enabled() -> bool {
    std::env::var("HF2Q_PROFILE_W5B22").is_ok()
}

/// True when `HF2Q_PROFILE_W5B26=1` is set in the environment.
///
/// Separate gate from W-5b.8/W-5b.17/W-5b.22 so the 11 `DnFfn*`
/// sub-buckets (audit of the `DnOuterFfnDispatch` 3,229 ms post-W-5b.24
/// baseline) only fire during W-5b.26 audits.  Sum of the 11 sub-buckets
/// should land within ±5 % of `DnOuterFfnDispatch` — see the
/// W-5b.26 SectionKind comment block above for the bucket breakdown.
#[inline]
pub fn w5b26_enabled() -> bool {
    std::env::var("HF2Q_PROFILE_W5B26").is_ok()
}

/// RAII guard. `Section::start(kind)` records the elapsed wall-clock
/// into the thread-local accumulator on drop; no-ops when the env gate
/// is off (the constructor still allocates an Instant, but that's a
/// few-ns rdtsc — negligible vs measurement noise).
pub struct Section {
    kind: SectionKind,
    t0: Option<Instant>,
}

impl Section {
    pub fn start(kind: SectionKind) -> Self {
        let t0 = if w5b8_enabled() { Some(Instant::now()) } else { None };
        Self { kind, t0 }
    }

    /// Variant gated on `HF2Q_PROFILE_W5B17=1`. Use for the four W-5b.17
    /// `Dn*` sub-buckets so they default off independently of W-5b.8.
    pub fn start_w5b17(kind: SectionKind) -> Self {
        let t0 = if w5b17_enabled() { Some(Instant::now()) } else { None };
        Self { kind, t0 }
    }

    /// Variant gated on `HF2Q_PROFILE_W5B22=1`. Use for the four W-5b.22
    /// `DnOuter*` sub-buckets so they default off independently of
    /// W-5b.8 and W-5b.17.
    pub fn start_w5b22(kind: SectionKind) -> Self {
        let t0 = if w5b22_enabled() { Some(Instant::now()) } else { None };
        Self { kind, t0 }
    }

    /// Variant gated on `HF2Q_PROFILE_W5B26=1`. Use for the 11 W-5b.26
    /// `DnFfn*` sub-buckets so they default off independently of
    /// W-5b.8, W-5b.17, and W-5b.22.
    pub fn start_w5b26(kind: SectionKind) -> Self {
        let t0 = if w5b26_enabled() { Some(Instant::now()) } else { None };
        Self { kind, t0 }
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        if let Some(t0) = self.t0 {
            let dur = t0.elapsed();
            W5B8.with(|cell| cell.borrow_mut().record(self.kind, dur));
        }
    }
}

/// Print a one-shot summary of all accumulated buckets to stderr and
/// reset the thread-local state. Called from `forward_gpu_impl` after
/// the per-layer loop completes — gated on `w5b8_enabled() ||
/// w5b17_enabled()` so W-5b.17 audits surface their sub-buckets even
/// when W-5b.8 is off.
pub fn w5b8_print_and_reset(label: &str) {
    if !w5b8_enabled() && !w5b17_enabled() && !w5b22_enabled() && !w5b26_enabled() {
        return;
    }
    W5B8.with(|cell| {
        let mut state = cell.borrow_mut();
        eprintln!("[W5B8_PROFILE] === section summary: {label} ===");
        eprintln!(
            "[W5B8_PROFILE] {:<26} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "section", "n", "sum_ms", "mean_ms", "min_ms", "max_ms", "p50_ms", "p95_ms"
        );
        // Iterate in the SectionKind order for stable output.
        let kinds = [
            SectionKind::UploadWeights,
            SectionKind::LayerOps1to3,
            SectionKind::LayerQkvDeinterleave,
            SectionKind::LayerChunkPrep,
            SectionKind::LayerChunkCall,
            SectionKind::ChunkGqaExpand,
            SectionKind::ChunkAllocs,
            SectionKind::ChunkEncBuild,
            SectionKind::ChunkCommitWait,
            SectionKind::LayerChunkOps8to9,
            SectionKind::LayerAutoregOps5to9,
            SectionKind::LayerLinearTotal,
            SectionKind::LayerFullTotal,
            // Wave 5b.9 FA sub-buckets:
            SectionKind::FaOps1to4,
            SectionKind::FaSdpaTotal,
            SectionKind::FaSdpaKvDownloadCopy,
            SectionKind::FaSdpaQDownloadPermuteUpload,
            SectionKind::FaSdpaKernel,
            SectionKind::FaSdpaOutDownloadPermuteUpload,
            SectionKind::FaOps6to7,
            // Wave 5b.11 post-attention sub-buckets:
            SectionKind::LayerPostAttnFusedNorm,
            SectionKind::LayerFfnDispatch,
            SectionKind::LayerFfnPostResidual,
            // Wave 5b.17 DN-wrapper-overhead sub-buckets (legacy CPU
            // qkv_download/cpu_loop/uploads removed in W-5b.19 alongside
            // the gate they tracked).
            SectionKind::DnStatePingpongMemcpy,
            // Wave 5b.18 GPU-side QKV split (the only path post-W-5b.19):
            SectionKind::DnQkvGpuSplit,
            // Wave 5b.22 outer-per-DN-layer-choreography sub-buckets:
            SectionKind::DnOuterPostAttnNorm,
            SectionKind::DnOuterFfnDispatch,
            SectionKind::DnOuterPostFfnResidual,
            SectionKind::DnOuterChoreographyTotal,
            // Wave 5b.26 FFN sub-buckets (11 phases inside
            // `build_moe_ffn_layer_gpu_q_into`):
            SectionKind::DnFfnProjRouter,
            SectionKind::DnFfnProjShLogit,
            SectionKind::DnFfnProjAS,
            SectionKind::DnFfnProjBS,
            SectionKind::DnFfnSoftmaxTopk,
            SectionKind::DnFfnGateMmId,
            SectionKind::DnFfnUpMmId,
            SectionKind::DnFfnSiluMulPhase,
            SectionKind::DnFfnSharedDown,
            SectionKind::DnFfnDownMmId,
            SectionKind::DnFfnWeightedReduce,
        ];
        for k in kinds {
            let acc = &state.accs[k.idx()];
            if acc.count() == 0 {
                continue;
            }
            eprintln!(
                "[W5B8_PROFILE] {:<26} {:>6} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
                k.label(),
                acc.count(),
                acc.sum_us() as f64 / 1000.0,
                acc.mean_us() as f64 / 1000.0,
                acc.min_us() as f64 / 1000.0,
                acc.max_us() as f64 / 1000.0,
                acc.percentile_us(0.50) as f64 / 1000.0,
                acc.percentile_us(0.95) as f64 / 1000.0,
            );
        }
        eprintln!("[W5B8_PROFILE] === end summary ===");
        // Reset for subsequent calls (the binary issues one forward at
        // pp4096 then exits, but unit tests may iterate).
        *state = W5b8State::new();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_records_only_when_enabled() {
        // gate off (default) — record() no-op
        std::env::remove_var("HF2Q_PROFILE_W5B8");
        {
            let _t = Section::start(SectionKind::UploadWeights);
            std::thread::sleep(Duration::from_millis(1));
        }
        W5B8.with(|cell| {
            let s = cell.borrow();
            assert_eq!(
                s.accs[SectionKind::UploadWeights.idx()].count(),
                0,
                "section should not record when env unset"
            );
        });

        // gate on — record() captures one sample
        std::env::set_var("HF2Q_PROFILE_W5B8", "1");
        {
            let _t = Section::start(SectionKind::UploadWeights);
            std::thread::sleep(Duration::from_millis(2));
        }
        W5B8.with(|cell| {
            let s = cell.borrow();
            assert_eq!(s.accs[SectionKind::UploadWeights.idx()].count(), 1);
            assert!(s.accs[SectionKind::UploadWeights.idx()].sum_us() >= 1_000);
        });
        // Cleanup so other tests don't see the env var.
        std::env::remove_var("HF2Q_PROFILE_W5B8");
        // Reset thread-local for downstream test isolation.
        W5B8.with(|cell| *cell.borrow_mut() = W5b8State::new());
    }
}
