//! ADR-040 Phase F M2.2 / S2+S3 — the `[N,hidden]` batched decode BODY.
//!
//! S1 batched the lm_head (post-body). S2/S3 batch the BODY itself — the layer
//! loop that dominates decode time (MoE 1.76 ms/token + dense projections). The
//! MoE sits mid-layer after per-slot attention, so batching it requires the N
//! slots' MoE inputs gathered: i.e. running the whole layer in `[N,hidden]`.
//!
//! `forward_decode_body_batched` replaces `decode_batch_gemma4` pass-1's N
//! sequential per-slot bodies with ONE batched pass, leaving each slot's final
//! hidden in `[N,hidden]`; the existing (proven) batched head + finalize then
//! complete the tick. Premise PROVEN at the kernel level before this restructure
//! (mantra): dense projections per-row bit-identical (H-S1-rowparity, mlx
//! 272b2a3), MoE `_id` per-token bit-identical across n_tokens (H-S2-tokenparity,
//! mlx 32b5045), rms_norm per-row identical (implicit via slot_aware_n4),
//! attention per-slot (N independent `flash_attn_vec_tq_hb` dispatches). So a
//! correct `[N,hidden]` body MUST be bit-identical to N serial bodies — the
//! `slot_aware_n1`/`slot_aware_n4` gates confirm it.
//!
//! This module is built incrementally with the gate held at each step:
//! 1. `BatchedDecodeBuffers` — the `[N,...]` activation scratch (this commit).
//! 2. `encode_one_layer_batched` — production hybrid-TQ `[N,hidden]` layer.
//! 3. `forward_decode_body_batched` — embed + layer loop, wired into pass-1.

use anyhow::Result;
use mlx_native::graph::GraphSession;
use mlx_native::{DType, GraphExecutor, KernelRegistry, MlxBuffer, MlxDevice};

use crate::debug::INVESTIGATION_ENV;
use crate::quantize::imatrix::ImatrixHint;
use crate::serve::config::LayerType;
use crate::serve::forward_mlx_shared::{
    dispatch_qmatmul, dispatch_rms_norm_unit_perhead, RmsNormPerHeadArgs,
};
use mlx_native::ops::encode_helpers::{encode_with_args, KernelArg};

use super::batched_head::BatchedHeadOut;
use super::kv_cache::{MultiSeqHbKvBuffers, MultiSeqHybridKvBuffers};
use super::model::{MlxActivationBuffers, MlxModelWeights};
use crate::serve::gpu::GpuContext;
use crate::serve::multi_seq_kv::SlotId;

/// ADR-040 M-SPEED-LC Stage 2 — typed KV-scaffold selector for the batched
/// decode body's per-layer attention phase.
///
/// Exactly one production KV regime is active per `HF2Q_HYBRID_KV`: the
/// hybrid F16-K + TQ-HB-V scaffold (`HF2Q_HYBRID_KV=1`, DEFAULT) or the
/// full-TQ byte-packed K+V scaffold (`HF2Q_HYBRID_KV=0`, opt-in). The HB
/// scaffold (`MultiSeqHbKvBuffers`) is ALWAYS provisioned at spawn
/// (engine.rs, H94) regardless of which regime is selected, so `FullTq` is
/// always constructible when `Hybrid` is unavailable — a typed enum (not
/// two `Option`s) makes "exactly one regime, never both, never neither"
/// a compile-time invariant instead of a runtime `unwrap()`.
pub(crate) enum BatchedKvRegime<'a> {
    Hybrid(&'a mut [MultiSeqHybridKvBuffers]),
    FullTq(&'a mut [MultiSeqHbKvBuffers]),
}

/// `[N,...]` mirror of the production decode activation scratch. Each buffer is
/// sized `N × (the scalar buffer's element count)` — read straight from the
/// model's existing `MlxActivationBuffers` so the per-token/per-head dimensions
/// are exactly the proven scalar sizes (largest-layer-sized), never re-derived.
/// Row-major: row `i` (slot `i`) of buffer `b` occupies `b[i*stride..(i+1)*stride]`
/// where `stride` is the scalar buffer's element count.
pub struct BatchedDecodeBuffers {
    /// Batch width (number of slots this scratch is sized for).
    pub n: usize,
    /// `[N, hidden]` residual-stream hidden state (the body's input + output).
    pub hidden: MlxBuffer,
    /// `[N, hidden]` RMS-norm output (pre-attn / pre-FF / final norms reuse it).
    pub norm_out: MlxBuffer,
    /// `[N, num_heads*head_dim]` Q projection.
    pub attn_q: MlxBuffer,
    /// `[N, num_kv_heads*head_dim]` K projection.
    pub attn_k: MlxBuffer,
    /// `[N, num_kv_heads*head_dim]` V projection.
    pub attn_v: MlxBuffer,
    /// `[N, num_kv_heads*head_dim]` V after per-head norm (the `!v_is_k` case —
    /// sliding gemma4 layers with a separate v_proj; mirrors scalar
    /// `moe_expert_out` as the V-norm output / KV-encode source).
    pub attn_v_normed: MlxBuffer,
    /// `[N, num_heads*head_dim]` Q after per-head norm + RoPE.
    pub attn_q_normed: MlxBuffer,
    /// `[N, num_kv_heads*head_dim]` K after per-head norm + RoPE.
    pub attn_k_normed: MlxBuffer,
    /// `[N, num_heads*head_dim]` SDPA output (one row per slot's attention).
    pub sdpa_out: MlxBuffer,
    /// `[N, hidden]` O-projection output.
    pub attn_out: MlxBuffer,
    /// `[N, intermediate]` dense-MLP gate.
    pub mlp_gate: MlxBuffer,
    /// `[N, intermediate]` dense-MLP up.
    pub mlp_up: MlxBuffer,
    /// `[N, max(intermediate, moe_intermediate)]` fused SwiGLU scratch.
    pub mlp_fused: MlxBuffer,
    /// `[N, hidden]` dense-MLP down output.
    pub mlp_down: MlxBuffer,
    /// `[N, hidden]` residual scratch.
    pub residual: MlxBuffer,
    /// `[N, hidden]` MoE router-input norm.
    pub moe_norm_out: MlxBuffer,
    /// `[N, hidden]` router norm (concurrent with pre-FF norm).
    pub router_norm_out: MlxBuffer,
    /// `[N, num_experts]` router logits.
    pub moe_router_logits: MlxBuffer,
    /// `[N, top_k]` selected expert ids (U32).
    pub moe_expert_ids: MlxBuffer,
    /// `[N, top_k]` pre-scaled routing weights.
    pub moe_routing_weights_gpu: MlxBuffer,
    /// `[N, top_k, 2*moe_intermediate]` gate_up `_id` output.
    pub moe_gate_up_id_out: MlxBuffer,
    /// `[N, top_k, moe_intermediate]` SwiGLU `_id` output.
    pub moe_swiglu_id_out: MlxBuffer,
    /// `[N, top_k, hidden]` down `_id` output.
    pub moe_down_id_out: MlxBuffer,
    /// `[N, hidden]` MoE accumulator (weighted sum of top_k expert outputs).
    /// Dedicated buffer mirroring scalar `self.activations.moe_accum` — never
    /// alias `norm_out` here (an untracked weighted_sum write into a buffer the
    /// conflict tracker last saw read by B9 produced stale post-FF-norm2 input).
    pub moe_accum: MlxBuffer,
    /// `[N, tmp]` per-query flash reduce scratch for the M4 BATCHED flash
    /// (`HF2Q_BATCHED_FLASH=1`) — sized `N ×` the scalar `sdpa_tmp` so all N
    /// queries' NWG partials + S/M land in disjoint regions. Unused by the
    /// default per-slot flash.
    pub sdpa_tmp: MlxBuffer,
}

/// Element count of an F32/U32 buffer (4 bytes/element).
fn elems(b: &MlxBuffer) -> usize {
    b.byte_len() / 4
}

impl BatchedDecodeBuffers {
    /// Allocate `[N,...]` scratch sized `N ×` each scalar buffer. `acts` is the
    /// model's live `MlxActivationBuffers` (the proven scalar sizes).
    pub fn new(device: &MlxDevice, acts: &MlxActivationBuffers, n: usize) -> Result<Self> {
        let f32n = |scalar: &MlxBuffer, name: &str| -> Result<MlxBuffer> {
            let count = elems(scalar) * n;
            device
                .alloc_buffer(count * 4, DType::F32, vec![count])
                .map_err(|e| {
                    anyhow::anyhow!("BatchedDecodeBuffers alloc {name} ({count} f32): {e}")
                })
        };
        let u32n = |scalar: &MlxBuffer, name: &str| -> Result<MlxBuffer> {
            let count = elems(scalar) * n;
            device
                .alloc_buffer(count * 4, DType::U32, vec![count])
                .map_err(|e| {
                    anyhow::anyhow!("BatchedDecodeBuffers alloc {name} ({count} u32): {e}")
                })
        };
        Ok(Self {
            n,
            hidden: f32n(&acts.hidden, "hidden")?,
            norm_out: f32n(&acts.norm_out, "norm_out")?,
            attn_q: f32n(&acts.attn_q, "attn_q")?,
            attn_k: f32n(&acts.attn_k, "attn_k")?,
            attn_v: f32n(&acts.attn_v, "attn_v")?,
            attn_v_normed: f32n(&acts.attn_v, "attn_v_normed")?,
            attn_q_normed: f32n(&acts.attn_q_normed, "attn_q_normed")?,
            attn_k_normed: f32n(&acts.attn_k_normed, "attn_k_normed")?,
            sdpa_out: f32n(&acts.sdpa_out, "sdpa_out")?,
            attn_out: f32n(&acts.attn_out, "attn_out")?,
            mlp_gate: f32n(&acts.mlp_gate, "mlp_gate")?,
            mlp_up: f32n(&acts.mlp_up, "mlp_up")?,
            mlp_fused: f32n(&acts.mlp_fused, "mlp_fused")?,
            mlp_down: f32n(&acts.mlp_down, "mlp_down")?,
            residual: f32n(&acts.residual, "residual")?,
            moe_norm_out: f32n(&acts.moe_norm_out, "moe_norm_out")?,
            router_norm_out: f32n(&acts.router_norm_out, "router_norm_out")?,
            moe_router_logits: f32n(&acts.moe_router_logits, "moe_router_logits")?,
            moe_expert_ids: u32n(&acts.moe_expert_ids, "moe_expert_ids")?,
            moe_routing_weights_gpu: f32n(
                &acts.moe_routing_weights_gpu,
                "moe_routing_weights_gpu",
            )?,
            moe_gate_up_id_out: f32n(&acts.moe_gate_up_id_out, "moe_gate_up_id_out")?,
            moe_swiglu_id_out: f32n(&acts.moe_swiglu_id_out, "moe_swiglu_id_out")?,
            moe_down_id_out: f32n(&acts.moe_down_id_out, "moe_down_id_out")?,
            moe_accum: f32n(&acts.moe_accum, "moe_accum")?,
            sdpa_tmp: f32n(&acts.sdpa_tmp, "sdpa_tmp")?,
        })
    }

    /// Per-slot stride (element count of one row) for buffer family `hidden`.
    #[inline]
    pub fn hidden_stride(&self) -> usize {
        elems(&self.hidden) / self.n
    }
}

/// Byte offset of slot `i`'s row in an `[N, stride]` buffer (f32/u32, 4 bytes).
#[inline]
fn row_off(stride: usize, i: usize) -> u64 {
    (i * stride * 4) as u64
}

/// ADR-040 iter-G decode-category split (`HF2Q_DECODE_CATSPLIT=1`).
///
/// MEASUREMENT-ONLY. When the env gate is OFF (the default), `cat_split` is a
/// no-op — the batched body runs in ONE session committed at `s.finish()`,
/// BYTE-UNCHANGED from HEAD. When ON, the body session is committed at each
/// category boundary via `finish_with_gpu_time()` (the real
/// `GPUEndTime-GPUStartTime` interval — no inserted CPU-busy-wait pollutes the
/// measurement, only the extra commits serialize what was one async CB) and the
/// GPU-busy ns is accumulated into the per-category bucket below; a fresh
/// session is then begun. The probe reads + reports these buckets per token.
///
/// The split serializes CBs that production runs as one pipelined CB, so the
/// SUM of buckets OVERSTATES the real GPU-busy step (no inter-CB overlap). The
/// probe therefore also reports total throughput with CATSPLIT on vs off so the
/// overhead is visible and the split is trusted as a RELATIVE ranking, not an
/// absolute step time.
pub(crate) mod catsplit {
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Category index. Order = report order; `LEN` sizes the bucket arrays.
    #[derive(Clone, Copy)]
    pub enum Cat {
        Embed = 0,
        RmsNorm = 1,
        DenseQ = 2,
        DenseK = 3,
        DenseV = 4,
        DenseO = 5,
        AttnPre = 6,
        AttnFlash = 7,
        DenseFfn = 8,
        MoeGateUp = 9,
        MoeDown = 10,
        MoeOther = 11,
    }
    pub const LEN: usize = 12;
    // Consumed only by the engine.rs bench/profiling tests (cfg(test));
    // dead in the production bin build by design.
    #[allow(dead_code)]
    pub const NAMES: [&str; LEN] = [
        "embed",
        "rms_norm",
        "dense_q_proj",
        "dense_k_proj",
        "dense_v_proj",
        "dense_o_proj",
        "attn_pre(qkv_norm_rope+vnorm+fwht)",
        "attention(kv_enc+flash)",
        "dense_ffn(gate+up+down+router)",
        "moe_gate_up(mv_id)",
        "moe_down(mv_id)",
        "moe_other(swiglu+routing+wsum+endlayer)",
    ];

    // One ns accumulator + one CB-count accumulator per category.
    macro_rules! buckets {
        ($($name:ident),*) => {
            $(static $name: [AtomicU64; LEN] = [
                AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
                AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
                AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
            ];)*
        };
    }
    buckets!(NS, CBS, DISP);

    /// Running dispatch_count() at the last boundary, for per-category deltas.
    static LAST_DISP: AtomicU64 = AtomicU64::new(0);

    pub static ENABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("HF2Q_DECODE_CATSPLIT").as_deref() == Ok("1"));

    /// Add `ns` GPU-busy time (and one CB) to category `c`, plus the
    /// dispatch_count() delta since the previous boundary (the dispatches just
    /// encoded into the category `c` command buffer).
    #[inline]
    pub fn add(c: Cat, ns: u64) {
        NS[c as usize].fetch_add(ns, Ordering::Relaxed);
        CBS[c as usize].fetch_add(1, Ordering::Relaxed);
        let cur = mlx_native::dispatch_count();
        let last = LAST_DISP.swap(cur, Ordering::Relaxed);
        DISP[c as usize].fetch_add(cur.saturating_sub(last), Ordering::Relaxed);
    }

    /// Snapshot `(name, total_ns, cb_count, dispatch_count)` per category.
    // Consumed only by the engine.rs bench/profiling tests (cfg(test)).
    #[allow(dead_code)]
    pub fn snapshot() -> Vec<(&'static str, u64, u64, u64)> {
        (0..LEN)
            .map(|i| {
                (
                    NAMES[i],
                    NS[i].load(Ordering::Relaxed),
                    CBS[i].load(Ordering::Relaxed),
                    DISP[i].load(Ordering::Relaxed),
                )
            })
            .collect()
    }

    /// Reset all buckets (called once before the timed decode).
    // Consumed only by the engine.rs bench/profiling tests (cfg(test)).
    #[allow(dead_code)]
    pub fn reset() {
        for i in 0..LEN {
            NS[i].store(0, Ordering::Relaxed);
            CBS[i].store(0, Ordering::Relaxed);
            DISP[i].store(0, Ordering::Relaxed);
        }
        LAST_DISP.store(mlx_native::dispatch_count(), Ordering::Relaxed);
    }
}

/// ADR-040 §22 host-phase timing (`HF2Q_HOST_PHASES=1`).
///
/// Wall-clock `Instant` accumulators for the NON-GPU per-step host phases that
/// fill the ~8.3ms/step GPU-idle window (wall 35.9 − GPU-busy 27.6). The
/// existing catsplit/GPU_BUSY timers only see GPU exec time; these capture where
/// the host blocks/works between the two per-step `commit_and_wait` syncs:
/// the two GPU waits, the two host readbacks (`to_vec`), the Pass-2 sample loop,
/// and the pre-forward gather/mount-clear. OFF by default (zero overhead).
pub(crate) mod host_phases {
    use std::sync::atomic::{AtomicU64, Ordering};

    #[derive(Clone, Copy)]
    pub enum Phase {
        BodyWait = 0,          // commit_and_wait on the 30-layer body (host idle on GPU)
        BodyReadback = 1,      // hidden [N,hidden] GPU->host to_vec
        LmheadWait = 2,        // commit_and_wait on lm_head (host idle on GPU)
        LmheadReadback = 3,    // logits+normed [N,vocab] GPU->host to_vec
        SampleLoop = 4,        // Pass-2 host: 8x argmax + finalize + detok + scheduler
        GatherMisc = 5,        // pre-forward mount-clear + per-slot token/pos gather
        SchedStep = 6,         // scheduler.step() (worker loop, outside decode_batch)
        Publish = 7,           // publish(scheduler stats) mutex (worker loop)
        ArgmaxFinalize = 8,    // 8x argmax_f32 + finalize_token_from_logits (critical path)
        DecodeTick = 9, // 8x decode_tick_finalize (detok + EOS + stop + emit + sched-advance)
        DecodeBatchTotal = 10, // whole decode_batch_gemma4 call (setup+gather+body+head+sample)
        WorkerIter = 11, // whole worker-loop iteration (admit+sched.step+decode+publish)
    }
    pub const LEN: usize = 12;
    // Consumed only by the engine.rs bench/profiling tests (cfg(test));
    // dead in the production bin build by design.
    #[allow(dead_code)]
    pub const NAMES: [&str; LEN] = [
        "body_wait(sync)",
        "body_readback(hidden)",
        "lmhead_wait(sync)",
        "lmhead_readback(logits)",
        "sample_loop(argmax+detok+sched)",
        "gather+mount_clear",
        "scheduler.step()",
        "publish(stats)",
        "  argmax+finalize",
        "  decode_tick(detok+sched)",
        "decode_batch_TOTAL",
        "worker_iter_TOTAL",
    ];

    static NS: [AtomicU64; LEN] = [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ];

    pub static ENABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("HF2Q_HOST_PHASES").as_deref() == Ok("1"));

    #[inline]
    pub fn add(p: Phase, ns: u64) {
        if *ENABLED {
            NS[p as usize].fetch_add(ns, Ordering::Relaxed);
        }
    }

    // Consumed only by the engine.rs bench/profiling tests (cfg(test)).
    #[allow(dead_code)]
    pub fn snapshot() -> Vec<(&'static str, u64)> {
        (0..LEN)
            .map(|i| (NAMES[i], NS[i].load(Ordering::Relaxed)))
            .collect()
    }

    // Consumed only by the engine.rs bench/profiling tests (cfg(test)).
    #[allow(dead_code)]
    pub fn reset() {
        for i in 0..LEN {
            NS[i].store(0, Ordering::Relaxed);
        }
    }
}

/// ADR-040 iter-G category boundary (`HF2Q_DECODE_CATSPLIT=1`).
///
/// When the gate is ON: finish the current session with `finish_with_gpu_time`
/// (real GPU `GPUEndTime-GPUStartTime`), attribute that GPU-busy interval to the
/// category of the work JUST encoded into it (`c`), and replace `*session` with
/// a fresh one so the NEXT category's dispatches land in a clean CB.
///
/// When the gate is OFF: NO-OP. `*session` is untouched, so the body keeps its
/// single async session — byte-for-byte the production path.
///
/// `c` labels the work that was encoded SINCE the previous boundary (or session
/// begin). Place a call AFTER each category's last dispatch.
#[inline]
fn cat_boundary<'a>(
    session: &mut GraphSession<'a>,
    exec: &'a GraphExecutor,
    c: catsplit::Cat,
) -> Result<()> {
    if !*catsplit::ENABLED {
        return Ok(());
    }
    // Replace the live session with a fresh one; finish the old one and read its
    // GPU-busy interval. `exec.begin()` is the same call the body uses to open
    // its session — buffers are owned by `bufs`/the model, not the session, so
    // they persist across the boundary. Each finish is a full GPU sync, so all
    // prior writes are complete + host-visible before the next category begins.
    let fresh = exec
        .begin()
        .map_err(|e| anyhow::anyhow!("catsplit re-begin: {e}"))?;
    let old = std::mem::replace(session, fresh);
    let ns = old
        .finish_with_gpu_time()
        .map_err(|e| anyhow::anyhow!("catsplit finish: {e}"))?;
    catsplit::add(c, ns);
    Ok(())
}

/// Dense `input[N,in_dim] · weightᵀ → output[N,out_dim]`, BYTE-IDENTICAL to N
/// serial m=1 decode matmuls.
///
/// `dispatch_qmatmul` routes a QUANTIZED weight to `kernel_mul_mv` (per-row
/// bit-identical to m=1) only while `m <= MM_ROUTING_THRESHOLD`; above it, and
/// for F32 (router `ffn_gate_inp`) / F16 (`ffn_down`, intermediate=2112) weights
/// at ANY m>1, it routes to a TILE kernel (`mul_mm` / `dense_matmul_*_tensor` /
/// `mm_v2_f16`) whose reduction order differs from the m=1 matvec the serial
/// slot-aware reference uses. That tile path is mathematically equal but NOT
/// byte-identical, so a batched body that used it would diverge from N serial
/// decodes (the slot_aware_n4 bar). Batch only when the m=N path is the SAME
/// per-row matvec; otherwise loop m=1 per row (correct, weight re-read N times —
/// these are the minority dense projections; the dominant MoE experts batch
/// bit-identically via `quantized_matmul_id_ggml`).
#[allow(clippy::too_many_arguments)]
fn dispatch_dense_rowident(
    session: &mut GraphSession<'_>,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    input: &MlxBuffer,
    weight: &crate::serve::forward_mlx_shared::MlxQWeight,
    output: &MlxBuffer,
    n: usize,
    in_stride: usize,
    out_stride: usize,
    tag: &'static str,
    layer_idx: usize,
) -> Result<()> {
    use mlx_native::GgmlType;
    // Measured (ADR-040 M4, 2026-06-24): for F32/F16 weights at decode N≤8 the
    // per-row m=1 matvec loop is FASTER than the m=N tile path (151.6 vs 138.5
    // tok/s @ N=4) — the 8×8 SIMD tile wastes most rows at small m, while the
    // m=1 matvec is bandwidth-optimal. So the byte-identical per-row loop is also
    // the throughput-optimal choice; no batched F16/F32 matvec kernel is needed.
    let batched_rowident = !matches!(weight.info.ggml_dtype, GgmlType::F32 | GgmlType::F16)
        && (n as u32) <= mlx_native::ops::quantized_matmul_ggml::MM_ROUTING_THRESHOLD;
    if batched_rowident {
        dispatch_qmatmul(
            session,
            reg,
            dev,
            input,
            weight,
            output,
            n as u32,
            ImatrixHint::Layered {
                tag,
                layer: layer_idx,
            },
        )
    } else {
        // Measured (ADR-040 M4): amortizing the F16/F32 weight read across rows
        // (dispatch row 0 only) was throughput-NEUTRAL (152.3 vs 151.8 @ N=4) —
        // these reads are L2/latency-bound, not bandwidth-bound, so a batched
        // F16/F32 mat-VEC kernel would NOT help. The per-row m=1 loop stays.
        for i in 0..n {
            let in_i = input.slice_view(row_off(in_stride, i), in_stride);
            let out_i = output.slice_view(row_off(out_stride, i), out_stride);
            dispatch_qmatmul(
                session,
                reg,
                dev,
                &in_i,
                weight,
                &out_i,
                1,
                ImatrixHint::Layered {
                    tag,
                    layer: layer_idx,
                },
            )?;
        }
        Ok(())
    }
}

impl MlxModelWeights {
    /// ADR-040 S2/S3 — one `[N,hidden]` decode layer (production hybrid-TQ path).
    ///
    /// Position-INDEPENDENT ops (input-norm, post-attn norm, fused MLP, MoE) run
    /// BATCHED on the full `[N,...]` buffers (rows=N / n_tokens=N — bit-identical
    /// per H-S1-rowparity + H-S2-tokenparity). Dense projections route through
    /// [`dispatch_dense_rowident`] (m=N `mul_mv` for quantized n≤8, else per-row
    /// m=1) so they are byte-identical to the serial m=1 decode. Position-
    /// DEPENDENT ops (Q/K norm+RoPE, V-norm, hybrid KV-encode,
    /// `flash_attn_vec_hybrid`) loop per-slot over `slice_view` row-views at the
    /// ACTUAL per-layer dims (gemma4 global head_dim=512 vs sliding=256), running
    /// the EXACT scalar ops — bit-identical by reuse.
    ///
    /// Proven byte-identical to N serial slot-aware decodes by `slot_aware_n1`
    /// (N=1) and `slot_aware_n4` (N=4 concurrent per-slot parity) at the
    /// production default. Opt-in via `HF2Q_BATCHED_BODY=1`.
    pub(crate) fn encode_one_layer_batched<'a>(
        &self,
        layer_idx: usize,
        bufs: &BatchedDecodeBuffers,
        n: usize,
        positions_buf: &MlxBuffer,
        slot_id_buf: &MlxBuffer,
        slot_ids: &[SlotId],
        seq_positions: &[usize],
        regime: &mut BatchedKvRegime<'_>,
        tq_scale_factor_d512: f32,
        tq_codebook_bits: u32,
        session: &mut GraphSession<'a>,
        exec: &'a GraphExecutor,
        reg: &mut KernelRegistry,
    ) -> Result<()> {
        let dev = exec.device();
        let metal_dev = dev.metal_device();
        let hs = self.hidden_size;
        let nu = n as u32;
        let hd = self.layers[layer_idx].head_dim;
        let nkv = self.layers[layer_idx].num_kv_heads;
        let nh = self.num_attention_heads;
        let is_sliding = self.layers[layer_idx].layer_type == LayerType::Sliding;
        let eps = self.rms_norm_eps;
        // Per-ROW strides for the attention buffers MUST be the ACTUAL
        // per-layer dims (nh*hd, nkv*hd), NOT elems(buf)/n. Those buffers are
        // allocated at the MAX layer size (num_heads*max_hd / max_kv_heads*max_hd
        // — gemma4 global layers use head_dim=512/kv_heads=2; sliding layers use
        // head_dim=256), but the batched qmatmul (m=N) packs rows CONTIGUOUSLY by
        // the actual output dim. Using elems/n (the max stride) misaligns rows>0
        // on sliding layers → garbage at N>1 while N=1 (row 0 at offset 0) passes.
        let q_stride = nh * hd;
        let k_stride = nkv * hd;
        let v_stride = nkv * hd;

        // -- Pre-attention RMS norm (BATCHED rows=N): hidden -> norm_out --
        // norm_params is [eps, hs], per-element and identical for every row.
        session.barrier_between(
            &[&bufs.hidden, &self.layers[layer_idx].norms.input_layernorm],
            &[&bufs.norm_out],
        );
        session
            .rms_norm(
                reg,
                metal_dev,
                &bufs.hidden,
                &self.layers[layer_idx].norms.input_layernorm,
                &bufs.norm_out,
                &self.activations.norm_params,
                nu,
                hs as u32,
            )
            .map_err(|e| anyhow::anyhow!("batched pre-attn norm L{layer_idx}: {e}"))?;
        cat_boundary(session, exec, catsplit::Cat::RmsNorm)?;

        // -- QKV projections (BATCHED m=N): all read norm_out, write disjoint --
        session.barrier_between(
            &[&bufs.norm_out],
            &[&bufs.attn_q, &bufs.attn_k, &bufs.attn_v],
        );
        dispatch_dense_rowident(
            session,
            reg,
            dev,
            &bufs.norm_out,
            &self.layers[layer_idx].attn.q_proj,
            &bufs.attn_q,
            n,
            hs,
            q_stride,
            "attn_q",
            layer_idx,
        )?;
        cat_boundary(session, exec, catsplit::Cat::DenseQ)?;
        dispatch_dense_rowident(
            session,
            reg,
            dev,
            &bufs.norm_out,
            &self.layers[layer_idx].attn.k_proj,
            &bufs.attn_k,
            n,
            hs,
            k_stride,
            "attn_k",
            layer_idx,
        )?;
        cat_boundary(session, exec, catsplit::Cat::DenseK)?;
        let v_is_k = self.layers[layer_idx].attn.v_proj.is_none();
        if !v_is_k {
            dispatch_dense_rowident(
                session,
                reg,
                dev,
                &bufs.norm_out,
                self.layers[layer_idx].attn.v_proj.as_ref().unwrap(),
                &bufs.attn_v,
                n,
                hs,
                v_stride,
                "attn_v",
                layer_idx,
            )?;
            cat_boundary(session, exec, catsplit::Cat::DenseV)?;
        }

        // -- Per-head RMS norm + RoPE on Q and K (PER-SLOT row-views) --
        // Position-dependent: each slot's row uses its own position. Mirrors the
        // scalar fused_head_norm_rope (gpu_full_attn.rs:150-170) per slot.
        let half_rope = (hd / 2) as u32;
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
        // ADR-040 M4 — HF2Q_BATCHED_ATTNPRE: fuse the per-slot Q/K norm+RoPE,
        // V-norm, and FWHT-undo loops into single grid-dim-N dispatches (the
        // kernels already index a flat array of heads/rows: base=head_id*dim,
        // pos=positions[head_id/n_heads] — widening the grid to N*heads /
        // rows=N*nkv processes all N queries query-major ⇒ per-row BIT-IDENTICAL
        // to the per-slot loops; N=1 reduces to the single dispatch).
        // ADR-040 iter-F-batched-default: DEFAULT-ON (opt out HF2Q_BATCHED_ATTNPRE=0);
        // per-row bit-identical to the per-slot loop (proven by n4/n8 parity).
        let attnpre = std::env::var("HF2Q_BATCHED_ATTNPRE").as_deref() != Ok("0");
        session.barrier_between(
            &[&bufs.attn_q, &bufs.attn_k],
            &[&bufs.attn_q_normed, &bufs.attn_k_normed],
        );
        if attnpre {
            // ONE batched Q + ONE batched K norm+RoPE over all N queries (grid
            // N*nh / N*nkv; full positions_buf [N] — kernel reads each query's
            // pos via seq_idx=head_id/n_heads). Buffers are tight [N, heads*hd].
            mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32_batched(
                session.encoder_mut(),
                reg,
                metal_dev,
                &bufs.attn_q,
                &bufs.attn_q_normed,
                Some(&self.layers[layer_idx].attn.q_norm_weight),
                positions_buf,
                ff_gpu,
                n as u32,
                nh as u32,
                hd as u32,
                half_rope,
                eps,
                theta,
            )
            .map_err(|e| anyhow::anyhow!("batched-attnpre Q norm+RoPE L{layer_idx}: {e}"))?;
            mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32_batched(
                session.encoder_mut(),
                reg,
                metal_dev,
                &bufs.attn_k,
                &bufs.attn_k_normed,
                Some(&self.layers[layer_idx].attn.k_norm_weight),
                positions_buf,
                ff_gpu,
                n as u32,
                nkv as u32,
                hd as u32,
                half_rope,
                eps,
                theta,
            )
            .map_err(|e| anyhow::anyhow!("batched-attnpre K norm+RoPE L{layer_idx}: {e}"))?;
        } else {
            for i in 0..n {
                let pos_i = positions_buf.slice_view((i * 4) as u64, 1);
                let q_in = bufs.attn_q.slice_view(row_off(q_stride, i), q_stride);
                let q_out = bufs
                    .attn_q_normed
                    .slice_view(row_off(q_stride, i), q_stride);
                mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32(
                    session.encoder_mut(),
                    reg,
                    metal_dev,
                    &q_in,
                    &q_out,
                    Some(&self.layers[layer_idx].attn.q_norm_weight),
                    &pos_i,
                    ff_gpu,
                    nh as u32,
                    hd as u32,
                    half_rope,
                    eps,
                    theta,
                )
                .map_err(|e| anyhow::anyhow!("batched Q norm+RoPE L{layer_idx} slot{i}: {e}"))?;
                let k_in = bufs.attn_k.slice_view(row_off(k_stride, i), k_stride);
                let k_out = bufs
                    .attn_k_normed
                    .slice_view(row_off(k_stride, i), k_stride);
                mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32(
                    session.encoder_mut(),
                    reg,
                    metal_dev,
                    &k_in,
                    &k_out,
                    Some(&self.layers[layer_idx].attn.k_norm_weight),
                    &pos_i,
                    ff_gpu,
                    nkv as u32,
                    hd as u32,
                    half_rope,
                    eps,
                    theta,
                )
                .map_err(|e| anyhow::anyhow!("batched K norm+RoPE L{layer_idx} slot{i}: {e}"))?;
            }
        }

        // -- V norm (PER-SLOT row-views) — mirrors scalar gpu_full_attn.rs:183-221.
        // v_is_k (V tied to K, full-attn k_eq_v layers): norm attn_k -> attn_v.
        // !v_is_k (separate v_proj, sliding layers): norm attn_v -> attn_v_normed.
        // The KV-encode V source is the NORMED buffer in BOTH cases (codex-found:
        // the missing !v_is_k branch quantized raw V → N=1 divergence).
        let hd_norm_params = if is_sliding {
            &self.activations.norm_params_sliding_hd
        } else {
            &self.activations.norm_params_global_hd
        };
        let v_normed_stride = nkv * hd; // actual per-layer dim (see q/k/v_stride note)
                                        // ATTNPRE: ONE V-norm over all N queries (rows = N*nkv; kernel base =
                                        // row*hd over the tight [N, nkv*hd] buffer ⇒ query-major, bit-identical).
        let (v_in_buf, v_out_buf) = if v_is_k {
            (&bufs.attn_k, &bufs.attn_v)
        } else {
            (&bufs.attn_v, &bufs.attn_v_normed)
        };
        session.barrier_between(&[v_in_buf], &[v_out_buf]);
        if attnpre {
            dispatch_rms_norm_unit_perhead(
                session.encoder_mut(),
                reg,
                metal_dev,
                &RmsNormPerHeadArgs {
                    input: v_in_buf,
                    output: v_out_buf,
                    params_buf: hd_norm_params,
                    rows: (n * nkv) as u32,
                    dim: hd as u32,
                },
            )?;
        } else if v_is_k {
            for i in 0..n {
                let vk_in = bufs.attn_k.slice_view(row_off(k_stride, i), k_stride);
                let v_out = bufs.attn_v.slice_view(row_off(v_stride, i), v_stride);
                dispatch_rms_norm_unit_perhead(
                    session.encoder_mut(),
                    reg,
                    metal_dev,
                    &RmsNormPerHeadArgs {
                        input: &vk_in,
                        output: &v_out,
                        params_buf: hd_norm_params,
                        rows: nkv as u32,
                        dim: hd as u32,
                    },
                )?;
            }
        } else {
            for i in 0..n {
                let vv_in = bufs.attn_v.slice_view(row_off(v_stride, i), v_stride);
                let v_out = bufs
                    .attn_v_normed
                    .slice_view(row_off(v_normed_stride, i), v_normed_stride);
                dispatch_rms_norm_unit_perhead(
                    session.encoder_mut(),
                    reg,
                    metal_dev,
                    &RmsNormPerHeadArgs {
                        input: &vv_in,
                        output: &v_out,
                        params_buf: hd_norm_params,
                        rows: nkv as u32,
                        dim: hd as u32,
                    },
                )?;
            }
        }
        // catsplit: Q/K head-norm+RoPE + V-norm (the pre-attention head ops).
        cat_boundary(session, exec, catsplit::Cat::AttnPre)?;

        // -- ATTENTION (PER-SLOT): hybrid KV-encode (F16-K copy + FWHT-V quant)
        // -> flash_attn_vec_hybrid -> fwht_sign_undo, each against this slot's
        // multi_seq_kv_hybrid[L] region (slice_view at slot byte offset). Mirrors
        // the scalar default hybrid path (gpu_full_attn.rs:416-494, 1180-1221,
        // FWHT-undo). The encode↔SDPA↔undo FWHT coherence is version-churned in
        // the scalar source; this build applies the undo, and the N=1 gate
        // (slot_aware_n1, bit-identical to serial) settles it — if N=1 diverges,
        // toggle the undo / params per the dumped divergence (RUNTIME PARITY,
        // not read-replication). --
        let q_norm_stride = nh * hd; // actual per-layer dim (see q/k/v_stride note)
        let k_norm_stride = nkv * hd;
        let sdpa_stride = nh * hd;
        // ADR-040 M-SPEED-LC Stage 2 — regime dispatch. The Hybrid arm below
        // is BYTE-FOR-BYTE the pre-existing body (only re-indented into the
        // match arm) — no dispatch reordering, no arithmetic changes. The
        // FullTq arm is NEW: same per-query addressing / same_bucket-gating
        // skeleton, byte-packed 5/6/8-bit TQ-HB K+V (mlx-native
        // `flash_attn_vec_tq_hb_batched`, ADR-040 M-SPEED-LC Stage 1) instead
        // of hybrid's F16-K + TQ-HB-V.
        match regime {
            BatchedKvRegime::Hybrid(multi_seq_kv_hybrid) => {
                // ADR-040 M4 — OPT-IN batched multi-seq flash (HF2Q_BATCHED_FLASH=1):
                // replace the N per-slot flash dispatches with ONE batched flash over all
                // N queries (GPU occupancy). KV-encode + FWHT-undo stay per-slot. Gated
                // to slots sharing the same (nwg, nsg) bucket so the per-query math is
                // bit-identical to the per-slot flash (proven by slot_aware_n1/n4).
                let gbuf = &multi_seq_kv_hybrid[layer_idx];
                let gcap = gbuf.capacity;
                let gring = gbuf.is_sliding;
                let ksl_of = |i: usize| -> u32 {
                    let sp = seq_positions[i];
                    if gring {
                        ((sp + 1).min(gcap)) as u32
                    } else {
                        (sp + 1) as u32
                    }
                };
                let nwg_bucket = |k: u32| if k > 512 { 32u32 } else { 16u32 };
                let max_ksl = (0..n).map(ksl_of).max().unwrap_or(1);
                let nsg_max = mlx_native::ops::flash_attn_vec_tq_hb::compute_nsg(max_ksl);
                let same_bucket = (0..n).all(|i| {
                    let k = ksl_of(i);
                    nwg_bucket(k) == nwg_bucket(max_ksl)
                        && mlx_native::ops::flash_attn_vec_tq_hb::compute_nsg(k) == nsg_max
                });
                // ADR-040 iter-F-batched-default: DEFAULT-ON (opt out HF2Q_BATCHED_FLASH=0),
                // gated to same-(nwg,nsg)-bucket slots so the per-query math is bit-identical
                // to the per-slot flash; mixed-bucket (e.g. staggered) slots fall back to the
                // per-slot loop below.
                let use_batched_flash =
                    std::env::var("HF2Q_BATCHED_FLASH").as_deref() != Ok("0") && same_bucket;
                // ROWDIFF probe (HF2Q_DECODE_TRACE): layer 0 only — dump per-slot
                // seq_positions / ksl / cache_pos so we can tell whether staggered slot-0
                // divergence is BOOKKEEPING (positions differ) or KV-CONTENT (identical).
                if layer_idx == 0 && std::env::var("HF2Q_DECODE_TRACE").is_ok() {
                    let dump: Vec<String> = (0..n)
                        .map(|i| {
                            let sp = seq_positions[i];
                            let cp = if gring { (sp % gcap) as u32 } else { sp as u32 };
                            format!("s{}:sp{} ksl{} cp{}", slot_ids[i].0, sp, ksl_of(i), cp)
                        })
                        .collect();
                    eprintln!(
                        "[ROWDIFF] L0 N={} max_ksl={} same_bucket={} bflash={} [{}]",
                        n,
                        max_ksl,
                        same_bucket,
                        use_batched_flash,
                        dump.join(" ")
                    );
                }
                if use_batched_flash {
                    let buf = gbuf;
                    let cap = gcap;
                    let is_ring = gring;
                    // PHASE 1 — KV-encode (F16-K + FWHT-V). Fuses the N per-slot
                    // dispatches into 2 grid-dim-N dispatches (one F16-K, one FWHT-V over
                    // all N queries) — bit-identical (per-query slot/pos addressing
                    // in-kernel). Falls back per-slot for the dummy-vnorms (FULL_F16_KV)
                    // case. ADR-040 M4. DEFAULT-ON (2026-06-27): measured +3.5% N=8 decode
                    // throughput + −19% dispatches/step (2147→1731); byte-parity GREEN
                    // (slot_aware_n8_per_slot_parity_vs_serial). Opt out: HF2Q_BATCHED_KVENC=0.
                    let v_src_buf: &MlxBuffer = if v_is_k {
                        &bufs.attn_v
                    } else {
                        &bufs.attn_v_normed
                    };
                    let vnorms_dummy = buf.v_norms.byte_len() == 4;
                    let use_batched_kvenc =
                        std::env::var("HF2Q_BATCHED_KVENC").as_deref() != Ok("0") && !vnorms_dummy;
                    if use_batched_kvenc {
                        // ONE barrier: norm-rope/V-norm wrote attn_k_normed / v_src;
                        // declare them as reads for the 2 batched encode dispatches.
                        session.barrier_between(
                            &[&bufs.attn_k_normed, v_src_buf],
                            &[&buf.k, &buf.v_packed, &buf.v_norms],
                        );
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16_batched(
                    session.encoder_mut(), reg, metal_dev,
                    &bufs.attn_k_normed, &buf.k, slot_id_buf, positions_buf,
                    n as u32, nkv as u32, hd as u32, cap as u32, is_ring,
                ).map_err(|e| anyhow::anyhow!("bf F16-K batched L{layer_idx}: {e}"))?;
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb_batched(
                    session.encoder_mut(), reg, metal_dev,
                    v_src_buf, &buf.v_packed, &buf.v_norms, slot_id_buf, positions_buf,
                    n as u32, nkv as u32, hd as u32, cap as u32, is_ring,
                    tq_scale_factor_d512, tq_codebook_bits,
                ).map_err(|e| anyhow::anyhow!("bf FWHT-V batched L{layer_idx}: {e}"))?;
                    } else {
                        for i in 0..n {
                            let slot = slot_ids[i].0 as u64;
                            let seq_pos_i = seq_positions[i];
                            let cache_pos: u32 = if is_ring {
                                (seq_pos_i % cap) as u32
                            } else {
                                seq_pos_i as u32
                            };
                            let k_elems = nkv * cap * hd;
                            let v_dtype_size = buf.v_packed.dtype().size_of() as u64;
                            let k_view = buf
                                .k
                                .slice_view(slot * (k_elems as u64) * 2, k_elems)
                                .with_shape(vec![nkv, cap, hd])
                                .map_err(|e| anyhow::anyhow!("bf K L{layer_idx} s{i}: {e}"))?;
                            let v_view = buf
                                .v_packed
                                .slice_view(slot * (k_elems as u64) * v_dtype_size, k_elems)
                                .with_shape(vec![nkv, cap, hd])
                                .map_err(|e| anyhow::anyhow!("bf V L{layer_idx} s{i}: {e}"))?;
                            let norms_per_pos = buf.norms_per_pos;
                            let v_norms_view = if vnorms_dummy {
                                buf.v_norms
                                    .slice_view(0, 1)
                                    .with_shape(vec![1])
                                    .map_err(|e| anyhow::anyhow!("bf Vn dummy: {e}"))?
                            } else {
                                let ne = nkv * cap * norms_per_pos;
                                let shp = if norms_per_pos == 1 {
                                    vec![nkv, cap]
                                } else {
                                    vec![nkv, cap, norms_per_pos]
                                };
                                buf.v_norms
                                    .slice_view(slot * (ne as u64) * 4, ne)
                                    .with_shape(shp)
                                    .map_err(|e| anyhow::anyhow!("bf Vn: {e}"))?
                            };
                            let kn_i = bufs
                                .attn_k_normed
                                .slice_view(row_off(k_norm_stride, i), k_norm_stride);
                            let v_i = if v_is_k {
                                bufs.attn_v.slice_view(row_off(v_stride, i), v_stride)
                            } else {
                                bufs.attn_v_normed
                                    .slice_view(row_off(v_normed_stride, i), v_normed_stride)
                            };
                            session.barrier_between(
                                &[&kn_i, v_src_buf],
                                &[&buf.k, &buf.v_packed, &buf.v_norms],
                            );
                            mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16(
                    session.encoder_mut(), reg, metal_dev, &kn_i, &k_view, nkv as u32, hd as u32, cap as u32, cache_pos,
                ).map_err(|e| anyhow::anyhow!("bf F16-K L{layer_idx} s{i}: {e}"))?;
                            mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
                    session.encoder_mut(), reg, metal_dev, &v_i, &v_view, &v_norms_view,
                    nkv as u32, hd as u32, cap as u32, cache_pos, is_ring, tq_scale_factor_d512, tq_codebook_bits,
                ).map_err(|e| anyhow::anyhow!("bf FWHT-V L{layer_idx} s{i}: {e}"))?;
                        }
                    }
                    // PHASE 2 — ONE batched flash over all N queries (per-query KV via
                    // slot_id_buf + positions_buf, derived in-kernel).
                    let p_hyb = mlx_native::ops::flash_attn_vec_hybrid::FlashAttnVecTqHbParams {
                        num_heads: nh as u32,
                        num_kv_heads: nkv as u32,
                        head_dim: hd as u32,
                        kv_seq_len: max_ksl,
                        kv_capacity: cap as u32,
                        scale: 1.0,
                        mask_type: if is_sliding { 2 } else { 1 },
                        sliding_window: if is_sliding {
                            self.sliding_window as u32
                        } else {
                            0
                        },
                        softcap: 0.0,
                        ring_start: 0,
                        scale_factor_d512: tq_scale_factor_d512,
                        codebook_bits: tq_codebook_bits,
                        fuse_fwht_pre: 0,
                        nsg: nsg_max,
                    };
                    session.barrier_between(
                        &[
                            &bufs.attn_q_normed,
                            &buf.k,
                            &buf.v_packed,
                            &buf.v_norms,
                            &bufs.sdpa_tmp,
                        ],
                        &[&bufs.sdpa_out, &bufs.sdpa_tmp],
                    );
                    mlx_native::ops::flash_attn_vec_hybrid::flash_attn_vec_hybrid_batched(
                        session.encoder_mut(),
                        reg,
                        dev,
                        n as u32,
                        &bufs.attn_q_normed,
                        &buf.k,
                        &buf.v_packed,
                        &buf.v_norms,
                        &bufs.sdpa_out,
                        &bufs.sdpa_tmp,
                        slot_id_buf,
                        positions_buf,
                        &p_hyb,
                    )
                    .map_err(|e| anyhow::anyhow!("batched flash L{layer_idx}: {e}"))?;
                    // PHASE 3 — FWHT-undo. ATTNPRE: ONE batched undo over all N queries
                    // (num_heads=N*nh; kernel base=head_idx*hd over tight [N, nh*hd] ⇒
                    // bit-identical). Else per-slot.
                    session.barrier_between(&[&bufs.sdpa_out], &[&bufs.sdpa_out]);
                    if attnpre {
                        mlx_native::ops::fwht_standalone::dispatch_fwht_sign_undo_f32(
                            session.encoder_mut(),
                            reg,
                            metal_dev,
                            &bufs.sdpa_out,
                            (n as u32) * nh as u32,
                            hd as u32,
                        )
                        .map_err(|e| anyhow::anyhow!("bf batched undo L{layer_idx}: {e}"))?;
                    } else {
                        for i in 0..n {
                            let sdpa_i = bufs
                                .sdpa_out
                                .slice_view(row_off(sdpa_stride, i), sdpa_stride);
                            session.barrier_between(&[&bufs.sdpa_out], &[&bufs.sdpa_out]);
                            mlx_native::ops::fwht_standalone::dispatch_fwht_sign_undo_f32(
                                session.encoder_mut(),
                                reg,
                                metal_dev,
                                &sdpa_i,
                                nh as u32,
                                hd as u32,
                            )
                            .map_err(|e| anyhow::anyhow!("bf undo L{layer_idx} s{i}: {e}"))?;
                        }
                    }
                } else {
                    // ADR-040 Phase F `iter-F-flashtmp` / `iter-F-batched-determinism-residual`
                    // (2026-06-25) — per-slot reduce scratch. The N×-sized `bufs.sdpa_tmp`
                    // (per-call, owned) replaces the shared `self.activations.sdpa_tmp` so
                    // each slot's flash reduces in its OWN disjoint region. The prior shared
                    // tmp relied on a WAW barrier for isolation, which held for a single
                    // stable body call (n4/n8 pass) but NOT under staggered continuous
                    // batching (changing N + mid-window admission/refill touch the same
                    // global activation scratch) — root cause of the ~2.5% batched-only
                    // staggered non-determinism (codex-confirmed; bisected: still flakes at
                    // FLASH=0/KVENC=0/ATTNPRE=0 because staggered slots fail `same_bucket`
                    // and fall back HERE even with FLASH=1).
                    let tmp_stride = elems(&bufs.sdpa_tmp) / n;
                    for i in 0..n {
                        let slot = slot_ids[i].0 as u64;
                        let seq_pos_i = seq_positions[i];
                        let buf = &multi_seq_kv_hybrid[layer_idx];
                        let cap = buf.capacity;
                        let is_ring = buf.is_sliding;
                        let cache_pos: u32 = if is_ring {
                            (seq_pos_i % cap) as u32
                        } else {
                            seq_pos_i as u32
                        };
                        // Per-slot KV views (offset math mirrors forward_prefill.rs:4636-4735).
                        let k_elems = nkv * cap * hd;
                        let v_dtype_size = buf.v_packed.dtype().size_of() as u64;
                        let k_view = buf
                            .k
                            .slice_view(slot * (k_elems as u64) * 2, k_elems)
                            .with_shape(vec![nkv, cap, hd])
                            .map_err(|e| {
                                anyhow::anyhow!("batched K slot-view L{layer_idx} s{i}: {e}")
                            })?;
                        let v_view = buf
                            .v_packed
                            .slice_view(slot * (k_elems as u64) * v_dtype_size, k_elems)
                            .with_shape(vec![nkv, cap, hd])
                            .map_err(|e| {
                                anyhow::anyhow!("batched V slot-view L{layer_idx} s{i}: {e}")
                            })?;
                        let norms_per_pos = buf.norms_per_pos;
                        let v_norms_view = if buf.v_norms.byte_len() == 4 {
                            buf.v_norms
                                .slice_view(0, 1)
                                .with_shape(vec![1])
                                .map_err(|e| {
                                    anyhow::anyhow!("batched Vnorms dummy L{layer_idx} s{i}: {e}")
                                })?
                        } else {
                            let ne = nkv * cap * norms_per_pos;
                            let shp = if norms_per_pos == 1 {
                                vec![nkv, cap]
                            } else {
                                vec![nkv, cap, norms_per_pos]
                            };
                            buf.v_norms
                                .slice_view(slot * (ne as u64) * 4, ne)
                                .with_shape(shp)
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "batched Vnorms slot-view L{layer_idx} s{i}: {e}"
                                    )
                                })?
                        };
                        // Row-views of the batched Q/K/V activations for this slot.
                        let q_i = bufs
                            .attn_q_normed
                            .slice_view(row_off(q_norm_stride, i), q_norm_stride);
                        let kn_i = bufs
                            .attn_k_normed
                            .slice_view(row_off(k_norm_stride, i), k_norm_stride);
                        // KV-encode V source: normed buffer (attn_v for v_is_k, else attn_v_normed).
                        let v_src_buf: &MlxBuffer = if v_is_k {
                            &bufs.attn_v
                        } else {
                            &bufs.attn_v_normed
                        };
                        let v_i = if v_is_k {
                            bufs.attn_v.slice_view(row_off(v_stride, i), v_stride)
                        } else {
                            bufs.attn_v_normed
                                .slice_view(row_off(v_normed_stride, i), v_normed_stride)
                        };
                        let sdpa_i = bufs
                            .sdpa_out
                            .slice_view(row_off(sdpa_stride, i), sdpa_stride);
                        // Per-slot reduce scratch (iter-F-flashtmp) — disjoint region of the
                        // N×-sized batched tmp; replaces the shared self.activations.sdpa_tmp.
                        let tmp_i = bufs.sdpa_tmp.slice_view(row_off(tmp_stride, i), tmp_stride);

                        // BARRIER (mirror scalar gpu_full_attn.rs:423): the Q/K norm+RoPE
                        // and V-norm dispatches above wrote attn_{q,k}_normed / v_src via
                        // raw `encoder_mut()` (untracked); the preceding `barrier_between`
                        // calls registered those buffers as WRITES in the conflict tracker.
                        // This barrier_between declares them as READS for the KV-encode →
                        // the tracker detects the RAW and emits the Metal memory_barrier so
                        // the F16-K copy / FWHT-V quant never read stale norm+RoPE output.
                        // (Omitting it produced garbage from the first decode token — the
                        // flash read pre-norm Q/K. ADR-040 S2/S3 root cause.)
                        session.barrier_between(
                            &[&bufs.attn_k_normed, v_src_buf],
                            &[&buf.k, &buf.v_packed, &buf.v_norms],
                        );
                        // F16-K copy: attn_k_normed -> hybrid K cache (F16).
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16(
                            session.encoder_mut(),
                            reg,
                            metal_dev,
                            &kn_i,
                            &k_view,
                            nkv as u32,
                            hd as u32,
                            cap as u32,
                            cache_pos,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("batched F16-K write L{layer_idx} s{i}: {e}")
                        })?;
                        // FWHT-V quantize: attn_v -> hybrid V (TQ-HB packed + norms).
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
                            session.encoder_mut(),
                            reg,
                            metal_dev,
                            &v_i,
                            &v_view,
                            &v_norms_view,
                            nkv as u32,
                            hd as u32,
                            cap as u32,
                            cache_pos,
                            is_ring,
                            tq_scale_factor_d512,
                            tq_codebook_bits,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("batched FWHT-V quant L{layer_idx} s{i}: {e}")
                        })?;

                        // SDPA: flash_attn_vec_hybrid (raw Q, F16-K, TQ-HB-V).
                        let kv_seq_len = if is_ring {
                            ((seq_pos_i + 1).min(cap)) as u32
                        } else {
                            (seq_pos_i + 1) as u32
                        };
                        let ring_start = if is_ring && kv_seq_len as usize >= cap {
                            ((seq_pos_i + 1) % cap) as u32
                        } else {
                            0u32
                        };
                        // BARRIER (mirror scalar gpu_full_attn.rs:1197): the F16-K copy /
                        // FWHT-V quant just wrote buf.k / buf.v_packed / buf.v_norms
                        // (untracked encoder dispatches; registered as writes by the
                        // barrier above). Flash reads attn_q_normed + the KV cache and
                        // writes sdpa_out — declare the RAW so the encode lands first.
                        // `tmp_i` is this slot's OWN disjoint reduce-scratch region of the
                        // N×-sized `bufs.sdpa_tmp` (iter-F-flashtmp). Each per-slot flash now
                        // reduces in isolation — no shared-scratch collision, so this is also
                        // genuinely per-slot-concurrent (the WAW barrier on the old shared
                        // `self.activations.sdpa_tmp` is no longer the isolation mechanism).
                        // The RAW barrier below still orders the KV-encode writes before the
                        // flash reads.
                        session.barrier_between(
                            &[
                                &bufs.attn_q_normed,
                                &buf.k,
                                &buf.v_packed,
                                &buf.v_norms,
                                &tmp_i,
                            ],
                            &[&bufs.sdpa_out, &tmp_i],
                        );
                        let p_hyb =
                            mlx_native::ops::flash_attn_vec_hybrid::FlashAttnVecTqHbParams {
                                num_heads: nh as u32,
                                num_kv_heads: nkv as u32,
                                head_dim: hd as u32,
                                kv_seq_len,
                                kv_capacity: cap as u32,
                                scale: 1.0,
                                mask_type: if is_sliding { 2 } else { 1 },
                                sliding_window: if is_sliding {
                                    self.sliding_window as u32
                                } else {
                                    0
                                },
                                softcap: 0.0,
                                ring_start,
                                scale_factor_d512: tq_scale_factor_d512,
                                codebook_bits: tq_codebook_bits,
                                fuse_fwht_pre: 0,
                                nsg: mlx_native::ops::flash_attn_vec_tq_hb::compute_nsg(kv_seq_len),
                            };
                        mlx_native::ops::flash_attn_vec_hybrid::flash_attn_vec_hybrid(
                            session.encoder_mut(),
                            reg,
                            dev,
                            &q_i,
                            &k_view,
                            &v_view,
                            &v_norms_view,
                            &sdpa_i,
                            &tmp_i,
                            &p_hyb,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("batched flash_attn_vec_hybrid L{layer_idx} s{i}: {e}")
                        })?;
                        // FWHT-undo (V was FWHT-rotated pre-quant ⇒ SDPA out in FWHT domain;
                        // mirrors scalar gpu_full_attn.rs:1350, applied for the TQ-HB-V regime).
                        // BARRIER (mirror scalar :1346): flash wrote sdpa_out (untracked;
                        // registered as write by the SDPA barrier above). FWHT-undo
                        // reads+writes sdpa_out in place → serialize.
                        session.barrier_between(&[&bufs.sdpa_out], &[&bufs.sdpa_out]);
                        mlx_native::ops::fwht_standalone::dispatch_fwht_sign_undo_f32(
                            session.encoder_mut(),
                            reg,
                            metal_dev,
                            &sdpa_i,
                            nh as u32,
                            hd as u32,
                        )
                        .map_err(|e| anyhow::anyhow!("batched FWHT-undo L{layer_idx} s{i}: {e}"))?;
                    }
                }
            } // end BatchedKvRegime::Hybrid
            BatchedKvRegime::FullTq(multi_seq_kv_hb) => {
                // ADR-040 M-SPEED-LC Stage 2/3 — FullTq (byte-packed 5/6/8-bit K+V)
                // attention phase. Same same_bucket-gating skeleton as the Hybrid arm
                // above; SDPA uses mlx-native's `flash_attn_vec_tq_hb_batched`
                // (M-SPEED-LC Stage 1) instead of `flash_attn_vec_hybrid_batched`.
                // Both K and V are FWHT-rotated + TQ-quantized (unlike hybrid's raw
                // F16-K), so Q must ALSO be FWHT sign-premultiplied before SDPA — this
                // mirrors the scalar production sequence at
                // `gemma4/gpu_full_attn.rs:1381-1392` (standalone `fwht_sign_premult`
                // dispatch ahead of the SDPA call, `fuse_fwht_pre: 0` in params).
                let gbuf = &multi_seq_kv_hb[layer_idx];
                let gcap = gbuf.capacity;
                let gring = gbuf.is_sliding;
                let ksl_of = |i: usize| -> u32 {
                    let sp = seq_positions[i];
                    if gring {
                        ((sp + 1).min(gcap)) as u32
                    } else {
                        (sp + 1) as u32
                    }
                };
                let nwg_bucket = |k: u32| if k > 512 { 32u32 } else { 16u32 };
                let max_ksl = (0..n).map(ksl_of).max().unwrap_or(1);
                let nsg_max = mlx_native::ops::flash_attn_vec_tq_hb::compute_nsg(max_ksl);
                let same_bucket = (0..n).all(|i| {
                    let k = ksl_of(i);
                    nwg_bucket(k) == nwg_bucket(max_ksl)
                        && mlx_native::ops::flash_attn_vec_tq_hb::compute_nsg(k) == nsg_max
                });
                let use_batched_flash =
                    std::env::var("HF2Q_BATCHED_FLASH").as_deref() != Ok("0") && same_bucket;
                if layer_idx == 0 && std::env::var("HF2Q_DECODE_TRACE").is_ok() {
                    let dump: Vec<String> = (0..n)
                        .map(|i| {
                            let sp = seq_positions[i];
                            let cp = if gring { (sp % gcap) as u32 } else { sp as u32 };
                            format!("s{}:sp{} ksl{} cp{}", slot_ids[i].0, sp, ksl_of(i), cp)
                        })
                        .collect();
                    eprintln!(
                        "[ROWDIFF-TQ] L0 N={} max_ksl={} same_bucket={} bflash={} [{}]",
                        n,
                        max_ksl,
                        same_bucket,
                        use_batched_flash,
                        dump.join(" ")
                    );
                }

                // Q FWHT sign-premult — ONE dispatch (attnpre) or N per-slot dispatches,
                // covering ALL N queries regardless of same_bucket (premult is a pure
                // per-row transform independent of kv_seq_len/bucket selection, so it
                // is hoisted ahead of the same_bucket branch below — both the batched
                // SDPA path and the per-slot fallback read the SAME already-rotated
                // `bufs.attn_q_normed`). `dispatch_fwht_sign_premult_f32` is row-count-
                // agnostic (grid.x = num_heads param, one threadgroup per row, no
                // cross-row state — same structure as `dispatch_fwht_sign_undo_f32`,
                // already relied on for the Hybrid arm's ATTNPRE batched undo above) ⇒
                // per-row bit-identical to N separate per-slot premult calls.
                session.barrier_between(&[&bufs.attn_q_normed], &[&bufs.attn_q_normed]);
                if attnpre {
                    mlx_native::ops::fwht_standalone::dispatch_fwht_sign_premult_f32(
                        session.encoder_mut(),
                        reg,
                        metal_dev,
                        &bufs.attn_q_normed,
                        (n as u32) * nh as u32,
                        hd as u32,
                    )
                    .map_err(|e| anyhow::anyhow!("tq batched Q premult L{layer_idx}: {e}"))?;
                } else {
                    for i in 0..n {
                        let q_i = bufs
                            .attn_q_normed
                            .slice_view(row_off(q_norm_stride, i), q_norm_stride);
                        session.barrier_between(&[&bufs.attn_q_normed], &[&bufs.attn_q_normed]);
                        mlx_native::ops::fwht_standalone::dispatch_fwht_sign_premult_f32(
                            session.encoder_mut(),
                            reg,
                            metal_dev,
                            &q_i,
                            nh as u32,
                            hd as u32,
                        )
                        .map_err(|e| anyhow::anyhow!("tq Q premult L{layer_idx} s{i}: {e}"))?;
                    }
                }

                if use_batched_flash {
                    let buf = gbuf;
                    let cap = gcap;
                    let is_ring = gring;
                    // PHASE 1 — KV-encode: K and V are BOTH byte-packed TQ-HB (unlike
                    // hybrid's F16-K), so this is 2 batched `dispatch_hadamard_quantize_
                    // kv_hb_batched` calls (K then V) rather than hybrid's F16-K-copy +
                    // FWHT-V-quantize pair. The scalar default production path fuses
                    // K+V into ONE `dispatch_hadamard_quantize_kv_hb_dual` dispatch
                    // (gpu_full_attn.rs:541); no batched-dual kernel exists yet, so this
                    // uses 2 separate dispatches instead. Bit-parity for the BATCHED
                    // multi-query kernel itself (not just fused-vs-2-scalar-dispatches,
                    // which is a DIFFERENT claim `test_hadamard_quantize_kv_hb_dual_
                    // byte_identity_d256` covers) is proven by mlx-native's
                    // `test_hadamard_quantize_kv_hb_batched_parity.rs`
                    // (`hadamard_quantize_kv_hb_batched_bit_parity_matrix`): N=8 queries
                    // in ONE batched dispatch (mixed ring-wrap + linear positions, full
                    // slot permutation, D=256/512, cbits 5/6/8) vs N separate scalar
                    // `dispatch_hadamard_quantize_kv_hb` calls, byte-compared zero-
                    // tolerance. Opt out: HF2Q_BATCHED_KVENC=0 (same env var as the
                    // Hybrid arm's KV-encode toggle).
                    let v_src_buf: &MlxBuffer = if v_is_k {
                        &bufs.attn_v
                    } else {
                        &bufs.attn_v_normed
                    };
                    let use_batched_kvenc =
                        std::env::var("HF2Q_BATCHED_KVENC").as_deref() != Ok("0");
                    if use_batched_kvenc {
                        session.barrier_between(
                            &[&bufs.attn_k_normed, v_src_buf],
                            &[&buf.k_packed, &buf.k_norms, &buf.v_packed, &buf.v_norms],
                        );
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb_batched(
                    session.encoder_mut(), reg, metal_dev,
                    &bufs.attn_k_normed, &buf.k_packed, &buf.k_norms, slot_id_buf, positions_buf,
                    n as u32, nkv as u32, hd as u32, cap as u32, is_ring,
                    tq_scale_factor_d512, tq_codebook_bits,
                ).map_err(|e| anyhow::anyhow!("tq HB-K batched L{layer_idx}: {e}"))?;
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb_batched(
                    session.encoder_mut(), reg, metal_dev,
                    v_src_buf, &buf.v_packed, &buf.v_norms, slot_id_buf, positions_buf,
                    n as u32, nkv as u32, hd as u32, cap as u32, is_ring,
                    tq_scale_factor_d512, tq_codebook_bits,
                ).map_err(|e| anyhow::anyhow!("tq HB-V batched L{layer_idx}: {e}"))?;
                    } else {
                        for i in 0..n {
                            let slot = slot_ids[i].0 as u64;
                            let seq_pos_i = seq_positions[i];
                            let cache_pos: u32 = if is_ring {
                                (seq_pos_i % cap) as u32
                            } else {
                                seq_pos_i as u32
                            };
                            let k_elems = nkv * cap * hd;
                            let norms_per_pos = buf.norms_per_pos;
                            let ne = nkv * cap * norms_per_pos;
                            let norms_shp = if norms_per_pos == 1 {
                                vec![nkv, cap]
                            } else {
                                vec![nkv, cap, norms_per_pos]
                            };
                            let k_view = buf
                                .k_packed
                                .slice_view(slot * (k_elems as u64), k_elems)
                                .with_shape(vec![nkv, cap, hd])
                                .map_err(|e| anyhow::anyhow!("tq K L{layer_idx} s{i}: {e}"))?;
                            let k_norms_view = buf
                                .k_norms
                                .slice_view(slot * (ne as u64) * 4, ne)
                                .with_shape(norms_shp.clone())
                                .map_err(|e| anyhow::anyhow!("tq Kn L{layer_idx} s{i}: {e}"))?;
                            let v_view = buf
                                .v_packed
                                .slice_view(slot * (k_elems as u64), k_elems)
                                .with_shape(vec![nkv, cap, hd])
                                .map_err(|e| anyhow::anyhow!("tq V L{layer_idx} s{i}: {e}"))?;
                            let v_norms_view = buf
                                .v_norms
                                .slice_view(slot * (ne as u64) * 4, ne)
                                .with_shape(norms_shp)
                                .map_err(|e| anyhow::anyhow!("tq Vn L{layer_idx} s{i}: {e}"))?;
                            let kn_i = bufs
                                .attn_k_normed
                                .slice_view(row_off(k_norm_stride, i), k_norm_stride);
                            let v_i = if v_is_k {
                                bufs.attn_v.slice_view(row_off(v_stride, i), v_stride)
                            } else {
                                bufs.attn_v_normed
                                    .slice_view(row_off(v_normed_stride, i), v_normed_stride)
                            };
                            session.barrier_between(
                                &[&kn_i, v_src_buf],
                                &[&buf.k_packed, &buf.k_norms, &buf.v_packed, &buf.v_norms],
                            );
                            mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
                    session.encoder_mut(), reg, metal_dev, &kn_i, &k_view, &k_norms_view,
                    nkv as u32, hd as u32, cap as u32, cache_pos, is_ring, tq_scale_factor_d512, tq_codebook_bits,
                ).map_err(|e| anyhow::anyhow!("tq HB-K L{layer_idx} s{i}: {e}"))?;
                            mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
                    session.encoder_mut(), reg, metal_dev, &v_i, &v_view, &v_norms_view,
                    nkv as u32, hd as u32, cap as u32, cache_pos, is_ring, tq_scale_factor_d512, tq_codebook_bits,
                ).map_err(|e| anyhow::anyhow!("tq HB-V L{layer_idx} s{i}: {e}"))?;
                        }
                    }
                    // PHASE 2 — ONE `flash_attn_vec_tq_hb_batched` dispatch over all N
                    // queries. The dispatcher OWNS undo semantics (fused reduce+undo at
                    // nwg>1 / standalone fwht_sign_undo at nwg==1 — mlx-native
                    // M-SPEED-LC Stage 1) — NO trailing FWHT-undo call here.
                    let p_tq = mlx_native::ops::flash_attn_vec_tq_hb::FlashAttnVecTqHbParams {
                        num_heads: nh as u32,
                        num_kv_heads: nkv as u32,
                        head_dim: hd as u32,
                        kv_seq_len: max_ksl,
                        kv_capacity: cap as u32,
                        scale: 1.0,
                        mask_type: if is_sliding { 2 } else { 1 },
                        sliding_window: if is_sliding {
                            self.sliding_window as u32
                        } else {
                            0
                        },
                        softcap: 0.0,
                        ring_start: 0,
                        scale_factor_d512: tq_scale_factor_d512,
                        codebook_bits: tq_codebook_bits,
                        fuse_fwht_pre: 0,
                        nsg: nsg_max,
                    };
                    session.barrier_between(
                        &[
                            &bufs.attn_q_normed,
                            &buf.k_packed,
                            &buf.k_norms,
                            &buf.v_packed,
                            &buf.v_norms,
                            &bufs.sdpa_tmp,
                        ],
                        &[&bufs.sdpa_out, &bufs.sdpa_tmp],
                    );
                    mlx_native::ops::flash_attn_vec_tq_hb::flash_attn_vec_tq_hb_batched(
                        session.encoder_mut(),
                        reg,
                        dev,
                        n as u32,
                        &bufs.attn_q_normed,
                        &buf.k_packed,
                        &buf.k_norms,
                        &buf.v_packed,
                        &buf.v_norms,
                        &bufs.sdpa_out,
                        &bufs.sdpa_tmp,
                        slot_id_buf,
                        positions_buf,
                        &p_tq,
                    )
                    .map_err(|e| anyhow::anyhow!("batched flash_tq_hb L{layer_idx}: {e}"))?;
                } else {
                    // Per-slot fallback (mixed-bucket, e.g. staggered slots): the
                    // SCALAR production-equivalent sequence per slot — Q is already
                    // premultiplied above; `flash_attn_vec_tq_hb_with_fused_undo` owns
                    // undo internally (fused reduce+undo at nwg>1 / standalone at
                    // nwg==1), matching the batched arm's contract (no trailing undo).
                    let tmp_stride = elems(&bufs.sdpa_tmp) / n;
                    for i in 0..n {
                        let slot = slot_ids[i].0 as u64;
                        let seq_pos_i = seq_positions[i];
                        let buf = &multi_seq_kv_hb[layer_idx];
                        let cap = buf.capacity;
                        let is_ring = buf.is_sliding;
                        let cache_pos: u32 = if is_ring {
                            (seq_pos_i % cap) as u32
                        } else {
                            seq_pos_i as u32
                        };
                        let k_elems = nkv * cap * hd;
                        let norms_per_pos = buf.norms_per_pos;
                        let ne = nkv * cap * norms_per_pos;
                        let norms_shp = if norms_per_pos == 1 {
                            vec![nkv, cap]
                        } else {
                            vec![nkv, cap, norms_per_pos]
                        };
                        let k_view = buf
                            .k_packed
                            .slice_view(slot * (k_elems as u64), k_elems)
                            .with_shape(vec![nkv, cap, hd])
                            .map_err(|e| {
                                anyhow::anyhow!("tq fallback K slot-view L{layer_idx} s{i}: {e}")
                            })?;
                        let k_norms_view = buf
                            .k_norms
                            .slice_view(slot * (ne as u64) * 4, ne)
                            .with_shape(norms_shp.clone())
                            .map_err(|e| {
                                anyhow::anyhow!("tq fallback Kn slot-view L{layer_idx} s{i}: {e}")
                            })?;
                        let v_view = buf
                            .v_packed
                            .slice_view(slot * (k_elems as u64), k_elems)
                            .with_shape(vec![nkv, cap, hd])
                            .map_err(|e| {
                                anyhow::anyhow!("tq fallback V slot-view L{layer_idx} s{i}: {e}")
                            })?;
                        let v_norms_view = buf
                            .v_norms
                            .slice_view(slot * (ne as u64) * 4, ne)
                            .with_shape(norms_shp)
                            .map_err(|e| {
                                anyhow::anyhow!("tq fallback Vn slot-view L{layer_idx} s{i}: {e}")
                            })?;

                        let q_i = bufs
                            .attn_q_normed
                            .slice_view(row_off(q_norm_stride, i), q_norm_stride);
                        let kn_i = bufs
                            .attn_k_normed
                            .slice_view(row_off(k_norm_stride, i), k_norm_stride);
                        let v_src_buf: &MlxBuffer = if v_is_k {
                            &bufs.attn_v
                        } else {
                            &bufs.attn_v_normed
                        };
                        let v_i = if v_is_k {
                            bufs.attn_v.slice_view(row_off(v_stride, i), v_stride)
                        } else {
                            bufs.attn_v_normed
                                .slice_view(row_off(v_normed_stride, i), v_normed_stride)
                        };
                        let sdpa_i = bufs
                            .sdpa_out
                            .slice_view(row_off(sdpa_stride, i), sdpa_stride);
                        let tmp_i = bufs.sdpa_tmp.slice_view(row_off(tmp_stride, i), tmp_stride);

                        session.barrier_between(
                            &[&bufs.attn_k_normed, v_src_buf],
                            &[&buf.k_packed, &buf.k_norms, &buf.v_packed, &buf.v_norms],
                        );
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
                            session.encoder_mut(),
                            reg,
                            metal_dev,
                            &kn_i,
                            &k_view,
                            &k_norms_view,
                            nkv as u32,
                            hd as u32,
                            cap as u32,
                            cache_pos,
                            is_ring,
                            tq_scale_factor_d512,
                            tq_codebook_bits,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("tq fallback HB-K write L{layer_idx} s{i}: {e}")
                        })?;
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
                            session.encoder_mut(),
                            reg,
                            metal_dev,
                            &v_i,
                            &v_view,
                            &v_norms_view,
                            nkv as u32,
                            hd as u32,
                            cap as u32,
                            cache_pos,
                            is_ring,
                            tq_scale_factor_d512,
                            tq_codebook_bits,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("tq fallback HB-V quant L{layer_idx} s{i}: {e}")
                        })?;

                        let kv_seq_len = if is_ring {
                            ((seq_pos_i + 1).min(cap)) as u32
                        } else {
                            (seq_pos_i + 1) as u32
                        };
                        let ring_start = if is_ring && kv_seq_len as usize >= cap {
                            ((seq_pos_i + 1) % cap) as u32
                        } else {
                            0u32
                        };
                        session.barrier_between(
                            &[
                                &bufs.attn_q_normed,
                                &buf.k_packed,
                                &buf.k_norms,
                                &buf.v_packed,
                                &buf.v_norms,
                                &tmp_i,
                            ],
                            &[&bufs.sdpa_out, &tmp_i],
                        );
                        let p_tq = mlx_native::ops::flash_attn_vec_tq_hb::FlashAttnVecTqHbParams {
                            num_heads: nh as u32,
                            num_kv_heads: nkv as u32,
                            head_dim: hd as u32,
                            kv_seq_len,
                            kv_capacity: cap as u32,
                            scale: 1.0,
                            mask_type: if is_sliding { 2 } else { 1 },
                            sliding_window: if is_sliding {
                                self.sliding_window as u32
                            } else {
                                0
                            },
                            softcap: 0.0,
                            ring_start,
                            scale_factor_d512: tq_scale_factor_d512,
                            codebook_bits: tq_codebook_bits,
                            fuse_fwht_pre: 0,
                            nsg: mlx_native::ops::flash_attn_vec_tq_hb::compute_nsg(kv_seq_len),
                        };
                        mlx_native::ops::flash_attn_vec_tq_hb::flash_attn_vec_tq_hb_with_fused_undo(
                    session.encoder_mut(), reg, dev,
                    &q_i, &k_view, &k_norms_view, &v_view, &v_norms_view,
                    &sdpa_i, &tmp_i, &p_tq,
                ).map_err(|e| anyhow::anyhow!("tq fallback flash_attn_vec_tq_hb_with_fused_undo L{layer_idx} s{i}: {e}"))?;
                    }
                }
            } // end BatchedKvRegime::FullTq
        }

        // catsplit: KV-encode + flash attention + FWHT-undo (the whole
        // per-slot/batched attention block above, either regime).
        cat_boundary(session, exec, catsplit::Cat::AttnFlash)?;

        let num_experts = self.num_experts;
        let top_k = self.layers[layer_idx].moe.top_k;
        let moe_int = self.layers[layer_idx].moe.moe_intermediate_size;
        let interm = self.intermediate_size;

        // -- O-proj (BATCHED m=N): sdpa_out -> attn_out --
        session.barrier_between(
            &[&bufs.sdpa_out, &self.layers[layer_idx].attn.o_proj.buffer],
            &[&bufs.attn_out],
        );
        dispatch_dense_rowident(
            session,
            reg,
            dev,
            &bufs.sdpa_out,
            &self.layers[layer_idx].attn.o_proj,
            &bufs.attn_out,
            n,
            sdpa_stride,
            hs,
            "attn_output",
            layer_idx,
        )?;
        cat_boundary(session, exec, catsplit::Cat::DenseO)?;

        // -- Fused post-attn norm + residual add (BATCHED rows=N): residual =
        // norm(attn_out, post_attn_w) + hidden. Default (non-split) path. --
        session.barrier_between(&[&bufs.hidden, &bufs.attn_out], &[&bufs.residual]);
        mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
            session.encoder_mut(),
            reg,
            metal_dev,
            &bufs.hidden,
            &bufs.attn_out,
            &self.layers[layer_idx].norms.post_attention_layernorm,
            &bufs.residual,
            hs as u32,
            nu,
            eps,
        )
        .map_err(|e| anyhow::anyhow!("batched post-attn norm+add L{layer_idx}: {e}"))?;

        // -- B8: pre-FF norm1 + pre-FF norm2 + router norm (BATCHED rows=N) --
        // Plain rms_norm rows=N (the scalar's `rms_norm_f32_hs_cached` is a
        // rows=1 pipeline-cache optimization; same math, per-row bit-identical).
        session.barrier_between(
            &[&bufs.residual],
            &[&bufs.norm_out, &bufs.moe_norm_out, &bufs.router_norm_out],
        );
        session
            .rms_norm(
                reg,
                metal_dev,
                &bufs.residual,
                &self.layers[layer_idx].norms.pre_feedforward_layernorm,
                &bufs.norm_out,
                &self.activations.norm_params,
                nu,
                hs as u32,
            )
            .map_err(|e| anyhow::anyhow!("batched pre-FF norm L{layer_idx}: {e}"))?;
        session
            .rms_norm(
                reg,
                metal_dev,
                &bufs.residual,
                &self.layers[layer_idx].norms.pre_feedforward_layernorm_2,
                &bufs.moe_norm_out,
                &self.activations.norm_params,
                nu,
                hs as u32,
            )
            .map_err(|e| anyhow::anyhow!("batched pre-FF norm 2 L{layer_idx}: {e}"))?;
        session
            .rms_norm(
                reg,
                metal_dev,
                &bufs.residual,
                &self.layers[layer_idx].moe.router_combined_weight,
                &bufs.router_norm_out,
                &self.activations.norm_params,
                nu,
                hs as u32,
            )
            .map_err(|e| anyhow::anyhow!("batched router norm L{layer_idx}: {e}"))?;
        // catsplit: post-attn norm+add + pre-FF norm1/norm2 + router norm.
        cat_boundary(session, exec, catsplit::Cat::RmsNorm)?;

        // -- B9: dense gate + dense up + router logits (BATCHED m=N) --
        session.barrier_between(
            &[&bufs.norm_out, &bufs.router_norm_out],
            &[&bufs.mlp_gate, &bufs.mlp_up, &bufs.moe_router_logits],
        );
        dispatch_dense_rowident(
            session,
            reg,
            dev,
            &bufs.norm_out,
            &self.layers[layer_idx].mlp.gate_proj,
            &bufs.mlp_gate,
            n,
            hs,
            interm,
            "ffn_gate",
            layer_idx,
        )?;
        dispatch_dense_rowident(
            session,
            reg,
            dev,
            &bufs.norm_out,
            &self.layers[layer_idx].mlp.up_proj,
            &bufs.mlp_up,
            n,
            hs,
            interm,
            "ffn_up",
            layer_idx,
        )?;
        // router_proj is F32 (ffn_gate_inp) → per-row m=1 (tile path not byte-identical).
        dispatch_dense_rowident(
            session,
            reg,
            dev,
            &bufs.router_norm_out,
            &self.layers[layer_idx].moe.router_proj,
            &bufs.moe_router_logits,
            n,
            hs,
            num_experts,
            "ffn_gate_inp",
            layer_idx,
        )?;
        // catsplit: dense MLP gate + up + router projection.
        cat_boundary(session, exec, catsplit::Cat::DenseFfn)?;

        // -- B10: fused_gelu_mul (BATCHED, elementwise over N*intermediate) +
        // fused_moe_routing (PER-SLOT: each token's router logits -> top_k) --
        session.barrier_between(
            &[&bufs.mlp_gate, &bufs.mlp_up, &bufs.moe_router_logits],
            &[
                &bufs.mlp_fused,
                &bufs.moe_expert_ids,
                &bufs.moe_routing_weights_gpu,
            ],
        );
        {
            let total = (interm * n) as u32;
            let n_elements_bytes = total.to_ne_bytes();
            let pipeline = reg.get_pipeline("fused_gelu_mul", metal_dev)?;
            encode_with_args(
                session.encoder_mut(),
                pipeline,
                &[
                    (0, KernelArg::Buffer(&bufs.mlp_gate)),
                    (1, KernelArg::Buffer(&bufs.mlp_up)),
                    (2, KernelArg::Buffer(&bufs.mlp_fused)),
                    (3, KernelArg::Bytes(&n_elements_bytes)),
                ],
                mlx_native::MTLSize::new(total as u64, 1, 1),
                mlx_native::MTLSize::new(std::cmp::min(256, total as u64), 1, 1),
            );
        }
        // MoE routing stays PER-SLOT: the existing prefill
        // `fused_moe_routing_batch_f32` kernel is NOT byte-identical to the
        // decode single-token `fused_moe_routing_f32` (different softmax/top-k
        // reduction — diverges at n=1 even with V3 off; ADR-040 M4 fork finding).
        // And the weighted-sum batched-seq kernel IS byte-identical but
        // throughput-NEUTRAL (the dispatch was never a cost) — so neither MoE
        // per-slot loop is a worthwhile fusion lever; the 202→243 residual is
        // elsewhere (batched-op efficiency). Both kept per-slot.
        let rl_stride = elems(&bufs.moe_router_logits) / n; // num_experts
        let ids_stride = elems(&bufs.moe_expert_ids) / n; // top_k
        for i in 0..n {
            let rl_i = bufs
                .moe_router_logits
                .slice_view(row_off(rl_stride, i), rl_stride);
            let ids_i = bufs
                .moe_expert_ids
                .slice_view(row_off(ids_stride, i), ids_stride);
            let w_i = bufs
                .moe_routing_weights_gpu
                .slice_view(row_off(ids_stride, i), ids_stride);
            mlx_native::ops::fused_norm_add::dispatch_fused_moe_routing_f32(
                session.encoder_mut(),
                reg,
                metal_dev,
                &rl_i,
                &ids_i,
                &w_i,
                &self.layers[layer_idx].moe.per_expert_scale,
                num_experts as u32,
                top_k as u32,
            )
            .map_err(|e| anyhow::anyhow!("batched MoE routing L{layer_idx} slot{i}: {e}"))?;
        }
        // catsplit: fused_gelu_mul (dense SwiGLU) + per-slot MoE routing/top-k.
        cat_boundary(session, exec, catsplit::Cat::MoeOther)?;

        // -- B11: dense down: mlp_fused -> mlp_down. down_proj is F16
        // (intermediate=2112 not 256-aligned) → per-row m=1 (tile path not
        // byte-identical to the m=1 matvec). --
        session.barrier_between(
            &[
                &bufs.mlp_fused,
                &self.layers[layer_idx].mlp.down_proj.buffer,
            ],
            &[&bufs.mlp_down],
        );
        dispatch_dense_rowident(
            session,
            reg,
            dev,
            &bufs.mlp_fused,
            &self.layers[layer_idx].mlp.down_proj,
            &bufs.mlp_down,
            n,
            interm,
            hs,
            "ffn_down",
            layer_idx,
        )?;
        // catsplit: dense MLP down projection.
        cat_boundary(session, exec, catsplit::Cat::DenseFfn)?;

        // -- MoE gate_up_id (BATCHED n_tokens=N — H-S2-tokenparity) --
        let stacked_gate_up = self.layers[layer_idx]
            .moe
            .stacked_gate_up
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "batched body requires fused _id MoE (stacked_gate_up) L{layer_idx}"
                )
            })?;
        let stacked_down = self.layers[layer_idx]
            .moe
            .stacked_down
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!("batched body requires fused _id MoE (stacked_down) L{layer_idx}")
            })?;
        session.barrier_between(
            &[&bufs.moe_norm_out, &bufs.moe_expert_ids, stacked_gate_up],
            &[&bufs.moe_gate_up_id_out],
        );
        let gu_params = mlx_native::GgmlQuantizedMatmulIdParams {
            n_tokens: nu,
            top_k: top_k as u32,
            n: (2 * moe_int) as u32,
            k: hs as u32,
            n_experts: num_experts as u32,
            expert_stride: self.layers[layer_idx].moe.gate_up_expert_stride,
            ggml_type: self.layers[layer_idx].moe.gate_up_ggml_dtype,
        };
        // ADR-040 Phase F `iter-F-moe-mvid`: force the per-token `mv_id` route
        // (byte-identical to the serial slot-aware ref). At the batched decode
        // width the down projection's `n_tokens = N*top_k` crosses the `mm_id`
        // grouped-kernel threshold (>32, i.e. N≥5) whose reduction order is NOT
        // bit-identical to serial — `slot_aware_n8_per_slot_parity_vs_serial`
        // diverged on the routed entry point and is byte-identical on this one.
        // `mm_id` is a prefill optimization + a measured regression at N≤8, so
        // there is no decode-width cost. gate_up is mv at N≤4 anyway; pinning it
        // keeps the whole MoE byte-identical for any future N.
        crate::inference::expert_dispatch::dispatch_expert_matmul_id_mv(
            session,
            reg,
            dev,
            &bufs.moe_norm_out,
            stacked_gate_up,
            &bufs.moe_expert_ids,
            &bufs.moe_gate_up_id_out,
            &gu_params,
        )
        .map_err(|e| anyhow::anyhow!("batched gate_up _id L{layer_idx}: {e}"))?;
        // catsplit: MoE gate_up `_id` (per-token mv_id) — the big expert read.
        cat_boundary(session, exec, catsplit::Cat::MoeGateUp)?;

        // -- swiglu (BATCHED over N*top_k expert rows) --
        session.barrier_between(&[&bufs.moe_gate_up_id_out], &[&bufs.moe_swiglu_id_out]);
        mlx_native::ops::moe_dispatch::moe_swiglu_batch_encode(
            session.encoder_mut(),
            reg,
            metal_dev,
            &bufs.moe_gate_up_id_out,
            &bufs.moe_swiglu_id_out,
            moe_int,
            top_k * n,
        )
        .map_err(|e| anyhow::anyhow!("batched swiglu L{layer_idx}: {e}"))?;
        // catsplit: MoE expert SwiGLU activation.
        cat_boundary(session, exec, catsplit::Cat::MoeOther)?;

        // -- down_id (BATCHED n_tokens=N*top_k) --
        session.barrier_between(
            &[&bufs.moe_swiglu_id_out, &bufs.moe_expert_ids, stacked_down],
            &[&bufs.moe_down_id_out],
        );
        let dn_params = mlx_native::GgmlQuantizedMatmulIdParams {
            n_tokens: (top_k * n) as u32,
            top_k: 1,
            n: hs as u32,
            k: moe_int as u32,
            n_experts: num_experts as u32,
            expert_stride: self.layers[layer_idx].moe.down_expert_stride,
            ggml_type: self.layers[layer_idx].moe.down_ggml_dtype,
        };
        // ADR-040 Phase F `iter-F-moe-mvid`: per-token `mv_id` (byte-identical).
        // This is THE divergence site at N≥5 (`n_tokens = N*top_k > 32`).
        crate::inference::expert_dispatch::dispatch_expert_matmul_id_mv(
            session,
            reg,
            dev,
            &bufs.moe_swiglu_id_out,
            stacked_down,
            &bufs.moe_expert_ids,
            &bufs.moe_down_id_out,
            &dn_params,
        )
        .map_err(|e| anyhow::anyhow!("batched down _id L{layer_idx}: {e}"))?;
        // catsplit: MoE down `_id` (per-token mv_id) — the second big expert read.
        cat_boundary(session, exec, catsplit::Cat::MoeDown)?;

        // -- post-FF norm 1 (BATCHED rows=N): mlp_down -> attn_out --
        session.barrier_between(&[&bufs.mlp_down], &[&bufs.attn_out]);
        session
            .rms_norm(
                reg,
                metal_dev,
                &bufs.mlp_down,
                &self.layers[layer_idx].norms.post_feedforward_layernorm_1,
                &bufs.attn_out,
                &self.activations.norm_params,
                nu,
                hs as u32,
            )
            .map_err(|e| anyhow::anyhow!("batched post-FF norm1 L{layer_idx}: {e}"))?;
        // catsplit: post-FF norm1.
        cat_boundary(session, exec, catsplit::Cat::RmsNorm)?;

        // -- weighted_sum (PER-SLOT: each token's top_k experts -> its accum row)
        // into the DEDICATED moe_accum buffer (mirrors scalar moe_accum). The
        // barrier_between registers moe_down_id_out as a read (RAW vs down_id)
        // AND moe_accum as a write, so the following post-FF-norm2 barrier sees
        // moe_accum and serializes — without it the (untracked) weighted_sum
        // races the norm2 read. Mirror of scalar gpu_full_attn.rs:2130. --
        let down_stride = elems(&bufs.moe_down_id_out) / n; // top_k*hs
        let w_stride = elems(&bufs.moe_routing_weights_gpu) / n; // top_k
        let acc_stride = hs; // moe_accum row = hidden
        session.barrier_between(
            &[&bufs.moe_down_id_out, &bufs.moe_routing_weights_gpu],
            &[&bufs.moe_accum],
        );
        // ADR-040 §24 iter-J — DISPATCH FUSION. The per-slot weighted_sum loop
        // (8 dispatches/layer = ~210/step) was kept per-slot on the stale M4
        // "dispatch was never a cost" finding; the §24 catsplit dispatch-count
        // localization showed moe_other (this + routing) is 536 disp/step (31%
        // of all dispatches, 1.9% of GPU time) — the dominant encode-time cost.
        // The batched-seq kernel computes the SAME per-token top_k·weight sum
        // (buffers are already [N,top_k,hs]/[N,top_k]/[N,hs] contiguous), so it
        // is byte-identical (gate: slot_aware_n8_per_slot_parity_vs_serial).
        // DEFAULT-ON (opt out HF2Q_BATCHED_WSUM=0). Parity-gate-proven
        // byte-identical (slot_aware_n8). NOTE: this −209 dispatch/step
        // reduction is THROUGHPUT-NEUTRAL (measured ~noise) — it refuted the
        // "encode is dispatch-count-bound" hypothesis (−12% dispatches → ~0 wall;
        // these small kernels encode cheaply). Kept as the cleaner batched form +
        // a strict dispatch reduction, not for a throughput claim.
        let batched_wsum = std::env::var("HF2Q_BATCHED_WSUM").as_deref() != Ok("0");
        if batched_wsum {
            let _ = (down_stride, w_stride, acc_stride);
            mlx_native::ops::moe_dispatch::moe_weighted_sum_seq_encode(
                session.encoder_mut(),
                reg,
                metal_dev,
                &bufs.moe_down_id_out,
                &bufs.moe_routing_weights_gpu,
                &bufs.moe_accum,
                hs,
                top_k,
                n,
            )
            .map_err(|e| anyhow::anyhow!("batched-seq weighted_sum L{layer_idx}: {e}"))?;
        } else {
            for i in 0..n {
                let din_i = bufs
                    .moe_down_id_out
                    .slice_view(row_off(down_stride, i), down_stride);
                let w_i = bufs
                    .moe_routing_weights_gpu
                    .slice_view(row_off(w_stride, i), w_stride);
                let acc_i = bufs
                    .moe_accum
                    .slice_view(row_off(acc_stride, i), acc_stride);
                mlx_native::ops::moe_dispatch::moe_weighted_sum_encode(
                    session.encoder_mut(),
                    reg,
                    metal_dev,
                    &din_i,
                    &w_i,
                    &acc_i,
                    hs,
                    top_k,
                )
                .map_err(|e| anyhow::anyhow!("batched weighted_sum L{layer_idx} slot{i}: {e}"))?;
            }
        }
        // catsplit: MoE per-slot weighted_sum (top_k expert combine).
        cat_boundary(session, exec, catsplit::Cat::MoeOther)?;

        // -- post-FF norm2 + combine + end-of-layer (BATCHED rows=N) --
        // Mirror the scalar's branch (gpu_full_attn.rs:2191/2253) on
        // INVESTIGATION_ENV.fused_end_of_layer (HF2Q_FUSED_END_OF_LAYER,
        // DEFAULT-ON) so the batched body is BYTE-IDENTICAL under BOTH settings:
        //   * fused ON (production default) → the single fused kernel. The
        //     unfused 2-dispatch decomposition is mathematically equal but NOT
        //     byte-identical (proven: the N=1 gate passed only under
        //     HF2Q_FUSED_END_OF_LAYER=0 before this branch was added).
        //   * fused OFF → the 2-dispatch path (post-FF-norm2 then end-of-layer).
        // The iter-367 wsum fusion (HF2Q_FUSED_MOE_WSUM_END_LAYER_V2) is
        // default-OFF, so the weighted_sum stays a separate dispatch above.
        let scalar_is_vector = self.layers[layer_idx].layer_scalar.element_count() > 1;
        if INVESTIGATION_ENV.fused_end_of_layer {
            // ONE kernel: mlp_down = norm(moe_accum, post_ff_norm2) + attn_out;
            // hidden = (norm(mlp_down, post_ff_norm) + residual) * layer_scalar.
            // The kernel dispatches one threadgroup per row ⇒ rows=N batches all
            // slots. Bit-identical to scalar gpu_full_attn.rs:2238.
            session.barrier_between(
                &[
                    &bufs.attn_out,
                    &bufs.moe_accum,
                    &bufs.residual,
                    &self.layers[layer_idx].layer_scalar,
                ],
                &[&bufs.mlp_down, &bufs.hidden],
            );
            mlx_native::ops::rms_norm::dispatch_fused_post_ff_norm2_endlayer_f32(
                session.encoder_mut(),
                reg,
                metal_dev,
                &bufs.attn_out,
                &bufs.moe_accum,
                &bufs.residual,
                &self.layers[layer_idx].norms.post_feedforward_layernorm_2,
                &self.layers[layer_idx].norms.post_feedforward_layernorm,
                &self.layers[layer_idx].layer_scalar,
                &bufs.mlp_down,
                &bufs.hidden,
                eps,
                nu,
                hs as u32,
                scalar_is_vector,
            )
            .map_err(|e| anyhow::anyhow!("batched fused end-of-layer L{layer_idx}: {e}"))?;
        } else {
            // post-FF norm2 + combine: mlp_down = norm(moe_accum, post_ff_norm2) + attn_out
            session.barrier_between(&[&bufs.attn_out, &bufs.moe_accum], &[&bufs.mlp_down]);
            mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
                session.encoder_mut(),
                reg,
                metal_dev,
                &bufs.attn_out,
                &bufs.moe_accum,
                &self.layers[layer_idx].norms.post_feedforward_layernorm_2,
                &bufs.mlp_down,
                hs as u32,
                nu,
                eps,
            )
            .map_err(|e| anyhow::anyhow!("batched post-FF norm2+combine L{layer_idx}: {e}"))?;

            // end-of-layer: hidden = (norm(mlp_down, post_ff_norm) + residual) * layer_scalar
            session.barrier_between(&[&bufs.residual, &bufs.mlp_down], &[&bufs.hidden]);
            mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_scalar_f32(
                session.encoder_mut(),
                reg,
                metal_dev,
                &bufs.residual,
                &bufs.mlp_down,
                &self.layers[layer_idx].norms.post_feedforward_layernorm,
                &bufs.hidden,
                &self.layers[layer_idx].layer_scalar,
                nu,
                hs as u32,
                eps,
                scalar_is_vector,
            )
            .map_err(|e| anyhow::anyhow!("batched end-of-layer L{layer_idx}: {e}"))?;
        }
        // catsplit: post-FF norm2 + end-of-layer norm+scale (fused or unfused).
        cat_boundary(session, exec, catsplit::Cat::RmsNorm)?;

        // Layer complete: bufs.hidden holds this layer's output for all N slots.
        Ok(())
    }

    /// ADR-040 S2/S3 — the `[N,hidden]` batched decode BODY: embed-gather N
    /// tokens → run the layer loop in `[N,hidden]` (per-slot attention against
    /// each slot's `multi_seq_kv_hybrid` region) → return each slot's final
    /// hidden row `[n, hidden]` (pre-final-norm — the same value scalar
    /// `forward_decode_capture_hidden` leaves in `self.activations.hidden`). The
    /// SlotAware worker feeds those rows to the proven batched head + finalize.
    ///
    /// `tokens[i]` / `slot_ids[i]` / `seq_positions[i]` describe slot `i`'s
    /// current decode step. Both production KV regimes (see
    /// [`BatchedKvRegime`]): hybrid F16-K + TQ-HB-V (`HF2Q_HYBRID_KV=1`,
    /// DEFAULT) and full-TQ byte-packed K+V (`HF2Q_HYBRID_KV=0`, opt-in;
    /// ADR-040 M-SPEED-LC Stage 2/3). Wired into `decode_batch_gemma4` under
    /// `HF2Q_BATCHED_BODY=1`; the hybrid regime is proven byte-identical to N
    /// serial slot-aware decodes by `slot_aware_n1` + `slot_aware_n4`.
    pub(crate) fn forward_decode_body_batched(
        &self,
        tokens: &[u32],
        slot_ids: &[SlotId],
        seq_positions: &[usize],
        regime: &mut BatchedKvRegime<'_>,
        // ADR-040 §25 iter-L — when HF2Q_FUSE_LMHEAD=1 (chunked path only), the
        // lm_head is encoded as the final pipeline chunk and its output is written
        // here (the body then returns an EMPTY Vec; the caller uses `head_out`).
        // When the fuse is off, this stays None and the body returns hidden.
        head_out: &mut Option<BatchedHeadOut>,
        gpu: &mut GpuContext,
    ) -> Result<Vec<f32>> {
        let n = tokens.len();
        let hs = self.hidden_size;
        if n == 0 {
            return Ok(Vec::new());
        }
        // tq params — read identically to the scalar (forward_gpu.rs:517/560).
        let tq_scale_factor_d512: f32 = match std::env::var("HF2Q_SCALE_FORMULA").as_deref() {
            Ok("sqrt256") => 16.0,
            Ok("sqrt512") => 512.0_f32.sqrt(),
            _ => 1.0,
        };
        let tq_codebook_bits =
            crate::serve::api::tq_packed_descriptor::effective_gemma_tq_codebook_bits();

        let (exec, reg) = gpu.split();
        let dev = exec.device();
        let metal_dev = dev.metal_device();

        let bufs = BatchedDecodeBuffers::new(dev, &self.activations, n)?;
        let h_stride = bufs.hidden_stride();

        // Positions buffer [N] u32 for per-slot RoPE.
        let mut positions_buf = dev
            .alloc_buffer(n * 4, DType::U32, vec![n])
            .map_err(|e| anyhow::anyhow!("body_batched positions alloc: {e}"))?;
        {
            let p: &mut [u32] = positions_buf
                .as_mut_slice()
                .map_err(|e| anyhow::anyhow!("body_batched positions write: {e}"))?;
            for (i, &sp) in seq_positions.iter().enumerate() {
                p[i] = sp as u32;
            }
        }
        // Physical slot-id buffer [N] u32 — constant across layers; feeds the M4
        // batched flash kernel (it derives each query's KV base offset).
        let mut slot_id_buf = dev
            .alloc_buffer(n * 4, DType::U32, vec![n])
            .map_err(|e| anyhow::anyhow!("body_batched slot_id alloc: {e}"))?;
        {
            let s: &mut [u32] = slot_id_buf
                .as_mut_slice()
                .map_err(|e| anyhow::anyhow!("body_batched slot_id write: {e}"))?;
            for (i, sid) in slot_ids.iter().enumerate() {
                s[i] = sid.0;
            }
        }

        let mut s = exec
            .begin()
            .map_err(|e| anyhow::anyhow!("body_batched session begin: {e}"))?;

        // Embed-gather (PER-SLOT): token i -> bufs.hidden row i, scaled sqrt(hs).
        let scale = (hs as f32).sqrt();
        for i in 0..n {
            let h_i = bufs.hidden.slice_view(row_off(h_stride, i), h_stride);
            mlx_native::ops::elementwise::embedding_gather_scale_f32(
                s.encoder_mut(),
                reg,
                metal_dev,
                &self.embed_weight,
                &h_i,
                tokens[i],
                hs,
                scale,
            )
            .map_err(|e| anyhow::anyhow!("body_batched embed slot{i}: {e}"))?;
        }
        // Register the embed's write to `hidden` in the conflict tracker. The
        // scalar runs embed in its own finished session (full sync before the
        // layer sessions); here embed + all layers share ONE session, so the
        // first layer's pre-attn `barrier_between([hidden,..],[norm_out])` must
        // see `hidden` as a prior write to emit the RAW barrier. Without this
        // the pre-attn norm can race the (untracked) embed dispatches.
        s.track_dispatch(&[&self.embed_weight], &[&bufs.hidden]);
        // catsplit: embedding gather (per-slot). Boundary commits the embed CB and
        // re-begins so the layer loop's categories start in a clean session. The
        // commit is a full GPU sync, so layer-0's pre-attn RAW vs `hidden` is
        // satisfied by completion (the fresh session's tracker re-declares it on
        // its first `barrier_between`).
        cat_boundary(&mut s, exec, catsplit::Cat::Embed)?;

        // Layer loop in [N,hidden].
        let num_layers = self.layers.len();
        let cksum_on = std::env::var("HF2Q_S019_CKSUM").is_ok();

        // ADR-040 §21 — intra-step command-buffer pipelining (HF2Q_DECODE_CB_CHUNKS=K,
        // default OFF=1). Splits the per-step layer loop into K command buffers,
        // ASYNC-committing each in order so the GPU executes chunk c while the CPU
        // ENCODES chunk c+1 (the peer's ggml-metal n_cb pattern, single-threaded so
        // NO cross-thread aliasing). Cross-CB ordering is the same-queue COMMIT order
        // (no MTLFence/Event needed — codex-reviewed): chunk c's CB fully executes
        // before chunk c+1's reads its output. BYTE-IDENTICAL by construction (same
        // dispatches, same order, just split across CBs) — VALIDATED:
        // slot_aware_n8_per_slot_parity_vs_serial GREEN at K=3. Each chunk's first
        // `encode_one_layer_batched` opens with a `barrier_between` so every CB starts
        // with a clean intra-CB conflict tracker. Disabled when the per-layer
        // cksum/catsplit debug paths own the session lifetime.
        //
        // ADR-040 §25 iter-K — DEFAULT-ON at K=4 (opt out HF2Q_DECODE_CB_CHUNKS=0/1).
        // RE-MEASURED 2026-06-28 (N=8 gemma4-ara Q5_K_M, post iter-I/iter-J): K=4 =
        // +2.5–3.1% (233→240 t/s), K=6/10 ≈ +3%. The earlier "+1.5–2.7%, marginal,
        // default-OFF" verdict was too pessimistic (and the baseline moved up). This
        // recovers the serial-encode-overlap part of the host gap: §25 localized the
        // ~1.66ms/step encode as raw Metal arg-encoding (NOT barriers, 0.22ms), and
        // overlapping it behind GPU exec via incremental async commit is the byte-
        // identical mechanism (record-reuse can't — it still pays the Metal calls;
        // codex-confirmed). BYTE-IDENTICAL VALIDATED at K=4: slot_aware_n8_per_slot
        // _parity_vs_serial GREEN + slot_aware_staggered_eviction_no_peer_perturbation
        // GREEN (continuous-batching mid-window admission). Cross-CB ordering = same-
        // queue commit order (no fences). Disabled under cksum/catsplit debug (they
        // own the session lifetime).
        let cb_chunks_req: usize = std::env::var("HF2Q_DECODE_CB_CHUNKS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4);
        let cb_chunks: usize = if cb_chunks_req >= 2 && !cksum_on && !*catsplit::ENABLED {
            cb_chunks_req.min(num_layers.max(1))
        } else {
            1
        };

        // ADR-040 §25 iter-L — fuse lm_head into the body's CB pipeline (one
        // commit_and_wait instead of two; lm_head encode overlaps the body GPU).
        // DEFAULT-ON (opt out HF2Q_FUSE_LMHEAD=0); chunked path only, Q6_K head, not under the
        // cksum/catsplit debug paths (they own session lifetime, already excluded
        // from cb_chunks>=2 above — re-asserted here for clarity).
        let fuse_lmhead = cb_chunks >= 2
            && std::env::var("HF2Q_FUSE_LMHEAD").as_deref() != Ok("0")
            && self.lm_head_q6k.is_some()
            && !cksum_on
            && !*catsplit::ENABLED;

        if cb_chunks >= 2 {
            // Async-committed chunks; only the LAST chunk waits (same-queue order ⇒
            // its completion implies all prior CBs completed). Hold committed
            // encoders alive until the final wait.
            let mut committed: Vec<mlx_native::CommandEncoder> = Vec::with_capacity(cb_chunks);
            // §25 iter-L: fused lm_head output buffers (read after the final wait).
            let mut fused_head: Option<(
                MlxBuffer,
                MlxBuffer,
                MlxBuffer,
                Option<super::batched_head::GpuSampleBuffers>,
            )> = None;
            let per = num_layers.div_ceil(cb_chunks);
            let mut layer_idx = 0usize;
            while layer_idx < num_layers {
                let chunk_end = (layer_idx + per).min(num_layers);
                while layer_idx < chunk_end {
                    self.encode_one_layer_batched(
                        layer_idx,
                        &bufs,
                        n,
                        &positions_buf,
                        &slot_id_buf,
                        slot_ids,
                        seq_positions,
                        regime,
                        tq_scale_factor_d512,
                        tq_codebook_bits,
                        &mut s,
                        exec,
                        reg,
                    )?;
                    layer_idx += 1;
                }
                if layer_idx >= num_layers {
                    // §25 iter-L: append lm_head (final-norm + Q6_K matmul + softcap)
                    // into THIS session as the final pipeline chunk, reading
                    // bufs.hidden directly (the last layer wrote it into this same
                    // session's tracker, so the first head barrier orders it). One
                    // commit_and_wait covers body + head.
                    if fuse_lmhead {
                        fused_head =
                            Some(self.encode_lm_head_into(&mut s, &bufs.hidden, n, dev, reg)?);
                    }
                    s.finish()
                        .map_err(|e| anyhow::anyhow!("body_batched chunked final finish: {e}"))?;
                    break;
                } else {
                    let fresh = exec
                        .begin()
                        .map_err(|e| anyhow::anyhow!("body_batched chunk re-begin: {e}"))?;
                    let old = std::mem::replace(&mut s, fresh);
                    committed.push(old.commit());
                }
            }
            // Final chunk waited; same-queue order ⇒ all earlier CBs done.
            // §25: fold the async-committed chunks' GPU-busy into the accumulator
            // (HF2Q_GPU_BUSY) — they bypass commit_and_wait's GPU-time hook, so
            // without this the profiler undercounts GPU-busy on the pipelined path.
            for enc in &committed {
                enc.accumulate_gpu_busy();
            }
            drop(committed);

            // §25 iter-L: fused head — read logits/normed after the single wait and
            // hand them back via head_out; the body returns an empty hidden Vec.
            if let Some((logits_b, normed_b, _softcap_params_b, gpu_sample_bufs)) = fused_head {
                let _hp = std::time::Instant::now();
                let logits: Vec<f32> = logits_b
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("fused lm_head read logits: {e}"))?
                    .to_vec();
                let normed: Vec<f32> = normed_b
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("fused lm_head read normed: {e}"))?
                    .to_vec();
                // §26 iter-M: read back the small GPU-sample buffers (top1 +
                // threshold candidates), if produced.
                let gpu_sample = match gpu_sample_bufs {
                    Some(b) => Some(super::batched_head::GpuSampleOut {
                        top1_idx: b
                            .top1_idx
                            .as_slice::<u32>()
                            .map_err(|e| anyhow::anyhow!("gpu_sample read top1_idx: {e}"))?
                            .to_vec(),
                        top1_val: b
                            .top1_val
                            .as_slice::<f32>()
                            .map_err(|e| anyhow::anyhow!("gpu_sample read top1_val: {e}"))?
                            .to_vec(),
                        cand_count: b
                            .cand_count
                            .as_slice::<u32>()
                            .map_err(|e| anyhow::anyhow!("gpu_sample read cand_count: {e}"))?
                            .to_vec(),
                        overflow: b
                            .overflow
                            .as_slice::<u32>()
                            .map_err(|e| anyhow::anyhow!("gpu_sample read overflow: {e}"))?
                            .to_vec(),
                        cand_ids: b
                            .cand_ids
                            .as_slice::<u32>()
                            .map_err(|e| anyhow::anyhow!("gpu_sample read cand_ids: {e}"))?
                            .to_vec(),
                        cap: b.cap,
                    }),
                    None => None,
                };
                host_phases::add(
                    host_phases::Phase::LmheadReadback,
                    _hp.elapsed().as_nanos() as u64,
                );
                *head_out = Some(BatchedHeadOut {
                    logits,
                    normed,
                    gpu_sample,
                });
                return Ok(Vec::new());
            }
        } else {
            for layer_idx in 0..num_layers {
                self.encode_one_layer_batched(
                    layer_idx,
                    &bufs,
                    n,
                    &positions_buf,
                    &slot_id_buf,
                    slot_ids,
                    seq_positions,
                    regime,
                    tq_scale_factor_d512,
                    tq_codebook_bits,
                    &mut s,
                    exec,
                    reg,
                )?;
                // ADR-040 §0.19 decode bisection (HF2Q_S019_CKSUM=1): checksum the
                // batched residual after each layer. finish()+restart commits this
                // layer's work so the host read is valid. Under the 2x-contention
                // gate (MAXTOK>1), the FIRST diverging layer = the cross-step/within-
                // step corrupted decode buffer's manifestation.
                if cksum_on {
                    s.finish()
                        .map_err(|e| anyhow::anyhow!("s019 dec finish L{layer_idx}: {e}"))?;
                    if let Ok(h) = bufs.hidden.as_slice::<f32>() {
                        let mut c: u64 = 0xcbf29ce484222325;
                        for &x in h[..(n * hs).min(h.len())].iter() {
                            c ^= x.to_bits() as u64;
                            c = c.wrapping_mul(0x100000001b3);
                        }
                        eprintln!("S019_DECHID L{layer_idx:02} cks={c:016x}");
                    }
                    if let Ok(p) = positions_buf.as_slice::<u32>() {
                        let mut c: u64 = 0xcbf29ce484222325;
                        for &x in p.iter() {
                            c ^= x as u64;
                            c = c.wrapping_mul(0x100000001b3);
                        }
                        eprintln!("S019_DECPOS L{layer_idx:02} cks={c:016x}");
                    }
                    if let Ok(sb) = slot_id_buf.as_slice::<u32>() {
                        let mut c: u64 = 0xcbf29ce484222325;
                        for &x in sb.iter() {
                            c ^= x as u64;
                            c = c.wrapping_mul(0x100000001b3);
                        }
                        eprintln!("S019_DECSLOT L{layer_idx:02} cks={c:016x}");
                    }
                    s = exec
                        .begin()
                        .map_err(|e| anyhow::anyhow!("s019 dec restart L{layer_idx}: {e}"))?;
                }
            }

            let _hp = std::time::Instant::now();
            s.finish()
                .map_err(|e| anyhow::anyhow!("body_batched session finish: {e}"))?;
            host_phases::add(
                host_phases::Phase::BodyWait,
                _hp.elapsed().as_nanos() as u64,
            );
        }

        // Final hidden rows [n, hidden] (pre-final-norm).
        let _hp = std::time::Instant::now();
        let out: &[f32] = bufs
            .hidden
            .as_slice()
            .map_err(|e| anyhow::anyhow!("body_batched read hidden: {e}"))?;
        let v = out[..n * hs].to_vec();
        host_phases::add(
            host_phases::Phase::BodyReadback,
            _hp.elapsed().as_nanos() as u64,
        );
        Ok(v)
    }
}
