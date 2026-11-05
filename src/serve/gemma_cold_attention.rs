//! Gemma 4 cold (cursor-0) prefill attention and O-projection: the one
//! per-sequence function used by a single-slot cold prefill and by every lane
//! of a cold multi-sequence prefill (ADR-063 Q4-GEMMA-COLD, decision D2).
//!
//! Row-parallel work (embedding, norms, Q/K/V projections, fused head norm and
//! RoPE, V norm, dense FFN, router, MoE, head) runs once over the concatenated
//! stream in `forward_prefill_batched`. This module runs, for each sequence,
//! the attention and O-projection the single-slot cold route runs for a prompt
//! of that sequence's length:
//!
//! * sliding layers: `permute_021_f32_to_f16` of the sequence's seq-major
//!   normed rows into sequence-local head-major F16 scratch at offset 0, the
//!   sequence's own `[rows, rows]` sliding mask (re-cast to F16 in this layer
//!   session, the ADR-040 §0.19 remask rule) and `blk`, F16 d256 flash
//!   attention, F16→BF16 cast;
//! * global layers: tensor-mm (`64 < rows <= 8192`, or `HF2Q_NO_FA=1` at
//!   `rows >= 32`) or F16 d512 flash attention otherwise, the predicate
//!   evaluated with the sequence's own row count;
//! * the O-projection at `m = rows`, writing the sequence's rows of the joint
//!   output.
//!
//! The joint fused head-norm/RoPE and V-norm kernels write the head-major
//! operands (`q_perm`, `k_perm`, `v_perm`, `q_perm_f32`) across all stream
//! rows, so a lane of a multi-sequence stream is a strided sub-block there.
//! A route that reads those operands (tensor-mm, or BF16 flash attention under
//! `HF2Q_FA_F16=0`) therefore re-runs the same fused dispatches per sequence
//! with `seq_len = rows` on views of the joint seq-major projection rows and
//! positions, writing sequence-local head-major operands at offset 0 (and the
//! sequence's own rows of the seq-major normed buffers, which the KV scatter
//! reads). No strided view, reinterpreting copy or block-diagonal mask is
//! involved.
//!
//! Everything that reaches an attention kernel is a function of the sequence
//! alone, with the same shapes and offset-0 operands as the single-slot call,
//! so a lane is bitwise equal to its own single-slot cold prefill whenever the
//! row-wise kernels are row-invariant (lanes above 32 rows; see
//! `gemma_warm_batch::MIN_SUFFIX_TOKENS`). A single sequence spanning the whole
//! stream binds the joint buffers themselves, so the single-slot cold route
//! issues exactly the dispatches it issued before this module existed.
use super::*;
use mlx_native::{GraphExecutor, GraphSession, KernelRegistry, MlxDevice};

use crate::serve::forward_mlx_shared::MlxQWeight;

/// ADR-040 §7.32K: upper bound of the global-layer tensor-mm (NO_FA) domain;
/// above it globals route through F16 D512 FA (O(seq) memory).
pub(super) const GLOBAL_NOFA_MAX_SEQ: usize = 8192;

/// The cold attention kernel choice, resolved once per forward from the
/// diagnostic environment and evaluated per sequence by row count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ColdRoute {
    /// `HF2Q_FA_F16` (default on): F16 flash attention.
    pub(crate) fa_f16: bool,
    /// `HF2Q_NO_FA` tri-state: explicit tensor-mm request for globals.
    pub(crate) no_fa: Option<bool>,
    /// `HF2Q_GLOBAL_FA=1`: flash attention for globals at any length.
    pub(crate) global_fa: bool,
}

impl ColdRoute {
    pub(crate) fn from_env() -> Self {
        Self::from_values(
            std::env::var("HF2Q_FA_F16").ok().as_deref(),
            std::env::var("HF2Q_NO_FA").ok().as_deref(),
            std::env::var("HF2Q_GLOBAL_FA").ok().as_deref(),
        )
    }

    pub(crate) fn from_values(
        fa_f16: Option<&str>,
        no_fa: Option<&str>,
        global_fa: Option<&str>,
    ) -> Self {
        Self {
            fa_f16: !matches!(fa_f16, Some("0") | Some("false") | Some("off")),
            no_fa: match no_fa {
                Some("0") | Some("false") | Some("off") => Some(false),
                Some("1") | Some("true") | Some("on") => Some(true),
                _ => None,
            },
            global_fa: global_fa == Some("1"),
        }
    }

    /// `HF2Q_NO_FA=1`, honored only at `rows >= 32` (the dense tensor-mm
    /// tile reduction is 32-aligned).
    pub(crate) fn no_fa(self, rows: usize) -> bool {
        self.no_fa == Some(true) && rows >= 32
    }

    /// ADR-040 §0.19 / §7.32K seq-bounded global routing: tensor-mm for
    /// `64 < rows <= 8192` unless `HF2Q_GLOBAL_FA=1`.
    pub(crate) fn force_global_nofa(self, rows: usize) -> bool {
        !self.global_fa && rows > 64 && rows <= GLOBAL_NOFA_MAX_SEQ
    }

    /// Whether a cold global layer of `rows` rows runs tensor-mm attention.
    pub(crate) fn global_tensor_mm(self, rows: usize) -> bool {
        self.no_fa(rows) || self.force_global_nofa(rows)
    }

    /// Whether this layer's attention reads the head-major operands the
    /// fused head-norm/RoPE and V-norm kernels write.
    pub(crate) fn reads_head_major(self, is_sliding: bool, rows: usize) -> bool {
        (!is_sliding && self.global_tensor_mm(rows)) || !self.fa_f16
    }
}

/// One sequence's square masks and tile-skip bytes, built once per forward
/// in the setup session.
pub(super) struct ColdLaneMasks {
    pub(super) sliding: MlxBuffer,
    pub(super) sliding_f16: Option<MlxBuffer>,
    pub(super) global: MlxBuffer,
    pub(super) global_f16: Option<MlxBuffer>,
    pub(super) blk_sliding: MlxBuffer,
    pub(super) blk_global: MlxBuffer,
}

/// One sequence of the cold stream: rows `[offset, offset + rows)`.
pub(super) struct ColdSeq<'a> {
    pub(super) offset: usize,
    pub(super) rows: usize,
    pub(super) masks: &'a ColdLaneMasks,
}

/// The layer's stream buffers and parameters.
pub(super) struct ColdLayer<'a> {
    pub(super) layer_idx: usize,
    pub(super) nh: usize,
    pub(super) nkv: usize,
    pub(super) hd: usize,
    pub(super) is_sliding: bool,
    pub(super) route: ColdRoute,
    pub(super) profile_buckets: bool,
    /// Rows of the whole stream.
    pub(super) total_rows: usize,
    /// Joint seq-major projection rows (inputs of the fused head norm).
    pub(super) q: &'a MlxBuffer,
    pub(super) k: &'a MlxBuffer,
    pub(super) v_input: &'a MlxBuffer,
    /// Joint seq-major normed rows.
    pub(super) q_normed: &'a MlxBuffer,
    pub(super) k_normed: &'a MlxBuffer,
    pub(super) v_normed: &'a MlxBuffer,
    pub(super) positions: &'a MlxBuffer,
    pub(super) q_norm_weight: &'a MlxBuffer,
    pub(super) k_norm_weight: &'a MlxBuffer,
    pub(super) freq_factors: Option<&'a MlxBuffer>,
    pub(super) hd_norm_params: &'a MlxBuffer,
    pub(super) half_rope: u32,
    pub(super) eps: f32,
    pub(super) theta: f32,
    pub(super) o_proj: &'a MlxQWeight,
    /// Joint seq-major O-projection output `[total_rows, hidden]`.
    pub(super) attn_out: &'a MlxBuffer,
    pub(super) hidden_size: usize,
}

/// Attention scratch. Head-major buffers are addressed from offset 0 per
/// sequence; `sdpa_out` is the joint seq-major `[total_rows, nh * hd]` F32.
pub(super) struct ColdScratch<'a> {
    pub(super) q_perm: &'a MlxBuffer,
    pub(super) k_perm: &'a MlxBuffer,
    pub(super) v_perm: &'a MlxBuffer,
    pub(super) sdpa_out_perm: &'a MlxBuffer,
    pub(super) q_perm_f16: Option<&'a MlxBuffer>,
    pub(super) k_perm_f16: Option<&'a MlxBuffer>,
    pub(super) v_perm_f16: Option<&'a MlxBuffer>,
    pub(super) sdpa_out_perm_f16: Option<&'a MlxBuffer>,
    pub(super) q_perm_f32: Option<&'a MlxBuffer>,
    pub(super) kq: Option<&'a MlxBuffer>,
    pub(super) v_perm_t: Option<&'a MlxBuffer>,
    pub(super) attn_f32: Option<&'a MlxBuffer>,
    pub(super) sdpa_out: &'a MlxBuffer,
}

/// Rows `[offset, offset + rows)` of a seq-major buffer with `row_elems`
/// elements of `elem_bytes` each. A sequence spanning the whole stream binds
/// the buffer itself.
fn rows_view(
    buffer: &MlxBuffer,
    offset: usize,
    rows: usize,
    total_rows: usize,
    row_elems: usize,
    elem_bytes: usize,
) -> MlxBuffer {
    if offset == 0 && rows == total_rows {
        buffer.clone()
    } else {
        buffer.slice_view((offset * row_elems * elem_bytes) as u64, rows * row_elems)
    }
}

fn fnv_f16(buffer: &MlxBuffer) -> Option<u64> {
    let values = buffer.as_slice::<half::f16>().ok()?;
    let mut c: u64 = 0xcbf29ce484222325;
    for &x in values.iter() {
        c ^= x.to_bits() as u64;
        c = c.wrapping_mul(0x100000001b3);
    }
    Some(c)
}

/// Encode one layer's cold attention and O-projection for every sequence,
/// in order. Returns the (possibly re-opened) session.
#[allow(clippy::too_many_lines)]
pub(super) fn encode_cold_layer_attention<'e, 'a>(
    mut s: GraphSession<'e>,
    exec: &'e GraphExecutor,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    layer: &ColdLayer<'_>,
    scratch: &ColdScratch<'_>,
    seqs: impl IntoIterator<Item = ColdSeq<'a>>,
) -> Result<GraphSession<'e>> {
    let metal_dev = dev.metal_device();
    let profile_buckets_on = layer.profile_buckets;
    let layer_idx = layer.layer_idx;
    let (nh, nkv, hd) = (layer.nh, layer.nkv, layer.hd);
    let is_sliding = layer.is_sliding;
    let route = layer.route;
    let total = layer.total_rows;
    let seqs: Vec<ColdSeq<'a>> = seqs.into_iter().collect();
    let joint_o_proj = stream_o_projection(layer, &seqs);
    for seq in seqs {
        let seq_len = seq.rows;
        let off = seq.offset;
        let masks = seq.masks;
        anyhow::ensure!(
            seq_len > 0 && off + seq_len <= total,
            "cold attention L{layer_idx}: sequence rows [{off}, {}) outside stream {total}",
            off + seq_len
        );
        let route_through_nofa = !is_sliding && route.global_tensor_mm(seq_len);
        let q_normed = rows_view(layer.q_normed, off, seq_len, total, nh * hd, 4);
        let k_normed = rows_view(layer.k_normed, off, seq_len, total, nkv * hd, 4);
        let v_normed = rows_view(layer.v_normed, off, seq_len, total, nkv * hd, 4);
        let sdpa_out = rows_view(scratch.sdpa_out, off, seq_len, total, nh * hd, 4);
        let attn_out = rows_view(layer.attn_out, off, seq_len, total, layer.hidden_size, 4);
        let pf_q_perm = scratch.q_perm;
        let pf_k_perm = scratch.k_perm;
        let pf_v_perm = scratch.v_perm;
        let pf_sdpa_out_perm = scratch.sdpa_out_perm;

        // A sequence that is a strided sub-block of the joint head-major
        // operands re-runs the single-slot fused dispatches at
        // `seq_len = rows`, writing sequence-local operands at offset 0.
        if seq_len != total && route.reads_head_major(is_sliding, seq_len) {
            let q_in = rows_view(layer.q, off, seq_len, total, nh * hd, 4);
            let k_in = rows_view(layer.k, off, seq_len, total, nkv * hd, 4);
            let v_in = rows_view(layer.v_input, off, seq_len, total, nkv * hd, 4);
            let positions = rows_view(layer.positions, off, seq_len, total, 1, 4);
            let q_perm_f32 = scratch.q_perm_f32.filter(|_| route_through_nofa);
            let mut writes = vec![&q_normed, &k_normed, pf_q_perm, pf_k_perm];
            if let Some(f32_perm) = q_perm_f32 {
                writes.push(f32_perm);
            }
            s.barrier_between(&[&q_in, &k_in, &positions], &writes);
            mlx_native::ops::fused_head_norm_rope::
                dispatch_fused_head_norm_rope_batch_f32_with_bf16_f32_perm(
                s.encoder_mut(), reg, metal_dev,
                &q_in,
                &q_normed,
                Some(pf_q_perm),
                q_perm_f32,
                Some(layer.q_norm_weight),
                &positions,
                layer.freq_factors,
                nh as u32, hd as u32, layer.half_rope,
                seq_len as u32,
                layer.eps, layer.theta,
                true,
            ).map_err(|e| anyhow::anyhow!("cold lane Q norm+RoPE L{layer_idx}: {e}"))?;
            mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_batch_f32_with_bf16(
                s.encoder_mut(), reg, metal_dev,
                &k_in,
                &k_normed,
                Some(pf_k_perm),
                Some(layer.k_norm_weight),
                &positions,
                layer.freq_factors,
                nkv as u32, hd as u32, layer.half_rope,
                seq_len as u32,
                layer.eps, layer.theta,
                true,
            ).map_err(|e| anyhow::anyhow!("cold lane K norm+RoPE L{layer_idx}: {e}"))?;
            s.barrier_between(&[&v_in], &[&v_normed, pf_v_perm]);
            dispatch_rms_norm_unit_perhead_dual_perm(
                s.encoder_mut(),
                reg,
                metal_dev,
                &v_in,
                &v_normed,
                pf_v_perm,
                layer.hd_norm_params,
                nkv as u32,
                seq_len as u32,
                hd as u32,
            )?;
        }

        if route_through_nofa {
            // ---- tensor-mm attention (ADR-040 §0.19 / §7.32K globals) ----
            //   1. Q f32 head-major (co-written by the head-norm+RoPE kernel)
            //   2. Q @ K^T -> kq f32  [nh, seq, seq]
            //   3. scale-mask-softmax -> kq in-place (Gemma 4 has no Q·K
            //      softcap; scale = 1.0, Q is pre-scaled upstream)
            //   4. transpose_last2 V bf16 -> V_t bf16 [nkv, hd, seq]
            //   5. scores @ V_t -> attn f32 [nh, seq, hd]
            //   6. permute_021 attn -> sdpa_out f32 [seq, nh, hd]
            let pf_q_perm_f32_ref = scratch
                .q_perm_f32
                .expect("pf_q_perm_f32 must be allocated for tensor-mm attention");
            let pf_kq_ref = scratch
                .kq
                .expect("pf_kq must be allocated for tensor-mm attention");
            let pf_v_perm_t_ref = scratch
                .v_perm_t
                .expect("pf_v_perm_t must be allocated for tensor-mm attention");
            let pf_attn_f32_ref = scratch
                .attn_f32
                .expect("pf_attn_f32 must be allocated for tensor-mm attention");

            let nofa_qk_t0 = if profile_buckets_on {
                s.finish()
                    .map_err(|e| anyhow::anyhow!("noFA QK pre-finish L{layer_idx}: {e}"))?;
                let t0 = std::time::Instant::now();
                s = exec
                    .begin()
                    .map(crate::inference::native_observation::session)
                    .map_err(|e| anyhow::anyhow!("noFA QK begin L{layer_idx}: {e}"))?;
                Some(t0)
            } else {
                None
            };
            s.barrier_between(&[pf_k_perm, pf_q_perm_f32_ref], &[pf_kq_ref]);
            mlx_native::ops::dense_mm_bf16::dense_matmul_bf16_f32_tensor(
                s.encoder_mut(),
                reg,
                dev,
                pf_k_perm,
                pf_q_perm_f32_ref,
                pf_kq_ref,
                &mlx_native::ops::dense_mm_bf16::DenseMmBf16F32Params {
                    m: seq_len as u32,
                    n: seq_len as u32,
                    k: hd as u32,
                    src0_batch: nkv as u32,
                    src1_batch: nh as u32,
                },
            )
            .map_err(|e| anyhow::anyhow!("non-FA Q@K^T L{layer_idx}: {e}"))?;
            if let Some(t0) = nofa_qk_t0 {
                s.finish()
                    .map_err(|e| anyhow::anyhow!("noFA QK post-finish L{layer_idx}: {e}"))?;
                let dt_ns = t0.elapsed().as_nanos() as u64;
                PROFILE_B_NOFA_QK_NS.fetch_add(dt_ns, Ordering::Relaxed);
                PROFILE_B_NOFA_QK_COUNT.fetch_add(1, Ordering::Relaxed);
                s = exec
                    .begin()
                    .map(crate::inference::native_observation::session)
                    .map_err(|e| anyhow::anyhow!("noFA QK reopen L{layer_idx}: {e}"))?;
            }

            let nofa_sms_t0 = if profile_buckets_on {
                s.finish()
                    .map_err(|e| anyhow::anyhow!("noFA SMS pre-finish L{layer_idx}: {e}"))?;
                let t0 = std::time::Instant::now();
                s = exec
                    .begin()
                    .map(crate::inference::native_observation::session)
                    .map_err(|e| anyhow::anyhow!("noFA SMS begin L{layer_idx}: {e}"))?;
                Some(t0)
            } else {
                None
            };
            let mask_ref = if is_sliding {
                &masks.sliding
            } else {
                &masks.global
            };
            s.barrier_between(&[pf_kq_ref, mask_ref], &[pf_kq_ref]);
            mlx_native::ops::scale_mask_softmax::dispatch_scale_mask_softmax_f32(
                s.encoder_mut(),
                reg,
                dev,
                pf_kq_ref,
                pf_kq_ref,
                mask_ref,
                &mlx_native::ops::scale_mask_softmax::ScaleMaskSoftmaxParams {
                    rows: (nh * seq_len) as u32,
                    cols: seq_len as u32,
                    seq_q: seq_len as u32,
                    scale: 1.0,
                },
            )
            .map_err(|e| anyhow::anyhow!("non-FA scale_mask_softmax L{layer_idx}: {e}"))?;
            if let Some(t0) = nofa_sms_t0 {
                s.finish()
                    .map_err(|e| anyhow::anyhow!("noFA SMS post-finish L{layer_idx}: {e}"))?;
                let dt_ns = t0.elapsed().as_nanos() as u64;
                PROFILE_B_NOFA_SMS_NS.fetch_add(dt_ns, Ordering::Relaxed);
                PROFILE_B_NOFA_SMS_COUNT.fetch_add(1, Ordering::Relaxed);
                s = exec
                    .begin()
                    .map(crate::inference::native_observation::session)
                    .map_err(|e| anyhow::anyhow!("noFA SMS reopen L{layer_idx}: {e}"))?;
            }

            let nofa_vtrans_t0 = if profile_buckets_on {
                s.finish()
                    .map_err(|e| anyhow::anyhow!("noFA Vtrans pre-finish L{layer_idx}: {e}"))?;
                let t0 = std::time::Instant::now();
                s = exec
                    .begin()
                    .map(crate::inference::native_observation::session)
                    .map_err(|e| anyhow::anyhow!("noFA Vtrans begin L{layer_idx}: {e}"))?;
                Some(t0)
            } else {
                None
            };
            s.barrier_between(&[pf_v_perm], &[pf_v_perm_t_ref]);
            mlx_native::ops::transpose::transpose_last2_bf16(
                s.encoder_mut(),
                reg,
                metal_dev,
                pf_v_perm,
                pf_v_perm_t_ref,
                nkv,
                seq_len,
                hd,
            )
            .map_err(|e| anyhow::anyhow!("non-FA V transpose L{layer_idx}: {e}"))?;
            if let Some(t0) = nofa_vtrans_t0 {
                s.finish()
                    .map_err(|e| anyhow::anyhow!("noFA Vtrans post-finish L{layer_idx}: {e}"))?;
                let dt_ns = t0.elapsed().as_nanos() as u64;
                PROFILE_B_NOFA_VTRANS_NS.fetch_add(dt_ns, Ordering::Relaxed);
                PROFILE_B_NOFA_VTRANS_COUNT.fetch_add(1, Ordering::Relaxed);
                s = exec
                    .begin()
                    .map(crate::inference::native_observation::session)
                    .map_err(|e| anyhow::anyhow!("noFA Vtrans reopen L{layer_idx}: {e}"))?;
            }

            let nofa_sv_t0 = if profile_buckets_on {
                s.finish()
                    .map_err(|e| anyhow::anyhow!("noFA SV pre-finish L{layer_idx}: {e}"))?;
                let t0 = std::time::Instant::now();
                s = exec
                    .begin()
                    .map(crate::inference::native_observation::session)
                    .map_err(|e| anyhow::anyhow!("noFA SV begin L{layer_idx}: {e}"))?;
                Some(t0)
            } else {
                None
            };
            s.barrier_between(&[pf_v_perm_t_ref, pf_kq_ref], &[pf_attn_f32_ref]);
            mlx_native::ops::dense_mm_bf16::dense_matmul_bf16_f32_tensor(
                s.encoder_mut(),
                reg,
                dev,
                pf_v_perm_t_ref,
                pf_kq_ref,
                pf_attn_f32_ref,
                &mlx_native::ops::dense_mm_bf16::DenseMmBf16F32Params {
                    m: seq_len as u32,
                    n: hd as u32,
                    k: seq_len as u32,
                    src0_batch: nkv as u32,
                    src1_batch: nh as u32,
                },
            )
            .map_err(|e| anyhow::anyhow!("non-FA scores@V L{layer_idx}: {e}"))?;
            if let Some(t0) = nofa_sv_t0 {
                s.finish()
                    .map_err(|e| anyhow::anyhow!("noFA SV post-finish L{layer_idx}: {e}"))?;
                let dt_ns = t0.elapsed().as_nanos() as u64;
                PROFILE_B_NOFA_SV_NS.fetch_add(dt_ns, Ordering::Relaxed);
                PROFILE_B_NOFA_SV_COUNT.fetch_add(1, Ordering::Relaxed);
                s = exec
                    .begin()
                    .map(crate::inference::native_observation::session)
                    .map_err(|e| anyhow::anyhow!("noFA SV reopen L{layer_idx}: {e}"))?;
            }

            let nofa_perm_t0 = if profile_buckets_on {
                s.finish()
                    .map_err(|e| anyhow::anyhow!("noFA perm pre-finish L{layer_idx}: {e}"))?;
                let t0 = std::time::Instant::now();
                s = exec
                    .begin()
                    .map(crate::inference::native_observation::session)
                    .map_err(|e| anyhow::anyhow!("noFA perm begin L{layer_idx}: {e}"))?;
                Some(t0)
            } else {
                None
            };
            s.barrier_between(&[pf_attn_f32_ref], &[&sdpa_out]);
            mlx_native::ops::transpose::permute_021_f32(
                s.encoder_mut(),
                reg,
                metal_dev,
                pf_attn_f32_ref,
                &sdpa_out,
                nh,
                seq_len,
                hd,
            )
            .map_err(|e| anyhow::anyhow!("non-FA attn permute L{layer_idx}: {e}"))?;
            if let Some(t0) = nofa_perm_t0 {
                s.finish()
                    .map_err(|e| anyhow::anyhow!("noFA perm post-finish L{layer_idx}: {e}"))?;
                let dt_ns = t0.elapsed().as_nanos() as u64;
                PROFILE_B_NOFA_PERM_NS.fetch_add(dt_ns, Ordering::Relaxed);
                PROFILE_B_NOFA_PERM_COUNT.fetch_add(1, Ordering::Relaxed);
                s = exec
                    .begin()
                    .map(crate::inference::native_observation::session)
                    .map_err(|e| anyhow::anyhow!("noFA perm reopen L{layer_idx}: {e}"))?;
            }
        } else {
            s.barrier_between(&[pf_q_perm, pf_k_perm, pf_v_perm], &[pf_sdpa_out_perm]);
            // Wave P4.0 — env-gated FA isolation for per-kernel timing.
            let profile_fa = std::env::var("HF2Q_PROFILE_FA").is_ok() || profile_buckets_on;
            let fa_start = if profile_fa {
                s.finish()
                    .map_err(|e| anyhow::anyhow!("FA-profile pre-finish L{layer_idx}: {e}"))?;
                let t0 = std::time::Instant::now();
                s = exec
                    .begin()
                    .map(crate::inference::native_observation::session)
                    .map_err(|e| anyhow::anyhow!("FA-profile begin L{layer_idx}: {e}"))?;
                Some(t0)
            } else {
                None
            };
            let (mask, mask_f16, blk) = if is_sliding {
                (
                    &masks.sliding,
                    masks.sliding_f16.as_ref(),
                    &masks.blk_sliding,
                )
            } else {
                (&masks.global, masks.global_f16.as_ref(), &masks.blk_global)
            };
            if route.fa_f16 {
                // ADR-032: F16 flash attention. F16 Q/K/V are sourced from
                // the F32 seq-major normed rows via permute_021_f32_to_f16
                // (one rounding step) into head-major scratch at offset 0.
                let q_f16 = scratch
                    .q_perm_f16
                    .ok_or_else(|| anyhow::anyhow!("FA_F16: pf_q_perm_f16 not allocated"))?;
                let k_f16 = scratch
                    .k_perm_f16
                    .ok_or_else(|| anyhow::anyhow!("FA_F16: pf_k_perm_f16 not allocated"))?;
                let v_f16 = scratch
                    .v_perm_f16
                    .ok_or_else(|| anyhow::anyhow!("FA_F16: pf_v_perm_f16 not allocated"))?;
                let out_f16 = scratch
                    .sdpa_out_perm_f16
                    .ok_or_else(|| anyhow::anyhow!("FA_F16: pf_sdpa_out_perm_f16 not allocated"))?;
                let mask_f16 =
                    mask_f16.ok_or_else(|| anyhow::anyhow!("FA_F16: F16 mask not allocated"))?;
                let q_elems = nh * seq_len * hd;
                let kind = if is_sliding { "" } else { "global " };

                s.barrier_between(&[&q_normed], &[q_f16]);
                mlx_native::ops::transpose::permute_021_f32_to_f16(
                    s.encoder_mut(),
                    reg,
                    metal_dev,
                    &q_normed,
                    q_f16,
                    seq_len,
                    nh,
                    hd,
                )
                .map_err(|e| anyhow::anyhow!("FA_F16 {kind}Q permute+cast L{layer_idx}: {e}"))?;
                s.barrier_between(&[&k_normed], &[k_f16]);
                mlx_native::ops::transpose::permute_021_f32_to_f16(
                    s.encoder_mut(),
                    reg,
                    metal_dev,
                    &k_normed,
                    k_f16,
                    seq_len,
                    nkv,
                    hd,
                )
                .map_err(|e| anyhow::anyhow!("FA_F16 {kind}K permute+cast L{layer_idx}: {e}"))?;
                s.barrier_between(&[&v_normed], &[v_f16]);
                mlx_native::ops::transpose::permute_021_f32_to_f16(
                    s.encoder_mut(),
                    reg,
                    metal_dev,
                    &v_normed,
                    v_f16,
                    seq_len,
                    nkv,
                    hd,
                )
                .map_err(|e| anyhow::anyhow!("FA_F16 {kind}V permute+cast L{layer_idx}: {e}"))?;

                // ADR-040 §0.19: checksum the ACTUAL sliding FA inputs.
                if is_sliding && std::env::var("HF2Q_S019_CKSUM").is_ok() {
                    if let Ok(s2) = exec
                        .begin()
                        .map(crate::inference::native_observation::session)
                    {
                        let _ = s2.finish();
                    }
                    for (buffer, name) in [
                        (q_f16, "FAQ16"),
                        (k_f16, "FAK16"),
                        (v_f16, "FAV16"),
                        (mask_f16, "FAMASK"),
                    ] {
                        if let Some(c) = fnv_f16(buffer) {
                            eprintln!("S019_{name} L{layer_idx:02} cks={c:016x}");
                        }
                    }
                    if let Ok(bytes) = blk.as_slice::<u8>() {
                        let mut c: u64 = 0xcbf29ce484222325;
                        for &x in bytes.iter() {
                            c ^= x as u64;
                            c = c.wrapping_mul(0x100000001b3);
                        }
                        eprintln!("S019_FABLK L{layer_idx:02} cks={c:016x}");
                    }
                }
                // ADR-040 §0.19 FIX (default-on; opt-out HF2Q_S019_NO_REMASK):
                // rebuild the F16 mask IN THIS LAYER'S SESSION right before
                // the FA reads it, from the intact bf16 source built in the
                // setup session. The per-session range tracker cannot bridge
                // the setup→layer cross-session hazard; this re-cast and the
                // FA barrier below are in one session, so it orders them.
                if std::env::var("HF2Q_S019_NO_REMASK").is_err() {
                    s.barrier_between(&[mask], &[mask_f16]);
                    mlx_native::ops::elementwise::cast(
                        s.encoder_mut(),
                        reg,
                        metal_dev,
                        mask,
                        mask_f16,
                        seq_len * seq_len,
                        mlx_native::ops::elementwise::CastDirection::BF16ToF16,
                    )
                    .map_err(|e| anyhow::anyhow!("§0.19 remask {kind}L{layer_idx}: {e}"))?;
                }
                s.barrier_between(&[q_f16, k_f16, v_f16, mask_f16, blk], &[out_f16]);
                if is_sliding {
                    mlx_native::ops::flash_attn_prefill::dispatch_flash_attn_prefill_f16_d256_with_blk(
                        s.encoder_mut(), dev, reg,
                        q_f16, k_f16, v_f16,
                        Some(mask_f16),
                        Some(blk),
                        out_f16,
                        &mlx_native::ops::flash_attn_prefill::FlashAttnPrefillParams {
                            n_heads: nh as u32,
                            n_kv_heads: nkv as u32,
                            head_dim: hd as u32,
                            seq_len_q: seq_len as u32,
                            seq_len_k: seq_len as u32,
                            batch: 1,
                            scale: 1.0,
                            do_causal: false,
                        },
                    ).map_err(|e| anyhow::anyhow!("FA_F16 sliding L{layer_idx}: {e}"))?;
                } else {
                    mlx_native::ops::flash_attn_prefill_d512::dispatch_flash_attn_prefill_f16_d512_with_blk(
                        s.encoder_mut(), dev, reg,
                        q_f16, k_f16, v_f16,
                        Some(mask_f16),
                        Some(blk),
                        out_f16,
                        &mlx_native::ops::flash_attn_prefill_d512::FlashAttnPrefillParams {
                            n_heads: nh as u32,
                            n_kv_heads: nkv as u32,
                            head_dim: hd as u32,
                            seq_len_q: seq_len as u32,
                            seq_len_k: seq_len as u32,
                            batch: 1,
                            scale: 1.0,
                            do_causal: false,
                        },
                    ).map_err(|e| anyhow::anyhow!("FA_F16 global D=512 L{layer_idx}: {e}"))?;
                }

                // Cast F16 output back to BF16 for the O-projection input.
                s.barrier_between(&[out_f16], &[pf_sdpa_out_perm]);
                mlx_native::ops::elementwise::cast(
                    s.encoder_mut(),
                    reg,
                    metal_dev,
                    out_f16,
                    pf_sdpa_out_perm,
                    q_elems,
                    mlx_native::ops::elementwise::CastDirection::F16ToBF16,
                )
                .map_err(|e| anyhow::anyhow!("FA_F16 {kind}O cast L{layer_idx}: {e}"))?;
            } else if is_sliding {
                // HF2Q_FA_F16=0: legacy BF16 instantiation (diagnostic A/B).
                mlx_native::ops::flash_attn_prefill::dispatch_flash_attn_prefill_bf16_d256_with_blk(
                    s.encoder_mut(), dev, reg,
                    pf_q_perm, pf_k_perm, pf_v_perm,
                    Some(mask),
                    Some(blk),
                    pf_sdpa_out_perm,
                    &mlx_native::ops::flash_attn_prefill::FlashAttnPrefillParams {
                        n_heads: nh as u32,
                        n_kv_heads: nkv as u32,
                        head_dim: hd as u32,
                        seq_len_q: seq_len as u32,
                        seq_len_k: seq_len as u32,
                        batch: 1,
                        scale: 1.0,
                        do_causal: false,
                    },
                ).map_err(|e| anyhow::anyhow!("batched sliding flash_attn_prefill L{layer_idx}: {e}"))?;
            } else {
                mlx_native::ops::flash_attn_prefill_d512::dispatch_flash_attn_prefill_bf16_d512_with_blk(
                    s.encoder_mut(), dev, reg,
                    pf_q_perm, pf_k_perm, pf_v_perm,
                    Some(mask),
                    Some(blk),
                    pf_sdpa_out_perm,
                    &mlx_native::ops::flash_attn_prefill_d512::FlashAttnPrefillParams {
                        n_heads: nh as u32,
                        n_kv_heads: nkv as u32,
                        head_dim: hd as u32,
                        seq_len_q: seq_len as u32,
                        seq_len_k: seq_len as u32,
                        batch: 1,
                        scale: 1.0,
                        do_causal: false,
                    },
                ).map_err(|e| anyhow::anyhow!("batched global flash_attn_prefill L{layer_idx}: {e}"))?;
            }
            if let Some(t0) = fa_start {
                if is_sliding {
                    bucket_finish!(
                        s,
                        exec,
                        t0,
                        &PROFILE_FA_SW_NS,
                        &PROFILE_FA_SW_COUNT,
                        1,
                        "fa_sw"
                    );
                } else {
                    bucket_finish!(
                        s,
                        exec,
                        t0,
                        &PROFILE_FA_GL_NS,
                        &PROFILE_FA_GL_COUNT,
                        1,
                        "fa_gl"
                    );
                }
            }
        }

        if joint_o_proj {
            // Stream O-projection below: leave this lane's attention output
            // seq-major F32 in its rows of `sdpa_out`, exactly the
            // intermediate `dispatch_qmatmul_head_major_bf16` builds for a
            // codec without a perm021 kernel (tensor-mm lanes already wrote
            // it).
            if !route_through_nofa {
                s.barrier_between(&[pf_sdpa_out_perm], &[&sdpa_out]);
                mlx_native::ops::transpose::permute_021_bf16_to_f32(
                    s.encoder_mut(),
                    reg,
                    metal_dev,
                    pf_sdpa_out_perm,
                    &sdpa_out,
                    nh,
                    seq_len,
                    hd,
                )
                .map_err(|e| anyhow::anyhow!("cold lane attention permute L{layer_idx}: {e}"))?;
            }
            continue;
        }
        // O-projection (m = rows): tensor-mm attention wrote seq-major F32;
        // flash attention wrote head-major BF16, read directly by the
        // perm021 tensor-MM variant (or permuted for other stored codecs).
        s = encode_o_projection(s, exec, reg, layer, |s, reg| {
            if route_through_nofa {
                s.barrier_between(&[&sdpa_out, &layer.o_proj.buffer], &[&attn_out]);
                dispatch_qmatmul(
                    s,
                    reg,
                    dev,
                    &sdpa_out,
                    layer.o_proj,
                    &attn_out,
                    seq_len as u32,
                    crate::quantize::imatrix::ImatrixHint::Layered {
                        tag: "attn_output",
                        layer: layer_idx,
                    },
                )
            } else {
                crate::serve::forward_mlx_shared::dispatch_qmatmul_head_major_bf16(
                    s,
                    reg,
                    dev,
                    pf_sdpa_out_perm,
                    &sdpa_out,
                    layer.o_proj,
                    &attn_out,
                    seq_len as u32,
                    nh,
                    hd,
                    crate::quantize::imatrix::ImatrixHint::Layered {
                        tag: "attn_output",
                        layer: layer_idx,
                    },
                )
                .map(|_| ())
                .map_err(|error| {
                    anyhow::anyhow!("batched native O-projection L{layer_idx}: {error}")
                })
            }
        })?;
    }
    if joint_o_proj {
        // One O-projection over every stream row (m = total): the ordinary
        // native route each lane's own projection takes, row-invariant for
        // m > 8 (the same property the joint Q/K/V projections rely on).
        s = encode_o_projection(s, exec, reg, layer, |s, reg| {
            s.barrier_between(&[scratch.sdpa_out, &layer.o_proj.buffer], &[layer.attn_out]);
            dispatch_qmatmul(
                s,
                reg,
                dev,
                scratch.sdpa_out,
                layer.o_proj,
                layer.attn_out,
                total as u32,
                crate::quantize::imatrix::ImatrixHint::Layered {
                    tag: "attn_output",
                    layer: layer_idx,
                },
            )
        })?;
    }
    Ok(s)
}

/// Dense native matmuls take the `mm` kernels for every `m > 8`, whose
/// per-row result does not depend on `m`.
const O_PROJ_ROW_INVARIANT_MIN_ROWS: usize = 9;

/// Whether one O-projection over the whole stream equals each lane's own:
/// two or more lanes, every lane on the ordinary native route (no perm021
/// kernel for the codec, no affine weight), and every lane and the stream
/// above 8 rows.
fn stream_o_projection(layer: &ColdLayer<'_>, seqs: &[ColdSeq<'_>]) -> bool {
    let ordinary_route = layer.o_proj.affine.is_none()
        && !crate::serve::native_projection::supports_native_perm021(
            layer.o_proj,
            layer.total_rows as u32,
            layer.hd as u32,
        );
    seqs.len() > 1
        && ordinary_route
        && seqs
            .iter()
            .all(|seq| seq.rows >= O_PROJ_ROW_INVARIANT_MIN_ROWS)
}

/// Encode one O-projection, bracketed by the `HF2Q_PROFILE_MM` bucket.
fn encode_o_projection<'e>(
    mut s: GraphSession<'e>,
    exec: &'e GraphExecutor,
    reg: &mut KernelRegistry,
    layer: &ColdLayer<'_>,
    encode: impl FnOnce(&mut GraphSession<'e>, &mut KernelRegistry) -> Result<()>,
) -> Result<GraphSession<'e>> {
    let layer_idx = layer.layer_idx;
    let o_t0 = if std::env::var("HF2Q_PROFILE_MM").is_ok() || layer.profile_buckets {
        s.finish()
            .map_err(|e| anyhow::anyhow!("MM-profile pre-finish (O) L{layer_idx}: {e}"))?;
        let t0 = std::time::Instant::now();
        s = exec
            .begin()
            .map(crate::inference::native_observation::session)
            .map_err(|e| anyhow::anyhow!("MM-profile begin (O) L{layer_idx}: {e}"))?;
        Some(t0)
    } else {
        None
    };
    encode(&mut s, reg)?;
    if let Some(t0) = o_t0 {
        bucket_finish!(
            s,
            exec,
            t0,
            &PROFILE_O_MM_NS,
            &PROFILE_O_MM_COUNT,
            1,
            "O_mm"
        );
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::ColdRoute;

    #[test]
    fn gemma_cold_attention_route_matches_single_slot_predicates() {
        let default = ColdRoute::from_values(None, None, None);
        assert!(default.fa_f16);
        // seq <= 64: F16 FA globals; 64 < seq <= 8192: tensor-mm; above: FA.
        assert!(!default.global_tensor_mm(33));
        assert!(!default.global_tensor_mm(64));
        assert!(default.global_tensor_mm(65));
        assert!(default.global_tensor_mm(458));
        assert!(default.global_tensor_mm(8192));
        assert!(!default.global_tensor_mm(8193));
        // Sliding layers never take tensor-mm; head-major operands are read
        // only by tensor-mm globals on the default F16 route.
        assert!(!default.reads_head_major(true, 458));
        assert!(default.reads_head_major(false, 458));
        assert!(!default.reads_head_major(false, 64));

        let global_fa = ColdRoute::from_values(None, None, Some("1"));
        assert!(!global_fa.global_tensor_mm(458));
        let no_fa = ColdRoute::from_values(None, Some("1"), Some("1"));
        assert!(no_fa.global_tensor_mm(32));
        assert!(!no_fa.global_tensor_mm(31));
        let bf16 = ColdRoute::from_values(Some("0"), None, None);
        assert!(!bf16.fa_f16);
        assert!(bf16.reads_head_major(true, 33));
    }
}
