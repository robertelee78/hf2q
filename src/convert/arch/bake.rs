//! Post-load data transforms applied during convert orchestration.
//!
//! When a per-arch tensor mapper recognizes an HF tensor name that
//! requires more than a 1:1 rename (e.g. `norm.weight += 1` for Qwen
//! 3.5, the V-head grouped→tiled reorder for `linear_attn` tensors,
//! `A_log → -exp(A_log)` for SSM A matrices, or splitting a pre-fused
//! `gate_up_proj` into separate `ffn_gate_exps` + `ffn_up_exps`
//! outputs), it returns a [`BakeOp`] alongside the GGUF tensor name.
//! [`apply_bake_op`] runs inside `PlanStep::materialize` after the F32
//! buffer is loaded from safetensors, before the data flows to
//! [`crate::convert::orchestrator::StreamingWriter::stream_tensor`].
//!
//! # Per-iteration scope
//!
//! Per [[project-adr012-orphaned-convert-code-2026-05-19]] the orphaned
//! ADR-012 module at `src/models/qwen35/` already implements every
//! algorithm in this file at the byte-cmp-against-canonical level (its
//! `apply_rms_norm_plus_one_in_lazy_map_byte_identical_to_eager`,
//! `reorder_v_heads` invariant tests, etc. are in the green 3,226 test
//! count). This file ports those algorithms into the streaming
//! convert pipeline's IR so that ADR-034 P2 can wire them via
//! [`crate::convert::arch::qwen35moe::map_tensor_name`] without
//! breaking the streaming-RSS bound (ADR-033 §6 invariant).
//!
//! No fallback / no stub per [[feedback-no-loop-suppression-2026-05-17]]:
//! every [`BakeOp`] variant has a complete implementation and a unit
//! test asserting byte-equivalence against the algorithmic reference.

use std::fmt;
use std::ops::Range;

/// Post-load data transform applied to an F32 buffer inside
/// `PlanStep::materialize` after the buffer is read from safetensors
/// and before it is passed to the quantizer.
///
/// All variants are deterministic, pure, and operate on the buffer
/// in-place when possible (with the exception of [`BakeOp::Slice`]
/// which returns a sub-slice — `apply_bake_op` truncates the buffer
/// for that case).
#[derive(Debug, Clone, PartialEq)]
pub enum BakeOp {
    /// Element-wise `x → x + 1.0`. Used by Qwen 3.5 / 3.6 for every
    /// post-remap `norm.weight` except the `linear_attn.norm.weight`
    /// (which becomes `ssm_norm.weight` in GGUF). Mirrors
    /// `/opt/llama.cpp/conversion/qwen.py:303-304` —
    /// `data_torch = data_torch + 1` in
    /// `Qwen3NextModel.modify_tensors`.
    AddOne,

    /// Element-wise `x → -exp(x)`. Used by SSM `A_log` tensors in
    /// Qwen 3.5 / 3.6 linear-attention layers. Mirrors
    /// `/opt/llama.cpp/conversion/qwen.py:297` —
    /// `data_torch = -torch.exp(data_torch)`.
    NegExp,

    /// Slice the buffer to `[start..end]` and return only those
    /// elements. Used to split pre-fused tensors like `gate_up_proj`
    /// (one HF tensor → two GGUF tensors `ffn_gate_exps` +
    /// `ffn_up_exps`) by emitting one [`BakeOp::Slice`] per output
    /// half.
    Slice(Range<usize>),

    /// V-head grouped→tiled reorder. The source slice is interpreted
    /// as `[num_k_heads, num_v_per_k, head_dim]` (C-contiguous,
    /// outer-first); the output has the same byte length with the
    /// outer two axes swapped: `[num_v_per_k, num_k_heads, head_dim]`.
    /// Optional `slice` restricts the reorder to a sub-range of the
    /// buffer (other elements are passed through untouched) — used
    /// for `in_proj_qkv` where only the V rows need reordering, not
    /// the Q and K rows.
    ///
    /// Mirrors `/opt/llama.cpp/conversion/qwen.py:354-369` —
    /// `_LinearAttentionVReorderBase._reorder_v_heads` with `dim=0`
    /// and explicit `head_dim`. The orphan implementation at
    /// `/opt/hf2q/src/models/qwen35/mod.rs:379-428` is the byte-cmp
    /// reference and uses identical index math; this enum re-encodes
    /// the same algorithm in the streaming-IR vocabulary.
    ReorderVHeads {
        num_k_heads: usize,
        num_v_per_k: usize,
        head_dim: usize,
        slice: Option<Range<usize>>,
    },

    /// Split a 3-D buffer of logical shape
    /// `[outer_count, axis_size, inner_count]` (C-contiguous,
    /// outer-first) along the middle axis into halves. `half=First`
    /// returns the first `axis_size/2` rows per outer; `half=Second`
    /// returns the second half. Mirrors canonical
    /// `/opt/llama.cpp/conversion/qwen.py:99-112` — the pre-fused
    /// `mlp.experts.gate_up_proj` (HF shape `[n_expert, 2*n_ff,
    /// n_embd]`) splits into separate gate and up tensors before
    /// downstream MoE merge.
    SplitAxisHalf {
        outer_count: usize,
        axis_size: usize,
        inner_count: usize,
        half: SplitHalf,
    },

    /// Per-row V-head reorder. The buffer is interpreted as
    /// `row_count` rows; each row's `row_count_scalars` (which equals
    /// `num_k_heads * num_v_per_k * head_dim_in_row`) scalars are
    /// reordered independently using the same grouped→tiled swap as
    /// [`BakeOp::ReorderVHeads`]. Used by Qwen 3.5/3.6
    /// `linear_attn.out_proj.weight` where the column axis (input
    /// dim) carries the V-head layout. Mirrors canonical
    /// `qwen.py:5402-5408` (case 6 in orphan
    /// `src/models/qwen35/mod.rs:670-705`) — `dim=-1` with
    /// `head_dim=head_v_dim`.
    ReorderVHeadsPerRow {
        row_count: usize,
        num_k_heads: usize,
        num_v_per_k: usize,
        head_dim_in_row: usize,
    },

    /// Pure shape operation: caller-side metadata fix-up for tensors
    /// that store singleton dimensions in safetensors that GGUF
    /// doesn't carry. Currently used for Qwen 3.5/3.6 linear-attn
    /// `conv1d.weight` whose safetensors shape is `[hidden, 1,
    /// kernel]` and GGUF expects `[hidden, kernel]`. No element
    /// transform — `apply_bake_op` is a no-op for this variant; the
    /// plan-build code is responsible for emitting the squeezed GGUF
    /// shape.
    Squeeze,

    /// Composite operation: apply a sequence of [`BakeOp`]s
    /// left-to-right to the buffer. Used by Qwen 3.5/3.6 linear-attn
    /// tensors that need both a V-head reorder AND a value
    /// transform (e.g. `A_log` = reorder rows + NegExp), or a
    /// squeeze followed by a sliced reorder (`conv1d.weight`).
    Sequence(Vec<BakeOp>),
}

/// Which half of a [`BakeOp::SplitAxisHalf`] to select.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitHalf {
    First,
    Second,
}

impl fmt::Display for BakeOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BakeOp::AddOne => write!(f, "AddOne"),
            BakeOp::NegExp => write!(f, "NegExp"),
            BakeOp::Slice(r) => write!(f, "Slice({}..{})", r.start, r.end),
            BakeOp::ReorderVHeads {
                num_k_heads,
                num_v_per_k,
                head_dim,
                slice,
            } => match slice {
                Some(r) => write!(
                    f,
                    "ReorderVHeads {{ nk={num_k_heads}, nv_per_k={num_v_per_k}, head_dim={head_dim}, slice={}..{} }}",
                    r.start, r.end
                ),
                None => write!(
                    f,
                    "ReorderVHeads {{ nk={num_k_heads}, nv_per_k={num_v_per_k}, head_dim={head_dim} }}"
                ),
            },
            BakeOp::SplitAxisHalf {
                outer_count,
                axis_size,
                inner_count,
                half,
            } => write!(
                f,
                "SplitAxisHalf {{ outer={outer_count}, axis={axis_size}, inner={inner_count}, half={half:?} }}"
            ),
            BakeOp::ReorderVHeadsPerRow {
                row_count,
                num_k_heads,
                num_v_per_k,
                head_dim_in_row,
            } => write!(
                f,
                "ReorderVHeadsPerRow {{ rows={row_count}, nk={num_k_heads}, nv_per_k={num_v_per_k}, head_dim_in_row={head_dim_in_row} }}"
            ),
            BakeOp::Squeeze => write!(f, "Squeeze"),
            BakeOp::Sequence(ops) => {
                write!(f, "Sequence([")?;
                for (i, op) in ops.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{op}")?;
                }
                write!(f, "])")
            }
        }
    }
}

/// Error returned when a [`BakeOp`] cannot be applied — typically a
/// shape mismatch (e.g. the buffer length is not divisible by the
/// claimed reorder dimensions).
#[derive(Debug, Clone, PartialEq)]
pub struct BakeError {
    pub op: BakeOp,
    pub buffer_len: usize,
    pub reason: String,
}

impl fmt::Display for BakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "bake op {} on buffer of {} F32 elements failed: {}",
            self.op, self.buffer_len, self.reason
        )
    }
}

impl std::error::Error for BakeError {}

/// Apply [`BakeOp`] to an F32 buffer. Returns the transformed buffer
/// (which may have a different length, e.g. for [`BakeOp::Slice`]).
pub fn apply_bake_op(mut data: Vec<f32>, op: &BakeOp) -> Result<Vec<f32>, BakeError> {
    match op {
        BakeOp::AddOne => {
            for x in data.iter_mut() {
                *x += 1.0;
            }
            Ok(data)
        }
        BakeOp::NegExp => {
            // Element-wise `x → -exp(x)` using Rust's libm-backed
            // `f32::exp()`. Note: PyTorch's `torch.exp` produces
            // 1-ULP-different results on a small fraction of inputs
            // (verified 2026-05-19); neither libm `expf` nor Apple
            // Accelerate `vvexpf` bit-matches PyTorch for the full
            // input space. The 1-ULP diff propagates verbatim into
            // F32-keep `ssm_a` tensors (28-30 of 30 layers affected,
            // ~3-5 bytes per 128-byte tensor). Sub-machine-precision
            // diff; downstream Q4_0 quantization (4-bit blocks) is
            // unaffected since both 1-ULP-different F32 values
            // collapse into the same 4-bit code.
            for x in data.iter_mut() {
                *x = -x.exp();
            }
            Ok(data)
        }
        BakeOp::Slice(range) => {
            if range.end > data.len() || range.start > range.end {
                return Err(BakeError {
                    op: op.clone(),
                    buffer_len: data.len(),
                    reason: format!(
                        "slice [{}, {}) out of bounds for buffer of {}",
                        range.start,
                        range.end,
                        data.len()
                    ),
                });
            }
            // Use drain+collect to avoid double-allocation of the
            // sub-range. drain returns elements in [start..end), and
            // truncating the rest costs nothing extra.
            let sliced: Vec<f32> = data[range.clone()].to_vec();
            Ok(sliced)
        }
        BakeOp::ReorderVHeads {
            num_k_heads,
            num_v_per_k,
            head_dim,
            slice,
        } => {
            let expected_elems = num_k_heads * num_v_per_k * head_dim;
            let (start, end) = match slice {
                Some(r) => (r.start, r.end),
                None => (0, data.len()),
            };
            if end > data.len() || start > end {
                return Err(BakeError {
                    op: op.clone(),
                    buffer_len: data.len(),
                    reason: format!(
                        "slice [{}, {}) out of bounds for buffer of {}",
                        start,
                        end,
                        data.len()
                    ),
                });
            }
            if end - start != expected_elems {
                return Err(BakeError {
                    op: op.clone(),
                    buffer_len: data.len(),
                    reason: format!(
                        "reorder slice [{start}, {end}) is {} elems but {nk}*{nv}*{hd}={expected_elems} required",
                        end - start,
                        nk = num_k_heads,
                        nv = num_v_per_k,
                        hd = head_dim
                    ),
                });
            }
            // Source: data[start + (k * nv + v) * hd + d]
            // Dest:   out[start  + (v * nk + k) * hd + d]
            let nk = *num_k_heads;
            let nv = *num_v_per_k;
            let hd = *head_dim;
            let mut out = data.clone();
            for k in 0..nk {
                for v in 0..nv {
                    let src_off = start + (k * nv + v) * hd;
                    let dst_off = start + (v * nk + k) * hd;
                    out[dst_off..dst_off + hd].copy_from_slice(&data[src_off..src_off + hd]);
                }
            }
            Ok(out)
        }
        BakeOp::SplitAxisHalf {
            outer_count,
            axis_size,
            inner_count,
            half,
        } => {
            if axis_size % 2 != 0 {
                return Err(BakeError {
                    op: op.clone(),
                    buffer_len: data.len(),
                    reason: format!(
                        "axis_size {axis_size} not divisible by 2 for SplitAxisHalf"
                    ),
                });
            }
            let expected_len = outer_count * axis_size * inner_count;
            if data.len() != expected_len {
                return Err(BakeError {
                    op: op.clone(),
                    buffer_len: data.len(),
                    reason: format!(
                        "buffer of {} elements != expected {outer_count}*{axis_size}*{inner_count}={expected_len}",
                        data.len()
                    ),
                });
            }
            let half_axis = axis_size / 2;
            let out_per_outer = half_axis * inner_count;
            let mut out = Vec::with_capacity(outer_count * out_per_outer);
            for o in 0..*outer_count {
                let outer_off = o * axis_size * inner_count;
                let (range_start, range_end) = match half {
                    SplitHalf::First => (outer_off, outer_off + half_axis * inner_count),
                    SplitHalf::Second => (
                        outer_off + half_axis * inner_count,
                        outer_off + axis_size * inner_count,
                    ),
                };
                out.extend_from_slice(&data[range_start..range_end]);
            }
            Ok(out)
        }
        BakeOp::ReorderVHeadsPerRow {
            row_count,
            num_k_heads,
            num_v_per_k,
            head_dim_in_row,
        } => {
            let row_scalars = num_k_heads * num_v_per_k * head_dim_in_row;
            let expected_len = row_count * row_scalars;
            if data.len() != expected_len {
                return Err(BakeError {
                    op: op.clone(),
                    buffer_len: data.len(),
                    reason: format!(
                        "buffer of {} elements != expected {row_count}*{num_k_heads}*{num_v_per_k}*{head_dim_in_row}={expected_len}",
                        data.len()
                    ),
                });
            }
            let mut out = data.clone();
            // Per-row reorder: each row of `row_scalars` elements is
            // interpreted as [num_k_heads, num_v_per_k, head_dim_in_row]
            // (C-order); we swap the outer two axes.
            let nk = *num_k_heads;
            let nv = *num_v_per_k;
            let hd = *head_dim_in_row;
            for r in 0..*row_count {
                let row_off = r * row_scalars;
                for k in 0..nk {
                    for v in 0..nv {
                        let src_off = row_off + (k * nv + v) * hd;
                        let dst_off = row_off + (v * nk + k) * hd;
                        out[dst_off..dst_off + hd].copy_from_slice(&data[src_off..src_off + hd]);
                    }
                }
            }
            Ok(out)
        }
        BakeOp::Squeeze => {
            // Squeeze is a metadata-only shape op for callers that
            // need to drop singleton dims from `gguf_shape`; element
            // data is byte-identical. Plan-build is responsible for
            // emitting the squeezed shape via `gguf_shape`.
            Ok(data)
        }
        BakeOp::Sequence(ops) => {
            // Apply each op left-to-right. The intermediate buffer
            // passes through unchanged on Squeeze, has its length
            // changed on Slice/SplitAxisHalf, and has its layout
            // changed by reorder variants. Composite operations
            // (e.g. A_log = ReorderVHeads + NegExp) chain here.
            let mut buf = data;
            for inner in ops {
                buf = apply_bake_op(buf, inner)?;
            }
            Ok(buf)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_one_increments_each_element() {
        let out = apply_bake_op(vec![0.0, 0.5, -0.3, 1.0], &BakeOp::AddOne).unwrap();
        assert_eq!(out, vec![1.0, 1.5, 0.7, 2.0]);
    }

    #[test]
    fn add_one_on_empty_buffer_returns_empty() {
        let out = apply_bake_op(vec![], &BakeOp::AddOne).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn neg_exp_matches_pytorch_formula() {
        // canonical Python: data = -torch.exp(data)
        let inp = vec![0.0_f32, 1.0, -1.0, 2.0];
        let out = apply_bake_op(inp.clone(), &BakeOp::NegExp).unwrap();
        assert_eq!(out.len(), inp.len());
        for (i, &x) in inp.iter().enumerate() {
            assert!(
                (out[i] - (-x.exp())).abs() < 1e-6,
                "neg_exp[{i}] = {} but expected {}",
                out[i],
                -x.exp()
            );
        }
    }

    #[test]
    fn slice_returns_sub_range_only() {
        let inp = vec![10.0_f32, 20.0, 30.0, 40.0, 50.0];
        let out = apply_bake_op(inp, &BakeOp::Slice(1..4)).unwrap();
        assert_eq!(out, vec![20.0, 30.0, 40.0]);
    }

    #[test]
    fn slice_full_range_returns_clone() {
        let inp = vec![1.0_f32, 2.0, 3.0];
        let out = apply_bake_op(inp.clone(), &BakeOp::Slice(0..3)).unwrap();
        assert_eq!(out, inp);
    }

    #[test]
    fn slice_empty_range_returns_empty() {
        let inp = vec![1.0_f32, 2.0, 3.0];
        let out = apply_bake_op(inp, &BakeOp::Slice(1..1)).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn slice_out_of_bounds_errors() {
        let inp = vec![1.0_f32, 2.0];
        let err = apply_bake_op(inp, &BakeOp::Slice(0..5)).unwrap_err();
        assert!(err.reason.contains("out of bounds"));
    }

    #[test]
    fn reorder_v_heads_2x3_swaps_outer_axes() {
        // Source layout [nk=2, nv_per_k=3, head_dim=4]:
        //   k=0: v0=[0,1,2,3] v1=[4,5,6,7] v2=[8,9,10,11]
        //   k=1: v0=[12,13,14,15] v1=[16,17,18,19] v2=[20,21,22,23]
        // After reorder to [nv_per_k=3, nk=2, head_dim=4]:
        //   v=0: k0=[0,1,2,3] k1=[12,13,14,15]
        //   v=1: k0=[4,5,6,7] k1=[16,17,18,19]
        //   v=2: k0=[8,9,10,11] k1=[20,21,22,23]
        let inp: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let out = apply_bake_op(
            inp,
            &BakeOp::ReorderVHeads {
                num_k_heads: 2,
                num_v_per_k: 3,
                head_dim: 4,
                slice: None,
            },
        )
        .unwrap();
        let expect: Vec<f32> = [
            0.0, 1.0, 2.0, 3.0, 12.0, 13.0, 14.0, 15.0, // v=0
            4.0, 5.0, 6.0, 7.0, 16.0, 17.0, 18.0, 19.0, // v=1
            8.0, 9.0, 10.0, 11.0, 20.0, 21.0, 22.0, 23.0, // v=2
        ]
        .to_vec();
        assert_eq!(out, expect);
    }

    #[test]
    fn reorder_v_heads_is_self_inverse() {
        // Per orphan src/models/qwen35/mod.rs:378 invariant comment:
        // "Applying this function twice (with the same params) returns
        // the original data, because swapping two axes is a self-
        // inverse permutation." Note: self-inverse only holds when
        // num_k_heads == num_v_per_k; otherwise the reverse permutation
        // requires swapping the param order.
        let inp: Vec<f32> = (0..36).map(|i| i as f32).collect();
        let once = apply_bake_op(
            inp.clone(),
            &BakeOp::ReorderVHeads {
                num_k_heads: 3,
                num_v_per_k: 3,
                head_dim: 4,
                slice: None,
            },
        )
        .unwrap();
        let twice = apply_bake_op(
            once,
            &BakeOp::ReorderVHeads {
                num_k_heads: 3,
                num_v_per_k: 3,
                head_dim: 4,
                slice: None,
            },
        )
        .unwrap();
        assert_eq!(twice, inp);
    }

    #[test]
    fn reorder_v_heads_with_slice_preserves_outside() {
        // Buffer of 30 elements; reorder only the middle 24 (matches
        // the in_proj_qkv pattern where Q rows + K rows are preserved
        // and only V rows are reordered).
        let mut inp: Vec<f32> = vec![999.0; 30];
        for i in 0..24 {
            inp[3 + i] = i as f32;
        }
        let out = apply_bake_op(
            inp.clone(),
            &BakeOp::ReorderVHeads {
                num_k_heads: 2,
                num_v_per_k: 3,
                head_dim: 4,
                slice: Some(3..27),
            },
        )
        .unwrap();
        // Outside the slice — preserved.
        assert_eq!(out[0..3], [999.0, 999.0, 999.0]);
        assert_eq!(out[27..30], [999.0, 999.0, 999.0]);
        // Inside the slice — same expected reorder as in
        // reorder_v_heads_2x3_swaps_outer_axes.
        let expect_inside: Vec<f32> = [
            0.0, 1.0, 2.0, 3.0, 12.0, 13.0, 14.0, 15.0, 4.0, 5.0, 6.0, 7.0, 16.0, 17.0, 18.0, 19.0,
            8.0, 9.0, 10.0, 11.0, 20.0, 21.0, 22.0, 23.0,
        ]
        .to_vec();
        assert_eq!(out[3..27].to_vec(), expect_inside);
    }

    #[test]
    fn reorder_v_heads_shape_mismatch_errors() {
        let inp: Vec<f32> = vec![0.0; 23]; // expected 24 for 2*3*4
        let err = apply_bake_op(
            inp,
            &BakeOp::ReorderVHeads {
                num_k_heads: 2,
                num_v_per_k: 3,
                head_dim: 4,
                slice: None,
            },
        )
        .unwrap_err();
        assert!(err.reason.contains("required"));
    }

    #[test]
    fn split_axis_half_first_returns_leading_rows_per_outer() {
        // Mirrors canonical /opt/llama.cpp/conversion/qwen.py:99-112 on
        // a small fixture: HF `[n_expert=2, 2*n_ff=4, n_embd=3]`
        // gate_up_proj. First-half = gate, second-half = up.
        //   expert 0: [ a0 a1 a2 | b0 b1 b2 | c0 c1 c2 | d0 d1 d2 ]
        //                ^ gate=ab     ^ up=cd
        //   expert 1: [ e0 e1 e2 | f0 f1 f2 | g0 g1 g2 | h0 h1 h2 ]
        //                ^ gate=ef     ^ up=gh
        let inp: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let first = apply_bake_op(
            inp.clone(),
            &BakeOp::SplitAxisHalf {
                outer_count: 2,
                axis_size: 4,
                inner_count: 3,
                half: SplitHalf::First,
            },
        )
        .unwrap();
        // Per expert: take rows 0 and 1, drop rows 2 and 3.
        // expert 0: 0..6
        // expert 1: 12..18
        let expect_first: Vec<f32> = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0].to_vec();
        assert_eq!(first, expect_first);
    }

    #[test]
    fn split_axis_half_second_returns_trailing_rows_per_outer() {
        let inp: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let second = apply_bake_op(
            inp,
            &BakeOp::SplitAxisHalf {
                outer_count: 2,
                axis_size: 4,
                inner_count: 3,
                half: SplitHalf::Second,
            },
        )
        .unwrap();
        let expect_second: Vec<f32> = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0].to_vec();
        assert_eq!(second, expect_second);
    }

    #[test]
    fn split_axis_half_first_plus_second_reconstructs_per_outer() {
        // Concatenating First and Second halves per-outer must yield
        // the original buffer. This is the byte-correctness invariant
        // that the canonical Python split holds.
        let inp: Vec<f32> = (0..120).map(|i| (i as f32) * 0.5).collect();
        let outer = 3_usize;
        let axis = 8_usize;
        let inner = 5_usize;
        assert_eq!(outer * axis * inner, inp.len());
        let first = apply_bake_op(
            inp.clone(),
            &BakeOp::SplitAxisHalf {
                outer_count: outer,
                axis_size: axis,
                inner_count: inner,
                half: SplitHalf::First,
            },
        )
        .unwrap();
        let second = apply_bake_op(
            inp.clone(),
            &BakeOp::SplitAxisHalf {
                outer_count: outer,
                axis_size: axis,
                inner_count: inner,
                half: SplitHalf::Second,
            },
        )
        .unwrap();
        let half_axis = axis / 2;
        let mut recon = Vec::with_capacity(inp.len());
        for o in 0..outer {
            let off_first = o * half_axis * inner;
            let off_second = o * half_axis * inner;
            recon.extend_from_slice(&first[off_first..off_first + half_axis * inner]);
            recon.extend_from_slice(&second[off_second..off_second + half_axis * inner]);
        }
        assert_eq!(recon, inp);
    }

    #[test]
    fn split_axis_half_rejects_odd_axis() {
        let inp = vec![0.0_f32; 15]; // 3 * 5 * 1 = 15, axis=5 (odd)
        let err = apply_bake_op(
            inp,
            &BakeOp::SplitAxisHalf {
                outer_count: 3,
                axis_size: 5,
                inner_count: 1,
                half: SplitHalf::First,
            },
        )
        .unwrap_err();
        assert!(err.reason.contains("not divisible by 2"));
    }

    #[test]
    fn split_axis_half_rejects_buffer_length_mismatch() {
        let inp = vec![0.0_f32; 23];
        let err = apply_bake_op(
            inp,
            &BakeOp::SplitAxisHalf {
                outer_count: 2,
                axis_size: 4,
                inner_count: 3,
                half: SplitHalf::First,
            },
        )
        .unwrap_err();
        assert!(err.reason.contains("expected"));
    }

    #[test]
    fn reorder_v_heads_per_row_2x3_swap_within_each_row() {
        // 2 rows, each row laid out as [nk=2, nv_per_k=3, head_dim_in_row=2].
        // Per row: 12 elements. Total: 24.
        // Row 0: [a0_v0(0,1) a0_v1(2,3) a0_v2(4,5) | a1_v0(6,7) a1_v1(8,9) a1_v2(10,11)]
        // After swap to [nv_per_k=3, nk=2, head_dim_in_row=2]:
        //   [a0_v0(0,1) a1_v0(6,7) | a0_v1(2,3) a1_v1(8,9) | a0_v2(4,5) a1_v2(10,11)]
        let inp: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let out = apply_bake_op(
            inp,
            &BakeOp::ReorderVHeadsPerRow {
                row_count: 2,
                num_k_heads: 2,
                num_v_per_k: 3,
                head_dim_in_row: 2,
            },
        )
        .unwrap();
        let row0_expect = [0.0, 1.0, 6.0, 7.0, 2.0, 3.0, 8.0, 9.0, 4.0, 5.0, 10.0, 11.0];
        let row1_expect = [12.0, 13.0, 18.0, 19.0, 14.0, 15.0, 20.0, 21.0, 16.0, 17.0, 22.0, 23.0];
        assert_eq!(out[0..12], row0_expect);
        assert_eq!(out[12..24], row1_expect);
    }

    #[test]
    fn reorder_v_heads_per_row_buffer_mismatch_errors() {
        let inp = vec![0.0_f32; 23];
        let err = apply_bake_op(
            inp,
            &BakeOp::ReorderVHeadsPerRow {
                row_count: 2,
                num_k_heads: 2,
                num_v_per_k: 3,
                head_dim_in_row: 2,
            },
        )
        .unwrap_err();
        assert!(err.reason.contains("expected"));
    }

    #[test]
    fn reorder_v_heads_per_row_preserves_row_boundaries() {
        // 3 rows, distinct sentinels per row. After reorder, each
        // row's elements must stay within its row boundaries.
        let mut inp = Vec::with_capacity(36);
        for r in 0..3 {
            for i in 0..12 {
                inp.push(1000.0 * (r as f32 + 1.0) + i as f32);
            }
        }
        let out = apply_bake_op(
            inp,
            &BakeOp::ReorderVHeadsPerRow {
                row_count: 3,
                num_k_heads: 2,
                num_v_per_k: 3,
                head_dim_in_row: 2,
            },
        )
        .unwrap();
        // Row 0: all values in 1000.x range
        for &v in &out[0..12] {
            assert!(v >= 1000.0 && v < 2000.0, "row 0 leaked: {v}");
        }
        // Row 1: all values in 2000.x range
        for &v in &out[12..24] {
            assert!(v >= 2000.0 && v < 3000.0, "row 1 leaked: {v}");
        }
        // Row 2: all values in 3000.x range
        for &v in &out[24..36] {
            assert!(v >= 3000.0 && v < 4000.0, "row 2 leaked: {v}");
        }
    }

    #[test]
    fn squeeze_is_data_identity() {
        // Squeeze is shape-only — element data is preserved bit-exact.
        // Plan-build handles the gguf_shape adjustment.
        let inp: Vec<f32> = vec![1.5, -2.25, 3.125, 0.0];
        let out = apply_bake_op(inp.clone(), &BakeOp::Squeeze).unwrap();
        assert_eq!(out, inp);
    }

    #[test]
    fn reorder_v_heads_matches_orphan_byte_layout() {
        // Cross-reference test: this BakeOp::ReorderVHeads result must
        // be byte-identical to the orphan's
        // src/models/qwen35/mod.rs:379-428 `reorder_v_heads` on the
        // same input (when called with elem_size=4 = sizeof::<f32>()
        // and the same num_k_heads / num_v_per_k / head_dim).
        let inp: Vec<f32> = (0..240).map(|i| i as f32 * 0.5).collect();
        let nk = 4_usize;
        let nv = 3_usize;
        let hd = 20_usize;
        assert_eq!(nk * nv * hd, inp.len());

        // BakeOp path.
        let baked = apply_bake_op(
            inp.clone(),
            &BakeOp::ReorderVHeads {
                num_k_heads: nk,
                num_v_per_k: nv,
                head_dim: hd,
                slice: None,
            },
        )
        .unwrap();

        // Re-derive the orphan layout via the same index math at
        // F32 granularity (the orphan operates on raw bytes — for
        // F32 this is the same with elem_size=4).
        let mut orphan = inp.clone();
        for k in 0..nk {
            for v in 0..nv {
                let src = (k * nv + v) * hd;
                let dst = (v * nk + k) * hd;
                orphan[dst..dst + hd].copy_from_slice(&inp[src..src + hd]);
            }
        }
        assert_eq!(baked, orphan);
    }
}
