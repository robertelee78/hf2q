//! Gemma 4 decode dense-projection route (`HF2Q_GEMMA_DECODE_MMA=1`).
//!
//! Off by default; the switch is read once per process. When on, every
//! decode-width (`m ∈ 1..=8`) GGUF dense matmul of Gemma 4 — attention
//! q/k/v/o, dense FFN gate/up/down and the lm head — whose weight the
//! column-invariant few-row MMA mat-vec admits runs
//! `mlx_native::quantized_matmul_ggml_column_invariant_mma`. That pipeline is
//! chosen from (type, N, K) alone, so a row's result is bitwise the same at
//! every width 1..=8: the solo m=1 decode, the `[N,hidden]` batched body, the
//! fused/batched lm head and the speculative verify heads/layers all produce
//! the same per-row values (batched ≡ serial, numerics class B against the
//! stock `mul_mv` routes).
//!
//! Admission ([`admits`]) depends only on (type, N, K): Q4_K, Q6_K, Q5_0 with
//! `K % 256` / `K % 64` (generic kind, 64-weight K step) and Q8_0 with
//! `K % 32`. Any other weight (F32 router, F16/BF16, affine, other block
//! types, misaligned K) keeps the existing route at every width, so a weight's
//! kernel never depends on `m`. Widths above 8 (prefill, wide tree verify)
//! keep the existing routes. The published identity is
//! [`published_kernel`] (`/hf2q/v1/runtime`
//! `active_controls.gemma_decode_dense_kernel`, ADR-063 Gemma Q1 receipts).
use std::sync::LazyLock;

use anyhow::Result;
use mlx_native::ops::quantized_matmul_ggml::{GgmlQuantizedMatmulParams, MM_ROUTING_THRESHOLD};
use mlx_native::{GgmlType, GraphSession, KernelRegistry, MlxBuffer, MlxDevice};

use crate::quantize::imatrix::ImatrixHint;
use crate::serve::forward_mlx_shared::{
    dispatch_qmatmul, dispatch_qmatmul_column_invariant_mma, MlxQWeight,
};

pub(crate) const ENV: &str = "HF2Q_GEMMA_DECODE_MMA";

static ENABLED: LazyLock<bool> =
    LazyLock::new(|| enabled_value(std::env::var(ENV).ok().as_deref()));

fn enabled_value(value: Option<&str>) -> bool {
    value == Some("1")
}

pub(crate) fn enabled() -> bool {
    *ENABLED
}

/// Resolved decode dense kernel of this process.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecodeDenseKernel {
    /// Existing `dispatch_qmatmul` routes (`mul_mv` / NR2 record).
    Stock,
    /// `quantized_matmul_ggml_column_invariant_mma` for admitted weights.
    ColumnInvariantMma,
}

impl DecodeDenseKernel {
    fn from_enabled(enabled: bool) -> Self {
        if enabled {
            Self::ColumnInvariantMma
        } else {
            Self::Stock
        }
    }

    /// Stable wire name (`/hf2q/v1/runtime` active controls, Q1 receipts).
    pub(crate) fn wire_name(self) -> &'static str {
        match self {
            Self::Stock => "Stock",
            Self::ColumnInvariantMma => "ColumnInvariantMma",
        }
    }
}

pub(crate) fn kernel() -> DecodeDenseKernel {
    DecodeDenseKernel::from_enabled(enabled())
}

/// The value the runtime-controls surface publishes.
pub(crate) fn published_kernel() -> &'static str {
    kernel().wire_name()
}

/// Whether the column-invariant MMA admits a GGUF block weight of this
/// (type, N, K). Independent of the decode width by construction.
pub(crate) fn admits(ggml_type: GgmlType, n: usize, k: usize) -> bool {
    let k_step = match ggml_type {
        GgmlType::Q4_K | GgmlType::Q6_K | GgmlType::Q5_0 => 64,
        GgmlType::Q8_0 => 32,
        _ => return false,
    };
    n > 0
        && k > 0
        && u32::try_from(n).is_ok()
        && u32::try_from(k).is_ok()
        && k % ggml_type.block_values() as usize == 0
        && k % k_step == 0
}

/// Routing decision for one decode-time dense matmul of width `m`.
fn uses_mma(enabled: bool, ggml_type: GgmlType, n: usize, k: usize, affine: bool, m: u32) -> bool {
    enabled && !affine && (1..=MM_ROUTING_THRESHOLD).contains(&m) && admits(ggml_type, n, k)
}

fn weight_uses_mma(weight: &MlxQWeight, m: u32) -> bool {
    uses_mma(
        enabled(),
        weight.info.ggml_dtype,
        weight.info.rows,
        weight.info.cols,
        weight.affine.is_some(),
        m,
    )
}

/// Gemma 4 decode dense matmul `input[m,K] · weightᵀ → output[m,N]`: the
/// column-invariant MMA when [`uses_mma`], else [`dispatch_qmatmul`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_decode_dense(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxQWeight,
    output: &MlxBuffer,
    m: u32,
    hint: ImatrixHint<'_>,
) -> Result<()> {
    if weight_uses_mma(weight, m) {
        dispatch_qmatmul_column_invariant_mma(
            session, registry, device, input, weight, output, m, hint,
        )
    } else {
        dispatch_qmatmul(session, registry, device, input, weight, output, m, hint)
    }
}

/// Encoder-based tree-verify projection (`gemma4_tree_verify_*`): the
/// column-invariant MMA when [`uses_mma`] at `m = seq_len`, else the shared
/// `apply_linear_projection_f32_qweight`. Allocates the `[seq_len, out]`
/// output like the shared helper.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_tree_verify_projection(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxQWeight,
    seq_len: u32,
    in_features: u32,
    out_features: u32,
) -> Result<MlxBuffer> {
    if !(weight.buffer.dtype() == mlx_native::DType::U8 && weight_uses_mma(weight, seq_len)) {
        return crate::inference::models::qwen35::gpu_full_attn::apply_linear_projection_f32_qweight(
            encoder,
            registry,
            device,
            input,
            weight,
            seq_len,
            in_features,
            out_features,
        );
    }
    anyhow::ensure!(
        weight.info.cols == in_features as usize && weight.info.rows == out_features as usize,
        "tree-verify projection shape [{}, {}] != weight [{}, {}]",
        out_features,
        in_features,
        weight.info.rows,
        weight.info.cols
    );
    let dst = device
        .alloc_buffer(
            seq_len as usize * out_features as usize * std::mem::size_of::<f32>(),
            mlx_native::DType::F32,
            vec![seq_len as usize, out_features as usize],
        )
        .map_err(|e| anyhow::anyhow!("alloc projection output: {e}"))?;
    let params = GgmlQuantizedMatmulParams {
        m: seq_len,
        n: out_features,
        k: in_features,
        ggml_type: weight.info.ggml_dtype,
    };
    mlx_native::quantized_matmul_ggml_column_invariant_mma(
        encoder,
        registry,
        device,
        input,
        &weight.buffer,
        &dst,
        &params,
    )
    .map_err(|e| {
        anyhow::anyhow!("quantized_matmul_ggml_column_invariant_mma (tree verify): {e}")
    })?;
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SUPPORTED: [GgmlType; 4] = [
        GgmlType::Q4_K,
        GgmlType::Q6_K,
        GgmlType::Q5_0,
        GgmlType::Q8_0,
    ];

    #[test]
    fn gemma_decode_mma_switch_is_exactly_one() {
        assert!(enabled_value(Some("1")));
        for raw in [
            None,
            Some("0"),
            Some(""),
            Some("true"),
            Some("on"),
            Some(" 1"),
        ] {
            assert!(!enabled_value(raw), "{raw:?}");
        }
        assert_eq!(DecodeDenseKernel::from_enabled(false).wire_name(), "Stock");
        assert_eq!(
            DecodeDenseKernel::from_enabled(true).wire_name(),
            "ColumnInvariantMma"
        );
    }

    #[test]
    fn gemma_decode_mma_off_keeps_every_weight_stock() {
        for ty in SUPPORTED {
            for m in 1..=8 {
                assert!(!uses_mma(false, ty, 2816, 2816, false, m));
            }
        }
    }

    #[test]
    fn gemma_decode_mma_admits_gemma_shapes_and_types() {
        // Gemma 4 26B-A4B dense/lm-head shapes (N, K).
        for (ty, n, k) in [
            (GgmlType::Q4_K, 8192, 2816),
            (GgmlType::Q4_K, 4096, 2816),
            (GgmlType::Q6_K, 262144, 2816),
            (GgmlType::Q6_K, 2112, 2816),
            (GgmlType::Q5_0, 2816, 2112),
            (GgmlType::Q8_0, 2816, 2112),
        ] {
            assert!(admits(ty, n, k), "{ty:?} {n}x{k}");
            assert!(uses_mma(true, ty, n, k, false, 1), "{ty:?} {n}x{k}");
        }
        // Q8_0 needs K % 32 only; Q5_0 needs the 64-weight generic step.
        assert!(admits(GgmlType::Q8_0, 16, 96));
        assert!(!admits(GgmlType::Q5_0, 16, 96));
        assert!(!admits(GgmlType::Q4_K, 16, 2816 + 64));
        assert!(!admits(GgmlType::Q4_K, 0, 2816));
    }

    #[test]
    fn gemma_decode_mma_rejects_unsupported_types_and_affine() {
        for ty in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::BF16,
            GgmlType::Q4_0,
            GgmlType::Q5_1,
            GgmlType::Q5_K,
        ] {
            for m in 1..=8 {
                assert!(!uses_mma(true, ty, 2816, 2816, false, m), "{ty:?} m={m}");
            }
        }
        // Affine weights carry an F32 sentinel type and are never block weights.
        assert!(!uses_mma(true, GgmlType::Q4_K, 2816, 2816, true, 1));
    }

    #[test]
    fn gemma_decode_mma_decision_is_independent_of_width() {
        for ty in [
            GgmlType::Q4_K,
            GgmlType::Q6_K,
            GgmlType::Q5_0,
            GgmlType::Q8_0,
            GgmlType::F32,
            GgmlType::F16,
        ] {
            for (n, k) in [
                (2816, 2816),
                (2816, 2112),
                (128, 2816),
                (16, 96),
                (8192, 2816 + 32),
            ] {
                for enabled in [false, true] {
                    let at_one = uses_mma(enabled, ty, n, k, false, 1);
                    for m in 2..=MM_ROUTING_THRESHOLD {
                        assert_eq!(
                            uses_mma(enabled, ty, n, k, false, m),
                            at_one,
                            "{ty:?} {n}x{k} enabled={enabled} m={m}"
                        );
                    }
                }
            }
        }
        // Prefill widths and m = 0 never take the decode kernel.
        assert!(!uses_mma(true, GgmlType::Q4_K, 2816, 2816, false, 0));
        assert!(!uses_mma(
            true,
            GgmlType::Q4_K,
            2816,
            2816,
            false,
            MM_ROUTING_THRESHOLD + 1
        ));
    }
}
