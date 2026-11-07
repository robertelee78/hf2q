//! Gemma 4 decode dense-projection selector (`HF2Q_GEMMA_DECODE_DENSE`).
//!
//! `stock` (default), `row_exact` or `mma`; any other value is refused when
//! the switch is first read (model load). The value is read once per process.
//!
//! The selector covers every decode-width GGUF dense matmul of Gemma 4 —
//! attention q/k/v/o, dense FFN gate/up/down and the lm head — in the solo
//! decode, the `[N,hidden]` batched body, the batched/fused lm head, the
//! kernel-profile decode, and the speculative verify heads and tree-verify
//! layers:
//! - `stock`: the existing `dispatch_qmatmul` / encoder routes.
//! - `row_exact`: `mlx_native::ops::row_exact_matvec` (row-exact across
//!   m = 1..8).
//! - `mma`: `mlx_native::quantized_matmul_ggml_column_invariant_mma`
//!   (column-invariant across m = 1..8, numerics class B).
//!
//! Both non-stock kernels give each row a result that is bitwise independent
//! of how many rows share the dispatch, so batched ≡ serial holds. A product
//! of more than 8 rows (wide verify heads, long trees) runs as consecutive
//! chunks of at most 8 rows, which keeps every row's bits unchanged.
//!
//! Admission ([`route`]) depends on the weight alone — (type, N, K), its
//! storage and its start alignment — never on `m`; a weight the selected
//! kernel does not admit keeps the stock route at every width. Router
//! projections (F32), F16/BF16, affine weights, expert matmuls and prefill are
//! not routed here. The resolved kernel is published as
//! `active_controls.gemma_decode_dense_kernel` in `/hf2q/v1/runtime` and
//! declared as `decode_dense_kernel` in ADR-063 Gemma Q1 packets.
use std::sync::LazyLock;

use anyhow::Result;
use mlx_native::ops::quantized_matmul_ggml::{GgmlQuantizedMatmulParams, MM_ROUTING_THRESHOLD};
use mlx_native::ops::row_exact_matvec::{
    dispatch_row_exact_matvec, RowExactMatvecShape, RowExactWeight,
};
use mlx_native::{
    CommandEncoder, DType, GgmlType, GraphSession, KernelRegistry, MlxBuffer, MlxDevice,
};

use crate::quantize::imatrix::ImatrixHint;
use crate::serve::forward_mlx_shared::{dispatch_qmatmul, qmatmul_preamble, MlxQWeight};

pub(crate) const ENV: &str = "HF2Q_GEMMA_DECODE_DENSE";

/// Decode dense kernel of this process.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecodeDenseKernel {
    /// Existing `dispatch_qmatmul` / encoder routes.
    Stock,
    /// `row_exact_matvec` for admitted weights.
    RowExact,
    /// `quantized_matmul_ggml_column_invariant_mma` for admitted weights.
    ColumnInvariantMma,
}

impl DecodeDenseKernel {
    fn parse(raw: Option<&str>) -> Result<Self, String> {
        match raw {
            None | Some("stock") => Ok(Self::Stock),
            Some("row_exact") => Ok(Self::RowExact),
            Some("mma") => Ok(Self::ColumnInvariantMma),
            Some(other) => Err(format!(
                "{ENV}={other:?} is not one of `stock`, `row_exact`, `mma`"
            )),
        }
    }

    /// Stable wire name (`/hf2q/v1/runtime` active controls, Q1 receipts).
    pub(crate) fn wire_name(self) -> &'static str {
        match self {
            Self::Stock => "Stock",
            Self::RowExact => "RowExact",
            Self::ColumnInvariantMma => "ColumnInvariantMma",
        }
    }
}

static RESOLVED: LazyLock<Result<DecodeDenseKernel, String>> =
    LazyLock::new(|| match std::env::var_os(ENV) {
        None => DecodeDenseKernel::parse(None),
        Some(raw) => match raw.to_str() {
            Some(value) => DecodeDenseKernel::parse(Some(value)),
            None => Err(format!("{ENV} is not valid UTF-8")),
        },
    });

/// The process's decode dense kernel; an invalid `HF2Q_GEMMA_DECODE_DENSE`
/// is an error. Gemma model load calls this before any allocation.
pub(crate) fn resolve() -> Result<DecodeDenseKernel> {
    RESOLVED.clone().map_err(anyhow::Error::msg)
}

/// The resolved kernel. Panics on an invalid switch, which model load has
/// already refused through [`resolve`].
pub(crate) fn kernel() -> DecodeDenseKernel {
    match &*RESOLVED {
        Ok(kernel) => *kernel,
        Err(error) => panic!("{error}"),
    }
}

/// The value the runtime-controls surface publishes.
pub(crate) fn published_kernel() -> &'static str {
    kernel().wire_name()
}

/// Row-invariant kernel of one admitted weight.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InvariantKernel {
    RowExact(RowExactWeight),
    Mma,
}

/// Facts of a weight that decide its route; no decode width.
#[derive(Clone, Copy, Debug)]
struct WeightFacts {
    ggml_type: GgmlType,
    n: usize,
    k: usize,
    affine: bool,
    u8_storage: bool,
    byte_offset: u64,
}

impl WeightFacts {
    fn of(weight: &MlxQWeight) -> Self {
        Self {
            ggml_type: weight.info.ggml_dtype,
            n: weight.info.rows,
            k: weight.info.cols,
            affine: weight.affine.is_some(),
            u8_storage: weight.buffer.dtype() == DType::U8,
            byte_offset: weight.buffer.byte_offset(),
        }
    }

    fn block_weight(self) -> bool {
        !self.affine
            && self.u8_storage
            && self.n > 0
            && self.k > 0
            && u32::try_from(self.n).is_ok()
            && u32::try_from(self.k).is_ok()
    }
}

/// `row_exact_matvec` admission: K % 256 (Q4_K, Q6_K) or K % 32 (Q5_0,
/// Q8_0) and the weight start aligned for the kernel's loads (16 bytes for
/// Q4_K, 2 bytes otherwise).
fn row_exact_weight(facts: WeightFacts) -> Option<RowExactWeight> {
    let (weight, k_multiple, align) = match facts.ggml_type {
        GgmlType::Q4_K => (RowExactWeight::Q4_K, 256, 16),
        GgmlType::Q6_K => (RowExactWeight::Q6_K, 256, 2),
        GgmlType::Q5_0 => (RowExactWeight::Q5_0, 32, 2),
        GgmlType::Q8_0 => (RowExactWeight::Q8_0, 32, 2),
        _ => return None,
    };
    (facts.block_weight() && facts.k % k_multiple == 0 && facts.byte_offset % align == 0)
        .then_some(weight)
}

/// Column-invariant MMA admission: Q4_K, Q6_K, Q5_0 with K % 64 (and the
/// block size), Q8_0 with K % 32.
fn mma_admits(facts: WeightFacts) -> bool {
    let k_step = match facts.ggml_type {
        GgmlType::Q4_K | GgmlType::Q6_K | GgmlType::Q5_0 => 64,
        GgmlType::Q8_0 => 32,
        _ => return false,
    };
    facts.block_weight()
        && facts.k % facts.ggml_type.block_values() as usize == 0
        && facts.k % k_step == 0
}

/// Route of one weight under `kernel`: `None` = stock. Independent of the
/// decode width by construction.
fn route(kernel: DecodeDenseKernel, facts: WeightFacts) -> Option<InvariantKernel> {
    match kernel {
        DecodeDenseKernel::Stock => None,
        DecodeDenseKernel::RowExact => row_exact_weight(facts).map(InvariantKernel::RowExact),
        DecodeDenseKernel::ColumnInvariantMma => mma_admits(facts).then_some(InvariantKernel::Mma),
    }
}

/// `(first_row, rows)` chunks of at most 8 rows covering `0..m`.
fn row_chunks(m: u32) -> impl Iterator<Item = (u32, u32)> {
    (0..m)
        .step_by(MM_ROUTING_THRESHOLD as usize)
        .map(move |first| (first, (m - first).min(MM_ROUTING_THRESHOLD)))
}

/// Rows `first..first + rows` of a contiguous F32 `[*, cols]` buffer.
fn rows_view(buffer: &MlxBuffer, first: u32, rows: u32, cols: usize) -> MlxBuffer {
    buffer.slice_view(
        u64::from(first) * cols as u64 * std::mem::size_of::<f32>() as u64,
        rows as usize * cols,
    )
}

/// Encode `y[rows, n] = x[rows, k] · Wᵀ` for `rows ∈ 1..=8`.
#[allow(clippy::too_many_arguments)]
fn encode_invariant(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    kernel: InvariantKernel,
    weight: &MlxQWeight,
    x: &MlxBuffer,
    y: &MlxBuffer,
    rows: u32,
) -> Result<()> {
    let n = weight.info.rows as u32;
    let k = weight.info.cols as u32;
    match kernel {
        InvariantKernel::RowExact(row_exact) => dispatch_row_exact_matvec(
            encoder,
            registry,
            device.metal_device(),
            row_exact,
            &weight.buffer,
            x,
            y,
            &RowExactMatvecShape::contiguous(rows, n, k),
        )
        .map_err(|e| anyhow::anyhow!("dispatch_row_exact_matvec failed: {e}")),
        InvariantKernel::Mma => mlx_native::quantized_matmul_ggml_column_invariant_mma(
            encoder,
            registry,
            device,
            x,
            &weight.buffer,
            y,
            &GgmlQuantizedMatmulParams {
                m: rows,
                n,
                k,
                ggml_type: weight.info.ggml_dtype,
            },
        )
        .map_err(|e| anyhow::anyhow!("quantized_matmul_ggml_column_invariant_mma failed: {e}")),
    }
}

/// Gemma 4 decode dense matmul `input[m,K] · weightᵀ → output[m,N]`
/// (contiguous rows): the selected row-invariant kernel for an admitted
/// weight, in chunks of at most 8 rows, else [`dispatch_qmatmul`].
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
    let Some(kernel) = route(kernel(), WeightFacts::of(weight)) else {
        return dispatch_qmatmul(session, registry, device, input, weight, output, m, hint);
    };
    anyhow::ensure!(m > 0, "decode dense matmul with zero rows");
    if m <= MM_ROUTING_THRESHOLD {
        qmatmul_preamble(session, input, weight, output, m, hint)?;
        return encode_invariant(
            session.encoder_mut(),
            registry,
            device,
            kernel,
            weight,
            input,
            output,
            m,
        );
    }
    for (first, rows) in row_chunks(m) {
        let x = rows_view(input, first, rows, weight.info.cols);
        let y = rows_view(output, first, rows, weight.info.rows);
        qmatmul_preamble(session, &x, weight, &y, rows, hint)?;
        encode_invariant(
            session.encoder_mut(),
            registry,
            device,
            kernel,
            weight,
            &x,
            &y,
            rows,
        )?;
    }
    Ok(())
}

/// Encoder-based tree-verify projection (`gemma4_tree_verify_*`): the
/// selected row-invariant kernel for an admitted weight, in chunks of at most
/// 8 rows, else the shared `apply_linear_projection_f32_qweight`. Allocates
/// the `[seq_len, out]` output like the shared helper.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_tree_verify_projection(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxQWeight,
    seq_len: u32,
    in_features: u32,
    out_features: u32,
) -> Result<MlxBuffer> {
    let Some(kernel) = route(kernel(), WeightFacts::of(weight)) else {
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
    };
    anyhow::ensure!(
        seq_len > 0
            && weight.info.cols == in_features as usize
            && weight.info.rows == out_features as usize,
        "tree-verify projection [{seq_len}, {in_features}] -> {out_features} != weight [{}, {}]",
        weight.info.rows,
        weight.info.cols
    );
    let dst = device
        .alloc_buffer(
            seq_len as usize * out_features as usize * std::mem::size_of::<f32>(),
            DType::F32,
            vec![seq_len as usize, out_features as usize],
        )
        .map_err(|e| anyhow::anyhow!("alloc projection output: {e}"))?;
    if seq_len <= MM_ROUTING_THRESHOLD {
        encode_invariant(
            encoder, registry, device, kernel, weight, input, &dst, seq_len,
        )?;
    } else {
        for (first, rows) in row_chunks(seq_len) {
            let x = rows_view(input, first, rows, weight.info.cols);
            let y = rows_view(&dst, first, rows, weight.info.rows);
            encode_invariant(encoder, registry, device, kernel, weight, &x, &y, rows)?;
        }
    }
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(ggml_type: GgmlType, n: usize, k: usize) -> WeightFacts {
        WeightFacts {
            ggml_type,
            n,
            k,
            affine: false,
            u8_storage: true,
            byte_offset: 0,
        }
    }

    const KERNELS: [DecodeDenseKernel; 3] = [
        DecodeDenseKernel::Stock,
        DecodeDenseKernel::RowExact,
        DecodeDenseKernel::ColumnInvariantMma,
    ];

    #[test]
    fn gemma_decode_dense_switch_parses_three_values_and_refuses_the_rest() {
        assert_eq!(DecodeDenseKernel::parse(None), Ok(DecodeDenseKernel::Stock));
        assert_eq!(
            DecodeDenseKernel::parse(Some("stock")),
            Ok(DecodeDenseKernel::Stock)
        );
        assert_eq!(
            DecodeDenseKernel::parse(Some("row_exact")),
            Ok(DecodeDenseKernel::RowExact)
        );
        assert_eq!(
            DecodeDenseKernel::parse(Some("mma")),
            Ok(DecodeDenseKernel::ColumnInvariantMma)
        );
        for raw in [
            "",
            "1",
            "0",
            "MMA",
            "Stock",
            "row-exact",
            "rowexact",
            " mma",
            "mma ",
        ] {
            let error = DecodeDenseKernel::parse(Some(raw)).unwrap_err();
            assert!(error.contains(ENV), "{raw:?}: {error}");
        }
        assert_eq!(
            KERNELS.map(DecodeDenseKernel::wire_name),
            ["Stock", "RowExact", "ColumnInvariantMma"]
        );
    }

    #[test]
    fn gemma_decode_dense_stock_routes_nothing() {
        for ty in [
            GgmlType::Q4_K,
            GgmlType::Q6_K,
            GgmlType::Q5_0,
            GgmlType::Q8_0,
        ] {
            assert_eq!(route(DecodeDenseKernel::Stock, facts(ty, 2816, 2816)), None);
        }
    }

    #[test]
    fn gemma_decode_dense_row_exact_admission_per_type() {
        let row_exact = |f| route(DecodeDenseKernel::RowExact, f);
        // Gemma 4 26B-A4B dense and lm-head shapes (N, K).
        for (ty, want, n, k) in [
            (GgmlType::Q4_K, RowExactWeight::Q4_K, 8192, 2816),
            (GgmlType::Q4_K, RowExactWeight::Q4_K, 4096, 2816),
            (GgmlType::Q6_K, RowExactWeight::Q6_K, 262144, 2816),
            (GgmlType::Q6_K, RowExactWeight::Q6_K, 2112, 2816),
            (GgmlType::Q5_0, RowExactWeight::Q5_0, 2816, 2112),
            (GgmlType::Q8_0, RowExactWeight::Q8_0, 2816, 2112),
        ] {
            assert_eq!(
                row_exact(facts(ty, n, k)),
                Some(InvariantKernel::RowExact(want)),
                "{ty:?} {n}x{k}"
            );
        }
        // K multiples: 256 for the K-quants, 32 for Q5_0 / Q8_0.
        assert_eq!(row_exact(facts(GgmlType::Q4_K, 16, 2816 + 64)), None);
        assert_eq!(row_exact(facts(GgmlType::Q6_K, 16, 2816 + 128)), None);
        assert!(row_exact(facts(GgmlType::Q5_0, 16, 96)).is_some());
        assert!(row_exact(facts(GgmlType::Q8_0, 16, 96)).is_some());
        assert_eq!(row_exact(facts(GgmlType::Q8_0, 16, 80)), None);
        // Q4_K needs a 16-byte weight start; the others 2 bytes.
        for (offset, admitted) in [(0, true), (16, true), (8, false), (2, false)] {
            let f = WeightFacts {
                byte_offset: offset,
                ..facts(GgmlType::Q4_K, 16, 256)
            };
            assert_eq!(row_exact(f).is_some(), admitted, "Q4_K offset {offset}");
        }
        for ty in [GgmlType::Q6_K, GgmlType::Q5_0, GgmlType::Q8_0] {
            let f = WeightFacts {
                byte_offset: 2,
                ..facts(ty, 16, 256)
            };
            assert!(row_exact(f).is_some(), "{ty:?}");
            let f = WeightFacts {
                byte_offset: 1,
                ..facts(ty, 16, 256)
            };
            assert_eq!(row_exact(f), None, "{ty:?}");
        }
    }

    #[test]
    fn gemma_decode_dense_mma_admission_per_type() {
        let mma = |f| route(DecodeDenseKernel::ColumnInvariantMma, f);
        for (ty, n, k) in [
            (GgmlType::Q4_K, 8192, 2816),
            (GgmlType::Q6_K, 262144, 2816),
            (GgmlType::Q5_0, 2816, 2112),
            (GgmlType::Q8_0, 2816, 2112),
        ] {
            assert_eq!(mma(facts(ty, n, k)), Some(InvariantKernel::Mma), "{ty:?}");
        }
        assert_eq!(
            mma(facts(GgmlType::Q8_0, 16, 96)),
            Some(InvariantKernel::Mma)
        );
        assert_eq!(mma(facts(GgmlType::Q5_0, 16, 96)), None);
        assert_eq!(mma(facts(GgmlType::Q4_K, 16, 2816 + 64)), None);
    }

    #[test]
    fn gemma_decode_dense_refuses_unsupported_weights_under_every_kernel() {
        for kernel in KERNELS {
            for ty in [
                GgmlType::F32,
                GgmlType::F16,
                GgmlType::BF16,
                GgmlType::Q4_0,
                GgmlType::Q5_1,
                GgmlType::Q5_K,
            ] {
                assert_eq!(
                    route(kernel, facts(ty, 2816, 2816)),
                    None,
                    "{kernel:?} {ty:?}"
                );
            }
            let base = facts(GgmlType::Q4_K, 2816, 2816);
            for f in [
                WeightFacts {
                    affine: true,
                    ..base
                },
                WeightFacts {
                    u8_storage: false,
                    ..base
                },
                WeightFacts { n: 0, ..base },
                WeightFacts { k: 0, ..base },
            ] {
                assert_eq!(route(kernel, f), None, "{kernel:?} {f:?}");
            }
        }
    }

    #[test]
    fn gemma_decode_dense_rows_split_into_width_invariant_chunks() {
        // The route takes no width; wider products only split into chunks of
        // at most 8 rows that cover every row once, in order.
        assert_eq!(row_chunks(0).count(), 0);
        for m in 1..=40u32 {
            let chunks: Vec<_> = row_chunks(m).collect();
            if m <= MM_ROUTING_THRESHOLD {
                assert_eq!(chunks, [(0, m)]);
            }
            let mut next = 0;
            for (first, rows) in chunks {
                assert_eq!(first, next);
                assert!((1..=MM_ROUTING_THRESHOLD).contains(&rows));
                next += rows;
            }
            assert_eq!(next, m);
        }
    }
}
