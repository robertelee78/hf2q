//! Row-exact K-quant projection selector (`HF2Q_QWEN_ROWEXACT_PROJ=1`).
//!
//! ADR-063 Q4-QWEN36 F3. With the switch on, every Qwen3.5-family dense GGML
//! projection of `m = 1..=8` rows whose weight is in [`ROW_EXACT_PROJ_SHAPES`]
//! runs `mlx_native::ops::row_exact_matvec` instead of the stock
//! `quantized_matmul_ggml` mat-vec. Each output row of that kernel depends
//! only on the weight and its own input row, so a row is bitwise the same at
//! `m = 1` (serial greedy and sampled decode, MTP drafting) and inside a joint
//! step of 2..=8 rows.
//!
//! The selector sits in the single Qwen GGML projection boundary
//! ([`super::execution_dispatch::quantized_matmul_ggml`]), so every caller
//! (the joint graph, the serial decode builders, the `seq_len = 1` forward of
//! sampled rows, MTP, verify heads and the lm head on every path) gets the
//! same decision for the same weight at every width. Products of more than 8
//! rows (prefill chunks) keep the stock route.
//!
//! Admission depends on the weight alone, never on `m`: the `(type, N, K)`
//! triple is in the measured set, the storage is raw GGML blocks (`U8`) with
//! `K` a multiple of 256, the weight starts on the kernel's load alignment
//! (16 bytes for Q4_K, 2 for Q6_K) and holds `N` full rows. A weight that
//! fails any of these keeps the stock route at every width. Activation
//! constraints (F32, 16-byte-aligned input start) are not a fallback: a
//! violating call is refused, so a row never silently changes kernels.
//!
//! Numerics: class B against stock (different reduction order, bounded by
//! `gamma_K * sum |w x|` against an F64 reference); bitwise row-exact across
//! widths.

use std::sync::LazyLock;

use mlx_native::ops::quantized_matmul_ggml::{GgmlQuantizedMatmulParams, GgmlType};
use mlx_native::ops::row_exact_matvec::{
    dispatch_row_exact_matvec, RowExactMatvecShape, RowExactWeight, ROW_EXACT_MATVEC_MAX_ROWS,
};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice, MlxError};

/// Switch for the row-exact projection route. Default off.
pub(crate) const QWEN_ROWEXACT_PROJ_ENV: &str = "HF2Q_QWEN_ROWEXACT_PROJ";

/// `(type, N, K)` of the Qwen3.6-35B-A3B projections routed to the row-exact
/// kernel: the shapes where it measured at or below the stock mat-vec over
/// widths 1..4 (quiet window, 2026-11-07). Q6_K 2048x512 and Q6_K 512x2048
/// measured slower and keep the stock kernel.
pub(crate) const ROW_EXACT_PROJ_SHAPES: [(GgmlType, u32, u32); 7] = [
    // DeltaNet attn_qkv, full-attention fused Q + gate.
    (GgmlType::Q4_K, 8192, 2048),
    // DeltaNet attn_gate (z), split Q / gate.
    (GgmlType::Q4_K, 4096, 2048),
    // DeltaNet ssm_out, full-attention attn_output.
    (GgmlType::Q4_K, 2048, 4096),
    // DeltaNet attn_qkv.
    (GgmlType::Q6_K, 8192, 2048),
    // attn_k / attn_v, native shared-expert gate and up.
    (GgmlType::Q4_K, 512, 2048),
    // Native shared-expert down.
    (GgmlType::Q4_K, 2048, 512),
    // lm head.
    (GgmlType::Q6_K, 248320, 2048),
];

/// Parse [`QWEN_ROWEXACT_PROJ_ENV`]; unset or blank means off.
pub(crate) fn parse_qwen_rowexact_proj(raw: Option<&str>) -> Result<bool, String> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(false);
    };
    match raw.to_ascii_lowercase().as_str() {
        "1" | "on" | "true" => Ok(true),
        "0" | "off" | "false" => Ok(false),
        _ => Err(format!(
            "{QWEN_ROWEXACT_PROJ_ENV} must be 1/on/true or 0/off/false, got {raw:?}"
        )),
    }
}

static REQUESTED: LazyLock<bool> = LazyLock::new(|| {
    parse_qwen_rowexact_proj(std::env::var(QWEN_ROWEXACT_PROJ_ENV).ok().as_deref()).unwrap_or_else(
        |error| {
            tracing::warn!(
                error,
                fallback = false,
                "invalid Qwen row-exact projection switch"
            );
            false
        },
    )
});

#[cfg(test)]
thread_local! {
    static TEST_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// [`QWEN_ROWEXACT_PROJ_ENV`] read once per process (at the first Qwen GGML
/// projection or the worker-start log). Invalid values warn and keep the
/// default (off).
pub(crate) fn qwen_rowexact_proj_requested() -> bool {
    #[cfg(test)]
    if let Some(enabled) = TEST_OVERRIDE.with(std::cell::Cell::get) {
        return enabled;
    }
    *REQUESTED
}

/// Run `operation` with the switch forced to `enabled` on this thread.
#[cfg(test)]
pub(crate) fn with_rowexact_proj_for_test<T>(enabled: bool, operation: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_OVERRIDE.with(|cell| cell.set(self.0));
        }
    }
    let _restore = Restore(TEST_OVERRIDE.with(|cell| cell.replace(Some(enabled))));
    operation()
}

/// Resolved route over a set of loaded projection weights: how many of the
/// weights whose `(type, N, K)` is in [`ROW_EXACT_PROJ_SHAPES`] (`eligible`)
/// the row-exact kernel admits under the switch (`admitted`). Weights of
/// other shapes always run stock and are not counted.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProjRouteCount {
    pub(crate) admitted: usize,
    pub(crate) eligible: usize,
}

impl ProjRouteCount {
    /// Count one loaded projection weight read by a decode dispatch as
    /// `[n, k]` of `ggml_type` (the Metal buffer the kernel reads).
    pub(crate) fn add(
        &mut self,
        enabled: bool,
        ggml_type: GgmlType,
        n: u32,
        k: u32,
        weight: &MlxBuffer,
    ) {
        if ROW_EXACT_PROJ_SHAPES.contains(&(ggml_type, n, k)) {
            self.eligible += 1;
            self.admitted +=
                usize::from(row_exact_weight(enabled, ggml_type, n, k, weight).is_some());
        }
    }

    /// Sum of two counts.
    pub(crate) fn merged(self, other: Self) -> Self {
        Self {
            admitted: self.admitted + other.admitted,
            eligible: self.eligible + other.eligible,
        }
    }

    /// `stock` when nothing is admitted (switch off, or no eligible weight
    /// passes admission), `row_exact` when every eligible weight is admitted,
    /// `mixed` otherwise.
    pub(crate) fn label(self) -> &'static str {
        if self.admitted == 0 {
            "stock"
        } else if self.admitted == self.eligible {
            "row_exact"
        } else {
            "mixed"
        }
    }
}

impl std::fmt::Display for ProjRouteCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.admitted, self.eligible)
    }
}

/// Resolved projection route of a loaded model: the main decode graph (every
/// layer plus the lm head) and the MTP block, when one is loaded.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProjRoute {
    pub(crate) main: ProjRouteCount,
    pub(crate) mtp: Option<ProjRouteCount>,
}

impl ProjRoute {
    /// Main graph and MTP block together.
    pub(crate) fn total(self) -> ProjRouteCount {
        self.mtp.map_or(self.main, |mtp| self.main.merged(mtp))
    }
}

/// Row-exact kernel of a weight, or `None` for the stock route. Depends on
/// the weight alone (`ggml_type`, `n`, `k`, storage, start alignment, size).
pub(crate) fn row_exact_weight(
    enabled: bool,
    ggml_type: GgmlType,
    n: u32,
    k: u32,
    weight: &MlxBuffer,
) -> Option<RowExactWeight> {
    if !enabled || !ROW_EXACT_PROJ_SHAPES.contains(&(ggml_type, n, k)) {
        return None;
    }
    let (kernel, align) = match ggml_type {
        GgmlType::Q4_K => (RowExactWeight::Q4_K, 16),
        GgmlType::Q6_K => (RowExactWeight::Q6_K, 2),
        _ => return None,
    };
    let rows_fit = kernel
        .row_bytes(k)
        .checked_mul(u64::from(n))
        .is_some_and(|bytes| bytes <= weight.data_byte_len() as u64);
    (weight.dtype() == DType::U8 && k % 256 == 0 && weight.byte_offset() % align == 0 && rows_fit)
        .then_some(kernel)
}

/// Kernel for one projection call: the weight's row-exact kernel when
/// `params.m` is a decode width (1..=8), else `None` (stock).
pub(crate) fn select(
    enabled: bool,
    weight: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> Option<RowExactWeight> {
    if params.m == 0 || params.m > ROW_EXACT_MATVEC_MAX_ROWS {
        return None;
    }
    row_exact_weight(enabled, params.ggml_type, params.n, params.k, weight)
}

/// Encode `output[m, n] = input[m, k] @ weight^T` on the row-exact kernel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    kernel: RowExactWeight,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> mlx_native::Result<()> {
    dispatch_row_exact_matvec(
        encoder,
        registry,
        device.metal_device(),
        kernel,
        weight,
        input,
        output,
        &RowExactMatvecShape::contiguous(params.m, params.n, params.k),
    )
    .map_err(|error| {
        MlxError::InvalidArgument(format!(
            "{QWEN_ROWEXACT_PROJ_ENV}: {:?} [{}x{}] m={}: {error}",
            params.ggml_type, params.n, params.k, params.m
        ))
    })
}
