//! Projection kernel for the packed products of a padded Qwen joint step.
//!
//! [`JointRouteMode::PaddedMvExt`] runs every packed Q4_K / Q6_K product at a
//! fixed width of [`PADDED_MV_EXT_WIDTH`] rows, so a step always pays the
//! width-4 cost, including 1-lane steps. Under
//! [`JointProjectionKernel::ColumnInvariantMma`] (switch
//! [`QWEN_JOINT_MMA_ENV`]) those products use mlx-native's few-row MMA
//! kernel ([`quantized_matmul_ggml_column_invariant_mma`]): one pipeline for
//! every width, and each row's result is independent of the width, of its
//! packed position and of the other rows. Against the `mul_mv_ext` route this
//! is a different accumulation order (Q1 Class B); cohort invariance holds by
//! construction.
//!
//! The kernel applies inside [`JointProjectionScope`] only: the joint layer
//! chains open it, the output head and every serial graph run outside it.
//!
//! [`JointRouteMode::PaddedMvExt`]: super::joint_decode_routes::JointRouteMode::PaddedMvExt
//! [`PADDED_MV_EXT_WIDTH`]: super::joint_decode_routes::PADDED_MV_EXT_WIDTH

use std::cell::Cell;

use mlx_native::ops::quantized_matmul_ggml::{GgmlQuantizedMatmulParams, GgmlType};
use mlx_native::{
    quantized_matmul_ggml_column_invariant_mma, CommandEncoder, KernelRegistry, MlxBuffer,
    MlxDevice,
};

use super::joint_decode_routes::JointRouteMode;

/// Switch for [`JointProjectionKernel::ColumnInvariantMma`]: `1`/`on`/`true`
/// selects it in the padded mode; unset, blank or `0`/`off`/`false` keeps
/// `mul_mv_ext`.
pub(crate) const QWEN_JOINT_MMA_ENV: &str = "HF2Q_QWEN_JOINT_MMA";

/// Switch for [`JointStepOptions::pooled_outputs`]: `1`/`on`/`true` takes the
/// joint layer chain's multi-row outputs from the decode pool
/// ([`super::decode_pool::StepPooledOutputs`]); unset, blank or
/// `0`/`off`/`false` keeps fresh device allocations. Either mode.
pub(crate) const QWEN_JOINT_POOLED_OUTPUTS_ENV: &str = "HF2Q_QWEN_JOINT_POOLED_OUTPUTS";

/// K step of the MMA kernel's generic (Q4_K / Q6_K) kind.
const MMA_K_STEP: u32 = 64;

/// Kernel for the packed K-quant products of a joint step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum JointProjectionKernel {
    /// The canonical dispatcher (`mul_mv_ext` at the padded width).
    Canonical,
    /// [`quantized_matmul_ggml_column_invariant_mma`].
    ColumnInvariantMma,
}

/// Per-step execution choices of a joint step that its route mode does not
/// fix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct JointStepOptions {
    pub projection: JointProjectionKernel,
    /// Multi-row layer outputs from the decode pool (Class A).
    pub pooled_outputs: bool,
}

impl JointStepOptions {
    /// The switches over the process environment for `mode`.
    pub(crate) fn from_process(mode: JointRouteMode) -> Result<Self, String> {
        let read = |name: &str| parse_switch(name, std::env::var(name).ok().as_deref());
        Ok(Self {
            projection: select_joint_projection_kernel(mode, read(QWEN_JOINT_MMA_ENV)?),
            pooled_outputs: read(QWEN_JOINT_POOLED_OUTPUTS_ENV)?,
        })
    }
}

/// Parse an on/off switch named `name`; unset or blank means off.
pub(crate) fn parse_switch(name: &str, raw: Option<&str>) -> Result<bool, String> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(false);
    };
    match raw.to_ascii_lowercase().as_str() {
        "1" | "on" | "true" => Ok(true),
        "0" | "off" | "false" => Ok(false),
        _ => Err(format!(
            "{name} must be 1/on/true or 0/off/false, got {raw:?}"
        )),
    }
}

/// Kernel for `mode` under the switch. Exact mode keeps the canonical
/// dispatcher: its rows must stay bitwise equal to serial decode.
pub(crate) fn select_joint_projection_kernel(
    mode: JointRouteMode,
    mma: bool,
) -> JointProjectionKernel {
    match mode {
        JointRouteMode::PaddedMvExt if mma => JointProjectionKernel::ColumnInvariantMma,
        JointRouteMode::Exact | JointRouteMode::PaddedMvExt => JointProjectionKernel::Canonical,
    }
}

thread_local! {
    /// Packed width of the open MMA scope, if any.
    static MMA_SCOPE_WIDTH: Cell<Option<u32>> = const { Cell::new(None) };
}

/// Routes this thread's packed products to the MMA kernel until dropped.
pub(crate) struct JointProjectionScope {
    previous: Option<u32>,
}

impl JointProjectionScope {
    /// Open a scope for products of exactly `width` rows. A
    /// [`JointProjectionKernel::Canonical`] scope changes nothing.
    pub(crate) fn enter(kernel: JointProjectionKernel, width: u32) -> Self {
        let scoped = (kernel == JointProjectionKernel::ColumnInvariantMma).then_some(width);
        Self {
            previous: MMA_SCOPE_WIDTH.with(|cell| cell.replace(scoped)),
        }
    }
}

impl Drop for JointProjectionScope {
    fn drop(&mut self) {
        MMA_SCOPE_WIDTH.with(|cell| cell.set(self.previous));
    }
}

/// Whether the open scope takes this product: the packed width, a Q4_K /
/// Q6_K weight, `K` a multiple of the kernel's K step and a 16-byte aligned
/// input.
fn admits(input: &MlxBuffer, params: &GgmlQuantizedMatmulParams) -> bool {
    MMA_SCOPE_WIDTH.with(Cell::get) == Some(params.m)
        && matches!(params.ggml_type, GgmlType::Q4_K | GgmlType::Q6_K)
        && params.k % MMA_K_STEP == 0
        && input.byte_offset() % 16 == 0
}

/// Dispatch an admitted product on the MMA kernel; `None` when the open
/// scope does not take it (the caller runs the canonical dispatcher).
pub(crate) fn dispatch_in_scope(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> Option<mlx_native::Result<()>> {
    admits(input, params).then(|| {
        quantized_matmul_ggml_column_invariant_mma(
            encoder, registry, device, input, weight, output, params,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switch_parses_on_off_and_rejects_other_values() {
        let parse = |raw| parse_switch(QWEN_JOINT_MMA_ENV, raw);
        assert_eq!(parse(None), Ok(false));
        assert_eq!(parse(Some("  ")), Ok(false));
        for on in ["1", "on", "TRUE"] {
            assert_eq!(parse(Some(on)), Ok(true));
        }
        for off in ["0", "off", "False"] {
            assert_eq!(parse(Some(off)), Ok(false));
        }
        assert!(parse(Some("mma")).is_err());
    }

    #[test]
    fn exact_mode_never_selects_mma() {
        assert_eq!(
            select_joint_projection_kernel(JointRouteMode::Exact, true),
            JointProjectionKernel::Canonical
        );
        assert_eq!(
            select_joint_projection_kernel(JointRouteMode::PaddedMvExt, true),
            JointProjectionKernel::ColumnInvariantMma
        );
        assert_eq!(
            select_joint_projection_kernel(JointRouteMode::PaddedMvExt, false),
            JointProjectionKernel::Canonical
        );
    }

    #[test]
    fn scope_admits_only_its_width_and_k_quants_and_restores_on_drop() {
        let params = |m, ggml_type, k| GgmlQuantizedMatmulParams {
            m,
            n: 64,
            k,
            ggml_type,
        };
        let device = MlxDevice::new().expect("device");
        let input = device
            .alloc_buffer(4 * 5120 * 4, mlx_native::DType::F32, vec![4, 5120])
            .expect("input");
        assert!(!admits(&input, &params(4, GgmlType::Q4_K, 5120)));
        {
            let _scope = JointProjectionScope::enter(JointProjectionKernel::ColumnInvariantMma, 4);
            assert!(admits(&input, &params(4, GgmlType::Q4_K, 5120)));
            assert!(admits(&input, &params(4, GgmlType::Q6_K, 17408)));
            assert!(!admits(&input, &params(1, GgmlType::Q4_K, 5120)));
            assert!(!admits(&input, &params(4, GgmlType::Q8_0, 5120)));
            assert!(!admits(&input, &params(4, GgmlType::Q4_K, 5120 + 32)));
            {
                let _inner = JointProjectionScope::enter(JointProjectionKernel::Canonical, 4);
                assert!(!admits(&input, &params(4, GgmlType::Q4_K, 5120)));
            }
            assert!(admits(&input, &params(4, GgmlType::Q4_K, 5120)));
        }
        assert!(!admits(&input, &params(4, GgmlType::Q4_K, 5120)));
    }
}
