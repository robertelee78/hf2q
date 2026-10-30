//! Fail-closed route admission for Qwen joint B-row decode.
//!
//! A joint step packs one decode token from each of several independent
//! slots. Every product it issues either runs once over the packed rows or
//! runs once per row through the exact helper the serial one-token graph
//! uses. The contract depends on the route mode:
//!
//! - [`JointRouteMode::Exact`]: each row must be bitwise identical to its
//!   serial one-token forward. Packed products are admitted only for block
//!   storage whose canonical `m = 2..=8` route keeps one-row arithmetic
//!   (`Mv`, the `mvN` K-quant tiles, canonical Q5_K), and only while
//!   `HF2Q_DECODE_MV_EXT` is not `1`. Float storage switches from a matvec at
//!   `m = 1` to tiled kernels at `m > 1`, so it runs per row.
//! - [`JointRouteMode::PaddedMvExt`]: the Qwen3.8 production policy
//!   (`HF2Q_DECODE_MV_EXT=1`) routes K-quants through the non-exact
//!   `mul_mv_ext` kernel at widths 4..=8. Rows are padded to a fixed width so
//!   a row's arithmetic never depends on how many tenants share the step;
//!   the gate is decision parity (ADR-044 precedent), not bitwise equality.
//!   Only dense-FFN artifacts are admitted in this mode.
//!
//! Any other block storage type is refused, so an unproven artifact keeps the
//! serial path instead of coupling one conversation's numerics to another's.

use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::DType;

/// Widest joint step the graph accepts; it stays on the `m <= 8` matvec
/// routes. Serving caps the default lower (see the scheduler's row cap).
pub(crate) const MAX_JOINT_DECODE_ROWS: usize = 8;

/// Fixed padded width for the decision-parity mode at the default cap.
const PADDED_MV_EXT_WIDTH: usize = 4;

/// How a joint step's packed products are allowed to differ from serial.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum JointRouteMode {
    /// Bitwise equal to serial decode per row.
    Exact,
    /// Packed products use `mul_mv_ext` at a fixed padded width; gated by
    /// decision parity against independent sequential forwards.
    PaddedMvExt,
}

impl JointRouteMode {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::PaddedMvExt => "padded_mv_ext",
        }
    }
}

/// How one product of a joint step is issued.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum JointProductRoute {
    /// One dispatch over every packed row.
    Batched,
    /// One serial-identical `m = 1` dispatch per row.
    PerRow,
}

/// Process routing switches that decide the joint contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct JointRoutingEnvironment {
    pub mv_ext: bool,
}

impl JointRoutingEnvironment {
    pub(crate) fn from_process() -> Self {
        Self {
            mv_ext: std::env::var("HF2Q_DECODE_MV_EXT").as_deref() == Ok("1"),
        }
    }
}

/// Route mode for this artifact family under the process routing policy,
/// or the reason joint decode must stay serial.
pub(crate) fn select_joint_route_mode(
    environment: JointRoutingEnvironment,
    moe: bool,
) -> Result<JointRouteMode, String> {
    match (environment.mv_ext, moe) {
        (false, _) => Ok(JointRouteMode::Exact),
        (true, false) => Ok(JointRouteMode::PaddedMvExt),
        (true, true) => Err(
            "HF2Q_DECODE_MV_EXT=1 routes packed MoE rows through non-exact mul_mv_ext \
             and no decision-parity gate covers MoE artifacts"
                .into(),
        ),
    }
}

/// Reject a mode the current routing policy cannot honour.
pub(crate) fn check_mode_environment(
    mode: JointRouteMode,
    environment: JointRoutingEnvironment,
) -> Result<(), String> {
    if mode == JointRouteMode::Exact && environment.mv_ext {
        return Err(
            "exact joint decode requires HF2Q_DECODE_MV_EXT!=1 (mul_mv_ext is not bitwise)".into(),
        );
    }
    Ok(())
}

/// Packed width for `rows` real rows. Exact steps pack only real rows; the
/// decision-parity mode pads to 4 (or 8 above four rows) so a row's
/// `mul_mv_ext` width never depends on its tenants.
pub(crate) fn joint_padded_width(mode: JointRouteMode, rows: usize) -> usize {
    match mode {
        JointRouteMode::Exact => rows,
        JointRouteMode::PaddedMvExt if rows <= PADDED_MV_EXT_WIDTH => PADDED_MV_EXT_WIDTH,
        JointRouteMode::PaddedMvExt => MAX_JOINT_DECODE_ROWS,
    }
}

/// Block storage whose packed route is proven (or, for `PaddedMvExt`,
/// width-consistent) on the default matvec family.
fn packed_block_type(ggml_type: GgmlType) -> bool {
    matches!(
        ggml_type,
        GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K
    )
}

/// Route for one projection with Metal storage `dtype` and GGUF type
/// `ggml_type`, or the reason the artifact must stay serial.
pub(crate) fn projection_route(
    name: &str,
    dtype: DType,
    ggml_type: GgmlType,
) -> Result<JointProductRoute, String> {
    match dtype {
        DType::F32 | DType::F16 | DType::BF16 => Ok(JointProductRoute::PerRow),
        DType::U8 if packed_block_type(ggml_type) => Ok(JointProductRoute::Batched),
        DType::U8 => Err(format!(
            "{name} is {ggml_type:?}, which has no proven joint decode route"
        )),
        other => Err(format!("{name} has unsupported joint storage {other:?}")),
    }
}

/// Dense FFN products always run packed (the fused gate/up kernels have no
/// per-row fallback in the joint graph), so both types must be admitted.
pub(crate) fn dense_ffn_route(
    layer: usize,
    gate_up: GgmlType,
    down: GgmlType,
) -> Result<(), String> {
    for (role, ggml_type) in [("gate/up", gate_up), ("down", down)] {
        if !packed_block_type(ggml_type) {
            return Err(format!(
                "layer {layer} dense FFN {role} is {ggml_type:?}, which has no proven joint decode route"
            ));
        }
    }
    Ok(())
}

/// Process-wide switches that change the serial one-token graph's
/// submission shape. Any of them keeps decode serial. (Layer-dump
/// diagnostics are checked by the graph, which owns their predicates.)
pub(crate) fn joint_decode_env_unavailable_reason() -> Option<String> {
    let flag = |name: &str| std::env::var(name).as_deref() == Ok("1");
    if flag("HF2Q_LEGACY_PER_LAYER_CB") {
        return Some("HF2Q_LEGACY_PER_LAYER_CB=1 selects the legacy serial graph".into());
    }
    if flag("MLX_UNRETAINED_REFS") {
        return Some("MLX_UNRETAINED_REFS=1 requires the serial buffer anchors".into());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_mode_is_selected_unless_mv_ext_is_on() {
        let off = JointRoutingEnvironment { mv_ext: false };
        let on = JointRoutingEnvironment { mv_ext: true };
        assert_eq!(
            select_joint_route_mode(off, false),
            Ok(JointRouteMode::Exact)
        );
        assert_eq!(
            select_joint_route_mode(off, true),
            Ok(JointRouteMode::Exact)
        );
        assert_eq!(
            select_joint_route_mode(on, false),
            Ok(JointRouteMode::PaddedMvExt)
        );
        assert!(select_joint_route_mode(on, true).is_err());
    }

    #[test]
    fn exact_mode_refuses_mv_ext_policy() {
        let on = JointRoutingEnvironment { mv_ext: true };
        let off = JointRoutingEnvironment { mv_ext: false };
        assert!(check_mode_environment(JointRouteMode::Exact, on).is_err());
        assert!(check_mode_environment(JointRouteMode::Exact, off).is_ok());
        assert!(check_mode_environment(JointRouteMode::PaddedMvExt, on).is_ok());
    }

    #[test]
    fn padded_width_never_depends_on_tenant_count() {
        for rows in 1..=4 {
            assert_eq!(joint_padded_width(JointRouteMode::PaddedMvExt, rows), 4);
            assert_eq!(joint_padded_width(JointRouteMode::Exact, rows), rows);
        }
        for rows in 5..=MAX_JOINT_DECODE_ROWS {
            assert_eq!(joint_padded_width(JointRouteMode::PaddedMvExt, rows), 8);
        }
    }

    #[test]
    fn float_storage_runs_per_row_and_unproven_blocks_are_refused() {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            assert_eq!(
                projection_route("router", dtype, GgmlType::F32),
                Ok(JointProductRoute::PerRow)
            );
        }
        for ggml_type in [
            GgmlType::Q4_0,
            GgmlType::Q8_0,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
        ] {
            assert_eq!(
                projection_route("wq", DType::U8, ggml_type),
                Ok(JointProductRoute::Batched)
            );
        }
        for ggml_type in [
            GgmlType::Q2_K,
            GgmlType::Q3_K,
            GgmlType::Q5_0,
            GgmlType::IQ4_NL,
            GgmlType::IQ4_XS,
        ] {
            let error = projection_route("attn_qkv", DType::U8, ggml_type)
                .expect_err("unproven block storage must stay serial");
            assert!(error.contains("attn_qkv"), "{error}");
        }
    }

    #[test]
    fn dense_ffn_requires_both_roles_admitted() {
        assert!(dense_ffn_route(3, GgmlType::Q4_K, GgmlType::Q6_K).is_ok());
        let error = dense_ffn_route(3, GgmlType::IQ4_NL, GgmlType::Q6_K)
            .expect_err("unproven gate/up storage");
        assert!(
            error.contains("layer 3") && error.contains("gate/up"),
            "{error}"
        );
        assert!(dense_ffn_route(3, GgmlType::Q4_K, GgmlType::Q2_K).is_err());
    }
}
