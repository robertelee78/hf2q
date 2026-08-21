//! Qwen3.5 production dispatch boundary.
//!
//! The default binary keeps only the ordinary, environment-backed MLX
//! dispatch path. ADR-046 evidence configuration, trace capture, and the
//! source-teacher graph scope are validation substrate and compile with the
//! tests that exercise them.

#[cfg(test)]
include!("execution_dispatch/evidence.rs");

mod production {
    use mlx_native::ops::quantized_matmul_ggml::{
        quantized_matmul_ggml as legacy_quantized_matmul_ggml, GgmlQuantizedMatmulParams,
    };
    use mlx_native::{CommandEncoder, KernelRegistry, MlxBuffer, MlxDevice};

    macro_rules! define_fused_gate_up_wrapper {
        ($name:ident, $normal:path, $args:path) => {
            #[allow(clippy::too_many_arguments)]
            pub(in crate::inference::models::qwen35) fn $name(
                encoder: &mut CommandEncoder,
                registry: &mut KernelRegistry,
                device: &MlxDevice,
                gate: &MlxBuffer,
                up: &MlxBuffer,
                input: &MlxBuffer,
                output: &MlxBuffer,
                args: $args,
            ) -> mlx_native::Result<()> {
                $normal(encoder, registry, device, gate, up, input, output, args)
            }
        };
    }

    define_fused_gate_up_wrapper!(
        dispatch_fused_gate_up_silu_q4_k,
        mlx_native::ops::fused_gate_up_silu_q4_K::dispatch_fused_gate_up_silu_q4_K,
        mlx_native::ops::fused_gate_up_silu_q4_K::FusedGateUpSiluQ4_KArgs
    );
    define_fused_gate_up_wrapper!(
        dispatch_fused_gate_up_silu_q5_k,
        mlx_native::ops::fused_gate_up_silu_q5_K::dispatch_fused_gate_up_silu_q5_K,
        mlx_native::ops::fused_gate_up_silu_q5_K::FusedGateUpSiluQ5_KArgs
    );
    define_fused_gate_up_wrapper!(
        dispatch_fused_gate_up_silu_q6_k,
        mlx_native::ops::fused_gate_up_silu_q6_K::dispatch_fused_gate_up_silu_q6_K,
        mlx_native::ops::fused_gate_up_silu_q6_K::FusedGateUpSiluQ6_KArgs
    );
    define_fused_gate_up_wrapper!(
        dispatch_fused_gate_up_silu_q8_0,
        mlx_native::ops::fused_gate_up_silu_q8_0::dispatch_fused_gate_up_silu_q8_0,
        mlx_native::ops::fused_gate_up_silu_q8_0::FusedGateUpSiluQ8_0Args
    );
    define_fused_gate_up_wrapper!(
        dispatch_fused_gate_up_silu_iq4_nl,
        mlx_native::ops::fused_gate_up_silu_iq4_nl::dispatch_fused_gate_up_silu_iq4_nl,
        mlx_native::ops::fused_gate_up_silu_iq4_nl::FusedGateUpSiluIq4NlArgs
    );

    #[allow(clippy::too_many_arguments)]
    pub(in crate::inference::models::qwen35) fn quantized_matmul_ggml(
        encoder: &mut CommandEncoder,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        input: &MlxBuffer,
        weight: &MlxBuffer,
        output: &MlxBuffer,
        params: &GgmlQuantizedMatmulParams,
    ) -> mlx_native::Result<()> {
        legacy_quantized_matmul_ggml(encoder, registry, device, input, weight, output, params)
    }

    pub(in crate::inference::models::qwen35) fn dense_gate_up_fusion_enabled() -> bool {
        !matches!(
            std::env::var("HF2Q_FUSED_GATE_UP_SILU").as_deref(),
            Ok("0") | Ok("false") | Ok("off")
        )
    }

    pub(in crate::inference::models::qwen35) fn fused_qkvg_enabled() -> bool {
        std::env::var("HF2Q_FUSED_QKVG").as_deref() == Ok("1")
    }

    pub(in crate::inference::models::qwen35) fn source_teacher_scope_active() -> bool {
        false
    }

    pub(in crate::inference::models::qwen35) fn dense_q_arena_reset_enabled() -> bool {
        std::env::var("HF2Q_DENSE_Q_ARENA_RESET").as_deref() != Ok("0")
    }

    pub(in crate::inference::models::qwen35) fn dense_q_split_profile_enabled() -> bool {
        std::env::var("HF2Q_PROFILE_DENSE_Q_SPLIT_COMMITS").as_deref() == Ok("1")
    }

    pub(in crate::inference::models::qwen35) fn chunk_scan_prefill_enabled(
        legacy_value: bool,
    ) -> bool {
        legacy_value
    }

    pub(in crate::inference::models::qwen35) fn vec_small_path_enabled() -> bool {
        std::env::var("HF2Q_NO_VEC_SMALL_PATH").as_deref() != Ok("1")
    }

    pub(in crate::inference::models::qwen35) fn fused_stage_ab_vec_enabled() -> bool {
        std::env::var("HF2Q_NO_FUSED_STAGE_AB_VEC").as_deref() != Ok("1")
    }
}

#[cfg(test)]
mod production_boundary_tests {
    use super::production;

    #[test]
    fn release_dispatch_surface_is_complete_and_keeps_plain_chunk_policy() {
        let _ = production::dispatch_fused_gate_up_silu_q4_k;
        let _ = production::dispatch_fused_gate_up_silu_q5_k;
        let _ = production::dispatch_fused_gate_up_silu_q6_k;
        let _ = production::dispatch_fused_gate_up_silu_q8_0;
        let _ = production::dispatch_fused_gate_up_silu_iq4_nl;
        let _ = production::quantized_matmul_ggml;
        let _ = production::dense_gate_up_fusion_enabled;
        let _ = production::fused_qkvg_enabled;
        let _ = production::source_teacher_scope_active;
        let _ = production::dense_q_arena_reset_enabled;
        let _ = production::dense_q_split_profile_enabled;
        let _ = production::vec_small_path_enabled;
        let _ = production::fused_stage_ab_vec_enabled;

        assert!(production::chunk_scan_prefill_enabled(true));
        assert!(!production::chunk_scan_prefill_enabled(false));
        assert!(!production::source_teacher_scope_active());
    }
}

#[cfg(not(test))]
pub(super) use production::*;
