//! Fixed graph/TLS scope for the source-precision dense-Qwen teacher path.
//!
//! This scope freezes only the hf2q graph switches routed through
//! `execution_dispatch`. Other diagnostic and native BF16 tensor-MM choices
//! remain outside it, so this is not a complete graph, kernel-route, or
//! execution receipt.

use anyhow::Result;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::marker::PhantomData;
use std::rc::Rc;

const SOURCE_TEACHER_GRAPH_PROFILE: &str = "dense_qwen35_source_bf16_graph_scope_v1";

/// Lifetime-bound, non-Clone and non-Send proof that the canonical source
/// graph scope is active on the current thread.
pub(in crate::inference::models::qwen35) struct SourceTeacherGraphScope {
    pub(super) _not_send: PhantomData<Rc<()>>,
}

#[derive(Serialize)]
struct SourceTeacherGraphPolicyV1 {
    profile: &'static str,
    weight_precision: &'static str,
    control_precision: &'static str,
    base_text_only: bool,
    ggml: bool,
    q4_repack: bool,
    dwq: bool,
    tq: bool,
    mtp: bool,
    vision: bool,
    chunk_scan_prefill: bool,
    fused_qkvg: bool,
    fused_quantized_gate_up: bool,
    dense_q_split_profile: bool,
    dense_q_arena_reset: bool,
    vec_small_path: bool,
    fused_stage_ab_vec: bool,
    complete_native_route_bound: bool,
}

fn policy() -> SourceTeacherGraphPolicyV1 {
    SourceTeacherGraphPolicyV1 {
        profile: SOURCE_TEACHER_GRAPH_PROFILE,
        weight_precision: "bf16",
        control_precision: "f32",
        base_text_only: true,
        ggml: false,
        q4_repack: false,
        dwq: false,
        tq: false,
        mtp: false,
        vision: false,
        chunk_scan_prefill: false,
        fused_qkvg: false,
        fused_quantized_gate_up: false,
        dense_q_split_profile: false,
        dense_q_arena_reset: true,
        vec_small_path: true,
        fused_stage_ab_vec: true,
        complete_native_route_bound: false,
    }
}

pub(in crate::inference::models::qwen35) fn source_teacher_graph_policy_sha256() -> Result<String> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(&policy())?)))
}

pub(in crate::inference::models::qwen35) fn with_source_teacher_graph_scope<T>(
    operation: impl FnOnce(&SourceTeacherGraphScope) -> Result<T>,
) -> Result<T> {
    super::with_source_teacher_graph_scope_inner(operation)
}

#[cfg(test)]
mod tests {
    use anyhow::{bail, Result};

    use super::*;
    use crate::inference::models::qwen35::execution_config::{
        Qwen35ExecutionConfiguration, Qwen35GateUpPolicy,
    };
    use mlx_native::GgmlRoutingPolicy;

    #[test]
    fn source_teacher_scope_is_canonical_nonreentrant_and_error_safe() {
        let first = source_teacher_graph_policy_sha256().unwrap();
        let second = source_teacher_graph_policy_sha256().unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 64);

        let copied = Qwen35ExecutionConfiguration::from_resolved(
            GgmlRoutingPolicy::default(),
            Qwen35GateUpPolicy::Separate,
        )
        .unwrap();
        let error: Result<()> = with_source_teacher_graph_scope(|_| {
            assert!(!super::super::dense_gate_up_fusion_enabled());
            assert!(!super::super::fused_qkvg_enabled());
            assert!(super::super::dense_q_arena_reset_enabled());
            assert!(!super::super::dense_q_split_profile_enabled());
            assert!(!super::super::chunk_scan_prefill_enabled(true));
            assert!(super::super::vec_small_path_enabled());
            assert!(super::super::fused_stage_ab_vec_enabled());
            assert!(with_source_teacher_graph_scope(|_| Ok(())).is_err());
            assert!(super::super::with_execution_configuration(&copied, || Ok(())).is_err());
            for operation in [
                "GGML projection dispatch",
                "fused q4_k gate/up dispatch",
                "fused q6_k gate/up dispatch",
                "fused q8_0 gate/up dispatch",
                "fused iq4_nl gate/up dispatch",
            ] {
                assert!(super::super::require_quantized_dispatch_for_test(operation).is_err());
            }
            bail!("synthetic source-scope failure")
        });
        assert!(error.is_err());
        assert!(super::super::require_quantized_dispatch_for_test("GGML projection").is_ok());
        with_source_teacher_graph_scope(|_| Ok(())).unwrap();
        super::super::with_execution_configuration(&copied, || {
            assert!(with_source_teacher_graph_scope(|_| Ok(())).is_err());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn source_teacher_scope_clears_after_panic_unwind() {
        let panic = std::panic::catch_unwind(|| {
            with_source_teacher_graph_scope(|_| -> Result<()> {
                panic!("synthetic source-scope panic")
            })
            .unwrap();
        });
        assert!(panic.is_err());
        with_source_teacher_graph_scope(|_| Ok(())).unwrap();
    }
}
