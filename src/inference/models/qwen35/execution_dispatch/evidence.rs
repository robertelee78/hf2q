// Test-only runtime consumption of the canonical dense-Qwen execution
// configuration. The evidence-bearing copied-GGML candidate enters a scoped
// configuration which routes every dense GGML projection through the exact
// resolved policy. The private source-BF16 runner instead enters a distinct
// fixed graph scope which rejects GGML dispatch. Both scopes are thread-local
// because Qwen's Metal cache and forward loop are themselves thread-local;
// neither confers runtime or solver authority by itself.

use anyhow::{bail, Result};
use serde::Serialize;
use std::cell::RefCell;
use std::collections::BTreeMap;

use mlx_native::ops::quantized_matmul_ggml::{
    quantized_matmul_ggml as legacy_quantized_matmul_ggml, quantized_matmul_ggml_with_policy,
    quantized_matmul_ggml_with_policy_and_trace, GgmlQuantizedMatmulParams,
};
use mlx_native::{
    CommandEncoder, GgmlResolvedDispatchTrace, GgmlWorkloadClass, KernelRegistry, MlxBuffer,
    MlxDevice, MlxError,
};

use super::execution_config::{Qwen35ExecutionConfiguration, Qwen35GateUpPolicy};

#[path = "source_scope.rs"]
mod source_scope;

pub(super) use source_scope::{
    source_teacher_graph_policy_sha256, with_source_teacher_graph_scope, SourceTeacherGraphScope,
};

thread_local! {
    static ACTIVE_CONFIGURATION: RefCell<Option<ActiveExecutionState>> =
        const { RefCell::new(None) };
}

struct TraceCaptureState {
    weight_slots: BTreeMap<Qwen35TraceWeightKey, Qwen35TraceWeightSlot>,
    workload: GgmlWorkloadClass,
    observations: Vec<Qwen35EncodedDispatchObservation>,
}

enum ActiveExecutionState {
    CopiedGgml {
        configuration: Qwen35ExecutionConfiguration,
        trace_capture: Option<TraceCaptureState>,
    },
    SourceTeacher,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct Qwen35EncodedDispatchObservation {
    pub operation_id: String,
    pub executed_tensor_node_ids: Vec<String>,
    pub trace: GgmlResolvedDispatchTrace,
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen35TraceWeightSlot {
    pub operation_id: String,
    pub executed_tensor_node_id: String,
}

pub(crate) type Qwen35TraceWeightKey = (usize, u64, usize);

struct ActiveExecutionGuard;

impl Drop for ActiveExecutionGuard {
    fn drop(&mut self) {
        ACTIVE_CONFIGURATION.with(|slot| {
            slot.replace(None);
        });
    }
}

pub(super) fn with_execution_configuration<T>(
    configuration: &Qwen35ExecutionConfiguration,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    configuration.validate()?;
    ACTIVE_CONFIGURATION.with(|slot| -> Result<()> {
        if slot.borrow().is_some() {
            bail!("nested Qwen evidence execution configuration is not admitted");
        }
        slot.replace(Some(ActiveExecutionState::CopiedGgml {
            configuration: configuration.clone(),
            trace_capture: None,
        }));
        Ok(())
    })?;
    let _guard = ActiveExecutionGuard;
    operation()
}

fn with_source_teacher_graph_scope_inner<T>(
    operation: impl FnOnce(&SourceTeacherGraphScope) -> Result<T>,
) -> Result<T> {
    ACTIVE_CONFIGURATION.with(|slot| -> Result<()> {
        if slot.borrow().is_some() {
            bail!("nested Qwen execution scope is not admitted");
        }
        slot.replace(Some(ActiveExecutionState::SourceTeacher));
        Ok(())
    })?;
    let _guard = ActiveExecutionGuard;
    operation(&SourceTeacherGraphScope {
        _not_send: std::marker::PhantomData,
    })
}

/// True only while the current thread owns the canonical source-teacher
/// scope. Source-only helpers use this to replace asynchronous commits with
/// checked waits; ordinary serving retains its existing scheduling.
pub(in crate::inference::models::qwen35) fn source_teacher_scope_active() -> bool {
    ACTIVE_CONFIGURATION.with(|slot| {
        matches!(
            slot.borrow().as_ref(),
            Some(ActiveExecutionState::SourceTeacher)
        )
    })
}

pub(super) fn with_execution_trace_capture<T>(
    configuration: &Qwen35ExecutionConfiguration,
    weight_slots: BTreeMap<Qwen35TraceWeightKey, Qwen35TraceWeightSlot>,
    workload: GgmlWorkloadClass,
    operation: impl FnOnce() -> Result<T>,
) -> Result<(T, Vec<Qwen35EncodedDispatchObservation>)> {
    configuration.validate()?;
    ACTIVE_CONFIGURATION.with(|slot| -> Result<()> {
        if slot.borrow().is_some() {
            bail!("nested Qwen evidence execution configuration is not admitted");
        }
        slot.replace(Some(ActiveExecutionState::CopiedGgml {
            configuration: configuration.clone(),
            trace_capture: Some(TraceCaptureState {
                weight_slots,
                workload,
                observations: Vec::new(),
            }),
        }));
        Ok(())
    })?;
    let guard = ActiveExecutionGuard;
    let value = operation()?;
    let state = ACTIVE_CONFIGURATION.with(|slot| slot.replace(None));
    std::mem::forget(guard);
    let observations = state
        .and_then(|state| match state {
            ActiveExecutionState::CopiedGgml { trace_capture, .. } => trace_capture,
            ActiveExecutionState::SourceTeacher => None,
        })
        .map(|capture| capture.observations)
        .ok_or_else(|| anyhow::anyhow!("Qwen trace capture state disappeared"))?;
    Ok((value, observations))
}

fn buffer_key(buffer: &MlxBuffer) -> Qwen35TraceWeightKey {
    (
        buffer.contents_ptr() as usize,
        buffer.byte_offset(),
        buffer.data_byte_len(),
    )
}

fn fused_gate_up_operation_id(
    capture: &TraceCaptureState,
    gate: &MlxBuffer,
    up: &MlxBuffer,
) -> mlx_native::Result<(String, Vec<String>)> {
    let gate_slot = capture
        .weight_slots
        .get(&buffer_key(gate))
        .ok_or_else(|| MlxError::InvalidArgument("traced gate weight has no Qwen slot".into()))?;
    let up_slot = capture
        .weight_slots
        .get(&buffer_key(up))
        .ok_or_else(|| MlxError::InvalidArgument("traced up weight has no Qwen slot".into()))?;
    let prefix = gate_slot
        .operation_id
        .strip_suffix("ffn_gate.weight")
        .ok_or_else(|| {
            MlxError::InvalidArgument("traced fused gate slot has the wrong semantic role".into())
        })?;
    if up_slot.operation_id != format!("{prefix}ffn_up.weight") {
        return Err(MlxError::InvalidArgument(
            "traced fused gate/up weights belong to different Qwen layers".into(),
        ));
    }
    Ok((
        format!("{prefix}ffn_gate_up_silu"),
        vec![
            gate_slot.executed_tensor_node_id.clone(),
            up_slot.executed_tensor_node_id.clone(),
        ],
    ))
}

fn reject_source_teacher_quantized_dispatch(
    active: Option<&ActiveExecutionState>,
    operation: &str,
) -> mlx_native::Result<()> {
    if matches!(active, Some(ActiveExecutionState::SourceTeacher)) {
        Err(MlxError::InvalidArgument(format!(
            "source-teacher graph scope forbids {operation}"
        )))
    } else {
        Ok(())
    }
}

#[cfg(test)]
fn require_quantized_dispatch_for_test(operation: &str) -> mlx_native::Result<()> {
    ACTIVE_CONFIGURATION
        .with(|slot| reject_source_teacher_quantized_dispatch(slot.borrow().as_ref(), operation))
}

macro_rules! define_fused_gate_up_wrapper {
    ($name:ident, $normal:path, $traced:path, $args:path) => {
        #[allow(clippy::too_many_arguments)]
        pub(super) fn $name(
            encoder: &mut CommandEncoder,
            registry: &mut KernelRegistry,
            device: &MlxDevice,
            gate: &MlxBuffer,
            up: &MlxBuffer,
            input: &MlxBuffer,
            output: &MlxBuffer,
            args: $args,
        ) -> mlx_native::Result<()> {
            ACTIVE_CONFIGURATION.with(|slot| {
                let mut active = slot.borrow_mut();
                if let Some(ActiveExecutionState::CopiedGgml {
                    configuration,
                    trace_capture,
                }) = active.as_mut()
                {
                    if let Some(capture) = trace_capture.as_mut() {
                        let (operation_id, executed_tensor_node_ids) =
                            fused_gate_up_operation_id(capture, gate, up)?;
                        let workload = capture.workload.clone();
                        let trace = $traced(
                            encoder,
                            registry,
                            device,
                            gate,
                            up,
                            input,
                            output,
                            args,
                            configuration.ggml_routing_policy(),
                            workload,
                        )?;
                        capture.observations.push(Qwen35EncodedDispatchObservation {
                            operation_id,
                            executed_tensor_node_ids,
                            trace,
                        });
                        Ok(())
                    } else {
                        $normal(encoder, registry, device, gate, up, input, output, args)
                    }
                } else {
                    reject_source_teacher_quantized_dispatch(
                        active.as_ref(),
                        "fused quantized gate/up dispatch",
                    )?;
                    $normal(encoder, registry, device, gate, up, input, output, args)
                }
            })
        }
    };
}

define_fused_gate_up_wrapper!(
    dispatch_fused_gate_up_silu_q4_k,
    mlx_native::ops::fused_gate_up_silu_q4_K::dispatch_fused_gate_up_silu_q4_K,
    mlx_native::ops::fused_gate_up_silu_q4_K::dispatch_fused_gate_up_silu_q4_K_with_trace,
    mlx_native::ops::fused_gate_up_silu_q4_K::FusedGateUpSiluQ4_KArgs
);
define_fused_gate_up_wrapper!(
    dispatch_fused_gate_up_silu_q6_k,
    mlx_native::ops::fused_gate_up_silu_q6_K::dispatch_fused_gate_up_silu_q6_K,
    mlx_native::ops::fused_gate_up_silu_q6_K::dispatch_fused_gate_up_silu_q6_K_with_trace,
    mlx_native::ops::fused_gate_up_silu_q6_K::FusedGateUpSiluQ6_KArgs
);
define_fused_gate_up_wrapper!(
    dispatch_fused_gate_up_silu_q8_0,
    mlx_native::ops::fused_gate_up_silu_q8_0::dispatch_fused_gate_up_silu_q8_0,
    mlx_native::ops::fused_gate_up_silu_q8_0::dispatch_fused_gate_up_silu_q8_0_with_trace,
    mlx_native::ops::fused_gate_up_silu_q8_0::FusedGateUpSiluQ8_0Args
);
define_fused_gate_up_wrapper!(
    dispatch_fused_gate_up_silu_iq4_nl,
    mlx_native::ops::fused_gate_up_silu_iq4_nl::dispatch_fused_gate_up_silu_iq4_nl,
    mlx_native::ops::fused_gate_up_silu_iq4_nl::dispatch_fused_gate_up_silu_iq4_nl_with_trace,
    mlx_native::ops::fused_gate_up_silu_iq4_nl::FusedGateUpSiluIq4NlArgs
);

/// Dense GGML dispatch used by every admitted Qwen text projection.
///
/// Under an evidence configuration this consumes the exact policy resolved at
/// load time.  Outside that scope it delegates to the legacy entrypoint and
/// therefore preserves existing serving behavior.
#[allow(clippy::too_many_arguments)]
pub(super) fn quantized_matmul_ggml(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> mlx_native::Result<()> {
    ACTIVE_CONFIGURATION.with(|slot| {
        let mut active = slot.borrow_mut();
        if let Some(ActiveExecutionState::CopiedGgml {
            configuration,
            trace_capture,
        }) = active.as_mut()
        {
            if let Some(capture) = trace_capture.as_mut() {
                let weight_slot = capture
                    .weight_slots
                    .get(&buffer_key(weight))
                    .cloned()
                    .ok_or_else(|| {
                        MlxError::InvalidArgument(
                            "Qwen traced GGML weight is absent from the executed catalog".into(),
                        )
                    })?;
                let trace = quantized_matmul_ggml_with_policy_and_trace(
                    encoder,
                    registry,
                    device,
                    input,
                    weight,
                    output,
                    params,
                    configuration.ggml_routing_policy(),
                    capture.workload.clone(),
                )?;
                capture.observations.push(Qwen35EncodedDispatchObservation {
                    operation_id: weight_slot.operation_id,
                    executed_tensor_node_ids: vec![weight_slot.executed_tensor_node_id],
                    trace,
                });
                Ok(())
            } else {
                quantized_matmul_ggml_with_policy(
                    encoder,
                    registry,
                    device,
                    input,
                    weight,
                    output,
                    params,
                    configuration.ggml_routing_policy(),
                )
            }
        } else {
            reject_source_teacher_quantized_dispatch(active.as_ref(), "GGML projection dispatch")?;
            legacy_quantized_matmul_ggml(encoder, registry, device, input, weight, output, params)
        }
    })
}

pub(super) fn dense_gate_up_fusion_enabled() -> bool {
    ACTIVE_CONFIGURATION.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|active| match active {
                ActiveExecutionState::CopiedGgml { configuration, .. } => {
                    configuration.dense_ffn_gate_up()
                        == Qwen35GateUpPolicy::PreferFusedWhenSupported
                }
                ActiveExecutionState::SourceTeacher => false,
            })
            .unwrap_or_else(|| {
                !matches!(
                    std::env::var("HF2Q_FUSED_GATE_UP_SILU").as_deref(),
                    Ok("0") | Ok("false") | Ok("off")
                )
            })
    })
}

pub(super) fn fused_qkvg_enabled() -> bool {
    ACTIVE_CONFIGURATION.with(|slot| {
        if slot.borrow().is_some() {
            false
        } else {
            std::env::var("HF2Q_FUSED_QKVG").as_deref() == Ok("1")
        }
    })
}

pub(super) fn dense_q_arena_reset_enabled() -> bool {
    ACTIVE_CONFIGURATION.with(|slot| {
        if slot.borrow().is_some() {
            true
        } else {
            std::env::var("HF2Q_DENSE_Q_ARENA_RESET").as_deref() != Ok("0")
        }
    })
}

pub(super) fn dense_q_split_profile_enabled() -> bool {
    ACTIVE_CONFIGURATION.with(|slot| {
        if slot.borrow().is_some() {
            false
        } else {
            std::env::var("HF2Q_PROFILE_DENSE_Q_SPLIT_COMMITS").as_deref() == Ok("1")
        }
    })
}

pub(super) fn chunk_scan_prefill_enabled(legacy_value: bool) -> bool {
    ACTIVE_CONFIGURATION.with(|slot| {
        if slot.borrow().is_some() {
            false
        } else {
            legacy_value
        }
    })
}

pub(super) fn vec_small_path_enabled() -> bool {
    ACTIVE_CONFIGURATION.with(|slot| {
        if slot.borrow().is_some() {
            true
        } else {
            std::env::var("HF2Q_NO_VEC_SMALL_PATH").as_deref() != Ok("1")
        }
    })
}

pub(super) fn fused_stage_ab_vec_enabled() -> bool {
    ACTIVE_CONFIGURATION.with(|slot| {
        if slot.borrow().is_some() {
            true
        } else {
            std::env::var("HF2Q_NO_FUSED_STAGE_AB_VEC").as_deref() != Ok("1")
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_native::{DType, GgmlRoutingPolicy, MlxDevice};

    #[test]
    fn trace_weight_identity_distinguishes_views_and_collapses_clones() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let Ok(device) = MlxDevice::new() else { return };
        let parent = device
            .alloc_buffer(64, DType::U8, vec![64])
            .expect("allocate trace identity fixture");
        let first = parent.slice_view(0, 32);
        let second = parent.slice_view(32, 32);
        assert_ne!(buffer_key(&first), buffer_key(&second));
        assert_eq!(buffer_key(&first), buffer_key(&first.clone()));
    }

    #[test]
    fn scoped_graph_switches_are_canonical_and_environment_independent() {
        let configuration = Qwen35ExecutionConfiguration::from_resolved(
            GgmlRoutingPolicy::default(),
            Qwen35GateUpPolicy::Separate,
        )
        .unwrap();
        with_execution_configuration(&configuration, || {
            assert!(!dense_gate_up_fusion_enabled());
            assert!(!fused_qkvg_enabled());
            assert!(dense_q_arena_reset_enabled());
            assert!(!dense_q_split_profile_enabled());
            assert!(!chunk_scan_prefill_enabled(true));
            assert!(vec_small_path_enabled());
            assert!(fused_stage_ab_vec_enabled());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn scope_is_not_reentrant_and_drains_after_error() {
        let configuration = Qwen35ExecutionConfiguration::from_resolved(
            GgmlRoutingPolicy::default(),
            Qwen35GateUpPolicy::PreferFusedWhenSupported,
        )
        .unwrap();
        let error: Result<()> = with_execution_configuration(&configuration, || {
            assert!(with_execution_configuration(&configuration, || Ok(())).is_err());
            bail!("synthetic operation failure")
        });
        assert!(error.is_err());
        with_execution_configuration(&configuration, || Ok(())).unwrap();
    }
}
