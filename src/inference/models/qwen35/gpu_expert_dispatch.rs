//! Typed routed-expert dispatch; storage never changes the selected arithmetic.
use anyhow::{anyhow, bail, Result};
use mlx_native::ops::quantized_matmul_id_ggml::{self as native, GgmlQuantizedMatmulIdParams, IdMmScratch, quantized_matmul_id_ggml_pooled};
use mlx_native::{CommandEncoder, GgmlRoutingPolicy, KernelRegistry, MlxBuffer, MlxDevice};
use super::expert_storage::ExpertRoleRef;
#[cfg(test)]
use super::expert_storage::{ExpertRole, MoeExpertStorage};

fn routing_policy(registry: &KernelRegistry) -> GgmlRoutingPolicy {
    registry.ggml_routing_policy().copied().unwrap_or_else(mlx_native::ggml_routing_policy_from_environment)
}

/// Borrow a GGML fixture without retaining an obsolete source in an affine role.
#[cfg(test)]
pub(super) fn raw_expert(storage: &MoeExpertStorage, role: ExpertRole) -> Result<&MlxBuffer> {
    match storage.role(role) {
        ExpertRoleRef::Ggml(weight) => Ok(weight),
        _ => bail!("GGML fixture access requires raw GGML expert storage"),
    }
}

/// The Q4_0 tree route also accepts the loaded DWQ overlay domain. Validate
/// the representation actually dispatched, not a retired GGML source.
#[allow(clippy::too_many_arguments)]
pub(super) fn validate_tree_expert(
    role: ExpertRoleRef<'_>, n_experts: usize, n: usize, k: usize,
    raw_bytes: usize, label: &str,
) -> Result<()> {
    use mlx_native::DType;
    match role {
        ExpertRoleRef::Ggml(weight) => {
            anyhow::ensure!(weight.dtype() == DType::U8,
                "qwen35_tree_verify_full_layer_q_moe: {label} dtype must be U8");
            anyhow::ensure!(weight.element_count() == raw_bytes,
                "qwen35_tree_verify_full_layer_q_moe: {label} has {} bytes, expected exactly {raw_bytes}",
                weight.element_count());
        }
        ExpertRoleRef::Affine(stack) => {
            anyhow::ensure!(stack.num_experts == n_experts && stack.n == n && stack.k == k,
                "qwen35_tree_verify_full_layer_q_moe: {label} affine geometry [{}, {}, {}] must match [{n_experts}, {n}, {k}]",
                stack.num_experts, stack.n, stack.k);
            // The Qwen DWQ loader publishes packed-U32, four-bit/group-32 stacks.
            anyhow::ensure!(stack.bits == 4 && stack.group_size == 32 && k % 32 == 0,
                "qwen35_tree_verify_full_layer_q_moe: {label} affine storage requires bits=4, group_size=32 and aligned K");
            let rows = n_experts.checked_mul(n)
                .ok_or_else(|| anyhow!("{label} affine row count overflows usize"))?;
            let words = rows.checked_mul(k / 8)
                .ok_or_else(|| anyhow!("{label} affine word count overflows usize"))?;
            let groups = rows.checked_mul(k / 32)
                .ok_or_else(|| anyhow!("{label} affine group count overflows usize"))?;
            anyhow::ensure!(stack.weight.dtype() == DType::U32 && stack.weight.element_count() == words,
                "qwen35_tree_verify_full_layer_q_moe: {label} affine weight must contain exactly {words} U32 words");
            for (name, buffer) in [("scales", &stack.scales), ("biases", &stack.biases)] {
                anyhow::ensure!(buffer.dtype() == DType::BF16 && buffer.element_count() == groups,
                    "qwen35_tree_verify_full_layer_q_moe: {label} affine {name} must contain exactly {groups} BF16 values");
            }
        }
        ExpertRoleRef::Prepared(_) => bail!(
            "qwen35_tree_verify_full_layer_q_moe: {label} does not support prepared expert storage"
        ),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_moe_id_routed(
    enc: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: ExpertRoleRef<'_>,
    ids: &MlxBuffer,
    output: &MlxBuffer,
    legacy_params: &GgmlQuantizedMatmulIdParams,
    pool_slot: super::decode_pool::MmIdSlot,
    pool_n_experts: u32,
    pool_rows: u32,
    label: &str,
    // ADR-033 §Pi Phase B Stage 3b.2 — imatrix intercept hint
    // (Layered{tag,layer} for GGML-quant MoE expert tensors, or None
    // to skip the intercept for paths where MoE imatrix collection
    // isn't applicable). The affine-stack branch below is NOT
    // intercepted yet — affine MoE GGUFs are a separate format and
    // their imatrix wiring lands in Stage 3b.3 when an operator
    // ships imatrix-aware DWQ. Per [[feedback-no-stub-2026-05-19]]
    // we accept the gap as documented scope, not as a silent
    // fallback.
    imatrix_hint: crate::quantize::imatrix::ImatrixHint<'_>,
) -> anyhow::Result<()> {
    if let ExpertRoleRef::Prepared(planes) = weight {
        anyhow::ensure!(!crate::quantize::imatrix::is_active(), "canonical expert storage cannot collect imatrix inputs");
        let routing = routing_policy(registry);
        return super::decode_pool::with_id_mm_scratch(
            pool_slot, device, pool_n_experts, pool_rows, |scratch| {
                native::quantized_matmul_id_ggml_prepared_pooled_with_policy(
                    enc, registry, device, input, planes, ids, output, scratch, legacy_params, &routing,
                )
            },
        ).map_err(|e| anyhow!("{label} prepared qmatmul_id_pooled: {e}"));
    }
    if let ExpertRoleRef::Affine(stack) = weight {
        // Affine path: kernel writes into `output` directly.  M/N/K
        // come from legacy_params (truth source for shape — the
        // affine stack carries n/k for shape validation only).
        debug_assert_eq!(
            stack.n as u32, legacy_params.n,
            "{label}: affine stack n ({}) != legacy_params.n ({})",
            stack.n, legacy_params.n
        );
        debug_assert_eq!(
            stack.k as u32, legacy_params.k,
            "{label}: affine stack k ({}) != legacy_params.k ({})",
            stack.k, legacy_params.k
        );
        let m = legacy_params.n_tokens;
        mlx_native::quantized_matmul_id_into(
            enc,
            registry,
            device,
            input,
            &stack.weight,
            &stack.scales,
            &stack.biases,
            ids,
            output,
            &mlx_native::QuantizedMatmulIdParams {
                m,
                k: legacy_params.k,
                n: legacy_params.n,
                group_size: stack.group_size,
                bits: stack.bits,
                n_expert_used: legacy_params.top_k,
                num_experts: legacy_params.n_experts,
            },
        )
        .map_err(|e| anyhow!("{label} qmatmul_id_into (affine): {e}"))
    } else {
        let ExpertRoleRef::Ggml(legacy_weight) = weight else { unreachable!() };
        // ADR-033 §Pi Phase B Stage 3b.2 — imatrix intercept BEFORE
        // the GGML-quant MoE-id dispatch. Captures the per-token
        // shared input + per-token routed expert IDs and fires
        // `record_moe` for each (token, expert) pair. No-op when no
        // collector installed (production decode/serve path).
        crate::quantize::imatrix::intercept_qmatmul_id_with_hint(
            imatrix_hint,
            legacy_params.n_tokens as usize,
            legacy_params.top_k as usize,
            legacy_params.k as usize,
            || {
                // `commit_wait_and_rotate` syncs the input buffer
                // (so we can safely read F32 host-side) AND rotates
                // to a fresh CB so the matmul dispatch below doesn't
                // hit the Metal `MTLCommandBufferStatusCommitted`
                // assertion at `setCurrentCommandEncoder:` line 323.
                if let Err(e) = enc.commit_wait_and_rotate() {
                    eprintln!(
                        "[hf2q imatrix moe intercept ({label})] commit_wait_and_rotate failed: {e}"
                    );
                    return None;
                }
                input.as_slice::<f32>().ok().map(|sl| sl.to_vec())
            },
            || ids.as_slice::<u32>().ok().map(|sl| sl.to_vec()),
        )
        .map_err(|e| anyhow!("imatrix moe intercept ({label}): {e}"))?;

        // Legacy path: pooled scratch + GGML kernel.  Same body as
        // pre-Iter C2.4 #4 (preserved for byte-identity on non-overlay
        // serves).
        super::decode_pool::with_id_mm_scratch(
            pool_slot,
            device,
            pool_n_experts,
            pool_rows,
            |scratch| {
                quantized_matmul_id_ggml_pooled(
                    enc,
                    registry,
                    device,
                    input,
                    legacy_weight,
                    ids,
                    output,
                    scratch,
                    legacy_params,
                )
            },
        )
        .map_err(|e| anyhow!("{label} qmatmul_id_pooled: {e}"))
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_expert_pair(
    enc: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, gate: ExpertRoleRef<'_>, up: ExpertRoleRef<'_>,
    ids: &MlxBuffer, gate_output: &MlxBuffer, up_output: &MlxBuffer,
    scratch: &mut IdMmScratch, params: &GgmlQuantizedMatmulIdParams,
) -> mlx_native::Result<()> {
    if crate::quantize::imatrix::is_active() && matches!(&gate, ExpertRoleRef::Prepared(_)) {
        return Err(mlx_native::MlxError::InvalidArgument("canonical expert storage cannot collect imatrix inputs".into()));
    }
    let routing = routing_policy(registry);
    match (gate, up) {
        (ExpertRoleRef::Ggml(gate), ExpertRoleRef::Ggml(up)) =>
            native::quantized_matmul_id_ggml_pooled_pair_with_policy(
                enc, registry, device, input, gate, up, ids, gate_output, up_output,
                scratch, params, &routing,
            ),
        (ExpertRoleRef::Prepared(gate), ExpertRoleRef::Prepared(up)) =>
            native::quantized_matmul_id_ggml_prepared_pooled_pair_with_policy(
                enc, registry, device, input, gate, up, ids, gate_output, up_output,
                scratch, params, &routing,
            ),
        _ => Err(mlx_native::MlxError::InvalidArgument(
            "paired expert schedule requires matching GGML or canonical storage".into(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_expert_slotted(
    enc: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, weight: ExpertRoleRef<'_>, ids: &MlxBuffer, output: &MlxBuffer,
    scratch: &mut IdMmScratch, params: &GgmlQuantizedMatmulIdParams,
) -> mlx_native::Result<()> {
    if crate::quantize::imatrix::is_active() && matches!(&weight, ExpertRoleRef::Prepared(_)) {
        return Err(mlx_native::MlxError::InvalidArgument("canonical expert storage cannot collect imatrix inputs".into()));
    }
    let routing = routing_policy(registry);
    match weight {
        ExpertRoleRef::Ggml(weight) =>
            native::quantized_matmul_id_ggml_pooled_slotted_reusing_schedule_with_policy(
                enc, registry, device, input, weight, ids, output, scratch, params, &routing,
            ),
        ExpertRoleRef::Prepared(weight) =>
            native::quantized_matmul_id_ggml_prepared_pooled_slotted_reusing_schedule_with_policy(
                enc, registry, device, input, weight, ids, output, scratch, params, &routing,
            ),
        ExpertRoleRef::Affine(_) => Err(mlx_native::MlxError::InvalidArgument(
            "slotted expert schedule does not support affine storage".into(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_expert_fused(
    enc: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, gate: ExpertRoleRef<'_>, up: ExpertRoleRef<'_>,
    ids: &MlxBuffer, output: &MlxBuffer, scratch: &mut IdMmScratch,
    params: &GgmlQuantizedMatmulIdParams,
) -> mlx_native::Result<()> {
    if crate::quantize::imatrix::is_active() && matches!(&gate, ExpertRoleRef::Prepared(_)) {
        return Err(mlx_native::MlxError::InvalidArgument("canonical expert storage cannot collect imatrix inputs".into()));
    }
    match (gate, up) {
        (ExpertRoleRef::Ggml(gate), ExpertRoleRef::Ggml(up)) => {
            let dispatch = native::GgmlIdMmDispatchParams {
                n_tokens: params.n_tokens, top_k: params.top_k, n: params.n, k: params.k,
                n_experts: params.n_experts, expert_stride: params.expert_stride,
                ggml_type: params.ggml_type,
            };
            let (htpe, hids) = scratch.buffers_mut();
            native::dispatch_id_mm_fused_gate_up_silu_for_test(
                enc, registry, device, input, gate, up, ids, htpe, hids, output, &dispatch,
            )
        }
        (ExpertRoleRef::Prepared(gate), ExpertRoleRef::Prepared(up)) =>
            native::quantized_matmul_id_ggml_prepared_fused_gate_up_silu_mm_q6_k(
                enc, registry, device, input, gate, up, ids, output, scratch, params, false,
            ),
        _ => Err(mlx_native::MlxError::InvalidArgument(
            "fused expert projection requires matching GGML or canonical storage".into(),
        )),
    }
}
