//! Payload- and Metal-device-free GGUF execution preflight for dense
//! Qwen3.5-family models.
//!
//! A GGUF's `general.file_type` is only a summary: mixed artifacts can place
//! different codecs on embeddings, attention projections, FFN projections,
//! and the output head. Production admission therefore follows the exact
//! tensor role consumed by the Qwen graph. This module reads tensor-directory
//! metadata only; it never creates a Metal device or loads tensor payloads.

use anyhow::{anyhow, bail, ensure, Context, Result};
use mlx_native::gguf::{GgufFile, TensorInfo};
use mlx_native::{
    DType, DenseMatmulIdInputLayout, DenseMatmulIdMultiplicity, DenseMatmulIdParams,
    DenseMatmulIdRoute, GgmlCapabilityRequest, GgmlExpertInputLayout, GgmlExpertShape,
    GgmlInvocation, GgmlRoutingPolicy, GgmlWorkloadClass, GGML_CAPABILITY_SCHEMA_VERSION,
};

use super::ffn::MoeExpertGeometry;
use super::weight_loader::validate_native_row_projection_info;
use super::{Qwen35Config, Qwen35LayerKind, Qwen35Variant};

mod contract;
use contract::{
    admit_mtp_tensor_presence, admit_storage_for_role, Qwen35GgufPreflightReceipt, TensorRole,
    TensorStorage,
};

#[cfg(test)]
pub(super) fn admitted_matrix_codecs() -> &'static [mlx_native::GgmlType] {
    contract::QWEN_ADMITTED_MATRIX_CODECS
}

/// Every matrix width exercised by the production scheduler. Keep the
/// non-power-of-two continuous/prompt widths and the first prompt width above
/// 16 in
/// preflight so admission cannot succeed on a narrower kernel contract than
/// serving uses.
const REQUIRED_MATRIX_WIDTHS: [(u32, GgmlWorkloadClass); 8] = [
    (1, GgmlWorkloadClass::DecodeSingle),
    (2, GgmlWorkloadClass::ContinuousWidth),
    (3, GgmlWorkloadClass::ContinuousWidth),
    (4, GgmlWorkloadClass::ContinuousWidth),
    (8, GgmlWorkloadClass::ContinuousWidth),
    (9, GgmlWorkloadClass::Prompt),
    (16, GgmlWorkloadClass::Prompt),
    (17, GgmlWorkloadClass::Prompt),
];

/// Exact largest source-token chunks admitted by Qwen serving: the multi-slot
/// scheduler width and the single-slot ceiling. These are expert-only because
/// routed scratch/row expansion must be checked at the graph's actual maximum.
const REQUIRED_EXPERT_SCHEDULER_WIDTHS: [u32; 2] = [2_048, 4_096];

/// Physical source-row widths whose scalar expert route is timed when a model
/// becomes live. Widths 1..=16 are the supported simultaneous decode surface;
/// the two larger values are the exact scheduler chunk contracts above.
pub(crate) fn scalar_expert_activation_widths() -> Vec<u32> {
    (1..=16).chain(REQUIRED_EXPERT_SCHEDULER_WIDTHS).collect()
}

fn checked_tensor_bytes(info: &TensorInfo) -> Result<usize> {
    let (&inner, outer) = info
        .shape
        .split_last()
        .ok_or_else(|| anyhow!("tensor '{}' has an empty shape", info.name))?;
    let block_values = info.ggml_type.block_values() as usize;
    ensure!(
        inner > 0 && !outer.contains(&0),
        "tensor '{}' has a zero dimension in {:?}",
        info.name,
        info.shape
    );
    ensure!(
        inner % block_values == 0,
        "tensor '{}' innermost dimension {inner} is not aligned to {:?}'s {block_values}-value blocks",
        info.name,
        info.ggml_type
    );
    outer
        .iter()
        .try_fold(inner / block_values, |elements, dim| {
            elements.checked_mul(*dim)
        })
        .and_then(|blocks| blocks.checked_mul(info.ggml_type.block_bytes() as usize))
        .ok_or_else(|| anyhow!("tensor '{}' byte length overflows usize", info.name))
}

fn require_tensor<'a>(
    gguf: &'a GgufFile,
    name: &str,
    expected_shape: &[usize],
    role: TensorRole,
    receipt: &mut Qwen35GgufPreflightReceipt,
) -> Result<&'a TensorInfo> {
    let info = gguf
        .tensor_info(name)
        .ok_or_else(|| anyhow!("Qwen GGUF preflight: required tensor '{name}' is missing"))?;
    ensure!(
        info.shape == expected_shape,
        "Qwen GGUF preflight: tensor '{name}' shape {:?} != expected {:?}",
        info.shape,
        expected_shape
    );
    let expected_bytes = checked_tensor_bytes(info)?;
    ensure!(
        info.byte_len == expected_bytes,
        "Qwen GGUF preflight: tensor '{name}' byte length {} != {expected_bytes} for {:?} shape {:?}",
        info.byte_len,
        info.ggml_type,
        info.shape
    );
    admit_storage_for_role(name, role, TensorStorage::Parsed(info.ggml_type))?;
    receipt.record_tensor(role, TensorStorage::Parsed(info.ggml_type), expected_bytes)?;
    Ok(info)
}

fn require_f32(
    gguf: &GgufFile,
    name: &str,
    shape: &[usize],
    receipt: &mut Qwen35GgufPreflightReceipt,
) -> Result<()> {
    require_tensor(gguf, name, shape, TensorRole::F32State, receipt)?;
    Ok(())
}

fn require_f32_matrix(
    gguf: &GgufFile,
    name: &str,
    shape: &[usize],
    receipt: &mut Qwen35GgufPreflightReceipt,
) -> Result<()> {
    require_tensor(gguf, name, shape, TensorRole::F32Matrix, receipt)?;
    Ok(())
}

fn ensure_dense_capability_dims(
    name: &str,
    info: &TensorInfo,
    rows: usize,
    cols: usize,
    routing: GgmlRoutingPolicy,
) -> Result<()> {
    let n = u32::try_from(rows).context("Qwen dense projection rows exceed u32")?;
    let k = u32::try_from(cols).context("Qwen dense projection cols exceed u32")?;
    for (m, workload) in REQUIRED_MATRIX_WIDTHS {
        let capability = mlx_native::ggml_capability(GgmlCapabilityRequest {
            schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
            invocation: GgmlInvocation::DenseAuto { m, n, k },
            ggml_type: info.ggml_type,
            workload,
            routing,
        });
        ensure!(
            capability.executable,
            "Qwen GGUF preflight: tensor '{name}' type {:?} is not executable at M={m} ({workload:?}): {}",
            info.ggml_type,
            capability.diagnostic
        );
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn ensure_dense_capability(name: &str, info: &TensorInfo) -> Result<()> {
    ensure_dense_capability_with_routing(
        name,
        info,
        mlx_native::ggml_routing_policy_from_environment(),
    )
}

fn ensure_dense_capability_with_routing(
    name: &str,
    info: &TensorInfo,
    routing: GgmlRoutingPolicy,
) -> Result<()> {
    let [rows, cols] = info.shape.as_slice() else {
        bail!(
            "Qwen GGUF preflight: dense tensor '{name}' must be rank 2, got {:?}",
            info.shape
        );
    };
    ensure_dense_capability_dims(name, info, *rows, *cols, routing)
}

fn require_projection(
    gguf: &GgufFile,
    name: &str,
    shape: &[usize],
    receipt: &mut Qwen35GgufPreflightReceipt,
    routing: GgmlRoutingPolicy,
) -> Result<()> {
    let info = require_tensor(gguf, name, shape, TensorRole::DenseProjection, receipt)?;
    ensure_dense_capability_with_routing(name, info, routing)
}

/// The shared-expert sigmoid gate is the sole Qwen matrix role whose GGUF
/// schema is a rank-1 row vector. It executes as logical `[1, hidden]` but the
/// stored payload stays exact; rank-2 or any other squeeze is rejected here.
fn require_row_projection(
    gguf: &GgufFile,
    name: &str,
    cols: usize,
    receipt: &mut Qwen35GgufPreflightReceipt,
    routing: GgmlRoutingPolicy,
) -> Result<()> {
    let info = require_tensor(gguf, name, &[cols], TensorRole::DenseProjection, receipt)?;
    validate_native_row_projection_info(name, info, cols)?;
    ensure_dense_capability_dims(name, info, 1, cols, routing)
}

fn require_embedding(
    gguf: &GgufFile,
    name: &str,
    hidden: usize,
    receipt: &mut Qwen35GgufPreflightReceipt,
    routing: GgmlRoutingPolicy,
) -> Result<usize> {
    let info = gguf
        .tensor_info(name)
        .ok_or_else(|| anyhow!("Qwen GGUF preflight: required tensor '{name}' is missing"))?;
    ensure!(
        info.shape.len() == 2 && info.shape[1] == hidden,
        "Qwen GGUF preflight: tensor '{name}' shape {:?} is not [rows,{hidden}]",
        info.shape
    );
    let rows = info.shape[0];
    require_tensor(gguf, name, &[rows, hidden], TensorRole::Embedding, receipt)?;

    let capability = mlx_native::ggml_capability(GgmlCapabilityRequest {
        schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
        invocation: GgmlInvocation::EmbeddingGather {
            n_tokens: 1,
            vocab_size: u32::try_from(rows).context("Qwen embedding rows exceed u32")?,
            embed_dim: u32::try_from(hidden).context("Qwen embedding width exceeds u32")?,
        },
        ggml_type: info.ggml_type,
        workload: GgmlWorkloadClass::Embedding,
        routing,
    });
    ensure!(
        capability.executable,
        "Qwen GGUF preflight: tensor '{name}' type {:?} has no native embedding route: {}",
        info.ggml_type,
        capability.diagnostic
    );
    Ok(rows)
}

fn require_dense_ffn(
    gguf: &GgufFile,
    layer_index: u32,
    hidden: usize,
    intermediate: usize,
    receipt: &mut Qwen35GgufPreflightReceipt,
    routing: GgmlRoutingPolicy,
) -> Result<()> {
    let prefix = format!("blk.{layer_index}");
    let gate_name = format!("{prefix}.ffn_gate.weight");
    let up_name = format!("{prefix}.ffn_up.weight");
    let down_name = format!("{prefix}.ffn_down.weight");
    let gate = require_tensor(
        gguf,
        &gate_name,
        &[intermediate, hidden],
        TensorRole::FfnGateUp,
        receipt,
    )?;
    let up = require_tensor(
        gguf,
        &up_name,
        &[intermediate, hidden],
        TensorRole::FfnGateUp,
        receipt,
    )?;
    let down = require_tensor(
        gguf,
        &down_name,
        &[hidden, intermediate],
        TensorRole::FfnDown,
        receipt,
    )?;

    super::weight_loader::dense_ffn_storage(
        layer_index,
        gate.ggml_type,
        up.ggml_type,
        down.ggml_type,
    )
    .with_context(|| format!("Qwen GGUF preflight: dense FFN layer {layer_index}"))?;
    for (name, info) in [(&gate_name, gate), (&up_name, up), (&down_name, down)] {
        ensure_dense_capability_with_routing(name, info, routing)?;
    }
    Ok(())
}

fn ensure_expert_capability(
    name: &str,
    info: &TensorInfo,
    n: usize,
    k: usize,
    top_k: u32,
    n_experts: u32,
    execution: ExpertExecution,
    routing: GgmlRoutingPolicy,
) -> Result<()> {
    ensure!(n_experts > 0, "Qwen expert stack has zero experts");
    ensure!(
        info.byte_len % n_experts as usize == 0,
        "Qwen expert tensor '{name}' byte length {} is not divisible by {n_experts} experts",
        info.byte_len
    );
    let expert_stride_bytes = u64::try_from(info.byte_len / n_experts as usize)
        .context("Qwen expert stride exceeds u64")?;
    let n = u32::try_from(n).context("Qwen expert output rows exceed u32")?;
    let k = u32::try_from(k).context("Qwen expert input cols exceed u32")?;
    if matches!(
        info.ggml_type,
        mlx_native::GgmlType::F32 | mlx_native::GgmlType::F16 | mlx_native::GgmlType::BF16
    ) {
        let dtype = match info.ggml_type {
            mlx_native::GgmlType::F32 => DType::F32,
            mlx_native::GgmlType::F16 => DType::F16,
            mlx_native::GgmlType::BF16 => DType::BF16,
            _ => unreachable!(),
        };
        for source_tokens in required_expert_source_widths(execution, top_k, routing)? {
            let params = DenseMatmulIdParams {
                m: source_tokens,
                n,
                k,
                top_k,
                n_experts,
                expert_stride_bytes,
                input_layout: if execution == ExpertExecution::FlattenedRoutedRows {
                    DenseMatmulIdInputLayout::Slotted
                } else {
                    DenseMatmulIdInputLayout::SharedPerToken
                },
                id_multiplicity: DenseMatmulIdMultiplicity::DistinctPerToken,
                route: DenseMatmulIdRoute::Direct,
            };
            let capability = mlx_native::dense_matmul_id_capability(dtype, &params)
                .with_context(|| format!("Qwen scalar expert '{name}' source M={source_tokens}"))?;
            ensure!(
                info.byte_len >= capability.required_weight_bytes,
                "Qwen scalar expert tensor '{name}' has {} bytes, route requires {}",
                info.byte_len,
                capability.required_weight_bytes
            );
        }
        return Ok(());
    }
    for source_tokens in required_expert_source_widths(execution, top_k, routing)? {
        let request = expert_capability_request(
            info.ggml_type,
            n,
            k,
            top_k,
            n_experts,
            expert_stride_bytes,
            source_tokens,
            execution,
            routing,
        )?;
        let GgmlInvocation::ExpertPooled { shape, .. } = request.invocation else {
            unreachable!("Qwen expert preflight only constructs pooled requests")
        };
        let capability = mlx_native::ggml_capability(request);
        ensure!(
            capability.executable,
            "Qwen GGUF preflight: expert tensor '{name}' type {:?} is not executable at source M={source_tokens} (runtime M={}, top_k={}, {:?}): {}",
            info.ggml_type,
            shape.n_tokens,
            shape.top_k,
            capability.request.workload,
            capability.diagnostic
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn ensure_expert_triplet_capability(
    gate_name: &str,
    gate: &TensorInfo,
    up_name: &str,
    up: &TensorInfo,
    down_name: &str,
    down: &TensorInfo,
    expert: usize,
    hidden: usize,
    top_k: u32,
    n_experts: u32,
    routing: GgmlRoutingPolicy,
) -> Result<()> {
    // Equal shapes and a shared activation row do not make sibling expert
    // projections share an artifact codec or execution capability.
    ensure_expert_capability(
        gate_name,
        gate,
        expert,
        hidden,
        top_k,
        n_experts,
        ExpertExecution::SharedPerSourceToken,
        routing,
    )?;
    ensure_expert_capability(
        up_name,
        up,
        expert,
        hidden,
        top_k,
        n_experts,
        ExpertExecution::SharedPerSourceToken,
        routing,
    )?;
    ensure_expert_capability(
        down_name,
        down,
        hidden,
        expert,
        top_k,
        n_experts,
        ExpertExecution::FlattenedRoutedRows,
        routing,
    )
}

/// Exact input geometry used by the production pooled expert dispatcher.
/// Gate/up read one shared row per source token and expand it by top-k. Down
/// consumes the already-expanded `h_all` rows, so its pooled call is a normal
/// shared-row invocation with `M = source_tokens * top_k` and runtime top-k 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpertExecution {
    SharedPerSourceToken,
    FlattenedRoutedRows,
}

fn workload_for_runtime_rows(rows: u32) -> GgmlWorkloadClass {
    match rows {
        1 => GgmlWorkloadClass::DecodeSingle,
        2..=8 => GgmlWorkloadClass::ContinuousWidth,
        _ => GgmlWorkloadClass::Prompt,
    }
}

/// Preserve every ordinary serving width, then add the exact expert route
/// boundary under the active policy and both scheduler maxima. Down's route
/// boundary is expressed in source tokens but derived from its expanded row
/// count, so default top-k 8 proves source M=4 (runtime M=32) and M=5 (40).
fn required_expert_source_widths(
    execution: ExpertExecution,
    configured_top_k: u32,
    routing: GgmlRoutingPolicy,
) -> Result<Vec<u32>> {
    ensure!(configured_top_k > 0, "Qwen expert top-k must be non-zero");
    ensure!(
        routing.expert_mm_threshold > 0,
        "Qwen expert MM routing threshold must be non-zero"
    );
    let mut widths = REQUIRED_MATRIX_WIDTHS
        .iter()
        .map(|(width, _)| *width)
        .chain(REQUIRED_EXPERT_SCHEDULER_WIDTHS)
        .collect::<Vec<_>>();
    let scheduler_max = *REQUIRED_EXPERT_SCHEDULER_WIDTHS
        .last()
        .expect("expert scheduler widths are non-empty");
    let mut add_if_served = |width: u32| {
        if (1..=scheduler_max).contains(&width) {
            widths.push(width);
        }
    };
    match execution {
        ExpertExecution::SharedPerSourceToken => {
            if let Some(before) = routing.expert_mm_threshold.checked_sub(1) {
                add_if_served(before);
            }
            add_if_served(routing.expert_mm_threshold);
            if let Some(first_mm) = routing.expert_mm_threshold.checked_add(1) {
                add_if_served(first_mm);
            }
        }
        ExpertExecution::FlattenedRoutedRows => {
            let last_mv_source = routing.expert_mm_threshold / configured_top_k;
            add_if_served(last_mv_source);
            if let Some(first_mm_source) = last_mv_source.checked_add(1) {
                if first_mm_source.checked_mul(configured_top_k).is_some() {
                    add_if_served(first_mm_source);
                }
            }
        }
    }
    widths.sort_unstable();
    widths.dedup();
    Ok(widths)
}

#[allow(clippy::too_many_arguments)]
fn expert_capability_request(
    ggml_type: mlx_native::GgmlType,
    n: u32,
    k: u32,
    configured_top_k: u32,
    n_experts: u32,
    expert_stride_bytes: u64,
    source_tokens: u32,
    execution: ExpertExecution,
    routing: GgmlRoutingPolicy,
) -> Result<GgmlCapabilityRequest> {
    let source_rows =
        usize::try_from(source_tokens).context("Qwen expert source row count exceeds usize")?;
    let geometry = MoeExpertGeometry::checked(source_rows, configured_top_k)
        .map_err(|error| anyhow!("Qwen expert production geometry: {error}"))?;
    let call = match execution {
        ExpertExecution::SharedPerSourceToken => geometry.gate_up,
        ExpertExecution::FlattenedRoutedRows => geometry.down,
    };
    Ok(GgmlCapabilityRequest {
        schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
        invocation: GgmlInvocation::ExpertPooled {
            shape: GgmlExpertShape {
                n_tokens: call.n_tokens,
                n,
                k,
                top_k: call.top_k,
                n_experts,
                expert_stride_bytes,
                ids_are_distinct_per_token: true,
                ids_within_expert_range: true,
            },
            input_layout: GgmlExpertInputLayout::SharedPerToken,
        },
        ggml_type,
        workload: workload_for_runtime_rows(call.n_tokens),
        routing,
    })
}

fn require_moe_ffn(
    gguf: &GgufFile,
    cfg: &Qwen35Config,
    layer_index: u32,
    receipt: &mut Qwen35GgufPreflightReceipt,
    routing: GgmlRoutingPolicy,
) -> Result<()> {
    let moe = cfg
        .moe
        .as_ref()
        .context("Qwen MoE GGUF preflight requires MoE configuration")?;
    let p = format!("blk.{layer_index}");
    let hidden = cfg.hidden_size as usize;
    let experts = moe.num_experts as usize;
    let expert = moe.moe_intermediate_size as usize;
    let shared = moe.shared_expert_intermediate_size as usize;

    require_projection(
        gguf,
        &format!("{p}.ffn_gate_inp.weight"),
        &[experts, hidden],
        receipt,
        routing,
    )?;
    require_row_projection(
        gguf,
        &format!("{p}.ffn_gate_inp_shexp.weight"),
        hidden,
        receipt,
        routing,
    )?;
    require_projection(
        gguf,
        &format!("{p}.ffn_gate_shexp.weight"),
        &[shared, hidden],
        receipt,
        routing,
    )?;
    require_projection(
        gguf,
        &format!("{p}.ffn_up_shexp.weight"),
        &[shared, hidden],
        receipt,
        routing,
    )?;
    require_projection(
        gguf,
        &format!("{p}.ffn_down_shexp.weight"),
        &[hidden, shared],
        receipt,
        routing,
    )?;

    let gate_name = format!("{p}.ffn_gate_exps.weight");
    let up_name = format!("{p}.ffn_up_exps.weight");
    let down_name = format!("{p}.ffn_down_exps.weight");
    let gate = require_tensor(
        gguf,
        &gate_name,
        &[experts, expert, hidden],
        TensorRole::ExpertStack,
        receipt,
    )?;
    let up = require_tensor(
        gguf,
        &up_name,
        &[experts, expert, hidden],
        TensorRole::ExpertStack,
        receipt,
    )?;
    let down = require_tensor(
        gguf,
        &down_name,
        &[experts, hidden, expert],
        TensorRole::ExpertStack,
        receipt,
    )?;
    ensure_expert_triplet_capability(
        &gate_name,
        gate,
        &up_name,
        up,
        &down_name,
        down,
        expert,
        hidden,
        moe.num_experts_per_tok,
        moe.num_experts,
        routing,
    )
}

fn require_full_attention(
    gguf: &GgufFile,
    cfg: &Qwen35Config,
    layer_index: u32,
    allow_split_q_gate: bool,
    receipt: &mut Qwen35GgufPreflightReceipt,
    routing: GgmlRoutingPolicy,
) -> Result<()> {
    let p = format!("blk.{layer_index}");
    let hidden = cfg.hidden_size as usize;
    let q_total = (cfg.num_attention_heads * cfg.head_dim) as usize;
    let kv_total = (cfg.num_key_value_heads * cfg.head_dim) as usize;
    let head_dim = cfg.head_dim as usize;

    require_f32(gguf, &format!("{p}.attn_norm.weight"), &[hidden], receipt)?;
    require_f32(
        gguf,
        &format!("{p}.post_attention_norm.weight"),
        &[hidden],
        receipt,
    )?;
    require_f32(
        gguf,
        &format!("{p}.attn_q_norm.weight"),
        &[head_dim],
        receipt,
    )?;
    require_f32(
        gguf,
        &format!("{p}.attn_k_norm.weight"),
        &[head_dim],
        receipt,
    )?;

    let q_name = format!("{p}.attn_q.weight");
    let q_shape = gguf
        .tensor_info(&q_name)
        .ok_or_else(|| anyhow!("Qwen GGUF preflight: required tensor '{q_name}' is missing"))?
        .shape
        .clone();
    if allow_split_q_gate && q_shape == [q_total, hidden] {
        require_projection(gguf, &q_name, &[q_total, hidden], receipt, routing)?;
        let gate_name = format!("{p}.attn_gate.weight");
        if gguf.tensor_info(&gate_name).is_some() {
            require_projection(gguf, &gate_name, &[q_total, hidden], receipt, routing)?;
        }
    } else {
        require_projection(gguf, &q_name, &[2 * q_total, hidden], receipt, routing)?;
    }
    require_projection(
        gguf,
        &format!("{p}.attn_k.weight"),
        &[kv_total, hidden],
        receipt,
        routing,
    )?;
    require_projection(
        gguf,
        &format!("{p}.attn_v.weight"),
        &[kv_total, hidden],
        receipt,
        routing,
    )?;
    require_projection(
        gguf,
        &format!("{p}.attn_output.weight"),
        &[hidden, q_total],
        receipt,
        routing,
    )?;
    Ok(())
}

fn require_linear_attention(
    gguf: &GgufFile,
    cfg: &Qwen35Config,
    layer_index: u32,
    receipt: &mut Qwen35GgufPreflightReceipt,
    routing: GgmlRoutingPolicy,
) -> Result<()> {
    let p = format!("blk.{layer_index}");
    let hidden = cfg.hidden_size as usize;
    let nk = cfg.linear_num_key_heads as usize;
    let nv = cfg.linear_num_value_heads as usize;
    let dk = cfg.linear_key_head_dim as usize;
    let dv = cfg.linear_value_head_dim as usize;
    let kernel = cfg.linear_conv_kernel_dim as usize;
    let qkv = 2 * nk * dk + nv * dv;
    let value = nv * dv;

    require_f32(gguf, &format!("{p}.attn_norm.weight"), &[hidden], receipt)?;
    require_f32(
        gguf,
        &format!("{p}.post_attention_norm.weight"),
        &[hidden],
        receipt,
    )?;
    require_f32_matrix(
        gguf,
        &format!("{p}.ssm_conv1d.weight"),
        &[qkv, kernel],
        receipt,
    )?;
    require_f32(gguf, &format!("{p}.ssm_dt.bias"), &[nv], receipt)?;
    require_f32(gguf, &format!("{p}.ssm_a"), &[nv], receipt)?;
    require_f32(gguf, &format!("{p}.ssm_norm.weight"), &[dv], receipt)?;

    require_projection(
        gguf,
        &format!("{p}.attn_qkv.weight"),
        &[qkv, hidden],
        receipt,
        routing,
    )?;
    require_projection(
        gguf,
        &format!("{p}.attn_gate.weight"),
        &[value, hidden],
        receipt,
        routing,
    )?;
    require_projection(
        gguf,
        &format!("{p}.ssm_alpha.weight"),
        &[nv, hidden],
        receipt,
        routing,
    )?;
    require_projection(
        gguf,
        &format!("{p}.ssm_beta.weight"),
        &[nv, hidden],
        receipt,
        routing,
    )?;
    require_projection(
        gguf,
        &format!("{p}.ssm_out.weight"),
        &[hidden, value],
        receipt,
        routing,
    )?;
    Ok(())
}

fn require_mtp(
    gguf: &GgufFile,
    cfg: &Qwen35Config,
    receipt: &mut Qwen35GgufPreflightReceipt,
    routing: GgmlRoutingPolicy,
) -> Result<()> {
    ensure!(
        cfg.mtp_num_hidden_layers == 1,
        "Qwen GGUF preflight supports exactly one MTP layer, got {}",
        cfg.mtp_num_hidden_layers
    );
    let layer = cfg.num_hidden_layers;
    let nextn = format!("blk.{layer}.nextn");
    let hidden = cfg.hidden_size as usize;
    let embed_name = format!("{nextn}.embed_tokens.weight");
    let head_name = format!("{nextn}.shared_head_head.weight");
    admit_mtp_tensor_presence(
        cfg.mtp_use_dedicated_embeddings,
        gguf.tensor_info(&embed_name).is_some(),
        gguf.tensor_info(&head_name).is_some(),
    )?;

    for suffix in ["enorm.weight", "hnorm.weight", "shared_head_norm.weight"] {
        require_f32(gguf, &format!("{nextn}.{suffix}"), &[hidden], receipt)?;
    }
    require_projection(
        gguf,
        &format!("{nextn}.eh_proj.weight"),
        &[hidden, 2 * hidden],
        receipt,
        routing,
    )?;

    if cfg.mtp_use_dedicated_embeddings {
        require_embedding(gguf, &embed_name, hidden, receipt, routing)?;
    } else {
        ensure!(
            gguf.tensor_info(&embed_name).is_none(),
            "Qwen GGUF preflight: shared-MTP metadata conflicts with present tensor '{embed_name}'"
        );
    }

    if cfg.mtp_use_dedicated_embeddings {
        let info = gguf.tensor_info(&head_name).ok_or_else(|| {
            anyhow!("Qwen GGUF preflight: dedicated MTP requires tensor '{head_name}'")
        })?;
        ensure!(
            info.shape.len() == 2 && info.shape[1] == hidden,
            "Qwen GGUF preflight: tensor '{head_name}' shape {:?} is not [vocab,{hidden}]",
            info.shape
        );
        require_projection(gguf, &head_name, &[info.shape[0], hidden], receipt, routing)?;
    } else {
        ensure!(
            gguf.tensor_info(&head_name).is_none(),
            "Qwen GGUF preflight: shared-MTP metadata conflicts with present tensor '{head_name}'"
        );
    }

    require_full_attention(gguf, cfg, layer, true, receipt, routing)?;
    match cfg.variant {
        Qwen35Variant::Dense => require_dense_ffn(
            gguf,
            layer,
            hidden,
            cfg.intermediate_size
                .context("dense Qwen MTP preflight requires feed_forward_length")?
                as usize,
            receipt,
            routing,
        ),
        Qwen35Variant::Moe => require_moe_ffn(gguf, cfg, layer, receipt, routing),
    }
}

/// Validate every tensor consumed by the Qwen graph before creating a
/// Metal device or reading any tensor payload.
pub(super) fn preflight_qwen35_gguf(
    gguf: &GgufFile,
    cfg: &Qwen35Config,
) -> Result<Qwen35GgufPreflightReceipt> {
    preflight_qwen35_gguf_with_routing(gguf, cfg, super::ggml_routing_policy_for_gguf(gguf))
}

pub(super) fn preflight_qwen35_gguf_with_routing(
    gguf: &GgufFile,
    cfg: &Qwen35Config,
    routing: GgmlRoutingPolicy,
) -> Result<Qwen35GgufPreflightReceipt> {
    let hidden = cfg.hidden_size as usize;
    let mut receipt = Qwen35GgufPreflightReceipt::default();

    let embedding_rows =
        require_embedding(gguf, "token_embd.weight", hidden, &mut receipt, routing)?;
    require_f32(gguf, "output_norm.weight", &[hidden], &mut receipt)?;
    if let Some(output) = gguf.tensor_info("output.weight") {
        ensure!(
            output.shape.len() == 2 && output.shape[1] == hidden,
            "Qwen GGUF preflight: output.weight shape {:?} is not [vocab,{hidden}]",
            output.shape
        );
        ensure!(
            embedding_rows >= output.shape[0],
            "Qwen GGUF preflight: token embedding rows {embedding_rows} < output rows {}",
            output.shape[0]
        );
        require_projection(
            gguf,
            "output.weight",
            &[output.shape[0], hidden],
            &mut receipt,
            routing,
        )?;
    } else {
        let token = gguf
            .tensor_info("token_embd.weight")
            .context("Qwen GGUF preflight: tied output head has no token embedding")?;
        admit_storage_for_role(
            "token_embd.weight (tied output head)",
            TensorRole::DenseProjection,
            TensorStorage::Parsed(token.ggml_type),
        )?;
        ensure_dense_capability_with_routing(
            "token_embd.weight (tied output head)",
            token,
            routing,
        )?;
    }

    for (layer_index, kind) in cfg.layer_types.iter().copied().enumerate() {
        let layer_index = u32::try_from(layer_index).context("Qwen layer index exceeds u32")?;
        match kind {
            Qwen35LayerKind::FullAttention => {
                require_full_attention(gguf, cfg, layer_index, false, &mut receipt, routing)?
            }
            Qwen35LayerKind::LinearAttention => {
                require_linear_attention(gguf, cfg, layer_index, &mut receipt, routing)?
            }
        }
        match cfg.variant {
            Qwen35Variant::Dense => require_dense_ffn(
                gguf,
                layer_index,
                hidden,
                cfg.intermediate_size
                    .context("dense Qwen GGUF preflight requires feed_forward_length")?
                    as usize,
                &mut receipt,
                routing,
            )?,
            Qwen35Variant::Moe => require_moe_ffn(gguf, cfg, layer_index, &mut receipt, routing)?,
        }
    }

    match cfg.mtp_num_hidden_layers {
        0 => {}
        1 => require_mtp(gguf, cfg, &mut receipt, routing)?,
        count => bail!("Qwen GGUF preflight supports at most one MTP layer, got {count}"),
    }

    tracing::debug!(
        required_tensors = receipt.required_tensor_count,
        matrix_tensors = receipt.matrix_tensor_count,
        matrix_bytes = receipt.matrix_bytes,
        roles = ?receipt.role_counts,
        storage = ?receipt.storage_counts,
        variant = ?cfg.variant,
        "Qwen GGUF execution preflight passed"
    );
    Ok(receipt)
}

#[cfg(test)]
pub(super) fn preflight_dense_qwen35_gguf(gguf: &GgufFile, cfg: &Qwen35Config) -> Result<()> {
    ensure!(
        cfg.variant == Qwen35Variant::Dense,
        "dense Qwen GGUF preflight received {:?}",
        cfg.variant
    );
    preflight_qwen35_gguf(gguf, cfg).map(|_| ())
}

#[cfg(test)]
#[path = "gguf_preflight_tests.rs"]
mod tests;
