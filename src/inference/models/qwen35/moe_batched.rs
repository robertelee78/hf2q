//! Batched MoE FFN decode route (`HF2Q_QWEN_MOE_BATCHED=1`).
//!
//! One MoE FFN layer over `rows = 1..=32` decode or DFlash2 target rows in one
//! encoder segment, with the router and shared expert read in their GGUF storage types:
//!
//! | phase | dispatch | kernel |
//! |---|---|---|
//! | A | router logits `[rows, n_experts]` | `dense_matvec_f32_rows` (F32) |
//! | A | shared-expert gate logit `[rows, 1]` | `dense_matvec_f32_rows` (F32) |
//! | A | shared gate, shared up `[rows, shared_i]` | `quantized_matmul_ggml` m = rows (`kernel_mul_mv_<q>_f32` at 1 row, `_mN_r1_<rows>` at 2..5) |
//! | B | softmax + top-k + renorm over all rows | `moe_softmax_topk_simd_f32` |
//! | B | shared SwiGLU | `silu_mul_f32` |
//! | C | expert gate, expert up over `rows * top_k` pairs | `kernel_mul_mv_id_q4_K_f32_nr2` (Q4_K) or `kernel_mul_mv_id_<q>_f32` (forced matvec route) |
//! | C | shared down `[rows, hidden]` | `quantized_matmul_ggml` m = rows |
//! | D | expert SwiGLU | `silu_mul_f32` |
//! | E | expert down over `rows * top_k` pairs | `kernel_mul_mv_id_q4_K_f32_nr2` (Q4_K) or `kernel_mul_mv_id_<q>_f32[_nr2]` (forced matvec route) |
//! | F | weighted reduce + sigmoid(shared gate) * shared + residual | `moe_weighted_reduce_f32` |
//!
//! Decode dispatches of one to eight rows preserve the established bitwise
//! single-row invariant. The DFlash2 verifier's width-32 pass uses dense MM
//! for the F32 router and shared-gate logits and is qualified against the F64
//! whole-layer reference instead. The serial decode builder
//! ([`super::gpu_ffn::build_moe_ffn_layer_gpu_q_into`] at one row) and the
//! joint step both run this route whenever the layer's weights were loaded
//! in native storage, so the switch is decided once at load.

use std::sync::LazyLock;

use anyhow::{anyhow, bail, ensure, Context, Result};
use mlx_native::ops::dflash2_attention_pack::{
    dispatch_dflash2_mask_moe_routes, DFlash2MoeRouteMaskParams,
};
use mlx_native::ops::dense_matvec_f32_rows::{
    dispatch_dense_matvec_f32_rows, DENSE_MATVEC_F32_ROWS_MAX_ROWS,
};
use mlx_native::ops::dense_mm_f32_f32::{dense_matmul_f32_f32_tensor, DenseMmF32F32Params};
use mlx_native::ops::moe_softmax_topk::dispatch_moe_softmax_topk_simd;
use mlx_native::ops::moe_weighted_reduce::dispatch_moe_weighted_reduce;
use mlx_native::ops::quantized_matmul_ggml::{GgmlQuantizedMatmulParams, GgmlType};
use mlx_native::ops::quantized_matmul_id_ggml::{
    quantized_matmul_id_ggml_mv, quantized_matmul_id_ggml_mv_masked,
    quantized_matmul_id_ggml_mv_q4_k_nr2, quantized_matmul_id_ggml_mv_q4_k_nr2_masked,
    GgmlQuantizedMatmulIdParams,
};
use mlx_native::ops::silu_mul::dispatch_silu_mul;
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::decode_pool::pooled_alloc_buffer;
use super::execution_dispatch::quantized_matmul_ggml;
use super::ffn::MoeFfnShape;
use super::gpu_ffn::MoeFfnWeightsGpuQ;

/// Switch for the batched native-storage MoE decode route. Default off.
pub(crate) const QWEN_MOE_BATCHED_ENV: &str = "HF2Q_QWEN_MOE_BATCHED";

/// Widest row count one batched MoE layer accepts: the DFlash2 target width.
pub(crate) const MOE_BATCHED_MAX_ROWS: usize = 32;

/// Parse [`QWEN_MOE_BATCHED_ENV`]; unset or blank means off.
pub(crate) fn parse_qwen_moe_batched(raw: Option<&str>) -> Result<bool, String> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(false);
    };
    match raw.to_ascii_lowercase().as_str() {
        "1" | "on" | "true" => Ok(true),
        "0" | "off" | "false" => Ok(false),
        _ => Err(format!(
            "{QWEN_MOE_BATCHED_ENV} must be 1/on/true or 0/off/false, got {raw:?}"
        )),
    }
}

/// [`QWEN_MOE_BATCHED_ENV`] read once per process (at the first MoE layer
/// load). Invalid values warn and keep the default (off).
pub(crate) fn qwen_moe_batched_requested() -> bool {
    static REQUESTED: LazyLock<bool> = LazyLock::new(|| {
        parse_qwen_moe_batched(std::env::var(QWEN_MOE_BATCHED_ENV).ok().as_deref())
            .unwrap_or_else(|error| {
                tracing::warn!(error, fallback = false, "invalid Qwen MoE batched switch");
                false
            })
    });
    *REQUESTED
}

/// Shared-expert storage types the batched route admits. Their GGML matmul
/// routes support the full `m = 1..=32` interval.
pub(crate) fn native_shared_types_supported(gate_up: GgmlType, down: GgmlType) -> bool {
    let exact = |t: GgmlType| matches!(t, GgmlType::Q4_K | GgmlType::Q6_K);
    exact(gate_up) && exact(down)
}

/// How a MoE layer stores its router and shared-expert weights on the GPU.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MoeDenseStorage {
    /// Router, shared gate logit and shared gate/up/down uploaded as BF16
    /// (dequantized at load). Decode runs the serial builder once per row.
    Bf16,
    /// GGUF storage: router and shared gate logit F32, shared gate/up/down
    /// raw GGML blocks. Decode runs [`build_moe_ffn_layer_gpu_batched_into`].
    Native {
        shared_gate_up: GgmlType,
        shared_down: GgmlType,
    },
}

impl MoeDenseStorage {
    pub fn is_native(self) -> bool {
        matches!(self, Self::Native { .. })
    }
}

/// One of the router / shared-expert projections of a MoE layer.
#[derive(Clone, Copy, Debug)]
pub(crate) enum MoeDenseRole {
    Router,
    SharedGateLogit,
    SharedGate,
    SharedUp,
    SharedDown,
}

impl MoeDenseRole {
    pub(crate) fn weight(self, weights: &MoeFfnWeightsGpuQ) -> &MlxBuffer {
        match self {
            Self::Router => &weights.router,
            Self::SharedGateLogit => &weights.shared_gate_inp,
            Self::SharedGate => &weights.shared_gate,
            Self::SharedUp => &weights.shared_up,
            Self::SharedDown => &weights.shared_down,
        }
    }
}

/// Encode `output[rows, out] = input[rows, in] @ W^T` for a native-storage
/// router / shared-expert weight. F32 weights take the row-exact
/// `dense_matvec_f32_rows` at `rows <= 8` (the tensor matmul above, for
/// prefill chunks); GGML weights take `quantized_matmul_ggml` at `m = rows`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_native_dense_proj(
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weights: &MoeFfnWeightsGpuQ,
    role: MoeDenseRole,
    output: &MlxBuffer,
    rows: u32,
    in_features: u32,
    out_features: u32,
) -> Result<()> {
    let MoeDenseStorage::Native {
        shared_gate_up,
        shared_down,
    } = weights.dense_storage
    else {
        bail!("{role:?}: native projection requested on BF16 MoE storage");
    };
    let weight = role.weight(weights);
    match role {
        MoeDenseRole::Router | MoeDenseRole::SharedGateLogit => {
            ensure!(
                weight.dtype() == DType::F32,
                "{role:?}: native router weights must be F32, got {:?}",
                weight.dtype()
            );
            if rows <= DENSE_MATVEC_F32_ROWS_MAX_ROWS {
                dispatch_dense_matvec_f32_rows(
                    enc,
                    registry,
                    device,
                    input,
                    weight,
                    output,
                    rows,
                    out_features,
                    in_features,
                )
                .with_context(|| format!("{role:?} dense_matvec_f32_rows"))
            } else {
                dense_matmul_f32_f32_tensor(
                    enc,
                    registry,
                    device,
                    weight,
                    input,
                    output,
                    &DenseMmF32F32Params {
                        m: rows,
                        n: out_features,
                        k: in_features,
                        src0_batch: 1,
                        src1_batch: 1,
                    },
                )
                .with_context(|| format!("{role:?} dense_matmul_f32_f32_tensor"))
            }
        }
        MoeDenseRole::SharedGate | MoeDenseRole::SharedUp | MoeDenseRole::SharedDown => {
            let ggml_type = if matches!(role, MoeDenseRole::SharedDown) {
                shared_down
            } else {
                shared_gate_up
            };
            quantized_matmul_ggml(
                enc,
                registry,
                device,
                input,
                weight,
                output,
                &GgmlQuantizedMatmulParams {
                    m: rows,
                    n: out_features,
                    k: in_features,
                    ggml_type,
                },
            )
            .map_err(|e| anyhow!("{role:?} quantized_matmul_ggml {ggml_type:?}: {e}"))
        }
    }
}

/// Expert matvec over `params.n_tokens * params.top_k` pairs on the forced
/// `mv_id` route (never the grouped `mm_id` route, whose reduction differs),
/// behind the same imatrix intercept as the serial builder. Q4_K experts take
/// the peer-layout kernel (`kernel_mul_mv_id_q4_K_f32_nr2`) at every width,
/// including one row, so a pair's result never depends on the width.
#[allow(clippy::too_many_arguments)]
fn dispatch_expert_mv(
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    ids: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulIdParams,
    tag: &'static str,
    layer: usize,
    masked: bool,
) -> Result<()> {
    crate::quantize::imatrix::intercept_qmatmul_id_with_hint(
        crate::quantize::imatrix::ImatrixHint::Layered { tag, layer },
        params.n_tokens as usize,
        params.top_k as usize,
        params.k as usize,
        || {
            if let Err(e) = enc.commit_wait_and_rotate() {
                eprintln!("[hf2q imatrix moe intercept ({tag})] commit_wait_and_rotate failed: {e}");
                return None;
            }
            input.as_slice::<f32>().ok().map(|sl| sl.to_vec())
        },
        || ids.as_slice::<u32>().ok().map(|sl| sl.to_vec()),
    )
    .map_err(|e| anyhow!("imatrix moe intercept ({tag}): {e}"))?;
    let dispatched = match (params.ggml_type, masked) {
        (GgmlType::Q4_K, false) => quantized_matmul_id_ggml_mv_q4_k_nr2(
            enc, registry, device, input, weight, ids, output, params,
        ),
        (GgmlType::Q4_K, true) => quantized_matmul_id_ggml_mv_q4_k_nr2_masked(
            enc, registry, device, input, weight, ids, output, params,
        ),
        (_, false) => {
            quantized_matmul_id_ggml_mv(enc, registry, device, input, weight, ids, output, params)
        }
        (_, true) => quantized_matmul_id_ggml_mv_masked(
            enc, registry, device, input, weight, ids, output, params,
        ),
    };
    dispatched.map_err(|e| anyhow!("{tag} mul_mv_id: {e}"))
}

/// Routing produced by one batched layer: `[rows * top_k]` expert ids and
/// renormalized weights, row-major. Pooled; valid until the next decode-pool
/// reset. Production callers drop it; the batched-vs-serial tests read the
/// routing back.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct MoeBatchedRouting {
    pub ids: MlxBuffer,
    pub weights: MlxBuffer,
}

#[derive(Clone, Copy)]
pub(crate) struct DFlash2MoeRouteMask<'a> {
    pub verify_rows: &'a MlxBuffer,
    pub params: DFlash2MoeRouteMaskParams,
}

fn pooled(device: &MlxDevice, elements: usize, dtype: DType, shape: Vec<usize>, label: &str) -> Result<MlxBuffer> {
    pooled_alloc_buffer(device, elements * dtype.size_of(), dtype, shape)
        .map_err(|e| anyhow!("allocate batched MoE {label}: {e}"))
}

/// Encode one MoE FFN layer over `rows` packed decode rows into `out`
/// (`[rows, hidden]`), adding `residual` (`[rows, hidden]`) when given.
/// `x` is `[rows, hidden]` F32. Scratch comes from the decode pool. Does not
/// commit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_moe_ffn_layer_gpu_batched_into(
    enc: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &MlxBuffer,
    weights: &MoeFfnWeightsGpuQ,
    shape: MoeFfnShape,
    residual: Option<&MlxBuffer>,
    out: &mut MlxBuffer,
    rows: u32,
    layer_idx: usize,
    route_mask: Option<DFlash2MoeRouteMask<'_>>,
) -> Result<MoeBatchedRouting> {
    ensure!(
        (1..=MOE_BATCHED_MAX_ROWS).contains(&(rows as usize)),
        "batched MoE layer {layer_idx} accepts 1..={MOE_BATCHED_MAX_ROWS} rows, got {rows}"
    );
    ensure!(
        weights.dense_storage.is_native(),
        "batched MoE layer {layer_idx} requires native router/shared-expert storage"
    );
    ensure!(
        weights.expert_gate_affine.is_none()
            && weights.expert_up_affine.is_none()
            && weights.expert_down_affine.is_none(),
        "batched MoE layer {layer_idx} does not support DWQ affine expert overlays"
    );
    let r = rows as usize;
    let h = shape.hidden_size as usize;
    let top_k = shape.num_experts_per_tok as usize;
    let m_moe = shape.moe_intermediate_size as usize;
    let m_sh = shape.shared_intermediate_size as usize;
    let ne = shape.num_experts;
    ensure!(
        x.element_count() >= r * h,
        "batched MoE layer {layer_idx}: input has {} values, need {rows} x {h}",
        x.element_count()
    );
    let pairs = r * top_k;

    let logits = pooled(device, r * ne as usize, DType::F32, vec![r, ne as usize], "router logits")?;
    let sh_logit = pooled(device, r, DType::F32, vec![r, 1], "shared gate logit")?;
    let a_s = pooled(device, r * m_sh, DType::F32, vec![r, m_sh], "shared gate")?;
    let b_s = pooled(device, r * m_sh, DType::F32, vec![r, m_sh], "shared up")?;
    let h_s = pooled(device, r * m_sh, DType::F32, vec![r, m_sh], "shared swiglu")?;
    let y_s = pooled(device, r * h, DType::F32, vec![r, h], "shared down")?;
    let ids = pooled(device, pairs, DType::U32, vec![pairs], "expert ids")?;
    let routing_weights = pooled(device, pairs, DType::F32, vec![pairs], "routing weights")?;
    let gate_all = pooled(device, pairs * m_moe, DType::F32, vec![pairs, m_moe], "expert gate")?;
    let up_all = pooled(device, pairs * m_moe, DType::F32, vec![pairs, m_moe], "expert up")?;
    let h_all = pooled(device, pairs * m_moe, DType::F32, vec![pairs, m_moe], "expert swiglu")?;
    let y_all = pooled(device, pairs * h, DType::F32, vec![pairs, h], "expert down")?;
    let mut silu_sh_params = pooled(device, 1, DType::U32, vec![1], "shared swiglu params")?;
    let mut silu_params = pooled(device, 1, DType::U32, vec![1], "expert swiglu params")?;
    let n_h_s = (r * m_sh) as u32;
    let n_h_all = (pairs * m_moe) as u32;
    silu_sh_params
        .as_mut_slice::<u32>()
        .map_err(|e| anyhow!("batched MoE shared swiglu params: {e}"))?[0] = n_h_s;
    silu_params
        .as_mut_slice::<u32>()
        .map_err(|e| anyhow!("batched MoE expert swiglu params: {e}"))?[0] = n_h_all;
    let dummy_residual;
    let residual_ref = match residual {
        Some(buffer) => buffer,
        None => {
            dummy_residual = pooled(device, 1, DType::F32, vec![1], "dummy residual")?;
            &dummy_residual
        }
    };

    let h32 = shape.hidden_size;
    let m_sh32 = shape.shared_intermediate_size;
    // ---- A: router, shared gate logit, shared gate/up (all read x) ----
    for (role, output, out_features) in [
        (MoeDenseRole::Router, &logits, ne),
        (MoeDenseRole::SharedGateLogit, &sh_logit, 1),
        (MoeDenseRole::SharedGate, &a_s, m_sh32),
        (MoeDenseRole::SharedUp, &b_s, m_sh32),
    ] {
        encode_native_dense_proj(
            enc, registry, device, x, weights, role, output, rows, h32, out_features,
        )
        .with_context(|| format!("batched MoE layer {layer_idx}"))?;
    }
    enc.memory_barrier();

    // ---- B: routing over every row, shared SwiGLU ----
    dispatch_moe_softmax_topk_simd(
        enc,
        registry,
        device,
        &logits,
        &ids,
        &routing_weights,
        rows,
        ne,
        shape.num_experts_per_tok,
    )
    .map_err(|e| anyhow!("batched MoE layer {layer_idx} top-k: {e}"))?;
    if let Some(mask) = route_mask {
        dispatch_dflash2_mask_moe_routes(
            enc,
            registry,
            device,
            mask.verify_rows,
            &ids,
            &routing_weights,
            mask.params,
        )
        .map_err(|e| anyhow!("batched MoE layer {layer_idx} DFlash2 route mask: {e}"))?;
    }
    dispatch_silu_mul(
        enc,
        registry,
        device.metal_device(),
        &a_s,
        &b_s,
        &h_s,
        &silu_sh_params,
        n_h_s,
    )
    .map_err(|e| anyhow!("batched MoE layer {layer_idx} shared swiglu: {e}"))?;
    enc.memory_barrier();

    // ---- C: expert gate/up over rows * top_k pairs (row t feeds pairs
    // t * top_k ..), shared down ----
    let gate_params = GgmlQuantizedMatmulIdParams {
        n_tokens: rows,
        top_k: shape.num_experts_per_tok,
        n: shape.moe_intermediate_size,
        k: h32,
        n_experts: ne,
        expert_stride: weights.expert_gate_stride,
        ggml_type: weights.ggml_type_gate_up,
    };
    let up_params = GgmlQuantizedMatmulIdParams {
        expert_stride: weights.expert_up_stride,
        ..gate_params
    };
    dispatch_expert_mv(
        enc, registry, device, x, &weights.expert_gate_q, &ids, &gate_all, &gate_params,
        "ffn_gate_exps", layer_idx, route_mask.is_some(),
    )?;
    dispatch_expert_mv(
        enc, registry, device, x, &weights.expert_up_q, &ids, &up_all, &up_params,
        "ffn_up_exps", layer_idx, route_mask.is_some(),
    )?;
    encode_native_dense_proj(
        enc,
        registry,
        device,
        &h_s,
        weights,
        MoeDenseRole::SharedDown,
        &y_s,
        rows,
        m_sh32,
        h32,
    )
    .with_context(|| format!("batched MoE layer {layer_idx}"))?;
    enc.memory_barrier();

    // ---- D: expert SwiGLU ----
    dispatch_silu_mul(
        enc,
        registry,
        device.metal_device(),
        &gate_all,
        &up_all,
        &h_all,
        &silu_params,
        n_h_all,
    )
    .map_err(|e| anyhow!("batched MoE layer {layer_idx} expert swiglu: {e}"))?;
    enc.memory_barrier();

    // ---- E: expert down, one pair per input row ----
    let down_params = GgmlQuantizedMatmulIdParams {
        n_tokens: pairs as u32,
        top_k: 1,
        n: h32,
        k: shape.moe_intermediate_size,
        n_experts: ne,
        expert_stride: weights.expert_down_stride,
        ggml_type: weights.ggml_type_down,
    };
    dispatch_expert_mv(
        enc, registry, device, &h_all, &weights.expert_down_q, &ids, &y_all, &down_params,
        "ffn_down_exps", layer_idx, route_mask.is_some(),
    )?;
    enc.memory_barrier();

    // ---- F: weighted reduce + sigmoid(shared gate) * shared + residual ----
    dispatch_moe_weighted_reduce(
        enc,
        registry,
        device,
        &routing_weights,
        &y_all,
        &sh_logit,
        &y_s,
        residual_ref,
        out,
        rows,
        shape.num_experts_per_tok,
        h32,
        residual.is_some(),
    )
    .map_err(|e| anyhow!("batched MoE layer {layer_idx} weighted reduce: {e}"))?;

    #[cfg(test)]
    {
        crate::adr063_invariance::qwen::retain_current("moe_router_logits", &logits, vec![r, ne as usize])
            .map_err(anyhow::Error::msg)?;
        crate::adr063_invariance::qwen::retain_current("moe_router_ids", &ids, vec![r, top_k])
            .map_err(anyhow::Error::msg)?;
        crate::adr063_invariance::qwen::retain_current(
            "moe_router_weights",
            &routing_weights,
            vec![r, top_k],
        )
        .map_err(anyhow::Error::msg)?;
    }

    Ok(MoeBatchedRouting {
        ids,
        weights: routing_weights,
    })
}
