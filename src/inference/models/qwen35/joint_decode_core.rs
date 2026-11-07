//! Per-layer attention cores for Qwen joint B-row decode.
//!
//! Each joint row is one decode token of an independent slot at its own
//! position. The functions here split the serial single-token helpers
//! ([`super::gpu_delta_net::build_delta_net_layer_decode_into`] and
//! [`super::gpu_full_attn::apply_gated_attn_layer_decode_into`]) into
//!
//! 1. projections over every packed row (or one serial-identical `m = 1`
//!    dispatch per row, see [`JointProductRoute`]),
//! 2. a per-slot core that reads and writes only that slot's cache region
//!    (DeltaNet conv + gated delta net against the slot's ping-pong parity;
//!    TQ/F32 KV write + SDPA against the slot's cursor), and
//! 3. a packed epilogue (gated norm / sigmoid gate + output projection).
//!
//! Row-independent kernels (RMS norms, IMROPE with per-row positions,
//! `compute_g_beta`, `ssm_norm_gate`, sigmoid gate) run once over the packed
//! rows: their per-row arithmetic does not depend on the row count. The
//! serial bodies are unchanged; this module repeats their dispatch order.
//!
//! A slot's rows form one [`JointWindow`] of consecutive positions: one row
//! in a decode step, the pending token plus its MTP drafts in a verify step
//! (`joint_verify`). DeltaNet runs one conv and one gated delta net dispatch
//! per window (`n_tokens` = its rows; window ≡ sequential one-token steps);
//! full attention runs each row's own write + SDPA in order, as decode does.
//! [`DeltaNetReplay`] retains a multi-row window's inputs for the verify
//! commit's replay ([`encode_delta_net_replay`]).
//!
//! Packed rows beyond the real rows (`mul_mv_ext` width-4 padding) carry copies
//! of row 0's core outputs so every packed kernel reads finite data; their
//! results are discarded and they never touch a cache.

use anyhow::{anyhow, bail, ensure, Context, Result};
use mlx_native::ops::compute_g_beta::dispatch_compute_g_beta;
use mlx_native::ops::copy::{dispatch_copy_f32, dispatch_strided_copy_f32, StridedCopyParams};
use mlx_native::ops::elementwise::scalar_mul_f32;
use mlx_native::ops::gated_delta_net::GatedDeltaNetParams;
use mlx_native::ops::gated_delta_net_decode::dispatch_gated_delta_net_decode;
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::ops::ssm_conv::{dispatch_ssm_conv, SsmConvParams};
use mlx_native::ops::ssm_norm_gate::dispatch_ssm_norm_gate;
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::decode_pool::pooled_alloc_buffer;
use super::delta_net::DeltaNetLayerShape;
use super::full_attn::FullAttnShape;
use super::gpu_delta_net::{
    apply_l2_norm_per_head, apply_l2_norm_per_head_into, apply_pre_norm, apply_proj_with_ggml_type,
    narrow_la_ping_pong_to_slot, DeltaNetWeightsGpu, FORWARD_DISPATCH_N_SEQS,
};
use super::gpu_full_attn::{
    apply_imrope, apply_linear_projection_f32_pooled_with_ggml_type, apply_pre_attn_rms_norm,
    apply_q_gate_projection_f32, apply_q_or_k_per_head_rms_norm,
    apply_sdpa_with_kv_cache_decode_into, apply_sigmoid_gate_multiply, FullAttnQGateWeightsGpu,
    FullAttnWeightsGpu,
};
use super::joint_decode_routes::{projection_route, JointProductRoute};
use super::kv_cache::{HybridKvCache, LayerSlot};
use super::Qwen35Config;
use crate::serve::multi_seq_kv::SlotId;

/// The packed rows of one joint step: the real rows' slots in packed order,
/// their lane windows, plus the packed width (`>= slots.len()`; larger only
/// when padded).
pub(crate) struct JointRows<'a> {
    pub slots: &'a [SlotId],
    /// Consecutive real rows of one slot, in packed order, covering every
    /// real row once. A decode step has one single-row window per slot; an
    /// MTP verify step has one window per lane (its pending token and
    /// drafts at consecutive positions).
    pub windows: &'a [JointWindow],
    pub width: usize,
}

/// One lane's rows of a joint step: packed rows `start..start + rows`, all
/// of `slot`, at consecutive positions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct JointWindow {
    pub slot: SlotId,
    pub start: usize,
    pub rows: usize,
}

impl JointWindow {
    /// One single-row window per slot: the decode step's layout.
    pub(crate) fn per_row(slots: &[SlotId]) -> Vec<Self> {
        slots
            .iter()
            .enumerate()
            .map(|(start, &slot)| Self {
                slot,
                start,
                rows: 1,
            })
            .collect()
    }
}

impl JointRows<'_> {
    pub(crate) fn real(&self) -> usize {
        self.slots.len()
    }

    fn width_u32(&self) -> u32 {
        self.width as u32
    }
}

/// Zero-copy view of rows `start..start + rows` of a row-major buffer with
/// `row_width` elements per row, shaped `[rows, row_width]`.
pub(crate) fn rows_view(
    buffer: &MlxBuffer,
    start: usize,
    rows: usize,
    row_width: usize,
) -> Result<MlxBuffer> {
    let elements = rows
        .checked_mul(row_width)
        .context("Qwen joint rows view overflow")?;
    let first = start
        .checked_mul(row_width)
        .context("Qwen joint rows view offset overflow")?;
    ensure!(
        first + elements <= buffer.element_count(),
        "Qwen joint rows {start}..{} of {row_width} elements exceed buffer {:?}",
        start + rows,
        buffer.shape()
    );
    let byte_offset = u64::try_from(first * buffer.dtype().size_of())
        .context("Qwen joint rows view byte offset overflow")?;
    buffer
        .slice_view(byte_offset, elements)
        .with_shape(vec![rows, row_width])
        .map_err(|e| anyhow!("reshape Qwen joint rows view: {e}"))
}

/// Zero-copy view of row `row` of a row-major buffer, reshaped to `shape`.
pub(crate) fn row_view(buffer: &MlxBuffer, row: usize, shape: Vec<usize>) -> Result<MlxBuffer> {
    let row_elements = shape
        .iter()
        .try_fold(1_usize, |count, &dim| count.checked_mul(dim))
        .context("Qwen joint row view shape overflow")?;
    let end = row
        .checked_add(1)
        .and_then(|rows| rows.checked_mul(row_elements))
        .context("Qwen joint row view end overflow")?;
    ensure!(
        end <= buffer.element_count(),
        "Qwen joint row {row} of {row_elements} elements exceeds buffer {:?}",
        buffer.shape()
    );
    let byte_offset = (row * row_elements)
        .checked_mul(buffer.dtype().size_of())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .context("Qwen joint row view byte offset overflow")?;
    buffer
        .slice_view(byte_offset, row_elements)
        .with_shape(shape)
        .map_err(|e| anyhow!("reshape Qwen joint row view: {e}"))
}

fn pooled_f32(device: &MlxDevice, shape: Vec<usize>, label: &str) -> Result<MlxBuffer> {
    let elements: usize = shape.iter().product();
    pooled_alloc_buffer(device, elements * 4, DType::F32, shape)
        .map_err(|e| anyhow!("allocate Qwen joint {label}: {e}"))
}

/// Copy `row_width` F32 values from `src` into packed row `row` of `dst`.
fn copy_row(
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    src: &MlxBuffer,
    dst: &MlxBuffer,
    row: usize,
    row_width: usize,
) -> Result<()> {
    dispatch_copy_f32(
        enc,
        registry,
        device.metal_device(),
        src,
        dst,
        0,
        row * row_width,
        row_width,
    )
    .map_err(|e| anyhow!("copy Qwen joint row {row}: {e}"))
}

/// Fill packed padding rows with copies of row 0 (`dst` rows `real..width`).
/// The caller orders the copies after row 0's producer with a barrier.
fn fill_padding_rows(
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    dst: &MlxBuffer,
    rows: &JointRows<'_>,
    row_width: usize,
) -> Result<()> {
    if rows.width == rows.real() {
        return Ok(());
    }
    let first = row_view(dst, 0, vec![row_width])?;
    for row in rows.real()..rows.width {
        copy_row(enc, registry, device, &first, dst, row, row_width)?;
    }
    Ok(())
}

/// Issue one projection stage over the packed rows. `project(enc, registry,
/// input, seq)` is the serial helper; it returns one or more `[seq, w]`
/// outputs. `Batched` calls it once with `seq = width`; `PerRow` calls it
/// once per packed row with `seq = 1` (exactly the serial dispatch) and
/// gathers the rows into packed `[width, w]` buffers.
pub(crate) fn project_rows<F>(
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    route: JointProductRoute,
    input: &MlxBuffer,
    width: usize,
    in_features: usize,
    mut project: F,
) -> Result<Vec<MlxBuffer>>
where
    F: FnMut(&mut CommandEncoder, &mut KernelRegistry, &MlxBuffer, u32) -> Result<Vec<MlxBuffer>>,
{
    if route == JointProductRoute::Batched {
        return project(enc, registry, input, width as u32);
    }
    let mut per_row = Vec::with_capacity(width);
    for row in 0..width {
        let input_row = row_view(input, row, vec![1, in_features])?;
        per_row.push(project(enc, registry, &input_row, 1)?);
    }
    let outputs = per_row.first().map_or(0, Vec::len);
    ensure!(
        outputs > 0 && per_row.iter().all(|row| row.len() == outputs),
        "Qwen joint per-row projection returned inconsistent outputs"
    );
    // The gathers read every row's projection output.
    enc.memory_barrier();
    let mut packed = Vec::with_capacity(outputs);
    for output in 0..outputs {
        let row_width = per_row[0][output].element_count();
        let joint = pooled_f32(device, vec![width, row_width], "per-row projection")?;
        for (row, row_outputs) in per_row.iter().enumerate() {
            ensure!(
                row_outputs[output].element_count() == row_width,
                "Qwen joint per-row projection width changed at row {row}"
            );
            copy_row(
                enc,
                registry,
                device,
                &row_outputs[output],
                &joint,
                row,
                row_width,
            )?;
        }
        packed.push(joint);
    }
    Ok(packed)
}

fn single(mut outputs: Vec<MlxBuffer>, label: &str) -> Result<MlxBuffer> {
    ensure!(
        outputs.len() == 1,
        "Qwen joint {label} projection returned {} outputs",
        outputs.len()
    );
    Ok(outputs.remove(0))
}

/// Route for the full-attention Q/gate projection.
pub(crate) fn q_gate_route(weights: &FullAttnQGateWeightsGpu) -> Result<JointProductRoute, String> {
    match weights {
        FullAttnQGateWeightsGpu::Fused { weight, ggml_type } => {
            projection_route("attn_q_gate", weight.dtype(), *ggml_type)
        }
        FullAttnQGateWeightsGpu::Split {
            wq,
            wq_ggml_type,
            w_gate,
            w_gate_ggml_type,
        } => {
            let q = projection_route("attn_q", wq.dtype(), *wq_ggml_type)?;
            let gate = projection_route("attn_gate", w_gate.dtype(), *w_gate_ggml_type)?;
            Ok(
                if q == JointProductRoute::Batched && gate == JointProductRoute::Batched {
                    JointProductRoute::Batched
                } else {
                    JointProductRoute::PerRow
                },
            )
        }
    }
}

/// Every product route of one full-attention layer.
pub(crate) struct FullAttnRoutes {
    q_gate: JointProductRoute,
    k: JointProductRoute,
    v: JointProductRoute,
    o: JointProductRoute,
}

impl FullAttnRoutes {
    pub(crate) fn plan(layer: usize, weights: &FullAttnWeightsGpu) -> Result<Self, String> {
        let at = |error: String| format!("layer {layer}: {error}");
        Ok(Self {
            q_gate: q_gate_route(&weights.q_gate).map_err(at)?,
            k: projection_route("attn_k", weights.wk.dtype(), weights.wk_ggml_type).map_err(at)?,
            v: projection_route("attn_v", weights.wv.dtype(), weights.wv_ggml_type).map_err(at)?,
            o: projection_route("attn_output", weights.wo.dtype(), weights.wo_ggml_type)
                .map_err(at)?,
        })
    }
}

/// Every product route of one DeltaNet layer.
pub(crate) struct DeltaNetRoutes {
    qkv: JointProductRoute,
    gate: JointProductRoute,
    alpha: JointProductRoute,
    beta: JointProductRoute,
    out: JointProductRoute,
}

impl DeltaNetRoutes {
    pub(crate) fn plan(layer: usize, weights: &DeltaNetWeightsGpu) -> Result<Self, String> {
        let at = |error: String| format!("layer {layer}: {error}");
        let route = |name: &str, buffer: &MlxBuffer, ggml_type: GgmlType| {
            projection_route(name, buffer.dtype(), ggml_type).map_err(at)
        };
        Ok(Self {
            qkv: route("attn_qkv", &weights.attn_qkv, weights.attn_qkv_ggml_type)?,
            gate: route("attn_gate", &weights.attn_gate, weights.attn_gate_ggml_type)?,
            alpha: route("ssm_alpha", &weights.ssm_alpha, weights.ssm_alpha_ggml_type)?,
            beta: route("ssm_beta", &weights.ssm_beta, weights.ssm_beta_ggml_type)?,
            out: route("ssm_out", &weights.ssm_out, weights.ssm_out_ggml_type)?,
        })
    }
}

fn check_slots(kv_cache: &HybridKvCache, rows: &JointRows<'_>) -> Result<()> {
    for slot in rows.slots {
        ensure!(
            slot.0 < kv_cache.n_seqs,
            "Qwen joint slot {} outside n_seqs={}",
            slot.0,
            kv_cache.n_seqs
        );
    }
    let mut next = 0;
    for (index, window) in rows.windows.iter().enumerate() {
        let end = window.start + window.rows;
        ensure!(
            window.rows > 0 && window.start == next && end <= rows.real(),
            "Qwen joint windows must cover the real rows in packed order"
        );
        ensure!(
            rows.slots[window.start..end]
                .iter()
                .all(|&slot| slot == window.slot),
            "Qwen joint window rows belong to slot {}",
            window.slot.0
        );
        ensure!(
            rows.windows[..index]
                .iter()
                .all(|earlier| earlier.slot != window.slot),
            "Qwen joint slot {} appears in two windows",
            window.slot.0
        );
        next = end;
    }
    ensure!(
        next == rows.real() && rows.width >= next,
        "Qwen joint windows cover {next} of {} real rows (width {})",
        rows.real(),
        rows.width
    );
    Ok(())
}

/// Elements per [`DeltaNetReplay`] allocation: 16-byte aligned so every
/// view starts on the alignment the MMA and copy kernels expect.
fn replay_elements(elements: usize) -> usize {
    elements.next_multiple_of(4)
}

/// What a verify step's DeltaNet layers retain for the replay commit
/// ([`encode_delta_net_replay`]), in one durable allocation that outlives
/// the decode pool: per DeltaNet layer the packed `g`/`beta` rows and, per
/// multi-row window, the scaled Q, L2-normed K, V and raw conv input rows
/// plus a replay output scratch. Single-row windows retain nothing: their
/// commit is a fast path (keep, or discard by a parity flip).
pub(crate) struct DeltaNetReplay {
    arena: MlxBuffer,
    used: usize,
    layers: Vec<DeltaNetReplayLayer>,
}

struct DeltaNetReplayLayer {
    rank: usize,
    lanes: Vec<DeltaNetReplayLane>,
}

/// One multi-row window's replay inputs in one DeltaNet layer.
struct DeltaNetReplayLane {
    slot: SlotId,
    rows: usize,
    /// `[rows, n_k_heads * d_k]`, already scaled.
    q: MlxBuffer,
    /// `[rows, n_k_heads * d_k]`, L2-normed.
    k: MlxBuffer,
    /// `[rows, n_v_heads * d_v]`.
    v: MlxBuffer,
    /// `[rows, n_v_heads]` views of the layer's packed `g` and `beta`.
    g: MlxBuffer,
    beta: MlxBuffer,
    /// `[K-1 + rows, channels]`: K-1 staging rows the commit fills with the
    /// original conv state, then the window's raw conv inputs.
    conv_rows: MlxBuffer,
    /// `[rows, n_v_heads * d_v]` replay output scratch (discarded).
    output: MlxBuffer,
}

/// Per-layer buffers a multi-row window stages during the verify forward.
struct WindowStage {
    q: MlxBuffer,
    k: MlxBuffer,
    v: MlxBuffer,
    conv_rows: MlxBuffer,
    output: MlxBuffer,
}

impl DeltaNetReplay {
    /// Storage for `linear_layers` DeltaNet layers of a step with `windows`
    /// at packed `width`; `None` when no window has more than one row.
    pub(crate) fn new(
        device: &MlxDevice,
        cfg: &Qwen35Config,
        windows: &[JointWindow],
        width: usize,
        linear_layers: usize,
    ) -> Result<Option<Self>> {
        if windows.iter().all(|window| window.rows == 1) {
            return Ok(None);
        }
        let shape = DeltaNetLayerShape::from_config(cfg);
        let q_span = (shape.n_k_heads * shape.d_k) as usize;
        let v_elems = (shape.n_v_heads * shape.d_v) as usize;
        let channels = 2 * q_span + v_elems;
        let k_minus_one = shape.conv_kernel as usize - 1;
        let nv = shape.n_v_heads as usize;
        let per_layer = 2 * replay_elements(width * nv)
            + windows
                .iter()
                .filter(|window| window.rows > 1)
                .map(|window| {
                    let n = window.rows;
                    replay_elements((k_minus_one + n) * channels)
                        + 2 * replay_elements(n * q_span)
                        + 2 * replay_elements(n * v_elems)
                })
                .sum::<usize>();
        let elements = per_layer
            .checked_mul(linear_layers)
            .context("Qwen joint verify replay storage overflow")?;
        let arena = device
            .alloc_buffer(elements * 4, DType::F32, vec![elements])
            .map_err(|e| anyhow!("allocate Qwen joint verify replay storage: {e}"))?;
        Ok(Some(Self {
            arena,
            used: 0,
            layers: Vec::with_capacity(linear_layers),
        }))
    }

    fn take(&mut self, elements: usize) -> Result<MlxBuffer> {
        let end = self.used + replay_elements(elements);
        ensure!(
            end <= self.arena.element_count(),
            "Qwen joint verify replay storage exhausted ({end} of {} elements)",
            self.arena.element_count()
        );
        let view = self
            .arena
            .slice_view((self.used * 4) as u64, elements)
            .with_shape(vec![elements])
            .map_err(|e| anyhow!("shape Qwen joint verify replay view: {e}"))?;
        self.used = end;
        Ok(view)
    }
}

fn ssm_conv_params_buffer(device: &MlxDevice, params: SsmConvParams) -> Result<MlxBuffer> {
    let mut buffer = pooled_alloc_buffer(device, 16, DType::U32, vec![4])
        .map_err(|e| anyhow!("allocate Qwen joint ssm params: {e}"))?;
    buffer
        .as_mut_slice::<u32>()
        .map_err(|e| anyhow!("{e}"))?
        .copy_from_slice(&[
            params.channels,
            params.n_tokens,
            params.n_seqs,
            params.k_width,
        ]);
    Ok(buffer)
}

fn write_gdn_params(buffer: &mut MlxBuffer, params: GatedDeltaNetParams) -> Result<()> {
    buffer
        .as_mut_slice::<u32>()
        .map_err(|e| anyhow!("{e}"))?
        .copy_from_slice(&[
            params.d_k,
            params.d_v,
            params.n_k_heads,
            params.n_v_heads,
            params.n_tokens,
            params.n_seqs,
            0,
            0,
            0,
        ]);
    Ok(())
}

/// Kernel parameter buffers of one layer, one per window length.
struct WindowParams(Vec<(usize, MlxBuffer)>);

impl WindowParams {
    fn new() -> Self {
        Self(Vec::new())
    }

    fn get(
        &mut self,
        rows: usize,
        make: impl FnOnce() -> Result<MlxBuffer>,
    ) -> Result<&MlxBuffer> {
        let index = match self.0.iter().position(|(len, _)| *len == rows) {
            Some(index) => index,
            None => {
                self.0.push((rows, make()?));
                self.0.len() - 1
            }
        };
        Ok(&self.0[index].1)
    }
}

fn gdn_params(shape: &DeltaNetLayerShape, rows: usize) -> GatedDeltaNetParams {
    GatedDeltaNetParams {
        d_k: shape.d_k,
        d_v: shape.d_v,
        n_k_heads: shape.n_k_heads,
        n_v_heads: shape.n_v_heads,
        n_tokens: rows as u32,
        n_seqs: FORWARD_DISPATCH_N_SEQS,
    }
}

/// Encode the replay commit of a verify step's DeltaNet state. For every
/// lane window with `accepted` rows in `1..rows` (`accepted` lists
/// `(slot, accepted rows)`; other lanes are skipped) and every DeltaNet
/// layer, rebuild the slot's current state as the state after its first
/// `accepted` rows:
///
/// - recurrent: one `gated_delta_net_decode` over those rows from the
///   untouched pre-window state (the verify flipped the parity, so it is
///   the slot's scratch role now) into the current role;
/// - conv: byte copies only, `new_state(i, c) = [old ‖ rows](accepted + i, c)`
///   (the `ssm_conv` state rule): the old state is staged transposed in
///   front of the retained raw rows, then the window starting at
///   `accepted` is copied back in the state layout.
///
/// No parity changes. The caller commits `enc`.
pub(crate) fn encode_delta_net_replay(
    enc: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    kv_cache: &HybridKvCache,
    replay: &DeltaNetReplay,
    accepted: &[(SlotId, usize)],
    cfg: &Qwen35Config,
) -> Result<()> {
    let shape = DeltaNetLayerShape::from_config(cfg);
    let (d_k, d_v, n_v_heads) = (shape.d_k, shape.d_v, shape.n_v_heads);
    let q_span = (shape.n_k_heads * d_k) as usize;
    let v_elems = (n_v_heads * d_v) as usize;
    let channels = 2 * q_span + v_elems;
    let k_minus_one = shape.conv_kernel - 1;
    let km1 = k_minus_one as usize;
    let nv = n_v_heads as usize;
    let partial = |lane: &DeltaNetReplayLane| {
        accepted
            .iter()
            .find(|(slot, _)| *slot == lane.slot)
            .map(|&(_, rows)| rows)
            .filter(|&rows| (1..lane.rows).contains(&rows))
    };
    // (pre-window state, current state) views of one lane's slot.
    let views = |rank: usize, slot: SlotId| -> Result<_> {
        let la = kv_cache
            .linear_attn
            .get(rank)
            .context("Qwen joint verify replay layer outside the cache")?;
        let (conv_cur, conv_prior) = la.conv_bufs_for_slot(slot);
        let (rec_cur, rec_prior) = la.recurrent_bufs_for_slot(slot);
        Ok(narrow_la_ping_pong_to_slot(
            conv_prior,
            conv_cur,
            rec_prior,
            rec_cur,
            slot,
            d_k,
            d_v,
            n_v_heads,
            channels as u32,
            k_minus_one,
        ))
    };
    for layer in &replay.layers {
        for lane in &layer.lanes {
            if partial(lane).is_none() {
                continue;
            }
            let view = views(layer.rank, lane.slot)?;
            dispatch_strided_copy_f32(
                enc,
                registry,
                device.metal_device(),
                &view.conv_state_in,
                &rows_view(&lane.conv_rows, 0, km1, channels)?,
                &StridedCopyParams {
                    rows: k_minus_one,
                    cols: channels as u32,
                    stride_row: 1,
                    stride_col: k_minus_one,
                },
            )
            .with_context(|| format!("stage Qwen joint verify conv state slot {}", lane.slot.0))?;
        }
    }
    enc.memory_barrier();
    let mut params = WindowParams::new();
    for layer in &replay.layers {
        for lane in &layer.lanes {
            let Some(rows) = partial(lane) else {
                continue;
            };
            let view = views(layer.rank, lane.slot)?;
            dispatch_strided_copy_f32(
                enc,
                registry,
                device.metal_device(),
                &rows_view(&lane.conv_rows, rows, km1, channels)?,
                &view.conv_state_out,
                &StridedCopyParams {
                    rows: channels as u32,
                    cols: k_minus_one,
                    stride_row: 1,
                    stride_col: channels as u32,
                },
            )
            .with_context(|| {
                format!("rebuild Qwen joint verify conv state slot {}", lane.slot.0)
            })?;
            let p = gdn_params(&shape, rows);
            let params_buf = params.get(rows, || {
                let mut buffer = device
                    .alloc_buffer(9 * 4, DType::U32, vec![9])
                    .map_err(|e| anyhow!("allocate Qwen joint replay GDN params: {e}"))?;
                write_gdn_params(&mut buffer, p)?;
                Ok(buffer)
            })?;
            dispatch_gated_delta_net_decode(
                enc,
                registry,
                device.metal_device(),
                &rows_view(&lane.q, 0, rows, q_span)?,
                &rows_view(&lane.k, 0, rows, q_span)?,
                &rows_view(&lane.v, 0, rows, v_elems)?,
                &rows_view(&lane.g, 0, rows, nv)?,
                &rows_view(&lane.beta, 0, rows, nv)?,
                &view.recurrent_in,
                &rows_view(&lane.output, 0, rows, v_elems)?,
                &view.recurrent_out,
                params_buf,
                p,
            )
            .with_context(|| format!("replay Qwen joint verify GDN slot {}", lane.slot.0))?;
        }
    }
    Ok(())
}

/// Encode one DeltaNet layer for every joint row into `enc` (no commit) and
/// flip each window's ping-pong parity once, exactly as the serial
/// single-token path does after encoding its layer. Each window runs one
/// `ssm_conv` and one gated delta net dispatch over its rows (`n_tokens` =
/// its row count), reading the slot's current state and writing its
/// opposite parity. Multi-row windows (MTP verify) need `replay`, which
/// retains their inputs for [`encode_delta_net_replay`]. Returns the packed
/// `[width, hidden]` attention output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_joint_delta_net_layer(
    enc: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &MlxBuffer,
    weights: &DeltaNetWeightsGpu,
    routes: &DeltaNetRoutes,
    kv_cache: &mut HybridKvCache,
    layer_idx: usize,
    rows: &JointRows<'_>,
    cfg: &Qwen35Config,
    mut replay: Option<&mut DeltaNetReplay>,
) -> Result<MlxBuffer> {
    check_slots(kv_cache, rows)?;
    let shape = DeltaNetLayerShape::from_config(cfg);
    let rank = match kv_cache.slot_index_for_layer(layer_idx as u32) {
        Some(LayerSlot::Linear(rank)) => rank as usize,
        other => bail!("Qwen joint layer {layer_idx}: expected DeltaNet slot, got {other:?}"),
    };
    let width = rows.width;
    let width_u32 = rows.width_u32();
    let hidden = shape.hidden_size;
    let eps = shape.rms_norm_eps;
    let (n_k_heads, n_v_heads, d_k, d_v) = (shape.n_k_heads, shape.n_v_heads, shape.d_k, shape.d_v);
    let k_width = shape.conv_kernel;
    let qkv_channels = 2 * n_k_heads * d_k + n_v_heads * d_v;
    let z_channels = n_v_heads * d_v;
    let q_span = (n_k_heads * d_k) as usize;
    let k_span = q_span;
    let qkv_ch = qkv_channels as usize;
    let nv = n_v_heads as usize;
    let v_elems = nv * d_v as usize;

    // ---- Packed: pre-norm and the four input projections ----
    let x_norm = apply_pre_norm(
        enc,
        registry,
        device,
        x,
        &weights.attn_norm,
        width_u32,
        hidden,
        eps,
    )?;
    enc.memory_barrier();
    let input_projection = |enc: &mut CommandEncoder,
                            registry: &mut KernelRegistry,
                            route: JointProductRoute,
                            weight: &MlxBuffer,
                            ggml_type: GgmlType,
                            out_features: u32,
                            label: &str| {
        project_rows(
            enc,
            registry,
            device,
            route,
            &x_norm,
            width,
            hidden as usize,
            |enc, registry, input, seq| {
                apply_proj_with_ggml_type(
                    enc,
                    registry,
                    device,
                    input,
                    weight,
                    ggml_type,
                    seq,
                    hidden,
                    out_features,
                )
                .map(|output| vec![output])
            },
        )
        .and_then(|outputs| single(outputs, label))
    };
    let qkv_raw = input_projection(
        enc,
        registry,
        routes.qkv,
        &weights.attn_qkv,
        weights.attn_qkv_ggml_type,
        qkv_channels,
        "attn_qkv",
    )?;
    let z = input_projection(
        enc,
        registry,
        routes.gate,
        &weights.attn_gate,
        weights.attn_gate_ggml_type,
        z_channels,
        "attn_gate",
    )?;
    let alpha_logit = input_projection(
        enc,
        registry,
        routes.alpha,
        &weights.ssm_alpha,
        weights.ssm_alpha_ggml_type,
        n_v_heads,
        "ssm_alpha",
    )?;
    let beta_logit = input_projection(
        enc,
        registry,
        routes.beta,
        &weights.ssm_beta,
        weights.ssm_beta_ggml_type,
        n_v_heads,
        "ssm_beta",
    )?;
    enc.memory_barrier();

    // ---- Per window: causal conv against the slot's ping-pong parity ----
    let qkv_conv = pooled_f32(device, vec![width, qkv_ch], "DeltaNet conv output")?;
    let k_minus_one = (k_width - 1) as usize;
    let mut ssm_params = WindowParams::new();
    let mut lanes = Vec::with_capacity(rows.windows.len());
    {
        let slot = &kv_cache.linear_attn[rank];
        for window in rows.windows {
            let n = window.rows;
            let (conv_cur, conv_scratch) = slot.conv_bufs_for_slot(window.slot);
            let (rec_cur, rec_scratch) = slot.recurrent_bufs_for_slot(window.slot);
            let view = narrow_la_ping_pong_to_slot(
                conv_cur,
                conv_scratch,
                rec_cur,
                rec_scratch,
                window.slot,
                d_k,
                d_v,
                n_v_heads,
                qkv_channels,
                k_width - 1,
            );
            let conv_params = SsmConvParams {
                channels: qkv_channels,
                n_tokens: n as u32,
                n_seqs: FORWARD_DISPATCH_N_SEQS,
                k_width,
            };
            dispatch_ssm_conv(
                enc,
                registry,
                device.metal_device(),
                &rows_view(&qkv_raw, window.start, n, qkv_ch)?,
                &weights.ssm_conv1d,
                &view.conv_state_in,
                &view.conv_state_out,
                &rows_view(&qkv_conv, window.start, n, qkv_ch)?,
                ssm_params.get(n, || ssm_conv_params_buffer(device, conv_params))?,
                conv_params,
            )
            .with_context(|| {
                format!(
                    "Qwen joint ssm_conv layer {layer_idx} rows {}..{}",
                    window.start,
                    window.start + n
                )
            })?;
            let stage = if n > 1 {
                let replay = replay
                    .as_deref_mut()
                    .context("Qwen joint multi-row window requires replay storage")?;
                let conv_rows = replay.take((k_minus_one + n) * qkv_ch)?;
                dispatch_copy_f32(
                    enc,
                    registry,
                    device.metal_device(),
                    &qkv_raw,
                    &conv_rows,
                    window.start * qkv_ch,
                    k_minus_one * qkv_ch,
                    n * qkv_ch,
                )
                .map_err(|e| anyhow!("retain Qwen joint conv rows layer {layer_idx}: {e}"))?;
                Some(WindowStage {
                    conv_rows,
                    q: replay.take(n * q_span)?,
                    k: replay.take(n * k_span)?,
                    v: replay.take(n * v_elems)?,
                    output: replay.take(n * v_elems)?,
                })
            } else {
                None
            };
            lanes.push((view, stage));
        }
    }
    enc.memory_barrier();

    // ---- Per row: L2 norms of Q and K; a multi-row window gathers its
    // ---- K and V rows contiguously for the window dispatch ----
    let mut q_l2 = Vec::with_capacity(rows.real());
    let mut window_k = Vec::with_capacity(rows.windows.len());
    for (window, (_, stage)) in rows.windows.iter().zip(&lanes) {
        let mut single_k = None;
        for r in 0..window.rows {
            let base = (window.start + r) * qkv_ch;
            let q_view = qkv_conv.slice_view((base * 4) as u64, q_span);
            let k_view = qkv_conv.slice_view(((base + q_span) * 4) as u64, k_span);
            q_l2.push(apply_l2_norm_per_head(
                enc, registry, device, &q_view, 1, n_k_heads, d_k, eps,
            )?);
            match stage {
                None => {
                    single_k = Some(apply_l2_norm_per_head(
                        enc, registry, device, &k_view, 1, n_k_heads, d_k, eps,
                    )?);
                }
                Some(stage) => {
                    let mut l2_params = pooled_alloc_buffer(device, 8, DType::F32, vec![2])
                        .map_err(|e| anyhow!("allocate Qwen joint l2 params: {e}"))?;
                    apply_l2_norm_per_head_into(
                        enc,
                        registry,
                        device,
                        &k_view,
                        &stage.k.slice_view((r * k_span * 4) as u64, k_span),
                        &mut l2_params,
                        1,
                        n_k_heads,
                        d_k,
                        eps,
                    )?;
                    dispatch_copy_f32(
                        enc,
                        registry,
                        device.metal_device(),
                        &qkv_conv,
                        &stage.v,
                        base + q_span + k_span,
                        r * v_elems,
                        v_elems,
                    )
                    .map_err(|e| anyhow!("gather Qwen joint window V row {r}: {e}"))?;
                }
            }
        }
        window_k.push(single_k);
    }
    enc.memory_barrier();

    // ---- Per row q scale; packed g/beta ----
    let q_scale = 1.0_f32 / (d_k as f32).sqrt();
    let mut window_q = Vec::with_capacity(rows.windows.len());
    let mut q_rows = q_l2.iter();
    for (window, (_, stage)) in rows.windows.iter().zip(&lanes) {
        let scaled = match stage {
            None => pooled_f32(device, vec![q_span], "scaled query")?,
            Some(stage) => stage.q.clone(),
        };
        for r in 0..window.rows {
            let q = q_rows.next().context("Qwen joint q rows")?;
            scalar_mul_f32(
                enc,
                registry,
                device.metal_device(),
                q,
                &scaled.slice_view((r * q_span * 4) as u64, q_span),
                q_span,
                q_scale,
            )
            .context("Qwen joint q scale")?;
        }
        window_q.push(scaled);
    }
    let (g_buf, beta_buf) = match replay.as_deref_mut() {
        Some(replay) => (replay.take(width * nv)?, replay.take(width * nv)?),
        None => (
            pooled_f32(device, vec![width * nv], "DeltaNet g")?,
            pooled_f32(device, vec![width * nv], "DeltaNet beta")?,
        ),
    };
    let mut g_params = pooled_alloc_buffer(device, 8, DType::U32, vec![2])
        .map_err(|e| anyhow!("allocate Qwen joint g params: {e}"))?;
    {
        let s = g_params.as_mut_slice::<u32>().map_err(|e| anyhow!("{e}"))?;
        s[0] = n_v_heads;
        s[1] = width_u32;
    }
    dispatch_compute_g_beta(
        enc,
        registry,
        device.metal_device(),
        &alpha_logit,
        &beta_logit,
        &weights.ssm_dt_bias,
        &weights.ssm_a,
        &g_buf,
        &beta_buf,
        &g_params,
        width_u32,
        n_v_heads,
    )
    .context("Qwen joint compute_g_beta")?;
    enc.memory_barrier();

    // ---- Per window: gated delta net against the slot's recurrent state ----
    let mut gdn_params_bufs = WindowParams::new();
    let attn_out = pooled_f32(device, vec![width * v_elems], "DeltaNet core output")?;
    for (index, (window, (view, stage))) in rows.windows.iter().zip(&lanes).enumerate() {
        let n = window.rows;
        let p = gdn_params(&shape, n);
        let (k, v) = match stage {
            None => (
                window_k[index]
                    .clone()
                    .context("Qwen joint single-row K")?,
                qkv_conv.slice_view(
                    ((window.start * qkv_ch + q_span + k_span) * 4) as u64,
                    v_elems,
                ),
            ),
            Some(stage) => (stage.k.clone(), stage.v.clone()),
        };
        let params_buf = gdn_params_bufs.get(n, || {
            let mut buffer = pooled_alloc_buffer(device, 9 * 4, DType::U32, vec![9])
                .map_err(|e| anyhow!("allocate Qwen joint GDN params: {e}"))?;
            write_gdn_params(&mut buffer, p)?;
            Ok(buffer)
        })?;
        dispatch_gated_delta_net_decode(
            enc,
            registry,
            device.metal_device(),
            &window_q[index],
            &k,
            &v,
            &g_buf.slice_view((window.start * nv * 4) as u64, n * nv),
            &beta_buf.slice_view((window.start * nv * 4) as u64, n * nv),
            &view.recurrent_in,
            &attn_out.slice_view((window.start * v_elems * 4) as u64, n * v_elems),
            &view.recurrent_out,
            params_buf,
            p,
        )
        .with_context(|| {
            format!(
                "Qwen joint gated delta net layer {layer_idx} rows {}..{}",
                window.start,
                window.start + n
            )
        })?;
    }
    enc.memory_barrier();
    if rows.width > rows.real() {
        fill_padding_rows(enc, registry, device, &attn_out, rows, v_elems)?;
        enc.memory_barrier();
    }

    // ---- Packed: gated RMS norm and output projection ----
    let gated = pooled_f32(device, vec![width * v_elems], "DeltaNet gated output")?;
    let mut op8_params = pooled_alloc_buffer(device, 8, DType::F32, vec![2])
        .map_err(|e| anyhow!("allocate Qwen joint norm-gate params: {e}"))?;
    {
        let s = op8_params
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow!("{e}"))?;
        s[0] = eps;
        s[1] = d_v as f32;
    }
    dispatch_ssm_norm_gate(
        enc,
        registry,
        device.metal_device(),
        &attn_out,
        &weights.ssm_norm,
        &z,
        &gated,
        &op8_params,
        width_u32 * n_v_heads,
        d_v,
    )
    .context("Qwen joint ssm_norm_gate")?;
    enc.memory_barrier();
    let gated = gated
        .with_shape(vec![width, z_channels as usize])
        .map_err(|e| anyhow!("shape Qwen joint gated output: {e}"))?;
    let output = project_rows(
        enc,
        registry,
        device,
        routes.out,
        &gated,
        width,
        z_channels as usize,
        |enc, registry, input, seq| {
            apply_proj_with_ggml_type(
                enc,
                registry,
                device,
                input,
                &weights.ssm_out,
                weights.ssm_out_ggml_type,
                seq,
                z_channels,
                hidden,
            )
            .map(|output| vec![output])
        },
    )
    .and_then(|outputs| single(outputs, "ssm_out"))?;

    // Same CPU-side parity flip the serial path performs after encoding,
    // once per window: a window's dispatches read the current parity and
    // wrote the opposite one.
    let slot = &mut kv_cache.linear_attn[rank];
    for window in rows.windows {
        slot.swap_for_slot(window.slot);
    }
    if let Some(replay) = replay {
        let lanes = rows
            .windows
            .iter()
            .zip(lanes)
            .filter_map(|(window, (_, stage))| {
                stage.map(|stage| {
                    let rows_nv = window.rows * nv;
                    let offset = (window.start * nv * 4) as u64;
                    DeltaNetReplayLane {
                        slot: window.slot,
                        rows: window.rows,
                        q: stage.q,
                        k: stage.k,
                        v: stage.v,
                        g: g_buf.slice_view(offset, rows_nv),
                        beta: beta_buf.slice_view(offset, rows_nv),
                        conv_rows: stage.conv_rows,
                        output: stage.output,
                    }
                })
            })
            .collect();
        replay.layers.push(DeltaNetReplayLayer { rank, lanes });
    }
    Ok(output)
}

/// Encode one gated full-attention layer for every joint row into `enc` (no
/// commit). Each real row writes its K/V at its own cursor and attends over
/// its own slot; the cursor update is the serial helper's. Returns the
/// packed `[width, hidden]` attention output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_joint_full_attn_layer(
    enc: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &MlxBuffer,
    positions: &MlxBuffer,
    weights: &FullAttnWeightsGpu,
    routes: &FullAttnRoutes,
    kv_cache: &mut HybridKvCache,
    layer_idx: usize,
    rows: &JointRows<'_>,
    cfg: &Qwen35Config,
) -> Result<MlxBuffer> {
    check_slots(kv_cache, rows)?;
    let shape = FullAttnShape::from_config(cfg);
    ensure!(
        shape.head_dim % 32 == 0,
        "Qwen joint full attention requires head_dim % 32 == 0, got {}",
        shape.head_dim
    );
    let rank = match kv_cache.slot_index_for_layer(layer_idx as u32) {
        Some(LayerSlot::Full(rank)) => rank as usize,
        other => bail!("Qwen joint layer {layer_idx}: expected FullAttn slot, got {other:?}"),
    };
    let width = rows.width;
    let width_u32 = rows.width_u32();
    let hidden = shape.hidden_size;
    let (n_heads, n_kv_heads, head_dim) = (shape.n_head, shape.n_kv, shape.head_dim);
    let q_total = (n_heads * head_dim) as usize;
    let kv_total = n_kv_heads * head_dim;

    // ---- Packed: pre-norm, projections, Q/K norms, IMROPE ----
    let x_norm = apply_pre_attn_rms_norm(
        enc,
        registry,
        device,
        x,
        weights,
        width_u32,
        hidden,
        shape.rms_norm_eps,
    )?;
    enc.memory_barrier();
    let mut q_gate = project_rows(
        enc,
        registry,
        device,
        routes.q_gate,
        &x_norm,
        width,
        hidden as usize,
        |enc, registry, input, seq| {
            let (q, gate) = apply_q_gate_projection_f32(
                enc,
                registry,
                device,
                input,
                &weights.q_gate,
                seq,
                n_heads,
                head_dim,
                hidden,
            )?;
            Ok(vec![q, gate])
        },
    )?;
    ensure!(q_gate.len() == 2, "Qwen joint Q/gate projection shape");
    let gate_flat = q_gate.pop().expect("gate output checked above");
    let q_flat = q_gate.pop().expect("query output checked above");
    let kv_projection = |enc: &mut CommandEncoder,
                         registry: &mut KernelRegistry,
                         route: JointProductRoute,
                         weight: &MlxBuffer,
                         ggml_type: GgmlType,
                         label: &str| {
        project_rows(
            enc,
            registry,
            device,
            route,
            &x_norm,
            width,
            hidden as usize,
            |enc, registry, input, seq| {
                apply_linear_projection_f32_pooled_with_ggml_type(
                    enc, registry, device, input, weight, ggml_type, seq, hidden, kv_total,
                )
                .map(|output| vec![output])
            },
        )
        .and_then(|outputs| single(outputs, label))
    };
    let k_flat = kv_projection(
        enc,
        registry,
        routes.k,
        &weights.wk,
        weights.wk_ggml_type,
        "attn_k",
    )?;
    let v_flat = kv_projection(
        enc,
        registry,
        routes.v,
        &weights.wv,
        weights.wv_ggml_type,
        "attn_v",
    )?;
    enc.memory_barrier();
    let q_normed = apply_q_or_k_per_head_rms_norm(
        enc,
        registry,
        device,
        &q_flat,
        &weights.attn_q_norm,
        width_u32,
        n_heads,
        head_dim,
        shape.rms_norm_eps,
    )?;
    let k_normed = apply_q_or_k_per_head_rms_norm(
        enc,
        registry,
        device,
        &k_flat,
        &weights.attn_k_norm,
        width_u32,
        n_kv_heads,
        head_dim,
        shape.rms_norm_eps,
    )?;
    enc.memory_barrier();
    let q_rope = apply_imrope(
        enc,
        registry,
        device,
        &q_normed,
        positions,
        width_u32,
        n_heads,
        head_dim,
        shape.rotary_dim,
        shape.rope_theta,
        shape.mrope_section,
    )?;
    let k_rope = apply_imrope(
        enc,
        registry,
        device,
        &k_normed,
        positions,
        width_u32,
        n_kv_heads,
        head_dim,
        shape.rotary_dim,
        shape.rope_theta,
        shape.mrope_section,
    )?;
    enc.memory_barrier();

    // ---- Per slot: KV write at the slot cursor + SDPA over the slot ----
    let max_seq_len = kv_cache.max_seq_len;
    let slot = &mut kv_cache.full_attn[rank];
    let mut row_outputs = Vec::with_capacity(rows.real());
    for (row, &slot_id) in rows.slots.iter().enumerate() {
        if row > 0 {
            enc.memory_barrier();
        }
        let q_row = row_view(&q_rope, row, vec![1, n_heads as usize, head_dim as usize])?;
        let k_row = row_view(
            &k_rope,
            row,
            vec![1, n_kv_heads as usize, head_dim as usize],
        )?;
        let v_row = row_view(&v_flat, row, vec![1, kv_total as usize])?;
        let output = apply_sdpa_with_kv_cache_decode_into(
            enc,
            device,
            registry,
            &q_row,
            &k_row,
            &v_row,
            slot,
            1,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq_len,
            slot_id,
        )
        .with_context(|| format!("Qwen joint SDPA layer {layer_idx} row {row}"))?;
        row_outputs.push(output);
    }
    enc.memory_barrier();
    let attention = pooled_f32(device, vec![width, q_total], "attention output")?;
    for (row, output) in row_outputs.iter().enumerate() {
        copy_row(enc, registry, device, output, &attention, row, q_total)?;
    }
    for row in rows.real()..width {
        copy_row(
            enc,
            registry,
            device,
            &row_outputs[0],
            &attention,
            row,
            q_total,
        )?;
    }
    enc.memory_barrier();

    // ---- Packed: sigmoid gate and output projection ----
    let gated = apply_sigmoid_gate_multiply(
        enc,
        registry,
        device,
        &attention,
        &gate_flat,
        (width * q_total) as u32,
    )?;
    enc.memory_barrier();
    let gated = gated
        .with_shape(vec![width, q_total])
        .map_err(|e| anyhow!("shape Qwen joint gated attention: {e}"))?;
    project_rows(
        enc,
        registry,
        device,
        routes.o,
        &gated,
        width,
        q_total,
        |enc, registry, input, seq| {
            apply_linear_projection_f32_pooled_with_ggml_type(
                enc,
                registry,
                device,
                input,
                &weights.wo,
                weights.wo_ggml_type,
                seq,
                q_total as u32,
                hidden,
            )
            .map(|output| vec![output])
        },
    )
    .and_then(|outputs| single(outputs, "attn_output"))
}
