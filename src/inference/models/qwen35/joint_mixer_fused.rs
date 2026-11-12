//! Fused mixer stage of a Qwen DFlash2 target verify step
//! (`HF2Q_QWEN_VERIFY_MIXER_FUSED=1`).
//!
//! A DeltaNet layer of a four-lane verify step is about 155 dispatches in
//! ten serialized stages: its three prepared input products each as a
//! split-grid pass, a barrier and a reduce pass, and around the recurrence
//! one dispatch per row for the Q norm, the K norm, the V gather and the
//! query scale and one per window for the convolution, the raw-row copy and
//! the recurrence itself. The fused stage issues
//!
//! - the prepared input products of a layer (DeltaNet alpha/beta, qkv and z
//!   gate; attention Q/gate, K and V) as one wave, one kernel pass each with
//!   no barrier between them ([`super::prepared_projection::Pass::One`]), and
//! - the DeltaNet core of every window (convolution, Q/K norms, query scale,
//!   gates, recurrence, gated norm, the rows the replay commit retains) as
//!   one dispatch that writes the BF16 rows the output projection reads
//!   (mlx-native `gdn_window_fused`).
//!
//! Class A: every output bit, the DeltaNet state and the retained rows are
//! those of the stock route. Padding rows of the gated output are zero
//! instead of row 0's core output under their own gate; no real row reads
//! them.

use std::sync::{LazyLock, OnceLock};

use anyhow::{anyhow, ensure, Context, Result};
use mlx_native::ops::gdn_window_fused::{
    dispatch_gdn_window_fused, gdn_window_fused_available, GdnWindowInputs, GdnWindowLane,
    GdnWindowRows, GdnWindowShape, GdnWindowState, GdnWindowTape, GdnWindowWeights, HEAD_DIM,
    MAX_LANES, WINDOW_ROWS,
};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::decode_pool::pooled_alloc_buffer;
use super::delta_net::DeltaNetLayerShape;
use super::gpu_delta_net::DeltaNetWeightsGpu;
use super::joint_decode_core::{AlphaBeta, DeltaNetReplay, JointRows, WindowStage};
use super::kv_cache::LinearAttnStateSlot;
use super::prepared_projection;

/// Opt-in switch of the fused mixer stage; exact value `1`.
pub(crate) const QWEN_VERIFY_MIXER_FUSED_ENV: &str = "HF2Q_QWEN_VERIFY_MIXER_FUSED";

/// Whether the serving worker requests the fused stage (read once).
pub(crate) fn enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        std::env::var(QWEN_VERIFY_MIXER_FUSED_ENV)
            .map(|value| value == "1")
            .unwrap_or(false)
    });
    *ENABLED
}

/// Padding lanes that cover packed rows `real..width`.
fn padding_lanes(real: usize, width: usize) -> usize {
    (width - real).div_ceil(WINDOW_ROWS as usize)
}

/// Whether the DeltaNet core of this step runs as one dispatch: the kernel
/// serves the geometry, every window has at most [`WINDOW_ROWS`] rows, the
/// lanes fit the kernel's table and the device runs its pipelines. Otherwise
/// the step keeps the stock chain.
pub(super) fn delta_core_admits(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    shape: &DeltaNetLayerShape,
    rows: &JointRows<'_>,
) -> Result<bool> {
    let geometry = shape.d_k == HEAD_DIM
        && shape.d_v == HEAD_DIM
        && shape.n_k_heads > 0
        && shape.n_v_heads % shape.n_k_heads == 0
        && (2..=WINDOW_ROWS + 1).contains(&shape.conv_kernel);
    let lanes = rows
        .windows
        .iter()
        .all(|window| (1..=WINDOW_ROWS as usize).contains(&window.rows))
        && rows.windows.len() + padding_lanes(rows.real(), rows.width) <= MAX_LANES;
    if !(geometry && lanes) {
        return Ok(false);
    }
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    if let Some(available) = AVAILABLE.get() {
        return Ok(*available);
    }
    let available = gdn_window_fused_available(registry, device)
        .map_err(|error| anyhow!("probe the fused DeltaNet window kernel: {error}"))?;
    if !available {
        tracing::warn!(
            "Qwen fused mixer: the fused DeltaNet window kernel exceeds this device's \
             threadgroup limits; DeltaNet layers keep the stock chain"
        );
    }
    Ok(*AVAILABLE.get_or_init(|| available))
}

/// What the fused core of one layer produced.
pub(super) struct DeltaCore {
    /// Gated rows `[width, n_v_heads * d_v]` in the requested dtype.
    pub(super) gated: MlxBuffer,
    /// Per window, the rows a multi-row window retains for the replay.
    pub(super) stages: Vec<Option<WindowStage>>,
    /// Packed `g` and `beta` rows `[width, n_v_heads]` (replay storage when
    /// the step has it; unwritten otherwise).
    pub(super) g: MlxBuffer,
    pub(super) beta: MlxBuffer,
}

/// Encode the DeltaNet core of every window of a joint step as one dispatch
/// (the caller checked [`delta_core_admits`]): reads the layer's projection
/// rows and each window's slot state at its current parity, writes the gated
/// rows as `gated_dtype` (F32 or BF16), each slot's state after its window
/// at the opposite parity and, for multi-row windows with `replay`, the rows
/// the replay commit reads. The caller flips the parities.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_delta_core(
    enc: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    weights: &DeltaNetWeightsGpu,
    slot: &LinearAttnStateSlot,
    shape: &DeltaNetLayerShape,
    rows: &JointRows<'_>,
    qkv_raw: &MlxBuffer,
    z: &MlxBuffer,
    alpha_beta: &AlphaBeta,
    mut replay: Option<&mut DeltaNetReplay>,
    gated_dtype: DType,
) -> Result<DeltaCore> {
    let width = rows.width;
    let nv = shape.n_v_heads as usize;
    let q_span = (shape.n_k_heads * shape.d_k) as usize;
    let v_elems = nv * shape.d_v as usize;
    let channels = 2 * q_span + v_elems;
    let k_minus_one = shape.conv_kernel as usize - 1;
    let recurrent_slot = v_elems * shape.d_k as usize;
    let slots = slot.recurrent.element_count() / recurrent_slot;
    ensure!(
        slots > 0 && slot.conv_state.element_count() == slots * channels * k_minus_one,
        "Qwen fused DeltaNet core: state buffers hold {} recurrent and {} conv elements",
        slot.recurrent.element_count(),
        slot.conv_state.element_count()
    );

    // Retained rows, in the stock route's storage order.
    let mut stages = Vec::with_capacity(rows.windows.len());
    for window in rows.windows {
        let n = window.rows;
        stages.push(if n > 1 {
            let replay = replay
                .as_deref_mut()
                .context("Qwen joint multi-row window requires replay storage")?;
            Some(WindowStage {
                conv_rows: replay.take((k_minus_one + n) * channels)?,
                q: replay.take(n * q_span)?,
                k: replay.take(n * q_span)?,
                v: replay.take(n * v_elems)?,
                output: replay.take(n * v_elems)?,
            })
        } else {
            None
        });
    }
    let pooled = |label: &str| {
        pooled_alloc_buffer(device, width * nv * 4, DType::F32, vec![width * nv])
            .map_err(|error| anyhow!("allocate Qwen joint DeltaNet {label}: {error}"))
    };
    let (g, beta) = match replay.as_deref_mut() {
        Some(replay) => (replay.take(width * nv)?, replay.take(width * nv)?),
        None => (pooled("g")?, pooled("beta")?),
    };

    let mut lanes = Vec::with_capacity(rows.windows.len() + padding_lanes(rows.real(), width));
    for (window, stage) in rows.windows.iter().zip(&stages) {
        let tape = match (stage, replay.as_deref()) {
            (Some(stage), Some(replay)) => Some(GdnWindowTape {
                q: replay.offset_of(&stage.q)?,
                k: replay.offset_of(&stage.k)?,
                v: replay.offset_of(&stage.v)?,
                conv_rows: replay.offset_of(&stage.conv_rows)?
                    + u32::try_from(k_minus_one * channels)?,
                g: replay.offset_of(&g)? + u32::try_from(window.start * nv)?,
                beta: replay.offset_of(&beta)? + u32::try_from(window.start * nv)?,
            }),
            _ => None,
        };
        // Physical buffer 0 is the named state, 1 its scratch; the slot's
        // parity says which one is current.
        let (current, _) = slot.conv_bufs_for_slot(window.slot);
        lanes.push(GdnWindowLane::Window {
            first_row: u32::try_from(window.start)?,
            rows: u32::try_from(window.rows)?,
            slot: window.slot.0,
            state_in: u32::from(!std::ptr::eq(current, &slot.conv_state)),
            tape,
        });
    }
    for first in (rows.real()..width).step_by(WINDOW_ROWS as usize) {
        lanes.push(GdnWindowLane::Padding {
            first_row: u32::try_from(first)?,
            rows: u32::try_from((width - first).min(WINDOW_ROWS as usize))?,
        });
    }

    let gated = pooled_alloc_buffer(
        device,
        width * v_elems * gated_dtype.size_of(),
        gated_dtype,
        vec![width, v_elems],
    )
    .map_err(|error| anyhow!("allocate Qwen joint DeltaNet gated output: {error}"))?;
    let beta_view;
    let (alpha_rows, beta_rows) = match alpha_beta {
        AlphaBeta::Separate(alpha, beta) => (
            GdnWindowRows {
                buffer: alpha,
                stride: shape.n_v_heads,
            },
            GdnWindowRows {
                buffer: beta,
                stride: shape.n_v_heads,
            },
        ),
        AlphaBeta::Fused(fused) => {
            // Alpha's columns, then beta's, in rows of the fused width.
            let stride = prepared_projection::alpha_beta_rows(shape.n_v_heads);
            beta_view = fused.slice_view(
                (nv * std::mem::size_of::<f32>()) as u64,
                width * stride as usize - nv,
            );
            (
                GdnWindowRows {
                    buffer: fused,
                    stride,
                },
                GdnWindowRows {
                    buffer: &beta_view,
                    stride,
                },
            )
        }
    };
    dispatch_gdn_window_fused(
        enc,
        registry,
        device,
        &GdnWindowShape {
            n_k_heads: shape.n_k_heads,
            n_v_heads: shape.n_v_heads,
            head_dim: shape.d_k,
            conv_taps: shape.conv_kernel,
            eps: shape.rms_norm_eps,
            q_scale: 1.0_f32 / (shape.d_k as f32).sqrt(),
            slots: u32::try_from(slots)?,
        },
        &GdnWindowInputs {
            qkv: GdnWindowRows {
                buffer: qkv_raw,
                stride: u32::try_from(channels)?,
            },
            z: GdnWindowRows {
                buffer: z,
                stride: u32::try_from(v_elems)?,
            },
            alpha: alpha_rows,
            beta: beta_rows,
        },
        &GdnWindowWeights {
            conv: &weights.ssm_conv1d,
            dt_bias: &weights.ssm_dt_bias,
            ssm_a: &weights.ssm_a,
            norm: &weights.ssm_norm,
        },
        &GdnWindowState {
            conv: [&slot.conv_state, &slot.conv_state_scratch],
            recurrent: [&slot.recurrent, &slot.recurrent_scratch],
        },
        &gated,
        replay.as_deref().map(DeltaNetReplay::storage),
        u32::try_from(width)?,
        &lanes,
    )
    .map_err(|error| anyhow!("dispatch fused DeltaNet window: {error}"))?;
    Ok(DeltaCore {
        gated,
        stages,
        g,
        beta,
    })
}
