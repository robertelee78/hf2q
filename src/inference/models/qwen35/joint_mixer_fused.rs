//! Fused mixer stage of a Qwen DFlash2 target verify step
//! (`HF2Q_QWEN_VERIFY_MIXER_FUSED=1`).
//!
//! A DeltaNet layer of a four-lane verify step is 159 dispatches in 14
//! serialized stages: its three prepared input products each as a
//! split-grid pass, a barrier and a reduce pass, and around the recurrence
//! one dispatch per row for the Q norm, the K norm, the V gather and the
//! query scale and one per window for the convolution, the raw-row copy and
//! the recurrence itself. The fused stage issues
//!
//! - the prepared input products of a layer (DeltaNet alpha/beta, qkv and z
//!   gate; attention Q/gate, K and V) as one wave, one kernel pass each with
//!   no barrier between them ([`super::prepared_projection::Pass::One`]),
//! - the pre-mixer RMS norm straight into the BF16 rows those products read
//!   when every product of the layer is prepared
//!   ([`super::prepared_projection::normed_bf16_input`]), and
//! - the DeltaNet core of every window (convolution, Q/K norms, query scale,
//!   gates, recurrence, gated norm, the rows the replay commit retains) as
//!   one dispatch that writes the BF16 rows the output projection reads
//!   (mlx-native `gdn_window_fused`).
//!
//! Class A: every output bit, the DeltaNet state and the retained rows are
//! those of the stock route. Padding rows of the gated output are zero
//! instead of row 0's core output under their own gate; no real row reads
//! them. The fused core's equality with the stock chain is qualified by test
//! on the Apple M5 Max; mlx-native `tests/test_gdn_window_fused.rs` is the
//! gate for any other GPU family.
//!
//! The core serves DeltaNet head dimension 128, two to nine convolution taps
//! and windows of at most eight rows, on a device whose limits admit its
//! 256-thread pipeline; a layer that does not qualify keeps the stock chain
//! behind the wave. The first use and the first refusal are logged.

use std::sync::{LazyLock, Once, OnceLock};

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

/// Whether this device runs the fused window pipelines (probed once).
fn device_runs_core(registry: &mut KernelRegistry, device: &MlxDevice) -> Result<bool> {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    if let Some(available) = AVAILABLE.get() {
        return Ok(*available);
    }
    let available = gdn_window_fused_available(registry, device)
        .map_err(|error| anyhow!("probe the fused DeltaNet window kernel: {error}"))?;
    Ok(*AVAILABLE.get_or_init(|| available))
}

/// Why the fused core does not serve this geometry or these rows, whatever
/// the device.
pub(super) fn geometry_refusal(
    shape: &DeltaNetLayerShape,
    rows: &JointRows<'_>,
) -> Option<&'static str> {
    if shape.d_k != HEAD_DIM || shape.d_v != HEAD_DIM {
        Some("the DeltaNet head dimension is not 128")
    } else if shape.n_k_heads == 0 || shape.n_v_heads % shape.n_k_heads != 0 {
        Some("the value heads are not a multiple of the key heads")
    } else if !(2..=WINDOW_ROWS + 1).contains(&shape.conv_kernel) {
        Some("the convolution has fewer than two or more than nine taps")
    } else if !rows
        .windows
        .iter()
        .all(|window| (1..=WINDOW_ROWS as usize).contains(&window.rows))
    {
        Some("a window has more than eight rows")
    } else if rows.windows.len() + padding_lanes(rows.real(), rows.width) > MAX_LANES {
        Some("the step has more lanes than the kernel's table")
    } else {
        None
    }
}

/// Whether the DeltaNet core of this step runs as one dispatch: head
/// dimension [`HEAD_DIM`], two to nine convolution taps, every window of at
/// most [`WINDOW_ROWS`] rows, the lanes within the kernel's table, and a
/// device whose thread and threadgroup-memory limits admit its pipelines.
/// Otherwise the layer keeps the stock chain; the first refusal is logged
/// with its reason.
pub(super) fn delta_core_admits(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    shape: &DeltaNetLayerShape,
    rows: &JointRows<'_>,
) -> Result<bool> {
    let refusal = match geometry_refusal(shape, rows) {
        Some(reason) => Some(reason),
        None if !device_runs_core(registry, device)? => {
            Some("the kernel exceeds this device's thread or threadgroup-memory limits")
        }
        None => None,
    };
    if let Some(reason) = refusal {
        // The switch asked for the stage; say once why a layer keeps the
        // stock chain.
        static FIRST_REFUSAL: Once = Once::new();
        FIRST_REFUSAL.call_once(|| {
            tracing::info!(
                reason,
                key_head_dim = shape.d_k,
                value_head_dim = shape.d_v,
                num_key_heads = shape.n_k_heads,
                num_value_heads = shape.n_v_heads,
                conv_taps = shape.conv_kernel,
                width = rows.width,
                "Qwen DFlash2 verify keeps the stock DeltaNet chain: fused DeltaNet core refused"
            );
        });
    }
    Ok(refusal.is_none())
}

/// What the fused core of one layer produced.
pub(super) struct DeltaCore {
    /// Gated rows `[width, n_v_heads * d_v]` in the requested dtype.
    pub(super) gated: MlxBuffer,
    /// Per window, the rows a multi-row window retains for the replay.
    pub(super) stages: Vec<Option<WindowStage>>,
    /// Packed `g` and `beta` rows `[width, n_v_heads]` in the replay
    /// storage; `None` for a step without it, which retains nothing.
    pub(super) gates: Option<(MlxBuffer, MlxBuffer)>,
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
    let gates = match replay.as_deref_mut() {
        Some(replay) => Some((replay.take(width * nv)?, replay.take(width * nv)?)),
        None => None,
    };

    let mut lanes = Vec::with_capacity(rows.windows.len() + padding_lanes(rows.real(), width));
    for (window, stage) in rows.windows.iter().zip(&stages) {
        let tape = match (stage, replay.as_deref(), &gates) {
            (Some(stage), Some(replay), Some((g, beta))) => Some(GdnWindowTape {
                q: replay.offset_of(&stage.q)?,
                k: replay.offset_of(&stage.k)?,
                v: replay.offset_of(&stage.v)?,
                conv_rows: replay.offset_of(&stage.conv_rows)?
                    + u32::try_from(k_minus_one * channels)?,
                g: replay.offset_of(g)? + u32::try_from(window.start * nv)?,
                beta: replay.offset_of(beta)? + u32::try_from(window.start * nv)?,
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
    static FIRST_USE: Once = Once::new();
    FIRST_USE.call_once(|| {
        tracing::info!(
            rows = rows.real(),
            width,
            lanes = rows.windows.len(),
            num_key_heads = shape.n_k_heads,
            num_value_heads = shape.n_v_heads,
            conv_taps = shape.conv_kernel,
            "Qwen DFlash2 verify selected the fused DeltaNet core"
        );
    });
    Ok(DeltaCore {
        gated,
        stages,
        gates,
    })
}
