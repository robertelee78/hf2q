//! Joint MTP verify windows over independent Qwen3.5/3.6/3.8 slots (H1) and
//! their replay commit (H2).
//!
//! One verify step carries several lanes. Lane `i` contributes a window of
//! `1 + k_i` rows: its pending committed token followed by `k_i` drafts, at
//! consecutive positions starting at the slot's committed cursor. The rows
//! run through the joint decode graph ([`Qwen35Model::encode_joint_decode`])
//! so every op a row sees is the op the spec-off joint step runs for the
//! same slot at the same position:
//!
//! - packed products at the verify width: column-invariant MMA uses the
//!   exact real-row count at `m <= 8`; canonical products use
//!   [`joint_verify_padded_width`]. The output head stays inside that width,
//!   preserving the spec-off kernel for every row;
//! - DeltaNet: one `ssm_conv` and one gated delta net dispatch per lane
//!   with `n_tokens` = the window's rows, reading the current parity and
//!   writing the opposite one (bitwise the sequential one-token steps);
//! - full attention: each row runs the spec-off single-row route itself,
//!   TQ write at its own position then SDPA over `position + 1` rows.
//!
//! A row's bytes therefore depend only on its lane's state and its own
//! input, never on the other lanes, their windows or the packing order.
//!
//! The step leaves each lane with its window's final DeltaNet state current
//! and its pre-window state intact in the scratch parity, and its attention
//! cursor after the window. [`Qwen35Model::commit_joint_verify`] then keeps
//! `accepted` rows per lane: all rows (no work), none (flip the parity back),
//! or a prefix (replay the prefix from the pre-window state; conv state by
//! byte copies), and truncates the attention cursor; positions past it hold
//! draft K/V that the next writes overwrite before any read.
//!
//! The caller's spec-off reference is a joint decode step with the same
//! [`JointRouteMode`], [`JointDenseFfnRoute`] and [`JointStepOptions`].

use anyhow::{anyhow, ensure, Context, Result};
use mlx_native::MlxBuffer;

use super::joint_decode::JointOutputHeadTail;
use super::{Qwen35JointOutput, Qwen35JointRow, Qwen35JointRowOutput};
use crate::inference::models::qwen35::joint_decode_core::{
    encode_delta_net_replay, rows_view, DeltaNetReplay, JointWindow,
};
use crate::inference::models::qwen35::joint_decode_routes::{
    joint_verify_padded_width, JointDenseFfnRoute, JointRouteMode, DFLASH2_VERIFY_WIDTH,
    MAX_JOINT_DECODE_ROWS, MAX_JOINT_VERIFY_ROWS,
};
use crate::inference::models::qwen35::joint_projection::JointProjectionKernel;
use crate::inference::models::qwen35::joint_step_options::JointStepOptions;
use crate::inference::models::qwen35::kv_cache::HybridKvCache;
use crate::inference::models::qwen35::model::Qwen35Model;
use crate::inference::spec_decode::dflash2::adapters::DFlash2TargetCapture;
use crate::serve::multi_seq_kv::SlotId;

/// One lane of a joint verify step.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen35JointVerifyLane<'a> {
    pub slot_id: SlotId,
    /// Position of `tokens[0]`: the slot's committed cursor.
    pub position: i32,
    /// The pending committed token, then the drafts.
    pub tokens: &'a [u32],
    /// What every row of the window returns.
    pub output: Qwen35JointOutput,
}

/// Result of a joint verify step.
pub(crate) struct Qwen35JointVerify {
    /// Per lane in input order, one output per window row.
    pub rows: Vec<Vec<Qwen35JointRowOutput>>,
    /// Per lane, its window's `[rows, hidden]` post-output-norm rows (a
    /// view of one durable buffer): the MTP catch-up's target hidden.
    pub hidden: Vec<MlxBuffer>,
    /// The state the commit settles.
    pub pending: Qwen35JointVerifyPending,
}

/// What [`Qwen35Model::commit_joint_verify`] needs from its verify step.
/// Until the commit runs, every lane's slot holds its whole window
/// (DeltaNet parity flipped, cursor after the window); dropping it
/// unsettled leaves that state, so it must be committed before the lanes'
/// next forward.
#[must_use = "a verify step must be committed before its lanes run again"]
pub(crate) struct Qwen35JointVerifyPending {
    lanes: Vec<PendingLane>,
    replay: Option<DeltaNetReplay>,
}

struct PendingLane {
    slot: SlotId,
    position: u32,
    rows: usize,
}

#[cfg(test)]
impl Qwen35JointVerifyPending {
    /// `(slot, window rows)` per lane, in the verify step's lane order.
    pub(crate) fn lanes(&self) -> impl Iterator<Item = (SlotId, usize)> + '_ {
        self.lanes.iter().map(|lane| (lane.slot, lane.rows))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VerifyLayout {
    /// Existing MTP diagnostic: preserve the caller's joint-decode route.
    JointMtp,
    /// DFlash2 target: dense targets fill the width-32 tile; MoE targets use
    /// the smallest admitted tile so padding does not trigger expert work.
    DFlash2,
}

impl VerifyLayout {
    fn max_rows(self) -> usize {
        match self {
            Self::JointMtp => MAX_JOINT_VERIFY_ROWS,
            Self::DFlash2 => DFLASH2_VERIFY_WIDTH,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::JointMtp => "joint MTP",
            Self::DFlash2 => "DFlash2",
        }
    }
}

fn dflash2_verify_width(is_moe: bool, total: usize) -> usize {
    if is_moe {
        total.next_power_of_two().clamp(4, DFLASH2_VERIFY_WIDTH)
    } else {
        DFLASH2_VERIFY_WIDTH
    }
}

impl Qwen35Model {
    /// Run one joint MTP verify step over `lanes` (distinct slots, `1 + k_i`
    /// rows each, at most [`MAX_JOINT_VERIFY_ROWS`] rows in total). Every
    /// row's logits, hidden and argmax are bitwise those of the joint
    /// decode step under the same `mode`, `dense_ffn` and `options` for that
    /// slot at that position with the same preceding tokens. Refusal
    /// happens before any cache write; a failure after it leaves every lane
    /// needing the caller's per-slot recovery.
    pub(crate) fn forward_gpu_joint_verify(
        &self,
        lanes: &[Qwen35JointVerifyLane<'_>],
        kv_cache: &mut HybridKvCache,
        mode: JointRouteMode,
        dense_ffn: JointDenseFfnRoute,
        options: JointStepOptions,
    ) -> Result<Qwen35JointVerify> {
        self.forward_gpu_joint_verify_layout(
            lanes,
            kv_cache,
            mode,
            dense_ffn,
            options,
            VerifyLayout::JointMtp,
            None,
            None,
            None,
        )
    }

    /// Run the DFlash2 target verify as one fixed-width temporal tile.
    ///
    /// The tile is always [`DFLASH2_VERIFY_WIDTH`] rows, so a lane's bytes
    /// do not depend on the number or lengths of its cohort's windows.
    /// Quantized projections at this width use the canonical matrix-matrix
    /// route and stream each dense target weight once. This is a Q1 Class B
    /// alternate route against one-token decode: its decision envelope and
    /// argmax equality are qualification gates, not assumptions.
    #[allow(
        dead_code,
        reason = "DFlash2 scheduler wiring follows the target-capture gate"
    )]
    pub(crate) fn forward_gpu_dflash2_verify(
        &self,
        lanes: &[Qwen35JointVerifyLane<'_>],
        kv_cache: &mut HybridKvCache,
        mode: JointRouteMode,
        dense_ffn: JointDenseFfnRoute,
        options: JointStepOptions,
        capture: Option<DFlash2TargetCapture<'_>>,
    ) -> Result<Qwen35JointVerify> {
        self.forward_gpu_joint_verify_layout(
            lanes,
            kv_cache,
            mode,
            dense_ffn,
            options,
            VerifyLayout::DFlash2,
            None,
            capture,
            None,
        )
    }

    /// Run DFlash2 verification from the GPU-composed fixed-width token tile.
    /// Host lane metadata still owns positions and cache windows; token values
    /// remain on the command queue until verification completes.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_gpu_dflash2_verify_from_gpu_tokens(
        &self,
        lanes: &[Qwen35JointVerifyLane<'_>],
        target_tokens: &MlxBuffer,
        kv_cache: &mut HybridKvCache,
        mode: JointRouteMode,
        dense_ffn: JointDenseFfnRoute,
        options: JointStepOptions,
        capture: Option<DFlash2TargetCapture<'_>>,
        output_tail: Option<&mut JointOutputHeadTail<'_>>,
    ) -> Result<Qwen35JointVerify> {
        self.forward_gpu_joint_verify_layout(
            lanes,
            kv_cache,
            mode,
            dense_ffn,
            options,
            VerifyLayout::DFlash2,
            Some(target_tokens),
            capture,
            output_tail,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_gpu_joint_verify_layout(
        &self,
        lanes: &[Qwen35JointVerifyLane<'_>],
        kv_cache: &mut HybridKvCache,
        mode: JointRouteMode,
        dense_ffn: JointDenseFfnRoute,
        options: JointStepOptions,
        layout: VerifyLayout,
        gpu_token_ids: Option<&MlxBuffer>,
        capture: Option<DFlash2TargetCapture<'_>>,
        output_tail: Option<&mut JointOutputHeadTail<'_>>,
    ) -> Result<Qwen35JointVerify> {
        ensure!(
            !lanes.is_empty(),
            "Qwen {} verify requires a lane",
            layout.label()
        );
        if layout == VerifyLayout::DFlash2 {
            ensure!(
                lanes.len() <= 4,
                "Qwen DFlash2 verify accepts at most four lanes, got {}",
                lanes.len()
            );
            ensure!(
                options.projection == JointProjectionKernel::Canonical,
                "Qwen DFlash2 verify requires the canonical width-32 projection route"
            );
        } else {
            ensure!(
                !options.verify_attention_tile,
                "Qwen joint MTP verify rows are the joint decode step's; the tiled verify \
                 attention is a DFlash2 route"
            );
        }
        let total = lanes.iter().map(|lane| lane.tokens.len()).sum::<usize>();
        let max_rows = layout.max_rows();
        ensure!(
            (1..=max_rows).contains(&total),
            "Qwen {} verify requires 1..={max_rows} rows, got {total}",
            layout.label()
        );
        let mut rows = Vec::with_capacity(total);
        let mut windows = Vec::with_capacity(lanes.len());
        let mut pending = Vec::with_capacity(lanes.len());
        for (index, lane) in lanes.iter().enumerate() {
            let slot = lane.slot_id;
            ensure!(
                slot.0 < kv_cache.n_seqs,
                "Qwen {} verify slot {} outside n_seqs={}",
                layout.label(),
                slot.0,
                kv_cache.n_seqs
            );
            ensure!(
                lanes[..index].iter().all(|earlier| earlier.slot_id != slot),
                "Qwen {} verify slot {} appears twice",
                layout.label(),
                slot.0
            );
            ensure!(
                !lane.tokens.is_empty(),
                "Qwen {} verify lane {} has no rows",
                layout.label(),
                slot.0
            );
            if layout == VerifyLayout::DFlash2 {
                ensure!(
                    lane.tokens.len() <= 8,
                    "Qwen DFlash2 verify lane {} has {} rows; maximum is eight",
                    slot.0,
                    lane.tokens.len()
                );
            }
            let position = u32::try_from(lane.position).with_context(|| {
                format!(
                    "Qwen {} verify position must be non-negative",
                    layout.label()
                )
            })?;
            kv_cache
                .validate_sequence_len_for_slot(slot, position as usize)
                .with_context(|| {
                    format!(
                        "Qwen {} verify lane {} must start at its committed cursor",
                        layout.label(),
                        slot.0
                    )
                })?;
            ensure!(
                position as usize + lane.tokens.len() <= kv_cache.max_seq_len as usize,
                "Qwen {} verify lane {} window {}..{} exceeds max_seq_len {}",
                layout.label(),
                slot.0,
                position,
                position as usize + lane.tokens.len(),
                kv_cache.max_seq_len
            );
            windows.push(JointWindow {
                slot,
                start: rows.len(),
                rows: lane.tokens.len(),
            });
            pending.push(PendingLane {
                slot,
                position,
                rows: lane.tokens.len(),
            });
            for (offset, &token) in lane.tokens.iter().enumerate() {
                rows.push(Qwen35JointRow {
                    slot_id: slot,
                    token,
                    position: lane.position + offset as i32,
                    output: lane.output,
                });
            }
        }
        self.check_joint_step_admission(kv_cache, mode, dense_ffn, options)?;
        let width = match layout {
            VerifyLayout::DFlash2 => dflash2_verify_width(self.cfg.moe.is_some(), total),
            VerifyLayout::JointMtp
                if options.projection == JointProjectionKernel::ColumnInvariantMma =>
            {
                ensure!(
                    total <= MAX_JOINT_DECODE_ROWS,
                    "Qwen joint verify column-invariant MMA supports at most \
                     {MAX_JOINT_DECODE_ROWS} rows, got {total}"
                );
                total
            }
            VerifyLayout::JointMtp => joint_verify_padded_width(mode, total),
        };
        let packing = super::joint_decode::JointPacking {
            rows: &rows,
            windows: &windows,
            width,
            attention_tile: options.verify_attention_tile,
        };
        let linear_layers = kv_cache.linear_attn.len();
        let (outputs, normed, replay) = self.with_joint_forward(
            mode,
            |device, registry, layer_weights, output_head, plan| {
                let mut replay =
                    DeltaNetReplay::new(device, &self.cfg, &windows, packing.width, linear_layers)?;
                let (outputs, normed) = self.encode_joint_decode(
                    device,
                    registry,
                    layer_weights,
                    output_head,
                    plan,
                    &packing,
                    kv_cache,
                    dense_ffn,
                    options,
                    gpu_token_ids,
                    replay.as_mut(),
                    capture,
                    output_tail,
                )?;
                Ok((outputs, normed, replay))
            },
        )?;
        let mut outputs = outputs.into_iter();
        let rows = windows
            .iter()
            .map(|window| outputs.by_ref().take(window.rows).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        ensure!(
            rows.iter()
                .zip(&windows)
                .all(|(lane, window)| lane.len() == window.rows),
            "Qwen {} verify returned fewer rows than it packed",
            layout.label()
        );
        let hidden_size = self.cfg.hidden_size as usize;
        let hidden = windows
            .iter()
            .map(|window| rows_view(&normed, window.start, window.rows, hidden_size))
            .collect::<Result<Vec<_>>>()?;
        Ok(Qwen35JointVerify {
            rows,
            hidden,
            pending: Qwen35JointVerifyPending {
                lanes: pending,
                replay,
            },
        })
    }

    /// Settle a verify step: lane `i` keeps its first `accepted[i]` rows
    /// (`0..=rows`; `accepted` follows the verify step's lane order). The
    /// lane's DeltaNet state and attention cursor become those of the
    /// spec-off steps over exactly those rows:
    ///
    /// - all rows: nothing to do;
    /// - none: the pre-window DeltaNet state is still intact in the scratch
    ///   parity, so the parity flips back;
    /// - a prefix: the DeltaNet state is replayed over the prefix from the
    ///   pre-window state (one command buffer for every such lane).
    ///
    /// The attention cursor is truncated to `position + accepted`. On error
    /// the caller must reset every lane of the step.
    pub(crate) fn commit_joint_verify(
        &self,
        kv_cache: &mut HybridKvCache,
        pending: Qwen35JointVerifyPending,
        accepted: &[usize],
    ) -> Result<()> {
        ensure!(
            accepted.len() == pending.lanes.len(),
            "Qwen joint verify commit got {} accept counts for {} lanes",
            accepted.len(),
            pending.lanes.len()
        );
        for (lane, &rows) in pending.lanes.iter().zip(accepted) {
            ensure!(
                rows <= lane.rows,
                "Qwen joint verify lane {} accepts {rows} of {} rows",
                lane.slot.0,
                lane.rows
            );
            kv_cache
                .validate_sequence_len_for_slot(lane.slot, lane.position as usize + lane.rows)
                .with_context(|| {
                    format!(
                        "Qwen joint verify lane {} cursor moved before its commit",
                        lane.slot.0
                    )
                })?;
        }
        let partial = pending
            .lanes
            .iter()
            .zip(accepted)
            .filter(|(lane, &rows)| (1..lane.rows).contains(&rows))
            .map(|(lane, &rows)| (lane.slot, rows))
            .collect::<Vec<_>>();
        if !partial.is_empty() {
            let replay = pending
                .replay
                .as_ref()
                .context("Qwen joint verify partial commit without replay storage")?;
            let cache: &HybridKvCache = kv_cache;
            self.with_gpu_cache_mut(|device, registry| {
                let mut enc = device
                    .command_encoder()
                    .map_err(|e| anyhow!("Qwen joint verify replay encoder: {e}"))?;
                encode_delta_net_replay(
                    &mut enc, device, registry, cache, replay, &partial, &self.cfg,
                )?;
                enc.commit_and_wait_labeled("qwen35.joint_verify.replay")
                    .map_err(|e| anyhow!("Qwen joint verify replay command buffer: {e}"))
            })?;
        }
        for (lane, &rows) in pending.lanes.iter().zip(accepted) {
            if rows == 0 {
                for layer in kv_cache.linear_attn.iter_mut() {
                    layer.swap_for_slot(lane.slot);
                }
            }
            if rows < lane.rows {
                kv_cache
                    .truncate_full_attn_to_for_slot(lane.slot, lane.position + rows as u32)
                    .with_context(|| {
                        format!("Qwen joint verify lane {} cursor truncate", lane.slot.0)
                    })?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dflash2_moe_width_avoids_unverified_padding_expert_work() {
        for (rows, width) in [(1, 4), (4, 4), (5, 8), (8, 8), (9, 16), (17, 32)] {
            assert_eq!(dflash2_verify_width(true, rows), width);
        }
        assert_eq!(dflash2_verify_width(false, 1), DFLASH2_VERIFY_WIDTH);
        assert_eq!(dflash2_verify_width(false, 17), DFLASH2_VERIFY_WIDTH);
    }
}
