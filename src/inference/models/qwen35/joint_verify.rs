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
//! - packed products at the verify width ([`joint_verify_padded_width`]),
//!   which keeps the spec-off kernel per row (column-invariant MMA at any
//!   `m <= 8`, the `mul_mv_ext` width-4 tile at `m` = 4 and 8, the one-row
//!   dispatch on per-row routes); the output head runs inside the same
//!   width, the canonical spec-off kernel;
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

use super::{Qwen35JointOutput, Qwen35JointRow, Qwen35JointRowOutput};
use crate::inference::models::qwen35::joint_decode_core::{
    encode_delta_net_replay, DeltaNetReplay, JointWindow,
};
use crate::inference::models::qwen35::joint_decode_routes::{
    joint_verify_padded_width, JointDenseFfnRoute, JointRouteMode, MAX_JOINT_DECODE_ROWS,
};
use crate::inference::models::qwen35::joint_step_options::JointStepOptions;
use crate::inference::models::qwen35::kv_cache::HybridKvCache;
use crate::inference::models::qwen35::model::Qwen35Model;
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

impl Qwen35JointVerifyPending {
    /// `(slot, window rows)` per lane, in the verify step's lane order.
    pub(crate) fn lanes(&self) -> impl Iterator<Item = (SlotId, usize)> + '_ {
        self.lanes.iter().map(|lane| (lane.slot, lane.rows))
    }
}

impl Qwen35Model {
    /// Run one joint verify step over `lanes` (distinct slots, `1 + k_i`
    /// rows each, at most [`MAX_JOINT_DECODE_ROWS`] rows in total). Every
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
        ensure!(!lanes.is_empty(), "Qwen joint verify requires a lane");
        let total = lanes.iter().map(|lane| lane.tokens.len()).sum::<usize>();
        ensure!(
            (1..=MAX_JOINT_DECODE_ROWS).contains(&total),
            "Qwen joint verify requires 1..={MAX_JOINT_DECODE_ROWS} rows, got {total}"
        );
        let mut rows = Vec::with_capacity(total);
        let mut windows = Vec::with_capacity(lanes.len());
        let mut pending = Vec::with_capacity(lanes.len());
        for (index, lane) in lanes.iter().enumerate() {
            let slot = lane.slot_id;
            ensure!(
                slot.0 < kv_cache.n_seqs,
                "Qwen joint verify slot {} outside n_seqs={}",
                slot.0,
                kv_cache.n_seqs
            );
            ensure!(
                lanes[..index].iter().all(|earlier| earlier.slot_id != slot),
                "Qwen joint verify slot {} appears twice",
                slot.0
            );
            ensure!(
                !lane.tokens.is_empty(),
                "Qwen joint verify lane {} has no rows",
                slot.0
            );
            let position = u32::try_from(lane.position)
                .context("Qwen joint verify position must be non-negative")?;
            kv_cache
                .validate_sequence_len_for_slot(slot, position as usize)
                .with_context(|| {
                    format!(
                        "Qwen joint verify lane {} must start at its committed cursor",
                        slot.0
                    )
                })?;
            ensure!(
                position as usize + lane.tokens.len() <= kv_cache.max_seq_len as usize,
                "Qwen joint verify lane {} window {}..{} exceeds max_seq_len {}",
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
        let packing = super::joint_decode::JointPacking {
            rows: &rows,
            windows: &windows,
            width: joint_verify_padded_width(mode, total),
        };
        let linear_layers = kv_cache.linear_attn.len();
        let (outputs, replay) = self.with_joint_forward(
            mode,
            |device, registry, layer_weights, output_head, plan| {
                let mut replay = DeltaNetReplay::new(
                    device,
                    &self.cfg,
                    &windows,
                    packing.width,
                    linear_layers,
                )?;
                let outputs = self.encode_joint_decode(
                    device,
                    registry,
                    layer_weights,
                    output_head,
                    plan,
                    &packing,
                    kv_cache,
                    dense_ffn,
                    options,
                    replay.as_mut(),
                )?;
                Ok((outputs, replay))
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
            "Qwen joint verify returned fewer rows than it packed"
        );
        Ok(Qwen35JointVerify {
            rows,
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
                    &mut enc,
                    device,
                    registry,
                    cache,
                    replay,
                    &partial,
                    &self.cfg,
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
