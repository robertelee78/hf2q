//! Joint MTP verify turn (`HF2Q_QWEN_JOINT_MTP`, design H4).
//!
//! One group (sorted by slot, at most `max_rows` lanes) runs one joint MTP
//! step (`run_joint_mtp_step`: roles, cap8 depths, batched drafting, one
//! joint verify forward, spec-off decisions per row, shared commit, MTP
//! catch-up). Then every lane settles through `settle_qwen35_decode_lane`,
//! and a window lane's remaining verified tokens are drained in the same
//! turn, one settlement per token: each token streams, counts and is
//! checked exactly as one spec-off tick, and lanes stay in step (the
//! design's in-turn drain; a one-token-per-turn drain would desynchronise
//! them). Lanes that finish inside a run leave the group; new requests
//! prefill between turns, as with spec-off joint decode.

use std::sync::Once;

use crate::serve::api::engine_qwen35::{run_joint_mtp_step, Qwen35JointMtpForward};
use crate::serve::api::registry::ModelRegistration;

use super::{
    is_fatal_command_buffer_error, reinstall_lanes, settle_qwen35_decode_lane, take_joint_lanes,
    EngineSupervisor, InflightBatchedScheduler, PromptCache, Qwen35FatalFailure, Qwen35FatalReply,
    Qwen35JointDecodeConfig, Qwen35KvGuard, Qwen35PromptAnchor, Qwen35RetainedPrefix, Qwen35Slot,
    Qwen35SlotWork, Qwen35WorkerFatalKind, SlotHandle,
};

/// One joint MTP step over `group`, then settle and drain every lane.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_qwen35_joint_mtp_group(
    guard: &mut Qwen35KvGuard<'_>,
    scheduler: &mut InflightBatchedScheduler,
    slots: &mut [Option<Qwen35Slot>],
    retained_tokens: &mut [Qwen35RetainedPrefix],
    prompt_caches: &mut [PromptCache],
    prompt_anchors: &mut [Option<Qwen35PromptAnchor>],
    registration: Option<&ModelRegistration>,
    group: Vec<SlotHandle>,
    serial_lanes: usize,
    turn_lanes: usize,
    kv_bytes_per_token: u64,
    config: Qwen35JointDecodeConfig,
    supervisor: &EngineSupervisor,
) -> Option<Qwen35FatalFailure> {
    let mut lanes = match take_joint_lanes(slots, group) {
        Ok(lanes) => lanes,
        Err(fatal) => return Some(*fatal),
    };
    for lane in &mut lanes {
        // The joint step serves the lane's tick; drop its plan.
        lane.state.consume_planned_tick();
    }
    static FIRST_JOINT_MTP: Once = Once::new();
    FIRST_JOINT_MTP.call_once(|| {
        tracing::info!(
            target: "hf2q::serve::api::engine_qwen35::progress",
            lanes = lanes.len(),
            mode = config.mode.label(),
            dense_ffn = config.dense_ffn.label(),
            "Qwen joint MTP verify engaged"
        );
    });
    tracing::debug!(
        target: "hf2q::serve::api::engine_qwen35::progress",
        lanes = lanes.len(),
        serial = serial_lanes,
        turn_lanes,
        "Qwen joint MTP turn"
    );

    let step = {
        let _native_invocation = scheduler.trace_transaction(
            "qwen35",
            "joint_mtp_step",
            &lanes.iter().map(|lane| lane.handle).collect::<Vec<_>>(),
        );
        let mut states = lanes
            .iter_mut()
            .map(|lane| &mut lane.state)
            .collect::<Vec<_>>();
        run_joint_mtp_step(
            guard.model,
            guard.kv.as_mut().expect("kv Some during loop"),
            &mut states,
            Qwen35JointMtpForward {
                mode: config.mode,
                dense_ffn: config.dense_ffn,
                options: config.step,
            },
            supervisor,
            &mut |qwen, kv, requests| qwen.draft_joint_mtp(kv, requests, supervisor),
            &|error| !supervisor.is_healthy() || is_fatal_command_buffer_error(error),
            None,
            None,
        )
    };
    let ticks = match step {
        Ok(ticks) => ticks,
        Err(error) => {
            let mut owned = lanes.into_iter();
            let first = owned.next().expect("a joint step has at least one lane");
            reinstall_lanes(slots, owned);
            return Some(Qwen35FatalFailure {
                handle: Some(first.handle),
                reply: Qwen35FatalReply::Slot(first.reply),
                error,
                kind: Qwen35WorkerFatalKind::Gpu,
            });
        }
    };

    let mut owned = lanes.into_iter().zip(ticks);
    while let Some((lane, tick)) = owned.next() {
        let handle = lane.handle;
        if let Some(fatal) = settle_qwen35_decode_lane(
            guard,
            scheduler,
            slots,
            retained_tokens,
            prompt_caches,
            prompt_anchors,
            registration,
            handle,
            lane.reply,
            lane.state,
            tick,
            kv_bytes_per_token,
            supervisor,
        ) {
            reinstall_lanes(slots, owned.map(|(lane, _)| lane));
            return Some(fatal);
        }
        // The lane was reinstalled unless it finished, failed or was
        // cancelled; drain it while it still holds verified output.
        let index = handle.slot_id.0 as usize;
        while matches!(
            &slots[index],
            Some((Qwen35SlotWork::Decode(state), _, installed))
                if *installed == handle && state.has_pending_speculation_output()
        ) {
            let Some((Qwen35SlotWork::Decode(mut state), reply, installed)) = slots[index].take()
            else {
                unreachable!("checked above");
            };
            let tick = state.joint_mtp_drain_tick(guard.model);
            if let Some(fatal) = settle_qwen35_decode_lane(
                guard,
                scheduler,
                slots,
                retained_tokens,
                prompt_caches,
                prompt_anchors,
                registration,
                installed,
                reply,
                state,
                tick,
                kv_bytes_per_token,
                supervisor,
            ) {
                reinstall_lanes(slots, owned.map(|(lane, _)| lane));
                return Some(fatal);
            }
        }
    }
    None
}
