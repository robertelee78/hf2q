//! Slot-aware Qwen DFlash2 decode group.

use std::sync::Once;

use crate::inference::models::qwen35::joint_projection::JointProjectionKernel;
use crate::inference::models::qwen35::joint_step_options::JointStepOptions;
use crate::inference::models::qwen35::joint_mixer_fused;
use crate::inference::models::qwen35::verify_attn_tile;
use crate::serve::api::engine::qwen35_joint_decode::Qwen35JointDecodeConfig;
use crate::serve::api::engine_qwen35::{
    anchor_store::Qwen35AnchorStore, Qwen35DFlash2Forward, Qwen35DecodeState,
};
use crate::serve::api::registry::ModelRegistration;

use super::{
    is_fatal_command_buffer_error, settle_qwen35_decode_lane, EngineSupervisor,
    InflightBatchedScheduler, PromptCache, Qwen35FatalFailure, Qwen35FatalReply, Qwen35KvGuard,
    Qwen35RetainedPrefix, Qwen35Slot, Qwen35SlotWork, Qwen35WorkerFatalKind, SlotHandle, SlotReply,
};

pub(super) enum Qwen35DFlash2Run {
    NotEligible,
    Ran { serial: Vec<SlotHandle> },
    Fatal(Qwen35FatalFailure),
}

struct Lane {
    handle: SlotHandle,
    reply: SlotReply,
    state: Qwen35DecodeState,
}

fn reinstall(slots: &mut [Option<Qwen35Slot>], lanes: impl IntoIterator<Item = Lane>) {
    for lane in lanes {
        let index = lane.handle.slot_id.0 as usize;
        debug_assert!(slots[index].is_none());
        slots[index] = Some((Qwen35SlotWork::Decode(lane.state), lane.reply, lane.handle));
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_qwen35_dflash2(
    guard: &mut Qwen35KvGuard<'_>,
    scheduler: &mut InflightBatchedScheduler,
    slots: &mut [Option<Qwen35Slot>],
    retained_tokens: &mut [Qwen35RetainedPrefix],
    prompt_caches: &mut [PromptCache],
    prompt_anchors: &mut [Qwen35AnchorStore],
    registration: Option<&ModelRegistration>,
    handles: &[SlotHandle],
    kv_bytes_per_token: u64,
    config: Option<Qwen35JointDecodeConfig>,
    supervisor: &EngineSupervisor,
) -> Qwen35DFlash2Run {
    let Some(config) = config else {
        return Qwen35DFlash2Run::NotEligible;
    };
    let Some(runtime) = guard.model.dflash2.as_ref() else {
        return Qwen35DFlash2Run::NotEligible;
    };
    if guard
        .kv
        .as_ref()
        .expect("kv Some during loop")
        .la_capture_active()
    {
        return Qwen35DFlash2Run::NotEligible;
    }
    let lane_lengths = runtime.lane_lengths().to_vec();
    let kv = guard.kv.as_ref().expect("kv Some during loop");
    let mut selected = Vec::new();
    let mut serial = Vec::new();
    for &handle in handles {
        let eligible = slots
            .get(handle.slot_id.0 as usize)
            .and_then(Option::as_ref)
            .is_some_and(|(work, reply, installed)| {
                let Qwen35SlotWork::Decode(state) = work else {
                    return false;
                };
                if *installed != handle || reply.client_closed() {
                    return false;
                }
                let slot = handle.slot_id.0 as usize;
                state.joint_decode_readiness(kv).is_ok()
                    && lane_lengths.get(slot).is_some_and(|&draft_cursor| {
                        kv.sequence_len_for_slot(handle.slot_id)
                            .is_ok_and(|target_cursor| draft_cursor == target_cursor as usize)
                    })
            });
        if eligible {
            selected.push(handle);
        } else {
            serial.push(handle);
        }
    }
    if selected.is_empty() {
        return Qwen35DFlash2Run::NotEligible;
    }
    selected.sort_by_key(|handle| handle.slot_id);

    let mut lanes = Vec::with_capacity(selected.len());
    for handle in selected {
        let index = handle.slot_id.0 as usize;
        match slots[index].take() {
            Some((Qwen35SlotWork::Decode(state), reply, installed)) if installed == handle => {
                lanes.push(Lane {
                    handle,
                    reply,
                    state,
                });
            }
            other => {
                slots[index] = other;
                reinstall(slots, lanes);
                return Qwen35DFlash2Run::Fatal(Qwen35FatalFailure {
                    handle: Some(handle),
                    reply: Qwen35FatalReply::None,
                    error: anyhow::anyhow!("Qwen DFlash2 lane changed before execution"),
                    kind: Qwen35WorkerFatalKind::Invariant,
                });
            }
        }
    }

    static FIRST_DFLASH2: Once = Once::new();
    FIRST_DFLASH2.call_once(|| {
        tracing::info!(
            target: "hf2q::serve::api::engine_qwen35::progress",
            lanes = lanes.len(),
            checkpoint = guard
                .model
                .dflash2
                .as_ref()
                .map_or("unknown", |runtime| runtime.descriptor.id),
            "Qwen DFlash2 decode engaged"
        );
    });
    let step = {
        let _native_invocation = scheduler.trace_transaction(
            "qwen35",
            "dflash2_step",
            &lanes.iter().map(|lane| lane.handle).collect::<Vec<_>>(),
        );
        let mut states = lanes
            .iter_mut()
            .map(|lane| &mut lane.state)
            .collect::<Vec<_>>();
        guard.model.run_dflash2_step(
            guard.kv.as_mut().expect("kv Some during loop"),
            &mut states,
            Qwen35DFlash2Forward {
                mode: config.mode,
                dense_ffn: config.dense_ffn,
                options: JointStepOptions {
                    projection: JointProjectionKernel::Canonical,
                    pooled_outputs: config.step.pooled_outputs,
                    verify_attention_tile: verify_attn_tile::enabled(),
                    verify_mixer_fused: joint_mixer_fused::enabled(),
                },
            },
            supervisor,
            &|error| !supervisor.is_healthy() || is_fatal_command_buffer_error(error),
        )
    };
    let ticks = match step {
        Ok(ticks) => ticks,
        Err(error) => {
            let mut owned = lanes.into_iter();
            let first = owned.next().expect("a DFlash2 step has at least one lane");
            reinstall(slots, owned);
            return Qwen35DFlash2Run::Fatal(Qwen35FatalFailure {
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
            reinstall(slots, owned.map(|(lane, _)| lane));
            return Qwen35DFlash2Run::Fatal(fatal);
        }
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
                reinstall(slots, owned.map(|(lane, _)| lane));
                return Qwen35DFlash2Run::Fatal(fatal);
            }
        }
    }
    Qwen35DFlash2Run::Ran { serial }
}
