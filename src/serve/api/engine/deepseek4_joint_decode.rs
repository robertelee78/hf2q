//! DeepSeek-V4 joint decode scheduling.
//!
//! One scheduler decode turn advances every eligible decoding slot with one
//! joint forward per token step instead of a per-slot serial quantum. Sampling,
//! grammar/tool-call constraints, stop strings, streaming, and cancellation
//! remain per slot exactly as in the serial path; only cache advancement is
//! shared. Ineligible handles (parked completions, closed clients, poisoned or
//! exhausted caches) stay on the serial loop, and the scheduler's mixed-work
//! budget still interleaves bounded prefill chunks between decode turns.

use std::sync::atomic::Ordering;
use std::sync::{LazyLock, Once};

use crate::inference::models::deepseek4::{joint_decode_available, MAX_JOINT_DECODE_ROWS};
use crate::serve::api::engine_deepseek4::{
    Deepseek4LoadedModel, Deepseek4Session, Deepseek4SlotState,
};
use crate::serve::api::registry::ModelRegistration;

use super::{
    deepseek4_effective_decode_quantum, is_worker_fatal, settle_deepseek4_decode_lane,
    Deepseek4DecodeLaneOutcome, Deepseek4Slot, Deepseek4SlotWork, EngineSupervisor,
    InflightBatchedScheduler, SlotAwareGpuFatal, SlotHandle, SlotReply,
};

/// Kill switch for A/B measurement: `1`/`on`/`true` selects joint decode,
/// `0`/`off`/`false` keeps the serial quantum (and the exact B=4 cohort).
const DEEPSEEK4_JOINT_DECODE_ENV: &str = "HF2Q_DEEPSEEK_JOINT_DECODE";
/// Joint decode is the default: it is bitwise equal to serial decode on the
/// official artifact (`official_artifact_joint_decode_matches_serial_decode_bitwise`)
/// and cut four-agent C262144 coding makespan by 33.5% (4 runs per arm,
/// 2026-10-30; `docs/research/adr063-s3-ds4-joint-decode.md`).
const DEEPSEEK4_JOINT_DECODE_DEFAULT: bool = true;

fn parse_deepseek4_joint_decode(raw: Option<&str>) -> Result<bool, String> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(DEEPSEEK4_JOINT_DECODE_DEFAULT);
    };
    match raw.to_ascii_lowercase().as_str() {
        "1" | "on" | "true" => Ok(true),
        "0" | "off" | "false" => Ok(false),
        _ => Err(format!(
            "{DEEPSEEK4_JOINT_DECODE_ENV} must be 1/on/true or 0/off/false, got {raw:?}"
        )),
    }
}

/// Process-wide joint decode selection, read once at first decode turn.
static JOINT_DECODE_ENABLED: LazyLock<bool> = LazyLock::new(|| {
    let raw = std::env::var(DEEPSEEK4_JOINT_DECODE_ENV).ok();
    let requested = parse_deepseek4_joint_decode(raw.as_deref()).unwrap_or_else(|error| {
        tracing::warn!(
            error,
            fallback = DEEPSEEK4_JOINT_DECODE_DEFAULT,
            "invalid DeepSeek-V4 joint decode switch"
        );
        DEEPSEEK4_JOINT_DECODE_DEFAULT
    });
    let available = joint_decode_available();
    tracing::info!(
        target: "hf2q::serve::api::engine_deepseek4::progress",
        requested,
        available,
        max_rows = MAX_JOINT_DECODE_ROWS,
        "DeepSeek-V4 joint decode configuration"
    );
    requested && available
});

pub(super) fn deepseek4_joint_decode_enabled() -> bool {
    *JOINT_DECODE_ENABLED
}

/// Split one decode turn into joint rows and serial leftovers. The whole turn
/// stays serial unless handles are unique per slot and at least two are
/// eligible. Joint rows are sorted by slot and capped at `max_rows`; serial
/// leftovers keep scheduler order.
fn partition_deepseek4_joint_lanes(
    handles: &[SlotHandle],
    max_rows: usize,
    mut eligible: impl FnMut(SlotHandle) -> bool,
) -> Option<(Vec<SlotHandle>, Vec<SlotHandle>)> {
    let unique = handles.iter().enumerate().all(|(index, handle)| {
        handles[..index]
            .iter()
            .all(|earlier| earlier.slot_id != handle.slot_id)
    });
    if !unique || max_rows < 2 {
        return None;
    }
    let mut joint = Vec::with_capacity(handles.len().min(max_rows));
    let mut serial = Vec::new();
    for &handle in handles {
        if joint.len() < max_rows && eligible(handle) {
            joint.push(handle);
        } else {
            serial.push(handle);
        }
    }
    if joint.len() < 2 {
        return None;
    }
    joint.sort_by_key(|handle| handle.slot_id);
    Some((joint, serial))
}

fn deepseek4_joint_lane_eligible(
    sessions: &[Deepseek4Session],
    slots: &[Option<Deepseek4Slot>],
    handle: SlotHandle,
) -> bool {
    let slot_index = handle.slot_id.0 as usize;
    slots
        .get(slot_index)
        .and_then(Option::as_ref)
        .is_some_and(|(work, reply, installed)| {
            *installed == handle
                && matches!(work, Deepseek4SlotWork::Decode { state, .. } if !state.is_finished())
                && !reply.client_closed()
        })
        && sessions
            .get(slot_index)
            .is_some_and(Deepseek4Session::joint_decode_ready)
}

pub(super) enum Deepseek4JointRun {
    /// Fewer than two eligible lanes; the caller runs the whole turn serially.
    NotEligible,
    /// Joint lanes are settled; `serial` handles still need the serial loop.
    Ran {
        serial: Vec<SlotHandle>,
    },
    Fatal(SlotAwareGpuFatal),
}

struct JointLane {
    slot_index: usize,
    handle: SlotHandle,
    state: Deepseek4SlotState,
    cold_wave: bool,
    reply: SlotReply,
    completed_ticks: usize,
    tick_error: Option<anyhow::Error>,
    client_closed: bool,
}

impl JointLane {
    fn active(&self) -> bool {
        !self.state.is_finished() && self.tick_error.is_none() && !self.client_closed
    }
}

fn reinstall_joint_lanes(slots: &mut [Option<Deepseek4Slot>], lanes: Vec<JointLane>) {
    for lane in lanes {
        debug_assert!(slots[lane.slot_index].is_none());
        slots[lane.slot_index] = Some((
            Deepseek4SlotWork::Decode {
                state: lane.state,
                cold_wave: lane.cold_wave,
            },
            lane.reply,
            lane.handle,
        ));
    }
}

fn joint_lanes_fatal(lanes: Vec<JointLane>, error: anyhow::Error) -> SlotAwareGpuFatal {
    let mut owned = lanes.into_iter();
    let first = owned.next().expect("joint decode has at least two lanes");
    let mut fatal = SlotAwareGpuFatal::slot(first.handle, first.reply, error);
    fatal.extend_slots(owned.map(|lane| (Some(lane.handle), lane.reply)));
    fatal
}

/// Run one decode turn for every eligible slot through joint steps, then
/// settle each joint lane with the serial path's slot semantics.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_deepseek4_joint_decode(
    model: &mut Deepseek4LoadedModel,
    sessions: &mut [Deepseek4Session],
    slots: &mut [Option<Deepseek4Slot>],
    last_used: &mut [u64],
    clock: &mut u64,
    scheduler: &mut InflightBatchedScheduler,
    registration: Option<&ModelRegistration>,
    handles: &[SlotHandle],
    kv_bytes_per_token: u64,
    decode_quantum: usize,
    cold_runnable_at_start: usize,
    supervisor: &EngineSupervisor,
) -> Deepseek4JointRun {
    let Some((joint, serial)) =
        partition_deepseek4_joint_lanes(handles, MAX_JOINT_DECODE_ROWS, |handle| {
            deepseek4_joint_lane_eligible(sessions, slots, handle)
        })
    else {
        return Deepseek4JointRun::NotEligible;
    };

    let mut lanes = Vec::with_capacity(joint.len());
    for handle in joint {
        let slot_index = handle.slot_id.0 as usize;
        match slots[slot_index].take() {
            Some((Deepseek4SlotWork::Decode { state, cold_wave }, reply, installed))
                if installed == handle =>
            {
                lanes.push(JointLane {
                    slot_index,
                    handle,
                    state,
                    cold_wave,
                    reply,
                    completed_ticks: 0,
                    tick_error: None,
                    client_closed: false,
                });
            }
            other => {
                slots[slot_index] = other;
                reinstall_joint_lanes(slots, lanes);
                return Deepseek4JointRun::Fatal(SlotAwareGpuFatal::invariant_handle(
                    handle,
                    anyhow::anyhow!("DeepSeek-V4 joint decode lane changed before execution"),
                ));
            }
        }
    }

    static FIRST_JOINT_DECODE: Once = Once::new();
    FIRST_JOINT_DECODE.call_once(|| {
        tracing::info!(
            target: "hf2q::serve::api::engine_deepseek4::progress",
            lanes = lanes.len(),
            "DeepSeek-V4 joint decode engaged"
        );
    });
    // One shared budget keeps every row in lockstep for the whole turn; the
    // scheduler re-forms the group (and interleaves prefill) next turn.
    let budget = lanes
        .iter()
        .map(|lane| deepseek4_effective_decode_quantum(decode_quantum, lane.cold_wave))
        .min()
        .unwrap_or(1);
    tracing::debug!(
        target: "hf2q::serve::api::engine_deepseek4::progress",
        lanes = lanes.len(),
        serial = serial.len(),
        budget,
        "DeepSeek-V4 joint decode turn"
    );

    for _ in 0..budget {
        for lane in lanes.iter_mut().filter(|lane| lane.active()) {
            if let SlotReply::Stream { events, cancel, .. } = &lane.reply {
                if events.is_closed() {
                    lane.state.cancelled();
                    if let Some(counter) = cancel {
                        counter.fetch_add(1, Ordering::Relaxed);
                    }
                    lane.client_closed = true;
                }
            }
        }

        let mut pending = Vec::with_capacity(lanes.len());
        for (index, lane) in lanes.iter_mut().enumerate() {
            if !lane.active() {
                continue;
            }
            let events = match &lane.reply {
                SlotReply::Unary(_) => None,
                SlotReply::Stream { events, .. } => Some(events),
            };
            match lane
                .state
                .prepare_tick(model, registration, events, supervisor)
            {
                Ok(Some(token)) => pending.push((index, token)),
                Ok(None) => lane.completed_ticks += 1,
                Err(error) => lane.tick_error = Some(error),
            }
        }

        let _native_invocation = crate::serve::phase_trace::enabled().then(|| {
            scheduler.trace_transaction(
                "deepseek4",
                "joint_decode_input",
                &pending
                    .iter()
                    .map(|&(index, _)| lanes[index].handle)
                    .collect::<Vec<_>>(),
            )
        });
        if let [(index, token)] = pending[..] {
            // A lone survivor keeps the exact one-row decode graph.
            let lane = &mut lanes[index];
            sessions[lane.slot_index].swap_with_loaded(model);
            let committed = model.commit_generated_token(token, supervisor);
            sessions[lane.slot_index].swap_with_loaded(model);
            match committed {
                Ok(logits) => {
                    lane.state.install_next_logits(logits);
                    lane.completed_ticks += 1;
                }
                Err(error) => lane.tick_error = Some(error),
            }
        } else if pending.len() > 1 {
            let commits = pending
                .iter()
                .map(|&(index, token)| (lanes[index].slot_index, token))
                .collect::<Vec<_>>();
            match model.commit_generated_tokens_joint(&commits, sessions, supervisor) {
                Ok(logits) => {
                    for (&(index, _), logits) in pending.iter().zip(logits) {
                        lanes[index].state.install_next_logits(logits);
                        lanes[index].completed_ticks += 1;
                    }
                }
                Err(error) if is_worker_fatal(supervisor, &error) => {
                    return Deepseek4JointRun::Fatal(joint_lanes_fatal(lanes, error));
                }
                Err(error) => {
                    // Every row shared the failed transaction (a submitted
                    // failure poisons every row's cache), so each request
                    // fails exactly as its own serial commit failure would.
                    let message = format!("{error:#}");
                    let mut error = Some(error);
                    for &(index, _) in &pending {
                        lanes[index].tick_error =
                            Some(error.take().unwrap_or_else(|| anyhow::anyhow!("{message}")));
                    }
                }
            }
        }
        if !lanes.iter().any(JointLane::active) {
            break;
        }
    }

    let mut owned = lanes.into_iter();
    while let Some(lane) = owned.next() {
        if let Some(mut fatal) = settle_deepseek4_decode_lane(
            Deepseek4DecodeLaneOutcome {
                handle: lane.handle,
                reply: lane.reply,
                state: lane.state,
                cold_wave: lane.cold_wave,
                was_parked: false,
                deferred_decode_ticks: 0,
                completed_ticks: lane.completed_ticks,
                tick_error: lane.tick_error,
                client_closed: lane.client_closed,
            },
            sessions,
            slots,
            last_used,
            clock,
            scheduler,
            registration,
            kv_bytes_per_token,
            cold_runnable_at_start,
            supervisor,
        ) {
            fatal.extend_slots(owned.map(|lane| (Some(lane.handle), lane.reply)));
            return Deepseek4JointRun::Fatal(fatal);
        }
    }
    Deepseek4JointRun::Ran { serial }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::multi_seq_kv::SlotId;

    fn handle(slot: u32, generation: u64) -> SlotHandle {
        SlotHandle {
            slot_id: SlotId(slot),
            generation,
        }
    }

    #[test]
    fn joint_decode_switch_defaults_on_and_parses_explicit_arms() {
        assert_eq!(
            parse_deepseek4_joint_decode(None),
            Ok(DEEPSEEK4_JOINT_DECODE_DEFAULT)
        );
        assert_eq!(
            parse_deepseek4_joint_decode(Some("  ")),
            Ok(DEEPSEEK4_JOINT_DECODE_DEFAULT)
        );
        assert!(DEEPSEEK4_JOINT_DECODE_DEFAULT);
        for raw in ["1", "on", "TRUE"] {
            assert_eq!(parse_deepseek4_joint_decode(Some(raw)), Ok(true));
        }
        for raw in ["0", "Off", "false"] {
            assert_eq!(parse_deepseek4_joint_decode(Some(raw)), Ok(false));
        }
        assert!(parse_deepseek4_joint_decode(Some("joint")).is_err());
    }

    #[test]
    fn ragged_eligible_lanes_join_sorted_and_the_rest_stay_serial() {
        let handles = [handle(3, 9), handle(0, 1), handle(2, 4), handle(1, 7)];
        let (joint, serial) =
            partition_deepseek4_joint_lanes(&handles, MAX_JOINT_DECODE_ROWS, |handle| {
                handle.slot_id != SlotId(2)
            })
            .expect("three eligible lanes form a joint step");
        assert_eq!(joint, [handle(0, 1), handle(1, 7), handle(3, 9)]);
        assert_eq!(serial, [handle(2, 4)]);
    }

    #[test]
    fn fewer_than_two_eligible_lanes_keep_the_whole_turn_serial() {
        let handles = [handle(0, 1), handle(1, 1)];
        assert!(
            partition_deepseek4_joint_lanes(&handles, MAX_JOINT_DECODE_ROWS, |handle| {
                handle.slot_id == SlotId(0)
            })
            .is_none()
        );
        assert!(partition_deepseek4_joint_lanes(&[], MAX_JOINT_DECODE_ROWS, |_| true).is_none());
        assert!(partition_deepseek4_joint_lanes(&handles, 1, |_| true).is_none());
    }

    #[test]
    fn duplicate_slots_never_form_a_joint_step() {
        let handles = [handle(0, 1), handle(1, 1), handle(0, 2)];
        assert!(
            partition_deepseek4_joint_lanes(&handles, MAX_JOINT_DECODE_ROWS, |_| true).is_none()
        );
    }

    #[test]
    fn joint_width_is_capped_and_overflow_lanes_stay_serial() {
        let handles = [handle(4, 1), handle(3, 1), handle(2, 1), handle(1, 1)];
        let (joint, serial) =
            partition_deepseek4_joint_lanes(&handles, 3, |_| true).expect("capped joint step");
        assert_eq!(joint, [handle(2, 1), handle(3, 1), handle(4, 1)]);
        assert_eq!(serial, [handle(1, 1)]);
    }
}
