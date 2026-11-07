//! Qwen joint B-row decode scheduling (ADR-063 S3, phase 1).
//!
//! One scheduler decode turn advances every eligible decoding slot with one
//! shared target forward instead of one forward per slot. Sampling, grammar
//! and tool-call constraints, thinking budgets, stop strings, streaming,
//! cancellation, retained-prefix publication and failure recovery stay per
//! slot: each joint lane settles through `settle_qwen35_decode_lane`, the
//! same tail the serial loop uses. Ineligible handles (closed clients, CPU-only
//! verified output, cursor/ledger mismatch) stay on the serial loop.
//!
//! Exact mode: a turn with fewer than two eligible lanes is serial, including
//! MTP K3 and history speculation, and lanes beyond `max_rows` stay serial;
//! both are bitwise equal to the joint row. Padded mode (#717): every eligible
//! lane runs in consecutive joint groups of at most `max_rows`, a lone
//! ordinary lane included, so an ordinary request's arithmetic does not depend
//! on which tenants share its turns; only a lone lane that would speculate
//! stays serial.

use std::sync::Once;

use crate::inference::models::qwen35::joint_decode_routes::{
    check_mode_environment, joint_decode_env_unavailable_reason, joint_ffn_shared_from_process,
    joint_mode_max_rows, select_dense_ffn_route, JointDenseFfnRoute, JointRouteMode,
    JointRoutingEnvironment, MAX_JOINT_DECODE_ROWS,
};
use crate::inference::models::qwen35::joint_projection::{
    select_joint_projection_kernel, JointProjectionKernel, QWEN_JOINT_MMA_ENV,
};
use crate::inference::models::qwen35::joint_step_options::{
    switch_from_process, JointStepOptions, QWEN_JOINT_POOLED_OUTPUTS_ENV,
};
use crate::inference::models::qwen35::model::Qwen35Model;
use crate::serve::api::engine_qwen35::{Qwen35JointLane, Qwen35TickOutcome};
use crate::serve::api::qwen35_speculation::QwenSpeculationRoute;
use crate::serve::api::registry::ModelRegistration;

use super::{
    is_fatal_command_buffer_error, settle_qwen35_decode_lane, EngineSupervisor,
    InflightBatchedScheduler, PromptCache, Qwen35FatalFailure, Qwen35FatalReply, Qwen35KvGuard,
    Qwen35PromptAnchor, Qwen35RetainedPrefix, Qwen35Slot, Qwen35SlotWork, Qwen35WorkerFatalKind,
    SlotHandle,
};

/// Kill switch for A/B measurement: `1`/`on`/`true` selects joint decode,
/// `0`/`off`/`false` keeps the serial per-slot turn. Unset selects joint
/// decode on every Qwen artifact that has a proven joint route:
/// - dense (Qwen3.8): the official-artifact gates passed and the four-agent
///   ABBA A/B cut median makespan 116.4 → 90.1 s with 8/8 `CalibrationPass`;
/// - MoE (Qwen3.6): on the declared repair-turn workload (ADR-063
///   Q4-QWEN36) the paired 24-instance task campaign is `Pass` against serial
///   decode, and the six-block canonical n4 ABBA cut makespan by a ratio of
///   0.800 (95% CI 0.776–0.824) with 26/26 `CalibrationPass`.
const QWEN_JOINT_DECODE_ENV: &str = "HF2Q_QWEN_JOINT_DECODE";

/// Widest joint step selected by default. Exact mode accepts 5..=8 when
/// raised explicitly (no serving measurement); the padded `mul_mv_ext` mode
/// caps at 4 so its packed width stays fixed.
const QWEN_JOINT_DECODE_MAX_ROWS_ENV: &str = "HF2Q_QWEN_JOINT_DECODE_MAX_ROWS";
const DEFAULT_QWEN_JOINT_DECODE_ROWS: usize = 4;

/// Kill switch for the padded-mode solo route (#717). Unset or
/// `1`/`on`/`true`: every eligible ordinary lane, a lone one included, runs
/// the padded joint forward. `0`/`off`/`false`: a lone lane and lanes beyond
/// `max_rows` run the serial one-row forward (the phase-1 partition).
const QWEN_JOINT_SOLO_ENV: &str = "HF2Q_QWEN_JOINT_SOLO";

/// Joint decode configuration resolved once at worker start.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Qwen35JointDecodeConfig {
    pub max_rows: usize,
    pub mode: JointRouteMode,
    /// Dense FFN gate/up route; `HF2Q_QWEN_JOINT_FFN_SHARED=0` keeps the
    /// serial builder in the padded mode (exact mode always keeps it).
    pub dense_ffn: JointDenseFfnRoute,
    /// Padded mode on a multi-slot worker with the solo route on: every
    /// eligible lane joins a joint group, and a lone ordinary lane runs a
    /// one-lane joint step instead of the serial forward.
    pub solo: bool,
    /// Per-step options; their switches are read here, once.
    pub step: JointStepOptions,
}

/// `Ok(None)` when unset, so the caller applies the default (on).
fn parse_qwen_joint_decode(raw: Option<&str>) -> Result<Option<bool>, String> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(None);
    };
    match raw.to_ascii_lowercase().as_str() {
        "1" | "on" | "true" => Ok(Some(true)),
        "0" | "off" | "false" => Ok(Some(false)),
        _ => Err(format!(
            "{QWEN_JOINT_DECODE_ENV} must be 1/on/true or 0/off/false, got {raw:?}"
        )),
    }
}

fn parse_qwen_joint_decode_max_rows(raw: Option<&str>) -> Result<usize, String> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(DEFAULT_QWEN_JOINT_DECODE_ROWS);
    };
    raw.parse::<usize>()
        .ok()
        .filter(|rows| (2..=MAX_JOINT_DECODE_ROWS).contains(rows))
        .ok_or_else(|| {
            format!(
                "{QWEN_JOINT_DECODE_MAX_ROWS_ENV} must be an integer in 2..={MAX_JOINT_DECODE_ROWS}, got {raw:?}"
            )
        })
}

/// Unset selects the solo route (`true`).
fn parse_qwen_joint_solo(raw: Option<&str>) -> Result<bool, String> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(true);
    };
    match raw.to_ascii_lowercase().as_str() {
        "1" | "on" | "true" => Ok(true),
        "0" | "off" | "false" => Ok(false),
        _ => Err(format!(
            "{QWEN_JOINT_SOLO_ENV} must be 1/on/true or 0/off/false, got {raw:?}"
        )),
    }
}

/// `solo` applies only to the padded mode on a worker with at least two
/// slots: exact-mode joint rows equal serial rows bitwise, and a one-slot
/// worker (N1) never shares a turn. `requested_step` holds the switches;
/// the column-invariant MMA projection applies only to the padded mode.
fn select_qwen_joint_decode(
    requested: bool,
    max_rows: usize,
    mode: Result<JointRouteMode, String>,
    ffn_shared: bool,
    solo_requested: bool,
    n_slots: usize,
    requested_step: JointStepOptions,
) -> Option<Qwen35JointDecodeConfig> {
    match (requested, mode) {
        (true, Ok(mode)) => Some(Qwen35JointDecodeConfig {
            max_rows: joint_mode_max_rows(mode, max_rows),
            mode,
            dense_ffn: select_dense_ffn_route(mode, ffn_shared),
            solo: solo_requested && mode == JointRouteMode::PaddedMvExt && n_slots >= 2,
            step: JointStepOptions {
                projection: select_joint_projection_kernel(
                    mode,
                    requested_step.projection == JointProjectionKernel::ColumnInvariantMma,
                ),
                pooled_outputs: requested_step.pooled_outputs,
            },
        }),
        _ => None,
    }
}

/// Resolve joint decode once when the Qwen worker starts and log the
/// decision, including why an artifact or configuration keeps serial decode.
pub(super) fn qwen35_joint_decode_config(
    model: &Qwen35Model,
    n_slots: usize,
) -> Option<Qwen35JointDecodeConfig> {
    let moe = model.cfg.moe.is_some();
    let default = true;
    let raw = std::env::var(QWEN_JOINT_DECODE_ENV).ok();
    let explicit = parse_qwen_joint_decode(raw.as_deref()).unwrap_or_else(|error| {
        tracing::warn!(
            error,
            fallback = default,
            "invalid Qwen joint decode switch"
        );
        None
    });
    let requested = explicit.unwrap_or(default);
    let raw_rows = std::env::var(QWEN_JOINT_DECODE_MAX_ROWS_ENV).ok();
    let max_rows = parse_qwen_joint_decode_max_rows(raw_rows.as_deref()).unwrap_or_else(|error| {
        tracing::warn!(
            error,
            fallback = DEFAULT_QWEN_JOINT_DECODE_ROWS,
            "invalid Qwen joint decode width"
        );
        DEFAULT_QWEN_JOINT_DECODE_ROWS
    });
    let mode = if requested {
        model.joint_decode_mode()
    } else {
        Err("disabled by HF2Q_QWEN_JOINT_DECODE".into())
    };
    let unavailable = mode.as_ref().err().cloned();
    let ffn_shared = joint_ffn_shared_from_process().unwrap_or_else(|error| {
        tracing::warn!(
            error,
            fallback = true,
            "invalid Qwen joint dense FFN switch"
        );
        true
    });
    let raw_solo = std::env::var(QWEN_JOINT_SOLO_ENV).ok();
    let solo_requested = parse_qwen_joint_solo(raw_solo.as_deref()).unwrap_or_else(|error| {
        tracing::warn!(error, fallback = true, "invalid Qwen joint solo switch");
        true
    });
    let switch = |name: &str, label: &str| {
        switch_from_process(name).unwrap_or_else(|error| {
            tracing::warn!(error, fallback = false, "invalid Qwen joint {label} switch");
            false
        })
    };
    let requested_step = JointStepOptions {
        projection: if switch(QWEN_JOINT_MMA_ENV, "MMA") {
            JointProjectionKernel::ColumnInvariantMma
        } else {
            JointProjectionKernel::Canonical
        },
        pooled_outputs: switch(QWEN_JOINT_POOLED_OUTPUTS_ENV, "pooled outputs"),
    };
    let selected = select_qwen_joint_decode(
        requested,
        max_rows,
        mode,
        ffn_shared,
        solo_requested,
        n_slots,
        requested_step,
    );
    tracing::info!(
        target: "hf2q::serve::api::engine_qwen35::progress",
        requested,
        explicit = explicit.is_some(),
        moe,
        max_rows = selected.map_or(max_rows, |config| config.max_rows),
        selected = selected.is_some(),
        mode = selected.map_or("serial", |config| config.mode.label()),
        dense_ffn = selected.map_or("serial", |config| config.dense_ffn.label()),
        solo = selected.is_some_and(|config| config.solo),
        column_invariant_mma = selected.is_some_and(|config| {
            config.step.projection == JointProjectionKernel::ColumnInvariantMma
        }),
        pooled_outputs = selected.is_some_and(|config| config.step.pooled_outputs),
        unavailable = unavailable.as_deref().unwrap_or("none"),
        "Qwen joint decode configuration"
    );
    selected
}

/// Re-check the process routing policy every turn: routing variables can
/// change after worker start; a turn that no longer satisfies the worker's
/// mode runs serially (logged once), and a later turn that satisfies it again
/// runs jointly.
pub(super) fn qwen35_joint_decode_turn_config(
    worker: Option<Qwen35JointDecodeConfig>,
) -> Option<Qwen35JointDecodeConfig> {
    let config = worker?;
    let reason = joint_decode_env_unavailable_reason().or_else(|| {
        check_mode_environment(config.mode, JointRoutingEnvironment::from_process()).err()
    });
    let Some(reason) = reason else {
        return Some(config);
    };
    static REVOKED: Once = Once::new();
    REVOKED.call_once(|| {
        tracing::warn!(
            target: "hf2q::serve::api::engine_qwen35::progress",
            reason = %reason,
            "Qwen joint decode revoked after worker start; decoding serially"
        );
    });
    None
}

/// One decode turn's joint groups and serial leftovers.
#[derive(Debug, PartialEq, Eq)]
struct JointPartition {
    /// Joint steps, run in order; each is sorted by slot.
    groups: Vec<Vec<SlotHandle>>,
    /// Handles for the serial loop, in scheduler order.
    serial: Vec<SlotHandle>,
}

/// Split one decode turn into joint groups and serial leftovers.
/// `classify` returns `None` for a lane that cannot join a joint step and
/// `Some(ordinary)` otherwise, where `ordinary` means its serial tick would
/// be the ordinary target forward (no MTP or history speculation state).
///
/// - Without `solo` (exact mode, one-slot workers, or the kill switch): one
///   group of the first `max_rows` eligible lanes, only when at least two are
///   eligible.
/// - With `solo` (padded mode, #717): every eligible lane, in consecutive
///   groups of at most `max_rows` in scheduler order. A group of one joins
///   only when its lane is ordinary, so a lone speculating lane keeps its
///   serial round.
///
/// `None` when handles repeat a slot or no group forms: the whole turn is
/// serial.
fn partition_qwen35_joint_lanes(
    handles: &[SlotHandle],
    max_rows: usize,
    solo: bool,
    mut classify: impl FnMut(SlotHandle) -> Option<bool>,
) -> Option<JointPartition> {
    let unique = handles.iter().enumerate().all(|(index, handle)| {
        handles[..index]
            .iter()
            .all(|earlier| earlier.slot_id != handle.slot_id)
    });
    if !unique || max_rows < 2 {
        return None;
    }
    let mut joined = vec![false; handles.len()];
    let mut groups: Vec<Vec<SlotHandle>> = Vec::new();
    if solo {
        let eligible = handles
            .iter()
            .enumerate()
            .filter_map(|(index, &handle)| classify(handle).map(|ordinary| (index, ordinary)))
            .collect::<Vec<_>>();
        for chunk in eligible.chunks(max_rows) {
            if let [(_, false)] = chunk {
                continue;
            }
            for &(index, _) in chunk {
                joined[index] = true;
            }
            groups.push(chunk.iter().map(|&(index, _)| handles[index]).collect());
        }
    } else {
        let mut group = Vec::with_capacity(handles.len().min(max_rows));
        for (index, &handle) in handles.iter().enumerate() {
            if group.len() < max_rows && classify(handle).is_some() {
                group.push(index);
            }
        }
        if group.len() >= 2 {
            for &index in &group {
                joined[index] = true;
            }
            groups.push(group.iter().map(|&index| handles[index]).collect());
        }
    }
    if groups.is_empty() {
        return None;
    }
    for group in &mut groups {
        group.sort_by_key(|handle| handle.slot_id);
    }
    let serial = handles
        .iter()
        .zip(&joined)
        .filter(|(_, &joined)| !joined)
        .map(|(&handle, _)| handle)
        .collect();
    Some(JointPartition { groups, serial })
}

/// A failed shared transaction fails exactly the rows that submitted it:
/// the first keeps the original error, the rest get its rendering.
fn distribute_step_error<T>(
    error: anyhow::Error,
    rows: usize,
) -> impl Iterator<Item = anyhow::Result<T>> {
    let message = format!("{error:#}");
    let mut first = Some(error);
    (0..rows).map(move |_| Err(first.take().unwrap_or_else(|| anyhow::anyhow!("{message}"))))
}

pub(super) enum Qwen35JointRun {
    /// No joint group forms; the caller runs the whole turn serially.
    NotEligible,
    /// Joint lanes are settled; `serial` handles still need the serial loop.
    Ran {
        serial: Vec<SlotHandle>,
    },
    Fatal(Qwen35FatalFailure),
}

struct JointLane {
    handle: SlotHandle,
    reply: super::SlotReply,
    state: crate::serve::api::engine_qwen35::Qwen35DecodeState,
    tick: Option<anyhow::Result<Qwen35TickOutcome>>,
}

fn reinstall_lanes(slots: &mut [Option<Qwen35Slot>], lanes: impl IntoIterator<Item = JointLane>) {
    for lane in lanes {
        let index = lane.handle.slot_id.0 as usize;
        debug_assert!(slots[index].is_none());
        slots[index] = Some((Qwen35SlotWork::Decode(lane.state), lane.reply, lane.handle));
    }
}

/// Run the turn's joint groups one after another, then hand the serial
/// leftovers back to the caller. Each group is one joint step whose lanes
/// settle with the serial path's slot semantics.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_qwen35_joint_decode(
    guard: &mut Qwen35KvGuard<'_>,
    scheduler: &mut InflightBatchedScheduler,
    slots: &mut [Option<Qwen35Slot>],
    retained_tokens: &mut [Qwen35RetainedPrefix],
    prompt_caches: &mut [PromptCache],
    prompt_anchors: &mut [Option<Qwen35PromptAnchor>],
    registration: Option<&ModelRegistration>,
    handles: &[SlotHandle],
    kv_bytes_per_token: u64,
    config: Qwen35JointDecodeConfig,
    supervisor: &EngineSupervisor,
) -> Qwen35JointRun {
    if guard
        .kv
        .as_ref()
        .expect("kv Some during loop")
        .la_capture_active()
    {
        // An active DeltaNet capture would make the joint transaction refuse
        // after lanes were prepared; keep such a turn serial up front.
        return Qwen35JointRun::NotEligible;
    }
    // Repeatable route: plan every lane's tick from its own state before the
    // partition, so a lane that speculates or drains verified output never
    // joins a joint step and the serial loop runs the same decision. A plan
    // error leaves the lane unplanned; its serial tick reports the error.
    for &handle in handles {
        if let Some(Some((Qwen35SlotWork::Decode(state), _, installed))) =
            slots.get_mut(handle.slot_id.0 as usize)
        {
            if *installed == handle && state.speculation_route() == QwenSpeculationRoute::Repeatable
            {
                let _ = state.plan_tick();
            }
        }
    }
    let Some(JointPartition { groups, mut serial }) = ({
        let kv = guard.kv.as_ref().expect("kv Some during loop");
        partition_qwen35_joint_lanes(handles, config.max_rows, config.solo, |handle| match slots
            .get(handle.slot_id.0 as usize)
            .and_then(Option::as_ref)
        {
            Some((Qwen35SlotWork::Decode(state), reply, installed))
                if *installed == handle
                    && !reply.client_closed()
                    && state.joint_decode_ready(kv) =>
            {
                state.joint_lane_class()
            }
            _ => None,
        })
    }) else {
        return Qwen35JointRun::NotEligible;
    };

    let mut groups = groups.into_iter();
    while let Some(group) = groups.next() {
        // A capture opened by an earlier group's settlement would make this
        // group's transaction refuse; its lanes and the rest decode serially.
        if guard
            .kv
            .as_ref()
            .expect("kv Some during loop")
            .la_capture_active()
        {
            serial.extend(group);
            serial.extend(groups.by_ref().flatten());
            break;
        }
        if let Some(fatal) = run_qwen35_joint_group(
            guard,
            scheduler,
            slots,
            retained_tokens,
            prompt_caches,
            prompt_anchors,
            registration,
            group,
            serial.len(),
            handles.len(),
            kv_bytes_per_token,
            config,
            supervisor,
        ) {
            return Qwen35JointRun::Fatal(fatal);
        }
    }
    Qwen35JointRun::Ran { serial }
}

/// One joint step over `group` (sorted by slot, 1..=`max_rows` lanes), then
/// settle each lane with the serial path's slot semantics.
#[allow(clippy::too_many_arguments)]
fn run_qwen35_joint_group(
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
    let mut lanes = Vec::with_capacity(group.len());
    for handle in group {
        let index = handle.slot_id.0 as usize;
        match slots[index].take() {
            Some((Qwen35SlotWork::Decode(state), reply, installed)) if installed == handle => {
                lanes.push(JointLane {
                    handle,
                    reply,
                    state,
                    tick: None,
                });
            }
            other => {
                slots[index] = other;
                reinstall_lanes(slots, lanes);
                return Some(Qwen35FatalFailure {
                    handle: Some(handle),
                    reply: Qwen35FatalReply::None,
                    error: anyhow::anyhow!("Qwen joint decode lane changed before execution"),
                    kind: Qwen35WorkerFatalKind::Invariant,
                });
            }
        }
    }

    for lane in &mut lanes {
        // The joint row serves the lane's ordinary tick; drop its plan.
        lane.state.consume_planned_tick();
        tracing::debug!(
            target: "hf2q::serve::api::engine_qwen35::progress",
            slot = lane.handle.slot_id.0,
            kind = "ordinary",
            path = "joint",
            turn_lanes,
            "Qwen decode tick route"
        );
    }

    static FIRST_JOINT_DECODE: Once = Once::new();
    FIRST_JOINT_DECODE.call_once(|| {
        tracing::info!(
            target: "hf2q::serve::api::engine_qwen35::progress",
            lanes = lanes.len(),
            mode = config.mode.label(),
            dense_ffn = config.dense_ffn.label(),
            "Qwen joint decode engaged"
        );
    });

    // Prepare each lane exactly as its serial tick would before the forward.
    let mut prepared: Vec<(usize, Qwen35JointLane)> = Vec::with_capacity(lanes.len());
    {
        let kv = guard.kv.as_ref().expect("kv Some during loop");
        for (index, lane) in lanes.iter_mut().enumerate() {
            match lane.state.prepare_joint_tick(kv) {
                Ok(joint_lane) => prepared.push((index, joint_lane)),
                Err(error) => lane.tick = Some(Err(error)),
            }
        }
    }
    tracing::debug!(
        target: "hf2q::serve::api::engine_qwen35::progress",
        lanes = lanes.len(),
        rows = prepared.len(),
        serial = serial_lanes,
        "Qwen joint decode turn"
    );

    if !prepared.is_empty() {
        let rows = prepared
            .iter()
            .map(|(_, lane)| lane.row)
            .collect::<Vec<_>>();
        let _native_invocation = scheduler.trace_transaction(
            "qwen35",
            "joint_ordinary_decode",
            &prepared
                .iter()
                .map(|&(index, _)| lanes[index].handle)
                .collect::<Vec<_>>(),
        );
        let forward = guard.model.forward_joint_decode(
            &rows,
            guard.kv.as_mut().expect("kv Some during loop"),
            config.mode,
            config.dense_ffn,
            config.step,
            supervisor,
        );
        match forward {
            Ok(outputs) => {
                for ((index, joint_lane), output) in prepared.into_iter().zip(outputs) {
                    let lane = &mut lanes[index];
                    lane.tick = Some(lane.state.finish_joint_tick(
                        guard.model,
                        guard.kv.as_mut().expect("kv Some during loop"),
                        supervisor,
                        joint_lane.target,
                        output,
                    ));
                }
            }
            Err(error) if !supervisor.is_healthy() || is_fatal_command_buffer_error(&error) => {
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
            Err(error) => {
                // Every prepared row shared the failed transaction, so each
                // fails exactly as its own serial forward failure would;
                // rows outside the step are untouched.
                let indices = prepared.iter().map(|&(index, _)| index).collect::<Vec<_>>();
                for (index, tick) in indices
                    .iter()
                    .zip(distribute_step_error(error, indices.len()))
                {
                    lanes[*index].tick = Some(tick);
                }
            }
        }
    }

    let mut owned = lanes.into_iter();
    while let Some(lane) = owned.next() {
        let tick = lane
            .tick
            .unwrap_or_else(|| Err(anyhow::anyhow!("Qwen joint lane produced no tick")));
        if let Some(fatal) = settle_qwen35_decode_lane(
            guard,
            scheduler,
            slots,
            retained_tokens,
            prompt_caches,
            prompt_anchors,
            registration,
            lane.handle,
            lane.reply,
            lane.state,
            tick,
            kv_bytes_per_token,
            supervisor,
        ) {
            reinstall_lanes(slots, owned);
            return Some(fatal);
        }
    }
    None
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
    fn joint_decode_switch_parses_explicit_arms() {
        assert_eq!(parse_qwen_joint_decode(None), Ok(None));
        assert_eq!(parse_qwen_joint_decode(Some("  ")), Ok(None));
        for raw in ["1", "on", "TRUE"] {
            assert_eq!(parse_qwen_joint_decode(Some(raw)), Ok(Some(true)));
        }
        for raw in ["0", "Off", "false"] {
            assert_eq!(parse_qwen_joint_decode(Some(raw)), Ok(Some(false)));
        }
        assert!(parse_qwen_joint_decode(Some("joint")).is_err());
    }

    #[test]
    fn joint_solo_switch_defaults_on_and_parses_explicit_arms() {
        assert_eq!(parse_qwen_joint_solo(None), Ok(true));
        assert_eq!(parse_qwen_joint_solo(Some(" ")), Ok(true));
        for raw in ["1", "On", "true"] {
            assert_eq!(parse_qwen_joint_solo(Some(raw)), Ok(true));
        }
        for raw in ["0", "OFF", "false"] {
            assert_eq!(parse_qwen_joint_solo(Some(raw)), Ok(false));
        }
        assert!(parse_qwen_joint_solo(Some("solo")).is_err());
    }

    #[test]
    fn joint_width_defaults_to_four_and_accepts_two_through_eight() {
        assert_eq!(parse_qwen_joint_decode_max_rows(None), Ok(4));
        assert_eq!(parse_qwen_joint_decode_max_rows(Some(" ")), Ok(4));
        for rows in 2..=MAX_JOINT_DECODE_ROWS {
            assert_eq!(
                parse_qwen_joint_decode_max_rows(Some(&rows.to_string())),
                Ok(rows)
            );
        }
        for raw in ["0", "1", "9", "four"] {
            assert!(parse_qwen_joint_decode_max_rows(Some(raw)).is_err());
        }
    }

    #[test]
    fn refused_artifacts_and_disabled_switch_keep_serial_decode() {
        let pooled = JointStepOptions {
            pooled_outputs: true,
            ..JointStepOptions::default()
        };
        assert_eq!(
            select_qwen_joint_decode(true, 4, Ok(JointRouteMode::Exact), true, true, 4, pooled),
            Some(Qwen35JointDecodeConfig {
                max_rows: 4,
                mode: JointRouteMode::Exact,
                dense_ffn: JointDenseFfnRoute::SerialBuilder,
                solo: false,
                step: pooled,
            })
        );
        assert_eq!(
            select_qwen_joint_decode(
                true,
                4,
                Err("layer 3: attn_qkv is IQ4_NL, which has no proven joint decode route".into()),
                true,
                true,
                4,
                JointStepOptions::default()
            ),
            None
        );
        assert_eq!(
            select_qwen_joint_decode(
                false,
                4,
                Ok(JointRouteMode::PaddedMvExt),
                true,
                true,
                4,
                JointStepOptions::default()
            ),
            None
        );
    }

    #[test]
    fn mma_projection_is_padded_mode_only() {
        let mma = JointStepOptions {
            projection: JointProjectionKernel::ColumnInvariantMma,
            ..JointStepOptions::default()
        };
        let projection = |mode| {
            select_qwen_joint_decode(true, 4, Ok(mode), true, true, 4, mma)
                .expect("admitted mode")
                .step
                .projection
        };
        assert_eq!(
            projection(JointRouteMode::PaddedMvExt),
            JointProjectionKernel::ColumnInvariantMma
        );
        assert_eq!(
            projection(JointRouteMode::Exact),
            JointProjectionKernel::Canonical
        );
    }

    #[test]
    fn padded_mode_caps_joint_width_at_four() {
        for requested in 2..=MAX_JOINT_DECODE_ROWS {
            let padded = select_qwen_joint_decode(
                true,
                requested,
                Ok(JointRouteMode::PaddedMvExt),
                true,
                true,
                4,
                JointStepOptions::default(),
            )
            .expect("admitted padded mode");
            assert_eq!(padded.max_rows, requested.min(4));
            let exact = select_qwen_joint_decode(
                true,
                requested,
                Ok(JointRouteMode::Exact),
                true,
                true,
                4,
                JointStepOptions::default(),
            )
            .expect("admitted exact mode");
            assert_eq!(exact.max_rows, requested);
        }
    }

    #[test]
    fn solo_route_is_padded_mode_on_multi_slot_workers_only() {
        let solo = |mode, solo_requested, n_slots| {
            select_qwen_joint_decode(
                true,
                4,
                Ok(mode),
                true,
                solo_requested,
                n_slots,
                JointStepOptions::default(),
            )
            .expect("admitted mode")
            .solo
        };
        for n_slots in 2..=8 {
            assert!(solo(JointRouteMode::PaddedMvExt, true, n_slots));
            assert!(!solo(JointRouteMode::PaddedMvExt, false, n_slots));
            assert!(!solo(JointRouteMode::Exact, true, n_slots));
        }
        // N1: a one-slot worker never shares a turn and keeps serial decode.
        assert!(!solo(JointRouteMode::PaddedMvExt, true, 1));
    }

    /// Every handle eligible; `ordinary` lists the slots whose serial tick is
    /// the ordinary target forward.
    fn classify_by(eligible: &[u32], ordinary: &[u32]) -> impl Fn(SlotHandle) -> Option<bool> {
        let eligible = eligible.to_vec();
        let ordinary = ordinary.to_vec();
        move |handle| {
            eligible
                .contains(&handle.slot_id.0)
                .then(|| ordinary.contains(&handle.slot_id.0))
        }
    }

    #[test]
    fn ragged_eligible_lanes_join_sorted_and_the_rest_stay_serial() {
        let handles = [handle(3, 9), handle(0, 1), handle(2, 4), handle(1, 7)];
        for solo in [false, true] {
            let partition = partition_qwen35_joint_lanes(
                &handles,
                4,
                solo,
                classify_by(&[0, 1, 3], &[0, 1, 3]),
            )
            .expect("three eligible lanes form a joint step");
            assert_eq!(
                partition,
                JointPartition {
                    groups: vec![vec![handle(0, 1), handle(1, 7), handle(3, 9)]],
                    serial: vec![handle(2, 4)],
                }
            );
        }
    }

    #[test]
    fn phase_one_partition_keeps_fewer_than_two_eligible_lanes_serial() {
        let handles = [handle(0, 1), handle(1, 1)];
        assert!(
            partition_qwen35_joint_lanes(&handles, 4, false, classify_by(&[0], &[0])).is_none()
        );
        assert!(partition_qwen35_joint_lanes(&[], 4, false, |_| Some(true)).is_none());
        assert!(partition_qwen35_joint_lanes(&handles, 1, false, |_| Some(true)).is_none());
    }

    #[test]
    fn solo_partition_runs_a_lone_ordinary_lane_jointly() {
        let handles = [handle(0, 1), handle(1, 1)];
        assert_eq!(
            partition_qwen35_joint_lanes(&handles, 4, true, classify_by(&[1], &[1])),
            Some(JointPartition {
                groups: vec![vec![handle(1, 1)]],
                serial: vec![handle(0, 1)],
            })
        );
        // A lone lane that would speculate keeps its serial round.
        assert!(partition_qwen35_joint_lanes(&handles, 4, true, classify_by(&[1], &[])).is_none());
        // Two lanes join whatever their speculation state, as in phase 1.
        assert_eq!(
            partition_qwen35_joint_lanes(&handles, 4, true, classify_by(&[0, 1], &[])),
            Some(JointPartition {
                groups: vec![vec![handle(0, 1), handle(1, 1)]],
                serial: vec![],
            })
        );
        assert!(partition_qwen35_joint_lanes(&[], 4, true, |_| Some(true)).is_none());
    }

    #[test]
    fn duplicate_slots_never_form_a_joint_step() {
        let handles = [handle(0, 1), handle(1, 1), handle(0, 2)];
        for solo in [false, true] {
            assert!(partition_qwen35_joint_lanes(&handles, 4, solo, |_| Some(true)).is_none());
        }
    }

    #[test]
    fn phase_one_partition_caps_the_joint_width_and_overflow_lanes_stay_serial() {
        let handles = [handle(4, 1), handle(3, 1), handle(2, 1), handle(1, 1)];
        let partition = partition_qwen35_joint_lanes(&handles, 3, false, |_| Some(true))
            .expect("capped joint step");
        assert_eq!(
            partition,
            JointPartition {
                groups: vec![vec![handle(2, 1), handle(3, 1), handle(4, 1)]],
                serial: vec![handle(1, 1)],
            }
        );
    }

    #[test]
    fn solo_partition_puts_every_eligible_lane_in_a_group_at_any_width() {
        let handles = (0..8).rev().map(|slot| handle(slot, 1)).collect::<Vec<_>>();
        for max_rows in 2..=4 {
            for lanes in 1..=handles.len() {
                let partition =
                    partition_qwen35_joint_lanes(&handles[..lanes], max_rows, true, |_| Some(true))
                        .expect("ordinary lanes always form groups");
                assert!(
                    partition.serial.is_empty(),
                    "max_rows={max_rows} lanes={lanes}"
                );
                assert_eq!(partition.groups.len(), lanes.div_ceil(max_rows));
                let mut joined = partition.groups.concat();
                assert!(partition.groups.iter().all(|group| {
                    (1..=max_rows).contains(&group.len())
                        && group
                            .windows(2)
                            .all(|pair| pair[0].slot_id < pair[1].slot_id)
                }));
                joined.sort_by_key(|handle| handle.slot_id);
                let mut expected = handles[..lanes].to_vec();
                expected.sort_by_key(|handle| handle.slot_id);
                assert_eq!(joined, expected);
            }
        }
        // Five lanes at width 4: the lone leftover speculates, so it alone is
        // serial; groups follow scheduler order.
        let partition = partition_qwen35_joint_lanes(
            &handles[..5],
            4,
            true,
            classify_by(&[3, 4, 5, 6, 7], &[4, 5, 6, 7]),
        )
        .expect("one full group");
        assert_eq!(
            partition,
            JointPartition {
                groups: vec![vec![handle(4, 1), handle(5, 1), handle(6, 1), handle(7, 1)]],
                serial: vec![handle(3, 1)],
            }
        );
    }

    #[test]
    fn a_failed_step_fails_exactly_its_rows_with_the_original_error_first() {
        let errors = distribute_step_error::<()>(
            anyhow::anyhow!("command buffer failed").context("Qwen joint decode"),
            3,
        )
        .collect::<Vec<_>>();
        assert_eq!(errors.len(), 3);
        for error in &errors {
            let rendered = format!("{:#}", error.as_ref().expect_err("every row fails"));
            assert!(rendered.contains("command buffer failed"), "{rendered}");
            assert!(rendered.contains("Qwen joint decode"), "{rendered}");
        }
        assert_eq!(
            distribute_step_error::<()>(anyhow::anyhow!("x"), 0).count(),
            0
        );
    }
}
