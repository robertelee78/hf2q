//! Qwen joint B-row decode scheduling (ADR-063 S3, phase 1).
//!
//! One scheduler decode turn advances every eligible decoding slot with one
//! shared target forward instead of one forward per slot. Sampling, grammar
//! and tool-call constraints, thinking budgets, stop strings, streaming,
//! cancellation, retained-prefix publication and failure recovery stay per
//! slot: each joint lane settles through `settle_qwen35_decode_lane`, the
//! same tail the serial loop uses. Ineligible handles (closed clients, CPU-only
//! verified output, cursor/ledger mismatch) stay on the serial loop. With
//! fewer than two eligible lanes the whole turn is serial, including MTP K3
//! and history speculation.

use std::sync::Once;

use crate::inference::models::qwen35::joint_decode_routes::{
    check_mode_environment, joint_decode_env_unavailable_reason, joint_mode_max_rows,
    JointRouteMode, JointRoutingEnvironment, MAX_JOINT_DECODE_ROWS,
};
use crate::inference::models::qwen35::model::Qwen35Model;
use crate::serve::api::engine_qwen35::{Qwen35JointLane, Qwen35TickOutcome};
use crate::serve::api::registry::ModelRegistration;

use super::{
    is_fatal_command_buffer_error, settle_qwen35_decode_lane, EngineSupervisor,
    InflightBatchedScheduler, PromptCache, Qwen35FatalFailure, Qwen35FatalReply, Qwen35KvGuard,
    Qwen35PromptAnchor, Qwen35RetainedPrefix, Qwen35Slot, Qwen35SlotWork, Qwen35WorkerFatalKind,
    SlotHandle,
};

/// Kill switch for A/B measurement: `1`/`on`/`true` selects joint decode,
/// `0`/`off`/`false` keeps the serial per-slot turn. Unset selects the
/// per-family default ([`default_joint_decode`]).
const QWEN_JOINT_DECODE_ENV: &str = "HF2Q_QWEN_JOINT_DECODE";

/// Per-family default when the switch is unset. Dense artifacts (Qwen3.8)
/// are on: the official-artifact gates passed and the four-agent ABBA A/B
/// cut median makespan 116.4 → 90.1 s with 8/8 `CalibrationPass`. MoE
/// artifacts (Qwen3.6) are off: their forward is bitwise exact, but joint
/// rows commit M1 tokens where stock commits K3 verify tokens, and the
/// incomplete A/B (2 runs per arm) failed the coding protocol on both joint
/// runs. They need `HF2Q_QWEN_JOINT_DECODE=1` until a complete ABBA shows
/// protocol parity.
fn default_joint_decode(moe: bool) -> bool {
    !moe
}

const MOE_DEFAULT_OFF_REASON: &str = "MoE artifacts default off pending a complete Qwen3.6 \
     four-agent A/B; set HF2Q_QWEN_JOINT_DECODE=1 to opt in";
/// Widest joint step selected by default. Exact mode accepts 5..=8 when
/// raised explicitly (no serving measurement); the padded `mul_mv_ext` mode
/// caps at 4 so its packed width stays fixed.
const QWEN_JOINT_DECODE_MAX_ROWS_ENV: &str = "HF2Q_QWEN_JOINT_DECODE_MAX_ROWS";
const DEFAULT_QWEN_JOINT_DECODE_ROWS: usize = 4;

/// Joint decode configuration resolved once at worker start.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Qwen35JointDecodeConfig {
    pub max_rows: usize,
    pub mode: JointRouteMode,
}

/// `Ok(None)` when unset, so the caller applies the per-family default.
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

fn select_qwen_joint_decode(
    requested: bool,
    max_rows: usize,
    mode: Result<JointRouteMode, String>,
) -> Option<Qwen35JointDecodeConfig> {
    match (requested, mode) {
        (true, Ok(mode)) => Some(Qwen35JointDecodeConfig {
            max_rows: joint_mode_max_rows(mode, max_rows),
            mode,
        }),
        _ => None,
    }
}

/// Resolve joint decode once when the Qwen worker starts and log the
/// decision, including why an artifact or configuration keeps serial decode.
pub(super) fn qwen35_joint_decode_config(model: &Qwen35Model) -> Option<Qwen35JointDecodeConfig> {
    let moe = model.cfg.moe.is_some();
    let default = default_joint_decode(moe);
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
    let mode = match (requested, explicit) {
        (true, _) => model.joint_decode_mode(),
        (false, None) => Err(MOE_DEFAULT_OFF_REASON.into()),
        (false, Some(_)) => Err("disabled by HF2Q_QWEN_JOINT_DECODE".into()),
    };
    let unavailable = mode.as_ref().err().cloned();
    let selected = select_qwen_joint_decode(requested, max_rows, mode);
    tracing::info!(
        target: "hf2q::serve::api::engine_qwen35::progress",
        requested,
        explicit = explicit.is_some(),
        moe,
        max_rows = selected.map_or(max_rows, |config| config.max_rows),
        selected = selected.is_some(),
        mode = selected.map_or("serial", |config| config.mode.label()),
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

/// Split one decode turn into joint lanes and serial leftovers. The whole
/// turn stays serial unless handles are unique per slot and at least two are
/// eligible. Joint lanes are sorted by slot and capped at `max_rows`; serial
/// leftovers keep scheduler order.
fn partition_qwen35_joint_lanes(
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
    /// Fewer than two eligible lanes; the caller runs the whole turn serially.
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

/// Run one joint step for every eligible slot, then settle each joint lane
/// with the serial path's slot semantics.
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
    let Some((joint, serial)) = ({
        let kv = guard.kv.as_ref().expect("kv Some during loop");
        // An active DeltaNet capture would make the joint transaction refuse
        // after lanes were prepared; keep such a turn serial up front.
        if kv.la_capture_active() {
            return Qwen35JointRun::NotEligible;
        }
        partition_qwen35_joint_lanes(handles, config.max_rows, |handle| {
            slots
                .get(handle.slot_id.0 as usize)
                .and_then(Option::as_ref)
                .is_some_and(|(work, reply, installed)| {
                    *installed == handle
                        && !reply.client_closed()
                        && matches!(work, Qwen35SlotWork::Decode(state) if state.joint_decode_ready(kv))
                })
        })
    }) else {
        return Qwen35JointRun::NotEligible;
    };

    let mut lanes = Vec::with_capacity(joint.len());
    for handle in joint {
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
                return Qwen35JointRun::Fatal(Qwen35FatalFailure {
                    handle: Some(handle),
                    reply: Qwen35FatalReply::None,
                    error: anyhow::anyhow!("Qwen joint decode lane changed before execution"),
                    kind: Qwen35WorkerFatalKind::Invariant,
                });
            }
        }
    }

    static FIRST_JOINT_DECODE: Once = Once::new();
    FIRST_JOINT_DECODE.call_once(|| {
        tracing::info!(
            target: "hf2q::serve::api::engine_qwen35::progress",
            lanes = lanes.len(),
            mode = config.mode.label(),
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
        serial = serial.len(),
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
                let first = owned.next().expect("joint decode has at least two lanes");
                reinstall_lanes(slots, owned);
                return Qwen35JointRun::Fatal(Qwen35FatalFailure {
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
            return Qwen35JointRun::Fatal(fatal);
        }
    }
    Qwen35JointRun::Ran { serial }
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
    fn joint_decode_switch_defaults_per_family_and_parses_explicit_arms() {
        assert_eq!(parse_qwen_joint_decode(None), Ok(None));
        assert_eq!(parse_qwen_joint_decode(Some("  ")), Ok(None));
        // Dense (Qwen3.8) on, MoE (Qwen3.6) off unless explicitly enabled.
        assert!(default_joint_decode(false));
        assert!(!default_joint_decode(true));
        for raw in ["1", "on", "TRUE"] {
            assert_eq!(parse_qwen_joint_decode(Some(raw)), Ok(Some(true)));
        }
        for raw in ["0", "Off", "false"] {
            assert_eq!(parse_qwen_joint_decode(Some(raw)), Ok(Some(false)));
        }
        assert!(parse_qwen_joint_decode(Some("joint")).is_err());
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
        assert_eq!(
            select_qwen_joint_decode(true, 4, Ok(JointRouteMode::Exact)),
            Some(Qwen35JointDecodeConfig {
                max_rows: 4,
                mode: JointRouteMode::Exact
            })
        );
        assert_eq!(
            select_qwen_joint_decode(
                true,
                4,
                Err("layer 3: attn_qkv is IQ4_NL, which has no proven joint decode route".into())
            ),
            None
        );
        assert_eq!(
            select_qwen_joint_decode(false, 4, Ok(JointRouteMode::PaddedMvExt)),
            None
        );
    }

    #[test]
    fn padded_mode_caps_joint_width_at_four() {
        for requested in 2..=MAX_JOINT_DECODE_ROWS {
            let padded = select_qwen_joint_decode(true, requested, Ok(JointRouteMode::PaddedMvExt))
                .expect("admitted padded mode");
            assert_eq!(padded.max_rows, requested.min(4));
            let exact = select_qwen_joint_decode(true, requested, Ok(JointRouteMode::Exact))
                .expect("admitted exact mode");
            assert_eq!(exact.max_rows, requested);
        }
    }

    #[test]
    fn ragged_eligible_lanes_join_sorted_and_the_rest_stay_serial() {
        let handles = [handle(3, 9), handle(0, 1), handle(2, 4), handle(1, 7)];
        let (joint, serial) =
            partition_qwen35_joint_lanes(&handles, 4, |handle| handle.slot_id != SlotId(2))
                .expect("three eligible lanes form a joint step");
        assert_eq!(joint, [handle(0, 1), handle(1, 7), handle(3, 9)]);
        assert_eq!(serial, [handle(2, 4)]);
    }

    #[test]
    fn fewer_than_two_eligible_lanes_keep_the_whole_turn_serial() {
        let handles = [handle(0, 1), handle(1, 1)];
        assert!(
            partition_qwen35_joint_lanes(&handles, 4, |handle| handle.slot_id == SlotId(0))
                .is_none()
        );
        assert!(partition_qwen35_joint_lanes(&[], 4, |_| true).is_none());
        assert!(partition_qwen35_joint_lanes(&handles, 1, |_| true).is_none());
    }

    #[test]
    fn duplicate_slots_never_form_a_joint_step() {
        let handles = [handle(0, 1), handle(1, 1), handle(0, 2)];
        assert!(partition_qwen35_joint_lanes(&handles, 4, |_| true).is_none());
    }

    #[test]
    fn joint_width_is_capped_and_overflow_lanes_stay_serial() {
        let handles = [handle(4, 1), handle(3, 1), handle(2, 1), handle(1, 1)];
        let (joint, serial) =
            partition_qwen35_joint_lanes(&handles, 3, |_| true).expect("capped joint step");
        assert_eq!(joint, [handle(2, 1), handle(3, 1), handle(4, 1)]);
        assert_eq!(serial, [handle(1, 1)]);
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
