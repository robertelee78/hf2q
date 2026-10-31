//! ADR-063 Stage 1: joint warm Gemma 4 prefill for same-step tool turns.
//!
//! With every slot idle, one admission quantum can hold several warm text
//! requests: full-input-tagged tool turns, or any request whose slot holds a
//! reusable prefix. Admitted one at a time, each prefills inline and the
//! worker yields, so later slots are admitted between decode steps and pay the
//! fixed per-forward cost again. Here a contiguous FIFO prefix of such
//! requests is admitted without GPU work and prefilled in one
//! `forward_prefill_batched_multi_seq_live` call with
//! [`GemmaLiveAttention::PerSlotExact`]:
//!
//! * projections, norms, dense FFN/MoE and the per-row head run once over
//!   every suffix row;
//! * attention runs per slot against its own KV through the one shared
//!   live attention a single-slot warm prefill uses, on the same route
//!   (tiled F16 by default, vec-hybrid under
//!   `HF2Q_GEMMA_WARM_TILED_ATTENTION=0`), chunking, staging and FWHT
//!   handling.
//!
//! Each lane is therefore bitwise equal to its per-slot prefill on either
//! route; the gates are `gemma_joint_warm_prefill_matches_per_slot_bitwise`
//! and `..._vec_hybrid` (real artifact).
//!
//! Lane eligibility: text only (no soft tokens, vision identity or stable cue
//! split), the slot selected at admission holds `cached > 0` tokens, the
//! suffix exceeds 32 rows, and the cohort stays within the 4,096-row
//! transaction cap. Planning and installation are model-free; a lane whose
//! actually selected slot fails the rule is admitted inline exactly as
//! before. Lanes are submitted, captured and seeded in FIFO admission order,
//! and the decode reserve passed to the forward is the binding lane's own
//! (`gemma4_batch_decode_reserve`), so a cohort fits whenever every lane fits
//! alone.
//!
//! Failure semantics:
//! * a pre-submission check failure (substrate, plan, cursor) leaves every
//!   lane installed and prefills the primary per slot; the remaining lanes
//!   are retried as a smaller cohort or per slot;
//! * a submitted, non-fatal forward error restores every cohort lane from its
//!   own request-start rollback anchor and re-prefills it per slot in FIFO
//!   order (`settle_gemma4_prefill_batch`). One lane's fault recurs in its own
//!   per-slot prefill and fails only that request, as serial prefill would;
//!   a lane that cannot be restored is failed request-locally. The
//!   fault-injection gate is `gemma_joint_warm_forward_fault_retries_each_lane_per_slot`
//!   (real artifact). A worker-fatal GPU error fails the worker, as every
//!   other Gemma transaction does;
//! * a client that disconnects before submission is left out of the cohort
//!   and cancelled by its per-slot advance; one that disconnects during the
//!   forward is restored from its just-published checkpoint after the
//!   transaction, while the other lanes continue.
use super::*;
use crate::serve::api::registry::ModelRegistration;

pub(super) const JOINT_WARM_PREFILL_ENV: &str = "HF2Q_GEMMA_JOINT_WARM_PREFILL";
/// On by default: the official-artifact bitwise gate is the contract, and the
/// root runs it before merge.
const JOINT_WARM_PREFILL_DEFAULT: bool = true;
/// The binding constraint is the MoE route. mlx-native routes the expert
/// matmuls to `mm_id` only when `n_tokens > MM_ID_ROUTING_THRESHOLD` (32 in
/// 0.15.15); dense GGUF matmuls take the mm kernels for every `m > 8`. A
/// per-slot warm prefill of `T` rows runs `gate_up` with `n_tokens = T`, the
/// joint stream with `n_tokens = Σ T` (> 32 for any cohort), so a lane with
/// `T = 32` would run `mv_id` alone but `mm_id` jointly. (Suffixes below 32
/// are also replayed through decode forwards by
/// `forward_prefill_with_soft_tokens_slot_aware_resume`.)
pub(super) const MIN_SUFFIX_TOKENS: usize = mlx_native::MM_ID_ROUTING_THRESHOLD as usize + 1;
/// The existing Gemma 4 transaction row cap.
pub(super) const ROW_CAP: usize = GEMMA4_SLOT_PREFILL_CHUNK_TOKENS as usize;

#[cfg(test)]
pub(super) static JOINT_WARM_TRANSACTIONS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(super) static JOINT_WARM_LANES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// ADR-063 S3 G-C3 fault injection (tests only). When armed, the next joint
/// warm transaction keeps its real forward's KV writes but reports a
/// non-fatal error, so settlement must restore every lane from its rollback
/// anchor and re-prefill it per slot. The arm is consumed by one transaction.
#[cfg(test)]
pub(super) static INJECT_FORWARD_FAULT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
/// `(slot, prompt tokens)` of the faulted transaction's lanes, in submission
/// order.
#[cfg(test)]
pub(super) static FAULTED_LANES: std::sync::Mutex<Vec<(u32, usize)>> =
    std::sync::Mutex::new(Vec::new());
/// `(slot, prompt tokens)` of every lane restored for a per-slot retry
/// (`settle_gemma4_prefill_batch`), in retry order.
#[cfg(test)]
pub(super) static RETRIED_LANES: std::sync::Mutex<Vec<(u32, usize)>> =
    std::sync::Mutex::new(Vec::new());

#[cfg(test)]
fn inject_forward_fault<T>(forward: Result<T>, lanes: &[Gemma4PrefillBatchLane]) -> Result<T> {
    use std::sync::atomic::Ordering;
    if forward.is_err() || !INJECT_FORWARD_FAULT.swap(false, Ordering::SeqCst) {
        return forward;
    }
    FAULTED_LANES.lock().expect("fault record").extend(
        lanes
            .iter()
            .map(|lane| (lane.handle.slot_id.0, lane.state.prompt_tokens.len())),
    );
    Err(anyhow::anyhow!(
        "injected non-fatal joint warm prefill fault"
    ))
}

fn parse_switch(raw: Option<&str>) -> std::result::Result<bool, String> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(JOINT_WARM_PREFILL_DEFAULT);
    };
    match raw.to_ascii_lowercase().as_str() {
        "1" | "on" | "true" => Ok(true),
        "0" | "off" | "false" => Ok(false),
        _ => Err(format!(
            "{JOINT_WARM_PREFILL_ENV} must be 1/on/true or 0/off/false, got {raw:?}"
        )),
    }
}

/// Resolve the switch once when the Gemma SlotAware worker starts. Diagnostic
/// settings that change the per-slot reference route keep it off.
pub(super) fn worker_enabled() -> bool {
    let raw = std::env::var(JOINT_WARM_PREFILL_ENV).ok();
    let requested = parse_switch(raw.as_deref()).unwrap_or_else(|error| {
        tracing::warn!(
            error,
            fallback = JOINT_WARM_PREFILL_DEFAULT,
            "invalid Gemma joint warm prefill switch"
        );
        JOINT_WARM_PREFILL_DEFAULT
    });
    let unavailable = if std::env::var("HF2Q_PREFILL_SLOT_BATCHED").as_deref() == Ok("0") {
        Some("HF2Q_PREFILL_SLOT_BATCHED=0 replaces the per-slot route lanes must equal")
    } else if std::env::var_os("HF2Q_MM_ID_ROUTING_THRESHOLD").is_some() {
        Some("HF2Q_MM_ID_ROUTING_THRESHOLD moves the MoE route boundary of the lane rule")
    } else {
        None
    };
    let active = requested && unavailable.is_none();
    tracing::info!(
        requested,
        active,
        unavailable,
        "Gemma joint warm prefill worker decision"
    );
    active
}

/// One lane's row rule: a reusable prefix and a suffix the joint stream
/// routes exactly like the per-slot prefill.
pub(super) fn lane_eligible(prompt_tokens: usize, cached_tokens: usize) -> bool {
    cached_tokens > 0
        && prompt_tokens <= ROW_CAP
        && prompt_tokens
            .checked_sub(cached_tokens)
            .is_some_and(|suffix| suffix >= MIN_SUFFIX_TOKENS)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PlannedLane {
    pub(super) slot: usize,
    pub(super) prompt_tokens: usize,
    pub(super) cached_tokens: usize,
}

/// Length of the joint cohort at the head of `lanes`: the longest prefix of
/// eligible lanes on distinct slots within `row_cap` suffix rows, or 0 when
/// fewer than two lanes qualify. `None` ends the prefix.
pub(super) fn plan_prefix(
    lanes: impl IntoIterator<Item = Option<PlannedLane>>,
    row_cap: usize,
) -> usize {
    let mut rows = 0usize;
    let mut claimed: Vec<usize> = Vec::new();
    for lane in lanes {
        let Some(lane) = lane else {
            break;
        };
        if !lane_eligible(lane.prompt_tokens, lane.cached_tokens) || claimed.contains(&lane.slot) {
            break;
        }
        let Some(next) = rows
            .checked_add(lane.prompt_tokens - lane.cached_tokens)
            .filter(|&next| next <= row_cap)
        else {
            break;
        };
        rows = next;
        claimed.push(lane.slot);
    }
    if claimed.len() >= 2 {
        claimed.len()
    } else {
        0
    }
}

/// The request shape a joint lane can take: plain text, one transaction.
fn warm_text_request(request: &Request) -> Option<(&[u32], &SamplingParams)> {
    let (prompt_tokens, params) = match request {
        Request::Generate {
            prompt_tokens,
            params,
            ..
        } => (prompt_tokens, params),
        Request::GenerateStream {
            prompt_tokens,
            params,
            soft_tokens,
            ..
        } if soft_tokens.is_empty() => (prompt_tokens, params),
        _ => return None,
    };
    (params.max_tokens > 0
        && u32::try_from(params.max_tokens).is_ok()
        && prompt_tokens.len() <= ROW_CAP
        && params.stable_prompt_prefix_tokens.is_none()
        && params.vision_fingerprint.is_none())
    .then_some((prompt_tokens.as_slice(), params))
}

/// Plan with an injected slot preference; `prefer` receives the slots
/// already claimed by earlier cohort lanes.
fn plan_requests(
    requests: &[Request],
    n_slots: usize,
    mut prefer: impl FnMut(&[u32], Option<[u8; 32]>, &[bool]) -> Option<Gemma4SlotPreference>,
) -> usize {
    let mut claimed = vec![false; n_slots];
    plan_prefix(
        requests.iter().map(|request| {
            let (prompt_tokens, params) = warm_text_request(request)?;
            let preference = prefer(prompt_tokens, params.vision_fingerprint, &claimed)?;
            let slot = preference.slot.0 as usize;
            *claimed.get_mut(slot)? = true;
            Some(PlannedLane {
                slot,
                prompt_tokens: prompt_tokens.len(),
                cached_tokens: preference.cached_tokens,
            })
        }),
        ROW_CAP,
    )
}

/// Joint cohort length at the head of one admission quantum.
pub(super) fn cohort_prefix_len(
    requests: &[Request],
    retained_tokens: &[Vec<u32>],
    prompt_anchors: &[Option<Gemma4PromptAnchor>],
    slots: &[Option<Gemma4Slot>],
) -> usize {
    plan_requests(
        requests,
        slots.len(),
        |prompt_tokens, vision_fingerprint, claimed| {
            preferred_gemma4_slot_for_request_excluding(
                retained_tokens,
                prompt_anchors,
                slots,
                prompt_tokens,
                vision_fingerprint,
                claimed,
            )
        },
    )
}

/// Admit a planned cohort in FIFO order, then prefill every installed lane.
/// A worker-fatal admission returns the requests not yet admitted.
#[allow(clippy::too_many_arguments)]
pub(super) fn admit_cohort(
    guard: &mut Gemma4KvGuard<'_>,
    scheduler: &mut InflightBatchedScheduler,
    slots: &mut [Option<Gemma4Slot>],
    retained_tokens: &mut [Vec<u32>],
    prompt_anchors: &mut [Option<Gemma4PromptAnchor>],
    registration: Option<&ModelRegistration>,
    scheduler_stats_snapshot: &Arc<Mutex<SchedulerStats>>,
    per_slot_kv_budget_bytes: u64,
    kv_bytes_per_token: u64,
    supervisor: &EngineSupervisor,
    requests: Vec<Request>,
) -> Option<(SlotAwareGpuFatal, Vec<Request>)> {
    let mut requests = requests.into_iter();
    while let Some(request) = requests.next() {
        let (prompt_tokens, params, reply) = match request {
            Request::Generate {
                prompt_tokens,
                params,
                reply,
            } => (prompt_tokens, params, SlotReply::Unary(reply)),
            Request::GenerateStream {
                prompt_tokens,
                params,
                events,
                cancellation_counter,
                admission,
                soft_tokens,
                ..
            } if soft_tokens.is_empty() => {
                let reply = SlotReply::family_stream(
                    events,
                    cancellation_counter,
                    admission,
                    registration,
                    &params,
                );
                (prompt_tokens, params, reply)
            }
            other => {
                fail_slotaware_buffered_request(
                    other,
                    "Gemma4 joint warm cohort planned a request it cannot admit",
                );
                continue;
            }
        };
        if let Some(fatal) = admit_gemma4_slot_routed(
            guard,
            scheduler,
            slots,
            retained_tokens,
            prompt_anchors,
            registration,
            scheduler_stats_snapshot,
            per_slot_kv_budget_bytes,
            kv_bytes_per_token,
            supervisor,
            prompt_tokens,
            Vec::new(),
            params,
            reply,
            true,
        ) {
            return Some((fatal, requests.collect()));
        }
    }
    run_installed(
        guard,
        scheduler,
        slots,
        retained_tokens,
        prompt_anchors,
        registration,
        scheduler_stats_snapshot,
        kv_bytes_per_token,
        supervisor,
    )
    .map(|fatal| (fatal, Vec::new()))
}

/// Next FIFO admission ordinal for an installed joint-warm lane. Cohorts
/// submit, capture, seed and (on failure) retry lanes in this order, so a
/// binding checkpoint quota sees captures in serial-admission order.
pub(super) fn next_admission_ordinal() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Drain every installed joint-warm lane before the scheduler steps: cohorts
/// of two or more run jointly, any remainder per slot, earliest admission
/// first. Each advance completes or removes its primary lane, so the loop
/// terminates.
#[allow(clippy::too_many_arguments)]
fn run_installed(
    guard: &mut Gemma4KvGuard<'_>,
    scheduler: &mut InflightBatchedScheduler,
    slots: &mut [Option<Gemma4Slot>],
    retained_tokens: &mut [Vec<u32>],
    prompt_anchors: &mut [Option<Gemma4PromptAnchor>],
    registration: Option<&ModelRegistration>,
    scheduler_stats_snapshot: &Arc<Mutex<SchedulerStats>>,
    kv_bytes_per_token: u64,
    supervisor: &EngineSupervisor,
) -> Option<SlotAwareGpuFatal> {
    while let Some((_, handle)) = slots
        .iter()
        .filter_map(|slot| match slot {
            Some((Gemma4SlotWork::Prefill(state), _, handle)) => {
                state.joint_warm.map(|ordinal| (ordinal, *handle))
            }
            _ => None,
        })
        .min_by_key(|(ordinal, _)| *ordinal)
    {
        if let Some(fatal) = advance_gemma4_prefill_quantum(
            guard,
            scheduler,
            slots,
            retained_tokens,
            prompt_anchors,
            registration,
            scheduler_stats_snapshot,
            handle,
            GEMMA4_SLOT_PREFILL_CHUNK_TOKENS,
            kv_bytes_per_token,
            supervisor,
            true,
        ) {
            return Some(fatal);
        }
    }
    None
}

/// Lanes of one joint transaction: the primary first, then the other
/// installed joint-warm lanes in FIFO admission order, while the cohort stays
/// within `row_cap` rows. `lanes[i]` is slot `i`'s `(admission ordinal,
/// unprefilled suffix rows)` when it is a live joint-warm lane. Fewer than
/// two entries: no cohort.
pub(super) fn select_installed(
    primary: usize,
    lanes: &[Option<(u64, usize)>],
    row_cap: usize,
) -> Vec<usize> {
    let eligible = |index: usize| {
        lanes[index].filter(|&(_, rows)| rows >= MIN_SUFFIX_TOKENS && rows <= row_cap)
    };
    if primary >= lanes.len() {
        return Vec::new();
    }
    let Some((_, mut rows)) = eligible(primary) else {
        return Vec::new();
    };
    let mut others: Vec<(u64, usize, usize)> = (0..lanes.len())
        .filter(|&index| index != primary)
        .filter_map(|index| eligible(index).map(|(ordinal, lane)| (ordinal, index, lane)))
        .collect();
    others.sort_unstable();
    let mut selected = vec![primary];
    for (_, index, lane) in others {
        let Some(next) = rows.checked_add(lane).filter(|&next| next <= row_cap) else {
            continue;
        };
        rows = next;
        selected.push(index);
    }
    selected
}

/// Whole-suffix plans for the selected lanes, or the reason the cohort must
/// not be submitted.
fn submission_plans(
    guard: &Gemma4KvGuard<'_>,
    slots: &[Option<Gemma4Slot>],
    selected: &[usize],
) -> Result<Vec<Gemma4PrefillPlan>> {
    anyhow::ensure!(
        guard.hybrid.is_some()
            && crate::debug::INVESTIGATION_ENV.hybrid_kv
            && std::env::var("HF2Q_DFLASH_XLEN_SDPA").as_deref() != Ok("1"),
        "joint warm prefill requires the production hybrid KV substrate"
    );
    selected
        .iter()
        .map(|&index| {
            let Some((Gemma4SlotWork::Prefill(state), _, installed)) = slots[index].as_ref() else {
                anyhow::bail!("slot {index} is not an installed prefill");
            };
            let rows = state
                .prompt_tokens
                .len()
                .saturating_sub(state.progress.cursor());
            let plan = state.progress.plan(rows.max(1))?;
            anyhow::ensure!(
                plan.ready
                    && !plan.capture_anchor
                    && plan.start > 0
                    && plan.end - plan.start >= MIN_SUFFIX_TOKENS,
                "slot {index} plan {plan:?} is not one whole warm suffix"
            );
            validate_gemma4_slot_cursor(guard, installed.slot_id, plan.start)?;
            Ok(plan)
        })
        .collect()
}

/// Advance the installed joint-warm lane `handle` (the scheduler/admission
/// primary): jointly with every other eligible installed joint-warm lane,
/// or per slot when no cohort forms or a pre-submission check fails.
#[allow(clippy::too_many_arguments)]
pub(super) fn advance(
    guard: &mut Gemma4KvGuard<'_>,
    scheduler: &mut InflightBatchedScheduler,
    slots: &mut [Option<Gemma4Slot>],
    retained_tokens: &mut [Vec<u32>],
    prompt_anchors: &mut [Option<Gemma4PromptAnchor>],
    registration: Option<&ModelRegistration>,
    scheduler_stats_snapshot: &Arc<Mutex<SchedulerStats>>,
    handle: SlotHandle,
    kv_bytes_per_token: u64,
    supervisor: &EngineSupervisor,
) -> Option<SlotAwareGpuFatal> {
    // The per-slot route is today's warm prefill: one whole-suffix
    // `forward_prefill_with_soft_tokens_slot_aware_resume` transaction.
    macro_rules! per_slot {
        () => {
            return advance_gemma4_prefill(
                guard,
                scheduler,
                slots,
                retained_tokens,
                prompt_anchors,
                registration,
                scheduler_stats_snapshot,
                handle,
                GEMMA4_SLOT_PREFILL_CHUNK_TOKENS,
                kv_bytes_per_token,
                supervisor,
            )
        };
    }
    let candidates: Vec<Option<(u64, usize)>> = slots
        .iter()
        .enumerate()
        .map(|(index, slot)| match slot {
            Some((Gemma4SlotWork::Prefill(state), reply, installed))
                if installed.slot_id.0 as usize == index
                    && state.soft_token_data.is_empty()
                    && !reply.client_closed() =>
            {
                let rows = state
                    .prompt_tokens
                    .len()
                    .checked_sub(state.progress.cursor())?;
                state.joint_warm.map(|ordinal| (ordinal, rows))
            }
            _ => None,
        })
        .collect();
    let selected = select_installed(handle.slot_id.0 as usize, &candidates, ROW_CAP);
    if selected.len() < 2 {
        per_slot!();
    }
    let plans = match submission_plans(guard, slots, &selected) {
        Ok(plans) => plans,
        Err(error) => {
            tracing::warn!(
                slot = handle.slot_id.0,
                error = %format!("{error:#}"),
                "Gemma4 joint warm prefill pre-submission check failed; prefilling per slot"
            );
            per_slot!();
        }
    };

    let mut lanes: Vec<Gemma4PrefillBatchLane> = Vec::with_capacity(selected.len());
    for (&index, plan) in selected.iter().zip(plans) {
        match slots[index].take() {
            Some((Gemma4SlotWork::Prefill(state), reply, installed)) => {
                lanes.push(Gemma4PrefillBatchLane {
                    handle: installed,
                    state,
                    reply,
                    plan,
                });
            }
            other => {
                slots[index] = other;
                return Some(gemma4_prefill_batch_fatal(
                    lanes,
                    anyhow::anyhow!("Gemma4 joint warm lane {index} changed before submission"),
                    SlotAwareFatalKind::Invariant,
                ));
            }
        }
    }

    let seqs: Vec<(Vec<u32>, SlotId, usize)> = lanes
        .iter()
        .map(|lane| {
            (
                lane.state.prompt_tokens[lane.plan.start..lane.plan.end].to_vec(),
                lane.handle.slot_id,
                lane.plan.start,
            )
        })
        .collect();
    // Every lane was validated for its own `prompt + max_tokens`; reserve only
    // what the binding lane needs beyond the longest lane's end.
    let max_decode_tokens = gemma4_batch_decode_reserve(
        lanes
            .iter()
            .map(|lane| (lane.plan.end, lane.state.params.max_tokens)),
    );
    let _native_invocation = crate::serve::phase_trace::enabled().then(|| {
        scheduler.trace_transaction(
            "gemma4",
            "actual_joint_warm_prefill_inputs",
            &lanes.iter().map(|lane| lane.handle).collect::<Vec<_>>(),
        )
    });
    clear_gemma4_self_mounts(guard.model);
    let started = Instant::now();
    let forward = match guard.hybrid.as_deref() {
        Some(scaffold) => {
            supervised_gemma4_gpu_call(supervisor, "gemma4_joint_warm_prefill", || {
                guard.model.weights.forward_prefill_batched_multi_seq_live(
                    &seqs,
                    scaffold,
                    max_decode_tokens,
                    GemmaLiveAttention::PerSlotExact,
                    &mut guard.model.ctx,
                )
            })
        }
        None => Err(anyhow::anyhow!(
            "Gemma4 joint warm prefill lost its hybrid scaffold"
        )),
    };
    let elapsed = started.elapsed();
    clear_gemma4_self_mounts(guard.model);
    #[cfg(test)]
    let forward = inject_forward_fault(forward, &lanes);
    #[cfg(test)]
    if forward.is_ok() {
        JOINT_WARM_TRANSACTIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        JOINT_WARM_LANES.fetch_add(lanes.len(), std::sync::atomic::Ordering::Relaxed);
    }
    settle_gemma4_prefill_batch(
        guard,
        scheduler,
        slots,
        retained_tokens,
        prompt_anchors,
        registration,
        scheduler_stats_snapshot,
        lanes,
        forward,
        elapsed,
        kv_bytes_per_token,
        supervisor,
        "joint warm",
        true,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four-agent tool-turn shape: histories and suffixes per agent.
    const HISTORIES: [usize; 4] = [465, 714, 943, 1254];
    const SUFFIXES: [usize; 4] = [56, 200, 311, 385];

    fn history(agent: usize, len: usize) -> Vec<u32> {
        (0..len as u32).map(|t| agent as u32 * 10_000 + t).collect()
    }

    fn prompt(agent: usize, history_len: usize, suffix: usize) -> Vec<u32> {
        let mut prompt = history(agent, history_len);
        prompt.extend((0..suffix as u32).map(|t| 90_000 + agent as u32 * 1_000 + t));
        prompt
    }

    fn generate(prompt_tokens: Vec<u32>) -> Request {
        let (reply, _) = oneshot::channel();
        Request::Generate {
            prompt_tokens,
            params: SamplingParams {
                max_tokens: 16,
                ..SamplingParams::default()
            },
            reply,
        }
    }

    /// Plan against CPU-only checkpoint owners through the production
    /// affinity primitive (no GPU cache is needed to test selection).
    fn plan(requests: &[Request], anchors: &[Vec<u32>]) -> usize {
        plan_requests(requests, anchors.len(), |prompt, _, claimed| {
            gemma4_slot_affinity(
                anchors.iter().enumerate().map(|(index, anchor)| {
                    (
                        index,
                        &[][..],
                        Some(anchor.as_slice()),
                        None,
                        false,
                        claimed[index],
                    )
                }),
                prompt,
            )
            .map(Gemma4SlotAffinity::preference)
        })
    }

    fn anchors() -> Vec<Vec<u32>> {
        HISTORIES
            .iter()
            .enumerate()
            .map(|(agent, &len)| history(agent, len))
            .collect()
    }

    #[test]
    fn gemma_joint_warm_four_tool_turns_form_one_cohort() {
        let requests: Vec<Request> = (0..4)
            .map(|agent| generate(prompt(agent, HISTORIES[agent], SUFFIXES[agent])))
            .collect();
        assert_eq!(plan(&requests, &anchors()), 4);
        // Two and three same-step lanes form smaller cohorts.
        assert_eq!(plan(&requests[..2], &anchors()), 2);
        assert_eq!(plan(&requests[1..], &anchors()), 3);
    }

    #[test]
    fn gemma_joint_warm_requires_more_than_32_suffix_rows_per_lane() {
        let lanes = |third_suffix: usize| {
            vec![
                generate(prompt(0, HISTORIES[0], SUFFIXES[0])),
                generate(prompt(1, HISTORIES[1], SUFFIXES[1])),
                generate(prompt(2, HISTORIES[2], third_suffix)),
                generate(prompt(3, HISTORIES[3], SUFFIXES[3])),
            ]
        };
        // 32 rows would take the MoE mv_id route per slot but mm_id jointly:
        // the cohort stops before that lane, which then admits per slot.
        assert_eq!(plan(&lanes(32), &anchors()), 2);
        assert_eq!(plan(&lanes(33), &anchors()), 4);
        let first_tiny = vec![
            generate(prompt(0, HISTORIES[0], 5)),
            generate(prompt(1, HISTORIES[1], SUFFIXES[1])),
        ];
        assert_eq!(plan(&first_tiny, &anchors()), 0);
        assert!(!lane_eligible(465 + 32, 465));
        assert!(lane_eligible(465 + 33, 465));
    }

    #[test]
    fn gemma_joint_warm_cold_or_non_text_lane_ends_the_prefix_in_fifo_order() {
        let cold_first = vec![
            generate((500_000..500_100).collect()),
            generate(prompt(1, HISTORIES[1], SUFFIXES[1])),
            generate(prompt(2, HISTORIES[2], SUFFIXES[2])),
        ];
        assert_eq!(plan(&cold_first, &anchors()), 0);
        let control_third = vec![
            generate(prompt(0, HISTORIES[0], SUFFIXES[0])),
            generate(prompt(1, HISTORIES[1], SUFFIXES[1])),
            Request::Shutdown,
            generate(prompt(3, HISTORIES[3], SUFFIXES[3])),
        ];
        assert_eq!(plan(&control_third, &anchors()), 2);
        let mut cue_split = generate(prompt(2, HISTORIES[2], SUFFIXES[2]));
        if let Request::Generate { params, .. } = &mut cue_split {
            params.stable_prompt_prefix_tokens = Some(HISTORIES[2]);
        }
        let stable_second = vec![
            generate(prompt(0, HISTORIES[0], SUFFIXES[0])),
            cue_split,
            generate(prompt(3, HISTORIES[3], SUFFIXES[3])),
        ];
        assert_eq!(
            plan(&stable_second, &anchors()),
            0,
            "a stable cue split stays in the stable batch class"
        );
    }

    #[test]
    fn gemma_joint_warm_cohort_stays_within_the_4096_row_cap() {
        let lanes = |suffixes: &[usize]| -> Vec<Request> {
            suffixes
                .iter()
                .enumerate()
                .map(|(agent, &suffix)| generate(prompt(agent, 465, suffix)))
                .collect()
        };
        let anchors: Vec<Vec<u32>> = (0..4).map(|agent| history(agent, 465)).collect();
        // Prompts must also stay single-transaction (<= 4096 rows).
        assert_eq!(plan(&lanes(&[2048, 2048]), &anchors), 2);
        assert_eq!(plan(&lanes(&[2048, 2049]), &anchors), 0);
        assert_eq!(plan(&lanes(&[2000, 2000, 200]), &anchors), 2);
        assert_eq!(plan(&lanes(&[3632, 100]), &anchors), 0, "prompt above cap");
    }

    #[test]
    fn gemma_joint_warm_lanes_claim_distinct_slots() {
        // The same conversation twice: the second request cannot reuse the
        // first lane's slot, and no other slot holds its prefix.
        let requests = vec![
            generate(prompt(0, HISTORIES[0], SUFFIXES[0])),
            generate(prompt(0, HISTORIES[0], SUFFIXES[1])),
        ];
        assert_eq!(plan(&requests, &anchors()), 0);
        assert_eq!(
            plan_prefix(
                [
                    Some(PlannedLane {
                        slot: 1,
                        prompt_tokens: 600,
                        cached_tokens: 465,
                    }),
                    Some(PlannedLane {
                        slot: 1,
                        prompt_tokens: 700,
                        cached_tokens: 465,
                    }),
                ],
                ROW_CAP,
            ),
            0
        );
    }

    #[test]
    fn gemma_joint_warm_installed_selection_is_fifo_and_fits_the_cap() {
        // Slot order 1,2,3,0 differs from admission order 1,3,0,2.
        let lanes = [
            Some((2, 56)),
            Some((0, 200)),
            Some((3, 311)),
            Some((1, 385)),
        ];
        assert_eq!(select_installed(1, &lanes, ROW_CAP), vec![1, 3, 0, 2]);
        let with_gap = [Some((7, 56)), Some((5, 200)), None, Some((6, 385))];
        assert_eq!(select_installed(1, &with_gap, ROW_CAP), vec![1, 3, 0]);
        assert_eq!(select_installed(2, &with_gap, ROW_CAP), Vec::<usize>::new());
        assert_eq!(
            select_installed(0, &[Some((0, 56)), Some((1, 32)), Some((2, 311))], ROW_CAP),
            vec![0, 2]
        );
        assert_eq!(
            select_installed(
                0,
                &[Some((0, 3000)), Some((1, 2000)), Some((2, 1000))],
                ROW_CAP
            ),
            vec![0, 2]
        );
        assert_eq!(select_installed(0, &[Some((0, 56))], ROW_CAP), vec![0]);
    }

    #[test]
    fn gemma_joint_warm_decode_reserve_fits_mixed_max_tokens_near_context() {
        // Context 32,768: the full-layer slot capacity is ctx + 1 rows. Each
        // lane fits alone; pairing the largest max_tokens with the longest
        // lane would demand 1,700 + 31,700 = 33,400 rows and fail the cohort.
        let capacity = 32_768 + 1;
        let lanes = [(1_000usize, 31_700usize), (1_700, 100)];
        for (end, max_tokens) in lanes {
            assert!(
                crate::serve::forward_prefill::gemma_flash_attn_vec_capacity(end + max_tokens)
                    <= capacity
            );
        }
        let reserve = gemma4_batch_decode_reserve(lanes);
        let max_end = 1_700;
        assert_eq!(max_end + reserve, 1_000 + 31_700, "binding lane's own need");
        assert!(
            crate::serve::forward_prefill::gemma_flash_attn_vec_capacity(max_end + reserve)
                <= capacity
        );
        let naive = lanes
            .iter()
            .map(|&(_, max_tokens)| max_tokens)
            .max()
            .unwrap();
        assert!(
            crate::serve::forward_prefill::gemma_flash_attn_vec_capacity(max_end + naive)
                > capacity,
            "the old reserve fails this cohort"
        );
        // Same lane order independence, the per-slot minimum of one decode
        // row, and a single lane's own value.
        assert_eq!(
            gemma4_batch_decode_reserve([(1_700, 100), (1_000, 31_700)]),
            reserve
        );
        assert_eq!(gemma4_batch_decode_reserve([(500, 0), (400, 0)]), 1);
        assert_eq!(gemma4_batch_decode_reserve([(500, 16)]), 16);
        assert_eq!(gemma4_batch_decode_reserve(std::iter::empty()), 1);
    }

    #[test]
    fn gemma_joint_warm_switch_defaults_on_with_a_kill_switch() {
        assert_eq!(parse_switch(None), Ok(true));
        assert_eq!(parse_switch(Some(" ")), Ok(true));
        for on in ["1", "on", "TRUE"] {
            assert_eq!(parse_switch(Some(on)), Ok(true));
        }
        for off in ["0", "off", "False"] {
            assert_eq!(parse_switch(Some(off)), Ok(false));
        }
        assert!(parse_switch(Some("2")).is_err());
    }
}
