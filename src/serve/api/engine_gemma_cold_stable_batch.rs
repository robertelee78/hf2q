//! ADR-063 Q4-GEMMA-COLD: one exact joint prefill for simultaneous cold Gemma
//! 4 first turns (design `docs/research/adr063-q4-gemma-cold-design.md`).
//!
//! Every four-agent turn-0 request carries an exact pre-generation boundary
//! (`S < P`) and no reusable slot prefix. Admitted per slot, each lane
//! prefills inline before the next one is admitted, so no lane decodes until
//! the last one is prefilled. A cold stable cohort (`gemma_cold_batch::
//! stable_plan`) runs instead as one transaction:
//!
//! ```text
//! reserve (FIFO) → reset → Phase B: joint boundary prefill → anchors (FIFO)
//!     → Phase C: joint cue → commit cursors (0, P) → publish (FIFO)
//! ```
//!
//! * Phase B is one `forward_prefill_batched_multi_seq` over every lane's
//!   boundary rows; attention and the O-projection run per lane through the
//!   shared per-sequence cold attention (`gemma_cold_attention`), so each lane
//!   equals its own single-slot boundary prefill.
//! * Each lane's `StableBoundary` anchor is captured in FIFO order, exactly
//!   where per-slot admission captures it. The planner refused the cohort
//!   unless the serial order's anchor reservations all fit.
//! * Phase C: cues shorter than 32 rows replay through the batched decode
//!   body (byte-identical to N serial `forward_decode_slot_aware` calls),
//!   one step per cue position, logits read from the decode tick's own
//!   source; cues of 33 rows or more (all lanes) take one joint live resume
//!   with `GemmaLiveAttention::PerSlotExact`; a 32-row cue, or a long cue
//!   beside a short one, runs the per-slot resume inside the transaction.
//!
//! Failure semantics (design §4.5):
//! * pre-submission (reservation, a lane that turns out warm at its serial
//!   planning point): reservations are released and every lane runs per slot
//!   in FIFO order; a failed reset is an invariant fatal;
//! * Phase B non-fatal forward error: every lane is reset and released, then
//!   admitted per slot in FIFO order, exactly as if it had been admitted
//!   alone (`gemma_joint_cold_forward_fault_retries_each_lane_per_slot`);
//! * anchor capture error other than a quota refusal: that lane is reset and
//!   failed request-locally; the others continue;
//! * Phase C non-fatal error: each lane is restored from its just-captured
//!   boundary anchor and finishes its cue per slot in FIFO order, keeping its
//!   reservation; a lane whose anchor was refused is reset, released and
//!   admitted afresh per slot (`…_cue_fault_restores_from_boundary_anchors`);
//! * worker-fatal GPU error: the worker fails, as for every Gemma
//!   transaction;
//! * publication failure of one lane: that lane is recovered and failed
//!   only;
//! * cancellation: a client closed before reservation or reset is dropped and
//!   cancelled; one that closes during the transaction is recovered after
//!   publication from its published anchor and released.
use super::*;
use crate::serve::api::registry::ModelRegistration;
use gemma_cold_batch::{CueRoute, StableLane, StablePlan};

#[cfg(test)]
pub(super) static COLD_STABLE_COHORTS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(super) static COLD_STABLE_LANES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
/// Fault injection (tests only): the next cohort's Phase B forward keeps its
/// real KV writes but reports a non-fatal error. Consumed by one transaction.
#[cfg(test)]
pub(super) static INJECT_COLD_FORWARD_FAULT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
/// Fault injection (tests only): the next cohort's Phase C reports a
/// non-fatal error after its real cue writes. Consumed by one transaction.
#[cfg(test)]
pub(super) static INJECT_COLD_CUE_FAULT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
/// `(slot, prompt tokens)` of every lane retried per slot after a Phase B
/// fault, in retry order.
#[cfg(test)]
pub(super) static RETRIED_LANES: std::sync::Mutex<Vec<(u32, usize)>> =
    std::sync::Mutex::new(Vec::new());
/// `(slot, prompt tokens)` of every lane restored from its boundary anchor
/// after a Phase C fault, in restore order.
#[cfg(test)]
pub(super) static CUE_RESTORED_LANES: std::sync::Mutex<Vec<(u32, usize)>> =
    std::sync::Mutex::new(Vec::new());
/// `(slot, prompt tokens)` of the last cohort's lanes in FIFO reservation
/// order, recorded before any reset.
#[cfg(test)]
pub(super) static LAST_COHORT_FIFO: std::sync::Mutex<Vec<(u32, usize)>> =
    std::sync::Mutex::new(Vec::new());
/// Test hook: runs once, between the reservation and the reset (closes a
/// client "during the transaction").
#[cfg(test)]
pub(super) static BEFORE_PHASE_B: std::sync::Mutex<Option<Box<dyn FnOnce() + Send>>> =
    std::sync::Mutex::new(None);

#[cfg(test)]
fn inject<T>(armed: &std::sync::atomic::AtomicBool, result: Result<T>, what: &str) -> Result<T> {
    if result.is_err() || !armed.swap(false, std::sync::atomic::Ordering::SeqCst) {
        return result;
    }
    Err(anyhow::anyhow!(
        "injected non-fatal joint cold {what} fault"
    ))
}

/// One replay step of the short cues (design §4.4): `(lane, position, last)`
/// for every lane with a cue row at this step, in FIFO order. `last` marks
/// the step whose logits row seeds the lane.
pub(super) fn replay_schedule(lanes: &[(usize, usize)]) -> Vec<Vec<(usize, usize, bool)>> {
    let steps = lanes.iter().map(|&(_, cue)| cue).max().unwrap_or(0);
    (0..steps)
        .map(|step| {
            lanes
                .iter()
                .enumerate()
                .filter(|(_, (_, cue))| step < *cue)
                .map(|(lane, &(boundary, cue))| (lane, boundary + step, step + 1 == cue))
                .collect()
        })
        .collect()
}

/// The slot preference per-slot admission computes for a lane after the
/// earlier cohort lanes were admitted: each earlier lane is the active owner
/// of its slot with its boundary anchor published and its live ledger
/// cleared, exactly the state `admit_gemma4_slot` leaves before the next
/// serial lane is planned.
fn serial_preference(
    retained_tokens: &[Vec<u32>],
    prompt_anchors: &[Option<Gemma4PromptAnchor>],
    prompt: &[u32],
    earlier: &[(usize, &[u32], usize)],
) -> Option<Gemma4SlotPreference> {
    match gemma4_slot_affinity(
        retained_tokens.iter().zip(prompt_anchors).enumerate().map(
            |(index, (retained, anchor))| {
                if let Some(&(_, earlier_prompt, boundary)) =
                    earlier.iter().find(|(slot, _, _)| *slot == index)
                {
                    let none: &[u32] = &[];
                    return (
                        index,
                        none,
                        Some(&earlier_prompt[..boundary]),
                        Some(earlier_prompt),
                        true,
                        false,
                    );
                }
                // Text lanes: an anchor is usable only without a vision
                // identity (`preferred_gemma4_slot_for_request`).
                let identity = anchor
                    .as_ref()
                    .is_some_and(|anchor| anchor.vision_fingerprint.is_none());
                (
                    index,
                    if identity { retained.as_slice() } else { &[] },
                    anchor
                        .as_ref()
                        .filter(|_| identity)
                        .map(|anchor| anchor.prompt_tokens.as_slice()),
                    None,
                    false,
                    false,
                )
            },
        ),
        prompt,
    ) {
        Some(Gemma4SlotAffinity::Idle(preference)) => Some(preference),
        Some(Gemma4SlotAffinity::Active(_)) | None => None,
    }
}

/// The scheduler reservation for one cohort lane at its serial planning
/// point: exactly the `AdmitRequest` `admit_gemma4_slot_routed` issues for a
/// cold lane (no preferred slot). `None` when the lane is warm at that point
/// or no longer a valid request, so the whole group goes per slot.
#[allow(clippy::too_many_arguments)]
fn cohort_admit_request(
    retained_tokens: &[Vec<u32>],
    prompt_anchors: &[Option<Gemma4PromptAnchor>],
    prompt: &[u32],
    params: &SamplingParams,
    earlier: &[(usize, &[u32], usize)],
    max_seq_len: usize,
    graft_len: usize,
    per_slot_kv_budget_bytes: u64,
    kv_bytes_per_token: u64,
) -> Option<AdmitRequest> {
    if serial_preference(retained_tokens, prompt_anchors, prompt, earlier).is_some() {
        return None;
    }
    let shape =
        validate_gemma4_generation_request(prompt.len(), params.max_tokens, max_seq_len, graft_len)
            .ok()?;
    Some(gemma4_slot_admit_request(
        shape,
        prompt.len(),
        graft_len,
        per_slot_kv_budget_bytes,
        kv_bytes_per_token,
        None,
    ))
}

/// One reserved cohort lane. Fields are visible to the real-artifact gates,
/// which drive the transaction phases directly.
pub(super) struct Lane {
    pub(super) handle: SlotHandle,
    pub(super) prompt: Vec<u32>,
    pub(super) params: SamplingParams,
    pub(super) reply: SlotReply,
    pub(super) shape: StableLane,
    pub(super) anchor: Option<Gemma4PromptAnchor>,
    /// First token and its logits row, once the cue is processed.
    pub(super) seed: Option<(u32, Vec<f32>)>,
}

/// Per-slot admission of `requests`, in order (today's path).
#[allow(clippy::too_many_arguments)]
fn admit_per_slot(
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
    requests: impl IntoIterator<Item = (Vec<u32>, SamplingParams, SlotReply)>,
) -> Option<SlotAwareGpuFatal> {
    let mut requests = requests.into_iter();
    while let Some((prompt_tokens, params, reply)) = requests.next() {
        if let Some(mut fatal) = admit_gemma4_slot(
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
        ) {
            fatal.extend_slots(requests.map(|(_, _, reply)| (None, reply)));
            return Some(fatal);
        }
    }
    None
}

fn fatal_lanes(
    error: anyhow::Error,
    kind: SlotAwareFatalKind,
    lanes: impl IntoIterator<Item = Lane>,
) -> SlotAwareGpuFatal {
    let mut fatal = SlotAwareGpuFatal {
        error,
        kind,
        owned: Vec::new(),
    };
    fatal.extend_slots(
        lanes
            .into_iter()
            .map(|lane| (Some(lane.handle), lane.reply)),
    );
    fatal
}

/// Release a reserved lane whose slot holds no request state.
fn release_unused(scheduler: &mut InflightBatchedScheduler, handle: SlotHandle) {
    scheduler.record_slot_high_water(handle, 0);
    scheduler.release(handle);
}

/// Run a planned cold stable cohort (design §3.2). `requests` and
/// `plan.lanes` are in FIFO order and of equal length.
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
    requests: Vec<(Vec<u32>, SamplingParams, SlotReply)>,
    plan: StablePlan,
) -> Option<SlotAwareGpuFatal> {
    debug_assert_eq!(requests.len(), plan.lanes.len());
    macro_rules! per_slot {
        ($requests:expr) => {
            admit_per_slot(
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
                $requests,
            )
        };
    }
    let max_seq_len = guard
        .model
        .context_length
        .unwrap_or(guard.model.config.max_position_embeddings as usize);
    let graft_len = guard.model.weights.kv_graft_len();

    // 1. Reserve, FIFO: the reservation `admit_gemma4_slot_routed` issues for
    //    each lane, planned at its serial admission point.
    let mut lanes: Vec<Lane> = Vec::with_capacity(requests.len());
    let mut pending = requests.into_iter().zip(plan.lanes);
    while let Some(((prompt, params, reply), shape)) = pending.next() {
        if reply.client_closed() {
            reply.record_client_cancellation();
            slot_fire_done(
                reply,
                Err(anyhow::anyhow!(
                    "client disconnected before Gemma4 admission"
                )),
                true,
            );
            continue;
        }
        // Per-slot admission fails an unusable grammar before reserving.
        if let Err(error) = validate_gemma4_request_grammar(&params) {
            slot_fire_done(reply, Err(error), false);
            continue;
        }
        let earlier: Vec<(usize, &[u32], usize)> = lanes
            .iter()
            .map(|lane| {
                (
                    lane.handle.slot_id.0 as usize,
                    lane.prompt.as_slice(),
                    lane.shape.boundary,
                )
            })
            .collect();
        let Some(request) = cohort_admit_request(
            retained_tokens,
            prompt_anchors,
            &prompt,
            &params,
            &earlier,
            max_seq_len,
            graft_len,
            per_slot_kv_budget_bytes,
            kv_bytes_per_token,
        ) else {
            // At its serial planning point this lane is warm (or no longer
            // valid): nothing was reset, so today's path runs every lane.
            for lane in &lanes {
                release_unused(scheduler, lane.handle);
            }
            let replay = lanes
                .into_iter()
                .map(|lane| (lane.prompt, lane.params, lane.reply))
                .chain(std::iter::once((prompt, params, reply)))
                .chain(pending.map(|((prompt, params, reply), _)| (prompt, params, reply)));
            return per_slot!(replay.collect::<Vec<_>>());
        };
        match scheduler.admit(request) {
            Ok(admitted) => match admitted.handle {
                Some(handle) => lanes.push(Lane {
                    handle,
                    prompt,
                    params,
                    reply,
                    shape,
                    anchor: None,
                    seed: None,
                }),
                None => slot_fire_done(
                    reply,
                    Err(EngineRequestError::invalid_request(
                        "max_tokens must be greater than zero",
                    )
                    .into_anyhow()),
                    false,
                ),
            },
            Err(error) => slot_fire_done(reply, Err(gemma4_slot_admit_error(error)), false),
        }
    }
    if let Ok(mut stats) = scheduler_stats_snapshot.lock() {
        *stats = scheduler.stats();
    }
    // A stream can close while later lanes are reserved: drop it before any
    // reset, publishing zero physical growth before release.
    let mut live = Vec::with_capacity(lanes.len());
    for lane in lanes {
        if lane.reply.client_closed() {
            lane.reply.record_client_cancellation();
            release_unused(scheduler, lane.handle);
            slot_fire_done(
                lane.reply,
                Err(anyhow::anyhow!("client disconnected before Gemma4 prefill")),
                true,
            );
        } else {
            live.push(lane);
        }
    }
    let mut lanes = live;
    if lanes.len() < 2 {
        for lane in &lanes {
            release_unused(scheduler, lane.handle);
        }
        return per_slot!(lanes
            .into_iter()
            .map(|lane| (lane.prompt, lane.params, lane.reply))
            .collect::<Vec<_>>());
    }

    let handles: Vec<SlotHandle> = lanes.iter().map(|lane| lane.handle).collect();
    #[cfg(test)]
    {
        *LAST_COHORT_FIFO.lock().expect("cold cohort FIFO record") = lanes
            .iter()
            .map(|lane| (lane.handle.slot_id.0, lane.prompt.len()))
            .collect();
    }
    let _native_invocation = scheduler.trace_transaction("gemma4", "cold_stable_cohort", &handles);
    for lane in &mut lanes {
        lane.reply.accept_stream_admission();
        let request_id = gemma4_operator_request_id(lane.handle);
        crate::serve::operator_ui::request_started(
            "gemma4",
            request_id,
            Some(lane.handle.slot_id.0),
            if matches!(lane.reply, SlotReply::Stream { .. }) {
                "slot-stream-cold-cohort"
            } else {
                "slot-unary-cold-cohort"
            },
            lane.prompt.len(),
            lane.params.max_tokens,
        );
        crate::serve::operator_ui::prefill_planned("gemma4", request_id, 0, lane.prompt.len());
    }
    #[cfg(test)]
    if let Some(hook) = BEFORE_PHASE_B.lock().expect("cold cohort hook").take() {
        hook();
    }

    // 2. Reset every lane, then clear the self-mounts (as `prefill_seed`).
    for lane in &lanes {
        let slot = lane.handle.slot_id.0 as usize;
        retained_tokens[slot].clear();
        prompt_anchors[slot] = None;
    }
    if let Some(error) = lanes
        .iter()
        .find_map(|lane| reset_gemma4_slot(guard, scheduler, lane.handle, kv_bytes_per_token).err())
    {
        return Some(fatal_lanes(error, SlotAwareFatalKind::Invariant, lanes));
    }
    clear_gemma4_self_mounts(guard.model);

    // 3. Phase B: one joint prefill of every lane's boundary rows.
    let started = Instant::now();
    let boundary = boundary_forward(guard, supervisor, &lanes);
    #[cfg(test)]
    let boundary = inject(&INJECT_COLD_FORWARD_FAULT, boundary, "boundary forward");
    if let Err(error) = boundary {
        if is_worker_fatal(supervisor, &error) {
            return Some(fatal_lanes(error, SlotAwareFatalKind::Gpu, lanes));
        }
        tracing::warn!(
            error = %error,
            lanes = lanes.len(),
            "Gemma4 cold stable cohort boundary forward failed; admitting each lane per slot"
        );
        // A cold lane's request-start state is the reset slot.
        if let Some(error) = lanes.iter().find_map(|lane| {
            reset_gemma4_slot(guard, scheduler, lane.handle, kv_bytes_per_token).err()
        }) {
            return Some(fatal_lanes(error, SlotAwareFatalKind::Invariant, lanes));
        }
        for lane in &lanes {
            release_unused(scheduler, lane.handle);
            #[cfg(test)]
            RETRIED_LANES
                .lock()
                .expect("cold retry record")
                .push((lane.handle.slot_id.0, lane.prompt.len()));
        }
        return per_slot!(lanes
            .into_iter()
            .map(|lane| (lane.prompt, lane.params, lane.reply))
            .collect::<Vec<_>>());
    }
    let boundary_elapsed = started.elapsed();

    // 4. Anchors, FIFO, where per-slot admission captures them.
    let mut captured = Vec::with_capacity(lanes.len());
    let mut remaining = lanes.into_iter();
    while let Some(mut lane) = remaining.next() {
        match capture_boundary_anchor(guard, &lane) {
            Ok(anchor) => {
                lane.anchor = anchor;
                captured.push(lane);
            }
            Err(error) => {
                let slot = lane.handle.slot_id.0 as usize;
                let reply = match recover_gemma4_slot_after_cancellation_for_reply(
                    guard,
                    scheduler,
                    lane.handle,
                    kv_bytes_per_token,
                    &mut retained_tokens[slot],
                    &mut prompt_anchors[slot],
                    None,
                    false,
                    lane.reply,
                ) {
                    Ok(reply) => reply,
                    Err(mut fatal) => {
                        fatal.extend_slots(
                            captured
                                .into_iter()
                                .chain(remaining)
                                .map(|lane: Lane| (Some(lane.handle), lane.reply)),
                        );
                        return Some(fatal);
                    }
                };
                scheduler.release(lane.handle);
                slot_fire_done(reply, Err(error), false);
            }
        }
    }
    let mut lanes = captured;
    let anchor_elapsed = started.elapsed() - boundary_elapsed;

    // 5. Phase C: every lane's generation cue.
    let cue = encode_cues(guard, supervisor, &mut lanes);
    #[cfg(test)]
    let cue = inject(&INJECT_COLD_CUE_FAULT, cue, "cue");
    clear_gemma4_self_mounts(guard.model);
    if let Err(error) = cue {
        if is_worker_fatal(supervisor, &error) {
            return Some(fatal_lanes(error, SlotAwareFatalKind::Gpu, lanes));
        }
        tracing::warn!(
            error = %error,
            lanes = lanes.len(),
            "Gemma4 cold stable cohort cue failed; finishing each lane's cue per slot"
        );
        let (restored, fatal) = recover_cues(
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
            lanes,
        );
        if fatal.is_some() {
            return fatal;
        }
        lanes = restored;
    }
    let elapsed = started.elapsed();
    let cue_elapsed = elapsed - boundary_elapsed - anchor_elapsed;
    if lanes.is_empty() {
        return None;
    }

    // 6. Commit (0, P) all-or-nothing: no forward moved the ledger cursor.
    let commits: Vec<(SlotId, usize, usize)> = lanes
        .iter()
        .map(|lane| (lane.handle.slot_id, 0, lane.shape.prompt_len))
        .collect();
    if let Err(error) = commit_gemma4_slot_cursors(guard, &commits) {
        return Some(fatal_lanes(error, SlotAwareFatalKind::Invariant, lanes));
    }
    #[cfg(test)]
    {
        COLD_STABLE_COHORTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        COLD_STABLE_LANES.fetch_add(lanes.len(), std::sync::atomic::Ordering::Relaxed);
    }
    tracing::info!(
        requests = lanes.len(),
        boundary_rows = lanes.iter().map(|lane| lane.shape.boundary).sum::<usize>(),
        cue_rows = lanes
            .iter()
            .map(|lane| lane.shape.cue_rows())
            .sum::<usize>(),
        boundary_ms = boundary_elapsed.as_secs_f64() * 1000.0,
        anchor_ms = anchor_elapsed.as_secs_f64() * 1000.0,
        cue_ms = cue_elapsed.as_secs_f64() * 1000.0,
        elapsed_ms = elapsed.as_secs_f64() * 1000.0,
        "Gemma4 cold stable cohort prefilled in one joint transaction"
    );

    // 7. Publish, FIFO: the per-slot admission tail.
    let mut lanes = lanes.into_iter();
    while let Some(lane) = lanes.next() {
        if let Some(mut fatal) = publish(
            guard,
            scheduler,
            slots,
            retained_tokens,
            prompt_anchors,
            registration,
            kv_bytes_per_token,
            lane,
            elapsed,
        ) {
            fatal.extend_slots(lanes.map(|lane| (Some(lane.handle), lane.reply)));
            return Some(fatal);
        }
    }
    if let Ok(mut stats) = scheduler_stats_snapshot.lock() {
        *stats = scheduler.stats();
    }
    None
}

/// Phase B (design §3.2 step 3): one `forward_prefill_batched_multi_seq`
/// over every lane's boundary rows `prompt[..S]`, each lane at start 0. The
/// head rows are computed and discarded, as `prefill_seed` discards the
/// boundary token; the self-mounts are cleared afterwards.
pub(super) fn boundary_forward(
    guard: &mut Gemma4KvGuard<'_>,
    supervisor: &EngineSupervisor,
    lanes: &[Lane],
) -> Result<()> {
    let seqs: Vec<(Vec<u32>, SlotId)> = lanes
        .iter()
        .map(|lane| {
            (
                lane.prompt[..lane.shape.boundary].to_vec(),
                lane.handle.slot_id,
            )
        })
        .collect();
    // Cold multi-sequence prefill keeps the largest `max_tokens`; its forward
    // bounds each lane by its own slot capacity.
    let max_decode = lanes
        .iter()
        .map(|lane| lane.params.max_tokens.max(1))
        .max()
        .unwrap_or(1);
    let forward = {
        let scaffold = guard
            .hybrid
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("Gemma4 cold cohort requires hybrid KV"))?;
        supervised_gemma4_gpu_call(supervisor, "gemma4_cold_cohort_boundary", || {
            guard.model.weights.forward_prefill_batched_multi_seq(
                &seqs,
                scaffold,
                max_decode,
                &mut guard.model.ctx,
            )
        })
    };
    clear_gemma4_self_mounts(guard.model);
    forward.map(|_| ())
}

/// The lane's `StableBoundary` anchor at `S`, exactly as `prefill_seed`
/// captures it; a quota refusal is `Ok(None)`.
pub(super) fn capture_boundary_anchor(
    guard: &Gemma4KvGuard<'_>,
    lane: &Lane,
) -> Result<Option<Gemma4PromptAnchor>> {
    let scaffold = guard
        .hybrid
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("Gemma4 cold cohort requires hybrid KV"))?;
    let kv = gemma_full_input::capture_or_skip(
        lane.handle.slot_id,
        gemma_full_input::AnchorRole::StableBoundary,
        crate::inference::models::gemma4::kv_cache::snapshot_gemma_hybrid_slot_anchor(
            scaffold,
            lane.handle.slot_id,
            lane.shape.boundary,
        ),
    )?;
    Ok(kv.map(|kv| Gemma4PromptAnchor {
        prompt_tokens: lane.prompt[..lane.shape.boundary].to_vec(),
        kv,
        vision_fingerprint: None,
    }))
}

/// Phase C over every lane (design §3.2 step 5, §4.4). Fills `lane.seed`.
pub(super) fn encode_cues(
    guard: &mut Gemma4KvGuard<'_>,
    supervisor: &EngineSupervisor,
    lanes: &mut [Lane],
) -> Result<()> {
    let vocab = guard.model.weights.vocab_size;
    // Short cues: one batched decode step per cue position.
    let replay: Vec<usize> = (0..lanes.len())
        .filter(|&index| lanes[index].shape.cue == CueRoute::Replay)
        .collect();
    let schedule = replay_schedule(
        &replay
            .iter()
            .map(|&index| (lanes[index].shape.boundary, lanes[index].shape.cue_rows()))
            .collect::<Vec<_>>(),
    );
    for step in schedule {
        let tokens: Vec<u32> = step
            .iter()
            .map(|&(lane, position, _)| lanes[replay[lane]].prompt[position])
            .collect();
        let slot_ids: Vec<SlotId> = step
            .iter()
            .map(|&(lane, _, _)| lanes[replay[lane]].handle.slot_id)
            .collect();
        // Graft length is 0 by rule: the physical position is the prompt
        // position, as the per-slot replay's `kv_graft_len() + S + offset`.
        let positions: Vec<usize> = step.iter().map(|&(_, position, _)| position).collect();
        let logits = supervised_gemma4_gpu_call(supervisor, "gemma4_cold_cohort_cue", || {
            let hybrid = guard
                .hybrid
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("Gemma4 cold cohort cue requires hybrid KV"))?;
            let mut regime =
                crate::inference::models::gemma4::batched_body::BatchedKvRegime::Hybrid(
                    hybrid.as_mut_slice(),
                );
            let mut head_out = None;
            let lm = &mut *guard.model;
            let hidden = lm.weights.forward_decode_body_batched(
                &tokens,
                &slot_ids,
                &positions,
                &mut regime,
                &mut head_out,
                &mut lm.ctx,
            )?;
            // The decode tick's own logits source (`decode_batch_gemma4`):
            // the fused head when the body encoded it, else one batched head.
            let head = match head_out {
                Some(head) => head,
                None if step.iter().any(|&(_, _, last)| last) => {
                    lm.weights
                        .lm_head_batched(&hidden, tokens.len(), &mut lm.ctx)?
                }
                None => return Ok(None),
            };
            anyhow::ensure!(
                head.logits.len() == tokens.len() * vocab,
                "Gemma4 cold cohort cue logits extent {} != {} rows x {vocab}",
                head.logits.len(),
                tokens.len()
            );
            Ok(Some(head.logits))
        })?;
        for (row, &(lane, position, last)) in step.iter().enumerate() {
            if !last {
                continue;
            }
            let lane = &mut lanes[replay[lane]];
            anyhow::ensure!(
                position + 1 == lane.shape.prompt_len,
                "Gemma4 cold cohort replay ended at {position}, prompt {}",
                lane.shape.prompt_len
            );
            let logits = logits
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Gemma4 cold cohort cue produced no head"))?;
            let row = logits[row * vocab..(row + 1) * vocab].to_vec();
            let (token, _) = argmax_f32_first_max(&row);
            lane.seed = Some((token, row));
        }
    }

    // Long cues on every lane: one joint live resume (the joint warm route).
    let joint: Vec<usize> = (0..lanes.len())
        .filter(|&index| lanes[index].shape.cue == CueRoute::JointLive)
        .collect();
    if !joint.is_empty() {
        let seqs: Vec<(Vec<u32>, SlotId, usize)> = joint
            .iter()
            .map(|&index| {
                let lane = &lanes[index];
                (
                    lane.prompt[lane.shape.boundary..].to_vec(),
                    lane.handle.slot_id,
                    lane.shape.boundary,
                )
            })
            .collect();
        let reserve = gemma4_batch_decode_reserve(joint.iter().map(|&index| {
            (
                lanes[index].shape.prompt_len,
                lanes[index].params.max_tokens,
            )
        }));
        let output = {
            let scaffold = guard
                .hybrid
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("Gemma4 cold cohort cue requires hybrid KV"))?;
            supervised_gemma4_gpu_call(supervisor, "gemma4_cold_cohort_cue", || {
                guard.model.weights.forward_prefill_batched_multi_seq_live(
                    &seqs,
                    scaffold,
                    reserve,
                    GemmaLiveAttention::PerSlotExact,
                    &mut guard.model.ctx,
                )
            })?
        };
        clear_gemma4_self_mounts(guard.model);
        for ((&index, token), logits) in joint.iter().zip(output.first_tokens).zip(output.logits) {
            lanes[index].seed = Some((token, logits));
        }
    }

    // A 32-row cue, or a long cue beside a short one: the per-slot resume.
    for lane in lanes
        .iter_mut()
        .filter(|lane| lane.shape.cue == CueRoute::PerSlot)
    {
        lane.seed = Some(per_slot_cue(guard, supervisor, lane)?);
    }
    Ok(())
}

/// The per-slot cue of `prefill_seed` (`gemma4_prefill_stable_cue`), with
/// its logits row copied before any later forward overwrites it.
fn per_slot_cue(
    guard: &mut Gemma4KvGuard<'_>,
    supervisor: &EngineSupervisor,
    lane: &Lane,
) -> Result<(u32, Vec<f32>)> {
    let token = supervised_gemma4_gpu_call(supervisor, "gemma4_prefill_stable_cue", || {
        guard
            .model
            .weights
            .forward_prefill_with_soft_tokens_slot_aware_resume(
                &lane.prompt,
                &[],
                lane.params.max_tokens.max(1),
                &mut guard.model.ctx,
                lane.handle.slot_id,
                &mut guard.kv,
                guard.hybrid.as_mut(),
                guard.dense.as_mut(),
                guard.mlx.as_mut(),
                lane.shape.boundary,
            )
    })?;
    clear_gemma4_self_mounts(guard.model);
    let vocab = guard.model.weights.vocab_size;
    let logits = guard.model.weights.logits_view()?;
    anyhow::ensure!(
        logits.len() >= vocab,
        "Gemma logits row {} < vocab {vocab}",
        logits.len()
    );
    Ok((token, logits[..vocab].to_vec()))
}

/// Phase C non-fatal failure: each lane resumes from its boundary anchor and
/// finishes its cue per slot, FIFO, keeping its reservation. A lane without
/// an anchor (quota refusal) is reset, released and admitted afresh.
#[allow(clippy::too_many_arguments)]
fn recover_cues(
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
    lanes: Vec<Lane>,
) -> (Vec<Lane>, Option<SlotAwareGpuFatal>) {
    let mut restored = Vec::with_capacity(lanes.len());
    let mut lanes = lanes.into_iter();
    while let Some(mut lane) = lanes.next() {
        let slot = lane.handle.slot_id.0 as usize;
        let Some(anchor) = lane.anchor.as_ref() else {
            if let Err(error) = reset_gemma4_slot(guard, scheduler, lane.handle, kv_bytes_per_token)
            {
                let mut fatal = fatal_lanes(error, SlotAwareFatalKind::Invariant, restored);
                fatal.extend_slots(
                    std::iter::once(lane)
                        .chain(lanes)
                        .map(|lane| (Some(lane.handle), lane.reply)),
                );
                return (Vec::new(), Some(fatal));
            }
            retained_tokens[slot].clear();
            release_unused(scheduler, lane.handle);
            if let Some(mut fatal) = admit_per_slot(
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
                [(lane.prompt, lane.params, lane.reply)],
            ) {
                fatal.extend_slots(
                    restored
                        .into_iter()
                        .chain(lanes)
                        .map(|lane| (Some(lane.handle), lane.reply)),
                );
                return (Vec::new(), Some(fatal));
            }
            continue;
        };
        #[cfg(test)]
        CUE_RESTORED_LANES
            .lock()
            .expect("cold cue restore record")
            .push((lane.handle.slot_id.0, lane.prompt.len()));
        let restore = guard
            .hybrid
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Gemma4 cold cohort restore requires hybrid KV"))
            .and_then(|hybrid| {
                crate::inference::models::gemma4::kv_cache::restore_gemma_hybrid_slot_anchor(
                    hybrid,
                    lane.handle.slot_id,
                    &anchor.kv,
                    lane.shape.boundary,
                )
            })
            // The transaction keeps the ledger cursor at 0 until its commit.
            .and_then(|()| install_gemma4_slot_cursor(guard, lane.handle.slot_id, 0));
        let cue = restore.and_then(|()| per_slot_cue(guard, supervisor, &lane));
        match cue {
            Ok(seed) => {
                lane.seed = Some(seed);
                restored.push(lane);
            }
            Err(error) if is_worker_fatal(supervisor, &error) => {
                let mut fatal = fatal_lanes(error, SlotAwareFatalKind::Gpu, restored);
                fatal.extend_slots(
                    std::iter::once(lane)
                        .chain(lanes)
                        .map(|lane| (Some(lane.handle), lane.reply)),
                );
                return (Vec::new(), Some(fatal));
            }
            Err(error) => {
                let reply = match recover_gemma4_slot_after_cancellation_for_reply(
                    guard,
                    scheduler,
                    lane.handle,
                    kv_bytes_per_token,
                    &mut retained_tokens[slot],
                    &mut prompt_anchors[slot],
                    None,
                    false,
                    lane.reply,
                ) {
                    Ok(reply) => reply,
                    Err(mut fatal) => {
                        fatal.extend_slots(
                            restored
                                .into_iter()
                                .chain(lanes)
                                .map(|lane| (Some(lane.handle), lane.reply)),
                        );
                        return (Vec::new(), Some(fatal));
                    }
                };
                scheduler.release(lane.handle);
                slot_fire_done(reply, Err(error), false);
            }
        }
    }
    (restored, None)
}

/// The per-slot admission tail for one committed lane
/// (`admit_gemma4_slot_routed` after `prefill_seed`).
#[allow(clippy::too_many_arguments)]
fn publish(
    guard: &mut Gemma4KvGuard<'_>,
    scheduler: &mut InflightBatchedScheduler,
    slots: &mut [Option<Gemma4Slot>],
    retained_tokens: &mut [Vec<u32>],
    prompt_anchors: &mut [Option<Gemma4PromptAnchor>],
    registration: Option<&ModelRegistration>,
    kv_bytes_per_token: u64,
    lane: Lane,
    elapsed: std::time::Duration,
) -> Option<SlotAwareGpuFatal> {
    let Lane {
        handle,
        prompt,
        params,
        mut reply,
        anchor,
        seed,
        ..
    } = lane;
    let slot = handle.slot_id.0 as usize;
    let Some((first_token, first_logits)) = seed else {
        return Some(SlotAwareGpuFatal::invariant_slot(
            handle,
            reply,
            anyhow::anyhow!(
                "Gemma4 cold cohort lane {} has no first token",
                handle.slot_id.0
            ),
        ));
    };
    let state = match Gemma4DecodeState::from_first_token(
        guard.model,
        &prompt,
        &params,
        registration,
        handle.slot_id,
        first_token,
        Some(&first_logits),
        elapsed,
        0,
    ) {
        Ok(state) => state,
        Err(error) => {
            reply = match recover_gemma4_slot_after_cancellation_for_reply(
                guard,
                scheduler,
                handle,
                kv_bytes_per_token,
                &mut retained_tokens[slot],
                &mut prompt_anchors[slot],
                None,
                false,
                reply,
            ) {
                Ok(reply) => reply,
                Err(fatal) => return Some(fatal),
            };
            scheduler.release(handle);
            slot_fire_done(reply, Err(error), false);
            return None;
        }
    };
    let request_id = gemma4_operator_request_id(handle);
    let (completed, work, rate) = state.operator_prefill_progress();
    crate::serve::operator_ui::prefill_progress("gemma4", request_id, completed, work, rate);
    let (generated, max_tokens, decode_rate) = state.operator_progress();
    crate::serve::operator_ui::decode_progress(
        "gemma4",
        request_id,
        generated,
        max_tokens,
        decode_rate,
    );
    if reply.client_closed() {
        reply.record_client_cancellation();
        reply = match recover_gemma4_slot_after_cancellation_for_reply(
            guard,
            scheduler,
            handle,
            kv_bytes_per_token,
            &mut retained_tokens[slot],
            &mut prompt_anchors[slot],
            anchor,
            false,
            reply,
        ) {
            Ok(reply) => reply,
            Err(fatal) => return Some(fatal),
        };
        scheduler.release(handle);
        slot_fire_done(
            reply,
            Err(anyhow::anyhow!("client disconnected during Gemma4 prefill")),
            true,
        );
        return None;
    }
    prompt_anchors[slot] = anchor;
    scheduler.advance_after_prefill(
        handle,
        u32::try_from(prompt.len()).expect("Gemma cold cohort request was validated"),
    );
    if slot_emit_token(&mut reply, &state.seed_tick()) {
        reply = match recover_gemma4_slot_after_cancellation_for_reply(
            guard,
            scheduler,
            handle,
            kv_bytes_per_token,
            &mut retained_tokens[slot],
            &mut prompt_anchors[slot],
            None,
            false,
            reply,
        ) {
            Ok(reply) => reply,
            Err(fatal) => return Some(fatal),
        };
        scheduler.release(handle);
        finish_gemma4_operator_request(handle, "cancelled");
        slot_fire_done(reply, state.finish(registration), true);
    } else if state.finished_at_seed() {
        retained_tokens[slot] = state.retained_prefix();
        record_retained_slot_kv(
            scheduler,
            handle,
            retained_tokens[slot].len(),
            kv_bytes_per_token,
        );
        scheduler.advance_after_decode(handle);
        scheduler.release(handle);
        finish_gemma4_operator_request(handle, "complete");
        slot_fire_done(reply, state.finish(registration), false);
    } else {
        slots[slot] = Some((Gemma4SlotWork::Decode(state), reply, handle));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reservation order (design §6.1): the cohort's back-to-back
    /// `AdmitRequest`s pick the same slots, in the same order, as the
    /// per-slot path, whose admissions interleave each lane's reset and
    /// prefill bookkeeping, including when a released slot retains KV.
    #[test]
    fn gemma_cold_cohort_reservation_order_matches_per_slot_path() {
        const KV: u64 = 1_000;
        const BUDGET: u64 = 8 * 4_096 * KV;
        let prompts: Vec<Vec<u32>> = (0..4u32)
            .map(|lane| (0..465).map(|token| lane * 10_000 + token).collect())
            .collect();
        let params = SamplingParams {
            max_tokens: 64,
            stable_prompt_prefix_tokens: Some(458),
            ..SamplingParams::default()
        };
        let retained = vec![Vec::new(); 4];
        let anchors: Vec<Option<Gemma4PromptAnchor>> = (0..4).map(|_| None).collect();
        for retained_slot in [None, Some(2u32)] {
            let fixture = || {
                let mut scheduler = InflightBatchedScheduler::new_with_kv_budget(8, 4, BUDGET);
                if let Some(slot) = retained_slot {
                    // A finished earlier request leaves measured KV on `slot`.
                    let mut held = Vec::new();
                    for _ in 0..=slot {
                        let request = gemma4_slot_admit_request(
                            Gemma4ValidatedRequestShape {
                                prompt_tokens: 600,
                                max_tokens: 64,
                            },
                            600,
                            0,
                            BUDGET,
                            KV,
                            None,
                        );
                        held.push(scheduler.admit(request).unwrap().handle.unwrap());
                    }
                    for handle in held {
                        let bytes = if handle.slot_id.0 == slot {
                            600 * KV
                        } else {
                            0
                        };
                        scheduler.record_slot_high_water(handle, bytes);
                        scheduler.release(handle);
                    }
                }
                scheduler
            };

            // Per-slot path: admit, reset bookkeeping, prefill, then the next.
            let mut serial = fixture();
            let mut serial_requests = Vec::new();
            let mut serial_slots = Vec::new();
            for prompt in &prompts {
                let req = gemma4_slot_admit_request(
                    validate_gemma4_generation_request(465, 64, 32_768, 0).unwrap(),
                    465,
                    0,
                    BUDGET,
                    KV,
                    preferred_gemma4_slot_for_request(
                        &retained,
                        &anchors,
                        &[None, None, None, None],
                        prompt,
                        None,
                    )
                    .map(|preference| preference.slot),
                );
                serial_requests.push(format!("{req:?}"));
                let handle = serial.admit(req).unwrap().handle.unwrap();
                record_retained_slot_kv(&mut serial, handle, 0, KV);
                serial.advance_after_prefill(handle, 465);
                serial_slots.push(handle.slot_id);
            }

            // Cohort: every reservation first, then the resets.
            let mut joint = fixture();
            let mut joint_requests = Vec::new();
            let mut handles: Vec<SlotHandle> = Vec::new();
            for prompt in &prompts {
                let earlier: Vec<(usize, &[u32], usize)> = handles
                    .iter()
                    .zip(&prompts)
                    .map(|(handle, prompt)| (handle.slot_id.0 as usize, prompt.as_slice(), 458))
                    .collect();
                let req = cohort_admit_request(
                    &retained, &anchors, prompt, &params, &earlier, 32_768, 0, BUDGET, KV,
                )
                .expect("cold lane");
                joint_requests.push(format!("{req:?}"));
                handles.push(joint.admit(req).unwrap().handle.unwrap());
            }
            for &handle in &handles {
                record_retained_slot_kv(&mut joint, handle, 0, KV);
                joint.advance_after_prefill(handle, 465);
            }
            let joint_slots: Vec<SlotId> = handles.iter().map(|handle| handle.slot_id).collect();
            assert_eq!(
                joint_requests, serial_requests,
                "retained={retained_slot:?}"
            );
            assert_eq!(joint_slots, serial_slots, "retained={retained_slot:?}");
        }
    }

    #[test]
    fn gemma_cold_replay_schedule_unequal_cues() {
        // Lanes {1, 7, 31} cue rows at boundaries 100, 200, 300.
        let schedule = replay_schedule(&[(100, 1), (200, 7), (300, 31)]);
        assert_eq!(schedule.len(), 31);
        assert_eq!(
            schedule[0],
            vec![(0, 100, true), (1, 200, false), (2, 300, false)]
        );
        assert_eq!(schedule[1], vec![(1, 201, false), (2, 301, false)]);
        assert_eq!(schedule[6], vec![(1, 206, true), (2, 306, false)]);
        assert_eq!(schedule[7], vec![(2, 307, false)]);
        assert_eq!(schedule[30], vec![(2, 330, true)]);
        // Every lane's head is read exactly once, at its last position.
        let reads: Vec<(usize, usize)> = schedule
            .iter()
            .flatten()
            .filter(|(_, _, last)| *last)
            .map(|&(lane, position, _)| (lane, position))
            .collect();
        assert_eq!(reads, [(0, 100), (1, 206), (2, 330)]);
        assert!(replay_schedule(&[]).is_empty());
    }

    #[test]
    fn gemma_cold_serial_preference_sees_earlier_lanes_as_active_owners() {
        let shared: Vec<u32> = (0..50).collect();
        let lane0 = [shared.clone(), vec![900; 10]].concat();
        let lane1 = [lane0.clone(), vec![901; 5]].concat();
        let retained = vec![Vec::new(); 4];
        let anchors: Vec<Option<Gemma4PromptAnchor>> = (0..4).map(|_| None).collect();
        // Alone: no reusable state anywhere → cold.
        assert!(serial_preference(&retained, &anchors, &lane1, &[]).is_none());
        // After lane 0 owns slot 2, lane 1 extends lane 0's prompt: serial
        // planning sees an active owner (no idle preference), exactly as
        // `preferred_gemma4_slot_for_request` does with lane 0 installed.
        assert!(serial_preference(&retained, &anchors, &lane1, &[(2, &lane0, 50)]).is_none());
        // A live ledger is credited only with an identity-matching anchor on
        // its slot (`preferred_gemma4_slot_for_request`); without one the
        // lane stays cold.
        let mut retained = retained;
        retained[1] = shared[..40].to_vec();
        assert!(serial_preference(&retained, &anchors, &lane1, &[]).is_none());
        // An earlier lane's slot replaces that slot's old state and is an
        // active owner, never an idle preference.
        let affinity = gemma4_slot_affinity(
            [(
                1usize,
                &[][..],
                Some(&lane0[..50]),
                Some(lane0.as_slice()),
                true,
                false,
            )],
            &lane1,
        );
        assert!(matches!(affinity, Some(Gemma4SlotAffinity::Active(_))));
        assert!(serial_preference(&retained, &anchors, &lane1, &[(1, &lane0, 50)]).is_none());
    }
}
