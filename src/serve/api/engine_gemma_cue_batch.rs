//! ADR-063: joint replay of same-step Gemma 4 generation cues.
//!
//! A native Gemma chat turn renders a stable prefix `S` (system, tools,
//! history) followed by a short generation cue (7 tokens on the canonical
//! workload). The per-slot stable route prefills `S`, captures the
//! stable-boundary checkpoint, then resumes the cue with
//! `forward_prefill_with_soft_tokens_slot_aware_resume`, which replays any
//! text suffix shorter than 32 tokens as one `forward_decode_slot_aware`
//! forward per token. Four cold agents therefore pay 4 × 7 = 28 serial
//! single-slot decode forwards at turn 0, after their four boundary
//! prefills.
//!
//! This module admits a same-step group of stable lanes in FIFO order exactly
//! as before (boundary prefill and checkpoint unchanged, per slot), defers each
//! tiny cue, then replays all cues together: cue position `k` of every lane
//! that still has one is one row of a single `forward_decode_body_batched`
//! call (the production decode body), and each lane's last cue row goes
//! through `lm_head_batched`. A 7-token cue costs 7 batched-body steps for the
//! whole cohort instead of 7 single-slot forwards per lane.
//!
//! Exactness: production decode already uses this body and head for every
//! decode step, and their rows are documented bit-identical to the scalar
//! `forward_decode_slot_aware` path (ADR-040 H-S1-rowparity, H-S2-tokenparity:
//! at ≤ 8 rows every dense and expert matmul takes a row-identical route).
//! The repo's existing N=1/4/8 engine tests only check generated text and
//! tokens, so this module's real-artifact gate
//! (`gemma_joint_cue_replay_matches_per_slot_bitwise`) is the bitwise proof:
//! first token, final logits bits, KV bytes and cursors versus the per-slot
//! replay. The route stays off under any non-default GGML routing policy
//! (for example `HF2Q_DECODE_MV_EXT=1`, set process-wide when a Qwen3.8 model
//! loads), where row identity across widths is not established.
//!
//! Failure semantics: a worker-fatal error fails the worker. A non-fatal
//! joint replay error restores each lane's sliding rings from its
//! stable-boundary checkpoint and replays its cue per slot in FIFO order,
//! exactly as inline admission would; a lane without a checkpoint (quota
//! refusal) is restored from its request-start anchor and failed
//! request-locally. Publication (checkpoint, seed, decode install) runs per
//! lane in FIFO order through `finish_gemma4_slot_seed`, the inline tail.
use super::*;
use crate::inference::models::gemma4::kv_cache::GemmaHybridSlotAnchor;
use crate::serve::api::registry::ModelRegistration;

pub(super) const JOINT_CUE_ENV: &str = "HF2Q_GEMMA_JOINT_CUE";
/// On by default only because the official-artifact bitwise gate is the
/// contract; the root runs it before merge.
const JOINT_CUE_DEFAULT: bool = true;

#[cfg(test)]
pub(super) static JOINT_CUE_COHORTS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(super) static JOINT_CUE_LANES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn parse_switch(raw: Option<&str>) -> std::result::Result<bool, String> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(JOINT_CUE_DEFAULT);
    };
    match raw.to_ascii_lowercase().as_str() {
        "1" | "on" | "true" => Ok(true),
        "0" | "off" | "false" => Ok(false),
        _ => Err(format!(
            "{JOINT_CUE_ENV} must be 1/on/true or 0/off/false, got {raw:?}"
        )),
    }
}

/// Resolve the switch once when the Gemma SlotAware worker starts. The route
/// needs the production batched decode body over the hybrid KV scaffold.
pub(super) fn worker_enabled() -> bool {
    let raw = std::env::var(JOINT_CUE_ENV).ok();
    let requested = parse_switch(raw.as_deref()).unwrap_or_else(|error| {
        tracing::warn!(
            error,
            fallback = JOINT_CUE_DEFAULT,
            "invalid Gemma joint cue switch"
        );
        JOINT_CUE_DEFAULT
    });
    let unavailable = if std::env::var("HF2Q_BATCHED_BODY").as_deref() == Ok("0") {
        Some("HF2Q_BATCHED_BODY=0 disables the batched decode body")
    } else if !crate::debug::INVESTIGATION_ENV.hybrid_kv {
        Some("the batched cue replay needs the hybrid KV scaffold")
    } else {
        None
    };
    let active = requested && unavailable.is_none();
    tracing::info!(
        requested,
        active,
        unavailable,
        "Gemma joint cue replay worker decision"
    );
    active
}

/// True when the per-slot stable route replays the cue after `boundary`
/// token by token: `forward_prefill_with_soft_tokens_slot_aware_resume`
/// decodes a non-empty text suffix shorter than
/// `GEMMA_SLOT_BATCHED_PREFILL_MIN_TOKENS` (32) one forward per token.
pub(super) fn cue_replays_per_token(prompt_len: usize, boundary: usize) -> bool {
    prompt_len.checked_sub(boundary).is_some_and(|cue| {
        cue > 0 && cue < crate::serve::forward_prefill::GEMMA_SLOT_BATCHED_PREFILL_MIN_TOKENS
    })
}

/// Row identity across decode widths is established only for the default
/// GGML routing policy (width-invariant mv/mvN dense routes, mv_id experts).
pub(super) fn routes_row_identical(policy: &mlx_native::GgmlRoutingPolicy) -> bool {
    *policy == mlx_native::GgmlRoutingPolicy::default()
}

/// Cue position `k` → the lanes that still replay a token there, each with
/// whether `k` is that lane's last cue token (its head row).
pub(super) fn cue_steps(cue_lens: &[usize]) -> Vec<Vec<(usize, bool)>> {
    let steps = cue_lens.iter().copied().max().unwrap_or(0);
    (0..steps)
        .map(|k| {
            cue_lens
                .iter()
                .enumerate()
                .filter(|&(_, &len)| k < len)
                .map(|(lane, &len)| (lane, k + 1 == len))
                .collect()
        })
        .collect()
}

/// One admitted stable lane whose boundary prefill and checkpoint are done
/// and whose generation cue is deferred (`Gemma4SeedOutcome::CueDeferred`).
pub(super) struct CueLane {
    pub(super) handle: SlotHandle,
    pub(super) reply: SlotReply,
    pub(super) prompt_tokens: Vec<u32>,
    pub(super) params: SamplingParams,
    pub(super) cached_tokens: usize,
    pub(super) shape_prompt_tokens: u32,
    pub(super) boundary: usize,
    pub(super) request_rollback_anchor: Option<Gemma4PromptAnchor>,
    pub(super) boundary_anchor: Option<GemmaHybridSlotAnchor>,
    pub(super) boundary_duration: Duration,
}

/// One lane's cue for [`replay_cues_jointly`]: `cue[k]` is decoded at
/// physical position `start + k` in `slot`.
pub(super) struct CueRows<'a> {
    pub(super) slot: SlotId,
    pub(super) start: usize,
    pub(super) cue: &'a [u32],
}

/// Head output of one lane's last cue token.
pub(super) struct CueOutput {
    pub(super) first_token: u32,
    pub(super) logits: Vec<f32>,
}

/// Replay every lane's cue through the production batched decode body, one
/// call per cue position, and the batched head for each lane's last token.
/// The first token is the first-max argmax of the logits row, the same
/// selection production decode applies. Writes each lane's cue K/V rows;
/// does not move the cursor ledger.
pub(super) fn replay_cues_jointly(
    guard: &mut Gemma4KvGuard<'_>,
    lanes: &[CueRows<'_>],
    supervisor: &EngineSupervisor,
) -> Result<Vec<CueOutput>> {
    let vocab = guard.model.weights.vocab_size;
    let cue_lens: Vec<usize> = lanes.iter().map(|lane| lane.cue.len()).collect();
    let mut outputs: Vec<Option<CueOutput>> = lanes.iter().map(|_| None).collect();
    for (k, rows) in cue_steps(&cue_lens).into_iter().enumerate() {
        let tokens: Vec<u32> = rows.iter().map(|&(lane, _)| lanes[lane].cue[k]).collect();
        let slot_ids: Vec<SlotId> = rows.iter().map(|&(lane, _)| lanes[lane].slot).collect();
        let positions: Vec<usize> = rows
            .iter()
            .map(|&(lane, _)| lanes[lane].start + k)
            .collect();
        clear_gemma4_self_mounts(guard.model);
        let mut fused_head = None;
        let hidden = supervised_gemma4_gpu_call(supervisor, "gemma4_joint_cue_body", || {
            let hybrid = guard
                .hybrid
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("Gemma joint cue replay requires hybrid KV"))?;
            let mut regime =
                crate::inference::models::gemma4::batched_body::BatchedKvRegime::Hybrid(
                    hybrid.as_mut_slice(),
                );
            let lm = &mut *guard.model;
            lm.weights.forward_decode_body_batched(
                &tokens,
                &slot_ids,
                &positions,
                &mut regime,
                &mut fused_head,
                &mut lm.ctx,
            )
        })?;
        let last: Vec<(usize, usize)> = rows
            .iter()
            .enumerate()
            .filter(|&(_, &(_, is_last))| is_last)
            .map(|(row, &(lane, _))| (row, lane))
            .collect();
        if last.is_empty() {
            continue;
        }
        // The fused head (HF2Q_FUSE_LMHEAD=1) already covers every row of
        // this step; otherwise only the finishing rows go through the head.
        let (logits, rows_are_step_rows) = match fused_head.take() {
            Some(head) => (head.logits, true),
            None => {
                let hs = guard.model.weights.hidden_size;
                let mut gathered = Vec::with_capacity(last.len() * hs);
                for &(row, _) in &last {
                    gathered.extend_from_slice(&hidden[row * hs..(row + 1) * hs]);
                }
                let head = supervised_gemma4_gpu_call(supervisor, "gemma4_joint_cue_head", || {
                    guard
                        .model
                        .weights
                        .lm_head_batched(&gathered, last.len(), &mut guard.model.ctx)
                })?;
                (head.logits, false)
            }
        };
        for (index, &(row, lane)) in last.iter().enumerate() {
            let at = if rows_are_step_rows { row } else { index };
            let row_logits = logits
                .get(at * vocab..(at + 1) * vocab)
                .ok_or_else(|| anyhow::anyhow!("Gemma joint cue head returned too few rows"))?;
            let (first_token, _) = argmax_f32_first_max(row_logits);
            outputs[lane] = Some(CueOutput {
                first_token,
                logits: row_logits.to_vec(),
            });
        }
    }
    clear_gemma4_self_mounts(guard.model);
    outputs
        .into_iter()
        .map(|output| {
            output.ok_or_else(|| anyhow::anyhow!("Gemma joint cue lane produced no head"))
        })
        .collect()
}

/// The inline per-slot cue replay (`prefill_seed`'s stable cue call). Leaves
/// the lane's last-token logits in the model's logits buffer.
pub(super) fn replay_cue_per_slot(
    guard: &mut Gemma4KvGuard<'_>,
    slot: SlotId,
    prompt_tokens: &[u32],
    boundary: usize,
    max_decode_tokens: usize,
    supervisor: &EngineSupervisor,
) -> Result<u32> {
    supervised_gemma4_gpu_call(supervisor, "gemma4_prefill_stable_cue", || {
        guard
            .model
            .weights
            .forward_prefill_with_soft_tokens_slot_aware_resume(
                prompt_tokens,
                &[],
                max_decode_tokens,
                &mut guard.model.ctx,
                slot,
                &mut guard.kv,
                guard.hybrid.as_mut(),
                guard.dense.as_mut(),
                guard.mlx.as_mut(),
                boundary,
            )
    })
}

fn cue_lanes_fatal(
    lanes: impl IntoIterator<Item = CueLane>,
    error: anyhow::Error,
    kind: SlotAwareFatalKind,
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

/// Admit a same-step group of stable requests in FIFO order with their tiny
/// cues deferred, then replay the deferred cues (jointly when two or more)
/// and publish every lane in FIFO order.
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
) -> Option<SlotAwareGpuFatal> {
    let mut lanes: Vec<CueLane> = Vec::new();
    let mut requests = requests.into_iter();
    while let Some((prompt_tokens, params, reply)) = requests.next() {
        if let Some(mut fatal) = admit_gemma4_slot_routed(
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
            Gemma4AdmitRoute::JointCueCandidate(&mut lanes),
        ) {
            fatal.extend_slots(lanes.drain(..).map(|lane| (Some(lane.handle), lane.reply)));
            fatal.extend_slots(requests.map(|(_, _, reply)| (None, reply)));
            return Some(fatal);
        }
    }
    replay_and_publish(
        guard,
        scheduler,
        slots,
        retained_tokens,
        prompt_anchors,
        registration,
        kv_bytes_per_token,
        supervisor,
        lanes,
    )
}

/// Rewind a lane whose joint replay failed to its stable-boundary checkpoint
/// (every sliding ring) and its admission cursor, so its per-slot replay
/// starts from the same bytes the inline path would.
fn restore_cue_lane(guard: &mut Gemma4KvGuard<'_>, lane: &CueLane) -> Result<()> {
    let anchor = lane
        .boundary_anchor
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Gemma cue lane has no stable-boundary checkpoint"))?;
    let graft_len = guard.model.weights.kv_graft_len();
    let hybrid = guard
        .hybrid
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("Gemma cue lane restore requires hybrid KV"))?;
    crate::inference::models::gemma4::kv_cache::restore_gemma_hybrid_slot_anchor(
        hybrid,
        lane.handle.slot_id,
        anchor,
        graft_len + lane.boundary,
    )?;
    install_gemma4_slot_cursor(guard, lane.handle.slot_id, lane.cached_tokens)
}

#[allow(clippy::too_many_arguments)]
fn replay_and_publish(
    guard: &mut Gemma4KvGuard<'_>,
    scheduler: &mut InflightBatchedScheduler,
    slots: &mut [Option<Gemma4Slot>],
    retained_tokens: &mut [Vec<u32>],
    prompt_anchors: &mut [Option<Gemma4PromptAnchor>],
    registration: Option<&ModelRegistration>,
    kv_bytes_per_token: u64,
    supervisor: &EngineSupervisor,
    lanes: Vec<CueLane>,
) -> Option<SlotAwareGpuFatal> {
    if lanes.is_empty() {
        return None;
    }
    let graft_len = guard.model.weights.kv_graft_len();
    let joint = lanes.len() >= 2
        && guard.hybrid.is_some()
        && routes_row_identical(&mlx_native::ggml_routing_policy_from_environment());
    let started = Instant::now();
    // `Some(output)`: joint head row; `None`: replay per slot.
    let mut joint_outputs: Vec<Option<CueOutput>> = lanes.iter().map(|_| None).collect();
    let mut retry_restored = false;
    if joint {
        let rows: Vec<CueRows<'_>> = lanes
            .iter()
            .map(|lane| CueRows {
                slot: lane.handle.slot_id,
                start: graft_len + lane.boundary,
                cue: &lane.prompt_tokens[lane.boundary..],
            })
            .collect();
        match replay_cues_jointly(guard, &rows, supervisor) {
            Ok(outputs) => {
                #[cfg(test)]
                {
                    JOINT_CUE_COHORTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    JOINT_CUE_LANES.fetch_add(lanes.len(), std::sync::atomic::Ordering::Relaxed);
                }
                tracing::info!(
                    requests = lanes.len(),
                    cue_tokens = lanes
                        .iter()
                        .map(|lane| lane.prompt_tokens.len() - lane.boundary)
                        .sum::<usize>(),
                    elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
                    "Gemma4 joint cue replay"
                );
                joint_outputs = outputs.into_iter().map(Some).collect();
            }
            Err(error) if is_worker_fatal(supervisor, &error) => {
                return Some(cue_lanes_fatal(lanes, error, SlotAwareFatalKind::Gpu));
            }
            Err(error) => {
                tracing::warn!(
                    requests = lanes.len(),
                    error = %format!("{error:#}"),
                    "Gemma4 joint cue replay failed; replaying each cue per slot"
                );
                retry_restored = true;
            }
        }
    }
    let cohort_cue = started.elapsed();

    let mut lanes = lanes.into_iter().zip(joint_outputs);
    while let Some((lane, joint_output)) = lanes.next() {
        let slot_idx = lane.handle.slot_id.0 as usize;
        let (first_token, logits, cue_elapsed) = match joint_output {
            Some(output) => (Ok(output.first_token), Some(output.logits), cohort_cue),
            None => {
                let lane_started = Instant::now();
                let restored = if retry_restored {
                    restore_cue_lane(guard, &lane)
                } else {
                    Ok(())
                };
                let replay = restored.and_then(|()| {
                    clear_gemma4_self_mounts(guard.model);
                    replay_cue_per_slot(
                        guard,
                        lane.handle.slot_id,
                        &lane.prompt_tokens,
                        lane.boundary,
                        lane.params.max_tokens.max(1),
                        supervisor,
                    )
                });
                (replay, None, lane_started.elapsed())
            }
        };
        // Per-slot replay leaves the logits in the model buffer, which
        // `from_first_token(.., None, ..)` reads, exactly as inline seeding.
        let seeded = first_token.and_then(|first_token| {
            Gemma4DecodeState::from_first_token(
                guard.model,
                &lane.prompt_tokens,
                &lane.params,
                registration,
                lane.handle.slot_id,
                first_token,
                logits.as_deref(),
                lane.boundary_duration + cue_elapsed,
                lane.cached_tokens,
            )
        });
        let CueLane {
            handle,
            mut reply,
            prompt_tokens,
            params,
            cached_tokens,
            shape_prompt_tokens,
            request_rollback_anchor,
            boundary_anchor,
            ..
        } = lane;
        let state = match seeded {
            Ok(state) => state,
            Err(error) if is_worker_fatal(supervisor, &error) => {
                let mut fatal = SlotAwareGpuFatal::slot(handle, reply, error);
                fatal.extend_slots(lanes.map(|(lane, _)| (Some(lane.handle), lane.reply)));
                return Some(fatal);
            }
            Err(error) => {
                reply = match recover_gemma4_slot_after_cancellation_for_reply(
                    guard,
                    scheduler,
                    handle,
                    kv_bytes_per_token,
                    &mut retained_tokens[slot_idx],
                    &mut prompt_anchors[slot_idx],
                    request_rollback_anchor,
                    false,
                    reply,
                ) {
                    Ok(reply) => reply,
                    Err(mut fatal) => {
                        fatal.extend_slots(lanes.map(|(lane, _)| (Some(lane.handle), lane.reply)));
                        return Some(fatal);
                    }
                };
                scheduler.release(handle);
                slot_fire_done(reply, Err(error), false);
                continue;
            }
        };
        if let Some(mut fatal) = finish_gemma4_slot_seed(
            guard,
            scheduler,
            slots,
            retained_tokens,
            prompt_anchors,
            registration,
            kv_bytes_per_token,
            handle,
            reply,
            prompt_tokens,
            params,
            cached_tokens,
            shape_prompt_tokens,
            request_rollback_anchor,
            state,
            boundary_anchor,
        ) {
            fatal.extend_slots(lanes.map(|(lane, _)| (Some(lane.handle), lane.reply)));
            return Some(fatal);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma_joint_cue_defers_only_the_per_token_cue_route() {
        // Canonical turn 0: 458-token stable prefix + 7-token generation cue.
        assert!(cue_replays_per_token(465, 458));
        // 31 tokens still replay per token; 32 take the batched resume.
        assert!(cue_replays_per_token(458 + 31, 458));
        assert!(!cue_replays_per_token(458 + 32, 458));
        // No cue, or a boundary past the prompt, defers nothing.
        assert!(!cue_replays_per_token(458, 458));
        assert!(!cue_replays_per_token(400, 458));
    }

    #[test]
    fn gemma_joint_cue_steps_drop_finished_lanes_and_mark_head_rows() {
        let four = cue_steps(&[7, 7, 7, 7]);
        assert_eq!(four.len(), 7);
        for (k, rows) in four.iter().enumerate() {
            assert_eq!(
                rows,
                &(0..4).map(|lane| (lane, k == 6)).collect::<Vec<_>>(),
                "step {k}"
            );
        }
        assert_eq!(
            cue_steps(&[3, 1, 2]),
            vec![
                vec![(0, false), (1, true), (2, false)],
                vec![(0, false), (2, true)],
                vec![(0, true)],
            ]
        );
        // Every lane gets exactly one head row, at its own last position.
        let mixed = [7usize, 3, 12, 31];
        let steps = cue_steps(&mixed);
        assert_eq!(steps.len(), 31);
        for (lane, &len) in mixed.iter().enumerate() {
            let heads: Vec<usize> = steps
                .iter()
                .enumerate()
                .filter(|(_, rows)| rows.contains(&(lane, true)))
                .map(|(k, _)| k)
                .collect();
            assert_eq!(heads, vec![len - 1], "lane {lane}");
            let rows = steps
                .iter()
                .filter(|rows| rows.iter().any(|&(row_lane, _)| row_lane == lane))
                .count();
            assert_eq!(rows, len, "lane {lane} replays each cue token once");
        }
        assert!(cue_steps(&[]).is_empty());
    }

    #[test]
    fn gemma_joint_cue_requires_the_default_routing_policy() {
        let default = mlx_native::GgmlRoutingPolicy::default();
        assert!(routes_row_identical(&default));
        let mv_ext = mlx_native::GgmlRoutingPolicy {
            dense_decode_mv_ext: true,
            ..default
        };
        assert!(
            !routes_row_identical(&mv_ext),
            "the padded multi-column route is not bit-exact across widths"
        );
        let no_mvn = mlx_native::GgmlRoutingPolicy {
            dense_decode_mvn: false,
            ..default
        };
        assert!(!routes_row_identical(&no_mvn));
    }

    #[test]
    fn gemma_joint_cue_switch_defaults_on_with_a_kill_switch() {
        assert_eq!(parse_switch(None), Ok(true));
        assert_eq!(parse_switch(Some("")), Ok(true));
        for on in ["1", "on", "True"] {
            assert_eq!(parse_switch(Some(on)), Ok(true));
        }
        for off in ["0", "OFF", "false"] {
            assert_eq!(parse_switch(Some(off)), Ok(false));
        }
        assert!(parse_switch(Some("yes")).is_err());
    }
}
