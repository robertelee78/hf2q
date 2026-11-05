//! Cold-admission policy, before any slot reset or GPU submission.
//!
//! * [`plan`]: the iter-G(a) no-boundary cold batch. Since ADR-063
//!   Q4-GEMMA-COLD (decision D2) its lanes run the per-sequence cold
//!   attention, so a lane must exceed 32 rows ([`MIN_LANE_ROWS`]): a 32-row
//!   lane's MoE runs `mv_id` alone but `mm_id` in the joint stream. Graft-bound
//!   workers keep per-slot cold admission until a graft bitwise gate exists.
//! * [`stable_plan`]: the cold stable cohort rule (design §3.1), the turn-0
//!   shape of the four-agent task: every lane cold, with an exact
//!   pre-generation boundary of at least 33 rows.
//! * [`worker_enabled`]: the `HF2Q_GEMMA_JOINT_COLD_PREFILL` kill switch and
//!   the worker substrate, resolved once per SlotAware worker.
use super::*;

pub(super) const JOINT_COLD_PREFILL_ENV: &str = "HF2Q_GEMMA_JOINT_COLD_PREFILL";
/// On by default (operator decision D3): the official-artifact bitwise gates
/// are the contract, and the root runs them before merge.
const JOINT_COLD_PREFILL_DEFAULT: bool = true;
/// Smallest cold multi-sequence lane whose row-wise kernels route exactly as
/// its single-slot prefill (`gemma_warm_batch::MIN_SUFFIX_TOKENS`).
pub(super) const MIN_LANE_ROWS: usize = gemma_warm_batch::MIN_SUFFIX_TOKENS;

#[derive(Debug, Eq, PartialEq)]
pub(super) struct Plan {
    pub(super) has_tiny_prefill: bool,
    transaction_too_large: bool,
    has_reusable_prefix: bool,
    graft_bound: bool,
}
impl Plan {
    pub(super) fn requires_slot_admission(&self) -> bool {
        self.has_tiny_prefill
            || self.transaction_too_large
            || self.has_reusable_prefix
            || self.graft_bound
    }
}

/// The caller supplies the actual request-aware retained-slot preference.
/// No generation cue (S == P) is not proof that a request has no reusable KV.
/// Every lane falls back in its original order if any lane needs slot reuse.
pub(super) fn plan(candidates: impl IntoIterator<Item = (usize, usize)>, graft_len: usize) -> Plan {
    let mut total_rows = Some(0usize);
    let mut result = Plan {
        has_tiny_prefill: false,
        transaction_too_large: false,
        has_reusable_prefix: false,
        graft_bound: graft_len > 0,
    };
    for (rows, cached) in candidates {
        total_rows = total_rows.and_then(|total| total.checked_add(rows));
        result.has_tiny_prefill |= rows < MIN_LANE_ROWS;
        result.has_reusable_prefix |= cached > 0;
    }
    result.transaction_too_large =
        total_rows.is_none_or(|rows| rows > GEMMA4_SLOT_PREFILL_CHUNK_TOKENS as usize);
    result
}

fn parse_switch(raw: Option<&str>) -> std::result::Result<bool, String> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(JOINT_COLD_PREFILL_DEFAULT);
    };
    match raw.to_ascii_lowercase().as_str() {
        "1" | "on" | "true" => Ok(true),
        "0" | "off" | "false" => Ok(false),
        _ => Err(format!(
            "{JOINT_COLD_PREFILL_ENV} must be 1/on/true or 0/off/false, got {raw:?}"
        )),
    }
}

/// The worker setting that keeps the cohort off, if any. Each one either
/// changes the per-slot reference route the cohort reproduces or bypasses
/// the route the cohort runs.
fn unavailable_reason(
    cross_slot_admit: bool,
    hybrid_kv: bool,
    env: impl Fn(&str) -> Option<String>,
) -> Option<&'static str> {
    let is = |name: &str, values: &[&str]| {
        env(name).is_some_and(|value| values.contains(&value.trim().to_ascii_lowercase().as_str()))
    };
    if !cross_slot_admit {
        Some("cross-slot admission is off")
    } else if !hybrid_kv {
        Some("the hybrid KV regime is off")
    } else if is("HF2Q_BATCHED_BODY", &["0"]) {
        Some("HF2Q_BATCHED_BODY=0 removes the batched decode body of the cue replay")
    } else if is("HF2Q_PREFILL_SLOT_BATCHED", &["0"]) {
        Some("HF2Q_PREFILL_SLOT_BATCHED=0 replaces the per-slot route lanes must equal")
    } else if env("HF2Q_MM_ID_ROUTING_THRESHOLD").is_some() {
        Some("HF2Q_MM_ID_ROUTING_THRESHOLD moves the MoE route boundary of the lane rule")
    } else if is("HF2Q_DFLASH_XLEN_SDPA", &["1"]) {
        Some("HF2Q_DFLASH_XLEN_SDPA=1 is not supported by multi-sequence prefill")
    } else if is("HF2Q_FA_F16", &["0", "false", "off"]) {
        Some("HF2Q_FA_F16=0 is a diagnostic attention override")
    } else if env("HF2Q_NO_FA").is_some() {
        Some("HF2Q_NO_FA is a diagnostic attention override")
    } else if env("HF2Q_GLOBAL_FA").is_some() {
        Some("HF2Q_GLOBAL_FA is a diagnostic attention override")
    } else {
        None
    }
}

#[derive(Debug, Eq, PartialEq)]
pub(super) struct WorkerDecision {
    pub(super) requested: bool,
    pub(super) active: bool,
    pub(super) unavailable: Option<&'static str>,
    pub(super) invalid: Option<String>,
}

pub(super) fn worker_decision(
    switch: Option<&str>,
    cross_slot_admit: bool,
    hybrid_kv: bool,
    env: impl Fn(&str) -> Option<String>,
) -> WorkerDecision {
    let (requested, invalid) = match parse_switch(switch) {
        Ok(requested) => (requested, None),
        Err(error) => (JOINT_COLD_PREFILL_DEFAULT, Some(error)),
    };
    let unavailable = unavailable_reason(cross_slot_admit, hybrid_kv, env);
    WorkerDecision {
        requested,
        active: requested && unavailable.is_none(),
        unavailable,
        invalid,
    }
}

/// Resolve the switch once when the Gemma SlotAware worker starts.
pub(super) fn worker_enabled(cross_slot_admit: bool) -> bool {
    let raw = std::env::var(JOINT_COLD_PREFILL_ENV).ok();
    let decision = worker_decision(
        raw.as_deref(),
        cross_slot_admit,
        crate::debug::INVESTIGATION_ENV.hybrid_kv,
        |name| std::env::var(name).ok(),
    );
    if let Some(error) = decision.invalid.as_deref() {
        tracing::warn!(
            error,
            fallback = JOINT_COLD_PREFILL_DEFAULT,
            "invalid Gemma joint cold prefill switch"
        );
    }
    tracing::info!(
        requested = decision.requested,
        active = decision.active,
        unavailable = decision.unavailable,
        "Gemma joint cold prefill worker decision"
    );
    decision.active
}

/// How one lane processes its generation cue `prompt[S..P]` (design §3.2
/// step 5).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CueRoute {
    /// Fewer than 32 rows: replayed through the batched decode body, one
    /// step per position, as the per-slot resume replays it through
    /// `forward_decode_slot_aware`.
    Replay,
    /// Every lane has at least 33 cue rows: one joint
    /// `forward_prefill_batched_multi_seq_live` with `PerSlotExact`.
    JointLive,
    /// A 32-row cue, or a long cue beside a short one: the per-slot resume
    /// inside the same transaction (exact by identity).
    PerSlot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct StableLane {
    pub(super) boundary: usize,
    pub(super) prompt_len: usize,
    pub(super) cue: CueRoute,
}

impl StableLane {
    pub(super) fn cue_rows(&self) -> usize {
        self.prompt_len - self.boundary
    }
}

/// A cold stable cohort, lanes in FIFO order.
#[derive(Debug, Eq, PartialEq)]
pub(super) struct StablePlan {
    pub(super) lanes: Vec<StableLane>,
}

/// The worker and quantum state the rule reads.
#[derive(Clone, Copy, Debug)]
pub(super) struct StableContext {
    /// [`worker_enabled`].
    pub(super) worker_active: bool,
    /// Every slot was idle when the quantum was collected.
    pub(super) all_slots_idle: bool,
    pub(super) graft_len: usize,
    pub(super) max_seq_len: usize,
}

fn cue_routes(cues: &[usize]) -> Vec<CueRoute> {
    let all_long = cues.iter().all(|&cue| cue >= MIN_LANE_ROWS);
    cues.iter()
        .map(|&cue| {
            if cue < crate::serve::forward_prefill::GEMMA_SLOT_BATCHED_PREFILL_MIN_TOKENS {
                CueRoute::Replay
            } else if cue >= MIN_LANE_ROWS && all_long {
                CueRoute::JointLive
            } else {
                CueRoute::PerSlot
            }
        })
        .collect()
}

/// The cold stable cohort rule (design §3.1). Pure: it reads token lengths,
/// params, the slot preference (`preference(i)` for lane `i`, the serial
/// admission's preference) and the anchor-quota pre-check, and performs no
/// reservation, reset or GPU work. `None` keeps today's path, in FIFO order.
pub(super) fn stable_plan<'a>(
    ctx: &StableContext,
    requests: impl IntoIterator<Item = (&'a [u32], &'a SamplingParams)>,
    mut preference: impl FnMut(usize) -> Option<Gemma4SlotPreference>,
    anchors_fit: impl FnOnce(&[StableLane]) -> bool,
) -> Option<StablePlan> {
    if !ctx.worker_active || !ctx.all_slots_idle || ctx.graft_len > 0 {
        return None;
    }
    let mut shapes = Vec::new();
    for (index, (prompt, params)) in requests.into_iter().enumerate() {
        let boundary = params.stable_prompt_prefix_tokens?;
        let text = params.vision_fingerprint.is_none()
            && gemma_full_input::effective(params).is_none()
            // A tool or response grammar is validated per lane at its
            // reservation point, as per-slot admission does.
            // `max_tokens == 1` finishes at the seed and releases its slot
            // before the next serial lane is planned; keep it per slot.
            && params.max_tokens > 1;
        let cold = preference(index).is_none_or(|preference| preference.cached_tokens == 0);
        if !text
            || !cold
            || boundary < MIN_LANE_ROWS
            || boundary >= prompt.len()
            || validate_gemma4_generation_request(
                prompt.len(),
                params.max_tokens,
                ctx.max_seq_len,
                ctx.graft_len,
            )
            .is_err()
        {
            return None;
        }
        shapes.push((boundary, prompt.len()));
    }
    if shapes.len() < 2
        || !gemma4_batch_rows_within_transaction_limit(shapes.iter().map(|&(_, prompt)| prompt))
    {
        return None;
    }
    let cues: Vec<usize> = shapes.iter().map(|&(s, p)| p - s).collect();
    let lanes: Vec<StableLane> = shapes
        .into_iter()
        .zip(cue_routes(&cues))
        .map(|((boundary, prompt_len), cue)| StableLane {
            boundary,
            prompt_len,
            cue,
        })
        .collect();
    anchors_fit(&lanes).then_some(StablePlan { lanes })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preference(
        retained: &[Vec<u32>],
        anchors: &[Vec<u32>],
        prompt: &[u32],
    ) -> Option<Gemma4SlotPreference> {
        // Actual production affinity primitive, with CPU-only token owners;
        // no fabricated GPU cache or snapshot is required to test selection.
        gemma4_slot_affinity(
            retained
                .iter()
                .zip(anchors)
                .enumerate()
                .map(|(i, (live, anchor))| {
                    (
                        i,
                        live.as_slice(),
                        Some(anchor.as_slice()),
                        None,
                        false,
                        false,
                    )
                }),
            prompt,
        )
        .map(Gemma4SlotAffinity::preference)
    }

    fn histories() -> (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let stable: Vec<Vec<u32>> = (0..4)
            .map(|agent| (0..458).map(|token| agent * 1000 + token).collect())
            .collect();
        let live = stable
            .iter()
            .map(|s| {
                let mut p = s.clone();
                p.extend([8000; 25]);
                p
            })
            .collect();
        let prompts = stable
            .iter()
            .map(|s| {
                let mut p = s.clone();
                p.extend([9000; 56]);
                p
            })
            .collect();
        (live, stable, prompts)
    }

    #[test]
    fn gemma_cold_batch_preserves_four_reusable_458_token_anchors_without_a_cue() {
        let (live, stable, prompts) = histories();
        let candidates = prompts.iter().enumerate().map(|(i, prompt)| {
            assert_eq!(prompt.len(), 514);
            // Deciding failed-run shape: current S == P, thus no strict cue
            // boundary and all four requests classify as the cold cohort.
            assert!(gemma4_cross_batch_eligible(514, 8192, false, false, false));
            let selected = preference(&live, &stable, prompt).unwrap();
            assert_eq!(selected.slot, SlotId(i as u32));
            assert_eq!(selected.cached_tokens, 458);
            assert!(selected.restore_prompt_anchor);
            (prompt.len(), selected.cached_tokens)
        });
        let decision = plan(candidates, 0);
        assert!(decision.requires_slot_admission());
        assert!(!decision.transaction_too_large);
        assert!(!decision.has_tiny_prefill);
    }

    #[test]
    fn gemma_cold_batch_still_batches_true_cold_and_divergent_histories() {
        let (live, stable, _) = histories();
        let prompts = vec![vec![42_000; 514]; 4];
        let decision = plan(
            prompts.iter().map(|prompt| {
                assert!(preference(&live, &stable, prompt).is_none());
                (prompt.len(), 0)
            }),
            0,
        );
        assert!(!decision.requires_slot_admission());
    }

    #[test]
    fn gemma_cold_batch_mixed_warm_cold_falls_back_without_reordering() {
        let (live, stable, prompts) = histories();
        let cold = vec![42_000; 514];
        let requests = [
            cold.as_slice(),
            prompts[2].as_slice(),
            prompts[0].as_slice(),
        ];
        let mut visited = Vec::new();
        let decision = plan(
            requests.iter().enumerate().map(|(i, prompt)| {
                visited.push(i);
                (
                    prompt.len(),
                    preference(&live, &stable, prompt).map_or(0, |p| p.cached_tokens),
                )
            }),
            0,
        );
        assert!(decision.requires_slot_admission());
        assert_eq!(visited, [0, 1, 2]);
    }

    #[test]
    fn gemma_cold_batch_preserves_existing_tiny_aggregate_and_overflow_refusals() {
        assert!(plan([(1, 0), (514, 0)], 0).has_tiny_prefill);
        assert!(plan([(2048, 0), (2049, 0)], 0).requires_slot_admission());
        assert!(plan([(usize::MAX, 0), (1, 0)], 0).requires_slot_admission());
        assert!(!plan([(2048, 0), (2048, 0)], 0).requires_slot_admission());
        // D2 lane floor: a 32-row lane goes per slot, a 33-row lane joins.
        assert!(plan([(32, 0), (514, 0)], 0).has_tiny_prefill);
        assert!(!plan([(33, 0), (514, 0)], 0).requires_slot_admission());
    }

    #[test]
    fn gemma_cold_batch_graft_bound_worker_admits_per_slot() {
        assert!(plan([(514, 0), (514, 0)], 3).requires_slot_admission());
        assert!(!plan([(514, 0), (514, 0)], 3).has_tiny_prefill);
    }

    fn ctx() -> StableContext {
        StableContext {
            worker_active: true,
            all_slots_idle: true,
            graft_len: 0,
            max_seq_len: 32_768,
        }
    }

    fn params(boundary: usize) -> SamplingParams {
        SamplingParams {
            max_tokens: 64,
            stable_prompt_prefix_tokens: Some(boundary),
            ..SamplingParams::default()
        }
    }

    fn lanes(shapes: &[(usize, usize)]) -> (Vec<Vec<u32>>, Vec<SamplingParams>) {
        shapes
            .iter()
            .enumerate()
            .map(|(lane, &(boundary, prompt))| {
                (
                    (0..prompt as u32)
                        .map(|token| lane as u32 * 10_000 + token)
                        .collect(),
                    params(boundary),
                )
            })
            .unzip()
    }

    fn plan_for(
        ctx: &StableContext,
        prompts: &[Vec<u32>],
        params: &[SamplingParams],
    ) -> Option<StablePlan> {
        stable_plan(
            ctx,
            prompts.iter().map(Vec::as_slice).zip(params),
            |_| None,
            |_| true,
        )
    }

    fn routes(shapes: &[(usize, usize)]) -> Option<Vec<CueRoute>> {
        let (prompts, params) = lanes(shapes);
        plan_for(&ctx(), &prompts, &params)
            .map(|plan| plan.lanes.iter().map(|lane| lane.cue).collect())
    }

    #[test]
    fn gemma_cold_stable_plan_accepts_turn0_shape_as_replay() {
        let (prompts, params) = lanes(&[(458, 465); 4]);
        let plan = plan_for(&ctx(), &prompts, &params).expect("turn-0 cohort");
        assert_eq!(
            plan.lanes,
            vec![
                StableLane {
                    boundary: 458,
                    prompt_len: 465,
                    cue: CueRoute::Replay,
                };
                4
            ]
        );
        assert_eq!(plan.lanes[0].cue_rows(), 7);
    }

    #[test]
    fn gemma_cold_stable_plan_cue_routing() {
        assert_eq!(
            routes(&[(400, 407), (500, 507)]),
            Some(vec![CueRoute::Replay; 2])
        );
        assert_eq!(
            routes(&[(500, 540), (600, 640)]),
            Some(vec![CueRoute::JointLive; 2])
        );
        assert_eq!(
            routes(&[(300, 332), (400, 432)]),
            Some(vec![CueRoute::PerSlot; 2])
        );
        assert_eq!(
            routes(&[(400, 407), (500, 540)]),
            Some(vec![CueRoute::Replay, CueRoute::PerSlot])
        );
        assert_eq!(
            routes(&[(300, 332), (500, 540)]),
            Some(vec![CueRoute::PerSlot, CueRoute::PerSlot])
        );
    }

    #[test]
    fn gemma_cold_stable_plan_boundary_and_cap_refusals() {
        // S = 32 is refused, S = 33 is accepted.
        assert!(routes(&[(32, 39), (458, 465)]).is_none());
        assert!(routes(&[(33, 40), (458, 465)]).is_some());
        // No cue (P = S).
        let (prompts, mut params) = lanes(&[(458, 465), (458, 465)]);
        params[1].stable_prompt_prefix_tokens = Some(465);
        assert!(plan_for(&ctx(), &prompts, &params).is_none());
        // Σ P = 4,097 is refused, 4,096 accepted.
        assert!(routes(&[(2000, 2048), (2041, 2049)]).is_none());
        assert!(routes(&[(2000, 2048), (2040, 2048)]).is_some());
        // A single lane.
        assert!(routes(&[(458, 465)]).is_none());
        // A lane without a boundary.
        let (prompts, mut params) = lanes(&[(458, 465), (458, 465)]);
        params[0].stable_prompt_prefix_tokens = None;
        assert!(plan_for(&ctx(), &prompts, &params).is_none());
    }

    #[test]
    fn gemma_cold_stable_plan_request_shape_refusals() {
        let (prompts, base) = lanes(&[(458, 465), (458, 465)]);
        let refused = |edit: &dyn Fn(&mut SamplingParams)| {
            let mut params = base.clone();
            edit(&mut params[1]);
            plan_for(&ctx(), &prompts, &params).is_none()
        };
        assert!(refused(&|p| p.vision_fingerprint = Some([7; 32])));
        assert!(refused(&|p| p.max_tokens = 1));
        assert!(refused(&|p| p.max_tokens = 0));
        // Tool-calling turns carry a grammar (the four-agent turn 0 does).
        assert!(!refused(&|p| {
            p.grammar = Some(crate::serve::api::grammar::parser::parse("root ::= \"a\"").unwrap())
        }));
        assert!(!refused(&|p| p.max_tokens = 2));
    }

    #[test]
    fn gemma_cold_stable_plan_substrate_refusals() {
        let (prompts, params) = lanes(&[(458, 465), (458, 465)]);
        for ctx in [
            StableContext {
                worker_active: false,
                ..ctx()
            },
            StableContext {
                all_slots_idle: false,
                ..ctx()
            },
            StableContext {
                graft_len: 1,
                ..ctx()
            },
            StableContext {
                max_seq_len: 400,
                ..ctx()
            },
        ] {
            assert!(plan_for(&ctx, &prompts, &params).is_none(), "{ctx:?}");
        }
    }

    #[test]
    fn gemma_cold_stable_plan_any_cached_lane_keeps_todays_route() {
        let (prompts, params) = lanes(&[(458, 465), (458, 465), (458, 465)]);
        let mut asked = Vec::new();
        let plan = stable_plan(
            &ctx(),
            prompts.iter().map(Vec::as_slice).zip(&params),
            |lane| {
                asked.push(lane);
                (lane == 1).then_some(Gemma4SlotPreference {
                    slot: SlotId(3),
                    cached_tokens: 400,
                    restore_prompt_anchor: true,
                })
            },
            |_| true,
        );
        assert!(plan.is_none());
        assert_eq!(asked, [0, 1]);
    }

    #[test]
    fn gemma_cold_stable_plan_anchor_quota_refusal() {
        let (prompts, params) = lanes(&[(458, 465); 4]);
        let mut seen = None;
        let plan = stable_plan(
            &ctx(),
            prompts.iter().map(Vec::as_slice).zip(&params),
            |_| None,
            |lanes| {
                seen = Some(lanes.len());
                // A ledger in which the serial order would refuse the fourth
                // capture (`anchor_budget::serial_captures_fit`).
                crate::inference::models::gemma4::kv_cache::anchor_budget::serial_captures_fit(
                    0,
                    3 * 100,
                    lanes.iter().map(|_| 100),
                )
            },
        );
        assert!(plan.is_none());
        assert_eq!(seen, Some(4));
    }

    fn env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    #[test]
    fn gemma_cold_worker_decision_switch_and_substrate() {
        let on = worker_decision(None, true, true, env(&[]));
        assert!(on.requested && on.active && on.unavailable.is_none());
        for raw in ["0", "off", "false", " FALSE "] {
            let off = worker_decision(Some(raw), true, true, env(&[]));
            assert!(!off.requested && !off.active, "{raw}");
        }
        for raw in ["1", "on", "true", ""] {
            assert!(
                worker_decision(Some(raw), true, true, env(&[])).active,
                "{raw}"
            );
        }
        let invalid = worker_decision(Some("maybe"), true, true, env(&[]));
        assert!(invalid.active && invalid.invalid.is_some());
        assert!(!worker_decision(None, false, true, env(&[])).active);
        assert!(!worker_decision(None, true, false, env(&[])).active);
        for pairs in [
            &[("HF2Q_BATCHED_BODY", "0")][..],
            &[("HF2Q_PREFILL_SLOT_BATCHED", "0")],
            &[("HF2Q_MM_ID_ROUTING_THRESHOLD", "32")],
            &[("HF2Q_DFLASH_XLEN_SDPA", "1")],
            &[("HF2Q_FA_F16", "0")],
            &[("HF2Q_NO_FA", "1")],
            &[("HF2Q_GLOBAL_FA", "1")],
        ] {
            let pairs: &'static [(&'static str, &'static str)] = pairs;
            let decision = worker_decision(None, true, true, env(pairs));
            assert!(
                decision.requested && !decision.active && decision.unavailable.is_some(),
                "{pairs:?}"
            );
        }
        assert!(worker_decision(None, true, true, env(&[("HF2Q_BATCHED_BODY", "1")])).active);
        assert!(worker_decision(None, true, true, env(&[("HF2Q_FA_F16", "1")])).active);
    }
}
