//! Server policy, isolation state, and telemetry for exact Qwen speculation.
//!
//! SlotAware request state, proposer cost gates, and KV transactions live in
//! `engine_qwen35`; this module owns admission policy and process telemetry.
//! Sampled requests still fail closed to ordinary target decode because the
//! server path has not qualified rejection sampling for every API constraint.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Server policy selected with `HF2Q_QWEN_SPECULATION`.
///
/// An explicit `off`/`auto` always wins. When the variable is unset the
/// default depends on the worker ([`default_server_policy`]): `off` for a
/// dense artifact on a SlotAware worker that selected joint decode and has
/// at least [`UNSET_OFF_MIN_SLOTS`] slots, where ADR-063 D1 measured a
/// four-agent makespan loss for `auto` (Qwen3.8); `auto` everywhere else
/// (ADR-044, updated 2026-08-21: every admission path fails closed, so the
/// worst case is ordinary decode plus telemetry).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenSpeculationPolicy {
    Off,
    Auto,
}

/// Narrowest SlotAware width at which an unset policy resolves to `off` for
/// dense joint-decode workers. ADR-063 D1 (#364): at four slots, Qwen3.8
/// `auto` vs `off` own-output makespan ratio 1.020 (95% CI 1.0095-1.0308,
/// 8 adjacent ABBA pairs, every pair slower under `auto`; receipts
/// `adr063-preflight-20261023/d1-speculation-ext-v2`).
pub const UNSET_OFF_MIN_SLOTS: usize = 4;

/// Unset default for a worker serving `max_slots` slots. Only the measured
/// configuration defaults `off`: a dense artifact whose worker selected joint
/// decode (where each MTP-holding lane pays its catch-up inside joint steps)
/// at [`UNSET_OFF_MIN_SLOTS`] or more slots. SerialFifo, narrower workers,
/// workers whose joint decode is off or unavailable, and MoE artifacts
/// (Qwen3.6, not measured as a loss) keep `auto`.
pub fn default_server_policy(
    max_slots: usize,
    moe: bool,
    joint_decode: bool,
) -> QwenSpeculationPolicy {
    if !moe && joint_decode && max_slots >= UNSET_OFF_MIN_SLOTS {
        QwenSpeculationPolicy::Off
    } else {
        QwenSpeculationPolicy::Auto
    }
}

const DENSE_WIDE_DEFAULT_OFF_REASON: &str = "unset: ADR-063 D1 measured a dense four-slot \
     joint-decode makespan loss for auto (ratio 1.020); set HF2Q_QWEN_SPECULATION=auto to opt in";
const DEFAULT_AUTO_REASON: &str = "unset: auto default (ADR-044); ADR-063 D1 disables it only \
     for dense joint-decode workers with max_slots >= 4";
const EXPLICIT_REASON: &str = "explicit HF2Q_QWEN_SPECULATION";

impl QwenSpeculationPolicy {
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        match value.map(str::trim).filter(|value| !value.is_empty()) {
            None | Some("off") | Some("0") => Ok(Self::Off),
            Some("auto") | Some("1") => Ok(Self::Auto),
            Some(value) => Err(format!(
                "HF2Q_QWEN_SPECULATION must be off or auto (got {value:?})"
            )),
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
        }
    }

    /// The operator's explicit choice: `None` when the variable is absent;
    /// a present value is parsed strictly and an invalid value warns and
    /// fails closed to an explicit `off` (it never inherits the default).
    /// Kept pure (the caller passes the env read) so it is unit-testable
    /// without mutating process state.
    pub fn requested(value: Option<&str>) -> Option<Self> {
        let value = value?;
        Some(Self::parse(Some(value)).unwrap_or_else(|message| {
            tracing::warn!(%message, "disabling Qwen speculation");
            Self::Off
        }))
    }

    pub fn requested_from_environment() -> Option<Self> {
        Self::requested(std::env::var("HF2Q_QWEN_SPECULATION").ok().as_deref())
    }

    /// Server-side resolution for the qwen35 engine: an explicit request
    /// wins; an unset one takes [`default_server_policy`] for a worker of
    /// `max_slots` slots serving a dense (`!moe`) or MoE artifact, with or
    /// without joint decode selected. Pure.
    pub fn resolve_server_policy(
        requested: Option<Self>,
        max_slots: usize,
        moe: bool,
        joint_decode: bool,
    ) -> Self {
        requested.unwrap_or_else(|| default_server_policy(max_slots, moe, joint_decode))
    }

    fn resolution_reason(
        requested: Option<Self>,
        max_slots: usize,
        moe: bool,
        joint_decode: bool,
    ) -> &'static str {
        match (requested, default_server_policy(max_slots, moe, joint_decode)) {
            (Some(_), _) => EXPLICIT_REASON,
            (None, Self::Off) => DENSE_WIDE_DEFAULT_OFF_REASON,
            (None, Self::Auto) => DEFAULT_AUTO_REASON,
        }
    }
}

/// Why a Qwen request used ordinary target decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenSpeculationDecision {
    Disabled,
    UnsupportedSemantics,
    PromptCacheHit,
    Unprofitable,
    RuntimeUnavailable,
    Eligible,
}

impl QwenSpeculationDecision {
    pub const fn metric_reason(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::UnsupportedSemantics => "unsupported_semantics",
            Self::PromptCacheHit => "prompt_cache_hit",
            Self::Unprofitable => "unprofitable",
            Self::RuntimeUnavailable => "runtime_unavailable",
            Self::Eligible => "eligible",
        }
    }
}

/// Classify a request without retaining conversation-local state. SlotAware
/// scheduling uses this at admission; the per-slot decoder owns all
/// request-local proposer and KV state and never shares a serial controller.
pub fn classify_request(
    policy: QwenSpeculationPolicy,
    exact_semantics: bool,
    prompt_cache_hit: bool,
) -> QwenSpeculationDecision {
    if policy == QwenSpeculationPolicy::Off {
        QwenSpeculationDecision::Disabled
    } else if !exact_semantics {
        QwenSpeculationDecision::UnsupportedSemantics
    } else if prompt_cache_hit {
        QwenSpeculationDecision::PromptCacheHit
    } else {
        QwenSpeculationDecision::Eligible
    }
}

/// SlotAware admission decision: [`classify_request`] plus the bound-graft
/// fallback (ADR-059). The MTP cache is a separate slot arena that never
/// receives the graft, so its drafts and any retained speculative boundary
/// describe the ungrafted context. Speculation is an optimization, never a
/// semantic path: an otherwise eligible grafted request is served by ordinary
/// grafted decode and reported as `RuntimeUnavailable`.
pub fn classify_slot_request(
    policy: QwenSpeculationPolicy,
    exact_semantics: bool,
    prompt_cache_hit: bool,
    graft_bound: bool,
) -> QwenSpeculationDecision {
    match classify_request(policy, exact_semantics, prompt_cache_hit) {
        QwenSpeculationDecision::Eligible if graft_bound => {
            QwenSpeculationDecision::RuntimeUnavailable
        }
        decision => decision,
    }
}

/// Whether a bounded SlotAware prefill chunk also runs the MTP prompt
/// catch-up that seeds K3 drafting. It needs `auto`, exact request semantics,
/// a text-only chunk, MTP weights with an MTP cache arena, and no bound KV
/// graft (same reason as [`classify_slot_request`]).
pub fn prefill_mtp_catchup_allowed(
    policy: QwenSpeculationPolicy,
    exact_semantics: bool,
    text_only: bool,
    mtp_runtime: bool,
    graft_bound: bool,
) -> bool {
    policy == QwenSpeculationPolicy::Auto
        && exact_semantics
        && text_only
        && mtp_runtime
        && !graft_bound
}

/// One-worker controller. It holds no KV tensors and only carries the
/// current conversation identity plus the previous speculative result.
#[derive(Debug, Clone)]
pub struct QwenSpeculationController {
    /// Operator choice (`None` = `HF2Q_QWEN_SPECULATION` unset).
    requested: Option<QwenSpeculationPolicy>,
    /// Policy every admission/prefill/decode consumer reads.
    policy: QwenSpeculationPolicy,
    conversation_prompt: Option<Vec<u32>>,
}

impl QwenSpeculationController {
    /// Controller with an explicit policy; worker resolution keeps it.
    pub fn new(policy: QwenSpeculationPolicy) -> Self {
        Self {
            requested: Some(policy),
            policy,
            conversation_prompt: None,
        }
    }

    /// Reads `HF2Q_QWEN_SPECULATION` once at engine load. Until a SlotAware
    /// worker calls [`Self::resolve_for_worker`], the policy is the
    /// one-slot (SerialFifo) resolution, which is `auto` when unset.
    pub fn from_environment() -> Self {
        let requested = QwenSpeculationPolicy::requested_from_environment();
        Self {
            requested,
            policy: requested.unwrap_or(QwenSpeculationPolicy::Auto),
            conversation_prompt: None,
        }
    }

    /// Resolve the unset default for a SlotAware worker of `max_slots`
    /// slots once at worker start (ADR-063 D1) and log the decision.
    /// `joint_decode` is whether the worker selected Qwen joint decode.
    pub fn resolve_for_worker(&mut self, max_slots: usize, moe: bool, joint_decode: bool) {
        let requested = self.requested;
        self.policy =
            QwenSpeculationPolicy::resolve_server_policy(requested, max_slots, moe, joint_decode);
        let reason =
            QwenSpeculationPolicy::resolution_reason(requested, max_slots, moe, joint_decode);
        tracing::info!(
            target: "hf2q::serve::api::engine_qwen35::progress",
            requested = requested.map_or("unset", QwenSpeculationPolicy::label),
            policy = self.policy.label(),
            max_slots,
            moe,
            joint_decode,
            reason,
            "Qwen speculation worker policy"
        );
    }

    pub fn policy(&self) -> QwenSpeculationPolicy {
        self.policy
    }

    /// Decide the transaction path. A non-extension resets the state before
    /// inspection, which prevents one chat's negative acceptance outcome from
    /// disabling (or enabling) another chat.
    pub fn decide(
        &mut self,
        prompt: &[u32],
        greedy_exact: bool,
        _has_mtp: bool,
        prompt_cache_hit: bool,
    ) -> QwenSpeculationDecision {
        let same_conversation = self
            .conversation_prompt
            .as_ref()
            .is_some_and(|prior| prompt.starts_with(prior));
        if !same_conversation {
            self.conversation_prompt = None;
            TELEMETRY
                .conversation_resets
                .fetch_add(1, Ordering::Relaxed);
        }
        self.conversation_prompt = Some(prompt.to_vec());

        classify_request(self.policy, greedy_exact, prompt_cache_hit)
    }
}

/// Process telemetry, emitted by the server `/metrics` endpoint and tracing.
#[derive(Debug, Default)]
pub struct QwenSpeculationTelemetry {
    pub drafted_tokens: AtomicU64,
    pub accepted_tokens: AtomicU64,
    pub rejected_tokens: AtomicU64,
    pub target_forwards: AtomicU64,
    pub cached_token_reuse: AtomicU64,
    pub fallback_requests: AtomicU64,
    pub fallback_disabled: AtomicU64,
    pub fallback_unsupported_semantics: AtomicU64,
    // Retained as a zero-valued compatibility metric while `auto` can use
    // history lookup without MTP weights.
    pub fallback_no_mtp_weights: AtomicU64,
    pub fallback_prompt_cache_hit: AtomicU64,
    pub fallback_unprofitable: AtomicU64,
    pub fallback_slot_aware_unavailable: AtomicU64,
    pub fallback_runtime_unavailable: AtomicU64,
    pub conversation_resets: AtomicU64,
    pub history_lookup_proposals: AtomicU64,
    pub mtp_proposals: AtomicU64,
    pub history_lookup_no_match: AtomicU64,
    pub history_lookup_drafted_tokens: AtomicU64,
    pub history_lookup_accepted_tokens: AtomicU64,
    pub history_lookup_rejected_tokens: AtomicU64,
    pub history_lookup_target_forwards: AtomicU64,
    pub mtp_drafted_tokens: AtomicU64,
    pub mtp_accepted_tokens: AtomicU64,
    pub mtp_rejected_tokens: AtomicU64,
    pub mtp_target_forwards: AtomicU64,
    pub history_lookup_cost_disabled: AtomicU64,
    pub mtp_cost_disabled: AtomicU64,
    pub history_lookup_round_nanos: AtomicU64,
    pub mtp_round_nanos: AtomicU64,
    pub history_lookup_equivalent_ordinary_nanos: AtomicU64,
    pub mtp_equivalent_ordinary_nanos: AtomicU64,
}

pub static TELEMETRY: QwenSpeculationTelemetry = QwenSpeculationTelemetry {
    drafted_tokens: AtomicU64::new(0),
    accepted_tokens: AtomicU64::new(0),
    rejected_tokens: AtomicU64::new(0),
    target_forwards: AtomicU64::new(0),
    cached_token_reuse: AtomicU64::new(0),
    fallback_requests: AtomicU64::new(0),
    fallback_disabled: AtomicU64::new(0),
    fallback_unsupported_semantics: AtomicU64::new(0),
    fallback_no_mtp_weights: AtomicU64::new(0),
    fallback_prompt_cache_hit: AtomicU64::new(0),
    fallback_unprofitable: AtomicU64::new(0),
    fallback_slot_aware_unavailable: AtomicU64::new(0),
    fallback_runtime_unavailable: AtomicU64::new(0),
    conversation_resets: AtomicU64::new(0),
    history_lookup_proposals: AtomicU64::new(0),
    mtp_proposals: AtomicU64::new(0),
    history_lookup_no_match: AtomicU64::new(0),
    history_lookup_drafted_tokens: AtomicU64::new(0),
    history_lookup_accepted_tokens: AtomicU64::new(0),
    history_lookup_rejected_tokens: AtomicU64::new(0),
    history_lookup_target_forwards: AtomicU64::new(0),
    mtp_drafted_tokens: AtomicU64::new(0),
    mtp_accepted_tokens: AtomicU64::new(0),
    mtp_rejected_tokens: AtomicU64::new(0),
    mtp_target_forwards: AtomicU64::new(0),
    history_lookup_cost_disabled: AtomicU64::new(0),
    mtp_cost_disabled: AtomicU64::new(0),
    history_lookup_round_nanos: AtomicU64::new(0),
    mtp_round_nanos: AtomicU64::new(0),
    history_lookup_equivalent_ordinary_nanos: AtomicU64::new(0),
    mtp_equivalent_ordinary_nanos: AtomicU64::new(0),
};

pub fn record_fallback(decision: QwenSpeculationDecision) {
    TELEMETRY.fallback_requests.fetch_add(1, Ordering::Relaxed);
    let counter = match decision {
        QwenSpeculationDecision::Disabled => &TELEMETRY.fallback_disabled,
        QwenSpeculationDecision::UnsupportedSemantics => &TELEMETRY.fallback_unsupported_semantics,
        QwenSpeculationDecision::PromptCacheHit => &TELEMETRY.fallback_prompt_cache_hit,
        QwenSpeculationDecision::Unprofitable => &TELEMETRY.fallback_unprofitable,
        QwenSpeculationDecision::RuntimeUnavailable => &TELEMETRY.fallback_runtime_unavailable,
        QwenSpeculationDecision::Eligible => return,
    };
    counter.fetch_add(1, Ordering::Relaxed);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenSpeculationProposer {
    HistoryLookup,
    Mtp,
}

pub fn record_history_lookup_no_match() {
    TELEMETRY
        .history_lookup_no_match
        .fetch_add(1, Ordering::Relaxed);
}

pub fn record_outcome(
    drafted: usize,
    accepted: usize,
    rejected: usize,
    target_forwards: usize,
    cached_tokens: usize,
) {
    TELEMETRY
        .drafted_tokens
        .fetch_add(drafted as u64, Ordering::Relaxed);
    TELEMETRY
        .accepted_tokens
        .fetch_add(accepted as u64, Ordering::Relaxed);
    TELEMETRY
        .rejected_tokens
        .fetch_add(rejected as u64, Ordering::Relaxed);
    TELEMETRY
        .target_forwards
        .fetch_add(target_forwards as u64, Ordering::Relaxed);
    TELEMETRY
        .cached_token_reuse
        .fetch_add(cached_tokens as u64, Ordering::Relaxed);
}

pub fn record_proposer_outcome(
    proposer: QwenSpeculationProposer,
    drafted: usize,
    accepted: usize,
    rejected: usize,
    target_forwards: usize,
    cached_tokens: usize,
) {
    match proposer {
        QwenSpeculationProposer::HistoryLookup => {
            TELEMETRY
                .history_lookup_proposals
                .fetch_add(1, Ordering::Relaxed);
            TELEMETRY
                .history_lookup_drafted_tokens
                .fetch_add(drafted as u64, Ordering::Relaxed);
            TELEMETRY
                .history_lookup_accepted_tokens
                .fetch_add(accepted as u64, Ordering::Relaxed);
            TELEMETRY
                .history_lookup_rejected_tokens
                .fetch_add(rejected as u64, Ordering::Relaxed);
            TELEMETRY
                .history_lookup_target_forwards
                .fetch_add(target_forwards as u64, Ordering::Relaxed);
        }
        QwenSpeculationProposer::Mtp => {
            TELEMETRY.mtp_proposals.fetch_add(1, Ordering::Relaxed);
            TELEMETRY
                .mtp_drafted_tokens
                .fetch_add(drafted as u64, Ordering::Relaxed);
            TELEMETRY
                .mtp_accepted_tokens
                .fetch_add(accepted as u64, Ordering::Relaxed);
            TELEMETRY
                .mtp_rejected_tokens
                .fetch_add(rejected as u64, Ordering::Relaxed);
            TELEMETRY
                .mtp_target_forwards
                .fetch_add(target_forwards as u64, Ordering::Relaxed);
        }
    };
    record_outcome(drafted, accepted, rejected, target_forwards, cached_tokens);
}

pub fn record_cost_disabled(proposer: QwenSpeculationProposer) {
    let counter = match proposer {
        QwenSpeculationProposer::HistoryLookup => &TELEMETRY.history_lookup_cost_disabled,
        QwenSpeculationProposer::Mtp => &TELEMETRY.mtp_cost_disabled,
    };
    counter.fetch_add(1, Ordering::Relaxed);
}

pub fn record_proposer_timing(
    proposer: QwenSpeculationProposer,
    elapsed: Duration,
    equivalent_ordinary: Duration,
) {
    let elapsed = elapsed.as_nanos().min(u128::from(u64::MAX)) as u64;
    let equivalent = equivalent_ordinary.as_nanos().min(u128::from(u64::MAX)) as u64;
    let (round, baseline) = match proposer {
        QwenSpeculationProposer::HistoryLookup => (
            &TELEMETRY.history_lookup_round_nanos,
            &TELEMETRY.history_lookup_equivalent_ordinary_nanos,
        ),
        QwenSpeculationProposer::Mtp => (
            &TELEMETRY.mtp_round_nanos,
            &TELEMETRY.mtp_equivalent_ordinary_nanos,
        ),
    };
    round.fetch_add(elapsed, Ordering::Relaxed);
    baseline.fetch_add(equivalent, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    static TELEMETRY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn policy_parser_is_explicit_and_fail_closed() {
        assert_eq!(
            QwenSpeculationPolicy::parse(None),
            Ok(QwenSpeculationPolicy::Off)
        );
        assert_eq!(
            QwenSpeculationPolicy::parse(Some("auto")),
            Ok(QwenSpeculationPolicy::Auto)
        );
        assert!(QwenSpeculationPolicy::parse(Some("always")).is_err());
    }

    #[test]
    fn requested_policy_is_explicit_or_unset_and_fails_closed() {
        assert_eq!(QwenSpeculationPolicy::requested(None), None);
        for (raw, expected) in [
            ("auto", QwenSpeculationPolicy::Auto),
            ("1", QwenSpeculationPolicy::Auto),
            ("off", QwenSpeculationPolicy::Off),
            ("0", QwenSpeculationPolicy::Off),
            ("", QwenSpeculationPolicy::Off),
            // Invalid values fail closed rather than inheriting the default.
            ("always", QwenSpeculationPolicy::Off),
        ] {
            assert_eq!(
                QwenSpeculationPolicy::requested(Some(raw)),
                Some(expected),
                "HF2Q_QWEN_SPECULATION={raw:?}"
            );
        }
    }

    /// ADR-063 D1: unset resolves to `off` only for dense joint-decode
    /// workers with at least four slots; explicit values always win.
    #[test]
    fn server_policy_resolution_table() {
        use QwenSpeculationPolicy::{Auto, Off};
        let invalid = QwenSpeculationPolicy::requested(Some("always"));
        // (requested, max_slots, moe, joint_decode, expected)
        let table = [
            (None, 4, false, true, Off),
            (None, 8, false, true, Off),
            (None, 3, false, true, Auto),
            (None, 1, false, true, Auto),
            (None, 4, false, false, Auto),
            (None, 8, false, false, Auto),
            (None, 4, true, false, Auto),
            (None, 4, true, true, Auto),
            (None, 8, true, true, Auto),
            (None, 1, true, false, Auto),
            (Some(Auto), 4, false, true, Auto),
            (Some(Auto), 8, false, true, Auto),
            (Some(Off), 1, false, false, Off),
            (Some(Off), 3, false, true, Off),
            (Some(Off), 4, false, true, Off),
            (Some(Off), 1, true, false, Off),
            (Some(Off), 4, true, true, Off),
            (invalid, 1, true, false, Off),
            (invalid, 4, false, true, Off),
        ];
        for (requested, max_slots, moe, joint, expected) in table {
            assert_eq!(
                QwenSpeculationPolicy::resolve_server_policy(requested, max_slots, moe, joint),
                expected,
                "requested={requested:?} max_slots={max_slots} moe={moe} joint={joint}"
            );
        }
    }

    #[test]
    fn worker_resolution_replaces_only_the_unset_default() {
        let unset = || QwenSpeculationController {
            requested: None,
            policy: QwenSpeculationPolicy::Auto,
            conversation_prompt: None,
        };
        let mut dense_joint_four = unset();
        dense_joint_four.resolve_for_worker(4, false, true);
        assert_eq!(dense_joint_four.policy(), QwenSpeculationPolicy::Off);
        let mut dense_serial_four = unset();
        dense_serial_four.resolve_for_worker(4, false, false);
        assert_eq!(dense_serial_four.policy(), QwenSpeculationPolicy::Auto);
        let mut moe_four = unset();
        moe_four.resolve_for_worker(4, true, false);
        assert_eq!(moe_four.policy(), QwenSpeculationPolicy::Auto);
        let mut dense_joint_three = unset();
        dense_joint_three.resolve_for_worker(3, false, true);
        assert_eq!(dense_joint_three.policy(), QwenSpeculationPolicy::Auto);

        let mut explicit_auto = QwenSpeculationController::new(QwenSpeculationPolicy::Auto);
        explicit_auto.resolve_for_worker(4, false, true);
        assert_eq!(explicit_auto.policy(), QwenSpeculationPolicy::Auto);
        let mut explicit_off = QwenSpeculationController::new(QwenSpeculationPolicy::Off);
        explicit_off.resolve_for_worker(1, true, false);
        assert_eq!(explicit_off.policy(), QwenSpeculationPolicy::Off);
    }

    #[test]
    fn sampled_requests_and_prompt_cache_hits_fall_back() {
        let mut controller = QwenSpeculationController::new(QwenSpeculationPolicy::Auto);
        assert_eq!(
            controller.decide(&[1, 2], false, true, false),
            QwenSpeculationDecision::UnsupportedSemantics
        );
        assert_eq!(
            controller.decide(&[1, 2, 3], true, true, true),
            QwenSpeculationDecision::PromptCacheHit
        );
    }

    #[test]
    fn eligibility_does_not_use_acceptance_as_a_profitability_proxy() {
        let mut controller = QwenSpeculationController::new(QwenSpeculationPolicy::Auto);
        assert_eq!(
            controller.decide(&[7, 8], true, true, false),
            QwenSpeculationDecision::Eligible
        );
        assert_eq!(
            controller.decide(&[7, 8, 9], true, true, false),
            QwenSpeculationDecision::Eligible
        );
    }

    /// ADR-063 D1 graft-restore fallback: a bound graft never reaches MTP,
    /// whether the request would start cold or resume a retained speculative
    /// boundary, while every non-eligible reason keeps its own label.
    #[test]
    fn bound_graft_keeps_eligible_requests_off_mtp_at_admission_and_prefill() {
        let auto = QwenSpeculationPolicy::Auto;
        assert_eq!(
            classify_slot_request(auto, true, false, false),
            QwenSpeculationDecision::Eligible
        );
        assert_eq!(
            classify_slot_request(auto, true, false, true),
            QwenSpeculationDecision::RuntimeUnavailable,
            "a warm grafted slot with reusable speculative metadata must not resume MTP"
        );
        for graft_bound in [false, true] {
            assert_eq!(
                classify_slot_request(QwenSpeculationPolicy::Off, true, false, graft_bound),
                QwenSpeculationDecision::Disabled
            );
            assert_eq!(
                classify_slot_request(auto, false, false, graft_bound),
                QwenSpeculationDecision::UnsupportedSemantics
            );
            assert_eq!(
                classify_slot_request(auto, true, true, graft_bound),
                QwenSpeculationDecision::PromptCacheHit
            );
        }

        assert!(prefill_mtp_catchup_allowed(auto, true, true, true, false));
        assert!(
            !prefill_mtp_catchup_allowed(auto, true, true, true, true),
            "a grafted prefill must not seed the ungrafted MTP arena"
        );
        assert!(!prefill_mtp_catchup_allowed(
            QwenSpeculationPolicy::Off,
            true,
            true,
            true,
            false
        ));
        assert!(!prefill_mtp_catchup_allowed(auto, false, true, true, false));
        assert!(!prefill_mtp_catchup_allowed(auto, true, false, true, false));
        assert!(!prefill_mtp_catchup_allowed(auto, true, true, false, false));
    }

    #[test]
    fn telemetry_records_every_transaction_counter() {
        let _guard = TELEMETRY_TEST_LOCK.lock().expect("telemetry test lock");
        let before = (
            TELEMETRY.drafted_tokens.load(Ordering::Relaxed),
            TELEMETRY.accepted_tokens.load(Ordering::Relaxed),
            TELEMETRY.rejected_tokens.load(Ordering::Relaxed),
            TELEMETRY.target_forwards.load(Ordering::Relaxed),
            TELEMETRY.cached_token_reuse.load(Ordering::Relaxed),
        );
        record_outcome(8, 5, 3, 4, 12);
        assert_eq!(
            TELEMETRY.drafted_tokens.load(Ordering::Relaxed),
            before.0 + 8
        );
        assert_eq!(
            TELEMETRY.accepted_tokens.load(Ordering::Relaxed),
            before.1 + 5
        );
        assert_eq!(
            TELEMETRY.rejected_tokens.load(Ordering::Relaxed),
            before.2 + 3
        );
        assert_eq!(
            TELEMETRY.target_forwards.load(Ordering::Relaxed),
            before.3 + 4
        );
        assert_eq!(
            TELEMETRY.cached_token_reuse.load(Ordering::Relaxed),
            before.4 + 12
        );
    }

    #[test]
    fn telemetry_keeps_proposer_and_cost_gate_counters_separate() {
        let _guard = TELEMETRY_TEST_LOCK.lock().expect("telemetry test lock");
        let before = (
            TELEMETRY.history_lookup_proposals.load(Ordering::Relaxed),
            TELEMETRY
                .history_lookup_accepted_tokens
                .load(Ordering::Relaxed),
            TELEMETRY.mtp_proposals.load(Ordering::Relaxed),
            TELEMETRY.mtp_rejected_tokens.load(Ordering::Relaxed),
            TELEMETRY
                .history_lookup_cost_disabled
                .load(Ordering::Relaxed),
            TELEMETRY.mtp_cost_disabled.load(Ordering::Relaxed),
        );

        record_proposer_outcome(QwenSpeculationProposer::HistoryLookup, 3, 2, 1, 2, 0);
        record_proposer_outcome(QwenSpeculationProposer::Mtp, 4, 1, 3, 2, 0);
        record_cost_disabled(QwenSpeculationProposer::HistoryLookup);
        record_cost_disabled(QwenSpeculationProposer::Mtp);

        assert_eq!(
            TELEMETRY.history_lookup_proposals.load(Ordering::Relaxed),
            before.0 + 1
        );
        assert_eq!(
            TELEMETRY
                .history_lookup_accepted_tokens
                .load(Ordering::Relaxed),
            before.1 + 2
        );
        assert_eq!(
            TELEMETRY.mtp_proposals.load(Ordering::Relaxed),
            before.2 + 1
        );
        assert_eq!(
            TELEMETRY.mtp_rejected_tokens.load(Ordering::Relaxed),
            before.3 + 3
        );
        assert_eq!(
            TELEMETRY
                .history_lookup_cost_disabled
                .load(Ordering::Relaxed),
            before.4 + 1
        );
        assert_eq!(
            TELEMETRY.mtp_cost_disabled.load(Ordering::Relaxed),
            before.5 + 1
        );
    }

    #[test]
    fn fallback_reason_is_counted_separately() {
        let _guard = TELEMETRY_TEST_LOCK.lock().expect("telemetry test lock");
        let total = TELEMETRY.fallback_requests.load(Ordering::Relaxed);
        let slot_aware = TELEMETRY
            .fallback_runtime_unavailable
            .load(Ordering::Relaxed);
        record_fallback(QwenSpeculationDecision::RuntimeUnavailable);
        assert_eq!(
            TELEMETRY.fallback_requests.load(Ordering::Relaxed),
            total + 1
        );
        assert_eq!(
            TELEMETRY
                .fallback_runtime_unavailable
                .load(Ordering::Relaxed),
            slot_aware + 1
        );
    }

    #[test]
    fn telemetry_records_measured_round_and_equivalent_baseline_time() {
        let _guard = TELEMETRY_TEST_LOCK.lock().expect("telemetry test lock");
        let before = (
            TELEMETRY.mtp_round_nanos.load(Ordering::Relaxed),
            TELEMETRY
                .mtp_equivalent_ordinary_nanos
                .load(Ordering::Relaxed),
        );
        record_proposer_timing(
            QwenSpeculationProposer::Mtp,
            Duration::from_millis(17),
            Duration::from_millis(24),
        );
        assert_eq!(
            TELEMETRY.mtp_round_nanos.load(Ordering::Relaxed),
            before.0 + 17_000_000
        );
        assert_eq!(
            TELEMETRY
                .mtp_equivalent_ordinary_nanos
                .load(Ordering::Relaxed),
            before.1 + 24_000_000
        );
    }
}
