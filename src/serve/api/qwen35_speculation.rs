//! Server policy, isolation state, and telemetry for exact Qwen speculation.
//!
//! SlotAware request state, proposer cost gates, and KV transactions live in
//! `engine_qwen35`; this module owns admission policy and process telemetry.
//! Sampled requests still fail closed to ordinary target decode because the
//! server path has not qualified rejection sampling for every API constraint.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Explicit server policy selected with `HF2Q_QWEN_SPECULATION`.
///
/// `auto` is the default for the qwen35 server engine when no policy is
/// supplied (ADR-044, updated 2026-08-21): the policy was already the
/// canonical Qwen3.8 launcher choice, and every admission path fails
/// closed — unsupported semantics, prompt-cache hits, unavailable
/// proposers, and measured-unprofitable rounds all stay on ordinary
/// target decode, so the worst case for a default-on `auto` is ordinary
/// decode plus telemetry. An explicit `off` remains the operator escape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenSpeculationPolicy {
    Off,
    Auto,
}

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

    /// Server-side resolution for the qwen35 engine: an absent variable
    /// defaults to `auto`; a present variable is parsed strictly and an
    /// invalid value warns and fails closed to `off`. Kept pure (the
    /// caller passes the env read) so the default is unit-testable
    /// without mutating process state.
    pub fn resolve_server_policy(value: Option<&str>) -> Self {
        match value {
            None => Self::Auto,
            Some(_) => match Self::parse(value) {
                Ok(policy) => policy,
                Err(message) => {
                    tracing::warn!(%message, "disabling Qwen speculation");
                    Self::Off
                }
            },
        }
    }

    pub fn from_environment() -> Self {
        Self::resolve_server_policy(std::env::var("HF2Q_QWEN_SPECULATION").ok().as_deref())
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

/// One-worker controller. It holds no KV tensors and only carries the
/// current conversation identity plus the previous speculative result.
#[derive(Debug, Clone)]
pub struct QwenSpeculationController {
    policy: QwenSpeculationPolicy,
    conversation_prompt: Option<Vec<u32>>,
}

impl QwenSpeculationController {
    pub fn new(policy: QwenSpeculationPolicy) -> Self {
        Self {
            policy,
            conversation_prompt: None,
        }
    }

    pub fn from_environment() -> Self {
        Self::new(QwenSpeculationPolicy::from_environment())
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
    fn server_policy_defaults_to_auto_and_preserves_explicit_off() {
        assert_eq!(
            QwenSpeculationPolicy::resolve_server_policy(None),
            QwenSpeculationPolicy::Auto
        );
        assert_eq!(
            QwenSpeculationPolicy::resolve_server_policy(Some("auto")),
            QwenSpeculationPolicy::Auto
        );
        assert_eq!(
            QwenSpeculationPolicy::resolve_server_policy(Some("1")),
            QwenSpeculationPolicy::Auto
        );
        assert_eq!(
            QwenSpeculationPolicy::resolve_server_policy(Some("off")),
            QwenSpeculationPolicy::Off
        );
        // Invalid values fail closed rather than inheriting the default.
        assert_eq!(
            QwenSpeculationPolicy::resolve_server_policy(Some("always")),
            QwenSpeculationPolicy::Off
        );
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
