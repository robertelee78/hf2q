//! Cost gate for one proposer in a speculative generation.
//!
//! The gate learns an ordinary target-decision cost first, then compares four
//! output-equivalent rounds against the complete proposal + verify/recovery
//! cost of those rounds, and disables the proposer after two consecutive
//! windows with no positive net gain.
//!
//! Two inputs, chosen per generation:
//! - **measured** ([`SpeculationCostController::new`]): wall-clock costs. The
//!   decision then depends on host load and cohort timing (Qwen
//!   `HF2Q_QWEN_SPECULATION_ROUTE=adaptive`).
//! - **modelled** ([`SpeculationCostController::modelled`]): an ordinary
//!   decision costs one unit and a round costs a fixed constant. A window is
//!   unprofitable exactly when its mean target-equivalent decisions per round
//!   are at most that constant, so the gate is an acceptance threshold against
//!   a fixed cost model and repeats exactly for an identical transcript (the
//!   repeatable route).
//!
//! Telemetry ([`SpeculationCostController::equivalent_ordinary_elapsed`])
//! always reports measured wall time, whichever input drives the gate.

use std::time::Duration;

const EVALUATION_ROUNDS: u8 = 4;
const UNPROFITABLE_WINDOWS_TO_DISABLE: u8 = 2;
/// Modelled cost resolution: one ordinary decision is this many units.
const MODELLED_DECISION_UNITS: u128 = 1_000;

#[derive(Debug, Clone, Default)]
pub struct SpeculationCostController {
    /// Gate inputs: nanoseconds (measured) or modelled units.
    ordinary_cost: u128,
    ordinary_samples: u32,
    speculative_cost: u128,
    equivalent_ordinary_cost: u128,
    rounds: u8,
    consecutive_unprofitable_windows: u8,
    speculation_enabled: bool,
    /// Fixed round cost in modelled units; `None` gates on wall time.
    modelled_round_cost: Option<u128>,
    /// Measured ordinary target time, for telemetry only.
    measured_ordinary_nanos: u128,
    measured_ordinary_samples: u32,
}

impl SpeculationCostController {
    /// Gate on measured wall time.
    pub fn new() -> Self {
        Self {
            speculation_enabled: true,
            ..Self::default()
        }
    }

    /// Gate on a fixed cost model: `round_cost_decisions` is one round's cost
    /// in ordinary target decisions (must be positive and finite).
    pub fn modelled(round_cost_decisions: f64) -> Self {
        assert!(
            round_cost_decisions.is_finite() && round_cost_decisions > 0.0,
            "modelled round cost must be positive and finite"
        );
        let units = (round_cost_decisions * MODELLED_DECISION_UNITS as f64).round() as u128;
        Self {
            speculation_enabled: true,
            modelled_round_cost: Some(units.max(1)),
            ..Self::default()
        }
    }

    pub fn observe_ordinary_target(&mut self, elapsed: Duration) {
        if !elapsed.is_zero() {
            self.measured_ordinary_nanos = self
                .measured_ordinary_nanos
                .saturating_add(elapsed.as_nanos());
            self.measured_ordinary_samples = self.measured_ordinary_samples.saturating_add(1);
        }
        let cost = match self.modelled_round_cost {
            Some(_) => MODELLED_DECISION_UNITS,
            None if elapsed.is_zero() => return,
            None => elapsed.as_nanos(),
        };
        self.ordinary_cost = self.ordinary_cost.saturating_add(cost);
        self.ordinary_samples = self.ordinary_samples.saturating_add(1);
    }

    /// A proposer is never warmed up on an unobserved target baseline.
    pub fn may_speculate(&self) -> bool {
        self.speculation_enabled && self.ordinary_samples > 0
    }

    /// Measured ordinary target time for the same number of output decisions,
    /// for telemetry. It is wall time even when the gate is modelled.
    pub fn equivalent_ordinary_elapsed(
        &self,
        equivalent_target_decisions: usize,
    ) -> Option<Duration> {
        if self.measured_ordinary_samples == 0 || equivalent_target_decisions == 0 {
            return None;
        }
        let per_decision =
            self.measured_ordinary_nanos / u128::from(self.measured_ordinary_samples);
        let nanos = per_decision.saturating_mul(equivalent_target_decisions as u128);
        Some(Duration::from_nanos(nanos.min(u128::from(u64::MAX)) as u64))
    }

    /// Record a completed target-equivalent speculative round.
    /// `equivalent_target_decisions` includes a terminal EOS decision even
    /// though EOS is not exposed as a completion token. `elapsed` is read only
    /// by the measured gate.
    /// Returns false after two consecutive full windows have no positive net
    /// gain. One negative window is treated as noise: draft acceptance and
    /// first-use pipeline cost are bursty enough that a permanent decision
    /// after four rounds can discard an otherwise profitable generation.
    pub fn observe_speculative_round(
        &mut self,
        equivalent_target_decisions: usize,
        elapsed: Duration,
    ) -> bool {
        if !self.may_speculate() || equivalent_target_decisions == 0 {
            return self.speculation_enabled;
        }
        let round_cost = match self.modelled_round_cost {
            Some(cost) => cost,
            None if elapsed.is_zero() => return self.speculation_enabled,
            None => elapsed.as_nanos(),
        };
        let ordinary = self.ordinary_cost / u128::from(self.ordinary_samples);
        self.equivalent_ordinary_cost = self
            .equivalent_ordinary_cost
            .saturating_add(ordinary.saturating_mul(equivalent_target_decisions as u128));
        self.speculative_cost = self.speculative_cost.saturating_add(round_cost);
        self.rounds = self.rounds.saturating_add(1);
        if self.rounds >= EVALUATION_ROUNDS {
            if self.speculative_cost >= self.equivalent_ordinary_cost {
                self.consecutive_unprofitable_windows =
                    self.consecutive_unprofitable_windows.saturating_add(1);
                if self.consecutive_unprofitable_windows >= UNPROFITABLE_WINDOWS_TO_DISABLE {
                    self.speculation_enabled = false;
                }
            } else {
                self.consecutive_unprofitable_windows = 0;
            }
            self.rounds = 0;
            self.speculative_cost = 0;
            self.equivalent_ordinary_cost = 0;
        }
        self.speculation_enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_ordinary_warmup_then_disables_two_non_positive_windows() {
        let mut controller = SpeculationCostController::new();
        assert!(!controller.may_speculate());
        controller.observe_ordinary_target(Duration::from_millis(10));
        assert!(controller.may_speculate());
        for _ in 0..7 {
            assert!(controller.observe_speculative_round(1, Duration::from_millis(10)));
        }
        assert!(!controller.observe_speculative_round(1, Duration::from_millis(10)));
        assert!(!controller.may_speculate());
    }

    #[test]
    fn profitable_window_clears_transient_negative_hysteresis() {
        let mut controller = SpeculationCostController::new();
        controller.observe_ordinary_target(Duration::from_millis(10));
        for _ in 0..4 {
            assert!(controller.observe_speculative_round(1, Duration::from_millis(11)));
        }
        for _ in 0..4 {
            assert!(controller.observe_speculative_round(2, Duration::from_millis(15)));
        }
        for _ in 0..4 {
            assert!(controller.observe_speculative_round(1, Duration::from_millis(11)));
        }
        assert!(controller.may_speculate());
    }

    #[test]
    fn keeps_proposer_when_equivalent_output_cost_is_positive() {
        let mut controller = SpeculationCostController::new();
        controller.observe_ordinary_target(Duration::from_millis(10));
        for _ in 0..4 {
            assert!(controller.observe_speculative_round(2, Duration::from_millis(15)));
        }
        assert!(controller.may_speculate());
    }

    #[test]
    fn reports_the_same_equivalent_baseline_used_by_the_gate() {
        let mut controller = SpeculationCostController::new();
        assert_eq!(controller.equivalent_ordinary_elapsed(3), None);
        controller.observe_ordinary_target(Duration::from_millis(10));
        controller.observe_ordinary_target(Duration::from_millis(14));
        assert_eq!(
            controller.equivalent_ordinary_elapsed(3),
            Some(Duration::from_millis(36))
        );
    }

    /// Modelled gate: a window is unprofitable exactly when its mean
    /// decisions per round are at most the round cost, whatever the times.
    #[test]
    fn modelled_gate_is_an_acceptance_threshold_independent_of_wall_time() {
        // Round cost 1.26: two decisions per round is profitable.
        let mut keeps = SpeculationCostController::modelled(1.26);
        keeps.observe_ordinary_target(Duration::from_millis(500));
        for round in 0..16 {
            let elapsed = Duration::from_millis(1 + round * 97);
            assert!(keeps.observe_speculative_round(2, elapsed));
        }
        // One decision per round (all drafts rejected) is not: disabled after
        // two windows, at exactly the eighth round, however fast the rounds.
        let mut drops = SpeculationCostController::modelled(1.26);
        drops.observe_ordinary_target(Duration::from_secs(9));
        for _ in 0..7 {
            assert!(drops.observe_speculative_round(1, Duration::from_nanos(1)));
        }
        assert!(!drops.observe_speculative_round(1, Duration::from_nanos(1)));
        // Mean exactly at the cost (1.25 vs round cost 1.25) is unprofitable.
        let mut edge = SpeculationCostController::modelled(1.25);
        edge.observe_ordinary_target(Duration::ZERO);
        assert!(
            edge.may_speculate(),
            "modelled warm-up counts a zero-time decision"
        );
        for _ in 0..2 {
            for decisions in [1, 1, 1, 2] {
                edge.observe_speculative_round(decisions, Duration::ZERO);
            }
        }
        assert!(!edge.may_speculate());
    }

    #[test]
    fn modelled_decisions_repeat_for_an_identical_transcript_under_any_timing() {
        let transcript = [4, 1, 1, 2, 1, 1, 1, 1, 3, 1, 1, 1];
        let run = |jitter: u64| {
            let mut controller = SpeculationCostController::modelled(1.49);
            controller.observe_ordinary_target(Duration::from_millis(30 + jitter));
            transcript
                .iter()
                .enumerate()
                .map(|(i, &d)| {
                    controller
                        .observe_speculative_round(d, Duration::from_millis(jitter * i as u64))
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(0), run(17));
        assert_eq!(run(0), run(1_000));
    }

    #[test]
    fn modelled_gate_keeps_measured_telemetry() {
        let mut controller = SpeculationCostController::modelled(1.26);
        controller.observe_ordinary_target(Duration::from_millis(20));
        controller.observe_ordinary_target(Duration::from_millis(40));
        assert_eq!(
            controller.equivalent_ordinary_elapsed(2),
            Some(Duration::from_millis(60))
        );
    }
}
