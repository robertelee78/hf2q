//! Pure measured cost gate for one proposer in a speculative generation.
//!
//! Acceptance is deliberately not a profitability proxy. The gate learns a
//! rolling ordinary target-token cost first, then compares four equivalent
//! output rounds against the complete proposal + verify/recovery elapsed time.

use std::time::Duration;

const EVALUATION_ROUNDS: u8 = 4;
const UNPROFITABLE_WINDOWS_TO_DISABLE: u8 = 2;

#[derive(Debug, Clone, Default)]
pub struct SpeculationCostController {
    ordinary_nanos: u128,
    ordinary_samples: u32,
    speculative_nanos: u128,
    equivalent_ordinary_nanos: u128,
    rounds: u8,
    consecutive_unprofitable_windows: u8,
    speculation_enabled: bool,
}

impl SpeculationCostController {
    pub fn new() -> Self {
        Self {
            speculation_enabled: true,
            ..Self::default()
        }
    }

    pub fn observe_ordinary_target(&mut self, elapsed: Duration) {
        if !elapsed.is_zero() {
            self.ordinary_nanos = self.ordinary_nanos.saturating_add(elapsed.as_nanos());
            self.ordinary_samples = self.ordinary_samples.saturating_add(1);
        }
    }

    /// A proposer is never warmed up on an unmeasured target baseline.
    pub fn may_speculate(&self) -> bool {
        self.speculation_enabled && self.ordinary_samples > 0
    }

    /// Measured ordinary target time for the same number of output decisions.
    /// This exposes the cost gate's comparison baseline for telemetry without
    /// making acceptance rate a profitability proxy.
    pub fn equivalent_ordinary_elapsed(
        &self,
        equivalent_target_decisions: usize,
    ) -> Option<Duration> {
        if self.ordinary_samples == 0 || equivalent_target_decisions == 0 {
            return None;
        }
        let per_decision = self.ordinary_nanos / u128::from(self.ordinary_samples);
        let nanos = per_decision.saturating_mul(equivalent_target_decisions as u128);
        Some(Duration::from_nanos(nanos.min(u128::from(u64::MAX)) as u64))
    }

    /// Record a completed target-equivalent speculative round.
    /// `equivalent_target_decisions` includes a terminal EOS decision even
    /// though EOS is not exposed as a completion token.
    /// Returns false after two consecutive full windows have no positive net
    /// gain. One negative window is treated as noise: draft acceptance and
    /// first-use pipeline cost are bursty enough that a permanent decision
    /// after four rounds can discard an otherwise profitable generation.
    pub fn observe_speculative_round(
        &mut self,
        equivalent_target_decisions: usize,
        elapsed: Duration,
    ) -> bool {
        if !self.may_speculate() || equivalent_target_decisions == 0 || elapsed.is_zero() {
            return self.speculation_enabled;
        }
        let ordinary = self.ordinary_nanos / u128::from(self.ordinary_samples);
        self.equivalent_ordinary_nanos = self
            .equivalent_ordinary_nanos
            .saturating_add(ordinary.saturating_mul(equivalent_target_decisions as u128));
        self.speculative_nanos = self.speculative_nanos.saturating_add(elapsed.as_nanos());
        self.rounds = self.rounds.saturating_add(1);
        if self.rounds >= EVALUATION_ROUNDS {
            if self.speculative_nanos >= self.equivalent_ordinary_nanos {
                self.consecutive_unprofitable_windows =
                    self.consecutive_unprofitable_windows.saturating_add(1);
                if self.consecutive_unprofitable_windows >= UNPROFITABLE_WINDOWS_TO_DISABLE {
                    self.speculation_enabled = false;
                }
            } else {
                self.consecutive_unprofitable_windows = 0;
            }
            self.rounds = 0;
            self.speculative_nanos = 0;
            self.equivalent_ordinary_nanos = 0;
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
}
