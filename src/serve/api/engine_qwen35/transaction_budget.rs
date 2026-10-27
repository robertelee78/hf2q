//! Finite Qwen worker budgets. Prefill cost grows with the retained context;
//! scalar/cohort decode and speculative verification keep their existing bound.
use std::time::Duration;

pub(super) enum Transaction {
    BoundedPrefill,
    Decode,
}

pub(super) const fn timeout(transaction: Transaction) -> Duration {
    match transaction {
        Transaction::BoundedPrefill => Duration::from_secs(120),
        Transaction::Decode => Duration::from_secs(30),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::api::engine_supervisor::EngineSupervisor;

    #[test]
    fn bounded_prefill_survives_old_deadline_and_finishes_inside_new_budget() {
        let supervisor = EngineSupervisor::without_monitor_for_test();
        let lease = supervisor
            .arm(
                "Qwen35 bounded prefill",
                timeout(Transaction::BoundedPrefill),
            )
            .unwrap();
        supervisor.elapse_armed_for_test(Duration::from_secs(31));
        assert!(supervisor.is_healthy());
        supervisor.elapse_armed_for_test(Duration::from_secs(88));
        assert!(supervisor.is_healthy());
        assert!(lease.finish().is_ok());
        assert!(supervisor.is_healthy());
    }

    #[test]
    fn decode_keeps_thirty_second_fail_closed_deadline() {
        let supervisor = EngineSupervisor::without_monitor_for_test();
        let lease = supervisor
            .arm("Qwen35 decode greedy", timeout(Transaction::Decode))
            .unwrap();
        supervisor.elapse_armed_for_test(Duration::from_secs(30));
        assert!(!supervisor.is_healthy());
        assert!(lease.finish().is_err());
        assert!(supervisor
            .arm("later decode", timeout(Transaction::Decode))
            .is_err());
        assert!(supervisor
            .unhealthy_message()
            .contains("Qwen35 decode greedy"));
    }

    #[test]
    fn bounded_prefill_still_poisoned_permanently_after_new_deadline() {
        let supervisor = EngineSupervisor::without_monitor_for_test();
        let lease = supervisor
            .arm(
                "Qwen35 bounded prefill",
                timeout(Transaction::BoundedPrefill),
            )
            .unwrap();
        supervisor.elapse_armed_for_test(Duration::from_secs(121));
        assert!(!supervisor.is_healthy());
        assert!(lease.finish().is_err());
        assert!(supervisor
            .arm("later prefill", timeout(Transaction::BoundedPrefill))
            .is_err());
        assert!(supervisor
            .unhealthy_message()
            .contains("Qwen35 bounded prefill"));
    }
}
