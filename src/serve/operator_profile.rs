//! Setup-persisted serving profile applied as process defaults.
//!
//! `hf2q setup` persists the qualified agentic-coding profile (repetition
//! penalty plus bounded thinking budgets) when the operator optimizes for
//! long agent and tool-use prompts. `hf2q serve` applies those values to
//! the process environment here — once, at the top of `cmd_serve`, before
//! engine construction and before the first request handler reads them —
//! so every downstream consumer (the `INVESTIGATION_ENV` snapshot, the
//! per-request thinking-budget reads in `api::handlers`) observes one
//! consistent set of values.
//!
//! Precedence is always: explicit operator environment > setup config >
//! built-in default. An operator-exported variable is never overwritten;
//! the config only fills variables that are absent.

use crate::setup::ServeDefaultsV2;

const REPETITION_PENALTY_ENV: &str = "HF2Q_DEFAULT_REPETITION_PENALTY";
const THINKING_BUDGET_ENV: &str = "HF2Q_DEFAULT_THINKING_TOKEN_BUDGET";
const TOOL_THINKING_BUDGET_ENV: &str = "HF2Q_DEFAULT_TOOL_THINKING_TOKEN_BUDGET";

pub(crate) fn apply_operator_serve_profile(defaults: Option<&ServeDefaultsV2>) {
    let Some(defaults) = defaults else {
        return;
    };
    for (key, configured) in [
        (
            REPETITION_PENALTY_ENV,
            defaults.repetition_penalty.map(|value| value.to_string()),
        ),
        (
            THINKING_BUDGET_ENV,
            defaults.thinking_token_budget.map(|value| value.to_string()),
        ),
        (
            TOOL_THINKING_BUDGET_ENV,
            defaults
                .tool_thinking_token_budget
                .map(|value| value.to_string()),
        ),
    ] {
        if let Some(value) = planned_assignment(std::env::var_os(key).is_some(), configured) {
            std::env::set_var(key, &value);
            tracing::info!(key, value, "applied setup-persisted serve profile default");
        }
    }
}

/// Pure decision: the config value becomes the process default only when
/// the operator has not exported the variable. Kept env-free so the
/// precedence rule is unit-testable without mutating process state.
fn planned_assignment(env_present: bool, configured: Option<String>) -> Option<String> {
    if env_present {
        None
    } else {
        configured
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_environment_always_wins_over_the_persisted_profile() {
        assert_eq!(planned_assignment(true, Some("1.05".to_owned())), None);
        assert_eq!(planned_assignment(true, None), None);
    }

    #[test]
    fn persisted_profile_fills_only_absent_environment() {
        assert_eq!(
            planned_assignment(false, Some("2048".to_owned())),
            Some("2048".to_owned())
        );
        assert_eq!(planned_assignment(false, None), None);
    }
}
