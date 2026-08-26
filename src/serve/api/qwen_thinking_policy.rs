//! Request-local Qwen reasoning ceilings for OpenAI chat serving.
//!
//! Qwen templates can seed a hidden reasoning span before generation. A
//! required or named tool grammar intentionally waits for the reasoning-close
//! marker, so constrained calls need a bounded close and useful capacity after
//! it. This module owns that composition as one testable policy unit.

use std::sync::Arc;

use super::engine;
use super::registry;
use super::schema::{ChatMessage, ToolChoiceValue};

const QWEN_TOOL_CONTINUATION_THINKING_CEILING: usize = 512;
const QWEN_REPEATED_CAP_THINKING_FLOOR: usize = 256;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct QwenToolChainState {
    pub(super) is_tool_continuation: bool,
    /// Assistant turns that requested tools since the latest user turn.
    /// Parallel results belong to one cycle and do not deepen the chain.
    pub(super) tool_cycles_since_user: usize,
}

pub(super) fn qwen_tool_chain_state(messages: &[ChatMessage]) -> QwenToolChainState {
    let is_tool_continuation = messages
        .iter()
        .rev()
        .find(|message| message.role != "system")
        .is_some_and(|message| message.role == "tool");
    let chain_start = messages
        .iter()
        .rposition(|message| message.role == "user")
        .map_or(0, |index| index.saturating_add(1));
    let tool_cycles_since_user = messages[chain_start..]
        .iter()
        .filter(|message| {
            message.role == "assistant"
                && message
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty())
        })
        .count();
    QwenToolChainState {
        is_tool_continuation,
        tool_cycles_since_user,
    }
}

pub(super) fn adaptive_qwen_default_thinking_budget(
    base: Option<usize>,
    continuation_override: Option<Option<usize>>,
    chain: QwenToolChainState,
) -> Option<usize> {
    if !chain.is_tool_continuation {
        return base;
    }
    let configured = continuation_override.unwrap_or_else(|| {
        base.map(|budget| budget.min(QWEN_TOOL_CONTINUATION_THINKING_CEILING))
    })?;
    let reductions = chain
        .tool_cycles_since_user
        .saturating_sub(1)
        .min(usize::BITS as usize - 1);
    let reduced = configured >> reductions;
    Some(reduced.max(configured.min(QWEN_REPEATED_CAP_THINKING_FLOOR)))
}

fn qwen_default_thinking_budget_for_mode(
    base: Option<usize>,
    continuation_override: Option<Option<usize>>,
    chain: QwenToolChainState,
    constrained_tool_choice: bool,
    max_tokens: usize,
) -> Option<usize> {
    let adaptive = adaptive_qwen_default_thinking_budget(base, continuation_override, chain);
    if constrained_tool_choice {
        adaptive.or(Some(max_tokens))
    } else {
        adaptive
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct QwenThinkingDefaults {
    base: Option<usize>,
    continuation_override: Option<Option<usize>>,
}

impl QwenThinkingDefaults {
    pub(super) const fn from_config(base: Option<u32>, continuation: Option<u32>) -> Self {
        Self {
            base: match base {
                Some(value) if value > 0 => Some(value as usize),
                _ => None,
            },
            continuation_override: match continuation {
                Some(value) if value > 0 => Some(Some(value as usize)),
                Some(_) => Some(None),
                None => None,
            },
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct QwenThinkingResolution {
    pub(super) default_budget: Option<usize>,
    pub(super) effective_budget: Option<usize>,
    pub(super) end_tokens: Option<Arc<Vec<u32>>>,
    pub(super) close_tokens: Option<Arc<Vec<u32>>>,
    pub(super) required_tool_mode: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ThinkingPolicyError {
    pub(super) message: String,
    pub(super) param: &'static str,
}

pub(super) fn qwen_thinking_mode(
    registration: Option<&registry::ModelRegistration>,
    reasoning_forced_open: bool,
) -> bool {
    reasoning_forced_open
        && registration.is_some_and(registry::ModelRegistration::uses_qwen_protocol)
}

/// Resolve the complete Qwen reasoning-close policy consumed by the handler.
/// Tests call this same unit with the real Qwen registration and both
/// constrained tool-choice variants; the handler supplies the live tokenizer.
pub(super) fn resolve_qwen_thinking_policy<F>(
    registration: Option<&registry::ModelRegistration>,
    reasoning_forced_open: bool,
    tool_choice: &ToolChoiceValue,
    explicit_budget: Option<usize>,
    max_tokens: usize,
    chain: QwenToolChainState,
    defaults: QwenThinkingDefaults,
    slot_aware: bool,
    mut encode: F,
) -> Result<QwenThinkingResolution, ThinkingPolicyError>
where
    F: FnMut(&str) -> Result<Arc<Vec<u32>>, String>,
{
    if !qwen_thinking_mode(registration, reasoning_forced_open) {
        return Ok(QwenThinkingResolution::default());
    }
    if explicit_budget == Some(0) {
        return Err(ThinkingPolicyError {
            message: "thinking_token_budget must be greater than zero".into(),
            param: "thinking_token_budget",
        });
    }

    let required_tool_mode = matches!(
        tool_choice,
        ToolChoiceValue::Required | ToolChoiceValue::Function(_)
    );
    let default_budget = if explicit_budget.is_none() {
        qwen_default_thinking_budget_for_mode(
            defaults.base,
            defaults.continuation_override,
            chain,
            required_tool_mode,
            max_tokens,
        )
    } else {
        None
    };
    let Some(configured_budget) = explicit_budget.or(default_budget) else {
        return Ok(QwenThinkingResolution {
            default_budget,
            required_tool_mode,
            ..QwenThinkingResolution::default()
        });
    };
    if !slot_aware {
        return Err(ThinkingPolicyError {
            message: "thinking_token_budget requires an inflight-batched scheduler".into(),
            param: "thinking_token_budget",
        });
    }

    let close = registration
        .and_then(|registration| registration.reasoning_close)
        .ok_or_else(|| ThinkingPolicyError {
            message: "thinking_token_budget requires a registered reasoning close marker".into(),
            param: "thinking_token_budget",
        })?;
    let transition = format!("\nI need to answer now.{close}");
    let end_tokens = encode(&transition).map_err(|error| ThinkingPolicyError {
        message: format!("failed to tokenize reasoning boundary: {error}"),
        param: "thinking_token_budget",
    })?;
    let close_tokens = encode(close).map_err(|error| ThinkingPolicyError {
        message: format!("failed to tokenize reasoning boundary: {error}"),
        param: "thinking_token_budget",
    })?;
    if end_tokens.is_empty() || close_tokens.is_empty() {
        return Err(ThinkingPolicyError {
            message: "reasoning boundary tokenized to an empty sequence".into(),
            param: "thinking_token_budget",
        });
    }
    if !end_tokens.ends_with(close_tokens.as_slice()) {
        return Err(ThinkingPolicyError {
            message:
                "forced reasoning transition must end with the standalone close-token sequence"
                    .into(),
            param: "thinking_token_budget",
        });
    }

    let answer_reserve_tokens = if required_tool_mode {
        qwen_required_tool_answer_reserve(max_tokens)
    } else {
        1
    };
    let effective_budget = effective_qwen_thinking_budget_with_reserve(
        Some(configured_budget),
        explicit_budget.is_some(),
        max_tokens,
        end_tokens.len(),
        answer_reserve_tokens,
    )
    .map_err(|message| ThinkingPolicyError {
        message,
        param: "thinking_token_budget",
    })?;
    if required_tool_mode && effective_budget.is_none() {
        return Err(ThinkingPolicyError {
            message: format!(
                "max_tokens ({max_tokens}) is too small for the Qwen reasoning transition and required tool call"
            ),
            param: "max_tokens",
        });
    }

    Ok(QwenThinkingResolution {
        default_budget,
        effective_budget,
        end_tokens: effective_budget.map(|_| end_tokens),
        close_tokens: effective_budget.map(|_| close_tokens),
        required_tool_mode,
    })
}

pub(super) fn effective_qwen_thinking_budget(
    configured: Option<usize>,
    explicit: bool,
    max_tokens: usize,
    transition_tokens: usize,
) -> Result<Option<usize>, String> {
    effective_qwen_thinking_budget_with_reserve(
        configured,
        explicit,
        max_tokens,
        transition_tokens,
        1,
    )
}

fn effective_qwen_thinking_budget_with_reserve(
    configured: Option<usize>,
    explicit: bool,
    max_tokens: usize,
    transition_tokens: usize,
    answer_reserve_tokens: usize,
) -> Result<Option<usize>, String> {
    let Some(budget) = configured else {
        return Ok(None);
    };
    let maximum_safe_budget =
        max_tokens.saturating_sub(transition_tokens.saturating_add(answer_reserve_tokens));
    if explicit && budget > maximum_safe_budget {
        return Err(if answer_reserve_tokens == 1 {
            format!(
                "thinking_token_budget ({budget}) plus its {transition_tokens}-token forced transition must be less than max_tokens ({max_tokens}) so answer capacity remains"
            )
        } else {
            format!(
                "thinking_token_budget ({budget}) plus its {transition_tokens}-token forced transition must leave {answer_reserve_tokens} answer tokens within max_tokens ({max_tokens})"
            )
        });
    }
    let effective = if explicit {
        budget
    } else {
        budget
            .min(max_tokens.saturating_mul(3) / 4)
            .min(maximum_safe_budget)
    };
    Ok((effective > 0).then_some(effective))
}

pub(super) fn qwen_required_tool_answer_reserve(max_tokens: usize) -> usize {
    // Half leaves useful capacity for wrappers, names, and arguments while
    // preserving bounded native reasoning on short OpenAI requests.
    (max_tokens / 2).max(1)
}

pub(super) fn constrained_thinking_budget_conflicts(
    thinking_token_budget: Option<usize>,
    tool_call_policy: engine::ToolCallPolicy,
    qwen_required_tool_mode: bool,
    deepseek_required_tool_mode: bool,
) -> bool {
    thinking_token_budget.is_some()
        && tool_call_policy == engine::ToolCallPolicy::Constrained
        && !qwen_required_tool_mode
        && !deepseek_required_tool_mode
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen_test_encode(text: &str) -> Result<Arc<Vec<u32>>, String> {
        if text == "</think>" {
            Ok(Arc::new(vec![7, 8]))
        } else {
            // Model the tokenizer invariant consumed by the runtime: the
            // forced transition ends in the standalone close-token sequence.
            Ok(Arc::new(vec![1, 2, 3, 4, 5, 6, 7, 8]))
        }
    }

    #[test]
    fn reserves_required_and_named_call_capacity_with_missing_or_zero_defaults() {
        let registration = registry::find_for("Qwen3.8").expect("Qwen registration");
        assert_eq!(registration.reasoning_close, Some("</think>"));
        for defaults in [
            QwenThinkingDefaults::from_config(None, None),
            QwenThinkingDefaults::from_config(Some(0), Some(0)),
        ] {
            for tool_choice in [
                ToolChoiceValue::Required,
                ToolChoiceValue::Function("lifecycle_probe".into()),
            ] {
                let resolved = resolve_qwen_thinking_policy(
                    Some(&registration),
                    true,
                    &tool_choice,
                    None,
                    128,
                    QwenToolChainState::default(),
                    defaults,
                    true,
                    qwen_test_encode,
                )
                .unwrap();
                assert!(resolved.required_tool_mode);
                assert_eq!(resolved.default_budget, Some(128));
                assert_eq!(resolved.effective_budget, Some(56));
                assert_eq!(resolved.end_tokens.as_deref().unwrap().len(), 8);
                assert_eq!(resolved.close_tokens.as_deref().unwrap(), &[7, 8]);
                assert!(resolved
                    .end_tokens
                    .as_deref()
                    .unwrap()
                    .ends_with(resolved.close_tokens.as_deref().unwrap()));
                assert_eq!(
                    resolved.effective_budget.unwrap()
                        + resolved.end_tokens.as_deref().unwrap().len()
                        + qwen_required_tool_answer_reserve(128),
                    128
                );
                assert!(!constrained_thinking_budget_conflicts(
                    resolved.effective_budget,
                    engine::ToolCallPolicy::Constrained,
                    resolved.required_tool_mode,
                    false,
                ));
            }
        }
    }

    #[test]
    fn preserves_ordinary_opt_out_and_rejects_tiny_required_window() {
        let registration = registry::find_for("Qwen3.8").expect("Qwen registration");
        let ordinary = resolve_qwen_thinking_policy(
            Some(&registration),
            true,
            &ToolChoiceValue::Auto,
            None,
            128,
            QwenToolChainState::default(),
            QwenThinkingDefaults::from_config(Some(0), Some(0)),
            true,
            qwen_test_encode,
        )
        .unwrap();
        assert_eq!(ordinary, QwenThinkingResolution::default());

        let error = resolve_qwen_thinking_policy(
            Some(&registration),
            true,
            &ToolChoiceValue::Required,
            None,
            16,
            QwenToolChainState::default(),
            QwenThinkingDefaults::default(),
            true,
            qwen_test_encode,
        )
        .unwrap_err();
        assert_eq!(error.param, "max_tokens");
        assert!(error.message.contains("too small"));
    }

    #[test]
    fn accepts_explicit_lifecycle_budget_across_qwen_protocol_spellings() {
        for model in ["Qwen3.5", "Qwen3.6", "Qwen3.7", "Qwen3.8"] {
            let registration = registry::find_for(model).expect("Qwen registration");
            let resolved = resolve_qwen_thinking_policy(
                Some(&registration),
                true,
                &ToolChoiceValue::Auto,
                Some(16),
                64,
                QwenToolChainState::default(),
                QwenThinkingDefaults::from_config(Some(0), Some(0)),
                true,
                qwen_test_encode,
            )
            .unwrap();
            assert!(!resolved.required_tool_mode, "{model}");
            assert_eq!(resolved.default_budget, None, "{model}");
            assert_eq!(resolved.effective_budget, Some(16), "{model}");
            assert_eq!(resolved.end_tokens.as_deref().unwrap().len(), 8, "{model}");
        }
    }

    #[test]
    fn pins_explicit_max_safe_boundary() {
        let registration = registry::find_for("Qwen3.8").expect("Qwen registration");
        let at_limit = resolve_qwen_thinking_policy(
            Some(&registration),
            true,
            &ToolChoiceValue::Required,
            Some(56),
            128,
            QwenToolChainState::default(),
            QwenThinkingDefaults::default(),
            true,
            qwen_test_encode,
        )
        .unwrap();
        assert_eq!(at_limit.effective_budget, Some(56));

        let over = resolve_qwen_thinking_policy(
            Some(&registration),
            true,
            &ToolChoiceValue::Required,
            Some(57),
            128,
            QwenToolChainState::default(),
            QwenThinkingDefaults::default(),
            true,
            qwen_test_encode,
        )
        .unwrap_err();
        assert_eq!(over.param, "thinking_token_budget");
        assert!(over.message.contains("leave 64 answer tokens"));
    }

    #[test]
    fn rejects_a_transition_without_the_standalone_close_suffix() {
        let registration = registry::find_for("Qwen3.8").expect("Qwen registration");
        let error = resolve_qwen_thinking_policy(
            Some(&registration),
            true,
            &ToolChoiceValue::Required,
            None,
            128,
            QwenToolChainState::default(),
            QwenThinkingDefaults::default(),
            true,
            |text| {
                Ok(if text == "</think>" {
                    Arc::new(vec![7, 8])
                } else {
                    Arc::new(vec![1, 2, 3, 4])
                })
            },
        )
        .unwrap_err();
        assert_eq!(error.param, "thinking_token_budget");
        assert!(error.message.contains("standalone close-token sequence"));
    }
}
