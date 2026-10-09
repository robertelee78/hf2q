//! Request-local Qwen reasoning ceilings for OpenAI chat serving.
//!
//! Qwen templates can seed a hidden reasoning span before generation. A
//! required or named tool grammar intentionally waits for the reasoning-close
//! marker, so constrained calls need a bounded close and useful capacity after
//! it. This module owns that composition as one testable policy unit; it is
//! the Qwen branch of the shared ADR-062 D5 budget enforcer
//! (`super::reasoning_controls::resolve_thinking_budget_policy`), which
//! every family flows through identically.

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

/// A thinking budget the request asked for by number (ADR-062 D5):
/// `strict` when the client sent the number itself (`thinking_token_budget`
/// or `reasoning.max_tokens`), so it must fit within `max_tokens` or the
/// request is a 400; `false` when the one effort table derived it from a
/// level or the server supplied a default, so the server clamps it into
/// the remaining window instead.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RequestedThinkingBudget {
    pub(super) tokens: usize,
    pub(super) strict: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct QwenThinkingResolution {
    pub(super) default_budget: Option<usize>,
    pub(super) effective_budget: Option<usize>,
    pub(super) end_tokens: Option<Arc<Vec<u32>>>,
    pub(super) close_tokens: Option<Arc<Vec<u32>>>,
    pub(super) required_tool_mode: bool,
    /// A budget that was resolved but not enforced, with its short reason
    /// (see `super::reasoning_controls`): a server-side or effort-derived
    /// budget is dropped with a warning instead of failing the request
    /// (ADR-062 D5 bullet 3).
    pub(super) dropped_budget: Option<(usize, &'static str)>,
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
        && registration.is_some_and(|registration| registration.family == "qwen35")
}

/// Resolve the complete Qwen reasoning-close policy consumed by the shared
/// ADR-062 D5 budget enforcer. Tests call this same unit with the real Qwen
/// registration and both constrained tool-choice variants; the enforcer
/// supplies the live tokenizer.
pub(super) fn resolve_qwen_thinking_policy<F>(
    registration: Option<&registry::ModelRegistration>,
    reasoning_forced_open: bool,
    tool_choice: &ToolChoiceValue,
    requested_budget: Option<RequestedThinkingBudget>,
    budget_param: &'static str,
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
    if requested_budget
        .as_ref()
        .is_some_and(|budget| budget.tokens == 0)
    {
        return Err(ThinkingPolicyError {
            message: format!("{budget_param} must be greater than zero"),
            param: budget_param,
        });
    }

    let required_tool_mode = matches!(
        tool_choice,
        ToolChoiceValue::Required | ToolChoiceValue::Function(_)
    );
    let default_budget = if requested_budget.is_none() {
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
    let Some(configured_budget) = requested_budget
        .map(|budget| budget.tokens)
        .or(default_budget)
    else {
        return Ok(QwenThinkingResolution {
            default_budget,
            required_tool_mode,
            ..QwenThinkingResolution::default()
        });
    };
    if !slot_aware {
        if requested_budget
            .as_ref()
            .is_some_and(|budget| budget.strict)
        {
            return Err(ThinkingPolicyError {
                message: format!("{budget_param} requires an inflight-batched scheduler"),
                param: budget_param,
            });
        }
        // ADR-062 D5 bullet 3: a server-default or effort-derived budget
        // under fifo-serial is dropped (warned by the caller), never a 4xx;
        // only an explicit client budget is rejected, identically on every
        // family.
        return Ok(QwenThinkingResolution {
            default_budget,
            required_tool_mode,
            dropped_budget: Some((
                configured_budget,
                super::reasoning_controls::DROPPED_FIFO_SERIAL,
            )),
            ..QwenThinkingResolution::default()
        });
    }

    let close = registration
        .and_then(|registration| registration.reasoning_close)
        .ok_or_else(|| ThinkingPolicyError {
            message: format!("{budget_param} requires a registered reasoning close marker"),
            param: budget_param,
        })?;
    let transition = format!("\nI need to answer now.{close}");
    let end_tokens = encode(&transition).map_err(|error| ThinkingPolicyError {
        message: format!("failed to tokenize reasoning boundary: {error}"),
        param: budget_param,
    })?;
    let close_tokens = encode(close).map_err(|error| ThinkingPolicyError {
        message: format!("failed to tokenize reasoning boundary: {error}"),
        param: budget_param,
    })?;
    if end_tokens.is_empty() || close_tokens.is_empty() {
        return Err(ThinkingPolicyError {
            message: "reasoning boundary tokenized to an empty sequence".into(),
            param: budget_param,
        });
    }
    if !end_tokens.ends_with(close_tokens.as_slice()) {
        return Err(ThinkingPolicyError {
            message:
                "forced reasoning transition must end with the standalone close-token sequence"
                    .into(),
            param: budget_param,
        });
    }

    let answer_reserve_tokens = if required_tool_mode {
        qwen_required_tool_answer_reserve(max_tokens)
    } else {
        1
    };
    let effective_budget = effective_qwen_thinking_budget_with_reserve(
        Some(configured_budget),
        requested_budget
            .as_ref()
            .is_some_and(|budget| budget.strict),
        max_tokens,
        end_tokens.len(),
        answer_reserve_tokens,
    )
    .map_err(|message| ThinkingPolicyError {
        message,
        param: budget_param,
    })?;
    if required_tool_mode && effective_budget.is_none() {
        if requested_budget
            .as_ref()
            .is_some_and(|budget| budget.strict)
        {
            return Err(ThinkingPolicyError {
                message: format!(
                    "max_tokens ({max_tokens}) is too small for the Qwen reasoning transition and required tool call"
                ),
                param: "max_tokens",
            });
        }
        // ADR-062 D5 bullet 3: the server-side required-tool budget is
        // dropped with a warning when max_tokens leaves no room, never a
        // 4xx (issue #278 finding 3).
        return Ok(QwenThinkingResolution {
            default_budget,
            required_tool_mode,
            dropped_budget: Some((
                configured_budget,
                super::reasoning_controls::DROPPED_NO_ROOM,
            )),
            ..QwenThinkingResolution::default()
        });
    }

    Ok(QwenThinkingResolution {
        default_budget,
        effective_budget,
        end_tokens: effective_budget.map(|_| end_tokens),
        close_tokens: effective_budget.map(|_| close_tokens),
        required_tool_mode,
        dropped_budget: None,
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
                    "thinking_token_budget",
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
    fn preserves_ordinary_opt_out_and_drops_tiny_required_window() {
        let registration = registry::find_for("Qwen3.8").expect("Qwen registration");
        let ordinary = resolve_qwen_thinking_policy(
            Some(&registration),
            true,
            &ToolChoiceValue::Auto,
            None,
            "thinking_token_budget",
            128,
            QwenToolChainState::default(),
            QwenThinkingDefaults::from_config(Some(0), Some(0)),
            true,
            qwen_test_encode,
        )
        .unwrap();
        assert_eq!(
            ordinary,
            QwenThinkingResolution {
                required_tool_mode: false,
                ..QwenThinkingResolution::default()
            }
        );

        // ADR-062 D5 bullet 3 (issue #278 finding 3): the server-side
        // required-tool budget is dropped with a warning when max_tokens
        // leaves no room for the transition and the answer reserve, never a
        // 4xx.
        let tiny = resolve_qwen_thinking_policy(
            Some(&registration),
            true,
            &ToolChoiceValue::Required,
            None,
            "thinking_token_budget",
            16,
            QwenToolChainState::default(),
            QwenThinkingDefaults::default(),
            true,
            qwen_test_encode,
        )
        .unwrap();
        assert!(tiny.required_tool_mode);
        assert_eq!(tiny.effective_budget, None);
        assert_eq!(
            tiny.dropped_budget,
            Some((16, super::super::reasoning_controls::DROPPED_NO_ROOM))
        );
    }

    #[test]
    fn pins_explicit_max_safe_boundary() {
        let registration = registry::find_for("Qwen3.8").expect("Qwen registration");
        let at_limit = resolve_qwen_thinking_policy(
            Some(&registration),
            true,
            &ToolChoiceValue::Required,
            Some(RequestedThinkingBudget {
                tokens: 56,
                strict: true,
            }),
            "thinking_token_budget",
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
            Some(RequestedThinkingBudget {
                tokens: 57,
                strict: true,
            }),
            "thinking_token_budget",
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
            "thinking_token_budget",
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
