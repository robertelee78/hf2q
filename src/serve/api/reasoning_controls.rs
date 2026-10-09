//! ADR-062 D5 — reasoning controls that behave the same on every family.
//!
//! One effort table accepts `reasoning_effort`, its nested
//! `reasoning.{effort,enabled,max_tokens}` aliases, and
//! `thinking_token_budget` on every family, resolves them into the loaded
//! family's native thinking channel (Qwen thinking budget, DeepSeek-V4
//! effort tier), and enforces thinking budgets through one shared policy on
//! `inflight-batched` — including Gemma 4, whose template has no thinking
//! channel and therefore rejects every reasoning control with one 400
//! instead of accepting-and-ignoring it. A server-side default budget never
//! causes a 4xx: under `fifo-serial` only an explicit client budget is
//! rejected, identically on every family.
//!
//! The Qwen-side adaptive machinery (tool-continuation ceilings, chain
//! reductions, the forced transition invariant) stays in
//! [`super::qwen_thinking_policy`]; this module owns the table, the request
//! parse, and the shared enforcer every family flows through.

use std::sync::Arc;

use super::qwen_thinking_policy::{
    effective_qwen_thinking_budget, resolve_qwen_thinking_policy, QwenThinkingDefaults,
    QwenToolChainState, RequestedThinkingBudget, ThinkingPolicyError,
};
use super::registry;
use super::schema::{ChatCompletionRequest, ToolChoiceValue};

/// Short reasons attached to a resolved-but-unenforced budget (ADR-062 D5:
/// a dropped server-side budget is warned, never a client-visible 4xx).
pub(super) const DROPPED_FIFO_SERIAL: &str = "fifo-serial scheduler";
pub(super) const DROPPED_NO_ROOM: &str = "no room within max_tokens";
const DROPPED_NO_CHANNEL: &str = "no thinking channel on this family";
const DROPPED_NO_OPEN_SPAN: &str = "no open reasoning span";

/// The one effort table (ADR-062 D5). `none`/`minimal` turn thinking off;
/// the ladder levels carry thinking-budget ceilings for budget-channel
/// families and map onto DeepSeek-V4's native tiers with `medium` -> `high`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum EffortLevel {
    Off,
    Low,
    Medium,
    High,
    Max,
}

impl EffortLevel {
    /// Parse one effort value, case-insensitively. `none`/`minimal` are the
    /// stock-client "off" sentinels (OpenCode sends `none` when no reasoning
    /// variant is selected); `xhigh` and `max` both mean "no ceiling".
    pub(super) fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "none" | "minimal" => Some(Self::Off),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "xhigh" | "max" => Some(Self::Max),
            _ => None,
        }
    }

    /// The thinking-budget ceiling this level carries on a budget-channel
    /// family. `Off` disables thinking (no span to bound) and `Max`/`xhigh`
    /// sets no ceiling.
    pub(super) const fn budget(self) -> Option<usize> {
        match self {
            Self::Off | Self::Max => None,
            Self::Low => Some(512),
            Self::Medium => Some(2_048),
            Self::High => Some(8_192),
        }
    }

    /// The native DeepSeek-V4 tier this level maps onto. `Off` disables
    /// thinking instead of naming a tier; `medium` maps to the native
    /// `high` tier because DeepSeek-V4 has no native medium.
    pub(super) const fn deepseek_tier(self) -> Option<&'static str> {
        match self {
            Self::Off => None,
            Self::Low => Some("low"),
            Self::Medium | Self::High => Some("high"),
            Self::Max => Some("max"),
        }
    }
}

/// The request's reasoning controls after one parse pass: the top-level
/// `reasoning_effort`, `hf2q_enable_thinking`, and `thinking_token_budget`
/// fields merged with their `reasoning.{effort,enabled,max_tokens}` aliases,
/// with contradictory combinations rejected (ADR-062 D5 consequences).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ReasoningControls {
    pub(super) effort: Option<EffortLevel>,
    /// The request field the effort level came from.
    effort_param: Option<&'static str>,
    /// The merged thinking on/off request: `hf2q_enable_thinking` and
    /// `reasoning.enabled` agree or the parse is a 400.
    pub(super) enabled: Option<bool>,
    /// The merged explicit client budget: `thinking_token_budget` and
    /// `reasoning.max_tokens` agree or the parse is a 400.
    pub(super) budget: Option<usize>,
    /// The request field the budget came from, so rejections name what the
    /// client actually sent (issue #278 finding 3: never a field the
    /// client never sent).
    pub(super) budget_param: &'static str,
    /// `true` when `reasoning.enabled` supplied the enabled override. The
    /// hf2q-private `hf2q_enable_thinking` predates D5 (ADR-005 iter-133
    /// W67) and is deliberately not counted as a D5 control.
    enabled_is_d5_alias: bool,
}

impl Default for ReasoningControls {
    fn default() -> Self {
        Self {
            effort: None,
            effort_param: None,
            enabled: None,
            budget: None,
            budget_param: "thinking_token_budget",
            enabled_is_d5_alias: false,
        }
    }
}

impl ReasoningControls {
    /// Whether the client sent any D5-named reasoning control
    /// (`reasoning_effort`, `reasoning.{effort,enabled,max_tokens}`, or
    /// `thinking_token_budget`). A family with no thinking channel rejects
    /// exactly these with one 400 (ADR-062 D5 bullet 4).
    pub(super) fn d5_control_present(&self) -> bool {
        self.effort.is_some() || self.budget.is_some() || self.enabled_is_d5_alias
    }

    /// The first D5-named control the client sent, for the no-channel 400's
    /// message and `param`.
    pub(super) fn first_d5_param(&self) -> &'static str {
        self.effort_param
            .or(if self.enabled_is_d5_alias {
                Some("reasoning.enabled")
            } else {
                None
            })
            .or_else(|| self.budget.map(|_| self.budget_param))
            .unwrap_or("reasoning_effort")
    }

    /// The pre-render thinking override: an effort level states it directly
    /// (`Off` disables, a ladder level enables); otherwise the explicit
    /// enabled value; `None` defers to the loaded template's native mode.
    pub(super) fn thinking_enabled_override(&self) -> Option<bool> {
        match self.effort {
            Some(EffortLevel::Off) => Some(false),
            Some(_) => Some(true),
            None => self.enabled,
        }
    }

    /// Whether the client's controls, taken together, ask for thinking OFF
    /// only (`none`/`minimal` with no enabled/budget contradiction) — an
    /// accepted no-op on a family with no thinking channel (ADR-062 D5
    /// bullet 4; stock OpenCode sends `reasoning_effort: "none"` on every
    /// request).
    pub(super) fn requested_thinking_off(&self) -> bool {
        self.effort == Some(EffortLevel::Off)
            && self.enabled != Some(true)
            && self.budget.is_none()
    }

    /// The DeepSeek-V4 native tier this request maps onto through the one
    /// effort table (`medium` -> `high`); `None` when no effort level was
    /// sent or the level turns thinking off.
    pub(super) fn deepseek_reasoning_effort_tier(&self) -> Option<&'static str> {
        self.effort.and_then(EffortLevel::deepseek_tier)
    }
}

/// Parse the request's reasoning controls through the one effort table.
/// Every disagreement between the top-level fields and their nested aliases
/// is a 400 naming the fields that disagree; a budget combined with a
/// thinking-off control is a 400 (contradictory reasoning controls,
/// ADR-062 D5 consequences — matching OpenAI behavior).
pub(super) fn reasoning_controls_from_request(
    req: &ChatCompletionRequest,
) -> Result<ReasoningControls, ThinkingPolicyError> {
    let mut controls = ReasoningControls {
        enabled: req.hf2q_enable_thinking,
        budget: req.thinking_token_budget,
        ..ReasoningControls::default()
    };
    if let Some(raw) = req.reasoning_effort.as_deref() {
        controls.effort = Some(parse_effort(raw, "reasoning_effort")?);
        controls.effort_param = Some("reasoning_effort");
    }
    if let Some(raw) = req
        .reasoning
        .as_ref()
        .and_then(|aliases| aliases.effort.as_deref())
    {
        let level = parse_effort(raw, "reasoning.effort")?;
        match controls.effort {
            Some(existing) if existing != level => {
                return Err(ThinkingPolicyError {
                    message: "reasoning_effort and reasoning.effort disagree".into(),
                    param: "reasoning.effort",
                });
            }
            None => {
                controls.effort = Some(level);
                controls.effort_param = Some("reasoning.effort");
            }
            _ => {}
        }
    }
    if let Some(value) = req.reasoning.as_ref().and_then(|aliases| aliases.enabled) {
        match controls.enabled {
            Some(existing) if existing != value => {
                return Err(ThinkingPolicyError {
                    message: "hf2q_enable_thinking and reasoning.enabled disagree".into(),
                    param: "reasoning.enabled",
                });
            }
            None => {
                controls.enabled = Some(value);
                controls.enabled_is_d5_alias = true;
            }
            _ => {}
        }
    }
    if let Some(value) = req
        .reasoning
        .as_ref()
        .and_then(|aliases| aliases.max_tokens)
    {
        match controls.budget {
            Some(existing) if existing != value => {
                return Err(ThinkingPolicyError {
                    message: "thinking_token_budget and reasoning.max_tokens disagree".into(),
                    param: "reasoning.max_tokens",
                });
            }
            None => {
                controls.budget = Some(value);
                controls.budget_param = "reasoning.max_tokens";
            }
            _ => {}
        }
    }
    if controls.enabled == Some(false) {
        if let Some(param) = controls
            .effort_param
            .filter(|_| controls.effort != Some(EffortLevel::Off))
        {
            return Err(ThinkingPolicyError {
                message: format!("{param} requires thinking, but thinking was explicitly disabled"),
                param,
            });
        }
        if controls.budget.is_some() {
            return Err(ThinkingPolicyError {
                message: format!(
                    "{} requires thinking, but thinking was explicitly disabled",
                    controls.budget_param
                ),
                param: controls.budget_param,
            });
        }
    }
    if controls.effort == Some(EffortLevel::Off) && controls.enabled == Some(true) {
        return Err(ThinkingPolicyError {
            message:
                "reasoning effort `none`/`minimal` turns thinking off; reasoning.enabled=true contradicts it"
                    .into(),
            param: "reasoning.enabled",
        });
    }
    if controls.effort == Some(EffortLevel::Off) && controls.budget.is_some() {
        return Err(ThinkingPolicyError {
            message: format!(
                "{} cannot be combined with reasoning effort `none`/`minimal`, which turns thinking off",
                controls.budget_param
            ),
            param: controls.budget_param,
        });
    }
    Ok(controls)
}

fn parse_effort(raw: &str, param: &'static str) -> Result<EffortLevel, ThinkingPolicyError> {
    EffortLevel::parse(raw).ok_or_else(|| ThinkingPolicyError {
        message: format!(
            "{param} must be one of none, minimal, low, medium, high, xhigh, or max; got {raw:?}"
        ),
        param,
    })
}

/// The loaded family's native thinking channel for reasoning controls
/// (ADR-062 D5): Qwen bounds its seeded reasoning span with a token budget,
/// DeepSeek-V4 maps effort levels onto its native tiers, and every other
/// family (Gemma 4, Qwen3-VL, unregistered models) has no channel and
/// rejects reasoning controls with one 400 instead of
/// accepting-and-ignoring them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ReasoningChannel {
    QwenBudget,
    DeepSeekTier,
    None,
}

/// Resolve the reasoning channel for a loaded engine: the family
/// registration is authoritative, and a Qwen template that does not
/// actually support `enable_thinking` has no thinking channel however the
/// family is labeled.
pub(super) fn reasoning_channel_for(
    registration: Option<&registry::ModelRegistration>,
    template_supports_thinking: bool,
) -> ReasoningChannel {
    match registration.map(|registration| registration.family) {
        Some("qwen35") if template_supports_thinking => ReasoningChannel::QwenBudget,
        Some("deepseek4") => ReasoningChannel::DeepSeekTier,
        _ => ReasoningChannel::None,
    }
}

/// The enforced thinking budget and its layout tokens, resolved through one
/// shared policy on every family (ADR-062 D5 bullet 2).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct ThinkingBudgetResolution {
    pub(super) thinking_budget: Option<usize>,
    pub(super) reasoning_end_tokens: Option<Arc<Vec<u32>>>,
    pub(super) reasoning_close_tokens: Option<Arc<Vec<u32>>>,
    pub(super) qwen_required_tool_mode: bool,
    pub(super) deepseek_required_tool_mode: bool,
    /// A budget that was resolved but not enforced, with its short reason.
    /// Warned by the handler; a server-side budget never becomes a 4xx.
    pub(super) dropped_budget: Option<(usize, &'static str)>,
}

/// The one shared budget enforcer (ADR-062 D5 bullet 2). Every family
/// resolves its thinking budget through this function on
/// `inflight-batched`, including Gemma 4; the tool-thinking budget has one
/// meaning on every family (a constrained tool call bounds thinking with
/// the tool budget unless the client supplied its own number). Under
/// `fifo-serial` only an explicit client budget is rejected, identically
/// on every family; a server-side default budget is dropped with a warning
/// instead.
#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_thinking_budget_policy<F>(
    registration: Option<&registry::ModelRegistration>,
    channel: ReasoningChannel,
    controls: &ReasoningControls,
    reasoning_forced_open: bool,
    tool_choice: &ToolChoiceValue,
    max_tokens: usize,
    chain: QwenToolChainState,
    server_base_budget: Option<u32>,
    server_tool_budget: Option<u32>,
    deepseek_required_tool_eligible: bool,
    slot_aware: bool,
    encode: F,
) -> Result<ThinkingBudgetResolution, ThinkingPolicyError>
where
    F: FnMut(&str) -> Result<Arc<Vec<u32>>, String>,
{
    if controls.budget == Some(0) {
        return Err(ThinkingPolicyError {
            message: format!("{} must be greater than zero", controls.budget_param),
            param: controls.budget_param,
        });
    }
    match channel {
        ReasoningChannel::QwenBudget => resolve_qwen_budget_channel(
            registration,
            controls,
            reasoning_forced_open,
            tool_choice,
            max_tokens,
            chain,
            server_base_budget,
            server_tool_budget,
            slot_aware,
            encode,
        ),
        ReasoningChannel::DeepSeekTier => resolve_deepseek_budget_channel(
            registration,
            controls,
            reasoning_forced_open,
            tool_choice,
            max_tokens,
            server_tool_budget,
            deepseek_required_tool_eligible,
            slot_aware,
            encode,
        ),
        ReasoningChannel::None => Ok(resolve_no_thinking_channel(
            server_base_budget,
            server_tool_budget,
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_qwen_budget_channel<F>(
    registration: Option<&registry::ModelRegistration>,
    controls: &ReasoningControls,
    reasoning_forced_open: bool,
    tool_choice: &ToolChoiceValue,
    max_tokens: usize,
    chain: QwenToolChainState,
    server_base_budget: Option<u32>,
    server_tool_budget: Option<u32>,
    slot_aware: bool,
    encode: F,
) -> Result<ThinkingBudgetResolution, ThinkingPolicyError>
where
    F: FnMut(&str) -> Result<Arc<Vec<u32>>, String>,
{
    if !reasoning_forced_open {
        // The rendered prompt does not end inside an open reasoning span, so
        // the budget-forcing machinery has no span to bound. An explicit
        // client budget number is a 400 (never accepted-and-ignored,
        // ADR-062 D5 bullet 4); an effort-derived budget is dropped.
        if controls.budget.is_some() {
            return Err(ThinkingPolicyError {
                message: format!(
                    "{} requires an open reasoning span; enable thinking or use a thinking-mode template",
                    controls.budget_param
                ),
                param: controls.budget_param,
            });
        }
        return Ok(ThinkingBudgetResolution {
            qwen_required_tool_mode: matches!(
                tool_choice,
                ToolChoiceValue::Required | ToolChoiceValue::Function(_)
            ),
            dropped_budget: controls
                .effort
                .and_then(EffortLevel::budget)
                .map(|budget| (budget, DROPPED_NO_OPEN_SPAN)),
            ..ThinkingBudgetResolution::default()
        });
    }
    let requested = controls
        .budget
        .map(|tokens| RequestedThinkingBudget {
            tokens,
            strict: true,
        })
        .or_else(|| {
            controls
                .effort
                .and_then(EffortLevel::budget)
                .map(|tokens| RequestedThinkingBudget {
                    tokens,
                    strict: false,
                })
        });
    let defaults = QwenThinkingDefaults::from_config(server_base_budget, server_tool_budget);
    let resolved = resolve_qwen_thinking_policy(
        registration,
        reasoning_forced_open,
        tool_choice,
        requested,
        controls.budget_param,
        max_tokens,
        chain,
        defaults,
        slot_aware,
        encode,
    )?;
    Ok(ThinkingBudgetResolution {
        thinking_budget: resolved.effective_budget,
        reasoning_end_tokens: resolved.end_tokens,
        reasoning_close_tokens: resolved.close_tokens,
        qwen_required_tool_mode: resolved.required_tool_mode,
        dropped_budget: resolved.dropped_budget,
        ..ThinkingBudgetResolution::default()
    })
}

#[allow(clippy::too_many_arguments)]
fn resolve_deepseek_budget_channel<F>(
    registration: Option<&registry::ModelRegistration>,
    controls: &ReasoningControls,
    reasoning_forced_open: bool,
    tool_choice: &ToolChoiceValue,
    max_tokens: usize,
    server_tool_budget: Option<u32>,
    deepseek_required_tool_eligible: bool,
    slot_aware: bool,
    mut encode: F,
) -> Result<ThinkingBudgetResolution, ThinkingPolicyError>
where
    F: FnMut(&str) -> Result<Arc<Vec<u32>>, String>,
{
    let client_budget = controls.budget;
    if client_budget.is_some() && !reasoning_forced_open {
        return Err(ThinkingPolicyError {
            message: format!(
                "{} requires an open reasoning span; enable thinking first",
                controls.budget_param
            ),
            param: controls.budget_param,
        });
    }
    // One meaning for the tool-thinking budget on every family (ADR-062 D5
    // bullet 2): under a constrained tool call the server bounds thinking
    // with the tool budget unless the client supplied its own number. An
    // explicit client budget under a constrained tool choice keeps the
    // same tool-mode handling as Qwen instead of tripping the
    // budget/tool_choice conflict check.
    let constrained_tool_choice = matches!(
        tool_choice,
        ToolChoiceValue::Required | ToolChoiceValue::Function(_)
    );
    let deepseek_required_tool_mode =
        deepseek_required_tool_eligible || (client_budget.is_some() && constrained_tool_choice);
    let server_budget = if deepseek_required_tool_eligible && client_budget.is_none() {
        enabled_default_thinking_budget(server_tool_budget)
    } else {
        None
    };
    let Some(configured_budget) = client_budget.or(server_budget) else {
        return Ok(ThinkingBudgetResolution {
            deepseek_required_tool_mode,
            ..ThinkingBudgetResolution::default()
        });
    };
    if !slot_aware {
        if client_budget.is_some() {
            return Err(ThinkingPolicyError {
                message: format!(
                    "{} requires an inflight-batched scheduler",
                    controls.budget_param
                ),
                param: controls.budget_param,
            });
        }
        // A server-default budget under fifo-serial is ignored with a
        // warning, never a 4xx (ADR-062 D5 bullet 3).
        return Ok(ThinkingBudgetResolution {
            deepseek_required_tool_mode,
            dropped_budget: Some((configured_budget, DROPPED_FIFO_SERIAL)),
            ..ThinkingBudgetResolution::default()
        });
    }
    let close = registration
        .and_then(|registration| registration.reasoning_close)
        .ok_or_else(|| ThinkingPolicyError {
            message: format!(
                "{} requires a registered reasoning close marker",
                controls.budget_param
            ),
            param: controls.budget_param,
        })?;
    let close_tokens = encode(close).map_err(|error| ThinkingPolicyError {
        message: format!("failed to tokenize reasoning boundary: {error}"),
        param: controls.budget_param,
    })?;
    if close_tokens.is_empty() {
        return Err(ThinkingPolicyError {
            message: "reasoning boundary tokenized to an empty sequence".into(),
            param: controls.budget_param,
        });
    }
    let effective_budget = effective_qwen_thinking_budget(
        Some(configured_budget),
        client_budget.is_some(),
        max_tokens,
        close_tokens.len(),
    )
    .map_err(|message| ThinkingPolicyError {
        message,
        param: controls.budget_param,
    })?;
    Ok(ThinkingBudgetResolution {
        thinking_budget: effective_budget,
        reasoning_end_tokens: effective_budget.map(|_| Arc::clone(&close_tokens)),
        reasoning_close_tokens: effective_budget.map(|_| close_tokens),
        qwen_required_tool_mode: false,
        deepseek_required_tool_mode,
        dropped_budget: (effective_budget.is_none() && client_budget.is_none())
            .then_some((configured_budget, DROPPED_NO_ROOM)),
    })
}

fn resolve_no_thinking_channel(
    server_base_budget: Option<u32>,
    server_tool_budget: Option<u32>,
) -> ThinkingBudgetResolution {
    // Reasoning controls were already rejected at the boundary (one 400
    // naming the control). A server-side default budget must never cause a
    // 4xx, so it is reported as dropped for the handler's warning instead
    // of being applied to a family that cannot bound a thinking span.
    ThinkingBudgetResolution {
        dropped_budget: server_base_budget
            .filter(|&budget| budget > 0)
            .or_else(|| server_tool_budget.filter(|&budget| budget > 0))
            .map(|budget| (budget as usize, DROPPED_NO_CHANNEL)),
        ..ThinkingBudgetResolution::default()
    }
}

/// A CLI/built-in thinking budget value: an explicit zero disables the
/// budget rather than deferring to a default.
pub(super) fn enabled_default_thinking_budget(value: Option<u32>) -> Option<usize> {
    value.and_then(|value| (value > 0).then_some(value as usize))
}
