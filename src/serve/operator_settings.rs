//! Shared resolution for the operator controls used by `serve` and `info`.
//!
//! This module is deliberately free of process-environment reads. The public
//! operator contract is CLI > the loaded family's built-in profile
//! (ADR-062 D1); the former `[serve]` behavior keys are retired (parse warns
//! and ignores), and the same functions drive both static preview and the
//! eventual model load.

use std::path::Path;

use mlx_native::gguf::GgufFile;

use crate::cli;
use crate::setup::{ServeDefaultsV2, GUIDE_MAX_SLOTS, GUIDE_SCHEDULER};

use super::api::engine::EngineMode;
use super::kv_persist::lcp_registry::LcpResumeBudget;

pub(crate) const DEFAULT_MAX_SLOTS_UNDER_INFLIGHT: u32 = GUIDE_MAX_SLOTS;

/// ADR-062 D3 (2026-10-08) — the slot count `--max-slots`, `[serve] max_slots`,
/// and the `hf2q setup` prompt accept on every family. Raising the bound
/// requires a hands-on qualification at the new width; it is not lifted by an
/// environment variable.
pub(crate) const MAX_SUPPORTED_SLOTS: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SettingOrigin {
    Cli,
    Config,
    Gguf,
    /// ADR-062 D1 — the built-in layer: the shared scheduler/slot constant,
    /// the built-in port (8081), and the per-family built-in serving
    /// profile. The former `[serve]` behavior keys retire into this layer.
    BuiltIn,
}

impl SettingOrigin {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Cli => "CLI",
            Self::Config => "config.toml",
            Self::Gguf => "GGUF",
            Self::BuiltIn => "built-in",
        }
    }

    /// Stable lowercase wire label for JSON origin fields such as
    /// `/hf2q/v1/runtime`'s per-value origin reporting.
    pub(crate) const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Cli => "cli",
            Self::Config => "config",
            Self::Gguf => "gguf",
            Self::BuiltIn => "builtin",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RequestedContext {
    pub(crate) tokens: u32,
    pub(crate) origin: SettingOrigin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResolvedContext {
    pub(crate) declared_tokens: u32,
    pub(crate) effective_tokens: u32,
    pub(crate) origin: SettingOrigin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResolvedKvBudget {
    pub(crate) bytes: Option<u64>,
    pub(crate) origin: Option<SettingOrigin>,
}

/// #727: built-in host byte budget of the Qwen SlotAware reclaimed-slot spill
/// store when neither `--slot-spill-budget` nor `[serve] slot_spill_budget`
/// sets one. A fixed constant, never derived from available memory, so the
/// spill/evict sequence depends only on the request sequence. One 32K-token
/// spill (cache rows, both DeltaNet ping-pong buffers, and the prompt
/// anchor) is about 1.6 GB on Qwen3.8 27B and 0.57 GB on Qwen3.6 35B-A3B,
/// so 16 GiB holds about ten or thirty such conversations.
pub(crate) const DEFAULT_SLOT_SPILL_BUDGET_BYTES: u64 = 16 << 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResolvedSlotSpillBudget {
    /// Host bytes the spill store may hold; `0` disables spilling.
    pub(crate) bytes: u64,
    pub(crate) origin: SettingOrigin,
}

/// ADR-062 D1 — provenance of the effective scheduler and slot count,
/// reported by `hf2q info` and `/hf2q/v1/runtime` (CLI flag, `[serve]`
/// config key, or the built-in default from one shared constant).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResolvedSchedulerOrigins {
    pub(crate) scheduler: SettingOrigin,
    pub(crate) max_slots: SettingOrigin,
}

pub(crate) fn resolve_scheduler_origins(
    planning: &cli::ServePlanningArgs,
    defaults: Option<&ServeDefaultsV2>,
) -> ResolvedSchedulerOrigins {
    // The same precedence chain as `resolve_scheduler`. Under `fifo-serial`
    // the slot count is structurally 1, so only the scheduler origin applies.
    let layer = if planning.scheduler.is_some() {
        SettingOrigin::Cli
    } else if defaults.is_some() {
        SettingOrigin::Config
    } else {
        SettingOrigin::BuiltIn
    };
    let max_slots = if planning.max_slots.is_some() {
        SettingOrigin::Cli
    } else if defaults.is_some() {
        SettingOrigin::Config
    } else {
        SettingOrigin::BuiltIn
    };
    ResolvedSchedulerOrigins {
        scheduler: layer,
        max_slots,
    }
}

/// Explicit CLI `--default-*` serving-behavior layer resolved at startup.
///
/// Every value field stays `None` until the operator passes the matching
/// flag. The family built-in layer ([`family_serve_profile`]) is applied per
/// loaded engine at request time (ADR-062 D1), so a serve with no startup
/// model resolves the same values once its engine loads. The `*_origin`
/// fields state where each effective value comes from: `Cli` when the flag
/// is passed, otherwise the family built-in layer (the former `[serve]`
/// behavior keys are retired, so no Config origin exists for these keys).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ResolvedServeBehavior {
    pub(crate) repetition_penalty: Option<f32>,
    /// `Cli` when `--default-repetition-penalty` is passed; otherwise
    /// `BuiltIn` — every family profile states a repetition penalty.
    pub(crate) repetition_penalty_origin: SettingOrigin,
    pub(crate) thinking_token_budget: Option<u32>,
    /// `Some(Cli)` when `--default-thinking-token-budget` is passed;
    /// otherwise `None` — whether the family built-in layer supplies a
    /// value is resolved per loaded engine.
    pub(crate) thinking_token_budget_origin: Option<SettingOrigin>,
    pub(crate) tool_thinking_token_budget: Option<u32>,
    /// `Some(Cli)` when `--default-tool-thinking-token-budget` is passed;
    /// otherwise `None` — resolved per loaded engine.
    pub(crate) tool_thinking_token_budget_origin: Option<SettingOrigin>,
}

/// Per-family built-in serving profile (ADR-062 D1). One table states every
/// key per family — an absent key is a decision, not an omission. The values
/// are the ones each canonical launcher already passes, so a plain
/// `hf2q serve` reproduces launcher-qualified behavior with no `HF2Q_*`
/// ritual, and the built-ins move with the binary when a family is
/// re-qualified.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FamilyServeProfile {
    pub(crate) repetition_penalty: f32,
    pub(crate) thinking_token_budget: Option<u32>,
    pub(crate) tool_thinking_token_budget: Option<u32>,
}

/// The built-in serving values for a loaded family. The `qwen35` registration
/// covers the `qwen35` and `qwen35moe` GGUF architectures.
pub(crate) fn family_serve_profile(family: Option<&str>) -> FamilyServeProfile {
    match family {
        // serve_qwen36_opencode.sh: loop mitigation plus bounded native
        // reasoning for fresh turns and tool-result continuations.
        Some("qwen35") => FamilyServeProfile {
            repetition_penalty: 1.05,
            thinking_token_budget: Some(2_048),
            tool_thinking_token_budget: Some(512),
        },
        // serve_gemma4_opencode.sh: loop mitigation only. The Gemma 4
        // template has no thinking channel, so there are no thinking
        // budgets (an absent key is a decision).
        Some("gemma4") => FamilyServeProfile {
            repetition_penalty: 1.05,
            thinking_token_budget: None,
            tool_thinking_token_budget: None,
        },
        // serve_deepseek4_opencode.sh values with the owner's 2026-10-08
        // retirement of its 8-token budget (ADR-062 D1/D5): no repetition
        // penalty (1.05 distorted constrained tool strings), a 512-token
        // tool-thinking budget, and no base thinking budget.
        Some("deepseek4") => FamilyServeProfile {
            repetition_penalty: 1.0,
            thinking_token_budget: None,
            tool_thinking_token_budget: Some(512),
        },
        // qwen3vl and unregistered families keep the neutral values.
        _ => FamilyServeProfile {
            repetition_penalty: 1.0,
            thinking_token_budget: None,
            tool_thinking_token_budget: None,
        },
    }
}

pub(crate) fn requested_context(
    cli_tokens: Option<u32>,
    defaults: Option<&ServeDefaultsV2>,
) -> Result<Option<RequestedContext>, String> {
    let requested = cli_tokens
        .map(|tokens| RequestedContext {
            tokens,
            origin: SettingOrigin::Cli,
        })
        .or_else(|| {
            defaults.and_then(|defaults| {
                defaults.ctx.map(|tokens| RequestedContext {
                    tokens,
                    origin: SettingOrigin::Config,
                })
            })
        });
    if let Some(RequestedContext { tokens: 0, origin }) = requested {
        return Err(format!(
            "context length from {} must be positive; omit `ctx` to use the GGUF maximum",
            origin.as_str()
        ));
    }
    Ok(requested)
}

pub(crate) fn declared_context_length(gguf: &GgufFile) -> Result<u32, String> {
    let arch = gguf
        .metadata_string("general.architecture")
        .filter(|arch| !arch.is_empty())
        .ok_or_else(|| "GGUF is missing required `general.architecture`".to_owned())?;
    let key = format!("{arch}.context_length");
    let declared = gguf
        .metadata_u32(&key)
        .ok_or_else(|| format!("GGUF is missing required declared context metadata `{key}`"))?;
    if declared == 0 {
        return Err(format!("GGUF declares invalid `{key}=0`"));
    }
    Ok(declared)
}

pub(crate) fn resolve_context_length(
    declared_tokens: u32,
    requested: Option<RequestedContext>,
) -> Result<ResolvedContext, String> {
    if declared_tokens == 0 {
        return Err("GGUF declared context maximum must be positive".to_owned());
    }
    let Some(requested) = requested else {
        return Ok(ResolvedContext {
            declared_tokens,
            effective_tokens: declared_tokens,
            origin: SettingOrigin::Gguf,
        });
    };
    if requested.tokens == 0 {
        return Err(format!(
            "context length from {} must be positive; omit `ctx` to use the GGUF maximum",
            requested.origin.as_str()
        ));
    }
    if requested.tokens > declared_tokens {
        return Err(format!(
            "requested context length {} from {} exceeds the GGUF declared maximum of {} tokens",
            requested.tokens,
            requested.origin.as_str(),
            declared_tokens
        ));
    }
    Ok(ResolvedContext {
        declared_tokens,
        effective_tokens: requested.tokens,
        origin: requested.origin,
    })
}

pub(crate) fn resolve_context_for_gguf(
    gguf: &GgufFile,
    requested: Option<RequestedContext>,
) -> Result<ResolvedContext, String> {
    resolve_context_length(declared_context_length(gguf)?, requested)
}

pub(crate) fn resolve_scheduler(
    planning: &cli::ServePlanningArgs,
    defaults: Option<&ServeDefaultsV2>,
) -> Result<EngineMode, String> {
    // CLI > config.toml > the guide profile. Without any of them, `serve`
    // runs exactly what `hf2q setup --accept-defaults` would have recorded,
    // so an unconfigured server handles long agent prompts.
    let scheduler = planning
        .scheduler
        .or_else(|| defaults.map(|defaults| defaults.scheduler.as_cli()))
        .unwrap_or(GUIDE_SCHEDULER.as_cli());
    if scheduler == cli::SchedulerArg::FifoSerial {
        if planning.max_slots.is_some_and(|max_slots| max_slots != 1) {
            return Err(
                "--max-slots greater than 1 requires `--scheduler inflight-batched` or an inflight-batched setup config"
                    .to_owned(),
            );
        }
        return Ok(EngineMode::SerialFifo);
    }

    let max_slots = planning
        .max_slots
        .or_else(|| defaults.map(|defaults| defaults.max_slots))
        .unwrap_or(DEFAULT_MAX_SLOTS_UNDER_INFLIGHT);
    validate_max_slots(max_slots)?;
    Ok(EngineMode::SlotAware { max_slots })
}

/// ADR-062 D3 (2026-10-08) — one slot-range authority shared by the CLI, the
/// `config.toml` load, and the `hf2q setup` prompt. Every family accepts
/// 1-[`MAX_SUPPORTED_SLOTS`]; the check is family-independent.
pub(crate) fn validate_max_slots(max_slots: u32) -> Result<(), String> {
    if max_slots > MAX_SUPPORTED_SLOTS {
        return Err(format!(
            "max_slots {max_slots} exceeds the supported maximum of {MAX_SUPPORTED_SLOTS}; \
             the accepted range is 1-{MAX_SUPPORTED_SLOTS}, and raising the bound requires \
             a hands-on qualification at the new width"
        ));
    }
    if max_slots == 0 {
        return Err(format!(
            "max_slots must be in 1-{MAX_SUPPORTED_SLOTS}; omit `max_slots` to use the \
             inflight default of {DEFAULT_MAX_SLOTS_UNDER_INFLIGHT}"
        ));
    }
    Ok(())
}

pub(crate) fn resolve_kv_cache_budget(
    cli_value: Option<&str>,
    defaults: Option<&ServeDefaultsV2>,
) -> Result<ResolvedKvBudget, String> {
    let selected = cli_value
        .map(|value| (value, SettingOrigin::Cli))
        .or_else(|| {
            defaults
                .and_then(|defaults| defaults.kv_cache_budget.as_deref())
                .map(|value| (value, SettingOrigin::Config))
        });
    let Some((value, origin)) = selected else {
        return Ok(ResolvedKvBudget {
            bytes: None,
            origin: None,
        });
    };
    let bytes = parse_byte_size(value).map_err(|error| {
        let setting = match origin {
            SettingOrigin::Cli => "--kv-cache-budget",
            SettingOrigin::Config => "serve.kv_cache_budget",
            SettingOrigin::Gguf => unreachable!("GGUF never supplies a KV-cache byte budget"),
            SettingOrigin::BuiltIn => {
                unreachable!("the built-in layer never supplies a KV-cache byte budget")
            }
        };
        format!("{setting}: {error}")
    })?;
    Ok(ResolvedKvBudget {
        bytes: (bytes != 0).then_some(bytes),
        origin: Some(origin),
    })
}

pub(crate) fn resolve_kv_persist_budget(
    cli_value: Option<&str>,
    defaults: Option<&ServeDefaultsV2>,
) -> Result<ResolvedKvBudget, String> {
    let selected = cli_value
        .map(|value| (value, SettingOrigin::Cli))
        .or_else(|| {
            defaults
                .and_then(|defaults| defaults.kv_persist_budget.as_deref())
                .map(|value| (value, SettingOrigin::Config))
        });
    let Some((value, origin)) = selected else {
        return Ok(ResolvedKvBudget {
            bytes: None,
            origin: None,
        });
    };
    let bytes = parse_byte_size(value).map_err(|error| {
        let setting = match origin {
            SettingOrigin::Cli => "--kv-persist-budget",
            SettingOrigin::Config => "serve.kv_persist_budget",
            SettingOrigin::Gguf => unreachable!("GGUF never supplies a persistent-KV budget"),
            SettingOrigin::BuiltIn => {
                unreachable!("the built-in layer never supplies a persistent-KV budget")
            }
        };
        format!("{setting}: {error}")
    })?;
    Ok(ResolvedKvBudget {
        bytes: (bytes != 0).then_some(bytes),
        origin: Some(origin),
    })
}

/// Effective in-memory LCP resume budget and the layer that supplied it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResolvedLcpResumeBudget {
    pub(crate) budget: LcpResumeBudget,
    pub(crate) origin: SettingOrigin,
}

impl ResolvedLcpResumeBudget {
    /// Label for the `[hf2q lcp] byte_budget=... (source=...)` startup line.
    pub(crate) const fn source_label(self) -> &'static str {
        match self.origin {
            SettingOrigin::Cli => "cli",
            SettingOrigin::Config => "config",
            SettingOrigin::Gguf | SettingOrigin::BuiltIn => "default",
        }
    }
}

/// Parse one LCP resume budget value. Zero is rejected: a zero-byte registry
/// would evict every snapshot on store.
pub(crate) fn parse_lcp_resume_budget(value: &str) -> Result<u64, String> {
    let bytes = parse_byte_size(value)?;
    if bytes == 0 {
        return Err("must be greater than zero".to_owned());
    }
    Ok(bytes)
}

/// Resolve the LCP resume registry budget: `--lcp-resume-budget` >
/// `[serve] lcp_resume_budget` > [`LcpResumeBudget::DEFAULT`]. A pure
/// function of operator input; host memory is never consulted.
pub(crate) fn resolve_lcp_resume_budget(
    cli_value: Option<&str>,
    defaults: Option<&ServeDefaultsV2>,
) -> Result<ResolvedLcpResumeBudget, String> {
    let selected = cli_value
        .map(|value| (value, SettingOrigin::Cli))
        .or_else(|| {
            defaults
                .and_then(|defaults| defaults.lcp_resume_budget.as_deref())
                .map(|value| (value, SettingOrigin::Config))
        });
    let Some((value, origin)) = selected else {
        return Ok(ResolvedLcpResumeBudget {
            budget: LcpResumeBudget::DEFAULT,
            origin: SettingOrigin::BuiltIn,
        });
    };
    let bytes = parse_lcp_resume_budget(value).map_err(|error| {
        let setting = match origin {
            SettingOrigin::Cli => "--lcp-resume-budget",
            SettingOrigin::Config => "serve.lcp_resume_budget",
            SettingOrigin::Gguf | SettingOrigin::BuiltIn => {
                unreachable!("only the CLI and config layers supply an LCP resume budget")
            }
        };
        format!("{setting}: {error}")
    })?;
    Ok(ResolvedLcpResumeBudget {
        budget: LcpResumeBudget::from_bytes(bytes),
        origin,
    })
}

/// #727: resolve the reclaimed-slot spill budget, CLI > `[serve]` config >
/// [`DEFAULT_SLOT_SPILL_BUDGET_BYTES`]. An explicit `0` disables spilling.
pub(crate) fn resolve_slot_spill_budget(
    cli_value: Option<&str>,
    defaults: Option<&ServeDefaultsV2>,
) -> Result<ResolvedSlotSpillBudget, String> {
    let selected = cli_value
        .map(|value| (value, SettingOrigin::Cli))
        .or_else(|| {
            defaults
                .and_then(|defaults| defaults.slot_spill_budget.as_deref())
                .map(|value| (value, SettingOrigin::Config))
        });
    let Some((value, origin)) = selected else {
        return Ok(ResolvedSlotSpillBudget {
            bytes: DEFAULT_SLOT_SPILL_BUDGET_BYTES,
            origin: SettingOrigin::BuiltIn,
        });
    };
    let bytes = parse_byte_size(value).map_err(|error| {
        let setting = match origin {
            SettingOrigin::Cli => "--slot-spill-budget",
            SettingOrigin::Config => "serve.slot_spill_budget",
            SettingOrigin::Gguf | SettingOrigin::BuiltIn => {
                unreachable!("only the CLI and config.toml supply a spill budget string")
            }
        };
        format!("{setting}: {error}")
    })?;
    Ok(ResolvedSlotSpillBudget { bytes, origin })
}

/// Reject a command-line disk budget that cannot affect the requested
/// invocation. A setup-level budget may remain dormant until a particular
/// serve/info invocation supplies `--kv-persist PATH`.
pub(crate) fn validate_kv_persist_plan(
    persist_path: Option<&Path>,
    budget: ResolvedKvBudget,
) -> Result<(), String> {
    if persist_path.is_some_and(|path| path.as_os_str().is_empty()) {
        return Err("--kv-persist PATH cannot be empty".to_owned());
    }
    if persist_path.is_none() && budget.origin == Some(SettingOrigin::Cli) {
        return Err(
            "--kv-persist-budget requires --kv-persist PATH so the disk ceiling has an active store"
                .to_owned(),
        );
    }
    Ok(())
}

/// Resolve the CLI behavior layer. Request-time precedence is
/// request field > these CLI values > [`family_serve_profile`] (ADR-062 D1);
/// the former `[serve]` behavior keys are retired (parse warns and ignores).
pub(crate) fn resolve_serve_behavior(
    cli: &cli::ServeBehaviorArgs,
) -> Result<ResolvedServeBehavior, String> {
    if let Some(repetition_penalty) = cli.default_repetition_penalty {
        if !repetition_penalty.is_finite() || repetition_penalty <= 0.0 {
            return Err("--default-repetition-penalty must be a finite positive number".to_owned());
        }
    }
    // Preserve an explicit zero through the typed configuration. Qwen's
    // policy treats a zero base as disabled and, importantly, a zero tool
    // override as "do not derive a continuation budget from the base".
    // Collapsing zero to `None` here would lose that distinction.
    Ok(ResolvedServeBehavior {
        repetition_penalty: cli.default_repetition_penalty,
        // ADR-062 D1 reporting: every family profile states a repetition
        // penalty, so an absent flag always resolves to the built-in layer.
        repetition_penalty_origin: if cli.default_repetition_penalty.is_some() {
            SettingOrigin::Cli
        } else {
            SettingOrigin::BuiltIn
        },
        thinking_token_budget: cli.default_thinking_token_budget,
        thinking_token_budget_origin: cli
            .default_thinking_token_budget
            .map(|_| SettingOrigin::Cli),
        tool_thinking_token_budget: cli.default_tool_thinking_token_budget,
        tool_thinking_token_budget_origin: cli
            .default_tool_thinking_token_budget
            .map(|_| SettingOrigin::Cli),
    })
}

/// ADR-062 D1 — the effective per-family serving values with the origin of
/// each value, as reported by `hf2q info` and the measurement snapshot.
/// Precedence is CLI `--default-*` > [`family_serve_profile`]; the former
/// `[serve]` behavior keys are retired, so the only possible origins are
/// `Cli` and `BuiltIn`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct EffectiveServeBehavior {
    pub(crate) repetition_penalty: f32,
    pub(crate) repetition_penalty_origin: SettingOrigin,
    pub(crate) thinking_token_budget: Option<u32>,
    pub(crate) thinking_token_budget_origin: Option<SettingOrigin>,
    pub(crate) tool_thinking_token_budget: Option<u32>,
    pub(crate) tool_thinking_token_budget_origin: Option<SettingOrigin>,
}

/// Fold the startup CLI behavior layer over the loaded family's built-in
/// profile. An explicit CLI zero is preserved (it disables the budget
/// instead of deferring to the profile).
pub(crate) fn resolve_effective_serve_behavior(
    cli_layer: &ResolvedServeBehavior,
    family: Option<&str>,
) -> EffectiveServeBehavior {
    let profile = family_serve_profile(family);
    EffectiveServeBehavior {
        repetition_penalty: cli_layer
            .repetition_penalty
            .unwrap_or(profile.repetition_penalty),
        repetition_penalty_origin: cli_layer.repetition_penalty_origin,
        thinking_token_budget: cli_layer
            .thinking_token_budget
            .or(profile.thinking_token_budget),
        thinking_token_budget_origin: cli_layer
            .thinking_token_budget_origin
            .or(profile.thinking_token_budget.map(|_| SettingOrigin::BuiltIn)),
        tool_thinking_token_budget: cli_layer
            .tool_thinking_token_budget
            .or(profile.tool_thinking_token_budget),
        tool_thinking_token_budget_origin: cli_layer
            .tool_thinking_token_budget_origin
            .or(profile
                .tool_thinking_token_budget
                .map(|_| SettingOrigin::BuiltIn)),
    }
}

/// Parse a non-negative byte count with an optional SI or IEC suffix.
/// Bare integers remain byte-compatible with the former raw-byte flag.
pub(crate) fn parse_byte_size(raw: &str) -> Result<u64, String> {
    let value = raw.trim();
    if value.is_empty() {
        return Err("size cannot be empty".to_owned());
    }
    let split = value
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(value.len());
    let (number, unit) = value.split_at(split);
    if number.is_empty() {
        return Err(byte_size_error(raw));
    }
    let amount = number.parse::<u64>().map_err(|_| byte_size_error(raw))?;
    let normalized = unit.trim().to_ascii_lowercase();
    let multiplier = match normalized.as_str() {
        "" | "b" => 1,
        "kb" => 1_000,
        "mb" => 1_000_000,
        "gb" => 1_000_000_000,
        "tb" => 1_000_000_000_000,
        "kib" => 1 << 10,
        "mib" => 1 << 20,
        "gib" => 1 << 30,
        "tib" => 1u64 << 40,
        _ => return Err(byte_size_error(raw)),
    };
    amount
        .checked_mul(multiplier)
        .ok_or_else(|| format!("size {raw:?} exceeds the u64 byte range"))
}

fn byte_size_error(raw: &str) -> String {
    format!("size {raw:?} is invalid; use a non-negative byte count or a unit such as `8GiB`")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_context_uses_the_gguf_declared_maximum() {
        assert_eq!(
            resolve_context_length(1_048_576, None).unwrap(),
            ResolvedContext {
                declared_tokens: 1_048_576,
                effective_tokens: 1_048_576,
                origin: SettingOrigin::Gguf,
            }
        );
    }

    #[test]
    fn explicit_context_is_per_slot_and_cannot_exceed_gguf() {
        let request = RequestedContext {
            tokens: 262_144,
            origin: SettingOrigin::Cli,
        };
        assert_eq!(
            resolve_context_length(1_048_576, Some(request))
                .unwrap()
                .effective_tokens,
            262_144
        );
        let error = resolve_context_length(
            262_144,
            Some(RequestedContext {
                tokens: 262_145,
                origin: SettingOrigin::Config,
            }),
        )
        .unwrap_err();
        assert!(error.contains("262145"), "{error}");
        assert!(error.contains("262144"), "{error}");
        assert!(error.contains("config.toml"), "{error}");
    }

    #[test]
    fn zero_context_is_rejected_instead_of_meaning_auto() {
        let error = resolve_context_length(
            1_048_576,
            Some(RequestedContext {
                tokens: 0,
                origin: SettingOrigin::Config,
            }),
        )
        .unwrap_err();
        assert!(error.contains("omit `ctx`"), "{error}");
    }

    #[test]
    fn lcp_resume_budget_default_is_exactly_8gib_and_host_independent() {
        assert_eq!(LcpResumeBudget::DEFAULT.bytes(), 8 * (1u64 << 30));
        assert_eq!(LcpResumeBudget::default(), LcpResumeBudget::DEFAULT);
        let resolved = resolve_lcp_resume_budget(None, None).unwrap();
        assert_eq!(resolved.budget, LcpResumeBudget::DEFAULT);
        assert_eq!(resolved.origin, SettingOrigin::BuiltIn);
        assert_eq!(resolved.source_label(), "default");
        // A config without the key also resolves to the constant.
        let config = crate::setup::OperatorConfigV2::guide_defaults().unwrap();
        let resolved = resolve_lcp_resume_budget(None, Some(&config.serve)).unwrap();
        assert_eq!(resolved.budget, LcpResumeBudget::DEFAULT);
        assert_eq!(resolved.origin, SettingOrigin::BuiltIn);
    }

    #[test]
    fn lcp_resume_budget_precedence_is_cli_then_config_then_default() {
        let mut config = crate::setup::OperatorConfigV2::guide_defaults().unwrap();
        config.serve.lcp_resume_budget = Some("12GiB".to_owned());

        let from_config = resolve_lcp_resume_budget(None, Some(&config.serve)).unwrap();
        assert_eq!(from_config.budget.bytes(), 12 * (1u64 << 30));
        assert_eq!(from_config.origin, SettingOrigin::Config);
        assert_eq!(from_config.source_label(), "config");

        let from_cli = resolve_lcp_resume_budget(Some("2GiB"), Some(&config.serve)).unwrap();
        assert_eq!(from_cli.budget.bytes(), 2 * (1u64 << 30));
        assert_eq!(from_cli.origin, SettingOrigin::Cli);
        assert_eq!(from_cli.source_label(), "cli");

        let bare = resolve_lcp_resume_budget(Some("8589934592"), None).unwrap();
        assert_eq!(bare.budget, LcpResumeBudget::DEFAULT);
    }

    #[test]
    fn lcp_resume_budget_rejects_zero_and_garbage_naming_the_layer() {
        let error = resolve_lcp_resume_budget(Some("0"), None).unwrap_err();
        assert!(error.contains("--lcp-resume-budget"), "{error}");
        assert!(error.contains("greater than zero"), "{error}");
        let mut config = crate::setup::OperatorConfigV2::guide_defaults().unwrap();
        config.serve.lcp_resume_budget = Some("lots".to_owned());
        let error = resolve_lcp_resume_budget(None, Some(&config.serve)).unwrap_err();
        assert!(error.contains("serve.lcp_resume_budget"), "{error}");
    }

    #[test]
    fn lcp_resume_budget_admits_entries_up_to_the_budget_inclusive() {
        let budget = LcpResumeBudget::from_bytes(1_000);
        assert!(budget.admits(999));
        assert!(budget.admits(1_000));
        assert!(!budget.admits(1_001));
    }

    #[test]
    fn lcp_resume_budget_path_reads_no_env_or_host_memory() {
        // #730 removed-guard: the retired env knob and the free-memory probe
        // must not return to the registry budget path.
        let registry_src = include_str!("kv_persist/lcp_registry.rs");
        let serve_src = include_str!("mod.rs");
        for src in [registry_src, serve_src] {
            assert!(!src.contains("std::env::var(\"HF2Q_KV_LCP_RESUME_CAPACITY\")"));
        }
        assert!(!registry_src.contains("sysinfo"));
        assert!(!registry_src.contains("available_memory"));
        assert!(serve_src.contains("operator_settings::resolve_lcp_resume_budget"));
        // Model swaps reuse the once-resolved value.
        assert!(serve_src.contains("lcp_resume_budget: config.lcp_resume_budget,"));

        // Both engines build their registry from the load options, and Gemma's
        // per-request snapshot gate reads the same value off the weights.
        let gemma_engine_src = include_str!("api/engine.rs");
        assert!(gemma_engine_src.contains("weights.lcp_resume_budget = opts.lcp_resume_budget;"));
        assert!(gemma_engine_src.contains("opts.lcp_resume_budget.bytes(),"));
        let qwen_engine_src = include_str!("api/engine_qwen35.rs");
        assert!(qwen_engine_src.contains("opts.lcp_resume_budget.bytes(),"));

        // The batched file keeps an unrelated per-request route probe
        // (`serve_batched_route_viable`), so the LCP gate is pinned by its exact
        // statement rather than by banning the probe file-wide.
        let gate = "let lcp_fits_budget = self.lcp_resume_budget.admits(lcp_est_bytes);";
        let prefill_src = include_str!("forward_prefill.rs");
        let batched_src = include_str!("forward_prefill_batched.rs");
        for src in [prefill_src, batched_src, gemma_engine_src, qwen_engine_src] {
            assert!(!src.contains("default_lcp_byte_budget"));
            assert!(!src.contains("gemma_lcp_snapshot_fits_budget"));
        }
        assert!(prefill_src.contains(gate));
        assert!(batched_src.contains(gate));
        assert!(!prefill_src.contains("available_memory"));
    }

    #[test]
    fn byte_budget_accepts_human_units_and_legacy_bare_bytes() {
        assert_eq!(parse_byte_size("8GiB").unwrap(), 8 * (1u64 << 30));
        assert_eq!(parse_byte_size(" 2 GB ").unwrap(), 2_000_000_000);
        assert_eq!(parse_byte_size("48000").unwrap(), 48_000);
        assert_eq!(parse_byte_size("0").unwrap(), 0);
        assert!(parse_byte_size("8GiBits").is_err());
        assert!(parse_byte_size("-1GiB").is_err());
    }

    #[test]
    fn behavior_cli_overrides_the_family_builtin_and_zero_disables_optional_budgets() {
        let resolved = resolve_serve_behavior(&cli::ServeBehaviorArgs {
            default_repetition_penalty: Some(1.1),
            default_thinking_token_budget: Some(0),
            default_tool_thinking_token_budget: Some(64),
        })
        .unwrap();
        assert_eq!(resolved.repetition_penalty, Some(1.1));
        assert_eq!(resolved.thinking_token_budget, Some(0));
        assert_eq!(resolved.tool_thinking_token_budget, Some(64));
        // ADR-062 D1 reporting: an explicit flag is the Cli origin on every
        // key, including an explicit zero.
        assert_eq!(resolved.repetition_penalty_origin, SettingOrigin::Cli);
        assert_eq!(
            resolved.thinking_token_budget_origin,
            Some(SettingOrigin::Cli)
        );
        assert_eq!(
            resolved.tool_thinking_token_budget_origin,
            Some(SettingOrigin::Cli)
        );

        let zero_tool = resolve_serve_behavior(&cli::ServeBehaviorArgs {
            default_repetition_penalty: None,
            default_thinking_token_budget: None,
            default_tool_thinking_token_budget: Some(0),
        })
        .unwrap();
        assert_eq!(zero_tool.tool_thinking_token_budget, Some(0));
        // Without flags the layer defers to the family built-in profile:
        // the repetition penalty is always built-in; a budget's origin is
        // resolved per loaded engine.
        assert_eq!(zero_tool.repetition_penalty_origin, SettingOrigin::BuiltIn);
        assert_eq!(zero_tool.thinking_token_budget_origin, None);
        assert_eq!(
            zero_tool.tool_thinking_token_budget_origin,
            Some(SettingOrigin::Cli)
        );
    }

    #[test]
    fn effective_behavior_folds_the_cli_layer_over_the_family_profile() {
        let empty_cli_layer = resolve_serve_behavior(&cli::ServeBehaviorArgs {
            default_repetition_penalty: None,
            default_thinking_token_budget: None,
            default_tool_thinking_token_budget: None,
        })
        .unwrap();
        // The qwen35 registration covers the qwen35 and qwen35moe GGUF
        // architectures; its profile supplies every key.
        let qwen = resolve_effective_serve_behavior(&empty_cli_layer, Some("qwen35"));
        assert_eq!(qwen.repetition_penalty, 1.05);
        assert_eq!(qwen.repetition_penalty_origin, SettingOrigin::BuiltIn);
        assert_eq!(qwen.thinking_token_budget, Some(2_048));
        assert_eq!(
            qwen.thinking_token_budget_origin,
            Some(SettingOrigin::BuiltIn)
        );
        assert_eq!(qwen.tool_thinking_token_budget, Some(512));
        assert_eq!(
            qwen.tool_thinking_token_budget_origin,
            Some(SettingOrigin::BuiltIn)
        );

        // A CLI flag overrides the profile value and carries the Cli origin;
        // an explicit zero stays a disable instead of deferring.
        let cli_layer = resolve_serve_behavior(&cli::ServeBehaviorArgs {
            default_repetition_penalty: Some(1.1),
            default_thinking_token_budget: Some(0),
            default_tool_thinking_token_budget: None,
        })
        .unwrap();
        let overridden = resolve_effective_serve_behavior(&cli_layer, Some("qwen35"));
        assert_eq!(overridden.repetition_penalty, 1.1);
        assert_eq!(overridden.repetition_penalty_origin, SettingOrigin::Cli);
        assert_eq!(overridden.thinking_token_budget, Some(0));
        assert_eq!(overridden.thinking_token_budget_origin, Some(SettingOrigin::Cli));
        assert_eq!(overridden.tool_thinking_token_budget, Some(512));
        assert_eq!(
            overridden.tool_thinking_token_budget_origin,
            Some(SettingOrigin::BuiltIn)
        );

        // An unregistered family keeps the neutral values; absent budgets
        // stay request-derived with no origin.
        let neutral = resolve_effective_serve_behavior(&empty_cli_layer, None);
        assert_eq!(neutral.repetition_penalty, 1.0);
        assert_eq!(neutral.repetition_penalty_origin, SettingOrigin::BuiltIn);
        assert_eq!(neutral.thinking_token_budget, None);
        assert_eq!(neutral.thinking_token_budget_origin, None);
        assert_eq!(neutral.tool_thinking_token_budget, None);
        assert_eq!(neutral.tool_thinking_token_budget_origin, None);
    }

    #[test]
    fn behavior_rejects_invalid_repetition_penalty() {
        let error = resolve_serve_behavior(&cli::ServeBehaviorArgs {
            default_repetition_penalty: Some(f32::NAN),
            default_thinking_token_budget: None,
            default_tool_thinking_token_budget: None,
        })
        .unwrap_err();
        assert!(error.contains("finite positive"), "{error}");
    }

    #[test]
    fn cli_planning_values_override_setup_config() {
        let mut config = crate::setup::OperatorConfigV2::guide_defaults().unwrap();
        config.serve.ctx = Some(524_288);
        config.serve.kv_cache_budget = Some("16GiB".to_owned());
        config.serve.kv_persist_budget = Some("64GiB".to_owned());
        let planning = cli::ServePlanningArgs {
            ctx: Some(262_144),
            scheduler: Some(cli::SchedulerArg::FifoSerial),
            max_slots: Some(1),
            kv_cache_budget: Some("2GiB".to_owned()),
            kv_persist_path: Some("/tmp/hf2q-kv".into()),
            kv_persist_budget: Some("32GiB".to_owned()),
            slot_spill_budget: None,
        };
        let requested = requested_context(planning.ctx, Some(&config.serve))
            .unwrap()
            .unwrap();
        assert_eq!(requested.tokens, 262_144);
        assert_eq!(requested.origin, SettingOrigin::Cli);
        assert_eq!(
            resolve_scheduler(&planning, Some(&config.serve)).unwrap(),
            EngineMode::SerialFifo
        );
        let budget =
            resolve_kv_cache_budget(planning.kv_cache_budget.as_deref(), Some(&config.serve))
                .unwrap();
        assert_eq!(budget.bytes, Some(2 * (1u64 << 30)));
        assert_eq!(budget.origin, Some(SettingOrigin::Cli));
        let persist_budget =
            resolve_kv_persist_budget(planning.kv_persist_budget.as_deref(), Some(&config.serve))
                .unwrap();
        assert_eq!(persist_budget.bytes, Some(32 * (1u64 << 30)));
        assert_eq!(persist_budget.origin, Some(SettingOrigin::Cli));
    }

    #[test]
    fn max_slots_is_not_silently_ignored_by_fifo() {
        let planning = cli::ServePlanningArgs {
            ctx: None,
            scheduler: Some(cli::SchedulerArg::FifoSerial),
            max_slots: Some(4),
            kv_cache_budget: None,
            kv_persist_path: None,
            kv_persist_budget: None,
            slot_spill_budget: None,
        };
        let error = resolve_scheduler(&planning, None).unwrap_err();
        assert!(error.contains("inflight-batched"), "{error}");
    }

    #[test]
    fn slot_spill_budget_is_cli_then_config_then_a_fixed_default() {
        let builtin = resolve_slot_spill_budget(None, None).unwrap();
        assert_eq!(builtin.bytes, DEFAULT_SLOT_SPILL_BUDGET_BYTES);
        assert_eq!(builtin.bytes, 16 * (1u64 << 30));
        assert_eq!(builtin.origin, SettingOrigin::BuiltIn);

        let mut config = crate::setup::OperatorConfigV2::guide_defaults().unwrap();
        let unset = resolve_slot_spill_budget(None, Some(&config.serve)).unwrap();
        assert_eq!(unset, builtin);

        config.serve.slot_spill_budget = Some("6GiB".to_owned());
        let configured = resolve_slot_spill_budget(None, Some(&config.serve)).unwrap();
        assert_eq!(configured.bytes, 6 * (1u64 << 30));
        assert_eq!(configured.origin, SettingOrigin::Config);

        let cli = resolve_slot_spill_budget(Some("0"), Some(&config.serve)).unwrap();
        assert_eq!(cli.bytes, 0, "an explicit zero disables spilling");
        assert_eq!(cli.origin, SettingOrigin::Cli);

        let error = resolve_slot_spill_budget(Some("lots"), None).unwrap_err();
        assert!(error.starts_with("--slot-spill-budget:"), "{error}");
        config.serve.slot_spill_budget = Some("lots".to_owned());
        let error = resolve_slot_spill_budget(None, Some(&config.serve)).unwrap_err();
        assert!(error.starts_with("serve.slot_spill_budget:"), "{error}");
    }

    #[test]
    fn explicit_persistent_budget_requires_an_active_store_path() {
        let cli_budget = ResolvedKvBudget {
            bytes: Some(32 * (1u64 << 30)),
            origin: Some(SettingOrigin::Cli),
        };
        let error = validate_kv_persist_plan(None, cli_budget).unwrap_err();
        assert!(error.contains("--kv-persist PATH"), "{error}");
        validate_kv_persist_plan(Some(Path::new("/tmp/hf2q-kv")), cli_budget).unwrap();

        let configured_budget = ResolvedKvBudget {
            bytes: Some(32 * (1u64 << 30)),
            origin: Some(SettingOrigin::Config),
        };
        validate_kv_persist_plan(None, configured_budget).unwrap();
    }

    #[test]
    fn persistent_budget_has_no_production_environment_reader() {
        let removed = ["HF2Q", "KV", "PERSIST", "BUDGET", "BYTES"].join("_");
        for source in [include_str!("mod.rs"), include_str!("api/engine_qwen35.rs")] {
            assert!(
                !source.contains(&removed),
                "persistent budget regressed to hidden environment UX"
            );
        }
    }
}
