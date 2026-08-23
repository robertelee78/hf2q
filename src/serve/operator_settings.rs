//! Shared resolution for the operator controls used by `serve` and `info`.
//!
//! This module is deliberately free of process-environment reads. The public
//! operator contract is CLI > `config.toml` > built-in/model default, and the
//! same functions drive both static preview and the eventual model load.

use std::path::Path;

use mlx_native::gguf::GgufFile;

use crate::cli;
use crate::setup::ServeDefaultsV2;

use super::api::engine::EngineMode;

pub(crate) const DEFAULT_MAX_SLOTS_UNDER_INFLIGHT: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SettingOrigin {
    Cli,
    Config,
    Gguf,
}

impl SettingOrigin {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Cli => "CLI",
            Self::Config => "config.toml",
            Self::Gguf => "GGUF",
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ResolvedServeBehavior {
    pub(crate) repetition_penalty: f32,
    pub(crate) thinking_token_budget: Option<u32>,
    pub(crate) tool_thinking_token_budget: Option<u32>,
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
    let scheduler = planning
        .scheduler
        .or_else(|| defaults.map(|defaults| defaults.scheduler.as_cli()));
    if matches!(scheduler, None | Some(cli::SchedulerArg::FifoSerial)) {
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
    if max_slots == 0 {
        return Err(format!(
            "max concurrent slots must be positive; omit `max_slots` to use the inflight default of {DEFAULT_MAX_SLOTS_UNDER_INFLIGHT}"
        ));
    }
    Ok(EngineMode::SlotAware { max_slots })
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
        };
        format!("{setting}: {error}")
    })?;
    Ok(ResolvedKvBudget {
        bytes: (bytes != 0).then_some(bytes),
        origin: Some(origin),
    })
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

pub(crate) fn resolve_serve_behavior(
    cli: &cli::ServeBehaviorArgs,
    defaults: Option<&ServeDefaultsV2>,
) -> Result<ResolvedServeBehavior, String> {
    let repetition_penalty = cli
        .default_repetition_penalty
        .or_else(|| {
            defaults.and_then(|defaults| defaults.repetition_penalty.map(|value| value as f32))
        })
        .unwrap_or(1.0);
    if !repetition_penalty.is_finite() || repetition_penalty <= 0.0 {
        return Err("--default-repetition-penalty must be a finite positive number".to_owned());
    }
    // Preserve an explicit zero through the typed configuration. Qwen's
    // policy treats a zero base as disabled and, importantly, a zero tool
    // override as "do not derive a continuation budget from the base".
    // Collapsing zero to `None` here would lose that distinction.
    let thinking_token_budget = cli
        .default_thinking_token_budget
        .or_else(|| defaults.and_then(|defaults| defaults.thinking_token_budget));
    let tool_thinking_token_budget = cli
        .default_tool_thinking_token_budget
        .or_else(|| defaults.and_then(|defaults| defaults.tool_thinking_token_budget));
    Ok(ResolvedServeBehavior {
        repetition_penalty,
        thinking_token_budget,
        tool_thinking_token_budget,
    })
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
    fn byte_budget_accepts_human_units_and_legacy_bare_bytes() {
        assert_eq!(parse_byte_size("8GiB").unwrap(), 8 * (1u64 << 30));
        assert_eq!(parse_byte_size(" 2 GB ").unwrap(), 2_000_000_000);
        assert_eq!(parse_byte_size("48000").unwrap(), 48_000);
        assert_eq!(parse_byte_size("0").unwrap(), 0);
        assert!(parse_byte_size("8GiBits").is_err());
        assert!(parse_byte_size("-1GiB").is_err());
    }

    #[test]
    fn behavior_cli_overrides_config_and_zero_disables_optional_budgets() {
        let config = crate::setup::OperatorConfigV2::guide_defaults().unwrap();
        let resolved = resolve_serve_behavior(
            &cli::ServeBehaviorArgs {
                default_repetition_penalty: Some(1.1),
                default_thinking_token_budget: Some(0),
                default_tool_thinking_token_budget: Some(64),
            },
            Some(&config.serve),
        )
        .unwrap();
        assert_eq!(resolved.repetition_penalty, 1.1);
        assert_eq!(resolved.thinking_token_budget, Some(0));
        assert_eq!(resolved.tool_thinking_token_budget, Some(64));

        let zero_tool = resolve_serve_behavior(
            &cli::ServeBehaviorArgs {
                default_repetition_penalty: None,
                default_thinking_token_budget: None,
                default_tool_thinking_token_budget: Some(0),
            },
            Some(&config.serve),
        )
        .unwrap();
        assert_eq!(zero_tool.tool_thinking_token_budget, Some(0));
    }

    #[test]
    fn behavior_rejects_invalid_repetition_penalty() {
        let error = resolve_serve_behavior(
            &cli::ServeBehaviorArgs {
                default_repetition_penalty: Some(f32::NAN),
                default_thinking_token_budget: None,
                default_tool_thinking_token_budget: None,
            },
            None,
        )
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
        };
        let error = resolve_scheduler(&planning, None).unwrap_err();
        assert!(error.contains("inflight-batched"), "{error}");
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
