use serde::{Deserialize, Serialize};

use super::SetupError;

pub(super) const MAX_CONFIG_BYTES: usize = 16 * 1024;
const CONFIG_KIND: &str = "hf2q.config";
const CONFIG_SCHEMA_VERSION: i64 = 2;
const PACKAGE: &str = "hf2q";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OperatorConfigV2 {
    kind: String,
    schema_version: u32,
    package: String,
    pub(crate) convert: ConvertDefaultsV2,
    pub(crate) serve: ServeDefaultsV2,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ConvertDefaultsV2 {
    pub(crate) quant: String,
}

/// ADR-062 D1 (2026-10-08): `hf2q setup` no longer writes behavior profile
/// keys. The former `[serve]` `repetition_penalty`, `thinking_token_budget`,
/// and `tool_thinking_token_budget` keys are retired — parse warns and
/// ignores them, and the per-family built-in table in
/// `src/serve/operator_settings.rs` (plus the CLI `--default-*` flags) is the
/// operator surface for those values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ServeDefaultsV2 {
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) scheduler: ConfiguredScheduler,
    pub(crate) max_slots: u32,
    /// Optional per-slot logical context cap. Omission means each model uses
    /// the maximum declared by its GGUF metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) ctx: Option<u32>,
    /// Optional aggregate physical KV-cache residency ceiling. Human-readable
    /// units such as `8GiB` are accepted; omission means no explicit ceiling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) kv_cache_budget: Option<String>,
    /// Optional on-disk ceiling for persistent KV data. Human-readable units
    /// such as `32GiB` are accepted; omission/zero means no explicit ceiling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) kv_persist_budget: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ConfiguredScheduler {
    FifoSerial,
    InflightBatched,
}

/// Scheduler the guide profile records, and the scheduler `serve` uses when
/// neither the CLI nor `config.toml` names one. A fresh install must serve
/// long agent prompts, which only the resumable inflight-batched prefill can.
pub(crate) const GUIDE_SCHEDULER: ConfiguredScheduler = ConfiguredScheduler::InflightBatched;

/// Active-slot count paired with [`GUIDE_SCHEDULER`], and the inflight
/// default whenever no slot count is configured. Idle slots reserve their
/// full-attention KV lazily; agent harnesses send side requests (titles,
/// summaries, sub-agents) alongside the main turn, so they run concurrently
/// instead of queueing.
pub(crate) const GUIDE_MAX_SLOTS: u32 = 4;

impl ConfiguredScheduler {
    pub(crate) const fn as_cli(self) -> crate::cli::SchedulerArg {
        match self {
            Self::FifoSerial => crate::cli::SchedulerArg::FifoSerial,
            Self::InflightBatched => crate::cli::SchedulerArg::InflightBatched,
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::FifoSerial => "fifo_serial",
            Self::InflightBatched => "inflight_batched",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ConfiguredShell {
    Bash,
    Fish,
    Zsh,
    Other,
}

impl ConfiguredShell {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Bash => "bash",
            Self::Fish => "fish",
            Self::Zsh => "zsh",
            Self::Other => "other",
        }
    }
}

impl OperatorConfigV2 {
    pub(crate) fn new(
        convert: ConvertDefaultsV2,
        serve: ServeDefaultsV2,
    ) -> Result<Self, SetupError> {
        let config = Self {
            kind: CONFIG_KIND.to_owned(),
            schema_version: CONFIG_SCHEMA_VERSION as u32,
            package: PACKAGE.to_owned(),
            convert,
            serve,
        };
        config.validate()?;
        Ok(config)
    }

    pub(crate) fn guide_defaults() -> Result<Self, SetupError> {
        let serve = ServeDefaultsV2 {
            host: "127.0.0.1".to_owned(),
            port: 8081,
            scheduler: GUIDE_SCHEDULER,
            max_slots: GUIDE_MAX_SLOTS,
            ctx: None,
            kv_cache_budget: None,
            kv_persist_budget: None,
        };
        Self::new(
            ConvertDefaultsV2 {
                quant: "q4_k_m".to_owned(),
            },
            serve,
        )
    }

    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, SetupError> {
        if bytes.is_empty() || bytes.len() > MAX_CONFIG_BYTES {
            return Err(SetupError::InvalidConfig(
                "config.toml is empty or exceeds 16 KiB".to_owned(),
            ));
        }
        let text = std::str::from_utf8(bytes)
            .map_err(|_| SetupError::InvalidConfig("config.toml is not UTF-8".to_owned()))?;
        let document: toml::Value = toml::from_str(text)
            .map_err(|error| SetupError::InvalidConfig(format!("invalid TOML: {error}")))?;
        reject_unsupported_schema(&document)?;
        let document = retire_serve_behavior_keys(document);
        let config: Self = document
            .try_into()
            .map_err(|error| SetupError::InvalidConfig(format!("invalid config.toml: {error}")))?;
        config.validate()?;
        Ok(config)
    }

    pub(crate) fn to_canonical_bytes(&self) -> Result<Vec<u8>, SetupError> {
        self.validate()?;
        let mut text = toml::to_string(self)
            .map_err(|error| SetupError::InvalidConfig(format!("cannot encode TOML: {error}")))?;
        if !text.ends_with('\n') {
            text.push('\n');
        }
        if text.len() > MAX_CONFIG_BYTES {
            return Err(SetupError::InvalidConfig(
                "encoded config.toml exceeds 16 KiB".to_owned(),
            ));
        }
        if Self::parse(text.as_bytes())? != *self {
            return Err(SetupError::InvalidConfig(
                "config.toml producer did not round-trip exactly".to_owned(),
            ));
        }
        Ok(text.into_bytes())
    }

    fn validate(&self) -> Result<(), SetupError> {
        if self.kind != CONFIG_KIND
            || self.schema_version != CONFIG_SCHEMA_VERSION as u32
            || self.package != PACKAGE
        {
            return Err(SetupError::InvalidConfig(
                "config identity or schema is unsupported".to_owned(),
            ));
        }
        self.convert.validate()?;
        self.serve.validate()
    }
}

impl ConvertDefaultsV2 {
    fn validate(&self) -> Result<(), SetupError> {
        crate::convert::QuantSelector::from_name(&self.quant)
            .map(|_| ())
            .map_err(|error| {
                SetupError::InvalidConfig(format!(
                    "convert.quant is not a supported hf2q quant selector: {error}"
                ))
            })
    }
}

impl ServeDefaultsV2 {
    fn validate(&self) -> Result<(), SetupError> {
        if !matches!(self.host.as_str(), "127.0.0.1" | "0.0.0.0") {
            return Err(SetupError::InvalidConfig(
                "serve.host must be 127.0.0.1 or 0.0.0.0".to_owned(),
            ));
        }
        if self.port == 0 {
            return Err(SetupError::InvalidConfig(
                "serve.port must be in 1..=65535".to_owned(),
            ));
        }
        if self.max_slots == 0 {
            return Err(SetupError::InvalidConfig(
                "serve.max_slots must be positive".to_owned(),
            ));
        }
        if self.scheduler == ConfiguredScheduler::FifoSerial && self.max_slots != 1 {
            return Err(SetupError::InvalidConfig(
                "serve.max_slots must be 1 for fifo_serial".to_owned(),
            ));
        }
        crate::serve::operator_settings::validate_max_slots(self.max_slots)
            .map_err(|error| SetupError::InvalidConfig(format!("serve.{error}")))?;
        if self.ctx == Some(0) {
            return Err(SetupError::InvalidConfig(
                "serve.ctx must be positive; omit it to use the GGUF maximum".to_owned(),
            ));
        }
        if let Some(value) = self.kv_cache_budget.as_deref() {
            crate::serve::operator_settings::parse_byte_size(value).map_err(|error| {
                SetupError::InvalidConfig(format!("serve.kv_cache_budget is invalid: {error}"))
            })?;
        }
        if let Some(value) = self.kv_persist_budget.as_deref() {
            crate::serve::operator_settings::parse_byte_size(value).map_err(|error| {
                SetupError::InvalidConfig(format!("serve.kv_persist_budget is invalid: {error}"))
            })?;
        }
        Ok(())
    }
}

/// ADR-062 D1: the former `[serve]` behavior profile keys are retired. A
/// config carrying them still parses — each retired key warns (naming its
/// CLI flag) and is ignored, so serving uses the per-family built-in values
/// unless the CLI `--default-*` flag overrides them.
fn retire_serve_behavior_keys(mut document: toml::Value) -> toml::Value {
    const RETIRED_SERVE_KEYS: [(&str, &str); 3] = [
        ("repetition_penalty", "--default-repetition-penalty"),
        ("thinking_token_budget", "--default-thinking-token-budget"),
        (
            "tool_thinking_token_budget",
            "--default-tool-thinking-token-budget",
        ),
    ];
    let Some(serve) = document
        .get_mut("serve")
        .and_then(toml::Value::as_table_mut)
    else {
        return document;
    };
    for (key, flag) in RETIRED_SERVE_KEYS {
        if let Some(value) = serve.remove(key) {
            eprintln!(
                "warning: config.toml `serve.{key} = {value}` is retired and ignored \
                 (ADR-062 D1); serving uses the family built-in, and \
                 `hf2q serve {flag}` remains the explicit override"
            );
        }
    }
    document
}

fn reject_unsupported_schema(document: &toml::Value) -> Result<(), SetupError> {
    let table = document.as_table().ok_or_else(|| {
        SetupError::InvalidConfig("config.toml must contain a TOML table".to_owned())
    })?;
    let kind = table.get("kind").and_then(toml::Value::as_str);
    let version = table
        .get("schema_version")
        .and_then(toml::Value::as_integer);
    if kind != Some(CONFIG_KIND) {
        return Err(SetupError::InvalidConfig(
            "config kind is unsupported".to_owned(),
        ));
    }
    match version {
        Some(CONFIG_SCHEMA_VERSION) => Ok(()),
        Some(1) => Err(SetupError::InvalidConfig(
            "provisional config schema 1 is no longer supported; move config.toml aside and rerun `hf2q setup`"
                .to_owned(),
        )),
        Some(other) => Err(SetupError::InvalidConfig(format!(
            "config schema {other} is unsupported; upgrade hf2q or rerun `hf2q setup`"
        ))),
        None => Err(SetupError::InvalidConfig(
            "config schema_version is missing or not an integer".to_owned(),
        )),
    }
}
