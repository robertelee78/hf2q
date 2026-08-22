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

/// Qualified agentic-coding serving profile (OpenCode acceptance host,
/// 2026-08-21). Persisted by `hf2q setup` when the operator optimizes for
/// long agent and tool-use prompts; applied by `hf2q serve` as process
/// defaults without overriding explicit operator environment.
pub(crate) const AGENTIC_PROFILE_REPETITION_PENALTY: f64 = 1.05;
pub(crate) const AGENTIC_PROFILE_THINKING_BUDGET: u32 = 2048;
pub(crate) const AGENTIC_PROFILE_TOOL_THINKING_BUDGET: u32 = 512;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ServeDefaultsV2 {
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) scheduler: ConfiguredScheduler,
    pub(crate) max_slots: u32,
    /// Server-wide repetition penalty for clients that omit the field
    /// (`HF2Q_DEFAULT_REPETITION_PENALTY`). `None` leaves the built-in
    /// default (1.0 = off).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) repetition_penalty: Option<f64>,
    /// Default Qwen thinking budget (`HF2Q_DEFAULT_THINKING_TOKEN_BUDGET`).
    /// `None` leaves thinking unbounded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) thinking_token_budget: Option<u32>,
    /// Tool-continuation thinking budget override
    /// (`HF2Q_DEFAULT_TOOL_THINKING_TOKEN_BUDGET`). `None` leaves the
    /// adaptive derivation from `thinking_token_budget`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tool_thinking_token_budget: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ConfiguredScheduler {
    FifoSerial,
    InflightBatched,
}

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
        let mut serve = ServeDefaultsV2 {
            host: "127.0.0.1".to_owned(),
            port: 8081,
            scheduler: ConfiguredScheduler::InflightBatched,
            max_slots: 1,
            repetition_penalty: None,
            thinking_token_budget: None,
            tool_thinking_token_budget: None,
        };
        // The guide journey optimizes for long agent and tool-use prompts
        // (the default setup answer), so the qualified profile is part of
        // the default configuration rather than an environment ritual.
        serve.apply_agentic_profile();
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
        reject_unsupported_schema(text)?;
        let config: Self = toml::from_str(text)
            .map_err(|error| SetupError::InvalidConfig(format!("invalid TOML: {error}")))?;
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
    /// Persist the qualified agentic-coding profile. Called when the
    /// operator answers yes to "Optimize serving for long agent and
    /// tool-use prompts?" (the default answer), so a plain `hf2q serve`
    /// picks up the same behavior the canonical OpenCode launchers export.
    pub(crate) fn apply_agentic_profile(&mut self) {
        self.repetition_penalty = Some(AGENTIC_PROFILE_REPETITION_PENALTY);
        self.thinking_token_budget = Some(AGENTIC_PROFILE_THINKING_BUDGET);
        self.tool_thinking_token_budget = Some(AGENTIC_PROFILE_TOOL_THINKING_BUDGET);
    }

    pub(crate) fn clear_agentic_profile(&mut self) {
        self.repetition_penalty = None;
        self.thinking_token_budget = None;
        self.tool_thinking_token_budget = None;
    }

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
        if let Some(penalty) = self.repetition_penalty {
            if !penalty.is_finite() || penalty <= 0.0 {
                return Err(SetupError::InvalidConfig(
                    "serve.repetition_penalty must be a finite positive number".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

fn reject_unsupported_schema(text: &str) -> Result<(), SetupError> {
    let document: toml::Value = toml::from_str(text)
        .map_err(|error| SetupError::InvalidConfig(format!("invalid TOML: {error}")))?;
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
