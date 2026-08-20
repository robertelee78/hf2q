use serde::{Deserialize, Serialize};

use super::SetupError;

pub(super) const MAX_CONFIG_BYTES: usize = 16 * 1024;
const CONFIG_KIND: &str = "hf2q.config";
const CONFIG_SCHEMA_VERSION: i64 = 2;
const PACKAGE: &str = "hf2q";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ServeDefaultsV2 {
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) scheduler: ConfiguredScheduler,
    pub(crate) max_slots: u32,
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
        Self::new(
            ConvertDefaultsV2 {
                quant: "q4_k_m".to_owned(),
            },
            ServeDefaultsV2 {
                host: "127.0.0.1".to_owned(),
                port: 8081,
                scheduler: ConfiguredScheduler::InflightBatched,
                max_slots: 1,
            },
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
