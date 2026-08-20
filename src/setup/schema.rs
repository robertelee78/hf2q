use serde::{Deserialize, Serialize};

use super::SetupError;

pub(super) const MAX_CONFIG_BYTES: usize = 16 * 1024;
const CONFIG_KIND: &str = "hf2q.config";
const CONFIG_SCHEMA_VERSION: u32 = 1;
const STATE_LAYOUT_SCHEMA: u32 = 1;
const PACKAGE: &str = "hf2q";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ConfigV1 {
    kind: String,
    schema_version: u32,
    state_layout_schema: u32,
    package: String,
    pub(super) hardware: HardwareProfileV1,
    pub(super) session_cache: SessionCachePolicyV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct HardwareProfileV1 {
    pub(super) target: String,
    pub(super) chip_model: String,
    pub(super) unified_memory_bytes: u64,
    pub(super) metal_device_name: String,
    pub(super) metal_recommended_working_set_bytes: u64,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SessionCachePolicyV1 {
    /// Zero is the sole disabled encoding. Positive values are exact bytes.
    pub(super) limit_bytes: u64,
}

impl ConfigV1 {
    pub(super) fn new(
        hardware: HardwareProfileV1,
        session_cache: SessionCachePolicyV1,
    ) -> Result<Self, SetupError> {
        let config = Self {
            kind: CONFIG_KIND.to_owned(),
            schema_version: CONFIG_SCHEMA_VERSION,
            state_layout_schema: STATE_LAYOUT_SCHEMA,
            package: PACKAGE.to_owned(),
            hardware,
            session_cache,
        };
        config.validate()?;
        Ok(config)
    }

    pub(super) fn parse(bytes: &[u8]) -> Result<Self, SetupError> {
        if bytes.is_empty() || bytes.len() > MAX_CONFIG_BYTES {
            return Err(SetupError::InvalidConfig(
                "config.toml is empty or exceeds 16 KiB".to_owned(),
            ));
        }
        let text = std::str::from_utf8(bytes)
            .map_err(|_| SetupError::InvalidConfig("config.toml is not UTF-8".to_owned()))?;
        let config: Self = toml::from_str(text)
            .map_err(|error| SetupError::InvalidConfig(format!("invalid TOML: {error}")))?;
        config.validate()?;
        Ok(config)
    }

    pub(super) fn to_canonical_bytes(&self) -> Result<Vec<u8>, SetupError> {
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
            || self.schema_version != CONFIG_SCHEMA_VERSION
            || self.state_layout_schema != STATE_LAYOUT_SCHEMA
            || self.package != PACKAGE
        {
            return Err(SetupError::InvalidConfig(
                "config identity or schema is unsupported".to_owned(),
            ));
        }
        self.hardware.validate()?;
        self.session_cache.validate()
    }
}

impl HardwareProfileV1 {
    pub(super) fn validate(&self) -> Result<(), SetupError> {
        if self.target != "aarch64-apple-darwin" {
            return Err(SetupError::InvalidConfig(
                "hardware target must be aarch64-apple-darwin".to_owned(),
            ));
        }
        require_bounded_text("chip_model", &self.chip_model)?;
        require_bounded_text("metal_device_name", &self.metal_device_name)?;
        if self.unified_memory_bytes == 0
            || self.unified_memory_bytes > i64::MAX as u64
            || self.metal_recommended_working_set_bytes == 0
            || self.metal_recommended_working_set_bytes > i64::MAX as u64
        {
            return Err(SetupError::InvalidConfig(
                "hardware profile contains an invalid byte limit".to_owned(),
            ));
        }
        Ok(())
    }
}

impl SessionCachePolicyV1 {
    fn validate(&self) -> Result<(), SetupError> {
        if self.limit_bytes > i64::MAX as u64 {
            return Err(SetupError::InvalidConfig(
                "session cache limit exceeds TOML's signed integer range".to_owned(),
            ));
        }
        Ok(())
    }
}

fn require_bounded_text(field: &str, value: &str) -> Result<(), SetupError> {
    if value.is_empty() || value.len() > 128 || value.chars().any(char::is_control) {
        return Err(SetupError::InvalidConfig(format!(
            "{field} must be 1..=128 non-control UTF-8 bytes"
        )));
    }
    Ok(())
}
