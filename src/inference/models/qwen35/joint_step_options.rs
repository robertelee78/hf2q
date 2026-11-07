//! Per-step execution choices of a Qwen joint decode step and their
//! switches.
//!
//! The serving worker parses the switches once at start
//! (`qwen35_joint_decode_config`); `forward_gpu_decode_joint_with_options`
//! takes the options explicitly, so one process can run several (the Q1
//! trace).

/// Switch for [`JointStepOptions::pooled_outputs`]: `1`/`on`/`true` takes the
/// joint layer chain's multi-row outputs from the decode pool
/// ([`super::decode_pool::StepPooledOutputs`]); unset, blank or
/// `0`/`off`/`false` keeps fresh device allocations. Either mode.
pub(crate) const QWEN_JOINT_POOLED_OUTPUTS_ENV: &str = "HF2Q_QWEN_JOINT_POOLED_OUTPUTS";

/// Per-step execution choices of a joint step that its route mode does not
/// fix. The default turns every choice off.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct JointStepOptions {
    /// Multi-row layer outputs from the decode pool (Class A).
    pub pooled_outputs: bool,
}

/// Parse an on/off switch named `name`; unset or blank means off.
pub(crate) fn parse_switch(name: &str, raw: Option<&str>) -> Result<bool, String> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(false);
    };
    match raw.to_ascii_lowercase().as_str() {
        "1" | "on" | "true" => Ok(true),
        "0" | "off" | "false" => Ok(false),
        _ => Err(format!(
            "{name} must be 1/on/true or 0/off/false, got {raw:?}"
        )),
    }
}

/// [`parse_switch`] over the process environment.
pub(crate) fn switch_from_process(name: &str) -> Result<bool, String> {
    parse_switch(name, std::env::var(name).ok().as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switch_parses_on_off_and_rejects_other_values() {
        let parse = |raw| parse_switch(QWEN_JOINT_POOLED_OUTPUTS_ENV, raw);
        assert_eq!(parse(None), Ok(false));
        assert_eq!(parse(Some("  ")), Ok(false));
        for on in ["1", "on", "TRUE"] {
            assert_eq!(parse(Some(on)), Ok(true));
        }
        for off in ["0", "off", "False"] {
            assert_eq!(parse(Some(off)), Ok(false));
        }
        assert!(parse(Some("pooled")).is_err());
    }
}
