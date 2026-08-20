mod fs;
mod host;
mod policy;
mod schema;

use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::cli::SetupArgs;

use self::host::{HostObservation, HostProbe, LiveHostProbe};
use self::policy::{format_bytes, recommended_limit, resolve_policy, PolicyResolution};
use self::schema::ConfigV1;

#[derive(Debug, Error)]
pub(super) enum SetupError {
    #[error("setup input: {0}")]
    Input(String),
    #[error("setup host inventory: {0}")]
    Host(String),
    #[error("setup configuration: {0}")]
    InvalidConfig(String),
    #[error("setup filesystem: {0}")]
    Filesystem(String),
    #[error("another hf2q setup is already running")]
    Busy,
    #[error("config.toml may have committed but durability could not be proven: {0}")]
    DurabilityUnknown(String),
    #[error("required setup state is missing")]
    Missing,
    #[error("setup I/O: {0}")]
    Io(#[from] std::io::Error),
}

impl SetupError {
    pub(super) fn is_input(&self) -> bool {
        matches!(
            self,
            Self::Input(_) | Self::Host(_) | Self::InvalidConfig(_)
        )
    }
}

pub(super) fn run(args: SetupArgs) -> Result<(), SetupError> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let interactive = stdin.is_terminal() && stdout.is_terminal();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    execute(args, &LiveHostProbe, interactive, &mut input, &mut output)
}

fn execute<P: HostProbe, R: BufRead, W: Write>(
    args: SetupArgs,
    probe: &P,
    interactive: bool,
    input: &mut R,
    output: &mut W,
) -> Result<(), SetupError> {
    let root = resolve_root(args.state_root.as_deref())?;
    let state_binding = crate::distribution::verify_setup_state_root(&root)
        .map_err(|error| SetupError::Filesystem(error.to_string()))?;
    let observed = fs::observe_existing_config(&root)?;
    let current = observed.config();
    let HostObservation {
        hardware,
        macos_version,
        configured_shell,
        performance_level0_name,
        performance_level0_cores,
        performance_level1_name,
        performance_level1_cores,
        open_file_soft_limit,
        volume_total_bytes,
        volume_available_bytes,
    } = probe.observe(&root)?;
    hardware.validate()?;
    let recommendation = recommended_limit(volume_total_bytes, volume_available_bytes)?;

    writeln!(
        output,
        "Detected {} on {}: {} / {} with {} bytes unified memory and a {}-byte Metal recommended working set (macOS {}, {} {} + {} {} logical cores, shell {}, RLIMIT_NOFILE {}).",
        hardware.target,
        root.display(),
        hardware.chip_model,
        hardware.metal_device_name,
        hardware.unified_memory_bytes,
        hardware.metal_recommended_working_set_bytes,
        macos_version,
        performance_level0_cores,
        performance_level0_name,
        performance_level1_cores,
        performance_level1_name,
        configured_shell.as_str(),
        open_file_soft_limit,
    )?;
    writeln!(
        output,
        "Containing volume: {} bytes total, {} bytes available; recommended session-cache limit {} bytes ({}).",
        volume_total_bytes,
        volume_available_bytes,
        recommendation,
        format_bytes(recommendation),
    )?;

    let session_cache = match resolve_policy(
        &args,
        current.map(|config| &config.session_cache),
        interactive,
        recommendation,
        input,
        output,
    )? {
        PolicyResolution::Cancelled => {
            writeln!(output, "Setup cancelled; no configuration was changed.")?;
            return Ok(());
        }
        PolicyResolution::Selected(policy) => policy,
    };
    let config = ConfigV1::new(hardware, session_cache)?;
    let bytes = config.to_canonical_bytes()?;
    let changed = fs::persist(&root, &config, &bytes, &observed, &state_binding)?;
    writeln!(
        output,
        "{} {}",
        if changed { "Configured" } else { "Verified" },
        root.join("config.toml").display()
    )?;
    writeln!(
        output,
        "Recorded session-cache policy: {} ({} bytes).",
        if config.session_cache.limit_bytes == 0 {
            "disabled"
        } else {
            "enabled"
        },
        config.session_cache.limit_bytes
    )?;
    writeln!(
        output,
        "No model was downloaded, converted, loaded, or served."
    )?;
    Ok(())
}

fn resolve_root(explicit: Option<&Path>) -> Result<PathBuf, SetupError> {
    let root = match explicit {
        Some(path) => path.to_owned(),
        None => {
            let home = std::env::var_os("HOME").ok_or_else(|| {
                SetupError::Input("HOME is unset; pass --state-root ABSOLUTE_PATH".to_owned())
            })?;
            if home.is_empty() {
                return Err(SetupError::Input(
                    "HOME is empty; pass --state-root ABSOLUTE_PATH".to_owned(),
                ));
            }
            PathBuf::from(home).join(".hf2q")
        }
    };
    if !root.is_absolute() {
        return Err(SetupError::Input("setup root must be absolute".to_owned()));
    }
    Ok(root)
}

#[cfg(test)]
mod tests;
