mod fs;
mod host;
mod policy;
mod schema;

use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::cli::SetupArgs;

use self::host::{HostObservation, HostProbe, LiveHostProbe};
use self::policy::{resolve_preferences, PreferenceResolution};
pub(crate) use self::schema::{OperatorConfigV2, ServeDefaultsV2};

#[derive(Debug, Clone)]
pub(crate) struct ConfigPurgePlan {
    pub(crate) root: PathBuf,
    pub(crate) paths: [PathBuf; 3],
}

#[derive(Debug, Error)]
pub(crate) enum SetupError {
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

pub(super) fn run(args: SetupArgs, explicit_root: Option<&Path>) -> Result<(), SetupError> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let interactive = stdin.is_terminal() && stdout.is_terminal();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    execute(
        args,
        explicit_root,
        &LiveHostProbe,
        interactive,
        &mut input,
        &mut output,
    )
}

pub(crate) fn load_operator_config(
    explicit_root: Option<&Path>,
) -> Result<Option<OperatorConfigV2>, SetupError> {
    let root = resolve_root(explicit_root)?;
    fs::load_config_if_present(&root)
}

pub(crate) fn prepare_config_purge(
    explicit_root: Option<&Path>,
) -> Result<ConfigPurgePlan, SetupError> {
    let root = resolve_root(explicit_root)?;
    fs::validate_purge_target(&root)?;
    Ok(ConfigPurgePlan {
        paths: [
            root.join("config.toml"),
            root.join(".config.toml.partial"),
            root.join(".config.toml.lock"),
        ],
        root,
    })
}

pub(crate) fn execute_config_purge(plan: &ConfigPurgePlan) -> Result<Vec<PathBuf>, SetupError> {
    fs::validate_purge_target(&plan.root)?;
    let removed = fs::purge_config(&plan.root)?;
    Ok(removed
        .into_iter()
        .map(|name| plan.root.join(name))
        .collect())
}

fn execute<P: HostProbe, R: BufRead, W: Write>(
    args: SetupArgs,
    explicit_root: Option<&Path>,
    probe: &P,
    interactive: bool,
    input: &mut R,
    output: &mut W,
) -> Result<(), SetupError> {
    let root = resolve_root(explicit_root)?;
    let observed = fs::observe_existing_config(&root)?;
    let current = observed.config();
    let HostObservation {
        hardware,
        macos_version,
        configured_shell,
        performance_levels,
        logical_cores,
        open_file_soft_limit,
        volume_total_bytes,
        volume_available_bytes,
    } = probe.observe(&root)?;
    hardware.validate()?;
    host::validate_performance_levels(&performance_levels, logical_cores)?;
    let performance_summary = performance_levels
        .iter()
        .map(|level| format!("{} {}", level.logical_cores, level.name))
        .collect::<Vec<_>>()
        .join(" + ");

    writeln!(
        output,
        "Detected {} on {}: {} / {} with {} bytes unified memory and a {}-byte Metal recommended working set (macOS {}, {} logical cores, configured shell {}, RLIMIT_NOFILE {}).",
        hardware.target,
        root.display(),
        hardware.chip_model,
        hardware.metal_device_name,
        hardware.unified_memory_bytes,
        hardware.metal_recommended_working_set_bytes,
        macos_version,
        performance_summary,
        configured_shell.as_str(),
        open_file_soft_limit,
    )?;
    writeln!(
        output,
        "Containing volume: {} bytes total, {} bytes available.",
        volume_total_bytes, volume_available_bytes,
    )?;
    writeln!(
        output,
        "Recommended defaults follow the tested Qwen3.8 coding-client journey; every value remains overrideable per command."
    )?;

    let config = match resolve_preferences(&args, current, interactive, input, output)? {
        PreferenceResolution::Cancelled => {
            writeln!(output, "Setup cancelled; no configuration was changed.")?;
            return Ok(());
        }
        PreferenceResolution::Selected(config) => config,
    };
    let bytes = config.to_canonical_bytes()?;
    let changed = fs::persist(&root, &config, &bytes, &observed)?;
    writeln!(
        output,
        "{} {}",
        if changed { "Configured" } else { "Verified" },
        root.join("config.toml").display()
    )?;
    writeln!(output, "Default convert quant: {}", config.convert.quant)?;
    writeln!(
        output,
        "Default serve endpoint: {}:{}",
        config.serve.host, config.serve.port
    )?;
    writeln!(
        output,
        "Default serve scheduler: {} (max slots {}).",
        config.serve.scheduler.as_str(),
        config.serve.max_slots
    )?;
    if config.serve.host == "0.0.0.0" {
        writeln!(
            output,
            "LAN serving requires --auth-token or HF2Q_AUTH_TOKEN before hf2q will bind."
        )?;
    }
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
        return Err(SetupError::Input(
            "hf2q state root must be absolute".to_owned(),
        ));
    }
    Ok(root)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod defaults_contract_tests;
