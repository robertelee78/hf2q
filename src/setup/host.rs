use std::path::{Path, PathBuf};

use sysinfo::Disks;

use super::schema::ConfiguredShell;
use super::SetupError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct HardwareProfile {
    pub(super) target: String,
    pub(super) chip_model: String,
    pub(super) unified_memory_bytes: u64,
    pub(super) metal_device_name: String,
    pub(super) metal_recommended_working_set_bytes: u64,
}

impl HardwareProfile {
    pub(super) fn validate(&self) -> Result<(), SetupError> {
        if self.target != "aarch64-apple-darwin" {
            return Err(SetupError::Host(
                "hardware target must be aarch64-apple-darwin".to_owned(),
            ));
        }
        for (field, value) in [
            ("chip model", self.chip_model.as_str()),
            ("Metal device name", self.metal_device_name.as_str()),
        ] {
            if value.is_empty() || value.len() > 128 || value.chars().any(char::is_control) {
                return Err(SetupError::Host(format!(
                    "{field} must be 1..=128 non-control UTF-8 bytes"
                )));
            }
        }
        if self.unified_memory_bytes == 0
            || self.metal_recommended_working_set_bytes == 0
            || self.unified_memory_bytes > i64::MAX as u64
            || self.metal_recommended_working_set_bytes > i64::MAX as u64
        {
            return Err(SetupError::Host(
                "hardware inventory contains an invalid byte value".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PerformanceLevel {
    pub(super) name: String,
    pub(super) logical_cores: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct HostObservation {
    pub(super) hardware: HardwareProfile,
    pub(super) macos_version: String,
    pub(super) configured_shell: ConfiguredShell,
    pub(super) performance_levels: Vec<PerformanceLevel>,
    pub(super) logical_cores: u32,
    pub(super) open_file_soft_limit: u64,
    pub(super) volume_total_bytes: u64,
    pub(super) volume_available_bytes: u64,
}

pub(super) trait HostProbe {
    fn observe(&self, state_root: &Path) -> Result<HostObservation, SetupError>;
}

pub(super) struct LiveHostProbe;

impl HostProbe for LiveHostProbe {
    fn observe(&self, state_root: &Path) -> Result<HostObservation, SetupError> {
        let (volume_total_bytes, volume_available_bytes) = volume_facts(state_root)?;
        let mut observation = read_host_profile()?;
        observation.volume_total_bytes = volume_total_bytes;
        observation.volume_available_bytes = volume_available_bytes;
        Ok(observation)
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn read_host_profile() -> Result<HostObservation, SetupError> {
    use mlx_native::MlxDevice;

    let chip_model = read_sysctl_string("machdep.cpu.brand_string")?;
    let macos_version = read_sysctl_string("kern.osproductversion")?;
    require_supported_macos(&macos_version)?;
    let unified_memory_bytes = read_sysctl_u64("hw.memsize")?;
    let logical_cores = read_sysctl_u32("hw.logicalcpu_max")?;
    let performance_levels = read_performance_levels(logical_cores)?;
    let device = MlxDevice::new()
        .map_err(|error| SetupError::Host(format!("cannot open the Metal device: {error}")))?;
    let working_set = device.metal_device().recommended_max_working_set_size();
    if working_set == 0 {
        return Err(SetupError::Host(
            "Metal reported a zero recommended working-set size".to_owned(),
        ));
    }
    Ok(HostObservation {
        hardware: HardwareProfile {
            target: "aarch64-apple-darwin".to_owned(),
            chip_model,
            unified_memory_bytes,
            metal_device_name: device.name(),
            metal_recommended_working_set_bytes: working_set,
        },
        macos_version,
        configured_shell: configured_shell(),
        performance_levels,
        logical_cores,
        open_file_soft_limit: open_file_soft_limit()?,
        volume_total_bytes: 0,
        volume_available_bytes: 0,
    })
}

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
fn read_host_profile() -> Result<HostObservation, SetupError> {
    Err(SetupError::Host(
        "hf2q setup requires Apple Silicon macOS".to_owned(),
    ))
}

#[cfg(target_os = "macos")]
fn read_sysctl_string(name: &str) -> Result<String, SetupError> {
    use sysctl::Sysctl;

    let control = sysctl::Ctl::new(name)
        .map_err(|error| SetupError::Host(format!("cannot open {name}: {error}")))?;
    let value = control
        .value_string()
        .map_err(|error| SetupError::Host(format!("cannot read {name}: {error}")))?;
    let value = value.trim().to_owned();
    if value.is_empty() || value.len() > 128 || value.chars().any(char::is_control) {
        return Err(SetupError::Host(format!("{name} returned invalid text")));
    }
    Ok(value)
}

#[cfg(target_os = "macos")]
fn read_sysctl_u64(name: &str) -> Result<u64, SetupError> {
    let value = read_sysctl_string(name)?
        .parse::<u64>()
        .map_err(|_| SetupError::Host(format!("{name} is not an unsigned integer")))?;
    if value == 0 {
        return Err(SetupError::Host(format!("{name} is zero")));
    }
    Ok(value)
}

#[cfg(target_os = "macos")]
fn read_sysctl_u32(name: &str) -> Result<u32, SetupError> {
    u32::try_from(read_sysctl_u64(name)?)
        .map_err(|_| SetupError::Host(format!("{name} exceeds u32")))
}

#[cfg(target_os = "macos")]
fn read_performance_levels(logical_cores: u32) -> Result<Vec<PerformanceLevel>, SetupError> {
    const MAX_PERFORMANCE_LEVELS: usize = 8;

    let level_count = usize::try_from(read_sysctl_u32("hw.nperflevels")?)
        .map_err(|_| SetupError::Host("hw.nperflevels exceeds usize".to_owned()))?;
    if !(1..=MAX_PERFORMANCE_LEVELS).contains(&level_count) {
        return Err(SetupError::Host(format!(
            "hw.nperflevels must be 1..={MAX_PERFORMANCE_LEVELS}"
        )));
    }
    let mut levels = Vec::with_capacity(level_count);
    for index in 0..level_count {
        let name_key = format!("hw.perflevel{index}.name");
        let cores_key = format!("hw.perflevel{index}.logicalcpu_max");
        levels.push(PerformanceLevel {
            name: read_sysctl_string(&name_key)?,
            logical_cores: read_sysctl_u32(&cores_key)?,
        });
    }
    validate_performance_levels(&levels, logical_cores)?;
    Ok(levels)
}

pub(super) fn validate_performance_levels(
    levels: &[PerformanceLevel],
    logical_cores: u32,
) -> Result<(), SetupError> {
    if levels.is_empty() || levels.len() > 8 {
        return Err(SetupError::Host(
            "macOS must report 1..=8 named performance levels".to_owned(),
        ));
    }
    let mut total = 0_u32;
    for (index, level) in levels.iter().enumerate() {
        if level.name.is_empty()
            || level.name.len() > 128
            || level.name.chars().any(char::is_control)
            || level.logical_cores == 0
            || levels[..index].iter().any(|prior| prior.name == level.name)
        {
            return Err(SetupError::Host(
                "performance-level names must be unique valid text with positive core counts"
                    .to_owned(),
            ));
        }
        total = total
            .checked_add(level.logical_cores)
            .ok_or_else(|| SetupError::Host("performance-level core count overflow".to_owned()))?;
    }
    if total != logical_cores {
        return Err(SetupError::Host(
            "named performance-level core counts do not match hw.logicalcpu_max".to_owned(),
        ));
    }
    Ok(())
}

pub(super) fn require_supported_macos(value: &str) -> Result<(), SetupError> {
    let components: Vec<&str> = value.split('.').collect();
    if !(2..=3).contains(&components.len())
        || components.iter().any(|part| {
            part.is_empty()
                || (part.len() > 1 && part.starts_with('0'))
                || part.parse::<u16>().is_err()
        })
    {
        return Err(SetupError::Host("macOS version is malformed".to_owned()));
    }
    let major = components[0]
        .parse::<u16>()
        .map_err(|_| SetupError::Host("macOS version is malformed".to_owned()))?;
    if major < 14 {
        return Err(SetupError::Host(format!(
            "macOS 14.0 or newer is required; detected {value}"
        )));
    }
    Ok(())
}

fn configured_shell() -> ConfiguredShell {
    let shell = std::env::var_os("SHELL")
        .and_then(|path| PathBuf::from(path).file_name().map(|name| name.to_owned()))
        .and_then(|name| name.to_str().map(str::to_owned));
    match shell.as_deref() {
        Some("bash") => ConfiguredShell::Bash,
        Some("fish") => ConfiguredShell::Fish,
        Some("zsh") => ConfiguredShell::Zsh,
        _ => ConfiguredShell::Other,
    }
}

fn open_file_soft_limit() -> Result<u64, SetupError> {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    let result = unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) };
    if result != 0 || limit.rlim_cur == 0 {
        return Err(SetupError::Host(
            "cannot read a positive RLIMIT_NOFILE soft limit".to_owned(),
        ));
    }
    Ok(limit.rlim_cur)
}

fn volume_facts(path: &Path) -> Result<(u64, u64), SetupError> {
    let existing = nearest_existing_directory(path)?;
    let disks = Disks::new_with_refreshed_list();
    let disk = disks
        .list()
        .iter()
        .filter(|disk| existing.starts_with(disk.mount_point()))
        .max_by_key(|disk| disk.mount_point().as_os_str().len())
        .ok_or_else(|| {
            SetupError::Host(format!(
                "no mounted filesystem contains {}",
                existing.display()
            ))
        })?;
    let total = disk.total_space();
    let available = disk.available_space();
    if total == 0 || available > total {
        return Err(SetupError::Host(
            "filesystem reported invalid capacity facts".to_owned(),
        ));
    }
    Ok((total, available))
}

pub(super) fn nearest_existing_directory(path: &Path) -> Result<PathBuf, SetupError> {
    let mut candidate = path.to_path_buf();
    loop {
        match candidate.metadata() {
            Ok(metadata) if metadata.is_dir() => {
                return candidate.canonicalize().map_err(|error| {
                    SetupError::Host(format!(
                        "cannot canonicalize storage ancestor {}: {error}",
                        candidate.display()
                    ))
                });
            }
            Ok(_) => {
                return Err(SetupError::Host(format!(
                    "storage ancestor {} is not a directory",
                    candidate.display()
                )));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if !candidate.pop() {
                    return Err(SetupError::Host(
                        "state root has no existing directory ancestor".to_owned(),
                    ));
                }
            }
            Err(error) => {
                return Err(SetupError::Host(format!(
                    "cannot inspect storage ancestor {}: {error}",
                    candidate.display()
                )));
            }
        }
    }
}
