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
pub(super) struct HostObservation {
    pub(super) hardware: HardwareProfile,
    pub(super) macos_version: String,
    pub(super) configured_shell: ConfiguredShell,
    pub(super) performance_level0_name: String,
    pub(super) performance_level0_cores: u32,
    pub(super) performance_level1_name: String,
    pub(super) performance_level1_cores: u32,
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
    let performance_level0_name = read_sysctl_string("hw.perflevel0.name")?;
    let performance_level0_cores = read_sysctl_u32("hw.perflevel0.logicalcpu_max")?;
    let performance_level1_name = read_sysctl_string("hw.perflevel1.name")?;
    let performance_level1_cores = read_sysctl_u32("hw.perflevel1.logicalcpu_max")?;
    let logical_cores = read_sysctl_u32("hw.logicalcpu_max")?;
    if performance_level0_name == performance_level1_name
        || performance_level0_cores
            .checked_add(performance_level1_cores)
            .filter(|total| *total == logical_cores)
            .is_none()
    {
        return Err(SetupError::Host(
            "named performance-level core counts do not match hw.logicalcpu_max".to_owned(),
        ));
    }
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
        performance_level0_name,
        performance_level0_cores,
        performance_level1_name,
        performance_level1_cores,
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
fn read_sysctl_string(name: &'static str) -> Result<String, SetupError> {
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
fn read_sysctl_u64(name: &'static str) -> Result<u64, SetupError> {
    let value = read_sysctl_string(name)?
        .parse::<u64>()
        .map_err(|_| SetupError::Host(format!("{name} is not an unsigned integer")))?;
    if value == 0 {
        return Err(SetupError::Host(format!("{name} is zero")));
    }
    Ok(value)
}

#[cfg(target_os = "macos")]
fn read_sysctl_u32(name: &'static str) -> Result<u32, SetupError> {
    u32::try_from(read_sysctl_u64(name)?)
        .map_err(|_| SetupError::Host(format!("{name} exceeds u32")))
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
