use std::path::{Path, PathBuf};

use sysinfo::Disks;

use super::{ModelPreparationError, VerifiedRecipeHost};
use crate::input::model_recipe::ModelRecipe;

impl ModelRecipe {
    /// Measure the current host and the filesystem that will hold prepared
    /// model state, then apply this exact recipe's closed hardware/disk policy.
    /// Caller-provided hardware facts cannot mint [`VerifiedRecipeHost`].
    pub fn verify_current_host_and_disk(
        &self,
        preparation_root: &Path,
    ) -> Result<VerifiedRecipeHost, ModelPreparationError> {
        let observation = HostObservation::read(preparation_root)?;
        self.verify_host_and_disk_facts(
            observation.target,
            &observation.chip_model,
            observation.total_unified_memory_bytes,
            observation.preflight_available_bytes,
        )
    }

    #[cfg(test)]
    pub(in crate::input) fn verify_host_and_disk(
        &self,
        target: &str,
        chip_model: &str,
        total_unified_memory_bytes: u64,
        available_bytes: u64,
    ) -> Result<VerifiedRecipeHost, ModelPreparationError> {
        self.verify_host_and_disk_facts(
            target,
            chip_model,
            total_unified_memory_bytes,
            available_bytes,
        )
    }

    #[cfg(test)]
    pub(in crate::input) fn available_bytes_for_path_for_test(
        path: &Path,
    ) -> Result<u64, ModelPreparationError> {
        available_bytes_for_path(path)
    }
}

struct HostObservation {
    target: &'static str,
    chip_model: String,
    total_unified_memory_bytes: u64,
    preflight_available_bytes: u64,
}

impl HostObservation {
    fn read(preparation_root: &Path) -> Result<Self, ModelPreparationError> {
        Ok(Self {
            target: compiled_target()?,
            chip_model: read_chip_model()?,
            total_unified_memory_bytes: read_total_unified_memory_bytes()?,
            preflight_available_bytes: available_bytes_for_path(preparation_root)?,
        })
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn compiled_target() -> Result<&'static str, ModelPreparationError> {
    Ok("aarch64-apple-darwin")
}

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
fn compiled_target() -> Result<&'static str, ModelPreparationError> {
    Err(host_error(format!(
        "unsupported compiled target {}-{}",
        std::env::consts::ARCH,
        std::env::consts::OS
    )))
}

#[cfg(target_os = "macos")]
fn read_chip_model() -> Result<String, ModelPreparationError> {
    let value = read_sysctl_string("machdep.cpu.brand_string")?;
    if value.is_empty() || value.len() > 128 || value.chars().any(char::is_control) {
        return Err(host_error("invalid machdep.cpu.brand_string"));
    }
    Ok(value)
}

#[cfg(not(target_os = "macos"))]
fn read_chip_model() -> Result<String, ModelPreparationError> {
    Err(host_error("chip model requires macOS sysctl"))
}

#[cfg(target_os = "macos")]
fn read_total_unified_memory_bytes() -> Result<u64, ModelPreparationError> {
    let value = read_sysctl_string("hw.memsize")?
        .parse::<u64>()
        .map_err(|_| host_error("hw.memsize is not an unsigned integer"))?;
    if value == 0 {
        return Err(host_error("hw.memsize is zero"));
    }
    Ok(value)
}

#[cfg(not(target_os = "macos"))]
fn read_total_unified_memory_bytes() -> Result<u64, ModelPreparationError> {
    Err(host_error("unified memory requires macOS sysctl"))
}

#[cfg(target_os = "macos")]
fn read_sysctl_string(name: &'static str) -> Result<String, ModelPreparationError> {
    use sysctl::Sysctl;

    let control = sysctl::Ctl::new(name)
        .map_err(|error| host_error(format!("cannot open {name}: {error}")))?;
    let value = control
        .value_string()
        .map_err(|error| host_error(format!("cannot read {name}: {error}")))?;
    Ok(value.trim().to_owned())
}

fn available_bytes_for_path(path: &Path) -> Result<u64, ModelPreparationError> {
    let existing = nearest_existing_directory(path)?;
    let disks = Disks::new_with_refreshed_list();
    let disk = disks
        .list()
        .iter()
        .filter(|disk| existing.starts_with(disk.mount_point()))
        .max_by_key(|disk| disk.mount_point().as_os_str().len())
        .ok_or_else(|| {
            host_error(format!(
                "no mounted filesystem contains {}",
                existing.display()
            ))
        })?;
    let available = disk.available_space();
    if available == 0 {
        return Err(host_error(format!(
            "filesystem containing {} reports zero available bytes",
            existing.display()
        )));
    }
    Ok(available)
}

fn nearest_existing_directory(path: &Path) -> Result<PathBuf, ModelPreparationError> {
    let mut candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| host_error(format!("cannot read current directory: {error}")))?
            .join(path)
    };
    loop {
        match candidate.metadata() {
            Ok(metadata) if metadata.is_dir() => {
                return candidate.canonicalize().map_err(|error| {
                    host_error(format!(
                        "cannot canonicalize preparation ancestor {}: {error}",
                        candidate.display()
                    ))
                });
            }
            Ok(_) => {
                return Err(host_error(format!(
                    "preparation ancestor {} is not a directory",
                    candidate.display()
                )));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if !candidate.pop() {
                    return Err(host_error("preparation path has no existing ancestor"));
                }
            }
            Err(error) => {
                return Err(host_error(format!(
                    "cannot inspect preparation ancestor {}: {error}",
                    candidate.display()
                )));
            }
        }
    }
}

fn host_error(reason: impl Into<String>) -> ModelPreparationError {
    ModelPreparationError::HostProbe {
        reason: reason.into(),
    }
}
