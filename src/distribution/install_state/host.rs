#[cfg(target_os = "macos")]
use std::ffi::CString;

use super::schema::MacOsVersion;
use super::InstallStateError;

pub(super) fn require_compatible_host(required: &MacOsVersion) -> Result<(), InstallStateError> {
    let floor = MacOsVersion::parse("minimum_macos", "14.0".to_owned())
        .expect("the v1 macOS floor is valid");
    let host = host_macos_version()?;
    require_version_supported(&host, &floor, required)
}

fn require_version_supported(
    host: &MacOsVersion,
    floor: &MacOsVersion,
    required: &MacOsVersion,
) -> Result<(), InstallStateError> {
    if required < floor {
        return Err(InstallStateError::InvalidLayout(
            "release manifest predates the v1 macOS deployment floor",
        ));
    }
    if host < required {
        return Err(InstallStateError::InvalidLayout(
            "host macOS version is below the release requirement",
        ));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn host_macos_version() -> Result<MacOsVersion, InstallStateError> {
    let name = CString::new("kern.osproductversion").expect("static sysctl name");
    let mut length = 0_usize;
    // SAFETY: `name` is NUL-terminated, the first call supplies no output
    // buffer as required by sysctlbyname, and `length` is a valid out pointer.
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            std::ptr::null_mut(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 || length == 0 || length > 64 {
        return Err(InstallStateError::std_io(
            "read host macOS version size",
            std::io::Error::last_os_error(),
        ));
    }
    let mut bytes = vec![0_u8; length];
    // SAFETY: `bytes` owns `length` writable bytes and the remaining pointers
    // follow sysctlbyname's read-only query contract.
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            bytes.as_mut_ptr().cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 || length == 0 || length > bytes.len() {
        return Err(InstallStateError::std_io(
            "read host macOS version",
            std::io::Error::last_os_error(),
        ));
    }
    bytes.truncate(length);
    if bytes.last() == Some(&0) {
        bytes.pop();
    }
    let text = std::str::from_utf8(&bytes).map_err(|_| {
        InstallStateError::InvalidLayout("host macOS version is not canonical UTF-8")
    })?;
    MacOsVersion::parse("host_macos", text.to_owned())
        .map_err(|_| InstallStateError::InvalidLayout("host macOS version is not canonical"))
}

#[cfg(not(target_os = "macos"))]
fn host_macos_version() -> Result<MacOsVersion, InstallStateError> {
    Err(InstallStateError::InvalidLayout(
        "standalone installation requires macOS",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn version(value: &str) -> MacOsVersion {
        MacOsVersion::parse("test", value.to_owned()).expect("test macOS version")
    }

    #[test]
    fn host_must_satisfy_floor_and_release_requirement() {
        let floor = version("14.0");
        assert!(require_version_supported(&version("14.0"), &floor, &version("14.0")).is_ok());
        assert!(require_version_supported(&version("14.9"), &floor, &version("14.10")).is_err());
        assert!(require_version_supported(&version("26.5.2"), &floor, &version("99.0")).is_err());
        assert!(require_version_supported(&version("26.5.2"), &floor, &version("13.0")).is_err());
    }
}
