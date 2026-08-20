fn enforce_free_floor(
    directory: &Directory,
    object_bytes: u64,
    catalog_bytes: usize,
) -> Result<(), ManagedSessionCacheError> {
    #[cfg(test)]
    let injected = TEST_VOLUME_SPACE.with(std::cell::Cell::get);
    #[cfg(not(test))]
    let injected: Option<(u64, u64, u64)> = None;
    let (volume, available, fragment) = if let Some((volume, available, fragment)) = injected {
        (volume, available, fragment.max(1))
    } else {
        let stat = unix::volume_space(directory)?;
        let fragment = stat.f_frsize.max(1);
        (
            stat.f_blocks
                .checked_mul(fragment)
                .ok_or(ManagedSessionCacheError::FreeSpaceFloor)?,
            stat.f_bavail
                .checked_mul(fragment)
                .ok_or(ManagedSessionCacheError::FreeSpaceFloor)?,
            fragment,
        )
    };
    let fifteen_percent = volume
        .checked_mul(15)
        .and_then(|value| value.checked_add(99))
        .map(|value| value / 100)
        .ok_or(ManagedSessionCacheError::FreeSpaceFloor)?;
    let reserve = MIN_FREE_BYTES.max(fifteen_percent);
    let required = reserve
        .checked_add(round_up(object_bytes, fragment)?)
        .and_then(|value| value.checked_add(round_up(catalog_bytes as u64, fragment).ok()?))
        .and_then(|value| value.checked_add(fragment.saturating_mul(METADATA_BLOCK_MARGIN)))
        .ok_or(ManagedSessionCacheError::FreeSpaceFloor)?;
    if available < required {
        return Err(ManagedSessionCacheError::FreeSpaceFloor);
    }
    Ok(())
}

fn round_up(value: u64, unit: u64) -> Result<u64, ManagedSessionCacheError> {
    value
        .checked_add(unit - 1)
        .map(|rounded| rounded / unit * unit)
        .ok_or(ManagedSessionCacheError::QuotaExceeded)
}

#[cfg(test)]
fn note_test_space_reclaimed(bytes: u64) {
    TEST_VOLUME_SPACE.with(|slot| {
        if let Some((volume, available, fragment)) = slot.get() {
            slot.set(Some((volume, available.saturating_add(bytes), fragment)));
        }
    });
}

#[cfg(not(test))]
fn note_test_space_reclaimed(_bytes: u64) {}

fn is_lower_hex(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_quarantine_name(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".checkpoint") else {
        return false;
    };
    let mut parts = stem.splitn(3, '-');
    let Some(sequence) = parts.next() else {
        return false;
    };
    let Some(reason) = parts.next() else {
        return false;
    };
    let Some(digest) = parts.next() else {
        return false;
    };
    sequence.len() == 20
        && sequence.bytes().all(|byte| byte.is_ascii_digit())
        && !reason.is_empty()
        && reason.len() <= 32
        && reason
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte == b'_')
        && digest.len() == 64
        && is_lower_hex(digest)
}

fn std_io(operation: &'static str, error: std::io::Error) -> ManagedSessionCacheError {
    match error.raw_os_error() {
        Some(libc::ENOSPC) | Some(libc::EDQUOT) => ManagedSessionCacheError::StorageFull,
        _ => ManagedSessionCacheError::Filesystem(format!("{operation}: {error}")),
    }
}

#[cfg(test)]
pub(super) fn test_std_io_error_mapping(error: std::io::Error) -> ManagedSessionCacheError {
    std_io("test managed cache std I/O error", error)
}
