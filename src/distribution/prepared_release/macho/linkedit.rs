use super::{u32_at, MachOError, LC_DATA_IN_CODE, LC_FUNCTION_STARTS};

pub(super) fn verify_symtab(
    record: &[u8],
    file_length: u64,
    payloads: &mut Vec<(u64, u64)>,
) -> Result<u32, MachOError> {
    if record.len() != 24 {
        return Err(MachOError::Invalid);
    }
    let symbol_count = u32_at(record, 12)?;
    push_optional_range(
        payloads,
        u32_at(record, 8)?,
        symbol_count.checked_mul(16).ok_or(MachOError::Invalid)?,
        file_length,
    )?;
    push_optional_range(
        payloads,
        u32_at(record, 16)?,
        u32_at(record, 20)?,
        file_length,
    )?;
    Ok(symbol_count)
}

pub(super) fn verify_dysymtab(
    record: &[u8],
    symbol_count: u32,
    file_length: u64,
    payloads: &mut Vec<(u64, u64)>,
) -> Result<(), MachOError> {
    if record.len() != 80 {
        return Err(MachOError::Invalid);
    }
    let local_start = u32_at(record, 8)?;
    let local_count = u32_at(record, 12)?;
    let external_start = u32_at(record, 16)?;
    let external_count = u32_at(record, 20)?;
    let undefined_start = u32_at(record, 24)?;
    let undefined_count = u32_at(record, 28)?;
    let local_end = local_start
        .checked_add(local_count)
        .ok_or(MachOError::Invalid)?;
    let external_end = external_start
        .checked_add(external_count)
        .ok_or(MachOError::Invalid)?;
    let undefined_end = undefined_start
        .checked_add(undefined_count)
        .ok_or(MachOError::Invalid)?;
    if local_start != 0
        || external_start != local_end
        || undefined_start != external_end
        || undefined_end != symbol_count
    {
        return Err(MachOError::Invalid);
    }
    for (offset, count, width) in [
        (32, 36, 8_u32),
        (40, 44, 56),
        (48, 52, 4),
        (56, 60, 4),
        (64, 68, 8),
        (72, 76, 8),
    ] {
        push_optional_range(
            payloads,
            u32_at(record, offset)?,
            u32_at(record, count)?
                .checked_mul(width)
                .ok_or(MachOError::Invalid)?,
            file_length,
        )?;
    }
    Ok(())
}

pub(super) fn verify_dyld_info(
    record: &[u8],
    file_length: u64,
    payloads: &mut Vec<(u64, u64)>,
) -> Result<(), MachOError> {
    if record.len() != 48 {
        return Err(MachOError::Invalid);
    }
    for offset in [8, 16, 24, 32, 40] {
        push_optional_range(
            payloads,
            u32_at(record, offset)?,
            u32_at(record, offset + 4)?,
            file_length,
        )?;
    }
    Ok(())
}

pub(super) fn verify_linkedit_data(
    record: &[u8],
    kind: u32,
    file_length: u64,
) -> Result<(u64, u64), MachOError> {
    if record.len() != 16 {
        return Err(MachOError::Invalid);
    }
    let size = u32_at(record, 12)?;
    if (kind == LC_FUNCTION_STARTS && size == 0) || (kind == LC_DATA_IN_CODE && size % 8 != 0) {
        return Err(MachOError::Invalid);
    }
    checked_file_range(u32_at(record, 8)?, size, file_length)
}

fn push_optional_range(
    payloads: &mut Vec<(u64, u64)>,
    offset: u32,
    size: u32,
    file_length: u64,
) -> Result<(), MachOError> {
    if size == 0 {
        if offset != 0 {
            return Err(MachOError::Invalid);
        }
        return Ok(());
    }
    payloads.push(checked_file_range(offset, size, file_length)?);
    Ok(())
}

fn checked_file_range(offset: u32, size: u32, file_length: u64) -> Result<(u64, u64), MachOError> {
    let start = u64::from(offset);
    let end = start
        .checked_add(u64::from(size))
        .ok_or(MachOError::Invalid)?;
    if end > file_length {
        return Err(MachOError::Invalid);
    }
    Ok((start, end))
}
