use std::collections::BTreeSet;

use super::{u32_at, u64_at, MachOError};

const VM_PROT_READ: u32 = 0x01;
const VM_PROT_WRITE: u32 = 0x02;
const VM_PROT_EXECUTE: u32 = 0x04;
const SECTION_TYPE: u32 = 0x0000_00ff;
const S_REGULAR: u32 = 0;
const S_ZEROFILL: u32 = 1;
const S_GB_ZEROFILL: u32 = 0x0c;
const S_THREAD_LOCAL_ZEROFILL: u32 = 0x12;
const S_ATTR_SOME_INSTRUCTIONS: u32 = 0x0000_0400;
const S_ATTR_PURE_INSTRUCTIONS: u32 = 0x8000_0000;
const TEXT_SECTION_FLAGS: u32 = S_REGULAR | S_ATTR_SOME_INSTRUCTIONS | S_ATTR_PURE_INSTRUCTIONS;
const SG_NORELOC: u32 = 0x04;
const SG_READ_ONLY: u32 = 0x10;
const ALLOWED_SEGMENT_FLAGS: u32 = SG_NORELOC | SG_READ_ONLY;

#[derive(Debug, Clone, Copy)]
pub(super) struct SegmentRange {
    vm_start: u64,
    vm_end: u64,
    file_start: u64,
    file_end: u64,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn verify_segment(
    record: &[u8],
    file_length: u64,
    commands_end: u64,
    executable_text_range: &mut Option<(u64, u64)>,
    executable_text_section: &mut Option<(u64, u64)>,
    linkedit_range: &mut Option<(u64, u64)>,
    segments: &mut Vec<SegmentRange>,
    segment_names: &mut BTreeSet<String>,
) -> Result<(), MachOError> {
    if record.len() < 72 {
        return Err(MachOError::Invalid);
    }
    let section_count = usize::try_from(u32_at(record, 64)?).map_err(|_| MachOError::Invalid)?;
    let expected_size = 72_usize
        .checked_add(section_count.checked_mul(80).ok_or(MachOError::Invalid)?)
        .ok_or(MachOError::Invalid)?;
    if record.len() != expected_size {
        return Err(MachOError::Invalid);
    }
    let name = fixed_name(record.get(8..24).ok_or(MachOError::Invalid)?)?;
    if !segment_names.insert(name.to_owned()) {
        return Err(MachOError::Invalid);
    }
    let file_offset = u64_at(record, 40)?;
    let file_size = u64_at(record, 48)?;
    let file_end = file_offset
        .checked_add(file_size)
        .ok_or(MachOError::Invalid)?;
    if file_end > file_length {
        return Err(MachOError::Invalid);
    }
    let maximum_protection = u32_at(record, 56)?;
    let initial_protection = u32_at(record, 60)?;
    if maximum_protection & !(VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE) != 0
        || initial_protection & !(VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE) != 0
        || initial_protection & !maximum_protection != 0
        || has_write_execute(maximum_protection)
        || has_write_execute(initial_protection)
        || (name != "__TEXT" && (maximum_protection | initial_protection) & VM_PROT_EXECUTE != 0)
        || u32_at(record, 68)? & !ALLOWED_SEGMENT_FLAGS != 0
    {
        return Err(MachOError::Invalid);
    }
    if name == "__TEXT" {
        if executable_text_range.is_some()
            || file_offset != 0
            || file_end < commands_end
            || maximum_protection != VM_PROT_READ | VM_PROT_EXECUTE
            || initial_protection != VM_PROT_READ | VM_PROT_EXECUTE
        {
            return Err(MachOError::Invalid);
        }
        *executable_text_range = Some((file_offset, file_end));
    } else if name == "__LINKEDIT" {
        if linkedit_range.is_some()
            || file_size == 0
            || file_offset < commands_end
            || file_end != file_length
            || maximum_protection != VM_PROT_READ
            || initial_protection != VM_PROT_READ
            || section_count != 0
        {
            return Err(MachOError::Invalid);
        }
        *linkedit_range = Some((file_offset, file_end));
    } else if file_size != 0 && file_offset < commands_end {
        return Err(MachOError::Invalid);
    }

    let vm_address = u64_at(record, 24)?;
    let vm_size = u64_at(record, 32)?;
    let vm_end = vm_address.checked_add(vm_size).ok_or(MachOError::Invalid)?;
    if file_size > vm_size {
        return Err(MachOError::Invalid);
    }
    let mut section_ranges = Vec::with_capacity(section_count);
    for index in 0..section_count {
        let section_offset = 72_usize
            .checked_add(index.checked_mul(80).ok_or(MachOError::Invalid)?)
            .ok_or(MachOError::Invalid)?;
        let section = record
            .get(section_offset..section_offset + 80)
            .ok_or(MachOError::Invalid)?;
        let section_name = fixed_name(&section[0..16])?;
        let section_segment = fixed_name(&section[16..32])?;
        if section_segment != name || u32_at(section, 52)? > 31 {
            return Err(MachOError::Invalid);
        }
        let section_address = u64_at(section, 32)?;
        let section_size = u64_at(section, 40)?;
        let section_vm_end = section_address
            .checked_add(section_size)
            .ok_or(MachOError::Invalid)?;
        if section_address < vm_address || section_vm_end > vm_end {
            return Err(MachOError::Invalid);
        }
        let section_flags = u32_at(section, 64)?;
        let section_type = section_flags & SECTION_TYPE;
        let section_file_offset = u64::from(u32_at(section, 48)?);
        let alignment = 1_u64
            .checked_shl(u32_at(section, 52)?)
            .ok_or(MachOError::Invalid)?;
        if section_address % alignment != 0
            || (!matches!(
                section_type,
                S_ZEROFILL | S_GB_ZEROFILL | S_THREAD_LOCAL_ZEROFILL
            ) && section_file_offset % alignment != 0)
        {
            return Err(MachOError::Invalid);
        }
        if matches!(
            section_type,
            S_ZEROFILL | S_GB_ZEROFILL | S_THREAD_LOCAL_ZEROFILL
        ) {
            if section_file_offset != 0 {
                return Err(MachOError::Invalid);
            }
            section_ranges.push(SegmentRange {
                vm_start: section_address,
                vm_end: section_vm_end,
                file_start: 0,
                file_end: 0,
            });
        } else {
            let section_file_end = section_file_offset
                .checked_add(section_size)
                .ok_or(MachOError::Invalid)?;
            if section_file_offset < file_offset || section_file_end > file_end {
                return Err(MachOError::Invalid);
            }
            if section_address
                .checked_sub(vm_address)
                .zip(section_file_offset.checked_sub(file_offset))
                .is_none_or(|(vm_delta, file_delta)| vm_delta != file_delta)
            {
                return Err(MachOError::Invalid);
            }
            if name == "__TEXT" && section_name == "__text" {
                if executable_text_section.is_some()
                    || section_flags != TEXT_SECTION_FLAGS
                    || section_size == 0
                    || u32_at(section, 68)? != 0
                    || u32_at(section, 72)? != 0
                    || u32_at(section, 76)? != 0
                {
                    return Err(MachOError::Invalid);
                }
                *executable_text_section = Some((section_file_offset, section_file_end));
            }
            section_ranges.push(SegmentRange {
                vm_start: section_address,
                vm_end: section_vm_end,
                file_start: section_file_offset,
                file_end: section_file_end,
            });
        }
        let relocation_offset = u64::from(u32_at(section, 56)?);
        let relocation_count = u32_at(section, 60)?;
        if relocation_offset != 0 || relocation_count != 0 {
            return Err(MachOError::Invalid);
        }
    }
    verify_disjoint_segments(&section_ranges)?;
    segments.push(SegmentRange {
        vm_start: vm_address,
        vm_end,
        file_start: file_offset,
        file_end,
    });
    Ok(())
}

pub(super) fn verify_disjoint_segments(ranges: &[SegmentRange]) -> Result<(), MachOError> {
    let mut vm_ranges = ranges
        .iter()
        .filter_map(|range| {
            (range.vm_start < range.vm_end).then_some((range.vm_start, range.vm_end))
        })
        .collect::<Vec<_>>();
    vm_ranges.sort_unstable();
    if vm_ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err(MachOError::Invalid);
    }

    let mut file_ranges = ranges
        .iter()
        .filter_map(|range| {
            (range.file_start < range.file_end).then_some((range.file_start, range.file_end))
        })
        .collect::<Vec<_>>();
    file_ranges.sort_unstable();
    if file_ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err(MachOError::Invalid);
    }
    Ok(())
}

fn fixed_name(bytes: &[u8]) -> Result<&str, MachOError> {
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    if bytes[end..].iter().any(|byte| *byte != 0) {
        return Err(MachOError::Invalid);
    }
    std::str::from_utf8(&bytes[..end])
        .ok()
        .filter(|value| value.is_ascii() && !value.is_empty())
        .ok_or(MachOError::Invalid)
}

fn has_write_execute(protection: u32) -> bool {
    protection & (VM_PROT_WRITE | VM_PROT_EXECUTE) == (VM_PROT_WRITE | VM_PROT_EXECUTE)
}
