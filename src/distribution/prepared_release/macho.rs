use std::collections::BTreeSet;
use std::io;

use crate::distribution::schema::ReleaseManifestV1;

mod linkedit;
mod segment;

use linkedit::{verify_dyld_info, verify_dysymtab, verify_linkedit_data, verify_symtab};
use segment::{verify_disjoint_segments, verify_segment};

const MACH_HEADER_64_BYTES: usize = 32;
const LOAD_COMMAND_HEADER_BYTES: usize = 8;
const MAX_LOAD_COMMANDS: u32 = 4_096;
const MAX_LOAD_COMMAND_BYTES: u32 = 1_048_576;

const MH_MAGIC_64: u32 = 0xfeedfacf;
const CPU_TYPE_ARM64: u32 = 0x0100_000c;
const CPU_SUBTYPE_ARM64_ALL: u32 = 0;
const MH_EXECUTE: u32 = 2;
const MH_NOUNDEFS: u32 = 0x0000_0001;
const MH_DYLDLINK: u32 = 0x0000_0004;
const MH_TWOLEVEL: u32 = 0x0000_0080;
const MH_ALLOW_STACK_EXECUTION: u32 = 0x0002_0000;
const MH_PIE: u32 = 0x0020_0000;
const MH_HAS_TLV_DESCRIPTORS: u32 = 0x0080_0000;
const MH_NO_HEAP_EXECUTION: u32 = 0x0100_0000;
const MH_APP_EXTENSION_SAFE: u32 = 0x0200_0000;
const ALLOWED_HEADER_FLAGS: u32 = MH_NOUNDEFS
    | MH_DYLDLINK
    | MH_TWOLEVEL
    | MH_PIE
    | MH_HAS_TLV_DESCRIPTORS
    | MH_NO_HEAP_EXECUTION
    | MH_APP_EXTENSION_SAFE;

const LC_SYMTAB: u32 = 0x02;
const LC_UNIXTHREAD: u32 = 0x05;
const LC_LOADFVMLIB: u32 = 0x06;
const LC_DYSYMTAB: u32 = 0x0b;
const LC_LOAD_DYLIB: u32 = 0x0c;
const LC_ID_DYLIB: u32 = 0x0d;
const LC_LOAD_DYLINKER: u32 = 0x0e;
const LC_PREBOUND_DYLIB: u32 = 0x10;
const LC_LOAD_WEAK_DYLIB: u32 = 0x8000_0018;
const LC_SEGMENT_64: u32 = 0x19;
const LC_UUID: u32 = 0x1b;
const LC_RPATH: u32 = 0x8000_001c;
const LC_CODE_SIGNATURE: u32 = 0x1d;
const LC_REEXPORT_DYLIB: u32 = 0x8000_001f;
const LC_LAZY_LOAD_DYLIB: u32 = 0x20;
const LC_ENCRYPTION_INFO: u32 = 0x21;
const LC_DYLD_INFO_ONLY: u32 = 0x8000_0022;
const LC_LOAD_UPWARD_DYLIB: u32 = 0x8000_0023;
const LC_VERSION_MIN_MACOSX: u32 = 0x24;
const LC_VERSION_MIN_IPHONEOS: u32 = 0x25;
const LC_FUNCTION_STARTS: u32 = 0x26;
const LC_DYLD_ENVIRONMENT: u32 = 0x27;
const LC_MAIN: u32 = 0x8000_0028;
const LC_DATA_IN_CODE: u32 = 0x29;
const LC_ENCRYPTION_INFO_64: u32 = 0x2c;
const LC_VERSION_MIN_TVOS: u32 = 0x2f;
const LC_VERSION_MIN_WATCHOS: u32 = 0x30;
const LC_BUILD_VERSION: u32 = 0x32;

const PLATFORM_MACOS: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(super) enum MachOError {
    #[error("the staged executable is outside the supported Mach-O profile")]
    Invalid,
    #[error("the staged executable could not be read completely")]
    Read,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct VerifiedMachO {
    code_signature_offset: u64,
    code_signature_length: u64,
}

impl VerifiedMachO {
    #[cfg(test)]
    pub(super) fn code_signature_range(&self) -> std::ops::Range<u64> {
        self.code_signature_offset..self.code_signature_offset + self.code_signature_length
    }
}

trait Source {
    fn len(&self) -> u64;
    fn read_exact_at(&self, offset: u64, buffer: &mut [u8]) -> io::Result<()>;
}

#[cfg(test)]
pub(super) fn verify_bytes(
    bytes: &[u8],
    manifest: &ReleaseManifestV1,
) -> Result<VerifiedMachO, MachOError> {
    verify_source(&ByteSource(bytes), manifest)
}

#[cfg(test)]
struct ByteSource<'a>(&'a [u8]);

#[cfg(test)]
impl Source for ByteSource<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_exact_at(&self, offset: u64, buffer: &mut [u8]) -> io::Result<()> {
        let start = usize::try_from(offset).map_err(|_| io::ErrorKind::InvalidData)?;
        let end = start
            .checked_add(buffer.len())
            .ok_or(io::ErrorKind::InvalidData)?;
        let source = self.0.get(start..end).ok_or(io::ErrorKind::UnexpectedEof)?;
        buffer.copy_from_slice(source);
        Ok(())
    }
}

fn verify_source(
    source: &impl Source,
    manifest: &ReleaseManifestV1,
) -> Result<VerifiedMachO, MachOError> {
    if !manifest.non_system_dynamic_dependencies().is_empty() {
        return Err(MachOError::Invalid);
    }
    let expected_binary = manifest
        .files()
        .iter()
        .find(|file| file.path().as_str() == "bin/hf2q")
        .ok_or(MachOError::Invalid)?;
    if source.len() != expected_binary.size() || source.len() < MACH_HEADER_64_BYTES as u64 {
        return Err(MachOError::Invalid);
    }

    let mut header = [0_u8; MACH_HEADER_64_BYTES];
    source
        .read_exact_at(0, &mut header)
        .map_err(|_| MachOError::Read)?;
    let header_flags = u32_at(&header, 24)?;
    if u32_at(&header, 0)? != MH_MAGIC_64
        || u32_at(&header, 4)? != CPU_TYPE_ARM64
        || u32_at(&header, 8)? != CPU_SUBTYPE_ARM64_ALL
        || u32_at(&header, 12)? != MH_EXECUTE
        || header_flags & (MH_NOUNDEFS | MH_DYLDLINK | MH_TWOLEVEL | MH_PIE)
            != (MH_NOUNDEFS | MH_DYLDLINK | MH_TWOLEVEL | MH_PIE)
        || header_flags & !ALLOWED_HEADER_FLAGS != 0
        || u32_at(&header, 28)? != 0
    {
        return Err(MachOError::Invalid);
    }
    let command_count = u32_at(&header, 16)?;
    let command_bytes = u32_at(&header, 20)?;
    if command_count == 0
        || command_count > MAX_LOAD_COMMANDS
        || command_bytes < command_count.saturating_mul(LOAD_COMMAND_HEADER_BYTES as u32)
        || command_bytes > MAX_LOAD_COMMAND_BYTES
    {
        return Err(MachOError::Invalid);
    }
    let commands_end = (MACH_HEADER_64_BYTES as u64)
        .checked_add(u64::from(command_bytes))
        .ok_or(MachOError::Invalid)?;
    if commands_end > source.len() {
        return Err(MachOError::Invalid);
    }
    let mut commands = vec![0_u8; command_bytes as usize];
    source
        .read_exact_at(MACH_HEADER_64_BYTES as u64, &mut commands)
        .map_err(|_| MachOError::Read)?;

    let expected_macos = encode_macos_version(manifest.minimum_macos().as_str())?;
    let mut cursor = 0_usize;
    let mut build_version_seen = false;
    let mut code_signature = None;
    let mut executable_text_range = None;
    let mut executable_text_section = None;
    let mut linkedit_range = None;
    let mut linkedit_payloads = Vec::new();
    let mut dylinker_seen = false;
    let mut main_entry_offset = None;
    let mut symtab_symbols = None;
    let mut dysymtab_seen = false;
    let mut dyld_info_seen = false;
    let mut uuid_seen = false;
    let mut function_starts_seen = false;
    let mut data_in_code_seen = false;
    let mut segments = Vec::new();
    let mut segment_names = BTreeSet::new();

    for _ in 0..command_count {
        let command = commands.get(cursor..).ok_or(MachOError::Invalid)?;
        let kind = u32_at(command, 0)?;
        let size = usize::try_from(u32_at(command, 4)?).map_err(|_| MachOError::Invalid)?;
        if size < LOAD_COMMAND_HEADER_BYTES || size % 8 != 0 {
            return Err(MachOError::Invalid);
        }
        let record = command.get(..size).ok_or(MachOError::Invalid)?;
        match kind {
            LC_SEGMENT_64 => {
                verify_segment(
                    record,
                    source.len(),
                    commands_end,
                    &mut executable_text_range,
                    &mut executable_text_section,
                    &mut linkedit_range,
                    &mut segments,
                    &mut segment_names,
                )?;
            }
            LC_SYMTAB => {
                if symtab_symbols.is_some() {
                    return Err(MachOError::Invalid);
                }
                symtab_symbols = Some(verify_symtab(record, source.len(), &mut linkedit_payloads)?);
            }
            LC_DYSYMTAB => {
                if dysymtab_seen {
                    return Err(MachOError::Invalid);
                }
                verify_dysymtab(
                    record,
                    symtab_symbols.ok_or(MachOError::Invalid)?,
                    source.len(),
                    &mut linkedit_payloads,
                )?;
                dysymtab_seen = true;
            }
            LC_DYLD_INFO_ONLY => {
                if dyld_info_seen {
                    return Err(MachOError::Invalid);
                }
                verify_dyld_info(record, source.len(), &mut linkedit_payloads)?;
                dyld_info_seen = true;
            }
            LC_UUID => {
                if uuid_seen || record.len() != 24 || record[8..24].iter().all(|byte| *byte == 0) {
                    return Err(MachOError::Invalid);
                }
                uuid_seen = true;
            }
            LC_FUNCTION_STARTS | LC_DATA_IN_CODE => {
                let seen = if kind == LC_FUNCTION_STARTS {
                    &mut function_starts_seen
                } else {
                    &mut data_in_code_seen
                };
                if *seen {
                    return Err(MachOError::Invalid);
                }
                linkedit_payloads.push(verify_linkedit_data(record, kind, source.len())?);
                *seen = true;
            }
            LC_BUILD_VERSION => {
                if build_version_seen {
                    return Err(MachOError::Invalid);
                }
                verify_build_version(record, expected_macos)?;
                build_version_seen = true;
            }
            LC_LOAD_DYLIB => verify_system_dylib(record)?,
            LC_LOAD_DYLINKER => {
                if dylinker_seen || command_string(record, 8, 12)? != "/usr/lib/dyld" {
                    return Err(MachOError::Invalid);
                }
                dylinker_seen = true;
            }
            LC_MAIN => {
                if main_entry_offset.is_some() || record.len() != 24 || u64_at(record, 16)? != 0 {
                    return Err(MachOError::Invalid);
                }
                main_entry_offset = Some(u64_at(record, 8)?);
            }
            LC_CODE_SIGNATURE => {
                if code_signature.is_some() || record.len() != 16 {
                    return Err(MachOError::Invalid);
                }
                let offset = u64::from(u32_at(record, 8)?);
                let length = u64::from(u32_at(record, 12)?);
                let end = offset.checked_add(length).ok_or(MachOError::Invalid)?;
                if length == 0 || offset < commands_end || end != source.len() {
                    return Err(MachOError::Invalid);
                }
                code_signature = Some((offset, length));
            }
            LC_UNIXTHREAD
            | LC_LOADFVMLIB
            | LC_ID_DYLIB
            | LC_PREBOUND_DYLIB
            | LC_LOAD_WEAK_DYLIB
            | LC_REEXPORT_DYLIB
            | LC_LAZY_LOAD_DYLIB
            | LC_LOAD_UPWARD_DYLIB
            | LC_RPATH
            | LC_ENCRYPTION_INFO
            | LC_ENCRYPTION_INFO_64
            | LC_VERSION_MIN_MACOSX
            | LC_VERSION_MIN_IPHONEOS
            | LC_VERSION_MIN_TVOS
            | LC_VERSION_MIN_WATCHOS
            | LC_DYLD_ENVIRONMENT => return Err(MachOError::Invalid),
            _ => return Err(MachOError::Invalid),
        }
        cursor = cursor.checked_add(size).ok_or(MachOError::Invalid)?;
    }
    if cursor != commands.len()
        || !build_version_seen
        || executable_text_range.is_none()
        || executable_text_section.is_none()
        || linkedit_range.is_none()
        || !dylinker_seen
        || main_entry_offset.is_none()
        || symtab_symbols.is_none()
        || !dysymtab_seen
        || !dyld_info_seen
        || !uuid_seen
        || !function_starts_seen
        || !data_in_code_seen
    {
        return Err(MachOError::Invalid);
    }
    let (code_signature_offset, code_signature_length) =
        code_signature.ok_or(MachOError::Invalid)?;
    let (linkedit_start, linkedit_end) = linkedit_range.ok_or(MachOError::Invalid)?;
    let (text_start, text_end) = executable_text_range.ok_or(MachOError::Invalid)?;
    verify_disjoint_segments(&segments)?;
    linkedit_payloads.sort_unstable();
    if code_signature_offset < linkedit_start
        || code_signature_offset + code_signature_length > linkedit_end
        || text_end > linkedit_start
        || linkedit_payloads.iter().any(|(start, end)| {
            *start < linkedit_start || *end > code_signature_offset || *end > linkedit_end
        })
        || linkedit_payloads
            .iter()
            .filter(|(start, end)| start < end)
            .zip(
                linkedit_payloads
                    .iter()
                    .filter(|(start, end)| start < end)
                    .skip(1),
            )
            .any(|((_, left_end), (right_start, _))| left_end > right_start)
    {
        return Err(MachOError::Invalid);
    }
    let (code_start, code_end) = executable_text_section.ok_or(MachOError::Invalid)?;
    let entry_offset = main_entry_offset.ok_or(MachOError::Invalid)?;
    if entry_offset < commands_end
        || entry_offset < text_start
        || entry_offset >= text_end
        || entry_offset < code_start
        || entry_offset >= code_end
        || entry_offset % 4 != 0
    {
        return Err(MachOError::Invalid);
    }
    Ok(VerifiedMachO {
        code_signature_offset,
        code_signature_length,
    })
}

fn verify_build_version(record: &[u8], expected_macos: u32) -> Result<(), MachOError> {
    if record.len() < 24 || u32_at(record, 8)? != PLATFORM_MACOS {
        return Err(MachOError::Invalid);
    }
    let tool_count = usize::try_from(u32_at(record, 20)?).map_err(|_| MachOError::Invalid)?;
    let expected_size = 24_usize
        .checked_add(tool_count.checked_mul(8).ok_or(MachOError::Invalid)?)
        .ok_or(MachOError::Invalid)?;
    if record.len() != expected_size
        || u32_at(record, 12)? != expected_macos
        || u32_at(record, 16)? < expected_macos
    {
        return Err(MachOError::Invalid);
    }
    Ok(())
}

fn verify_system_dylib(record: &[u8]) -> Result<(), MachOError> {
    if record.len() < 24 {
        return Err(MachOError::Invalid);
    }
    let path = command_string(record, 8, 24)?;
    let allowed = path
        .strip_prefix("/usr/lib/")
        .or_else(|| path.strip_prefix("/System/Library/Frameworks/"))
        .is_some_and(|suffix| {
            !suffix.is_empty()
                && !suffix.ends_with('/')
                && suffix
                    .split('/')
                    .all(|component| !component.is_empty() && !matches!(component, "." | ".."))
        });
    if !allowed {
        return Err(MachOError::Invalid);
    }
    Ok(())
}

fn command_string(
    record: &[u8],
    offset_field: usize,
    minimum_offset: usize,
) -> Result<&str, MachOError> {
    let offset = usize::try_from(u32_at(record, offset_field)?).map_err(|_| MachOError::Invalid)?;
    if offset < minimum_offset {
        return Err(MachOError::Invalid);
    }
    let bytes = record.get(offset..).ok_or(MachOError::Invalid)?;
    let nul = bytes
        .iter()
        .position(|byte| *byte == 0)
        .ok_or(MachOError::Invalid)?;
    if nul == 0 || bytes[nul + 1..].iter().any(|byte| *byte != 0) {
        return Err(MachOError::Invalid);
    }
    std::str::from_utf8(&bytes[..nul])
        .ok()
        .filter(|value| value.is_ascii())
        .ok_or(MachOError::Invalid)
}

fn encode_macos_version(value: &str) -> Result<u32, MachOError> {
    let mut components = value.split('.');
    let major = components
        .next()
        .and_then(|part| part.parse::<u16>().ok())
        .ok_or(MachOError::Invalid)?;
    let minor = components
        .next()
        .and_then(|part| part.parse::<u8>().ok())
        .ok_or(MachOError::Invalid)?;
    let patch = components
        .next()
        .map_or(Ok(0), str::parse::<u8>)
        .map_err(|_| MachOError::Invalid)?;
    if components.next().is_some() {
        return Err(MachOError::Invalid);
    }
    Ok((u32::from(major) << 16) | (u32::from(minor) << 8) | u32::from(patch))
}

fn u32_at(bytes: &[u8], offset: usize) -> Result<u32, MachOError> {
    let raw: [u8; 4] = bytes
        .get(offset..offset + 4)
        .ok_or(MachOError::Invalid)?
        .try_into()
        .map_err(|_| MachOError::Invalid)?;
    Ok(u32::from_le_bytes(raw))
}

fn u64_at(bytes: &[u8], offset: usize) -> Result<u64, MachOError> {
    let raw: [u8; 8] = bytes
        .get(offset..offset + 8)
        .ok_or(MachOError::Invalid)?
        .try_into()
        .map_err(|_| MachOError::Invalid)?;
    Ok(u64::from_le_bytes(raw))
}
