use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, SeekFrom};

use sha2::{Digest, Sha256};
use zip::{CompressionMethod, ZipArchive};

use super::PreparedReleaseError;
use crate::distribution::schema::{
    BundleEntryType, ReleaseManifestV1, MAX_BUNDLE_FILES, MAX_RELEASE_ARCHIVE_BYTES,
    MAX_RELEASE_MANIFEST_BYTES,
};

const MANIFEST_NAME: &str = "release-manifest.json";
const END_RECORD_BYTES: u64 = 22;
const CENTRAL_HEADER_BYTES: u64 = 46;
const LOCAL_HEADER_BYTES: u64 = 30;
const END_RECORD_SIGNATURE: u32 = 0x0605_4b50;
const CENTRAL_HEADER_SIGNATURE: u32 = 0x0201_4b50;
const LOCAL_HEADER_SIGNATURE: u32 = 0x0403_4b50;
const UNIX_CREATOR: u16 = 3;
const REGULAR_FILE_TYPE: u32 = 0o100000;
const FILE_TYPE_MASK: u32 = 0o170000;
const MODE_MASK: u32 = 0o7777;
const MAX_ENTRY_NAME_BYTES: usize = 512;
const CANONICAL_DOS_TIME: u16 = 0;
const CANONICAL_DOS_DATE: u16 = 0x21;

struct ExpectedEntry<'a> {
    size: u64,
    mode: u32,
    sha256: Option<&'a str>,
}

#[derive(Debug)]
pub(super) struct ClassicEntry {
    name: String,
    version_needed: u16,
    flags: u16,
    pub(super) method: u16,
    modified_time: u16,
    modified_date: u16,
    crc32: u32,
    pub(super) compressed_size: u32,
    pub(super) uncompressed_size: u32,
    local_header_start: u32,
    pub(super) data_start: u64,
    central_header_start: u64,
}

pub(super) fn verify_archive<R: Read + Seek>(
    reader: &mut R,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
) -> Result<(), PreparedReleaseError> {
    if exact_manifest.is_empty()
        || exact_manifest.len() > MAX_RELEASE_MANIFEST_BYTES
        || manifest
            .to_deterministic_json()
            .map_err(|_| PreparedReleaseError::ArchiveProfile)?
            != exact_manifest
    {
        return Err(PreparedReleaseError::ArchiveProfile);
    }
    let expected = expected_entries(exact_manifest, manifest)?;
    let classic = verify_classic_structure(reader, &expected)?;
    super::deflate::verify_exact_streams(reader, &classic)?;
    verify_library_view(reader, &expected, &classic)?;
    verify_contents(reader, exact_manifest, &expected)?;
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|_| PreparedReleaseError::ArchiveRead)?;
    Ok(())
}

fn expected_entries<'a>(
    exact_manifest: &'a [u8],
    manifest: &'a ReleaseManifestV1,
) -> Result<BTreeMap<&'a str, ExpectedEntry<'a>>, PreparedReleaseError> {
    let expected_count = manifest
        .files()
        .len()
        .checked_add(1)
        .ok_or(PreparedReleaseError::ArchiveProfile)?;
    if manifest.files().is_empty() || expected_count > MAX_BUNDLE_FILES + 1 {
        return Err(PreparedReleaseError::ArchiveProfile);
    }
    let mut expected = BTreeMap::new();
    expected.insert(
        MANIFEST_NAME,
        ExpectedEntry {
            size: exact_manifest.len() as u64,
            mode: 0o644,
            sha256: None,
        },
    );
    for file in manifest.files() {
        if file.file_type() != BundleEntryType::Regular
            || file.size() >= u64::from(u32::MAX)
            || expected
                .insert(
                    file.path().as_str(),
                    ExpectedEntry {
                        size: file.size(),
                        mode: file.mode().as_octal(),
                        sha256: Some(file.sha256().as_str()),
                    },
                )
                .is_some()
        {
            return Err(PreparedReleaseError::ArchiveProfile);
        }
    }
    Ok(expected)
}

/// Parse the bounded classic-ZIP structure independently of `zip`.
///
/// `zip` is used only after this pass, as a Stored/Deflate decoder. This pass
/// preserves every central-directory record, binds it to its exact local
/// header, and rejects layouts that a higher-level name index could normalize.
fn verify_classic_structure<R: Read + Seek>(
    reader: &mut R,
    expected: &BTreeMap<&str, ExpectedEntry<'_>>,
) -> Result<Vec<ClassicEntry>, PreparedReleaseError> {
    let length = reader
        .seek(SeekFrom::End(0))
        .map_err(|_| PreparedReleaseError::ArchiveRead)?;
    if length < END_RECORD_BYTES || length > MAX_RELEASE_ARCHIVE_BYTES as u64 {
        return Err(PreparedReleaseError::ArchiveProfile);
    }
    reader
        .seek(SeekFrom::End(-(END_RECORD_BYTES as i64)))
        .map_err(|_| PreparedReleaseError::ArchiveRead)?;
    let end = read_array::<22>(reader)?;
    let disk = little_u16(&end[4..6]);
    let central_disk = little_u16(&end[6..8]);
    let disk_entries = little_u16(&end[8..10]);
    let total_entries = little_u16(&end[10..12]);
    let central_bytes = little_u32(&end[12..16]);
    let central_offset = little_u32(&end[16..20]);
    let comment_bytes = little_u16(&end[20..22]);
    if little_u32(&end[..4]) != END_RECORD_SIGNATURE
        || disk != 0
        || central_disk != 0
        || disk_entries != total_entries
        || total_entries as usize != expected.len()
        || expected.len() > MAX_BUNDLE_FILES + 1
        || total_entries == u16::MAX
        || central_bytes == u32::MAX
        || central_offset == u32::MAX
        || comment_bytes != 0
        || u64::from(central_offset).checked_add(u64::from(central_bytes))
            != length.checked_sub(END_RECORD_BYTES)
    {
        return Err(PreparedReleaseError::ArchiveProfile);
    }

    let central_start = u64::from(central_offset);
    let central_end = central_start
        .checked_add(u64::from(central_bytes))
        .ok_or(PreparedReleaseError::ArchiveProfile)?;
    reader
        .seek(SeekFrom::Start(central_start))
        .map_err(|_| PreparedReleaseError::ArchiveRead)?;
    let canonical_names: Vec<_> = std::iter::once(MANIFEST_NAME)
        .chain(
            expected
                .keys()
                .copied()
                .filter(|name| *name != MANIFEST_NAME),
        )
        .collect();
    let mut entries = Vec::with_capacity(expected.len());
    let mut seen = BTreeSet::new();
    for expected_name in &canonical_names {
        let central_header_start = reader
            .stream_position()
            .map_err(|_| PreparedReleaseError::ArchiveRead)?;
        let header = read_array::<46>(reader)?;
        let version_made_by = little_u16(&header[4..6]);
        let version_needed = little_u16(&header[6..8]);
        let flags = little_u16(&header[8..10]);
        let method = little_u16(&header[10..12]);
        let modified_time = little_u16(&header[12..14]);
        let modified_date = little_u16(&header[14..16]);
        let crc32 = little_u32(&header[16..20]);
        let compressed_size = little_u32(&header[20..24]);
        let uncompressed_size = little_u32(&header[24..28]);
        let name_bytes = little_u16(&header[28..30]) as usize;
        let extra_bytes = little_u16(&header[30..32]);
        let comment_bytes = little_u16(&header[32..34]);
        let disk_start = little_u16(&header[34..36]);
        let internal_attributes = little_u16(&header[36..38]);
        let external_attributes = little_u32(&header[38..42]);
        let local_header_start = little_u32(&header[42..46]);
        let raw_name = read_name(reader, name_bytes)?;
        let name = std::str::from_utf8(&raw_name)
            .map_err(|_| PreparedReleaseError::ArchiveProfile)?
            .to_owned();
        let Some(spec) = expected.get(name.as_str()) else {
            return Err(PreparedReleaseError::ArchiveProfile);
        };
        if little_u32(&header[..4]) != CENTRAL_HEADER_SIGNATURE
            || version_made_by >> 8 != UNIX_CREATOR
            || !matches!((method, version_needed), (0, 10) | (8, 20))
            || flags != 0
            || modified_time != CANONICAL_DOS_TIME
            || modified_date != CANONICAL_DOS_DATE
            || compressed_size == u32::MAX
            || uncompressed_size == u32::MAX
            || u64::from(uncompressed_size) != spec.size
            || (method == 0 && compressed_size != uncompressed_size)
            || extra_bytes != 0
            || comment_bytes != 0
            || disk_start != 0
            || internal_attributes != 0
            || external_attributes & 0xffff != 0
            || !exact_regular_mode(Some(external_attributes >> 16), spec.mode)
            || local_header_start == u32::MAX
            || name != *expected_name
            || !seen.insert(name.clone())
        {
            return Err(PreparedReleaseError::ArchiveProfile);
        }
        let after_record = central_header_start
            .checked_add(CENTRAL_HEADER_BYTES)
            .and_then(|position| position.checked_add(name_bytes as u64))
            .ok_or(PreparedReleaseError::ArchiveProfile)?;
        if after_record > central_end {
            return Err(PreparedReleaseError::ArchiveProfile);
        }
        entries.push(ClassicEntry {
            name,
            version_needed,
            flags,
            method,
            modified_time,
            modified_date,
            crc32,
            compressed_size,
            uncompressed_size,
            local_header_start,
            data_start: 0,
            central_header_start,
        });
    }
    if reader
        .stream_position()
        .map_err(|_| PreparedReleaseError::ArchiveRead)?
        != central_end
        || seen.len() != expected.len()
    {
        return Err(PreparedReleaseError::ArchiveProfile);
    }

    entries.sort_by_key(|entry| entry.local_header_start);
    if entries
        .iter()
        .map(|entry| entry.name.as_str())
        .ne(canonical_names.iter().copied())
    {
        return Err(PreparedReleaseError::ArchiveProfile);
    }
    let mut next_local_start = 0_u64;
    for entry in &mut entries {
        let local_start = u64::from(entry.local_header_start);
        if local_start != next_local_start {
            return Err(PreparedReleaseError::ArchiveProfile);
        }
        reader
            .seek(SeekFrom::Start(local_start))
            .map_err(|_| PreparedReleaseError::ArchiveRead)?;
        let local = read_array::<30>(reader)?;
        let name_bytes = little_u16(&local[26..28]) as usize;
        let extra_bytes = little_u16(&local[28..30]);
        let raw_name = read_name(reader, name_bytes)?;
        if little_u32(&local[..4]) != LOCAL_HEADER_SIGNATURE
            || little_u16(&local[4..6]) != entry.version_needed
            || little_u16(&local[6..8]) != entry.flags
            || little_u16(&local[8..10]) != entry.method
            || little_u16(&local[10..12]) != entry.modified_time
            || little_u16(&local[12..14]) != entry.modified_date
            || little_u32(&local[14..18]) != entry.crc32
            || little_u32(&local[18..22]) != entry.compressed_size
            || little_u32(&local[22..26]) != entry.uncompressed_size
            || extra_bytes != 0
            || raw_name != entry.name.as_bytes()
        {
            return Err(PreparedReleaseError::ArchiveProfile);
        }
        let data_start = local_start
            .checked_add(LOCAL_HEADER_BYTES)
            .and_then(|position| position.checked_add(name_bytes as u64))
            .ok_or(PreparedReleaseError::ArchiveProfile)?;
        let data_end = data_start
            .checked_add(u64::from(entry.compressed_size))
            .ok_or(PreparedReleaseError::ArchiveProfile)?;
        if data_end > central_start {
            return Err(PreparedReleaseError::ArchiveProfile);
        }
        entry.data_start = data_start;
        next_local_start = data_end;
    }
    if next_local_start != central_start {
        return Err(PreparedReleaseError::ArchiveProfile);
    }
    entries.sort_by_key(|entry| entry.central_header_start);
    Ok(entries)
}

fn verify_library_view<R: Read + Seek>(
    reader: &mut R,
    expected: &BTreeMap<&str, ExpectedEntry<'_>>,
    classic: &[ClassicEntry],
) -> Result<(), PreparedReleaseError> {
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|_| PreparedReleaseError::ArchiveRead)?;
    let mut archive =
        ZipArchive::new(&mut *reader).map_err(|_| PreparedReleaseError::ArchiveProfile)?;
    if archive.offset() != 0
        || archive.len() != classic.len()
        || !archive.comment().is_empty()
        || archive.zip64_comment().is_some()
    {
        return Err(PreparedReleaseError::ArchiveProfile);
    }
    for (index, descriptor) in classic.iter().enumerate() {
        let entry = archive
            .by_index(index)
            .map_err(|_| PreparedReleaseError::ArchiveProfile)?;
        let raw_name = std::str::from_utf8(entry.name_raw())
            .map_err(|_| PreparedReleaseError::ArchiveProfile)?;
        let spec = expected
            .get(raw_name)
            .ok_or(PreparedReleaseError::ArchiveProfile)?;
        if raw_name != descriptor.name
            || raw_name != entry.name()
            || entry.encrypted()
            || !entry.comment().is_empty()
            || !matches!(
                entry.compression(),
                CompressionMethod::Stored | CompressionMethod::Deflated
            )
            || entry.size() != spec.size
            || entry.size() != u64::from(descriptor.uncompressed_size)
            || entry.compressed_size() != u64::from(descriptor.compressed_size)
            || entry.crc32() != descriptor.crc32
            || entry.header_start() != u64::from(descriptor.local_header_start)
            || entry.data_start() != descriptor.data_start
            || entry.central_header_start() != descriptor.central_header_start
            || !exact_regular_mode(entry.unix_mode(), spec.mode)
        {
            return Err(PreparedReleaseError::ArchiveProfile);
        }
    }
    Ok(())
}

fn verify_contents<R: Read + Seek>(
    reader: &mut R,
    exact_manifest: &[u8],
    expected: &BTreeMap<&str, ExpectedEntry<'_>>,
) -> Result<(), PreparedReleaseError> {
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|_| PreparedReleaseError::ArchiveRead)?;
    let mut archive =
        ZipArchive::new(&mut *reader).map_err(|_| PreparedReleaseError::ArchiveProfile)?;
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|_| PreparedReleaseError::ArchiveRead)?;
        let name = std::str::from_utf8(entry.name_raw())
            .map_err(|_| PreparedReleaseError::ArchiveProfile)?;
        let spec = expected
            .get(name)
            .ok_or(PreparedReleaseError::ArchiveProfile)?;
        if name == MANIFEST_NAME {
            if !stream_matches_bytes(&mut entry, exact_manifest)? {
                return Err(PreparedReleaseError::ArchiveProfile);
            }
        } else if !stream_matches_digest(
            &mut entry,
            spec.size,
            spec.sha256.ok_or(PreparedReleaseError::ArchiveProfile)?,
        )? {
            return Err(PreparedReleaseError::ArchiveProfile);
        }
    }
    Ok(())
}

fn read_array<const N: usize>(reader: &mut impl Read) -> Result<[u8; N], PreparedReleaseError> {
    let mut bytes = [0_u8; N];
    reader
        .read_exact(&mut bytes)
        .map_err(|_| PreparedReleaseError::ArchiveRead)?;
    Ok(bytes)
}

fn read_name(reader: &mut impl Read, length: usize) -> Result<Vec<u8>, PreparedReleaseError> {
    if length == 0 || length > MAX_ENTRY_NAME_BYTES {
        return Err(PreparedReleaseError::ArchiveProfile);
    }
    let mut bytes = vec![0_u8; length];
    reader
        .read_exact(&mut bytes)
        .map_err(|_| PreparedReleaseError::ArchiveRead)?;
    Ok(bytes)
}

fn stream_matches_bytes(
    reader: &mut impl Read,
    expected: &[u8],
) -> Result<bool, PreparedReleaseError> {
    let mut offset = 0_usize;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|_| PreparedReleaseError::ArchiveRead)?;
        if count == 0 {
            break;
        }
        let Some(end) = offset.checked_add(count) else {
            return Ok(false);
        };
        if end > expected.len() || buffer[..count] != expected[offset..end] {
            return Ok(false);
        }
        offset = end;
    }
    Ok(offset == expected.len())
}

fn stream_matches_digest(
    reader: &mut impl Read,
    expected_size: u64,
    expected_sha256: &str,
) -> Result<bool, PreparedReleaseError> {
    let mut size = 0_u64;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|_| PreparedReleaseError::ArchiveRead)?;
        if count == 0 {
            break;
        }
        size = size
            .checked_add(count as u64)
            .ok_or(PreparedReleaseError::ArchiveProfile)?;
        if size > expected_size {
            return Ok(false);
        }
        hasher.update(&buffer[..count]);
    }
    Ok(size == expected_size && hex::encode(hasher.finalize()) == expected_sha256)
}

fn exact_regular_mode(actual: Option<u32>, expected: u32) -> bool {
    actual.is_some_and(|mode| {
        mode & FILE_TYPE_MASK == REGULAR_FILE_TYPE && mode & MODE_MASK == expected
    })
}

fn little_u16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

fn little_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}
