use std::io::{Cursor, Read, Seek, SeekFrom, Write};

use serde_json::json;
use sha2::{Digest, Sha256};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

use super::archive::verify_archive;
use super::{ArchiveIntegrity, PreparedReleaseError};
use crate::distribution::install_state::ArtifactStageError;
use crate::distribution::schema::ReleaseManifestV1;

const PAYLOADS: [(&str, &[u8], u32); 4] = [
    ("bin/hf2q", b"signed hf2q binary\n", 0o755),
    (
        "libexec/serve_qwen38_opencode.sh",
        b"#!/bin/sh\nexec hf2q serve Qwen/Qwen3.8-27B\n",
        0o755,
    ),
    (
        "share/doc/hf2q/README.md",
        b"packaged documentation\n",
        0o644,
    ),
    ("share/licenses/hf2q/LICENSE-APACHE", b"Apache-2.0\n", 0o644),
];

struct CountingArchive {
    bytes: Cursor<Vec<u8>>,
    revalidations: usize,
    fail_on: Option<usize>,
}

impl Read for CountingArchive {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.bytes.read(buffer)
    }
}

impl Seek for CountingArchive {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.bytes.seek(position)
    }
}

impl ArchiveIntegrity for CountingArchive {
    fn revalidate_for_preparation(&mut self) -> Result<(), PreparedReleaseError> {
        self.revalidations += 1;
        if self.fail_on == Some(self.revalidations) {
            return Err(PreparedReleaseError::ArchiveIntegrity(
                ArtifactStageError::Integrity,
            ));
        }
        Ok(())
    }
}

#[test]
fn stored_and_deflated_archives_bind_exact_manifest_and_inventory() {
    let (manifest, manifest_bytes) = manifest();
    for compression in [CompressionMethod::Stored, CompressionMethod::Deflated] {
        let bytes = archive(&manifest_bytes, &PAYLOADS, compression, None);
        verify_archive(&mut Cursor::new(bytes), &manifest_bytes, &manifest)
            .expect("valid exact archive");
    }
}

#[test]
fn binding_composes_pre_and_post_archive_revalidation() {
    let (manifest, manifest_bytes) = manifest();
    let bytes = archive(&manifest_bytes, &PAYLOADS, CompressionMethod::Stored, None);
    let mut checked = CountingArchive {
        bytes: Cursor::new(bytes.clone()),
        revalidations: 0,
        fail_on: None,
    };
    super::validate_archive_reader(&mut checked, &manifest_bytes, &manifest)
        .expect("archive is revalidated around profile binding");
    assert_eq!(checked.revalidations, 2);

    let mut changed_after_profile = CountingArchive {
        bytes: Cursor::new(bytes),
        revalidations: 0,
        fail_on: Some(2),
    };
    assert!(matches!(
        super::validate_archive_reader(&mut changed_after_profile, &manifest_bytes, &manifest),
        Err(PreparedReleaseError::ArchiveIntegrity(
            ArtifactStageError::Integrity
        ))
    ));
    assert_eq!(changed_after_profile.revalidations, 2);
}

#[test]
fn embedded_manifest_must_be_byte_identical() {
    let (manifest, manifest_bytes) = manifest();
    let mut changed = manifest_bytes.clone();
    changed.insert(0, b' ');
    let bytes = archive(&changed, &PAYLOADS, CompressionMethod::Stored, None);
    assert!(matches!(
        verify_archive(&mut Cursor::new(bytes), &manifest_bytes, &manifest),
        Err(PreparedReleaseError::ArchiveProfile)
    ));
}

#[test]
fn external_manifest_must_use_the_deterministic_wire_encoding() {
    let (manifest, manifest_bytes) = manifest();
    let parsed: serde_json::Value = serde_json::from_slice(&manifest_bytes).expect("manifest JSON");
    let alternate = serde_json::to_vec_pretty(&parsed).expect("pretty manifest");
    let bytes = archive(&alternate, &PAYLOADS, CompressionMethod::Stored, None);
    assert!(verify_archive(&mut Cursor::new(bytes), &alternate, &manifest).is_err());
}

#[test]
fn archive_entry_set_is_exact() {
    let (manifest, manifest_bytes) = manifest();
    let missing = archive(
        &manifest_bytes,
        &PAYLOADS[..PAYLOADS.len() - 1],
        CompressionMethod::Stored,
        None,
    );
    assert!(verify_archive(&mut Cursor::new(missing), &manifest_bytes, &manifest).is_err());

    let extra = archive(
        &manifest_bytes,
        &PAYLOADS,
        CompressionMethod::Stored,
        Some(("../escape", b"nope", 0o644)),
    );
    assert!(verify_archive(&mut Cursor::new(extra), &manifest_bytes, &manifest).is_err());
}

#[test]
fn archive_rejects_wrong_mode_and_payload_digest() {
    let (manifest, manifest_bytes) = manifest();
    let mut wrong_mode = PAYLOADS;
    wrong_mode[0].2 = 0o644;
    let bytes = archive(
        &manifest_bytes,
        &wrong_mode,
        CompressionMethod::Stored,
        None,
    );
    assert!(verify_archive(&mut Cursor::new(bytes), &manifest_bytes, &manifest).is_err());

    let mut wrong_bytes = PAYLOADS;
    wrong_bytes[0].1 = b"different binary\n";
    let bytes = archive(
        &manifest_bytes,
        &wrong_bytes,
        CompressionMethod::Stored,
        None,
    );
    assert!(verify_archive(&mut Cursor::new(bytes), &manifest_bytes, &manifest).is_err());
}

#[test]
fn archive_rejects_comments_trailing_bytes_and_zip64_sentinels() {
    let (manifest, manifest_bytes) = manifest();
    let mut commented = archive(&manifest_bytes, &PAYLOADS, CompressionMethod::Stored, None);
    set_archive_comment(&mut commented, b"comment");
    assert!(verify_archive(&mut Cursor::new(commented), &manifest_bytes, &manifest).is_err());

    let mut trailing = archive(&manifest_bytes, &PAYLOADS, CompressionMethod::Stored, None);
    trailing.extend_from_slice(b"trailing");
    assert!(verify_archive(&mut Cursor::new(trailing), &manifest_bytes, &manifest).is_err());

    let mut zip64 = archive(&manifest_bytes, &PAYLOADS, CompressionMethod::Stored, None);
    let eocd = zip64.len() - 22;
    zip64[eocd + 10..eocd + 12].copy_from_slice(&u16::MAX.to_le_bytes());
    assert!(verify_archive(&mut Cursor::new(zip64), &manifest_bytes, &manifest).is_err());
}

#[test]
fn archive_preflight_rejects_declared_entry_count_before_parser_allocation() {
    let (manifest, manifest_bytes) = manifest();
    let mut bytes = archive(&manifest_bytes, &PAYLOADS, CompressionMethod::Stored, None);
    let eocd = bytes.len() - 22;
    let too_many = (crate::distribution::schema::MAX_BUNDLE_FILES as u16).saturating_add(2);
    bytes[eocd + 8..eocd + 10].copy_from_slice(&too_many.to_le_bytes());
    bytes[eocd + 10..eocd + 12].copy_from_slice(&too_many.to_le_bytes());
    assert!(verify_archive(&mut Cursor::new(bytes), &manifest_bytes, &manifest).is_err());
}

#[test]
fn archive_rejects_duplicate_and_noncanonical_entry_order() {
    let (manifest, manifest_bytes) = manifest();
    let mut duplicate = PAYLOADS;
    duplicate[3].0 = "tmp/xxxx";
    let mut bytes = archive(&manifest_bytes, &duplicate, CompressionMethod::Stored, None);
    let central = central_headers(&bytes);
    let local = local_headers(&bytes, &central);
    assert_eq!(duplicate[3].0.len(), PAYLOADS[0].0.len());
    bytes[central[4] + 46..central[4] + 46 + PAYLOADS[0].0.len()]
        .copy_from_slice(PAYLOADS[0].0.as_bytes());
    bytes[local[4] + 30..local[4] + 30 + PAYLOADS[0].0.len()]
        .copy_from_slice(PAYLOADS[0].0.as_bytes());
    assert!(verify_archive(&mut Cursor::new(bytes), &manifest_bytes, &manifest).is_err());

    let reversed: Vec<_> = PAYLOADS.iter().copied().rev().collect();
    let bytes = archive(&manifest_bytes, &reversed, CompressionMethod::Stored, None);
    assert!(verify_archive(&mut Cursor::new(bytes), &manifest_bytes, &manifest).is_err());
}

#[test]
fn archive_rejects_local_central_mismatch_and_overlapping_records() {
    let (manifest, manifest_bytes) = manifest();
    let original = archive(&manifest_bytes, &PAYLOADS, CompressionMethod::Stored, None);
    let central = central_headers(&original);
    let local = local_headers(&original, &central);

    let mut method_mismatch = original.clone();
    method_mismatch[local[1] + 8..local[1] + 10].copy_from_slice(&8_u16.to_le_bytes());
    assert!(verify_archive(
        &mut Cursor::new(method_mismatch),
        &manifest_bytes,
        &manifest
    )
    .is_err());

    let mut overlap = original;
    let first_offset = u32::try_from(local[0]).expect("classic local offset");
    overlap[central[1] + 42..central[1] + 46].copy_from_slice(&first_offset.to_le_bytes());
    assert!(verify_archive(&mut Cursor::new(overlap), &manifest_bytes, &manifest).is_err());
}

#[test]
fn archive_rejects_invalid_method_version_pairs_and_physical_order() {
    let (manifest, manifest_bytes) = manifest();
    let mut wrong_version = archive(
        &manifest_bytes,
        &PAYLOADS,
        CompressionMethod::Deflated,
        None,
    );
    let central = central_headers(&wrong_version);
    let local = local_headers(&wrong_version, &central);
    wrong_version[central[0] + 6..central[0] + 8].copy_from_slice(&10_u16.to_le_bytes());
    wrong_version[local[0] + 4..local[0] + 6].copy_from_slice(&10_u16.to_le_bytes());
    assert!(verify_archive(&mut Cursor::new(wrong_version), &manifest_bytes, &manifest).is_err());

    let original = archive(&manifest_bytes, &PAYLOADS, CompressionMethod::Stored, None);
    let reordered = reorder_local_records(&original, &[0, 2, 1, 3, 4]);
    assert!(verify_archive(&mut Cursor::new(reordered), &manifest_bytes, &manifest).is_err());
}

#[test]
fn deflate_stream_must_consume_its_exact_declared_range() {
    let (manifest, manifest_bytes) = manifest();
    let original = archive(
        &manifest_bytes,
        &PAYLOADS,
        CompressionMethod::Deflated,
        None,
    );
    let central = central_headers(&original);
    let local = local_headers(&original, &central);
    let old_central_start = central[0];
    let mut with_junk = original.clone();
    with_junk.insert(old_central_start, 0);
    let shifted_last_central = central[4] + 1;
    let last_compressed = u32::from_le_bytes([
        with_junk[shifted_last_central + 20],
        with_junk[shifted_last_central + 21],
        with_junk[shifted_last_central + 22],
        with_junk[shifted_last_central + 23],
    ]) + 1;
    with_junk[shifted_last_central + 20..shifted_last_central + 24]
        .copy_from_slice(&last_compressed.to_le_bytes());
    with_junk[local[4] + 18..local[4] + 22].copy_from_slice(&last_compressed.to_le_bytes());
    let eocd = with_junk.len() - 22;
    let shifted_central_offset = u32::try_from(old_central_start + 1).expect("classic offset");
    with_junk[eocd + 16..eocd + 20].copy_from_slice(&shifted_central_offset.to_le_bytes());
    assert!(verify_archive(&mut Cursor::new(with_junk), &manifest_bytes, &manifest).is_err());

    // A high-level Deflate `Read` adapter can emit the complete plaintext for
    // this shape even though the terminal marker byte is gone. The low-level
    // verifier must still reject it because `StreamEnd` was never observed.
    let mut truncated = original;
    truncated.remove(old_central_start - 1);
    let shifted_last_central = central[4] - 1;
    let last_compressed = u32::from_le_bytes([
        truncated[shifted_last_central + 20],
        truncated[shifted_last_central + 21],
        truncated[shifted_last_central + 22],
        truncated[shifted_last_central + 23],
    ]) - 1;
    truncated[shifted_last_central + 20..shifted_last_central + 24]
        .copy_from_slice(&last_compressed.to_le_bytes());
    truncated[local[4] + 18..local[4] + 22].copy_from_slice(&last_compressed.to_le_bytes());
    let eocd = truncated.len() - 22;
    let shifted_central_offset = u32::try_from(old_central_start - 1).expect("classic offset");
    truncated[eocd + 16..eocd + 20].copy_from_slice(&shifted_central_offset.to_le_bytes());
    assert!(verify_archive(&mut Cursor::new(truncated), &manifest_bytes, &manifest).is_err());
}

#[test]
fn archive_rejects_noncanonical_dos_timestamp_even_when_headers_agree() {
    let (manifest, manifest_bytes) = manifest();
    let mut bytes = archive(&manifest_bytes, &PAYLOADS, CompressionMethod::Stored, None);
    let central = central_headers(&bytes);
    let local = local_headers(&bytes, &central);
    bytes[central[0] + 12..central[0] + 14].copy_from_slice(&1_u16.to_le_bytes());
    bytes[central[0] + 14..central[0] + 16].copy_from_slice(&0x22_u16.to_le_bytes());
    bytes[local[0] + 10..local[0] + 12].copy_from_slice(&1_u16.to_le_bytes());
    bytes[local[0] + 12..local[0] + 14].copy_from_slice(&0x22_u16.to_le_bytes());
    assert!(verify_archive(&mut Cursor::new(bytes), &manifest_bytes, &manifest).is_err());
}

#[test]
fn archive_rejects_flags_methods_special_modes_and_invalid_names() {
    let (manifest, manifest_bytes) = manifest();
    let original = archive(&manifest_bytes, &PAYLOADS, CompressionMethod::Stored, None);
    let central = central_headers(&original);
    let local = local_headers(&original, &central);

    let mut data_descriptor = original.clone();
    data_descriptor[central[0] + 8..central[0] + 10].copy_from_slice(&8_u16.to_le_bytes());
    data_descriptor[local[0] + 6..local[0] + 8].copy_from_slice(&8_u16.to_le_bytes());
    assert!(verify_archive(
        &mut Cursor::new(data_descriptor),
        &manifest_bytes,
        &manifest
    )
    .is_err());

    let mut unsupported_method = original.clone();
    unsupported_method[central[0] + 10..central[0] + 12].copy_from_slice(&99_u16.to_le_bytes());
    unsupported_method[local[0] + 8..local[0] + 10].copy_from_slice(&99_u16.to_le_bytes());
    assert!(verify_archive(
        &mut Cursor::new(unsupported_method),
        &manifest_bytes,
        &manifest
    )
    .is_err());

    let mut setuid = original.clone();
    let special_attributes = (0o104755_u32) << 16;
    setuid[central[1] + 38..central[1] + 42].copy_from_slice(&special_attributes.to_le_bytes());
    assert!(verify_archive(&mut Cursor::new(setuid), &manifest_bytes, &manifest).is_err());

    let mut invalid_utf8 = original;
    invalid_utf8[central[1] + 46] = 0xff;
    invalid_utf8[local[1] + 30] = 0xff;
    assert!(verify_archive(&mut Cursor::new(invalid_utf8), &manifest_bytes, &manifest).is_err());
}

#[test]
fn archive_rejects_prepended_bytes_and_corrupt_payload_crc() {
    let (manifest, manifest_bytes) = manifest();
    let original = archive(&manifest_bytes, &PAYLOADS, CompressionMethod::Stored, None);
    let mut prepended = b"polyglot".to_vec();
    prepended.extend_from_slice(&original);
    assert!(verify_archive(&mut Cursor::new(prepended), &manifest_bytes, &manifest).is_err());

    let central = central_headers(&original);
    let local = local_headers(&original, &central);
    let mut corrupt = original;
    let manifest_name_length = "release-manifest.json".len();
    let first_payload = local[1] + 30 + PAYLOADS[0].0.len();
    assert_eq!(local[0] + 30 + manifest_name_length, 51);
    corrupt[first_payload] ^= 0xff;
    assert!(verify_archive(&mut Cursor::new(corrupt), &manifest_bytes, &manifest).is_err());
}

fn manifest() -> (ReleaseManifestV1, Vec<u8>) {
    let files: Vec<_> = PAYLOADS
        .iter()
        .map(|(path, bytes, mode)| {
            json!({
                "path": path,
                "type": "regular",
                "size": bytes.len(),
                "mode": if *mode == 0o755 { "0755" } else { "0644" },
                "sha256": digest(bytes),
            })
        })
        .collect();
    let raw = serde_json::to_vec(&json!({
        "kind": "hf2q.release-manifest",
        "schema_version": 1,
        "package": "hf2q",
        "version": "0.2.0",
        "target": "aarch64-apple-darwin",
        "minimum_macos": "14.0",
        "source_commit": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "channel": "stable",
        "code_signing": {
            "team_id": "A1B2C3D4E5",
            "identifier": "us.hf2q.cli",
            "certificate_common_name": "Developer ID Application: hf2q (A1B2C3D4E5)"
        },
        "compatibility": {
            "minimum_installer_protocol": 1,
            "minimum_updater_protocol": 1,
            "launcher_registry_schema": 1
        },
        "files": files,
        "non_system_dynamic_dependencies": []
    }))
    .expect("manifest JSON");
    let manifest = ReleaseManifestV1::parse_and_validate(&raw).expect("valid manifest");
    let bytes = manifest
        .to_deterministic_json()
        .expect("canonical manifest");
    (manifest, bytes)
}

fn archive(
    manifest: &[u8],
    payloads: &[(&str, &[u8], u32)],
    compression: CompressionMethod,
    extra: Option<(&str, &[u8], u32)>,
) -> Vec<u8> {
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    write_entry(
        &mut writer,
        "release-manifest.json",
        manifest,
        0o644,
        compression,
    );
    for (name, bytes, mode) in payloads {
        write_entry(&mut writer, name, bytes, *mode, compression);
    }
    if let Some((name, bytes, mode)) = extra {
        write_entry(&mut writer, name, bytes, mode, compression);
    }
    writer.finish().expect("finish ZIP").into_inner()
}

fn write_entry(
    writer: &mut ZipWriter<Cursor<Vec<u8>>>,
    name: &str,
    bytes: &[u8],
    mode: u32,
    compression: CompressionMethod,
) {
    let options = SimpleFileOptions::default()
        .compression_method(compression)
        .unix_permissions(mode);
    writer.start_file(name, options).expect("start ZIP entry");
    writer.write_all(bytes).expect("write ZIP entry");
}

fn set_archive_comment(bytes: &mut Vec<u8>, comment: &[u8]) {
    let eocd = bytes.len() - 22;
    bytes[eocd + 20..eocd + 22].copy_from_slice(&(comment.len() as u16).to_le_bytes());
    bytes.extend_from_slice(comment);
}

fn central_headers(bytes: &[u8]) -> Vec<usize> {
    let eocd = bytes.len() - 22;
    let count = u16::from_le_bytes([bytes[eocd + 10], bytes[eocd + 11]]) as usize;
    let mut cursor = u32::from_le_bytes([
        bytes[eocd + 16],
        bytes[eocd + 17],
        bytes[eocd + 18],
        bytes[eocd + 19],
    ]) as usize;
    let mut headers = Vec::with_capacity(count);
    for _ in 0..count {
        headers.push(cursor);
        let name = u16::from_le_bytes([bytes[cursor + 28], bytes[cursor + 29]]) as usize;
        let extra = u16::from_le_bytes([bytes[cursor + 30], bytes[cursor + 31]]) as usize;
        let comment = u16::from_le_bytes([bytes[cursor + 32], bytes[cursor + 33]]) as usize;
        cursor += 46 + name + extra + comment;
    }
    headers
}

fn local_headers(bytes: &[u8], central: &[usize]) -> Vec<usize> {
    central
        .iter()
        .map(|offset| {
            u32::from_le_bytes([
                bytes[offset + 42],
                bytes[offset + 43],
                bytes[offset + 44],
                bytes[offset + 45],
            ]) as usize
        })
        .collect()
}

fn reorder_local_records(bytes: &[u8], order: &[usize]) -> Vec<u8> {
    let central = central_headers(bytes);
    let local = local_headers(bytes, &central);
    assert_eq!(order.len(), local.len());
    let central_start = central[0];
    let mut chunks = Vec::with_capacity(local.len());
    for (index, start) in local.iter().copied().enumerate() {
        let end = local.get(index + 1).copied().unwrap_or(central_start);
        chunks.push(bytes[start..end].to_vec());
    }
    let mut result = Vec::with_capacity(bytes.len());
    let mut new_offsets = vec![0_u32; chunks.len()];
    for original_index in order.iter().copied() {
        new_offsets[original_index] = u32::try_from(result.len()).expect("classic local offset");
        result.extend_from_slice(&chunks[original_index]);
    }
    result.extend_from_slice(&bytes[central_start..]);
    for (index, central_start) in central.iter().copied().enumerate() {
        result[central_start + 42..central_start + 46]
            .copy_from_slice(&new_offsets[index].to_le_bytes());
    }
    result
}

fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
