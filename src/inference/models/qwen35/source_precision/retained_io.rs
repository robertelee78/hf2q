use std::fs::File;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Component, Path};

use anyhow::{ensure, Context, Result};
use rustix::fs::{self, Mode, OFlags};
use sha1::Sha1;
use sha2::{Digest, Sha256};

use crate::core::integrity::ShardIntegrity;

use super::types::SOURCE_READ_CHUNK_BYTES;

const RETAINED_DIRECTORY_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::DIRECTORY)
    .union(OFlags::NOFOLLOW)
    .union(OFlags::CLOEXEC);
const RETAINED_FILE_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::NOFOLLOW)
    .union(OFlags::NONBLOCK)
    .union(OFlags::CLOEXEC);

#[derive(Debug)]
pub(super) struct RetainedSourceFile {
    pub(super) filename: String,
    pub(super) file: File,
    pub(super) byte_len: u64,
    pub(super) device: u64,
    pub(super) inode: u64,
    pub(super) sha256: String,
}

pub(super) fn require_safe_leaf(name: &str) -> Result<()> {
    let mut components = Path::new(name).components();
    ensure!(
        !name.is_empty()
            && name.is_ascii()
            && !name.bytes().any(|byte| byte.is_ascii_control())
            && matches!(components.next(), Some(Component::Normal(_)))
            && components.next().is_none(),
        "source snapshot v1 requires a safe root-level filename"
    );
    Ok(())
}

pub(super) fn open_retained_directory(path: &Path) -> Result<File> {
    let descriptor = fs::open(path, RETAINED_DIRECTORY_FLAGS, Mode::empty())
        .with_context(|| format!("open retained source directory {}", path.display()))?;
    let file = File::from(descriptor);
    ensure!(
        file.metadata()?.file_type().is_dir(),
        "source root is not a retained directory"
    );
    Ok(file)
}

pub(super) fn open_retained_file(root: &File, filename: &str) -> Result<File> {
    require_safe_leaf(filename)?;
    let descriptor = fs::openat(root, filename, RETAINED_FILE_FLAGS, Mode::empty())
        .with_context(|| format!("open retained source file {filename}"))?;
    Ok(File::from(descriptor))
}

pub(super) fn open_and_read_config(
    root: &File,
    record: &ShardIntegrity,
) -> Result<(RetainedSourceFile, Vec<u8>)> {
    let file = open_retained_file(root, &record.filename)?;
    let metadata = file.metadata()?;
    ensure!(
        metadata.file_type().is_file() && metadata.len() == record.bytes,
        "source config is not the expected regular file"
    );
    let identity = (metadata.dev(), metadata.ino());
    let mut bytes = vec![0_u8; usize::try_from(record.bytes)?];
    read_exact_at(&file, &mut bytes, 0)?;
    let sha256 = hex::encode(Sha256::digest(&bytes));
    ensure!(
        source_record_matches(record, &bytes, &sha256),
        "retained config differs from its verified source record"
    );
    let after = file.metadata()?;
    ensure!(
        after.file_type().is_file()
            && after.len() == record.bytes
            && (after.dev(), after.ino()) == identity,
        "source config changed identity during verification"
    );
    Ok((
        RetainedSourceFile {
            filename: record.filename.clone(),
            file,
            byte_len: record.bytes,
            device: identity.0,
            inode: identity.1,
            sha256,
        },
        bytes,
    ))
}

fn source_record_matches(record: &ShardIntegrity, bytes: &[u8], sha256: &str) -> bool {
    if record
        .sha256
        .as_deref()
        .is_some_and(|expected| expected.eq_ignore_ascii_case(sha256))
    {
        return true;
    }
    if record.is_lfs {
        return false;
    }
    let mut git = Sha1::new();
    git.update(format!("blob {}\0", bytes.len()).as_bytes());
    git.update(bytes);
    hex::encode(git.finalize()).eq_ignore_ascii_case(record.hf_etag.trim().trim_matches('"'))
}

pub(super) fn read_exact_at(file: &File, mut output: &mut [u8], mut offset: u64) -> Result<()> {
    while !output.is_empty() {
        let read = file.read_at(output, offset)?;
        ensure!(
            read != 0,
            "retained source file ended before its declared length"
        );
        offset = offset
            .checked_add(u64::try_from(read).unwrap())
            .context("retained source read offset overflow")?;
        output = &mut output[read..];
    }
    Ok(())
}

pub(super) fn hash_region(
    file: &File,
    offset: u64,
    byte_len: u64,
    whole_file_hasher: &mut Sha256,
    scratch: &mut [u8],
) -> Result<String> {
    ensure!(!scratch.is_empty(), "source hash scratch cannot be empty");
    let mut hasher = Sha256::new();
    let mut remaining = byte_len;
    let mut position = offset;
    while remaining != 0 {
        let wanted = usize::try_from(remaining.min(scratch.len() as u64)).unwrap();
        read_exact_at(file, &mut scratch[..wanted], position)?;
        hasher.update(&scratch[..wanted]);
        whole_file_hasher.update(&scratch[..wanted]);
        let consumed = u64::try_from(wanted).unwrap();
        position = position
            .checked_add(consumed)
            .context("retained source hash offset overflow")?;
        remaining -= consumed;
    }
    Ok(hex::encode(hasher.finalize()))
}

pub(super) fn require_same_file(source: &RetainedSourceFile) -> Result<()> {
    let metadata = source.file.metadata()?;
    ensure!(
        metadata.file_type().is_file()
            && metadata.len() == source.byte_len
            && metadata.dev() == source.device
            && metadata.ino() == source.inode,
        "retained source file {} changed identity",
        source.filename
    );
    Ok(())
}

pub(super) fn hash_retained_file(source: &RetainedSourceFile) -> Result<String> {
    require_same_file(source)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; SOURCE_READ_CHUNK_BYTES];
    let mut remaining = source.byte_len;
    let mut offset = 0_u64;
    while remaining != 0 {
        let wanted = usize::try_from(remaining.min(buffer.len() as u64)).unwrap();
        read_exact_at(&source.file, &mut buffer[..wanted], offset)?;
        hasher.update(&buffer[..wanted]);
        let consumed = u64::try_from(wanted).unwrap();
        offset = offset
            .checked_add(consumed)
            .context("retained source hash offset overflow")?;
        remaining -= consumed;
    }
    require_same_file(source)?;
    Ok(hex::encode(hasher.finalize()))
}

pub(super) fn visit_hashed_tensor_region<F>(
    source: &RetainedSourceFile,
    payload_offset: u64,
    byte_len: u64,
    expected_sha256: &str,
    scratch: &mut Vec<u8>,
    mut visit: F,
) -> Result<()>
where
    F: FnMut(usize, &[u8]) -> Result<()>,
{
    ensure!(byte_len % 2 == 0, "source tensor has an odd byte length");
    require_same_file(source)?;
    if scratch.len() != SOURCE_READ_CHUNK_BYTES {
        scratch.resize(SOURCE_READ_CHUNK_BYTES, 0);
    }
    let mut hasher = Sha256::new();
    let mut remaining = byte_len;
    let mut offset = payload_offset;
    let mut element_offset = 0_usize;
    while remaining != 0 {
        let wanted = usize::try_from(remaining.min(scratch.len() as u64)).unwrap();
        ensure!(wanted % 2 == 0, "source tensor has an odd read chunk");
        read_exact_at(&source.file, &mut scratch[..wanted], offset)?;
        hasher.update(&scratch[..wanted]);
        visit(element_offset, &scratch[..wanted])?;
        element_offset = element_offset
            .checked_add(wanted / 2)
            .context("source tensor element offset overflow")?;
        let consumed = u64::try_from(wanted).unwrap();
        offset = offset
            .checked_add(consumed)
            .context("source tensor read offset overflow")?;
        remaining -= consumed;
    }
    ensure!(
        u64::try_from(element_offset)?
            .checked_mul(2)
            .context("source tensor visited byte count overflow")?
            == byte_len
            && hex::encode(hasher.finalize()) == expected_sha256,
        "source tensor changed after snapshot verification"
    );
    require_same_file(source)?;
    Ok(())
}
