use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;

use sha2::{Digest, Sha256};

use super::{identity_with_size, ExpectedFile, ExtractionError};
use crate::distribution::install_state::unix::{self, EntryIdentity};

#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(test)]
static ABORT_AFTER_EXTRACTION_WRITE: AtomicBool = AtomicBool::new(false);

/// Reconstruct one exact authenticated file without deleting or truncating it.
///
/// Existing bytes are compared first. Any mismatching range is replaced only
/// after the stage's exact name/type/owner/mode/link/device identity has been
/// established under the held installation lock. This makes a length-ahead-
/// of-data power-crash residue repairable while preserving fail-closed layout
/// handling for anything outside this single reserved private file.
pub(super) fn reconstruct_exact(
    file: &File,
    before: EntryIdentity,
    source: &mut dyn Read,
    expected: &ExpectedFile,
) -> Result<(), ExtractionError> {
    let mut total = 0_u64;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = source.read(&mut buffer).map_err(ExtractionError::read_io)?;
        if count == 0 {
            break;
        }
        let next = total
            .checked_add(count as u64)
            .ok_or(ExtractionError::Integrity)?;
        if next > expected.size {
            return Err(ExtractionError::Integrity);
        }
        hasher.update(&buffer[..count]);
        let prefix = before.size.saturating_sub(total).min(count as u64) as usize;
        if prefix != 0 {
            let mut existing = vec![0_u8; prefix];
            read_exact_at(file, &mut existing, total)?;
            if existing != buffer[..prefix] {
                write_all_at(file, &buffer[..prefix], total)?;
            }
        }
        if prefix < count {
            write_all_at(file, &buffer[prefix..count], total + prefix as u64)?;
        }
        total = next;
    }
    if total != expected.size || hex::encode(hasher.finalize()) != expected.sha256 {
        return Err(ExtractionError::Integrity);
    }
    unix::full_sync_file(file)?;
    verify_file(file, identity_with_size(before, expected.size), expected)
}

pub(super) fn verify_file(
    file: &File,
    expected_identity: EntryIdentity,
    expected: &ExpectedFile,
) -> Result<(), ExtractionError> {
    let actual = unix::regular_file_identity(file, expected_identity.device)?;
    if actual != expected_identity || actual.size != expected.size {
        return Err(ExtractionError::Integrity);
    }
    let mut total = 0_u64;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read_at(&mut buffer, total)
            .map_err(ExtractionError::read_io)?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .ok_or(ExtractionError::Integrity)?;
        if total > expected.size {
            return Err(ExtractionError::Integrity);
        }
        hasher.update(&buffer[..count]);
    }
    let after = unix::regular_file_identity(file, expected_identity.device)?;
    if after != actual
        || total != expected.size
        || hex::encode(hasher.finalize()) != expected.sha256
    {
        return Err(ExtractionError::Integrity);
    }
    Ok(())
}

fn read_exact_at(
    file: &File,
    mut bytes: &mut [u8],
    mut offset: u64,
) -> Result<(), ExtractionError> {
    while !bytes.is_empty() {
        let count = file
            .read_at(bytes, offset)
            .map_err(ExtractionError::read_io)?;
        if count == 0 {
            return Err(ExtractionError::Integrity);
        }
        offset += count as u64;
        bytes = &mut bytes[count..];
    }
    Ok(())
}

fn write_all_at(file: &File, mut bytes: &[u8], mut offset: u64) -> Result<(), ExtractionError> {
    while !bytes.is_empty() {
        let count = file
            .write_at(bytes, offset)
            .map_err(ExtractionError::write_io)?;
        if count == 0 {
            return Err(ExtractionError::write_io(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "release extraction made no write progress",
            )));
        }
        #[cfg(test)]
        if ABORT_AFTER_EXTRACTION_WRITE.swap(false, Ordering::SeqCst) {
            std::process::abort();
        }
        offset += count as u64;
        bytes = &bytes[count..];
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn abort_after_next_extraction_write() {
    ABORT_AFTER_EXTRACTION_WRITE.store(true, Ordering::SeqCst);
}
