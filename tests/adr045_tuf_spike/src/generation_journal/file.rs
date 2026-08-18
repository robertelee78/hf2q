use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

use rustix::fs::{self, AtFlags, Mode, OFlags};

use super::unix::{
    identity, require_regular_policy, require_same_identity, validate_component, Directory,
    EntryIdentity,
};
use super::JournalError;

const READ_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::NOFOLLOW)
    .union(OFlags::NONBLOCK)
    .union(OFlags::CLOEXEC);

pub(super) struct ReadPrivateFile {
    pub(super) file: File,
    pub(super) bytes: Vec<u8>,
    _identity: EntryIdentity,
}

pub(super) fn read_private_file(
    parent: &Directory,
    name: &str,
    max_bytes: usize,
) -> Result<ReadPrivateFile, JournalError> {
    validate_component(name)?;
    let named =
        fs::statat(parent.fd(), name, AtFlags::SYMLINK_NOFOLLOW).map_err(JournalError::errno)?;
    require_regular_policy(&named, 0o600, parent.device())?;
    let fd =
        fs::openat(parent.fd(), name, READ_FLAGS, Mode::empty()).map_err(JournalError::errno)?;
    let mut file = File::from(fd);
    let opened = fs::fstat(&file).map_err(JournalError::errno)?;
    require_same_identity(&named, &opened)?;
    require_regular_policy(&opened, 0o600, parent.device())?;
    let opened_identity = identity(&opened);
    let mut bytes = Vec::with_capacity((opened.st_size.max(0) as usize).min(max_bytes));
    (&mut file)
        .take(max_bytes.saturating_add(1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > max_bytes {
        return Err(JournalError::Invalid("private file exceeds its bound"));
    }
    let after = fs::fstat(&file).map_err(JournalError::errno)?;
    if identity(&after) != opened_identity || bytes.len() as i64 != opened.st_size {
        return Err(JournalError::Invalid("private file changed while reading"));
    }
    Ok(ReadPrivateFile {
        file,
        bytes,
        _identity: opened_identity,
    })
}

pub(super) fn write_or_resume_private_file(
    parent: &Directory,
    name: &str,
    expected: &[u8],
) -> Result<File, JournalError> {
    validate_component(name)?;
    let create_flags = OFlags::RDWR
        | OFlags::CREATE
        | OFlags::EXCL
        | OFlags::NOFOLLOW
        | OFlags::NONBLOCK
        | OFlags::CLOEXEC;
    let (mut file, created) =
        match fs::openat(parent.fd(), name, create_flags, Mode::from_raw_mode(0o600)) {
            Ok(fd) => (File::from(fd), true),
            Err(rustix::io::Errno::EXIST) => {
                let fd = fs::openat(
                    parent.fd(),
                    name,
                    OFlags::RDWR | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
                    Mode::empty(),
                )
                .map_err(JournalError::errno)?;
                (File::from(fd), false)
            }
            Err(error) => return Err(JournalError::errno(error)),
        };
    if created {
        super::unix::sync_directory(parent)?;
    }
    let opened = fs::fstat(&file).map_err(JournalError::errno)?;
    require_regular_policy(&opened, 0o600, parent.device())?;
    if opened.st_size < 0 || opened.st_size as usize > expected.len() {
        return Err(JournalError::Invalid("partial private file is too large"));
    }
    let mut prefix = Vec::with_capacity(opened.st_size as usize);
    (&mut file)
        .take(expected.len().saturating_add(1) as u64)
        .read_to_end(&mut prefix)?;
    if !expected.starts_with(&prefix) {
        return Err(JournalError::Invalid(
            "partial private file has a conflicting prefix",
        ));
    }
    file.seek(SeekFrom::Start(prefix.len() as u64))?;
    file.write_all(&expected[prefix.len()..])?;
    fs::fsync(&file).map_err(JournalError::errno)?;
    let after = fs::fstat(&file).map_err(JournalError::errno)?;
    require_same_identity(&opened, &after)?;
    require_regular_policy(&after, 0o600, parent.device())?;
    if after.st_size as usize != expected.len() {
        return Err(JournalError::Invalid("private file write is incomplete"));
    }
    Ok(file)
}
