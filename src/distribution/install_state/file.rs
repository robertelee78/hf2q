use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

use rustix::fs::{self, AtFlags, Mode, OFlags};
use sha2::{Digest, Sha256};

use super::unix::{
    self, identity, require_regular_policy, require_same_identity, validate_component, Directory,
    EntryIdentity,
};
use super::InstallStateError;

const READ_FILE_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::NOFOLLOW)
    .union(OFlags::NONBLOCK)
    .union(OFlags::CLOEXEC);

pub(super) fn read_regular_file(
    parent: &Directory,
    name: &str,
    expected_mode: u32,
    max_bytes: usize,
) -> Result<(File, Vec<u8>, EntryIdentity), InstallStateError> {
    let (mut file, opened) = open_regular_file(parent, name, expected_mode)?;
    let mut bytes = Vec::with_capacity((opened.size as usize).min(max_bytes));
    (&mut file)
        .take(max_bytes.saturating_add(1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| InstallStateError::std_io("read bounded regular file", error))?;
    if bytes.len() > max_bytes {
        return Err(InstallStateError::InvalidLayout(
            "regular file exceeds its input bound",
        ));
    }
    revalidate_read_file(parent, &file, opened, bytes.len() as u64)?;
    Ok((file, bytes, opened))
}

pub(super) fn hash_regular_file(
    parent: &Directory,
    name: &str,
    expected_mode: u32,
    expected_size: u64,
) -> Result<(File, String), InstallStateError> {
    let (mut file, opened) = open_regular_file(parent, name, expected_mode)?;
    if opened.size != expected_size {
        return Err(InstallStateError::InvalidLayout(
            "regular file size does not match its signed inventory",
        ));
    }
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut total = 0_u64;
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| InstallStateError::std_io("hash bounded regular file", error))?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(read as u64)
            .ok_or(InstallStateError::InvalidLayout(
                "regular file size overflowed while hashing",
            ))?;
        if total > expected_size {
            return Err(InstallStateError::InvalidLayout(
                "regular file grew beyond its signed size while hashing",
            ));
        }
        hasher.update(&buffer[..read]);
    }
    revalidate_read_file(parent, &file, opened, total)?;
    Ok((file, hex::encode(hasher.finalize())))
}

pub(super) fn write_or_resume_private_file(
    parent: &Directory,
    name: &str,
    expected: &[u8],
) -> Result<File, InstallStateError> {
    write_or_resume_private_file_with_create_hook(parent, name, expected, || Ok(()))
}

pub(super) fn write_or_resume_private_file_with_create_hook(
    parent: &Directory,
    name: &str,
    expected: &[u8],
    after_create: impl FnOnce() -> Result<(), InstallStateError>,
) -> Result<File, InstallStateError> {
    validate_component(name)?;
    let create_flags = OFlags::RDWR
        | OFlags::CREATE
        | OFlags::EXCL
        | OFlags::NOFOLLOW
        | OFlags::NONBLOCK
        | OFlags::CLOEXEC;
    let (mut file, newly_created) =
        match fs::openat(parent.fd(), name, create_flags, Mode::from_raw_mode(0o600)) {
            Ok(fd) => (File::from(fd), true),
            Err(rustix::io::Errno::EXIST) => {
                let fd = fs::openat(
                    parent.fd(),
                    name,
                    OFlags::RDWR | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
                    Mode::empty(),
                )
                .map_err(|error| InstallStateError::io("open partial private file", error))?;
                (File::from(fd), false)
            }
            Err(error) => return Err(InstallStateError::io("create partial private file", error)),
        };
    if newly_created {
        unix::sync_directory(parent)?;
        after_create()?;
    }

    let opened = fs::fstat(&file)
        .map_err(|error| InstallStateError::io("inspect partial private file", error))?;
    require_regular_policy(&opened, 0o600, parent.device())?;
    if opened.st_size < 0 || opened.st_size as usize > expected.len() {
        return Err(InstallStateError::InvalidLayout(
            "partial private file exceeds its expected bytes",
        ));
    }
    let mut prefix = Vec::with_capacity(opened.st_size as usize);
    (&mut file)
        .take(expected.len().saturating_add(1) as u64)
        .read_to_end(&mut prefix)
        .map_err(|error| InstallStateError::std_io("read partial private file", error))?;
    if !expected.starts_with(&prefix) {
        return Err(InstallStateError::InvalidLayout(
            "partial private file conflicts with expected bytes",
        ));
    }
    file.seek(SeekFrom::Start(prefix.len() as u64))
        .map_err(|error| InstallStateError::std_io("seek partial private file", error))?;
    file.write_all(&expected[prefix.len()..])
        .map_err(|error| InstallStateError::std_io("resume partial private file", error))?;
    fs::fsync(&file).map_err(|error| InstallStateError::io("sync private file", error))?;
    let after = fs::fstat(&file)
        .map_err(|error| InstallStateError::io("reinspect partial private file", error))?;
    require_same_identity(
        &opened,
        &after,
        "partial private file changed while resuming",
    )?;
    require_regular_policy(&after, 0o600, parent.device())?;
    if after.st_size as usize != expected.len() {
        return Err(InstallStateError::InvalidLayout(
            "resumed private file has the wrong size",
        ));
    }
    Ok(file)
}

fn open_regular_file(
    parent: &Directory,
    name: &str,
    expected_mode: u32,
) -> Result<(File, EntryIdentity), InstallStateError> {
    validate_component(name)?;
    let named = fs::statat(parent.fd(), name, AtFlags::SYMLINK_NOFOLLOW)
        .map_err(|error| InstallStateError::io("inspect regular file", error))?;
    require_regular_policy(&named, expected_mode, parent.device())?;
    let fd = fs::openat(parent.fd(), name, READ_FILE_FLAGS, Mode::empty())
        .map_err(|error| InstallStateError::io("open regular file", error))?;
    let opened = fs::fstat(&fd)
        .map_err(|error| InstallStateError::io("inspect opened regular file", error))?;
    require_same_identity(&named, &opened, "regular file changed while opening")?;
    require_regular_policy(&opened, expected_mode, parent.device())?;
    Ok((File::from(fd), identity(&opened)))
}

fn revalidate_read_file(
    parent: &Directory,
    file: &File,
    opened: EntryIdentity,
    bytes_read: u64,
) -> Result<(), InstallStateError> {
    let after = fs::fstat(file)
        .map_err(|error| InstallStateError::io("reinspect opened regular file", error))?;
    require_regular_policy(&after, opened.mode, parent.device())?;
    if identity(&after) != opened || bytes_read != opened.size {
        return Err(InstallStateError::InvalidLayout(
            "regular file size changed while reading",
        ));
    }
    Ok(())
}
