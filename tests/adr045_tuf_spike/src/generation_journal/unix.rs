use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs::File;
use std::os::fd::OwnedFd;
use std::path::{Component, Path};

use rustix::fs::{self, AtFlags, Dir, FileType, FlockOperation, Mode, OFlags, RenameFlags, Stat};

use super::JournalError;

const DIRECTORY_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::DIRECTORY)
    .union(OFlags::NOFOLLOW)
    .union(OFlags::CLOEXEC);

#[derive(Debug)]
pub(super) struct Directory {
    fd: OwnedFd,
    stat: Stat,
}

impl Directory {
    pub(super) fn fd(&self) -> &OwnedFd {
        &self.fd
    }

    pub(super) fn device(&self) -> u64 {
        self.stat.st_dev as u64
    }

    pub(super) fn same_object(&self, other: &Self) -> bool {
        self.stat.st_dev == other.stat.st_dev
            && self.stat.st_ino == other.stat.st_ino
            && FileType::from_raw_mode(self.stat.st_mode)
                == FileType::from_raw_mode(other.stat.st_mode)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct EntryIdentity {
    device: u64,
    inode: u64,
    file_type: FileType,
    mode: u32,
    uid: u32,
    links: u64,
    size: u64,
}

pub(super) fn open_existing_root(path: &Path) -> Result<Directory, JournalError> {
    let components = authorized_components(path)?;
    let fd = fs::open("/", DIRECTORY_FLAGS, Mode::empty()).map_err(JournalError::errno)?;
    let stat = fs::fstat(&fd).map_err(JournalError::errno)?;
    let mut current = Directory { fd, stat };
    for component in components {
        current = open_directory_policy(&current, &component, None, false, false)?;
    }
    require_directory_policy(&current.stat, Some(0o700), true, None)?;
    Ok(current)
}

pub(super) fn ensure_private_directory(
    parent: &Directory,
    name: &str,
) -> Result<Directory, JournalError> {
    match open_directory(parent, name) {
        Ok(directory) => Ok(directory),
        Err(JournalError::Missing) => {
            fs::mkdirat(parent.fd(), name, Mode::from_raw_mode(0o700))
                .or_else(|error| {
                    (error == rustix::io::Errno::EXIST)
                        .then_some(())
                        .ok_or(error)
                })
                .map_err(JournalError::errno)?;
            sync_directory(parent)?;
            open_directory(parent, name)
        }
        Err(error) => Err(error),
    }
}

pub(super) fn open_directory(parent: &Directory, name: &str) -> Result<Directory, JournalError> {
    open_directory_policy(parent, name, Some(0o700), true, true)
}

fn open_directory_policy(
    parent: &Directory,
    name: &str,
    expected_mode: Option<u32>,
    require_owner: bool,
    require_same_device: bool,
) -> Result<Directory, JournalError> {
    validate_component(name)?;
    let named = match fs::statat(parent.fd(), name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) => stat,
        Err(rustix::io::Errno::NOENT) => return Err(JournalError::Missing),
        Err(error) => return Err(JournalError::errno(error)),
    };
    if FileType::from_raw_mode(named.st_mode) != FileType::Directory {
        return Err(JournalError::Invalid(
            "expected directory has the wrong type",
        ));
    }
    let fd = fs::openat(parent.fd(), name, DIRECTORY_FLAGS, Mode::empty())
        .map_err(JournalError::errno)?;
    let opened = fs::fstat(&fd).map_err(JournalError::errno)?;
    require_same_identity(&named, &opened)?;
    require_directory_policy(
        &opened,
        expected_mode,
        require_owner,
        require_same_device.then(|| parent.device()),
    )?;
    Ok(Directory { fd, stat: opened })
}

pub(super) fn entry_identity(
    parent: &Directory,
    name: &str,
) -> Result<Option<EntryIdentity>, JournalError> {
    validate_component(name)?;
    match fs::statat(parent.fd(), name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) => Ok(Some(identity(&stat))),
        Err(rustix::io::Errno::NOENT) => Ok(None),
        Err(error) => Err(JournalError::errno(error)),
    }
}

pub(super) fn list_names(directory: &Directory) -> Result<BTreeSet<String>, JournalError> {
    let mut entries = Dir::read_from(directory.fd()).map_err(JournalError::errno)?;
    let mut names = BTreeSet::new();
    while let Some(entry) = entries.read() {
        let entry = entry.map_err(JournalError::errno)?;
        let bytes = entry.file_name().to_bytes();
        if bytes == b"." || bytes == b".." {
            continue;
        }
        let name = std::str::from_utf8(bytes)
            .map_err(|_| JournalError::Invalid("directory name is not UTF-8"))?;
        validate_component(name)?;
        if !names.insert(name.to_string()) {
            return Err(JournalError::Invalid("directory has duplicate names"));
        }
    }
    Ok(names)
}

pub(super) fn acquire_nonblocking_lock(
    update: &Directory,
) -> Result<(File, EntryIdentity), JournalError> {
    const NAME: &str = "install.lock";
    let flags = OFlags::RDWR
        | OFlags::CREATE
        | OFlags::EXCL
        | OFlags::NOFOLLOW
        | OFlags::NONBLOCK
        | OFlags::CLOEXEC;
    let fd = match fs::openat(update.fd(), NAME, flags, Mode::from_raw_mode(0o600)) {
        Ok(fd) => {
            sync_directory(update)?;
            fd
        }
        Err(rustix::io::Errno::EXIST) => fs::openat(
            update.fd(),
            NAME,
            OFlags::RDWR | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(JournalError::errno)?,
        Err(error) => return Err(JournalError::errno(error)),
    };
    let file = File::from(fd);
    let opened = fs::fstat(&file).map_err(JournalError::errno)?;
    require_regular_policy(&opened, 0o600, update.device())?;
    match fs::flock(&file, FlockOperation::NonBlockingLockExclusive) {
        Ok(()) => {}
        Err(rustix::io::Errno::WOULDBLOCK) => return Err(JournalError::Busy),
        Err(error) => return Err(JournalError::errno(error)),
    }
    let named =
        fs::statat(update.fd(), NAME, AtFlags::SYMLINK_NOFOLLOW).map_err(JournalError::errno)?;
    require_same_identity(&opened, &named)?;
    Ok((file, identity(&opened)))
}

pub(super) fn verify_named_identity(
    parent: &Directory,
    name: &str,
    expected: EntryIdentity,
) -> Result<(), JournalError> {
    let named =
        fs::statat(parent.fd(), name, AtFlags::SYMLINK_NOFOLLOW).map_err(JournalError::errno)?;
    if identity(&named) != expected {
        return Err(JournalError::Invalid(
            "named entry changed after verification",
        ));
    }
    Ok(())
}

pub(super) fn regular_file_identity(
    file: &File,
    expected_device: u64,
) -> Result<EntryIdentity, JournalError> {
    let stat = fs::fstat(file).map_err(JournalError::errno)?;
    require_regular_policy(&stat, 0o600, expected_device)?;
    Ok(identity(&stat))
}

pub(super) fn rename_noreplace(
    from_parent: &Directory,
    from: &str,
    to_parent: &Directory,
    to: &str,
) -> Result<(), JournalError> {
    validate_component(from)?;
    validate_component(to)?;
    fs::renameat_with(
        from_parent.fd(),
        from,
        to_parent.fd(),
        to,
        RenameFlags::NOREPLACE,
    )
    .map_err(JournalError::errno)
}

pub(super) fn rename_replace(parent: &Directory, from: &str, to: &str) -> Result<(), JournalError> {
    validate_component(from)?;
    validate_component(to)?;
    fs::renameat(parent.fd(), from, parent.fd(), to).map_err(JournalError::errno)
}

pub(super) fn sync_directory(directory: &Directory) -> Result<(), JournalError> {
    fs::fsync(directory.fd()).map_err(JournalError::errno)
}

pub(super) fn full_sync_file(file: &File) -> Result<(), JournalError> {
    #[cfg(target_os = "macos")]
    {
        fs::fcntl_fullfsync(file).map_err(JournalError::errno)
    }
    #[cfg(not(target_os = "macos"))]
    {
        fs::fsync(file).map_err(JournalError::errno)
    }
}

pub(super) fn require_regular_policy(
    stat: &Stat,
    expected_mode: u32,
    expected_device: u64,
) -> Result<(), JournalError> {
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile
        || stat.st_uid != rustix::process::geteuid().as_raw()
        || stat.st_nlink != 1
        || permission_bits(stat) != expected_mode
        || stat.st_dev as u64 != expected_device
        || stat.st_size < 0
    {
        return Err(JournalError::Invalid(
            "regular file type, owner, mode, links, or device is invalid",
        ));
    }
    Ok(())
}

pub(super) fn require_same_identity(first: &Stat, second: &Stat) -> Result<(), JournalError> {
    if first.st_dev != second.st_dev
        || first.st_ino != second.st_ino
        || FileType::from_raw_mode(first.st_mode) != FileType::from_raw_mode(second.st_mode)
    {
        return Err(JournalError::Invalid(
            "entry identity changed while opening",
        ));
    }
    Ok(())
}

pub(super) fn identity(stat: &Stat) -> EntryIdentity {
    EntryIdentity {
        device: stat.st_dev as u64,
        inode: stat.st_ino,
        file_type: FileType::from_raw_mode(stat.st_mode),
        mode: permission_bits(stat),
        uid: stat.st_uid,
        links: stat.st_nlink as u64,
        size: stat.st_size.max(0) as u64,
    }
}

fn authorized_components(path: &Path) -> Result<Vec<String>, JournalError> {
    let mut saw_root = false;
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            Component::RootDir if !saw_root && components.is_empty() => saw_root = true,
            Component::Normal(value) if saw_root => {
                let value = utf8_component(value)?;
                validate_component(value)?;
                components.push(value.to_string());
            }
            _ => {
                return Err(JournalError::Invalid(
                    "root path is not absolute and normalized",
                ))
            }
        }
    }
    if !saw_root || components.is_empty() {
        return Err(JournalError::Invalid(
            "filesystem root is not an authorized journal root",
        ));
    }
    Ok(components)
}

fn utf8_component(value: &OsStr) -> Result<&str, JournalError> {
    value
        .to_str()
        .ok_or(JournalError::Invalid("root path component is not UTF-8"))
}

pub(super) fn validate_component(value: &str) -> Result<(), JournalError> {
    if value.is_empty()
        || value == "."
        || value == ".."
        || value.len() > 255
        || value.contains('/')
        || value.contains('\\')
        || value.bytes().any(|byte| byte == 0)
        || value.chars().any(char::is_control)
    {
        return Err(JournalError::Invalid("filesystem component is invalid"));
    }
    Ok(())
}

fn require_directory_policy(
    stat: &Stat,
    expected_mode: Option<u32>,
    require_owner: bool,
    expected_device: Option<u64>,
) -> Result<(), JournalError> {
    if FileType::from_raw_mode(stat.st_mode) != FileType::Directory
        || (require_owner && stat.st_uid != rustix::process::geteuid().as_raw())
        || expected_mode.is_some_and(|mode| permission_bits(stat) != mode)
        || expected_device.is_some_and(|device| stat.st_dev as u64 != device)
    {
        return Err(JournalError::Invalid(
            "directory type, owner, mode, or device is invalid",
        ));
    }
    Ok(())
}

fn permission_bits(stat: &Stat) -> u32 {
    Mode::from_raw_mode(stat.st_mode).as_raw_mode() as u32 & 0o7777
}
