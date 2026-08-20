use std::fs::File;
use std::os::fd::{AsFd, OwnedFd};
use std::path::{Component, Path};

use rustix::fs::{self, AtFlags, FileType, FlockOperation, Mode, OFlags, Stat};

use crate::setup::SetupError;

const DIRECTORY_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::DIRECTORY)
    .union(OFlags::NOFOLLOW)
    .union(OFlags::CLOEXEC);
const FILE_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::NOFOLLOW)
    .union(OFlags::NONBLOCK)
    .union(OFlags::CLOEXEC);

pub(super) struct Directory {
    pub(super) fd: OwnedFd,
    stat: Stat,
}

impl Directory {
    pub(super) fn same_object(&self, other: &Self) -> bool {
        self.stat.st_dev == other.stat.st_dev && self.stat.st_ino == other.stat.st_ino
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Identity {
    pub(super) device: u64,
    pub(super) inode: u64,
    pub(super) size: u64,
    file_type: FileType,
    mode: u32,
    owner: u32,
    links: u64,
}

pub(in crate::setup) struct SetupLock {
    file: File,
    identity: Identity,
}

pub(super) fn open_or_create_root(path: &Path) -> Result<(Directory, bool), SetupError> {
    let components = authorized_components(path)?;
    let root_fd = fs::open("/", DIRECTORY_FLAGS, Mode::empty())
        .map_err(|error| io_error("open filesystem root", error))?;
    let root_stat = fs::fstat(&root_fd).map_err(|error| io_error("inspect root", error))?;
    let mut current = Directory {
        fd: root_fd,
        stat: root_stat,
    };
    let mut created = false;
    for (index, component) in components.iter().enumerate() {
        let final_component = index + 1 == components.len();
        current = match open_directory(&current, component, final_component) {
            Ok(directory) => directory,
            Err(SetupError::Missing) if final_component => {
                match fs::mkdirat(
                    current.fd.as_fd(),
                    component.as_str(),
                    Mode::from_raw_mode(0o700),
                ) {
                    Ok(()) => {
                        created = true;
                        finish_created_directory(&current, component)?
                    }
                    Err(rustix::io::Errno::EXIST) => {
                        checked_directory(open_directory(&current, component, false)?, true)?
                    }
                    Err(error) => return Err(io_error("create state root", error)),
                }
            }
            Err(error) => return Err(error),
        };
    }
    Ok((current, created))
}

pub(super) fn reopen_root(path: &Path) -> Result<Directory, SetupError> {
    let components = authorized_components(path)?;
    let root_fd = fs::open("/", DIRECTORY_FLAGS, Mode::empty())
        .map_err(|error| io_error("open filesystem root", error))?;
    let root_stat = fs::fstat(&root_fd).map_err(|error| io_error("inspect root", error))?;
    let mut current = Directory {
        fd: root_fd,
        stat: root_stat,
    };
    for (index, component) in components.iter().enumerate() {
        current = open_directory(&current, component, index + 1 == components.len())?;
    }
    Ok(current)
}

pub(super) fn verify_root(path: &Path, expected: &Directory) -> Result<(), SetupError> {
    let live = reopen_root(path)?;
    if !live.same_object(expected) {
        return Err(SetupError::Filesystem(
            "named state root changed during setup".to_owned(),
        ));
    }
    Ok(())
}

fn open_directory(
    parent: &Directory,
    name: &str,
    require_private: bool,
) -> Result<Directory, SetupError> {
    let named = match fs::statat(parent.fd.as_fd(), name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) => stat,
        Err(rustix::io::Errno::NOENT) => return Err(SetupError::Missing),
        Err(error) => return Err(io_error("inspect directory entry", error)),
    };
    if FileType::from_raw_mode(named.st_mode) != FileType::Directory {
        return Err(SetupError::Filesystem(
            "expected directory entry has the wrong type".to_owned(),
        ));
    }
    let fd = fs::openat(parent.fd.as_fd(), name, DIRECTORY_FLAGS, Mode::empty())
        .map_err(|error| io_error("open directory entry", error))?;
    let stat = fs::fstat(&fd).map_err(|error| io_error("inspect directory", error))?;
    if stat.st_dev != named.st_dev || stat.st_ino != named.st_ino {
        return Err(SetupError::Filesystem(
            "directory changed while opening".to_owned(),
        ));
    }
    checked_directory(Directory { fd, stat }, require_private)
}

fn checked_directory(directory: Directory, require_private: bool) -> Result<Directory, SetupError> {
    let mode = directory.stat.st_mode as u32 & 0o7777;
    if require_private
        && (mode != 0o700
            || directory.stat.st_uid != rustix::process::geteuid().as_raw()
            || directory.stat.st_nlink < 1)
    {
        return Err(SetupError::Filesystem(
            "state directory is not owned mode 0700".to_owned(),
        ));
    }
    Ok(directory)
}

fn finish_created_directory(parent: &Directory, name: &str) -> Result<Directory, SetupError> {
    let created = fs::statat(parent.fd.as_fd(), name, AtFlags::SYMLINK_NOFOLLOW)
        .map_err(|error| io_error("inspect newly created setup directory", error))?;
    if FileType::from_raw_mode(created.st_mode) != FileType::Directory
        || created.st_uid != rustix::process::geteuid().as_raw()
        || created.st_nlink < 1
        || created.st_dev != parent.stat.st_dev
    {
        return Err(SetupError::Filesystem(
            "new setup directory changed before mode normalization".to_owned(),
        ));
    }
    fs::chmodat(
        parent.fd.as_fd(),
        name,
        Mode::from_raw_mode(0o700),
        AtFlags::SYMLINK_NOFOLLOW,
    )
    .map_err(|error| io_error("set setup-directory mode", error))?;
    let directory = open_directory(parent, name, false)?;
    if directory.stat.st_dev != created.st_dev
        || directory.stat.st_ino != created.st_ino
        || directory.stat.st_uid != created.st_uid
        || directory.stat.st_nlink != created.st_nlink
    {
        return Err(SetupError::Filesystem(
            "new setup directory changed during mode normalization".to_owned(),
        ));
    }
    let directory = checked_directory(directory, true)?;
    sync_directory(&directory)?;
    sync_directory(parent)?;
    verify_directory(parent, name, &directory)?;
    Ok(directory)
}

fn verify_directory(
    parent: &Directory,
    name: &str,
    expected: &Directory,
) -> Result<(), SetupError> {
    let actual = open_directory(parent, name, true)?;
    if !actual.same_object(expected) {
        return Err(SetupError::Filesystem(
            "named setup directory changed during the transaction".to_owned(),
        ));
    }
    Ok(())
}

pub(super) fn acquire_lock(root: &Directory, name: &str) -> Result<SetupLock, SetupError> {
    let file = match try_create_private_file(root, name)? {
        Some(file) => file,
        None => open_private_file(root, name, true)?,
    };
    fs::flock(&file, FlockOperation::NonBlockingLockExclusive).map_err(|error| {
        if error == rustix::io::Errno::WOULDBLOCK {
            SetupError::Busy
        } else {
            io_error("acquire setup lock", error)
        }
    })?;
    let identity = private_identity(&file, root)?;
    if identity.size != 0 {
        return Err(SetupError::Filesystem(
            "setup lock file must remain empty".to_owned(),
        ));
    }
    verify_named(root, name, identity)?;
    Ok(SetupLock { file, identity })
}

pub(super) fn verify_lock(
    root: &Directory,
    name: &str,
    lock: &SetupLock,
) -> Result<(), SetupError> {
    if private_identity(&lock.file, root)? != lock.identity {
        return Err(SetupError::Filesystem(
            "held setup lock changed during the transaction".to_owned(),
        ));
    }
    verify_named(root, name, lock.identity)
}

pub(super) fn full_sync_lock(lock: &SetupLock) -> Result<(), SetupError> {
    full_sync(&lock.file)
}

pub(super) fn create_private_file(root: &Directory, name: &str) -> Result<File, SetupError> {
    try_create_private_file(root, name)?
        .ok_or_else(|| SetupError::Filesystem("private setup file already exists".to_owned()))
}

fn try_create_private_file(root: &Directory, name: &str) -> Result<Option<File>, SetupError> {
    let flags = OFlags::RDWR
        .union(OFlags::CREATE)
        .union(OFlags::EXCL)
        .union(OFlags::NOFOLLOW)
        .union(OFlags::CLOEXEC);
    let fd = match fs::openat(root.fd.as_fd(), name, flags, Mode::from_raw_mode(0o600)) {
        Ok(fd) => fd,
        Err(rustix::io::Errno::EXIST) => return Ok(None),
        Err(error) => return Err(io_error("create private setup file", error)),
    };
    let file = File::from(fd);
    fs::fchmod(&file, Mode::from_raw_mode(0o600))
        .map_err(|error| io_error("set private setup-file mode", error))?;
    let identity = private_identity(&file, root)?;
    verify_named(root, name, identity)?;
    full_sync(&file)?;
    sync_directory(root)?;
    verify_named(root, name, identity)?;
    Ok(Some(file))
}

pub(super) fn open_private_file(
    root: &Directory,
    name: &str,
    writable: bool,
) -> Result<File, SetupError> {
    let flags = if writable {
        FILE_FLAGS.union(OFlags::RDWR)
    } else {
        FILE_FLAGS
    };
    let named = match fs::statat(root.fd.as_fd(), name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) => stat,
        Err(rustix::io::Errno::NOENT) => return Err(SetupError::Missing),
        Err(error) => return Err(io_error("inspect setup file", error)),
    };
    let named_identity = private_identity_from_stat(&named, root)?;
    let fd = fs::openat(root.fd.as_fd(), name, flags, Mode::empty())
        .map_err(|error| io_error("open setup file", error))?;
    let file = File::from(fd);
    let opened = private_identity(&file, root)?;
    if opened != named_identity {
        return Err(SetupError::Filesystem(
            "setup file changed while opening".to_owned(),
        ));
    }
    Ok(file)
}

pub(super) fn private_identity(file: &File, root: &Directory) -> Result<Identity, SetupError> {
    let stat = fs::fstat(file).map_err(|error| io_error("inspect setup file", error))?;
    private_identity_from_stat(&stat, root)
}

fn private_identity_from_stat(stat: &Stat, root: &Directory) -> Result<Identity, SetupError> {
    let mode = stat.st_mode as u32 & 0o7777;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile
        || mode != 0o600
        || stat.st_uid != rustix::process::geteuid().as_raw()
        || stat.st_nlink != 1
        || stat.st_dev != root.stat.st_dev
        || stat.st_size < 0
    {
        return Err(SetupError::Filesystem(
            "setup file is not an owned single-link mode-0600 regular file".to_owned(),
        ));
    }
    Ok(Identity {
        device: stat.st_dev as u64,
        inode: stat.st_ino as u64,
        size: stat.st_size as u64,
        file_type: FileType::from_raw_mode(stat.st_mode),
        mode,
        owner: stat.st_uid,
        links: stat.st_nlink as u64,
    })
}

pub(super) fn entry_identity(root: &Directory, name: &str) -> Result<Option<Identity>, SetupError> {
    match fs::statat(root.fd.as_fd(), name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) => Ok(Some(Identity {
            device: stat.st_dev as u64,
            inode: stat.st_ino as u64,
            size: stat.st_size as u64,
            file_type: FileType::from_raw_mode(stat.st_mode),
            mode: stat.st_mode as u32 & 0o7777,
            owner: stat.st_uid,
            links: stat.st_nlink as u64,
        })),
        Err(rustix::io::Errno::NOENT) => Ok(None),
        Err(error) => Err(io_error("inspect named setup file", error)),
    }
}

pub(super) fn verify_named(
    root: &Directory,
    name: &str,
    expected: Identity,
) -> Result<(), SetupError> {
    let actual = entry_identity(root, name)?.ok_or(SetupError::Missing)?;
    if actual != expected {
        return Err(SetupError::Filesystem(
            "named setup file changed during the transaction".to_owned(),
        ));
    }
    Ok(())
}

fn authorized_components(path: &Path) -> Result<Vec<String>, SetupError> {
    if !path.is_absolute() || path == Path::new("/") || path.as_os_str().len() > 1024 {
        return Err(SetupError::Input(
            "setup root must be an absolute non-root path of at most 1024 bytes".to_owned(),
        ));
    }
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            Component::RootDir => {}
            Component::Normal(value) => {
                let value = value.to_str().ok_or_else(|| {
                    SetupError::Input("setup root must be valid UTF-8".to_owned())
                })?;
                if value.is_empty() || value.len() > 255 || value.chars().any(char::is_control) {
                    return Err(SetupError::Input(
                        "setup root has an invalid path component".to_owned(),
                    ));
                }
                components.push(value.to_owned());
            }
            _ => {
                return Err(SetupError::Input(
                    "setup root may not contain relative components".to_owned(),
                ));
            }
        }
    }
    if components.is_empty() {
        return Err(SetupError::Input("setup root has no components".to_owned()));
    }
    Ok(components)
}

pub(super) fn sync_directory(directory: &Directory) -> Result<(), SetupError> {
    fs::fsync(&directory.fd).map_err(|error| io_error("sync setup directory", error))
}

pub(super) fn full_sync(file: &File) -> Result<(), SetupError> {
    #[cfg(target_os = "macos")]
    {
        fs::fcntl_fullfsync(file).map_err(|error| io_error("full-sync setup file", error))
    }
    #[cfg(not(target_os = "macos"))]
    {
        fs::fsync(file).map_err(|error| io_error("sync setup file", error))
    }
}

pub(super) fn io_error(operation: &'static str, error: rustix::io::Errno) -> SetupError {
    SetupError::Filesystem(format!("{operation}: {error}"))
}
