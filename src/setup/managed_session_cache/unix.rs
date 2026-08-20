use std::collections::BTreeSet;
use std::fs::File;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};

use rustix::fs::{
    self, AtFlags, Dir, FileType, FlockOperation, Mode, OFlags, RenameFlags, Stat, StatVfs,
};

use super::ManagedSessionCacheError;

const DIRECTORY_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::DIRECTORY)
    .union(OFlags::NOFOLLOW)
    .union(OFlags::CLOEXEC);
const READ_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::NOFOLLOW)
    .union(OFlags::NONBLOCK)
    .union(OFlags::CLOEXEC);
const WRITE_FLAGS: OFlags = OFlags::RDWR
    .union(OFlags::NOFOLLOW)
    .union(OFlags::NONBLOCK)
    .union(OFlags::CLOEXEC);

pub(super) struct Directory {
    fd: OwnedFd,
    identity: EntryIdentity,
}

pub(super) struct StoreLock {
    file: File,
    identity: EntryIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct EntryIdentity {
    device: u64,
    inode: u64,
    size: u64,
    blocks: u64,
    file_type: FileType,
    mode: u32,
    owner: u32,
    links: u64,
}

impl Directory {
    pub(super) fn duplicate(source: BorrowedFd<'_>) -> Result<Self, ManagedSessionCacheError> {
        let fd =
            rustix::io::dup(source).map_err(|error| io("duplicate sessions directory", error))?;
        let stat = fs::fstat(&fd).map_err(|error| io("inspect sessions directory", error))?;
        let identity = directory_identity(&stat)?;
        Ok(Self { fd, identity })
    }

    pub(super) fn fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }

    pub(super) const fn device(&self) -> u64 {
        self.identity.device
    }
}

impl EntryIdentity {
    pub(super) const fn size(self) -> u64 {
        self.size
    }

    pub(super) const fn links(self) -> u64 {
        self.links
    }

    pub(super) const fn charge(self) -> u64 {
        let allocated = self.blocks.saturating_mul(512);
        if allocated > self.size {
            allocated
        } else {
            self.size
        }
    }
}

pub(super) fn ensure_directory(
    parent: &Directory,
    name: &str,
) -> Result<Directory, ManagedSessionCacheError> {
    validate_component(name)?;
    match open_directory(parent, name) {
        Ok(directory) => Ok(directory),
        Err(ManagedSessionCacheError::Missing) => {
            match fs::mkdirat(parent.fd(), name, Mode::from_raw_mode(0o700)) {
                Ok(()) => {}
                Err(rustix::io::Errno::EXIST) => return open_directory(parent, name),
                Err(error) => return Err(io("create managed cache directory", error)),
            }
            let named = fs::statat(parent.fd(), name, AtFlags::SYMLINK_NOFOLLOW)
                .map_err(|error| io("inspect new managed cache directory", error))?;
            if FileType::from_raw_mode(named.st_mode) != FileType::Directory
                || named.st_uid != rustix::process::geteuid().as_raw()
                || named.st_dev as u64 != parent.device()
                || named.st_nlink < 1
            {
                return Err(ManagedSessionCacheError::InvalidLayout(
                    "new managed cache directory changed before normalization",
                ));
            }
            fs::chmodat(
                parent.fd(),
                name,
                Mode::from_raw_mode(0o700),
                AtFlags::SYMLINK_NOFOLLOW,
            )
            .map_err(|error| io("normalize managed cache directory", error))?;
            let directory = open_directory(parent, name)?;
            let expected = stat_identity(&named)?;
            if directory.identity.device != expected.device
                || directory.identity.inode != expected.inode
                || directory.identity.owner != expected.owner
            {
                return Err(ManagedSessionCacheError::InvalidLayout(
                    "managed cache directory changed during normalization",
                ));
            }
            sync_directory(&directory)?;
            sync_directory(parent)?;
            verify_directory(parent, name, &directory)?;
            Ok(directory)
        }
        Err(error) => Err(error),
    }
}

pub(super) fn open_directory(
    parent: &Directory,
    name: &str,
) -> Result<Directory, ManagedSessionCacheError> {
    validate_component(name)?;
    let named = match fs::statat(parent.fd(), name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) => stat,
        Err(rustix::io::Errno::NOENT) => return Err(ManagedSessionCacheError::Missing),
        Err(error) => return Err(io("inspect managed cache directory", error)),
    };
    let expected = directory_identity(&named)?;
    if expected.device != parent.device() {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache directory crosses the sessions filesystem",
        ));
    }
    let fd = fs::openat(parent.fd(), name, DIRECTORY_FLAGS, Mode::empty())
        .map_err(|error| io("open managed cache directory", error))?;
    let opened = fs::fstat(&fd).map_err(|error| io("inspect opened cache directory", error))?;
    let actual = directory_identity(&opened)?;
    if actual != expected {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache directory changed while opening",
        ));
    }
    Ok(Directory {
        fd,
        identity: actual,
    })
}

pub(super) fn verify_directory(
    parent: &Directory,
    name: &str,
    expected: &Directory,
) -> Result<(), ManagedSessionCacheError> {
    let actual = open_directory(parent, name)?;
    if actual.identity.device != expected.identity.device
        || actual.identity.inode != expected.identity.inode
    {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache directory changed during the transaction",
        ));
    }
    Ok(())
}

pub(super) fn directory_charge(directory: &Directory) -> Result<u64, ManagedSessionCacheError> {
    let stat = fs::fstat(directory.fd())
        .map_err(|error| io("inspect managed cache directory charge", error))?;
    let identity = directory_identity(&stat)?;
    if identity.device != directory.identity.device || identity.inode != directory.identity.inode {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache directory changed while measuring charge",
        ));
    }
    Ok(identity.charge())
}

pub(super) fn acquire_lock(
    parent: &Directory,
    name: &str,
) -> Result<StoreLock, ManagedSessionCacheError> {
    validate_component(name)?;
    let (file, identity) = match open_private(parent, name, true) {
        Ok(pair) => pair,
        Err(ManagedSessionCacheError::Missing) => create_private(parent, name)?,
        Err(error) => return Err(error),
    };
    if identity.size != 0 {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache lock is not empty",
        ));
    }
    match fs::flock(&file, FlockOperation::NonBlockingLockExclusive) {
        Ok(()) => {}
        Err(rustix::io::Errno::WOULDBLOCK) => return Err(ManagedSessionCacheError::Busy),
        Err(error) => return Err(io("lock managed session cache", error)),
    }
    let lock = StoreLock { file, identity };
    verify_lock(parent, name, &lock)?;
    Ok(lock)
}

pub(super) fn verify_lock(
    parent: &Directory,
    name: &str,
    lock: &StoreLock,
) -> Result<(), ManagedSessionCacheError> {
    if private_identity(&lock.file, parent.device())? != lock.identity {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "held managed cache lock changed",
        ));
    }
    verify_named(parent, name, lock.identity)
}

pub(super) fn full_sync_lock(lock: &StoreLock) -> Result<(), ManagedSessionCacheError> {
    full_sync(&lock.file)
}

pub(super) fn create_private(
    parent: &Directory,
    name: &str,
) -> Result<(File, EntryIdentity), ManagedSessionCacheError> {
    validate_component(name)?;
    let fd = fs::openat(
        parent.fd(),
        name,
        WRITE_FLAGS | OFlags::CREATE | OFlags::EXCL,
        Mode::from_raw_mode(0o600),
    )
    .map_err(|error| io("create managed cache file", error))?;
    let file = File::from(fd);
    fs::fchmod(&file, Mode::from_raw_mode(0o600))
        .map_err(|error| io("normalize managed cache file", error))?;
    let identity = private_identity(&file, parent.device())?;
    verify_named(parent, name, identity)?;
    sync_directory(parent)?;
    verify_named(parent, name, identity)?;
    Ok((file, identity))
}

pub(super) fn open_private(
    parent: &Directory,
    name: &str,
    writable: bool,
) -> Result<(File, EntryIdentity), ManagedSessionCacheError> {
    validate_component(name)?;
    let named = match fs::statat(parent.fd(), name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) => stat,
        Err(rustix::io::Errno::NOENT) => return Err(ManagedSessionCacheError::Missing),
        Err(error) => return Err(io("inspect managed cache file", error)),
    };
    let expected = private_identity_from_stat(&named, parent.device())?;
    let fd = fs::openat(
        parent.fd(),
        name,
        if writable { WRITE_FLAGS } else { READ_FLAGS },
        Mode::empty(),
    )
    .map_err(|error| io("open managed cache file", error))?;
    let file = File::from(fd);
    let actual = private_identity(&file, parent.device())?;
    if actual != expected {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache file changed while opening",
        ));
    }
    Ok((file, actual))
}

pub(super) fn inspect_owned_regular(
    parent: &Directory,
    name: &str,
    allowed_links: u64,
) -> Result<EntryIdentity, ManagedSessionCacheError> {
    validate_component(name)?;
    let stat = fs::statat(parent.fd(), name, AtFlags::SYMLINK_NOFOLLOW)
        .map_err(|error| io("inspect managed cache regular file", error))?;
    let identity = stat_identity(&stat)?;
    if identity.file_type != FileType::RegularFile
        || identity.mode != 0o600
        || identity.owner != rustix::process::geteuid().as_raw()
        || identity.links != allowed_links
        || identity.device != parent.device()
    {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache regular file violates its private policy",
        ));
    }
    Ok(identity)
}

pub(super) fn identity_after_io(
    file: &File,
    parent: &Directory,
    expected: EntryIdentity,
) -> Result<EntryIdentity, ManagedSessionCacheError> {
    let actual = private_identity(file, parent.device())?;
    if actual.device != expected.device || actual.inode != expected.inode {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache file identity changed during I/O",
        ));
    }
    Ok(actual)
}

pub(super) fn verify_named(
    parent: &Directory,
    name: &str,
    expected: EntryIdentity,
) -> Result<(), ManagedSessionCacheError> {
    let stat = fs::statat(parent.fd(), name, AtFlags::SYMLINK_NOFOLLOW)
        .map_err(|error| io("rebind managed cache file", error))?;
    let actual = stat_identity(&stat)?;
    if actual != expected {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache file changed during the transaction",
        ));
    }
    Ok(())
}

pub(super) fn entry_identity(
    parent: &Directory,
    name: &str,
) -> Result<Option<EntryIdentity>, ManagedSessionCacheError> {
    validate_component(name)?;
    match fs::statat(parent.fd(), name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) => Ok(Some(stat_identity(&stat)?)),
        Err(rustix::io::Errno::NOENT) => Ok(None),
        Err(error) => Err(io("inspect managed cache entry", error)),
    }
}

pub(super) fn rename_noreplace(
    source_parent: &Directory,
    source: &str,
    target_parent: &Directory,
    target: &str,
) -> Result<bool, ManagedSessionCacheError> {
    validate_component(source)?;
    validate_component(target)?;
    match fs::renameat_with(
        source_parent.fd(),
        source,
        target_parent.fd(),
        target,
        RenameFlags::NOREPLACE,
    ) {
        Ok(()) => Ok(true),
        Err(rustix::io::Errno::EXIST) => Ok(false),
        Err(error) => Err(io("publish managed cache file no-replace", error)),
    }
}

pub(super) fn remove_file(
    parent: &Directory,
    name: &str,
    expected: EntryIdentity,
) -> Result<(), ManagedSessionCacheError> {
    verify_named(parent, name, expected)?;
    fs::unlinkat(parent.fd(), name, AtFlags::empty())
        .map_err(|error| io("remove managed cache file", error))
}

pub(super) fn list_names_bounded(
    directory: &Directory,
    maximum: usize,
) -> Result<BTreeSet<String>, ManagedSessionCacheError> {
    let mut stream = Dir::read_from(directory.fd())
        .map_err(|error| io("open managed cache inventory", error))?;
    let mut names = BTreeSet::new();
    while let Some(entry) = stream.read() {
        let entry = entry.map_err(|error| io("read managed cache inventory", error))?;
        let bytes = entry.file_name().to_bytes();
        if bytes == b"." || bytes == b".." {
            continue;
        }
        let name = std::str::from_utf8(bytes).map_err(|_| {
            ManagedSessionCacheError::InvalidLayout("managed cache name is not UTF-8")
        })?;
        validate_component(name)?;
        if names.len() >= maximum {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed cache inventory exceeds its cap",
            ));
        }
        if !names.insert(name.to_owned()) {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed cache inventory has duplicate names",
            ));
        }
    }
    Ok(names)
}

pub(super) fn sync_directory(directory: &Directory) -> Result<(), ManagedSessionCacheError> {
    fs::fsync(directory.fd()).map_err(|error| io("sync managed cache directory", error))
}

pub(super) fn full_sync(file: &File) -> Result<(), ManagedSessionCacheError> {
    #[cfg(target_os = "macos")]
    {
        fs::fcntl_fullfsync(file).map_err(|error| io("full-sync managed cache file", error))
    }
    #[cfg(not(target_os = "macos"))]
    {
        fs::fsync(file).map_err(|error| io("sync managed cache file", error))
    }
}

pub(super) fn full_sync_named(
    parent: &Directory,
    name: &str,
    expected: EntryIdentity,
) -> Result<(), ManagedSessionCacheError> {
    let (file, identity) = open_private(parent, name, true)?;
    if identity != expected {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache final changed before full-sync",
        ));
    }
    full_sync(&file)?;
    let after = identity_after_io(&file, parent, identity)?;
    if after != expected {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache final changed while full-syncing",
        ));
    }
    verify_named(parent, name, expected)
}

pub(super) fn volume_space(directory: &Directory) -> Result<StatVfs, ManagedSessionCacheError> {
    fs::fstatvfs(directory.fd()).map_err(|error| io("inspect managed cache volume", error))
}

fn private_identity(file: &File, device: u64) -> Result<EntryIdentity, ManagedSessionCacheError> {
    let stat = fs::fstat(file).map_err(|error| io("inspect managed cache file", error))?;
    private_identity_from_stat(&stat, device)
}

fn private_identity_from_stat(
    stat: &Stat,
    device: u64,
) -> Result<EntryIdentity, ManagedSessionCacheError> {
    let identity = stat_identity(stat)?;
    if identity.file_type != FileType::RegularFile
        || identity.mode != 0o600
        || identity.owner != rustix::process::geteuid().as_raw()
        || identity.links != 1
        || identity.device != device
    {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache file is not an owned single-link mode-0600 regular file",
        ));
    }
    Ok(identity)
}

fn directory_identity(stat: &Stat) -> Result<EntryIdentity, ManagedSessionCacheError> {
    let identity = stat_identity(stat)?;
    if identity.file_type != FileType::Directory
        || identity.mode != 0o700
        || identity.owner != rustix::process::geteuid().as_raw()
        || identity.links < 1
    {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache directory is not owned mode 0700",
        ));
    }
    Ok(identity)
}

fn stat_identity(stat: &Stat) -> Result<EntryIdentity, ManagedSessionCacheError> {
    if stat.st_size < 0 || stat.st_blocks < 0 {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache entry has a negative size",
        ));
    }
    Ok(EntryIdentity {
        device: stat.st_dev as u64,
        inode: stat.st_ino as u64,
        size: stat.st_size as u64,
        blocks: stat.st_blocks as u64,
        file_type: FileType::from_raw_mode(stat.st_mode),
        mode: stat.st_mode as u32 & 0o7777,
        owner: stat.st_uid,
        links: stat.st_nlink as u64,
    })
}

fn validate_component(name: &str) -> Result<(), ManagedSessionCacheError> {
    if name.is_empty()
        || name.len() > 255
        || name == "."
        || name == ".."
        || name.contains('/')
        || name.contains('\\')
        || name.chars().any(char::is_control)
    {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache name is not a safe component",
        ));
    }
    Ok(())
}

fn io(operation: &'static str, error: rustix::io::Errno) -> ManagedSessionCacheError {
    match error {
        rustix::io::Errno::NOSPC | rustix::io::Errno::DQUOT => {
            ManagedSessionCacheError::StorageFull
        }
        _ => ManagedSessionCacheError::Filesystem(format!("{operation}: {error}")),
    }
}
