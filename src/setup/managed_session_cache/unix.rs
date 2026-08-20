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

    pub(super) const fn identity(&self) -> EntryIdentity {
        self.identity
    }
}

impl EntryIdentity {
    pub(super) const fn same_node(self, other: Self) -> bool {
        self.device == other.device && self.inode == other.inode
    }

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
    match open_recoverable_directory(parent, name) {
        Ok(directory) => {
            sync_directory(&directory)?;
            sync_directory(parent)?;
            verify_directory(parent, name, &directory)?;
            Ok(directory)
        }
        Err(ManagedSessionCacheError::Missing) => {
            match fs::mkdirat(parent.fd(), name, Mode::from_raw_mode(0o700)) {
                Ok(()) => {}
                Err(rustix::io::Errno::EXIST) => return open_recoverable_directory(parent, name),
                Err(error) => return Err(io("create managed cache directory", error)),
            }
            #[cfg(test)]
            super::tests::abort_at_managed_cache_barrier(
                super::ManagedCacheBarrier::DirectoryCreatedBeforeMode,
            );
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

pub(super) fn open_recoverable_directory(
    parent: &Directory,
    name: &str,
) -> Result<Directory, ManagedSessionCacheError> {
    validate_component(name)?;
    let named = match fs::statat(parent.fd(), name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) => stat,
        Err(rustix::io::Errno::NOENT) => return Err(ManagedSessionCacheError::Missing),
        Err(error) => return Err(io("inspect recoverable managed cache directory", error)),
    };
    let before = stat_identity(&named)?;
    if before.file_type != FileType::Directory
        || before.owner != rustix::process::geteuid().as_raw()
        || before.device != parent.device()
        || before.links < 1
        || before.mode & !0o700 != 0
    {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache directory is not an exact recoverable private directory",
        ));
    }
    if before.mode != 0o700 {
        fs::chmodat(
            parent.fd(),
            name,
            Mode::from_raw_mode(0o700),
            AtFlags::SYMLINK_NOFOLLOW,
        )
        .map_err(|error| io("recover managed cache directory mode", error))?;
    }
    let directory = open_directory(parent, name)?;
    if !directory.identity.same_node(before) {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache directory changed during mode recovery",
        ));
    }
    if before.mode != 0o700 {
        sync_directory(&directory)?;
        sync_directory(parent)?;
        verify_directory(parent, name, &directory)?;
    }
    Ok(directory)
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

include!("unix/private.rs");
