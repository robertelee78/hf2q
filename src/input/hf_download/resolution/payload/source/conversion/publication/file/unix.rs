use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::fd::OwnedFd;
use std::path::{Component, Path};

use rustix::fs::{self, AtFlags, FileType, Mode, OFlags, Stat};

use super::super::{publication_error, publication_require, ModelPreparationPublicationError};

pub(super) const PRIVATE_MODE: rustix::fs::RawMode = 0o600;
const DIRECTORY_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::DIRECTORY)
    .union(OFlags::NOFOLLOW)
    .union(OFlags::CLOEXEC);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::input::hf_download::resolution::payload::source::conversion::publication) struct Identity
{
    device: u64,
    inode: u64,
}

impl Identity {
    pub(in crate::input::hf_download::resolution::payload::source::conversion::publication) fn device(
        self,
    ) -> u64 {
        self.device
    }

    #[cfg(test)]
    pub(in crate::input::hf_download::resolution::payload::source::conversion::publication) const fn for_test(
        value: u64,
    ) -> Self {
        Self {
            device: value,
            inode: value,
        }
    }
}

pub(super) struct Directory {
    pub(super) fd: OwnedFd,
    pub(super) identity: Identity,
}

pub(super) fn named_file_identity(
    parent: &Directory,
    name: &str,
    links: u64,
    size: Option<u64>,
) -> Result<Identity, ModelPreparationPublicationError> {
    let stat =
        fs::statat(&parent.fd, name, AtFlags::SYMLINK_NOFOLLOW).map_err(std::io::Error::from)?;
    require_private_stat(&stat, parent.identity.device, links)?;
    if let Some(size) = size {
        publication_require(stat.st_size as u64 == size, "model record length differs")?;
    }
    Ok(stat_identity(&stat))
}

pub(super) fn open_named_file(
    parent: &Directory,
    name: &str,
    expected_identity: Identity,
    read_only: bool,
) -> Result<File, ModelPreparationPublicationError> {
    let access = if read_only {
        OFlags::RDONLY
    } else {
        OFlags::RDWR
    };
    let fd = fs::openat(
        &parent.fd,
        name,
        access | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(std::io::Error::from)?;
    let file = File::from(fd);
    let stat = fs::fstat(&file).map_err(std::io::Error::from)?;
    require_private_stat(&stat, parent.identity.device, stat.st_nlink as u64)?;
    publication_require(
        stat_identity(&stat) == expected_identity,
        "model record changed while opening",
    )?;
    Ok(file)
}

pub(super) fn verify_open_file(
    file: &mut File,
    expected_identity: Identity,
    expected: &[u8],
    links: u64,
    expected_device: u64,
) -> Result<(), ModelPreparationPublicationError> {
    let stat = fs::fstat(&*file).map_err(std::io::Error::from)?;
    require_private_stat(&stat, expected_device, links)?;
    publication_require(
        stat_identity(&stat) == expected_identity && stat.st_size as u64 == expected.len() as u64,
        "open model record identity or length changed",
    )?;
    file.seek(SeekFrom::Start(0))?;
    let mut actual = Vec::with_capacity(expected.len());
    file.take(expected.len() as u64 + 1)
        .read_to_end(&mut actual)?;
    publication_require(actual == expected, "model record bytes differ")
}

pub(super) fn verify_named_file(
    parent: &Directory,
    name: &str,
    expected_identity: Identity,
    expected: &[u8],
    links: u64,
) -> Result<(), ModelPreparationPublicationError> {
    verify_named_file_with(parent, name, expected_identity, expected, links, || Ok(()))
}

pub(super) fn verify_named_file_with(
    parent: &Directory,
    name: &str,
    expected_identity: Identity,
    expected: &[u8],
    links: u64,
    after_read: impl FnOnce() -> Result<(), ModelPreparationPublicationError>,
) -> Result<(), ModelPreparationPublicationError> {
    let named = named_file_identity(parent, name, links, Some(expected.len() as u64))?;
    publication_require(
        named == expected_identity,
        "named model record identity changed",
    )?;
    let mut file = open_named_file(parent, name, expected_identity, true)?;
    verify_open_file(
        &mut file,
        expected_identity,
        expected,
        links,
        parent.identity.device,
    )?;
    after_read()?;
    let after = fs::fstat(&file).map_err(std::io::Error::from)?;
    require_private_stat(&after, parent.identity.device, links)?;
    publication_require(
        stat_identity(&after) == expected_identity && after.st_size as u64 == expected.len() as u64,
        "open model record changed after reading",
    )?;
    verify_named_identity(parent, name, expected_identity)
}

pub(super) fn verify_named_identity(
    parent: &Directory,
    name: &str,
    expected: Identity,
) -> Result<(), ModelPreparationPublicationError> {
    let stat =
        fs::statat(&parent.fd, name, AtFlags::SYMLINK_NOFOLLOW).map_err(std::io::Error::from)?;
    publication_require(
        stat_identity(&stat) == expected,
        "named model record identity changed",
    )
}

pub(super) fn remove_named_file(
    parent: &Directory,
    name: &str,
    expected: Identity,
) -> Result<(), ModelPreparationPublicationError> {
    verify_named_identity(parent, name, expected)?;
    fs::unlinkat(&parent.fd, name, AtFlags::empty()).map_err(std::io::Error::from)?;
    Ok(())
}

pub(super) fn full_sync_named_file(
    parent: &Directory,
    name: &str,
    expected_identity: Identity,
    expected: &[u8],
) -> Result<(), ModelPreparationPublicationError> {
    let mut file = open_named_file(parent, name, expected_identity, true)?;
    verify_open_file(
        &mut file,
        expected_identity,
        expected,
        1,
        parent.identity.device,
    )?;
    full_sync(&file)
}

pub(super) fn full_sync(file: &File) -> Result<(), ModelPreparationPublicationError> {
    file.sync_all()?;
    #[cfg(target_os = "macos")]
    fs::fcntl_fullfsync(file).map_err(std::io::Error::from)?;
    Ok(())
}

pub(super) fn sync_directory(
    directory: &Directory,
) -> Result<(), ModelPreparationPublicationError> {
    fs::fsync(&directory.fd).map_err(std::io::Error::from)?;
    Ok(())
}

pub(super) fn open_exact_directory(
    path: &Path,
    expected: Option<Identity>,
) -> Result<Directory, ModelPreparationPublicationError> {
    publication_require(path.is_absolute(), "model record parent is not absolute")?;
    let root_fd = fs::open("/", DIRECTORY_FLAGS, Mode::empty()).map_err(std::io::Error::from)?;
    let root_stat = fs::fstat(&root_fd).map_err(std::io::Error::from)?;
    let mut directory = Directory {
        fd: root_fd,
        identity: stat_identity(&root_stat),
    };
    let mut saw_root = false;
    for component in path.components() {
        match component {
            Component::RootDir if !saw_root => saw_root = true,
            Component::Normal(name) if saw_root => {
                let named = fs::statat(&directory.fd, name, AtFlags::SYMLINK_NOFOLLOW)
                    .map_err(std::io::Error::from)?;
                publication_require(
                    FileType::from_raw_mode(named.st_mode) == FileType::Directory,
                    "model record ancestor is not a directory",
                )?;
                let fd = fs::openat(&directory.fd, name, DIRECTORY_FLAGS, Mode::empty())
                    .map_err(std::io::Error::from)?;
                let opened = fs::fstat(&fd).map_err(std::io::Error::from)?;
                publication_require(
                    stat_identity(&opened) == stat_identity(&named),
                    "model record ancestor changed while opening",
                )?;
                directory = Directory {
                    fd,
                    identity: stat_identity(&opened),
                };
            }
            _ => return Err(publication_error("model record parent is not canonical").into()),
        }
    }
    let final_stat = fs::fstat(&directory.fd).map_err(std::io::Error::from)?;
    publication_require(
        FileType::from_raw_mode(final_stat.st_mode) == FileType::Directory
            && final_stat.st_uid == rustix::process::geteuid().as_raw(),
        "model record parent is not an exact owned directory",
    )?;
    if let Some(expected) = expected {
        publication_require(
            directory.identity == expected,
            "model record parent differs from authenticated identity",
        )?;
    }
    Ok(directory)
}

pub(super) fn require_directory_rebound(
    path: &Path,
    expected: Identity,
) -> Result<(), ModelPreparationPublicationError> {
    publication_require(
        open_exact_directory(path, Some(expected))?.identity == expected,
        "model record parent changed during publication",
    )
}

pub(super) fn entry_exists_at(parent: &Directory, name: &str) -> Result<bool, std::io::Error> {
    match fs::statat(&parent.fd, name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(_) => Ok(true),
        Err(rustix::io::Errno::NOENT) => Ok(false),
        Err(error) => Err(std::io::Error::from(error)),
    }
}

pub(super) fn require_created_file(
    stat: &Stat,
    expected_device: u64,
) -> Result<(), ModelPreparationPublicationError> {
    publication_require(
        FileType::from_raw_mode(stat.st_mode) == FileType::RegularFile
            && stat.st_uid == rustix::process::geteuid().as_raw()
            && stat.st_dev as u64 == expected_device
            && stat.st_nlink == 1,
        "created model record ownership, device, type, or links are invalid",
    )
}

pub(super) fn require_private_stat(
    stat: &Stat,
    expected_device: u64,
    links: u64,
) -> Result<(), ModelPreparationPublicationError> {
    publication_require(
        FileType::from_raw_mode(stat.st_mode) == FileType::RegularFile
            && stat.st_uid == rustix::process::geteuid().as_raw()
            && stat.st_dev as u64 == expected_device
            && stat.st_nlink as u64 == links
            && permission_bits(stat) == PRIVATE_MODE,
        "model record ownership, device, links, or mode are invalid",
    )
}

pub(super) fn validate_name(name: &str) -> Result<(), ModelPreparationPublicationError> {
    publication_require(
        !name.is_empty()
            && name != "."
            && name != ".."
            && !name.contains('/')
            && !name.contains('\0'),
        "model record name is invalid",
    )
}

pub(super) fn stat_identity(stat: &Stat) -> Identity {
    Identity {
        device: stat.st_dev as u64,
        inode: stat.st_ino as u64,
    }
}

fn permission_bits(stat: &Stat) -> rustix::fs::RawMode {
    stat.st_mode & 0o7777
}

pub(super) fn publication_as_io(error: ModelPreparationPublicationError) -> std::io::Error {
    std::io::Error::other(error.to_string())
}
