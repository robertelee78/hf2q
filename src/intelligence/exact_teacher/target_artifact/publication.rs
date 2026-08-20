//! Descriptor-relative, no-clobber publication for a retained target inode.

use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::os::fd::OwnedFd;
use std::path::{Path, PathBuf};

use rand::RngCore;
use rustix::fs::{self, AtFlags, FileType, Mode, OFlags, RenameFlags, Stat};

use super::ExactTeacherTargetError;

#[cfg(test)]
#[path = "publication_tests.rs"]
mod tests;

const DIRECTORY_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::DIRECTORY)
    .union(OFlags::NOFOLLOW)
    .union(OFlags::CLOEXEC);
const TEMP_FLAGS: OFlags = OFlags::RDWR
    .union(OFlags::CREATE)
    .union(OFlags::EXCL)
    .union(OFlags::NOFOLLOW)
    .union(OFlags::NONBLOCK)
    .union(OFlags::CLOEXEC);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Identity {
    device: u64,
    inode: u64,
}

pub(super) struct RetainedTargetTemp {
    parent: OwnedFd,
    parent_path: PathBuf,
    parent_identity: Identity,
    temporary_name: Option<OsString>,
    final_name: OsString,
    output: PathBuf,
    identity: Identity,
    file: Option<File>,
}

impl RetainedTargetTemp {
    pub(super) fn create(output: &Path) -> Result<Self, ExactTeacherTargetError> {
        let parent = output
            .parent()
            .ok_or_else(|| invalid("teacher target path has no parent"))?;
        let final_name = output
            .file_name()
            .filter(|name| !name.is_empty())
            .ok_or_else(|| invalid("teacher target path has no file name"))?
            .to_owned();
        std::fs::create_dir_all(parent)
            .map_err(|error| ExactTeacherTargetError::io(parent, error))?;
        let parent_path = std::fs::canonicalize(parent)
            .map_err(|error| ExactTeacherTargetError::io(parent, error))?;
        let parent = fs::open(&parent_path, DIRECTORY_FLAGS, Mode::empty())
            .map_err(std::io::Error::from)
            .map_err(|error| ExactTeacherTargetError::io(&parent_path, error))?;
        let parent_stat = fs::fstat(&parent)
            .map_err(std::io::Error::from)
            .map_err(|error| ExactTeacherTargetError::io(&parent_path, error))?;
        if FileType::from_raw_mode(parent_stat.st_mode) != FileType::Directory {
            return Err(invalid("teacher target parent is not a directory"));
        }
        let parent_identity = identity(&parent_stat);
        require_absent(&parent, &final_name, &parent_path.join(&final_name))?;

        let (temporary_name, file, file_identity) = create_temporary(&parent, &parent_path)?;
        let output = parent_path.join(&final_name);
        Ok(Self {
            parent,
            parent_path,
            parent_identity,
            temporary_name: Some(temporary_name),
            final_name,
            output,
            identity: file_identity,
            file: Some(file),
        })
    }

    pub(super) fn output(&self) -> &Path {
        &self.output
    }

    pub(super) fn as_file_mut(&mut self) -> &mut File {
        self.file
            .as_mut()
            .expect("retained teacher target file must be present")
    }

    pub(super) fn as_file(&self) -> &File {
        self.file
            .as_ref()
            .expect("retained teacher target file must be present")
    }

    pub(super) fn publish_noclobber(
        self,
        expected_len: u64,
        mut verify: impl FnMut(&mut File) -> Result<(), ExactTeacherTargetError>,
    ) -> Result<File, ExactTeacherTargetError> {
        self.publish_noclobber_inner(expected_len, &mut verify, || {})
    }

    #[cfg(test)]
    fn publish_noclobber_with_before_rename_for_test(
        self,
        expected_len: u64,
        mut verify: impl FnMut(&mut File) -> Result<(), ExactTeacherTargetError>,
        before_rename: impl FnOnce(),
    ) -> Result<File, ExactTeacherTargetError> {
        self.publish_noclobber_inner(expected_len, &mut verify, before_rename)
    }

    fn publish_noclobber_inner(
        mut self,
        expected_len: u64,
        verify: &mut impl FnMut(&mut File) -> Result<(), ExactTeacherTargetError>,
        before_rename: impl FnOnce(),
    ) -> Result<File, ExactTeacherTargetError> {
        self.require_parent_rebound()?;
        self.verify_temporary(expected_len, 1)?;
        verify(self.as_file_mut())?;
        require_absent(&self.parent, &self.final_name, &self.output)?;
        let temporary_name = self
            .temporary_name
            .as_ref()
            .expect("retained target temp must have a name");
        self.as_file()
            .sync_all()
            .map_err(|error| ExactTeacherTargetError::io(&self.output, error))?;
        fs::fsync(&self.parent)
            .map_err(std::io::Error::from)
            .map_err(|error| ExactTeacherTargetError::io(&self.output, error))?;
        self.verify_temporary(expected_len, 1)?;
        self.require_parent_rebound()?;
        before_rename();
        fs::renameat_with(
            &self.parent,
            temporary_name,
            &self.parent,
            &self.final_name,
            RenameFlags::NOREPLACE,
        )
        .map_err(std::io::Error::from)
        .map_err(|error| ExactTeacherTargetError::io(&self.output, error))?;
        // `renameat_with(NOREPLACE)` is deliberately the last fallible
        // operation. It atomically removes the private name and creates the
        // final name for the same inode. No rollback-by-path race exists.
        self.temporary_name = None;
        Ok(self
            .file
            .take()
            .expect("retained teacher target file must be present"))
    }

    fn require_parent_rebound(&self) -> Result<(), ExactTeacherTargetError> {
        let rebound = fs::open(&self.parent_path, DIRECTORY_FLAGS, Mode::empty())
            .map_err(std::io::Error::from)
            .map_err(|error| ExactTeacherTargetError::io(&self.parent_path, error))?;
        let rebound = fs::fstat(&rebound)
            .map_err(std::io::Error::from)
            .map_err(|error| ExactTeacherTargetError::io(&self.parent_path, error))?;
        if identity(&rebound) != self.parent_identity {
            return Err(invalid("teacher target parent changed during publication"));
        }
        Ok(())
    }

    fn verify_temporary(
        &self,
        expected_len: u64,
        links: u64,
    ) -> Result<(), ExactTeacherTargetError> {
        let name = self
            .temporary_name
            .as_ref()
            .ok_or_else(|| invalid("teacher target temporary name is absent"))?;
        self.verify_named(name, expected_len, links)?;
        self.verify_open(self.as_file(), expected_len, links)
    }

    fn verify_named(
        &self,
        name: &OsStr,
        expected_len: u64,
        links: u64,
    ) -> Result<(), ExactTeacherTargetError> {
        let stat = fs::statat(&self.parent, name, AtFlags::SYMLINK_NOFOLLOW)
            .map_err(std::io::Error::from)
            .map_err(|error| ExactTeacherTargetError::io(&self.output, error))?;
        require_file(
            &stat,
            self.parent_identity.device,
            self.identity,
            expected_len,
            links,
        )
    }

    fn verify_open(
        &self,
        file: &File,
        expected_len: u64,
        links: u64,
    ) -> Result<(), ExactTeacherTargetError> {
        let stat = fs::fstat(file)
            .map_err(std::io::Error::from)
            .map_err(|error| ExactTeacherTargetError::io(&self.output, error))?;
        require_file(
            &stat,
            self.parent_identity.device,
            self.identity,
            expected_len,
            links,
        )
    }
}

impl Drop for RetainedTargetTemp {
    fn drop(&mut self) {
        if let Some(name) = self.temporary_name.take() {
            remove_named_if_identity(&self.parent, &name, self.identity);
        }
    }
}

fn create_temporary(
    parent: &OwnedFd,
    parent_path: &Path,
) -> Result<(OsString, File, Identity), ExactTeacherTargetError> {
    for _ in 0..32 {
        let mut random = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut random);
        let name = OsString::from(format!(".hf2q-exact-teacher-{}.tmp", hex::encode(random)));
        match fs::openat(parent, &name, TEMP_FLAGS, Mode::from_raw_mode(0o600)) {
            Ok(fd) => {
                let file = File::from(fd);
                let stat = fs::fstat(&file)
                    .map_err(std::io::Error::from)
                    .map_err(|error| ExactTeacherTargetError::io(parent_path, error))?;
                let parent_stat = fs::fstat(parent)
                    .map_err(std::io::Error::from)
                    .map_err(|error| ExactTeacherTargetError::io(parent_path, error))?;
                let file_identity = identity(&stat);
                if let Err(error) =
                    require_file(&stat, identity(&parent_stat).device, file_identity, 0, 1)
                {
                    remove_named_if_identity(parent, &name, file_identity);
                    return Err(error);
                }
                return Ok((name, file, file_identity));
            }
            Err(rustix::io::Errno::EXIST) => continue,
            Err(error) => {
                return Err(ExactTeacherTargetError::io(
                    parent_path,
                    std::io::Error::from(error),
                ));
            }
        }
    }
    Err(invalid(
        "could not allocate a unique teacher target temporary",
    ))
}

fn require_absent(
    parent: &OwnedFd,
    name: &OsStr,
    output: &Path,
) -> Result<(), ExactTeacherTargetError> {
    match fs::statat(parent, name, AtFlags::SYMLINK_NOFOLLOW) {
        Err(rustix::io::Errno::NOENT) => Ok(()),
        Err(error) => Err(ExactTeacherTargetError::io(
            output,
            std::io::Error::from(error),
        )),
        Ok(_) => Err(invalid("teacher target destination already exists")),
    }
}

fn require_file(
    stat: &Stat,
    parent_device: u64,
    expected: Identity,
    expected_len: u64,
    links: u64,
) -> Result<(), ExactTeacherTargetError> {
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile
        || stat.st_dev as u64 != parent_device
        || identity(stat) != expected
        || stat.st_size as u64 != expected_len
        || stat.st_nlink as u64 != links
    {
        return Err(invalid("teacher target retained inode identity is invalid"));
    }
    Ok(())
}

fn remove_named_if_identity(parent: &OwnedFd, name: &OsStr, expected: Identity) {
    if fs::statat(parent, name, AtFlags::SYMLINK_NOFOLLOW)
        .map(|stat| identity(&stat) == expected)
        .unwrap_or(false)
    {
        let _ = fs::unlinkat(parent, name, AtFlags::empty());
    }
}

fn identity(stat: &Stat) -> Identity {
    Identity {
        device: stat.st_dev as u64,
        inode: stat.st_ino as u64,
    }
}

fn invalid(message: impl Into<String>) -> ExactTeacherTargetError {
    ExactTeacherTargetError::Invalid(message.into())
}
