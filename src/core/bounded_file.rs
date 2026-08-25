//! Stable bounded reads for small authority and metadata files.

use std::fs::{self, OpenOptions};
use std::io::{Read, Result, Seek, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::os::unix::io::AsRawFd;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StableFileIdentity {
    device: u64,
    inode: u64,
    length: u64,
    mode: u32,
    mtime: i64,
    mtime_nsec: i64,
    ctime: i64,
    ctime_nsec: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StableLinkIdentity {
    device: u64,
    inode: u64,
    length: u64,
    mode: u32,
}

fn snapshot(metadata: &fs::Metadata) -> StableFileIdentity {
    StableFileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        length: metadata.len(),
        mode: metadata.mode(),
        mtime: metadata.mtime(),
        mtime_nsec: metadata.mtime_nsec(),
        ctime: metadata.ctime(),
        ctime_nsec: metadata.ctime_nsec(),
    }
}

fn link_snapshot(metadata: &fs::Metadata) -> StableLinkIdentity {
    StableLinkIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        length: metadata.len(),
        mode: metadata.mode(),
    }
}

fn link_snapshot_stat(metadata: &rustix::fs::Stat) -> StableLinkIdentity {
    StableLinkIdentity {
        device: metadata.st_dev as u64,
        inode: metadata.st_ino as u64,
        length: metadata.st_size.max(0) as u64,
        mode: metadata.st_mode as u32,
    }
}

pub(crate) struct StableRegularFile {
    file: fs::File,
    path: std::path::PathBuf,
    identity: StableFileIdentity,
    namespace: Option<WalkedNamespace>,
    direct_symlink: Option<StableLinkIdentity>,
}

pub(crate) struct StableDirectory {
    file: fs::File,
    public_path: std::path::PathBuf,
    canonical_path: std::path::PathBuf,
    identity: StableFileIdentity,
}

impl StableRegularFile {
    /// Pathname that reopens this retained descriptor rather than the public
    /// filesystem name. The descriptor must outlive every open through this
    /// path. macOS and the supported Unix development hosts expose `/dev/fd`.
    pub(crate) fn activation_path(&self) -> Result<std::path::PathBuf> {
        // macOS implements /dev/fd opens as descriptor duplication, so the
        // new handle shares this open-file description's seek offset. Reset
        // immediately before handing out the activation name.
        let mut shared = self.file.try_clone()?;
        shared.rewind()?;
        Ok(std::path::PathBuf::from(format!(
            "/dev/fd/{}",
            self.file.as_raw_fd()
        )))
    }
}

impl StableDirectory {
    pub(crate) fn create_and_open(path: &Path) -> Result<Self> {
        use rustix::fs::{Mode, OFlags};

        fs::create_dir_all(path)?;
        let canonical_path = path.canonicalize()?;
        let file = fs::File::from(rustix::fs::open(
            &canonical_path,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )?);
        let metadata = file.metadata()?;
        if !metadata.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "retained destination authority is not a directory",
            ));
        }
        Ok(Self {
            file,
            public_path: path.to_path_buf(),
            canonical_path,
            identity: snapshot(&metadata),
        })
    }

    pub(crate) fn try_clone(&self) -> Result<fs::File> {
        self.file.try_clone()
    }

    pub(crate) fn device_id(&self) -> u64 {
        self.identity.device
    }

    pub(crate) fn available_bytes(&self) -> Option<u64> {
        rustix::fs::fstatvfs(&self.file)
            .ok()
            .and_then(|stats| stats.f_bavail.checked_mul(stats.f_frsize))
    }

    pub(crate) fn canonical_path(&self) -> &Path {
        &self.canonical_path
    }

    pub(crate) fn is_current(&self) -> Result<bool> {
        let descriptor = snapshot(&self.file.metadata()?);
        if !descriptor.same_inode(self.identity)
            || self
                .public_path
                .canonicalize()
                .ok()
                .is_none_or(|path| path != self.canonical_path)
        {
            return Ok(false);
        }
        let reopened = Self::open_existing_canonical(&self.canonical_path)?;
        Ok(reopened.is_some_and(|file| {
            file.metadata()
                .ok()
                .map(|metadata| snapshot(&metadata).same_inode(self.identity))
                .unwrap_or(false)
        }))
    }

    fn open_existing_canonical(path: &Path) -> Result<Option<fs::File>> {
        use rustix::fs::{Mode, OFlags};

        match rustix::fs::open(
            path,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        ) {
            Ok(fd) => Ok(Some(fs::File::from(fd))),
            Err(error)
                if matches!(
                    error,
                    rustix::io::Errno::NOENT | rustix::io::Errno::NOTDIR | rustix::io::Errno::LOOP
                ) =>
            {
                Ok(None)
            }
            Err(error) => Err(std::io::Error::from_raw_os_error(error.raw_os_error())),
        }
    }
}

struct WalkedNamespace {
    root: fs::File,
    root_path: std::path::PathBuf,
    relative: std::path::PathBuf,
    leaf_symlink: Option<StableLinkIdentity>,
}

impl StableRegularFile {
    pub(crate) fn from_open_file(
        file: fs::File,
        path: &Path,
        expected_bytes: u64,
    ) -> Result<Option<Self>> {
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.len() != expected_bytes {
            return Ok(None);
        }
        Ok(Some(Self {
            file,
            path: path.to_path_buf(),
            identity: snapshot(&metadata),
            namespace: None,
            direct_symlink: None,
        }))
    }

    pub(crate) fn open_exact(path: &Path, expected_bytes: u64) -> Result<Option<Self>> {
        let file = match OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path)
        {
            Ok(file) => file,
            Err(error)
                if error.kind() == std::io::ErrorKind::NotFound
                    || error.raw_os_error() == Some(libc::ELOOP) =>
            {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        Self::from_open_file(file, path, expected_bytes)
    }

    /// Open one operator-placed regular file or direct file symlink while
    /// retaining both the target descriptor and the symlink identity. Receipt
    /// and publication authorities continue to use strict [`Self::open_exact`].
    pub(crate) fn open_operator_path_exact(
        path: &Path,
        expected_bytes: u64,
    ) -> Result<Option<Self>> {
        let before = match fs::symlink_metadata(path) {
            Ok(metadata) if metadata.file_type().is_symlink() => link_snapshot(&metadata),
            Ok(metadata) if metadata.is_file() => return Self::open_exact(path, expected_bytes),
            Ok(_) => return Ok(None),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let file = match OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NONBLOCK)
            .open(path)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let target = file.metadata()?;
        let after = match fs::symlink_metadata(path) {
            Ok(metadata) if metadata.file_type().is_symlink() => link_snapshot(&metadata),
            _ => return Ok(None),
        };
        if before != after || !target.is_file() || target.len() != expected_bytes {
            return Ok(None);
        }
        Ok(Some(Self {
            file,
            path: path.to_path_buf(),
            identity: snapshot(&target),
            namespace: None,
            direct_symlink: Some(before),
        }))
    }

    pub(crate) fn from_walked_file(
        root: fs::File,
        root_path: std::path::PathBuf,
        relative: std::path::PathBuf,
        path: &Path,
        file: fs::File,
        expected_bytes: u64,
    ) -> Result<Option<Self>> {
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.len() != expected_bytes {
            return Ok(None);
        }
        Ok(Some(Self {
            file,
            path: path.to_path_buf(),
            identity: snapshot(&metadata),
            namespace: Some(WalkedNamespace {
                root,
                root_path,
                relative,
                leaf_symlink: None,
            }),
            direct_symlink: None,
        }))
    }

    fn from_walked_symlink(
        root: fs::File,
        root_path: std::path::PathBuf,
        relative: std::path::PathBuf,
        path: &Path,
        file: fs::File,
        link_identity: StableLinkIdentity,
    ) -> Result<Option<Self>> {
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Ok(None);
        }
        Ok(Some(Self {
            file,
            path: path.to_path_buf(),
            identity: snapshot(&metadata),
            namespace: Some(WalkedNamespace {
                root,
                root_path,
                relative,
                leaf_symlink: Some(link_identity),
            }),
            direct_symlink: None,
        }))
    }

    pub(crate) fn try_clone(&self) -> Result<fs::File> {
        self.file.try_clone()
    }

    pub(crate) fn identity(&self) -> StableFileIdentity {
        self.identity
    }

    pub(crate) fn has_operator_symlink_leaf(&self) -> bool {
        self.direct_symlink.is_some()
            || self
                .namespace
                .as_ref()
                .is_some_and(|namespace| namespace.leaf_symlink.is_some())
    }

    /// Canonical public name for the exact retained inode. This is used only
    /// to discover small authority files adjacent to an operator-linked GGUF;
    /// activation continues through the retained descriptor.
    pub(crate) fn canonical_path_for_identity(&self) -> Result<Option<std::path::PathBuf>> {
        if !self.is_stable()? {
            return Ok(None);
        }
        // Resolve the name from the retained descriptor, not by following the
        // mutable public symlink again. Otherwise a swap to a same-inode
        // hardlink namespace could redirect adjacent receipt/pair-journal
        // authority without changing the retained payload identity.
        let canonical = match descriptor_identity_path(&self.file) {
            Ok(path) => path,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let metadata = match fs::symlink_metadata(&canonical) {
            Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => metadata,
            _ => return Ok(None),
        };
        if snapshot(&metadata) != self.identity || !self.is_stable()? {
            return Ok(None);
        }
        Ok(Some(canonical))
    }

    pub(crate) fn device_id(&self) -> u64 {
        self.identity.device
    }

    pub(crate) fn descriptor_is_stable(&self) -> Result<bool> {
        Ok(snapshot(&self.file.metadata()?) == self.identity)
    }

    /// Compare the current descriptor state with a newly opened final path.
    /// This intentionally re-baselines ctime after an atomic rename, which
    /// changes inode metadata without changing the published inode or bytes.
    pub(crate) fn current_descriptor_matches_path(&self, path: &Path) -> Result<bool> {
        let current = snapshot(&self.file.metadata()?);
        let Some(opened) = Self::open_exact(path, current.length)? else {
            return Ok(false);
        };
        Ok(opened.identity == current && opened.is_stable()?)
    }

    #[cfg(target_vendor = "apple")]
    pub(crate) fn clone_to_at(
        &self,
        destination_directory: &fs::File,
        destination_name: &std::ffi::OsStr,
    ) -> Result<()> {
        rustix::fs::fclonefileat(
            &self.file,
            destination_directory,
            destination_name,
            rustix::fs::CloneFlags::NOFOLLOW | rustix::fs::CloneFlags::NOOWNERCOPY,
        )?;
        Ok(())
    }

    #[cfg(not(target_vendor = "apple"))]
    pub(crate) fn clone_to_at(
        &self,
        _destination_directory: &fs::File,
        _destination_name: &std::ffi::OsStr,
    ) -> Result<()> {
        Err(std::io::Error::from_raw_os_error(libc::EXDEV))
    }

    pub(crate) fn read_bounded(&mut self, maximum: u64) -> Result<Option<Vec<u8>>> {
        if self.identity.length > maximum {
            return Ok(None);
        }
        self.file.rewind()?;
        let mut bytes = Vec::with_capacity(self.identity.length as usize);
        (&mut self.file)
            .take(maximum.saturating_add(1))
            .read_to_end(&mut bytes)?;
        let result = self
            .is_stable()
            .map(|stable| (stable && bytes.len() as u64 == self.identity.length).then_some(bytes));
        self.file.rewind()?;
        result
    }

    pub(crate) fn is_stable(&self) -> Result<bool> {
        let descriptor = snapshot(&self.file.metadata()?);
        let path = match &self.namespace {
            Some(namespace) => {
                let Some(reopened) = reopen_walked_path(namespace)? else {
                    return Ok(false);
                };
                let metadata = reopened.metadata()?;
                if !metadata.is_file() {
                    return Ok(false);
                }
                snapshot(&metadata)
            }
            _ if self.direct_symlink.is_some() => {
                let Some(reopened) =
                    Self::open_operator_path_exact(&self.path, self.identity.length)?
                else {
                    return Ok(false);
                };
                if reopened.direct_symlink != self.direct_symlink {
                    return Ok(false);
                }
                snapshot(&reopened.file.metadata()?)
            }
            _ => match fs::symlink_metadata(&self.path) {
                Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {
                    snapshot(&metadata)
                }
                _ => return Ok(false),
            },
        };
        Ok(descriptor == self.identity && path == self.identity)
    }

    pub(crate) fn sha256(&mut self) -> Result<Option<String>> {
        self.sha256_with_progress(|_| {})
    }

    #[cfg(test)]
    pub(crate) fn sha256_with_hook(
        &mut self,
        after_first_chunk: impl FnOnce(),
    ) -> Result<Option<String>> {
        let mut hook = Some(after_first_chunk);
        self.sha256_with_progress(|_| {
            if let Some(hook) = hook.take() {
                hook();
            }
        })
    }

    pub(crate) fn sha256_with_progress(
        &mut self,
        mut after_chunk: impl FnMut(u64),
    ) -> Result<Option<String>> {
        use sha2::{Digest, Sha256};

        self.file.rewind()?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0_u8; 1024 * 1024];
        let mut total = 0_u64;
        let mut limited = (&mut self.file).take(self.identity.length.saturating_add(1));
        loop {
            let read = limited.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            total = total.saturating_add(read as u64);
            after_chunk(total);
        }
        let result = self.is_stable().map(|stable| {
            (stable && total == self.identity.length).then(|| hex::encode(hasher.finalize()))
        });
        self.file.rewind()?;
        result
    }

    pub(crate) fn copy_and_hash(
        &mut self,
        destination: &mut impl Write,
    ) -> Result<Option<(u64, String)>> {
        use sha2::{Digest, Sha256};

        self.file.rewind()?;
        let mut hasher = Sha256::new();
        let mut total = 0_u64;
        let mut buffer = vec![0_u8; 1024 * 1024];
        let mut limited = (&mut self.file).take(self.identity.length.saturating_add(1));
        loop {
            let read = limited.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            destination.write_all(&buffer[..read])?;
            hasher.update(&buffer[..read]);
            total = total.saturating_add(read as u64);
        }
        let result = self.is_stable().map(|stable| {
            (stable && total == self.identity.length)
                .then(|| (total, hex::encode(hasher.finalize())))
        });
        self.file.rewind()?;
        result
    }
}

#[cfg(target_os = "linux")]
fn descriptor_identity_path(file: &fs::File) -> Result<std::path::PathBuf> {
    fs::read_link(format!("/proc/self/fd/{}", file.as_raw_fd()))
}

#[cfg(target_os = "macos")]
fn descriptor_identity_path(file: &fs::File) -> Result<std::path::PathBuf> {
    use std::ffi::CStr;
    use std::os::unix::ffi::OsStrExt;

    let mut buffer = [0_i8; libc::PATH_MAX as usize];
    // SAFETY: F_GETPATH writes at most PATH_MAX bytes to this live buffer and
    // does not retain the pointer. The descriptor is owned for the call.
    let result = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETPATH, buffer.as_mut_ptr()) };
    if result == -1 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: a successful F_GETPATH call returns one NUL-terminated path in
    // the PATH_MAX-sized output buffer.
    let bytes = unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_bytes();
    Ok(std::path::PathBuf::from(std::ffi::OsStr::from_bytes(bytes)))
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn descriptor_identity_path(file: &fs::File) -> Result<std::path::PathBuf> {
    fs::read_link(format!("/dev/fd/{}", file.as_raw_fd()))
}

impl StableFileIdentity {
    pub(crate) fn same_inode(self, other: Self) -> bool {
        self.device == other.device && self.inode == other.inode
    }

    pub(crate) fn length(self) -> u64 {
        self.length
    }
}

pub(crate) fn regular_path_matches_identity(
    path: &Path,
    expected: StableFileIdentity,
) -> Result<bool> {
    let Some(file) = StableRegularFile::open_operator_path_exact(path, expected.length)? else {
        return Ok(false);
    };
    Ok(file.identity == expected && file.is_stable()?)
}

fn reopen_walked_path(namespace: &WalkedNamespace) -> Result<Option<fs::File>> {
    use rustix::fs::{AtFlags, Mode, OFlags};

    let root_now = match rustix::fs::open(
        &namespace.root_path,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    ) {
        Ok(fd) => fs::File::from(fd),
        Err(_) => return Ok(None),
    };
    let expected_root = snapshot(&namespace.root.metadata()?);
    if snapshot(&root_now.metadata()?) != expected_root {
        return Ok(None);
    }
    let mut directory = namespace.root.try_clone()?;
    let components = namespace.relative.components().collect::<Vec<_>>();
    for (index, component) in components.iter().enumerate() {
        let std::path::Component::Normal(name) = component else {
            return Ok(None);
        };
        let last = index + 1 == components.len();
        let mut flags = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC;
        if !last {
            flags |= OFlags::DIRECTORY;
        }
        if last {
            if let Some(expected_link) = namespace.leaf_symlink {
                let before = match rustix::fs::statat(&directory, *name, AtFlags::SYMLINK_NOFOLLOW)
                {
                    Ok(metadata) => link_snapshot_stat(&metadata),
                    Err(_) => return Ok(None),
                };
                if before != expected_link
                    || before.mode & libc::S_IFMT as u32 != libc::S_IFLNK as u32
                {
                    return Ok(None);
                }
                let opened = match rustix::fs::openat(
                    &directory,
                    *name,
                    OFlags::RDONLY | OFlags::NONBLOCK | OFlags::CLOEXEC,
                    Mode::empty(),
                ) {
                    Ok(fd) => fs::File::from(fd),
                    Err(_) => return Ok(None),
                };
                let after = match rustix::fs::statat(&directory, *name, AtFlags::SYMLINK_NOFOLLOW) {
                    Ok(metadata) => link_snapshot_stat(&metadata),
                    Err(_) => return Ok(None),
                };
                return Ok((after == before && opened.metadata()?.is_file()).then_some(opened));
            }
        }
        let opened = match rustix::fs::openat(&directory, *name, flags, Mode::empty()) {
            Ok(fd) => fs::File::from(fd),
            Err(_) => return Ok(None),
        };
        if last {
            return Ok(Some(opened));
        }
        directory = opened;
    }
    Ok(None)
}

pub(crate) fn open_walked_operator_symlink(
    root: fs::File,
    root_path: std::path::PathBuf,
    relative: std::path::PathBuf,
    directory: &fs::File,
    name: &std::ffi::OsStr,
    path: &Path,
) -> Result<Option<StableRegularFile>> {
    use rustix::fs::{AtFlags, Mode, OFlags};

    let before = match rustix::fs::statat(directory, name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(metadata) => link_snapshot_stat(&metadata),
        Err(_) => return Ok(None),
    };
    if before.mode & libc::S_IFMT as u32 != libc::S_IFLNK as u32 {
        return Ok(None);
    }
    let file = match rustix::fs::openat(
        directory,
        name,
        OFlags::RDONLY | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    ) {
        Ok(fd) => fs::File::from(fd),
        Err(_) => return Ok(None),
    };
    let target = file.metadata()?;
    let after = match rustix::fs::statat(directory, name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(metadata) => link_snapshot_stat(&metadata),
        Err(_) => return Ok(None),
    };
    if before != after || !target.is_file() {
        return Ok(None);
    }
    StableRegularFile::from_walked_symlink(root, root_path, relative, path, file, before)
}

pub(crate) fn walked_directory_is_current(
    root: &fs::File,
    root_path: &Path,
    relative: &Path,
) -> Result<bool> {
    use rustix::fs::{Mode, OFlags};

    let root_now = match rustix::fs::open(
        root_path,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    ) {
        Ok(fd) => fs::File::from(fd),
        Err(_) => return Ok(false),
    };
    if snapshot(&root_now.metadata()?) != snapshot(&root.metadata()?) {
        return Ok(false);
    }
    let mut directory = root.try_clone()?;
    for component in relative.components() {
        let std::path::Component::Normal(name) = component else {
            return Ok(false);
        };
        let opened = match rustix::fs::openat(
            &directory,
            name,
            OFlags::RDONLY
                | OFlags::DIRECTORY
                | OFlags::NOFOLLOW
                | OFlags::NONBLOCK
                | OFlags::CLOEXEC,
            Mode::empty(),
        ) {
            Ok(fd) => fs::File::from(fd),
            Err(_) => return Ok(false),
        };
        directory = opened;
    }
    Ok(true)
}

pub(crate) fn sha256_regular_nofollow_exact(
    path: &Path,
    expected_bytes: u64,
) -> Result<Option<String>> {
    let Some(mut file) = StableRegularFile::open_exact(path, expected_bytes)? else {
        return Ok(None);
    };
    file.sha256()
}

pub(crate) fn sha256_operator_path_exact(
    path: &Path,
    expected_bytes: u64,
) -> Result<Option<String>> {
    let Some(mut file) = StableRegularFile::open_operator_path_exact(path, expected_bytes)? else {
        return Ok(None);
    };
    file.sha256()
}

/// Read one regular file through a retained no-follow, nonblocking descriptor.
/// Returns `None` for missing, symlinked, non-regular, oversized, replaced, or
/// concurrently mutated paths.
pub(crate) fn read_bounded_regular_nofollow(path: &Path, maximum: u64) -> Result<Option<Vec<u8>>> {
    read_bounded_regular_nofollow_with_hook(path, maximum, || {})
}

pub(crate) fn read_bounded_regular_nofollow_with_hook(
    path: &Path,
    maximum: u64,
    after_open: impl FnOnce(),
) -> Result<Option<Vec<u8>>> {
    let file = match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
    {
        Ok(file) => file,
        Err(error)
            if error.kind() == std::io::ErrorKind::NotFound
                || error.raw_os_error() == Some(libc::ELOOP) =>
        {
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let before_metadata = file.metadata()?;
    if !before_metadata.is_file() || before_metadata.len() > maximum {
        return Ok(None);
    }
    let before = snapshot(&before_metadata);
    after_open();
    let mut bytes = Vec::with_capacity(before.length as usize);
    (&file)
        .take(maximum.saturating_add(1))
        .read_to_end(&mut bytes)?;
    let after = snapshot(&file.metadata()?);
    let path_after = match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {
            snapshot(&metadata)
        }
        _ => return Ok(None),
    };
    if before != after
        || before != path_after
        || bytes.len() as u64 != before.length
        || bytes.len() as u64 > maximum
    {
        return Ok(None);
    }
    Ok(Some(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::symlink;

    #[test]
    fn bounded_authority_reads_reject_symlink_fifo_oversize_and_mutation() {
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("target.json");
        fs::write(&target, b"{}").unwrap();
        let link = directory.path().join("link.json");
        symlink(&target, &link).unwrap();
        assert!(read_bounded_regular_nofollow(&link, 1024)
            .unwrap()
            .is_none());

        let fifo = directory.path().join("metadata.fifo");
        let fifo_c = CString::new(fifo.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(fifo_c.as_ptr(), 0o600) }, 0);
        assert!(read_bounded_regular_nofollow(&fifo, 1024)
            .unwrap()
            .is_none());
        assert!(sha256_regular_nofollow_exact(&fifo, 0).unwrap().is_none());

        let oversized = directory.path().join("oversized.json");
        fs::write(&oversized, vec![b'x'; 1025]).unwrap();
        assert!(read_bounded_regular_nofollow(&oversized, 1024)
            .unwrap()
            .is_none());

        let receipt = directory.path().join("receipt.json");
        let prior = directory.path().join("prior.json");
        fs::write(&receipt, b"old receipt").unwrap();
        let replaced = read_bounded_regular_nofollow_with_hook(&receipt, 1024, || {
            fs::rename(&receipt, &prior).unwrap();
            fs::write(&receipt, b"replacement").unwrap();
        })
        .unwrap();
        assert!(replaced.is_none(), "path replacement must fail closed");

        let mutated = directory.path().join("mutated.json");
        fs::write(&mutated, b"original").unwrap();
        let unstable = read_bounded_regular_nofollow_with_hook(&mutated, 1024, || {
            fs::write(&mutated, b"mutation").unwrap();
        })
        .unwrap();
        assert!(unstable.is_none(), "same-inode mutation must fail closed");

        let model = directory.path().join("manual.gguf");
        let prior_model = directory.path().join("prior.gguf");
        fs::write(&model, b"exact model bytes").unwrap();
        let mut opened = StableRegularFile::open_exact(&model, 17).unwrap().unwrap();
        fs::rename(&model, &prior_model).unwrap();
        fs::write(&model, b"other model bytes").unwrap();
        assert_eq!(
            fs::read(opened.activation_path().unwrap()).unwrap(),
            b"exact model bytes",
            "activation must reopen the retained inode, not the replaced pathname"
        );
        assert!(opened.sha256().unwrap().is_none());

        let growing = directory.path().join("growing.gguf");
        fs::write(&growing, vec![b'a'; 2 * 1024 * 1024]).unwrap();
        let mut opened = StableRegularFile::open_exact(&growing, 2 * 1024 * 1024)
            .unwrap()
            .unwrap();
        let unstable = opened
            .sha256_with_hook(|| {
                let mut writer = fs::OpenOptions::new().append(true).open(&growing).unwrap();
                writer.write_all(b"growth").unwrap();
            })
            .unwrap();
        assert!(unstable.is_none(), "append during hash must fail bounded");
    }

    #[test]
    fn retained_sha256_rewinds_after_a_shared_descriptor_read() {
        let directory = tempfile::tempdir().unwrap();
        let projector = directory.path().join("mmproj.gguf");
        let bytes = b"complete retained projector bytes";
        fs::write(&projector, bytes).unwrap();
        let expected = crate::core::sha256::compute_file_sha256(&projector).unwrap();
        let mut retained = StableRegularFile::open_exact(&projector, bytes.len() as u64)
            .unwrap()
            .unwrap();
        let mut shared = retained.try_clone().unwrap();
        let mut prefix = [0_u8; 9];
        shared.read_exact(&mut prefix).unwrap();

        assert_eq!(
            retained.sha256().unwrap().as_deref(),
            Some(expected.as_str())
        );
    }

    #[test]
    fn direct_operator_file_symlink_retains_target_and_detects_retargeting() {
        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("first.gguf");
        let second = directory.path().join("second.gguf");
        let link = directory.path().join("model.gguf");
        fs::write(&first, b"first-model").unwrap();
        fs::write(&second, b"other-model").unwrap();
        symlink(&first, &link).unwrap();

        let retained = StableRegularFile::open_operator_path_exact(&link, 11)
            .unwrap()
            .expect("direct file symlink");
        assert_eq!(
            fs::read(retained.activation_path().unwrap()).unwrap(),
            b"first-model"
        );
        assert!(retained.is_stable().unwrap());

        fs::remove_file(&link).unwrap();
        symlink(&second, &link).unwrap();
        assert!(!retained.is_stable().unwrap());
        assert_eq!(
            fs::read(retained.activation_path().unwrap()).unwrap(),
            b"first-model",
            "activation must remain pinned to the admitted target inode"
        );
    }

    #[test]
    fn descriptor_identity_path_cannot_be_redirected_to_a_hardlink_namespace_by_swap_restore() {
        let original = tempfile::tempdir().unwrap();
        let alternate = tempfile::tempdir().unwrap();
        let target = original.path().join("model.gguf");
        let alias = alternate.path().join("model.gguf");
        let link = original.path().join("logical.gguf");
        let parked_link = original.path().join("logical.parked");
        fs::write(&target, b"same retained inode").unwrap();
        fs::hard_link(&target, &alias).unwrap();
        symlink(&target, &link).unwrap();
        let retained = StableRegularFile::open_operator_path_exact(&link, 19)
            .unwrap()
            .unwrap();

        fs::rename(&link, &parked_link).unwrap();
        symlink(&alias, &link).unwrap();
        fs::remove_file(&link).unwrap();
        fs::rename(&parked_link, &link).unwrap();

        let expected = target.canonicalize().unwrap();
        assert_eq!(
            retained.canonical_path_for_identity().unwrap().as_deref(),
            Some(expected.as_path()),
            "adjacent authority must stay in the descriptor's original target namespace"
        );
    }
}
