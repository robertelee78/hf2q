//! Stable bounded reads for small authority and metadata files.

use std::fs::{self, OpenOptions};
use std::io::{Read, Result, Seek, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
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

pub(crate) struct StableRegularFile {
    file: fs::File,
    path: std::path::PathBuf,
    identity: StableFileIdentity,
    namespace: Option<WalkedNamespace>,
}

pub(crate) struct StableDirectory {
    file: fs::File,
    public_path: std::path::PathBuf,
    canonical_path: std::path::PathBuf,
    identity: StableFileIdentity,
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
            }),
        }))
    }

    pub(crate) fn try_clone(&self) -> Result<fs::File> {
        self.file.try_clone()
    }

    pub(crate) fn identity(&self) -> StableFileIdentity {
        self.identity
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
        self.is_stable()
            .map(|stable| (stable && bytes.len() as u64 == self.identity.length).then_some(bytes))
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
        self.sha256_with_hook(|| {})
    }

    pub(crate) fn sha256_with_hook(
        &mut self,
        after_first_chunk: impl FnOnce(),
    ) -> Result<Option<String>> {
        use sha2::{Digest, Sha256};

        self.file.rewind()?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0_u8; 1024 * 1024];
        let mut total = 0_u64;
        let mut hook = Some(after_first_chunk);
        let mut limited = (&mut self.file).take(self.identity.length.saturating_add(1));
        loop {
            let read = limited.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            total = total.saturating_add(read as u64);
            if let Some(hook) = hook.take() {
                hook();
            }
        }
        self.is_stable().map(|stable| {
            (stable && total == self.identity.length).then(|| hex::encode(hasher.finalize()))
        })
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
        self.is_stable().map(|stable| {
            (stable && total == self.identity.length)
                .then(|| (total, hex::encode(hasher.finalize())))
        })
    }
}

impl StableFileIdentity {
    pub(crate) fn same_inode(self, other: Self) -> bool {
        self.device == other.device && self.inode == other.inode
    }
}

pub(crate) fn regular_path_matches_identity(
    path: &Path,
    expected: StableFileIdentity,
) -> Result<bool> {
    let Some(file) = StableRegularFile::open_exact(path, expected.length)? else {
        return Ok(false);
    };
    Ok(file.identity == expected && file.is_stable()?)
}

fn reopen_walked_path(namespace: &WalkedNamespace) -> Result<Option<fs::File>> {
    use rustix::fs::{Mode, OFlags};

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
}
