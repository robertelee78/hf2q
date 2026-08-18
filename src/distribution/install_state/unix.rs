use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs::File;
use std::os::fd::OwnedFd;
use std::path::{Component, Path};

use rustix::fs::{self, AtFlags, Dir, FileType, FlockOperation, Mode, OFlags, RenameFlags, Stat};
use rustix::io::fcntl_dupfd_cloexec;

use super::InstallStateError;

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

    pub(super) fn inode(&self) -> u64 {
        self.stat.st_ino as u64
    }

    pub(super) fn same_object(&self, other: &Self) -> bool {
        self.device() == other.device() && self.inode() == other.inode()
    }
}

pub(super) fn duplicate_directory(directory: &Directory) -> Result<Directory, InstallStateError> {
    let fd = fcntl_dupfd_cloexec(directory.fd(), 0)
        .map_err(|error| InstallStateError::io("duplicate directory descriptor", error))?;
    let stat = fs::fstat(&fd)
        .map_err(|error| InstallStateError::io("inspect duplicated directory", error))?;
    require_same_identity(
        &directory.stat,
        &stat,
        "directory changed while duplicating its descriptor",
    )?;
    Ok(Directory { fd, stat })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct EntryIdentity {
    pub(super) device: u64,
    pub(super) inode: u64,
    pub(super) file_type: FileType,
    pub(super) mode: u32,
    pub(super) uid: u32,
    pub(super) links: u64,
    pub(super) size: u64,
}

pub(super) fn open_or_create_root(path: &Path) -> Result<Directory, InstallStateError> {
    let components = authorized_components(path)?;
    let root_fd = fs::open("/", DIRECTORY_FLAGS, Mode::empty())
        .map_err(|error| InstallStateError::io("open filesystem root", error))?;
    let root_stat = fs::fstat(&root_fd)
        .map_err(|error| InstallStateError::io("inspect filesystem root", error))?;
    let mut current = Directory {
        fd: root_fd,
        stat: root_stat,
    };

    for (index, component) in components.iter().enumerate() {
        let is_final = index + 1 == components.len();
        current = match open_directory_at_policy(&current, component, None, false, false) {
            Ok(directory) => directory,
            Err(InstallStateError::Missing(_)) if is_final => {
                fs::mkdirat(current.fd(), component.as_str(), Mode::from_raw_mode(0o700))
                    .or_else(|error| {
                        if error == rustix::io::Errno::EXIST {
                            Ok(())
                        } else {
                            Err(error)
                        }
                    })
                    .map_err(|error| InstallStateError::io("create authorized root", error))?;
                sync_directory(&current)?;
                open_directory_at_policy(&current, component, Some(0o700), true, false)?
            }
            Err(InstallStateError::Missing(_)) => {
                return Err(InstallStateError::Missing("explicit root ancestor"))
            }
            Err(error) => return Err(error),
        };
        if is_final {
            require_directory_policy(&current.stat, Some(0o700), true, None, "state root")?;
        }
    }
    Ok(current)
}

pub(super) fn open_existing_root(path: &Path) -> Result<Directory, InstallStateError> {
    let components = authorized_components(path)?;
    let root_fd = fs::open("/", DIRECTORY_FLAGS, Mode::empty())
        .map_err(|error| InstallStateError::io("open filesystem root", error))?;
    let root_stat = fs::fstat(&root_fd)
        .map_err(|error| InstallStateError::io("inspect filesystem root", error))?;
    let mut current = Directory {
        fd: root_fd,
        stat: root_stat,
    };

    for (index, component) in components.iter().enumerate() {
        current = open_directory_at_policy(&current, component, None, false, false)?;
        if index + 1 == components.len() {
            require_directory_policy(&current.stat, Some(0o700), true, None, "state root")?;
        }
    }
    Ok(current)
}

pub(super) fn ensure_private_directory(
    parent: &Directory,
    name: &str,
) -> Result<Directory, InstallStateError> {
    validate_component(name)?;
    match open_directory_at(parent, name, Some(0o700), true) {
        Ok(directory) => Ok(directory),
        Err(InstallStateError::Missing(_)) => {
            fs::mkdirat(parent.fd(), name, Mode::from_raw_mode(0o700))
                .or_else(|error| {
                    if error == rustix::io::Errno::EXIST {
                        Ok(())
                    } else {
                        Err(error)
                    }
                })
                .map_err(|error| InstallStateError::io("create private directory", error))?;
            sync_directory(parent)?;
            open_directory_at(parent, name, Some(0o700), true)
        }
        Err(error) => Err(error),
    }
}

pub(super) fn open_directory_at(
    parent: &Directory,
    name: &str,
    expected_mode: Option<u32>,
    require_owner: bool,
) -> Result<Directory, InstallStateError> {
    open_directory_at_policy(parent, name, expected_mode, require_owner, true)
}

fn open_directory_at_policy(
    parent: &Directory,
    name: &str,
    expected_mode: Option<u32>,
    require_owner: bool,
    require_same_device: bool,
) -> Result<Directory, InstallStateError> {
    validate_component(name)?;
    let named = match fs::statat(parent.fd(), name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) => stat,
        Err(rustix::io::Errno::NOENT) => return Err(InstallStateError::Missing("directory")),
        Err(error) => return Err(InstallStateError::io("inspect directory entry", error)),
    };
    if FileType::from_raw_mode(named.st_mode) != FileType::Directory {
        return Err(InstallStateError::InvalidLayout(
            "expected directory entry has the wrong type",
        ));
    }
    let fd = fs::openat(parent.fd(), name, DIRECTORY_FLAGS, Mode::empty())
        .map_err(|error| InstallStateError::io("open directory entry", error))?;
    let opened =
        fs::fstat(&fd).map_err(|error| InstallStateError::io("inspect opened directory", error))?;
    require_same_identity(&named, &opened, "directory changed while opening")?;
    require_directory_policy(
        &opened,
        expected_mode,
        require_owner,
        require_same_device.then(|| parent.device()),
        "directory",
    )?;
    Ok(Directory { fd, stat: opened })
}

pub(super) fn entry_identity(
    parent: &Directory,
    name: &str,
) -> Result<Option<EntryIdentity>, InstallStateError> {
    validate_component(name)?;
    match fs::statat(parent.fd(), name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) => Ok(Some(identity(&stat))),
        Err(rustix::io::Errno::NOENT) => Ok(None),
        Err(error) => Err(InstallStateError::io("inspect named entry", error)),
    }
}

pub(super) fn sync_directory(directory: &Directory) -> Result<(), InstallStateError> {
    fs::fsync(directory.fd()).map_err(|error| InstallStateError::io("sync directory", error))
}

pub(super) fn full_sync_file(file: &File) -> Result<(), InstallStateError> {
    #[cfg(target_os = "macos")]
    {
        fs::fcntl_fullfsync(file)
            .map_err(|error| InstallStateError::io("full-sync committed installation", error))
    }
    #[cfg(not(target_os = "macos"))]
    {
        fs::fsync(file).map_err(|error| InstallStateError::io("sync committed installation", error))
    }
}

pub(super) fn create_symlink(
    parent: &Directory,
    name: &str,
    target: &str,
) -> Result<(), InstallStateError> {
    validate_component(name)?;
    if target.is_empty() || target.as_bytes().len() > 512 || target.as_bytes().contains(&0) {
        return Err(InstallStateError::InvalidLayout(
            "symlink target is outside the bounded contract",
        ));
    }
    fs::symlinkat(target, parent.fd(), name)
        .map_err(|error| InstallStateError::io("create bounded symlink", error))
}

pub(super) fn read_symlink(parent: &Directory, name: &str) -> Result<String, InstallStateError> {
    validate_component(name)?;
    let before = fs::statat(parent.fd(), name, AtFlags::SYMLINK_NOFOLLOW)
        .map_err(|error| InstallStateError::io("inspect symlink", error))?;
    if FileType::from_raw_mode(before.st_mode) != FileType::Symlink
        || before.st_uid != rustix::process::geteuid().as_raw()
        || before.st_nlink != 1
        || before.st_dev as u64 != parent.device()
    {
        return Err(InstallStateError::InvalidLayout(
            "symlink ownership or type is invalid",
        ));
    }
    let target = fs::readlinkat(parent.fd(), name, Vec::new())
        .map_err(|error| InstallStateError::io("read bounded symlink", error))?;
    let after = fs::statat(parent.fd(), name, AtFlags::SYMLINK_NOFOLLOW)
        .map_err(|error| InstallStateError::io("reinspect symlink", error))?;
    require_same_identity(&before, &after, "symlink changed while reading")?;
    let bytes = target.as_bytes();
    if bytes.len() > 512 {
        return Err(InstallStateError::InvalidLayout(
            "symlink target exceeds its input bound",
        ));
    }
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|_| InstallStateError::InvalidLayout("symlink target is not UTF-8"))
}

pub(super) fn rename_noreplace(
    from_parent: &Directory,
    from: &str,
    to_parent: &Directory,
    to: &str,
) -> Result<(), InstallStateError> {
    validate_component(from)?;
    validate_component(to)?;
    fs::renameat_with(
        from_parent.fd(),
        from,
        to_parent.fd(),
        to,
        RenameFlags::NOREPLACE,
    )
    .map_err(|error| InstallStateError::io("publish no-replace entry", error))
}

pub(super) fn rename_replace(
    parent: &Directory,
    from: &str,
    to: &str,
) -> Result<(), InstallStateError> {
    validate_component(from)?;
    validate_component(to)?;
    fs::renameat(parent.fd(), from, parent.fd(), to)
        .map_err(|error| InstallStateError::io("atomically replace named entry", error))
}

pub(super) fn regular_file_identity(
    file: &File,
    expected_device: u64,
) -> Result<EntryIdentity, InstallStateError> {
    let stat = fs::fstat(file)
        .map_err(|error| InstallStateError::io("inspect opened regular file", error))?;
    require_regular_policy(&stat, 0o600, expected_device)?;
    Ok(identity(&stat))
}

pub(super) fn open_private_regular_file(
    parent: &Directory,
    name: &str,
) -> Result<(File, EntryIdentity), InstallStateError> {
    validate_component(name)?;
    let named = fs::statat(parent.fd(), name, AtFlags::SYMLINK_NOFOLLOW)
        .map_err(|error| InstallStateError::io("inspect private regular file", error))?;
    require_regular_policy(&named, 0o600, parent.device())?;
    let fd = fs::openat(
        parent.fd(),
        name,
        OFlags::RDWR | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|error| InstallStateError::io("open private regular file", error))?;
    let file = File::from(fd);
    let opened = fs::fstat(&file)
        .map_err(|error| InstallStateError::io("inspect opened private regular file", error))?;
    require_same_identity(
        &named,
        &opened,
        "private regular file changed while opening",
    )?;
    require_regular_policy(&opened, 0o600, parent.device())?;
    Ok((file, identity(&opened)))
}

pub(super) fn create_private_regular_file(
    parent: &Directory,
    name: &str,
) -> Result<(File, EntryIdentity), InstallStateError> {
    validate_component(name)?;
    let fd = fs::openat(
        parent.fd(),
        name,
        OFlags::RDWR
            | OFlags::CREATE
            | OFlags::EXCL
            | OFlags::NOFOLLOW
            | OFlags::NONBLOCK
            | OFlags::CLOEXEC,
        Mode::from_raw_mode(0o600),
    )
    .map_err(|error| InstallStateError::io("create private regular file", error))?;
    let file = File::from(fd);
    let opened = fs::fstat(&file)
        .map_err(|error| InstallStateError::io("inspect created private regular file", error))?;
    require_regular_policy(&opened, 0o600, parent.device())?;
    let identity = identity(&opened);
    verify_named_identity(parent, name, identity)?;
    sync_directory(parent)?;
    verify_named_identity(parent, name, identity)?;
    Ok((file, identity))
}

pub(super) fn remove_named_regular_file(
    parent: &Directory,
    name: &str,
    expected: EntryIdentity,
) -> Result<(), InstallStateError> {
    validate_component(name)?;
    verify_named_identity(parent, name, expected)?;
    fs::unlinkat(parent.fd(), name, AtFlags::empty())
        .map_err(|error| InstallStateError::io("remove verified regular file", error))
}

pub(super) fn remove_empty_directory(
    parent: &Directory,
    name: &str,
    expected: &Directory,
) -> Result<(), InstallStateError> {
    validate_component(name)?;
    let named = open_directory_at(parent, name, Some(0o700), true)?;
    if !named.same_object(expected) || !list_names(&named)?.is_empty() {
        return Err(InstallStateError::InvalidLayout(
            "directory changed or is not empty before removal",
        ));
    }
    fs::unlinkat(parent.fd(), name, AtFlags::REMOVEDIR)
        .map_err(|error| InstallStateError::io("remove verified empty directory", error))
}

pub(super) fn preflight_noreplace(
    parent: &Directory,
    existing_source: &str,
    existing_target: &str,
) -> Result<(), InstallStateError> {
    validate_component(existing_source)?;
    validate_component(existing_target)?;
    match fs::renameat_with(
        parent.fd(),
        existing_source,
        parent.fd(),
        existing_target,
        RenameFlags::NOREPLACE,
    ) {
        Err(rustix::io::Errno::EXIST) => Ok(()),
        Ok(()) => Err(InstallStateError::InvalidLayout(
            "filesystem did not enforce no-replace semantics",
        )),
        Err(error) => Err(InstallStateError::io(
            "preflight no-replace rename support",
            error,
        )),
    }
}

pub(super) fn list_names(directory: &Directory) -> Result<BTreeSet<String>, InstallStateError> {
    list_names_bounded(directory, usize::MAX)
}

pub(super) fn list_names_bounded(
    directory: &Directory,
    maximum: usize,
) -> Result<BTreeSet<String>, InstallStateError> {
    let mut stream = Dir::read_from(directory.fd())
        .map_err(|error| InstallStateError::io("open directory inventory", error))?;
    let mut names = BTreeSet::new();
    while let Some(entry) = stream.read() {
        let entry =
            entry.map_err(|error| InstallStateError::io("read directory inventory", error))?;
        let bytes = entry.file_name().to_bytes();
        if bytes == b"." || bytes == b".." {
            continue;
        }
        let name = std::str::from_utf8(bytes)
            .map_err(|_| InstallStateError::InvalidLayout("directory entry name is not UTF-8"))?;
        validate_component(name)?;
        if names.len() >= maximum {
            return Err(InstallStateError::InvalidLayout(
                "directory inventory exceeds its bounded maximum",
            ));
        }
        if !names.insert(name.to_owned()) {
            return Err(InstallStateError::InvalidLayout(
                "directory inventory contains a duplicate entry",
            ));
        }
    }
    Ok(names)
}

pub(super) fn acquire_nonblocking_lock(
    update: &Directory,
) -> Result<(File, EntryIdentity), InstallStateError> {
    const LOCK_NAME: &str = "install.lock";
    let create_flags = OFlags::RDWR
        | OFlags::CREATE
        | OFlags::EXCL
        | OFlags::NOFOLLOW
        | OFlags::NONBLOCK
        | OFlags::CLOEXEC;
    let fd = match fs::openat(
        update.fd(),
        LOCK_NAME,
        create_flags,
        Mode::from_raw_mode(0o600),
    ) {
        Ok(fd) => {
            sync_directory(update)?;
            fd
        }
        Err(rustix::io::Errno::EXIST) => fs::openat(
            update.fd(),
            LOCK_NAME,
            OFlags::RDWR | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|error| InstallStateError::io("open installation lock", error))?,
        Err(error) => return Err(InstallStateError::io("create installation lock", error)),
    };
    let file = File::from(fd);
    let opened = fs::fstat(&file)
        .map_err(|error| InstallStateError::io("inspect installation lock", error))?;
    require_regular_policy(&opened, 0o600, update.device())?;
    match fs::flock(&file, FlockOperation::NonBlockingLockExclusive) {
        Ok(()) => {}
        Err(rustix::io::Errno::WOULDBLOCK) => return Err(InstallStateError::Busy),
        Err(error) => return Err(InstallStateError::io("lock installation state", error)),
    }
    let named = fs::statat(update.fd(), LOCK_NAME, AtFlags::SYMLINK_NOFOLLOW)
        .map_err(|error| InstallStateError::io("reinspect installation lock", error))?;
    require_same_identity(&opened, &named, "installation lock name changed")?;
    Ok((file, identity(&opened)))
}

pub(super) fn verify_named_identity(
    parent: &Directory,
    name: &str,
    expected: EntryIdentity,
) -> Result<(), InstallStateError> {
    let stat = fs::statat(parent.fd(), name, AtFlags::SYMLINK_NOFOLLOW)
        .map_err(|error| InstallStateError::io("revalidate named entry", error))?;
    if identity(&stat) != expected {
        return Err(InstallStateError::InvalidLayout(
            "named entry changed after verification",
        ));
    }
    Ok(())
}

fn authorized_components(path: &Path) -> Result<Vec<String>, InstallStateError> {
    let mut saw_root = false;
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            Component::RootDir if !saw_root && components.is_empty() => saw_root = true,
            Component::Normal(value) if saw_root => {
                let value = utf8_component(value)?;
                validate_component(value)?;
                components.push(value.to_owned());
            }
            _ => {
                return Err(InstallStateError::InvalidRoot(
                    "root must be an absolute path of normal UTF-8 components",
                ))
            }
        }
    }
    if !saw_root || components.is_empty() {
        return Err(InstallStateError::InvalidRoot(
            "root cannot be relative or the filesystem root",
        ));
    }
    Ok(components)
}

fn utf8_component(value: &OsStr) -> Result<&str, InstallStateError> {
    value
        .to_str()
        .ok_or(InstallStateError::InvalidRoot("root path must be UTF-8"))
}

pub(super) fn validate_component(value: &str) -> Result<(), InstallStateError> {
    if value.is_empty()
        || value == "."
        || value == ".."
        || value.as_bytes().len() > 255
        || value.contains('/')
        || value.contains('\\')
        || value.as_bytes().contains(&0)
        || value.chars().any(char::is_control)
    {
        return Err(InstallStateError::InvalidLayout(
            "filesystem component is outside the bounded contract",
        ));
    }
    Ok(())
}

fn require_directory_policy(
    stat: &Stat,
    expected_mode: Option<u32>,
    require_owner: bool,
    expected_device: Option<u64>,
    label: &'static str,
) -> Result<(), InstallStateError> {
    if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
        return Err(InstallStateError::InvalidLayout(
            "expected directory has the wrong type",
        ));
    }
    if require_owner && stat.st_uid != rustix::process::geteuid().as_raw() {
        return Err(InstallStateError::InvalidLayout(
            "expected directory has the wrong owner",
        ));
    }
    if expected_mode.is_some_and(|mode| permission_bits(stat) != mode) {
        return Err(InstallStateError::InvalidLayout(match label {
            "state root" => "state root permissions must be 0700",
            _ => "private directory permissions must be 0700",
        }));
    }
    if expected_device.is_some_and(|device| stat.st_dev as u64 != device) {
        return Err(InstallStateError::InvalidLayout(
            "directory crosses the installation device boundary",
        ));
    }
    Ok(())
}

pub(super) fn require_regular_policy(
    stat: &Stat,
    expected_mode: u32,
    expected_device: u64,
) -> Result<(), InstallStateError> {
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile
        || stat.st_uid != rustix::process::geteuid().as_raw()
        || stat.st_nlink != 1
        || permission_bits(stat) != expected_mode
        || stat.st_dev as u64 != expected_device
        || stat.st_size < 0
    {
        return Err(InstallStateError::InvalidLayout(
            "regular file type, owner, mode, link count, or device is invalid",
        ));
    }
    Ok(())
}

pub(super) fn require_same_identity(
    first: &Stat,
    second: &Stat,
    reason: &'static str,
) -> Result<(), InstallStateError> {
    if first.st_dev != second.st_dev
        || first.st_ino != second.st_ino
        || FileType::from_raw_mode(first.st_mode) != FileType::from_raw_mode(second.st_mode)
    {
        return Err(InstallStateError::InvalidLayout(reason));
    }
    Ok(())
}

fn permission_bits(stat: &Stat) -> u32 {
    Mode::from_raw_mode(stat.st_mode).as_raw_mode() as u32 & 0o7777
}

pub(super) fn identity(stat: &Stat) -> EntryIdentity {
    EntryIdentity {
        device: stat.st_dev as u64,
        inode: stat.st_ino as u64,
        file_type: FileType::from_raw_mode(stat.st_mode),
        mode: permission_bits(stat),
        uid: stat.st_uid,
        links: stat.st_nlink as u64,
        size: stat.st_size.max(0) as u64,
    }
}
