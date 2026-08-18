//! Best-effort, zero-config installation of clap shell completions.
//!
//! Release builds keep hf2q-owned static completion scripts current in the
//! standard per-user Bash, Zsh, and Fish locations. The preferred Bash or Zsh
//! startup file receives one bounded source block so completion works in a new
//! shell without a manual `source` command. Debug/test binaries write only when
//! explicit destinations are supplied, which prevents a local build from
//! replacing the operator's installed completion.
//!
//! Every error is contained here: completion setup must never make an hf2q
//! command fail. Existing unmarked files, symlinks, non-regular files, and
//! malformed startup blocks are preserved.

use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};

use clap::CommandFactory as _;
use clap_complete::{generate, Shell};

use super::Cli;

const MARKER: &str = "# hf2q-managed shell completion — auto-provisioned; edits are overwritten";
const MARKER_PREFIX: &str = "# hf2q-managed shell completion";
const BEGIN: &[u8] = b"# >>> hf2q managed completion >>>";
const END: &[u8] = b"# <<< hf2q managed completion <<<";

const OPT_OUT_VAR: &str = "HF2Q_NO_COMPLETION_INSTALL";
const ZSH_DIR_VAR: &str = "HF2Q_ZSH_COMPLETIONS_DIR";
const FISH_DIR_VAR: &str = "HF2Q_FISH_COMPLETIONS_DIR";
const STARTUP_FILE_VAR: &str = "HF2Q_COMPLETION_STARTUP_FILE";
const ZSH_STARTUP_DIR_VAR: &str = "HF2Q_ZSH_STARTUP_DIR";

#[derive(Clone, Copy)]
struct CompletionShell {
    name: &'static str,
    file: &'static str,
    generator: Shell,
    marker_line: usize,
}

const BASH: CompletionShell = CompletionShell {
    name: "bash",
    file: "hf2q",
    generator: Shell::Bash,
    marker_line: 0,
};
const ZSH: CompletionShell = CompletionShell {
    name: "zsh",
    file: "_hf2q",
    generator: Shell::Zsh,
    // `#compdef hf2q` must remain the first line for zsh autoloading.
    marker_line: 1,
};
const FISH: CompletionShell = CompletionShell {
    name: "fish",
    file: "hf2q.fish",
    generator: Shell::Fish,
    marker_line: 0,
};

/// Reconcile managed completion files and the preferred shell's startup block.
///
/// This is intentionally infallible at the CLI boundary. Each shell is handled
/// independently, and any path, permission, rendering, or write failure is
/// swallowed so normal hf2q behavior is unchanged.
pub(crate) fn reconcile() {
    if std::env::var_os(OPT_OUT_VAR).is_some() {
        return;
    }

    let automatic = automatic_destinations_enabled();
    let bash = completion_dir(&BASH, automatic)
        .and_then(|dir| reconcile_registration(&BASH, &dir).ok().flatten());
    let zsh = completion_dir(&ZSH, automatic)
        .and_then(|dir| reconcile_registration(&ZSH, &dir).ok().flatten());
    let _fish = completion_dir(&FISH, automatic)
        .and_then(|dir| reconcile_registration(&FISH, &dir).ok().flatten());

    let shell = std::env::var_os("SHELL")
        .and_then(|value| Path::new(&value).file_name().map(OsStr::to_os_string));
    let registration = match shell.as_deref() {
        Some(name) if name == OsStr::new("bash") => bash.as_deref(),
        Some(name) if name == OsStr::new("zsh") => zsh.as_deref(),
        _ => None,
    };
    let Some(registration) = registration else {
        return;
    };
    let Some(block) = startup_block(shell.as_deref().unwrap_or_default(), registration) else {
        return;
    };
    for startup in startup_files(shell.as_deref().unwrap_or_default(), automatic) {
        let _ = reconcile_startup_file(&startup, &block);
    }
}

fn automatic_destinations_enabled() -> bool {
    if cfg!(debug_assertions) {
        return false;
    }
    #[cfg(unix)]
    {
        // SAFETY: `geteuid` has no memory or aliasing preconditions and cannot
        // fail. A sudo/root invocation must not leave root-owned shell files in
        // another account's HOME; explicit destinations remain available.
        return unsafe { libc::geteuid() } != 0;
    }
    #[cfg(not(unix))]
    true
}

fn completion_dir(shell: &CompletionShell, automatic: bool) -> Option<PathBuf> {
    match shell.name {
        "bash" => {
            if let Some(base) = non_empty_env("BASH_COMPLETION_USER_DIR") {
                return Some(PathBuf::from(base).join("completions"));
            }
            automatic
                .then(data_home)?
                .map(|base| base.join("bash-completion").join("completions"))
        }
        "zsh" => {
            if let Some(dir) = non_empty_env(ZSH_DIR_VAR) {
                return Some(PathBuf::from(dir));
            }
            automatic
                .then(data_home)?
                .map(|base| base.join("zsh/site-functions"))
        }
        "fish" => {
            if let Some(dir) = non_empty_env(FISH_DIR_VAR) {
                return Some(PathBuf::from(dir));
            }
            automatic
                .then(config_home)?
                .map(|base| base.join("fish/completions"))
        }
        _ => None,
    }
}

fn non_empty_env(name: &str) -> Option<OsString> {
    std::env::var_os(name).filter(|value| !value.is_empty())
}

fn data_home() -> Option<PathBuf> {
    non_empty_env("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| non_empty_env("HOME").map(|home| PathBuf::from(home).join(".local/share")))
}

fn config_home() -> Option<PathBuf> {
    non_empty_env("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| non_empty_env("HOME").map(|home| PathBuf::from(home).join(".config")))
}

/// Write or refresh one managed registration. `Ok(Some(path))` means the
/// exact desired regular file is installed and safe for a startup block to
/// source. Foreign occupants and non-regular paths return `Ok(None)`.
fn reconcile_registration(shell: &CompletionShell, dir: &Path) -> io::Result<Option<PathBuf>> {
    fs::create_dir_all(dir)?;
    let target = dir.join(shell.file);
    let desired = render_registration(shell)?;

    match fs::symlink_metadata(&target) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            write_atomic(&target, &desired, None, 0o644)?;
        }
        Ok(metadata) if metadata.file_type().is_file() => {
            let existing = fs::read(&target)?;
            if !is_managed(&existing, shell.marker_line) {
                return Ok(None);
            }
            if existing != desired {
                write_atomic(
                    &target,
                    &desired,
                    Some(&existing),
                    file_mode_or(&metadata, 0o644),
                )?;
            }
        }
        Ok(_) => return Ok(None),
        Err(error) => return Err(error),
    }

    let metadata = fs::symlink_metadata(&target)?;
    if metadata.file_type().is_file() && fs::read(&target)? == desired {
        Ok(Some(target))
    } else {
        Ok(None)
    }
}

fn render_registration(shell: &CompletionShell) -> io::Result<Vec<u8>> {
    let mut command = Cli::command();
    let mut generated = Vec::new();
    generate(shell.generator, &mut command, "hf2q", &mut generated);

    let mut managed = Vec::with_capacity(generated.len() + MARKER.len() + 1);
    if shell.marker_line == 0 {
        writeln!(managed, "{MARKER}")?;
        managed.extend_from_slice(&generated);
        return Ok(managed);
    }

    let newline = generated
        .iter()
        .position(|byte| *byte == b'\n')
        .ok_or_else(|| io::Error::other("zsh completion has no first-line newline"))?;
    managed.extend_from_slice(&generated[..=newline]);
    writeln!(managed, "{MARKER}")?;
    managed.extend_from_slice(&generated[newline + 1..]);
    Ok(managed)
}

fn is_managed(bytes: &[u8], marker_line: usize) -> bool {
    bytes
        .split(|byte| *byte == b'\n')
        .nth(marker_line)
        .map(|line| line.strip_suffix(b"\r").unwrap_or(line))
        .is_some_and(|line| line.starts_with(MARKER_PREFIX.as_bytes()))
}

fn write_atomic(
    target: &Path,
    desired: &[u8],
    expected: Option<&[u8]>,
    mode: u32,
) -> io::Result<()> {
    let parent = target
        .parent()
        .ok_or_else(|| io::Error::other("completion target has no parent"))?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(desired)?;
    temporary.flush()?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        temporary
            .as_file()
            .set_permissions(fs::Permissions::from_mode(mode))?;
    }

    #[cfg(not(unix))]
    let _ = mode;

    match expected {
        None => match temporary.persist_noclobber(target) {
            Ok(_) => Ok(()),
            Err(error) if error.error.kind() == io::ErrorKind::AlreadyExists => Ok(()),
            Err(error) => Err(error.error),
        },
        Some(expected) => {
            let metadata = fs::symlink_metadata(target)?;
            if !metadata.file_type().is_file() || fs::read(target)? != expected {
                return Err(io::Error::other(
                    "completion target changed during reconciliation",
                ));
            }
            temporary
                .persist(target)
                .map(|_| ())
                .map_err(|error| error.error)
        }
    }
}

fn file_mode_or(metadata: &fs::Metadata, _default: u32) -> u32 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        metadata.permissions().mode() & 0o7777
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        _default
    }
}

fn startup_files(shell: &OsStr, automatic: bool) -> Vec<PathBuf> {
    if let Some(path) = non_empty_env(STARTUP_FILE_VAR) {
        return vec![PathBuf::from(path)];
    }
    if !automatic {
        return Vec::new();
    }

    if shell == OsStr::new("zsh") {
        let root = non_empty_env(ZSH_STARTUP_DIR_VAR)
            .or_else(|| non_empty_env("ZDOTDIR"))
            .or_else(|| non_empty_env("HOME"));
        return root
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
            .map(|path| vec![path.join(".zshrc")])
            .unwrap_or_default();
    }
    if shell != OsStr::new("bash") {
        return Vec::new();
    }

    let Some(home) = non_empty_env("HOME").map(PathBuf::from) else {
        return Vec::new();
    };
    let mut paths = vec![home.join(".bashrc")];
    let login = [".bash_profile", ".bash_login", ".profile"]
        .into_iter()
        .map(|name| home.join(name))
        .find(|path| path.exists())
        .unwrap_or_else(|| home.join(".profile"));
    paths.push(login);
    paths
}

fn shell_quote(path: &Path) -> Option<String> {
    path.to_str()
        .map(|value| format!("'{}'", value.replace('\'', "'\"'\"'")))
}

fn startup_block(shell: &OsStr, registration: &Path) -> Option<Vec<u8>> {
    let registration = shell_quote(registration)?;
    let block = if shell == OsStr::new("zsh") {
        format!(
            "{}\nif [[ -z ${{{OPT_OUT_VAR}+x}} ]]; then\n  () {{\n    local completion_file={registration}\n    local compdef_line ownership_line\n    if [[ -f \"$completion_file\" && ! -L \"$completion_file\" ]]; then\n      {{\n        IFS= read -r compdef_line\n        IFS= read -r ownership_line\n      }} < \"$completion_file\"\n      if [[ \"$compdef_line\" == '#compdef hf2q' &&\n            \"$ownership_line\" == '{MARKER_PREFIX}'* ]]; then\n        autoload -Uz compinit\n        (( $+functions[compdef] )) || compinit -i\n        source \"$completion_file\" || true\n      fi\n    fi\n    return 0\n  }}\nfi\n{}\n",
            String::from_utf8_lossy(BEGIN),
            String::from_utf8_lossy(END),
        )
    } else if shell == OsStr::new("bash") {
        format!(
            "{}\nif [ -z \"${{{OPT_OUT_VAR}+x}}\" ]; then\n  case $- in\n    *i*)\n      __hf2q_completion_file={registration}\n      __hf2q_completion_owner=\n      if [ -f \"$__hf2q_completion_file\" ] &&\n         [ ! -L \"$__hf2q_completion_file\" ] &&\n         IFS= read -r __hf2q_completion_owner < \"$__hf2q_completion_file\"; then\n        case \"$__hf2q_completion_owner\" in\n          '{MARKER_PREFIX}'*) . \"$__hf2q_completion_file\" || : ;;\n        esac\n      fi\n      unset __hf2q_completion_file __hf2q_completion_owner\n      ;;\n  esac\nfi\n{}\n",
            String::from_utf8_lossy(BEGIN),
            String::from_utf8_lossy(END),
        )
    } else {
        return None;
    };
    Some(block.into_bytes())
}

fn reconcile_startup_file(logical: &Path, block: &[u8]) -> io::Result<()> {
    let target = match fs::symlink_metadata(logical) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            let target = fs::canonicalize(logical)?;
            if !fs::metadata(&target)?.is_file() {
                return Ok(());
            }
            target
        }
        Ok(metadata) if metadata.file_type().is_file() => logical.to_path_buf(),
        Ok(_) => return Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => logical.to_path_buf(),
        Err(error) => return Err(error),
    };

    let target_metadata = fs::symlink_metadata(&target).ok();
    let existing = match fs::read(&target) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error),
    };
    let Some(updated) = reconcile_block(&existing, block) else {
        return Ok(());
    };
    if updated == existing {
        return Ok(());
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    write_atomic(
        &target,
        &updated,
        target_metadata.as_ref().map(|_| existing.as_slice()),
        target_metadata
            .as_ref()
            .map_or(0o644, |metadata| file_mode_or(metadata, 0o644)),
    )
}

/// Replace one exact managed block, append when none exists, and preserve an
/// ambiguous or malformed marker layout.
fn reconcile_block(existing: &[u8], block: &[u8]) -> Option<Vec<u8>> {
    let begins = exact_line_ranges(existing, BEGIN);
    let ends = exact_line_ranges(existing, END);
    match (begins.as_slice(), ends.as_slice()) {
        ([], []) => {
            let mut updated = existing.to_vec();
            if !updated.is_empty() && !updated.ends_with(b"\n") {
                updated.push(b'\n');
            }
            updated.extend_from_slice(block);
            Some(updated)
        }
        ([(begin, _)], [(_, end)]) if begin < end => {
            let mut updated = Vec::with_capacity(existing.len() + block.len());
            updated.extend_from_slice(&existing[..*begin]);
            updated.extend_from_slice(block);
            updated.extend_from_slice(&existing[*end..]);
            Some(updated)
        }
        _ => None,
    }
}

fn exact_line_ranges(bytes: &[u8], marker: &[u8]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < bytes.len() {
        let newline = bytes[start..]
            .iter()
            .position(|byte| *byte == b'\n')
            .map(|offset| start + offset);
        let end = newline.unwrap_or(bytes.len());
        let line = bytes[start..end]
            .strip_suffix(b"\r")
            .unwrap_or(&bytes[start..end]);
        if line == marker {
            ranges.push((start, newline.map_or(end, |index| index + 1)));
        }
        start = newline.map_or(bytes.len(), |index| index + 1);
    }
    ranges
}

#[cfg(test)]
#[path = "completion_install_tests.rs"]
mod tests;
