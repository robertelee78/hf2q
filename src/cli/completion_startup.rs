//! Preferred-shell startup bootstrap for managed completion (ADR-045).
//!
//! Registration files alone are not sufficient on a shell that has no
//! completion loader. A release build therefore owns small, bounded blocks in
//! the preferred shell's effective startup files. Zsh uses `.zshrc`; Bash uses
//! `.bashrc` plus the first login file Bash will read. The blocks are narrower
//! than an rc-file generator: bytes outside the two exact markers are never
//! rewritten, ambiguous marker layouts are preserved, and a startup-file
//! symlink remains a symlink while its regular referent is updated atomically.

use std::ffi::OsStr;
use std::fs;
use std::os::unix::ffi::OsStringExt as _;
use std::path::{Path, PathBuf};

use super::completion_install::{capture_regular_target, ExpectedTarget};

const BEGIN: &[u8] = b"# >>> hf2q managed completion >>>";
const END: &[u8] = b"# <<< hf2q managed completion <<<";
const OPT_OUT_VAR: &str = "HF2Q_NO_COMPLETION_INSTALL";
const STARTUP_FILE_VAR: &str = "HF2Q_COMPLETION_STARTUP_FILE";
const ZSH_STARTUP_DIR_VAR: &str = "HF2Q_ZSH_STARTUP_DIR";
const ZDOTDIR_PROBE_VAR: &str = "HF2Q_COMPLETION_ZDOTDIR_PROBE";
const ZDOTDIR_PROBE_MARKER: &[u8] = b"\0HF2Q_ZDOTDIR_V1\0";

#[derive(Debug)]
pub(super) enum Outcome {
    Wrote(PathBuf),
    UpToDate(PathBuf),
    PreservedMalformed(PathBuf),
    PreservedNonRegular(PathBuf),
    Failed(String),
}

/// Reconcile the canonical interactive startup file for `$SHELL`.
///
/// The caller supplies the exact registration location it already reconciled.
/// Fish has an official autoload directory and needs no startup bootstrap.
pub(super) fn reconcile_preferred_shell(
    bash_registration: Option<(&Path, &str)>,
    zsh_functions_dir: Option<(&Path, &str)>,
    allow_automatic: bool,
) -> Vec<(&'static str, Outcome)> {
    let explicit_startup = std::env::var_os(STARTUP_FILE_VAR)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);
    if explicit_startup.is_none() && !allow_automatic {
        return Vec::new();
    }
    let Some(shell) = std::env::var_os("SHELL") else {
        return Vec::new();
    };
    let shell_path = Path::new(&shell);
    match shell_path.file_name() {
        Some(name) if name == OsStr::new("zsh") => {
            let Some((dir, binding)) = zsh_functions_dir else {
                return Vec::new();
            };
            let path = match explicit_startup {
                Some(path) => path,
                None => match zsh_startup_root(shell_path) {
                    Ok(root) => root.join(".zshrc"),
                    Err(error) => return vec![("zsh startup", Outcome::Failed(error))],
                },
            };
            let outcome = match zsh_block(dir, binding) {
                Ok(block) => reconcile_file(&path, &block),
                Err(error) => Outcome::Failed(error),
            };
            vec![("zsh startup", outcome)]
        }
        Some(name) if name == OsStr::new("bash") => {
            let Some(home) = std::env::var_os("HOME").filter(|value| !value.is_empty()) else {
                return Vec::new();
            };
            let Some((registration, binding)) = bash_registration else {
                return Vec::new();
            };
            let home = PathBuf::from(home);
            let block = match bash_block(registration, binding) {
                Ok(block) => block,
                Err(error) => return vec![("bash startup", Outcome::Failed(error))],
            };
            if let Some(path) = explicit_startup {
                return vec![("bash startup", reconcile_file(&path, &block))];
            }
            let mut outcomes = vec![(
                "bash startup",
                reconcile_file(&home.join(".bashrc"), &block),
            )];
            let login = match bash_login_startup_path(&home) {
                Ok(path) => reconcile_file(&path, &block),
                Err(error) => Outcome::Failed(error),
            };
            outcomes.push(("bash login startup", login));
            outcomes
        }
        _ => Vec::new(),
    }
}

/// Resolve the directory an interactive preferred Zsh will actually use for
/// `.zshrc`. `ZDOTDIR` is a shell parameter and is commonly assigned without
/// `export` in `$HOME/.zshenv`, so inspecting only this process's environment
/// silently writes the wrong startup file. Querying the operator-selected Zsh
/// lets that normal startup contract resolve itself without parsing dotfiles.
fn zsh_startup_root(shell: &Path) -> Result<PathBuf, String> {
    for variable in [ZSH_STARTUP_DIR_VAR, "ZDOTDIR"] {
        if let Some(value) = std::env::var_os(variable).filter(|value| !value.is_empty()) {
            return validate_zsh_startup_root(PathBuf::from(value), variable);
        }
    }

    if std::env::var_os(ZDOTDIR_PROBE_VAR).is_some() {
        return Err(zsh_startup_recovery(
            "recursive effective-ZDOTDIR discovery was refused",
        ));
    }
    probe_effective_zdotdir(shell)
}

fn validate_zsh_startup_root(path: PathBuf, source: &str) -> Result<PathBuf, String> {
    if path.is_absolute() {
        Ok(path)
    } else {
        Err(zsh_startup_recovery(&format!(
            "{source} resolves to a relative directory"
        )))
    }
}

fn zsh_startup_recovery(reason: &str) -> String {
    format!(
        "{reason}; set {ZSH_STARTUP_DIR_VAR} to the absolute directory containing .zshrc, then run hf2q again"
    )
}

fn probe_effective_zdotdir(shell: &Path) -> Result<PathBuf, String> {
    probe_effective_zdotdir_with_runner(shell, |program, args, timeout_secs| {
        bounded_output(program, args, timeout_secs)
    })
}

fn probe_effective_zdotdir_with_runner<F>(shell: &Path, run: F) -> Result<PathBuf, String>
where
    F: FnOnce(&Path, &[&str], u64) -> Option<std::process::Output>,
{
    let shell = shell
        .to_str()
        .ok_or_else(|| zsh_startup_recovery("the preferred Zsh executable path is not UTF-8"))?;
    // Reuse the repository's one bounded synchronous process site. ADR-041's
    // physical-constructor gate intentionally forbids adding a second launch
    // here before its typed front door exists. `/usr/bin/env` applies the two
    // probe-only variables before Zsh reads `.zshenv`; a nested hf2q therefore
    // cannot recursively reconcile completion while the outer probe waits.
    let args = [
        "HF2Q_COMPLETION_ZDOTDIR_PROBE=1",
        "HF2Q_NO_COMPLETION_INSTALL=1",
        shell,
        "-c",
        "builtin print -rn -- $'\\0HF2Q_ZDOTDIR_V1\\0'\"${ZDOTDIR:-$HOME}\"$'\\0'",
    ];
    let output = run(Path::new("/usr/bin/env"), &args, 2).ok_or_else(|| {
        zsh_startup_recovery("the preferred Zsh startup probe failed or timed out")
    })?;
    if !output.status.success() {
        return Err(zsh_startup_recovery(
            "the preferred Zsh could not resolve its startup directory",
        ));
    }
    let marker = output
        .stdout
        .windows(ZDOTDIR_PROBE_MARKER.len())
        .rposition(|window| window == ZDOTDIR_PROBE_MARKER)
        .ok_or_else(|| zsh_startup_recovery("the preferred Zsh returned no startup directory"))?;
    let value = &output.stdout[marker + ZDOTDIR_PROBE_MARKER.len()..];
    let end = value.iter().position(|byte| *byte == 0).ok_or_else(|| {
        zsh_startup_recovery("the preferred Zsh returned an incomplete startup directory")
    })?;
    if end == 0 {
        return Err(zsh_startup_recovery(
            "the preferred Zsh has neither ZDOTDIR nor HOME",
        ));
    }
    validate_zsh_startup_root(
        PathBuf::from(std::ffi::OsString::from_vec(value[..end].to_vec())),
        "the preferred Zsh",
    )
}

/// Bash reads only the first resolved login startup file in this order. Its
/// existence check follows symlinks, so a dangling higher-priority link is
/// skipped while the link itself remains untouched. When no candidate resolves,
/// writing `.profile` adds completion without creating a new `.bash_profile`
/// that would shadow future/operator `.profile` content.
fn bash_login_startup_path(home: &Path) -> Result<PathBuf, String> {
    for name in [".bash_profile", ".bash_login", ".profile"] {
        let path = home.join(name);
        match fs::metadata(&path) {
            Ok(_) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("stat {}: {error}", path.display())),
        }
    }
    Ok(home.join(".profile"))
}

fn shell_quote(path: &Path) -> Result<String, String> {
    let value = path
        .to_str()
        .ok_or_else(|| format!("completion path is not UTF-8: {}", path.display()))?;
    Ok(format!("'{}'", value.replace('\'', "'\"'\"'")))
}

fn zsh_block(functions_dir: &Path, binding: &str) -> Result<Vec<u8>, String> {
    let dir = shell_quote(functions_dir)?;
    Ok(format!(
        r#"{}
if [[ -z ${{{OPT_OUT_VAR}+x}} ]]; then
  () {{
    emulate -L zsh
    local completion_dir={dir}
    local completion_file="$completion_dir/_hf2q"
    if [[ -f "$completion_file" && ! -L "$completion_file" ]]; then
      local compdef_line ownership_line binding_line
      {{
        IFS= read -r compdef_line
        IFS= read -r ownership_line
        IFS= read -r binding_line
      }} < "$completion_file"
      if [[ "$compdef_line" == '#compdef hf2q' &&
            "$ownership_line" == '{marker_prefix}'* &&
            "$binding_line" == '{binding}' ]]; then
        # Do not expose the candidate directory to compinit until the exact
        # managed headers above have matched. Initialize compdef from the
        # operator's existing fpath, then register the header-verified file
        # directly; a preserved foreign #compdef file is never scanned.
        if (( ! $+functions[compdef] )); then
          autoload -Uz compinit
          compinit -i
        fi
        if (( $+functions[compdef] )); then
          if [[ -d "$completion_dir" ]] && (( ${{fpath[(Ie)$completion_dir]}} == 0 )); then
            fpath=("$completion_dir" "${{fpath[@]}}")
          fi
          source "$completion_file" || return 0
        fi
      fi
    fi
    return 0
  }}
fi
{}
"#,
        String::from_utf8_lossy(BEGIN),
        String::from_utf8_lossy(END),
        marker_prefix = super::completion_install::MARKER_PREFIX,
        binding = binding,
    )
    .into_bytes())
}

fn bash_block(registration: &Path, binding: &str) -> Result<Vec<u8>, String> {
    let registration = shell_quote(registration)?;
    Ok(format!(
        r#"{}
if [ -n "${{BASH_VERSION-}}" ] && [ -z "${{{OPT_OUT_VAR}+x}}" ]; then
  case $- in
    *i*)
      __hf2q_managed_completion_bootstrap_v2() {{
        __hf2q_completion_file={registration}
        __hf2q_completion_owner=
        __hf2q_completion_binding=
        if [ -f "$__hf2q_completion_file" ] && [ ! -L "$__hf2q_completion_file" ] &&
           {{
             IFS= read -r __hf2q_completion_owner
             IFS= read -r __hf2q_completion_binding
           }} < "$__hf2q_completion_file"; then
          case "$__hf2q_completion_owner" in
            '{marker_prefix}'*)
              if [ "$__hf2q_completion_binding" = '{binding}' ]; then
                . "$__hf2q_completion_file" || :
              fi
              ;;
          esac
        fi
        unset __hf2q_completion_file __hf2q_completion_owner __hf2q_completion_binding
        return 0
      }}
      __hf2q_managed_completion_bootstrap_v2
      unset -f __hf2q_managed_completion_bootstrap_v2
      ;;
  esac
fi
{}
"#,
        String::from_utf8_lossy(BEGIN),
        String::from_utf8_lossy(END),
        marker_prefix = super::completion_install::MARKER_PREFIX,
        binding = binding,
    )
    .into_bytes())
}

fn reconcile_file(logical_path: &Path, desired_block: &[u8]) -> Outcome {
    reconcile_file_fallible(logical_path, desired_block).unwrap_or_else(Outcome::Failed)
}

fn reconcile_file_fallible(logical_path: &Path, desired_block: &[u8]) -> Result<Outcome, String> {
    let Some(target) = editable_target(logical_path)? else {
        return Ok(Outcome::PreservedNonRegular(logical_path.to_path_buf()));
    };
    let expected = match fs::symlink_metadata(&target) {
        Ok(metadata) if metadata.file_type().is_file() => capture_regular_target(&target)
            .map_err(|error| format!("reading {}: {error}", target.display()))?,
        Ok(_) => return Ok(Outcome::PreservedNonRegular(logical_path.to_path_buf())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => ExpectedTarget::Absent,
        Err(error) => return Err(format!("stat {}: {error}", target.display())),
    };
    let existing = expected.bytes();

    let Some(updated) = reconcile_block(existing, desired_block)? else {
        return Ok(Outcome::PreservedMalformed(logical_path.to_path_buf()));
    };
    if updated == existing {
        return Ok(Outcome::UpToDate(logical_path.to_path_buf()));
    }

    let parent = target
        .parent()
        .ok_or_else(|| format!("startup file has no parent: {}", target.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("creating {}: {error}", parent.display()))?;
    atomic_replace_startup_with_hook(&target, &updated, &expected, || {})
        .map_err(|error| format!("writing {}: {error}", target.display()))?;
    Ok(Outcome::Wrote(logical_path.to_path_buf()))
}

/// Resolve only the startup filename itself when it is a symlink. Updating the
/// regular referent preserves the operator's link; dangling/non-regular links
/// are returned as a preserve outcome by the caller.
fn editable_target(logical_path: &Path) -> Result<Option<PathBuf>, String> {
    match fs::symlink_metadata(logical_path) {
        Ok(metadata) if metadata.file_type().is_symlink() => match fs::canonicalize(logical_path) {
            Ok(target) => Ok(Some(target)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(format!(
                "resolving startup symlink {}: {error}",
                logical_path.display()
            )),
        },
        Ok(_) => Ok(Some(logical_path.to_path_buf())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(Some(logical_path.to_path_buf()))
        }
        Err(error) => Err(format!("stat {}: {error}", logical_path.display())),
    }
}

/// Return the complete updated file, or `None` when ownership markers are
/// ambiguous and the file must be preserved.
fn reconcile_block(existing: &[u8], desired: &[u8]) -> Result<Option<Vec<u8>>, String> {
    if !desired.starts_with(BEGIN) || !desired.ends_with(b"\n") {
        return Err("invalid managed startup block".to_owned());
    }
    let lines = line_ranges(existing);
    let begins = marker_lines(existing, &lines, BEGIN);
    let ends = marker_lines(existing, &lines, END);
    match (begins.as_slice(), ends.as_slice()) {
        ([], []) => {
            let mut updated = existing.to_vec();
            if !updated.is_empty() {
                if !updated.ends_with(b"\n") {
                    updated.push(b'\n');
                }
                if !updated.ends_with(b"\n\n") {
                    updated.push(b'\n');
                }
            }
            updated.extend_from_slice(desired);
            Ok(Some(updated))
        }
        ([begin], [end]) if begin < end => {
            let mut updated = Vec::with_capacity(existing.len() + desired.len());
            updated.extend_from_slice(&existing[..lines[*begin].0]);
            updated.extend_from_slice(desired);
            updated.extend_from_slice(&existing[lines[*end].2..]);
            Ok(Some(updated))
        }
        _ => Ok(None),
    }
}

fn managed_block_range(bytes: &[u8]) -> Option<(usize, usize)> {
    let lines = line_ranges(bytes);
    let begins = marker_lines(bytes, &lines, BEGIN);
    let ends = marker_lines(bytes, &lines, END);
    match (begins.as_slice(), ends.as_slice()) {
        ([begin], [end]) if begin < end => Some((lines[*begin].0, lines[*end].2)),
        _ => None,
    }
}

pub(super) fn managed_block_digest(logical_path: &Path) -> Result<Option<String>, String> {
    use sha2::{Digest as _, Sha256};

    let Some(target) = editable_target(logical_path)? else {
        return Ok(None);
    };
    let bytes = match fs::read(&target) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("reading {}: {error}", target.display())),
    };
    let Some((start, end)) = managed_block_range(&bytes) else {
        return Ok(None);
    };
    Ok(Some(format!("{:x}", Sha256::digest(&bytes[start..end]))))
}

/// Remove only the exact startup block named by an ownership receipt. Bytes
/// outside the block are preserved byte-for-byte. A modified or ambiguous
/// block is never removed and is returned as an actionable preservation.
pub(super) fn remove_managed_block(
    logical_path: &Path,
    expected_sha256: &str,
) -> Result<StartupCleanup, String> {
    use sha2::{Digest as _, Sha256};

    let Some(target) = editable_target(logical_path)? else {
        return Ok(StartupCleanup::Preserved(
            "startup path is a dangling or non-regular symlink".to_owned(),
        ));
    };
    let expected = match fs::symlink_metadata(&target) {
        Ok(metadata) if metadata.file_type().is_file() => capture_regular_target(&target)
            .map_err(|error| format!("reading {}: {error}", target.display()))?,
        Ok(_) => {
            return Ok(StartupCleanup::Preserved(
                "startup path is not a regular file".to_owned(),
            ));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(StartupCleanup::Absent);
        }
        Err(error) => return Err(format!("stat {}: {error}", target.display())),
    };
    let bytes = expected.bytes();
    let Some((start, end)) = managed_block_range(bytes) else {
        return Ok(StartupCleanup::Preserved(
            "managed startup markers are missing or ambiguous".to_owned(),
        ));
    };
    let digest = format!("{:x}", Sha256::digest(&bytes[start..end]));
    if digest != expected_sha256 {
        return Ok(StartupCleanup::Preserved(
            "managed startup block was modified after installation".to_owned(),
        ));
    }
    let mut updated = Vec::with_capacity(bytes.len() - (end - start));
    updated.extend_from_slice(&bytes[..start]);
    updated.extend_from_slice(&bytes[end..]);
    if updated.is_empty() && target == logical_path {
        expected
            .revalidate(&target)
            .map_err(|error| format!("revalidating {}: {error}", target.display()))?;
        fs::remove_file(&target)
            .map_err(|error| format!("removing {}: {error}", target.display()))?;
    } else {
        atomic_replace_startup_with_hook(&target, &updated, &expected, || {})
            .map_err(|error| format!("writing {}: {error}", target.display()))?;
    }
    Ok(StartupCleanup::Removed)
}

pub(super) enum StartupCleanup {
    Removed,
    Absent,
    Preserved(String),
}

/// `(start, end_without_newline, end_with_newline)` for every logical line.
fn line_ranges(bytes: &[u8]) -> Vec<(usize, usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0;
    for (index, byte) in bytes.iter().enumerate() {
        if *byte == b'\n' {
            let end = index - usize::from(index > start && bytes[index - 1] == b'\r');
            ranges.push((start, end, index + 1));
            start = index + 1;
        }
    }
    if start < bytes.len() {
        let end = bytes.len() - usize::from(bytes.last() == Some(&b'\r'));
        ranges.push((start, end, bytes.len()));
    }
    ranges
}

fn marker_lines(bytes: &[u8], lines: &[(usize, usize, usize)], marker: &[u8]) -> Vec<usize> {
    lines
        .iter()
        .enumerate()
        .filter_map(|(index, (start, end, _))| (&bytes[*start..*end] == marker).then_some(index))
        .collect()
}

fn atomic_replace_startup_with_hook<F>(
    target: &Path,
    updated: &[u8],
    expected: &ExpectedTarget,
    before_commit: F,
) -> std::io::Result<()>
where
    F: FnOnce(),
{
    let parent = target
        .parent()
        .ok_or_else(|| std::io::Error::other("missing parent"))?;
    super::completion_install::atomic_replace_with_hook(
        parent,
        target,
        updated,
        expected.mode_or(0o600),
        expected,
        "startup",
        before_commit,
    )
}

fn bounded_output(
    program: &Path,
    args: &[&str],
    timeout_secs: u64,
) -> Option<std::process::Output> {
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return child.wait_with_output().ok(),
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}

#[cfg(test)]
#[path = "completion_startup_tests.rs"]
mod tests;
