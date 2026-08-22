//! Zero-config tab-completion self-provisioning — **bash, zsh, and Fish**.
//!
//! See ADR-045's shell-completion amendment. A proven standalone or Cargo
//! installation should not require a manual `source <(HF2Q_COMPLETE=… hf2q)`
//! line or an `--install` verb, so its release binary keeps a dynamic
//! registration script current for each supported auto-installed shell:
//! - **bash**: `…/bash-completion/completions/hf2q`, which bash-completion's lazy
//!   `_comp_load` sources on the first `hf2q<TAB>` — in the current shell.
//! - **zsh**: `_hf2q` in a site-functions dir, which `compinit` autoloads at the
//!   next shell start (zsh has no lazy first-tab loader). A release binary writes
//!   both (a) a safe, standard Homebrew `site-functions` candidate when present
//!   and (b) the per-user `~/.local/share/zsh/site-functions`. The preferred-
//!   shell startup bootstrap adds the per-user directory to `$fpath`, initializes
//!   `compinit` when needed, and registers `_hf2q` in each new interactive shell.
//! - **Fish**: `hf2q.fish` in Fish's official per-user autoload directory,
//!   `${XDG_CONFIG_HOME:-$HOME/.config}/fish/completions`. Fish discovers the
//!   command-named file without an rc edit.
//!
//! Automatic destinations require a proven standalone/Cargo release install:
//! source, debug, unmanaged, ambiguous, and root binaries must never replace a
//! live registration with a disposable or unowned binary path. Explicit
//! destinations remain available in every profile for isolated tests and local
//! development.
//!
//! This is strictly best-effort: **every** failure path is swallowed so a
//! completion-provisioning problem can never fail `hf2q <verb>`; each shell is
//! provisioned independently. The one-run-before-registration property is
//! inherent (a shell cannot invoke hf2q to create a file that does not yet
//! exist). Bash's lazy loader and Fish's autoload path can discover the new file;
//! zsh needs a new shell after that first normal hf2q process; a child cannot
//! mutate the already-running parent shell's completion state.
//!
//! Boundary: hf2q is pre-alpha and runs on the trusted bare host under the
//! operator's own account (ADR-041). Writing into the operator's own
//! `~/.local/share`, `~/.config`, a writable Homebrew/distro site-functions dir,
//! or hf2q's bounded managed block in the preferred shell's startup file is
//! in-posture. A hostile `HOME` or XDG directory owner is outside ADR-041's
//! accepted trusted-host threat model. The residual in scope is an ordinary
//! permission, race, or I/O failure, contained by ownership markers, atomic
//! replacement, and the best-effort failure boundary described above.

use std::ffi::{CString, OsString};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::ffi::OsStrExt as _;
use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use clap::Parser as _;
use clap_complete::env::{Bash, Elvish, EnvCompleter, Fish, Powershell, Shells, Zsh};

use super::complete::public_completion_command;

/// Ownership marker inserted into every hf2q-provisioned completion file.
/// An existing file that carries it at the shell's designated marker line is
/// refreshed in place. A regular file that does **not** carry it is adopted
/// losslessly (ADR-045 amendment 2026-08-21): regenerable static `hf2q
/// completions` output is replaced outright, and anything else is committed
/// byte-for-byte to an inert same-directory backup before the destination is
/// replaced — never clobbered, never nagged about on every run.
const MARKER: &str = "# hf2q-managed dynamic completion — auto-provisioned, edits are overwritten";

/// Stable, committed prefix that the first line must **start with** for a file
/// to count as hf2q-owned. Anchored at byte 0 (a leading `# ` comment) so it is
/// an ownership assertion, not merely a mention — a foreign line that happens to
/// contain the phrase later does not match. Kept a strict prefix of [`MARKER`]
/// so the trailing wording can evolve without orphaning already-written files.
pub(super) const MARKER_PREFIX: &str = "# hf2q-managed dynamic completion";

/// Generated-script binding header checked by startup blocks before sourcing.
/// The digest covers the complete generated artifact (marker, canonical writer
/// path, protected adapter, and shell-specific wrapper) with this line's digest
/// normalized to zeros. Shell startup compares the header; it does not portably
/// recompute the body digest.
pub(super) const BINDING_PREFIX: &str = "# hf2q-completion-binding sha256:";
const BINDING_PLACEHOLDER: &str =
    "# hf2q-completion-binding sha256:0000000000000000000000000000000000000000000000000000000000000000";

/// Opt-out: offline / distro-managed / read-only-HOME installs set this to skip
/// provisioning entirely.
const OPT_OUT_VAR: &str = "HF2Q_NO_COMPLETION_INSTALL";

/// The namespaced dynamic-completion trigger (shared with `main`'s `CompleteEnv`
/// hook). If it is set we are inside a candidate-generation run and must not
/// recurse into provisioning (belt-and-suspenders: the hook self-exits first).
const TRIGGER_VAR: &str = "HF2Q_COMPLETE";

/// Bash and Zsh need small option-localization repairs around the exact
/// registration emitted by the pinned clap_complete release. Every dynamic
/// shell also shares hf2q's public-surface filtering policy. These
/// adapters are shared by direct `HF2Q_COMPLETE=<shell> hf2q` output and managed
/// files so the two activation paths cannot drift.
#[derive(Copy, Clone, Debug)]
struct ProtectedBash;

#[derive(Copy, Clone, Debug)]
struct ProtectedZsh;

#[derive(Copy, Clone, Debug)]
struct ProtectedFish;

const PROTECTED_BASH: ProtectedBash = ProtectedBash;
const PROTECTED_ZSH: ProtectedZsh = ProtectedZsh;
const PROTECTED_FISH: ProtectedFish = ProtectedFish;

#[derive(Copy, Clone, Debug)]
enum CandidateSeparator {
    EnvironmentOrNewline,
    Newline,
}

/// Completion-policy decorator that structurally removes internal-only values
/// from every dynamic shell protocol. The public command projection is the
/// primary boundary; this is a fail-closed second boundary for protocol drift.
#[derive(Copy, Clone, Debug)]
struct PublicOnly<S> {
    inner: S,
    separator: CandidateSeparator,
}

impl<S> EnvCompleter for PublicOnly<S>
where
    S: EnvCompleter,
{
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn is(&self, name: &str) -> bool {
        self.inner.is(name)
    }

    fn write_registration(
        &self,
        var: &str,
        name: &str,
        bin: &str,
        completer: &str,
        buf: &mut dyn Write,
    ) -> std::io::Result<()> {
        self.inner
            .write_registration(var, name, bin, completer, buf)
    }

    fn write_complete(
        &self,
        cmd: &mut clap::Command,
        args: Vec<OsString>,
        current_dir: Option<&Path>,
        buf: &mut dyn Write,
    ) -> std::io::Result<()> {
        let separator = match self.separator {
            CandidateSeparator::EnvironmentOrNewline => std::env::var("_CLAP_IFS")
                .ok()
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "\n".to_owned()),
            CandidateSeparator::Newline => "\n".to_owned(),
        };
        let mut rendered = Vec::new();
        self.inner
            .write_complete(cmd, args, current_dir, &mut rendered)?;
        let rendered = std::str::from_utf8(&rendered).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("completion protocol was not UTF-8: {error}"),
            )
        })?;
        let filtered = rendered
            .split(&separator)
            .filter(|record| record.is_empty() || public_record(record))
            .collect::<Vec<_>>()
            .join(&separator);
        buf.write_all(filtered.as_bytes())
    }
}

fn public_record(record: &str) -> bool {
    const HIDDEN: &[&str] = &[
        "__standalone-install",
        "__fetch-hub-gguf",
        "__catalog-hub-gguf",
        "__verify-local-gguf",
        "source-teacher",
        "source-teacher-reference",
        "source-teacher-acceptance-verify",
        "--chat-parent-lifeline-fd",
    ];
    !HIDDEN.iter().any(|hidden| {
        record == *hidden
            || record
                .strip_prefix(hidden)
                .is_some_and(|suffix| suffix.starts_with('\t') || suffix.starts_with(':'))
    })
}

const PUBLIC_BASH: PublicOnly<ProtectedBash> = PublicOnly {
    inner: PROTECTED_BASH,
    separator: CandidateSeparator::EnvironmentOrNewline,
};
const PUBLIC_ELVISH: PublicOnly<Elvish> = PublicOnly {
    inner: Elvish,
    separator: CandidateSeparator::EnvironmentOrNewline,
};
const PUBLIC_FISH: PublicOnly<ProtectedFish> = PublicOnly {
    inner: PROTECTED_FISH,
    separator: CandidateSeparator::Newline,
};
const PUBLIC_POWERSHELL: PublicOnly<Powershell> = PublicOnly {
    inner: Powershell,
    separator: CandidateSeparator::Newline,
};
const PUBLIC_ZSH: PublicOnly<ProtectedZsh> = PublicOnly {
    inner: PROTECTED_ZSH,
    separator: CandidateSeparator::EnvironmentOrNewline,
};

fn completion_shells() -> Shells<'static> {
    Shells(&[
        &PUBLIC_BASH,
        &PUBLIC_ELVISH,
        &PUBLIC_FISH,
        &PUBLIC_POWERSHELL,
        &PUBLIC_ZSH,
    ])
}

/// Run the namespaced dynamic-completion bootstrap before normal command
/// initialization. This is public only for the binary entry point.
pub fn complete_env() {
    clap_complete::CompleteEnv::with_factory(public_completion_command)
        .var(TRIGGER_VAR)
        .shells(completion_shells())
        .complete();
}

impl EnvCompleter for ProtectedBash {
    fn name(&self) -> &'static str {
        Bash.name()
    }

    fn is(&self, name: &str) -> bool {
        Bash.is(name)
    }

    fn write_registration(
        &self,
        var: &str,
        name: &str,
        bin: &str,
        completer: &str,
        buf: &mut dyn Write,
    ) -> std::io::Result<()> {
        const COMPLETER_PLACEHOLDER: &str = "__HF2Q_COMPLETER_PATH_PLACEHOLDER__";
        let mut upstream = Vec::new();
        Bash.write_registration(var, name, bin, COMPLETER_PLACEHOLDER, &mut upstream)?;
        buf.write_all(&protect_bash_registration(
            &upstream,
            COMPLETER_PLACEHOLDER,
            completer,
        )?)
    }

    fn write_complete(
        &self,
        cmd: &mut clap::Command,
        args: Vec<OsString>,
        current_dir: Option<&Path>,
        buf: &mut dyn Write,
    ) -> std::io::Result<()> {
        Bash.write_complete(cmd, args, current_dir, buf)
    }
}

impl EnvCompleter for ProtectedZsh {
    fn name(&self) -> &'static str {
        Zsh.name()
    }

    fn is(&self, name: &str) -> bool {
        Zsh.is(name)
    }

    fn write_registration(
        &self,
        var: &str,
        name: &str,
        bin: &str,
        completer: &str,
        buf: &mut dyn Write,
    ) -> std::io::Result<()> {
        const COMPLETER_PLACEHOLDER: &str = "__HF2Q_COMPLETER_PATH_PLACEHOLDER__";
        let mut upstream = Vec::new();
        Zsh.write_registration(var, name, bin, COMPLETER_PLACEHOLDER, &mut upstream)?;
        buf.write_all(&protect_zsh_registration(
            &upstream,
            COMPLETER_PLACEHOLDER,
            completer,
        )?)
    }

    fn write_complete(
        &self,
        cmd: &mut clap::Command,
        args: Vec<OsString>,
        current_dir: Option<&Path>,
        buf: &mut dyn Write,
    ) -> std::io::Result<()> {
        Zsh.write_complete(cmd, args, current_dir, buf)
    }
}

impl EnvCompleter for ProtectedFish {
    fn name(&self) -> &'static str {
        Fish.name()
    }

    fn is(&self, name: &str) -> bool {
        Fish.is(name)
    }

    fn write_registration(
        &self,
        var: &str,
        name: &str,
        bin: &str,
        completer: &str,
        buf: &mut dyn Write,
    ) -> std::io::Result<()> {
        const COMPLETER_PLACEHOLDER: &str = "__HF2Q_COMPLETER_PATH_PLACEHOLDER__";
        let mut upstream = Vec::new();
        Fish.write_registration(var, name, bin, COMPLETER_PLACEHOLDER, &mut upstream)?;
        buf.write_all(&protect_fish_registration(
            &upstream,
            COMPLETER_PLACEHOLDER,
            completer,
        )?)
    }

    fn write_complete(
        &self,
        cmd: &mut clap::Command,
        args: Vec<OsString>,
        current_dir: Option<&Path>,
        buf: &mut dyn Write,
    ) -> std::io::Result<()> {
        Fish.write_complete(cmd, args, current_dir, buf)
    }
}

/// Rewrap clap's one-line Fish registration into an autoload-shaped function so
/// the completer invocation can re-verify its pinned path at Tab time (ADR-045
/// self-recovery contract; see `protect_bash_registration`). The upstream
/// argument expression is carried into the function body byte-for-byte with
/// only the completer word replaced, so the tab-time protocol cannot drift.
fn protect_fish_registration(
    upstream: &[u8],
    completer_placeholder: &str,
    completer: &str,
) -> std::io::Result<Vec<u8>> {
    let upstream = std::str::from_utf8(upstream).map_err(|error| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Fish registration UTF-8: {error}"),
        )
    })?;
    const ARGUMENTS_OPEN: &str = "--arguments \"(";
    const ARGUMENTS_CLOSE: &str = ")\"\n";
    if upstream.matches(ARGUMENTS_OPEN).count() != 1
        || !upstream.ends_with(ARGUMENTS_CLOSE)
        || upstream.matches(completer_placeholder).count() != 1
    {
        return Err(std::io::Error::other(
            "pinned clap_complete Fish registration shape changed",
        ));
    }
    let open = upstream.find(ARGUMENTS_OPEN).expect("counted above") + ARGUMENTS_OPEN.len();
    let inner = &upstream[open..upstream.len() - ARGUMENTS_CLOSE.len()];
    if !inner.contains(completer_placeholder) {
        return Err(std::io::Error::other(
            "pinned clap_complete Fish registration must invoke the completer in --arguments",
        ));
    }
    // Fish single quotes escape only `\` and `'`, each with a backslash.
    let shell_quoted_completer =
        format!("'{}'", completer.replace('\\', "\\\\").replace('\'', "\\'"));
    let invocation = inner.replace(completer_placeholder, "$_hf2q_completer");
    let registration = &upstream[..open];
    // `-f` as well as `-x`: a searchable directory recreated at the dead pin's
    // pathname must fall back, not be executed.
    let protected = format!(
        "function __hf2q_dynamic_completer\n    \
         set -l _hf2q_completer {shell_quoted_completer}\n    \
         if not test -f \"$_hf2q_completer\" -a -x \"$_hf2q_completer\"\n        \
         set _hf2q_completer (command -v hf2q)\n        \
         or return\n    \
         end\n    \
         {invocation}\n\
         end\n\n\
         {registration}__hf2q_dynamic_completer{ARGUMENTS_CLOSE}"
    );
    Ok(protected.into_bytes())
}

fn protect_bash_registration(
    upstream: &[u8],
    completer_placeholder: &str,
    completer: &str,
) -> std::io::Result<Vec<u8>> {
    let upstream = std::str::from_utf8(upstream).map_err(|error| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Bash registration UTF-8: {error}"),
        )
    })?;
    const ASSIGNMENT: &str = "    COMPREPLY=( $( \\\n";
    const STATUS: &str = "    if [[ $? != 0 ]]; then\n";
    let quoted_placeholder = format!("\"{completer_placeholder}\"");
    if upstream.matches(ASSIGNMENT).count() != 1
        || upstream.matches(STATUS).count() != 1
        || upstream.matches(&quoted_placeholder).count() != 1
    {
        return Err(std::io::Error::other(
            "pinned clap_complete Bash registration shape changed",
        ));
    }
    // The pinned completer path may legitimately die (a reclaimed worktree
    // target, a moved install tree). ADR-045: re-verify it at Tab time; fall
    // back to the `hf2q` on PATH for this invocation; with no fallback, produce
    // no candidates and no diagnostics instead of a shell error. `-f` as well
    // as `-x`: a searchable directory recreated at the dead pin's pathname
    // must fall back, not be executed. clap_complete 4.6.7 shell-quotes
    // `completer` and then places that result inside double quotes, which
    // turns the quote marks into literal filename bytes when the executable
    // path contains spaces — the command word is the preamble's re-verified
    // `$_hf2q_completer` instead. The placeholder is rewritten BEFORE the real
    // path is spliced in, so a pathological pin path containing the
    // placeholder bytes cannot be corrupted.
    let shell_quoted_completer = format!("'{}'", completer.replace('\'', "'\"'\"'"));
    // Every template rewrite that matches on upstream text must finish before
    // the real path is spliced in: the ASSIGNMENT replacement carries the
    // quoted path, so it runs LAST, after the placeholder and STATUS rewrites.
    let protected = upstream.replace(&quoted_placeholder, "\"$_hf2q_completer\"");
    let protected = protected.replace(
        STATUS,
        "    local _hf2q_complete_status=$?\n\
    if [[ -n $_hf2q_restore_glob ]]; then\n\
        set +f\n\
    fi\n\
    if [[ $_hf2q_complete_status != 0 ]]; then\n",
    );
    let protected = protected.replace(
        ASSIGNMENT,
        &format!(
            "    local _hf2q_completer={shell_quoted_completer}\n\
    if [[ ! -f $_hf2q_completer || ! -x $_hf2q_completer ]]; then\n\
        _hf2q_completer=$(type -P hf2q) && [[ -f $_hf2q_completer && -x $_hf2q_completer ]] || {{ unset COMPREPLY; return 0; }}\n\
    fi\n\
    local _hf2q_restore_glob=\n\
    if [[ $- != *f* ]]; then\n\
        set -f\n\
        _hf2q_restore_glob=1\n\
    fi\n\
    COMPREPLY=( $( \\\n"
        ),
    );
    Ok(protected.into_bytes())
}

fn protect_zsh_registration(
    upstream: &[u8],
    completer_placeholder: &str,
    completer: &str,
) -> std::io::Result<Vec<u8>> {
    let upstream = std::str::from_utf8(upstream).map_err(|error| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Zsh registration UTF-8: {error}"),
        )
    })?;
    const LOCALIZE_ARRAYS: &str = "    setopt localoptions noksharrays\n";
    if upstream.matches(completer_placeholder).count() != 1 {
        return Err(std::io::Error::other(
            "pinned clap_complete Zsh registration must invoke the completer exactly once",
        ));
    }
    // Rewrite the placeholder BEFORE the real path is spliced in, so a
    // pathological pin path containing the placeholder bytes cannot be
    // corrupted.
    let upstream = upstream.replace(completer_placeholder, "\"$_hf2q_completer\"");
    // The function name is rendered before this wrapper sees the script, so
    // locate the stable function prefix and its opening newline exactly once.
    let Some(start) = upstream.find("function _clap_dynamic_completer_") else {
        return Err(std::io::Error::other(
            "pinned clap_complete Zsh registration shape changed",
        ));
    };
    let Some(relative_open) = upstream[start..].find("() {\n") else {
        return Err(std::io::Error::other(
            "pinned clap_complete Zsh function opening changed",
        ));
    };
    let insertion = start + relative_open + "() {\n".len();
    if upstream[insertion..].contains("    emulate -L zsh\n")
        || upstream[insertion..].contains(LOCALIZE_ARRAYS)
    {
        return Err(std::io::Error::other(
            "pinned clap_complete Zsh registration unexpectedly localizes options",
        ));
    }
    // The pinned completer path may legitimately die; ADR-045's Tab-time
    // self-recovery contract (see `protect_bash_registration`). clap already
    // routes the completer's stderr to /dev/null for zsh, so without this a
    // dead pin fails silently on every Tab and never recovers. `whence -p`
    // forces a fresh PATH search — `$commands` is zsh's hash table and can be
    // stale in exactly the dead-binary scenario this recovers from. `-f` as
    // well as `-x`: a directory at the pathname must fall back, not execute.
    let shell_quoted_completer = format!("'{}'", completer.replace('\'', "'\"'\"'"));
    let fallback = format!(
        "    local _hf2q_completer={shell_quoted_completer}\n    \
         if [[ ! -f $_hf2q_completer || ! -x $_hf2q_completer ]]; then\n        \
         _hf2q_completer=$(whence -p hf2q) || return 0\n        \
         [[ -f $_hf2q_completer && -x $_hf2q_completer ]] || return 0\n    \
         fi\n"
    );
    let mut protected =
        String::with_capacity(upstream.len() + LOCALIZE_ARRAYS.len() + fallback.len());
    protected.push_str(&upstream[..insertion]);
    // `emulate -L zsh` is too broad here: it clears completion-framework
    // options inherited from `_main_complete`, which makes `_describe` reject
    // otherwise valid matches during a literal Tab. Only KSH_ARRAYS needs
    // normalization. LOCAL_OPTIONS makes that one change function-local and
    // restores the operator's prior option state on return.
    protected.push_str(LOCALIZE_ARRAYS);
    protected.push_str(&fallback);
    protected.push_str(&upstream[insertion..]);
    Ok(protected.into_bytes())
}

/// Explicit override for the zsh completions drop dir. Point it at a dir
/// already on your `$fpath` — e.g. Homebrew's
/// `/opt/homebrew/share/zsh/site-functions` — to make `_hf2q` directly
/// discoverable by `compinit`. Unlike automatic discovery, this explicit
/// override is honored by unproven binaries too, which lets tests, package
/// maintainers, and developers use an isolated directory safely.
const ZSH_DIR_VAR: &str = "HF2Q_ZSH_COMPLETIONS_DIR";

/// Explicit override for the Fish completions drop dir. Fish autoloads
/// `hf2q.fish` from this directory. The override is honored in every build
/// profile; automatic `${XDG_CONFIG_HOME:-$HOME/.config}/fish/completions`
/// discovery is installation-owned release-only.
const FISH_DIR_VAR: &str = "HF2Q_FISH_COMPLETIONS_DIR";

/// One shell's provisioning shape. The files differ only in the drop-file name,
/// completer key, and where the ownership marker sits (bash/Fish: line 0; zsh:
/// line 1, because `#compdef hf2q` MUST be the first line for `compinit`
/// autoload). Everything else — marker gate, atomic write, symlink preservation
/// — is shared.
struct Shell {
    /// `HF2Q_COMPLETE` value + `clap_complete` completer key.
    name: &'static str,
    /// Provisioned file name: bash `hf2q`, zsh `_hf2q`, or Fish `hf2q.fish`.
    file: &'static str,
    /// 0-based line index the ownership marker occupies.
    marker_line: usize,
    /// Explicit destination variable honored by every build profile.
    explicit_dir_var: &'static str,
}

const BASH: Shell = Shell {
    name: "bash",
    file: "hf2q",
    marker_line: 0,
    explicit_dir_var: "BASH_COMPLETION_USER_DIR",
};
const ZSH: Shell = Shell {
    name: "zsh",
    file: "_hf2q",
    marker_line: 1,
    explicit_dir_var: ZSH_DIR_VAR,
};
const FISH: Shell = Shell {
    name: "fish",
    file: "hf2q.fish",
    marker_line: 0,
    explicit_dir_var: FISH_DIR_VAR,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct FileIdentity {
    dev: u64,
    ino: u64,
    size: u64,
    mtime: i64,
    mtime_nsec: i64,
    ctime: i64,
    ctime_nsec: i64,
}

impl FileIdentity {
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        Self {
            dev: metadata.dev(),
            ino: metadata.ino(),
            size: metadata.size(),
            mtime: metadata.mtime(),
            mtime_nsec: metadata.mtime_nsec(),
            ctime: metadata.ctime(),
            ctime_nsec: metadata.ctime_nsec(),
        }
    }
}

/// The target state observed before rendering a replacement. Absent creation
/// is committed with the OS's no-replace primitive; an existing managed file
/// is revalidated by type, identity, metadata, and bytes immediately before
/// rename. Unix does not expose an atomic compare-and-replace-by-inode syscall,
/// so an external writer in the final revalidation-to-rename interval remains
/// an honest trusted-host residual.
#[derive(Clone, Debug)]
pub(super) enum ExpectedTarget {
    Absent,
    Regular {
        identity: FileIdentity,
        bytes: Vec<u8>,
        mode: u32,
    },
}

pub(super) fn capture_regular_target(path: &Path) -> std::io::Result<ExpectedTarget> {
    let before = fs::symlink_metadata(path)?;
    if !before.file_type().is_file() {
        return Err(std::io::Error::other("target is not a regular file"));
    }
    let identity = FileIdentity::from_metadata(&before);
    let bytes = fs::read(path)?;
    let after = fs::symlink_metadata(path)?;
    if !after.file_type().is_file() || FileIdentity::from_metadata(&after) != identity {
        return Err(std::io::Error::other(
            "target changed while it was being inspected",
        ));
    }
    Ok(ExpectedTarget::Regular {
        identity,
        bytes,
        mode: before.permissions().mode() & 0o7777,
    })
}

impl ExpectedTarget {
    pub(super) fn bytes(&self) -> &[u8] {
        match self {
            Self::Absent => &[],
            Self::Regular { bytes, .. } => bytes,
        }
    }

    pub(super) fn mode_or(&self, default: u32) -> u32 {
        match self {
            Self::Absent => default,
            Self::Regular { mode, .. } => *mode,
        }
    }

    pub(super) fn revalidate(&self, target: &Path) -> std::io::Result<()> {
        match self {
            Self::Absent => match fs::symlink_metadata(target) {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Ok(_) => Err(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    "target appeared during reconciliation",
                )),
                Err(error) => Err(error),
            },
            Self::Regular {
                identity, bytes, ..
            } => {
                let before = fs::symlink_metadata(target)?;
                if !before.file_type().is_file()
                    || FileIdentity::from_metadata(&before) != *identity
                {
                    return Err(std::io::Error::other(
                        "target identity changed during reconciliation",
                    ));
                }
                if fs::read(target)? != *bytes {
                    return Err(std::io::Error::other(
                        "target content changed during reconciliation",
                    ));
                }
                let after = fs::symlink_metadata(target)?;
                if !after.file_type().is_file() || FileIdentity::from_metadata(&after) != *identity
                {
                    return Err(std::io::Error::other(
                        "target changed during final reconciliation check",
                    ));
                }
                Ok(())
            }
        }
    }
}

/// `Some(v)` unless `v` is empty — used to treat an empty env var as unset.
fn non_empty(v: std::ffi::OsString) -> Option<std::ffi::OsString> {
    if v.is_empty() {
        None
    } else {
        Some(v)
    }
}

/// What reconciliation did. Logging and the operator reporter do not exist when
/// `reconcile()` runs in `main`, so the outcome is stashed and emitted by
/// [`report_outcome`] after CLI initialization.
#[derive(Debug)]
enum Outcome {
    /// File written (created or refreshed) at this path.
    Wrote(PathBuf),
    /// Existing file already byte-identical; nothing to do.
    UpToDate(PathBuf),
    /// An unmarked regular file occupied the managed destination and was
    /// adopted (ADR-045 amendment 2026-08-21). `backup` names the inert
    /// same-directory copy of the prior content; `None` means the occupant was
    /// byte-identical to this binary's static `hf2q completions <shell>` output
    /// — regenerable hf2q debris preserved by regenerability, not by copy.
    Adopted {
        path: PathBuf,
        backup: Option<PathBuf>,
    },
    /// An unmarked regular file could not be adopted because its lossless
    /// backup could not be committed; the destination was left untouched.
    /// Losslessness precedes progress.
    PreservedForeign { path: PathBuf, backup_error: String },
    /// A symlink that resolves to a regular file — an explicit operator
    /// arrangement. Preserved untouched and not an incomplete install.
    PreservedOperatorLink(PathBuf),
    /// A symlink whose referent is missing. Preserved — the referent may only
    /// be temporarily absent — but completion is factually broken, so this
    /// stays an actionable warning.
    PreservedDanglingLink(PathBuf),
    /// The target exists but is neither a regular file nor a symlink handled
    /// above (a directory, a device, …). Never replaced — we neither read
    /// through it nor `rename` over it.
    PreservedNonRegular(PathBuf),
    /// Opt-out env var set; provisioning skipped.
    OptedOut,
    /// Inside a completion run; provisioning skipped.
    CompletionRun,
    /// Uninstall or rollback owns completion cleanup for this invocation.
    LifecycleCleanup,
    /// This binary is not authorized for implicit destinations. The value
    /// names the explicit destination variable that enables an isolated write.
    PolicySkip(&'static str),
    /// Best-effort failure (dir/HOME resolution, render, or write).
    Failed(String),
    /// Preferred-shell startup bootstrap result.
    Startup(super::completion_startup::Outcome),
}

static LAST_OUTCOME: OnceLock<Vec<(&'static str, Outcome)>> = OnceLock::new();

/// Best-effort: ensure the bash, zsh, and Fish dynamic-completion files exist
/// for this binary. Linux and macOS are both first-class; release builds
/// discover standard destinations, while every unproven binary requires
/// explicit destination environment variables.
///
/// Called once, early in `main`, before argv parsing — so it fires for every
/// real invocation (including `--help`/`--version`/parse-error exits) except a
/// completion run. Never panics, never returns an error to the caller; each
/// shell's provisioning succeeds or is swallowed independently.
pub fn reconcile(raw_args: &[OsString]) {
    let outcomes: Vec<(&'static str, Outcome)> = if completion_trigger_active() {
        vec![("completion", Outcome::CompletionRun)]
    } else if lifecycle_cleanup_requested(raw_args) {
        vec![("completion", Outcome::LifecycleCleanup)]
    } else if std::env::var_os(OPT_OUT_VAR).is_some() {
        vec![("completion", Outcome::OptedOut)]
    } else {
        let allow_automatic = automatic_destinations_enabled();
        // Proven installed release builds discover managed destinations; all
        // other binaries resolve only explicitly configured destinations.
        // Reconcile every resolved directory independently.
        let mut v = Vec::new();
        for shell in [&BASH, &ZSH, &FISH] {
            let dirs = completions_dirs(shell, allow_automatic);
            if dirs.is_empty() {
                let outcome = if !allow_automatic {
                    Outcome::PolicySkip(shell.explicit_dir_var)
                } else {
                    Outcome::Failed(format!(
                        "cannot resolve a {} completions dir (HOME/XDG base unset)",
                        shell.name
                    ))
                };
                v.push((shell.name, outcome));
                continue;
            }
            for dir in dirs {
                let o = try_reconcile_in(shell, &dir).unwrap_or_else(Outcome::Failed);
                v.push((shell.name, o));
            }
        }
        // A startup block may only point at an artifact whose ownership gate
        // succeeds *after* reconciliation. In particular, do not turn a
        // preserved foreign file, operator symlink, or directory into shell
        // startup code merely because it occupies the managed pathname. Keep
        // this entire discovery pass installation-owned so source/debug and
        // unmanaged binaries do not even probe implicit live locations.
        let bash_registration = startup_bash_registration();
        let zsh_functions_dir = preferred_zsh_startup_registration();
        for (shell, outcome) in super::completion_startup::reconcile_preferred_shell(
            bash_registration
                .as_ref()
                .map(|(path, binding)| (path.as_path(), binding.as_str())),
            zsh_functions_dir
                .as_ref()
                .map(|(path, binding)| (path.as_path(), binding.as_str())),
            allow_automatic,
        ) {
            v.push((shell, Outcome::Startup(outcome)));
        }
        let registrations = v
            .iter()
            .filter_map(|(_, outcome)| match outcome {
                Outcome::Wrote(path) | Outcome::UpToDate(path) => Some(path.clone()),
                Outcome::Adopted { path, .. } => Some(path.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let startup_files = v
            .iter()
            .filter_map(|(_, outcome)| match outcome {
                Outcome::Startup(
                    super::completion_startup::Outcome::Wrote(path)
                    | super::completion_startup::Outcome::UpToDate(path),
                ) => Some(path.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        if let Err(error) = super::completion_receipt::record(&registrations, &startup_files) {
            v.push(("ownership receipt", Outcome::Failed(error)));
        }
        v
    };
    // If somehow called twice, keep the first result; the work is idempotent.
    let _ = LAST_OUTCOME.set(outcomes);
}

/// Surface the one fact a child process cannot make automatic: an already
/// running parent shell cannot observe a newly installed registration. Normal
/// up-to-date and policy-skip outcomes stay silent; incomplete ownership or
/// filesystem states remain actionable instead of failing the requested CLI
/// command.
pub fn report_outcome() {
    let Some(outcomes) = LAST_OUTCOME.get() else {
        return;
    };
    let mut updated = Vec::new();
    let mut backups = Vec::new();
    let mut problems = Vec::new();
    for (shell, outcome) in outcomes {
        let family = shell.split_whitespace().next().unwrap_or(shell);
        if matches!(outcome, Outcome::Wrote(_) | Outcome::Adopted { .. })
            || matches!(
                outcome,
                Outcome::Startup(super::completion_startup::Outcome::Wrote(_))
            )
        {
            if !updated.contains(&family) {
                updated.push(family);
            }
        }
        match outcome {
            Outcome::Wrote(path) => {
                tracing::debug!(shell, path = %path.display(), "provisioned completion");
            }
            Outcome::UpToDate(path) => {
                tracing::debug!(shell, path = %path.display(), "completion already current");
            }
            Outcome::Adopted { path, backup } => {
                tracing::debug!(shell, path = %path.display(), ?backup, "adopted completion destination");
                if let Some(path) = backup {
                    backups.push(path.display().to_string());
                }
            }
            Outcome::PreservedForeign { path, backup_error } => problems.push(format!(
                "{shell} destination {} was preserved because its backup failed: {backup_error}",
                path.display()
            )),
            Outcome::PreservedOperatorLink(path) => {
                tracing::debug!(shell, path = %path.display(), "preserved operator completion symlink");
            }
            Outcome::PreservedDanglingLink(path) => problems.push(format!(
                "{shell} destination {} is a dangling symlink",
                path.display()
            )),
            Outcome::PreservedNonRegular(path) => problems.push(format!(
                "{shell} destination {} is not a regular file",
                path.display()
            )),
            Outcome::OptedOut | Outcome::CompletionRun | Outcome::LifecycleCleanup => {}
            Outcome::PolicySkip(variable) => {
                tracing::debug!(
                    shell,
                    explicit_destination = *variable,
                    "automatic completion provisioning is not authorized for this binary"
                );
            }
            Outcome::Failed(reason) => problems.push(format!("{shell}: {reason}")),
            Outcome::Startup(startup) => match startup {
                super::completion_startup::Outcome::Wrote(path) => {
                    tracing::debug!(shell, path = %path.display(), "provisioned completion startup block");
                }
                super::completion_startup::Outcome::UpToDate(path) => {
                    tracing::debug!(shell, path = %path.display(), "completion startup block already current");
                }
                super::completion_startup::Outcome::PreservedMalformed(path) => problems.push(
                    format!("{shell} markers in {} are ambiguous", path.display()),
                ),
                super::completion_startup::Outcome::PreservedNonRegular(path) => {
                    problems.push(format!(
                        "{shell} destination {} is not a regular file",
                        path.display()
                    ))
                }
                super::completion_startup::Outcome::Failed(reason) => {
                    problems.push(format!("{shell}: {reason}"));
                }
            },
        }
    }
    updated.sort_unstable();
    backups.sort();
    problems.sort();
    if !updated.is_empty() {
        eprintln!(
            "hf2q: installed Tab completion for {}; open a new shell to activate it",
            updated.join(", ")
        );
        for backup in backups {
            eprintln!("hf2q: preserved the previous completion file at {backup}");
        }
    }
    for problem in problems {
        eprintln!("hf2q: completion setup incomplete: {problem}");
    }
}

fn lifecycle_cleanup_requested(raw_args: &[OsString]) -> bool {
    matches!(
        super::Cli::try_parse_from(raw_args.iter().cloned()),
        Ok(super::Cli {
            command: super::Command::Uninstall(_),
            ..
        }) | Ok(super::Cli {
            command: super::Command::Update(super::UpdateArgs { rollback: true, .. }),
            ..
        })
    )
}

/// Automatic mutation is reserved for a release binary whose owner can be
/// proven by the same fail-closed installation resolver used by update and
/// uninstall. Source builds, unmanaged copies, ambiguous receipts, and root
/// invocations remain inert. Explicit completion destinations are independent
/// of this gate so tests and package maintainers can use isolated paths.
fn automatic_destinations_enabled() -> bool {
    if cfg!(debug_assertions) || rustix::process::geteuid().is_root() {
        return false;
    }
    let Ok(executable) = std::env::current_exe() else {
        return false;
    };
    matches!(
        crate::distribution::installation::detect(&executable),
        Ok(crate::distribution::installation::Installation::Standalone { .. })
            | Ok(crate::distribution::installation::Installation::Cargo { .. })
    )
}

/// Mirror clap_complete's activation rule exactly. An exported empty value or
/// `0` is a normal invocation; only a nonempty, nonzero value is a completion
/// request. `HF2Q_NO_COMPLETION_INSTALL` remains the sole install opt-out.
fn completion_trigger_active() -> bool {
    std::env::var_os(TRIGGER_VAR).is_some_and(|value| !value.is_empty() && value != "0")
}

/// The fallible core, for one shell and one already-resolved drop `dir`. Returns
/// a human-readable reason string on failure so the caller can stash it for the
/// debug diagnostic. A failure on one dir never aborts the others — each is
/// reconciled independently by [`reconcile`].
fn try_reconcile_in(shell: &Shell, dir: &Path) -> Result<Outcome, String> {
    let target = dir.join(shell.file);
    let desired = render_registration(shell).map_err(|e| format!("render failed: {e}"))?;

    // Probe the target WITHOUT following symlinks (`symlink_metadata` = lstat).
    // A symlink is never read through or `rename`d over — replacing it would
    // silently destroy the operator's link. A link that resolves to a regular
    // file is a working operator arrangement (quiet); a dangling link is
    // preserved but stays an actionable warning. Only a genuine regular file
    // is a candidate for rewrite or adoption.
    let mut adoption_backup: Option<Option<PathBuf>> = None;
    let expected = match std::fs::symlink_metadata(&target) {
        Ok(meta) if meta.file_type().is_symlink() => {
            return Ok(match std::fs::metadata(&target) {
                Ok(referent) if referent.file_type().is_file() => {
                    Outcome::PreservedOperatorLink(target)
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    Outcome::PreservedDanglingLink(target)
                }
                _ => Outcome::PreservedNonRegular(target),
            });
        }
        Ok(meta) if !meta.file_type().is_file() => {
            return Ok(Outcome::PreservedNonRegular(target));
        }
        Ok(_) => {
            // Regular file: take a stable identity+content snapshot before
            // applying the ownership gate. The same snapshot is revalidated
            // immediately before replacement, so whatever the replacement
            // destroys is provably the content that was inspected (and, on
            // the adoption path, backed up).
            let expected = capture_regular_target(&target)
                .map_err(|e| format!("reading {}: {e}", target.display()))?;
            let existing = expected.bytes();
            if existing == desired {
                return Ok(Outcome::UpToDate(target));
            }
            if !is_hf2q_managed(existing, shell.marker_line) {
                // Unmarked occupant of hf2q's own destination pathname: adopt
                // it losslessly instead of refusing forever (ADR-045
                // amendment 2026-08-21). Bytes identical to this binary's
                // static `hf2q completions <shell>` output are regenerable hf2q
                // debris — replaced outright. Anything else is committed
                // byte-for-byte (mode included) to an inert same-directory
                // backup BEFORE the destination is touched; a failed backup
                // aborts adoption and preserves the old refusal outcome.
                let backup = if is_current_static_output(shell, existing) {
                    None
                } else {
                    match commit_adoption_backup(dir, shell.file, &expected) {
                        Ok(slot) => Some(slot),
                        Err(e) => {
                            return Ok(Outcome::PreservedForeign {
                                path: target,
                                backup_error: e.to_string(),
                            });
                        }
                    }
                };
                adoption_backup = Some(backup);
            }
            // hf2q-owned but stale (e.g. binary moved), or adopted — refresh
            // below against the captured snapshot.
            expected
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => ExpectedTarget::Absent,
        Err(e) => return Err(format!("stat {}: {e}", target.display())),
    };

    std::fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
    // An adopted file becomes a fresh managed artifact and gets the
    // fresh-install mode; the operator's original mode lives on the backup.
    // A managed refresh keeps the managed file's existing mode.
    let mode = if adoption_backup.is_some() {
        0o644
    } else {
        expected.mode_or(0o644)
    };
    atomic_replace_with_hook(dir, &target, &desired, mode, &expected, "completion", || {})
        .map_err(|e| format!("writing {}: {e}", target.display()))?;
    Ok(match adoption_backup {
        Some(backup) => Outcome::Adopted {
            path: target,
            backup,
        },
        None => Outcome::Wrote(target),
    })
}

/// True when `existing` is byte-identical to the static completion script this
/// binary's `hf2q completions <shell>` prints today — content with zero
/// information beyond what hf2q regenerates on demand, so adoption may discard
/// it without a backup copy.
fn is_current_static_output(shell: &Shell, existing: &[u8]) -> bool {
    let static_shell = match shell.name {
        "bash" => clap_complete::Shell::Bash,
        "zsh" => clap_complete::Shell::Zsh,
        "fish" => clap_complete::Shell::Fish,
        _ => return false,
    };
    {
        let mut command = public_completion_command();
        let mut generated = Vec::new();
        clap_complete::generate(static_shell, &mut command, "hf2q", &mut generated);
        generated == existing
    }
}

/// The inert, content-addressed backup slot for an adopted foreign occupant of
/// `file_name`. Dot-prefixed so no shell loader can discover it:
/// bash-completion looks files up by command name, `compinit` autoloads only
/// `_*` names, and Fish sources only `*.fish` — and none of the three globs
/// dotfiles. The digest prefix makes re-adoption of identical content
/// idempotent and distinct contents collision-free in practice.
fn adoption_backup_slot(dir: &Path, file_name: &str, bytes: &[u8]) -> PathBuf {
    use sha2::{Digest as _, Sha256};
    let digest = format!("{:x}", Sha256::digest(bytes));
    dir.join(format!(".hf2q-backup.{file_name}.{}", &digest[..12]))
}

/// Commit the captured foreign content to its backup slot before the
/// destination is replaced. Fully written, synchronized, and atomically
/// committed with the same machinery as the managed file itself; the foreign
/// file's mode is preserved on the copy. An existing identical slot is reused
/// (idempotent re-adoption); a torn regular-file slot in our own backup
/// namespace is atomically rewritten; any other occupant of the slot is an
/// error — adoption then aborts and the destination stays untouched.
fn commit_adoption_backup(
    dir: &Path,
    file_name: &str,
    expected: &ExpectedTarget,
) -> std::io::Result<PathBuf> {
    let ExpectedTarget::Regular { bytes, mode, .. } = expected else {
        return Err(std::io::Error::other(
            "adoption backup requires a captured regular file",
        ));
    };
    let slot = adoption_backup_slot(dir, file_name, bytes);
    let slot_state = match std::fs::symlink_metadata(&slot) {
        Ok(meta) if meta.file_type().is_file() => {
            let captured = capture_regular_target(&slot)?;
            if captured.bytes() == bytes {
                return Ok(slot);
            }
            captured
        }
        Ok(_) => {
            return Err(std::io::Error::other(format!(
                "backup slot {} is occupied by a non-regular file",
                slot.display()
            )));
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => ExpectedTarget::Absent,
        Err(e) => return Err(e),
    };
    atomic_replace_with_hook(dir, &slot, bytes, *mode, &slot_state, "backup", || {})?;
    Ok(slot)
}

/// The completions drop dir(s) for `shell`, in loader priority order. Bash and
/// Fish have exactly one per-user directory; zsh has one or more (see
/// [`zsh_target_dirs`]). Empty means either no release destination was
/// resolvable or an unproven binary intentionally received no explicit target;
/// [`reconcile`] records those cases distinctly.
fn completions_dirs(shell: &Shell, allow_automatic: bool) -> Vec<PathBuf> {
    match shell.name {
        "bash" => bash_target_dirs_with_automatic_user(allow_automatic),
        "zsh" => zsh_target_dirs_with_automatic_locations(allow_automatic),
        "fish" => fish_target_dirs_with_automatic_user(allow_automatic),
        _ => Vec::new(),
    }
}

/// Debug/test binaries may write bash completion only when the caller exports
/// the completion-specific destination. This prevents a local debug run from
/// replacing a release registration in the operator's normal XDG/HOME tree.
fn bash_target_dirs_with_automatic_user(allow_automatic_user: bool) -> Vec<PathBuf> {
    let explicit = std::env::var_os("BASH_COMPLETION_USER_DIR")
        .and_then(non_empty)
        .is_some();
    if !explicit && !allow_automatic_user {
        return Vec::new();
    }
    user_completions_dir().into_iter().collect()
}

/// Resolve the bash-completion user completions directory, mirroring
/// bash-completion's own lookup:
/// `${BASH_COMPLETION_USER_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/bash-completion}/completions`.
///
/// Only hf2q's own (exported) environment is visible here; a shell-local
/// `BASH_COMPLETION_USER_DIR` not exported to this child is invisible — the
/// guarantee is scoped to the default or exported environment.
fn user_completions_dir() -> Option<PathBuf> {
    let base = if let Some(d) = std::env::var_os("BASH_COMPLETION_USER_DIR").and_then(non_empty) {
        PathBuf::from(d)
    } else {
        let data_home = if let Some(x) = std::env::var_os("XDG_DATA_HOME").and_then(non_empty) {
            PathBuf::from(x)
        } else {
            let home = std::env::var_os("HOME").and_then(non_empty)?;
            PathBuf::from(home).join(".local/share")
        };
        data_home.join("bash-completion")
    };
    Some(base.join("completions"))
}

/// Resolve the zsh completion drop dir(s). zsh has
/// no default per-user completion dir, so:
/// - `$HF2Q_ZSH_COMPLETIONS_DIR` if exported ⇒ **exactly that dir** — the
///   operator's explicit choice (point it at any `$fpath` dir);
/// - otherwise, for release builds, **the additive set**
///   `[safe Homebrew dir?, XDG dir?]`:
///   1. a Homebrew `site-functions` dir that already exists and is
///      safe to write (owned by our euid, owner-writable, not group/world-
///      writable) — the macOS zero-config candidate (see
///      [`safe_on_fpath_zsh_dir`]);
///   2. `${XDG_DATA_HOME:-$HOME/.local/share}/zsh/site-functions` —
///      the HOME-isolated per-user dir. The preferred-shell startup bootstrap
///      adds it to `$fpath` and initializes `compinit` when needed.
///
/// Writing **both** in an owned release install is deliberate: hf2q never
/// *abandons* the XDG dir, so an operator who already put it on their `$fpath`
/// keeps receiving updates there
/// (no regression), while the Homebrew dir adds zero-config for everyone else.
/// A stray `_hf2q` in a dir that turns out not to be on `$fpath` is inert, so the
/// additive write is never worse than the XDG-only status quo — and hf2q does not
/// (and cannot cheaply) *prove* `$fpath` membership without spawning the
/// operator's shell, which it declines to do on every startup. Unproven
/// binaries intentionally skip **all** automatic destinations.
/// Policy-parametric core used to pin owned and unowned behavior in tests.
fn zsh_target_dirs_with_automatic_locations(allow_automatic_locations: bool) -> Vec<PathBuf> {
    if let Some(d) = std::env::var_os(ZSH_DIR_VAR).and_then(non_empty) {
        return vec![PathBuf::from(d)];
    }
    if !allow_automatic_locations {
        return Vec::new();
    }
    let mut dirs = Vec::new();
    if let Some(hb) = safe_on_fpath_zsh_dir(homebrew_site_functions_candidates()) {
        dirs.push(hb);
    }
    if let Some(xdg) = xdg_zsh_site_functions() {
        dirs.push(xdg);
    }
    dirs
}

/// The HOME-isolated per-user zsh fallback dir:
/// `${XDG_DATA_HOME:-$HOME/.local/share}/zsh/site-functions`. Not on zsh's
/// default `$fpath`; the managed startup block adds it. `None` only if both
/// `XDG_DATA_HOME` and `HOME` are unset/empty.
fn xdg_zsh_site_functions() -> Option<PathBuf> {
    let data_home = if let Some(x) = std::env::var_os("XDG_DATA_HOME").and_then(non_empty) {
        PathBuf::from(x)
    } else {
        PathBuf::from(std::env::var_os("HOME").and_then(non_empty)?).join(".local/share")
    };
    Some(data_home.join("zsh/site-functions"))
}

/// Select the exact-current Zsh artifact the startup block should expose. A
/// marker alone is insufficient: a failed refresh must not make an old script
/// outrank a successfully refreshed fallback. If no candidate is current,
/// there is no safe startup registration to reconcile.
fn preferred_zsh_startup_registration() -> Option<(PathBuf, String)> {
    let desired = render_registration(&ZSH).ok()?;
    let binding = registration_binding_line(&desired, &ZSH)?.to_owned();
    if let Some(explicit) = std::env::var_os(ZSH_DIR_VAR).and_then(non_empty) {
        let dir = PathBuf::from(explicit);
        return is_exact_regular_registration(&dir.join(ZSH.file), &desired)
            .then_some((dir, binding));
    }

    let xdg = xdg_zsh_site_functions();
    let homebrew = safe_on_fpath_zsh_dir(homebrew_site_functions_candidates());
    for dir in [xdg.as_ref(), homebrew.as_ref()].into_iter().flatten() {
        if is_exact_regular_registration(&dir.join(ZSH.file), &desired) {
            return Some((dir.clone(), binding));
        }
    }
    None
}

fn startup_bash_registration() -> Option<(PathBuf, String)> {
    let desired = render_registration(&BASH).ok()?;
    let binding = registration_binding_line(&desired, &BASH)?.to_owned();
    let path = user_completions_dir()?.join(BASH.file);
    is_exact_regular_registration(&path, &desired).then_some((path, binding))
}

/// Resolve Fish's official per-user autoload directory. An explicit
/// `$HF2Q_FISH_COMPLETIONS_DIR` wins in every profile; otherwise owned release
/// installs use `${XDG_CONFIG_HOME:-$HOME/.config}/fish/completions`, while
/// unproven binaries decline to discover an implicit live destination.
/// Policy-parametric core used to pin owned and unowned behavior in tests.
fn fish_target_dirs_with_automatic_user(allow_automatic_user: bool) -> Vec<PathBuf> {
    if let Some(dir) = std::env::var_os(FISH_DIR_VAR).and_then(non_empty) {
        return vec![PathBuf::from(dir)];
    }
    if !allow_automatic_user {
        return Vec::new();
    }
    fish_user_completions_dir().into_iter().collect()
}

/// Fish's official per-user command-completion directory:
/// `${XDG_CONFIG_HOME:-$HOME/.config}/fish/completions`. `None` only when both
/// `XDG_CONFIG_HOME` and `HOME` are unset or empty.
fn fish_user_completions_dir() -> Option<PathBuf> {
    let config_home = if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME").and_then(non_empty) {
        PathBuf::from(xdg)
    } else {
        PathBuf::from(std::env::var_os("HOME").and_then(non_empty)?).join(".config")
    };
    Some(config_home.join("fish/completions"))
}

/// Candidate Homebrew `site-functions` directories, most-specific first. Each is
/// placed on `$fpath` by `brew shellenv` on a standard install, so it is a
/// discoverable drop target for zero-config completion. `$HOMEBREW_PREFIX`
/// (exported by `brew shellenv`, so inherited by an hf2q launched from the
/// operator's shell) wins; the two canonical prefixes — `/opt/homebrew` (Apple
/// Silicon) and `/usr/local` (Intel) — are the fallback probe set.
fn homebrew_site_functions_candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(p) = std::env::var_os("HOMEBREW_PREFIX").and_then(non_empty) {
        v.push(PathBuf::from(p).join("share/zsh/site-functions"));
    }
    v.push(PathBuf::from("/opt/homebrew/share/zsh/site-functions"));
    v.push(PathBuf::from("/usr/local/share/zsh/site-functions"));
    v
}

/// The first candidate dir that already exists as a **real directory owned by our
/// effective uid, owner-writable, and not group- or world-writable** — the safety
/// gate that lets hf2q auto-write into an on-`$fpath` Homebrew dir without a
/// cross-user clobber hazard. Rationale for each rejection:
/// - **running as root** (`euid == 0`, e.g. under `sudo`): a root-owned
///   `/usr/local/...` would otherwise pass the ownership gate and hf2q would write
///   a *machine-wide* completion — never auto-install system files; root uses its
///   own HOME's XDG dir like any account;
/// - **not a real dir** (missing, a file, or a symlink): `symlink_metadata` +
///   `is_dir()` rejects symlinks too, so we never write *through* an operator- or
///   attacker-managed link;
/// - **owned by another uid** (e.g. root-owned `/usr/local/...`): not ours to
///   manage — fall through to the per-user XDG dir instead;
/// - **group/world-writable**: a slot another account can write is exactly the
///   cross-install/cross-user hazard the original design avoided;
/// - **not owner-writable/searchable** (e.g. a `0555` dir we own): the later
///   write would fail, so reject it up front and let the always-written XDG dir
///   carry completion instead.
///
/// Mode bits are the gate, not a full access oracle — ACLs, read-only mounts, and
/// file flags can still deny a write the bits permit. TOCTOU between this lstat
/// and the later write is not hardened; it remains an explicit open obligation,
/// not an ADR-041 containment claim or accepted deferral. The always-written XDG
/// dir is only a functional backstop when the Homebrew write fails. Returns
/// `None` if nothing qualifies.
fn safe_on_fpath_zsh_dir(candidates: Vec<PathBuf>) -> Option<PathBuf> {
    use std::os::unix::fs::MetadataExt;
    // SAFETY: `geteuid` always succeeds, is reentrant, and touches no memory.
    let euid = unsafe { libc::geteuid() };
    if euid == 0 {
        return None; // never auto-install a system-wide completion (sudo/root)
    }
    for dir in candidates {
        let Ok(meta) = std::fs::symlink_metadata(&dir) else {
            continue; // missing or unstattable
        };
        if !meta.file_type().is_dir() {
            continue; // a file or symlink — never write through it
        }
        if meta.uid() != euid {
            continue; // someone else's dir (e.g. root) — not ours to manage
        }
        let mode = meta.mode();
        if mode & 0o022 != 0 {
            continue; // group- or world-writable — cross-user clobber hazard
        }
        if mode & 0o300 != 0o300 {
            continue; // owner lacks write+search — the write would fail; use XDG
        }
        return Some(dir);
    }
    None
}

/// Render the dynamic-completion registration script for THIS binary + `shell`,
/// carrying the ownership marker. Byte-identical to `HF2Q_COMPLETE=<shell> hf2q`
/// except for the marker line, with the in-function invocation pinning the
/// canonicalized `current_exe()` — completion calls the exact binary that
/// wrote the file while that binary exists, and self-recovers through the
/// `hf2q` on PATH when the pin dies (ADR-045 Tab-time self-recovery).
///
/// The marker sits at `shell.marker_line`: **line 0** for bash and Fish (a
/// leading `#` comment), but **line 1** for zsh — its first line MUST be
/// `#compdef hf2q` for `compinit` to autoload the `_hf2q` file, so the marker
/// comment is inserted immediately after it (still a comment, invisible to
/// zsh).
fn render_registration(shell: &Shell) -> std::io::Result<Vec<u8>> {
    let exe = std::env::current_exe()?;
    // Canonicalize to bind completion to the symlink target (a stable real
    // path) rather than a bin-dir symlink that may be re-pointed.
    let exe = std::fs::canonicalize(&exe).unwrap_or(exe);
    let completer = exe.to_str().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "executable path is not valid UTF-8",
        )
    })?;

    let shells = completion_shells();
    let completer_shell = shells.completer(shell.name).ok_or_else(|| {
        std::io::Error::other(format!(
            "clap_complete has no builtin {} completer",
            shell.name
        ))
    })?;

    // Render the bare clap registration first. (var, name, bin, completer):
    // `HF2Q_COMPLETE` is the trigger, `hf2q` names the function + `complete`/
    // `compdef` target, `completer` is the absolute binary the function invokes.
    let mut clap = Vec::new();
    completer_shell.write_registration("HF2Q_COMPLETE", "hf2q", "hf2q", completer, &mut clap)?;

    // Splice the ownership marker and generation-binding header at the
    // shell-specific marker line.
    let mut buf = Vec::new();
    if shell.marker_line == 0 {
        writeln!(buf, "{MARKER}")?;
        writeln!(buf, "{BINDING_PLACEHOLDER}")?;
        buf.extend_from_slice(&clap);
    } else {
        // Keep the clap first line (e.g. `#compdef hf2q`) at line 0, then the
        // marker, then the rest. A newline-less clap output (shouldn't happen)
        // is an ERROR, not a prepend: prepending the marker before `#compdef`
        // would break `compinit` autoload AND leave a file `is_hf2q_managed`
        // rejects at marker line 1 (an un-refreshable orphan). Erroring is
        // swallowed best-effort — no broken file is written.
        let nl = clap
            .iter()
            .position(|&b| b == b'\n')
            .ok_or_else(|| std::io::Error::other("zsh registration has no first-line newline"))?;
        buf.extend_from_slice(&clap[..=nl]); // includes the '\n'
        writeln!(buf, "{MARKER}")?;
        writeln!(buf, "{BINDING_PLACEHOLDER}")?;
        buf.extend_from_slice(&clap[nl + 1..]);
        // Make the `#compdef` autoload body actually COMPLETE on the first tab.
        // clap's registration only defines/registers the dynamic helper; when
        // compinit autoloads `_hf2q`, that otherwise makes only the *next* Tab
        // work. The same artifact is also sourced by our verified startup block,
        // where an eager completion call would be wrong. Localize only array
        // semantics (a full `emulate` would clear `_main_complete`'s active
        // completion options), detect the `_hf2q` autoload frame, and invoke the
        // helper only in that mode. The anonymous function makes the check
        // independent of an operator's KSH_ARRAYS setting.
        writeln!(
            buf,
            "() {{\n    setopt localoptions noksharrays\n    if (( ${{funcstack[(Ie)_hf2q]}} > 0 )); then\n        _clap_dynamic_completer_hf2q\n    fi\n}}"
        )?;
    }
    finalize_registration_binding(buf)
}

fn registration_binding(registration: &[u8]) -> String {
    use sha2::{Digest as _, Sha256};
    let digest = Sha256::digest(registration);
    format!("{BINDING_PREFIX}{digest:x}")
}

fn finalize_registration_binding(mut artifact: Vec<u8>) -> std::io::Result<Vec<u8>> {
    let placeholder = BINDING_PLACEHOLDER.as_bytes();
    let matches = artifact
        .windows(placeholder.len())
        .enumerate()
        .filter_map(|(index, candidate)| (candidate == placeholder).then_some(index))
        .collect::<Vec<_>>();
    let [offset] = matches.as_slice() else {
        return Err(std::io::Error::other(
            "generated completion artifact must contain one binding placeholder",
        ));
    };
    let binding = registration_binding(&artifact);
    debug_assert_eq!(binding.len(), placeholder.len());
    artifact[*offset..*offset + placeholder.len()].copy_from_slice(binding.as_bytes());
    Ok(artifact)
}

fn registration_binding_line<'a>(bytes: &'a [u8], shell: &Shell) -> Option<&'a str> {
    let line = bytes
        .split(|byte| *byte == b'\n')
        .nth(shell.marker_line + 1)?;
    let line = std::str::from_utf8(line.strip_suffix(b"\r").unwrap_or(line)).ok()?;
    line.starts_with(BINDING_PREFIX).then_some(line)
}

/// True if the file's line at `marker_line` **starts with** the hf2q ownership
/// prefix (a trailing CR from a CRLF file is tolerated). Anchored to that exact
/// line (bash/Fish line 0; zsh line 1, after `#compdef`), not a substring search,
/// so only a file whose marker line asserts hf2q ownership is a rewrite candidate.
fn is_hf2q_managed(bytes: &[u8], marker_line: usize) -> bool {
    let Some(line) = bytes.split(|&b| b == b'\n').nth(marker_line) else {
        return false;
    };
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    line.starts_with(MARKER_PREFIX.as_bytes())
}

/// True only for a real regular file byte-identical to the registration this
/// binary would write. `symlink_metadata` deliberately rejects operator links.
fn is_exact_regular_registration(path: &Path, desired: &[u8]) -> bool {
    let Ok(metadata) = std::fs::symlink_metadata(path) else {
        return false;
    };
    if !metadata.file_type().is_file() {
        return false;
    }
    std::fs::read(path).is_ok_and(|bytes| bytes == desired)
}

/// Write a complete same-directory temporary file, then commit against the
/// state observed by the caller. The hook exists for deterministic race tests;
/// production always passes a no-op closure.
pub(super) fn atomic_replace_with_hook<F>(
    dir: &Path,
    target: &Path,
    data: &[u8],
    mode: u32,
    expected: &ExpectedTarget,
    temp_label: &str,
    before_commit: F,
) -> std::io::Result<()>
where
    F: FnOnce(),
{
    let mut opened = None;
    for attempt in 0..64_u8 {
        let path = dir.join(format!(
            ".hf2q-{temp_label}.{}.{attempt}.tmp",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => {
                opened = Some((path, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
    }
    let Some((temp, mut file)) = opened else {
        return Err(std::io::Error::other("no unique temporary filename"));
    };
    let result = (|| {
        file.set_permissions(fs::Permissions::from_mode(mode))?;
        file.write_all(data)?;
        file.sync_all()?;
        drop(file);

        before_commit();
        match expected {
            ExpectedTarget::Absent => rename_no_replace(&temp, target),
            ExpectedTarget::Regular { .. } => {
                expected.revalidate(target)?;
                // There is no macOS/Linux compare-and-swap-by-inode rename for
                // replacing an existing path. This rename follows the strongest
                // available immediate identity/content revalidation; the final
                // syscall interval remains documented as a trusted-host race.
                fs::rename(&temp, target)
            }
        }
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn c_path(path: &Path) -> std::io::Result<CString> {
    CString::new(path.as_os_str().as_bytes())
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "path contains NUL"))
}

#[cfg(target_os = "linux")]
fn rename_no_replace_platform(from: &Path, to: &Path) -> std::io::Result<()> {
    let from = c_path(from)?;
    let to = c_path(to)?;
    // SAFETY: both C strings are NUL-terminated and live through the call;
    // AT_FDCWD makes both paths relative to the process cwd as normal paths.
    let result = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            from.as_ptr(),
            libc::AT_FDCWD,
            to.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    (result == 0)
        .then_some(())
        .ok_or_else(std::io::Error::last_os_error)
}

#[cfg(target_os = "macos")]
fn rename_no_replace_platform(from: &Path, to: &Path) -> std::io::Result<()> {
    let from = c_path(from)?;
    let to = c_path(to)?;
    // SAFETY: both C strings are NUL-terminated and live through the call;
    // RENAME_EXCL atomically refuses an existing destination.
    let result = unsafe {
        libc::renameatx_np(
            libc::AT_FDCWD,
            from.as_ptr(),
            libc::AT_FDCWD,
            to.as_ptr(),
            libc::RENAME_EXCL,
        )
    };
    (result == 0)
        .then_some(())
        .ok_or_else(std::io::Error::last_os_error)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn rename_no_replace_platform(_from: &Path, _to: &Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "no platform no-replace rename",
    ))
}

fn rename_no_replace(from: &Path, to: &Path) -> std::io::Result<()> {
    match rename_no_replace_platform(from, to) {
        Ok(()) => Ok(()),
        Err(error)
            if error.raw_os_error().is_some_and(|code| {
                code == libc::ENOSYS || code == libc::EINVAL || code == libc::ENOTSUP
            }) =>
        {
            // Same-directory hard-link creation is also atomic and refuses an
            // existing name. It is the portable fallback for filesystems that
            // do not implement the platform-specific exclusive rename.
            fs::hard_link(from, to)?;
            fs::remove_file(from)
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
#[path = "completion_install_tests.rs"]
mod tests;
