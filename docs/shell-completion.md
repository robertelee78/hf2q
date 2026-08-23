# Shell completion

hf2q installs Tab completion as part of the normal installed experience. There
is no separate completion command to remember after the standalone installer:
the installer invokes the stable installed binary once. Cargo has no
post-install hook, so `cargo install hf2q` provisions completion on the first
normal `hf2q` invocation.

The managed adapters are dynamic. Each Tab asks the installed hf2q binary for
candidates from the current public Clap grammar, so upgrades immediately expose
new commands and values. Hidden installer, transfer, source-teacher, and
process-lifeline surfaces are removed structurally before either dynamic or
static completion is generated.

For every user-facing argument that accepts a local GGUF model or projector,
an empty or bare-name completion starts under
`${XDG_DATA_HOME:-$HOME/.local/share}/hf2q/models` rather than the shell's
working directory. This covers `chat --model`, `generate --model`/`--mmproj`,
`serve --model`/`--embedding-model`/`--mmproj`,
`info --model`/`--mmproj`, and both parity `--model` arguments. Directory
candidates are ordered by name; decoder completion offers non-projector
`.gguf` files, while projector completion offers `.gguf` files whose
conventional filename contains `mmproj`. Returned candidates carry the full
path, so selecting `qwen3.8/` produces a value that is valid from any working
directory. `chat --model` remains free to accept endpoint model IDs and Hugging
Face repository IDs, while `cache clear --model` remains a repository selector
and does not receive local-path candidates.

Because the adapters project the live public Clap grammar, both `serve` and
`info` complete `--ctx`, `--scheduler`, `--max-slots`, and
`--kv-cache-budget` plus the independent disk-backed
`--kv-persist` / `--kv-persist-budget` pair. Setup completes
`--serve-kv-persist-budget`. Serve
additionally completes its typed behavior-default
flags. Removed `--max-seq-len`, `--kv-cache-budget-bytes`, and old
`info --input`/`--repo` spellings are absent from dynamic and regenerated
static output; they are not retained as silent aliases.

An explicit path remains explicit. Values containing a path separator, including
`Desktop/`, `./`, `../`, `~/`, and absolute paths, complete from that location.
A bare name with no managed-model match falls back to the working directory.
Completion reads only the selected directory and returns at most 256 globally
name-sorted candidates: it does not recurse, create the model root, inspect GGUF
contents, access the network, or initialize the inference runtime. Static
completion snapshots retain ordinary shell filesystem completion; the
managed-model preference is provided by hf2q's dynamic adapters.

## Managed locations

- Bash: `${BASH_COMPLETION_USER_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/bash-completion}/completions/hf2q`.
- Zsh: `${XDG_DATA_HOME:-$HOME/.local/share}/zsh/site-functions/_hf2q`, plus a safe current-user-owned Homebrew `site-functions` directory when one exists.
- Fish: `${XDG_CONFIG_HOME:-$HOME/.config}/fish/completions/hf2q.fish`.

For a preferred Bash or Zsh login shell, hf2q adds one bounded block between
`# >>> hf2q managed completion >>>` and
`# <<< hf2q managed completion <<<`. Bytes outside that block are preserved.
Fish discovers its command-named file without startup configuration. A child
process cannot mutate the already-running parent shell, so start a new shell
after the one-time setup notice.

hf2q writes only for a proven standalone or Cargo release installation owned
by the current non-root account. Source builds, debug builds, unmanaged copies,
ambiguous installation receipts, and root invocations do not auto-install.

## Ownership and lifecycle

Every registration contains an hf2q ownership marker, its canonical writer
path, and a generation-binding digest. Updates atomically refresh only a
managed regular file. A foreign regular file at the managed pathname is copied
byte-for-byte to an inert, content-addressed same-directory backup before hf2q
adopts the pathname. Symlinks, directories, and ambiguous startup markers are
preserved.

A private ownership receipt under
`${XDG_STATE_HOME:-$HOME/.local/state}/hf2q/completion-ownership-v1.json`
binds the exact registration bytes and startup blocks. `hf2q uninstall --yes`
and standalone rollback remove only receipt-bound, unchanged artifacts. A file
or block edited after installation is preserved and reported. Update and
rollback invoke the newly active binary once so completion never remains pinned
to a deleted temporary or previous executable.

## Opt out and custom paths

Set `HF2Q_NO_COMPLETION_INSTALL=1` to disable all automatic writes. Presence is
the opt-out signal, so an exported empty value also disables provisioning.

Package maintainers and isolated tests may select explicit destinations:

- `BASH_COMPLETION_USER_DIR`
- `HF2Q_ZSH_COMPLETIONS_DIR`
- `HF2Q_FISH_COMPLETIONS_DIR`
- `HF2Q_COMPLETION_STARTUP_FILE` for one explicitly selected Bash/Zsh startup file

Explicit destinations work for source/debug builds, which makes the behavior
testable without touching an operator's real home. For a fully package-owned
layout, set the opt-out and generate a static snapshot:

```bash
HF2Q_NO_COMPLETION_INSTALL=1 hf2q completions --shell zsh > _hf2q
```

Static snapshots must be regenerated after an hf2q upgrade.

## Troubleshooting

1. Run `hf2q --version` once and read any `completion setup incomplete`
   message on stderr.
2. Start a new shell. For Zsh, confirm the managed block is present in the
   effective `${ZDOTDIR:-$HOME}/.zshrc`; hf2q probes shell-local `ZDOTDIR`
   assignments rather than assuming `$HOME`.
3. Check that a destination is a current-user-owned regular file or safe
   directory. hf2q never writes through a completion-file symlink or replaces
   a non-regular occupant.
4. If an installed binary moved, run the new `hf2q --version`. Managed adapters
   also fall back to the executable currently found on `PATH` when their pinned
   executable disappears.
