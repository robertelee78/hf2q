# Configure hf2q

`hf2q setup` is a small local configuration step for Apple Silicon. It
inventories the host and records whether future hf2q serving may use a bounded
inactive-session cache. It does not download a model, convert weights, start a
server, edit another application, or calibrate inference.

## Interactive setup

Run:

```bash
hf2q setup
```

Setup shows the selected state root, Apple chip and Metal device, unified
memory, Metal recommended working-set size, macOS and core facts, configured shell,
`RLIMIT_NOFILE`, containing-volume capacity/free space, and the exact
disk-aware recommendation. It then asks:

```text
Keep inactive sessions on disk for fast resume? [Y/n]
```

Enter accepts the displayed default. On a rerun, the current policy is the
default: an enabled policy keeps `[Y/n]` and its current positive limit; a
disabled policy uses `[y/N]`. EOF or interrupted input cancels successfully
without creating the root, lock, partial, config, or cache directories.

## Non-interactive setup

Automation must state the policy completely:

```bash
hf2q setup --session-cache off
hf2q setup --session-cache on --session-cache-limit 32GiB
```

Sizes are canonical unsigned integers with optional `B`, `KiB`, `MiB`, `GiB`,
or `TiB`. Zero is accepted only as the stored disabled policy; `on` requires a
positive limit. A positive override above the recommendation is allowed with a
warning. If the safe recommendation is zero, enabling is refused.

The default state root is `$HOME/.hf2q`. Select a custom standalone root only
with an absolute path:

```bash
hf2q setup --state-root /Volumes/Private/hf2q --session-cache off
```

Existing ancestors must already exist. Setup may create only the final state
root and its private `cache/sessions` directories. It never creates
installation identity, versions, activations, or `current`. If installation
identity already exists, its exact descriptor-bound capability must remain the
same across prompting and publication. Setup validates that identity namespace;
this slice does not certify the full versions/activations/current layout.

## Managed state

Setup writes canonical, bounded UTF-8 TOML at `<state-root>/config.toml`. The
stable v1 fields are the package/schema identity, target, chip, unified memory,
Metal device and recommended working-set size, and `session_cache.limit_bytes`.
Volatile free-space, shell, macOS, core, rlimit, and timestamp facts are never
persisted.

The state root and session directories are mode `0700`; `config.toml`, the
persistent `.config.toml.lock`, and transaction `.config.toml.partial` are
owned, single-link, same-device regular files at mode `0600`. Exact canonical
reruns preserve config inode and mtime. Malformed or future config is retained
unchanged. Precommit interruption leaves a resumable exact prefix; a returned
postcommit durability error is explicit and an exact retry revalidates all
barriers.

## Current boundary

The setup-owned read boundary can now reopen this policy without mutation. It
returns an explicit absent decision when no config exists, a disabled proof for
zero, or a descriptor-bound nonzero proof for a positive limit. The proof
retains and can revalidate the exact state root, optional installation identity,
config bytes/inode, and private `cache/sessions` directories; malformed or
changed state is an error, never a fallback to unlimited.
Every retained regular-file descriptor is read-only, including the optional
installation-identity and lock inodes; this boundary acquires no lock and
retains no write authority.

The positive proof has one consuming transition into a setup-private dormant
managed store; absent and disabled proofs cannot create it. The store uses the
fixed `<state-root>/cache/sessions` descriptor authority, one aggregate hard
cap, a pre-admission volume reserve, immutable checksummed object/catalog
publication, bounded hostile inventories, and exact crash recovery. It remains
inaccessible to serving, models, CLI dispatch, and the legacy zero-unlimited
persistors. A family-specific Qwen/SerialFifo compatibility adapter, bounded
restore, request pinning, access-LRU, and safe replay fallback must land before
this recorded policy becomes active. Corrupt selected dormant-store evidence
currently fails closed and is preserved; it is not yet converted to a cache
miss or quarantine action.
