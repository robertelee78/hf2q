# Releasing hf2q

Users get hf2q from three places: the GitHub Release, crates.io, and hf2q.us
(the stable record that `hf2q update` reads and the `install.sh` one-liner
redirect). A release MUST reach all three. This checklist is the whole
process; ADR-061 records why.

The key words MUST, MUST NOT, SHOULD, and MAY are to be interpreted as
described in RFC 2119.

QE is not a step of this checklist. A release never waits on it. Agents run
the ad hoc QE playbook (`docs/qe-playbook.md`) only when the owner asks.

## 1. Bump the version and check the crate fits

On a topic branch from current `main`:

```bash
# Edit the version line in the [package] section of Cargo.toml, e.g. 0.1.22
cargo update --workspace --offline   # refresh the hf2q entry in Cargo.lock
# Move the [Unreleased] notes in CHANGELOG.md under a new [0.1.22] heading
cargo package --locked --allow-dirty --no-verify
ls -l target/package/hf2q-0.1.22.crate   # MUST be under 10 MiB
```

If the crate is 10 MiB or larger, crates.io will reject it with HTTP 413.
Trim it through `exclude` in `Cargo.toml` before going further.

Commit as `chore(release): prepare X.Y.Z`, merge to `main`, and note the
merged commit SHA.

## 2. Run the release workflow

Dispatch **Release** (`.github/workflows/release.yml`) with that SHA and
version:

```bash
gh workflow run release.yml -f commit_sha=<sha> -f version=X.Y.Z
```

The self-hosted model qualification job is off by default and runs only if
the owner asks for it (`-f skip_qualification=false`). QE is ad hoc
(ADR-061 D2); it never gates a release.

The workflow tags, builds, signs, notarizes, publishes the crate, verifies
the crate bytes, and only then makes the GitHub Release public. If any step
fails, the draft release MUST NOT be made public by hand. Fix the cause and
re-run the workflow; it resumes from what already exists.

## 3. Update hf2q.us

In the hf2q.us repository, follow `docs/release-activation.md` → "Future
releases" for the new version: take `install.sh` and
`stable-aarch64-apple-darwin.json` from the exact versioned GitHub release,
update the Apache redirect, the stable record, and the pinned hashes, run
`npm run verify`, deploy, `apache2ctl configtest`, reload, and run the
transport verifier.

## Done

A release is done when all three of these work:

```bash
# on a machine running the previous standalone release
hf2q update && hf2q --version              # reports X.Y.Z

# anywhere with Rust
cargo install hf2q --version X.Y.Z --locked

# fresh $HOME
curl -fsSL https://hf2q.us/install.sh | sh && hf2q --version   # reports X.Y.Z
```

If any of them fails, the release is partial. Say which channel is missing
and fix it; do not describe it as released.
