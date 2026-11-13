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
# Update the release line in docs/shipping-contract.md and the README status row
# (scripts/test_shipping_contract.sh checks it)
cargo package --locked --allow-dirty --no-verify
ls -l target/package/hf2q-0.1.22.crate   # MUST be under 10 MiB
```

When the published `mlx-native` pin changes, update the one reviewed identity
in `scripts/verify_release_dependency_provenance.sh` together with Cargo.toml and
Cargo.lock. Its `identity` entry checks both the actual manifest requirement and
registry lock version/source/checksum; release/cache/cohort/matched-reference
producers and validators consume that identity. Run
`bash scripts/test_release_dependency_provenance_contract.sh` before release:
it checks the real source inputs as well as synthetic refusal cases, so a
self-consistent stale fixture cannot hide a missed pin update. Historical
receipts remain bound to their original source; do not rewrite them for a new
release. A release that failed because of its source (a build, test, check,
or audit failure) must be corrected/reviewed/merged before a new explicitly
authorized workflow attempt, never retried as the same source. A failure
caused only by infrastructure (a hosted runner never acquired) or by a dead
credential is not a source failure: resume the same run (section 2).

If the crate is 10 MiB or larger, crates.io will reject it with HTTP 413.
Trim it through `exclude` in `Cargo.toml` before going further.

Commit as `chore(release): prepare X.Y.Z`, merge to `main`, and note the
merged commit SHA.

## 2. Run the release workflow

Run the operator script from a checkout with an authenticated `gh`:

```bash
scripts/release.sh X.Y.Z <sha>
```

It dispatches **Release** (`.github/workflows/release.yml`), watches the run,
and prints the release URL on success. When GitHub cancels a job because no
hosted macOS runner was acquired (the job's "not acquired by Runner"
annotation), it re-runs the failed jobs of the same run with backoff
(`HF2Q_RELEASE_MAX_RERUNS`, default 6). Any other failure stops it at once and
prints the failing job and step. To keep retrying a run later, use
`scripts/release.sh --resume <run-id>`. The plain dispatch is
`gh workflow run release.yml -f commit_sha=<sha> -f version=X.Y.Z`.

The self-hosted model qualification job is off by default and runs only if
the owner asks for it (`-f skip_qualification=false`). QE is ad hoc
(ADR-061 D2); it never gates a release.

The workflow builds the candidate (without secrets), then checks the
crates.io and Apple credentials at the start of the protected sign job,
before it touches the candidate. It then signs, notarizes, tags, publishes
the crate, verifies the crate bytes, and only then makes the GitHub Release
public. If any step fails, the draft release MUST NOT be made public by hand.

Resume versus fresh dispatch:

- **Resume** (`gh run rerun <run-id> --failed`, or `scripts/release.sh
  --resume <run-id>`) reuses that run's candidate artifacts (kept 30 days), so
  the draft release receives identical bytes. Use it for runner-capacity and
  credential failures once the credential is fixed. A rerun uses the workflow
  files of the original run.
- **Fresh dispatch** builds and signs a new candidate. The signed binary is
  not byte-reproducible between runs, so the workflow replaces any differing
  asset on a still-draft release (logged as a warning). It never alters a
  published release: a differing or missing asset there fails the run.
- A commit cut before these release-reliability changes cannot be released
  by a fresh dispatch: the build job stops at once with "commit predates the
  release-reliability changes". Finish it by resuming its existing run, or
  release a new version cut after the merge.

### crates.io credential

The publish job exchanges a short-lived crates.io token through Trusted
Publishing (`rust-lang/crates-io-auth-action`) right before `cargo publish`.
One-time setup on crates.io (hf2q → Settings → Trusted Publishing → GitHub):
owner `robertelee78`, repository `hf2q`, workflow filename `release.yml`,
environment `crates-publish`. Restrict that GitHub environment to `main`.
Until it is configured, the exchange fails and publish falls back to the
`CARGO_REGISTRY_TOKEN` secret with a warning. Once a release has published
through Trusted Publishing, delete that secret: while it exists, the sign
job's preflight fails the release if crates.io rejects it (expired or
revoked). A scoped publish token passes the preflight. The exchanged token
lives until the action's post step revokes it at the end of the publish job.

## 3. Update hf2q.us

In the hf2q.us repository, follow `docs/release-activation.md` → "Future
releases" for the new version: take `install.sh` and
`stable-aarch64-apple-darwin.json` from the exact versioned GitHub release,
update the stable record and its pinned hashes, registry identity and evidence,
run `npm run verify`, deploy the built site, and run the transport verifier.
The version-agnostic `public/install.sh` bootstrap and Apache configuration
stay unchanged during a normal release activation; no Apache reload is needed.

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
