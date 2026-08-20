# ADR-045: Frictionless distribution, updates, and guided onboarding

- Status: Proposed; product interview, shell-completion bootstrap,
  distribution schemas, crash-durable root identity capability,
  first-activation transaction, signed-update verifier selection, shared
  installation lock, durable metadata journal, dormant
  transport-free production verifier, commit-freshness capability, and
  restart-discard, root-authorized online-role recovery, and dormant
  channel-pointer/selected-target binding, origin-locked artifact transport,
  lock-reauthenticated fetch capability, and same-descriptor streamed archive
  staging, dormant exact embedded-manifest/classic-ZIP validation, and
  lock-held descriptor-relative inert extraction, and dormant native Developer
  ID requirement/signing-information verification, descriptor-bound native
  verification, crash-resumable signed-mode normalization, and dormant
  crash-durable prepared-version publication, plus the dormant live-installed-
  release semantic floor and explicit already-current planning outcome, and
  the closed verifier-request metadata route plus fresh-session
  durable commit coordinator are reconciled;
  canonical Hugging Face model-reference parsing, immutable native Hub
  resolution, bounded index-selected source transfer, Git/LFS byte
  verification, conversion receipt v3, and the closed checked-in Qwen3.8
  source/artifact/hardware/disk recipe, bounded canonical preparation-pair
  receipt, OS-bound host/disk preflight, and sealed host/conversion/pair proofs
  plus the canonical inert no-options preparation layout, exact Hub resolution,
  and complete recipe-metadata transfer authorization are also reconciled;
  the real
  release trust root, real compiled Team ID plus protected positive fixture,
  public update/install/onboarding
  implementation, and exact-artifact proof remain pending
- Date: 2026-08-17
- Updated: 2026-08-19
- Owners: hf2q release engineering and operator experience
- Related: `docs/ADR-044-qwen38-native.md`,
  `docs/ADR-017-persistent-block-prefix-cache.md`,
  `docs/ADR-027-qwen35-tq-kv-cache-and-persist-family.md`,
  `docs/shipping-contract.md`

## Context

hf2q can convert Hugging Face source weights, quantize them, and serve the
result through its Rust and `mlx-native` implementation. Installing and
operating that surface is still a contributor workflow rather than a product
workflow:

- the README starts with a Rust checkout and release build;
- the GitHub release workflow publishes the crate and checksum, but no
  end-user Apple Silicon bundle;
- the CLI has no self-update or first-run setup command;
- remote conversion now has a native immutable-reference/download boundary,
  and a checked-in exact Qwen3.8 preparation recipe, but does not yet compose
  those boundaries into the complete no-options paired text/projector
  preparation, retention, registration, and calibration transaction;
- the canonical `scripts/serve_*_opencode.sh` launchers assume paths under
  `/opt/hf2q`, so copying them out of the checkout is not sufficient; and
- OpenCode and Agentic Kit setup is scattered outside one tested guide.

The desired first-run experience is a user-owned, no-`sudo` installation,
followed by a small hf2q-only setup step and explicit Hugging Face operations.
The user keeps the upstream identity visible from preparation through serving:
`Qwen/Qwen3.8-27B` is downloaded from its official repository, converted and
quantized by hf2q, then served under that same identity. Qwen3.8 is the
reference demo because it exercises text, vision, tool calling, source
provenance, prefix reuse, and restart-safe session restoration.

This ADR governs distribution and onboarding. It does not weaken the product
boundary: installation and official-source conversion may not introduce
Python, `hf`, llama.cpp, MLX-LM, a vendor converter, a pre-quantized substitute,
or an external inference process as an hf2q runtime dependency. An explicitly
named third-party GGUF may be downloaded and served through hf2q's Rust and
`mlx-native` runtime, but it is recorded as upstream-prequantized and never
presented as an hf2q conversion result. This reconciles the conversion boundary
with the existing documented ability to serve explicitly supported external
GGUF files.

## Observable contract

For the first supported target, `aarch64-apple-darwin`, the intended public
workflow is:

```sh
curl -fsSL https://hf2q.us/install.sh | bash
hf2q setup
hf2q convert Qwen/Qwen3.8-27B
hf2q serve Qwen/Qwen3.8-27B
```

Before the branded redirect is enabled, the direct GitHub equivalent is:

```sh
curl -fsSL https://github.com/robertelee78/hf2q/releases/latest/download/install.sh | bash
```

`https://hf2q.us/install.sh` issues an HTTP redirect to the immutable
`releases/download/vX.Y.Z/install.sh` asset selected for the stable channel. It
never hosts a second mutable copy. The GitHub URL is sufficient for the first
public installer and remains usable when the branded front door is unavailable;
enabling the branded route is not a first-release blocker.

The installation contains only hf2q-owned files: the hf2q binary, canonical
supported model launchers, release manifest, licenses, and local hf2q
documentation. It does not install model weights, OpenCode, Node/npm, Agentic
Kit, SearXNG, Firecrawl, Crawl4AI, or another third-party tool or service.
Models are downloaded only when the user later invokes `hf2q convert` or
explicitly asks `hf2q serve` to acquire a published GGUF.

The core guide is complete without an integration. Separate optional guides
show how to install OpenCode through its official distribution, connect that
existing installation to hf2q, optionally add Agentic Kit, and optionally add
search/crawling services. `--opencode` or an `hf2q opencode` command means
"configure an already installed OpenCode"; it never means "install OpenCode."

Every hf2q invocation evaluates trusted local update state. When a newer stable
release is known, hf2q prints one non-forcing notice to stderr and continues
the requested command. `hf2q update` is the universal user-facing update
command: it self-updates standalone installs atomically and delegates through
the detected package owner when the executable is package-managed.

Every normal release-binary invocation also best-effort reconciles hf2q-owned
static clap completion scripts in the standard per-user Bash, Zsh, and Fish
locations. The preferred Bash or Zsh startup file receives one bounded managed
source block, so a newly started shell has Tab completion without a manual
activation command. Setup failures never fail the requested hf2q command;
foreign completion files, symlinks, non-regular paths, and malformed managed
startup blocks are preserved. `HF2Q_NO_COMPLETION_INSTALL=1` opts out, while
debug/test binaries require explicit isolated destinations and never discover
live user paths. `hf2q completions --shell <shell>` remains the packaging and
manual-generation surface. A child process cannot modify the already-running
parent shell, so the zero-config guarantee begins with the next shell after the
first normal hf2q invocation.

## Domain model

Distribution and model preparation use explicit records rather than inferring
state from PATH aliases or filenames:

- **Release bundle**: an immutable archive for one version and target,
  containing the binary, launchers, docs, licenses, and release manifest.
- **Release manifest**: version, target, minimum macOS version, source commit,
  exact file inventory and modes, digests, signing identity, update channel,
  and compatibility/schema versions. The manifest wire schema, minimum
  installer protocol, minimum updater protocol, and launcher-registry schema
  are independent integers: an unsupported document schema or a required
  protocol/schema above the verifier's capability fails closed.
- **Installed-version marker**: immutable, version-local standalone ownership
  state binding one installation ID and root to one version, target, release
  manifest/archive digest, nonzero installation sequence, preparation time,
  and the four authenticated metadata-role versions that prepared it. The
  recorded role versions make the exact activation receipt reconstructible
  after a crash or later metadata refresh. They are local correlation and
  audit evidence, never transmitted update identity or authentication.
- **Install receipt**: for standalone ownership, an immutable activation record
  selected atomically with its version; for manager/manual ownership, a
  non-authoritative last-observation record. It contains the configured state
  root, installation root, active and two retained versions, last observed owner
  family (`standalone`, `homebrew`, `cargo-registry`, or `unknown/manual`),
  selected update route (`standalone`, `brew`, `cargo-install`,
  `cargo-binstall`, or absent), release-bundle identity where that channel
  provides one, standalone marker identity only for hf2q-owned versions, and
  last successful transition. Cargo and cargo-binstall
  deliberately share an owner family because both install into Cargo's
  registry/bin layout; the route is a separately recorded preference rather
  than invented history. A completed migration is a transition from
  `unknown/manual` to `standalone`, after which the durable route is
  `standalone`; one-time consent is never a reusable update route.
- **Update metadata**: signed, expiring, monotonically versioned metadata that
  binds a channel and version to an exact release-manifest and archive digest.
- **Hugging Face model reference**: the user's original input plus normalized
  repository ID/type, canonical URL, immutable revision, and optional exact
  filename. Repository IDs and equivalent `huggingface.co` model, `tree`,
  `blob`, and `resolve` URLs share one parser and canonical identity.
- **Model recipe**: a checked-in supported recipe binding the canonical model
  reference, accepted source files, text/projector outputs, independently
  accepted quantization candidates, exact proven hardware profile, disk
  preflight, source-retention policy, and acceptance status. Recipe bytes are
  embedded policy, not caller, cache, or network input; parsing them alone
  grants no conversion, serving, installation, or activation authority.
- **Prepared model**: hf2q-produced local artifacts plus a conversion receipt
  binding source revision and hashes, converter identity, recipe, output
  digests, and exact text/projector relationship.
- **Verified external GGUF**: an explicitly requested upstream-prequantized
  artifact plus repository revision, filename, size, digest, projector binding,
  embedded metadata, and honest external provenance.
- **Hardware profile**: measured hardware, unified memory, disk state, and the
  candidate-selection inputs produced by `hf2q setup`.
- **Artifact calibration receipt**: bounded runtime measurements for one exact
  artifact/hardware/hf2q combination, including memory residency, context,
  slots, shared KV budget, prefill batch, TTFT, prefill, and decode results.
- **Session restore policy**: setup-recorded consent, storage root, byte limit,
  permissions, eviction rules, and eligible family/scheduler combinations for
  persisted prefix/KV checkpoints.

Model and session data are not stored below the versioned installation root
and are never removed, replaced, or migrated by `hf2q update`.

## Decision

### 1. Build and promote one exact release bundle

The release pipeline produces a versioned Apple Silicon archive such as
`hf2q-vX.Y.Z-aarch64-apple-darwin.zip` with this logical layout:

```text
bin/hf2q
libexec/serve_qwen38_opencode.sh
libexec/serve_qwen36_opencode.sh
libexec/serve_gemma4_opencode.sh
libexec/serve_deepseek4_opencode.sh
share/doc/hf2q/
share/licenses/hf2q/
release-manifest.json
```

The four launcher names shown above are the complete v1 path allowlist, not a
requirement that every release contain all four. Each release includes only
the subset whose family-specific exact-artifact gates have already passed.

`release-manifest.json` is the bundle's reserved envelope. Its sorted file
array inventories every payload entry except itself; every representable entry
is a regular file with an exact path, type, size, mode, and SHA-256. Archive
verification requires the entry set to equal
`{release-manifest.json} ∪ manifest.files`. Signed update targets bind the
external manifest bytes, and the embedded copy must be byte-for-byte identical
to that separately verified external target. Neither the manifest nor its
embedded copy attempts the impossible operation of hashing itself or its
containing archive.

The v1 archive profile is deliberately narrower than general ZIP. It is a
classic single-disk archive whose EOCD occupies exactly the final 22 bytes, with
no prefix, trailing bytes, archive comment, ZIP64 record, split disk, extra
field, per-entry comment, encryption, data descriptor, or discretionary flag.
Entry zero is `release-manifest.json`; the remaining entries follow the
manifest's already sorted file order. Every raw name is the exact safe ASCII
manifest path. Local records are contiguous from offset zero, the central
directory begins at the exact end of compressed data, and every local header
must equal its central record for name, method, flags, timestamp, CRC, and
sizes. Only Stored and Deflate are accepted. Creator metadata must identify
Unix and each external mode must be an exact regular-file `0644` or `0755`.
Stored requires extraction version 1.0 and Deflate requires 2.0; both local and
central timestamps use the canonical DOS epoch. Stored compressed size must
equal uncompressed size. Every file is smaller than the classic-ZIP
`0xffffffff` sentinel. After the bounded hf2q structural pass, pinned
`flate2` 1.1.9 with the Rust `zlib-rs` backend must observe a real raw-Deflate
`StreamEnd`, consume the exact declared compressed range without ignored suffix
bytes, and emit the exact declared uncompressed length. Pinned `zip` 7.2.0 then
supplies the independent Stored/Deflate decode, CRC, and payload-digest view;
hf2q never calls its generic extractor. The embedded manifest must also equal
the deterministic external manifest encoding. The archive is reverified on its
original unlinked descriptor before and after this pass. This validation
remains inert and creates no filesystem-write, prepared-version, activation, or
install authority.

The next dormant boundary consumes that exact profile and extracts only into
`update/extractions/.extract-v{VERSION}-{ARCHIVE_SHA256}` while holding the
shared installation lock. The stage name is derived solely from the
authenticated stable SemVer and full lowercase archive SHA-256. There may be
at most eight retained extraction stages; an existing requested stage remains
resumable at the limit, while a ninth distinct stage fails closed. This slice
has no deletion or garbage-collection authority, so exhaustion is recoverable
only through a later explicitly authorized cleanup policy. At the 4 GiB
expanded-payload ceiling, the deliberately conservative worst-case retained
payload budget is about 32 GiB plus manifest and filesystem overhead.

The manifest and every payload are decoded in canonical archive order into an
exact descriptor-relative tree. The tree admits at most 4,096 unique derived
directories. Raw extraction leaves every stage/derived directory private
`0700` and every file—including payloads whose signed final mode is `0755`—
inert `0600`. Extraction never calls a generic ZIP extractor and returns no
path, file descriptor, marker, receipt, prepared-version, or activation
capability. The sealed next transition verifies the same descriptor-bound
`bin/hf2q`. That ephemeral proof is bound to the live extraction namespace and
stage identities, executable path/device/inode/size/mode, and exact manifest
digest; normalization rederives the complete binding before consuming it, so a
proof from another root or stage cannot be substituted. It then applies the signed
`0644`/`0755` modes within the same private stage root, replays current TUF
state, verifies the same path/inode again, and finally reopens and rehashes the
complete normalized tree. The stage root remains `0700`, and no name under
`versions/` can exist in this boundary.

Restart recovery treats a correctly named, current-user, single-link,
same-device `0600` expected file as reconstructible scratch. A newly
authenticated copy of the exact archive is decoded from byte zero; retained
bytes are compared, mismatching ranges are overwritten in the same inode, and
missing suffixes are appended. No file is truncated, replaced, or deleted.
Oversized files, non-prefix file ordering, extra names, wrong types, links,
owners, devices, or modes remain diagnostic evidence and fail closed. This is
required because Apple's [`fsync(2)` documentation](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fsync.2.html)
does not promise power-loss write ordering. Each reconstructed file is checked
for exact size/SHA-256, `F_FULLFSYNC`ed, reopened and rechecked; all directories
are synced bottom-up, the named root/update/extractions/stage namespace is
reopened and rebound, and the installation lock provides the final full-sync
endpoint. A process crash or later corruption can therefore leave only inert
reconstructible state: no retained byte is trusted, and a file that cannot be
reconstructed exactly from a freshly authenticated archive remains fail closed
rather than becoming release authority.

The same selected metadata identity is replayed from the compiled anchor at
current time before opening the stage and again after all archive and
filesystem I/O. The archive's anonymous descriptor is independently
length/hash-revalidated on both sides of extraction. Generation drift, pointer
or descriptor mismatch, expiry at the final sample, clock rollback, namespace
replacement, or archive mutation returns no extracted-release capability. The
successful sealed result retains the shared lock and the unforgeable classic-
ZIP profile, but is still inert and cannot authorize publication.

Developer ID validation, signed-mode normalization, installed-marker creation,
and prepared-version publication are one first-install-only transaction under
that retained installation lock. An internal code-signed-tree proof may divide
the implementation, but it is neither durable nor reusable authority. The
transaction moves the already authenticated tree without copying its payload:

```text
update/extractions/.extract-v{VERSION}-{ARCHIVE_SHA256}
  -> update/prepared/.pending-v{VERSION}-{ARCHIVE_SHA256}
  -> versions/{VERSION}
```

`update/prepared/` and each hidden pending root are private `0700`. Before the
final rename, `versions/` is absent or empty and no `current` or activation may
exist. This slice introduces no update-retention, pruning, or generic cleanup
authority. Its bounded inventory admits only one exact transaction shape; an
unrelated name, multiple intents, pending plus final, or other ambiguous state
is retained and fails closed.

The marker's one nondeterministic input is persisted in a bounded canonical
intent name before any tree move:

```text
.marker-v{VERSION}-{ARCHIVE_SHA256}-t{UNIX_SECONDS:020}.partial
.marker-v{VERSION}-{ARCHIVE_SHA256}-t{UNIX_SECONDS:020}.ready
```

The exact marker and fully derived receipt are rebuilt from the authenticated
installation identity, release descriptors, four selected role versions, and
that timestamp. The partial file uses exact-prefix recovery, is full-synced,
and becomes `ready` by a no-replace rename. The ready inode is then moved—not
copied—into the pending tree as `version-installation.json`, so the published
tree can contain no torn marker. Exact marker digest and full canonical receipt
equality are required on every recovery.

Mode normalization accepts only a prefix of the canonical order: manifest,
payload files in manifest order, then derived directories deepest-first. Files
move from inert `0600` to their exact signed `0644`/`0755` modes; derived
directories move from `0700` to `0755`; the version root stays `0700` and the
marker stays `0600`. Each transition operates on the retained descriptor,
rechecks device/inode/owner/link count/size/hash, full-syncs, rehashes, and
revalidates its name. Any non-prefix mixture fails closed.

The deterministic extraction stage is also the normalization crash journal.
A final-mode file is accepted only as part of the exact canonical file prefix,
at its complete authenticated length and SHA-256; it is never rewritten or
repaired. A final-mode derived directory is accepted only as part of the exact
deepest-first/path-sorted directory prefix after every file is final. Returned
errors and real process aborts after every file-mode, file-full-sync,
directory-mode, directory-sync, namespace-sync, live-rebind, and endpoint
full-sync barrier must be recoverable by a fresh exact archive replay. The
successful normalized capability retains the same installation lock and grants
no path, descriptor, marker, receipt, publication, activation, or deletion
authority. A mutation of any non-binary payload between the native brackets is
also rejected by the final whole-tree identity/mode/size/hash replay.

After all files and directories are full-synced bottom-up, the transaction
replays the same selected TUF generation at current time, reopens and rebinds
root/update/prepared/versions/pending/marker/lock/binary identities, and repeats
the complete tree, marker, receipt, Mach-O, and Developer ID checks. The sole
prepared-version commit is descriptor-relative no-replace rename from the
hidden pending name to `versions/{VERSION}`. Before it, every error means no
version was published. After it, every error is a typed
`PreparedVersionDurabilityUnknown { version }`; exact recovery reopens the final
inode, repeats all content and media barriers, and adopts only that same
marker/receipt. Install-state accepts only a one-use update-auth commit guard,
not an arbitrary successful callback; that guard samples after the final
namespace rebind and immediately before the rename. A second one-use receipt-
bound token samples again after all postcommit tree/media/native verification,
and only then constructs `AuthenticatedPreparedVersion`.

While a prepared intent exists—or while an unactivated first version exists—
metadata selection may not advance. The shared lock prevents concurrency, and
this additional inventory guard preserves the exact role versions needed to
reconstruct a crash-interrupted marker. Recovery completes or fails closed
before a new metadata transcript can stage.

The binary is Developer ID signed with Hardened Runtime and a secure timestamp.
The stable v1 signing identifier is the compiled literal `us.hf2q.cli`; the
final ten-character Apple Team ID is likewise compiled policy and may not be
selected by the manifest, environment, command line, or network. The
authenticated manifest must repeat both exact literals before any code-signing
operation. Its certificate common name is cross-checked against the actual leaf
certificate for evidence, but it is not a trust root and is never interpolated
into a code requirement. No production verification constructor exists until
the real Team ID is checked in and a real Developer ID Application fixture has
passed the protected release gate.

V1 accepts exactly one little-endian thin arm64-ALL `MH_EXECUTE`, not a
fat/universal or arm64e binary. The filesystem-free bounded read-at parser
requires `MH_NOUNDEFS`, `MH_DYLDLINK`, `MH_TWOLEVEL`, and `MH_PIE`, permits only
the explicitly frozen safe header bits, and rejects all other header flags. Its
positive load-command profile consists only of structurally exact segment,
dyld-info, symbol/dynamic-symbol table, UUID, function-starts, data-in-code,
build-version, dylinker, public system-dylib, modern-entry-point, and terminal
code-signature commands. Unknown commands and legacy, weak, re-export, lazy,
upward, encryption, rpath, environment, and thread-entry variants fail closed.

Segments, sections, and link-edit payloads have checked sizes and offsets,
nonoverlapping file and VM ranges, valid protection subsets, no writable-
executable mapping, and exact parent/mapping/alignment relationships. There is
one nonempty regular instruction-only `__TEXT,__text`; its relocation and
reserved fields are zero and the four-byte-aligned `LC_MAIN` entry lies inside
it. There is one read-only `__LINKEDIT`; required link-edit payloads precede and
do not overlap the nonempty `LC_CODE_SIGNATURE`, which ends exactly at EOF.
`LC_DATA_IN_CODE` may be empty, otherwise its length is a whole number of
eight-byte entries. One macOS `LC_BUILD_VERSION` has an SDK no older than its
minimum deployment target, and that minimum equals the manifest. The canonical
dylinker is `/usr/lib/dyld`; `LC_LOAD_DYLIB` dependencies are absolute public
paths below `/usr/lib/` or `/System/Library/Frameworks/`. The v1 manifest
therefore requires an empty `non_system_dynamic_dependencies`; bundled
libraries require a later schema revision.

Runtime verification uses native Security.framework, never parsed `codesign`
output. A requirement built only from the two compiled literals requires the
Apple generic and trusted anchors, the Developer ID Application leaf and issuer
OIDs, and the exact Team/identifier. `SecStaticCodeCheckValidity` runs with all
architectures, strict validation, trusted-anchor checking, and no network.
After that succeeds, a small private FFI bridge type-checks signing information
and requires the exact Team and identifier, a real TSA `CFDate` timestamp,
Hardened Runtime, no ad-hoc or linker-signed flags, no raw or dictionary
entitlements, and the manifest-bound leaf common name. Ordinary certificate
expiry remains compatible with a valid secure timestamp; install-time
verification does not force current-expiration or online revocation checks.

The dormant verifier pins `core-foundation` 0.10.1,
`security-framework` 3.7.0, and `security-framework-sys` 2.17.0 on macOS. The
safe wrapper owns requirement/static-code construction and validation; one
small private FFI bridge owns only `SecCodeCopySigningInformation` and the
documented `kSecCodeInfo*` keys absent from the wrapper crate. Every returned
Core Foundation value is type-checked, every create-rule object is released
once, the certificate chain is bounded to one through eight and type-checked
entry by entry, and neither errors nor `Debug` expose paths or signing data. A
test-only policy exercises the complete fail-closed field matrix and the FFI
bridge against an Apple-signed system binary. The current test executable is
rejected. A protected real Developer ID Application positive test exists but
remains explicitly ignored in hosted CI; no production policy constructor may
exist until the real public Team ID is compiled and that gate passes.

Security.framework is path-oriented. The updater keeps the descriptor-relative
binary open, obtains its macOS path, requires a no-follow reopen of that path to
match the retained device/inode, and brackets the native validation with repeat
signature checks plus descriptor hash, named identity, stage identity, and
manifest-byte validation. This detects ordinary replacement and persistent
swaps but, like Apple's API, cannot defeat an actively malicious same-EUID ABA
swap inside the path-only call. That actor remains part of the already excluded
fully compromised local-account threat.

The final ZIP is submitted with `notarytool`; release promotion requires an
`Accepted` result and retained submission log for those exact bytes. [Apple
documents](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow)
that ZIP archives are accepted for notarization but tickets cannot be stapled
to a ZIP or standalone executable, so notarization is release evidence, not an
install-time `spctl` dependency. The archive and manifest are
checksummed and covered by GitHub artifact attestation. The pipeline records
the exact source commit, Developer ID team identifier, and minimum supported
macOS version.

The candidate archive is created before real-hardware release acceptance. The
Apple Silicon gate installs and runs that archive, not a neighboring workspace
binary. Publication promotes those identical archive bytes. It does not
rebuild, re-sign, re-archive, or replace an asset after the hardware gate.
The manifest also lists every non-system dynamic dependency, and the clean-host
gate proves the bundle does not accidentally depend on a Rust toolchain, source
checkout, Homebrew library, or build-machine-only path.

The existing crate publication remains supported, but the current release
workflow's mutable `gh release upload --clobber` path is not used for immutable
binary releases. A release is drafted, all final assets are attached once, and
the draft is published only after exact-artifact gates pass. A published asset
or tag is never overwritten. GitHub immutable releases are enabled before the
first bundle is published, and every GitHub release eligible to become
`latest` carries an `install.sh` asset so
`releases/latest/download/install.sh` cannot point at a crate-only release.
Crate-only publication either creates no GitHub release or marks it as a
prerelease that cannot become the stable `latest` target.

The launcher list is an allowlist, not an assertion that a filename alone is
supported. A launcher enters a public bundle only after its governing model
family's exact-artifact release gates have passed. Every included launcher also
has an installed-layout test that proves relative binary/profile resolution and
that missing, incompatible, or conflicting prerequisites fail before a large
model load with the correct recovery command.

The accepted bundle is bound to the hardware gate's exact head SHA. Publication
requires that SHA to remain an ancestor of `origin/main`; it does not require
the gated SHA to remain the current tip after a long hardware run. This
supersedes the release workflow's tip-equality rule for binary-bundle
promotion, while the tag and every receipt remain bound to the gated SHA.

### 2. Install without root and preserve rollback

The standalone installer is a small auditable script compatible with Bash 3.2
and with zsh, the standard interactive shell on current macOS. It installs all
hf2q-owned state below one root, except for the user PATH entry point:

```text
~/.hf2q/
├── versions/X.Y.Z/
│   └── version-installation.json
├── activations/N/
│   ├── install-receipt.json
│   └── version -> ../../versions/X.Y.Z
├── current -> activations/N
├── models/
├── cache/sessions/
├── receipts/
├── update/
│   ├── install.lock
│   ├── .noreplace-source
│   ├── .noreplace-target
│   ├── installation-identity.json
│   ├── downloads/
│   ├── extractions/
│   ├── prepared/
│   └── metadata/
│       ├── current.json
│       └── generations/
│           └── N/
│               ├── generation.json
│               ├── anchor-root.json
│               ├── root-chain/R.root.json
│               ├── trusted-root.json
│               ├── timestamp.json
│               ├── snapshot.json
│               └── targets.json
└── config.toml
~/.local/bin/hf2q
```

`current` is the sole standalone activation commit point. It selects one
immutable activation directory containing both the install receipt and one
fixed relative link to the verified version directory, so executable and
receipt change with the same rename. Each complete version directory contains
its immutable local `version-installation.json` in addition to the signed
bundle contents. The marker is not an archive entry and is verified separately;
the installed entry set is therefore the manifest envelope, its exact payload
inventory, and this one fixed local marker. Its receipt digest binds the exact
canonical marker bytes; hashing parsed or reserialized JSON is forbidden. No
mutation proceeds merely because the activation receipt or marker parses; live
descriptor-relative ownership, manifest, link, and entry-point evidence must
agree. The activation directory's exact entry set is `install-receipt.json`
and `version`; ownership verification and cleanup use descriptor-relative
`lstat`/`fstatat`-style no-follow operations and reject any extra entry.

`N` is the receipt's nonzero transition sequence rendered as one canonical
20-digit zero-padded decimal component. The raw `current` target is exactly
`activations/N`, and the activation's raw `version` target is exactly
`../../versions/X.Y.Z`; absolute targets, traversal variants, extra components,
and link chains fail ownership verification. Under the nonblocking installation
lock, a transition builder verifies the live prior activation and requires
`N = prior N + 1` without wraparound; the initial standalone activation is
sequence one. The receipt schema additionally requires install, update, and
confirmed-migration sequences to equal the newly installed version sequence,
and rollback sequences to exceed every active or retained installation
sequence. A parsed receipt alone cannot establish this live monotonic floor.

The v1 standalone filesystem contract uses mode `0700` for the state root,
`versions`, `activations`, `update`, version roots, activation roots, and
private transaction staging directories; manifest-derived payload directories
use `0755`. The immutable activation receipt and installed-version marker use
`0600`, the manifest envelope uses `0644`, and payload modes come from the
manifest. `update/install.lock` is a regular `0600`, single-link, current-user
file opened without following links, locked exclusively without blocking, and
never unlinked. The two empty private no-replace markers are fixed owned state,
not temporary cruft; attempting to rename source over target must fail with
`EEXIST` before any activation is published. Every descendant must remain on
the root device. An explicit
custom root is traversed component-by-component from `/` without following
symlinks; only normal UTF-8 components are accepted. Existing ancestors need
not be private or user-owned, but they must already exist: explicit authority
to create one root never authorizes creation of missing ancestors. The final
root may be created and must be current-user owned mode `0700`.

Before metadata, download, extraction, prepared-version, activation, or
uninstall state can exist, hf2q commits one immutable root identity at
`update/installation-identity.json`. The canonical v1 wire is compact JSON plus
one LF, capped at 16 KiB, and contains only kind
`hf2q.installation-identity`, schema version 1, state-layout schema 1, package
`hf2q`, one canonical lowercase UUIDv4 installation ID, and the canonical
absolute state root. It deliberately has no creation timestamp: every byte is
reconstructible from the reserved intent name and explicit root authority.
Unknown/duplicate fields, trailing documents, noncanonical encoding, UUID
aliases, a non-v4 UUID, and noncanonical roots fail closed.

Bootstrap first creates and syncs the private root/update/no-replace/lock
scaffold under the shared nonblocking lock. It then creates at most one
`update/.installation-identity-v1-{UUID}.partial` regular `0600`, single-link,
same-device intent. Existing intent bytes must be an exact prefix of the one
canonical record; conflicting, oversized, wrongly typed/mode/owned/linked,
malformed, multiple, or over-cap residue is retained without deletion and
fails closed. The complete intent is synced and `F_FULLFSYNC`ed, the live
root/update/lock/intent namespace and exact bytes are rebound, and a
descriptor-relative no-replace rename to `installation-identity.json` is the
sole commit point. After that rename, any reopen or
file/update/root/lock-durability failure is
`IdentityCommittedDurabilityUnknown { installation_id }`, and an exact retry
recovers the UUID from the final or intent name, repeats all barriers, and
returns the same identity. No retry generates a replacement UUID when either
durable name exists.

The final identity is not authority merely because its JSON parses. Ordinary
open retains the exact root, update, lock, and identity-file inodes and exact
identity bytes, then repeats the named snapshot. Lock acquisition and every
metadata, artifact, extraction, and activation reopen require those same live
bindings; replacing an inode with byte-identical content cannot authorize a
transition. The bounded `update/` inventory recognizes only the fixed
no-replace/lock/identity entries plus metadata, downloads, extractions, and
prepared-version staging;
the state-root inventory is intentionally not globally closed because hf2q
also owns preserved state such as `ruvector`, models, and caches. When no final
identity exists, any metadata/download/extraction/version/activation/current
or uninstall state is inconsistent and bootstrap rejects it before mutation.
A truly empty or unrelated preidentity root remains eligible, and read-only
absence checks do not create it.

The first filesystem implementation is deliberately narrower than ownership
or update. Given explicit root authorization and an already authenticated,
fully prepared version, it may create only an `install` transition at
standalone sequence one; confirmed migration remains gated on a separate
one-time consent capability and live source evidence. Its
lock-bound, descriptor-backed prepared capability is non-cloneable and grants
no update, overwrite, entry-point, manager, pruning, or deletion authority.
If a crash publishes the complete version or activation before `current`, a
retry may adopt it only after re-verifying the currently authenticated exact
manifest, marker, receipt, inventory, and raw link bytes. Partial or conflicting
state fails closed; it is never blindly deleted or adopted. A temporary entry
is resumed only when every existing byte/type/mode is the exact expected
prefix, so a successful retry leaves no transaction cruft. Receipt publication
uses the one fixed `.install-receipt.json.partial` name: an absent file is
created privately, a bounded byte-for-byte prefix is resumed, conflicting
bytes fail closed, and only the complete synced file is renamed no-replace to
`install-receipt.json`.

Root and `update/` creation are a minimal race-safe lock bootstrap. `update/`
is opened and the nonblocking lock acquired before `activations/` or any other
transition state is created; an already prepared `versions/` must exist. A
fresh-root race adopts an exact private root/update directory and then yields
one lock holder plus `Busy`, never an `EEXIST` transition failure. Preparation,
the immediate precommit check, and postcommit recovery reopen the root-relative
`update`, `versions`, exact version, and `activations` names, bind their inode
identities, and reverify the resolved content rather than trusting stale file
descriptors or link strings. Every version file is full-synced and every
derived directory plus `versions/` is synced bottom-up before selection.

`current` is committed with a no-replace descriptor-relative rename. Before
that rename, every error means no activation was selected. After that rename,
an error from directory sync or macOS `F_FULLFSYNC` is reported distinctly as
`CommittedDurabilityUnknown { sequence }`: the caller must treat sequence one
as potentially active and re-open/verify instead of retrying as though nothing
changed. Unsupported no-replace or full-sync behavior fails in preflight. The
v1 binary deployment target and public release floor are macOS 14.0. The
future binary-bundle release job must set `MACOSX_DEPLOYMENT_TARGET=14.0` and
reject the packed executable unless `vtool`/`otool` proves that exact minimum;
source-crate workflows do not pretend to prove a binary deployment target.
The activation verifier reads the actual host version without a subprocess,
compares versions numerically rather than lexically, and rejects both a host
below the release requirement and a release below the v1 floor. Public README
requirements change only when that packed binary gate passes.

If postcommit durability is unknown, reopening a matching activation repeats
version-file, version-directory, activation-directory, root-directory, and
receipt full-sync barriers before returning durable `AlreadyCommitted`.
The advisory lock and repeated namespace checks reject observable accidental
or non-cooperating swaps, but no name-based POSIX protocol can defeat an
actively malicious same-EUID process in the final check-to-rename instruction
window. That case is part of the already excluded fully compromised local
account threat, not an ownership guarantee claimed by this capability.

Activation and authenticated-metadata transitions use one non-cloneable,
descriptor-backed lock capability over the same `update/install.lock` entry.
The first-activation behavior does not change when that lock ownership is
shared: the root, named `update` directory, and lock inode are reopened and
revalidated before either bounded context reaches a commit point. A metadata
refresh therefore cannot race an installation transition, and neither path may
acquire an independent lock with a different namespace meaning.

Install-receipt v1 has independent receipt, state-layout, and
installation-layout schema integers. It records a local-only random
installation ID; active/retained standalone releases bind the target, manifest
digest, archive digest, exact installed-marker digest, and nonzero installation
sequence. Retained order is activation recency, not SemVer order, and contains
at most two releases distinct from the active release. The last successful
transition records a monotonically increasing nonzero sequence, type, from/to
owner and release, its recorded evidence class, and a diagnostic completion
time. Verified standalone
install/update/migration transitions bind the versions of all four update
metadata roles; rollback binds the retained manifest; manager updates bind the
selected manager route. These are audit claims until live evidence is
reverified. A manager/manual active release may omit bundle identity, or may
bind a manifest/archive when that channel ships them, but it never claims a
standalone marker or retained release. No receipt contains arbitrary deletion
paths or manager argv.

Installed-version marker v2 is the first public-state candidate. It adds the
closed `prepared_from` evidence object with kind `verified-update-metadata` and
nonzero root, timestamp, snapshot, and targets versions. A narrow typed builder
emits the canonical marker bytes, hashes those exact bytes into install-receipt
v1, and derives the receipt transition evidence and completion time from the
same marker inputs. This makes a completely published version self-sufficient
for exact receipt reconstruction after restart. First-activation verification
reconstructs that record from the exact durable marker bytes and requires the
candidate receipt to equal the derived receipt in full; a matching marker
digest alone is insufficient. The marker's `installed_at_unix_seconds` is the
single diagnostic time fixed for first-install preparation and copied into the
receipt's `completed_at_unix_seconds`; recovery never resamples or rewrites it,
and it is not a claim about the wall-clock instant of a later `current` rename.
The earlier marker-v1 bytes
were reachable only from dormant code with no production entry point; they are
retained as a regression fixture and rejected as unsupported rather than
silently migrated. Once any public writer exists, a future marker change
requires an explicit dual-reader migration release before a new writer.

The executable in `~/.local/bin` resolves `current/version/bin/hf2q`. Installed
launcher entry points resolve paths relative to that version and use the
model/profile registry; they contain no `/opt/hf2q` assumption.

The installer:

1. fails early unless the host and target are supported;
2. supports a pinned `--version`, `--no-modify-path`, and an explicit install
   root override;
3. downloads the archive and its release-attached signed bootstrap metadata
   snapshot into a private temporary directory; the expected metadata location
   is an immutable versioned release URL, not the mutable update endpoint;
4. performs a listing-only archive validation, rejecting duplicate or
   unexpected entries, absolute paths, traversal, device nodes, and links
   before extracting anything;
5. extracts only the uniquely named regular `bin/hf2q` into private staging,
   checks its SHA-256 against the target-specific value embedded in the
   immutable versioned `install.sh`, validates it with
   `codesign --verify --strict`, and requires the exact release-pinned Developer
   ID team identifier before executing it;
6. uses that staged binary's constrained verification mode and embedded
   threshold root to authenticate the bootstrap metadata, then obtains the
   expected archive and manifest digests only from that verified metadata;
7. verifies the archive digest with macOS `shasum`, extracts the remaining
   allowlisted files, and verifies the manifest, every file digest/mode, and the
   complete inventory before activation;
8. writes and syncs the immutable installed-version marker and installs the
   fully verified directory under a never-replaced version name;
9. creates and syncs a never-replaced activation directory containing the
   complete install receipt and exact relative version link, then atomically
   switches and syncs `current` as the only activation commit while retaining
   the active version plus two prior verified versions; and
10. updates shell PATH idempotently when needed, then runs a
   packaged-install-aware
   `hf2q doctor`.

The live update-metadata service is not required while a release-attached
signed snapshot is still valid. The normal and high-assurance local-file flows
can then finish when the live update endpoint and Apple's notary service are
unreachable. An explicitly selected older `--version` whose attached metadata
has expired must obtain current threshold-signed metadata or fail before
activation; explicit downgrade intent never bypasses signature or expiry
checks. The release pipeline's accepted notarization record is checked before
publication, while installer trust rests on local Developer ID verification,
threshold-signed metadata, and exact digests. It does not treat `spctl`
acceptance of an extracted bare CLI as a portable gate.

PATH mutation supports zsh and Bash in the first release. The installer first
checks the effective PATH and recognizes equivalent `$HOME/.local/bin` forms.
If no change is required it writes nothing. Otherwise it preserves existing
PATH logic and permissions and atomically adds exactly one marked hf2q-owned
block to the active shell's appropriate user startup file, with a backup.
It never edits Bash files merely because they exist when zsh is the active
shell. `--no-modify-path` skips the edit, unknown shells receive copyable
instructions, repeat installs do not duplicate text, and uninstall removes
only the exact owned block.

Before the `current` commit, interruption leaves the prior activation usable.
After the commit, `current` selects the complete new activation receipt,
relative version link, directory, and immutable marker. Partial activation
directories are never selected, and no independent active-receipt rename can
produce a receipt/current mismatch.
The installer never requires `sudo` and never writes `/usr/local` or another
system prefix by default. It installs no model or optional integration, invokes
no third-party installer, and never runs npm, Docker, Homebrew, or an OpenCode
installation command. Its final action is to print `hf2q setup`; it does not
run an interactive wizard inside the `curl | bash` pipeline.

`curl | bash` necessarily trusts the initial HTTPS name resolution and
response before the script can verify anything. The guide states that
bootstrap limitation and also provides a high-assurance, version-pinned flow:
download the installer and release assets, inspect the script, verify the
GitHub attestation and published digests, then execute it locally.

`hf2q uninstall` is receipt-driven. It removes hf2q-owned version directories,
the hf2q entry points it created, update state, its receipt, and only the exact
PATH stanza it owns. It preserves configuration, downloaded source weights,
converted models, calibration receipts, session snapshots, OpenCode
configuration, Agentic Kit state, and every third-party service by default,
and prints the preserved data location. A separate explicit `--purge-data`
confirmation is required to remove model and session data.

The immutable activation receipt is never repurposed as mutable uninstall
state. Before the first removal, uninstall durably creates the separate bounded
`~/.hf2q/uninstall/uninstall-state.json` journal with a closed schema: package,
schema version, installation ID, starting activation sequence, purge-data
choice, and a phase enum. It contains no arbitrary paths. Each phase is written
with private permissions, file sync, atomic same-directory rename, and parent
directory sync before the next schema-derived removal begins. Resume requires
the same live ownership evidence and installation identity; mismatch refuses
mutation. The journal is removed only after the entry point, `current`, owned
installation/update files, and owned PATH stanza reach the recorded terminal
state. Default uninstall keeps the state root and user data, so this journal
remains available across interruption. Purge-data is a subsequent explicit
data transaction and cannot weaken the installation-uninstall recovery rule.
The data-preserving default retains the root identity as the ownership anchor
for preserved state; a full purge may remove it only as the final
identity-bound step immediately before removing an otherwise empty state root.

### 3. Update the whole managed installation atomically

The CLI adds:

```text
hf2q update --check
hf2q update
hf2q update --version X.Y.Z
hf2q update --rollback
```

The default channel is stable. Update metadata uses a TUF-style trust model:
the binary embeds a multi-key threshold root, metadata has signed roles and
expiry, versions are monotonic, and the target binds the exact archive and
release-manifest hashes. The eventual v1 wire contract must support root
rotation through a threshold-signed rollover chain; reinstalling through a
newly verified installer is the documented recovery of last resort. This must
resist rollback, freeze, mix-and-match, and replayed-metadata attacks. Plain
unsigned `latest` JSON is not an acceptable updater authority. The completed
Rust 1.88 comparison and hostile-metadata corpus selected only `sigstore-tuf`
0.11.0's transport-free `TrustedMetadataSet` behind hf2q-owned bounds and
floors; it did not select the stock updater or store.

Top-level targets independently bind the stable channel pointer, versioned
external release manifest, and versioned archive by target name, byte length,
and SHA-256. The signed pointer names the exact manifest and archive targets
and repeats their length/digest descriptors; disagreement with the enclosing
targets metadata is a hard failure. V1 uses top-level targets only and does not
use delegations. The manifest never contains the archive digest because an
identical embedded manifest would make that relationship self-referential.

The initial 2026-08-17 Rust 1.88 source comparison found no production-ready
whole-client selection. It used the exact crates.io archives for
`sigstore-tuf` 0.11.0
(SHA-256
`eedac50883a917b7b434db22e2e6e853ace8c00f4a9c27f53e1e9c87e6d89fe4`,
upstream commit `ef17cacdbd357befea4c1c768ef02ed9bf52672c`) and `tough`
0.24.0 (SHA-256
`35b378d98765c2ae9cdc3e9963ea7e670da8cdd9ee39611b8d722083c7f1ac11`,
upstream commit `98d8eb8b2ce63515d9b4981c938ef6453c5b5771`). Both compile with
Rust 1.88; no unpublished revision is an acceptable landing dependency.

[`sigstore-tuf` 0.11.0](https://docs.rs/sigstore-tuf/0.11.0/sigstore_tuf/)
declares Rust 1.70, exposes a promising I/O-independent
`TrustedMetadataSet`, and its release tree runs the official TUF conformance
suite pinned at v2.4.0 with an empty expected-failure list. That run predates
the conformance suite's later
[multi-star path cases](https://github.com/theupdateframework/tuf-conformance/pull/388),
so it does not prove the post-release delegation behavior. Its stock restart
behavior fails hf2q's persistent rollback-floor contract, however. A temporary
external regression used a generated P-256 key and a fixed initial time of
`2026-06-01T00:00:00Z`, seeded the metadata store with correctly signed
timestamp version 2 expiring `2026-06-15T00:00:00Z`, then restarted at
`2026-07-01T00:00:00Z` from the same pinned root while the repository offered
a correctly signed version 1 expiring in 2999. The refresh succeeded and
trusted version 1. The test patch remained outside the hf2q landing diff; the
exact command against the published source plus that regression was:

```text
cargo +1.88.0 test --locked -p sigstore-tuf --test end_to_end \
  expired_cached_timestamp_still_blocks_restart_rollback -- --exact --nocapture
```

This is an expected-failing regression: it exited 101 because `refresh`
returned `Ok(())` instead of the required rollback error.

The root cause is `Updater::seed_lower_from_store(now)`: expired cached lower
roles are silently skipped before the network refresh, so their versions stop
being floors. `FileStore` also provides neither a cross-role transaction nor
the file and parent-directory sync barriers required here. This rejects the
published stock `Updater` plus `FileStore` as update authority; it does not yet
reject using only the transport-free verifier behind hf2q-owned durable state.
The published crate already used `literal_separator(true)` for delegated
patterns; the later
[`sigstore-tuf` wildcard hardening](https://github.com/sigstore/sigstore-rust/pull/174)
fixes additional multi-star and segment semantics and is not present in 0.11.0.
Another unpublished change re-verifies cached delegated roles. V1 therefore
rejects any delegations before target resolution rather than depending on
either unpublished fix.

[`tough` 0.24.0](https://docs.rs/tough/0.24.0/tough/) retains an expired prior
timestamp or snapshot as a rollback floor, but its private datastore rewrites
each JSON file independently with `tokio::fs::write`; root replacement removes
then recreates `root.json`, and no file sync, parent-directory sync, or
cross-role commit exists. Its public `Transport` can capture the exact bytes
of each response keyed by the requested metadata URL. After a successful
top-level-only load, an adapter may associate captured response bytes with the
successful verification only after parsing them and requiring equality with
the corresponding public `Repository` role getter; each captured root must
additionally form the exact gapless N+1 chain. This makes `tough` the mature
comparator only with an isolated per-attempt scratch datastore. Because
`load_root` currently treats any error opening the next root as the end of the
chain, the adapter must record that probe result and reject a successful load
unless termination was an actual not-found response. Because
`RepositoryLoader::load` eagerly fetches delegated metadata, the attempt
transport must reject every metadata request after the one expected top-level
targets response, and the returned top-level role must have no `delegations`
block.

The landed comparative spike under `tests/adr045_tuf_spike/` keeps both stock
stores out of the authority path. It feeds the last committed raw root and
metadata generation into isolated verifier attempts, captures exact fetched
bytes, and—when using `tough`—correlates them by requested name and complete
parsed-role equality with the successful library result. Application code
never substitutes its own serde reserialization for fetched metadata bytes.
Both verifier paths reject delegations and compare every returned role version
plus an hf2q-owned clock sample against a locked durable generation journal.
The `tough` wrapper additionally brackets its separately sampled internal
expiry time and accepts root-chain termination only after an actual not-found
response. Those experimental floors never decrease; the production adapter
adds the TUF 1.0.36 root-authorized timestamp/snapshot recovery exception
described below rather than inheriting the spike's over-strict behavior.

The Rust 1.88 locked corpus proves rollback, expiry/freeze, same-version byte
replay, mix-and-match, duplicate/oversized metadata, wrong-role signatures,
old/new multi-key thresholds, sequential single- and multi-root rotation,
cross-channel binding, crash-at-every-write, selector replay, namespace swaps,
and cross-process concurrency. It composes application binding with the actual
production `ReleaseManifestV1` parser, all three top-level target descriptors,
the byte-identical embedded manifest, streamed archive digest/length, and an
exact regular-file inventory including full mode bits. The retained
normalization corpus is generated deterministically, regenerated byte-for-byte
in two independent directories, and pinned by SHA-256. The gate passed with 63
tests, zero failures, one explicitly ignored regeneration utility, strict
clippy, a clean audit of 177 locked dependencies, and package-exclusion proof.

This evidence selects only `sigstore-tuf` 0.11.0's transport-free
`TrustedMetadataSet` as the production verification engine. Its stock
`Updater`, transport, and `FileStore` remain rejected. hf2q owns transport,
request sequencing, exact-byte capture, byte/role limits, the clock sample,
version and digest floors, the narrowly root-authorized online-role reset, the
durable generation transaction, and application target binding. `tough`
0.24.0 remains a dev-only independent
comparator over the same hostile corpus; neither its transport policy nor its
datastore becomes production authority. V1 continues to reject all
delegations. The spike's journal and pointer wire representations are
disposable evidence rather than compatibility promises; the production
adapter freezes its schemas only after reusing these invariants in the main
bounded distribution context.

The main crate now contains that dormant transport-free adapter under
`src/distribution/update_auth/`. The exact normal dependency is
`sigstore-tuf = 0.11.0` with default features disabled. `TrustedMetadataSet`
is the only library verification state machine that hf2q imports or uses; the
stock `Updater`, `FileStore`, and `Repository` APIs are not imported or used,
and the dependency's fetch/HTTP/TLS features are disabled. No URL policy,
transport, or metadata store participates in this authority path. One-use
request tokens derive the only accepted next role and wire name. Responses are
bounded before parsing, reject duplicate/trailing or over-depth JSON, and
preserve the exact authenticated bytes. The v1 profile
requires positive versions, canonical expiry strings, exact lowercase SHA-256
and length parent pins, timestamp/snapshot singleton metadata, no delegations,
and a maximum of 256 lifetime root rotations. Every root key-map ID, role-
binding key ID, and envelope-signature key ID is exactly 64 lowercase
hexadecimal characters. Each root key-map ID must equal SHA-256 of the
canonical core key object, and that object is exactly Ed25519/Ed25519 with a
64-character lowercase raw public key and no extension fields. This is hf2q's
closed v1 POUF: it rejects aliased IDs, mismatched keytype/scheme pairs,
case-folded or whitespace-normalized public keys, and producer extensions even
when the pinned library could otherwise verify their signatures. Root-chain
termination advances only on an explicit not-found response; any other
transport outcome remains a failure for the future transport layer.

Authority remains deliberately staged. Structurally valid journal bytes are
not cryptographic authority. A complete transcript becomes a non-cloneable
`VerifiedMetadataCandidate` only after replay from the compiled anchor,
gapless dual-threshold root verification, rollback/equivocation floors, and
freshness checks. An advancing candidate is replayed again against the live
selected floor while the shared installation lock is held. Only the TUF
coordinator can construct its sealed `AdvancingCommitGuard`; the raw journal
has no production advancing-commit surface. The guard parses and caches the
four final role expiries, samples time after lock-held replay and again after
all namespace/staged-byte revalidation immediately before the selector
rename, rejects a backward second sample, and requires every expiry to be
strictly greater than both samples. A selected exact retry instead repairs
historical durability as a rollback floor without current freshness and
returns no target bytes or lookup authority.

TUF 1.0.36 client root-update step 11 (stable tag `v1.0.36`, commit
`59e601ed29c0d2e497264ae8b31c11b8ef07df1e`) requires discarding trusted
timestamp and snapshot state when either online role's keys rotate so a
repository can recover from a fast-forward compromise. hf2q treats the
role's complete effective authorization as that binding: the authorized key
IDs, their exact public-key objects, and threshold. Because step 11 leaves
"rotated" underspecified, hf2q applies the safe recovery predicate at the end
of the gapless root update: fewer byte-identical keys from the prior selected
authorization survive in the final role than the final threshold requires.
Only that quorum invalidation clears both in-memory floors for one transcript.
An additive key, threshold decrease, key ordering, or transient A-to-B-to-A
chain therefore cannot create a rollback window; recovery ceremonies must
actually revoke the prior online quorum. It does not clear the authenticated
root or targets floor, and a root version bump, consistent-snapshot change,
root-key rotation, or targets-key rotation alone does not trigger it. The
sealed candidate records the prior selected root from which this exception
began. Whenever a receipt claims this exception, lock-held and fresh-process
replay must derive the same endpoint predicate from the exact dual-threshold
root chain before treating that receipt as an authenticated floor.

Preserving the authenticated targets floor is an intentional boundary. A
snapshot-only compromise can fast-forward a targets-version *claim* but cannot
produce authenticated targets bytes, so clearing timestamp and snapshot state
is sufficient for recovery. If targets authority itself authenticates and
commits a maliciously high targets version, ordinary online-role recovery must
not lower it: operators must rotate the targets keys and publish actual targets
metadata above the retained floor, or use the separately authenticated
fresh-installer recovery path. The production key-custody runbook must make
that consequence explicit before public self-update ships.

After commit, the coordinator authenticates the exact selected bytes while
the lock remains held, releases the lock, performs an ordinary fail-closed
reopen, authenticates the bytes again from the compiled anchor, and requires
both durable proofs to match. `DurableMetadataBaseline` exposes only the
generation sequence and receipt digest. It is not target lookup, archive,
prepared-version, activation, or update authority.

The dormant application-target boundary now freezes `ChannelPointerV1`. Its
deterministic producer encoding is compact JSON plus one LF and contains kind
`hf2q.update-channel-pointer`, schema version 1, package and logical repository
ID `hf2q`, stable channel, canonical stable SemVer, target
`aarch64-apple-darwin`, and exact manifest/archive descriptors. Each descriptor
contains only the canonical logical name, nonzero byte length, and lowercase
SHA-256. Parsing is structural, bounded to 16 KiB, hostile-input safe, and
never grants update authority. The manifest descriptor is capped at 1 MiB.
The compressed tool-only release archive is separately capped at 512 MiB;
the manifest's 4 GiB expanded-payload ceiling remains a distinct zip-bomb and
inventory bound. Model weights are never release-bundle members.

Logical target names are generated from typed values, never accepted as
equivalent caller paths or URLs:

- `channels/stable/aarch64-apple-darwin.json`;
- `releases/vV/aarch64-apple-darwin/release-manifest.json`; and
- `releases/vV/aarch64-apple-darwin/hf2q-vV-aarch64-apple-darwin.zip`.

The stable repository profile requires authenticated
`consistent_snapshot=true`. Signed targets and the pointer always carry the
unprefixed logical name. The physical object name preserves its parent
directories and prefixes only the basename with the full SHA-256 from the
authenticated TUF target descriptor. A repeated digest inside the pointer
never chooses that prefix. The future transport maps that exact typed physical
name to an origin; neither the pointer nor TUF custom metadata contains a URL.
GitHub Pages retains the full nested physical path for the channel pointer.
GitHub Release assets are flat, so the manifest/archive origin mapper must use
only the route-validated physical basename (including its authenticated digest
prefix) while retaining the canonical logical path as the signed identity.

Top-level targets remain bounded by the existing 4 MiB and 4,096-entry limits.
The application profile accepts exactly one canonical stable pointer plus one
or more complete canonical `(release-manifest, archive)` pairs for retained
versions. It rejects orphan members, unrelated or hash-prefixed logical names,
delegations, target custom/extra fields, zero/over-limit lengths, and anything
other than one lowercase SHA-256. Retained older pairs are inert. Pointer
binding consumes the fresh set and exposes exactly three role-specific
descriptors: the pointer, its selected manifest, and its selected archive.
There is no arbitrary target lookup surface.

After the first selected generation enters this stable application profile,
versioned release pairs form an append-only semantic floor across every
successor `targets.json`. While the shared installation lock is held, hf2q
replays the selected predecessor and candidate from the compiled anchor, then
requires every prior canonical manifest/archive name, length, and SHA-256 to
remain byte-identical. The pointer may move and new complete pairs may be
appended; an old member may never be changed, renamed, or removed. This
pairwise check occurs before the selector commit and is transitive, so normal
postcommit predecessor cleanup loses no release-identity floor and requires no
duplicate receipt field. Timestamp/snapshot floor recovery does not reset this
release floor.

`AuthenticatedTargetSet` is non-cloneable and can be created only by an
ordinary fail-closed reread of the selected v2 journal followed by replay from
the compiled anchor. It samples current time before and after replay, rejects
clock rollback, and requires root, timestamp, snapshot, and targets expiry to
be strictly later than the second sample. It binds installation/state identity,
selected sequence and receipt digest, all four metadata versions, the final
consistent-snapshot policy, and earliest expiry. Exact pointer bytes must then
match the authenticated pointer descriptor before `ChannelPointerV1` is parsed;
its repeated manifest/archive descriptors must exactly equal the selected TUF
pair. The resulting sealed release-target plan still authorizes no URL,
download, filesystem write, archive extraction, prepared version, activation,
or downgrade. A later mutation must reacquire the shared installation lock,
prove the same selected journal identity, and recheck freshness after I/O.

Automatic planning also reads a sealed live installed-release floor while that
same lock is held. The reader follows only the canonical `current` activation,
requires the exact bounded activation inventory, and descriptor-relatively
verifies the canonical receipt, prepared version directory, installed marker,
manifest, and complete payload before it retains the active version, target,
manifest/archive SHA-256 values, installation and activation sequences, and
receipt digest. Two independently reopened snapshots must agree in both bytes
and inode identities. `current` being absent produces an explicit no-active-
release state; prepared, extracted, or otherwise unselected residue never
becomes a downgrade floor or adoption capability.

The authenticated pointer is compared numerically against that live floor
before artifact transport authority exists. A lower SemVer is rollback, an
equal version with either release digest changed is equivocation, an exact
equal version/digest tuple returns a sealed `AlreadyCurrent` outcome with no
download authority, and only a higher version (or the explicit no-active-
release state) can produce the one-use fetch capability. The exact floor is
read again before archive staging, after external I/O, and when the retained-
lock preparation session begins. Any change in active release, receipt, or
activation identity invalidates the plan. An intentional pinned downgrade or
rollback remains a separate future one-use user-intent capability; it cannot
reuse this automatic path.

The v1 network routes are closed policy, not runtime configuration. Production
accepts no caller-supplied base URL, mirror, path, redirect host, proxy, or
HTTP client as update authority. Typed verifier requests and authenticated
target descriptors map to exactly these HTTPS locations:

- metadata: `https://robertelee78.github.io/hf2q/updates/stable/metadata/R`,
  where `R` is the verifier-issued relative name;
- the stable channel pointer:
  `https://robertelee78.github.io/hf2q/updates/stable/targets/P`, where `P`
  is its full authenticated consistent-snapshot physical path; and
- manifest/archive assets:
  `https://github.com/robertelee78/hf2q/releases/download/vV/B`, where `V`
  is the authenticated release version and `B` is the route-validated,
  digest-prefixed physical basename.

The dormant transport implements all three routes. For metadata, a fresh
session first lock-authenticates the durable selected floor and crash-durably
discards only an exact never-selected write prefix. It then consumes the
transport-free verifier's outstanding request token exactly once. The route
accepts a direct 200 with a nonempty body no larger than the role cap; a
present `Content-Length` must be nonzero, within the cap, and equal the bytes
read. Every body is independently read through cap plus one. Only the exact
next-root request may map a direct 404 to `ConfirmedNotFound`; timestamp,
snapshot, and targets 404s fail, as do redirects, transformed content, all
other statuses, malformed headers, and read failures. A completed transcript
still becomes durable only through the existing lock-held current-time
reauthentication, selector-boundary freshness check, commit, reopen, and exact
ordinary-reader replay. The returned commit outcome exposes no target lookup
or artifact authority. No production root factory or public caller exists yet.

Pages requests never redirect. A release-asset request accepts either an
immediate 200 response or exactly one 302 from the exact `github.com` route to
HTTPS host `release-assets.githubusercontent.com`, followed by a final 200.
The redirect may carry GitHub's opaque expiring query, but it may not contain
userinfo, a fragment, a non-HTTPS scheme, an IP literal, or a nondefault port.
A second redirect, hostname suffix/lookalike, `/latest`, REST/API discovery,
`hf2q.us`, or pointer/custom-metadata URL fails closed. Redirect URLs and
queries are never logged.

The dedicated client explicitly selects rustls, HTTPS-only requests, no
referer, no automatic redirect, no automatic retry, and identity content
encoding. Connects are bounded to 10 seconds; metadata, pointer, and manifest
requests are bounded to 60 seconds; an archive is bounded to 30 minutes. A
non-identity `Content-Encoding` fails. Standard HTTPS/system proxy settings
are honored for corporate usability, but there is no runtime proxy override
in this authority surface and a proxy remains hostile availability-only
transport. Exact origins, WebPKI TLS, signed bytes, and the post-I/O TUF replay
remain authoritative.

`Content-Type`, `ETag`, `Age`, `Last-Modified`, CDN cache headers, and
GitHub's release-API digest are diagnostic or publisher corroboration only;
signed length and SHA-256 remain the client authority. A present
`Content-Length` must equal the authenticated length, and every body is
independently read only through expected length plus one byte. Status 206/range
resume is not part of v1.

Pointer and manifest bytes are bounded in memory. The archive is streamed
through a fixed-size buffer into a private 0600 file created descriptor-
relatively on the authorized state-root device and unlinked immediately after
open. The same descriptor is counted and hashed while writing, synced, rewound,
and re-read for exact length/SHA-256 before it can become a non-cloneable
staged-file proof. The file-owning wrapper exposes only `Read`/`Seek`, uses
redacted debug output, and can repeat the same-FD identity/length/hash check
immediately before its later consumer. A crash between durable create and
unlink may leave at most
one exact empty UUIDv4 residue; the next shared-lock holder removes it only
after a bounded complete-inventory and identity check. V1 restarts a failed
archive download from the canonical GitHub URL. `ENOSPC`/`EDQUOT`,
short/long bodies, digest mismatch, timeout, read failure, write failure, and
sync failure yield no artifact or install capability.

Fetching occurs outside the installation lock. Before the large archive and
again after all I/O, the coordinator reacquires the shared installation lock,
replays the current selected journal from the compiled anchor at current time,
and requires the same installation/state identity, selected sequence and
receipt digest, pointer/manifest/archive descriptors, live installed-release
floor, and fresh role expiries. The same floor is checked once more when the
lock-retaining preparation session starts.
Any change discards the inert staged bytes and restarts planning. The fetched
result alone still grants no extraction, codesign verification,
prepared-version, activation, or update authority.

TUF metadata versions do not prevent newly signed metadata from moving the
stable pointer to an older hf2q SemVer. `ReleaseVersion` therefore uses numeric
SemVer ordering (`0.10.0 > 0.9.0`) and the dormant binding compares only with
the fully verified live floor described above, never a parsed or caller-
supplied receipt. Public update mutation remains blocked on the production
root, Team-ID policy, protected positive fixture, and public coordinator. Any
user-requested pinned downgrade or rollback still
requires a separate one-use intent capability.

Large archives are streamed and checked against the sealed descriptor by hf2q
rather than buffered through either client's convenience target API. The
origin-locked external-byte layer, exact classic-ZIP/embedded-manifest
validator, private descriptor-relative extraction, descriptor-bound native
verification, crash-resumable signed-mode normalization, and crash-durable
prepared-version publication are implemented but remain deliberately dormant.
The real compiled Team-ID policy and protected positive signing fixture remain
pending. Notarization is
publisher promotion evidence for the exact archive, not runtime preparation
authority. The implemented metadata layer
already preserves the distinction between ordinary read authority and
lock-held recovery: any partial or published-but-unselected
successor is ambiguous and fails closed for an ordinary reader. A same-process
retry may resume exact staged bytes only while it still owns the sealed live
candidate, including the explicit root-chain not-found proof that is not
serialized in the journal. A fresh process never promotes residue: it
historically authenticates and repairs the selected rollback floor, removes
only the structurally exact never-selected transaction under the shared lock,
then requires a wholly fresh transcript.

The production v2 local metadata journal is frozen independently of the
network verifier. It remains crate-private and unreachable from command
dispatch. The earlier pre-publication v1 experiment had no floor-reset record
and was never reachable from a released CLI, installer, network transport, or
production entry point. Rather than silently redefining those canonical v1
bytes when root-authorized online-role recovery was added, this ADR retracts
that dormant experiment and makes v2 the first public-state candidate. The
reader rejects the retained exact v1 golden bytes; there is deliberately no
migration authority for state that no shipped writer could create. Once a
public writer exists, this pre-publication rebaseline exception no longer
applies: every later wire change requires a dual-reader migration release
before a new writer may select it. The canonical v2 generation receipt is
compact JSON followed by one LF and contains:

- kind `hf2q.update-metadata-generation`, schema version 2, state-layout
  schema 1, and package `hf2q`;
- the nonzero 64-bit generation sequence and, except for sequence one, the
  exact SHA-256 of the predecessor generation receipt;
- the installation UUID and canonical state root, preventing cross-root
  journal substitution;
- logical repository ID `hf2q` and channel `stable`; transport origins and a
  future branded mirror remain repository policy, not durable trust identity;
- exact canonical RFC 3339 verification start and completion instants with
  subsecond precision; a successor start cannot precede prior completion, and
  completion cannot precede start; and
- exact request name, positive role version, byte length, and lowercase
  SHA-256 descriptors for the embedded anchor root, complete gapless root
  history, final trusted root, timestamp, snapshot, and top-level targets; and
- an optional timestamp/snapshot floor-reset record containing exact prior and
  final trusted-root descriptors. It is valid only on a successor whose root
  history extends the predecessor, whose prior descriptor equals the selected
  root, and whose exact authenticated endpoint roots prove that the prior
  timestamp or snapshot quorum cannot satisfy the final threshold.

The selector has kind `hf2q.update-metadata-selector`, schema version 2,
sequence, and the exact generation-receipt SHA-256. Receipts are limited to
64 KiB, selectors to 16 KiB, roots/timestamp/snapshot to 1 MiB each, and
targets to 4 MiB. Hostile input rejects unknown or duplicate fields, trailing
documents, noncanonical encoding or timestamps, zero versions, version/name
disagreement, wrong repository/channel/root identity, invalid digests, and
rollback or same-version byte equivocation outside the authenticated
timestamp/snapshot recovery exception. Root and targets floors remain
monotonic through that exception. The complete gapless root history
starts at the compiled-in anchor and is capped at 256 lifetime rotations.
That exceptional root-rotation recovery bound is not an update-count limit;
ordinary metadata generations use a checked 64-bit sequence and do not stop
after an arbitrary number of updates.

All journal directories are `0700`; regular files are `0600` and single-link.
Both are owned by the effective user, opened no-follow, and on the state-root
device. A generation is written or exact-prefix-resumed below
`generations/.pending-N`, each file is synced, the exact inventory is verified,
and it is published by no-replace rename. The staged `.current-N.json` is then
synced and rebound to its named inode. Replacing `current.json` is the sole
successor commit point; the initial selector uses no-replace. Precommit checks
reopen the authorized root, `update`, `metadata`, `generations`, published
generation, lock, and staged selector, rejecting stale descriptors or a
changed namespace. Postcommit checks reopen and reverify the selection and
repeat file, generation, parent, update, root, and `F_FULLFSYNC` barriers. A
postcommit failure is reported as
`CommittedDurabilityUnknown { sequence }`. Every subsequent lock-held
transition must repair that selected generation's durability before cleanup or
new staging; an exact retry then returns `AlreadyCommitted`.

The journal remains bounded without the spike's disposable 1,024-update
exhaustion rule. During a transition it permits only the selected generation,
one exact successor transaction, and one exact predecessor cleanup residue.
An ordinary unlocked reader rejects a partial or published-but-unselected
successor. A live same-process sealed candidate may exact-resume it, but a
fresh process has no authority to infer the missing transport termination
proof from disk. Fresh-process recovery first replays any selected generation
from the compiled anchor at its historical completion time, repeats its
postcommit barriers, and completes prior cleanup. It then classifies only the
derived next sequence. A published successor must pass its canonical receipt,
state identity, predecessor, exact file inventory, role descriptor, and staged
selector binding before it can be renamed no-replace back to `.pending-N`. A
pending generation must be an exact bounded prefix of the reserved write
order: root-history directory, anchor, trusted root, timestamp, snapshot,
targets, then receipt. Corrupt, non-prefix, symlinked, hard-linked,
wrong-mode, oversized, or namespace-swapped residue is preserved fail-closed.
No stored successor bytes become a verifier candidate.

Authorized discard removes only `.current-N.json`, `N`, and `.pending-N`
derived from the selected sequence. Fixed entries are removed in reverse
creation order, each unlink/rmdir is followed by its containing-directory
sync, and every crash leaves the same recognizable prefix. Discard then syncs
the generations, metadata, update, and state-root directories and ends at
`F_FULLFSYNC` on the exact selected selector, or the held lock file when no
selection exists. The next network attempt starts from a wholly fresh TUF
transcript and may reuse sequence N. No generic path, recursive deletion,
quarantine, or candidate-construction API is granted.

After a successor is selected, its independently verifiable complete root
history and role floors allow the old generation to be renamed no-replace to
the one derived
`.prune-(N-1)` name and removed through an exact, descriptor-relative,
no-follow inventory. A crash before or during cleanup leaves the newer
selection authoritative and a bounded, receipt-bound cleanup residue; the
next lock-held transition completes it before staging anything new. Every
entry removal is followed by its containing-directory sync; cleanup then syncs
the generation, metadata, update, and state-root directories and ends at a
reopened, byte-exact selected-selector `F_FULLFSYNC` before later selection may
advance. No generic path or recursive deletion API is granted, and corrupt
selected state never falls back to an older floor.

Acceptance proof pins literal v1 receipt and selector bytes, exercises bounded
hostile parsing and namespace/inode/attribute swaps, and runs every initial,
successor, and postcommit barrier through both returned-error recovery and real
child-process abort recovery. The single-rotation matrix covers every cleanup
barrier, while an independent nine-root history covers every root-file removal,
the root-history-directory boundary, every role and receipt removal, and exact
retry through both failure modes. The process matrix requires an actual
`SIGABRT`, and a separately corrupted surviving root-history suffix fails closed
without deleting its diagnostic residue. Cross-process tests prove that
activation and metadata paths exclude one another through the shared lock. A
blocking test also commits 1,025 successive stable-root generations, retains
one selected generation, and demonstrates that the discarded spike's 1,024
limit is not present in the production journal; schema proof separately accepts
exactly 256 lifetime root rotations and rejects 257.

Production-verifier proof adds strict-expiry equality, backward-clock,
rollback/equivocation, old/new threshold, request-name, root-limit,
duplicate/trailing/over-depth JSON, wrong-role, delegation, and mixed-parent
adversarial cases. Advancing commits are rejected both before staging and at
the selector-boundary guard. The journal-layer fresh-process discard runs every
removal/sync barrier through returned-error and real `SIGABRT` recovery for
both an empty journal and selected-N plus unselected-N+1, proving that the
selected selector and receipt remain byte-exact. The TUF coordinator separately
proves the full composition: historically authenticate selected N, discard an
unselected N+1, reopen the same durable floor, and commit a wholly fresh N+1
transcript. A selected floor also accepts a later dual-threshold root rotation,
commits the newly signed lower roles, and rejects a version rollback against
that new durable floor. A separate fast-forward recovery proof commits high
timestamp/snapshot floors, rotates those online bindings through an
offline-authorized root, accepts lower recovered timestamp/snapshot roles,
preserves the targets floor, reopens the auditable receipt, and rejects a
second rollback without another qualifying rotation. Root-only, targets-only,
additive, surviving-threshold, and transient A-to-B-to-A transitions cannot
mint the exception. Truncated write prefixes are discardable while hostile
shapes are preserved. An independently generated Python-TUF 7.0.0
corpus uses canonical key IDs, a complete old/new two-of-two root rotation,
two-of-two lower roles, consistent-snapshot wire names, exact parent pins, a
fully hashed dependency lock, and retained provenance/checksums. Rust pins the
five metadata digests, proves positive commit/reopen, and derives missing-old,
missing-new, and missing-lower-signature failures from those independent
bytes.

The journal, verifier, dormant application layer, and sibling transport now
compose a fresh, generation-bound selection of the stable pointer, fetch its
exact external manifest/archive bytes from closed origins, and reauthenticate
the same selected generation under the shared lock before and after archive
I/O. The private preparation boundary additionally proves the exact classic-ZIP
layout, canonical embedded manifest, complete payload inventory, modes, sizes,
CRCs, and SHA-256 values on the same archive descriptor, then materializes the
exact bytes into a private inert tree under the retained shared lock. These
boundaries still do not produce update authority. The preparation boundary now
brackets the same descriptor-bound binary with Mach-O and native Developer ID
verification, consumes the first ephemeral proof to normalize exact signed
modes, repeats current-time TUF replay, and requires the same path/inode to pass
the native verifier again. It then creates the exact marker/receipt, commits the
prepared version by no-replace rename, repeats durability and current-time
authentication, and only then constructs an authenticated prepared version.
The next production-enablement slice must embed the real stable trust root,
compile the real public Team ID, and pass the protected positive Developer ID
fixture.
Neither a receipt, parsed role, provisional candidate, durable baseline, nor
selected target plan can download bytes or mutate an installation by itself.

The initial stable metadata repository is served from GitHub Pages under the
hf2q repository (for example,
`https://robertelee78.github.io/hf2q/updates/stable/`). A branded route may
redirect to or mirror those bytes, but signed metadata—not either transport—is
the update authority. GitHub Pages enablement, key separation, offline-root
handling, rotation, expiry refresh, and recovery are release prerequisites;
the unsigned GitHub Releases `latest` response is only installer discovery.
The key-generation, offline-root, online-role, rotation, revocation, expiry,
and disaster-recovery runbook must exist and be exercised before self-update is
made public; these choices are not deferred until after the first trust root is
embedded.

For a receipt-owned standalone install, update downloads and verifies the
complete bundle, installs a new version directory, and atomically changes the
active version. It never patches the running binary in place. A concurrent
installer/updater is rejected by an installation-scoped lock. Rollback may
select only a retained, previously verified release and records the change.

`hf2q update` must work as the single front door regardless of installation
method; "universal" does not mean overwriting across ownership boundaries. For
Homebrew or Cargo-registry ownership, hf2q invokes the selected `brew`,
`cargo install`, or `cargo binstall` route, propagates its result, and verifies
the resulting hf2q version. For unknown/manual ownership, it explains the
ambiguity and offers an explicitly confirmed migration to the managed
standalone layout; it never silently overwrites the executable. Package
ownership determines how an update is performed, not whether the user can
invoke `hf2q update`.

Ownership resolution is evidence-based and ordered: a live standalone layout
whose entry point, raw `current` target, version marker, manifest, and receipt
all verify within the opened recorded root; a Homebrew formula query
whose Cellar path owns the running executable; then Cargo's installed-package
registry plus its configured install root. Location alone is never sufficient.
Cargo's installed-package records are an internal, unstabilized part of
[Cargo home](https://doc.rust-lang.org/cargo/guide/cargo-home.html), and
[cargo-binstall](https://github.com/cargo-bins/cargo-binstall) describes itself
as a drop-in `cargo install` replacement. They do not provide a supported
historical-installer distinction, so hf2q records a route preference: an
existing valid preference wins; otherwise an available cargo-binstall in the
same Cargo home is selected and plain Cargo is the fallback. Ambiguous or
contradictory evidence becomes `unknown/manual` and cannot authorize an
overwrite.

The receipt's owner and route are only the last observation. Every mutation
reruns the ordered live evidence while holding the nonblocking installation
lock. The valid route matrix is `standalone -> standalone`,
`homebrew -> brew`, `cargo-registry -> cargo-install|cargo-binstall`, and
`unknown/manual -> absent`. Manager/manual receipts have no retained releases;
their active release-manifest identity may be absent when that package channel
does not install an hf2q release bundle. The shared state root and the
owner-controlled installation root are distinct fields; they are equal for a
standalone install and may differ for manager/manual ownership.

Automatic `--rollback` is a standalone-only operation because only the
standalone layout owns retained versions. Under Homebrew or Cargo-registry
ownership it fails closed without changing the executable and prints the
tested owner-specific version/downgrade command when that manager supports one,
or the manager's recovery instructions when it does not. `--version` likewise
delegates an exact version only when the detected manager can express it;
otherwise it fails without mutation. Unknown/manual ownership may migrate only
after explicit confirmation.

### 4. Check for updates on every launch with bounded refresh cost

Startup evaluates a trusted local metadata cache before argument parsing, but
does not emit the resulting notice until output format is known. The CLI uses a
non-exiting parser path so a cached notice is emitted before command dispatch
or before returning help, version, or parse-error output. A syntactically valid
`--log-format json` selects a structured notice even for help/version exits;
otherwise parser failures use one text stderr line. Stdout and
machine-readable command output remain untouched.

When the cache is stale, at most one process attempts a refresh, no more than
once per 24 hours. That elected stale-cache invocation performs one foreground
refresh with a 500 ms connection timeout and 1,500 ms total timeout. This
explicit bounded delay, at most once per 24 hours for the shared state root, is
the accepted cost of allowing the current invocation to report newly verified
metadata; all other launches use only local state. Network, DNS, clock,
signature, lock, or parse failure is silent at normal verbosity and never fails
the requested hf2q command or exceeds those bounds. Invalid metadata is not
cached.

In text log mode, the notice is a single stderr line with installed/new
versions and the correct owner-aware update instruction. Under
`--log-format json`, it is one valid structured log event so the existing
one-JSON-object-per-line contract is preserved. It never downloads or installs
automatically, and notice emission is not suppressed merely because stderr is
piped or non-interactive. `HF2Q_NO_UPDATE_CHECK=1` disables automatic network
refresh and notices but never disables an explicit `hf2q update`; it is
classified in `docs/operator-env-vars.md` and the shipping contract.
Metadata and its single-flight lock live under the configured hf2q state root,
`~/.hf2q/update/` by default, regardless of executable owner. Reading cached
state creates nothing; only an elected stale refresh or explicit update may
create that private directory. The request uses a fixed generic User-Agent and
carries no installation identifier, exact installed version, telemetry, model
inventory, or command name. "Every launch checks" means every launch evaluates
the verified local state; it does not mean an unconditional network request on
every launch.

### 5. Configure hf2q with one small setup command

`hf2q setup` is an idempotent hf2q-only host configuration step. It inventories
Apple Silicon generation/GPU shape, unified memory, filesystem capacity and
free space, active shell, existing hf2q configuration, and safe system limits,
then writes `~/.hf2q/config.toml` atomically. It downloads no model, performs no
conversion, starts no server, installs no integration, and does not edit
OpenCode.

Setup asks one simple session-persistence question:

```text
Keep inactive sessions on disk for fast resume? [Y/n]
```

When the user accepts, setup displays a disk-aware recommended byte limit and
lets Enter accept it or an explicit value override it. The recommendation is
the smallest of 100 GiB, 10% of the containing volume, 25% of currently free
space, and the bytes remaining after reserving the greater of 20 GiB or 15% of
the volume. Checked arithmetic floors a negative result to zero; when no safe
positive band remains, setup refuses to enable persistence. The answer and
exact limit are recorded. Non-interactive setup requires either
`--session-cache off` or
`--session-cache on --session-cache-limit <SIZE>` and never guesses consent.
For the stored policy, zero means persistence is disabled; it never means
unlimited, and `--session-cache on` rejects a zero limit. Re-running setup
merges the current configuration without duplicate keys or stale managed
fragments.

Setup records a hardware profile used to rank accepted model candidates, but
does not pretend to calibrate inference before a model exists. Exact runtime
calibration occurs only after conversion or a requested published GGUF has
been acquired.

### 6. Treat Hugging Face identities as the public model interface

One resolver accepts and normalizes:

```text
Qwen/Qwen3.8-27B
https://huggingface.co/Qwen/Qwen3.8-27B
https://huggingface.co/Qwen/Qwen3.8-27B/tree/<revision>
https://huggingface.co/<owner>/<repo>/blob/<revision>/<file.gguf>
https://huggingface.co/<owner>/<repo>/resolve/<revision>/<file.gguf>
```

A URL-embedded revision and `--revision` must agree or resolution fails. A
branch or tag is resolved to an immutable commit before transfer. Receipts keep
the user's original string and the normalized repository ID/type, canonical
URL, exact revision, selected files, byte sizes, and digests. Public repositories
need no credential; private/gated repositories use standard Hugging Face token
discovery without copying tokens into hf2q configuration. Local paths remain
supported explicitly.

The guide uses `Qwen/Qwen3.8-27B` everywhere. A private local alias is not part
of the documented workflow and never governs receipts, cache identity,
diagnostics, or `/v1/models` identity.

**Landed boundary (2026-08-19).** The positional model ID/URL grammar and the
`--repo` compatibility spelling now converge on one bounded parser. It accepts
the forms above, rejects credentials, alternate origins/ports, query/fragment,
ambiguous routes, traversal, malformed percent escapes, mismatched revisions,
and over-cap repository/revision/file components. File-specific `blob` and
`resolve` forms are structural identities only; repository conversion rejects
them until the separately recipe-bound external-GGUF path lands. Existing,
absolute, `./`, and `../` paths remain explicitly local.

The in-process `hf-hub` path pins the official endpoint, resolves repository
information to a 40-hex commit before selected transfer, and requires each
file's metadata to name that same commit. The complete repository name
inventory is capped at 4,096; selected paths at 1,024 bytes/64 components;
small metadata and the safetensors index at 16 MiB; tokenizer/vocabulary files
at 512 MiB; and the index at 262,144 tensor entries. The authenticated index
rejects duplicate/unknown structure and selects only required safetensors;
unrelated weights, `.bin`, ONNX, and pre-quantized outputs remain inert.
Safetensors require an LFS SHA-256. Git-managed metadata is checked through its
canonical Git blob SHA-1 and every selected local file receives a SHA-256 in
the schema-v3 conversion receipt. The receipt additionally binds the original
reference, normalized repository ID/type, canonical URL, immutable revision,
source bundle, converter commit, quant selector, and final GGUF identity.
Standard Hub token/cache discovery is used without persisting a token in hf2q.

### 7. Make official-source preparation native, explicit, and device-aware

The canonical official-source preparation command is:

```sh
hf2q convert Qwen/Qwen3.8-27B
```

That no-options command remains the target contract, not a claim about the
current CLI. The landed immutable-source slice still requires explicit
`--quant` and `--output`; it does not guess either. The checked-in recipe now
provides the exact source/artifact/hardware/disk decision input, but the next
preparation slice must consume that sealed evidence in one source-matched
text/projector conversion, retention, registration, prepared-profile, and
bounded-runtime-calibration transaction before removing those requirements.

**Landed recipe boundary (2026-08-19).**
`data/model-recipes/qwen38-27b-official-v1.json` is a canonical compact JSON
document plus one LF, capped at 64 KiB and embedded into the binary. Its exact
SHA-256 is
`47a4cec7eb3b19ad68727f557ff47e83f1ef88c791734a76b5bd052d921c9d9d`.
The v1 parser denies unknown, duplicate, trailing, over-depth, noncanonical,
and over-cap input. It admits only the official `Qwen/Qwen3.8-27B` model at
immutable revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, with 29 sorted
basename-only source records. Each record binds exact bytes, SHA-256, Hub
ETag, and—where applicable—the identical LFS SHA-256. The LFS inventory also
derives the exact source-bundle digest rather than trusting a repeated total.

The recipe admits exactly the ADR-044 accepted pair:

- `Qwen3.8-27B-Q4_K_M.gguf`, 16,810,714,752 bytes, SHA-256
  `0fa8acc661d0edc60276c43705619fd848682dbf768ced9fe46cd8a572b8043d`;
- `Qwen3.8-27B-mmproj-F16.gguf`, 927,606,848 bytes, SHA-256
  `6fa039b75244c0a28a013da30b92b1d221c61029acc19f9efa882b75a495b0d0`.

Automatic selection is deliberately closed to the one independently proven
profile: `aarch64-apple-darwin`, exact chip model `Apple M5 Max`, at least
128 GiB unified memory, Q4_K_M text plus F16 projector, with runtime
calibration still required. Other hardware fails before preparation; this
slice does not extrapolate an unmeasured profile. The disk floor is derived
from 55,586,101,511 source bytes, 17,738,321,600 output bytes, and an explicit
8 GiB reserve: 81,914,357,703 free bytes. Below that exact floor fails before
large work. Interactive retention defaults to keep; noninteractive deletion
requires a future explicit choice and is limited to recipe-owned source.

`VerifiedRecipeSource` is minted only when the complete immutable Hub manifest
matches the recipe and Git-managed support files also hash locally. A separate
sealed artifact proof requires exact size and SHA-256. Neither proof can be
cloned or used as serving/install authority, and the two roles cannot be
cross-bound. Live metadata proof covers all 29 official files at the exact
revision; the opt-in local gate rehashes the already accepted 17.7 GB output
pair.

**Landed preparation-pair boundary (2026-08-19).** Before either conversion
can enter a pair, `VerifiedRecipeHost` binds the exact target, chip, observed
unified memory, selected recipe profile, and observed free-space floor.
Production construction accepts no caller-supplied host facts: the target is
compile-time fixed, `machdep.cpu.brand_string` and `hw.memsize` are read from
fixed sysctl keys in-process, and free space is read from the mounted
filesystem containing the chosen preparation root's nearest existing
directory. That ancestor is canonicalized before mount selection, so an
operator-chosen model-root symlink measures its actual destination filesystem;
this proof grants no filesystem mutation authority. The probe rejects a file
ancestor and fails closed when the target, sysctl values, mount, or nonzero
free-space value is unavailable. No PATH-resolved subprocess participates. The
future no-options coordinator must choose the preparation root and consume this
sealed proof before authentication or large work. The protected Apple Silicon
proof gate is `HF2Q_TEST_QWEN38_HOST_PREFLIGHT=1 cargo +1.88.0 test --locked
--bin hf2q --all-features input::model_recipe_preparation_tests:: --
--test-threads=1`; ordinary hosted CI still proves the fail-closed and
filesystem-selection branches without pretending its hardware matches the M5
Max profile. Each conversion receipt is independently capped at 64 KiB, parsed
with a closed schema, required to equal its deterministic pretty-JSON encoding
plus LF, and cross-bound to the complete recipe source inventory, accepted
artifact path, size, digest, quantization, role-specific conversion strategy,
zero excluded tensors, and canonical hf2q converter identity. The artifact
proof is consumed when that conversion proof is minted, so a text receipt
cannot be paired with a projector artifact or with another recipe.

Only the sealed source proof, the sealed host proof, and one text plus one
projector conversion from the same converter identity can mint
`VerifiedModelPreparation`. Its canonical 1,334-byte v1 receipt has SHA-256
`5b20ca00d6757b285927e6f464271143e8820b1126ae85bc5786a532374ed69b` and
binds the recipe/source/profile/converter identities, both accepted artifacts,
and both exact conversion-receipt SHA-256 values. Its only state is
`awaiting_runtime_calibration`. Parsing those bytes is structural and cannot
recreate the non-cloneable sealed value; neither form grants serving,
registration, retention deletion, or filesystem-mutation authority.

**Landed no-options preparation-plan boundary (2026-08-19).** A bare official
reference or its exact accepted-revision URL can now mint one non-cloneable
`ModelPreparationPlan`. Construction selects only the embedded recipe, derives
the accepted revision without pretending that Hub resolution has already
occurred, and owns the OS-bound host proof for the exact future model location.
It canonicalizes the absolute models root through existing directory symlinks,
accepts only UTF-8 paths capped at 4,096 bytes, 64 components, and 255 bytes per
component, and rejects dot/parent traversal, file ancestors, over-cap paths,
file-specific Hub references, other repositories, and unaccepted revisions.
Planning creates no directories or files.

The exact planned tree is
`<models-root>/huggingface/Qwen/Qwen3.8-27B/<accepted-revision>/` with
`source/`, `artifacts/`, `receipts/`, and `profile.json`. Artifact names remain
the two recipe names. Their final conversion receipts are named
`<artifact>.receipt.json` under `receipts/`, and the pair receipt is
`receipts/model-preparation.json`. The plan exposes only read-only typed paths
and policy facts; it grants no download, conversion, source deletion,
registration, calibration, serving, or filesystem-mutation authority. A later
coordinator must consume it, resolve Hub metadata to the same exact revision,
and revalidate the namespace before mutation.

**Landed exact-resolution boundary (2026-08-19).** The pinned in-process Hub
client can now resolve the plan's exact original reference without transferring
model payloads. `ResolvedModelRepository` is a private-field, non-cloneable
token containing the normalized immutable 40-hex reference and a bounded,
validated repository-name inventory. The model-recipe module retains no Hub,
HTTP, subprocess, or filesystem-write authority: the download module alone
consumes the host-checked plan, performs the pinned lookup, and passes the
sealed result back to the plan's narrow binder. Binding requires the exact
original input, normalized repository and canonical URL, accepted immutable
revision, no file route, and the presence of all 29 recipe-owned source names.
Unrelated repository entries remain inert and private. The returned
`ResolvedModelPreparationPlan` retains the sealed resolution and planned
layout but still grants no payload transfer, conversion, deletion,
registration, calibration, serving, or filesystem-mutation authority.

**Landed pre-transfer metadata authorization (2026-08-19).** Before a recipe
payload can be requested, the download module queries the pinned Hub metadata
endpoint for each of the 29 recipe names at the already resolved immutable
commit. Request order is the recipe's canonical order. Every response must
name that commit and exactly match the checked-in relative name, byte length,
Git or LFS classification, ETag, and LFS SHA-256 when present. Missing, extra,
reordered, renamed, resized, reclassified, or rehashed records fail before any
payload transfer. The non-cloneable `AuthorizedModelPreparationTransfer`
consumes the resolved plan, canonicalizes the accepted records back to the
checked-in values, keeps the record inventory private, and exposes no generic
file selection. It still grants no payload transfer, conversion,
filesystem-mutation, deletion, registration, calibration, or serving
authority. The opt-in live proof is
`HF2Q_TEST_QWEN38_TRANSFER_AUTH=1 cargo +1.88.0 test --locked --bin hf2q
--all-features input::model_recipe_plan_tests::current_plan_authorizes_all_recipe_metadata_before_payload_transfer -- --exact --test-threads=1`.

The authorized payload transfer, no-options conversion invocation,
source-retention mutation, prepared-model registry, and calibration receipt
remain pending and must consume these proofs rather than reconstructing policy
from strings.

It executes an hf2q-only resumable plan:

1. resolve the checked-in recipe and ADR-044-accepted immutable revision;
2. run host, free-space, memory, credential, and packaged-file preflight;
3. download the official safetensors, configuration, tokenizer, template, and
   multimodal preprocessing assets through the in-process `hf-hub` path;
4. verify every recipe file by identity, size, and SHA-256 before conversion;
5. choose the strongest independently acceptance-gated quantization that fits
   measured unified memory, disk, temporary-space, and safety constraints;
6. convert/quantize the language model and emit the source-matched F16 vision
   projector through hf2q's Rust implementation;
7. write immutable conversion receipts and register the artifact by canonical
   Hugging Face identity; and
8. run the bounded exact-artifact runtime calibration defined below.

`--quant` remains an expert override, but cannot select a candidate that lacks
the model-family quality and compatibility gate. Insufficient hardware fails
before authentication or a large transfer and reports exact requirements. The
product path never shells out to `hf` or `huggingface-cli`; external-CLI and pip
remediation paths are unreachable from this command.

The default layout is:

```text
~/.hf2q/models/huggingface/Qwen/Qwen3.8-27B/<revision>/
├── source/
├── artifacts/
├── receipts/
└── profile.json
```

The source download is resumable and shared by text/projector conversion. At
successful completion, interactive conversion reports the exact source bytes
and asks whether to retain them. Non-interactive use requires
`--source-retention keep|delete`. Deletion occurs only after every output and
receipt verifies, never after a failed/interrupted conversion, and never
removes source data owned by an existing external Hugging Face cache.

### 8. Resolve, calibrate, and serve exact artifacts

The bundle contains the Qwen3.8, Qwen3.6, Gemma 4, and DeepSeek-V4 launchers,
but they become thin relocatable entry points over a typed Rust profile
registry. They resolve the installed binary and `~/.hf2q/models`, never
`/opt/hf2q` or another checkout path.

The canonical prepared-model command is:

```sh
hf2q serve Qwen/Qwen3.8-27B
```

Resolution order is: an explicit revision/`--quant`; the most recently
successfully served compatible local artifact for that repository; the
device-appropriate compatible cached artifact; otherwise a clear failure that
prints `hf2q convert Qwen/Qwen3.8-27B`. A failed load never changes the
preference. Every launch prints the chosen immutable revision, quantization,
projector, and calibrated profile. Serving an official source ID never performs
an implicit conversion or large source download.

After conversion, or after first acquisition of a published GGUF, hf2q spends
a bounded approximately one-minute run measuring the exact artifact's resident
memory, safe prefill batch, KV bytes/token, slot count, shared physical KV
budget, semantic TTFT, prefill, decode, text health, and image health when
applicable. This is runtime calibration, not imatrix/DWQ quantization-quality
calibration, and it cannot promote an unaccepted quantization. The receipt is
keyed by artifact/projector digests, hardware fingerprint, hf2q/MLX ABI,
tokenizer/template, scheduler, and inference settings. An incompatible change
invalidates it. On commands that would initiate calibration—official-source
`hf2q convert` and first-time external-GGUF `hf2q serve` acquisition—an explicit
`--skip-calibration` chooses the accepted hardware-table profile immediately
and records that fallback in the receipt. An ordinary serve of an already
prepared artifact does not silently recalibrate it.

Profiles optimize for agentic coding rather than minimum-memory conservatism.
For Qwen3.8, every slot retains the model's 262,144-token logical context; slot
count never divides that advertised limit. hf2q targets up to eight slots and,
when measured safe, about 1,048,576 aggregate resident tokens in one dynamically
shared physical KV budget. Admission, spill, and eviction enforce physical
capacity without silently truncating a slot's logical context. Active requests
are never evicted.

An explicitly requested upstream-prequantized model uses a distinct path:

```sh
hf2q serve unsloth/Qwen3.8-27B-GGUF
hf2q serve unsloth/Qwen3.8-27B-GGUF --quant Q5_K_M
```

hf2q resolves an immutable repository revision, deterministically selects only
a runtime-supported GGUF candidate compatible with the hardware, downloads it
plus the matching F16 projector, verifies bytes/metadata/binding, records
upstream-prequantized provenance, calibrates it, caches it, and serves it with
hf2q's runtime. It never claims hf2q converted that artifact. In the first
release, external GGUF acquisition is limited to checked-in recipes that pin
the repository revision, allowed GGUF/projector filenames, sizes, and SHA-256
digests. A specific Hugging Face `blob`/`resolve` GGUF URL selects that exact
file only when it normalizes into such a recipe; other external GGUF URLs fail
closed rather than treating bytes and a digest from the same untrusted fetch as
independent verification. Vision is disabled unless the recipe binds the exact
F16 projector. Once an artifact has downloaded and served successfully, later
serves use the cached immutable revision without checking Hugging Face for
changes; network access requires an explicit `--refresh` or a missing cache
entry.

Qwen3.8 advertises image input only when its exact projector is present, bound,
loaded, and reported as multimodal by `/v1/models`. The first public installer
waits for ADR-044's packaged exact-artifact text-plus-vision gate; there is no
public text-first installer shortcut under this ADR.

### 9. Persist inactive sessions as a bounded acceleration cache

hf2q builds on ADR-017 and ADR-027 rather than creating a second persistence
system. Active and recently used sessions form the hot unified-memory tier;
stable request-boundary prefix/KV checkpoints form a cold SSD tier under
`~/.hf2q/cache/sessions/`. The client transcript remains authoritative: a
checkpoint is only an acceleration artifact and its loss never loses the
conversation.

After each successful turn, hf2q atomically commits a sparse latest-useful
checkpoint. On memory pressure it spills or reuses the durable checkpoint for
the least-recently-used inactive session, frees its memory, and restores the
longest exact matching prefix when that conversation returns. Active requests
are never spilled or evicted. Graceful shutdown drains bounded pending writes;
after a crash, only the last completely renamed verified checkpoint is
eligible.

Restore requires exact model and projector digests, tokenizer/template,
rendered prompt prefix, KV substrate/codec and ABI, scheduler/inference
configuration, and vision fingerprint. A client session identifier may improve
indexing but is neither sufficient nor required because OpenAI-compatible
clients normally resend the transcript. Any mismatch, corruption, unsafe
permission, or incompatible version becomes a cache miss and normal prefill.

The setup-recorded byte limit is mandatory. A disabled policy is encoded as
zero; every enabled policy has a positive limit, and zero never means unlimited.
Writes use private directories/files, checksums, atomic publication, sparse
checkpoints, and LRU disk eviction before the limit or free-space guard is
crossed. A disk write failure warns and falls back to safe transcript replay
rather than blocking unrelated new work. The current tree
already proves selected SerialFifo Qwen restart hydration; multi-slot Qwen3.8
vision restoration remains release work and must pass exact-artifact semantic,
isolation, disk-budget, cancellation, crash, and restart gates before the guide
claims it.

### 10. Keep every external integration optional

The core installer, setup, conversion, server, direct API smoke test, and update path
have no integration prerequisite. OpenCode, Agentic Kit, SearXNG, Firecrawl,
Crawl4AI, Node/npm, container runtimes, and their dependencies are never
installed, updated, started, stopped, or removed by the hf2q installer or
updater.

After a user installs OpenCode separately,
`hf2q opencode configure Qwen/Qwen3.8-27B`
emits an `@ai-sdk/openai-compatible` provider pointing at the local `/v1`
endpoint, with a stable provider/model ID, context/output limits, tool support,
and text/image modalities derived from the live server rather than hardcoded
optimism. If OpenCode is absent, hf2q makes no system change and points to the
tested official installation step in the guide. If the endpoint is unreachable
or `/v1/models` does not expose the exact selected artifact and capabilities,
the command writes nothing and instructs the user to start
`hf2q serve Qwen/Qwen3.8-27B` first.

The default target is the user's global
`~/.config/opencode/opencode.json`, matching OpenCode's user-wide provider/model
scope and merge behavior. An explicit
`hf2q opencode configure Qwen/Qwen3.8-27B --project .` writes a project override.
Existing global/project configuration, providers, models, plugins, and Agentic
Kit additions are preserved. A write uses a lock, same-directory temporary
file, fsync, atomic rename, restrictive permissions, and a backup. Strict JSON
may be merged structurally. JSONC with comments is changed only through a
parser that preserves its syntax; otherwise hf2q fails with a generated snippet
and makes no edit. It never replaces an unparseable file. `hf2q setup` does not
perform this optional integration step.

OpenCode and Node.js remain third-party tools with their own release and trust
boundaries. The guide uses the current official OpenCode installation method
and records the exact version used by the acceptance demo.

OpenCode and Agentic Kit have the same status: both are optional integrations,
not hf2q requirements. The full agentic-demo guide adds Agentic Kit only after
OpenCode is installed. Its reproducible acceptance path pins the exact
prerelease resolved in the demo receipt:

```sh
npm install -g @pacphi/agentic-kit@<accepted-version>
ak setup --opencode
ak sync
```

The published guide replaces `<accepted-version>` with a concrete version; it
does not ship that placeholder. It may also show
`npm install -g @pacphi/agentic-kit@next` as an explicitly moving opt-in update
channel, but `@next` is not the version-bound acceptance command. The guide
states the Node/npm prerequisite and third-party trust boundary.
Running Agentic Kit before or after hf2q configuration must converge without
clobbering either tool's provider/plugin entries.

SearXNG, Firecrawl, and Crawl4AI each receive a separate optional integration
guide. Each guide uses the service's official installation and update boundary,
records the tested version, names its ports/data/credential locations, shows
how to connect it through the supported OpenCode or Agentic Kit mechanism,
runs one deterministic smoke test, and explains shutdown and removal. The
service remains independently owned; hf2q does not proxy, supervise, configure,
or claim support for its internals. New services follow the same guide template
instead of expanding the hf2q installer.

`/var/tmp/search-fetch-setup.md` is lab-notebook source material only. Public
guides must not reproduce its user names, absolute home paths, local tokens,
temporary-file dependencies, launch-agent state, or machine-specific claims.
Required wrappers/plugins/templates are checked into the repository with
portable paths. Commands and versions are revalidated against official service
documentation. Fetch/crawl examples bind locally and include redirect-aware
SSRF defenses, private/link-local address rejection, `file:` rejection,
response/time limits, and explicit credential handling; they do not promise
that a free public search engine will remain available.

### 11. Provide one progressive guide

Implementation adds a progressive documentation set:

```text
docs/guides/getting-started.md
docs/guides/qwen38.md
docs/guides/integrations/opencode.md
docs/guides/integrations/agentic-kit.md
docs/guides/integrations/searxng.md
docs/guides/integrations/firecrawl.md
docs/guides/integrations/crawl4ai.md
```

The two core guides cover:

- supported host/storage prerequisites and Hugging Face authentication;
- installer, version-pinned verification, update, rollback, and uninstall;
- `hf2q setup`, the canonical Hugging Face ID/URL grammar, and source-retention
  and session-cache choices;
- `hf2q convert Qwen/Qwen3.8-27B`, automatic quantization selection,
  source-matched projector conversion, calibration, receipt inspection, and
  `hf2q serve Qwen/Qwen3.8-27B`;
- the explicit `hf2q serve unsloth/Qwen3.8-27B-GGUF` upstream-prequantized
  path and its different provenance guarantees;
- direct curl examples for unary/SSE text, image, tool call, and tool-result
  continuation; and
- troubleshooting by stable error and recovery action.

The optional integration guides then cover OpenCode installation/configuration,
the full OpenCode plus Agentic Kit demo, and each external search/crawling
service. Every third-party command is researched against its current official
documentation, version-bound in the acceptance receipt, and clearly separated
from commands owned by hf2q.

The README leads with the hf2q-only install/setup/convert/serve path. Optional integrations
appear afterward and are never prerequisites for core success. Contributor
build instructions remain available but are no longer presented as the normal
installation route. CLI help, `doctor`, README, and guides are audited together
so the clean-account path never recommends Python or claims stale family
support.

## Threat model and failure behavior

| Threat or failure | Required behavior |
|---|---|
| Compromised mirror/CDN asset | A uniquely staged binary must pass exact Developer ID/team validation before it authenticates threshold-signed bootstrap metadata; only that metadata supplies the archive/manifest digests. Any mismatch fails closed, and the high-assurance flow additionally verifies GitHub attestation. |
| Replayed old metadata or release | Version/expiry/role checks reject metadata rollback and freeze. The descriptor-verified live installed-release floor rejects a lower stable SemVer and same-version digest equivocation before transport, returns exact equality as no-download `AlreadyCurrent`, and is rechecked at every later lock boundary. No downgrade occurs without a future explicit one-use intent. |
| Published asset replacement | Immutable draft-to-publish flow forbids overwrite; clients bind exact hashes rather than trusting a mutable tag alone. |
| Malicious archive path/link | A bounded custom classic-ZIP pass preserves every raw record and rejects duplicates, noncanonical order, flags, links/types, ZIP64, comments/extras, local/central disagreement, gaps, overlaps, prefixes, and trailing bytes before the decoder or any extraction runs. The exact embedded manifest and every streamed payload digest must then match the signed inventory. |
| Crash or torn private extraction | The shared-lock stage has a deterministic authenticated name, exact bounded inventory, private `0600`/`0700` modes, in-place exact reconstruction, per-file `F_FULLFSYNC`, bottom-up directory barriers, namespace rebinding, and a final metadata replay. Safe scratch is resumable; hostile shape is retained and fails closed; no version is published. |
| Crash or drift during signed-mode normalization | The first descriptor-bound Developer ID proof is one-use; final-mode files and directories must form exact canonical prefixes, and final files are never repaired. Returned-error and `SIGABRT` matrices cover every mode/full-sync/namespace/endpoint barrier. Current-time TUF replay plus a second same-path/inode native check is mandatory before the inert normalized capability exists; expiry, rollback, selected-generation drift, or namespace replacement returns no capability and publishes nothing. |
| Interrupted install/update | Before activation, old `current` selects the complete old activation. After the sole commit, new `current` selects one complete immutable receipt, relative version link, version, and marker. Partial staging is never executable. |
| Concurrent updater | Installation lock admits one transition and leaves no ambiguous active version. |
| Package-manager collision | Receipt plus manager-database ownership evidence prevents self-overwrite; Cargo route history is never guessed, and `hf2q update` delegates through the recorded/selected route and verifies the result. |
| Offline or hostile update endpoint | Normal commands continue and only verified cached metadata may produce a notice. Installation uses the immutable release's signed bootstrap snapshot rather than the live endpoint; absent or invalid required release files fail before activation. |
| Corrupt/partial model download | Shard identity, size, and SHA-256 fail before conversion; resumable state is retained or quarantined safely. |
| Ambiguous or hostile Hugging Face reference | Only normalized Hugging Face IDs/URLs and explicit local paths are accepted; host, revision, filename, and URL/flag conflicts fail closed. |
| Untrusted external GGUF | The first release accepts only recipe-pinned revision/file/size/digest/projector combinations; architecture, metadata, runtime codec, and projector checks fail before load, and provenance remains upstream-prequantized. |
| Wrong text/projector pair | Preparation, startup, `/v1/models`, and image requests fail closed. |
| Stale or incompatible calibration | Artifact, hardware, ABI, template, scheduler, or settings mismatch invalidates the profile and requires recalibration or a hardware-table profile. |
| KV checkpoint leakage or corruption | Exact prefix/identity/codec/vision gates reject cross-session reuse; corruption becomes a cache miss and is quarantined. |
| Session cache fills disk | Setup-recorded byte/free-space guards and LRU eviction stop writes before the boundary; unlimited mode is not accepted. |
| Existing optional-integration config | Parse/merge validation or no write; never truncate, replace, or silently discard entries. |
| Port or memory conflict | Calibration/launcher refuses before loading the large model and gives a direct recovery command. |

The bootstrap HTTPS request, a compromised maintainer signing identity, a
compromised Apple/GitHub trust root, and a fully compromised local account are
outside what a shell installer can eliminate. Key rotation/revocation and
update-root recovery require the exercised operational runbook specified above
before public self-update ships.

## Implementation sequence

1. The release manifest, install receipt/version marker, crash-durable
   descriptor-bound installation identity, durable first activation,
   comparative TUF spike, shared lock, and production v2 metadata journal land
   first. The tokenized transport-free authenticated-update
   verifier, sealed selector-boundary freshness capability, durable-baseline
   replay, fresh-process discard recovery, root-authorized online-role floor
   reset, independent Python-TUF corpus, canonical channel-pointer schema,
   current-time authenticated target inventory, sealed pointer cross-binding,
   origin-locked transport, lock-reauthenticated fetch capability, and exact
   external manifest/same-FD streamed archive binding plus exact
   embedded-manifest/classic-ZIP inventory verification, descriptor-relative
   inert extraction with current-time lock-held replay, marker-v2 exact
   preparation evidence, and the first-standalone marker/receipt builder have
   landed as dormant bounded contexts. The native Developer ID requirement and
   typed signing-information verifier, descriptor-bound double native check,
   crash-resumable signed-mode normalization, and post-normalization TUF replay
   have also landed without a production Team-ID constructor. Exact marker-
   intent recovery, no-replace prepared-version publication, typed postcommit
   durability-unknown recovery, current-time commit gating, and the activation
   capability bridge have now landed as another dormant bounded context. The
   descriptor-verified active-release reader, numeric rollback/equivocation
   policy, no-download already-current outcome, and repeated floor checks at
   pointer, archive, post-I/O, and preparation boundaries have also landed.
   The verifier-issued Pages metadata route and fresh-session durable commit
   coordinator have landed with root-only 404 termination, role caps, selected-
   floor replay, exact restart discard, and no target or artifact authority.
   Canonical Hugging Face reference grammar, immutable native resolution,
   bounded exact source selection, Git/LFS verification, and conversion
   receipt v3 have now landed. The closed checked-in Qwen3.8 recipe now binds
   its immutable 29-file source, exact accepted text/projector pair, sole
   proven M5 Max 128 GiB selection profile, derived disk floor, and retention
   decision boundary. The bounded canonical pair receipt now additionally
   consumes the exact source, host/disk, artifact, and conversion-receipt
   proofs; its production host proof is minted only from fixed in-process
   macOS sysctl reads plus the selected target filesystem's observed free
   space. The canonical no-options preparation plan now owns that proof and
   derives the bounded source/artifact/receipt/profile layout without creating
   it; its exact original reference is now consumed by the pinned Hub resolver
   and bound to the accepted commit plus complete recipe-owned name inventory.
   A second metadata-only transition authenticates the size and Git/LFS
   identity of every recipe file before any payload transfer. All remain
   explicitly calibration-pending and inert. Next, embed
   the real stable root, compile the real
   public Team ID, pass the protected positive signing fixture, and compose
   that recipe into the no-options paired conversion, source-retention
   transaction, prepared/external artifact provenance, calibration receipt,
   and session policy. Every schema lands with bounded hostile input and
   golden-byte fixtures; schema parsing alone never creates an authenticated
   or ownership-verified capability.
   Before uninstall implementation, freeze and adversarially test its separate
   bounded journal schema and recovery state machine; activation receipts stay
   immutable.
2. Implement `hf2q setup`, the unified `~/.hf2q` state layout, idempotent
   Bash/zsh PATH ownership, hardware inventory, and the one-question bounded
   session policy.
3. Converge conversion on the exact-revision native downloader, add positional
   canonical model references, the accepted Qwen3.8 quantization/profile
   matrix, source-retention transaction, and receipt-bound text/projector
   output.
4. Add canonical-ID prepared-model resolution and the separately labeled
   external-GGUF resolver; pin the exact accepted Unsloth revision, filenames,
   sizes, and SHA-256 values in release recipes.
5. Implement exact-artifact runtime calibration and ambitious shared-KV
   profile selection, then make every canonical launcher relocatable over the
   same typed profile registry.
6. Lift ADR-017/ADR-027 persistence into bounded setup-controlled multi-slot
   Qwen3.8 vision sessions, with isolation, crash, quota, and semantic replay
   gates.
7. Build the candidate bundle; add signing, notarization promotion evidence,
   GitHub attestation,
   immutable-release enforcement, draft publication, and installed-artifact
   hardware gates.
8. Implement and adversarially test the standalone installer, atomic switch,
   active-plus-two rollback retention, ownership detection, universal
   manager-delegating `hf2q update`, launch notices, and receipt-driven
   uninstall.
9. Implement safe global-by-default optional OpenCode configuration and live
   capability verification without adding an OpenCode installation path.
10. Research, write, and run the core and optional integration guides, then
    update `docs/shipping-contract.md` and the README only for surfaces whose
    exact packaged-artifact gates have passed.

Each phase lands with focused tests before the next expands the public surface.
No installer URL is advertised until it returns an immutable, verified bundle.

## Acceptance gates

The release candidate is tested from a clean supported Apple Silicon account
with no Rust toolchain, Python, `hf` CLI, source checkout, or pre-existing hf2q
files.

### Core distribution gates

- the direct GitHub asset installs the exact hardware-accepted archive without
  `sudo`; when enabled, `hf2q.us/install.sh` redirects to that version's
  immutable asset and never serves independent bytes;
- every stable-`latest` GitHub release contains `install.sh`, while crate-only
  releases cannot become that pointer;
- stock macOS Bash 3.2 and zsh both parse and run the installer, including
  version-pinned and no-PATH-mutation flows;
- signed-bootstrap verification passes with the live update endpoint and Apple
  notarization service offline while attached metadata is valid; missing,
  expired, or tampered bootstrap state and a wrong Developer ID team block
  activation, while a non-accepted notarization record blocks promotion;
- the installed binary, all packaged launchers/docs/licenses/manifests,
  receipts, smart Bash/zsh PATH behavior, `doctor`, uninstall, and shell restart
  work from `~/.hf2q` plus `~/.local/bin/hf2q`;
- default uninstall removes only receipt-owned installation files and the exact
  owned PATH stanza, preserves and reports the location of config, models,
  source weights, conversion/calibration receipts, session snapshots, OpenCode,
  Agentic Kit, and third-party service state, and never follows a changed path
  or symlink outside that inventory; `--purge-data` without explicit
  confirmation removes nothing, and interruption leaves either the prior
  runnable installation or a journal-marked retryable uninstall state with no
  dangling `current` target;
- each packaged launcher has already passed its governing family release gate,
  resolves only installed relative paths, and rejects missing/incompatible
  artifacts or port/memory conflicts before model load with the right canonical
  recovery command;
- the installed inventory contains only hf2q-owned files and invokes no model,
  OpenCode, npm, Agentic Kit, container, or service installer;
- setup passes fresh, cancelled, repeated, malformed-existing-config, and
  non-interactive cases, including no-positive-safe-disk-band and zero-limit
  rejection, without destructive or duplicate changes;
- tampered metadata/archive/manifest, traversal/link archives, interrupted and
  concurrent transitions, unsupported host/macOS, and owner mismatch fail as
  specified;
- `hf2q update` passes a real standalone lifecycle plus PATH-isolated recording
  manager fixtures for Homebrew, Cargo, cargo-binstall, contradictory/unknown
  ownership, exact-version, manager-specific rollback refusal, concurrent,
  offline, and manager-failure cases without touching model, calibration,
  session, or integration data; each actual package channel adds a real-manager
  installed-artifact gate before its support claim is published;
- launch notices evaluate cached state on every launch, refresh no more than
  once per 24 hours with a 500 ms connect and 1,500 ms total timeout, never
  contaminate stdout, remain valid JSON events, cover help/version/parse-error
  exits, print the owner-aware update route, honor opt-out, preserve privacy,
  create no state on cache-only reads, and never force an update; and
- threshold-root rotation, expired/replayed/mixed metadata, and recovery by a
  newly verified installer pass before self-update is advertised.

### Official Qwen3.8 gates

- Hugging Face ID, canonical repository URL, and pinned `tree` URL normalize to
  one identity; hostile/ambiguous URL, revision, and filename combinations fail;
- `hf2q convert Qwen/Qwen3.8-27B` downloads the accepted official revision
  once, verifies every shard, performs hf2q-owned quantization/projector
  conversion, safely implements keep/delete source retention, and produces
  complete reproducible receipts;
- every automatically selectable quantization independently passes quality,
  memory, conversion, text, vision, and agentic serving acceptance; a host
  below all accepted profiles fails before authentication or transfer;
- post-artifact calibration is bounded and reproducible, and stale artifact,
  hardware, ABI, template, scheduler, or setting combinations invalidate it;
  explicit `--skip-calibration` on a calibration-triggering command selects
  only an accepted hardware-table profile and records that fallback;
- `hf2q serve Qwen/Qwen3.8-27B` resolves only the matching local prepared
  artifact, obeys explicit/MRU/device-compatible selection, and performs no
  surprise source download or conversion; and
- the exact installed text/projector pair passes ADR-044's remaining vision
  promotion gate plus `/readyz`, `/v1/models`, unary/SSE text, a real image,
  native tools, tool-result continuation, semantic TTFT, cancellation, and
  unchanged-prefix reuse.

### External GGUF and session-restoration gates

- one exact `unsloth/Qwen3.8-27B-GGUF` revision, selected GGUF(s), F16
  projector, sizes, and hashes are recipe-pinned and served end-to-end with
  upstream-prequantized provenance;
- unsupported architecture/quantization, corrupt bytes, incompatible metadata,
  missing/mismatched projector, and unsafe URL forms fail before model load;
- a successfully served cached external artifact triggers no remote update
  check, while explicit `--refresh` is deterministic and receipt-bound;
- setup's `[Y/n]` policy records the displayed byte limit, enforces sparse LRU
  disk use and the free-space guard, and never permits an unlimited product
  cache;
- cold-process Qwen3.8 text and image conversations restore the longest exact
  prefix with correct output/tool semantics, while unrelated conversation,
  tenant, image, model, template, codec, scheduler, and settings cases cannot
  reuse it; and
- disk corruption/full, cancellation, SIGKILL during write, graceful restart,
  memory-pressure spill, and restore failure degrade to safe replay without
  cross-session leakage or blocking unrelated work.

### Documentation and optional-integration gates

- the core direct-API guide completes the packaged Qwen3.8 scenario with no
  optional tool installed;
- the optional OpenCode guide installs OpenCode through its official method,
  configures the live `Qwen/Qwen3.8-27B` provider globally without manual JSON,
  and completes the same text/image/tool scenario;
- global/project, JSON/JSONC, malformed, repeat-run, and Agentic Kit convergence
  fixtures preserve every unrelated provider, model, plugin, and comment, while
  server-down or model/capability-mismatch cases perform no write and print the
  exact serve-first recovery command;
- the optional Agentic Kit guide's setup/sync preserves the hf2q provider and
  completes the OpenCode scenario using the exact receipted package version;
  any displayed `@next` alternative is labeled moving and is not acceptance
  evidence; and
- each SearXNG, Firecrawl, and Crawl4AI guide passes its pinned-version smoke
  test without adding files to or ownership claims inside the hf2q install.

An optional integration failure blocks publication of that guide's verified
claim, not the unrelated core installer. The first public installer itself is
blocked on working self-update, launch notices, and the packaged Qwen3.8 vision
proof enumerated above.

Receipts bind the exact archive, source commit, target, macOS/hardware, model
source/output hashes, prompt/settings, cached-token counts, TTFT,
prefill/decode rates, tool semantics, image identity, and client versions.

## Consequences and non-claims

The primary installation becomes independent of Rust, Python, and a checkout,
while the standalone supported operator surface moves as one rollback-capable
unit.
The cost is a larger release/security surface: signing credentials, notarized
artifacts, trusted metadata rotation, immutable release discipline, installer
testing, and clean-account hardware acceptance become release-blocking work.

The first bundle supports Apple Silicon only. Homebrew formula publication,
Linux, Intel macOS, Windows, background daemons, automatic third-party tool
installation, delta updates, and nightly channels are deferred. cargo-dist may
help generate archives or formulae, but
its installer model does not replace this bundle's scripts/docs/receipt and
atomic-update contract.

Nothing in this ADR is shipped merely because the document exists. Until the
implementation and acceptance gates pass, the checkout build, existing CLI,
and current release assets remain the truthful public behavior. Qwen3.8 text
acceptance and vision-candidate status remain exactly as recorded in ADR-044.
