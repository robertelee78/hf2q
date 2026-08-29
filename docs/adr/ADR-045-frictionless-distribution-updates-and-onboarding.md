# ADR-045: Frictionless distribution, updates, and guided onboarding

- Status: Proposed; product scope corrected on 2026-08-20 and 2026-08-21,
  release gate simplified, the first public cross-version standalone journey
  shipped on 2026-08-21, and channel-aware standalone/Cargo/source lifecycle
  plus explicit purge landed the same day; the issue-146 guide contract was
  corrected after a post-landing regression review; installed completion
  ownership was restored on 2026-08-21 and managed-model completion scope was
  corrected on 2026-08-22; the issue-181 search fallback was corrected after
  a second-host portability failure on 2026-08-28; distinct-account proof remains
- Date: 2026-08-17
- Updated: 2026-08-28
- Owners: hf2q release engineering and operator experience
- Related: `docs/shipping-contract.md`,
  `docs/adr/diary/ADR-005-inference-server.md`,
  `docs/adr/ADR-033-unified-quant-convert-pipeline.md`,
  `docs/adr/diary/ADR-017-persistent-block-prefix-cache.md`,
  `docs/adr/ADR-027-qwen35-tq-kv-cache-and-persist-family.md`

## Scope correction

This ADR governs how an operator installs hf2q, configures it for one Mac,
learns the existing conversion/quantization/serving workflow, updates the
installation, and uninstalls it.

Earlier revisions expanded that product goal into new code for no-options
model preparation, a prepared-model registry, runtime calibration and model
resolution, session persistence, and automatic third-party client setup. That
was a scope error. Those mechanisms are not required by ADR-045.

The corrected product boundary is:

1. familiar installation choices, presented like a modern CLI product;
2. `hf2q setup`, which learns the host and operator preferences and writes
   defaults consumed by later hf2q commands;
3. one tested guide for downloading an exact published hf2q artifact pair and
   using the existing multimodal serve, API, OpenCode, Agentic Kit, and research
   surfaces; generic source conversion remains documented separately;
4. `hf2q update`, which updates hf2q through the channel that installed it;
5. safe, channel-aware uninstall behavior; and
6. a clean-Mac proof that the whole journey works.

Existing code written under the broader wording is not accepted merely
because it exists. It will be audited separately and either retained under
the ADR that actually owns it, simplified, or removed. In particular,
session persistence belongs to ADR-017/ADR-027 if pursued. Model-family,
conversion, quantization, inference, and cache correctness remain governed by
their existing decisions and shipping gates.

### Audited disposition of the earlier implementation

The Slice A audit at exact main commit
`8eecbf720ac991cbc49aba34454c1e4097c9593f` found approximately 35,500 lines
under `src/distribution`, 7,700 under `src/setup`, and 13,000 more in dormant
model-preparation, recipe, payload, and TUF-spike code and tests. Despite that
surface, the public CLI has no `update` or `uninstall` command, the release
workflow publishes source crates rather than a signed native installer
artifact, and the dormant preparation and managed-session paths have no
production caller.

The implementation disposition is therefore:

| Existing subsystem | Disposition | Governing reason |
|---|---|---|
| Public `convert`, `serve`, generic Hugging Face resolution/download, integrity receipts, and explicit imatrix support | **Keep and relabel** under the existing input/conversion/inference decisions, especially ADR-005 and ADR-033 | These are reachable product capabilities used by the guide; ADR-045 does not wrap or replace them. |
| Existing block and Qwen persistence implementations | **Keep outside ADR-045** under ADR-017 and ADR-027 | Setup and onboarding do not create a second cache implementation. |
| Stable Mac/Metal/memory/storage probes and reusable private atomic config-file mechanics | **Keep and simplify** for the corrected setup schema | These directly support system learning and safe config publication. |
| Exact release revision/tag/checksum lineage and the existing Cargo source-package channel | **Keep** | They are useful release evidence and an advanced install channel, but are not proof of the standalone native channel. |
| Session-cache-only setup policy, dormant runtime authorization, and the second generic managed-session store | **Remove** | The runtime authorization and second store are removed. Schema 2 replaces the temporary cache field with defaults consumed by convert and serve. |
| No-options model recipes, prepared-model profiles/registry/publication, source-retention orchestration, and post-conversion calibration state | **Remove** | They have no production caller and replace guide steps with an unrequested orchestration system. Any useful exact model evidence moves to the relevant model/conversion ADR or guide proof. |
| Custom TUF client/spike, transport sealed to it, TUF metadata journal, first-activation graph, custom archive/Mach-O preparation, and their structural CI sentinels | **Removed** | They had no production caller and did not participate in the shipped installer or updater. Their experiments remain in git history. |
| Setup's read-only installation-identity coupling | **Removed** | It was reachable from setup, but required an identity tree that no shipped installation path created or used. Setup retains its own descriptor-bound root, lock, config, and race checks. |
| Reachable standalone record, Apple signature/notarization checks, and atomic publication mechanics | **Keep** | These are the small mechanisms used by the real signed installer, updater, rollback, and uninstaller. |
| Installation-owned shell completion | **Keep narrowly under ADR-045** | The standalone/Cargo installation owns dynamic public completion, refresh, and exact cleanup. Proven installed release binaries may reconcile it on startup because Cargo has no post-install hook; source/debug/unmanaged/root binaries remain non-mutating unless explicit isolated destinations are supplied. |

Removing dormant code does not erase its history. Git and the project decision
ledger retain the experiments and their evidence. Main should carry the
smallest implementation that serves the accepted product contract.

## Context

hf2q already owns the hard product core: Rust-native Hugging Face conversion,
quantization, and Apple-Silicon inference through `mlx-native`. The missing
piece is a coherent operator journey.

Today an operator still has to infer too much:

- how to obtain a trustworthy hf2q binary;
- which installation methods are real and supported;
- which paths, quantization defaults, and serving defaults make sense for
  this Mac and this operator;
- how the existing commands fit together;
- how to connect an optional OpenAI-compatible client; and
- how to update or remove hf2q without losing models or configuration.

The hf2q.us product surface should take the same useful lesson as OpenCode's
installation experience: make the installation choice obvious and show exact
copyable commands. This is
an interaction model, not a request to copy OpenCode's visual identity or to
turn hf2q into a JavaScript application.

## Product intent

An Apple-Silicon operator should be able to:

```sh
curl -fsSL https://hf2q.us/install.sh | sh
hf2q setup
```

and then follow one guide that uses hf2q's existing command surfaces, for
example:

```sh
hf2q convert <HF_REPOSITORY_OR_SOURCE_DIR> \
  --quant <SUPPORTED_QUANTIZATION> \
  --output <MODEL.gguf>

hf2q serve --model <MODEL.gguf> --port <PORT>
```

The exact examples in the published guide must be copied from tested,
currently supported invocations. ADR-045 does not create a new no-options
model workflow and does not hide the upstream model identity from the
operator.

## Decision

### 1. Offer familiar, independently verifiable install methods

hf2q.us will present an install-method selector as a primary product surface.
The intended channels are:

- a standalone installer, with
  `curl -fsSL https://hf2q.us/install.sh | sh` as the primary command;
- direct versioned release downloads; and
- Cargo/source installation for contributors and advanced operators.

These are product targets, not permission to advertise placeholders. A method
is marked available only when its exact published artifact, clean install,
version output, setup invocation, update behavior, and uninstall behavior have
been proven. Until then, hf2q.us labels it planned or does not show a copyable
command.

Homebrew and npm-family distribution are explicitly outside ADR-045. They are
not deferred acceptance requirements, planned website methods, or runtime
ownership cases. A later product decision may add either channel with its own
artifact and lifecycle proof; this ADR does not reserve or implement them.

The first supported production target is `aarch64-apple-darwin`. Additional
platforms require their own truthful artifact and runtime proof.

### 2. Ship one authenticated Apple-Silicon release artifact

All binary channels must resolve to an exact hf2q release built from one
source revision. The Apple-Silicon artifact must be:

- built with the repository's locked Rust dependencies;
- code-signed with the hf2q Developer ID identity;
- notarized with an Apple `Accepted` result whose ticket binds the exact
  signed executable CDHash;
- versioned and immutable after publication;
- accompanied by an exact checksum and release manifest; and
- tested from the bytes that the channel actually installs.

The installer must default to a user-owned location and avoid `sudo`. It must
not install model weights, Python, llama.cpp, MLX-LM, OpenCode, Agentic Kit,
or another third-party runtime. Downloads remain limited to acquiring hf2q's
own release bytes.

The release implementation needs enough authenticated metadata and atomic
replacement behavior to install and update safely. Those mechanisms are
supporting implementation, not additional user-facing product scope. Prefer
the smallest design that proves the observable contract.

The first standalone layout is intentionally small. By default it owns only
these data names in `$HOME/.local/bin`:

- `hf2q`, the active executable and the sole activation point;
- `.hf2q-standalone.json`, a bounded marker that records only that the
  standalone channel owns this executable path; and
- `.hf2q-previous`, the one retained executable used by explicit rollback.

A fourth persistent empty file, `.hf2q-standalone.lock`, serializes install,
update, rollback, and uninstall. Fixed hidden partial names may exist only
while that lock is held or as recognized crash residue.

The install directory may be overridden explicitly. The marker does not
duplicate active-version, digest, transition-history, model, or configuration
state. The running executable reports its own version, and the exact file
bytes provide its digest. A stale auxiliary document must therefore never be
required to decide which executable is active.

The mutable stable-channel document is a locator, not executable authority.
Before activation, the standalone installer/updater verifies the bounded
download, exact checksum and size, Apple-Silicon target, pinned Developer ID
team and executable identifier, and the version reported by the candidate.
Normal update rejects a downgrade. The release workflow separately proves
accepted notarization and exact ticket-CDHash binding for the published bytes. This is
the smallest trust chain for the first Apple-only channel; the dormant custom
TUF client and multi-role local journal are not part of it.

The distributed artifact is a thin `arm64` Mach-O with a macOS 14 deployment
floor, hardened runtime, secure timestamp, the fixed `us.hf2q.cli` signing
identifier, and the expected Developer ID team. A ZIP containing that exact
signed executable is only the Apple notary submission carrier. Apple creates
an online ticket for a standalone executable but cannot currently staple that
ticket to the raw file, so the release retains the accepted submission/log and
rebinds its ticket CDHash to the exact distributed executable. Apple's `spctl`
assessment is defined for top-level app bundles and rejects this valid raw CLI
as “not an app”; it is therefore not used as a raw-executable trust oracle.
The installer and updater instead verify the exact checksum, thin-arm64 shape,
strict Developer ID signature and authority chain, hardened runtime, secure
timestamp, team, identifier, and version. They also combine `codesign`'s
`--check-notarization` online-ticket lookup with the explicit
`--test-requirement '=notarized'` code requirement. The online option alone is
not sufficient: the measured local negative control accepted ad-hoc and Apple
platform code unless the explicit notarized requirement was also present. The
ZIP is not a product download.

The distribution candidate workflow is deliberately short and reusable. A
no-secret job builds the locked packed-source candidate, and a protected
`apple-release` job signs and notarizes that exact input in an ephemeral
keychain. The protected job invokes its signer only from the verified exact
checkout and treats the unsigned and signed executables as data: it never runs
candidate code while Apple credentials are present. One `Release` dispatch
invokes those jobs and then revalidates the packed crate, dependency
provenance, exact unsigned-input binding, Developer ID identity, hardened
runtime, minimum OS and architecture, accepted notary log, ticket CDHash, and
signed executable bytes before publication. The candidate workflow remains
directly dispatchable for diagnostics and optional model qualification; an
operator does not copy a run ID between workflows to publish a release. The
requested source SHA must be an ancestor of main and is bound by candidate
receipts; GitHub's workflow-definition `headSha` is not misused as candidate
identity, so unrelated later merges do not invalidate an in-flight release.
The publication verifier drops its GitHub API credentials immediately after
artifact download and before executing even a verified candidate.

Model-family quality, cache, and performance qualification is independent. It
may consume the same signed candidate and remains required when a governing
model or serving ADR calls for it, but it does not authorize the CLI artifact
and cannot block an otherwise unchanged distribution release because an
unrelated model, runner workload, or performance environment is unavailable.
That workflow assembles a complete draft, downloads and compares every asset,
publishes the already-proven crate, then makes the complete GitHub release
public and verifies its unauthenticated downloads and production-trust
installer. Immutable standalone assets are never uploaded with overwrite
authority, and stable-release workflow runs are globally serialized.

The installer never owns `$HOME/.hf2q`, model directories, Hugging Face
caches, or another application. If `$HOME/.local/bin` is not on `PATH`, it
prints one exact shell instruction instead of silently rewriting unrelated
shell files.

### 3. Make `hf2q setup` configure hf2q, not perform the workflow

`hf2q setup` is the system-learning and operator-configuration step. It must:

1. inspect stable facts about the selected Mac, including Apple chip/model,
   architecture, unified memory, usable Metal device information, OS version,
   and relevant storage capacity;
2. ask only questions that affect how later hf2q commands should behave;
3. explain recommended defaults in operator language;
4. write a versioned hf2q configuration under the selected state root; and
5. be safe and idempotent to run again.

The version-2 config freezes only five stable defaults with immediate
production consumers:

- `convert.quant`, validated by the existing `QuantSelector`;
- `serve.host` and `serve.port`;
- `serve.scheduler`; and
- `serve.max_slots`.

The guide-proven recommendation is `q4_k_m`, `127.0.0.1:8081`, and
`inflight_batched` with one active slot. Interactive setup explains and may
change each value. Setup observes hardware and storage to inform the operator,
but does not persist a hardware snapshot or claim that a model-free probe can
derive a safe model size.

Model source/revision, output path, cache roots, source-retention policy,
authentication secrets, model-specific conversion inputs, and experimental
runtime knobs are deliberately absent. The current commands do not expose one
honest shared cache or source-retention setting, and model-specific paths must
remain explicit. A setup field is allowed only when a later production command
consumes it or the guide clearly uses it.

Explicit command-line arguments override configured defaults. Existing serve
environment overrides remain between CLI and config. Malformed, provisional,
or unsupported config fails with an actionable error before download, model
load, or listener bind; it never silently becomes an unsafe or unlimited
default.

`hf2q setup` does not:

- choose or download a model;
- convert, quantize, benchmark, calibrate, or serve a model;
- create a prepared-model registry;
- enable or implement session persistence;
- install or configure OpenCode or another client;
- start background services; or
- update hf2q.

Setup must probe and decide before mutating the state root. Cancellation leaves
no partial configuration. A successful write is private, bounded,
crash-durable, and atomically replaces only hf2q-owned configuration.

### 4. Publish one canonical guide that uses existing hf2q commands

The onboarding workflow after setup is documentation, not a new orchestration
layer. One canonical getting-started guide will cover:

1. verifying the installation with `hf2q --version` and `hf2q doctor`;
2. reading the supported model/family matrix;
3. downloading and checksum-verifying the exact published text/projector pair;
4. serving both artifacts with the qualified `hf2q serve` settings;
5. proving unary, SSE, and image generation directly;
6. installing full Agentic Kit and connecting stock OpenCode Build without
   removing any coding capability;
7. installing and proving search, fetch, crawl, and extraction; and
8. disabling, re-enabling, stopping, troubleshooting, and uninstalling only
   the exact components created by the guide.

Model recommendations in the guide must be explicit, dated, and grounded in
the checked-in support matrix and measured hardware requirements. They are not
a hidden runtime recommender.

The guide and its provenance record must distinguish:

- hf2q-converted output from the exact source repository named by the guide;
- explicitly supported external GGUF input; and
- unsupported or approximate family compatibility, which must not be
  presented as supported.

The issue-146 guide's primary path is the exact hf2q-produced Q4_K_M text GGUF
and source-matched F16 projector published at an immutable Hugging Face commit.
It verifies both hashes, serves the pair with explicit qualified settings,
proves unary/SSE/image generation, then configures stock OpenCode Build, full
Agentic Kit, and the local research stack. Generic native conversion remains a
separate product capability and provenance source; it is not a second
onboarding path or a substitute for the exact bytes accepted by this guide.

The guide may merge the hf2q provider into third-party configuration only
after backing it up and must preserve existing agents, tools, permissions,
plugins, instructions, and MCP settings. It must never replace the stock coding
prompt, disable tool schemas, or make a restricted agent the default.

Every published command in the guide is an acceptance surface. CI or a
reproducible release gate must prove its syntax, and hardware/model claims must
name the exact artifact, settings, and host evidence.

The first guide model is
`jenerallee78/Qwen3.8-27B-Abliterated-SFT`. Its source provenance remains bound
to revision `08c2f075b43bc06456382db6b918a3dcabdcf4dd`, while the downloadable
guide pair is bound to artifact commit
`40d771ee15d826017f297261f5bedcf2c32cf4c2`, text digest
`1ee55c653644d6f645c6b2f39fc56a3ce28093620fd34dd43678875f348f2e1a`, and
projector digest
`463b264713f8e081f0fae753c80d8089308e01b1e2ac0948dd9966d0711d8f1b`.
The guide exercises both text and vision and does not treat a community
checkpoint as an official upstream release.

#### Research-stack reliability amendment (issue 181)

The guide's optional local OpenCode research stack is accepted only when the
installer proves usable discovery rather than process liveness. SearXNG remains
the primary route with a curated multi-engine pool. Zero usable URLs accompanied
by engine failures are an infrastructure failure, never a truthful "no results"
answer, and an explicit engine, category, language, or time constraint is never
silently discarded.

Nonempty is not synonymous with useful. The runtime and installer apply a
small deterministic query-term relevance gate before accepting primary or
fallback results. The live installation matrix covers a current fact, an obscure
attribution, and company research so a search engine returning unrelated popular
pages cannot satisfy onboarding.

The first implementation passed its three live probes on the development host
but failed on a second Mac: DuckDuckGo returned a CAPTCHA, Yahoo returned a
protocol error, and the nominal Bing fallback returned Price.com, Price
Industries, and Priceline for the gold-price query. The fail-closed activation
gate behaved correctly, but the fallback was not portable because its browser
and stealth routes were two transports to the same evidence origin and the
service itself treated any structurally safe nonempty result as success.

For an unconstrained general query only, the fetch service may make one bounded
fixed-provider discovery cascade after the primary route errors or returns no
usable URLs. The ordered providers are Brave's static search page, Bing RSS,
and Bing browser followed by stealth transport. The caller supplies only the
query and optional language, never a provider URL. Each route uses a focused
query, explicit provenance, public-target validation, and server-side
query-term relevance filtering before it may return `ok: true`; the plugin
repeats the relevance filter as a defense in depth. The entire worst-case
cascade remains below the plugin's 150-second deadline.

An HTTP 200, installed stealth package, consent wall, CAPTCHA page, or unrelated
popular result is not success. Brave supplies a genuinely different search
origin from Bing, but the cascade is still a best-effort local-host fallback,
not a claim that every CAPTCHA is solvable or that it provides an independent
network failure domain.

Search-discovered result URLs are untrusted. Automatic reads therefore use a
server-enforced public-only static path: globally routable DNS answers only,
validated and pinned connections with original TLS SNI/Host, proxy-environment
isolation, standard web ports, no URL credentials, bounded bodies and redirects,
and revalidation at every redirect hop. Browser and stealth fetches remain
available for explicit user-directed URLs but are not automatic result readers
until they can enforce an equivalent connect-time boundary.

The installer activates the plugin and visible `/search` command only after the
functional search gate passes. On a late failure it stops the managed services
and exits nonzero. `--status` likewise exits nonzero when neither the primary nor
bounded fallback returns a usable result; package import is reported as
`stealth_installed`, not as proof of anti-bot capability.

### 5. Make `hf2q update` honor the installation channel

`hf2q update` is the universal operator-facing update command, but it does not
pretend every channel is the standalone installer.

Each installation records or can unambiguously determine its channel and the
channel-specific package identity. Update behavior is:

- standalone/direct-release installs authenticate and atomically replace the
  native hf2q release using the selected release channel;
- Cargo installs update through Cargo; and
- source/development checkouts are explicitly unmanaged and receive exact
  source-update instructions rather than an unsafe automated repository edit.

For every advertised end-user channel, `hf2q update` performs the update by
using that recorded channel; it may show the exact action and ask for
confirmation first. A missing Cargo executable or invalid channel receipt is
an actionable failure, not a fallback to another channel. It must never guess
ownership, cross channels silently, reinterpret an arbitrary binary on
`PATH`, or replace a Cargo-owned binary behind Cargo's back.

An update changes hf2q-owned release files only. It preserves configuration,
downloaded source weights, converted models, caches, logs, and other operator
data unless a separately documented migration is required. A migration must
be versioned, crash-safe, and reversible or fail before changing the active
installation.

The standalone channel retains the previous known-good hf2q version until the
new version passes validation and activation. Cargo rollback is unsupported
unless Cargo itself gains an exact, recorded rollback contract; hf2q reports
that limitation instead of improvising one.

For standalone update, the new executable is downloaded to an OS-managed
temporary file, bounded and fully verified, then copied to the fixed private
sibling `.hf2q-candidate.partial` before any active-path mutation. The current
executable is first retained as `.hf2q-previous`; one same-directory atomic
rename of that verified candidate over `hf2q` is the activation point. Offline,
malformed, corrupt, unsigned, wrong-identity, wrong-target, and interrupted
pre-activation attempts leave `hf2q` unchanged. An error after the rename is
reported as activation-possibly-complete and reconciled from the executable
bytes, not guessed from a transition ledger. `hf2q update --rollback` swaps
only the two exact standalone-owned executable files.

#### Measured installation-ownership contract

The first universal resolver is deliberately limited to channels that can be
proved from current artifacts. A 2026-08-21 spike on the supported Cargo
1.88.0 toolchain installed a tiny `hf2q` package through both `--path` and
`--git file://...` into isolated `--root` directories. Cargo wrote the active
binary at `<root>/bin/hf2q` and kept two adjacent receipts in sync:
`.crates.toml` v1 and `.crates2.json`. Both identified one package ID, exact
version and source, and the exact `hf2q` bin. A separate installed-registry
sample used Cargo's canonical crates.io source ID
`registry+https://github.com/rust-lang/crates.io-index`, even though the local
registry transport used Cargo's sparse cache. `cargo uninstall --root <root>
hf2q` removed the executable and removed the package from both receipts.

A follow-up selector spike then tested default Git, `--branch`, `--tag`,
`--rev`, and `--path` installs. Cargo retained `branch`, `tag`, or `rev` as an
explicit query selector and kept the separately resolved commit after `#`;
the default-branch form retained no selector, only the resolved commit. Path
receipts retained the exact local file URL. Cargo 1.88 also exposes `--index`
for replaying an exact custom-registry index URL. This falsified the earlier
hypothesis that all non-crates.io Cargo updates lacked stable selection data.

The resolver therefore collects all applicable evidence before choosing an
owner:

- a standalone marker is valid only when the existing standalone verifier
  binds it to the canonical running executable;
- a Cargo root is derived only from a canonical `<root>/bin/hf2q` executable,
  never from `CARGO_HOME` or another ambient setting; both current Cargo
  receipts must be bounded, non-symlink, current-user-controlled files and
  must agree on exactly one `hf2q` owner, version, source, and bin;
- a source/development build is recognized only in Cargo's standard
  `target/{debug,release}/hf2q` or target-triple variant below a manifest whose
  package name is exactly `hf2q`; custom target directories and copied
  executables are unmanaged; and
- malformed present evidence is an invalid installation, while two valid
  owners are ambiguous. Neither condition falls back to another channel.

Cargo update delegates through one direct-argv `cargo install` invocation
using the exact derived root, source selector, selected `hf2q` binary,
features, profile, target, and version requirement retained by Cargo. The
canonical registry uses `--registry crates-io`; a path receipt uses its exact
`--path` without updating that checkout; Git uses the exact repository and
retained default/branch/tag/rev selector; and a custom registry uses its exact
`--index` URL. Credential-bearing or otherwise non-replayable source forms
remain owned for uninstall but receive an explicit recovery instruction
rather than leaking credentials or guessing. After Cargo succeeds, hf2q
proves the binary and both receipts still bind the same root, source selector,
and options. Cargo uninstall delegates without a shell to `cargo uninstall
--root <detected-root> --package hf2q@<installed-version> --bin hf2q` and
proves that the binary and receipt entry are gone. A standard source build
similarly tells the operator to update the checkout with its chosen VCS
workflow and rebuild with `cargo build --release --locked`; hf2q never runs
`git pull`, `cargo clean`, or deletes a checkout.

`hf2q update --check` remains a real remote comparison for standalone. For a
Cargo or source installation it reports the detected owner and the exact next
action without mutating anything; Cargo has no stable install dry-run in the
supported toolchain. `--rollback` remains standalone-only because neither
Cargo nor a source checkout records hf2q's one-file previous-version slot. A
raw copied release binary without the standalone marker is unmanaged, even if
its bytes match a release; the recovery message points to the versioned
installer or the owning installation method instead of adopting it silently.

### 6. Make uninstall channel-aware and preserve operator data

`hf2q uninstall` removes hf2q through the installation channel that owns it.
By default it removes only hf2q release files and exact Cargo receipts. It
preserves configuration, model sources, converted artifacts, caches, and logs.

Destructive data removal requires a separate explicit purge request, an exact
preview of the owned paths, and confirmation. hf2q must never recursively
delete a broad home, state, cache, or model directory based only on a guessed
path.

The first explicit surface is `hf2q uninstall --purge-config` and
`hf2q uninstall --purge-cache`, each still requiring `--yes`. Without
`--yes`, uninstall resolves and validates the installation owner and selected
purge roots, prints the exact release action and purge paths, and changes
nothing. Config purge reuses setup's descriptor-relative ownership checks and
removes only `config.toml`, `.config.toml.partial`, and
`.config.toml.lock` from the selected private state root; it does not remove
unknown siblings. Cache purge reuses the existing manifest-owned model-cache
operation: it clears the validated hf2q cache `models/` tree and atomically
resets its manifest while preserving cache locks, Hugging Face's cache,
operator-selected model directories, persistent-KV roots, and logs. A purge
target must be an exact, non-root, current-user-controlled path; missing data
is an idempotent no-op. Release removal is reconciled before purge, and any
post-removal purge failure reports the completed release action and exact
survivors rather than claiming full success.

The first standalone uninstall removes only `hf2q`,
`.hf2q-standalone.json`, `.hf2q-previous`, and an exact in-progress temporary
or lock name owned by the same installation. Missing or inconsistent channel
ownership fails closed. Configuration and model data are not even inputs to
the default uninstall implementation.

### 7. Make installed Tab completion automatic and lifecycle-owned

The earlier removal made a correct observation—an arbitrary source binary must
not edit an operator's shell—but drew the wrong product boundary. It left the
standalone installer and Cargo channel with no completion installation owner,
made the only public surface a stale static snapshot, and contradicted the
required install-and-use experience. The governing distinction is installation
ownership, not whether the process was launched as an ordinary command.

The 2026-08-21 investigation tested three hypotheses:

1. Raw static and dynamic Clap generation were sufficient. Falsified: neither
   installs or refreshes itself, and raw generation retained hidden internal
   subcommands/arguments in the completion grammar.
2. The standalone candidate could provision completion before activation.
   Falsified: `install.sh` executed a temporary download and deleted it after
   publication, leaving generated adapters pinned to a dead path.
3. The installed binary can safely self-provision when installation ownership
   is proven. Confirmed by the existing fail-closed ADR-045 standalone/Cargo
   resolver and by the protected dynamic-adapter pattern in `/opt/repo-to-cve`.
   The reusable result is the shell protocol, binding, atomic-update, and
   startup-block mechanism; r2c-specific command-domain filtering is not
   copied.

The resulting contract is:

- dynamic completion is the first side-effect-free process branch and exits
  before logging, configuration, cache, network, Metal, or model work;
- both dynamic and explicit static generation use a recursively projected
  public Clap command, so hidden installer, transfer, source-teacher, and
  process-lifeline surfaces are absent structurally;
- the exact `clap_complete` protocol version is pinned, with protected Bash,
  Zsh, and Fish adapters and semantic quant/architecture candidates;
- dynamic completion for every user-facing local GGUF model path prefers the
  canonical `${XDG_DATA_HOME:-$HOME/.local/share}/hf2q/models` tree for an
  empty or bare value. This includes `chat --model`, `generate --model`,
  `generate --mmproj`, `serve --model`, `serve --embedding-model`, `serve
  --mmproj`, and both parity `--model` arguments. Decoder and projector GGUF
  filenames are filtered separately, and ordinary filesystem completion
  resumes as soon as the operator supplies an explicit path or no managed
  bare-name candidate exists. Chat remains a hybrid surface: explicit endpoint
  model IDs and Hugging Face repository IDs remain valid even though local
  managed paths are suggested;
- the standalone installer suppresses reconciliation in the temporary
  candidate and invokes an accepted, side-effect-light command on the stable
  installed binary after activation; Cargo provisions on its first accepted
  normal command because Cargo has no post-install hook. Clap help, version,
  and parse-error exits remain protocol-clean and non-mutating;
- automatic destinations require a non-root release binary proven as
  standalone- or Cargo-owned. Source/debug/unmanaged/ambiguous binaries are
  inert unless the caller gives completion-specific isolated destinations;
- registration files and preferred Bash/Zsh startup blocks are reconciled
  atomically. Foreign regular files are backed up before adoption; symlinks,
  non-regular entries, and ambiguous marker layouts are preserved;
- a private bounded receipt binds exact registration bytes and exact managed
  startup blocks. Update activates the new binary, rollback removes old
  bindings then activates the restored binary, and uninstall removes only
  unchanged receipt-bound artifacts. Operator-modified artifacts are preserved
  and reported; and
- `HF2Q_NO_COMPLETION_INSTALL` is the presence-based opt-out. Explicit static
  Bash, Elvish, Fish, PowerShell, and Zsh generation remains for package-owned
  snapshots.

The unavoidable activation boundary is documented: a child cannot alter its
already-running parent shell. Bash/Fish loaders can discover their managed
files, while the one-time notice tells the operator to open a new shell; Zsh's
verified startup block makes the next shell deterministic.

### 8. Keep hf2q.us truthful and product-led

The website will use hf2q's own identity while adopting the useful install
selector pattern: one exact primary command, direct-release and Cargo/source
alternatives, and clear platform and prerequisite notes.

The website, README, release notes, installer, Cargo metadata, and canonical
guide must agree on:

- supported platforms;
- current version and artifact provenance;
- available installation methods;
- state/config locations;
- update and uninstall behavior; and
- which later hf2q commands are existing product behavior.

The branded installer URL is a mutable selector, not another copy of the
installer. It uses a temporary no-cache redirect to the exact immutable
versioned GitHub release asset, never `/latest`. The updater's small stable
record remains a direct same-origin `200` because update transport rejects
redirects. A permanent or cacheable installer redirect, parked page, HTML
error, mutable release target, or missing asset is a release blocker.

## Explicit non-goals

ADR-045 does not require or authorize:

- a no-options model-selection, download, conversion, or serving command;
- a prepared-model registry or canonical-ID runtime resolver;
- automatic runtime calibration or benchmark-driven model selection;
- a new cache, checkpoint, session-persistence, or eviction subsystem;
- changes to model-family conversion, quantization, inference, templates, or
  tool-calling semantics;
- automatic installation or configuration of OpenCode, Agentic Kit, search,
  crawling, or another third-party product;
- background daemons or forced updates;
- downloading a pre-quantized model as a substitute for an hf2q conversion;
  or
- Homebrew or npm-family distribution under this ADR.

If one of those outcomes is needed, it requires its own governing decision or
the already applicable ADR. It must not be smuggled into onboarding as an
implementation detail.

ADR-051 subsequently supplies that separate governing decision for the first
two bullets: repository model operands, managed local authority, local-first
resolution, hosted-GGUF preparation, native-conversion fallback, and local
inventory are now an explicit product surface. The remaining ADR-045
non-goals and all installation/distribution decisions above are unchanged.

## Configuration precedence

The schema-2 setup slice freezes the governing precedence as:

1. an explicit command-line argument;
2. an explicit environment override already supported by that command;
3. hf2q's versioned config; and
4. the command's safe built-in default.

Convert has no quantization built-in: `--quant` wins over config, and absence
of both is an actionable input error. Serve retains its pre-setup built-ins
when config is absent. The global `--state-root` selects the config root for
setup, convert, and serve; a custom root is never inferred from the executable
or a model path.

Security-sensitive choices, such as non-loopback serving or disabling
authentication, require explicit operator intent and must not be inferred from
hardware. Setup-derived defaults must be explainable and reproducible from
stable host facts.

## Failure behavior

The public journey fails closed at ownership and trust boundaries while
remaining understandable to an operator:

- an unsupported platform or unsigned release is rejected before installation,
  and the release rail refuses to publish a non-Accepted notarization result;
- an unavailable package channel is not presented as available;
- setup probe failure or cancellation leaves existing config unchanged;
- invalid config never becomes an unlimited or externally exposed default;
- a guide/model mismatch is a documentation/release failure, not permission
  for approximate runtime routing;
- an update-channel mismatch stops and explains the correct recovery command;
- update failure preserves the current working installation and user data;
  and
- uninstall refuses ambiguous ownership and preserves user data by default.

Diagnostic output must name the failed stage and the next operator action. It
must not print secrets, tokens, signed URLs, credentials, or private model
metadata.

## Implementation sequence

Each slice follows the hf2q Kata and lands a complete observable outcome before
the next slice begins.

### Slice A: correct the decision and audit prior drift

1. Land this corrected product contract.
2. Inventory code, tests, docs, and workflow gates previously attributed to
   ADR-045.
3. Classify each item as required by this ADR, owned by another ADR, useful
   generic infrastructure, or removable scope drift.
4. Update status/docs and remove misleading product claims before expanding
   implementation.

### Slice B: freeze and prove the guide against today's product

1. Write the canonical getting-started guide using current commands.
2. Use bounded synthetic fixtures to prove generic command syntax and
   conversion behavior in the fast hosted gate, without claiming that a
   synthetic model proves chat serving. Use one exact published Apple-Silicon
   model/projector pair for the protected end-to-end serving gate.
3. Preserve source conversion as provenance evidence, then prove the published
   pair, multimodal serving, direct API use, stock OpenCode Build, full Agentic
   Kit, and research tools without adding a second preparation workflow.
4. Feed any actual usability gaps into the setup schema rather than inventing
   a parallel workflow.

Slice B is complete for today's source-install product at hf2q commit
`efa3da2c67daed823ad35c0135e474bb99ac61df`. The protected Apple-Silicon proof
used an Apple M5 Max with 128 GiB unified memory and the exact guide source
`jenerallee78/Qwen3.8-27B-Abliterated-SFT` revision
`08c2f075b43bc06456382db6b918a3dcabdcf4dd`. The selected 21-file source set
was 55,583,125,949 bytes. Its schema-v3 conversion receipt recorded the
canonical 14-LFS-entry source-bundle SHA-256
`8531e68e43a4a28ed6c0b9b41ac33dc9484e0fbb5eae96198a8c45ba9caf18d0`.
Its config, index, and chat-template SHA-256 values were respectively
`7c45051a516d27c45714ce6ca3285f88194b389f6d8ef71b840478903808271c`,
`e0c5e013a335880aba95c437b37d34fce7868c58b9f57f06ae91f91b3c359981`,
and `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.

Native remote conversion completed in 2,959.05 seconds and produced an
866-tensor, 16,810,714,848-byte Q4_K_M GGUF with SHA-256
`d2ea096cf688ebb02a233ee19b66ade4dc48fdff543793c35631bc5e6291aaaf`.
The schema-v3 receipt independently rebound the same source, converter,
selector, size, and digest. The server loaded the text-only artifact with its
embedded tokenizer and template, advertised 262,144 tokens through
`/v1/models`, and passed readiness, coherent unary output, reconstructable SSE
ending in `[DONE]`, required and automatic tool calls, tool-result
continuation, and warm-versus-cold semantic replay. The warm direct
continuation reused 297 of 374 prompt tokens and exactly matched the cold
result.

The automatic multimodal-pair follow-up at converter commit
`54fd9a089c2d9ebf2ec3ac20b8d24fdc1236c318` reused that exact source revision
and source-bundle identity. One default conversion command produced a bound
16,810,714,944-byte text GGUF with SHA-256
`1ee55c653644d6f645c6b2f39fc56a3ce28093620fd34dd43678875f348f2e1a` and a
927,606,848-byte F16 projector with SHA-256
`463b264713f8e081f0fae753c80d8089308e01b1e2ac0948dd9966d0711d8f1b`.
The two schema-v3 receipts share the exact source and converter identities;
their selectors are `q4_k_m` and `f16-mmproj`. Runtime startup reopened 866
text tensors and 496 projector tensors, verified the projector digest embedded
in the text header, and loaded the Qwen3-VL SigLIP/merger path.

The exact first-image cache gate then used an 86,077-token cold text turn and
a first-image follow-up with 86,172 prompt tokens. The follow-up reused 86,072
tokens, performed GPU vision inference, answered that the fixture was red in
73 completion tokens, stopped normally, and left readiness at HTTP 200. This
supersedes the earlier text-only artifact as the native self-conversion proof.
Those exact hf2q-produced bytes were then published at immutable artifact
commit `40d771ee15d826017f297261f5bedcf2c32cf4c2`. The canonical guide downloads
and verifies that pair; it does not ask each new user to repeat the provenance
conversion or substitute a different model-author artifact.

The guide and its retained evidence bind that accepted digest. A separately
dispatched model-qualification workflow may require the runner's
`QWEN38_MODEL_SHA256` setting to match it, but routine CLI publication does not
rerun or reinterpret this model proof. The filesystem path remains
runner-configurable for qualification; a mutable repository variable cannot
silently substitute another Qwen3.8 artifact for the guide's accepted
community checkpoint.

The first OpenCode spike also found and corrected one real documentation gap:
OpenCode 1.18.18 rendered a 7,105-token agent prompt, which the default
SerialFifo path correctly rejected above its 2,048-token bounded transaction.
The existing `--scheduler inflight-batched --max-slots 1` path then performed
the real file-read tool call, continued with the exact file content, and
reported 7,100 cached input tokens. The guide now documents that measured
server invocation. No model-selection, preparation, client-configuration, or
cache subsystem was added.

### Slice C: realign setup and make its config effective

1. Inventory stable host facts with in-process platform APIs.
2. Derive the smallest schema from existing conversion/quantization/serve
   flags and the guide.
3. Implement idempotent prompting and crash-safe private config publication.
4. Make the existing commands consume the config with the documented
   precedence.
5. Prove that explicit flags override config and setup itself performs no
   model or integration work.

The slice implements schema 2 with the five fields above. The old schema 1 was
provisional and is not auto-migrated: its exact bytes remain untouched and the
operator is told to move it aside and rerun setup. Publication retains the
private descriptor-relative lock, exact-prefix recovery, atomic rename, and
durability barriers already proven by setup's filesystem tests.

### Slice D: publish the standalone Apple-Silicon channel — completed 2026-08-21

1. Freeze the three-name standalone layout from the local lifecycle spike and
   implement its focused install/update/rollback/uninstall tests.
2. Produce the exact signed, notarized, immutable release artifact.
3. Prove the installer from a clean Mac and a user-owned prefix.
4. Publish the reviewed versioned script and then enable
   `https://hf2q.us/install.sh`.
5. Add the method to the website only after the live bytes pass post-publication
   verification.

### Slice E: finish universal update, uninstall, and clean-account proof

1. Prove the full channel matrix and mismatch failures.
2. Prove standalone rollback and truthful Cargo/source recovery.
3. Prove default data preservation and explicit purge boundaries.
4. Run the complete install -> setup -> guide -> update -> uninstall journey
   from a clean Apple-Silicon account using only published artifacts.

## Acceptance gates

ADR-045 is not complete until all applicable gates below pass against exact
published bytes.

### Installation and release

- The primary standalone command installs a signed/notarized Apple-Silicon
  binary from hf2q.us without a source checkout or `sudo`.
- Each website method installs the same declared hf2q release and reports the
  expected `hf2q --version`.
- Direct release checksums/manifests and post-publication downloaded bytes
  match the release candidate.
- Planned/unavailable channels cannot be mistaken for working commands.

### Setup

- Setup reports real stable host facts and explains its recommendations.
- Fresh interactive, fully noninteractive, cancellation, rerun, malformed
  config, concurrent invocation, and filesystem-failure paths are proven.
- The resulting config affects existing conversion/quantization and serving
  defaults; explicit flags win.
- Setup performs no download, conversion, quantization, serving, session
  persistence, update, or third-party integration mutation.

### Guide

- Every command parses against the shipped CLI.
- A clean-account operator can follow the guide from its exact pinned Hugging
  Face artifact commit to checksum-verified text/projector bytes and valid
  text, SSE, and image responses.
- The exact protected Apple-Silicon proof records source provenance, published
  artifact digests, quantization, settings, hardware, output correctness, and
  cleanup.
- Stock OpenCode Build proves Bash/file tool calls and continuation with full
  Agentic Kit plus live search/fetch/crawl/extract; resolved configuration names
  alone are not acceptance.

### Update and uninstall

- `hf2q update` uses or clearly delegates to the recorded installation channel
  for standalone, Cargo, and source/development cases.
- Channel mismatch, offline, corrupt download, signature failure, release-time
  notarization failure,
  interrupted update, and already-current behavior preserve the active
  installation.
- Update preserves config, sources, converted models, caches, and logs.
- Uninstall removes only channel-owned release files by default; explicit
  purge previews and removes only exact hf2q-owned data.

### Shell completion

- A standalone install provisions against the stable installed binary, never
  the temporary candidate; Cargo provisions on the first accepted normal
  command. Help, version, and parse-error exits remain non-mutating.
- Bash 3.2 and Zsh 5.9 literal dispatch execute the generated adapters and
  return public command candidates. Static and direct dynamic Bash, Elvish,
  Fish, PowerShell, and Zsh requests contain no hidden surface.
- Quant and architecture candidates are drawn from the shipped parser and
  architecture registry; reserved quant names are never advertised.
- Empty or bare values for every user-facing local GGUF model/projector path
  enumerate the canonical managed-model root in stable name order, while
  explicit relative, home, and absolute paths remain available. Tests cover
  chat, generate, serve model/embedding/projector, and both parity model
  surfaces. Repository-ID-only `cache clear --model` is not given local-path
  candidates. Decoder completion excludes conventional mmproj filenames and
  projector completion excludes decoder filenames.
- Dynamic completion protocol requests produce stdout-only data and perform no
  reconciliation. Package-owned static snapshots use the documented opt-out;
  ordinary source/debug runs without explicit destinations perform no
  completion or startup writes.
- Reconciliation is idempotent, handles broken pipes as success, preserves
  foreign/symlink/non-regular/racing targets, and records exact ownership.
- Update, rollback, and uninstall tests prove refresh or exact cleanup;
  modified artifacts survive and are named to the operator. The rollback
  regression is included in the published crate and runs against the unpacked
  release artifact, not only the source worktree.

### Traceability

- The website, README, guide, CLI help, release notes, and
  governing ADR describe the same current behavior.
- Source and packaged-artifact tests execute the load-bearing install/setup/
  update/uninstall and guide selectors.
- Existing code formerly attributed to broader ADR-045 wording has an explicit
  keep/move/remove disposition.

## Current implementation truth

As of 2026-08-21, ADR-045 remains **Proposed**.

What exists:

- hf2q's core conversion, quantization, and serving commands;
- the signed and notarized standalone install path plus Cargo and exact-source
  alternatives;
- the canonical tested Qwen3.8 guide that downloads the exact published
  multimodal pair and proves text, SSE, image generation, stock OpenCode Build,
  full Agentic Kit, research tools, and cleanup;
- a `setup` command that records conversion and serving defaults consumed by
  the existing commands through a selected state root;
- the published standalone installer, hidden exact-byte bootstrap, public
  `hf2q update`/`hf2q update --rollback`, and marker-gated
  `hf2q uninstall --yes` implementation;
- fail-closed standalone, Cargo, source-development, and unmanaged ownership
  resolution, including direct-argv Cargo update/uninstall delegation;
- ownership-gated dynamic Bash/Zsh/Fish completion with stable installer
  activation, Cargo first-run activation, protected public-only adapters, and
  receipt-bound lifecycle cleanup, including canonical managed-model and
  mmproj path candidates across chat, generation, serving, embeddings, and
  parity; and
- explicit non-mutating config/cache purge previews whose execution preserves
  unknown state siblings, cache locks, external model/Hugging Face data,
  persistent-KV roots, and logs.

The first Slice-D filesystem hypothesis was tested locally on 2026-08-20 with
two distinct real arm64 hf2q executables. Their SHA-256 values were
`e1affca950361961c58cb886cbd0d5307366c188d4214776ac354bfcc43c3d10`
and
`f2e87911234790e615df3e244f55681f5a858a4f2dfbf22e764982e3546d9e52`.
The spike proved clean install, exact-byte update, explicit rollback, offline
failure, corrupt-download failure, interruption before activation, and
default uninstall preservation of a separate `config.toml` and model file.
It showed that the active executable plus one channel marker, one persistent
lock, and one retained executable are sufficient for the observable
standalone lifecycle. The subsequent local implementation adds a canonical
bounded stable-release record, exact size/SHA verification, same-Developer-ID
continuity, release-time notarization proof, stable-version checks, and the same atomic
publisher for install and update. The source tree now also contains the
two-stage standalone-candidate rail, ephemeral Apple credential handling,
accepted-notary proof receipt, immutable draft publication, exact public-byte
verification, and a real-signature clean-prefix gate that runs the installed
binary's noninteractive setup, revalidates its canonical config, and proves
uninstall preserves that config and model data. The protected signing job uses
only its verified checkout script and never executes candidate bytes. The first
credentialed run on 2026-08-20 produced an Apple `Accepted`, zero-issue
submission whose ticket CDHash exactly matched the signed hf2q binary. It also
falsified the prior `spctl` assumption: current macOS rejects a raw Mach-O as
“not an app” even when its code and notarization ticket are valid. No artifact
was published and the hardware job was skipped. A follow-up local falsifier
proved the replacement boundary: the signed hf2q passed the combined online
check plus `=notarized` requirement, while the local ad-hoc hf2q and `/bin/ls`
failed the explicit requirement. The rail now verifies both that runtime check
and the independently accepted notary log/exact CDHash. A subsequent attempt
also demonstrated that coupling publication to the full cross-family model
gate was a product error: unrelated compiler activity on the shared model
runner invalidated a performance phase after signing and notarization had
already succeeded. The rail is now split into a short standalone candidate
workflow, the distribution release, and optional model qualification. Only the
first two govern publication of unchanged CLI bytes. The release's unpacked
crate check is correspondingly limited to compilation, installed and explicit completions,
setup, standalone distribution, and installed CLI behavior. Main CI owns broad
source regressions; model workflows own family-specific correctness, quality,
cache, and performance qualification. That trust separation no longer creates
two operator steps: the release calls the candidate workflow and consumes its
artifacts within one serialized run.

The reformulated rail then completed against exact source
`cfa487a829d6e15508d41089e8d892f18bdb86b0`: main CI run `32470582081`,
Apple candidate run `32472225070`, and release run `32472465388` all passed.
The published v0.1.7 Apple-Silicon binary is 38,269,584 bytes with SHA-256
`ff8a7735b1eac7c7b0c7076f684999460d268a1c596e84daa9333be1c9d56eef`,
Team ID `3T2D2YNTVW`, and identifier `us.hf2q.cli`. GitHub and crates.io
package bytes matched; the public release passed clean install, setup, and
data-preserving uninstall.

Website main `20b0bbcd5a2f05e07c21f567c7aca82b039c77a2` now sends a no-store
temporary redirect from `https://hf2q.us/install.sh` to the exact immutable
v0.1.7 GitHub release asset. It serves the canonical stable record directly
with explicit revalidation. Live transport verification and an isolated
physical-HOME journey passed:
install -> setup -> doctor -> already-current update check/update -> uninstall,
with configuration and model bytes preserved. The measured order matters:
setup establishes the private state root before doctor or another command uses
it.

The v0.1.8 release then completed against exact source
`5bca834264479257a086bfc3d82ac98e17f2df8c`: exact-main CI run
`32509856510` and release run `32511435203` passed. The published signed and
notarized Apple-Silicon binary is 37,688,896 bytes with SHA-256
`c66f3203839a9a6c4d78ffff985c0f2547b1c58fb7bb5f6be53f77b8bf94ff0a`.
The installer SHA-256 is
`bdf6d545f6d2bad8625cde4ca65bb5d008270c925978bebbdde1561ec3561e0c`,
and the stable-record SHA-256 is
`6e0bdce77127965f9f5ccb6d3f960e57a1062e5a75bf6604b248c8f836f2563a`.
The workflow verified the packed crate, ephemeral Developer ID signing and
notarization, crates.io bytes, public GitHub assets, and a clean-prefix
installation. This release includes the client-independent `hf2q chat`
operator surface used by the canonical guide.

Website PR `robertelee78/hf2q.us#5` passed exact-head verification and merged
as `b1d7a1b3fad17981e27f4b64dfb5e137cc46e651`; main verification run
`32514514110` passed. The deployed `/install.sh` now returns a no-store
temporary redirect to the exact immutable v0.1.8 installer, while the stable
record remains a direct same-origin `200` with mandatory revalidation. Live
transport verification matched both public surfaces to the release bytes.

The first public cross-version lifecycle then passed from an isolated physical
path on the Apple M5 Max:

`v0.1.7 install -> setup -> doctor -> update check -> v0.1.8 -> hf2q chat
help -> rollback to v0.1.7 -> update to v0.1.8 -> uninstall`.

The fail-closed run exited successfully, left the install directory empty, and
preserved the generated `config.toml` and a separate model sentinel. This
closes the prior lack of published cross-version standalone evidence without
adding another installation or update mechanism.

The Cargo/source hypothesis was then measured on Cargo 1.88.0 with isolated
path and Git installs. Default-branch, branch, tag, revision, path, and
registry receipt shapes were captured before implementation. Thirteen focused
resolver/manager tests now cover matching and hostile receipts, exact
direct-argv reconstruction, selector retention, option/root reconciliation,
credential-bearing source redaction, and a real offline Cargo path install ->
update -> uninstall round trip. The source-development CLI check is
non-mutating; update without `--check` and uninstall both refuse with exact
checkout instructions. A separate black-box standalone CLI test proves that
explicit config/cache purge preview changes nothing and confirmed execution
removes only the named data while preserving unknown state/cache siblings and
cache locks.

The channel-lifecycle slice landed through PR #151 at exact reviewed head
`c2141e4c42d0e3455c362a3813dbd97dc1597d1d` and main merge commit
`2c12bb43a68d61cec93f292f04b25c2e48144406`. Before landing it passed
`cargo check --locked --all-targets
--all-features`, `cargo build --release --locked`, and the complete
`cargo test --locked` suite: 51 library tests, 4,638 binary tests with 54
declared ignores, and every executed integration target completed with zero
failures. The changed Rust files are rustfmt-clean; whole-tree
`cargo fmt --check` still reports only pre-existing formatting debt outside
this slice. Parser-focused Agentic-QE SAST reported zero findings in
`src/distribution` and `src/setup`; no configured external model provider was
available for its optional consensus pass. This is software/filesystem proof,
not the still-required distinct-account Apple-Silicon guide and real-model
acceptance run.

The source installer for the next release is intentionally readable and adds
only measured boundary hardening: one parse-before-execute compound command,
closed candidate stdin, physical Apple-Silicon detection under Rosetta,
bounded HTTPS transfer, test-only local fixtures, and current-user-owned,
non-group/world-writable install directories checked again at native
activation. Its lifecycle test falsifies truncated inputs, stdin capture,
unsafe directories, and oversized downloads before proving install, setup, and
data-preserving uninstall. The updater accepts exact HTTP `200` only.

The unreachable second managed-session store, its runtime authorization, and
the provisional session-cache setup field have been removed. `hf2q setup` does
not create or authorize a separate `cache/sessions` hierarchy.
The former unconditional shell-completion experiment was removed because it
could not distinguish a live install from an ephemeral source binary. The
2026-08-21 amendment restores the capability behind ADR-045's proven
standalone/Cargo ownership boundary, adds exact lifecycle cleanup, and retains
`hf2q completions --shell <shell>` as the explicit static generation surface.
The dormant no-options model recipe, preparation plan, paired-artifact
publication, prepared-profile registry, retention, and calibration-pending
state have been removed as well. Generic remote conversion remains owned by
ADR-033. The only exact source projection retained is ADR-046's private,
read-only Qwen source-teacher manifest; it cannot download, convert, register,
or serve a model.

The unreachable custom TUF verifier and spike, sealed transport, metadata
journal, custom archive/Mach-O preparation, and first-activation graph have
also been removed; they were not called by `hf2q update`, the installer,
setup, or the release workflow. Setup's separate read-only installation-
identity verifier was reachable, but it coupled configuration publication to
an identity tree that no shipped installation path created or used, so that
coupling was removed as well. The reachable standalone record, Apple trust
checks, exact download, rollback, uninstall, and atomic publisher remain the
sole distribution implementation. That removal landed on main in merge commit
`ccfa4dc368e2d2274d3234e75e10a69d9da56e0f` after exact-head CI passed.

What is not yet the corrected product:

- the complete guide/update/uninstall journey from a distinct clean macOS
  account, including the protected real-model acceptance path.

## Consequences

### Positive

- The ADR now describes the product the operator asked for.
- Installation, setup, documentation, update, and uninstall have clear owners
  and observable acceptance gates.
- Existing conversion/quantization/serving work is reused instead of wrapped
  in a second orchestration system.
- Additional install-channel support can be proposed later without
  advertising fiction now.
- Cache and model-runtime engineering can proceed under the decisions that
  actually govern those subsystems.

### Trade-offs

- Correcting setup does not itself ship an installer or lifecycle channel.
- Previously landed ADR-045-labeled code requires an explicit follow-up audit.
- Channel-aware update/uninstall behavior must be implemented and tested for
  each advertised method rather than hidden behind one generic mechanism.
- The guide becomes a release artifact and therefore requires continuing
  command and hardware proof.

These trade-offs are intentional. ADR-045 succeeds when a new operator can
install hf2q, configure it for their Mac and preferences, follow the existing
workflow, update it through the correct channel, and remove it safely—not when
the repository accumulates more onboarding-adjacent machinery.
