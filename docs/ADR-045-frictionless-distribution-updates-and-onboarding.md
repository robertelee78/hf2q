# ADR-045: Frictionless distribution, updates, and guided onboarding

- Status: Proposed; product scope corrected on 2026-08-20
- Date: 2026-08-17
- Updated: 2026-08-20
- Owners: hf2q release engineering and operator experience
- Related: `docs/shipping-contract.md`,
  `docs/ADR-005-inference-server.md`,
  `docs/ADR-033-unified-quant-convert-pipeline.md`,
  `docs/ADR-017-persistent-block-prefix-cache.md`,
  `docs/ADR-027-qwen35-tq-kv-cache-and-persist-family.md`

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
3. one tested guide for choosing a supported model and using the existing
   convert-with-quantization, serve, and API surfaces;
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
| Custom TUF client/spike, transport sealed to it, TUF metadata journal, first-activation graph, and their structural CI sentinels | **Remove** | The current code cannot install, update, or uninstall hf2q and does not shorten the path to the first real channel. |
| Release manifest/receipt and installation-identity concepts, archive/signature validation, and atomic publication mechanics | **Keep, then simplify behind the reachable standalone channel** | The first channel needs a signed/notarized native artifact, a small manifest and channel receipt, atomic replacement with one known-good fallback, and observable behavior tests. The artifact spike decides which existing validators remain necessary. |
| Automatic shell-completion mutation on ordinary CLI startup | **Remove** | Completion installation belongs to an explicit setup choice or the owning package/installer, not an unrelated command invocation. Explicit completion generation remains useful. |

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
installation experience: make the installation choice obvious, show exact
copyable commands, and offer familiar package-manager alternatives. This is
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
- Homebrew;
- an npm-compatible native-binary package usable from npm and, where the
  package managers are compatible, Bun, pnpm, or Yarn;
- direct versioned release downloads; and
- Cargo/source installation for contributors and advanced operators.

These are product targets, not permission to advertise placeholders. A method
is marked available only when its exact published artifact, clean install,
version output, setup invocation, update behavior, and uninstall behavior have
been proven. Until then, hf2q.us labels it planned or does not show a copyable
command.

The npm package, if offered, is a thin distributor for the native hf2q binary.
It must not introduce a Node.js implementation of conversion, quantization,
or inference. Homebrew and other package-manager recipes similarly distribute
the same native product.

The first supported production target is `aarch64-apple-darwin`. Additional
platforms require their own truthful artifact and runtime proof.

### 2. Ship one authenticated Apple-Silicon release artifact

All binary channels must resolve to an exact hf2q release built from one
source revision. The Apple-Silicon artifact must be:

- built with the repository's locked Rust dependencies;
- code-signed with the hf2q Developer ID identity;
- notarized and accepted by Gatekeeper;
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
notarization and Gatekeeper acceptance for the exact published bytes. This is
the smallest trust chain for the first Apple-only channel; the dormant custom
TUF client and multi-role local journal are not part of it.

The distributed artifact is a thin `arm64` Mach-O with a macOS 14 deployment
floor, hardened runtime, secure timestamp, the fixed `us.hf2q.cli` signing
identifier, and the expected Developer ID team. A ZIP containing that exact
signed executable is only the Apple notary submission carrier. Apple creates
an online ticket for a standalone executable but cannot currently staple that
ticket to the raw file, so the release retains the accepted submission/log and
requires online Gatekeeper assessment of the exact raw bytes. The ZIP is not a
product download.

The exact-artifact workflow is deliberately three-stage. A no-secret job
builds the locked packed-source candidate, a short protected `apple-release`
job signs and notarizes it in an ephemeral keychain, and a no-secret
Apple-Silicon hardware job runs the full cross-family gate against those exact
signed bytes. The protected job invokes its signer only from the verified exact
checkout and treats the unsigned and signed executables as data: it never runs
candidate code while Apple credentials are present. Version execution belongs
to the surrounding no-secret jobs. Only a successful exact-SHA gate may feed
the release workflow.
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
3. choosing a model appropriate for the operator's task and measured Mac;
4. acquiring the selected Hugging Face source weights through the supported
   existing hf2q path, with an exact revision and clear upstream lineage;
5. converting and quantizing with the current `hf2q convert` syntax;
6. serving the produced GGUF with the current `hf2q serve` syntax;
7. checking the OpenAI-compatible endpoint directly; and
8. optionally connecting an already installed client such as OpenCode.

Model recommendations in the guide must be explicit, dated, and grounded in
the checked-in support matrix and measured hardware requirements. They are not
a hidden runtime recommender.

The guide must distinguish:

- hf2q-converted output from the exact source repository named by the guide;
- explicitly supported external GGUF input; and
- unsupported or approximate family compatibility, which must not be
  presented as supported.

Optional client sections show configuration snippets and verification steps.
hf2q does not install, rewrite, or take ownership of third-party client
configuration. The core guide must remain complete without OpenCode or any
other integration.

Every published command in the guide is an acceptance surface. CI or a
reproducible release gate must prove its syntax, and hardware/model claims must
name the exact artifact, settings, and host evidence.

The first guide model is
`jenerallee78/Qwen3.8-27B-Abliterated-SFT` at exact revision
`08c2f075b43bc06456382db6b918a3dcabdcf4dd`. The guide covers its accepted
Qwen3.8 text path only. It does not broaden the still-separate Qwen3.8 vision
candidate or treat a community checkpoint as an official upstream release.

### 5. Make `hf2q update` honor the installation channel

`hf2q update` is the universal operator-facing update command, but it does not
pretend every channel is the standalone installer.

Each installation records or can unambiguously determine its channel and the
channel-specific package identity. Update behavior is:

- standalone/direct-release installs authenticate and atomically replace the
  native hf2q release using the selected release channel;
- Homebrew installs update through the hf2q Homebrew formula/tap;
- npm-family installs update through the package manager that installed the
  package;
- Cargo installs update through Cargo; and
- source/development checkouts are explicitly unmanaged and receive exact
  source-update instructions rather than an unsafe automated repository edit.

For every advertised end-user channel, `hf2q update` performs the update by
using that recorded channel; it may show the exact package-manager action and
ask for confirmation first. A missing manager or invalid channel receipt is an
actionable failure, not a fallback to another channel. It must never guess a
manager, cross channels silently, reinterpret an arbitrary binary on `PATH`,
or replace a package-manager-owned binary behind that manager's back.

An update changes hf2q-owned release files only. It preserves configuration,
downloaded source weights, converted models, caches, logs, and other operator
data unless a separately documented migration is required. A migration must
be versioned, crash-safe, and reversible or fail before changing the active
installation.

The standalone channel retains the previous known-good hf2q version until the
new version passes validation and activation. Package-manager rollback follows
the guarantees of that package manager and must be documented truthfully.

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

### 6. Make uninstall channel-aware and preserve operator data

`hf2q uninstall` removes hf2q through the installation channel that owns it.
By default it removes only hf2q release files and package-manager records. It
preserves configuration, model sources, converted artifacts, caches, and logs.

Destructive data removal requires a separate explicit purge request, an exact
preview of the owned paths, and confirmation. hf2q must never recursively
delete a broad home, state, cache, or model directory based only on a guessed
path.

The first standalone uninstall removes only `hf2q`,
`.hf2q-standalone.json`, `.hf2q-previous`, and an exact in-progress temporary
or lock name owned by the same installation. Missing or inconsistent channel
ownership fails closed. Configuration and model data are not even inputs to
the default uninstall implementation.

### 7. Keep hf2q.us truthful and product-led

The website will use hf2q's own identity while adopting the useful install
selector pattern: clear method tabs, one exact copyable command, platform and
prerequisite notes, and a visible distinction between available and planned
channels.

The website, README, release notes, installer, package recipes, and canonical
guide must agree on:

- supported platforms;
- current version and artifact provenance;
- available installation methods;
- state/config locations;
- update and uninstall behavior; and
- which later hf2q commands are existing product behavior.

The branded installer URL is enabled only after it returns a real reviewed
script with the expected content type and an authenticated, immutable release
path. A parked page, HTML error, mutable placeholder, or missing release asset
is a release blocker.

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
- advertising unbuilt package-manager channels.

If one of those outcomes is needed, it requires its own governing decision or
the already applicable ADR. It must not be smuggled into onboarding as an
implementation detail.

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

- an unsupported platform or unsigned/unnotarized release is rejected before
  installation;
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
2. Use bounded synthetic fixtures to prove command syntax and conversion
   behavior in the fast hosted gate, without claiming that a synthetic model
   proves chat serving. Use one exact supported Apple-Silicon model artifact
   for the protected end-to-end conversion and serving gate.
3. Prove conversion, quantization, serving, direct API use, and the optional
   OpenCode instructions without adding orchestration code.
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

### Slice D: publish the standalone Apple-Silicon channel

1. Freeze the three-name standalone layout from the local lifecycle spike and
   implement its focused install/update/rollback/uninstall tests.
2. Produce the exact signed, notarized, immutable release artifact.
3. Prove the installer from a clean Mac and a user-owned prefix.
4. Publish the reviewed versioned script and then enable
   `https://hf2q.us/install.sh`.
5. Add the method to the website only after the live bytes pass post-publication
   verification.

### Slice E: add package-manager channels one at a time

For Homebrew, npm-family distribution, and any later channel:

1. build the native-binary package/recipe;
2. prove clean installation and artifact identity;
3. implement and prove channel-aware update/uninstall behavior;
4. verify the installed guide/setup journey; and
5. only then mark the channel available on hf2q.us.

### Slice F: finish universal update, uninstall, and clean-account proof

1. Prove the full channel matrix and mismatch failures.
2. Prove standalone rollback and truthful package-manager recovery.
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
  Face source weights to an hf2q-converted GGUF and a valid OpenAI-compatible
  response.
- The exact protected Apple-Silicon proof records source revision, model
  artifact, quantization, settings, hardware, output correctness, and cleanup.
- The optional OpenCode section proves connection and a realistic tool-call
  continuation, while the core hf2q journey remains independent of it.

### Update and uninstall

- `hf2q update` uses or clearly delegates to the recorded installation channel
  for standalone, Homebrew, npm-family, Cargo, and source/development cases.
- Channel mismatch, offline, corrupt download, signature/notarization failure,
  interrupted update, and already-current behavior preserve the active
  installation.
- Update preserves config, sources, converted models, caches, and logs.
- Uninstall removes only channel-owned release files by default; explicit
  purge previews and removes only exact hf2q-owned data.

### Traceability

- The website, README, guide, CLI help, release notes, package recipes, and
  governing ADR describe the same current behavior.
- Source and packaged-artifact tests execute the load-bearing install/setup/
  update/uninstall and guide selectors.
- Existing code formerly attributed to broader ADR-045 wording has an explicit
  keep/move/remove disposition.

## Current implementation truth

As of 2026-08-20, ADR-045 remains **Proposed**.

What exists:

- hf2q's core conversion, quantization, and serving commands;
- a source/Cargo-oriented install path;
- the canonical tested text-only Qwen3.8 guide for conversion, serving,
  direct API use, and optional OpenCode;
- partial dormant distribution/update security infrastructure;
- a `setup` command that records conversion and serving defaults consumed by
  the existing commands through a selected state root;
- a local standalone installer template, hidden exact-byte bootstrap, public
  `hf2q update`/`hf2q update --rollback`, and marker-gated
  `hf2q uninstall --yes` implementation; and
- dormant model-preparation components created under the prior over-broad ADR
  wording.

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
continuity, Gatekeeper assessment, stable-version checks, and the same atomic
publisher for install and update. The source tree now also contains the
three-stage signed-byte release rail, ephemeral Apple credential handling,
accepted-notary proof receipt, immutable draft publication, exact public-byte
verification, and a real-trust clean-prefix gate that runs the installed
binary's noninteractive setup, revalidates its canonical config, and proves
uninstall preserves that config and model data. The protected signing job uses
only its verified checkout script and never executes candidate bytes. That rail
has not yet run with the project's Apple credentials and has not published a
real signed/notarized hf2q artifact, so the channel remains unavailable.

The unreachable second managed-session store, its runtime authorization, and
the provisional session-cache setup field have been removed. `hf2q setup` does
not create or authorize a separate `cache/sessions` hierarchy.
Automatic shell-completion mutation on ordinary startup has also been removed;
`hf2q completions --shell <shell>` remains the explicit generation surface.

What is not yet the corrected product:

- the live standalone installer at hf2q.us;
- verified Homebrew/npm/direct user channels;
- published-byte proof for the standalone `update`/`uninstall` commands and
  package-manager adapters for later channels; and
- the clean-account installed-artifact acceptance proof.

The remaining dormant components listed above do not become ADR-045
requirements. The scope audit has classified them for removal or retention
under their actual governing decisions before more onboarding code is added.

## Consequences

### Positive

- The ADR now describes the product the operator asked for.
- Installation, setup, documentation, update, and uninstall have clear owners
  and observable acceptance gates.
- Existing conversion/quantization/serving work is reused instead of wrapped
  in a second orchestration system.
- Package-manager support can grow incrementally without advertising fiction.
- Cache and model-runtime engineering can proceed under the decisions that
  actually govern those subsystems.

### Trade-offs

- Correcting setup does not itself ship an installer or package-manager
  channel.
- Previously landed ADR-045-labeled code requires an explicit follow-up audit.
- Channel-aware update/uninstall behavior must be implemented and tested for
  each advertised method rather than hidden behind one generic mechanism.
- The guide becomes a release artifact and therefore requires continuing
  command and hardware proof.

These trade-offs are intentional. ADR-045 succeeds when a new operator can
install hf2q, configure it for their Mac and preferences, follow the existing
workflow, update it through the correct channel, and remove it safely—not when
the repository accumulates more onboarding-adjacent machinery.
