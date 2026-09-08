# ADR-051: Frictionless local-first model resolution

- **Status:** Accepted; v0.1.17 and the v0.1.18 native-Xet transport
  amendment shipped; the v0.1.19 observable-transfer and cache-link amendment
  shipped; the v0.1.20 same-origin Git-metadata redirect and canonical Qwen
  shared-expert admission corrections are implemented with exact-artifact
  publication validation pending
- **Updated:** 2026-09-05 — accepted metadata-driven admission and exact hosted
  selector amendment; implementation and validation are in progress (not shipped).
- **Date:** 2026-08-23; native-Xet transfer amendment 2026-08-26; observable
  transfer/cache-link amendment 2026-08-26; qualified-host Xet policy accepted
  2026-08-27; Git-metadata redirect and canonical Qwen admission corrections
  2026-08-27
- **Related:** ADR-005, ADR-018, ADR-033, ADR-045, ADR-046, ADR-047
- **Research:** `docs/research/hf-download-sota-2026-08-26.md` and
  `docs/research/hf-download-ux-rca-2026-08-26.md`
- **Supersedes in part:** ADR-045's statement that onboarding does not create a
  no-options model workflow; ADR-045's installation and distribution decisions
  are unchanged

## Context

The public product journey should be complete after installation, setup, and
doctor:

```sh
hf2q chat jenerallee78/Qwen3.8-27B-Abliterated-SFT:Q4_K_M
```

The same model identity must work with `serve` and `convert`. Operators should
not have to remember whether bytes came from a prior native conversion, a
hosted GGUF download, the hf-hub cache, or a manual download. They should not
repeat a multi-gigabyte transfer merely because the artifact is outside one
legacy cache manifest.

The baseline spike on 2026-08-23 used the existing debug binary from source
commit `b2d2ae0f`. All three intended commands failed in Clap before resolver
code ran:

| Command shape | Exit | Observed failure |
|---|---:|---|
| `hf2q serve owner/repo:Q8_0` | 2 | unexpected positional argument |
| `hf2q chat owner/repo:Q4_K_M` | 2 | unexpected positional argument |
| `hf2q convert owner/repo:Q8_0` | 2 | required `--output` missing |

This falsifies the narrow hypothesis that the problem is only a missed cache
lookup. The shared operand grammar, artifact authority, managed layout, and
fallback rules are one product contract.

A metadata-only spike against the operator's concrete repository resolved
commit `0a72776892f98db49381fdf69f4b9982222ec9dc`. It exposes selectable Q4,
Q5, Q6, and Q8 text artifacts plus two projector companions: one exact
Q4-text-stem companion and one generic Qwen projector. The operator's managed
root already contains Q5, Q6, and Q8 files whose byte lengths exactly match
the immutable Hub inventory. This measured case reformulated two details:

- Loose discovery scans once, orders candidates by modification time, and
  admits a unique matching quant/byte-length candidate through bounded GGUF
  metadata and tensor-directory validation. It never hashes a multi-gigabyte
  manual model merely to decide whether local inference can start.
- Projector matching prefers an exact `<text-stem>-mmproj.gguf`, then one
  generic `mmproj-*` companion. Multiple candidates after those rules remain
  ambiguous and produce the documented text-only warning.

ADR-047 already supplies useful mechanisms: receipt-backed local discovery,
exact hosted metadata, immutable revisions, strong LFS digests for hf2q-owned
downloads, bounded GGUF admission, and a safe hosted activation path. It does not make canonical
model identities a first-class CLI operand, does not scan the canonical data
directory by default, and deliberately kept ordinary serve conversion-first.
This ADR changes that product policy without weakening ADR-047's authority
checks.

## Decision

### 2026-09-05 amendment: metadata-driven hosted selection and admission

The source-bound spike at `626a62e3` reproduced two failures: ambiguity is
declared before candidate compatibility is probed, and the bounded GGUF reader
rejects BF16 storage type 30. The runtime pin `mlx-native 0.11.2` also cannot
read that storage. A hosted publisher label is not a runtime quantization
policy: distinct artifacts can both declare `general.file_type = 7` while
having different tensor types. See `docs/research/hosted-resolution-kata-2026-09-05.md`.

The resolver accepts a bounded literal hosted selector after `:`. It matches
an exact case-sensitive repository-relative GGUF filename or a
delimiter-bounded filename suffix, case-insensitively. Canonical quant selectors retain native conversion
fallback; a publisher-only selector never silently becomes a different quant
or a native conversion policy. Filenames select requested artifacts but never
prove their role, architecture, storage compatibility, or immutable identity.

Candidate compatibility is established before reporting genuine ambiguity.
Admission uses the runtime's exact architecture/configuration/tensor contract,
shared by hosted preflight and local loading; it has no repository-name,
model-name, directory-name, MTP-name, or publisher-name exceptions. An unknown
architecture is not routed through another family's loader. The bounded
network parser performs generic format and byte-geometry validation; execution
support comes from the operation that consumes each tensor and its native
backend capability. Adding a supported runtime contract makes it available to
admission without a separate hosted architecture allowlist.

Publisher selectors and header-derived runtime quant identities remain
separate. Hosted local reuse and resident pool identity preserve the original
repository, immutable revision, exact filename, and digest. A cached artifact
with the same `general.file_type` cannot satisfy a different explicit selector.
New hosted destinations include a digest of the full repository-relative
filename, avoiding basename collisions within the bounded scan depth. Adoption
preserves that original filename. Existing canonical conversion receipts and
cache entries remain readable.

BF16 storage is admitted only when the published, locked runtime dependency
and hf2q dispatch paths support the actual dense and expert operations. Merely
recognizing its byte width is insufficient. No metadata relabeling, implicit
weight conversion, external inference, or approximate-family fallback is used.

Completion requires both reported commands to load their selected artifact,
focused negative-path regressions, locked build/test checks, and exact-artifact
Apple Silicon multi-turn unary/SSE/tool-result/prefix-reuse validation. Until
that evidence is recorded, implementation remains under validation and must
not be described as a proven serving fix.

### 0. Native Xet is the default remote payload transport

The local-first order is unchanged. When a hosted GGUF or native source weight
is genuinely absent, hf2q uses pinned `hf-hub 1.0.0` and its native Xet client
instead of the former synchronous single-stream `hf-hub 0.5` path. A hosted
artifact is one exact-revision Xet-aware file request. Native source weights
are one bounded snapshot operation containing only the authenticated index's
glob-escaped literal shard paths, with eight file workers. Large model payloads
must advertise Xet; there is no production downgrade to the legacy large-file
HTTP transport.

Small Git-backed metadata can return a 307 whose `Content-Length` describes
only the redirect response body. When that response does not provide
`x-linked-size`, hf2q follows at most four HTTPS redirects on the exact
`huggingface.co` origin and uses only the final successful representation's
length. Every hop must preserve the immutable commit and ETag (and Xet identity
when present); loops, cross-origin targets, embedded credentials, malformed
locations, or identity changes fail closed. Bearer credentials therefore never
cross an origin boundary. Xet/LFS responses that provide `x-linked-size` keep
the existing single-request fast path and do not follow their signed payload
redirects.

hf2q selects one resource-qualified policy before the first Xet session. On
Apple Silicon with at least 64 GiB physical unified memory, and only when the
operator has not set either `HF_XET_HIGH_PERFORMANCE` or `HF_XET_HP`, hf2q
enables upstream Xet high-performance mode. Smaller or non-Apple hosts retain
upstream adaptive defaults. Either explicit upstream variable remains
authoritative, including an explicit opt-out. This is one native-Xet transport,
not a second downloader or an hf2q tuning matrix.

Transport success grants no serving or conversion authority. Exact revision,
selected filename, linked size, strong LFS SHA-256, exact snapshot parent, and
the full local digest pass remain mandatory. Complete standard Hub cache blobs
are reused; incomplete objects are never published as cache hits. See
`docs/research/hf-download-rca-2026-08-26.md` for the source-bound RCA and the
checked-in 16.81 GB cold-cache native-Xet benchmark contract.

### 1. One model operand grammar

`serve`, `chat`, and remote `convert` accept:

```text
owner/repository
owner/repository:QUANT
```

`QUANT` is case-insensitive at input and normalized to its canonical GGUF name.
An existing path, an absolute path, or an explicit `./`, `../`, or `~` path is
always a path. The literal `list` is reserved for the local inventory in
`serve` and `chat`. `--model` remains a compatibility spelling; supplying both
spellings is an input error.

For `convert`, `owner/repository:QUANT` and `--quant QUANT` are equivalent.
Different values fail before network or filesystem mutation. Hosted
pre-quantized bytes never satisfy `convert`: conversion still downloads source
weights and runs hf2q's native converter and quantizer.

### 2. Managed layout and identity receipt

The default durable root is:

```text
${XDG_DATA_HOME:-$HOME/.local/share}/hf2q/models
```

Repository artifacts live under:

```text
<root>/<owner>/<repository>/<immutable-revision>/<artifact>
```

Each adopted, downloaded, or converted artifact has bounded sidecar authority
that records the repository, immutable revision, exact artifact bytes, SHA-256,
quant identity, origin, materialization time, and last successful use. A
projector binding additionally records its own immutable filename, bytes, and
SHA-256. Schema-v3 conversion receipts and the existing canonical `ModelCache`
remain valid authorities and are merged into the same inventory.
The readable components use the same bounded Hugging Face owner/repository
grammar as the public operand. Legacy `v2-<hex(owner/repository)>` and
`owner__repository` directories remain read-compatible only when bounded
receipt/manifest authority proves the exact repository and revision; new
managed artifact writes always use the readable hierarchy. Internal cache
lock names remain injective and case-fold stable, and old/new lock aliases are
acquired in sorted order so binaries from different layout eras cannot publish
the same quant concurrently.

Inventory admits a regular managed artifact while its sidecar and exact size
remain valid. It admits a managed final-leaf symlink only when the regular
sidecar binds an hf2q-authenticated hosted artifact and the retained target is
the digest-named blob in the active standard Hugging Face repository cache;
the exact-revision snapshot must still resolve to that same inode. Projector
links use the same rule. Arbitrary sidecar-authorized symlinks remain rejected.
Repeat discovery performs no model-sized hash.

`--output FILE` keeps its exact-file meaning. `--output DIR` places the derived
artifact name in that directory. Without `--output`, `convert` writes the
revision-bound managed path. Serve/chat hosted downloads use the managed path
unless their command supplies an explicit output destination.

### 3. Local-first resolution without filename trust

Resolution is deterministic:

1. Find verified, repository-bound local candidates in schema-v3 conversion
   receipts, managed-artifact sidecars, and the canonical `ModelCache`.
2. If an exact quant was requested, use only that quant. Otherwise prefer the
   compatible candidate with the newest successful-use timestamp; if none has
   use history, prefer the newest verified materialization timestamp. A
   successfully used bound candidate returns immediately after one locked,
   bounded GGUF metadata/tensor-directory admission. Serving never rehashes
   its complete payload.
3. Scan bounded configured roots and exact-revision canonical hf-hub snapshots
   for loose GGUFs. Exact Hub metadata supplies an immutable revision and the
   candidate quant/byte-length set. A local candidate wins only when that set
   maps it uniquely and its bounded GGUF metadata, tokenizer, tensor directory,
   shapes, encoded storage types, and runtime support contract all pass. A
   filename, substring, modification time, or quant-looking suffix is never
   sufficient authority, and ambiguity fails instead of guessing. The stable
   operator file is served in place without a full-payload hash or managed
   copy. SHA-256 remains mandatory when hf2q downloads or publishes bytes and
   claims immutable payload identity.
   A direct child directory symlink beneath a configured root is treated as an
   operator-configured model root after resolving it to one canonical
   directory. The scanner opens that canonical target with `NOFOLLOW`, retains
   its directory/file descriptors, and reopens against the same authority
   before acceptance. A symlink may also appear as the final regular-file leaf
   of this bounded walk. hf2q records the leaf link identity before and after
   following it, retains the target file descriptor, and revalidates both the
   link and target identities; it never traverses a symlink as a directory or
   accepts a symlinked configured root. This supports operator libraries that
   link individual multi-GiB GGUFs into the managed directory while closing
   resolve-to-load retargeting races.
   When the retained target has an adjacent schema-v3 hf2q conversion receipt,
   that bounded non-symlink receipt supplies repository/revision/quant identity
   even when the source repository publishes no hosted GGUF row. The logical
   link remains the displayed and activated model path, and a symlinked receipt
   or sidecar is never authority. A matching logical sidecar may contribute
   only successful-use history after its artifact fields match the conversion
   receipt authority. Successful-use publication is bound to the exact
   repository, revision, quant, and retained inode that actually loaded; a
   retarget to another otherwise valid same-repository quant cannot inherit
   that history. Receipt discovery requires the same complete schema-v3 model,
   source-bundle, converter-commit, and output identity predicate as inventory.
   It derives the adjacent namespace from the retained descriptor path rather
   than recanonicalizing the mutable public link, carries the retained target
   identity into final GGUF admission, and keeps malformed receipts local to
   their candidate.
   When the receipt binds a projector target and the logical directory contains
   one exact-inode projector leaf, hf2q preserves that logical alias so
   generation-marked pair locking and same-directory validation remain intact.
   If no logical projector alias exists, a target-adjacent receipt pair locks
   its shared descriptor-derived conversion namespace. A separately retrieved
   exact projector beside the logical text link instead locks that logical
   shared namespace; retained descriptors still supply both activation bytes.
   Runtime activation opens `/dev/fd/N` for the retained text descriptor, so a
   public-path swap-and-restore cannot redirect the loader after admission.
   Pool resident-byte accounting stats that same descriptor activation path,
   never the mutable public link; retargeting the logical name to a smaller
   file therefore cannot weaken memory-budget admission while the retained
   model is loading.
   Pool identity, banners, and APIs continue to publish the operator's logical
   path rather than the descriptor alias. If `general.name` is absent, the
   logical filename-derived model ID is applied before the inference worker
   captures family registration, reasoning, and tool-call routing; the late
   display-path rewrite is not treated as sufficient. The explicit repository
   operand is operator opt-in to this structural-match policy; hf2q reports
   that narrower claim and does not claim cryptographic fine-tune identity for
   manual bytes.
4. If no local candidate wins, query hosted metadata. An exact quant downloads
   exactly that supported quant after disk preflight. An unqualified request
   chooses the setup/live hardware recommendation, then the nearest lower
   supported hosted tier. If no recommended-or-lower artifact exists, it uses
   the nearest higher tier that already passed the automatic runtime memory
   and pool-budget admission check before attempting native conversion. An
   exact quant never steps tiers. Unsupported or ambiguous hosted choices fail
   with the available exact options.
5. If no supported hosted GGUF exists, `serve` and owned local `chat` fall back
   to hf2q native source conversion. `convert` always uses this source path.

Normal serve/chat does not move, delete, hard-link, copy, or hash operator
bytes. It retains the structurally admitted descriptor and serves that stable
file in place. If an operator explicitly requests a different output, the
existing adoption path creates an independent CoW clone on a supported Apple
filesystem or performs a copy-and-hash fallback, publishes through one
retained canonical destination-parent descriptor with atomic `NOREPLACE`, and
writes immutable authority only after exact size/SHA-256 proof agrees. A
conflicting destination fails closed and is never overwritten implicitly.

Hosted disk preflight models the actual materialization plan. Hub cache and an
managed destination require exactly the uncached Hub payload extent. After
full size/digest authentication, hf2q atomically publishes a tiny absolute
symlink from the managed artifact name to the digest-named Hub blob. It does
not clone, copy, hard-link, move, or rehash the payload during managed
publication, including when the managed root is on another filesystem. An
already cached exact artifact therefore requires no model-sized destination
extent. Local operator adoption and native-conversion outputs retain their
independent clone/copy publication contracts; they are not reclassified as
Hub-cache links.

Compatibility means the current runtime supports the GGUF architecture and
file type, the bounded header/tensor directory and tokenizer/chat contracts
pass, and existing memory/admission and pair preflights accept it. Recency
never overrides those checks.

For Qwen hosted or manually adopted artifacts, bounded admission uses the same
GGUF metadata/tokenizer parser and validates executable scalar values, every
normal-layer tensor topology and shape, supported one-layer MTP/NextN topology,
exact packed byte geometry, and storage-role support before payload transfer
or managed publication. The v0.1.15 hosted fast path is closed-admitted for
Qwen3.5/Qwen3.8 GGUF architecture identifiers; other source families use
native conversion until they gain their own complete hosted admission
contract. A semantic incompatibility tries the next compatible hosted tier and
ultimately native source conversion; transport or immutable-identity failures
remain fatal.

The Qwen3.5/3.6 shared-expert router follows hf2q's conversion and runtime
contract: Hugging Face `[1, hidden]` is squeezed to canonical GGUF `[hidden]`
so the small router remains F32 and is consumed as one length-`hidden` dot
product. Main-stack and MTP hosted admission validate that same one-dimensional
shape; requiring the pre-conversion `[1, hidden]` form would reject hf2q's own
canonical output and compatible published GGUFs.

### 4. Quant defaults and idempotent conversion

Remote conversion quant precedence is:

1. a non-conflicting operand suffix or explicit `--quant`;
2. the setup-recorded recommendation; or
3. live hardware selection when setup has not run.

A matching verified hf2q conversion at the final destination is an idempotent
success and does not re-download or re-quantize. Default multimodal conversion
is a no-op only when both text and projector receipts, digests, and the pair
generation/binding contract verify; a prior text-only output cannot silently
satisfy the paired default. An external or hosted GGUF is never reclassified
as a completed conversion.

Conversion operation locking follows the conditional product plan. The text
destination is always locked; an explicit projector destination or a
derivable `.gguf` sibling is locked with it. An extensionless text destination
does not fail before source inspection because a text-only model has no
projector product. If that source proves multimodal, the paired driver still
rejects the non-`.gguf` text name before writing either artifact.

Before the first source-weight transfer, remote conversion resolves and HEADs
the complete immutable inventory, aggregates the exact uncached metadata
extent, and stages verified bounded metadata before any weight write. It then
range-fetches only authenticated safetensors headers, builds private sparse
logical-length shards, and runs the production converter in dry-run mode to
obtain exact text and projector product sizes for each candidate quant. The
selected source-cache plus exact product plan is preflighted before weight
transfer; after download, the production plan must match before output
creation. Automatic native fallback steps down through admissible tiers using
those exact product sizes. An exact `repo:QUANT` remains authoritative but
does not bypass disk preflight. Unsupported projector storage fails during
the sparse production plan, before payload transfer.

### 5. Multimodal companion behavior

Before transferring a selected hosted text payload, authenticated bounded GGUF
metadata/token markers determine whether the model declares a supported
multimodal architecture and cause text plus projector to be admitted as one
aggregate cache/destination disk plan. For markerless source repositories
only, hf2q may additionally inspect the exact revision's bounded
`config.json` metadata for a `vision_config` marker. That optional metadata is
not copied into the managed directory and is never required to serve an
existing GGUF. For a selected local artifact, this fallback is resolved at
the candidate's bound revision; a mutable repository-HEAD catalog is never
used to select its companion. A trustworthy locally bound matching projector
is loaded automatically. If it is absent and the exact repository
revision exposes one unambiguous matching `mmproj` companion, hf2q prepares one
owned action containing the exact revision, filename, size, digest, final
destination, and either a retained local/cache descriptor or an exact hosted
download. The aggregate pair preflight returns that action; text publication
consumes it without companion reselection or filesystem rediscovery, then
verifies, binds, and loads it before inference.

For a structurally admitted manual text GGUF, that order starts with an
already-present unambiguous regular sibling (or retained final-leaf symlink)
in the text directory. hf2q structurally admits the smaller projector, binds
it by its complete digest, and retains it for activation. When the immutable
catalog has one unambiguous companion, the local digest must equal that exact
companion; otherwise hf2q warns and continues to the hosted exact companion.
When the catalog has no recognized companion row, the structurally valid
operator-owned sibling remains eligible. A freshly downloaded hosted
projector may be returned by `hf-hub` as a snapshot symlink, so hf2q
authenticates that pointer into the exact revision's canonical repository blob
store before opening the retained no-follow descriptor. Failure to retain any
automatically selected projector clears its path and digest, warns, and
continues text-only rather than aborting repository resolution.

All local text multimodal markers and expected-projector digests used during
planning are read from the already retained text descriptor, never by reopening
the public model pathname. A bound projector is fully digest-authenticated and
stability-checked before its GGUF metadata is parsed or allowed to suppress an
exact hosted repair. Its final activation descriptor is digest-checked again;
a same-size replacement after admission visibly degrades to text-only before
runtime warmup rather than aborting server startup. This complete projector
hash does not introduce a full text-model hash on the serve-in-place path.

Automatic companion failure is a visible warning followed by text-only
serving. It never substitutes an unrelated projector. An explicitly supplied
`--mmproj` retains fail-closed behavior. The existing text/projector pair
preflight remains the final authority; a failed automatic pair preflight warns
and falls back to text-only.

The same local behavior applies to an explicit GGUF path. hf2q first uses a
valid managed sidecar or matched conversion receipt, then the text GGUF's
bound projector digest, then an unambiguous sibling projector. It never sends
a local-path model to the Hub. Ambiguity or failed pair admission warns and
continues text-only.

### 6. Chat and inventory behavior

`hf2q serve list` and `hf2q chat list` call the same read-only inventory and
show repository, revision, quant, origin, recency, projector availability, and
path. Listing performs no Hub request and no full-file hashing.

`hf2q chat owner/repository[:QUANT]` uses the same preparation resolver as
serve by prebinding and retaining the exact loopback TCP listener, then
starting an owned server child with that listener and exact operand. Existing
DNS-SD advertisements do not carry immutable repository/revision/quant
identity, so targeted chat deliberately does not risk reusing an unrelated
large resident model. Plain `hf2q chat` still discovers and reuses existing
servers. A selected global `--state-root` is passed to the owned `serve` child,
so setup-selected quant and serving defaults are identical for direct serve
and targeted chat. The child inherits `HF2Q_AUTH_TOKEN`; its actual bound
loopback port is authorized only when its READY message matches that retained
listener on a private inherited Unix socket. DNS-SD PID/TXT hints never
receive credentials or endpoint authority. Chat renders the server's typed
preparation events as one scrollback-safe live row plus durable phase
milestones. It distinguishes local discovery, bounded local GGUF inspection,
Hub metadata, hosted payload verification/transfer, native conversion, text
load/warmup, projector load/warmup, and authenticated endpoint readiness. A
local hit says explicitly that no model download is needed. Indeterminate work
uses a spinner and elapsed time. Every native-Xet hosted payload publishes
bounded completed bytes, total bytes, measured bytes/second when available,
and elapsed time through the private child/parent startup channel. Interactive
chat and direct serve render bytes, percentage, rate, and ETA in one live byte
bar. Non-TTY output prints the same facts on the first update, each
five-percentage-point milestone, every 30 seconds without a milestone, and
completion, avoiding both silence and roughly 10 Hz log spam. The TUI does not
create a second downloader or converter and never treats an empty SSE role
event or an unauthenticated child message as readiness.

The native-Xet worker callback never writes the terminal or startup socket. It
updates output-agnostic atomics; the synchronous foreground owner samples and
coalesces them at 100 ms before invoking the renderer or nonblocking datagram
publisher. This preserves hf-hub/Xet concurrency and prevents a slow terminal
or backpressured parent from throttling payload workers.

### 7. Global operator presentation and exact brand asset

hf2q has one global operator banner for interactive human commands, not a
chat-only decoration. The only rabbit artwork authority is the exact
`head.svg` published by hf2q.us, 1,387 bytes with SHA-256
`645f8a42049a9a1fd7074a98568c35ec0da947d2e2e997151a1d88c8ce9f2c4c`.
hf2q packages those exact SVG bytes in lossless base64 form and compiles
deterministic terminal rasters from the decoded bytes with the pure-Rust
`resvg` renderer. This avoids inventing a trailing text-file newline absent
from the published asset. Raster generation is a build-time step, not repeated
process-start work. It does not redraw, reinterpret, recolor, approximate, or
substitute character-cell rabbit art.

The published SVG is transparent white/black artwork. Terminal rasters place
those unchanged mark pixels on hf2q.us's exact `#0b0c0e` canvas so the white
silhouette and dark cuts remain visible on both light and dark terminal themes.
The canvas is presentation behind the source mark, not a replacement or
recoloring of it.

On a supported terminal, the raster is emitted through the terminal's native
Kitty graphics or iTerm2 inline-image protocol. cmux is positively identified
by its protected `CMUX_SURFACE_ID` environment and uses Kitty graphics through
its Ghostty terminal core. Alacritty and Apple's Terminal do not expose a
native inline-raster protocol, so they receive an ANSI truecolor half-block
raster sampled mechanically from the same exact SVG. That universal renderer
is a pixel backend, not independently drawn character art; it never invents or
substitutes a rabbit shape.

The implementation preserves the main screen and scrollback: it does not enter
raw mode, take terminal input, or enter an alternate screen merely to show the
banner. Automatic selection is conservative and uses positive terminal-family
evidence, then falls back to the ANSI raster. Multiplexed sessions without a
proven passthrough path use ANSI. The source-derived raster is shown only when
at least 24 columns are available; narrower terminals receive the wordmark
alone rather than a wrapped or distorted mark.
An explicit global terminal-graphics selector permits `auto`, `kitty`,
`iterm2`, `ansi`, or `off`. `off` suppresses the entire logo and wordmark
banner. No selector overrides non-TTY, redirected-output, CI, quiet,
completion, internal-helper, or structured-log suppression.

The banner is written to interactive stderr so stdout remains a stable command
data boundary. It appears once per top-level invocation, including the bare
`hf2q` command overview, `setup`, `doctor`, `convert`, `serve`, `chat`, and
inventory commands. Bare `hf2q` is an interactive landing surface and emits
the banner before Clap's missing-subcommand overview exits; it is not treated
as an explicit help protocol request. Owned chat server children use the
existing quiet/plain arguments and do not print a second banner. Explicit Clap
`--help`/`--version` exits, completion protocols, malformed invocations, hidden
helper commands, redirected output, `serve --quiet`, and `--log-format json`
remain free of graphics and wordmarks.

Direct `serve` uses the same preparation milestones on interactive stderr.
While its single startup row is live, tracing output is retained in a bounded
buffer rather than written through the row. Once preparation is complete and
the listener is bound, hf2q clears the row with one explicitly non-readiness
`listener bound; starting HTTP service` line and flushes retained diagnostics.
The optional long-lived dashboard begins in a yellow listener-bound state and
may turn green only after the concurrently polled Axum service answers a real
authenticated `/health` request. A model-less serve says no model is preloaded
and never claims model preparation. Early failure drops also clear the row and
flush retained diagnostics. The internal native-conversion child has captured
stdio and explicitly disables global branding, so it cannot paint a second
rabbit or progress bar through the parent surface.

The performance spike first rasterized the full 544×764 SVG at runtime. Across
warm batches of ten pseudo-terminal invocations this cost about 194 ms per
Kitty invocation. Reducing the native raster to 224×315 still cost about 150
ms, and the runtime ANSI decode/downsample path cost about 76 ms. This rejected
runtime rasterization as the global-command design. The accepted build-time
assets measured on an Apple M5 Max (`arm64`, macOS 26.5.2) across 50 warm
pseudo-terminal invocations of the locked debug binary's `cache size` command:
8.0 ms/invocation with graphics off, 9.0 ms with the ANSI raster, and 48.8 ms
with the native Kitty transfer. Each run used `/usr/bin/script -q /dev/null`
to preserve a real TTY while discarding emitted bytes. The remaining native
delta is bounded terminal image transport; ANSI is effectively at the
no-graphics process baseline.

The first v0.1.16 release proof exercised normal commands through real PTYs but
did not exercise the literal bare `hf2q` journey. That omission allowed Clap's
missing-subcommand exit to bypass the post-parse banner hook even though
`serve list`, Apple Terminal ANSI, explicit Kitty, and suppression cases were
green. Stable-channel activation was held at v0.1.15 after the operator found
the defect. The corrective hypothesis was that only the early bare-command
control flow differed. A regression run against the released v0.1.16 binary
failed because its captured bare PTY contained no tagline or raster. The
v0.1.17 implementation emits the default banner before that one Clap exit and
keeps explicit help/version, malformed arguments, redirected output, CI,
completion, and structured paths clean. The accepted packed-binary proof pins
Apple Terminal and Alacritty to source-derived ANSI pixels, pins cmux to Kitty,
extracts the cmux PNG and matches its SHA-256 to the compiled exact-rabbit
asset, and rejects alternate-screen entry on every branded path. The first
corrective PR gate (Actions run `32805771272`) then rejected stale `v0.1.16`
declarations in the README and shipping contract even though all Rust tests
shown before the assertion passed. The v0.1.17 candidate therefore updates
those authoritative release declarations in the same correction and must pass
that hosted shipping-contract assertion before merge.

The first protected v0.1.17 release attempt (Actions run `32809448201`) passed
the exact-SHA CI binding, package, audit, signed candidate, and notarization
stages, then failed closed in the packed installed-binary smoke before any tag,
release, or registry mutation. The original smoke used bare `set -e`
assertions, so the hosted log identified neither the failing phase nor the
command; the same Cargo-installed artifact path passed locally both with and
without `CI=1`. The reformulated release-gate requirement is fail-closed *and*
evidence-preserving: every frictionless smoke names its active phase, reports
the exact failed line, status, and command, retains its isolated files on
failure, and the release workflow uploads those files. A negative contract
test drives `/usr/bin/false` through the inventory phase and proves that the
diagnostic survives. The protected release must be rerun from a new exact
merged-main SHA; a passing local retry alone cannot activate v0.1.17.

That retry (Actions run `32814223369`, exact merged-main SHA
`55d88d0ddfeeb325b1639103e672bc6ffb78001a`) failed in the newly identified
`terminal-banner-matrix` phase. Its retained packed-artifact capture contained
the correct rabbit and command overview, but Clap's styled PTY output inserted
SGR controls between `Usage:` and `hf2q`; the smoke incorrectly searched the
raw capture for the contiguous bytes `Usage: hf2q`. The developer shell had
`NO_COLOR=1`, so the same raw-byte assertion passed locally. Removing that
variable reproduced the hosted failure against the already-built artifact and
falsified the hypothesis that the runner or product output was intermittent.

Terminal proof is therefore hermetic and split by concern. PTY helpers remove
ambient color controls, the matrix explicitly exercises styled and
`NO_COLOR` Clap branches, semantic text assertions strip CSI presentation
controls, and protocol assertions continue to inspect raw bytes. Every PR runs
this exact release-binary journey immediately after `cargo build --release`,
before audit and the long hosted-safe suite; the protected release is final
confirmation rather than first discovery for this class of failure.

### 8. Concurrency and failure safety

Per-repository/revision/quant locks cover adoption, download, conversion, and
sidecar publication. A global manifest lock plus reload/merge protects every
cache-manifest mutation, while per-quant successful-use timestamps prevent
two quant processes from publishing stale repository-level state. A
successful-use cache touch additionally requires the current manifest's exact
revision, quant, byte length, and GGUF pathname to resolve to the retained
inode that actually loaded; a concurrently replaced same-quant entry cannot
inherit another artifact's recency. Shared
projector publication rechecks an exact destination after a concurrent
`EEXIST` winner. Every expensive path rechecks after acquiring its lock.
hf-hub/Xet owns partial download state in the standard cache. Managed hosted
publication stages only a same-directory temporary symlink, validates that it
resolves to the retained authenticated blob, and atomically renames it with
`NOREPLACE`; no managed artifact or even its revision directory is created
before payload authentication completes. Local copies retain their existing
same-directory temporary files and atomic rename. Disk space is checked before
transfer using the exact uncached hosted cache extent or exact aggregate native
source/product plan. Native conversion publishes with no-clobber semantics
under an exclusive intent, then a durable journal covers text, projector,
receipts, and pair binding. Retry recovers a crash before journal creation and
terminal committed or rolled-back journal state. Interrupted work leaves no
authoritative partial artifact. Zero-byte,
non-regular, unstable/retargeted symlink, structurally ambiguous,
stale-revision, or unsupported candidates never win resolution; hf2q-owned
downloads and published copies additionally fail on digest mismatch.

Hub-cache inventory retains the Hub root, model, snapshots, revision, and blob
directories and descends with no-follow `openat` authorities; replacing any
public ancestor cannot redirect a scan. Local adoption likewise retains the
verified text/projector files and canonical destination directories before
aggregate disk admission. Device identity and available capacity come from
those descriptors, publication consumes the same authorities, and both an
already-exact destination and a new destination parent are revalidated before
the first pair write.

## Acceptance gates

- CLI tests prove all approved operands, compatibility spellings, conflict
  failures, and no-output conversion parsing.
- Pure resolver tests prove exact-quant selection, use/materialization recency,
  local-over-hosted precedence, ambiguity, nearest-lower hosted preference,
  and resource-admitted nearest-higher fallback before native conversion.
- Filesystem tests prove bounded descriptor-relative discovery, exact digest
  adoption, independent-inode CoW/copy publication for local artifacts,
  authenticated digest-named Hub-cache symlink publication for hosted
  artifacts, idempotent link reuse, snapshot/blob retarget rejection, readable
  new-write paths, legacy v2-layout reads, source-hardlink mutation isolation,
  clone-unsupported local copy fallback, Hub-ancestor replacement exclusion,
  retained cross-device pair accounting, exact-destination and
  destination-parent replacement refusal, atomic sidecars,
  conflicting-destination refusal, and interrupted-partial exclusion.
- Download tests prove immutable metadata revalidation, final-representation
  sizing across bounded same-origin Git-metadata redirects, cross-origin and
  identity-change rejection, unchanged `x-linked-size` fast-path behavior,
  exact quant, disk preflight, full SHA-256, and unique companion selection
  without payload on ambiguous or declined paths.
- Qwen admission tests pin the canonical one-dimensional shared-expert router
  shape across main-stack and MTP validation, and an opt-in exact-revision live
  gate proves that the repository's admitted Q5 wins before native conversion.
- Conversion tests prove setup/live quant precedence, suffix/flag conflict,
  conditional pair-lock planning, managed revision output, receipt-backed
  no-op, and that hosted GGUF never satisfies convert.
- Serve/chat tests prove one shared inventory, owned-chat reuse of serve
  preparation, manual-local admission without invoking a full-payload hash,
  retained-descriptor runtime activation, pre-spawn logical model registration,
  exact automatic-projector digest binding, manual sibling reuse, fresh Hub
  snapshot-to-blob retention, automatic projector load, post-preparation
  text-only warning fallback, and explicit-projector fail-closed behavior.
- Receipt/pair race tests prove first-use history for regular and final-link
  conversions, revision/inode-bound retarget rejection, retained text metadata
  planning through swap/restore on both loose/cache branches, stale-sidecar
  non-authority, bound-projector digest/replacement rejection, descriptor
  namespace resistance to same-inode hardlink aliasing, target-adjacent receipt
  pair locking, and logical-namespace locking for a separately retrieved exact
  projector.
- A retained-authority pool test retargets the public model link to a smaller
  file and restores it after admission. The original retained size still
  drives oversized-budget rejection, while link-identity revalidation detects
  the swap independently.
- Inventory exposes runnable text-model choices only; a companion-only GGUF is
  represented by its text row's `MMPROJ` field and never repeated as an
  unbound model option. Every row field escapes structural controls and
  Unicode bidirectional controls/isolates before it reaches a terminal.
- Operator-UI tests prove the embedded SVG's exact source digest, deterministic
  nonempty rasterization, Kitty, iTerm2, cmux, Alacritty, Apple Terminal, and
  ANSI paths, global suppression rules, native-Xet aggregate progress
  monotonicity, child/parent progress-wire admission, interactive
  bytes/percentage/rate/ETA rendering, bounded line-oriented non-TTY startup
  telemetry, and exact-installed-binary PTY lifecycle behavior: banner-only
  invocations never enter the alternate screen, direct dashboard startup
  enters/restores it exactly once across SIGINT, and owned-chat preparation
  never enters it.
- The exact 16,810,714,944-byte cold-cache Xet A/B runs three trials per arm,
  alternates arm order, verifies the same SHA-256 after every run, and records
  wall time, throughput, CPU, and peak RSS. High-performance mode is accepted
  for the >=64 GiB qualified host only when its median improves over adaptive;
  a favorable single trial is not evidence.
- Locked focused tests, all-target check, release build, full hosted-safe test
  suite, and nonzero Rust coverage evidence at the exact branch head.
  Agentic-QE complexity/audit output is advisory because the installed tool
  does not instrument Rust coverage; a zero-test or heuristic-only report is
  never represented as a passing gate.
- A real operator-owned local artifact is detected without payload transfer;
  any real model load is run only on an uncontended Apple Silicon host and is
  reported separately from source/unit proof.

## Consequences

The common path becomes short and deterministic, and already-owned bytes win
without imposing model-sized startup I/O. Managed sidecars and complete
digests remain immutable authority for hf2q-owned downloads/conversions; a
manual local serve instead makes the narrower claim that one stable GGUF is
structurally valid and executable by this build.

Metadata lookup may be required to obtain the repository's immutable revision
and unique quant/byte-length candidate set. That small request avoids payload
duplication without turning filenames into identity. Bounded compatibility is
then read entirely from the local GGUF; the payload is neither downloaded nor
hashed. If the Hub is unavailable, already-bound artifacts remain usable while
unbound loose files stay visible but ineligible.

A later Qwen3.6 spike disproved the hypothesis that the operator had only a
partial source download. The partial 3-of-42 shard state existed only in the
Hugging Face source cache. The managed library instead contained a direct
`qwen3.6` directory symlink to a 25,043,007,488-byte `APEX-Q5_K_M.gguf` and an
899,283,264-byte `mmproj-qwen36-F16.gguf`. Exact immutable Hub catalog metadata
reported those same byte lengths. The
measured miss was therefore the no-follow walker skipping the directory link,
not absent model bytes. The accepted reformulation admits a bounded canonical
direct-directory target and performs bounded GGUF structural admission without
reading the full payload; no filename or fuzzy model-name match alone becomes
authority.

A subsequent Qwen3.8 spike found the operator's requested Q4_K_M text GGUF and
its projector already present as final file symlinks beneath the managed
`qwen3.8` directory, targeting the release-model library. Blanket rejection of
file symlinks therefore reproduced the reported duplicate-download UX. The
reformulated rule admits only a final regular-file leaf, retains the target
descriptor, and revalidates both identities; nested directory-symlink
traversal remains prohibited. The 16.8 GB text payload is not hashed on this
serve-in-place path. Both targets carry schema-v3 conversion receipts for
`jenerallee78/Qwen3.8-27B-Abliterated-SFT` at revision
`08c2f075b43bc06456382db6b918a3dcabdcf4dd`, so local identity does not depend
on that source repository publishing a pre-quantized GGUF.

ADR-047's explicit diagnostic activation remains supported for remote or
pre-existing endpoints. This ADR adds the simpler owned-local path; it does not
weaken multi-model admission, process ownership, or OpenAI compatibility.

## Validation evidence

The pre-refinement Apple Silicon inventory spike used immutable Qwen3.8
repository revision `0a72776892f98db49381fdf69f4b9982222ec9dc` and established
that the operator already owned exact Q5, Q6, and Q8 text bytes plus a matching
projector. It proved the identity/adoption hypothesis and preserved those
artifacts, but an earlier release-binary chat attempt did not complete a valid
new runtime proof; this ADR does not count that attempt as acceptance evidence.

The v0.1.18 release then shipped the native-Xet transport but did not satisfy
the observable-transfer or managed-publication experience. A real operator run
of `hf2q chat jenerallee78/Qwen3.8-27B-Abliterated-SFT:Q4_K_M` selected a
15.7 GiB hosted artifact and displayed only a spinner plus elapsed time for
more than five minutes. During transfer the managed root exposed an empty
opaque `v2-<hex>` revision directory, while the actual partial bytes lived in
the standard Hugging Face cache. Source tracing proved that hf-hub emitted
aggregate completed/total/rate events, but the owned server rendered them into
its hidden stderr log; the private startup protocol carried only a one-shot
`HostedDownload` event, and the parent therefore intentionally selected its
indeterminate spinner. The release benchmark redirected helper stdout/stderr,
the progress unit test used a hidden bar, and the packed PTY smoke never ran a
cold hosted download. Those gates proved transport throughput and integrity,
not the user-visible contract above. The terminated operator process is not
counted as an hf2q defect because another development agent sent that signal.
The complete source-bound analysis is
`docs/research/hf-download-ux-rca-2026-08-26.md`.

The observable-transfer/cache-link amendment must produce fresh exact-head
evidence. Passing v0.1.18 gates are historical context and do not prove this
amendment. Until the focused progress, path, symlink, retarget, disk-plan,
packed non-TTY, and cold-download PTY gates pass, its status remains under
validation and no speed or UX claim may be published from source inspection
alone.

The resource-policy A/B completed on 2026-08-27 with candidate binary SHA-256
`069e3f3cb0fdb79ead4cfe3cc98e747cf33d02fd444b0be909904380ff352d07`
on arm64 macOS 26.5.2. All six cold outputs matched 16,810,714,944 bytes and
SHA-256 `1ee55c653644d6f645c6b2f39fc56a3ce28093620fd34dd43678875f348f2e1a`.
Adaptive median was 251.23 seconds (63.814 MiB/s) at 3,301,113,856 bytes RSS;
high-performance median was 225.99 seconds (70.941 MiB/s) at 10,223,222,784
bytes RSS, a 10.05% median wall-time improvement with about 6.92 GB additional
median RSS. One HP trial took 260.04 seconds, so the decision claims a median
improvement, not elimination of network variance. This evidence accepts the
>=64 GiB policy floor; it does not waive the remaining packed UX and release
gates.

The cold end-to-end product journey then ran the actual `hf2q chat` command
against a fresh Hugging Face cache and a fresh managed root. The exact 15.7 GiB
Q4_K_M artifact completed in 4 minutes 24 seconds, passed its pinned byte count
and SHA-256 checks, loaded, reached the interactive prompt, and exited cleanly
through `/quit`. The final readable managed path was
`jenerallee78/Qwen3.8-27B-Abliterated-SFT/<revision>/<artifact>.gguf`; its leaf
was a symlink to the authenticated Hugging Face digest blob and both resolved
to inode `131573324`, proving that the journey did not create a second payload
extent. A deterministic recording-terminal test separately proves that the
interactive frame renders cache-to-managed-link context, completed and total
bytes, percentage, transfer rate, and ETA; this avoids treating a PTY capture
tool's suppression of in-place redraw frames as product evidence.

The current branch has blocking focused tests for operand parsing,
setup/exact/live quant precedence, exact/nearest-lower selection including Q2,
local recency, local-only loose-byte compatibility, bounded hostile GGUF
parsing, complete Qwen metadata/tokenizer/normal-layer/MTP topology and storage
admission, canonical hf-hub cache reuse, aggregate source/text/projector disk
planning, destination reuse/conflict, durable
pre-journal and journal recovery, receipt-backed pair reuse, direct-path
projector resolution, exact prepared-projector action consumption, concurrent
publication, markerless candidate-revision config/projector selection,
existing-projector hard-link mutation detection,
independent-inode retained adoption, authenticated sparse native planning,
retained-listener/private-READY targeted-chat authentication, state-root
propagation, malformed-config read-only list parity, heartbeat, and stale
inventory exclusion. The measured branch proof is recorded below; the release
SHA is added only after the immutable release candidate passes every gate.

The final Apple-Silicon local-reuse proof used the operator-owned Q4_K_M pair
for `jenerallee78/Qwen3.8-27B-Abliterated-SFT` with networking disabled. The
inventory command found the text artifact and projector in 0.15 seconds. Serve
reported no model download and no full-file hash, loaded/warmed the 15.7 GiB
text GGUF in 2 seconds, loaded/warmed the 884.6 MiB projector in 1 second, and
published `text,image` modality with the projector present. The smallest
16-token spike exhausted its budget in Qwen reasoning and was rejected as
semantic acceptance evidence. Reformulating the request with 64 output tokens
returned the exact answer `local reuse works`. On the final release binary the
cold request completed in 2.104 seconds with 1,043.12 ms time to first semantic
token; repeating the identical prompt completed in 1.308 seconds with 54
cached prompt tokens and 367.11 ms time to first semantic token. Authenticated
health, model inventory, graceful shutdown, process exit, and listener removal
all passed.

At the same branch state, `cargo check --locked --all-targets --all-features`,
`cargo build --release --locked`, and the serialized locked all-feature test
suite passed. The full suite included 51 library tests, 4,875 binary tests with
55 explicitly ignored, the 36-test persistent-KV harness, the standalone
installer fixture, completion/rollback lifecycle tests, getting-started and
shipping-contract scripts, and the release-binary frictionless journey.

The 2026-08-27 v0.1.20 candidate correction proof used
`jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated` at immutable revision
`afde6ca7c35272a4b5eefb3b97576fdac0f74ba0`. The Hub's initial `config.json`
HEAD returned a 307 with a 348-byte redirect body; the final same-origin
representation was 2,317 bytes. The focused synthetic redirect/security tests
and exact live metadata test passed. The live catalog/header gate then selected
the sole 25,043,007,488-byte Q5_K_M artifact for an automatically admitted
128-GiB host instead of the unsupported native-conversion path. A patched debug
binary using the repository operand and an isolated test port structurally
reused the exact local Q5 bytes, validated 733 tensors, loaded and warmed text
plus projector through `mlx-native`, and reached confirmed HTTP health.
A strict OpenAI-compatible test produced the required `lookup_key` tool call,
continued from its tool result with `ACK`, reused 2,837 of 2,898 prompt tokens,
and emitted semantic SSE content plus one usage event and `[DONE]`. SIGINT
drained the worker and removed the listener cleanly. This is candidate-branch
evidence; the protected exact-artifact workflow remains release publication
authority.
