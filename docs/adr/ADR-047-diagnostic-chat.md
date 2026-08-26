# ADR-047: Diagnostic chat over the native inference server

- **Status:** Accepted; model-swap proof-integrity corrections and the one-process universal generative-family hardware matrix are sealed; the exact Qwen3.8 cross-format single-hash swap matrix is hardware-sealed; embedding and projector lifecycle execution remains separate
- **Date:** 2026-08-20
- **Updated:** 2026-08-25
- **Related:** ADR-005, ADR-017, ADR-040, ADR-043

## Context

hf2q owns conversion, quantization, and native inference for agentic clients.
It is not an agent harness. Operators nevertheless need a deliberately plain
way to ask the loaded model a few questions when separating a model or server
defect from behavior introduced by OpenCode, BrowserCode, or another client.
Using a second feature-rich chat harness for that comparison defeats the
diagnostic purpose.

The existing server already supplies most of the required path:

- OpenAI-compatible multi-turn chat requests and SSE responses;
- separate content, reasoning-content, and structured tool-call deltas;
- prompt-prefix/KV reuse across an unchanged transcript;
- a request-time, memory-bounded multi-model `HotSwapManager`;
- model and pool telemetry through `/v1/models` and `/metrics`; and
- graceful whole-process shutdown through `POST /shutdown`.

Three isolated spikes at source commit `15318b1b` tested the missing seams
before this decision was accepted.

1. A byte-buffered client decoded deliberately fragmented SSE, including a
   UTF-8 sequence split across transport chunks, and reconstructed content,
   reasoning, tool calls, usage, timing, and the next-turn transcript. Five of
   five spike tests passed. The server already produces all desired usage
   fields, but its final streaming chunk currently discards timing fields held
   in `StreamStats`.
2. On macOS 26.5.2 arm64, twelve of twelve direct DNS-SD LocalOnly trials
   registered an OS-assigned TCP port, browsed and resolved the exact port,
   connected to it, and observed removal after an ungraceful registrar death.
   `async-dnssd 0.5.1` separately passed the same lifecycle on Rust 1.88.
3. Pool-policy simulation can predict exact LRU victims without mutation, but
   the current `HotSwapManager::evict` immediately returns logical bytes while
   an in-flight request `Arc` can keep the physical model alive. The existing
   test and a fail-first spike both proved this. `Engine::shutdown` drains its
   queued and active work, but permanently stops the engine and is not by
   itself a per-model admission gate.

The third result falsifies the original idea that explicit switching can be a
thin call to the existing eviction function.

## Decision

### Product boundary and terminal behavior

`hf2q chat` is a diagnostic, scrollback-preserving terminal client for an
OpenAI-compatible server. It does not use an alternate screen and does not
host agents, execute tools, run evaluations, edit arbitrary request JSON, or
persist conversations.

The client keeps context only for its current process. Every successful turn
resends the complete ephemeral transcript; failed or truncated responses are
not committed. The default request contains only:

- `model`;
- `messages`;
- `stream: true`; and
- `stream_options.include_usage: true`.

Operator flags may add `--system`, `--temperature`, `--top-p`,
`--max-tokens`, `--seed`, and `--reasoning-effort`. The terminal commands are
`/new`, `/model`, `/thinking auto|on|off`, `/status`, `/detach`, and `/quit`.
The thinking command maps to the existing `hf2q_enable_thinking` request
field. Reasoning is visibly separated and dimmed only when the server emits
it. Tool calls are assembled and displayed as structured data but are never
executed.

`--url` may name any OpenAI-compatible endpoint. The basic chat path does not
require that endpoint to claim hf2q identity. hf2q-specific lifecycle actions
are enabled only when the corresponding capability endpoint is actually
available. Process ownership is never inferred from endpoint identity: a
server is process-owned by chat only when that chat process spawned it.

### Local instance discovery and ownership

Every macOS `hf2q serve` process registers `_hf2q._tcp` through DNS-SD with
`kDNSServiceInterfaceIndexLocalOnly` after the HTTP listener has successfully
bound. Its registration handle lives through the server lifetime. The
advertisement contains only non-secret routing hints such as schema version,
PID, start time, and actual bound port; it never contains authentication
tokens. LocalOnly is machine-local, not Unix-user-local, so all names and TXT
records are untrusted discovery candidates.

`hf2q chat` browses LocalOnly only, resolves candidates, then verifies each
endpoint over HTTP. Automatic discovery uses the resolved port only and
connects through loopback; it never follows a TXT-provided host or URL. A PID
from TXT is a display/correlation hint and is never process-control authority.
Chat neither scans arbitrary ports nor maintains an ephemeral registry file.
A manually launched `hf2q serve --port 9123` is therefore as discoverable as
a server started by chat.

DNS-SD does not authenticate the process behind a loopback port. Automatic
discovery therefore never sends credentials while probing candidates. When
`HF2Q_AUTH_TOKEN` is set, chat requires an explicit `--url`; naming the endpoint
is the operator's trust decision and bearer authentication is then used for
that endpoint only.

- One verified local server is selected automatically.
- Multiple verified servers produce a numbered picker showing endpoint and
  resident models.
- With no server, chat starts the current hf2q executable on a loopback,
  OS-assigned port, waits for verified discovery, and selects the requested
  model or offers the server's local model catalog.

Exiting chat stops only a server it spawned, using the existing graceful
shutdown path. `/detach` and `--keep-serving` relinquish that responsibility.
Pre-existing servers are never stopped by chat.

The owned-child contract includes abnormal terminal exit, not only `/quit` or
EOF. A chat-started server runs in its own process group and has an explicit
parent-lifetime channel. Normal exits run bounded cleanup. If SIGINT, SIGTERM,
terminal loss, or another abnormal exit kills chat before cleanup, loss of the
lifetime channel makes the owned server terminate its own process group.
Detach is an explicit message on that channel, never inferred from a closed
descriptor. Bounded graceful shutdown escalates to the owned process group and
reaps it; it never signals a discovered or explicit endpoint.

The owned server's stdout/stderr are not inherited by the interactive
terminal. Stderr is redirected directly to a private durable log. On startup,
session, or cleanup failure the log is retained and its path is reported, but
its arbitrary contents are never copied into terminal errors automatically;
on detach the log is likewise retained and reported. This prevents a child
download progress bar from continuing to paint over a shell prompt after chat
exits or blocking on a pipe whose TUI-side reader disappeared.

DNS-SD registration is isolated behind an hf2q-owned module. Non-macOS builds
retain explicit-URL chat but do not claim automatic local discovery. DNS-SD
failure must be visible in logs and must not make the established HTTP server
unusable.

### Mixed Hub repositories and exact GGUF selection

A bare Hugging Face repository is not an artifact identity. ADR-005 retains
its conversion-first behavior for ordinary `serve --model owner/repo` and
ordinary OpenAI request-time resolution, but diagnostic chat must not silently
choose that path when the same repository already contains GGUF artifacts.

For an hf2q-capable endpoint, a unique resident repository match returns
immediately without Hub access. An ambiguous or nonresident repository asks
the server for a metadata-only hosted catalog before activation. The server
uses its own Hub credentials; neither the token nor raw repository inventory
crosses into the TUI. The server retains repository, exact commit, safe
filename, byte size, and strong LFS SHA-256 behind a short-lived opaque
candidate ID. The TUI receives display fields only: filename, bytes, role,
selectability, unavailable reason, and a filename-derived `quant_hint`.
Catalog metadata does not claim the actual GGUF header type or loader
compatibility.

The narrow hosted bridge makes Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, and Q8_0
selectable. The server binds Q5_K_M to GGUF file type 17 instead of reusing a
conversion-policy default. BF16, split GGUFs, and `mmproj` companions remain
visible with explicit reasons. Catalog resolution transfers no model payload.
Transfer admission reuses the catalog classifier instead of maintaining a
second quant allowlist, and activation compares the downloaded GGUF header to
the selected exact file type before model loading.

### Receipt-backed local artifact discovery amendment (2026-08-21)

The hosted-only boundary above was correct for legacy cache entries, but it is
not correct for every artifact hf2q now produces. Schema-v3 conversion
receipts bind the exact Hugging Face repository, source revision, emitted byte
length, output SHA-256, converter identity, and quant selector. Modern
`ModelCache` quant entries likewise bind a canonical managed path, byte length,
output SHA-256, and quant. Those are sufficient authorities for discovery as
long as activation revalidates the bytes rather than trusting a filename.

The server therefore advertises the additive
`hf2q.local-artifact-resolution.v1` capability and exposes authenticated
`GET /hf2q/v1/models/local-artifacts`. Hosted
`hf2q.artifact-resolution.v2` remains unchanged so old clients and servers
retain their current behavior. The local route accepts an optional bare
repository filter; without one it supplies the initial picker for an empty
chat-owned server.

Local means local to the server, never to an arbitrary HTTP client. The
inventory examines only the server startup directory's `models/` tree,
repeatable explicit `serve --model-dir DIR` roots, and the canonical hf2q
`ModelCache` manifest. Roots, traversal depth, visited entries, receipts,
receipt bytes, candidates, public strings, and warnings are bounded. Roots and
descendants are rechecked without following symlinks. A cache entry is eligible
only when its path equals `cache_model_path(root, repository, quant)`; a
manifest cannot grant authority to another path.

For a conversion receipt, the sibling `<artifact>.gguf.receipt.json` names the
candidate. Its recorded `output.path` is relocatable evidence and is never
dereferenced. The sibling must be a regular non-symlink file inside the
configured root with the recorded size. Cataloging performs a cheap supported-
quant/header preflight but deliberately does not hash every potentially large
artifact. Unsupported local selectors such as BF16 remain visible as disabled
rows with the current loader limitation rather than disappearing.

Paths and output digests never cross the HTTP catalog boundary. The server
retains them behind the same bounded, ten-minute opaque candidate authority
used by hosted selection. Receipt-backed local rows precede managed-cache rows;
both precede hosted rows. Duplicate output digests are deterministic, with a
schema-v3 receipt outranking matching cache metadata, while conflicting size or
quant claims for one digest fail closed. Public rows contain only repository,
revision, basename, bytes, quant hint, origin, role, selectability, and reason.

Resolution order is resident, receipt-backed local, managed cache, then hosted.
`--quant` and `--artifact` return immediately for one local match and perform
zero Hub work. Local ambiguity fails before Hub access. Without a selector,
the picker presents local rows and an explicit `Browse hosted artifacts` row;
a Hub outage cannot make an already-local candidate unusable. Missing paths
such as `./models/missing.gguf` remain paths and never become accidental Hub
repository requests. Every string received from an endpoint is escaped before
terminal rendering.

Admission still runs before expensive preparation. After a non-evicting plan
fits, or after the operator confirms an exact switch receipt, the server starts
one independently bounded direct-child verifier. It rechecks root containment,
regular-file and non-symlink status, byte length, complete SHA-256, supported
GGUF header type, quant identity, and the file snapshot before returning a path
receipt. The existing `PreparationSupervisor` owns cancellation and exact
reaping; `--no-integrity` never bypasses this candidate check. Pool state is
replanned before publication. A loaded engine retains its GGUF path so a later
catalog of the same local artifact returns the existing resident rather than
creating an alias.

The filesystem boundary is cooperative between processes running as the same
OS user. Re-stat and digest checks detect ordinary replacement before load, but
eliminating the final pathname race completely would require an fd-bound
`mlx-native` loader. hf2q does not claim protection from a same-user process
actively rewriting model files during activation.

When one repository exposes multiple selectable hosted GGUFs, interactive
chat shows a numbered picker. `--quant` is non-interactive only when it
identifies exactly one selectable candidate; `--artifact` names one exact
repository filename and wins no implicit fallback. They are mutually exclusive
and require `--model`. Ambiguity, EOF, declining the picker, admission conflict,
and stale switch confirmation transfer zero payload bytes. Source conversion
is outside this diagnostic selection flow and never silently outranks a hosted
GGUF.

Admission preflight uses the server-retained exact bytes before transfer.
Only after a no-evict plan fits, or after the operator confirms the exact
switch receipt, may hf2q fetch the selected filename at the pinned revision.
It revalidates exact Hub metadata, verifies size and LFS SHA-256, parses the
downloaded GGUF header, confirms that header matches the selectable hint, and
only then loads and warms it. Pool publication remains the final compatibility
test.

The hosted pool subject is the immutable
`hf://repo@commit/filename#sha256` identity, not merely the source repository.
This is a hosted-only identity bridge, not a global pool/cache type migration.
Activation returns that authoritative `request_model`; the TUI uses it for
every subsequent OpenAI request. Two resident hosted artifacts from one
repository therefore do not alias. Ordinary conversion cache and ADR-005
policy types remain unchanged.

Generic OpenAI endpoints retain opaque model strings and `/v1/models` only.
They never cause Hub catalog traffic, and artifact-selection flags fail with an
actionable hf2q-capability requirement.

Hosted catalog and transfer work runs in direct-child hf2q helpers supervised
by the server. Metadata helpers and transfer helpers each have an independent
two-child cap; catalog output, error output, candidate count, and candidate
lifetime are bounded. The transfer helper, not the HTTP handler, owns the
payload operation until it is explicitly killed and reaped. Dropping an
activation request cancels preparation and never publishes an incomplete
artifact. Cancelling chat against an external server leaves that server alive;
cancelling chat against an owned server additionally invokes the private
parent-lifetime cleanup above.

Artifact transfer occurs outside the exclusive lifecycle admission gate so a
slow download does not block requests for resident models. The gate is
reacquired and admission is recalculated before publication. Once load or
explicit switch crosses its irreversible commit boundary, an AppState-owned
task finishes it to one consistent terminal state even if the client
disconnects. Server shutdown cancels preparation before HTTP drain and waits
for all supervised work; an HTTP-drain or supervisor deadline fails before any
normal engine snapshot or teardown can race the still-active work.

### Non-evicting admission and explicit switching

Ordinary OpenAI clients retain ADR-005's current request-time auto-swap
behavior. Diagnostic chat does not silently cause eviction.

`LoadedPool` gains one canonical pure admission calculation, reused by both
planning and insertion, with these results:

```text
Resident
FitsAlongside { projected_bytes }
WouldEvict { exact_lru_victims, projected_bytes }
Impossible { reason }
```

The estimate uses the pool's current GGUF-byte accounting. `FitsAlongside` is
not a physical-memory promise: it becomes proven only when a non-evicting
load and synchronous warmup succeeds. The non-evicting load path must never
invoke the loader when the plan requires eviction and must never publish an
unplanned victim.

hf2q exposes `GET /hf2q/v1/runtime` as a versioned capability/runtime view and
`POST /hf2q/v1/models/activate` as an authenticated model-activation action. A
normal activation returns immediately for a resident
model, attempts a non-evicting load when it fits, and otherwise returns a
conflict containing the exact candidate, pool revision, and victims. The
terminal may then offer an explicit `Switch to X` action. It never performs
that action merely because the user selected or typed a model.

The switch is coordinated above `HotSwapManager` because it crosses async
request and worker lifecycles:

1. take the exclusive model-admission gate and revalidate the user-confirmed
   pool revision and exact victim plan;
2. before any drain mutation, reopen the materialized candidate and preflight
   every model-allocation-free fact consumed by the selected family: GGUF extents,
   architecture/config/context/quant identity, required tensor names and
   shapes, exact native storage/route capability, chat template, tokenizer,
   and configured DWQ/projector sidecars;
3. mark those victims draining so no new unary, embedding, or streaming
   request lease can be acquired;
4. wait for every request lease, including the response-body lifetime of SSE,
   to reach zero;
5. run the existing pre-eviction KV-spill hook while each idle worker is
   alive;
6. call `Engine::shutdown` for every confirmed worker. The first shutdown
   call is the destructive boundary: continue attempting the complete victim
   set even after an error, then commit removal of all confirmed victims so
   the pool cannot advertise a potentially dead engine;
7. if any shutdown failed, return restart-required without loading a
   replacement; otherwise load and warm the requested model through the
   non-evicting path; and
8. fail closed on a stale plan or drain timeout, never loading the replacement
   while a victim may still own its model memory.

This preserves the existing serve process and endpoint. It does not kill a
manually started process.

Model swap is a first-class serving operation, not merely a diagnostic-chat
convenience. Its hardware proof must use two distinct physical artifacts and
the production revision-bound switch route. A second pool key for a symlink to
the same GGUF does not exercise model, tokenizer, template, cache, teardown, or
reload behavior and is not swap evidence. The required sequence is A -> B -> A
under a budget that permits either artifact alone but not both together. It
must prove one resident generation after every transition, a fresh generation
when A returns, exact deterministic A-result replay, bounded load-and-warm time
on both legs, process-RSS and host-wired-memory reclamation when moving from
the larger artifact to the smaller one, and absence of a double-residency peak.
macOS `footprint` output is retained as a diagnostic only: the 2026-08-22 spike
showed that its process physical-footprint charge can rise while both RSS and
host wired pages fall, so it is not an authority for current Metal residency.
This corrected gate covers pool-resident generative engines through their
native chat endpoint. It does not count `/v1/embeddings` as proof for the
dedicated BERT/Nomic subsystem: those models are process-global in the current
source, while `/v1/embeddings` against a pool-resident generative engine uses
that engine's last-state pooling path. Dedicated embedding-model lifecycle is
a separate required implementation and hardware gate, not evidence supplied
by this test. Family-specific cache and template state must never cross a
switch.

Likewise, the startup vision projector is process-global in the current
source. Strongly bound incompatible swaps fail closed, but a shape-compatible
external vision model without source/projector digests can inherit and execute
the wrong startup projector; projector weights and its vision cache are also
outside pool accounting. The required correction makes one resident
generation own an atomic text/projector pair, resolves and warms the complete
pair before publication, leases both from that same generation, and accounts
the pair's unique allocations and cache bytes in admission/eviction. This work
is an active correctness lane and cannot be credited to the text-only gate.

The runtime capability also advertises the exact
`x-hf2q-diagnostic-no-evict: 1` request header. After activation, diagnostic
chat sends that header on OpenAI chat requests. If another client changes
residency between activation and generation, request-time resolution returns
409 instead of reopening ADR-005's ordinary auto-eviction path. The JSON body
therefore retains the zero-hidden-parameter contract above. Clients without
the header retain ADR-005 behavior.

### Diagnostic telemetry

The final SSE chunk gains an optional `x_hf2q_timing` object populated from
the already-measured `StreamStats`. Fields remain optional because not every
family instruments every counter. The chat footer combines that object with
standard usage details and the hf2q runtime view to report, when available:

- model and finish reason;
- prompt, cached, output, and reasoning tokens;
- time to first token, prefill/decode duration and rate;
- GPU synchronization/dispatch counts; and
- current pool residency and budget.

Chat does not enable logprobs or calculate perplexity by default. The former
changes the current execution/cache path and the latter requires a corpus and
teacher-forced scoring, so neither is zero-interference diagnostic telemetry.

## Failure behavior

- A discovered record that does not resolve, pass HTTP verification, or
  answer before the bounded discovery deadline is ignored and reported when
  useful.
- A protected endpoint requires explicit `--url`. Automatic discovery never
  sends credentials to an untrusted DNS-SD candidate, and credentials are
  never copied into DNS-SD.
- A non-hf2q `--url` endpoint can chat but cannot perform hf2q model-pool or
  owned-process actions unless it exposes and passes the exact capability
  contract.
- An admission conflict never loads the candidate. A changed plan requires a
  new explicit confirmation.
- A model-allocation-free candidate preflight failure happens before draining and
  leaves the old generation callable. Once worker shutdown begins, an error
  removes the entire confirmed victim set, leaves no dead engine advertised,
  and returns restart-required without a second physical load. A drain timeout
  leaves the victim unavailable for new leases and is likewise
  restart-required.
- A malformed, error-finished, or incomplete SSE response remains visible but
  is not appended to the transcript.

## Acceptance gates

Implementation is not complete until all of the following are proven:

1. pure pool tests cover resident, fits, exact multi-victim LRU order,
   impossible admission, no mutation, no loader call on conflict, and
   prediction/execution identity;
2. lifecycle tests cover unary, embedding, queued work, SSE-body lease,
   stale-plan rejection, missing-required-tensor and wrong-native-type
   preflight failures that preserve callable A, spill-before-shutdown,
   commit-after-shutdown, a two-victim first-dead/second-error removal, and
   timeout fail-closed behavior;
3. router tests prove authentication, capability discovery, non-evicting
   activation, conflict receipts, and explicit switch semantics;
4. SSE tests prove optional timing on the terminal chunk while all existing
   OpenAI chunk fields remain compatible;
5. client tests fragment bytes and UTF-8 arbitrarily, reconstruct reasoning
   and tool calls without execution, preserve only successful transcript
   turns, and prove the exact second-turn request;
6. real macOS tests prove LocalOnly registration/browse/resolve/removal,
   simultaneous instances, name collision handling, and HTTP verification;
7. subprocess tests prove normal owned-server shutdown, abnormal parent-
   lifeline EOF, leader-exit descendant cleanup, detach/keep-serving with a
   durable log, and that a pre-existing server is never stopped;
8. a real hf2q model test proves multi-turn unary/SSE compatibility, TUI
   tool-call display, direct-API tool-result continuation, prefix-cache reuse,
   timing/usage, and unchanged model output under matched settings. The direct
   API proves the agentic serving contract; the diagnostic TUI remains a
   display-only client and never becomes a tool harness;
9. real TCP tests prove request disconnect kills and reaps the exact hosted
   helper while an external server remains healthy, server-root cancellation
   reaps preparation before HTTP drain, and pipe-retaining descendants make
   shutdown fail closed; and
10. hosted-selection tests prove immutable candidate binding across mutable
    branch changes, bounded metadata/transfer concurrency, exact Q5_K_M file
    type 17 selection, BF16/split/mmproj rejection, zero transfer on conflict
    or stale switch, authoritative request identity, and no implicit
    safetensors conversion; and
11. local-selection tests prove schema-v3 and canonical-cache discovery,
    bounded/symlink-safe traversal, no path or digest serialization, local-first
    selector behavior with zero Hub work, repository-bound opaque authority,
    post-catalog digest/header rejection before loading, verifier cancellation
    and reaping, and successful activation of a real hf2q-produced GGUF; and
12. a real macOS pool-resident generative test executes the exact 13-phase
    sequence in `data/generative_swap_matrix.v1.json` twice through one
    long-lived server process: Qwen dense -> DeepSeek -> Qwen MoE -> DeepSeek
    -> Gemma -> DeepSeek -> Qwen dense, then the same three spokes again. All
    four artifacts are distinct physical files. DeepSeek is the eviction hub,
    and the largest-artifact byte budget makes every adjacent pair require
    replacement. Every response must join its family-specific semantic canary
    and execution receipt to the fresh resident generation, expected
    load-time SHA-256, and exact GGUF architecture; report zero cached tokens
    for that generation; retain the same process and model policy; prove the
    evicted mapping is absent; keep load-and-warm under the fixed 60-second
    bound; remain inside independently recomputed RSS and host-wired memory
    ceilings; and reproduce each returning family's deterministic result
    exactly. A source/mutation pass is not this hardware result; and
13. dedicated BERT/Nomic A -> B -> A tests prove model/tokenizer/registry
    replacement, exact embedding replay, generation isolation, memory
    reclamation, and no double-residency peak; and
14. multimodal A+P_A -> text B -> C+P_C -> A+P_A tests prove the projector and
    vision cache are generation-owned, admission-accounted, reclaimed with the
    text engine, and never inherited by an unbound shape-compatible model.
15. the sealed Qwen3.8 exact-artifact gate executes every directed A -> B -> A
    diagnostic row in `data/qwen38_exact_swap_matrix.v1.json` and, as the
    runtime authority, two complete
    BF16 -> Q4_K_M -> Q5_K_M -> Q6_K -> Q8_0 -> BF16 cycles in one long-lived
    server process. The build must embed the clean source commit and expose it
    through `__build-info`; the verifier joins that commit, the executed binary
    SHA-256, the crates.io `mlx-native` version/checksum, and all five immutable
    artifact identities. Every activation must publish a fresh resident
    generation, execute on that generation, return the exact
    `HF2Q_SWAP_OK` assistant message with `stop`, report a nonzero completion
    token count and zero generation-local cached tokens, complete load-and-warm
    in less than the fixed ten-second budget, and prove eviction plus bounded
    RSS and host-wired peaks. The verifier independently recomputes every
    bound from measured endpoints, including replay bounds; coordinated
    inflation of a claimed peak and its claimed bound is a failure. Both BF16
    cycle returns must reproduce the original result and configuration. The
    sealed matrix must be embedded by digest and value in the protected cache-
    lifecycle manifest and accepted by its dedicated final verifier. Missing,
    unsealed, pairwise-only, or unmanifested evidence is a failure.

## Validation evidence

### Original diagnostic-chat landing (2026-08-20)

The evidence in this subsection validates the original TUI, discovery,
generation, and explicit-switch implementation. It predates the hosted-GGUF
selection and abnormal-exit correction below and is not evidence for those
new boundaries.

The implementation candidate was frozen at `477172af` after two independent
review findings were reproduced and fixed. After integrating the current
`origin/main`, the exact validation tree was `b9c64c34`:

- replacement loading previously ran synchronous model warmup inside the Axum
  Tokio runtime. A nested-runtime sentinel test now proves the complete switch
  route loads off-runtime, and the real Qwen-to-Gemma switch completed without
  a panic;
- automatic discovery previously attached `HF2Q_AUTH_TOKEN` to an untrusted
  LocalOnly candidate. Real loopback HTTP tests now prove both `/health` and
  `/v1/models` probes carry no authorization header, and authenticated
  automatic discovery fails closed until the operator supplies `--url`.

On 2026-08-20 the release M5 Max host (macOS 26.5, arm64, 128 GiB) produced the
following source-bound evidence:

- `cargo check --locked --all-targets --all-features` and
  `cargo build --release --locked` passed;
- the serial full-bin suite passed 4,807 tests with zero failures and 53
  explicitly hardware/fixture-gated ignored tests; focused chat, lifecycle,
  router, SSE, multi-model, and discovery tests also passed, including every
  possible single SSE split boundary plus byte-at-a-time delivery;
- Agentic-QE SAST and both Claude-Flow full/input-validation scans reported
  zero findings;
- `hf2q chat` found no server, spawned the current release binary on port
  `61856`, activated the 16,810,714,752-byte Qwen3.8 27B Q4_K_M artifact,
  completed two streamed turns, reused 61 of 124 prompt tokens on turn two,
  displayed reasoning/usage/timing/pool telemetry, changed thinking mode, and
  gracefully stopped only that owned child on exit;
- a manually launched server remained PID `21648` on port `61947` while an
  explicit, revision-bound switch removed Qwen and loaded the
  20,576,631,488-byte Gemma4 Ara Q5_K_M artifact under a 24,700,000,000-byte
  pool budget. Chat returned `GEMMA_OK`, reported pool revision 3, identified
  itself as external, and left the manually launched server running;
- matched direct-API tool requests produced one structured `read_file` call.
  Exact unary replay reused 123 of 123 prompt tokens, SSE emitted the same call
  and one terminal `[DONE]`, and a role=`tool` continuation reused 130 of 240
  prompt tokens before returning exactly `HF2Q_CHAT_AGENTIC_OK`; and
- matched normal and `x-hf2q-diagnostic-no-evict: 1` requests produced equal
  OpenAI choices with the exact content `HF2Q_HEADER_PARITY_OK`.

Two failed spikes remain evidence rather than accepted claims. The repository's
7,102-token agentic fixture correctly exceeded Gemma SerialFifo's documented
4,096-token transaction bound. A reduced legacy harness then changed
`tool_choice` from `required` to `auto` and observed zero reused tokens, so the
final cache proof held request settings constant instead of treating a changed
grammar contract as the same prefix.

### Hosted-selection and Ctrl-C RCA correction (2026-08-21)

The reported failure was one causal chain. Diagnostic activation of a bare Hub
repository entered ADR-005's synchronous source-conversion path, so it began a
safetensors download. Ctrl-C terminated chat before its normal owned-child
cleanup. The server received shutdown too, but Axum retained the in-flight
blocking activation for its drain window, and that server inherited the
terminal's stderr. The surviving download therefore continued repainting the
restored shell prompt.

The correction replaces that chain rather than masking its output: diagnostic
Hub activation is conversion-free and selects one immutable hosted GGUF;
preparation is cancellable and explicitly reaped; owned servers use a verified
private process-group lifeline and durable log; transfer does not hold the
admission write gate; irreversible lifecycle transactions finish under server
supervision; and shutdown returns before engine teardown if HTTP or supervised
work has not reached a terminal state.

### Owned-startup telemetry amendment (2026-08-24)

Keeping the owned child's arbitrary stderr private is still mandatory, but a
generic periodic heartbeat does not tell an operator whether hf2q found local
bytes or began a transfer. The child therefore receives a third inherited
private descriptor: a nonblocking Unix datagram pair used only for bounded,
typed startup events. It reports local search/candidate/verification,
metadata-only Hub lookup, hosted selection, native conversion, text
load/warmup, projector load/warmup, and text-only projector fallback. The
parent renders those events as one scrollback-safe live row plus durable
milestones, or stable non-TTY lines. Only measured completed/total byte counts
produce a byte bar and ETA.

This telemetry channel is deliberately non-authoritative and fail-open.
Unknown, invalid, overlong, malformed, and oversized frames are ignored;
closed or backpressured presentation never delays model work. One parent tick
drains at most 32 events. The event schema has no READY variant. Endpoint
authority remains exclusively on the separate lifeline stream, must match the
retained listener, and becomes operator-visible readiness only after `/health`
and `/v1/models` verify over loopback. The server's private error log remains
private and is not replayed through telemetry.

The first small real-artifact spike then exposed a separate Qwen3.5 inference
defect: the selected hosted ggml-org 0.8B Q8_0 GGUF legally ties its output
projection to `token_embd.weight`, while hf2q required a separate
`output.weight`. The artifact catalog had correctly made no compatibility
claim; native load was the final compatibility gate. ADR-013 now records the
family correction and explicit buffer-sharing contract.

Load failures also crossed an unsafe diagnostic boundary. Ordinary
`anyhow::Error` display hid the causal leaf, while returning the full chain
would expose operator paths or credentials through HTTP. The server now emits
typed, allow-listed public diagnostics (such as a bounded missing tensor name),
keeps arbitrary context only in its private log, and retains and reports only
that log's path whenever an owned chat session fails. Successful non-detached
sessions still delete the temporary log; external endpoints confer no log or
process authority. This preserves local postmortem evidence without undoing
the HTTP redaction boundary at the terminal.

Focused fail-first and regression coverage now includes fragmented client
protocol, exact candidate binding, bounded catalog/transfer slots, the Q5
closed boundary, admission and post-commit classification, real-TCP request
drop to helper reap, server-root helper cancellation, pipe-retaining child
failure, leader-exit process-group cleanup, and generated zsh completion.

The reconciled source implementation candidate is `7bd89799`, based on
`origin/main` `84384d65`; the only later change is this ADR-only validation
receipt. On the macOS 26.5 M5 Max release host:

- focused correction suites passed 37/37 chat tests, 24/24 Qwen3.5 model
  tests, and 15/15 MTP tests. These include one combined owned activation-500
  test that receives the safe HTTP detail, shuts down and reaps the exact
  process group, retains the private log, and proves a fake credential and
  private path are absent from the terminal error; a separate startup test
  proves the same path-only boundary before discovery succeeds;
- the exact rebased tree passed the locked all-targets/all-features check,
  `cargo build --release --locked` with zero warnings, and
  the full single-threaded `cargo test --locked` workspace gate. The latter
  included 51/51 library tests, 4,928 passed binary tests with zero failures
  and 55 explicitly ignored hardware/fixture tests, every integration target,
  and doc tests;
- `cargo audit --file Cargo.lock` found zero vulnerabilities. It reported the
  three already-allowed unmaintained-dependency warnings for transitive
  `bincode` and `paste` paths through `ruvector-core`, `tokenizers`, and
  `mlx-native`; this change introduced none of them;
- the exact hosted artifact was
  `ggml-org/Qwen3.5-0.8B-GGUF` revision
  `8fea620810c4afa23dd6443f999a48574c1611a3`, file
  `Qwen3.5-0.8B-Q8_0.gguf`, 833,592,096 bytes, SHA-256
  `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`.
  Its immutable `hf://` identity, not the mutable repository name, became the
  pool and request identity;
- direct native generation from those bytes returned exactly
  `HF2Q_TIED_OK`. The pinned peer at
  `521a64cd01979bb5b1a466152c576a9d809b068d`
  returned the identical content from the same file, prompt, 16-token limit,
  greedy decoding, and reasoning-off settings;
- a real owned TUI session selected and loaded that hosted Q8 without entering
  safetensors conversion, reported 795.0 MiB resident, answered `ALPHA` and
  `BETA`, and reused 119 of 141 second-turn prompt tokens. TTFT moved from
  55.0 ms to 18.1 ms; `/status` reported the exact model identity, pool
  revision, and owned lifecycle, and `/quit` left no child or listener;
- Ctrl-C during an active 8,192-token SSE generation terminated chat and the
  owned server group immediately. No server, helper, or listener remained on
  the advertised port, and redirected child progress did not repaint the
  restored terminal; and
- `--keep-serving` retained a 0600 log, detached PID 89302 on port 52197, and
  returned from chat without signaling it. `/health` remained 200 until the
  operator explicitly requested `/shutdown`, after which the process exited;
- a real invalid-artifact activation returned the typed safe 400, then
  stopped the owned server and retained a 0600 private log. The terminal
  displayed only its path and the safe HTTP message; no process remained.

These receipts close the correction's complete local Kata gate. Publication
and merge remain subject to the repository's exact-commit GitHub checks.

### Receipt-backed local selection correction (2026-08-21)

The reported local-selection failure had a separate root cause from hosted
selection: `chat --model owner/repository` called only the hosted catalog even
when the exact artifact and its schema-v3 receipt were already below the
server's `models/` directory. The obsolete design assumption was that local
cache records did not identify emitted bytes. The smallest spike against the
operator's current receipt disproved that assumption: it binds repository,
source revision, byte length, output SHA-256, converter identity, and quant,
while the sibling GGUF header independently binds its file type.

The fail-first receipt test returned zero candidates before the new inventory
was connected. The corrected implementation is based on `origin/main`
`ccfa4dc3` and produced the following evidence on the same arm64 128 GiB host:

- `cargo check --locked --all-targets --all-features`,
  `cargo build --release --locked`, and the full `cargo test --locked`
  workspace gate passed. Focused gates additionally passed 42/42 chat tests,
  10/10 local-inventory/verifier tests, and the generated-completion tests;
- adversarial tests reject symlink roots and artifacts, traversal, stale size,
  wrong repository, malformed receipt authority, noncanonical cache paths,
  digest or quant mutation after cataloging, and late-created or oversized
  input outside the bounded policy. HTTP serialization tests prove local paths
  and output digests stay server-private;
- the Claude-Flow full/input-validation security scan reported zero findings.
  `cargo audit --file Cargo.lock` found zero vulnerabilities and repeated only
  the three already-allowed unmaintained transitive `bincode` and `paste`
  warnings recorded by the preceding correction;
- a release server launched from `/opt/hf2q` with no model and no explicit
  model root discovered the real schema-v3 artifact for
  `jenerallee78/Qwen3.8-27B-Abliterated-SFT`: revision
  `08c2f075b43bc06456382db6b918a3dcabdcf4dd`, basename
  `Qwen3.8-27B-Abliterated-SFT-Q4_K_M.gguf`, 16,810,714,848 bytes, and selector
  Q4_K_M. Its public JSON contained an opaque candidate id but neither the
  `/opt/hf2q` path nor output SHA-256;
- the real TUI displayed that artifact first as `[local hf2q] Q4_K_M`, followed
  by one explicit `[hosted] Browse hosted artifacts` action. With
  `--quant Q4_K_M`, selection reached local integrity verification without a
  Hub catalog request or safetensors transfer;
- Ctrl-C during full-file verification of that 15.66 GiB candidate exited chat,
  reaped the `__verify-local-gguf` child, left the manually launched server
  healthy, and left no test listener after explicit server shutdown; and
- after the concurrent ADR-046 source-teacher process exited, a fresh
  no-model release server completed the same local selection and full SHA-256
  verification, loaded all 64 Qwen layers through `mlx-native`, and published
  pool revision 1 with exactly 16,810,714,848 resident bytes. Thinking-off chat
  returned exactly `LOCAL_OK` (17 prompt, 2 completion tokens; 211.0 ms TTFT;
  80.6 prompt tok/s; 27.1 decode tok/s). `/status` reported the opaque
  `local://` request identity and external lifecycle. `/quit` left that
  manually launched server healthy and resident; explicit Ctrl-C then stopped
  the server and left no listener or model process.

### A -> B -> A proof correction spike (2026-08-22)

Source inspection invalidated the old `tests/multi_model_swap.rs` claim before
new implementation work began. It admitted a differently named symlink to the
same GGUF alongside the original engine, expected two resident pool entries,
and never evicted or reloaded A. That was alias admission, not model swap.

The first replacement spike used the production explicit-switch route with the
20,576,631,488-byte Gemma artifact and the 16,810,714,944-byte Qwen artifact
under a 20,576,631,488-byte pool budget. A -> B loaded in 2.64 seconds and
B -> A in 3.31 seconds. The deterministic A response replayed exactly; process
RSS fell from 29,735,092,224 to 24,981,094,400 bytes before returning to
29,820,174,336 bytes. The spike nevertheless was not acceptable: logs proved
startup A used `SlotAware { max_slots: 4 }` while reloaded A silently used
`SerialFifo`. Both diagnostic activation and ordinary request-time loading had
reconstructed an incomplete `EngineConfig` with a hard-coded scheduler.

The same run falsified process `footprint` as a Metal-residency authority. A
later fail-first run charged B 59,033,797,192 physical-footprint bytes even as
RSS fell to 22,028,140,544 bytes and host wired memory fell from
35,143,843,840 to 7,180,435,456 bytes. The corrected gate therefore samples
process RSS and `vm_stat` host wired pages during both switch transactions to
bound transient double residency; it records `footprint` only for diagnosis.
The engine configuration correction stores one process-wide dynamic-load
template plus canonical-artifact-and-filesystem-stamp overrides for explicit
tokenizer, config, overlay, and projector policy. A replacement at the same
pathname cannot inherit the old artifact's sidecars. Resident lookup compares
only the inexpensive file stamp; it does not hash the multi-GB GGUF again. A
stamp mismatch fails closed to the process template with every model-local
path cleared. The local runtime view continues to expose only a path-free
configuration identity.

The post-fix hardware gate passed on the same M5 Max with both reloads using
`SlotAware { max_slots: 4 }`. Gemma -> Qwen loaded in 2.277 seconds and
Qwen -> Gemma in 4.249 seconds. Process RSS was 29,743,431,680 bytes for the
first Gemma generation, 22,017,998,848 bytes for Qwen after eviction, and
30,837,932,032 bytes for the reloaded Gemma generation. Host wired memory was
35,074,883,584, 7,182,794,752, and 35,095,085,056 bytes respectively. The
sampled transition peaks were 29,743,513,600 / 35,074,899,968 bytes (RSS / host
wired) for Gemma -> Qwen and 31,998,066,688 / 36,528,816,128 bytes for
Qwen -> Gemma, both within the gate's no-double-residency bound. The reloaded
Gemma engine published a fresh generation, retained the exact path-free engine
configuration identity, and reproduced its deterministic response exactly.

That run remains lifecycle and replay evidence, but its inference assertion
accepted an HTTP 200 plus a separately sampled pool row. Source review on
2026-08-23 showed that those observations did not prove the response executed
the sampled generation. Successful unary and SSE chat responses now carry
private `x-hf2q-execution-*` headers derived from the actual leased
`LoadedEngine`: base64 pool key, generation, the already-retained load-time
text-artifact SHA-256, coarse architecture family, and exact GGUF
architecture. The response path compares the lease and engine identities
before inference and never re-hashes resident weights. The hardware swap gate
must join every A1/B/A2 response receipt to the contemporaneous resident row
and expected artifact digest/architecture; HTTP success alone is invalid.

At the end of the 2026-08-22 correction, the proposed
`data/generative_swap_matrix.v1.json` contract used four separate directed
A -> B -> A pairs. It had no family-matrix hardware receipt. That design is
historical: source review showed that a max-artifact pool budget still permits
several smaller pairs to co-reside, and fresh-server rows cannot reveal
cumulative leaks or stale cross-family state.

### One-process universal generative swap contract (2026-08-23)

Commit `b2c760f8` replaces that proposed matrix with one fixed 13-phase chain:
Qwen dense -> DeepSeek -> Qwen MoE -> DeepSeek -> Gemma -> DeepSeek -> Qwen
dense, followed by the same three spokes a second time. DeepSeek is the
eviction hub. Its artifact is the largest pool budget and every adjacent
artifact pair exceeds that budget, so all twelve transitions must physically
replace rather than co-reside. One server PID across both cycles exposes
cumulative leaks, process-policy drift, stale generations, and family state
that fresh-server rows would hide.

The fail-closed receipt binds clean source, embedded binary commit and digest,
the exact crates.io `mlx-native` identity, four immutable artifacts, all 13
fresh generations, family-specific semantic canaries, zero generation-local
cached tokens, execution-to-residency joins, process policy, eviction mapping
absence, load time, and independently recomputed RSS/host-wired transition and
replay bounds. The returning family result must replay exactly. BERT and Nomic
remain explicitly excluded because acceptance gate 13 owns their separate
process-global embedding lifecycle.

The structural and mutation gate is checked in. No 13-phase Apple-Silicon
artifact receipt has been produced yet, so this is source authority only, not
universal runtime acceptance or a performance claim.

The first 2026-08-25 hardware spike reached Qwen-dense admission and correctly
exposed an input-contract mismatch: the chosen text pathname had an automatic
sibling projector, so the pool charged text bytes plus the projector's mapped
physical bytes and full vision-cache reservation. Gate 12 is deliberately the
text-only generative-family chain; gate 14 owns projector lifecycle. The gate
now rejects projector/other GGUF siblings before model load, and qualified
text artifacts are presented through isolated same-inode hard links. This
preserves the immutable text identity while preventing a local directory's
optional sidecar layout from silently changing Gate 12's resident-byte and
eviction contract.

The corrected text-only spike passed the initial Qwen phase, then the first
DeepSeek activation took 202.787 seconds and exceeded the unchanged 60-second
client-observed switch bound. A live process sample placed that interval in
`TextArtifactIdentity::inspect` after the Qwen victim had already shut down;
the subsequent DeepSeek mapping and synchronous warmup took about 14.8
seconds. This was not a 107 GB model-load limit. The universal runner had
omitted the already-implemented schema-v2 receipt directory used by the exact
Qwen matrix, so the server correctly fell back to a full content hash during
activation.

The reformulated universal contract now creates or reuses one schema-v2
receipt for each of its four exact artifacts before the server starts, admits
the closed receipt directory through
`HF2Q_MODEL_VERIFICATION_RECEIPT_DIR`, and seals all four receipts in the
evidence manifest. The server remains the nanosecond-stamp authority and each
transition is still timed end to end against 60 seconds; no authentication
time is subtracted and no second runtime identity mechanism is introduced.
The shell/static/mutation battery is green. A fresh 13-phase hardware receipt
remains required before Gate 12 is accepted.

The first exact-candidate invocation then stopped before hashing or model
admission because the runner inherited an operator Cargo configuration that
its own exact-source policy forbids. The runner now owns an absolute isolated
Cargo home outside both source and evidence trees, rejects configuration there,
and passes it to every build, metadata, and test command. This preserves the
existing no-ambient-patch authority instead of requiring operators to alter
their normal Cargo setup.

The next exact run proved the preverified DeepSeek transition was inside its
60-second budget, then failed at DeepSeek -> Qwen-MoE on host-wired memory.
Diagnostic-only replay measured a 115,267,158,016-byte peak against a
115,271,286,784-byte pre-switch DeepSeek endpoint and a 7,777,222,656-byte
post-switch endpoint. The old destination-only formula produced a
34,967,713,792-byte bound and therefore rejected the already-resident source
at sampler start. The corrected source/destination-symmetric formula below
admits either endpoint plus margin or one destination above the lower endpoint,
while a simultaneous 107.4 GB source plus 25.0 GB destination remains over
bound. The model-free source-dominant mutant and independent receipt arithmetic
are green; the complete hardware chain reruns from a fresh exact candidate.

That rerun passed Qwen-Dense, Qwen-MoE, and DeepSeek, then exposed a distinct
proof-harness defect at the Gemma phase. `lsof` proved the evicted Qwen-MoE
artifact closed, while the `vmmap` fallback called it live only because it
searched the basename `APEX-Q5_K_M.gguf` as a substring of the current
`gemma4-ara-2pass-APEX-Q5_K_M.gguf` filename. The probe now matches the exact
canonical path as a complete trailing `vmmap -wide` field. A focused mutant
proves that the exact Qwen path matches and the longer Gemma path does not;
the eviction invariant itself remains fail-closed. The full chain still must
pass from a freshly built exact candidate before this hardware gate closes.

The corrected runtime then completed all thirteen phases and twelve forced
family replacements in one production process. Its independent validator
correctly refused to seal the receipt because two older proof assumptions had
drifted from the production telemetry: it required `serial_fifo` even though
the process reported the required `slot_aware` policy with four slots, and it
required strictly positive process-wired bytes even though macOS `footprint`
legitimately reported zero after several Metal-backed replacements. The
authority now requires `slot_aware` with at least four slots, keeps RSS,
physical-footprint, and host-wired measurements strictly positive, and accepts
process-wired bytes only as a nonnegative integer. New mutations reject serial
fallback and negative wired data. The captured runtime validates under that
reformulated model-free authority; the final exact-lineage rerun remains the
hardware seal.

The exact-lineage rerun passed at runtime commit `cb622acc`. One production
four-slot process completed all thirteen phases and twelve forced replacements
across Qwen3.8 dense, Qwen3.6 MoE, Gemma4, and DeepSeek4. Exact replay,
semantic canaries, fresh generations, process-policy stability, eviction,
mapping, and memory bounds all held. Switches measured 0.560691–4.838193
seconds with a 3.306021-second median under the immutable 60-second ceiling.
The independently revalidated matrix is
`/opt/hf2q-evidence/universal-release-cb622acc-generative-swap/matrix.json`
(SHA-256 `86e6739c476b9a4c0fc3ea2e28dced0fca5650d4ca85a1f6ff6ea1d434026bd6`).

### Exact Qwen3.8 cross-format swap matrix contract (2026-08-23)

The prior Qwen3.8 artifact catalog and four-position gate proved immutable
storage identity and per-artifact inference independently. The corrected
Gemma -> Qwen -> Gemma run proved the pool lifecycle independently. Neither
receipt proved that all five contracted Qwen3.8 formats can replace one
another without stale generation, route, mapping, or result state. Treating
those independent facts as a cross-format model-swap result would be a proof
composition error.

`data/qwen38_exact_swap_matrix.v1.json` seals a five-row directed cycle:
BF16 -> Q4_K_M -> BF16, Q4_K_M -> Q5_K_M -> Q4_K_M, Q5_K_M -> Q6_K ->
Q5_K_M, Q6_K -> Q8_0 -> Q6_K, and Q8_0 -> BF16 -> Q8_0. This is the smallest
complete matrix in which every contracted artifact is both an evicted/replayed
A and a loaded B. Those cells remain useful pairwise diagnostics, but they are
not cumulative lifecycle authority because each cell starts a new server. The
acceptance run additionally executes two BF16-hub cycles through the
production revision-bound switch route in one server process: BF16 -> Q4_K_M
-> BF16 -> Q5_K_M -> BF16 -> Q6_K -> BF16 -> Q8_0 -> BF16, then the same
four spokes again. A max-artifact pool permits adjacent smaller formats to
co-reside, so the earlier direct 11-phase sequence did not actually force
replacement and was rejected by the gate. The five isolated cells retain the
direct small-format edges; the 17-phase hub chain forces all 16 cumulative
transitions and makes sub-threshold leaks, stale generation state, and
cross-format residue visible.

Before the first load, the runner requires a clean exact source tree, rejects
ambient Cargo configuration, derives the crates.io `mlx-native` identity from
the exact manifest and lock entry, and verifies the size and SHA-256 of all
five immutable artifacts. It builds once with `GIT_COMMIT_SHA` set to the
clean checkout commit, or consumes the protected signed candidate, then
requires that exact executable's hidden `__build-info` receipt to report the
same commit. Binary SHA-256, embedded commit, source commit, dependency
identity, and artifact snapshots are rechecked through the run. The fixed
load-and-warm budget is ten seconds; no inherited environment variable may
raise it.

Each activation publishes a path-free receipt joining the leased pool-key
digest, generation, artifact digest, architecture, resident bytes, process
identity, memory samples, and switch peaks. Generation coherence is semantic,
not merely non-null JSON: greedy inference must return the exact assistant
sentinel `HF2Q_SWAP_OK`, finish with `stop`, report at least one completion
token, and report zero cached prompt tokens for the fresh generation. The
long-lived chain requires seventeen unique generations, stable per-format pool
and configuration identities, five distinct format pool identities, absence
of every evicted artifact mapping, and exact BF16 replay at both cycle ends.

The receipt may record bounds, but it is not authority for its own arithmetic.
The shell contract independently derives each transition and replay margin
from the endpoint RSS, process footprint, process wired, and host-wired
measurements and rejects both a peak above the derived bound and a coordinated
increase of peak plus recorded bound. Process `footprint` remains diagnostic;
RSS and host wired memory remain residency authorities. The outer receipt
embeds all five diagnostic cells and the long-lived chain, seals their logs,
and is then embedded by exact digest and value in the protected cache-lifecycle
manifest. A dedicated final verifier revalidates the seal against the exact
checkout before the aggregate manifest can pass.

An adversarial source review falsified six earlier proof assumptions: a binary
SHA did not prove the binary contained the named source; an inherited swap-
budget variable could relax the target; non-null output could be empty or
incoherent and the generation cache was unchecked; pairwise server restarts
could hide cumulative defects; the real runtime matrix was absent from the
protected final manifest; and self-authored memory bounds allowed coordinated
peak/bound inflation. The corrected source-only mutation suite now rejects
each of those cases, plus missing formats or rows, self-swaps, artifact or
execution-generation drift, BF16 replay divergence, changed process identity,
and evidence-log tamper. It is blocking in hosted CI without loading a model.
This establishes the contract and its failure semantics only; it is not a
runtime acceptance result.

At source commit `9768b3d0`, `Cargo.toml` required unpublished
`mlx-native = 0.12.3` and the lock entry had no crates.io source or checksum.
The production runner correctly failed closed before build or model load.
That is retained as historical blocker evidence.

The 2026-08-23 candidate resolved the blocker differently: `mlx-native 0.13.0`
was published from commit `1d9073a5d31565bee79bf99a516b8781cab0a284`, and
fail-closed workflow run `32685066084` verified its exact source, package,
registry archive, crates.io bytes, tag, and GitHub release bytes. hf2q commit
`1b1f3811` pins that registry package and checksum
`19bc89cd60cd6416ce9d562ac51c50851d88f345059331cce8a4baca15265356`.
The cross-format runner was therefore no longer registry-blocked at that
boundary. Its five hardware rows, two long-lived cycles, and final seal had not
yet executed; publication and pinning were prerequisites, not runtime
acceptance.

The current candidate now pins published `mlx-native 0.14.0` from exact commit
`32f076c7502151e7ca9cb20c06d0f3fe5e1d5641`; fail-closed workflow run
`32873363483` binds its source, packed crate, crates.io bytes, tag, and GitHub
release to SHA-256
`c7b359aa9ea2603f58b49151ba54e37ed1aac10e76faf530865ea30a95f051b4`.
This updates the reproducible dependency boundary only; the swap matrix still
requires its own exact-artifact execution seal.

### Single-hash swap-catalog correction under validation (2026-08-24)

The first exact-matrix hardware spike at hf2q commit `e323f878` failed before
its first BF16 -> Q4_K_M cell. The runner authenticated all five artifacts,
but the no-startup-model server had authority for none of those receipts.
Explicit-local activation then replaced any registry evidence with an
ordinary stamp-only binding, so the loader read the complete GGUF again. The
BF16 startup did not become ready for roughly 100 seconds and the following
Q4_K_M switch took 29.553 seconds, exceeding the immutable ten-second bound.
This was duplicate integrity work, not model initialization throughput, and
raising the bound would conceal the defect.

The reformulated contract records one schema-v2 receipt per exact physical
artifact in a bounded operator-owned directory before server launch. Server
startup admits the directory atomically: it rejects relative or symlinked
directories, empty or oversized catalogs, non-JSON or duplicate entries,
legacy receipts, and any stale member. Each admitted identity enters the
existing canonical-path plus full-file-stamp configuration registry. Both
activation actions preserve that identity when their explicit-local payload
does not carry a newer verified identity; same-path replacement still drops
to the process template and cannot inherit the digest or model-local policy.
The loader remains the final authority and revalidates the full stamp before
using the retained digest. No client-provided digest, path alias, or coarse
shell snapshot can authorize hash reuse.

The exact runner now creates or reuses those five receipts through the same
sealed binary recorder, passes their directory to every pairwise and
long-lived server, and seals the receipts in its evidence manifest. The first
post-correction attempt exposed a separate proof-integrity defect before its
first cell could be accepted: `cargo test --release` replaced the supplied
`target/release/hf2q` after the runner captured its digest. Startup reached
ready in six seconds, but the executed bytes correctly failed the exact-binary
assertion. The runner now copies the candidate to a private bounded temporary
directory before any integration-test compilation, attests and executes only
that immutable copy, and removes it on exit.

The next run passed four pairwise cells and then falsified the original
host-wired bound on Q8_0 -> BF16. Process RSS stayed near 4.4 GiB and both
settled host-wired endpoints were about 8.5 GiB, while the 100 ms system
sampler observed 60.88 GB during admission of the 54.66 GB BF16 destination.
That peak is consistent with one destination artifact becoming temporarily
wired; the rejected two-artifact case would add the 29.05 GB Q8_0 source and
cross roughly 92 GB above the same host baseline. Comparing a transient peak
only with settled endpoints therefore rejected legitimate one-model loading
and made the gate sampling-luck dependent.

The corrected universal arithmetic retains the strict process-RSS endpoint
bound. For system-wired memory it independently computes the maximum of
`before + margin`, `after + margin`, and
`min(before, after) + destination_artifact_bytes + margin`. The first two
terms admit either measured single-resident endpoint; the third admits one
destination becoming transiently wired above the lower host baseline. Source
plus destination double residency still exceeds the bound. Pairwise Qwen, the
long-lived Qwen chain, and the cross-family generative chain use the same
formula, and their contract validators recompute it from sealed artifact bytes
rather than trusting the receipt's bound.

The first 17-phase hub run then completed every transition and failed only the
old final host-wired replay ceiling, which still compared a potentially wired
BF16 generation with its settled initial endpoint plus 2 GiB. Replay now uses
the same independently recomputed one-resident rule: the lowest host-wired
phase baseline plus the replay artifact bytes plus margin. Process RSS,
process footprint/wired memory, semantic replay, mapping absence, and all
transition bounds remain unchanged; adding any evicted source artifact to a
fully wired replay still exceeds the ceiling.

The following exact run passed all five pair cells and all 17 long-lived
generations, then failed final sealing because the old seal allowlist did not
include the new `preflight/` receipt directory. The seal now requires exactly
five named schema-v2 receipts, binds each digest to its contracted format and
artifact suffix, rejects any extra/missing/symlinked entry, and verifies their
hashes through the evidence manifest. Runtime success without that evidence
closure remains a failed gate.

Commit `3c79d56f` with exact binary SHA-256
`c64e11ba3f70f0c400b7858d9c1b5a3be535c590b260bab7cfb3c139ef6e2ea8`
then passed and sealed the complete matrix. All five pair cells returned the
exact `HF2Q_SWAP_OK` semantic result at A1/B/A2 with zero cached tokens and
fresh generations. Nine of ten pair legs loaded in 0.368--0.391 seconds; the
cold Q8_0 -> BF16 leg took 9.332 seconds and remains inside, but close to, the
immutable ten-second ceiling. All 16 BF16-hub transitions in the single PID
took 0.361--0.383 seconds, produced 17 unique generations, retained five
stable format identities, removed every evicted mapping, and replayed BF16 at
phases 8 and 16. The sealed `matrix.json` SHA-256 is
`32a3327fa1c8372b88a58b88d67e0bdeaf50b90916932daeebb175e463eedf7e`;
its evidence manifest SHA-256 is
`be57f4d04e4ac773e179e0738b243f28104790353768be427e9cc4476dfc2b3b`.

This accepts the exact cross-format swap lane. The later provenance-bound
candidate at commit `9138cfaa`, binary SHA-256
`3a3339a327224ea73e00351853a8237c053ad7302bc3320ad52d75f4caccfb76`,
also sealed the five-format by five-width physical matrix with exact scalar
replay. Its matrix SHA-256 is
`d0f4d215a776a17a24cfc13df6c3ad09c3df9eed2ec835f589e8e1f2ecfc6800`.
The same binary passed the matched Q4_K_M one-slot ABBA gate at 1.053834x for
code and 1.159750x for exact repetition. The comparison pin then advanced, so
the current universal acceptance gate remains the all-format, all-width
matched-physical matrix built from the refreshed pin; no prior receipt is
silently relabeled as current.

## Consequences

The implementation adds a small terminal client and explicit operational
control plane, not a second inference stack. It also tightens model-pool
semantics by distinguishing estimated logical admission from proven physical
release. The lifecycle coordinator is intentional complexity: the measured
alternative can temporarily double-reside models while claiming the victim's
bytes are free, which is unacceptable for an explicit diagnostic switch.

Persistent chat history, automatic tool execution, a general request editor,
agent orchestration, evaluations, and opt-in logprob/perplexity analysis remain
out of scope.
