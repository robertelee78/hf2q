# ADR-047: Diagnostic chat over the native inference server

- **Status:** Accepted; text-engine A -> B -> A proof passed; embedding and projector lifecycle execution active
- **Date:** 2026-08-20
- **Updated:** 2026-08-22 — exact-candidate embedding activation and the
  synthetic public A -> B -> A lifecycle gate are green; the converted real
  BERT -> Nomic-BERT -> BERT quality, mapping-reclaim, and latency receipt
  remains an acceptance gate and is not claimed by this revision
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
2. mark those victims draining so no new unary, embedding, or streaming
   request lease can be acquired;
3. wait for every request lease, including the response-body lifetime of SSE,
   to reach zero;
4. run the existing pre-eviction KV-spill hook while each idle worker is
   alive;
5. call `Engine::shutdown` and join each worker;
6. commit pool removal only after shutdown succeeds;
7. load and warm the requested model through the non-evicting path; and
8. fail closed on a stale plan or drain/shutdown timeout, never loading the
   replacement while a victim may still own its model memory.

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
native chat endpoint. It does not count `/v1/embeddings` as that proof:
pool-resident generative embeddings use the text engine's last-state pooling
path. The dedicated BERT/Nomic subsystem instead owns a separate
single-generation slot binding its mapped weights, tokenizer, vocabulary, and
warmed registry. It refuses replacement while a request lease is live, drops
the old generation before invoking the new loader, and remains
dedicated-but-unavailable after a load failure rather than falling through to
the generative embedding path. Model activation schema v3 adds an explicit
`kind: "embedding"` domain to the existing authenticated activation route;
omitting `kind` retains the v2 generative-pool behavior. The server never
infers this domain from a filename or GGUF architecture. `action: "load"`
returns a conflict receipt when another encoder generation is resident, and
`action: "switch"` requires that receipt's exact embedding generation through
`expected_revision`. The success receipt publishes the exact candidate
selection, new generation, architecture, logical resident bytes, reclaimed
bytes, and `load_ready`/`post_warm` timing milestones without exposing a local
path.
Artifact catalog queries accept the same explicit kind. Local hf2q conversion
receipts classify `bert` and `nomic-bert` artifacts as `embedding_model` and
retain the exact source revision, artifact SHA-256, GGUF file type, and private
path behind an opaque candidate id. This identity is deliberately not routed
through the narrower generative `QuantType` enum: doing so would reject valid
Q4_0/F16/BF16 encoder artifacts. The child verifier authenticates the digest
and exact file type before the public switch invokes the native loader.
Embedding load and switch require that opaque `candidate_id`; a bare local
path remains startup-only through `--embedding-model`. Catalog-driven
residency compares the exact published identity, so two artifacts with the
same basename cannot alias. A candidate-free probe may inspect the configured
startup path, but it has no mutation authority.
The runtime capability view reports the same slot state, including a
fail-closed load error or in-progress replacement. Its independent A -> B -> A
hardware gate must drive all three A -> B -> A activations through this public
route, prove exact embedding replay plus process-RSS and wired-memory
reclamation, and include conflict, load-ready, post-warm, first-semantic, and
steady-state latency. It must also prove that an exact-candidate load failure
leaves the dedicated route unavailable and that an explicit public recovery
publishes a fresh coherent generation. Direct calls to an internal swap helper
are not evidence. The generative switch gate is not evidence for it.
Family-specific cache and template state must never cross a switch.

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
- A drain or worker-shutdown timeout leaves the model unavailable for new
  leases and returns an actionable restart-required error. It does not risk a
  second physical load.
- A malformed, error-finished, or incomplete SSE response remains visible but
  is not appended to the transcript.

## Acceptance gates

Implementation is not complete until all of the following are proven:

1. pure pool tests cover resident, fits, exact multi-victim LRU order,
   impossible admission, no mutation, no loader call on conflict, and
   prediction/execution identity;
2. lifecycle tests cover unary, embedding, queued work, SSE-body lease,
   stale-plan rejection, spill-before-shutdown, commit-after-shutdown, and
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
12. real macOS pool-resident generative A -> B -> A tests use distinct GGUF
    bytes and exact conflict
    receipts, keep one resident generation, bound both load-and-warm legs,
    measure process RSS and host wired memory across each transition, reject a
    double-residency peak, publish a fresh A generation, and reproduce A's
    deterministic chat message exactly after reload; and
13. dedicated BERT/Nomic A -> B -> A tests prove model/tokenizer/registry
    replacement, exact embedding replay, generation isolation, memory
    reclamation, and no double-residency peak; and
14. multimodal A+P_A -> text B -> C+P_C -> A+P_A tests prove the projector and
    vision cache are generation-owned, admission-accounted, reclaimed with the
    text engine, and never inherited by an unbound shape-compatible model.

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
template plus canonical-artifact overrides for explicit tokenizer, config, and
overlay paths, and exposes a path-free configuration identity in the local
runtime view.

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
