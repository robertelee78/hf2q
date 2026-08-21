# ADR-047: Diagnostic chat over the native inference server

- **Status:** Accepted and implemented; landing remains CI-gated
- **Date:** 2026-08-20
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

DNS-SD registration is isolated behind an hf2q-owned module. Non-macOS builds
retain explicit-URL chat but do not claim automatic local discovery. DNS-SD
failure must be visible in logs and must not make the established HTTP server
unusable.

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
7. a subprocess test proves owned-server shutdown, detach/keep-serving, and
   that a pre-existing server is never stopped; and
8. a real hf2q model test proves multi-turn unary/SSE compatibility, TUI
   tool-call display, direct-API tool-result continuation, prefix-cache reuse,
   timing/usage, and unchanged model output under matched settings. The direct
   API proves the agentic serving contract; the diagnostic TUI remains a
   display-only client and never becomes a tool harness.

## Validation evidence

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
