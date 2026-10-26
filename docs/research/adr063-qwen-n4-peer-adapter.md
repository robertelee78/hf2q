# Explicit latest-peer N4 coding diagnostic adapter

Source-only candidate based on HF1f9fda16. No tests, imports, builds, HTTP, model
or GPU execution by author. This is harness support, not production inference or
permission to launch. Existing HF mode remains default `--engine-kind hf2q`;
`--engine-kind peer` is explicit and restricted to the four-agent coordinator.
One externally owned server, four independent own-output conversations, seven
requests and six actual sandboxed Rust fixture tools each, transport matrix,
8192 output allowances, barrier, full final tests/source/patch oracles and
cleanup remain shared. Three tools retain strict:false and requests remain
image-free; no alternate compatibility wire or hidden prompt rewrite is added.

Peer source identity inspected: d854d98bae598388ecea1ed84396587e5edc326d,
with retained help `adr063-peer-d854d98b-help.txt`. Adapter source is not a binary
attestation: outer approval still binds exact source/build executable, model,
settings and host evidence. It never launches either engine or loads a model.

Peer readiness uses actual /health status=ok, /props total_slots=4,
default_generation_settings.n_ctx=262144 and model_alias, plus enabled /slots
with IDs0..3 and EVERY actual n_ctx=262144. Raw replies are saved. These fields
come from tools/server/server-context.cpp:797,5552–5572,5681; slot context is
llama_n_ctx_seq at4955, not presumed CLI total divided correctly. A peer launch
that yields65536 per slot is refused even if it passed --ctx-size262144. Bind
actual per-sequence semantics; do not copy HF --ctx flags as equivalent without
checking /props and /slots. Server must expose --slots for this diagnostic.
Readiness alone proves neither four active physical lanes nor workload cache
reuse. Actual physical-cohort trace remains Unavailable until separately captured.

Peer preflight replaces HF-specific profile24g/TQ8/GLP/native-registry assumptions
with explicit peer build/settings/template/generation/cache/readiness bindings.
No HF runtime endpoint or profile boolean is forged. Approval carries engine_kind
through coordinator/agents and rejects mixed-engine results. Peer source hashes
include the new adapter and its pure tests. All real sandbox/tool provenance
requirements remain; same coding fixture and workflow are used.

Peer tools/server/server-task.cpp:357–363 emits OpenAI usage totals and cached
prompt count. Shared unary/SSE parser routes explicitly to the peer usage
validator; response identity/tool/finish/SSE grammar checks remain shared. Raw
responses remain archived. Missing cached count stays null/Unavailable, never0;
reported0 stays observed0 and does not become positive reuse. Invalid negative,
boolean or >prompt counts and contradictory totals fail. Unlike HF's existing
positive-followup cache guard, peer protocol completion can retain unavailable
cache evidence; full_prefix_reuse remains PendingPostOracle and such a result
cannot be accepted as a qualified cache/performance win until the independent
prefix oracle closes it. Protocol/tool success is not renamed full model quality.
HF default usage validation remains unchanged and still rejects missing/zero
followup cache. This explicit difference preserves raw unknowns; it does not waive
qualification requirements for either engine.

The current wire includes hf2q_overflow_policy=reject; peer does not thereby gain
HF rejection semantics. Peer launch must bind its own context-shift/truncation
settings and generation controls. No missing phase-time/draft/queue/prefill/decode
metric becomes0. Outer recorder must retain peer timing raw fields/logs and
actual configuration (including optional projector disposition and caches).
Single-shot readiness is not a server-memory watchdog; root's external owned
process/memory/thermal supervisor remains required. This patch supplies no
performance statistics, peer-best tuning verdict, all-four result or F2 closure.

Six proposed pure adapter controls cover actual per-sequence readiness and
contradictions, missing cache preservation, invalid counts, engine-specific
preflight separation, and unchanged default HF usage. Existing unary/SSE/N4
controls still apply. Author performed only AST syntax parsing and diff hygiene;
root owns all tests and real execution after independent source review.

Root validation: 75 focused Python controls passed, including six peer-adapter
controls and the retained HF controls. Independent source review checked the
metadata/cache mapping against pinned peer source d854d98b. No peer inference,
cache qualification, model-memory admission or speed win is established here.
