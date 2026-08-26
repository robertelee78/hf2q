# FreeToken elastic model-parking spike — 2026-08-24

## Question

Can hf2q make an A -> B -> A model swap cheaper on Apple unified memory by
retaining A's immutable, file-backed weights and compiled pipelines while
releasing A's idle mutable runtime state at a drained worker boundary?

The spike is based on hf2q `0da66164`. It updates the implementation decision
recorded as an open hypothesis in ADR-049; it does not change that ADR's status
or make a hardware performance claim.

## Source-grounded finding

The current multi-model pool charges artifact and projector bytes. It does not
separately charge KV, recurrent, prefix, anchor, capture, or scratch ownership.
Therefore a worker park receipt is valid evidence of bytes released by that
worker, but is **not** pool-capacity credit until the same mutable bytes are
first added to the admission ledger. Subtracting a receipt from the current
pool total would create fictitious capacity.

Apple unified memory also changes the mechanism from FreeToken's discrete-GPU
pool resizing. hf2q cannot demote an arbitrary mapped Metal weight allocation
to a separate host tier. The reachable operation is narrower and useful:

1. stop admitting work to one engine generation;
2. drain its worker and atomically enter a parked state;
3. release registered mutable caches and scratch while retaining native mapped
   weights, their file mapping, native route activation, and pipelines;
4. commit deferred Metal residency removals before acknowledging the release;
5. reactivate as a fresh, zero-cache generation, or fully evict if immutable
   residents do not co-fit with the incoming model.

The relevant pool and lifecycle seams are `src/serve/multi_model.rs` and
`src/serve/model_lifecycle.rs`. The family-owned mutable state lives in
`src/serve/api/engine.rs`, `src/serve/api/engine_qwen35.rs`,
`src/serve/api/engine_deepseek4.rs`, and the architecture cache/scratch modules
under `src/inference/models/`.

## Implemented worker primitive

This spike implements the smallest coherent primitive, not the manager policy:

- `PrepareIdlePark` and `Reactivate` are serialized against every cloned
  request handle. Ordinary work fails closed while transitioning or parked.
- Park succeeds only after scheduler, slots, pending queues, transactions, and
  scratch pools prove drained. Every fallible preflight runs before ownership
  mutation.
- Receipts report checked `before`, `after`, and `before - after` values for the
  explicitly registered reclaimable set. Immutable weights and pipelines, and
  retained family-fixed activation state, are outside that set.
- Slot generations advance and stale scheduler handles cannot name the cold
  replacement state.
- Reactivation never restores prefix identity. Qwen and Gemma prompt, anchor,
  LCP, speculation, and slot KV authority are cleared. DeepSeek receives fresh
  sessions. Reactivation failure leaves the engine parked.
- A lost or timed-out park acknowledgement leaves the handle in the
  indeterminate transition state. It cannot serve or reactivate; lifecycle
  must shut down and fully evict it. This avoids racing reactivation against a
  delayed park failure on a worker that actually remained active.
- Qwen, Gemma, and DeepSeek SlotAware workers implement the contract. Qwen and
  Gemma SerialFifo workers implement it through their existing lazy cold-cache
  allocation. DeepSeek SerialFifo returns a typed unsupported result because
  its mandatory base cache has no cold reconstruction seam; lifecycle must use
  full eviction for that mode.
- Buffer drops only stage Metal residency-set removals. Each successful park
  creates its release encoder before mutation and commits after all drops, so
  lifecycle cannot admit B on an acknowledgement that has not yet flushed A's
  removals.

Exact registered ownership includes Metal buffer byte lengths and directly
owned host controls that disappear on park: outer cache vectors, per-slot
cursor vectors, anchor controls/payloads, token vectors where released, and
live DeepSeek cache plans. Retained control capacity is either reported in
`after` or deliberately excluded from both sides.

## Reformulated hypothesis

The original broad "elastic pool rebuild" hypothesis splits into two parts:

1. **Worker parking is possible and source-coherent.** The implementation has
   a fail-closed protocol and exact release receipts without changing model
   arithmetic.
2. **A -> B -> A is faster and fits more useful model pairs** remains a hardware
   hypothesis. It requires manager integration so admission reserves the
   complete incoming generation before destructive action, uses full eviction
   when immutable residents do not co-fit, publishes a fresh generation on
   reactivation, and never routes requests to a parked entry.

The manager must also cold-barrier any disk LCP hydration and keep vision
projectors generation-bound. Those are lifecycle responsibilities, not reasons
to weaken the worker primitive.

## Proof at this commit

Model-free gates:

```text
cargo check --locked --all-targets --all-features
cargo test --locked --bin hf2q idle_park_rejects_work_until_cold_reactivation
cargo test --locked --bin hf2q idle_runtime_reset
cargo fmt -- --check
git diff --check
```

The transition test proves park -> ordinary-work rejection -> cold reactivation.
The scheduler tests prove drain preflight, generation invalidation, and removal
of released arena high-water.

## Required hardware decision

After lifecycle integration, run one checked-in A -> B -> A gate on each
applicable generative family. For A1, B, and A2 bind the response to the active
generation and artifact identity; prove A2 is a zero-cache generation, A1/A2
greedy output and configuration are identical, no request reaches a parked
entry, and physical/RSS/wired measurements show no forbidden double-residency
peak. Report park receipts separately from pool charges and OS residency.

Compare full eviction/reload with retained-weight parking using matched
artifacts, prompts, settings, and repeated medians. The implementation earns a
shipping performance claim only if the measured swap latency or admissible
pair envelope improves without a coherence regression.
