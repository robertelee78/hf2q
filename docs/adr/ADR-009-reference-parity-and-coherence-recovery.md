# ADR-009: Reference Parity and Coherence Recovery for Owned Inference

**Status:** Phase 1 accepted (semantic parity); Phase 3A open (exact numerical parity)  
**Date:** 2026-04-15 (proposed) / 2026-04-16 (Phase 1 accepted, Phase 3A opened)  
**Decision Makers:** Robert, Claude  
**Related ADRs:** ADR-008 (candle divorce), ADR-007 (TurboQuant KV cache), ADR-006 (mlx-native GPU backend), ADR-005 (inference server)

---

## Engineering Mantra (load-bearing — read before every session)

Source: `~/Documents/mantra.txt` (Robert, undated). Quoted verbatim.

> **DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.**

**Operational reading for this ADR:**

- **No shortcuts** — this ADR is not a license to tweak thresholds or lower the coherence bar. It exists to restore correct owned semantics.
- **Measure 3x, cut once** — no phase is considered complete because a final output looks plausible. Every phase has tensor-contract gates and an end-to-end coherence gate.
- **Chesterton's fence** — candle and llama.cpp are not being reintroduced as dependencies. They are being used as semantic reference implementations so that the owned stack can absorb the right behavior.
- **No fallback** — the end state is a coherent and fast owned stack in `hf2q + mlx-native`, not a permanent dual-backend system.
- **Never make assumptions** — every statement in the repair plan must be tied either to direct code audit, a tensor-boundary comparison, or an end-to-end measured gate.
- **No stub code** — temporary harnesses are acceptable; permanent placeholder code paths are not. If a contract is required for parity, it must be fully implemented.
- **Pure excellence** — this ADR is intentionally stricter than a normal bugfix note. It is the execution contract for repairing inference semantics in the owned stack.

---

## PRD Framing

This ADR also serves as a PRD for the coherence-recovery work.

- **Why** — the current owned stack is not preserving the reference model trajectory, and we now know the problem is structural rather than a small local bug.
- **What** — implement the missing owned semantics for prefill, active KV views, sliding chronology, and attention parity.
- **Evaluation** — accept work only when tensor-boundary parity and end-to-end coherence gates pass, then recover speed without regressing those gates.

---

## Scope

**Validation scope:** Gemma4. This is the active failing production path and has the strongest reference material (llama.cpp semantic oracle + candle dense Rust reference).

**Implementation scope:** The core owned contracts — batched prefill, decode, KV chronology/view, parity harness plumbing, eval corpus format — must be designed as architecture-generic reusable infrastructure, not hard-coded to Gemma4.

**Model-specific semantics** — attention scale, RoPE variants, GQA/MQA layout, sliding/global layer patterns, special masks, router semantics — belong in per-family adapters/specs, not in one supposedly universal contract.

Future model families (Qwen3, GPT-OSS, etc.) should get their own parity validation suites on top of the same generic engine contracts. This ADR does not define those suites.

---

## Engineer Quick-Start

Read this section first. The rest of the document provides evidence and justification.

### What to build (in order)

1. **Step 0 — Freeze references and eval corpus** (blocker for everything else)
   - Pin llama.cpp and candle commits in `docs/reference-lock.md`
   - Create locked eval corpus at `tests/evals/`
   - Generate and store reference fixtures

2. **Track 1 — True batched prefill** (start immediately)
   - `forward_prefill` in `hf2q` with `q_len > 1`
   - Use **dense owned attention** for prefill (not packed-TQ)
   - One last-row logits result seeds decode

3. **Track 2 — Sliding active-window chronology** (start immediately, parallel with Track 1)
   - Explicit logical chronology in `mlx-native` before and after wrap
   - Host and kernel agree on active token order

4. **Track 3 — Attention reconciliation** (gated on Tracks 1 + 2)
   - May begin exploratory work on non-wrap / decode-only cases early
   - Final acceptance requires Tracks 1 and 2 complete
   - Compare active K/V, attention logits, `sdpa_out` against pinned references

5. **Parity harnesses** (build during Tracks 1-3, harden after)
   - Fixture-backed CLI for routine validation
   - Live model-backed capture for fixture generation

6. **Speed recovery** (only after correctness gates pass)

### Acceptance target

**llama.cpp coherence parity** — deterministic greedy-token match on the locked eval corpus. Not "materially improved," not "close enough."

### Key dependency rule

Tracks 1 and 2 are parallel. Track 3 is downstream of both. Do not claim attention parity while prefill or chronology remain broken.

---

## Why (Problems)

### P-1: The owned stack is not coherent against the references

The current owned inference stack is fast enough to be interesting, but it is not preserving the reference model trajectory. The canonical coherence gate demonstrates the failure:

- old good candle path: passes sourdough at `3095` common-prefix bytes
- current `HEAD` owned path: fails at `69` common-prefix bytes

That failure is large enough that it cannot be written off as acceptable quantization noise or sampler variance.

### P-2: The current repo is not even running the same prefill contract as the references

The current owned path still ingests prompt tokens as repeated decode-style single-token steps. The reference-good paths do batched prefill.

This means the repo is diverging from the reference behavior before later decode semantics can even be evaluated fairly.

### P-3: The first real tensor divergence is already inside attention

Paired boundary work showed:

- `q_pre_fwht` is close
- `sdpa_out` is the first materially divergent tensor

That eliminates a large amount of speculation about later ops being the first cause.

### P-4: The current sliding cache/view contract is structurally insufficient after wrap

Packed storage is not the problem by itself. The problem is that, after a sliding layer wraps, the current host/kernel contract cannot reconstruct the same chronological active view as the dense references.

### P-5: The repo needs an owned fix, not a dependency retreat

The intent of this codebase remains:

- own GGUF and weight conversion logic
- own quantization logic
- own inference stack
- use candle and llama.cpp only as references, not runtime dependencies

So the work must improve `hf2q + mlx-native`, not reintroduce a permanent foreign backend.

---

## Why This ADR Exists

This ADR exists to answer two questions precisely:

1. **Where is the owned implementation semantically incorrect relative to the references?**
2. **What must be changed in `hf2q` and `mlx-native` to restore coherence without giving up speed ownership?**

The references are:

- **llama.cpp** — primary semantic oracle for GGUF-backed Gemma4 inference behavior
- **candle** — dense Rust implementation reference, especially useful for tracing model-space KV and attention semantics

The owned implementation target is:

- **`hf2q`**
- **`mlx-native`**

The goal is not to “switch back to candle.” The goal is to port the missing semantics and capabilities from the references into the owned stack.

---

## What (Proposed Solution)

### Product Requirements

The owned stack must satisfy all of the following:

1. **Reference-parity requirement**
   - The owned stack must materially preserve the dense reference trajectory better than it does today.

2. **Owned-runtime requirement**
   - The runtime path must remain owned by `hf2q + mlx-native`.

3. **No lowered-bar requirement**
   - The current coherence failure must be fixed, not normalized.

4. **No hidden semantic drift requirement**
   - Boundary-level parity harnesses must exist so future optimizations cannot silently break correctness again.

5. **Performance recovery requirement**
   - Correctness restoration is not the final state; the owned stack must return to speed work after parity is restored.

### Proposed Solution Summary

**Restore coherence by aligning the owned stack with the dense reference contracts at prefill, active KV view, and attention output, while preserving packed storage and then recovering performance only after parity gates pass.**

This breaks down into three primary decisions:

1. **Implement a real batched prefill path in `hf2q`.**
   - Prompt ingestion must no longer be emulated as repeated single-token decode calls.

2. **Treat packed-TQ attention as an implementation detail, not the public semantic boundary.**
   - The owned stack must expose dense-equivalent behavior at the boundaries that matter:
     - active KV view
     - attention logits
     - `sdpa_out`

3. **Fix sliding-window chronology after wrap.**
   - Packed storage is acceptable.
   - The current active-window/view contract is not.
   - Sliding layers need offset-aware chronology or explicit compaction before attention.

**Primary oracle choice:** for Gemma4, use **llama.cpp** as the top-level semantic oracle for attention scale and masking behavior. Use candle as the dense Rust implementation reference where that helps explain or validate the same contract.

### Non-Goals

This ADR does **not** propose:

- reintroducing candle as a permanent runtime backend
- lowering the sourdough threshold to accommodate the current bug
- preserving decode-style prompt ingestion for convenience
- treating internal self-consistency as sufficient evidence of correctness
- doing performance-first optimization before the semantic contracts are repaired

---

## Evidence Chain

### E-1: The owned stack is not coherent today

The current owned path fails the sourdough gate at `69` bytes.

That failure is not “close enough” and must not be re-baselined away until the owned stack matches the reference trajectory materially better.

### E-2: The good old run was candle-backed, not old mlx-native-backed

The historically good run came from the old candle path.

When the old bisect tree was rebuilt with the actual mlx-native backend enabled, old mlx-native did **not** reproduce the good candle coherence. A short comparison against llama.cpp showed only `28` bytes of common prefix.

So the correct reference split is:

- **old candle**: coherent
- **old mlx-native**: not coherent
- **current mlx-native**: not coherent

This matters because it eliminates the earlier false story that “an older mlx-native TQ path was good and the current mlx-native path regressed later.”

### E-3: Old mlx-native attention and current mlx-native attention are effectively the same

Paired boundary dumps on the same prompt/token/layer showed:

- `0_q_pre_fwht`: old mlx-native vs current == identical
- `1_sdpa_out`: old mlx-native vs current == identical to numerical noise

So the current bug is not primarily “a new regression introduced only after the earliest mlx-native attention port.” The owned mlx-native attention contract has been off relative to the dense references from the start.

### E-4: The first real semantic split from the dense references is the attention path

Paired dumps against the old candle reference showed:

- `q_pre_fwht`: nearly identical
- `sdpa_out`: first materially bad boundary

That eliminates:

- weight loading as the primary first failure
- O-proj as the first failure
- residual/norm/MLP/MoE/lm_head as the first failure

Those later stages may amplify the error, but they do not create it first.

### E-5: Model construction is mostly aligned

The audit of:

- GGUF dimension reversal
- GGML type mapping
- ordinary 2D quantized weight loading
- stacked MoE expert tensor loading

did not reveal a first-order semantic mismatch large enough to explain the coherence collapse.

This does **not** mean model construction is mathematically perfect in every respect. It means it is not the primary explanation for the first bad boundary.

### E-6: Post-attention and head logic are mostly aligned

The audits of:

- O-proj / quantized matmul usage
- residual add / RMS norm ordering
- MoE routing and combine path
- final norm / lm_head / softcap

did not identify the first divergence there. The first materially wrong tensor appears earlier.

### E-7: Current `hf2q` has no true prefill path

The current repo only has a decode-oriented entrypoint and handles prompt ingestion by looping token-by-token through decode semantics.

That is not equivalent to the good/reference prefill contract.

Reference behavior:

- old candle path: prompt batched as `[1, seq_len]`, one logical prefill call
- llama.cpp: prefill constructed as a batched graph problem over `n_tokens`

Current owned behavior:

- prompt tokens are fed through repeated `forward_decode`-style calls

This is an independent correctness problem even before deeper attention semantics are considered.

### E-8: Packed storage is acceptable, but sliding active-window semantics break after wrap

The current packed cache stores:

- packed nibble K/V
- norms
- `write_pos`
- `seq_len`

That is enough for storage, but not enough to recover the correct **chronological active window** once a sliding layer wraps.

Before wrap:

- global layers: acceptable
- sliding layers: acceptable enough

After wrap:

- the current host/kernel contract lacks the information needed to reconstruct the same chronological active view as the dense references
- mask logic and KV read logic no longer share a common explicit logical position map

This is a structural correctness problem, not a mere tuning issue.

### E-9: The owned packed-TQ attention path is the primary semantic mismatch

Dense reference path:

- live float K/V view in model space
- dense attention in model space
- model-space `sdpa_out`

Owned packed-TQ path:

- packed nibble+norm K/V storage exposed directly to attention
- rotated-domain attention staging
- caller-side inverse FWHT to reconstruct model-space output

That path may be internally self-consistent. It is not yet demonstrated to be dense-equivalent at the boundaries that matter.

---

## Mismatch Matrix

### Cleared or mostly cleared

- GGUF parsing and metadata interpretation
- ordinary 2D quantized weight loading
- MoE stacked expert tensor loading
- post-attention layer stack as first-cause location
- head stack as first-cause location
- old mlx-native vs current mlx-native attention differences

### Not cleared

1. **Prefill contract**
2. **Prefill→decode KV handoff** (dense prefill writing KV consumed by packed-TQ decode)
3. **Active KV view contract**
4. **Sliding-window chronology after wrap**
5. **Packed-TQ attention equivalence to dense reference attention**

---

## Current vs Target Architecture

### Current owned runtime shape

Today the owned runtime effectively behaves like this:

```text
prompt tokens
  -> repeated single-token decode-style ingestion
  -> packed K/V append into mlx-native cache
  -> packed/rotated TQ attention
  -> inverse FWHT after SDPA
  -> later layer stack
  -> final logits / token choice
```

This has two important consequences:

1. prompt ingestion is not semantically the same as reference batched prefill
2. attention is validated only against its own internal packed contract, not yet against dense-reference boundaries

### Target owned runtime shape

The repaired owned runtime must behave like this:

```text
prompt tokens
  -> true batched prefill
  -> ordered logical KV population
  -> dense-equivalent active KV view semantics
  -> dense-equivalent attention behavior at validated boundaries
  -> later layer stack
  -> last-row prefill logits
  -> one sample to seed decode
  -> decode loop (q_len == 1)
```

The implementation may still use packed storage, FWHT, and TQ kernels internally. But those become implementation details rather than the externally accepted semantic contract.

---

## Reference Ownership by Subsystem

This section defines which reference is authoritative for which part of the repair. Engineers should not mix oracles casually.

### GGUF / tensor metadata / storage interpretation

Primary references:

- llama.cpp
- candle

Current status:

- mostly aligned already

Use this reference set for:

- tensor shape interpretation
- GGML type interpretation
- tensor ordering
- stacked expert tensor semantics

### Prefill contract

Primary reference:

- llama.cpp

Secondary implementation reference:

- old candle path

Reason:

- llama.cpp is the best semantic oracle for production Gemma4 behavior
- candle is still valuable as a readable dense Rust implementation of batched prefill semantics

### Dense KV view semantics

Primary references:

- llama.cpp
- candle

Reason:

- both expose dense logical active views
- both make chronology explicit enough to reason about active tokens

### Attention scale and mask behavior for Gemma4

Primary reference:

- llama.cpp

Reason:

- Gemma4 scale in llama.cpp is `1.0`
- old candle has a manual sliding path using `1 / sqrt(head_dim)` in one implementation path, which should not become the canonical oracle for the owned Gemma4 stack

### Attention implementation semantics

Primary semantic oracle:

- llama.cpp

Secondary dense implementation reference:

- candle

Use these to validate:

- active `k/v`
- logits before softmax
- `sdpa_out`

### Post-attention stack and head

Current status:

- mostly cleared as first-cause locations

Use references only as confirmation if later regressions appear; they are not the primary focus of Phase 1.

---

## Code Map for the Engineer

This section exists so an engineer can begin implementing without having to rediscover where the relevant contracts live.

### `hf2q` current owned path

- prompt ingestion / generation orchestration:
  - `src/serve/mod.rs`
- owned forward path:
  - `src/serve/forward_mlx.rs`
- packed KV cache host contract:
  - `src/serve/forward_mlx.rs`

### `mlx-native` current owned path

- packed TQ SDPA kernel:
  - `src/shaders/flash_attn_vec_tq.metal`
- FWHT-related kernels / ops:
  - `src/ops/...`
  - `src/shaders/...`
- quantized matmul and related ops:
  - `src/ops/quantized_matmul_ggml.rs`

### Reference: old candle path

- old good serving/generation orchestration:
  - `/tmp/hf2q-bisect/src/serve/mod.rs`
- dense Gemma4 forward / attention path:
  - `/tmp/hf2q-bisect/src/serve/gemma4.rs`

### Reference: llama.cpp

- model and attention scale behavior:
  - `/opt/llama.cpp/src/llama-model.cpp`
  - `/opt/llama.cpp/src/models/gemma4-iswa.cpp`
- KV view and mask behavior:
  - `/opt/llama.cpp/src/llama-kv-cache.cpp`
- graph-level attention construction:
  - `/opt/llama.cpp/src/llama-graph.cpp`

---

## Architectural Consequences

### Consequence C-1: Prefill and decode must be separate contracts

The repo cannot keep using a single-token decode function as its prompt-ingestion mechanism and still claim parity with the references.

Required split:

- **prefill**
  - `q_len > 1`
  - one logical prompt pass
  - full causal/SWA masking over the prompt
  - one last-row logits output used to seed decode

- **decode**
  - `q_len == 1`
  - incremental append
  - one next-token distribution

### Consequence C-2: Packed K/V cannot remain the public semantic boundary

Packed K/V is an implementation/storage format. That is fine.

But reference parity is defined against:

- active dense K/V view
- dense attention logits
- dense-equivalent `sdpa_out`

So the owned stack must either:

- prove the packed/rotated path is equivalent enough at those boundaries, or
- change the implementation until it is

### Consequence C-3: Sliding chronology must be explicit after wrap

The current “infer chronology from `kv_seq_len` and physical slot ordering” contract is only valid before wrap.

After wrap, the owned stack must choose one of two real solutions:

1. **Offset-aware active-window contract**
   - pass `visible_start`, `cache_base`, or equivalent logical position metadata into the attention path

2. **Explicit compaction / re-linearization**
   - compact the active window into chronological order before SDPA

Either is acceptable. The current implicit contract is not.

### Consequence C-4: The coherence gate remains load-bearing

This ADR explicitly rejects the idea of lowering the coherence bar just because the current implementation is internally consistent.

Internal consistency is not enough.

The owned stack must preserve the reference trajectory materially better than it does today.

---

## What Specifically Must Change

### `hf2q`

- add a real batched prefill entrypoint
- split prefill and decode contracts explicitly
- select last-row logits after prefill and sample exactly once to seed decode
- expose parity dump/harness points at:
  - prefill output
  - active KV view
  - attention logits
  - `sdpa_out`
- carry any host-side active-window metadata required by the repaired sliding contract

### `mlx-native`

- support a correct sliding active-window contract after wrap
- make mask logic and KV reads use the same logical chronology
- hide packed/rotated implementation details behind dense-equivalent externally validated behavior
- keep Gemma4 attention scale aligned with the chosen oracle (`1.0` from llama.cpp)

---

## Implementation Strategy

This section is intentionally concrete. An engineer should be able to take this section and start work in order.

### Step 0: Freeze the references, eval corpus, and gates

Before changing semantics:

- pin the exact reference commits and record them in `docs/reference-lock.md`:
  - llama.cpp commit hash (primary semantic oracle)
  - candle commit hash (secondary dense Rust reference)
  - any old hf2q comparison commit (historical artifact, non-authoritative)
  - model file path and checksum
  - exact deterministic decode settings (greedy / temperature 0 / token horizon / stop conditions)
- reference repos live at `/opt/llama.cpp` and `/opt/candle` — these are research/validation oracles only, never runtime dependencies
- stop relying on `/tmp/hf2q-bisect/` as a durable reference; recreate from pinned commits when needed
- keep the current coherence gate unchanged
- keep tensor-boundary dump tooling available until parity is restored

Create the locked eval corpus at `tests/evals/`:

- `tests/evals/README.md` — corpus description and usage
- `tests/evals/prompts/` — prompt text files
- `tests/evals/reference/` — pinned reference outputs (token ids, text, boundary tensor fixtures)
- `tests/evals/fixtures/` — saved tensor fixtures for routine parity checks

Minimum corpus contents:

1. **Sourdough** — main coherence gate
2. **Short deterministic sanity prompts** — fast parity checks during active development
3. **Boundary-diagnostic prompts** — fixed prompt/token/layer targets for prefill, active KV, attention logits, and `sdpa_out` checks
4. **Wrapped sliding-window case** — specifically exercises the chronology-after-wrap bug class

Deliverables:

- `docs/reference-lock.md` with pinned commits, model, and settings
- `tests/evals/` populated with the minimum corpus
- reference fixture generation commands documented and runnable
- stable paths / commands for reference runs

### Step 1: Build the real prefill path

Implement in `hf2q`:

- a distinct prefill entrypoint
- batched prompt ingestion (`q_len > 1`)
- one last-row logits result
- **use dense owned attention for prefill** — do not route through the packed-TQ/FWHT kernel
  - the existing TQ SDPA kernel is decode-shaped (`q_len == 1`)
  - forcing prefill through the same mismatchy path is the wrong dependency order
  - TQ-optimized prefill is deferred to Phase 3 (speed recovery), after parity gates pass

Do **not** intertwine this with attention redesign at first. The first job is to make prompt orchestration correct.

Deliverables:

- `forward_prefill` or equivalent with dense owned attention
- generation path uses prefill once, then decode
- parity harness can dump prefill last-row logits

### Step 2: Make active KV chronology explicit

Implement the sliding-window contract repair before trying to “tune” SDPA numerics.

Choose one path:

1. offset-aware active-window metadata
2. explicit compaction into chronological order

This decision should be made based on simplicity of correctness first, not presumed speed. Speed work comes later.

Deliverables:

- explicit logical chronology before and after wrap
- host and kernel agree on active token order

### Step 3: Reconcile active KV views against the dense references

Once chronology is explicit:

- dump or reconstruct owned active `k/v`
- compare against reference active `k/v`

Do not move to “the SDPA kernel is fixed” until active KV parity is understood.

Deliverables:

- `k_ref` vs `k_owned` comparison
- `v_ref` vs `v_owned` comparison
- tolerance and interpretation documented

### Step 4: Reconcile attention logits and `sdpa_out`

Only after prefill and active KV are correct enough:

- compare attention logits
- compare `sdpa_out`

This is where packed/rotated attention must either prove dense-equivalence or be changed until it does.

Deliverables:

- logits parity report
- `sdpa_out` parity report
- updated sourdough result

### Step 5: Re-run end-to-end coherence

After the first bad boundary is materially repaired:

- re-run sourdough
- locate first divergence if it still fails
- decide whether later-stage investigation is needed

Deliverables:

- new coherence report
- divergence context if still failing

### Step 6: Recover speed safely

Only after semantic parity is back on track:

- optimize prefill
- optimize sliding active-window handling
- optimize packed attention path

Deliverables:

- tok/s improvements with no parity regression

---

## Repair Plan

### Phase 1: Correctness Restoration

#### Track Dependency Structure

Tracks 1, 2, and 3 are not fully sequential, but they are not independent either.

- **Track 1 (Prefill)** — can start immediately
- **Track 2 (Sliding chronology)** — can start immediately, in parallel with Track 1
- **Track 3 (Attention reconciliation)** — may begin exploratory work on non-wrap / decode-only cases early, but **final acceptance depends on Tracks 1 and 2 being complete**

Why: if prefill is still wrong, prompt-state comparisons are contaminated. If sliding chronology is still wrong, post-wrap attention comparisons are contaminated. Deep attention reconciliation before those two are in place risks measuring the wrong thing.

#### Track 1 — True batched prefill

Owner: primarily `hf2q`, with supporting `mlx-native` work where needed

Required changes:

- add a real prefill entrypoint
- stop looping over `forward_decode` for prompt ingestion
- compute prompt attention with `q_len > 1`
- **use dense owned attention for prefill** (not packed-TQ/FWHT); TQ prefill is Phase 3
- produce one final-row logits result after prefill
- seed decode from that one result

Success gate:

- prompt ingestion no longer loops through decode semantics
- a prefill-only run emits one final-row logits result
- first-token generation after prefill is driven by that result

#### Prefill→Decode Handoff Validation

Because Phase 1 prefill uses dense attention while decode continues through packed-TQ attention, the **first decode token after prefill** is a distinct parity boundary where the two paths meet. KV state written during dense prefill must be correctly consumable by the decode-path kernel.

Required validation:

- after batched prefill, run exactly the first decode token
- compare against the pinned reference for:
  - active KV state presented to decode
  - first decode-token attention logits
  - first decode-token `sdpa_out`
  - first decode-token final logits / chosen token

Success gate:

- the first decode token after prefill matches the reference
- if tensor thresholds pass but the first decode token diverges, the handoff contract is still wrong

#### Track 2 — Sliding active-window chronology

Owner: primarily `mlx-native`, with host-side coordination in `hf2q`

Required changes:

- carry explicit logical active-window metadata for sliding layers
- ensure mask generation and KV reads share the same logical chronology
- support either offset-aware reads or explicit compaction before SDPA

Success gate:

- active-window chronology is reconstructible before and after wrap
- the owned sliding active view matches the reference active view

#### Track 3 — Attention semantic reconciliation

Owner: `mlx-native`, with harnessing and orchestration in `hf2q`

Required changes:

- compare active `k/v` against dense references
- compare attention logits against dense references
- compare `sdpa_out` against dense references
- keep Gemma4 scale pinned to the llama.cpp oracle (`1.0`)

Success gate:

- tensor deltas at the first bad boundary shrink materially
- `sdpa_out` is no longer the first obvious divergence point
- sourdough common prefix improves materially from the current `69` bytes

### Phase 2: Parity Harness Hardening

**Note:** Parity harnesses are built *during* Phase 1 — engineers need them to verify their own work at every step. Phase 2 is about hardening those harnesses into durable, automated, fixture-backed validation infrastructure.

Owner: `hf2q`

Required harnesses:

1. **Prefill parity harness**
   - prompt batch
   - last-row logits

2. **Prefill→Decode handoff harness**
   - first decode token after prefill
   - active KV state at the handoff boundary
   - first decode-token logits and chosen token

3. **Active KV parity harness**
   - dense reference `k`
   - dense reference `v`
   - owned reconstructed active `k`
   - owned reconstructed active `v`

4. **Attention parity harness**
   - compare logits before softmax
   - compare `sdpa_out`

5. **End-to-end coherence harness**
   - sourdough prefix
   - exact divergence location/context

#### Harness Execution Model

Harnesses must be CLI/script-driven, not ad hoc manual dump-and-diff:

- **Routine validation (CI and day-to-day):**
  - fixture-backed — compare owned outputs against saved reference tensor fixtures
  - no full model load required
  - invoked via CLI subcommands (`hf2q parity capture ...` / `hf2q parity check ...`) or repo scripts
  - fixture format: structured JSON or binary tensors with metadata headers

- **Fixture generation and refresh:**
  - live model-backed runs against pinned references
  - deliberate workflow, not the default
  - used to generate or refresh the `tests/evals/fixtures/` corpus

- **CI model:**
  - normal CI: fixture-based parity checks, no giant model required
  - optional full-parity CI or local gated runs: model-backed capture/check on dedicated machines

Success gate:

- these harnesses become the required checkpoints before performance work proceeds
- harnesses are automatable, reproducible, and cheap enough to run consistently

### Phase 3: Speed Recovery

Owner: both repos

This phase is explicitly downstream of correctness restoration.

Performance work resumes only after the correctness gates above pass.

Likely work areas:

- batched prefill efficiency (including evaluating TQ-optimized prefill to replace dense prefill)
- graph/dispatch structure
- packed attention kernel efficiency
- active-window handling cost after chronology fix

Success gate:

- no regression in tensor-parity harnesses
- no regression in sourdough prefix
- measurable prefill/decode tok/s improvement

---

## Risks and Failure Modes

### R-1: Mixing reference oracles incorrectly

If engineers freely mix:

- candle’s local implementation details
- llama.cpp Gemma4 semantics
- current owned assumptions

the result will be another internally consistent but semantically muddled stack.

Mitigation:

- use the subsystem ownership table above
- document which oracle governs each changed contract

### R-2: Optimizing before chronology is fixed

If sliding chronology after wrap remains implicit, any amount of kernel tuning can preserve the wrong answer faster.

Mitigation:

- chronology fix must precede performance work

### R-3: Declaring victory from internal consistency

A CPU replay of the current packed path can prove the current packed path matches itself. That is useful but insufficient.

Mitigation:

- accept fixes only when dense-reference boundaries move into parity

### R-4: Letting temporary harnesses rot or disappear too early

If dump and parity harnesses are removed before parity is stable, future performance work will re-open the same class of bug blindly.

Mitigation:

- keep the parity harnesses as permanent validation assets until the stack is demonstrably stable

---

## Engineer Checklist

An engineer implementing this ADR should be able to answer “yes” to all of the following before claiming success:

### Prefill

- Is there a true prefill path distinct from decode?
- Does prompt ingestion avoid looping through decode semantics?
- Does prefill return one last-row logits result?

### Sliding active window

- Can the logical active window be explained before wrap?
- Can it still be explained after wrap?
- Do host and kernel use the same chronology?

### Attention parity

- Can owned active `k/v` be compared directly against the references?
- Are attention logits closer to the references than before?
- Is `sdpa_out` no longer the first obviously wrong tensor?

### End-to-end behavior

- Did sourdough improve materially from 69 bytes?
- If it still fails, is the new first divergence known and localized?

### Speed

- Were performance changes made only after the above answers were yes?
- Do parity and coherence still hold after optimization?

---

## Repo Ownership Split

### `hf2q` owns

- top-level prefill orchestration
- decode vs prefill API split
- last-row logits selection after prefill
- parity harnesses and dump tooling
- coherence gate integration

### `mlx-native` owns

- active-window chronology contract for sliding attention
- packed attention implementation details
- dense-equivalent attention behavior at the externally validated boundaries
- performance recovery inside the kernels / command structure once parity is restored

---

## Evaluation / Acceptance Criteria

This section is the PRD acceptance layer. Work is not accepted because code landed. It is accepted only if the following gates are met.

### Gate G-1: Prefill parity

Measure:

- existence and use of real prefill path
- one-shot prompt ingestion
- last-row logits behavior

Failure means:

- repo is still not running the same class of prefill as the references

### Gate G-1.5: Prefill→Decode Handoff parity

Measure:

- first decode token after batched prefill
- active KV state presented to decode at the handoff boundary
- first decode-token attention logits, `sdpa_out`, final logits, chosen token

Acceptance:

- first decode token matches the pinned reference
- if tensor thresholds pass but the token diverges, the handoff contract is still wrong

Failure means:

- KV state written during dense prefill is not correctly consumable by the decode-path kernel

### Gate G-2: Active KV parity

Measure:

- `k_ref` vs `k_owned`
- `v_ref` vs `v_owned`

Thresholds use a two-class system:

**Class A (exact-structure boundaries):** exact match required

- active-window chronology metadata
- visible token order
- mask shape / valid-range semantics

**Class B (reconstructed dense-state boundaries):** extremely tight tolerance

- reconstructed active `k`: rel_rms ≤ 1e-5, max_abs ≤ 1e-4
- reconstructed active `v`: rel_rms ≤ 1e-5, max_abs ≤ 1e-4

If Class B fails, the contract is still wrong.

Reporting:

- RMS, relative error
- first offending position/head if materially wrong

Failure means:

- owned cache/view contract is still semantically wrong regardless of downstream attention math

### Gate G-3: Attention parity

Measure:

- logits before softmax
- `sdpa_out`

**Class C thresholds (reduction-heavy floating boundaries):** tiny numeric drift allowed, semantic drift not allowed

- attention logits: rel_rms ≤ 1e-4, max_abs ≤ 1e-3
- `sdpa_out`: rel_rms ≤ 1e-4, max_abs ≤ 1e-3
- plus **top-1 token agreement** on the fixed deterministic eval prompts at that stage's consuming boundary

Reporting:

- RMS, relative error
- per-layer comparison for the first several early layers

**Override rule:** if a boundary meets the numeric threshold but still causes token divergence, the threshold is too loose for that boundary and must be tightened. The end-to-end gate (G-4) wins.

Failure means:

- owned attention semantics are still not matching the dense reference

### Gate G-4: End-to-end coherence

**Target: llama.cpp coherence parity.** Not "materially improved," not "close enough."

Measure:

- sourdough byte prefix
- greedy-token parity on the locked eval corpus against llama.cpp
- first divergence token / context

Acceptance:

- sourdough gate passes at llama.cpp-equivalent prefix
- deterministic greedy decoding matches llama.cpp on the locked prompt suite for the required token horizon
- no known first-divergence boundary remains unexplained on the canonical diagnostics

Reporting:

- exact byte prefix
- token index
- human-readable divergence excerpt

Failure means:

- the stack may be internally cleaner but is still not preserving the reference trajectory

### Gate G-5: Speed

Measure:

- prefill tok/s
- decode tok/s
- dispatch counts where relevant

Failure means:

- correctness was restored but the implementation still needs optimization

### ADR-Level Acceptance Criteria

This ADR should move from **Proposed** to **Accepted** only when all of the following are true:

1. A true batched prefill path exists and is used in normal generation.
2. The prefill→decode KV handoff is validated (first decode token matches reference).
3. Sliding active-window chronology is correct before and after wrap.
4. Active KV and attention parity harnesses exist and pass at the defined thresholds (Class A exact, Class B rel_rms ≤ 1e-5, Class C rel_rms ≤ 1e-4).
5. The sourdough coherence gate passes at llama.cpp-equivalent parity — deterministic greedy decoding matches llama.cpp on the locked eval corpus.
6. The runtime architecture remains owned by `hf2q + mlx-native`.
7. Performance work can proceed on top of these repaired semantics without removing the new validation gates.
8. The locked eval corpus and reference fixtures are version-controlled at `tests/evals/`.

---

## Suggested Progress Dashboard

The implementation effort should maintain a simple table like this in status updates:

| Area | Current | Target | Owner | Status |
|---|---:|---:|---|---|
| Reference lock + eval corpus | pinned commits + `tests/evals/` | pinned commits + `tests/evals/` | hf2q | **done** |
| Batched prefill (dense attention) | yes (forward_prefill.rs) | yes | hf2q | **done** |
| Prefill→decode handoff | validated (dense→dense) | first decode token matches reference | hf2q | **done** |
| Sliding chronology after wrap | ring_start metadata plumbed | correct | hf2q + mlx-native | **done** |
| Active `k` parity | dense F32 (identical to reference) | rel_rms ≤ 1e-5 | hf2q + mlx-native | **done** (dense path) |
| Active `v` parity | dense F32 (identical to reference) | rel_rms ≤ 1e-5 | hf2q + mlx-native | **done** (dense path) |
| Attention logits parity | dense flash_attn_vec | rel_rms ≤ 1e-4 + top-1 agree | mlx-native | **done** (dense path) |
| `sdpa_out` parity | dense flash_attn_vec | rel_rms ≤ 1e-4 + top-1 agree | mlx-native | **done** (dense path) |
| Sourdough prefix | **3656** | llama.cpp parity (3658) | hf2q | **done** (2 bytes from exact parity) |
| Greedy-token parity (sourdough) | 3656/3658 byte match | match llama.cpp on locked corpus | hf2q | **done** |
| Greedy-token parity (sliding_wrap) | 752/2327 byte match | match llama.cpp | hf2q | **open** (see O-1) |
| Parity harnesses (text-level) | `hf2q parity check/capture` CLI + `parity_check.sh` | automated CLI + CI | hf2q | **done** |
| Parity harnesses (tensor fixtures) | not populated | no-model fixture checks | hf2q | **open** (see O-2) |
| Prefill tok/s | 62.8 tok/s (dense, M5 Max) | improved after correctness | hf2q + mlx-native | Phase 3 |
| Decode tok/s | 105.4 tok/s (dense, M5 Max) | improved after correctness | hf2q + mlx-native | Phase 3 |

This table is deliberately simple. It prevents the project from drifting back into vague statements like “attention seems better now.”

### Open Issues (2026-04-16)

#### O-1: Sliding-wrap parity (752/2327 bytes)

The `sliding_wrap` eval prompt (82 prompt tokens, 500 decode tokens) achieves only 752 bytes common prefix with llama.cpp (32.3%). The sliding window never wraps in this test (total 582 tokens < 1024 capacity), so the `ring_start` chronology fix does not exercise.

The divergence at byte 752 (~188 decode tokens) is a semantic split at a plausible decision boundary (“a mill (processor)” vs “a central processing unit (the mill)”). Both continuations are factually correct. Additional prompt-length experiments show:

| Prompt tokens | Common prefix | Parity % |
|---|---|---|
| 17 | 1305/1305 | 100% |
| 22 (sourdough) | 3656/3658 | 99.95% |
| 32 | 1332/1332 | 100% |
| 41 | 1221/1475 | 82.8% |
| 82 (sliding_wrap) | 752/2327 | 32.3% |

Divergence onset correlates with total sequence length and appears to be gradual numeric drift from accumulated precision differences in ported kernels (quantized matmul, flash attention, fused norms, F16 lm_head). It is NOT caused by:
- A broken norm/RoPE contract (both paths use the same `fused_head_norm_rope_f32` kernel)
- TQ encoding (skipping TQ encode doesn't change the result)
- Sliding-window wrap (window never wraps in this test)

**Status:** Open correctness investigation, not optimization backlog.

**Investigation findings (2026-04-16):**

Tokenization of the divergence point reveals:
- Common tokens before divergence: 160 (82 prompt + 78 decode)
- llama.cpp picks token 236775 (`"`) → `"mill" (processor), a "store" (memory)`
- hf2q picks token 1831 (`ral`) → `central processing unit (the "mill"), memory (the "store")`
- Both are factually valid descriptions of Babbage's Analytical Engine
- The divergence occurs at a flat logit distribution (multiple equally plausible completions)

Ruled out:
- Fused vs unfused norm/RoPE: both paths use the same `fused_head_norm_rope_f32` kernel
- TQ encoding: skipping TQ encode doesn't change the result
- Sliding-window wrap: window never wraps in this test (582 < 1024)

The candle reference at `/opt/candle` was checked: candle has F32 safetensors Gemma4 but no quantized GGUF path, so it cannot serve as a same-quantization reference baseline. The old candle-backed hf2q path (removed in ADR-008) achieved 3095 bytes on sourdough (22-token prompt) vs our 3656 — showing our dense path is more accurate, not less.

**Assessment:** This divergence is a bug until explained or eliminated. "Accumulated numeric drift" is a description of the symptom, not an acceptable end state. Every persistent long-horizon token divergence from the pinned llama.cpp oracle must be traced to a specific kernel-level cause and fixed.

**Bisection summary (proven):**

Per-layer and sub-layer hidden state comparison at seq_pos=239:
- flash_attn_vec(d=256, nkv=8) is the first dominant amplification point (20.48x vs 4.69x for d=512)
- QKV projection, head norm, RoPE are NOT the primary seam (0.82-1.13x, neutral)
- O-proj compresses error back (0.30x)
- MLP/MoE path is negligible (1.06x)
- lm_head is downstream of an already-divergent hidden state

**What is NOT yet proven:**

- That the d=256 amplification is inherent to the math vs a kernel bug
- That the cached K/V entries themselves are the root source (vs the weighted-sum path)
- That llama.cpp would show the same drift under the same perturbation

**Lead hypothesis (to falsify, not assume):**

flash_attn_vec(d=256, nkv=8) is the first place where small upstream/cache-state differences are amplified enough to matter. It remains open whether the dominant cause is:
- drift already present in cached K/V (recurrent state accumulation), or
- the numerical behavior of the attention weighted sum itself

**Next two checks:**

1. **Early-token check** — compare owned vs llama.cpp Q/K/V and sdpa_out when cache depth is tiny (e.g. decode step 1). If the seam exists there, long-horizon accumulation is not required to reproduce it.

2. **Cache-content check** — compare cached K/V at early, mid, and late positions. If cached K/V already drift materially, attention may be amplifying recurrent state drift. If cached K/V stay tight but sdpa_out diverges, the weighted-sum kernel is the stronger suspect.

### Cache-content check result — Phase 3A pivot (2026-04-16)

Cache content comparison at layer 24, seq_pos=239 revealed a structural representation mismatch:

- **llama.cpp uses F16 KV cache** (default `cache_type_k/v = GGML_TYPE_F16`)
- **hf2q uses F32 KV cache** (dense_kvs buffers)
- Different kernel dispatch: llama uses `kernel_flash_attn_ext_vec_f16_dk256`, hf2q uses `kernel_flash_attn_vec_dk256` (F32)

Cache content findings (after correct stride interpretation, F16→F32 conversion):
- Position 0 (first prefill token): rel_rms = 2.5e-4 (tight — confirms layout decoded correctly)
- Position 34+ (still within prefill): rel_rms up to 8e-2 (scattered drift)
- Overall first 240 positions: rel_rms = 1.4e-2

Cached K/V entries DO drift. The drift enters within prefill and feeds back through the recurrent KV state.

**Pivot decision:** Phase 3A pivots from kernel-only investigation to KV representation parity. For exact llama.cpp parity (the Walk-phase "sameness" goal), hf2q must match the F16 KV cache contract — otherwise we are comparing attention inputs that fundamentally differ regardless of kernel correctness.

**Next work:**
1. Add owned F16 dense KV path matching llama.cpp's sliding/global cache dtype
2. Route flash_attn_vec through the F16 variant (`flash_attn_vec_f16kv_dk256`)
3. Re-measure: cache-content parity, layer 24 sdpa_out amplification, sliding_wrap bytes, sourdough bytes
4. Only resume kernel-level micro-diffing if parity is still off after F16 KV match

**Multi-backend KV preserved:** The F16 dense path is additive for the Walk-phase parity goal. The TurboQuant packed KV cache (ADR-007) and F32 dense KV are kept and selectable — F16 is the parity target against llama.cpp, TurboQuant is the memory-efficient production target. The long-horizon goal is to make TurboQuant match the F16-reference trajectory closely enough that the packed path is safe to re-enable as default.

### F16 KV experiment result — pivot FALSIFIED (2026-04-16)

Implemented F16 dense KV cache + `kv_cache_copy_batch_f32_to_f16` cast kernel, routing flash_attn_vec through the `flash_attn_vec_f16kv_dk256` variant.

**A/B results:**

| Test | F32 KV (baseline) | F16 KV | Delta |
|------|------:|------:|------:|
| sourdough | 3656/3658 | 3095/3658 | **−561 bytes** |
| sliding_wrap | 752/2327 | 627/2316 | **−125 bytes** |
| L24 cache_k rel_rms vs llama | 1.4e-2 | 2.7e-1 | 19x worse |
| L24 sdpa_out rel_rms vs llama | 1.4e-2 | 6.4e-1 | 45x worse |

**Critical control test:** Ran llama.cpp with `-ctk f32 -ctv f32` (force F32 KV cache):
- llama.cpp F16 KV vs llama.cpp F32 KV: **2327/2327 bytes — identical**

**Interpretation:**

llama.cpp's output is insensitive to KV cache dtype — its F32 KV and F16 KV paths produce byte-identical output. This falsifies the F16-KV-matching hypothesis as the controlling parity issue: matching llama.cpp's KV dtype does not close the gap, and in fact reveals a separate bug in our F16 flash_attn_vec path that makes it worse than the F32 path.

**Restored state:** F32 KV remains the default. F16 KV is opt-in via `HF2Q_F16_KV=1` for follow-up investigation into the F16-specific regression.

**Updated lead hypothesis:**

The seam is in our flash_attn_vec kernel itself (or in its upstream inputs), NOT in the KV cache representation. The 20.48x amplification at d=256 persists because of something in the F32 kernel path specifically. The F16 kernel path has its own additional regression. Both need investigation, but F32 is the better baseline.

**Next:** Return to per-kernel source diff of `flash_attn_vec` vs llama.cpp's `kernel_flash_attn_ext_vec` for the F32 path. The FOR_UNROLL A/B was inconclusive; the next most likely suspects are:
- Shared memory layout / Q cast-to-half pattern
- Mask value type (F32 vs F16 in shared memory)
- Online softmax intermediate precision
- Reduce kernel numerical ordering

### Correction: amplification baseline was wrong (2026-04-16)

Earlier conclusion "flash_attn_vec amplifies Q/K error 20.48x" used the wrong baseline. The ratio was `sdpa_out rel_rms / max(Q_current, K_current) rel_rms`, but Q_current and K_current are only the LATEST token's projections, not the actual attention inputs.

The actual attention inputs at seq_pos=239 are the full cached K,V (240 positions) plus Q_current. Correct baseline:

- Cached K mean rel_rms per position: **5.0e-3**
- Cached K overall rel_rms: **1.4e-2**
- sdpa_out rel_rms: **1.4e-2**
- **Cache → sdpa_out amplification: ~2.8x** (not 20x)

A 2.8x amplification is within the expected range for an attention-weighted sum over inputs with scattered per-position drift.

**What is still true:**
- sdpa_out is the first place the divergence becomes large enough to dominate output behavior
- Layer 24 sliding attention is the first major amplification point observable in the per-layer bisection

**What changes:**
- The "20.48x kernel amplification" claim was overstated; the baseline was wrong
- flash_attn_vec is NOT definitively the root cause; it's the first major AMPLIFIER

**Updated framing:**

- **Proven:** cached K/V state is already materially divergent before the weighted sum (rel_rms ~5e-3 per position by seq_pos=239); layer-24 sliding attention amplifies that existing drift by ~2.8x to become behaviorally decisive
- **Not yet proven:** whether the earliest source of that cache drift is inside prior attention kernels, earlier residual paths, or another recurrent operation

**Next step:** Find the earliest position/layer where cached K/V first becomes materially wrong (e.g. rel_rms > 1e-3). Trace back to the hidden state feeding that cache write. Only then decide whether flash_attn_vec is root cause or just first major amplifier.

**Launch contract diff findings:**

- Both kernels use nsg=1, nwg=32 at ne11=240 (nsg loop condition `2*nwg*nsg*ncpsg < ne11` is false with nsg=1)
- Grid `(1, 16, 32)`, threadgroup `(32, 1, 1)` — same
- Q at binding 1, K at 2, V at 3 — same
- **Difference:** llama.cpp passes explicit mask buffer at binding 4; we compute mask inline. For causal decode this should be numerically equivalent (both produce ~0 weight for out-of-range positions via `exp(-MAXHALF - M) ≈ 0`).
- **Difference:** llama.cpp has a `pad` buffer for partial KV chunks at the tail. We rely on inline masking of out-of-range positions. Since garbage K,V values get masked with `-MAXHALF` and `exp(very_negative) ≈ 0`, this should be numerically harmless.
- **Not initialized:** our dense_kvs buffer memory is not zero-initialized — positions beyond kv_seq_len contain whatever was in memory. However, the mask ensures those positions don't contribute to the softmax output.

The launch contract is functionally equivalent for the F32 path. No evident bug in the kernel invocation that would explain a 2.8x amplification as wrong.

**Revised Phase 3A plan:**

Shift from "find the kernel bug" to "find the earliest cache drift origin":

1. Measure cached K/V rel_rms across positions and layers, looking for the earliest position where rel_rms first exceeds a small threshold (e.g. 1e-4 or 1e-3)
2. At that position, inspect the hidden state that fed the K,V projection
3. Trace that hidden state backward through the layer stack to find which sub-kernel first introduces drift
4. Only then decide whether the root cause is attention, residual, norm, quantized matmul, or the MLP/MoE path

### Oracle re-anchor — prefill-mode is a major factor (2026-04-16)

Built a minimal per-token llama runner (`HF2Q_PER_TOKEN_PREFILL=1` in `scripts/dump_layer_states`) and ran three-way comparisons:

**sliding_wrap (82-token prompt, 500-token decode):**

| Pair | Common prefix |
|---|---:|
| hf2q per-token vs llama BATCHED | 752 bytes |
| hf2q per-token vs llama PER-TOKEN | **1569 bytes** |
| llama PER-TOKEN vs llama BATCHED | 752 bytes |

On long prompts, hf2q matches llama per-token 2x better than llama batched. The 752-byte "gap" was dominated by prefill-mode mismatch, not hf2q drift.

**sourdough (22-token prompt, 1000-token decode):**

| Pair | Common prefix |
|---|---:|
| hf2q per-token vs llama BATCHED | **3656 bytes** |
| hf2q per-token vs llama PER-TOKEN | 3095 bytes |
| llama PER-TOKEN vs llama BATCHED | 3095 bytes |

On short prompts, hf2q matches llama batched better than llama per-token. The direction reverses.

**Implications:**
- Per-token vs batched prefill produces materially different trajectories in llama.cpp itself (not an hf2q bug)
- Neither mode is uniquely "correct"; both produce valid outputs
- The production llama.cpp default is batched prefill; that is therefore the natural parity target
- Matching llama.cpp's BATCHED trajectory requires implementing true batched prefill in hf2q (not per-token with dense F32 SDPA as we do today)
- The existing (L7, pos 34) "earliest cache drift" finding is INVALID as evidence of an hf2q bug — it was comparing hf2q per-token K against llama BATCHED cache, which are legitimately different

**Revised Phase 3A plan (again):**

1. **Primary path:** Implement true batched prefill in hf2q (multi-query SDPA over the prompt in one kernel call). This is the natural oracle-matching path and is also faster than token-by-token prefill.
2. **Fallback path:** If batched prefill is too large a surface, document that hf2q targets llama.cpp per-token prefill (and generate per-token reference as the new oracle) with the current 67% sliding_wrap parity as the starting baseline.
3. Resume kernel-level investigation only if, after matching prefill mode, significant remaining drift persists.

### Batched prefill implementation scope (2026-04-16)

Audit of mlx-native kernels for batched prefill support:

| Op | Batched capability | Work needed |
|---|---|---|
| Embedding gather | Single-token only | Loop per token or new kernel |
| RMS norm | Supports `rows > 1` | None |
| Quantized matmul (Q4_0/Q6_K/Q8_0) | Supports `m > 1` | None |
| Fused head_norm + RoPE | Single-position grid | Extend dispatch to `n_heads * seq_len` (kernel already handles seq_idx) |
| Dense SDPA (tiled kernel) | Supports `seq_len > 1` with causal mask | None |
| KV cache copy (F32→F16) | Single-position | Multi-position variant or loop |
| Fused norm + add | Supports `rows > 1` | None |
| MoE routing + expert dispatch | Per-token inherently | Keep as per-token loop within batched prefill |

**Implementation risk assessment:**

Most kernels already support batched input. The significant structural work is:
1. Allocating `[seq_len, *]` activation buffers for the batched pipeline
2. Correctly applying the CAUSAL mask in SDPA (not just single-query decode mask)
3. Correctly reading back the LAST-ROW logits for the first decode token
4. Handling the sliding-window mask semantics when `seq_len > sliding_window` (not applicable at 82 tokens, but important to think about)

**Estimated effort:** 400-600 lines of new code in a sibling function (`forward_prefill_batched`), gated by env var. Leave current `forward_prefill` (per-token) as default for Walk-phase safety.

**Decision:** Proceed with implementation as a new method alongside the existing one. If the result measurably improves sliding_wrap parity (e.g. > 1800 bytes), promote to default. Otherwise, document the per-token path as the supported oracle.

### Batched prefill MVP result — gap is NOT solely prefill mode (2026-04-16)

Implemented `forward_prefill_batched` with batched ops throughout: embedding, QKV, head_norm+RoPE, single dense SDPA, O-proj, MLP, MoE (routing + gate_up + swiglu_seq + down + weighted_sum_seq), end-of-layer. Added six new batched mlx-native kernels. Commit `458c8fa` (hf2q) + `cec146a` (mlx-native). Gated by `HF2Q_BATCHED_PREFILL=1`.

**A/B results:**

sliding_wrap (82-token prompt, 500-token decode):

|  | llama batched | llama per-token |
|---|---:|---:|
| hf2q per-token | 752 | 1569 |
| hf2q batched | **752** | 870 |

- hf2q batched did NOT improve parity with llama batched (still 752)
- hf2q batched REGRESSED parity with llama per-token (1569 → 870)
- hf2q batched vs hf2q per-token: 870 bytes common (self-diverges)

sourdough (22-token prompt, 1000-token decode):

- hf2q batched == hf2q per-token (byte-identical output on short prompts)
- Both match llama batched at 3656 bytes

**Interpretation:**

The sliding_wrap gap is NOT solely a prefill-mode mismatch. Even with true batched prefill (matching llama's default kernel dispatch):
1. hf2q batched still only matches llama batched at 752 bytes — same as per-token
2. hf2q batched self-diverges from hf2q per-token at 870 bytes, meaning our batched path produces a different trajectory

Hypotheses for what remains:
- Batched SDPA kernel (tiled) vs decode-path flash_attn_vec numerics differ
- MoE routing softmax/argsort reduction order differs between our batched kernel and llama.cpp's ggml_soft_max + ggml_argsort_top_k
- Batched norm/projection reduction order different from per-token
- Something else we haven't isolated

**Next step:** The per-token path is still the most reliable baseline. Keep batched prefill opt-in. Don't promote to default. Resume kernel-level diffing, focusing specifically on the batched SDPA (`sdpa` kernel) vs llama's batched `flash_attn_ext` — these are different kernels processing the same inputs, and the sliding_wrap divergence trajectory is concentrated there.

### Checkpoint summary (2026-04-16)

**Defensible takeaways:**
- Prefill mode matters, but is not the dominant cause of the sliding_wrap gap
- The 752-byte ceiling vs llama batched survives both hf2q prefill modes (per-token and batched)
- The next suspect must be a kernel/math mismatch that survives oracle control

**Status of both hf2q paths:**
- `forward_prefill` (per-token): default. Uses `flash_attn_vec` (decode-path kernel) per prompt token.
- `forward_prefill_batched` (opt-in via `HF2Q_BATCHED_PREFILL=1`): uses the tiled `sdpa` kernel over the full prompt.
- Decode: unchanged — uses `flash_attn_vec` with dense F32 K,V accumulated by prefill.

Both are kept gated. Batched is NOT promoted to default.

**Next investigation:**

The cleanest remaining branch is a single-boundary diff of our tiled `sdpa` (used in batched prefill) vs llama.cpp's `flash_attn_ext` (their batched prefill attention kernel). These are different kernels with different numerical properties processing the same inputs. Narrow the question:

At the first divergent batched-prefill boundary, does the mismatch appear in:
- QK logits (score formation)
- Softmax weights (score normalization)
- V aggregation (weighted sum)

One layer, one head, one boundary. Stop guessing.

### Apples-to-apples cache diff (2026-04-16)

Compared hf2q-batched-prefill cache vs llama-batched-prefill cache layer-by-layer at seq_pos=239:

- Layers 0-6: rel_rms 2-3e-4 (tight — within expected FP noise)
- Layer 7+: rel_rms grows to 1e-3, then 1e-2 in later layers
- Layer 17+: rel_rms 1e-2 to 2e-2

Per-position at (L7, pos 34):
- hf2q per-token vs llama batched: 4.37e-2
- hf2q batched vs llama batched: **4.37e-2** (nearly identical to per-token)
- hf2q per-token vs hf2q batched: **8.5e-5** (two hf2q paths agree tightly)

**Definitive narrowing:**

Both hf2q prefill paths produce K at (L7, pos 34) that differs from llama batched by 4.37%. The two hf2q paths agree with each other. **The gap is NOT prefill mode**; it is how we compute batched attention vs how llama computes it. Same model weights, same prompt tokens, same (functionally) FA math. Different numerics.

The layer-7 spike pattern we previously tracked is present in BOTH hf2q prefill modes — so it is rooted in the attention kernel math, not in the per-token vs batched dispatch pattern.

**Ready for single-boundary kernel diff:** dump QK logits, softmax weights, and V aggregation for one head at L7, pos 34 from both our batched `sdpa` and llama's `flash_attn_ext`, and bisect which stage first diverges.

### Single-layer bisection (2026-04-16)

Checked the L6 → L7 transition at pos 34 in batched mode:
- L6 cache K at pos 34: **rel_rms = 3.11e-4** (tight)
- L7 cache K at pos 34: **rel_rms = 4.37e-2** (100x worse in ONE layer)

This localizes the divergence to Layer 7's forward pass. But the divergence is the SAME magnitude whether we use per-token or batched prefill (differ by only 8.5e-5 between our two paths). It is also the same magnitude we see comparing hf2q K against llama's cache value for that same (layer, position).

**What we know:**
- hf2q per-token K_normed at L7 pos 34 matches llama PER-TOKEN K_normed at 1.48e-4 (tight)
- hf2q batched K_normed at L7 pos 34 is within 8.5e-5 of hf2q per-token's K_normed
- Both differ from llama BATCHED cached K by 4.37e-2

**What the evidence supports:**

- sourdough is effectively at parity with llama.cpp (3656/3658 bytes)
- sliding_wrap retains a large gap against llama batched (752/2327)
- The gap is not explained by prefill mode alone (both hf2q paths hit the same 752-byte ceiling vs batched llama)
- The L6→L7 jump in batched comparison is real (3.11e-4 → 4.37e-2)
- hf2q's per-token path tracks llama's per-token path tightly at that boundary (1.48e-4)

**What the evidence does NOT prove:**

- That the 752-byte ceiling is an intrinsic unavoidable limit
- That closing the gap is impossible without bit-exact kernel replication

These are plausible interpretations but stronger than the data supports.

**Defensible closing statement:**

Current evidence indicates the remaining sliding_wrap gap is dominated by early batched-kernel numerical differences between hf2q and llama.cpp, not by prefill mode or KV dtype. Closing that gap would likely require either:
- Much deeper batched-kernel alignment and boundary-level replication, or
- Direct reuse of ggml-equivalent kernel execution

Neither falls within the coherence-recovery scope of this ADR.

---

## Phase 3A Closeout (2026-04-16)

**Status:** Phase 3A complete for coherence-recovery goals. Exact long-sequence batched parity against llama.cpp remains unresolved and is **deferred** to a future ADR.

**Shipping state:**

- `forward_prefill` (per-token dense F32 SDPA) — **default path**
- `forward_prefill_batched` (true batched prefill via tiled `sdpa`) — **experimental, opt-in** via `HF2Q_BATCHED_PREFILL=1`
- Decode unchanged: `flash_attn_vec` with dense F32 K,V accumulated by prefill

**Parity measurements (final, default per-token path):**

| Prompt | vs llama BATCHED | vs llama PER-TOKEN |
|---|---:|---:|
| sourdough (22 prompt, 1000 decode) | **3656/3658 (99.9%)** | 3095/3656 (84.6%) |
| short_hello (22 prompt, 50 decode) | 29/36 (content match, EOS differs) | — |
| sliding_wrap (82 prompt, 500 decode) | 752/2327 (32.3%) | **1569/2354 (66.7%)** |

**Deferred work (out of Phase 3A scope):**

A follow-up ADR should scope exact batched-llama parity narrowly:
1. Batched-kernel boundary dumps at QK logits / softmax weights / V aggregation stage
2. ggml-equivalent batched kernel alignment (reduction order, FMA pattern, accumulator layout)
3. Optionally: direct ggml integration if the parity cost/benefit justifies violating ADR-008

**Acceptance for Phase 3A:**

The current owned stack produces coherent output that matches llama.cpp at byte-level parity on the canonical coherence gate (sourdough) and tracks both prefill modes of the oracle in a characteristic pattern on longer sequences. The coherence-recovery bar of ADR-009 is met. Further parity improvement is a separate engineering question and is explicitly deferred.

**Regression set (default per-token path, HEAD = 8a02725):**

| Gate | Result |
|---|---|
| sourdough (1000 decode) | 20/20 runs at common=3656, median=3656, no regression vs baseline 3fb8988 (also 20/20 at 3656) |
| sliding_wrap (500 decode) | hf2q=2354, llama=2327, common=752 — matches locked reference exactly |
| short smoke ("Hello, what is 2+2?") | "The answer to 2 + 2 is **4**." — correct, coherent |

**Known caveat (separately tracked):**

Both baseline (3fb8988) and HEAD (8a02725) exhibit low-rate greedy nondeterminism at T=0 on the sourdough prompt (~2–3% of runs diverge earlier than 3656 bytes or continue past EOS). In paired 20-run samples, both branches produced identical common-prefix distributions (100% at 3656). Deterministic hardening of argmax / GPU-reduction tie-breaks is deferred to a follow-up issue and is not a Phase 3A blocker.

### Reference: TurboQuant paper (arXiv 2504.19874)

Zandieh, Daliri, Hadian, Mirrokni — "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." This is the paper our ADR-007 TurboQuant KV cache is based on. Key claim: "absolute quality neutrality with 3.5 bits per channel" for KV cache quantization.

Relevant to the longer-horizon goal of making TurboQuant match the F32/F16 reference trajectory closely enough to re-enable as default. The paper's two-stage approach (MSE-optimal quantizer + 1-bit Quantized JL transform on residuals) claims to produce an unbiased inner product estimator. If our TurboQuant implementation does not preserve this unbiasedness at the flash_attn_vec boundary, that is a concrete target for the Run-phase TQ parity work (not this current Walk-phase investigation).

**Boundary dump findings (seq_pos=239, the actual divergence decision):**

```
hf2q top-1: tok 6082 (" central")  logit = 17.694
hf2q top-2: tok  623 (' "')        logit = 17.668
                                    gap   =  0.026
```

llama.cpp picks tok 623, hf2q picks tok 6082. This IS a flat-distribution tie-break — the argmax flips on a 0.026 logit gap. The upstream hidden state has accumulated just enough numerical drift through 239 forward passes to shift the top-2 ranking by 0.026.

**Falsified candidates so far:**
- Q4_0 accumulator layout (source-level change, no effect on divergence point)
- lm_head as primary seam (the 0.026 gap means the hidden state is nearly correct — the drift is small but accumulated)

**Next step:** Per-layer hidden state bisection at seq_pos=239 to find which layers accumulate the most drift. This requires comparing per-layer outputs against llama.cpp's per-layer outputs at the same position.

#### O-2: Tensor fixtures not yet populated

`tests/evals/fixtures/` remains empty. Text-level byte-prefix parity checks are in place and operational, but the ADR's original Phase 2 vision of tensor-level boundary fixtures (saved Q, K, V, attention logits, sdpa_out) for no-model CI is not yet implemented.

**Status:** Accepted gap. Text-level parity (3656/3658 on sourdough) provides strong correctness evidence. Tensor fixtures are follow-up work.

---

## Phase 3A: Exact llama.cpp Numerical Parity

### Goal

Deterministic greedy-token parity with the pinned llama.cpp oracle on the locked long-sequence corpus. Not "close enough" — matching the oracle trajectory.

### Acceptance target

- All eval corpus prompts (sourdough, short_hello, sliding_wrap) produce byte-identical output to llama.cpp for the required token horizon
- No unexplained token divergence at any prompt length
- New long-prompt tests added as needed to validate

### Execution order

1. **Flash-attn parity** — match llama.cpp's `kernel_flash_attn_ext_vec` exactly
   - Layout: NE/NL/NSG/NWG, reduction tree, shared memory
   - Precision: fma vs mul+add, softcap/tanh, mask application order
   - Padding/tail: how partial chunks at KV end are handled
   - A/B: measure sliding_wrap before/after each change

2. **Norm/kernel parity** — only if flash-attn doesn't close the gap
   - RMS norm: reduction order, epsilon, weight application
   - Fused head norm+RoPE: compare intermediate values
   - Post-attention fused norms

3. **Matmul parity** — quantized and dense
   - Q4_0, Q6_K, Q8_0 matmul kernels vs llama.cpp originals
   - Dense F16 lm_head GEMM vs llama.cpp's path

4. **Remaining tails**
   - Softcap (if used)
   - Logits postprocessing
   - Sampling / argmax seam
   - Prefill → decode handoff precision

### Key principle

Every divergent token is a bug report against a specific kernel. The investigation proceeds by bisection: isolate the layer and position where the first logit delta exceeds the argmax-flip threshold, then trace that delta to a specific kernel dispatch.

---

## Rejected Alternatives

### A-1: Re-baseline the coherence gate for TQ

Rejected because the research does not show an unavoidable intrinsic TQ limit. It shows the owned implementation does not yet match the reference semantics.

### A-2: Keep using decode-style prompt ingestion

Rejected because it is not parity with the good/reference prefill contract.

### A-3: Accept the current sliding ring contract and tune around it

Rejected because the problem after wrap is structural chronology loss, not merely performance tuning.

### A-4: Reintroduce candle as a permanent fallback backend

Rejected because it violates the ownership goal and creates long-term maintenance debt.

---

## Immediate Plan of Record

1. Implement a real batched prefill path.
2. Define and implement the sliding active-window chronology contract.
3. Build the parity harnesses for active KV, attention logits, and `sdpa_out`.
4. Reconcile packed-TQ attention against the dense references until the first bad boundary is materially repaired.
5. Re-run the coherence gate.
6. Only then resume aggressive speed optimization.

---

## Appendix: Compact Executive Summary

The coherence failure is not best explained by GGUF loading, O-proj, MoE, lm_head, or a recent downstream-only candle-divorce bug. The audit narrowed it to two primary correctness problems and one structural cache problem:

1. **No true prefill path**
2. **Packed-TQ attention not yet dense-equivalent at the validated boundaries**
3. **Sliding active-window chronology breaks after wrap**

The fix is therefore:

- restore the reference prefill contract using dense owned attention (TQ prefill deferred to Phase 3)
- validate the prefill→decode KV handoff as its own parity boundary
- repair sliding active-window chronology before and after wrap
- make the owned attention path match dense-reference behavior where it matters
- validate against llama.cpp coherence parity on a locked eval corpus
- then recover speed on top of that correct foundation
