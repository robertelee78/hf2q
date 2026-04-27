# ADR-015: mlx-native — single-CB decode forward pass for qwen35

- **Status:** Proposed
- **Date:** 2026-04-26
- **Authors:** Robert E. Lee + Claude Code
- **Successor of:** ADR-012 §Optimize / Task #15 (closed at 0.94× of llama.cpp; cited the need for a "new mlx-native perf ADR")
- **Sibling of:** ADR-013 (qwen35 inference — owns the qwen35 forward path being rewritten)

## Context

ADR-012 closed the qwen35moe **conversion** scope at 0.94× of llama.cpp on
`dwq46` `n=256` decode (109.1 vs 116.0 t/s, M5 Max, same-day median per
`project_end_gate_reality_check`). Per-CB GPU time matches llama.cpp; the
gap is **CPU-side encoding/scheduling overhead**.

### The structural diagnosis

| Path | CBs / decode token | CPU encode overhead at ~5µs/CB |
|---|---:|---:|
| **llama.cpp** (`ggml-metal-context.m:458`) | 1–2 | ~5–10 µs |
| **hf2q Gemma** (`forward_mlx.rs:1309` — "one GraphSession per layer (30/pass)") | ~30 | ~150 µs |
| **hf2q qwen35** (95 `device.command_encoder()` call-sites in `inference/models/qwen35/`) | ~102 (measured `fa3f9d6`) | ~510 µs |

llama.cpp's own measurement: *"tests on M1 Pro and M2 Ultra using LLaMA
models, show that optimal values for n_cb are 1 or 2"* (`ggml-metal-context.m:458`).
Their `dispatch_apply` parallel-encoding mechanism (`:550`) targets
**prefill**, where `n_cb` may be larger; for decode, `n_cb=1` is fastest.

### Why prior CB-fusion attempts moved the needle by noise

`fa3f9d6` (small-scope CB fusion, FullAttn `sdpa+ops6-7` → 1 CB) recovered
~40 µs/token at +0.5% within noise — exactly the slope predicted by ~5 µs/CB:
the experiment removed 10 CBs / token. That data point **confirms the
theory** at one decimal place; it just couldn't statistically distinguish
+0.5% from zero at n=256 with 3 cold runs. Removing all ~100 CBs (going
to 1–2) projects to ~500 µs / token saved ≈ the entire 6% deficit.

### Why dispatch_apply alone isn't the lever for decode

`dispatch_apply` parallelizes encoding **across CPU threads**; CB count
**stays the same**. To recover 0.5 ms / token of encode overhead the CB
count itself has to drop by ~50× — not the encode wall-clock per CB.
The lever for decode is **single-CB forward pass**, not parallel encoding.

(Prefill is a separate axis — see Non-Goals.)

## Decision

**D1.** Migrate `qwen35` decode forward to a **single `GraphSession`** (1 CB
per forward pass), mirroring llama.cpp's `encode_async`
(`ggml-metal-context.m:676–722`) — one logical command buffer encoded with
explicit `encoder.memory_barrier()` between dependent op stages.

**D2.** Keep the existing per-layer-encoder path as the legacy
implementation behind `HF2Q_LEGACY_PER_LAYER_CB=1`. After a 7-day soak +
sourdough-pass on the new path, delete the legacy path entirely (per
`feedback_no_broken_windows`).

**D3.** Bit-exact parity gate vs the current `forward_gpu_greedy` is
mandatory (within 1e-5 max-abs logit error on the 16-prompt smoke set).
No fallback path is shipped — if parity fails, the lever is wrong, fix
it; don't gate the primary path off (per
`feedback_never_ship_fallback_without_rootcause`).

**D4.** Exit criteria: `dwq46` `n=256` decode median **≥ 1.00× of llama.cpp**
same-day median, 3 cold runs each, M5 Max, cold SoC per
`feedback_perf_gate_thermal_methodology`.

## Phasing

| Phase | Deliverable | Definition of done |
|---|---|---|
| **P1 — Audit** | Map every cross-CB synchronization point in `forward_gpu` for the qwen35 path. List of `(commit/commit_and_wait → next encoder)` transitions and the buffer dependencies that justify each one. Output: a markdown table in this ADR. | Every CB boundary in the live qwen35 path is annotated with the data dependency it protects, OR documented as legacy / removable. |
| **P2 — Empty-CB cost calibration** | Stand-alone bench: encode N (1, 10, 100, 1000) empty CBs in a tight loop on M5 Max; time wall-clock; compute µs/CB. Confirms or refutes the ~5 µs/CB working assumption. | Number is published in this ADR; if µs/CB ≪ 5, the upper bound on this ADR's win shrinks proportionally and we re-scope before P3. |
| **P3 — Implementation** | New `forward_gpu_single_cb` in `src/inference/models/qwen35/forward_gpu.rs` — one `GraphSession`, all dispatches encoded, explicit `memory_barrier()` for cross-stage dependencies, single `commit()` at end. | Builds in release. Compiles without `unsafe` beyond what `mlx-native` already requires. |
| **P4 — Parity gate** | 16-prompt smoke (`scripts/smoke-dwq46.sh` or equivalent) on the new path produces logits within 1e-5 max-abs of legacy path. | Logged in commit message. |
| **P5 — Bench gate** | 3 cold runs of `hf2q --benchmark` on `dwq46` at `n=256` (cold SoC). Same-day llama-bench `-p 0 -n 256 -r 3` on the same model. Ratio computed. | Ratio ≥ 1.00× recorded in ADR + commit message. |
| **P6 — Default cut-over** | `HF2Q_LEGACY_PER_LAYER_CB=1` is the only way back to legacy. Default = single-CB. | Smoke + bench gate green on default-default settings. |
| **P7 — Soak + delete legacy** | 7-day window, no regressions. Then delete `forward_gpu_greedy` (legacy) entirely. | Diff-stat ≥ 1000 LOC removed. |

## Open questions / risks

| ID | Risk / open question | Mitigation |
|---|---|---|
| **R1** | Barrier correctness: single-CB requires every cross-stage dependency expressed as `enc.memory_barrier()`. A missed barrier = silent corruption. | P4 bit-exact parity gate is mandatory. Spot-check with `mlx_native::ops::memory_barrier` debug logging during P3. |
| **R2** | Encoder lifetime + buffer pool: a long-lived encoder may hold buffer refs across the full forward pass, blocking pool reuse and inflating peak working set. | P3 measures pool stats; if `pool.alloc_count` > pre-ADR + 10%, size the pool up or refactor to release. |
| **R3** | `MLX_PROFILE_CB` per-label timing relies on CB granularity. With 1 CB / token, per-stage attribution is gone. | Switch to per-dispatch labels (`commit_labeled` becomes `dispatch_labeled` — already partially supported). Document the change in the P3 commit. |
| **R4** | The 6% gap may not be 100% encode overhead. If P2 shows µs/CB << 5, single-CB wins shrink. | P2 is the cost calibration. If the upper bound drops below ~3% headroom, re-scope to a different lever (Option B below) before P3. |
| **R5** | MoE expert dispatch is the largest contributor to dispatch count (256 experts × N selected per layer). Single-CB doesn't compress dispatch count, only CB count. If dispatch overhead dominates encode cost, single-CB still wins (one `commit` not 100), but the CPU encode cost itself doesn't shrink. | P2 also measures per-dispatch encode cost (encode N noop dispatches into 1 CB). If per-dispatch dominates, the lever is fewer dispatches, not fewer CBs — branch to a separate ADR (e.g., persistent-MoE-router or shader-side expert loop). |
| **R6** | Decode-only pivot: prefill is unchanged. Long-prefill parity inversion (`project_long_prefill_parity_inverts`) remains open. | Out of scope for ADR-015 — covered by Non-Goal 1. |

## Non-goals

1. **Prefill parity** — `pp≥1024` parity inversion at `-ub 512` matched batching is a separate axis, owned by a future ADR. ADR-015 measures decode only.
2. **Speculative / MTP decode** — ADR-013 P14 territory.
3. **Sampler upgrades beyond greedy** — ADR-008 placement decision stands; sampling lives in hf2q `sampler_pure.rs`.
4. **Apply to non-qwen35 paths** — Gemma's 30-CB / pass is a separate refactor. If P5 succeeds and the architecture pattern proves sound, a successor ADR can port Gemma.

## Alternatives considered

- **Option A: dispatch_apply parallel encoding (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:550`).** *Rejected for decode* — llama.cpp's own data shows `n_cb=1` is optimal for decode (`:458`). dispatch_apply is a prefill mechanism. Would still help our prefill — but per Non-Goal 1, that's a separate ADR. Not committed in this iteration.
- **Option B: cross-token speculative pipelining (draft + verify decode).** *Deferred* — substantial program-structure change. Becomes attractive only if Option A (single-CB) hits the wall before 1.00×. Land in a successor ADR if needed.
- **Option C: source-level kernel fusion (combine adjacent compute kernels into one shader).** *Falsified 9× in the M5 Max static-evidence kernel-hypothesis register* — most recent: `swiglu` Q4_0 `mv_id` (mlx-native@`4efeec0`, hf2q@`502364d`, −1.5%). The Metal compiler hoists what llama.cpp hand-tunes (`project_metal_compiler_auto_optimizes_static_levers`). Not a productive lever on M5 Max for current workloads.
- **Option D: keep status quo at 0.94×.** *Rejected.* Per `feedback_shippability_standing_directive` and `feedback_no_shortcuts`, "as fast as our peers" is the standing exit bar. 0.94× is not shippable.

## References

- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:448-614` — main encode loop; `:550` `dispatch_apply`; `:458` "optimal n_cb is 1 or 2"
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:670-722` — `encode_async` block, the per-CB encoding worker
- `/opt/hf2q/docs/ADR-012-qwen35moe-conversion.md:215-256` — diagnosis + closure
- `/opt/hf2q/src/inference/models/qwen35/forward_gpu.rs` — current per-layer-encoder qwen35 forward (target of P3 rewrite)
- `/opt/hf2q/src/serve/forward_mlx.rs:1309` — Gemma's "one GraphSession per layer" baseline (30 CBs / pass)
- mlx-native: `mlx_native::GraphSession`, `mlx_native::CommandEncoder::memory_barrier()`
- Memory pins: `feedback_perf_gate_thermal_methodology`, `feedback_shippability_standing_directive`, `feedback_never_ship_fallback_without_rootcause`, `feedback_no_broken_windows`, `project_metal_compiler_auto_optimizes_static_levers`, `project_end_gate_reality_check`

## Changelog

- **2026-04-26 — Proposed.** Diagnosis pivot from ADR-012 §Optimize: gap is CB-count, not CB-encode-time. Single-CB forward pass selected over `dispatch_apply` based on llama.cpp's own n_cb data.
