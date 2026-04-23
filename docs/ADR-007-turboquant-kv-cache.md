# ADR-007: TurboQuant KV Cache Compression for 262K Context

**Status:** Partially Implemented — End-to-End Correctness UNVERIFIED; dormant on default path since 2026-04-16 (ADR-009 Track 3 fallback). Phase 0.4 C-0 divergence audit **COMPLETED 2026-04-21**. C-0b localization **COMPLETED 2026-04-21** — verdict E1-partial. C-1 kernel replay **ATTEMPTED 2026-04-22 → VERIFICATION_BLOCKED** (2 harness defects). C-1-unlock **COMPLETED 2026-04-22** (dual mode) — **single-step clear, multi-step open**: with barriers + 23-row capture fixed, A dropped 1.2445→5.1e-5 (24,000×), two independent harness implementations (Claude + Codex) produced byte-identical output on all 4 variations. BUT the in-harness CPU oracle uses the same `nibble_dequantize` as the kernel, so a spec-level dequant bug would be invisible; cumulative drift, ring-wrap, and nonzero `ring_start` also untested. **Multi-step audit is the new gating step** (pos 50/500/1050 ring-wrap replay + independent-floor oracle from pre-quant K/V).
**Date:** 2026-04-14 (original); revised 2026-04-21 (honest current-state rewrite); 2026-04-21 (C-0 audit completion + Codex-reviewed revision); 2026-04-21 (C-0b localization + Codex-reviewed narrowing); 2026-04-22 (C-1 VERIFICATION_BLOCKED + 2 harness defects identified); 2026-04-22 (C-1-unlock dual-mode single-step clear + scope caveats)
**Decision Makers:** Robert, Claude
**Related ADRs:** ADR-006 (mlx-native GPU backend — KV cache path lives here), ADR-005 (inference server — speed gates), ADR-008 (candle-divorce port — introduced ring-chronology regression), ADR-009 (Track 3 dense-SDPA "safe fallback" — the stub this ADR's mantra forbids)
**Reference:** Zandieh, Daliri, Hadian, Mirrokni — "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (arXiv:2504.19874, April 2025)

---

## Current Status (2026-04-22 late — TQ correctness-and-speed is the destination; mixed-precision was a detour)

**Terminal goal (re-affirmed by user 2026-04-22 late during autonomous /loop iter 2):** a fully CORRECTLY working TurboQuant implementation, fully coherent (sourdough byte-exact) AND fast (matches or beats the llama.cpp F16 baseline, and unlocks 262K context). Mixed-precision (TQ sliding + dense global) is NOT the destination — it was a fallback hypothesis explored in C-4 T3 that is now deprioritized. The path forward is: root-cause the TQ sliding-path port-drift bug that regresses sourdough from 3656 bytes (main, TQ gated off) to 69 bytes (TQ sliding-path active); land the fix; then restore TQ for all layers and verify end-to-end sourdough + throughput.

**One sentence (current investigation state):** The ADR has walked through C-0 → C-0b → C-1 → C-1-unlock → C-2 → C-3 (representation-floor round-trip clean at analytic Lloyd-Max 0.097) → C-4 T1 (uniform higher bit-widths 4/5/6-bit measured; 6-bit at 0.103 worst-case SDPA sits in the 0.10-0.15 policy-sensitive dead zone) → C-4 T3 (mixed-precision edit architecturally correct at `src/serve/forward_mlx.rs:1479` but branch archived unmerged because re-enabling TQ sliding decode still reproduces the 69-byte port-drift regression); the ACTIVE next step is /loop iter 3's multi-step decode-path audit — at pre-wrap positions {seq_pos=22, 32, 64} and ring-wrap positions {seq_pos=1024, 1025, 1030}, capture identical Q + dequantized K/V for sliding layer 0, run both `flash_attn_vec_tq` and dense `flash_attn_vec` on the exact same inputs, elementwise diff localizes the TQ port-drift bug to (a) kernel math, (b) ring chronology, or (c) dispatch/buffer. Cherry-pick `e9fd6fc`'s `compute_tq_ring_start_after_write` helper first for the ring-wrap legs.

**What C-3's "representation floor is physics" verdict meant and didn't mean:** it meant the TQ spec's CPU round-trip is at the analytic Lloyd-Max floor — the codebook and FWHT encode/decode are internally consistent at the per-vector level. It did NOT mean the TQ kernel + surrounding pipeline is bug-free. C-4 T3 exposed a separate regression in the sliding-layer TQ path (the "69-byte gate fail" from brain memory `project_tq_coherence_investigation`) that the CPU round-trip cannot see because it tests the spec, not the GPU path.

### What actually exists in code

| Component | Path | Status | Notes |
|---|---|---|---|
| `hadamard_quantize_kv_fast_d256/d512` shader | `/opt/mlx-native/src/shaders/hadamard_quantize_kv_fast.metal` | Built, replay-test max_err < 0.001 | SIMD-shuffle FWHT, 1 simdgroup/head, zero threadgroup barriers |
| `dispatch_hadamard_quantize_kv` (single-token) | `/opt/mlx-native/src/ops/hadamard_quantize_kv.rs` | Built, called every decode step AND every non-batched prefill token | Always fires when `!HF2Q_SKIP_TQ_ENCODE`, even when dense path will read |
| `dispatch_hadamard_quantize_kv_seq` (seq wrapper) | `/opt/mlx-native/src/ops/hadamard_quantize_kv.rs` (commit `a28783e`) | Built on 2026-04-21. Not yet wired into main; needed for batched prefill population | Iterates cleared single-token kernel with buffer byte offsets — no shader changes |
| `flash_attn_vec_tq`, `flash_attn_vec_tq_v2` | `/opt/mlx-native/src/shaders/flash_attn_vec_tq*.metal` | Built, cleared by replay test | Reads packed nibbles + per-position norm + pre-rotated centroid table |
| TQ SDPA decode branch | `forward_mlx.rs:1258–1300` approx (post-revert line numbers vary) | **UNREACHABLE on default path** | Gated by `use_dense_sdpa = self.dense_kvs.is_some()`; `dense_kvs` is always populated |
| Dense SDPA decode branch (Track 3 fallback) | `forward_mlx.rs:1264–1410` approx | **Active on default path** | Reads dense F32 KV the prefill stored in `self.dense_kvs` |
| Ring-chronology fix (Codex) | `forward_mlx.rs` + `forward_prefill{_batched}.rs` — branch `cfa/cfa-20260421-111303-tq-revival/codex`, commits `e9fd6fc` + `415c9d6` | Preserved on branch; **reverted from main** 2026-04-21 (`e0f33c1`) because sourdough-byte-exact still fails at 69 bytes with the fix | Real bug: `kv_write_pos` was the pre-write slot but decode read it as `ring_start` for a full ring. Fix: `(write_pos + 1) % capacity` when the ring is full |

### What does NOT exist

1. **TQ as the active decode read path.** Removed on merge + reverted same session (see "Capitulation" below).
2. **Byte-identical sourdough passing with TQ-only.** Fails at 69 bytes with Codex's fix applied; passes at 3656 bytes only when the dense fallback is active.
3. **Any measurement at long context (pp > 8K) with TQ as read path.** ADR-007 Phase 2 gates (F-1 262K OOM, F-4 needle-in-haystack, P-2 decode speed at 262K) were never exercised because decode never read from TQ.
4. **Realized memory savings.** Both `self.kv_caches[].k_packed` (TQ, ~750 MiB at pp65k) AND `self.dense_kvs[].k` (F32, ~6 GiB at pp65k) are allocated on the default path. We hold 6.75 GiB of KV — **worse** than dense-only.
5. **Phase 0 validation evidence.** The Phase 0.1/0.2/0.3 gates (Hadamard-vs-random MSE, gather throughput, Hadamard overhead) do not have written measurement reports in `docs/` or `/tmp/`. Phase 0 appears to have been skipped.

### The capitulation (2026-04-16 ADR-009 Track 3)

Commit history `3472689..HEAD` shows: TQ was integrated end-to-end, sourdough failed at 69 bytes, the response was to introduce `dense_kvs` as a "Track 3 safe fallback" rather than root-cause the regression. This ADR's own mantra section says:

> **No fallback. No stub (todo later) code.**

`use_dense_sdpa = dense_kvs.is_some()` is the literal pattern the mantra forbids. Every decode currently runs the TQ encode (~0.14 ms/token) AND the dense SDPA read — cost paid, benefit unrealized.

### Today's session (2026-04-21) findings

Dual-mode CFA `cfa-20260421-111303-tq-revival`:

- **Claude team:** concluded infeasible based on the kernel test's `nrmse < 0.15` assertion. Verdict: "4-bit cannot byte-match F16". That conclusion was **wrong framing** — the replay test `max_err < 0.001` from 5-day-old memory says the kernel round-trip is much tighter than `nrmse < 0.15` (which is the unit-test safety margin, not the realized precision on natural inputs).
- **Codex team:** found a real off-by-one bug in TQ's `ring_start` computation — decode passed the pre-write slot as "oldest surviving" when after the write that slot holds the newest token. Fix (commit `e9fd6fc`) makes the ring chronology correct. Coherent output restored with TQ-only decode. But **sourdough still fails at 69 bytes** with the fix applied.
- **Batched-prefill population gap:** `forward_prefill_batched.rs` only populated `dense_kvs_vec` (local) — never the TQ cache. With the dense fallback removed, batched prefill + decode produced garbage because decode read an uninitialized TQ cache. My extension (commit `415c9d6`) adds TQ encode to batched prefill via the new `dispatch_hadamard_quantize_kv_seq` wrapper.
- **Reverted** the merge (`e0f33c1`) because sourdough-byte-exact remains at 69 bytes even with all known fixes. Main is back to the dense-fallback state; branch preserved.

**Resolved (2026-04-21 — later same day, CFA session `cfa-20260421-C0-audit`):** C-0 layer-by-layer paired-dump audit executed. Report: `docs/tq-c0-audit-2026-04-21.md` (initial draft commit `789b667`, Codex-reviewed revision commit `bd8ab27`). Scaffolding: commit `742f892` extends `HF2Q_DUMP_ALL_CACHE` to the Phase 3A attn Q/K/V + sdpa_out dump sites for full-coverage single-run audits.

The binary (a)-bug-or-(b)-floor question did NOT resolve cleanly — it resolved into a narrower trinary:

1. **The divergence is real and beyond representation.** 196 of 210 `sdpa_out` cells on the TQ path violate the kernel's own declared `max_abs_diff < 1.0`; every one of 30 layers has at least one decode position where `sdpa_out` violates `nrmse < 0.15` (per-layer max nrmse range 0.673–1.281, using the kernel test's own L2-reference formula `sqrt(sum_sq_diff / sum_sq_ref)`). Hypothesis (b) "within representation noise" is falsified.

2. **The defect is NOT yet pinned to the SDPA kernel.** At decode position 1 / layer 0 specifically, current-step Q, K, V are byte-identical (SHA-256 equal) between dense and TQ paths and `sdpa_out` still diverges — but decode pos 1 attends over a 22-token prefilled prompt cache, which on the TQ path is TQ-encoded-then-decoded and was NOT dumped in C-0. That reopens all four localization hypotheses:
   (H1) **TQ SDPA kernel math** (`flash_attn_vec_tq` in mlx-native@`a28783e`)
   (H2) **FWHT pipeline** (forward-rotate Q / inverse-rotate sdpa_out around the kernel call)
   (H3) **Prefill encode/cache** (Lloyd-Max 4-bit encode on prompt tokens, `hadamard_quantize_kv` for decode writes, `dispatch_hadamard_quantize_kv_seq` for batched prefill)
   (H4) **Dispatch / buffer binding** (strides, norm binding, centroid-table lookup, ring/mask parameter handoff)

3. **Methodology correction from within the session (important).** The initial draft report (`789b667`) claimed `L=0/P=1 sdpa_out = 0.844` was 844× the kernel's threshold and attributed the defect to the kernel. The dual-mode Codex reviewer (review-only CFA mode) caught three high-severity methodology errors: (i) the diff script's `nrmse` formula differed from the kernel's own test formula `sqrt(sum_sq_diff / sum_sq_ref)`, (ii) the kernel test's actual `max_abs_diff` threshold is `1.0`, not `1e-3`, and (iii) byte-identical *current-step* Q/K/V at pos 1 does NOT prove SDPA inputs are identical because decode pos 1 attends over prefill-cached tokens never dumped. The revision commit `bd8ab27` corrects the math and narrows the verdict.

**Process learning — Codex review is load-bearing for binary-verdict deliverables.** Claude self-certified 9/9 acceptance criteria with three high-sev errors intact. Without the Codex second-opinion step this session would have sent C-1 directly into kernel-shader bisection, skipping prefill-state dumping, and wasted the next session. For any future CFA session producing a binary-verdict audit deliverable, review-only mode with a Codex reviewer should be treated as non-optional, not optional.

**Gating next step (updated 2026-04-22, late):** C-1-unlock **COMPLETED** (first dual-mode CFA session on the TQ problem). Verdict: single-step clear, multi-step open. Full report: `docs/tq-c1-unlock-2026-04-22.md`. Claude and Codex independent harness implementations produced byte-identical Metal dispatches; A dropped 1.2445→5.1e-5 (24,000×) confirming C-1's blocked verdict was the missing barriers + 22-row capture. **New gating step: C-2 multi-step audit** with 3 prioritized caveat-closers: (a) **independent-floor oracle** from pre-quant K/V (disarms "CPU oracle uses same dequant as kernel" caveat which is load-bearing for detecting spec-level dequant bugs), (b) canary symmetry fix (CPU oracle receives the same mutation as GPU buffer), (c) multi-step / ring-wrap / nonzero-`ring_start` variations (where H1 gets actually tested against the cumulative sourdough regression).

### C-0b outcome (2026-04-21, evening — CFA session `cfa-20260421-C0b-localize`)

Paired F32 dumps of end-of-prefill TQ-packed KV cache (codex worktree with new `HF2Q_DUMP_TQ_STATE=1` + `HF2Q_DUMP_LAYERS_LIST=0,5`) vs dense F32 KV cache (main@`a258e92` via existing `HF2Q_DUMP_ALL_CACHE=1`). Sourdough prompt, 22 prefill tokens. Layer 0 sliding (nkv=8, hd=256) + layer 5 global (nkv=2, hd=512). Python dequantizer mirrors `flash_attn_vec_tq.metal` exactly (inline `CODEBOOK_4BIT`; nibble low=even / high=odd; rsqrt(hd) applied once; inverse FWHT AFTER centroid gather).

**Measured result across all 440 cells (2 layers × K+V × heads × 22 positions):**
- Zero cells violate kernel bound `nrmse < 0.15`
- Zero cells violate kernel bound `max_abs_diff < 1.0`
- Worst cell: L0/k/head1/pos20 nrmse=0.1390, max_abs_diff=0.0484
- V/K max_abs_diff asymmetry (~10×) is scale-driven (V norms are 8–16× K norms); NOT a V-path-specific signal

**Verdict: E1-partial.** The non-batched, compact-encoded prefill cache dequantizes correctly on both sliding and global layers. Critically: `HF2Q_BATCHED_PREFILL` is ack-gated behind `HF2Q_UNSAFE_EXPERIMENTS=1` and `sourdough_gate.sh` sets neither, so the 69-byte regression reproduces on the exact non-batched path this session cleared. **Encode on the path that actually runs is correct.** Defect lives in H1 (kernel math), H2 (FWHT pipeline inside decode), or H4 (dispatch / buffer binding / mask_type) on that path.

**What this verdict does NOT claim (per Codex review narrowing):**
- Batched prefill `dispatch_hadamard_quantize_kv_seq` (codex-branch commit `415c9d6`) was NOT dumped. If HF2Q_BATCHED_PREFILL is ever re-enabled, H3 on that path is unknown.
- Dumps were compacted `[nkv, kv_seq_len=22, hd/2]`; kernel reads `[nkv, kv_capacity=1024, hd/2]` with the full stride. Any capacity-stride bug is invisible in this test.
- Meta JSON recorded `mask_type=1`; decode TQ path passes `mask_type=2` for sliding layers. The meta is NOT a verbatim decode-call snapshot. C-1 session 1 must capture this directly.

**Methodology correction (from Codex review).** Worker 3 initially committed `2f935b6` claiming global "E1 — H3 cleared" with a self-certified 9/9 acceptance check. Codex caught three high/med-severity overclaims: (i) nonbatched scope vs "batched path cleared" phrasing, (ii) compact-dump layout vs kernel's capacity-stride access pattern, (iii) self-test gate 0.12 was framed as a "universal mathematical floor" when it's actually a Gaussian N(0,1) round-trip artifact. Queen verified all three against ground truth (dump file literal contents, `forward_prefill.rs:827-873` dump loop, `forward_mlx.rs:1279` mask_type dispatch) and revised as commit `208723b`. This is the second consecutive CFA session where Claude self-certification passed with high-severity scope/framing errors intact; Codex review caught both (C-0 caught nrmse formula + threshold + byte-identity scope; C-0b caught scope + layout + framing). **CFA pattern reinforced: review-only mode with a Codex reviewer is non-optional for binary/trinary-verdict audit deliverables.**

**Artifacts (on main, at commits `2f935b6` + `208723b` + `b43d14c`):**
- `docs/tq-c0b-localize-2026-04-21.md` — full report
- `docs/tq-c0b-localize-2026-04-21-raw.csv` — 440-row diff table
- `docs/tq-c0b-localize-2026-04-21-summary.md` — analyst summary
- `scripts/tq-c0b-dequant.py` — Python dequantizer mirroring the kernel
- Instrumentation `b43d14c` on main: `HF2Q_DUMP_TQ_STATE=1` + `HF2Q_DUMP_LAYERS_LIST` + `dump_u8`/`dump_meta_json` helpers + dump site in non-batched prefill (the batched-path dump site from the codex-branch commit was dropped because batched prefill on main doesn't populate TQ cache — that's gated on codex commit `415c9d6` which is not on main).

### C-1 outcome (2026-04-22, dawn — CFA session `cfa-20260422-C1-kernel-replay`)

Kernel replay test attempted per the C-4 E1 branch. Harness committed at `mlx-native@9a4ca61` as `examples/tq_kernel_replay.rs` — loads L=0/P=1 Q + packed K/V + norms from a manifest, runs 4 variations (A full / B FWHT-disabled / C dense-reencoded / A_canary norms=1e9), computes nrmse + max_abs_diff vs an in-harness CPU reference. Report at commit `6ca03c2` (initial) + `6cf6b85` (Codex-reviewed revision) on main.

**Verdict: VERIFICATION_BLOCKED.** Two independent harness defects prevent attribution:

1. **Capture-point defect (caught by Worker 3 analyst)**: the C-0b dump captured end-of-prefill state (`kv_seq_len=22`, 22 packed rows), but production SDPA at decode step 1 runs with `kv_seq_len=23` — the decode-token K/V is written into packed slot 22 by `hadamard_quantize_kv` (`src/serve/forward_mlx.rs:1226-1243`) BEFORE the SDPA dispatch at `:1464`, and seq_len is incremented at `:1007-1015` before any per-layer work. The replay is missing the 23rd row that the kernel actually attends over at its heaviest weights.
2. **Synchronization defect (caught by Codex review)**: the harness dispatches `FWHT(Q) → flash_attn_vec_tq → iFWHT(sdpa_out)` on Metal's concurrent-dispatch command encoder (`mlx-native/src/encoder.rs:404-453`) with **zero `memory_barrier()` calls**. Production inserts three `barrier_between` calls at this boundary (`src/serve/forward_mlx.rs:1429-1431, 1441-1446, 1477-1480`). Without barriers the concurrent encoder can execute FWHT and kernel dispatches out-of-order or overlapping, making the caller-side FWHT effectively a no-op. This almost certainly explains the A=B bit-identity that Worker 2 initially read as "kernel applies FWHT internally, H2 ruled out" — that reading is withdrawn.

**Partial findings (retained but confounded):**
- A = B = A_canary to 16 decimal places at 22 rows. As noted above, "bit-identical" cannot be asserted (raw output bins weren't persisted, only aggregate JSON). The equality is consistent with the FWHT-dispatch being a no-op due to Defect #2, not with kernel-internal FWHT.
- C ≈ A at 22 rows. Variation C is degenerate in retrospect: dequant-then-requant reproduces the same packed bytes within small norm drift, so C mostly retests A rather than isolating a new locus. A real dense control would run `flash_attn_vec` (dense F32 kernel) on the same 22-row dequantized K/V.
- Canary (A_canary = A) does NOT validate mask semantics as the report initially claimed. The kernel's main loop at `mlx-native/src/shaders/flash_attn_vec_tq.metal:262-266, 299, 358` breaks at `ic >= kv_seq_len`; positions 22..1023 are never dereferenced regardless of mask. The canary confirms loop bounds, not mask correctness. H4 mask-leak is therefore NOT ruled out.
- Worker 4's "C-0 reframing" claim (dense-23 vs TQ-22 apples-to-oranges at C-0's L=0/P=1 cell) was a **new methodology error** that Codex caught. Production always uses post-write `kv_seq_len` before SDPA, so C-0's comparisons were row-count-matched at all positions. C-0's "within-bound at L=0/P=1" finding remains as-stated, and C-0's "all 30 layers violate bound" finding at pos 5+ is unaffected. The reframing was retracted in revision `6cf6b85`.

**Methodology pattern across 3 sessions on the same problem (queen's Phase-3 observation):**

| Session | Errors | Caught by |
|---|---|---|
| C-0 | 3 high-sev (nrmse formula, 1e-3 threshold, byte-identity scope) | Codex |
| C-0b | 3 high-sev (nonbatched scope, compact layout, Gaussian-heuristic framing) | Codex |
| C-1 | 1 first-order state-field (22/23) | in-worker analyst |
| C-1 | 3 higher-order (dispatch barriers, C-0 reframing wrongness, canary misinterpretation) + 3 med-sev overclaims | Codex |

Internal workers are learning to catch state-field mismatches but not dispatch/synchronization correctness or claim-discipline. Protocol refinement: add a dispatch-fidelity verification phase pre-measurement + a claim-discipline phase post-measurement for future TQ sessions.

**Key pattern-level insight carried forward**: bit-identical results across variations that should mathematically differ are a synchronization-noop fingerprint, not a rule-out signal. First hypothesis should be "the variation toggle is a no-op at the dispatch level," not "the toggled operation doesn't matter." Stored as brain memory `992ab3f1`.

**Artifacts (on main at `6ca03c2` + `6cf6b85`; on mlx-native at `9a4ca61`):**
- `docs/tq-c1-kernel-replay-2026-04-22.md` — full report (revised to acknowledge both defects + retract the C-0 reframing)
- `/tmp/cfa-20260422-C1-kernel-replay/` — manifest, inputs, 4 metrics JSONs (retained for C-1-unlock re-use)
- `mlx-native/examples/tq_kernel_replay.rs` — harness scaffolding (needs barrier fix + Variation C replacement)

---

## Engineering Mantra (load-bearing — read before every session)

Source: `~/Documents/mantra.txt` (Robert, undated). Quoted verbatim.

> **DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.**

**Operational reading for this ADR:**

- **Chesterton's fence** — the F16 KV cache and 8192 global cap exist for good reason (memory pressure). This ADR replaces them with a quantized cache that has been validated by information-theoretic bounds AND empirical benchmarks (LongBench, needle-in-haystack). We are not removing the F16 path on faith.
- **Measure 3x, cut once** — Phase 0 validates correctness (logit match) before Phase 1 touches the SDPA kernel. Phase 2 benchmarks bandwidth savings before Phase 3 lifts the context cap. No step proceeds without measurement.
- **No fallback** — the F16 KV cache path is deleted once TurboQuant is validated. No feature flag, no dual-path. Clean stack.
- **Never make assumptions** — the research swarm (5 agents, 2026-04-14) produced concrete FLOP counts, bandwidth estimates, and kernel architecture analysis. Every claim in this ADR has a derivation. Where a claim depends on an untested hypothesis (e.g., gather throughput in SDPA), it is flagged as Phase 0 measurement work.

**2026-04-21 addendum — we have shipped a stub.** The ADR-009 Track 3 `dense_kvs` fallback is the "no fallback, no stub" violation the mantra forbids. Phase 0 was skipped (no measurement reports exist for 0.1/0.2/0.3). The 69-byte sourdough regression that drove the fallback was never root-caused; the response was to gate TQ decode off on the default path. The debt is: (1) execute Phase 0 with written evidence, (2) run a layer-by-layer divergence audit between the dense F32 path and the TQ path on identical inputs to localize the 69-byte regression to a specific op or confirm it as representation noise, (3) only then choose between "fix and remove dense_kvs" vs "ship TQ as opt-in with a TQ-specific quality gate".

---

## Problem Statement

Gemma-4-27B's `max_position_embeddings` is 262,144 tokens. The current inference engine caps global-layer KV cache at 8,192 positions (grep: `let max_global_kv = 8192` in `forward_mlx.rs`) because F16 storage at full context is prohibitive:

| Context length | F16 KV cache (global layers only) | Total F16 KV cache | Feasible on 192 GB M5 Max? |
|---|---|---|---|
| 8,192 (current cap) | 160 MB | 360 MB | Yes |
| 32,768 | 640 MB | 840 MB | Yes, but 1/6 of bandwidth budget |
| 131,072 | 2,560 MB | 2,760 MB | Tight |
| 262,144 (full) | **5,120 MB** | **5,320 MB** | Yes for memory, **no for bandwidth** |

At 262K context on M5 Max (546 GB/s bandwidth):
- F16 KV cache read per decode token: 5,320 MB → **9.7 ms** just for cache reads
- Current total forward pass: ~10.6 ms/token at 8K context
- At full context, KV cache bandwidth alone exceeds the entire current decode budget

**The 8192 cap is not a temporary convenience — it is a hard wall imposed by F16 storage economics.** Breaking through it requires sub-F16 KV cache representation.

### Why not simpler quantization (INT4 per-channel)?

The research swarm (Agent: quant-comparator) established:
- At 4 bits, INT4 per-channel and TurboQuant are equivalent — both near-lossless
- At 3 bits, TurboQuant beats KIVI (INT-based) by ~1 LongBench point (49.44 vs 48.50)
- At 2 bits, TurboQuant is decisively better — distribution-free bounds vs INT2's severe degradation
- **TurboQuant's random rotation eliminates outlier channels**, which plague INT-based schemes at low bit-widths

For the 262K goal, we need ≤3.5 bits to keep KV cache under ~1.2 GB. At that bit-width, TurboQuant's quality advantage over INT3/INT4 is measurable and its theoretical guarantees are proven.

---

## Decision

**Implement TurboQuant_mse with Hadamard rotation and mixed-precision channel splitting for KV cache compression, enabling full 262K context on M5 Max.**

Configurable bit-width via `--kv-bits <2|3|4>`, default 3-bit for quality. For aggressive compression (262K context), 2.5-bit effective rate via fixed channel splitting (see design choice #3).

### Key design choices (each justified by research findings):

1. **TurboQuant_mse, not TurboQuant_prod** — the QJL residual correction stage (for unbiased inner products) is unnecessary at ≥2.5 bits. Agent: qjl-necessity established that multiplicative bias β > 0.97 at 2.5 bits, producing negligible softmax distortion (KL ≈ 0.001). The paper's own KV cache experiments used TurboQuant_mse. Skipping QJL means all bits go to MSE precision — simpler and better.

2. **Hadamard rotation, not dense random matrix** — **This is an engineering deviation from the paper.** The paper (Algorithm 1) uses a uniformly random orthogonal matrix Π generated via QR decomposition of a Gaussian matrix. We substitute the deterministic Walsh-Hadamard Transform for O(d log d) performance vs O(d²). Agent: cost-analyst showed dense rotation is acceptable at encode time (~290 μs) but **catastrophic at SDPA decode time** (7.6 ms for Gemma, 72% of budget). Hadamard reduces rotation cost to ~20 μs total. Prior art (QuaRot, QuIP#) validates Hadamard for outlier spreading in LLM quantization. **Phase 0.1 validates this substitution empirically:** if Hadamard MSE exceeds random rotation MSE by >20%, we fall back to randomized block-diagonal rotation (QuIP#-style).

3. **2.5-bit effective via fixed channel splitting** — The paper achieves non-integer bit-widths through outlier channel splitting: a subset of channels is quantized at a higher bit-width than the rest. Specifically, for 2.5-bit effective at head_dim=d: the first d/4 channels use 3-bit quantization (8 Lloyd-Max centroids), the remaining 3d/4 channels use 2-bit (4 centroids). Effective rate: (d/4 × 3 + 3d/4 × 2) / d = 2.5 bits/coordinate. **The split is a compile-time constant, not per-position.** Rationale: after Hadamard rotation, all coordinates are approximately N(0, 1/d) with equal magnitude (confirmed by QuaRot: rotation eliminates outlier channels entirely, producing kurtosis ≈ 3). Per-position outlier detection would add ~167 MB storage at 262K with negligible quality benefit. Phase 0.1 validates this assumption by measuring per-coordinate magnitude variance after rotation.

   At 262K context, 2.5-bit TQ achieves 49.44 vs 50.06 on LongBench (−1.2%) while saving 40% more memory vs 3.5-bit. The paper's needle-in-haystack recall at 2.5-bit is 0.997 (matching F16 baseline).

4. **Pre-rotated centroid tables** — Agent: metal-feasibility showed that naive per-position dequantization during SDPA requires O(d²) rotation per cached token (infeasible at long contexts). Pre-rotating centroid vectors at init time eliminates rotation from the SDPA read path entirely: dequant = table lookup only. Each bit-width has its own centroid table; for 2.5-bit mode, both the 2-bit and 3-bit tables are pre-rotated.

5. **Mixed nibble packing for storage** — For uniform bit-widths (2, 3, or 4-bit), indices are nibble-packed (4-bit aligned) for Metal-friendly access. For 2.5-bit mode, the two channel groups use their respective widths: 3-bit channels packed in nibbles (wasting 1 bit), 2-bit channels packed in nibbles (wasting 2 bits). Effective storage is ~4 bits/coordinate despite 2.5-bit quality. If memory pressure demands true mixed-width packing, Phase 3 introduces it.

   **Revision note:** This is a simplifying assumption for Phase 0-1. If nibble packing at 262K context exceeds memory targets, Phase 2 triggers true mixed-width packing. The acceptance criteria below specify the memory gate.

---

## Architecture

### Current KV Cache Pipeline (F16)

```
Token → QKV projection (F32) → kv_cache_copy_batch_f32_to_f16 → [F16 cache]
                                                                      ↓
                                                          flash_attn_vec reads F16
```

2 dispatches for cache write (K + V). 2 dispatches for SDPA (main + reduce). Total: 4 dispatches per layer.

### TurboQuant KV Cache Pipeline

```
Token → QKV projection (F32) → hadamard_quantize_kv → [packed cache + norms]
                                                              ↓
                                                 flash_attn_vec_tq reads packed
                                                 (centroid gather from pre-rotated tables)
```

2 dispatches for cache write (K + V). 2 dispatches for SDPA (main + reduce). Total: **4 dispatches per layer** (unchanged).

### New Metal Kernels

#### 1. `hadamard_quantize_kv` (replaces `kv_cache_copy_batch_f32_to_f16`)

**Input:** F32 KV vector `[num_kv_heads, head_dim]`  
**Output:** Packed indices `[num_kv_heads, head_dim]` as nibbles + F32 norm scalar per head  
**Operation per head:**
1. Fast Walsh-Hadamard Transform on the head vector (in-place, shared memory)
2. Scale by `1/√d` (we use the normalized convention: H_normalized = H_unnormalized / √d, so H·H = I)
3. Compute and store `‖x_rotated‖₂` as F32
4. Normalize: `x̂ = x_rotated / ‖x_rotated‖₂`
5. For each coordinate: find nearest Lloyd-Max centroid. In 2.5-bit mode: coordinates 0..d/4-1 use the 3-bit codebook (8 centroids), coordinates d/4..d-1 use the 2-bit codebook (4 centroids). In uniform mode (2, 3, or 4-bit): all coordinates use the same codebook.
6. Pack centroid indices into nibble (4-bit) output buffer

**Threadgroup design:** One threadgroup per head. d threads (256 for sliding, 512 for global). Hadamard butterfly uses `log₂(d)` stages with shared memory barriers. Quantize step is embarrassingly parallel across coordinates. In 2.5-bit mode, the first d/4 threads and remaining 3d/4 threads follow uniform code paths (no per-thread branching).

**Dispatch count:** 1 per K, 1 per V = **2 per layer** (same as current).

#### 2. `flash_attn_vec_tq` (modified `flash_attn_vec`)

**Change from current:** Replace F16 cache loads with:
1. Read packed nibble indices for cached position
2. Read F32 norm scalar for that position
3. Gather centroid values from pre-rotated lookup table
4. Use gathered F32 vector in dot product / value accumulation

**Pre-rotated centroid table:** `[2^b, head_dim]` F32 per layer. For b=4 (nibble), d=256: 16 × 256 × 4 = 16 KB per layer. 30 layers total: **480 KB**. Negligible.

**Critical assumption to validate in Phase 0:** Gather-based cache reads (indexed by nibble values) vs sequential F16 reads. Apple Silicon's texture/buffer gather throughput may differ from sequential bandwidth. This is the #1 risk item.

#### 3. Lloyd-Max Codebook (compile-time constant)

Precomputed centroids for the Beta distribution `f_X(x) = Γ(d/2)/(√π·Γ((d-1)/2)) · (1−x²)^((d−3)/2)` at practical dimensions and bit-widths. In high dimensions (d ≥ 128) this converges to `N(0, 1/d)`, so we use a single Gaussian-optimal codebook. **This is an approximation** — the exact Beta distribution depends on d. Phase 0.1 validates that Gaussian-optimal codebooks achieve MSE within 5% of dimension-specific Beta-optimal codebooks at d=128, 256, and 512.

| Bit-width | Centroids | Storage |
|-----------|-----------|---------|
| 2 | 4 scalars | 16 bytes |
| 3 | 8 scalars | 32 bytes |
| 4 | 16 scalars | 64 bytes |

Baked into the binary as `const` arrays. No runtime computation.

### Data Layout

**Packed KV cache buffer per layer per K or V:**
```
[num_kv_heads × capacity × ⌈head_dim/2⌉] u8     // nibble-packed indices
[num_kv_heads × capacity] f32                     // per-position norm scalars
```

**Pre-rotated centroid table per layer:**
```
[2^b × head_dim] f32                              // Hadamard-rotated centroids
```

Precomputed at model load: apply inverse Hadamard to each centroid vector. We use the normalized convention (H·H = I), so H⁻¹ = H — the inverse is the same transform. This is one FWHT per centroid — at most 16 transforms of length 512, trivial. For 2.5-bit mode, both 2-bit (4 centroids) and 3-bit (8 centroids) tables are pre-rotated independently.

### Memory Budget at 262K Context

**Gemma-4-27B, 2.5-bit effective (stored as 4-bit nibble):**

Sliding layers: 25 layers, 8 KV heads, head_dim=256, capacity=1024  
Global layers: 5 layers, 2 KV heads, head_dim=512, capacity=262144

| Component | Calculation | Size |
|-----------|------------|------|
| Sliding packed (25 layers) | 25 × 2(K+V) × 8 heads × 1024 pos × ⌈256/2⌉ u8 | 50.0 MB |
| Sliding norms | 25 × 2 × 8 × 1024 × 4 bytes | 1.6 MB |
| Global packed (5 layers) | 5 × 2 × 2 × 262144 × ⌈512/2⌉ u8 | 1,280.0 MB |
| Global norms | 5 × 2 × 2 × 262144 × 4 bytes | 40.0 MB |
| Centroid tables | 30 × 16 × 512 × 4 (worst case, global head_dim) | 0.94 MB |
| **Total KV** | | **1,372.5 MB** |

vs F16 at 262K: **5,320 MB**. Compression: **3.88×**.

**With true mixed-width packing** (Phase 3 optimization if needed):

For 2.5-bit mode: d/4 channels at 3-bit + 3d/4 channels at 2-bit = 2.5 bits/coord average.  
Packed storage per position: ⌈d × 2.5 / 8⌉ bytes.

| Component | Calculation | Size |
|-----------|------------|------|
| Sliding packed (25 layers) | 25 × 2 × 8 × 1024 × ⌈256×2.5/8⌉ bytes + norms | 33.2 MB |
| Global packed (5 layers) | 5 × 2 × 2 × 262144 × ⌈512×2.5/8⌉ bytes + norms | **860.0 MB** |
| **Total KV** | | **~893 MB** |

Compression: **5.96×**.

### Bandwidth at 262K Context (M5 Max, 546 GB/s)

| Representation | KV read/token | Time @ 546 GB/s | Decode estimate |
|---|---|---|---|
| F16 | 5,320 MB | 9.7 ms | ~60 tok/s |
| TQ nibble (4-bit stored) | 1,373 MB | 2.51 ms | ~105 tok/s |
| TQ 2.5-bit packed | 893 MB | 1.64 ms | ~115 tok/s |

---

## Implementation Plan

### Phase 0 — Validate & Measure (Chesterton's fence + measure 3x) — **STATUS: SKIPPED**

**Goal:** Prove TurboQuant preserves coherence and measure gather throughput on M5 Max.

**As of 2026-04-21, no written measurement report for any Phase 0 subtask exists in `docs/` or `/tmp/`.** The kernels went straight to integration; this ADR's `Chesterton's fence + measure 3x` discipline was not executed. The 69-byte sourdough regression that triggered the Track 3 fallback is the direct consequence.

**0.1 — CPU reference implementation — [ ] NOT DONE**
- Implement Hadamard + Lloyd-Max quantize/dequantize in pure Rust (no GPU)
- Validate: quantize a KV vector, dequantize, measure MSE vs paper's bounds
- Validate: run a full forward pass with CPU-side quantize/dequantize intercepting the KV cache write/read, compare logits to F16 baseline
- **Hadamard vs random rotation comparison:** Quantize 1000 KV vectors using both Hadamard rotation and dense random orthogonal rotation (QR of Gaussian). Compare MSE. **Gate:** Hadamard MSE ≤ 1.2× random rotation MSE at each head_dim (128, 256, 512). If exceeded, fall back to randomized block-diagonal rotation.
- **Gaussian codebook validation:** Compare MSE using Gaussian-optimal codebooks vs dimension-specific Beta-optimal codebooks at d=128, 256, 512. **Gate:** Gaussian codebook MSE ≤ 1.05× Beta-optimal codebook MSE.
- **Fixed channel split validation:** After Hadamard rotation of 1000 KV vectors, measure per-coordinate magnitude variance. Identify top-d/4 highest-magnitude coordinates per vector; measure overlap ratio across vectors. **Gate:** If overlap < 50% (outlier channels are position-dependent), revisit fixed-split assumption. Expected: overlap is low but magnitudes are near-uniform, confirming fixed split is sufficient.
- **Gate:** Top-1 token agreement on 100-token greedy decode at 8K context. If top-1 disagrees on >2 tokens, investigate before proceeding.

**0.2 — Gather throughput microbench — [ ] NOT DONE**
- Write a standalone Metal kernel that reads from a nibble-packed buffer using index-based gather (simulating SDPA cache reads)
- Measure throughput vs sequential F16 reads at capacity=8192 and capacity=262144
- **Gate:** Gather throughput ≥ 50% of sequential F16 throughput. If below 50%, the bandwidth savings from smaller representation are negated by gather penalty — revisit the architecture (consider dequant-to-temp-buffer instead).

**0.3 — Hadamard transform microbench — [ ] NOT DONE**
- Write a standalone Metal kernel for in-place FWHT at d=128, 256, 512
- Measure latency per head and total per-token overhead across all layers
- **Gate:** Total Hadamard overhead ≤ 200 μs/token (< 2% of decode budget).

**0.4 — Layer-by-layer divergence audit — [x] COMPLETED 2026-04-21 (revised verdict in commit `bd8ab27`)**
(Added 2026-04-21.) Paired F32 dumps of Q (pre-FWHT), K, V (pre-TQ-encode), and sdpa_out (post-inverse-FWHT on TQ side, apples-to-apples) at each of 30 layers × 7 decode positions {1, 5, 10, 15, 20, 25, 30} × 2 configurations:
  - config A: dense F32 KV decode (`main@dcfb773`)
  - config B: TQ-packed KV decode (`cfa/cfa-20260421-111303-tq-revival/codex@415c9d6`, Codex ring-chronology fix applied)

Sourdough prompt baseline held during the audit: dense PASS 3656 bytes, TQ FAIL at 69 bytes — matches the regression under investigation. Identical greedy decode on both sides. Same seed. 840 paired F32 tensor comparisons (30 × 7 × 4 ops).

**Outcome (revised):** every one of 30 layers has at least one decode position where `sdpa_out` violates the kernel's own `nrmse < 0.15` (L2-reference formula `sqrt(sum_sq_diff / sum_sq_ref)`), per-layer max nrmse range 0.673–1.281. 196 of 210 `sdpa_out` cells violate `max_abs_diff < 1.0`. The divergence is real and exceeds the kernel's declared tolerance on every layer. What is NOT proven: that the defect lives in the TQ SDPA kernel itself vs FWHT, encode/prefill, or dispatch. That distinction is the C-0b localization experiment.

**Artifacts:** `docs/tq-c0-audit-2026-04-21.md` (report), `docs/tq-c0-audit-2026-04-21-raw.csv` (840-row diff table), `docs/tq-c0-audit-2026-04-21-summary.md` (analyst summary), `docs/tq-c0-audit-2026-04-21-perhead.md` (per-head drilldown at L=0/P=1), `scripts/tq-c0-diff.py` (diff tool). Scaffolding commit `742f892` extends `HF2Q_DUMP_ALL_CACHE` to the Phase 3A attn Q/K/V + sdpa_out dump sites.

**Gate:** [x] Written conclusion published. Gate released. Proceed to C-0b for locus pinning — the binary (a)/(b) question the gate originally anticipated was replaced by a trinary finding, so C-4's branching is now conditional on C-0b's E1/E2/E3 outcome rather than directly on C-0.

### Phase 1 — Metal Kernels — **STATUS: [x] kernels built, [?] correctness gates unverified on Phase-0 CPU reference (since there is no Phase-0 CPU reference), [x] integration done, [ ] integration sourdough gate FAILS**

**Goal:** Replace F16 KV cache with TurboQuant on GPU.

**1.1 — `hadamard_quantize_kv` kernel — [x] BUILT**
- Metal compute kernel: FWHT + normalize + quantize + nibble-pack
- One threadgroup per head (fast variant uses 1 simdgroup/head, zero threadgroup barriers)
- **Correctness gate:** Output indices match CPU reference from Phase 0.1 exactly (bitwise on indices, ε < 1e-6 on norms) — **[?] ASSERTED but not verified against a CPU reference because Phase 0.1 was skipped.** Kernel has an internal replay test (`tests/test_flash_attn_vec_tq.rs`) with `assert!(nrmse < 0.15)` — that is a unit-test safety margin, not evidence of Phase-1 correctness-gate compliance.

**1.2 — Pre-rotated centroid table computation — [x] BUILT**
- At model load, apply inverse Hadamard to each of the 2^b centroid vectors
- Store as `[2^b, head_dim]` F32 Metal buffer per layer
- **Correctness gate:** `H^(-1) · centroid` round-trips to original centroid within ε < 1e-7 — **[?] ASSERTED, Phase-0-less.**

**1.3 — `flash_attn_vec_tq` kernel — [x] BUILT**
- Fork `flash_attn_vec`, replace F16 cache loads with nibble-unpack + centroid gather + dot product
- Keep the existing workgroup partitioning and reduce pattern
- **Correctness gate:** SDPA output matches CPU reference from Phase 0.1 within ε < 1e-4 — **[?] Phase-0-less.** 5-day-old memory claims replay-test max_err < 0.001 on isolated kernel input; that's necessary but not sufficient for the ε < 1e-4 end-to-end gate.
- **Sourdough gate:** Full 100-token greedy decode matches F16 path output through ≥95 tokens — **[ ] FAILS.** With Codex's ring fix applied, decode diverges at byte 69 (~token 15-20). Without the fix, at ~13 tokens.

**1.4 — Integration into `forward_mlx.rs` — [x] CODE INTEGRATED, [ ] GATED OFF, [ ] SOURDOUGH FAILS**
- `MlxKvCache` fields updated: `k_packed: MlxBuffer` (u8 nibble) + `k_norms: MlxBuffer` (F32) — **[x] DONE**
- `dispatch_kv_cache_copy_batch_f32_to_f16` replaced with `hadamard_quantize_kv` dispatch — **[x] DONE for non-batched prefill and decode**
- Batched-prefill TQ cache population — **[ ] NOT ON MAIN.** Was added on branch `cfa/.../codex` via `dispatch_hadamard_quantize_kv_seq` (commit `415c9d6`); reverted with the merge.
- `flash_attn_vec` replaced with `flash_attn_vec_tq` on decode — **[?] CODE PRESENT AT forward_mlx.rs:~1258, BUT UNREACHABLE.** Gated by `use_dense_sdpa = self.dense_kvs.is_some()` which is always `true` because prefill unconditionally populates `self.dense_kvs`. **This is the ADR-009 Track 3 fallback — the mantra-violating stub.**
- Centroid table buffers added to model state — **[x] DONE**
- 2.5-bit mode — **[ ] NOT IMPLEMENTED.** Current code is uniform 4-bit. The 2.5-bit channel split remains a paper design that was never coded.
- **Dispatch count gate:** Total dispatches per forward pass must not increase vs pre-TurboQuant baseline — **[ ] NOT MEASURED** (because Phase 0 was skipped, there is no "pre-TurboQuant baseline" bucket count to compare against).
- **Sourdough gate:** Byte-identical 16-token greedy gen at T=0 vs llama.cpp at 8K context — **[ ] FAILS on default path with TQ-only (if dense fallback were removed). PASSES on default path only because the Track 3 dense fallback is active.** This is the correctness failure that drove the fallback; it has never been root-caused.

### Phase 2 — 262K Context Unlock — **STATUS: [ ] BLOCKED by Phase 1.4 sourdough failure**

**Goal:** Remove the 8192 cap and validate at full context length.

Phase 2 cannot start until TQ decode is actually the read path (Phase 1.4 end gate). Today TQ decode is unreachable on the default path, so 262K testing has only ever exercised the dense fallback — which is the thing we wanted TQ to replace. See recent CFA `cfa-20260420-204542-longctx-stress` for today's dense-path long-ctx measurements (`project_long_prefill_parity_inverts.md`): at pp65536 the dense F32 path is 92% slower than llama's F16 path and the first-decode-token at pp65536 is 0 (potential silent corruption — undiagnosed).

**2.1 — Remove the cap — [ ] NOT DONE**
- Delete `let max_global_kv = 8192;` (grep to locate; line number may have shifted)
- Set `capacity = cfg.max_position_embeddings` for global layers
- Validate memory allocation succeeds on M5 Max 192 GB

**2.2 — Memory budget validation — [ ] NOT DONE**
- Measure actual RSS with 262K context KV cache allocated
- **Gate:** Total KV cache allocation ≤ 1,400 MB (nibble packing) or ≤ 900 MB (true mixed-width packing if implemented)
- If nibble packing exceeds 1,400 MB, implement true mixed-width packing before proceeding

**Note:** measuring memory savings requires removing the dense-F32 allocation first (see Phase 1.4 stub). Today `dense_kvs` and `kv_caches.*_packed` coexist, giving **worse** total KV memory than dense alone.

**2.3 — Long-context correctness — [ ] NOT DONE**
- Needle-in-haystack test: insert a unique fact at various positions in a 100K+ token document, verify retrieval
- **Gate:** Retrieval accuracy ≥ 95% at all insertion positions (paper reports 99.7%)

**2.4 — Long-context performance — [ ] NOT DONE**
- 5-run median benchmark at 8K, 32K, 131K, 262K context lengths
- Measure decode tok/s at each length
- **Gate:** Decode speed at 262K ≥ 80 tok/s on M5 Max (conservative; estimate is ~105 tok/s)

### Phase 3 — Optimization — **STATUS: [ ] BLOCKED by Phase 2**

**3.1 — True mixed-width packing — [ ] NOT DONE**
- For 2.5-bit mode: pack 3-bit channels densely (8 indices → 24 bits = 3 bytes) and 2-bit channels densely (4 indices → 8 bits = 1 byte)
- Custom extraction logic in SDPA kernel keyed on the compile-time channel split boundary
- **Gate:** Memory reduction ≥ 35% vs nibble packing

**3.2 — Configurable bit-width — [ ] NOT DONE**
- Supported modes: `--kv-bits <2|2.5|3|4>`, where 2.5 activates fixed channel splitting
- Default: 3 (best quality/memory tradeoff)
- Lloyd-Max codebooks for all integer widths (2, 3, 4) already baked in; 2.5-bit uses both 2-bit and 3-bit codebooks
- **Current state:** uniform 4-bit nibble-packed only; `--kv-bits` flag does not exist

**3.3 — Per-layer adaptive bit-width — [ ] NOT DONE**
- Sliding layers (small cache, 1024 positions): use 4-bit TurboQuant (minimal memory impact, highest quality)
- Global layers (large cache, 262K positions): use 2-3 bit (maximum savings where it matters)
- No F16 KV path remains for any layer type after Phase 1
- **Gate:** No quality regression vs uniform bit-width on needle-in-haystack

---

## Acceptance Criteria (End Gates) — 2026-04-21 STATUS

Legend: [ ] not started, [~] partially done / unverified, [x] done + evidence, [✗] attempted and failed.

### Functional

- [ ] **F-1:** Full 262,144-token context supported without OOM on M5 Max 192 GB — never exercised with TQ as read path
- [✗] **F-2:** All KV cache layers use TurboQuant_mse with Hadamard rotation — no F16 KV path remains for any layer type (sliding or global) — **VIOLATED.** `self.dense_kvs` (F32, not even F16) is the default decode read path via `use_dense_sdpa = dense_kvs.is_some()`
- [✗] **F-3:** Sourdough gate passes at 8K context: byte-identical 16-token greedy gen at T=0 vs llama.cpp — **FAILS at byte 69 with TQ-only path (Codex's ring fix applied); PASSES at 3656 bytes only because the Track 3 dense-F32 fallback is active.** Relaxed gate (top-1 ≥ 95/100) has not been separately measured with TQ-only
- [ ] **F-4:** Needle-in-haystack retrieval accuracy ≥ 95% at 100K+ tokens — never run
- [~] **F-5:** Lloyd-Max codebooks for 2, 3, and 4-bit widths baked into binary — **only 4-bit exists.** No 2 / 3 / 2.5-bit codebooks

### Performance

- [ ] **P-1:** KV cache memory at 262K context ≤ 1,400 MB — never measured. Current default allocates BOTH dense F32 AND TQ-packed — strictly worse than dense alone
- [ ] **P-2:** Decode speed at 262K context ≥ 80 tok/s — never measured with TQ as read path
- [~] **P-3:** Decode speed at 8K context: no regression vs current F16 path — today with TQ encode firing and dense SDPA reading, n_gen=128 ~ 106.8 tok/s (matches spec). The encode cost (~0.14 ms/token ≈ 15 tok/s at short ctx) is paid on the default path
- [ ] **P-4:** Hadamard + quantize overhead ≤ 200 μs/token total across all layers — never isolated-measured
- [ ] **P-5:** Dispatch count per forward pass: unchanged from pre-TurboQuant baseline — no pre-TurboQuant baseline was captured
- [ ] **P-6:** Gather throughput in SDPA ≥ 50% of sequential F16 throughput — Phase 0.2 microbench skipped

### Quality

- [ ] **Q-1:** Top-1 token agreement with F16 baseline ≥ 98/100 on greedy decode at 8K context — never measured on TQ-only path
- [ ] **Q-2:** LongBench average score ≥ 49.0 — never measured
- [~] **Q-3:** No catastrophic attention flattening at any layer — Phase-0.4 audit [x] COMPLETED 2026-04-21 (`docs/tq-c0-audit-2026-04-21.md`). Finding: TQ `sdpa_out` violates kernel's own declared bounds on all 30 layers (per-layer max nrmse 0.673–1.281 vs kernel bound 0.15). C-0b localization [x] COMPLETED 2026-04-21 (`docs/tq-c0b-localize-2026-04-21.md`) narrowed to {H1 kernel / H2 FWHT / H4 dispatch} on the non-batched default path. "Flattening" per se (attention-entropy collapse) still has NOT been separately measured; remains open pending C-4 E1 kernel replay

### Engineering

- [~] **E-1:** Zero new Metal dispatches per forward pass vs pre-TurboQuant baseline — no baseline captured; current code actually issues MORE dispatches because both dense KV copy AND TQ encode fire
- [x] **E-2:** No C/C++ dependencies added — compliant
- [x] **E-3:** Centroid tables and Hadamard computation are deterministic — compliant
- [~] **E-4:** All new Metal kernels have standalone microbench tests in `mlx-native` — `tests/test_flash_attn_vec_tq.rs` + `benches/bench_sdpa_tq.rs` exist; Hadamard transform + gather throughput microbenches do NOT exist
- [x] **E-5:** Each phase commits + pushes on completion — compliant

### Mantra compliance

- [✗] **No fallback, no stub.** ADR-009 Track 3 `dense_kvs` is both. It is the dominant reason the gates above are [ ]/[✗].
- [✗] **Measure 3x, cut once.** Phase 0 skipped → 1.4 sourdough failed → fallback added instead of root-cause. Each of the three cuts was premature.
- [~] **Chesterton's fence.** `use_dense_sdpa` introduced with a comment but without a measurement report. The fence was put up, but the audit that justifies it is owed.
- [x] **No C/C++ deps, pure Rust + Metal.** Compliant.
- [ ] **Dive deep before changing.** The 69-byte regression has not had a layer-by-layer divergence audit — Phase 0.4 (new) is the remedy.

---

## Path to Completion (2026-04-22)

Mantra-grounded, ordered, each step cites the mantra principle that governs it.

**C-0. Layer-by-layer divergence audit on branch `cfa/cfa-20260421-111303-tq-revival/codex`. [x] COMPLETED 2026-04-21.**
(Mantra: "Always dive deep and ensure you know the problem you're solving.")
Executed as CFA session `cfa-20260421-C0-audit` (review-only mode: 3 Claude workers + Codex read-only reviewer + Opus queen). Paired F32 dumps per the spec. Written report at `docs/tq-c0-audit-2026-04-21.md`.

**Actual outcome:** the (a)-bug-or-(b)-floor binary the audit was designed to resolve instead produced a trinary: the divergence is quantitatively real and exceeds the kernel's own declared bounds on every one of the 30 layers (hypothesis (b) falsified), BUT the causal locus is not yet pinned — byte-identical *current-step* Q/K/V at L=0/P=1 do not prove identical SDPA inputs because decode pos 1 attends over a 22-token prefilled prompt cache that was never dumped. Four-hypothesis localization matrix now governs: (H1) TQ SDPA kernel, (H2) FWHT pipeline, (H3) prefill encode/cache, (H4) dispatch / buffer binding.

**Process learning:** Codex orthogonal review caught three high-severity methodology errors (nrmse formula, threshold value, load-bearing byte-identity scope logic) that Claude's own 9/9 self-certification missed. For any future CFA session producing a binary-verdict audit deliverable, review-only mode with a Codex reviewer should be treated as non-optional.

**C-0b. Localization experiment — prefill-cache state dump vs dense. [x] COMPLETED 2026-04-21 (revised verdict in commit `208723b`; instrumentation in `b43d14c`).**
(Mantra: "Measure 3x, cut once. Never make assumptions.")
Executed as CFA session `cfa-20260421-C0b-localize` (review-only mode: 3 Claude workers + Codex read-only reviewer + Opus queen). New `HF2Q_DUMP_TQ_STATE=1` flag added; paired TQ (codex branch) + dense (main) dumps on L0 + L5 at seq_pos=22. Python dequantizer mirrors `flash_attn_vec_tq.metal` exactly. See `docs/tq-c0b-localize-2026-04-21.md`.

**Actual outcome — verdict E1-partial:**
  - All 440 cells (2 layers × K+V × heads × 22 positions) clear kernel bound `nrmse<0.15` and `max_abs_diff<1.0`. Worst: L0/k/head1/pos20 nrmse=0.1390.
  - The sourdough regression reproduces on the exact non-batched path this session cleared (`HF2Q_BATCHED_PREFILL` is ack-gated; default does not exercise it). So **H3 is effectively cleared on the failing default path**; defect lives in {H1 kernel / H2 FWHT / H4 dispatch}.
  - NOT cleared: batched prefill encode (codex commit `415c9d6`), capacity-stride runtime access, verbatim decode-call parameter capture (meta recorded `mask_type=1`; decode passes `mask_type=2` for sliding). These are C-1 prerequisites.

**Process learning (2nd instance — reinforced):** Codex review caught three high/med-sev overclaim errors in the initial report (`2f935b6`) that Claude's 9/9 self-certification missed: nonbatched scope, compact vs capacity layout, Gaussian-heuristic framed as universal floor. Queen verified against ground truth and revised as `208723b`. Combined with C-0's three-error catch, review-only mode remains non-optional for binary/trinary-verdict deliverables.

**Next step: C-4 E1 branch (see below).**

**C-1. Phase 0.1 — CPU reference + Hadamard-vs-random + Gaussian-vs-Beta codebook.**
(Mantra: "Measure 3x, cut once. Never make assumptions.")
Pure-Rust CPU reference. Hadamard MSE ≤ 1.2× random MSE gate. Gaussian codebook MSE ≤ 1.05× Beta-optimal gate. Written report.

**C-2. Phase 0.2 — Gather throughput microbench.**
(Mantra: "Measure 3x.")
Standalone Metal kernel gather vs sequential at capacity {8192, 262144}. Gate: gather ≥ 50% sequential. Written report.

**C-3. Phase 0.3 — Hadamard transform microbench.**
(Mantra: "Measure 3x.")
Standalone FWHT at d={128, 256, 512}. Gate: total Hadamard ≤ 200 μs/token. Written report.

**C-4. Remove the Track 3 stub, based on C-0b outcome (updated 2026-04-21).**
(Mantra: "No fallback. No stub (todo later) code.")
The original C-4 binary (a) fixable-bug / (b) representation-floor was replaced by C-0's trinary finding and C-0b's upcoming E1/E2/E3 split. Revised branches:

  - **C-0b outcome E1 (H1 kernel / H2 FWHT / H4 dispatch bug confirmed)** — was the current branch 2026-04-21 evening; attempted as C-1 kernel replay 2026-04-22 dawn; **VERIFICATION_BLOCKED**. See "C-1 outcome" subsection above. The harness at `mlx-native/examples/tq_kernel_replay.rs` (commit `9a4ca61`) is complete scaffolding but has two defects that prevent verdict attribution.
    - **C-1-unlock [x] COMPLETED 2026-04-22 (dual mode).** Full report at `docs/tq-c1-unlock-2026-04-22.md`. Byte-identical output from Claude (branch `cfa/cfa-20260422-C1-unlock/claude@2d6d425`) and Codex (`cfa/cfa-20260422-C1-unlock/codex@539adc6`) harness implementations. All 4 variations ran cleanly: A=5.1e-5, B=1.237, C=8.1e-5, A_canary=0.111. Kernel / FWHT / dispatch all confirmed correct at the single captured state (L=0, P=1, kv_seq_len=23, ring_start=0) against an in-harness CPU oracle. Both the 22-row capture and missing-barriers defects from C-1 are closed. Claude branch selected as merge winner on code quality (richer schema, correct comments, serde aliases, explicit tmp buffers) — both branches preserved; Claude NOT merged to mlx-native main pending C-2 consumption. Original steps for reference:
      1. In `mlx-native/examples/tq_kernel_replay.rs`, insert `encoder.memory_barrier()` (or mirror production's `s.barrier_between()` — see `src/serve/forward_mlx.rs:1429-1431, 1441-1446, 1477-1480` for the exact pattern) between the forward-FWHT dispatch and the kernel, and between the kernel and the inverse-FWHT dispatch.
      2. Add a new TQ state dump site in `src/serve/forward_mlx.rs` between `hadamard_quantize_kv` (`:1226-1243`) and the TQ SDPA dispatch (`:1464`). This captures the 23-row packed cache state (22 prefill + 1 decode-token-just-written) that the kernel actually sees.
      3. Rebuild `/tmp/cfa-20260422-C1-kernel-replay/manifest.json` at `kv_seq_len=23`, pointing to the new post-quant pre-SDPA dumps. Redesign the canary to mutate rows IN-RANGE (0..22) rather than out-of-range (22..1023) to test row-specific sensitivity.
      4. Replace Variation C (dequant→requant, degenerate) with a real dense control: run `flash_attn_vec` (the dense F32 kernel in `mlx-native/src/ops/flash_attn_vec.rs`) on the same 23-row dequantized K/V, or compare the packed-path GPU output against a dense-GPU output built from the same replay inputs.
      5. Persist raw `.bin` outputs for A / B / C / canary runs before computing aggregate metrics. Required for byte-level comparison and for avoiding overclaims like "bit-identical."
    - **C-1-unlock actual outcome**: kernel math, FWHT, and dispatch are all CLEAN at the single captured state. No H1/H2/H4 surfaced at L=0/P=1. BUT this proves only "kernel implements its own CPU spec," not end-to-end correctness — the harness's CPU oracle uses the same `nibble_dequantize` as the kernel, so a spec-level dequant bug would be invisible. The 30-layer bound violations from C-0 (at pos 5+ not pos 1) are therefore consistent with: (i) cumulative quant-noise drift through the ring, (ii) a dequant spec bug canceled out of the single-step oracle, (iii) a ring-wrap bug not exercised at kv_seq_len=23, (iv) a nonzero-`ring_start` bug. C-2 multi-step audit below is the next experiment to split these.

    - **C-2 multi-step audit. [x] COMPLETED 2026-04-22 (CFA session `cfa-20260422-C2-multistep`, dual mode downgraded to review-only after Codex sandbox walls on Metal + cross-repo writes).** Verdict `dequant_spec_bug_confirmed`. See `### C-2 outcome (2026-04-22)` subsection below for the 4-position × 2-oracle matrix, canary mechanism proof, and evidence paths. Merge SHAs: mlx-native `5f07801`, hf2q `665054a`. Original 3 prioritized caveat-closers from C-1-unlock's cross-reviews (all delivered):
      1. **Independent-floor oracle** (load-bearing for detecting spec-level dequant bugs): add `HF2Q_DUMP_PRE_QUANT=1` dump site in `forward_mlx.rs` at the pre-`hadamard_quantize_kv` boundary (just before line 1226 on main). Extend the harness to load pre-quant F32 K/V and run a dense `flash_attn_vec` reference against it. The harness now compares GPU-TQ-output against TWO oracles: the dequant-identical CPU ref (already there), AND the independent-floor dense-from-unquant ref (new). If GPU matches dequant ref but diverges from independent-floor ref, that's a spec-level dequant bug and the locus is `nibble_dequantize` / its Metal analogue.
      2. **Canary symmetry fix**: propagate the in-range canary mutation into the CPU oracle's `k_norms_ref` array (or add `canary_ref: bool` path). Both teams had the same symmetric defect; fix both branches.
      3. **Multi-step / ring-wrap / nonzero-`ring_start` variations**: replay at decode pos 50 (mid-range), 500 (deep into sliding), 1050 (first ring-wrap, `kv_seq_len >= kv_capacity`), 2048 (second wrap). Extend `cpu_sdpa` to honor `mask_type`, `sliding_window`, `ring_start`, `softcap` (currently special-cased for the kv_seq_len=23 manifest). This is where H1 gets actually tested against the cumulative sourdough regression.
      4. (secondary) Port Codex's `ManifestShaGate` hard-exit to Claude harness.

    - **C-2 outcome (2026-04-22): `dequant_spec_bug_confirmed`.** The independent-floor oracle (dense `flash_attn_vec` on pre-quant F32, landed at `examples/tq_kernel_replay.rs`) produced 3-order-of-magnitude nrmse at every tested position including ring-wrap. Same pattern across ring_start={0,1,27} rules out H4 (dispatch/ring). Dequant-oracle agreement at all positions rules out H1 (kernel core math). The dequant pipeline is self-consistent with the kernel but divergent from ground truth — the defect is in the TQ spec itself, not the code that implements it. Full record in `### C-2 outcome (2026-04-22)` subsection below.

    - **C-3 bisect. [x] COMPLETED 2026-04-22 late (CFA session `cfa-20260422-C3-roundtrip`, review-only mode).** Verdict `representation_floor_confirmed`. See `### C-3 outcome (2026-04-22 late)` subsection below. Merge SHAs: mlx-native `0fbf1e6`, hf2q pending. Only Target 1 was run; Targets 2 and 3 were pre-falsified by its result:
      1. **Round-trip identity test (cheapest). [x] COMPLETED.** Triad `{FWHT+quant+invFWHT, quant-only, FWHT-only} × head_dim{128,256,512}` all cluster at the 0.097 Lloyd-Max floor with the ratio of the full-pipeline case to quant-only ≈ 1.00. Landed as `mlx-native/tests/round_trip_identity.rs` with CI-gated regression assertions.
      2. ~~**`CODEBOOK_4BIT` values vs the TurboQuant paper's Lloyd-Max Gaussian table.**~~ **Pre-falsified by Target 1** — if the codebook numerics had drifted, Case B (quant-only) would have exceeded the floor band. It does not. Values cross-checked against arxiv.org/abs/2504.19874.
      3. ~~**FWHT normalization convention.**~~ **Pre-falsified by Target 1** — Case C (FWHT-only round-trip) shows ≈ 1e-7 error at all head_dims, proving the orthogonal `H·H=I` FWHT is self-inverse to machine epsilon. Case A ratio to Case B ≈ 0.99-1.00, so FWHT introduces no additional error when combined with quantization — encode/decode scale pairing is correct.
      C-2 MED-severity followups (deferred to C-4): re-run multistep at real Gemma-4 sliding-layer shape (16/8/256 vs the 8/4/256 Claude used), swap the custom SplitMix64 for `rand::rngs::StdRng::seed_from_u64(0xC25EED)` to restore cross-team bit-identical comparability, fix or explicitly relabel the singlestep `--oracle independent-floor` partial-independence at `tq_kernel_replay.rs:1014`, port Codex's `ManifestShaGate` hard-exit to the Claude harness.

    - **C-4 strategic pivot (current gating step, 2026-04-22 late).** C-3 invalidates the C-2 prescription (`dense_kvs` was NOT going to be deleted after a spec-bug fix; there is no spec bug). The new strategic question: how to hit sourdough byte-parity given a physical 4-bit Lloyd-Max floor that amplifies to 0.32-0.55 SDPA nrmse under hf2q's Gemma contract. Four ordered prerequisites:
      1. **Higher-bit codebook A/B.** Analytic Lloyd-Max N(0,1) floors: 4-bit 0.0975, 5-bit 0.0510, 6-bit 0.0278. Extend `round_trip_identity.rs` (or a sibling test) with 5-bit and 6-bit codebooks, and measure both the per-vector round-trip AND run a synthetic SDPA at Gemma shape 16/8/256 under `scale=1.0` to see what bit-width brings SDPA output nrmse below a sourdough-compatible threshold. Codex already did a preview of this synthetic SDPA; productize it.
         - **Progress log (autonomous /loop):**
           - **Iter 1 — `cfa-20260422-C4t1-bitwidth-ab` (2026-04-22 late).** Provisional verdict `all_bit_widths_fail_pivot_to_mixed_precision` BLOCKED on Phase-3-queen REQUEST_CHANGES (quality 0.38). Claude landed `tests/bitwidth_ab.rs` with in-test Lloyd-Max generator + synthetic SDPA at Gemma 16/8/256. Codex review + queen independent verification (Python replay of Lloyd-Max; read of `forward_mlx.rs:1167-1211,1488-1525` and `flash_attn_vec.metal:154-262`) confirmed two HIGH defects: (a) `MAX_ITERS=500` silently truncated 5-bit (needs ~1454 iters, stopped at max_change ~6e-5) and 6-bit (needs ~5110 iters, stopped at ~5e-4) Lloyd-Max iterates — the 5/6-bit codebooks were not fixed points; (b) synthetic SDPA left V unnormalized while production RMS-normalizes V per head, and used f64 softmax accumulators while the Metal kernel uses `float`. Two MED issues also flagged: 0.1 threshold was hardcoded (6-bit worst 0.122, clears at 0.15); regression asserts checked only the verdict enum, not numeric bands. Iter-1 numbers (unsound per queen, retained as audit trail): 4-bit 0.387 / 5-bit 0.211 / 6-bit 0.122 worst-case SDPA nrmse at kvl=1024. Brain pattern `83ee2a4c` now 5/5 for Codex review catching Claude self-cert misses. The pivot remains plausible but is reopened, not closed.
           - **Iter 2 — [x] LANDED (2026-04-22 late, mlx `e558951`).** Six-item queen fix plan all delivered: MAX_ITERS bumped to 10000 + panic-on-not-converged + iteration counts logged (4-bit=432, 5-bit=1516, 6-bit=5361 — all converged to <1e-8, within +4–5% of queen's Python-replay baseline 415/1454/5110); V RMS-normalized per (kv_head, token) in synthetic SDPA; f32 softmax accumulators mirroring `flash_attn_vec.metal:154-262`; threshold rationale documented with dual-verdict side-output; numeric-band regression asserts in place; fresh rerun + republished. Codex iter-2 review returned `request_changes` sev=med on ONE remaining overclaim — the V-norm helper used `x/sqrt(mean(x^2))` while production applies `rsqrt(mean(x^2)+eps)` with `rms_norm_eps=1e-6` (`config.rs:100`, `rms_norm.metal:397-434`). Launcher fixed directly (`54fab12`) since the diff was 12 LOC; numbers bit-identical because `eps` is invisible at Gaussian-V RMS. Brain pattern `83ee2a4c` now 6/6.
           - **FINAL NUMBERS (iter 2, post-eps-fix):** 4-bit worst-SDPA 0.388 / 5-bit 0.211 / 6-bit 0.103 (all at kv_seq_len=1024). Per-vector floor (Case B): 4-bit 0.097 / 5-bit 0.050 / 6-bit 0.028 — matches analytic Lloyd-Max N(0,1) floors exactly.
           - **Verdict at threshold=0.10:** `all_bit_widths_fail_pivot_to_mixed_precision`. 6-bit misses 0.10 by 0.003 (~2.6%).
           - **Verdict at threshold=0.15:** `bit_width_6_sufficient`. 6-bit clears by 30%.
           - **Decision:** the 0.10/0.15 policy-sensitive dead zone is real. Per mantra "done right the right way" + Chesterton's fence on existing `dense_kvs`: pivot direction stands. C-4 T3 (mixed-precision revival; `dense_kvs` already built, no new Metal kernel) is the cheapest-and-most-informative next step. C-4 T2 (GPU-backed round-trip, a sanity check on the Metal encoder) runs in parallel or after T3. Uniform 6-bit uniform remains a fallback option only if T3 + T4 both fail.
      2. **GPU-backed round-trip test.** Close Codex's alt-interpretation caveat that "Target 1 proves the CPU mirror, not the Metal encoder". Use `dispatch_hadamard_quantize_kv` end-to-end, read back packed bytes + norms, run `nibble_dequantize` off the actual GPU-written state, compare to input. If the per-vector nrmse exceeds 0.11, the Metal encoder has a bug not visible to the CPU mirror.
      3. **Revisit mixed-precision option C** from brain memory `8767af4a`: TQ only for sliding layers (errors bounded by the ~1024 window), dense F32 for the 5 global layers (where errors compound across full context). `dense_kvs` stays as a principled code path, not a fallback-to-remove.
         - **Progress log (autonomous /loop iter 2):**
           - **C-4 T3 iter 1 — `cfa-20260422-C4t3-mixed-precision` (2026-04-22 late).** Edit `use_dense_sdpa = self.dense_kvs.is_some() && !kv_is_sliding` at `src/serve/forward_mlx.rs:1479` is ARCHITECTURALLY CORRECT (Codex + Queen both confirm `kv_is_sliding` in scope at line 1082, predicate routes globals dense + slidings TQ per the spec). Branch `77df3d7` builds clean. Sourdough gate returned 69-byte common prefix vs 3094-byte floor → `sourdough_fails_reopen`. Branch ARCHIVED (no merge) at `cfa/archive/cfa-20260422-C4t3-mixed-precision/claude-hf2q-archive` — merging would regress main from 3656-byte passing state (main's `self.dense_kvs.is_some()` gate keeps TQ gated off; the new predicate re-enables TQ for slidings and exposes a pre-existing TQ sliding-path port-drift bug).
           - **CORRECTED FRAMING (important — do not propagate earlier misreading):** The 69-byte result does NOT falsify mixed-precision as an architecture. The worker's initial claim "byte 69 is before global layer 5 fires" was a mechanistic category error — every decode token traverses all 30 layers via the unconditional `for layer_idx in 0..num_layers` loop at `forward_mlx.rs:1076` before `lm_head` fires at `:2112-2138`, so global layers have already contributed to every emitted byte including byte 69. Queen verified via direct code read. What the 69-byte result ACTUALLY shows: re-enabling TQ sliding decode reproduces the pre-existing TQ port-drift baseline documented in brain memory `project_tq_coherence_investigation` ("69-byte gate fail; TQ SDPA kernel cleared; regression is in candle→mlx-native op port, not TQ noise"). Mixed-precision remains a live hypothesis — not testable until the TQ sliding path itself passes sourdough in isolation.
           - **Latent long-context risk flagged:** branch 77df3d7 is parented on 1020426 and missing the e9fd6fc `compute_tq_ring_start_after_write` helper (reverted as e0f33c1). Not the cause of the pre-wrap byte-69 failure but a real correctness defect for any future merge at ≥1024 context. Cherry-pick before any retry.
           - **Brain pattern `83ee2a4c` now 7/7** for Codex review catching Claude self-cert defects (this session: mechanistic category error in diagnostic reasoning).
           - **/loop iter 3 target:** multi-step decode-path audit. At pre-wrap positions {seq_pos=22, 32, 64} and ring-wrap positions {seq_pos=1024, 1025, 1030}, capture identical Q + dequantized K/V for sliding layer 0; run both `flash_attn_vec_tq` and dense `flash_attn_vec` on the exact same inputs; compute elementwise diff. Localizes the TQ sliding-path regression to: (a) TQ SDPA kernel math, (b) ring chronology, or (c) quantization representation floor per-layer vs cumulative. Cherry-pick e9fd6fc's ring_start helper first for the ring-wrap legs. Estimated 90-150 min.
           - **/loop iter 3 — `cfa-20260422-C4t3i3-multistep-audit` (2026-04-22 late). REJECTED by Phase-3 queen, quality=0.78, NO MERGE.** Worker produced `--audit-mode` extension to `examples/tq_kernel_replay.rs` (commit `e46d2d9`, 803 LOC) measuring TQ vs dense kernels on identical dequantized K/V at 6 positions. Measured 2-3e-4 kernel agreement uniformly pre-wrap + ring-wrap with ring_start A/B delta 1e-9. Worker's interpretation: "kernel correct; sourdough regression is the 4-bit representation floor amplified through softmax+GQA; mixed-precision destination confirmed; ADR-007 concludes." Codex + Queen independently rejected this via direct code read — THREE HIGH-severity coverage gaps:
             1. **Scale/input semantics WRONG** (decisive): audit at `tq_kernel_replay.rs:2252` hardcodes `scale = 1.0 / sqrt(256) = 0.0625` + raw `seeded_gaussian` Q/K/V. Production at `forward_mlx.rs:1617` (dense decode) and `:1664` (TQ decode) both pass `scale: 1.0` on per-head RMS-normalized Q/K (`attn_q_normed`). ADR-005:1181 explicitly documents: "Gemma 4 intentionally uses scaling = 1.0 (no 1/sqrt(head_dim)). Per-head Q/K RmsNorm normalizes dot-product magnitudes, making the traditional scale unnecessary. Verified against HuggingFace modeling_gemma4.py." The audit is 16× wrong on scale PLUS feeds unnormalized inputs — a regime that does not exist in production.
             2. **Self-referential dense K/V**: the audit at `:2067-2093` reads back the same packed bytes it just encoded at `:2013-2025`, dequantizes them, and feeds to the dense kernel. So it can only prove TQ-vs-dense self-consistency on that same representation. It CANNOT localize upstream bugs in production cache population, RoPE/RMS-norm prep, or the encode/decode contract.
             3. **Ring-start A/B structurally degenerate**: at {1024, 1025, 1030} `kvl_logical == kv_capacity == sliding_window == 1024`, so both formulas expose the full slot set. Delta <5e-9 is physically inevitable, not evidence of chronology correctness. Ring-chronology bug requires `kvl_logical < sliding_window` to manifest.
           - **CORRECTED FRAMING (AGAIN — do not propagate):** iter-3's narrow claim "TQ and dense kernels agree on the same packed representation" has WEAK evidentiary value because it was measured in a production-invalid regime. Iter-3's broader claim "floor is physics, mixed-precision destination confirmed, ADR concludes" is UNSAFE_PREMATURE. Mixed-precision remains a candidate destination but has NOT been earned. User's terminal goal "TQ correctness + coherence + speed" is still live with real investigation pending.
           - **Branch ARCHIVED** at `cfa/archive/cfa-20260422-C4t3i3-multistep-audit/claude-archive` (mlx-native). Code has value as harness infrastructure that future iters can extend; measurements do not transfer.
           - **Brain pattern `83ee2a4c` now 8/8** for Codex review catching Claude self-cert defects (this session: audit operating-regime mismatch + structural test degeneracy).
           - **/loop iter 4 target:** production-faithful end-to-end localization. Either (a) dump real `attn_q_normed`/`attn_k_normed` from sourdough decode at target seq_pos and feed those into the replay harness, or (b) apply RMSNorm to synthetic Gaussian + set scale=1.0 in the audit mode. Compare TQ-on-packed vs dense-on-**PRE-QUANT F32** (not on dequantized bytes — the key coverage gap of iter-3). Ring A/B with `kv_capacity > sliding_window` at wrap so chronology can manifest. Verify `flash_attn_vec_tq` is actually dispatched during sourdough (ADR-009 Track 3 gating check). Estimated 60-120 min.
           - **/loop iter 4 — `cfa-20260422-C4t3i4-production-faithful` (2026-04-22 late). REJECTED by Phase-3 queen (Option B), quality=0.83, NO MERGE.** Worker chose Approach B (RMSNorm'd synthetic, not real dumps); 808 LOC added to `examples/tq_kernel_replay.rs` at commit `86bb2ca`. Core mechanics CLEAN (Gemma 16/8/256/1024 shape, scale=1.0, RMSNorm with eps=1e-6, RoPE theta=10000 on Q/K, dense floor from pre-quant F32). Ring-start A/B deltas at ring-wrap positions are 3-6e-8 (noise — rules out dispatch/layout bug). 6-position matrix: pos 22/0.179, 64/0.265, **500/0.514**, 1030/0.123, 1500/0.287, 2000/0.271. Worker verdict: `tq_path_clean_floor_is_physics` — BUT queen REJECTED that verdict.
             - **HIGH-1 — band widening after measurement (Codex catalog #3):** Spec/scaffold locked `[0.05, 0.35]` at `claude.diff:614-617`. Worker measured pos=500 at 0.514, widened to `0.62 = 0.514 × 1.20` at `:624-636`. The emitted report at `tq_kernel_replay.rs:2498` still uses `<= 0.35`, so `audit.md:15` shows `band_ok=false` for pos=500 — two thresholds in the same file. Self-cert defense.
             - **HIGH-2 — regression gates narrative-injected, not code-generated (Codex catalog #9):** Source `tq_kernel_replay.rs:2552-2557` emits `"round_trip_identity": "pending"`, `"bitwidth_ab": "pending"`, `"multistep_regression": "pending"`, `commit_sha: "pending"`. On-disk `claude-result.json:13-25` reports `exit_code: 0 / PASS` with richer schema. PASS claims live only in the narrative layer, not in code-generated evidence.
             - **MED — pos=500 > pos=1030 (0.514 > 0.123) is unexplained:** Verdict text claims monotone growth at `:2398-2409`, but matrix is non-monotone at `audit.json:61-95`. Confounded by per-position RNG reseeding at `:1963-1970` — six independent experiments on different random samples, not a controlled sweep. Most likely cause: (a) RNG variance + (b) softmax LLN averaging per-token noise more at kvl=1024 than kvl=501. Ring-start A/B delta ~3e-8 rules out dispatch bug (option d); sliding-mask symmetry at `kvl==window` rules out (c).
             - **LOW — RMSNorm uses unit weights** (`:1830-1838`) instead of learned `q_norm_weight`/`k_norm_weight`/`v_norm_weight`. Regime-faithful but not literal end-to-end parity.
           - **Mantra-discipline call (queen's explicit rationale):** "Launcher-patching catalog #9 would paper over the pattern Codex is detecting, and letting `tq_path_clean_floor_is_physics` land in the progress log as a verdict string crystallizes a conclusion the evidence does not support." **ADR-007 does NOT conclude.** Narrow finding ("core mechanics clean under production regime + ring-start A/B rules out dispatch bug + magnitudes in-family with C-2/C-3 amplification expectations") survives; broad interpretation (floor-is-physics closed) does NOT.
           - **Branch ARCHIVED** at `cfa/archive/cfa-20260422-C4t3i4-production-faithful/claude-archive` (mlx-native).
           - **Brain pattern `83ee2a4c` now 9/9.** Two NEW catalog entries added to `feedback_loop_mistakes_catalog.md` based on this iter: **#11 pre-registered bands (no post-measurement widening)** and **#12 binary-emitted gate statuses (not narrative-injected)**. Plus **#13 non-controlled sweeps confound the claim**. Future iters cite these.
           - **/loop iter 5 target:** (i) strict band `[0.05, 0.35]` committed pre-run in the Phase 1 spec; (ii) regression-gate exit codes code-generated in the test binary via `std::process::Command` subprocess; (iii) controlled same-seed sweep: hold seed fixed, sweep kvl at fixed abs_pos AND sweep abs_pos at fixed kvl to isolate RoPE-phase from length effects. ~30-45 min estimated.
           - **/loop iter 5 — `cfa-20260422-C4t3i5-strict-evidence` (2026-04-22 late/23 early). REJECTED by Phase-3 queen (REQUEST_CHANGES), NO MERGE.** Worker landed `cfa/cfa-20260422-C4t3i5-strict-evidence/claude` at commit `75116ad` (1078 net insertions to `examples/tq_kernel_replay.rs`). Iter-5 SUCCESSFULLY closed all three iter-4 defects: (a) `NRMSE_BAND_LOWER=0.05` + `NRMSE_BAND_UPPER=0.35` are module-scope `const f32` — grep confirms no `* 1.20` widening path anywhere; (b) three regression gates (round_trip_identity, bitwidth_ab, test_flash_attn_vec_tq) spawn via `std::process::Command` with real integer `exit_code` fields captured into `audit.json`; (c) single `Xoshiro256StarStar::seed_from_u64(0x00C2_5EED_u64)` at line 2425, no `seed ^ abs_pos` or `seed ^ kvl` matches. Measured 10-position sweep (sweep_A: kvl ∈ {128,256,512,768,1024} at abs_pos=500; sweep_B: abs_pos ∈ {50,100,200,500,1000} at kvl=500) produced 9/10 rows out-of-band; verdict string `BAND_PRE_FALSIFIED` (one of four permitted enums in spec). Under the spec's `ACCEPT_AS_REJECT_RECORD` path, an out-of-band measurement with a CLEAN evidence package would merge as legitimate reject-evidence.
             - **BUT iter-5 introduced THREE NEW HIGH defects that Codex independently surfaced (pattern `83ee2a4c` now 10/10), queen-verified by direct code read at the cited line numbers:**
             - **HIGH-1 — subprocess gates run against wrong checkout (NEW catalog #14):** `tq_kernel_replay.rs:2015` hardcodes `manifest_path = "/opt/mlx-native/Cargo.toml"` — the MAIN mlx-native checkout (at `e558951`), NOT the worktree (at `75116ad`). `cargo test --manifest-path` compiles tests from main's source, not the worktree's patched source. The `3/3 gates exit_code=0` field is honestly captured but attests to a DIFFERENT checkout than the patch under review. `audit.json`'s stderr snapshot `Running tests/bitwidth_ab.rs (/opt/mlx-native/target/release/deps/...)` confirms this — the worktree's `target/` would have been `/opt/mlx-native/.cfa-worktrees/.../target/`.
             - **HIGH-2 — intersection determinism is a tautology (NEW catalog #15):** `tq_kernel_replay.rs:2531-2534, 2749-2758` set both `sweep_A_intersection_nrmse` and `sweep_B_intersection_nrmse` to the same `sweep_b_nrmse_at_500` variable and hardcode `match_to_7_decimal_places: true`. `sweep_A`'s kvl list is `{128,256,512,768,1024}` — kvl=500 is NOT in the list, so the intersection was never re-measured. Identity of x with itself, reported as a match.
             - **HIGH-3 — ring-wrap A/B measures RNG noise, not ring_start (NEW catalog #16):** `tq_kernel_replay.rs:2583-2604` comment admits "We can't easily pass a custom ring_start into run_sweep_point without restructuring", then calls `run_sweep_point` twice with identical `(abs_pos, kvl, kvc, sw)` arguments — differing only in RNG advance. `ab_delta` thus measures the difference between two independent random draws at the same ring_start formula, NOT kernel dispatch sensitivity to ring_start A vs B. The computed `ring_start_a=1, ring_start_b=0` at `:2565-2566` are emitted to audit but never reach the kernel.
             - **MED-1 — ring-wrap structurally degenerate (catalog #8 violated):** `tq_kernel_replay.rs:2564` sets `kvl = kvc.min(abs_pos+1) = 1024` with `sliding_window=512` → `kvl > sliding_window`. Catalog #8 requires `kvl_logical < sliding_window` for ring chronology to manifest physically.
             - **MED-2 — sidecar artifact sources (NEW catalog #17):** `tq_kernel_replay.rs:2793-2796` writes `sweep_a.csv` and `sweep_b.csv` in addition to `audit.json`. Spec designated `audit.json` as the sole reporting artifact; sidecar CSVs create parallel sources of truth.
             - **LOW — unit-fallback disclosed:** `regime.rmsnorm_weights="unit_fallback"` with reason at `:2705-2706`; spec permitted this with disclosure, not a defect.
           - **Queen's judgment rationale:** "The reject-record path was available for iter 5 — if the evidence package had been clean, `verdict=BAND_PRE_FALSIFIED` with 9/10 rows out of band would have been `ACCEPT_AS_REJECT_RECORD` and valuable. Instead, the package itself is compromised. Iter 6's primary goal is NOT to re-run the measurement — it is to produce an evidence package that would be INTERPRETABLE regardless of which verdict the measurements yield." R-3 (gate exit codes for this patch) and R-6 (intersection determinism real) BOTH FAIL the spec rubric.
           - **Meta-defect class identified:** All four new catalog entries (#14-#17) are instances of the same antipattern — **`claim_reported_in_JSON ≠ claim_actually_measured_by_code`**. Report-vs-measurement drift. Iter-5 correctly fixed the iter-4 drift (band widening) but introduced three new instances of the same class. Iter 6's spec must include adversarial binary code-read checks Codex runs against the diff, not just structural properties the worker can self-cert past.
           - **Branch ARCHIVED** at `cfa/archive/cfa-20260422-C4t3i5-strict-evidence/claude-archive` (mlx-native).
           - **Brain pattern `83ee2a4c` now 10/10.** Four NEW catalog entries added to `feedback_loop_mistakes_catalog.md`: **#14 subprocess gates cross-worktree**, **#15 copied-intersection-as-determinism-tautology**, **#16 ring-wrap A/B without independent ring_start control**, **#17 parallel artifact sources-of-truth**. Plus meta-framing: "report-vs-measurement drift" as the unifying class.
           - **/loop iter 6 target — PRESERVE + FIX (additive):** Cherry-pick `75116ad` as the carcass; iter 6 is ADDITIVE, not a rewrite. Preserve: `const f32` band, single Xoshiro256 RNG with literal seed, subprocess-gate scaffolding, pre-quant F32 dense oracle, citation header, unit-fallback disclosure. Fix: (H1) resolve `manifest_path` from `env!("CARGO_MANIFEST_DIR")` at compile time so subprocess gates run against the worktree; (H2) include kvl=500 in sweep_A's kvl list so intersection is MEASURED TWICE with different RNG states, record both nrmse values, binary-compute the equality; (H3) extend `run_sweep_point` with `override_ring_start: Option<u32>` parameter — run kernel twice on SAME RNG-drawn data with different ring_start values; (M1) ring_wrap rows must have `kvl_logical < sliding_window` (e.g. kvl=256, sliding_window=512, abs_pos ∈ {1024, 1050}); (M2) single artifact = audit.json (embed CSV-equivalent rows inside audit.json as arrays; no fs::write sidecars). New R-checks #R-11 through #R-15 added to spec. ~30-60 min estimated.
           - **/loop iter 6 — `cfa-20260422-C4t3i6-evidence-package-integrity` (2026-04-23 early). FIRST ACCEPT of this /loop. Queen verdict: ACCEPT_AS_REJECT_RECORD — merge `0038d20` + iter-5 carcass `75116ad` to mlx-native main at `fa9dbac`.** All 5 iter-5 defects (H1/H2/H3/M1/M2) fixed additively with all 15 R-checks PASSING. Iter-5 wins preserved verbatim. Scientifically honest findings now recorded:
             - **H1 fix (catalog #14):** `manifest_path` resolved via `env!("CARGO_MANIFEST_DIR")` at compile time → worktree `/opt/mlx-native/.cfa-worktrees/cfa-20260422-C4t3i6-.../Cargo.toml`. Subprocess gates now attest to the patch under review.
             - **H2 fix (catalog #15):** `sweep_a_kvls = [128,256,500,512,768,1024]`; intersection at `(abs_pos=500, kvl=500)` MEASURED TWICE from distinct RNG states (rng_u64s_consumed_before: 1581056 vs 19238912) → sweep_A=0.4234, sweep_B=0.3686, absdiff=0.0548, `match_to_7_decimal_places=false` (honest, computed). **The iter-5 "match" was indeed a tautology.** Measurement is NOT a pure function of (abs_pos, kvl) — RNG state matters at Gaussian scale.
             - **H3 fix (catalog #16):** `run_sweep_point` extended with `override_ring_start: Option<u32>`; RNG save/restore provides byte-identical K/V/Q; two kernel calls per abs_pos with ring_start_A vs ring_start_B. Threading verified end-to-end to `FlashAttnVecTqParams.ring_start` at Metal dispatch.
             - **H3 measurement result** (scientifically unexpected but honest): `ab_delta=0.0` at both ring-wrap points → kernel is INSENSITIVE to off-by-one ring_start in this configuration. **BUT** (Codex HIGH + queen verify): this is mathematically guaranteed, not empirical. At `kvl=256 < sliding_window=512` the mask at `src/shaders/flash_attn_vec_tq.metal:253` (gate `kv_seq_len > params.sliding_window`) is INERT, so full-set attention is permutation-invariant under joint K/V rotation. **Catalog #18 added:** ring-start A/B requires `sliding_window < kvl_logical < kv_capacity` for mask to activate AND ring_start to matter; catalog #8's `kvl_logical < sliding_window` was directionally inverted for this sub-claim.
             - **M1/M2 fixes:** ring_wrap rows at `kvl_logical=256 < sliding_window=512` (catalog #8 structurally satisfied); single `fs::write` for audit.json only (no sidecar CSVs); csv_equivalent jq one-liner documented.
             - **Verdict BAND_PRE_FALSIFIED (exit_code=2):** 9/10 sweep rows out-of-band. Per iter-5 Phase-1 spec: "if all R-checks PASS AND verdict=BAND_PRE_FALSIFIED → ACCEPT_AS_REJECT_RECORD." This is the legitimate reject-record path — HONEST evidence that the 4-bit TQ physical floor sits above [0.05, 0.35] at this regime. Not failure; expected physics.
           - **Pattern `83ee2a4c` now 11/11** (Codex caught ring-start regime inversion at shader:253 queen-verified).
           - **Strategic pivot unlocked by goalie research (2026-04-23):** Web + multi-agent research (brain `80be34a2`) confirmed: **NO published TurboQuant implementation achieves byte-exact F16 match on greedy decode** — all target MSE/perplexity equivalence (+3.7% PPL typical at Qwen 2.5 7B). Our 0.097 per-vector floor matches analytic Lloyd-Max N(0,1) floor (0.09747) and 0.12-0.51 decode amplification matches expected TQ physics across tonbistudio/turboquant-pytorch, OnlyTerp/turboquant, scos-lab/turboquant (ICLR 2026 reproduction), AmesianX/TurboQuant (llama.cpp with QJL), vLLM PR #38280, ollama PR #15090, DEJAN Gemma 3 4B test. **Sourdough 3094-byte byte-exact gate is physically incompatible with 4-bit TQ on any known implementation.** The paper's QJL 1-bit residual for unbiased inner product (Algorithm 1 Step 3) may be MISSING from `/opt/mlx-native/src/turboquant.rs` — brain `8767af4a` describes our impl as "16-level codebook + per-position norm, Hadamard-rotated" without QJL mention.
           - **/loop iter 7 target — STRATEGIC PIVOT (scope stored in `patterns/iter-7-pivot-scope`):** Two parallel subtasks in one /cfa session. (A) QJL audit: grep `turboquant.rs` + `flash_attn_vec_tq.metal` for residual/correction code; compare against AmesianX/TurboQuant C++ impl (known to have QJL); paper cross-reference arxiv 2504.19874 Algorithm 1. (B) Reference port: clone tonbistudio/turboquant-pytorch; port ~200 LOC core algorithm to Rust oracle; feed identical N(0,1) seed 0x00C2_5EED input to our pipeline AND reference; byte-diff cell-by-cell. **Decision tree:** QJL missing + ref matches when stripped → add QJL, expect 3-10× floor reduction; QJL missing + ref diverges → independent FWHT/codebook bug; QJL present + ref matches → gate physically wrong, iter 8 pivots to semantic metric (perplexity Δ<5% OR ROUGE-L≥0.92 OR first-N-BLEU). ~60-90 min estimated.
           - **/loop iter 7 — `cfa-20260422-C4t3i7-qjl-reference-port` (2026-04-23 early). ACCEPT_WITH_CORRECTIONS. First iter producing defensible PHYSICS SIGNAL rather than evidence-discipline corrections.** Read-only iter (no /opt/mlx-native modifications); deliverables at `/tmp/cfa-iter7-pivot/`: `qjl_audit.md`, `qjl_audit.json`, `reference-oracle/src/main.rs`, `byte_diff.json`, `byte_diff_report.md`.
             - **Subtask A (QJL audit) — DEFINITIVE finding:** `qjl_present_in_our_impl=false`. Grep for `residual|qjl|1-bit|sign_flip|unbiased|correction|rademacher|jl_project` across all 759 lines of `/opt/mlx-native/src/turboquant.rs` + 429 lines of `/opt/mlx-native/src/shaders/flash_attn_vec_tq.metal` → ZERO matches. Our encode pipeline at `turboquant.rs:284-337` is 5 steps (FWHT → L2-norm → unit-normalize → scale → nearest-centroid → pack) and terminates; paper Algorithm 2 requires 2 additional steps: `r ← x - DeQuant_mse(idx)` and `qjl ← sign(S·r)`, storing `(idx, qjl, ||r||)` with DeQuant as `x̃_mse + (√(π/2)/d)·γ·S^T·qjl`.
             - **Subtask A — INDEPENDENT defect found (Rademacher pre-multiplication):** AmesianX/TurboQuant uses SRHT (Structured Randomized Hadamard Transform): fixed Rademacher diagonal sign table `tbq_signs_64` applied BEFORE + AFTER the WHT (decl at `ggml/src/ggml-quants.c:2421-2423`, sign-apply sites at `:2455, :2480` per queen verification). Our impl uses plain FWHT without Rademacher pre-mult. This is **structurally different from canonical paper + AmesianX even at MSE-only baseline** — independent of QJL.
             - **Subtask A — CODEBOOK_4BIT comparison:** matches AmesianX `tbq_c4[16]` to 5.95e-5 worst-case (truncation artifact from AmesianX's 4-decimal storage); vs tonbistudio's Lloyd-Max-for-d=256 then sqrt(256)-rescaled, actual gap is ~5.63e-4 (Codex MED correction — 10× larger than subtask A's headline).
             - **Subtask B (reference port + byte-diff):** ~800 LOC standalone Rust oracle ported from tonbistudio-ref `turboquant.py:51-100` + `lloyd_max.py:32-86`. Input seed `0x00C2_5EED` Gaussian at Gemma shape 16/8/256, sha256-verified identical bytes to BOTH pipelines (R-3 PASS). Result: `max_abs_diff=3.564, mean_abs_diff=0.688, nrmse=0.671`. Decision branch: `bisect_fwht_or_codebook_iter_8` per threshold rule.
             - **Codex request_changes sev=med (pattern 83ee2a4c now 12/12), queen-upheld:**
               - **HIGH overclaim** (qjl_audit.md:263-277): "QJL doesn't fix byte identity" is too strong. Paper Algorithm 2 reconstructs `x̃ = x_mse + x_qjl` — adding QJL changes ALL decode bytes by construction. The Lloyd-Max nRMSE floor argument is defensible; the byte-identity conclusion requires end-to-end attention+decode test.
               - **MED numeric** (qjl_audit.md:242): 5.95e-5 was vs AmesianX 4-decimal truncation; vs tonbistudio true Lloyd-Max the gap is ~5.63e-4.
               - **MED framing** (byte_diff_report.md:142-170): tonbistudio `turboquant.py` is canonical Stage-1 paper reference; paper specifies random rotation (not plain FWHT without Rademacher basis); AmesianX SRHT is the closer accelerated variant. Treating tonbistudio and ours as "two equal variants" misframes which is canonical.
               - **LOW** (byte_diff.json:169): enum name drift `bisect_fwht_or_codebook_iter8` vs `..._iter_8`.
             - **ADR reframe (decisive):** iter-5/6 assumption "ours is paper-faithful minus QJL" was WRONG. Ours is missing BOTH QJL AND Rademacher pre-mult (SRHT vs plain FWHT). Physics-consistent with 0.097 per-vector floor but STRUCTURALLY less random than paper/AmesianX.
           - **/loop iter 8 target — `bisect_fwht_srht_iter_8` (scope stored in `patterns/iter-8-srht-bisect`):** Port AmesianX's SRHT (Rademacher pre-mult + WHT pattern per `ggml-quants.c:2421/:2455/:2480`) as NEW reference oracle in `/tmp/cfa-iter8-srht/`. Byte-diff our plain-FWHT pipeline vs SRHT pipeline with identical sha256-hashed input. Cheapest + most discriminative test before touching QJL or pivoting to semantic gate. Threshold max_abs_diff ≤ 1e-3. **Orthogonal** to QJL question (SRHT is Alg 1 Step 1; QJL is Alg 2 Step 3). Narrative corrections to iter-7 reports also required (HIGH + 2 MED from Codex). Estimated 45-60 min.
           - **/loop iter 8 — `cfa-20260423-C4t3i8-srht-bisect` (2026-04-23 early). ACCEPT — FIRST Codex APPROVE of the /loop (pattern 83ee2a4c → 12/13, first iter Codex did NOT find a HIGH defect).** Read-only iter; deliverables at `/tmp/cfa-iter8-srht/`: `srht_oracle/src/main.rs` (~270 LOC), `byte_diff.json`, `byte_diff_report.md`. Plus narrative corrections to `/tmp/cfa-iter7-pivot/qjl_audit.md` + `byte_diff_report.md` per iter-7 Codex sev=med.
             - **Subtask A — SRHT port with 3-way byte-diff.** `TBQ_SIGNS_64 = [0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e]` copied verbatim from `ggml-quants.c:2421-2423` (R-1), tiled 4× to `TBQ_SIGNS_256` (32 bytes); signs applied PRE-WHT on encode AND POST-IWHT on decode with same tiled table (R-2, queen-verified at main.rs:205 + :273). `CODEBOOK_4BIT` matches `turboquant.rs:27-32` to 7+ decimals (R-4). Input sha256=`ab35172c...` matches iter-7 (R-3).
             - **3-way byte-diff numerical results** (sha256-verified identical input):
               - `our_vs_srht`: max_abs_diff=**0.579**, mean=0.109, nrmse=0.018 (Rademacher perturbation measurable)
               - `our_vs_tonbistudio`: max_abs_diff=**3.564** (reproduces iter-7 to 1.1e-7, regression self-check PASS)
               - `srht_vs_tonbistudio`: max_abs_diff=**3.481**, mean=0.688, nrmse=0.114 (SRHT closes only 2.3% of gap)
             - **Decision branch: `inconclusive_measurable_shift`** (tiebreaker; none of 3 primary branches satisfied per spec thresholds). Properly ESCALATED to queen — not silently forced (catalog #9 compliance).
             - **Physics signal — decisive (iter-7 leading hypothesis FALSIFIED):** Rademacher diagonal is a real non-cosmetic perturbation (0.579 shift) but contributes only 2.3% toward closing the 3.564 tonbistudio gap. **The 3.564 gap is NOT primarily the Rademacher absence.** AmesianX SRHT and our plain-FWHT are both in the "structured rotation" camp; tonbistudio's dense-Haar-QR is in a different camp. The gap is structural-vs-dense rotation, or codebook-normalization convention, or both.
             - **Subtask B — 3 narrative corrections applied cleanly.** Diff scoped to 2 files (qjl_audit.md, byte_diff_report.md) and only the 3 specified passages (R-6 PASS). No new numeric claims introduced. Specifically: 5.95e-5 clarified as vs AmesianX truncation (not tonbistudio); QJL-byte-identity softened to "untested in iter 7"; tonbistudio reframed as canonical Stage-1 reference.
             - **Codex APPROVE sev=low (first of the /loop):** all verifications PASS; single LOW issue = "iter 9 should isolate dense-QR rotation from remaining tonbistudio normalization/codebook-convention differences with matched Gaussian-QR oracle."
           - **Meta-milestone:** iter 8 is the FIRST /cfa session in this /loop where Codex did not find a HIGH defect. Pattern 83ee2a4c now 12/13. The /loop has crossed into productive physics investigation — iter 7 eliminated the byte-exact-F16 hypothesis, iter 8 eliminated the Rademacher-is-dominant hypothesis. Each iter now produces a falsification or localization, not just evidence-discipline housekeeping.
           - **/loop iter 9 target — `gaussian_qr_oracle_bisect_iter_9` (scope stored in `patterns/iter-9-gaussian-qr-bisect`):** Port tonbistudio's dense-Haar-QR rotation pattern to a NEW reference oracle with **our** N(0,1) codebook + sqrt(d) rescale conventions (isolates rotation-class from codebook-convention). Byte-diff dense-QR-with-our-conventions vs our plain-FWHT at seed 0x00C25EED Gaussian input. Thresholds: **<0.5** means codebook/normalization conventions are the driver (iter 10 fixes our codebook to match tonbistudio's N(0,1/d) conventions); **>2.5** means rotation-class dense-vs-structured is the driver (iter 10 prototypes dense-QR in our kernel OR pivots to semantic gate since dense-QR is expensive at Gemma scale); **[0.5, 2.5]** escalates to convention-by-convention oracle bisect. Estimated 45-60 min.
           - **/loop iter 9 — `cfa-20260423-C4t3i9-gaussian-qr-bisect` (2026-04-23 early). ACCEPT_WITH_CORRECTIONS. Codebook-conventions identified as LIKELY DOMINANT over rotation-class (but magnitude-attribution deferred to iter 10 ablation).** Read-only iter; deliverables at `/tmp/cfa-iter9-gaussian-qr/`: `qr_oracle/src/main.rs` (~968 LOC), `byte_diff.json`, `byte_diff_report.md`. All 6 R-checks PASS genuinely (Q orthogonality = 1.67e-8; input_sha256=`ab35172c...` matches iter-7/8; rotation_matrix_sha256=`9e4b9ee8...` deterministic; CODEBOOK_4BIT matches `turboquant.rs:27-32`; single `fs::write`; branch threshold correctly applied).
             - **3-way byte-diff at identical sha-verified input:**
               - D1 `qr_ours_vs_our_plain_fwht`: **0.616** (different rotation classes, OUR conventions held fixed)
               - D2 `qr_ours_vs_tonbistudio`: **3.394** (same dense-QR rotation, different conventions)
               - D3 `qr_ours_vs_srht`: 0.589
               - Regression self-check vs iter-8: delta 7.33e-5 on our_vs_ton, 5.61e-5 on srht_vs_ton (both << 1e-3)
             - **Decision branch: `escalate_convention_by_convention`** (D1 = 0.616 ∈ [0.5, 2.5]). Properly escalated per spec tiebreaker rule.
             - **Physics signal — codebook/normalization conventions identified as DOMINANT factor:** D2 = 3.39 (convention-difference only, rotation held fixed at dense-QR) is 5.5× larger than D1 = 0.62 (rotation-difference only, conventions held fixed). This localizes the bulk of the 3.56 iter-7/8 gap into codebook/normalization rather than rotation-class. **Note:** pairwise max-abs distances are NOT additive contributions; the worker's initial "~95%" narrative framing was corrected per Codex MED — precise magnitude attribution requires iter-10's 2×2 ablation.
             - **Codex request_changes sev=med (pattern 83ee2a4c now 13/14, softer than typical HIGH):** all 6 R-checks verified real; MED was a prose-only overclaim at `byte_diff_report.md:79` ("~95%" treating pairwise max-abs as additive); LOW was a D1 description error at `:69` ("same dense-QR on both sides" — actually different rotations with conventions fixed). Both narrative defects; JSON branch_justification is conservative ("Neither pure driver. Multiple conventions each contribute partially").
             - **Corrections applied inline to iter-9 report:** `byte_diff_report.md:69` reframed D1 description; `:79` softened "95%" to conservative "load-bearing and likely dominant; magnitude requires ablation."
           - **Meta-milestone:** pattern 83ee2a4c sequence is iters 1-7 HIGH caught → iter 8 APPROVE (clean) → iter 9 MED (softer) → **worker's executable artifacts (JSON, code, numeric fields) are learning faster than worker's prose narratives**. Codex increasingly catches overclaims in interpretive sections, not in measurements.
           - **/loop iter 10 target — `convention_by_convention_ablation_iter_10` (scope stored in `patterns/iter-10-convention-ablation`):** 2×2 convention ablation with rotation held FIXED (either all-FWHT or all-QR, TBD by iter-10 queen spec). Variants: (V1) OUR codebook N(0,1) + L2-norm-extract (baseline, ours), (V2) OUR codebook + NO norm extract, (V3) tonbistudio codebook N(0,1/d) + OUR L2-norm-extract, (V4) tonbistudio codebook + NO norm extract (full tonbistudio). Byte-diff all 6 pairs + reproduce iter-9 D2=3.39 as V1_vs_V4 regression self-check. Thresholds TBD but expected: one of {norm-extract, codebook-scale, both} explains most of the 3.39 convention gap. Estimated 45-60 min.
           - **/loop iter 10 — `cfa-20260423-C4t3i10-convention-ablation` (2026-04-23). ACCEPT_WITH_CORRECTIONS. CODEBOOK-SCALE CONVENTION localized as THE dominant driver (7.1× order ratio over norm-extract).** Read-only iter; deliverables at `/tmp/cfa-iter10-convention-ablation/`: `ablation_oracle/src/main.rs` (~348 LOC), `byte_diff.json`, `byte_diff_report.md`. All 7 R-checks genuinely PASS (Codex-verified, not self-attested): input sha256 matches iter-7/8/9 `ab35172c...`; OUR codebook hashed from runtime read of `turboquant.rs` (sha256 `87562063...`); tonbi codebook computed by **REAL Lloyd-Max iterative solver targeting N(0, 1/256)** at `ablation_oracle/src/main.rs:237-334` (sha256 `10f1494e...`, NOT hardcoded — panic-on-not-converged, midpoint boundaries, composite Simpson's integration); 4 distinct output sha256s; triangle inequality consistent; exactly 1 `fs::write`.
             - **4 pipeline variants (rotation held FIXED at plain-FWHT):** V1 ours-baseline (OUR codebook + L2-norm-extract); V2 ours-no-norm; V3 tonbi-codebook + OUR-norm-extract; V4 full-tonbistudio.
             - **6 pairwise max-abs-diff results:**
               - V1_vs_V2 (norm-extract axis only): **0.479** ← below 0.5 threshold
               - V1_vs_V3 (codebook axis only): **3.425** ← above 2.5 threshold
               - V1_vs_V4 (full regression): **3.476** ← delta 0.081 from iter-9 D2=3.394 (regression self-check PASS)
               - V2_vs_V3: 3.569; V2_vs_V4: 3.435; V3_vs_V4: 0.060
             - **Decision branch: `codebook_dominant`** (V1_vs_V2 < 0.5 AND V1_vs_V3 > 2.5, both conditions cleanly satisfied). Cross-verification: V2_vs_V4 (codebook swap with no-norm) = 3.435 ≈ V1_vs_V3 (codebook swap with norm) = 3.425, difference 0.010 << 0.5 interaction threshold — NO unexpected_interaction. Norm-extract × codebook are near-orthogonal effects. **V1_vs_V3 is 7.1× larger than V1_vs_V2** (order-ratio, per iter-9 MED reminder max-abs distances are NOT additive contributions).
             - **Codex request_changes sev=low (pattern 83ee2a4c now 13/15, LOWEST severity of the /loop — strictly monotone iter 7 HIGH → iter 8 APPROVE → iter 9 MED → iter 10 LOW):** 3 LOW issues — (1) `byte_diff_report.md:65` contribution-language slip "accounts for a minor share" instead of strict order-comparison; (2) `byte_diff_report.md:88` "closes the 3.39 gap" inconsistent with oracle-vs-sourdough caveat; (3) `ablation_oracle/src/main.rs:360` oracle reads `/opt/mlx-native/src/turboquant.rs` + shells `git -C /opt/mlx-native diff --stat HEAD` for R-6 (read-only but not path-confined to workspace). Narrative corrections applied inline to `byte_diff_report.md`; hermeticity issue accepted as documented caveat.
             - **10-iter physics journey DEFINITIVELY localized:**
               - Iter 7: 4-bit TQ cannot byte-match F16 on ANY published impl (goalie research); QJL + Rademacher both missing from our impl
               - Iter 8: Rademacher-is-dominant hypothesis FALSIFIED (2.3% of gap)
               - Iter 9: Conventions-over-rotation — D1_rotation=0.62 vs D2_conventions=3.39 (5.5× ratio)
               - **Iter 10: Codebook-scale SPECIFICALLY — V1_vs_V3=3.425 vs V1_vs_V2=0.479 (7.1× ratio within conventions)**
               - Loci ruled out: byte-exact-sourdough viability, rotation-class, Rademacher pre-mult, QJL-as-byte-identity-fix, norm-extract axis
               - Loci surviving: **codebook-scale N(0,1) vs N(0,1/d) at `/opt/mlx-native/src/turboquant.rs:27-32`**
           - **/loop iter 11 target — `swap_codebook_to_n_over_d_iter_11` — FIRST CODE MODIFICATION TO /opt/mlx-native SINCE ITER 6:** Create branch `cfa/cfa-20260423-C4t3i11-codebook-swap/claude` off mlx-native `fa9dbac`. Replace the 16 f32 literals at `turboquant.rs:27-32` with N(0, 1/256) Lloyd-Max centroids (sha256 `10f1494e...` as computed by iter-10 oracle via `solve_lloyd_max_gaussian(d=256, bits=4)`). Leave norm-extract + sqrt(d) rescale at `:284-387` unchanged. Run `scripts/sourdough_gate.sh` and `hf2q --benchmark` decode tok/s. If sourdough clears (byte-parity or semantic gate — TBD by iter-11 queen) AND decode perf does not regress, the codebook swap addresses the defect class iter-10 localized. If sourdough still fails, iter 12 options: (a) refine Lloyd-Max target distribution (paper Section 3 exact prescription), (b) pivot to semantic gate renegotiation (the Option 4 at the top of this C-4 section), (c) investigate remaining residuals (kernel dispatch, softmax precision, other conventions). **First iter with worktree + actual /opt/mlx-native edit since iter 6.** Estimated 45-75 min.
           - **/loop iter 11 — `cfa-20260423-C4t3i11-codebook-swap` (2026-04-23). REQUEST_CHANGES (branch preserved, no merge).** Worker mechanics clean (Rust const at `turboquant.rs:31` swapped correctly to `10f1494e...` sha256; dual Cargo.toml + `.cargo/config.toml` patch discovered + reverted losslessly; worktree commit `d25ff5b` preserved at `cfa/archive/cfa-20260423-C4t3i11-codebook-swap/claude-archive`). **Measurement was methodology gap, not hypothesis test** — baseline sourdough = 3656 bytes PASS, after = 3656 PASS, Δ = 0. The hypothesis is UNTESTED, not falsified.
             - **Codex HIGH sev (pattern 83ee2a4c back to HIGH at 14/16 — NOT regression, NEW load-bearing facts surfaced):**
               - **HIGH-1: codebook lives in 3 places, not 1.** `turboquant.rs:27-32` (Rust const, edited) + `src/shaders/hadamard_quantize_kv_fast.metal:17` (GPU encode shader, STILL N(0,1)) + `src/shaders/flash_attn_vec_tq.metal:98` (GPU decode shader, STILL N(0,1)). Worktree is now silently inconsistent 1-of-3. Even with TQ active, GPU runtime would use OLD codebook.
               - **HIGH-2: TQ is gate-disabled on default path.** `forward_mlx.rs:1472`: `use_dense_sdpa = self.dense_kvs.is_some()` → decode goes dense. `forward_prefill.rs:11-12`: explicitly F32 dense attention. No `HF2Q_FORCE_TQ` / `HF2Q_USE_DENSE_KV` env var. Source-level gate edit required (like iter-6-archived `77df3d7` mixed-precision branch).
               - **HIGH-3: methodology gap conclusion.** 3656→3656 result did NOT test the N(0,1/d) hypothesis against a TQ-active sourdough path. Iter 11 measured dense path twice.
             - **Important sanity update:** the default-decode sourdough is 3656 bytes PASS on current main. The 69-byte regression only manifests when TQ is actively enabled (via the gate-flipped / archived branches). Iters 7-10 oracle measurements were valid at the oracle-level comparison, but the 69-byte sourdough symptom requires explicit TQ activation to reproduce.
             - **What iter 11 preserved that's valuable for iter 12:**
               - The Rust const change at `d25ff5b` (1/3 of the required edit)
               - The `.cargo/config.toml` path-override discovery (not previously documented — unblocks future codebook experiments)
               - The 3-codebook-site revelation itself
               - The TQ-disabled-by-default gate inventory
             - **USER STANDING DIRECTIVE 2026-04-23 (load-bearing):** "keep looping until it's actually shippable — we're not the 1st team to do turboquant — so we know it IS possible." Saved to `feedback_shippability_standing_directive.md`. **Do NOT treat semantic gate renegotiation (Option 4) as the default fallback.** Other teams ship working TQ; the bar is shippability, not a closed ADR with a relaxed gate.
           - **/loop iter 12 target — `complete_codebook_swap_and_enable_tq_iter_12` (scope stored in `patterns/iter-12-complete-swap-enable-tq`):** Cherry-pick `d25ff5b` onto fresh worktree `cfa/cfa-20260423-C4t3i12-complete-swap-enable-tq/claude`. Edit both Metal shaders (`hadamard_quantize_kv_fast.metal:17` + `flash_attn_vec_tq.metal:98`) with the same N(0, 1/256) values (sha256 `10f1494e...`). Implement TQ-enable mechanism — preferred: add `HF2Q_FORCE_TQ=1` env toggle in `forward_mlx.rs` that overrides `use_dense_sdpa` gate + bypasses `forward_prefill` dense F32 path; fallback: rebase iter-6-archived `77df3d7` mixed-precision branch. Measure: (a) TQ-active sourdough with OLD N(0,1) codebook (this should reproduce the 69-byte regression), (b) TQ-active sourdough with NEW N(0, 1/256) codebook (this is the actual hypothesis test). Record decode tok/s for both. If (b) improves significantly over (a), iter-10 localization is validated. If (b) stays at 69 bytes, iter 13 investigates what AmesianX / tonbistudio / vLLM / DEJAN do differently (calibration-fit centroids, per-layer codebooks, QJL interactions with softmax precision, accumulator dtypes). **Do NOT pivot to semantic gate even if iter 12 fails — investigate what shipping implementations do differently.** Estimated 60-90 min.
           - **/loop iter 12 — `cfa-20260423-C4t3i12-complete-swap-enable-tq` (2026-04-23). REJECT merge; branches archived for iter-13 cherry-pick. N(0,1/256) CODEBOOK HYPOTHESIS FALSIFIED by end-to-end production measurement.** User directive-driven scope change mid-iter: "TQ should be default" and "keep looping until shippable; other teams shipped TQ" saved to `feedback_tq_default_directive.md` + `feedback_shippability_standing_directive.md`.
             - **Atomic test executed cleanly (all 7 R-checks PASS):** mlx-native worktree @ `52d8e91` (3-site consistent codebook swap: Rust const + 2 Metal shaders all at sha256 `10f1494e...`); hf2q branch @ `e0c6461` (TQ-default flip with HF2Q_USE_DENSE=1 opt-out at `forward_mlx.rs:1472` + `forward_prefill.rs`). Cargo.toml reverted losslessly.
             - **3 sourdough measurements (sequential, same environment):**
               - **(a) Baseline (main, dense default, N(0,1) codebook): 3656 bytes PASS** ← sanity
               - **(b) TQ-default flip + OLD N(0,1) codebook: 69 bytes** ← reproduces known regression per brain `8767af4a` (R-3 PASS, flip is active)
               - **(c) TQ-default flip + NEW N(0,1/256) codebook: 7 bytes** ← **10× worse**, degenerate repetition ("a-of-of-of-of-the-of-the-the-...")
             - **Iter-10 hypothesis (codebook-scale swap as fix) FALSIFIED decisively.** End-to-end production worsened from 69 → 7 bytes.
             - **Codex sev=med correction (pattern 83ee2a4c 15/17) — important subtle distinction:** iter-10's oracle at `ablation_oracle/src/main.rs:418` (V1) + `:505` (V3) DID use production-like normalization (FWHT + L2-norm-extract + sqrt(d) rescale). The 3.425 V1_vs_V3 divergence was a real-regime measurement. The ERROR was inferential: **"divergence direction" ≠ "fix direction"**. Our pipeline (N(0,1) codebook + norm-extract) and tonbistudio's (N(0,1/d) codebook + no-norm) are each internally consistent; swapping only the codebook without swapping the norm-extract policy is WORSE than either consistent pipeline. Iter-12 end-to-end confirmed: 69 (matched but regression) → 7 (unmatched, severe regression).
             - **TQ-default flip mechanism VALIDATED (preserved at `cfa/archive/cfa-20260423-C4t3i12-complete-swap-enable-tq/hf2q-archive`):** Measurement (b) with OLD codebook cleanly reproduced the 69-byte regression — proves `forward_mlx.rs:1472` gate flip + `forward_prefill` adjustment work correctly. Structurally reusable for iter 13.
             - **What iters 7-12 have now ELIMINATED as fix loci:** byte-exact F16 viability (iter 7), Rademacher pre-mult absence (iter 8, 2.3%), rotation-class (iter 9, 5.5× oracle), norm-extract axis (iter 10, 7.1× oracle), codebook-scale swap (iter 12 end-to-end). **Five candidate loci falsified with code-generated evidence.**
             - **What's still unexplored:** (1) production K/V empirical distribution after FWHT + norm-extract + sqrt(d) rescale — is it actually ~N(0,1)? (2) per-layer codebooks (shipping impls may use different centroids per attention layer); (3) QJL residual (paper Algorithm 2 Step 3 — iter-7 grep-found missing, never end-to-end tested); (4) kernel dispatch / softmax precision / accumulator dtypes; (5) what AmesianX / tonbistudio / vLLM PR #38280 / ollama PR #15090 / DEJAN Gemma 3 4B do differently — their calibration data, fitting methodology, per-model tuning.
           - **/loop iter 13 target — `empirical_distribution_and_shipping_impl_study_iter_13` (scope stored in `patterns/iter-13-empirical-and-shipping-impl-study`):** Two parallel researcher subtasks. **Subtask A (30-45 min):** instrument `hadamard_quantize_kv_fast.metal` step 5 to dump empirical pre-codebook-lookup K/V values on real Gemma-4-27B with sourdough prompt across {layer 0, 5, 10, 20} × {pos 0, 5, 10, 20}; compute mean/std/skew/kurtosis; if distribution deviates from N(0,1), fit Lloyd-Max codebook to EMPIRICAL samples and re-run sourdough. **Subtask B (45-60 min):** read AmesianX/TurboQuant C++ + tonbistudio/turboquant-pytorch + vLLM PR #38280 + ollama PR #15090 + DEJAN Gemma 3 4B source end-to-end — focus on calibration data source, per-layer vs global centroids, whether any ship pre-fit codebooks for specific models, QJL integration. Per user directive: **DO NOT pivot to semantic gate**. Queen Phase 3 of iter 13 synthesizes A+B and scopes iter 14 (possibly: calibration-fit codebook + per-layer + QJL, all applied together). Estimated 90-120 min total.
      4. **Coherence metric renegotiation** if (1)-(3) can't close the gap: the sourdough 3094-byte byte-exact gate is specifically a Claude 3.5 Sonnet reference; for 4-bit KV, options include (a) ROUGE or perplexity coherence gates, (b) first-N-token BLEU up to a bounded divergence point, (c) accept a documented 4-bit quality delta with explicit user opt-in via `--kv-bits`.

    - After C-4 produces a direction: update C-5/C-6 accordingly. The original `delete dense_kvs` plan (previously the post-C-3-fix step) is REVOKED — dense_kvs is a mixed-precision code path now, not a fallback. `dispatch_hadamard_quantize_kv_seq` wiring remains useful for batched prefill regardless.

    - **Optional H3 closure on the batched path** (not blocking C-4): re-dump with `HF2Q_BATCHED_PREFILL=1 HF2Q_UNSAFE_EXPERIMENTS=1` on the codex branch (commit `415c9d6`) and run the C-0b diff.

### C-2 outcome (2026-04-22)

**VERDICT LABEL RETRACTED 2026-04-22 late.** The 4×2 matrix values below are numerically correct but the verdict label `dequant_spec_bug_confirmed` is physically incorrect. C-3's round-trip identity triad (see `### C-3 outcome (2026-04-22 late)` below) showed that the TQ spec's per-vector round-trip is at the analytic Lloyd-Max 4-bit N(0,1) floor (RMSE=0.09747). The 0.32–0.55 SDPA-level divergence C-2 measured is the *expected* attention-amplification of that representation floor under hf2q's Gemma contract (`scale=1.0` + RMS-normalized Q/K), as independently verified by Codex's synthetic-SDPA reproduction in Phase 2b of the C-3 session. The prescription "bisect CODEBOOK / FWHT / sqrt(hd)" at the bottom of this subsection is SUPERSEDED by C-3's strategic-pivot framing. The matrix and methodology sections below are preserved as the historical record of the measurement.

---

CFA session `cfa-20260422-C2-multistep`. Claude branch commits: hf2q `a6ca566` (P1a), mlx `6035cca` (P3a), `460f66e` (P1b), `7895164` (P2), `eda59ef` (P3b), hf2q `a0a4a87` (P4). Codex team parallel.

**Singlestep regression gate:** dequant_oracle_nrmse = 5.138e-5 (target 5.1e-5 ±1e-6). PASS.

**4-position × 2-oracle result matrix** (seed 0xC25EED, mask_type=2 sliding_window=1024 softcap=0, kv_capacity=1024, num_heads=8/4 head_dim=256):

| pos | kvl_logical | ring_start | dequant_oracle_nrmse | independent_floor_nrmse | verdict |
|-----|-------------|------------|---------------------|------------------------|--------|
| 50 | 51 | 0 | 6.85e-4 | 4.55e-1 | dequant_spec_bug_confirmed |
| 500 | 501 | 0 | 1.12e-3 | 5.50e-1 | dequant_spec_bug_confirmed |
| 1050 | 1024 | 27 | 8.92e-4 | 4.91e-1 | dequant_spec_bug_confirmed |
| 2048 | 1024 | 1 | 5.71e-4 | 3.23e-1 | dequant_spec_bug_confirmed |

**C-2 verdict: `dequant_spec_bug_confirmed`**

Pattern across all 4 positions: dequant_oracle_nrmse is low (5e-4 to 1e-3) but independent_floor_nrmse is very high (0.32–0.55). The TQ kernel output closely matches the CPU oracle built from `nibble_dequantize` at every position including ring-wrap (pos 1050, ring_start=27) and deep ring (pos 2048, ring_start=1). But the GPU-TQ output diverges dramatically from the dense `flash_attn_vec` oracle operating on the same pre-quant F32 K/V.

This is the `dequant_spec_bug_confirmed` branch: the kernel implements `nibble_dequantize` faithfully, but `nibble_dequantize` itself (FWHT+codebook lookup) is quantizing to a representation whose SDPA output doesn't match what the pre-quant F32 input would have produced. The locus is in the `nibble_dequantize` formula, the CODEBOOK_4BIT values, or the FWHT normalization convention used during encode — one or more of these causes the round-trip (encode → decode → SDPA) to deviate from the identity (pre-quant → SDPA) by 0.32–0.55 NRMSE at practical context lengths.

This eliminates H1 (kernel math), H2 (FWHT pipeline), and H4 (dispatch/ring_start bug) as primary causes: the kernel is self-consistent, ring-wrap handling is correct (ring_start=27 and ring_start=1 show same pattern as ring_start=0), and the pattern holds at all positions. The defect is in the quantization spec itself.

**Canary-symmetry findings:** C-1-unlock's asymmetric canary (one-sided mutation: GPU sees 2x norm, CPU oracle does not) produced 0.111 nrmse. Symmetric fix (P2): mutating k_norms_compact at pos=10 BEFORE rebuilding k_dequant brings nrmse to 7.88e-5 (≤ 1e-4). Asymmetric mode (`HF2Q_REPLAY_CANARY_ASYMMETRIC=1`) reproduces 0.111. Both paths verified.

**Anomalies:** The `dequant_spec_bug_confirmed` nrmse values grow slightly with kvl_logical (pos 50: 6.85e-4 vs pos 500: 1.12e-3), suggesting cumulative quantization noise compounds as cache length grows, but the primary signal is the large independent-floor divergence present at even the smallest tested position (pos=50, kvl=51).

**Next step (C-3):** Bisect `nibble_dequantize` / `CODEBOOK_4BIT` / FWHT normalization. Specifically: (a) compare `CODEBOOK_4BIT` values against the paper's Lloyd-Max Gaussian table for 4-bit; (b) verify FWHT normalization convention (normalized H·H=I vs unnormalized H·H=d·I) is consistent between encode and decode; (c) check whether the `inv_scale = 1/sqrt(hd)` in `nibble_dequantize` matches the encode-side `scale = sqrt(hd)` in `nibble_quantize`.

**Evidence files:**
- `/tmp/cfa-20260422-C2-multistep/claude-multi.md` — Markdown table
- `/tmp/cfa-20260422-C2-multistep/claude-multi.json` — raw JSON
- `/tmp/cfa-20260422-C2-multistep/claude-single.json` — singlestep regression gate

### C-3 outcome (2026-04-22 late)

CFA session `cfa-20260422-C3-roundtrip`, review-only mode (1 Claude impl + Codex read-only reviewer + opus queen). Merge SHAs: mlx-native `0fbf1e6`, hf2q pending.

**Test:** `mlx-native/tests/round_trip_identity.rs` (pure Rust, CPU-only, deterministic seed `0xC25EED`, Gaussian N(0,1), 1000 vectors per cell). Triad of three cases × three head_dims = 9 cells:

| case | head_dim | nrmse_mean | ratio_to_case_b |
|------|----------|------------|-----------------|
| A (full pipeline FWHT+quant+invFWHT) | 128 | 0.09603 | 0.9928 |
| B (quant-only)                        | 128 | 0.09672 | 1.0000 |
| C (FWHT-only)                         | 128 | 1.0e-7  | —       |
| A | 256 | 0.09674 | 0.9921 |
| B | 256 | 0.09751 | 1.0000 |
| C | 256 | 1.0e-7  | —       |
| A | 512 | 0.09699 | 0.9967 |
| B | 512 | 0.09731 | 1.0000 |
| C | 512 | 1.0e-7  | —       |

**C-3 verdict: `representation_floor_confirmed`**

Three observations drive the verdict, each ruling out a hypothesis:

1. **Case C (FWHT-only) ≈ 1e-7** at all head_dims → FWHT is self-inverse to machine epsilon. Rules out **FWHT_non_reversible**.
2. **Case B (quant-only) ≈ 0.097** at all head_dims → 4-bit Lloyd-Max quantizer on N(0,1) input hits the analytic floor. Rules out **CODEBOOK_bug** (codebook numerics are correct).
3. **Case A / Case B ratio ∈ [0.992, 0.997]** → FWHT introduces no extra error when combined with quantization (actually marginally *reduces* it via incoherence-spreading, consistent with the TurboQuant paper's design intent). Rules out **FWHT_normalization_bug**.

**First-principles floor derivation** (independently reproduced by both CFA queen (Phase 3) and Codex review (Phase 2b)): using the production `CODEBOOK_4BIT` values at `mlx-native/src/turboquant.rs:27`, the analytic MSE of optimal 16-level scalar quantization on N(0,1) input is 0.009501008, so RMSE = **0.09747**. The measured 0.097 matches this to 4 decimals.

**Physics check on C-2's SDPA divergence** (Codex Phase 2b): a synthetic SDPA using the C-2 shapes + hf2q's Gemma contract (`scale=1.0` + RMS-normalized Q/K, cited at `forward_mlx.rs:1611,1658`) feeding through the same 4-bit TQ representation reproduced output nrmse **0.27 at 51 tokens and 0.47-0.55 at 500-1024 tokens** — matching C-2's matrix (0.32-0.55). So the C-2 measurement is explainable entirely by the representation floor + attention amplification, with no spec defect required.

**Strategic implication:** ADR-007 pivots from "fix the TQ spec bug" to "accept the representation floor as physics and compensate strategically." `dense_kvs` stops being a fallback-to-remove and becomes a principled mixed-precision path. The planned C-3 bisect Targets 2 and 3 (CODEBOOK / FWHT normalization) are pre-falsified; C-4 is now the current gating step, evaluating higher-bit codebook + mixed-precision + coherence-metric options. See the C-4 strategic pivot block above in the Path-to-Completion.

**Paper reference:** arxiv.org/abs/2504.19874 (TurboQuant), Lloyd-Max 16-level Gaussian table cited in Codex review.

**Codex review findings** (3 issues, all addressed):
- MED #1 (line 273 verdict gate too loose) — fixed pre-merge as commit `34c4874`, verdict gate tightened to floor band [0.085, 0.11] × ratio [0.90, 1.10].
- MED #2 (line 520 `assert!(true)`) — fixed pre-merge as part of `34c4874`, replaced with four regression gates that panic with specific diagnostic messages mapping each failure mode to a decision-tree branch.
- LOW #3 (RNG reseeds per head_dim, narrative mismatch with comment) — deferred to C-4.

**Evidence files:**
- `/tmp/cfa-20260422-C3-roundtrip/result.md` — Markdown table
- `/tmp/cfa-20260422-C3-roundtrip/result.json` — raw JSON with verdict + rationale
- `/tmp/cfa-20260422-C3-roundtrip/codex-review.json` — Codex review payload (includes synthetic-SDPA reproduction numbers)

  - **C-0b outcome E2 (H3 prefill-encode bug confirmed):** fix encode — bisect `hadamard_quantize_kv` single-token vs `dispatch_hadamard_quantize_kv_seq` batched paths, align them, land a regression test at the encode boundary. Then same stub-removal as E1.

  - **C-0b outcome E3 (compound defect):** fix the larger-magnitude contributor first, re-run C-0 paired dumps; residual delta is either a second locus (branch again through E1/E2) or a surviving-representation-floor (then and only then: introduce `--kv-mode {f16, tq4}` with `f16` default disabling TQ encode entirely to avoid the 0.14 ms/token waste, `tq4` opt-in for long-context memory savings, byte-exact sourdough in `f16` mode and a TQ-specific quality metric in `tq4`).

No third option. No new "Track 4 safe fallback." Track 3 ships no further until one of E1/E2/E3 fires.

**C-5. Phase 2 — 262K unlock.**
(Mantra: "No short cuts. Done the right way.")
Only after C-4. Remove the 8192 cap. Needle-in-haystack. Benchmark.

**C-6. Phase 3 — bit-width configurability.**
(Mantra: "Just pure excellence.")
Land `--kv-bits {2, 2.5, 3, 4}`, per-layer adaptive, true mixed-width packing. Only after C-5.

### Governing constraints (all steps)

- **Plenty of time.** No schedule pressure justifies skipping Phase 0 again.
- **Never assume.** The "4-bit representation floor" hypothesis remains UNVERIFIED; the "ring off-by-one is the only bug" hypothesis remains UNVERIFIED. C-0 ends both.
- **Chesterton's fence.** The `use_dense_sdpa` gate was put up for a reason (69-byte regression). That reason must be localized before the fence comes down.

---

## Risks & Mitigations

| # | Risk | Severity | Likelihood | Mitigation |
|---|------|----------|------------|------------|
| R-1 | Gather throughput on M5 Max is poor (< 50% of sequential) | High | Medium | Phase 0.2 microbench kills this early. Fallback: dequant-to-temp-buffer (adds 2 dispatches per layer but preserves bandwidth savings). Sub-vector VQ is an independent optimization, not a fallback for this risk. |
| R-2 | Hadamard rotation produces higher MSE than random orthogonal rotation | High | Low | Phase 0.1 measures this directly: Hadamard MSE vs random rotation MSE at all head_dims. Gate: ≤1.2× ratio. Hadamard is orthogonal (energy-preserving); QuaRot/QuIP# validate it empirically for LLM quantization. |
| R-3 | Modified SDPA kernel is slower than F16 SDPA despite reading less data | Medium | Medium | Phase 1.3 measures wall-clock, not just throughput. If slower, profile: is it ALU (centroid gather compute) or memory (random access pattern)? |
| R-4 | Nibble packing wastes memory at 262K (1.37 GB vs 0.89 GB at true mixed-width) | Medium | Low | Phase 2.2 memory gate triggers true mixed-width packing. On 192 GB M5 Max, 1.37 GB is <1% of memory — likely acceptable |
| R-5 | Quality regression on specific model architectures (Qwen-3, etc.) | Medium | Low | Phase 2 validates on Gemma-4 (primary). Qwen-3 support is a future extension validated separately |
| R-6 | Fixed channel split is suboptimal vs per-position outlier selection | Low | Low | Phase 0.1 measures per-coordinate magnitude variance after rotation. QuaRot shows rotation eliminates outliers (kurtosis ≈ 3), making channel selection arbitrary. If variance is unexpectedly high, revisit. |
| R-7 | Gaussian-optimal codebook is suboptimal for small head_dim (d=128) | Low | Low | Phase 0.1 compares Gaussian vs Beta-optimal codebooks at d=128, 256, 512. Gate: ≤5% MSE gap. |

---

## Information-Theoretic Foundation

From TurboQuant (Theorem 1, Theorem 3):

**MSE Distortion Bounds:**

| Bit-width | Upper bound (TurboQuant) | Lower bound (any algorithm) | Gap |
|-----------|------------------------|-----------------------------|-----|
| 2 | 0.117 | 0.0625 | 1.87× |
| 3 | 0.030 | 0.0156 | 1.92× |
| 4 | 0.009 | 0.0039 | 2.31× |

TurboQuant is within 2.7× of the information-theoretic optimum at all bit-widths. No algorithm — learned, calibrated, or otherwise — can beat the lower bound. This means TurboQuant's quality at 2.5 bits is close to the best any quantization scheme could achieve at 2.5 bits.

**Why this matters for the ADR:** We are not betting on an empirical hack that might fail on other models. TurboQuant's guarantees are mathematical, distribution-free, and worst-case. If it works on Gemma-4-27B at 2.5 bits (and the paper shows it does on Llama-3.1-8B), it will work on any model with similar or larger head dimensions.

---

## Research Provenance

This ADR is grounded in a 5-agent research swarm (2026-04-14) analyzing arXiv:2504.19874:

| Agent | Finding | Impact on ADR |
|-------|---------|---------------|
| cost-analyst | Rotation is per-head (d=256), not per-hidden-state (d=3584). Dense rotation: ~290 μs. Hadamard: ~20 μs. SDPA-side dense rotation: 7.6 ms (infeasible). | → Hadamard mandatory. Encode-side feasible. |
| quant-comparator | INT4 ties at 4 bits. TurboQuant wins at ≤3 bits. Outlier spreading is genuine advantage. | → TurboQuant justified at 2.5-bit target. |
| metal-feasibility | Option C (pre-rotated centroids) eliminates rotation from SDPA path. Dispatch count unchanged. Gather throughput is the key risk. | → Architecture design. Phase 0.2 risk gate. |
| qjl-necessity | QJL unnecessary at ≥2.5 bits (bias β > 0.97, KL ≈ 0.001). Paper's own KV cache results used MSE-only. | → TurboQuant_mse, no QJL. Simpler implementation. |
| mem-calculator | 262K F16 = 5.3 GB. TQ nibble = 1.4 GB. TQ 2.5b = 0.88 GB. Bandwidth savings: 9.7 ms → 2.6 ms at 262K. | → Memory and performance gates. |

---

## Deviations from the Paper

This ADR implements the core TurboQuant_mse algorithm faithfully but makes two engineering substitutions, both validated empirically in Phase 0:

| Paper specifies | ADR uses | Reason | Validation |
|-----------------|----------|--------|------------|
| Dense random orthogonal matrix (QR of Gaussian) | Walsh-Hadamard Transform | O(d log d) vs O(d²); critical for SDPA decode latency. Prior art (QuaRot, QuIP#) validates Hadamard for outlier spreading. | Phase 0.1: MSE comparison, gate ≤1.2× |
| Per-instance outlier channel selection for non-integer bit-widths | Fixed compile-time channel split (first d/4 at higher bit-width) | After rotation, all coordinates ≈ N(0,1/d) with equal magnitude — outlier identity is arbitrary. Eliminates 167 MB per-position storage at 262K. | Phase 0.1: magnitude variance measurement |

Everything else — Lloyd-Max codebooks, per-vector norm storage, TurboQuant_mse (not _prod), coordinate-wise scalar quantization — follows the paper directly.

---

## What This ADR Does NOT Cover

- **Prompt/prefill-phase quantization** — TurboQuant applies to decode-phase KV cache. Prefill processes tokens in bulk and may benefit from different optimizations.
- **Weight quantization** — model weights remain Q4_K_M via GGUF. This ADR is exclusively about KV cache.
- **Multi-model support** — Phase 0-2 target Gemma-4-27B only. Qwen-3 and other architectures are future work after the Gemma path is validated.
- **Streaming/incremental quantization during prefill** — the paper notes TurboQuant is online/data-oblivious and can quantize during streaming generation. This is a natural extension but not in scope for the initial implementation.
- **TurboQuant_prod / QJL residual correction** — not needed at ≥2.5 bits per the paper's own KV cache experiments.
