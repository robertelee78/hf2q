# ADR-007: TurboQuant KV Cache Compression for 262K Context

**Status:** Partially Implemented — End-to-End Correctness UNVERIFIED; dormant on default path since 2026-04-16 (ADR-009 Track 3 fallback). Phase 0.4 C-0 divergence audit **COMPLETED 2026-04-21**. C-0b localization **COMPLETED 2026-04-21** — verdict E1-partial. C-1 kernel replay **ATTEMPTED 2026-04-22 → VERIFICATION_BLOCKED** (2 harness defects). C-1-unlock **COMPLETED 2026-04-22** (dual mode) — **single-step clear, multi-step open**: with barriers + 23-row capture fixed, A dropped 1.2445→5.1e-5 (24,000×), two independent harness implementations (Claude + Codex) produced byte-identical output on all 4 variations. BUT the in-harness CPU oracle uses the same `nibble_dequantize` as the kernel, so a spec-level dequant bug would be invisible; cumulative drift, ring-wrap, and nonzero `ring_start` also untested. **Multi-step audit is the new gating step** (pos 50/500/1050 ring-wrap replay + independent-floor oracle from pre-quant K/V).
**Date:** 2026-04-14 (original); revised 2026-04-21 (honest current-state rewrite); 2026-04-21 (C-0 audit completion + Codex-reviewed revision); 2026-04-21 (C-0b localization + Codex-reviewed narrowing); 2026-04-22 (C-1 VERIFICATION_BLOCKED + 2 harness defects identified); 2026-04-22 (C-1-unlock dual-mode single-step clear + scope caveats)
**Decision Makers:** Robert, Claude
**Related ADRs:** ADR-006 (mlx-native GPU backend — KV cache path lives here), ADR-005 (inference server — speed gates), ADR-008 (candle-divorce port — introduced ring-chronology regression), ADR-009 (Track 3 dense-SDPA "safe fallback" — the stub this ADR's mantra forbids)
**Reference:** Zandieh, Daliri, Hadian, Mirrokni — "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (arXiv:2504.19874, April 2025)

---

## Current Status (2026-04-22 — C-2 complete, spec-level defect confirmed)

**One sentence:** The kernels exist, pass isolated replay tests, AND (as of C-2, 2026-04-22) the causal locus of the sourdough regression is pinned: the independent-floor oracle landed in `examples/tq_kernel_replay.rs` shows a 3-order-of-magnitude nrmse gap between the GPU-TQ output and a dense `flash_attn_vec` reference on pre-quant F32 K/V at every tested position (50, 500, 1050, 2048) including ring-wrap — ruling out {kernel math H1, FWHT pipeline H2, dispatch/ring_start H4} and placing the defect inside the TQ specification itself (CODEBOOK_4BIT Lloyd-Max values, FWHT normalization convention, or the encode/decode sqrt(hd) pairing). C-3 will bisect.

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

    - **C-3 bisect (current gating step, 2026-04-22 late).** Three targets, ordered by cost:
      1. **Round-trip identity test (cheapest).** Encode a known F32 K/V → decode immediately via `nibble_dequantize` → compare to original. Expected near-zero. If NOT near-zero, the bug is in the codebook values or the FWHT normalization. If it IS near-zero but full pipeline (encode → decode → SDPA) still diverges 0.32–0.55 nrmse vs ground truth, the bug is in how the representation interacts with attention (not in the K/V vector round-trip).
      2. **`CODEBOOK_4BIT` values vs the TurboQuant paper's Lloyd-Max Gaussian table.** Inline Lloyd-Max centroids at `mlx-native/src/ops/turboquant.rs:27-32` and the Metal mirror in `flash_attn_vec_tq.metal:98-103`. Cross-reference against the paper's 4-bit Gaussian centroids. Any numerical drift from the paper's values is the locus.
      3. **FWHT normalization convention.** Verify (a) encode `scale = sqrt(hd)` at `turboquant.rs:94-97` and decode `inv_scale = 1/sqrt(hd)` in the kernel are literal reciprocals (not, say, reciprocal up to a `1/d` factor); (b) both sides use the same H·H=I-vs-H·H=d·I orthogonality convention; (c) the `rsqrt(head_dim)` factor in `flash_attn_vec_tq.metal` matches what `hadamard_quantize_kv.metal` wrote.
      C-2 MED-severity followups (not blocking C-3): re-run multistep at real Gemma-4 sliding-layer shape (16/8/256 vs the 8/4/256 Claude used), swap the custom SplitMix64 for `rand::rngs::StdRng::seed_from_u64(0xC25EED)` to restore cross-team bit-identical comparability, fix or explicitly relabel the singlestep `--oracle independent-floor` partial-independence at `tq_kernel_replay.rs:1014` (history rows backfilled from dequant, only newest row from pre-quant), port Codex's `ManifestShaGate` hard-exit to the Claude harness.

    - **Optional H3 closure on the batched path** (not blocking stub removal since default exercise is non-batched): re-dump with `HF2Q_BATCHED_PREFILL=1 HF2Q_UNSAFE_EXPERIMENTS=1` on the codex branch (commit `415c9d6`) and run the C-0b diff.
    - After C-3 fix lands and sourdough clears: delete `dense_kvs` / `dense_sdpa_tmp` fields, the `use_dense_sdpa` gate, the dense SDPA decode branch, and the `self.dense_kvs = Some(...)` assignments in both prefills. Keep `dispatch_hadamard_quantize_kv_seq` wiring in batched prefill. Sourdough passes byte-exact at 3094+.

### C-2 outcome (2026-04-22)

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
