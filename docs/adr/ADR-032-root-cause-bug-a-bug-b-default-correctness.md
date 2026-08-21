# ADR-032: Root-Cause Bug A and Bug B; Ship Best-Outcome Defaults for Gemma & Qwen

- **Status**: proposed
- **Date**: 2026-05-17
- **Deciders**: operator (robert@loveathome.us); claude (impl)
- **Tags**: coherence, flash-attention, softmax, default-correctness, gemma-4, qwen-3, TQ-KV, root-cause

## Context

Two coherence bugs were surfaced during /loop investigation on
`models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf`:

- **Bug A** — FA-D=512 BF16-Q path produces a `Format: Hemoglobin` greedy
  loop at decode-pos ~70 on 300-item enumeration prompts. llama.cpp and
  hf2q's sequential prefill (per-token, ground truth) stay coherent through
  95+ items.
- **Bug B** — `scale_mask_softmax_f32` cross-simdgroup reduction at
  `tg_size=33` produces token 237372 (`서`) instead of `<|channel>` (100)
  as the first decoded token on the chemistry+backticks `--enable-thinking`
  probe.

What was shipped this session:

| Bug | Currently in tree | Status of root-cause work |
|---|---|---|
| A | F16-Q dispatcher behind `HF2Q_FA_F16=1` opt-in (default OFF); NO_FA tensor-mm routing around FA at D=512 default ON | **Mitigation + workaround.** Neither is a peer-validated kernel fix. |
| B | sg0-only reduction + broadcast in `scale_mask_softmax` V1/V4 max/sum (mlx-native `5461c81`) | **Likely correct, not yet audited at the dispatcher layer.** Who picks `tg_size=33`? Is the dispatcher picking pathological sizes that affect *every* simd-reduce kernel? |

Operator directive (2026-05-17): "we need to be as coherent as llama.cpp
and as fast as (or faster than) llama.cpp, with TQ enabled, for gemma and
qwen families … we need our defaults to use the best settings to enable
the best outcome for the operator … I wanted us to identify the root
cause of the bug and fix the bug" (not side-step it).

Per mantra: "DO NOT BE LAZY. Always dive deep and ensure you know the
problem you're solving. No fallback. Just pure excellence."

The current code-tree state violates the mantra: Bug A's root mechanism
has not been compared against the peer reference (`flash_attn_prefill_llamacpp_bf16_d512`
is *named* after llama.cpp but has never been algorithmically diffed
against it); the env-flag mitigation was chosen because more-mantissa-bits
makes the symptom go away, not because we proved that BF16-vs-F16 is the
actual peer divergence. Bug B is plausibly fixed but the *dispatcher* that
chose `tg_size=33` has not been audited.

This ADR records the plan to find the true root cause for both, and ships
**defaults that match operator's best-outcome criteria** (peer coherence,
peer-or-better speed, TQ enabled) — not env-gated escape hatches.

## Decision

Execute a four-phase investigation-then-fix plan. **No code changes ship
until phase 1-4 investigations either confirm or supersede the current
mitigation.**

### Success criteria for ship

Ship gate (all must hold under DEFAULT — zero env vars set):

1. **Gemma 4 26B-A4B APEX-Q5_K_M coherence**
   - 300-Format `--no-thinking` greedy: coherent through 95+ distinct items (matches llama.cpp diversity character-class)
   - chemistry+backticks `--enable-thinking`: first decode token = 100 (`<|channel>`), coherent through ≥30 decoded tokens
   - Both batched-prefill paths AND sequential prefill produce coherent output
2. **Qwen 3.6 APEX-Q5_K_M coherence + speed**
   - sourdough byte-parity gate: 242-byte common prefix byte-identical to llama.cpp
   - tg200 ≥ 1.05× llama.cpp
   - tg1500 ≥ 1.05× llama.cpp
   - TQ-V active by default (load-banner `tq_kv = active`)
3. **Gemma 4 ara speed**
   - tg200 ≥ llama.cpp `-fa 1` (parity or AHEAD)
   - tg2000 ≥ llama.cpp `-fa 1` (parity or AHEAD)
4. **Regression-free**
   - coherence_smoke: 2/2 PASS
   - No new flaky tests
   - No CLAUDE.md / mantra violations (no fallback, no env-gating-around-bug)

### Phase 1 — Peer kernel comparison (low cost, high info)

**Goal**: determine whether `flash_attn_prefill_llamacpp_bf16_d512` actually
matches llama.cpp's flash-attn-ext D=512 algorithm, or whether we deviated.

Actions:
- Read `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` (kernel) and
  `ggml-metal-ops.cpp` (dispatcher) for flash_attn_ext at hd=128 / hd=512.
- Read `/opt/candle/candle-metal-kernels/src/metal_src/` flash-attn variants if present.
- Diff against `/opt/mlx-native/src/shaders/flash_attn_prefill_d512.metal`:
  - Q/K/V shmem precision
  - Tile sizes (BQ, BK)
  - Scale partitioning (pre-MMA vs post-MMA)
  - Online softmax normalization (max-tracking, log-sum-exp invariants)
  - Mask handling (broadcast vs per-element)
  - KV chunk reload precision

Effect: signed determination of one of:
- (A1) Our kernel matches peer exactly — BF16-Q is peer-validated;
  Bug A is somewhere else (likely outside FA entirely).
- (A2) Our kernel deviates in a specific way that causes the precision
  drift — fix the deviation, not the symptom.
- (A3) Peer also uses F16-Q in shmem — current F16 mitigation is correct
  algorithmic alignment; default-flip with confidence.

### Phase 2 — Per-layer dump bisection for Bug A (medium cost)

**Goal**: localize Bug A to a specific dispatch, not just "FA-D=512".

Actions:
- Configure `HF2Q_NO_FA=0 HF2Q_FA_F16=0` (forces broken FA-BF16 path)
- Use `HF2Q_BATCHED_LAYER_SCAN=85` (or wherever the Hemoglobin loop starts)
  to dump per-layer residual stream entry at the first-divergent decode
  position
- Same dump from llama.cpp at the same decode position (instrument
  `examples/main` or use existing dump infra)
- Compare residual streams layer-by-layer; find first layer with
  divergence > BF16 noise floor (~2e-3 relative)
- At that layer, narrow to dispatch (head_norm? RoPE? Q@K^T? scale? softmax? P@V? o_proj?)

Effect: precise per-dispatch localization. Reveals whether the bug is
FA-internal or elsewhere (e.g., head_norm or RoPE upstream).

### Phase 3 — Bug B dispatcher re-audit (low cost)

**Goal**: confirm kernel-layer fix is sufficient; check for latent
dispatcher bug affecting all simd-reduce kernels.

Actions:
- `grep -rn "threadgroup_size\|threads_per_threadgroup\|MTLSize\|dispatchThreadgroups" mlx-native/src/ops/scale_mask_softmax*`
- Identify caller and dispatch-size selection logic
- Determine: does the dispatcher pick `tg_size=33` because of a row-count
  fencepost (e.g., `seq_len + 1` for causal mask?)? Or is `33` the natural
  shape from upstream tile sizing?
- If picker has bug (e.g., doesn't round up to simdgroup-aligned): patch
  at picker layer too. Kernel fix becomes defense-in-depth.
- Audit *all* kernels using same `shared[tiisg] + simd_max/simd_sum`
  pattern for the same latent bug (mantra: "Always understand current
  fully before changing it" — Chesterton's fence).

Effect: confirmed-or-extended Bug B fix. Defense in depth.

### Phase 4 — Scale-partition prior-art audit (low cost)

**Goal**: check if the existing scale-post-matmul technique
(comment `flash_attn_prefill_d512.metal:458-468`) can absorb Q precision
without F16 promotion.

Actions:
- Re-read the historical comment block
- Determine if Q can stay BF16 in shmem if we apply scale in F32 *after*
  the tensor MMA accumulator dump
- Compare to peer (phase 1 finding)

Effect: alternative root-fix candidate ranked against F16-Q.

### Phase 5 — Design

Pick the root-fix layer based on phase 1-4. Priority order:
- (a) Peer-algorithm-divergence fix (phase 1, outcome A2)
- (b) Specific-non-FA-dispatch fix (phase 2, if bug isn't FA-internal)
- (c) Scale partitioning (phase 4)
- (d) F16-Q in shmem (current mitigation; selected only if a/b/c yield no deeper cause)

### Phase 6 — Implementation

- Apply chosen fix at the root layer
- Default-flip `HF2Q_NO_FA` back to **false** (no more side-step workaround)
- Default-flip `HF2Q_FA_F16` to **true** if option (d) is chosen, or
  remove the env-flag entirely if (a)/(b)/(c) supersedes it
- Retire outdated comments framing fixes as "work around"
- Keep `HF2Q_NO_FA=0` and `HF2Q_FA_F16=0` as escape hatches for debugging only

### Phase 7 — Verification

Run the full ship-gate matrix from Success Criteria above:
- Coherence: 300-Format, chemistry+backticks, sourdough byte-parity, coherence_smoke
- Speed: Gemma tg200 + tg2000, Qwen tg200 + tg1500, all vs llama.cpp
- TQ-V verification: load-banner check, KV memory savings

### Phase 8 — Ship

- Atomic commits to mlx-native (kernel changes) and hf2q (wiring + default-flips)
- `docs/operator-env-vars.md`: HF2Q_FA_F16 and HF2Q_NO_FA reclassified as
  debug/diagnostic flags; remove all "required for coherence" language
- Update auto-memory with **actual root-cause** documentation
  (superseding the current "F16 better than BF16" framing)
- ADR-032 status: `proposed` → `accepted` with final root-cause findings

## Consequences

### Positive

- **Bug A actually root-caused**, not symptomically mitigated. The "BF16 has
  fewer mantissa bits" explanation is replaced with a peer-validated
  algorithmic explanation.
- **Defaults reflect best-outcome configuration** for the operator. No env
  vars needed to get peer-coherent peer-fast TQ-enabled inference.
- **Mantra compliance**: no fallback, no side-step. The investigation
  *earns* the fix decision.
- **Defense in depth on Bug B**: dispatcher-layer fix (if found) prevents
  any future `tg_size % 32 != 0` kernel from hitting the same trap.
- **Cross-family validation**: success criteria explicitly cover both
  Gemma (D=256/D=512 mixed) and Qwen (TQ-KV with D=128) families, so
  fixes can't silently regress one while improving the other.

### Negative

- Investigation cost ≈ 2-4 /loop iterations before any code lands.
- If phase 1 outcome A1 (peer matches our kernel exactly), then the
  "deep root cause" hunt may dead-end at phase 2's per-layer dump.
  Mitigation: phase 2 will *find* a divergence somewhere (the bug exists);
  it just may not be in FA. That's still a more correct fix than what's
  currently shipped.
- Default-flipping `HF2Q_NO_FA` to false may regain ~118 MB at seq=1359
  on long-context users that the workaround currently saves. Mitigation:
  if F16-Q is the chosen fix, FA is the path everywhere by default and
  long-context users don't lose anything (FA scales O(seq) not O(seq²)).

### Neutral

- Phase 3 (Bug B dispatcher audit) outcome could go either way — if
  dispatcher is fine and `tg_size=33` is the natural shape from upstream
  tile sizing, kernel-only fix stands as-is.
- Perf cost of F16 promotion (if it's the chosen fix) is currently
  measured at ~2.4% tg2000 *when stacked on NO_FA*; the *correct*
  apples-to-apples comparison (F16-Q-FA vs llama.cpp `-fa 1`) is part of
  phase 7 and may differ.

## Links

- ADR-022: kernel-coverage parity with llama.cpp (peer-port methodology)
- ADR-028: peer parity coherence and speed (the prior cross-family goal)
- ADR-029: gemma4 MoE pipeline is the gap (sibling perf investigation)
- ADR-027: Qwen 3.5 TQ-KV cache (TQ-V correctness foundation)
- Memory: `[[bug-a-fa-d512-bf16-precision-2026-05-17]]` (Bug A characterization, now superseded)
- Memory: `[[bug-a-fa-d512-FIXED-2026-05-17]]` (current mitigation entry; will be replaced by root-cause entry on ADR ship)
- Memory: `[[bug-softmax-partial-simdgroup-FIXED-2026-05-17]]` (Bug B current kernel-layer fix)
- Memory: `[[bug-gemma4-batched-prefill-dual-open-2026-05-17]]` (original dual-bug observation)
- Mantra: `~/Documents/mantra.txt` (operating principle)
- Operator directive 2026-05-17: "as coherent as llama.cpp and as fast as (or faster than) llama.cpp, with TQ enabled, for gemma and qwen families"
- Operator directive 2026-05-17: "defaults to use the best settings to enable the best outcome for the operator"
