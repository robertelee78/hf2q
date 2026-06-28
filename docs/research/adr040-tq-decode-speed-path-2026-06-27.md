# ADR-040 §22 — Making TQ-HB-V a decode SPEEDUP: 3-stream research synthesis + ranked path (2026-06-27)

Three independent web-research agents (QuaRot/SpinQuant fold; 4-bit-KV decode crossover;
in-kernel Hadamard + Apple-Silicon Metal) + our own QuaRot-fold derivation. All citations
in the agent outputs under `tasks/a2e18351a8f189597`, `a3166278ef0c1764f`, `a656242e3b1ec1d03`.

## The reframe (this is the load-bearing finding)

**Low-bit KV is a MEMORY/BATCH feature, not a single-stream short-context decode-speed feature.**
The premise "TQ should make decode faster" is **false at our benchmark regime**:

- KIVI's headline 2.35–3.47× is **batch-size** (4× smaller cache → 4× more concurrent
  sequences), NOT faster per-token decode of one stream. (arXiv 2402.02750)
- The V-cache is a **minority of bytes-per-token** at short/medium context — model weights
  dominate. KVQuant's own fused 4-bit kernels net only **1.1–1.4× on the K/V mat-vec op
  itself**, diluted to ~nothing across the full step. (arXiv 2401.18079)
- **Crossover ≈ 7k tokens** (vLLM FP8 data): below ~7k, quantized KV *decodes slightly
  slower* (fixed dequant intercept > tiny bandwidth saving); the win only appears past ~7k.
  (vllm-project.github.io/2026/04/22/fp8-kvcache)
- **llama.cpp measured net-neutral-to-negative at ALL lengths** (gemma Q4_0 @64k: F16 4.75
  vs Q4_0 4.57 tok/s — *slower*); "dequant cost and bandwidth savings roughly cancel."
  (llama.cpp disc. #24109)

**Implication for us:** our N=8 decode benches run at *short* context — exactly the regime
where 4-bit V is *expected* net-neutral-or-slower **even before** our extra FWHT overhead.
So the measured TQ slowdown is NOT primarily a bug in our TQ — it is the known short-context
crossover, *plus* avoidable runtime Hadamard work that the SOTA folds away entirely.

## Our current TQ-HB-V has THREE runtime costs; SOTA eliminates two of them offline

Per-layer, per-step, our hybrid path pays:
- **(a) FWHT-V rotate on KV-write** — rotate V by H before 4-bit quantize.
- **(b) 4-bit V dequant in the flash inner loop** — inherent to any low-bit KV.
- **(c) SEPARATE FWHT-undo dispatch on sdpa_out** + a full read+write round-trip of that
  buffer per layer (our flash kernel has no in-kernel inverse-FWHT).

**(a) and (c) are BOTH foldable offline. (b) is inherent and already fused.**

### The full QuaRot fold (eliminates BOTH runtime FWHT dispatches)

H acts on `head_dim` (256), softmax/QK act on `seq` — they commute. H = D1·FWHT is
orthogonal (H⁻¹ = Hᵀ); D1 sign is a FIXED constant (TBQ_SIGNS_256), FWHT fixed → H is one
fixed 256×256 matrix applied to *every* head (head-independent, so GQA does not complicate it).

- **Fold H into `v_proj` offline** → V emerges already-rotated; the runtime FWHT-on-write (a)
  disappears. Only the inherent per-block scale-compute + 4-bit pack remains. This is QuaRot's
  `W_v ← W_v·H` (SpinQuant `rotate_ov_proj`).
- **Fold H⁻¹ into `o_proj` offline** → `attn_out_rotated = H·attn_out_true`, and
  `o_proj·H⁻¹` un-rotates it as a *free matmul side-effect*; the separate undo dispatch +
  sdpa_out round-trip (c) disappears. This is QuaRot's `W_out ← H·W_out` / SpinQuant
  `apply_exact_had_to_linear(o_proj, output=False, R2)` — **zero online op on the value path**.

Derivation (exact in inf. precision):
```
V_stored      = quant(H · V_true)                         # H folded into v_proj
attn_rotated  = Σ_j softmax_j · dequant(V_stored_j) ≈ H · (Σ_j softmax_j · V_true_j) = H·attn_true
o_proj_true(attn_true) = o_proj · attn_true
             = o_proj · H⁻¹ · (H·attn_true) = (o_proj·H⁻¹) · attn_rotated   # H⁻¹ folded into o_proj
```
After folding, **re-quantize both weights** (quantize `H·v_proj` and `o_proj·H⁻¹`, not the
originals — this is the step a hand-roll gets wrong). The per-256-block RMS norms stay at
runtime (they're data-dependent scale computation, NOT rotation) — correct and unchanged.

Caveats (all real, from agent #1):
- Per-head **block-diagonal** fold — we only rotate within head_dim, so it folds cleanly.
- **RoPE is the wedge** — this works for V→o_proj (no positional op between) but NOT for Q/K
  (RoPE sits in the middle). → do **not** also rotate+fold K. K stays F16 (correct as-is).
- **Quality is equivalent-or-better but NOT bit-identical** to the current model: we trade an
  *exact* f32 runtime un-rotation for a *folded-into-quantized-o_proj* un-rotation. QuaRot
  reports rotation improves quantizability (outlier suppression), but it requires a model
  re-conversion + coherence validation. This is a conversion-time artifact change.

## Apple-Silicon-specific evidence (agent #3 — most actionable)

- **The decisive in-kernel lever: do dequant + any residual rotation INLINE in one fused Metal
  SDPA dispatch — NEVER materialize an fp16 KV copy and never a separate round-trip.** That is
  *exactly* our sin (c). mlx-qsdpa (packed-uint32 load → bit-shift → scale → `simd_sum`, no
  fp16 buffer) is **1.28× over FP16 at 128K, 1.17× at 64K**, and switches FP16 SDPA <16K /
  fused kernel above (`cache_sdpa`). (github.com/Thump604/mlx-qsdpa)
- **RotorQuant (llama.cpp Metal, rotation-KV that is actually FASTER):** block-diagonal 2D/4D
  rotations **O(d) not O(d log d)**, drop-in, **119 vs 93 tok/s decode (+28%)**, with the
  *critical inverse-rotation-on-V-dequant fix* (their PPL 15369→7.05 bug is the same class as
  our undo). (github.com/scrya-com/rotorquant)
- **"When Quantization Is Free" (arXiv 2605.05699):** proof rotation-KV CAN be net-positive on
  Apple — SRFT (not SRHT, because mixed-radix FFT handles non-pow2 + maps to AMX), ~25 ns/vec,
  **−3% to −8% ms/tok on Gemma-3-1B**, ~3× compression, ~0 ΔPPL. Mechanism: "a 3× compressed
  cache transfers 3× less per step … kernel cost is below the bandwidth saved."
- FA3: a fused Hadamard is `O(d log d)` and **bandwidth-bound, so it fuses with an adjacent
  bandwidth-bound op (rotary) 'for free'** — but the shipped fusions are **QK-side and
  cache-boundary**; "rotate post-softmax·V in-register before the epilogue write is the logical
  next fusion but is NOT a documented shipped pattern." → our in-kernel-fuse fallback is
  unproven; **the fold is strictly better (zero runtime cost vs ≤7%).**

## Ranked path (highest ROI first)

1. **Full QuaRot fold: H→v_proj + H⁻¹→o_proj, offline, re-quantized.** Eliminates BOTH runtime
   FWHT dispatches (a)+(c): ~30 undo dispatches/step + 30 sdpa_out read+write round-trips/step
   gone (dispatch-count AND bandwidth win — both levers we already know matter on this
   memory-bound path), AND ~30 FWHT-on-write rotations/step gone. Leaves only inherent (b).
   Quality equiv-or-better, **not bit-identical** → conversion-time change, needs coherence
   validation. **This is the SOTA-standard structure and the highest-ROI single change.**
2. **Context-adaptive V (F16 short / TQ-4bit past the ~7k crossover).** Matches mlx-qsdpa
   `cache_sdpa` / vLLM / llama.cpp config. At short context (our benches, and most chat turns)
   use F16 V → no dequant, no rotation, equal-to-llama. Engage TQ-V only when the cache is big
   enough to pay off. This is what actually makes us "as fast as llama.cpp" at short context
   AND a memory win at long context. (Independent of, and composes with, lever 1.)
3. **Verify (b) is truly fused — no fp16 V materialization.** Confirm our flash inner loop
   unpacks 4-bit V inline (mlx-qsdpa style) and does not stage an fp16 V buffer. If it does,
   that's a separate bandwidth leak to close.
4. **(Research/later) cheaper structured rotation** — only relevant if any online rotation
   survives the fold. RotorQuant block-diagonal O(d) beat full-WHT on Metal; SRFT for Apple.
   With lever 1 the output/write rotations are folded → likely moot for us.

## Recommendation

Start with **lever 1 (the fold)** — it is the SOTA structure, removes the exact avoidable
overhead the user flagged, and is a self-contained offline conversion change. Spike it: fold H
into v_proj + H⁻¹ into o_proj at conversion (gemma4 `value.weight` / `attn_output.weight`),
re-quantize, drop the runtime FWHT-write + FWHT-undo dispatches, validate N=8 coherence
(0/120 single-process) + decode speed + PPL vs current. Then layer **lever 2 (context-adaptive
V)** for the short-context regime. Both are quality-affecting (lever 1) / behavior-changing
(lever 2) → codex plan-check + user go-ahead before building, per ADR-040 process.

## EXECUTION LOG — codex plan-check + falsification spike (2026-06-27)

**v1 (fold BOTH v_proj + o_proj, all layers): codex REJECT.** Root flaw: Gemma global/full-attn
layers under `attention_k_eq_v` have `v_proj = None` (V derived from k_proj; model.rs:1050) — no
independent `attn_v.weight` to fold into; folding into k_proj would rotate K and break QK/RoPE.
Also the D=512 no-FWHT encoder norms are per-256-half so a 512-pt H breaks quant equivalence.

**v2 (RE-SCOPED: fold ONLY o_proj, all layers; V-write UNCHANGED): codex APPROVE-WITH-CHANGES.**
The o_proj fold (H⁻¹ → attn_output.weight) is independent of where V comes from — it works on
sliding AND global layers because `sdpa_out = H·true_sdpa` holds as long as runtime still
produces H·V (which the kept FWHT-V-write does). This sidesteps BOTH the v_proj=None blocker and
the D=512 norm concern (V quant byte-unchanged). codex confirmed: o_proj always independent (Q1);
H head-independent so one per-head-dim H⁻¹ block applies to all q-head blocks (Q3); fold axis =
o_proj INPUT columns, head-major, matches runtime contract (Q4); global 512-block well-defined
(Q5). **Required change (Q2/Q6):** more undo sites than first listed — batched_body.rs:739/747 +
:892/898; gpu_full_attn.rs:1350 + :1470/1475 (legacy HB) + :1540/1547 (4-bit TQ) + :1441
(fused-undo HF2Q_TQ_HB_OUT_FUSED) + flash_attn_vec_tq_hb.rs:491 internal. A folded model through
ANY live undo path silently DOUBLE-un-rotates. Fix: a model-level `fwht_folded` GGUF-metadata
flag set at conversion that GATES EVERY undo site off (folded + legacy both correct, double-undo
impossible). No non-debug consumer reads sdpa_out expecting it rotated (only post-undo dumps).

**Algebraic simplification (verified):** H⁻¹ = D1·FWHT_norm and the fold multiplies o_proj's
input columns ⇒ `W_o'[row, head-block] = sign_premult_fwht(W_o[row, head-block])` — the fold is
LITERALLY the existing V-encode rotation applied to each row of o_proj per head block. Reuse the
shipped primitive (turboquant.rs apply_d1_sign_mask_inplace + fwht_inplace); no new transform.

**FALSIFICATION SPIKE PASSED (mlx-native/tests/adr_040_tq_fold_oproj_shadow.rs):** pure-F32, no
quant, asserts `folded_o_proj·rotated == original_o_proj·undo(rotated)` for one sliding (256) and
one global (512) layer. Result: sliding max_rel=2.3e-6, global max_rel=3.6e-6, undo_err≈1e-7 —
fp roundoff only. The fold math is CONFIRMED; correctness rests on H being orthonormal
(inner-product-preserving), which normalized-FWHT + ±1 D1 sign guarantees.

**REMAINING BUILD (v2, codex-approved):** (1) conversion: apply the row-wise per-head encode_rot
to attn_output.weight before quant (orchestrator.rs:460), per-layer head_dim 256/512 + matching
sign table, set GGUF `fwht_folded` flag; (2) runtime: gate ALL undo sites on !folded; (3)
validate on a real converted model — N=8 coherence (0/120 single-process), slot_aware parity,
PPL vs current, decode tok/s. Add quant only after the F32 path is green (done). v_proj
sliding-only fold (kills the write rotation too) deferred as a follow-on.

## PAYOFF SPIKE — REFUTED: the fold yields ~0% decode speedup (2026-06-27)

Before building the multi-file conversion+gating plumbing, measured the UPPER-BOUND decode payoff
of the fold by gating the runtime FWHT-undo dispatches behind `HF2Q_TQ_SKIP_UNDO=1` (output
incoherent — TIMING ONLY) and running the N=8 throughput probe (gemma4 Ara Q5_K_M, M5 Max,
HF2Q_BENCH_N=8) interleaved, undo-on vs undo-removed:

| undo | tok/s (3 trials) | GPU-busy ms/step |
|---|---|---|
| ON  (skip=0) | 223.3, 222.8 | 27.56, 27.65 |
| OFF (skip=1) | 223.8, 221.6 | 27.62, 27.70 |

**Removing the undo dispatch + sdpa_out round-trip = ZERO measurable speedup** (Δ within
run-to-run noise; skip=1 fractionally SLOWER in one pair). The undo is a negligible fraction of
the GPU-WORK-BOUND 27.6ms/step. ⇒ The QuaRot o_proj fold, though mathematically CORRECT
(F32-shadow spike passed) and SOTA-standard, is **NOT a decode-speed lever for our short-context
N=8**. Building the conversion+gating plumbing would be effort for no measurable gain — MANTRA
call: don't build it. (The v_proj write-rotation fold removes an even smaller in-kernel cost →
refuted by extension.)

**This CONFIRMS the deep-research reframe empirically:** TQ's decode cost is NOT the avoidable
Hadamard machinery; at short context 4-bit-V is inherently ~net-neutral (the ~7k crossover).
The decode is GPU-WORK BOUND at 27.6ms/step ≈ llama's ENTIRE step (27.5ms); the residual gap to
llama (223 vs 291 t/s) is the **wall-vs-GPU-busy delta** (35.9ms wall vs 27.6ms GPU = 77%
utilization → ~23%/8.3ms per-step is NON-GPU host/scheduling/sync, NOT decode-body GPU work).
That host-overlap gap (task #21) — not TQ — is the real "as fast as llama.cpp" lever.

**DECISION:** iter-H-TQ-fold CLOSED as a speed lever (correct but no payoff). The F32-shadow
spike + this refutation are the durable artifacts. Re-target the 23% wall-vs-GPU host gap.
