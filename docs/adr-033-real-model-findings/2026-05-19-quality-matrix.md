# ADR-033 Exit Criterion — Quality Matrix vs Canonical llama.cpp

**Date:** 2026-05-19
**Model under test:** `google/gemma-4-26b-a4b-it` (MoE, 128 experts, 8 used, 30 layers, 26B params)
**Source:** `/opt/hf2q/models/google-gemma-4-26b-a4b-it/` (BF16 safetensors, ~48 GB)
**Canonical F16 ref:** `/opt/hf2q/tmp/byte-cmp/gemma4-llama-canonical-f16.gguf` (50.5 GB, produced by `convert_hf_to_gguf.py` at pinned llama.cpp SHA)
**Calibration corpus:** `/opt/hf2q/data/calibration/cdv3.txt` (bartowski's `calibration_datav3.txt`, 273 KB)
**PPL settings:** `llama-perplexity -c 2048 --chunks 20 -ngl 999` on M5 Max
**Tooling:** `llama-quantize` (canonical) vs `hf2q convert` (under test); both consume the same canonical F16 GGUF / source HF dir

---

## Exit criterion (operator-stated 2026-05-19)

> "We need to be sure to end-to-end test our quant/convert — check model quality
> etc. … Are the models we create the same as the quality llama.cpp's tooling
> does? That's the exit criteria."

Concrete acceptance: **|ratio − 1| < 0.05** per cell (perplexity within 5 %
of canonical llama.cpp output on the same corpus + context length). The §P1
closure target was ±2 %; this doc uses the more permissive ±5 % envelope to
account for FP non-associativity across the longer per-cell pipeline (HF →
F16 GGUF → `llama-quantize <type>` ≠ HF → `hf2q convert --quant <type>`).

---

## Matrix — UPDATED 2026-05-19 POST-CLOSURE (commit `50fd89c2`)

| Quant   | Canonical PPL          | hf2q PPL              | Ratio  | Tensors | Verdict |
|---------|------------------------|-----------------------|--------|---------|---------|
| Q4_K_M  | 13183.4003 ± 697.92    | 13183.4003 ± 697.92   | 1.0000 | **658/658 byte-identical** | 🏆 EXACT |
| Q5_K_M  | 5458.7019 ± 283.97     | 5458.7019 ± 283.97    | 1.0000 | **658/658 byte-identical** | 🏆 EXACT |
| Q6_K    | 4150.6105 ± 211.32     | 4150.6105 ± 211.32    | 1.0000 | **658/658 byte-identical** | 🏆 EXACT |
| Q8_0    | 4119.9252 ± 206.98     | 4119.9252 ± 206.98    | 1.0000 | **658/658 byte-identical** | 🏆 EXACT |

All four cells: per-tensor byte-cmp via `scripts/byte_cmp_gguf.py` reports zero diffs across every F32/Q4_K/Q5_K/Q5_0/Q5_1/Q6_K/Q8_0 tensor; PPL is bit-for-bit identical between canonical `convert_hf_to_gguf.py --outtype f16 | llama-quantize <type>` and `hf2q convert --quant <type>`. File-size delta remains 448 bytes (header-KV ordering only — not a tensor-data difference).

Two root causes closed the historical gaps:

1. **FMA non-associativity** between clang (`-O3 -march=native` + `-ffp-contract=on` default fuses `a*b+c` into single-rounded `fmadd`) and rustc (`--release` defaults `fp-contract=off`, no auto-FMA). Fix: explicit `f32::mul_add` at every accumulator hotspot, `a*b - c*d` determinant pattern (encoded as `a.mul_add(b, -c*d)`), and `scale*l + min - x` error-evaluation pattern (encoded as `scale.mul_add(l, min) - x`). Applied across `make_qx_quants`, `make_qkx2_quants`, `make_qkx3_quants`, `make_qp_quants` in `src/quantize/ggml_quants/common.rs`.

2. **F16 intermediate round-trip**. Canonical pipeline stores BF16/F32 weight tensors as F16 in the intermediate GGUF (`/opt/llama.cpp/conversion/base.py:875-876`) before `llama-quantize` reads them back to F32; this round-trip is lossy below F16's normal range (~6.1e-5). hf2q's previous direct `BF16 → F32 → Q4_K` path produced ~779/5.77M element divergence on `blk.0.attn_k.weight` alone. Fix: `src/convert/orchestrator.rs::stream_tensor` applies `F32 → F16 → F32` to the input vector at the quantizer branch, mirroring canonical's behavior.

### Earlier (pre-closure) measurements — historical

| Quant   | Canonical PPL          | hf2q PPL              | Ratio  | Verdict | Notes |
|---------|------------------------|-----------------------|--------|---------|-------|
| Q5_K_M  | 5471.84 ± 284.38       | 5411.20 ± 281.65      | 0.9889 | ✅ PASS | Pre-FMA-fix; was the §P1 byte-mix-equivalent measurement at `b03915af`. Closed exactly post-`50fd89c2`. |
| Q4_K_M  | 13183.40 ± 697.92      | 11502.00 ± 607.41     | 0.8725 | ❌ FAIL | Pre-FMA-fix; 12.8% delta from FMA + F16-roundtrip artifacts. Closed exactly post-`50fd89c2`. |
| Q6_K    | 4150.61 ± 211.32       | 4193.80 ± 213.74      | 1.0104 | ✅ PASS | Pre-FMA-fix; closed exactly post-`50fd89c2`. |
| Q8_0    | 4119.9252 ± 206.98     | 4119.9252 ± 206.98    | 1.0000 | ✅ PASS | Q8_0 was unaffected by either bug (single-byte scalar quant, no iterative scale search). |

Remaining matrix cells (per ADR §1 AC#2) are operator-time follow-ups:
`q4_0`, `q4_k_s`, `q5_k_s`, `iq4_nl`, plus the APEX tier matrix.

**Cross-family verification (Qwen)**: blocked by arch support. The supported set
in hf2q convert (`src/convert/arch/`) is `llama, gemma3, bert, nomic_bert,
qwen3_moe, qwen3_vl, minimax_m2`. The downloaded `Qwen/Qwen3.5-35B-A3B`
declares `architectures: ["Qwen3_5MoeForConditionalGeneration"]` with
`model_type: qwen3_5_moe` and an interleaved `linear_attention` + `full_attention`
layer schedule — this is a new architecture distinct from `qwen3_moe`. Adding
`qwen3_5_moe` support is a separate work item (per-arch tensor mapping +
linear-attention layer handling), not a §P1 issue. Once arch support lands,
the same byte-cmp + PPL gates from this matrix should re-run against
Qwen3.5/3.6 to satisfy the [[feedback-test-both-families]] rule. Until then,
the §P1 closure stands on Gemma 4 alone.

## Verdict summary

**3 of 4 measured cells PASS the exit criterion** (|ratio − 1| < 0.05):
- Q5_K_M: 0.989 ✅ (1.1% below canonical)
- Q6_K:   1.010 ✅ (1.0% above canonical)
- Q8_0:   1.000 ✅ (identical to 4 decimal places — empirically byte-equivalent at the data-flow level)

**1 cell FAILS:**
- Q4_K_M: 0.872 ❌ (12.8% below canonical, outside ±5% envelope)

The Q5_K + Q6_K + Q8_0 kernels are confirmed clean. The **Q4_K kernel
is the lone identified divergent kernel** (Q4_K_M uses Q4_K + Q6_K;
Q6_K is independently clean; therefore Q4_K is the divergent layer).

The Q4_K kernel's byte-cmp fixture tests
(`q4_k::tests::byte_cmp_noim` + `byte_cmp_im`) pass at HEAD, so the
divergence is input-distribution-dependent: the fixture's specific
input distribution doesn't exercise the divergent code path that
real Gemma 4 weights trigger.

**Direction note:** hf2q's Q4_K_M PPL (11502) is LOWER (better) than
canonical's (13183) — an unexpected divergence direction. Two possible
explanations:

1. **PPL-variance hypothesis:** at PPL ~10K-13K (Gemma 4 26B on cdv3 is
   higher than typical PPL ranges because cdv3 is biomedical and Gemma
   4 is general-purpose) the ±5% envelope may not capture sample
   variance. The ±5.3% error bars on each measurement compound to
   ~7.5% relative uncertainty on the ratio — making 0.872 statistically
   meaningfully different from 1.0 but the magnitude could be partially
   noise.

2. **Q4_K kernel "accidentally better" hypothesis:** hf2q's Q4_K
   implementation may have a subtle numerical difference (e.g., one
   rounding boundary off by ½ ULP) that happens to produce a slightly
   different quantization choice on real-model weight distributions,
   landing in a regime that has marginally better quality on cdv3.
   Byte-cmp fixture tests pass because the fixture input doesn't
   straddle the boundary.

## Iter-21 deeper-dive: per-type byte-cmp across all 4 files

Extracted per-tensor bytes via `gguf-py` and SHA-compared against
canonical for every tensor in every file. Per-quant-type summary:

| Kernel | Files | Total tensors | Bytes-equal count | Diff% | Files exercising |
|--------|-------|---------------|-------------------|-------|------------------|
| F32    | all   | 1568          | 1568              | 0.000% | passthrough — exact match |
| Q5_0   | Q4_K_M | 32           | 32                | 0.000% | byte-equivalent ✅ |
| Q5_1   | Q5_K_M | 32           | 26 (6 outliers)   | ~0%   | 14 diff bytes total — likely matches all |
| Q8_0   | all   | 354           | 354               | 0.000% | byte-equivalent ✅ |
| **Q5_K** | Q5_K_M | 192        | **0**             | **0.002%** | all tensors diverge |
| **Q6_K** | all  | 234           | **0**             | **0.038%** | all tensors diverge |
| **Q4_K** | Q4_K_M | 192        | **0**             | **0.066%** | all tensors diverge |

**Diagnostic conclusion: ALL K-QUANT KERNELS** (Q4_K, Q5_K, Q6_K)
**diverge from canonical** at the byte level. Non-K-quants (Q8_0,
Q5_0, Q5_1) are byte-equivalent. The divergence rate scales with
quant aggressiveness (Q4_K most aggressive → highest divergence
rate per byte; Q5_K least aggressive K-quant → lowest rate).

PPL impact correlates with per-bit-error sensitivity of each kernel:
- Q8_0 (no divergence) → ratio 1.0000
- Q6_K (0.038%) → ratio 1.010 (within ±2% noise)
- Q5_K_M (Q5_K 0.002% + Q6_K 0.034%) → ratio 0.989 (within ±2%)
- Q4_K_M (Q4_K 0.066% + Q6_K 0.038%) → ratio 0.872 (FAIL ±5%)

## First differing byte localization (Q4_K_M, `blk.0.attn_k.weight`)

- First diff at offset 6914 (Q4_K block 48, `dmin` field, F16)
- `d` field IDENTICAL → `max_scale` matches canonical
- `dmin` field DIFFERS → `max_min` (output of `make_qkx2_quants`)
  diverges in at least one of 8 sub-blocks within block 48
- 278 of 22,528 blocks differ in this tensor (1.2%)

## Root-cause: FMA non-associativity — **EMPIRICALLY VERIFIED at assembly level (iter-22)**

Disassembled both compiled `make_qkx2_quants` symbols and counted
FMA-class instructions:

| Binary | Total instructions | fmadd | fmul | fadd | fsub |
|--------|-------------------|-------|------|------|------|
| `libggml-base.0.dylib::_make_qkx2_quants` (clang -O3 -march=native) | 178 | **12** | 7 | 4 | 6 |
| `hf2q::quantize::ggml_quants::common::make_qkx2_quants` (cargo --release) | 1319 | **0** | 126 | 241 | 47 |

Canonical (clang) emits 12 `fmadd` instructions for the accumulator
hot-spots (`sum_x += w * x[i]`, `sum_l += w * li`, `sum_l2 += w * li *
li`, `sum_xl += w * li * x[i]`, `best_error += w * diff`,
`cur_error += w * diff`). Each `fmadd` is a single-rounded
multiply-add. Rust at `--release` emits **zero FMA instructions**
because rustc's default `-Cllvm-args=-fp-contract=off` is stricter
than clang's `-ffp-contract=on` default.

ULP-level divergences in the FMA-fused accumulators propagate
through `make_qkx2_quants`'s iterative refinement loop, producing
different `the_min` outputs that, when rounded to F16 for storage
in the `dmin` block field, cross an F16-precision boundary in
0.04-0.07% of blocks.

This explains all observations:
- Only K-quants diverge (only they use `make_qkx2_quants`)
- All K-quants diverge consistently (Q4_K, Q5_K, Q6_K all use it)
- Non-K-quants byte-match (Q8_0, Q5_0, Q5_1 don't use FMA-sensitive ops)
- Byte-cmp fixture tests pass (fixture inputs don't cross F16
  `dmin` boundaries → FMA-vs-non-FMA produce the same rounded output)
- Real-model weights produce divergence (some sub-blocks cross
  the boundary)

## Fix candidates (operator-decision-pending)

If operator chooses **option 2** (force Rust to match canonical):

Replace accumulator hot-spots in `common.rs::make_qkx2_quants`:
```rust
sum_x += w * x[i];           // → sum_x = w.mul_add(x[i], sum_x);
sum_l += w * li_f;           // → sum_l = w.mul_add(li_f, sum_l);
sum_xl += w * li_f * x[i];   // → sum_xl = (w * li_f).mul_add(x[i], sum_xl);
sum_l2 += w * li_f * li_f;   // → sum_l2 = (w * li_f).mul_add(li_f, sum_l2);
best_error += w * diff;      // → best_error = w.mul_add(diff, best_error);
cur_error += w * diff;       // → cur_error = w.mul_add(diff, cur_error);
```

Plus matching changes in `make_qkx3_quants` (used by K-quant
imatrix paths). Total: ~12 line edits.

After change: re-run Q4_K_M PPL, expect ratio → ~1.0 (matching
canonical's `make_qkx2_quants` output exactly via byte-equivalent
FMA computation).

Verification gate: byte-cmp `blk.0.attn_k.weight` between
canonical and fixed-hf2q — should be 0 differing bytes.

Non-K-quant kernels use simpler O(n) scale-and-round patterns
without the determinant computation → no FMA-sensitive ops → byte
match.

The Q4_K kernel's byte-cmp fixture tests
(`q4_k::tests::byte_cmp_noim` + `byte_cmp_im`) pass at HEAD
because the fixture's specific F32 inputs don't cross the F16
`dmin` boundary that real Gemma 4 weights cross in ~0.07% of blocks.

## Three options for operator

1. **Accept the divergence as-is.** Q5_K_M, Q6_K, Q8_0 all pass the
   ±5% envelope. Q4_K_M deviates 13% in the BETTER direction
   (lower PPL on cdv3). Document and ship.

2. **Force Rust to match canonical via `f32::mul_add`.** Rewrite the
   3 FMA-sensitive lines in `make_qkx2_quants` using explicit
   `mul_add` calls. Re-test all K-quant matrix cells. Should bring
   Q4_K_M to ratio ~1.0 at the cost of slightly worse PPL on Gemma 4.

3. **Accept with caveat.** Document that hf2q's K-quants produce
   marginally different (sometimes-better) output vs canonical on
   real-model weight distributions; add a `--strict-canonical` flag
   if a downstream consumer needs byte-exact match.

Per [[feedback-codex-review-loop-rule-2026-05-17]] + the mantra
("create hypotheses that are testable before changing code"), option
(2) requires testing the FMA hypothesis first — likely via a
minimal repro that builds the same C function with `-O0 -ffp-contract=off`
and compares against the `-O3 -ffp-contract=on` build. Operator-time.

---

## Cell-level provenance

### Q5_K_M — PASS at ratio 0.9889

- Canonical: `tmp/byte-cmp/gemma4-llama-canonical-q5_k_m.gguf` (19.13 GB),
  produced by `llama-quantize gemma4-llama-canonical-f16.gguf <out> Q5_K_M`
  at pinned llama.cpp SHA.
- hf2q: `tmp/byte-cmp/gemma4-hf2q-q5_k_m-postfix2.gguf` (19.13 GB), produced
  by `hf2q convert /opt/hf2q/models/google-gemma-4-26b-a4b-it --quant q5_k_m`
  at HEAD (commit `eaa8727b` or earlier — code path unchanged since
  `b03915af` for non-imatrix StandardPolicy quants).
- Measurement timestamps: re-verified iter-13 at HEAD; numbers match
  `b03915af`'s recorded 5471.84 / 5411.20 byte-for-byte.

### Q4_K_M — _pending_

(Filled in once background `llama-quantize` + `hf2q convert` finish and
`llama-perplexity` reports.)

### Q6_K — _pending_

### Q8_0 — _pending_

---

## What this matrix does NOT cover (and why)

1. ~~**Per-tensor byte-cmp** between canonical and hf2q output~~ — **NOW COVERED**. Commit `50fd89c2` achieved strict per-tensor byte-equivalence (`cmp 0` semantics) via `scripts/byte_cmp_gguf.py`. The earlier "byte-mix-equivalent" framing assumed FP-accumulation-order divergence was structural; the d_det `a*b - c*d` FMA pattern + F16 intermediate round-trip turned out to be the actual sources. Strict byte-cmp now holds for Q4_K_M, Q5_K_M, Q6_K, Q8_0 on Gemma 4 26B-A4B-IT.

2. **Cross-arch coverage.** Only Gemma 4 26B is exercised today. The
   source-level transitive proof (§Pa for non-I APEX, §P4b for I-tier
   APEX) covers Qwen 3.5/3.6 MoE + MiniMax M2.7 at the policy layer
   without real-model PPL. End-to-end real-model PPL on those arches
   remains operator-time (each is a 30 GB+ HF download).

3. **APEX tier matrix.** Same logic: source-level proofs discharge the
   policy layer; real-model end-to-end is operator-time per family ×
   tier. The §P1 closure's Q5_K_M result is the only real-model
   measurement performed against canonical so far.
