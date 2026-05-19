# Quality-Equivalence Audit — Gemma 4 26B-A4B-IT Q5_K_M (hf2q vs bartowski/llama.cpp)

**Date:** 2026-05-19
**Reviewer:** worker (under ADR-033 §P1 acceptance gate) + post-audit verification by coordinator
**Model:** google/gemma-4-26b-a4b-it (MoE, 128 experts, 8 used, 30 layers, 26B params)

---

## 0. **CRITICAL AMENDMENT (2026-05-19, post-worker, coordinator-verified)**

> The original worker audit below compares hf2q against **bartowski's** Q5_K_M as
> the "llama.cpp reference". Source verification (`grep` of `/opt/llama.cpp/src/`)
> shows this is the **wrong reference**:
>
> 1. **`use_more_bits` IS ALREADY PORTED in hf2q** at `src/quantize/ggml_quants/standard_policy.rs:317`,
>    and is called at the correct branches for Q5_K_M (lines 556, 657). So the
>    worker's headline recommendation "port `use_more_bits`" is moot — it's there.
>
> 2. **Bartowski's mix is NOT canonical Q5_K_M output**. Per
>    `/opt/llama.cpp/src/llama-quant.cpp:511-545`, the `attn_v → Q8_0` branch
>    fires only on `n_expert == 8` (Mixtral, etc.). Gemma 4 has **n_expert=128**,
>    so canonical Q5_K_M would promote attn_v to **Q6_K under `use_more_bits`**,
>    not Q8_0. Bartowski's 25× Q8_0 is a custom override (likely via
>    `--token-embedding-type` / `--output-tensor-type` / `--type-attn-v` flags
>    or a per-tensor override file), not the canonical Q5_K_M output.
>
> 3. **hf2q's mix appears to match canonical Q5_K_M**: 13× Q6_K + 12× Q5_K + 5× residual
>    on attn_v is exactly the `use_more_bits(i, 30)` ratio (≈14 of 30 layers true).
>    This is **correct canonical behavior** for Gemma 4 + Q5_K_M.
>
> 4. **The `final_logit_softcapping` "missing key" is NOT quality-damaging on Gemma 4**.
>    Per `/opt/llama.cpp/src/models/gemma4.cpp:20`, the key is read with the optional
>    flag (`false`), and the default in `llama-hparams.h:103` is `30.0f` — which is
>    exactly the gemma4 value. Missing this KV is a no-op for output quality.
>
> **Implication:** The §P1 acceptance gate requires comparison against
> `convert_hf_to_gguf.py | llama-quantize Q5_K_M` (canonical, no overrides),
> **not** against bartowski. The worker's "FAIL" verdict against bartowski is
> measuring a real perplexity gap, but most of that gap is bartowski's custom
> attn-Q8_0 promotion, not a defect in hf2q's port of canonical Q5_K_M.
>
> Next mandatory step (before any code change): generate
> `tmp/byte-cmp/gemma4-llama-canonical-q5_k_m.gguf` via `convert_hf_to_gguf.py |
> llama-quantize Q5_K_M` and re-measure perplexity ratio against THAT reference.
> Only then can we claim §P1 PASS/FAIL meaningfully.
>
> Source verification for both "missing quality KVs" the worker flagged:
>
> - **`final_logit_softcapping`**: `gemma4.cpp:20` reads it with `false` (optional);
>   default in `llama-hparams.h:103` is `30.0f` — exactly the gemma4 value.
>   **Missing this KV is a no-op on Gemma 4.**
> - **`rope.freq_base_swa`**: `gemma4.cpp:13` reads it with `false` (optional);
>   default in `llama-hparams.h:118` is `10000.0f` — exactly the gemma4 value.
>   **Missing this KV is also a no-op on Gemma 4.**
>
> Both happen to be coincidentally-correct defaults for Gemma 4 specifically.
> Emitting them is still good hygiene (the defaults could change in future
> llama.cpp versions, and other architectures with different gemma4-like
> values would break), but they do NOT cause the measured perplexity gap.
>
> What survives from the worker audit unchanged:
> - **Coherence is PASS** on both hf2q and bartowski.
> - **No imatrix consumer in hf2q** is genuine but orthogonal to the canonical-quant
>   question (canonical Q5_K_M from `llama-quantize` does NOT use imatrix unless
>   `--imatrix` is explicitly passed).
> - **`quantize.imatrix.*` metadata block** on bartowski is the signature that bartowski
>   ran with `--imatrix`. The canonical reference we need to generate should also
>   be run without imatrix to match what hf2q produces.
>
> Codex-review per `[[feedback-codex-review-loop-rule-2026-05-17]]` is pending for
> both the worker doc and this amendment.

---

## 1. TL;DR (original worker — NOT FINAL per §0 amendment)

hf2q's `--quant q5_k_m` produces **functionally-coherent** GGUF output (loads, runs,
answers correctly) but is **NOT byte-mix-equivalent** to llama.cpp's reference Q5_K_M.
The two files use materially different per-tensor type mixes; on the same biomedical
calibration corpus (`cdv3.txt`, ctx=2048, 20 chunks) hf2q's perplexity is
**6500.07 ± 341 vs bartowski's 3962.26 ± 199 — a ratio of 1.64x (worse)**. The same
delta replicates at ctx=512 (10 chunks): 3134.97 vs 2428.76, **1.29x worse**. Absolute
PPL values are inflated by the corpus being out-of-distribution (biomedical jargon
on an instruct model), but the *relative* gap is stable, multi-context, well outside
the ±400 stderr envelope, and traceable directly to the tensor-mix divergence
identified in §2.

**Verdict: FAIL** the ADR-033 §P1 quality-equivalence gate (ratio outside [0.98, 1.02]).
We can produce a *coherent* Q5_K_M GGUF. We cannot yet produce a *quality-equivalent*
Q5_K_M GGUF. Root causes are mechanical and well-scoped (§7).

---

## 2. Tensor-Type Breakdown — Side by Side

Both files contain **658 tensors** (627 weights + 31 housekeeping/F32 norms) for
the same architecture. Both share **392 F32 housekeeping tensors** (norms, scales,
gates) byte-identical in structure. The divergence is entirely on the **227 quantized
weight tensors**:

### Per-tensor-type counts

| ggml_type | hf2q count | bartowski count | delta | approx-MB hf2q | approx-MB bart |
|-----------|-----------:|----------------:|------:|---------------:|---------------:|
| Q5_K  | 192 | 106 |  +86 | 10895.2 | 10337.5 |
| Q8_0  |  34 |  83 |  −49 |  4473.6 |  3990.9 |
| Q5_1  |  26 |  32 |   −6 |  2414.8 |  2972.1 |
| Q6_K  |  14 |  45 |  −31 |   636.2 |  1064.8 |
| F32   | 392 | 392 |    0 |    43.9 |    43.9 |
| **Sum** | **658** | **658** | **0** | **≈18.0 GB** | **≈17.9 GB** |

Net file delta: hf2q = 19,376,360,992 B vs bartowski = 19,319,197,088 B — hf2q is
**57.2 MB larger**, consistent with the approx-byte summary above (the 1.6× perplexity
penalty is **NOT** "we shaved bits and got smaller"; we shaved bits in the wrong
places and have a *larger* file with *worse* perplexity).

### Where the mix actually diverges (per-pattern, 30 layers each)

| pattern (per layer × 30) | hf2q | bartowski | analysis |
|---|---|---|---|
| `attn_k.weight`     | **30× Q5_K** | **30× Q8_0** | bart promotes K to Q8_0 |
| `attn_output.weight`| **30× Q5_K** | **30× Q6_K** | bart promotes output to Q6_K |
| `attn_q.weight`     | **30× Q5_K** | 16× Q5_K + 14× Q6_K | bart promotes ~half of Q to Q6_K |
| `attn_v.weight`     | 13× Q6_K + 12× Q5_K + 5× Q5_K (residual) | **25× Q8_0** + 5× (residual) | bart promotes V to Q8_0 on most layers |
| `ffn_down.weight`   | 17× Q8_0 + 13× Q5_1 | 14× Q8_0 + 16× Q5_1 | very close, minor shift |
| `ffn_down_exps.weight` | 17× Q8_0 + 13× Q5_1 | 14× Q8_0 + 16× Q5_1 | same as above (MoE down) |
| `ffn_gate_up_exps.weight` | 30× Q5_K | 30× Q5_K | identical |
| `ffn_gate/up.weight` | 30× Q5_K | 30× Q5_K | identical |
| `token_embd.weight` | 1× Q6_K | 1× Q6_K | identical |
| `output_norm.weight` | F32 | F32 | identical |

### Interpretation — bartowski runs the llama.cpp **`use_more_bits` + imatrix** ruleset

Bartowski's mix is the canonical `LLAMA_FTYPE_MOSTLY_Q5_K_M` output as produced by
`llama-quantize` *with* an imatrix calibration. The promotions follow llama.cpp's
`use_more_bits()` heuristic in `src/llama-quant.cpp`:

- attention K/V/output get **bumped up** (Q5_K_M's "M" = mostly Q5_K but more bits
  on important tensors), driven by per-tensor importance scores from imatrix
- ffn_down has a layer-wise mix (early/late layers Q8_0, middle Q5_1)
- this is why the file is **smaller** despite *more* Q6_K/Q8_0: imatrix-guided
  spend re-allocates the bit budget rather than naively quantizing everything to Q5_K

hf2q's mix appears to follow a **simpler, uniform** policy that defaults attention
weights to Q5_K and only uses Q8_0 on a third of ffn_down. Without imatrix
guidance, importance-weighted promotions never happen.

This is the single biggest finding: **the 1.64x perplexity gap is mechanically
explained by hf2q producing a worse-quality bit allocation, not by a kernel bug
or numerical drift**.

---

## 3. Perplexity Comparison

Corpus: `/opt/hf2q/data/calibration/cdv3.txt` (biomedical / scientific paragraphs,
280 KB, 2481 lines).

NB: The absolute PPL numbers are inflated because (a) cdv3 is out-of-distribution
for an instruct model, (b) it is biomedical jargon-heavy, and (c) wikitext-2 raw
was not available locally. **The ratio is meaningful even if the absolute is not**
— both files saw the EXACT same tokenization and the EXACT same evaluation,
so the relative comparison is fair.

### Run 1: ctx=512, 10 chunks

| file | PPL | stderr | ratio |
|---|---:|---:|---:|
| hf2q     | 3134.97 | ±468.51 |  |
| bartowski| 2428.76 | ±353.73 | 1.291x worse |

### Run 2: ctx=2048, 20 chunks (statistically tighter)

| file | PPL | stderr | ratio |
|---:|---:|---:|---:|
| hf2q     | 6500.07 | ±341.43 |  |
| bartowski| 3962.26 | ±198.58 | **1.640x worse** |

The gap is consistent across context lengths and chunk counts and is well outside
the ±stderr envelopes (a 1.0x null hypothesis would require either run to be
within ~1 stderr of the other, which it is not).

Acceptance criterion per ADR-033 §P1: ratio in [0.98, 1.02] is quality-equivalent;
> 1.05 is a regression. **Observed 1.29x and 1.64x — fail by an order of magnitude
relative to the acceptance bar.**

---

## 4. Coherence — Side by Side

Both files were exercised with `llama-cli ... --seed 42 --temp 0 -ngl 999` on
identical prompts. Output is qualitatively equivalent on the probes attempted.

### Probe 1: "What is the capital of France?" (n=64)

**hf2q**
```
[Start thinking]
The user is asking for the capital of France.
The capital of France is Paris.
State the answer clearly.
[End thinking]

The capital of France is **Paris**.
```
[ Prompt: 140.0 t/s | Generation: 113.6 t/s ]

**bartowski**
```
[Start thinking]
The user is asking for the capital of France.
France.
Paris.
State the answer clearly.
[End thinking]

The capital of France is **Paris**.
```
[ Prompt: 192.1 t/s | Generation: 106.8 t/s ]

Both produce the same final answer with very similar reasoning traces. (The minor
difference in the "thinking" block is exactly the kind of low-amplitude argmax flip
expected when bit allocation differs.)

### Probe 2: "Explain in 3 short sentences how photosynthesis works." (n=128)

Both produced coherent on-topic 3-sentence-style outputs that ran past 128 tokens
mid-second-sentence (i.e. both are doing structured "think then answer" — neither
loops, garbages, or diverges in tone). Outputs differ in wording (chlorophyll vs
chloroplasts focus, $H_2O$/$CO_2$ vs prose) — different but quality-equivalent.

**Coherence verdict: PASS.** No loops, no garbage, no first-token flip. The
perplexity gap shows up as wider-margin secondary token choices but does not
break either model's output on common prompts.

---

## 5. Metadata Diff Summary

hf2q kv_count=40 (after recent gemma4-metadata work b514b5c2 + 55ba3abd).
bartowski kv_count=57 (older file, May 17, but more general.* publisher metadata).

### Keys hf2q has but bartowski does not

- `tokenizer.ggml.pre = 'gemma4'` — pre-tokenizer name

### Keys bartowski has but hf2q does not

#### Quality-affecting (genuine omissions in hf2q)

- **`gemma4.final_logit_softcapping = 30.0`** — logit softcap, used at the output
  head in llama.cpp's gemma4 implementation. If llama.cpp's loader requires it,
  it falls back to a default; if it interprets "missing" as "off" this is a
  silent behavior change. **Confirm before claiming the field is optional.**
- **`gemma4.rope.freq_base_swa = 10000.0`** — RoPE base for sliding-window layers
  (vs `freq_base = 1000000.0` for global layers). Without it, llama.cpp likely
  defaults the SWA freq base to the same as global, which **would** affect
  position encoding on the 25 SWA layers. **Same caveat.**
- `tokenizer.ggml.mask_token_id = 4` — only matters for mask-token aware decoding

#### Imatrix lineage (hf2q does not currently produce imatrix metadata)

- `quantize.imatrix.file = '/models_out/...'`
- `quantize.imatrix.dataset = '/training_dir/calibration_datav5.txt'`
- `quantize.imatrix.entries_count = 295`
- `quantize.imatrix.chunks_count = 822`
- `general.quantization_version = 2`

This is the *signature* that bartowski ran llama-quantize **with an imatrix**.
hf2q does not currently consume one (it doesn't write any of these markers).

#### Publisher metadata (cosmetic — does not affect quality)

- `general.type`, `general.tags`, `general.size_label`, `general.basename`,
  `general.finetune`, `general.license`, `general.license.link`,
  `general.sampling.top_k / top_p / temp` — these are bartowski's release card
  info. hf2q omitting them is **fine** for quality.

### Chat templates: byte-identical

Both files contain the same 16,934-byte `tokenizer.chat_template` (verified via
gguf reader). This is a hf2q **win** over what we shipped pre-b514b5c2 and is
required for serve-time chat behavior — **no regression here**.

---

## 6. Verdict

**FAIL — quality-regression vs llama.cpp/bartowski reference at the §P1 gate.**

Specifically:

- **Coherence: PASS** (functional output matches on probed prompts)
- **Perplexity: FAIL** at 1.29×–1.64× the reference (acceptance bar is ±2%)
- **Tensor mix: FAIL** (no `use_more_bits` promotion, no imatrix guidance)
- **Metadata: PARTIAL** (3 quality-affecting KVs missing: `final_logit_softcapping`,
  `rope.freq_base_swa`, plus the imatrix lineage block)
- **Chat template: PASS** (byte-identical)
- **File size: NEUTRAL** (we're 57 MB *larger* than bartowski while being worse —
  i.e. our bit budget is genuinely mis-allocated, not just "smaller and worse")

Per operator standing rule: "ensure we're now able to make gguf/quants at the
same quality level as llama.cpp" — **not yet met**. Functional output is fine;
quality parity is not.

---

## 7. Recommendations

In rough order of leverage:

1. **(BIG) Implement `use_more_bits`-style attention promotion in the Q5_K_M policy.**
   The single biggest contributor to the gap is hf2q quantizing
   `attn_k / attn_v / attn_output / attn_q` to **Q5_K** while llama.cpp's
   Q5_K_M policy promotes them to **Q6_K** or **Q8_0** based on importance.
   Port llama.cpp's `src/llama-quant.cpp::use_more_bits()` rule (with the layer-aware
   tweaks for "first 1/8 + last 1/8 layers stay higher precision"). This alone
   should close most of the gap **even without imatrix**, since the static rule
   is well-tested.

2. **(BIG, downstream of #1) Add imatrix-guided tensor classification.**
   bartowski's `entries_count=295, chunks_count=822` means an imatrix was
   computed on ~820 chunks of natural text and used to compute per-row
   importance during `quantize_tensor`. hf2q currently has no imatrix consumer.
   This needs an ADR — it touches the entire codec dispatcher. **But:** ADR-033
   already covers convergence on llama.cpp's codec; an imatrix consumer is the
   natural follow-on. Note: without imatrix, llama.cpp's static
   `Q5_K_M.use_more_bits` rule still produces materially better output than
   uniform Q5_K, so #1 is independently valuable and ship-able before #2.

3. **(MEDIUM) Surface `gemma4.rope.freq_base_swa` and
   `gemma4.final_logit_softcapping` from the HF config into the GGUF writer.**
   These two KVs are quality-affecting and present in the HF config files
   we're loading from. Cross-check llama.cpp's gemma4 loader to confirm whether
   missing values default safely or cause silent behavior change; either way,
   emit them. This is a small `gguf_writer.add_*` patch.

4. **(SMALL) Add `general.quantization_version = 2`.** Cosmetic but signals
   "produced by a Q-spec-v2-aware quantizer" to downstream tools; bartowski emits
   it, we don't.

5. **(METHODOLOGY) Add a perplexity-ratio gate to the test matrix.**
   The current ADR-033 §P1 acceptance language calls for ±2% perplexity equivalence
   but there's no harness running it on real models. Make the gate concrete:
   `tests/quality/ppl_ratio_gate.sh <hf2q.gguf> <reference.gguf>` returns nonzero
   if ratio outside [0.98, 1.02]. Use cdv3 + at least one in-distribution corpus
   (real wiki.test.raw, downloadable via HF datasets) so the absolute numbers are
   also sane.

6. **(METHODOLOGY) Per the standing test-both-families rule, repeat this audit
   on Qwen 3.6 35B-A3B once we have a non-hf2q reference of the same family for
   side-by-side comparison.** This audit only covers Gemma 4 so far.

---

## Provenance

- hf2q file: `/opt/hf2q/tmp/byte-cmp/gemma4-hf2q-q5_k_m.gguf` (19,376,360,992 B),
  produced by `hf2q convert /opt/hf2q/models/google-gemma-4-26b-a4b-it --quant q5_k_m`
  on hf2q HEAD ending `b1cbcd69` (post-`b514b5c2` Gemma 4 metadata fix).
- Reference file: `/opt/hf2q/models/gemma-4-26b-a4b-it-bartowski/google_gemma-4-26B-A4B-it-Q5_K_M.gguf`
  (19,319,197,088 B), published by bartowski (canonical `convert_hf_to_gguf.py | llama-quantize`
  pipeline, with imatrix from `calibration_datav5.txt`, 822 chunks).
- Tools: llama-perplexity built fresh from `/opt/llama.cpp` (build-id `b9223-e15384a5c`)
  using `SDKROOT=$(xcrun --show-sdk-path)`. Metadata dumps via
  `gguf-py/scripts/gguf_dump.py`.
- Cmdlines:
  - `llama-perplexity -m <gguf> -f cdv3.txt -ngl 999 --ctx-size {512,2048} -t 8 --chunks {10,20}`
  - `llama-cli -m <gguf> -p "<probe>" -n {64,128} --seed 42 --temp 0 -ngl 999 -st`
