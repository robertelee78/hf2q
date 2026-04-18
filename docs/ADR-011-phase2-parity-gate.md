# ADR-011 Phase 2 — End-to-End Parity Gate

**Status:** Proposed  
**Author:** Agent #6 (research-parity-gate), swarm-1776516482254-ft5mwj  
**Date:** 2026-04-17  
**Scope:** Phase 2 correctness + prefill parity gate design  
**Companion file:** `scripts/adr-011-phase2-gate.sh` (produced below)

---

## 0. Purpose

Phase 2 success criterion (user directive, verbatim):

> "You'll know you got it right if when we run hf2q it produces coherent
> (basically identical) output as llama.cpp, and just as fast."

This document defines the measurement harness that lets CI or a human
verify both halves on every commit: **text parity** and **prefill
tok/s parity**.

All claims cite `file:line` or a log path.  
The gate is self-contained: a human unfamiliar with this ADR can run it
end-to-end by following §10.

---

## 1. Ground truth from prior work

### 1.1 Binaries

| Tool | Path | Build tag |
|---|---|---|
| llama-completion | `/opt/llama.cpp/build/bin/llama-completion` | b3d758750 (version 8807) |
| llama-bench | `/opt/llama.cpp/build/bin/llama-bench` | b3d758750 |
| hf2q | `target/release/hf2q` | built with `cargo build --release --features metal` |

Established at:
- `scripts/sourdough_gate.sh:95-101` — llama-completion location discovery
- `docs/ADR-011-phase1-port-source-decision.md §3` — llama-bench path confirmed

### 1.2 Model under test

```
/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/
  gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf
```

SHA-256 (from `tests/evals/reference/MANIFEST.json`):
`ae19574dab588e0a742a8cfa1282ccf358ed8a58c5a3483f2bf596020a4f8e6f`

### 1.3 Re-measured llama.cpp peer baselines (2026-04-17, M5 Max)

Source: `docs/ADR-011-phase1-port-source-decision.md §3`

#### Prefill (pp=N, `-n 0`, `-r 3`, median of 3)

| seq_len | fa=1 tok/s | fa=0 tok/s | stdev (fa=1) |
|---|---|---|---|
| 128 | 1114.32 | — | ±34.85 |
| 512 | 3722.23 | — | ±1.91 |
| 1024 | 3604.84 | — | ±8.88 |
| 2048 | 3384.86 | — | ±14.30 |
| 2455 | 3313.62 | 3455.88 | ±5.30 / ±16.09 |

Important finding: **fa=0 beats fa=1 at pp≥2455 on this MoE model**
(`docs/ADR-011-phase1-port-source-decision.md §3.2`). The non-FA peer
(fa=0) is the more conservative reference for hf2q's non-FA prefill path.

#### Decode

| Path | tok/s |
|---|---|
| llama.cpp tg128 fa=1 | 106.31 ±0.27 |
| llama.cpp tg128 fa=0 | 105.35 ±0.30 |

Decode is at parity with hf2q already (`docs/ADR-011-phase1-port-source-decision.md §3.3`).

---

## 2. Canonical prompts

All four prompts are checked into the repo. The fourth (`prefill_2048`) is
used only for prefill benchmarking, not text parity.

| Name | File | Size | Token count |
|---|---|---|---|
| `short_hello` | `tests/evals/prompts/short_hello.txt` | 21 bytes | ~5 tokens |
| `sourdough` | `tests/evals/prompts/sourdough.txt` | 55 bytes | 22 tokens (load-bearing typo — do not fix) |
| `sliding_wrap` | `tests/evals/prompts/sliding_wrap.txt` | 388 bytes | ~80 tokens |
| `prefill_2048` | `tests/evals/prompts/prefill_2048.txt` | 11,592 bytes | 2455 tokens (hf2q) / 2443 tokens (llama.cpp) |

`sourdough.txt` note (`scripts/sourdough_gate.sh:54-56`): the typo
`"Complrehensive"` is intentional and load-bearing — the token trajectory
is locked to this spelling.

`prefill_2048.txt` note (`docs/spike-gate-a-prefill.md §Method`):
constructed as `adversarial_1000.txt × 3`; 11,592 bytes. Token-count
discrepancy (2455 vs 2443) is a known BOS/Unicode path difference, not a bug.

---

## 3. Text parity procedure

### 3.1 What "basically identical" means concretely

For greedy sampling (temp=0, fixed seed), the output is deterministic on each
tool independently. The question is whether the *two tools* agree.

Prior art from `scripts/sourdough_gate.sh` established the working definition:

> "hf2q's decode output is byte-identical to llama.cpp on the DWQ GGUF for
> the first ~830 decode tokens." (`sourdough_gate.sh:7-9`)

The existing measurements (from `tests/evals/reference/MANIFEST.json`):

| Prompt | llama.cpp bytes | hf2q bytes | Common prefix | Current gate floor |
|---|---|---|---|---|
| sourdough | 3658 | 3656 | 3656 | ≥ 3094 |
| short_hello | 46 | 36 | 29 | ≥ 29 (EOS differs only) |
| sliding_wrap | 2327 | 2354 | 752 | ≥ 700 |

**Concrete pass/fail thresholds for Phase 2:**

The Phase 2 gate raises the bar from the Phase 1b floors. After Phase 2
flash-attn integration (ADR-011 §Phase 2–3), the prefill path changes; the
decode output must not regress. Thresholds:

| Prompt | Phase 1b floor | Phase 2 floor | Rationale |
|---|---|---|---|
| sourdough | 3094 bytes | 3094 bytes | Locked to known drift point (token 840 case-flip); do not raise until the case-flip is investigated |
| short_hello | 29 bytes | 29 bytes | EOS divergence is structural; content match is what matters |
| sliding_wrap | 700 bytes | 700 bytes | ADR-010-deferred; lower parity expected on this longer prompt |

**Divergence interpretation guide:**

| Common prefix | Verdict |
|---|---|
| ≥ 3094 bytes on sourdough (~830 tokens) | Basically identical — PASS |
| 100–3094 bytes | Partial coherence — investigate before declaring Phase 2 closed |
| < 100 bytes (~26 tokens) | Fundamentally wrong — a forward-pass regression |
| < 10 bytes (~3 tokens) | Catastrophic regression — attention, RoPE, or norm is broken |

The 100-token (~380-byte) boundary is the "coherent" floor. The 10-token
boundary is the "garbage output" floor.

### 3.2 Identical-input requirement

For byte-level text comparison to be meaningful, both tools must see
**byte-identical token sequences**. This is non-trivial because:

1. **BOS handling differs.** hf2q applies the Gemma chat template internally
   (prepends literal `<bos>`). llama-completion requires the template to be
   pre-rendered and the leading `<bos>` stripped before passing `--file`.
   Source: `scripts/sourdough_gate.sh:119-137`.

2. **Both tools must load the same GGUF file.** The model SHA-256 is pinned
   in `tests/evals/reference/MANIFEST.json` and must match.

3. **Sampling must be deterministic.** Both tools use `--temp 0` (greedy).
   llama-completion is also passed `--seed 42` (`sourdough_gate.sh:144`).
   hf2q greedy mode is fully deterministic (no RNG involved at temp=0).

4. **Chat template.** hf2q reads the template from GGUF metadata.
   llama-completion uses `-no-cnv` (no conversation mode) and receives the
   already-templated prompt string — not the raw user text.
   Source: `sourdough_gate.sh:141-146`.

5. **No repetition penalty, no top-p/k filtering.** Both tools run greedy
   (temp=0 implies top-1 sampling). hf2q default `--repetition-penalty 1.0`.
   llama-completion does not apply rep-penalty by default.

**Enforcement procedure (identical to `sourdough_gate.sh:119-137`):**

```bash
# Step 1: render the chat template via hf2q, save to a temp file
HF2Q_DUMP_RENDERED_PROMPT=/tmp/rendered.txt \
  target/release/hf2q generate --model <gguf> --prompt "$USER_PROMPT" \
    --max-tokens 1 --temperature 0 >/dev/null 2>/dev/null

# Step 2: strip the leading literal "<bos>" for llama-completion
python3 -c "
import sys
data = open('/tmp/rendered.txt', 'rb').read()
assert data.startswith(b'<bos>'), 'BOS missing'
open('/tmp/rendered_nobos.txt', 'wb').write(data[5:])
"

# Step 3: run llama-completion on the BOS-stripped rendered prompt
llama-completion --model <gguf> --file /tmp/rendered_nobos.txt \
  --predict <N> --temp 0 --seed 42 \
  --no-display-prompt -no-cnv -st -ngl 999 </dev/null > /tmp/llama_out.txt

# Step 4: run hf2q on the raw user prompt (applies template internally)
target/release/hf2q generate --model <gguf> --prompt "$USER_PROMPT" \
  --max-tokens <N> --temperature 0 > /tmp/hf2q_out.txt
```

Then strip hf2q's 4-line stdout header before comparing. The header
format is documented at `scripts/sourdough_gate.sh:162-173` and the
stripping regex at `sourdough_gate.sh:188-207`.

### 3.3 Preferred metric: byte common-prefix

The existing gate uses **common byte-prefix length** (`sourdough_gate.sh:209-216`).
This is the right metric for greedy decoding because:

- Byte-identical prefix up to byte K means token-identical up to approximately
  token K/3.8 (average UTF-8 bytes per English token on Gemma 4).
- First divergence point pinpoints exactly where the two tools split.
- Tool-level total byte counts differ legitimately (hf2q may stop at EOS
  before max_tokens; the prefix metric is length-agnostic).

Alternative — token-sequence comparison — is more precise but requires
both tools to expose their token IDs. Neither tool currently does so in a
machine-readable way in their default output mode. Byte-prefix is sufficient
and already battle-tested across hundreds of runs.

---

## 4. Prefill tok/s parity procedure

### 4.1 llama.cpp benchmark command

```bash
/opt/llama.cpp/build/bin/llama-bench \
  --model <gguf> \
  -p <seq_len> \
  -n 0 \
  --flash-attn 0 \
  -r 3 \
  --output csv
```

Flags:
- `-p <seq_len>`: prefill seq length (prompt tokens processed)
- `-n 0`: zero decode tokens (pure prefill measurement)
- `--flash-attn 0`: use non-FA path. Rationale: at pp≥2455 on this MoE
  model, fa=0 beats fa=1 by 4.1% (`docs/ADR-011-phase1-port-source-decision.md
  §3.2`). Since hf2q's new flash-attn prefill path (ADR-011 §Phase 3) will
  be compared against the best available llama.cpp number, and fa=0 is the
  faster peer for this model shape, fa=0 is the conservative (harder) target.
- `-r 3`: 3 repetitions; report median.
- `--output csv`: machine-parseable.

Run at all four canonical shapes: pp=128, pp=512, pp=1024, pp=2455.

The pp=2455 shape uses an actual prompt file (not a synthetic sequence) so
the tokenizer follows the same Unicode/BOS path. Generate a 2455-token
prompt by passing the content of `tests/evals/prompts/prefill_2048.txt`.

### 4.2 hf2q benchmark command

hf2q has no dedicated `bench` subcommand (confirmed from `hf2q --help`).
The prefill measurement is extracted from `hf2q generate` stderr, which
emits a prefill summary line:

```
Batched prefill complete: N tokens in X.X ms (Y.Y tok/s)
```

Source: `scripts/release-check.sh:180-181` — grep pattern for this line.

For the batched-prefill path (the Phase 2 target):

```bash
HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_BATCHED_PREFILL=1 \
  target/release/hf2q generate \
    --model <gguf> \
    --prompt-file tests/evals/prompts/prefill_2048.txt \
    --max-tokens 1 --temperature 0 \
    >/dev/null 2>/tmp/hf2q_prefill.log

grep -oE 'Batched prefill complete: [0-9]+ tokens in [0-9.]+ ms \(([0-9.]+) tok/s\)' \
  /tmp/hf2q_prefill.log | tail -1
```

For pp=128, pp=512, pp=1024: use a truncated prompt or a synthetic prompt of
that many tokens. The prompt-file path is the cleanest approach; alternatively,
use `--prompt` with a known short text and measure the tokens from hf2q's
prefill log line (which reports actual token count, not byte count).

Repeat 3 times and take the median. The `scripts/release-check.sh` single-run
method (`release-check.sh:165-196`) is sufficient for a pass/fail gate but
3-run median removes thermal jitter.

**Note on seq_len < 1025:** At seq_len ≤ 1024 (below the sliding-window cap),
the batched prefill path works today (`docs/spike-gate-a-prefill.md §Addendum
— Finding 2`). At seq_len > 1024 (including pp=2455), the batched prefill
path is currently blocked on the `sdpa_sliding` kernel fix (same doc, Finding 2).
The Phase 2 gate for pp=2455 becomes meaningful only once that blocker resolves.

### 4.3 Pass thresholds

The peer baselines from §1.3 (re-measured 2026-04-17, fa=0, 3-rep median):

| seq_len | llama.cpp fa=0 tok/s | Phase 2 pass floor (−10%) | Note |
|---|---|---|---|
| 128 | ~1000 (fa=0 not separately measured at p128) | 900 | Extrapolated; measure fresh |
| 512 | ~3722 (fa=0 ~ same as fa=1 at short seqs) | 3350 | fa=0/fa=1 gap is small at pp=512 |
| 1024 | ~3605 (fa=0 not measured separately at p1024) | 3245 | Extrapolated |
| 2455 | 3455.88 | **3110** | fa=0 peer from `ADR-011-phase1-port-source-decision.md §3.2` |

The −10% tolerance captures measurement noise (llama-bench stdev at pp=2455
fa=1 is ±5.30 = ±0.16%; the thermal floor from prior gates as in
`release-check.sh:161-163` accounts for the rest) plus a reasonable run-to-run
jitter budget for hf2q on a thermal-loaded M5 Max.

For **Phase 2 "just as fast"**, the target at pp=2455 is ≥ 3110 tok/s. This
is approximately 90% of the fa=0 peer. True ±5% parity would require ≥ 3283
tok/s, which requires the Phase 4/5 SWA tile-skip and D=512 carve-outs from
`ADR-011-phase1-port-source-decision.md §9.5`. Set the Phase 2 floor at −10%
and track toward ±5% in Phase 3–5.

---

## 5. Tolerance and noise analysis

### 5.1 Text: is llama.cpp deterministic across runs?

From `scripts/sourdough_gate.sh`: yes, at `--temp 0 --seed 42` with the same
GGUF, llama-completion produces byte-identical output on repeated runs on
the same hardware. The gate's `--min-prefix 3094` was established from
multiple runs without observation of inter-run drift.

For the Phase 2 gate, `parity_check.sh` already runs each prompt 3× and
requires all 3 to pass (`parity_check.sh:59-80`). This encodes Gate F
(determinism). Any run that produces different output from the locked reference
fails the determinism check.

### 5.2 Prefill: llama-bench noise floor

From `docs/ADR-011-phase1-port-source-decision.md §3.1`:
- pp=512 fa=1: ±1.91 tok/s on 3722 = **±0.05%** — effectively zero noise.
- pp=2455 fa=1: ±5.30 tok/s on 3313 = **±0.16%**.
- pp=128 fa=1: ±34.85 tok/s on 1114 = **±3.1%** — kernel-launch regime.

The llama-bench noise floor for seq_len ≥ 512 is < 0.5%. For seq_len=128
it is ~3%. Set the pass threshold conservatively:

| seq_len | Noise floor | Margin added | Gate threshold |
|---|---|---|---|
| 128 | ~3% | 7% | peer − 10% |
| 512–2455 | < 0.5% | 9.5% | peer − 10% |

The −10% gate is generous enough to survive thermal variation (M5 Max can
throttle ~5% over a 45-second gate sequence per `release-check.sh:159-163`)
while still catching real regressions (the old per-token-prefill path at
94.50 tok/s at pp=2455 is 33× below the floor; any regression that reverts
to per-token prefill would fail by ~33×).

### 5.3 hf2q run-to-run noise

From `release-check.sh:113-133`: the Gate B (decode) gate uses median of 3
runs because thermal jitter on a 1000-token decode causes ~5 tok/s swings.
Apply the same median-of-3 discipline for prefill measurements.

---

## 6. Recipe per canonical prompt

### 6.1 short_hello

- **Prompt file:** `tests/evals/prompts/short_hello.txt` — `Hello, what is 2+2?`
- **Max tokens:** 50
- **Reference outputs:** `tests/evals/reference/short_hello_llama.txt` (46 bytes), `short_hello_hf2q.txt` (36 bytes)
- **Common prefix floor:** 29 bytes (content match; only EOS format differs)
- **How to compare:** `hf2q parity check --model <gguf> --prompt short_hello --min-prefix 29`
- **Pass criterion:** common prefix ≥ 29 bytes. If common prefix = 0, catastrophic regression.

### 6.2 sourdough

- **Prompt file:** `tests/evals/prompts/sourdough.txt` — `Complrehensive instructions for making sourdough bread.`  
  (typo is load-bearing — `sourdough_gate.sh:54-56`)
- **Max tokens:** 1000
- **Reference outputs:** `tests/evals/reference/sourdough_llama.txt` (3658 bytes), `sourdough_hf2q.txt` (3656 bytes)
- **Common prefix floor:** 3094 bytes
- **How to compare:** `hf2q parity check --model <gguf> --prompt sourdough --min-prefix 3094`
- **Pass criterion:** common prefix ≥ 3094 bytes. Known first divergence is at byte 3095 (a single letter case-flip at decode token ~840): `sourdough_gate.sh:17-19`.
- **Note on prefill_2048:** the sourdough prompt is the only text-parity prompt that runs a live llama-completion comparison. The `prefill_2048` prompt is benchmark-only.

### 6.3 sliding_wrap

- **Prompt file:** `tests/evals/prompts/sliding_wrap.txt` — 388-byte essay request covering computing history
- **Max tokens:** 500
- **Reference outputs:** `tests/evals/reference/sliding_wrap_llama.txt` (2327 bytes), `sliding_wrap_hf2q.txt` (2354 bytes)
- **Common prefix floor:** 700 bytes (ADR-010 deferred; lower parity than sourdough is expected)
- **How to compare:** `hf2q parity check --model <gguf> --prompt sliding_wrap --min-prefix 700`
- **Pass criterion:** common prefix ≥ 700 bytes.

### 6.4 prefill_2048 (benchmark only — no text parity check)

- **Prompt file:** `tests/evals/prompts/prefill_2048.txt` — 11,592 bytes (2455 hf2q tokens)
- **Used for:** prefill tok/s measurement only
- **Max tokens:** 1 (prefill dominates; one decode step confirms no crash)
- **Expected hf2q tok/s (Phase 2 pass):** ≥ 3110 tok/s
- **Expected llama.cpp tok/s (peer):** 3455.88 tok/s (fa=0, from `ADR-011-phase1-port-source-decision.md §3.2`)
- **Note:** currently blocked on `sdpa_sliding` fix (`docs/spike-gate-a-prefill.md §Addendum`). This check will produce `ERROR: seq_len=2455 exceeds dense cap=1024` until that blocker resolves.

---

## 7. Gate script

The gate script is `scripts/adr-011-phase2-gate.sh`. Its logic:

1. Run `scripts/parity_check.sh` (all 3 text-parity prompts, 3 runs each).
2. Run llama-bench at pp=128, 512, 1024, 2455 (fa=0, 3 reps each).
3. Run hf2q prefill at pp=128, 512, 1024, 2455 (batched path, 3 reps each).
4. Compare ratios and print summary table.
5. Exit 0 if all pass; exit 2 if any fail.

See `scripts/adr-011-phase2-gate.sh` for the implementation.

---

## 8. Debugging when the gate fails

### 8.1 Text divergence at token K

**Step 1 — Locate the divergence byte.** The gate prints the first 120 chars
after the divergence point for both tools (same as `sourdough_gate.sh:234-237`).

**Step 2 — Identify which forward-pass component diverged.** Divergence at
token K means the K-th logit distribution differed. Candidates:
- Attention output (SDPA, RoPE, KV cache)
- MoE dispatch (expert selection or blending)
- Layer norms
- `lm_head` projection

**Step 3 — Compare logit distributions.** Neither tool currently has a
production `--log-logits` mode. Options:

- **hf2q side:** Add `HF2Q_DUMP_LOGITS=1` (not yet implemented) to emit the
  top-K logits per step to stderr. Check whether this is in the roadmap for
  Phase 2 tooling.
- **llama.cpp side:** `llama-cli` has `--logits-all` which dumps the full
  vocabulary logit vector for every output position. This is usable but
  voluminous at vocab=256K for Gemma 4.
- **Intermediate approach:** run both tools with `--max-tokens K` (to the
  divergence point) and compare the final hidden state. Requires adding a
  `HF2Q_DUMP_HIDDEN` env var analogous to the existing
  `HF2Q_DUMP_RENDERED_PROMPT` (`sourdough_gate.sh:120-121`).

**Step 4 — Bisect.** If the divergence is at a new token K' < K_prev (earlier
than the prior clean baseline), use `git bisect` over the recent src/ changes
touching: attention/SDPA, MoE, norms, RoPE, lm_head, KV cache, SDPA. The
gate's exit code 2 is suitable as the bisect test.

### 8.2 Prefill tok/s miss

If hf2q prefill tok/s is below the gate floor:

1. **Confirm batched prefill path is active.** The stderr log should contain
   `Batched prefill complete: ...` — if absent, the env vars
   `HF2Q_BATCHED_PREFILL=1 HF2Q_UNSAFE_EXPERIMENTS=1` were not set, or the
   path fell back to per-token prefill.

2. **Per-op profiling.** Use Metal Frame Capture (Xcode Instruments → GPU
   Counters) or `HF2Q_DUMP_COUNTERS=1` to measure dispatch and sync counts
   per forward pass. The Gate G thresholds in `release-check.sh:205-257`
   establish the expected counter envelope (≤ 1300 dispatches/decode_tok,
   ≤ 60 total syncs).

3. **Check the bottleneck layer.** At pp=2455, the 34× gap vs llama.cpp
   (when running per-token prefill) is structural: hf2q runs 2455 sequential
   forward passes vs llama.cpp's 1 batched forward pass. After Phase 2 lands
   the batched-prefill path, a remaining gap at pp=2455 means the
   `sdpa_sliding` kernel or the fused-norm-rope kernel is underperforming.
   Per `docs/spike-gate-a-prefill.md §Why per-token prefill is decode-speed`:
   batched prefill at ≤1024 tokens already runs at 2.85× per-token speed
   (283 vs 99 tok/s at pp=576); the same speedup should apply at pp=2455
   once ring-wrap lands.

4. **SWA tile-skip gap.** If hf2q is within 15% of the peer after Phase 3,
   the remaining gap is likely the missing `flash_attn_ext_blk` SWA tile-skip
   pre-pass (Phase 5 carve-out from `ADR-011-phase1-port-source-decision.md §5.3`).
   The expected gain is ~15% on this model (50% SWA layers × 2.4× per-tile
   speedup at window=1024, seq=2455).

---

## 9. Actionable checklist

- [ ] **Run `scripts/adr-011-phase2-gate.sh <gguf>`** end-to-end on a clean HEAD
      (after `cargo build --release --features metal`). Record baseline.
- [ ] **Capture fresh reference outputs if needed:**
      `hf2q parity capture --model <gguf> --prompt all`
      (overwrites `tests/evals/reference/*_hf2q.txt` — commit the result).
- [ ] **Unblock pp=2455 batched prefill.** Fix `sdpa_sliding` in mlx-native
      (`docs/spike-gate-a-prefill.md §Addendum, Finding 2`). Then re-run
      Gate 2 (prefill pp=2455).
- [ ] **Wire gate into CI** (optional for Phase 2, required for Phase 3 close).
      Add a `make gate-phase2` target that runs `scripts/adr-011-phase2-gate.sh`.

---

## 10. Self-contained run instructions

Prerequisites:
1. `cargo build --release --features metal` — builds `target/release/hf2q`
2. `llama-completion` and `llama-bench` present at `/opt/llama.cpp/build/bin/`
   (confirmed at `scripts/sourdough_gate.sh:95-101`)
3. GGUF at the path you pass as `$1`

```bash
cd /opt/hf2q
scripts/adr-011-phase2-gate.sh \
  /opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf
```

Exit 0 = all pass. Exit 2 = gate failed (output contains which check failed
and the measured values). Exit 3 = a tool crashed.

---

## 11. References

| File | What it establishes |
|---|---|
| `scripts/sourdough_gate.sh` | BOS-strip protocol, llama-completion flags, byte-prefix comparison algorithm |
| `scripts/parity_check.sh` | Multi-run (Gate F) parity check driver |
| `scripts/release-check.sh` | Perf gate structure (median of 3), counter gate thresholds |
| `tests/evals/reference/MANIFEST.json` | Locked reference byte counts, generation settings, llama.cpp commit |
| `tests/evals/README.md` | Parity check CLI usage |
| `docs/ADR-011-phase1-port-source-decision.md §3` | Re-measured llama.cpp peer baselines (2026-04-17) |
| `docs/spike-gate-a-prefill.md` | Per-token vs batched prefill measurements; sdpa_sliding blocker |
| `docs/ADR-011-flash-attn-prefill.md` | Phase 2–5 plan, success criteria |
