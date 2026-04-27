# Converting a Model with hf2q

Generic reference for the `hf2q convert` command. Covers all supported model
classes, every quantization variant in the ADR-014 P8 menu, both output
formats (GGUF + safetensors), and the user-facing escape-hatch env vars
landed under ADR-014 P11-prereq.

For model-specific details:
- Qwen3.5 / Qwen3.6: see `docs/converting-qwen35.md`
- Gemma-4: canonical command is documented in this file (see below)

---

## Quick start

```bash
hf2q convert \
  --repo google/gemma-4-26b-it \
  --format gguf \
  --quant q4_k_m \
  --output models/gemma-4-26b-it-q4_k_m/out.gguf
```

Expected output: a `.gguf` file at the chosen path plus the HF tokenizer +
config sidecars copied alongside it (see "Sidecar files" below).

```bash
hf2q convert \
  --input /path/to/local/safetensors/dir \
  --format safetensors \
  --quant dwq-4-6 \
  --output models/gemma-4-26b-it-dwq46/
```

For the safetensors format the `--output` path is treated as a **directory**
(mlx-lm convention — see "Output options" below).

---

## Synopsis

```bash
hf2q convert [OPTIONS]
```

Exactly one of `--input` or `--repo` is required.

---

## Input options

### `--input <PATH>`

Convert a model that has already been downloaded to a local directory.
The directory must contain `config.json` and at least one `.safetensors`
file.

```bash
hf2q convert \
  --input /path/to/model \
  --format gguf \
  --quant q4_k_m \
  --output /path/to/output
```

### `--repo <REPO_ID>`

Download from HuggingFace Hub and convert in one step. Requires a valid
HF token for gated models (see Authentication below). `--input` and
`--repo` are mutually exclusive.

---

## Format options (`--format`)

| Value | Description |
|---|---|
| `gguf` | GGUF binary (default consumer format; loads in llama.cpp, LM Studio, Ollama). Single-file emit. |
| `safetensors` | mlx-lm-style directory layout (loads in mlx-lm, Candle, vLLM, hf2q's own serve loader). |

The choice of `--format` selects the **container** format. The `--quant`
choice (next section) selects the per-tensor codec. The two axes are
orthogonal: every quant variant works under both formats subject to the
loader-side restrictions documented under "Safetensors directory layout"
below.

---

## Quantization options (`--quant`)

The 17-variant Decision-12 menu (ADR-014 P8). The menu spans the diagonal
of the (Calibrator × OutputFormat) matrix; off-diagonal cells are reachable
via the orthogonal `--calibration` + `--output-format` flags (see "For
maintainers" at the bottom).

### Passthrough float (no quantization)

| Variant | Calibrator | Codec | When to use |
|---|---|---|---|
| `f16` | none | flat float-16 | Half-precision passthrough; reference for KL/PPL diffs against quants. |
| `bf16` | none | flat bfloat-16 | Preserves bf16 source dtype byte-identically (no f16 cast loss). |

`--quant f16 --format safetensors` and `--quant bf16 --format safetensors`
emit a **single** `model.safetensors` (Decision 17 byte-identity gate;
not a sharded directory).

### Legacy flat block formats (uncalibrated)

| Variant | Codec | When to use |
|---|---|---|
| `q2` | Q2_K block format (~2 bpw) | Aggressive 2-bit; quality cliff is real, use only for size-bound deployments. |
| `q4` | Q4_0 block format (~4 bpw) | Legacy 4-bit; community-standard for the smoke harness (`hf2q smoke ... --quant q4_0`). |
| `q8` | Q8_0 block format (~8 bpw) | High-quality 8-bit; smallest perceptible quality drop vs `f16`. |

`q4` accepts the alias `q4_0`; `q8` accepts `q8_0` (clap-level, not a
display rename — both names produce bit-identical output).

### Uncalibrated K-quant (community-standard `_M` upgrades)

| Variant | Base codec | `_M` upgrades |
|---|---|---|
| `q4_k_m` | Q4_K | `output.weight` and `token_embd.weight` → Q6_K; `attn_v` and `ffn_down` on `use_more_bits` layers → Q6_K. |
| `q5_k_m` | Q5_K | Same `_M` policy. |
| `q6_k` | Q6_K | Higher-precision flat K-quant; no `_M` policy needed (already Q6_K everywhere). |

Default modern starting point: `q4_k_m`. Better quality than `q4` at the
same bpw; mature codebook search (`make_qkx2_quants`) ported from
`ggml-quants.c:1395`.

### Imatrix-calibrated K-quant (llama.cpp PR #4861 style)

| Variant | Calibrator | Codec | Notes |
|---|---|---|---|
| `imatrix-q4_k_m` | imatrix | Q4_K (`_M`) | Per-column importance-weighted codebook search. Quality matches llama.cpp's `--imatrix` path (verified via `cross_validate_imatrix_gguf` against `llama-imatrix`, ADR-014 P6 close iter-1; abs ≤ 1e-3, rel ≤ 1e-2). |
| `imatrix-q5_k_m` | imatrix | Q5_K (`_M`) | Same imatrix policy at 5-bit base. |
| `imatrix-q6_k` | imatrix | Q6_K | Imatrix-weighted Q6_K. |
| `imatrix-adaptive` | imatrix | per-tensor target | Per-tensor optimal precision via `VariantKQuantizer` + `layer_mix::target_for`. Replaces the deleted `apex` variant (Decision 13); preserves per-tensor optimal-precision behaviour. |

All imatrix variants require a forward pass to capture activations. The
calibrator uses the model's own arch-specific forward driver (qwen35 /
qwen35moe via ADR-013's `RealActivationCapture`; Gemma-4 via
`gemma4/forward_cpu.rs`). Architectures whose forward driver is not
present surface a typed `CalibrationError::ForwardPassUnavailable` error
— **no silent fallback to NoneCalibrator**.

### DWQ-calibrated K-quant (ADR-014 P11-prereq Iter C)

DWQ — Apple/MLX's distilled weight quantization — uses activation-derived
per-layer sensitivity to split tensors into a "base" bucket (most layers)
and a "sensitive" bucket (a handful of layers where higher precision pays
off most). Every DWQ variant emits **Q4_K_M-family GGUFs by default** as
of P11-prereq Iter C (commit `975a67a`):

| Variant | Base target | Sensitive target | Default emit |
|---|---|---|---|
| `dwq-4-6` | Q4_K | Q6_K | Q4_K_M base + Q6_K sensitive |
| `dwq-4-8` | Q4_K | Q8_0 | Q4_K_M base + Q8_0 sensitive |
| `dwq-6-8` | Q6_K | Q8_0 | Q6_K base + Q8_0 sensitive |
| `dwq-2-8` | Q2_K (codec pending) | Q8_0 | Sensitive emits as Q8_0; base surfaces a typed `QuantizeError::TensorQuantizeFailed` pointing at the deferred Q2_K codec port — no panics, no silent fallback. |

**Legacy Q4_0-base path**: pre-Iter-C, every DWQ variant emitted Q4_0 for
the base bucket. The legacy path is preserved behind the
`HF2Q_USE_LEGACY_DWQ_Q4_0=1` env var for back-to-back comparison runs;
see "Escape-hatch env vars" below.

DWQ variants on `qwen35` / `qwen35moe` require the ADR-013
`RealActivationCapture` forward driver. Architectures without a forward
driver surface the same `CalibrationError::ForwardPassUnavailable` as
imatrix variants — no weight-space fallback (per
`feedback_never_ship_fallback_without_rootcause.md`).

### Auto

| Variant | Behaviour |
|---|---|
| `auto` | AutoResolver Decision-18 routing table picks a concrete variant from {dense, MoE} × {hardware bandwidth, free memory}. The resolver substitutes a real variant into `ConvertConfig.quant` before downstream dispatch. |

---

## Calibration corpus options

For variants that require calibration (`imatrix-*` and `dwq-*`), the
following knobs control the calibration corpus:

| Flag | Default | Effect |
|---|---|---|
| `--calibration-samples <N>` | `1024` | Number of synthetic calibration tokens fed through the forward pass. Higher = better calibration accuracy at higher convert wall-clock cost. |
| `--sensitive-layers <RANGE>` | (auto) | Override the activation-derived sensitive-layer set with a literal range list (e.g. `"13-24"` or `"1,5,13-24"`). Use only for A/B experiments — the auto-derived set is the production path. |

The synthetic corpus is built by `build_calibration_corpus` in
`src/main.rs` (deterministic ramp drawn from `[0, vocab_size)`). The
real wikitext-2 corpus used by P10's peer-parity harness is fetched
separately by `scripts/fetch_wikitext2.sh` and consumed by
`tests/peer_parity_gates.rs`; per-convert calibration uses the synthetic
ramp because activation magnitudes (the actual sensitivity signal) are
similar across token distributions and the cache key (`corpus_sha`)
already pins token-level reproducibility.

DWQ variants additionally consult the **sensitivity cache** at
`${XDG_CACHE_HOME:-$HOME/.cache}/hf2q/sensitivity/<sha>.json` — the
second `--quant dwq-4-8` invocation on the same model + corpus reuses
the `--quant dwq-4-6` run's per-layer sensitivity scores and skips the
forward pass entirely (P5).

---

## Output options

### `--output <PATH>`

For `--format gguf`:
- If the path ends in `.gguf`, the file is written there directly.
- Otherwise the path is treated as a directory; the GGUF file is written
  inside it with an auto-generated name.

For `--format safetensors`:
- Quantized variants (every variant except `f16` / `bf16`) emit a
  **directory** at `<output>/`. The directory is created if missing.
- `--quant f16` / `--quant bf16` emit a **single** `model.safetensors`
  file inside `<output>/` (Decision 17 byte-identity).

### `--shard-size-gb <FLOAT>`

Target shard size in GB for the safetensors directory layout. **Default
5.0 GB** (mlx-lm / HuggingFace community convention). **Range
0.5..=50.0**; out-of-range or non-finite values are rejected at clap
parse time. Ignored for `--format gguf` and for the single-file
`--quant f16` / `--quant bf16` safetensors path.

```bash
hf2q convert \
  --input /path/to/model \
  --format safetensors \
  --quant dwq-4-6 \
  --shard-size-gb 2.0 \
  --output models/dwq46-2gb-shards/
```

### Safetensors directory layout

When `--format safetensors` is used with a quantized variant, the output
directory contains:

```text
<output>/
  ├── config.json                              # injected per mlx-lm save_config schema
  ├── tokenizer.json + tokenizer_config.json   # copied from HF source (sidecars)
  ├── chat_template.jinja                      # copied (when present)
  ├── generation_config.json                   # copied (when present)
  ├── special_tokens_map.json                  # copied (when present)
  ├── quantization_config.json                 # legacy hf2q sidecar (kept for back-compat)
  ├── model.safetensors                        # single-shard case (≤ shard_size_gb)
  │   OR
  ├── model-NNNNN-of-MMMMM.safetensors         # multi-shard (5-digit zero-padded names)
  └── model.safetensors.index.json             # only when sharded (MMMMM > 1)
```

Per-tensor schema:

- **DWQ variants** (`dwq-4-6` / `dwq-4-8` / `dwq-6-8` / `dwq-2-8`): each
  weight tensor `<name>.weight` lives in the U8 slot as packed bits;
  companion `<name>.scales` (F16) and `<name>.biases` (F16) tensors
  carry the unpacked codebook per the mlx-lm convention
  (`mlx_lm.utils:154-155`). `bits` and `group_size` live in the
  top-level `config.json#quantization` block.
- **K-quant variants** (`q4_k_m` / `q5_k_m` / `q6_k` / `imatrix-*`):
  each weight tensor `<name>.weight` lives in the U8 slot as **opaque
  GGUF block bytes**. No companion tensors — scales pack inline. Per-
  shard `__metadata__` carries a `quant_method` discriminator (e.g.
  `k_quant_q4_k_m`) so a downstream loader can route to the right
  dequantizer. mlx-lm cannot load these natively (it would TypeError on
  the U8 dtype mismatch); hf2q's own serve loader is the consumer.

---

## Sidecar files

After writing the `.gguf` (or after writing the safetensors directory),
hf2q copies these files from the HF source directory into the output
directory byte-identically:

```
chat_template.jinja
tokenizer.json
tokenizer_config.json
config.json
generation_config.json
special_tokens_map.json
```

Files missing from the source are silently skipped. For `--format
safetensors`, the backend-injected `config.json` (with the mlx-lm
`quantization` block) takes precedence — the sidecar copier skips files
already present at the destination so the injected config is not
clobbered.

---

## Authentication

For gated or private models:

```bash
# Environment variable (recommended for scripting)
export HF_TOKEN=hf_xxxx

# Or write the token to the standard path
echo hf_xxxx > ~/.huggingface/token
chmod 600 ~/.huggingface/token
```

Token resolution order (mirrors hf-hub):
1. `HF_TOKEN` env var
2. `HUGGING_FACE_HUB_TOKEN` env var (legacy)
3. `~/.cache/huggingface/token`
4. `~/.huggingface/token`

---

## Disk preflight

hf2q checks available disk space before starting any download (Decision 14):

| Model class | Minimum free |
|---|---|
| Qwen3.5-MoE 35B | 150 GB |
| Qwen3.5 27B dense | 55 GB |
| Other models (Gemma-4, Llama, etc.) | 100 GB |

Override the cache directory to use a different disk:

```bash
export HF_HUB_CACHE=/mnt/large-disk/hf-cache
```

---

## Other options

| Flag | Description |
|---|---|
| `--skip-quality` | Skip KL divergence / perplexity measurement (faster; use for CI). |
| `--quality-gate` | Fail with exit code 2 if quality thresholds are exceeded. |
| `--dry-run` | Run preflight + auto resolution, print plan, exit without converting. |
| `--json-report` | Emit a structured JSON report for CI / automation. |
| `--bits <2..=8>` | Override the default bit width (advanced; conflicts with most calibrated variants). |
| `--group-size <32\|64\|128>` | Override the quantization group size (default 64). |
| `--target-bpw <FLOAT>` | Target average bits/weight for adaptive paths (default 4.5). |
| `--unsupported-layers passthrough` | Pass unsupported layer types through at f16 instead of failing. |
| `--emit-vision-tower` | ADR-012 P10 — also emit `mmproj-<slug>-F16.gguf` when the HF repo has a `vision_config`. Silently skipped when no `vision_config` is present (Gemma-4, Qwen3.6-35B-A3B MoE). |
| `--no-integrity` | Skip post-download per-shard SHA-256 integrity verification against HF's `x-linked-etag`. **NOT recommended** — corruption + MITM + silent force-push will not be detected. |
| `--yes` | Non-interactive mode — skip confirmation prompts. |

---

## Escape-hatch env vars (user-facing)

These are documented and stable; they will not be removed without an ADR.
The full classification matrix lives in `docs/shipping-contract.md`; the
two that affect the convert pipeline are:

### `HF2Q_USE_LEGACY_DWQ_Q4_0=1`

Restore the pre-P11-prereq DWQ emit path that produces Q4_0-base GGUFs
instead of the modern Q4_K_M-family default. Use only for back-to-back
PPL/quality comparison runs against pre-Iter-C artefacts. The default
path (Q4_K_M family) is the production target; the legacy escape exists
because rolling-back a kernel-level emit path requires a real env var,
not a "did the convert behave like last week?" guess.

```bash
HF2Q_USE_LEGACY_DWQ_Q4_0=1 \
  hf2q convert --repo X --format gguf --quant dwq-4-6 --output legacy.gguf
```

### `HF2Q_UNSAFE_EXPERIMENTS=1`

Required acknowledgment for the orthogonal `--calibration X
--output-format Y` selector when the requested cell is **off-diagonal**
(not one of the 17 `--quant` variants). Diagonal cells (e.g.
`--calibration imatrix --output-format k-quant-q4_k_m`, equivalent to
`--quant imatrix-q4_k_m`) are accepted unconditionally; off-diagonal
cells (e.g. `--calibration imatrix --output-format bit-pair-4-6`)
require the env gate to surface accidental misconfigurations.

---

## Common errors

The convert pipeline fails fast with typed, user-actionable errors. The
most common surfaces:

- **`No config.json found in <dir>. Is this a HuggingFace model directory?`**
  — `--input` points at a directory missing `config.json`. Verify the
  path; HF repos always include this file at the snapshot root.

- **`calibration: forward-pass infrastructure unavailable for arch '<arch>'`**
  (`CalibrationError::ForwardPassUnavailable`) — `--quant imatrix-*` or
  `--quant dwq-*` was selected for an architecture without a forward
  driver. There is **no weight-space fallback**; pick a NoneCalibrator
  variant (`q4_k_m` / `q5_k_m` / `q6_k`) or implement the forward
  driver for the target arch.

- **`calibration: corpus is empty`** (`CalibrationError::EmptyCorpus`) —
  `--calibration-samples 0` or an internal corpus generation bug.
  Re-run with the default sample count.

- **`Insufficient free disk: <model class> requires ≥<N> GB free in
  <path>; found <M> GB. Free space or change --cache-dir.`** — disk
  preflight failure; either free space on the cache disk or set
  `HF_HUB_CACHE` to a larger volume.

- **`--quant apex was removed in ADR-014 P8 (Decision 13). Use --quant
  imatrix-adaptive ...`** (and the `mixed-N-M` / `dwq-mixed-N-M`
  family) — pre-P8 variant names with no aliases. The error message
  names the modern equivalent; rename the script and re-run. Mapping
  table is at `cli::map_deleted_quant_hint`.

- **`--calibration X --output-format Y is not a validated cell ... See
  docs/converting-a-model.md §maintainers, or use one of the 17 --quant
  variants instead.`** — off-diagonal orthogonal selector without
  `HF2Q_UNSAFE_EXPERIMENTS=1`.

- **`tensor quantize failed: <name>: Q2_K codec port pending — see
  src/quantize/dwq_k_quantizer.rs ::P28`** — `--quant dwq-2-8` on
  base-target tensors. The Q2_K codec port is queued for a future iter;
  sensitive-target tensors still quantize correctly under `dwq-2-8`.

---

## Gemma-4 26B — canonical command

```bash
hf2q convert \
  --repo google/gemma-4-26b-it \
  --format gguf \
  --quant dwq-4-6 \
  --output models/gemma-4-26b-it-dwq46/out.gguf
```

**Expected output size:** 12–14 GB.

Smoke test:

```bash
llama-cli --model models/gemma-4-26b-it-dwq46/out.gguf -p "Hello" -n 8
```

---

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Conversion error |
| 2 | Quality threshold exceeded (`--quality-gate`) |
| 3 | Input / validation error |

---

## For maintainers — orthogonal `--calibration` × `--output-format`

The 17-variant `--quant` menu is the diagonal of a (Calibrator ×
OutputFormat) matrix. The two axes are also exposed as separate flags
for future-proofing; the diagonal cells reachable via `--quant` are:

| `--quant` variant | `--calibration` | `--output-format` |
|---|---|---|
| `f16` | `none` | `flat-f16` |
| `bf16` | `none` | `flat-bf16` |
| `q2` | `none` | `flat-q2` |
| `q4` | `none` | `flat-q4` |
| `q8` | `none` | `flat-q8` |
| `q4_k_m` | `none` | `k-quant-q4_k_m` |
| `q5_k_m` | `none` | `k-quant-q5_k_m` |
| `q6_k` | `none` | `k-quant-q6_k` |
| `imatrix-q4_k_m` | `imatrix` | `k-quant-q4_k_m` |
| `imatrix-q5_k_m` | `imatrix` | `k-quant-q5_k_m` |
| `imatrix-q6_k` | `imatrix` | `k-quant-q6_k` |
| `imatrix-adaptive` | `imatrix` | `k-quant-adaptive` |
| `dwq-4-6` | `dwq` | `bit-pair-4-6` |
| `dwq-4-8` | `dwq` | `bit-pair-4-8` |
| `dwq-6-8` | `dwq` | `bit-pair-6-8` |
| `dwq-2-8` | `dwq` | `bit-pair-2-8` |

Off-diagonal cells (e.g. `--calibration imatrix --output-format
bit-pair-4-6`) are accepted only when `HF2Q_UNSAFE_EXPERIMENTS=1`. The
off-diagonal pair is logged as a tracing breadcrumb; the live dispatch
through these cells lands in a future iter (currently the
`--quant`-equivalent diagonal cell is reachable via `--quant`).

To add a new Calibrator implementation (Imatrix, DWQ, future ones), see
`docs/calibrator-onboarding.md`.

---

## References

- `docs/converting-qwen35.md` — Qwen3.5 / Qwen3.6 specific guide.
- `docs/calibrator-onboarding.md` — developer guide for adding new
  Calibrator implementations.
- `docs/shipping-contract.md` — product contract, env-var classification,
  peer-parity gates.
- `docs/operator-env-vars.md` — complete env var reference.
- `docs/ADR-014-streaming-convert-pipeline.md` — ADR with the full
  Decision-12 menu rationale + Decision-15 peer-parity gate matrix.
