# Converting a Model with hf2q

Generic reference for the `hf2q convert` command. Covers all supported model
classes and quantization schemes.

For model-specific details:
- Qwen3.5 / Qwen3.6: see `docs/converting-qwen35.md`
- Gemma-4: canonical command is documented in this file (see below)

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
The directory must contain `config.json` and at least one `.safetensors` file.

```bash
hf2q convert \
  --input /path/to/model \
  --format gguf \
  --quant q4 \
  --output /path/to/output
```

### `--repo <REPO_ID>`

Download from HuggingFace Hub and convert in one step.
Requires a valid HF token for gated models (see Authentication below).

```bash
hf2q convert \
  --repo google/gemma-4-26b-it \
  --format gguf \
  --quant dwq-mixed-4-6 \
  --output models/gemma-4-26b-it-dwq46/out.gguf
```

---

## Format options (`--format`)

| Value | Description |
|---|---|
| `gguf` | GGUF binary (default; compatible with llama.cpp, LM Studio, Ollama) |
| `safetensors` | HuggingFace safetensors format with quantization config |

---

## Quantization options (`--quant`)

| Value | Description | Output bits/weight |
|---|---|---|
| `q4` | Static 4-bit | ~4 bpw |
| `q8` | Static 8-bit | ~8 bpw |
| `f16` | Half-precision (no quantization) | 16 bpw |
| `q2` | Static 2-bit (aggressive) | ~2 bpw |
| `mixed-4-6` | Per-layer 4-bit base, 6-bit sensitive | ~4–6 bpw |
| `mixed-3-6` | Per-layer 3-bit base, 6-bit sensitive | ~3–6 bpw |
| `mixed-2-6` | Per-layer 2-bit base, 6-bit sensitive | ~2–6 bpw |
| `dwq-mixed-4-6` | DWQ 4-bit base, 6-bit sensitive | ~4–6 bpw |
| `dwq-mixed-4-8` | DWQ 4-bit base, 8-bit sensitive | ~4–8 bpw |
| `dwq-mixed-6-8` | DWQ 6-bit base, 8-bit sensitive | ~6–8 bpw |
| `dwq-mixed-2-8` | DWQ 2-bit base, 8-bit sensitive | ~2–8 bpw |
| `auto` | hf2q selects based on hardware and model fingerprint | varies |
| `apex` | Adaptive per-layer allocation targeting a bits/weight budget | varies |

DWQ variants use Dynamic Weight Quantization (activation-aware sensitivity
scoring). They require an `ActivationCapture` implementation from the inference
session for qwen35/qwen35moe (see ADR-012 Decision 13).

---

## Output options

### `--output <PATH>`

If the path ends in `.gguf`, the file is written there directly.
If the path is a directory (or does not end in `.gguf`), the directory is
created and the GGUF file is written inside it with an auto-generated name.

If `--output` is omitted with `--quant auto`, the output path is derived from
the input model name and selected quantization.

---

## Sidecar files

After writing the `.gguf`, hf2q copies these files from the HF source
directory into the output directory byte-identically (Decision 15):

```
chat_template.jinja
tokenizer.json
tokenizer_config.json
config.json
generation_config.json
special_tokens_map.json
```

Files missing from the source are silently skipped.

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
| `--skip-quality` | Skip quality measurement (faster; use for CI) |
| `--dry-run` | Print the conversion plan without writing any files |
| `--json-report` | Write a JSON report with quality metrics |
| `--quality-gate` | Fail with exit code 2 if quality thresholds are exceeded |
| `--sensitive-layers <RANGE>` | Layer indices to promote to higher bit width |
| `--calibration-samples <N>` | Number of calibration samples for DWQ |
| `--group-size <N>` | Quantization group size (default: 64) |
| `--yes` | Write JSON report to stdout instead of a file |

---

## Gemma-4 26B — canonical command

```bash
hf2q convert \
  --repo google/gemma-4-26b-it \
  --format gguf \
  --quant dwq-mixed-4-6 \
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

## References

- `docs/converting-qwen35.md` — Qwen3.5 / Qwen3.6 specific guide.
- `docs/shipping-contract.md` — product contract and accepted model classes.
- `docs/operator-env-vars.md` — complete env var reference.
