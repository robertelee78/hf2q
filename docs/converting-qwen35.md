# Converting Qwen3.5 / Qwen3.6 Models with hf2q

Canonical reference for converting the Qwen3.5 / Qwen3.6 model family to DWQ GGUF.
Two variants are covered: the 35B MoE and the 27B dense.

See `docs/ADR-012-qwen35moe-conversion.md` for the full architectural rationale.

---

## Prerequisites

- `hf2q` built from this repo (`cargo build --release --bin hf2q`).
- An HuggingFace token with access to the target repo, either via
  `HF_TOKEN=hf_xxxx` env var or `~/.huggingface/token`.
- Sufficient free disk space (see below).

---

## Disk preflight

hf2q checks available disk space before starting any download. If the check
fails, the download is aborted immediately with a user-actionable error.

| Model | Minimum free disk |
|---|---|
| Qwen3.5-MoE 35B | 150 GB |
| Qwen3.5 27B dense | 55 GB |
| Other models | 100 GB |

The check uses `--cache-dir` (or `HF_HUB_CACHE` / `~/.cache/huggingface/hub`)
as the target path. To use a different disk, set:

```bash
export HF_HUB_CACHE=/path/on/large/disk
```

Example error when disk space is insufficient:

```
Error: Qwen3.5-MoE 35B requires ≥150 GB free in /home/user/.cache/huggingface/hub; found 80 GB. Free space or change --cache-dir.
```

---

## Shard resumption

hf-hub skips shards that have already been fully downloaded. If a download is
interrupted (e.g. `Ctrl+C`), re-invoking `hf2q` with the same `--repo` will
re-fetch only the in-flight shard and resume from there. Completed shards are
not re-downloaded.

Manual verification: after `Ctrl+C`, inspect
`~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<hash>/` — fully
downloaded shards are present; the interrupted shard may be partial. On
re-invoke, only the partial shard is re-fetched.

---

## Sidecar files

After writing the `.gguf`, hf2q copies these files from the HF source
directory into the output directory alongside the model file, byte-identical:

- `chat_template.jinja`
- `tokenizer.json`
- `tokenizer_config.json`
- `config.json`
- `generation_config.json`
- `special_tokens_map.json`

Files missing from the source are silently skipped. The sidecar set is the
same for all model classes (Gemma, Qwen3.5-MoE, Qwen3.5 dense).

---

## Converting Qwen3.6-35B-A3B (MoE variant)

```bash
hf2q convert \
  --repo jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated \
  --format gguf \
  --quant dwq-mixed-4-6 \
  --output models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/out.gguf
```

**Expected output size:** 18–22 GB (DWQ-4-6 of a 35B MoE; expert-merged).

Alternative with 4-bit base / 8-bit sensitive layers:

```bash
hf2q convert \
  --repo jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated \
  --format gguf \
  --quant dwq-mixed-4-8 \
  --output models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48/out.gguf
```

**Expected output size:** 22–28 GB (DWQ-4-8; sensitive layers promoted to 8-bit).

---

## Converting Qwen3.5-27B (dense variant)

```bash
hf2q convert \
  --repo <qwen35-dense-repo> \
  --format gguf \
  --quant dwq-mixed-4-8 \
  --output models/qwen35-27b-dense-dwq48/out-dwq48.gguf
```

Replace `<qwen35-dense-repo>` with the HuggingFace repo ID of the 27B dense
model (e.g. `Qwen/Qwen3.5-27B-Instruct` or a fine-tune).

**Expected output size:** 14–18 GB (DWQ-4-8 of a 27B dense).

---

## Manual smoke test

After conversion, verify that llama.cpp can load the file without errors:

```bash
llama-cli --model models/.../out.gguf -p "Hello" -n 8
```

Expected: llama.cpp prints the model load summary and emits 8 tokens without
error. Inference coherence is out of scope for the convert acceptance contract
(see ADR-013).

---

## Acceptance criteria (shipping contract)

A converted qwen35 / qwen35moe GGUF is accepted when:

1. `.gguf` is structurally valid per hf2q's reader (magic `GGUF`, version 3,
   tensor_count > 0, kv_count > 0).
2. Every metadata key in the ADR-012 catalog is present.
3. Every tensor name follows the ADR-012 P4 naming spec (Decision 8).
4. `llama-cli --model ...out.gguf -p "Hello" -n 8` loads without error.

Inference coherence (sourdough gate, sliding window parity) is delegated to
ADR-013 (Qwen3.5 inference engine).

---

## References

- `docs/ADR-012-qwen35moe-conversion.md` — full decision record.
- `docs/converting-a-model.md` — generic convert-command reference.
- `docs/shipping-contract.md` — overall product contract including qwen35 section.
- `docs/operator-env-vars.md` — HF_TOKEN, HF_HUB_CACHE, and other env vars.
