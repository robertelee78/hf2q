# Reference Lock — ADR-009

Pinned reference commits, model, and deterministic decode settings for the
coherence-recovery work.  Nothing in this file is a runtime dependency —
these are validation oracles only.

---

## Pinned Commits

| Repo | Commit | Date | Note |
|------|--------|------|------|
| **llama.cpp** (primary semantic oracle) | `b3d758750a268bf93f084ccfa3060fb9a203192a` | 2026-04-15 | `/opt/llama.cpp` |
| **candle** (dense Rust reference) | `c42e1fefa928335660c7877753903e59c87fd7ff` | 2026-04-15 | `/opt/candle` |
| **mlx-native** (owned GPU backend) | `636d5cb87ff3e8eb3a254f24aba5e59da325ec4d` | 2026-04-15 | `/opt/mlx-native` |
| **hf2q** (baseline at ADR-009 start) | `7a9612e6430101a63bf60c89bf55b21099600561` | 2026-04-15 | `/opt/hf2q` |

## Pinned Model

| Field | Value |
|-------|-------|
| **Path** | `models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf` |
| **Size** | 16,922,246,304 bytes (15.8 GB) |
| **MD5** | `c350aa0e317c91f7bc2c5b6d1e456713` |
| **SHA-256** | `ae19574dab588e0a742a8cfa1282ccf358ed8a58c5a3483f2bf596020a4f8e6f` |
| **Architecture** | Gemma 4 26B A4B (MoE, ISWA) |
| **Quantization** | DWQ (dynamic weight quantization) |

## Deterministic Decode Settings

All coherence comparisons must use these exact settings:

| Setting | Value |
|---------|-------|
| **Temperature** | 0 (greedy) |
| **Seed** | 42 (llama.cpp only; hf2q uses greedy argmax) |
| **Max tokens** | 1000 |
| **Top-p** | 1.0 (disabled) |
| **Top-k** | 0 (disabled) |
| **Repetition penalty** | 1.0 (disabled) |
| **Stop conditions** | EOS token IDs: `[1, 106]` |

## Reference Commands

### llama.cpp reference run

```bash
# Pre-render prompt via hf2q, strip BOS for llama-completion
HF2Q_DUMP_RENDERED_PROMPT=/tmp/rendered.txt \
  target/release/hf2q generate --model "$GGUF" --prompt "$PROMPT" --max-tokens 1 --temperature 0

python3 -c "
import sys; d=open('/tmp/rendered.txt','rb').read()
assert d.startswith(b'<bos>'); open('/tmp/rendered_nobos.txt','wb').write(d[5:])
"

/opt/llama.cpp/build/bin/llama-completion \
  --model "$GGUF" --file /tmp/rendered_nobos.txt \
  --predict 1000 --temp 0 --seed 42 \
  --no-display-prompt -no-cnv -st -ngl 999
```

### hf2q reference run

```bash
target/release/hf2q generate \
  --model "$GGUF" --prompt "$PROMPT" \
  --max-tokens 1000 --temperature 0
```

### Sourdough gate

```bash
scripts/sourdough_gate.sh "$GGUF" --min-prefix 3094
```

## Baseline Coherence Measurements

| Path | Sourdough common-prefix bytes | Date |
|------|-------------------------------|------|
| **llama.cpp** (oracle) | 3658 bytes output | 2026-04-11 |
| **hf2q** (old candle path) | 3095 bytes common prefix | 2026-04-11 |
| **hf2q** (current mlx-native) | 69 bytes common prefix | 2026-04-15 |

**Target:** hf2q owned stack matches llama.cpp at >= 3094 bytes common prefix
on the sourdough prompt with the pinned model and settings above.
