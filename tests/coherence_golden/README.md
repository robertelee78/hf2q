# Coherence golden outputs

ADR-015 iter41 — captured 2026-04-28 from llama-completion (peer reference) on
M5 Max as the truth-of-the-day for hf2q's coherence parity gates.

## What's stored here

12 files: `<fixture>-<prompt-slug>.txt`, each containing the exact decoded
text llama-completion produced for `(fixture, prompt)` at `--temp 0.0 -n 16`
with `-no-cnv --no-display-prompt`. These files are bytes-as-captured; do NOT
hand-edit. If a model fixture changes (re-quantization, new conversion), the
goldens MUST be re-captured against that exact GGUF.

Empty trailing newlines are preserved. Multi-line outputs (e.g. the
`What is 2+2?` cells) keep their newlines verbatim.

## Fixtures

| Slug          | Path                                                                                                   | Architecture     |
|---------------|--------------------------------------------------------------------------------------------------------|------------------|
| `27b-dwq46`   | `/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf`                                            | Qwen3.6 dense    |
| `dwq46`       | `/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/...-dwq46.gguf`                      | Qwen3.6 MoE 35B  |
| `apex`        | `/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/...-apex.gguf`                        | Qwen3.6 MoE 35B  |
| `gemma`       | `/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/...-dwq.gguf`                                  | Gemma3 26B-A4B   |

## Prompts

| Slug                  | Prompt                |
|-----------------------|-----------------------|
| `hello-my-name-is`    | `Hello, my name is`   |
| `the-quick-brown-fox` | `The quick brown fox` |
| `what-is-22`          | `What is 2+2?`        |

## Re-capture procedure

If a fixture or llama-completion version changes, re-capture as follows:

```sh
cd <repo>/tests/coherence_golden

for FIXTURE in 27b-dwq46 dwq46 apex gemma; do
  case "$FIXTURE" in
    27b-dwq46) M=/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf ;;
    dwq46)     M=/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46.gguf ;;
    apex)      M=/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf ;;
    gemma)     M=/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf ;;
  esac
  for P in "Hello, my name is" "The quick brown fox" "What is 2+2?"; do
    SLUG=$(echo "$P" | tr -cd 'A-Za-z0-9 ' | tr ' ' '-' | tr 'A-Z' 'a-z' | sed 's/--*/-/g')
    /opt/homebrew/bin/llama-completion -m "$M" -p "$P" -n 16 --temp 0.0 \
      -no-cnv --no-display-prompt < /dev/null 2>/dev/null \
      | head -c 1000 > "${FIXTURE}-${SLUG}.txt"
  done
done
```

## What "coherence" means here

The goldens are llama-completion's output, not "ideal" text. Some cells (e.g.
`gemma-the-quick-brown-fox`) show repetitive degenerate patterns — that is
llama's behavior at temp 0 on that fixture/prompt and is the reference hf2q
must match. The contract is **peer parity**, not absolute coherence.

## Pass tiers (cf. `tests/coherence_matrix.rs`)

- **EXACT** — byte-identical to golden. PASS, log `EXACT`.
- **COHERENT** — first 5 tokens of golden share ≥3 with hf2q output AND no
  >3× repetition of any token AND no degenerate-pattern markers. PASS+WARN.
- **GIBBERISH** — none of the above. FAIL with golden vs actual diff.

## When goldens disagree with the smoke heuristics

`tests/coherence_smoke.rs` flags degenerate output unconditionally; if a
golden itself trips those heuristics (gemma `-ing-ing-ing`), the smoke test
documents the cell as `KNOWN_DEGENERATE_PEER` rather than failing. Add the
`(fixture, prompt-slug)` pair to `KNOWN_DEGENERATE_PEER` in
`tests/coherence_smoke.rs` whenever a peer-degenerate golden is captured.

## History

- 2026-04-28 — Initial capture (iter41) on M5 Max. llama.cpp ggml 0.9.11.
