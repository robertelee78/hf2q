# Eval Corpus — ADR-009 Parity Validation

Locked evaluation corpus for coherence-recovery work. See `docs/reference-lock.md`
for pinned commits, model, and deterministic decode settings.

## Directory Structure

```
tests/evals/
├── README.md                          # this file
├── prompts/                           # prompt text files
│   ├── sourdough.txt                  # main coherence gate (22 tokens)
│   ├── short_hello.txt                # fast sanity check
│   └── sliding_wrap.txt               # long prompt for sliding-window
├── reference/                         # locked reference outputs
│   ├── MANIFEST.json                  # generation metadata + parity measurements
│   ├── sourdough_llama.txt            # llama.cpp oracle output (3658 bytes)
│   ├── sourdough_hf2q.txt             # hf2q baseline output (3656 bytes)
│   ├── short_hello_llama.txt          # llama.cpp oracle output (46 bytes)
│   ├── short_hello_hf2q.txt           # hf2q baseline output (36 bytes)
│   ├── sliding_wrap_llama.txt         # llama.cpp oracle output (2327 bytes)
│   └── sliding_wrap_hf2q.txt          # hf2q baseline output (2354 bytes)
└── fixtures/                          # tensor fixtures (Phase 2+)
    └── .gitkeep
```

## Parity Checks

### CLI subcommands

```bash
# Check hf2q output against locked llama.cpp reference
hf2q parity check --model <gguf> --prompt sourdough
hf2q parity check --model <gguf> --prompt sourdough --min-prefix 3094
hf2q parity check --model <gguf> --prompt short_hello --min-prefix 29
hf2q parity check --model <gguf> --prompt sliding_wrap --min-prefix 700

# Capture fresh hf2q output (overwrites reference/X_hf2q.txt)
hf2q parity capture --model <gguf> --prompt all
hf2q parity capture --model <gguf> --prompt sourdough --max-tokens 1000
```

### Validation suite script

```bash
# Run all parity checks with standard thresholds
scripts/parity_check.sh <gguf_path>
```

### Sourdough gate (legacy, still works)

```bash
# Original byte-prefix gate — compares hf2q vs live llama.cpp
scripts/sourdough_gate.sh <gguf_path> --min-prefix 3094
```

## Current Parity Measurements (2026-04-16)

| Prompt | llama.cpp | hf2q | Common prefix | Gate |
|--------|-----------|------|---------------|------|
| sourdough | 3658 bytes | 3656 bytes | **3656** bytes | >= 3094 |
| short_hello | 46 bytes | 36 bytes | **29** bytes | content identical, EOS differs |
| sliding_wrap | 2327 bytes | 2354 bytes | **752** bytes | >= 700 |

## Reference Generation

References are generated from the pinned llama.cpp commit and model
specified in `docs/reference-lock.md`. The MANIFEST.json records the
exact generation settings and commit hashes.

To regenerate llama.cpp references (requires llama-completion):

```bash
# Sourdough (main gate)
scripts/sourdough_gate.sh <gguf> --min-prefix 0
# Output is captured in /tmp/sourdough_gate_llama.txt

# Short hello
llama-completion --model <gguf> --file <rendered_prompt> \
  --predict 50 --temp 0 --seed 42 \
  --no-display-prompt -no-cnv -st -ngl 999
```

## Fixture Format

The `fixtures/` directory is reserved for tensor-level boundary fixtures
(saved Q, K, V, attention logits, sdpa_out tensors). These provide
finer-grained regression detection than text-level comparison.

Fixture population is planned as future work — the current text-level
parity checks (3656/3658 byte match on sourdough) provide strong
correctness evidence.
