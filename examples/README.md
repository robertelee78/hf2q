# hf2q examples

Runnable end-to-end scenarios.  Each one is self-contained, ~50 LOC,
and copy-paste-ready.  Built and tested on Apple Silicon (M-series)
with macOS 14+.

## Quick start by intent

| What you want | Script | Notes |
|---|---|---|
| HF → GGUF in one command | [`01-convert-gemma4-q4km.sh`](01-convert-gemma4-q4km.sh) | Default convert path with K-quant output |
| Spin up an OpenAI-API server | [`02-serve-and-curl.sh`](02-serve-and-curl.sh) | `hf2q serve` + a `curl` smoke probe |
| Drive it from Python | [`03-openai-python-client.py`](03-openai-python-client.py) | Stock OpenAI SDK works as-is |
| BERT embeddings | [`04-embeddings-bert.sh`](04-embeddings-bert.sh) | `nomic-bert` / `bert-base` via `/v1/embeddings` |
| Qwen3-VL with an image | [`05-qwen-vision-image.sh`](05-qwen-vision-image.sh) | Vision chat completion |

## Prerequisites

- macOS on Apple Silicon (M1 or newer).
- Rust 1.81+ toolchain (`rustup default 1.81.0`).
- `hf2q` built in release: `cargo build --release` at the repo root.
- (Examples that download models) Enough free disk space — see the
  per-arch floors in [`src/arch/entries/`](../src/arch/entries/).

Every script `set -euo pipefail`'s and prints what it's doing.
Read before running.

## Why so few examples?

Each script demonstrates one user goal end-to-end rather than one
flag.  For the full flag surface run `hf2q <subcommand> --help`; for
every supported model architecture see
[`docs/converting-a-model.md`](../docs/converting-a-model.md) and the
per-arch deep-dives like
[`docs/converting-qwen35.md`](../docs/converting-qwen35.md).
