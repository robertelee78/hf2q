# hf2q

Pure-Rust CLI for converting HuggingFace models to hardware-optimized
formats — and serving them through an OpenAI-compatible HTTP API on
Apple Silicon.

| | |
|---|---|
| **License** | Apache-2.0 |
| **Rust** | 1.81+ |
| **Inference backend** | [`mlx-native`](https://crates.io/crates/mlx-native) (Apple Metal) — ADR-008 |
| **Output formats** | GGUF (`llama.cpp` consumers), mlx-lm safetensors |
| **Status** | Production usage on M-series Macs; structural performance parity with `llama.cpp` peer at HEAD |

```bash
# Convert a HuggingFace model to a Q4_K_M GGUF
hf2q convert \
  --repo google/gemma-4-26b-it \
  --format gguf \
  --quant q4_k_m \
  --output models/gemma-4-26b-it-q4_k_m/out.gguf

# Serve it over an OpenAI-compatible HTTP API
hf2q serve --model models/gemma-4-26b-it-q4_k_m/out.gguf --port 8080
```

---

## What it does

`hf2q` is two tools fused into one binary:

1. **A conversion pipeline.** Read HuggingFace `config.json` +
   `*.safetensors`, normalize tensor names per architecture, run
   quantization (legacy `Qx_0`, K-quants `Qx_K_{S,M,L}`, DWQ-calibrated
   mixed-bit, or APEX adaptive) and emit GGUF or mlx-lm safetensors.
   No `llama.cpp` or `candle` is involved at build, test or runtime
   (ADR-008 — "candle divorce"; sovereignty rule in
   `docs/arch-onboarding.md`).

2. **An inference + serving engine.** Load a GGUF, run prefill +
   speculative-or-vanilla decode on the GPU via `mlx-native`, expose
   it through an OpenAI-style `/v1/chat/completions`, `/v1/embeddings`
   and `/v1/models` HTTP API. Supports tools / function-calling,
   streaming SSE, vision (`qwen3vl`), grammar-constrained sampling,
   and a persistent block-prefix KV cache.

Supported architectures today: **Gemma 4 (dense + MoE)**, **Qwen 3.5 /
3.6 (dense + MoE + multi-token-prediction)**, **Qwen 3-VL (vision +
text)**, **BERT / Nomic-BERT** (embedding-only). Each lives under a
single `src/inference/models/<arch>/` module — the arch-registry
(`src/arch/`) is the single source of truth for tensor catalogs,
quality thresholds, smoke prompts and MTP/vision flags.

## Install

`hf2q` is a Cargo crate. Apple Silicon is currently the only supported
target — the inference path is Metal-only.

```bash
git clone git@github.com:robertelee78/hf2q.git
cd hf2q
cargo build --release
./target/release/hf2q --help
```

The build path-pins a local checkout of
[`mlx-native`](https://crates.io/crates/mlx-native) (`Cargo.toml:53`).
For a clean checkout without the sibling repo, replace the `path =
"/opt/mlx-native"` override with `mlx-native = "0.8"` from crates.io.

`cargo build` requires:

- macOS with Metal Performance Shaders (M1 or newer).
- A working Rust toolchain at the version pinned in `Cargo.toml`
  (`rust-version = "1.81.0"`).
- ~25 GB free disk for the full test corpus + scratch GGUF.

`hf2q doctor` enumerates the runtime checks (hardware detection, disk
space, optional RuVector backend); run it after `cargo install` if
anything misbehaves.

## CLI subcommands

| Command | What it does |
|---|---|
| `hf2q convert` | HuggingFace → GGUF or mlx-lm safetensors (with quantization). |
| `hf2q gguf-patch` | Rewrite a GGUF's metadata in place (e.g. inject a chat template). |
| `hf2q info` | Inspect an HF or GGUF model without converting. |
| `hf2q validate` | Quality check (cosine similarity, KL, perplexity) against a reference. |
| `hf2q generate` | Single-shot text generation from a GGUF on the local GPU. |
| `hf2q serve` | OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/embeddings`). |
| `hf2q parity` | ADR-009 parity validation against locked reference outputs. |
| `hf2q smoke` | ADR-012 end-gate smoke test for a registered architecture. |
| `hf2q cache` | Manage `~/.cache/hf2q/` (list / size / clear). |
| `hf2q dwq-train` | DWQ training over every `Linear` in a GGUF (ADR-020). |
| `hf2q dwq-overlay-drift` | Measure DWQ overlay drift through the round-trip. |
| `hf2q doctor` | Diagnose hardware, cache, RuVector, disk. |
| `hf2q completions` | Generate shell completions. |

Run `hf2q <command> --help` for the full flag surface.

### Quantization variants

The convert pipeline supports four families of `--quant` values:

| Family | Variants | Notes |
|---|---|---|
| Legacy block | `q4_0`, `q5_0`, `q8_0` | `llama.cpp`-compatible block formats. |
| K-quant | `q2_k`, `q2_k_s`, `q3_k_{s,m,l}`, `q4_k_{s,m}`, `q5_k_{s,m}`, `q6_k`, `q8_0` | Per-row scale + min, 256-element super-blocks. |
| Imatrix-K-quant | `imatrix-q{n}_k_{s,m,l}`, `imatrix-adaptive` | Same K-quant codecs, sensitivity-weighted with an activation imatrix. |
| Dynamic / DWQ | `dynamic-quant-{4-6,4-8,6-8,2-8}`, `dwq-4` (overlay) | Mixed-precision per-layer or scale/bias-trained per-Linear. |

Float passthroughs: `f16`, `bf16` (single-file mlx-lm safetensors only).

The full menu and per-arch availability live in
`docs/converting-a-model.md`.

## Quick start: serve a model

```bash
# Convert (one-time, ~10-30 min depending on model size + quant)
hf2q convert \
  --repo Qwen/Qwen3.5-35B-A3B-Instruct \
  --format gguf \
  --quant q4_k_m \
  --output models/qwen35-q4_k_m/out.gguf

# Serve
hf2q serve \
  --model models/qwen35-q4_k_m/out.gguf \
  --port 8080

# Use it (OpenAI SDK works out of the box)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen35","messages":[{"role":"user","content":"hello"}]}'
```

## Architecture

A full source-grounded architecture map lives in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). One-paragraph version:

```
   ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
HF │ input/       │ -> │ models/<arch>/   │ -> │ backends/    │
   │ - safetensors│    │ - tensor rename  │    │ - gguf       │
   │ - config     │    │ - MoE merge      │    │ - safetensors│
   └──────────────┘    │ - DWQ targets    │    └──────────────┘
                       └──────────────────┘            │ GGUF
                                                       v
                                              ┌──────────────────┐
                                              │ inference/       │
                                              │ - load + warmup  │
                                              │ - forward (mlx)  │
                                              │ - KV cache (TQ)  │
                                              │ - spec-decode    │
                                              └──────────────────┘
                                                       │
                                              ┌──────────────────┐
                                              │ serve/           │
                                              │ - OpenAI HTTP    │
                                              │ - SSE streaming  │
                                              │ - block-prefix$  │
                                              │ - multi-model    │
                                              └──────────────────┘
```

## Performance

Measured on M5 Max at HEAD against `llama.cpp` peer with identical
GGUFs (thermal-fair alt-pair protocol, `σ < 1%` per arm — see
`docs/peer-parity-baselines-2026-04-26.md` for the methodology):

- **Decode** — `0.93–0.94× peer-FA` (Flash-Attention path) across
  `tg100 / tg2000 / tg5000` regimes. Gap is structural; closing the
  last 5–6% requires a multi-month parallel-encode refactor and is
  intentionally not scheduled.
- **Prefill** — `1.07–1.09× peer-FA` (AHEAD) at `pp1800–pp3700`,
  driven by the ADR-011 Flash-Attention kernel + ADR-015 batched
  prefill path.
- **KV-cache footprint** — `~3.94× advantage` vs peer-F16-KV under
  TurboQuant (ADR-007), which Hadamard-quantizes K and V to ≈4 bits
  with negligible quality loss.
- **DWQ-quantized models** can match `Q4_0` size at noticeably lower
  KL divergence vs the FP32 teacher; the canonical
  `qwen3.6-35B-APEX-Q5_K_M` ships at `~1.34× peer` on decode at
  full quality.

Performance work is investigation-driven and tracked in numbered
ADR-029 (Gemma 4 decode), ADR-028 (peer-parity baseline), ADR-030
(speculative decode) iter-logs under `docs/`.

## Repository layout

```
src/
├── arch/          single source of truth for per-arch conformance
├── backends/      GGUF + mlx-lm safetensors writers
├── calibrate/     DWQ training, autograd, imatrix
├── inference/     per-arch forward graphs, spec-decode, vision
├── input/         HF config + safetensors loaders, HF Hub download
├── intelligence/  hardware probe, auto-quant heuristics, RuVector
├── ir/            internal tensor / metadata representation
├── models/        per-arch tensor rename + MoE merge
├── quality/       cosine / KL / perplexity scorers
├── quantize/      Q-format codecs (legacy / K-quant / DWQ / mixed)
└── serve/         OpenAI HTTP API, block-prefix KV cache, multi-model
docs/              ADRs 004–030 + per-feature runbooks
tests/             77 integration test files
scripts/           104 bench / repro / runbook scripts
```

## Development

```bash
cargo build              # debug build
cargo test               # full test suite
cargo build --release    # release binary
cargo run -- doctor      # diagnostic
```

The project is TDD-heavy: every ADR closes only when its acceptance
tests + smoke prompts pass. New architectures must be onboarded via
the checklist in `docs/arch-onboarding.md` — registry entry + tensor
catalog + smoke prompt before any forward-pass code lands.

## Documentation index

- `docs/ARCHITECTURE.md` — source-grounded architecture map.
- `docs/converting-a-model.md` — generic convert reference.
- `docs/converting-qwen35.md` — Qwen 3.5/3.6 specifics.
- `docs/operating-kv-cache.md` — TurboQuant KV cache operator guide.
- `docs/operator-env-vars.md` — every `HF2Q_*` env var, what it gates.
- `docs/ADR-004…ADR-030` — every architectural decision, with rationale and verification status.

## License

Apache-2.0. See `Cargo.toml` for crate metadata and
`docs/ADR-008-candle-divorce.md` for the dependency philosophy.
