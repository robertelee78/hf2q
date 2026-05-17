# hf2q

[![CI](https://github.com/robertelee78/hf2q/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/robertelee78/hf2q/actions/workflows/ci.yml)
[![License: Apache-2.0 OR MIT](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](#license)
[![Rust 1.81+](https://img.shields.io/badge/rust-1.81%2B-orange.svg)](https://www.rust-lang.org)
[![Platform: Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey.svg)](#install)
[![Backend: mlx-native](https://img.shields.io/badge/backend-mlx--native%200.9-purple.svg)](https://crates.io/crates/mlx-native)

Pure-Rust CLI for converting HuggingFace models to hardware-optimized
formats — and serving them through an OpenAI-compatible HTTP API on
Apple Silicon. **No C++ at build, test, or runtime** (ADR-008
sovereignty rule); the inference path runs entirely on `mlx-native`
Metal kernels we own end-to-end.

> **Performance** — on M5 Max at HEAD (2026-05-17 re-bench, 3-run
> median, default config including the HF2Q_NO_FA hybrid-attn fix from
> commit `03328ee5`):
> * **Gemma-4 26B-A4B Q6_K decode** — `tg200` 105.2 t/s vs llama.cpp
>   `-fa 1` 104.32 t/s (**1.01× peer-FA AHEAD**); `tg2000` 93.5 t/s vs
>   96.69 t/s (**0.97× peer-FA**).
> * **Qwen 3.6 35B-A3B APEX-Q5_K_M decode** — `tg200` 130.6 t/s vs
>   llama.cpp `-fa 1` 101.31 t/s (**1.29× peer-FA AHEAD**).
> * **Gemma-4 prefill** — `pp1800` 2734 t/s vs llama.cpp 2837 t/s
>   (**0.96× peer-FA**); `pp3700` 2703 t/s vs 2181 t/s (**1.24×
>   peer-FA AHEAD**) — hf2q's prefill rate drops only ~1% from
>   pp1800→pp3700 while llama's drops ~23%, so the cross-over is in
>   the lower part of this range.
> * **TurboQuant 8-bit KV cache** — Qwen 3.6 35B-A3B at 32K context:
>   340 MiB vs 1.34 GiB F32 baseline = **3.94× memory savings**
>   (ADR-027 iter-34, regression-pinned by
>   `tests/qh35_no_f32_kv_alloc_with_tq_kv.rs`).
>
> Methodology references in
> [`docs/peer-parity-baselines-2026-04-26.md`](docs/peer-parity-baselines-2026-04-26.md);
> the historical 1.05× decode + 1.07-1.09× prefill claims (ADR-029
> iter-175) were measured at a pre-HF2Q_NO_FA HEAD and do not hold
> at current main per the re-bench above.

| | |
|---|---|
| **License** | Apache-2.0 OR MIT (dual) |
| **Rust** | 1.81+ |
| **Inference backend** | [`mlx-native`](https://crates.io/crates/mlx-native) 0.9 (Apple Metal) — ADR-008 |
| **Output formats** | GGUF (`llama.cpp` consumers), mlx-lm safetensors |
| **Status** | Pre-release on M-series Macs. Some paths are fast and well-tested (batched prefill, TQ KV cache, Qwen 3.5 / 3.6 convert + serve); others are incomplete or actively under investigation (spec-decode wire-up, multi-arch coverage). See the ADR ledger for per-feature status. |

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
   quantization (legacy block `Q4_0` / `Q8_0`, K-quants
   `Q{2..6}_K_{S,M,L}`, imatrix-weighted K-quants including
   `imatrix-adaptive`, or mixed-bit `dynamic-quant-*`) and emit GGUF
   or mlx-lm safetensors.
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

The default `mlx-native = "0.9"` declaration at `Cargo.toml:105` resolves
from `crates.io`.  For local mlx-native development place a path
override in a gitignored `.cargo/config.toml` (template at
`Cargo.toml:217+`) — out-of-the-box `cargo build` does NOT path-pin
to a sibling checkout.

`cargo build` requires:

- macOS with Metal Performance Shaders (M1 or newer).
- A working Rust toolchain at the version pinned in `Cargo.toml`
  (`rust-version = "1.81.0"`).
- Per-arch disk floor for convert (`src/arch/entries/`): **100 GB** for
  Qwen 3.5 dense, **150 GB** for Qwen 3.5 MoE. Smoke preflight refuses
  to start below `disk_floor_gb + 10`.

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

The convert pipeline supports four families of `--quant` values (the
canonical `QuantMethod` enum lives at `src/cli.rs:1167`):

| Family | Variants | Notes |
|---|---|---|
| Float passthrough | `f16`, `bf16`, `auto` | No quantization; `auto` selects per-tensor. |
| Legacy block | `q2`, `q4` (alias `q4_0`), `q8` (alias `q8_0`) | `llama.cpp`-compatible block formats. |
| K-quant | `q2_k`, `q2_k_s`, `q3_k_{s,m,l}`, `q4_k_{s,m}`, `q5_k_{s,m}`, `q6_k` | Per-row scale + min, 256-element super-blocks. |
| Imatrix-K-quant | `imatrix-q2_k{,_s}`, `imatrix-q3_k_{s,m,l}`, `imatrix-q4_k_{s,m}`, `imatrix-q5_k_{s,m}`, `imatrix-q6_k`, `imatrix-adaptive` | Same K-quant codecs, sensitivity-weighted with an activation imatrix. |
| Dynamic / mixed | `dynamic-quant-{4-6,4-8,6-8,2-8}` | Mixed-precision per-layer sensitivity-driven bit pair. |

DWQ training (`hf2q dwq-train`, ADR-020) produces a separate
mlx-format safetensors overlay layered on top of an existing GGUF; it
is not selectable through `--quant`.

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

Re-bench at HEAD 2026-05-17 on M5 Max against `llama.cpp` peer
(build `389ff61d7`, `-fa 1`) with identical GGUFs.  3-run median;
hf2q uses default config including the HF2Q_NO_FA hybrid-attn
fix from commit `03328ee5`.  See
[`docs/peer-parity-baselines-2026-04-26.md`](docs/peer-parity-baselines-2026-04-26.md)
for the full thermal-fair alt-pair protocol used by ADR-029 baselines.

- **Decode (Gemma-4 26B-A4B Q6_K)** — `tg200` **1.01× peer-FA AHEAD**
  (hf2q 105.2 t/s vs llama-bench 104.32 t/s); `tg2000` **0.97× peer-FA**
  (hf2q 93.5 t/s vs 96.69 t/s).  The historical ADR-029 iter-175
  `~1.05× AHEAD across tg200/tg2000/tg5000` claim was measured at a
  pre-HF2Q_NO_FA HEAD; re-bench at current main shows it holding at
  tg200 only.
- **Prefill (Gemma-4 26B)** — crossover regime: `pp1800` **0.96×
  peer-FA** (hf2q 2734 t/s vs llama-bench 2837 t/s); `pp3700`
  **1.24× peer-FA AHEAD** (hf2q 2703 t/s vs 2181 t/s).  hf2q's
  prefill rate drops ~1% from pp1800→pp3700 while llama's drops
  ~23% (FA tile-skip helps less at longer K), so the cross-over
  sits early in this range.  The historical `1.07-1.09× AHEAD`
  claim across the whole range no longer holds at current main.
- **Decode (Qwen 3.6 35B-A3B APEX-Q5_K_M)** — `tg200` **1.29× peer-FA
  AHEAD** (hf2q 130.6 t/s vs 101.31 t/s).  Historical ADR-028
  `~1.34×` measurement is within ~4% of current re-bench (thermal /
  build drift).
- **KV-cache footprint** — TurboQuant 8-bit (ADR-007 + ADR-027 iter-34)
  drops F32 K/V allocations entirely on Qwen 3.6 35B-A3B at 32K
  context, **340 MiB vs 1.34 GiB F32-only baseline = 3.94× memory
  savings**.  This is the only major performance claim with an
  in-tree regression pin (`tests/qh35_no_f32_kv_alloc_with_tq_kv.rs`).

Regression protection for the decode path: 8 parity tests
(V2/V3 unbatched + V3 batched), `coherence_smoke` (2 cells),
200-token byte-identity verification.  No automated bench-vs-peer
gate is currently in CI — these numbers are operator-driven
re-bench, not continuously verified.

Note: DWQ at the production-default `perturb=1.0` is mathematically
equivalent to the underlying K-quant baseline (ADR-020 finding
2026-05-08); DWQ wins materialize only at lower perturb values that
move the scales/biases off the K-quant projection.

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
docs/              ADRs 004–031 + per-feature runbooks
tests/             77 integration test files
scripts/           109 bench / repro / runbook scripts
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
- `docs/ADR-004…ADR-031` — every architectural decision, with rationale and verification status.

## License

Dual-licensed under Apache-2.0 OR MIT (`Cargo.toml` `license` field;
`LICENSE-APACHE` and `LICENSE-MIT` files at repo root).  See
`docs/ADR-008-candle-divorce.md` for the dependency philosophy.
