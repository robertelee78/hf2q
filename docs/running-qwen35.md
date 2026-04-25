# Running Qwen3.5 / Qwen3.6 Inference with hf2q

Canonical reference for invoking `hf2q generate` against Qwen3.5 (`qwen35`)
and Qwen3.5-MoE / Qwen3.6-MoE (`qwen35moe`) GGUFs on Apple Silicon.

See `docs/ADR-013-qwen35-inference.md` for the full architectural rationale,
phase plan, and end-gate methodology.

For the conversion side (HuggingFace → DWQ GGUF) see
`docs/converting-qwen35.md` and `docs/ADR-012-qwen35moe-conversion.md`.

---

## Hardware requirements

hf2q's Qwen3.5 inference path is a pure-Rust forward pass driven by the
`mlx-native` Metal GPU backend. Apple Silicon is required.

| Resource | Minimum | Recommended (apex MoE) |
|---|---|---|
| Chip | M-series (any) | M4 Pro / M3 Ultra / M5 Max |
| Unified memory | 32 GB | 48 GB+ |
| Free Metal working set per process | ~30 GB (35B-A3B MoE @ DWQ) | — |
| Disk for model | 25 GB (apex GGUF) | — |

The 35B-A3B MoE working set on the local apex GGUF (Q5_K experts, BF16
activations) sits around 29.8 GB Metal-resident at ctx 4096 (model 23.9 GB +
context 5.2 GB + compute 776 MB), measured 2026-04-23 against llama.cpp at
matching shape. The 27B dense variant runs in roughly half that footprint.

**One-model-at-a-time rule.** Loading a 35B-class model consumes ~30 GB of
unified memory in a single process. Running two concurrent inference
processes on the apex GGUF will OOM the M5 Max. If you launch a benchmark
or sourdough run, wait for it to finish before launching another. See
`feedback_oom_prevention.md`.

---

## Model layout

hf2q expects the GGUF and its sidecars in a single directory. The
conventional layout (matching ADR-012 conversion output and the local apex
checkpoint) is:

```
/opt/hf2q/models/<model-name>/
├── <model-name>.gguf            # the weights + GGUF metadata
├── tokenizer.json               # HF-style tokenizer (preferred)
├── config.json                  # HF source config (sidecar; not required for inference)
├── chat_template.jinja          # optional; consumed by `hf2q serve`
├── tokenizer_config.json        # optional sidecar
├── special_tokens_map.json      # optional sidecar
├── generation_config.json       # optional sidecar
└── mmproj-*.gguf                # optional vision tower (loaded but not executed; ADR-013 is text-only)
```

`hf2q generate` resolves `tokenizer.json` from the directory containing
the `.gguf`, falling back to GGUF-embedded vocabulary if the sidecar is
absent. The `--tokenizer <PATH>` and `--config <PATH>` flags override the
defaults.

The local apex MoE is at:

```
/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/
└── qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf  (25 GB)
```

---

## One-liner: text generation (apex MoE)

```bash
hf2q generate \
  --model /opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf \
  --prompt "Explain the difference between latency and throughput." \
  --max-tokens 256 \
  --temperature 0
```

Generated tokens stream to stdout; a 4-line hf2q header + per-step prefill /
decode timing + a final `--- mlx-native (Qwen3.5): N tokens in Xs (Y tok/s) ---`
footer print to stderr.

For dense (27B `qwen35`) GGUFs, swap the path; the dispatcher routes both
`qwen35` and `qwen35moe` arch strings through the same Rust entry point and
selects the dense vs. MoE FFN at load time.

`--temperature 0` selects the greedy fast-path
(`Qwen35Model::forward_gpu_greedy`, GPU argmax over the lm_head). Any
`--temperature > 0` switches to `forward_gpu` returning full vocab logits
for CPU-side temperature/top-k/top-p sampling. The greedy fast-path is the
match-or-beat target for the bench gate; sampling adds a per-step
`vocab×4` byte download from Metal that costs a few tok/s.

---

## One-liner: sourdough byte-prefix gate

```bash
scripts/sourdough_qwen35.sh \
  /opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf
```

Runs hf2q and llama-cli on the fixed prompt in
`tests/sourdough_qwen35_prompt.txt` at T=0 greedy and asserts the common
byte prefix of their generated output is at least 160 bytes (the calibrated
floor; current measured prefix is 180 bytes against HEAD `23e1128`). The
llama-cli binary is invoked as an external black-box reference at gate
time; it is never linked into hf2q at build, test, or CI time
(`feedback_hf2q_sovereignty.md`).

Override the floor with `--min-prefix N` and the decode budget with
`--max-tokens N`. Exit codes: `0` PASS, `2` drift earlier than floor, `3`
tool invocation failure.

---

## One-liner: benchmark gate (match-or-beat llama.cpp)

```bash
scripts/qwen35_bench.sh \
  /opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf
```

Runs both `llama-bench` and `hf2q generate --benchmark` across a matrix of
prefill × decode token counts and asserts hf2q ≥ 0.95× llama.cpp at every
data point (5% drift budget; tighten with `--drift-budget`). Use
`--skip-llama` mid-development to run only the hf2q side.

Current numbers at HEAD `23e1128` (M5 Max, mlx-native HEAD `25d4c4b`):

| Decode length | hf2q tok/s | llama.cpp tok/s | Ratio |
|---|---|---|---|
| tg64  | 110.7 | 97.3 | 1.138× |
| tg256 | 106.7 | 97.3 | 1.097× |
| tg1024 | 97.9 | 97.3 | 1.006× |

All three points clear the 0.95× gate.

---

## Known caveats

- **Greedy vs. sampled.** `forward_gpu_greedy` (GPU argmax) is the
  match-or-beat fast path. Any non-zero `--temperature` falls back to
  `forward_gpu` (full logit download to CPU for the sampler). The decode
  fast-path is exercised by the bench gate; the sampling path is correct
  but ~5% slower in tok/s on the apex GGUF.

- **Chat-template stop at token 106.** The apex GGUF's chat template terminates
  on the `<turn|>` boundary token (id 106) at ~978 tokens. Long-decode runs
  beyond ~1000 tokens or coherence stress beyond the chat-template horizon
  will short-circuit early. Tracked in `project_hf2q_generate_chat_template_stop.md`;
  a `--ignore-eos` flag is the documented unblock and is not yet wired.
  Until then, cap your `--max-tokens` at 1024 if you want guaranteed
  observation of the full decode budget.

- **Q5_K experts on the apex GGUF.** The apex MoE checkpoint stores its
  256 per-expert tensors as Q5_K. Inference requires the
  `dispatch_q5k_dequant` + `mul_mat_id_q5k_f32_bf16` Metal kernels in
  mlx-native — landed at commit `dd087a9` (mlx-native 2026-04-22). Building
  hf2q against an older mlx-native checkout will fail at GGUF load with an
  unsupported-quant error.

- **Hybrid KV cache geometry.** The Qwen3.5 architecture interleaves
  full-attention layers (3 of them in a 40-layer MoE) with linear-attention
  / Gated DeltaNet layers (37 of them). The KV cache holds two distinct
  state types: standard KV tensors for full-attn layers, and the
  recurrent state + 1D-conv state ring buffer for DeltaNet layers. Memory
  scales with token count for the full-attn slots only; DeltaNet state is
  fixed-size per layer regardless of context length.

- **BF16 K-cache decode constraint.** All decode-path SDPA kernels expect
  BF16 K and V caches with `head_dim ≥ 32`. The apex GGUF satisfies this
  natively (head_dim 256). Synthetic test models smaller than this will not
  exercise the production fast-path.

---

## What hf2q does NOT support (Qwen3.5 / Qwen3.6 path)

These are out of scope for ADR-013 and will not be added without a
follow-up ADR:

- **Parallel batches.** `hf2q generate` is single-prompt, single-stream.
  Batched-prefill (`HF2Q_BATCHED_PREFILL=1`) is gated to the Gemma-4 path
  only (`tooling_hf2q_batched_prefill_requires_unsafe_ack.md`); the
  Qwen3.5 dispatcher in `src/serve/mod.rs::cmd_generate_qwen35` walks one
  prompt at a time.

- **Tool use / function calling.** No tool-call grammar parsing, no
  structured-output enforcement. The chat template emits raw text; any
  tool-call wrapping is the caller's responsibility.

- **Multi-turn conversation.** `hf2q generate` is single-turn. The chat
  template is applied once at prefill; no turn-boundary KV-cache reuse,
  no `<|im_end|>` re-entry. `hf2q serve` (when wired for Qwen3.5 in a
  follow-up ADR) will expose the OpenAI-compatible multi-turn surface;
  for now use `hf2q generate` with one full-context prompt.

- **Vision tower.** `mmproj-qwen36-F16.gguf` is loaded by the convert path
  but never executed; ADR-013 explicitly scopes text-only inference.
  Adding the 27-layer ViT (patch_size 16, hidden_size 1152) is a
  follow-up ADR.

- **MTP execution.** Multi-token-prediction tensors are loaded
  (`Qwen35Model::mtp: Option<MtpWeights>`) but the MTP head is not
  executed at decode time; speculative-decode integration is a separate
  ADR.

---

## References

- `docs/ADR-013-qwen35-inference.md` — full ADR with phase plan.
- `docs/ADR-012-qwen35moe-conversion.md` — conversion contract (DWQ GGUF emission).
- `docs/converting-qwen35.md` — HF → GGUF conversion howto.
- `scripts/sourdough_qwen35.sh` — byte-prefix correctness gate.
- `scripts/qwen35_bench.sh` — match-or-beat performance gate.
- `tests/integration_qwen35moe.rs` / `tests/integration_qwen35_dense.rs` —
  CI-visible (opt-in via `--ignored`) generate-smoke tests.
