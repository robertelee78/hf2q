# hf2q Shipping Contract

This document defines the canonical product surface for `hf2q` as
shipped today, and the policy each environment variable is classified
under. Per-variable effects live in `docs/operator-env-vars.md`; this
document sits one level above and defines *what is supported*.

**Anything not listed in categories 1–3 below may be removed or
relocated without notice** — it is not part of the supported surface.

---

## Category 1 — Production contract (shipped default)

What the default release binary does with **no environment variables
set**, on the proven model class (Gemma-4 26B DWQ GGUF):

- Per-token `forward_prefill` (not batched).
- Dense **F32** KV cache.
- Default decode (single-buffer or dual-buffer internal tuning; not
  user-configurable).
- **Auto Q8 lm_head** with exact F32 rerank, selected when
  `hidden_size % 32 == 0` **and** F16 lm_head weight > 256 MB;
  otherwise F16.

### Required gates before merging

Every change that could affect the forward pass or lm_head must pass
`scripts/release-check.sh`:

| Gate | Floor |
|---|---|
| `short_hello` exact-byte match vs locked llama.cpp reference | ≥ 29 bytes |
| `sourdough` common-byte-prefix with llama.cpp | ≥ 3094 bytes |
| `sliding_wrap` common-byte-prefix with locked hf2q reference | ≥ 700 bytes |
| Decode perf sanity on the sourdough prompt | ≥ 95 tok/s |

---

## Category 2 — Supported operator knobs

User-facing escape hatches. Stable in the contractual sense: we will
not remove or silently change them without an ADR.

| Var | Values | Purpose |
|---|---|---|
| `HF2Q_LMHEAD_Q8` | `1`, `0`, unset | Force Q8 on, force F16, or auto-select. Escape hatch for models the auto heuristic classifies incorrectly. |

---

## Category 3 — Benchmarking-only (user-triggerable but unsafe)

Documented knobs for controlled measurement. Activating them requires
an explicit acknowledgment: `HF2Q_UNSAFE_EXPERIMENTS=1`.

| Var | Unsafe-ack | Purpose |
|---|---|---|
| `HF2Q_LMHEAD_RERANK=0` | **required** | Measure raw Q8 argmax cost. Reintroduces the rare near-tiebreak flip (observed as mid-decode `<pad>` emission). |

---

## Category 4 — Investigation-only (not part of product surface)

Internal scaffolding. Not listed in `docs/operator-env-vars.md` as
operator-facing; loaded through `src/debug/investigation_env.rs`
(centralized), not read ad-hoc in hot paths.

**Ack-required (known to risk correctness or runtime reliability):**

| Var | Notes |
|---|---|
| `HF2Q_F16_KV` | Known-worse KV cache representation; separate bug vs F32 path. |
| `HF2Q_BATCHED_PREFILL` | Experimental; errors when `seq_len > sliding_window`. |
| `HF2Q_SKIP_TQ_ENCODE` | Bisection scaffolding; produces garbage output. |
| `HF2Q_SKIP_TQ_SDPA` | Bisection scaffolding; produces garbage output. |

**Warn-on-activation, no ack (ineffective but safe):**

| Var | Notes |
|---|---|
| `HF2Q_GRAPH_OPT` | No measured win; reorder aborts on unannotated dispatches. |
| `HF2Q_LMHEAD_COMPARE` | Keeps both F16 and Q8 resident; inert (not wired into live decode). |
| `HF2Q_DUAL_BUFFER` | Internal perf tuning; default (3) is part of category 1. |

**Silent / read-only diagnostics (no warning, no ack):**

| Var | Notes |
|---|---|
| `HF2Q_PREFILL_DUMP`, `HF2Q_BATCHED_DUMP`, `HF2Q_BATCHED_LAYER_SCAN`, `HF2Q_DUMP_LAYERS`, `HF2Q_DUMP_BOUNDARY`, `HF2Q_DUMP_ALL_CACHE`, `HF2Q_DUMP_LAYER_DETAIL`, `HF2Q_DUMP_NORM_WEIGHT`, `HF2Q_DUMP_DIR` | Hidden-state / cache dumps; output-only, cannot affect decode. |
| `HF2Q_DUMP_RENDERED_PROMPT`, `HF2Q_DUMP_PROMPT_TOKENS` | Prompt-path diagnostics. |
| `HF2Q_MLX_TIMING`, `HF2Q_SPLIT_TIMING`, `HF2Q_MLX_KERNEL_PROFILE`, `HF2Q_MLX_PROFILE` | Timing / kernel-attribution diagnostics. |
| `HF2Q_QWEN36_AUTOREG` | Wave 5a opt-in: when `=1`, dispatch Qwen3.6 GGUFs (`general.name` substring `qwen3.6`) through the existing autoregressive Qwen3.5 forward path (`inference::models::qwen35::*`). When unset, Qwen3.6 GGUFs soft-error with operator-actionable bail. Dispatch gate only — does not modify forward-pass math; no `HF2Q_UNSAFE_EXPERIMENTS` ack required. Removed once Wave 5b chunk-scan kernel lands (long-prefill SOTA path covers all Qwen3.x without env gate). |

---

## Classification rule

A toggle requires the `HF2Q_UNSAFE_EXPERIMENTS=1` acknowledgment when
it is **known to risk correctness or runtime reliability** — not
merely because it is experimental or inert. Toggles that are
ineffective-but-safe get a startup warning, not a gate. Toggles that
are read-only diagnostics get neither.

When a new toggle is introduced, classify it by this rule and register
it in `src/debug/investigation_env.rs` (for category 4) or update this
document (for categories 2–3).

---

---

## Qwen3.5 / Qwen3.6 conversion acceptance (ADR-012)

`qwen35` (dense 27B) and `qwen35moe` (MoE 35B) are shipped as **convert-only**
model classes as of ADR-012. Inference coherence is delegated to ADR-013.

### Acceptance gates for a converted GGUF

| Gate | Criterion |
|---|---|
| Structural validity | File begins with magic `GGUF`, version 3, tensor_count > 0, kv_count > 0 |
| Metadata completeness | Every key in the ADR-012 Decision 7 catalog is present |
| Tensor naming | Every tensor name matches the ADR-012 Decision 8 naming spec |
| llama.cpp load | `llama-cli --model out.gguf -p "Hello" -n 8` exits 0 |
| Sidecar set | `tokenizer.json`, `tokenizer_config.json`, `config.json`, `generation_config.json`, `special_tokens_map.json` (and `chat_template.jinja` when present) are byte-identical copies alongside the GGUF |
| MTP tensors (when `mtp_num_hidden_layers > 0`) | Round-trip integrity gate at `tests/convert_qwen35_mtp_roundtrip.rs` (Decision 19); 4 tensors land at `blk.{num_hidden_layers}.nextn.{enorm,hnorm,embed_tokens,eh_proj}.weight` |
| mmproj (when `--emit-vision-tower` and `vision_config` present) | Pure-Rust emitter at `src/models/vit/`; produces `mmproj-<slug>-F16.gguf` per Decision 18 with three layers of structural / round-trip / spec-driven test coverage |
| Smoke harness | `hf2q smoke --arch <qwen35\|qwen35moe> --quant q4_0` exits 0 with byte-identical transcripts across two fresh runs (Decision 16) |

### DWQ activation-based quantization for qwen35/qwen35moe

**Shipped 2026-04-25** under ADR-012 P9 + P9b (formerly listed as
"out-of-scope" pending ADR-013 P12). The convert pipeline now runs
the full two-pass activation calibration end-to-end:

  1. Emit intermediate F16 GGUF from the in-memory tensor_map
     (`backends::gguf::emit_gguf_from_tensor_map`, P9b.1).
  2. Construct `RealActivationCapture::new(intermediate_gguf, tokenizer)`
     which loads via the ADR-013 `Qwen35Model::load_from_gguf` path
     (P9b.3b).
  3. Run `quantize::dwq_activation::run_dwq_activation_calibration`
     which generates calibration tokens, runs the CPU forward pass
     through the loaded model, computes per-layer sensitivity, and
     produces a derived `MixedBitQuantizer` configured with
     activation-driven sensitive layers (P9b.3a).
  4. Final GGUF is emitted at the user-specified output path. The
     intermediate is dropped via `tempfile::TempDir` RAII (P9b.5).

No weight-space fallback for these architectures (Decision 13).

Real-model artifact production for the four end-deliverable GGUFs
(qwen35/qwen35moe × dwq46/dwq48) is gated only on environment
(HF_TOKEN + ~150 GB disk + Metal-validated llama.cpp build).

### Out-of-scope for ADR-012

- Inference coherence (sourdough gate, sliding-window parity) — ADR-013.
- MTP head **inference** (speculative decoding) — ADR-013 P14. ADR-012 P11
  ships the conversion-side tensor round-trip integrity gate; runtime
  draft/accept loops are owned by ADR-013.
- ViT compute path for the converted mmproj — ADR-005 phase 2c. ADR-012 P10
  ships the GGUF emitter; forward-pass execution is ADR-005's deliverable.

### CI integration tests

`tests/convert_qwen35_integration.rs` and
`tests/convert_qwen35moe_integration.rs` run the full convert pipeline on
synthetic tiny models (4 layers, hidden=64, 4 experts) to validate structural
correctness and sidecar behavior without downloading real model weights.

---

## Known out-of-scope

These are deliberately not part of any category:

- Inference coherence for models other than Gemma-4 26B DWQ. Qwen3.5
  inference coherence is ADR-013.
- Byte-identical batched-prefill parity with llama.cpp at the ~752-byte
  `sliding_wrap` level (see `docs/ADR-010-exact-batched-kernel-parity.md`;
  deferred).
- An OpenAI-compatible server. The `hf2q serve` CLI subcommand exists
  but is a stub; it is not part of the shipping contract.

---

## References

- `docs/operator-env-vars.md` — per-variable effects and defaults.
- `docs/ADR-009-reference-parity-and-coherence-recovery.md` — why
  F32-KV and per-token prefill are the defaults.
- `docs/ADR-010-exact-batched-kernel-parity.md` — why batched-prefill
  parity is deferred.
- `docs/ADR-012-qwen35moe-conversion.md` — qwen35/qwen35moe convert spec.
- `docs/converting-qwen35.md` — canonical convert commands for Qwen3.5/3.6.
- `docs/converting-a-model.md` — generic convert reference including Gemma.
- `scripts/release-check.sh` — the reproducible gate runner.
