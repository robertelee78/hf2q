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

## Known out-of-scope

These are deliberately not part of any category:

- Models other than Gemma-4 26B DWQ. Other Gemma variants and
  Llama/Qwen have not been exercised against the gates.
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
- `scripts/release-check.sh` — the reproducible gate runner.
