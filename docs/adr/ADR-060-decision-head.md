# ADR-060: Discriminative decision head — `laya` model family and `POST /decide`

- **Status:** Proposed — design recorded 2026-09-22; implementation gated
  behind the flags and gates below. No default behavior changes. The
  intervention-taxonomy framing (selection / computation / context /
  choice) lands in `docs/gcd-in-hf2q.md` alongside both ADRs.
- **Date:** 2026-09-22
- **Related:** ADR-053/ADR-057 (GCD serving surface, schema arm, locked
  policy), ADR-059 (grafts — the other half of this brief), ADR-017
  (per-family status), `docs/arch-onboarding.md` (the family order this
  follows), ADR-008 (mlx-native sole backend)

## Attribution

- **Laya** — Convai Innovations and upstream contributors: the typed-
  decisions concept, pretrained weights, RLCD training, prompt format,
  output schema, and fitted calibration temperatures.
- **laya-mlx** (mizorewww, `/opt/laya-mlx`): the independent MLX port
  whose architecture this ADR follows operation-for-operation, and whose
  published parity fixtures (63/63 validation questions, 378/378 FP32+
  FP16 comparisons) become our parity gate. Apache-2.0; see its NOTICE.
- hf2q implements the **Rust / mlx-native family** in-process. No Python
  runs in the production path — the boundary decision that forced this
  path (a sibling laya-mlx service would be an external inference
  process; hf2q owns inference).

## Problem

The taxonomy's fourth surface is **choice**: calibrated selection among
an enumerated action set, with an abstention path. Today hf2q can only
approximate this with the generative model — ask it to emit an option and
read the logit. That is not calibrated confidence: generative option
logits are not probabilities, and the GCD grammar guarantees only that
the emitted token is *in* the authorized set, not that the choice among
authorized options is measured.

The design rationale from the brief: steering (GLP/grafts) aligns model
*preference* with policy — probabilistic, no guarantee. Grammar (GCD)
supplies the constitutive *boundary* — guaranteed, but fights the model
when preference disagrees (the W1 campaign's 823/1024 token-limit
truncations). The decision head provides calibrated *choice* among the
allowed actions: it picks, it doesn't write. Combined with GCD compiled
from the same policy, the head decides *within* the boundary the grammar
enforces.

## Decision

### 1. New model family `laya`, in the established order

Registry + tensor catalog → conversion mapping → inference graph →
smoke/parity proof → serve. **Do not open the serve path before the
forward pass is proven** (the standing family rule). This is hf2q's
first **bidirectional** (non-causal, encoder-only) family — a new graph
class, not a variant of the causal LM path.

### 2. Architecture spec (operation-for-operation from laya-mlx)

`DecisionModel` = encoder + decision head + scorer + action head:

- **Encoder (ModernBERT-large, 421M)**: embeddings (token embedding +
  LayerNorm); per layer — attn_norm (**Identity at layer 0**, LayerNorm
  elsewhere), fused `Wqkv` (3×hidden, no bias) + `Wo`, bidirectional SDPA
  with **boolean key masks**; attention pattern: local sliding
  (window 128, **inclusive** distance ≤ 64) with RoPE base 10000, global
  every `global_attn_every_n_layers` (=3) with RoPE base 160000; MLP =
  `Wi` (2×intermediate, split) → GELU(value) × gate → `Wo`; final
  LayerNorm. Norm eps 1e-5, no biases.
- **Head**: `type_emb` (3 × hidden, added per question by qtype);
  `head_layers` (=2) HeadLayers — MHA (heads = dims/64) + LayerNorm +
  **ReLU** FFN (4×) (ReLU is a deliberate upstream quirk: the head uses
  ReLU where the encoder uses GELU).
- **Scorer**: LayerNorm → Linear → GELU → Linear → 1, applied at
  **marker positions** (the `[MASK]` token before each option); logits →
  softmax over options.
- **Action head**: Linear(hidden+4 → 256) → GELU → Linear →
  n_actions+1, on pooled `[CLS]` ⊕ features `[top1, top1−top2,
  normalized entropy, k/255]`.
- **Calibration**: temperatures from `rl_agent_config.json` — per-qtype
  `[3]` plus per-bucket `temperature_by_options` keyed
  `"{qtype}:{option-count-bucket}"` — **clamped to [0.5, 5.0] before
  use**; clamped buckets are reported as uncalibrated at load (the
  laya-mlx finding: the shipped `choice:11+` bucket is 0.1006 and would
  sharpen a coin flip into near-certainty). Raw values retained for
  inspection.
- **Confidence**: `1 − H(p)/log(k)` (normalized Shannon entropy).

### 3. Prompt format — verbatim, no reinvention

`[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state
[SEP]`, with `head_max_len` (=192) budget, per-option truncation at 48
tokens, `max_len` from config (512 English / 1024 typed-decisions).
Option rendering rules (label `": " description`, score levels
`"level %d: %s"`, noul false/true defaults) follow laya-mlx `common.py`
exactly. The tokenizer is HF-format via the existing `tokenizers` crate;
`cls`/`sep`/`mask` special tokens are required and validated at load.

### 4. Conversion — HF safetensors → GGUF (`laya.*` metadata)

hf2q owns conversion: `hf2q convert <laya-checkpoint>` emits a GGUF with
`laya.*` config keys (encoder config, head_layers, act_costs, max_len,
head_max_len, temperatures raw + clamped flag, layer_types) and tensors
under the family's tensor catalog. Parameter-name mapping follows
laya-mlx's `sanitize_weights` table (`.in_proj_weight` → fused qkv;
scorer/act_head sequential renumbering). Existing output directories are
never overwritten (house rule). Checkpoint identity and hashes follow the
established receipt pattern.

### 5. Single source of truth — one policy, two compiled artifacts

The option set and the GCD authorization grammar compile from the **same
authorized-action specification** (`PolicySpec`): one document, two
compilers — the existing JSON-Schema→GBNF path (ADR-057) for the
constitutive boundary, and a new option-set renderer for `/decide`
questions. Both artifacts carry the `PolicySpec` hash; a `/decide`
response's option-set hash and the served grammar's policy hash are
traceable to the same source. **No parallel hand-maintained policy
representations.** Under `--gcd-schema-locked`, decision requests accept
no caller-supplied grammar or tool definitions (`tool_choice: "none"`
semantics), and option sets must derive from server policy — a caller
cannot widen the choice set the grammar constrains.

### 6. Endpoint — `POST /decide`

Request: `state` (text | JSON object | conversation list) + `questions`
(keyed, typed: `choice` / `score` / `noul`, with instructions and
criteria). Response per question: calibrated probability distribution,
chosen option (argmax), confidence, token usage; plus abstention status
(§7) and the dossier fields (§8). Validation mirrors laya-mlx's
`_to_internal` rules (unique choice labels, nonempty criteria, typed
errors before inference). Batching: questions processed in bounded
chunks (batch 16 default), each question an independent row — the
runtime does **not** claim one state encoding reused across arbitrary
questions (the laya-mlx honesty note, preserved).

### 7. Abstention — first-class result

Server config `--decide-abstain-threshold <p>` (default: off). Below
threshold on the calibrated distribution, the response is a structured
abstain: `decision: null`, probabilities, confidence, threshold — HTTP
200, not an error. Routing to a human or a supervised generative fallback
is the caller's policy; hf2q ships no generative fallback routing in v1
(Non-goal).

### 8. Dossiers — hash-integrity in v1, signing spec'd and flagged

Optional decision record: policy version, `PolicySpec` hash, option-set
hash, probability distribution, chosen option or abstain, threshold,
model + checkpoint identity, timestamps — integrity-chained with sha256
(the existing `sha2`/`hex` deps). **Ed25519 signing is specified but not
added**: no signing crate exists in `Cargo.toml`, and adding
`ed25519-dalek` (or equivalent) is a dependency decision for the
integration owner, flagged here per the brief — not slipped in. The
dossier format reserves a signature envelope so signing lands without a
format change.

### 9. Scheduling and Metal contention

`/decide` is a bounded **prefill-class job**: ≤512/1024 tokens,
bidirectional, batched ≤16 questions, no decode loop, no KV cache. It
shares the Metal device with the generative engine; the scheduler treats
it as a prefill-class unit with a bounded budget. The contention trade
(brief's open question) is **measured, not assumed**: gate 6 runs the
head standalone (laya-mlx's reference: 13.4 ms P50 one-question on M3
Max) and under a sustained decode load, reporting both. If contention
measures unacceptable, the fallback is serializing `/decide` behind the
decode scheduler — decided by the measurement, not here.

### 10. mlx-native surface — exists vs to-author

We own `/opt/mlx-native`; gaps are work items there, each following the
publish-and-pin rule (a landed hf2q result is reproducible only after
the mlx-native revision is published and pinned in
`Cargo.toml`/`Cargo.lock` — no local `[patch.crates.io]` in landed
claims).

- **Exists**: `gelu` (f32/f16/bf16), `rope`/`rope_multi` (custom bases),
  `sdpa`/`sdpa_sliding`/`scale_mask_softmax`, `softmax`,
  `embedding_dense`, `dense_mm*` family, `gather`/`take_along_axis`
  (marker gather), elementwise ops.
- **To author**: (a) **LayerNorm-with-scale** op (encoder/head norms are
  LayerNorm, not RMSNorm — mean-center + variance + affine; bias-less
  variant suffices); (b) **ReLU elementwise** (head FFN); (c) a
  **bidirectional padded-mask attention path** — existing sdpa ops are
  causal-oriented; ModernBERT needs arbitrary boolean key masks with
  padded-query rows that see valid keys (the laya-mlx mask semantics:
  padded queries are never keys and never pooled, so valid-token results
  are unchanged). (c) may compose from `scale_mask_softmax` + dense
  matmuls first (correctness before kernel speed), with a fused kernel as
  a measured follow-up.

## Non-goals

- **No RLCD training or fine-tuning in hf2q.** Upstream owns training;
  hf2q converts, loads, and serves (the GLP/phantom producer-consumer
  split again).
- **No generative fallback routing in v1** (abstain returns to the
  caller).
- **No multilingual router, no shortlisting, no `predict_shortlist`** in
  v1 — single-checkpoint serving; the router is a follow-up if measured
  demand exists.
- **No universal-calibration claim.** Port fidelity = parity with
  laya-mlx probabilities (4-decimal), which itself is fidelity to
  upstream on their fixtures — calibration is *measured* here (ECE,
  gate 5), never assumed from a checkpoint temperature.
- **No new signing dependency** (above).

## Gates before this ships

1. **Conversion round-trip**: strict parameter-name/shape validation on
   load; every exported tensor checked against its source (the
   prepare-hub discipline).
2. **Parity**: the laya-mlx 63/63 validation questions × FP32 and FP16
   (378/378 comparisons), checked-in hash-identified fixtures; selected
   answer agreement and 4-decimal probability agreement. Zero tolerance
   on selected-label disagreement.
3. **Tokenizer contract**: cls/sep/mask handling, option truncation
   boundaries, state serialization (text vs JSON vs conversation list).
4. **`/decide` surface tests**: schema validation (typed errors),
   abstain behavior at and below threshold, locked-policy interaction
   (caller grammar/tool definitions rejected; option sets policy-derived),
   dossier hash chain.
5. **Calibration report**: ECE on a held-out labeled set in the eval
   harness, per qtype and option-count bucket, alongside the clamped-
   bucket warnings. Reported, not thresholded, in v1.
6. **Latency + contention**: standalone P50/P95 and under-sustained-
   decode-load medians, multiple runs, per §9.
7. **Paired decision-head arms** per the measurement doctrine where the
   head participates in an intervention comparison (e.g. head-choice vs
   generative-choice under the same policy grammar), with truncation and
   errors recorded alongside.

## Evidence behind the design

- laya-mlx `README.md`/`BENCHMARKS.md`: 13.42 ms P50 one-question (M3
  Max, 421M FP16), 146.8 q/s at batch 64, port fidelity 63/63 × both
  precisions, temperature-clamp finding, memory-growth stability (100
  repeated calls).
- laya-mlx `laya_mlx/{model,common,agent}.py`: the architecture spec in
  §2 and prompt format in §3 are transcribed from this source (read
  2026-09-22), not from memory.
- hf2q source grounding: `src/input/` (HF config/tokenizer/safetensors
  input), `src/arch/` (registry), `src/backends/` (GGUF writer),
  `src/serve/api/grammar/json_schema.rs` (the PolicySpec grammar
  compiler), `src/serve/api/gcd_policy.rs` (locked-policy precedent),
  `Cargo.toml` (no signing crate; `sha2`/`hex` present; `tokenizers`
  present).
- mlx-native `src/ops/` inventory (2026-09-22): exists/to-author table in
  §10.

## Open questions resolved / deferred

- **In-process vs sibling service** (brief's open question): resolved —
  in-process, forced by the inference-ownership boundary; contention is
  measured in gate 6 with a named fallback (scheduler serialization).
- **Calibration path** (brief's "discriminative encoder vs interim"):
  resolved — the discriminative encoder ships now (the Rust port IS the
  discriminative path); the interim generative-logit arm is not needed
  and is not built.
- **Ed25519**: spec'd, envelope reserved, dependency decision flagged to
  the owner (§8).
- **Router/shortlist/multilingual**: deferred (Non-goals).
