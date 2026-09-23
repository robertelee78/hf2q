# ADR-059: KV-cache grafts — runtime unalignment artifacts (`--kv-graft`)

- **Status:** Revised 2026-09-23 — **the product success criterion reads
  YES on the core claim.** A trained K/V bank, served by hf2q as a
  graft file on the unmodified Qwen3.5-4B (Q4_K_M), removes 83% of
  refusals on the phantom-kv harmful scoreboard (46/60 → 8/60) with the
  harmless suite fully preserved (0/20 both arms) — no weight edits,
  byte-identical base artifact, reversible by unbinding. The reference
  result is CONFIRMED in its own stack through the ported pipeline
  (46/60 → 4/60 on bf16); the 8-vs-4 gap is the bf16-trained-bank-on-
  Q4-weights parity cost. Open honesty items: the capability cost is
  not yet measured on the hf2q side (the reference measured GSM8K
  45/75 → 27/75 under their trained bank — a real cost to expect);
  family generality (the site matrix beyond qwen35 `full_attn_kv`) is
  the declared expansion; the residual-refusal count is conservative
  (the lexical judge flags late-window phrases on substantively
  engaging answers).
- **Date:** 2026-09-22
- **Related:** ADR-053 (GLP serving surface — the artifact/binding pattern
  this mirrors), ADR-054 (GLP calibration and its measured null result),
  ADR-027 (TQ KV quantization — the graft encode path), ADR-040 (multi-seq
  slots), ADR-042 (DeepSeek-V4 compressed attention), ADR-013 (Qwen3.5
  hybrid stack), ADR-017 (per-family status), ADR-060 (decision head — the
  other half of this brief)

## Attribution

- **phantom-kv** (lordx64, `/opt/phantom-kv`): the graft concept — a
  trained per-layer K/V bank spliced into the attention cache as fabricated
  history at invariant positions `0..N` — the container semantics, the
  training pipeline (dual objective: CE on harmful + KL-anchor to base on
  harmless), and the measured record (Qwen3-4B: 25/60 → 5/60 refusals,
  KL mean 0.015; persistence half-life ≈ 2–4k tokens; GSM8K capability
  cost 45/75 → 27/75 at a 256-token budget; classifier-recall caveats).
- hf2q implements the **native Rust reader/splice** in `mlx-native` paths.
  phantom-kv (Python/PyTorch) remains the derivation toolchain — the same
  producer/consumer split as GLP vs weightless (ADR-053 non-goal: "hf2q
  consumes only").

## Problem

**The product: unalign a model without touching its weights.**

A vanilla aligned model refuses a class of requests. The established
remedy is a weight-level unalignment fine-tune (abliterix-class) —
permanent, per-artifact, requires training and shipping a new model
file. The product goal of this ADR is the same outcome as a RUNTIME
artifact: a small, checkpoint-bound file that, when bound to the
serving process, shifts the forward-path probabilities so the model
answers questions it would otherwise refuse. No weight edits, no
fine-tune, instant rollback (unbind + fresh state restores the
baseline on byte-identical weights).

**Success criterion (binary, the only scoreboard):** on a stock
aligned model with measured refusal headroom, a bound graft
measurably reduces refusal on a fixed probe set, at an acceptable
capability cost. Refusal rate drops = the feature works. Refusal
rate unchanged = it does not, whatever the plumbing proves.

The supporting architecture argument (why this mechanism class):
the intervention taxonomy has two shipped surfaces — **selection**
(GCD — masks logits at sampling) and **computation** (GLP — modifies
activations mid-forward, and measured behaviorally inert in
ADR-054). Missing: **context** — what the model conditions on.
phantom-kv's mechanism is architecturally unlike GLP: no per-token
hook, no 1-D direction assumption. Steering is **attention-gated** —
the model's own attention decides how strongly to weight the graft,
per head, per token. That is the mechanism by which the refusal
probabilities are meant to move.

**Measured so far (hf2q, honest record):**

- The MECHANISM is proven end-to-end: bound banks splice into every
  full-attention layer on both engine paths and both KV substrates,
  demonstrably shift the token stream (participation canaries), and
  restore the baseline exactly when unbound. Zero-slot grafts are
  byte-identical no-ops. Cost: ~5% decode tok/s at 64 slots on a
  333-token context, scaling as `n_slots / context_length`.
- The DONOR-bank class is FALSIFIED for behavioral transfer: banks
  derived from the model's own prefill of a donor TEXT (its state of
  READING a compliance/style request) produced zero behavioral
  movement at any dose — 0/12 style transfer at 64/128/256 slots on
  Qwen3.5-4B (2026-09-23, isolated dose-response, pre-registered
  metrics), and 0/3 refusal restoration on the already-unaligned
  APEX 35B at 64 slots (2026-09-22 panel). Reading about complying is
  not complying: the K/V of a user-turn request does not carry the
  assistant-side compliance state.
- phantom-kv's measured record (their stack, Qwen3-4B): TRAINED banks
  moved refusals 25/60 → 5/60 at KL 0.015 — the product outcome
  exists in their pipeline, at a real capability cost (GSM8K 45/75 →
  27/75). Their donor/v1 prefill banks were correspondingly weak.
  Their training objective is the product objective: CE on harmful
  (compliance) + KL anchor to base on harmless (capability
  preservation).

**The reformulated path to YES — MEASURED, and it reads YES:**

The reference pipeline (phantom-kv), ported to the hybrid architecture
(`scripts/graft_probe/phantom_port.py`; their donor text, suites,
judge, run format, three-role objective ce 50% / sup 25% / kl 25%,
their hyperparameters — 400 steps, AdamW 1e-3, sup_margin 3.0, L2
anchor 1e-2 — with exactly one seam adapted: the bank covers the 8
full-attention layers and every cache splice lands at indices
{3, 7, …, 31}, matching hf2q's serve-time splice semantics):

1. **Base run** (their stack, HF bf16): 46/60 harmful refusals,
   0/20 harmless — more headroom than their Qwen3-4B (25/60).
2. **v1 donor run**: 43/60 — the donor bank flips only 3 (its real
   job is the data generator: 3 genuine compliant completions for the
   `ce` targets, plus 13 base-complies; the 46 recorded refusals are
   the `sup` targets; 20 harmless completions are the `kl` anchor).
3. **Training**: sup loss 2.14 → 0.00 (every recorded refusal driven
   below the margin), ce 0.59 → 0.29, kl 0.039, anchor drift 1.6e-2.
4. **Reference confirmation** (their stack, bf16): **46/60 → 4/60
   refusals, harmless 0/20 unchanged** — their demonstrated result
   class, reproduced on hybrid Qwen3.5.
5. **PRODUCT** (hf2q serving the trained bank as a `graft.*` GGUF on
   the unmodified Q4_K_M artifact, TQ substrate, SerialFifo):
   **46/60 → 8/60 refusals, harmless 0/20 unchanged.** The trained
   effect survives quantized weights, mlx-native kernels, and the TQ
   KV substrate. Compliance is genuine (substantive, coherent direct
   answers — verified by content inspection, not just the lexical
   judge). The 8-vs-4 gap vs the bf16 stack is the bank-precision
   parity cost; serving an F16 conversion is the named tightening if
   8 is not acceptable.
6. **Capability cost (hf2q side, the honesty metric): preserved
   within noise** — GSM8K 21/75 → 31/75 (+10, ~2σ; the graft's
   direct-answer style appears to AID final-answer extraction on a
   4B that over-hedges at baseline), MMLU 19/100 → 15/100 (−4,
   exactly 1σ). The reference measured a real GSM8K cost (45/75 →
   27/75) under their ALL-layer bank on a standard-attention model;
   ours touches 8 of 32 layers (the hybrid `full_attn_kv` site), a
   plausible mechanical reason the capability interference is milder.
   The full qwen product scoreboard: refusal 46→8, harmless 0/20
   intact, capability preserved — regenerated end-to-end
   deterministically after a /tmp wipe (same distilled counts, same
   bank sha, same 4/60 reference result; artifacts now live under
   `artifacts/grafts/`, gitignored).

The falsified path that got here (kept as evidence): donated banks —
both user-turn reading state and assistant-ack multi-shot state — are
behaviorally inert at every dose (0/12 style transfer at 64/128/256
slots; refusal unchanged under the ack donor), and K/V amplification
of donor rows is a bludgeon (V×4 suppresses the refusal register by
degrading generation into template word salad — content-verified
dead). The bank must be TRAINED against the refusal objective; the
reference trainer is the proven way to produce it.

## Splice-site reality (the load-bearing constraint)

**Universality principle.** The graft *container* is family-independent
data (per-layer K/V tensors + metadata — cache-data, not an engine
patch). The *splice site* is per-architecture — but by the models' own
construction, not by hf2q's convenience, and the set of sites is small,
closed, and enumerable. Every attention-based architecture exposes at
least one. What changes per site is where tensors land and how the bank
must be derived, not whether the concept applies. A future spec version
may let one graft file carry **multiple site sections** for one
checkpoint (e.g. full-attn tensors + recurrent-state tensors) for maximum
coverage; v1 files carry exactly one site. The artifact itself is always
checkpoint-bound (post-RoPE K at fixed positions, exact revision) —
universality of the format and toolchain, never of one bank across
models (phantom-kv's own stated split).

phantom-kv's serving path is HF transformers `DynamicCache` — full
attention, every layer has K/V, one site covers everything. hf2q's
served families are hybrid/compressed, so the same concept distributes
across the site matrix:

| Site | Where it lands | Families that expose it | v1 rollout |
|---|---|---|---|
| `full_attn_kv` | Reserved leading slots of full-attention layers (global, append-only) | Qwen3.5/3.6/3.8 (1-in-4 layers), Gemma-4 (Full layers), — any layer with full attention over past K/V | **Ships in v1.** No eviction, no ring, no hot-path kernel change. |
| `window_tail_kv` | Sliding-window layers: graft rides the window tail and is re-injected as the window slides (phantom §6.11 refresh semantics — their measured mitigation) | Gemma-4 (Sliding layers) | Staged site 2. The trained receptive field of a sliding layer excludes old positions, so a stable prefix graft is invisible there by the model's own attention pattern; tail-riding is the correct placement. |
| `compressed_kv` | DeepSeek-V4 compressor output region (post-`attn_compressor_kv` space) | DeepSeek-V4 | Staged site 3. Positions `0..N` are merged into compressor state immediately, so a raw prefix splice is meaningless; a graft must be *derived in compressed space* (gradients through the frozen compressor — phantom's pipeline can in principle). No derivation tooling exists yet. |
| `recurrent_state` | DeltaNet conv/recurrent state buffers | Qwen3.5/3.6/3.8 (3-in-4 layers) | Staged site 4. These layers have no K/V at all — the graft medium is recurrent state, a different tensor geometry and derivation. This is the site that lifts Qwen coverage from 25% toward full. |

The Qwen v1 coverage limitation (graft touches full-attn layers only) is
a **measured question, not an assumption**: phantom-kv never grafted a
hybrid architecture. The canary and paired-arm gates exist to answer
whether 25% layer coverage steers at all on this family — and the
`recurrent_state` site is the named escalation if it does not. A null
result is reported honestly and is a successful gate (cf. ADR-054).

Unimplemented sites are **fatal at load** (the `glp.mode` pattern): the
reader rejects a site it does not implement rather than silently
mis-splicing — this is what makes staging safe.

## Decision

### 1. Artifact format — GGUF with `graft.*` keys

First-class artifact mirroring the GLP reader-conformance discipline
(fail-closed on malformed/ambiguous, reject unimplemented operations):

- `graft.spec_version` (u32, =1)
- `graft.mode` (string): v1 implements `splice_prefix` only. Any other
  value is **fatal** at load (the `glp.mode` pattern).
- `graft.kind` (string): provenance — `prefill_kv` | `softprompt_kv` |
  `direct_kv` (phantom's three kinds; all three splice identically, the
  kind records how the bank was derived). Unknown kinds are fatal.
- `graft.hook_point` (string): the splice site — v1 implements
  `full_attn_kv` only; the staged sites `window_tail_kv`,
  `compressed_kv`, and `recurrent_state` are reserved names (see the
  site matrix above). Unknown values are fatal; **known-but-unimplemented
  values are fatal too** — the reader never silently mis-splices a site
  it does not implement (the `glp.mode` pattern).
- `graft.n_slots` (u32): bank length N; positions `0..N-1`.
- `graft.content_sha256` (string, optional but expected): sha256 over raw
  F32 K-then-V tensor bytes in increasing layer order, excluding metadata
  and padding — the `glp.content_sha256` rule. Verified **before** any
  device placement.
- `general.base_model.0.*` + conversion-receipt binding: checkpoint
  identity uses the GGUF-conventional keys (the same keys GLP and the
  weightless ecosystem use), so ADR-053's `CheckpointIdentity` trust
  boundary — output-verified hf2q conversion receipts supply the actual
  source repository/revision for converted GGUFs; model-card ancestors
  never identify the served checkpoint; conflicts are fatal;
  unverifiable declared revisions are rejected — applies verbatim with
  no duplicated identity logic.
- **RoPE binding keys** (new, graft-specific): cache K is post-RoPE, so a
  graft bank is (checkpoint, RoPE configuration, position base)-bound.
  `graft.rope_theta`, `graft.rotary_dim`, `graft.position_base` (=0),
  plus family flags (e.g. `graft.mrope_interleaved` for qwen35). Binding
  validates these against the model config; mismatch is fatal. Cross-
  quantization loads remain possible but behaviorally unverified — the
  artifact records its quant lane and the loader warns below the spec's
  revalidation threshold, mirroring GLP.
- Tensors: `graft.k.<layer>` / `graft.v.<layer>` F32,
  `[n_slots, n_kv_heads, head_dim]`, present **only for layers the hook
  covers** (qwen35: full-attn layer indices; gemma4: Full layer indices).
  A tensor at a non-covered layer index is fatal.

### 2. CLI surface

- `hf2q serve <model> --kv-graft <file.gguf>` — explicit file only.
  **No Hub auto-discovery** — a deliberate divergence from bare `--glp`,
  recorded here: graft banks are checkpoint+RoPE+quant bound, discovery
  cannot safely choose among variants, and the brief mandates no silent
  resolution. `hf2q info --kv-graft <ref>` performs reader-conformance
  preflight without tensor residency. `hf2q chat` forwards the flag to a
  server it spawns (same scope semantics as `--gcd`/`--glp`).
- Per-request graft selection is **deferred** (v1 is a server-level
  modifier of the resolved model, the `--mmproj` shape). A per-request
  surface needs its own policy review (a caller-chosen graft is a
  caller-chosen steering intervention — the GCD locked-policy analogy
  applies) and is a named follow-up.

### 3. Splice semantics and the position offset

The graft occupies cache positions `0..N-1`; the rendered prompt begins at
position N. This matches phantom-kv's invariant-position contract (banks
are trained there; K is post-RoPE at those positions). Blast radius, all
behind the flag:

- RoPE position offsets for prefill/decode (`positions` vectors, decode
  position bases)
- Cache write positions (`write_pos` starts at N)
- `retained_prefix` / transcript reconstruction (graft slots are **not**
  conversational history — excluded)
- Prefix/LCP cache identity (below)
- Speculative-decode prefix boundaries and kv_persist snapshots (below)

Gate 5 (below) proves the offset plumbing is a bit-exact no-op when no
graft is bound.

### 4. Cache discipline (the security-critical part)

- **Graft slots are tagged, not anonymous**: a `GraftRegion` marker in the
  slot metadata spans positions `0..N`. Live-compaction, eviction, and
  summarization paths must treat it as immutable reserved state. On the
  v1 families the full-attn caches are append-only with fixed capacity,
  so the enforcement is: (a) the tag exists and is checked, (b) any future
  compaction path (ADR-040 Phase D paged eviction, DeepSeek-style
  compressors if ever added to these families) must skip the region —
  enforced by tests that construct a graft region and run the compaction
  entry points.
- **Never served as history**: graft slots are excluded from
  `retained_prefix`, transcript reconstruction, and any client-visible
  context accounting. A request that did not ask for the graft never sees
  its bytes; a request on a grafted server sees the graft's *effect* only
  through attention (that is the intervention, and it is auditable at
  boot).
- **Prefix-cache identity includes the graft hash**: the LCP identity
  extends the `glp_steering_params_hash` pattern to
  steering-and-graft identity — a grafted prefix can never masquerade as
  a clean one, and differently-grafted servers never share saved KV.
  Same graft content → same key; any change to bank bytes, mode, hook, or
  n_slots → different key; ungrafted → stable and distinct.
- **kv_persist snapshots exclude the graft**: disk snapshots store only
  real prompt/decode state; hydrate re-splices the graft from the bound
  artifact and re-verifies `graft.content_sha256`. Single source of
  truth, smaller snapshots, no graft bytes on disk.
- **Reset semantics**: existing KV state retains steering effects (the
  phantom persistence measurement is the precedent). Changing or
  disabling a graft mid-session **invalidates the affected slots' state**
  (documented, enforced — the slot is rebuilt, not patched). Fresh-state
  disable restores the ungrafted baseline bit-for-bit (gate 4).
- **TQ interaction**: if a slot is TQ-active, graft K/V pass through the
  same `hadamard_quantize_kv` encode path once at splice (device-resident,
  encoded into the slot's TQ buffers with the same codebook-bits
  resolution as this process's prefill rows — byte-identical to encoding
  the bank rows as a prefill chunk, enforced by test). The splice refuses
  by name if head_dim is not 256/512 (the kernel encodes no other dim);
  the F32 control path (`HF2Q_TQ_KV=0`) is the fixture substrate.

### 5. Determinism

A graft changes the forward pass, not the determinism story: same seed +
same graft + same prompt = same trace. The replay-cache identity includes
the graft hash. No sampling-path changes.

### 6. Canaries (the GLP pattern, extended)

1. **Zero-slot canary**: a graft with `n_slots=0` (or unbound) must be a
   bit-exact no-op on logits — proves the plumbing.
2. **Live graft canary**: a bound graft must shift logits by > 1e-3 on a
   fixed probe prompt — proves the splice actually participates in
   attention.
3. **Disable-restore canary**: unbinding with fresh state must restore
   the ungrafted baseline bit-for-bit.
4. **KL canary**: teacher-forced KL(grafted ‖ base) on a fixed benign
   probe set, reported per run — phantom-kv's own honesty instrument
   (their KL floor was partly memorization; ours is a different probe set
   and is reported, not thresholded).

## Non-goals

- **No graft training/derivation in hf2q.** phantom-kv owns the toolchain
  (mirror of ADR-053's non-goal on GLP directions). A future
  `hf2q calibrate`-style capture path would be a separate ADR.
- **v1 ships one site (`full_attn_kv`); the other three are staged, not
  rejected.** Each staged site opens only with (a) its splice contract
  specified here or in a successor ADR, (b) derivation tooling that can
  produce banks in that site's tensor geometry (phantom-kv side), and
  (c) its own canary + paired-arm gates. Until then the reader fails
  closed on those `graft.hook_point` values.
- **No Hub auto-discovery; no per-request selection API in v1** (above).
- **No behavioral-effect claim without paired arms.** phantom-kv's
  numbers are theirs; the GSM8K capability cost they measured is a
  standing warning that capability spot-checks are mandatory here.

## Gates before this ships

1. **Reader conformance**: spec clauses; malformed metadata, ambiguous
   shapes, unknown mode/kind/hook, hash mismatch, non-covered-layer
   tensors — all fatal, all tested. **DONE** (23 tests).
2. **Bind validation matrix**: layer indices vs site coverage,
   n_kv_heads, head_dim, dtype, RoPE keys vs model config, checkpoint
   identity (receipt-bound), TQ encode compatibility. **DONE** (9
   tests; complete-coverage contract).
3. **Cache-identity separation**: grafted ≠ clean ≠ differently-grafted
   keys; cross-slot isolation under multi-seq (ADR-040 slots). **DONE**
   (splice primitive 11 tests incl. byte-level row verification,
   slot isolation, graft-aware snapshot/restore round trip; LCP
   identity test).
4. **Canaries 1–4** above, on hardware, per family that ships.
   **MEASURED 2026-09-22 on Qwen3.6-35B-Abliterix-APEX Q5_K_M**
   (qwen35moe, F32 control path `HF2Q_TQ_KV=0`,
   `HF2Q_QWEN_SPECULATION=off`, thinking unbudgeted — all arms
   configuration-matched per the doctrine; harness:
   `scripts/graft_probe/canary.sh`, one campaign per scheduler):
   - **SerialFifo campaign** (`--scheduler fifo-serial`): zero-slot
     canary (n_slots=0, no tensors) output **byte-identical** to the
     ungrafted baseline — the plumbing is a no-op when absent; live bank
     (64 slots, every full-attention layer, deterministic synthetic
     rows, scale 8.0) output **diverges** from baseline — the splice is
     read by attention and shifts the forward pass (participation
     proven; behavioral quality is the paired-arm panel's job, not the
     canary's); disable with a fresh process output **identical** to
     baseline. Dose note: 8 slots at scale 0.5 left greedy argmax
     unchanged — small synthetic doses can be invisible under greedy
     decoding; the canary uses the measured guaranteed-shift dose.
   - **SlotAware campaign** (`--scheduler inflight-batched`, after the
     continuous-batching graft wiring): **ALL PASS** — zero-slot
     byte-identical, live bank diverges (with the same degenerate
     prefix as the serial live arm — the splice effect is consistent
     across engine paths), disable restores. The SlotAware wiring:
     splice at cold admission in `Qwen35PrefillState::begin`, warm-path
     graft-region invariant (a reset clears anchors, so a warm
     admission with a missing region tag is an invariant failure that
     refuses by name), graft-shifted RoPE positions (prefill chunks and
     `decode_position_base`), graft-aware slot anchors (the anchor
     boundary is the PHYSICAL cursor — graft rows + prompt rows — and
     the append-only contract keeps the rows intact across
     capture/restore), per-request capacity and KV-byte accounting that
     counts the graft rows, and MTP/spec-prefix suppression (the MTP
      arena never sees the graft). Vision/extension generation and
      embeddings still refuse grafts by name at their own gates.
    - **TQ-substrate campaigns** (`HF2Q_TQ_KV=1`, the production KV
      substrate; the graft splice encodes bank rows through the same
      hadamard kernel prefill uses — byte-identical to a prefill-chunk
      encode of the same rows, enforced by test): the full campaign
      matrix {SerialFifo, SlotAware} × {F32, TQ} is **ALL PASS**
      (zero-slot byte-identical, live bank diverges, disable restores,
      in every combo). The zero-slot no-op on TQ also proves the
      splice's substrate dispatch is a true no-op when absent. Each
      campaign's manifest records its `scheduler` + `tq_kv` fields.
    Remaining for this gate: the same canary on additional families as
    their splices ship.
5. **Position-offset equivalence**: ungrafted engine vs grafted engine
   with `n_slots=0` — bit-identical logits across prefill and decode,
   including spec-decode and kv_persist round-trips. The zero-slot
   canary above proves the serve-path equivalence end-to-end (output
   level); kv_persist is refused under graft at boot pending the
   graft-aware disk codec.
 6. **Throughput report**: decode and prefill tok/s, grafted vs
    ungrafted, median of multiple runs. Expected cost is attention over
    N extra slots at covered layers only (≈ a prompt N tokens longer at
    those layers); acceptance is "within the measured noise band or the
    delta is documented and accepted," never silently shipped.
    **MEASURED 2026-09-22** on Qwen3.6-35B-Abliterix-APEX Q5_K_M
    (TQ substrate, SerialFifo, speculation off, 333-token prompt,
    128-token completions, 5 runs/arm, A-B-A design with fresh process
    per arm; harness: `scripts/graft_probe/throughput.sh`):
    - **Decode: a 64-slot graft costs ~5% tok/s** (100.4 → 94.6 vs
      baseline, -4.3% vs the trailing baseline2) — OUTSIDE the measured
      machine-drift band (±1.6% A-vs-A after a 3-minute pre-warm) and
      mechanically consistent: the graft grows every covered layer's
      attention rows by 64/333 ≈ +19%, and the TQ read path (per-row
      Hadamard dequant) is a significant share of per-token time at
      full-attention layers. The cost scales as
      `n_slots / context_length` — ~0.8% at 8K context, larger at very
      short contexts. Documented and accepted: steering buys a decode
      tax proportional to graft size over context length.
    - **Prefill: within the noise band** — the A-B deltas straddle
      zero (+5.7% / +21.7%) inside a ±13% A-vs-A drift band; at a
      333-token prompt the TTFT is ~60 ms and client-side jitter
      dominates. Resolving a prefill delta needs a server-side
      chunk-timing harness or a much longer prompt; the mechanical
      expectation is the same proportional row growth at covered
      layers.
    - **Methodology (load-bearing)**: an un-warmed A-B-A showed +13.2%
      decode / +31.1% prefill A-vs-A drift — the host's GPU warm-up
      ramp swamps any graft effect for the first ~10 minutes of
      sustained load. The harness pre-warms ~3 minutes before the
      first measured arm; without that, ANY A-B throughput comparison
      on this host is noise.
 7. **Paired-arm behavioral panels** per the measurement doctrine:
    baseline / graft / GCD / graft+GCD, identical model artifacts,
    prompts, sampling settings, budgets; engagement, capability
    (GSM8K-class spot check — phantom measured a real cost here),
    completion, truncation, and KL reported separately. A null result is
    recorded honestly and closes the gate.
    **MEASURED 2026-09-22** on Qwen3.6-35B-Abliterix-APEX Q5_K_M with a
    64-slot SELF-DONOR bank (derived through hf2q's own donor prefill —
    `HF2Q_GRAFT_DERIVE` + `wrap_bank.py`, checkpoint-bound, proven to
    bind `Named` and shift the token stream in the output-level
    canary); harness `scripts/graft_probe/panel.sh` (TQ substrate,
    SerialFifo, speculation off, thinking off per request, temperature
    0, 15 GSM8K-class + 12 engagement probes per arm):
    - **Capability: no measurable graft cost** — baseline 9/15 vs
      graft 11/15; the ±2-item difference is ~1σ at n=15 (noise band
      ≈ ±1.9 items). Phantom's v3 TRAINED arm measured a real cost
      (45/75 → 27/75 on their stack); this v1-style donor bank shows
      none at this dose.
    - **Engagement: null** — safety refusals 0/3 in EVERY arm (the
      de-aligned Abliterix fine-tune's refusals are not restored by a
      generic compliance donor — an honest null for the strongest
      phantom-style claim); verbosity (373 vs 377 mean words) and
      concise-compliance (1/3 vs 1/3) unchanged.
    - **GCD arms**: the schema contract structurally constrains output
      (median 11–13 completion tokens, zero truncations — the grammar
      closes every response); capability under schema 7/15 (baseline)
      and 6/15 (graft) — the grammar's own cost, within noise of the
      plain arms' pairwise differences.
    - **Verdict: NULL behavioral result at this dose, recorded
      honestly.** The graft demonstrably shifts the token stream
      (output-level canary) but moves no behavioral metric on this
      probe set — consistent with phantom's own finding that the v1
      prefill_kv arm is their weakest (their engagement results needed
      the v3 trained banks). Closing the gate on the null; a TRAINED
      bank (phantom v3-style optimization against a target objective)
      is the named follow-up if behavioral steering is wanted.
    - **Harness lesson (recorded for reruns)**: the first campaign ran
      unbudgeted thinking and the model spent the whole 512-token
      budget in reasoning on complex probes — empty content, vacuous
      metrics. The panel sends `hf2q_enable_thinking: false` per
      request; the falsifier run's manifest is preserved.

## Evidence behind the design

- phantom-kv README + `docs/TECHNIQUE.md` §§6–7 (their measurements:
   scoreboard, persistence §6.9/§6.11, capability §6.10, judge-audit
   recall floors §6.10.1, pill matrix §7.5–§7.6) — quoted as their
   evidence, on their stack (Qwen3-4B dense, MPS/bf16).
- ADR-054 gate-3/4 null result (this repo, DeepSeek-V4 panel).
- hf2q source grounding: `src/inference/models/qwen35/kv_cache.rs`
   (FullAttnKvSlot, TQ encode, global append-only capacity),
   `src/inference/models/qwen35/mod.rs` (`Qwen35LayerKind`,
   `default_layer_types`, interval 4), `src/inference/models/gemma4/
   kv_cache.rs` (ring buffers, capacity=sliding_window),
   `src/inference/models/deepseek4/cache.rs` + `attention.rs` (128-token
   window + compressor), `src/serve/api/engine_qwen35.rs:11138`
   (`glp_cache_identity_tests` — the identity pattern to extend),
   `src/serve/multi_seq_kv.rs` (SlotId/SeqId discipline),
   `src/serve/sampler_pure.rs` (seed determinism).
- mlx-native surface: `ops::kv_cache_copy`, `ops::hadamard_quantize_kv`
   (splice + TQ encode need no new kernels).

## Open questions resolved / deferred

- **Graft × live compaction** (brief's open question): resolved per site
  in the matrix above — the v1 site's covered caches are append-only; the
  GraftRegion tag + compaction-skip tests are the enforcement seam for
  future paged/compacted caches and for the staged sites whose media is
  windowed or compressed.
- **Versioning against converted-GGUF checkpoints** (brief's open
  question): resolved by reusing ADR-053's conversion-receipt binding
  verbatim (output size + SHA-256 verified receipt supplies source
  identity; ancestors never do).
- **Per-request selection**: deferred with the policy rationale above.
- **Multi-site graft files** (one file, several site sections for one
  checkpoint): reserved for a spec v2 once two sites actually ship;
  v1's one-site-per-file keeps reader conformance testable.
