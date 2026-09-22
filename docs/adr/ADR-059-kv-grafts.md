# ADR-059: KV-cache grafts — context-space steering artifacts (`--kv-graft`)

- **Status:** Proposed — design recorded 2026-09-22; implementation gated
  behind the flags and gates below. No default behavior changes. This ADR
  is the hypothesis + design step of the Kata; every behavioral claim below
  is phantom-kv's measurement on their stack, not an hf2q result, until the
  paired-arm gates run here.
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

The intervention taxonomy has two shipped surfaces: **selection** (GCD —
masks logits at sampling) and **computation** (GLP — modifies activations
mid-forward). Missing: **context** — what the model conditions on.

Two facts motivate a second intervention class now:

1. ADR-054's measured null result: the calibrated `d_disp` direction at
   the FFN-writer site is behaviorally inert on the 48-prompt panel
   (28→28→28→29→29 across the dose ladder). The site-transfer warning in
   spec/GLP.md was confirmed. A structurally different mechanism is
   warranted, not another dose.
2. phantom-kv's mechanism is architecturally unlike GLP: no per-token
   hook, no 1-D direction assumption, nothing projected out of any
   activation. Steering is **attention-gated** — the model's own attention
   decides how strongly to weight the graft, per head, per token. Unload
   the graft and (with fresh inference state) the baseline is restored on
   byte-identical weights.

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
   **OPEN** — the paired-arm harness extension measures this.
7. **Paired-arm behavioral panels** per the measurement doctrine:
   baseline / graft / GCD / graft+GCD, identical model artifacts,
   prompts, sampling settings, budgets; engagement, capability
   (GSM8K-class spot check — phantom measured a real cost here),
   completion, truncation, and KL reported separately. A null result is
   recorded honestly and closes the gate. **OPEN** — requires a
   phantom-kv-derived bank (the synthetic canary bank proves
   participation, not behavior).

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
