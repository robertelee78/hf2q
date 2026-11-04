# ADR-063 Q4-GEMMA-SPEC-SERVE: serving-side speculative decoding for Gemma 4 26B-A4B with the Gemma 4 assistant (MTP) drafter — HLD/LLD

Status: design for approval (2026-11-04). Revision 2 applies the
ACCEPT-WITH-CONDITIONS review (ReviewGemmaSpec, findings 1–9). The review's
recommendations, adopted by the root, are recorded in §13. No code, build or
hardware run backs this document. Every number is quoted from a retained
receipt or source line, with its path, or is marked `[INFERENCE]` (estimates
and arithmetic included).

Story: **Q4-GEMMA-SPEC-SERVE** (new, v0.1.30, operator decision 2026-11-04).
It is one of four stories that close the Gemma 4 gap left after #668
(Q4-GEMMA-ATTN). Its predecessors are Q4-GEMMA (#660, the profile) and Q5
(#371). Q4-GEMMA-ATTN's closing campaign consumes its configuration.

Sources:
- ADR-063 sections Q4, Q4-GEMMA, Q4-GEMMA-ATTN and Q5; the Q1 numeric policy
  `hf2q.q1.numeric-policy.v1` (`docs/research/adr063-q1-numeric-policy.md`).
- The #660 result, `docs/research/adr063-q4-gemma-profile-result.md` (branch
  `feat/q4-gemma-profile`, PR #687), and its receipts under `$G`.
- ADR-065 (Qwen joint speculative verify) and the J1 design
  (`docs/research/adr065-j1-design.md`), for the K3 transaction pattern.
- ADR-034 P7 and its Workstream C/D tables (Gemma DFlash and the deferred
  `-assistant` path).
- hf2q `main` at `ff4af4bd` and mlx-native 0.15.17 (`/opt/mlx-native`), read
  only.
- The HF checkpoint `google/gemma-4-26B-A4B-it-assistant@6e5aaaf4`
  (`config.json`, `generation_config.json` and the safetensors header, local
  copy `/Users/robert/.local/state/hf2q/ext/models/q4-gemma/src/MT`). Also the
  HF reference code `transformers/models/gemma4_assistant/modeling_gemma4_assistant.py`
  (main) and `docs/source/en/model_doc/gemma4_assistant.md` at `effde209`.
- The peer reference tree at `e86f5fd1`: `src/models/gemma4-assistant.cpp`,
  `src/models/gemma4.cpp`, its model-registry source (memory construction),
  hparams header and graph builder, `common/speculative.cpp` and
  `conversion/gemma.py`.

`$G` = `/Users/robert/.local/state/hf2q/v0130/q4-gemma`.

## 1. Decision summary

1. **What ships.** hf2q's Gemma 4 serving worker drafts with the Gemma 4
   assistant. Each speculative tick has:
   - one drafter pass of K greedy steps for every eligible slot at once;
   - **one** joint target verify forward holding every slot's window (K+1 rows
     for a speculating slot, 1 row for an ordinary one);
   - the ordinary tick's per-token commit sequence, walked row by row in
     order (greedy exact match);
   - commit, or rollback of the rejected rows.

   The depth K is chosen by M0 from {1, 2, 3} (§8.1). **K = 1 is measured
   first and is the working value.** Then a 4-slot verify has 8 rows, so no
   dense or head grouping change is needed and the sliding layers run only
   two phases. The peer's best Gemma arm (`draft-mtp`, n-max 3, greedy
   drafts) is the model for this transaction. Today hf2q has it only in the
   offline `generate` paths.
2. **The verify route is Class A, bitwise equal to the serial M1 decode.**
   The Gemma batched decode body is built from row-identical dispatch:
   - quantized dense projections run the m=N `mul_mv` at N ≤ 8;
   - routed MoE is pinned to the per-token `mv_id` route;
   - attention runs per query
   (`src/inference/models/gemma4/batched_body.rs:440-490, 2053-2063`).

   By construction, therefore, its rows do not depend on the batch. The
   `slot_aware_n1`/`n4`/`n8` tests check this only on greedy outputs (text
   and logprobs) at short positions, not on bitwise logits, so A1 (§7.3)
   carries the bitwise proof. The verify route (R-A) is that body run over
   *virtual rows* `(slot, position)`, laid out **rank-major**:
   - quantized dense projections in groups of ≤ 8 rows (unchanged at K = 1);
   - MoE per token;
   - sliding-layer KV write and attention in rank-ordered phases, each over
     a contiguous row slice, so no row reads a ring slot that a later row of
     its own window has overwritten.

   Each verified row's logits therefore equal the row M1 would have produced,
   so speculation on versus off must give `ExactTranscripts`. Quality is
   inherited (Q1 §3.6), and there is no Class B surface. This is the main
   difference from Qwen K3, whose verify graph is a Class B route (ADR-065
   :89).
3. **Rollback is exact at the cursor.**
   - Full-attention layers are append-only: rejected rows ≥ the cursor are
     never read and are overwritten later.
   - The sliding ring holds exactly 1,024 rows (`kv_cache.rs:373`), so verify
     rows `pos0+1..pos0+K` overwrite history that the next decode still
     needs. Before the verify, the round saves those ring rows' pre-images on
     the GPU, using mlx-native's `dispatch_kv_cache_linearize_ring_bytes`
     (`src/ops/kv_cache_copy.rs:961`). After acceptance it restores the
     rejected rows' pre-images, using a new byte-exact inverse that takes a
     source row offset (§5.5).
   - The Class A store gate compares what is **readable at the cursor**: full
     layers `[0, c)`, the ring's logical window `[c−1023, c−1]`, the cursors,
     and the next eight decodes, all against the serial path after the same
     tokens.
4. **The drafter is a sidecar, not a runtime.**
   - `gemma4-assistant` stays out of the primary `CONTRACTS` (the kata test
     `src/inference/gguf_contract.rs:81-86` keeps passing).
   - It gains a drafter contract that loads only beside a paired `gemma4`
     target, behind a fail-closed pairing guard.
   - hf2q's converter learns `Gemma4AssistantForCausalLM` and emits Q8_0 with
     the peer-compatible tensor names. The released configuration obtains
     the artifact through managed download, pinned by sha256 (§4.4).
   - The 26B-A4B checkpoint has **no centroid-masked head**: its config has
     `use_ordered_embeddings: false` and the safetensors carry no
     `masked_embedding.*` tensors. The head is the dense tied 262,144×1,024
     embedding. A checkpoint with `use_ordered_embeddings: true` (the
     E2B/E4B assistants) is refused at conversion, never approximated (§3.3).
   - The drafter's step-0 input is the post-final-norm row that the target's
     head already computes on the GPU. The fused head (the production
     default for this Q6_K-head target) stays on, and its normed operand is
     retained for the drafter (§5.2). Nothing new crosses to the host.
5. **The deciding measurement comes first.** ADR-034 measured MoE batched
   verify as not amortizing (Qwen 35B-A3B MoE spec at 0.83× base; Gemma
   26B DFlash at 0.40× base, single slot). On Gemma, the peer's own MTP
   saves only −0.48 s of decode and −0.35 s of makespan (#660 §1).

   So the implementation starts with the R-A verify and its transaction,
   driven by forced drafts, and measures the **round-cost curve M0**. The
   drafter's cost is bounded by a model-free synthetic bench of its exact
   shapes, and acceptance by a frozen L(K) model fitted to the peer, so G0
   is computable before any drafter runtime exists. Gate G0 (frozen in §8.1)
   stops the story with a recorded falsification if the projected saving is
   below 0.40 s per run. It is re-checked on hf2q's own acceptance (M1).

   On a miss, the windowed verify code is deleted unless a zero-cost drafter
   (n-gram or prompt lookup) is opened to reuse it. No Class B alternative
   is in scope (§13, D1).
6. **Switch.**
   - `--gemma-assistant <gguf>` supplies or overrides the drafter.
   - `HF2Q_GEMMA_SPECULATION` takes `0`/`off` (the kill switch) or `mtp`.
   - Before promotion, unset means off. After promotion, unset means `mtp`
     whenever an assistant is resolved.
   - Only greedy requests speculate. The Gemma requests reach temperature 0
     through the server default (`req.temperature.unwrap_or(0.0)`,
     `src/serve/api/handlers.rs:2399`), since the workload omits the field.
   - Repetition penalty, logit bias, grammar and tool triggers are honoured
     row by row, through the same per-token commit sequence the ordinary
     tick runs (§5.4).

## 2. Evidence

### 2.1 The gap and this story's share (#660, L1, 7 paired blocks)

From `$G/profile/l1/compare-hf2q-vs-peer.json` (#660 result §2):

| Stage | hf2q s | peer MT s | Δ s (95% CI) | Share |
|---|---:|---:|---:|---:|
| makespan | 14.867 | 12.477 | +2.734 (+2.285, +3.072) | — |
| decode | 9.203 | 7.958 | +1.170 (+0.961, +2.012) | 0.43 |

A Q4 Win needs about 11.85 s, which is about 3.0 s below hf2q today
(#660 §7). #660 §7 books this lever at **0.4–0.5 s**: "Gemma 4
assistant-drafter MTP … in hf2q serving at 4 slots … MT vs P0 on the peer:
−0.48 s decode and −0.35 s makespan (CI includes 0)".

### 2.2 The peer's arm

Case `$G/cases/peer/G-MT-mtp-assistant.json`: `--spec-type draft-mtp
--spec-draft-model {MT} --spec-draft-n-max 3 --spec-draft-n-min 0
--spec-draft-p-min 0 --spec-draft-sampling greedy
--spec-draft-backend-sampling`, with an F16 unified KV. The drafter
`e1145ee2…` is BF16 (855,228,512 bytes, `$G/MT.receipt.json`).

Run `$G/profile/l1/002-b01-peer-20261104T200604Z/server.stderr`, aggregated
over all `print_timing` lines [arithmetic]:
- 1,452 drafted and 1,151 accepted (0.793 per drafted; the #660 median is
  0.79);
- 1,643 decode tokens over about 492 verify rounds;
- **L ≈ 3.34 tokens emitted per round** (2.34 accepted per verify, #660 §7);
- per-request acceptance ranges from 0.81 to 0.94 on the long-decode turns
  (`mean len` 3.4–3.8).

The peer logs only aggregate acceptance at n-max 3, not acceptance per
depth, so L(1) and L(2) are not measured. §8.1 therefore freezes a model for
them, and M1 replaces it with hf2q's own measurement.

hf2q's matching run (`001-b01-hf2q-…`) emitted about 1,657 completion tokens
[arithmetic from the client usage records]. Tokens per agent are therefore
about equal, and decode per output token is about 22 ms per agent at four
slots [INFERENCE: 9.20 s / ~414 tokens].

**Implied peer round cost.** A 5.7% decode saving (−0.48 s of about 8.44 s)
at L = 3.34 implies the peer's round costs about 3.15× one ordinary step
[INFERENCE]. Even the peer's amortizing `mm_id` verify barely pays at four
slots, which is why §8.1 measures cost before building the rest.

### 2.3 hf2q today (source facts)

| Fact | Where |
|---|---|
| `gemma4-assistant` has no runtime contract, by test | `src/inference/gguf_contract.rs:16-35, 81-86` |
| HF `Gemma4*` architectures map to `ArchName::Gemma4` by prefix, so `Gemma4AssistantForCausalLM` would be misrouted | `src/core/model_arch.rs:26, 43-50` |
| The converter maps Gemma 4 names (`layer_scalar` → `layer_output_scale`, `q_norm`, `rope_freqs`, `shared_kv_layers`, `sliding_window_pattern`, `key_length_swa`) but has no `pre_projection`/`post_projection` | `src/convert/arch/gemma4.rs:192-335, 458-801` |
| Gemma speculation exists only in offline `generate` (DFlash 26B, EAGLE-3 31B, n-gram), on global non-slot caches, temperature 0 only | `src/serve/spec_decode_cli.rs:85, 662, 817`; `src/serve/mod.rs:1375-1442, 2489` |
| Serving worker `run_slot_aware_gemma4`; decode step `decode_batch_gemma4` → `forward_decode_body_batched` → head → per-row `decode_tick_finalize` | `src/serve/api/engine.rs:11973, 16371, 7788`; `batched_body.rs:2362`; `batched_head.rs:58` |
| **The lm head is fused into the body by default** (opt out `HF2Q_FUSE_LMHEAD=0`). It applies with `cb_chunks ≥ 2` (default 4) and a Q6_K head, and the target's tied `token_embd` is Q6_K. When fused, the body returns no hidden rows (`Vec::new()`). Its normed operand `_normed_b` stays on the GPU. The comments saying "`=1`" (`batched_body.rs:2368`, `engine.rs:16445, 16810`) are out of date | `batched_body.rs:2506-2524, 2589, 2601` |
| Per-tick bookkeeping outside finalize: `valid_tokens += 1` per written input token (it feeds `retained_prefix` and warm resume); `commit_gemma4_decode_cursors` (a +1 step); `scheduler.advance_after_decode` once per token (auto-release at `max_tokens`); stop at the first failed emit; finish/cancel handling (retained prefix, `record_retained_slot_kv`, recovery from `seq_lens` and anchors) | `engine.rs:16605-16615, 7701-7707, 17218, 16930, 16942`; `src/serve/scheduler.rs:1565-1599` |
| Batched flash takes per-query slot id and position, so several queries may address one slot. It is used only when all queries share an `(nwg, nsg)` bucket; otherwise it falls back to per-query dispatch | `batched_body.rs:880-901, 1031-1086` |
| Hybrid KV (default): F16 K, TQ-HB V (FWHT-rotated) plus F32 norms; sliding layers are a ring of exactly `sliding_window` = 1,024 rows; full layers are linear | `src/inference/models/gemma4/kv_cache.rs:373, 1133` |
| Cursor install can move a cursor back; commit refuses to | `engine.rs:17131, 17200` |
| The head's host payload is logits only, by test | `batched_head.rs:28-46` |
| Gemma's default repetition penalty is 1.05, so sampler params are always set and finalize runs `sample_logits_with_grammar` (greedy-with-penalty at temperature 0) | `src/serve/operator_settings.rs:166-170`; `engine.rs:575-640, 7534-7546` |
| The four-agent Gemma workload omits `temperature`; the server default makes it 0 | `src/serve/api/handlers.rs:2399` (`req.temperature.unwrap_or(0.0)`); `scripts/four_agent/family_peer.py:51` |
| Qwen K3 transaction, planner, cost gate and telemetry | `src/serve/api/engine_qwen35.rs:5760-6149, 4729`; `src/inference/spec_decode/cost_controller.rs:9-10`; `src/serve/api/qwen35_speculation.rs:330-458` |

### 2.4 Why the verify cost is the risk

ADR-034's Workstream A and C tables
(`docs/ADR/ADR-034-speculative-decode-end-to-end.md`):
- Qwen 35B-A3B MoE MTP: "T_v(N)/T_v(1) ratio = 2.4× … MoE batched-verifier
  overhead doesn't amortize".
- Gemma 26B DFlash: 0.40× base.

Under R-A, MoE stays per token. Its expert reads scale with rows unless the
sibling Q4-GEMMA-DECODE expert-grouped `mv_id` lands; that design is bitwise
equal to stock and admits up to 16 tokens × top-8, as agreed with that
design's author on 2026-11-04. The round-cost curve (§8.1) decides.

## 3. The assistant model (verified)

### 3.1 Checkpoint facts

The HF `config.json` of `6e5aaaf4`:
- `Gemma4AssistantForCausalLM`, `model_type: gemma4_assistant`.
- `backbone_hidden_size: 2816`, `num_centroids: 2048`,
  `centroid_intermediate_top_k: 32`, `use_ordered_embeddings: false`,
  `tie_word_embeddings: true`.
- `text_config`:
  - hidden 1,024, 4 layers `[sliding ×3, full]`, `num_kv_shared_layers: 4`;
  - 16 Q heads, head_dim 256 (sliding) and 512 (full);
  - `num_key_value_heads: 8`, `num_global_key_value_heads: 2`,
    `attention_k_eq_v: true`;
  - dense GELU-tanh MLP 8,192, `enable_moe_block: false`;
  - `sliding_window: 1024`;
  - RoPE: sliding θ=1e4; full proportional, θ=1e6,
    `partial_rotary_factor 0.25`;
  - `final_logit_softcapping: null`, vocab 262,144.

`generation_config.json`: `num_assistant_tokens: 6`, constant schedule. The
peer runs n-max 3. This design measures K ∈ {1, 2, 3}, with K = 1 as the
working value (§5.7).

Safetensors header, 48 tensors, all BF16 (the peer's BF16 GGUF has 49,
including a synthesized `rope_freqs`):
- `model.embed_tokens.weight [262144,1024]`, `model.norm.weight`;
- per layer: `input_layernorm`, `q_proj`, `q_norm`, `o_proj`,
  `post_attention_layernorm`, `pre_feedforward_layernorm`,
  `mlp.{gate,up,down}_proj`, `post_feedforward_layernorm`, `layer_scalar`;
- `pre_projection.weight [1024,5632]`, `post_projection.weight [2816,1024]`.

There is **no** `k_proj`, `v_proj`, `k_norm`, `lm_head` or
`masked_embedding.*` tensor.

### 3.2 Graph (the HF reference, the peer, and this design)

Per draft step, for one slot with last token `x` at position `pos0` (the
position of the token the target has sampled but not yet processed):

1. `e = target.embed_tokens(x) · √2816`. This is the *target's* embedding
   table; the peer uses `model_other->tok_embd` scaled by
   `sqrtf(n_embd_backbone)`, `gemma4-assistant.cpp:117-118`.
2. `u = concat(e, h)` [5632]. Here `h` is, at step 0, the target's
   **post-final-norm** hidden state of the row that predicted `x`
   (`gemma4.cpp:413-422`: "matches the reference (transformers/vLLM/SGLang)").
   At step i ≥ 1 it is the drafter's own `h_next` from step i−1, raw.
3. `z = pre_projection · u` [1024].
4. Four layers, each:
   - `n = rms(z; attn_norm)`, `q = W_q n`, `q = rms_head(q; q_norm)`,
     `q = rope(q, pos0)`;
   - **no K/V projection**: `a = attn(q, K_t, V_t)` over the target's cache
     (§3.4);
   - `z = z + rms(W_o a; post_attention_norm)`;
   - `z = z + rms(W_down(gelu_tanh(W_gate m) ⊙ W_up m); post_ffw_norm)` with
     `m = rms(z; ffn_norm)`;
   - `z = z · layer_scalar`.
5. `y = rms(z; output_norm)`.
6. `logits = E_assistant · y` (tied, 262,144 × 1,024, dense), then the draft
   token `d = argmax(logits)`.
7. `h_next = post_projection · y` [2816].

Attention scale is 1.0 (`f_attention_scale = 1.0`, peer `:12`), with no
softcap. HF: `projected_state = post_projection(last_hidden_state)`; logits
are `lm_head(last_hidden_state)` unless `use_ordered_embeddings`.

### 3.3 The centroid-masked head is absent here

The HF `Gemma4AssistantMaskedEmbedder` (2,048 centroid logits, top-32
clusters, gather of 32 × 128 candidate embeddings, scatter with
`min − 1` fill) is built only `if self.config.use_ordered_embeddings`. For
this checkpoint that flag is false and the tensors are absent.

The peer declares `masked_embd_centroids`/`masked_embd_ordering` as
`TENSOR_NOT_REQUIRED` and never uses them in its graph
(`gemma4-assistant.cpp:39-40, 196`).

Design rule: the converter refuses `use_ordered_embeddings: true` with
"Gemma 4 assistant ordered-embedding (centroid) head is not supported". The
drafter contract refuses any `masked_embd_*` tensor. No silent dense
fallback is allowed for such checkpoints.

### 3.4 KV sharing, position and mask

**Layer mapping.** Sliding drafter layers read target layer `n_layer − 2` =
28 (sliding, 8 KV heads, d256). The full layer reads `n_layer − 1` = 29
(full, 2 KV heads, d512). On layer 29 `attention_k_eq_v` means V is produced
by the K projection, but the **V cache differs from the K cache**: V gets a
weightless RMS norm and no RoPE. The drafter reads layer 29's V cache, never
its K cache as V.

Sources:
- the peer's model-registry memory construction (`share = is_swa(il) ?
  n_layer(other) − 2 : n_layer(other) − 1`, lines 3274-3285);
- HF `shared_kv_states["sliding_attention"|"full_attention"]`, "the last
  layer of each `layer_type`".

hf2q's target has full layers at `(i+1) % 6 == 0` (`src/serve/config.rs`
test `:490`), so 28 is sliding and 29 is full.

**Position.** Every draft step uses position `pos0`. The HF doc says "the
assistant predicts all tokens from the same position ID"; the peer uses
`batch.add(id, dp.pos0, …)` (`common/speculative.cpp:1881-1886`).

**Visible keys.** The target's cached rows `p < pos0`. On sliding layers the
target's own sliding mask is evaluated at `pos0`: `pos0 − p < 1024`, so
`p ∈ [pos0−1023, pos0−1]`, which is 1,023 keys (the peer's `is_masked_swa`,
`STANDARD` type, hparams header lines 519-528). In the peer, the token at
`pos0` has not yet been processed, and rejected rows are removed after
verify. [INFERENCE] The HF `flip`ped bidirectional sliding mask over a
sliding cache that holds 1,023 past rows gives the same set. Test D1 (§7.3)
pins it against the HF reference.

**Mapping onto hf2q's kernel.** The batched `flash_attn_vec_hybrid` derives
each query's extent from its `seq_pos` as `sp + 1`. It uses
`params.kv_seq_len` only to pick nwg (`flash_attn_vec_hybrid.metal:807-818`,
`flash_attn_vec_hybrid.rs:343`). The drafter therefore:
- passes **`seq_pos = pos0 − 1`** for attention, so the full layer reads
  `[0, pos0)` and never the stale row at `pos0`;
- passes **`sliding_window = 1023`** on sliding layers. At `pos0 ≥ 1024`
  this gives `ksl = 1024` and ring start `pos0 mod 1024`, and
  `window_start = 1` masks position `pos0 − 1024`, so the keys are exactly
  `[pos0−1023, pos0−1]`. Below 1024 every cached row is visible;
- keeps a separate RoPE positions buffer holding **`pos0`**.

**Exactness scope.** This choice affects acceptance only. A drafter
difference can never reach committed output (§6).

**Writes.** The drafter never writes target state: no KV write, no cursor
move (in the peer's graph builder, `if (store_kv && k_cur)` never fires, line 3551).

### 3.5 Sizes (arithmetic)

- Parameters:
  - head: 268.4 M;
  - sliding layer: 33.6 M each; full layer: 41.9 M;
  - pre projection: 5.8 M; post projection: 2.9 M;
  - **total: about 420 M**.
- Weights: BF16 about 840 MB (the peer file is 855 MB); Q8_0 about 446 MB
  (the hosted `MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf` is 461,766,816 bytes,
  `docs/research/hosted-resolution-kata-2026-09-05.md:36`).
- Per draft step at four slots, the drafter reads its weights about once
  (mat-vec, m ≤ 4): about 446 MB in Q8_0, of which the head is 64%.
  [INFERENCE] At about 450 GB/s that is about 1 ms per step, which is
  ≤ 0.05 of one ordinary step. §8.1 bounds it with a synthetic bench rather
  than relying on this estimate.
- Scratch per step is small: 4 × 262,144 × 4 B = 4 MiB of logits plus
  activations.

## 4. Conversion and loading (LLD)

### 4.1 Converter

- `src/core/model_arch.rs`: a `gemma4_assistant` model type and the
  `Gemma4AssistantForCausalLM` architecture map to a new
  `ArchName::Gemma4Assistant` (`"gemma4-assistant"`,
  `src/quantize/ggml_quants/tensor_ref.rs`). The match arm precedes the
  `starts_with("Gemma4")` prefix arm. A unit test pins that the assistant is
  not routed to `Gemma4`.
- New `src/convert/arch/gemma4_assistant.rs`. It reuses the gemma4 name map
  for the shared names (`attn_q`, `attn_q_norm`, `attn_output`,
  `attn_norm`, `post_attention_norm`, `ffn_norm`, `post_ffw_norm`,
  `ffn_{gate,up,down}`, `layer_output_scale`, `token_embd`, `output_norm`)
  and adds `pre_projection` → `nextn.pre_projection.weight` and
  `post_projection` → `nextn.post_projection.weight`. These are the peer's
  GGUF names (`gguf-py/gguf/constants.py:2175-2176`), so a hf2q artifact is
  also loadable by the peer (decision D4).
  - Refuse: `use_ordered_embeddings: true`, any `masked_embedding.*` tensor,
    any `k_proj`/`v_proj`, and `num_kv_shared_layers ≠ num_hidden_layers`.
- Metadata, under `gemma4-assistant.`:
  - the gemma4 key set: block_count, embedding_length, head counts, the
    head_count_kv array, key/value lengths (global and `_swa`),
    sliding_window, `sliding_window_pattern`, `shared_kv_layers`, rope bases
    and dimension counts;
  - plus `embedding_length_out` = 2816 and `nextn_predict_layers` = 4 (as
    the peer converter, `conversion/gemma.py:955-972`);
  - `rope_freqs.weight` synthesized as for gemma4 (`gemma4.rs:859`).
- Quantization (decision D2):
  - `q8_0` (`QuantSelector::from_name("q8_0")`, `quant_selector.rs:133`)
    applies to every 2-D weight, including the tied embedding/head;
  - norms, `layer_output_scale` and `rope_freqs` stay F32;
  - the stored-evidence v1 path admits q8_0 (`stored_evidence.rs:123-127`).
- The tokenizer is copied from the checkpoint. The pairing guard (§4.2)
  compares it with the target's.
- The artifact gets a receipt: source revision, safetensors sha256 (`006c8713…`
  per `$G/MT.receipt.json`), converter commit, output sha256.

### 4.2 Drafter contract and pairing guard

`gguf_contract.rs` gains `DRAFTER_CONTRACTS`, with one entry
`gemma4-assistant` paired to `gemma4`, and
`drafter_contract(arch) -> Option<&DrafterContract>`. The kata test keeps its
assertion and gains
`drafter_contract("gemma4-assistant").is_some() && runtime_contract(..).is_none()`.
`validate_architecture_for_load` (`engine.rs:2431`) is unchanged, so an
assistant passed as `--model` is still refused.

New `src/inference/models/gemma4_assistant/admission.rs`. It refuses at load,
fail-closed, naming the mismatch, unless all of these hold:
1. `embedding_length_out` = target `hidden_size` (2816).
2. Vocab sizes are equal. If the assistant carries `tokenizer.ggml.tokens`,
   the token-list digest equals the target's. The DS precedent is one
   differing token text (#660 §1); the ids must mean the same tokens.
3. Layer pattern `[sliding ×3, full]`. Target layers `n−2` and `n−1` are
   sliding and full.
4. The drafter's sliding and full head_dim equal the target's KV head_dim
   (256, 512). Its Q heads are divisible by the target's KV heads (16:8,
   16:2).
5. Equal `sliding_window`. Rope bases equal the target's per layer type, and
   `rope_freqs` is equal bitwise.
6. Tensor inventory and shapes are exactly §3.1 in GGUF names. Dtypes are
   Q8_0 or BF16 for 2-D weights and F32 for 1-D.
7. The target runs:
   - hybrid KV (`HF2Q_HYBRID_KV` on) with TQ-HB V, so not
     `HF2Q_FULL_F16_KV=1`, which changes the V payload the drafter reads and
     the ring rows it saves;
   - the batched decode body (`HF2Q_BATCHED_BODY` not `0`);
   - not `HF2Q_DFLASH_XLEN_SDPA`.

   Either head path is accepted: the fused head (default, Q6_K) or
   `HF2Q_FUSE_LMHEAD=0`. Both retain their normed operand for the drafter
   (§5.2), so both S1 arms run the same head path as production. The FullTq
   regime, the serial decode path and the F16-V regime are refused with
   "Gemma speculation requires the hybrid batched decode body".
8. No KV graft is bound **on the model**. The graft is model-wide
   (`model.rs:806-811`), so a bound graft refuses `mtp` at load, not per
   lane.

### 4.3 Runtime objects

- `Gemma4AssistantConfig::from_gguf` (new; `Gemma4Config::from_gguf` keeps
  `ensure!(arch == "gemma4")`).
- `Gemma4AssistantWeights`: GPU tensors uploaded on the target's
  `GpuContext` device.
- `Gemma4AssistantScratch`, sized for `max_slots` rows: pre/post buffers,
  Q, attention output, MLP, logits `[max_slots, vocab]`, argmax ids
  `[K, max_slots]` and `h_next [max_slots, 2816]`.
- `LoadOptions` (`src/serve/mod.rs:4030-4041`) gains
  `gemma_assistant_path: Option<PathBuf>`, threaded from a new serve flag
  `--gemma-assistant <PATH>`, as `kv_graft_path`/`glp_path` are.
  `GemmaLoadedModel` owns `Option<Gemma4Assistant>`.
- The assistant is loaded after the target and before readiness. Its
  sha256 and contract verdict are logged and published.

### 4.4 Distribution (decision D3: managed download, sha256 pin)

- The hf2q-built Q8_0 artifact (§4.1) is published as a managed artifact,
  in an hf2q-owned location fixed at release time, with its receipt.
- Its sha256 is pinned in hf2q's managed-artifact table beside the target's
  pairing (`src/serve/managed_artifacts.rs:344`; the pair-guard precedent is
  `automatic_pair_guard_authority_paths`, `src/serve/mod.rs:4328`).
- When serving a target whose managed entry names an assistant, hf2q
  resolves and verifies the assistant artifact. A digest mismatch refuses
  `mtp` at load.
- `--gemma-assistant <PATH>` overrides the managed resolution. It is the
  only path the Q5 cases use, so every receipt binds an explicit file and
  sha256.
- The promotion commit (§8.6) cannot land until the artifact is published
  and its pin is merged, because the Q4 Win counts only a published
  configuration.

## 5. Serving round (HLD → LLD)

### 5.1 Tick composition

`decode_batch_gemma4` (`engine.rs:16371`) partitions the decoding slots:
- **speculating windows**: eligible (§5.7) slots, each with a window
  `[x, d0, …, d_{K−1}]` at positions `pos0 .. pos0+K`;
- **ordinary rows**: everything else, width 1.

A tick with no speculating window is today's tick, byte for byte (A5). The
scheduler is unchanged: `Mixed` still runs decode first and then one prefill
quantum (`engine.rs:13178-13230`).

Round, for a tick with at least one speculating window:
1. **Entry**, per lane. Check eligibility. Check that the cursor == `pos0`
   and that a carry row index is present (§5.2). On failure the lane stays
   ordinary.
2. **Draft pass** (§5.2): one command buffer of K steps for all speculating
   lanes. Read back the `K × lanes` draft ids.
3. **Pre-image save** (§5.5) for lanes whose ring has wrapped over the
   window.
4. **Verify forward** (§5.3) over all rows in rank-major layout, with the
   head (fused by default) over all rows.
5. **Commit walk**, per lane (§5.4): the ordinary tick's per-token commit
   sequence, row by row, with finish effects deferred.
6. **Rollback**, then deferred finishes: ring restore (§5.5), cursor commit,
   then the deferred finish and release effects (§5.4).
7. **Accounting** (§5.9).

### 5.2 Draft pass

- **Carry row.** Both head paths compute the post-final-norm rows on the
  GPU:
  - the fused head's `_normed_b` (`batched_body.rs:2589`), which today is
    discarded;
  - `lm_head_batched`'s `normed_b` (`batched_head.rs:92-147`).

  In `mtp` mode, the head keeps that `[R, 2816]` F32 buffer as the
  worker's **carry buffer** until the next tick's draft pass ends. It is
  double-buffered, so the next tick's head does not overwrite it while the
  drafter reads it. Each lane records the row index of its last committed
  position, the row that predicted its `next_token`.
  - Nothing crosses to the host. The logits-only host-payload gate
    (`batched_head.rs:28-46`) is unchanged, and so is the fused head.
  - Ordinary ticks in `mtp` mode retain the buffer too, so a lane can draft
    on the tick after any ordinary tick.
  - A lane's first decode tick after any prefill is ordinary, because the
    prefill paths do not retain their last row's normed operand. Prefill code
    is therefore not touched, and the cost is one ordinary tick per request
    (28 of about 1,650 emitted tokens per run [arithmetic]).
  - Under `HF2Q_GEMMA_SPECULATION=0`, the buffer is not retained. That is
    today's behaviour exactly, and it allocates nothing.
- **Step 0 inputs.** The tokens are each lane's `next_token`. The rows are
  gathered from the carry buffer by the lanes' row indices
  (`dispatch_gather_f32`, `src/ops/gather.rs:58`). This is exactly the
  post-final-norm row the drafter needs (§3.2), with no re-norm and no
  upload.
- **Each step**, for all lanes in one `[lanes, ·]` batch:
  - target `embed_tokens` gather × √2816 (`model.rs:826`) from a GPU token
    buffer;
  - `dispatch_feature_concat_f32` ×2;
  - the drafter graph of §3.2, using `quantized_matmul_ggml` (Q8_0, m ≤ 4,
    mv) or `dense_gemv_bf16_f32` for BF16;
  - `fused_head_norm_rope_f32` with the full layer's `freq_factors`, using
    the RoPE positions buffer holding `pos0`;
  - `flash_attn_vec_hybrid_batched` over the lane's target layer 28 (K and
    V caches) or layer 29 (K cache, and its **separate V cache**). It uses
    per-query slot ids, attention positions buffer **`pos0 − 1`**,
    `sliding_window` 1023 on sliding layers, scale 1.0 and softcap 0
    (§3.4), followed by the FWHT sign undo the target applies after hybrid
    SDPA (`batched_body.rs:1092`);
  - the raw `fused_gelu_mul` pipeline as the target uses it
    (`forward_gpu.rs:1932`);
  - output norm, tied head, `dispatch_argmax_f32` into `ids[step]`, and the
    post projection into `h_next`.

  Step i+1 gathers its embedding from `ids[step i]` on the GPU, so the K
  steps need no host sync.
- **Drafts.** Raw drafter argmax, greedy, the same as the peer's
  `--spec-draft-sampling greedy`. No penalty or grammar is applied to drafts:
  they are proposals, and the target decides.
- **No target-state writes.** The drafter's command buffer declares reads
  only on target KV buffers and the carry buffer (`barrier_between`). A3
  proves the target state is unchanged.

### 5.3 Verify route R-A

`forward_decode_body_batched` is generalized to windows. It keeps its name
and its w=1 dispatch sequence:

```rust
pub(crate) struct DecodeWindow { pub slot: SlotId, pub start_pos: usize, pub tokens: ArrayVec<u32, 4> }
pub(crate) struct WindowRows { pub rank_offsets: ArrayVec<usize, 5>, pub row_of: Vec<ArrayVec<usize, 4>> } // row_of[window][rank]
pub(crate) fn forward_decode_body_batched(&self, windows: &[DecodeWindow], regime: &mut BatchedKvRegime<'_>,
    head_out: &mut Option<BatchedHeadOut>, carry: Option<&mut GemmaCarryBuffer>, gpu: &mut GpuContext)
    -> Result<(Vec<f32>, WindowRows)>
```

**Row layout: rank-major.**
- Rows `[rank_offsets[j], rank_offsets[j+1])` hold the rank-j row of every
  window that has one, in window order.
- Rank 0 is every lane (speculating and ordinary), so the rank-0 slice is
  exactly today's one-row-per-slot batch in today's order.
- Each phase's rows are one contiguous slice. The batched KV-encode, flash
  and FWHT-undo kernels take contiguous `[n, …]` row buffers with per-row
  position and slot-id buffers (`batched_body.rs:958-1102`), so each phase
  is a `slice_view` with no gather.
- `WindowRows::row_of` maps `(window, rank)` to the row, for the head
  readback and the walk. U8 asserts the layout and the map.

Per layer:

- **Position-independent ops** (norms, fused MLP, MoE, post norms) run
  batched over all R rows exactly as today. MoE stays on the per-token
  `mv_id` route, which is row-identical at any N, and nothing switches to
  `mm_id` at 16 rows (`batched_body.rs:2053-2063`; `ggml_capability.rs:987-999`).
  It moves to Q4-GEMMA-DECODE's bitwise grouped `mv_id` if that lands.
- **Dense projections.** At K = 1 and four lanes, R ≤ 8, so
  `dispatch_dense_rowident` (`batched_body.rs:451`) is used unchanged: the
  m=N `mul_mv`, byte-identical to m=1 while m ≤ `MM_ROUTING_THRESHOLD`
  (`:440-447`). Byte-parity tests for m = 2..8 against m = 1 exist for Q4_K
  and Q6_K, and Q5_0 and Q8_0 are identical by construction (same kernel,
  `grid.y = m`). Only if M0 selects K ≥ 2, so that R > 8: the n > 8 branch
  for quantized weights changes from the per-row m=1 loop to m ≤ 8 groups.
  F16, F32 and BF16 weights keep the per-row loop.
- **Q/K norm and RoPE**: one batched dispatch over all rows with per-row
  positions, as today.
- **Full layers.** KV-encode all R rows (append-only; each row's K/V bytes
  are its own), then one batched flash over all R queries with per-query
  slot id and position. Each query's extent is `pos_r + 1`, so its keys
  `≤ pos_r` are exactly the bytes M1 would have written. The batched kernel
  is used only if every query shares the `(nwg, nsg)` bucket. Otherwise each
  rank slice is split by bucket, as the existing mixed-bucket fallback does
  (`:891-901`).
- **Sliding layers** (ring of 1,024). For `j = 0 .. w_max−1`: KV-encode
  slice j, then the batched flash on slice j, then the FWHT undo on slice j.
  - Once `ksl == cap` the kernel reads all 1,024 ring slots with no
    logical-position mask (`flash_attn_vec_hybrid.metal:807-818, 918-944`).
    So "write every row, then attend" would be wrong for windows, and the
    phases are required.
  - Phase j's ring write `(pos0+j) mod 1024` overwrites position
    `pos0+j−1024`. Ranks below j have already attended, and ranks at or
    above j do not read it (`pos_r − 1023 > pos0+j−1024`). So every query
    reads exactly M1's key set with M1's bytes.
- **Graft.** The graft is model-wide, and `mtp` refuses a bound graft at
  load (§4.2 rule 8), so windows never meet graft-indexed rings.

**Head.** The head (fused or not) over R rows is per-row bit-identical by
row-identical dispatch (`batched_head.rs:57`), in m ≤ 8 groups when R > 8.
It also fills the carry buffer (§5.2). The logits rows are read row by row
during the walk, through `row_of`. No logits beyond the first rejected row
are copied into new host vectors [LLD; M0 measures the readback].

**Why this is Class A.** Each row's arithmetic is a composition of operators
whose dispatch is row-identical across N by construction, applied to inputs
that are bitwise M1's:
- the same token embeddings;
- K/V rows of earlier positions in the window, computed by the same per-row
  arithmetic;
- for sliding layers, the same ring bytes at read time.

The existing `slot_aware_*` tests check greedy outputs, not logits bits, so
**A1 (§7.3) is the proof**, and a single differing byte falsifies R-A.

**Cost structure** [INFERENCE, measured by M0]:
- dense weights are read ⌈R/8⌉ times (once at K = 1);
- MoE expert reads scale with R unless grouped;
- attention scales with R, the same per query as M1;
- sliding layers add (w_max − 1) × 3 dispatches per layer (KV-encode,
  flash, undo): 75 per verify at K = 1 and 225 at K = 3.

### 5.4 Commit walk

The ordinary tick's per-token tail is factored into one function, used by
both paths:
- `gemma_commit_decoded_row(lane, logits_row, greedy) -> RowOutcome`;
- `GemmaDeferredFinish`, returned instead of applied, for the release and
  finish effects.

Ordinary ticks call the function and then apply the finish immediately, so
their behaviour is unchanged; A5 and A6 gate this. The function performs,
per row, in today's order:
1. `state.valid_tokens += 1` for the input token this row wrote
   (`engine.rs:16605-16615`). This feeds `retained_prefix` (`:7701-7707`)
   and warm resume.
2. `decode_tick_finalize` (`engine.rs:7788`), unchanged: logit bias, then
   `sample_logits_with_grammar` over `self.generated_tokens` (so the
   penalty window includes earlier rows of this window), grammar accept,
   EOS, push, the reasoning and tool-call splitters (`ToolCallOpen` →
   `rt.trigger()`), stop strings and grammar-dead. At temperature 0 the
   greedy-with-penalty pick returns before any random draw
   (`sampler_pure.rs:98-107`).
3. Emit the fragment. The **first failed emit stops** the walk at this row,
   with today's failure handling for that request (`engine.rs:16930,
   16942`).
4. `scheduler.advance_after_decode(handle)`, once per row
   (`scheduler.rs:1565-1599`). If it auto-releases at `max_tokens`, the
   release is captured as a deferred finish and the walk stops.

The walk over a speculating lane:

```text
m = 0
for j in 0..w:                                      # w = K+1 rows, rows via row_of[lane][j]
    out = gemma_commit_decoded_row(lane, logits(row_of[lane][j]), greedy_j); m += 1
    if out.stopped: break        # finished (EOS/stop/grammar-dead/max_tokens) or failed emit
    if j == w-1: break           # bonus row: next_token = its decision
    if lane.next_token != d_j: break                 # correction: next_token = target's decision
c = pos0 + m; carry row index := row_of[lane][m-1]
```

After all lanes' walks, in this order:
1. **Ring restore** for lanes with `m < w` (§5.5).
2. **Cursor commit.** `commit_gemma4_decode_cursors` is generalized from
   +1 to +m per lane (`engine.rs:17218`). It stays forward-only, and the
   cursor is never moved back, because nothing was committed beyond `c`.
   It is validated against `valid_tokens`.
3. **Deferred finishes.** Retained prefix, `record_retained_slot_kv`,
   release, and cancellation that arrived during the round. These read
   `seq_lens` and the ring, so they must see the restored state. A
   cancellation never interrupts the walk: it is observed after the round,
   as for an ordinary tick.

**Transaction exactness.** Each row passes through the same function, with
the same logits (§5.3) and the same request and slot state, as the serial
tick at that position. Only the timing of finish effects moves, and it moves
to a point where the state they read equals the serial state. ADR-065 :90
asks the same of Qwen's K3 through `plan_qwen35_verified_block`; here the
shared function is the planner. U5 covers every exit.

**Delivery.** Fragments are emitted as each row is committed, so a round's
1 to K+1 tokens reach the stream in one burst. SSE ordering is the serial
order.

### 5.5 Rollback and the ring pre-image

After the walk, the state **readable at the cursor** `c = pos0 + m` must
equal the serial path's.

- **Full layers.** Rows `≥ c` keep rejected bytes. Nothing reads them:
  - decode's extent is `pos+1`;
  - the drafter reads `[0, pos0)`;
  - the next write at each position overwrites them;
  - anchors copy a slot's whole sliding region, but only at prefill or
    admission (`kv_cache.rs:1605-1633`), never between a verify and its
    restore. U6b asserts that no snapshot path runs in that interval.
- **Sliding ring.** The rejected row at position r overwrote ring slot
  `r mod 1024`.
  - If `r ≥ 1024`, that slot held logical position `r − 1024`. The next
    decode at `c` reads logical `[c−1023, c−1]`, so every rejected row
    `r ≥ c+1` destroyed readable history.
  - Row `r = c` destroyed `c − 1024`, which is not readable and is rewritten
    next. Restoring it too keeps the rule uniform when it has a pre-image.
  - If `r < 1024`, the slot held nothing readable, because the ring has not
    wrapped there. The stale bytes are never read and need no pre-image.
- **Save**, before the verify, for each speculating lane with
  `pos0 + K ≥ 1024`. Let `r_lo = max(pos0+1, 1024)`. Save the ring rows at
  logical positions `r_lo−1024 .. pos0+K−1024` (`n_saved = pos0+K−r_lo+1`
  rows, 1 to K). Pre-image row i is the one verify position `r_lo+i`
  overwrites.
  - This covers every sliding layer (25) and every payload (F16 K, TQ-HB V
    codes, F32 V norms), with
    `dispatch_kv_cache_linearize_ring_bytes(src = lane's layer view, dst =
    pre-image, n_heads = 8, src_capacity = 1024, dst_capacity = K,
    history_len = n_saved, logical_end = pos0+K+1−1024, row_bytes)`.
  - The kernel maps `src_slot = (logical_end − history_len + i) mod
    src_capacity`, with `logical_end` exclusive (`kv_cache_copy.metal:432-456`).
    The bytes are opaque per row, so one primitive serves all three
    payloads.
  - Size: 25 × K × (8 × (512 + 256 + 4)) B, about 0.15 MiB per lane at
    K = 1 and 0.45 MiB at K = 3 [arithmetic]. V codes are 1 byte per
    element whatever the codebook bits are
    (`hadamard_quantize_kv_fast.metal:618, 717-726`).
- **Restore**, after the walk, if `m < w`: positions
  `r ∈ [max(c, r_lo), pos0+K]` return to their pre-images. This is the
  **tail** of the pre-image: `src_row_offset = max(c, r_lo) − r_lo`,
  `n_restore = pos0+K − max(c, r_lo) + 1`, `logical_end = pos0+K+1−1024`.
  Accepted rows `pos0+1 .. c−1` are not touched, and the next decode reads
  them.
- **The verify's rank-0 write** at `pos0` overwrote `pos0−1024`, which is
  never readable at a cursor `≥ pos0+1`, so it has no pre-image.
- **New mlx-native primitive N1**, `dispatch_kv_cache_restore_ring_bytes(
  encoder, registry, device, src /*pre-image*/, dst /*ring*/, n_heads,
  src_capacity /*pre-image rows per head*/, dst_capacity /*ring*/,
  src_row_offset, history_len, logical_end, row_bytes)`. It writes
  `dst[h, (logical_end − history_len + i) mod dst_capacity] =
  src[h, src_row_offset + i]` for `i < history_len`.
  - Bounds are checked against each buffer view's **`data_byte_len`**, not
    `byte_len`, which is the whole Metal buffer (`buffer.rs:338-340`). This
    way a per-slot view cannot write past its slot.
  - The existing linearize checks `byte_len`, so on the save side hf2q checks
    each per-slot view's bounds itself before dispatch (U6). Changing
    linearize is left to mlx-native.
  - Contract: `restore(offset, n) ∘ linearize` is the identity on exactly
    the `n` restored rows and leaves every other ring byte unchanged. This
    holds for wrap and no-wrap, full and partial (`offset > 0`) restores,
    and 1 to 4 rows, through both metallib paths.
  - Native policy: no external engine names. It is published and pinned
    before the hf2q integration lands.
  - A batched-over-seqs variant that cuts the dispatch count from
    lanes × 75 to 75 is allowed under the same contract. M0 decides whether
    it is needed.
- **Cursor.** It is committed forward by m (§5.4). The assistant has no
  cursor or state of its own, so there is nothing else to roll back.

### 5.6 Failure semantics (ADR-065 :91, carried over)

- **Lane-local refusal before submission** (entry or draft-pass error): that
  lane runs as an ordinary row in the same tick, with state untouched.
- **A failure of the submitted shared verify or restore**: every
  participating lane is reset through the existing Gemma slot-reset path,
  and its request terminates with an error. Non-participants' state and
  committed prefixes are unmutated, because their rows are rank 0 and write
  only their own `pos` row, exactly as an M1 tick would.
- **A failure after the verify but before the restore and cursor commit
  complete**: worker-fatal, fail-stop. No slot state is reused, because the
  ring may hold unrestorable rows. Deferred finishes do not run.
- Cancellation during a round takes effect after the round's restore and
  cursor commit, as a deferred finish (§5.4).

### 5.7 Eligibility, depth and policy

A lane speculates in a tick iff all of these hold:
1. The switch resolves to `mtp` and an assistant is loaded. Load already
   refused any configuration §4.2 excludes, including a bound graft.
2. The request is greedy: `temperature < SAMPLING_EPS`. The four-agent
   requests omit `temperature` and get 0 from the server default
   (`handlers.rs:2399`). Sampled requests never speculate and consume no RNG
   for drafts. `top_k`/`top_p`, penalty, bias, grammar and logprobs are all
   allowed: at temperature 0 they are deterministic per-row selections
   inside `decode_tick_finalize`, and logprob records are produced per
   committed row.
3. The lane has a carry row index (§5.2).
4. `max_tokens − generated ≥ 2`. The window is `min(K+1, remaining)`.
5. The cost controller allows it.

**Depth.** K is fixed per release. It is chosen by M0 from {1, 2, 3} under
the frozen L(K) model (§8.1), confirmed by M1 on hf2q's measured acceptance,
and frozen before S1.
- **K = 1 is measured first and is the working value.** At four lanes R = 8,
  so dense and head dispatch are unchanged and the sliding layers run only
  two phases.
- The review's prior estimate [INFERENCE]: K = 3 gives about
  c(3,4) ≈ 3.3 > 3.19, which misses G0. K = 1 gives c ≈ 1.64, which passes
  if L(1) ≥ 1.71.
- There is no adaptive depth. Under R-A, depth and the cost gate cannot
  change output, only speed.

**Cost controller.** `SpeculationCostController` (`cost_controller.rs`,
`EVALUATION_ROUNDS = 4`, `UNPROFITABLE_WINDOWS_TO_DISABLE = 2`).
- The worker keeps one per decoding width (1–4). Each compares the round's
  wall time per emitted token with the ordinary tick's time per token at the
  same width.
- The controller has no re-enable API (Qwen keeps one per request). So "a
  disabled width stays off until the next request admission" is implemented
  by **recreating** that width's controller at every request admission.
- It is deterministic in its inputs, but timing makes it nondeterministic
  across runs. That is allowed because transcripts do not depend on it (A6).

**Mixed steps.** Speculation runs in `Decode` and `Mixed` steps alike. The
pre-declared alternative arm `HF2Q_GEMMA_SPECULATION=mtp-decode-only`
suppresses windows in a tick that also runs a prefill quantum. It exists
only if S1's TTFT fairness fails (§8.3, decision D7).

### 5.8 Switch, default, kill switch

`HF2Q_GEMMA_SPECULATION` is resolved once per worker:

| Value | Before promotion | After promotion (§8.6) |
|---|---|---|
| unset, assistant resolved (managed §4.4 or `--gemma-assistant`) | off | `mtp` |
| unset, no assistant | off | off |
| `0` / `off` / `false` | off (kill switch) | off (kill switch) |
| `mtp` / `1` / `on` | `mtp` (refuses to start without an assistant) | `mtp` |
| `mtp-decode-only` | exists only under D7 | as decided |
| anything else | warning, then off | warning, then the default |

Model load fails closed when `mtp` resolves but the assistant or any of its
pipelines (drafter kernels, the N1 primitive, the sliding phase kernels)
cannot be built or prewarmed. There is no silent fallback to off.

### 5.9 Observability and telemetry

- **Startup log:** `Gemma speculation mode=mtp depth=<K> assistant=<sha256>
  quant=Q8_0 head=fused|unfused`.
- **`/hf2q/v1/runtime` `active_controls`** (`src/serve/api/measurement.rs`):
  `gemma_speculation`, `gemma_speculation_depth` and
  `gemma_assistant_sha256`, so that Q5 receipts bind the arm and the drafter.
- **Prometheus**, mirroring Qwen (`handlers.rs:8684-8745`):
  - `hf2q_gemma_speculation_{rounds,drafted_tokens,accepted_tokens,rejected_tokens,ring_restores,cost_disabled}_total`;
  - `…_round_seconds_total`, `…_equivalent_ordinary_seconds_total`;
  - `…_fallback_requests_by_reason_total{reason}`;
  - `hf2q_gemma_decode_ticks_by_width_total{width,speculative}` and
    `hf2q_gemma_decode_tick_seconds_total{width,speculative}`. In a
    spec-off baseline these give the time-weighted width histogram that the
    §8.1 projection needs.
- **Dev-only `HF2Q_GEMMA_SPEC_PHASE_PROFILE=1`:** one line per round with
  draft, save, verify, head, walk and restore times (as
  `HF2Q_MTP_PHASE_PROFILE` does for Qwen).

### 5.10 Surfaces updated in the same change

- `scripts/serve_gemma4_opencode.sh`:
  - `ASSISTANT` is forwarded as `--gemma-assistant`;
  - `SPECULATION` is forwarded as `HF2Q_GEMMA_SPECULATION`.
- `scripts/four_agent/q5_cases/gemma4.json`:
  - new variables `GEMMA_ASSISTANT`, `GEMMA_ASSISTANT_SHA256` and
    `GEMMA_SPECULATION`;
  - the runner verifies `shasum -a 256` of the assistant before launch, as
    the peer cases do with `DRAFT_SHA256`;
  - readiness checks `active_controls.gemma_speculation` per arm.
- `test_q5_calibration.py`.
- `docs/operator-env-vars.md`, `docs/shipping-contract.md`,
  `docs/env-var-inventory.md` and CHANGELOG.
- `scripts/audit_hf2q_env_surface.sh` inventory.
- `src/quality/adr063_invariance/gemma_actual.rs` pins the speculation
  default in `gemma_controls`.

## 6. Exactness and quality contract

Policy `hf2q.q1.numeric-policy.v1`, thresholds unchanged.

| Surface | Class | Requirement |
|---|---|---|
| Verified rows (every logit row the walk finalizes, and every row R-A computes) | **A** | Bitwise equal to teacher-forced serial M1 rows for the same tokens and prefix: logits, hidden and every KV byte written. Store verdict `ExactEqual`; one byte is `DivergedExactInvariant` |
| Cohort invariance | **A** | A window's rows and its lane's post-commit state are byte-identical alone and in any cohort of 1–4 windows of mixed widths, with ordinary rows, in any slot order, across ring wrap and `(nwg, nsg)` bucket boundaries |
| Non-participants | **A** | Ordinary rows in a speculative tick are byte-identical to the same rows in an all-ordinary tick |
| Transaction | **A** | Given identical logits, the committed tokens, stream fragments, request state (penalty history, grammar, splitters, stop handling, logprobs) and slot state equal the serial ticks (§5.4 identity) |
| Rollback store | **A** | After a round with *a* accepted, the state **readable at the cursor** c equals the serial path's after a+1 tokens: full layers `[0, c)`, the ring's logical window `[max(0, c−1023), c−1]`, cursors, `valid_tokens`. The next eight ordinary decodes are also bitwise equal. Stale bytes outside the readable window (full rows `≥ c`; unwrapped ring slots beyond c) are not compared, because the serial path never wrote them |
| Draft side effects | **A** | The drafter pass changes no target byte and no cursor (Q1 §3.1(5): "draft generation and discard must not change committed state") |
| Serving transcripts, spec on vs off | **A** | `ExactTranscripts` (`q1_decision_policy.py transcripts`) in every S1 block. Quality is then inherited (Q1 §3.6) |
| Drafter logits | outside Q1 (proposal-only, never committed) | Correctness is checked against an F64 reference of §3.2 on identical inputs (D1, §7.3). Acceptance is a speed quantity (§8.2) |
| hf2q vs the peer | **C** | Not a gate on rows. The Q4 paired quality gates apply on the combined candidate |

Two notes on scope:
- Nothing in this story is Class B, so there is no envelope, near-tie or
  canonical-coordinate test. No Class B verify route (grouped `mm_id`, `mm`
  above 8 rows, or the prefill multi-seq path) is in scope. If R-A misses G0,
  the windowed verify code is deleted (§13, D1).
- §3.7 still applies: no subsets, no retry-until-pass, no learned
  tolerances.

## 7. Tests

Model-free tests run in `cargo test --locked --bin hf2q` (and the converter
crate path). Real-artifact tests are ignored, one process each,
`--exact --ignored --test-threads=1`, on the root's GPU lane.

### 7.1 Model-free (hf2q)

| Id | Test | Asserts |
|---|---|---|
| U1 | `gemma4_assistant_arch_detection_precedes_gemma4_prefix`, `gemma4_assistant_tensor_map_and_metadata`, `gemma4_assistant_refuses_ordered_embeddings` | Synthetic config/header: names (`nextn.pre_projection`, …), keys (`embedding_length_out`, `nextn_predict_layers`, `shared_kv_layers`), Q8_0 policy; refusals for `use_ordered_embeddings`, `masked_embedding.*`, `k_proj`, partial KV sharing |
| U2 | `gemma4_assistant_pairing_guard_matrix` | Accepts the §4.2 pair under both head paths; refuses each of rules 1–8 in isolation with its message (including `HF2Q_FULL_F16_KV=1` and a bound graft); the kata test plus the drafter-contract assertion |
| U3 | `gemma_speculation_switch_table` | §5.8 table before and after promotion; `mtp` without an assistant fails startup |
| U4 | `gemma_speculation_eligibility` | Sampled is refused (and no RNG is drawn); an omitted `temperature` resolves greedy; penalty, bias, grammar and logprobs allowed; budget window `min(K+1, remaining)`; first tick after prefill is ordinary; per-width cost controller recreated at admission |
| U5 | `gemma_spec_walk_matches_serial_ticks` | With synthetic logits through `gemma_commit_decoded_row`, the walk equals the same rows as successive ordinary ticks in: fragments, request state, `valid_tokens`, the cursor commit by m, the count and order of `advance_after_decode` calls, and the deferred finish/release (retained prefix length = c, `record_retained_slot_kv`), applied after restore. Cases: every combination of full, partial and zero acceptance; EOS, stop string, grammar trigger and grammar-dead at each row; `max_tokens` auto-release at each row; first failed emit at each row; cancellation arriving mid-round; penalty sensitivity across rows (a row whose argmax flips only because of an earlier row's token in the window) |
| U6 | `gemma_spec_ring_preimage_plan` | For each `(pos0, K, m)`, including pos0 ∈ {1020..1026} and deep wrap: `r_lo`, `n_saved`, save `logical_end`; restore `src_row_offset`, `n_restore`; accepted rows never restored; no save before wrap; per-slot view bounds checked before the save dispatch |
| U6b | `gemma_spec_no_snapshot_between_verify_and_restore` | No anchor, prefix-cache, retained-prefix or warm-resume snapshot path runs between a round's verify submission and its restore plus cursor commit; deferred finishes run only after them |
| U7 | `gemma_spec_failure_semantics` | Injected lane-local refusal leaves the lane ordinary; a shared verify failure resets participants and leaves non-participants unmutated; a failure after verify fail-stops and runs no deferred finish |
| U8 | `gemma_spec_window_rows_layout` | **Rank-major** layout: slice j holds rank j of every window in window order, and slice 0 equals today's one-row-per-slot batch; `rank_offsets`, the window-major `row_of` map, per-row positions and slot ids; each phase is a contiguous slice; dense grouping ⌈R/8⌉; full-layer bucket split |
| U9 | `gemma_spec_metrics_and_controls` | Metric names and runtime control fields |

### 7.2 mlx-native

| Id | Test | Asserts |
|---|---|---|
| N1 | `kv_cache_restore_ring_bytes_inverts_linearize` | Bitwise round trip for row_bytes ∈ {4, 256, 512, 1024}, heads 2 and 8, capacity 1024, history 1–4, wrap and no-wrap, **full and partial restores (`src_row_offset` 1..3)**, untouched-byte checks on the ring and the rest of the buffer, and refusal when a write would exceed a view's `data_byte_len`; both metallib paths (default and `MLX_PRECOMPILED_METALLIB=0`) |

### 7.3 Real artifact (target `bf65b7e1…`, assistant from §4.1)

| Id | Test | Required |
|---|---|---|
| A1 | `gemma_spec_verify_rows_match_serial_m1_bitwise` | For windows w = 1..4, teacher-forced tokens from a frozen greedy transcript, and cohorts {one window}, {4 windows w=4}, {4,1,3,2}, {2,4} plus two ordinary rows, in both slot orders, at positions below 1024, crossing 1024, deep (≥ 2,048), and straddling an `(nwg, nsg)` bucket edge: logits rows, hidden rows and all KV bytes written equal serial M1 decodes. Bitwise |
| A2 | `gemma_spec_rollback_store_matches_serial` | For a ∈ {0..K} with forced drafts, wrapped, unwrapped and straddling 1024: after commit, the readable state at the cursor (full layers `[0, c)`, ring logical `[max(0, c−1023), c−1]`, cursors, `valid_tokens`) `ExactEqual` serial after a+1 tokens, and the following 8 ordinary decodes are bitwise equal (logits and readable state) |
| A3 | `gemma_spec_draft_pass_is_side_effect_free` | Target store bytes and cursors are unchanged by a drafter pass. Bitwise |
| A4 | `gemma_spec_ordinary_rows_invariant` | Ordinary rows in a speculative tick equal the same rows in an all-ordinary tick. Bitwise |
| A5 | existing `slot_aware_n1`, `slot_aware_n4`, `slot_aware_n8_per_slot_parity_vs_serial` and the Gemma joint warm prefill Class A gates | Pass unchanged. The w=1 path is today's |
| A6 | Q5 S1 arms, `q1_decision_policy.py transcripts` on vs off, per block | `ExactTranscripts` |
| A7 | `gemma_spec_fail_closed_injection` | §5.6 on hardware, with an injected verify error |
| D1 | `gemma_assistant_drafter_reference_parity` | Drafter logits and `h_next` against an F64 numpy reference of §3.2 computed from the GGUF's dequantized weights and **hf2q's own exported** target layer 28 K/V and layer 29 K and V caches (dequantized V) plus the carry row. Positions: pos0 < 1023, pos0 = 1023/1024/1025, and deep; 3 steps. Frozen bounds: rel-RMS ≤ 1e-4, and top-1 equal on every row whose reference top-2 margin is ≥ 0.01. Also asserts that attention used `seq_pos = pos0 − 1` and RoPE used `pos0`. The sliding-window variant (1023 vs 1024 keys) is pinned against the HF reference code on the same arrays; the design default is 1023 (§3.4). Any change is a recorded fix, never a threshold change |
| D2 | `gemma_assistant_acceptance_replay` (diagnostic; M1, §8.2) | Teacher-forced replay over the recorded spec-off four-agent transcripts. For every decode position, draft 3 steps from the committed prefix and record per-depth acceptance, so L(1), L(2) and L(3) come from one replay; Q8_0 vs BF16; the real `td(K, n)` |

## 8. Measurements and gates

Every hardware run records:
- the source commit and binary SHA-256;
- the target and assistant digests;
- the mlx-native version and metallib SHA-256;
- the case file and its SHA-256.

Every run passes Q5 `CalibrationPass` with the accepted negative-control
receipt. Evidence goes to `~/.local/state/hf2q/v0130/q4-gemma-spec-serve/`.
Only the root, holding the GPU lane, runs these, one model process at a time.

### 8.1 M0: round-cost curve and gate G0 (before any drafter runtime)

Harness: `gemma_spec_round_cost_curve`, a real-artifact ignored bench. It
loads the target only. It builds 4 live slots from the canonical turn-2
prompts (contexts about 700–1,400 rows; the #660 per-turn table) and times,
as the median of 200 ticks after 20 warm-up ticks:
- `t1(n)`: the ordinary tick at n = 1..4 decoding slots;
- `tv(w, n)`: the R-A tick with n speculating windows of width w, forced
  drafts, both head paths. **w = 2 (K = 1) first**, then w = 3 and 4;
- `ts`: the pre-image save, and `tr`: the restore;
- `td_bound(K, n)`: the drafter-pass bound, measured without a drafter
  runtime (below).

Each figure includes host walk and readback. The round-cost ratio is
`c(K, n) = (td_bound(K, n) + ts + tv(K+1, n) + E[tr]) / t1(n)`.

**Drafter bound** [frozen]:
`td_bound(K, n) = 1.25 · K · (Σ_s t_mv(s, n) + t_argmax(n) + 4·t_flash(n) + 30·t_disp)`.
- `t_mv(s, n)`: `quantized_matmul_ggml` Q8_0 mat-vec at m = n on random
  weights of each §3.1 shape `s`: pre 1024×5632; per layer q, o, gate, up,
  down; head 262,144×1,024; post 2,816×1,024.
- `t_argmax(n)`: `dispatch_argmax_f32` over 262,144 × n.
- `t_flash(n)`: one target hybrid flash plus FWHT undo with n queries on the
  live slots' layer 28 and layer 29 caches, averaged.
- `t_disp`: the harness's measured empty-dispatch cost, covering norms,
  RoPE, concat, gather and scalar ops.
- The 1.25 covers dispatch-sequence differences. All of these are existing
  mlx-native ops, so no drafter code is needed.

**Acceptance model** [frozen until M1]: geometric per-depth acceptance
fitted to the peer's measured L(3) = 3.34 (§2.2).
`1 + α + α² + α³ = 3.34` gives `α = 0.881`, so L(1) = 1.881, L(2) = 2.657
and L(3) = 3.341 [arithmetic].

**Time-weighted width histogram.** `h(n) = tick_seconds(n) / Σ_m
tick_seconds(m)`, from `hf2q_gemma_decode_tick_seconds_total{width=n,
speculative=false}` in one spec-off Q5 run of the same binary. It is
equivalently `ticks(n)·t1(n) / Σ ticks(m)·t1(m)`, so `D·h(n)` is the decode
time spent at width n.

**Projection** [frozen formula]: `ΔD(K) = Σ_n D·h(n)·(1 − c(K, n)/L(K))`,
where `D` = 9.203 s, hf2q's L1 decode median (§2.1). M0's provisional K* is
the K ∈ {1, 2, 3} that maximizes ΔD(K).

**G0 (frozen).** `ΔD(K*) ≥ 0.40 s`, the lower end of #660 §7's estimate.
At width 4 only, that means:
- K = 1: c(1, 4) ≤ 1.881 × (1 − 0.40/9.203) = 1.80;
- K = 3: c(3, 4) ≤ 3.19 [arithmetic].

- **On a miss**, the story stops. Record M0 and the falsification. The
  windowed verify code is deleted unless a zero-cost drafter (n-gram or
  prompt lookup) is opened to reuse it (D1). N1, once released, stays in
  mlx-native. Q4's Gemma budget then loses this line (#660 §7).
- **On a pass**, M0 also records whether the batched N1 variant is needed
  (save + restore > 3% of `tv`).

### 8.2 M1: acceptance replay and final depth (after the drafter, before S1)

D2 runs over the 28 spec-off turns of three Q5 runs. It reports per-depth
acceptance L(1..3), the real `td(K, n)` and the Q8_0-vs-BF16 difference.

Decision rule (frozen):
- Q8_0 ships unless its accepted-per-drafted falls more than 0.02 below BF16
  (D2).
- ΔD(K) is recomputed with hf2q's measured L(K) and real `td` in place of
  the model and the bound. **The final K\*** is the maximizer, frozen before
  S1. G0 must hold at it; a miss is the same stop as §8.1.
- Context only: the peer's 0.79 accepted per drafted (#660).

### 8.3 S1: speed A/B, same binary (ADR-065 two-arm rules)

The arms differ only in `GEMMA_SPECULATION ∈ {0, mtp}`, with the assistant
loaded in both, so memory and load are identical. Settings: 12 ABBA blocks,
fixed before qualification, on the canonical four-agent workload:

```sh
for ARM in 0 mtp mtp 0; do   # one block; 12 blocks
  HF2Q_Q5_GEMMA_SPECULATION=$ARM HF2Q_Q5_GEMMA_ASSISTANT=$ASSIST HF2Q_Q5_GEMMA_ASSISTANT_SHA256=$ASSIST_SHA \
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json \
    --negative-control-receipt "$NC" \
    --output ~/.local/state/hf2q/v0130/q4-gemma-spec-serve/s1-$ARM-$(date -u +%Y%m%dT%H%M%SZ)
done
```

Frozen gates:
- **S1-a (speed).** The ADR-065 verdict is **Win**: the paired
  log-makespan 95% interval excludes zero favourably, with half-width ≤ 2.5
  points. **And** the median makespan reduction is ≥ 0.30 s (75% of the
  0.40 s booked) **and** ≥ 60% of M0's projected ΔD(K*).
- **S1-b (stage).** The decode stage, by
  `scripts/four_agent/gemma_stage_profile.py` on the A/B receipts, drops by
  ≥ 0.35 s median. The prefill and residual stages do not grow by more than
  0.10 s (the #660 CI half-widths).
- **S1-c (exactness).** A6 `ExactTranscripts` in every block.
- **S1-d (fairness).** The ADR-065 paired test against the `0` arm:
  worst-slot p95 semantic TTFT and worst-slot max SSE gap, each failing if
  the lower bound of the ratio interval exceeds 1.025. Every run stays ≤ 1.5×
  the frozen B1-style ceilings from the `0` arm. Context: #660 measured
  hf2q's worst SSE gap at 62–66 ms (tool-call end, turn 1) and the peer's at
  98–190 ms from draft-verify bursts. The long decodes are the unary turns 2
  and 6, but streamed turns also speculate. A TTFT failure triggers the
  pre-declared `mtp-decode-only` arm (D7) as one more S1 run of 12 blocks.
  An SSE failure falsifies the default.
- **Telemetry**, reported and not gated: accepted per drafted, L, rounds,
  restores, cost-disabled windows and `h(n)`.

### 8.4 Quality

R-A is Class A, so S1-c (`ExactTranscripts`) carries the quality verdict to
the `0` arm (Q1 §3.6). **No separate 24-instance campaign is needed for
promotion.** A `DivergedTranscripts` block is a defect, not a quality
question: it blocks promotion and is localized through A1–A4. The Q4
closing campaign still runs the paired quality gates on the combined
candidate against v0.1.28 and against the peer (ADR-063 Q4).

### 8.5 Memory (Q5)

The artifact set changes (target + assistant), so the `gemma4` case is
recalibrated:
- a supervised run of the largest declared four-agent workload with `mtp`;
- `CalibrationPass` against the Gemma effective floor (5.3 GiB, ADR-063
  Q5);
- the receipt binds both sha256s.

Expected additions [arithmetic]:
- weights about 0.46 GB;
- drafter scratch ≤ 16 MiB;
- pre-images ≤ 2 MiB;
- verify buffers for 8 rows at K = 1 (16 at K = 3);
- the double-buffered carry buffer, 2 × R × 2816 × 4 B ≤ 0.4 MiB.

A failure blocks promotion.

### 8.6 Promotion

`mtp` becomes the default when an assistant is resolved (§5.8) only when
all of these hold:
- N1 is published and pinned;
- U1–U9 (with U6b) and A1–A5, A7 and D1 passed;
- G0 passed on M0 and again on M1, and K\* is frozen;
- S1-a through S1-d passed;
- the Q5 recalibration passed;
- the Q8_0 assistant artifact is published as a managed artifact, with its
  sha256 pin merged (§4.4, D3).

The promotion commit:
- flips the default;
- keeps `0` as the kill switch;
- updates the §5.10 surfaces;
- records the receipts in this document and in ADR-063.

### 8.7 Q4 contribution

This story has no Win of its own. Its configuration enters Q4-GEMMA-ATTN's
combined candidate, which is run once per release candidate (the
`q3_campaign.py` family `gemma4` invocation in the tiled design §9.3) with:
- `--experiment-var GEMMA_SPECULATION=mtp` and the assistant variables;
- the peer arm of record, per D4.

The Q4 gates apply there:
- the CI upper bound ≤ 0.95 against the peer's best;
- TTFT and SSE within 1.025 against v0.1.28;
- quality `Pass` against v0.1.28 and against the peer;
- a fresh Q5.

## 9. Risks

| Risk | Effect | Mitigation / disposition |
|---|---|---|
| R1 MoE does not amortize under per-token `mv_id` (ADR-034 evidence) | G0 miss | M0 first, at K = 1 (8 rows), where MoE costs at most 2× an ordinary tick. Q4-GEMMA-DECODE's bitwise grouped `mv_id` (≤ 16 tokens × top-8, agreed) amortizes expert reads while staying Class A, and M0 is re-run on it if it lands. No Class B route (D1) |
| R2 Dispatch overhead: sliding-phase dispatches (75 at K = 1), pre-image save/restore (lanes × 75 without the batched N1), logits readback | Raises c | Measured inside c; the batched N1 variant; lazy row readback |
| R3 Lower acceptance than the peer: hf2q's V is TQ-HB quantized, while the peer's KV is F16 and the drafter was trained on exact KV; Q8_0 drafter; sliding-window choice | Lower L | D2 measures it; BF16 is the fallback (D2 rule); the D1 window pin |
| R3b The L(K) model is wrong at depth 1 | M0's provisional K\* differs from M1's | K\* and G0 are final only at M1, on measured L |
| R4 SSE/TTFT fairness: a round is about c× an ordinary tick, and Mixed steps delay prefill quanta | S1-d failure | Bursts deliver several tokens per gap; `mtp-decode-only` is pre-declared (D7); the cost controller |
| R5 Bucket edge cases (mixed `(nwg, nsg)` buckets inside a rank slice) | A1 failure | The rank slice is split by bucket, as the existing fallback does; A1 covers bucket edges; a graft is refused at load |
| R6 Bookkeeping drift between the walk and ordinary ticks (`valid_tokens`, cursor commit, `advance_after_decode`, emit failure, finish/cancel) | Wrong retained prefix or release; non-exact transcripts | One shared `gemma_commit_decoded_row`; finishes deferred until after restore; U5; A6 |
| R7 Ring restore omission (a snapshot or read between verify and restore) | Silent state corruption | Restore and cursor commit run in the same worker step before any deferred finish; U6b; A2 includes the next 8 ordinary decodes |
| R8 Carry-buffer lifetime (the next head overwrites it before the draft pass reads it) | Wrong drafts (speed only) | Double-buffered; A3 and D1 read through it |
| R9 Interaction with siblings: #668 changes warm prefill only; Q4-GEMMA-DECODE changes decode kernels bitwise | R-A stays Class A only if M1 and verify call the same kernels | Both use `forward_decode_body_batched`; A1 runs on the final combined candidate too |
| R10 Peer fairness: hf2q's Q8_0 drafter could also help the peer | Unfair comparison | D4 |

## 10. GPU validation plan (root, in order; each step gates the next)

1. **mlx-native N1:** parity on both metallib paths; release; crates.io
   verification; hf2q pin.
2. **R-A with forced drafts, K = 1 first:** A5 (unchanged M1, both head
   paths), then A1, A2, A4 and A7.
3. **M0:** round-cost curve (target only; `td_bound`, frozen L(K) model,
   time-weighted h), G0, and provisional K\*. Stop on a miss (§8.1).
4. **Converter, drafter and managed artifact:** convert the assistant (Q8_0
   and BF16) with receipts; A3; D1 (reference parity; window pin;
   positions).
5. **M1:** D2 replay over the spec-off Q5 runs; real `td`; G0 on hf2q's L;
   final K\*; Q8_0 vs BF16.
6. **Q5:** recalibration with `mtp`.
7. **S1:** 12 ABBA blocks, with A6 per block. The `mtp-decode-only` arm runs
   only on an S1-d TTFT miss.
8. **Promotion commit** (§8.6).
9. **Q4-GEMMA-ATTN combined candidate campaign** (§8.7).

Every step records the identities in §8. A failure is a preserved case, not
a retry trigger (Q5).

## 11. Stage ownership against #660

| #660 stage | Owner |
|---|---|
| Decode, speculation part (peer MT −0.48 s decode; booked 0.4–0.5 s) | **this story** |
| Decode, per-step kernels (MoE 40%, dense 38%) | Q4-GEMMA-DECODE. Its bitwise kernels also lower this story's c |
| Warm-prefill attention | #668 Q4-GEMMA-ATTN |
| Warm-prefill per-row MoE/dense | the sibling prefill-MoE story |
| Cold-turn admission | the sibling cold-admission story |
| SSE cadence | inside #668's acceptance. This story must not regress it (S1-d) |

## 12. ADR-063 story section (extraction-ready)

### Q4-GEMMA-SPEC-SERVE. Serve Gemma 4 with its assistant (MTP) drafter at four slots

Outcome: hf2q's Gemma 4 26B-A4B serving worker speculates with the Gemma 4 assistant drafter at four slots, as the peer's best configuration does, and finishes the canonical four-agent coding task measurably sooner than the same binary without speculation, with transcripts unchanged.

Requirement: The program MUST convert and load `google/gemma-4-26B-A4B-it-assistant@6e5aaaf4` as a paired drafter and MUST provide a joint four-slot draft/verify transaction in the Gemma serving worker whose verified rows, commits and rollbacks are Class A exact against the serial decode route.

Acceptance:

- A written design, approved before implementation, states the verify route, rollback, switch and kill switch: `docs/research/adr063-q4-gemma-spec-serve-design.md`. In it: the assistant comes from managed download pinned by sha256, with `--gemma-assistant` as the override; `HF2Q_GEMMA_SPECULATION=mtp` enables speculation; it is off by default until every gate passes, then on when an assistant is resolved; `0` is the kill switch; only greedy requests speculate; the depth K ∈ {1, 2, 3} is measured with K = 1 first.
- The drafter loads only through a drafter contract paired to a `gemma4` target, behind a fail-closed pairing guard. `gemma4-assistant` stays out of the primary runtime contracts. The 26B-A4B checkpoint's head is the dense tied embedding (`use_ordered_embeddings: false`); ordered-embedding (centroid) assistants are refused, never approximated. The drafter's step-0 input is the target head's retained post-final-norm operand; the fused head stays on.
- The round-cost curve is measured on the verify route before any drafter runtime, with a frozen drafter-cost bound, a frozen acceptance model fitted to the peer, and a time-weighted width histogram. Gate G0: the projected decode saving at the chosen depth must be ≥ 0.40 s per run (the lower end of the Q4-GEMMA booked 0.4–0.5 s). G0 is re-checked, and the depth fixed, on hf2q's measured acceptance. A miss ends the story as a recorded falsification, and the windowed verify is removed unless a zero-cost drafter reuses it.
- Class A gates pass:
  - verified rows are bitwise equal to teacher-forced serial decode rows in every cohort;
  - ordinary rows are invariant;
  - after a round with *a* accepted drafts, the state readable at the cursor (full layers below it, the sliding ring's window, cursors) equals serial decoding of a+1 tokens, and so do the next eight decodes;
  - the per-token bookkeeping (valid tokens, cursor commit, scheduler advance, emit failure, finish and cancellation after restore) equals the ordinary ticks';
  - the drafter pass leaves the target state unchanged;
  - serving transcripts with speculation on and off are `ExactTranscripts` in every A/B block.
- Same-binary A/B under the ADR-065 two-arm rules (12 ABBA blocks): verdict Win, median makespan reduction ≥ 0.30 s and ≥ 60% of the M0 projection, decode stage −0.35 s or more, and paired TTFT/SSE fairness within 1.025 against the speculation-off arm. A fresh Q5 calibration of the target plus assistant artifact set passes.
- Every native primitive (the byte-exact ring-row restore with a source row offset) is published in mlx-native and consumed at an exact pin. The Q8_0 assistant artifact is published and pinned before promotion.

Repository ownership: hf2q owns the converter, drafter contract, serving transaction, policy, telemetry and campaigns. mlx-native owns the ring-row restore primitive, its parity tests and release.

Validation: Independent review of the design, the Class A receipts, the M0 round-cost curve and G0 decision, the acceptance replay and the A/B receipts.

Predecessors: Q4-GEMMA (#660, profiling result), Q5 (#371). Q4-GEMMA-ATTN's closing campaign consumes the accepted configuration.

Priority: High. Core scope (second-tier family).

Delivery boundary: Release v0.1.30, or the first published hf2q release containing the accepted Gemma 4 configuration; a G0 miss delivers the recorded falsification instead.

## 13. Recorded decisions (review ReviewGemmaSpec, adopted by the root, 2026-11-04)

1. **D1, the verify route: R-A only.** No Class B grouped-verify design is
   in scope. On a G0 miss (M0 or M1), the windowed verify, the rank-phase
   code and the walk are deleted in a clean cutover, unless a zero-cost
   drafter (n-gram or prompt lookup) is opened at that point to reuse them.
   N1, once released, stays in mlx-native.
2. **D2, the drafter artifact: Q8_0 by default.** hf2q converts `6e5aaaf4`
   to Q8_0 2-D weights with F32 norms and peer-compatible names. BF16 is
   used only if M1's accepted-per-drafted drop exceeds 0.02.
3. **D3, distribution: managed download (b), with a sha256 pin** (§4.4).
   `--gemma-assistant` remains the override and the Q5 path.
4. **D4, peer fairness: approved.** The peer's MT arm is re-qualified with
   hf2q's Q8_0 assistant GGUF as a tie-break arm against the BF16
   `e1145ee2…` arm, and the faster becomes the peer arm of record for the Q4
   closing campaign.
5. **D5, the default after promotion: approved.** `mtp` when an assistant
   is resolved, off otherwise; `0` is the kill switch.
6. **D6, the G0 threshold: 0.40 s**, applied to the frozen formula of §8.1
   (drafter bound, L(K) model, time-weighted h) and re-checked at M1.
7. **D7, Mixed-step fallback: approved.** `mtp-decode-only` is the one
   pre-declared extra arm if S1's TTFT fairness fails.
8. **Depth: measure K = 1 first; K = 1 is the working value.** The review's
   prior [INFERENCE]: c(3, 4) ≈ 3.3 misses G0, while c(1, 4) ≈ 1.64 passes
   if L(1) ≥ 1.71 (the model gives 1.88).
9. **Cost controller: one per decoding width, recreated at each request
   admission** (it has no re-enable API).
10. **Temperature:** the Gemma four-agent requests omit `temperature`; they
    are greedy only through `handlers.rs:2399`'s default of 0. A client that
    sends a positive temperature does not speculate.

Review findings applied in revision 2:
1. Fused head kept; its normed operand is the carry (§4.2 rule 7, §5.2).
2. Rank-major rows with a window-major map (§5.3, U8).
3. Drafter attention at `pos0 − 1`, RoPE at `pos0` (§3.4, §5.2, D1).
4. Restore takes `src_row_offset`; partial N1 case; `data_byte_len` bounds
   (§5.5, N1, U6).
5. Rollback gate is the state readable at the cursor plus eight follow-on
   decodes; U6b restated (§6, A2, U6b).
6. Per-row bookkeeping in one shared function, finishes deferred until after
   restore (§5.4, U5).
7. G0 computable before any drafter runtime: `td_bound`, the frozen L(K)
   model, and K\* final at M1 (§8.1, §8.2).
8. Time-weighted h(n) (§8.1, §5.9).
9. Fact corrections: 48 tensors; V codes 1 byte per element; layer 29's V
   cache differs from K; the graft is model-wide; the `slot_aware_*` tests
   compare greedy outputs; `HF2Q_FULL_F16_KV` is refused (§2.3, §3, §4.2,
   §5.3, §5.5).

## 14. Source identities read

- hf2q `main` `ff4af4bd` (`/opt/hf2q`):
  - `src/inference/gguf_contract.rs`
  - `src/core/model_arch.rs`
  - `src/convert/arch/gemma4.rs`
  - `src/serve/spec_decode_cli.rs`
  - `src/serve/api/engine.rs` (lines 575–640, 7520–7900, 11973, 16371, 17131–17222)
  - `src/inference/models/gemma4/{batched_body.rs, batched_head.rs, kv_cache.rs, model.rs}`
  - `src/serve/forward_prefill_batched.rs:6832`
  - `src/serve/operator_settings.rs`
  - `src/inference/spec_decode/cost_controller.rs`
  - `src/serve/api/{engine_qwen35.rs, qwen35_speculation.rs}`
  - `scripts/four_agent/{family_peer.py, q5_cases/gemma4.json}`
- mlx-native 0.15.17 (`/opt/mlx-native`):
  - `src/ops/{kv_cache_copy.rs, copy.rs, gather.rs, flash_attn_vec.rs, flash_attn_vec_hybrid.rs, quantized_matmul_ggml.rs, dense_gemv_bf16.rs, rope.rs, fused_head_norm_rope.rs, feature_concat.rs, argmax.rs}`
- The peer reference `e86f5fd1`:
  - `src/models/gemma4-assistant.cpp`
  - `src/models/gemma4.cpp:395-442`
  - the model-registry source, lines 3245-3320; the hparams header, lines 519-528; the graph builder, line 3551
  - `common/speculative.cpp:1489-1931`
  - `conversion/gemma.py:671-972`
- HF: the `6e5aaaf4` `config.json`, `generation_config.json` and safetensors
  header; `modeling_gemma4_assistant.py` (transformers main, fetched
  2026-11-04); `gemma4_assistant.md` at `effde209`.
- Receipts:
  - `$G/MT.receipt.json`, `$G/drafters.json`
  - `$G/cases/peer/G-MT-mtp-assistant.json`
  - `$G/profile/l1/{001-b01-hf2q-…, 002-b01-peer-…}`
  - `$G/profile/l1/compare-hf2q-vs-peer.json` (via the #660 result)
