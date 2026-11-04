# ADR-063 Q4-GEMMA-SPEC-SERVE: serving-side speculative decoding for Gemma 4 26B-A4B with the Gemma 4 assistant (MTP) drafter — HLD/LLD

Status: design for approval (2026-11-04). No code, build or hardware run backs
this document. Every number is quoted from a retained receipt or source line,
with its path, or is marked `[INFERENCE]` (estimates and arithmetic included).

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
   - one drafter pass of K=3 greedy steps for every eligible slot at once;
   - **one** joint target verify forward holding every slot's window (K+1 rows
     for a speculating slot, 1 row for an ordinary one);
   - the existing per-row finalize, walked in order (greedy exact match);
   - commit, or rollback of the rejected rows.

   This is the peer's best Gemma arm (`draft-mtp`, n-max 3, greedy drafts) as
   an hf2q serving transaction. Today it exists only in the offline
   `generate` paths.
2. **The verify route is Class A, bitwise equal to the serial M1 decode.**
   The Gemma batched decode body is built from row-identical dispatch:
   - quantized dense projections run the m=N `mul_mv` at N ≤ 8;
   - routed MoE is pinned to the per-token `mv_id` route;
   - attention runs per query.

   It is proven byte-identical to N serial decodes (`slot_aware_n1`,
   `slot_aware_n4`, `slot_aware_n8_per_slot_parity_vs_serial`;
   `src/inference/models/gemma4/batched_body.rs:2349-2362, 440-490,
   2053-2063`). The verify route (R-A) is that body run over *virtual rows*
   `(slot, position)`:
   - quantized dense projections in groups of ≤ 8 rows;
   - MoE per token;
   - sliding-layer KV write and attention in rank-ordered phases, so no row
     reads a ring slot that a later row of its own window has overwritten.

   Each verified row's logits therefore equal the row M1 would have produced,
   so speculation on versus off must give `ExactTranscripts`. Quality is
   inherited (Q1 §3.6), and there is no Class B surface. This is the main
   difference from Qwen K3, whose verify graph is a Class B route (ADR-065
   :89).
3. **Rollback is exact.** Full-attention layers are append-only, so moving
   the cursor back is enough. The sliding ring holds exactly 1,024 rows
   (`kv_cache.rs:373`), so verify rows `pos0+1..pos0+K` overwrite history
   that the next decode still needs. Before the verify, the round saves
   those ring rows' pre-images on the GPU. After acceptance it restores the
   rejected rows' pre-images. Saving reuses mlx-native's
   `dispatch_kv_cache_linearize_ring_bytes` (`src/ops/kv_cache_copy.rs:961`).
   Restoring needs its byte-exact inverse, a new mlx-native primitive (§5.5).
   A Class A store gate requires that a round with *a* accepted drafts leaves
   every KV byte and cursor `ExactEqual` to the serial path after a+1 tokens.
4. **The drafter is a sidecar, not a runtime.**
   - `gemma4-assistant` stays out of the primary `CONTRACTS` (the kata test
     `src/inference/gguf_contract.rs:81-86` keeps passing).
   - It gains a drafter contract that loads only beside a paired `gemma4`
     target, behind a fail-closed pairing guard.
   - hf2q's converter learns `Gemma4AssistantForCausalLM` and emits Q8_0 with
     the peer-compatible tensor names.
   - The 26B-A4B checkpoint has **no centroid-masked head**: its config has
     `use_ordered_embeddings: false` and the safetensors carry no
     `masked_embedding.*` tensors. The head is the dense tied 262,144×1,024
     embedding. A checkpoint with `use_ordered_embeddings: true` (the
     E2B/E4B assistants) is refused at conversion, never approximated (§3.3).
5. **The deciding measurement comes first.** ADR-034 measured MoE batched
   verify as not amortizing (Qwen 35B-A3B MoE spec at 0.83× base; Gemma
   26B DFlash at 0.40× base, single slot). On Gemma, the peer's own MTP
   saves only −0.48 s of decode and −0.35 s of makespan (#660 §1). So the
   implementation starts with the R-A verify and its transaction, driven by
   forced drafts, and measures the **round-cost curve M0** before any drafter
   or serving-policy work. Gate G0 (frozen in §8.1) stops the story with a
   recorded falsification if the projected saving is below 0.40 s per run.
   The Class B grouped-`mm_id` alternative needs a new design and an
   operator decision (D1).
6. **Switch.**
   - `--gemma-assistant <gguf>` supplies the drafter.
   - `HF2Q_GEMMA_SPECULATION` takes `0`/`off` (the kill switch) or `mtp`.
   - Before promotion, unset means off. After promotion, unset means `mtp`
     whenever an assistant is supplied.
   - Only greedy requests (temperature 0) speculate. Repetition penalty,
     logit bias, grammar and tool triggers are honoured row by row, through
     the same `decode_tick_finalize` the serial path uses.

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
| Serving worker `run_slot_aware_gemma4`; decode step `decode_batch_gemma4` → `forward_decode_body_batched` (one row per slot, pre-final-norm hidden to host) → `lm_head_batched` → per-row `decode_tick_finalize` | `src/serve/api/engine.rs:11973, 16371, 7788`; `batched_body.rs:2362`; `batched_head.rs:58` |
| Batched flash takes per-query slot id and position, so several queries may address one slot. It is used only when all queries share an `(nwg, nsg)` bucket; otherwise it falls back to per-query dispatch | `batched_body.rs:880-901, 1031-1086` |
| Hybrid KV (default): F16 K, TQ-HB V (FWHT-rotated) plus F32 norms; sliding layers are a ring of exactly `sliding_window` = 1,024 rows; full layers are linear | `src/inference/models/gemma4/kv_cache.rs:373, 1133` |
| Cursor install can move a cursor back; commit refuses to | `engine.rs:17131, 17200` |
| The head's host payload is logits only, by test | `batched_head.rs:28-46` |
| Gemma's default repetition penalty is 1.05, so sampler params are always set and finalize runs `sample_logits_with_grammar` (greedy-with-penalty at temperature 0) | `src/serve/operator_settings.rs:166-170`; `engine.rs:575-640, 7534-7546` |
| The four-agent Gemma workload sends temperature 0 | `scripts/four_agent/family_peer.py:51` |
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
peer runs n-max 3, and so does this design (§5.7).

Safetensors header, 49 tensors, all BF16:
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
(full, 2 KV heads, d512, K==V). Sources: the peer's model-registry memory
construction (`share = is_swa(il) ? n_layer(other) − 2 : n_layer(other) − 1`, lines 3274-3285);
HF `shared_kv_states["sliding_attention"|"full_attention"]`, "the last layer
of each `layer_type`". hf2q's target has full layers at `(i+1) % 6 == 0`
(`src/serve/config.rs` test `:490`), so 28 is sliding and 29 is full.

**Position.** Every draft step uses position `pos0`. The HF doc says "the
assistant predicts all tokens from the same position ID"; the peer uses
`batch.add(id, dp.pos0, …)` (`common/speculative.cpp:1881-1886`).

**Visible keys.** The target's cached rows `p < pos0`. On sliding layers the
target's own sliding mask is evaluated at `pos0`: `pos0 − p < 1024`, so
`p ∈ [pos0−1023, pos0−1]`, which is 1,023 keys (the peer's `is_masked_swa`,
`STANDARD` type, hparams header lines 519-528). [INFERENCE] The HF `flip`ped bidirectional
sliding mask over a sliding cache that holds 1,023 past rows gives the same
set. Test D1 (§7.3) pins it against the HF reference.

**Mapping onto hf2q's kernel.** hf2q's `flash_attn_vec_hybrid` shows "the
last `sliding_window` slots of `[0, kv_seq_len)`"
(`flash_attn_vec_hybrid.metal:489-516`), so the drafter passes
`sliding_window = 1023` for its sliding layers.

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
  [INFERENCE] At about 450 GB/s that is about 1 ms per step, so three steps
  cost ≤ 0.15 of one ordinary step.
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
7. The target runs hybrid KV (`HF2Q_HYBRID_KV` on), the batched decode body
   (`HF2Q_BATCHED_BODY` not `0`), not `HF2Q_DFLASH_XLEN_SDPA`, and not the
   fused head (`HF2Q_FUSE_LMHEAD=1`). The fused body returns no hidden rows,
   so there would be no carry. The FullTq regime, the serial decode path and
   the fused head are refused with "Gemma speculation requires the hybrid
   batched decode body".

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
1. **Entry**, per lane. Check eligibility. Check cursor == `pos0` and that a
   carry hidden row is present (§5.2). On failure the lane stays ordinary.
2. **Draft pass** (§5.2): one command buffer of K steps for all speculating
   lanes. Read back the `K × lanes` draft ids.
3. **Pre-image save** (§5.5) for lanes whose ring has wrapped.
4. **Verify forward** (§5.3) over all windows and ordinary rows, then
   `lm_head_batched` over all rows.
5. **Finalize walk**, per lane (§5.4), then commit or roll back (§5.5).
6. **Accounting** (§5.9).

### 5.2 Draft pass

- **Carry hidden.** Every target forward already returns each row's
  pre-final-norm hidden to host (`forward_decode_body_batched` → `Vec<f32>`).
  After each tick, the lane keeps the row for its last committed position as
  `carry_hidden` [2816]. That row predicted the lane's `next_token`. A lane's
  first decode tick after any prefill is ordinary, because prefill paths do
  not return the last prompt row's hidden. Prefill code is therefore not
  touched, and the cost is one ordinary tick per request (28 of about 1,650
  emitted tokens per run [arithmetic]).
- **Step 0 inputs.** The tokens are each lane's `next_token`. The rows are
  `carry_hidden` uploaded, then normed with the target's final norm using the
  **same dispatch `lm_head_batched` uses** (`batched_head.rs:92-147`). No new
  host payload is added, so the logits-only gate stays as it is.
- **Each step**, for all lanes in one `[lanes, ·]` batch:
  - target `embed_tokens` gather × √2816 (`model.rs:826`) from a GPU token
    buffer;
  - `dispatch_feature_concat_f32` ×2;
  - the drafter graph of §3.2, using `quantized_matmul_ggml` (Q8_0, m ≤ 4,
    mv) or `dense_gemv_bf16_f32` for BF16;
  - `fused_head_norm_rope_f32` with the full layer's `freq_factors`;
  - `flash_attn_vec_hybrid_batched` over the lane's target layer 28/29
    region, with per-query slot ids, `kv_seq_len` covering `[0, pos0)`,
    `sliding_window` 1023 on sliding layers, scale 1.0 and softcap 0,
    followed by the FWHT sign undo the target applies after hybrid SDPA
    (`batched_body.rs:1092`);
  - the raw `fused_gelu_mul` pipeline as the target uses it
    (`forward_gpu.rs:1932`);
  - output norm, tied head, `dispatch_argmax_f32` into `ids[step]`, and the
    post projection into `h_next`.

  Step i+1 gathers its embedding from `ids[step i]` on the GPU, so the K
  steps need no host sync.
- **Drafts.** Raw drafter argmax, greedy, the same as the peer's
  `--spec-draft-sampling greedy`. No penalty or grammar is applied to drafts:
  they are proposals, and the target's finalize decides.
- **No target-state writes.** The drafter's command buffer declares reads
  only on target KV buffers (`barrier_between`). A3 proves the target state
  is unchanged.

### 5.3 Verify route R-A

`forward_decode_body_batched` is generalized to windows; it keeps its name
and its w=1 dispatch sequence:

```rust
pub(crate) struct DecodeWindow { pub slot: SlotId, pub start_pos: usize, pub tokens: ArrayVec<u32, 4> }
pub(crate) fn forward_decode_body_batched(&self, windows: &[DecodeWindow], regime: &mut BatchedKvRegime<'_>,
    head_out: &mut Option<BatchedHeadOut>, gpu: &mut GpuContext) -> Result<Vec<f32>>  // [Σw, hidden], window-major
```

Rows are window-major: row r is `(slot_r, pos_r)`, with rank `j_r` inside
its window. Per layer:

- **Position-independent ops** (norms, fused MLP, MoE, post norms) run
  batched over all R rows exactly as today. MoE stays on the per-token
  `mv_id` route, which is row-identical at any N
  (`batched_body.rs:2053-2063`), or on Q4-GEMMA-DECODE's bitwise grouped
  `mv_id` if it lands.
- **Dense projections.** `dispatch_dense_rowident` (`batched_body.rs:451`)
  changes its n > 8 branch for quantized weights. Today that branch loops
  m=1 per row; it becomes **m ≤ 8 groups of the m=N `mul_mv`**, which is
  byte-identical to m=1 while m ≤ `MM_ROUTING_THRESHOLD` (`:440-447`). F16,
  F32 and BF16 weights keep the per-row loop. For n ≤ 8 nothing changes.
- **Q/K norm and RoPE**: one batched dispatch over all rows with per-row
  positions, as today.
- **Full layers.** KV-encode all R rows (append-only; each row's K/V bytes
  are its own), then one batched flash over all R queries with per-query
  slot id and position. Each query's extent is `pos_r + 1`, and its keys
  `≤ pos_r` are exactly the bytes M1 would have written. The batched kernel
  is used only if every query shares the `(nwg, nsg)` bucket; otherwise each
  query is dispatched alone, which generalizes the existing mixed-bucket
  fallback (`:891-901`).
- **Sliding layers** (ring of 1,024): for rank `j = 0 .. w_max−1`,
  KV-encode the rank-j rows of every window, then run the batched flash for
  those rows, then the FWHT undo. Phase j's ring write
  `(pos0+j) mod 1024` overwrites position `pos0+j−1024`. Ranks below j have
  already attended, and ranks at or above j do not read it
  (`pos_r − 1023 > pos0+j−1024`), so every query reads exactly M1's key set
  with M1's bytes. Ordinary rows are rank 0.
- **Graft.** Sliding rings are prompt-indexed (`batched_body.rs:2414-2420`).
  A lane with a bound graft is not eligible (§5.7), so a speculative tick
  never mixes graft arithmetic into windows.

The head `lm_head_batched` over R rows is "per-row bit-identical to the
scalar head by row-identical dispatch" (`batched_head.rs:57`), in m ≤ 8
groups. Its logits buffer is read lazily, row by row, during the finalize
walk. No logits beyond the first rejected row are copied into new host
vectors [LLD; M0 measures the readback].

**Why this is Class A by construction.** Each row's arithmetic is a
composition of operators that are already proven row-identical across N
(the `slot_aware_n1`/`n4`/`n8` bars), applied to inputs that are bitwise
M1's:
- the same token embeddings;
- K/V rows of earlier positions in the window computed by the same per-row
  arithmetic;
- for sliding layers, the same ring bytes at read time.

A1 (§7.3) is the gate, and a single differing byte falsifies R-A.

**Cost structure** [INFERENCE, measured by M0]:
- dense weights are read ⌈R/8⌉ times;
- MoE expert reads scale with R unless grouped;
- attention scales with R, the same per query as M1;
- sliding layers add (w_max − 1) × 3 dispatches per layer (KV-encode,
  flash, undo), about 225 per verify at w=4.

### 5.4 Finalize walk and commit

For each speculating lane, in slot order (lanes are independent):

```text
for j in 0..w:                                  # w = K+1 rows
    out_j = decode_tick_finalize(lane, greedy_j, logits_row_j)   # the serial function, unchanged
    stream out_j (fragment, reasoning flag) exactly as the serial tick does
    if out_j.finished: committed = j+1; break           # EOS / stop string / grammar-dead / max_tokens
    if j == w-1: committed = w; break                   # bonus row: next_token = its decision
    if lane.next_token != d_j: committed = j+1; break   # correction: next_token = target's decision
committed rows = pos0 .. pos0+committed-1; cursor := pos0 + committed
carry_hidden := hidden row (pos0 + committed - 1)
```

`decode_tick_finalize` (`engine.rs:7788`) is the serial path's finalize. It
applies, in this order:
1. logit bias;
2. `sample_logits_with_grammar` with `self.generated_tokens`, so the
   penalty window includes earlier rows of this window;
3. grammar accept, EOS, the tool-call splitter (`ToolCallOpen` →
   `rt.trigger()`), stop strings and grammar-dead.

The walk calls it on row j only if rows `< j` were accepted. Each call
therefore sees the same logits (§5.3) and the same request state as the
serial tick at that position.

This gives **transaction exactness by identity**. ADR-065 :90 asks the same
of Qwen's K3 through `plan_qwen35_verified_block`; here no separate planner
or semantic state exists to drift.

**Delivery.** Fragments are emitted as each row is finalized, so a round's
1–4 tokens reach the stream in one burst. SSE ordering is the serial order.

### 5.5 Rollback and the ring pre-image

After commit, a lane's state must equal the serial path's at cursor
`c = pos0 + committed`:

- **Full layers.** Rows `≥ c` hold stale bytes. Later writes overwrite
  them, so only the cursor moves, **provided** every reader bounds its reads
  by the cursor. Decode (extent `pos+1`) and the drafter (`[0, pos0)`) do so
  by construction. Warm resume, anchors and prefix-cache snapshots must read
  only `[0, cursor)`. Unit test U6b asserts this for every snapshot path the
  worker can reach between rounds, and A2's eight follow-on decodes cover it
  on hardware.
- **Sliding ring.** The rejected row at position r overwrote ring slot
  `r mod 1024`, which held position `r − 1024`. The next decode at `c`
  needs positions `≥ c − 1023`, so every rejected row `r ≥ c+1` destroyed
  needed history. Row `r = c` destroyed position `c − 1024`, which nobody
  needs and the next decode rewrites; restoring it too keeps the rule
  uniform.
  - **Save**, before the verify, for each speculating lane whose ring has
    wrapped over the window (`pos0 + K ≥ 1024`): the ring rows at the
    logical positions that exist in
    `max(0, pos0+1−1024) .. pos0+K−1024`, so 1 to K rows. Positions below 0
    were never written and need no pre-image. This covers every sliding
    layer (25) and every payload (F16 K, TQ-HB V codes, F32 V norms) with
    `dispatch_kv_cache_linearize_ring_bytes(src = lane's layer view, dst =
    pre-image, n_heads = 8, src_capacity = 1024, dst_capacity = K,
    history_len = rows, logical_end = pos0+K+1−1024, row_bytes)`. The
    kernel's mapping is `src_slot = (logical_end − history_len + i) mod
    src_capacity` (`kv_cache_copy.metal:435-456`), and the bytes are opaque
    per row, so one primitive serves all three payloads. Size: 25 × K ×
    (8 × (512 + 256 + 4)) B ≈ 0.45 MiB per lane at K=3 [arithmetic; the
    V-code row bytes follow `effective_gemma_tq_codebook_bits`].
  - **Restore**, after commit, if `committed < w`: rows `c .. pos0+K`
    return to their pre-images through the inverse primitive (below).
  - The verify's rank-0 write at `pos0` overwrote `pos0−1024`, which is
    never needed at cursor `≥ pos0+1`, so it has no pre-image.
- **New mlx-native primitive N1**, `dispatch_kv_cache_restore_ring_bytes`.
  It is the byte-exact inverse of `linearize_ring_bytes`: same arguments,
  writing `dst[h, (logical_end − history_len + i) mod src_capacity] =
  src[h, i]`.
  - Contract: `restore ∘ linearize` is the identity on the touched rows and
    every other byte is unchanged, for wrap and no-wrap and for 1–4 rows,
    through both metallib paths.
  - Native policy: no external engine names. It is published and pinned
    before the hf2q integration lands.
  - A batched-over-seqs variant that cuts the dispatch count from
    lanes × 75 to 75 is allowed under the same contract; M0 decides whether
    it is needed.
- **Cursor.** `install_gemma4_slot_cursor(slot, c)` (`engine.rs:17131`),
  then `validate`: target cursor == committed tokens + graft + prompt. The
  assistant has no cursor or state of its own, so there is nothing else to
  roll back.

### 5.6 Failure semantics (ADR-065 :91, carried over)

- **Lane-local refusal before submission** (entry or draft-pass error): that
  lane runs as an ordinary row in the same tick, with state untouched.
- **A failure of the submitted shared verify or restore**: every
  participating lane is reset through the existing Gemma slot-reset path,
  and its request terminates with an error. Non-participants' state and
  committed prefixes are unmutated, because their rows are rank 0 and write
  only their own `pos` row, exactly as an M1 tick would.
- **A failure after the verify but before restore completes**: worker-fatal,
  fail-stop. No slot state is reused, because the ring may hold
  unrestorable rows.
- Cancellation during a round takes effect after the round, as for an
  ordinary tick.

### 5.7 Eligibility, depth and policy

A lane speculates in a tick iff all of these hold:
1. The switch resolves to `mtp` and an assistant is loaded.
2. The request is greedy: `temperature < SAMPLING_EPS`. Sampled requests
   never speculate and consume no RNG for drafts. `top_k`/`top_p`, penalty,
   bias, grammar and logprobs are all allowed: at temperature 0 they are
   deterministic per-row selections inside `decode_tick_finalize`, and
   logprob records are produced per finalized row.
3. No graft is bound on the slot.
4. `carry_hidden` is present.
5. `max_tokens − generated ≥ 2`. The window is `min(K+1, remaining)`.
6. The cost controller allows it.

**Depth.** K is fixed per release, chosen by M0 from {1, 2, 3} (§8.1) and
frozen before S1. The working value is K=3, like the peer's n-max 3 and
Qwen K3. There is no adaptive depth. Under R-A, depth and the cost gate
cannot change output, only speed.

**Cost controller.** `SpeculationCostController`
(`cost_controller.rs`, `EVALUATION_ROUNDS = 4`,
`UNPROFITABLE_WINDOWS_TO_DISABLE = 2`), one per worker, keyed by the
tick's decoding width (1–4). It compares the round's wall time per emitted
token with the ordinary tick's time per token at the same width, and may
disable speculation for that width until the next request admission. It is
deterministic in its inputs, but timing makes it nondeterministic across
runs. That is allowed because transcripts do not depend on it (A6).

**Mixed steps.** Speculation runs in `Decode` and `Mixed` steps alike. The
pre-declared alternative arm `HF2Q_GEMMA_SPECULATION=mtp-decode-only`
suppresses windows in a tick that also runs a prefill quantum. It exists
only if S1's TTFT fairness fails (§8.3, decision D7).

### 5.8 Switch, default, kill switch

`HF2Q_GEMMA_SPECULATION` is resolved once per worker:

| Value | Before promotion | After promotion (§8.6) |
|---|---|---|
| unset, assistant supplied | off | `mtp` |
| unset, no assistant | off | off |
| `0` / `off` / `false` | off (kill switch) | off (kill switch) |
| `mtp` / `1` / `on` | `mtp` (refuses to start without `--gemma-assistant`) | `mtp` |
| `mtp-decode-only` | exists only under D7 | as decided |
| anything else | warning, then off | warning, then the default |

Model load fails closed when `mtp` resolves but the assistant or any of its
pipelines (drafter kernels, the N1 primitive, the sliding phase kernels)
cannot be built or prewarmed. There is no silent fallback to off.

### 5.9 Observability and telemetry

- **Startup log:** `Gemma speculation mode=mtp depth=3 assistant=<sha256>
  quant=Q8_0`.
- **`/hf2q/v1/runtime` `active_controls`** (`src/serve/api/measurement.rs`):
  `gemma_speculation`, `gemma_speculation_depth` and
  `gemma_assistant_sha256`, so that Q5 receipts bind the arm and the drafter.
- **Prometheus**, mirroring Qwen (`handlers.rs:8684-8745`):
  - `hf2q_gemma_speculation_{rounds,drafted_tokens,accepted_tokens,rejected_tokens,ring_restores,cost_disabled}_total`;
  - `…_round_seconds_total`, `…_equivalent_ordinary_seconds_total`;
  - `…_fallback_requests_by_reason_total{reason}`;
  - `hf2q_gemma_decode_ticks_by_width_total{width,speculative}`. This one
    is also the histogram M0's projection needs from a spec-off baseline
    (§8.1).
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
| Rollback store | **A** | After a round with *a* accepted: every layer's KV bytes (ring included) and cursors are `ExactEqual` to the serial path after a+1 tokens |
| Draft side effects | **A** | The drafter pass changes no target byte and no cursor (Q1 §3.1(5): "draft generation and discard must not change committed state") |
| Serving transcripts, spec on vs off | **A** | `ExactTranscripts` (`q1_decision_policy.py transcripts`) in every S1 block. Quality is then inherited (Q1 §3.6) |
| Drafter logits | outside Q1 (proposal-only, never committed) | Correctness is checked against an F64 reference of §3.2 on identical inputs (D1, §7.3). Acceptance is a speed quantity (§8.2) |
| hf2q vs the peer | **C** | Not a gate on rows. The Q4 paired quality gates apply on the combined candidate |

Two notes on scope:
- Nothing in this story is Class B, so there is no envelope, near-tie or
  canonical-coordinate test. If R-A cannot pass G0, any route that would
  make verified rows Class B (grouped `mm_id` or `mm` above 8 rows, or the
  prefill multi-seq path) is **out of this design**. It needs its own Class B
  design: envelope over every verified row against teacher-forced M1, the
  E2 coordinate, and the paired 24-instance quality campaign (D1).
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
| U2 | `gemma4_assistant_pairing_guard_matrix` | Accepts the §4.2 pair; refuses each of rules 1–7 in isolation with its message; the kata test plus the drafter-contract assertion |
| U3 | `gemma_speculation_switch_table` | §5.8 table before and after promotion; `mtp` without an assistant fails startup |
| U4 | `gemma_speculation_eligibility` | Sampled is refused (and no RNG is drawn); penalty, bias, grammar and logprobs allowed; graft refused; budget window `min(K+1, remaining)`; first tick after prefill is ordinary |
| U5 | `gemma_spec_walk_matches_serial_finalize` | With synthetic logits, every combination of full, partial and zero acceptance, EOS, stop string, grammar trigger and grammar-dead inside the window, `max_tokens` at each row, and penalty sensitivity across rows (a row whose argmax flips only because of an earlier row's token in the window) gives fragments, request state and cursor equal to serial `decode_tick_finalize` calls |
| U6 | `gemma_spec_ring_preimage_plan` | For each `(pos0, K, committed)`, including pos0 ∈ {1020..1026} and deep wrap: the saved logical rows (clipped at 0), `history_len`/`logical_end`, the restored rows, and no save before wrap |
| U6b | `gemma_spec_snapshot_paths_bound_by_cursor` | Every anchor, prefix-cache and warm-resume snapshot path reachable between rounds reads only `[0, cursor)` of full layers and only the window ending at the cursor of sliding rings |
| U7 | `gemma_spec_failure_semantics` | Injected lane-local refusal leaves the lane ordinary; a shared verify failure resets participants and leaves non-participants unmutated; a failure after verify fail-stops |
| U8 | `gemma_spec_window_rows_layout` | Window-major row order, ranks, per-row positions, dense grouping ⌈R/8⌉, and full-layer bucket fallback decisions |
| U9 | `gemma_spec_metrics_and_controls` | Metric names and runtime control fields |

### 7.2 mlx-native

| Id | Test | Asserts |
|---|---|---|
| N1 | `kv_cache_restore_ring_bytes_inverts_linearize` | Bitwise round trip for row_bytes ∈ {4, 256, 512, 1024}, heads 2 and 8, capacity 1024, history 1–4, wrap and no-wrap, plus untouched-byte checks; both metallib paths (default and `MLX_PRECOMPILED_METALLIB=0`) |

### 7.3 Real artifact (target `bf65b7e1…`, assistant from §4.1)

| Id | Test | Required |
|---|---|---|
| A1 | `gemma_spec_verify_rows_match_serial_m1_bitwise` | For windows w = 1..4, teacher-forced tokens from a frozen greedy transcript, and cohorts {one window}, {4 windows w=4}, {4,1,3,2}, {2,4} plus two ordinary rows, in both slot orders, at positions below 1024, crossing 1024, deep (≥ 2,048), and straddling an `(nwg, nsg)` bucket edge: logits rows, hidden rows and all KV bytes written equal serial M1 decodes. Bitwise |
| A2 | `gemma_spec_rollback_store_matches_serial` | For a ∈ {0, 1, 2, 3} with forced drafts, wrapped and unwrapped: after commit, the full slot store (every layer, ring, cursors) `ExactEqual` serial after a+1 tokens, and the following 8 ordinary decodes are bitwise equal |
| A3 | `gemma_spec_draft_pass_is_side_effect_free` | Target store bytes and cursors are unchanged by a drafter pass. Bitwise |
| A4 | `gemma_spec_ordinary_rows_invariant` | Ordinary rows in a speculative tick equal the same rows in an all-ordinary tick. Bitwise |
| A5 | existing `slot_aware_n1`, `slot_aware_n4`, `slot_aware_n8_per_slot_parity_vs_serial` and the Gemma joint warm prefill Class A gates | Pass unchanged. The w=1 path is today's |
| A6 | Q5 S1 arms, `q1_decision_policy.py transcripts` on vs off, per block | `ExactTranscripts` |
| A7 | `gemma_spec_fail_closed_injection` | §5.6 on hardware, with an injected verify error |
| D1 | `gemma_assistant_drafter_reference_parity` | Drafter logits and `h_next` against an F64 numpy reference of §3.2 computed from the GGUF's dequantized weights and **hf2q's own exported** target layer 28/29 K and dequantized V plus the carry row, at 3 positions (pos0 < 1023, pos0 = 1023/1024/1025, deep), 3 steps. Frozen bounds: rel-RMS ≤ 1e-4 and top-1 equal on every row whose reference top-2 margin is ≥ 0.01. The sliding-window variant (1023 vs 1024 keys) is pinned against the HF reference code on the same arrays; the design default is 1023 (§3.4). Any change is a recorded fix, never a threshold change |
| D2 | `gemma_assistant_acceptance_replay` (diagnostic; M1, §8.2) | Teacher-forced replay over the recorded spec-off four-agent transcripts. For every decode position: draft K=3 from the committed prefix, and record per-depth acceptance, L(K) for K = 1..3, and Q8_0 vs BF16 |

## 8. Measurements and gates

Every hardware run records:
- the source commit and binary SHA-256;
- the target and assistant digests;
- the mlx-native version and metallib SHA-256;
- the case file and its SHA-256.

Every run passes Q5 `CalibrationPass` with the accepted negative-control
receipt. Evidence goes to `~/.local/state/hf2q/v0130/q4-gemma-spec-serve/`.
Only the root, holding the GPU lane, runs these, one model process at a time.

### 8.1 M0: round-cost curve and gate G0 (before drafter and policy work)

Harness: `gemma_spec_round_cost_curve`, a real-artifact ignored bench. It
builds 4 live slots from the canonical turn-2 prompts (contexts
about 700–1,400 rows; the #660 per-turn table) and times, as the median of
200 ticks after 20 warm-up ticks:
- `t1(n)`: the ordinary tick at n = 1..4 decoding slots;
- `tv(w, n)`: the R-A tick with n speculating windows of width w ∈ {2, 3, 4}
  (forced drafts);
- `ts`: the pre-image save, and `tr`: the restore;
- `td(K, n)`: the drafter pass. It uses the hosted Q8_0 GGUF as a stand-in
  until §4.1's artifact exists; its Q5 and pairing receipts are recorded.

Each figure includes host walk and readback. The round-cost ratio is
`c(K, n) = (td + ts + tv(K+1, n) + E[tr]) / t1(n)`.

**Projection** [frozen formula]:
`ΔD = Σ_n D·h(n)·(1 − c(K, n)/L(K))`, where:
- `D` = 9.203 s, hf2q's L1 decode median (§2.1);
- `h(n)` = the share of decode ticks at width n, from
  `hf2q_gemma_decode_ticks_by_width_total` in a spec-off Q5 run of the same
  binary;
- `L(K)` = the peer-measured per-depth acceptance (L(3) = 3.34, §2.2) until
  D2 replaces it with hf2q's own.

K* is the K ∈ {1, 2, 3} that maximizes ΔD.

**G0 (frozen).** `ΔD(K*) ≥ 0.40 s`, the lower end of #660 §7's estimate.
At D = 9.20 s and L = 3.34, that means c ≤ 3.19 at width 4 [arithmetic].

- **On a miss**, the story stops. Record M0 and the falsification, and keep
  the R-A code only if A1–A7 passed and the operator wants it (D1). Q4's
  Gemma budget then loses this line (#660 §7).
- **On a pass**, M0 also fixes K* and whether the batched N1 variant is
  needed (save + restore > 3% of `tv`).

### 8.2 M1: acceptance replay (after the drafter, before S1)

D2 over the 28 spec-off turns of three Q5 runs. It reports per-depth
acceptance, L(K*) and the Q8_0-vs-BF16 difference.

Decision rule (frozen):
- Q8_0 ships unless its accepted-per-drafted falls more than 0.02 below BF16
  (D2).
- G0 is recomputed with hf2q's L. A miss there is the same stop as §8.1.
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
- verify buffers for 16 rows.

A failure blocks promotion.

### 8.6 Promotion

`mtp` becomes the default when an assistant is supplied (§5.8) only when all
of these hold:
- N1 is published and pinned;
- U1–U9 and A1–A5, A7 and D1 passed;
- G0 passed on M0 and again on M1's L;
- S1-a through S1-d passed;
- the Q5 recalibration passed.

The promotion commit:
- flips the default;
- keeps `0` as the kill switch;
- updates the §5.10 surfaces;
- records the receipts in this document and in ADR-063.

The distribution of the assistant (D3) must be settled first, because the
Q4 Win counts only a published configuration.

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
| R1 MoE does not amortize under per-token `mv_id` (ADR-034 evidence) | G0 miss | M0 first. Q4-GEMMA-DECODE's bitwise grouped `mv_id` (≤ 16 tokens × top-8, agreed) amortizes expert reads while staying Class A, and M0 is re-run on it if it lands. K* may be < 3. The Class B route is D1 |
| R2 Dispatch overhead: about 225 sliding-phase dispatches, pre-image save/restore (lanes × 75 without the batched N1), 16 MiB logits readback | Raises c | Measured inside c; the batched N1 variant; lazy row readback |
| R3 Lower acceptance than the peer: hf2q's V is TQ-HB quantized, while the peer's KV is F16 and the drafter was trained on exact KV; Q8_0 drafter; sliding-window choice | Lower L | D2 measures it; BF16 is the fallback (D2 rule); the D1 window pin |
| R4 SSE/TTFT fairness: a round is about c× an ordinary tick, and Mixed steps delay prefill quanta | S1-d failure | Bursts deliver several tokens per gap; `mtp-decode-only` is pre-declared (D7); the cost controller |
| R5 Cohort or bucket edge cases (mixed `(nwg, nsg)` buckets across window ranks; graft) | A1 failure | Per-query fallback generalizes the existing one; graft lanes are ineligible; A1 covers bucket edges |
| R6 Ring restore omission (a path that snapshots or reads the ring between verify and restore) | Silent state corruption | Restore runs in the same worker step before any other slot work; A2 includes the next 8 ordinary decodes; U6 |
| R7 Interaction with siblings: #668 changes warm prefill only; Q4-GEMMA-DECODE changes decode kernels bitwise | R-A stays Class A only if M1 and verify call the same kernels | Both use `forward_decode_body_batched`; A1 runs on the final combined candidate too |
| R8 Distribution: a Win counted on an operator-supplied drafter path | Q4 validity | D3 |
| R9 Peer fairness: hf2q's Q8_0 drafter could also help the peer | Unfair comparison | D4 |

## 10. GPU validation plan (root, in order; each step gates the next)

1. **mlx-native N1:** parity on both metallib paths; release; crates.io
   verification; hf2q pin.
2. **R-A with forced drafts:** A5 (unchanged M1), then A1, A2, A4 and A7.
3. **M0:** round-cost curve, G0, and K*. Stop on a miss (§8.1).
4. **Converter and drafter:** convert the assistant (Q8_0 and BF16) with
   receipts; A3; D1 (reference parity; window pin).
5. **M1:** D2 replay over the spec-off Q5 runs; G0 on hf2q's L; Q8_0 vs
   BF16.
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

- A written design, approved before implementation, states the verify route, rollback, switch and kill switch: `docs/research/adr063-q4-gemma-spec-serve-design.md`. In it: `--gemma-assistant` supplies the drafter; `HF2Q_GEMMA_SPECULATION=mtp` enables speculation; it is off by default until every gate passes; `0` is the kill switch; only greedy requests speculate.
- The drafter loads only through a drafter contract paired to a `gemma4` target, behind a fail-closed pairing guard. `gemma4-assistant` stays out of the primary runtime contracts. The 26B-A4B checkpoint's head is the dense tied embedding (`use_ordered_embeddings: false`); ordered-embedding (centroid) assistants are refused, never approximated.
- The round-cost curve is measured on the verify route before drafter and policy work. Gate G0: the projected decode saving at the chosen depth must be ≥ 0.40 s per run (the lower end of the Q4-GEMMA booked 0.4–0.5 s). A miss ends the story as a recorded falsification, and any Class B alternative needs a new design.
- Class A gates pass:
  - verified rows are bitwise equal to teacher-forced serial decode rows in every cohort;
  - ordinary rows are invariant;
  - a round with *a* accepted drafts leaves every KV byte (sliding ring included) and cursor equal to serial decoding of a+1 tokens;
  - the drafter pass leaves the target state unchanged;
  - serving transcripts with speculation on and off are `ExactTranscripts` in every A/B block.
- Same-binary A/B under the ADR-065 two-arm rules (12 ABBA blocks): verdict Win, median makespan reduction ≥ 0.30 s and ≥ 60% of the M0 projection, decode stage −0.35 s or more, and paired TTFT/SSE fairness within 1.025 against the speculation-off arm. A fresh Q5 calibration of the target plus assistant artifact set passes.
- Every native primitive (the byte-exact ring-row restore) is published in mlx-native and consumed at an exact pin.

Repository ownership: hf2q owns the converter, drafter contract, serving transaction, policy, telemetry and campaigns. mlx-native owns the ring-row restore primitive, its parity tests and release.

Validation: Independent review of the design, the Class A receipts, the M0 round-cost curve and G0 decision, the acceptance replay and the A/B receipts.

Predecessors: Q4-GEMMA (#660, profiling result), Q5 (#371). Q4-GEMMA-ATTN's closing campaign consumes the accepted configuration.

Priority: High. Core scope (second-tier family).

Delivery boundary: Release v0.1.30, or the first published hf2q release containing the accepted Gemma 4 configuration; a G0 miss delivers the recorded falsification instead.

## 13. Decisions for the operator

1. **D1, the verify route.** This design approves only R-A (Class A,
   bitwise). If G0 misses, do you want a separate Class B grouped-verify
   design (`mm_id`/`mm` above 8 rows: envelope over every verified row, E2
   coordinate, paired quality campaign), or a recorded falsification with
   the lever dropped from the Gemma plan? And should the R-A code be kept or
   removed in that case? The recommendation is to remove it unless a Class B
   design is opened.
2. **D2, the drafter artifact.** hf2q converts `6e5aaaf4` to Q8_0 2-D
   weights with F32 norms and peer-compatible names. BF16 is used only if
   the M1 acceptance drop exceeds 0.02. Approve Q8_0 as the default?
3. **D3, distribution.** How does the released configuration get the
   assistant?
   - (a) The operator passes `--gemma-assistant` through the launcher and
     the Q5 case.
   - (b) hf2q's managed or hosted artifact resolution fetches the hf2q-built
     drafter beside the target.

   A Q4 Win counts only a published configuration, so (b), or a documented
   (a), must be in place before promotion.
4. **D4, peer fairness.** hf2q's drafter GGUF uses the peer's tensor names.
   Re-qualify the peer's MT arm with it (as a tie-break arm against the
   BF16 `e1145ee2…` arm) and keep the faster as the peer arm of record?
5. **D5, the default after promotion.** `mtp` when an assistant is supplied
   and off otherwise, with `0` as the kill switch. Approve?
6. **D6, the G0 threshold.** 0.40 s projected (c ≤ 3.19 at width 4 with the
   peer's L). Lowering it would accept a lever worth less than #660 books.
7. **D7, Mixed-step fallback.** Pre-declare `mtp-decode-only` (no windows in
   ticks that run a prefill quantum) as the one extra arm if S1's TTFT
   fairness fails?

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
