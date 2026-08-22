# ADR-041 — Qwen3-VL Text-LM Engine-Seam Wire-Up (iter-9b)

- **Status**: Accepted; execution in progress. The runtime remains fail-closed at
  spawn until every acceptance criterion below is proven together.
- **Updated**: 2026-08-22 — code audit added the native-matrix, host-readback,
  model-generation, and cross-family serving contracts. The current guarded
  text implementation expands the F16 token table to F32 and downloads the
  full table for each forward. For the 151,936 × 2,048 canonical table, the
  expansion alone adds 622,329,856 bytes (593.5 MiB). Those are rejected
  implementation details, not an accepted cost of multimodal inference.
- **Created**: 2026-05-30
- **Owner**: Robert E. Lee
- **Related**:
  - **ADR-005** Decision #1 — engine seam shape (`worker_run` dispatch)
  - **ADR-040** §6.1.27→§6.1.55 — Continuous-batching multi-seq KV scaffold; this ADR closes the **Qwen3-VL specific gap** ADR-040 §6.1.52 (iter-C2e) + §6.1.55 (iter-C2e-cont) explicitly typed-deferred onto a separate ADR.
  - **`src/inference/models/qwen3vl_text/forward.rs`** — `forward_text_prefill_logits_last` (iter-8a-2 LANDED) — the dense forward path this ADR wires through the engine seam.
  - **`src/serve/api/engine_qwen3vl.rs:245-287`** — `handle_qwen3vl_slot_aware_n_gt_0_sentinel` — the existing witness take/restore harness this ADR replaces with real cache dispatch.

---

## 1. Problem Statement

`qwen3vl_text_forward_pending_err` (sentinel at `src/inference/models/qwen3vl_text/forward.rs:130`) is still the single dispatch arm called by `serve/api/engine.rs::worker_run` for every `LoadedModel::Qwen3VlText` request — Generate / GenerateStream / GenerateWithSoftTokens / Embed all return HTTP 501 with the pending message.

The forward function itself (`forward_text_prefill_logits_last`) is **shipped and tested** (iter-8a-2). What's missing is the **engine seam wire-up**: the worker-arm dispatch, the persistent KV cache type (`Qwen3VlTextKvCache` does not exist yet), the multi-seq sibling for continuous batching (`MultiSeqQwen3VlKvBuffers` does not exist yet), and the four-arm worker dispatch (Generate / GenerateStream / Embed / SoftTokens) into the forward path.

This is the **same engine-seam pattern** Qwen35 and Gemma 4 already use end-to-end (ADR-040 iter-C2d-cont-kernel-iter-{1,2,3,4} for Qwen35; iter-B4c-kernel-iter-{1..5}/iter-2-decode-* for Gemma 4). The work scales similarly: ~2000-5000 LOC of structural lift across:

- forward path: KV-incremental decode loop (today is full re-prefill per token)
- KV cache type: persistent + multi-seq sibling
- engine worker arms: 4 worker-arm dispatchers
- sampler / grammar / stop-strings / logprobs surface (mirrors iter-2-decode-C)
- vision splicing: `<|image_pad|>` override-row pass through ViT-projected embeddings (iter-8a-2 punted this; iter-9b adds it)
- DeepStack injection for image grids (already exists in the guarded forward,
  but still needs to be plumbed through the engine seam)
- native mapped matrix ownership: the token embedding and tied/untied output
  head remain in their GGUF representation; no matrix is expanded, dequantized,
  or requantized during load or inference
- device-side embedding gather and output projection: only final result data
  may cross back to the host, never the full token table or hidden matrix
- first-class model-generation ownership for an atomic text+projector pair,
  including exact A+P_A→B→C+P_C→A+P_A replay and memory reclamation

## 2. Why Carved Out of ADR-040

ADR-040 closed Phase E1 with `KEEP-SerialFifo` and shipped continuous-batching scaffolding for two architectures (Qwen35 + Gemma 4) that **already had working engine seams**. Adding a **third architecture whose engine seam doesn't exist** is structurally out of scope:

1. **ADR-040 invariant** — every iter is "lift existing single-seq engine seam onto multi-seq cache". Qwen3-VL has no single-seq engine seam to lift.
2. **Multi-day vs multi-iter** — Qwen35 + Gemma 4 worker lifts ran 4 + 5 iters each (~1500-3000 LOC per arch). A from-scratch engine seam plus the dense forward + KV cache + multi-seq plumbing is closer to 2000-5000 LOC and structurally a single ADR's worth of work.
3. **Path-B-clamp is the explicit ADR-040 contract** — iter-C2e (§6.1.52) shipped the spawn-time activation, iter-C2e-cont (§6.1.55) shipped the witness take/restore harness, and BOTH typed-deferred the forward-path replacement onto **"a separate ADR per iter-228a-followup"**. This ADR is that separate ADR.

## 3. Acceptance Criteria (when work starts)

- **AC-1** — `qwen3vl_text_forward_pending_err` callers DELETED from `serve/api/engine.rs::worker_run`. All 4 dispatch arms (Generate, GenerateStream, GenerateWithSoftTokens, Embed) call into real forward paths.
- **AC-2** — `Qwen3VlTextKvCache` type exists with persistent-rollback semantics (mirror `Qwen35Model::persistent_kv_cache`).
- **AC-3** — `MultiSeqQwen3VlKvBuffers` sibling struct exists, implements `MultiSeqKvCache` trait, supports `reset_for_slot` (mirror `MultiSeqHybridKvCache` for Qwen35).
- **AC-4** — `Qwen3VlTextLoadedModel::handle_qwen3vl_slot_aware_n_gt_0_sentinel` (currently returns 501) is DELETED OR replaced with real `forward_with_slot_id(slot_id, …)` dispatch. ADR-040 §6.1.52/§6.1.55 cross-references updated.
- **AC-5** — KV-incremental decode loop replaces the per-token full re-prefill. Single-seq throughput should be within 1.0× of Qwen3.5/3.6 dense (the architectures have similar hidden + layers).
- **AC-6** — Vision splicing (override-row pass for `<|image_pad|>` positions with ViT-projected embeddings) closes the iter-8a-2 punt. Functional acceptance: image+text prompt returns coherent output (not gibberish from un-replaced placeholder embeddings).
- **AC-7** — DeepStack injection (`cur += t_inp_embd_slab[il+1]` for
  `il < n_deepstack_layers`) is plumbed through the engine seam for
  image-augmented prompts.
- **AC-8** — N=4 SlotAware bench (mirror ADR-040 §6.1.56 D3 AC-4 harness with `--model <qwen3vl-gguf>` instead of Qwen3.6-A3B Q4_0) returns measurements without 501s. Throughput target relaxed to ≥0.5× SerialFifo at N=4 (Qwen3-VL dense, not MoE — likely closer to SerialFifo parity than MoE's 0.99× because of dense-attention KV-cache hit pattern).
- **AC-9** — Every matrix role, including `token_embd.weight` and the tied or
  dedicated output head, is loaded in its native GGUF storage through shared
  mapped ownership. Preflight rejects an unsupported role/codec before Metal
  allocation. A source canary proves the production loader contains no
  matrix-sized `load_tensor_f32`, dequantize, or requantize fallback.
- **AC-10** — Embedding gather, soft-token row replacement, and final output
  projection stay on device. The production forward never downloads the full
  token table, residual stream, or normalized hidden matrix. Only requested
  embeddings, sampled token/logit results, and bounded diagnostics may cross
  to the host.
- **AC-11** — The public model selector resolves one server-private atomic
  generation `{text, digest-matched projector}`. A missing or mismatched
  projector fails before publication. The exact
  A+P_A→B→C+P_C→A+P_A gate proves identical first semantic output after reload,
  no stale tokenizer/template/cache/projector state, one-generation byte
  accounting, switch-to-first-result latency, and RSS/wired reclamation.
- **AC-12** — ADR-049 multi-anchor state reuse and any row-aggregation lever
  whose family spike passes are present at engine bring-up, rather than added
  as later family-specific work.
- **AC-13** — Matched current-pinned-reference parity and performance run on
  the same artifact, prompt, settings, context, and hardware. Coherence is a
  hard gate; accepted single-slot and N=4 throughput must be at least parity,
  with cold-load, post-warm, first-semantic, and steady-state timings reported
  separately.

## 4. Out of Scope

- **External**: tooling for fine-tuning Qwen3-VL ViT projector weights. Iter-8a-2 loads pre-trained projector from GGUF; this ADR uses those weights unchanged.
- There is no performance carve-out for the current CPU permutation or host
  embedding-table path. If the engine seam needs a `head_dim=128` attention
  kernel or another native primitive to meet AC-10/AC-13, that primitive is
  part of this execution.
- **CFA empirical regime gate** — N=8+ regression prediction (per ADR-040 §6.1.55 dossier) applies to MoE architectures; the prediction for Qwen3-VL dense is structurally different and needs its own bench before any cross-ADR claim.

## 5. Open Questions

1. **KV cache shape** — `Option<Qwen3VlTextKvCache>` (Qwen35-shaped) vs `Vec<MultiSeqQwen3VlKvBuffers>` (Gemma 4 shaped)? Architecture is dense not MoE; ADR-040 §6.1.52 left this OQ for this ADR to decide. Default lean: Qwen35-shaped (simpler; matches the dense vs MoE distinction).
2. **Vision splicing timing** — override-row pass before or after embed_tokens? Iter-8a-2 deferred this to iter-9b; the engine-seam decision picks which.
3. **Per-image-grid IMROPE recomputation** — iter-9b's KV-incremental decode loop needs to recompute positions when an image hits the KV window; needs design before AC-5.
4. **Soft-token KV invalidation** — when a soft-token replaces an `<|image_pad|>` row, does the KV cache for that position need explicit invalidation? Open per ADR-040 iter-228a punt.

## 6. Sequencing

1. Add native mapped matrix ownership and device-side embedding/head primitives;
   prove no implicit transform or whole-table readback before opening serving.
2. Add `Qwen3VlTextKvCache`, persistent rollback, the multi-sequence sibling,
   and model-neutral ADR-049 state reuse.
3. Implement incremental prefill/decode with device-side vision splicing and
   DeepStack injection; run exact text and image forward parity.
4. Wire Generate, GenerateStream, Embed, and GenerateWithSoftTokens; delete all
   pending sentinels in the same change.
5. Publish text+projector as one generation and pass the exact A/B/C/A swap,
   reclamation, unary/SSE, tool, cache, and isolation gates.
6. Run physical N=1/2/4/8/16 where the artifact fits, matched-reference
   parity/performance, tail latency, memory, and thermal gates; update this ADR
   from exact receipts before changing Status to Implemented.

## 7. References

- `src/inference/models/qwen3vl_text/forward.rs:212-880` — `forward_text_prefill_logits_last` (iter-8a-2 LANDED)
- `src/serve/api/engine_qwen3vl.rs:245-287` — `handle_qwen3vl_slot_aware_n_gt_0_sentinel` (iter-C2e-cont witness scaffold)
- `src/serve/api/engine.rs:1804` — `LoadedModel::Qwen3VlText` variant
- `docs/adr/ADR-040-continuous-batching-reopen.md` §6.1.52 (iter-C2e), §6.1.55 (iter-C2e-cont) — the two ADR-040 sections that explicitly carved this work out
- Pinned reference source recorded by `data/llama_cpp_pin.txt` — parity and
  DeepStack comparison only; never a production dependency
