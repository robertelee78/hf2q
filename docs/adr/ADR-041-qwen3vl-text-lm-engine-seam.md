# ADR-041 — Qwen3-VL Text-LM Engine-Seam Wire-Up (iter-9b)

- **Status**: 📋 STUB — scope carved 2026-05-30 from ADR-040 §6.1.27→§6.1.55. Implementation deferred (multi-day structural work; not in any single ADR-040 iter). When work starts: this ADR moves to PROPOSED with an iter sequencing table.
- **Updated**: 2026-08-20 — **interim spawn guard landed (guarantees tune-up item 1)**: the serve-side dense-arm dispatch in `serve/api/engine.rs::LoadedModel::load` bails at spawn again (operator-actionable message naming this ADR), replacing iter-228a's load-then-501 state that violated the published "refused up front" guarantee. iter-9b-4 (Generate-arm worker dispatch) DELETES that bail in the same commit that wires the real forward; the pins to flip then are `load_engine_refuses_dense_qwen3vl_until_adr041_engine_seam` + `..._upstream_arch_...` in `src/serve/mod.rs`. Robert approved executing this ADR (2026-08-20, tune-up conversation); targeted post-0.1.7 (0.1.8 scope per release-cut decision).
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
- DeepStack injection for image grids (already exists in forward, but needs to be plumbed through engine seam — peer's `qwen3vl.cpp:146-150`)

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
- **AC-7** — DeepStack injection (peer's `qwen3vl.cpp:146-150` `cur += t_inp_embd_slab[il+1]` for `il < n_deepstack_layers`) plumbed through engine seam for image-augmented prompts.
- **AC-8** — N=4 SlotAware bench (mirror ADR-040 §6.1.56 D3 AC-4 harness with `--model <qwen3vl-gguf>` instead of Qwen3.6-A3B Q4_0) returns measurements without 501s. Throughput target relaxed to ≥0.5× SerialFifo at N=4 (Qwen3-VL dense, not MoE — likely closer to SerialFifo parity than MoE's 0.99× because of dense-attention KV-cache hit pattern).

## 4. Out of Scope

- **External**: tooling for fine-tuning Qwen3-VL ViT projector weights. Iter-8a-2 loads pre-trained projector from GGUF; this ADR uses those weights unchanged.
- **Performance**: Flash Attention prefill kernel for `head_dim=128` — currently CPU-permute path per `forward.rs` comment 3. Optimization headroom but not engine-seam work.
- **CFA empirical regime gate** — N=8+ regression prediction (per ADR-040 §6.1.55 dossier) applies to MoE architectures; the prediction for Qwen3-VL dense is structurally different and needs its own bench before any cross-ADR claim.

## 5. Open Questions

1. **KV cache shape** — `Option<Qwen3VlTextKvCache>` (Qwen35-shaped) vs `Vec<MultiSeqQwen3VlKvBuffers>` (Gemma 4 shaped)? Architecture is dense not MoE; ADR-040 §6.1.52 left this OQ for this ADR to decide. Default lean: Qwen35-shaped (simpler; matches the dense vs MoE distinction).
2. **Vision splicing timing** — override-row pass before or after embed_tokens? Iter-8a-2 deferred this to iter-9b; the engine-seam decision picks which.
3. **Per-image-grid IMROPE recomputation** — iter-9b's KV-incremental decode loop needs to recompute positions when an image hits the KV window; needs design before AC-5.
4. **Soft-token KV invalidation** — when a soft-token replaces an `<|image_pad|>` row, does the KV cache for that position need explicit invalidation? Open per ADR-040 iter-228a punt.

## 6. Sequencing (TBD when work starts)

- iter-9b-1: `Qwen3VlTextKvCache` type + persistent rollback scaffold
- iter-9b-2: `MultiSeqQwen3VlKvBuffers` sibling + `MultiSeqKvCache` impl
- iter-9b-3: forward path KV-incremental decode loop (mirror Qwen35 incremental path)
- iter-9b-4: Generate-arm worker dispatch in `engine.rs::worker_run`
- iter-9b-5: GenerateStream-arm worker dispatch
- iter-9b-6: Embed-arm worker dispatch
- iter-9b-7: GenerateWithSoftTokens-arm worker dispatch (vision splicing)
- iter-9b-8: DeepStack injection plumbing for image-augmented prompts
- iter-9b-9: Delete `qwen3vl_text_forward_pending_err` + `handle_qwen3vl_slot_aware_n_gt_0_sentinel` (cleanup; AC-1 + AC-4)
- iter-9b-10: N=4 SlotAware bench (AC-8)

## 7. References

- `src/inference/models/qwen3vl_text/forward.rs:212-880` — `forward_text_prefill_logits_last` (iter-8a-2 LANDED)
- `src/serve/api/engine_qwen3vl.rs:245-287` — `handle_qwen3vl_slot_aware_n_gt_0_sentinel` (iter-C2e-cont witness scaffold)
- `src/serve/api/engine.rs:1804` — `LoadedModel::Qwen3VlText` variant
- `docs/adr/ADR-040-continuous-batching-reopen.md` §6.1.52 (iter-C2e), §6.1.55 (iter-C2e-cont) — the two ADR-040 sections that explicitly carved this work out
- Peer reference: `qwen3vl.cpp:146-150` — DeepStack injection pattern
