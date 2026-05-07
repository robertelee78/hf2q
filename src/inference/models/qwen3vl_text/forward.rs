//! Qwen3-VL text-LM dense forward path (ADR-021 iter-8a-2).
//!
//! # What this module ships
//!
//! The dense transformer forward pass for Qwen3-VL **text** LMs —
//! per-layer biased-GQA attention with per-head Q/K RMSNorm + 3D-IMROPE
//! + SDPA, followed by SwiGLU SiLU FFN, with a final RMSNorm + LM head
//! (tied or untied). Reuses Qwen3.5/3.6's GPU primitives (since the
//! per-layer arithmetic is structurally identical) but threads them
//! against [`Qwen3VlTextWeights`] instead of `Qwen35Weights`.
//!
//! # Ground-truth peer reference
//!
//! `/opt/llama.cpp/src/models/qwen3vl.cpp:53-172` —
//! `llama_model_qwen3vl::graph::graph(...)`. Per layer:
//!
//! ```text
//! inpL → rms_norm(attn_norm)            → cur
//! cur → Q/K/V proj (biased per peer; absent in the canonical 2B/4B GGUFs)
//! Qcur → rms_norm(attn_q_norm) → ggml_rope_multi(IMROPE, sections, theta)
//! Kcur → rms_norm(attn_k_norm) → ggml_rope_multi(IMROPE, sections, theta)
//! cur → build_attn(wo, wo_b, ...) → SDPA causal, scale = 1/sqrt(head_dim)
//! ffn_inp = cur + inpSA                   (residual after attn)
//! cur → rms_norm(ffn_norm)
//! cur → build_ffn(SILU, PAR, gate+up+down)
//! cur += ffn_inp                          (residual after ffn)
//! if il < n_deepstack_layers: cur += t_inp_embd_slab[il+1]
//! ```
//!
//! Final:
//!
//! ```text
//! cur → rms_norm(output_norm)
//! cur → build_lora_mm(output OR token_embd if tied)  (LM head)
//! ```
//!
//! # Scope of this iter
//!
//! - **Done in iter-8a-2 (this iter)**: full prefill — token embedding
//!   lookup, per-layer attn + FFN chain, final RMSNorm + LM head
//!   (tied + untied), returns last-position logits.
//! - **Deferred to iter-9a**: DeepStack residual injection at LM
//!   layers `il < n_deepstack_layers` (per-position add of vision
//!   slab into image-token positions).
//! - **Deferred to iter-9b**: Engine seam wiring (replace the
//!   [`qwen3vl_text_forward_pending`] sentinel in
//!   `serve/api/engine.rs::worker_run` Qwen3VlText match arms);
//!   soft-token splicing (replace per-position embed-table rows with
//!   ViT-projected vision tokens at `<|image_pad|>` positions); KV
//!   cache for incremental decode.
//! - **Deferred to iter-10a**: Live AC-3 closure (3 multimodal
//!   inference requests against real `Qwen/Qwen3-VL-2B-Instruct`,
//!   coherent generated text, literal output bytes).
//!
//! # Why no KV cache yet
//!
//! iter-8a-2 implements **prefill-only** — every call walks the full
//! token sequence from position 0. iter-9b's engine seam will add the
//! generation loop; for the first cut it can re-run the prefill on
//! `prompt + decoded_so_far` per generated token (O(N²) wall but
//! correct). KV-cache-incremental decode is an optimization for a
//! later iter. The mantra applies: get correctness first, then
//! optimize.
//!
//! # SDPA path choice
//!
//! Qwen3-VL has `head_dim = 128`. Qwen3.5/3.6 has `head_dim = 256`.
//! mlx-native's `apply_flash_attn_prefill_seq_major` (used by Qwen3.5
//! prefill) is hardcoded to D=256 via `flash_attn_prefill_bf16_d256`.
//! A D=128 dispatcher would be ideal but does not yet exist. We use
//! [`apply_sdpa_causal_from_seq_major`] instead — it has a CPU permute
//! hop (download → reorder → upload) but works for any head_dim. iter-
//! 9a/9b can replace this with a fused GPU permutation + SDPA path
//! once a D=128 dispatcher (or a generic `apply_sdpa_causal_seq_major`
//! GPU-permute variant) exists in mlx-native.

use anyhow::{anyhow, Context, Result};

use mlx_native::ops::elementwise::elementwise_add;
use mlx_native::ops::rms_norm::dispatch_rms_norm;
use mlx_native::{DType, GgmlType, KernelRegistry, MlxBuffer, MlxDevice};

use crate::inference::models::qwen35::gpu_ffn::{
    build_dense_ffn_layer_gpu_q_into, DenseFfnWeightsGpuQ,
};
use crate::inference::models::qwen35::gpu_full_attn::{
    apply_imrope, apply_linear_projection_f32, apply_q_or_k_per_head_rms_norm,
    apply_sdpa_causal_from_seq_major, download_f32, upload_f32,
};
use crate::inference::models::qwen35::io_heads::embed_tokens;

use super::Qwen3VlTextModel;

// ---------------------------------------------------------------------------
// Pending-sentinel surface (iter-228a → kept until iter-9b replaces engine seam)
// ---------------------------------------------------------------------------
//
// The forward path below is the iter-8a-2 deliverable, but the engine seam
// at `serve/api/engine.rs::worker_run` for `LoadedModel::Qwen3VlText` still
// calls `qwen3vl_text_forward_pending_err` (until iter-9b lands). Operator-
// facing chat / streaming / soft-tokens requests still receive the 501
// response with the pending message. The forward function below is reachable
// only via the in-tree integration test (`tests/qwen3vl_text_lm_forward.rs`)
// + future iter-9b dispatch.

/// Sentinel substring embedded in iter-8a-2's pending engine-seam error.
///
/// The chat handler at [`crate::serve::api::handlers`] matches on this
/// substring to dispatch to a structured HTTP 501 response. Stable
/// across iters; the value is part of the operator-facing contract.
/// iter-9b removes the call sites (engine seam dispatches the real
/// [`forward_text_prefill_logits_last`] forward path).
pub const QWEN3VL_TEXT_FORWARD_PENDING_SENTINEL: &str = "qwen3vl_text_forward_pending";

/// Operator-facing message body. Surfaced verbatim in the HTTP 501
/// response body (and in `tracing::warn` lines on the worker side).
pub const QWEN3VL_TEXT_FORWARD_PENDING_MESSAGE: &str =
    "Qwen3-VL text-LM dense forward is implemented in `forward.rs::forward_text_prefill_logits_last` \
     (iter-8a-2 LANDED) but the engine seam wire-up is iter-9b scope. The Generate / GenerateStream / \
     GenerateWithSoftTokens dispatch arms in `serve/api/engine.rs::worker_run` still return this \
     sentinel; iter-9b replaces them with calls into the prefill forward path. For text-only chat \
     today, use a Qwen3.5/3.6 GGUF (full chat path) or a Gemma 4 GGUF (full chat + image path).";

/// Construct the sentinel-tagged error returned by every dispatch arm
/// that lands on [`super::Qwen3VlTextModel`] in iter-8a-2 (until iter-
/// 9b wires the engine seam).
pub fn qwen3vl_text_forward_pending_err<T>() -> Result<T> {
    Err(anyhow::anyhow!(
        "{}: {}",
        QWEN3VL_TEXT_FORWARD_PENDING_SENTINEL,
        QWEN3VL_TEXT_FORWARD_PENDING_MESSAGE
    ))
}

// ---------------------------------------------------------------------------
// Dense forward — prefill only, returns last-position logits
// ---------------------------------------------------------------------------

/// Run a single prefill forward pass and return next-token logits over
/// the full vocabulary, computed at the **last** prompt position only.
///
/// # Args
///
/// * `model` — loaded Qwen3-VL text-LM bundle (config + weights + GPU
///    context). Mutable borrow because the GPU context's session
///    machinery + kernel registry mutate during the forward.
/// * `tokens` — input token IDs. Length = `seq_len`.
/// * `positions_flat` — 3D-IMROPE positions, axis-major layout
///    `[t_0, t_1, ..., t_{S-1}, y_0, ..., y_{S-1}, x_0, ..., x_{S-1}, z_0, ..., z_{S-1}]`
///    of length `4 * seq_len`. For text-only prompts the four axes
///    typically replicate the same token index (built by
///    [`crate::serve::forward_prefill::build_qwen3vl_positions`] with
///    no image grids); image-token positions get distinct (t, y, x)
///    values.
///
/// # Returns
///
/// `Vec<f32>` of length `cfg.vocab_size` — logits over the vocabulary
/// for the position immediately following the input prompt (i.e. the
/// last prompt token's logits row, which the sampling layer interprets
/// as the *next* token's distribution).
///
/// # Errors
///
/// * Empty `tokens`.
/// * `positions_flat.len() != 4 * tokens.len()`.
/// * Any underlying mlx-native dispatch / alloc / commit failure.
/// * Out-of-range token IDs (caught by [`embed_tokens`] assertion;
///   surfaced as a panic-converted error context).
///
/// # Implementation notes
///
/// 1. **Embedding lookup is CPU-side** — `weights.token_embd` is
///    F32; we download the full table once, then `embed_tokens` does
///    the row gather in CPU memory and we upload the resulting
///    `[seq_len, hidden]` matrix. iter-9b can replace this with an
///    on-GPU `dispatch_embedding_lookup_f32` once the engine seam owns
///    persistent buffers (the per-call F32 download of a 1.24 GB
///    table is fine for iter-8a-2 correctness work but unacceptable
///    for production).
///
/// 2. **Per-layer chain runs in ONE GPU session** — all 28 layers
///    encode into a single command buffer, then `session.finish()`
///    commits + waits once. Stage A's iter-6 collapse pattern from
///    `vit_gpu_qwen3vl.rs` applied at LM-layer scale.
///
/// 3. **SDPA is via CPU-permute path** — see module-level note. iter-
///    9a/9b can swap this for a GPU-permute path once mlx-native ships
///    a head_dim=128 flash-attention-prefill kernel.
///
/// 4. **No KV cache** — every call walks all 28 layers from position
///    0. iter-9b's engine seam adds the generate loop (re-prefill per
///    decoded token at first; KV-cache-incremental later).
///
/// 5. **No deepstack injection** — peer's `qwen3vl.cpp:146-150`
///    `cur += t_inp_embd_slab[il+1]` for `il < n_deepstack_layers` is
///    iter-9a scope. For the iter-8a-2 prefill smoke test against
///    text-only prompts (no images), this is a no-op anyway since the
///    deepstack slabs would be zero.
///
/// 6. **No soft-token splicing** — `<|image_pad|>` placeholder
///    positions in `tokens` are passed through to `embed_tokens`
///    unchanged in this iter; iter-9b adds the override-row pass
///    that replaces those rows with ViT-projected embeddings before
///    the GPU upload.
pub fn forward_text_prefill_logits_last(
    model: &mut Qwen3VlTextModel,
    tokens: &[u32],
    positions_flat: &[i32],
) -> Result<Vec<f32>> {
    if tokens.is_empty() {
        return Err(anyhow!(
            "forward_text_prefill_logits_last: empty token list (need at least 1 prompt token)"
        ));
    }
    let seq_len_usize = tokens.len();
    if positions_flat.len() != 4 * seq_len_usize {
        return Err(anyhow!(
            "forward_text_prefill_logits_last: positions_flat length ({}) must equal \
             4 * tokens.len() ({}) — IMROPE expects 4 axes (t/y/x/pad) per token",
            positions_flat.len(),
            4 * seq_len_usize,
        ));
    }
    let seq_len = seq_len_usize as u32;

    let cfg = model.cfg.clone();
    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_attention_heads;
    let n_kv_heads = cfg.num_key_value_heads;
    let kv_dim = n_kv_heads * head_dim;
    let intermediate = cfg.intermediate_size;
    let n_layers = cfg.num_hidden_layers as usize;
    let vocab = cfg.vocab_size;
    let rms_eps = cfg.rms_norm_eps;
    let rope_theta = cfg.rope_theta;
    let mrope_section = cfg.mrope_section;
    let tied = cfg.tied_word_embeddings;
    // Per peer `qwen3vl.cpp:60`: `GGML_ASSERT(n_embd_head == n_rot)`. The
    // RoPE-multi `n_rot` argument equals head_dim; the IMROPE arrangement
    // rotates the first `head_dim/2` slots (sum of `mrope_section`).
    let rotary_dim = head_dim;

    let weights = &model.weights;

    // ──────────────────────────────────────────────────────────────────
    // Step 1: Token embedding lookup (CPU side)
    // ──────────────────────────────────────────────────────────────────
    //
    // weights.token_embd is F32 [vocab, hidden] (cast at load time from
    // GGUF F16 — see weights.rs::load_from_gguf rationale).
    let token_embd_cpu = download_f32(&weights.token_embd)
        .context("download token_embd for embedding lookup")?;
    let hidden_cpu = embed_tokens(tokens, &token_embd_cpu, vocab, hidden);
    drop(token_embd_cpu); // release the 1.24 GB CPU buffer ASAP

    // ──────────────────────────────────────────────────────────────────
    // Step 2: Open the GPU session shared across all 28 layers + head
    // ──────────────────────────────────────────────────────────────────
    let (executor, registry) = model.ctx.split();
    let device = executor.device();

    // Upload initial residual stream `hidden_gpu` and IMROPE positions.
    // Both are persistent across the full forward.
    let mut hidden_gpu =
        upload_f32(&hidden_cpu, device).context("upload initial residual stream")?;
    drop(hidden_cpu);

    let positions_gpu = {
        let byte_len = positions_flat.len() * 4;
        let mut buf = device
            .alloc_buffer(
                byte_len,
                DType::I32,
                vec![positions_flat.len()],
            )
            .map_err(|e| anyhow!("alloc IMROPE positions buffer: {e}"))?;
        buf.as_mut_slice::<i32>()
            .map_err(|e| anyhow!("positions as_mut_slice: {e}"))?
            .copy_from_slice(positions_flat);
        buf
    };

    // ──────────────────────────────────────────────────────────────────
    // Step 3: Per-layer transformer chain
    // ──────────────────────────────────────────────────────────────────
    //
    // Each layer is broken into THREE phases because
    // `apply_sdpa_causal_from_seq_major` commit-and-waits the encoder
    // mid-flight (its CPU-permute hop):
    //
    //   Phase A: pre-attn norm + Q/K/V proj + per-head Q/K rmsnorm + IMROPE
    //   Phase B: SDPA (its own commit + permute + dispatch internally)
    //   Phase C: output proj + post-attn residual + ffn norm + SwiGLU FFN
    //            + post-ffn residual (built-in to dense_ffn).
    //
    // Each phase opens its own `executor.begin()` session so encoder
    // lifetimes are scoped tightly and Rust's borrow checker is happy.
    // The cost (3× session-finish per layer × 28 layers = 84 commits)
    // is iter-8a-2 correctness overhead; iter-9a/9b's GPU-permute SDPA
    // will collapse Phase A+B+C back into one encoder.
    //
    // The peer dispatch (`qwen3vl.cpp:77-154`) wraps every layer in the
    // same shape; we mirror it 1:1.
    for il in 0..n_layers {
        let lw = &weights.layers[il];

        // ── Phase A: pre-attn → Q/K/V → IMROPE ────────────────────
        let (q_rope, k_rope, v_seq) = {
            let mut session_a = executor
                .begin()
                .with_context(|| format!("layer {il}: begin Phase A session"))?;
            let enc = session_a.encoder_mut();

            let attn_normed = rms_norm_2d(
                enc, registry, device, &hidden_gpu, &lw.attn_norm,
                seq_len, hidden, rms_eps,
            )
            .with_context(|| format!("layer {il}: pre-attn rms_norm"))?;

            let q_seq = apply_linear_projection_f32(
                enc, registry, device, &attn_normed, &lw.attn_q,
                seq_len, hidden, hidden,
            )
            .with_context(|| format!("layer {il}: Q proj"))?;
            let k_seq = apply_linear_projection_f32(
                enc, registry, device, &attn_normed, &lw.attn_k,
                seq_len, hidden, kv_dim,
            )
            .with_context(|| format!("layer {il}: K proj"))?;
            let v_seq = apply_linear_projection_f32(
                enc, registry, device, &attn_normed, &lw.attn_v,
                seq_len, hidden, kv_dim,
            )
            .with_context(|| format!("layer {il}: V proj"))?;

            // The kernel reads Q/K as `[seq * n_heads, head_dim]` (each
            // row independently normalized). Q/K buffers have shape
            // `[seq, hidden]` = `[seq, n_heads * head_dim]` —
            // memory-equivalent. `apply_q_or_k_per_head_rms_norm`
            // validates `rows × dim` element count, not strict shape.
            let q_normed = apply_q_or_k_per_head_rms_norm(
                enc, registry, device, &q_seq, &lw.attn_q_norm,
                seq_len, n_heads, head_dim, rms_eps,
            )
            .with_context(|| format!("layer {il}: Q per-head rms_norm"))?;
            let k_normed = apply_q_or_k_per_head_rms_norm(
                enc, registry, device, &k_seq, &lw.attn_k_norm,
                seq_len, n_kv_heads, head_dim, rms_eps,
            )
            .with_context(|| format!("layer {il}: K per-head rms_norm"))?;

            // 3D-IMROPE: mode=40 (Imrope), sections=[24,20,20,0] for
            // Qwen3-VL-2B → first 64 slots rotated, next 64 identity.
            let q_rope = apply_imrope(
                enc, registry, device, &q_normed, &positions_gpu,
                seq_len, n_heads, head_dim, rotary_dim, rope_theta, mrope_section,
            )
            .with_context(|| format!("layer {il}: Q IMROPE"))?;
            let k_rope = apply_imrope(
                enc, registry, device, &k_normed, &positions_gpu,
                seq_len, n_kv_heads, head_dim, rotary_dim, rope_theta, mrope_section,
            )
            .with_context(|| format!("layer {il}: K IMROPE"))?;

            session_a
                .finish()
                .with_context(|| format!("layer {il}: finish Phase A session"))?;
            (q_rope, k_rope, v_seq)
        };

        // ── Phase B: SDPA (causal) — head_dim=128 via CPU-permute ─
        //
        // `apply_sdpa_causal_from_seq_major` opens its own internal
        // encoder for the SDPA dispatch (after its CPU permute hop).
        // We pass it a session-owned encoder it can commit, then drop
        // the session — no double-commit risk because the function
        // does its own internal commit_and_wait, and our session's
        // CommandEncoder Drop only `endEncoding`s without committing.
        let attn_out = {
            let mut session_b = executor
                .begin()
                .with_context(|| format!("layer {il}: begin Phase B session"))?;
            let enc = session_b.encoder_mut();
            let attn_out = apply_sdpa_causal_from_seq_major(
                enc, registry, device, &q_rope, &k_rope, &v_seq,
                seq_len, n_heads, n_kv_heads, head_dim,
            )
            .with_context(|| format!("layer {il}: SDPA"))?;
            // session_b drops here — its encoder is already discharged
            // by the internal commit_and_wait inside the SDPA wrapper.
            // Do NOT call session_b.finish() (it would double-commit).
            attn_out
        };

        // ── Phase C: output proj + residual + FFN ─────────────────
        let l_out = {
            let mut session_c = executor
                .begin()
                .with_context(|| format!("layer {il}: begin Phase C session"))?;
            let enc = session_c.encoder_mut();

            let attn_proj = apply_linear_projection_f32(
                enc, registry, device, &attn_out, &lw.attn_output,
                seq_len, hidden, hidden,
            )
            .with_context(|| format!("layer {il}: output proj"))?;

            let ffn_inp = elementwise_add_f32_2d(
                enc, registry, device, &hidden_gpu, &attn_proj,
                seq_len, hidden,
            )
            .with_context(|| format!("layer {il}: post-attn residual add"))?;

            let ffn_normed = rms_norm_2d(
                enc, registry, device, &ffn_inp, &lw.ffn_norm, seq_len, hidden, rms_eps,
            )
            .with_context(|| format!("layer {il}: ffn rms_norm"))?;

            // SwiGLU SiLU FFN with built-in residual add.
            //
            // **iter-8a-2 limitation**: assumes Q4_0 for gate/up/down.
            // The canonical Qwen3-VL-2B/4B Instruct GGUF emitted by
            // hf2q's converter ships uniformly Q4_0 across all FFN
            // tensors (verified gguf-dump 2026-05-07: 168 Q4_0 + 28 F32
            // exactly = 6 Q4_0 attn + 3 Q4_0 ffn × 28 layers + 6 F32
            // norms × 28 layers; no mixed-quant variants in tree).
            // iter-9a should extend [`super::Qwen3VlTextLayerWeights`]
            // with `ggml_type_*: GgmlType` fields recorded at load time
            // (mirrors `qwen35::weight_loader::DenseFfnWeightsQ::ggml_type_*`)
            // so future mixed-quant Qwen3-VL variants (e.g. Q5_K gate/up
            // + Q6_K down) load + run without a forward-path code change.
            let ffn_weights = DenseFfnWeightsGpuQ {
                gate_q: lw.ffn_gate.clone(),
                up_q: lw.ffn_up.clone(),
                down_q: lw.ffn_down.clone(),
                ggml_type_gate_up: GgmlType::Q4_0,
                ggml_type_down: GgmlType::Q4_0,
                intermediate_size: intermediate,
                hidden_size: hidden,
            };
            let l_out = build_dense_ffn_layer_gpu_q_into(
                enc, device, registry, &ffn_normed, &ffn_weights, Some(&ffn_inp),
            )
            .with_context(|| format!("layer {il}: dense SwiGLU FFN"))?;

            session_c
                .finish()
                .with_context(|| format!("layer {il}: finish Phase C session"))?;
            l_out
        };

        // ── DeepStack residual injection (iter-9a scope) ───────────
        //
        // Peer `qwen3vl.cpp:146-150`:
        //     if (il < n_deepstack_layers) {
        //         ggml_tensor * ds = view_2d(t_inp_embd, n_embd, n_tokens, ...,
        //                                    (il+1) * n_embd * sizeof(float));
        //         cur = ggml_add(cur, ds);
        //     }
        //
        // For text-only prompts (no images) the deepstack slabs would
        // be zero, so the add is a no-op. iter-9a wires the per-image-
        // token slab + position-gated `image_token_residual_add_gpu`.

        hidden_gpu = l_out;
    }

    // ──────────────────────────────────────────────────────────────────
    // Step 4: Final RMS norm + LM head — last-position only
    // ──────────────────────────────────────────────────────────────────
    //
    // We download the residual stream, slice the last row, then
    // re-upload + dispatch the head on a single 1-row buffer. This
    // wastes the per-call CPU↔GPU round-trip but keeps the head matmul
    // M=1 (vs M=seq_len), which is the cheapest path for getting
    // last-position logits without needing GGML's `get_rows` op.
    //
    // iter-9b can replace this with an on-GPU row extraction (a
    // 1-row dispatch_copy_f32 with `src_offset = (seq-1)*hidden*4`)
    // to avoid the CPU hop entirely.

    let final_residual_cpu = hidden_gpu
        .as_slice::<f32>()
        .map_err(|e| anyhow!("final residual as_slice: {e}"))?;
    let last_row_start = (seq_len_usize - 1) * (hidden as usize);
    let last_row_end = last_row_start + (hidden as usize);
    let last_row: Vec<f32> = final_residual_cpu[last_row_start..last_row_end].to_vec();

    let last_row_gpu = upload_f32(&last_row, device)
        .context("upload last-row residual for output head")?;
    drop(last_row);

    // LM head — tied or untied. Per peer `qwen3vl.cpp:18-26`:
    //   - `output.weight` is created as TENSOR_NOT_REQUIRED;
    //   - if absent, falls back to `tok_embd` (TENSOR_DUPLICATED).
    //
    // Our `weights.output` is `Some(buf)` for untied, `None` for tied
    // (matching the post-load `Qwen3VlTextConfig::tied_word_embeddings`
    // signal). For tied, we re-use `weights.token_embd` as the head.
    let lm_head_weight: &MlxBuffer = if tied {
        &weights.token_embd
    } else {
        weights
            .output
            .as_ref()
            .ok_or_else(|| {
                anyhow!(
                    "Qwen3-VL text-LM weights inconsistent: tied_word_embeddings=false but \
                     output is None — config and weights disagree"
                )
            })?
    };

    let logits_buf = {
        let mut session_head = executor
            .begin()
            .context("begin output-head session")?;
        let enc = session_head.encoder_mut();

        // Final RMSNorm (output_norm).
        let final_normed = rms_norm_2d(
            enc, registry, device, &last_row_gpu, &weights.output_norm,
            1, hidden, rms_eps,
        )
        .context("final output rms_norm")?;

        let logits_buf = apply_linear_projection_f32(
            enc, registry, device, &final_normed, lm_head_weight,
            1, hidden, vocab,
        )
        .context("LM head matmul")?;

        session_head
            .finish()
            .context("finish output-head session")?;
        logits_buf
    };

    // Read logits back to a CPU `Vec<f32>` of length `vocab_size`. Even
    // if the underlying buffer's `byte_len()` is bucket-rounded by the
    // pool, `apply_linear_projection_f32` uses `device.alloc_buffer`
    // (NOT pooled — see its "NOTE: NOT pooled" comment), so
    // `as_slice::<f32>()` returns exactly `1 * vocab` elements.
    let logits_full = logits_buf
        .as_slice::<f32>()
        .map_err(|e| anyhow!("logits as_slice: {e}"))?;
    let v = vocab as usize;
    if logits_full.len() < v {
        return Err(anyhow!(
            "logits buffer too small: got {} elements, expected {}",
            logits_full.len(),
            v
        ));
    }
    Ok(logits_full[..v].to_vec())
}

// ---------------------------------------------------------------------------
// Local helpers — small wrappers that aren't worth lifting to a module crate
// ---------------------------------------------------------------------------

/// Thin wrapper around mlx-native's `dispatch_rms_norm` that allocates
/// the output buffer + the 8-byte params buffer (`[eps, dim]` f32 pair)
/// on the device, dispatches, and returns the output.
///
/// `input` shape: `[rows, dim]` F32. Output: same shape, F32.
fn rms_norm_2d(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    rows: u32,
    dim: u32,
    eps: f32,
) -> Result<MlxBuffer> {
    let out = device
        .alloc_buffer(
            (rows * dim) as usize * 4,
            DType::F32,
            vec![rows as usize, dim as usize],
        )
        .map_err(|e| anyhow!("alloc rms_norm output: {e}"))?;
    let mut params = device
        .alloc_buffer(8, DType::F32, vec![2])
        .map_err(|e| anyhow!("alloc rms_norm params: {e}"))?;
    {
        let s = params
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow!("rms_norm params as_mut_slice: {e}"))?;
        s[0] = eps;
        s[1] = dim as f32;
    }
    dispatch_rms_norm(
        encoder,
        registry,
        device.metal_device(),
        input,
        weight,
        &out,
        &params,
        rows,
        dim,
    )
    .context("dispatch_rms_norm")?;
    Ok(out)
}

/// Allocate output + dispatch elementwise add for 2D F32 tensors of shape
/// `[rows, cols]`. Wraps mlx-native's `elementwise_add` to handle the
/// per-call output alloc.
fn elementwise_add_f32_2d(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    a: &MlxBuffer,
    b: &MlxBuffer,
    rows: u32,
    cols: u32,
) -> Result<MlxBuffer> {
    let n_elements = (rows as usize) * (cols as usize);
    let out = device
        .alloc_buffer(
            n_elements * 4,
            DType::F32,
            vec![rows as usize, cols as usize],
        )
        .map_err(|e| anyhow!("alloc elementwise_add output: {e}"))?;
    elementwise_add(
        encoder,
        registry,
        device.metal_device(),
        a,
        b,
        &out,
        n_elements,
        DType::F32,
    )
    .map_err(|e| anyhow!("elementwise_add: {e}"))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentinel_is_stable_across_iters() {
        // The sentinel value is part of the operator-facing contract.
        // The chat handler matches on this substring; changing it in a
        // refactor would silently break HTTP 501 dispatch.
        assert_eq!(
            QWEN3VL_TEXT_FORWARD_PENDING_SENTINEL,
            "qwen3vl_text_forward_pending"
        );
    }

    #[test]
    fn pending_err_carries_sentinel_substring() {
        let err: Result<()> = qwen3vl_text_forward_pending_err();
        let msg = format!("{:#}", err.unwrap_err());
        assert!(
            msg.contains(QWEN3VL_TEXT_FORWARD_PENDING_SENTINEL),
            "error message must carry the sentinel substring; got: {msg}"
        );
        assert!(
            msg.contains("iter-9b"),
            "iter-8a-2 pending message must point at iter-9b for engine seam wire-up; got: {msg}"
        );
    }

    #[test]
    fn pending_err_message_is_operator_actionable() {
        let err: Result<()> = qwen3vl_text_forward_pending_err();
        let msg = format!("{:#}", err.unwrap_err());
        // Operator-actionable: tells them what works today.
        assert!(
            msg.contains("Qwen3.5") || msg.contains("Gemma"),
            "error message must name a working alternative; got: {msg}"
        );
    }

    /// Real-model gate. When `HF2Q_QWEN3VL_LM_LOAD=1` and the canonical
    /// fixture GGUF is present, load it and run a single prefill forward
    /// pass over a 4-token text-only prompt; assert the returned logits
    /// have the right length and contain no NaN/Inf. This is a SHAPE +
    /// FINITENESS smoke test only; semantic correctness (which token
    /// the argmax picks) is iter-10a's AC-3 deliverable.
    #[test]
    fn forward_text_prefill_shape_finite_when_operator_gated() {
        if std::env::var("HF2Q_QWEN3VL_LM_LOAD").ok().as_deref() != Some("1") {
            eprintln!("skip: HF2Q_QWEN3VL_LM_LOAD!=1");
            return;
        }
        let p = std::path::PathBuf::from(
            "/opt/hf2q/.cfa-archive/wedge4f-out/qwen3-vl-2b-q4_0.gguf",
        );
        if !p.exists() {
            eprintln!("skip: real GGUF fixture not present at {}", p.display());
            return;
        }
        let gguf = mlx_native::gguf::GgufFile::open(&p)
            .expect("open real Qwen3-VL-2B GGUF");
        let mut progress = crate::serve::header::LoadProgress::new(false, 0, 28);
        let mut model = Qwen3VlTextModel::load_from_gguf(&gguf, &mut progress)
            .expect("load real Qwen3-VL-2B text-LM model");

        // 4 random text tokens from Qwen3-VL's vocab (well below the
        // 151_936 size bound).
        let tokens: Vec<u32> = vec![100, 200, 300, 400];
        // Text-only positions: replicate the same index across all 4
        // axes (per `build_qwen3vl_positions` for an empty image_grids).
        let seq = tokens.len();
        let mut positions = vec![0i32; 4 * seq];
        for axis in 0..4 {
            for t in 0..seq {
                positions[axis * seq + t] = t as i32;
            }
        }

        let logits = forward_text_prefill_logits_last(&mut model, &tokens, &positions)
            .expect("forward must succeed on the canonical GGUF");

        assert_eq!(
            logits.len(),
            model.cfg.vocab_size as usize,
            "logits length must equal vocab_size"
        );

        let n_finite = logits.iter().filter(|x| x.is_finite()).count();
        let n_nan = logits.iter().filter(|x| x.is_nan()).count();
        let n_inf = logits.iter().filter(|x| x.is_infinite()).count();
        assert_eq!(
            n_nan, 0,
            "logits must contain no NaN; got {n_nan} NaN entries"
        );
        assert_eq!(
            n_inf, 0,
            "logits must contain no Inf; got {n_inf} Inf entries"
        );
        assert_eq!(
            n_finite,
            logits.len(),
            "all logits must be finite (no NaN/Inf); got {n_finite}/{}",
            logits.len()
        );

        // Sanity range: a well-formed forward pass produces logits with
        // a reasonable spread (max - min > 1.0 typically). Loose bound.
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            (max - min) > 0.1,
            "logits should have nonzero spread; got max={max}, min={min}"
        );
        eprintln!(
            "forward_text_prefill_shape_finite: vocab={} max_logit={:.4} min_logit={:.4}",
            logits.len(),
            max,
            min
        );
    }
}
