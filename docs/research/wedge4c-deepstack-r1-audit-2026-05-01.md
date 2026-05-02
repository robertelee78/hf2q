# Wedge-4c DeepStack — Risk R1 audit (2026-05-01)

**Status**: settled by source-side inspection of `/opt/llama.cpp` (production-correct reference).
**Author**: main-thread audit during /loop iter (Workers AC + AD in flight on AC's B-FragReplay and AD's Wedge-4b).
**Trigger**: Worker W's plan §F Risk R1 — "DeepStack architecture is novel; concat-along-sequence may degrade quality vs. per-layer LM injection. Mitigation: implement concat first, measure parity, escalate to per-layer injection only if quality gap > 5%."

## TL;DR

**Worker W's plan was wrong on this risk.** llama.cpp's production Phase-1 implementation is **per-layer LM injection via concatenated-feature transport**. Implementing Worker W's "concatenate along sequence dim" idea would not converge to llama.cpp's logits within the 5% top-1 disagreement budget — Qwen3-VL was trained assuming per-layer injection.

**Wedge-4c should implement per-layer LM injection from day 1.** Adds ~150-200 LOC vs Worker W's plan; eliminates the Phase-2 rework risk for free; matches the trained model's expectations.

## ViT side — `tools/mtmd/models/qwen3vl.cpp` (clip path)

Per-layer DeepStack heads attached to specific ViT layers:

1. **Layer flag**: `clip.vision.is_deepstack_layers` GGUF metadata is a per-layer boolean array (`Sequence[bool]`, indexed `[0..n_layer)`). `convert_hf_to_gguf.py:4875-4877` populates from HF config's `deepstack_visual_indexes` (e.g., `[5, 11, 17]` → flags at indices 5/11/17).

2. **Per-layer head structure** (`clip-model.h:185-191`):
   ```
   v.deepstack.{N}.norm.{weight,bias}    // LayerNorm
   v.deepstack.{N}.fc1.{weight,bias}     // Linear (in: n_embd*merge_factor=n_embd*4)
   v.deepstack.{N}.fc2.{weight,bias}     // Linear (out: mm_1_b->ne[0] = LM hidden)
   ```
   Functionally a 2-layer MLP: `LayerNorm → Linear → GELU → Linear`.

3. **Per-head forward** (`qwen3vl.cpp:150-165`):
   ```cpp
   if (layer.has_deepstack()) {
       // reshape [n_embd, n_pos, B] → [n_embd*4, n_pos/4, B] (2×2 spatial-merge)
       feat = ggml_reshape_3d(cur, n_embd * merge_factor, n_pos / merge_factor, B);
       feat = build_norm(feat, layer.deepstack_norm_w, layer.deepstack_norm_b, NORM_NORMAL, eps, il);
       feat = build_ffn(feat, fc1, fc1_b, nullptr, nullptr, fc2, fc2_b, FFN_GELU, il);
       // feat now has shape [mm_1_b->ne[0], n_pos/4, B] = [LM_hidden, n_image_tokens, B]

       if (!deepstack_features) deepstack_features = feat;
       else deepstack_features = ggml_concat(ctx, deepstack_features, feat, 0);
       // concat along DIM 0 = FEATURE dim, growing channel count
   }
   ```

4. **Final composition** (`qwen3vl.cpp:185-187`):
   ```cpp
   // main projector: embeddings = build_ffn(reshape_3d(inpL, n_embd*4, n_pos/4, B), mm_0, mm_1, GELU)
   //   shape: [mm_1_b->ne[0], n_pos/4, B]
   if (deepstack_features) {
       embeddings = ggml_concat(ctx, embeddings, deepstack_features, 0);  // feature dim
   }
   // final shape: [mm_1_b->ne[0] * (1 + n_deepstack_layers), n_pos/4, B]
   ```

5. **Exposed embed-dim per image token** (`clip.cpp:3808-3809`):
   ```cpp
   return ctx->model.mm_1_b->ne[0] * (1 + ctx->model.n_deepstack_layers);
   ```
   For Qwen3-VL-2B: 2048 × (1 + 3) = **8192 dim per image token**, **n_image_tokens unchanged** = (img_h/16/2) × (img_w/16/2).

## LM side — `src/models/qwen3vl.cpp` (the receiver)

**This is the half Worker W's plan missed.** The LM treats the augmented embedding as `(1+N_deepstack) × hidden_size` channels per token and splits at injection time:

```cpp
// llm_build_qwen3vl::llm_build_qwen3vl, src/models/qwen3vl.cpp:96-100
if (il < (int) n_deepstack_layers) {
    ggml_tensor * ds = ggml_view_2d(
        ctx0, res->t_inp_embd,
        n_embd, n_tokens,
        res->t_inp_embd->nb[1],
        (il + 1) * n_embd * sizeof(float)   // offset = (il+1) chunk in feature dim
    );
    cur = ggml_add(ctx0, cur, ds);          // add to LM post-FFN residual
    cb(cur, "deepstack_out", il);
}
```

Reading: at LM layer `il < n_deepstack_layers`, slice the `(il+1)`-th `n_embd`-sized chunk out of the input embedding and **add** it to the post-FFN-residual `cur`.

**Concretely** for Qwen3-VL with 3 DeepStack layers (e.g., trained at ViT layers 5/11/17):
- LM layer 0: `cur += chunk[1]` (DeepStack output of ViT layer 5)
- LM layer 1: `cur += chunk[2]` (DeepStack output of ViT layer 11)
- LM layer 2: `cur += chunk[3]` (DeepStack output of ViT layer 17)
- LM layers 3..end: no DeepStack add

The "input token embedding" itself (chunk 0) goes in via the standard `build_inp_embd(model.tok_embd)` path at LM layer 0 input.

Note: the same pattern lives at `/opt/llama.cpp/src/models/qwen3vl-moe.cpp:103` for the MoE variant.

## Implications for hf2q Wedge-4c

### What Worker W's plan got right
- Per-layer DeepStack heads inside the ViT (LayerNorm + 2-layer MLP at specific ViT layers)
- Spatial merge (2×2 patch-pool reshape)
- Tap-points at ViT layers 5/11/17 for Qwen3-VL-2B (GGUF metadata `clip.vision.is_deepstack_layers`)

### What Worker W's plan got wrong (Risk R1 — RESOLVED)
- ❌ "Concatenate along sequence dim, yielding 3× the tokens" — production llama.cpp does NOT do this
- ❌ "Per-layer LM injection in Phase 2 if quality gap > 5%" — production llama.cpp does this in Phase 1
- ❌ The ViT output shape is NOT `[hidden, 3*n_image_tokens]`; it IS `[hidden*(1+N_deepstack), n_image_tokens]`

### What hf2q Wedge-4c should do
1. **ViT path**: implement per-layer DeepStack heads inside `vit_gpu_qwen3vl.rs`, output shape `[hidden*(1+N_deepstack), n_image_tokens, B]`.
2. **soft_tokens transport**: a single `SoftTokenData` per image with the augmented embedding (no per-layer fan-out at the engine boundary; the LM splits internally).
3. **Qwen35Model::forward_*_with_soft_tokens injection**:
   - At input embed step: place chunk 0 (`augmented[0..n_embd]`) at the image-token positions.
   - At post-FFN-residual of each LM layer `il < n_deepstack_layers`: add chunk `(il+1)` (`augmented[(il+1)*n_embd..(il+2)*n_embd]`) at the image-token positions.
4. **soft_tokens shape contract**: extend `SoftTokenData.embeddings: MlxBuffer` to carry the augmented `[hidden*(1+N_deepstack), n_image_tokens]` block. Either:
   - (A) Bake the split-and-add into `Qwen35Model::forward_gpu_*_with_soft_tokens` — keeps `SoftTokenData` API identical, splitter logic local to the model.
   - (B) Extend `SoftTokenData` with a `Vec<MlxBuffer>` of length `(1+N_deepstack)`, one per LM-injection point — explicit fan-out at the engine boundary; clearer audit trail but more pipe surface.
   Recommend (A) — minimal API surface change, matches llama.cpp's transport.

### Falsification criterion (Wedge-4c)
- `cargo test vit_gpu_qwen3vl_parity_real_mmproj`: top-50 elements of the augmented post-projector embedding bitwise tolerance ≤ 1e-3 vs llama.cpp golden.
- New test `qwen3vl_per_layer_deepstack_injection`: synthesize a 3-layer DeepStack mini-fixture; verify post-FFN-residual at LM layer `il` has `cur + chunk[il+1]` byte-equal to a manual-Python reference computation.

### LOC delta vs Worker W's plan
- Worker W: ~950 LOC (concat-along-seq variant)
- Per-layer-injection variant: ~1100-1150 LOC (+150-200 for the LM-side split-and-add hooks at `forward_gpu_qwen35.rs`)
- Trade: small LOC bump for production correctness from day 1 (no Phase-2 rework + risk-free closure of Risk R1)

## Sources

- `/opt/llama.cpp/tools/mtmd/models/qwen3vl.cpp:1-193` (full ViT graph build)
- `/opt/llama.cpp/tools/mtmd/clip-model.h:185-228` (DeepStack tensor names + has_deepstack())
- `/opt/llama.cpp/tools/mtmd/clip.cpp:1697-1705` (per-layer head loading + n_deepstack_layers count)
- `/opt/llama.cpp/tools/mtmd/clip.cpp:3808-3809` (exposed embed-dim formula)
- `/opt/llama.cpp/tools/mtmd/clip-impl.h:50` (`KEY_IS_DEEPSTACK_LAYERS = "clip.vision.is_deepstack_layers"`)
- `/opt/llama.cpp/tools/mtmd/clip-impl.h:117-119` (`TN_DEEPSTACK_NORM/FC1/FC2`)
- `/opt/llama.cpp/src/models/qwen3vl.cpp:1-122` (LM graph with per-layer injection at L96-100)
- `/opt/llama.cpp/src/models/qwen3vl-moe.cpp:103` (MoE variant, same pattern)
- `/opt/llama.cpp/src/llama-hparams.cpp:76-77` (`n_embd_inp += n_embd * n_deepstack_layers`)
- `/opt/llama.cpp/convert_hf_to_gguf.py:4875-4877, 4895-4896` (HF→GGUF conversion of `deepstack_visual_indexes`)
- `/opt/llama.cpp/gguf-py/gguf/constants.py:130, 320` (`{arch}.n_deepstack_layers` LM-side key + clip-side `IS_DEEPSTACK_LAYERS`)
