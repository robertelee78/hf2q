# Wedge-4 Risks R3/R5/R6 audit (2026-05-02)

**Status**: settled by source-side inspection while Wedge-4c.1 scaffold worker in flight.
**Trigger**: Worker W's plan §F lists 7 risks; R1 was audited 2026-05-01. This pass settles R3, R5, R6 — all directly impact Wedge-4c.{2..5} implementation specs.

## R3 — 2D-RoPE inside the ViT

**Worker W said**: "this is a new kernel call inside the ViT only; mlx-native's `rope_multi` already exists. ~150 LOC of orchestration."

**Audit finds R3 is partly wrong — there's a cross-repo dependency.**

`/opt/llama.cpp/ggml/include/ggml.h:1840-1846`:
```
// example vision RoPE:
//  given sections = [y=4, x=4, 0, 0] (last 2 sections are ignored)
//  given a single head with size = 8 --> [00000000]
//  GGML_ROPE_TYPE_VISION  n_dims = 4 --> [yyyyxxxx]
```

llama.cpp Qwen3-VL ViT call site (`/opt/llama.cpp/tools/mtmd/models/qwen3vl.cpp:111-116`):
```cpp
Qcur = ggml_rope_multi(
    ctx0, Qcur, positions, nullptr,
    d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
```
Mode value `GGML_ROPE_TYPE_VISION = 24` (per ggml.h:253), `mrope_sections = {d_head/4, d_head/4, d_head/4, d_head/4}` (per qwen3vl.cpp:14 — but the LAST 2 sections are ignored in vision mode).

**hf2q's `mlx-native` rope_multi does NOT support mode=24.**

`/opt/mlx-native/src/ops/rope_multi.rs:46-55`:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum RopeMultiMode {
    /// Standard multi-section RoPE; contiguous sections.
    Mrope = 8,
    /// Interleaved multi-section RoPE; `sector % 3` cycles through 3 axes.
    /// Used by Qwen3.5 / Qwen3.6.
    Imrope = 40,
}
```
Only `Mrope=8` and `Imrope=40`. No `Vision=24`.

The vision rope's `[yyyyxxxx]` per-section layout is structurally different from MROPE/IMROPE — per ggml.h:1845-1846 "the theta for each dim is computed differently for each section, in other words, idx used for theta: [0123] for y section, then [0123] for x section". MROPE/IMROPE use a unified theta sequence across sections.

**Implication for Wedge-4c.3 (ViT per-block forward)**:
Cross-repo dependency. Either:
- (A) **Add `RopeMultiMode::Vision = 24` to `/opt/mlx-native/src/ops/rope_multi.rs`** + matching shader logic — preferred per mantra ("no shortcuts, no fallback")
- (B) Build h/w positions inside hf2q and abuse IMROPE's 3-axis cycle by setting `t-axis = h, h-axis = w, w-axis = 0` — fragile workaround; would not match llama.cpp logits within the 5% top-1 budget

**Recommendation**: Wedge-4c.3 must be **paired with an mlx-native PR** adding `RopeMultiMode::Vision`. New mlx-native LOC estimate: ~80 (enum arm + shader branch + 1 unit test). hf2q-side caller: ~20 LOC. Total cross-repo addition: ~100 LOC + a unit test on each side. Sequence: mlx-native PR first (separate /cfa cycle in `/opt/mlx-native`), then hf2q-side wedge-4c.3 fires citing mlx-native commit hash.

**Falsifiable closure for the mlx-native PR**: synthetic input where `[yyyyxxxx]` dim layout produces specific output values from a hand-computed reference; bitwise tolerance ≤ 1 ULP at FP32.

## R5 — Chat-template mismatch

**Worker W said**: "qwen3-chatml.jinja already writes `<|vision_start|><|image_pad|><|vision_end|>` (line 18); a `<|image|>` literal anywhere in the rendered prompt would break expansion."

**Audit confirms R5 is correct.** No action needed for the template itself.

`/opt/hf2q/src/backends/chat_templates/qwen3-chatml.jinja:18`:
```jinja
{{- '<|vision_start|><|image_pad|><|vision_end|>' }}
```

The jinja correctly renders the Qwen-VL trio for any `image_url` content part. Wedge-4d's family branch lives in `handlers.rs::rewrite_messages_for_vision_placeholders` (`:2323-2341`) — that function rewrites pre-template content from OpenAI shape to family-specific marker. Currently it writes `<|image|>` (Gemma's marker). For Qwen-VL, that path must NOT rewrite (the jinja itself emits the trio; rewriting to `<|image|>` would clash).

**Implication for Wedge-4d**: Family branch logic at `handlers.rs:2323-2341` becomes:
- Gemma (current): rewrite to `<|image|>`
- Qwen-VL: passthrough (jinja handles the marker emission)

Tokenizer-side `expand_image_placeholders` lookup at `:2451` still needs the family branch:
- Gemma: lookup `<|image|>` token id
- Qwen-VL: lookup `<|image_pad|>` token id (151655 per `Qwen/Qwen3-VL-2B-Instruct/config.json`)

## R6 — `embed_tokens_gpu` CPU-gather + upload override pattern

**Worker W said**: "`embed_tokens_gpu` does CPU gather + upload (`forward_gpu.rs:408-427`); injecting an override row mid-batch requires either (a) splitting the upload into segments, or (b) post-upload patching via `dispatch_copy_f32` at known byte offsets. Recommend (b)."

**Audit finds Worker W's recommendation is wrong for Wedge-4c.5 — option (a) was already adopted in Wedge-4a, AND option (a) doesn't work for DeepStack residual-add anyway.**

Wedge-4a (LANDED `cbfffa3`) implemented `embed_tokens_gpu_with_soft_tokens` at `forward_gpu.rs:464-538` using **option (a) — CPU-side overwrite-then-upload**. The function does:
1. Validate ranges (no overlap, fits in seq, embedding size ≥ range × hidden × 4)
2. Standard CPU gather of token rows
3. **For each soft_token range**: `cpu[dst_off..dst_off+h].copy_from_slice(&src[src_off..src_off+h])` — direct CPU memcpy
4. `upload_f32(&cpu, device)` — single GPU upload of the patched buffer

That handles the **input-embed step (chunk 0)** of the augmented embedding for Qwen3-VL. Wedge-4c.5 inherits this for free.

**For Wedge-4c.5's NEW work** — DeepStack chunks 1..N added to **post-FFN-residual at LM layers 0..(N-1)**, not at input — the CPU-overwrite-then-upload pattern does NOT apply. By the time the LM is at layer `il`, the GPU residual is a live `MlxBuffer`, not a CPU vector pre-upload.

**Wedge-4c.5 needs a NEW GPU dispatch**: `image_token_residual_add(residual_buf, chunk_buf, image_token_positions, n_pos, hidden_size)` — Metal shader that adds `chunk_buf[p]` rows to `residual_buf[p]` at the listed image-token positions only, leaving text positions untouched.

LOC estimate for the new dispatch:
- Metal shader: ~30 LOC (1 simple add-with-position-mask kernel)
- Rust wrapper: ~80 LOC (validate + dispatch + kernel registration)
- 1 synthetic unit test: ~50 LOC

This adds ~160 LOC to Wedge-4c.5 vs Worker W's "150-200" estimate; lands within the same envelope.

**Falsifiable closure for Wedge-4c.5**:
- `qwen3vl_per_layer_deepstack_injection` synthetic test: build a 2-layer LM hidden state, inject a known augmented embedding `[hidden*(1+3), n_image_tokens]`, run `forward_gpu_with_soft_tokens`, assert post-FFN-residual at LM layers 0/1/2 byte-equals manual Python reference computation (chunk add at image positions only; text positions unchanged).

## Cross-cutting: Wedge-4 schedule is a chain, not parallel

R3 forces **mlx-native PR before Wedge-4c.3**. Cumulative:

```
Wedge-4c.1 (in flight)        scaffold + dispatch arm                ~200 LOC, ~1 day
Wedge-4c.2                    patch_embedding + position_embedding   ~250 LOC, ~1 day
mlx-native R3                 RopeMultiMode::Vision = 24             ~80 LOC,  ~0.5 day  ← cross-repo
Wedge-4c.3                    per-block ViT forward (uses Vision)    ~300 LOC, ~1-1.5 days
Wedge-4c.4                    deepstack heads + spatial merger       ~350 LOC, ~1.5 days
Wedge-4c.5                    LM-side split-and-add + image_token_residual_add  ~360 LOC, ~1.5 days
                              (~160 of which is the new GPU dispatch)
Wedge-4c TOTAL                                                       ~1540 LOC, ~6.5-7 days
```

Worker W's original Wedge-4c estimate was 4-5 man-days at ~950 LOC. The corrected scope (R1 audit per-layer LM injection, R3 cross-repo Vision rope, R6 new GPU dispatch for residual add) puts it closer to ~6.5-7 man-days at ~1540 LOC.

## Sources
- `/opt/llama.cpp/ggml/include/ggml.h:253, 1840-1862` (rope mode + vision-rope semantics)
- `/opt/llama.cpp/tools/mtmd/models/qwen3vl.cpp:14, 111-116` (vision-rope call site)
- `/opt/mlx-native/src/ops/rope_multi.rs:46-69` (current enum + params)
- `/opt/hf2q/src/backends/chat_templates/qwen3-chatml.jinja:18` (vision token emit)
- `/opt/hf2q/src/inference/models/qwen35/forward_gpu.rs:464-538` (Wedge-4a's overwrite-then-upload pattern)
- `Qwen/Qwen3-VL-2B-Instruct/config.json` (image_token_id=151655 — referenced; verified earlier in session)
