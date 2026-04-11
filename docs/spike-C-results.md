# ADR-005 Phase 1b — Spike Report: C (per-layer hidden-state bisect)

**Date:** 2026-04-11
**Runner:** Claude (investigation-only; no `main` commits)
**Scope:** Localize the owner of the residual `The`/`To` Walk-correctness argmax
drift (+0.77 toward `The`, opposite to llama.cpp's `To`) by dumping per-layer
hidden states on both hf2q and a patched llama.cpp, comparing element-wise on
the canonical 187-token bench prompt, and drilling into the first-divergence
layer to identify the owning op.
**Baseline binary:** `main` HEAD `67efbf1` (post-1bNEW.17 spike ADR update;
58.51 tok/s median).
**Model:** `models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf`
(Gemma 4 26B MoE DWQ, mixed Q4_0/Q6_K/F16).
**Hardware:** Apple M5 Max, 128 GB unified memory.
**Worktree discipline:** all hf2q scratch instrumentation reverted before
returning. `git diff --stat src/` empty at return; `git rev-parse HEAD ==
67efbf1` unchanged. llama.cpp left in a dirty scratch state (eval-callback
dump patch in `src/llama-context.cpp`) — documented in section "llama.cpp
patch" below so it can be re-created or reverted.

---

## TL;DR

**First-divergence layer: layer 5** (the first full-attention / global layer).
**Outcome class: A (concentrated).** Layers 0-4 (sliding attention, head_dim=256)
are numerically f32-faithful against llama.cpp at `max |Δ|` ≤ 5e-3 at the
last-token position. Layer 5 jumps by ~300x to `max |Δ|` ≈ 0.81 at the
last-token position, and the step-change is entirely introduced inside
`DecoderLayer::forward` for layer 5.

**Owning op: Q/K RoPE for global (full-attention) layers.** Drill into layer 5's
sub-tensors isolates the divergence to `Qcur_pos` and `Kcur_pos` — the tensors
after `rotary_emb.apply(&q, &k, seqlen_offset)` at `/opt/hf2q/src/serve/gemma4.rs:693`.
All pre-RoPE sub-tensors (`attn_norm`, `Qcur_normed`, `Kcur_normed`, `Vcur_normed`)
match llama.cpp at max |Δ| ≤ 0.08 (compounded from the layer-4 input carry-over).
`Qcur_pos` max |Δ| jumps to **30.08** and `Kcur_pos` to **1.50**, with the
difference growing linearly with token position (token 0: 2e-4, token 100:
17.08) — the characteristic signature of a frequency-base mismatch in RoPE.

**Two combined causes**, both Walk-citable:

1. **Missing `rope_freqs.weight` tensor binding.** Gemma 4's GGUF ships a
   `rope_freqs.weight` F32 tensor of shape `[256]` with the pattern
   `[1.0] × 64 + [1e+30] × 192`. llama.cpp uses it as `freq_factors` in
   `ggml_rope_ext` at `/opt/llama.cpp/src/models/gemma4-iswa.cpp:73,97-98` on
   the **full-attention layers only** (`gemma4-iswa.cpp:56-59`), dividing each
   per-pair theta by that factor in the `kernel_rope_neox` kernel at
   `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:4353-4355`. The 1e+30
   values effectively zero the rotation (`theta/1e30 → 0` → `cos=1, sin=0` →
   identity) for pair indices `[64..256)`, leaving only pairs `[0..64)`
   rotated. **hf2q's serve-side code never reads `rope_freqs.weight`.** The
   fused RoPE kernel at `/opt/hf2q/src/serve/rope_kernel.rs:261,319` has the
   `freq_factor` branch correctly implemented (byte-for-byte port from
   llama.cpp) but the caller hardcodes `src2: 0` at
   `/opt/hf2q/src/serve/rope_kernel.rs:721`, permanently disabling the
   `args.src2 != 0` gate. Also `src/serve/` contains **zero references** to
   `rope_freqs` (verified by grep across all files).
2. **Wrong pair offset for partial rotary on global layers.** llama.cpp's
   `kernel_rope_neox` reads each rotation pair at `(src[ic], src[ic +
   n_dims/2])` where `n_dims = n_rot(il)` is 512 for global layers (from GGUF
   metadata `gemma4.rope.dimension_count = 512`). So pair `ic ∈ [0..256)` uses
   elements at offsets `ic` and `ic + 256`. hf2q's loop-mode RoPE at
   `/opt/hf2q/src/serve/gemma4.rs:508-515` operates on a narrow slice
   `q[..rotary_dim]` and pairs elements at offset `rotary_dim/2`. With hf2q's
   config default `partial_rotary_factor_global = 0.25` (at
   `/opt/hf2q/src/serve/config.rs:88`), `rotary_dim = 128`, so hf2q pairs
   elements at offset `64`, i.e., `(q[i], q[64+i])` for `i ∈ [0..64)`. **llama.cpp
   and hf2q rotate completely different component pairs of the same head
   vector.**

**Walk-correctness closable: YES (WALK-CITABLE).** Both causes are
reference-citable one-site ports against the existing llama.cpp Gemma 4
implementation — the kernel code on hf2q's side already supports
`freq_factors`, only the load-time weight binding and `rope_fused` call-site
are missing. The partial-rotary pairing fix is to replace the "narrow(0,
rotary_dim) + rope_apply + cat" dance with "apply rope to all 512 dims,
pairing at offset 256". In the fused kernel this is already a one-arg change
(pass `n_dims = head_dim` instead of `n_dims = rotary_dim`). The combined
scratch test demonstrated in section "Proof of fix" below (`sin[64:128]=0,
cos[64:128]=1` mask on the global pre-computed table) partially closes the
gap from +0.771 to +0.435 (−0.336 toward llama.cpp) in a single toggle in
loop mode; a structurally complete fix that also corrects the pair offset
is projected to close the bulk of the remaining ~0.435 drift.

**End-gate impact:** Walk-correctness End gate (ADR line ~711:
"hf2q's top-1 token at decode 1 matches llama.cpp's top-1 token") is **closable**
via a single reference-citable item. Speed impact is zero to minimal — the
fused kernel already handles `src2` on the GPU side; wiring `rope_freqs` through
adds one Metal buffer binding per global-layer RoPE dispatch (5 layers × 2 RoPE
calls = 10 additional bindings per forward pass, no new dispatches).

---

## Methodology

### Instrumentation

#### hf2q side (scratch, reverted at return)

Added an env-var-gated per-layer hidden-state dump inside
`Gemma4Model::forward` after each `DecoderLayer::forward` call, mirroring
the existing `HF2Q_DUMP_LOGITS` pattern at `/opt/hf2q/src/serve/mod.rs:167-180`:

```rust
// Inside the layer loop at gemma4.rs:1922-1924
for (_i, layer) in self.layers.iter_mut().enumerate() {
    xs = layer.forward(&xs, seqlen_offset)?;
    if let Some(ref dir) = std::env::var("HF2Q_DUMP_LAYER_STATES").ok() {
        // dump xs as F32 LE bytes, shape [1, seq_len, hidden_size]
    }
}
// ...after self.norm.forward(&last_hidden): dump layer_30_final.bin
```

Zero runtime cost when unset. For the layer-5 drill-down, a second env-var
gate `HF2Q_DUMP_DRILL=<dir>` was added to a scratch variant
`DecoderLayer::forward_with_idx` that dumps 15 intra-layer tensors with
names matching the ggml `cb()` callback names in
`/opt/llama.cpp/src/models/gemma4-iswa.cpp` (`attn_norm`, `Qcur_normed`,
`Kcur_normed`, `Vcur_normed`, `Qcur_pos`, `Kcur_pos`, `attn_post_norm`,
`attn_out`, `ffn_norm_1`, `ffn_mlp`, `ffn_norm_2`, `ffn_moe`,
`ffn_moe_combined`, `ffn_post_norm`, `out_scaled`).

All scratch edits reverted before return. Verified clean:
```
$ git diff --stat src/
$ git rev-parse HEAD
67efbf153074419d950946b18640d99ca6d65f04
```

#### llama.cpp side (scratch, left as dirty worktree)

Patched `/opt/llama.cpp/src/llama-context.cpp` in two places:

1. In `process_ubatch`, when `LLAMA_DUMP_LAYER_STATES` is set **and**
   `ubatch.n_tokens >= 10` (to skip the 2-token graph-reservation warmup call
   that otherwise overwrites the real prefill dump), install a per-tensor
   `ggml_backend_sched_eval_callback` that fires during graph execution for
   every tensor whose name matches `l_out-*`, `result_norm`, or any of the 15
   intra-layer drill tap names. The callback copies the tensor to CPU via
   `ggml_backend_tensor_get` and writes it to disk as F32 LE bytes with the
   filename `drill_<name>-<il>.bin` (or `layer_<il>.bin` / `layer_30_final.bin`
   for the l_out and result_norm cases).

2. An earlier attempt set `ggml_set_output(cur)` inside the build-time `cb()`
   callback and read back tensor data in a post-compute loop, but this
   produced `inf` values at layer 4 — the backend scheduler was re-using the
   layer-4 output tensor's memory after that tensor was consumed by layer 5's
   input, so the data was garbage by the time the post-compute loop ran. The
   eval-callback approach avoids this because it fires **during** compute, at
   the exact moment the tensor's data is valid on CPU-side. This code path is
   preserved but gated behind `&& false` in the patch.

Build: `cmake --build /opt/llama.cpp/build --target llama-completion -j8`. Both
builds succeeded cleanly (hf2q `cargo build --release --features metal`,
llama.cpp metal build).

**llama.cpp patch summary** (for re-creation or revert):
- `/opt/llama.cpp/src/llama-context.cpp:12-20` — add `<cstdio>`, `<string>`,
  `<sys/stat.h>`, `<vector>` includes.
- `/opt/llama.cpp/src/llama-context.cpp:1197-1265` — in `process_ubatch`,
  replace the unconditional
  `ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);`
  with a conditional that installs the SPIKE_C dump callback when
  `LLAMA_DUMP_LAYER_STATES` is set and `ubatch.n_tokens >= 10`.
- `/opt/llama.cpp/src/llama-context.cpp:2197-2250` (`graph_get_cb`) — SPIKE_C
  comment block; no flag-setting (removed in favor of the eval-callback
  approach).
- `/opt/llama.cpp/src/llama-context.cpp:1230-1310` — post-compute dump loop,
  disabled via `&& false`.

### Prompt alignment (CRITICAL — pre-existing `crawl_verify.sh` hazard)

Before any layer-state comparison, discovered that `crawl_verify.sh`
post-1bNEW.0c produces **187 tokens on hf2q but 188 tokens on llama.cpp** for
the same rendered prompt file. Root cause: the hf2q-rendered prompt starts
with the literal text `<bos>`, which hf2q's tokenizer parses as BOS token
(id 2) one time. llama.cpp's `common_tokenize(ctx, prompt, /*add_special=*/
true, /*parse_special=*/true)` at `/opt/llama.cpp/tools/completion/completion.cpp:322`
**also** adds an implicit BOS when the GGUF metadata has
`tokenizer.ggml.add_bos_token = true` (verified via `gguf.GGUFReader` on the
Gemma 4 GGUF). Net result: llama.cpp's token sequence is `[2, 2, 105, 2364,
...]` (two BOS) vs hf2q's `[2, 105, 2364, ...]` (one BOS), shifting every
position by one and making all per-position hidden states structurally
non-comparable.

**Fix (spike-local):** strip the leading `<bos>` text from the rendered
prompt file before passing it to llama-completion:
```bash
python3 -c "data = open('/tmp/rendered.txt','rb').read(); \
  assert data.startswith(b'<bos>'); \
  open('/tmp/rendered_noleadingbos.txt','wb').write(data[5:])"
```

Verified both tools tokenize to 187 tokens after the fix
(`llama-completion --verbose-prompt` → `main: number of tokens in prompt = 187`,
`HF2Q_DUMP_PROMPT_TOKENS=1` → `total=187`).

**Side finding with cross-cutting implications:** When both tools see the
same 187-token sequence, **llama.cpp's top-1 token at decode 1 becomes `The`,
not `To`**. Reproduced 3x:

- `llama-completion --file /tmp/rendered.txt` (188 tokens, double BOS) → `To`
- `llama-completion --file /tmp/rendered_noleadingbos.txt` (187 tokens) → `The`
- `hf2q generate --prompt-file tests/bench_prompt_128.txt` (187 tokens) → `The`

The historical Walk-correctness disagreement ("hf2q picks `The`, llama.cpp
picks `To`") recorded in `crawl_verify.sh`'s crawl verification output and in
ADR line 194-196 is at least in part an artifact of the 188-vs-187 token
count mismatch. **The ~0.89 logit gap between the two tools is partly a
non-comparable-input artifact.** This does NOT dismiss the residual drift —
the per-layer bisect below shows layer 5 still diverges by ~0.8 logit at the
last-token position even on byte-identical 187-token input — but it does
move the end-gate framing: the "hf2q top-1 == llama.cpp top-1" goal is
**already met on matched inputs**, and the remaining Walk-correctness work
is about byte-level prefix agreement across the generated continuation,
which depends on the per-layer drift converging to floating-point noise.

### Canonical bench and comparison

- hf2q run: `HF2Q_DUMP_LAYER_STATES=/tmp/spikeC/hf2q HF2Q_DUMP_DRILL=/tmp/spikeC/hf2q
  target/release/hf2q generate --model <gguf> --prompt-file tests/bench_prompt_128.txt
  --max-tokens 1 --temperature 0`.
- llama.cpp run: `LLAMA_DUMP_LAYER_STATES=/tmp/spikeC/llama
  llama-completion --model <gguf> --file /tmp/rendered_noleadingbos.txt
  --predict 1 --temp 0 --seed 42 --no-display-prompt -no-cnv -st -ngl 999 </dev/null`.

Both produce `l_out-N.bin` (N ∈ [0..29]) + `layer_30_final.bin` (post-`result_norm`)
on the 187-token prompt. Layer-state tensors are F32 flat arrays:
- Layers 0..28: `[seq_len=187, hidden_size=2816]` = 2,106,368 bytes.
- Layer 29 on llama.cpp: `[1, 2816]` = 11,264 bytes (reduced via `inp_out_ids`
  optimization in `gemma4-iswa.cpp:113-115` — at the final layer only the
  last token is kept for the lm_head).
- `layer_30_final.bin` on both: `[1, 2816]` = 11,264 bytes.

Comparison script at `/tmp/compare_layers.py` (scratch, not committed):
reshapes both sides to `(187, 2816)`, computes per-layer `max |Δ| all` (over
the full sequence) and `max |Δ| last` (restricted to the last-token position,
which is what produces the lm_head logits), plus `mean |Δ|` and `rms`.

---

## Part 1 — Per-layer divergence table (all 30 layers + final)

Numbers from the canonical 187-token bench prompt at HEAD `67efbf1`, hf2q
defaults (`--moe-kernel fused --rms-norm-kernel fused --rope-kernel fused
--lm-head-kernel fused`), vs llama.cpp patched with the eval-callback dump on
the BOS-stripped 187-token prompt:

| layer | shape | max \|Δ\| all | max \|Δ\| last | mean \|Δ\| last | rms last | verdict |
|-------|-------|--------------|----------------|-----------------|----------|---------|
| 0     | (187, 2816) | 1.307e-02 | **2.070e-03** | 5.03e-05 | 1.39e-04 | f32 floor |
| 1     | (187, 2816) | 6.983e-01 | **2.532e-03** | 4.89e-05 | 1.35e-04 | f32 floor |
| 2     | (187, 2816) | 3.054e-01 | **2.182e-03** | 6.13e-05 | 1.68e-04 | f32 floor |
| 3     | (187, 2816) | 2.668e-01 | **3.540e-03** | 8.23e-05 | 2.25e-04 | f32 floor |
| 4     | (187, 2816) | 2.076e-01 | **4.118e-03** | 8.61e-05 | 2.22e-04 | f32 floor |
| **5** | (187, 2816) | **7.101e+00** | **8.078e-01** | 3.18e-02 | 7.53e-02 | **JUMP** |
| 6     | (187, 2816) | 9.748e+00 | 7.303e-01 | 3.01e-02 | 7.01e-02 | compounding |
| 7     | (187, 2816) | 2.303e+01 | 5.358e-01 | 2.75e-02 | 6.19e-02 | compounding |
| 8     | (187, 2816) | 4.849e+01 | 7.297e-01 | 3.04e-02 | 6.50e-02 | compounding |
| 9     | (187, 2816) | 7.177e+01 | 1.145e+00 | 4.06e-02 | 8.33e-02 | compounding |
| 10    | (187, 2816) | 8.503e+01 | 8.490e-01 | 4.37e-02 | 8.38e-02 | compounding |
| 11    | (187, 2816) | 8.462e+01 | 1.709e+00 | 1.01e-01 | 1.77e-01 | 2nd global jump |
| 12    | (187, 2816) | 9.587e+01 | 1.424e+00 | 9.22e-02 | 1.55e-01 | compounding |
| 13    | (187, 2816) | 8.382e+01 | 1.242e+00 | 1.09e-01 | 1.72e-01 | compounding |
| 14    | (187, 2816) | 5.516e+01 | 1.134e+00 | 9.57e-02 | 1.46e-01 | compounding |
| 15    | (187, 2816) | 3.435e+01 | 1.447e+00 | 9.55e-02 | 1.44e-01 | compounding |
| 16    | (187, 2816) | 2.098e+01 | 1.064e+00 | 1.00e-01 | 1.52e-01 | compounding |
| 17    | (187, 2816) | 5.120e+01 | 2.909e+00 | 1.55e-01 | 2.33e-01 | 3rd global jump |
| 18    | (187, 2816) | 5.424e+01 | 3.036e+00 | 1.48e-01 | 2.19e-01 | compounding |
| 19    | (187, 2816) | 4.435e+01 | 1.252e+00 | 1.43e-01 | 2.02e-01 | compounding |
| 20    | (187, 2816) | 3.483e+01 | 1.249e+00 | 1.30e-01 | 1.83e-01 | compounding |
| 21    | (187, 2816) | 2.875e+01 | 1.030e+00 | 1.25e-01 | 1.72e-01 | compounding |
| 22    | (187, 2816) | 2.822e+01 | 1.072e+00 | 1.15e-01 | 1.58e-01 | compounding |
| 23    | (187, 2816) | 2.540e+01 | 8.706e-01 | 1.56e-01 | 2.11e-01 | 4th global jump |
| 24    | (187, 2816) | 2.174e+01 | 1.130e+00 | 1.54e-01 | 2.07e-01 | compounding |
| 25    | (187, 2816) | 1.760e+01 | 9.145e-01 | 1.51e-01 | 2.00e-01 | compounding |
| 26    | (187, 2816) | 1.440e+01 | 1.007e+00 | 1.50e-01 | 1.98e-01 | compounding |
| 27    | (187, 2816) | 1.144e+01 | 7.424e-01 | 1.37e-01 | 1.84e-01 | compounding |
| 28    | (187, 2816) | 5.653e+00 | 7.425e-01 | 1.24e-01 | 1.63e-01 | compounding |
| 29    | (187,→1, 2816) | — | 1.770e+00 | 3.65e-02 | 6.19e-02 | 5th global jump (1-token slice) |
| 30F   | [1, 2816] | — | **3.194e+00** | 3.13e-01 | 4.09e-01 | post-final-norm |

Observations:

1. **Layers 0-4 are f32-faithful.** Max |Δ| at the last-token position is
   ≤ 4.1e-3; mean |Δ| is ~5-8e-5; rms is ~1-2e-4. These are the residual floor
   of candle's vs ggml's floating-point reduction order differences and
   are **below even the most aggressive "bit-identity" test bar used in any
   Walk item** (1bNEW.1/4/6 Phase A all tested at 1e-5 ε and this is above
   that, but still far below any "systematic drift" threshold).
2. **Layer 5 is the first jump: 4.1e-3 → 0.808 at last-token.** A ~300×
   step-change in a single layer. Layer 5 is **the first full-attention /
   global layer** (per `config.is_full_attention(5) == true`, the five
   full-attention layers are {5, 11, 17, 23, 29}).
3. **Visible secondary jumps at layers 11, 17, 23** — the other full-attention
   layers. Layer 11's `max |Δ| last` is 1.71 (almost 2× the layer-5 jump),
   layer 17's is 2.91, layer 23's is 0.87 (slightly reduced from layer 17
   — the drift is not monotonic because the MoE branch redistributes it).
4. **Layers 6-10, 12-16, 18-22, 24-28 (sliding)** do not introduce new
   divergence of their own; they just compound the drift already in the
   residual stream. Max |Δ| all typically shrinks slightly across these
   (the residual-stream smoothing effect).
5. **Layer 30 final** (post-`result_norm`, 1-token) has max |Δ| 3.19 — this
   is what projects through the lm_head into the ~0.89 logit difference
   observed in Spike B.

**Outcome class: A (concentrated) with periodic re-injection.** The drift is
introduced only on full-attention layers {5, 11, 17, 23, 29} and smears
across sliding layers in between via the residual stream. The non-monotonic
profile is consistent with a per-layer systematic bias that gets partially
averaged away by sliding-layer attention.

---

## Part 2 — Layer 5 drill: which op inside the DecoderLayer introduces the drift?

For layer 5, dumped 15 intra-layer tap tensors on both sides at the named
ggml callback points (see `gemma4-iswa.cpp:53,71,75,94,95,100,120,123,132,143,149,176,179,197,229`
on the llama side; `input_layernorm`, `q_norm`, `k_norm`, `rms_norm_unit`,
`rotary_emb.apply`, `post_attention_layernorm`, `(xs + attn_out)`, and the MoE
pipeline on the hf2q side). Compared by flat sorted L∞ norm (layouts
differ — hf2q is `[1, n_heads, seq, head_dim]` post-transpose, llama is
`[head_dim, n_heads, seq]` col-major which is the same row-major `[seq,
n_heads, head_dim]` — both preserve total element count so sort-comparison
is meaningful):

| tap | hf2q size | llama size | max \|Δ\| | mean \|Δ\| | rms | hf2q rms | llama rms |
|-----|-----------|------------|----------|------------|-----|---------|----------|
| `attn_norm` (pre-attn input)       | 526,592 | 526,592 | 4.90e-02 | 2.87e-04 | 1.13e-03 | 0.800 | 0.800 |
| `Qcur_normed` (post-q_norm)        | 1,531,904 | 1,531,904 | 4.70e-02 | 3.69e-04 | 1.26e-03 | 1.008 | 1.008 |
| `Kcur_normed` (post-k_norm)        | 191,488 | 191,488 | **5.17e-03** | 3.74e-05 | 1.33e-04 | 0.0623 | 0.0623 |
| `Vcur_normed` (rms_norm_unit(v))   | 191,488 | 191,488 | 8.31e-02 | 6.01e-04 | 2.13e-03 | 1.000 | 1.000 |
| **`Qcur_pos`** (post-RoPE)         | 1,531,904 | 1,531,904 | **3.008e+01** | 8.79e-01 | 1.38e+00 | 1.008 | 1.008 |
| **`Kcur_pos`** (post-RoPE)         | 191,488 | 191,488 | **1.502e+00** | 6.38e-02 | 8.41e-02 | 0.0623 | 0.0623 |
| `attn_post_norm` (post_attn_norm ∘ attn)  | 526,592 | 526,592 | 3.06e+00 | 4.60e-02 | 1.21e-01 | 0.536 | 0.519 |
| `attn_out` (xs + attn_out residual)| 526,592 | 526,592 | 3.06e+00 | 4.60e-02 | 1.21e-01 | 2.118 | 2.119 |
| `ffn_norm_1` (pre-MLP norm)        | 526,592 | 526,592 | 3.48e+00 | 2.23e-02 | 6.44e-02 | 0.368 | 0.362 |
| `ffn_mlp` (MLP branch output)      | 526,592 | 526,592 | 1.93e+02 | 5.72e-01 | 1.84e+00 | 19.22 | 19.43 |
| `ffn_norm_2` (pre-MoE norm)        | 526,592 | 526,592 | 6.69e-01 | 2.22e-02 | 3.57e-02 | 0.278 | 0.278 |
| `ffn_moe` (MoE branch output)      | 526,592 | 526,592 | 8.85e+01 | 5.05e-01 | 1.21e+00 | 10.74 | 10.71 |
| `ffn_moe_combined` (mlp + moe)     | 526,592 | 526,592 | 2.26e+02 | 8.27e-01 | 2.30e+00 | 26.12 | 26.25 |
| `ffn_post_norm` (post-FFW norm)    | 526,592 | 526,592 | 8.31e+01 | 6.07e-01 | 2.11e+00 | 3.299 | 1.599 |

**Decisive:** pre-RoPE tensors match at `max |Δ|` ≤ 0.083 (carry-over noise
from the layer-4 input and the Q/K/V projection + norm chain). Post-RoPE
tensors `Qcur_pos` and `Kcur_pos` jump to **30.08** and **1.50** respectively.
**The RoPE application on layer 5 is the first site where hf2q and llama.cpp
disagree by more than the floating-point reduction-order envelope.**

Position-dependent analysis of `Qcur_pos` (reshape hf2q to `[1, 16 heads, 187
tokens, 512 head_dim]`, llama to `[187 tokens, 16 heads, 512 head_dim]`,
compare head 0 across token positions):

| token | head | max \|Δ\| | rms |
|-------|------|----------|-----|
| 0     | 0    | 2.01e-04 | 5.99e-05 |
| 0     | 5    | 3.16e-04 | 7.70e-05 |
| 0     | 15   | 4.87e-04 | 1.07e-04 |
| 50    | 0    | 5.81e+00 | 1.06e+00 |
| 50    | 5    | 6.01e+00 | 1.10e+00 |
| 50    | 15   | 4.97e+00 | 6.05e-01 |
| 100   | 0    | 1.71e+01 | 1.12e+00 |
| 100   | 5    | 1.68e+01 | 1.01e+00 |
| 100   | 15   | 1.27e+01 | 7.83e-01 |
| 186   | 0    | 1.64e+01 | 1.01e+00 |
| 186   | 5    | 1.34e+01 | 9.12e-01 |
| 186   | 15   | 3.25e+00 | 3.27e-01 |

Token 0 has `max |Δ|` ≈ 2e-4 — floating-point floor (since `cos(0) = 1, sin(0) =
0` regardless of frequency, so the two tools trivially agree on the zero-position
rotation). Token 50 jumps to 5.8, token 100 to 17.0, token 186 to 16.4. **This
is the unambiguous signature of a frequency-mismatched RoPE**: divergence
scales with position because each pair's rotation angle `theta_ic = pos *
(freq_base)^(-...)` accumulates the frequency error over position.

**Verdict:** `Qcur_pos-5` and `Kcur_pos-5` disagree at position-proportional
magnitude; the underlying op is the **RoPE application on the first
full-attention layer**.

---

## Part 3 — Root cause analysis

### 3.1 llama.cpp's Gemma 4 RoPE path

Traced end-to-end:

1. **`gemma4-iswa.cpp:49`** — `const int n_rot_l = hparams.n_rot(il)`. This
   reads per-layer rotary dimension count from hparams.

2. **`llama-hparams.cpp:65-71`** — `n_rot(il)` returns `is_swa(il) ? n_rot_swa :
   n_rot_full`. Populated from GGUF metadata. Verified via
   `gguf.GGUFReader`:
   ```
   gemma4.rope.dimension_count = 512    (full / global)
   gemma4.rope.dimension_count_swa = 256 (sliding)
   gemma4.rope.freq_base = 1e+06
   gemma4.rope.freq_base_swa = 10000
   gemma4.attention.head_count = 16
   gemma4.attention.head_count_kv = 8
   ```
   **For global layers, `n_rot_l = 512 = head_dim`** — full rotary, no
   pass-through tail in the sense hf2q's code assumes.

3. **`gemma4-iswa.cpp:55-59`** — `freq_factors` is set to
   `model.layers[il].rope_freqs` only on non-SWA (full-attention) layers:
   ```cpp
   ggml_tensor * freq_factors = nullptr;
   if (!hparams.is_swa(il)) {
       // full_attention layers use rope_freqs for proportional rope
       freq_factors = model.layers[il].rope_freqs;
   }
   ```

4. **`gemma4-iswa.cpp:73-75, 97-98`** — `ggml_rope_ext(ctx0, Q, inp_pos,
   freq_factors, n_rot_l, rope_type, ...)` on the global path passes
   `n_rot_l = 512` and `freq_factors = rope_freqs`.

5. **`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:4376-4426`** —
   `kernel_rope_neox<T>` with `args.n_dims = 512`:
   ```
   for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
       if (i0 < args.n_dims) {
           const int ic = i0/2;
           const float theta = theta_base * pow(args.freq_base, inv_ndims*i0);
           const float freq_factor = (args.src2 != 0) ? ((device const float *) src2)[ic] : 1.0f;
           rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, i0,
                     args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
           ...
           const float x0 = src[0];
           const float x1 = src[args.n_dims/2];   // <-- PAIR OFFSET = n_dims/2 = 256
           dst_data[0]             = x0*cos_theta - x1*sin_theta;
           dst_data[args.n_dims/2] = x0*sin_theta + x1*cos_theta;
       } else {
           // pass-through (i0 >= n_dims)
       }
   }
   ```
   For global layer: `inv_ndims = -1/512`, so `theta = theta_base *
   (10^6)^(-i0/512)`. For pair `ic` (where `i0 = 2*ic`), `theta_ic =
   theta_base * (10^6)^(-ic/256)` for `ic ∈ [0..256)`. **Pair offset is 256**
   (`n_dims/2`).

6. **GGUF `rope_freqs.weight`:** F32, shape `[256]`. Verified via Python:
   ```
   Count near 1.0:  64   (indices [0..64))
   Count at 1e+30: 192   (indices [64..256))
   ```
   So `freq_factor = 1.0` for `ic ∈ [0..64)` and `1e+30` for `ic ∈
   [64..256)`. The `theta / 1e+30` division produces `~0`, giving `cos_theta
   = 1, sin_theta = 0` — **identity rotation**, writes the input values
   unchanged. Effective behavior: pairs `[0..64)` rotate normally with
   `theta = pos * (10^6)^(-ic/256)`; pairs `[64..256)` are identity-rotated
   (unchanged).

**llama.cpp's effective global-layer RoPE:**
- For `ic ∈ [0..64)`: rotate pair `(q[ic], q[ic+256])` by angle
  `pos * (10^6)^(-ic/256)`.
- For `ic ∈ [64..256)`: leave pair `(q[ic], q[ic+256])` unchanged.

### 3.2 hf2q's Gemma 4 RoPE path

Traced end-to-end:

1. **`/opt/hf2q/src/serve/config.rs:88`** — `partial_rotary_factor_global:
   full_rope.partial_rotary_factor.unwrap_or(0.25)`. **Default value is
   `0.25`.** The GGUF does not set this key, so the default applies.

2. **`/opt/hf2q/src/serve/gemma4.rs:1668-1672`** — `rope_global =
   RotaryEmbedding::new_partial(cfg.global_head_dim=512, ...,
   cfg.partial_rotary_factor_global=0.25, ...)`.

3. **`/opt/hf2q/src/serve/gemma4.rs:375-387`** — `new_partial`:
   ```rust
   let rope_angles = ((partial_rotary_factor * head_dim as f64 / 2.0).floor()
                       as usize).min(head_dim / 2);
   let rotary_dim = rope_angles * 2;
   Self::build(head_dim, rotary_dim, ...)
   ```
   For `partial_rotary_factor = 0.25, head_dim = 512`:
   `rope_angles = (0.25 * 512 / 2).floor() = 64`, `rotary_dim = 128`.

4. **`/opt/hf2q/src/serve/gemma4.rs:389-428`** — `build`:
   ```rust
   let half = rotary_dim / 2;   // 64
   let inv_freq: Vec<f32> = (0..half)
       .map(|i| 1f32 / rope_theta.powf(2.0 * i as f64 / head_dim as f64))
       .collect();
   // inv_freq[i] = (10^6)^(-2i/512) = (10^6)^(-i/256), i ∈ [0..64)
   let freqs = t.matmul(&inv_freq)?;  // [max_seq_len, 64]
   let sin = freqs.sin()?;             // [max_seq_len, 64]
   let cos = freqs.cos()?;             // [max_seq_len, 64]
   ```
   **`self.sin/self.cos` only contain 64 frequency pairs.**

5. **`/opt/hf2q/src/serve/gemma4.rs:484-504`** — `apply` (loop mode):
   ```rust
   let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;  // [seq, 64]
   let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;  // [seq, 64]

   if self.rotary_dim == head_dim { ... }
   else {   // rotary_dim=128 < head_dim=512
       let pass_len = head_dim - self.rotary_dim;           // 384
       let q_rot_part = q.narrow(D::Minus1, 0, self.rotary_dim)?;  // [B,H,S,128]
       let q_pass = q.narrow(D::Minus1, self.rotary_dim, pass_len)?; // [B,H,S,384]
       let q_rot = Self::rope_apply(&q_rot_part, &cos, &sin)?;
       Tensor::cat(&[q_rot, q_pass], D::Minus1)?   // back to [B,H,S,512]
   }
   ```

6. **`/opt/hf2q/src/serve/gemma4.rs:508-515`** — `rope_apply`:
   ```rust
   fn rope_apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
       let half = x.dim(D::Minus1)? / 2;   // 64 (of the 128-dim slice)
       let x1 = x.narrow(D::Minus1, 0, half)?;       // x[0..64]
       let x2 = x.narrow(D::Minus1, half, half)?;    // x[64..128]
       let r1 = (x1 * cos - x2 * sin)?;
       let r2 = (x1 * sin + x2 * cos)?;
       Tensor::cat(&[r1, r2], D::Minus1)
   }
   ```
   **Pair offset is 64** (`rotary_dim/2` of the 128-dim slice).

**hf2q's effective global-layer RoPE:**
- For `i ∈ [0..64)`: rotate pair `(q[i], q[i+64])` by angle
  `pos * (10^6)^(-i/256)`.
- For `i ∈ [64..512)`: pass-through (copied unchanged).

### 3.3 Side-by-side comparison

| what | llama.cpp | hf2q | same? |
|------|-----------|------|-------|
| angular frequency for pair `i` | `pos * (10^6)^(-i/256)` | `pos * (10^6)^(-i/256)` | **YES** |
| which indices get rotated | ic ∈ [0..64), via `rope_freqs` mask | i ∈ [0..64), via `rotary_dim=128` slice | superficially similar |
| pair member offset | `n_dims/2 = 256` → pair `(q[i], q[i+256])` | `rotary_dim/2 = 64` → pair `(q[i], q[i+64])` | **NO — critical** |
| what happens to `q[64..256]` | rotated as pair `(q[ic], q[ic+256])` with `freq_factor=1e+30` → identity rotation → unchanged | passed through (as part of `q_pass`) → unchanged | same result |
| what happens to `q[256..320]` | **rotated** — paired with `q[0..64]` via pair-offset=256 | **passed through** — part of `q_pass` | **NO** |
| what happens to `q[320..512]` | identity rotation, unchanged | pass-through, unchanged | same result |

**hf2q rotates pairs `(q[0..64], q[64..128])`. llama.cpp rotates pairs
`(q[0..64], q[256..320])`.** These are completely different component pairs
of the same vector. The rotation math is applied correctly in both tools,
but to **different data**. The position-proportional drift pattern observed
in `Qcur_pos-5` (0 at token 0, linearly growing to ~17 at token 100+) is
exactly what this produces: the rotation is applied with correct angles but
to the wrong operand pairs, so the net perturbation is a pure rotation in
the wrong subspace, which (since rotations preserve norm) produces a
mean-zero, position-scaled divergence. The `hf2q rms = llama rms = 1.008`
row for `Qcur_pos` is the smoking gun for this: both tensors have the same
L2 norm, they just point in different directions in the head-dim subspace.

### 3.4 Secondary finding: fused kernel is also affected

The fused Metal RoPE kernel in
`/opt/hf2q/src/serve/rope_kernel.rs:283-339` is a byte-for-byte port of
llama.cpp's `kernel_rope_neox`, including the `freq_factor = (args.src2 !=
0) ? src2[ic] : 1.0f` branch at line 319 and the correct pair offset `x1 =
src[args.n_dims/2]` at line 327. **The kernel itself is correct**. The
problem is two-fold on the caller side:

1. **`rope_kernel.rs:721`** — `src2: 0` is hardcoded, permanently disabling
   the `args.src2 != 0` gate regardless of whether the caller has a
   `rope_freqs` buffer to pass.
2. **`gemma4.rs:453`** — the caller passes `self.rotary_dim` as `n_dims` to
   `rope_fused()`. For global layers with `rotary_dim = 128`, the kernel
   computes `theta = theta_base * pow(freq_base, inv_ndims*i0)` with
   `inv_ndims = -1/128`, and reads pair members at offset `n_dims/2 = 64`.
   **Both the pair offset AND the frequency base exponent denominator are
   wrong on the fused path as well.**

So both loop and fused RoPE paths produce equivalent-wrong output, which is
why Spike B's kernel-toggle sweep showed `--rope-kernel loop` moving the top-2
gap by −0.01430 (a BF16 interaction artifact, not a structural difference) —
the two paths implement the same (wrong) math.

### 3.5 Why the Phase A unit tests pass

The fused RoPE Phase A tests at `rope_kernel.rs:980-1110` all pass because
they compare the fused kernel output against the `reference_rope_apply`
function at `rope_kernel.rs:834` — **which is hf2q's own buggy reference
implementation**. Both paths apply the same wrong pair offset and the same
wrong frequency-base denominator, so they agree byte-for-byte at ε=1e-5 on
synthetic inputs. The tests validate internal consistency of hf2q's RoPE
port against itself, **not** fidelity against llama.cpp. The test at
`rope_kernel.rs:992-1001`
(`rope_neox_decode_partial_rotary_global`) specifically exercises the
`rotary_dim = 128, head_dim = 512` case and passes — confirming the two
paths agree with each other, not with ggml. This is a gap in hf2q's test
discipline that Spike C should flag for a follow-up reference parity test
(see "Action items" section).

---

## Part 4 — Proof-of-fix test

To confirm the diagnosis, applied a scratch patch masking the global RoPE's
pre-computed `sin/cos` tables to match llama.cpp's effective identity
rotation for pairs `[64..128)` (i.e., set `sin[:, 64:128] = 0, cos[:,
64:128] = 1`). This is a **partial** fix — it zeroes out rotation for pairs
[64..128) but does NOT correct the underlying pair offset (hf2q still pairs
`(q[i], q[i+64])` instead of `(q[i], q[i+256])`).

Scratch gate: `HF2Q_SPIKE_C_ROPE_MASK=1`, reverted at return.

Run with `HF2Q_SPIKE_C_ROPE_MASK=1 --rope-kernel loop`:

| config | `(The, 818)` | `(To, 2021)` | gap | Δ vs HEAD |
|--------|-------------|-------------|-----|-----------|
| HEAD defaults                   | 27.108929 | 26.337908 | +0.771021 | 0 |
| `HF2Q_SPIKE_C_ROPE_MASK=1` + `--rope-kernel loop` | 27.108929 | 26.673597 | **+0.435332** | **−0.336** |

**The partial-mask scratch fix closes 0.336 of the 0.89 logit total
disagreement in a single toggle.** That is **38% of the total gap**, from a
fix that only addresses one of the two root-cause components (the "mask
pairs [64..128) to identity" sub-fix, not the "pair offset should be 256
not 64" sub-fix).

Per-layer bisect under the partial fix (HF2Q_SPIKE_C_ROPE_MASK=1) vs llama.cpp:

| layer | HEAD max \|Δ\| last | MASK max \|Δ\| last | HEAD max \|Δ\| all | MASK max \|Δ\| all |
|-------|---------------------|---------------------|--------------------|--------------------|
| 5     | 0.808               | **0.769**           | 7.10               | **7.04** |
| 6     | 0.730               | 0.488               | 9.75               | 7.80 |
| 7     | 0.536               | 0.742               | 23.03              | 9.47 |
| 8     | 0.730               | 0.734               | 48.49              | 13.99 |
| 12    | 1.424               | 1.282               | 95.87              | 20.64 |
| 28    | 0.742               | 0.437               | 5.65               | 5.18 |
| 30F   | 3.194               | 2.926               | —                  | — |

Max |Δ| all shrinks by ~4-5x on downstream layers (layer 8: 48 → 14, layer
12: 96 → 21). Layer-5's max |Δ| last shrinks only marginally (0.808 →
0.769) because the partial mask does not fix the wrong pair offset — it
only removes one of the two erroneous rotations. A full fix (correct pair
offset + correct `freq_factors`) is projected to drop layer-5's last-token
max |Δ| to the ~1e-3 f32 floor and collapse the entire downstream chain to
the noise level seen on layers 0-4, closing the remaining ~0.43 logit gap to
well below the Walk-correctness End gate threshold.

The partial-mask test is sufficient to **verify the diagnosis** (RoPE is
the owner, on global layers only, and the fix direction is correct). The
full structural fix is deferred to a landing item.

---

## Part 5 — Proposed fix

### Fix #1: Load `rope_freqs.weight` from the GGUF and bind it to the fused RoPE `src2` input

**Sites to change:**

- **`/opt/hf2q/src/serve/gemma4.rs` — `Gemma4Model::load`**: after
  constructing `rope_global`, load `rope_freqs.weight` from the GGUF via
  the existing `GgufModel::get_tensor` helper. The tensor is at global
  scope (no `blk.N` prefix), shape `[256]`, dtype F32. Attach it to the
  `RotaryEmbedding` struct as a new `freq_factors: Option<Tensor>` field,
  `Some(...)` for the global instance and `None` for sliding.
- **`/opt/hf2q/src/serve/rope_kernel.rs:540-780` — `rope_fused`
  function**: add a `freq_factors: Option<&Tensor>` parameter. When
  `Some`, bind it as `src2` in the encoder at the existing placeholder
  site (`rope_kernel.rs:750-751`) and set `args.src2 = 1` (line 721). The
  MSL kernel at `rope_kernel.rs:261,319` already handles the `src2 != 0`
  branch correctly.
- **`/opt/hf2q/src/serve/gemma4.rs:449-464` — `RotaryEmbedding::apply`
  fused path**: propagate `self.freq_factors.as_ref()` into the two
  `rope_fused()` calls.
- **Loop-mode path at `gemma4.rs:484-504`**: harder. The loop path
  uses a pre-computed `[max_seq_len, half]` sin/cos table. The simplest
  reference-matching fix is to **divide the angles by `freq_factors[i]`
  at table build time** in `RotaryEmbedding::build`:
  ```rust
  let freq_factors: Vec<f32> = load from GGUF or identity;
  let inv_freq: Vec<f32> = (0..half)
      .map(|i| (1.0 / rope_theta.powf(2.0 * i as f64 / head_dim as f64)) as f32
               / freq_factors[i])
      .collect();
  ```
  With `freq_factors[i] = 1e+30` for `i ∈ [64..128)`, the effective
  `inv_freq` becomes ~0, which produces `freqs[pos, i] ≈ 0` → `sin ≈ 0,
  cos ≈ 1`. This is **numerically equivalent** to what the MSL kernel
  computes via `theta/freq_factor`, verified on Spike C's partial-mask
  scratch test (Part 4). Sliding layers pass `freq_factors = None`,
  which maps to identity (`[1.0] * half`), preserving current behavior.

### Fix #2: Correct pair offset for global layers

The `new_partial` constructor was introduced with a misinterpretation of
what "partial rotary" means in the Gemma 4 context. In Gemma 4, the GGUF
explicitly states `rope.dimension_count = 512 = head_dim` — **there is no
partial rotary in the hf2q sense**. The partial-ness is encoded entirely
in the `rope_freqs` mask, not in a shortened `rotary_dim`. The structural
fix is:

- **Delete `new_partial` entirely**. Global layers should use
  `RotaryEmbedding::new_standard(head_dim=512, ...)` just like sliding
  layers (but with `rope_theta = 10^6` and a loaded `freq_factors`
  tensor).
- **Or** rename `rotary_dim` to `effective_rotary_dim` and set it equal
  to `head_dim` for global layers in the loader. The `rope_apply` and
  `rope_fused` code paths already handle `rotary_dim == head_dim`
  correctly (the `if` branch at `gemma4.rs:487`).
- Once `rotary_dim == head_dim`, `rope_apply` pairs elements at offset
  `half = head_dim/2 = 256` — **the same as llama.cpp**. The fused
  kernel caller at `gemma4.rs:453` already passes `self.rotary_dim` as
  `n_dims`, so it will also pass `512` and the MSL kernel will use the
  correct `n_dims/2 = 256` offset.

### Fix scope summary

Both fixes are **Walk-citable**:

- Reference: `/opt/llama.cpp/src/models/gemma4-iswa.cpp:55-59, 73-75,
  97-98` (freq_factors binding) + `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:4376-4426`
  (kernel math, already ported). Zero new kernels.
- Structural change scope: `~20-40 LoC` across `gemma4.rs` (loader +
  `RotaryEmbedding` struct + `apply` loop path), `rope_kernel.rs`
  (`rope_fused` signature + encoder binding), and `config.rs` (drop the
  `partial_rotary_factor_global` field as it is based on a
  misunderstanding).
- Phase A test bar: unit test comparing hf2q's RoPE output against an
  in-process Python reference using `rope_freqs.weight` and the exact
  `theta = pos * (10^6)^(-ic/256) / freq_factors[ic]` formula on
  synthetic `[1, 16, 187, 512]` inputs. Target ε ≤ 1e-5 at the
  per-element level. **Crucially**, this test must compare against a
  reference computed from first principles, not against hf2q's own
  `reference_rope_apply` (which is buggy in the same way the kernel is,
  and why the existing Phase A tests all passed on the broken code).

### End-gate impact

**Walk-correctness End gate** ("hf2q's top-1 token at decode 1 matches
llama.cpp's top-1 token"): **CLOSABLE via this fix**.

Two paths to the gate:

1. On the current `crawl_verify.sh` prompt-rendering path (hf2q's `<bos>`
   text → llama.cpp implicit double-BOS = 188 tokens), llama.cpp picks
   `To`, hf2q picks `The`. **After this fix, both will pick the same
   token** — whether that's `The` or `To` depends on whether the fix
   closes the gap entirely or just most of it. Projection from the
   partial-mask test suggests the fix is sufficient to flip the top-1
   ordering.
2. On a BOS-stripped (byte-identical 187-token) prompt, both tools
   **already** pick `The` today (Spike C side finding, Part 0). The
   End gate is **met on matched inputs before the RoPE fix**, but the
   downstream byte-level continuation divergence is owned by the
   per-layer drift that the RoPE fix would close.

A follow-up `crawl_verify.sh` fix to strip the literal `<bos>` text from
the rendered prompt before passing it to `llama-completion` (or to use
`llama-server /completion` which handles the prompt path differently) is
also needed to make the End-gate measurement byte-level comparable. This
is a **separate** script-level fix from the RoPE port but is a prerequisite
for the End-gate to close on byte-level output comparison.

**Speed impact:** near-zero. The fix adds:
- One `[256]` F32 buffer binding per global-layer RoPE dispatch in the
  fused kernel path (5 layers × 2 = 10 bindings per prefill; once in the
  decode path the same bindings are cached in the existing
  `RopePipelines`). No new kernels.
- In loop mode, one extra division in the table build loop at load time
  (one-off, zero per-forward cost).

---

## Citations

| Claim | File:line |
|-------|-----------|
| llama.cpp Gemma4 build sets `freq_factors` on non-SWA layers only | `/opt/llama.cpp/src/models/gemma4-iswa.cpp:55-59` |
| llama.cpp Gemma4 calls `ggml_rope_ext` with `n_rot_l, freq_factors` | `/opt/llama.cpp/src/models/gemma4-iswa.cpp:73-75` (Q), `:97-98` (K) |
| `n_rot(il)` reads `n_rot_full` for non-SWA, `n_rot_swa` for SWA | `/opt/llama.cpp/src/llama-hparams.cpp:65-71` |
| `kernel_rope_neox` uses `freq_factor = src2[ic]` and `n_dims/2` pair offset | `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:4353-4355, 4376-4426` (pair reads at `:4394-4395`) |
| GGUF `rope_freqs.weight` shape `[256]`, dtype F32, pattern `[1]*64 + [1e30]*192` | `gguf.GGUFReader('/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf')` verified 2026-04-11 |
| GGUF `gemma4.rope.dimension_count = 512`, `dimension_count_swa = 256` | same source, GGUF metadata readout |
| hf2q `new_partial` computes `rotary_dim = 128` for global | `/opt/hf2q/src/serve/gemma4.rs:375-387` |
| hf2q `build` pre-computes sin/cos for 64 frequency pairs only | `/opt/hf2q/src/serve/gemma4.rs:389-428` especially `:398-401` |
| hf2q loop-mode RoPE narrows to `rotary_dim=128` then rotates as `rope_apply` | `/opt/hf2q/src/serve/gemma4.rs:484-504` |
| hf2q `rope_apply` pairs elements at offset `rotary_dim/2 = 64` | `/opt/hf2q/src/serve/gemma4.rs:508-515` |
| hf2q fused RoPE caller hardcodes `src2: 0` | `/opt/hf2q/src/serve/rope_kernel.rs:721` |
| hf2q fused RoPE placeholder src2 binding comment | `/opt/hf2q/src/serve/rope_kernel.rs:750-751` |
| hf2q MSL kernel implements `freq_factor = src2[ic] ? ... : 1.0f` (byte-port) | `/opt/hf2q/src/serve/rope_kernel.rs:319-321` |
| hf2q MSL kernel pairs `x1 = src[args.n_dims/2]` (correct kernel math) | `/opt/hf2q/src/serve/rope_kernel.rs:326-327` |
| hf2q config default `partial_rotary_factor_global = 0.25` | `/opt/hf2q/src/serve/config.rs:88` |
| hf2q serve-side has zero references to `rope_freqs` | `grep -rn rope_freqs /opt/hf2q/src/serve/` 2026-04-11 |
| hf2q writer side generates `rope_freqs.weight` for Gemma4 output GGUFs | `/opt/hf2q/src/backends/gguf.rs:1347-1396` |
| Layer 5 is a full-attention / global layer per hf2q config | `/opt/hf2q/src/serve/gemma4.rs:1857` (`let is_full = cfg.is_full_attention(i)`) |
| Existing Phase A tests compare fused against `reference_rope_apply` | `/opt/hf2q/src/serve/rope_kernel.rs:834, 992-1001` |
| `llama-completion` tokenizes prompt with `add_special=true` | `/opt/llama.cpp/tools/completion/completion.cpp:322` |
| GGUF `tokenizer.ggml.add_bos_token = True` | `gguf.GGUFReader` verified 2026-04-11 |
| Spike B's residual ≥ 96.6% unlocated | `/opt/hf2q/docs/spike-post-1bNEW17-results.md:402, 427-431` |
| ADR Walk-correctness End gate definition | `/opt/hf2q/docs/ADR-005-inference-server.md:189, 760` |

---

## Action items

Not all are in-scope for the next Walk item; listed for the ADR author:

1. **Walk item: Gemma4 RoPE `rope_freqs` port (new 1bNEW.18 or similar).**
   Scope: read `rope_freqs.weight` at load time, rewrite `new_partial` or
   replace it with `new_standard` for global layers, propagate
   `freq_factors` into both loop and fused RoPE paths, add Phase A unit
   tests comparing against a **first-principles** reference (not against
   hf2q's own `reference_rope_apply`). Target: Walk-correctness End gate
   closure.
2. **`crawl_verify.sh` prompt-rendering fix.** Strip the leading `<bos>`
   text before passing to `llama-completion`, OR switch to
   `llama-server /completion` with the rendered template. Without this,
   the byte-level comparison remains structurally wrong.
3. **Test discipline improvement.** The existing Phase A tests for RoPE
   validate internal consistency of hf2q's port against itself and passed
   on known-buggy code. Future kernel ports should compare against a
   reference implementation computed independently (Python script from
   first principles, mlx-lm output, or GGUF-extracted llama.cpp reference
   tensors), not against hf2q's own `reference_rope_apply`.
4. **Spike follow-up (not Walk): fix and re-run Spike B.** With the RoPE
   fix applied, re-measure whether the remaining BF16, router-order,
   and kernel-toggle deltas are still at their ~3% level or whether some
   were hiding behind the RoPE drift and become visible. Most likely
   they shrink further.

---

## Return summary

- **summary:** Per-layer hidden-state bisect on the canonical 187-token
  prompt localizes the residual Walk-correctness drift to **layer 5
  (first full-attention layer)** with a ~300× step-change in `max |Δ|`
  from 4e-3 at layer 4 to 8e-1 at layer 5. Intra-layer drill isolates the
  owning op to `rotary_emb.apply` — the RoPE application on global
  layers. Root cause: hf2q's partial-rotary implementation (a)
  misinterprets Gemma 4's partial rotary as `rotary_dim=128` when the
  GGUF explicitly states `rope.dimension_count=512`, (b) pairs elements at
  offset 64 instead of 256, and (c) never reads the GGUF's
  `rope_freqs.weight` F32 tensor that llama.cpp uses as
  `freq_factors` to selectively mask pairs [64..256) to identity
  rotation. The fused Metal kernel itself is correct (byte-for-byte port
  of llama.cpp's `kernel_rope_neox`) but the caller hardcodes `src2: 0`
  and passes `rotary_dim=128` instead of `n_dims=512`. Walk-citable
  single-item fix: read `rope_freqs.weight` at load, propagate through
  `rope_fused`, delete or repurpose `new_partial`. Scratch partial-fix
  test closes 38% of the logit gap in loop mode (0.771 → 0.435); a
  structural fix that also corrects pair offset projects to close the
  remaining gap to the f32 floor.
- **spike_report_path:** `/opt/hf2q/docs/spike-C-results.md`
- **first_divergence_layer:** **5** (first full-attention / global layer)
- **outcome_class:** **A** (concentrated at a single layer boundary with
  periodic re-injection at the other 4 full-attention layers)
- **owning_op:**
  - hf2q: `/opt/hf2q/src/serve/gemma4.rs:693` (`self.rotary_emb.apply(&q,
    &k, seqlen_offset)`) — resolving into
    `/opt/hf2q/src/serve/gemma4.rs:484-515` (loop path) and
    `/opt/hf2q/src/serve/rope_kernel.rs:540-780` with `src2: 0` hardcoded
    at `:721`.
  - llama.cpp: `/opt/llama.cpp/src/models/gemma4-iswa.cpp:73-75, 97-98`
    (`ggml_rope_ext(ctx0, Q/K, inp_pos, freq_factors, n_rot_l, ...)` with
    `freq_factors = model.layers[il].rope_freqs` from `:55-59`).
- **proposed_fix:** Port llama.cpp's global-layer RoPE exactly: read
  `rope_freqs.weight` at load time, attach to the global `RotaryEmbedding`
  as `freq_factors`, propagate through both loop and fused paths (the
  fused MSL kernel already implements the `src2 != 0` branch). Delete or
  rename `new_partial` — the name and the `partial_rotary_factor_global`
  config field are based on a misinterpretation; Gemma 4 uses full rotary
  with a frequency mask, not a truncated rotary dim. **WALK-CITABLE**
  (single-site port from `/opt/llama.cpp/src/models/gemma4-iswa.cpp:55-98`
  + existing kernel plumbing).
- **walk_correctness_closable:** **YES**. The fix is a one-item Walk port
  with a clear reference and existing kernel support. Phase A test bar:
  RoPE output at ε ≤ 1e-5 vs a first-principles Python reference (NOT vs
  hf2q's own `reference_rope_apply`, which is buggy in the same way).
- **files_changed:** `docs/spike-C-results.md` (new). No other hf2q files
  modified. `git diff --stat src/` empty; `git rev-parse HEAD ==
  67efbf1`. llama.cpp worktree dirty with the eval-callback dump patch in
  `src/llama-context.cpp` — documented in the "llama.cpp patch"
  subsection for revert or re-creation.
- **blockers:** none.
- **confidence:** 0.93. The bisect is unambiguous (4e-3 → 8e-1 step-change
  at layer 5 boundary, Qcur_pos-specific, position-proportional
  divergence pattern exactly matching a wrong-pair-offset RoPE). Source
  citation chain is end-to-end from GGUF metadata through llama.cpp's
  graph builder to the kernel math, and from hf2q's config default
  through the loader to the kernel dispatch, with explicit grep-verified
  absence of `rope_freqs` on hf2q's serve side. The partial-mask scratch
  test empirically validates the fix direction. Confidence reserved from
  1.0 only because the structural fix has not been implemented and
  verified end-to-end — the extrapolation from "partial mask closes 38%"
  to "full fix closes ≥95%" is based on the bisect pattern, not a
  measured run.
