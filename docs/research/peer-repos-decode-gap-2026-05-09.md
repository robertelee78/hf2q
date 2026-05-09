# Peer-Repo + Reddit Research Synthesis — hf2q decode gap (gemma-4-26b)

**Date**: 2026-05-09 (ADR-028 iter-99)
**Inputs**: 8 parallel /swarm-advanced researcher agents (mesh topology) over `/opt/{dflash,ds4,llama.cpp,candle,omlx,vllm}` and `docs/reddit/reddit-{atlas,heretic,mtp}.txt`.
**Mantra**: "as coherent as peers, as fast as or faster than peers."
**Live status (per ADR-028 iter-98)**: prefill MET (1.13×–1.87× faster than llama.cpp HEAD); decode 0.65–0.70× peer at FA=1, only remaining mantra-violation; 990 dispatches/token at GPU pipelined floor (~16 µs each).

Each finding tagged **TESTABLE** (measure-then-decide), **SPECULATIVE** (not yet falsifiable), or **OUT-OF-SCOPE**.

---

## 1. Reddit threads — `docs/reddit/reddit-{atlas,heretic,mtp}.txt`

### `reddit-atlas.txt` — agent `a539debaf8082136a`

- **GB10 = NVIDIA Grace-Blackwell DGX Spark, NOT Apple Silicon.** All "3×" claims trace to either GB10 (irrelevant to our M5 Max target) or to Qwen MTP (see below). **OUT-OF-SCOPE for direct hardware comparison.**
- Real Atlas tg ceiling (counter-data from `dtdisapointingresult`): **13.9 t/s** on Qwen3.5-27B NVFP4 — HF2Q's gemma 63 t/s baseline already crushes this. **Atlas is not the peer to chase.**
- The 3× claim that DOES apply (Qwen3.5-35B): **MTP K=2 self-speculative decode**. Routed to ADR-027 §11, not ADR-028.

### `reddit-heretic.txt` — agent `a0c38e5dfedda7802`

- **OUT-OF-SCOPE.** "Heretic 1.3" is a **version number**, not a speedup. Heretic is an *abliteration* tool (refusal-bypass via weight ablation), not an inference engine. Zero decode-optimization signal. Closing the lane.

### `reddit-mtp.txt` — agent `acc3ff03e3778887b`

- **TESTABLE — algorithm.** MTP K=2-3 self-speculative decode. The primary model emits token T, then the built-in MTP head proposes N speculative tokens (typical optimum N=3); main model verifies in one batched forward. Output quality preserved (only consensus tokens accepted).
- **TESTABLE — architecture requirement.** MTP heads must be **baked into the GGUF**. llama.cpp PR #22673 wires MTP support; converter is in that PR. Existing GGUFs converted without it lack the heads.
- **TESTABLE — local model status.** `models/qwen3.6-27b-mtp-q4_0/config.json` declares `mtp_num_hidden_layers: 1`, but the `.gguf` was hf2q-built Apr 30 — **predates PR #22673's converter**. Tensor-name dump required: look for `*.nextn.*` / `mtp.*` / `embed_tokens_extended`. If absent, must re-convert.
- **TESTABLE — M5 Max baseline.** Direct M5 Max 128GB datapoint: **MTP-on Q8_0 = 37 t/s @1K, 33 t/s @16K** (~2× over MLX baseline, NOT 2.5×). PR caveats: `-np 1` only (no batched serving); vision incompatible; 20% prefill regression; F16 6× slower than Q8_0 (quant choice load-bearing); some Hugging Face GGUFs ship broken (`<|box_end|>` token storm) — use RDson/Radamanthys11/eepos quants.
- **OUT-OF-SCOPE for gemma-4-26b.** Gemma was NOT trained with MTP heads. **MTP cannot close the gemma decode gap.** Belongs in a qwen3.6 ADR branch, not ADR-028.

---

## 2. `/opt/dflash` — block-diffusion speculative decode — agent `a8435840ce4d35a91`

- **TESTABLE — algorithm.** Block-diffusion drafter (default `block_size=16`): tiny transformer takes `[last_verified_token, MASK*15]` and denoises all mask positions in **one parallel non-causal forward pass**. Verifier runs the target on `[tok] + 15_drafts` (16 tokens at once); accepted prefix = longest run where draft argmax == target argmax. KV cache trims `bs - accepted - 1` on reject (`/opt/dflash/dflash/model_mlx.py:561-566`).
- **Net: 1 dispatch chain/token → 2 chain/16 tokens — 8× dispatch reduction at perfect acceptance.** Exact axis where hf2q lags peer.
- **Drafter conditioned on target hidden states** (`model_mlx.py:139, 188-194`) — concat of multi-layer features projected through a Linear; explains higher acceptance than EAGLE/Medusa.
- **Embeddings + lm_head shared with target** (`model_mlx.py:153-168, 188`) — small download, but bind step reaches into target's `embed_tokens`/`lm_head`. Q5_K_M target ≠ F16 draft today: needs **dequant-on-load of draft** (drafter is tiny, cost negligible) or a quantized-aware draft.
- **TESTABLE — gemma-4-26B-A4B-it draft EXISTS**: `z-lab/gemma-4-26B-A4B-it-DFlash` (`/opt/dflash/README.md:14`). The exact target hf2q runs.
- **TESTABLE — M5 Max applicability.** MLX path tested on Apple M5 Pro with Qwen3, Qwen3.5, Gemma-4 (`README.md:145`). M5 Max is uphill.
- **SPECULATIVE — speedup.** README does not publish numeric speedup; benchmark harness (`benchmark.py:120-135`) compares `bs=1` vs `bs=16` TPOT. oMLX integration doc reports **45.3 tok/s with 87.2% acceptance** on Qwen3.5-27B (`/opt/omlx/docs/experimental/dflash_mlx_integration.md:249`); 3-4× claim on Apple Silicon for that model.

---

## 3. `/opt/ds4` — DeepSeek V4 Flash engine — agent `aa5d4c280cb47f2ae`

- **OUT-OF-SCOPE for K-quants.** DS4 has **zero Q5_K or Q6_K Metal kernels**. Only IQ2_XXS / Q2_K / Q4_K (MoE expert quants) and Q8_0. NOT a K-quant reference for gemma-4-26b Q5_K_M.
- **TESTABLE — fusion patterns hf2q likely lacks (dense-applicable):**
  - `kernel_dsv4_qkv_rms_norm_f32_4` (`norm.metal:102`): single dispatch RMS-norms BOTH Q-lora and KV rows. Eliminates one dispatch per attn block. **Pre-norm gemma is structurally a clean fit.**
  - `kernel_dsv4_shared_gate_up_swiglu_q8_0` (`dense.metal:203-271`): **one kernel** = gate matvec + up matvec + SiLU + mul, reusing the input row in registers. Comment at `:200-202` is explicit: "the point is not to fuse two independent weight streams into one matmul; it is to remove the separate activation pass." For gemma-4 dense FFN: 3 dispatches → 1 per layer × 28 layers = ~56 dispatches/token saved.
  - **TESTABLE — persistent batch encoder** (`ds4_metal.m:223-235`): ONE compute encoder open across entire decode step (`g_batch_enc`). `endEncoding` only on (a) blit insert, (b) explicit `flush_commands`, (c) `end_commands` at step boundary. Lets Metal's automatic in-encoder serialization track RAW hazards without per-dispatch encoder/barrier overhead. **If hf2q's 16 µs/dispatch floor is partly encoder-rebuild overhead, this could reclaim a fraction of the 30% peer gap.**
- **TESTABLE — env-gated A/B knobs.** All fusions gated by `DS4_METAL_DISABLE_*_FUSION` envs (`ds4.c:8569-8602`) — explicit falsifier infrastructure. We should mirror this pattern.
- **OUT-OF-SCOPE — MoE fusions** (`kernel_mul_mv_id_*_pair_swiglu_f32`, `_sum6_f32`): gemma-4-26b is dense.
- **SPECULATIVE — MTP.** DS4 implements 1-2 draft tokens via `mtp.0.*` weights with prefix-1 commit (`ds4.c:15093`). README is honest: **"currently provides at most a slight speedup, not a meaningful generation-speed win."** Validates that MTP gain is model-specific, not engine-universal.

---

## 4. `/opt/llama.cpp` recent — agent `abafeb0b5d59f415a`

- **Recent commits (last 30d) — only 13, mostly non-decode:**
  - `d1649047a` (Apr 25 #20962) — splits `kernel_mul_mm` Tensor API path from legacy `simdgroup_matrix`. **Likely contributes to peer prefill, not decode** (mm not mv) — bench at decode `n=1` to confirm null effect.
  - `8635e221c`, `7fc1c4ef7`, `c0de6eda7` — event-sync fix / GPU watchdog workaround / FA support-logic fix. **OUT-OF-SCOPE.**
  - **No decode-mv kernel changes in 30d.** The decode pipeline is mature.
- **TESTABLE — fusion patterns hf2q lacks:**
  - `kernel_rms_norm_fuse_impl<vec_type, F>` template (`ggml-metal.metal:2986-3059`), `F=1/2/3` instantiated as `_f32_4` variants. **One dispatch replaces 3 kernels** (rms_norm + mul(weight) + add(bias)).
    - **Cross-check vs hf2q**: we already have `fused_norm_add_f32` (3-op fusion) per ADR-028 iter-93. The remaining gap is the **`_4` (vec4) suffix** path — quad-loaded vectorized variant, ~2.8-3.4% delta per iter-93 ROI estimate.
  - `kernel_bin_fuse_impl` template (`ggml-metal.metal:1209-1364`) with `_f32_f32_f32_4` chains contiguous BIN ops in one dispatch. Open task #14.
  - **Graph-walker fusion driver** (`ggml-metal-ops.cpp:3339-3430`): `ggml_metal_op_norm` walks the graph forward via `ctx->can_fuse(idx+n_fuse, fops, 2)` and merges up to 2 follow-on ops; per-op dispatch table at `:268-385` returns `n_fuse` 1-3.
- **TESTABLE — FA-vec split-K threshold** (`ggml-metal-ops.cpp:2944-3052`): `nwg=1` for short kL; `nwg=32` when `2*nwg*nsg*ncpsg < ne11 && nsg < 4`; `nwg=32` triggers a 2-stage path with `flash_attn_ext_vec_reduce` post-pass over `ne01_max=min(ne01,32)*ne02*ne03*nwg*(ne20+2)` F32 tmp buffer (`:2619-2625`). Decode (ne01=1) hits split-K when **kL ≥ ~512**. **Cross-check vs hf2q**: `nwg` parameter exists in `flash_attn_vec.metal:43,107,355,362` — confirm we're using `nwg=32` at decode kL ≥ 512.
- **TESTABLE — host-side dispatch / command-buffer count.**
  - `GGML_METAL_MAX_COMMAND_BUFFERS = 8` (`ggml-metal-context.m:20`); `n_cb = MIN(n_cb, 8)`; warns when `n_cb > 2`. Empirically recommended `n_cb ∈ {1,2}` per M1 Pro / M2 Ultra comment at `:458`.
  - `n_main = MAX(64, 0.1 * gf->n_nodes)` (`:445`) — nodes encoded inline by main thread, remainder split across `n_cb` async threads via `dispatch_apply` (`:551`). For a 26B-model decode graph (~1200 nodes): ~120 nodes inline, rest split → **typically 2-3 MTLCommandBuffer commits per decode step**.
  - All command buffers use `commandBufferWithUnretainedReferences` (`:512, :532`) — same path hf2q gates behind `MLX_UNRETAINED_REFS=1`.
- **SPECULATIVE — speculative decoding** (`common/speculative.cpp:21-41`): supports `draft`, `eagle3`, `ngram_simple`, `ngram_map_k/k4v`, `ngram_mod`, `ngram_cache`. **No flags in `llama-bench`** — public bench numbers (88-97 t/s on gemma-4) are **non-speculative**. **The 30% peer gap is NOT speculative-driven.** Speculative is additive on top of fixing the kernel-time gap.

---

## 5. `/opt/candle` — Rust inference — agent `a6ba138004ee73108`

**MOST IMPORTANT CORRECTION.**

- **TESTABLE — ADR-028's "candle baseline ~105 dispatches/token" claim is unsourced and almost certainly wrong.**
  - Cited at ADR-028 lines 732 and 775; not derivable from candle's source. CHANGELOG, READMEs, model code contain no such number.
  - Per `quantized_llama.rs:558-586` and `quantized_qwen3_moe.rs:125-219`, each candle decoder layer issues separately: attention_norm, Q/K/V proj (3 mv), Q/K rotary (2), Q/K-norm (qwen3, 2), kv-cache cat (2), SDPA (1), o-proj, residual-add, ffn_norm, gate/up/down (3 mv per active expert), routing-softmax, residual-add. **~14 dispatches × 28 layers ≈ 400/token** for dense Llama-7B; **~40/layer × layers** for 8-active-expert MoE.
  - **Same order of magnitude as hf2q's 990.** "9.4× headroom" claim collapses.
- **TESTABLE — candle has minimal fusion.** No fused `rmsnorm_mul`, no `silu_mul` (SwiGLU), no `add_residual_norm`. `Mlp::forward` (`quantized_llama.rs:57-62`) does silu, mul, matmul as 3 separate ops. Only fused decode primitive is **SDPA-vector** (vendored from MLX) — same fusion hf2q has via `flash_attn_vec`.
- **TESTABLE — candle K-quant path.** Uses ggml-metal's `kernel_mul_mv_q5_K_f32` (`quantized.metal:5159`) and `kernel_mul_mv_q6_K_f32` (`:5269`) — bit-identical to llama.cpp HEAD. `metal.rs:227 fwd_mv` issues **one dispatch per batch row** at decode m=1. Same primitive class hf2q is on.
- **TESTABLE — no Apple tensor-core direct invocation.** candle uses `simdgroup_matrix<8,8>` + `simdgroup_multiply_accumulate` (`mlx_gemm.metal:302-374`). **No `mpp::tensor_ops::matmul2d`** anywhere. M3+ AMX/AME not directly invoked. Same Metal primitive class hf2q is on. **Apple tensor cores remain a greenfield optimization for both engines.**
- **TESTABLE — adopt candle/MLX 2-pass SDPA-vector pattern for long context.** `scaled_dot_product_attention.metal:434 sdpa_vector_2pass_1` + `:577 sdpa_vector_2pass_2` splits long-K SDPA-vec into a partial-reduce + final-merge pair to keep more SMs busy at low Q-rows. Addresses the FA_GL D=512 12.59 ms/call hot path (ADR-028 iter-92 perf table). **Kernel-time optimization (real lever), not a dispatch-count one.**
- **OUT-OF-SCOPE — candle has no published Apple-Silicon Gemma/Qwen decode tok/s** in CHANGELOG / READMEs / examples. Cannot validate ADR-028's "105" baseline against any candle figure.
- **The structural gap is per-dispatch GPU time** (~16 µs hf2q vs ~11 µs llama.cpp HEAD per ADR-028 iter-90 table line 740), **not graph density.**

---

## 6. `/opt/omlx` + `/opt/vllm` — agent `add7f3535440fde93`

### oMLX (`/opt/omlx`)
- **TESTABLE — oMLX has no native kernels.** Python wrapper around `mlx-lm` (`omlx/models/llm.py:151,204` calls `mlx_lm.generate`/`stream_generate`). Kernel inventory and dispatch policy live in upstream MLX/mlx-lm.
- **TESTABLE — `mx.compile` is used selectively, NOT in the LLM decode loop.** Only embedding (`embedding.py:392`) and reranker (`reranker.py:692`) wrap forward in `mx.compile`. **Autoregressive LLM step has no fused single-dispatch kernel.** Same kernel-launch model hf2q faces. **mx.compile-style graph fusion for gemma-4 decode is greenfield, not "follow the leader."**
- **OUT-OF-SCOPE — oQ quantization.** Software quantization (bits 2/3/3.5/4/5/6/8 mixed-precision per layer) producing standard mlx-lm safetensors. NOT GGUF K-quants. Different format, separate scope.
- **TESTABLE — DFlash-on-Qwen3.5-27B M5 Max datapoint**: 45.3 tok/s with 87.2% acceptance, 38 cycles (`docs/experimental/dflash_mlx_integration.md:249`). 3-4× single-request speedup claimed (`engine/dflash.py:163`). Bundled draft checkpoints exist for Qwen3.5 family only — **no Gemma-4 draft yet** (z-lab has one per `/opt/dflash/README.md:14`, but oMLX has not bundled it).

### vLLM (`/opt/vllm`)
- **OUT-OF-SCOPE — no Apple Metal backend.** Platforms are `cuda/rocm/tpu/xpu/cpu/zen_cpu` (`/opt/vllm/vllm/platforms/`). On macOS-arm64 falls through to CPU platform (`platforms/cpu.py:54-66`). Attention backends CUDA/Triton/ROCm only.
- **TESTABLE — N-gram speculative decode** (`v1/spec_decode/ngram_proposer.py:1-50`): **no draft model needed**. Pure CPU-side proposer + 1 batched verify forward. With hf2q's 16 µs/dispatch × 990 floor, accepting k=3-5 tokens/cycle yields effective 2-4× decode speedup. **Lowest risk, highest leverage; fits gemma-4-26b today.**
- **SPECULATIVE — algorithm catalog**: `v1/spec_decode/{eagle.py,medusa.py,dflash.py,ngram_proposer.py,mtp.py,suffix_decoding.py,draft_model.py}`. Reported 2-3× latency reduction (`docs/features/speculative_decoding/speculators.md:22`).
- **OUT-OF-SCOPE — PagedAttention, continuous batching.** CUDA/Triton-only; no Metal port in repo.

---

## Synthesized verdict

### What changes in ADR-028
1. **RETRACT the "candle baseline ~105 dispatches/token" claim** at lines 732 and 775. Replace with "candle ≈ 400-1000 dispatches/token (same order as hf2q's 990)". Per Chesterton's fence: the 9.4× headroom number was a fence with no foundation; tearing it down is correct.
2. **Reframe the gap.** Per-dispatch GPU time (~16 µs hf2q vs ~11 µs llama.cpp HEAD) is the structural lever, NOT dispatch density. Iter-71's empirical "dispatch count is at parity with llama.cpp" stands.

### Action items ranked by ROI × risk
| # | Action | Tag | Source | Est ROI | Risk |
|---|--------|-----|--------|---------|------|
| A | Verify FA-vec `nwg=32` engaged at decode kL ≥ 512 | TESTABLE | llama.cpp ggml-metal-ops.cpp:2944-3052 | small (long-ctx only) | low |
| B | Fuse gate+up+SwiGLU into one Q5_K mv_id kernel for gemma-4 dense FFN | TESTABLE | DS4 dense.metal:203-271 | ~56 dispatches/token (~5%) | medium |
| C | Persistent batch-encoder pattern (one compute encoder per decode step) | TESTABLE | DS4 ds4_metal.m:223-235 | unknown — measure encode-CPU first | medium |
| D | Port `kernel_rms_norm_fuse_impl<F=2/3>` `_4` (vec4) suffix variant | TESTABLE | llama.cpp ggml-metal.metal:2986-3059 | ~2.8-3.4% per iter-93 | low |
| E | 2-pass SDPA-vector for long-K (kL ≥ 512) | TESTABLE | candle/MLX scaled_dot_product_attention.metal:434+577 | hot path FA_GL D=512 12.59 ms/call | medium |
| F | N-gram speculative decode (verify-batched, no draft model) | TESTABLE | vLLM v1/spec_decode/ngram_proposer.py | 2-4× at acceptance ≥ 60% | medium |
| G | DFlash block-diffusion drafter (z-lab gemma-4-26B-A4B-it-DFlash exists) | SPECULATIVE | dflash + omlx | 3-4× claimed Apple Silicon | high |
| H | MTP K=3 self-spec for **qwen3.6**, NOT gemma | OUT-OF-SCOPE for ADR-028 | reddit-mtp + ds4 | 2× M5 Max for qwen | high |
| I | `bin_fuse_f32_f32_f32_4` chain | TESTABLE | llama.cpp bin_fuse_impl 1209-1364 | ~6% per ADR-028 #14 | low |
| J | Apple `mpp::tensor_ops::matmul2d` for K-quants | SPECULATIVE | greenfield (neither candle nor llama.cpp use it) | unknown — research-grade | very high |

### Scope split
- **ADR-028 (gemma-4-26b decode parity)**: items A, B, C, D, E, I (kernel/scheduling) → close 0.65× → ≥ 1.0×.
- **ADR-027 §11 (qwen MTP / spec decode)**: items F, G, H — separate workstream; MTP architecturally inapplicable to gemma.
- **OUT-OF-SCOPE for now**: J (Apple tensor cores) — research-grade, no peer reference.

### Not pursuing
- Heretic (abliteration tool, no inference signal).
- vLLM PagedAttention (no Metal port).
- oQ quantization (different format, gemma-4-26b ships as Q5_K_M GGUF).
- Atlas (real Atlas decode 13.9 t/s — slower than us).

### Mantra status post-research
- Coherence: ✅ MET.
- Prefill: ✅ MET (1.13×–1.87× faster than llama.cpp HEAD).
- Decode: ❌ 0.65–0.70× peer at FA=1. Items A→I form the closure plan; J is research overhang.
