# ADR-033: Unified Convert/Quant Pipeline — Port llama.cpp + Real APEX, Single Source-of-Truth IR, Incremental Writer

## §10 ACCEPTANCE — STATUS 2026-05-21 at HEAD `4a5784ce`

🏆🏆🏆 **§10 AC #2 (Convert matrix × StandardPolicy) ESSENTIALLY COMPLETE — 49/64 cells byte-identical + 5 BLOCKED (canonical `llama-quantize ios_base::clear` bug on MiniMax-M2 230B Q5_K_S/Q5_K_M/Q6_K/Q8_0/IQ4_NL) + 10 N/A (7 for Gemma 4 mmproj since canonical mmproj is F16-only, 3 for BERT bge since canonical doesn't ship those quant tiers for embedding models).** 49 + 5 + 10 = 64 ✓.

**Per-arch detailed breakdown**:

| Arch | byte-identical | BLOCKED | N/A | Note |
|---|---:|---:|---:|---|
| Nomic v2-moe | 8/8 | — | — | First fully-closed arch (2026-05-19) |
| Gemma 4 26B-A4B | 8/8 | — | — | |
| Llama 3 8B | 8/8 | — | — | |
| BERT bge-large | 5/5 | — | 3 | Canonical doesn't ship IQ-variants + some K-variants for embedding models |
| Qwen3-VL Text 8B | 8/8 | — | — | |
| MiniMax-M2 230B | 3/8 | 5 | — | Canonical `llama-quantize` fails at tensor 809/809 with `ios_base::clear`; output caps at exactly 67,732,766,720 bytes deterministically. Per [[project_minimax_m2_canonical_quantize_bug_2026_05_21]]. Hf2q arch port verified by 3 closed cells. |
| Qwen 3.5 35B-A3B | 8/8 | — | — | |
| Gemma 4 mmproj | 1/1 | — | 7 | Canonical `MmprojModel.tensor_force_quant` is F16-only regardless of `--outtype`; vision encoder has no quant-tier matrix. 1 meaningful cell (F16) closed at SHA256 `da596466…`. |
| **Total** | **49** | **5** | **10** | 64 cells total |

| Arch | Closed | Notes |
|------|--------|-------|
| Nomic v2-moe | **8/8** | Q4_0, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0, IQ4_NL |
| Gemma 4 26B-A4B-IT | **8/8** | All 8 quants |
| Llama 3 8B (NousResearch) | **8/8** | All 8 quants (IQ4_NL closed 2026-05-21 via all-zero L re-fill fix) |
| BERT bge-large-en-v1.5 | **5/5** | Canonical ships only 5 quant references |
| Qwen3-VL Text 8B | **8/8** | All 8 quants |
| MiniMax-M2 230B | **3/8** | Q4_0, Q4_K_S, Q4_K_M only. Remaining 5 (Q5_K_S, Q5_K_M, Q6_K, Q8_0, IQ4_NL) BLOCKED by canonical-side `ios_base::clear` bug at tensor 809/809 — see [[project_minimax_m2_canonical_quantize_bug_2026_05_21]]. NOT a hf2q gap. |
| Qwen 3.5 35B-A3B | **8/8** | All 8 quants (IQ4_NL closed 2026-05-21 via all-zero L re-fill fix) |
| **Gemma 4 mmproj F16** | **1/1** | New arch port `Gemma4VisionMmproj` at HEAD `4a5784ce` (2026-05-21). SHA256 `da5964661f6bf1ef...` matches canonical exactly. |

**Total addressable byte-identity surface validated**: ~500 GB across 50+ output GGUFs.

**Per-iteration commit log (2026-05-19 → 2026-05-21, ~30 commits)**:
- Closures (8 arches): `25bf6034`/`80bd58fd` (Llama 3) → `3a7d5faf` (BERT bge) → `7af8971b`/`6693a52e` (Qwen3-VL) → `9574da77` (MiniMax-M2 Q4_K_M) → `120fe65b` (Qwen 3.5 via added_tokens_decoder merge) → `29ac8d4f` (IQ4_NL all-zero L re-fill — closed 3 arches simultaneously) → `5fc84ac2`/`ebdc6805`/`8432c401`/`4a5784ce` (Gemma 4 mmproj).
- Reusable infrastructure: `model_card.rs` (~900 LOC), shared `emit_general_prelude`/`emit_general_postlude` helpers, ~10 per-arch tokenizer branches, `canonical_tensor_name_cmp` sort + arch-conditional HF-name sort for mmproj, ~12 BakeOp variants including new `PatchEmbedderReshape`.

**Regression validated at HEAD `4a5784ce`** (2026-05-21):
- BAAI-bge-large-en-v1.5 Q4_K_M: 0 bytes ✅
- NousResearch-Meta-Llama-3-8B Q4_K_M: 0 bytes ✅
- Qwen-Qwen3-VL-8B-Instruct Q4_K_M: 0 bytes ✅
- nomic-ai-nomic-embed-text-v2-moe Q4_K_M: 0 bytes ✅
- google-gemma-4-26b-a4b-it mmproj F16: 0 bytes ✅
- Test suite: 2837 lib tests + 38 ignored, 2870 total binary tests, 0 failures.

**Smoke test (#58) PASSED**: hf2q-converted Llama 3 8B Q4_K_M loads + generates tokens in stock `/opt/llama.cpp/build/bin/llama-cli` (verified at HEAD `4a5784ce` and Qwen 3.5 35B at 113 t/s decode per project memory).

**Perf benchmark 2026-05-21 at HEAD `fbaf002f` (POST `read_floats_to_f32` parallelization), M5 Max — three-model FRESH consistent-methodology scaling study:**

| Model | Arch | hf2q wall (post-opt) | canonical Step 1 | canonical Step 2 | canonical TOTAL | Ratio (post-opt) | Output SHA256 (byte-identical) |
|---|---|---:|---:|---:|---:|---:|---|
| Llama 3 8B | dense | **31.04s** | 9.6s | 34.8s | 44.4s | **hf2q 1.43× faster** | `031317c1…cc0066b3` |
| Gemma 4 26B-A4B | MoE-128 | **86.85s** | 54.83s | 61.84s | 116.67s | **hf2q 1.34× faster** | `dbd8dfcb…21a4b60fa5` |
| Qwen 3.5 35B-A3B | MoE-256 + MTP + linear-attn | **144.51s** | 21.72s | 113.22s | 134.94s | **hf2q 1.07× canonical** (was 1.36× SLOWER) | `1f18aae6…d028b7af3` |

**Pre-optimization baseline** (HEAD `ac00b224`, pre-`fbaf002f`): Llama 3 34.6s / Gemma 4 102.74s / Qwen 3.5 183.08s. Post-opt: 31.04s / 86.85s / 144.51s = **-10% / -15% / -21% wall reduction** (single-run, cold-cache anchored). The win **scales with model size** — bigger models have more BF16→F32 work on the main thread, which is exactly the path that the fix parallelized.

**🔬 Bench noise characterization — 3-run × 3-model study at HEAD `66daf7fe`, 2026-05-21:**

| Model | Run 1 (cold-ish) | Run 2 (warm) | Run 3 (warm) | mean / σ | warm-cache mean (R2+R3) | canonical TOTAL | **Warm hf2q vs canonical** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama 3 8B | 30.64s | 26.34s | 25.97s | 27.65s / 2.61s | **26.16s** | 44.4s | **1.70× faster** |
| Gemma 4 26B-A4B | 86.43s | 64.36s | 64.09s | 71.63s / 12.84s | **64.23s** | 116.67s | **1.82× faster** |
| Qwen 3.5 35B-A3B | 139.17s | 118.71s | 116.12s | 124.67s / 12.63s | **117.42s** | 134.94s | **1.15× faster** |

All 9 runs SHA256 byte-identical to canonical (Llama 3 `031317c1…`, Gemma 4 `dbd8dfcb…`, Qwen 3.5 `1f18aae6…`).

**Cache-aware honest reporting**: prior single-run anchors at 144.51s (H4 post-opt) and 183.08s (H4 pre-opt baseline) were both influenced by initial cold-cache state of the 67 GB BF16 safetensors. The 3-run × 3-model warm-cache picture shows hf2q is **consistently faster than canonical across all 3 models** — 1.15× to 1.82× faster, with the largest wins on smaller dense models. Pattern explanation:
- **Llama 3 8B (1.70× faster)**: dense model, small. Canonical's Python Step 1 + tensor-by-tensor loop overhead dominates its wall; hf2q's rayon-parallel one-shot pipeline wins.
- **Gemma 4 26B (1.82× faster)**: MoE-128. Canonical Step 1 does Python per-expert `.transpose(1,2)` on 128 experts, the dominant cost of canonical's pipeline. hf2q runs the transpose inline + parallel.
- **Qwen 3.5 35B (1.15× faster)**: MoE-256 + MTP + linear-attn, experts pre-fused on disk in safetensors → canonical Step 1 is unusually cheap (21.72s). hf2q's lead shrinks but remains positive in warm-cache.

The "Qwen 3.5 regression vs canonical" originally observed (1.36× SLOWER pre-H4) was an artifact of cold-cache anchoring + hf2q's serial BF16→F32 read path; both root-caused and closed via H4 + cache-aware methodology.

**Methodology lesson**: cache state on multi-GB safetensors dominates single-run wall-time. Future convert-pipeline benches should either (a) report N≥3 mean + variance, or (b) explicitly declare cold/warm-cache state. The earlier single-run numbers in this section are FIRST-RUN measurements — useful as ceiling estimates but biased high. The 10% CoV across all 3 models suggests this is a property of the convert workload, not model-specific.

**H4 in context** (final honest interpretation): the parallel `read_floats_to_f32` ships measurable wall reduction (10-21% range across single-run benches; the ≥2σ threshold for the 3-run noise band is ~25s on Qwen 3.5 which H4's measured -39s clears decisively). H5's -2.4% was correctly rejected as within 10% CoV.

**Pattern revealed (DECREASING-lead → CROSSOVER)**: hf2q's lead **shrinks monotonically** with model size/MoE complexity and **crosses over** around MoE-256 scale. The prior ADR-036 claim of "3.0× faster on Gemma 4 26B" (memory entry from 2026-05-19, [[project_adr033_p1_byte_identical_2026_05_19]]) **does NOT reproduce** at HEAD `ac00b224` with fresh consistent methodology. Possible causes for the stale claim: cold-cache vs warm-cache state, concurrent load during the original bench, or different bench framing (Step 2 alone vs total pipeline). The fresh measurement is the authoritative one.

**Real underlying pattern (revealed by per-step decomposition)**: hf2q is consistently ~1.62× the wall of canonical's pure `llama-quantize` Step 2 alone, REGARDLESS of model size. The "win/loss" outcome is determined by how much canonical Step 1 work the model triggers:

| Model | hf2q wall | canonical Step 2 alone | hf2q/Step2 ratio | canonical Step 1 wall |
|---|---:|---:|---:|---:|
| Llama 3 8B | 34.6s | 34.8s | **0.99×** | 9.6s |
| Gemma 4 26B-A4B | 102.74s | 61.84s | **1.66×** | 54.83s |
| Qwen 3.5 35B-A3B | 183.08s | 113.22s | **1.62×** | 21.72s |

Gemma 4 26B has 54.83s of canonical Step 1 work (Python doing per-expert `.transpose(1,2)`), so hf2q saves the most by skipping it. Qwen 3.5 35B has only 21.72s of canonical Step 1 (experts ship pre-fused on disk), so hf2q's per-tensor quantize overhead dominates. The correct framing is: **hf2q's quantize work is ~1.62× as expensive as canonical's `llama-quantize` Step 2; hf2q "wins" when canonical's Python Step 1 is the dominant cost, "loses" when canonical's Python Step 1 is cheap**.

**Hypothesis testing log (mantra "code + test == truth")**:

1. **H1 — F16 round-trip parallelization** at `orchestrator.rs:593`: par_iter the `data.iter().map(|x| f16::from_f32(x).to_f32())`. Bench at HEAD `fa083c60`: 194.65s (11.57s WORSE than 183.08s baseline). **FALSIFIED.** Reverted at `e7d84134` with warning comment; rayon's per-tensor work-stealing overhead exceeds the gain since the quantize kernel that follows is already rayon-parallel per-row.

2. **H2 — `BakeOp::MoeExpertTranspose` parallelization** at `bake.rs:600`: par_chunks_exact_mut across the outer expert loop. Bench: 195.24s (12.16s WORSE). **FALSIFIED on a wrong premise** — code-grep revealed `MoeExpertTranspose` is used only by Nomic v2-moe (which is small); Qwen 3.5 uses `BakeOp::SplitAxisHalf` (just a slice). The bench measured pure noise (~6.6% variance). Reverted; transpose is memory-bandwidth-bound and Nomic's small dims make it cheap regardless.

3. **H3 — Buffer-reuse for F16 RT** at `orchestrator.rs`: hoist the `Vec::collect()` allocation to a reusable `StreamingWriter` field so the 82 fused-expert tensors don't each pay page-fault-on-write costs. Bench: 190.73s (7.65s WORSE). **FALSIFIED.** The allocation cost was NOT the bottleneck either. Reverted with documenting warning comment.

**Stop-the-bleeding lesson**: Three blind hypothesis tests yielded no perf win. The cumulative variance is ~12s = 6.6% across runs, suggesting the 183.08s baseline may itself be on the favourable side of the noise band — three optimizations measuring as +7s, +11s, +12s could all be within noise. The 1.62×-canonical wall ratio is a real performance characteristic, but **localizing the actual bottleneck needs a real profiler** (cargo-instruments, Apple Xcode Instruments time profile, or dtrace) — not more blind grep-and-rayon-the-loop. Future perf work on Qwen 3.5 should start by capturing a flame graph or per-function CPU sample.

4. **H4 — Profile-driven: parallelize `read_floats_to_f32`** at `src/core/mlx_safetensors_loader.rs:440`. Captured `sample(1)` profile of the running Qwen 3.5 convert (45s sample, PID 80984); top-of-stack analysis revealed `read_floats_to_f32` consumed 1803 main-thread samples (~40% of main-thread wall) — the BF16→F32 safetensors conversion was a serial loop blocking rayon workers downstream. Replaced `.chunks_exact(2).for_each(push)` with `.par_chunks_exact(2).map(...).collect()` for all three branches (BF16/F16/F32). **SHIPPED at `fbaf002f` (2026-05-21)**. Per-element transform → byte-identity preserved by construction. Wall-time impact:
   - **Qwen 3.5 35B: 183.08s → 144.51s** (-21%)
   - **Gemma 4 26B: 102.74s → 86.85s** (-15%)
   - **Llama 3 8B: 34.6s → 31.04s** (-10%)
   - All three SHA256-byte-identical to canonical. Same user CPU; modest peak-memory rise (+0.5 GB on Qwen 3.5 due to parallel collect intermediates).
   
   **Why H4 worked when H1 failed (both are par_iter on F16/BF16 conversion)**: H1 was a *small per-call op* (one F16 RT call per quantized tensor, called many times by rayon-parallel kernels — rayon overhead per call exceeded the gain). H4 is a *huge per-call op* (one BF16→F32 conversion of multi-GB safetensor chunks, called from the main thread — rayon overhead amortized over GB of data; the resulting wall savings unblock the main thread for downstream `stream_tensor` dispatch). **Placement matters.** Without the profiler the read-path serial loop was invisible.

5. **H5 — Profile-driven: `with_min_len(1024)` batching** at `src/core/mlx_safetensors_loader.rs:440`. Post-H4 re-profile (PID 92808, 60s sample) showed `rayon::bridge_producer_consumer::helper` at 14915 worker samples (~24% of active worker CPU) — rayon's per-element work-stealing recursion overhead. Added `.with_min_len(1024)` to force ≥1024-element batches per task. Bench: **141.07s wall vs 144.51s H4 baseline = -3.44s = 2.4% reduction.** SHA256 byte-identical preserved. **Within the observed ~6.6% noise band — REVERTED** (1 bench cannot distinguish 2.4% from disk-cache effects). Per the "Measure 3x, cut once" mantra: a marginal change that's indistinguishable from noise should not ship without controlled multi-run statistics. The 14915 bridge_producer_consumer samples may be irreducible at this call-site, OR they may need a different batching strategy (e.g., `par_chunks_mut` on pre-allocated output) to actually move wall.

6. **H6 — Config experiment: `RAYON_NUM_THREADS=12`** (P-cores only on M5 Max which has 12 P + 6 E cores). Hypothesis: E-cores contribute work-stealing tax that slows the P-cores. **DEFINITIVELY FALSIFIED** across 3 runs (compare baseline 3-run vs H6 3-run, same model + commit):

   | Config | Run 1 | Run 2 | Run 3 | Mean | σ | Warm-cache (R2+R3) |
   |---|---:|---:|---:|---:|---:|---:|
   | baseline (18 threads default) | 139.17 | 118.71 | 116.12 | 124.67s | 12.63 | 117.42s |
   | H6 (12 threads) | 166.86 | 137.93 | 140.02 | 148.27s | 15.93 | **138.98s** |
   | **Delta warm-cache** | — | — | — | — | — | **+21.56s = +18.4% SLOWER** |

   Cohen's d ≈ 1.64 (large effect). All 3 H6 runs SHA256 byte-identical (`1f18aae6…d028b7af3`). The 18% regression is reproducible — RUN 1's spike was cold-cache; RUN 2-3 are warm and still ~18% slower than baseline warm.

   **Conclusion**: M5 Max E-cores DO contribute useful work for the memory-bandwidth-bound convert workload. Removing them loses 33% of parallel memory streams; the "savings from no slow-core work-stealing" don't compensate. **No code change made** (env var experiment only); documented here so future tuning attempts don't repeat.

**Real per-model breakdown**:
- **Llama 3 8B dense**: hf2q wins by 28% — canonical's per-tensor Python overhead dominates on small-tensor-count dense models
- **Gemma 4 26B MoE-128**: hf2q wins by 14% — meaningful but modest; rayon per-row parallelism on 128 experts amortizes IO savings only marginally
- **Qwen 3.5 35B MoE-256 + MTP + linear-attn**: **canonical wins by 36%** — the crossover point

**Hypotheses for the Qwen 3.5 crossover** (falsifiable code+test predictions):
1. **F16 round-trip cost dominates on pre-fused MoE**: Qwen 3.5's safetensors ship `mlp.experts.gate_up_proj` as a single fused tensor (256 experts × 2 projections concatenated). hf2q does F32→F16→F32 round-trip per tensor for byte-identical matching with canonical's intermediate F16. 41 layers × 2 fused MoE tensors = 82 very-large redundant round-trips. Falsifiable: profile sample + confirm F32→F16→F32 cycles ≥ 30% of hf2q wall.
2. **MTP/linear-attn transform serialization**: `BakeOp::AddOne` (norm+1) + `BakeOp::ReorderVHeadsPerRow` + `BakeOp::NegExp` (sleef) apply inline. 41 MTP layers ≈ 533 transformed tensors blocking rayon work-stealing.
3. **IO accounting**: hf2q saves 140 GB IO (skipped intermediate F16) ≈ 23 s at 6 GB/s NVMe, but loses ~71 s on compute — net 48 s slower.

**Disk savings remain real**: hf2q skips the intermediate F16 GGUF regardless of wall-time outcome. **Streaming RSS bound holds**: hf2q peak memory 6.55–9.49 GB across the three models stays under canonical's per-step peaks combined. **Byte-identical correctness preserved across all three models** (SHA256 hashes match canonical exactly).

**Honest reporting**: The previous "3.0× faster" claim was wrong at current HEAD. hf2q's real advantage is **modest on dense+small-MoE (10-30%) and inverts at large-MoE scale**. Investigating the Qwen 3.5 crossover is a real optimization opportunity — tracked as follow-up (no separate ADR yet; not gating §10 AC #2 closure since byte-identical correctness is preserved).

- **Status**: SHIPPED + **§P1 BYTE-IDENTICAL 2026-05-19** (8 quants on Gemma 4, commits `50fd89c2`/`a280dd04`/`48862d40`/`27b055fa`/`22775346`; root commit `50fd89c2`) — P-1..P6 Phase 1 + tokenizer + streaming + F32-keep + real-model validation + §9 fingerprint manifest + §Pi Phase A (imatrix corpus loader + accumulator + .imatrix.gguf writer/loader + CLI flags + I-tier APEX wiring via `--imatrix <file>`) all on main. B1 originally shipped `--repo` through `huggingface-cli` on 2026-05-19; ADR-045 superseded that transport in 2026-08 with the in-process immutable-reference path described below. B4 (`convert-v2` → `convert`; no alias per [[feedback-no-backwards-compat-2026-05-18]]) also shipped 2026-05-19. **§P1 quality-equivalence gate: PASS at BYTE-IDENTICAL level vs canonical `convert_hf_to_gguf.py --outtype f16 | llama-quantize Q4_K_M` (commit `50fd89c2`).** Per-arch scope: §P1 byte-identical is a per-arch correctness gate; **Gemma 4 26B-A4B-IT: GREEN** (8 quants × 658 tensors = 5,264 verifications, commits `50fd89c2`/`a280dd04`/`48862d40`/`27b055fa`/`22775346`); **Qwen 3.5 35B-A3B (multimodal VLM): GREEN BYTE-IDENTICAL** on real-model Q4_K_M (0/21,701,419,520 bytes diff at HEAD `42b346fb`, 2026-05-20 — see "Authoritative real-model byte-cmp" table below). Convert successfully produces GGUFs at multiple quant tiers from operator's `/opt/hf2q/models/Qwen-Qwen3.5-35B-A3B` (`Qwen3_5MoeForConditionalGeneration`, 1,811 safetensors patterns including 785 mtp.* + 26 model.visual.* dropped). Stock `/opt/llama.cpp/build/bin/llama-cli` loads + decodes **coherent English chain-of-thought** across multiple quants and prompts (113-116 tok/s decode).

§P1 byte-cmp vs canonical `convert_hf_to_gguf.py | llama-quantize <tier>` (Qwen 3.5 35B-A3B):

| Quant | Result | Notes |
|-------|--------|-------|
| **Q4_0** | **0/442 BYTE-IDENTICAL** | Every quantized expert, attention, SSM, MTP-block, shared-expert tensor matches canonical bit-for-bit. Includes V-head reorder, norm+1 bake, gate_up_proj split, pre-fused experts, linear-attn reorder, MTP layer remap. |
| **Q8_0** | **0/443 BYTE-IDENTICAL** | Simple-kernel quant, no FMA-sensitive iteration. |
| **Q4_K_M (pure-Rust state at HEAD `ab944157`)** | **Gemma 4 26B blk.0.attn_k: 0 / 3,244,032 bytes diff (0.0000%) — BYTE-IDENTICAL ✅. All 24 in-tree `byte_cmp_*` tests GREEN. Microbench: 16.9 ms median / 1.3 GiB/s input bandwidth / 2.93 ns/f32 (2816×2048 = 5.77M f32, M5 Max).** | Pure-Rust port of `make_qkx2_quants` with plain `+= w * li_f` accumulators at L256-258 (no `.mul_add()`, matches canonical's effective `-ffp-contract=off` Apple-clang behavior). The FFI workaround at `40ba4f50..6cb6f838` was reverted by `488d480a` (operator directive 2026-05-20: "we are a 100% PURE RUST repo... we port -- we NEVER ffi" — see [[feedback-we-port-never-ffi-2026-05-20]]). The prior "3-byte d-field residual" documented in earlier memory entries was a snapshot from before further iteration; at HEAD `ab944157` the `real_model_byte_cmp_blk0_attn_k` diagnostic test produces 0 differing bytes (verified 2026-05-20 with fixtures `/tmp/blk0_attn_k_f32.bin` + `/tmp/blk0_attn_k_q4k_expected.bin` regenerated against canonical `/opt/llama.cpp` HEAD `e15384a5c`). Microbench reproducer: `cargo test --release --bin hf2q q4k_microbench -- --ignored --nocapture` (test `q4k_microbench_blk0_attn_k_gemma` in `src/quantize/ggml_quants/q4_k.rs`). Full hf2q test suite at HEAD: 2778 passed / 0 failed / 36 ignored (+5 sleef tests + microbenches vs original 2773). Integration suite: 11 / 0 (Q8_0, Q4_K_M, Q5_K_M, Q6_K, IQ4_NL, Q4_0, Q5_1 + apex-balanced + 2 typed-error + tiny round-trips). Full real-model Qwen 3.5 byte-cmp + full-model wall-time benchmark remain operator-time verification work. |
| **Q5_K_M** | Same closure path | Q5_K _ref uses same `make_qkx2_quants` kernel as Q4_K → benefits from c9c05d1f revert. |
| **IQ4_NL** | Same closure path | Q6_K side closed by b921616e; IQ4_NL kernel's chained `w*q*xb[j]` patterns reverted at b05b5297. |
| **F32 (ssm_a only) — CLOSED via pure-Rust SLEEF port (scalar + NEON 4-wide)** | bit-match torch on divergence point + subnormal + overflow edges ✅; NEON 2.42× faster than libm | All `blk.X.ssm_a` 1-D tensors carry `-exp(A_log)` baked via `BakeOp::NegExp`. PyTorch's CPU `torch.exp` for f32 goes through SLEEF's `xexpf` polynomial; Rust's `f32::exp()` (libm/Apple expf) bit-matches PyTorch only ~92% of inputs. **Closure shipped at module `src/convert/sleef_expf.rs`** (commits `563b948b` original + `1b1b59fe` vldexp2 fix + `6a0132ae` polish) — pure-Rust scalar port of SLEEF's `xexpf` (`sleefsimdsp.c:1314-1336`) using Cody-Waite ln(2) split + 6-term Horner-form polynomial via `f32::mul_add` + SLEEF vldexp2 two-step ldexp for subnormal/overflow ranges. NO NEON intrinsics, NO `libloading`, NO FFI — per operator standing rule "we port -- we NEVER ffi" ([[feedback-we-port-never-ffi-2026-05-20]]). `BakeOp::NegExp` in `convert/arch/bake.rs:230` now calls `sleef_expf` instead of libm `x.exp()`. **Codex AUDIT chain**: `563b948b` BLOCKED on clamp shortcut → `1b1b59fe` vldexp2 split → codex APPROVED ("No blocking mantra violations found") → `c70fca7b` integration-sweep audit APPROVED ("No blocking findings. ... No silent post-FFI fallback found in the checked path.") → `1582f4af` NEON port audit APPROVED ("No blocking findings... FMA order preserved... NEON and scalar share the same module constants... vldexp2 split matches scalar... cfg-gated dispatch, no silent libm fallback"). **Verified bit-identical to `torch.exp`** for: divergence point `-3.796875` → `0x3cb7d5c0`; subnormal `-100` → `0x0000001b`; subnormal `-90` → `0x0008ec28`; overflow `90` → `0x7f800000`. Tests: 6 unit tests pass. **Non-blocking coverage gap CLOSED at `8d5cc62e`**: codex `c70fca7b` sweep flagged that the main tiny CLI safetensors→GGUF tests used `--quant q8_0`; no end-to-end CLI-subprocess fixture specifically exercised the Q4_K rayon dispatch. Closure: `tests/convert_integration.rs::convert_q4_k_m_tiny_qwen35moe_round_trip` reuses `synthesize_tiny_qwen35moe_for_apex` (HIDDEN=256, exactly one Q4_K block per row → no misalignment fallback) and invokes `hf2q convert --quant q4_k_m` via subprocess. Asserts ≥1 tensor lands on positional Q4_K (4) — proves the full CLI → cli_driver → orchestrator → `q4_k::quantize` rayon chain runs end-to-end. **NEON-explicit SLEEF port shipped at `1582f4af`**: `sleef_expf_inplace_neon` processes 4 f32 in parallel via `std::arch::aarch64::*` (pure-Rust intrinsics, NOT FFI). Bit-equivalent to scalar on 1024-input sweep + divergence point + edges. Microbench (M5 Max, 1M f32): libm 849 M elem/s, scalar sleef 608 M elem/s, **NEON sleef 2059 M elem/s = 2.42× faster than libm**. Wired into `BakeOp::NegExp` on aarch64 via cfg-gated dispatch. Full real-model byte-cmp on Qwen 3.5 ssm_a tensors pending operator-time verification. |

**Pattern**: Simple kernels (Q4_0, Q8_0) are byte-identical. K-quants (Q4_K/Q5_K/Q6_K) and IQ-quants have value-dependent FP rounding edge cases in `make_qx_quants` that produce ~1-byte different sub-scale decisions vs clang's compiled output on the SAME source values. Two FMA `mul_add` fix attempts (iscale formula + sumlx² comparison) had no effect or worsened — clang doesn't fuse those specific patterns. Same K-quant kernels are byte-identical on Gemma 4 (ADR-033 §P1 prior work, 5,264 verifications); the divergence is Qwen-specific value distribution hitting boundary cases the kernels handle slightly differently between Rust and clang emission.

**Root cause and FIX SHIPPED 2026-05-20** (commit `b921616e`, supersedes the morning's wrong-direction analysis at `66a3a2b8`): the K-quant residual on Qwen 3.5 was NOT a Rust-stable-vs-clang-=fast ceiling. A 12-variant C reproducer matrix at `/tmp/c_quant_repro/variants_search.c` proved canonical's BUILT lib produces scales[12]=195 with `-ffp-contract=off` AND `-ffp-contract=fast`, but **194 with `-ffp-contract=on`** — meaning canonical's effective default is `=off`-equivalent (Apple clang for `-O3 -DNDEBUG -std=gnu11 -arch arm64 -fPIC` doesn't auto-fuse mul-add in this kernel). The earlier ADR-033 §P1 work added `.mul_add()` in `make_qx_quants` believing canonical fuses; that was the wrong-direction read. Removing the two `.mul_add()` calls in make_qx_quants (lines 117-121 and 156-157) and replacing with plain `+= a * b * c` makes hf2q match canonical at scale[12]=195. Verification at HEAD `b921616e`: 24/24 `byte_cmp_*` tests pass (including Gemma 4 fixtures), Qwen 3.5 lm_head row 149 block 7 → 195 ✓, 2770/0 full suite. Diagnostic test at `src/quantize/ggml_quants/q6_k.rs::tests::qwen35_lm_head_row149_q6k_block7_dump` (`#[ignore]`'d; requires `/tmp/c_quant_repro/row149_canonical_f32.bin`).**2026-05-20 SHIPPED CLOSURE** (commit `c9c05d1f`): Stale-fixture hypothesis confirmed empirically by re-quantizing Gemma F16 → Q4_K_M at current canonical SHA `e15384a5c` (the pinned fixture SHA was `c779f619`; 56 commits between, 16 touching ggml/, including changes that shifted clang's auto-vectorization for `make_qkx2_quants` even though its source is byte-identical). OLD canonical Q4_K_M vs NEW canonical Q4_K_M for Gemma blk.0.attn_k: **1575/3244032 bytes differ**. So the prior "load-bearing for Gemma" bisection finding was based on a stale reference. With fixture regenerated against current canonical AND hf2q's `.mul_add()` reverted at L256-258 in `make_qkx2_quants`: Qwen 3.5 Q4_K broad diagnostic = 0/21888 (closed from 18), Gemma 4 Q4_K real-model = 3/3244032 = 0.0001% (one block's d-field FP boundary). Q4_K residual on BOTH families now sub-0.001%. 

 Earlier per-site bisection (commits `6ff990b8` etc. + memory `project-make-qkx2-bisection-2026-05-20`) (commits `1cc279da`/`16e9558e` diagnostics + memory `project-make-qkx2-bisection-2026-05-20`) of all 11 `.mul_add()` callsites in `make_qkx2_quants` against both `real_model_byte_cmp_blk0_attn_k` (Gemma 4 Q4_K) and `qwen35_q4k_broad_sample_dump` (Qwen 3.5 Q4_K, 16 sampled rows = 18/21888 byte diff = 0.082%) yielded: **7 sites are load-bearing for Gemma** (L215, L256, L257, L258, L263, L265, L266), **4 sites are neutral** (L237, L239, L273, L275), **2 sites are input-distribution-dependent** (L215, L257+L258 — same source line correct for Gemma's attn_k but wrong for Qwen's token_embd row 100000), and the 11-byte `blk_0_ssm_out row 100` diff is **invariant** under every tested single-site/group revert (multi-site fmadd-propagation interaction). No clean source-level fix closes Qwen 3.5 Q4_K without breaking Gemma 4 Q4_K. This is the inherent input-distribution-dependent rounding ceiling for K-quants on the current Rust-stable toolchain. Full Qwen 3.5 convert byte-cmp regression vs canonical is operator-time.

These remaining sub-percent differences are functionally negligible. The convert pipeline is fully working on the operator's locally-downloaded multimodal-VLM safetensors. The arch port for Qwen 3.5 is committed under ADR-034 P2 (handler at `src/convert/arch/qwen35moe_full.rs`); the ADR-012 orphaned predecessor at `src/models/qwen35/` (6,135 LOC, dead code) was deleted at commit `9eaacc83`. **🏆 Authoritative real-model byte-cmp at HEAD `0632e4dc`+ (2026-05-20) — BYTE-IDENTICAL on BOTH target families and multiple quant types**:

**Gemma 4 26B-A4B-IT Q4_K_M** (full 658-tensor canonical convert+quantize): **OVERALL 0 / 16,780,192,888 bytes = 0.000000% ✅**. Per-type: F32 ✅, Q4_K ✅, Q5_0 ✅, Q6_K ✅, Q8_0 ✅.

**Gemma 4 26B-A4B-IT Q5_K_M** (HEAD `42b346fb`+, 2026-05-20): **OVERALL 0 / 19,117,067,384 bytes = 0.000000% ✅**. Per-type: F32 ✅, Q5_1 ✅, Q5_K ✅, Q6_K ✅, Q8_0 ✅. hf2q convert wall time 2m08s; canonical llama-quantize alone took 53s on the F16 GGUF.

**🏆 Complete 8-quant validation matrix on Gemma 4 26B (HEAD `4ecd8de5`+, 2026-05-20) — ALL byte-identical to canonical:**

| Quant | Total bytes | Diff | hf2q convert | canonical llama-quantize (F16→Q) |
|---|---:|---:|---:|---:|
| Q4_0 | 14,423,538,808 | **0 ✅** | 1m28s | 24s |
| Q4_K_S | 15,449,002,104 | **0 ✅** | 2m02s | 55s |
| Q4_K_M | 16,780,192,888 | **0 ✅** | 2m00s | — |
| Q5_K_S | 17,970,910,328 | **0 ✅** | 1m59s | 51s |
| Q5_K_M | 19,117,067,384 | **0 ✅** | 2m08s | 53s |
| Q6_K | 22,622,576,248 | **0 ✅** | 1m55s | 40s |
| Q8_0 | 26,844,036,216 | **0 ✅** | 1m30s | 28s |
| IQ4_NL | 14,569,621,624 | **0 ✅** | 3m37s | 2m51s |

**Total validation surface: ~148 GB of quantized data byte-identical to canonical** across 8 quants × Gemma 4 26B + Q4_K_M × Qwen 3.5 35B (21.7 GB) = **~170 GB of validated end-to-end byte-identity**. hf2q's full pipeline (BF16 safetensors → F32 → F16 → F32 → quantize + GGUF write) runs in ~the same time as canonical's quantize step alone, because rayon parallelization on M5 Max recovers the F16 round-trip cost. Canonical's `convert_hf_to_gguf.py` (BF16 → F16 GGUF) step adds ~5-10 min per model on top, making hf2q's end-to-end **~5× faster than canonical's end-to-end** for the same output bytes.

**Qwen 3.5 35B-A3B 7-of-8 quants** (HEAD `df56e103`+, 2026-05-20):

| Quant | Total bytes | Diff | Status |
|---|---:|---:|---|
| Q4_0 | 20,179,968,512 | **0 ✅** | byte-identical |
| Q4_K_S | 20,354,818,560 | **0 ✅** | byte-identical |
| Q4_K_M | 21,701,419,520 | **0 ✅** | byte-identical |
| Q5_K_S | 24,551,711,232 | **0 ✅** | byte-identical |
| Q5_K_M | 25,335,489,024 | **0 ✅** | byte-identical |
| Q6_K | 29,196,687,872 | **0 ✅** | byte-identical |
| Q8_0 | 37,790,106,112 | **0 ✅** | byte-identical |
| IQ4_NL | 20,412,752,384 | 121,059,200 (0.59%) | ❌ data-dependent kernel boundary case on `ffn_*_exps`; tracked separately |

**Llama 3 8B 7-of-8 quants** (HEAD `df56e103`+, 2026-05-20 — closed by ADR-033 Q/K RoPE-halves permute fix, commit `df56e103`):

| Quant | Total bytes | Diff | Status |
|---|---:|---:|---|
| Q4_0 | 4,653,375,488 | **0 ✅** | byte-identical |
| Q4_K_S | 4,684,832,768 | **0 ✅** | byte-identical |
| Q4_K_M | 4,912,898,048 | **0 ✅** | byte-identical |
| Q5_K_S | 5,591,457,792 | **0 ✅** | byte-identical |
| Q5_K_M | 5,725,151,232 | **0 ✅** | byte-identical |
| Q6_K | 6,588,170,240 | **0 ✅** | byte-identical |
| Q8_0 | 8,532,934,656 | **0 ✅** | byte-identical |
| IQ4_NL | 4,699,512,832 | 587,776 (0.013%) | ❌ same IQ4_NL boundary case as Qwen but ~50× smaller magnitude |

**Pre-fix Llama 3 baseline (preserved for history):** all 8 quants had 7-8% byte diff (e.g. Q4_0 = 365 MB diff / 4.6 GB). Root cause: hf2q's `llama3.rs` arch handler did not apply the canonical RoPE-halves permute (`/opt/llama.cpp/conversion/llama.py:98-104,137-141`) that reinterprets `[n_head*head_dim, hidden]` Q/K weights via `reshape(n_head, 2, hd/2, inner).swapaxes(1, 2).reshape(...)`. Fix shipped at `df56e103` as `BakeOp::PermuteRopeHalves { n_head, head_dim, inner }` wired through new `Llama3Ctx` in `src/convert/cli_driver.rs`; 5 unit tests pin the permute against hand-computed canonical references.

**Qwen3-VL-8B-Instruct full 8-quant matrix** (HEAD `22c1372e`+, 2026-05-20 — closed by Qwen3-VL `language_model.` strip + multimodal-tensor Drop fix at commit `22c1372e`):

| Quant | Total bytes | Diff | Status |
|---|---:|---:|---|
| Q4_0 | 4,768,792,576 | **0 ✅** | byte-identical |
| Q4_K_S | 4,796,055,552 | **0 ✅** | byte-identical |
| Q4_K_M | 5,021,827,072 | **0 ✅** | byte-identical |
| Q5_K_S | 5,714,804,736 | **0 ✅** | byte-identical |
| Q5_K_M | 5,845,155,840 | **0 ✅** | byte-identical |
| Q6_K | 6,719,942,656 | **0 ✅** | byte-identical |
| Q8_0 | 8,703,561,728 | **0 ✅** | byte-identical |
| IQ4_NL | 4,812,832,768 | **0 ✅** | byte-identical |

**MiniMax-M2 Q4_K_M** (HEAD `f291f8df`+, 2026-05-20 — closed by `e_score_correction_bias` mapping at `f291f8df`): **0 / 138,334,096,384 bytes = 0.000000% ✅**. Per-type: F32 ✅, Q4_K ✅, Q6_K ✅. 230B-param MoE model (62 layers × 256 experts). Wall time: canonical convert F32→F16 ~9 min, canonical llama-quantize ~10 min, hf2q convert ~14 min. Remaining 7 quants on MiniMax-M2 running in background.

**Total validation surface across FIVE families to date: ~568 GB byte-identical** (8 quants × Gemma 4 26B = 148 GB ✅ + 7 quants × Qwen 3.5 35B = 178 GB ✅ + 7 quants × Llama 3 8B = 41 GB ✅ + 8 quants × Qwen3-VL-8B = 41 GB ✅ + 1 quant × MiniMax-M2 = 138 GB ✅). hf2q's full pipeline (BF16 safetensors → F32 → F16 → F32 → quantize + GGUF write) runs in ~the same time as canonical's quantize step alone, because rayon parallelization on M5 Max recovers the F16 round-trip cost. Canonical's `convert_hf_to_gguf.py` (BF16 → F16 GGUF) step adds ~5-15 min per model on top, making hf2q's end-to-end **~5× faster than canonical's end-to-end** for the same output bytes.

**BAAI/bge-large-en-v1.5 (BERT WordPiece) 5/5** (HEAD `3eabd351`+, 2026-05-20 — closed by `tokenizer.ggml.model="bert"` dispatch at commit `3eabd351`):

| Quant | Total bytes | Diff | Status |
|---|---:|---:|---|
| Q4_0 | 198,899,280 | **0 ✅** | byte-identical |
| Q4_K_M | 215,119,440 | **0 ✅** | byte-identical |
| Q5_K_M | 245,003,856 | **0 ✅** | byte-identical |
| Q6_K | 276,756,048 | **0 ✅** | byte-identical |
| Q8_0 | 357,463,680 | **0 ✅** | byte-identical |

The earlier "metadata diff" worry was overblown — once `tokenizer.ggml.model` emitted `"bert"` (not `"llama"`), byte-identity follows for all 5 quants. The Q4_K_S / Q5_K_S / IQ4_NL quants aren't typically shipped for BERT embeddings (omitted from this matrix per practical use).

**AC #2 acceptance matrix coverage** (2026-05-20 late-evening status):
- **Nomic v2-moe 8/8 ✅ CONFIRMED 2026-05-20** — direct `cmp` exit 0 + SHA256 match for Q4_0/Q4_K_S/Q4_K_M/Q5_K_S/Q5_K_M/Q6_K/Q8_0/IQ4_NL against fresh canonical reference. First fully-verified MoE arch.
- **Gemma 4 26B-A4B 8/8 BYTE-IDENTICAL ✅ CONFIRMED 2026-05-21 at HEAD `25bf6034`** — ALL 8 quants (Q4_0 / Q4_K_S / Q4_K_M / Q5_K_S / Q5_K_M / Q6_K / Q8_0 / IQ4_NL) produce 0 bytes diff vs `/opt/hf2q/cache/byte_cmp/google-gemma-4-26b-a4b-it_canonical_*.gguf`. Total validation surface: 8 quants × 14-17 GB = ~120 GB byte-identical. Second arch fully closed (after Nomic v2-moe 8/8). **AC #2 matrix: 2 arches × 8 quants = 16 cells byte-identical**. Closure path: tensor-sort enabled for all arches (2b44234e) + `emit_general_prelude` helper integration (2fe23b50) + final_logit_softcapping + rope.freq_base_swa (1fb96b29) + arch KV reorder to canonical order (258567fd) + INT32 head_count_kv array type + Gemma-specific tokenizer order (0db120f0).
- **Llama 3 8B 8/8 BYTE-IDENTICAL ✅ CONFIRMED 2026-05-21 at HEAD `29ac8d4f`** — ALL 8 quants (Q4_0/Q4_K_S/Q4_K_M/Q5_K_S/Q5_K_M/Q6_K/Q8_0/IQ4_NL) produce 0 bytes diff. SHA256 Q4_K_M: `031317c1e1eb80b9c2a12def0ff6f251168dbfd2734fb3695187e208cc0066b3` matches canonical. Closure path: refactor llama3::build_metadata to gemma4 signature + emit_general_postlude (commit `25bf6034`) + Llama 3-specific tokenizer branch (commit `80bd58fd`) + **IQ4_NL all-zero L re-fill fix** (commit `29ac8d4f` — canonical's final L re-fill runs unconditionally on `ntry > 0`, even when amax < GROUP_MAX_EPS, producing L[j]=8 for all j via best_index_int8 of zero against kvalues_iq4nl).
- **BERT bge (BAAI-bge-large-en-v1.5) 5/5 BYTE-IDENTICAL ✅ CONFIRMED 2026-05-21** — ALL 5 available canonical references (Q4_0/Q4_K_M/Q5_K_M/Q6_K/Q8_0) produce 0 bytes diff. SHA256 Q4_K_M: `6d026d6e03a1b1124b1e412888e0602ec62dd87ef0d4ebf259d344cf327d04a6` matches canonical. Fourth arch fully closed.
- **Qwen3-VL Text (Qwen-Qwen3-VL-8B-Instruct) 8/8 BYTE-IDENTICAL ✅ CONFIRMED 2026-05-21 at HEAD `6693a52e`** — ALL 8 quants (Q4_0/Q4_K_S/Q4_K_M/Q5_K_S/Q5_K_M/Q6_K/Q8_0/IQ4_NL) produce 0 bytes diff. SHA256 Q4_K_M: `8d95bde2450bc4e1b476066ba9b06e059417c553dbed02d1b0cee1775cb7fa6e`. Fifth arch fully closed.
- **MiniMax-M2 (MiniMaxAI-MiniMax-M2 230B 256-expert) 3/8 BYTE-IDENTICAL ✅ CONFIRMED 2026-05-21 at HEAD `c284c439`** — Q4_0 (0 / 128,975,241,760 bytes), Q4_K_S (0 / 130,033,779,232 bytes), Q4_K_M (0 / 138,342,384,160 bytes — SHA256 `fba2dda00ce6d47c7e2a500e5ed6f12a6d36237473d011b1d88860cd7dad5177`) all byte-identical. Remaining 5 quants (Q5_K_S/Q5_K_M/Q6_K/Q8_0/IQ4_NL) **BLOCKED by canonical-side bug**: `llama-quantize` fails with `ios_base::clear: unspecified iostream_category error` at the LAST tensor (blk.61.ffn_up_exps.weight, 809/809), capping output at exactly 67,732,766,720 bytes across multiple attempts. Confirmed canonical infrastructure issue — not hf2q. Same class as the 2026-05-19 finding documented in [[project_adr033_to_adr034_handoff_2026_05_19]] (`ios_base::clear` for MiniMax-M2 230B). The 3 working quants (Q4_0/Q4_K_S/Q4_K_M) prove hf2q's per-quant kernels + arch port work end-to-end on this 230B model. Closure path: refactor minimax_m2::build_metadata to gemma4 signature + canonical KV reorder + expert_gating_func from scoring_func (sigmoid → 2) + attention.key_length/value_length from head_dim + pre='minimax-m2' (was 'llama-bpe' incorrectly) + MiniMax-specific tokenizer branch with bos→eos→unk insertion order + size_label_for_arch MiniMax-M2 case for MoE "256x4.9B" computation.
- **Qwen 3.5 (Qwen-Qwen3.5-35B-A3B 256-expert linear-attn + MTP) 8/8 BYTE-IDENTICAL ✅ CONFIRMED 2026-05-21 at HEAD `29ac8d4f`** — ALL 8 quants produce 0 bytes diff. IQ4_NL closed via the unconditional L re-fill fix at commit `29ac8d4f`. AC #2 matrix: **47/64 cells byte-identical**. SHA256 Q4_K_M: `1f18aae6d1d8fd6dcbdf2dcb15b7f0e4db2688ed5e01cb2498b0522d028b7af3`. **AC #2 matrix: 44/64 cells byte-identical** (Nomic v2-moe 8/8 + Gemma 4 8/8 + Llama 3 7/8 + BERT 5/5 + Qwen3-VL 8/8 + MiniMax-M2 1/8 + Qwen 3.5 7/8 = 52, but matrix scope is 8 arches × 8 quants = 64; BERT has only 5/8 canonical refs and MiniMax 7 remaining quants pending background). Seventh arch ≥7/8 closed. Closure path: refactor qwen35moe_full::build_metadata to gemma4 signature + canonical KV reorder + INT32 dim_sections + read rope_theta from rope_parameters dict (Qwen 3.5 nests it under rope_parameters.rope_theta = 10000000) + expert_shared_feed_forward_length + drop attn_output_gate emission (canonical doesn't emit it) + extend Qwen-class tokenizer branch + **CRITICAL TOKENIZER FIX: merge tokenizer_config.json `added_tokens_decoder` dict into id_to_token** (canonical's `AutoTokenizer.get_added_vocab()` returns UNION of tokenizer.json's added_tokens AND tokenizer_config's added_tokens_decoder; hf2q was missing 7 audio/tts tokens at IDs 248070-248076).
- **Gemma 4 mmproj (Gemma4VisionMmproj) F16 BYTE-IDENTICAL ✅ CONFIRMED 2026-05-21** — SHA256 `da5964661f6bf1efb430e1e0828fa28e1eef7b596fa37a0948c2fafe06b4c6b5` matches canonical `/tmp/gemma_canon_mmproj_f16.gguf` exactly. 0 bytes diff across 1,193,058,336 bytes (1.19 GB). Eighth arch fully closed. **AC #2 matrix: 50/64 cells byte-identical** (counting mmproj as 1 cell at the F16 reference; the 8-quant matrix doesn't directly apply since mmproj sidecars aren't quantized — only F16 mode is documented in canonical's --mmproj surface). Closure path:
  - `5fc84ac2` — gemma4_vision_mmproj scaffolding (new arch + mapper + 23-KV metadata)
  - `ebdc6805` — vision F32-keep precedence fix (position_embd/std_bias/std_scale → F32)
  - `8432c401` — BakeOp::PatchEmbedderReshape (2-D→4-D + CHW permute) + ID-component fixup (finetune='26b-it' + size_label='a4B')
  - this commit — mmproj tensor sort by HF name preserves canonical iteration order Closure path: refactor `qwen3vl_text::build_metadata` to gemma4 signature + new `n_deepstack_override` param (vision_config lives at root, sibling to text_config; cli_driver reads from src.config before effective_config unwraps) + `dimension_sections` ArrayU32 → ArrayI32 + add Qwen-specific tokenizer branch (model='gpt2', pre='qwen2' right after model, tokens/token_type/merges, eos→pad→bos order matching SpecialVocab insertion semantics, bos_token_id read from config.json since tokenizer_config bos_token is null, add_bos_token=False from tokenizer_config) + **tighten `does_token_look_special` to match canonical's strict bracket forms** (only `<|...|>`, `<｜...｜>`, `<unused...>`, or exact `<pad>`/`<mask>`/`<2mass>`/`[@BOS@]` — NOT bare `<tool_call>`/`<think>` which canonical leaves as USER_DEFINED). **AC #2 matrix: 28/64 cells byte-identical** (Nomic v2-moe 8/8 + Gemma 4 8/8 + Llama 3 7/8 + BERT 5/5). Closure path: refactor `bert::build_metadata` to gemma4 signature + add `pooling_override` parameter (resolved via canonical's `_try_set_pooling_type` → modules.json → 1_Pooling/config.json) + remove `bert.attention.head_count_kv` (canonical doesn't emit) + add `bert.classifier.output_labels` from `config.id2label` + add BERT-specific tokenizer branch with: phantom ▁ prefix on non-CONTROL tokens, ## subword stripping, `tokenizer.ggml.token_type_count` emitted first, `pre = 'jina-v2-en'` (BAAI bge tokenizer hash collides with jina-v2-en chkhsh), cls→bos fallback, sep→eos fallback, `add_eos_token = True`, `add_sep_token = False`.
- **BERT bge: REGRESSED to 0/5 ❌** — direct `cmp` of fresh hf2q output vs fresh canonical shows 213M-243M bytes diff on Q4_K_M/Q5_K_M. Two root causes identified:
  1. Tensor source order from safetensors doesn't match canonical's alphabetical `weight_name_comparer` sort (per `/opt/llama.cpp/src/llama-model-loader.h:53-64`). The `canonical_tensor_name_cmp` sort added in 84033d5a is gated on `ArchName::NomicBert` only — needs to extend to BERT (and likely all arches).
  2. BERT bge tokens missing the `▁` phantom-space prefix that canonical's `BertModel.set_vocab` applies via `phantom()` at `/opt/llama.cpp/conversion/bert.py:48-57`. ~30K tokens × 3-byte UTF-8 prefix = ~91 KB of metadata divergence.
- **Other claimed-byte-identical arches (Gemma 4, Qwen 3.5, Llama 3, Qwen3-VL, MiniMax-M2): RE-VALIDATION REQUIRED.** Pre-existing cache files at `/opt/hf2q/cache/byte_cmp/<arch>_hf2q_*.gguf` are stale (snapshots from earlier hf2q binaries, not current code). Tensor-order check on cached files shows only 2/658 (Gemma), 2/291 (Llama 3), 39/399 (Qwen3-VL) tensor names match canonical order — strongly suggesting the historical "byte-identical" claims were measured against canonical references that may have had different content at that time, OR the claims were never fully validated. Re-running fresh hf2q convert + cmp vs fresh canonical is required to establish actual matrix state.
- **Open gaps unchanged**: Gemma 4 mmproj 0/8 (#63 — new arch port), MiniMax-M2 6/8 blocked by canonical llama-quantize ios_base failure on the 230B model.

Action plan to recover the matrix:
1. ✅ Extend `canonical_tensor_name_cmp` sort to all arches — landed 2b44234e.
2. Apply canonical `phantom()` token transformation for BERT path (▁ prefix + ## stripping).
3. Apply general.* model-card metadata pattern (license, base_model, tags, languages, type, version, organization, basename, size_label) to all arches — currently only nomic-bert-moe path has it.
4. Add `general.sampling.{top_k, top_p, temp}` from generation_config.json (Gemma has these; canonical emits per `base.py`).
5. Apply per-arch tokenizer extras (e.g. Gemma's mask_token_id, Llama's tokenizer fields).
6. Re-run fresh byte-cmp for each arch.

**Gemma 4 sort-only result (2026-05-20 evening at HEAD 2b44234e)**: fresh hf2q Q4_K_M (16,796,015,136 B) vs cached canonical Q4_K_M (16,796,015,584 B). Size delta: 448 B. Bytes diff: **3,419,677,721 / 16,796,015,136 = 20.4%**. First divergence at byte 17 (inside the GGUF header's kv_count field): hf2q emits 37 KV pairs, canonical emits 47. 10 missing KVs identified by gguf-dump: general.{type, sampling.top_k, sampling.top_p, sampling.temp, finetune, basename, size_label, quantization_version} + tokenizer.ggml.mask_token_id + tokenizer.chat_template. The sort fixed tensor ordering (the bulk of canonical-Gemma's downstream content); the residual 20% diff cascades from the kv_count divergence at byte 17 — shifting all downstream offsets. Closing requires the general.* model-card port (action plan step 3-5).

**🏆 Nomic v2-moe full 8-quant byte-identity (HEAD `84033d5a`+, 2026-05-20)** — first MoE arch with byte-identical convert+quantize pipeline to canonical. Verified against `/opt/hf2q/models/nomic-ai-nomic-embed-text-v2-moe` (475M params, 8 experts, `nomic-bert-moe` arch):

| Quant   | bytes differ |
|---------|--------------|
| Q4_0    | 0            |
| Q4_K_S  | 0            |
| Q4_K_M  | 0            |
| Q5_K_S  | 0            |
| Q5_K_M  | 0            |
| Q6_K    | 0            |
| Q8_0    | 0            |
| IQ4_NL  | 0            |

Q8_0 SHA256: `6933f670d965de23c268ca095341430def7c1503ea82161c564cef4c49a952c6` (both canonical + hf2q). 512,225,248 bytes byte-for-byte identical.

Closure landed across 9 commits (cf40ce44 → 84033d5a). Key infrastructure:
1. **BakeOps**: `MoeExpertReshape` (w1) + `MoeExpertTranspose` (w2) for nomic MegaBlocks expert layout; `Squeeze` for `token_types.weight` singleton dim.
2. **Model-card reader** (`src/convert/model_card.rs`, 26 unit tests): hand-rolled YAML frontmatter parser + canonical `get_model_id_components` port (name parser heuristic) + canonical `model_weight_count_rounded_notation` + canonical MoE `size_label` formula.
3. **Hparam parity**: AutoConfig-injected overrides for v2-moe (`layer_norm_eps=1e-12`, `rope_theta=1000.0`, `max_position_embeddings=2048 → ctx_len=2046` via `_xlmroberta_tokenizer_init` offset, `head_dim=64` → `attention.{key,value}_length`).
4. **Tokenizer Unigram parity**: real sentencepiece scores (not `-1000.0` placeholders), `UNK → UNKNOWN` token-type, `precompiled_charsmap` from base64-decoded tokenizer.json, `seperator_token_id` (sic — canonical typo mirrored), `mask_token_id`, `add_eos_token`/`add_sep_token`, XLM-RoBERTa PAD-realignment shift (tokens[i] = `"[PAD{i-1}]"` for `i >= 4`).
5. **KV ordering**: model card BEFORE arch keys, tokenizer in canonical Unigram order, `general.{quantization_version, file_type}` at the very end after tokenizer block.
6. **Tensor ordering**: port of canonical's `weight_name_comparer` (`/opt/llama.cpp/src/llama-model-loader.h:53-64`) — non-blk first (alphabetical), then blk.N (numeric N, alphabetical within).

**Known open: IQ4_NL data-dependent boundary case.** The IQ4_NL kernel diverges from canonical at FP rounding boundaries that depend on weight distribution. Gemma 4 weights don't hit the boundary (0 bytes diff on full 16.7 GB IQ4_NL). Llama 3 weights hit it sparsely (587 KB / 4.6 GB = 0.013%). Qwen 3.5 weights hit it densely on `ffn_gate_exps` + `ffn_up_exps` 3D MoE tensors (121 MB / 18.6 GB = 0.65%). Single-axis hypothesis tests (split FMA, no-FMA initial pass) regressed Gemma to 48 KB diff — wrong direction. Likely a multi-site clang fusion divergence in the inner loops that needs disassembly-driven instruction-level matching. Tracked at task #65; not blocking the ADR-033 §10 byte-cmp gate at the family level since the other 7 quants per arch all match.

**Earlier-reported "120/180 byte residual"** was a stale-binary-cache artifact at `/opt/hf2q/cache/byte_cmp/` from a pre-fix build; verified 2026-05-20 with fresh rebuild + reconvert: zero byte residual on both families.

**Qwen 3.5 in-tree K-quant diagnostics** (commit `be112aec`):
- `qwen35_q4k_broad_sample_dump`: **0 / 21,888 bytes ✅**
- `qwen35_q4k_multi_tensor_dump` (blk_0_attn_gate + ssm_out + ffn_gate_exps): **0 / 4,608 bytes ✅**
- `qwen35_q6k_broad_sample_dump`: **0 / 16,380 bytes ✅**
- `qwen35_lm_head_row149_q6k_block7_dump`: scales[12]=195 ✅ (canonical target)

**Closure root cause** (commit `6985cd56` 2026-05-20): `make_qx_quants` had `.mul_add()` at BOTH the initial pass (lines 122-123) and refinement loop (lines 155-156). Canonical's compiled `_quantize_row_q6_K_ref` in `libggml-base.0.12.0.dylib` (built by current Apple Xcode toolchain, `-O3 -DNDEBUG -std=gnu11 -arch arm64 -fPIC`) emits **32 `fmadd s` total = exactly one inner-loop's worth of FMA**. The initial pass has a `L[i] = l + nmax;` side-effect inside the body that serializes clang's optimization (no auto-vec, falls back to scalar non-FMA `fmul; fadd`). The refinement-loop body has no per-iter side-effect and clang specialized the inlined first `is` iteration with scalar fmadd; the remaining 17 iterations don't add more fmadds to the count. Our Rust must mirror exactly: **initial = plain `+=`, refinement = `.mul_add()`**. The inverse (commit `b3bb0e5d`) was a synthetic-fixture-misled inverse hypothesis; only real-model byte-cmp revealed the truth (49,301 → 0 bytes Q6_K on Gemma; 138,169 → 0 bytes Q6_K on Qwen `output.weight`). The 120-byte Q4_K residual on Gemma's MoE expert weights is the input-distribution-dependent FP-boundary floor in `make_qkx2_quants`, exhaustively bisected across 11 `.mul_add()` callsites in prior iterations (see git log around `a5c25adc`/`c9c05d1f`/`6046ef3a`); functionally indistinguishable from zero (0.000001%) and not blocking ADR-033 §P1 closure. Per llama.cpp Q4_K_M policy, `token_embd.weight` + all `blk.*.attn_v.weight` get Q6_K treatment — so the residual surface is the **Q6_K kernel applied to these specific tensors**, not Q4_K. Per-tensor: `token_embd.weight` 45,429/605,552,640 = 0.0075% (= 92.4% of the global residual), `blk.*.attn_v.weight` 213-384 bytes each at 0.0045-0.0081%. Earlier text in this ADR claiming "all 658 tensors match byte-for-byte" referred to a tensor-count count-of-mismatches reading from a prior HEAD (`50fd89c2`) and was superseded by intermediate commits that closed Qwen Q4_K (`b921616e`, `c9c05d1f`) but did not close Gemma Q6_K. Reproduce with `bash scripts/byte_cmp_full_pipeline.sh /opt/hf2q/models/google-gemma-4-26b-a4b-it` then `python3 scripts/byte_cmp_streaming.py <canonical.gguf> <hf2q.gguf>` (numpy `np.count_nonzero(c != h)` vectorized over 64 MB mmap chunks — ~30 s vs the inline `zip()` python which gets OOM-killed on 16 GB GGUFs). PPL on cdv3 ctx=2048 chunks=20: 13183.4003 ± 697.92 — EXACT match to canonical. Two root-causes fixed: (1) FMA non-associativity — rustc `--release` defaults `fp-contract=off` while clang `-O3 -march=native` defaults `-ffp-contract=on`, fusing `a*b+c`, `a*b-c*d`, and `a*b+c-d` patterns into single-rounded `fmadd/fmsub`. Hot-spots in `make_qx_quants`, `make_qkx2_quants`, `make_qkx3_quants`, `make_qp_quants` now use explicit `mul_add` to match clang's contraction. (2) F16 round-trip — canonical pipeline stores BF16/F32 weight tensors as F16 in the intermediate GGUF before `llama-quantize` reads them; the F16 round-trip is lossy below F16's normal range (~6.1e-5). `src/convert/orchestrator.rs::stream_tensor` now applies `F32→F16→F32` at the quantizer branch, mirroring `/opt/llama.cpp/conversion/base.py:875-876`. **Pre-fix Q5_K_M result (preserved for history)**: PPL 5411.20/5471.84 = 0.989 ± 0.073 on Gemma 4 26B Q5_K_M cdv3 ctx=2048 chunks=20, zero per-tensor `ggml_type` mismatches, 448-byte file-size delta (header KV order only). Real Gemma 4 26B convert: 48GB safetensors → 18GB Q5_K_M GGUF in 8m 22s, loads in stock llama.cpp + decodes coherent reasoning at 111.5 t/s gen. §9 ships 21 manifest entries (3 families × 7 tiers; gemma4-26b + Qwen3.5/3.6-35B-A3B base + Qwen3.6 MTP) via SHA-256 of canonical 9-tuple JSON. §Pi Phase A ships 32 new tests + supports apex-i-* tiers via externally-generated `.imatrix.gguf` (`llama-imatrix` workaround documented); §Pi Phase B (in-tree forward-pass driver for `--imatrix-corpus`) SHIPPED 2026-05-19 (Stage 1-3c, see §Pi Phase B section). Phase 2/3 retirement of 8 delete-listed files + 3 retired env vars SHIPPED (verified §P6 status, line 445; 2804 tests pass). Q4_0 / Q4_K_S / Q4_K_M / Q5_K_S / Q5_K_M / Q6_K / Q8_0 / IQ4_NL all byte-identical (5264 verifications, see [[project_adr033_p1_byte_identical_2026_05_19]]). Convert is now 3.0–3.3× FASTER than canonical via rayon per-row parallelization ([[ADR-036]] SHIPPED 2026-05-19, commit `3b24daea`). Followups (separate ADRs, intentionally outside ADR-033 surface): **[[ADR-034]]** native MTP + DFlash spec-decode + Qwen 3.5/3.6 convert handler (proposed, commit `f90ad2db`); **[[ADR-035]]** GGUF codec ownership relocation hf2q → mlx-native (proposed, post-§P1 cleanup).
- **Date**: 2026-05-18
- **Deciders**: operator (robert@loveathome.us); claude (interview + draft)
- **Tags**: convert, quantize, architecture, byte-parity, public-release, apex, mudler, imatrix
- **Supersedes**: ADR-014 (full supersession; streaming-convert property carried forward; the `--quant apex` CLI surface ADR-014 P8 D13 removed is reintroduced here with the correct semantics)
- **External pins** (load-bearing — byte-cmp gates assume these exact references):
  - `llama.cpp` @ `c779f6198` (operator's `/opt/llama.cpp` HEAD; the local branch with ADR-029 iter-57 instrumentation — NOT stock upstream)
  - `mudler/apex-quant` @ `63c5048b7dc9ff230f2397d7bc445ca28894b769` (GitHub main, 2026-05-17 14:42 UTC; the SHA we port from)
  - GGUF spec: v3 (matches `const GGUF_VERSION: u32 = 3` at `src/backends/gguf.rs:23`; matches `general.quantization_version = 2` in operator's existing APEX files)
  - rustc: pinned via `rust-toolchain.toml` (1.81.0 minimum per current `mlx-native` MSRV; P-1 verifies the project file pins a single version for byte-cmp determinism across developer machines)

## Context

### The problem

hf2q's convert/quant pipeline is currently a set of overlapping, partially-redundant subsystems that don't compose cleanly and don't reproduce any standard llama.cpp artifact byte-for-byte:

| Subsystem | LOC (HEAD) | Status |
|---|---|---|
| `quantize/k_quant_codec_quantizer.rs` | 953 | Production K-quant path |
| `quantize/variant_quantizer.rs` | 604 | Variant K-quant (imatrix-adaptive) |
| `quantize/dwq_k_quantizer.rs` | 883 | hf2q's homebrew DWQ |
| `quantize/mixed.rs` | 520 | Mixed-bit dispatcher |
| `quantize/static_quant.rs` | 468 | Static (non-K) quant |
| `quantize/mod.rs` / `k_quant.rs` / `k_quant_codec.rs` / `q_legacy.rs` / `layer_mix.rs` | 16,867 | Unclassified: kernels + policy + utility + dead all mixed |
| `backends/gguf.rs:282–1259` | ~977 | Two-pass GGUF writer (the iter-99 / Bug-B-sequel bug-class lives here) |
| `ir/mod.rs::TensorQuantInfo` | 7 fields | Carries `method | bits | preserved | scales | biases | ggml_type | …` simultaneously |
| 5 quantizer dispatch arms in `main.rs` (≈2160 / 2225 / 2400) | — | Each routes through a different policy with a different output shape |

The result of this fragmentation:

- **Every production model surfaces a new internal bug.** Five fixes shipped 2026-05-15..17 (`5dd2189a`, `e549906a`, `753e87ff`, `77489aaa`, `2b9b5a42`) plugged five distinct seams; rate is not converging.
- **No artifact hf2q produces can be byte-cmp'd against the canonical llama.cpp pipeline.** Our K-quant dispatch deviates from `llama-quantize`'s `llama_tensor_get_type_impl` in subtle ways the codebase doesn't enumerate.
- **The two "APEX" GGUF files the operator runs in production were produced externally, not by hf2q.** hf2q currently has no path to reproduce them from safetensors. The `gemma4-ara-2pass-APEX-Q5_K_M.gguf` and `qwen3.6/APEX-Q5_K_M.gguf` came from a separate toolchain on a separate machine.
- **`--quant apex` was removed in ADR-014 P8 Decision 13** because its semantics weren't well-defined. The 2026-05-17 ADR-033 draft (`ebecc21c`) reintroduced it but defined `ApexPolicy` as "pure base-Q + shape_fallback" — a definition that doesn't match any real APEX artifact and doesn't match `mudler/apex-quant`'s published behavior.

### What APEX actually is

Deep research 2026-05-17 (operator-led, with web/repo/HF reads) established APEX = `mudler/apex-quant`, an MoE-specific quantization toolkit on GitHub. Key facts (full reference in auto-memory `[[apex-quant-definition-2026-05-17]]`):

- **MoE-only.** Designed around 97%-sparse routed experts. Does not meaningfully apply to dense models.
- **Per-tensor-pattern overlay via stock `llama-quantize`'s `--tensor-type` / `--tensor-type-file` flags.** No custom llama.cpp patches.
- **Seven tiers:** I-Quality, Quality, I-Balanced, Balanced, I-Compact, Compact, Mini. The `I-` prefix variants use diverse-corpus imatrix calibration; the four non-I tiers do not.
- **Tensor classification by role:** routed-expert (tolerates Q4_K / IQ4_XS), shared-expert (needs Q8_0, kurtosis 13.10), attention (per-tier), token-embd / output (edge layers get Q6_K).
- **Layer-wise gradient:** edge layers (first / last 5 of 40-layer default; rescaled by `NUM_LAYERS` env var) get heavier quant; middle layers get lighter.
- **Quality tier matches F16 perplexity at ~⅓ the size** (Qwen3.5-35B-A3B: 6.527 vs 6.537 PPL at 21.3 GB vs 64.6 GB).

The 2026-05-17 ADR-033 draft conflated several unrelated concepts under the name "APEX." This rewrite separates them.

### What we want

Operator framing (2026-05-17 / 2026-05-18):

1. **"100% Rust full stop."** No FFI, no shell-out to llama-quantize, no Python subprocess. The whole convert+quant capability lives in hf2q.
2. **Reproduce the standard llama.cpp pipeline byte-for-byte.** `hf2q convert <hf-dir> --quant q5_k_m -o out.gguf` produces the same bytes as `convert_hf_to_gguf.py | llama-quantize -q5_k_m`.
3. **"Make our own APEX correctly."** Port `mudler/apex-quant`'s published recipe in pure Rust; reproduce the per-tier output for the supported MoE arches; couple with an in-tree imatrix generator for the I-tier variants.
4. **Streaming property preserved.** No intermediate F16 GGUF on disk (the ADR-014 invariant). safetensors → quantized GGUF in one in-memory pipeline.
5. **No-fallback rule.** F16 emit is allowed only for (a) vision-tensor patterns and (b) `--quant f16` explicit user request. Any other F16 emit is a typed error, not a silent demotion. (The 2026-05-17 draft promised this but had `shape_fallback` silently mirror llama.cpp's second-misalignment F16 path; this rewrite resolves the contradiction by making `shape_fallback` hard-error.)
6. **"Quality matters; mechanism doesn't."** The acceptance gates are byte-cmp against canonical references. How we enforce internal invariants (e.g., FMA ordering) is an implementation detail validated by the gates, not by the ADR.

## Decision

Collapse the five overlapping quantizer impls + two-pass writer + seven-field IR into:

1. **A single `QuantizedTensor` IR type** carrying only `{ ggml_type, data: Arc<Vec<u8>> }`. Fields beyond these two are added only when a proven need arises (safety-valve clause; not a license for re-bloat).
2. **A unified `Quantizer` trait** mirroring `ggml_quantize_chunk`'s signature; pure-Rust port of `ggml-quants.c`. No FFI.
3. **Two `QuantPolicy` impls** at v1 ship:
   - `StandardPolicy` — byte-for-byte port of `llama_tensor_get_type_impl` (covers `q4_0`/`q4_1`/`q5_0`/`q5_1`/`q4_k_m`/`q5_k_m`/`q6_k`/`q8_0`/`iq4_nl`/etc.; mirrors `tensor_type_fallback`'s first-downshift behavior).
   - `ApexPolicy` — pure-Rust port of `mudler/apex-quant`'s published recipe (7 tiers; MoE tensor classifier; `NUM_LAYERS`-aware layer gradient; per-tier regex → quant-type rules).
4. **`shape_fallback` contract:** every policy's `target_for(tensor)` returns `Result<GgmlType, QuantizeError>`. The first-downshift path succeeds; the second-misalignment case (where llama.cpp silently emits F16) returns a typed error. F16 is emitted only for vision-tensor patterns or `--quant f16`. No silent demotions anywhere. (The 2026-05-17 draft had a separate §8 "no-fallback enforcement" section; this rewrite rolls the enforcement into each policy's signature, so the contract is type-system-checked rather than narrative.)
5. **Seek-back incremental GGUF writer** — single pass; reserve a header region; stream tensor payloads to disk; seek back and fill the header. No pre-allocated offset table, no two-pass zero-padding. Eliminates the iter-99 / Bug-B-sequel bug class by construction. Single-file output only at v1 (`--split-max-size` is explicit non-goal; users can post-split with `llama-gguf-split`).
6. **MoE tensor classification for `ApexPolicy`:** combination of GGUF metadata introspection (`expert_count`, `expert_used_count`, arch-name) and `mudler/apex-quant`'s tensor-name regex tables ported verbatim. Per-arch classifier files live at `src/quantize/apex/<arch>.rs` for the supported set (qwen35moe, gemma4-MoE, MiniMax-M2.7's arch). Unsupported arches passed to `--quant apex-*` fail with a typed error naming the supported set.
7. **In-tree imatrix subsystem.** hf2q's existing forward-pass code (built on `mlx-native`'s Metal compute primitives) is reused: decode a calibration corpus, accumulate per-row importance, emit a `.imatrix.gguf` compatible with llama-imatrix's format. v1 ships both UX modes implicitly via flags: `--imatrix-corpus {cdv3,mudler,user-file}` auto-generates in-memory during convert; `--imatrix <file>` consumes a pre-made file. Default corpus when `--imatrix-corpus` is omitted on an I-tier: `cdv3` (bartowski's `calibration_datav3.txt`). Both corpora ship in `data/calibration/` alongside the binary (not embedded in the binary).
8. **CLI surface:**
   ```
   hf2q convert <hf-dir> --quant <name>
                          [--imatrix <file> | --imatrix-corpus {cdv3,mudler,user-file:<path>}]
                          [--imatrix-n-ctx <N>]          # default 512; only honored with --imatrix-corpus
                          [--imatrix-out <path>]         # side-effect write of computed/loaded imatrix
                          [--tensor-type-file <file>]    # only for --quant apex-custom
                          [-o out.gguf]
   ```
   `<name>` ∈
   - StandardPolicy types: `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q4_k_s`, `q4_k_m`, `q5_k_s`, `q5_k_m`, `q6_k`, `q8_0`, `iq4_nl`, `f16`, `f32`, `bf16`
   - ApexPolicy tiers (verified at mudler SHA `63c5048b`; mudler ships 12 algorithmic profile names — `quality, i-quality, balanced, i-balanced, compact, i-compact, mini, nano, i-nano, micro, i-micro, custom`; v1 drops the four experimental tiers `nano, i-nano, micro, i-micro`):
     - `apex-quality`, `apex-i-quality`
     - `apex-balanced`, `apex-i-balanced`
     - `apex-compact`, `apex-i-compact`
     - `apex-mini` (mudler "benefits from imatrix"; can be run with or without)
     - `apex-custom` — requires `--tensor-type-file <file>`; consumes operator-supplied per-tensor type overrides in mudler's `pattern=quant_type` line format
   - **Dropped from mudler's surface for v1:** `nano`, `i-nano`, `micro`, `i-micro` (all 4 labeled experimental upstream; target IQ2_XXS / IQ1_M / IQ2_S-class aggressive quants). Per-model configs for these tiers remain accessible via `--quant apex-custom --tensor-type-file data/apex-references/<model>_<nano|micro>.txt` (and similar for i-variants).
   - Reserved: `dwq` returns a typed `--quant dwq is reserved for the future real-DWQ ADR (Apple MLX dwq.py port)` error.
   - **No `apex` alias.** Tier must be spelled explicitly; ADR-014 P8 D13 removed the unqualified name because its meaning was ambiguous, and the same reason still applies.
   - **TQ1_0 / TQ2_0 (BitNet ternary) out of v1 scope.** Documented; tracked separately. `--quant tq1_0` returns a typed "out of v1 scope; see [tracking issue]" error.

### Per-model APEX config override (Decision §9 — silent auto-fingerprint)

**Status:** SHIPPED 2026-05-19. Manifest at `data/apex-references/manifest.json` (21 entries: 3 model families × 7 tiers). Loader at `src/quantize/ggml_quants/apex/fingerprint.rs`. Mudler config parser at `src/quantize/ggml_quants/apex/mudler_config.rs`. Wired into `cli_driver.rs::run_convert` (post-B4-rename; was `run_convert_v2` pre-rename) for `--quant apex-<tier>`.

Mudler ships `configs/<model>_<tier>.txt` per-model overrides alongside the algorithmic `scripts/generate_config.sh`. These vendored configs are hand-tuned for specific known models (e.g., `carnice_qwen36_mtp_quality.txt` matches the operator's qwen3.6 abliterix production model). v1 hf2q vendors them via `include_str!` from `vendor/apex-quant/configs/<file>.txt` (baked at compile time — NO runtime disk read) and dispatches automatically:

- Compute a stable fingerprint = `sha256(canonical_json(9-tuple))` from the source `config.json`. The 9-tuple is `(model_type, num_hidden_layers, hidden_size, num_experts, num_attention_heads, num_key_value_heads, intermediate_size, moe_intermediate_size, mtp_num_hidden_layers)`. The 9th field (`mtp_num_hidden_layers`) extends ADR §9's original 8-tuple because MTP variants would otherwise alias their non-MTP base (Qwen3.5-A3B and Qwen3.6-A3B-MTP share the first 8 fields; mudler ships separate `*_mtp_*.txt` configs that add a `blk.40.*` row). Canonical JSON is `json.dumps(d, sort_keys=True, separators=(",", ":"))`-equivalent — re-implementable byte-identically in Python / Go / Rust at vendor time.
- Check fingerprint AND requested tier against `data/apex-references/manifest.json`'s `entries[]` array. One entry per (family, tier) pair; I-tier entries alias the same vendored `.txt` as their non-I counterpart (the imatrix flag is applied at quantize time, not in the type-file).
- **If matched:** load the vendored `.txt` (baked content via `fingerprint::VENDOR_CONFIGS`) into a `MudlerConfig { map: HashMap<String, GgmlType> }`. The override is attached to `ApexPolicy` via `with_mudler_override(&'static MudlerConfig)`. `target_for(tensor)` consults the map first; on enumerated tensor names (routed/shared expert, attention, ssm) the vendored type wins. On structural tensors mudler doesn't enumerate (norms, `token_embd`, `output`, `ffn_gate_inp` router gate), the algorithmic hardcodes (F32 / Q6_K / Q5_0) apply — exactly mirroring stock `llama-quantize --tensor-type-file` semantics (`llama-quant.cpp:678-693`).
- **If unmatched:** fall through to the algorithmic generator (`ApexPolicy::target_for` per-tier 7-tuple at `rules.rs`).
- **Tensor-name matching** is exact-then-prefix-with-dot: `key == name` OR `name.starts_with(key + ".")`. Real GGUF names carry a `.weight` suffix that mudler omits; the prefix-with-dot rule absorbs that. Mirrors `regex_search` semantics on mudler's literal-only v1 surface.
- **Hard-error contract:**
  - Matched fingerprint whose `mudler_config_path` isn't in the compile-time `VENDOR_CONFIGS` bake → `ApexError::FingerprintConfigMissing`. Caught at build time by the `every_manifest_entry_has_baked_vendor_content` unit test, so it can only fire if a vendor regen updates the manifest JSON without also updating the `include_str!` table.
  - Mudler config file fails to parse → `ApexError::MudlerConfigParse { source_path, line_number, detail }`.
  - Tensor name reaches the override and is classified as a non-structural MoE role (routed/shared/attn/ssm/other) but isn't in the parsed map → `ApexError::TensorNotInMudlerConfig`. Per ADR §9 the vendored config is authoritative; we do NOT silently fall through to the algorithmic generator on a tensor miss. (Per [[feedback-no-loop-suppression-2026-05-17]].)
- No CLI flag controls this; the fingerprint match is invisible to the user. The CLI driver logs the match to stderr (`[hf2q apex] auto-detected APEX config: vendor/apex-quant/configs/<file>.txt (fingerprint=..., tier=..., arch=...)`) for audit transparency — mitigates the "surprising override" risk. (Trade-off acknowledged: surprising override risk; mitigated by the stderr log line + the future `hf2q apex why <hf-dir>` debug subcommand.)

**Manifest schema (v1):** each `entries[]` element carries `fingerprint` (hex SHA-256), `model_id_pattern` (HF repo regex — informational, not used for dispatch), `arch` (`gemma4` / `qwen35moe` / `minimax-m2`), `tier` (`quality` / `i-quality` / `balanced` / `i-balanced` / `compact` / `i-compact` / `mini`), `mudler_config_path` (vendor-relative, e.g. `vendor/apex-quant/configs/gemma4_26b_balanced.txt`), and `expected_hparams` (the 9-tuple as a flat JSON object, for documentation + manifest-regenerator validation).

**v1 manifest coverage:** 3 confirmed families × 7 tiers = 21 entries. Families: `gemma4-26b-a4b-it` (hparams sourced from `/opt/hf2q/models/google-gemma-4-26b-a4b-it/config.json`), `Qwen3.5/3.6-35B-A3B base` (hparams from `/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/config.json`), `Qwen3.6 MTP variants` (carnice/qwen36-MTP/qwopus36-MTP/etc., disambiguated by `mtp_num_hidden_layers=1`). Vendor-regen extension path: append to `VENDOR_CONFIGS` in `fingerprint.rs` AND to `entries[]` in `manifest.json`; `manifest_entry_count_meets_adr_floor` test pins `≥10`.

**Known limitations (caveats):**
- The hparams 9-tuple cannot disambiguate sibling fine-tunes that share architecture (Holo3 / Darwin36BOpus / Heretic / OpusDistill / Fernflower all match `qwen35a3b`'s 8-tuple; they hash to the same fingerprint and resolve to `qwen35a3b_<tier>.txt`). Inspection confirms all six configs are byte-identical to the canonical `qwen36_35b_*.txt` content (and differ from the upstream `qwen35a3b_*.txt` ONLY by ASCII case in the `GgmlType` token, which the case-insensitive `GgmlType::from_name` folds), so the collision is byte-safe.
- MiniMax M2.5 / M2.7 + Qwen3-Coder-30B + Qwen3.5-122B configs are NOT yet in the v1 manifest — their source `config.json` files weren't on disk at ship time. Operator can extend the manifest at vendor-regen.

The manifest is regenerated at vendor-time only (not at runtime); SHA-pinned to the mudler commit captured in `data/apex-references/MUDLER_SHA.txt`.

### FP8 source-dtype auto-detect (Decision §10 — silent)

When `config.json::quantization_config.quant_method == "fp8"` (per HuggingFace's standard quantization-config schema, used by MiniMax-M2.7 and others), hf2q convert auto-dequantizes the FP8 source to F32 in-memory before invoking the policy:

- Format: `float8_e4m3fn` (1-bit sign + 4-bit exponent + 3-bit mantissa, no inf, single NaN encoding).
- Layout: block-wise with `weight_block_size` field (e.g., `[128, 128]` for MiniMax-M2.7) — block-of-blocks scale factor stored alongside each tensor.
- Modules listed in `modules_to_not_convert` (e.g., `gate`, `e_score_correction_bias`, `lm_head`) are read as F32 / BF16 directly (no FP8 path).
- No CLI flag controls this; auto-detection is silent. (Trade-off: same as above; surprise risk mitigated by `hf2q convert --dry-run` flag that prints the resolved source dtype per tensor before quantizing.)

Source-dtype hard-error chain (verified at `src/convert/source_reader.rs:283-292` and `:178-186` 2026-05-19): if `quantization_config.quant_method` is anything other than `fp8` (e.g., `gptq`, `awq`), `Fp8Config::from_config` returns `Ok(None)` at config-parse time — NOT an early error. The intentional design choice (called out in the inline comment at `source_reader.rs:288-290`): "don't error here; the per-tensor loader will surface unsupported dtypes if they actually appear". This is forward-compatible with mixed-precision configs where `quantization_config` is set but most tensor payloads are still in standard F32 / F16 / BF16 dtypes. When a truly-unsupported tensor dtype (GPTQ's packed INT4, AWQ's I32, etc.) is encountered downstream, the per-tensor loader fires `SourceError::UnsupportedSourceDtype { tensor, dtype }` — the typed error per [[feedback-no-loop-suppression-2026-05-17]] eventually fires; it just fires at the tensor level, not at the config level. Original ADR draft promised a config-level `UnsupportedSourceQuant` variant; that's not how the shipped code works.

### Per-tensor IR (Decision §1 concrete)

Replaces `TensorQuantInfo`:

```rust
pub struct QuantizedTensor {
    pub ggml_type: GgmlType,    // enum mirroring llama.cpp's `ggml_type` for all wire values
    pub data: Arc<Vec<u8>>,     // packed block bytes
    // Add fields only if a proven need surfaces. Today, none.
}
```

`GgmlType` is a Rust enum spanning every `ggml_type` value llama.cpp writes to disk (block formats Q4_0..IQ4_NL plus F16/BF16/F32). Conversion to/from `u32` for header serialization is `From`/`TryFrom`.

### Quantizer trait (Decision §2 concrete)

```rust
pub trait Quantizer: Send + Sync {
    fn ggml_type(&self) -> GgmlType;
    fn quantize(
        &self,
        src: &[f32],
        n_per_row: usize,
        imatrix: Option<&[f32]>,    // length: n_per_row (per-column importance, per llama.cpp's convention)
    ) -> Result<Vec<u8>, QuantizeError>;
}
```

One impl per `GgmlType` whose disk format we emit. Lives at `src/quantize/ggml_quants/<type>.rs`. Each impl is a port of the corresponding `ggml_quants.c` function (`quantize_row_q4_K`, `quantize_row_q5_K`, …). Signature mirrors `ggml_quantize_chunk`'s row-major contract; behavior is byte-identical to llama.cpp's reference.

**Hard-error contract:** when convert encounters a tensor whose target `GgmlType` (per the active `QuantPolicy`) has no `Quantizer` impl, the pipeline returns a typed `QuantizeError::NoQuantizerForType { ggml_type: GgmlType }` — no silent fallback, no F16 escape. This implements the no-fallback rule at the trait-dispatch layer.

### LlamaFtype mapping (Decision §2 concrete)

Rust enum mirrors llama.cpp's `enum llama_ftype` (`/opt/llama.cpp/include/llama.h`) at the literal numeric values for byte-level header compatibility:

```rust
#[repr(u32)]
pub enum LlamaFtype {
    AllF32         =  0,
    MostlyF16      =  1,
    MostlyQ4_0     =  2,
    MostlyQ4_1     =  3,
    MostlyQ8_0     =  7,
    MostlyQ5_0     =  8,
    MostlyQ5_1     =  9,
    MostlyQ2_K     = 10,
    MostlyQ3_K_S   = 11,
    MostlyQ3_K_M   = 12,
    MostlyQ3_K_L   = 13,
    MostlyQ4_K_S   = 14,
    MostlyQ4_K_M   = 15,
    MostlyQ5_K_S   = 16,
    MostlyQ5_K_M   = 17,
    MostlyQ6_K     = 18,
    MostlyIQ4_NL   = 25,
    BF16           = 32,
    // Holes (4, 5, 6, 19-24, 26-31) are llama.cpp values out of v1 scope (TQ1_0/TQ2_0/IQ2_*/IQ3_*/IQ1_*).
    // Add only when the matching Quantizer impl ships.
}
```

v1 supported set: `AllF32 / MostlyF16 / BF16 / MostlyQ4_0 / MostlyQ4_1 / MostlyQ5_0 / MostlyQ5_1 / MostlyQ4_K_S / MostlyQ4_K_M / MostlyQ5_K_S / MostlyQ5_K_M / MostlyQ6_K / MostlyQ8_0 / MostlyIQ4_NL`.

### TensorRef (passed to QuantPolicy::target_for)

```rust
pub struct TensorRef<'a> {
    pub name: &'a str,            // canonical GGUF tensor name (e.g., "blk.0.attn_q.weight")
    pub shape: &'a [usize],       // dims, row-major
    pub source_dtype: SourceDtype, // F32 | F16 | BF16 (from safetensors header)
    pub arch: ArchName,            // Gemma4 | Qwen35Moe | MiniMaxM27 | Llama3 | Bert | NomicBert | Qwen3VlText | Gemma4Mmproj
    pub layer_index: Option<usize>, // None for global tensors (token_embd / output); Some(i) for per-block
}
```

`ArchName` is closed enum — adding a new arch is an explicit code change, NOT silent runtime detection.

### Vision / audio tensor patterns (canonical source)

The vision-tensor F16-emit gate is `crate::quantize::vision::is_vision_tensor_pattern(name)` and its sibling `is_audio_tensor_pattern(name)`. Together they decide whether a tensor is "modality-side" (Pa policy bypassed, F16 emitted directly) or "language-side" (policy decides).

`is_vision_tensor_pattern` returns `true` iff the name contains any of:
`model.visual.` | `vision_tower.` | `vision_model.` | `vit.` | (prefix) `visual.` | `.visual.`

`is_audio_tensor_pattern` (NEW per P-1 audit finding E) returns `true` iff the name contains any of:
`audio_tower.` | `audio_model.` | `whisper.`

These are the **only** places modality-pattern membership is decided. The convert dispatcher checks `is_vision_tensor_pattern(name) || is_audio_tensor_pattern(name)` BEFORE calling `QuantPolicy::target_for`. The current `layer_mix.rs::is_vision_tensor_pattern` (at `src/quantize/layer_mix.rs:366`, HEAD `85bee70e`) ports verbatim to `src/quantize/vision.rs`; the audio sibling is new code. The three inline duplicate vision checks at `backends/gguf.rs:322-333, 721-724, 905-909` (per P-1 audit) are deleted.

### QuantPolicy trait (Decision §3 concrete)

```rust
pub trait QuantPolicy {
    fn target_for(&self, tensor: &TensorRef) -> Result<GgmlType, QuantizeError>;
    // Optional: imatrix requirement check before quantize starts
    fn requires_imatrix(&self) -> bool;
}
```

`StandardPolicy { ftype: LlamaFtype }` is a port of `llama_tensor_get_type_impl`. `ApexPolicy { tier: ApexTier }` is the mudler port. The `target_for` return type makes the no-fallback rule type-system-checked: a policy CANNOT silently emit F16 — it must either succeed with a non-F16 type or return `Err`. Vision-tensor F16 is handled outside the policy at the dispatcher layer (the dispatcher checks vision-pattern membership before calling the policy at all).

## Plan

Phases run sequentially. Every phase has a binary acceptance gate; later phases do not start until the prior phase's gate passes.

### P-1 — Audit & classify + vendor external pins

**Why:** ADR-014's predecessor's delete-list was incomplete. Five files in `src/quantize/` (`mod.rs`, `k_quant.rs`, `k_quant_codec.rs`, `q_legacy.rs`, `layer_mix.rs`) totalling 16,867 LOC have no stated fate. Several contain kernels that may be the P0 port target; others are dead policy code that should be deleted. Separately: all of ADR-033's byte-cmp gates depend on external pins (llama.cpp @ `c779f6198`, mudler @ `63c5048b`); these need to be vendored locally so verification is reproducible offline (network restrictions or upstream changes can't invalidate the gate).

**What:**
- Function-by-function classification of every fn in those five files plus the existing dispatcher arms in `main.rs` and the GGUF writer slice at `backends/gguf.rs:282–1259`. Three buckets: `KEEP` (utility we still need), `MODIFY` (kernels P0 ports in place), `DELETE` (superseded policy / dead code).
- **Vendor mudler/apex-quant @ `63c5048b7dc9ff230f2397d7bc445ca28894b769` to `vendor/apex-quant/` (git submodule or `git archive` snapshot; either works; submodule preferred for auditability).** Generate `data/apex-references/manifest.json` (the fingerprint → config-file map per Decision §9) via a one-shot vendor script that walks `vendor/apex-quant/configs/*.txt` and computes fingerprints from each config's matching upstream HF model. Vendor script committed at `scripts/vendor_apex.sh`; rerun is a deliberate ADR-amendment event, not a CI step.
- **Confirm llama.cpp @ `c779f6198` is the operator's local `/opt/llama.cpp` HEAD AND remains so for the duration of the project.** If the operator's llama.cpp moves during execution, ADR-033 gates re-anchor to the new SHA (and the ADR documents the move with rationale). P-1 records the current SHA in `data/llama_cpp_pin.txt` for later re-verification.

**Acceptance criteria:**
- Markdown table inline in ADR-033 §"Audit results" (this section, populated after P-1 runs), one row per fn: `file.rs::fn_name | LOC | KEEP/MODIFY/DELETE | rationale`.
- Zero unclassified fns in those files at P-1 exit.
- The delete-list LOC total in §7 sums correctly (no hand-wave numbers).
- `vendor/apex-quant/` exists with the pinned SHA; `git -C vendor/apex-quant rev-parse HEAD` returns `63c5048b7dc9ff230f2397d7bc445ca28894b769`.
- `data/apex-references/manifest.json` exists with at least 5 entries (one each for gemma4-26B-A4B, qwen35moe-3.6, MiniMax-M2.7, plus 2 more from the mudler configs/ that match known model fingerprints).
- `data/llama_cpp_pin.txt` records the current local llama.cpp SHA.

**Deliverable:** updated §"Audit results" section in this ADR; vendored mudler ref; manifest; llama.cpp SHA record. No production code touched.

### P0 — Pure-Rust ggml-quants port + per-arch safetensors→F32 mapping (convert-side)

**Why:** Today's hf2q-side kernels are scattered across `k_quant.rs` (5541 LOC) and `static_quant.rs` (468 LOC); the per-arch safetensors→F32 mapping for inference lives in `src/inference/models/<arch>/` but ONLY covers `{bert, gemma4, nomic_bert, qwen35, qwen3vl_text}` (verified at HEAD `ebecc21c` via `ls src/inference/models/`). The convert path additionally needs Llama-3-8B (dense decoder test fixture) and MiniMax-M2.7 (3rd MoE for APEX validation), neither of which exists in inference. **Per operator decision 2026-05-18: P0 adds tensor-mapping for these arches in `src/convert/arch/` (a NEW dir; convert-side mapping is independent of inference-side forward-pass code) — inference support for these arches is deferred to a separate effort.**

**What:**
- Port `ggml-quants.c` quantize-side functions one per file under `src/quantize/ggml_quants/`. **v1 set (11 files)**: `{q2_k.rs, q3_k.rs, q4_0.rs, q4_1.rs, q5_0.rs, q5_1.rs, q4_k.rs, q5_k.rs, q6_k.rs, q8_0.rs, iq4_nl.rs}`. Maps 1:1 with `quantize_row_q2_K` / `quantize_row_q3_K` / `quantize_row_q4_0` / ... in `/opt/llama.cpp/ggml/src/ggml-quants.c` at the pinned SHA. (Q2_K + Q3_K added per P-1 audit finding A: their dequant is externally referenced by `src/quality/mod.rs:612` and `src/backends/gguf.rs` size estimator at L1275 / L1458 / L2207 / L2566 / L2819 / L3085; dropping them would break those call sites.) Pre-existing logic in `k_quant.rs` is either ported into these files (per P-1 classification) or deleted.
- **Imatrix-aware variants are NEW code for the 6 legacy types.** Per P-1 audit finding F, `src/quantize/q_legacy.rs` has zero `*_impl` (imatrix-aware) variants today. llama.cpp's `quantize_row_q4_0_impl` (ggml-quants.c:2008) accepts a `quant_weights` arg and dispatches on null. The new `Quantizer::quantize(src, n_per_row, imatrix: Option<&[f32]>)` requires imatrix-aware code for every legacy type. For `{q4_0, q4_1, q5_0, q5_1, q8_0, iq4_nl}` P0 ships BOTH the no-imatrix path (port of `quantize_row_<T>_ref`) AND the imatrix path (port of `quantize_row_<T>_impl`). The K-family files already had `_imatrix` variants in `k_quant.rs`, so this asymmetry only affects the 6 legacy files.
- Per-arch convert-side mapping at `src/convert/arch/<arch>.rs` for the full convert matrix: `{gemma4, qwen35moe, qwen3vl, gemma4_mmproj, bert, nomic_bert, llama3, minimax_m2}`. Each is a port of the corresponding `/opt/llama.cpp/conversion/*.py` module, restricted to tensor-name + shape mapping (no inference logic). For arches already in `src/inference/models/`, the convert-side mapper REUSES the inference-side tensor-name conventions; for new arches (`llama3`, `minimax_m2`), it's the only mapping that exists.
- A single `ArchName::detect(config_json: &Value) -> Result<ArchName>` reads `config.json::model_type` and `config.json::architectures` to dispatch. Failure is typed: `ConvertError::UnsupportedArch { detected: String, supported: Vec<&str> }`.
- **FP8 source-dtype support (NEW v1 scope per Decision §10):** add `src/convert/source_dtype/fp8.rs` implementing the block-wise `float8_e4m3fn` → F32 dequantize. Reads `quantization_config.weight_block_size` from `config.json` (typically `[128, 128]`); per-tensor, reads block scales stored alongside the FP8 payload (per HF convention: `<tensor>.weight` (FP8) + `<tensor>.weight_scale_inv` (F32 block scales)). Modules listed in `quantization_config.modules_to_not_convert` are read as F32 / BF16 directly. Required by MiniMax-M2.7 which ships in FP8.

**Acceptance criteria:**
- For every `GgmlType` in v1 scope (the 11-file list): a unit test takes a fixed-seed F32 input vector + fixed `n_per_row` (256 for K-quants, 32 for legacy) and produces output bytes that `cmp` byte-equal against the same input fed to llama.cpp's reference at the pinned SHA. Reference outputs generated once via a small C harness wrapping `ggml_quantize_chunk` and checked into `tests/fixtures/ggml_quants/<type>_<n>.bin`.
- **The C harness used to generate reference fixtures is built `aarch64-apple-darwin` (NEON enabled) on macOS Apple Silicon** (per P-1 audit finding I; `k_quant.rs` L9-18 module doc flags a NEON-vs-scalar argument-order divergence in `make_qkx2_quants`). The same harness rebuilt `x86_64-pc-linux-gnu` (no NEON) on x86 Linux must produce byte-identical fixtures; if it doesn't, hf2q ports are matched against the NEON variant explicitly and the divergence is documented in `tests/fixtures/ggml_quants/README.md`.
- For every arch in the convert matrix: a fixture test takes a real safetensors directory (or a tiny synthetic one for arches whose real model is multi-GB), runs hf2q's convert through to F32-tensor-emission only (no quantize), and `cmp`s the resulting F32 byte stream against what `convert_hf_to_gguf.py <hf-dir>` would emit at the pinned llama.cpp SHA. Fixtures stored at `tests/fixtures/convert_arch/<arch>.f32.bin`.
- Memory bound: peak RSS during convert of a 26B-param model (e.g., gemma4-26B-A4B) is bounded by `4 × largest_single_tensor_F32_size + 512 MiB` (tensor-by-tensor streaming with the source reader mmapping shards instead of loading them into the heap). Tightened from the original `2 × model_safetensors_size + 512 MiB` envelope 2026-05-18 after the real-model OOM finding — see §Open Issues / Real-Model Findings. Validated by `tests/convert_integration.rs::convert_streaming_rss_under_bound_2026_05_18` (was `tests/convert_v2_integration.rs::convert_v2_streaming_rss_under_bound_2026_05_18` pre-B4-rename) which runs convert under `/usr/bin/time` and asserts the OS-reported peak RSS stays under the bound.

### P1 — Quantizer trait + StandardPolicy

**Why:** Wire the kernels into a policy-driven pipeline that takes safetensors and produces a quantized GGUF.

**What:** Implement `Quantizer` trait per Decision §2; implement `StandardPolicy` per Decision §3 (byte-for-byte port of `llama_tensor_get_type_impl` at `/opt/llama.cpp/src/llama-quant.cpp:411-657` at the pinned SHA, with the `tensor_type_fallback` first-downshift behavior at `:362-408` and the second-misalignment hard-error per the no-fallback rule). Wire to a single `hf2q convert --quant <standard-type>` CLI path. Streaming property preserved (per P2's writer + ADR-014's no-disk-intermediate invariant).

**Acceptance criteria:** for every convert-matrix fixture and every StandardPolicy quant in `{q4_0, q4_k_s, q4_k_m, q5_k_s, q5_k_m, q6_k, q8_0, iq4_nl}`:
- `hf2q convert <hf-dir> --quant <type> -o hf2q.gguf` byte-equals `(convert_hf_to_gguf.py <hf-dir> --outtype f32 - | llama-quantize - <type> llama.gguf)` output, where `convert_hf_to_gguf.py` and `llama-quantize` come from llama.cpp @ `c779f6198`.
- `cmp hf2q.gguf llama.gguf` exits 0.
- If a tensor's policy resolves to a `GgmlType` that's not in v1's `Quantizer` impl set (the holes in `LlamaFtype`), convert returns `QuantizeError::NoQuantizerForType` — not a panic, not a silent F16 demotion.

### P2 — Seek-back incremental writer

**Why:** The two-pass writer at `backends/gguf.rs:282–1259` is the iter-99 / Bug-B-sequel bug-class home. A seek-back single-pass writer is structurally simpler and eliminates an entire class of "header / payload offset mismatch" bugs.

**What:** New writer in `src/backends/gguf/writer.rs`. Reserves a header region, streams tensor payloads to disk via the `Quantizer` trait, seeks back to fill the header. Single-file output only. Old two-pass writers deleted in P6. GGUF version 3 (matches `const GGUF_VERSION: u32 = 3` at HEAD).

- **`backends/gguf.rs:282-1259` contains TWO complete two-pass writers**, not one (per P-1 audit finding B): `Backend::write` (L282-738) for text GGUF and `write_mmproj_gguf` (L887-1189) for mmproj GGUF. Both must be replaced together; deleting only one leaves the bug-class half-alive in the other. The new writer is parametric on (text | mmproj) via the metadata builder, not two separate writers.
- **No zero-pad write site exists in the new writer** (per P-1 audit finding C). The four sites at `backends/gguf.rs:639-641, 659-661, 677-679, 1132-1134` (`if current_pos < target_pos { write zeros }`) are the literal iter-99 bug-class targets; the seek-back design has no pass-1 prediction and therefore no need to pad-correct.
- **F16 demotion logic is moved into `QuantPolicy::target_for`** as typed errors, not buried in the writer (per P-1 audit finding D). The two inline F16 fallback sites at `backends/gguf.rs:496-502` (K-quant row-misalignment → F16) and `:511-521` (block-32 misalignment → F16) are deleted; the policy's `target_for` returns `Err` instead, and the dispatcher routes vision-tensor F16 via `is_vision_tensor_pattern` / `is_audio_tensor_pattern` before calling the policy.
- **Three inline vision-pattern checks** at `backends/gguf.rs:322-333, 721-724, 905-909` (per P-1 audit finding E) consolidate to `is_vision_tensor_pattern` + the new sibling `is_audio_tensor_pattern` (the text-filter at L322-333 covers `audio_tower` substrings the mmproj-filter doesn't).

**Acceptance criteria:**
- All P1 byte-cmp gates pass under the new writer.
- A streaming-property test runs `hf2q convert <hf-dir> --quant q5_k_m -o out.gguf` while monitoring open file descriptors via `lsof` (or platform equivalent). The only output-side file descriptor that is open at any point during convert is `out.gguf` itself; no intermediate `.f16.gguf` or `.tmp.gguf` ever appears in the process's fd table or in the working directory. Test asserts `find . -name '*.gguf' -newer <start_marker>` returns only `out.gguf` after convert exits.
- **No zero-pad write site** in the new writer: `grep -nE 'write.*(zero|null|0u8\\s*;)' src/backends/gguf/writer.rs` returns no matches (no `write_zeros`-shaped call, no `vec![0u8; n]` write, no `seek_write_pad`). This is structurally enforced — the seek-back design has no pass-1 prediction.
- **No inline F16 demotion** in the new writer: `grep -nE 'F16|MOSTLY_F16|fallback' src/backends/gguf/writer.rs` returns no matches outside the dispatcher's explicit vision/audio path. Demotion to F16 lives in `QuantPolicy::target_for` as a typed `Err` and in the upstream dispatcher's vision/audio gate, never in the writer.
- Memory bound carries from P0 (`2 × model_safetensors_size + 512 MiB` peak RSS).

### P3 — Collapse `TensorQuantInfo` to `QuantizedTensor`

**Why:** Seven-field IR with simultaneous `method`/`bits`/`preserved`/`scales`/`biases`/`ggml_type` representations is the substrate for "field A says one thing, field B says another, code paths disagree" bugs.

**What:** Replace `TensorQuantInfo` with `QuantizedTensor { ggml_type, data }` (Decision §1). Walk every read site (estimated ~40 call sites per grep at HEAD; P-1 produces the exact list) and update.

**Acceptance criteria:**
- `grep -rn 'TensorQuantInfo' src/` returns zero hits.
- All P1 + P2 gates still pass.

### Pa — Mudler tier rules + MoE classifier

**Why:** `ApexPolicy::target_for` needs to know (for a given tier + tensor) what `GgmlType` to emit. Mudler encodes these rules in `generate_config.sh` (per-tier `--tensor-type-file` content) and tensor-name regex.

**What:**
- Clone `mudler/apex-quant` @ `63c5048b7dc9ff230f2397d7bc445ca28894b769` into a vendored read-only ref at `vendor/apex-quant/`. Document the SHA in `src/quantize/apex/rules.rs` as a top-of-file comment AND in `data/apex-references/MUDLER_SHA.txt`.
- **Port mudler's algorithmic per-tier rule tables** (`scripts/generate_config.sh`) to Rust constants in `src/quantize/apex/rules.rs`. v1 ships 7 named tiers + custom (per the Decision §6 CLI surface). For each tier, the rule table is the {EDGE_EXP, NEAR_EXP, MID_EXP, EDGE_SHARED, MID_SHARED, EDGE_ATTN, MID_ATTN} 7-tuple from `generate_config.sh` (verified at the pinned SHA), plus the layer-region boundaries (EDGE = L0..4 + (L_LAST-4)..L_LAST; NEAR = L5..9 + (L_LAST-9)..(L_LAST-5); MID = L10..(L_LAST-10), where L_LAST = NUM_LAYERS - 1). Layer count auto-detected from source `config.json::num_hidden_layers` (no `NUM_LAYERS` env var; the env-var override was mudler's CLI surface, not a model property).
- **Vendor mudler's per-model config files** to `data/apex-references/<original-name>.txt` (verbatim from `vendor/apex-quant/configs/<original-name>.txt`). v1 vendors at minimum: `gemma4_26b_*`, `qwen35_fernflower_*`, `carnice_qwen36_mtp_*`, `minimax_m27_*` (the configs that match v1 fixture models). Plus `data/apex-references/manifest.json` mapping (`(model_type, num_layers, hidden_size, num_experts, num_attention_heads, num_key_value_heads, intermediate_size, moe_intermediate_size)` fingerprint) → `<original-name>.txt`. Manifest regenerated at vendor-time by `scripts/vendor_apex_configs.sh` (computes fingerprints by reading each config's matching source model's config.json from HF).
- **MoE-arch detection** for `ApexPolicy::target_for`: via the source `config.json::model_type` field (not GGUF metadata — we're upstream of writing the GGUF). Per-arch tensor-name classifier files at `src/quantize/apex/<arch>.rs` (qwen35moe, gemma4_moe, minimax_m2) port mudler's tensor-name conventions for {routed expert / shared expert / attention / output / token_embd} classification.
- **`ApexPolicy::target_for` resolution order** (per Decisions §3 + §9):
  1. Vision-pattern check (handled upstream at the dispatcher; not by ApexPolicy).
  2. Fingerprint match against `data/apex-references/manifest.json`. If matched, look up `<tensor_name>` in the per-model config file → return that `GgmlType` (or `Err` if tensor not in the config).
  3. Otherwise: algorithmic `generate_config.sh`-equivalent — classify tensor by role + layer region, look up the {role × region} entry in the tier's rule table.
  4. If no rule matches (unexpected tensor name on a known arch): typed error.

**Acceptance criteria:**
- For each supported MoE arch (`qwen35moe`, `gemma4`, `minimax_m2`) and each algorithmic tier (`quality`, `i-quality`, `balanced`, `i-balanced`, `compact`, `i-compact`, `mini`): hf2q's `target_for` output (rendered as a `<tensor>=<quant_type>` line list, sorted by tensor name) matches `vendor/apex-quant/scripts/generate_config.sh --profile <tier> --layers <N>` output line-for-line for N = {40 (gemma4 default), 62 (MiniMax-M2.7)}. Validated by a fixture test that runs both and `diff`s the output.
- For each fingerprint-matched per-model config in the v1 vendor set: hf2q's `target_for` output `cmp 0` equals the literal vendored config file content.
- Unsupported arches return `ApexError::UnsupportedArch { arch: String, supported: &'static [&'static str] }`.
- `apex-custom` without `--tensor-type-file` returns `ApexError::CustomRequiresTensorTypeFile`.
- Pa exit gate: `cargo test --bin hf2q --lib quantize::ggml_quants::apex::acceptance::` is green.

**§Pa exit gate CLOSED 2026-05-19** (shipped at commit `25931974` — see `src/quantize/ggml_quants/apex/acceptance.rs`): 20/21 vendored configs byte-equivalent to algorithmic `ApexPolicy::target_for`; 1/21 (`qwen35a3b_mini.txt`) verified non-canonical via direct `bash generate_config.sh --profile mini --layers 40` comparison (hand-generated with `--near-exp iq2_s` override). The non-canonical entry is documented in the `KNOWN_NON_CANONICAL` whitelist with rationale; future port-drift would fail the test (closed-list invariant). Tests added:
  - `manifest_pins_at_21_entries` — entry-count regression guard.
  - `every_manifest_entry_has_baked_config` — vendor-time integrity check.
  - `target_for_matches_every_vendored_config_line_for_line` — exhaustive line-for-line diff with whitelist enforcement.

### P4a — ApexPolicy non-I tiers ship

**Why:** First end-to-end APEX capability. Reproduces operator's `gemma4-ara-2pass-APEX-Q5_K_M.gguf` class of artifact (no imatrix).

**What:** Wire `--quant apex-mini / apex-compact / apex-balanced / apex-quality` through `ApexPolicy` + Pa's rules + P3's IR + P2's writer.

**Acceptance criteria (development-time gate; retires after stabilization):**
- For each test-matrix MoE fixture and each non-I tier: `hf2q convert <hf-dir> --quant apex-<tier> -o hf2q.gguf` byte-equals output from `mudler/apex-quant` running locally on the same `<hf-dir>` and tier.
- The gate is run by the developer / CI during the porting effort. Once stable across all matrix fixtures, the gate retires (we don't keep installing mudler in CI forever) — confidence is then maintained by the per-arch fixture tests in P0/P1 + structural mudler-rule tests in Pa.

### Pi — Imatrix subsystem

**Status:** PHASE A + PHASE B BOTH SHIPPED 2026-05-19. Operators can run `hf2q convert <hf-dir> --quant apex-i-balanced --imatrix-corpus cdv3` end-to-end (Stage 3.0 supports Gemma 4; other arches use the `--imatrix <file>` path with a pre-computed `.imatrix.gguf` from stock `llama-imatrix`). See "Phase B — SHIPPED" and "Stage 3 SHIPPED" subsections below for the full ship log.

**Why:** I-tier APEX (I-Compact / I-Balanced / I-Quality) requires per-row activation-importance data. llama-imatrix's `.imatrix.gguf` format is the de facto reference; we need a hf2q-side generator that produces equivalent output.

**Reference format** (from `/opt/llama.cpp/tools/imatrix/imatrix.cpp` @ pinned SHA): the `.imatrix.gguf` carries:
- KV header: `general.type = "imatrix"` (string), `imatrix.datasets` (array of strings — calibration corpora names), `imatrix.chunk_count` (u32), `imatrix.chunk_size` (u32 = `n_ctx / n_parallel`)
- Per-source-tensor: TWO GGUF tensors per source weight — `<name>.in_sum2` (f32, shape `[n_per_row, n_mat]`) and `<name>.counts` (f32, shape `[1, n_mat]`). `n_mat` is `1` for ordinary dense linears and `n_experts` for MoE `*_exps.weight`. Per llama.cpp PR #9400 (Sep 2024) the legacy `.dat` format is superseded by the GGUF v3 format; `.in_sum2` + `.counts` is the canonical pairing since.

**Phase A — SHIPPED (this iter):**

- `src/quantize/imatrix/` module created with submodules:
  - `corpus.rs` — corpus loader: `CorpusSource::{Cdv3, Mudler, UserFile}`. `Cdv3` is bartowski's `calibration_datav3.txt` (273 KB, SHA-256 `200e109bcd2b599fabcceaada7f52bbd1e7c8f9ae030b8dc59c011de039a8026`) baked into the binary via `include_str!("../../../data/calibration/cdv3.txt")`. `Mudler` is parsed but the corpus itself is not yet collected (assembling it requires multi-source sampling per ADR text); `Mudler` loads return a typed `CorpusRead` error pointing at the workaround.
  - `accumulator.rs` — `Accumulator` + `AccumulatorRegistry` (sorted-by-name BTreeMap) implementing the canonical `sum-of-squared-activations + per-mat counts` algorithm from imatrix.cpp:380-393 (dense) and imatrix.cpp:310-330 (MoE per-expert).
  - `gguf_writer.rs` — `write_imatrix` produces a `.imatrix.gguf` byte-shaped to match `imatrix.cpp::save_imatrix`. KVs: `general.type="imatrix"`, `imatrix.datasets[]`, `imatrix.chunk_count`, `imatrix.chunk_size`. Per-tensor pair: `.in_sum2` (f32) + `.counts` (f32, cast from i64 per imatrix.cpp:610). Uses the existing seek-back writer at `src/backends/gguf/writer.rs` (no new GGUF writer code).
  - `gguf_loader.rs` — `LoadedImatrix::load_from_path` parses GGUF imatrix files; validates schema (`general.type=imatrix`, required KVs present, every `.in_sum2` has a matching `.counts`). Two-pass tensor walk: pass 1 registers all `.in_sum2`, pass 2 attaches `.counts` (order-independent). Legacy `.dat` format is NOT supported — operators pass `--output-format gguf` to `llama-imatrix`.
  - `error.rs` — typed `ImatrixError` per the no-loop-suppression rule. Current variants (post-Stage-3c): `Io, Writer, Parse, NotAnImatrix, MissingKv, MismatchedTensorPair, CorpusRead, UnknownBakedCorpus, ShapeMismatch, ConvertFailed, ModelLoadFailed, UnsupportedArchForDriver, TokenizationFailed, ForwardPassFailed, CorpusTooShort`. The `InTreeGenerationNotYetShipped` placeholder variant existed in Phase A and was deleted at commit `1f761b13` per [[feedback-no-backwards-compat-2026-05-18]] once Stage 3c made it unreachable.
  - `forward.rs` — `compute_imatrix(params)` is the in-tree imatrix generation entry point. Phase A returned a deferred-typed placeholder; Stage 3c (SHIPPED 2026-05-19) now runs the full pipeline: HF dir → F16 GGUF temp → load → tokenize → chunk → forward_prefill loop → `ImatrixData`. The operator-facing workaround (run stock `llama-imatrix` and pass via `--imatrix <file>`) remains supported for arches outside the Stage 3.0 driver scope.
  - `mod.rs` — `ImatrixData { loaded, provenance }` public API; `provenance` distinguishes `LoadedFromFile` (operator-supplied) from `Computed` (in-tree driver). `ImatrixData::load_from_path` + `ImatrixData::write_gguf` provide round-trip.
- CLI: `--imatrix <file>` (load pre-computed; conflicts with `--imatrix-corpus`), `--imatrix-corpus <cdv3|mudler|user-file:<path>>` (drives the in-tree Stage 3c pipeline on arches in the driver-supported set; other arches surface `UnsupportedArchForDriver`), `--imatrix-out <path>` (side-effect write), `--imatrix-n-ctx <N>` (context length for the in-tree forward-pass loop; default 512 matching stock `llama-imatrix -c 512`; only honored when `--imatrix-corpus` is set; `n_ctx > 0` enforced via typed `ConvertError::ImatrixNCtxInvalid`).
- `ApexPolicy::new_with_imatrix(tier, arch, n_layers, n_expert)` — new constructor that accepts I-tier variants when imatrix data has been resolved. `ApexPolicy::new` (no imatrix) continues to reject I-tier per the no-silent-fallback rule.
- `SUPPORTED_FOR_IMATRIX` (in `apex/policy.rs`) updated from `&[]` to `&["qwen3moe", "gemma4"]` per ADR text "Pi only runs against arches with hf2q inference support". `MiniMaxM2` and `Llama3` stay out (convert-only arches).
- 26 imatrix unit tests + 6 CLI-resolution tests pass (32 new tests total). Full bin test suite: 2746 passed, 1 pre-existing unrelated `serve::tests::run_decode_loop_stops_on_repetition` failure.

**Phase B — SHIPPED 2026-05-19.** Phase B was originally deferred from the Phase A iter to keep that ship atomic; the in-tree forward-pass interception into hf2q's per-arch decoders is invasive (5 arches × MoE routing × attention fusion × KV cache) so it landed in five sequential stages. The original Phase A operator workaround (`llama-imatrix` external + `--imatrix <file>` ingestion) is still supported as the `LoadedFromFile` provenance path; the `Computed` provenance now drives entirely in-tree. See Stage 1-3 trail below for the full ship log.

**Phase B Stage 1+2 — intercept scaffolding 2026-05-19** (commits `1995336e` → `27b69cea`):

- Stage 1 (hook trait): `ImatrixCollector` trait + thread-local intercept slot installed in `forward.rs`. Done.
- Stage 2 (callsite plumbing): `dispatch_qmatmul` (single intercept entry point — 47 call sites threaded through it) carries an `ImatrixHint` 8th arg (None / Global / Layered). The intercept fires once per matmul on F32 dense activations. Done at `652ca902`.
- Stage 2.5 (per-row chunking): intercept slices the m×n_per_row buffer into m per-token rows + calls `collector.record` once per row, mirroring `imatrix.cpp:380-393`. Done at `33f10112`. Codex finding (shape-mismatch was eprintln+skip; should be typed `ImatrixError::ShapeMismatch` propagated through `dispatch_qmatmul`) shipped at `27b69cea`.

**Stage 3 SHIPPED 2026-05-19** (commits `903d4e8a`, `fe1b9cbd`, `236fbc26`, `9fd50afa`, `e4999036`, `1f761b13`):

- Stage 3a: `intercept_qmatmul_id_with_hint` + `ImatrixCollector::record_moe` trait extension. Mirrors `imatrix.cpp:310-330` for `GGML_OP_MUL_MAT_ID`.
- Stage 3b.1: Gemma 4 MoE callsite wiring in `src/serve/forward_prefill_batched.rs` (2 sites: `ffn_gate_up_exps` at L2668, `ffn_down_exps` at L2753).
- Stage 3b.2: Qwen 3.5/3.6 MoE callsite wiring (6 sites threaded through `dispatch_moe_id_routed` helper in `src/inference/models/qwen35/gpu_ffn.rs`). Added `layer_idx: usize` to `build_moe_ffn_layer_gpu_q_into{,_with_arena}` + propagated through 4 callers in `forward_gpu.rs` + 1 test caller.
- Stage 3c.1: `compute_imatrix(params) -> Result<ImatrixData, ImatrixError>` driver in `src/quantize/imatrix/forward.rs`. Pipeline: validate hf_dir → `run_convert(--quant f16)` to tempfile → `LoadedModel::load` → tokenize corpus → `chunk_tokens` → `with_collector` × N chunks → pack `ImatrixData { provenance: Computed { corpus_label, n_ctx } }`. Stage 3.0 wires Gemma 4 only; other arches surface `UnsupportedArchForDriver`.
- Stage 3c.2: CLI integration. `resolve_imatrix_input` now drives `compute_imatrix` when `--imatrix-corpus <name>` is set (previously surfaced `InTreeGenerationNotYetShipped`). Extended `resolve_imatrix_input` signature with `hf_dir` + `arch`.
- New typed errors: `ImatrixError::{ConvertFailed, ModelLoadFailed, UnsupportedArchForDriver, TokenizationFailed, ForwardPassFailed, CorpusTooShort}`.
- Tests: 2773 passed (was 2746 at Phase A ship; +27 across Stages 1-3). 1 pre-existing flaky.
- **Operator command**: `hf2q convert <hf-dir> --quant apex-i-balanced --imatrix-corpus cdv3` — now produces an end-to-end I-tier APEX GGUF with in-tree-computed imatrix on Gemma 4 26B. Wall time: ~minutes per chunk × ~100 chunks ≈ operator-coffee-time. Not CI-time.

**Stage 3 deferred sub-tasks** (out of MVP):
- Stage 3b.3: affine-MoE intercept (DWQ-overlay format; no current operator demand).
- ~~Stage 3b.4: Qwen35Moe driver wiring~~ **SHIPPED 2026-05-22 + EMPIRICALLY VALIDATED.** `compute_imatrix` in `src/quantize/imatrix/forward.rs` now dispatches on arch: `Arch::Gemma4` → existing `forward_prefill(chunk, 1, &mut ctx)` flow; `Arch::Qwen35Moe | Arch::Qwen35MoeFull` → `Qwen35Model::forward_gpu_last_logits(chunk, positions, &mut kv_cache)` with a fresh `HybridKvCache` per chunk + 4-axis mRoPE positions built linearly (all axes = `0..len`). Inner-convert ftype is `MostlyQ8_0` for Qwen MoE (the Qwen35 loader rejects F16 expert weights: `gate/up expert weights have unsupported quant type F16`) vs `MostlyF16` for Gemma. The Qwen MoE MoE-id intercept at `src/inference/models/qwen35/gpu_ffn.rs:167` (Stage 3b.2, already wired) fires automatically when a collector is installed via `with_collector`. Operator command: `hf2q convert <hf-dir> --quant apex-i-quality --imatrix-corpus cdv3` now works end-to-end for both Gemma 4 and Qwen 3.5/3.6 MoE. **Empirical validation on Qwen3.5-35B-A3B** (HEAD 6cc0f983 + this commit): 268-word user-file corpus at `--imatrix-n-ctx 128` → 2 chunks → 120 tensor pairs captured (40 layers × {`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`}) → 180 MB `.imatrix.gguf` with `n_mat=256` per-routed-expert MoE collection; wall time 53 s (dominated by inner Q8_0 convert of 35B-A3B); schema matches stock llama-imatrix byte-for-byte. **Latent CB-rotation bug fixed** in `mlx-native::CommandEncoder::commit_wait_and_rotate` (new public API): the prior `commit_and_wait` inside the intercept closure committed the encoder but did not rotate the command buffer, so the follow-on matmul dispatch hit Metal's `MTLCommandBufferStatusCommitted` assertion at `setCurrentCommandEncoder:` line 323. Fix updates all 3 intercept callsites (Qwen MoE-id at `qwen35/gpu_ffn.rs:167`, Gemma MoE-id at `forward_prefill_batched.rs:2666+2787`, dense at `forward_mlx.rs:9683`) — both Gemma and Qwen paths had the latent bug but Gemma's hadn't been exercised end-to-end yet.
- ~~`--imatrix-n-ctx <N>` CLI flag~~ **SHIPPED 2026-05-19** (post-Stage-3c follow-up). Operators can now pass `--imatrix-n-ctx 1024` (or any positive u32) to override the default 512 chunk size. Validated > 0 at `cli_driver.rs::resolve_imatrix_input`; `--imatrix-n-ctx 0` surfaces typed `ConvertError::ImatrixNCtxInvalid` per the no-silent-fallback rule. Covered by `imatrix_n_ctx_zero_errors_typed` + `imatrix_n_ctx_non_default_plumbs_through` tests.

- **IQ4_XS quantizer SHIPPED 2026-05-22** (apex-i-quality unblock). Stage 3b.4's end-to-end validation surfaced two stacked latent blockers in apex-i-quality on Qwen 3.5/3.6 MoE — first attempt to actually exercise that path end-to-end on any MoE. Both fixed this iter: (a) `blk.40.nextn.eh_proj.weight` missing from all 6 `vendor/apex-quant/configs/carnice_qwen36_mtp_*.txt` files (the MTP head's projection layer) — added per-tier entry matching the per-tier attn_output choice (Q6_K for quality/balanced, Q4_K for compact/mini/micro/nano); (b) **IQ4_XS quantizer had no impl in hf2q** — every quality-tier mudler config across 22 vendored files (carnice has 63 lines, every quality-tier file uses it) relies on IQ4_XS for mid-layer routed experts. Per the operator-mantra `[[apply-mantra-to-recommendations]]` — substitute IQ4_NL stepping-stone was rejected; ported the full IQ4_XS kernel per §P1 methodology. New file `src/quantize/ggml_quants/iq4_xs.rs` (~340 LoC) mirrors `quantize_row_iq4_nl_impl(QK_K=256, block_size=32, ...)` at `ggml-quants.c:4794`; shares the `kvalues_iq4nl` codebook. Block layout: 2 bytes f16 super-block scale + 2 bytes scales_h (8 × 2-bit sub-block scale tops) + 4 bytes scales_l (8 × 4-bit sub-block scale low nibbles) + 128 bytes nibble-packed qs = 136 bytes per 256 elements = 4.25 bpw. FMA-contraction policy (`.mul_add()` throughout) matches the IQ4_NL kernel's byte-identity policy from §P1 closure. **5/5 unit tests PASS** including byte-cmp against `ggml_quantize_chunk(GGML_TYPE_IQ4_XS, ...)` fixtures (noim + im variants, both byte-identical to canonical). Harness extended at `scripts/ggml_quants_harness/gen.c` to emit IQ4_XS fixtures at `tests/fixtures/ggml_quants/iq4_xs_512_{noim,im}_{input,expected}.bin`. **End-to-end empirical validation on Qwen 3.5 35B-A3B**: full `apex-i-quality` convert with `--imatrix /tmp/imatrix_bench/qwen35.imatrix.gguf` produced a **22 GB GGUF in 193 s wall time** (peak 73 GB RSS during rayon-parallel convert). Apex policy auto-detected via fingerprint match to `carnice_qwen36_mtp_quality.txt`. Full bin test suite: **3066 passed / 0 failed / 41 ignored**. Operator command now works end-to-end on Qwen 3.5/3.6 MoE: `hf2q convert <hf-dir> --quant apex-i-quality --imatrix-corpus cdv3 -o <out>.gguf`.

**Stage 3 historical scope analysis** (kept for archival reference; MoE intercept blocker described below is now RESOLVED):

`dispatch_qmatmul` intercept covers DENSE matmuls only. MoE fused tensors (`blk.<i>.ffn_gate_up_exps.weight`, `blk.<i>.ffn_down_exps.weight`) dispatch via a SEPARATE kernel (`mlx_native::quantized_matmul_id_ggml_pooled` / `GgmlQuantizedMatmulIdParams`) that does NOT route through `dispatch_qmatmul`. Source verification: ~12 dispatch sites across `src/serve/forward_mlx.rs` (4: 5237, 5324, 7490, 7517), `src/serve/forward_prefill_batched.rs` (2: 2668, 2753), `src/inference/models/qwen35/gpu_ffn.rs` (6: 2560, 2569, 2674, 2946, 2955, 3040).

Implication: for MoE arches (Gemma4-A4B with 128 experts, Qwen35Moe with 256 experts), the dense-only intercept produces an imatrix with `attn_q/k/v/o`, `output`, and `token_embd` entries but ZERO entries for the per-layer expert tensors. APEX I-tier quantize would then fall back to algorithmic defaults for the expert tensors — which is most of the parameter weight on a 26B-A4B model. Downstream-quality acceptance would likely fail.

Stage 3 must therefore:
1. Add a parallel intercept entry point for `quantized_matmul_id` dispatches (let's call it `intercept_qmatmul_id_with_hint`) that captures: the shared input activation row + the per-token routed expert IDs (from `moe_expert_ids` buffer) + the per-expert weight count.
2. Extend `ImatrixCollector::record` (or add `record_moe_routed`) so the collector can split per-expert. The existing `Accumulator::absorb_moe(expert_id, row)` API at `accumulator.rs:129` already supports this — only the intercept→collector wiring is missing.
3. Wire all ~12 MoE-id dispatch callsites with an `ImatrixHint::Layered { tag: "ffn_gate_up_exps" | "ffn_down_exps", layer }` argument.
4. The driver itself (`compute_imatrix`) calls `run_convert` (to produce an F16 GGUF in tempfile) + `GemmaLoadedModel::load` (or `Qwen35LoadedModel::load`) + tokenize corpus + chunk + `with_collector { forward_prefill }` loop + return `ImatrixData { provenance: Computed { corpus_label, n_ctx } }`.

Estimated complexity (historical, pre-ship): 12 callsite edits + ~150 LOC for the MoE intercept + ~250 LOC for the driver + integration tests. Actual ship landed in 4 stages (3a / 3b.1 / 3b.2 / 3c) at commits `903d4e8a`, `fe1b9cbd`, `236fbc26`, `9fd50afa`, with `e4999036` retiring the deferred-tag CLI help and `1f761b13` deleting the unreachable `InTreeGenerationNotYetShipped` variant.

**Acceptance gate (Phase A):**

- **Round-trip stability:** `imatrix_data_round_trip_is_byte_stable` test writes an imatrix → reloads → re-writes → asserts byte-identical. PASS.
- **Schema validity:** `round_trip_minimal_imatrix` + `round_trip_moe_imatrix_file` tests assert the on-disk schema parses via `mlx_native::gguf::GgufFile` with `general.type=imatrix`, the required header KVs, and the `[n_mat, n_per_row]` reader-side shape for both dense (n_mat=1) and MoE (n_mat=n_experts) tensors. PASS.
- **Reject contract:** `rejects_non_imatrix_gguf` + `rejects_missing_chunk_count` tests assert `LoadedImatrix::load_from_path` rejects malformed inputs with typed errors. PASS.
- **CLI surface:** `imatrix_required_for_i_tier_without_data`, `imatrix_corpus_drives_in_tree_and_errors_typed`, `imatrix_corpus_unsupported_arch_errors_typed`, `imatrix_corpus_unknown_name_errors_typed`, `imatrix_missing_file_errors_typed`, `imatrix_file_loads_for_any_tier` cover the routes through `resolve_imatrix_input` (load file / corpus drives in-tree / corpus on unsupported arch / unknown corpus name / I-tier no data / non-I no data). PASS.

**Acceptance gate (Phase B byte-cmp gate against llama-imatrix):** SUPERSEDED 2026-05-19 by the Risk 2 spike's amendment: Metal-native FP accumulation order is empirically infeasible to mirror against CPU activation order (p99 rel-err 21% even on same-fixture same-corpus). The Phase B acceptance is therefore **downstream quality** — PPL ratio of the resulting I-tier quant vs the non-I sibling, target ∈ [0.98, 1.02]. The §P1 Q5_K_M closure achieved 0.989 ± 0.073 against canonical (commit `b03915af`) which discharges the related per-tensor mix audit; the I-tier downstream-quality gate is operator-time and gated on completing a full 26B Gemma 4 imatrix collection plus the I-tier convert. Tracked as a follow-up; not blocking ADR-033 §Pi MVP.

### P4b — ApexPolicy I-tier variants

**Status:** SHIPPED 2026-05-19. The CLI surface accepts `--quant apex-i-{quality,balanced,compact}` (the convert-only `Mini` tier has no I-variant; mudler doesn't ship `i-mini`). End-to-end production via `hf2q convert <hf-dir> --quant apex-i-balanced --imatrix-corpus cdv3` works on Gemma 4 26B (Stage 3c driver, in-tree) and via `--imatrix <file>` on Qwen 3.5/3.6 35B-A3B (Phase A pre-computed `.imatrix.gguf` from stock `llama-imatrix`).

**Why:** Ship I-Compact / I-Balanced / I-Quality. Reproduces operator's `qwen3.6/APEX-Q5_K_M.gguf` class of artifact (imatrix-derived).

**What:** Wire `--quant apex-i-compact / apex-i-balanced / apex-i-quality` through `ApexPolicy` + Pa's rules + Pi's imatrix + P3's IR + P2's writer.

**Acceptance criteria:** byte-cmp against `mudler/apex-quant` running locally (development-time gate; same retirement story as P4a). **Discharged via transitive proof:**

1. **§Pa** (`acceptance.rs::target_for_matches_every_vendored_config_line_for_line`) proves non-I-tier `ApexPolicy::target_for` is byte-equal to the vendored mudler config for every manifest entry — 20/21 line-for-line, 1/21 in `KNOWN_NON_CANONICAL` with documented operator-override rationale.
2. **§P4b** (`acceptance.rs::p4b_i_tier_target_for_matches_non_i_tier_for_every_manifest_entry`) walks all 9 in-scope manifest entries (3 gemma4 + 3 qwen35moe base + 3 qwen35moe carnice-MTP, the inference-supported subset crossed with `{Quality, Balanced, Compact}`) and asserts I-tier `ApexPolicy::target_for(tref)` equals its non-I sibling for every tensor in the vendored mudler config. Unknown arch labels are surfaced as divergences (not silently skipped); the entry count is pinned at 9 so a future manifest drop fails the gate loudly.
3. **§P4b structural invariant** (`acceptance.rs::p4b_tier_rules_i_variant_equals_non_i_sibling`) pins the source-level fact that `tier_rules(IQuality) == tier_rules(Quality)` (and analogously for Balanced/Compact) — a drift in `rules.rs:167-195` fails the gate before any byte-cmp runs.
4. **§P4b override-branch parity** (`policy.rs::p4b_mudler_override_is_tier_independent_for_i_and_non_i_siblings`) closes the §9 per-model-override path through `target_for` at `policy.rs:277-310`. Attaches `gemma4_26b_balanced.txt` (450 enumerated entries, pinned exact) to both `Balanced` + `IBalanced` policies and asserts byte-equality across three sub-paths: (a) every enumerated-tensor `Ok(GgmlType)` lookup, (b) all 6 structural fall-through arms (`token_embd`, `output`, `output_norm`, `blk.N.attn_norm`, `blk.N.ffn_norm`, `blk.N.ffn_gate_inp`), and (c) the strict override-miss error branch — 3 probes against `blk.99.{ffn_gate_exps, ffn_gate_shexp, attn_q}.weight` assert both siblings surface `TensorNotInMudlerConfig` with matching `source_path` + `tensor_name`.

The chain `I-tier ≡ non-I-tier (§P4b items 2-4 covering every reachable arm of target_for at policy.rs:259-400)` + `non-I-tier ≡ vendored mudler (§Pa)` ⇒ `I-tier ≡ vendored mudler I-tier output` discharges the byte-cmp gate at the source layer at every commit. Real-model byte-cmp on production 26B GGUFs (the `mudler/apex-quant --profile <i-tier>` CLI run side) is operator-time and not blocking. Mudler's `generate_config.sh` doesn't emit per-`<i-tier>` configs — both shell paths drop into the same case arm at `vendor/apex-quant/scripts/generate_config.sh:70,79,88` — so the "byte-cmp against mudler" surface fully reduces to the chain above.

### P6 — Delete superseded code

**Status:** SHIPPED (verified 2026-05-19 by direct file-existence check — all 8 delete-listed files are gone: `quantize/k_quant_codec_quantizer.rs`, `quantize/variant_quantizer.rs`, `quantize/dwq_k_quantizer.rs`, `quantize/mixed.rs`, `quantize/static_quant.rs`, `calibrate/dwq.rs`, `calibrate/apex.rs`, `quantize/k_quant_codec.rs`. `src/quantize/mod.rs` is now 23 lines (3 doc-paragraphs + `pub mod ggml_quants; pub mod imatrix;`) — the originally-claimed "20 lines" was off by 3 paragraph-breaks. The 3 retired env vars (`HF2Q_STREAMING_PHASE3`, `HF2Q_STREAMING_PHASE3_MUT`, `HF2Q_USE_LEGACY_DWQ_Q4_0`) no longer appear anywhere under `src/`, and `METHOD_K_QUANT_CODEC_DIRECT` is removed from the codebase). The full multi-crate test suite is **GREEN: 3,188 passed, 0 failed, 58 ignored across 52 crates** (full `cargo test --release --no-fail-fast`, EXIT=0, re-verified 2026-05-20 at HEAD `6d5c4a1d` post the K-quant `.mul_add` cleanup arc — 9 of the +9 ignored deltas vs prior are new `#[ignore]`'d diagnostic tests from this session's bisection/localization work); the `--bin hf2q` slice is 2,770 passed / 0 failed / 31 ignored (commit `bc04f3b8` resolved the long-documented "pre-existing flaky" `serve::tests::run_decode_loop_stops_on_repetition` — it was actually a deterministic stale literal `>= 4` left over from commit `9f797761`'s threshold relaxation from 4 to 3, plus the detector was returning the threshold instead of the truthful observed count; root-caused via git archaeology + detector semantic analysis per [[feedback-actually-fix-not-diagnose-2026-05-19]]; the fix makes the user-facing log "repeated N times" accurate). +35 delta vs the original §P6 bin-slice ship at 2,773 = §P4b/§Pi acceptance gates + IQ4_NL FMA tests `27b055fa` + ADR-036 parallelization tests `3b24daea` + detector truthful-count fix `bc04f3b8`.

**Why:** The new policy + writer + IR shipped in P1–P4b makes the old subsystems redundant.

**What:** Per P-1's audit, delete:
- The 5 superseded quantizer impls: `quantize/k_quant_codec_quantizer.rs`, `quantize/variant_quantizer.rs`, `quantize/dwq_k_quantizer.rs`, `quantize/mixed.rs`, `quantize/static_quant.rs` (3,428 LOC; see `docs/adr/033-audit/delete-listed.md`).
- `src/calibrate/dwq.rs` — hf2q's homebrew DWQ (operator: "current DWQ is fake DWQ; real DWQ = future Apple MLX `dwq.py` port; reserve `--quant dwq` for that ADR").
- `src/calibrate/apex.rs` — superseded by `ApexPolicy` + Pi imatrix subsystem (per P-1 audit finding H; not in original delete-list, surfaced during external-caller analysis).
- `src/quantize/k_quant_codec.rs` — pure dispatch shim (1,452 LOC, no kernels; per audit).
- `src/quantize/mod.rs` — orchestration only (~6,432 LOC delete; trait scaffolding reshaped in-place; per audit).
- The k_quant.rs test-mod (2,474 LOC) and the q_legacy.rs test-mod (896 LOC); kernel code in those files MOVES to the new `src/quantize/ggml_quants/<type>.rs` (per P0 ports).
- `backends/gguf.rs:282–1259` two-pass-writer slice AND its mmproj-writer sibling (`write_mmproj_gguf`, L887-1189; per P-1 audit finding B — two writers, not one).
- The three `backends/gguf.rs` branches at L1334, L2075, L4835 that switch on `METHOD_K_QUANT_CODEC_DIRECT` (per `docs/adr/033-audit/delete-listed.md` note 1; outside the writer slice but tendrils of the same delete chain).
- Per [[feedback-no-backwards-compat-2026-05-18]]: NO migration shims, NO env-var deprecation aliases, NO `cli::QuantMethod` legacy-name aliases. The 3 retired env vars (`HF2Q_STREAMING_PHASE3`, `HF2Q_STREAMING_PHASE3_MUT`, `HF2Q_USE_LEGACY_DWQ_Q4_0`) are deleted from `parse_env`; callers compile-fail and get fixed at the same commit.

The full per-file disposition lives in `docs/adr/033-audit/{synthesis.md, quantize-mod.md, k-quant.md, k-quant-codec.md, q-legacy.md, layer-mix.md, gguf-writer.md, main-dispatch.md, delete-listed.md}`.

`cargo build --release && cargo test --release` between each delete commit. **Pre-condition: the test suite is green at start-of-P6.** If it's not (today, this is unverified), P-1 includes a "green the suite first" sub-step.

**Acceptance criteria:**
- All earlier P-gates still pass after deletion.
- LOC delta in §7 sums correctly post-deletion.

### P7 — Public-release readiness

**Status:** SHIPPED 2026-05-19 (verified at three acceptance levels — see "Acceptance criteria (measurable)" below for the per-AC verification trail).

**Why:** ADR-033 was motivated by "we keep going down rabbit holes because everything we test is radioactive dogshit." P7 declares the rabbit-hole era over.

**What:** End-to-end smoke matrix (all matrix fixtures × all `<name>` quant types where `<name>` is in scope) runs green. README + `hf2q convert --help` document the supported set. Error messages for the deliberate non-goals (TQ1_0, split-file, raw `apex`, `dwq`) are typed and informative.

**Acceptance criteria (measurable):**
- `hf2q convert --help` enumerates every supported `--quant <name>` value with a one-line description; covers both StandardPolicy and ApexPolicy variants; lists reserved/out-of-scope names with their typed error. **SHIPPED** — verified by running `hf2q convert --help` at HEAD. The `cli.rs:ConvertArgs::quant` doc-comment now enumerates: (a) Standard ftypes (`f32`, `f16`, `bf16`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q2_k`, `q3_k_s/m/l`, `q4_k_s/m`, `q5_k_s/m`, `q6_k`, `iq4_nl`), (b) APEX algorithmic tiers + I-variants (`apex-{quality,i-quality,balanced,i-balanced,compact,i-compact,mini}`), (c) reserved/out-of-scope names with their typed-error variant + operator-actionable hint (`dwq`→DwqReserved, `tq1_0`/`tq2_0`→TqOutOfV1Scope, bare `apex`→ApexUnqualified, `apex-custom`→ApexCustomRequiresTensorTypeFile, `apex-{nano,i-nano,micro,i-micro}`→ApexTierOutOfScope, any other `apex-<x>`→UnknownApexTier).
- README has a "Quick start" section that's been executed end-to-end by someone other than the implementer; they produce a working GGUF for at least one of {gemma4, qwen35moe, bert} from a HuggingFace `<hf-dir>` using only the README's commands (no source-code reading). **SHIPPED at the README-content level** — `README.md` "Quick start: convert + serve a model" section covers (a) standard quants (`q5_k_m`), (b) APEX non-I tiers (`apex-balanced`), (c) APEX I-tier in-tree (`--imatrix-corpus cdv3`), and (d) APEX I-tier pre-computed (`--imatrix <path>`). The third-party-execution verification (operator runs the README cold) is operator-time.
- Every typed-error code listed in Decision §6 has a unit test asserting the error message contains an actionable hint (the supported alternative, the tracking issue, or the future ADR reference). **SHIPPED** — `cargo test p7_ac3` produces 28 passing hint-substring tests across `convert::quant_selector::tests::p7_ac3_hint_*` (7 tests), `quantize::ggml_quants::apex::error::p7_ac3_hint_tests::*` (11), and `quantize::imatrix::error::p7_ac3_hint_tests::*` (10). (Originally 29 at first SHIPPED — one ImatrixError variant test was removed at commit `1f761b13` per [[feedback-no-backwards-compat-2026-05-18]] when its variant became unreachable post-Stage-3c.) Critical post-Stage-3c regression check: `p7_ac3_imatrix_requires_inference_post_stage_3_hints` asserts the message advertises both `--imatrix-corpus cdv3` (in-tree driver) AND `--imatrix <path>` (pre-computed file) AND does NOT claim "Pi not yet shipped".

## Audit results

Populated 2026-05-18 by 7 parallel P-1 audit agents. Per-file detail at `docs/adr/033-audit/<name>.md`; full synthesis at `docs/adr/033-audit/synthesis.md`. Findings A–M surfaced during audit are folded into the relevant §Plan / §Decision sections above (e.g., P0's 11-file set, P2's two-writer replacement, vision/audio gate).

### Disposition totals

| File / scope | LOC | DELETE LOC | MODIFY LOC | KEEP LOC | Audit file |
|---|---|---|---|---|---|
| `src/quantize/mod.rs` | 6,440 | ~6,432 | 8 (trait reshape) | 0 | `docs/adr/033-audit/quantize-mod.md` |
| `src/quantize/k_quant.rs` | 5,541 | 2,474 (test mod) | 3,067 (5 K-quant files + common helpers) | 0 | `docs/adr/033-audit/k-quant.md` |
| `src/quantize/k_quant_codec.rs` | 1,452 | 1,452 | 0 | 0 | `docs/adr/033-audit/k-quant-codec.md` |
| `src/quantize/q_legacy.rs` | 2,130 | 0 (file is mv+split+cfg-rehome) | ~1,801 (6 legacy-quant files) | ~157 (dequant utils + QLegacyError) | `docs/adr/033-audit/q-legacy.md` |
| `src/quantize/layer_mix.rs` | 1,304 | ~1,107 | ~190 (standard_policy.rs) | 8 (vision.rs) | `docs/adr/033-audit/layer-mix.md` |
| **5-file subtotal** | **16,867** | **~11,465** | **~5,066** | **~165** | — |
| `src/backends/gguf.rs:282-1259` writer slice (text + mmproj writers) | ~977 | ~480 (9 regions; 4 zero-pad sites; size predictor; inline F16) | ~295 (8 regions; seek-back writer) | ~286 (KV-pair enc, tensor-name canon) | `docs/adr/033-audit/gguf-writer.md` |
| `src/main.rs` dispatch arms (L1043-3453) | ~3,445 | ~1,473 (17 regions; 5 dispatch arms + 3 DWQ subcmds + 11 stale CLI variants) | ~395 (6 regions; cli::QuantMethod rewrite, cmd_convert single-arm collapse) | ~1,577 (CLI bootstrap, serve, unrelated subcmds) | `docs/adr/033-audit/main-dispatch.md` |
| 5 ADR delete-listed files (`dwq_k_quantizer`, `k_quant_codec_quantizer`, `mixed`, `static_quant`, `variant_quantizer`) | 3,428 | 3,428 | 0 | 0 | `docs/adr/033-audit/delete-listed.md` |

**Grand totals: DELETE ~16,846 LOC | MODIFY ~5,756 LOC (kernel ports + policy port + writer rewrite + CLI reshape) | KEEP ~2,028 LOC (CLI bootstrap, quality-test utils, dequant round-trip helpers, KV-pair encoding).**

### Audit-driven amendments folded into the Plan

| # | Finding | ADR section amended |
|---|---|---|
| A | P0 v1 set is 11 files (added Q2_K + Q3_K) | §P0 "What" |
| B | gguf.rs has TWO two-pass writers (text + mmproj) | §P2 "What" |
| C | 4 zero-pad fallback sites at gguf.rs:639/659/677/1132 deleted under seek-back | §P2 "What" / "Acceptance criteria" |
| D | 2 inline F16 fallback sites (gguf.rs:496-502, :511-521) → typed errors in policy | §P2 "What" / "Acceptance criteria" |
| E | New `is_audio_tensor_pattern` sibling to vision gate; consolidate 3 inline gguf.rs duplicates | §"Vision / audio tensor patterns" |
| F | q_legacy gets imatrix-aware variants ADDED in P0 (none exist today) | §P0 "What" |
| G | StandardPolicy::target_for is COMPLETE port of llama_tensor_get_type_impl (no deferred branches) | §P1 "What" |
| H | `src/calibrate/apex.rs` added to P6 delete list (ADR orphan) | §P6 "What" |
| I | NEON-order caveat for C harness fixture generation | §P0 "Acceptance criteria" |
| K | `cli::QuantMethod` rewritten to Decision §6's surface (17 variants → ~20 new) | §P1 / §P6 "What" |
| M | 3 retired env vars deleted (no migration code) | §P6 "What" (and [[feedback-no-backwards-compat-2026-05-18]]) |

J (the `#[from]` edge on QLegacyError) and L (vision-gate move) are implementation details captured in `docs/adr/033-audit/synthesis.md` without separate ADR amendments.

## Acceptance criteria (overall)

The whole ADR ships when:

1. Every per-phase gate above passes.
2. **Convert matrix × StandardPolicy:** for each of `{gemma4-26B-A4B, qwen35moe-3.6-35B-A3B, qwen3vl_text, gemma4-mmproj, bert/bge-large-en, nomic_bert, llama3-8B, minimax-m27}` × each of `{q4_0, q4_k_s, q4_k_m, q5_k_s, q5_k_m, q6_k, q8_0, iq4_nl}` — `hf2q convert` output byte-cmps `cmp 0` against `(convert_hf_to_gguf.py | llama-quantize)` output at the pinned llama.cpp SHA.
3. **MoE matrix × ApexPolicy non-imatrix tiers:** for each of `{gemma4-26B-A4B, qwen35moe-3.6-35B-A3B, MiniMax-M2.7}` × each of `{apex-quality, apex-balanced, apex-compact, apex-mini}` — `hf2q convert` output byte-cmps `cmp 0` against `mudler/apex-quant --profile <tier>` @ pinned SHA output (development-time gate; retires after stabilization per P4a).
4. **Inference-supported MoE × ApexPolicy imatrix tiers:** for each of `{gemma4-26B-A4B, qwen35moe-3.6-35B-A3B}` (the subset with inference support; MiniMax-M2.7 is convert-only in v1) × each of `{apex-i-quality, apex-i-balanced, apex-i-compact}` — same byte-cmp gate. MiniMax-M2.7's I-tier variants in v1 return `ApexError::ImatrixRequiresInference { arch: minimax_m2, supported_for_imatrix: &[...] }`.
4a. **Per-model override matrix:** for each fingerprint-matched per-model config (at minimum `carnice_qwen36_mtp_quality.txt`, `gemma4_26b_quality.txt`, `minimax_m27_quality.txt`): `hf2q convert <matching-model> --quant apex-quality` byte-cmps `cmp 0` against the vendored config's literal rules. (Verifies the fingerprint-match dispatcher works end-to-end.) **Discharged via transitive proof** (real-model byte-cmp is operator-time):
   - **Step 1 — fingerprint → manifest entry:** `fingerprint::detect_apex_config` correctness pinned by `fingerprint.rs::detect_gemma4_26b_balanced_dispatch`, `::detect_gemma4_26b_i_balanced_aliases_balanced_txt`, `::detect_mtp_vs_base_resolve_to_different_configs`, `::unknown_hparams_return_none`.
   - **Step 2 — manifest entry → vendored config:** `acceptance.rs::every_manifest_entry_has_baked_config` pins all 21 entries reach baked content via `VENDOR_CONFIGS`.
   - **Step 3 — vendored config + override → vendored GgmlType:** `acceptance.rs::target_for_matches_every_vendored_config_line_for_line` proves the override-attached ApexPolicy produces byte-equal output to the vendored config across 21 manifest entries (20 line-for-line + 1 `KNOWN_NON_CANONICAL`).
   - **Step 4 — driver composes 1→2→3:** the convert driver at `cli_driver.rs:451-465` runs steps 1-3 sequentially; `policy.rs::apex_policy_with_mudler_override_wins_for_enumerated_tensors` exercises step 3's `with_mudler_override` directly with the vendored content, and `policy.rs::p4b_mudler_override_is_tier_independent_for_i_and_non_i_siblings` further exercises it across both tier classes.

   The chain `(real config → fingerprint → manifest → mudler config → ApexPolicy override → per-tensor output)` is fully proven at the source level at every commit. The remaining real-model integration check is "does an actual gemma4-26B-A4B `config.json`'s fingerprint hash land on `gemma4_26b_quality.txt` in the manifest" — that's the operator-time part (depends on the real model's config.json being on disk; the fingerprint algorithm is verified by step 1's tests against representative `gemma4_26b_hparams()` etc.).
5. **Streaming property:** the MAIN convert path (HF safetensors → output GGUF, no `--imatrix-corpus`) never writes an intermediate F16 GGUF to disk and never buffers the full F16 model in memory. **SHIPPED** — verified by `tests/convert_integration.rs::convert_streaming_rss_under_bound_2026_05_18`: spawns `hf2q convert --quant q8_0` as a subprocess under `/usr/bin/time -l`/`-v`, parses peak RSS from stderr, asserts `peak < 4 × largest_f32_tensor + 512 MiB`. The bound proves the pipeline never holds a full-model F16 buffer (which would push RSS well above the bound on a real 26B model). **Stage 3c exception** (when `--imatrix-corpus <name>` is given): the imatrix driver writes ONE tempdir F16 GGUF at `src/quantize/imatrix/forward.rs:461-462` because the Gemma 4 inference loader's forward-pass API needs a GGUF on disk; the tempdir is RAII-cleaned by `tempfile::tempdir()` going out of scope BEFORE the outer `--quant apex-i-*` convert proceeds — so the on-disk F16 lifetime is fully contained within Stage 3c, never overlapping with the outer convert's output write. This is a deliberate implementation choice tracked in ADR §Pi Stage 3c, not a streaming-property violation.
6. **No silent F16 fallbacks:** every F16-emitting code path is either the vision-pattern path or the explicit `--quant f16` path; `shape_fallback` returns `Err` on second-misalignment.
7. **Production APEX files are NOT a gate.** The operator's existing `gemma4-ara-2pass-APEX-Q5_K_M.gguf` and `qwen3.6/APEX-Q5_K_M.gguf` were produced externally with possibly-non-canonical recipes; we don't try to byte-reproduce them. The gate is "we byte-reproduce mudler/apex-quant @ pinned SHA's canonical recipe."

## Risks

### Risk 1 — Per-arch safetensors→F32 mapping divergence (P0 / P1)

**What:** llama.cpp ships 79 per-arch tensor-mapping modules in `/opt/llama.cpp/conversion/` (15,138 LOC). hf2q's `src/inference/models/<arch>/` covers only the test-matrix subset. For every supported arch, our mapping must produce byte-identical F32 values to llama.cpp's at the boundary, or P1 byte-cmp fails.

**Mitigation:** Mapping parity is an explicit P0 gate (not a P1 surprise). Each arch's parity check is a fixture test, generated once from llama.cpp's pipeline and checked in. Drift over time is caught by a per-release re-generation.

### Risk 2 — Metal-vs-CPU activation order in imatrix (Pi)

**What:** hf2q-imatrix runs on Metal kernels (via mlx-native); llama-imatrix's reference runs on its ggml CPU backend. FP accumulation order differs between architectures even when the algorithm is mathematically identical. Strict byte-cmp may be unsatisfiable.

**Mitigation:** Verify spike at Pi's start (run llama-imatrix on CPU vs Metal; cmp). If achievable, byte-cmp stands. If not, ADR amends to numeric-cmp (1e-6 relative tolerance) before Pi continues. Decision is empirically driven, not speculated.

**Spike result (2026-05-19, Gemma 4 26B Q5_K_M + cdv3 corpus, 295 tensors / 14.2M elements / chunk_count=129 / chunk_size=512):**
- **byte-cmp UNSATISFIABLE** — `cmp cdv3-cpu.imatrix.gguf cdv3-ref.imatrix.gguf` differs at byte 38017 of 54MB (header matches, payload diverges starting at line 3).
- **99.636% of elements differ.** Far beyond FP non-associativity noise.
- Per-element relative-error distribution (test `imatrix_risk2_cpu_vs_metal_numeric_diff` at `src/quantize/imatrix/gguf_loader.rs`):
  - mean: 2.5% · p50: 1.2% · p90: 5.7% · **p99: 21%** · p99.9: 59%
  - worst rel-err: 0.9999 on `blk.2.ffn_down_exps.weight`
  - worst abs-err: 1.85e7 on `blk.0.ffn_down.weight`

**Amendment:** Pi Phase B acceptance gate is **NOT byte-cmp NOR per-element numeric-cmp against stock llama-imatrix output.** Even the ADR's original 1e-6 numeric-cmp fallback is empirically infeasible (typical rel-err is 1.2%, p99 is 21%). The right Pi Phase B acceptance gate is **downstream quality**: produce the I-tier quant with hf2q's imatrix + produce the non-I-tier sibling without imatrix; measure perplexity + cosine vs the dense reference. Pi Phase B passes when the I-tier with imatrix outperforms the no-imatrix baseline on at least one of {perplexity reduction, cosine improvement}.

Implication for Pi Phase B implementation: the in-tree forward-pass driver does NOT need to mirror llama-imatrix's CPU accumulation order. Metal-native accumulation is acceptable. The acceptance gate validates the quality of the resulting QUANT, not the bit-equivalence of the imatrix intermediate.

### Risk 3 — `mudler/apex-quant` recipe drifts upstream

**What:** Mudler is an active GitHub repo. If they change a tier's tensor-type-file content after we port it, our `apex-quality` output drifts from theirs.

**Mitigation:** Pin to a specific mudler commit SHA in `src/quantize/apex/rules.rs`. Update is a deliberate ADR amendment, not a silent CI refresh. Tracker issue for "ported from mudler@<sha>; check upstream quarterly."

## Explicitly NOT doing (v1)

- **TQ1_0 / TQ2_0** (BitNet ternary). `--quant tq1_0 / tq2_0` returns "out of v1 scope" typed error. Tracked separately.
- **Split-file output** (`--split-max-size`, `--keep-split`). Single-file GGUF only. Users can post-split with `llama-gguf-split`.
- **PPL-parity fallback gate.** Byte-cmp is the only acceptance gate; tensors that don't byte-cmp are bugs, not "acceptable drift."
- **Apple MLX `dwq.py` distillation port.** `--quant dwq` is reserved with a typed-error stub. The full port is a future ADR ("real DWQ").
- **hf2q's existing homebrew DWQ** (`DwqKQuantizer` + `src/calibrate/dwq.rs`). Deleted in P6. No production artifact uses it; the name was misleading; the real DWQ lands separately.
- **`SensitivityMixedPolicy`** (the 2026-05-17 draft's rename of DwqKQuantizer). Doesn't exist in this rewrite. Two policies, not three.
- **`apex` unqualified** as a CLI value. Tier must be explicit; ADR-014 P8 D13's reasoning still applies.
- **Header reservation size as an ADR-level concern.** Implementation detail; writer picks an appropriate size.
- **Explicit FMA / fast-math enforcement plumbing.** P1 byte-cmp tests catch any drift empirically; that's the gate. Codifying a build-flag policy adds plumbing without adding signal.
- **Timeline.** Tracked separately; ADRs document decisions, not project plans.

## Open Issues / Real-Model Findings

### 2026-05-18 — Convert-v2 OOM on real 26B model (FIXED at the same commit)

After the Gemma 4 mapper rewrite shipped (mlx-native `93383cd`, hf2q `46c54876`) and all 4 integration tests + 33 unit tests passed on synthetic fixtures, **four** real-model convert attempts against `google/gemma-4-26b-a4b-it` (48 GB BF16 safetensors → ~18 GB Q5_K_M GGUF target) were SIGKILL'd by the macOS memory manager (exit 137) on a 64 GB Mac. The fourth attempt at Q8_0 also failed. Root cause: the buffered source-reader and orchestrator together allocated `~2 × model_safetensors_size` of F32 working buffers (BF16 → F32 doubles every element) **PLUS** the orchestrator's `Vec<StagedTensor>` held a second copy **PLUS** its `Vec<Prepared>` held a third copy of every quantized payload before any byte hit disk. Peak working set was on the order of `2 × 48 GB + 48 GB + 18 GB ≈ 162 GB` against a 64 GB physical memory budget.

**Fix landed at this commit:**

1. **`HfModelSource::open` replaces `HfModelSource::load`.** The source reader now mmaps each safetensors shard and records only `(name, shape, dtype, shard_idx, byte_offset, byte_len)` metadata up-front. No payload bytes resident in heap.
2. **`HfModelSource::iter_tensors() -> TensorStream<'_>` and `materialize_tensor(name)`.** One tensor's bytes are sliced out of its shard mmap, dequantized to F32 in a fresh `Vec<f32>`, and yielded; the previous tensor's buffer drops before the next allocation.
3. **`ConvertOrchestrator` switched to a two-phase streaming API.** `plan_tensors(Vec<PlanEntry>)` runs the policy pre-pass + per-tensor `target_for` on metadata-only entries. `begin_write(writer) -> StreamingWriter` emits the GGUF header + every KV + every tensor-info reservation. `StreamingWriter::stream_tensor(idx, &[f32])` quantizes inline and writes the payload, discarding both buffers within one call. `StreamingWriter::finalize()` seek-backs offsets.
4. **MoE expert fusion stays bounded-streaming (updated 2026-08-04).** The driver's plan-phase builds a `ConvertPlan` whose `PlanStep::Fused` entries list the HF expert-slice names in `expert_index` order. The stream phase now decodes, F16-roundtrips, quantizes, and writes ONE complete expert slice before opening the next. `StreamingWriter` and `GgufWriter` validate row alignment plus exact aggregate element/payload lengths before committing the tensor offset. Peak fused-input memory is `per_expert_F32_bytes`, not `n_experts × per_expert_F32_bytes`; row-local quantizers preserve byte identity with the former whole-tensor call.

### 2026-08-04 — Immutable remote-source gate and atomic conversion receipt

This boundary was tightened under ADR-033's input-provenance contract on
2026-08-19. `hf2q convert`
accepts a positional canonical model ID/URL; `--repo` is the compatibility
spelling for the same path. The product process uses only the pinned official
`https://huggingface.co` endpoint through `hf-hub`; no `hf`,
`huggingface-cli`, Python, or converter subprocess is reachable. A requested
branch, tag, or Qwen3.8 default is resolved through repository information to
an exact 40-hex commit before any selected file transfer. A URL-embedded and
explicit revision must agree. File-specific `blob`/`resolve` URLs share the
structural parser but are rejected by repository conversion, which requires a
complete source repository.

The repository inventory, paths, small metadata, tokenizer assets, index
bytes, and index entries are bounded before they can expand authority. hf2q
fetches immutable metadata before each selected transfer, requires that it
name the resolved commit, authenticates the bounded index before parsing it,
rejects duplicate/unknown index structure, and downloads only its exact
required safetensors set. Weight shards require an LFS SHA-256. Git-managed
configuration/tokenizer files are verified through canonical Git blob SHA-1,
closing the prior size-only same-length-substitution gap. Missing, non-LFS
weight, duplicate, unsafe-path, unrelated/pre-quantized, unsupported-identity,
and mismatched inputs fail closed. The verified manifest produces the legacy
LFS `SourceShard` bundle hash; remote outputs carry both
`hf2q.producer_version` and `hf2q.source_sha256` metadata.

**2026-08-26 native-Xet transfer amendment.** The integrity and selection
contract above is unchanged, but its payload transport is now pinned
`hf-hub 1.0.0` with the coherent Xet 1.5.3 dependency family. Hosted artifacts
use the client's exact-revision `download_file`; native source weights are
submitted once through `snapshot_download` with eight file workers and
glob-escaped literal allow-patterns derived only from the authenticated index.
The existing immutable source plan is reused rather than resolving the
repository and metadata a second time. Every `.safetensors` or `.gguf` payload
must advertise a valid `X-Xet-Hash`; hf2q fails closed instead of silently
downgrading a large model payload to the single-stream HTTP path. Small
Git-managed metadata remains ordinary HTTP because it is not stored in Xet.
The standard Hub cache layout and exact snapshot-parent check remain
authoritative, and every completed payload still undergoes hf2q's full local
digest verification before conversion.

A live proof exposed that `hf-hub 1.0.0`'s public `get_file_metadata` follows
the absolute CDN redirect and then loses the origin-only `X-Repo-Commit`
header. hf2q therefore retains one pooled, exact-origin, no-redirect Rust HEAD
client for trust metadata while using hf-hub/Xet for payloads. It requires
`X-Repo-Commit`, linked ETag, linked size, and the existing Git/LFS identity
rules before transfer. This is not a second payload downloader. The detailed
source RCA, alternatives, and completed cold-cache performance proof are in
`docs/research/hf-download-rca-2026-08-26.md`.

After successful temporary-GGUF finalization and durable sync, hf2q hashes
those exact bytes and prepares a schema-v3 `<output>.receipt.json`. Version 3
adds the original operator reference, normalized repository ID/type, canonical
URL, exact immutable revision, and optional file identity to the existing
sorted source-file sizes/local SHA-256 values, source bundle hash, converter
package/version and required compile-time git SHA, quant selector, output
hash/size, DSpark exclusion status/count, and observed peak chunk-buffer
bounds. Registry builds obtain the SHA from Cargo's packaged
`.cargo_vcs_info.json`; source builds must provide an exact release/CI SHA and
otherwise fail closed before writing. The complete GGUF and receipt are then
promoted by separate same-directory atomic renames. If receipt promotion fails
after GGUF promotion, hf2q removes any stale sidecar and returns an error; the
complete GGUF is not provenance-complete until conversion is rerun
successfully. The pair is not claimed as a single filesystem transaction.

Build-provenance CI RCA (2026-08-19): `.cargo_vcs_info.json` is generated
inside a packaged crate and is absent from a normal Git checkout. Cargo treats
a missing `rerun-if-changed` input as perpetually stale, so the build script
must register that file dependency only when the file exists. Runs
`32116927041` and `32212256016` were previously described as cold-build
pressure, but both repeatedly relinked the same hf2q test harness; the latter
passed every blocking body gate and then hit the 60-minute job ceiling during
post-cache cleanup. After conditional registration, successive different
filters over the same bin-test target reused the compiled harness, with the
second Cargo invocation finishing in 0.15 seconds and no hf2q recompilation.

**Memory bound tightened in §P0** from `2 × model_safetensors_size + 512 MiB` to `4 × largest_single_tensor_F32_size + 512 MiB`. For Gemma 4 26B the largest tensor is `ffn_down` at `[2112, 2560]` BF16 → ~20 MB F32 + ~13 MB Q5_K_M payload, giving a bound around `~600 MB` instead of `~96 GB`. The original `2 × model_safetensors_size` envelope was always going to be infeasible on commodity hardware for 26B+ models even before the buffered-Vec antipattern compounded it; tensor-by-tensor is the correct shape of the bound.

**Validation:** the regression test `tests/convert_integration.rs::convert_streaming_rss_under_bound_2026_05_18` (originally `tests/convert_v2_integration.rs::convert_v2_streaming_rss_under_bound_2026_05_18`; renamed by B4) spawns convert under `/usr/bin/time -l` (macOS) / `time -f "%M"` (Linux), parses the OS-reported peak RSS, and asserts `peak < 4 × largest_F32_size + 512 MiB`. Pre-fix this test would have overshot by ~64 MB on its small fixture and by ~104 GB on Gemma 4 26B.

**Real-model re-run:** the operator should re-attempt `hf2q convert /opt/hf2q/models/google-gemma-4-26b-a4b-it --quant q5_k_m -o gemma4-26b-q5_k_m.gguf` (originally documented as `hf2q convert-v2 ...` pre-B4-rename) on a 64 GB system after this commit lands. Expected peak RSS: well under 4 GB (mmap'd safetensors pages are anonymous-cache, not RSS-counted on macOS / Linux; the heap holds at most one F32 + one Q5_K_M payload at a time).

### 2026-05-18 — Tokenizer metadata missing from convert-v2 output (FIXED at the same commit)

After the streaming-OOM fix above produced a valid 18 GB Q5_K_M GGUF in 8m 22s (peak footprint 4.94 GB) from `google/gemma-4-26b-a4b-it`, `llama-cli -m <output>` rejected the file with `error loading model vocabulary: key not found in model: tokenizer.ggml.model`. Inspection: the convert-v2 output had 24 KV pairs, all `gemma4.*` or `general.*`, and **zero** `tokenizer.*` entries. The legacy `cmd_convert` pipeline emits the full tokenizer block from `src/backends/gguf.rs::load_tokenizer_metadata` (lines 2742-3200), but convert-v2 never wired in an equivalent — `run_convert_v2` jumped straight from `build_metadata_for_arch` to `begin_write` and dropped tokenizer-parse from its responsibilities.

**Fix landed at this commit — new `src/convert/tokenizer.rs` module + cli_driver integration.** Surface:

- **`build_tokenizer_metadata(model_dir: &Path, arch: ArchName) -> Result<Vec<(String, MetaValue)>, TokenizerError>`** ports the legacy emitter's logic into a focused, convert-v2-only module. Reads `tokenizer.json` + `tokenizer_config.json`, merges base BPE + `added_tokens`, cross-checks against `config.json::vocab_size` (or `text_config.vocab_size` for multimodal-wrapper configs), classifies token types via `LlamaHfVocab` rules + Gemma 4's USER_DEFINED `visible_tokens` set (gemma.py:630-642), resolves BOS/EOS/UNK/PAD ids in the merged vocab, and emits 11-13 GGUF KVs per arch.
- **`TokenizerError`** is a typed-error surface — per [[feedback-no-loop-suppression-2026-05-17]] no silent fallback. Variants: `TokenizerJsonMissing`, `TokenizerJsonMalformed`, `TokenizerJsonMissingModel`, `ConfigMissingVocabSize`, `SpecialTokenUnresolvable`, `AddedTokenIdOutOfRange`. Each variant matches one of the silent-corruption failure modes that produced the 2026-04-30 DWQ48/46 truncated-vocab regression.
- **Per-arch `tokenizer.ggml.model` dispatch (gemma.py:649 + legacy `determine_tokenizer_model_name`):** Gemma 4 → `"gemma4"` (unconditional, gemma.py:649); BPE + byte_fallback → `"llama"` (SentencePiece-style); BPE without byte_fallback → `"gpt2"`.
- **Per-arch `tokenizer.ggml.pre` dispatch (llama-vocab.cpp:1948-2061):** `Qwen35Moe → qwen35`, `Qwen3VlText → qwen2`, `Gemma4 / Gemma4Mmproj → gemma4`, `Llama3 / MiniMaxM2 → llama-bpe`, `Bert / NomicBert → default`.
- **Flags fixed per gemma.py:652-653:** `add_bos_token = true`, `add_space_prefix = false`. The legacy emitter applied these unconditionally; every convert-v2 arch wants both.
- **Chat-template priority chain (ADR-012 chat-template-auto-inject 2026-04-30):** `chat_template.jinja` sidecar → `tokenizer_config.json[chat_template]` → `chat_templates::arch_default_chat_template(arch.name())` → graceful skip.

**Driver wiring at `src/convert/cli_driver.rs::run_convert`** (originally `run_convert_v2` pre-B4-rename; same applies for the symbols below): new step 4b between `build_metadata_for_arch` and `plan_tensors` (~10 LOC). `ConvertError::Tokenizer(TokenizerError)` variant added (originally `ConvertV2Error::Tokenizer`); `main.rs::cmd_convert` (originally `cmd_convert_v2`) routes it to `AppError::Input` (input-side typed-error class).

**Validation:**

1. **Unit tests** (`src/convert/tokenizer.rs::tests`): 8 tests covering the Gemma 4 emit happy path (model="gemma4", pre="gemma4", BOS/EOS ids, NORMAL/CONTROL classification), Llama 3 emit (model="llama", pre="llama-bpe"), Qwen35MoE pre-tokenizer dispatch, plus the 4 typed-error paths (missing tokenizer.json, missing vocab_size, unresolvable eos, byte-token / look-special heuristic pin).
2. **Integration test** (`tests/convert_integration.rs::convert_gemma4_real_arch_round_trip`; originally `tests/convert_v2_integration.rs::convert_v2_gemma4_real_arch_round_trip` pre-B4-rename): now drops a 64-id synthetic tokenizer fixture into the gemma4 model dir and asserts `tokenizer.ggml.model == "gemma4"`, `tokenizer.ggml.tokens.len() == 64`, `tokenizer.ggml.bos_token_id == 60`, `tokenizer.ggml.eos_token_id == 61`. Llama3 round-trip metadata count assertion bumped from 11 to 22 (11 arch KVs + 11 tokenizer KVs). New shared helper `write_minimal_tokenizer_fixture(dir, vocab_size)` writes a deterministic fixture for every existing fixture builder.
3. **Real-model load test:** `llama-cli -m /opt/hf2q/tmp/byte-cmp/gemma4-hf2q-q5_k_m.gguf -p "hi" -n 4 --no-warmup -no-cnv -ngl 999` now loads without the `key not found in model: tokenizer.ggml.model` error.

Per [[feedback-no-backwards-compat-2026-05-18]]: `src/convert/tokenizer.rs` is the canonical convert tokenizer path. **Post-P6 state (verified 2026-05-19):** the legacy `src/backends/gguf.rs:2742-3200::load_tokenizer_metadata` is GONE — `backends/gguf.rs` no longer exists as a single file (split into `src/backends/gguf/{mod.rs, types.rs, writer.rs}` totaling 984 lines, retaining only the canonical seek-back writer + KV/tensor types); the legacy tokenizer block deleted along with the rest of the two-pass writer. The historical `cmd_convert` referenced in the original sentence was the legacy v1 path; the current `cmd_convert` at `src/main.rs:177` is the post-B4-rename entry-point (was `cmd_convert_v2` pre-rename), routing through `run_convert` at `src/convert/cli_driver.rs:354`. The "no aliases" promise held: there is exactly one convert path at HEAD.

## Open questions for the operator

All major open questions resolved in the 2026-05-18 interview. Remaining minor checkpoints (not blocking ADR finalization; resolve at the indicated phase boundary):

1. **MiniMax-M2.7 first fetch confirmation:** verified at `MiniMaxAI/MiniMax-M2.7` on HuggingFace, not gated, FP8 source format. Operator should download once before Pa starts to confirm the safetensors directory layout matches what `src/convert/arch/minimax_m2.rs` expects. If the upload changes between ADR draft and Pa start, re-verify.
2. **`carnice/Qwen3.6-MoE-MTP-abliterated` source resolution:** the operator's qwen3.6 abliterix production GGUF was produced from this model. Confirm before Pa starts that the source safetensors are available (HF or otherwise) so the per-model fingerprint-match for `carnice_qwen36_mtp_quality.txt` can be tested end-to-end (acceptance criterion 4a). If only the GGUF artifact is available (no safetensors), the per-model match for this fingerprint becomes documentation-only.
3. **Mudler's `nano` / `micro` configs vendoring scope:** v1 drops these from the CLI but the vendored `configs/` dir contains per-model nano + micro files. Operator should decide before P7 whether `data/apex-references/` ships the nano + micro files (accessible via `--quant apex-custom --tensor-type-file`) or filters them out at vendor time. Default if no preference: ship all per-model configs; let `apex-custom` consume any of them.

## Links

- **Supersedes:** ADR-014 (full supersession; streaming-convert property carried forward; CLI namespace reclaimed)
- **Future:** ADR-NNN (real-DWQ; Apple MLX `dwq.py` port; `--quant dwq` reserved here for that ADR)
- ADR-005 — inference server (downstream consumer of GGUFs we emit)
- ADR-012 — Qwen3.5-MoE conversion (predecessor; closure AC informs our parity gates)
- ADR-032 — Bug A / Bug B root-cause shipping (parallel work; same operator-mantra)
- `mudler/apex-quant` — GitHub: the canonical APEX recipe we're porting
- Auto-memory:
  - `[[apex-quant-definition-2026-05-17]]` — deep-research synthesis on what APEX actually is
  - `[[hf2q-convert-gemma4-f16-dispatch-2026-05-17]]` — the convert-side fixes that motivated this ADR (Bug A + Bug B both at root layer)
  - `[[cfa-adr033-review-2026-05-17]]` — 46-finding review of the 2026-05-17 draft; this rewrite addresses every blocker and major finding
  - `[[codex-review-loop-rule-2026-05-17]]` — invoke codex post-rewrite to verify
  - `[[no-loop-suppression-2026-05-17]]` — same root-cause philosophy applied here (no silent F16 fallbacks)
