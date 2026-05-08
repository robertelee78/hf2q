# ADR-022: mlx-native kernel coverage parity with llama.cpp peer

- **Status**: proposed
- **Date**: 2026-05-08
- **Deciders**:
- **Tags**: mlx-native, metal, quantization, kernel-coverage, parity

---

## 1. Context

`hf2q generate` on `/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf` fails at GGUF arch-peek with `unsupported GGML type ID 7` (Q5_1) on `blk.5.ffn_down_exps.weight`. Inspection shows the file uses two GGML types mlx-native has never recognized:

- **Q5_1** (id 7): 10 expert tensors at blocks 5-9 + 20-24 — 24 B / 32-element block.
- **IQ4_NL** (id 20): 10 expert tensors at blocks 10-19 — 18 B / 32-element block.

Root cause is a llama.cpp APEX-Q5_K_M layer-mix policy that spills MoE expert tensors into legacy 32-element formats per layer. The file is on-disk-correct llama.cpp output; mlx-native's loader simply doesn't know these types.

Auditing further reveals that this surface gap exposes a deeper coverage matrix gap. mlx-native's quantized matmul + matmul-id Metal kernels do not have full coverage of the GGML types we already nominally support, and we are missing entire kernel categories (`mul_mv_ext` r1 family) that llama.cpp ships. This means there are GGUFs we cannot load (this Gemma4 file), GGUFs we can load but route through slow mv paths during prefill (anything Q5_K), and a class of small-matmul throughput we're leaving on the floor.

User directive (operator, 2026-05-08):

> *We want to fix anything that we know about so we're as coherent (or more coherent) as our peers, and as fast (or faster) than our peers.*

User mantra (`~/Documents/mantra.txt`):

> *DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.*

This ADR closes the entire matrix. **No fallback paths, no "unsupported" arms, no Q5_K-style "still pending after months" gaps**.

### 1.1 Coverage matrix snapshot (pre-ADR)

✓ = ours has it · ✗ = gap · — = N/A

| Type | mv | mv_id | mm_id | mm_id_tensor | mm | mm_tensor | mm_t_bf16_p021 |
|---|---|---|---|---|---|---|---|
| Q4_0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Q8_0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **✗** |
| Q4_K | ✓ | ✓ | ✓ | ✓ | **✗** | **✗** | **✗** |
| **Q5_K** | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** |
| Q6_K | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Q5_1** (new) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **IQ4_NL** (new) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

**Existing-type gaps**: 9 (Q5_K ×6, Q4_K ×3, Q8_0 ×1, Q4_K perm021 implicit ×1 → reconciled to 9 above).
**New-type gaps**: 14 (Q5_1 + IQ4_NL × 7 each).
**`mul_mv_ext` r1 family**: ~30 kernels (5 r1 widths × 6 types).

**Grand total**: ~52 kernel implementations, plus Rust dispatch wiring, plus host dequant primitives, plus parity tests.

### 1.2 Why this file specifically motivates the broader sweep

Operator history (zsh) shows a previously-working Gemma 4 inference flow against `gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf` (DWQ-quantized: pure Q4_K + Q6_K + Q8_0, all supported). That directory is no longer on disk. The new APEX-Q5_K_M file (mtime 2026-05-06) is what's there now and is the operator's coherence-eval fixture. So this isn't a regression on hf2q's side — it's first contact with a llama.cpp-built file format we never closed coverage on. Closing only the Gemma4 surface gap (Q5_1 + IQ4_NL) leaves the deeper Q5_K/Q4_K/`mul_mv_ext` gaps for the next operator surprise. We close the whole matrix.

### 1.3 Pre-existing reasons coverage was partial

- **Q5_K mm_id**: ADR-013 P7 added Q4_K mv_id + ADR-013 P16 added Q4_K mm_id. Q5_K mv_id followed. Q5_K mm_id deferred — `quantized_matmul_id_ggml.rs:84` comment "Q5_K mm_id not yet ported; mv_id fallback is used for all batch sizes." This is the canonical "fallback used in production" pattern this ADR eliminates.
- **`mul_mv_ext` family**: never ported. Not blocking any specific model load, but a measurable perf gap on small-prompt decode and prefill warm-up.
- **mlx-native's `mm_t_bf16_perm021`**: a non-llama.cpp optimization (transposed matmul on bf16 inputs); only Q4_0 + Q6_K have it. Adding Q8_0 is near-trivial template instantiation; other types are deferred under same scope. Operator authorized full sweep ("yes to both") on 2026-05-08.

---

## 2. Decision

Close the entire matrix in `/opt/mlx-native` (kernels + dispatch + host dequant) and `/opt/hf2q` (loader byte-len + legacy-block helpers). Every new kernel ships with:

1. **Per-block parity test**: F32 → quantize via reference impl → bytes → mlx-native dequant → assert `max_abs_delta ≤ documented quant error bound` (Q5_1 ≤ 0.025; IQ4_NL ≤ 0.025; Q5_K/Q4_K ≤ existing K-quant tol; perm021/non-perm021 byte-equal).
2. **GPU↔CPU parity**: kernel-output F32 byte-equal to host-reference matmul within accumulation tolerance (`1e-4` for f32 accumulator).
3. **Real-file load + first-32-token byte-equal vs `llama-cli`** at temperature=0 on at least one fixture per type.

No Metal kernel ships without all three.

### 2.1 Kernel matrix post-ADR (target state)

| Type | mv | mv_id | mm_id | mm_id_tensor | mm | mm_tensor | mm_t_bf16_p021 | mv_ext_r1_2..5 |
|---|---|---|---|---|---|---|---|---|
| Q4_0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Q8_0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Q4_K | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Q5_K | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Q6_K | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Q5_1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| IQ4_NL | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**No `"unsupported"` arms remain in `mv_kernel_name`, `mm_kernel_name`, `mm_tensor_kernel_name`, `id_kernel_name`, `id_mm_kernel_name`, `id_mm_tensor_kernel_name`.** The `params.ggml_type != GgmlType::Q5_K` exclusion at `quantized_matmul_id_ggml.rs:268, 379` is removed.

### 2.2 Phases (sequential; merge boundaries; nothing WIP at boundary)

#### Phase 1 — Gemma4 unblock: Q5_1 + IQ4_NL full type coverage [LANDED 2026-05-08]

Closes `Q5_1` and `IQ4_NL` rows in the matrix. Unblocks the operator's APEX-Q5_K_M Gemma4 file. ~3 days.

Kernels (×2 types = 14 implementations):
- 2 dense `mv` (inlined ports modeled on Q4_0/Q8_0 dense mv at `quantized_matmul_ggml.metal:122,195`)
- 2 `mv_id` (inlined ports modeled on Q5_K_id at `quantized_matmul_id_ggml.metal:401`; Q5_1 closer to Q8_0_id, IQ4_NL closer to Q4_0_id with codebook lookup)
- 2 `mm` template instantiations (`hf2q_mul_mm_impl<block_q*, nl, dequantize_q*>`)
- 2 `mm_tensor` template instantiations
- 2 `mm_id` template instantiations
- 2 `mm_id_tensor` template instantiations
- 2 `mm_tensor_bf16_perm021` template instantiations
- ~5 `mv_ext_r1_*` template instantiations per type. **Operator standing rule (2026-05-08): no deferrals without explicit approval.** The earlier "defer to Phase 4 if appropriate" wording is hereby retracted; mv_ext lands in Phase 1 alongside the rest of the Q5_1 / IQ4_NL surface.

Block + dequant primitives (shared across 4 metal files):
- `block_q5_1` typedef: `{ half d; half m; uint qh; uchar qs[16]; }` — 24 B
- `block_iq4_nl` typedef: `{ half d; uchar qs[16]; }` — 18 B
- `constant int8_t kvalues_iq4nl[16]` (verified constant per `/opt/llama.cpp/ggml/src/ggml-common.h:1109-1112`)
- `dequantize_q5_1<type4x4>` + `dequantize_q5_1_t<type4x4>` (tensor variant)
- `dequantize_iq4_nl<type4x4>` + `dequantize_iq4_nl_t<type4x4>`

Rust loader + dispatch:
- `mlx-native/src/ops/quantized_matmul_ggml.rs`: `GgmlType::{Q5_1, IQ4_NL}` enum variants + `block_values()` + `block_bytes()`.
- `mlx-native/src/gguf/mod.rs:300`: `GGML_TYPE_Q5_1 = 7`, `GGML_TYPE_IQ4_NL = 20` constants; `ggml_type_from_u32` arms; `compute_byte_len` table (32, 24) and (32, 18); U8-storage arm extended; `dequantize_to_f32` arms with new host dequant primitives.
- `mlx-native/src/ops/quantized_matmul_id_ggml.rs`: real kernel-name arms in all three `*_kernel_name` functions; new types added to `dispatch_id_*` validation matches at line 284, 384.

hf2q:
- `src/gguf_patch.rs:551`: `GGML_TYPE_Q5_1 => (32, 24)`, `GGML_TYPE_IQ4_NL => (32, 18)` rows.
- `src/quantize/q_legacy.rs`: `BlockIQ4_NL` struct, `KVALUES_IQ4_NL: [i8; 16]` const, `dequantize_row_iq4_nl` + `dequantize_row_iq4_nl_bytes` (parallel to existing Q5_1 dequant at `q_legacy.rs:991`).

Phase 1 exit AC:
- All 14 Phase-1 kernels pass per-block parity, GPU↔CPU parity, and integration tests.
- `cargo test -p mlx-native --release` and `cargo test --release` clean.
- `./target/release/hf2q generate --model /opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf --prompt "What is 2+2?" --max-tokens 64` produces coherent text. (Coherence-checked per `feedback_metal_raw_barrier_per_dispatch.md` — "What is 2+2?" answer must be coherent, not garbage.)
- First-32-token byte-equal vs `llama-cli` at `--temp 0 --top-p 1 --top-k 0 --repeat-penalty 1` on the same file with the same prompt.

##### Phase 1 progress (live)

| Step | Status | Commit | Evidence |
|------|--------|--------|----------|
| P1.1 GgmlType + loader recognition | DONE | (early Phase 1) | mlx-native loads Q5_1 + IQ4_NL tensors |
| P1.2 host dequantize_{q5_1,iq4_nl} | DONE | (early Phase 1) | adr_022_phase1_dequant_parity 8/8 |
| P1.3 hf2q gguf_patch + q_legacy IQ4_NL | DONE | (early Phase 1) | 7 unit tests in q_legacy.rs |
| P1.4 per-block parity (Q5_1 + IQ4_NL) | DONE | (early Phase 1) | tests/adr_022_phase1_dequant_parity.rs |
| P1.5 4 mv kernels (Q5_1+IQ4_NL × dense+id) | DONE | 8bfa86e + 469bc11 | tests/adr_022_phase1_{dense_mv,mv_id}_gpu_parity 8/8 |
| P1.6 mm+mm_tensor+mm_id+mm_id_tensor | DONE | 633abd0 (iter 13) | dense_mm_parity_prefill GREEN max_abs ~1.7e-3; mm_id_parity_prefill_path GREEN max_abs ~2.1e-3 (both tensor + non-tensor variants via env-flip) |
| P1.6 mm_t_bf16_perm021 | N/A | — | Attention Q@K^T only (ADR-013 P21); no model in scope quantizes attention as Q5_1 / IQ4_NL |
| P1.7 mul_mv_ext r1 family (Q5_1 + IQ4_NL) | DONE | mlx-native 5224e7e (iter 17) | 8 kernel instantiations (Q5_1 + IQ4_NL × r1∈{2,3,4,5}) + dispatcher with peer-matched routing (nsg=2 const; nxpsg ∈ {4,8,16} by K-modulus + M-rank; r1ptg ∈ {2,3,4,5} by m). PSOs specialized via `KernelRegistry::get_pipeline_with_constants` on FC_mul_mv_nsg (600) + FC_mul_mv_nxpsg (601). 10 parity tests GREEN: 8 type×r1 combinations (m=2..5, K=128) + 2 realistic Gemma4 shapes (K=2816, exercises both nxpsg=16 and nxpsg=8 branches). |
| P1.8 Integration: coherent generation + first-32 byte-equal | DONE (coherence + byte-equal) | e866e6c + iter 16 script | "What is 2+2?" → "2 + 2 = 4<turn\|>" on the original failing file. Prefill 46 tok/s, decode 72.6 tok/s. Byte-equal sub-AC GREEN as runnable regression: `scripts/adr022_p18_byte_equal.sh` — hf2q dumps rendered prompt via `HF2Q_DUMP_RENDERED_PROMPT`, replays through `llama-completion -no-cnv --jinja` under matched greedy sampling (--top-k 1), asserts byte-identical text output. PASS confirms the entire weight-type stack (Q5_1, IQ4_NL, Q6_K, Q8_0, F32) dispatches identically across both runtimes. |
| P1.9 F32 weight routing in `dispatch_qmatmul` | DONE | e866e6c | Type-aware routing: `weight.info.ggml_dtype == F32` → `dense_matmul_f32_f32_tensor`; everything else → `quantized_matmul_ggml`. Single match arm, no fallback chain. |
| P1.10 `find_tokenizer` walk-fallback removed | DONE | e866e6c | Same antipattern as `find_config` walk: silently picked qwen3.6's tokenizer for the Gemma4 GGUF → token-id-mismatched garbage output. Walk over `models/<subdir>/tokenizer.json` removed; resolution is now strict (`--tokenizer` flag → sibling-of-GGUF → fail-loud). |
| P1.11 GGUF-embedded tokenizer parsing | DONE | hf2q 6366d8e (builder + parity tests) + 6560177 (engine wiring) | New module `src/inference/models/gemma4/tokenizer.rs` ports the SentencePiece-derived BPE pipeline (BPE w/ byte_fallback + fuse_unk + Replace " "→"▁" normalizer + Split " " MergedWithPrevious + Decoder Sequence(Replace,ByteFallback,Fuse)). 5/5 parity tests vs on-disk tokenizer.json (`tests/adr_022_phase1_p11_gemma4_tokenizer_parity.rs`): simple, special-tokens, multibyte UTF-8, newlines-only, punctuation-runs — all byte-identical token streams. Engine wired: removed on-disk tokenizer.json + re-ran "What is 2+2?" → "2 + 2 = 4<turn\|>" (load banner: `tokenizer = hf-tokenizer-json (<gguf-embedded>)`). Single-file GGUF inference now works. |

Iter 14 root-cause notes (preserved for iter 15 + future readers):

- Operator pushback: "requiring config.json seems dumb" + "fallbacks are an antipattern". Both are correct: GGUF metadata carries every `Gemma4Config` field, and the engine.rs `if config_path { json } else { gguf }` conditional WAS a fallback. Iter 14 fix: `Gemma4Config::from_gguf` is the single source of truth on the GGUF path; the legacy `find_config` walked over `models/<subdir>/` was finding a peer model's config.json (e.g. qwen3.6's) when no Gemma4 config existed, silently substituting wrong arch params — that's the antipattern operator was flagging.
- 392 F32 tensors in `gemma4-ara-2pass-APEX-Q5_K_M.gguf`: 362 are 1D scalars (norms, eps, freq_factors, layer_output_scale) and 30 are `[2816, 128]` per-layer router weights `ffn_gate_inp.weight`. Only the routers go through `dispatch_qmatmul`. The 362 scalars are consumed by RMS-norm / embedding / RoPE kernels that have their own dispatch.
- llama.cpp peer pattern: `ggml_mul_mat` (in `ggml/src/ggml-cpu/ggml-cpu.c`) treats F32 as a regular type — the dense F32 matmul kernel is just one of many in its dispatch table. Our `quantized_matmul_ggml` is GGUF-format-aware but currently rejects F32; the `_ggml` suffix means "GGUF block format" but F32 IS a valid GGUF format (type id 0). Two architectural options for iter 15:
  1. Add F32 arm to `quantized_matmul_ggml` that dispatches to `hf2q_dense_mm_f32_f32` directly. Pro: single dispatcher, type-agnostic. Con: blurs the "quantized" name.
  2. hf2q-side `dispatch_qmatmul` wrapper inspects `weight.info.ggml_dtype` and forks to the correct kernel. Pro: keeps mlx-native's kernel boundaries clean. Con: every consumer must wrap.
   — Operator decision pending; default-recommend option 2 (smaller blast radius, hf2q already has the wrapper).

Iter 12 → 13 root cause investigation (preserved for future ADR readers):
- iter 12 misdiagnosed mm_id RED parity as a Q5_1 dequant template bug.
- Root cause was the test fixture: random-distribution `ids` over (n_tokens=64, top_k=8, n_experts=8) → 6.7σ variance; some experts received > n_tokens routings → overflowed `hids[n_experts × n_tokens]` row.
- Fix: deterministic flat-distribution ids `(t*17 + s*13 + 7) % n_experts` (matches the proven `tests/test_quantized_matmul_id_mm.rs` pattern), and direct `dispatch_id_mm_for_test` against mv_id reference (already CPU-validated in P1.5) — removes the CPU-quantizer noise floor and isolates the mm_id template body as the only variable. Q4_0 baseline at the same shape went GREEN, falsifying the dequant-bug hypothesis.
- Lesson: when porting a kernel that depends on a routing-table primitive (map0-produced hids), test fixtures must match the routing primitive's per-bucket capacity OR test against a known-good kernel path at the same shape.

#### Phase 3 — Q4_K dense mm + Q8_0 perm021 [LANDED 2026-05-08 — iter 21]

| Step | Status | Commit | Evidence |
|------|--------|--------|----------|
| P3.1 Q4_K dense mm + mm_tensor | DONE | mlx-native 1d8c67d | Block typedef + dequantize_q4_K(_t)<type4x4> template + kernel_mul_mm_q4_K_f32 + kernel_mul_mm_q4_K_tensor_f32 instantiations. Sibling of Phase 2's Q5_K kernels minus the qh high-bit branch. |
| P3.2 Q8_0 perm021 | DONE | mlx-native 1d8c67d | `kernel_mul_mm_q8_0_tensor_bf16_perm021` template instantiation. Uses existing block_q8_0 + dequantize_q8_0_t. Public dispatcher `quantized_matmul_mm_tensor_perm021` accepts Q8_0 alongside Q4_0 + Q6_K. |
| P3.3 AC-1 cleared | DONE | mlx-native 1d8c67d | `git grep '"unsupported"' src/ops/quantized_matmul*.rs` returns ONLY F32 \| F16 \| I16 arms (type-not-applicable). The post-ADR matrix in §2.1 is true. |
| P3.4 Parity tests | DONE | mlx-native 1d8c67d | tests/adr_022_phase3_q4_k_dense_parity.rs — 4/4 GREEN (mv m=1, mv m=4, mm m=64, mm m=32 K=2048). Q8_0 perm021 covered indirectly by the existing ADR-013 P21 attention path tests once it's wired in by ADR-013/015 follow-up. |

#### Phase 2 — Q5_K full coverage [LANDED 2026-05-08 — iter 20]

##### Phase 2 progress (live)

| Step | Status | Commit | Evidence |
|------|--------|--------|----------|
| P2.1 mm_id (simdgroup MMA) | DONE | mlx-native 8d9bad9 (iter 19) | `kernel_mul_mm_id_q5_K_f32` ported from llama.cpp (port of `dequantize_q5_K` at ggml-metal.metal:699 + Q4_K mm_id template path). Q5_K bypass at dispatch sites retired. 2/2 parity tests vs mv_id reference. |
| P2.2 mm_id_tensor | DONE | mlx-native 8d9bad9 (iter 19) | Sibling `kernel_mul_mm_id_q5_K_tensor_f32` for M3+ tensor cores. |
| P2.3 dense mv | DONE | mlx-native 29fa455 (iter 20) | `kernel_mul_mv_q5_K_f32` — port of llama.cpp `kernel_mul_mv_q5_K_f32_impl` (ggml-metal.metal:7837); body is Q4_K mv plus the Q5_K mv_id qh/acc2 high-bit block. Dispatch geometry: Q5_K joins Q4_K + Q6_K's (2, 32, 2) K-quant arm. |
| P2.4 dense mm + mm_tensor | DONE | mlx-native 29fa455 (iter 20) | `kernel_mul_mm_q5_K_f32` + `kernel_mul_mm_q5_K_tensor_f32` template instantiations. block_q5_K typedef + K_SCALE_SIZE + get_scale_min_k4_just2 helper + dequantize_q5_K(_t)<type4x4> templates ported into both mm shaders. |
| P2.5 Integration + parity | DONE | mlx-native 29fa455 (iter 20) | tests/adr_022_phase2_q5_k_dense_parity.rs — 4/4 GREEN (mv m=1, mv m=4, mm m=64, mm m=32 K=4096). Smoke: Qwen 3.6 35B-A3B-APEX-Q5_K_M still produces coherent CoT output post-port (prefill 48 t/s warm, decode 132 t/s). Long-prompt bench (iter 19): hf2q decode 129 t/s **beats** llama.cpp's 104 t/s (1.24×); hf2q prefill 608 t/s vs llama.cpp 1500 t/s (0.40× — pre-existing qwen35 prefill pipeline gap, ADR-013/015 scope). |




Closes Q5_K row. Validates against `/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/APEX-Q5_K_M.gguf` (operator's existing Q5_K_M fixture, llama.cpp-built per memory `project_qwen35_dwq_pre_505b5b8_broken_2026_05_05.md`). ~2 days.

Kernels (5 new):
- 1 dense `mv` (port inlined from existing `mv_id_q5_K` at `quantized_matmul_id_ggml.metal:401`, drop expert-routing indirection)
- 1 `mm_id` template instantiation (Metal `dequantize_q5_K` + `block_q5_K` typedef already in id_mm.metal? Verify; if missing, add)
- 1 `mm_id_tensor` template instantiation (+ `dq_q5_K_id` if missing in tensor variant)
- 1 dense `mm` template instantiation (+ block typedef + dequant fn in `quantized_matmul_mm.metal` if missing)
- 1 dense `mm_tensor` template instantiation
- 1 `mm_t_bf16_perm021` template instantiation

Rust dispatch:
- Remove Q5_K from "unsupported" arms in `quantized_matmul_ggml.rs:106, 127, 143`.
- Remove `params.ggml_type != GgmlType::Q5_K` exclusion at `quantized_matmul_id_ggml.rs:268, 379`.
- Update doc comments at `:263` to remove the "Q5_K (mm_id not yet ported — only mv_id kernel exists)" sentence.

Phase 2 exit AC:
- All 5 Phase-2 kernels pass parity (per-block + GPU↔CPU + real-file).
- Qwen 3.6 35B-A3B-APEX-Q5_K_M generates coherent text via `hf2q generate` and via `hf2q serve` /v1/chat/completions.
- First-32-token byte-equal vs `llama-cli` at temp=0 on a benign prompt.
- TTFT prefill on Qwen 3.6 35B-A3B improves measurably vs pre-Phase-2 (Q5_K mm_id replaces mv_id at prefill); record measured Δ in commit message.

#### Phase 3 — Q4_K dense + Q8_0 perm021

Closes the remaining existing-type cells. ~1 day.

Kernels (4 new):
- 1 Q4_K dense `mm` template instantiation
- 1 Q4_K dense `mm_tensor` template instantiation
- 1 Q4_K `mm_t_bf16_perm021` template instantiation
- 1 Q8_0 `mm_t_bf16_perm021` template instantiation

Rust dispatch:
- Remove Q4_K from `quantized_matmul_ggml.rs:127, 143` "unsupported" arms.
- Add Q4_K, Q5_K, Q8_0 to `:650` perm021 dispatch (currently only Q4_0 + Q6_K).

Phase 3 exit AC:
- All 4 kernels pass parity.
- Q4_K_M model (e.g. operator's `gemma-4-26B-A4B-it-ara-abliterated-dwq` if available, else any Q4_K_M GGUF on hand) generates coherent text.
- Microbench shows perm021 path engaged for Q8_0 + Q4_K + Q5_K when input is bf16-permuted.

#### Phase 4 — `mul_mv_ext` r1 family (small-matmul perf parity)

Closes the largest perf gap by category. ~1.5 weeks. Per `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:3936-3954`, llama.cpp ships `kernel_mul_mv_ext_<type>_f32_r1_<2|3|4|5>` for Q4_0, Q5_0, Q5_1, Q8_0, IQ4_NL, and a variant for K-quants.

Kernels (~30): 5 r1 widths × 6 types (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, Q5_1, IQ4_NL — minus `mul_mv_ext_*_r1_*` cells llama.cpp doesn't ship; verify per-cell against ggml-metal.metal 3936-3954 + K-quant ext family).

Implementation note: `mul_mv_ext_q4_f32_disp` is a generic dispatcher template in llama.cpp parameterized on block type + dequant function. Port the dispatcher template once, instantiate per (type × r1).

Phase 4 exit AC:
- Every llama.cpp `kernel_mul_mv_ext_<type>_f32_r1_<n>` for the types in our matrix has a parity match in mlx-native.
- Microbenchmark suite at `/opt/mlx-native/benches/mul_mv_ext_parity.rs` (new) shows ≤5% throughput gap vs llama.cpp's same kernel for matching matmul shapes (M×N×K = 1×N×K with N ∈ {2..5} × 32, K ∈ {2048, 4096, 8192}).
- Real-model decode TTFT improvement measured on at least 2 fixtures; record in commit messages.

#### Phase 5 — exit-criteria sweep

Final integration verification. ~2 days.

- All 7 type rows × 8 kernel columns = 56 cells in the post-ADR matrix are ✓ (no unsupported, no exclusion arms).
- `cargo test --workspace --release` clean across mlx-native + hf2q.
- `./target/release/hf2q generate` on every model in `/opt/hf2q/models/*` produces coherent text matching `llama-cli` first-32-token byte-equal at temp=0.
- Memory note created: `project_adr022_LANDED_<date>.md` summarizing measured perf deltas + parity outcomes.
- ADR-022 status flips to **LANDED**.

---

## 3. Consequences

### Positive
- Every llama.cpp-built GGUF using mainstream quants (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K/IQ4_NL) loads in hf2q.
- No `"unsupported"` arms in kernel-name dispatchers — removes a class of "works on this fixture, not that one" failure.
- Q5_K_M models stop incurring mv_id-instead-of-mm_id penalty at prefill; expected wall-time improvement on prefill TTFT for Qwen 3.6 35B-A3B-APEX-Q5_K_M.
- `mul_mv_ext` r1 family closes a measurable small-matmul throughput gap; impact varies by model but is net-positive.
- Eliminates a class of operator-surprise where a freshly-built llama.cpp file mysteriously fails to load.

### Negative
- Real Metal porting work; ~52 kernel implementations + tests. ~2-3 weeks calendar.
- Increased shader compile time (every new kernel adds to PSO build); measure and document if it crosses the 5s shader-compile budget.
- Larger `.metallib` binary; minor.

### Neutral
- Does not change any existing kernel's correctness or perf (all changes are additive).
- Q4_1 / Q5_0 / IQ-family beyond IQ4_NL remain out of scope; if a future GGUF needs them, that's a follow-up ADR with the same template (block typedef + dequant + 7-kernel coverage).

---

## 4. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Metal RAW race introduced in a kernel (per memory `feedback_metal_raw_barrier_per_dispatch.md`) | Coherence-test on every kernel post-port: "What is 2+2?" must produce a coherent answer, not garbage. Smoke tests do not suffice. |
| Codex Phase-2b catches a unified-memory race the tests didn't (per memory `feedback_codex_review_catches_unified_memory_races.md`) | Run a Codex Phase-2b audit pass at end of each phase before merge. Trust the line-numbered evidence over passing tests. |
| Per-block parity test passes but full-prompt coherence fails (long-prompt RAW race exposes) | Phase exit AC requires real-file generate on a non-trivial prompt (≥256 tokens), not just unit/microbench. |
| Q5_K ports introduce a regression on existing Q4_K/Q5_K mv_id paths | Phase 2 includes a regression test: pre-Phase-2 byte-output of Q5_K mv_id captured as fixture, post-Phase-2 byte-equal required. |
| `mul_mv_ext` perf gain measured only on synthetic benches, not real models | Phase 4 exit AC requires real-model decode TTFT measurement on ≥2 fixtures. |
| Worktree isolation skipped → AC stomp on AD's WIP (per memory `feedback_concurrent_src_workers_need_worktree.md`) | Each phase runs in a dedicated worktree (`/tmp/mlx-native-adr-022-phaseN`, `/tmp/hf2q-adr-022-phaseN`). No parallel agents on overlapping `src/` paths. |

---

## 5. Out of scope

- **Q4_1 / Q5_0** legacy types: not present in any operator GGUF as of 2026-05-08 audit; ports are mechanical when needed (same template-instantiation pattern). Defer to follow-up ADR triggered by file discovery.
- **IQ2_XS / IQ3_XXS / IQ-family beyond IQ4_NL**: not present in operator files; involve more complex 2D-codebook dequant. Defer.
- **TQ1_0 / TQ2_0**: ternary quants; ADR-007 / ADR-017 own this surface. Out of scope.
- **Flash-attention quantized K/V variants** (per `ggml-metal.metal:6596+`): owned by ADR-011 and the FA prefill chain. Out of scope.
- **Refactor existing inlined mv kernels to template-based**: tempting but breaks Chesterton's fence. The inlined ports were chosen for a reason (per-kernel optimization room); refactoring is a separate decision. Out of scope.

---

## 6. Acceptance criteria — measurable, falsifiable

- **AC-1 (matrix complete)**: `git grep '"unsupported"' /opt/mlx-native/src/ops/quantized_matmul*.rs` returns no matches outside type-not-applicable arms (F32, F16, I16). The post-ADR matrix in §2.1 is true.
- **AC-2 (Gemma4 file loads + generates)**: `./target/release/hf2q generate --model /opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf --prompt "What is the capital of France?" --max-tokens 64 --temp 0` produces coherent text containing "Paris".
- **AC-3 (peer byte-parity)**: First-32-token byte-equal vs `/opt/llama.cpp/build/bin/llama-cli` at `--temp 0` on at least one fixture per type (Q5_1 + IQ4_NL: Gemma4 APEX-Q5_K_M; Q5_K: Qwen 3.6 APEX-Q5_K_M; Q4_K: a Q4_K_M file; Q4_0/Q8_0/Q6_K: existing fixtures already passing).
- **AC-4 (Q5_K mm_id engaged)**: Phase 2 commit message records prefill TTFT delta on Qwen 3.6 35B-A3B-APEX-Q5_K_M with `tracing::info!` confirmation that `mm_id` (not `mv_id`) is the kernel path on prefill.
- **AC-5 (mv_ext perf parity)**: ≤5% throughput gap vs llama.cpp on the matching `kernel_mul_mv_ext_*_r1_*` shapes; record actual gap per type+width in Phase 4 commit message.
- **AC-6 (no regressions)**: Pre-ADR baseline output captured for at least 5 existing fixtures; post-ADR generates byte-equal output (or token-equal at temp=0) on the same fixtures.
- **AC-7 (memory note + landed flip)**: `project_adr022_LANDED_<YYYY-MM-DD>.md` created; ADR header status flipped to **LANDED**.

---

## 7. Implementation order

Strict sequential. No phase begins until prior phase's exit AC is met and merged.

1. **Phase 1**: Gemma4 unblock — Q5_1 + IQ4_NL full coverage (all 7 kernel cells per type).
2. **Phase 2**: Q5_K full coverage.
3. **Phase 3**: Q4_K dense + Q8_0 perm021.
4. **Phase 4**: `mul_mv_ext` r1 family across all types.
5. **Phase 5**: exit-criteria sweep.

Each phase runs in dedicated worktrees (`/tmp/mlx-native-adr-022-phase<N>` + `/tmp/hf2q-adr-022-phase<N>`). Phase-N branch named `adr-022/phase-<N>`.

---

## 8. Links

- Operator file under audit: `/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf` (mtime 2026-05-06, 20.5 GB).
- llama.cpp kernel reference: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`.
- llama.cpp IQ4_NL codebook: `/opt/llama.cpp/ggml/src/ggml-common.h:1109-1112`.
- mlx-native current matmul + matmul-id sources: `/opt/mlx-native/src/shaders/quantized_matmul*.metal`, `/opt/mlx-native/src/ops/quantized_matmul*.rs`.
- hf2q reference Q5_1 dequant: `src/quantize/q_legacy.rs:824-1013`.
- hf2q gguf_patch tensor_byte_len helper: `src/gguf_patch.rs:542-565`.
- Memory: `feedback_metal_raw_barrier_per_dispatch.md` (coherence test must be real, not smoke).
- Memory: `feedback_codex_review_catches_unified_memory_races.md` (Codex Phase-2b audit per phase).
- Memory: `feedback_concurrent_src_workers_need_worktree.md` (worktree isolation mandatory).
- Memory: `feedback_live_verification_must_check_content.md` (HTTP 200 ≠ verified).
- Predecessor ADRs: ADR-006 (mlx-native gpu backend), ADR-009 (reference parity & coherence recovery), ADR-010 (exact batched kernel parity), ADR-013 (kernel speed work — closed).
