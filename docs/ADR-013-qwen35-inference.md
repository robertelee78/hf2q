# ADR-013: Qwen3.5 / Qwen3.5-MoE (qwen35 + qwen35moe) Inference Support — Pure-Rust Forward Pass + Metal Kernels

**Status:** Proposed (2026-04-23)
**Decision Makers:** Robert, Claude
**Related ADRs:** ADR-004 (GGUF compatibility), ADR-005 (inference server), ADR-006 (mlx-native GPU backend), ADR-008 (candle divorce), ADR-009 (reference parity and coherence recovery), ADR-010 (exact batched kernel parity), ADR-011 (flash-attn prefill port), **ADR-012 (qwen35 + qwen35moe conversion — companion ADR)**
**Related memories:** `project_qwen36_architecture.md`, `project_model_class_split.md`, `project_pure_rust_crate_factory.md`, `project_mlx_native_is_the_strategic_destination.md`, `feedback_hf2q_sovereignty.md`, `feedback_llama_cpp_over_candle.md`, `feedback_mantra.md`, `feedback_walk_means_port_llama_cpp_to_rust.md`, `feedback_prove_in_code.md`, `feedback_no_broken_windows.md`, `feedback_no_shortcuts.md`, `feedback_correct_outcomes.md`, `reference_decode_benchmark_methodology.md`, `project_decode_parity_achieved.md`, `project_end_gate_reality_check.md`

---

## Engineering Mantra (load-bearing — read before every session)

Source: `~/Documents/mantra.txt` (Robert, undated). Quoted verbatim.

> **DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.**

**Operational reading for this ADR:**

- **No fallback.** When a Qwen3.5/3.6-specific kernel or op is required (GATED_DELTA_NET, TRI_SOLVE, CumSum, L2_NORM, IMROPE), it lands as a real Metal shader in `/opt/mlx-native` with a correct spec-driven test. No CPU-only "we'll GPU-ify later" paths in the shipped hot loop; no `unimplemented!()`; no stubbed autoregressive-path that silently degrades quality. The **decomposed-autoregressive path in Rust on CPU is allowed for debug / parity cross-check only** (it never executes in a production inference call; it exists solely to validate the fused Metal kernel).
- **Chesterton's fence.** The current Gemma-4 inference surface (`src/serve/forward_mlx.rs`, `src/serve/config.rs`, `src/backends/gguf.rs:1469`) encodes heterogeneous-attention dispatch, mixed head-dim per layer, V=K tying, frequency-factor-masked RoPE, 7-norm per-layer stack, and MoE expert stacking — all explicitly for Gemma-4's mixed sliding-vs-global + MoE shape. Before adding any dispatch for Qwen3.5, fully understand *why* each Gemma-shaped branch exists; do not generalize Gemma's `LayerType::{Sliding, Full}` into a shared enum that loses Gemma-specific semantics. Per `project_model_class_split.md`: Qwen3.5 gets its own per-variant layer-kind enum in its own module.
- **Dive deep.** The authoritative reference for every decision in this ADR is the llama.cpp tree at `/opt/llama.cpp` (commit `8bc492ebb` as of 2026-04-24): `src/models/qwen35.cpp` (dense graph builder), `src/models/qwen35moe.cpp` (MoE graph builder), `src/models/delta-net-base.cpp` (Gated DeltaNet — three paths: chunked, autoregressive, fused), `ggml/src/ggml-metal/ggml-metal.metal` (`kernel_gated_delta_net_impl`), `ggml/src/ggml-metal/ggml-metal-ops.cpp:1584-1657` (dispatch). Every TODO in this ADR cites a specific file:line range. Read them. Additionally, for the sigmoid-vs-swish output-gate question: HF transformers `modeling_qwen3_5.py:689` and vLLM `qwen3_next.py:312-314` are the authoritative tiebreakers — both apply `torch.sigmoid(gate)`, not swish. The llama.cpp implementation is correct; the HF config's `output_gate_type: "swish"` field is a vestigial label the transformers code does not read.
- **Absolute sovereignty (`feedback_hf2q_sovereignty.md`).** Pure Rust; hf2q and mlx-native are the only repos in our deliverables. No code, Cargo crates, utils, binaries, or derivative artifacts from `/opt/candle` or `/opt/llama.cpp` enter our deliverables — candle is a Rust crate we don't own, same rule as llama.cpp's Python and C++. This applies at build time, test time, and CI time. **We port the best ideas, borrow math and logic, but we write the Rust and Metal ourselves, in our repos.** Reading `qwen35.cpp` / `delta-net-base.cpp` / `ggml-metal.metal` to derive the mathematical specification for IMROPE section rotation, Gated DeltaNet state-update, tri-diagonal decay masking, etc., is legitimate — that's what reference repos are for. Copying a .metal kernel source file, linking against `libllama`, depending on a `candle-*` crate, or using llama.cpp binary output as a test oracle is not. Correctness proofs derive from the specification, small hand-authored fixtures, and — for byte-parity end-gates — llama.cpp's *live output on the same input* treated as an authoritative black-box (Robert's eyes looking at it is fine; a CI assertion target it is not).

---

## Context

### Business problem

Robert needs Qwen3.6 inference running on the hf2q + mlx-native stack on Apple Silicon. The immediate on-disk target is `/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf` (25 GB, `general.architecture = qwen35moe`, produced externally by convert_hf_to_gguf.py + llama-quantize and treated as a *convenience reference for eyes*, not a CI asset — per sovereignty rule). The 27B dense variant (`qwen35`) shares ≥90% of the inference surface and is scoped jointly.

Qwen3.6 is architecturally identical to Qwen3.5 and routes through llama.cpp's two architecture constants:

- **Dense variant** — `Qwen3_5ForCausalLM` → `LLM_ARCH_QWEN35` (`general.architecture="qwen35"`)
- **MoE variant** — `Qwen3_5MoeForCausalLM` → `LLM_ARCH_QWEN35MOE` (`general.architecture="qwen35moe"`)

### Technical problem

The existing hf2q serve/generate path (`src/serve/forward_mlx.rs`) is Gemma-4-shaped. Its `LayerType` enum is `{Sliding, Full}` — semantically the wrong axis for Qwen3.5 (which is `{LinearAttention, FullAttention}`). Its op sequence assumes:

- Mixed head_dim per layer (sliding=256, global=512) — Qwen3.5 is uniform 256.
- Standard RoPE with a frequency-factor mask — Qwen3.5 uses multi-section **interleaved** RoPE (IMROPE) with 3 sections `[11, 11, 10]` and `rope_theta=10,000,000`.
- Q·K softcap config option — Qwen3.5 has none.
- V=K tying on global layers — Qwen3.5 loads Q, K, V separately on full-attention layers.
- Dense decoder blocks with MoE-only FFN variance — Qwen3.5 has a *second* novel attention variant (Gated DeltaNet / linear attention) that shares tensor names (`attn_qkv`, `attn_gate`) with full-attention but means something completely different.

`/opt/mlx-native/src/ops/` has 40+ ops and `/opt/mlx-native/src/shaders/` has 29+ `.metal` kernels, **none of which implement the Qwen3.5-novel primitives**: no `gated_delta_net`, no `tri_solve`, no `cumsum`, no `l2_norm`, no `ssm_conv` (distinct from the Mamba-style conv already needed), no `rope_multi` with IMROPE mode. `/opt/hf2q/src/inference/models/` exists as empty scaffolding — there is no per-model forward-pass file registered today; Gemma-4's forward lives in `src/serve/`, not `src/inference/models/`. This ADR is the first population of `src/inference/models/`.

### Current state inventory (what exists today)

Direct reads on 2026-04-23:

| Component | State |
|---|---|
| `src/inference/models/` | **Empty.** No files. This ADR creates its first contents. |
| `src/serve/forward_mlx.rs` | Gemma-4-shaped: `is_sliding` per-layer branch, `head_dim_for_layer(i)`, frequency-factor mask, MoE 128-experts/top-8 routing. Load-bearing, do not refactor for Qwen3.5 — add sibling module instead. |
| `src/serve/config.rs` | `Gemma4Config` with `LayerType::{Sliding, Full}` enum. Do not generalize; new `Qwen35Config` lives in `src/inference/models/qwen35/`. |
| `src/backends/gguf.rs:1469` | Arch-aware tensor naming handles `gemma4`, `llama`, `mistral`, `qwen2`, `qwen3`, `phi`. **No `qwen35` and no `qwen35moe` branches** — Decision 1 adds both. |
| `src/preflight.rs::SUPPORTED_LAYER_TYPES` | Per ADR-012 Decision 3, `linear_attention` is not yet in the list. ADR-012 adds it. We depend on that. |
| `/opt/mlx-native/src/ops/` | 40+ ops; missing GATED_DELTA_NET, TRI_SOLVE, CUMSUM, L2_NORM, SSM_CONV (Qwen variant), ROPE_MULTI (IMROPE). |
| `/opt/mlx-native/src/shaders/` | 29 `.metal` shaders; missing shaders for all of the above. |

### Primary-source hparam summary (source: `project_qwen36_architecture.md`; re-verified 2026-04-23 against HF config.json)

Shared by both variants:

| Field | Value |
|---|---|
| `head_dim` | 256 |
| `linear_key_head_dim` / `linear_value_head_dim` | 128 |
| `linear_num_key_heads` | 16 |
| `linear_conv_kernel_dim` | 4 |
| `full_attention_interval` | 4 |
| `layer_types` | explicit per-layer string array (`"linear_attention"` / `"full_attention"`) |
| `partial_rotary_factor` | 0.25 → rotary_dim = 64 |
| `rope_theta` | 10,000,000 |
| `mrope_section` | `[11, 11, 10]` (sum = 32 = rotary_dim / 2) |
| `mrope_interleaved` | true |
| `max_position_embeddings` | 262,144 |
| `rms_norm_eps` | 1e-6 |
| `vocab_size` | 248,320 |
| `attn_output_gate` | true |
| `mamba_ssm_dtype` | float32 |
| `mtp_num_hidden_layers` | 1 |
| `dtype` | bfloat16 |

Variant-specific:

| Field | 27B dense (qwen35) | 35B-A3B MoE (qwen35moe) |
|---|---|---|
| `hidden_size` | 5120 | 2048 |
| `num_hidden_layers` | 64 | 40 |
| `num_attention_heads` | 24 | 16 |
| `num_key_value_heads` | 4 (GQA 6:1) | 2 (GQA 8:1) |
| `linear_num_value_heads` | 48 | 32 |
| `intermediate_size` | 17408 (dense SwiGLU) | — |
| `moe_intermediate_size` | — | 512 |
| `num_experts` / `num_experts_per_tok` | — | 256 / 8 |
| `shared_expert_intermediate_size` | — | 512 (YES shared experts) |

### Empirical reference baseline (2026-04-23, M5 Max, llama.cpp build `b8914-8bc492ebb`, Metal GPU)

Canonical inference on the local 25 GB apex GGUF:

```
Prompt prefill : 364.8 tok/s
Decode         :  97.3 tok/s
Output         : coherent, grammatical English reasoning output
Metal memory   : 29.8 GB (23.9 model + 5.2 context + 776 MB compute)
```

These are our **match-or-beat targets** for the hf2q port, consistent with `project_end_gate_reality_check.md` (re-measure on the day of the end-gate check). The output produced by this run is our **sourdough byte-parity oracle** for correctness (Decision 15) — treated as a black-box deterministic function from (GGUF, prompt, seed=42, temp=0.0) to token stream, NOT as a checked-in fixture.

---

## Strategic Decision

**hf2q implements Qwen3.5 (dense) and Qwen3.5-MoE forward-pass inference natively in pure Rust, backed by new Metal kernels in `/opt/mlx-native`. llama.cpp is the spec source for every op we write; no llama.cpp code, binary, or output enters our deliverables at build, test, or CI time. Both variants ship in one ADR because they share ≥90% of the inference surface.**

This is the application of `feedback_hf2q_sovereignty.md` to the inference question: we port the best ideas — the math of Gated DeltaNet, the section-interleaved rotation of IMROPE, the tri-diagonal decay-mask solve, the gated-output full-attention shape — but we write every Rust struct, every .metal shader, every test harness ourselves, in our repos. The on-disk apex GGUF is consumed (it's a valid Qwen3.5-MoE GGUF per Robert's external conversion today, and per hf2q's own convert path post-ADR-012) but the llama.cpp binary that produced our reference baseline is not a dependency of any hf2q test — llama.cpp is always an *external* tool we run at the end-gate, never linked or imported.

**Joint dense + MoE scope** (parallel to ADR-012's rationale): the linear-attention sub-layer, the gated full-attention sub-layer, MROPE, MTP, tokenizer dispatch, and every mlx-native kernel are bit-identical across the two variants. Only the FFN block differs (dense SwiGLU vs. 256-expert routing + shared expert). Splitting into ADR-013 (MoE) + ADR-014 (dense) would duplicate 90% of the spec.

---

## Non-Goals

Each of these is explicitly *not* in this ADR's scope.

1. **Conversion pipeline (HF safetensors → DWQ GGUF).** Owned by ADR-012. This ADR consumes GGUFs; it doesn't produce them. Cross-ADR coordination point: the `ActivationCapture` trait defined in ADR-012 Decision 13, implemented here in Decision 10.
2. **Multimodal vision tower.** Qwen3.6-27B dense ships with a 27-layer ViT (patch_size=16, hidden_size=1152). The 35B-A3B on-disk target has `mmproj-qwen36-F16.gguf` alongside the main GGUF but the text config drops vision. This ADR scopes text-only inference for both variants. Vision is a follow-up ADR that can layer on top.
3. **MTP speculative-decoding execution.** `mtp_num_hidden_layers: 1` is present in both variants' configs; the corresponding GGUF tensors are loaded but unused in the v1 hot path. Implementing speculative decoding (MTP head prediction + verification loop) is a follow-up ADR. This ADR ensures MTP weights load without error and the main forward produces correct output *without* MTP execution. (The on-disk apex GGUF has MTP tensors stripped per the 2026-04-23 gguf-dump — but conversion via ADR-012 will emit them, so inference must load them gracefully.)
4. **Tokenizer runtime changes beyond what GGUF-embedded metadata provides.** The Qwen3.5 `gpt2`-family BPE (pre-type `qwen35`, vocab 248,320) is loaded from the GGUF's `tokenizer.ggml.*` keys and `tokenizer.chat_template`. If hf2q's existing tokenizer pipeline requires a new pre-tokenizer regex for `pre: qwen35`, we add it; if it requires a new chat-template engine, that's a follow-up ADR.
5. **Dynamic KV-cache quantization (TQ / TurboQuant) for Qwen3.5's hybrid KV state.** ADR-007 is TurboQuant for Gemma; per `project_tq_state_2026_04_21.md` it's gated off pending C-0 audit. We ship Qwen3.5 inference with dense F32 KV cache for full-attention layers and dense F32 SSM recurrent state for linear-attention layers — matching the llama.cpp default. Extending TQ to Qwen3.5 is out of scope here.
6. **Multi-GPU / multi-device inference.** Single M-series device. llama.cpp's `auto_fgdn` fallback-on-device-mismatch logic is not ported; we target a single-device hot path.
7. **Performance optimization beyond match-or-beat llama.cpp baseline.** Beating 97.3 tok/s decode / 364.8 tok/s prompt on M5 Max is the ship bar. Going substantially faster (e.g., 2× llama.cpp) is a follow-up ADR after we're first coherent then at-parity.
8. **Qwen3-family-coverage beyond qwen35 / qwen35moe.** Qwen3, Qwen3-MoE, Qwen3-VL, Qwen3Next, Qwen3-Omni, Qwen3-Coder are out of scope. If the linear-attention machinery proves reusable for Qwen3Next (likely), that's a follow-up port ADR.

---

## Reference Implementations (authoritative — read for spec, do not copy)

Every decision in this ADR has an upstream reference. All file:line citations are into `/opt/llama.cpp` at commit `8bc492ebb` (2026-04-24). Verify against current HEAD before starting each phase.

### Graph builders

1. `src/models/qwen35.cpp` — dense variant (~15 KB). Per-layer dispatch on `hparams.is_recurrent(il)`: full-attention branch lines 117-196, linear-attention branch delegates to `delta-net-base.cpp`. Per-layer residual at line 49, post-attention norm at line 56.
2. `src/models/qwen35moe.cpp` — MoE variant (~17 KB). Same structure as qwen35.cpp but FFN is `build_moe_ffn(...)` + shared expert at lines 406-420 (shared expert is **gated** by `ffn_gate_inp_shexp` with sigmoid, then added to routed output — contradicting an earlier assumption that shared experts are always-on).
3. `src/models/delta-net-base.cpp` — Gated DeltaNet shared by both variants (~16 KB). Three inference paths:
   - `build_delta_net_chunking()` lines 15-286 (prefill; ~17 heavy ops: matmul, solve_tri, cumsum, tri-diagonal decay mask)
   - `build_delta_net_autoregressive()` lines 288-370 (decode; ~2 heavy ops, 10 light)
   - `build_delta_net_fused()` lines 372-421 (single `ggml_gated_delta_net` op wrapper)
   - Dispatcher lines 423-445: `cparams.fused_gdn_ar` / `cparams.fused_gdn_ch` default to `true` (fused is llama.cpp's production default; decomposed paths are fallbacks).

### Metal kernel

4. `ggml/src/ggml-metal/ggml-metal.metal` — `kernel_gated_delta_net_impl<NSG>` (4 NSG variants: 1, 2, 4, 8). ~300 Metal lines. Spec source for our mlx-native `gated_delta_net.metal`.
5. `ggml/src/ggml-metal/ggml-metal-ops.cpp:1584-1657` — dispatch (`ggml_metal_op_gated_delta_net`). Thread-group shape, pipeline selection, buffer binding order.

### MROPE (IMROPE mode)

6. `ggml/include/ggml.h:1847-1861` — `ggml_rope_multi(...)` signature.
7. `ggml/src/ggml.c:4183-4201` — `ggml_rope_multi()` wrapper over `ggml_rope_impl`.
8. `ggml/src/ggml-cpu/ops.cpp:5643-5711` — `ggml_mrope_cache_init` with the `is_imrope` branch (`mode == GGML_ROPE_TYPE_IMROPE == 40`) and the `sector % 3`-cycling section-to-theta assignment.
9. `ggml/src/ggml-cpu/ops.cpp:5714-5731` — `rotate_pairs<T>` inner loop; NeoX-style `(x[i], x[i + n_dims/2])` pair indexing.
10. `src/models/qwen35.cpp:162-172` — Qwen3.5's `ggml_rope_multi` call site with sections `[11, 11, 10, 0]`, mode IMROPE, freq_base 10^7.

### Gated full-attention output gate

11. `src/models/qwen35.cpp:117-196` — full-attention block. Q+gate fused in `wq` (output dim `2 × head_dim × n_head`), gate extracted via `ggml_view_3d` at lines 152-157, activation `ggml_sigmoid(gate)` at line 186, merge `cur = ggml_mul(ctx0, cur, gate_sigmoid)` at line 189, then `wo` projection at line 194.
12. `src/llama-model.h:213-265` — `llama_layer` struct fields: `wq` (shape `[head_dim*2*n_head, n_embd]`), `wk`, `wv`, `wo`, `attn_q_norm`, `attn_k_norm`. **No separate `wq_gate` / `attn_gate` field on full-attention layers.** The GGUF's `attn_gate.weight` tensor only appears on linear-attention layers, where it means DeltaNet Z-gate.

### Authoritative activation tiebreakers (sigmoid vs swish)

13. `/Users/robert/.pyenv/versions/3.13.12/lib/python3.13/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:689` — `attn_output = attn_output * torch.sigmoid(gate)`
14. `/opt/vllm/vllm/model_executor/models/qwen3_next.py:312-314` — `gate = torch.sigmoid(gate); attn_output = attn_output * gate`

Both authoritative implementations apply **sigmoid**, not swish. llama.cpp matches. HF config's `output_gate_type: "swish"` is a vestigial label the transformers code doesn't read.

### Existing local reference: apex.gguf on disk

`/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf` (25 GB, `general.file_type=17` = MOSTLY_Q5_K_M baseline label, actually mixed-bit APEX: Q6_K / Q8_0 / F32 / 2× I16 on embed + output). **NOT a test oracle; NOT a fixture; NOT a CI dependency** per the sovereignty directive. Convenience for eyes only — like reading llama.cpp source. Hf2q's test of "does this file load + infer coherently" is the end-gate; the GGUF's byte content is not checked in or asserted against in any unit test.

---

## Architecture Decisions

Each decision below specifies: **Problem** (what breaks today), **Decision** (what we do), **Acceptance Criteria** (how we know it's done). Decisions are numbered for citation from commits and follow-up issues.

### 1. Arch detection + model registry

**Problem.** `src/backends/gguf.rs:1469` doesn't recognize `qwen35` or `qwen35moe` as loadable architectures. Currently only Gemma-4 loads through hf2q's serve/generate path.

**Decision.** Extend the arch-dispatch table in `src/backends/gguf.rs` with entries for `qwen35` and `qwen35moe`, both routing to a new module `src/inference/models/qwen35/`. The dispatch reads `general.architecture` from the GGUF metadata and routes to the matching handler. Dense vs. MoE is disambiguated inside the qwen35 module by the same arch string.

Module file layout (per `project_model_class_split.md`):

- `src/inference/models/qwen35/mod.rs` — shared across both variants: linear-attention forward, gated full-attention forward, MROPE dispatch, tokenizer, MTP load-path, hybrid KV cache management, `Qwen35Config` parser, `LayerKind` enum.
- `src/inference/models/qwen35/dense.rs` — dense-specific: dense SwiGLU FFN, dense tensor-name resolution, dense forward entry point.
- `src/inference/models/qwen35/moe.rs` — MoE-specific: 256-expert dispatch, shared-expert-gate, MoE tensor-name resolution, MoE forward entry point.
- `src/inference/models/qwen35/kernels.rs` — thin wrappers around `mlx-native`'s new ops for Qwen3.5 (keeps forward-pass code readable; no math here, just dispatch).
- `src/inference/models/qwen35/kv_cache.rs` — hybrid cache (full-attention KV + linear-attention SSM state).
- `src/inference/models/qwen35/tests/` — unit + integration tests.

**Acceptance criteria.**
- `src/backends/gguf.rs` dispatch switch includes `"qwen35" => Box::new(Qwen35Dense::load(...))` and `"qwen35moe" => Box::new(Qwen35Moe::load(...))`.
- Attempting to load a qwen35 GGUF through the existing Gemma path errors with a clear message pointing to the qwen35 module.
- Gemma-4 loading path byte-identical before and after (regression guard).

### 2. Qwen3.5 config struct + per-variant layer-kind enum

**Problem.** Gemma-4's `LayerType::{Sliding, Full}` enum encodes a different semantic axis than Qwen3.5's `{linear_attention, full_attention}`. Generalizing the existing enum (option A) would either strip meaning or add cases that don't apply to Gemma. Per `project_model_class_split.md`: model-specific enums live in model-specific files.

**Decision.** Define a new `Qwen35Config` struct and `Qwen35LayerKind` enum **inside** `src/inference/models/qwen35/mod.rs`. Do not touch `src/serve/config.rs::LayerType`.

```rust
pub enum Qwen35LayerKind { LinearAttention, FullAttention }

pub struct Qwen35Config {
    pub variant: Qwen35Variant,                // Dense | Moe
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,                         // 256
    pub linear_num_key_heads: u32,             // 16
    pub linear_num_value_heads: u32,           // 32 (MoE) or 48 (dense)
    pub linear_key_head_dim: u32,              // 128
    pub linear_value_head_dim: u32,            // 128
    pub linear_conv_kernel_dim: u32,           // 4
    pub full_attention_interval: u32,          // 4
    pub layer_types: Vec<Qwen35LayerKind>,     // authoritative; source of truth
    pub partial_rotary_factor: f32,            // 0.25
    pub rope_theta: f64,                       // 1e7
    pub mrope_section: [u32; 4],               // [11, 11, 10, 0]
    pub mrope_interleaved: bool,               // true
    pub max_position_embeddings: u32,          // 262144
    pub rms_norm_eps: f32,                     // 1e-6
    pub vocab_size: u32,                       // 248320
    pub attn_output_gate: bool,                // true
    pub mtp_num_hidden_layers: u32,            // 1
    pub dtype: DType,                          // BF16
    pub intermediate_size: Option<u32>,        // dense: 17408; moe: None
    pub moe: Option<Qwen35MoeConfig>,          // dense: None
}

pub struct Qwen35MoeConfig {
    pub moe_intermediate_size: u32,            // 512
    pub num_experts: u32,                      // 256
    pub num_experts_per_tok: u32,              // 8
    pub shared_expert_intermediate_size: u32,  // 512
    pub router_aux_loss_coef: f32,             // 1e-3 (unused at inference, loaded for completeness)
}
```

`Qwen35Config::from_gguf_metadata(reader)` parses `qwen35.*` / `qwen35moe.*` keys (the set ADR-012 Decision 7 emits). Prefer explicit `layer_types` array when present; fall back to computing from `full_attention_interval` (sanity-check they agree when both present).

**Acceptance criteria.**
- Unit test loads the local apex GGUF's metadata and asserts every field parses with the correct value (num_hidden_layers=40, layer_types[3] = FullAttention, layer_types[0] = LinearAttention, shared_expert_intermediate_size=512, mrope_section=[11,11,10,0], etc.).
- Unit test asserts Gemma-4 config parsing is unaffected (no hidden dependency on new code).
- Malformed GGUF (qwen35moe arch string but missing `expert_count` key) errors clearly, pointing at the missing key.

### 3. mlx-native: L2_NORM kernel

**Problem.** Gated DeltaNet applies L2 norm to Q and K after the conv1d state update, before the core DeltaNet computation. Reference: `delta-net-base.cpp:320-321` (`q_conv = ggml_l2_norm(...)`). No `l2_norm` kernel exists in mlx-native today.

**Decision.** Implement `mlx-native/src/shaders/l2_norm.metal` and `mlx-native/src/ops/l2_norm.rs`. The math is:

```
l2_norm(x, eps) = x / sqrt(sum(x^2) + eps)
```

Applied per-row (over the head_dim axis). Dtype: BF16 and F32 variants. Kernel spec derived from the math above — no file copied from llama.cpp.

**Acceptance criteria.**
- Shader compiles; op wrapper exposes `ops::l2_norm(buf, shape, eps) -> MlxBuffer`.
- Unit test: for a hand-constructed small tensor with known Euclidean norm, output matches `x / ||x||` within 1e-5 for F32, 1e-3 for BF16.
- Unit test: backprop-style round-trip — `|l2_norm(x) * ||x|| - x| < eps` for random inputs.
- Spec-driven test: test comment derives formula from mathematical definition; expected outputs hand-authored.

### 4. mlx-native: CUMSUM kernel (inclusive prefix sum)

**Problem.** Chunked DeltaNet computes a cumulative sum across the gate tensor to produce the decay-mask base. Reference: `delta-net-base.cpp:88` (`ggml_cumsum(g_transposed)`). Required for the AR-decomposed debug path *and* used internally by the fused kernel. No `cumsum` in mlx-native today.

**Decision.** Implement `mlx-native/src/shaders/cumsum.metal` and `mlx-native/src/ops/cumsum.rs`. Inclusive prefix sum along a single axis. Use a parallel-prefix / Hillis-Steele style algorithm in Metal for O(log N) rounds on small axes, or segmented scan for longer ones. Dtype: BF16 and F32.

**Acceptance criteria.**
- Shader + op wrapper exist.
- Unit test: `cumsum([1,2,3,4]) = [1,3,6,10]`; multi-batched input; axis parameterization.
- Spec-driven test: formula `out[i] = sum(x[0..=i])` hand-derived in test comment; no reference-tool-produced values.

### 5. mlx-native: TRI_SOLVE kernel (triangular system solve)

**Problem.** Chunked DeltaNet solves a lower-triangular linear system per chunk: `solve(lhs, attn)` where `lhs` is lower-triangular with identity on the diagonal. Reference: `delta-net-base.cpp:165` (`ggml_solve_tri(lhs, attn, true, true, false)` = lower, unit-diagonal, no-transpose).

Status: `TRI_SOLVE` in llama.cpp is a recent add (Hexagon, Metal, WebGPU support landed in commits around `c1258830b`). We derive spec from the operation: forward substitution on a lower-triangular unit-diagonal matrix against a multi-column right-hand side.

**Decision.** Implement `mlx-native/src/shaders/tri_solve.metal` and `mlx-native/src/ops/tri_solve.rs`. Lower-triangular unit-diagonal forward substitution; multi-column RHS; batched over leading dimensions. Numerical clamp on the decay-mask inputs (the `[-50, 50]` log-space clamp at `delta-net-base.cpp:191-192`) is applied in the caller, not inside the kernel.

**Acceptance criteria.**
- Shader + op wrapper exist.
- Unit test: 4×4 lower-triangular with random off-diagonal, 4×8 RHS, compute X = solve(L, B), verify `|L @ X - B| < 1e-4` (F32) / 1e-2 (BF16).
- Spec-driven test: forward-substitution formula derived in comment; golden outputs hand-computed for a 3×3 example and checked in the test file.
- Used path-internally by Decision 8 debug path only (fused kernel subsumes it for production).

### 6. mlx-native: GATED_DELTA_NET fused kernel

**Problem.** The central novel kernel. llama.cpp's `kernel_gated_delta_net_impl<NSG>` in `ggml-metal.metal` is ~300 lines with 4 NSG variants (threadgroup sizes 1/2/4/8). We write ours from the algorithmic specification, not by copying shader source.

**Decision.** Implement `mlx-native/src/shaders/gated_delta_net.metal` and `mlx-native/src/ops/gated_delta_net.rs`.

The fused operation is a single op taking `(q, k, v, g, beta, state_in)` and producing `(output, state_out)`:

```
// Spec (derived from delta-net-base.cpp:372-421 fused path + qwen35.cpp math)
// Inputs:
//   q: [head_k_dim, num_k_heads, n_tokens, n_seqs]
//   k: [head_k_dim, num_k_heads, n_tokens, n_seqs]
//   v: [head_v_dim, num_v_heads, n_tokens, n_seqs]
//   g: gates [num_v_heads, n_tokens, n_seqs] (post-sigmoid/softplus on host side)
//   beta: decay [num_v_heads, n_tokens, n_seqs]
//   state_in: [head_v_dim, head_v_dim, num_v_heads, n_seqs] (per-seq recurrent state)
// Outputs:
//   output: [head_v_dim, num_v_heads, n_tokens, n_seqs]
//   state_out: same shape as state_in
//
// Per token t within a seq:
//   state[t+1] = alpha(t) * state[t] + beta(t) * (v[t] - state[t] @ k[t]) * k[t]^T
//   output[t] = state[t+1] @ q[t]
// where alpha(t) = exp(-g(t)) decays state between tokens.
```

Implementation details to match llama.cpp's performance:
- Multi-NSG dispatch (4 kernel variants parameterized on threadgroup size) selected per-shape at encoder time.
- Threadgroup scratch memory for state accumulation.
- Simdgroup matmul primitives where shape permits (Apple10 supports `has tensor = true` per today's Metal init dump).
- F32 accumulation regardless of input BF16 (per `mamba_ssm_dtype: float32`).

**Acceptance criteria.**
- Shader + op wrapper exist.
- **Spec-driven unit test**: for a synthetic 1-seq, 1-head, 4-token, 8-dim example, hand-compute the expected state evolution and output token-by-token, assert the kernel's output matches to 1e-3 (F32) / 1e-1 (BF16). The hand-computation is documented in a code comment with derivation from the spec above.
- **CPU parity test**: a pure-Rust scalar implementation of the spec above (in `kv_cache.rs` debug module) is written first; fused-kernel output matches the scalar CPU implementation on random inputs to 1e-3 (F32). The scalar CPU implementation is our internal "ground truth" for kernel correctness — it is not shipped in production.
- **Performance test**: benchmark against the M5 Max baseline (llama.cpp reports decode ~97 tok/s on 40-layer 3B-active MoE). hf2q+mlx-native's fused GATED_DELTA_NET kernel contributes no more than X ms/token on the decode path, where X is chosen so the total decode budget nets ≥97 tok/s. Measurement method per `reference_decode_benchmark_methodology.md`.

### 7. mlx-native: SSM_CONV kernel (Qwen3.5 linear-attention conv1d)

**Problem.** Linear-attention layers apply a 4-kernel-wide causal conv1d across the QKV projection's output, then split into q_conv / k_conv / v_conv. Reference: `delta-net-base.cpp:~240` and the `ssm.conv_kernel: 4` GGUF metadata (verified from local apex.gguf dump).

**Decision.** Implement `mlx-native/src/shaders/ssm_conv.metal` and `mlx-native/src/ops/ssm_conv.rs`. 1D causal convolution with kernel width 4, SiLU activation fused, concatenation with previous conv-state (per-seq ring buffer).

Spec (derived from delta-net-base.cpp and Mamba-family conv1d literature):

```
ssm_conv(x, kernel, state) -> (y, new_state)
  where x is [channels, n_tokens, n_seqs], kernel is [K=4, channels],
        state is [K-1=3, channels, n_seqs] holding previous conv inputs,
        y[c, t, s] = silu( sum_k(kernel[k,c] * extended[c, t+k-K+1, s]) )
        extended = concat(state, x) along token axis,
        new_state = last 3 tokens of extended (ring-buffered for next call).
```

**Acceptance criteria.**
- Shader + op wrapper exist; accepts per-seq state buffer.
- Spec-driven test: 1-seq, 4-channel, 6-token input with random kernel; hand-compute causal conv output with SiLU; match to 1e-4 (F32).
- Ring-buffer correctness test: run conv on tokens [0..4], save state, run conv on tokens [4..8] using saved state, compare against monolithic run on [0..8]; outputs byte-identical.

### 8. Gated DeltaNet forward in Rust (shared by dense and MoE)

**Problem.** The Rust graph builder for linear-attention layers needs to orchestrate: input norm → QKV+Z projection → conv → gate (alpha = softplus(·)·A, beta = sigmoid(·)) → optional Q/K L2 norm → GATED_DELTA_NET → gated output norm → out-projection → residual.

**Decision.** Implement `src/inference/models/qwen35/mod.rs::build_delta_net_layer(cfg, layer, layer_weights, kv_cache, input) -> output`. Production path calls the fused `gated_delta_net` kernel from Decision 6; a **debug-only scalar-CPU path** implements the same math in pure Rust (the AR-decomposed `delta-net-base.cpp:288-370` spec) and is gated behind `#[cfg(feature = "qwen35-debug")]` so it's never compiled into release.

The debug path is **only for kernel parity validation and per-layer activation capture** (feeds Decision 10's `ActivationCapture` impl if the production GPU path is uninstrumented). It never enters a production hot loop and never appears as a fallback — per the no-fallback mantra.

**Acceptance criteria.**
- `build_delta_net_layer` unit test exercises a 1-seq, 4-token input with a synthetic layer weight set; compares against the debug-path scalar output to 1e-3 (F32).
- Gemma-4 inference test produces byte-identical output before and after adding this module (qwen35 code must be zero-cost for non-Qwen models; compile-time guarded).
- Debug feature path is not in the default `Cargo.toml` feature set; release build size unchanged.

### 9. Gated full-attention forward in Rust (shared by dense and MoE)

**Problem.** Full-attention layers have a gated output: Q + gate are fused in `wq` (output dim `2 × head_dim × n_head`); Q goes through RMSNorm, MROPE, SDPA with K (also normed + MROPE'd) and V; attention output is element-wise multiplied by `sigmoid(gate)`; result goes through `wo` projection; residual added. No Q·K softcap, no logit softcap at the attention level, kq_scale = `1 / sqrt(head_dim)`.

**Decision.** Implement `src/inference/models/qwen35/mod.rs::build_gated_attn_layer(cfg, layer, layer_weights, kv_cache, input, position_ids) -> output`. Uses mlx-native's existing `flash_attn_vec` / `flash_attn_prefill` for the core SDPA (shape-compatible with head_dim=256 per our existing `flash_attn_prefill_d512.metal` which handles head_dim ≤ 512), existing `rms_norm`, plus the new `rope_multi` op (Decision 11) and a new sigmoid-elementwise-multiply fusion if perf-profitable (Decision 16 addresses perf).

Op order (verbatim from the spec at `qwen35.cpp:117-196`):
1. Q+gate projection via `wq` → shape `[2 × head_dim × n_head, n_tokens]`
2. Q view = lower half, reshape to `[head_dim, n_head, n_tokens]`
3. Q norm (RMS, `attn_q_norm`, post-reshape, pre-RoPE)
4. K projection via `wk` → reshape `[head_dim, n_kv, n_tokens]` → K norm
5. V projection via `wv` → reshape `[head_dim, n_kv, n_tokens]`
6. Gate view = upper half of wq output, reshape to `[head_dim × n_head, n_tokens]`
7. Q, K ← MROPE(·, position_ids, sections=[11,11,10,0], mode=IMROPE, theta=1e7)
8. SDPA(Q, K, V, kq_scale = 1/sqrt(256), causal mask, GQA repeat = n_head / n_kv)
9. gate_sigmoid = sigmoid(gate)
10. cur = attn_output ⊙ gate_sigmoid (elementwise)
11. cur = wo @ cur
12. residual + cur (in caller)

**Acceptance criteria.**
- Unit test: for a synthetic 1-seq, 4-token, head_dim=16, n_head=4, n_kv=2 input, compute expected full-attention output from a pure-Rust scalar implementation of the spec; mlx-native-backed implementation matches to 1e-3 (F32).
- The sigmoid activation is verified against HF transformers behavior (reference 13/14 in the Reference Implementations section); swish would be a silent-corruption bug and is explicitly *not* implemented.

### 10. MROPE — mlx-native kernel + Rust dispatcher (IMROPE mode)

**Problem.** Multi-section interleaved RoPE is new. Sections `[11, 11, 10, 0]`, NeoX-style pair indexing (`x[i]` paired with `x[i + n_dims/2]`), per-section theta assignment by `sector % 3`. No `rope_multi` kernel exists in mlx-native today.

**Decision.** Implement `mlx-native/src/shaders/rope_multi.metal` and `mlx-native/src/ops/rope_multi.rs`. Supports IMROPE mode (`mode == 40`) and standard MROPE (`mode == 8`) for future-proofing (Qwen-VL / Qwen3Next use non-interleaved MROPE). The Rust wrapper for Qwen3.5 always passes IMROPE; the kernel dispatches on mode.

Pseudocode (ported from the spec in `project_qwen36_architecture.md` and `ggml-cpu/ops.cpp:5643-5731`):

```rust
// In mlx-native/src/ops/rope_multi.rs — Rust dispatcher
pub fn rope_multi(
    qk: &mut MlxBuffer,
    position_ids: &MlxBuffer,
    sections: [u32; 4],     // [11, 11, 10, 0] for Qwen3.5
    mode: RopeMultiMode,    // Imrope or Mrope
    n_rot: u32,             // 64 (rotary_dim)
    freq_base: f32,         // 1e7
    // yarn params zeroed for Qwen3.5 text
) -> Result<()>;
```

Metal kernel computes `cos/sin` per rotary-index per section, then applies NeoX-style pair rotation. The `sector % 3`-cycling section selection is the IMROPE distinguisher.

**Acceptance criteria.**
- Shader + op wrapper exist.
- Spec-driven unit test: for n_rot=8, sections=[2,2,1,0], a hand-computed table of cos/sin values per rotary index matches the kernel's output. The test file documents the derivation from the spec in a comment.
- Integration unit test: Qwen3.5 full parameters (n_rot=64, sections=[11,11,10,0], freq_base=1e7) applied to a synthetic Q tensor produces bit-stable output across repeated calls (determinism).
- Cross-check: a pure-Rust scalar implementation in `kernels.rs` (debug feature) matches the Metal kernel to 1e-5 (F32).

### 11. Hybrid KV cache (full-attn KV + linear-attn SSM state)

**Problem.** Full-attention layers need a standard KV cache (token-indexed). Linear-attention layers need TWO caches: (a) the conv1d ring-buffer state (3 tokens × channels per seq), (b) the recurrent state matrix (`[head_v_dim, head_v_dim, num_v_heads, n_seqs]`).

**Decision.** `src/inference/models/qwen35/kv_cache.rs::HybridKvCache`:

```rust
pub struct HybridKvCache {
    pub full_attn: Vec<FullAttnKvSlot>,     // len = # full-attn layers
    pub linear_attn: Vec<LinearAttnStateSlot>,
}

pub struct FullAttnKvSlot {
    pub k: MlxBuffer,  // [head_dim, n_kv, max_seq_len, n_seqs]
    pub v: MlxBuffer,  // same shape
    pub current_len: u32,
}

pub struct LinearAttnStateSlot {
    pub conv_state: MlxBuffer,    // [K-1=3, channels, n_seqs]
    pub recurrent: MlxBuffer,     // [head_v_dim, head_v_dim, num_v_heads, n_seqs]
}
```

Allocation strategy: allocate full-cache-size upfront sized by `cfg.max_position_embeddings` (262K), use F32 for SSM state per `mamba_ssm_dtype: float32`, F32 or BF16 for full-attn KV (default F32 matching llama.cpp's production behavior — matches our Gemma decode-parity baseline methodology per `project_decode_parity_achieved.md`).

**Acceptance criteria.**
- Cache initialization for a 40-layer MoE config produces 10 full-attn slots and 30 linear-attn slots (every 4th = full; 10 × 4 = 40).
- Unit test: cache update from token 0 to token 100 through a synthetic forward pass preserves correctness (outputs at token 100 match a fresh-allocation full-context forward pass within eps).

### 12. Weight loading from GGUF (both variants)

**Problem.** hf2q's GGUF reader handles Gemma-4 tensor names. Qwen3.5 has different names; I16 quantization (embeddings + output) is unsupported; per-layer heterogeneity requires dispatched loading.

**Decision.** Two sub-tasks:

(a) **Tensor-name table** for `qwen35` + `qwen35moe` in `src/inference/models/qwen35/{dense,moe}.rs` — the exhaustive lists extracted by reading `src/llama-arch.cpp` tensor-name tables for `LLM_ARCH_QWEN35` / `LLM_ARCH_QWEN35MOE`. Every tensor a production GGUF is known to contain is enumerated as a constant with a code comment citing the llama.cpp file:line the name was sourced from. Loader dispatches on (layer_index, layer_kind) to select which tensor set to load for that layer.

(b) **I16 dequantization** in `src/backends/gguf.rs`. Add `GgmlQuantType::I16` to the match statement; dequant formula is `f32_val = i16_val * scale` where `scale` is a single f32 per tensor, stored in the GGUF's per-tensor metadata. ~20 LOC.

**Acceptance criteria.**
- Unit test: load the apex GGUF's tensor list, assert every expected tensor is found (full set of qwen35moe tensors for a 40-layer MoE with 256 experts: global tensors + 10 × full-attn block × 10 tensors + 30 × linear-attn block × 14 tensors + 40 × MoE block × (3 stacked expert tensors + 4 shared + 1 router) + tokenizer = ~1500 tensors total; actual count from apex dump was 733 *post-merge*).
- I16 dequant test: hand-constructed I16 buffer with known scale → float output matches `i16 * scale` to 1e-6.
- Gemma-4 weight loading byte-identical before and after (regression guard).

### 13. MoE FFN forward (MoE variant only)

**Problem.** 256 experts, top-8 routing, one always-gated shared expert. Existing Gemma-4 MoE path is 128 experts / top-8 (different count) with no shared expert.

**Decision.** Implement `src/inference/models/qwen35/moe.rs::build_moe_ffn_layer(...)`. Reuses mlx-native's existing `moe_gate` + `moe_dispatch` kernels (verified present in `src/ops/`) since the *mechanics* of top-k gating and expert dispatch are shape-agnostic; only counts/dims differ. Ensure:

- Router projection via `ffn_gate_inp` (F32), softmax, top-8 selection, renormalized routing weights.
- Gated shared expert: project via `ffn_gate_inp_shexp` → sigmoid → multiply shared-expert FFN output → add to routed MoE output (per reference `qwen35moe.cpp:406-420`).
- Stacked expert tensors (`ffn_gate_up_exps`, `ffn_down_exps`) accessed via indexed matmul (`quantized_matmul_id_ggml` in mlx-native, already present).

**Acceptance criteria.**
- Unit test with 4 synthetic experts + 1 shared, known routing produces expected output; shared-expert gate path verified (gate=0 → shared contribution = 0; gate=1 → shared contribution = full FFN output).
- Gemma-4 128-expert path untouched and byte-identical.

### 14. Dense FFN forward (dense variant only)

**Problem.** Dense Qwen3.5 has standard SwiGLU — gate_proj, up_proj, down_proj.

**Decision.** Implement `src/inference/models/qwen35/dense.rs::build_dense_ffn_layer(...)`. Straightforward: `down_proj(silu(gate_proj(x)) * up_proj(x))`. Uses existing mlx-native matmul + SiLU + elementwise ops.

**Acceptance criteria.**
- Unit test: synthetic FFN weights produce expected SwiGLU output.
- Zero overlap with MoE code path (compile-time variant dispatch).

### 15. MTP tensors (load-only, no execution)

**Problem.** `mtp_num_hidden_layers: 1` in config. GGUF tensors named `blk.{N}.nextn.*` (where N = num_hidden_layers) per ADR-012 Decision 11. These must load without error even though the v1 hot path doesn't execute speculative decoding.

**Decision.** Load MTP tensors into memory as `Option<MtpWeights>` on `Qwen35Model`. Production forward ignores them. A follow-up ADR (speculative decoding) consumes them.

If the GGUF doesn't contain MTP tensors (like the apex.gguf on disk today, which had them stripped per the 2026-04-23 dump), `Option` is None and no error.

**Acceptance criteria.**
- Load on apex.gguf succeeds (MTP absent → None, no error).
- Load on a (future hf2q-produced) GGUF with MTP succeeds; `MtpWeights` is populated; tensor shapes match spec.

### 16. `ActivationCapture` trait implementation (ADR-012 cross-coordination)

**Problem.** ADR-012 Decision 13 defines:

```rust
pub trait ActivationCapture {
    fn run_calibration_prompt(&mut self, tokens: &[u32]) -> Result<LayerActivations>;
}
```

ADR-012's P6 (DWQ calibration for Qwen3.5) blocks on our implementation. Per ADR-012 R3: no weight-space fallback is offered; we must implement this.

**Decision.** Implement `ActivationCapture` on `src/inference/models/qwen35/mod.rs::Qwen35Model`. The implementation runs the full forward pass with hooks that capture per-layer inputs and outputs into `LayerActivations`. Captures happen on host-side copies (MlxBuffer → F32 Vec) aggregated over calibration prompts. Capture is opt-in and gated by a `CaptureConfig` parameter (not always-on, to avoid memory cost in production).

**Acceptance criteria.**
- Trait impl lands in hf2q before ADR-012's P6 starts wiring (minimizes their rework).
- Mock implementation (deterministic synthetic activations) exists for ADR-012-side tests in their repo path, per their Decision 13 acceptance criteria.
- Real implementation runs a small prompt through a tiny test model (3 linear-attn + 1 full-attn, 4 experts) and produces sensible `LayerActivations` tensor shapes.
- No dependency on a working end-to-end inference session at merge time — activation capture is per-layer-testable with synthetic weights.

### 17. Sourdough byte-parity gate (correctness end-gate)

**Problem.** We need a deterministic proof that hf2q's Qwen3.5 forward produces the same tokens as llama.cpp's on the same inputs. Per ADR-009's reference-parity methodology and ADR-005's inference-server gate.

**Decision.** `scripts/sourdough_qwen35.sh` — mirrors `scripts/sourdough_gate.sh` (the Gemma-4 gate, per memory `project_decode_parity_achieved.md`):

1. Fixed prompt file: `tests/sourdough_qwen35_prompt.txt` (byte-content checked in; human-authored, short: ~10 tokens).
2. Run llama-cli on the GGUF: `llama-cli -m <gguf> -p @<prompt> -n 64 --temp 0.0 --seed 42 -ngl 99 -st --no-display-prompt > /tmp/llama_qwen35_output.txt`. **llama-cli is treated as an external black-box tool, run at gate-time; it is not linked into hf2q at any build stage.**
3. Run hf2q generate on the same GGUF + prompt + seed: `hf2q generate -m <gguf> -p @<prompt> -n 64 --temp 0.0 --seed 42 > /tmp/hf2q_qwen35_output.txt`.
4. Compare byte-by-byte on the generated token stream (exclude formatting differences like llama-cli's banner); ≥N prefix bytes must match where N is chosen such that divergence from tokenizer/sampler noise is negligible (Gemma precedent: 3656-byte prefix).

**Acceptance criteria.**
- Script runs green on the apex GGUF (or, once ADR-012 ships, on hf2q's own dwq46 output for the same HF repo).
- Script runs in <5 minutes wall-clock on M5 Max (loading llama.cpp + loading hf2q serially + generating 64 tokens each).
- Divergence early enough to catch V-head reorder mis-application, MROPE section mis-assignment, sigmoid-vs-swish, kv-cache off-by-one.
- Included as a Phase P13 gate; any PR merging qwen35 forward-pass changes must pass the sourdough test.
- When the script fails, the diff points to the first diverging byte; the developer's debug loop uses Decision 8's scalar CPU path + per-layer dumps via the `ActivationCapture` trait to bisect.

### 18. Performance baseline + match-or-beat gate

**Problem.** Per `project_end_gate_reality_check.md`, perf targets are re-measured on the day. Today's measurement (2026-04-23, M5 Max): llama.cpp `b8914-8bc492ebb` at 97.3 tok/s decode / 364.8 tok/s prompt on the apex GGUF.

**Decision.** `scripts/qwen35_bench.sh` — wraps `llama-bench` and `hf2q bench` on the same GGUF. Reports tok/s for prompt prefill and decode at a standard test harness (prompt lengths: 128, 2455, 16384; decode lengths: 64, 256, 1024). Gate: hf2q ≥ 0.95× llama.cpp at each data point (5% drift budget for implementation differences / non-determinism).

**Acceptance criteria.**
- Bench script exists and produces a pair of tables (llama vs hf2q).
- Final PR closing the ADR must show bench tables with hf2q numbers within drift budget. Re-measure llama.cpp on the same day per the ground-truth-is-what-we-can-measure-now rule.
- No claim of "parity achieved" without a concurrent measurement; historical numbers (like today's 97.3 tok/s) are starting hints, not ship gates.

---

## Phase plan

Phases are **dependency-ordered**, not priority-ordered. Per `feedback_swarm_sequential_when_shared_build.md`, each phase has a single owner claim when touching shared Cargo target.

### P0 — mlx-native foundations (L2_NORM + CUMSUM + SSM_CONV, parallel-safe)

**Scope:** Decisions 3, 4, 7. Independent kernels with no ordering dependencies. Smallest, simplest kernels first to establish the spec-driven-test pattern in mlx-native.

**Deliverables:**
- `l2_norm.metal` + `l2_norm.rs` + unit tests
- `cumsum.metal` + `cumsum.rs` + unit tests
- `ssm_conv.metal` + `ssm_conv.rs` + unit tests (includes ring-buffer state handling)

**Acceptance:** All three ops pass spec-driven unit tests; no existing mlx-native kernel regresses (`cargo test` in mlx-native green).

**Estimated LOC:** ~600 Rust + ~400 Metal.

### P1 — mlx-native: TRI_SOLVE kernel

**Scope:** Decision 5. Standalone, but only used by the decomposed-debug path and (internally) by the fused GATED_DELTA_NET kernel. Lands before P2.

**Deliverables:** `tri_solve.metal` + `tri_solve.rs` + unit tests.

**Acceptance:** Spec-driven test passes; |L @ solve(L, B) - B| < 1e-4 (F32).

**Estimated LOC:** ~300 Rust + ~300 Metal.

### P2 — mlx-native: MROPE kernel (IMROPE mode)

**Scope:** Decision 11 (mlx-native side only; Rust dispatcher wiring is P5).

**Deliverables:** `rope_multi.metal` + `rope_multi.rs` + unit tests.

**Acceptance:** Spec-driven test (small n_rot) matches hand-computed cos/sin table; CPU reference matches Metal kernel to 1e-5 (F32); determinism test passes.

**Estimated LOC:** ~300 Rust + ~250 Metal.

### P3 — mlx-native: GATED_DELTA_NET fused kernel (the centerpiece)

**Scope:** Decision 6. Depends on P0 (uses cumsum internally) and P1 (uses tri_solve internally for chunked sub-path if we include it). Largest single kernel.

**Deliverables:** `gated_delta_net.metal` (4 NSG variants) + `gated_delta_net.rs` wrapper + unit tests (spec-driven + CPU parity) + microbench.

**Acceptance:** Spec-driven test + CPU parity test pass. Microbench within 20% of llama.cpp's Metal kernel at matching shape.

**Estimated LOC:** ~500 Rust + ~700 Metal.

### P4 — hf2q: module scaffold + config + arch detection

**Scope:** Decisions 1, 2. Pure Rust scaffolding; no forward-pass code yet. Unblocks P5+ work.

**Deliverables:**
- `src/inference/models/qwen35/` module created with `mod.rs`, `dense.rs`, `moe.rs`, `kernels.rs`, `kv_cache.rs`, `tests/`
- `Qwen35Config` struct + parser
- `src/backends/gguf.rs` dispatch extended for `qwen35` and `qwen35moe` arch strings (handlers return `unimplemented!` stubs)
- Module registered in `src/inference/models/mod.rs`
- Unit tests for config parsing against apex.gguf metadata

**Acceptance:** `cargo build --release` succeeds; loading apex.gguf reaches the `unimplemented!` with a qwen35moe-specific panic message; Gemma-4 regression byte-identical.

**Estimated LOC:** ~400.

### P5 — hf2q: weight loading + I16 dequant

**Scope:** Decision 12. Complements ADR-012 (emission side); this is the load side.

**Deliverables:**
- Tensor-name table constants in `dense.rs` + `moe.rs`
- I16 dequant in `src/backends/gguf.rs`
- Weight-loading wire-up in `Qwen35Dense::load` / `Qwen35Moe::load`
- Unit tests

**Acceptance:** Loading apex.gguf produces a fully-populated `Qwen35MoeModel` struct (all tensors loaded, all shapes match `Qwen35Config`); Gemma regression byte-identical.

**Estimated LOC:** ~500.

### P6 — hf2q: KV cache (hybrid)

**Scope:** Decision 11. Pure Rust.

**Deliverables:** `kv_cache.rs` with `HybridKvCache`, per-slot allocation, update hooks.

**Acceptance:** Unit test against synthetic layer; integration-ready (final wiring in P8/P9).

**Estimated LOC:** ~300.

### P7 — hf2q: Gated full-attention layer (shared Rust)

**Scope:** Decision 9. Uses mlx-native's existing flash-attn + new MROPE (P2).

**Deliverables:** `mod.rs::build_gated_attn_layer` + scalar CPU parity impl (debug feature) + unit tests.

**Acceptance:** Synthetic-weight test matches CPU parity reference to 1e-3 (F32). Spec test with hand-computed 1-seq 4-token expected output passes.

**Estimated LOC:** ~450.

### P8 — hf2q: Gated DeltaNet layer (shared Rust)

**Scope:** Decision 8. Uses P3 fused kernel; P0 (conv, l2_norm); P1 (tri_solve, debug-only).

**Deliverables:** `mod.rs::build_delta_net_layer` + scalar CPU parity impl + unit tests.

**Acceptance:** Synthetic test matches CPU parity reference; kernel-level parity ≤1e-3 (F32).

**Estimated LOC:** ~600.

### P9 — hf2q: MoE FFN + Dense FFN forward + variant dispatch

**Scope:** Decisions 13, 14.

**Deliverables:**
- `moe.rs::build_moe_ffn_layer` (256 experts, shared-expert gate, reuses mlx-native moe_dispatch)
- `dense.rs::build_dense_ffn_layer` (SwiGLU)
- Variant dispatch glue in `mod.rs::build_layer_ffn`

**Acceptance:** Synthetic 4-expert + 1-shared test passes (MoE); synthetic SwiGLU test passes (dense); no cross-variant code path execution.

**Estimated LOC:** ~400.

### P10 — hf2q: MTP load path

**Scope:** Decision 15. Load-only, no execution.

**Deliverables:** `Qwen35Model::mtp: Option<MtpWeights>`, loader handles presence/absence gracefully.

**Acceptance:** Load on apex.gguf (no MTP) → None, no error. Load on synthetic MTP-bearing GGUF → Some(weights), shapes correct.

**Estimated LOC:** ~100.

### P11 — hf2q: End-to-end forward wire-up (dense + MoE entry points)

**Scope:** Integrate P4–P10 into a working `forward(model, input_tokens, position_ids, kv_cache) -> logits` for both variants. Register entry points in `src/serve/` (mirroring Gemma's integration point, without generalizing the Gemma struct).

**Deliverables:**
- `Qwen35Model::forward` dispatches per `Qwen35Variant` to dense or MoE path.
- `src/serve/generate.rs` (or equivalent) registers qwen35/qwen35moe as supported architectures.
- `hf2q generate -m <qwen35-gguf> -p '...' -n 64` works end-to-end (may produce wrong output if bugs remain; Phase P13 is the correctness gate).

**Acceptance:** End-to-end flow runs; loss=NaN is absent; logits shape `[vocab=248320, n_tokens]`; Gemma regression byte-identical.

**Estimated LOC:** ~300.

### P12 — hf2q: `ActivationCapture` trait impl

**Scope:** Decision 16. Cross-ADR coordination.

**Deliverables:** `Qwen35Model` implements `ActivationCapture`. Hooks into P11 forward to capture per-layer activations. Unblocks ADR-012 P6.

**Acceptance:** Mock-friendly interface; synthetic-weight test produces expected-shape `LayerActivations`; documented coordination with ADR-012 engineer.

**Estimated LOC:** ~250.

### P13 — Correctness gate: sourdough + bench + integration tests + docs

**Scope:** Decisions 17, 18.

**Deliverables:**
- `scripts/sourdough_qwen35.sh` + fixed prompt in `tests/sourdough_qwen35_prompt.txt`
- `scripts/qwen35_bench.sh`
- `tests/integration_qwen35.rs` — synthetic tiny-model E2E for dense; same for MoE
- `docs/running-qwen35.md` — canonical inference invocations
- `docs/shipping-contract.md` update: qwen35 + qwen35moe inference acceptance criteria

**Acceptance:**
- Sourdough byte-parity passes on apex.gguf against llama.cpp's live output (Robert's 2026-04-23 baseline recomputed on gate day).
- Bench within 5% of llama.cpp on M5 Max.
- Integration tests green.
- Docs cover both dense and MoE invocations.

**Estimated LOC:** ~400 (excluding docs word count).

### Totals

- Total code LOC: ~4,700 (hf2q: ~3,200; mlx-native: ~1,500) plus ~1,650 Metal shader lines.
- Tests: ≥ 1 spec-driven unit test per kernel and per shared Rust function; 2 E2E integration tests (dense + MoE); 1 sourdough byte-parity end-gate; 1 benchmark gate.
- Docs: 2 new/updated user-facing docs.
- Timeline: dependency-sequential within each repo. Cross-repo parallelism possible (mlx-native P0-P3 parallel with hf2q P4-P6). Expect ~3–5 weeks on single-threaded execution; less with cross-repo pipelining.

---

## Test strategy

### Unit tests (per-op / per-function)

Located alongside implementations. Every kernel (L2_NORM, CUMSUM, TRI_SOLVE, SSM_CONV, GATED_DELTA_NET, ROPE_MULTI) and every Rust graph-builder function (delta-net layer, gated-attn layer, MoE FFN, dense FFN) has at least one spec-driven test whose expected outputs are hand-computed in a code comment derived from the mathematical specification, and at least one random-input parity test against a scalar CPU reference implementation.

### CPU scalar reference implementations (for parity testing)

Per Decisions 8 and 9: scalar-CPU implementations of GATED_DELTA_NET and IMROPE live in `src/inference/models/qwen35/kernels.rs` behind `#[cfg(feature = "qwen35-debug")]`. These are internal test oracles. They never run in production. They never ship in the default release build. They exist to:

1. Validate the Metal kernels on random inputs during development.
2. Allow per-layer debug dumps when sourdough parity fails.
3. Provide a forward-compatible reference if Apple releases a Metal API change that breaks our kernel.

### Regression tests (Chesterton's fence)

- **Gemma-4 inference byte-identical** before and after every phase. This is the only way to know qwen35 code didn't accidentally change the Gemma path. `scripts/sourdough_gate.sh` (existing) continues to pass unchanged.
- **Gemma-4 decode perf byte-identical** before and after: llama-bench on Gemma-4 GGUF produces the same tok/s ±0.1%.
- **Existing mlx-native kernels byte-identical** before and after: each existing flash-attn, SDPA, etc. kernel test passes unchanged.

### Spec-driven tests (the heart of sovereignty)

For V-head reorder-like tensor transforms (inherited from ADR-012 and applied at inference load-time): hand-author expected permutation maps; derive from ggml broadcast semantics. For IMROPE: hand-compute cos/sin per sector per position for a small n_rot. For GATED_DELTA_NET: hand-compute state evolution for 4 tokens, 1 head, 8-dim state. The test IS the spec — the test file cites the llama.cpp source that motivated it but doesn't consume llama.cpp output as a reference.

### Sourdough byte-parity end-gate

Decision 17. This is the only place a llama.cpp binary is invoked, and it's invoked **at gate-time by a developer or CI runner**, not from within any unit test, not from `cargo test`. llama.cpp's output is a live-measured black-box reference, not a checked-in fixture.

### Integration tests

- `tests/integration_qwen35_dense.rs`: tiny synthetic dense model, end-to-end forward, asserts deterministic output on a fixed prompt/seed.
- `tests/integration_qwen35_moe.rs`: same for MoE.
- No binary GGUF checked in; synthetic weights generated deterministically from a fixed seed in the test itself.

---

## Risks and mitigations

### R1: Metal kernel performance substantially below llama.cpp

**Likelihood:** Medium-high — their kernel is hand-tuned over multiple releases.
**Impact:** hf2q inference is slower than llama.cpp despite correct output. Violates Decision 18's match-or-beat gate.
**Mitigation:** Decision 6 includes a performance microbench. If our first cut is >20% behind, profile with HF2Q_PROFILE_GPU_TS (per `tooling_hf2q_profile_gpu_ts.md`), identify bottleneck (likely simdgroup-matmul utilization or threadgroup scratch-memory pressure), iterate. Per `project_metal_compiler_auto_optimizes_static_levers.md`, static Metal compiler optimizations often close the gap — measure before hand-tuning.

### R2: Silent-corruption bug due to V-head reorder, MROPE section mis-assignment, or sigmoid-vs-swish

**Likelihood:** Medium — this is the canonical "loads but wrong" failure mode.
**Impact:** Inference produces plausible-looking garbage that no unit test catches without full-forward execution.
**Mitigation:** Decision 17's sourdough byte-parity is the primary catch. Scalar CPU reference implementations (behind the debug feature) provide per-layer bisection when sourdough diverges. Decision 9's sigmoid citation (HF transformers `modeling_qwen3_5.py:689`) eliminates the swish landmine at spec time.

### R3: GGUF tensor-name inconsistency between ADR-012's emission and llama.cpp's expectation

**Likelihood:** Medium — llama.cpp tensor-name tables change occasionally; ADR-012 will land before or alongside us and may expose drift.
**Impact:** hf2q's GGUFs load in our inference but fail in llama.cpp (or vice-versa for external DWQ GGUFs like the apex file).
**Mitigation:** Both ADR-012 Decision 7 (emission) and our Decision 12 (loading) cite the same `LLM_KV_*` enum and `LLM_TENSOR_*` names from `/opt/llama.cpp/src/llama-arch.cpp`. Drift is caught by Decision 17's sourdough on a real GGUF. When llama.cpp updates names, that's a cross-ADR addendum PR updating both sides.

### R4: ADR-012 P6 blocks on our `ActivationCapture` trait impl

**Likelihood:** Medium — cross-session coordination is the hardest schedule to align.
**Impact:** ADR-012 stalls.
**Mitigation:** Per ADR-012 R3 (mirrored): we implement Decision 16 as early as possible (Phase P12). If our entire forward pass isn't ready yet, we provide a trait impl backed by synthetic activations for their unit tests, with the real one landing as P11 completes. Fix the blocker, don't fall back.

### R5: The apex GGUF on disk is slightly non-standard (e.g., I16 on embeddings)

**Likelihood:** Happened — confirmed I16 on token_embd and output; `ffn_norm` is absent (abliterated).
**Impact:** Our Decision 12 loader must handle the oddities.
**Mitigation:** Decision 12 explicitly covers I16 dequant; the loader treats missing optional tensors gracefully. Sourdough tests on apex.gguf flush out any remaining oddities.

### R6: hf2q produces GGUFs (via ADR-012) that have different tensor-set than llama.cpp's production output

**Likelihood:** Medium — our DWQ pipeline differs from llama-quantize; tensor *shapes* and *names* should be identical but quant types differ.
**Impact:** hf2q's GGUF loads in hf2q but not in llama.cpp (blocks sourdough comparison).
**Mitigation:** ADR-012 P7 integration tests assert llama.cpp loads hf2q's output. Decision 17's sourdough runs against hf2q's own GGUF (not the apex) for the canonical end-gate.

### R7: Long-context (>2K tokens) correctness drift

**Likelihood:** Medium — per `project_long_prefill_parity_inverts.md`, hf2q's existing long-prefill path has known divergence from llama.cpp at pp≥1024.
**Impact:** Qwen3.5 inference correct at short context but gibberish at long context.
**Mitigation:** Sourdough test runs at multiple prompt lengths (16, 128, 2048); any divergence at longer lengths is caught. Root-cause analysis piggybacks on the existing long-prefill investigation.

### R8: Metal kernel memory-pressure at decode time (35B MoE × 256K context)

**Likelihood:** Low-medium — today's decode baseline measured 29.8 GB Metal working set on M5 Max; that's well under the 110 GB recommended max working set size reported by Metal. But `max_position_embeddings=262144` is a lot.
**Impact:** OOM at long context.
**Mitigation:** Per `feedback_oom_prevention.md`. KV cache allocation is configurable (`Qwen35Config::max_seq_len` override); default to the model's `max_position_embeddings`. Document disk / VRAM requirements in `docs/running-qwen35.md`.

---

## Open questions

1. **Does hf2q's existing tokenizer pipeline handle the `qwen35` pre-tokenizer regex?** The GGUF's `tokenizer.ggml.pre = qwen35` is a new BPE pre-type; llama.cpp has a specific regex for it at `src/llama-vocab.cpp:2029-2031`. If hf2q's tokenizer doesn't match, chat output will be subtly wrong. Investigate in P11 before sourdough runs.
2. **Does the local apex.gguf's chat template work through hf2q's chat formatter?** The template is embedded in GGUF metadata (`tokenizer.chat_template`, ~500+ chars). Jinja or raw-string? hf2q's existing chat code path targeted Gemma; verify during P11.
3. **BOS/EOS discrepancy:** HF config says `eos=248044` (same as BOS); GGUF metadata dump showed `eos=248046`. Which is the real stop? Per `project_hf2q_generate_chat_template_stop.md` this sort of thing has bitten before. Empirical check in P13.
4. **Does `attn_post_norm` exist on every layer (both kinds) or only on full-attn layers?** The local apex.gguf dump showed `post_attention_norm.weight` on layer 0 (linear-attn); llama.cpp's `qwen35.cpp:56` applies it in the main loop outside the attention branch. Confirm in P7.
5. **Per-expert scale tensor handling:** Gemma-4 MoE has `per_expert_scale` tensors; does Qwen3.5-MoE have an analog? apex.gguf didn't show one; worth confirming against a hf2q-produced GGUF once ADR-012 ships.
6. **Do we ever need the chunked-decomposed Gated DeltaNet path?** We said no (fused is llama.cpp's production default). If perf or correctness demands it later, the chunked path becomes a follow-up phase. Today's decision is: skip it entirely at shipping time; AR-scalar-CPU remains the only non-fused path, debug-only.

---

## Dependencies on other work (cross-ADR)

- **ADR-012 (conversion):** Defines `ActivationCapture` trait that we implement (our Decision 16 / Phase P12). Emits the GGUF metadata keys we consume (our Decision 12). Cross-ADR coordination surface is the trait definition and the hand-transcribed `LLM_KV_*` catalog, which both ADRs share.
- **ADR-005 (inference server):** hf2q's existing serve/generate subcommand integrates qwen35 forward at Phase P11. We add to the arch-dispatch table; we don't refactor the serve layer.
- **ADR-008 (candle divorce):** The sovereignty directive inherits from ADR-008's in-flight work. No new code in this ADR introduces a candle dependency. Existing candle uses are ADR-008's cleanup; we don't add to the pile.
- **ADR-009 (reference parity):** Our Decision 17 extends ADR-009's sourdough methodology from Gemma-4 to Qwen3.5. Same pattern: llama.cpp as live black-box reference, not checked-in fixture.
- **ADR-011 (flash-attn prefill):** Existing `flash_attn_prefill_*.metal` shaders in mlx-native are reused for Qwen3.5 full-attention SDPA (head_dim=256 is within their supported range per the phase-2 port). No changes to ADR-011's kernels; only reuse.

---

## Glossary

- **A3B / A4B.** MoE shorthand: "active 3B / 4B parameters per token." Total parameter count (e.g., 35B for A3B) is much larger.
- **Abliterated.** Community term for a model with refusal-producing attention heads zero'd out; a fine-tune variant of the base weights.
- **APEX.** Community quantization variant with mixed-bit imatrix-guided calibration, producing a GGUF with diverse per-tensor quant types.
- **DeltaNet (Gated).** A linear-attention variant using a selective state-space-like recurrence with learned gating. Qwen3.5 / Qwen3.6's replacement for Mamba-style state updates.
- **DWQ.** Dynamic Weight Quantization. hf2q's activation-calibrated mixed-precision quant (ADR-012).
- **GQA.** Grouped-Query Attention. `num_attention_heads > num_key_value_heads`; K/V tensors are repeated (logically) to match Q.
- **IMROPE.** Interleaved Multi-section RoPE. `mode == GGML_ROPE_TYPE_IMROPE == 40`. Sections rotate `sector % 3`-cycling across 3 theta bases. Qwen3.5 uses this.
- **LayerKind (Qwen3.5).** Enum `{LinearAttention, FullAttention}`. Distinct from Gemma-4's `LayerType::{Sliding, Full}`.
- **LLM_ARCH_QWEN35 / QWEN35MOE.** llama.cpp architecture constants. Arch strings `qwen35` / `qwen35moe`.
- **MROPE.** Multi-axis RoPE. Rotary dimensions partitioned into multiple sections with independent theta bases.
- **MTP.** Multi-Token Prediction. Extra transformer block after main stack for speculative decoding. Load-only in this ADR; execution is a follow-up.
- **NSG.** Metal "Num SimdGroups" — threadgroup-size parameterization of a kernel.
- **Sourdough gate.** ADR-009 methodology: byte-parity check against a black-box reference (llama.cpp) on fixed prompt/seed/temp=0 input.
- **SSM.** State-Space Model. Mamba-family recurrence; a primitive Gated DeltaNet is related to but not identical to.

---

## Appendix A: Target inference commands (once all phases land)

### Qwen3.6-35B-A3B MoE on local apex GGUF

```bash
hf2q generate \
  --model /opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf \
  --prompt "Write one short paragraph explaining what a transformer neural network is." \
  --max-tokens 80 \
  --temperature 0.0 \
  --seed 42
```

### Qwen3.6-27B dense on an hf2q-produced DWQ GGUF (via ADR-012)

```bash
hf2q generate \
  --model models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf \
  --prompt "..." \
  --max-tokens 80 \
  --temperature 0.0 \
  --seed 42
```

### Sourdough gate

```bash
scripts/sourdough_qwen35.sh /path/to/qwen35moe.gguf
# Compares hf2q generate output against llama-cli output on the same (prompt, seed, temp=0).
# Exits non-zero on byte divergence.
```

### Benchmark gate

```bash
scripts/qwen35_bench.sh /path/to/qwen35moe.gguf
# Reports prompt prefill + decode tok/s for hf2q and llama.cpp side-by-side.
# hf2q must be ≥ 0.95× llama.cpp at each data point.
```

---

## Appendix B: Canonical gotcha cross-reference

Every gotcha from `project_qwen36_architecture.md` is addressed in this ADR. ADR-012 addresses #1-#6, #8, #9; this ADR addresses the inference-only #7 and #10, plus owns the verification of #2-#6 and #8 at load-time.

| Gotcha from memory | Convert-side (ADR-012) | Infer-side (this ADR) |
|---|---|---|
| #1 V-head grouped→tiled | Decision 5 | Decision 12 (verified on load) |
| #2 A_log negation | Decision 6 | Decision 12 (verified on load) |
| #3 dt_bias rename | Decision 6 | Decision 12 (verified on load) |
| #4 Conv1d squeeze | Decision 6 | Decision 12 (verified on load) |
| #5 RMS norm +1 | Decision 6 (audit) | Decision 12 (applied if present) |
| #6 in_proj_qkvz reordering | Decision 6 | Decision 12 (verified on load) |
| #7 Decay-mask log-space clamp | — | **Decision 6 (kernel-internal; applied at runtime in gated_delta_net)** |
| #8 MROPE sections | Decision 7 (metadata) | **Decision 10 (runtime op: rope_multi IMROPE mode)** |
| #9 `full_attention_interval` vs `layer_types` | Decision 2 (parser) | Decision 2 (parser — same logic) |
| #10 Expert weights scale | — | **Decision 13 (applied in MoE forward)** |

Gotchas #7 and #10 are runtime concerns exclusive to this ADR. Conversion does not transform them; the inference engine applies them at op execution time.

---

## Progress log (reverse chronological)

### 2026-04-23 — /loop iter 3 · P1 COMPLETE (TRI_SOLVE landed)

**Scope:** Phase P1 Decision 5 (lower-triangular unit-diagonal solve).

**Delivered in `mlx-native`:**
- `src/shaders/tri_solve.metal` — 2 kernels (f32 + bf16). Forward-substitution kernel `X = L \ B` with L implicitly unit-diagonal; one thread per (col, batch), serial walk over rows, f32 accumulation. Multi-column RHS + batched over a leading dim.
- `src/ops/tri_solve.rs` — `TriSolveParams { n, m, batch }` + `dispatch_tri_solve()`. Validates shape + dtype + overflow; rejects any zero dim.
- `tests/test_tri_solve.rs` — 7 tests:
  - **ADR spec-driven 3×3** with hand-computed golden in file comments (L[1,0]=2, L[2,0]=0.5, L[2,1]=-1; B=[10,4,0]; X=[10,-16,-21]; verified via L·X=B in comments).
  - **ADR acceptance 4×4 random + 4×8 RHS**: measured `|L·X - B|_∞ < 1e-4` (passes).
  - Batched 4×4 × 3 RHS × 5 batches (cross-batch independence verified).
  - Identity (strict-lower=0 → X=B pass-through).
  - BF16 4×4 × 4 RHS with tolerance 1e-2.
  - Zero-N rejection.
  - Element-count mismatch rejection.
- `src/kernel_registry.rs` / `src/ops/mod.rs` — registered both kernels.

**Verification:**
- 7/7 TRI_SOLVE tests green.
- Library test suite: 95/95 pass (no regression).

**Phase map status:**

| Phase | Decisions | Status |
|---|---|---|
| P0  | 3, 4, 7   | COMPLETE |
| P1  | 5         | **COMPLETE** |
| P2  | 10 (mlx)  | Next iter — ROPE_MULTI (IMROPE mode) |
| P3  | 6         | Pending |
| P4–P13 | 1, 2, 8–18 | Pending |

**Aggregate so far:** 31 spec-driven kernel tests, ~900 lines Rust dispatch + ~900 lines Metal, 4 of 6 novel Qwen3.5 mlx-native kernels landed.

**Next iter target:** P2 ROPE_MULTI with IMROPE mode (sections [11,11,10,0], `sector % 3`-cycling theta selection, NeoX-style pair indexing).

### 2026-04-23 — /loop iter 2 · P0 COMPLETE (SSM_CONV landed)

**Scope:** Phase P0 Decision 7 (SSM depthwise causal 1D conv + SiLU).

**Delivered in `mlx-native`:**
- `src/shaders/ssm_conv.metal` — 4 kernels (forward + state-update × f32/bf16). Depthwise causal conv with K=4 (parameterized), fused SiLU, per-seq ring-buffer state. Memory layout `x/y [channels, n_tokens, n_seqs]`, `kernel [K, channels]`, `state [K-1, channels, n_seqs]`. f32 accumulation regardless of input dtype.
- `src/ops/ssm_conv.rs` — `SsmConvParams` struct + `dispatch_ssm_conv()` that encodes BOTH the forward and state-update kernels back-to-back (separate dispatches to avoid `old_state`/`new_state` aliasing when `n_tokens < K-1`). Full shape/dtype validation.
- `tests/test_ssm_conv.rs` — 8 tests including:
  - **ADR acceptance #2** spec-driven: 1-seq × 4-ch × 6-tok × K=4 with random kernel/state; hand-coded scalar CPU reference (`cpu_reference()`) with `|delta| < 1e-4`.
  - **ADR acceptance #3** ring-buffer correctness: monolithic `[0..8]` vs chunked `[0..4] + [4..8]` byte-identical (tolerance 1e-6).
  - Tiny hand-computed 1-ch × 2-tok × K=4 with all taps derived in comments.
  - Multi-seq independence (2 sequences with different state).
  - Decode regime (n_tokens=1 < K-1, exercises state-update's `old_state` branch).
  - BF16 path at 1e-2 tolerance.
  - Zero-channels / K=1 rejection.
- `src/kernel_registry.rs` / `src/ops/mod.rs` — registered all 4 kernels.

**Verification:**
- 8/8 SSM_CONV tests green.
- Library test suite: 95/95 pass (no regression).

**Phase map status (updated):**

| Phase | Decisions | Status |
|---|---|---|
| P0  | 3, 4, 7   | **COMPLETE** (L2_NORM ✓, CUMSUM ✓, SSM_CONV ✓ @ mlx-native) |
| P1  | 5         | Next iter — TRI_SOLVE |
| P2  | 10 (mlx)  | Pending |
| P3  | 6         | Pending |
| P4–P13 | 1, 2, 8–18 | Pending |

**Total P0 footprint:** 24 spec-driven tests green, ~350 lines Rust dispatch + ~550 lines Metal. Cross-repo: `mlx-native@<HEAD+1>`.

**Next iter target:** P1 TRI_SOLVE (forward-substitution on lower-triangular unit-diagonal matrix, multi-column RHS, batched over leading dims).

### 2026-04-23 — /loop iter 1 · P0 partial (L2_NORM + CUMSUM)

**Scope:** Phase P0 Decisions 3 & 4 (SSM_CONV deferred to iter 2).

**Delivered in `mlx-native`:**
- `src/shaders/l2_norm.metal` — f32/f16/bf16 kernels; per-row Euclidean normalization with f32 accumulation. Formula `x / sqrt(sum(x^2) + eps)` derived from the mathematical definition; no llama.cpp source referenced.
- `src/ops/l2_norm.rs` — Rust dispatch wrapper + `register()` helper. 8 spec-driven tests: 3-4-5 triangle, multirow independence, round-trip reconstruction, zero-row with eps, eps damping, bf16 path, zero-rows rejection, dtype-mismatch rejection.
- `src/shaders/cumsum.metal` — f32/bf16 kernels; Hillis-Steele scan with per-thread chunk (CHUNK=32); f32 accumulation. Covers `dim ≤ tg_size × 32 = 8192`.
- `src/ops/cumsum.rs` — Rust dispatch wrapper. 8 spec-driven tests: textbook `[1,2,3,4]→[1,3,6,10]`, negatives/zeros, multirow, large-ones (dim=512), non-power-of-2 random parity (dim=257), bf16 path, zero-dim rejection, oversized-dim rejection.
- `tests/bench_sdpa_tq.rs` — pre-existing broken-window fix: added missing `ring_start: 0` to `FlashAttnVecTqParams` literal (unblocks test-crate compile).
- `src/kernel_registry.rs` / `src/ops/mod.rs` — wired both new kernels into the static registry.

**Verification:**
- 16 new tests all green.
- Full mlx-native lib test suite: 95/95 pass (no regression).
- Pre-existing `test_q4_0_id_vs_norid*` failures (3) confirmed pre-existing — unrelated to this iter; will be tracked as a separate broken window.
- Full hf2q build clean.

**Phase map status:**

| Phase | Decisions | Status |
|---|---|---|
| P0  | 3, 4, 7   | **2/3 done** (L2_NORM ✓, CUMSUM ✓, SSM_CONV pending iter 2) |
| P1  | 5         | Pending |
| P2  | 10 (mlx side) | Pending |
| P3  | 6         | Pending |
| P4–P13 | 1, 2, 8–18 | Pending |

**Next iter target:** SSM_CONV (completes P0), then P1 TRI_SOLVE.
