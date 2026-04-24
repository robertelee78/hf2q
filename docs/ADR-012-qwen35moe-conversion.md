# ADR-012: Qwen3.5-MoE (qwen35moe) Conversion Support — Pure-Rust HF → DWQ GGUF

**Status:** Proposed (2026-04-23)
**Decision Makers:** Robert, Claude
**Related ADRs:** ADR-004 (GGUF compatibility), ADR-006 (mlx-native GPU backend), ADR-008 (candle divorce)
**Related memories:** `project_qwen36_architecture.md`, `project_model_class_split.md`, `project_pure_rust_crate_factory.md`, `project_mlx_native_is_the_strategic_destination.md`, `feedback_hf2q_sovereignty.md`, `feedback_llama_cpp_over_candle.md`, `feedback_no_broken_windows.md`, `feedback_correct_outcomes.md`

---

## Engineering Mantra (load-bearing — read before every session)

Source: `~/Documents/mantra.txt` (Robert, undated). Quoted verbatim.

> **DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.**

**Operational reading for this ADR:**

- **No fallback.** When a Qwen3.5-MoE-specific code path is required (V-head reorder, A_log negation, MROPE section emission, SSM tensor naming, linear-attention layer type), implement it fully in the first cut. No `unimplemented!()`, no "we'll handle shared experts in a follow-up." Either the phase ships conversion-capable code or the phase hasn't shipped.
- **Chesterton's fence.** The hardcoded `base_bits: 4, sensitive_bits: 6` at `src/main.rs:447-448` exists because DWQ-4-6 is the single validated sensitivity profile for Gemma-4. Before parameterizing, read `src/quantize/dwq.rs` and `src/intelligence/auto_quant.rs` to understand *which* layers DWQ-4-6 chose to promote, and *why* — then widen the preset surface without silently changing gemma's behavior.
- **Dive deep.** The authoritative references for every decision in this ADR are `/opt/llama.cpp/convert_hf_to_gguf.py` (Qwen2MoeModel → Qwen3NextModel → _LinearAttentionVReorderBase → Qwen3_5MoeTextModel chain) and `/opt/llama.cpp/src/llama-arch.{h,cpp}` (LLM_ARCH_QWEN35MOE metadata key table). Every TODO in this ADR cites a specific file:line range. Read them.
- **No Python at runtime.** Sovereignty directive (`feedback_hf2q_sovereignty.md`). llama.cpp's Python is a study reference, not a build-time or runtime dependency. Test fixtures produced by `convert_hf_to_gguf.py` are allowed as test oracles (binary data held in-tree); Python in the critical path is not.

---

## Context

### Business problem

Robert released `jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated` on HuggingFace and needs it converted to a DWQ GGUF named `qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq` in two variants (4-bit and 8-bit base precision) for local inference. This mirrors the existing `gemma-4-26B-A4B-it-ara-abliterated-dwq` workflow that hf2q produced.

### Technical problem

hf2q's current convert pipeline (HF safetensors → DWQ → GGUF) is validated only against Gemma-4. Qwen3.6 is the HuggingFace name for a model whose architecture is identical to Qwen3.5 and routes through llama.cpp's `LLM_ARCH_QWEN35MOE` (`general.architecture="qwen35moe"`). This is a hybrid transformer + linear-attention (Gated DeltaNet / SSM) arch with novel structural elements. Per `project_qwen36_architecture.md`, these are confirmed by direct reading of the target model's `config.json`:

- **Hybrid layer sequence.** 40 layers; 3:1 ratio of `linear_attention` to `full_attention` (full_attention at indices 3, 7, 11, ..., 39). Enforced by `full_attention_interval: 4` and explicitly enumerated in `layer_types: [...]`.
- **Gated DeltaNet linear-attention blocks.** Distinct tensor family: `linear_attn.in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`, `conv1d`, `A_log`, `dt_bias`, `dt_proj`. SSM state is f32 (`mamba_ssm_dtype: float32`).
- **Gated full-attention blocks.** `attn_output_gate: true` — new output-gate weight per full-attention block, absent from Gemma.
- **V-head grouped → tiled reorder.** Linear attention uses `num_k_heads=16, num_v_heads=32` → 2 V heads per K head. HF stores V heads grouped by K head; ggml binary ops expect tiled broadcast. Off-by-one in the reorder = silent corruption of decode output.
- **MROPE (multi-axis RoPE) with partial rotary.** `partial_rotary_factor: 0.25` → 64 rotated dims per head of 256. `mrope_section: [11, 11, 10]` (sum = 32 = rotary_dim / 2). `mrope_interleaved: true`. `rope_theta: 10,000,000`.
- **256 experts, 8 active, plus shared experts.** `num_experts: 256`, `num_experts_per_tok: 8`, `shared_expert_intermediate_size: 512` (YES shared experts), `moe_intermediate_size: 512`. Total MoE layer expert tensors: 256 × 3 (gate/up/down) + 3 shared = 771 per MoE layer × 40 layers ≈ 30,840 expert tensors.
- **MTP head.** `mtp_num_hidden_layers: 1` — one Multi-Token Prediction block (DeepSeek-V3 style). Open question whether conversion emits or drops these tensors.
- **New vocabulary.** `vocab_size: 248320`, not the 151K Qwen2/Qwen3 vocab. `bos_token_id == eos_token_id == 248044`.

### Current state inventory (what exists in hf2q today)

The gap between "Gemma works" and "Qwen3.5-MoE works" was mapped by direct reading of the codebase:

| Component | State |
|---|---|
| `src/input/config_parser.rs` | Extracts generic HF fields (architectures, hidden_size, num_experts, num_experts_per_tok, num_attention_heads, num_key_value_heads, num_hidden_layers). **Missing** every Qwen3.5-specific field. |
| `src/backends/gguf.rs` | Handles `gemma4`, `llama`-family default tensor renames (line 1469: covers `llama, mistral, qwen2, qwen3, phi`). **No `qwen35moe` arch string, no `qwen35moe.*` metadata emission.** Unit test at `:2382` asserts `qwen3` dense tensor rename only. |
| `src/preflight.rs::SUPPORTED_LAYER_TYPES` | Supports `attention, full_attention, sliding_attention, moe_attention, linear, mlp, feedforward, ffn, dense, self_attention, cross_attention, grouped_query_attention`. **`linear_attention` is absent.** |
| `src/models/` | Gemma-specific model file(s). **No qwen35moe file.** |
| `src/quantize/dwq.rs` | `DwqConfig { base_bits, sensitive_bits, ... }` parameterized internally. Default `base_bits: 4, sensitive_bits: 6`. |
| `src/main.rs:447-448` | DWQ dispatch **hardcodes** `base_bits: 4, sensitive_bits: 6` regardless of `config.bits`. `--bits` flag silently ignored for DWQ path. **Broken window.** |
| `src/cli.rs:336-337` | `QuantMethod::DwqMixed46` is the **only** DWQ enum variant. No DwqMixed48 or parameterized parser. |
| `src/intelligence/auto_quant.rs` | Routes `arch.contains("qwen")` to `DenseDecoder`; MoE Qwen falls into `GenericMoE`. Sensitivity heuristic **tuned on Gemma's homogeneous transformer blocks**; not hybrid-arch-aware. |
| `src/input/hf_download.rs` | Uses `hf_hub::api::sync::ApiBuilder` with `huggingface-cli` fallback. Unknown behavior for 35B × ~40 safetensors shards, LFS resumption, disk-space preflight. |
| `docs/` | Zero hits for `hf2q convert` as canonical example. Gemma conversion command is oral history. |
| `tests/convert_integration.rs` | Gemma-only. No hybrid-arch or qwen35moe coverage. |

### Why this blows up the Gemma pattern

Every architectural element above maps to a gate in the existing code that currently only lets Gemma through. Attempting `hf2q convert --repo jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated --format gguf --quant dwq-mixed-4-6` today will fail at config parsing before it reaches weight loading — `layer_types` is an unrecognized field, `full_attention_interval` doesn't exist in the parser, `linear_attention` fails preflight.

---

## Strategic Decision

**hf2q implements Qwen3.5-MoE conversion natively in pure Rust. No Python prerequisite. No runtime linkage to candle or llama.cpp. `/opt/mlx-native` is the only sibling repo we depend on.**

This is the application of `feedback_hf2q_sovereignty.md` to the specific question of Qwen3.5-MoE support. Path B (ride on llama.cpp's `convert_hf_to_gguf.py` as a prerequisite, and implement only a GGUF → DWQ → GGUF re-quantization pass in hf2q) was considered and explicitly rejected. Rationale: it would collapse hf2q's identity as a full-pipeline pure-Rust alternative to the Python + C++ status quo. Every future arch that llama.cpp adds (Qwen3-Next, Qwen3.5-VL, DeepSeek-V3, Llama-5, Grok, …) is ported into hf2q on landing.

---

## Non-Goals

Each of these is explicitly *not* in this ADR's scope. They are named so that the ADR's acceptance criteria can be measured against a bounded surface.

1. **Inference / forward-pass graph construction.** The inference session owns the Rust port of `Qwen3_5MoeTextModel` forward for execution, including GATED_DELTA_NET / SSM_CONV / TRI_SOLVE / L2_NORM / CumSum Metal kernels in `/opt/mlx-native`. This ADR consumes the inference engine only as a callable activation-capture backend for DWQ calibration (Phase P5). The inference port's ADR is separate.
2. **Tokenizer runtime changes.** The Qwen3.5 tokenizer (248K vocab) is embedded in the output GGUF via the existing tokenizer-embedding pipeline. If the existing `_set_vocab_qwen` equivalent in hf2q is sufficient for the new vocab, no work. If not, a follow-up ADR.
3. **Qwen3.6 dense (27B) variant.** This ADR targets the MoE variant only, which is what Robert released. The dense variant (`Qwen3_5ForCausalLM`, `LLM_ARCH_QWEN35`) shares most of the linear-attention machinery but is out of scope here. The qwen35moe work should be structured so that a subsequent qwen35 ADR is ≤ 200 LOC.
4. **Multimodal / vision tower.** The local target (`qwen3.6-35b-a3b-abliterix-ega-abliterated`) config drops `vision_config`. The `mmproj-qwen36-F16.gguf` file already in the model directory is a pre-existing artifact from an external converter; hf2q does not produce it. Vision support is a follow-up ADR.
5. **MTP head execution.** Conversion **emits** MTP tensors losslessly (see Decision 11). Whether the inference engine uses them for speculative decoding is the inference session's choice. This ADR does not implement speculative decoding infrastructure.
6. **Qwen3.5-MoE inference coherence gate.** The analogue of `scripts/sourdough_gate.sh` for qwen35moe is blocked on the inference session producing bit-stable output first. This ADR's acceptance is measured against structural GGUF validity and load-acceptance in llama.cpp, not against byte-level inference parity.
7. **Rewrite of the DWQ sensitivity heuristic.** We extend it for the hybrid case; we don't redesign it from scratch. If Phase P4 reveals the existing approach is fundamentally wrong for hybrid arch, that is a follow-up ADR.

---

## Reference Implementations (authoritative)

Every porting decision in this ADR has an upstream reference. These file:line citations are entry points — llama.cpp is a moving target, so verify against the current HEAD of `/opt/llama.cpp` (commit `8bc492ebb` as of 2026-04-24 per `project_qwen36_architecture.md`) before starting.

### Python converter class chain (study order, top-down)

1. `/opt/llama.cpp/convert_hf_to_gguf.py:4553` — `class Qwen2MoeModel(TextModel)`. Base for MoE tensor naming, expert merge, router handling.
2. `/opt/llama.cpp/convert_hf_to_gguf.py:4770` — `class Qwen3NextModel(Qwen2MoeModel)`. Adds linear-attention conversion: A_log negation, dt_bias rename, conv1d squeeze, in_proj_qkvz reordering.
3. `/opt/llama.cpp/convert_hf_to_gguf.py:5259` — `class _LinearAttentionVReorderBase(Qwen3NextModel)`. The grouped-to-tiled V-head reorder. Six cases (in_proj_qkv, in_proj_z, in_proj_a/b, A_log / dt_bias / dt_proj, conv1d, out_proj). Implementation at `:5375-5424`.
4. `/opt/llama.cpp/convert_hf_to_gguf.py:5433` — `class Qwen3_5MoeTextModel(_LinearAttentionVReorderBase)`. Terminal class; `model_arch = gguf.MODEL_ARCH.QWEN35MOE`.

### C++ loader & metadata keys

5. `/opt/llama.cpp/src/llama-arch.cpp:43` — `{ LLM_ARCH_QWEN35MOE, "qwen35moe" }`. The arch string hf2q must emit as `general.architecture`.
6. `/opt/llama.cpp/src/llama-arch.cpp:860` — `case LLM_ARCH_QWEN35MOE:` in `llm_arch_is_hybrid` (or equivalent predicate). Confirms upstream treats qwen35moe as a hybrid state-space/attention architecture.
7. `/opt/llama.cpp/src/llama-arch.h` + `llama-arch.cpp` — `LLM_KV_*` enum and the `LLM_KV_NAMES[]` table (starts around `:180`). Every key the loader reads must be emitted by hf2q. This is the authoritative list; do not derive from blog posts or the GGUF file's observed keys alone.
8. `/opt/llama.cpp/src/models/qwen35moe.cpp` — graph builder (~17 KB). Tells us which tensors the loader expects to find by name for each layer class. Conversion must produce all of them.
9. `/opt/llama.cpp/src/models/delta-net-base.cpp` — linear-attention compute (~16 KB). Informs which SSM-family tensors are load-bearing and which are optional.

### Existing local reference: the `apex.gguf` on disk

`/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf` (25 GB, `general.file_type=17` i.e. IQ2_XS) was produced by an external invocation of `convert_hf_to_gguf.py` + `llama-quantize`. It contains 733 tensors and 43 metadata keys. For Phase P7 integration testing, diffing its **non-weight** metadata against hf2q's output is the primary correctness signal.

---

## Architecture Decisions

Each decision below specifies: **Problem** (what breaks today), **Decision** (what we do), **Acceptance Criteria** (how we know it's done). Decisions are numbered for citation from commits and follow-up issues.

### 1. Arch string and routing

**Problem.** `src/backends/gguf.rs` has no routing for `qwen35moe`. Without it, `general.architecture` is wrong and llama.cpp refuses to load the file.

**Decision.** Add `qwen35moe` as a recognized arch. Detection source: `config.json::architectures[0] == "Qwen3_5MoeForCausalLM"` → arch string `"qwen35moe"`. Dispatch per-arch tensor-rename tables and metadata-emission logic off this arch string. Follow `project_model_class_split.md`: arch-specific code lives in a new file `src/models/qwen35moe.rs`; `src/backends/gguf.rs` holds only dispatch.

**Acceptance criteria.**
- Unit test in `src/backends/gguf.rs` mirroring the existing test at `:2382`: for arch `qwen35moe`, `hf_name_to_gguf("model.layers.0.post_attention_layernorm.weight", "qwen35moe")` returns the correct GGUF name (`blk.0.post_attention_norm` per llama.cpp's Gemma convention OR `blk.0.ffn_norm` per LLaMA — determined by reading `llama-arch.cpp`; the test MUST match what the loader reads).
- `hf2q convert --repo <any Qwen3.5-MoE repo> --format gguf --quant dwq-mixed-4-6 ... ` emits `general.architecture = "qwen35moe"` as readable via any GGUF reader.
- Integration test Phase P7 asserts this key is present and correct.

### 2. Config parser extensions

**Problem.** `src/input/config_parser.rs` doesn't extract the Qwen3.5-specific fields. `project_qwen36_architecture.md` enumerates them all; the parser is missing ≥12 of them.

**Decision.** Extend the parsed config struct with:
- `layer_types: Option<Vec<String>>` — per-layer attention type string array
- `full_attention_interval: Option<u32>`
- `attn_output_gate: Option<bool>`
- `head_dim: Option<u32>` (explicitly decoupled; must not be derived from hidden_size/num_heads when present)
- `partial_rotary_factor: Option<f32>`
- `rope_parameters: Option<RopeParameters>` with nested fields: `mrope_interleaved: bool`, `mrope_section: Vec<u32>`, `rope_theta: f64`, `rope_type: String`, `partial_rotary_factor: f32`
- `linear_conv_kernel_dim: Option<u32>`
- `linear_key_head_dim: Option<u32>`
- `linear_num_key_heads: Option<u32>`
- `linear_value_head_dim: Option<u32>`
- `linear_num_value_heads: Option<u32>`
- `mamba_ssm_dtype: Option<String>` (validated as one of `"float32"`, `"bfloat16"`, `"float16"`)
- `moe_intermediate_size: Option<u32>`
- `shared_expert_intermediate_size: Option<u32>`
- `mtp_num_hidden_layers: Option<u32>`
- `mtp_use_dedicated_embeddings: Option<bool>`
- `output_router_logits: Option<bool>`
- `router_aux_loss_coef: Option<f32>`

Support both `full_attention_interval` (computed) and `layer_types` (explicit enumeration); prefer explicit when present (gotcha #9 from `project_qwen36_architecture.md`).

**Acceptance criteria.**
- Unit test loads `/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/config.json` and asserts all fields parse with the correct values (head_dim=256, num_experts=256, num_experts_per_tok=8, rope_theta=10_000_000, mrope_section=[11,11,10], full_attention_interval=4, layer_types.len()==40, layer_types[3]==`"full_attention"`, layer_types[0]==`"linear_attention"`, shared_expert_intermediate_size=512, etc.).
- Unit test loads Gemma-4-26B-A4B `config.json` and asserts existing parsing is unaffected (regression guard).
- Malformed config (missing required-for-qwen35moe fields when arch is qwen35moe) returns an error with a clear message identifying the missing field — not silent zero-filling.

### 3. Preflight: `linear_attention` layer type

**Problem.** `src/preflight.rs::SUPPORTED_LAYER_TYPES` at `:8-22` doesn't include `linear_attention`. Preflight fails before conversion starts.

**Decision.** Add `"linear_attention"` to the `SUPPORTED_LAYER_TYPES` constant. Cross-reference that if `config.layer_types.contains("linear_attention")`, at least one `full_attention` entry exists (sanity check against misconfigured models).

**Acceptance criteria.**
- Preflight passes on the apex model's config.
- Preflight on a synthetic config with 100% `linear_attention` layers fails with a clear error (hybrid arch requires at least one full-attention layer for KV state anchoring).

### 4. New file: `src/models/qwen35moe.rs`

**Problem.** No file exists to hold Qwen3.5-MoE-specific logic. Per `project_model_class_split.md`, arch-specific code must live in a per-model file under `models/`, not in generic infrastructure.

**Decision.** Create `src/models/qwen35moe.rs` containing:
- `pub struct Qwen35MoeConvertContext { /* parsed hparams + arch constants */ }`
- `pub fn hf_tensor_name_to_gguf(hf_name: &str, ctx: &Qwen35MoeConvertContext) -> String`
- `pub fn transform_tensor(hf_name: &str, data: TensorData, ctx: &Qwen35MoeConvertContext) -> TensorData` — dispatches to V-head reorder, A_log negation, conv1d squeeze, in_proj_qkvz reordering, expert merge
- `pub fn emit_metadata(writer: &mut GgufWriter, ctx: &Qwen35MoeConvertContext) -> Result<()>` — writes all `qwen35moe.*` keys
- `pub fn is_linear_attention_layer(layer_idx: usize, ctx: &Qwen35MoeConvertContext) -> bool` — consults `layer_types` or derives from `full_attention_interval`
- `pub fn is_full_attention_layer(...)` — complement
- Unit tests per transform (each gotcha from `project_qwen36_architecture.md` gets its own test with known input → known output pairs)

Gemma logic stays where it is; no refactor of Gemma in this ADR.

**Acceptance criteria.**
- File exists at `src/models/qwen35moe.rs`, public API as above.
- Module registered in `src/models/mod.rs`.
- Per-transform unit tests cover all 10 gotchas from `project_qwen36_architecture.md`.

### 5. V-head grouped → tiled reorder

**Problem.** Linear attention has `num_k_heads=16, num_v_heads=32` (Qwen3.5-MoE 35B) or `num_k_heads=16, num_v_heads=48` (Qwen3.5 27B dense). HF stores V heads grouped by K head (`[G0_v0..v{r-1}, G1_v0..v{r-1}, ...]`); ggml binary ops expect tiled broadcast (`[K0, K1, ..., K0, K1, ...]`). Off-by-one = silent corruption.

**Decision.** Port the six cases in `/opt/llama.cpp/convert_hf_to_gguf.py:5375-5424` verbatim to Rust. Each case matches a suffix in `hf_name`:
- `.linear_attn.in_proj_qkv.weight` — reorder V rows only (positions after `q_dim + k_dim`)
- `.linear_attn.in_proj_z.weight` — reorder rows (num_v_heads * head_v_dim)
- `.linear_attn.in_proj_b.weight` / `.linear_attn.in_proj_a.weight` — reorder rows (num_v_heads, head_dim=1)
- `.A_log` / `.dt_bias` / `.dt_proj` — 1D parameters with num_v_heads elements; reorder along last dim
- `.conv1d` — reorder only the V channel portion (channels after `head_k_dim * num_k_heads * 2`)
- `.linear_attn.out_proj.weight` — reorder columns (input dimension)

Use the same helper logic: `reorder_v_heads(tensor, dim, num_k_heads, num_v_per_k, head_dim)`.

**Acceptance criteria.**
- Unit test per case: construct a tensor with known marker values at each V-head position, apply the transform, assert the output has markers at the expected tiled positions.
- Round-trip test: apply the reorder, apply its inverse (derivable from same params), assert byte-identical to input. This protects against bugs where the reorder is partially correct (permutation subset applied twice).
- Oracle test: if a llama.cpp-produced BF16 `.linear_attn.*` tensor fixture is checked in, hf2q's output must match it byte-for-byte after the reorder.

### 6. Qwen-specific tensor transforms (non-V-reorder)

**Problem.** Multiple conversion-time tensor transforms Qwen3.5 requires that Gemma doesn't.

**Decision.** Implement each transform in `src/models/qwen35moe.rs`:

| Transform | Trigger | Action |
|---|---|---|
| A_log negation | Tensor name ends with `.A_log` | Output = `-exp(input)`. Miss it → NaN on inference. Gotcha #2 from project_qwen36_architecture.md. |
| dt_bias rename | Tensor name ends with `.dt_bias` | Rename to `.dt_proj.bias` per llama.cpp convention. Gotcha #3. |
| conv1d squeeze | Tensor name ends with `.conv1d.weight` and shape is `[k, 1, d]` | Squeeze dim 1 to produce `[k, d]`. Gotcha #4. |
| in_proj_qkvz reorder | Tensor name ends with `.linear_attn.in_proj_qkvz.weight` (fused variant) | Split into Q/K/V/Z head-grouped reorder. Gotcha #6. |
| RMS norm +1 bias | Tensor name ends with `_norm.weight` or similar norm tensor | Qwen uses `gamma + 1` convention. Gotcha #5. **Audit decision pending**: verify whether Qwen3.5 specifically still uses +1 (some Qwen2/Qwen3 variants don't) — check `qwen35moe.cpp`'s `build_norm` calls. |

**Acceptance criteria.**
- One unit test per transform with hand-constructed fixtures.
- A_log transform test asserts `|output + exp(input)| < 1e-6` for random inputs.
- Conv1d squeeze test asserts shape change and byte-identical data.
- An audit note in `src/models/qwen35moe.rs` documents whether Qwen3.5 RMS norm adds +1 (with citation to the specific llama.cpp line consulted).

### 7. GGUF metadata emission for `qwen35moe.*`

**Problem.** llama.cpp's `qwen35moe` loader reads a specific set of `qwen35moe.*` keys. Without them, the file loads and silently produces wrong output.

**Decision.** Emit every metadata key the loader reads. The authoritative list is extracted by reading `LLM_KV_*` references in `/opt/llama.cpp/src/models/qwen35moe.cpp` and `/opt/llama.cpp/src/llama-arch.{h,cpp}`. Initial catalog (engineer must verify against HEAD):

**General / standard transformer keys (all emitted under `qwen35moe.*`):**
- `block_count` (= num_hidden_layers = 40)
- `context_length` (= max_position_embeddings = 262144)
- `embedding_length` (= hidden_size = 2048)
- `feed_forward_length` (the dense FFN width if any dense layers; 0/absent if all-MoE)
- `attention.head_count` (= num_attention_heads = 16)
- `attention.head_count_kv` (= num_key_value_heads = 2)
- `attention.head_key_length` (= head_dim = 256; MUST emit explicitly since decoupled from hidden_size/num_heads)
- `attention.head_value_length` (= head_dim = 256)
- `attention.layer_norm_rms_epsilon` (= rms_norm_eps = 1e-6)
- `attention.output_gate` (= attn_output_gate = true)

**MoE:**
- `expert_count` (= num_experts = 256)
- `expert_used_count` (= num_experts_per_tok = 8)
- `expert_feed_forward_length` (= moe_intermediate_size = 512)
- `expert_shared_feed_forward_length` (= shared_expert_intermediate_size = 512)

**Hybrid / linear-attention:**
- `full_attention_interval` (= 4)
- `layer_types` as a string array OR a compact encoding per llama.cpp's convention (verify against `qwen35moe.cpp` load path)
- Linear-attention hparams: conv_kernel_dim, key_head_dim, num_key_heads, value_head_dim, num_value_heads (exact key names per llama.cpp — use `grep LLM_KV_SSM` and `grep LLM_KV_LINEAR` in `/opt/llama.cpp/src/llama-arch.cpp`)

**RoPE:**
- `rope.dimension_count` (= rotary_dim = 64)
- `rope.freq_base` (= rope_theta = 10_000_000)
- `rope.dimension_sections` as `[mrope_section..., 0]` padded to 4 slots (see `convert_hf_to_gguf.py:1149-1155` — pad to length 4 with zeros, call `add_rope_dimension_sections`)
- `rope.mrope_interleaved` (= true)

**MTP:**
- `mtp_num_hidden_layers` (= 1) — or the llama.cpp-named `nextn_predict_layers` (see `llama-arch.cpp:194`). Use whatever the loader reads.

**Tokenizer / vocab (standard pipeline, but verify the Qwen3.5 vocab loads correctly):**
- `tokenizer.ggml.model`, `tokens`, `merges`, `scores`, `token_type`, `bos_token_id`, `eos_token_id`, `pad_token_id`, `chat_template`, `pre` — per ADR-004 decision 5.

**Acceptance criteria.**
- An "oracle diff" test loads both the hf2q-produced GGUF and the pre-existing `apex.gguf`, extracts all metadata keys from each, and asserts every `qwen35moe.*` key present in apex.gguf is also present in hf2q's output with either (a) the same value for invariants (arch, vocab, rope_theta, head_count, …) or (b) a documented diff (e.g. `general.file_type` legitimately differs because apex is IQ2_XS and hf2q output is DWQ-mixed-4-6).
- Missing any `qwen35moe.*` key that apex.gguf has → integration test fails loudly with the missing key name in the assertion message.

### 8. Tensor naming for hybrid layers

**Problem.** Full-attention layers (indices 3, 7, ..., 39) and linear-attention layers (all others) have completely different tensor sets. GGUF block indices must be the same, but tensor names must differ per layer type.

**Decision.** In `hf_tensor_name_to_gguf()`:

**For full-attention layers** (HF `model.layers.{L}.self_attn.*`):
- `q_proj.weight` → `blk.{L}.attn_q.weight`
- `k_proj.weight` → `blk.{L}.attn_k.weight`
- `v_proj.weight` → `blk.{L}.attn_v.weight`
- `o_proj.weight` → `blk.{L}.attn_output.weight`
- `q_norm.weight` → `blk.{L}.attn_q_norm.weight`
- `k_norm.weight` → `blk.{L}.attn_k_norm.weight`
- Output gate (new in qwen35moe; locate exact HF name — likely `self_attn.output_gate.weight` or `self_attn.gate_proj.weight` — verify in HF model): → `blk.{L}.attn_gate.weight` (LLM_TENSOR_ATTN_GATE at `llama-arch.cpp:370`)

**For linear-attention layers** (HF `model.layers.{L}.linear_attn.*`):
- Map each of `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`, `conv1d`, `A_log`, `dt_bias`, `dt_proj`, norms to the corresponding llama.cpp `blk.{L}.ssm.*` / `blk.{L}.linear_attn.*` names. Exact names per llama.cpp's `LLM_TENSOR_*` enum in `src/llama-arch.h` — verify, do not guess.

**For both layer types:**
- `input_layernorm.weight` → `blk.{L}.attn_norm.weight`
- `post_attention_layernorm.weight` → depends on llama.cpp's Qwen3.5 convention — either `blk.{L}.post_attention_norm` (Gemma convention) or `blk.{L}.ffn_norm` (LLaMA convention). **Read qwen35moe.cpp and verify.** Do not assume.

**MoE FFN (every layer):**
- `mlp.gate.weight` → `blk.{L}.ffn_gate_inp.weight` (router)
- `mlp.experts.{E}.gate_proj.weight` for E in 0..256 → merged into `blk.{L}.ffn_gate_exps.weight` shape `[256, moe_intermediate_size, hidden_size]`
- Same for `up_proj` → `ffn_up_exps` and `down_proj` → `ffn_down_exps`
- `mlp.shared_experts.gate_proj.weight` → `blk.{L}.ffn_gate_shexp.weight`
- Same for `up_proj` and `down_proj`

**Output head:**
- `model.embed_tokens.weight` → `token_embd.weight`
- `lm_head.weight` → `output.weight` (or tied to embed if `tie_word_embeddings=false` … verify; the target model has `tie_word_embeddings: false`, so lm_head is independent)
- `model.norm.weight` → `output_norm.weight`

**MTP head (see Decision 11):**
- Emitted as `blk.{40}.nextn.*` tensors per the `LLM_TENSOR_NEXTN_*` names at `llama-arch.cpp:447-450`.

**Acceptance criteria.**
- Unit test per layer type (full vs linear) with a synthetic tensor name list → expected GGUF name list.
- Integration test (Phase P7) extracts all tensor names from hf2q's output GGUF and asserts the set matches the set in apex.gguf modulo mmproj tensors and quantization-type suffixes.

### 9. MoE expert tensor merge

**Problem.** HF stores each expert as a separate tensor (`model.layers.{L}.mlp.experts.{E}.gate_proj.weight` for E in 0..256). GGUF expects a single 3D tensor per projection per layer.

**Decision.** In the convert pipeline, collect all 256 per-expert tensors for a given layer × projection combination, stack along a new first axis, and emit as a single 3D tensor. Perform this per-layer to avoid 40 × 256 × 3 = 30,720 tensor holds in memory simultaneously. Streaming: emit layer N's expert-merged tensor before starting layer N+1. Mirror `_LinearAttentionVReorderBase`'s memory discipline.

Shared experts are **not** merged — they're singletons: `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp`.

**Acceptance criteria.**
- Unit test builds 4 fake expert tensors with distinguishable values, merges them, asserts shape is `[4, inter, hidden]` and each expert's data appears at its expected slice.
- Memory profile during 35B conversion stays under 64 GB peak (target: comparable to Gemma-4 26B's ~54 GB peak per ADR-004 decision 8). If the layer-streaming discipline isn't enough for 256 experts × 3 projections × 40 layers, sub-layer streaming (per-projection) is the documented fallback — but only invoke after measuring.

### 10. DWQ bit-pair parameterization + close the broken window

**Problem.** `src/main.rs:447-448` hardcodes `base_bits: 4, sensitive_bits: 6` regardless of `config.bits`. `--bits 8 --quant dwq-mixed-4-6` accepts the flag and silently ignores it. `feedback_no_broken_windows.md` applies. Separately, the 4-AND-8 ask is unrepresentable in the current CLI vocabulary (`DwqMixed46` is the only enum variant at `src/cli.rs:336-337`).

**Decision.** Both changes in one commit:

(a) Add enum variants `DwqMixed48`, `DwqMixed68`, `DwqMixed28` to `QuantMethod`. Display as `"dwq-mixed-4-8"`, `"dwq-mixed-6-8"`, `"dwq-mixed-2-8"`.

(b) At `src/main.rs:447-448`, replace literal `4` / `6` with a lookup from `config.quant` → `(base_bits, sensitive_bits)` tuple. Keep `config.bits` for non-DWQ paths; for DWQ, error clearly if `--bits N` is passed (rather than silently ignoring) with message `"--bits is not used for DWQ; use --quant dwq-mixed-N-M to choose bit-pair variants"`.

(c) Output-filename auto-naming includes the DWQ variant: default output when `--quant dwq-mixed-4-6` is `{model-name}-dwq46.gguf`; for `dwq-mixed-4-8` it's `{model-name}-dwq48.gguf`. User-provided `--output` overrides.

**Non-goal** in this ADR: re-tuning the sensitivity heuristic for the wider bit gap. The current heuristic picks layers based on an activation-sensitivity score; the score is orthogonal to the promotion target (6 vs 8). Promotion is a magnitude change, not a selection change. Heuristic audit = Decision 12.

**Acceptance criteria.**
- `hf2q convert --help` lists all three new variants.
- Unit test: dispatch table maps each variant to the correct `(base, sensitive)` tuple.
- Unit test: parsing `--bits 5 --quant dwq-mixed-4-6` errors with the documented message.
- Integration test: two side-by-side outputs produced from the same input (dwq46 and dwq48) collide on neither filename nor internal tensor data (they differ in quant type of sensitive tensors).
- Gemma-4 convert with `dwq-mixed-4-6` produces byte-identical output to pre-change HEAD. **This is the Chesterton's-fence regression guard.**

### 11. MTP head handling

**Problem.** `mtp_num_hidden_layers: 1` → the model has a Multi-Token Prediction head (one small transformer block after the main stack). HF stores these as `model.mtp.*` tensors. llama.cpp represents them as `blk.{n_layer}.nextn.*` per `LLM_TENSOR_NEXTN_*` at `llama-arch.cpp:447-450`. Inference session decides whether to use MTP for speculative decoding; conversion must preserve the tensors losslessly.

**Decision.** Emit MTP tensors. Map HF `model.mtp.layers.0.*` → GGUF `blk.{num_hidden_layers}.nextn.*` (i.e. block 40 for the 40-layer MoE). Specific mappings per `LLM_TENSOR_NEXTN_*` names: `nextn.eh_proj`, `nextn.embed_tokens`, `nextn.enorm`, `nextn.hnorm`, plus any attention/FFN tensors the MTP block contains. Emit `qwen35moe.nextn_predict_layers = 1` metadata key per `llama-arch.cpp:194`.

**Acceptance criteria.**
- Tensor-name test: `model.mtp.layers.0.embed_tokens.weight` → `blk.40.nextn.embed_tokens.weight`.
- If `mtp_num_hidden_layers == 0` (future Qwen variants without MTP), the MTP emission path is skipped cleanly with no empty tensors.
- `qwen35moe.nextn_predict_layers` metadata key present and correct.

### 12. DWQ sensitivity heuristic for hybrid arch

**Problem.** `src/intelligence/auto_quant.rs` and `src/quantize/dwq.rs`'s sensitivity heuristic was tuned on Gemma's homogeneous transformer blocks. Qwen3.5-MoE has three tensor cohorts that respond to quantization differently:
- **Full-attention tensors** (standard transformer; existing heuristic applies)
- **Linear-attention / SSM tensors** (`A_log`, `dt_*`, `conv1d`, `in_proj_*` — small, numerically load-bearing; `A_log` in particular is an exponentiated parameter where quantization error compounds)
- **MoE tensors** (router gate — tiny, must stay high-precision; experts — 256× the count, individually less sensitive; shared experts — always hot, higher sensitivity than individual experts)

**Decision.** Extend the sensitivity scorer with a three-cohort prior: SSM state tensors (`A_log`, `dt_bias`, `dt_proj`, `conv1d`) are **always** promoted to `sensitive_bits` (never base_bits), regardless of activation score. Router gate tensors (`ffn_gate_inp`) same. Shared expert tensors (`ffn_*_shexp`) same. Individual expert tensors (`ffn_*_exps`) use the existing activation-score-driven heuristic but with a higher promotion threshold (details: exact threshold tuned empirically against measured KL-divergence per ADR-004 decision 7's fallback methodology). Full-attention tensors unchanged from current heuristic.

**Audit requirement** (Chesterton's fence): before wiring this in, read `src/intelligence/auto_quant.rs::build_component_overrides` and `src/quantize/dwq.rs::calibrate` in full, and explain in a code comment how the new cohort priors interact with existing per-tensor overrides. Do not introduce cohort priors that silently override a user-supplied `--sensitive-layers` range.

**Acceptance criteria.**
- Test: on a synthetic qwen35moe-shaped config, the sensitivity scorer marks all `A_log`, `dt_*`, `conv1d`, `ffn_gate_inp`, `ffn_*_shexp` tensors as "promote to sensitive_bits" before any activation data is collected.
- Test: `--sensitive-layers 0-10` user override is still honored.
- Test: Gemma-4 sensitivity choices are byte-identical pre- and post-change (the new cohort priors only fire for qwen35moe arch).
- Measured KL divergence on a calibration subset for Gemma-4 DWQ-4-6: ≤ pre-change value (never degrades).

### 13. DWQ activation calibration for hybrid arch

**Problem.** Activation-based DWQ calibration requires a forward pass. The existing forward pass at `src/quantize/dwq.rs` hooks into hf2q's Gemma inference engine (layer-streamed per ADR-004 decision 8). qwen35moe forward passes through linear-attention layers require SSM state, GATED_DELTA_NET Metal kernels, and MROPE — all of which live in `/opt/mlx-native` and are being built by the parallel inference session.

**Decision.** Define a stable activation-capture API that the inference session implements, and the DWQ pipeline consumes. Interface sketch:

```rust
// in src/quantize/dwq/forward.rs or equivalent
pub trait ActivationCapture {
    fn run_calibration_prompt(&mut self, tokens: &[u32]) -> Result<LayerActivations>;
}

pub struct LayerActivations {
    pub per_layer_inputs: Vec<TensorF32>,   // [n_layers][hidden_size] aggregated over calibration prompts
    pub per_layer_outputs: Vec<TensorF32>,
    // ... any other stats the sensitivity scorer reads
}
```

The inference session's qwen35moe forward implements `ActivationCapture`. hf2q's DWQ pipeline owns the trait definition and the sensitivity scorer; the inference engine owns the implementation. Cross-session coordination point: the trait definition.

**Acceptance criteria.**
- Trait definition lands in hf2q before inference session starts coding its implementation.
- A mock `ActivationCapture` that returns zeros is used for hf2q-side unit tests; the real inference-engine-backed implementation is used only for end-to-end conversion.
- If the inference session's engine isn't ready when the rest of Phase 1-4 lands, DWQ falls back to weight-space calibration (existing fallback path at `src/quantize/dwq.rs` when `use_activations: false`). **This fallback is named and dated — it expires when the inference engine lands. Per `feedback_never_ship_fallback_without_rootcause.md`.**

### 14. HF download robustness for 35B

**Problem.** `src/input/hf_download.rs` uses `hf_hub::api::sync::ApiBuilder` with `huggingface-cli` fallback. Unknown whether it correctly handles: (a) ~40 safetensors shards for a 35B model; (b) LFS resumption mid-shard on network interruption; (c) gated/auth-required repos; (d) disk-space preflight before starting a ~70 GB download.

**Decision.** Add preflight:
- Before starting download, `df` the target directory and assert ≥ 150 GB free (70 GB bf16 weights + 73 GB DWQ intermediate peak per ADR-004 decision 8 extrapolated + 10 GB margin).
- Error with a clear message if insufficient: `"Qwen3.5-MoE 35B requires ≥150 GB free in {path}; found {N} GB. Free space or change --cache-dir."`
- Verify shard resumption: if download is interrupted and `hf2q convert` is re-invoked, it must resume, not restart. Existing `hf_hub` API sync behavior — verify and add regression test.

**Acceptance criteria.**
- Unit test: mocks `fs2::available_space` to return < 150 GB, asserts error is raised with the documented message.
- Manual test (documented in `docs/converting-qwen35moe.md`): `Ctrl+C` during download → re-invoke → resumes from the partial shard, does not redownload completed shards.

### 15. Sidecar file preservation

**Problem.** Gemma's output dir contains `chat_template.jinja`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json` alongside the `.gguf`. Unknown whether hf2q writes these automatically or they were manually copied. For qwen35moe the tokenizer and chat template are both different from Gemma's.

**Decision.** During convert, copy the HF repo's sidecar files (`chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json`, `config.json`, `generation_config.json`, `special_tokens_map.json`) into the output directory alongside the produced `.gguf`. Preserve file content byte-identically. If any are missing in the HF repo, skip silently (not all models ship all sidecars).

**Acceptance criteria.**
- Integration test: output directory for `apex`-equivalent input has the 5+ sidecar files present with byte-identical content to the HF source.
- Test that Gemma-4 conversion still produces the same sidecar set it did before.

---

## Phase plan

Phases are **dependency-ordered**, not priority-ordered. Each phase has a single owner claim per `feedback_swarm_sequential_when_shared_build.md` (shared Cargo target = sequential).

### P0 — Broken-window fix (standalone, do first)

**Scope:** Decision 10 only. DWQ bit-pair parameterization + `--bits`-silently-ignored fix + output filename disambiguation.

**Why first:** Orthogonal to every other phase. Unblocks the "4 AND 8" user ask independent of qwen35moe timing. Lands as a small, reviewable PR.

**Deliverables:**
- `src/cli.rs`: new `DwqMixed48`, `DwqMixed68`, `DwqMixed28` enum variants
- `src/main.rs:447-448`: dispatch-driven `(base_bits, sensitive_bits)` lookup
- `src/quantize/dwq.rs`: ensure `DwqConfig` honors the dispatched values
- Output-filename naming logic
- Unit tests per Decision 10's acceptance criteria
- Gemma-4 regression test (byte-identical output for `dwq-mixed-4-6`)

**Acceptance:**
- All Decision 10 acceptance criteria met
- `hf2q convert --repo <gemma> --quant dwq-mixed-4-8 ...` produces a different, loadable GGUF than the 4-6 variant
- CI green, Gemma regression byte-identical

**Estimated LOC:** ~150

### P1 — Config ingestion

**Scope:** Decisions 2 and 3.

**Deliverables:**
- Extended `Config` struct in `src/input/config_parser.rs` with the 18 new fields
- Nested `RopeParameters` struct
- `layer_types` / `full_attention_interval` dual-support logic
- `SUPPORTED_LAYER_TYPES` extension
- Unit tests loading the apex model's `config.json` and Gemma-4's `config.json`

**Acceptance:**
- Decision 2 & 3 criteria met
- No regression on Gemma config parsing

**Estimated LOC:** ~300

### P2 — Scaffold `src/models/qwen35moe.rs` and V-head reorder

**Scope:** Decisions 4 and 5.

**Deliverables:**
- New file `src/models/qwen35moe.rs` with public API per Decision 4
- V-head reorder implementation with 6 case handlers
- Unit tests per Decision 5's acceptance criteria (per-case + round-trip + oracle)
- Fixtures: small hand-constructed tensors exhibiting the grouped-vs-tiled ordering

**Acceptance:**
- Decision 4 & 5 criteria met
- Module builds, tests pass

**Estimated LOC:** ~500

### P3 — Qwen-specific non-reorder transforms

**Scope:** Decision 6.

**Deliverables:**
- A_log negation, dt_bias rename, conv1d squeeze, in_proj_qkvz reorder, RMS norm +1 audit note
- One unit test per transform
- Code comment documenting the RMS norm +1 audit with citation

**Acceptance:** Decision 6 criteria met.

**Estimated LOC:** ~250

### P4 — GGUF metadata + tensor naming

**Scope:** Decisions 1, 7, 8, 11.

**Deliverables:**
- Arch routing in `src/backends/gguf.rs`
- `emit_metadata()` in `qwen35moe.rs` with all `qwen35moe.*` keys per Decision 7's catalog
- Tensor-name mapping per Decision 8 (full-attn, linear-attn, MoE, shared experts, output head, MTP)
- "Oracle diff" test framework that extracts metadata from both hf2q output and a reference GGUF and asserts set-equality
- Integration-ready but not yet end-to-end (needs P6 and P7)

**Acceptance:** Decisions 1, 7, 8, 11 criteria met.

**Estimated LOC:** ~600

### P5 — Expert merge + MoE pipeline

**Scope:** Decision 9.

**Deliverables:**
- Layer-streaming expert merge for 256 experts × 3 projections per MoE layer
- Shared expert singleton handling
- Memory profile measurement on the 35B target (must stay < 64 GB peak)
- Unit test with 4 synthetic experts

**Acceptance:** Decision 9 criteria met; memory target met on real 35B convert.

**Estimated LOC:** ~250

### P6 — DWQ integration for hybrid arch

**Scope:** Decisions 12 and 13.

**Deliverables:**
- Cohort-prior extension to sensitivity scorer per Decision 12
- `ActivationCapture` trait definition per Decision 13
- Mock implementation for hf2q-side tests
- Wire into convert pipeline with documented fallback when inference engine isn't ready
- KL-divergence regression guard on Gemma-4 DWQ-4-6

**Acceptance:** Decisions 12 & 13 criteria met.

**Estimated LOC:** ~300

### P7 — Integration test, HF download hardening, sidecar, docs

**Scope:** Decisions 14, 15; full end-to-end convert test.

**Deliverables:**
- `tests/convert_qwen35moe_integration.rs` — synthetic tiny qwen35moe end-to-end, asserts all metadata and tensor-name invariants vs. a checked-in llama.cpp-produced reference fixture
- HF download preflight (Decision 14)
- Sidecar file copy (Decision 15)
- `docs/converting-qwen35moe.md` — canonical command with the `--repo` + `--quant dwq-mixed-4-6` and `--quant dwq-mixed-4-8` invocations
- `docs/converting-a-model.md` — generic convert-command reference (also documents Gemma retroactively)
- Update `docs/shipping-contract.md` to indicate qwen35moe is an additional model class with its own acceptance criteria (lighter than Gemma since inference coherence is out of scope here)

**Acceptance:**
- End-to-end integration test passes on synthetic data
- End-to-end real-model test produces a GGUF that loads in llama.cpp with no errors (loading is acceptance; inference coherence is out of scope and delegated to the inference session's future ADR)
- Docs cover both DWQ variants and both model classes

**Estimated LOC:** ~400 (excluding docs word count)

### Totals

- Code LOC: ~2,250 across P0–P7
- Tests: ≥ 1 unit test per transform, 3 integration tests (full GGUF extraction, Gemma regression, qwen35moe end-to-end), 1 memory-profile measurement
- Docs: 3 new/updated docs
- Timeline: sequential due to shared Cargo target; P0 parallelizable with any phase since it touches different files

---

## Test strategy

### Unit tests (per-transform)

Located alongside implementations in `src/models/qwen35moe.rs`. Each tensor transform in Decisions 5, 6, and 9 has a test with hand-constructed tiny inputs and known outputs.

### Regression tests (Chesterton's fence)

- **Gemma-4 DWQ-4-6 byte-identical output** before vs. after P0 (bit-pair parameterization). This is the only way to know P0 didn't accidentally change the Gemma codepath.
- **Gemma-4 sensitivity choices byte-identical** before vs. after P6 (hybrid cohort priors). Cohort priors must fire only when arch is qwen35moe.

### Integration tests

- `tests/convert_qwen35moe_integration.rs`: synthetic tiny-model end-to-end. 2 linear + 1 full-attn layers, 4 experts, hidden_size=64, head_dim=16. Weights are deterministic from a fixed seed.
- Output GGUF is read back by hf2q's own GGUF reader and asserted byte-identical (where possible) or structurally-identical (metadata keys, tensor names, tensor shapes) against a checked-in reference fixture produced by `convert_hf_to_gguf.py --outtype bf16` on the same synthetic weights. Reference fixture is held in-tree under `tests/fixtures/qwen35moe_tiny_reference.gguf`. Production of the fixture is a one-time offline step documented in `tests/fixtures/README.md`; Python is **not** invoked at test time.

### Oracle tests

- For select tensor transforms (V-reorder, A_log negation), fixtures of known-correct output from llama.cpp's Python script are checked into `tests/fixtures/` and used as bit-exact oracles.

### Real-model smoke test

Documented manual test in `docs/converting-qwen35moe.md`:

```
hf2q convert --repo jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated \
  --format gguf --quant dwq-mixed-4-6 \
  --output models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/out.gguf

llama-cli --model models/.../out.gguf -p "Hello" -n 8
```

Acceptance: llama-cli loads the file without errors and emits 8 tokens. (Coherence is out of scope; this is a "file is not corrupted" check.)

---

## Risks and mitigations

### R1: llama.cpp changes its `qwen35moe.*` key naming

**Likelihood:** Medium — `qwen35moe` is new; names may churn.
**Impact:** hf2q output stops loading in newer llama.cpp.
**Mitigation:** Pin the reference commit in `tests/fixtures/` to the version used to generate fixtures. CI test alerts on divergence. Doc `docs/converting-qwen35moe.md` lists the minimum llama.cpp version that loads hf2q's output. When upstream breaks, treat it as a new ADR addendum, not a CVE.

### R2: V-head reorder has a subtle bug that loads but produces garbage

**Likelihood:** Medium — this is the named silent-failure mode in `project_qwen36_architecture.md` gotcha #1.
**Impact:** GGUF loads, inference produces plausible-looking nonsense. No automated test catches it without inference.
**Mitigation:** Oracle tests (Decision 5 acceptance) against llama.cpp-produced BF16 tensor fixtures. The inference session's sourdough-gate analogue for Qwen will catch this downstream. Do not ship to Robert's end-user workflow without either the oracle passing or the inference gate green.

### R3: DWQ calibration can't run because inference session isn't ready

**Likelihood:** High — cross-session coordination is the hardest schedule to align.
**Impact:** P6 stalls; we can only produce weight-space DWQ, which is inferior.
**Mitigation:** Weight-space fallback is explicitly allowed with dated exit condition (Decision 13 acceptance). First-ship uses weight-space; activation-based is a landed-but-gated improvement. `feedback_never_ship_fallback_without_rootcause.md` satisfied by the dated exit.

### R4: 35B conversion OOMs

**Likelihood:** Medium — Gemma-4 26B hit 54 GB peak; 35B MoE at ~256 experts has 10× the expert tensor count.
**Impact:** Convert crashes partway. User loses hours.
**Mitigation:** Preflight disk check (Decision 14). Per-layer expert-merge streaming (Decision 9). If even that is insufficient, per-projection sub-streaming — measure first, don't implement preemptively.

### R5: The target model's chat template / tokenizer needs pipeline changes

**Likelihood:** Low-medium — Qwen3.5 uses a new 248K vocab; existing hf2q tokenizer-embed logic was tested on Gemma's different format.
**Impact:** Output GGUF loads but decodes gibberish due to wrong chat template or vocab encoding.
**Mitigation:** Non-goal 2 explicitly reserves follow-up ADR scope for tokenizer changes. Phase P7 sidecar preservation (Decision 15) ensures the HF tokenizer files are present next to the GGUF for external tools. If hf2q's embed produces wrong output, open a follow-up ADR on the embed path rather than extending this one.

### R6: `--repo` download of ~70 GB fails and re-download wastes a day

**Likelihood:** Medium.
**Impact:** Lost time.
**Mitigation:** Decision 14 preflight + resumption test.

### R7: The existing `--bits` flag had users who relied on the silent-ignore behavior

**Likelihood:** Low — the flag was silently ignored, so nobody had any real dependency on it for DWQ.
**Impact:** If any automation passed `--bits` + `--quant dwq-mixed-4-6` "just in case," it now errors.
**Mitigation:** Accept the breaking change. Per `feedback_no_broken_windows.md` this is the right call. Error message (Decision 10c) clearly explains the migration.

---

## Open questions

1. **Exact llama.cpp loader key names for linear-attention hparams.** `LLM_KV_SSM_CONV_KERNEL`, `LLM_KV_LINEAR_ATTN_*`, or something different? Must be extracted from `/opt/llama.cpp/src/llama-arch.cpp` at engineer start time. Do not guess.
2. **Post-attention layernorm name** for qwen35moe: `blk.{L}.post_attention_norm` (Gemma convention) or `blk.{L}.ffn_norm` (LLaMA convention)? Resolved by reading `qwen35moe.cpp`'s `build_norm` calls.
3. **Does Qwen3.5 RMS norm use the +1 Qwen convention?** Gotcha #5 in `project_qwen36_architecture.md`. Audit in P3.
4. **Output-gate weight HF tensor name.** Likely `self_attn.output_gate.weight` or `self_attn.gate_proj.weight` — verify by inspecting the actual HF safetensors keys in the target model.
5. **MTP tensor full set.** Decision 11 lists the `nextn.*` names in `llama-arch.cpp:447-450` — confirm which subset Qwen3.5-MoE's MTP produces.
6. **DWQ sensitivity heuristic — is per-tensor the right granularity, or do we need per-expert?** For 256 experts per MoE layer, treating them as a single `ffn_gate_exps` tensor may mask a subset of experts that are individually sensitive. Investigation during P6.

---

## Dependencies on other work (cross-ADR)

- **Inference session (separate ADR, in flight):** Implements qwen35moe forward pass in Rust including linear-attention, MROPE, SSM state, Metal kernels in `/opt/mlx-native`. Activation-capture trait (Decision 13) is the cross-session coordination point. **Needed by P6 for activation-based DWQ; weight-space fallback unblocks P6 landing without it.**
- **`/opt/mlx-native` Metal kernels:** GATED_DELTA_NET, SSM_CONV, TRI_SOLVE, L2_NORM, CumSum per `project_qwen36_architecture.md`. **Not needed for conversion;** needed by inference session. Conversion happens on safetensors weights and does not invoke these kernels.
- **Tokenizer pipeline:** Existing hf2q tokenizer embed path is assumed sufficient for 248K vocab. **If it isn't, follow-up ADR; not in this scope.**

---

## Glossary

- **A3B / A4B.** MoE shorthand: "active 3B / 4B parameters per token." Total parameters (e.g. 35B for A3B) is much larger.
- **DWQ.** Dynamic Weight Quantization. hf2q's activation-calibrated mixed-precision quant: base bits for most tensors, promoted to sensitive bits for layers whose activations show high quantization error. Default Gemma preset: 4-base, 6-sensitive.
- **Gated DeltaNet.** A linear-attention variant using a selective state-space model (SSM) with a learned gating mechanism. Qwen3.5's replacement for Mamba-style state updates.
- **GGUF.** The file format used by the ggml ecosystem (llama.cpp, ollama). Contains weights + metadata + tokenizer in a single file.
- **LLM_ARCH_QWEN35MOE.** The llama.cpp architecture constant for Qwen3.5-MoE. Arch string: `qwen35moe`.
- **MoE.** Mixture of Experts. Each FFN layer is replaced by N experts and a router; each token routes through top-K experts.
- **MROPE.** Multi-axis RoPE. Partitions the rotary dimensions into multiple sections with different theta bases, originally for video/image positional encoding; used in Qwen3.5-MoE for long-context efficiency.
- **MTP.** Multi-Token Prediction. An extra small transformer block after the main stack that predicts the *next* next token, enabling speculative decoding. Originated in DeepSeek-V3.
- **Shared experts.** In some MoE architectures (including Qwen3.5-MoE), a small set of experts are activated for *every* token regardless of the router. Distinct tensors from the routed experts.
- **V-head grouped vs tiled.** Two memory layouts for multi-head tensors when `num_v_heads > num_k_heads`. Grouped: V heads adjacent per K head (`G0_v0, G0_v1, G1_v0, G1_v1, ...`). Tiled: all K heads' first V, then all K heads' second V (`G0_v0, G1_v0, G0_v1, G1_v1, ...`). HF uses grouped; ggml binary ops expect tiled.

---

## Appendix A: Target convert commands (once all phases land)

```bash
# 4-bit DWQ
hf2q convert \
  --repo jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated \
  --format gguf \
  --quant dwq-mixed-4-6 \
  --calibration-samples 1024 \
  --output models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46.gguf

# 8-bit DWQ
hf2q convert \
  --repo jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated \
  --format gguf \
  --quant dwq-mixed-4-8 \
  --calibration-samples 1024 \
  --output models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48.gguf
```

---

## Appendix B: Canonical gotcha cross-reference

Every gotcha from `project_qwen36_architecture.md` is addressed in a specific decision in this ADR. The table below is the engineer's checklist for "did I handle everything?"

| Gotcha from memory | ADR Decision |
|---|---|
| #1 V-head grouped→tiled | Decision 5 |
| #2 A_log negation | Decision 6 |
| #3 dt_bias rename | Decision 6 |
| #4 Conv1d squeeze | Decision 6 |
| #5 RMS norm +1 | Decision 6 (audit) |
| #6 in_proj_qkvz reordering | Decision 6 |
| #7 Decay-mask log-space clamp | **Inference-only** — not a convert-time transform; belongs to inference session's ADR |
| #8 MROPE sections | Decision 7 (metadata emission) |
| #9 `full_attention_interval` vs `layer_types` | Decision 2 (parser dual-support) |
| #10 Expert weights scale | **Inference-only** — runtime scaling, not persisted in tensor data |

Gotchas #7 and #10 are runtime concerns owned by the inference session; conversion does not transform them.
