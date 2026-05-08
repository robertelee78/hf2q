# hf2q Codebase Map for DWQ Port Completion (CFA worker 3)

Path roots used below — all absolute paths under `/opt/hf2q-adr020-iter10/`.

## 1. What's shipped and working

All paths under `/opt/hf2q-adr020-iter10/src/calibrate/`. Status legend: **prod** = wired into a CLI subcommand and runs in conversion path; **iter-19c** = real-BF16 single-Linear KL test landed; **synth-tested** = unit/synthetic only; **stub** = referenced but not implemented.

| Module | LOC | Public API | Status |
|---|---:|---|---|
| `mod.rs` | 62 | re-exports | prod (module index) |
| `apex.rs` | 680 | `ApexConfig`, `compute_importance_matrix`, `select_kquant_types`, `run_apex_quantization` (line 311) | prod (mixed-precision K-quant) |
| `imatrix.rs` | 1762 | `ImatrixCollector`, `Stats`, `ImatrixError` | prod |
| `imatrix_calibrator.rs` | 806 | `ImatrixCalibrator` impl `Calibrator` | prod |
| `imatrix_xvalidate.rs` | 576 | imatrix cross-validation | prod |
| `sensitivity.rs` | 236 | `LayerSensitivity`, `compute_layer_sensitivity`, `allocate_bits_by_sensitivity` | prod (variance-magnitude proxy — to be replaced) |
| `sensitivity_comparison.rs` | 444 | `LayerAggregator`, `aggregate_per_layer_scores`, `spearman_rank_correlation`, `top_k_overlap` | synth-tested (iter-11g) |
| `dwq.rs` | 909 | `DwqArch`, `DwqConfig`, `DwqQuantizer`, `run_dwq_calibration`, `run_dwq_calibration_internal` | prod (LEGACY weight-space DWQ-46/48; misnamed — see ADR-020 §1.3) |
| `dwq_calibrator.rs` | 885 | `DwqCalibrator` impl `Calibrator` | prod |
| `dwq_activation.rs` | 616 | `run_dwq_activation_calibration`, `capture_activations_to_sensitive_ranges`, `run_dwq_with_sensitive_ranges` | prod |
| `calibrator.rs` | 719 | `Calibrator` trait, `NoneCalibrator`, `CalibrationCorpus`, `CalibrationData` | prod |
| `cache.rs` | 691 | sensitivity JSON cache | prod |
| `dynamic_quant.rs` | 568 | `estimate_threshold` + `compute_bits_per_weight` (iter-7, pure CPU) | prod |
| `dynamic_quant_gpu.rs` | 514 | `kl_div_loss_per_row`, `estimate_sensitivities`, `SyntheticTwoLinearModel`, `QuantizableInput` | synth-tested (iter-9) |
| `qdq_gpu.rs` | 343 | `qdq_q4_0_gpu`, `qdq_q8_0_gpu`, `qdq_q{4,8}_0_to_tensor` | synth-tested (iter-10a) |
| `autograd.rs` | 1360 | CPU oracle (test-only) — Tape + 7 ops + finite-diff falsifier | test-only oracle |
| `autograd_gpu.rs` | 464 | `matmul_forward_f32`, `matmul_backward_f32` standalone primitives | synth-tested (iter-8b) |
| `autograd_gpu_tape.rs` | 3931 | `GpuTape`, `GpuTensor`, all op factories + `backward` | synth-tested (8a–13b) |
| `adam.rs` | 416 | `AdamConfig`, `AdamOptimizer` | synth-tested (iter-13a) |
| `dwq_loop.rs` | 1191 | `init_affine_params_gpu` (line 49), `box_muller_gaussian` (line 139), `buffer_from_f32` (line 175); rest is test-only loops | synth-tested + iter-13d/13e real-tensor |
| `dwq_targets.rs` | 696 | `TeacherLogitsProvider` trait (line 58), `ComputeTargetsConfig`, `CalibrationSplit`, `compute_dwq_targets` (line 103), `load_dwq_target` (line 359) | iter-14 synth-tested. **No production impl of the trait exists.** |
| `mlx_safetensors_loader.rs` | 1220 | `MlxQuantConfig`, `MlxAffineLinear::from_safetensors`/`to_safetensors_bytes`, `pack_u32_codes`, `unpack_u32_packed`, `read_floats_to_f32`, `write_floats_from_f32`, `discover_shards`, `MlxAffineLinearBytes` | synth-tested + load-bearing iter-19c reader use |
| `dwq_e2e.rs` | 881 | three #[test]s: `e2e_synthetic_train_save_load_infer_cycle_closes` (line 170), F32 sibling (line 790), `iter_19c_single_linear_kl_parity_vs_bf16` (line 481, `#[ignore]`) | iter-19c **PASS**: KL=2.73e-3 on layer 0 `linear_attn.out_proj` BF16 |
| `qwen35_attention_block.rs` | 1045 | `AttentionBlockConfig/Weights/Leaves`, `forward`, `multi_head_sdpa` (line 330), `estimate_attention_block_sensitivities` (line 271), `estimate_attention_block_sensitivities_streaming` (line 437) | synth-tested (iter-10c, iter-11a) |
| `qwen35_ffn.rs` | 316 | `FfnConfig`, `FfnWeights`, `FfnLeaves`, `forward` | synth-tested (iter-11b) |
| `qwen35_layer.rs` | 584 | `LayerConfig`, `LayerWeights`, `LayerLeaves`, `forward` (composes RMSNorm → Q/K/V → SDPA → O → residual → RMSNorm → SwiGLU → residual) | synth-tested (iter-11c) |
| `qwen35_model.rs` | 542 | `ModelConfig`, `ModelWeights`, `ModelLeaves`, `forward` (embedding → N×layer → final RMSNorm → lm_head) | synth-tested (iter-11d) |
| `qwen35_gguf_adapter.rs` | 331 | `weights_from_gguf_tensors` | synth-tested (iter-11e). MHA-only — does NOT support GQA, MoE, or linear_attn |
| `calibration_batcher.rs` | 350 | `CalibrationBatcher`, `whitespace_hash_tokenize` | synth-tested (iter-11f). Stub tokenizer — real BPE plug-in not landed |

## 2. Autograd backbone

`GpuTape` (Rc<RefCell<...>>) at `autograd_gpu_tape.rs:228`. Holds an `MlxDevice` + `KernelRegistry` + `Vec<GpuNode>`. `tape.reset()` (line 275) clears nodes between batches keeping the device warm — proven load-bearing for streaming after the iter-11b "Metal residency-set contention" flake.

`OpKind` enum (lines 67–208) — closed list of 17 variants:

| Variant | Forward fn | Backward kernels (mlx-native) | Tested by |
|---|---|---|---|
| `Leaf` | `GpuTensor::from_vec` / `from_buffer` | — | basic |
| `Matmul` | `matmul` (line 399) | `dense_matmul_f32_f32_tensor` + 2× `transpose_2d`. **k ≥ 32 floor** (line 411) | iter-8b.1 (two-matmul chain parity) |
| `ElementwiseAdd` / `Sub` / `Mul` | `add` / `sub` / `mul` / `square` (lines 1978, 1993, 2087, 2103) | `elementwise_add` / `elementwise_mul` / `scalar_mul_f32(-1)` | iter-8c (incl. `square_via_mul_self`) |
| `Softmax` | `softmax` (line 1451) | `dispatch_softmax_backward` | iter-8d (CPU oracle parity 1e-4) |
| `Log` | `log` (line 1416) | `dispatch_log_backward_f32` | iter-8e |
| `RowSum` | `row_sum` (line 1357) | `dispatch_row_sum_backward_f32` | iter-8f (KL-div composition test) |
| `Embedding` | `embedding` (line 1520) | `dispatch_embedding_scatter_add_f32` | iter-11d |
| `SiLU` | `silu` (line 1609) | `dispatch_silu_backward_f32` | iter-11b (FD falsifier 5e-3) |
| `Slice2dCols` / `Concat2Cols` | `slice_cols` (line 1655), `concat_cols` (line 1729) | `dispatch_slice_2d_cols_f32` / `dispatch_copy_2d_cols_into_f32` | iter-11a |
| `Transpose2d` | `transpose` (line 1829) | self-inverse `transpose_2d` | iter-11a |
| `RmsNorm` | `rms_norm` (line 1887) | 3-kernel chain: `rms_norm_compute_rms_inv` + `rms_norm_backward_dx` + `rms_norm_backward_dw` | iter-10b (FD falsifier 5e-3) |
| `View` | `view` (line 2188) | identity (zero-copy reshape) | iter-13c |
| `ScalarMul` | `scalar_mul` (line 2226) | `scalar_mul_f32(s)` | iter-13c |
| `QdqAffine` | `qdq_affine` (line 2289) | `dispatch_qdq_affine_backward_{scales,biases}_f32`. **q_int frozen** | iter-13b |

**Critical kernel-side floor at `autograd_gpu_tape.rs:649-663`:** every backward matmul requires `m, n, k ≥ 32`. Production Qwen 3.6 35B-A3B-Abliterix has dims (per iter-19c) hidden=2048, head_dim varies — all comfortably ≥ 32, but small-batch decode (m=1) cannot run through this tape. Forward/backward are training-batch-only.

**What's MISSING from the GpuTape for full Qwen 3.5/3.6 (iter-11h gap):**

1. **Rotary embeddings (RoPE)**. Zero references to RoPE in calibrate/. The Q/K projections in `qwen35_layer.rs:248-249` go straight into `multi_head_sdpa` with no positional encoding. Production decode-side RoPE lives in `mlx-native` ops but has no GpuTape op-kind binding.
2. **Grouped-Query Attention (GQA)**. `qwen35_attention_block.rs:330` `multi_head_sdpa(q, k, v, n_heads, head_dim)` assumes `q.shape == k.shape == v.shape`. Qwen 3.5 35B-A3B has `n_heads=16, n_kv_heads=2` (8:1 ratio); K/V need broadcast across the head ratio. `qwen35_gguf_adapter.rs:15-20` doc-comments concede this gap explicitly: *"Scope (iter-11e): standard multi-head attention only. Grouped-query attention (GQA, where n_kv_heads < n_heads) is deferred."*
3. **MoE FFN routing**. `qwen35_ffn.rs` is a vanilla SwiGLU dense FFN. No router, no expert dispatch, no top-K gather. Qwen 3.5 35B-A3B has 128 experts, top-8 routing — completely absent.
4. **Linear-attention / DeltaNet hybrid layers**. The cached BF16 model in iter-19c is `Qwen3.6-35B-A3B-Abliterix-EGA-abliterated` whose tensor-name structure `model.language_model.layers.0.linear_attn.out_proj.weight` (dwq_e2e.rs:516) reveals 3:1 hybrid linear+full attention. The calibrate/ stack has zero linear_attn op support.
5. **Causal mask** in `multi_head_sdpa`. The forward at `qwen35_attention_block.rs:330` does softmax over `Q @ K^T` with no upper-triangular mask. For training/distillation that's fine on a single token-position prediction, but for full-sequence forward the lack of causal masking is a behavioral mismatch with the production model.
6. **Logit scaling, lm_head tie**, and `output_norm` mid-block hooks present in some Qwen 3.5 variants — none are wired.

## 3. DWQ training loop

`dwq_loop.rs` is **NOT a production loop** — only `init_affine_params_gpu`, `box_muller_gaussian`, and `buffer_from_f32` are non-test public functions. The loop body lives inside the test functions starting at line 215 (`dwq_loop_synthetic_recovers_perturbed_affine_params`), 407 (`dwq_loop_synthetic_2linear_kl_div_converges_under_adam`), 696 (`dwq_loop_real_gguf_attn_qkv_converges_under_adam`).

The pattern (lines 268–325, 522–598, 817–843) is consistent and ready to lift:

```rust
// per-step shape (paraphrased from dwq_loop.rs:268-325)
let s_leaf = GpuTensor::from_vec(tape, &adam.read_param("s")?, vec![s.len()])?;
let b_leaf = GpuTensor::from_vec(tape, &adam.read_param("b")?, vec![b.len()])?;
let qdq = qdq_affine(&s_leaf, &b_leaf, &q_int, group_size)?;
// ...build KL forward chain through view → matmul → ... → kl_div_loss_per_row...
let grads = backward(&kl, dy_buf)?;
let g_s = grads[s_leaf.node_idx()].as_ref().unwrap().clone();
adam.step(&{ let mut m=BTreeMap::new(); m.insert("s".to_string(), g_s); m })?;
tape.reset();  // critical for memory bound
```

Hardcoded vs configurable:
- **Hardcoded**: q_int (frozen, no learnable codes), `lr=1e-3` (test default; `dwq.py` uses 1e-6 — see ADR-020 §3 row 1), batch via `m` parameter, `T=2.0` temperature.
- **Configurable**: `n_steps`, `group_size`, `bits`, perturbation factor, `AdamConfig`. Per-tensor scales/biases stored in `AdamOptimizer` indexed by string name.

Per-Linear-grad flow: `grads[node_idx]` is `Option<MlxBuffer>`; nodes that didn't participate are `None`. After `backward`, the caller pulls the grad for each scales/biases leaf by `node_idx()` and feeds them into `Adam.step` keyed by string name.

**Sample-batch handling**: completely absent in the loop. Each test pre-computes ALL teacher logits as a host FP64 oracle before the loop (e.g. dwq_loop.rs:486-513). There is NO per-batch streaming integrated with `compute_dwq_targets` or `TeacherLogitsProvider`. The mlx-lm `del grads + mx.eval(grad_accum)` rhythm at `dwq.py:178-179` has no analog in the test loop.

## 4. Model-side machinery

`qwen35_attention_block.rs`, `qwen35_ffn.rs`, `qwen35_layer.rs`, `qwen35_model.rs` are **CALIBRATION-ONLY** — they live entirely on the GpuTape, share zero code with the production inference path at `src/serve/forward_mlx.rs`, `src/serve/forward_prefill.rs`, etc. Confirmed by:

- `qwen35_layer.rs:236` `pub fn forward(cfg, x: &GpuTensor, ...) -> Result<GpuTensor>` — only callable with a `GpuTensor`.
- Zero hits for `MlxModelWeights` / `forward_decode` / `forward_prefill` in calibrate/.
- The production engine at `serve/forward_mlx.rs` (5665 LOC) defines its own forward path: `MlxModelWeights::forward_decode` (line 1595), `forward_prefill` (referenced at 3768), `logits_view` (4599) — none reachable from a GpuTape consumer.

Production prefill IS exposed (line 3768 mentions `forward_prefill`) but its return value is consumed internally; the public surface for "give me a `[B, S, vocab]` logits tensor" does not exist. Closest is `logits_view() -> &[f32]` (line 4599) which returns the **last token only** post-decode-step.

## 5. Calibration + targets

`dwq_targets.rs` ports `mlx_lm/quant/dwq.py:29-66` directly. Public surface:

- `TeacherLogitsProvider` trait (line 58) — `forward_logits(tokens, batch, seq, vocab) -> Vec<f32>` of `[B, S, V]` flat fp32.
- `compute_dwq_targets` (line 103) — drives a teacher across `CalibrationSplit`s, top-K reduction host-side via `BinaryHeap`, writes `<save_dir>/<split>/<i:010d>.safetensors` with `{logits: [B, S-1, K] f32, indices: [B, S-1, K] u32}`.
- `load_dwq_target` (line 359) — consumer-side reader.

**Format parity with mlx-lm**: confirmed byte-compatible per ADR-020 row 14: *"Output format byte-compatible with mlx-lm: `{"logits": [B, S-1, K] f32, "indices": [B, S-1, K] u32}` per safetensors file"*. Top-K via `BinaryHeap` instead of mlx-native's `top_k_f32` (which caps at K=128 vs mlx-lm's K=1024).

**THE GAP**: only `SyntheticTeacher` (test fixture, line 437) implements `TeacherLogitsProvider`. There is NO `GgufTeacherProvider`, NO real teacher hooked up to the production engine. This is iter-14b.

**Calibration corpus reachability**: not yet. `calibration_batcher.rs:104` `whitespace_hash_tokenize` is a deterministic stub for tests; real BPE plug-in is iter-11g/11h scope. mlx-lm's calibration corpus (`mlx_lm.quant.utils.load_data`) is hooked up only via the vendored `scripts/dwq_kl_parity/kld.py` shim (not in calibrate/).

## 6. mlx-format I/O

`mlx_safetensors_loader.rs` is the **single most production-ready** Track 2 module. Read path (lines 143-238):

- `MlxAffineLinear::from_safetensors(st, path, bits, group_size)` — pulls `<path>.{weight, scales, biases}` triplet, unpacks U32-packed weight bits → `Vec<u8>` (one byte per code), casts BF16/F16 scales+biases to F32.
- `MlxQuantConfig::from_config_json` parses `quantization.{bits, group_size, mode}` + per-layer overrides.
- `discover_shards` handles both single-file and sharded.

Write path (lines 388-463):

- `MlxAffineLinear::to_safetensors_bytes(float_dtype) -> MlxAffineLinearBytes` — round-trips through the same canonical `pack_u32_codes` + `write_floats_from_f32` (BF16/F16/F32 LE).
- Verified against `mlx/ops.cpp:4762-4798` per ADR-020 row 16/16b.

**Format='mlx' metadata**: I see no automatic top-level `quantization` block emission in `MlxAffineLinearBytes::to_safetensors_views` — that block has to be written separately into a `config.json`. The `from_config_json` reader knows the schema but the writer doesn't yet emit one. **MINOR GAP** for iter-17b shipping.

iter-19c uses this loader successfully on layer-0 `linear_attn.out_proj.weight` from the cached BF16 reference — the `discover_shards` + `read_floats_to_f32` (BF16→F32) path is **REAL-MODEL-TESTED**.

## 7. Mixed-precision K-quant (already shipped)

This is hf2q's already-shipped product line and is NOT what iter-19d half-2 is about. Quick audit so the team knows it's separate:

- `imatrix.rs` (1762 LOC) — port of llama.cpp's `tools/imatrix/imatrix.cpp`. `Stats`/`ImatrixCollector` accumulate `x[col]² · 1.0` per column during a calibration forward pass.
- `imatrix_calibrator.rs` — `Calibrator` trait impl that consumes Stats and produces a per-column importance matrix the K-quant codec consumes.
- `apex.rs:311` `run_apex_quantization` — Phase 2 entrypoint that combines imatrix scores with `select_kquant_types` to allocate per-tensor `Q4_K_M`/`Q5_K_M`/`Q6_K`/`Q8_0` choices.
- `sensitivity.rs` — variance-magnitude proxy ranker. ADR-020 §1.3 marks this as architecturally inferior to mlx-lm's gradient-Taylor sensitivity (which iter-9 `estimate_sensitivities` aims to replace it with).

**ImatrixAdaptive CLI routing** at `cli.rs:1002-1007`: `QuantMethod::ImatrixAdaptive` → `main.rs:1685` selects `ImatrixAdaptive` → calibrator factory at `main.rs:347` returns `ImatrixCalibrator::new(...)` for arches that support forward-pass capture.

The legacy DWQ-46/48 path: `cli.rs:1010-1019` declares `Dwq46/48/68/28`; `main.rs:1827-1844` maps to `quantize::dwq_k_quantizer::DwqKVariant::P{46,48,68,28}` and feeds through `DwqKQuantizer::new(...)` (line 1871). This is the path ADR-020 §1.3 plans to **REPLACE**, but it is currently the only DWQ path users invoke.

**Wires from new (iter-9+) code into this CLI: ZERO.** None of `estimate_sensitivities`, `qwen35_layer::forward`, `compute_dwq_targets`, `MlxAffineLinear::to_safetensors_bytes` is reachable from `hf2q convert`. They're all `#[cfg(test)]` reachable only.

## 8. iter-11h + iter-14b: explicit gap analysis

### iter-11h — Real-model multi-layer Qwen 3.5/3.6MoE forward on GpuTape

**ADR mandate** (row 11h, line 209): *"Real-Qwen3-0.6B-base spot-check (gated on model availability) — download Qwen3-0.6B-base GGUF, load via iter-11e adapter, run iter-11g comparison harness against hf2q's variance-magnitude ranker, verify lm_head + output_norm rank in top half of both rankings."*

**ADR row 19b half-2** (line 232) escalates this dramatically: *"Gated on iter-14b ... + iter-11h (full multi-layer Qwen3.5MoE forward on GpuTape including hybrid linear+full attention layer types)."*

The 11h-as-written (Qwen3-0.6B-base spot check) is a tiny scope that needs only model download + the existing `weights_from_gguf_tensors` adapter. The 19b half-2 reformulation needs a much bigger lift.

**What's missing for the larger 19b reformulation**:

1. **GQA broadcast in `multi_head_sdpa`** — currently asserts `q.shape == k.shape == v.shape`. Need a `n_kv_heads`-aware path where K/V are repeated `n_heads / n_kv_heads` times across the head dim before the score matmul, with corresponding backward through the broadcast (sum over the head ratio). Estimated **~150 LOC** in `qwen35_attention_block.rs` + 1 new mlx-native broadcast kernel or composition.

2. **RoPE op kind** — new `OpKind::Rope { rows, n_heads, head_dim, base, position_ids }` with forward via existing mlx-native `rope` ops + backward (RoPE is unitary so backward is just the inverse rotation). **~250 LOC** including a Metal kernel (or composed via cos/sin tables as leaves and existing elementwise ops, ~100 LOC less).

3. **Causal mask** — option on `multi_head_sdpa` to add `-inf` to upper-triangular before softmax. Forward only (mask is constant, no grad). **~50 LOC**.

4. **MoE FFN** — full-A3B router + top-8 expert dispatch. Mlx-native already has `quantized_matmul_simd_bf16_expert` for inference (see ADR-020 §6 line 162). For training/distillation, a router-softmax + scatter/gather + per-expert matmul is required. The existing `qwen35_ffn.rs::forward` is a single-expert SwiGLU; needs to fan out. Estimated **~500 LOC** + 1-2 new mlx-native kernels for the expert gather/scatter (or composition via existing slice/concat at NWG=128 — slow but functional). Top-K argmax also needs an op kind.

5. **Linear-attention / DeltaNet** — the abliterated cached model has `linear_attn` in tensor names. This is Mamba-ish state-space. Implementing it differentiable is multi-week work. **2000+ LOC** + multiple Metal kernels. Probably NOT feasible in iter-19d scope; recommend descoping iter-19b to a non-hybrid Qwen3 (e.g. 0.6B-base or 1.5B) until linear_attn is needed.

6. **GGUF adapter expansion** — `qwen35_gguf_adapter.rs` doc explicitly: GQA + linear_attn + MoE all defer. **~200 LOC** to handle MoE expert tensor names (`blk.{i}.ffn_gate_exps.weight` family).

7. **Real BPE tokenizer integration** — `calibration_batcher.rs::whitespace_hash_tokenize` is a stub. mlx-lm's calibration data is BPE-tokenized. Need to wire `tokenizers` crate or mlx-native's BPE. **~100 LOC**.

**Tests that would prove iter-11h** (in priority order):
- Existing iter-11d `model_forward_shape_and_finite` extended to a real-GGUF Qwen3-0.6B forward, compare logits to llama.cpp on the same prompt within 1e-3 rel.
- Sensitivity ranking test from iter-11g executed against the real model — verify `lm_head` + `output_norm` rank in top half (per ADR mandate).
- KL parity round: feed identical first-1k-token batch into hf2q's GpuTape forward AND into the production `MlxModelWeights::forward_prefill`; assert per-row KL < 1e-4.

### iter-14b — `GgufTeacherProvider`

**ADR mandate** (row 14b, line 218): *"`GgufTeacherProvider` — `TeacherLogitsProvider` impl backed by hf2q's existing GGUF model loader + GPU forward path. Spot-check on one calibration batch (2K tokens) against a peer (e.g. llama.cpp same model) for top-K parity. Defensive: cap teacher RSS via `model.drop()` before returning to dwq_quantize."*

**Input it needs**: the production engine at `src/serve/forward_mlx.rs` exposes `forward_decode` / `forward_prefill` but the public output is restricted to:

```rust
// forward_mlx.rs:4599
pub fn logits_view(&self) -> Result<&[f32]> { ... }  // last-token only, [vocab]
```

The dossier says **"exposes batched logits from `forward_prefill_batched`"** (memory note implies this exists at `forward_prefill_batched.rs`). Indeed, `MEMORY.md` references `forward_prefill_batched.rs:2131` for ADR-015 work. Confirmed `forward_mlx.rs:806` mentions `forward_prefill_batched.rs` as a sibling. So the prefill-batched path emits per-position logits internally — but there's no `pub fn forward_logits_batched(&mut self, tokens: &[u32], B, S, V) -> Vec<f32>` shaped to fit `TeacherLogitsProvider::forward_logits`.

**What's missing**:
- A new public method on `MlxModelWeights` (or a wrapper struct in `serve::api::engine`) that accepts `(tokens: &[u32], batch_size: usize, seq_len: usize)` and returns the full `[B, S, V]` flat fp32. Estimated **~150 LOC** (mostly buffer copy + reshape + the existing batched prefill plumbing).
- A new `src/calibrate/gguf_teacher_provider.rs` (~200 LOC) that:
  1. Takes a `MlxModelWeights` (loaded via the existing GGUF loader with appropriate quant level — for half-2 parity, BF16 model loaded directly).
  2. Implements `TeacherLogitsProvider::forward_logits` by calling the new public method.
  3. Implements `Drop` to release model GPU memory before student loads (mirrors `dwq.py:386-387`).
- An integration test that compares the GgufTeacherProvider's logits against `llama-completion` output on the same fixture for top-K parity.
- **CLI plumbing** to invoke `compute_dwq_targets` with this provider — does not exist anywhere today. Estimated **~300 LOC** for a new `hf2q calibrate compute-targets --teacher /path/to.gguf --corpus /path --save-dir /path` subcommand.

**Total iter-14b LOC estimate**: ~650 LOC (excluding any model-load refactor needed if BF16 isn't currently first-class through the GGUF loader; the cached BF16 reference is in HuggingFace safetensors format, not GGUF, so iter-14b may need a `SafetensorsTeacherProvider` variant or model-format abstraction — **add ~200 LOC for that**).

**Tests that would prove iter-14b**:
- `gguf_teacher_provider_returns_finite_logits_on_real_qwen` — load Qwen3.6 27B Q4_0, run 1 batch of 2K tokens, assert `[B, S, V]` shape + all finite.
- `gguf_teacher_provider_top_k_matches_llama_cpp` — same fixture, top-1024 indices match llama.cpp's logits within tolerance.
- `compute_dwq_targets_with_real_teacher_round_trips` — drives `compute_dwq_targets` over 4 batches, then `load_dwq_target` reads them back byte-identical.

## 9. CLI surface for full-model DWQ

There is currently NO CLI invocation that exercises any of iter-7+ work. The hypothetical native invocation per ADR §8.3 row 6:

```bash
hf2q convert --quant dwq-native-4 --input /path/to/bf16.safetensors --output /path/out.safetensors
```

does not exist. Today `cli.rs:1002-1019` recognizes only the LEGACY DWQ-46/48/68/28 (`--quant dwq-4-6` etc.) and `imatrix-adaptive`, all routed to `DwqKQuantizer` or `ImatrixCalibrator`.

**To wire end-to-end DWQ-native shipping**:

1. New `QuantMethod::DwqNative4 { group_size: u32 }` variant in `cli.rs` (~30 LOC).
2. New arm in `main.rs::resolve_convert_config` (~50 LOC).
3. New top-level orchestrator `src/calibrate/dwq_native.rs` (~600 LOC) that:
   - Reads source weights (BF16 safetensors or GGUF Q4_0).
   - Iterates layers (or all linears if model fits).
   - For each Linear: `init_affine_params_gpu` → DWQ training loop using `compute_dwq_targets` outputs → `MlxAffineLinear::to_safetensors_bytes`.
   - Aggregates into a sharded mlx-format safetensors output.
   - Writes a `quantization`-stanza `config.json`.
4. Output sink expansion — `src/quantize/output/` currently only emits GGUF; needs a `MlxSafetensors` output format. **~400 LOC**.

Note: the iter-13d `dwq_loop_real_gguf_attn_qkv_converges_under_adam` test has the per-Linear loop body (`HF2Q_TEST_GGUF` env-driven) that could serve as the lift template.

## 10. Test surface

What exists that exercises full-model GpuTape forward: **only synthetic-fixture tests** (`qwen35_model.rs:288` `model_forward_shape_and_finite`, n_layers=2, vocab=64, hidden=32). No real-model tests. iter-19c proves `qmm_affine_t_f32` + `MlxAffineLinear` + `qdq_affine` on a real BF16 weight, but it's a **single Linear in isolation** — no multi-layer composition, no embedding lookup against real vocab, no production tokens.

What's needed to add for iter-11h sign-off:
1. `qwen35_model_real_gguf_forward_parity` (`#[ignore]`) — load Qwen3-0.6B-base GGUF via `weights_from_gguf_tensors`, forward through `qwen35_model::forward`, compare top-1 token vs llama.cpp on the same prompt. Per-token agreement ≥ 95%.
2. `qwen35_model_real_gguf_sensitivity_ranking_top_half` (`#[ignore]`) — runs the iter-11g harness against the real model, asserts lm_head + output_norm rank in top half.
3. `gguf_teacher_provider_basic` + `compute_dwq_targets_real_teacher_round_trip` (gated `#[ignore]`).
4. `dwq_native_full_model_e2e_qwen3_0_6b` (gated `#[ignore]`) — train all layers via DWQ over a 100-batch corpus, save mlx-format, reload, run `qmm_affine_t_f32` inference, compare logits to teacher.

## 11. Crate-level structure / boundaries

Per `Cargo.toml:160-185` and `lib.rs`, hf2q has BOTH a `[[bin]]` named `hf2q` (the CLI) AND a narrow `[lib]` target. The `[lib]` exposes ONLY the `serve::kv_persist` block-store / writer / recovery / format / index / metrics / lcp_registry submodules — added in ADR-017 §A.2 specifically so `tests/kv_persist_writer_kill_minus_9.rs` could fork-test the SIGKILL durability invariant.

```rust
// lib.rs:39-71 (paraphrased)
pub mod serve {
    pub mod kv_persist {
        #[path = "../../serve/kv_persist/block_store.rs"] pub mod block_store;
        // ... format, index, metrics, recovery, writer, lcp_registry only ...
    }
}
```

Implications:
- The `calibrate::*` modules are **bin-private** — no integration test under `tests/` can import `crate::calibrate::dwq_targets::compute_dwq_targets`. Tests live as `#[cfg(test)] mod tests` blocks inside the bin source.
- Adding iter-11h or iter-14b external tests would require **either** expanding `lib.rs` to re-export `calibrate::dwq_*` (high blast radius — pulls in `mlx-native`, `safetensors`, the autograd tape, and transitively most of the GPU stack), **or** adding a new `[[bin]]` named `dwq_calibrate` that the test driver invokes via subprocess (matches how the cached `extract_dwq_sensitivity` and `dump_gguf_types` bins work today, per Cargo.toml:172-185).
- Per `MEMORY.md` "hf2q has no [lib] target — unit tests structurally expensive" (2026-05-01): the situation has improved post-ADR-017 but only for kv_persist. For calibrate work, the structural friction stands. The `#[ignore]`-gated `#[cfg(test)] mod tests` pattern (used by iter-13d, iter-13e, iter-19c) is the proven low-cost path.
- Memory-pressure-wise: every calibrate test currently spins up a fresh `MlxDevice::new()`. This was the source of the iter-11b "Metal residency-set contention" flake; the fix was tape-reset shared-device. For multi-test runs of `#[ignore]` real-model tests, a shared-device test fixture would help.

---

## Summary

The 3-5 highest-impact missing pieces, ordered by cost-benefit for unblocking iter-19b half-2:

1. **`GgufTeacherProvider` + a public batched-logits API on `MlxModelWeights`** (iter-14b core, ~650 LOC + ~150 LOC engine surface). Without this there is no way to feed real Qwen logits into `compute_dwq_targets` and the entire half-2 measurement is impossible. **Prerequisite for everything below.**

2. **GQA broadcast + RoPE op kinds on the GpuTape** (iter-11h fundamentals, ~400 LOC + ~1-2 mlx-native kernels). Without these the multi-layer forward at production dimensions cannot run end-to-end. RoPE alone is decisive — without it, attention scores are wrong and KL distillation can't converge to a meaningful number.

3. **Top-level orchestrator `src/calibrate/dwq_native.rs` + CLI subcommand `--quant dwq-native-4`** (~600 LOC + ~80 LOC CLI). Even if iter-14b lands, today there is no entrypoint that ties `compute_dwq_targets` → per-Linear DWQ loop → `to_safetensors_bytes` → sharded mlx-output into a single `hf2q convert` invocation. Without this CLI, iter-19b half-2 has to be driven from inside test functions.

4. **MoE FFN op kinds** (~500 LOC + 1-2 mlx-native kernels). Required for the actual cached BF16 reference (`Qwen3.6-35B-A3B-Abliterix-EGA-abliterated`). If iter-19b half-2 is descoped to dense Qwen3-1.5B-base, this can be deferred — but the memory `project_qwen35_dwq_pre_505b5b8_broken_2026_05_05.md` makes clear that the user's mission target IS the 35B-A3B MoE family. So MoE must land before the v0.1 ship.

5. **Real BPE tokenizer wiring in `calibration_batcher.rs`** (~100 LOC). Cheap but currently blocking the calibration corpus from actually being mlx-lm-comparable. Without this, the "same calibration corpus" half of iter-19b half-2's parity claim is structurally unachievable — `whitespace_hash_tokenize` produces different token IDs than mlx-lm's BPE tokenizer.

The iter-19c PASS (KL=2.73e-3 on a real BF16 single Linear) is real and load-bearing, but it tells us only that the **per-Linear primitive** works. The end-to-end half-2 measurement requires composing that primitive across 47 layers + hybrid attention + MoE routing + RoPE + GQA + a real teacher producing batched logits over a real BPE-tokenized corpus — and **none of that composition is wired today**. The iter-13/14/15/16/17/19c iterations have built every piece in isolation; iter-11h + iter-14b are the wiring iterations.
