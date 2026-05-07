# Researcher 7 — File Format Compatibility: MLX safetensors vs GGUF Q4_K

`/cfa cfa-20260506-mlxlm-research` — read-only.

## Bottom line

mlx-lm and GGUF Q4_K are **algorithmically the same family** (both per-group affine 4-bit) but **byte-incompatible**. The trained scales+biases produced by `mlx_lm.quant.dwq` cannot be `memcpy`'d into a Q4_K block. They can be **dequantized to FP and re-quantized**, which loses the trained-scale benefit, OR **fitted into Q4_K's compressed-scale slots** with measurable but bounded loss. Track 1 (`dynamic_quant`) maps cleanly to GGUF's mixed-tensor scheme; Track 2 (DWQ) does not.

---

## 1. MLX safetensors with quantized weights — exact byte layout

### 1.1 Tensors per `nn.QuantizedLinear` layer

From `/opt/homebrew/lib/python3.14/site-packages/mlx/nn/layers/quantized.py:223-248` and `/opt/mlx-lm/mlx_lm/utils.py:153-156`:

```python
self.weight, self.scales, *biases = mx.quantize(weight, group_size, bits, mode="affine")
self.biases = biases[0] if biases else None
```

A single quantized linear layer produces **three tensors** in the safetensors file:

| Tensor name | Shape | dtype | Notes |
|---|---|---|---|
| `model.layers.{i}.self_attn.q_proj.weight` | `[out_features, in_features // pack_factor]` | `uint32` | packed `32//bits` 4-bit weights per int32 |
| `model.layers.{i}.self_attn.q_proj.scales` | `[out_features, in_features // group_size]` | model dtype (FP16 / BF16) | one scale per group |
| `model.layers.{i}.self_attn.q_proj.biases` | `[out_features, in_features // group_size]` | model dtype (FP16 / BF16) | one bias per group |

For `bits=4`: `pack_factor = 32 // 4 = 8`. Defaults from `utils.py:800-808`:

```python
mode_defaults = {
    "affine": (64, 4),     # group_size=64, bits=4 — DEFAULT for DWQ
    "mxfp4":  (32, 4),
    "nvfp4":  (16, 4),
    "mxfp8":  (32, 8),
}
```

The three tensors live alongside any non-quantized tensors (norms, embeddings if not quantized) inside one of the standard sharded `model-{i:05d}-of-{n:05d}.safetensors` files. `mlx_lm.utils.save_model` (utils.py:714-771) writes them with `mx.save_safetensors(..., metadata={"format": "mlx"})` and emits the standard `model.safetensors.index.json` HF-style weight map.

### 1.2 `config.json["quantization"]` schema

From `utils.py:813-845`:

```json
{
  "quantization": {
    "group_size": 64,
    "bits": 4,
    "mode": "affine"
  },
  "quantization_config": { ... same dict mirrored ... }
}
```

Per-layer override (used by `dynamic_quant` and DWQ-fine-grained):

```json
{
  "quantization": {
    "group_size": 64,
    "bits": 4,
    "mode": "affine",
    "model.layers.0.self_attn.q_proj": { "bits": 5, "group_size": 64 },
    "model.layers.7.mlp.down_proj":   { "bits": 6, "group_size": 64 }
  }
}
```

`save_config` (utils.py:899-922) mirrors `quantization` to `quantization_config` for HF-tree compatibility.

### 1.3 Group-size semantics — per-group along which axis?

Quantization groups run **along the input (last/contiguous) axis** of the weight matrix. From `utils.py:826`: `if module.weight.shape[-1] % group_size != 0: return False`. With `weight.shape == [out_features, in_features]`, you get `n_groups = in_features // group_size` — one (scale, bias) pair per `(out_row, group_along_in)`.

### 1.4 Quantization formula — affine

Per `utils.py:144-147`:

```
MLX dequant: w_fp = q_int * scale + bias
```

`mx.quantize(weight, group_size, bits, mode="affine")` computes per-group `(min, max)` of the original weight, then:

```
scale = (max - min) / (2^bits - 1)
bias  = min                          # so q_int=0 reproduces min
q_int = round((w_fp - min) / scale)  ∈ [0, 2^bits - 1]
```

The packed `weight` tensor stores the **8 four-bit values per uint32**, little-endian-shifted (`utils.py:128-131`):

```python
shifts = mx.arange(pack_factor) * bits  # [0, 4, 8, 12, 16, 20, 24, 28]
weight = (repacked.astype(mx.uint32) << shifts).sum(axis=-1).astype(mx.uint32)
```

So `uint32` slot `k` packs the 8 weights for input columns `[8k, 8k+1, ..., 8k+7]` of one output row, with weight `j` stored in bits `[4j, 4j+4)`.

### 1.5 DWQ output vs initial `quantize_model` output — same file format?

**Same format. No marker.** From `dwq.py:97-99,389-411`:

```python
def unfreeze(_, m):
    if hasattr(m, "bits") and m.mode == "affine" and m.bits < 8:
        m.unfreeze(keys=["scales", "biases"], recurse=False)
...
save(args.mlx_path, args.model, q_model, tokenizer, config)
```

DWQ takes an already-quantized model, **unfreezes only `scales` and `biases`** (the packed `weight` integers stay frozen), runs KL-divergence distillation against a teacher's logits, then writes the model back through the standard `mlx_lm.utils.save` path. The output is structurally identical to a fresh `mlx_lm.convert` quantize — same three tensors per layer, same `config.json["quantization"]` block — only the FP scale/bias values differ from what naive min-max would have produced. There is no DWQ-specific marker in the safetensors or config; you cannot tell DWQ output from base-quantize output by inspecting bytes alone.

This is critical for hf2q's import path: there is no signal to distinguish "trained DWQ scales worth preserving" from "uniform-min-max scales easily replaced."

---

## 2. GGUF Q4_K block layout

From `/opt/llama.cpp/ggml/src/ggml-common.h:317-328` (also mirrored in `/opt/hf2q/src/quantize/k_quant.rs:36-69`):

```c
#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    union {
        struct { ggml_half d, dmin; };  // FP16 super-block scale + super-block min-scale
        ggml_half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE];       // 12 bytes packing 8×6-bit sub-block scales + 8×6-bit sub-block mins
    uint8_t qs[QK_K/2];                 // 128 bytes — 256 four-bit weights, two per byte
} block_q4_K;
// sizeof = 4 + 12 + 128 = 144 bytes — confirmed by static_assert at line 328
```

**Block geometry**:

- 1 super-block = **256 weights** (fixed, not configurable)
- 1 super-block = **8 sub-blocks of 32 weights each**
- Per super-block: 2 × FP16 (`d`, `dmin`) + 12 packed bytes + 128 packed weight bytes = **144 bytes**
- Effective bits-per-weight: `144 × 8 / 256 = 4.5 bpw`

**Quantization formula** (from `ggml-quants.c:1479-1484` dequantize):

```c
// For each pair of 32-weight sub-blocks j, j+1:
get_scale_min_k4(is + 0, x[i].scales, &sc, &m);     // sc, m ∈ [0, 63]
const float d1 = d * sc;  const float m1 = min * m;
get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
const float d2 = d * sc;  const float m2 = min * m;
for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;     // sub-block j
for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >>  4) - m2;     // sub-block j+1
```

So the per-sub-block dequant is:

```
w_fp = (d * sc) * q_int - (dmin * m)
       └── d_eff ──┘    └── m_eff ──┘
```

This is **affine** with the sign convention `w = scale * q - min` (note `-min`, not `+bias`). Compared to MLX:

```
MLX:   w = scale * q + bias
GGUF:  w = d_eff  * q - m_eff
       (so MLX bias === -GGUF m_eff, modulo scale geometry)
```

The signs differ but the family is the same; the conversion is `bias = -m_eff`.

### Per-tensor-dtype options in GGUF

From `/opt/llama.cpp/gguf-py/gguf/constants.py:4083-4097, 4268-4270`:

```python
class GGMLQuantizationType(IntEnum):
    F32     = 0      F16     = 1
    Q4_0    = 2      Q4_1    = 3
    Q5_0    = 6      Q5_1    = 7
    Q8_0    = 8      Q8_1    = 9
    Q2_K    = 10     Q3_K    = 11
    Q4_K    = 12     Q5_K    = 13
    Q6_K    = 14     Q8_K    = 15
    # ... IQ-family quantizations follow ...

# (block_size_in_weights, bytes_per_block):
Q4_K: (256, 2 + 2 + QK_K // 2 + 12)            #   = 144 bytes
Q5_K: (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12) #  = 176 bytes
Q6_K: (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16) # = 210 bytes
```

GGUF stores **one type code per tensor**, set in the per-tensor header. A "Q4_K_M" model is just a heterogeneous mix: most FFN/attn tensors use Q4_K, attn_v + FFN-down promote to Q6_K, output.weight stays F16, etc. There is no global "Q4_K_M" type code at the byte level — it is purely a build-time recipe.

---

## 3. The fundamental compatibility question — affine vs block

| Property | MLX `affine` mode | GGUF Q4_K |
|---|---|---|
| Group size | 64 (default) — configurable | **256 fixed** (with internal 8×32 sub-blocks) |
| Scale storage | 1 FP16/BF16 scalar **per group of 64** | 1 FP16 super-block scale `d` × 1 6-bit sub-scale per 32 |
| Min/bias storage | 1 FP16/BF16 scalar **per group of 64** | 1 FP16 super-block min `dmin` × 1 6-bit sub-min per 32 |
| Sign convention | `w = scale·q + bias` | `w = scale·q − min` |
| Bit-packing | 8 weights/uint32, axis-aligned along `in_features` | 2 weights/uint8, axis-aligned within super-block |
| Per-tensor metadata | 3 separate tensors (`weight`, `scales`, `biases`) | 1 contiguous blob of 144-byte super-blocks |
| Bit width per scale | FP16 (16 bits) | 6-bit sub-scale × FP16 super-scale (≈10 bits effective) |
| Effective bpw | ~4.5 bpw at gs=64 (4 + 32/64 = 4.5) | 4.5 bpw at QK_K=256 (4 + (4+4+12·8/256)) |

The two formats land on **the same effective bpw (4.5)** because both have the same metadata-overhead / weight ratio, but they spend that overhead very differently:

- MLX: **fewer, fatter scales** (FP16 per group of 64 = 1 full-precision affine pair per 64 weights).
- Q4_K: **more, thinner scales** (one 6-bit sub-scale per 32 weights, anchored on a single FP16 super-block scale).

---

## 4. The diff concretely — `q_proj.weight` shape `[4096, 4096]`

### MLX safetensors after `quantize_model(bits=4, group_size=64)`

```
weight  : shape [4096,  512]  dtype uint32          # 4096 × 512 × 4 B = 8.39 MB
scales  : shape [4096,   64]  dtype FP16            # 4096 ×  64 × 2 B = 524 KB
biases  : shape [4096,   64]  dtype FP16            # 4096 ×  64 × 2 B = 524 KB
TOTAL ≈ 9.43 MB
```

(`512 = 4096 / 8` packed columns; `64 = 4096 / 64` groups along input axis.)

### GGUF Q4_K equivalent

```
n_super_blocks = 4096 × (4096 / 256) = 4096 × 16 = 65,536 super-blocks
bytes          = 65,536 × 144 = 9,437,184 B ≈ 9.00 MiB ≈ 9.43 MB
TOTAL ≈ 9.43 MB  (within rounding — both at 4.5 bpw)
```

### The conclusion — same total size, different bytes

Both formats land at ≈9.43 MB, but **the byte sequences are entirely different**:

- MLX: three SoA tensors with FP16 scales contiguous and uint32-packed weights contiguous.
- Q4_K: one AoS tensor where each 144-byte super-block interleaves header + scales + weights.

There is no `memcpy` path. There is no rearrangement that makes them byte-identical. The closest mapping is:

| MLX field | Maps how to Q4_K |
|---|---|
| 4 MLX groups of 64 | = 1 Q4_K super-block of 256 |
| 4 MLX `scales[i]` (FP16) | → must be projected onto `(d, sc[0..3])` — `d` is one shared FP16 super-scale; each `sc[i]` is 6-bit |
| 4 MLX `biases[i]` (FP16) | → must be projected onto `(dmin, m[0..3])` (with sign flip) — same 6-bit compression |
| MLX `weight` uint32 (8 values × 4 bits) | → must be re-binned against the new `(d_eff, m_eff)` rather than the original MLX `(scale, bias)` |

The 2 sub-blocks/group ratio (Q4_K's 32-weight sub-block is half of MLX's 64-weight group) is the cleanest part. The hard part is the **6-bit compression of the per-sub-block scale and min** — Q4_K's `inv_scale = 63/max_scale` quantizes scales to 6 bits, throwing away precision the trained MLX FP16 scales relied on (`ggml-quants.c:1425-1431`):

```c
float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
for (int j = 0; j < QK_K/32; ++j) {
    uint8_t ls = nearest_int(inv_scale*scales[j]);  // ≤ 63
    uint8_t lm = nearest_int(inv_min  *mins[j]);    // ≤ 63
```

DWQ trains FP16 scales freely; Q4_K constrains all scales in a 256-block to lie on a 64-step grid scaled by one super-FP16. **This is the lossy step.**

---

## 5. Can hf2q consume mlx-lm DWQ output and emit GGUF without losing trained scales?

Three paths, ordered by quality preservation:

### Path A — dequantize → F16 GGUF (preserves DWQ quality, loses 4× compression)

```
load mlx safetensors
  → for each quantized layer: w_fp = q_int * scale_trained + bias_trained
  → emit GGUF with type=F16
```

Implementation: trivial. Quality: lossless w.r.t. the DWQ training. Storage: 4× larger (~36 MB instead of 9.43 MB for the example tensor). Useful for V0 validation but not a shipping format.

### Path B — dequantize → re-quantize via standard `quantize_row_q4_K` (drops DWQ benefit)

```
load mlx safetensors
  → dequantize via trained scales (yields the DWQ-optimised F16 weights)
  → run /opt/llama.cpp/ggml/src/ggml-quants.c::quantize_row_q4_K_ref OR
    /opt/hf2q/src/quantize/k_quant.rs::quantize_row_q4_k on the F16 row
  → emit GGUF with type=Q4_K
```

This is what `mlx_lm.convert` then `llama.cpp/quantize` would do today via a round-trip. Quality: **the DWQ-trained scales are entirely thrown away** because `make_qkx2_quants` re-derives its own `(scale, min)` per sub-block from the raw F16 input. The only DWQ contribution that survives is whatever signal got baked into the F16 reconstruction itself — which, since DWQ trains scales but holds the integer codebook fixed, is a reconstruction whose F16 values *would* re-quantize to a slightly different codebook on the way out. Empirically (from the ADR-014 P11-prereq analysis at `dwq_k_quantizer.rs:42-56`):

> Q4_K's codebook search (`make_qkx2_quants` / `quantize_row_q4_K`) already optimises (sub-block scale, sub-block min) jointly via a search over candidate inverse-scales and per-element re-binning; a post-hoc scalar scale tweak would degrade that joint optimum.

**Implication: re-quantization roughly resets the model to base-Q4_K quality.**

### Path C — fitted Q4_K emitter (preserves DWQ benefit, novel work, bounded loss)

```
load mlx safetensors (trained scales_64, biases_64)
  → for each 256-weight super-block aligned with 4 MLX groups of 64:
      derive a sub-block-32 view by either (a) splitting each 64-group in half
      (re-deriving sub-scale+sub-min from the F16 weights restricted to each half)
      or (b) accepting MLX's coarser granularity by using the same scale/bias
      for both 32-element halves of each 64-group
      → fit (d, dmin) FP16 super-scales: max(scales)/63, max(mins)/63
      → quantize each sub-scale to 6 bits via ls = round(63·scales[j] / max_scale)
      → re-bin q_int against the new d_eff = d·ls (NOT the trained MLX scale —
        the trained MLX integer codebook is invalidated by the scale snap)
  → emit Q4_K bytes directly
```

Quality cost relative to true DWQ: bounded by the 6-bit scale-snap (max ~1/64 = 1.5% relative scale error per sub-block) plus the inevitable re-binning of integer codes against the snapped scale. This is the **same loss** that any K-quant path imposes on any FP16 input, including DWQ's own dequantized weights. It is **not the same** as Path B because the trained scales are used to *derive* `(d, dmin, sc, m)` directly (no `make_qkx2_quants` search) — preserving the DWQ optimization within the bounds K-quant's metadata can express.

The open empirical question: is the trained-scale signal worth retaining if 6-bit scale compression already throws away ~6 bits of precision per sub-block? PPL/quality measurements would settle this. The hf2q codebase already has the infrastructure to A/B this: `KQuantCodecQuantizer` (`k_quant_codec_quantizer.rs`) emits `METHOD_K_QUANT_CODEC_DIRECT` blobs that the GGUF backend at `gguf.rs:1308-1320` passes through unchanged — a fitted-emitter could reuse that exact pipeline.

---

## 6. What hf2q's quantizers actually do

### `k_quant.rs::quantize_row_q4_k` (ref:1202)

Direct port of llama.cpp's `quantize_row_q4_K_ref`. Takes an F32 row, runs `make_qkx2_quants` per 32-weight sub-block, packs the result as 144-byte `BlockQ4K` structs that match `block_q4_K` byte-for-byte (verified by the iter-3b byte-identity test). Output: GGUF Q4_K bytes ready for emit. Sign convention matches GGUF (`w = d·sc·q − dmin·m`).

### `dwq_k_quantizer.rs::DwqKQuantizer`

**Misnomer**: this is **not** mlx-lm DWQ. From the module doc (`dwq_k_quantizer.rs:1-56`):

> `DwqKQuantizer` is the algorithmic replacement: it preserves the *exact same* per-tensor base-vs-sensitive dispatch policy (`extract_layer_index(tensor_name) ∈ sensitive_indices`) but routes through `KQuantCodecQuantizer` with a `KQuantTarget` chosen per variant. Output bytes are final GGUF block bytes carrying the `METHOD_K_QUANT_CODEC_DIRECT` sentinel.

Variants:

| Variant | Base | Sensitive |
|---|---|---|
| `P46` (dwq46) | Q4_K | Q6_K |
| `P48` (dwq48) | Q4_K | Q8_0 |
| `P68` (dwq68) | Q6_K | Q8_0 |
| `P28` (dwq28) | Q2_K | Q8_0 |

Sensitivity is **per-layer scalar** (a layer either is or isn't "sensitive"), drives **bit-width promotion** (Q4_K → Q6_K for sensitive layers), and writes **standard GGUF Q4_K/Q6_K bytes via the K-quant codec**. There is **no scale training**. The misleading name comes from the legacy `MixedBitQuantizer` + `dwq.rs::scale_cal` orchestrator (`dwq_k_quantizer.rs:42-56`), which used the closed-form `optimal_scale = dot(W,Q)/dot(Q,Q)` post-hoc rescale — itself only a rough approximation of what mlx-lm DWQ trains via gradient descent on KL-divergence.

### Format comparison: mlx-lm DWQ vs hf2q "DWQ"

| Aspect | mlx-lm `dwq.py` | hf2q `dwq_k_quantizer.rs` |
|---|---|---|
| Output format | MLX safetensors (3 tensors/layer) | GGUF Q4_K/Q6_K/Q8_0 bytes |
| Scale optimization | Gradient descent on KL-divergence (real DWQ) | None; layer bit-promotion only |
| Sensitivity granularity | Per-tensor (gradient × QDQ-residual signal) | Per-layer-index (RangeInclusive) |
| Group size | 64 (per `defaults_for_mode`) | 32 sub × 256 super (Q4_K-fixed) |
| Bit-packing | 8×4-bit per uint32 | 2×4-bit per uint8 (GGUF Q4_K) |
| Trained scales preserved? | Yes (the whole point) | N/A — no scales are trained |

These are different things wearing the same name. The hf2q "DWQ" is closer in spirit to mlx-lm's `dynamic_quant.py` (mixed-bit dispatch by sensitivity).

---

## 7. Recommendation by track

### Track 1 — `dynamic_quant.py` port → easy

mlx-lm's `dynamic_quant` uses gradient × QDQ-residual to score each linear's sensitivity, binary-searches a threshold to hit a target BPW, then dispatches each layer to either `low_bits` or `high_bits`. **The output is a uniformly-affine MLX-safetensors model with per-layer `bits` overrides in `config.json["quantization"]`.**

This maps cleanly onto GGUF's per-tensor heterogeneity:

- "Low-bits" tensors → emit Q4_K
- "High-bits" tensors → emit Q5_K or Q6_K
- Use llama.cpp's existing `quantize_row_q*_K` (or hf2q's `KQuantCodecQuantizer`) as the per-tensor backend
- Sensitivity ranking maps to hf2q's existing `LayerSensitivity` machinery (`src/calibrate/sensitivity.rs`) — but with the per-tensor (not per-layer) granularity mlx-lm uses

**No trained scales are involved. No format mismatch. Implementation is mostly the sensitivity-estimation loop + a threshold solver.** The hf2q `DwqKQuantizer` infrastructure already handles the dispatch side; what's missing is the `estimate_sensitivities` gradient step.

### Track 2 — DWQ proper port → harder; three options

**Option (a): Port DWQ training loop and output as MLX safetensors** (skip GGUF entirely)
- Trivial format-wise (same path mlx-lm already ships)
- Loses the llama.cpp-compat target — not what hf2q is for
- Useful only if hf2q wants to add an MLX-safetensors output mode

**Option (b): Port DWQ training loop and develop a fitted-Q4_K emitter** (Path C above)
- The trained scales survive format conversion to the extent Q4_K's 6-bit sub-scale grid + FP16 super-scale can express them
- Quality loss bounded by 6-bit scale-snap (~1.5% relative per sub-block) plus integer re-binning
- Novel work: ~500 LOC of scale-projection math + 6-bit-grid fitting; hf2q's `KQuantCodecQuantizer::METHOD_K_QUANT_CODEC_DIRECT` already passes the bytes through to GGUF emit
- Empirical question: A/B PPL against Path B (re-quantize from F16) on a 7B-class model. If the gap is ≤ noise (single-percent PPL), Path B is simpler and the trained-scale work is wasted.
- Open subquestion: how to handle the MLX-group-of-64 vs Q4_K-sub-block-of-32 mismatch — split each MLX group in half (re-deriving sub-statistics from the trained F16 reconstruction) is the principled answer; using the same trained scale for both halves is the lazy answer

**Option (c): Port DWQ training loop, hold weights in F16, emit F16 GGUF** (Path A)
- Lossless preservation of DWQ quality
- 4× larger files — defeats the compression goal
- Useful as a quality-validation baseline for option (b): if option (b) PPL ≈ option (c) PPL, the fitted emitter is working

### Concrete first step

Port `dynamic_quant.py` to hf2q first. It exercises the sensitivity estimation, the per-tensor bit dispatch, and the GGUF emit path with **no format-incompat surprises**. The resulting infrastructure (sensitivity solver + bit-width threshold) is a strict prerequisite for option (b) anyway. Decide on Track 2 path only after measuring how much DWQ adds over a well-tuned mixed-bit GGUF on the same model.

---

## File paths referenced

- `/opt/mlx-lm/mlx_lm/utils.py` (lines 85-172: AWQ→MLX repack; 714-771: save_model; 774-850: quantize_model; 853-896: dequantize_model; 925-950: save)
- `/opt/mlx-lm/mlx_lm/quant/dwq.py` (lines 69-150: dwq_quantize loop; 97-99: scales+biases unfreeze; 405-411: save invocation)
- `/opt/mlx-lm/mlx_lm/quant/dynamic_quant.py` (lines 38-106: estimate_sensitivities; 109-146: estimate_threshold; 229-254: quant_predicate + quantize_model + save)
- `/opt/homebrew/lib/python3.14/site-packages/mlx/nn/layers/quantized.py` (lines 121-176: QuantizedEmbedding; 200-263: QuantizedLinear init/dequant)
- `/opt/llama.cpp/ggml/src/ggml-common.h` (lines 87-89: QK_K=256; 90: K_SCALE_SIZE=12; 317-328: block_q4_K struct)
- `/opt/llama.cpp/ggml/src/ggml-quants.c` (lines 1395-1465: quantize_row_q4_K_ref; 1467-1490: dequantize_row_q4_K)
- `/opt/llama.cpp/gguf-py/gguf/constants.py` (lines 4083-4097: GGMLQuantizationType enum; 4268-4270: Q4_K/Q5_K/Q6_K block sizes)
- `/opt/hf2q/src/quantize/k_quant.rs` (lines 36-96: block-size docstrings; 122-200: BlockQ4K struct; 1202-1340: quantize_row_q4_k)
- `/opt/hf2q/src/quantize/dwq_k_quantizer.rs` (lines 1-100: doc explaining hf2q-DWQ vs mlx-lm-DWQ)
- `/opt/hf2q/src/backends/gguf.rs` (lines 46: GGML_TYPE_Q4_K=12; 1233: type-name resolution; 1308-1325: METHOD_K_QUANT_CODEC_DIRECT pass-through; 2063-2095: K-quant emit gating)
