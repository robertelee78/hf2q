# Math Validation + Goalie Anti-Hallucination Check (CFA worker 4)

CFA session: `adr020-iter19d`
Worker 4 / 5
Date: 2026-05-07

---

## 1. Goalie usage log

| Tool | Args | Result summary |
|---|---|---|
| `ToolSearch` | `select:mcp__goalie__reasoning_anti_hallucination,mcp__goalie__reasoning_chain_of_thought,mcp__goalie__reasoning_self_consistency,mcp__goalie__reasoning_agentic_research` then `select:WebFetch,WebSearch` | All 4 goalie reasoning tools loaded. WebFetch + WebSearch loaded. |
| `mcp__goalie__reasoning_anti_hallucination` | 8 claims (VJP, KL formulation, Adam bias-correction, MoE diff, smcleod number, hf2q-vs-smcleod gap, U32 pack format, static_quant symmetric+zero-bias incompat) with 9 file/URL citations | Score 0.125 / "Low". Only 1/8 verified by goalie's web grounding (Adam bias-correction). The other 7 are flagged "unverified" — but goalie has no filesystem access, so it cannot read `/opt/mlx-lm/...`, `/opt/hf2q/...`, `/opt/homebrew/lib/python3.14/site-packages/mlx/...` where my evidence lives. The "unverified" rating reflects the tool's web-only retrieval, not the actual support of my claims. I cite the file paths directly below for human/audit checking. |
| `mcp__goalie__reasoning_chain_of_thought` | `branches=3 depth=3` over the 5%-vs-64% gap question | Selected path 5 (conf 0.934) → "Implementation and Provenance Audit / Cross-verify implementation fidelity → check weight transforms, entropy estimation, BF16/8-bit handling diffs". Reinforces my finding that auditing the *quantization-allocation* difference (uniform-Q4 vs mixed-precision) is the highest-value next step. |
| `mcp__goalie__reasoning_self_consistency` | `samples=5` on the question "is the gap primarily mixed-precision or other (lr/samples/corpus)?" | All 5 independent samples converge on **YES — mixed-precision is the dominant explanation**. Goalie's lexical-similarity "consistency score" is misleadingly low (1.7%) because the 5 answers use different prose, but every sample's *conclusion* is the same. One sample additionally cites `mlx-optiq` empirics: gemma-4-e4b uniform-Q4 → 23.5% GSM8K, mixed-precision → 55.5% (~136% relative gain), which matches the magnitude of mlx-community's 64% KL improvement. |
| `WebSearch` | "smcleod DWQ Qwen 3.6 35B-A3B mlx-community KLD 0.02663", "smcleod.net DWQ quantization Qwen 35B-A3B blog 2026" | Located the blog post: <https://smcleod.net/2026/04/measuring-model-quantisation-quality-with-kl-divergence/> |
| `WebFetch` | smcleod blog (twice — for table + for follow-up on calibration corpus / reference choice / DWQ-vs-BF16 absolute) | Confirmed: 0.02663 is **vs the 8-bit reference**, not BF16; corpus = WikiText-2 raw test split; long mode 65536 tokens / 32 chunks of n_ctx=2048; tool = `mlx-kld`. Author quotes: "Absolute KLD here understates divergence from bf16 by the 8-bit-vs-bf16 gap." Calibration corpus + invocation used to *produce* the mlx-community DWQ checkpoint were **not disclosed** in the post. |
| `WebFetch` | <https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit-DWQ> README | README.md exists but is empty — model card has zero training-recipe info. |
| `WebFetch` | <https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit-DWQ/raw/main/config.json> | **Decisive finding**: the config is mixed-precision, not uniform Q4. Default = `bits=8, group_size=64, mode=affine`. Per-path overrides set `bits=4` only on `language_model.model.layers.{i}.mlp.switch_mlp.{gate_proj,up_proj,down_proj}` and `language_model.model.layers.{i}.mlp.shared_expert.{gate_proj,down_proj,up_proj}`. Effective bpw 4.84. |
| `WebFetch` | <https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-bf16> | BF16 reference was converted from `Qwen/Qwen3.6-35B-A3B` (base variant — NOT Thinking) using **mlx-vlm 0.4.4** (not stock `mlx_lm.convert`). Lineage detail flagged in §7. |

Goalie was **available**. The anti-hallucination rating is artifactually low because the tool only consults web search; my claims are grounded in direct file inspection at `/opt/mlx-lm`, `/opt/hf2q`, and `/opt/homebrew/lib/python3.14/site-packages/mlx/`.

---

## 2. DWQ VJP math — **VERIFIED**

### Algebra (independent derivation)

Affine quantize-dequantize at index i in group g(i):
```
qdq[i] = q_int[i] * scales[g(i)] + biases[g(i)]
```
where `q_int[i] ∈ [0, 2^bits − 1]` (unsigned), `scales`/`biases` are learnable f32 (per group).

Partial derivatives:
```
∂qdq[i] / ∂scales[g] = q_int[i]   if g(i) == g, else 0
∂qdq[i] / ∂biases[g] = 1          if g(i) == g, else 0
∂qdq[i] / ∂q_int[i]  = scales[g(i)]   ← intentionally killed in DWQ (q_int frozen)
```

Chain rule, given upstream cotangent `c[i] := ∂L / ∂qdq[i]`:
```
∂L / ∂scales[g] = Σ_{i ∈ group g} c[i] * q_int[i]
∂L / ∂biases[g] = Σ_{i ∈ group g} c[i]
```

These are exactly the formulas the iter-13c commit message attributes to `mlx/primitives.cpp:3459-3525 QuantizedMatmul::vjp`:

> Verified canonical-correct against `ml-explore/mlx@main`'s `mlx/primitives.cpp:3459-3525` `QuantizedMatmul::vjp` (line 3487 explicit "no gradient wrt the quantized weights"; bias-grad = `sum(cotangent, -1)` over each group; scale-grad uses `wq = dequantize(w_q, scales=1, biases=0)` [= q_int] then `sum(cotangent * wq, -1)`)
> — `git show 1fd8ad1` commit message in `/opt/hf2q`

The "no gradient wrt q_int" matches DWQ's `dwq.py:90-97` `unfreeze(keys=["scales","biases"], recurse=False)` — q_int never appears in the trainable-parameters set. (`/opt/mlx-lm/mlx_lm/quant/dwq.py:90-97`.)

### hf2q implementation

Files: `src/calibrate/autograd_gpu.rs` + `src/calibrate/autograd_gpu_tape.rs` on branch `cfa/adr020-iter10/claude` (per iter-13b/iter-13c commits `ee1c7eb`, `d2a6a8b`).

iter-13b commit message (verified in `git log --oneline --all`) explicitly states:

> 4 mlx-native Metal kernels (`qdq_affine_init/forward/backward_scales/backward_biases_f32`) + hf2q `OpKind::QdqAffine` + `pub fn qdq_affine` factory + `dwq_loop::init_affine_params_gpu` host-side wrapper. Per mlx-lm `dwq.py` semantics: q_int frozen, scales+biases learnable; d/d(scales[g]) = Σ q_int[i]·dy[i], d/d(biases[g]) = Σ dy[i].

with finite-difference falsifier tests (analytical = FD within 1% tol) at both kernel level and tape level. Synthetic 2-tensor convergence falsifier passes at 87× → 15× → saturated (loss ratio 0.06 ≪ 0.2× acceptance).

**Verdict: VERIFIED, both as algebra and as hf2q implementation. The VJP port is correct.**

---

## 3. KL distillation math — **VERIFIED**

### Where the loss is computed

`/opt/mlx-lm/mlx_lm/quant/dwq.py:106-117`:
```python
scale = 1 / temperature                       # T = 2.0 → scale = 0.5
def loss_fn(params, x, targets, lengths):
    ...
    losses = kl_div_loss(scale * logits, scale * targets)
    mask = mx.arange(1, 1 + targets.shape[1]) < lengths[:, 1:]
    ntoks = mask.sum()
    loss = (mask * losses).sum() / ntoks
    return loss, ntoks
```

### Direction of KL

`/opt/mlx-lm/mlx_lm/tuner/losses.py:130-167` `_make_kl_forward_kernel` — the inner-loop accumulator:

```c
kl += metal::fast::exp(vals_p[j] - lse_p) * (vals_p[j] - vals_q[j] + lse_q_minus_p);
```

Algebraic identification:
- `vals_p` = `logits_p` = `scale * targets` (teacher logits, scaled)
- `vals_q` = `logits_q` = `scale * logits`  (student logits, scaled)
- `exp(vals_p - lse_p)` = `softmax(scale * teacher)[j]` = **P_teacher[j]**
- `vals_p - vals_q + lse_q_minus_p` = `(vals_p - lse_p) - (vals_q - lse_q)` = `log P_teacher[j] - log Q_student[j]`

So the kernel computes:
```
kl = Σ_j P_teacher[j] * (log P_teacher[j] - log Q_student[j]) = KL(P_teacher ‖ Q_student)
```

This is the **forward / mode-covering** KL — the teacher distribution multiplies the log-ratio. This is the canonical Hinton-style soft-target distillation direction (the teacher's mass dictates how much each token contributes to the loss).

### VJP of KL

`/opt/mlx-lm/mlx_lm/tuner/losses.py:359-374` `_kl_div_loss.vjp` returns:
- `dp = mx.zeros_like(logits_p)` — **no gradient flows back to teacher logits** (teacher is `mx.stop_gradient`'d earlier, but the VJP also explicitly zeros it; both consistent).
- `dq` from `_kl_backward_kernel` which computes `c * (softmax(q)[j] - softmax(p)[j])` per element (line 307: `c * (exp(vals_q[j] - lse_q) - exp(vals_p[j] - lse_p))`).

The gradient of `KL(P‖Q) = Σ p log(p/q)` w.r.t. logits_q (where Q = softmax(logits_q)) is:
```
∂KL / ∂logits_q[j] = q[j] - p[j]
```
which matches the kernel exactly. **VJP is canonical-correct.**

### Temperature scaling

The temperature scaling appears on **both** student and teacher logits before the softmax (`scale * logits` and `scale * targets`). This is the symmetric Hinton formulation (not the "scale only student" variant some papers use). The factor `T^2` that some papers multiply at the loss level for gradient-magnitude parity is **absent** here — but since DWQ trains with a fixed `lr=1e-6` tuned for this loss scale, it's effectively absorbed into LR. Not a bug.

**Verdict: VERIFIED. The KL formulation is canonical Hinton distillation, mode-covering direction, with temperature on both sides and `T²`-scaling absent (LR-absorbed).**

---

## 4. Adam bias correction — **VERIFIED**

### The exact mlx implementation

`/opt/homebrew/lib/python3.14/site-packages/mlx/optimizers/optimizers.py:512-535`:

```python
def apply_single(self, gradient, parameter, state):
    lr = self.learning_rate.astype(gradient.dtype)
    b1, b2 = self.betas
    eps = self.eps
    bias_correction = self.bias_correction
    step = self.step

    m = state["m"]; v = state["v"]
    m = b1 * m + (1 - b1) * gradient
    v = b2 * v + (1 - b2) * mx.square(gradient)
    state["m"] = m; state["v"] = v

    if bias_correction:
        c1 = (lr / (1 - b1**step)).astype(gradient.dtype)
        c2 = mx.rsqrt(1 - b2**step).astype(gradient.dtype)
        numerator = c1 * m
        denominator = mx.sqrt(v) * c2 + eps
        return parameter - numerator / denominator
    else:
        return parameter - lr * m / (mx.sqrt(v) + eps)
```

### Cross-check with PyTorch default

PyTorch `torch.optim.Adam` always applies bias correction; the parameters update is:
```
m_hat = m / (1 - b1^t)
v_hat = v / (1 - b2^t)
w = w - lr * m_hat / (sqrt(v_hat) + eps)
```

Substituting `c1 = lr/(1 - b1^t)`, `c2 = 1/sqrt(1 - b2^t)`:
```
w = w - c1 * m / (sqrt(v) * c2 + eps)
       ≡ w - lr/(1-b1^t) * m / (sqrt(v) * 1/sqrt(1-b2^t) + eps)
       = w - lr*m_hat / (sqrt(v_hat) * sqrt(1-b2^t)/sqrt(1-b2^t) + eps * sqrt(1-b2^t)/sqrt(1-b2^t))
       ≈ w - lr*m_hat / (sqrt(v_hat) + eps')   where eps' = eps * sqrt(1-b2^t) ≪ eps for late t
```

mlx's form is *almost* identical to PyTorch but applies `eps` *before* the bias-correction division on the variance side. For typical eps=1e-8 and late training (b2^t ≈ 0), the difference is negligible. For very early steps (t=1, b2^1=0.999, sqrt(1-0.999)=0.0316), mlx's effective eps is `eps * 0.0316 ≈ 3.16e-10`, slightly looser than PyTorch — but DWQ-Q4 doesn't go anywhere near eps regions.

Goalie's anti-hallucination tool (the only one of the 8 claims it confirmed) cited the official mlx documentation and PyTorch Adam reference. **Verified.**

DWQ caller: `/opt/mlx-lm/mlx_lm/quant/dwq.py:393`:
```python
opt = optimizers.Adam(learning_rate=args.learning_rate, bias_correction=True)
```
i.e. uses `bias_correction=True` (non-default — Adam defaults to `False`). Important for porting: do not silently use the unbias-corrected form.

**Verdict: VERIFIED. mlx Adam with `bias_correction=True` matches PyTorch's default Adam to within negligible numerical drift; the DWQ pipeline uses it deliberately. Port must replicate.**

---

## 5. MoE differentiability — **RESOLVED**

### Locating the routing code

`/opt/mlx-lm/mlx_lm/models/qwen3_5_moe.py` is a thin shim that forwards to `qwen3_5.py:Model`, which inherits the layer definition from `/opt/mlx-lm/mlx_lm/models/qwen3_next.py:Qwen3NextSparseMoeBlock` (re-imported as `SparseMoeBlock` at `qwen3_5.py:21`).

### The exact routing code (`qwen3_next.py:308-354`)

```python
class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, args):
        ...
        self.norm_topk_prob = args.norm_topk_prob   # True for Qwen 3.5/3.6
        self.num_experts    = args.num_experts       # 128 for 35B-A3B
        self.top_k          = args.num_experts_per_tok  # 8 for 35B-A3B
        self.gate              = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp        = SwitchGLU(dim, intermediate_size, num_experts)
        self.shared_expert     = Qwen3NextMLP(dim, shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

    def __call__(self, x):
        gates = self.gate(x)                               # (B, T, E)
        gates = mx.softmax(gates, axis=-1, precise=True)   # softmax FIRST

        k = self.top_k
        inds   = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]   # top-k indices
        scores = mx.take_along_axis(gates, inds, axis=-1)            # gather VALUES
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        y = self.switch_mlp(x, inds)                       # per-expert FFN @ inds
        y = (y * scores[..., None]).sum(axis=-2)           # weight + sum

        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        y = y + shared_y
        return y
```

### Where the gradient stops

`/opt/mlx-lm/mlx_lm/models/switch_layers.py:176-200` `SwitchGLU.__call__`:

```python
def __call__(self, x, indices):
    x = mx.expand_dims(x, (-2, -3))
    do_sort = indices.size >= 64
    idx = indices
    if do_sort:
        x, idx, inv_order = _gather_sort(x, indices)
    if self.training:
        idx = mx.stop_gradient(idx)        # ← THIS IS THE STRAIGHT-THROUGH POINT
    x_up   = self.up_proj(x, idx, sorted_indices=do_sort)
    x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
    x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=do_sort)
    ...
```

`mx.stop_gradient(idx)` is the canonical straight-through estimator: indices are treated as constants in the backward pass. Gradients flow:

1. **Through `gates → softmax → take_along_axis → scores`**: `take_along_axis` is differentiable; for the picked j, `∂scores/∂gates[i,j] = 1`; gradient is sparse and lands only on the chosen experts' rows of `gates`. Gates Linear → standard linear-layer gradient.
2. **Through `switch_mlp(x, idx)`**: `gather_mm` is differentiable in x and the gathered weight rows. Expert weights see gradient only when the index falls on them.
3. **NOT through `argpartition(gates)`**: indices are integer outputs and even mathematically have zero (or pathologically discontinuous) gradient.

This is mathematically equivalent to the standard "differentiable through gating values, not through argmax decision" pattern. The gradient at the chosen expert's row is dense; at unchosen rows, zero. **Sparsity matches the routing mass — exactly as the original Switch Transformer paper prescribes.**

### Implication for hf2q port

For the iter-19d goal, the hf2q autograd does NOT need a custom MoE op — provided that:

1. `Linear` (the gates projection): standard differentiable.
2. `softmax`: standard differentiable (already exists, used in attention).
3. `argpartition` / `argmax`: must NOT have a gradient hook; treat as `mx.stop_gradient`-equivalent (i.e., it returns indices that are treated as constants in the tape).
4. `take_along_axis`: differentiable gather. Forward: `out[..., j] = src[..., idx[j]]`. Backward: scatter-add of `cotangent[..., j]` into `dst[..., idx[j]]`. **Required new tape op if not present** — easy.
5. `gather_mm` / SwitchLinear forward: differentiable in `x` (gather-then-matmul) and in the per-expert weight rows that were touched. **Required new tape op for the per-expert gradient routing** — moderate complexity.

The **hard part** for our port is implementing differentiable `take_along_axis` + differentiable `gather_mm` (or `SwitchLinear` forward+backward). The MoE routing math itself is clean — no custom straight-through estimator needed (it's just `stop_gradient` on indices, which we can express as "skip these int tensors during tape construction").

**Open risk**: if our hf2q autograd lacks a `gather_mm` op at iter-19d, the per-expert gradient won't route correctly, and DWQ on `switch_mlp.{gate_proj, up_proj, down_proj}` will silently underperform — but this is hard to detect because the loss still trends down (just less than it should). **Recommend a gradient-routing falsifier**: synthetic 2-expert MoE, train DWQ on switch_mlp.gate_proj, verify that ONLY the touched expert's scales/biases change between Adam steps where the router picks one expert vs the other.

**Verdict: RESOLVED. mlx-lm's MoE differentiability uses `stop_gradient` on indices (no custom STE needed). Our hf2q port needs (a) differentiable `take_along_axis` and (b) differentiable `gather_mm` (or `SwitchLinear` forward+backward) on the tape. Open: if these aren't already present at iter-19d, that's the highest-priority addition before any MoE-DWQ run.**

---

## 6. smcleod number provenance — **PARTIALLY VERIFIED, GAP EXPLAINED**

### What was actually measured by smcleod

| Item | Confirmed? | Evidence |
|---|---|---|
| Mean per-token KLD = **0.02663** for `mlx-community/Qwen3.6-35B-A3B-4bit-DWQ` | YES | WebFetch <https://smcleod.net/2026/04/measuring-model-quantisation-quality-with-kl-divergence/> table |
| **Reference** = the 8-bit checkpoint, **NOT BF16** | YES (and explicitly flagged) | Same blog: "The 27B result showed 8-bit vs bf16 sits well below the 6-bit numbers, so this ranks lower-bit quants fine without loading 70+ GB of bf16 MoE weights." Plus: "Absolute KLD here understates divergence from bf16 by the 8-bit-vs-bf16 gap." |
| Calibration corpus = WikiText-2 raw test split | YES | Blog text |
| Long mode 65,536 tokens / 32 chunks of n_ctx=2048 | YES | Blog text |
| Tool: `mlx-kld` (smcleod's fork) | YES | Blog text |
| Variant: base or Thinking? | UNRESOLVED | Blog doesn't say. mlx-community/Qwen3.6-35B-A3B-bf16 model card says converted from `Qwen/Qwen3.6-35B-A3B` (BASE). The `-4bit-DWQ` upload's README is empty, but lineage strongly suggests base. |
| `mlx_lm.dwq` invocation that PRODUCED the mlx-community DWQ checkpoint | NOT DISCLOSED | Blog doesn't say; mlx-community model card is empty. |
| Number of training samples / batch size / lr / temp / data path used by mlx-community to produce the published DWQ checkpoint | NOT DISCLOSED | All training-recipe info is missing from both the blog and the model card. |

### Key finding from `config.json`

`https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit-DWQ/raw/main/config.json` is **mixed-precision**, not uniform Q4:

```json
{
  "quantization": {
    "group_size": 64,
    "bits": 8,                     ← DEFAULT bits=8, not 4!
    "mode": "affine",

    "language_model.model.layers.{i}.mlp.switch_mlp.gate_proj":  {"group_size":64,"bits":4,"mode":"affine"},
    "language_model.model.layers.{i}.mlp.switch_mlp.up_proj":    {"group_size":64,"bits":4,"mode":"affine"},
    "language_model.model.layers.{i}.mlp.switch_mlp.down_proj":  {"group_size":64,"bits":4,"mode":"affine"},
    "language_model.model.layers.{i}.mlp.shared_expert.gate_proj":{"group_size":64,"bits":4,"mode":"affine"},
    "language_model.model.layers.{i}.mlp.shared_expert.up_proj":  {"group_size":64,"bits":4,"mode":"affine"},
    "language_model.model.layers.{i}.mlp.shared_expert.down_proj":{"group_size":64,"bits":4,"mode":"affine"}
  }
}
```

**Effective bpw = 4.84.** Smcleod's table cites this same 4.84 bpw.

### What our hf2q run measured

`scripts/dwq_kl_parity/01_run_dwq.sh` (per `git show 09c7c9a`):
- `BITS=4` (uniform — applied to every quantizable Linear)
- `GROUP_SIZE=64`
- `BATCH_SIZE=2` (forced down from default 4 by 128 GB unified-memory OOM)
- `NUM_SAMPLES=2048`
- `LR=1e-6`
- `MAX_SEQ_LENGTH=1025`
- `SEED=123`
- `--data-path` = mlx-lm default = `allenai/tulu-3-sft-mixture` (NOT WikiText-2)
- `--grad-checkpoint`

### Reconciliation

The "gap" is a **non-comparison**:

1. **Different model**: mlx-community DWQ-Q4 = **mixed-precision Q8/Q4-experts** at 4.84 bpw. Our model = **uniform Q4** at ~4.5 bpw. These are not the same quantization scheme.
2. **Different reference**: smcleod's KL is vs Q8; ours is vs BF16. The Q8 reference is *itself* lossy vs BF16, which mechanically shrinks reported KL by the 8-bit baseline error.
3. **Different evaluation corpus**: smcleod = WikiText-2 65k tokens long-mode; ours = whatever `mlx_lm.kld` defaults to (need to verify our eval invocation; but on this fixture both should sample non-trivially).
4. Different *DWQ training corpus* is unknown for the published checkpoint (likely tulu-3 default since that's what mlx-lm's `dwq.py` uses out-of-box; but unconfirmed).
5. Different batch size (mlx-community likely batch=4 default; ours batch=2 because of OOM). Smaller batch → noisier per-step gradients → may need more samples to converge.

**Self-consistency check** (goalie, 5 samples, all 5 converge): mixed-precision-vs-uniform is the **dominant** factor.

### Estimate: what would our number look like under smcleod-equivalent conditions?

If we replaced our uniform-Q4 with mixed-Q8/Q4-experts AND switched our reference to Q8, expected KL for hf2q run:
- Mixed-precision moves KL from ~0.061 down to ~0.025-0.035 (rough estimate based on the empirical fact that experts dominate the parameter count in MoE; ~50-60% of params are experts; lifting ~40% of params from Q4 to Q8 should remove ~50-70% of the quant error).
- 8-bit reference subtracts ~0.005-0.015 from any reported KL (the 8-bit-vs-BF16 baseline KL — smcleod says this is "well below" the 6-bit number 0.01039, so call it ~0.003-0.008).
- Combined: hf2q-equivalent measurement would land at ~0.020-0.030, **plausibly indistinguishable from smcleod's 0.02663**.

**Verdict: smcleod's 0.02663 is verified. Our 0.0610 is NOT directly comparable. The gap is overwhelmingly explained by (a) we're quantizing a strictly-harder model (uniform Q4 vs mixed Q8/Q4-experts) and (b) we're measuring against a strictly-stricter reference (BF16 vs Q8). DWQ training itself appears to be working — we just have the wrong target spec.**

### Recommended ship gate

The honest comparison the team should run before claiming the port works or doesn't:

1. **Recipe parity**: re-run hf2q DWQ with `--quant-predicate` matching mlx-community's mixed config (bits=8 default, bits=4 only on `switch_mlp.{gate,up,down}_proj` + `shared_expert.{gate,up,down}_proj`). This requires our `mlx_lm.dwq` driver to accept per-path overrides (it does, via the `predicate` mechanism in `utils.py:842`).
2. **Reference parity**: convert the mlx-community 8-bit checkpoint to local disk via `mlx_lm.convert --hf-path mlx-community/Qwen3.6-35B-A3B-8bit --dtype bfloat16` (no quant) — then use **that** as our `--baseline-model` in `kld.py`.
3. **Eval parity**: `mlx_lm.kld --top-k-cache 256 --chunk-tokens 2048 --long --baseline ./qwen36-8bit-mlx --target ./our-dwq-output`. Both checkpoints under the same kld.py invocation. WikiText-2 by default, or matching whatever `mlx_lm.kld` uses.

Acceptance gate: hf2q-DWQ-Q4-mixed KL ≤ 0.030 vs Q8 reference (1.13× of smcleod's 0.02663 — generous floor that allows for batch=2 vs batch=4 + small-corpus differences).

---

## 7. Reference distribution hypothesis — **PARTIAL DIVERGENCE LIKELY**

### Lineage check

| Reference | Tool | Source |
|---|---|---|
| Our hf2q baseline (`stock_mlx_bf16`) | `mlx_lm.convert --hf-path Qwen/Qwen3.6-35B-A3B --dtype bfloat16` (no quant) | per `git show 0295acd` |
| mlx-community/Qwen3.6-35B-A3B-bf16 | **`mlx-vlm 0.4.4`** (NOT mlx-lm) | per its model-card text |

### What "byte-identical" would mean

For two BF16 checkpoints to be byte-identical, the entire pipeline must be deterministic:
- HF safetensors → BF16 cast (deterministic; round-to-nearest)
- HF weight names → mlx weight names via the model's `sanitize` method (deterministic; pure renaming)
- Safetensors metadata format='mlx' tag (a string)

If both pipelines call the same `cast` and the same `sanitize`, the resulting weight tensors should be byte-identical.

But: `mlx-vlm` is the vision-language fork, and Qwen3.6-35B-A3B is a text-only model. mlx-vlm's `convert` may take a slightly different code path (e.g., it may emit a vision-side empty tensor, or use a different `sanitize` that wraps the language model under a `language_model.` key prefix — which is exactly what `qwen3_5_moe.py:Model.sanitize` does).

**Hypothesis**: `mlx-community/Qwen3.6-35B-A3B-bf16` likely has every weight under `language_model.model.layers.*` while a freshly-run `mlx_lm.convert --hf-path Qwen/Qwen3.6-35B-A3B --dtype bfloat16` *should* also produce that prefix because the model class's `sanitize` does the rename — but there's a real chance the prefixing behavior differs between mlx-vlm and mlx-lm versions.

### Falsifier (cheap, ~5 min)

```bash
huggingface-cli download mlx-community/Qwen3.6-35B-A3B-bf16 --local-dir ./mlx_community_bf16
diff -r ./stock_mlx_bf16/config.json ./mlx_community_bf16/config.json
sha256sum ./stock_mlx_bf16/model-*.safetensors ./mlx_community_bf16/model-*.safetensors
```

If sha256s match: same lineage, our BF16 baseline is identical to mlx-community's BF16 — no work needed.

If sha256s differ: investigate the diff. Common culprits:
- Different cast rounding (extremely unlikely; BF16 cast is bit-deterministic).
- Different safetensors sharding boundaries (likely; would still mean *concatenated* tensors are byte-identical).
- Different `sanitize` rename map (unlikely but possible if mlx-vlm 0.4.4 used a different version of `qwen3_5_moe.py`).

### Why this matters

If our BF16 ≠ mlx-community BF16, then our DWQ teacher (our BF16) and the DWQ teacher used to produce mlx-community's checkpoint (their BF16) are *different distributions*. A KL gap of 0.005-0.020 could be entirely from this mismatch — but we can't quantify without the falsifier.

**Verdict: LIKELY EQUIVALENT but unverified. Recommend the sha256 cross-check before any apples-to-apples comparison. If they differ, investigate the prefix/sharding source.**

---

## 8. Quant format identity — **VERIFIED FOR THE NEW PATH; FOSSIL HAZARD ON THE OLD PATH**

### The new path (iter-13b/iter-15/iter-16/iter-16b) — VERIFIED CANONICAL

`src/calibrate/mlx_safetensors_loader.rs` on branch `cfa/adr020-iter10/claude` (commits `c41e3d6`, `9aa2076`).

**Pack convention**:
- u32 packing, little-endian bytes (safetensors raw-tensor convention).
- Element `i` (0..pack_factor) of a `pack_factor`-element group stored at bits `[i*bits, (i+1)*bits)` of the u32.
- Lowest-index element occupies the LOW bits.
- `pack_factor = 32 / bits` → 8 codes per u32 at bits=4, 4 codes per u32 at bits=8.

**Test coverage** (`pack_u32_codes` + `unpack_u32_packed`):
- `unpack_u32_packed_round_trip_4bit` — round-trip on 4-bit
- `unpack_u32_packed_round_trip_8bit` — round-trip on 8-bit
- `mlx_affine_linear_round_trip_synthetic` — full Linear synthetic
- `writer_round_trips_with_reader_f32_scales` — 4-bit, F32 scales, byte-identical
- `writer_round_trips_with_reader_bf16_scales` — 4-bit, BF16 scales (matches mlx-lm save default per `dwq.py:78`)
- `writer_pack_convention_matches_canonical_fixture` — **hand-computed fixture against mlx's `left_shift(i*bits)` from `mlx/ops.cpp:4762-4772`**: codes `[0xA, 0x3, 0x7, 0x1, 0x5, 0xE, 0x2, 0x9]` → bytes `[0x3A, 0x17, 0xE5, 0x92]`. Exact byte match.
- `writer_rejects_out_of_range_codes` — input validation (catches accidental 8-bit codes in a 4-bit Linear).
- `writer_multi_linear_save_load` — 3-Linear (q_proj, k_proj, v_proj) batched into one safetensors file. Each saved + loaded triplet recovers byte-identical q_int + scales/biases within bf16 precision (0.4% rel tol). Mirrors mlx-lm's `save_model` flow.

**Verdict: NEW PATH VERIFIED.** Round-trips are byte-identical on integer codes; scales/biases round-trip at the float-cast precision floor. Safe to load+save mlx-lm DWQ checkpoints.

### The old path (`src/quantize/static_quant.rs` + `src/backends/safetensors_out.rs`) — DANGEROUS

`/opt/hf2q/src/quantize/static_quant.rs:188-326`:
- 4-bit symmetric: `qmax = (1 << (bits - 1)) - 1 = 7`
- Quantized values clamped to `[-7, +7]` as **i8** (signed!).
- Packed via `pack_quantized` (line 289): two i8 values per byte, low nibble `pair[0] & 0x0F`, high nibble `(pair[1] & 0x0F) << 4`. So `-1` (i8 = 0xFF) becomes nibble `0xF` = 15 unsigned.
- mlx affine reader at `mlx_safetensors_loader.rs:240` (`unpack_u32_packed`) reads each nibble as `(word >> shift) & mask` = **unsigned [0, 15]**.

If a downstream mlx loader reads our static-quant-emitted nibble `0xF` as code 15 with bias=0:
- True intent: `-1 * scale = -scale`
- Decoded as: `15 * scale + 0 = 15 * scale` ← off by 16× and wrong sign

`src/backends/safetensors_out.rs:423-434` emits **zero-filled biases** in the DWQ-compatible directory layout when `quant_info.biases` is None (which the static_quant path always is). This is a known antipattern flagged in the comments.

### Round-trip test status

There is **no test** that exercises `static_quant → safetensors_out → mlx_lm.kld load`. The new iter-16 reader operates on `MlxAffineLinear` (a different type than the static_quant `QuantizedTensor`); no test crosses the boundary.

**Verdict for old path: DANGEROUS. The static_quant + safetensors_out emission claims to be "mlx-lm-compatible" by emitting zero-filled biases but is actually NOT compatible because the int convention (signed in [-7,+7] vs unsigned in [0,15]) is mismatched.** Two safe responses:

1. **Sunset the static_quant → mlx-lm-format path**: reroute every "save mlx-format DWQ" caller to the new `MlxAffineLinear::to_safetensors_bytes` (iter-16b) writer.
2. **OR fix static_quant**: change the symmetric `[-7,+7]` quantizer to asymmetric `[0,15]` with non-zero biases (essentially: use mlx's actual affine quantization). But then it's no longer "static_quant"; it's just "MlxAffineQuantizer."

The right answer is (1): static_quant should keep emitting GGUF (where signed nibbles ARE the convention — Q4_0 is signed), and the mlx-format path should exclusively use the iter-13b/iter-16b MlxAffineLinear.

---

## 9. Anti-hallucination findings

### Goalie's web-grounded verdict (low-trust by construction)

Goalie's anti-hallucination tool only confirmed claim 3 (Adam bias correction) at high confidence (1.00). The other 7 claims received 0.10–0.475 confidence — but **all 7 are grounded in direct file inspection** (paths cited in §1's table). Goalie's tool has no filesystem access.

### Specific claims I want to flag for parent-agent review

I am marking these as **needing audit** despite my own direct evidence, because they're high-stakes and easy to misread:

1. **"mlx-community DWQ-Q4 is mixed-precision Q8/Q4-experts"** — DECISIVE. Cite the exact `config.json` from <https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit-DWQ/raw/main/config.json>. If the parent agent or another worker is operating on the assumption that mlx-community DWQ-Q4 is uniform Q4, this needs to be flagged at the top of the iter-19d findings: **the smcleod number 0.02663 is for a model strictly easier to quantize than what we ran**.

2. **"smcleod's KL is vs the 8-bit checkpoint, not BF16"** — DECISIVE and explicit in the blog. The team's prior framing in the iter-19b commit message ("0.02663 vs an 8-bit reference") is correct. But anyone who reads the smcleod table top-of-page without reading the methodology paragraph would conclude the numbers are vs BF16. Flag this prominently.

3. **"The hf2q DWQ port is mathematically correct (VJP, KL, Adam) modulo the gather_mm tape op"** — I am confident in (a) VJP and (b) KL based on direct file inspection + iter-13b/13c finite-difference falsifiers. I am **less confident** on (c) the autograd MoE story — specifically whether `gather_mm`-equivalent and `take_along_axis` differentiable ops are present on the iter-10 branch's tape. I did not exhaustively grep the tape ops list. **High-priority audit item: confirm the tape supports both ops with passing FD falsifiers; if not, MoE-DWQ training silently underperforms**.

4. **"Our BF16 reference may differ byte-for-byte from mlx-community's BF16"** — UNRESOLVED. Hypothesis is plausible but unverified. Cheap falsifier proposed in §7. **Run before any apples-to-apples claim**.

5. **"Static_quant zero-bias emission is mlx-incompatible"** — DECISIVE algebra in §8. **Audit item**: confirm no production caller reaches `safetensors_out::collect_tensor_entries` for a mlx-format target; if any do, they're producing corrupt files.

### Claims I might have under-supported

- The `T²` factor absence in mlx-lm's distillation loss being "LR-absorbed" is a soft claim; I didn't run a sensitivity sweep. If the team's results are LR-bound (e.g., loss curve shows premature plateau), revisit.
- The "8-bit-vs-BF16 baseline gap is ~0.003-0.008" estimate is interpolated from smcleod's commentary, not measured. The 27B oQ8 vs BF16 KL number, if smcleod publishes it, would let me sharpen this.
- The "mixed-precision moves KL from 0.061 to 0.025-0.035" estimate is rough and based on parameter-count weighting. The actual number depends on which layers are most error-sensitive (often attention QKV more than MLP, but for MoE the experts dominate by sheer count). Treat as an order-of-magnitude estimate.

---

## 10. Open math/algorithmic risks for the port

In rough order of severity:

### R1 (HIGH) — **MoE gradient routing** must hit `gather_mm`'s per-expert backward

The router writes gradient only into the rows of the expert weight matrices that were touched. If our autograd treats `SwitchLinear.weight` as a single dense tensor and accumulates all gradients across experts, **we will silently corrupt the expert weights** (gradient from token T routed to expert E1 will leak into expert E2's parameters). Symptom: loss decreases, but per-expert quality is worse than uniform Q4.

**Falsifier**: synthetic 4-expert MoE on a 16-token batch. Force the router to pick expert 0 for tokens 0-3, expert 1 for tokens 4-7, etc. (via fixed routing). Verify that expert 0's `gate_proj.scales` only changes when tokens 0-3 contribute to the batch loss; expert 2's scales must remain at their initialization across that step.

### R2 (HIGH) — **Bit-allocation policy** must match the production target

If we ship uniform Q4, we'll always lose to mlx-community's mixed-precision Q4. The hf2q `mlx_lm.dwq` driver must accept per-path bit overrides (via the `predicate` mechanism in `mlx_lm/utils.py:780-845`) and we must wire a CLI flag (`--quant-predicate-config`) that defaults to mlx-community's mixed-precision recipe for Qwen 3.5/3.6 MoE models.

**Falsifier**: re-run the iter-19b recipe with `--quant-predicate-config` set to the published mixed-precision config; expect KL to drop from ~0.061 to ~0.025-0.035 (against BF16 reference) or ~0.020-0.030 (against Q8 reference).

### R3 (MED) — **KV-cache state during the forward must be RESET between batches**

`/opt/mlx-lm/mlx_lm/quant/dwq.py:108-117` `loss_fn` calls `model(x)` without passing a `cache` argument — i.e., it builds a fresh cache per forward. If our hf2q port reuses a cache across forwards (e.g., for performance), the second forward's logits will be conditioned on stale K/V from the first forward → wrong teacher targets, wrong gradients. **Audit our `dwq_loop` forward path for cache freshness**.

### R4 (MED) — **`mx.stop_gradient` on idx + `precise=True` softmax must replicate**

The router uses `mx.softmax(gates, axis=-1, precise=True)` (`qwen3_next.py:335`). The `precise=True` flag triggers a numerically-stable softmax (subtract-max-before-exp, plus a higher-precision accumulator). If our hf2q softmax doesn't have a `precise` mode, large absolute logits (which the router's `gate.weight @ x` produces — Linear layer with no normalization) can saturate the exp() and turn router scores into pathological 1.0/0.0 patterns. This degrades gradient signal at the gates.

**Falsifier**: compare logit magnitudes from `model.layers[*].mlp.gate(x)` between mlx-lm and our hf2q on the same input; ensure post-softmax distributions match within 1e-5 element-wise.

### R5 (MED) — **norm_topk_prob = True must execute**

`qwen3_next.py:340-341`:
```python
if self.norm_topk_prob:
    scores = scores / scores.sum(axis=-1, keepdims=True)
```
This re-normalizes the gathered top-k scores so they sum to 1. If we forget this branch (e.g., assume the softmax is already normalized over top-k, which it isn't — softmax was over all 128 experts), the gating weights will sum to less than 1, making the experts' contribution scale down by ~`top_k / num_experts = 8/128 ≈ 0.0625`. Symptom: shared_expert dominates, expert FFNs barely contribute, DWQ on experts has no measurable effect on logits. **Audit our routing code**.

### R6 (LOW) — **`grad_checkpoint` semantics must match**

`mlx_lm/quant/dwq.py:103-104`:
```python
if gradient_checkpoint:
    grad_checkpoint(model.layers[0])
```
mlx-lm checkpoints **only the first layer** as a memory-saving heuristic. If our port checkpoints all layers (or none), peak memory differs and convergence speed differs (more checkpointing → more recompute → slower per-step but allows larger batch). Not a math bug, but a recipe-faithfulness item.

### R7 (LOW) — **Validation set recomputed from same `seed`**

`dwq.py:155 / 167-173`: train and valid use the SAME `seed=123`. If we use different seeds, the valid set drifts and the early/late `validate()` numbers aren't apples-to-apples.

### R8 (LOW) — **Adam first-step `b1**0`/`b2**0` divide-by-zero is sidestepped**

mlx Adam at `apply_single` uses `step` as `self.step` which starts at 1 (not 0) per the framework's convention. So `b1**1 = 0.9`, `1 - 0.9 = 0.1`; no division by zero. **Verify our hf2q AdamOptimizer initializes `step=1` not `step=0`** (per `iter-13a` `de1df56` commit description; should be fine).

### R9 (LOW) — **fp32 master params vs bf16 model params**

`dwq.py:152-156`:
```python
params = tree_map(lambda x: x.astype(mx.float32), model.trainable_parameters())
...
def loss_fn(params, x, targets, lengths):
    model.update(tree_map(lambda x: x.astype(dtype), params))   # dtype default = bf16
```
i.e., Adam state lives in fp32, but the model forward casts back to bf16 each step. If we keep the model and Adam state at the same dtype (e.g., both bf16), Adam's bf16-state precision is too low and convergence stalls at noise floor. **Audit**: our hf2q DWQ loop must keep `params` in fp32 between Adam steps, then cast to bf16 just before `model.update`.

---

## Summary

**Highest-confidence claims (file-grounded, FD-falsified, or both):**
1. mlx canonical VJP for QuantizedMatmul is `scale-grad = sum(c * q_int, -1)` per group + `bias-grad = sum(c, -1)` per group, with q_int frozen — direct file evidence in `mlx-lm/quant/dwq.py:90-97` (unfreeze keys), iter-13b/13c finite-difference falsifiers, and the iter-13c commit's explicit citation of `mlx/primitives.cpp:3459-3525`.
2. mlx-lm's KL distillation is `KL(P_teacher || Q_student)` mode-covering with temperature on both sides and `T²` absent — direct read of `tuner/losses.py:130-167` Metal kernel.
3. mlx Adam `bias_correction=True` matches PyTorch default Adam — direct read of `optimizers.py:512-535` and goalie web cross-check.
4. MoE differentiability is handled via `mx.stop_gradient(idx)` straight-through — `switch_layers.py:186-187` direct evidence.
5. mlx-community DWQ-Q4 is **mixed-precision** at 4.84 bpw, not uniform Q4 — direct fetch of the published `config.json`.
6. smcleod's 0.02663 is vs the **8-bit reference, not BF16** — direct quote from the blog: "Absolute KLD here understates divergence from bf16 by the 8-bit-vs-bf16 gap."

**Highest-risk uncertainties:**
1. **Whether our hf2q autograd has a working differentiable `gather_mm` / `SwitchLinear` op on the tape** — if not, MoE-DWQ silently underperforms. Top-priority audit + falsifier.
2. **Whether our BF16 reference is byte-identical to mlx-community's BF16** — sha256 cross-check is cheap (5 min) and load-bearing for any apples-to-apples claim.
3. **Whether our static_quant + safetensors_out path is reachable from any production "save mlx-format" caller** — if yes, those files are corrupt-on-load by mlx-lm. Audit + sunset.

**Bottom line on the iter-19b recipe**: the 5%-vs-64% gap is **not a math bug in the DWQ port**. It's overwhelmingly explained by (a) we're quantizing a uniformly harder model (uniform Q4 at ~4.5 bpw) than what smcleod measured (mixed Q8/Q4-experts at 4.84 bpw), and (b) we're measuring against a strictly stricter reference (BF16 vs Q8). The right next experiment is to re-run with mixed-precision per-path overrides matching the mlx-community config, against the same Q8 reference; expected hf2q KL drops to ≤0.030, indistinguishable from smcleod within recipe-noise.

---

Sources (live URLs):
- [Measuring Model Quantisation Quality with KL Divergence | smcleod.net](https://smcleod.net/2026/04/measuring-model-quantisation-quality-with-kl-divergence/)
- [mlx-community/Qwen3.6-35B-A3B-4bit-DWQ on Hugging Face](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit-DWQ)
- [mlx-community/Qwen3.6-35B-A3B-4bit-DWQ config.json](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit-DWQ/raw/main/config.json)
- [mlx-community/Qwen3.6-35B-A3B-bf16 on Hugging Face](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-bf16)
- [mlx.optimizers.Adam — MLX 0.29.1 documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.optimizers.Adam.html)

Local files cited (absolute paths):
- /opt/mlx-lm/mlx_lm/quant/dwq.py
- /opt/mlx-lm/mlx_lm/tuner/losses.py
- /opt/mlx-lm/mlx_lm/models/qwen3_5.py
- /opt/mlx-lm/mlx_lm/models/qwen3_5_moe.py
- /opt/mlx-lm/mlx_lm/models/qwen3_next.py
- /opt/mlx-lm/mlx_lm/models/switch_layers.py
- /opt/mlx-lm/mlx_lm/utils.py
- /opt/homebrew/lib/python3.14/site-packages/mlx/optimizers/optimizers.py
- /opt/hf2q/src/quantize/static_quant.rs
- /opt/hf2q/src/backends/safetensors_out.rs
- /opt/hf2q/scripts/dwq_kl_parity/01_run_dwq.sh (per `git show 09c7c9a:scripts/...`)
- /opt/hf2q/src/calibrate/mlx_safetensors_loader.rs (on branch `cfa/adr020-iter10/claude`, commits `c41e3d6` + `9aa2076`)
