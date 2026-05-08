# Acceptance Criteria for hf2q DWQ Port (CFA worker 6)

CFA session: `adr020-iter19d`
Worker: 6 of 5 (acceptance criteria, path-agnostic)
Date: 2026-05-07
Reference: ADR-020 §8.2/§8.3, scripts/dwq_kl_parity/queue_stock_run.sh

---

## 1. Acceptance philosophy

"Done" means the hf2q-native DWQ port produces a quantized model whose
mean per-token KL divergence vs BF16 is within a measured tolerance of
the canonical mlx-lm reference (Layer 3 — the load-bearing gate), built
from primitives that each pass an analytical or finite-difference
falsifier (Layer 1 — algorithmic correctness), composed correctly across
layers (Layer 2 — integration), and shippable as a CLI flag with bounded
memory and reproducible output (Layer 4 — ship gate). Each layer exists
because the layer above can pass while a lower layer hides a bug:
end-to-end KL parity within ±15% can be hit by a Linear layer that
happens to cancel a sign error in Adam, so we falsify Adam separately.
Conversely, every primitive can be unit-correct while their composition
diverges from mlx-lm — hence the parity gate is independent and
load-bearing on its own.

The hierarchy resolves disputes: a green Layer-1 + Layer-2 with a red
Layer-3 means we ship nothing and bisect. A red Layer-1 with green
Layer-3 means we have a fragile system and must fix the unit before
shipping. Layer-3 is the only number that justifies a v1 release.

---

## 2. Layer 1: Unit-level criteria (per-component tests)

All tests live under `src/calibrate/` or `tests/dwq_unit/`, run via
`cargo test --release --test dwq_unit_*`, and complete in <60s on M5 Max.

### 2.1 Differentiable Q4 affine quantization (forward + VJP)

**Name:** `dwq_qdq_affine_q4_forward_vjp`

**Falsifier:** A weight matrix `W ∈ R^[256, 256]` with known
group-of-64 affine codebook (scales `s ∈ R^[256, 4]`, biases `b ∈ R^[256, 4]`)
is fed through `qdq_affine_q4(W, s, b)`. The output must satisfy:
- forward: `dequant(quant(W, s, b), s, b)` ≡ `W̃` per the `mlx_lm.quant.affine`
  formula (codes ∈ [0, 15], `W̃ = s · code + b`); byte-equivalence to a
  reference scalar implementation (`mlx-native::weight::dequantize_q4_scalar`)
  on a 4096-element fixture.
- backward (∂L/∂W̃ = G): finite-difference check on a 64-element strip with
  ε=1e-3 must agree with analytical VJP to within `5e-3` relative on
  every component of (∂L/∂s, ∂L/∂b). Codes are frozen — no gradient flows
  to integer indices.

**Test command:**
```bash
cargo test --release --test dwq_unit_qdq -- --nocapture
```

**Pass criterion:**
1. Forward byte-identical to scalar reference on 4096-element fixture.
2. VJP relative error ≤ `5e-3` on 50 random seeds.
3. No NaN or Inf produced when input weight contains values in `[-2.5, 2.5]`
   range (DWQ canonical operating range).
4. Finite-difference epsilon-stability: same VJP at ε ∈ {1e-2, 1e-3, 1e-4}
   within 10× relative.

### 2.2 KL-divergence loss with temperature

**Name:** `dwq_kl_div_loss_temperature_forward_backward`

**Falsifier:** Build a 2-element fixture: teacher logits `[2.0, 0.0]`,
student logits `[1.0, 1.0]`, T=2.0. Analytical KL is
`p_t · (log p_t - log p_s)` with softmax temperature 2.0.
- forward: hf2q result must match Python `scipy.special.kl_div`
  reference within `1e-6` absolute.
- backward: ∂L/∂student_logits via finite-difference (ε=1e-3) must
  match analytical reverse pass within `5e-3` relative.

**Test command:**
```bash
cargo test --release --test dwq_unit_kl_loss
```

**Pass criterion:**
1. Forward matches scipy reference to `1e-6` on 100 random fixtures.
2. Backward FD-vs-analytical ≤ `5e-3` rel on 100 random fixtures.
3. Temperature parameter `T` correctly scales: KL(T=1) > KL(T=2) for
   the same logits (smoother distribution).
4. Numerical stability: no NaN when teacher_logit_max - teacher_logit_min > 30
   (log-sum-exp must be properly stabilized).

### 2.3 Adam with bias correction

**Name:** `dwq_adam_optimizer_bias_correction`

**Falsifier:** Drive Adam(lr=1e-3, β1=0.9, β2=0.999, ε=1e-8) on a
quadratic loss `L(x) = ‖x - x*‖²` where `x*` is a fixed target.
Reference: PyTorch `torch.optim.Adam` on the same fixture and seed.

**Test command:**
```bash
cargo test --release --test dwq_unit_adam
```

**Pass criterion:**
1. After 100 steps, hf2q's `x` matches PyTorch's `x` to `1e-5` absolute
   (computed against a frozen `tests/fixtures/adam_pytorch_reference.json`).
2. Bias correction is correctly applied — at step 1, the effective lr
   must be `lr · sqrt(1 - β2¹) / (1 - β1¹)` ≈ `lr · 0.316`. Test asserts
   this against analytical formula.
3. Convergence: after 1000 steps on the quadratic, ‖x - x*‖ ≤ `1e-4`.

### 2.4 LR schedules (warmup + cosine decay)

**Name:** `dwq_lr_schedule_warmup_cosine`

**Falsifier:** Schedule `warmup=100, total_steps=1024, base=1e-6, final_frac=0.1`.
- step 0: lr = 0
- step 100: lr = 1e-6 (peak after linear warmup)
- step 1024: lr = 1e-7 (final = base · final_frac)
- step 562 (mid-decay): lr ≈ `1e-7 + (1e-6 - 1e-7) · 0.5 · (1 + cos(π · 0.5))` ≈ 5.5e-7

**Test command:**
```bash
cargo test --release --test dwq_unit_lr_schedule
```

**Pass criterion:**
1. All four boundary points within `1e-9` absolute of analytical formula.
2. Monotone non-decreasing on `[0, warmup]`, monotone non-increasing on
   `[warmup, total_steps]`. Asserted by sampling 256 random step indices.
3. Schedule is deterministic — same inputs produce byte-identical outputs
   across 10 runs.

### 2.5 Per-Linear sensitivity computation

**Name:** `dwq_per_linear_sensitivity_two_linear_synthetic`

**Falsifier:** 2-Linear synthetic MLP fixture (50 KB on disk). Hand-derive
∇_W via chain rule (`X^T @ (dq @ next_W^T)`); compare to hf2q's computed
sensitivity vector. This is the iter-9 pattern from ADR-020 §8.2 row 9.

**Test command:**
```bash
cargo test --release --test dwq_unit_sensitivity
```

**Pass criterion:**
1. Per-tensor sensitivity agrees with analytical reference to `5e-3` rel
   tolerance.
2. Ranking (argsort of per-tensor sensitivities) matches analytical
   ranking exactly — no tied-rank ambiguity in fixture.
3. Determinism: 10 runs produce byte-identical sensitivity vectors.

### 2.6 mlx-format safetensors round-trip

**Name:** `dwq_safetensors_roundtrip_mlx_format`

**Falsifier:** Write a known set of quantized tensors (codes ∈ uint32,
scales ∈ bf16, biases ∈ bf16) via hf2q, read back via `mlx_lm` Python
loader (`mlx_lm.utils.load`). Round-trip must produce identical
in-memory representation.

**Test command:**
```bash
cargo test --release --test dwq_unit_safetensors --features python_compat
python3 tests/dwq_unit/verify_mlx_roundtrip.py /tmp/dwq_test_out
```

**Pass criterion:**
1. Header `format` field == `"mlx"` (kld.py line 328 hard-rejects
   anything else).
2. All quantization metadata fields present and correctly typed:
   `weight.scales`, `weight.biases`, `weight.group_size=64`, `weight.bits=4`.
3. Round-trip (write → read) produces byte-identical scales+biases for
   100 random fixtures.
4. mlx_lm Python loader successfully calls `model.load_weights(...)` on
   the output without exception.

### 2.7 Calibration corpus loader

**Name:** `dwq_calibration_corpus_loader_reproducibility`

**Falsifier:** Load `allenai/tulu-3-sft-mixture` with `seed=123,
num_samples=2048, max_seq_length=1025`. The resulting token id stream
must be byte-identical to a frozen reference fixture
(`tests/fixtures/tulu3_seed123_n2048_l1025.tokens.bin`,
SHA256 anchored — see §8).

**Test command:**
```bash
cargo test --release --test dwq_unit_calibration_loader
```

**Pass criterion:**
1. SHA256 of generated token stream matches frozen reference to byte.
2. Sample count matches request (2048).
3. All sequences ≤ max_seq_length (1025).
4. Tokenizer used is the model's own tokenizer (Qwen3.5 vocabulary,
   248320 tokens — NOT a stale 248044 truncated vocab; see existing
   memory `project_qwen35_dwq_pre_505b5b8_broken_2026_05_05`).

---

## 3. Layer 2: Integration criteria

### 3.1 Single-Linear DWQ training reaches KL ≤ 5e-3 (matches iter-19c)

**Test:** `tests/dwq_integration/single_linear_real_bf16.rs`

Loads `model.language_model.layers.0.linear_attn.out_proj.weight` from
the cached BF16 model (jenerallee78 abliterated 35B-A3B), runs
`DwqTrainer::train_single(linear, steps=100, lr=1e-3, T=2.0)`, measures
per-row KL between BF16 teacher and DWQ student on 64 Gaussian activations.

**Pass criterion:** post-train KL ≤ `5e-3` (iter-19c measured 2.73e-3 —
budget 2× headroom for nondeterminism). KL at analytical init must be
≤ `1e-3` (iter-19c measured 8.77e-4 — strict gate, this is the QDQ
correctness check).

### 3.2 Multi-layer DWQ on 1-2 layer toy model converges

**Test:** `tests/dwq_integration/two_layer_toy_convergence.rs`

Synthetic 2-Linear MLP (input 256, hidden 256, output 256). Train DWQ
for 50 Adam steps. Validation KL must strictly decrease from initial.

**Pass criterion:**
1. `validation_kl[step=50] < validation_kl[step=0]` (strict).
2. `validation_kl[step=50] < 0.5 · validation_kl[step=0]` (50% reduction).
3. No NaN/Inf at any step.
4. Adam state monotone: ‖m_t‖ and ‖v_t‖ both bounded, no overflow.

### 3.3 Full Qwen 3.6 35B-A3B forward parity vs mlx-lm reference

**Test:** `tests/dwq_integration/full_forward_parity_qwen35.rs`
(`#[ignore]` by default; runs only with `cargo test -- --ignored
--test full_forward_parity_qwen35`)

Load BF16 Qwen 3.6 35B-A3B via hf2q's autograd-aware forward, run on a
fixed 32-token input ("the quick brown fox jumps over the lazy dog"
chat-templated), compare logits to `mlx_lm.utils.load(...)` Python
forward on the same input.

**Pass criterion:**
1. L2-norm of logit difference ≤ `1e-2` per token (BF16 numerical
   tolerance — small differences from RoPE / norm reductions are expected).
2. Top-1 token agreement ≥ 31/32 positions.
3. Top-5 token Jaccard ≥ 0.95 averaged over 32 positions.

### 3.4 Target precomputation matches mlx-lm

**Test:** `tests/dwq_integration/compute_dwq_targets_parity.rs`

Run hf2q's `compute_dwq_targets` on stock Qwen 3.6 35B-A3B, compare
to mlx-lm's targets directory output (top-1024 logit indices + values
per position).

**Pass criterion:**
1. Top-1024 indices: byte-identical (exact match — same softmax,
   same top_k).
2. Top-1024 values: relative tolerance `1e-3` (BF16 reduction-order
   differences acceptable).
3. File count and naming match mlx-lm's `train/` and `valid/` layout.
4. Total disk size within ±1% of mlx-lm output (sanity for layout).

---

## 4. Layer 3: Parity criteria — the LOAD-BEARING numbers

### 4.1 The v1 ship gate (DWQ parity)

**Gate:** hf2q-DWQ KL ≤ mlx-lm-DWQ KL × 1.15 on the same model + same
calibration + same seed.

**Reference numbers (already measured by iter-19b/iter-19d):**
| Run | mlx-lm KL | RTN baseline KL |
|---|---|---|
| Stock Qwen 3.6 35B-A3B | **0.0610** | 0.0644 |
| Abliterated 35B-A3B | 0.0646 | 0.0670 |

**Justification of ±15% tolerance:**
- 512-sample kld.py with batch=4 on 1025-token sequences exposes ~2M
  per-token measurements. Empirical sample-noise analysis from running
  kld.py twice with different shuffle seeds shows ±3-5% variance.
- BF16 reduction-order non-determinism between hf2q (Metal kernels) and
  mlx-lm (Python/MLX) accounts for another ~5%.
- Algorithmic-equivalence drift (different sample sequencing under
  same `seed=123`, slightly different gradient accumulation order)
  budgets ~5%.
- Sum: ±13% RSS, rounded to 15% as the hard gate.

**Test fixture:**
- Model: stock Qwen 3.6 35B-A3B (canonical recipe target)
- Corpus: `allenai/tulu-3-sft-mixture` with `seed=123, num_samples=2048,
  max_seq_length=1025` for training; same dataset/seed/512-samples for
  kld.py evaluation
- Hyperparams: `bits=4, group_size=64, lr=1e-6, batch_size=2,
  T_distill=2.0, --grad-checkpoint`
- Seed: 123 for both training and evaluation

**Test command:**
```bash
# Pre-built DWQ output already in /tmp/cfa-adr020-iter19d/hf2q_stock_dwq_output/
cd /opt/hf2q-adr020-iter10/scripts/dwq_kl_parity
python3 kld.py \
    --model /tmp/cfa-adr020-iter19d/hf2q_stock_dwq_output \
    --baseline-model ./stock_mlx_bf16 \
    --top-k 1024 \
    --data-path allenai/tulu-3-sft-mixture \
    --sequence-length 1025 \
    --num-samples 512 \
    --batch-size 4 \
    --seed 123 \
    > hf2q_stock_dwq_kld.txt 2>&1
HF2Q_KL=$(grep -aoE '"mean_kl_per_token":\s*[0-9.eE+-]+' hf2q_stock_dwq_kld.txt | tail -1 | grep -oE '[0-9.eE+-]+$')
MLX_LM_KL=0.0610
RATIO=$(python3 -c "print($HF2Q_KL / $MLX_LM_KL)")
echo "hf2q KL = $HF2Q_KL ; mlx-lm KL = $MLX_LM_KL ; ratio = $RATIO"
# PASS: ratio ≤ 1.15 ; SOFT_PASS: 1.15 < ratio ≤ 1.30 ; FAIL: ratio > 1.30
```

**Failure conditions (numerical, not vibes):**
- `hf2q_KL > mlx_lm_KL × 1.30`: HARD FAIL. Block release.
- `mlx_lm_KL × 1.15 < hf2q_KL ≤ mlx_lm_KL × 1.30`: SOFT_PASS — write
  audit doc to `docs/dwq-parity-audit-<date>.md` listing the exact
  delta and a hypothesis (sample-noise / kernel reduction drift /
  fixture mismatch). Tag the v1 release as "soft-parity".
- `hf2q_KL ≤ mlx_lm_KL × 1.15`: HARD PASS. Ship.
- `hf2q_KL > 0.100` (smcleod's broken floor): HARD FAIL regardless of ratio.

### 4.2 RTN parity gate (sanity check)

**Gate:** hf2q-RTN KL ≤ mlx-lm-RTN KL × 1.05 on the same model.

**Justification:** RTN is a closed-form quantization (no training, no
optimizer state, no calibration corpus dependence). Any deviation
beyond ±5% indicates a quantization-format identity bug — almost
certainly incorrect group_size, group axis, or scale/bias dtype.
Tighter than DWQ because there's no algorithmic drift budget.

**Reference numbers:**
- Stock RTN-Q4: mlx-lm KL = **0.0644** → hf2q must be ≤ 0.0676
- Abliterated RTN-Q4: mlx-lm KL = **0.0670** → hf2q must be ≤ 0.0704

**Test command:**
```bash
hf2q convert --quant rtn-q4 --hf-path ./stock_working_dir \
    --mlx-path /tmp/hf2q_stock_rtn_output --bits 4 --group-size 64 --seed 123
python3 /opt/hf2q-adr020-iter10/scripts/dwq_kl_parity/kld.py \
    --model /tmp/hf2q_stock_rtn_output \
    --baseline-model /opt/hf2q-adr020-iter10/scripts/dwq_kl_parity/stock_mlx_bf16 \
    --top-k 1024 --data-path allenai/tulu-3-sft-mixture \
    --sequence-length 1025 --num-samples 512 --batch-size 4 --seed 123
```

**Failure conditions:**
- `hf2q_RTN_KL > 0.0676` (stock): HARD FAIL — bit/group encoding mismatch.
- `hf2q_RTN_KL > 0.100`: HARD FAIL regardless.

---

## 5. Layer 4: Ship-gate criteria

For hf2q v1 to claim "DWQ supported", ALL of:

1. **Parity gate (§4.1)**: `hf2q_DWQ_KL ≤ 0.0610 × 1.15 = 0.0702` on
   stock Qwen 3.6 35B-A3B (HARD PASS or SOFT_PASS with audit).
2. **RTN parity gate (§4.2)**: `hf2q_RTN_KL ≤ 0.0676` on stock 35B.
3. **Cross-family smoke**: one additional model class working
   (Llama 3.x 8B-Instruct OR Mistral 7B-v0.3) with `KL ≤ 0.045` (Q4
   reference per smcleod's published bands for dense models).
4. **CLI flag**: `hf2q convert --quant dwq-q4 --hf-path X --mlx-path Y
   --calibration-data allenai/tulu-3-sft-mixture --num-samples N
   --seed 123 [--bits 4] [--group-size 64] [--learning-rate 1e-6]
   [--batch-size 2] [--max-seq-length 1025] [--grad-checkpoint]`
   exists, parses, and runs end-to-end.
5. **Determinism gate**: re-running the same command (same model,
   same seed, same hyperparams) produces byte-identical safetensors
   output (modulo embedded timestamps in safetensors metadata if any —
   those must be either absent or pinned to a deterministic value).
   Verified via `sha256sum *.safetensors` over 2 runs.
6. **Memory budget**: peak unified memory ≤ 100 GB (fits 128 GB Macs)
   measured via `mach_task_basic_info` polled every 100 ms during the
   full pipeline. SIGTERM at 110 GB.
7. **Documentation**: `docs/quantization-recipes.md` (or new
   `docs/dwq-quickstart.md`) contains a working example, the parity
   number, the soft-fail policy, and a troubleshooting section.
8. **Test coverage**: ≥90% line coverage on `src/calibrate/dwq*.rs`
   measured via `cargo llvm-cov --release` (excluding `#[cfg(test)]`
   blocks). Branch coverage ≥80%.
9. **Coherence check**: `hf2q serve --model <output>` generates 32
   tokens of non-degenerate text on prompt "the quick brown fox" with
   no `<|_end|>` literal-byte leak (defends against the
   pre-505b5b8 vocab-truncation bug class — see existing memory
   `project_qwen35_dwq_pre_505b5b8_broken_2026_05_05`).
10. **GGUF/safetensors invariants**: `safetensors-info` (or equivalent
    on the output) shows `format=mlx`, expected tensor count, no NaN
    in any scale or bias.

---

## 6. Anti-criteria (concrete falsifiers)

These are the failure modes that pass-by-luck inputs would mask — each
must have an active test that catches it.

1. **Tests pass but inference is gibberish** — Catch: §5.9 coherence
   check generates 32 tokens and asserts UTF-8 validity + no
   `<|_end|>` / `<|_im_end|>` literal-byte leak + no
   ascii_ratio > 0.99 with 0% non-printable bytes (which would
   indicate special-token-leak-as-ascii degenerate mode).
2. **Parity gate passes on a single seed but the pipeline is
   non-deterministic** — Catch: §5.5 determinism gate requires 2 runs
   byte-identical.
3. **Forward parity passes but gradient direction is wrong** — Catch:
   §3.2 toy convergence requires `KL[50] < 0.5 · KL[0]` strict, plus
   §2.1/§2.2 finite-difference VJP checks.
4. **DWQ output is "good" because RTN is broken (lower bar)** — Catch:
   §4.2 RTN parity gate is independent and tight (±5%).
5. **Memory passes on the test fixture but blows up on the full model**
   — Catch: §5.6 measures peak via `mach_task_basic_info` on the actual
   full-35B run, not a synthetic.
6. **Output safetensors load in mlx-lm but produce wrong logits** —
   Catch: §3.3 full-forward parity vs mlx-lm reference checks logits,
   not just file format.
7. **Parity number is 0.0701 (just under 0.0702 hard gate) but the
   training loss is ascending** — Catch: log per-step training KL,
   assert validation_loss[final] < validation_loss[initial] (matches
   mlx-lm's `dwq.py:202-207` self-guard).

---

## 7. Test commands (concrete shell)

CI-runnable workflow. Assumes `/opt/hf2q-adr020-iter10/` is the
worktree, `target/release/hf2q` is built, and the parity dir at
`scripts/dwq_kl_parity/` is populated by Worker-side prep.

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /opt/hf2q-adr020-iter10

# ----- Phase 0: build -----
cargo build --release --bin hf2q

# ----- Phase 1: Layer 1 unit tests -----
cargo test --release --test dwq_unit_qdq
cargo test --release --test dwq_unit_kl_loss
cargo test --release --test dwq_unit_adam
cargo test --release --test dwq_unit_lr_schedule
cargo test --release --test dwq_unit_sensitivity
cargo test --release --test dwq_unit_safetensors --features python_compat
cargo test --release --test dwq_unit_calibration_loader

# ----- Phase 2: Layer 2 integration tests -----
cargo test --release --test dwq_integration_single_linear_real_bf16
cargo test --release --test dwq_integration_two_layer_toy
cargo test --release --test dwq_integration_compute_targets_parity
# Full-forward parity (slow, gated):
RUN_SLOW_TESTS=1 cargo test --release --test full_forward_parity_qwen35 -- --ignored

# ----- Phase 3: Layer 3 parity (the load-bearing gate) -----
DWQ_OUT=/tmp/hf2q_stock_dwq_output
RTN_OUT=/tmp/hf2q_stock_rtn_output
SOURCE=/opt/hf2q-adr020-iter10/scripts/dwq_kl_parity/stock_working_dir
BF16=/opt/hf2q-adr020-iter10/scripts/dwq_kl_parity/stock_mlx_bf16

# RTN baseline (cheap)
target/release/hf2q convert --quant rtn-q4 \
    --hf-path "$SOURCE" --mlx-path "$RTN_OUT" \
    --bits 4 --group-size 64 --seed 123

# DWQ training (4-6 hours on M5 Max)
target/release/hf2q convert --quant dwq-q4 \
    --hf-path "$SOURCE" --mlx-path "$DWQ_OUT" \
    --calibration-data allenai/tulu-3-sft-mixture --num-samples 2048 \
    --max-seq-length 1025 --bits 4 --group-size 64 \
    --learning-rate 1e-6 --batch-size 2 --seed 123 --grad-checkpoint

# kld.py parity measurements
cd /opt/hf2q-adr020-iter10/scripts/dwq_kl_parity
python3 kld.py --model "$DWQ_OUT" --baseline-model "$BF16" \
    --top-k 1024 --data-path allenai/tulu-3-sft-mixture \
    --sequence-length 1025 --num-samples 512 --batch-size 4 --seed 123 \
    > hf2q_stock_dwq_kld.txt 2>&1
python3 kld.py --model "$RTN_OUT" --baseline-model "$BF16" \
    --top-k 1024 --data-path allenai/tulu-3-sft-mixture \
    --sequence-length 1025 --num-samples 512 --batch-size 4 --seed 123 \
    > hf2q_stock_rtn_kld.txt 2>&1

# Gate evaluation
HF2Q_DWQ=$(grep -aoE '"mean_kl_per_token":\s*[0-9.eE+-]+' hf2q_stock_dwq_kld.txt | tail -1 | grep -oE '[0-9.eE+-]+$')
HF2Q_RTN=$(grep -aoE '"mean_kl_per_token":\s*[0-9.eE+-]+' hf2q_stock_rtn_kld.txt | tail -1 | grep -oE '[0-9.eE+-]+$')
python3 - <<EOF
hf2q_dwq, hf2q_rtn = $HF2Q_DWQ, $HF2Q_RTN
mlx_dwq, mlx_rtn = 0.0610, 0.0644
dwq_ratio = hf2q_dwq / mlx_dwq
rtn_ratio = hf2q_rtn / mlx_rtn
print(f"DWQ: hf2q={hf2q_dwq:.5f} mlx={mlx_dwq:.5f} ratio={dwq_ratio:.3f}")
print(f"RTN: hf2q={hf2q_rtn:.5f} mlx={mlx_rtn:.5f} ratio={rtn_ratio:.3f}")
import sys
fail = False
if hf2q_dwq > 0.100: print("FAIL: DWQ above smcleod broken floor"); fail=True
if dwq_ratio > 1.30: print("FAIL: DWQ ratio > 1.30"); fail=True
elif dwq_ratio > 1.15: print("SOFT_PASS: DWQ in 1.15-1.30 — write audit")
else: print("PASS: DWQ ≤ 1.15")
if rtn_ratio > 1.05: print("FAIL: RTN ratio > 1.05 (encoding bug)"); fail=True
sys.exit(1 if fail else 0)
EOF

# ----- Phase 4: Layer 4 ship gates -----
# Coherence
target/release/hf2q serve --model "$DWQ_OUT" --prompt "the quick brown fox" --max-tokens 32 > /tmp/dwq_coherence.txt
python3 -c "
import sys
with open('/tmp/dwq_coherence.txt','rb') as f: data=f.read()
assert b'<|_end|>' not in data, 'literal-byte token leak'
assert b'<|_im_end|>' not in data, 'literal-byte token leak'
assert data.decode('utf-8'), 'invalid UTF-8'
"
# Determinism (re-run with same seed)
target/release/hf2q convert --quant dwq-q4 --hf-path "$SOURCE" \
    --mlx-path /tmp/hf2q_stock_dwq_output_run2 \
    --calibration-data allenai/tulu-3-sft-mixture --num-samples 2048 \
    --max-seq-length 1025 --bits 4 --group-size 64 \
    --learning-rate 1e-6 --batch-size 2 --seed 123 --grad-checkpoint
diff <(cd "$DWQ_OUT" && sha256sum *.safetensors) \
     <(cd /tmp/hf2q_stock_dwq_output_run2 && sha256sum *.safetensors)
# Coverage
cargo llvm-cov --release --test 'dwq_*' --html --output-dir /tmp/dwq_coverage
# Coverage ≥90% asserted by parsing /tmp/dwq_coverage/html/index.html or
# by `cargo llvm-cov --json --summary-only` and checking line_pct.
```

---

## 8. Calibration data integrity

**Reproducibility anchor:** all training and evaluation use:
- Corpus: `allenai/tulu-3-sft-mixture` (HuggingFace dataset)
- Tokenizer: model's own (Qwen 3.5 vocab = 248320 tokens, eos=248046,
  per fix `505b5b8` — must NOT be the truncated 248044 vocab from
  pre-`505b5b8` builds)
- Seed: `123`
- Sample count: `2048` (training), `512` (evaluation)
- Max sequence length: `1025`
- Tokenization: chat-template applied (see `scripts/dwq_kl_parity/queue_stock_run.sh:198-221`)

**Frozen reference fixtures (commit to repo at
`tests/fixtures/dwq_calibration/`):**
- `tulu3_seed123_n2048_l1025.tokens.bin` — concatenated u32 token IDs;
  SHA256 anchored. (Generated once from kld.py's `load_eval_tokens`
  shim with above settings; size ≈ 8 MB.)
- `tulu3_seed123_n2048_l1025.sha256` — single-line SHA256 hex of above.
- `tulu3_seed123_n512_l1025_eval.tokens.bin` — 512-sample evaluation
  subset; same anchoring.

CI step verifies `sha256sum -c tulu3_seed123_n2048_l1025.sha256` before
any DWQ training run. Failure → block test (corpus drift would
silently invalidate all parity numbers).

**Smcleod's published numbers** for cross-validation against community
references (don't gate on these — they used a different tokenizer
sample, but they're sanity within ±2×):
- Qwen 3.6 35B-A3B mlx-community DWQ Q4 vs 8-bit ref: `0.02663`
- Qwen 3.6 35B-A3B RTN Q4 vs 8-bit ref: `0.07418`

---

## 9. Performance acceptance criteria

Measured on M5 Max (128 GB unified memory) at AC power, no other
heavy GPU consumers.

| Metric | Target | Hard ceiling | Notes |
|---|---|---|---|
| Per-step training time | ≤ 20 s/step at batch=2 | ≤ 30 s/step | mlx-lm achieves ~13 s/step; hf2q-native gets ~50% headroom for first release. Ceiling is 30 s (would imply >20 hour total run = impractical). |
| Peak unified memory | ≤ 80 GB at batch=2 | ≤ 100 GB | mlx-lm 3-pass recipe peaked at 60.6 GB. hf2q gets +20 GB headroom. 100 GB is the absolute ceiling; SIGTERM at 110 GB. |
| Disk I/O (target cache) | ≤ 10 GB | ≤ 12 GB | mlx-lm's `compute_dwq_targets` writes ~6.6 GB on stock 35B for top-1024. hf2q port writes the same shape; 50% growth budget. |
| Total wall time | ≤ 6 hours on stock 35B | ≤ 8 hours | mlx-lm 3-pass took ~3 hours (Pass 2: 12.6 min targets + Pass 3: ~2h45 train at 65-75 t/s). hf2q gets 2× headroom. |
| Coverage (`cargo llvm-cov`) | ≥ 90% line / ≥ 80% branch | — | |

**Test command:**
```bash
/usr/bin/time -l target/release/hf2q convert --quant dwq-q4 ... 2>&1 | tee dwq_run.log
# Parse "maximum resident set size" from time -l; convert pages → GB
# Run mach_task_basic_info watchdog in parallel (poll every 100 ms,
# log peak, SIGTERM at 110 GB).
```

---

## 10. Documentation acceptance criteria

1. **ADR row 19b status update**: After parity gate passes, edit
   ADR-020 §8.2 row "19b (half-2)" status from `NEXT (gated on 14b +
   11h)` → `DONE — hf2q KL = X.XXXX vs mlx-lm 0.0610 (ratio Y.YYY)
   — see commit ABCDEF`.
2. **README "DWQ quickstart"**: 20-line section in `README.md` showing:
   ```bash
   hf2q convert --quant dwq-q4 --hf-path Qwen/Qwen3.6-35B-A3B \
       --mlx-path my-dwq-q4 --num-samples 2048 --seed 123
   ```
   plus expected wall time, memory peak, and the parity number.
3. **CFA experiment scripts retained**: `scripts/dwq_kl_parity/` is
   kept as-is for reproducibility. Add a `README.md` in that
   directory pointing at this acceptance-criteria file and the
   ADR-020 row 19a/19b history.
4. **Soft-fail audit template**: if SOFT_PASS triggers, write
   `docs/dwq-parity-audit-<YYYY-MM-DD>.md` with: measured ratio,
   delta vs target, hypothesis, additional measurements taken, and
   either remediation plan or accepted-risk justification.

---

## Summary — the load-bearing gate and soft-fail policy

The single number the team has to hit is **mean per-token KL ≤ 0.0702**
(= mlx-lm's 0.0610 × 1.15) on stock Qwen 3.6 35B-A3B, measured via the
vendored `kld.py` against the `stock_mlx_bf16` baseline at
`seed=123, num_samples=512, batch=4, sequence_length=1025` over the
`allenai/tulu-3-sft-mixture` corpus. A measurement in `(0.0702, 0.0793]`
(= 1.15-1.30× ratio) is SOFT_PASS, ships as v1 with a public audit
documenting the delta and a hypothesis. A measurement above 0.0793, or
above 0.100 absolute regardless of ratio, is HARD FAIL — the port is
not shipped. RTN parity (KL ≤ 0.0676 on the same fixture, ±5% of
mlx-lm's 0.0644) is a tight independent guard that catches
quantization-format identity bugs. All Layer 1+2 falsifiers must be
green before the parity gate is run on the full model — a Layer-3 PASS
on a Layer-1-broken pipeline is a coincidence we will not exploit.

---

Worker 6 complete — acceptance criteria written to `/tmp/cfa-adr020-iter19d/worker6-acceptance-criteria.md`.
