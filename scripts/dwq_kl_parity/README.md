# ADR-020 iter-19b — DWQ KL parity harness vs mlx-lm

Closes the question "is our hf2q DWQ port at or below mlx-lm's
production KL on the user's reference model?" by running the
canonical mlx-lm tooling against `jenerallee78/Qwen3.6-35B-A3B-
Abliterix-EGA-abliterated` and capturing the target KL number
locally.

## Reference model

- HF: https://huggingface.co/jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated
- Local cache: `~/.cache/huggingface/hub/models--jenerallee78--Qwen3.6-35B-A3B-Abliterix-EGA-abliterated/`
- Architecture: `Qwen3_5MoeForCausalLM` (hybrid 3:1 linear+full attention, 35B-A3B MoE)
- Source dtype: BF16, 65 GB, 42 safetensors shards
- Already cached locally (verified 2026-05-07) — no download required.

## What's in this directory

| File | Purpose |
|---|---|
| `kld.py` | Vendored from open mlx-lm PR #1146, plus a `load_eval_tokens` shim around `mlx_lm.quant.utils.load_data` (PR's expected import isn't shipped yet — see below). |
| `01_run_dwq.sh` | Drives `mlx_lm.dwq` to produce a Q4 affine-quantized DWQ output from the BF16 reference. |
| `02_run_kld.sh` | Drives the vendored `kld.py` to measure mean per-token KLD between the BF16 reference (cached) and the DWQ Q4 output. |
| `03_run_rtn_baseline.sh` | Same KLD measurement, but for a naive RTN-Q4 (no DWQ training) — needed to confirm DWQ actually beats the naive baseline by the expected ~2.8× per smcleod's published numbers. |

## Acceptance gate (per ADR §8.2 row 19a)

- **Target**: hf2q's port produces final per-token KL ≤ **0.030** on
  Qwen 3.6 35B-A3B at Q4 (matches mlx-lm published 0.02663 + 13%
  margin).
- **Floor**: > **0.100** = broken (substantial divergence per
  smcleod's published thresholds).

## What the harness does NOT do

- It does NOT run hf2q's port — that requires iter-14b (real
  GgufTeacherProvider) + iter-11h (full multi-layer Qwen3.5MoE
  forward on GpuTape) to land first.
- This harness is the FIRST HALF of iter-19b: it captures the
  canonical mlx-lm number.  The SECOND HALF runs hf2q on the same
  inputs and compares.

## Why the `load_eval_tokens` shim is needed

PR #1146 (the open `mlx_lm.kld` PR) imports `load_eval_tokens` from
`mlx_lm.utils`, but that function is added by a separate downstream
PR not yet merged.  We shim it locally to call the existing
`mlx_lm.quant.utils.load_data` (the same calibration-v5 corpus that
`mlx_lm.dwq` uses), which produces a compatible output.  When the
downstream PR merges, the shim can be removed.

## Runtime + memory expectations on M5 Max

- `01_run_dwq.sh`:  **~hours** (training a Q4 student on 35B BF16
  teacher).  Peak unified memory: ~75 GB.  Output: another safetensors
  shard set (~18 GB Q4).
- `02_run_kld.sh`:  **~30-60 min** (full forward over the
  calibration corpus + per-batch KL).  Peak: ~80 GB (both teacher +
  student loaded).
- `03_run_rtn_baseline.sh`:  Same shape as 02.

Total: budget a half-day for the full sequence.  Run on AC power.

## Logging output

Each script prints final numbers to stdout AND appends a short JSON
record to `results.jsonl` for easy diffing across runs:

```json
{"ts": "2026-05-07T12:34:56Z", "step": "01_dwq", "model": "...", "duration_sec": 12345, "valid_loss_initial": 0.087, "valid_loss_final": 0.027}
{"ts": "2026-05-07T18:45:01Z", "step": "02_kld", "candidate": ".../dwq", "baseline": "...BF16", "mean_kld": 0.02663}
```
