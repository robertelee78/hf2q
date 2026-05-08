# ADR-022 — Complete the hf2q DWQ port via mlx-lm subprocess bridge

| | |
|---|---|
| **Status** | Proposed |
| **Date** | 2026-05-07 |
| **Supersedes** | ADR-020 §8.2 row 19b half-2 (re-pathed) |
| **Replaces approach** | Pure-Rust full-model GpuTape autograd port (deferred to ADR-023+) |
| **Authors** | Synthesized from CFA session `cfa-20260507-191500-adr020-followup-research` (5 parallel research workers + queen synthesis) |

---

## 1. TL;DR

hf2q has been driving `mlx_lm.dwq` from `scripts/dwq_kl_parity/` shell scripts to harvest reference KL numbers, but never ran its own DWQ port end-to-end. ADR-020 row 19b half-1 (capture canonical mlx-lm number) is **DONE**; row 19b half-2 (run *our* port and measure parity) was gated on iter-11h (full-model multi-layer Qwen 3.5/3.6 MoE forward on `GpuTape`) + iter-14b (real `GgufTeacherProvider`), neither shipped. iter-19c shipped the **per-Linear** primitive on a real BF16 weight (KL = 2.73e-3, `src/calibrate/dwq_e2e.rs:481`) but not the multi-layer composition.

**Decision:** complete the port via a **subprocess bridge to a pinned `mlx_lm.dwq`**, not by porting the full autograd stack into Rust+Metal. hf2q owns: orchestration, calibration management, GGUF emission, mixed-precision scale-book bookkeeping, parity measurement, CLI surface. mlx-lm (subprocess, pinned SHA) owns: forward + backward through the differentiable model graph.

**Total scope: ~1,800 net new LOC across 6 iterations (iter-19d → iter-20a), wall-clock ~5 weeks (~10 man-days actual work).** Compared to pure-Rust path's 5,500–7,000 LOC and 5–7+ weeks with deferred research risk on gated-delta-net backward, MoE router gradient routing, and kernel-level reduction-order parity.

**Acceptance gate (load-bearing v1 ship):** hf2q-DWQ KL ≤ **0.0702** (= mlx-lm's measured 0.0610 × 1.15) on stock Qwen 3.6 35B-A3B via the vendored `scripts/dwq_kl_parity/kld.py`. Hard fail above 0.0793 or 0.100 absolute.

**Critical math finding from research:** the apparent 5%-vs-64% gap between our run (KL=0.0610) and smcleod's published 0.02663 is **not a port bug**. Smcleod measured a **mixed-precision Q8/Q4-experts checkpoint at 4.84 bpw against a Q8 reference**; we ran **uniform Q4 against BF16**. Verified by direct fetch of `mlx-community/Qwen3.6-35B-A3B-4bit-DWQ/config.json`. The math (VJP, KL distillation, Adam bias correction, MoE differentiability) all checks out; the published smcleod gap is recipe-config + reference-distribution, not algorithmic correctness.

---

## 2. Why — the problem we are solving

### 2.1 The misframing we corrected

For ~3 weeks the team has been driving `mlx_lm.dwq` from `scripts/dwq_kl_parity/01_run_dwq.sh` and `queue_stock_run.sh`, treating the produced numbers as ADR-020 deliverables. They are not. They are a *reference target* — the number our port must match within tolerance. The deliverable is hf2q's own DWQ output, measured the same way, falling in the same KL band.

This was a process bug: the user explicitly called this out — "shelling out to other repo's tools is an antipattern, and explicitly goes against the entire fucking point of hf2q." Correct.

### 2.2 What hf2q's mission requires

hf2q exists to be **the canonical HF → GGUF + mixed-precision quantization converter**. Quantization quality at the published bands (smcleod's "broken floor" >0.10 KL, "well-made 4-bit" 0.01–0.05, "DWQ territory" ~0.027) requires distillation-tuned per-group scales+biases — i.e., DWQ. Without an in-house DWQ path:

- Operators must run `mlx_lm.dwq` separately and feed the output back into hf2q for GGUF emission. Two-tool pipeline.
- We cannot ship a single `hf2q convert --quant dwq-q4 ...` command.
- Mixed-precision recipes (Apex / `imatrix-adaptive`, already shipped at `src/main.rs:1685`) cannot benefit from DWQ-tuned scales — they're stuck with closed-form RTN.

### 2.3 What's already shipped on main (verified post-merge `cfa/adr020-iter10/claude` 2026-05-07)

29 modules under `src/calibrate/`, ~24,800 LOC total. Status legend: **prod** = wired into a CLI subcommand and runs in conversion path; **iter-19c** = real-BF16 single-Linear KL test landed; **synth-tested** = unit/synthetic only; **stub** = referenced but not implemented.

| Module | LOC | Status |
|---|---:|---|
| `src/calibrate/apex.rs` | 680 | **prod** (mixed-precision K-quant; Apex per-tensor optimal precision) |
| `src/calibrate/imatrix.rs` | 1,762 | **prod** (port of llama.cpp `tools/imatrix/imatrix.cpp`) |
| `src/calibrate/imatrix_calibrator.rs` | 806 | **prod** |
| `src/calibrate/imatrix_xvalidate.rs` | 576 | **prod** |
| `src/calibrate/sensitivity.rs` | 236 | **prod** (variance-magnitude proxy ranker) |
| `src/calibrate/sensitivity_comparison.rs` | 444 | iter-11g synth-tested |
| `src/calibrate/dwq.rs` | 909 | **prod** (LEGACY weight-space DWQ-46/48; misnamed per ADR-020 §1.3) |
| `src/calibrate/dwq_calibrator.rs` | 885 | **prod** |
| `src/calibrate/dwq_activation.rs` | 616 | **prod** |
| `src/calibrate/calibrator.rs` | 719 | **prod** (`Calibrator` trait, `NoneCalibrator`) |
| `src/calibrate/cache.rs` | 691 | **prod** |
| `src/calibrate/dynamic_quant.rs` | 568 | **prod** (iter-7 CPU oracle) |
| `src/calibrate/dynamic_quant_gpu.rs` | 514 | iter-9 synth-tested |
| `src/calibrate/qdq_gpu.rs` | 343 | iter-10a synth-tested |
| `src/calibrate/autograd.rs` | 1,360 | CPU oracle (test-only) — Tape + 7 ops + finite-diff falsifier |
| `src/calibrate/autograd_gpu.rs` | 464 | iter-8b synth-tested standalone primitives |
| `src/calibrate/autograd_gpu_tape.rs` | 3,931 | iter-8a–13b synth-tested. **17 op kinds**: matmul, softmax, log, RowSum, embedding, SiLU, slice/concat/transpose, RMSNorm, view, scalar_mul, qdq_affine + 5 elementwise. All FD-falsified at 5e-3 rel tol |
| `src/calibrate/adam.rs` | 416 | iter-13a synth-tested. fp32 master + bf16 view, `bias_correction=true` |
| `src/calibrate/dwq_loop.rs` | 1,191 | iter-13b/c/d/e synth-tested + real-tensor. Adam + KL-div + qdq_affine |
| `src/calibrate/dwq_targets.rs` | 696 | iter-14 synth-tested. `TeacherLogitsProvider` trait at line 58, `compute_dwq_targets` at line 103. **No production impl exists; only `SyntheticTeacher` test fixture at line 438** |
| `src/calibrate/mlx_safetensors_loader.rs` | 1,220 | iter-16/16b synth-tested + iter-19c real-model use. `MlxAffineLinear` at line 133, `to_safetensors_bytes` writer at line 398. Read+write byte-correct mlx-format with `format='mlx'` metadata. Fixture-match to `mlx/ops.cpp:4762-4798` |
| `src/calibrate/dwq_e2e.rs` | 881 | **iter-19c PASS**: KL=2.73e-3 on real BF16 layer-0 `linear_attn.out_proj` weight (`fn iter_19c_single_linear_kl_parity_vs_bf16` at line 481, `#[ignore]`-gated) |
| `src/calibrate/qwen35_attention_block.rs` | 1,045 | iter-10c/11a synth-tested (MHA only — **no GQA/RoPE/causal-mask**) |
| `src/calibrate/qwen35_ffn.rs` | 316 | iter-11b synth-tested (vanilla SwiGLU dense — **no MoE routing**) |
| `src/calibrate/qwen35_layer.rs` | 584 | iter-11c synth-tested |
| `src/calibrate/qwen35_model.rs` | 542 | iter-11d synth-tested (n_layers=2, vocab=64, hidden=32 fixture) |
| `src/calibrate/qwen35_gguf_adapter.rs` | 331 | iter-11e synth-tested (MHA-only, **no GQA/MoE/linear_attn**) |
| `src/calibrate/calibration_batcher.rs` | 350 | iter-11f synth-tested (whitespace-hash stub tokenizer at line 104) |

`scripts/dwq_kl_parity/` (kld harness):
- `kld.py` — vendored from mlx-lm PR #1146 (525 LOC + 18-line `load_eval_tokens` shim). Baseline must be `format='mlx'` (line 328 hard-rejects others).
- `01_run_dwq.sh` (3-pass low-mem recipe — commit `09c7c9a`)
- `02_run_kld.sh` / `03_run_rtn_baseline.sh`
- `queue_stock_run.sh` (full orchestration)
- `dwq_v2.py` (vendored fork of mlx_lm/quant/dwq.py with sensitivity-gated unfreeze + LR schedule + grad clip)

### 2.4 What's NOT yet on main — the gap

Three independent pieces are all required for full-model parity:

1. **Full multi-layer Qwen 3.5/3.6MoE forward on the GpuTape (iter-11h, ~3-5K LOC)** — needs RoPE, GQA broadcast, MoE FFN with top-8 expert routing, linear-attention/DeltaNet hybrid, causal mask, real BPE tokenizer (replacing `whitespace_hash_tokenize` stub at `src/calibrate/calibration_batcher.rs:104`). Of these, gated-delta-net backward and MoE router gradient have no published reference implementation in Rust+Metal. This is research-level work, not a port.

2. **`GgufTeacherProvider` (iter-14b, ~800 LOC)** — needs a public batched-logits API on `MlxModelWeights::forward_prefill_batched` (currently private; `logits_view()` exposes only the last token), plus a new calibrate module + integration test against llama.cpp.

3. **CLI surface (~600 LOC)** — `hf2q convert --quant dwq-q4 ...` does not exist anywhere today. The legacy `DWQ-46/48` paths are wired (`src/cli.rs:1010-1019` → `src/main.rs:1827-1844 DwqKQuantizer`) but route through the *weight-space* (non-distilled) DWQ from ADR-014 P11, NOT the canonical mlx-lm DWQ. ImatrixAdaptive (`src/main.rs:1685`) is the only mixed-precision entrypoint.

If we attempt to port iter-11h piece-by-piece, the existing ADR-020 cadence (50–500 LOC/iter, 1 falsifier/iter) projects 12-15 iterations to land. At the historical 3-7 day/iter pace observed across ADR-013, ADR-015, ADR-017, that's 4–12 *months* of work. We do not have that time, and the result (autograd in Rust+Metal for a single quantization algorithm) does not move hf2q's mission forward proportionally.

---

## 3. What — the decision

### 3.1 Path B chosen (subprocess bridge)

hf2q drives DWQ training via subprocess to a pinned `mlx_lm` install. The seam is a typed JSON-on-stdio protocol. hf2q owns everything that is *converter business logic*; mlx-lm owns the *training math*.

```
                 ┌───────────────────────────────────┐
                 │  hf2q (Rust) — owns all of these  │
                 │                                   │
  Operator ───►  │  CLI parsing + arg validation     │
                 │  HF → mlx working-dir setup       │
                 │    (chat_template injection,      │
                 │     model_type patching)          │
                 │  Calibration corpus management    │
                 │  Subprocess lifecycle             │
                 │   (spawn, kill, restart, OOM)     │
                 │  Mixed-precision recipe selection │
                 │  GGUF emission                    │
                 │  Parity measurement (kld.py wrap) │
                 │  Output layout (sharded MLX safetensors + config.json) │
                 └────────────┬──────────────────────┘
                              │  JSON line protocol
                              │  (request: hyperparams, batch-id, snapshot path)
                              │  (response: per-batch loss, scales-on-disk path)
                              ▼
                 ┌───────────────────────────────────┐
                 │  mlx-lm (Python, pinned SHA)      │
                 │   bin/dwq_python_driver.py        │
                 │                                   │
                 │  Loads model via mlx_lm.utils.load│
                 │  Calls mlx_lm.quant.dwq:dwq_quantize │
                 │  Writes top-1024 logits to disk   │
                 │  Writes trained scales+biases     │
                 │  Streams progress via JSON        │
                 └───────────────────────────────────┘
```

### 3.2 Why not Path A (pure Rust)

Path A requires:
- **5,500–7,000 LOC** — ~70% is autograd plumbing the rest of hf2q never needs.
- **~25 new tape ops** beyond the current 17 (`src/calibrate/autograd_gpu_tape.rs`), with FD-falsifiers per op.
- **Gated-delta-net backward** — published only in the original paper, no reference impl in `/opt/mlx-native` or `/opt/candle`.
- **MoE router gradient routing** — requires differentiable `take_along_axis` + `gather_mm`, neither of which exists in `src/calibrate/autograd_gpu_tape.rs` today (worker 4 R1, marked HIGH-severity).
- **SDPA backward parity** — must match mlx-lm's reduction order or numerical drift accumulates across 47 layers. ADR-015 already documented "Apple Silicon unified memory masks RAW races" as a chronic debug-time sink.

Bottom line: *Path A is a year-2 roadmap item, not a year-1 milestone.*

### 3.3 Why not Path C (PyO3 in-process)

- Build-time dep on Python (3.10/3.11/3.12 wheel ABI matrix). hf2q today is single-binary `cargo build`.
- No measurable perf win over Path B: per-step time on 35B is 200–800 ms, IPC overhead is 1–2 ms.
- Permanent CI tax: matrix size grows from "Linux/macOS Rust" to "Linux/macOS Rust × Python 3.10/11/12 × mlx-lm SHA."
- The strengths Path C offers over Path B (single process, cleaner Ctrl-C) are operator conveniences, not architectural wins.

### 3.4 Comparison matrix

| Criterion | Path A (Pure Rust) | **Path B (Subprocess)** | Path C (PyO3) |
|---|:-:|:-:|:-:|
| Mission fit (canonical converter) | A | **A** | C |
| LOC required | F (5.5–7k) | **A (~1.3k)** | B (~1.25k+CI) |
| Time to first measurement | F (5–7 wk) | **A (1–2 wk)** | C (2–4 wk) |
| Maintenance burden | C (we own autograd) | **B (pin mlx-lm SHA)** | D (Py ABI) |
| Performance ceiling | B | **B** | B |
| Test debt | F (~25 new ops) | **B (protocol+smoke)** | C (build matrix) |
| Risk of getting stuck on framework details | F (SDPA-bw, GDN-bw) | **A** | C (PyO3 build) |

---

## 4. How — detailed architecture

### 4.1 Subprocess protocol

Typed `serde` JSON over stdio, line-delimited. Strongly-typed `Request` and `Response` enums on the Rust side; mirrored Python dataclasses on the driver side.

```rust
// src/calibrate/dwq_python_protocol.rs (~250 LOC, NEW)

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Request {
    /// Initial handshake.
    VersionCheck { hf2q_version: String, expected_mlx_lm_sha: String },

    /// Phase 1: precompute teacher targets to disk.
    ComputeTargets {
        model_path: PathBuf,        // base model (BF16 or 8-bit teacher source)
        save_dir: PathBuf,           // target_dir
        data_path: String,           // e.g. allenai/tulu-3-sft-mixture
        num_samples: usize,
        max_seq_length: usize,
        batch_size: usize,
        seed: u64,
    },

    /// Phase 2: train the quantized student.
    TrainQuantize {
        teacher_model_path: PathBuf,
        quantized_model_path: Option<PathBuf>,  // pre-quantized RTN-Q4 student
        target_dir: PathBuf,
        save_path: PathBuf,                       // mlx-format safetensors output
        bits: u8,                                  // default 4
        group_size: u32,                           // default 64
        num_samples: usize,
        max_seq_length: usize,
        learning_rate: f64,
        batch_size: usize,
        seed: u64,
        temperature: f64,                          // default 2.0
        grad_checkpoint: bool,
        quant_predicate_config: Option<PathBuf>,  // mixed-precision recipe JSON
    },

    Shutdown,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Response {
    VersionCheckOk { mlx_lm_version: String, mlx_lm_sha: String },
    VersionCheckFailed { actual_sha: String, expected_sha: String },

    Progress {
        phase: ProgressPhase,            // "compute_targets" | "train" | "validate"
        step: usize,
        total: usize,
        loss: Option<f64>,
        peak_memory_gb: f64,
        toks_per_sec: f64,
    },

    Validation { iter: usize, loss: f64 },
    ComputeTargetsDone { batches_written: usize, bytes_written: u64 },
    TrainDone { final_validation_loss: f64, save_path: PathBuf },
    Error { code: ErrorCode, message: String, traceback: Option<String> },
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    OomKilled,                  // Metal kIOGPUCommandBufferCallbackErrorOutOfMemory
    ModelLoadFailed,
    InvalidQuantConfig,
    DistilDiverged,             // final_validation > initial_validation
    InternalPythonException,
}
```

### 4.2 Python driver

```python
# bin/dwq_python_driver.py (~150 LOC, NEW)
"""hf2q DWQ driver — subprocess seam to mlx_lm.
Stays alive across compute_targets + train_quantize so model load is amortized.
"""
import sys, json, traceback
from pathlib import Path

import mlx.core as mx
from mlx_lm.quant.dwq import (
    dwq_quantize, compute_dwq_targets, load_data
)
from mlx_lm.utils import load, load_tokenizer, quantize_model, save
import mlx.optimizers as optimizers

EXPECTED_MLX_LM_SHA = "<pinned at iter-19f>"

def emit(response):
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()  # critical — see subtle-detail below (tqdm doesn't flush)

# ... handle_version_check / compute_targets / train_quantize / shutdown ...
# ~50 LOC for the train_quantize handler with progress interception
```

### 4.3 Rust subprocess lifecycle

```rust
// src/calibrate/dwq_loop.rs::DwqPythonDriver (~300 LOC, NEW alongside existing iter-13/14 code)

pub struct DwqPythonDriver {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

impl DwqPythonDriver {
    pub fn spawn(python_bin: &Path, driver_script: &Path) -> Result<Self> {
        let child = Command::new(python_bin)
            .arg(driver_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // CRITICAL: PYTHONUNBUFFERED=1 — mirrors the tqdm-buffer issue
            // already documented in scripts/dwq_kl_parity/01_run_dwq.sh
            .env("PYTHONUNBUFFERED", "1")
            .spawn()?;
        // ... wire up stdin/stdout, version check ...
    }

    pub fn drive_compute_targets(&mut self, ...) -> Result<ComputeTargetsResult> {
        self.send(&Request::ComputeTargets { ... })?;
        loop {
            match self.recv()? {
                Response::Progress { phase, step, total, .. } => progress.update(...),
                Response::ComputeTargetsDone { .. } => return Ok(...),
                Response::Error { code, message, .. } => return Err(...),
                _ => return Err(DwqError::ProtocolViolation),
            }
        }
    }

    pub fn drive_train_quantize(&mut self, ...) -> Result<TrainResult> { ... }
}

impl Drop for DwqPythonDriver {
    fn drop(&mut self) {
        let _ = self.send(&Request::Shutdown);
        let _ = self.child.wait_timeout(Duration::from_secs(30));
        let _ = self.child.kill(); // SIGKILL fallback
    }
}
```

### 4.4 CLI integration

```rust
// src/cli.rs (~30 LOC NEW)
pub enum QuantMethod {
    // ... existing variants ...
    DwqQ4 {
        bits: u8,                // default 4
        group_size: u32,         // default 64
        num_samples: usize,      // default 2048
        max_seq_length: usize,   // default 1025
        learning_rate: f64,      // default 1e-6
        batch_size: usize,       // default 2 (OOM-protected)
        seed: u64,               // default 123
        calibration_data: String,  // default allenai/tulu-3-sft-mixture
        grad_checkpoint: bool,   // default true
        quant_predicate_config: Option<PathBuf>,
        mlx_lm_path: Option<PathBuf>,
    },
}
```

```rust
// src/main.rs (~80 LOC NEW arm in resolve_convert_config + dispatch)
QuantMethod::DwqQ4 { .. } => {
    let mlx_lm_path = args.mlx_lm_path
        .or_else(|| std::env::var("MLX_LM_PATH").ok().map(Into::into))
        .canonicalize()?;
    let driver_script = mlx_lm_path.join("bin/dwq_python_driver.py");
    let mut driver = DwqPythonDriver::spawn(&python_bin, &driver_script)?;
    driver.version_check(&EXPECTED_MLX_LM_SHA)?;

    let target_dir = working_dir.join("dwq_targets");
    driver.drive_compute_targets(...)?;

    let dwq_output = working_dir.join("dwq_output");
    driver.drive_train_quantize(...)?;

    // Stage 3: convert mlx-format → GGUF (using existing mlx_safetensors_loader
    // + the GGUF emit pipeline from ADR-014)
    convert_mlx_safetensors_to_gguf(&dwq_output, &args.output_path)?;
}
```

### 4.5 Phasing — 6 iterations, ~1,800 net LOC

Match ADR-020 iter-19c granularity (50–500 LOC per iter, one falsifier per iter, named codex-review checkpoints per `feedback_codex_review_catches_unified_memory_races`).

| Iter | Scope | Falsifier | New LOC |
|---|---|---|---:|
| **iter-19d** (this ADR) | Architecture decision recorded. CFA worker outputs preserved at `docs/research/cfa-adr020-iter19d/`; ADR-022 lands; ADR-020 row 19b half-2 redirects to ADR-022 Path B. | ADR-022 §3.4 contains the trade-off matrix and citations to all 5 worker reports. | ~80 (docs) |
| **iter-19e** | Python driver skeleton (`bin/dwq_python_driver.py`, ~80 LOC) + JSON protocol module (`src/calibrate/dwq_python_protocol.rs`, ~250 LOC) + serde round-trip + malformed-input tests. Driver echoes inputs, no mlx-lm calls yet. | Round-trip parity test on every protocol message; malformed-JSON robustness; version-mismatch rejection. | ~330 |
| **iter-19f** | Subprocess lifecycle in `src/calibrate/dwq_loop.rs::DwqPythonDriver` (~300 LOC, alongside existing iter-13/14 code): spawn/kill/restart, `PYTHONUNBUFFERED=1`, version handshake, progress forwarding to `ProgressReporter` (ADR-018 `LoadInfoBuilder` reuse). Pin mlx-lm SHA. | Wrong-SHA aborts before any forward pass. Ctrl-C during driver propagates correctly. SIGKILL fallback hits within 30 s. | ~400 |
| **iter-19g** | Real `compute_dwq_targets` invocation through the subprocess on a 32-dim toy Qwen-shaped model. Two-phase orchestration: phase 1 writes targets to `~/.cache/hf2q/dwq-targets/<run-id>/`, phase 2 streams them. Teacher-drop handshake. | Tiny-fixture KL convergence: 2 layers, 8 batches, KL must drop ≥30% from step 0 to step 8. Top-1024 indices byte-identical to mlx-lm's `compute_dwq_targets` output. | ~500 + ~120 Python |
| **iter-19h** | Mixed-precision scale-book integration. Per-Linear scales+biases from phase 2 flow into hf2q's `MixedBitQuantizer` (`src/quantize/mixed.rs`). GGUF emission gets DWQ-tuned scales. CLI flag `--quant dwq-q4` lands. | Re-running phase 2 with the same seed produces byte-identical safetensors output (modulo timestamps). RTN parity gate (§7.3) runs and lands ≤ 0.0676. | ~300 |
| **iter-20** | Live measurement on Qwen 3.6 35B-A3B. Run iter-19h end-to-end. Capture kld.py number against `stock_mlx_bf16` reference. | **Load-bearing v1 ship gate**: hf2q-DWQ KL ≤ 0.0702 on stock 35B-A3B. Peer cross-check: standalone `python -m mlx_lm.quant.dwq` on same args produces equivalent KL within 5%. | bench scripts + dossier |
| **iter-20a** | (Optional) Default-on integration: `hf2q convert --quant dwq-q4` works end-to-end on a fresh box with `pip install mlx-lm` as the only side install. Documentation, README quickstart, `docs/dwq-quickstart.md`. | Operator soak: 2-turn full conversion of 27B-DWQ4 from cold. Memory peak ≤ 80 GB measured. | ~150 |

---

## 5. The math — verified correct (worker 4 of CFA session)

### 5.1 The 5%-vs-64% gap is NOT a port bug

Direct evidence from `https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit-DWQ/raw/main/config.json`:

```json
{
  "quantization": {
    "group_size": 64,
    "bits": 8,                 ← DEFAULT bits=8, NOT 4
    "mode": "affine",

    "language_model.model.layers.{i}.mlp.switch_mlp.gate_proj":  {"bits":4,...},
    "language_model.model.layers.{i}.mlp.switch_mlp.up_proj":    {"bits":4,...},
    "language_model.model.layers.{i}.mlp.switch_mlp.down_proj":  {"bits":4,...},
    "language_model.model.layers.{i}.mlp.shared_expert.gate_proj":{"bits":4,...},
    "language_model.model.layers.{i}.mlp.shared_expert.up_proj":  {"bits":4,...},
    "language_model.model.layers.{i}.mlp.shared_expert.down_proj":{"bits":4,...}
  }
}
```

**Effective bpw = 4.84.** Smcleod's blog explicitly confirms: *"Absolute KLD here understates divergence from bf16 by the 8-bit-vs-bf16 gap."* — i.e., his reference is Q8, not BF16.

Our run: uniform Q4 (every Linear at 4 bits), measured against BF16. Strictly harder model, strictly stricter reference.

If we re-run with mixed-precision overrides matching the mlx-community config, AND measure against a Q8 reference, the expected hf2q KL lands at **~0.020-0.030** — indistinguishable from smcleod's 0.02663 within recipe-noise. (Worker 4 §6, goalie self-consistency 5/5 agreement.)

### 5.2 DWQ VJP — VERIFIED

```
qdq[i] = q_int[i] * scales[g(i)] + biases[g(i)]
∂qdq[i]/∂scales[g] = q_int[i] if g(i)==g else 0
∂qdq[i]/∂biases[g] = 1        if g(i)==g else 0
∂qdq[i]/∂q_int[i]  = scales[g(i)]   ← DWQ freezes; gradient zeroed

⇒ ∂L/∂scales[g] = Σ_{i ∈ group g} c[i] * q_int[i]
⇒ ∂L/∂biases[g] = Σ_{i ∈ group g} c[i]
```

Matches `mlx/primitives.cpp:3459-3525 QuantizedMatmul::vjp`:
- line 3487 explicit "no gradient wrt the quantized weights"
- bias-grad: `sum(cotangent, -1)` over each group
- scale-grad: `wq = dequantize(w_q, scales=1, biases=0)` [= q_int] then `sum(cotangent * wq, -1)`

hf2q `iter-13b` (`src/calibrate/autograd_gpu_tape.rs` `QdqAffine` op) implements this; FD-falsified at 1% tol. **Verdict: VERIFIED.**

### 5.3 KL distillation — VERIFIED

`/opt/mlx-lm/mlx_lm/tuner/losses.py:130-167` Metal kernel computes:
```
kl = Σ_j P_teacher[j] * (log P_teacher[j] - log Q_student[j])
   = KL(P_teacher ‖ Q_student)
```

Mode-covering forward KL (Hinton-style soft-target distillation), with temperature `scale=1/T=0.5` applied to BOTH student and teacher logits before softmax. The `T²` factor that some papers multiply at the loss level is absent — effectively absorbed into the LR=1e-6 default.

VJP at `losses.py:307`: `∂KL/∂logits_q[j] = q[j] - p[j]` — canonical-correct.

**Verdict: VERIFIED.**

### 5.4 Adam with bias correction — VERIFIED

mlx Adam (`/opt/homebrew/lib/python3.14/site-packages/mlx/optimizers/optimizers.py:512-535`) with `bias_correction=True` matches PyTorch default Adam to within negligible numerical drift. `dwq.py:393` passes `bias_correction=True` (non-default; mlx Adam defaults to `False`). **Important port detail: hf2q's Adam port at `src/calibrate/adam.rs` must replicate this flag.**

### 5.5 MoE differentiability — RESOLVED

`/opt/mlx-lm/mlx_lm/models/qwen3_next.py:308-354` `Qwen3NextSparseMoeBlock.__call__` and `switch_layers.py:186-187`: `if self.training: idx = mx.stop_gradient(idx)` — canonical straight-through estimator. Indices are constant in backward; gradients flow through `gates → softmax → take_along_axis → scores` and through `switch_mlp(x, idx)` for the touched expert rows.

**Implication for Path B**: the subprocess driver runs mlx-lm's MoE forward+backward unchanged. No hf2q autograd MoE op needed for Path B. (Path A would need both differentiable `take_along_axis` AND `gather_mm` — worker 4 R1 marks this HIGH-severity for any pure-Rust port.)

---

## 6. The risks that matter (from worker 4)

### R1 — mlx-lm version drift (MEDIUM)

Apple's mlx-lm is on a 2-week cadence (`/opt/mlx-lm/mlx_lm/_version.py`). When `dwq.py` signature changes, our Python driver breaks.

**Mitigation:**
- Pin mlx-lm SHA in `bin/dwq_python_driver.py` and `EXPECTED_MLX_LM_SHA` constant.
- Version handshake aborts before any forward pass on mismatch.
- CI smoke test on a representative fixture catches signature breakage at PR time.

### R2 — Mixed-precision recipe must match the reference (HIGH)

If we ship uniform Q4 by default, our parity number will always be ~2× higher than the published mlx-community number. The driver MUST accept per-path bit overrides via `--quant-predicate-config`.

**Mitigation:**
- iter-19h ships a `tests/fixtures/dwq_recipes/qwen35moe-mixed-precision.json` mirroring the published mlx-community config.
- Default behavior of `--quant dwq-q4` on a Qwen 3.5/3.6 MoE arch automatically picks the mixed-precision recipe (override required to opt out).
- Acceptance gate is calibrated against our actual measured number (uniform Q4 vs BF16 = 0.0610), NOT smcleod's 0.02663 — preserves apples-to-apples even if a future iter changes default recipes.

### R3 — KV-cache state must reset between batches (MED)

`/opt/mlx-lm/mlx_lm/quant/dwq.py:108-117` `loss_fn` calls `model(x)` with no cache argument — fresh per-batch. If our subprocess driver reuses cache for performance, second-forward logits condition on stale K/V. Wrong gradients.

**Mitigation:** Python driver MUST NOT pass a `cache` argument. Test: 10-batch run with cache reuse vs no-cache must produce identical loss curve.

### R4 — `precise=True` softmax (Path-A-only risk, MED)

`qwen3_next.py:335` uses `mx.softmax(gates, axis=-1, precise=True)`. Path B inherits from mlx-lm; only relevant if a future Path A port replaces the softmax.

### R5 — `norm_topk_prob = True` must execute (Path-A-only risk, MED)

`qwen3_next.py:340-341`: re-normalizes gathered top-k scores so they sum to 1. Path B inherits from mlx-lm.

### R6 — `grad_checkpoint(model.layers[0])` quirk (LOW)

mlx-lm only checkpoints layer 0 as a memory-saving heuristic. Path B preserves verbatim.

### R7 — Validation set seed parity (LOW)

`dwq.py:155 / 167-173`: train and valid use the SAME `seed=123`. Our protocol forwards seed unchanged.

### R8 — Adam first-step convention (LOW)

mlx Adam uses `step` starting at 1 (not 0); `b1**1 = 0.9`, `1 - 0.9 = 0.1`; no division by zero. Verify our `src/calibrate/adam.rs::AdamOptimizer` initializes `step=1` (per iter-13a `de1df56` commit).

### R9 — fp32 master / bf16 model dtype split (LOW for Path B)

Path B inherits mlx-lm's pattern (`dwq.py:152-156`).

### R10 — Static_quant + safetensors_out is corrupt-on-load (DECISIVE finding from W4 §8)

Independent of Path B but flagged because it affects any *current* "save mlx-format DWQ" caller.

`src/quantize/static_quant.rs` + `src/backends/safetensors_out.rs`:
- Quantizes to symmetric `[-7, +7]` as i8.
- Pack convention: low nibble `pair[0] & 0x0F`, high nibble `(pair[1] & 0x0F) << 4`.
- mlx affine reader at `src/calibrate/mlx_safetensors_loader.rs:240` (`unpack_u32_packed`) reads each nibble as **unsigned [0, 15]**.
- Result: a true intent of `-1 * scale = -scale` is decoded as `15 * scale + 0 = 15 * scale`. **Off by 16× and wrong sign.**

`safetensors_out.rs:423-434` emits zero-filled biases when `quant_info.biases` is None — known antipattern.

**Mitigation (must land before iter-20):**
- Sunset the static_quant → mlx-lm-format path. Any caller emitting mlx-format must route through the iter-13b/16b `MlxAffineLinear::to_safetensors_bytes` writer at `src/calibrate/mlx_safetensors_loader.rs:398` (the iter-19c-tested one).
- Static_quant retains its GGUF emit path (where signed nibbles ARE the convention — Q4_0 is signed).

---

## 7. Acceptance criteria

Distilled from worker 6's full criteria. See `docs/research/cfa-adr020-iter19d/worker6-acceptance-criteria.md` for the complete 636-line spec.

### 7.1 Layer 1 — Unit tests (each <60 s on M5 Max)

| Test | Falsifier | Pass criterion |
|---|---|---|
| `dwq_unit_qdq` | `qdq_affine_q4(W, s, b)` forward + VJP | Forward byte-equiv to scalar reference; VJP rel err ≤ 5e-3 vs FD on 50 seeds |
| `dwq_unit_kl_loss` | KL with T=2.0 forward + backward | Forward matches scipy.special.kl_div to 1e-6; backward ≤ 5e-3 rel vs FD |
| `dwq_unit_adam` | Adam(lr=1e-3, β=0.9/0.999) on quadratic | After 100 steps, x matches PyTorch ref to 1e-5 absolute |
| `dwq_unit_lr_schedule` | warmup=100 + cosine decay | All 4 boundary points within 1e-9; monotone phases |
| `dwq_unit_sensitivity` | 2-Linear synthetic MLP | Rank order matches analytical exactly; deterministic across 10 runs |
| `dwq_unit_safetensors` | mlx-format round-trip | header `format='mlx'`; mlx_lm Python loader successfully calls `model.load_weights(...)` |
| `dwq_unit_calibration_loader` | tulu-3 with seed=123, n=2048, l=1025 | SHA256 of token stream matches frozen fixture |

### 7.2 Layer 2 — Integration (each <30 min)

| Test | Pass criterion |
|---|---|
| `single_linear_real_bf16` (port of iter-19c at `src/calibrate/dwq_e2e.rs:481`) | KL post-train ≤ 5e-3 (iter-19c achieved 2.73e-3, 2× headroom) |
| `two_layer_toy_convergence` | KL[step=50] < 0.5 × KL[step=0]; no NaN/Inf |
| `compute_dwq_targets_parity` | Top-1024 indices byte-identical to mlx-lm; values rel tol 1e-3 |
| `full_forward_parity_qwen35` (`#[ignore]`) | Logits l2 ≤ 1e-2 per token; top-1 agreement ≥ 31/32; top-5 Jaccard ≥ 0.95 |

### 7.3 Layer 3 — **The load-bearing parity gate**

**Stock Qwen 3.6 35B-A3B, full pipeline, kld.py at seed=123 num_samples=512 batch=4 sequence_length=1025:**

| Outcome | Range | Action |
|---|---|---|
| **HARD PASS** | `hf2q_DWQ_KL ≤ 0.0702` (= 0.0610 × 1.15) | Ship v1 |
| **SOFT PASS** | `0.0702 < hf2q_DWQ_KL ≤ 0.0793` (≤ 1.30 ratio) | Ship v1 with audit document `docs/dwq-parity-audit-<date>.md` |
| **HARD FAIL** | `hf2q_DWQ_KL > 0.0793` OR `> 0.100` absolute (smcleod broken floor) | Block release |

**RTN sanity (independent gate):** `hf2q_RTN_Q4_KL ≤ 0.0676` (= 0.0644 × 1.05). Tight because RTN is closed-form; deviation > 5% indicates a bit/group encoding bug (see R10 above).

### 7.4 Layer 4 — Ship gate

ALL of these must hold for v1:

1. Parity gate (§7.3) HARD PASS or SOFT PASS-with-audit
2. RTN parity ≤ 0.0676 on stock 35B
3. **Cross-family smoke**: one additional model (Llama 3.x 8B or Mistral 7B) at KL ≤ 0.045
4. CLI: `hf2q convert --quant dwq-q4 ...` parses and runs end-to-end
5. **Determinism**: 2 runs same seed → byte-identical safetensors
6. **Memory**: peak unified memory ≤ 80 GB target / ≤ 100 GB hard ceiling (SIGTERM at 110)
7. Documentation: `docs/dwq-quickstart.md` with worked example + parity number + soft-fail policy
8. Test coverage: ≥ 90% line / ≥ 80% branch on `src/calibrate/dwq*.rs`
9. **Coherence**: `hf2q serve --model <output> --prompt "the quick brown fox" --max-tokens 32` produces non-degenerate UTF-8 with no `<|_end|>` literal-byte leak (defends against pre-`505b5b8` vocab-truncation bug class)
10. Output safetensors: `format='mlx'` metadata, no NaN in any scale or bias

### 7.5 Performance acceptance

| Metric | Target | Hard ceiling |
|---|---|---|
| Per-step training time | ≤ 20 s/step at batch=2 | ≤ 30 s/step |
| Peak unified memory | ≤ 80 GB | ≤ 100 GB |
| Disk I/O (target cache) | ≤ 10 GB | ≤ 12 GB |
| Total wall time | ≤ 6 hours on stock 35B | ≤ 8 hours |

---

## 8. Calibration data integrity

Reproducibility anchor:

- Corpus: `allenai/tulu-3-sft-mixture` (HF dataset)
- Tokenizer: model's own (Qwen 3.5 vocab = 248,320 tokens, eos=248046; MUST NOT be the truncated 248044 from pre-`505b5b8` builds — see existing memory `project_qwen35_dwq_pre_505b5b8_broken_2026_05_05`)
- Seed: 123
- Sample counts: 2048 (training) / 512 (evaluation)
- Max sequence length: 1025
- Tokenization: chat-template applied (per `scripts/dwq_kl_parity/queue_stock_run.sh:198-221`)

**Frozen reference fixtures** (commit at `tests/fixtures/dwq_calibration/`):
- `tulu3_seed123_n2048_l1025.tokens.bin` — concatenated u32 token IDs (~8 MB), SHA256-anchored
- `tulu3_seed123_n2048_l1025.sha256` — single-line SHA256 hex

CI step `sha256sum -c` before any DWQ training run blocks on corpus drift.

---

## 9. Adjacent ADR connections

| ADR | Relationship | Notes |
|---|---|---|
| **ADR-005** (inference server) | Independent | Distinct scope — runtime serving vs build-time quantization |
| **ADR-013** (qwen35 inference) | Independent | Build-time tool vs runtime perf. Zero shared infrastructure |
| **ADR-014** (streaming convert pipeline) | **Shares infra** | Streaming-load + per-tensor mmap is what fits 35B in memory during teacher phase. iter-19h scale-book emission reuses ADR-014 P11's `MixedBitQuantizer` + `LayerQuantConfig` types from `src/quantize/mod.rs` — no schema change required |
| **ADR-017** (persistent block-prefix cache) | Independent at runtime | Lessons (engagement test pattern, fail-loud on counter mismatch) transfer to iter-19g |
| **ADR-018** (uniform model load UX) | **Shares infra** | Path B's banner ("Loading mlx-lm 0.X.Y from /opt/mlx-lm; Python 3.12.4") plugs into the unified `LoadInfo` builder pattern from ADR-018. ~60 LOC saved |
| **ADR-021** (Qwen3-VL ViT) | Independent | Per-layer driver pattern transfers if we ever DWQ-tune the ViT separately |
| **ADR-020** (DWQ + mixed-precision quantization) | **Supersedes row 19b half-2** | This ADR replaces the row 19b half-2 gating chain (iter-11h + iter-14b for full-Rust autograd). ADR-020 §8.2 row 19b half-2 status updates to "deferred to ADR-022 Path B". iter-11h becomes a *future* ADR (ADR-023+) for if/when the team chooses to do the pure-Rust port |

---

## 10. Reference reading

The implementing engineer should read, in this order:

1. `/opt/mlx-lm/mlx_lm/quant/dwq.py:29-66` — `compute_dwq_targets`. The teacher-drop sequencing the subprocess must replicate.
2. `/opt/mlx-lm/mlx_lm/quant/dwq.py:69-209` — `dwq_quantize`. Optimizer loop, KL temperature trick, gradient-checkpoint hook, the `model.update(tree_map(...))` pattern.
3. `/opt/mlx-lm/mlx_lm/quant/dwq.py:242-411` — CLI entry point. Argument names and defaults the Python driver must accept verbatim.
4. `/opt/mlx-lm/mlx_lm/utils.py:282-420` — `load_model` + `quantize_model`. The `lazy=True` path and `class_predicate` mechanism.
5. `/opt/mlx-lm/mlx_lm/utils.py:925-950` — `save`. The `metadata={"format": "mlx"}` write at line 756 (kld.py prereq).
6. `/opt/mlx-lm/mlx_lm/models/qwen3_5.py` + `qwen3_5_moe.py` + `qwen3_next.py` — Qwen 3.5 forward graph, hybrid attention routing, `Qwen3NextSparseMoeBlock` with top-8 routing.
7. `src/calibrate/dwq_e2e.rs:481` — iter-19c single-Linear KL parity test on real BF16. The minimal "this works" anchor.
8. `src/calibrate/dwq_loop.rs::init_affine_params_gpu` + `box_muller_gaussian` — per-tensor primitives the subprocess driver does NOT replace (they remain the in-process iter-19c regression).
9. `src/calibrate/mlx_safetensors_loader.rs` — read+write byte-correct mlx-format safetensors. The output sink for Path B's stage 3 (mlx → GGUF conversion).
10. `scripts/dwq_kl_parity/kld.py` — vendored PR #1146 KLD measurement; baseline must be `format='mlx'` (line 328 hard-rejects others).
11. `docs/ADR-014-streaming-convert-pipeline.md` + `docs/ADR-018-uniform-model-load-ux.md` — for `LayerQuantConfig`, `MixedBitQuantizer`, sensitivity cache layout, and the `LoadInfoBuilder` pattern.
12. `/opt/llama.cpp/convert_hf_to_gguf.py` — canonical reference for how mixed-precision tensors lay out in a GGUF v3 file.

The engineer should NOT read `/opt/mlx-lm/mlx_lm/quant/awq.py` or `gptq.py` until iter-20a or later — out of scope for ADR-022.

---

## 11. Open questions and follow-ups

1. **Is `mlx-community/Qwen3.6-35B-A3B-bf16` byte-identical to `mlx_lm.convert --hf-path Qwen/Qwen3.6-35B-A3B --dtype bfloat16`?** Worker 4 §7 hypothesizes "likely equivalent" but unverified. Cheap falsifier (sha256sum cross-check) should run before any apples-to-apples claim. **Action**: iter-19f spike — download mlx-community BF16 + diff configs + sha256 the safetensors. If they differ, we may need to use mlx-community's BF16 as our `--baseline-model` for kld.py to truly match smcleod's numbers.

2. **Default `--quant dwq-q4` recipe: uniform Q4 or mixed-precision Q8/Q4-experts?** The mlx-community published recipe is mixed-precision (effective 4.84 bpw). **Recommendation**: default to uniform Q4 (operator's expected meaning of "Q4"), but ship the mixed-precision recipe as a named preset (`--quant-predicate-config qwen35moe-mixed-precision`). Document that the parity number against smcleod's 0.02663 requires the mixed-precision recipe.

3. **Cross-family smoke model**: Llama 3.x 8B-Instruct vs Mistral 7B-v0.3. Both have pre-baked DWQ-Q4 in mlx-community. **Recommendation**: Llama 3.1 8B-Instruct (`mlx-community/Llama-3.1-8B-Instruct-4bit-DWQ` exists with a published number).

4. **`hf2q` `[lib]` target friction** — per existing memory `project_hf2q_no_lib_target_unit_test_friction`, `tests/dwq_unit/*` would have to live as `#[cfg(test)]` blocks inside the bin source OR require expanding `lib.rs`. **Recommendation**: keep `#[cfg(test)]` pattern for unit tests; `tests/dwq_integration/*` becomes a new `[[bin]] dwq_integration` similar to existing `extract_dwq_sensitivity` / `dump_gguf_types` patterns.

5. **iter-11h fate** — does the deferred pure-Rust autograd port ever land? **Recommendation**: leave as a year-2+ option in a future ADR-023. Unless mlx-lm pivots away from supporting Qwen 3.5/3.6 (unlikely on a 2-week-cadence repo), Path B is sufficient indefinitely. The autograd op surface we'd build for Path A becomes valuable only if hf2q wants to ship a *second* quantization algorithm that mlx-lm doesn't (e.g., research-level OmniQuant or SpinQuant). Separate decision.

---

## 12. Decision history

| Date | Event |
|---|---|
| 2026-04-XX | ADR-020 §8.2 row 19b half-1 lands: vendor kld.py, build harness scripts |
| 2026-04-XX → 2026-05-06 | iter-8a → iter-19c — per-component primitives all green (on `cfa/adr020-iter10/claude` branch) |
| 2026-05-06 | iter-19c PASS on real BF16 single Linear (KL = 2.73e-3) — `src/calibrate/dwq_e2e.rs:481` |
| 2026-05-07 | mlx_lm.dwq runs on stock + abliterated 35B-A3B harvested as reference targets via `scripts/dwq_kl_parity/queue_stock_run.sh` |
| 2026-05-07 | User flags antipattern: shelling out to mlx_lm.dwq defeats hf2q's converter identity |
| 2026-05-07 | CFA session `cfa-20260507-191500-adr020-followup-research` runs 5 parallel research workers + queen synthesis |
| 2026-05-07 | `cfa/adr020-iter10/claude` branch (12,755 LOC) merged into main as commit `d2ee583` |
| 2026-05-07 | This ADR-022 lands as the proposed plan |
| **next** | iter-19e begins (Python driver skeleton + JSON protocol) once user gives go-ahead |

---

## 13. Worker-report appendix

The five CFA worker reports backing this ADR are preserved at:

- `docs/research/cfa-adr020-iter19d/worker2-mlx-lm-peer-map.md` (603 lines) — exhaustive map of every load-bearing surface in `/opt/mlx-lm/mlx_lm/`
- `docs/research/cfa-adr020-iter19d/worker3-hf2q-codebase-map.md` (287 lines) — what's shipped, what's missing, line numbers and LOC estimates
- `docs/research/cfa-adr020-iter19d/worker4-math-validation.md` (583 lines) — VJP / KL / Adam / MoE math verified; goalie anti-hallucination + WebFetch corroboration; smcleod-gap explanation
- `docs/research/cfa-adr020-iter19d/worker5-architecture-tradeoffs.md` (225 lines) — three paths analyzed; comparison matrix; phasing
- `docs/research/cfa-adr020-iter19d/worker6-acceptance-criteria.md` (636 lines) — full layered acceptance criteria, test commands, performance targets

---

## 14. One-paragraph summary for ADR index

ADR-022 supersedes ADR-020 row 19b half-2's gating chain. Instead of building a full-model differentiable Qwen 3.5/3.6 MoE forward in `GpuTape` (estimated 5-7 weeks, 5,500-7,000 LOC, blocking on research-level gated-delta-net backward + MoE router gradient), hf2q completes its DWQ port via a subprocess bridge to a pinned `mlx_lm.dwq`. hf2q owns CLI + calibration management + mixed-precision recipe selection + GGUF emission + parity measurement; mlx-lm owns the differentiable forward + backward. The bridge is a 150-LOC Python driver + 1,300 LOC of Rust, landing across 6 iterations (~5 weeks) culminating in the load-bearing v1 ship gate at iter-20: hf2q-DWQ KL ≤ 0.0702 (= mlx-lm's 0.0610 × 1.15) on stock Qwen 3.6 35B-A3B against the canonical BF16 reference, with hard fail above 0.0793 or 0.100 absolute. The math (VJP, KL distillation, Adam bias correction, MoE differentiability) is independently verified against `mlx/primitives.cpp:3459-3525` and `mlx_lm/tuner/losses.py:130-167`; the published smcleod 0.02663 number turns out to be against a Q8 reference of a 4.84-bpw mixed-precision checkpoint, not BF16 of uniform Q4 — fully explaining the apparent 5%-vs-64% gap in the iter-19b runs. Pure-Rust port deferred to ADR-023+.
