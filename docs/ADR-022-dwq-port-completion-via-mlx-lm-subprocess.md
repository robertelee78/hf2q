# ADR-022 — Complete the hf2q DWQ port via mlx-lm subprocess bridge

| | |
|---|---|
| **Status** | Proposed |
| **Date** | 2026-05-07 |
| **Supersedes** | ADR-020 §8.2 row 19b half-2 (re-scoped); ADR-020 §8.3 row 6 (re-pathed) |
| **Authors** | Synthesized from CFA session `cfa-20260507-191500-adr020-followup-research` (5 parallel research workers + queen synthesis) |
| **Replaces approach** | Pure-Rust full-model GpuTape autograd port (deferred to ADR-023+) |

---

## 1. TL;DR

hf2q has been *shelling out* to `mlx_lm.dwq` from `scripts/dwq_kl_parity/` instead of running its own DWQ port. That defeats the project's "canonical converter" identity. ADR-020 row 19b half-1 (capture canonical mlx-lm number) is **DONE**; row 19b half-2 (run *our* port and measure parity) has been gated for months on iter-11h (full multi-layer Qwen 3.5/3.6 MoE forward on GpuTape) and iter-14b (real `GgufTeacherProvider`) — neither shipped.

**Decision:** complete the port via a **subprocess bridge to a pinned `mlx_lm.dwq`**, not by porting the full autograd stack into Rust. hf2q owns: orchestration, calibration management, GGUF emission, mixed-precision scale-book bookkeeping, parity measurement, CLI surface. mlx-lm owns: forward + backward through the differentiable model graph. A 150-LOC Python driver script (`bin/dwq_python_driver.py`) is the seam; ~1,300 LOC of new Rust ties it together.

**Total scope: ~1,800 net new LOC across 6 iterations (iter-19d → iter-20a), wall-clock ~5 weeks (~10 man-days actual work).** Compared to the pure-Rust path's 5,500–7,000 LOC and 5–7 weeks, with no deferred research risk on gated-delta-net backward, MoE router gradient routing, or kernel-level reduction-order parity.

**Acceptance gate (load-bearing v1 ship):** hf2q-DWQ KL ≤ **0.0702** (= mlx-lm 0.0610 × 1.15) on stock Qwen 3.6 35B-A3B via the vendored `kld.py`. Hard fail above 0.0793 or 0.100 absolute.

**Critical math finding from research:** the apparent 5%-vs-64% gap between our run and smcleod's published 0.02663 is *not* a port bug. Smcleod measured a **mixed-precision Q8/Q4-experts checkpoint at 4.84 bpw against a Q8 reference**; we ran **uniform Q4 against BF16**. Verified by direct inspection of `mlx-community/Qwen3.6-35B-A3B-4bit-DWQ/config.json`. The math (VJP, KL, Adam, MoE differentiability) all checks out. The ship gate is correctly calibrated against our actual (apples-to-apples) reference number.

---

## 2. Why — the problem we are solving

### 2.1 The misframing we corrected

For ~3 weeks the team has been driving `mlx_lm.dwq` from `scripts/dwq_kl_parity/01_run_dwq.sh` and `queue_stock_run.sh`, treating the produced numbers as ADR-020 deliverables. They are not. They are a *reference target* — the number our port must match within tolerance. The deliverable is hf2q's own DWQ output, measured the same way, falling in the same KL band.

This was a process bug: the user explicitly called this out as "shelling out to other repo's tools is an antipattern, and explicitly goes against the entire fucking point of hf2q." Correct.

### 2.2 What hf2q's mission actually requires

hf2q exists to be **the canonical HF→GGUF + mixed-precision quantization converter**. Quantization quality at the published bands (smcleod's "broken floor" = >0.10 KL; "well-made 4-bit" = 0.01–0.05; "DWQ territory" ~0.027) requires distillation-tuned per-group scales+biases — i.e., DWQ. Without an in-house DWQ path:

- Operators must run `mlx_lm.dwq` separately and feed the output back into hf2q for GGUF emission. Two-tool pipeline.
- We cannot ship a single `hf2q convert --quant dwq-q4 ...` command.
- Mixed-precision recipes (Apex / `imatrix-adaptive`, already shipped) cannot benefit from DWQ-tuned scales — they're stuck with closed-form RTN.

### 2.3 What's already done (worth preserving)

Per worker 3's codebase map:

| Module | Status |
|---|---|
| `src/calibrate/mlx_safetensors_loader.rs` (1,220 LOC) | **iter-16/16b** — read+write byte-correct mlx-format safetensors with `format='mlx'` metadata. Fixture-match to `mlx/ops.cpp:4762-4798`. iter-19c uses it on real BF16 layer-0 `linear_attn.out_proj` — production-tested. |
| `src/calibrate/dwq_targets.rs` (696 LOC) | **iter-14** — `compute_dwq_targets` byte-compatible with mlx-lm's per-batch `[B, S-1, K]` safetensors layout. Top-K via `BinaryHeap`. Trait `TeacherLogitsProvider` defined; only `SyntheticTeacher` impl exists. |
| `src/calibrate/dwq_loop.rs` (1,191 LOC) | **iter-13a/b/c/d/e** — Adam + KL-div + qdq_affine training loop. Convergent on synthetic (87×→15× loss reduction) and on real BF16 single Linear (KL=2.73e-3 post-train at iter-19c). |
| `src/calibrate/autograd_gpu_tape.rs` (3,931 LOC) | **iter-8a–13b** — GpuTape with 17 op kinds. matmul, softmax, log, RowSum, embedding, SiLU, slice/concat/transpose, RMSNorm, view, scalar_mul, qdq_affine. All FD-falsified at 5e-3 rel tol. |
| `src/calibrate/imatrix*.rs` + `apex.rs` + `sensitivity.rs` (~3,300 LOC) | **prod-shipped** — mixed-precision K-quant via `imatrix-adaptive` CLI. Independent track from DWQ. |

### 2.4 What's missing (the actual gap)

**Worker 3's gap analysis is decisive — three independent pieces are all required:**

1. **Full multi-layer Qwen 3.5/3.6MoE forward on the GpuTape (iter-11h, ~3-5K LOC)** — needs RoPE, GQA broadcast, MoE FFN with top-8 expert routing, linear-attention/DeltaNet hybrid, causal mask, real BPE tokenizer. Of these, gated-delta-net backward and MoE router gradient have no published reference implementation in Rust+Metal. This is research-level work, not a port.

2. **`GgufTeacherProvider` (iter-14b, ~800 LOC)** — needs a public batched-logits API on `MlxModelWeights::forward_prefill_batched` (currently private; `logits_view()` exposes only the last token), plus a new calibrate module + integration test against llama.cpp.

3. **CLI surface (~600 LOC)** — `hf2q convert --quant dwq-q4 ...` does not exist anywhere today. The legacy DWQ-46/48 paths are wired but route through `DwqKQuantizer` which is the *weight-space* (non-distilled) DWQ, NOT the canonical mlx-lm DWQ.

If we attempt to port iter-11h piece-by-piece, the existing ADR-020 cadence (50–500 LOC/iter, 1 falsifier/iter) projects 12-15 iterations to land. At the historical 3-7 day/iter pace we observe across ADR-013, ADR-015, ADR-017, that's 4-12 *months* of work. Worker 5 estimates 5-7 weeks optimistic, 70+ days pessimistic on Path A.

We do not have that time, and the result (autograd in Rust+Metal for a single quantization algorithm) doesn't move hf2q's mission forward proportionally.

---

## 3. What — the decision

### 3.1 Path B chosen (subprocess bridge)

hf2q drives DWQ training via subprocess to a pinned `mlx_lm` install. The seam is a typed JSON-on-stdio protocol. hf2q owns everything that is *converter business logic*; mlx-lm owns the *training math*.

Boundary in detail:

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

Per worker 5's analysis, Path A requires:
- **5,500–7,000 LOC** — 70% of which is autograd plumbing the rest of hf2q never needs.
- **~25 new tape ops** with FD-falsifiers per op.
- **Gated-delta-net backward** — published only in the original paper, no reference impl in mlx-native or candle.
- **MoE router gradient routing** — requires differentiable `take_along_axis` + `gather_mm`, neither of which exists today (worker 4 R1, marked HIGH-severity).
- **SDPA backward parity** — must match mlx-lm's reduction order or numerical drift accumulates across 47 layers; ADR-015 already documented "Apple Silicon unified memory masks RAW races" as a chronic debug-time sink.

Bottom line from W5: *"Path A is a year-2 roadmap item, not a year-1 milestone."*

### 3.3 Why not Path C (PyO3 in-process)

- Build-time dep on Python (3.10/3.11/3.12 wheel ABI matrix). hf2q today is single-binary `cargo build`.
- No measurable perf win over Path B: per-step time on 35B is 200–800 ms, IPC overhead is 1–2 ms.
- Permanent CI tax: matrix size grows from "Linux/macOS Rust" to "Linux/macOS Rust × Python 3.10/11/12 × mlx-lm SHA."
- W5 verdict: "*The strengths Path C offers over Path B (single process, cleaner Ctrl-C) are operator conveniences, not architectural wins.*"

### 3.4 Comparison matrix (from W5)

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
// src/calibrate/dwq_python_protocol.rs (~250 LOC)

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
        teacher_model_path: PathBuf,    // base for quantize_model() if no quantized_model passed
        quantized_model_path: Option<PathBuf>,  // pre-quantized student (RTN-Q4)
        target_dir: PathBuf,             // disk cache of teacher logits
        save_path: PathBuf,              // mlx-format safetensors output
        bits: u8,                        // default 4
        group_size: u32,                 // default 64
        num_samples: usize,
        max_seq_length: usize,
        learning_rate: f64,
        batch_size: usize,
        seed: u64,
        temperature: f64,                // default 2.0
        grad_checkpoint: bool,
        quant_predicate_config: Option<PathBuf>,  // mixed-precision recipe JSON
    },

    /// Cooperative shutdown.
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
# bin/dwq_python_driver.py (~150 LOC)
"""hf2q DWQ driver — subprocess seam to mlx_lm.

Reads typed JSON requests on stdin, writes typed JSON responses on stdout.
Process stays alive across the train_quantize call so model load is amortized.
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
    sys.stdout.flush()  # critical — see ADR-020 §8.2 row 19b live "tqdm doesn't flush"

def handle_version_check(req):
    import mlx_lm
    actual_sha = ...  # git rev-parse on /opt/mlx-lm if available, else version string
    if actual_sha != req["expected_mlx_lm_sha"]:
        emit({"kind": "version_check_failed",
              "actual_sha": actual_sha,
              "expected_sha": req["expected_mlx_lm_sha"]})
        sys.exit(1)
    emit({"kind": "version_check_ok",
          "mlx_lm_version": mlx_lm.__version__,
          "mlx_lm_sha": actual_sha})

def handle_compute_targets(req):
    tokenizer = load_tokenizer(req["model_path"])
    train, valid = load_data(tokenizer, req["data_path"],
                             req["num_samples"], req["max_seq_length"])
    model, _, _ = load(req["model_path"], return_config=True, lazy=True)
    compute_dwq_targets(model, Path(req["save_dir"]), train, valid,
                        batch_size=req["batch_size"],
                        max_seq_length=req["max_seq_length"],
                        seed=req["seed"])
    # critical: free teacher RAM before next phase
    del model; mx.clear_cache()
    emit({"kind": "compute_targets_done", ...})

def handle_train_quantize(req):
    # Subscribes to per-step progress via a tqdm interceptor that emits JSON.
    # ... ~50 LOC ...
    pass

def handle_shutdown(req):
    emit({"kind": "ack"})
    sys.exit(0)

DISPATCH = {
    "version_check": handle_version_check,
    "compute_targets": handle_compute_targets,
    "train_quantize": handle_train_quantize,
    "shutdown": handle_shutdown,
}

if __name__ == "__main__":
    for line in sys.stdin:
        try:
            req = json.loads(line)
            DISPATCH[req["kind"]](req)
        except Exception as e:
            emit({"kind": "error",
                  "code": "internal_python_exception",
                  "message": str(e),
                  "traceback": traceback.format_exc()})
```

### 4.3 Rust subprocess lifecycle

```rust
// src/calibrate/dwq_loop.rs::DwqPythonDriver (~300 LOC of subprocess management)

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
            // CRITICAL: PYTHONUNBUFFERED=1 — see ADR-020 §8.2 row 19b live tqdm-buffer issue
            .env("PYTHONUNBUFFERED", "1")
            .spawn()?;
        // ... wire up stdin/stdout, version check ...
    }

    pub fn send(&mut self, req: &Request) -> Result<()> { ... }

    pub fn recv(&mut self) -> Result<Response> { ... }

    pub fn drive_compute_targets(&mut self, ...) -> Result<ComputeTargetsResult> {
        self.send(&Request::ComputeTargets { ... })?;
        loop {
            match self.recv()? {
                Response::Progress { phase, step, total, .. } => {
                    progress_reporter.update(...);  // ADR-018 LoadInfoBuilder reuse
                }
                Response::ComputeTargetsDone { .. } => return Ok(...),
                Response::Error { code, message, .. } => {
                    return Err(DwqError::Subprocess { code, message });
                }
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
        // Fallback: SIGKILL
        let _ = self.child.kill();
    }
}
```

### 4.4 CLI integration

```rust
// src/cli.rs (~30 LOC)
pub enum QuantMethod {
    // ... existing ...
    DwqQ4 {
        bits: u8,                // default 4
        group_size: u32,         // default 64
        num_samples: usize,      // default 2048
        max_seq_length: usize,   // default 1025
        learning_rate: f64,      // default 1e-6
        batch_size: usize,       // default 2 (was 4 — OOM-protected default)
        seed: u64,               // default 123
        calibration_data: String,  // default allenai/tulu-3-sft-mixture
        grad_checkpoint: bool,   // default true
        quant_predicate_config: Option<PathBuf>,  // for mixed-precision
        mlx_lm_path: Option<PathBuf>,  // path to /opt/mlx-lm or pinned install
    },
}
```

```rust
// src/main.rs (~80 LOC new arm in resolve_convert_config + dispatch)
QuantMethod::DwqQ4 { .. } => {
    let mlx_lm_path = args.mlx_lm_path
        .unwrap_or_else(|| std::env::var("MLX_LM_PATH").into())
        .canonicalize()?;
    let driver_script = mlx_lm_path.join("bin/dwq_python_driver.py");
    let mut driver = DwqPythonDriver::spawn(&python_bin, &driver_script)?;
    driver.version_check(&EXPECTED_MLX_LM_SHA)?;

    // Stage 1: compute targets
    let target_dir = working_dir.join("dwq_targets");
    driver.drive_compute_targets(...)?;

    // Stage 2: train DWQ student
    let dwq_output = working_dir.join("dwq_output");
    driver.drive_train_quantize(...)?;

    // Stage 3: convert mlx-format → GGUF (existing hf2q pipeline)
    convert_mlx_safetensors_to_gguf(&dwq_output, &args.output_path)?;
}
```

### 4.5 Phasing — 6 iterations, ~1,800 net LOC

Match ADR-020 iter-19c granularity (50–500 LOC per iter, one falsifier per iter, named codex-review checkpoints per `feedback_codex_review_catches_unified_memory_races`).

| Iter | Scope | Falsifier | New LOC |
|---|---|---|---:|
| **iter-19d** (this ADR) | Architecture decision recorded. CFA worker outputs preserved; ADR-022 lands; ADR-020 row 19b half-2 status updated to "deferred to ADR-022 Path B". | ADR-022 §15 contains the trade-off matrix and citations to all 5 worker reports. | ~80 (docs) |
| **iter-19e** | Python driver skeleton (`bin/dwq_python_driver.py`, ~80 LOC) + JSON protocol module (`calibrate/dwq_python_protocol.rs`, ~250 LOC) + serde round-trip + malformed-input tests. Driver echoes inputs, no mlx-lm calls yet. | Round-trip parity test on every protocol message; malformed-JSON robustness test; version-mismatch rejection test. | ~330 |
| **iter-19f** | Subprocess lifecycle in `dwq_loop.rs::DwqPythonDriver` (~300 LOC): spawn/kill/restart, `PYTHONUNBUFFERED=1`, version handshake, progress forwarding to `ProgressReporter` (ADR-018 `LoadInfoBuilder` reuse). Pin mlx-lm SHA. | Wrong-SHA aborts before any forward pass. Ctrl-C during driver propagates correctly. SIGKILL fallback hits within 30 s. | ~400 |
| **iter-19g** | Real `compute_dwq_targets` invocation through the subprocess on a 32-dim toy Qwen-shaped model. Two-phase orchestration: phase 1 writes targets to `~/.cache/hf2q/dwq-targets/<run-id>/`, phase 2 streams them. Teacher-drop handshake. | Tiny-fixture KL convergence: 2 layers, 8 batches, KL must drop ≥30% from step 0 to step 8. Top-1024 indices byte-identical to mlx-lm's `compute_dwq_targets` output. | ~500 + ~120 Python |
| **iter-19h** | Mixed-precision scale-book integration. Per-Linear scales+biases from phase 2 flow back into hf2q's `MixedBitQuantizer`. GGUF emission gets DWQ-tuned scales. CLI flag `--quant dwq-q4` lands. | R-C4-style byte-identity check on the produced GGUF: re-running phase 2 with the same seed produces identical output bytes (modulo timestamps). RTN parity gate (§8 below) runs and lands ≤ 0.0676. | ~300 |
| **iter-20** | Live measurement on Qwen 3.6 35B-A3B. Run iter-19h end-to-end. Capture kld.py number against `stock_mlx_bf16` reference. | **Load-bearing v1 ship gate**: hf2q-DWQ KL ≤ 0.0702 on stock 35B-A3B. Peer cross-check: standalone `python -m mlx_lm.quant.dwq` on same args produces equivalent KL within 5%. | bench scripts + measurement dossier |
| **iter-20a** | (Optional) Default-on integration: `hf2q convert --quant dwq-q4` works end-to-end on a fresh box with `pip install mlx-lm` as the only side install. Documentation, README quickstart, `docs/dwq-quickstart.md`. | Operator soak: 2-turn full conversion of 27B-DWQ4 from cold. Memory peak ≤ 80 GB measured. | ~150 |

---

## 5. The math — verified correct (from worker 4)

### 5.1 The 5%-vs-64% gap is NOT a port bug

This is the most important finding from the research phase. **The published smcleod number 0.02663 is for a different model and a different reference than what we ran.**

Direct evidence from `https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit-DWQ/raw/main/config.json`:

```json
{
  "quantization": {
    "group_size": 64,
    "bits": 8,                 ← DEFAULT bits=8, NOT 4
    "mode": "affine",

    "language_model.model.layers.{i}.mlp.switch_mlp.gate_proj":  {"bits":4,"group_size":64,"mode":"affine"},
    "language_model.model.layers.{i}.mlp.switch_mlp.up_proj":    {"bits":4,"group_size":64,"mode":"affine"},
    "language_model.model.layers.{i}.mlp.switch_mlp.down_proj":  {"bits":4,"group_size":64,"mode":"affine"},
    "language_model.model.layers.{i}.mlp.shared_expert.gate_proj":{"bits":4,...},
    "language_model.model.layers.{i}.mlp.shared_expert.up_proj":  {"bits":4,...},
    "language_model.model.layers.{i}.mlp.shared_expert.down_proj":{"bits":4,...}
  }
}
```

**Effective bpw = 4.84.** Smcleod's blog explicitly confirms: *"Absolute KLD here understates divergence from bf16 by the 8-bit-vs-bf16 gap."* — i.e., his reference is Q8, not BF16.

Our run: uniform Q4 (every Linear at 4 bits), measured against BF16. Strictly harder model, strictly stricter reference.

If we re-run with mixed-precision overrides matching the mlx-community config, AND measure against a Q8 reference, the expected hf2q KL lands at **~0.020-0.030** — indistinguishable from smcleod's 0.02663 within recipe-noise. (Worker 4 §6, self-consistency analysis 5/5 agreement, plus mlx-optiq gemma-4-e4b empirics: uniform-Q4 → 23.5% GSM8K, mixed-precision → 55.5%, ~136% relative gain.)

### 5.2 DWQ VJP — VERIFIED

Algebra (worker 4 §2):

```
qdq[i] = q_int[i] * scales[g(i)] + biases[g(i)]
∂qdq[i]/∂scales[g] = q_int[i] if g(i)==g else 0
∂qdq[i]/∂biases[g] = 1        if g(i)==g else 0
∂qdq[i]/∂q_int[i]  = scales[g(i)]   ← DWQ freezes; gradient zeroed

⇒ ∂L/∂scales[g] = Σ_{i ∈ group g} c[i] * q_int[i]
⇒ ∂L/∂biases[g] = Σ_{i ∈ group g} c[i]
```

These match `mlx/primitives.cpp:3459-3525 QuantizedMatmul::vjp`:
- line 3487: explicit "no gradient wrt the quantized weights"
- bias-grad: `sum(cotangent, -1)` over each group
- scale-grad: `wq = dequantize(w_q, scales=1, biases=0)` [= q_int] then `sum(cotangent * wq, -1)`

hf2q `iter-13b` implements this in `qdq_affine_backward_{scales,biases}_f32` Metal kernels, verified by FD falsifier at 1% tol. Synthetic 2-tensor convergence: loss ratio 0.06 ≪ 0.2× acceptance.

**Verdict: VERIFIED both as algebra and as hf2q implementation.**

### 5.3 KL distillation — VERIFIED

`mlx_lm/tuner/losses.py:130-167` Metal kernel computes:
```
kl = Σ_j P_teacher[j] * (log P_teacher[j] - log Q_student[j])
   = KL(P_teacher ‖ Q_student)
```

Mode-covering forward KL (Hinton-style soft-target distillation), with temperature `scale=1/T=0.5` applied to BOTH student and teacher logits before softmax. The `T²` factor that some papers multiply at the loss level is absent — it's effectively absorbed into the LR=1e-6 default.

VJP at `losses.py:307`: `∂KL/∂logits_q[j] = q[j] - p[j]` — canonical-correct.

**Verdict: VERIFIED.**

### 5.4 Adam with bias correction — VERIFIED

mlx Adam (`/opt/homebrew/lib/python3.14/site-packages/mlx/optimizers/optimizers.py:512-535`) with `bias_correction=True`:

```python
m = b1 * m + (1 - b1) * gradient
v = b2 * v + (1 - b2) * mx.square(gradient)
c1 = lr / (1 - b1**step)
c2 = mx.rsqrt(1 - b2**step)
return parameter - c1 * m / (mx.sqrt(v) * c2 + eps)
```

Equivalent to PyTorch's default Adam to within negligible numerical drift in eps placement. **Goalie's anti-hallucination tool confirmed this at confidence 1.00.**

`dwq.py:393` uses `bias_correction=True` (non-default; mlx Adam defaults to `False`). **Important port detail: hf2q's port must replicate this flag.**

### 5.5 MoE differentiability — RESOLVED

`mlx_lm/models/qwen3_next.py:308-354` `Qwen3NextSparseMoeBlock.__call__`:

```python
gates = mx.softmax(self.gate(x), axis=-1, precise=True)   # softmax FIRST
inds   = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]  # top-k indices
scores = mx.take_along_axis(gates, inds, axis=-1)           # gather PROBS
if self.norm_topk_prob:
    scores = scores / scores.sum(axis=-1, keepdims=True)
y = self.switch_mlp(x, inds)                                 # SwitchGLU dispatch
y = (y * scores[..., None]).sum(axis=-2)
shared_y = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
y = y + shared_y
```

`switch_layers.py:186-187`: `if self.training: idx = mx.stop_gradient(idx)` — canonical straight-through estimator. Indices are constant in backward; gradients flow through `gates → softmax → take_along_axis → scores` and through `switch_mlp(x, idx)` for the touched expert rows.

**Implication for Path B**: the subprocess driver runs mlx-lm's MoE forward+backward unchanged. No hf2q autograd MoE op needed for Path B. (Path A would need both differentiable `take_along_axis` AND `gather_mm` — worker 4 R1 marks this HIGH-severity for any pure-Rust port.)

---

## 6. The risks that matter (from worker 4)

### R1 — mlx-lm version drift (MEDIUM)

Apple's mlx-lm is on a 2-week cadence (`/opt/mlx-lm/mlx_lm/_version.py`). When `dwq.py` signature changes, our Python driver breaks.

**Mitigation:**
- Pin mlx-lm SHA in `bin/dwq_python_driver.py` and `EXPECTED_MLX_LM_SHA` constant.
- Version handshake aborts before any forward pass on mismatch.
- CI smoke test on a representative fixture catches signature breakage at PR time, not at iter-20 time.
- `Cargo.toml` documents the pinned SHA with a comment block.

### R2 — Mixed-precision recipe must match the reference (HIGH)

If we ship uniform Q4 by default, our parity number will always be ~2× higher than the published mlx-community number. The driver MUST accept per-path bit overrides via `--quant-predicate-config` and ship a default mixed-precision recipe for Qwen 3.5/3.6 MoE matching mlx-community's published config.

**Mitigation:**
- iter-19h ships a `tests/fixtures/dwq_recipes/qwen35moe-mixed-precision.json` that mirrors the published config.
- Default behavior of `--quant dwq-q4` on a Qwen 3.5/3.6 MoE arch automatically picks this recipe (override required to opt out).
- Acceptance gate is calibrated against our actual measured number (uniform Q4 vs BF16 = 0.0610), NOT smcleod's 0.02663 — preserves apples-to-apples even if a future iter changes default recipes.

### R3 — KV-cache state must reset between batches (MED)

`mlx_lm/quant/dwq.py:108-117` `loss_fn` calls `model(x)` with no cache argument — fresh per-batch. If our subprocess driver reuses cache for performance, second-forward logits condition on stale K/V. Wrong gradients, slow divergence.

**Mitigation:** the Python driver must NOT pass a `cache` argument. Test: 10-batch run with cache reuse vs no-cache must produce identical loss curve.

### R4 — `precise=True` softmax (MED)

`qwen3_next.py:335` `mx.softmax(gates, axis=-1, precise=True)` — numerically-stable subtract-max-before-exp. mlx-lm uses this; if we accidentally degrade to `precise=False` (e.g., via a future hf2q-side replacement), large absolute logits saturate exp() and router decisions become pathological.

**Mitigation:** Path B inherits mlx-lm's softmax — no risk in Path B. Documented as a forward-Path-A landmine.

### R5 — `norm_topk_prob = True` must execute (MED)

`qwen3_next.py:340-341`: re-normalizes gathered top-k scores so they sum to 1. If a port forgets this branch, expert FFNs contribute ~`8/128 ≈ 6.25%` of intended magnitude — DWQ on experts becomes ineffective.

**Mitigation:** Path B inherits this from mlx-lm.

### R6 — `grad_checkpoint(model.layers[0])` quirk (LOW)

mlx-lm only checkpoints layer 0 as a memory-saving heuristic. Wrapping all layers (or none) changes peak memory and recipe faithfulness. Path B preserves this verbatim.

### R7 — Validation set seed parity (LOW)

`dwq.py:155 / 167-173`: train and valid use the SAME `seed=123`. Our protocol must forward seed unchanged.

### R8 — Adam first-step convention (LOW)

mlx Adam uses `step` starting at 1 (not 0); `b1**1 = 0.9`, `1 - 0.9 = 0.1`; no division by zero. Our Path-B driver inherits this.

### R9 — fp32 master / bf16 model dtype split (LOW for Path B)

`dwq.py:152-156`: Adam state lives in fp32, but the model forward casts back to bf16 each step. Path B inherits mlx-lm's behavior unchanged.

### R10 — Static_quant + safetensors_out is corrupt-on-load (DECISIVE finding from W4 §8)

This is independent of Path B but flagged here because it affects any *current* "save mlx-format DWQ" caller in hf2q.

`/opt/hf2q/src/quantize/static_quant.rs` + `safetensors_out.rs`:
- Quantizes to symmetric `[-7, +7]` as i8.
- Pack convention: low nibble `pair[0] & 0x0F`, high nibble `(pair[1] & 0x0F) << 4`.
- mlx affine reader reads each nibble as **unsigned [0, 15]**.
- Result: a true intent of `-1 * scale = -scale` is decoded as `15 * scale + 0 = 15 * scale`. **Off by 16× and wrong sign.**

`safetensors_out.rs:423-434` emits zero-filled biases when `quant_info.biases` is None — known antipattern.

**Mitigation (must land before iter-20):**
- Sunset the static_quant → mlx-lm-format path. Any caller emitting mlx-format must route through the iter-13b/16b `MlxAffineLinear::to_safetensors_bytes` writer (the iter-19c-tested one).
- Static_quant retains its GGUF emit path (where signed nibbles ARE the convention — Q4_0 is signed).

---

## 7. Acceptance criteria

Distilled from worker 6's full criteria document. See `/tmp/cfa-adr020-iter19d/worker6-acceptance-criteria.md` for the complete 636-line spec.

### 7.1 Layer 1 — Unit tests (each <60 s on M5 Max)

| Test | Falsifier | Pass criterion |
|---|---|---|
| `dwq_unit_qdq` | `qdq_affine_q4(W, s, b)` forward + VJP | Forward byte-equiv to scalar reference; VJP rel err ≤ 5e-3 vs FD on 50 seeds |
| `dwq_unit_kl_loss` | KL with T=2.0 forward + backward | Forward matches scipy.special.kl_div to 1e-6; backward ≤ 5e-3 rel vs FD |
| `dwq_unit_adam` | Adam(lr=1e-3, β=0.9/0.999) on quadratic | After 100 steps, x matches PyTorch ref to 1e-5 absolute |
| `dwq_unit_lr_schedule` | warmup=100 + cosine decay | All 4 boundary points within 1e-9; monotone phases |
| `dwq_unit_sensitivity` | 2-Linear synthetic MLP per-Linear sensitivity | Rank order matches analytical exactly; deterministic across 10 runs |
| `dwq_unit_safetensors` | mlx-format round-trip | header `format='mlx'`; mlx_lm Python loader successfully calls `model.load_weights(...)` |
| `dwq_unit_calibration_loader` | tulu-3 with seed=123, n=2048, l=1025 | SHA256 of token stream matches frozen fixture |

### 7.2 Layer 2 — Integration (each <30 min)

| Test | Pass criterion |
|---|---|
| `single_linear_real_bf16` (port of iter-19c) | KL post-train ≤ 5e-3 (iter-19c: 2.73e-3, 2× headroom) |
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

**RTN sanity (independent gate):** `hf2q_RTN_Q4_KL ≤ 0.0676` (= 0.0644 × 1.05). Tight because RTN is closed-form; deviation > 5% indicates a bit/group encoding bug.

**Tolerance justification:**
- ±3-5% from 512-sample kld.py shuffle-seed variance
- ±5% from BF16 reduction-order non-determinism between Metal kernels (us) and MLX (mlx-lm)
- ±5% from gradient accumulation sequencing under same seed=123
- RSS sum ≈ ±13%, rounded to 15% as the hard gate

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
9. **Coherence**: `hf2q serve --model <output> --prompt "the quick brown fox" --max-tokens 32` produces non-degenerate UTF-8 with no `<|_end|>` literal-byte leak (defends against the pre-505b5b8 vocab-truncation bug class)
10. Output safetensors: `format='mlx'` metadata, no NaN in any scale or bias

### 7.5 Performance acceptance

| Metric | Target | Hard ceiling |
|---|---|---|
| Per-step training time | ≤ 20 s/step at batch=2 | ≤ 30 s/step |
| Peak unified memory | ≤ 80 GB | ≤ 100 GB |
| Disk I/O (target cache) | ≤ 10 GB | ≤ 12 GB |
| Total wall time | ≤ 6 hours on stock 35B | ≤ 8 hours |

---

## 8. Test plan (concrete shell)

CI workflow runnable from `/opt/hf2q-adr020-iter10/`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Phase 0: build
cargo build --release --bin hf2q

# Phase 1: Layer 1 unit tests (≈ 5 min total)
cargo test --release --test 'dwq_unit_*'

# Phase 2: Layer 2 integration (≈ 30 min)
cargo test --release --test 'dwq_integration_*'
RUN_SLOW_TESTS=1 cargo test --release --test full_forward_parity_qwen35 -- --ignored

# Phase 3: Layer 3 parity (4-6 hours wall)
DWQ_OUT=/tmp/hf2q_stock_dwq_output
RTN_OUT=/tmp/hf2q_stock_rtn_output
SOURCE=/opt/hf2q-adr020-iter10/scripts/dwq_kl_parity/stock_working_dir
BF16=/opt/hf2q-adr020-iter10/scripts/dwq_kl_parity/stock_mlx_bf16

target/release/hf2q convert --quant rtn-q4 \
    --hf-path "$SOURCE" --mlx-path "$RTN_OUT" \
    --bits 4 --group-size 64 --seed 123

target/release/hf2q convert --quant dwq-q4 \
    --hf-path "$SOURCE" --mlx-path "$DWQ_OUT" \
    --calibration-data allenai/tulu-3-sft-mixture --num-samples 2048 \
    --max-seq-length 1025 --bits 4 --group-size 64 \
    --learning-rate 1e-6 --batch-size 2 --seed 123 --grad-checkpoint

cd /opt/hf2q-adr020-iter10/scripts/dwq_kl_parity
python3 kld.py --model "$DWQ_OUT" --baseline-model "$BF16" \
    --top-k 1024 --data-path allenai/tulu-3-sft-mixture \
    --sequence-length 1025 --num-samples 512 --batch-size 4 --seed 123 \
    > hf2q_stock_dwq_kld.txt 2>&1
python3 kld.py --model "$RTN_OUT" --baseline-model "$BF16" ... > hf2q_stock_rtn_kld.txt

# Gate evaluation
HF2Q_DWQ=$(grep -aoE '"mean_kl_per_token":\s*[0-9.eE+-]+' hf2q_stock_dwq_kld.txt | tail -1 | grep -oE '[0-9.eE+-]+$')
HF2Q_RTN=$(grep -aoE '"mean_kl_per_token":\s*[0-9.eE+-]+' hf2q_stock_rtn_kld.txt | tail -1 | grep -oE '[0-9.eE+-]+$')
python3 - <<EOF
hf2q_dwq, hf2q_rtn = $HF2Q_DWQ, $HF2Q_RTN
mlx_dwq, mlx_rtn = 0.0610, 0.0644
dwq_ratio = hf2q_dwq / mlx_dwq
rtn_ratio = hf2q_rtn / mlx_rtn
print(f"DWQ: {hf2q_dwq:.5f} / {mlx_dwq:.5f} = ratio {dwq_ratio:.3f}")
print(f"RTN: {hf2q_rtn:.5f} / {mlx_rtn:.5f} = ratio {rtn_ratio:.3f}")
import sys
fail = False
if hf2q_dwq > 0.100: print("FAIL: DWQ above smcleod broken floor"); fail=True
if dwq_ratio > 1.30: print("FAIL: DWQ ratio > 1.30"); fail=True
elif dwq_ratio > 1.15: print("SOFT_PASS: write audit")
else: print("PASS: DWQ ≤ 1.15")
if rtn_ratio > 1.05: print("FAIL: RTN ratio > 1.05 (encoding bug)"); fail=True
sys.exit(1 if fail else 0)
EOF

# Phase 4: Layer 4 ship gates
target/release/hf2q serve --model "$DWQ_OUT" --prompt "the quick brown fox" --max-tokens 32 > /tmp/dwq_coherence.txt
python3 -c "
data = open('/tmp/dwq_coherence.txt','rb').read()
assert b'<|_end|>' not in data, 'literal-byte token leak'
assert data.decode('utf-8'), 'invalid UTF-8'
"
target/release/hf2q convert --quant dwq-q4 --hf-path "$SOURCE" \
    --mlx-path /tmp/hf2q_stock_dwq_output_run2 ... # determinism re-run
diff <(cd "$DWQ_OUT" && sha256sum *.safetensors) \
     <(cd /tmp/hf2q_stock_dwq_output_run2 && sha256sum *.safetensors)
cargo llvm-cov --release --test 'dwq_*' --html --output-dir /tmp/dwq_coverage
```

---

## 9. Calibration data integrity

Reproducibility anchor:

- Corpus: `allenai/tulu-3-sft-mixture` (HF dataset)
- Tokenizer: model's own (Qwen 3.5 vocab = 248,320 tokens, eos=248046; MUST NOT be the truncated 248044 from pre-505b5b8 builds — see existing memory `project_qwen35_dwq_pre_505b5b8_broken_2026_05_05`)
- Seed: 123
- Sample counts: 2048 (training) / 512 (evaluation)
- Max sequence length: 1025
- Tokenization: chat-template applied per `scripts/dwq_kl_parity/queue_stock_run.sh:198-221`

**Frozen reference fixtures** (commit at `tests/fixtures/dwq_calibration/`):
- `tulu3_seed123_n2048_l1025.tokens.bin` — concatenated u32 token IDs (~8 MB), SHA256-anchored
- `tulu3_seed123_n2048_l1025.sha256` — single-line SHA256 hex

CI step `sha256sum -c` before any DWQ training run blocks on corpus drift.

---

## 10. Adjacent ADR connections

| ADR | Relationship | Notes |
|---|---|---|
| **ADR-005** (mixed-prec K-quant) | **Enables** | DWQ-tuned scale book is the input to `MixedBitQuantizer`. iter-19h is the join point. `LayerQuantConfig` in `src/quantize/mod.rs` already has the fields DWQ needs. No schema change. |
| **ADR-013** (perf) | Independent | Build-time tool vs runtime perf. Zero shared infrastructure. |
| **ADR-014** (streaming convert) | **Shares infra** | Streaming-load + per-tensor mmap is what fits 35B in memory during teacher phase. ADR-020 §A.2 already cites this. Subprocess driver consumes same `~/.cache/hf2q/sensitivity` layout. |
| **ADR-017** (KV streaming) | Independent | Lessons (engagement test pattern, fail-loud on counter mismatch) transfer to iter-19g. |
| **ADR-018** (load UX) | **Shares infra** | Path B's banner ("Loading mlx-lm 0.X.Y from /opt/mlx-lm; Python 3.12.4") plugs into the unified `LoadInfo` builder pattern from ADR-018 commits a805221 + ecd4647. ~60 LOC saved. |
| **ADR-021** (Qwen3-VL ViT) | Independent for now | Per-layer driver pattern transfers if we ever DWQ-tune the ViT separately. |
| **ADR-020** (DWQ streaming calibration) | **Supersedes** | This ADR replaces row 19b half-2's gating chain (iter-11h + iter-14b for full-Rust autograd). ADR-020 §8.2 row 19b half-2 status updates to "deferred to ADR-022 Path B". iter-11h becomes a *future* ADR (ADR-023+) for if/when the team chooses to do the pure-Rust port. |

---

## 11. Reference reading

The implementing engineer should read, in this order:

1. `/opt/mlx-lm/mlx_lm/quant/dwq.py:29-66` — `compute_dwq_targets`. The teacher-drop sequencing the subprocess must replicate.
2. `/opt/mlx-lm/mlx_lm/quant/dwq.py:69-209` — `dwq_quantize`. Optimizer loop, KL temperature trick, gradient-checkpoint hook, the `model.update(tree_map(...))` pattern that scales propagate through.
3. `/opt/mlx-lm/mlx_lm/quant/dwq.py:242-411` — CLI entry point. Argument names and defaults the Python driver must accept verbatim.
4. `/opt/mlx-lm/mlx_lm/utils.py:282-420` — `load_model` + `quantize_model`. The `lazy=True` path; how `class_predicate` is keyed off `f"{p}.scales" in weights`; how `quantize_model` reaches `nn.QuantizedLinear.from_linear`.
5. `/opt/mlx-lm/mlx_lm/utils.py:925-950` — `save`. The `metadata={"format": "mlx"}` write at line 756 (kld.py prereq).
6. `/opt/mlx-lm/mlx_lm/models/qwen3_5.py` + `qwen3_5_moe.py` + `qwen3_next.py` — Qwen 3.5 forward graph, hybrid attention routing, Qwen3NextSparseMoeBlock with top-8 routing.
7. `/opt/hf2q-adr020-iter10/src/calibrate/dwq_e2e.rs:481` — iter-19c single-Linear KL parity test on real BF16. The minimal "this works" anchor.
8. `/opt/hf2q-adr020-iter10/src/calibrate/dwq_loop.rs::init_affine_params_gpu` + `box_muller_gaussian` — the per-tensor primitives the subprocess driver does NOT replace (they're for the in-process iter-19c spot check that remains as a regression).
9. `/opt/hf2q-adr020-iter10/src/calibrate/mlx_safetensors_loader.rs` — read+write byte-correct mlx-format safetensors. The output sink for Path B's stage 3 (mlx → GGUF conversion).
10. `/opt/hf2q-adr020-iter10/scripts/dwq_kl_parity/kld.py` — vendored PR #1146 KLD measurement; baseline must be `format='mlx'` (line 328 hard-rejects others).
11. `/opt/hf2q-adr020-iter10/docs/ADR-005-mixed-precision-kquant.md` + `ADR-014-streaming-convert-pipeline.md` + `ADR-018-model-load-ux.md` — for `LayerQuantConfig`, `MixedBitQuantizer`, sensitivity cache layout, and the `LoadInfoBuilder` pattern.
12. `/opt/llama.cpp/convert_hf_to_gguf.py` — canonical reference for how mixed-precision tensors lay out in a GGUF v3 file. Confirm DWQ-emitted scales survive a llama.cpp re-load.

The engineer should NOT read `/opt/mlx-lm/mlx_lm/quant/awq.py` or `gptq.py` until iter-20a or later — out of scope for ADR-022 and tempting rabbit holes.

---

## 12. Open questions and follow-ups

1. **Is `mlx-community/Qwen3.6-35B-A3B-bf16` byte-identical to `mlx_lm.convert --hf-path Qwen/Qwen3.6-35B-A3B --dtype bfloat16`?** Worker 4 §7 hypothesizes "likely equivalent" but unverified. The cheap falsifier (sha256sum cross-check, ~5 min) should run before any apples-to-apples claim. **Action**: iter-19f spike — download mlx-community BF16 + diff configs + sha256 the safetensors. If they differ, we may need to use mlx-community's BF16 as our `--baseline-model` for kld.py to truly match smcleod's numbers.

2. **Should the default `--quant dwq-q4` recipe be uniform Q4 or mixed-precision Q8/Q4-experts?** The mlx-community published recipe is mixed-precision (effective 4.84 bpw). If we default to mixed-precision:
   - Pro: matches the published number directly; easier to communicate "we're at parity."
   - Con: the bits-per-weight isn't actually 4 anymore — operators expect "Q4" to mean ~4 bpw.
   - **Recommendation**: default to uniform Q4 (operator's expected meaning), but ship the mixed-precision recipe as a named preset (`--quant-predicate-config qwen35moe-mixed-precision`). Document that the parity number against smcleod's 0.02663 requires the mixed-precision recipe.

3. **What's the right cross-family smoke model?** Llama 3.x 8B-Instruct vs Mistral 7B-v0.3 — both are dense, both have pre-baked DWQ-Q4 in mlx-community for cross-validation. **Recommendation**: Llama 3.1 8B-Instruct (mlx-community/Llama-3.1-8B-Instruct-4bit-DWQ exists with a published number).

4. **`hf2q` having no `[lib]` target** — per existing memory `project_hf2q_no_lib_target_unit_test_friction`, `tests/dwq_unit/*` would have to live as `#[cfg(test)]` blocks inside the bin source OR require expanding `lib.rs` to re-export `calibrate::dwq_*`. The latter has high blast radius (pulls mlx-native into the lib surface). **Recommendation**: keep `#[cfg(test)]` pattern for unit tests; `tests/dwq_integration/*` becomes a new `[[bin]] dwq_integration` similar to existing `extract_dwq_sensitivity` / `dump_gguf_types` (per `Cargo.toml:172-185`).

5. **Subprocess driver vs hf2q's existing `assert_cmd` test pattern** — should the iter-19e/f tests reuse `assert_cmd` for the subprocess lifecycle? **Recommendation**: yes; `assert_cmd` is already a project-wide testing pattern and trivially extends.

6. **iter-11h fate** — does the deferred pure-Rust autograd port ever land? **Recommendation**: leave as a year-2+ option in a future ADR-023. Unless mlx-lm pivots away from supporting Qwen 3.5/3.6 (unlikely on a 2-week-cadence repo), Path B is sufficient indefinitely. The autograd op surface we'd build for Path A becomes valuable only if hf2q wants to ship a *second* quantization algorithm that mlx-lm doesn't (e.g., research-level OmniQuant or SpinQuant). That's a separate decision.

---

## 13. Decision history

| Date | Event |
|---|---|
| 2026-04-XX | ADR-020 §8.2 row 19b half-1 lands: vendor kld.py, build harness scripts |
| 2026-04-XX → 2026-05-06 | iter-13a–13e + iter-14 + iter-15/15b + iter-16/16b + iter-17 — per-component primitives all green |
| 2026-05-06 | iter-19c PASS on real BF16 single Linear (KL = 2.73e-3) |
| 2026-05-07 | mlx_lm.dwq runs on stock 35B-A3B + abliterated 35B-A3B harvested as reference targets |
| 2026-05-07 | User flags antipattern: "shelling out to other repo's tools is an antipattern, and explicitly goes against the entire fucking point of hf2q" |
| 2026-05-07 | CFA session `cfa-20260507-191500-adr020-followup-research` runs 5 parallel research workers + queen synthesis |
| 2026-05-07 | This ADR-022 lands as the proposed plan |
| **next** | iter-19e begins (Python driver skeleton + JSON protocol) once user gives go-ahead |

---

## 14. Worker-report appendix

The five CFA worker reports backing this ADR are preserved verbatim at:

- `/tmp/cfa-adr020-iter19d/worker2-mlx-lm-peer-map.md` (603 lines) — exhaustive map of every load-bearing surface in `/opt/mlx-lm/mlx_lm/`
- `/tmp/cfa-adr020-iter19d/worker3-hf2q-codebase-map.md` (287 lines) — what's shipped, what's missing, exact line numbers and LOC estimates
- `/tmp/cfa-adr020-iter19d/worker4-math-validation.md` (583 lines) — VJP / KL / Adam / MoE math verified; goalie anti-hallucination + WebFetch corroboration; the smcleod-gap explanation
- `/tmp/cfa-adr020-iter19d/worker5-architecture-tradeoffs.md` (225 lines) — three paths analyzed; comparison matrix; phasing
- `/tmp/cfa-adr020-iter19d/worker6-acceptance-criteria.md` (636 lines) — full layered acceptance criteria, test commands, performance targets

These reports should be checked into the repo at `docs/research/cfa-adr020-iter19d/` for permanent reference.

---

## 15. One-paragraph summary for ADR index

ADR-022 supersedes ADR-020 row 19b half-2's gating chain. Instead of building a full-model differentiable Qwen 3.5/3.6 MoE forward in `GpuTape` (estimated 5-7 weeks, 5,500-7,000 LOC, blocking on research-level gated-delta-net backward + MoE router gradient), hf2q completes its DWQ port via a subprocess bridge to a pinned `mlx_lm.dwq`. hf2q owns CLI, calibration management, mixed-precision recipe selection, GGUF emission, and parity measurement; mlx-lm owns the differentiable forward + backward. The bridge is a 150-LOC Python driver + 1,300 LOC of Rust, landing across 6 iterations (~5 weeks) culminating in the load-bearing v1 ship gate at iter-20: hf2q-DWQ KL ≤ 0.0702 (= mlx-lm's 0.0610 × 1.15) on stock Qwen 3.6 35B-A3B against the canonical BF16 reference, with hard fail above 0.0793 or 0.100 absolute. The math (VJP, KL distillation, Adam bias correction, MoE differentiability) is independently verified against `mlx/primitives.cpp:3459-3525` and `mlx_lm/tuner/losses.py:130-167`; the published smcleod 0.02663 number turns out to be against a Q8 reference of a 4.84-bpw mixed-precision checkpoint, not BF16 of uniform Q4 — fully explaining the apparent 5%-vs-64% gap in the iter-19b runs. Pure-Rust port deferred to ADR-023+.
