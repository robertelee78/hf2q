# ADR-020 iter-19d — Worker 5: Architecture Trade-off Analysis

**Author:** Worker 5 (System Architecture Designer)
**Session:** cfa-adr020-iter19d
**Date:** 2026-05-07
**Scope:** Three implementation paths for full-model DWQ on Qwen 3.5 / 3.6 35B-A3B in hf2q.

---

## Ground state (2026-05-07)

Before the per-path analysis, four facts pin the trade-space:

1. **`GpuTape` op set is currently 6 operators** — `matmul`, `softmax`, `add`, `sub`, `mul`, `square` (`/opt/hf2q/src/calibrate/autograd_gpu_tape.rs:233-1039`). That is the entire forward+backward op surface available today. Anything past a two-Linear MLP requires new tape ops and new backward dispatch arms.
2. **Qwen 3.5/3.6 model code in hf2q is ~6,135 LOC** (`models/qwen35/mod.rs` 3,425 + `moe.rs` 2,194 + `dense.rs` 516) and is hand-written for *inference* — `&self` forwards, no autograd hooks, scalar-quant FFN paths. Re-using it for training requires either teeing every op into the tape (intrusive) or duplicating the forward in tape ops (large).
3. **mlx-lm's Qwen 3.5/3.6 model is 531 + 52 = 583 LOC of Python** (`models/qwen3_5.py`, `qwen3_5_moe.py`) and gets autograd for free via `mx.value_and_grad`. The DWQ training driver is 411 LOC (`quant/dwq.py`). Total mlx-lm DWQ surface = ~1,000 LOC, leveraging ~30k LOC of MLX framework.
4. **`hf2q/Cargo.toml` has no PyO3 today**; mlx-native is path-pinned to `/opt/mlx-native` at version 0.6. Adding PyO3 is a real dependency decision — it brings a CPython linkage requirement, GIL semantics, and a wheel-distribution problem the project has so far avoided.

These four facts asymmetrically penalize Path A and asymmetrically reward Paths B/C.

---

## Path A — Pure hf2q port

**Rough mechanism.** Build the full Qwen 3.5/3.6 forward inside `GpuTape` such that every op (RoPE, RMSNorm, SDPA, gated-delta-net, top-k MoE router, sigmoid gating, residual sum, embedding, LM head) has a forward kernel and a backward dispatch arm. Replace `MlxModelWeights` with a tape-aware `MlxModelWeightsAutograd` that materializes the same numerics but threads `GpuTensor` rather than `MlxBuffer`. `dwq_loop.rs` (does not yet exist) drives `value_and_grad` over learnable scales+biases of the quantized blocks, against teacher-precomputed top-1024 logits cached on disk per `dwq.py:29-66`.

**LOC estimate.** Bulk goes here:
- `autograd_gpu_tape.rs` extensions: +2,500–3,500 LOC. Need RoPE (forward+backward), RMSNorm (fw+bw), SDPA fw+bw (fused or unfused), `silu`, `sigmoid`, `gather` (for embedding + top-k MoE indexing), `scatter_add` (for MoE expert combine), `transpose`, `reshape`, `concat`, `broadcast_add`, `broadcast_mul`, gated-delta-net cell fw+bw. Each new op needs forward kernel (likely re-using `mlx-native` ops) plus a backward dispatch arm in `backward_dispatch` and an accumulator path in `accumulate`.
- New `models/qwen35/autograd.rs`: ~1,500–2,000 LOC. Tape-aware twin of `mod.rs` decode/prefill — except it must also handle MoE training (router gradient, expert-mixture gradient).
- `calibrate/dwq_loop.rs`: ~600 LOC. Optimizer state (Adam/AdamW), batch iterator, gradient checkpoint scaffolding, KL loss, save/restore.
- `calibrate/teacher_provider.rs`: ~400 LOC. `compute_dwq_targets` port — top-k argpartition, safetensors stream-to-disk, two-phase orchestrator with explicit teacher-drop between phase 1 and phase 2.
- mlx-native side: 1–3 new differentiable kernels if SDPA/RoPE backward isn't expressible from existing primitives. Conservatively +500 LOC.
- **Total: 5,500–7,000 net new LOC**, of which ~70% is autograd plumbing the rest of hf2q never needs.

**Test surface.**
- Per-op finite-difference parity tests against an analytical reference for every new tape op (~25 new tests, copying the iter-19c pattern at `dynamic_quant_gpu.rs:482` — synthetic small fixture, hand-derived reference).
- Two-Linear and three-Linear synthetic chain tests (already exist for `matmul`+`mul`; extend to RMSNorm + SDPA chains).
- Single-layer Qwen-block parity test: full attention + MLP forward+backward, vs. mlx-lm Python reference recorded into a fixture file. **This is new and load-bearing.** It is also the test most likely to expose nondeterminism between mlx-native and MLX (different reduction order, different RoPE phase).
- Full-model end-to-end PPL drop test on a tiny 1-block toy model (sanity).
- Real Qwen 3.6 35B run as the integration test — but that is a 4-hour smoke, not a unit test.
- iter-19c-style fixtures cover ~10% of what is needed; the rest is new ground.

**Risks.**
- *Technical:* SDPA backward on MoE attention with KV cache semantics is a research problem in our codebase, not a port. Gated-delta-net backward is published only in the original paper — no reference implementation exists in mlx-native or candle. **High risk** of getting stuck on "the gradient through `gdn.ops5-9` doesn't match mlx-lm's autograd." Numerical parity at the 1e-4 KL level requires matching reduction order, which mlx-native does not currently guarantee (per ADR-015 iter-92's silent-drift findings).
- *Maintenance:* When mlx-lm adds a new model family (cf. Apertus, exaone4, deepseek_v32 — all landed in mlx-lm in the last quarter), we have to port the model AND its autograd path. We become the Apple-MLX team but in Rust, with one engineer.
- *Performance:* Best-case ceiling is parity with mlx-lm. Realistic case is 0.5×–0.8× because we don't have MLX's lazy graph fusion. ADR-015 already documents 30%+ CB overhead vs. llama.cpp on inference; training amplifies this because every backward op opens a new CB.

**Strengths.**
- Total isolation. No Python at runtime. Single `cargo build` produces a self-contained binary. This is the only path that preserves hf2q's "pure Rust converter" identity.
- Reusable infrastructure: tape ops added for DWQ would be available for future quant algorithms (GPTQ, AWQ, SqueezeLLM, OmniQuant) without re-bridging.
- Determinism: with mlx-native's `MLX_UNRETAINED_REFS=0` default we already have byte-identical inference across runs. Training-time parity becomes auditable.

**Time-to-first-measurement (first end-to-end DWQ KL number on real Qwen 3.6 35B).**
- Optimistic: 18 man-days (assumes existing GpuTape extends cleanly, no SDPA-backward research stall).
- Realistic: 35–45 man-days (1 month full-time). Matches the historical pace of ADR-013 and ADR-015 (multi-iter stalls are the norm in our repo).
- Pessimistic: 70+ man-days if gated-delta-net backward gets stuck. Memory.md already records ADR-015 burning weeks on a single CB ordering bug.

**Long-term fit with hf2q's mission (HF→GGUF + mixed-precision).** **High in principle, contingent in practice.** hf2q's stated mission is "the canonical converter from HF to GGUF + GGUF-with-mixed-precision-quant." A pure Rust DWQ training loop would let hf2q ship a *one-stop* tool: input HF safetensors, output GGUF with a mixed-precision DWQ-tuned scale book, with no Python in the pipeline. That is a real differentiator vs. mlx-lm (Python-only) and vs. llama.cpp (no DWQ). However: the mission is "converter," not "trainer." Training infrastructure is a means to an end (better quant scales). If we can get the same scales 70% as well via Path B/C in 20% of the time, the pure port is a vanity feature.

---

## Path B — Hybrid (Rust drives, Python forward via subprocess)

**Rough mechanism.** `dwq_loop.rs` stays in Rust and owns: argument parsing, dataset chunking, GGUF emission, mixed-precision scale-book bookkeeping, KL aggregation, checkpointing, and the resumable two-phase orchestrator. The forward+backward pass is delegated to a Python subprocess: hf2q ships a small Python module `hf2q.dwq_forward` that invokes `mlx_lm.quant.dwq.dwq_quantize` directly. Communication is over a typed JSON-on-stdio protocol (request: batch-id, params-version, hyperparams; response: per-batch loss, gradient L2, scale updates as tensor blob path on disk). Python writes scale updates to safetensors; Rust reads them and applies them to the in-progress GGUF.

**LOC estimate.**
- `bin/dwq_python_driver.py`: 150–250 LOC. Thin wrapper around `mlx_lm.quant.dwq:69 dwq_quantize`. Handles the JSON protocol, loads model from a temp safetensors snapshot, returns per-step deltas.
- `calibrate/dwq_loop.rs`: ~700 LOC. Subprocess spawn (`std::process::Command` already widely used in hf2q via `assert_cmd` in tests), JSON line protocol, kill-on-Ctrl-C, restart-on-OOM, per-batch progress emission via existing `ProgressReporter`.
- `calibrate/dwq_python_protocol.rs`: ~250 LOC. Strongly-typed `serde` request/response, version handshake, fixture-replayable for testing.
- `calibrate/teacher_provider.rs`: optional — could either be kept on Python side (1 LOC, just call `compute_dwq_targets` first) or ported to Rust as a thin wrapper that also subprocesses (~50 LOC).
- **Total: 1,100–1,300 net new LOC in hf2q + 150–250 LOC in a sibling Python module.**

**Test surface.**
- Protocol unit tests: serde round-trip, version-mismatch rejection, malformed-JSON robustness — all cheap.
- A **pinned mlx-lm version** in `/opt/mlx-lm` with a hash-checked install banner. `bin/dwq_python_driver.py` asserts `mlx_lm.__version__ == EXPECTED_VERSION` on startup.
- A "tiny fixture" integration test: 32-dim toy Qwen-shaped model, 2 batches, expected KL after 1 step recorded as a snapshot. Detects subprocess-protocol drift without needing a real model.
- Live smoke on Qwen 3.6 35B as the integration gate.
- iter-19c synthetic fixtures cover the *Rust-side* sensitivity logic; they don't apply to Path B's Python side.
- Re-uses ~40% of iter-19c test scaffolding (the hf2q-side orchestration pieces).

**Risks.**
- *Technical:* Subprocess overhead per batch (~80–150 ms cold-start amortized away after first batch since the driver stays alive for the whole run). IPC payload size: Python writes scales to disk per step (~few MB on 35B-A3B with affine 4-bit), Rust reads them — well under the 100 MB/s NVMe ceiling. **Low technical risk.**
- *Maintenance:* mlx-lm version drift. Apple's mlx-lm is on a 2-week cadence (`/opt/mlx-lm/mlx_lm/_version.py`). When `dwq.py` signature changes, our Python driver breaks. Mitigation: pin the mlx-lm SHA and gate updates on a CI-run smoke test. **Medium risk** but bounded.
- *Maintenance:* Distribution. hf2q today ships as a single binary; Path B requires either (a) bundling Python + mlx-lm wheel as a sibling install, or (b) requiring users to `pip install mlx-lm`. (b) is the pragmatic answer and matches what `oq.py` users do today.
- *Performance:* Per-step overhead is dominated by the forward+backward, which is mlx-lm's job. Subprocess IPC is sub-1% of step time. Negligible perf overhead.

**Strengths.**
- *Speed to first measurement.* This is the path that gets us a real KL number on Qwen 3.6 35B fastest, by an order of magnitude over Path A.
- *Risk isolation.* When mlx-lm fixes a bug in `dwq.py`, we get the fix for free. When mlx-lm breaks, we keep our pinned version.
- *Honest layering.* The mission is "canonical converter." Quant *training* is delegated to a battle-tested upstream; we own the conversion pipeline, the GGUF emission, and the mixed-precision scale-book — all of which Apple has no interest in.
- *Operator transparency.* Subprocess boundary makes it trivial to swap Python for a future Rust impl. Path A and Path C don't have this seam.

**Time-to-first-measurement.**
- Optimistic: 4 man-days (subprocess wrapper + JSON protocol + smoke).
- Realistic: 7–10 man-days.
- Pessimistic: 15 man-days if mlx-lm's `dwq.py` requires unexpected hyperparam wiring or distributed init shims.

**Long-term fit.** **Excellent.** This is the path most aligned with the mission's stated scope. hf2q-as-converter does not need to own training; it needs to own conversion. Path B explicitly says "we drive, mlx-lm forwards" — which is the right separation of concerns. The only counter-argument is "Python in the deployment surface," and the counter-counter is "operators already use Python for HF tooling" — anyone running an HF→GGUF pipeline on Apple Silicon already has Python on the box.

---

## Path C — FFI/PyO3 in-process

**Rough mechanism.** Same data flow as Path B, but the Python interpreter is embedded into the hf2q binary via PyO3. `dwq_loop.rs` calls `mlx_lm.quant.dwq.dwq_quantize` as a Rust function whose body is a PyO3 `Python::with_gil(|py| ...)` block. Tensors marshal through `numpy::PyArray2` or directly via the Python C-API.

**LOC estimate.**
- `Cargo.toml`: +3 deps (`pyo3`, `pyo3-build-config`, optionally `numpy` for ndarray bridge). Plus a `build.rs` for `auto-initialize` linkage discovery.
- `calibrate/dwq_pyo3_bridge.rs`: ~600 LOC. PyO3 wrappers, GIL-acquisition discipline, error translation, progress callback bridge (Rust → Python `tqdm` proxy), Ctrl-C handling.
- `calibrate/dwq_loop.rs`: ~500 LOC (slightly smaller than Path B because no JSON protocol).
- `calibrate/teacher_provider.rs`: ~150 LOC.
- **Total: ~1,250 LOC + a real CI/build-system tax.**

**Test surface.**
- PyO3 round-trip tests on every type bridged — `Vec<f32>` ↔ `np.ndarray`, `BTreeMap<String, f32>` ↔ `dict`, custom error class.
- The same tiny-fixture integration test as Path B.
- **GIL re-entry tests**: any progress callback that tries to acquire the GIL while it's already held will deadlock. Worth a dedicated falsifier.
- Build-system tests: `cargo build` on a system with no Python, on Python 3.11 vs 3.12, with and without venv, on macOS native vs. brew Python.
- Re-uses ~30% of iter-19c test scaffolding.

**Risks.**
- *Technical (build):* PyO3 needs a Python interpreter at build time and at runtime, with version-matching ABI. macOS's system Python is 3.13.x but mlx-lm wheels target 3.10/3.11/3.12. **High build-system risk.** ADR-018 (model-load UX uniformity) recently spent effort on banner clarity — Path C re-introduces "binary works on my machine but not on operator's machine" failure modes.
- *Technical (runtime):* GIL contention during Rust-driven progress reporting; segfaults on bad PyO3 type casts; Python exceptions that don't translate cleanly. Memory.md repeatedly notes that races are hard to diagnose in Apple's unified memory model — adding a GIL on top is a force-multiplier on debug time.
- *Maintenance:* hf2q's CI matrix grows from "Linux/macOS Rust" to "Linux/macOS Rust × Python 3.10/3.11/3.12 × mlx-lm SHA." Quadratic surface area.
- *Performance:* In-process is *not* meaningfully faster than subprocess for this workload. Per-step time on 35B is on the order of 200–800 ms; Path B's IPC is 1–2 ms; PyO3's GIL acquisition is microseconds. The savings are noise. The "no subprocess overhead" headline does not survive contact with the workload profile.

**Strengths.**
- Single-process semantics: Ctrl-C is cleaner, OOM is observable from Rust, no need to marshal scales through disk.
- The Python driver script disappears — the binary is self-describing.
- Future-proofs an in-process Python escape hatch for *other* algorithms (AWQ, GPTQ) at near-zero marginal cost once the bridge exists.

**Time-to-first-measurement.**
- Optimistic: 6 man-days.
- Realistic: 12–18 man-days (build-system rabbit holes are the killer).
- Pessimistic: 30 man-days if cross-Python-version compatibility becomes a real problem on operator boxes.

**Long-term fit.** **Mediocre.** The mission is "canonical converter," and the converter does not benefit from in-process Python. The strengths Path C offers over Path B (single process, cleaner Ctrl-C) are operator conveniences, not architectural wins. The maintenance tax — keeping Python ABI alignment across releases — is permanent and grows with every hf2q release. This path optimizes for the wrong axis.

---

## Comparison Matrix

| Criterion                                  | Path A (Pure Rust) | Path B (Subprocess) | Path C (PyO3) |
| ------------------------------------------ | :----------------: | :-----------------: | :-----------: |
| Mission fit (canonical converter)          | A                  | A                   | C             |
| LOC required                               | F (5.5–7k)         | A (~1.3k)           | B (~1.25k+CI) |
| Time to first measurement                  | F (5–7 wk)         | A (1–2 wk)          | C (2–4 wk)    |
| Maintenance burden                         | C (we own autograd)| B (pin mlx-lm SHA)  | D (Py ABI)    |
| Performance ceiling                        | B (theoretical 1×) | B (mlx-lm parity)   | B (mlx-lm parity) |
| Test debt                                  | F (~25 new ops)    | B (protocol+smoke)  | C (build matrix) |
| Risk of getting stuck on framework details | F (SDPA-bw, GDN-bw)| A                   | C (PyO3 build) |

A = excellent, B = good, C = acceptable, D = poor, F = failing.

---

## Recommendation

**Take Path B.** Ship a subprocess-bridged DWQ in iter-19d→iter-20, get a real KL number on Qwen 3.6 35B inside two weeks, and use the measurement to decide whether Path A is worth the 5-week investment later.

**Why not Path A right now.** The "pure Rust" identity is real and worth defending — but not at the cost of 35–45 man-days against a workstream that already has 4 prior ADRs (013, 015, 017, 018) competing for the same engineer. The current `GpuTape` op set (6 ops) is nowhere near the ~30 ops needed to express a Qwen 3.5 forward, and at least 2 of those ops (gated-delta-net backward, MoE router gradient) have no published reference. Memory.md's standing lessons — "single-shot falsifiers don't catch cumulative state-change bugs," "Apple Silicon unified memory masks RAW races until they don't," "hf2q has no [lib] target" — all warn that we will burn iterations debugging numerical drift, not algorithm correctness. Path A is a year-2 roadmap item, not a year-1 milestone.

**Why not Path C.** Path C trades Path B's only real cost (a 150-LOC Python driver script) for a permanent CI/build-system tax. The "no subprocess overhead" pitch evaporates once you measure: per-step time is 200–800 ms on 35B-A3B; IPC is 1–2 ms. We are not building a microservice with millisecond budgets — we are building a quantization pipeline with minute budgets per layer. The PyO3 build-system surface is a maintenance burden that will outlive the DWQ feature itself.

**Addressing Path A's strengths under Path B.**
- "Reusable autograd infrastructure": deferred. If/when we need a second quant algorithm that mlx-lm doesn't ship, we can revisit. The current ROI is poor.
- "Total isolation from Python": acknowledged but not paid for. Operators on Apple Silicon already have Python (mlx-lm install instructions, HF safetensors tooling). Adding a hard dep on a `pip install mlx-lm` matches the existing operator surface.
- "Determinism": Path B can match Path A's determinism by pinning mlx-lm SHA + recording per-step random seed. Reproducibility is a process-level property, not a process-boundary property.

---

## Phasing (Path B, ADR-020 iter-by-iter)

Match the ADR-020 iter-19c granularity (~50–500 LOC per iter, one falsifier per iter, named codex-review checkpoints).

| Iter | Scope | Falsifier | LOC |
| --- | --- | --- | --- |
| **iter-19d** (this CFA) | **Architecture decision recorded.** Worker outputs (5 paths analyzed in parallel). One ADR-020 update commit pinning Path B as the chosen direction. | ADR-020 §15 contains the trade-off matrix and a citation to this CFA session. | ~80 LOC docs |
| **iter-19e** | Cargo wiring + minimal Python driver skeleton. `bin/dwq_python_driver.py` (~80 LOC), JSON protocol module `calibrate/dwq_python_protocol.rs` (~250 LOC), `serde` round-trip + malformed-input tests. No mlx-lm call yet — driver echoes inputs. | Round-trip parity test on every protocol message; malformed-JSON Robustness test. | ~350 LOC |
| **iter-19f** | Subprocess lifecycle in `dwq_loop.rs`: spawn/kill/restart, version handshake, progress forwarding to `ProgressReporter`. Driver gains `import mlx_lm; print(mlx_lm.__version__)` as the version handshake. | Test that wrong mlx-lm version aborts before any forward pass. | ~400 LOC |
| **iter-19g** | Real `compute_dwq_targets` invocation through the subprocess on a 32-dim toy Qwen-shaped model. Two-phase orchestration: phase 1 writes targets to `~/.cache/hf2q/dwq-targets/<run-id>/`, phase 2 streams them. Teacher-drop handshake (driver re-execs Python between phases). | Tiny-fixture KL convergence check: 2 layers, 8 batches, KL must drop ≥30% from step 0 to step 8. | ~500 LOC + ~120 LOC Python |
| **iter-19h** | Mixed-precision scale-book integration. Output of phase 2 (per-Linear scales+biases) flows back into hf2q's existing `MixedBitQuantizer` (`/opt/hf2q/src/quantize/mixed.rs`). GGUF emission gets the DWQ-tuned scales. Verify byte-identical re-load on `cmd_serve`. | R-C4-style byte-identity check on the produced GGUF: re-running phase 2 with the same seed produces identical output bytes. | ~300 LOC |
| **iter-20** | Live measurement on Qwen 3.6 35B-A3B. PPL drop vs. iter-19c single-Linear DWQ. Bench peer comparison: same 35B model quantized via standalone `mlx_lm.quant.dwq` should produce equivalent PPL ±2%. | Peer cross-check (pattern: `feedback_peer_crosscheck_before_perf_bisect`) — if standalone mlx-lm produces materially different PPL, regression is system-wide and we audit the bridge before ourselves. | bench scripts + dossier |
| **iter-20a** (optional follow-up) | Default-on integration: `hf2q convert --quant dwq-4` works end-to-end on a fresh box with `pip install mlx-lm` as the only side install. | Operator soak: 2-turn full conversion of 27B-DWQ4 from cold. | ~150 LOC |

This is 1,800 net new LOC across 6 iters, mirroring the ADR-017 Phase E.a iter cadence (~3-7 days/iter) for a ~5-week total pipeline-to-prod.

---

## Adjacent ADR connections

| ADR | Relationship | Notes |
| --- | --- | --- |
| **ADR-005** (mixed-prec K-quant) | **Enables.** DWQ-tuned scale book is the input to MixedBitQuantizer. iter-19h is the join point. | The `LayerQuantConfig` type at `/opt/hf2q/src/quantize/mod.rs` already has the fields DWQ needs to populate. No schema change required. |
| **ADR-013** (perf) | **Independent.** DWQ is a build-time tool; ADR-013 is runtime perf. Zero shared infrastructure. | Worth noting that ADR-013's mlx-native CB-count work is not on the critical path for DWQ training perf (Python forward dominates). |
| **ADR-014** (streaming convert) | **Shares infrastructure.** ADR-014's streaming load + per-tensor mmap path is what lets a 35B model fit in memory during the teacher phase. ADR-020 §A.2 already cites this. | Subprocess driver should consume the same `~/.cache/hf2q/sensitivity` layout where applicable. |
| **ADR-017** (KV streaming) | **Independent at runtime.** No interaction. | But: lessons from ADR-017's "engagement test" pattern (default-ON falsifier, fail-loud on counter mismatch) directly transfer to iter-19g's tiny-fixture KL test. |
| **ADR-018** (load UX) | **Shares infrastructure.** Path B's banner ("Loading mlx-lm 0.X.Y from /opt/mlx-lm; Python 3.12.4") should plug into the unified `LoadInfo` builder pattern landed in ADR-018 commits a805221+ecd4647. | Direct re-use opportunity — saves ~60 LOC of bespoke banner code. |
| **ADR-021** (Qwen3-VL ViT) | **Independent for now.** | But if we ever DWQ-tune the ViT separately, the per-layer driver pattern transfers without modification. |

The largest infrastructure share is with **ADR-014 + ADR-018** — both for memory-budget-aware streaming and for the load banner. That sharing is a positive: it argues against introducing a parallel Python-only memory-management path (Path A's natural failure mode).

---

## Reference reading

Before writing the first line of iter-19e, the implementing engineer should read, in this order:

1. **`/opt/mlx-lm/mlx_lm/quant/dwq.py:29-66`** — `compute_dwq_targets`. The exact teacher-drop sequencing the subprocess must replicate.
2. **`/opt/mlx-lm/mlx_lm/quant/dwq.py:69-280`** — `dwq_quantize`. The optimizer loop, the KL temperature trick (`scale = 1/temperature`), the gradient-checkpoint hook, and the `model.update(tree_map(...))` pattern that scales propagate through.
3. **`/opt/mlx-lm/mlx_lm/quant/dwq.py:281-411`** — CLI entry point. The exact argument names and defaults the Python driver must accept verbatim, so operators can swap our binary for `python -m mlx_lm.quant.dwq` for parity testing.
4. **`/opt/hf2q/src/calibrate/dynamic_quant_gpu.rs:161-280`** — existing `kl_div_loss_per_row` + `estimate_sensitivities`. The KL formulation Path B must produce *equivalent* numbers from. iter-19g's KL convergence falsifier should compare these two paths on the same toy fixture.
5. **`/opt/hf2q/src/calibrate/autograd_gpu_tape.rs:1138-1826`** — the complete suite of finite-difference parity tests. This is the test-style template iter-19g and iter-19h should follow.
6. **`/opt/hf2q/src/quantize/dwq_k_quantizer.rs`** — the iter-19c single-Linear DWQ training path. Its failure modes (gradient explosion at temp<2.0, KL plateau when scales saturate) are the failure modes the subprocess driver will inherit. Knowing them in advance saves a debug cycle.
7. **`/opt/hf2q/src/quantize/mixed.rs`** — `MixedBitQuantizer` + `LayerQuantConfig`. The exact data structure iter-19h's scale-book consumer must populate.
8. **`/opt/llama.cpp/convert_hf_to_gguf.py`** — Apple does not have a GGUF emit path; llama.cpp is the canonical reference for how mixed-precision tensors lay out in a GGUF v3 file. Confirm DWQ-emitted scales survive a llama.cpp re-load.
9. **`/opt/candle/candle-transformers/src/models/quantized_qwen3_moe.rs`** — the only Rust-language Qwen 3.x MoE forward in the workspace. Useful as a sanity check on shapes; *not* useful for autograd (Candle's autograd is limited).
10. **`/opt/hf2q/docs/ADR-018-model-load-ux.md`** + commits ecd4647→a805221 — the `LoadInfoBuilder` pattern Path B should re-use for its driver-version banner.

The engineer should *not* read `/opt/mlx-lm/mlx_lm/quant/awq.py` or `gptq.py` until iter-20a or later — they are out of scope for ADR-020 and tempting rabbit holes.

---

## Closing assertion

Path B is not the most elegant answer, but it is the answer that ships a measured DWQ on Qwen 3.6 35B inside the next two-week window with a defensible fallback. Path A is a real future option — but only after Path B has produced a real KL number, a real mixed-precision scale book, and a real PPL improvement on a real production model. Until then, autograd-infrastructure work in `GpuTape` is speculation, not deliverables.
