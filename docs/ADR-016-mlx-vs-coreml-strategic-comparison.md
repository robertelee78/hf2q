# ADR-016: mlx-native vs coreml-native — strategic comparison (stub)

- **Status:** Proposed (stub — not yet executed)
- **Date:** 2026-04-26
- **Authors:** Robert E. Lee + Claude Code
- **Successor of (potential):** ADR-015 if the perf ceiling on mlx-native + Metal GPU proves insufficient for shippability, OR runs in parallel as a hybrid path
- **Sibling crates:** `/opt/mlx-native` (Metal-GPU compute), `/opt/coreml-native` (CoreML inference: CPU/GPU/ANE)

## Why this is its own ADR

ADR-015 closes the qwen35 / gemma decode gap **within the existing mlx-native + Metal GPU substrate** (single-CB rewrite + Rust-orchestration sweep).  Whether a CoreML/ANE path could deliver larger wins — or where it could complement mlx-native in a hybrid pipeline — is a strategic comparison whose evidence requires substantial conversion + benchmark work that doesn't belong in the ADR-015 implementation scope.

## Two-phase question

### Q1 — Is there a Qwen3.5-MoE / Gemma-4 forward pass that runs efficiently through CoreML on M5 Max's ANE?

Open.  Likely answer is **no** for the full stack, because:

- **DWQ tensor format** is not a CoreML quantization scheme.  Conversion path would require re-quantizing through `coremltools` (likely losing our calibration) OR teaching CoreML to ingest GGUF blocks (significant tooling work).
- **MoE expert routing + Gated DeltaNet** are not standard CoreML ops; CoreML's compiler routes unknown ops to CPU, defeating ANE acceleration on our hot path.
- **Stateful KV cache via `MLState`** (macOS 15+) is supported in `coreml-native` but its perf characteristics on long-decode + 256-expert MoE are unmeasured.

A two-day spike using `coremltools` to convert a small Qwen3.5 or Gemma model + run the `coreml-native` `predict_async` decode path + measure n=256 tokens vs same model on mlx-native would settle this.  Until that spike, we operate on the assumption that mlx-native is the right substrate for these two model families.

### Q2 — Where *does* CoreML/ANE win on adjacent workloads we already ship?

Strong candidates:

- **mmproj / ViT prefill** in vision models (Gemma-4V, Qwen vision).  ViT is well-trodden, ANE has good op coverage, prefill is a one-shot non-stateful workload that maps cleanly to `Model::predict`.  `feedback_perf_gate_thermal_methodology` cold-SoC bench would be straightforward.
- **BERT-style embeddings** (the `bert-test` target in `models/bert-test/`).  Classic transformer arch, no novel ops, perfect ANE candidate.
- **Whisper / ASR** if we ever pick up speech.  Apple ships ANE-tuned Whisper variants and their speedup over CPU/GPU is well-documented.

Each of these is a separate, focused experiment.  ADR-016 is the umbrella; per-workload experiments would land as ADR-016a (mmproj), ADR-016b (BERT), etc., or as their own numbered ADRs depending on scope.

## Hybrid-pipeline question

Even if Q1 says "no Qwen-on-ANE", a **hybrid path** is interesting:

- **mmproj/ViT** runs on ANE via `coreml-native`, producing image embeddings handed to the LLM body.
- **LLM body (Qwen3.5-MoE / Gemma-4)** runs on Metal GPU via `mlx-native`, consuming those embeddings.
- ANE and GPU run **in parallel** during multimodal generation, exploiting Apple Silicon's heterogeneous compute.

This is the sweet spot the M5 Max architecture suggests but no upstream framework currently exposes.  Worth pursuing once ADR-015 closes the LLM decode gap.

## Decision criteria for "use mlx-native vs coreml-native"

Until ADR-016's two-day spike happens, this table is the operating heuristic:

| Workload property | Reach for `mlx-native` | Reach for `coreml-native` |
|---|---|---|
| Custom quantization (DWQ, GGUF Q4_K, our own) | ✅ | ❌ |
| MoE / novel arch / DeltaNet / hybrid attention | ✅ | ❌ |
| Need bit-exact parity vs llama.cpp / mlx-lm | ✅ | ❌ |
| Standard ViT / BERT / Whisper / classic transformer | ⚠️ acceptable but not the win | ✅ |
| Stateless one-shot prediction (no KV cache) | ⚠️ | ✅ |
| Want ANE acceleration (heterogeneous compute) | ❌ | ✅ |
| Need fine-grained Metal kernel control | ✅ | ❌ |
| `.mlmodelc` / `.mlpackage` format already on disk | ⚠️ | ✅ |

## Non-goals of ADR-016 itself

- ADR-016 does **not** commit to a CoreML pivot for Qwen / Gemma; it commits to *measuring* whether one is viable.
- ADR-016 does **not** override ADR-008 ("full candle divorce" — pure Rust, mlx-native sole inference backend).  CoreML is an additional sibling, not a replacement.

## Phasing (high level — to be expanded if/when ADR-016 is opened for execution)

| Phase | Deliverable |
|---|---|
| **P1 — coremltools conversion spike** | Convert Qwen3.5-3B (smallest in family) to `.mlmodelc`; measure n=256 decode wall-clock on M5 Max with `ComputeUnits::All`.  Compare to mlx-native baseline.  Decision: pursue Q1 further, or close it. |
| **P2 — mmproj / ViT spike** | Convert one Gemma-4V mmproj to `.mlmodelc`; measure prefill wall-clock vs current mlx-native ViT path.  If ANE wins ≥1.5×, schedule ADR-016a. |
| **P3 — embeddings spike** | Convert BERT (`bert-test` target) to `.mlmodelc`; measure embed wall-clock vs current mlx-native BERT path.  If ANE wins, schedule ADR-016b. |
| **P4 — Hybrid-pipeline design** | If P1 says no but P2/P3 say yes: design the multimodal split (ANE prefill + GPU LLM body), with explicit dispatch handoff. |

## References

- `/opt/coreml-native/README.md` — pure-Rust CoreML bindings; ANE / GPU / CPU compute-unit selection; stateful prediction via `MLState`
- `/opt/hf2q/docs/ADR-008-candle-divorce.md` — pure-Rust mandate (mlx-native is sole backend for current models; coreml-native is sibling, not replacement)
- ADR-015 §Why-ADR-015-stays-in-mlx-native — the immediate "why not now for our hot path"
- Memory pin: `feedback_hf2q_sovereignty` ("Pure Rust, mlx-native is the only sibling dep")
- Memory pin: `project_mlx_native_crate` ("Separate pure-Rust crate at /opt/mlx-native for Metal GPU compute, following coreml-native pattern")

## Status note

This ADR is a **stub** — it sketches the strategic question, defines the decision criteria, and outlines the phasing.  No code change lands under ADR-016 until it is moved to **Accepted** status with an explicit go/no-go on Q1.
