# ADR-016: coreml-native opportunistic encoder offload — P2 ViT + P3 BERT, MLComputePlan-gated

- **Status:** Proposed-Accepted (narrow scope per research dossier 2026-04-26).  Was *Stub* until the dossier graduation.
- **Date:** 2026-04-26 (initial stub); 2026-04-26 (graduated to Proposed-Accepted via dossier + deep technical research)
- **Authors:** Robert E. Lee + Claude Code
- **Predecessor:** ADR-016 *Stub* of the same date (full strategic-comparison framing).  See git history.
- **Successor of (potential):** ADR-015 if the perf ceiling on mlx-native + Metal GPU proves insufficient for shippability, OR runs in parallel as a hybrid path
- **Sibling crates:** `/opt/mlx-native` (Metal-GPU compute, including M5 Neural Accelerators via Metal 4 TensorOps), `/opt/coreml-native` (CoreML inference: CPU/GPU/ANE)
- **Authoritative evidence:** [`/opt/hf2q/docs/ADR-016-research-dossier-2026-04-26.md`](./ADR-016-research-dossier-2026-04-26.md) — 5,846 words, 100 verified URLs, 7-researcher CFA dual-research swarm.

## Why this is its own ADR

ADR-015 closes the qwen35 / gemma decode gap **within the existing mlx-native + Metal GPU substrate** (single-CB rewrite + Rust-orchestration sweep + per the new §"Capturing M5's GPU Neural Accelerators" infusion, M5-tuned Metal 4 TensorOps for prefill).  Whether a CoreML/ANE path could deliver larger wins for the LLM hot path — or where it could complement mlx-native by offloading non-LLM-body workloads — is a strategic comparison whose evidence required substantial 2026-cutoff literature work that did not belong in the ADR-015 implementation scope.

That evidence has now been gathered.  The strategic comparison is closed; ADR-016's operative scope is the *narrow execution plan* it implies.

## Status update — 2026-04-26 (graduated via research dossier)

The 7-researcher CFA dual-research swarm (6 Claude territory-split + 1 Codex orthogonal-bias) produced [the dossier](./ADR-016-research-dossier-2026-04-26.md) on 2026-04-26.  Its load-bearing conclusions:

1. **Q1 (Qwen3.5-MoE / Gemma-4 LLM body on ANE) is closed: HARD NO-GO.**  Six convergent reasons — DWQ Q4_K/Q6_K cannot round-trip into CoreML quant; MoE expert-routing gather + Gated DeltaNet SSM force CPU fallback; ~0.095 ms IOSurface/XPC × ~10 chunks per Qwen3.5-35B token = ~21 tok/s ceiling before compute; ANEMLL caps at 8B dense and disclaims MoE; Orion (arXiv:2603.06728) GPT-2 124M ANE 170 tok/s **loses to CPU** 283 tok/s on M4 Max; Apple's own ML Research blog deliberately targets GPU not ANE for Llama-3.1-8B because LLM body is bandwidth-bound.  Dossier §3.

2. **The empirical falsification (AtomGradient `hybrid-ane-mlx-bench`, March 2026) closes the public-CoreML+MLX hybrid path for Qwen-class.**  On macOS 26.3, CoreML `ComputeUnits::All` routes Qwen3.5 prefill to GPU not ANE (ANE power ~0 W); CoreML+MLX hybrid TTFT slower than pure MLX-GPU; no 9B crossover.  Scope limit: M2 Ultra not M5 Max, non-peer-reviewed.  Dossier §2.

3. **The strategic reframe.**  *Not* "can CoreML replace mlx-native for Qwen / Gemma?"  *Instead* "**Can `coreml-native` opportunistically offload non-LLM-body encoder work (mmproj/ViT, BERT, ASR) while `mlx-native` keeps the decode critical path?**"  Matches ADR-016's existing sibling-crate structure, preserves ADR-008's pure-Rust commitment.  Dossier §10.

4. **Apple's M5 4× AI compute headline lives in the GPU, not the ANE.**  M5 introduces per-GPU-core Neural Accelerators (NAX) accessed via Metal 4 TensorOps via MLX/mlx-native.  The standalone 16-core ANE is "faster" per Apple but with no published TOPS comparison vs M4.  *That path is ADR-015 territory*, not ADR-016 — see ADR-015 §"Capturing M5's GPU Neural Accelerators".  Dossier §1.

5. **MLComputePlan must be the placement gate.**  Codex orthogonal-bias check surfaced this; none of the six Claude territory-split researchers had it.  Per-op compute-device-usage + estimated cost reporting BEFORE wall-clock benchmarking.  Available macOS 14.4+ — covers M3 Max on Sonoma, no Sequoia required.  Dossier §6.

## Two-phase question — resolved

**Q1 — Is there a Qwen3.5-MoE / Gemma-4 forward pass that runs efficiently through CoreML on M5 Max's ANE?**  *Closed.* Hard no-go.  Six convergent reasons in dossier §3.  Reopen only with new evidence (e.g., a 2027+ Apple ANE generation that publishes a comparable TOPS jump and exposes new public APIs).

**Q2 — Where *does* CoreML/ANE win on adjacent workloads we already ship?**  *This ADR's narrow scope.*  Strong candidates per dossier §4:

- **mmproj / ViT prefill** in vision models (Gemma-4V, Qwen vision) — P2 below.
- **BERT-style embeddings** (the `bert-test` target in `models/bert-test/`) — P3 below.
- **ASR / Whisper-class** — strongest shipping ANE evidence (WhisperKit MLState 45 % latency reduction on M3 ANE; FluidInference Parakeet 110–190× RTF on M4 Pro).  Dossier §4.3 honourable-mention.

## Hybrid-pipeline question — deferred to ADR-017

The concurrent CoreML-prefill + MLX-decode hybrid is closed (AtomGradient).  The **sequential ANE-encoder → mlx-native-GPU-body** hybrid remains plausible after P2/P3 produce real ANE placement and a measured handoff cost.  The Metal 4 ML passes path (CoreML model on the GPU timeline inside a Metal command buffer, dossier §5.3) is a third path that bypasses ANE entirely.  Both are deferred to a successor ADR (working name **ADR-017 — multimodal hybrid pipeline**) that opens only after P2 and P3 in this ADR clear their MLComputePlan gates.

## Decision criteria for "use mlx-native vs coreml-native"

| Workload property | Reach for `mlx-native` | Reach for `coreml-native` |
|---|---|---|
| Custom quantization (DWQ, GGUF Q4_K, our own) | ✅ | ❌ |
| MoE / novel arch / DeltaNet / hybrid attention | ✅ | ❌ |
| Need bit-exact parity vs llama.cpp / mlx-lm | ✅ | ❌ |
| Need M5 GPU Neural Accelerators (Metal 4 TensorOps) | ✅ | ❌ |
| Standard ViT / BERT / Whisper / classic transformer | ⚠️ acceptable but not the win | ✅ |
| Stateless one-shot prediction (no KV cache) | ⚠️ | ✅ |
| Want ANE acceleration for power-efficiency | ❌ | ✅ |
| Need fine-grained Metal kernel control | ✅ | ❌ |
| `.mlmodelc` / `.mlpackage` format already on disk | ⚠️ | ✅ |

Row 4 (M5 Neural Accelerators) is the clarifying addition vs the original stub: the 4× AI compute headline is GPU-side, not ANE-side.

## Decisions

**D1.** **MLComputePlan placement gate is mandatory** for every P2 / P3 spike.  Per-op `MLComputePlan.computeDeviceUsageForMLProgramOperation:` + `estimatedCostOfMLProgramOperation:` MUST report ≥ **90 % ANE op fraction AND ≥ 90 % ANE cost fraction** before any wall-clock benchmark fires.  If either fraction is below threshold, **fail-fast** and re-check model conversion (compute units, op-coverage gaps) — do NOT measure wall-clock and "see what happens".  This prevents the AtomGradient failure mode (50–97 minutes spent benchmarking before discovering ANE never fired via power readings).  Available macOS 14.4+; matches `coreml-native`'s minimum target.

**D2.** **P0 — coreml-native binding additions, in three independent PRs.**  PR-1 lands first (prerequisite for any spike); PR-2 + PR-3 land in parallel.  Detailed scope in §"P0 — coreml-native binding additions" below.

**D3.** **P2 — ViT / mmproj on ANE.**  Concrete first model: a Gemma-4V mmproj export.  Conversion via `coremltools 8.3+` to `.mlmodelc` with `mlprogram` format, FP16 precision, `EnumeratedShapes` for sequence dimension.  Compute units: `CpuAndNeuralEngine` (forces ANE consideration; production may use `All`).  Required: D1 MLComputePlan gate green BEFORE wall-clock.  Exit criterion: ANE wall-clock prefill ≥ 1.5× current mlx-native ViT path on cold M5 Max.

**D4.** **P3 — BERT / embeddings on ANE.**  Concrete first model: the `bert-test` target in `models/bert-test/`.  Same conversion methodology as P2.  D1 gate green.  Exit criterion: ANE embed wall-clock ≥ 2.0× current mlx-native BERT path on cold M5 Max.

**D5.** **Q1 (Qwen3.5-MoE / Gemma-4 on ANE) closed.**  Do NOT reopen without new evidence: a fresh Apple ANE generation announcement, a public API change (e.g., direct `_ANEClient`-style access), OR a peer-reviewed measurement on M5+ silicon that breaks Orion's "ANE GPT-2 124M loses to CPU" finding.

**D6.** **P4 sequential hybrid** (ANE encoder → mlx-native GPU LLM body) **deferred to ADR-017**.  Opens only after both P2 and P3 here have D1 gate green AND wall-clock exit-criteria green.

**D7.** **P5 ASR / Whisper-class** — added as honourable-mention spike per dossier §4.3.  Strongest shipping ANE evidence in the entire 2026 literature (WhisperKit + FluidInference Parakeet).  Lower priority than P2/P3 because the model class is not currently in hf2q's roadmap, but a 2-week spike would establish coreml-native's audio-encoder readiness for any future ASR product.

## P0 — coreml-native binding additions (3 PRs against `/opt/coreml-native`)

Source: technical brief at `/tmp/cfa-adr016/research-coreml-bindings.md` (2026-04-26).  All API signatures verified against MacOSX15.4.sdk + MacOSX26.4.sdk headers and against the local `objc2-core-ml 0.3.2` generated source — no training-data recall used.

### P0/PR-1 — `MLComputePlan` placement-report module

- **Scope:** New file `src/compute_plan.rs`.  Public API:
  ```rust
  pub struct ComputePlanReport {
      pub ane_op_fraction: f64,       // ops with preferredComputeDevice == ANE
      pub ane_cost_fraction: f64,     // sum of cost.weight for ANE-preferred ops
      pub total_ops: usize,
      pub ane_ops: usize, pub gpu_ops: usize, pub cpu_ops: usize,
      pub op_details: Vec<OpDeviceRecord>,  // name, preferred device, cost
  }
  pub fn compute_plan_report(model_path: &Path, config: &ModelConfig)
      -> impl Future<Output = Result<ComputePlanReport>>;
  ```
- **Files touched:** `src/compute_plan.rs` (new), `src/lib.rs` (re-export), `Cargo.toml` (10 new feature flags on `objc2-core-ml`: `MLComputePlan`, `MLComputePlanCost`, `MLComputePlanDeviceUsage`, `MLComputeDeviceProtocol`, `MLModel_MLComputeDevice`, `MLModelStructure`, `MLModelStructureProgram`, `MLModelStructureProgramFunction`, `MLModelStructureProgramBlock`, `MLModelStructureProgramOperation`).
- **Min macOS:** 14.4 (`MLComputePlan` SDK availability).  Already inside coreml-native's overall macOS 12+ support bracket.
- **Async-bridge:** reuse the existing `CompletionFuture` pattern in `src/async_bridge.rs` (load is completion-handler-only, no synchronous init).
- **Test plan:** integration test loads `tests/fixtures/test_linear.mlmodelc`, asserts report returns without panic, asserts `total_ops > 0` and `ane_op_fraction ∈ [0.0, 1.0]`.
- **Implementation risk (open Q1):** the concrete `MLNeuralEngineComputeDevice` / `MLGPUComputeDevice` / `MLCPUComputeDevice` device-type structs may not be auto-generated in `objc2-core-ml 0.3.2`.  Fallback: classify via `objc2::runtime::AnyClass::get("MLNeuralEngineComputeDevice")` + `isKindOfClass:`.  Resolve before merge; emit `ComputeDevice::Unknown` with a `// TODO:` only as a last resort.
- **Unlocks:** P1 smoke test (Qwen3.5-0.8B placement report), P2/P3 fail-fast gate.

### P0/PR-2 — `MLPredictionOptions.outputBackings` (zero-copy output binding)

- **Scope:** `Model::predict_with_options()` + `PredictionOptions` builder.
  ```rust
  pub struct PredictionOptions {
      output_backings: Vec<(String, OwnedTensor)>,
  }
  impl PredictionOptions { pub fn new() -> Self; pub fn with_output_backing(mut self, name: &str, tensor: OwnedTensor) -> Self; }
  impl Model {
      pub fn predict_with_options(&self, inputs: &[(&str, &dyn AsMultiArray)],
                                  options: &PredictionOptions) -> Result<Prediction>;
  }
  ```
- **Critical caveat (gotcha):** `outputBackings` is a *proposal*, not a command.  Apple's `MLPredictionOptions.h`: *"The framework may not use the specified backing object … if … the model doesn't support the user allocated buffers"*.  We MUST identity-compare the returned output array against the backing after each prediction; expose `Prediction::backing_was_used(&self, name: &str) -> bool`.  When the backing is ignored, we fall through to the existing stride-aware `copy_array_to_f32` slow path.
- **Files touched:** `src/lib.rs` (new method on `Model`), new `src/options.rs`, `Cargo.toml` (no new feature flags — `MLPredictionOptions` already enabled).
- **Buffer alignment:** `aligned_alloc(vm_page_size, round_page(size))` per Apple's documented best-practice.
- **MTLBuffer→MLMultiArray:** does **not** exist in the public API.  Two zero-copy paths only: `initWithDataPointer:` (page-aligned heap) or `initWithPixelBuffer:` (IOSurface-backed CVPixelBuffer, macOS 12+).
- **MLState interaction:** `MLPredictionOptions` is NOT accepted by `predictionFromFeatures:usingState:error:` — stateful (KV-cache) and outputBackings are mutually exclusive in the public API.  Not a constraint for P2/P3 (both are non-stateful encoders).
- **Min macOS:** 11.0 (already covered).
- **Test plan:** smoke test with `test_linear.mlmodelc`; identity-check verifies the backing-was-used flag.
- **Unlocks:** P4 (ADR-017) sequential-hybrid handoff.

### P0/PR-3 — `MLModelConfiguration.preferredMetalDevice` (shared MTLDevice handle)

- **Scope:** Add `preferred_metal_device: Option<MetalDeviceHandle>` to a new `ModelConfig` (or extend the existing config struct).  `MetalDeviceHandle::system_default() -> Option<Self>` for the single-GPU M5 Max common case.
- **Two-repo dependency (open Q7):** `mlx-native` does NOT currently expose its underlying `MTLDevice` handle.  PR-3 cannot land production-ready until **`mlx-native` adds `metal_device() -> MetalDeviceHandle`**.  Track as a precondition; PR-3 itself can land with `MetalDeviceHandle::system_default()` only, and the `mlx-native`-handle bridging is a follow-up after the mlx-native API exists.
- **Honor-rate caveats:** `preferredMetalDevice` does NOT affect ANE routing — only CoreML's GPU fallback paths.  On single-GPU M5 Max (one internal GPU), `MTLCreateSystemDefaultDevice()` and CoreML's automatic selection produce the same device, so the property's practical value is most significant when ADR-017's sequential-hybrid pipeline fires alongside mlx-native's existing Metal session and the two would otherwise pick different `MTLDevice` instances.
- **Files touched:** `src/lib.rs` (new method `Model::load_with_config`), new `src/options.rs` shared with PR-2 (or separate file), `Cargo.toml` (add optional `objc2-metal = { version = "0.3", optional = true }` + `metal` feature flag, and `objc2-core-ml` `objc2-metal` feature).
- **Min macOS:** 10.15 (well within crate support).
- **Breaking-change risk:** minimal if `Model::load` signature is preserved and a parallel `Model::load_with_config` overload is added.

**PR ordering:** PR-1 → (PR-2 ‖ PR-3).  PR-2 and PR-3 are independent and can land in either order or in parallel after PR-1.  Bundling all three into one mega-PR would block the highest-value addition (PR-1, the gate) on the `objc2-metal` dep chain in PR-3.

## D1 — MLComputePlan placement gate (mandatory workflow for P2 / P3)

```rust
// After P0/PR-1 lands, every P2/P3 spike begins with this 4-step gate
// BEFORE any wall-clock measurement.

// Step 1 — convert model to .mlmodelc (Python-side, one-shot)
//   coremltools 8.3+, mlprogram format, FP16, EnumeratedShapes for seq dim
//   xcrun coremlc compile <model>.mlpackage <out>/

// Step 2 — load MLComputePlan and emit report
let report = coreml_native::compute_plan_report(
    Path::new("vit_encoder.mlmodelc"),
    &ModelConfig { compute_units: ComputeUnits::CpuAndNeuralEngine, ..Default::default() },
).await?;

println!("ANE op fraction:   {:.1}% ({}/{} ops)",
    report.ane_op_fraction * 100.0, report.ane_ops, report.total_ops);
println!("ANE cost fraction: {:.1}%", report.ane_cost_fraction * 100.0);
println!("GPU/CPU spillover: {} GPU ops, {} CPU ops",
    report.gpu_ops, report.cpu_ops);

// Step 3 — fail-fast assertion BEFORE any wall-clock benchmark
const ANE_OP_THRESHOLD: f64   = 0.90;  // ≥ 90% of ops on ANE
const ANE_COST_THRESHOLD: f64 = 0.90;  // ≥ 90% of compute cost on ANE
assert!(report.ane_op_fraction   >= ANE_OP_THRESHOLD,
    "GATE FAIL: ANE placement {:.1}% < required {:.0}%; check op coverage / re-convert with CpuAndNeuralEngine",
    report.ane_op_fraction * 100.0, ANE_OP_THRESHOLD * 100.0);
assert!(report.ane_cost_fraction >= ANE_COST_THRESHOLD,
    "GATE FAIL: ANE cost {:.1}% < required {:.0}%; some hot ops fell to GPU/CPU",
    report.ane_cost_fraction * 100.0, ANE_COST_THRESHOLD * 100.0);

// Step 4 — only NOW run wall-clock benchmark
//   Cold-SoC per `feedback_perf_gate_thermal_methodology`: cold run first,
//   thermal-stable run second; 3 cold runs, median reported.
```

**Gate-vs-production compute-units gotcha (open Q6):** the gate measurement uses `CpuAndNeuralEngine` to force ANE consideration.  If production inference uses `ComputeUnits::All` (the crate default), CoreML may still route some ops to GPU.  The gate fraction measured under `CpuAndNeuralEngine` is therefore an *upper bound* on production ANE utilization.  The exit-criteria wall-clock numbers MUST be measured under the same compute-units configuration as production, not under the gate's `CpuAndNeuralEngine`.

**Why this prevents AtomGradient's failure mode:** AtomGradient spent ~50–97 min on seq256/seq512 Qwen3.5 conversion + benchmarking and only discovered ANE never fired via powermetrics.  MLComputePlan reports placement at compile time — the gate exits in under 1 second before any inference runs.  Power readings remain useful as corroboration (post-gate), but they are not the inspectable signal — MLComputePlan is.

## Phasing

| Phase | Deliverable | Definition of done |
|---|---|---|
| **P0/PR-1 — MLComputePlan binding** | `src/compute_plan.rs` in coreml-native; `compute_plan_report()` async fn; `ComputePlanReport` + `OpDeviceRecord` types; 10 new objc2-core-ml feature flags; integration test on `test_linear.mlmodelc`. | Builds.  Test passes on M3 Max + M5 Max.  Concrete device-type classification working (or documented `Unknown` fallback per open Q1). |
| **P0/PR-2 — outputBackings** | `predict_with_options()` + `PredictionOptions`; identity-check exposed via `Prediction::backing_was_used()`. | Builds.  Smoke test verifies the backing-used flag round-trips correctly. |
| **P0/PR-3 — preferredMetalDevice** | `Model::load_with_config()` + `MetalDeviceHandle::system_default()`; optional `objc2-metal` dep; `metal` feature flag. | Builds with and without `metal` feature.  `MetalDeviceHandle::system_default()` returns `Some` on Apple Silicon. |
| **P0-precondition — mlx-native MTLDevice export** | New `mlx_native::metal_device() -> MetalDeviceHandle` public API.  Two-repo PR. | mlx-native exports the handle; coreml-native PR-3 follow-up wires it through. |
| **P2 — Gemma-4V mmproj on ANE** | Convert mmproj from float HF checkpoint to `.mlmodelc` (FP16, mlprogram, EnumeratedShapes for image-token-count buckets); D1 MLComputePlan gate green; cold-SoC ANE wall-clock vs current mlx-native ViT path. | D1 gate ≥ 90 % ANE ops + ≥ 90 % ANE cost.  Wall-clock prefill ≥ 1.5× current mlx-native ViT (3 cold runs, median, M5 Max). |
| **P3 — `bert-test` on ANE** | Convert BERT test target to `.mlmodelc` (FP16, mlprogram, EnumeratedShapes for seq dim); D1 gate green; cold-SoC ANE wall-clock vs current mlx-native BERT path. | D1 gate green.  Wall-clock embed ≥ 2.0× current mlx-native BERT (3 cold runs, median, M5 Max). |
| **P5 — ASR / Whisper-class spike (honourable mention)** | Convert Whisper-Large-v3-Turbo encoder to `.mlmodelc`; D1 gate; wall-clock vs CPU baseline. | D1 gate green.  Wall-clock encoder ≥ 3× CPU baseline (matches WhisperKit's published 65 % latency reduction on M3). |
| **P-close — feed evidence into ADR-017** | Sequential-hybrid (ANE encoder → mlx-native GPU LLM body) and Metal 4 ML passes path opens as ADR-017 only after P2 and P3 clear. | Dossier §5 + ADR-017 successor opens with the measured handoff cost from P2/P3. |

## Open questions / risks

| ID | Risk / open question | Mitigation |
|---|---|---|
| **Q1** | Concrete device-type structs (`MLNeuralEngineComputeDevice` etc.) may not be auto-generated in `objc2-core-ml 0.3.2`. | PR-1 implementation falls back to `AnyClass::get("MLNeuralEngineComputeDevice") + isKindOfClass:`.  Resolve before merge. |
| **Q2** | Does ANE actually honour `outputBackings` for FP16 `MLMultiArray` on real `mlprogram` ViT/BERT models, or fall through to internal buffer? | Live test on M5 Max during P2 spike.  Identity-check determines whether PR-2 delivers real zero-copy or best-effort hint. |
| **Q3** | macOS 26 (Tahoe) ANE stride regressions reported in Apple Developer Forums (dossier §6 forum citations).  Are P2/P3 outputs row-major even with backing pre-formatted? | macOS 26 CI run during P2 spike.  If strided despite backing, the existing `copy_array_to_f32` slow path takes over. |
| **Q4** | Does `MLNeuralEngineComputeDevice.totalCoreCount` distinguish M3 ANE vs M4 ANE vs M5 ANE? | Read via the binding; not a blocker but useful for documenting which silicon the gate ran on. |
| **Q5** | What's the actual Gemma-4V mmproj op coverage on ANE?  CoreML-LLM (john-rocky) reports its Gemma 4 E2B vision encoder runs on `cpuAndGPU` deliberately — *not* ANE.  This is a direct counter-warning that ViT-on-ANE is not automatic. | D1 gate catches this; if D1 fails on the first conversion, iterate the conversion (alternate ops, simpler attention pattern) or close P2 as no-go and document why. |
| **Q6** | `CpuAndNeuralEngine` (gate) vs `All` (production) compute-units divergence (above). | Exit-criteria wall-clock measured under same compute-units as production, not under the gate's. |
| **Q7** | mlx-native MTLDevice export precondition for PR-3. | Two-repo PR.  PR-3 lands with `system_default()` only; mlx-native handle bridging is a follow-up. |
| **Q8** | Does Apple FoundationModels' on-device LLM (~3B, 3.7 bpw, iOS 18+/macOS 15+) compete with this scope? | Out of scope.  FoundationModels is closed (no `.mlmodelc` exposed); we cannot inspect or replicate it.  Track as adjacent reference data only. |

## Non-goals

1. **Q1 reopen.**  Pure CoreML/ANE replacement of the LLM hot path is closed.  Out of scope until new evidence (D5).
2. **Concurrent CoreML+MLX hybrid for the LLM body.**  Falsified by AtomGradient.  Out of scope.
3. **Sequential ANE-encoder + mlx-native-GPU-body.**  Deferred to ADR-017 (post-P2/P3).
4. **Metal 4 ML passes (CoreML on GPU timeline).**  Deferred to ADR-017 §3rd-path.
5. **M5 Max standalone ANE TOPS investigation.**  Apple hasn't published it.  Not blocking any decision here.
6. **MoE-on-ANE.**  Closed via D5.  Even with op coverage parity (which doesn't exist), the 2.3 ms XPC dispatch ceiling makes Qwen3.5-35B-class MoE non-viable on ANE.

## References

- [`/opt/hf2q/docs/ADR-016-research-dossier-2026-04-26.md`](./ADR-016-research-dossier-2026-04-26.md) — 5,846-word dossier; 100 verified URLs; 84 inline `[R2]–[R8]` audit-trail markers.  Authoritative evidence base.
- `/tmp/cfa-adr016/research-coreml-bindings.md` — deep technical brief driving §P0 (3-PR plan, all signatures verified against MacOSX15.4.sdk + objc2-core-ml 0.3.2 generated source, 2026-04-26).  Also stored at memory key `swarm-cfa-adr016/deep-research/coreml-bindings`.
- `/opt/coreml-native/README.md` + `/opt/coreml-native/src/` — current binding surface.  Verified gaps (zero references): `setOutputBackings`, `setPreferredMetalDevice`, `MLComputePlan`, `predictionFromFeatures_options_error`.
- `/opt/coreml-native/docs/research-native-rust-coreml.md` — 2026-03-23 prior research; covers `MLPredictionOptions` and `MLState` but says "not yet exposed" — this ADR is the "now exposed" plan.
- ADR-008 — pure-Rust mandate (mlx-native is sole backend for current LLM hot path; coreml-native is sibling, not replacement).
- ADR-015 — sibling perf ADR; §"Capturing M5's GPU Neural Accelerators (Metal 4 TensorOps + NAX)" handles the 4× AI compute headline that lives in the GPU pathway, not the ANE pathway.
- WWDC25 session 262 — "Combine Metal 4 machine learning and graphics" (`developer.apple.com/videos/play/wwdc2025/262/`).
- Apple Developer Tech Talk 111432 — "Accelerate your machine learning workloads with the M5 and A19 GPUs" (`developer.apple.com/videos/play/tech-talks/111432/`).
- Apple ML Research M5/MLX post — `https://machinelearning.apple.com/research/exploring-llms-mlx-m5`.
- Memory pins: `feedback_perf_gate_thermal_methodology` (cold-SoC bench discipline), `feedback_hf2q_sovereignty` (Pure Rust), `project_mlx_native_is_the_strategic_destination`, `feedback_correct_outcomes` (no shortcuts), and the new `project_adr016_dossier_completed`, `project_m5_two_tier_ai_architecture`, `feedback_codex_invocation_correctness`.

## Status note

This ADR moves from **Stub → Proposed-Accepted (narrow scope)** with the explicit go/no-go on Q1 (closed: no-go) and the narrow execution path of P0 → P2 → P3 → ADR-017.  P4 hybrid pipeline does NOT land under ADR-016 and is the responsibility of its successor.

## Changelog

- **2026-04-26 — Stub (initial).** Strategic-comparison framing with Q1 / Q2 / hybrid open questions and P1–P4 phasing placeholders.
- **2026-04-26 — Graduated to Proposed-Accepted (narrow scope).** 7-researcher CFA dual-research swarm produced the dossier; 2 deep technical research briefs (Metal 4 TensorOps + CoreML bindings) verified all API claims against SDK headers / objc2-core-ml generated source / MLX commits.  Q1 closed (HARD NO-GO); P4 deferred to ADR-017; P0 (3-PR coreml-native bindings) + D1 (MLComputePlan placement gate) added as concrete execution plan; P2 + P3 narrowed with cold-SoC exit criteria; P5 ASR honourable-mention added.  Original §"Why this is its own ADR" + decision-criteria table preserved with the Metal-4-TensorOps row added.  Reference dossier graduates ADR-015's acknowledged "Apple Silicon literature gap".
