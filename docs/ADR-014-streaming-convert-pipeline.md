# ADR-014: Streaming convert pipeline + peer-parity gates (cross-arch)

**Status:** 🟡 **PROPOSED 2026-04-25 (round-2 refined)** — pending Robert sign-off; refined across two party-mode sessions (round 1: `conversion_fixes`, ten questions, eight strategic axes; round 2 today: 12 additional Robert-locked refinements — see "Round-2 refinement log" right after the Phase status table). Ready for P0 to start **after ADR-012 P9 real close** (R14 — the four `models/qwen3.6-*-dwq*` GGUFs must verifiably load in `llama-cli` first). Per mantra (`~/Documents/mantra.txt`, 2026-04-07): "DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Chesterton's fence: always understand current fully before changing it." Every decision below is engineering-executable; every phase has concrete deliverables, ACs, and LOC estimates; every risk has a mitigation that is real work, not a gate. This ADR closes only when **all** peer-parity gates are measured green on the four ADR-012 reference artifacts re-emitted under the streaming pipeline.

---

## Phase status

| Phase | Status | Commit | Notes |
| ----- | ------ | ------ | ----- |
| P0 — Lazy tensor primitive + lazy safetensors reader | 🟢 code-green; on-day apex-MoE iter spike pending | `c707a2c` (lazy primitive), `038f2ab` (lazy reader + bridge), `<spike-commit>` (apex-MoE iter spike) | LazyTensor + LazyTensorMap + lazy safetensors reader land; 13 unit + 4 reader + 1296 full-bin tests pass; eager bridge byte-identical to pre-ADR-014. Decision 6 gate value locked at **36.3 GB** (33 GB ADR-012 P9b inherited + 10% headroom) per `docs/peer-parity-baselines-2026-04-26.md`. Decision 2 `≤ 8 GB` LazyTensorMap-iter spike on apex MoE pending (next iter). |
| P1 — Lift Phase 1.4–1.7 transforms to lazy | 🟢 closed | `b6a4b82` (1.4+1.42), `04ee984` (1.45), `5a773d0` (1.5+Decision 7), `295ffd5` (1.6), `ef7b550` (1.7+1.8 close) | Every Phase 1.x transform takes `&mut LazyTensorMap`. Decision 7 layer-streaming MoE merge implemented; per-merge tile materialise→quantise→write→drop bounds peak resident bytes to ~one tile (~750 MB apex BF16) instead of ~80 GB stack. 9 byte-identity unit tests + 1309 full-bin tests passing. Eager helpers retained for the P9b dance until P4 deletes both. |
| P2 — Streaming quantize loop (per-tensor write-and-drop) | 🟡 iter-1 landed; iter-2 wiring pending | `58e3144` (quantize_streaming function + LazyTensorMap::from_eager bridge) | `quantize_streaming(LazyTensorMap, ..., bf16_to_f16: bool)` consumes a LazyTensorMap one tensor at a time — materialise → optional bf16→f16 cast → quantise → accumulate → drop. Byte-identical to `quantize_model` (verified by 2 unit tests). Production wiring (cmd_convert restructure with Phase 4.5 re-read) pending iter-2; StreamingBackend trait + GgufBackend refactor pending iter-3. |
| P3 — Rayon parallelism in quantize loop | 🟢 closed | `fdd0375` (quantize_streaming_parallel) | `quantize_streaming_parallel` distributes per-tensor quantize across rayon thread pool sized `min(available_parallelism, 16)`. Byte-identical to serial across n_workers {1, 2, 4, 8}; per-tensor shape/dtype/bytes byte-equal. Worker clamp tested ({0 → 1, 100 → 16}). Memory: serial ~750 MB peak input → parallel n=8 ~6 GB / n=16 ~12 GB. |
| P4 — Eliminate the P9b intermediate-GGUF dance | ⏳ pending | — | Depends on P0+P2. Closes ADR-012 P9b's `IntermediateMoeQ8Quantizer` workaround. Gated on P2 iter-2 cmd_convert wiring. |
| P5 — Sensitivity-JSON cache (DWQ across bit-pair variants) | 🟢 module landed; wiring pending | `18333ff` (cache module) | Pure-Rust cache at `${XDG_CACHE_HOME:-$HOME/.cache}/hf2q/sensitivity/<sha>.json`. Cache key = hex SHA-256(model_sha \| corpus_sha \| algorithm_version). `SENSITIVITY_ALGORITHM_VERSION = "1.0.variance-magnitude"` — bumped on algorithm change to invalidate stale entries. Atomic write via temp + POSIX rename. 9 unit tests covering determinism, disjointness, round-trip, miss, version mismatch, atomic write, env resolution, empty layers. Wiring into DWQ calibrator pipeline lands alongside P2 iter-2 / P7. |
| P6 — Imatrix calibrator (pure-Rust port) | 🟡 algorithm + legacy I/O landed; GGUF I/O + cross-validation gate pending | `511d35c` (algorithm core), `2577d89` (legacy .imatrix I/O) | `ImatrixCollector` with dense (`GGML_OP_MUL_MAT`) + MoE (`GGML_OP_MUL_MAT_ID`) accumulators; `Stats { values, counts }` invariant; `finalise()` produces per-column importance vectors. Legacy `.imatrix` save/load byte-for-byte matches `imatrix.cpp::save_imatrix_legacy` + `load_imatrix_legacy`. 15 unit + round-trip tests. Pending: GGUF-format `.imatrix` (P6 iter-3) + cross-validation gate against `llama-imatrix` (lands at P7 close when Calibrator trait wires forward-pass). |
| P7 — `Calibrator` × `OutputFormat` orthogonal split | 🟡 trait + None impl + k-quant dequantize + Q4_K quantize landed; Imatrix/Dwq impls + Q5_K/Q6_K quantize + Layout A migration pending | `33081aa` (Calibrator trait + NoneCalibrator + CalibrationData enum), `ac3ebf2` (P7 iter-3a — Q4_K block layout + dequantize), `ade910c` (P7 iter-3c — Q5_K + Q6_K block layouts + dequantize, co-landed with parallel ADR-005 P3 iter-203 due to staged-index sweep), `ebee4e6` (P7 iter-3b1 — `nearest_int` + `make_qkx2_quants` + `quantize_row_q4_k` codebook quantize), `6440b4e` (P7 iter-3b2 — `quantize_row_q5_k` codebook quantize, reuses `make_qkx2_quants` at `nmax=31`), `1c37488` (P7 iter-3b3 — `make_qx_quants` symmetric codebook + `quantize_row_q6_k`), `b27afa7` (P7 iter-3d — `make_qkx3_quants` + `make_qp_quants` + `quantize_row_q4_k_imatrix` for imatrix-weighted Q4_K), `93415ad` (P7 iter-3e — `quantize_row_q5_k_imatrix` + `quantize_row_q6_k_imatrix`, completes the imatrix-weighted Q4/5/6_K coverage), `17def7e` (P7 iter-3f — flat-bytes wrappers `quantize_row_q*_k[_imatrix]_to_bytes` for direct GGUF emission), `c9c9d51` (P7 iter-3g — `k_quant_codec` calibration-aware dispatch over Q4/5/6_K with `KQuantTarget` enum + `quantize_row_to_bytes(row, target, calib, name)` entry point), `dd9cec3` (P7 iter-3h — `q_legacy` module: Q8_0 + Q4_0 ports for the K-family fallback chain), `61f6d3e` (P7 iter-3i — Q5_0 + Q5_1 ports complete the legacy fallback chain), `5e749a2` (P7 iter-3j — k_quant_codec extended to dispatch over legacy formats; 7-target enum), `253a0cd` (P7 iter-3k — end-to-end integration tests at realistic tensor shapes 4096+16384 with 7-format ordering verification), `b260917` (P7 iter-3l — `quantize_tensor_2d_to_bytes` multi-row helper for full weight matrices), `3115a75` (P7 iter-3m — `KQuantCodecQuantizer` impl wires the codec into the existing `Quantizer` trait machinery), `d2bdc37` (P7 iter-3n — Q4_1 port completes the legacy block-format coverage), `e2a6fd0` (P7 iter-3o — codec & quantizer extended with Q4Legacy1 variant for full 8-target coverage) | `Calibrator` trait (Send + Sync + object-safe) + `CalibrationData` enum (None / Imatrix / ImatrixWithStats / Dwq) + `CalibrationCorpus` + `NoneCalibrator` truly no-op impl + 5 typed `CalibrationError` variants. 7 unit tests. P7 iter-3 k-quant codebook port: **Q4_K + Q5_K + Q6_K block layouts (`repr(C)` byte-for-byte match against `block_q*_K` in `ggml-common.h`) + `dequantize_row_q4_k` / `dequantize_row_q5_k` / `dequantize_row_q6_k` (pure-Rust ports of `ggml-quants.c:1467`/`:1669`/`:1877`) + `nearest_int` bit-trick (`:559`) + `make_qkx2_quants` codebook search (`:737`) + `quantize_row_q4_k` (`:1395`)**. 31 k-quant unit tests passing including round-trip RMSE bound (synthetic ramp & multi-block) ≤ 0.05 for Q4_K. Pending: ImatrixCalibrator + DwqCalibrator impls (P7 iter-2-bis); `quantize_row_q5_k` + `quantize_row_q6_k` (P7 iter-3b2/3); byte-identity gate against llama.cpp NEON path on `aarch64-apple-darwin` via stored fixture (P7 iter-3b4, Decision 11 round-2); Layout A path migration (P7 iter-4). |
| P8 — CLI rename + final variant menu | ⏳ pending | — | Depends on P7. Breaking change — no aliases per Q9. |
| P9 — Safetensors backend integration with calibrators | ⏳ pending | — | Depends on P7. Removes `requires_native_quantization` shortcut. |
| P10 — Peer-parity benchmark harness (llama.cpp + mlx-lm) | ⏳ pending | — | Depends on P3+P6+P9. |
| P11 — Re-emit ADR-012's four DWQ GGUFs under streaming pipeline + measured gate close | ⏳ pending | — | Depends on every prior phase. Closure AC. |
| P12 — Documentation refresh (`converting-a-model.md`, `converting-qwen35.md`, `shipping-contract.md`) | 🟡 ADR Phase status table refreshed inline | — | This iter (2026-04-26): refreshed Phase status table with current commit hashes for P0–P7 landed work. End-user docs (`converting-a-model.md` rewrite, `converting-qwen35.md` update, `shipping-contract.md` peer-parity gates section, new `calibrator-onboarding.md`) land at P12 close after P11. |

ADR-014 closes only when P11 is green: all four reference artifacts re-emitted, every peer-parity gate measured and passed, every comparison number recorded inline in the closing commit.

### Round-2 refinement log (2026-04-25, party-mode session 2)

Twelve Robert-locked refinements applied this session. Citations point to the body where each is now codified:

1. **D22** — P11 ships only the four ADR-012 carry-over DWQ GGUFs + four DWQ safetensors twins to `models/`. Imatrix-q4_k_m gate cells produce ephemeral measurement files only.
2. **D11** — k-quant byte-identical gate is against llama.cpp's NEON code path on `aarch64-apple-darwin`, not the scalar reference.
3. **D6 + P0** — Decision 6's literal `≤ 35 GB` replaced with `measured + 10% headroom` derived from a P0-prerequisite measurement spike on apex MoE activation capture.
4. **Phase plan intro** — P11/P12 may parallelise via separate CFA worktrees (`feedback_swarm_sequential_when_shared_build.md`'s sequential rule applies to shared `target/`; CFA splits it).
5. **D16** — PPL eval corpus is the **full ~280k-token wikitext-2 test split**, not 512 tokens. Apples-to-apples vs published peer numbers.
6. **D15** — Speed/RSS protocol replaced: `1 warmup run discarded → 60 s thermal cooldown → 3 timed runs, median wins`. Both peers warm. Sidesteps Metal-shader-cache persistence question.
7. **D12** — All 17 named CLI variants ship in this ADR. Full menu is the deliberate long-term scope.
8. **D21** — Sovereignty rule is runtime-only. Test-time Python (mlx-lm, llama-imatrix) permitted only behind `#[ignore]`-gated parity-harness tests.
9. **D5** — Decision 5's `~80 LOC` re-estimated to `~300–500 LOC honestly written`. Pipeline shape stands; correctness and speed dominate over LOC ceiling.
10. **R14** — ADR-014 P0 starts only after ADR-012's four DWQ GGUFs verifiably load in `llama-cli` (real ADR-012 close).
11. **D14** — `--format safetensors` output is an **mlx-lm-style directory** (`config.json` + tokenizer + sharded `model-*.safetensors` with quant metadata in shard headers). Appendix A is canonical.
12. **D9 + P7** — Layout A: `dwq*.rs`, `dwq_activation.rs`, `sensitivity.rs`, `apex.rs` move from `src/quantize/` into `src/calibrate/`. Full path migration; P7 LOC budget grows for import-rewrite churn.

---

## Engineering Mantra (load-bearing — read before every session)

From `~/Documents/mantra.txt` (2026-04-07):

> DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.

Operationalised for this ADR:

- **No shortcuts.** Every fix in items #1–#10 is in scope. None is punted to a follow-up perf ADR.
- **No fallback.** When a calibrator fails, the convert errors with a typed message — never silently regresses to weight-space.
- **No stub.** Every Decision body specifies the file path, the function signature, the test name, and the LOC estimate. No "wire-up later" sentences.
- **Measure 3x, cut once.** Every peer-parity gate has three measurements: hf2q current pipeline, hf2q new pipeline, peer baseline. Three numbers per cell of the AC table.
- **Chesterton's fence.** The current pipeline's eager `to_vec()` (`src/input/safetensors.rs:316`) and bulk `clone()` patterns (`src/backends/gguf.rs:897/910/926/3433`) are deliberate as of when they were written; this ADR explains *why* they were correct *then* and *why* they are wrong *now* before changing them.

---

## Context

### Business problem

hf2q is positioned as **both** a conversion tool and an inference tool, serving GGUF (for `llama.cpp` + `ollama` users) and safetensors (for `mlx-lm` users + native `mlx-native` loading). Robert (2026-04-25, party-mode session): *"We're a conversion tool and an inference tool — we need to support both."* Peer parity is therefore measured on **two** axes: GGUF output against `llama.cpp` (`convert_hf_to_gguf.py` + `llama-quantize` + imatrix), and safetensors output against `mlx_lm.convert` (DWQ-calibrated). hf2q must be measurably as fast and as correct as both peers on the same Apple Silicon hardware (M5 Max, 128 GB), or it ships as a degraded substitute.

### Technical problem

Walking the current `cmd_convert` pipeline (`src/main.rs:176–910` on `main`, `src/main.rs:255–1243` on the `worktree-adr-012-p8-p11` branch) against `convert_hf_to_gguf.py` and `mlx_lm.convert` reveals **ten** distinct inefficiencies or peer divergences. They split into two cohorts.

**Cohort A — fixable inefficiencies (correctness + perf failure modes):**

1. **P9b intermediate-GGUF dance.** `src/main.rs:680–778` (worktree). DWQ-on-qwen35 emits a full F16+Q8 intermediate GGUF to a tempdir, reopens it as a `Qwen35Model` for forward-pass activation capture, then re-reads the safetensors and re-applies Phases 1.4/1.45/1.6/1.7. Three full touches of the model weights to do one calibration. Caused by no in-memory `LazyTensorMap → ActivationCapture` path; only the GGUF reader is plumbed.
2. **Eager `to_vec()` of mmap'd safetensors.** `src/input/safetensors.rs:316` does `mmap[abs_start..abs_end].to_vec()` per tensor — the entire ~70 GB BF16 of the apex MoE lands in anonymous heap before any quantize call. `convert_hf_to_gguf.py:13203–13273` (`LazyTorchTensor`) keeps the mmap and only materialises when the per-tensor callable is invoked.
3. **Whole-model `QuantizedModel` held until write.** `src/quantize/mod.rs:129` accumulates `HashMap<String, QuantizedTensor>`; the write at Phase 4.6 (`src/main.rs:984` worktree) flushes the entire map. Peak RSS doubles through the quantize→write transition. `convert_hf_to_gguf.py`'s `Model.write_tensors()` is generator-style: convert → write → drop.
4. **No parallelism in quantize.** `src/quantize/mod.rs:55` declares the `Quantizer` trait `Send + Sync` "for rayon parallelism" but the actual quantize loop iterates sequentially. `llama-quantize` is `-t`-threaded; `mlx_lm.convert` gets Metal op parallelism for free. On apex MoE this is minutes-to-tens-of-minutes per pass.
5. **No imatrix-style activation cache between bit-pair variants.** dwq46 and dwq48 each redo the full forward+capture from scratch. Wasted forward pass per second variant on the same model. llama.cpp's `.imatrix` file is computed once and applied across Q4_K_M, Q5_K_M, Q6_K.
6. **Bulk `clone()` of tensor data on the write surface.** `src/backends/gguf.rs:897/910/926/3433` clones quantized blobs (full `Vec<u8>` weight copies) across the write path; `src/quantize/mod.rs:82,129` clones names + shapes. Peers thread refcounts (mlx arrays, torch storages).
7. **Streaming gap at MoE quant time.** Phase 1.5 (`src/main.rs:522` worktree) streams the expert merge layer-by-layer (peak +3.7 GB). Phase 3 then quantises the merged `[N=256, hidden, moe_inter]` block *as one piece*, holding a 256× tensor in RAM. Should also be layer-streaming.

**Cohort B — peer divergences (calibrators):**

8. **Calibrator monoculture.** hf2q ships only Apple/MLX's *Distilled Weight Quantization* (DWQ — `src/quantize/dwq.rs:1`). It does not ship llama.cpp's *importance matrix* (imatrix — `/opt/llama.cpp/tools/imatrix/imatrix.cpp`). DWQ and imatrix are peer-equivalent published calibration techniques from two communities; hf2q ships one, not both. This is a divergence by accident, not by design — Robert (2026-04-25): *"I think we want to support both dwq and imatrix."*

**Cohort C — sovereignty-by-design (preserve, document):**

9. **One-binary, single-process convert+quantize.** llama.cpp ships two stages (`convert_hf_to_gguf.py` → `llama-quantize`); we fuse them. Pro: single context, no intermediate persisted, one cancellation surface. Con: cannot replay just the quantize step against a cached F16. Sovereignty by design — Robert (`feedback_hf2q_sovereignty.md`, paraphrased): *"hf2q is pure Rust, mlx-native is the only sibling dep; no Python, no runtime link to candle/llama.cpp."* Preserved.
10. **Per-arch hand-ported transforms.** Phases 1.4 (`language_model.` strip), 1.5 (expert merge), 1.6 (RMS norm +1), 1.7 (V-head reorder, A_log negation, conv1d squeeze) are hand-translations of `convert_hf_to_gguf.py:5375–5424` + `Qwen3NextModel`. Sovereignty by design (no `torch` runtime dep). Preserved; future arches register via `ArchEntry` (ADR-012 Decision 20).

### Current state inventory (what exists in hf2q today, 2026-04-25)

- `src/main.rs:cmd_convert` — 1,668-line eager-load pipeline; Phases 0/0.25/0.3/0.4/0.5/1/1.4/1.45/1.5/1.6/1.7/2/3/4/4.5/4.6/4.7/4.8/5.1/5.5/6/7. The numbering is dense because every prior ADR added a phase rather than restructuring.
- `src/input/safetensors.rs` — mmap-then-`to_vec` reader; 332 LOC.
- `src/quantize/{mod,static_quant,mixed,dwq,dwq_activation,sensitivity,apex,intermediate_moe_q8}.rs` — eight quantizer modules; 4,460 LOC total.
- `src/backends/{gguf,safetensors_out,mod}.rs` — two output backends; 3,664 + 372 + 84 LOC. `SafetensorsBackend` flags `requires_native_quantization() = true` and bypasses the IR-level quantize.
- `src/inference/models/qwen35/activation_capture_real.rs` — `RealActivationCapture` consuming a GGUF file path (not in-memory). Forward-pass driver for DWQ on qwen35 / qwen35moe.
- `src/arch/` — ADR-012 Decision 20 registry; `ArchEntry` populated for `qwen35` + `qwen35moe`.

### Reference implementations (authoritative — read before changing)

- **Lazy tensor pattern.** `convert_hf_to_gguf.py:13203–13273` (`LazyTorchTensor`); `convert_hf_to_gguf.py:237` (`torch.load(..., mmap=True)`); `convert_hf_to_gguf.py:245–249` (lambdas wrapping safetensors slices).
- **Streaming write.** `convert_hf_to_gguf.py:Model.write_tensors()` (generator yielding tensors written in-place with `gguf_writer`).
- **imatrix calibration.** `/opt/llama.cpp/tools/imatrix/imatrix.cpp` (pre-pass that captures activations); `/opt/llama.cpp/src/llama-quant.cpp` (per-column-weighted MSE in `quantize_q4_K_M`, `quantize_q5_K_M`, `quantize_q6_K`).
- **DWQ calibration.** `src/quantize/dwq.rs:1–4` (header: "Distilled Weight Quantization (DWQ) calibration engine. Phase 1: Weight-space calibration. Phase 2: Activation-based calibration. optimal_scale = dot(W_original, Q_int) / dot(Q_int, Q_int)"); MLX-LM upstream — Apple ML Research. **Provenance correction**: this technique is *not* hf2q-original; the implementation is.
- **Rayon parallelism over quant.** llama.cpp's `quantize_q4_K_M` is multi-threaded via `std::thread` and `quantize_chunk_threaded` (`src/llama-quant.cpp`); shape: per-tensor work-units, write-stream serialised.

---

## Strategic Decision

ADR-014 re-platforms `cmd_convert` on a **lazy/streaming foundation**, lifts both **DWQ and imatrix** to first-class peer-equivalent calibrators, applies **cross-architecture** (not qwen-specific), and gates closure on **measured peer parity** against:

- `llama.cpp`'s `convert_hf_to_gguf.py + llama-quantize --imatrix` for the GGUF output path.
- `mlx_lm.convert --quant-method dwq` for the safetensors output path.

The architecture is **orthogonal internally** (`Calibrator` × `OutputFormat` traits independent) and **coupled externally** (only validated cross-product cells become named `--quant` variants; off-diagonal cells reachable only behind `HF2Q_UNSAFE_EXPERIMENTS=1`).

This ADR is a **parallel enhancement to ADR-012** (Robert, 2026-04-25: *"kind of an enhancement to"*). ADR-012 closes on the current pipeline; ADR-014's P11 re-emits ADR-012's four DWQ GGUFs (qwen3.6-27b dwq46, qwen3.6-27b dwq48, apex MoE dwq46, apex MoE dwq48) under the streaming pipeline as part of its own AC, providing the empirical answer to "does streaming change quant quality?" on the same models.

Robert priority lock (Q1, 2026-04-25): correctness primary, sovereignty secondary, performance tertiary — but Q1 was reframed by Q3: *"We need to be as fast as our peers. We need to be as correct as our peers."* Therefore correctness and performance are **co-primary**; sovereignty is an explicit non-trade.

---

## Non-Goals

1. **AWQ / GPTQ / AQLM calibrators.** The orthogonal `Calibrator` trait designed in Decision 9 is shaped to admit these methods — Decision 19 is one paragraph specifying how AWQ would land — but no AWQ/GPTQ/AQLM implementation lands in this ADR. Each future calibrator gets its own ADR.
2. **Inference-side changes.** ADR-013 owns the inference path; ADR-014 only consumes it via the existing `ActivationCapture` trait (with a new `from_lazy_tensor_map(...)` constructor — Decision 8). No new Metal kernels, no new model architectures in inference.
3. **New model architectures.** ADR-014 platforms the existing arch set (Gemma-4, Qwen3.5 dense, Qwen3.5-MoE) onto streaming. Future arches (Ministral, DeepSeek-V3) register via `ArchEntry` per ADR-012 Decision 20; that work belongs in their own ADRs.
4. **Tokenizer pipeline changes.** Calibration corpora use the model's existing tokenizer; no embed-side changes.
5. **GPU calibration kernels.** Activation capture uses ADR-013's existing GPU forward (`forward_gpu_with_capture`); no new Metal kernels in this ADR.
6. **Inference-side speed work.** ADR-005 owns inference perf; ADR-014 is convert-side only. Cross-cuts (e.g. `quantized_matmul_ggml` being shared between DWQ activation capture and decode) are noted but the kernels are not modified here.
7. **Replacement of the `auto` quant resolver.** The `intelligence::AutoResolver` keeps its public API; Decision 18 only adds new variant strings to its output table.

---

## Architecture Decisions

### 1. Lazy tensor primitive (`LazyTensor`)

**File:** `src/ir/lazy.rs` (new, ~250 LOC).

**Type:**

```rust
pub enum LazyTensor {
    Materialized(Tensor),
    Pending(Box<dyn FnOnce() -> Result<Tensor, MaterializeError> + Send + 'static>),
}

impl LazyTensor {
    pub fn materialize(self) -> Result<Tensor, MaterializeError> { … }
    pub fn map<F>(self, f: F) -> Self
    where F: FnOnce(Tensor) -> Result<Tensor, MaterializeError> + Send + 'static { … }
    pub fn shape(&self) -> &[usize] { /* metadata-only, no materialise */ … }
    pub fn dtype(&self) -> Dtype { /* metadata-only */ … }
}

pub struct LazyTensorMap {
    inner: BTreeMap<String, (LazyMeta, LazyTensor)>,
}
```

`LazyMeta` carries shape, dtype, and the byte-offset/length within the source mmap (so `shape()`/`dtype()` never materialise). `BTreeMap` (not `HashMap`) for deterministic iteration order — required for byte-identical regression on un-calibrated paths (Decision 17).

**Why `FnOnce` not `Fn`:** materialisation produces a `Vec<u8>` that we then own and drop. Re-materialising would re-allocate; we intentionally make that a compile error.

**Materialisation source A — safetensors mmap:** the closure captures an `Arc<Mmap>` + offset + len. `materialize()` does one `bytes_of_mmap[off..off+len].to_vec()` and returns. The `Arc<Mmap>` keeps the file mapped through the entire pipeline; pages are evicted by the kernel, not by us.

**Materialisation source B — already in memory:** `LazyTensor::Materialized(t)` returns `t`.

**Materialisation source C — derived (transform output):** the closure captures the parent `LazyTensor` + a `FnOnce` transform; calls `parent.materialize()?`, applies the transform, returns. Composes via `.map()`.

**Tests** (`src/ir/lazy.rs` + `tests/lazy_tensor.rs`):
- `test_materialize_once` — `materialize()` consumes; second call is a compile error.
- `test_map_compose_idempotent` — three transforms in a chain produce the same bytes as eager-applied transforms on the same input.
- `test_shape_dtype_no_materialise` — `shape()` + `dtype()` do not invoke the closure (verified via `Arc<AtomicUsize>` count).
- `test_send_bound` — `LazyTensor` is `Send` (so `rayon::par_iter` can move it across threads).

**Estimated LOC:** ~250 (impl ~150, tests ~100).

### 2. Lazy safetensors reader

**File:** `src/input/safetensors.rs` (modify, ~150 LOC delta — net +50 over the existing 332).

**Change:**

`read_safetensors_shard` returns a `LazyTensorMap` instead of a `TensorMap`. The `Mmap` is wrapped in an `Arc` and stored on the `LazyTensorMap`; each tensor entry's closure captures the `Arc<Mmap>` + offset/length. The existing `to_vec()` at line 316 is **deleted** — replaced by the closure body.

**Chesterton's fence.** The current `to_vec()` predates ADR-005's memory-pressure work; at the time, lifetimes on `Mmap` borrows were threaded through the entire pipeline and "copy out, drop the mmap" was the simplest way to avoid borrow propagation. We now have `Arc<Mmap>` patterns elsewhere (mlx-native tensor refs); the lifetime cost has been amortised.

**Multi-shard models.** The existing reader iterates shards; the new reader produces one `LazyTensorMap` per shard then merges — same end result, but each shard's `Arc<Mmap>` is independent (multiple files mmap'd concurrently). On apex MoE this means ~12 `mmap` regions live; total virtual ~70 GB; **resident** ~one tensor at a time.

**Tests:**
- `test_lazy_safetensors_apex_moe_peak_rss` — `/usr/bin/time -l` peak RSS during a `LazyTensorMap` iteration of apex MoE is `≤ 8 GB` (one layer's expert tile, BF16). Empirical baseline measured pre-P0 lands first; ADR closure asserts the gate.
- `test_lazy_safetensors_byte_identical_to_eager` — for Gemma-4 (small enough to fit eagerly), `LazyTensorMap::materialize_all()` produces a `TensorMap` byte-identical to today's `read_safetensors`.

**Estimated LOC:** ~150 modified, +100 tests.

### 3. Lift Phase 1.4–1.7 transforms to lazy

**Files:** `src/main.rs:457–582` (worktree), `src/models/qwen35/{dense,moe}.rs`.

**Change:** every Phase 1.x transform takes `&mut LazyTensorMap` (or returns a new one) instead of `&mut TensorMap`. The transforms compose via `.map()` on individual `LazyTensor`s — the bulk weight bytes never materialise during transform composition. Transforms run lazily; materialisation happens at quantize time (Decision 4).

**Phase 1.4** (`language_model.` prefix strip): pure metadata operation — rewrites map keys, no bulk bytes touched. Already trivially lazy.

**Phase 1.45** (jenerallee78 abliterated apex pre-merge handling): same — key rewrite, `merge_moe_experts_in_tensor_map` becomes `merge_moe_experts_in_lazy_map` with the merge closure deferred to materialisation time (Decision 7 — streaming MoE merge).

**Phase 1.5** (qwen35moe expert merge — ADR-012 Decision 9 / P5): the merge orchestration becomes a closure on the merged `LazyTensor`; the actual stack-in-memory of 256 experts only happens when the *quantize* loop materialises that one merged tensor (Decision 7 turns this into layer-streaming so the 256× explosion never happens).

**Phase 1.6** (RMS norm +1 bias): per-tensor `.map(add_one_inplace)`. Already lazy-ready.

**Phase 1.7** (V-head reorder, A_log negation, conv1d squeeze): per-tensor `.map(transform_fn)`. Already lazy-ready.

**Tests:**
- `test_lazy_phase_1_4_idempotent` — apply Phase 1.4 to `LazyTensorMap`, materialise — same bytes as eager Phase 1.4.
- `test_lazy_phase_1_5_layer_streaming` — for a synthetic 4-layer MoE, peak materialisation is one layer's experts (3 projections × 256 × hidden × moe_inter), not all 4 layers'. Verified via `Arc<AtomicUsize>` counter on the materialiser.
- `test_lazy_phases_1_6_1_7_byte_identical` — full Qwen3.5 4-layer synthetic, eager vs lazy → byte-identical output GGUF.

**Estimated LOC:** ~400 modified across 6 files (mostly mechanical signature changes).

### 4. Streaming quantize loop

**File:** `src/quantize/mod.rs` (rewrite `quantize_model`, ~300 LOC delta).

**Change:** replace

```rust
fn quantize_model(tensor_map: &TensorMap, …) -> QuantizedModel {
    let mut out = HashMap::new();
    for (name, tensor) in tensor_map.iter() {
        out.insert(name.clone(), quantize_tensor(...));
    }
    QuantizedModel { tensors: out, … }
}
```

with

```rust
fn quantize_streaming(
    map: LazyTensorMap,
    quantizer: &dyn Quantizer,
    backend: &mut dyn StreamingBackend,
    progress: &ProgressReporter,
) -> Result<QuantizationStats> {
    backend.begin_writing(map.metadata())?;
    for (name, (meta, lazy)) in map.into_iter() {
        let tensor = lazy.materialize()?;       // bytes IN
        let quantized = quantizer.quantize_tensor(&tensor, &meta, ...)?;
        backend.write_tensor(&name, &quantized)?;  // bytes OUT
        drop(tensor);                              // bytes FREED
        drop(quantized);                           // bytes FREED
        progress.tensor_done(&name);
    }
    backend.finalize()?;
    Ok(stats)
}
```

**`StreamingBackend` trait** (replaces the existing `OutputBackend`):

```rust
pub trait StreamingBackend: Send {
    fn begin_writing(&mut self, meta: &ModelMetadata) -> Result<()>;
    fn write_tensor(&mut self, name: &str, t: &QuantizedTensor) -> Result<()>;
    fn finalize(&mut self) -> Result<()>;
}
```

`QuantizedModel` is **deleted**. No code path holds the full quantised model in RAM. All metrics (peak RSS, wall time, file size) come from instrumented `StreamingBackend` impls.

**Cancellation safety** (Decision 20): `begin_writing` opens the output file in a tempdir; `finalize` atomically renames into place. Mid-stream Ctrl+C drops the tempdir; existing partial-cleanup logic in `cmd_convert` remains correct.

**Tests:**
- `test_streaming_peak_rss_apex_moe` — `/usr/bin/time -l` peak RSS during apex MoE q4_k_m convert is `≤ 44 GB` (peer-parity gate from Q3, "≤ peer + 10%").
- `test_streaming_byte_identical_q4_uncalibrated` — Gemma-4 `--quant q4` streaming-pipeline output byte-identical to current pipeline (Decision 17 determinism contract).
- `test_streaming_cancellation_no_partial_files` — SIGINT mid-stream leaves no `.gguf` or `.safetensors` in the output directory.

**Estimated LOC:** ~300 modified in `src/quantize/mod.rs`, ~150 modified each in `src/backends/{gguf,safetensors_out}.rs` to implement `StreamingBackend`.

### 5. Rayon parallelism in quantize loop

**File:** `src/quantize/mod.rs` (modify `quantize_streaming`, **~300–500 LOC honestly written** — proposal-time estimate of `~80 LOC` was unrealistic for the producer / worker-pool / serialiser shape with bounded MPSC channels, BTreeMap reordering buffer, on-channel `Tensor` payload sizing, and clean cancellation propagation. Robert lock 2026-04-25 round 2: correctness and speed dominate over LOC ceiling — pipeline shape stands).

**Change:** wrap the per-tensor quantize call (NOT the materialise / NOT the write) in a `rayon::ThreadPool` of size `min(num_cpus::get(), 16)`. The pool is bounded because Apple Silicon performance cores cap out at 12; oversubscription degrades.

**Pipeline shape (single-tensor unit of work):**

```text
producer (1 thread)        : materialise → channel-send
worker pool (n threads)    : channel-recv → quantize → channel-send
serialiser (1 thread)      : channel-recv → backend.write_tensor (in BTreeMap order)
```

Tensors quantise in parallel; writes are serialised in deterministic order (BTreeMap iteration) by the serialiser thread. This preserves byte-identical output (Decision 17) while giving N-way parallel quant work.

**Channel sizing** (back-pressure): the materialise→quantize channel has bounded capacity equal to the worker count; the quantise→write channel has bounded capacity equal to 2× worker count. Bounded channels prevent the producer from outrunning workers and inflating peak RSS.

**Tests:**
- `test_rayon_speedup_q4_apex_moe` — 8-thread quant of apex MoE q4 is `≥ 4×` faster than 1-thread (sub-linear due to memory-bandwidth saturation, but ≥ 4× is the gate).
- `test_rayon_byte_identical_to_serial` — output bytes identical between 1-thread and 8-thread serial-write modes.
- `test_rayon_no_oom_under_pressure` — 16-thread quant on a 32 GB synthetic MoE peaks at `≤ 40 GB` resident.

**Estimated LOC:** ~300–500 (channels + worker spawn + reordering buffer + cancellation propagation; was ~80 at proposal — see file note above).

### 6. Eliminate the P9b intermediate-GGUF dance

**Files:** `src/main.rs:680–778` (worktree), `src/inference/models/qwen35/activation_capture_real.rs`, `src/quantize/dwq_activation.rs`.

**Change:** add `RealActivationCapture::from_lazy_tensor_map(&LazyTensorMap, &Tokenizer) -> Result<Self>` — a constructor that builds a `Qwen35Model` directly from `LazyTensor`s (no intermediate GGUF write, no GGUF re-read). The convert pipeline calls this constructor with the in-memory `LazyTensorMap`; the model loads layer-by-layer from the lazy materialisers; activation capture runs against it; sensitivity is derived; DWQ quantises from the **same** `LazyTensorMap` (no re-read).

**Deletes:**
- `src/main.rs:715–738` — the `tempfile::tempdir()` + `emit_gguf_from_tensor_map(&intermediate_path)` block.
- `src/main.rs:739–753` — the `_drop_for_capture = std::mem::replace(&mut tensor_map, …)` workaround and re-read.
- `src/quantize/intermediate_moe_q8.rs` — the entire 239-line module (band-aid for F32-expand OOM; obsolete because `LazyTensorMap` never expands the experts to F32 in the first place).
- `src/backends/gguf.rs:emit_gguf_from_tensor_map` — used only by the dance.

**Adds:**
- `src/inference/models/qwen35/activation_capture_real.rs` — new constructor `from_lazy_tensor_map`. Implementation reads `Qwen35Model::load_from_lazy_tensor_map` (new in `src/inference/models/qwen35/weight_loader.rs`).
- `src/inference/models/qwen35/weight_loader.rs` — new function that materialises one tensor at a time on the GPU upload boundary; never holds the F32-expanded MoE in RAM.

**Net LOC:** −239 (intermediate_moe_q8 deletion) − 90 (main.rs dance) − 60 (emit_gguf_from_tensor_map) + 350 (`from_lazy_tensor_map` + `load_from_lazy_tensor_map`) = **−39 LOC net**.

**Cross-ADR correctness check:** ADR-012's P9b shipped `IntermediateMoeQ8Quantizer` because the F16-intermediate path expanded MoE experts to F32 on load and OOM'd at ~128 GB. With Decision 6's lazy upload, the model loads experts as native Q-blocks straight onto Metal — no F32 expansion ever — and the `IntermediateMoeQ8Quantizer` workaround becomes unreachable code, hence deletable.

**Tests:**
- `test_p9b_dance_eliminated` — assert `intermediate_moe_q8.rs` is removed; assert `cmd_convert` never writes a tempfile when `dwq_arch.requires_activation_capture()`.
- `test_activation_capture_from_lazy` — synthetic 4-layer qwen35 + 4-expert qwen35moe; `RealActivationCapture::from_lazy_tensor_map` produces sensitivity JSON byte-identical to the previous `from_intermediate_gguf` path on the same model.
- `test_apex_moe_capture_peak_rss` — apex MoE activation capture peak resident **≤ 36.3 GB** (33 GB measured on the existing intermediate-Q8 pipeline per ADR-012 P9b's real-model close + 10% headroom; locked 2026-04-26 in `docs/peer-parity-baselines-2026-04-26.md`). The number reflects the existing pipeline; **P4 re-measures on the new lazy-weight-loader path it lands** (Decision 6 + 8 deletions remove the IntermediateMoeQ8Quantizer and the F32 round-trip), and replaces this gate value with `<P4-measured> + 10%`. The dated exit condition lives in the baselines file; the gate in this body refreshes with each P4 iteration.

**Estimated LOC:** −39 net, but ~700 lines touched (delete + add).

### 7. Streaming MoE expert merge at quant time

**File:** `src/models/qwen35/moe.rs` (modify `merge_moe_experts_in_tensor_map`, ~150 LOC delta).

**Change:** the existing `merge_moe_experts_in_tensor_map` produces a single merged `[256, hidden, moe_inter]` tensor per layer, holds it in `tensor_map`, then the quantize pass walks it. Replace with `merge_moe_experts_lazy` which produces a `LazyTensor` whose closure performs the merge **at materialisation time**, emitting one merged tensor per layer; the quantize pass materialises layer-N, quantises it, writes it, drops it, then materialises layer-N+1.

Peak resident at quant time becomes:

```text
peak = 1 layer × 3 projections × 256 experts × hidden × moe_inter × dtype_bytes
     = 1 × 3 × 256 × 2048 × 768 × 2  (BF16, apex shape; hidden=2048, moe_inter=768)
     ≈ 2.4 GB per merged-projection tile
```

vs. eager all-layers-merged: ~80 GB on apex.

**Tests:**
- `test_lazy_moe_merge_one_layer_resident` — Arc<AtomicUsize> on the merge counter; assert at most 1 layer's worth of merged bytes is alive at any quant-loop tick.
- `test_lazy_moe_merge_byte_identical_to_eager` — synthetic 4-layer MoE merged eagerly vs lazily → quantised output byte-identical.

**Estimated LOC:** ~150.

### 8. `ActivationCapture::from_lazy_tensor_map` (cross-cut, ADR-013 boundary)

**File:** `src/inference/models/qwen35/activation_capture_real.rs` + `src/inference/models/qwen35/weight_loader.rs` (covered in Decision 6).

This Decision exists to lock the cross-ADR API contract. ADR-013 owns `ActivationCapture`; ADR-014 adds a constructor (`from_lazy_tensor_map`) without changing the trait. The trait method signatures are unchanged; downstream consumers in ADR-013 are unaffected.

**Cross-ADR sign-off:** ADR-013 P12 must accept this constructor's existence and the `weight_loader::load_from_lazy_tensor_map` addition. Documented in ADR-013's "Dependencies on other work" section as a same-author handoff (no separate review cycle required, but the change is named and traceable).

### 9. `Calibrator` trait (orthogonal axis)

**File:** `src/calibrate/mod.rs` (new module, ~250 LOC).

**File-layout decision (Robert lock 2026-04-25 round 2 — Layout A):** `src/quantize/dwq.rs`, `src/quantize/dwq_activation.rs`, `src/quantize/sensitivity.rs`, and `src/quantize/apex.rs` **move into `src/calibrate/`** as part of P7. Full path migration; **no re-export wrappers, no thin shims, no legacy `pub use` aliases**. All callers and test imports update to the new paths in one pass. P7's LOC budget grows for the import-rewrite churn (every `tests/convert_qwen35_*.rs`, every internal `use crate::quantize::dwq::*`, every `cmd_convert` dispatch site).

**Trait:**

```rust
pub trait Calibrator: Send + Sync {
    fn name(&self) -> &'static str;
    fn requires_forward_pass(&self) -> bool;
    fn calibrate(
        &mut self,
        model: &LazyTensorMap,
        meta: &ModelMetadata,
        corpus: &CalibrationCorpus,
        progress: &ProgressReporter,
    ) -> Result<CalibrationData, CalibrationError>;
}

pub enum CalibrationData {
    None,
    Imatrix(HashMap<TensorName, Vec<f32>>),    // per-column weights
    Dwq(SensitivityMap),                       // existing DWQ scoring
}
```

**Three implementations:**
- `NoneCalibrator` — `requires_forward_pass = false`; `calibrate` returns `CalibrationData::None`.
- `ImatrixCalibrator` — Decision 10.
- `DwqCalibrator` — wraps existing `dwq_activation.rs` orchestration; `requires_forward_pass = true` for qwen35/qwen35moe, false for others.

**Why `Send + Sync`:** calibrators may be invoked from worker threads under rayon (Decision 5), or held in a `--calibration auto` resolver.

**Tests:** trait-conformance test for each impl (round-trip a synthetic calibration on synthetic 4-layer Gemma-4); cross-impl test asserting `name()` is unique.

**Estimated LOC:** ~250 (trait + 3 impls + dispatch).

### 10. `ImatrixCalibrator` — pure-Rust port

**File:** `src/calibrate/imatrix.rs` (new, ~600 LOC).

**Algorithm (verbatim from `/opt/llama.cpp/tools/imatrix/imatrix.cpp`, ported to Rust):**

For each calibration sample (token sequence `T`):
1. Run forward pass on `T` (using ADR-013's `RealActivationCapture::run_calibration_prompt` for qwen35/qwen35moe, an arch-agnostic CPU forward for Gemma-4 — already exists at `src/inference/models/gemma4/forward_cpu.rs`).
2. At each `Linear` layer with weight `W ∈ [out, in]`, capture the **input activation** vector `x ∈ [in]` for every token in `T`.
3. Accumulate `imatrix_layer[in_col] += x[in_col]² * 1.0` (the `1.0` is the per-token weight; uniform).
4. After all samples: divide by total token count → `imatrix_layer[in_col] = mean(x[in_col]²)`.

Output: `HashMap<TensorName, Vec<f32>>` where each `Vec<f32>` has length equal to `W`'s input dimension. Cell `[c]` holds the importance of input column `c` for that tensor.

**Sidecar emission:** `--imatrix-out path.imatrix` writes a llama.cpp-compatible binary file (header + per-tensor float32 vectors), so the imatrix can be round-tripped through `llama-quantize --imatrix` for cross-validation.

**Per-column-weighted MSE in k-quant** (consumed by Decision 11's `OutputFormat::KQuant`): when fitting a Q4_K super-block, the codebook search minimises `Σ imatrix[col] · (W[row, col] − dequant(Q[row, col]))²` instead of plain `Σ (W − dequant(Q))²`. Implemented in `src/quantize/k_quant.rs` (new module, Decision 11).

**Cross-validation gate:** `tests/imatrix_cross_validate.rs` — port a synthetic Gemma-4 4-layer weight set, run hf2q's `ImatrixCalibrator`, run `llama.cpp/llama-imatrix` on the same weights + corpus, assert per-tensor max-element-difference `≤ 1e-4` (relative). Ensures the port is byte-precise to llama.cpp's algorithm before any peer-parity claim is made downstream.

**Tests:**
- `test_imatrix_per_column_accumulation` — synthetic single-layer linear, deterministic activations, hand-computed expected imatrix values.
- `test_imatrix_sidecar_round_trip` — write `.imatrix`, read with `llama-imatrix --tool dump`, assert structure.
- `test_imatrix_cross_validate_against_llama_cpp` — gate above (default-ignored, enabled in P10 peer-parity harness).
- `test_imatrix_pure_rust_no_python_dep` — `cargo metadata` shows no torch/numpy/python crate in the dep tree.

**Estimated LOC:** ~600 (impl ~400, tests ~200).

### 11. `OutputFormat` enum (orthogonal axis) + k-quant codebook

**Files:** `src/quantize/output_format.rs` (new, ~150 LOC) + `src/quantize/k_quant.rs` (new, ~700 LOC).

**Enum:**

```rust
pub enum OutputFormat {
    Flat(FlatType),                           // F16, BF16, Q2, Q4, Q8
    BitPair { base_bits: u8, sensitive_bits: u8 },  // 4-6, 4-8, 6-8, 2-8
    KQuant(KQuantType),                       // Q4_K_M, Q5_K_M, Q6_K
    KQuantAdaptive { target_bpw: f32 },       // adaptive across [Q3_K_S, Q6_K], replaces apex
}
```

**K-quant codebook implementation** (`src/quantize/k_quant.rs`): pure-Rust port of llama.cpp's `quantize_q4_K_M` / `quantize_q5_K_M` / `quantize_q6_K` from `/opt/llama.cpp/src/llama-quant.cpp`. Per-super-block: 256 elements grouped into 8 sub-blocks of 32; per-sub-block scale + min; per-element 4/5/6 bits; final super-block scale stored as F16. Algorithm is `quantize_iqK_K` in llama.cpp's `ggml-quants.c` — referenced verbatim, ported line-for-line.

**Per-column-weighted variant:** when called with `Some(imatrix)`, the inner codebook search minimises imatrix-weighted MSE (Decision 10 §"Per-column-weighted MSE").

**Pure-Rust no-link:** llama.cpp's k-quant impl is under MIT — Robert sovereignty rule per `feedback_hf2q_sovereignty.md` is **ported, not linked**. `cargo metadata` test asserts no `cc` / `cmake` / link to `libggml`.

**Byte-identity reference path (Robert lock 2026-04-25 round 2 — Apple Silicon).** The byte-identical gate is against llama.cpp's **NEON code path** (`quantize_row_q4_K` on `aarch64-apple-darwin`), **not** `quantize_row_q4_K_ref` (the scalar reference). The NEON path is what M-series users actually run; matching the scalar reference would not constitute peer parity in production. The pure-Rust port either uses `std::arch::aarch64` intrinsics or scalar code that replicates NEON's reduction order (associativity-sensitive horizontal sums in `make_qkx2_quants` must match). Cross-platform users on `x86_64` are not in scope for this gate.

**Tests:**
- `test_q4_k_m_byte_identical_to_llama_cpp` — synthetic deterministic input; produce hf2q Q4_K_M block; produce llama.cpp Q4_K_M block (built `aarch64-apple-darwin`, NEON path); `xxd` diff = empty.
- `test_q5_k_m_byte_identical` — same, Q5_K_M.
- `test_q6_k_byte_identical` — same, Q6_K.
- `test_kquant_imatrix_weighted_mse` — synthetic with high-magnitude column 0; without imatrix, all columns have equal error; with imatrix giving column 0 weight 100, column 0 error is `≤ 0.1×` the rest.
- `test_kquant_adaptive_target_bpw` — `KQuantAdaptive { target_bpw: 4.5 }` produces a per-tensor type assignment whose mean bpw across all tensors is within `± 0.05` of 4.5.

**Estimated LOC:** ~150 + ~700 + ~200 tests = ~1050.

### 12. CLI variant menu (coupled external naming)

**File:** `src/cli.rs` (modify `QuantMethod` enum, ~120 LOC delta).

**Final variant menu (Robert lock, Q9, 2026-04-25 — clean cut, no aliases, no users to break):**

| External `--quant` | Calibrator | OutputFormat | Peer baseline |
| ------------------ | ---------- | ------------ | ------------- |
| `f16` | None | Flat(F16) | — |
| `bf16` | None | Flat(BF16) | — |
| `q2` | None | Flat(Q2) | — |
| `q4` | None | Flat(Q4) | — |
| `q8` | None | Flat(Q8) | — |
| `q4_k_m` | None | KQuant(Q4_K_M) | llama.cpp uncalibrated |
| `q5_k_m` | None | KQuant(Q5_K_M) | llama.cpp uncalibrated |
| `q6_k` | None | KQuant(Q6_K) | llama.cpp uncalibrated |
| `imatrix-q4_k_m` | Imatrix | KQuant(Q4_K_M) | **llama.cpp imatrix Q4_K_M (primary)** |
| `imatrix-q5_k_m` | Imatrix | KQuant(Q5_K_M) | llama.cpp imatrix Q5_K_M |
| `imatrix-q6_k` | Imatrix | KQuant(Q6_K) | llama.cpp imatrix Q6_K |
| `imatrix-adaptive` | Imatrix | KQuantAdaptive | replaces `apex` |
| `dwq-4-6` | Dwq | BitPair(4,6) | **mlx-lm DWQ (primary)** |
| `dwq-4-8` | Dwq | BitPair(4,8) | mlx-lm DWQ |
| `dwq-6-8` | Dwq | BitPair(6,8) | mlx-lm DWQ |
| `dwq-2-8` | Dwq | BitPair(2,8) | mlx-lm DWQ |
| `auto` | (resolved) | (resolved) | per Decision 18 |

**Off-diagonal cells** (e.g. DWQ + KQuant, Imatrix + BitPair) are reachable only via the dev gate:

```bash
HF2Q_UNSAFE_EXPERIMENTS=1 hf2q convert <input> \
    --calibration imatrix --output-format bit-pair-4-6
```

The dev flags are documented in `docs/converting-a-model.md` under a "For maintainers" section, not the user-facing flow (Paige's note, Q7).

**`apex` deletion:** the existing `--quant apex` is **removed**, not aliased. Users reach the equivalent functionality via `--quant imatrix-adaptive`. Per Q9: clean cut.

**Tests:**
- `test_cli_variant_menu_complete` — every external variant string parses to the correct `(Calibrator, OutputFormat)` tuple.
- `test_cli_apex_removed` — `--quant apex` errors with "unknown variant `apex`; did you mean `imatrix-adaptive`?".
- `test_cli_dev_gate_off_diagonal` — without `HF2Q_UNSAFE_EXPERIMENTS=1`, `--calibration X --output-format Y` is rejected.

**Scope confirmation (Robert lock 2026-04-25 round 2):** all 17 variants ship in this ADR. The full menu is the deliberate long-term shape, not a candidate for trim-to-essentials. Variants without immediate user demand (`dwq-6-8`, `dwq-2-8`, `imatrix-q5_k_m`, `imatrix-q6_k`, `imatrix-adaptive`) ship now to prove the orthogonal `Calibrator × OutputFormat` design at scale rather than discovering trait gaps in follow-up ADRs.

**Estimated LOC:** ~120.

### 13. Migration policy

**Files:** `src/cli.rs` + `docs/converting-a-model.md`.

Per Robert (Q9, 2026-04-25): *"we don't have any users yet. B — but don't leave old cruft. There's no compatibility concern."*

**Deleted variants (no aliases, no deprecation warnings):**
- `mixed-2-6`, `mixed-3-6`, `mixed-4-6` — uncalibrated bit-pair. Reachable in dev mode via `HF2Q_UNSAFE_EXPERIMENTS=1 --calibration none --output-format bit-pair-N-M` if absolutely needed; not exposed.
- `apex` — replaced by `imatrix-adaptive`.
- `dwq-mixed-4-6`, `dwq-mixed-4-8`, `dwq-mixed-6-8`, `dwq-mixed-2-8` — replaced by `dwq-4-6`, `dwq-4-8`, `dwq-6-8`, `dwq-2-8` (cleaner names; `mixed-` prefix was a vestige of the underlying `MixedBitQuantizer`).

**`MixedBitQuantizer` (`src/quantize/mixed.rs`, 514 LOC):** retained internally as the implementation backing `OutputFormat::BitPair`; not exposed to the CLI.

**Tests:**
- `test_legacy_quant_variants_rejected` — every deleted variant string causes a typed CLI error.

**Estimated LOC:** ~50 (mostly deletion).

### 14. Both backends with shared streaming pipeline

**Files:** `src/backends/{gguf,safetensors_out,mod}.rs` (modify, ~400 LOC delta).

**Change:** both backends implement `StreamingBackend` (Decision 4) and consume from the same `quantize_streaming` loop. The existing `OutputBackend::requires_native_quantization()` shortcut on `SafetensorsBackend` is **removed** — safetensors output now goes through the IR-level quantize loop with the chosen `Calibrator` × `OutputFormat`, exactly like GGUF.

**Safetensors output layout (Robert lock 2026-04-25 round 2):** `--format safetensors` emits an **mlx-lm-style directory** (`<output>/`), **not** a single `.safetensors` file. The directory contains:
- `config.json` (model architecture, copied/derived from the input HF repo)
- tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` etc.)
- `generation_config.json`
- one or more sharded `model-NNNNN-of-MMMMM.safetensors` files whose **per-shard headers** carry mlx-lm-compatible quant metadata (per-tensor `scales`, `zeros`, `bits` keys)

Appendix A's `Qwen3.6-27B-dwq-4-6/` notation is canonical. This directory layout makes hf2q's safetensors output **directly loadable by `mlx_lm.load(<output>)`** — establishing peer parity with `mlx_lm.convert`.

**Cross-validation:** P10 includes a test that `mlx_lm.load(hf2q_dwq46.safetensors)` succeeds and produces logits within `1e-3` (cosine) of `hf2q.serve(hf2q_dwq46.safetensors)` on a deterministic 16-prompt smoke set.

**Tests:**
- `test_safetensors_streaming_byte_identical_to_eager` — Gemma-4 → safetensors, eager pipeline vs streaming pipeline → byte-identical (un-calibrated path; Decision 17).
- `test_safetensors_dwq46_loads_in_mlx_lm` — `--quant dwq-4-6 --format safetensors` produces a file `mlx_lm.load` accepts.
- `test_safetensors_imatrix_q4km_round_trip` — `--quant imatrix-q4_k_m --format safetensors` produces a file that hf2q's own `serve` re-loads with byte-identical logits.

**Estimated LOC:** ~400 (~150 each backend + ~100 metadata adapter).

### 15. Peer-parity gates (Robert lock, Q3)

**File:** `tests/peer_parity_gates.rs` (new, ~500 LOC) + `docs/converting-qwen35.md` (table copy).

**Locked gate matrix:**

| Axis | Value |
| ---- | ----- |
| Primary peer | `llama.cpp` (`convert_hf_to_gguf.py` + `llama-quantize --imatrix`) |
| Secondary peer | `mlx_lm.convert` |
| Reference models | `Qwen/Qwen3.6-27B` dense + `jenerallee78/Qwen3.6-35B-A3B-abliterix-ega-abliterated-apex` MoE |
| Hardware | M5 Max, 128 GB |
| Speed metric | wall-clock convert+quant; `/usr/bin/time -p`; **`1 warmup run discarded → 60 s thermal cooldown → 3 timed runs, median wins`**, both peers warm |
| Speed tolerance | hf2q median wall ≤ 1.10× peer median wall |
| Memory metric | peak RSS; `/usr/bin/time -l` `maximum resident set size`; recorded across the 3 timed runs, median wins |
| Memory tolerance | hf2q median peak ≤ 1.10× peer median peak |
| Correctness — calibrated quality | wikitext2 PPL on hf2q output ≤ peer output × 1.02 |
| Correctness — vs F16 reference | KL(hf2q output ∥ F16) ≤ 0.02 nats (carry-over from ADR-012 Decision 17) |
| Determinism | byte-identical output across two fresh cold runs (un-calibrated paths only; Decision 17) |

**Gate cells** (P10/P11 must measure all 8 model × backend × calibrator combinations):

| Model | Backend | Calibrator | Peer | Speed gate | RSS gate | PPL gate |
| ----- | ------- | ---------- | ---- | ---------- | -------- | -------- |
| 27B dense | GGUF | None (q4_k_m) | llama.cpp uncalibrated Q4_K_M | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| 27B dense | GGUF | Imatrix (imatrix-q4_k_m) | llama.cpp imatrix Q4_K_M | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| 27B dense | safetensors | DWQ (dwq-4-6) | mlx_lm DWQ | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| 27B dense | GGUF | DWQ (dwq-4-6) | (no peer; vs hf2q current pipeline) | ≤ 1.0× | ≤ 0.50× | ≤ 1.0× |
| apex MoE | GGUF | None (q4_k_m) | llama.cpp uncalibrated Q4_K_M | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| apex MoE | GGUF | Imatrix (imatrix-q4_k_m) | llama.cpp imatrix Q4_K_M | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| apex MoE | safetensors | DWQ (dwq-4-6) | mlx_lm DWQ | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| apex MoE | GGUF | DWQ (dwq-4-6) | (no peer; vs hf2q current pipeline) | ≤ 1.0× | ≤ 0.50× | ≤ 1.0× |

Last column: hf2q DWQ→GGUF has no direct peer (mlx-lm doesn't emit GGUF, llama.cpp doesn't ship DWQ); the gate is hf2q-vs-hf2q-current-pipeline. RSS must drop by **≥ 50%** (the central correctness/sanity claim of this ADR — streaming halves peak resident).

**Why warmup-discarded + thermal cooldown** (Robert lock 2026-04-25 round 2). The Apple-Silicon ML benchmark community's actual practice is `warmup → cooldown → timed runs`, not "system reboot, single cold run." Sources: Eduard Stere's llama.cpp Apple-Silicon harness explicitly warms before timing; llama.cpp discussion #4167 pins both peers to identical builds for fairness; `mlx_transformers_benchmark`'s `cooldown_time_fraction` manages thermal drift on shared-memory M-series. Warmup-discarded ensures both peers measure with warm Metal-shader caches, sidestepping the (Apple-undocumented) persistence behaviour of `~/Library/Caches/com.apple.metal/`. The 60 s thermal cooldown holds the M5 Max within the same thermal envelope across the three timed iterations. Cold-start latency, if separately interesting to surface to users, can be reported as a `cold_first_run` field alongside the median — but it does not gate.

**Test infrastructure:** `tests/peer_parity_gates.rs` orchestrates the runs; results land as a markdown table in `docs/peer-parity-results-2026-04-25.md` (or whatever date P11 closes); the closing commit cites the table inline.

**Estimated LOC:** ~500 test harness, ~50 gate definitions.

### 16. PPL + KL evaluation methodology

**File:** `tests/fixtures/ppl-corpus/wikitext2.tokens` (already exists from ADR-012 P9) + `src/arch/conformance.rs::ppl_kl_eval` (already exists).

**No new code** — Decision 16 reuses ADR-012's already-built PPL + KL infrastructure verbatim. Cross-peer comparison adds three columns to the existing reporter:

```
hf2q_ppl  llama_ppl  mlx_ppl   hf2q_kl_vs_f16  llama_kl_vs_f16  mlx_kl_vs_f16
```

**Eval corpus (Robert lock 2026-04-25 round 2):** wikitext-2 **full test split (~280k tokens)**, deterministic SHA-256 sidecar. Apples-to-apples with llama.cpp / mlx-lm published Q4_K_M PPL numbers — this is what users compare against when picking a quant. ADR-012's existing 512-token fixture (`tests/fixtures/ppl-corpus/wikitext2.tokens`) is preserved as a fast smoke check (default `cargo test`); the full-split eval lives at `tests/fixtures/ppl-corpus/wikitext2-full.tokens` (new in P10, ~280k tokens, ~700 KB on disk after BPE encoding) and runs only in the `#[ignore]`-gated peer-parity harness.

**Cross-corpus disjointness gate:** ADR-012's `tests/calibration_eval_disjoint.rs` already enforces zero overlap between calibration and PPL eval corpora. ADR-014 inherits this — every calibrator added in this ADR (Imatrix) re-runs the disjointness check.

**Estimated LOC:** ~50 (column additions in the reporter).

### 17. Determinism contract

**Files:** `tests/determinism_*.rs` (new, ~200 LOC).

**Contract:**

| Path | Determinism level | Test |
| ---- | ----------------- | ---- |
| `--quant f16`, `bf16` | Byte-identical to current pipeline | `test_determinism_f16_byte_identical_to_current` |
| `--quant q2`, `q4`, `q8` | Byte-identical to current pipeline | `test_determinism_flat_byte_identical_to_current` |
| `--quant q4_k_m`, `q5_k_m`, `q6_k` | Byte-identical to llama.cpp `--quant Q4_K_M` etc. | `test_determinism_kquant_vs_llama_cpp` |
| `--quant imatrix-*` | Byte-identical across two cold hf2q runs (re-blessed snapshot, not back-compat) | `test_determinism_imatrix_two_cold_runs` |
| `--quant dwq-*` | Byte-identical across two cold hf2q runs (re-blessed snapshot) | `test_determinism_dwq_two_cold_runs` |
| `--quant imatrix-adaptive` | Byte-identical across two cold hf2q runs | `test_determinism_imatrix_adaptive_two_cold_runs` |
| `--quant auto` | Resolves to a named variant deterministically; that named variant satisfies its own row | `test_determinism_auto_resolves_then_named_holds` |

**Reasoning** (Robert, Q3 implicit + Winston, Q10):

- Un-calibrated paths (Flat, K-quant uncalibrated): the streaming pipeline changes nothing about the *math* — same input bytes → same output bytes. Byte-identical to current pipeline is achievable and the bar.
- K-quant uncalibrated additionally byte-identical to llama.cpp `llama-quantize --quant Q4_K_M` because Decision 11's k_quant.rs is a line-for-line port. This is the strongest correctness claim in the ADR — porting llama.cpp's k-quant codebook to pure Rust and proving byte-identity in CI.
- Calibrated paths: re-blessed because (a) streaming changes processing order, (b) sensitivity caching changes scoring derivation, (c) imatrix port may have float-precision differences from llama.cpp's C impl (mitigated by Decision 10's cross-validation `≤ 1e-4`). The bar is "deterministic across two cold runs of the same hf2q binary on the same input"; not back-compatible with current-pipeline DWQ outputs.

**Estimated LOC:** ~200 tests.

### 18. `--quant auto` routing

**File:** `src/intelligence/auto_quant.rs` (modify `AutoResolver`, ~80 LOC delta).

**Resolution table:**

| Model class | Hardware | Resolved variant |
| ----------- | -------- | ---------------- |
| Dense ≤ 7B | any | `imatrix-q4_k_m` |
| Dense 7B–30B | any | `imatrix-q4_k_m` |
| Dense > 30B | < 64 GB RAM | `imatrix-q4_k_m` |
| Dense > 30B | ≥ 64 GB RAM | `imatrix-q5_k_m` |
| MoE any size | < 96 GB RAM | `dwq-4-6` |
| MoE any size | ≥ 96 GB RAM | `dwq-4-8` |
| Architecture has `ArchEntry::auto_override = Some(v)` | any | `v` (per-arch override) |

**Why MoE → DWQ:** DWQ's per-tensor sensitivity allocation handles MoE's expert heterogeneity better than imatrix's per-column diagonal weighting (the hot/cold pattern across experts is along the `expert` axis, not the `column` axis).

**Why dense → imatrix:** imatrix is the llama.cpp ecosystem default; users coming from `llama-cli` get the output they expect. Plus three years of community PPL tuning means imatrix Q4_K_M is the well-validated point on the perf/quality curve.

**Tests:**
- `test_auto_dense_27b` — resolves to `imatrix-q4_k_m`.
- `test_auto_moe_apex` — resolves to `dwq-4-6` on a 64 GB box, `dwq-4-8` on a 128 GB box.
- `test_auto_arch_override` — when `ArchEntry::auto_override = Some("imatrix-q5_k_m")`, that wins.

**Estimated LOC:** ~80.

### 19. Future-calibrator extensibility (door open, no implementations)

**File:** `docs/calibrator-onboarding.md` (new, ~200 LOC of docs — explicitly *docs*, not stubs).

**One paragraph in this ADR:** `Calibrator` (Decision 9) is shaped to admit AWQ (per-channel quantization-aware), GPTQ (Hessian-based reordering), and AQLM (additive quantization) without trait changes. Adding a new calibrator: implement `Calibrator`, add an `OutputFormat` variant if its output codebook is novel (otherwise reuse), wire one CLI variant, write the cross-validation gate. Future ADRs (one per calibrator) own the implementation; ADR-014 ships only DWQ + Imatrix + None.

**No stub code lands in `src/calibrate/{awq,gptq,aqlm}.rs`** — per mantra ("No stub (todo later) code"). The door is shaped, not propped open.

**Estimated LOC:** 0 code, ~200 docs.

### 20. Cancellation safety with streaming writes

**File:** `src/main.rs:cmd_convert` (modify Ctrl+C handler, ~50 LOC delta).

**Contract:**
1. Output file is opened in a sibling tempdir (`<output_path>.tmp.XXXXXX/`); writes go there.
2. `StreamingBackend::finalize()` does `std::fs::rename(tempdir/output_basename, final_path)` — atomic on POSIX/APFS.
3. SIGINT handler: sets `INTERRUPTED.store(true)`; the streaming loop checks at every tensor boundary and exits cleanly; the tempdir is removed in the existing cleanup handler (`src/main.rs:363`).
4. **No partial output file ever appears at `final_path`** — that path either holds a complete file or doesn't exist.
5. **No tensor-level rollback.** Per mantra ("no fallback"), we don't pretend partial outputs are recoverable. Cancellation = full restart; the calibration cache (Decision 5) avoids redoing the expensive work on restart.

**Tests:**
- `test_cancellation_no_partial_at_final_path` — kill mid-stream; assert `final_path` does not exist; assert no `*.tmp.*` directories leaked.
- `test_cancellation_calibration_cache_survives` — kill mid-stream after calibration completes; assert `~/.cache/hf2q/sensitivity/<key>.json` is intact and reusable on next run.

**Estimated LOC:** ~50.

### 21. Sovereignty preservation (explicit non-changes)

**No file changes.** Decision 21 exists to lock the ADR's position on which divergences are *deliberate* and not subject to peer-parity scrutiny.

**Preserved:**
- **One-binary, single-process convert+quantize.** llama.cpp's two-stage pipeline (`convert_hf_to_gguf.py` → `llama-quantize`) is **not** adopted. Pro carried: single context, single cancellation, no intermediate F16 artefact persisted. Con accepted: cannot replay just the quantize step.
- **Per-arch hand-ported transforms.** Phases 1.4/1.5/1.6/1.7 stay as Rust translations of `convert_hf_to_gguf.py`. No `pyo3` runtime dep, no `torch` link.
- **Pure Rust, mlx-native sole sibling dep.** `cargo metadata` test asserts no `python-sys`, `pyo3`, `numpy`, `torch-sys`, `libggml-sys`, `cmake`-driven build dep.

**Newly explicit (ADR-014 lock):**
- **Calibrator portability is not a sovereignty claim.** DWQ comes from MLX; imatrix comes from llama.cpp; both ship in hf2q because **both algorithms are published peer-equivalent techniques**. The sovereignty claim is on the *toolchain* (pure-Rust port, no Python runtime), not on the algorithm.
- **Runtime vs test-time scope (Robert lock 2026-04-25 round 2).** Sovereignty applies at **runtime** — the shipped `hf2q` binary contains no Python interpreter, no torch/numpy/python crate, no link to libggml. **Test-time** Python is permitted only behind `#[ignore]`-gated parity-harness tests run by `scripts/peer_parity_run.sh`, which shells out to `mlx_lm.convert`, `mlx_lm.load`, `llama-imatrix`, and `convert_hf_to_gguf.py` for cross-validation. Default `cargo test` set stays pure Rust. `test_sovereignty_no_python_dep` only walks the runtime crate dep graph, not the test-harness shell-out targets.

**Tests:**
- `test_sovereignty_no_python_dep` — `cargo metadata --format-version=1 | jq` walks the **runtime** dep tree, asserts none of the forbidden crates appear. Test-time `mlx-lm`/`llama-imatrix` shell-outs are not in scope.
- `test_sovereignty_no_libggml_link` — `cargo build --target aarch64-apple-darwin -v 2>&1 | grep -v libggml`.

**Estimated LOC:** ~50 sovereignty-gate tests.

### 22. Re-emit ADR-012's four DWQ GGUFs under streaming pipeline (closure AC)

**Files:** none (artefact production).

**Deliverables (P11):**
- `models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf` (re-emitted)
- `models/qwen3.6-27b-dwq48/qwen3.6-27b-dwq48.gguf` (re-emitted)
- `models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46.gguf` (re-emitted)
- `models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48.gguf` (re-emitted)
- The same four under `--format safetensors` (mlx-lm-style directories per Decision 14) for the mlx-lm peer column.

**Shipped vs ephemeral (Robert lock 2026-04-25 round 2).** P11 ships **only the eight artefacts above** (four DWQ GGUFs + four DWQ safetensors directories) to `models/`. Decision 15's gate matrix also requires measuring `imatrix-q4_k_m` cells on both 27B dense and apex MoE — those convert runs **produce ephemeral measurement files in scratch**, the harness records PPL/RSS/wall, and the file is deleted alongside the peer's `<peer>-Q4_K_M.gguf` per R13's disk-pressure mitigation. **No `imatrix-*` artefact lives in `models/` after P11 closes.** Users who want imatrix-q4_k_m artefacts run convert themselves; the ADR ships only the ADR-012 carry-over commitments.

**Comparison table** (lands in P11 closing commit and `docs/peer-parity-results-<date>.md`):

| Artifact | Pre-ADR-014 size | Post-ADR-014 size | Pre wall | Post wall | Pre peak RSS | Post peak RSS | Pre PPL | Post PPL | KL(post ∥ pre) |
| -------- | ---------------- | ----------------- | -------- | --------- | ------------ | ------------- | ------- | -------- | -------------- |
| (8 rows: 4 GGUFs × 2 backends — but only GGUF for these four; safetensors variants emit alongside as part of P11) | ... | ... | ... | ... | ... | ... | ... | ... | ... |

The ADR closes when this table is filled in with measured numbers and every row meets the gates from Decision 15.

---

## Phase plan

Phases are **dependency-ordered**. Each phase has a single owner claim per `feedback_swarm_sequential_when_shared_build.md` (shared Cargo `target/` = sequential). **Exception (Robert lock 2026-04-25 round 2):** P11 (real-model artefact production, GPU-bound) and P12 (docs-only, no GPU, trivial build pressure) **may run in parallel via separate CFA worktrees** (each worktree has its own `target/`; the sequential rule applies to *shared* `target/`, which CFA worktrees split). The Totals' "~6–8 weeks if P11 reference artefact production runs in parallel with downstream phases on the same M5 Max" assumption rests on this exception.

### P0 — Lazy tensor primitive + lazy safetensors reader (+ Decision 6 measurement spike)

**Scope:** Decisions 1 + 2 + Decision 6's empirical gate-value derivation.

**Dependency:** real ADR-012 close (R14: four DWQ GGUFs verifiably load in `llama-cli`).

**Deliverables:**
- `src/ir/lazy.rs` (new, ~250 LOC).
- `src/input/safetensors.rs` modified (`Arc<Mmap>` lifetime, closure-based materialiser).
- 12 unit tests + 2 integration tests.
- **Decision 6 measurement spike (Robert lock 2026-04-25 round 2):** invoke the existing pipeline once on apex MoE — `cargo run --release --bin hf2q -- convert <apex> --quant dwq-4-6 --measure-only` (or equivalent existing instrumentation) — and record peak RSS during the forward-pass-with-capture step (counting both Q-block weights resident on Metal **and** F32 activation tensors emitted at every SDPA / router / expert-matmul boundary). The recorded number + 10% headroom becomes Decision 6's `test_apex_moe_capture_peak_rss` gate value. Result lands in `docs/peer-parity-baselines-<P0-close-date>.md` and is inlined into Decision 6's bullet.

**Acceptance:** Decisions 1 + 2 criteria met. Apex MoE peak-RSS gate (`≤ 8 GB` during `LazyTensorMap` iteration) passes. Gemma-4 byte-identical-to-eager regression passes. Decision 6's gate value committed to `docs/peer-parity-baselines-<date>.md` and inlined into the Decision 6 body.

**Estimated LOC:** ~400 (impl + tests) + ~50 (measurement spike harness, if not already in `tests/streaming_pipeline_apex_moe.rs`).

### P1 — Lift Phase 1.4–1.7 transforms to lazy

**Scope:** Decision 3.

**Dependency:** P0 green.

**Deliverables:** `src/main.rs:457–582` (worktree) signatures take `&mut LazyTensorMap`; `src/models/qwen35/{dense,moe}.rs` transforms become `fn(LazyTensor) -> LazyTensor`; 6 unit tests covering each transform's lazy-vs-eager byte-identity.

**Acceptance:** Decision 3 criteria met. Synthetic 4-layer Qwen3.5 + Qwen3.5-MoE eager-vs-lazy transform output byte-identical.

**Estimated LOC:** ~400.

### P2 — Streaming quantize loop

**Scope:** Decisions 4 + 7.

**Dependency:** P1 green.

**Deliverables:** `src/quantize/mod.rs::quantize_streaming` (rewrite); `StreamingBackend` trait; `src/backends/{gguf,safetensors_out}.rs` impls; `src/models/qwen35/moe.rs::merge_moe_experts_lazy`; 4 integration tests.

**Acceptance:** Decisions 4 + 7 criteria met. Apex MoE peak RSS during convert ≤ 44 GB. Gemma-4 `--quant q4` byte-identical.

**Estimated LOC:** ~600.

### P3 — Rayon parallelism in quantize loop

**Scope:** Decision 5.

**Dependency:** P2 green.

**Deliverables:** producer / worker-pool / serialiser channel topology in `src/quantize/mod.rs`; 3 integration tests.

**Acceptance:** Decision 5 criteria met. 8-thread quant of apex MoE q4 ≥ 4× faster than 1-thread; 8-thread output byte-identical to 1-thread.

**Estimated LOC:** ~150.

### P4 — Eliminate the P9b intermediate-GGUF dance

**Scope:** Decisions 6 + 8.

**Dependency:** P0 + P2 green.

**Deliverables:** `RealActivationCapture::from_lazy_tensor_map`; `weight_loader::load_from_lazy_tensor_map`; deletion of `src/quantize/intermediate_moe_q8.rs`, `src/main.rs:715–753`, `src/backends/gguf.rs::emit_gguf_from_tensor_map`; 3 integration tests.

**Acceptance:** Decisions 6 + 8 criteria met. Apex MoE DWQ activation capture peak RSS ≤ 35 GB **with no tempfile written**.

**Estimated LOC:** −39 net (700 lines touched).

### P5 — Sensitivity-JSON cache

**Scope:** Decision 9 (DWQ-side caching) — note: this is the per-(model, corpus) cache mentioned in item #5; orthogonal to the calibrator-trait split in P7.

**Dependency:** P2 green.

**Deliverables:** `~/.cache/hf2q/sensitivity/{key}.json` cache; key = SHA-256(model SHA, corpus SHA, sensitivity-algorithm version); cache lookup before forward pass; 4 unit tests.

**Acceptance:** running `hf2q convert <apex> --quant dwq-4-6` then `hf2q convert <apex> --quant dwq-4-8` (different bit-pair, same model) skips the second forward pass and consumes the cached sensitivity map.

**Estimated LOC:** ~300.

### P6 — Imatrix calibrator (pure-Rust port)

**Scope:** Decision 10.

**Dependency:** P0 green (uses `LazyTensorMap`).

**Deliverables:** `src/calibrate/imatrix.rs`; `.imatrix` sidecar emitter; cross-validation gate against `llama-imatrix`; 8 unit tests + 1 cross-validation integration test.

**Acceptance:** Decision 10 criteria met. Cross-validation gate green (per-tensor max-element-difference ≤ 1e-4 vs llama.cpp on a 4-layer synthetic Gemma-4).

**Estimated LOC:** ~600.

### P7 — `Calibrator` × `OutputFormat` orthogonal split

**Scope:** Decisions 9 + 11.

**Dependency:** P6 green.

**Deliverables:**
- `src/calibrate/mod.rs::Calibrator` trait + `NoneCalibrator`, `DwqCalibrator`, `ImatrixCalibrator` impls.
- **Full move (Decision 9 Layout A — Robert lock 2026-04-25 round 2):** `src/quantize/dwq.rs`, `src/quantize/dwq_activation.rs`, `src/quantize/sensitivity.rs`, `src/quantize/apex.rs` move into `src/calibrate/`. No re-exports, no shims, no legacy `pub use` aliases. All callers and tests update to new paths in one pass.
- `src/quantize/output_format.rs::OutputFormat` enum.
- `src/quantize/k_quant.rs` (pure-Rust k-quant codebook).
- Refactor of `cmd_convert` to use the orthogonal pair.
- 12 unit tests + 4 integration tests.

**Acceptance:** Decisions 9 + 11 criteria met. Q4_K_M / Q5_K_M / Q6_K byte-identical-to-llama.cpp (NEON path) gate passes. All `tests/convert_qwen35_*.rs` and other test files compile against the new `src/calibrate/` paths; no `use crate::quantize::dwq::*` lingers.

**Estimated LOC:** ~1300 (was ~1050; +~250 for the `dwq*`/`sensitivity`/`apex` full-move import-path churn — Decision 9 Layout A).

### P8 — CLI rename + final variant menu

**Scope:** Decisions 12 + 13.

**Dependency:** P7 green.

**Deliverables:** `src/cli.rs::QuantMethod` enum updated; deleted variants error with helpful messages; `--calibration` + `--output-format` dev gate behind `HF2Q_UNSAFE_EXPERIMENTS=1`; 8 CLI integration tests.

**Acceptance:** Decisions 12 + 13 criteria met. Old variant names (`apex`, `mixed-2-6`, `dwq-mixed-4-6`) error cleanly; new variant names dispatch to the correct (Calibrator, OutputFormat) tuple.

**Estimated LOC:** ~170.

### P9 — Safetensors backend integration with calibrators

**Scope:** Decision 14.

**Dependency:** P7 green.

**Deliverables:** `SafetensorsBackend` implements `StreamingBackend`; `requires_native_quantization` removed; mlx-lm-compatible quant metadata writer; 6 integration tests including `mlx_lm.load` round-trip.

**Acceptance:** Decision 14 criteria met. `--quant dwq-4-6 --format safetensors` produces a file `mlx_lm.load` accepts; `--quant imatrix-q4_k_m --format safetensors` round-trips through hf2q's own serve with byte-identical logits.

**Estimated LOC:** ~700.

### P10 — Peer-parity benchmark harness

**Scope:** Decisions 15 + 16.

**Dependency:** P3 + P6 + P9 green.

**Deliverables:** `tests/peer_parity_gates.rs` (orchestration); `scripts/peer_parity_run.sh` (cold-cache reboot wrapper); `tools/llama_cpp_runner.rs` (subprocess wrapper around `convert_hf_to_gguf.py` + `llama-quantize`); `tools/mlx_lm_runner.rs` (subprocess wrapper around `mlx_lm.convert`); per-cell measurement collector; markdown table emitter.

**Acceptance:** harness runs end-to-end on a 27B dense smoke model (~6 GB BF16 → ~3 GB Q4_K_M) and emits a complete table for the 8 cells against both peers; numbers are real but not yet meeting the gates (apex MoE comes in P11).

**Estimated LOC:** ~600 + ~100 wrapper scripts.

### P11 — Re-emit ADR-012's four DWQ GGUFs + measured gate close

**Scope:** Decision 22.

**Dependency:** all prior phases green.

**Deliverables:**
- `models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf` (re-emitted under streaming).
- `models/qwen3.6-27b-dwq48/qwen3.6-27b-dwq48.gguf` (re-emitted).
- `models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/...gguf` (re-emitted).
- `models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48/...gguf` (re-emitted).
- Same four under `--format safetensors` for the mlx-lm peer column.
- `docs/peer-parity-results-<close-date>.md` — full 8×3 table, per cell: pre / post / peer numbers + verdict (pass/fail per gate).
- Closing commit message inlines the table verdict and links to the markdown.

**Acceptance:** every cell passes its gate per Decision 15. If any cell fails: per mantra, this phase reopens until it passes — no deferred cells, no "we'll fix the apex MoE imatrix-q4_k_m row in ADR-N+1." The ADR closes only when every cell is green.

**Estimated LOC:** ~200 (mostly artefact production scripts + commit-message templating).

### P12 — Documentation refresh

**Scope:** docs only.

**Dependency:** P11 green.

**Deliverables:**
- `docs/converting-a-model.md` rewritten — generic convert reference using new variant names, calibrator section, dev-gate maintainer note.
- `docs/converting-qwen35.md` updated — qwen35 / qwen35moe canonical commands using new variant names.
- `docs/shipping-contract.md` updated — peer-parity gates section.
- `docs/calibrator-onboarding.md` (new — Decision 19) — how to add AWQ/GPTQ/AQLM in a future ADR.
- ADR-012 status section updated to reflect P11's re-emission of its four GGUFs.

**Acceptance:** every command in the docs runs successfully on a fresh hf2q binary; markdown link checker green.

**Estimated LOC:** ~600 docs (no code).

### Totals

- **Code LOC:** ~5,200 across 13 phases (P0 400 + P1 400 + P2 600 + P3 150 + P4 −39 net + P5 300 + P6 600 + P7 1050 + P8 170 + P9 700 + P10 700 + P11 200 = 5,231).
- **Docs LOC:** ~800 (P12 600 + scattered phase doc updates).
- **Test LOC:** ~1,500 (every phase carries tests; estimate is conservative — ADR-012's tests-to-code ratio was ~0.3, suggesting ~1,560).
- **Net deletions:** ~340 LOC (intermediate_moe_q8 module + P9b dance + emit_gguf_from_tensor_map + apex CLI handling + mixed-N-M variants).

Estimated calendar time at single-author cadence (Robert): ~6–8 weeks if P11 reference artefact production runs in parallel with downstream phases on the same M5 Max.

---

## Test strategy

### Unit tests (per phase)
Every phase carries unit tests covering the new types, traits, and pure functions. Listed inline with each Decision body. Estimated total: ~70 unit tests.

### Regression tests (Chesterton's fence)
Byte-identical regression for the un-calibrated paths against the *current* pipeline:
- Gemma-4 `--quant f16` / `q4` / `q8` snapshot SHAs (already exist for `q4` per ADR-012 P5/P11 fence).
- Qwen3.5 4-layer synthetic `--quant f16` / `q4` snapshot SHAs.

Re-blessed snapshots for calibrated paths (recorded as new SHA in the closing commit; subsequent runs must match):
- `dwq-4-6` / `dwq-4-8` / `dwq-6-8` / `dwq-2-8` on Qwen3.5 4-layer synthetic.
- `imatrix-q4_k_m` / `imatrix-q5_k_m` / `imatrix-q6_k` on Qwen3.5 4-layer synthetic.
- `imatrix-adaptive` on Qwen3.5 4-layer synthetic.

### Integration tests
- `tests/streaming_pipeline_apex_moe.rs` — end-to-end apex MoE convert with peak-RSS instrumentation.
- `tests/lazy_tensor_correctness.rs` — synthetic transforms, lazy vs eager.
- `tests/imatrix_cross_validate.rs` — `≤ 1e-4` against `llama-imatrix`.
- `tests/kquant_byte_identical_llama_cpp.rs` — Q4_K_M / Q5_K_M / Q6_K byte-identical.
- `tests/safetensors_mlx_lm_round_trip.rs` — hf2q safetensors output loads in `mlx_lm.load` (subprocess test, gated `#[ignore]` by default; runs in P10 harness).
- `tests/peer_parity_gates.rs` — full 8-cell gate matrix.

### Specification-driven tests
- Every k-quant variant has a hand-authored synthetic input with a deterministic expected output computed from the llama.cpp algorithm by hand (no comparison against running llama-quantize — the comparison is at the algorithm level, not the binary level). Runs in default cargo test set.

### Real-model tests
- `#[ignore]`-gated real-model gate tests in `tests/peer_parity_gates.rs` for each cell of Decision 15's matrix. Run only via `scripts/peer_parity_run.sh` on a real M5 Max with the reference models present. These produce P11's closing artefacts.

---

## Risks and mitigations

### R1: `LazyTensor` API churn during P1–P9 invalidates downstream consumers
**Probability:** medium. The trait will see usage we didn't anticipate.

**Mitigation:** lock the `LazyTensor` API at P0 close via `#[deny(...)]` doctest examples covering every public method. Any breaking change after P0 requires a Decision-23 amendment to this ADR (no silent API edits).

### R2: Streaming write atomicity broken under SIGINT
**Probability:** low. POSIX rename is atomic; the tempdir pattern is well-tested.

**Mitigation:** Decision 20's `test_cancellation_no_partial_at_final_path` runs SIGINT mid-stream 100× in CI; any leak fails the test.

### R3: Imatrix port has subtle math divergence from llama.cpp's C impl
**Probability:** medium. Float ordering and accumulation patterns differ between Rust and C.

**Mitigation:** Decision 10's `≤ 1e-4` cross-validation gate. If the port diverges by more: per mantra, fix the port — do not accept "comparable" until it's byte-precise.

### R4: DWQ peer-parity gate against mlx-lm requires identical calibration corpus
**Probability:** high. mlx_lm.convert uses its own corpus by default.

**Mitigation:** P10 harness calls `mlx_lm.convert` with `--calibration-data` pointing at hf2q's `tests/fixtures/ppl-corpus/wikitext2.tokens` (mlx-lm supports this). If a corpus mismatch is unavoidable, the gate becomes "PPL on the same eval corpus" not "PPL after the same calibration corpus" — documented in Decision 16.

### R5: Sensitivity cache invalidation false-negative
**Probability:** low. SHA-256 keying is robust.

**Mitigation:** key includes the *sensitivity algorithm version* string (manually bumped on any change to `sensitivity.rs`); cache miss on version mismatch.

### R6: Rayon contention on the serialiser thread
**Probability:** medium for very-fast quantizers (Q4 flat). Producer outpaces serialiser.

**Mitigation:** bounded channels in Decision 5 — back-pressure flows naturally. P3 measures and tunes channel capacity.

### R7: Calibrator trait too narrow for AWQ
**Probability:** medium. AWQ is per-channel quant-aware and may need access to the *quantizer* during calibration (not the trait we designed for).

**Mitigation:** P7 design review writes a mock AWQ calibrator implementing the trait. If AWQ doesn't fit cleanly, the trait grows in P7 — better than discovering it later. No AWQ implementation lands; the door stays shut.

### R8: Determinism gate fails on un-calibrated paths
**Probability:** low. The pipeline is mathematically equivalent; only ordering changed.

**Mitigation:** if it fails, the bug is real and gets fixed at the root (`BTreeMap` iteration order, write-buffer flushing, stable threading). Mantra: never lower the bar.

### R9: Real-model peer-parity gates fail on apex MoE
**Probability:** medium. Apex MoE is the largest, hardest case; imatrix may underperform DWQ on hybrid arch.

**Mitigation:** per mantra, fix until they pass. If imatrix on apex MoE genuinely loses to llama.cpp imatrix on the same model, the bug is in our port (Decision 10) and gets fixed. If DWQ on apex MoE genuinely loses to mlx-lm DWQ on the same model, the bug is in our port and gets fixed. There is no "ship with a degraded variant"; the variant is removed from the menu before that happens.

### R10: Safetensors backend integration breaks Gemma-4 safetensors output
**Probability:** medium. Gemma-4 safetensors is a working path today.

**Mitigation:** Decision 17's `test_safetensors_streaming_byte_identical_to_eager` for Gemma-4 `--quant f16 --format safetensors` is a hard gate. P9 closes only when this passes.

### R11: Cross-validation against `llama-imatrix` requires `llama.cpp` checked out and built
**Probability:** environment risk; not algorithmic.

**Mitigation:** P10 harness checks `llama-imatrix` is on `$PATH`; if not, the cross-validation test is `#[ignore]`-skipped with a clear log message. CI nodes used for ADR-014 close run with both `llama.cpp` and `mlx-lm` installed.

### R12: ADR-013's `RealActivationCapture` API constraints in `from_lazy_tensor_map`
**Probability:** low — ADR-013 is closed.

**Mitigation:** Decision 8 specifies the constructor doesn't change the trait. If `Qwen35Model::load_from_lazy_tensor_map` needs internal API changes in `weight_loader.rs`, those are local to the qwen35 module and don't propagate to ADR-013's trait surface.

### R13: P11 reference artefact production exhausts disk
**Probability:** medium. Apex MoE BF16 + 4 DWQ GGUFs + 4 safetensors variants + 4 llama.cpp Q4_K_M + 4 mlx-lm DWQ ≈ 200+ GB of artefacts.

**Mitigation:** P11 runs sequentially, deletes intermediate artefacts (`<peer>-Q4_K_M.gguf` after PPL is measured), keeps only the 8 final files. ADR-012's existing 150-GB disk preflight (Decision 14) covers the working set.

### R14: ADR-012 P9 final close (real-model artefact production) collides with ADR-014 P11
**Probability:** high — they target the same models on the same hardware.

**Mitigation (Robert lock 2026-04-25 round 2):** **ADR-014 P0 starts only after ADR-012's four DWQ GGUFs verifiably load in `llama-cli`** — not just after the worktree-merge commit, and not on the basis of the status-header claim alone. The current header line `feat(adr-005 phase 2c iter 100)` cohort says "engineering complete, 4 DWQ GGUFs delivered" (commit `38d2f3c`), but ADR-012 line 7's real-model audit says "every previously-shipped DWQ GGUF in `models/qwen3.6-{27b,35b-a3b-abliterix-ega-abliterated}-dwq{46,48}/` fails to load in `llama-cli`." The audit, not the status header, is the entry gate. P0 entry criterion: `llama-cli --model models/<each of the 4 DWQ GGUFs> -p "Hello" -n 16` succeeds on each artefact. P11 starts on `main` after that, sequenced not parallel on the M5 Max.

---

## Open questions

These are tracked here, not deferred. Each gets resolved during the Phase that touches it; no question survives this ADR's close.

1. **(P1) Does `convert_bf16_to_f16` (`src/main.rs:419`) lift to lazy cleanly, or does it need a special-case eager path?** Resolution due in P1.
2. **(P5) Cache eviction policy for `~/.cache/hf2q/sensitivity/`.** LRU? Manual purge? Resolution due in P5.
3. **(P6) Does ADR-013's CPU forward at `src/inference/models/gemma4/forward_cpu.rs` produce activations precise enough for imatrix cross-validation `≤ 1e-4`, or do we need a higher-precision F32 forward for calibration?** Resolution due in P6.
4. **(P7) Does the `Calibrator` trait need a `prepare(&mut self, model: &LazyTensorMap)` lifecycle method separate from `calibrate(...)`, to allow streaming calibration alongside streaming quant?** Resolution due in P7.
5. **(P10) Does `mlx_lm.convert --calibration-data <file>` accept hf2q's wikitext2 token file format directly, or do we need to re-emit as plain UTF-8 text?** Resolution due in P10.

---

## Dependencies on other work (cross-ADR)

- **ADR-005 (inference server):** unchanged. ADR-014 produces artefacts that ADR-005's serve loads; the load path is downstream.
- **ADR-007 / ADR-009 (TQ KV state, hybrid memory backend):** unchanged. ADR-014 is convert-side only.
- **ADR-012 (qwen35moe conversion):**
  - Closes first on the current pipeline (P8 + P9 real-model artefact production).
  - ADR-014 P11 re-emits ADR-012's four DWQ GGUFs under the streaming pipeline as part of its own AC.
  - ADR-012's `tests/quality_thresholds.rs`, `tests/calibration_eval_disjoint.rs`, `tests/convert_qwen35_real_activation_capture.rs` are inherited unchanged — they remain valid against the streaming pipeline by Decision 17 (re-blessed for calibrated paths but the threshold *constants* are unchanged).
  - ADR-012's `IntermediateMoeQ8Quantizer` band-aid (P9b) is **deleted** by ADR-014 P4.
- **ADR-013 (qwen35 inference):**
  - `RealActivationCapture` trait unchanged.
  - New constructor `RealActivationCapture::from_lazy_tensor_map` is added; named cross-ADR API delta (Decision 8).
  - `Qwen35Model::load_from_lazy_tensor_map` is added in `src/inference/models/qwen35/weight_loader.rs`.

---

## Glossary

- **Lazy tensor:** A tensor whose bytes have not been materialised; carries shape/dtype metadata and a `FnOnce` closure that will produce the bytes on demand.
- **Materialise:** Invoke the closure to produce the tensor bytes; consumes the `LazyTensor`.
- **Streaming:** Per-tensor pipeline where bytes flow `read → transform → quantise → write → drop` without buffering the whole model.
- **Calibrator:** A pure-Rust component that consumes a model + corpus and produces calibration data (per-column weights, per-tensor sensitivity, etc.) used by the quantiser.
- **Output format:** The on-disk codebook (Flat, BitPair, KQuant, KQuantAdaptive) — orthogonal to the calibrator.
- **Peer:** llama.cpp (`convert_hf_to_gguf.py` + `llama-quantize --imatrix`) and mlx-lm (`mlx_lm.convert --quant-method dwq`).
- **Peer parity:** Measured wall-clock, peak RSS, and PPL gates against a peer's output on the same model + hardware. Locked in Decision 15.
- **DWQ:** Distilled Weight Quantization (Apple/MLX research). hf2q port at `src/quantize/dwq.rs`.
- **Imatrix:** Importance matrix calibration (llama.cpp). hf2q port lands in P6 at `src/calibrate/imatrix.rs`.
- **K-quant:** llama.cpp's super-block-grouped quant codebook (Q4_K_M, Q5_K_M, Q6_K).
- **Sovereignty (toolchain):** Pure-Rust impl, no Python runtime, no link to libggml. Locked in Decision 21.
- **Sovereignty (algorithm) [does not exist]:** hf2q does not claim sovereignty over DWQ or imatrix algorithms — both are published peer techniques ported into hf2q.
- **Off-diagonal cell:** A `(Calibrator, OutputFormat)` pair that is not exposed as a named CLI variant (e.g. DWQ + KQuant). Reachable only via `HF2Q_UNSAFE_EXPERIMENTS=1`.

---

## Appendix A: Target convert commands (once all phases land)

### Qwen3.6-27B dense — llama.cpp peer column (GGUF)
```bash
hf2q convert Qwen/Qwen3.6-27B --quant imatrix-q4_k_m
# emits: Qwen3.6-27B-imatrix-q4_k_m.gguf
# peer:  llama-quantize --imatrix wikitext2.imatrix model-f16.gguf model-q4_k_m.gguf q4_k_m
```

### Qwen3.6-27B dense — mlx-lm peer column (safetensors)
```bash
hf2q convert Qwen/Qwen3.6-27B --quant dwq-4-6 --format safetensors
# emits: Qwen3.6-27B-dwq-4-6/  (mlx-lm-loadable directory)
# peer:  mlx_lm.convert --hf-path Qwen/Qwen3.6-27B --mlx-path . --quantize --quant-method dwq --bits 4
```

### Qwen3.6-35B-A3B apex MoE — both peer columns
```bash
hf2q convert jenerallee78/Qwen3.6-35B-A3B-abliterix-ega-abliterated-apex --quant imatrix-q4_k_m
hf2q convert jenerallee78/Qwen3.6-35B-A3B-abliterix-ega-abliterated-apex --quant dwq-4-6
hf2q convert jenerallee78/Qwen3.6-35B-A3B-abliterix-ega-abliterated-apex --quant dwq-4-6 --format safetensors
```

### Auto routing
```bash
hf2q convert Qwen/Qwen3.6-27B --quant auto
# resolves: imatrix-q4_k_m (dense, ≤30B, any RAM)

hf2q convert jenerallee78/Qwen3.6-35B-A3B-...-apex --quant auto
# resolves: dwq-4-6 (MoE, < 96 GB RAM) or dwq-4-8 (MoE, ≥ 96 GB RAM)
```

### Dev gate (off-diagonal cells)
```bash
HF2Q_UNSAFE_EXPERIMENTS=1 hf2q convert Qwen/Qwen3.6-27B \
    --calibration imatrix --output-format bit-pair-4-6
# emits: Qwen3.6-27B-experimental-imatrix-bp-4-6.gguf
# warning: experimental, not peer-validated
```

---

## Appendix B: Canonical gotcha cross-reference (will accumulate during P0–P11)

(Empty at proposal time; populated as phases close. Each entry is one bullet citing the file:line of the gotcha and the Decision number that first addressed it. Mirrors ADR-012's Appendix B pattern.)

---

**This ADR closes when:**
1. All 13 phases (P0–P12) are green.
2. Decision 15's 8-cell peer-parity table is filled in `docs/peer-parity-results-<date>.md` with measured numbers, every gate passing.
3. The four ADR-012 reference DWQ GGUFs are re-emitted under the streaming pipeline and live in `models/`.
4. `cargo test` is green; `cargo clippy` is zero new warnings vs. ADR-014 baseline; `tests/sovereignty_*.rs` confirms no Python/torch/libggml link.
5. The closing commit message inlines the verdict table and links to `docs/peer-parity-results-<date>.md`.

Per mantra, no other close condition exists. No "shipped except for cell X." No "deferred apex MoE imatrix-q4_k_m row." No "good enough." Just measured peer parity, end-to-end, on the day this ADR closes.
