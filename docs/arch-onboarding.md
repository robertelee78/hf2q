# Adding a new model family to hf2q

> **ADR-012 Decision 20 canonical checklist** — derived from the
> qwen35 / qwen35moe registration diff landed in 2026-04-24's P8
> scaffolding commit `ebec4a1`.
>
> This document is the "one-file-per-arch add" guide. The arch
> registry front-loads the conformance surface so every new family
> (Gemma4 parity, Ministral ADR-015, DeepSeek-V3 ADR-016, Qwen3.7,
> …) costs roughly **~50 LOC registry registration + ~200–400 LOC
> arch-specific transforms** rather than the ~1500 LOC harness
> rewrite every single new arch paid pre-`src/arch/`.

---

## Sovereignty ground rules (read before writing any code)

Per `feedback_hf2q_sovereignty.md` (tightened 2026-04-23) and
ADR-012 §Engineering Mantra:

- **Pure Rust** in hf2q and mlx-native. No code, crate, binary, or
  build artifact from `/opt/candle` or `/opt/llama.cpp` enters our
  deliverables. This applies at build time, test time, AND CI time.
- **Spec sources are read-only.** `llama-arch.cpp`, `convert_hf_to_gguf.py`,
  `clip.cpp`, `clip-model.h`, `llama-model.cpp` are **read** to derive
  mathematical specs. Every catalog entry you transcribe cites the
  source line (see `src/arch/entries/qwen35.rs` for the canonical
  citation format).
- **No external oracles in tests.** Correctness is proven by:
  1. Hand-authored expected values (for permutation / transpose tests).
  2. Spec-driven synthetic inputs (for shape + broadcast tests).
  3. Round-trip gates (emit → load via our own loader).
  Never by "compare to llama.cpp's output on this input."
- **No stubs.** A populated placeholder is still a stub. Ship the
  arch fully convert-capable or ship nothing.

---

## Checklist: onboard `<arch>` (replace with `ministral`, `deepseekv3`, …)

### 1. Open an ADR (docs/ADR-NNN-<arch>-conversion.md)

ADR opens with:
- Business reason (who asked for it, target deliverable GGUFs).
- Architectural problem (what's different from qwen35 / Gemma4).
- Reference-implementation citations (Python convert class chain +
  llama-arch.cpp LLM_ARCH_* line numbers + convert_hf_to_gguf.py
  class hierarchy).
- Phase plan — copy the P0-shipped-first / P1-config-ingestion /
  P2-module-scaffold / P3-transforms / P4-metadata-emission /
  P5-variant-merge / P6-DWQ-calibration / P7-docs template from
  ADR-012. Adjust for arch's unique surface.
- Non-goals.

### 2. Add HF config parsing (src/input/config_parser.rs)

- Extend `ModelMetadata` only if the arch introduces genuinely-new
  hparams (ideally not). Prefer `Option<T>` for arch-specific fields
  so non-target arches stay `None`.
- Extend `validate_required_<arch>_fields()` with the arch's
  mandatory set. Name the function publicly so the preflight path
  can call it.

### 3. Add the arch transform module (src/models/<arch>/)

```
src/models/<arch>/
├── mod.rs       — <Arch>ConvertContext, ConvertError variants, layer-kind helpers
├── dense.rs     — hf_tensor_name_to_gguf_dense + emit_metadata_dense (if applicable)
└── moe.rs       — hf_tensor_name_to_gguf_moe + expert merge + emit_metadata_moe (if MoE)
```

Structure mirrors `src/models/qwen35/`. Every mapper table entry
needs a `// citation: llama-arch.cpp:LINE LLM_TENSOR_XXX` comment
above it. Hand-transcribed values fail loudly if the spec source
drifts upstream — the test catalog catches the delta.

### 4. Wire the arch into the GGUF backend (src/backends/gguf.rs)

- Extend `layer_map_for_arch()` with the arch's post-attention /
  FFN norm mapping. Qwen3.5 maps `post_attention_layernorm` to
  `post_attention_norm`; Ministral / DeepSeek have their own
  convention — cite the llama-arch.cpp constant.
- If the arch has MTP, extend the `mtp.layers.` dispatch in
  `hf_name_to_gguf` — the existing branch handles qwen35 /
  qwen35moe; add your arch to the `if arch == "..." || ...` guard.
- Extend `emit_metadata()` to call the arch-specific metadata
  emitter (`emit_metadata_dense/moe`).

### 5. DWQ integration (src/quantize/dwq.rs)

- Add `DwqArch::<Arch><Variant>` if the arch needs architecture-
  specific DWQ sensitivity priors. See ADR-012 Decision 12 for
  the SSM/router/shared-expert cohort split as a template.
- `DwqArch::from_hf_architecture()` must match the arch's HF
  `architectures[0]` string BEFORE any generic fallback.
- `requires_activation_capture()` returns `true` if the arch needs
  real forward-pass activations (hybrid / MoE architectures); `false`
  for weight-space-only (Llama-class dense).

### 6. Register in src/arch/

This is the point of Decision 20 — each new arch adds ONE file:

```rust
// src/arch/entries/<arch>.rs
use crate::arch::catalog::{LayerScope, TensorCatalog, TensorCatalogEntry, TensorDtype};
use crate::arch::registry::{ArchEntry, EvalCorpus, QualityThresholds};

const CATALOG: TensorCatalog = TensorCatalog {
    entries: &[
        TensorCatalogEntry { name_template: "...", scope: ..., dtype: ..., citation: "..." },
        // ... hand-transcribed per llama-arch.cpp LINE
    ],
};

pub const ENTRY: ArchEntry = ArchEntry {
    arch: "<arch>",
    // List EVERY HF architecture alias that maps to this arch. Multimodal
    // checkpoints typically ship `*ForConditionalGeneration` while text-
    // only ones ship `*ForCausalLM`; both must resolve to the same
    // ArchEntry or `get_by_hf_architecture` diverges from `arch_gguf_name`.
    // See ADR-012 fix `57d4bcc` for the qwen35 / qwen35moe alias divergence
    // bug that motivates listing both.
    hf_architectures: &["<Arch>ForCausalLM", "<Arch>ForConditionalGeneration"],
    tensor_catalog: &CATALOG,
    has_mtp: false,
    has_vision: false,
    smoke_prompts: &["..."],
    ppl_corpus: EvalCorpus { id: "...", token_count: 512, sha256_hex: "..." },
    quality_thresholds: QualityThresholds::ADR_012_DEFAULT,
    disk_floor_gb: ..., // per Decision 14 model-class routing
    hf_repos: &["<repo>"],
};
```

Then add two lines to `src/arch/entries/mod.rs`:

```rust
pub mod <arch>;
```

And one line to `src/arch/registry.rs::GLOBAL_REGISTRY`:

```rust
&super::entries::<arch>::ENTRY,
```

### 7. Tests to add

**Unit tests (inside `src/arch/entries/<arch>.rs`):**
- `catalog_has_expected_entry_count` — pins the catalog size so
  silent additions cannot drift.
- `<real_model>_tensor_count_folds_correctly` — folds known
  hparams through `expected_tensor_count()` and asserts the exact
  production model's loaded-tensor count. Catches off-by-one in
  catalog scope calculations.
- `hf_architectures_routes_correctly` — `get_by_hf_architecture`
  returns the right entry.

**Integration tests (`tests/convert_<arch>_integration.rs`):**
- Synthetic tiny model (4 layers, hidden=64) convert → read GGUF
  header → assert expected tensor names + metadata keys present.
- Sidecar preservation (if applicable).
- Format / quant matrix (F16, Q4_0, any new arch-gated variant).

**Integration tests (`tests/smoke_conformance.rs`):** extend if any
new failure mode; the existing harness already covers unknown-arch
uniform rejection for any unregistered key.

### 8. Docs to write

- `docs/converting-<arch>.md` — canonical convert commands + smoke
  test section (mirror `docs/converting-qwen35.md`). One section per
  variant (dense / MoE).
- `docs/shipping-contract.md` — append an arch-specific acceptance
  block.
- Update `docs/ADR-NNN-<arch>-conversion.md` phase-status table as
  each P0→P7 ships.

### 9. Gate via `hf2q smoke`

```bash
HF_TOKEN=xxx cargo run --release -- smoke --arch <arch> --quant q4_0
```

Once this exits 0 and produces
`tests/fixtures/smoke-transcripts/<arch>-q4_0.txt` byte-identically
across two fresh runs, the arch is shipped.

### 10. Gemma-4 regression check

Every arch-onboarding commit MUST demonstrate Gemma4 byte-
identical output (SHA-256 on a representative convert). Regression
test lives under `tests/convert_gemma4_regression.rs`; add assertion
that the new arch's code paths do not touch Gemma4's. This is the
load-bearing proof that new-arch code does not silently break
shipped products.

---

## Anti-patterns to avoid

| Anti-pattern | Why it fails |
|---|---|
| `blk.mtp0.nextn.*` placeholder for MTP block index | llama.cpp + our own loader expect `blk.{num_hidden_layers}.nextn.*` — placeholder silently drops MTP tensors. Caught by ADR-012 P11; re-introduction forbidden by the round-trip gate's negative assertion. |
| Copy-pasting llama.cpp C++ into Rust | Sovereignty violation. Read C++ → derive spec → write Rust from spec. No substring transliteration. |
| `todo!()` / `unimplemented!()` in the convert path | P4-class stub. Mantra: "No stub, no fallback." Either ship conversion-capable or the phase hasn't shipped. |
| Placeholder arch entries (e.g. empty `gemma4.rs`) | Populated stub is still a stub. Decision 20 acceptance: registry contains ONLY fully-populated arches. |
| Shared build dir swarm parallelism | Disjoint file claims are insufficient. Two agents compiling against the same `target/` racing `cargo build` will corrupt the output. Sequence workers when they share a Cargo workspace. |
| Smoke harness in bash | Decision 16 acceptance §"Ownership" rejects bash — the harness is a Rust binary subcommand. Testable, single-entry, dispatched via `ArchRegistry::get(arch)`. |
| External mmproj oracle / llama-cpp output as correctness reference | Sovereignty violation. Correctness = hand-authored expected values + spec-driven synthetic tests + round-trip through our own loader. |
| Reading `metadata.layer_types` directly when you need to know whether a layer is full / linear attention | Decision 2 contract violation: the parser populates raw `layer_types` as `["attention"; N]` for any config without an explicit per-layer enumeration, so a Qwen3.5-style config that supplies only `full_attention_interval` silently bypasses anything reading the raw field. Use `metadata.resolved_layer_types()` — verified against this trap by ADR-012 fixes `83b6618` (preflight hybrid validator) and `ae3a9cf` (format_info diagnostic). |
| Listing only `*ForCausalLM` in `hf_architectures` | Multimodal checkpoints ship `*ForConditionalGeneration`. Registry-side `get_by_hf_architecture` will diverge from GGUF-side `arch_gguf_name` if you list only one alias. List EVERY known alias for the arch — see ADR-012 fix `57d4bcc`. |
| Duplicating the arch-string string-set check in multiple call sites | Drift across copies silently corrupts the convert pipeline (e.g. one site applies V-head reorder for an alias but another site skips expert merge). Centralize via a `pub(crate) fn is_<arch>_architecture(arch, model_type) -> bool` and call it from every gate site. See `is_qwen35_family_architecture` / `is_qwen35moe_architecture` in `src/models/qwen35/mod.rs` (commit `d37daa4`). |

---

## References

- `src/arch/entries/qwen35.rs` — canonical example (dense variant).
- `src/arch/entries/qwen35moe.rs` — canonical example (MoE variant).
- `docs/ADR-012-qwen35moe-conversion.md` — full Decision set (20 decisions, 12 phases).
- `docs/converting-qwen35.md` — canonical user-facing convert guide.
- `src/arch/smoke.rs` — the dispatcher every arch inherits for free.
- `docs/shipping-contract.md` — product-level acceptance clauses.
