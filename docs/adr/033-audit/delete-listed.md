# Audit: ADR delete-list external callers (P-1, ADR-033)

ADR-033 §"P6 — Delete superseded code" delete-lists these five files. This audit confirms whether each is cleanly deletable by enumerating external callers (outside `src/quantize/`).

Pinned external references: hf2q HEAD `85bee70e` (post-ADR-033-rewrite commit).

| File | LOC | External callers | Disposition | Notes |
|---|---|---|---|---|
| `src/quantize/dwq_k_quantizer.rs` | 883 | `src/main.rs:2365,2368,2371,2374,2409` (5 references, all inside one dispatcher arm) | DELETE | clean — delete with the main.rs dispatcher arm in P6 |
| `src/quantize/k_quant_codec_quantizer.rs` | 953 | `src/main.rs:2167` (dispatcher arm) + `src/quality/mod.rs:1331,1463` (two quality-test sites) + `src/backends/gguf.rs:1334,2075,4835` (THREE writer-side branches on `METHOD_K_QUANT_CODEC_DIRECT`) | DELETE | requires P2 (seek-back writer) to land first — the three `gguf.rs` branches at 1334/2075/4835 are OUTSIDE the writer slice 282-1259 audited by the gguf-writer agent; they're the "iter-99 codec-direct re-entry" branches that the seek-back writer eliminates by NOT having a method-direct fast path |
| `src/quantize/mixed.rs` | 520 | `src/calibrate/dwq.rs:18` (only) | DELETE | `src/calibrate/dwq.rs` is itself delete-listed per ADR §P6 ("hf2q's homebrew DWQ; reserve `--quant dwq` for future real-DWQ ADR"). Deletion chains. |
| `src/quantize/static_quant.rs` | 468 | `src/main.rs:1104` (dispatcher) + `src/calibrate/apex.rs:22` (**EXISTING apex calibrator NOT on ADR delete-list**) | DELETE-AFTER-RESOLVE | **OPEN QUESTION for synthesizer:** `src/calibrate/apex.rs` is hf2q's existing apex calibrator. ADR-033 §P6 doesn't list it for deletion but the new `ApexPolicy` doesn't use a separate calibrator — it uses the unified `Quantizer` trait per Decision §2. Either (a) `src/calibrate/apex.rs` is also delete-listed and the ADR P6 list should be extended, or (b) it survives and rewrites against the new trait. Default presumption: DELETE (it duplicates what the new policy does). |
| `src/quantize/variant_quantizer.rs` | 604 | `src/cli.rs:1263` (doc-comment reference only) + `src/main.rs:2225` (dispatcher arm) | DELETE | clean — delete with the main.rs dispatcher arm in P6; remove the dangling doc-comment ref in cli.rs |

**Totals:** DELETE: 5 files, 3,428 LOC (sum: 883 + 953 + 520 + 468 + 604).

## Notes for the synthesizer

1. The `k_quant_codec_quantizer.rs` external footprint extends BEYOND the writer slice the `gguf-writer` agent audits. The three `gguf.rs` branches at 1334/2075/4835 need a separate delete-pass in P6, not just removal of the writer slice 282-1259. The new seek-back writer will not have a `METHOD_K_QUANT_CODEC_DIRECT` fast path — the trait-dispatch model removes the need for one. Synthesizer should add a row to the gguf-writer audit covering these three branches OR add a separate "P6 cross-file cleanup" section.

2. `src/calibrate/apex.rs` is the load-bearing P-1 surprise. The ADR §"Plan" / Pa says "imatrix subsystem (Pi) reuses hf2q's existing forward-pass code on mlx-native" but is silent on the existing `src/calibrate/apex.rs`. The ADR §P0 introduces `src/quantize/apex/<arch>.rs` for tensor classification (NOT a calibrator). Either the existing apex calibrator survives and gets refactored, or it's redundant under the new policy and gets deleted. **Recommend: ADR amendment to clarify.** Until then, the audit's working assumption is DELETE.

3. `src/quality/mod.rs:1331,1463` calls `KQuantCodecQuantizer` in quality-test code. Those test sites need to be rewritten against the new `Quantizer` trait when P6 lands. They're not delete-list themselves; they're MODIFY targets in the new policy world.

4. `src/cli.rs:1263` has a dangling doc-comment reference to `VariantKQuantizer`. Remove in P6 cleanup.

5. The delete-listed `src/calibrate/dwq.rs` is a transitive dependency of the `mixed.rs` deletion chain. Confirmed delete-listed in ADR-033 §P6 ("hf2q's existing homebrew DWQ ... Deleted in P6").
