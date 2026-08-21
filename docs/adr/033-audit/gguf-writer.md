# Audit: src/backends/gguf.rs:282-1259 writer slice (P-1, ADR-033)

Pinned external references: GGUF spec v3 (matches `const GGUF_VERSION: u32 = 3` at src/backends/gguf.rs:23). Two-pass writer slice the ADR §"Plan" P2 calls out is replaced by a seek-back single-pass writer at the new path `src/backends/gguf/writer.rs`.

## Scope note

The slice 282–1259 contains two complete two-pass writers and their helpers:

- `Backend::write` (L282–L738) — the dense / text GGUF writer
- `write_mmproj_gguf` (L887–L1189) — the parallel mmproj GGUF writer (separate file output for vision tensors)

Both share the same two-pass shape (size-predict → write-tensor-info → repack-and-zero-pad) and both should be replaced by ONE seek-back writer parameterized over `{tensor selection, metadata builder}`. The mmproj path is structurally the same bug-class; deleting one without the other leaves the bug-class half-alive.

## Region table

| Region (line range) | LOC | Disposition | Rationale (one line) |
|---|---|---|---|
| `Backend::write` path/dir resolution (L282–L313) | 32 | KEEP | Filename/dir handling reused by new writer; unchanged. |
| Text-tensor filter (inline vision/audio skip list) (L315–L335) | 21 | MODIFY | Vision-pattern membership moves to `crate::quantize::layer_mix::is_vision_tensor_pattern` per ADR Decision §"Vision tensor pattern"; the inline `n.contains("vision_tower") \|\| ...` list is the SECOND vision-pattern source — ADR P-1 requires a single canonical source. New writer's dispatcher checks this above the policy. |
| `generate_synthetic_tensors` call + Gemma4 V=K duplicates (L337–L371) | 35 | KEEP | Arch-specific synthetic-tensor + V=K-duplicate logic survives; new writer streams these too. (`generate_synthetic_tensors` body is outside slice at L1983; flagged for mod.rs agent territory.) |
| Progress bar setup (L373–L376) | 4 | KEEP | UX; reused. |
| `build_metadata` call + provenance append + counts (L378–L399) | 22 | KEEP | Metadata-build is byte-format-stable; reused by seek-back writer. (`build_metadata` body at L3356 is outside slice — mod.rs agent territory.) |
| File creation + `BufWriter` setup (L401–L404) | 4 | MODIFY | New writer needs `File` (not `BufWriter`) because seek-back requires `Seek`; otherwise unchanged. |
| Header magic+version+counts write (L406–L410) | 5 | MODIFY | Same 5 writes happen in the seek-back writer, but counts are written into reserved positions; otherwise byte-identical. |
| Metadata KV-pair encoding loop (L412–L415) | 4 | KEEP | Calls `write_metadata_kv` (L4007, outside slice) whose byte format is GGUF-spec v3; reused unchanged. |
| **Pass-1 size predictor + per-tensor type-resolution loop (L417–L567)** | **151** | **DELETE** | Pre-allocated offset table is exactly the iter-99 / Bug-B-sequel bug class home. `ggml_tensor_size` table at L1889 (outside slice) is the size-predictor; the seek-back writer eliminates pre-prediction by construction. |
| Inline F16 fallback at K-quant misalignment (L496–L502) | 7 | DELETE | Silent F16 demotion violates ADR §"No-fallback rule"; `shape_fallback` becomes typed error in `QuantPolicy::target_for`. iter-99 bug class. |
| Inline F16 fallback at block-32 misalignment (L511–L521) | 11 | DELETE | Same — silent F16 demotion; same bug class. |
| Pass-1 sizing for synthetic tensors (L569–L580) | 12 | DELETE | Pre-allocated offset table; replaced by streaming write in seek-back writer. |
| Pass-1 sizing for V=K duplicate tensors (L582–L612) | 31 | DELETE | Same bug class — predicts size of a duplicate-of-K tensor and zero-pads on mismatch. |
| `write_tensor_info` offset-table flush (L614–L617) | 4 | MODIFY | Offset-table write moves AFTER tensor data is streamed (the seek-back step); the per-tensor encoding helper `write_tensor_info` (L4174, outside slice) is byte-format-stable and KEPT. |
| Header-end padding to ALIGNMENT (L619–L627) | 9 | KEEP | Alignment padding before data block is GGUF-spec; the seek-back writer needs the same. |
| **Pass-2 payload loop with zero-pad fallback (L629–L652)** | **24** | **DELETE** | `if current < target { write zeros }` at L639–L641 is the literal zero-pad fallback the ADR P2 names; new writer streams in one pass, no pad-to-predicted-offset. **Bug-B-sequel zero-pad target.** |
| Pass-2 synthetic-tensor zero-pad (L654–L664) | 11 | DELETE | Same zero-pad pattern (L659–L661) for synthetics; same bug class. |
| Pass-2 V=K duplicate zero-pad (L666–L692) | 27 | DELETE | Same zero-pad pattern (L677–L679) for V=K duplicates; same bug class. The K→V data clone itself (L682–L690 calling `repack_to_ggml_blocks`) survives as a streaming step in the new writer. |
| Flush + file-size + manifest assembly (L694–L719) | 26 | KEEP | Output-manifest assembly is post-write bookkeeping; reused. |
| Vision-presence check + mmproj dispatch (L720–L730) | 11 | MODIFY | Vision-pattern membership check duplicates the inline filter at L315–L335 — both move to `is_vision_tensor_pattern` per ADR. The dispatch to a separate mmproj writer is preserved. |
| `OutputManifest` return (L732–L738) | 7 | KEEP | Unchanged. |
| `mmproj_path_from_text` (L749–L753) | 5 | KEEP | Pure path helper; reused unchanged. |
| `build_mmproj_metadata` (L760–L876) | 117 | KEEP | Mmproj-specific metadata builder (clip arch + vision geometry); byte-format-stable; reused unchanged. |
| `write_mmproj_gguf` path/dir + filter + progress (L887–L927) | 41 | MODIFY | Vision-tensor filter at L902–L909 is the THIRD copy of the vision-pattern list — consolidates to `is_vision_tensor_pattern`. Otherwise structurally same as the text path's setup. |
| `write_mmproj_gguf` metadata + header write (L929–L947) | 19 | KEEP | Same byte-format as text path; reused. |
| **`write_mmproj_gguf` Pass-1 size predictor + type resolution (L949–L1104)** | **156** | **DELETE** | Pre-allocated offset table for vision tensors — same iter-99 / Bug-B-sequel bug class. Includes inline F16→F32 promotion for clamp scalars (L978–L989), 1-D / norm / position-embd promotion (L991–L1018), and patch-embd Conv2d 2D→4D reshape (L1020–L1072). The PROMOTION RULES survive as type-resolution policy (move into a mmproj-specific dispatcher above the policy, MODIFY), but the pre-allocate-then-zero-pad PATTERN is deleted. |
| `write_mmproj_gguf` offset-table flush (L1106–L1109) | 4 | MODIFY | Same as text path — offset-table write moves AFTER streaming in seek-back writer. |
| `write_mmproj_gguf` header padding to ALIGNMENT (L1111–L1122) | 12 | KEEP | Alignment padding before data block; same as text path. |
| **`write_mmproj_gguf` Pass-2 payload loop with zero-pad (L1124–L1173)** | **50** | **DELETE** | Same `if current < target { write zeros }` pattern at L1132–L1134; same bug class. The HWC→CHW transpose call at L1149–L1169 survives as a streaming step (MODIFY into the per-tensor write path of the new writer). |
| `write_mmproj_gguf` flush + manifest (L1175–L1189) | 15 | KEEP | Post-write bookkeeping; reused. |
| `transpose_patch_embd_hwc_to_chw` (L1203–L1227) | 25 | KEEP | Pure byte-permute utility for gemma4v patch-embd; called by the streaming write path in new writer; unchanged. |
| `ggml_type_from_name` (L1237–L1259+) | 23 | MODIFY | String→numeric mapping moves to `GgmlType::from_str` / `TryFrom<&str>` on the new `GgmlType` enum (ADR Decision §"Per-tensor IR"); behavior preserved but loses the loose `Q3_K_S \| Q3_K_M \| Q3_K_L → Q3_K` aliasing (those become distinct `LlamaFtype` variants resolved by `StandardPolicy`, not collapsed at the type-name layer). |

## Totals

- **DELETE:** 9 regions, 480 LOC (pass-1 size predictors + zero-pad fallback loops in both text + mmproj writers; the iter-99 / Bug-B-sequel bug-class targets)
- **MODIFY:** 8 regions, 295 LOC (vision-pattern filter → consolidate to `is_vision_tensor_pattern`; promotion-rule logic → move to dispatcher-above-policy; offset-table write order → after streaming; file handle → needs `Seek`; type-name lookup → `GgmlType::from_str`)
- **KEEP:** 14 regions, 286 LOC (path/dir handling, metadata builders, KV-pair encoding, alignment padding, manifest assembly, mmproj path helper, patch-embd transpose)
- **Slice total:** ~1061 LOC accounted (slice is 978 LOC including blanks/comments between fns; structural region accounting includes some overlap-rounding).

## Notes — flagged items

### Vision-tensor F16-passthrough checks (THREE inline copies in the slice)

The ADR Decision §"Vision tensor pattern" says vision-pattern membership lives at `crate::quantize::layer_mix::is_vision_tensor_pattern(name)` (the canonical source). The writer slice today contains **three separate inline vision-pattern checks** that duplicate this logic:

1. **L322–L333** — text-tensor filter (exclude vision from text GGUF). Substrings: `vision_tower`, `embed_vision`, `audio_tower`, `model.visual.`.
2. **L721–L724** — `has_vision` detect (decide whether to write mmproj GGUF). Substrings: `vision_tower`, `embed_vision`.
3. **L905–L909** — mmproj-tensor filter (include only vision in mmproj GGUF). Substrings: `vision_tower`, `embed_vision`.

These three lists are **not identical** (the text-filter has `audio_tower` + `model.visual.` that the mmproj-filter lacks). Per ADR Decision §"Vision tensor pattern", all three move to `is_vision_tensor_pattern` at `src/quantize/layer_mix.rs:366` (or its post-P6 successor `src/quantize/vision.rs`) and the dispatcher above the new writer applies one canonical check. Audio-tensor membership needs its own canonical fn (analogous to vision); the audit doesn't see a `is_audio_tensor_pattern` today — flag for the synthesizer.

### Zero-pad fallback paths (the P2 bug-class targets)

Three zero-pad sites in the slice — all `if current < target { write zeros }`:

- **L639–L641** — text writer Pass-2 main tensor loop
- **L659–L661** — text writer Pass-2 synthetic-tensor loop
- **L677–L679** — text writer Pass-2 V=K-duplicate loop
- **L1132–L1134** — mmproj writer Pass-2 main tensor loop

All four are the iter-99 / Bug-B-sequel pattern: pass-1 mis-predicts a tensor size → pass-2 finds `current_pos < predicted_target_pos` → pads with zeros to land on the predicted offset → file inflates AND the next tensor's actual offset still mismatches the offset table. The seek-back writer eliminates all four by construction (no predicted offsets exist).

### Out-of-slice helpers called by the slice (mod.rs / main.rs agent territory)

Referenced from inside the slice but defined outside (not audited here; flagged for the synthesizer):

- `build_metadata` (L3356) — produces `Vec<(String, MetaValue)>` from `ModelMetadata`
- `write_metadata_kv` (L4007) — encodes one KV pair to the wire format
- `write_tensor_info` (L4174) — encodes one tensor-info entry
- `align_up` (L4192) — `(offset + align - 1) & !(align - 1)` helper
- `ggml_tensor_size` (L1889) — **the load-bearing size-predictor table** (the iter-99 / Bug-B-sequel home). Already has `panic!` on unknown types (2026-05-17 fix per `2b9b5a42`), but the entire function is **DELETE** under P2 — the seek-back writer never asks "how big will this be"; it writes the bytes and reports actual size.
- `ggml_type_name` (L1868) — debug-string helper; moves to `Display` on `GgmlType` enum.
- `quant_info_to_ggml_type` (L2059) — `TensorQuantInfo → u32`; DELETE under P3 (`TensorQuantInfo` collapses to `QuantizedTensor { ggml_type, data }`).
- `hf_name_to_gguf` (L2435) — HF→GGUF tensor name mapping; KEEP (utility used by both writers).
- `arch_gguf_name` (L4216) — arch-string resolver; KEEP.
- `generate_synthetic_tensors` (L1983) — Gemma4 rope_freqs etc.; KEEP.
- `repack_to_ggml_blocks` (L1303) — the per-tensor block-encoder called by Pass-2; MODIFY under P0/P1 — the `Quantizer` trait per Decision §2 replaces this dispatcher, and the new writer calls `Quantizer::quantize` per tensor instead.

### Block-size + bytes-per-block lookup

`ggml_tensor_size` (L1889, outside slice) is the `match ggml_type` table the ADR P2 names. Under P3 + P-1, this becomes `GgmlType::block_bytes() -> (block_elems: usize, block_size: usize)` (or two methods) on the new enum. The seek-back writer doesn't need to call it for prediction (it streams), but the new `Quantizer` impls still need block-size constants for their own per-row math — those constants move to the `GgmlType` enum impl block.

### Streaming property note

The current two-pass writer DOES already stream per-tensor (`repack_to_ggml_blocks` returns `Vec<u8>` that's dropped after the `write_all`, see L644–L650). The seek-back rewrite preserves this property — what changes is that the offset table is written AFTER the data stream (via seek-back) instead of BEFORE (with pre-predicted sizes). Memory bound from ADR P0 (`2 × model_safetensors_size + 512 MiB`) is unaffected.
