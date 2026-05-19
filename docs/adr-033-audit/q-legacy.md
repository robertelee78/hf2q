# Audit: src/quantize/q_legacy.rs (P-1, ADR-033)

Pinned external references: llama.cpp @ c779f6198 (`/opt/llama.cpp/ggml/src/ggml-quants.c`)

File size: 2130 LOC, 92 fn-like items (15 module-level fns incl. nearest_kvalue_iq4_nl_index; 36 impl methods/tests/inherent fns + 41 `#[test]` cases). Production code: L1–L1233. Test mod: L1235–L2130.

Legacy kernel mapping (verified line ranges in `/opt/llama.cpp/ggml/src/ggml-quants.c`):
- `quantize_row_q4_0_ref` → L71+ (no imatrix); `quantize_row_q4_0_impl` → L2008+ (imatrix-aware); `quantize_q4_0` wrapper → L2049+
- `quantize_row_q4_1_ref` → L108+; `quantize_row_q4_1_impl` → L2067+; `quantize_q4_1` → L2094+
- `quantize_row_q5_0_ref` → L145+; `quantize_row_q5_0_impl` → L2112+; `quantize_q5_0` → L2148+
- `quantize_row_q5_1_ref` → L189+; `quantize_row_q5_1_impl` → L2166+; `quantize_q5_1` → L2201+
- `quantize_row_q8_0_ref` → L234+; `quantize_q8_0` → L2218+
- `quantize_row_iq4_nl_ref` → L4928+; `quantize_row_iq4_nl_impl` → L4794+; `quantize_iq4_nl` → L4911+
- `dequantize_row_q4_0` → L397; `dequantize_row_q4_1` → L417; `dequantize_row_q5_0` → L438; `dequantize_row_q5_1` → L464; `dequantize_row_q8_0` → L491; `dequantize_row_iq4_nl` → L2649

| Symbol (line range) | LOC | Disposition | Rationale (one line) |
|---|---|---|---|
| `pub const QK8_0/QK4_0/QK4_1/QK5_0/QK5_1/QK4_NL` (L35–L71) | 6 | MODIFY | move → `src/quantize/ggml_quants/{q8_0,q4_0,q4_1,q5_0,q5_1,iq4_nl}.rs`; mirrors `ggml-quants.h` block constants |
| `pub const BLOCK_*_SIZE` × 6 (L42–L75) | 6 | MODIFY | move alongside corresponding QK; used by `backends/gguf.rs:1347-1349,1927-1935` for byte-size estimation — must remain importable |
| `pub const KVALUES_IQ4_NL` (L80–L84) | 5 | MODIFY | port → `src/quantize/ggml_quants/iq4_nl.rs`; byte-equal to `kvalues_iq4nl[16]` in ggml-quants.c |
| `pub enum QLegacyError` (L86–L99) | 14 | KEEP | external use: `quantize/k_quant_codec.rs:212` `#[from] QLegacyError`, `quantize/mod.rs:4518` test-helper return type — fold into the new `QuantizeError` taxonomy in P0/P1 |
| `pub struct BlockQ8_0` (L108–L113) | 6 | MODIFY | port → `src/quantize/ggml_quants/q8_0.rs`; mirrors `block_q8_0` struct in ggml-quants.h |
| `impl BlockQ8_0::from_bytes / to_bytes / d` (L115–L146) | 32 | MODIFY | port → q8_0.rs; byte-layout helpers (no llama.cpp analog — Rust-side serdes for `block_q8_0`) |
| `pub fn quantize_row_q8_0` (L163–L195) | 33 | MODIFY | port → `src/quantize/ggml_quants/q8_0.rs`; mirrors `quantize_row_q8_0_ref` @ ggml-quants.c:234 |
| `pub fn dequantize_row_q8_0` (L201–L219) | 19 | MODIFY | port → q8_0.rs; mirrors `dequantize_row_q8_0` @ ggml-quants.c:491 |
| `pub fn quantize_row_q8_0_to_bytes` (L226–L247) | 22 | MODIFY | port → q8_0.rs as helper feeding `Quantizer::quantize` return; called from `k_quant_codec.rs:313` |
| `pub fn dequantize_row_q8_0_bytes` (L253–L276) | 24 | KEEP | external callers: `quality/mod.rs:653`, `k_quant_codec.rs:868,966,1183`, `quantize/mod.rs:4524` — round-trip / quality verification helpers; rehome to q8_0.rs |
| `pub struct BlockQ4_0` (L287–L295) | 9 | MODIFY | port → `src/quantize/ggml_quants/q4_0.rs`; mirrors `block_q4_0` |
| `impl BlockQ4_0::from_bytes / to_bytes / d` (L297–L322) | 26 | MODIFY | port → q4_0.rs; byte-layout helpers |
| `pub fn quantize_row_q4_0` (L343–L386) | 44 | MODIFY | port → q4_0.rs; mirrors `quantize_row_q4_0_ref` @ ggml-quants.c:71 |
| `pub fn dequantize_row_q4_0` (L394–L413) | 20 | MODIFY | port → q4_0.rs; mirrors `dequantize_row_q4_0` @ ggml-quants.c:397 |
| `pub fn quantize_row_q4_0_to_bytes` (L416–L437) | 22 | MODIFY | port → q4_0.rs; called from `k_quant_codec.rs:309` AND `core/mlx_safetensors_loader.rs:301` |
| `pub fn dequantize_row_q4_0_bytes` (L440–L463) | 24 | KEEP | external: `quality/mod.rs:641`, `core/mlx_safetensors_loader.rs:303`, `k_quant_codec.rs:877,953`, `calibrate/qdq_gpu.rs` test, `quantize/mod.rs:4520` — rehome to q4_0.rs |
| `pub struct BlockQ4_1` (L475–L482) | 8 | MODIFY | port → `src/quantize/ggml_quants/q4_1.rs`; mirrors `block_q4_1` |
| `impl BlockQ4_1::from_bytes / to_bytes / d / m` (L484–L515) | 32 | MODIFY | port → q4_1.rs; byte-layout helpers |
| `pub fn quantize_row_q4_1` (L528–L580) | 53 | MODIFY | port → q4_1.rs; mirrors `quantize_row_q4_1_ref` @ ggml-quants.c:108 |
| `pub fn dequantize_row_q4_1` (L588–L608) | 21 | MODIFY | port → q4_1.rs; mirrors `dequantize_row_q4_1` @ ggml-quants.c:417 |
| `pub fn quantize_row_q4_1_to_bytes` (L611–L633) | 23 | MODIFY | port → q4_1.rs; called from `k_quant_codec.rs:310` |
| `pub fn dequantize_row_q4_1_bytes` (L636–L659) | 24 | KEEP | external: `quality/mod.rs:644`, `k_quant_codec.rs:957`, `quantize/mod.rs:4521` — rehome to q4_1.rs |
| `pub struct BlockQ5_0` (L671–L682) | 12 | MODIFY | port → `src/quantize/ggml_quants/q5_0.rs`; mirrors `block_q5_0` (qh array layout matches ggml) |
| `impl BlockQ5_0::from_bytes / to_bytes / d` (L684–L707) | 24 | MODIFY | port → q5_0.rs; byte-layout helpers |
| `pub fn quantize_row_q5_0` (L711–L758) | 48 | MODIFY | port → q5_0.rs; mirrors `quantize_row_q5_0_ref` @ ggml-quants.c:145 |
| `pub fn dequantize_row_q5_0` (L762–L785) | 24 | MODIFY | port → q5_0.rs; mirrors `dequantize_row_q5_0` @ ggml-quants.c:438 |
| `pub fn quantize_row_q5_0_to_bytes` (L788–L810) | 23 | MODIFY | port → q5_0.rs; called from `k_quant_codec.rs:311` |
| `pub fn dequantize_row_q5_0_bytes` (L813–L836) | 24 | KEEP | external: `quality/mod.rs:647`, `k_quant_codec.rs:871,960`, `quantize/mod.rs:4522` — rehome to q5_0.rs |
| `pub struct BlockQ5_1` (L844–L853) | 10 | MODIFY | port → `src/quantize/ggml_quants/q5_1.rs`; mirrors `block_q5_1` |
| `impl BlockQ5_1::from_bytes / to_bytes / d / m` (L855–L889) | 35 | MODIFY | port → q5_1.rs; byte-layout helpers |
| `pub fn quantize_row_q5_1` (L893–L949) | 57 | MODIFY | port → q5_1.rs; mirrors `quantize_row_q5_1_ref` @ ggml-quants.c:189 |
| `pub fn dequantize_row_q5_1` (L953–L976) | 24 | MODIFY | port → q5_1.rs; mirrors `dequantize_row_q5_1` @ ggml-quants.c:464 |
| `pub fn quantize_row_q5_1_to_bytes` (L979–L1002) | 24 | MODIFY | port → q5_1.rs; called from `k_quant_codec.rs:312` |
| `pub fn dequantize_row_q5_1_bytes` (L1005–L1028) | 24 | KEEP | external: `quality/mod.rs:650`, `k_quant_codec.rs:874,963`, `quantize/mod.rs:4523` — rehome to q5_1.rs |
| `pub struct BlockIQ4_NL` (L1037–L1042) | 6 | MODIFY | port → `src/quantize/ggml_quants/iq4_nl.rs`; mirrors `block_iq4_nl` |
| `impl BlockIQ4_NL::from_bytes / to_bytes / d` (L1044–L1065) | 22 | MODIFY | port → iq4_nl.rs; byte-layout helpers |
| `pub fn quantize_row_iq4_nl` (L1080–L1140) | 61 | MODIFY | port → iq4_nl.rs; mirrors `quantize_row_iq4_nl_ref` @ ggml-quants.c:4928 (uses `quantize_row_iq4_nl_impl` @ :4794) |
| `fn nearest_kvalue_iq4_nl_index` (L1143–L1154) | 12 | MODIFY | port → iq4_nl.rs as private helper; codebook lookup over `KVALUES_IQ4_NL` |
| `pub fn dequantize_row_iq4_nl` (L1158–L1180) | 23 | MODIFY | port → iq4_nl.rs; mirrors `dequantize_row_iq4_nl` @ ggml-quants.c:2649 |
| `pub fn quantize_row_iq4_nl_to_bytes` (L1183–L1204) | 22 | MODIFY | port → iq4_nl.rs; no current external caller but symmetric API with sister types — keep as `Quantizer::quantize` helper |
| `pub fn dequantize_row_iq4_nl_bytes` (L1207–L1233) | 27 | MODIFY | port → iq4_nl.rs; no external caller today; keep for ADR-022 round-trip parity tests |
| `mod tests { … }` 41 `#[test]` fns (L1235–L2130) | 896 | MODIFY | split per-type; move to `src/quantize/ggml_quants/{q4_0,q4_1,q5_0,q5_1,q8_0,iq4_nl}.rs#[cfg(test)]` alongside the kernels they exercise — incl. `adr022_iq4_nl_*` codebook-byte-equal + idempotency suite |

**Totals:** DELETE: 0 fns, 0 LOC | MODIFY: 81 fns/items, ~1801 LOC (kernels + tests; 6 module-level pubs ported to the 6 per-type files in `src/quantize/ggml_quants/`) | KEEP: 6 pub fns + 1 enum, ~157 LOC (5× `dequantize_row_q*_bytes` + `dequantize_row_q8_0_bytes` + `QLegacyError`; rehomed but NOT deleted — external callers in `quality/mod.rs`, `core/mlx_safetensors_loader.rs`, `calibrate/qdq_gpu.rs`, plus intra-quantize callers in `k_quant_codec.rs` + `quantize/mod.rs` test helpers).

**Notes:** follow-ups for synthesizer.

1. **No DELETE-only rows.** Every kernel maps to a P0 port target (`ggml-quants.c::quantize_row_<type>_ref`); every utility (`*_to_bytes`, `*_bytes` round-trip helpers) has at least one external caller. The file's net P6 outcome is `mv` + `split` + `cfg`-rehome, not `rm -rf`.
2. **`QLegacyError` is consumed by `k_quant_codec.rs::KQuantCodecError::QLegacy` via `#[from]`.** P0/P1 must fold its three variants (`NotBlockAligned { actual, qk }`, `OutputTooSmall`, `BufferTooSmall`) into ADR-033's typed `QuantizeError` taxonomy; the `#[from]` edge in `k_quant_codec.rs:212` will follow `k_quant_codec.rs`'s own DELETE/MODIFY classification (out-of-scope here).
3. **`*_bytes` round-trip helpers are NOT the ADR-033 `Quantizer::quantize` surface.** They predate the trait. v1 keeps them as utility (quality-metric round-trip in `quality/mod.rs:641-653`, MLX-safetensors-loader Q4_0 drift check at `core/mlx_safetensors_loader.rs:275`, calibrate QDQ-GPU parity tests at `calibrate/qdq_gpu.rs:155`). Synthesizer: place them in the same `ggml_quants/<type>.rs` file as the kernel, marked `pub(crate)` if the trait makes them redundant downstream.
4. **`quantize_row_iq4_nl_to_bytes` / `dequantize_row_iq4_nl_bytes` have no external callers today.** They are MODIFY (not DELETE) because (a) ADR-022 idempotency test at L2117 exercises them, and (b) symmetric API with the 5 Q-variants is load-bearing for ADR-033's `Quantizer` trait impls.
5. **The 6 `BLOCK_*_SIZE` constants are externally imported by `backends/gguf.rs:1347-1349,1927-1935`.** P0 ports leave these as re-exports from `ggml_quants/<type>.rs`; the GGUF writer's byte-size estimator is unchanged by ADR-033 P0 (P2 rewrites the writer, but byte-size estimation is orthogonal).
6. **Test mod is 896 LOC = 42% of the file.** Split by type. The `adr022_iq4_nl_codebook_byte_equal_to_llama_cpp` test at L2009 is a load-bearing parity gate — moves verbatim to `ggml_quants/iq4_nl.rs#[cfg(test)]`.
7. **No `*_impl` (imatrix-aware) variants exist in q_legacy.rs.** llama.cpp's `quantize_row_q4_0_impl` (ggml-quants.c:2008) accepts a `quant_weights` argument and dispatches on `quant_weights == NULL`. ADR-033 P0 will need to ADD imatrix-aware variants to the new `ggml_quants/<type>.rs` files (`Quantizer::quantize` takes `imatrix: Option<&[f32]>`); these are NEW code, not ports of q_legacy.rs content.
