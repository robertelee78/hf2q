# Audit: src/quantize/k_quant.rs (P-1, ADR-033)

Pinned external references: llama.cpp @ `c779f6198` (`/opt/llama.cpp/ggml/src/ggml-quants.c`)

File: `/opt/hf2q/src/quantize/k_quant.rs` — 5541 LOC. Production code = L1..L3066; `#[cfg(test)] mod tests` = L3068..L5541. Top-level `pub const` items (`QK_K`, `BLOCK_Q{2,3,4,5,6}_K_SIZE`, `KQuantError`) are not `fn`s and therefore not rows in this table; see Notes.

| Symbol (line range) | LOC | Disposition | Rationale (one line) |
|---|---|---|---|
| `impl BlockQ4K::from_bytes` (L151-170) | 20 | MODIFY | port → `src/quantize/ggml_quants/q4_k.rs`; little-endian decoder for `block_q4_K` in `ggml-common.h` (referenced by `ggml-quants.c::dequantize_row_q4_K`) |
| `impl BlockQ4K::to_bytes` (L172-180) | 9 | MODIFY | port → `src/quantize/ggml_quants/q4_k.rs`; little-endian encoder for `block_q4_K` (writer side of P0/P2) |
| `impl BlockQ4K::d` (L182-185) | 4 | MODIFY | port → `src/quantize/ggml_quants/q4_k.rs`; F16 super-block scale accessor used by `dequantize_row_q4_K` |
| `impl BlockQ4K::dmin` (L187-190) | 4 | MODIFY | port → `src/quantize/ggml_quants/q4_k.rs`; F16 super-block min accessor used by `dequantize_row_q4_K` |
| `pub fn get_scale_min_k4` (L199-232) | 34 | MODIFY | port → `src/quantize/ggml_quants/q4_k.rs`; mirrors `ggml-quants.c::get_scale_min_k4` (L818). Shared by Q4_K + Q5_K dequant per llama.cpp |
| `pub fn dequantize_row_q4_k` (L234-284) | 51 | MODIFY | port → `src/quantize/ggml_quants/q4_k.rs`; mirrors `ggml-quants.c::dequantize_row_q4_K` (L1467) |
| `pub fn dequantize_row_q4_k_bytes` (L286-316) | 31 | MODIFY | port → `src/quantize/ggml_quants/q4_k.rs`; bytes-in wrapper around `dequantize_row_q4_K`; external callers in `src/quality/mod.rs:612` + `src/backends/gguf.rs:1275` rewire to new path in P3 |
| `impl BlockQ5K::from_bytes` (L336-357) | 22 | MODIFY | port → `src/quantize/ggml_quants/q5_k.rs`; little-endian decoder for `block_q5_K` |
| `impl BlockQ5K::to_bytes` (L359-368) | 10 | MODIFY | port → `src/quantize/ggml_quants/q5_k.rs`; little-endian encoder for `block_q5_K` |
| `impl BlockQ5K::d` (L370-373) | 4 | MODIFY | port → `src/quantize/ggml_quants/q5_k.rs`; F16 super-block scale accessor |
| `impl BlockQ5K::dmin` (L375-378) | 4 | MODIFY | port → `src/quantize/ggml_quants/q5_k.rs`; F16 super-block min accessor |
| `pub fn dequantize_row_q5_k` (L397-449) | 53 | MODIFY | port → `src/quantize/ggml_quants/q5_k.rs`; mirrors `ggml-quants.c::dequantize_row_q5_K` (L1669) |
| `pub fn dequantize_row_q5_k_bytes` (L451-481) | 31 | MODIFY | port → `src/quantize/ggml_quants/q5_k.rs`; bytes-in wrapper; external callers in `src/quality/mod.rs:612` + `src/backends/gguf.rs:1458` rewire in P3 |
| `impl BlockQ6K::from_bytes` (L498-520) | 23 | MODIFY | port → `src/quantize/ggml_quants/q6_k.rs`; little-endian decoder for `block_q6_K` |
| `impl BlockQ6K::to_bytes` (L522-533) | 12 | MODIFY | port → `src/quantize/ggml_quants/q6_k.rs`; little-endian encoder for `block_q6_K` |
| `impl BlockQ6K::d` (L535-538) | 4 | MODIFY | port → `src/quantize/ggml_quants/q6_k.rs`; F16 super-block scale accessor |
| `pub fn dequantize_row_q6_k` (L558-610) | 53 | MODIFY | port → `src/quantize/ggml_quants/q6_k.rs`; mirrors `ggml-quants.c::dequantize_row_q6_K` (L1877) |
| `pub fn dequantize_row_q6_k_bytes` (L612-656) | 45 | MODIFY | port → `src/quantize/ggml_quants/q6_k.rs`; bytes-in wrapper; external callers in `src/quality/mod.rs:612` + `src/backends/gguf.rs:2207` rewire in P3 |
| `impl BlockQ3K::from_bytes` (L672-692) | 21 | MODIFY | port → `src/quantize/ggml_quants/q3_k.rs`; little-endian decoder for `block_q3_K`. NOTE: Q3_K is not in P0 v1's 9-file target set (see Notes) |
| `impl BlockQ3K::to_bytes` (L694-703) | 10 | MODIFY | port → `src/quantize/ggml_quants/q3_k.rs`; out-of-v1 (see Notes) |
| `impl BlockQ3K::d` (L705-708) | 4 | MODIFY | port → `src/quantize/ggml_quants/q3_k.rs`; out-of-v1 (see Notes) |
| `pub fn unpack_q3_k_scales` (L734-780) | 47 | MODIFY | port → `src/quantize/ggml_quants/q3_k.rs`; mirrors scales-unpack idiom inlined in `ggml-quants.c::dequantize_row_q3_K` (L1261-L1266, kmask1/kmask2 dance). hf2q factored it out; canonical lives inline. Out-of-v1 |
| `pub fn dequantize_row_q3_k` (L782-839) | 58 | MODIFY | port → `src/quantize/ggml_quants/q3_k.rs`; mirrors `ggml-quants.c::dequantize_row_q3_K` (L1243). External callers in `src/quality/mod.rs:612` + `src/backends/gguf.rs:2566` rewire in P3. Out-of-v1 |
| `pub fn dequantize_row_q3_k_bytes` (L841-885) | 45 | MODIFY | port → `src/quantize/ggml_quants/q3_k.rs`; bytes-in wrapper. Out-of-v1 |
| `impl BlockQ2K::from_bytes` (L902-920) | 19 | MODIFY | port → `src/quantize/ggml_quants/q2_k.rs`; little-endian decoder for `block_q2_K`. Out-of-v1 |
| `impl BlockQ2K::to_bytes` (L922-931) | 10 | MODIFY | port → `src/quantize/ggml_quants/q2_k.rs`; little-endian encoder. Out-of-v1 |
| `impl BlockQ2K::d` (L933-936) | 4 | MODIFY | port → `src/quantize/ggml_quants/q2_k.rs`; F16 super-block scale accessor. Out-of-v1 |
| `impl BlockQ2K::dmin` (L938-941) | 4 | MODIFY | port → `src/quantize/ggml_quants/q2_k.rs`; F16 super-block min accessor. Out-of-v1 |
| `pub fn dequantize_row_q2_k` (L958-1008) | 51 | MODIFY | port → `src/quantize/ggml_quants/q2_k.rs`; mirrors `ggml-quants.c::dequantize_row_q2_K` (L899). External callers in `src/quality/mod.rs:612` + `src/backends/gguf.rs:2819, 3085` rewire in P3. Out-of-v1 |
| `pub fn dequantize_row_q2_k_bytes` (L1010-1046) | 37 | MODIFY | port → `src/quantize/ggml_quants/q2_k.rs`; bytes-in wrapper. Out-of-v1 |
| `pub fn nearest_int` (L1048-1081) | 34 | MODIFY | port → `src/quantize/ggml_quants/common.rs`; mirrors `ggml-quants.c::nearest_int` (L559); shared inline helper used by every quantize fn |
| `pub fn make_qkx2_quants` (L1083-1200) | 118 | MODIFY | port → `src/quantize/ggml_quants/common.rs`; mirrors `ggml-quants.c::make_qkx2_quants` (L737); shared codebook search routine used by Q4_K, Q5_K ref quantize |
| `pub fn quantize_row_q4_k` (L1202-1337) | 136 | MODIFY | port → `src/quantize/ggml_quants/q4_k.rs`; mirrors `ggml-quants.c::quantize_row_q4_K_ref` (L1395) |
| `pub fn quantize_row_q4_k_imatrix` (L1339-1471) | 133 | MODIFY | port → `src/quantize/ggml_quants/q4_k.rs`; mirrors `ggml-quants.c::quantize_row_q4_K_impl` (L1491) |
| `pub fn quantize_row_q5_k_imatrix` (L1473-1615) | 143 | MODIFY | port → `src/quantize/ggml_quants/q5_k.rs`; mirrors `ggml-quants.c::quantize_row_q5_K_impl` (L1696) |
| `pub fn quantize_row_q6_k_imatrix` (L1617-1741) | 125 | MODIFY | port → `src/quantize/ggml_quants/q6_k.rs`; mirrors `ggml-quants.c::quantize_row_q6_K_impl` (L1908) |
| `pub fn quantize_row_q5_k` (L1743-1894) | 152 | MODIFY | port → `src/quantize/ggml_quants/q5_k.rs`; mirrors `ggml-quants.c::quantize_row_q5_K_ref` (L1582) |
| `pub fn make_qkx3_quants` (L1896-2010) | 115 | MODIFY | port → `src/quantize/ggml_quants/common.rs`; mirrors `ggml-quants.c::make_qkx3_quants` (L931); shared by `_impl` (imatrix) routines for Q2_K/Q3_K/Q4_K/Q5_K |
| `pub fn make_qp_quants` (L2012-2130) | 119 | MODIFY | port → `src/quantize/ggml_quants/common.rs`; mirrors `ggml-quants.c::make_qp_quants` (L1014); shared positive-quant scales search used by `_impl` paths |
| `pub fn make_qx_quants` (L2132-2252) | 121 | MODIFY | port → `src/quantize/ggml_quants/common.rs`; mirrors `ggml-quants.c::make_qx_quants` (L566); shared symmetric-quant routine used by Q3_K/Q6_K ref |
| `pub fn quantize_row_q6_k` (L2254-2362) | 109 | MODIFY | port → `src/quantize/ggml_quants/q6_k.rs`; mirrors `ggml-quants.c::quantize_row_q6_K_ref` (L1807) |
| `pub fn quantize_row_q4_k_to_bytes` (L2364-2386) | 23 | MODIFY | port → `src/quantize/ggml_quants/q4_k.rs`; bytes-out wrapper over `quantize_row_q4_K_ref` — composes with `Quantizer::quantize` trait signature in ADR-033 §"Quantizer trait" |
| `pub fn quantize_row_q4_k_imatrix_to_bytes` (L2388-2412) | 25 | MODIFY | port → `src/quantize/ggml_quants/q4_k.rs`; bytes-out wrapper over `quantize_row_q4_K_impl`; maps onto `quantize_q4_K` top-level dispatcher (`ggml-quants.c::L1564`) for the imatrix branch |
| `pub fn quantize_row_q5_k_to_bytes` (L2414-2436) | 23 | MODIFY | port → `src/quantize/ggml_quants/q5_k.rs`; bytes-out wrapper over ref path |
| `pub fn quantize_row_q5_k_imatrix_to_bytes` (L2438-2463) | 26 | MODIFY | port → `src/quantize/ggml_quants/q5_k.rs`; bytes-out wrapper; maps onto `quantize_q5_K` (`ggml-quants.c::L1789`) |
| `pub fn quantize_row_q6_k_to_bytes` (L2465-2486) | 22 | MODIFY | port → `src/quantize/ggml_quants/q6_k.rs`; bytes-out wrapper over ref path |
| `pub fn quantize_row_q6_k_imatrix_to_bytes` (L2488-2555) | 68 | MODIFY | port → `src/quantize/ggml_quants/q6_k.rs`; bytes-out wrapper; maps onto `quantize_q6_K` (`ggml-quants.c::L1992`) |
| `pub fn quantize_row_q3_k` (L2557-2562) | 6 | MODIFY | port → `src/quantize/ggml_quants/q3_k.rs`; thin ref-path entry, mirrors `quantize_row_q3_K_ref` (L1167). Out-of-v1 |
| `pub fn quantize_row_q3_k_imatrix` (L2564-2576) | 13 | MODIFY | port → `src/quantize/ggml_quants/q3_k.rs`; thin imatrix entry, mirrors `quantize_row_q3_K_impl` (L1293). Out-of-v1 |
| `fn quantize_row_q3_k_inner` (L2578-2727) | 150 | MODIFY | port → `src/quantize/ggml_quants/q3_k.rs`; shared body for ref + imatrix dispatch — mirrors `ggml-quants.c::quantize_row_q3_K_impl` (the ref path is the same impl with `quant_weights=null`). Out-of-v1 |
| `pub fn quantize_row_q3_k_to_bytes` (L2729-2750) | 22 | MODIFY | port → `src/quantize/ggml_quants/q3_k.rs`; bytes-out wrapper. Out-of-v1 |
| `pub fn quantize_row_q3_k_imatrix_to_bytes` (L2752-2797) | 46 | MODIFY | port → `src/quantize/ggml_quants/q3_k.rs`; bytes-out wrapper; maps onto `quantize_q3_K` (L1377). Out-of-v1 |
| `pub fn quantize_row_q2_k` (L2799-2811) | 13 | MODIFY | port → `src/quantize/ggml_quants/q2_k.rs`; thin ref-path entry, mirrors `quantize_row_q2_K_ref` (L829). Out-of-v1 |
| `pub fn quantize_row_q2_k_imatrix` (L2813-2825) | 13 | MODIFY | port → `src/quantize/ggml_quants/q2_k.rs`; thin imatrix entry, mirrors `quantize_row_q2_K_impl` (L1087). Out-of-v1 |
| `fn quantize_row_q2_k_inner` (L2827-3003) | 177 | MODIFY | port → `src/quantize/ggml_quants/q2_k.rs`; shared body for ref + imatrix — mirrors `ggml-quants.c::quantize_row_q2_K_impl`. Out-of-v1 |
| `fn pack_q2_k_qs` (L3005-3017) | 13 | MODIFY | port → `src/quantize/ggml_quants/q2_k.rs`; private packing helper for the 2-bit qs payload. Out-of-v1 |
| `pub fn quantize_row_q2_k_to_bytes` (L3019-3040) | 22 | MODIFY | port → `src/quantize/ggml_quants/q2_k.rs`; bytes-out wrapper. Out-of-v1 |
| `pub fn quantize_row_q2_k_imatrix_to_bytes` (L3042-3066) | 25 | MODIFY | port → `src/quantize/ggml_quants/q2_k.rs`; bytes-out wrapper; maps onto `quantize_q2_K` (L1149). Out-of-v1 |
| `#[cfg(test)] mod tests` — 96 inline test fns (L3068-L5541) | 2474 | DELETE | superseded by P0 acceptance criteria's byte-equivalence fixture tests against llama.cpp `ggml_quantize_chunk` at the pinned SHA (per ADR §"P0 Acceptance criteria" bullet 1); hf2q-internal RMSE/round-trip tests don't add coverage the byte-cmp gate doesn't already supply, and live alongside soon-to-be-deleted code paths (`k_quant_codec_quantizer`, `dwq_k_quantizer`) per per-test grep above |

**Totals:** DELETE: 96 fns, 2474 LOC | MODIFY: 56 fns, 3067 LOC | KEEP: 0 fns, 0 LOC | **File total:** 152 fns, 5541 LOC

**Notes:**

- **Top-level `pub const` items** are not `fn`s so not table rows: `QK_K` (L65), `BLOCK_Q4_K_SIZE` (L69), `BLOCK_Q5_K_SIZE` (L74), `BLOCK_Q6_K_SIZE` (L79), `BLOCK_Q3_K_SIZE` (L87), `BLOCK_Q2_K_SIZE` (L96), plus `pub enum KQuantError` (L100-113). These constants are referenced externally from `src/backends/gguf.rs:1336,1337,1339,1947,1951,1955,1959,4900,4918,4952` (block-size byte-budget calculations). They MODIFY-port alongside their type's kernel file in P0 (e.g., `BLOCK_Q4_K_SIZE` → `src/quantize/ggml_quants/q4_k.rs`) and external call sites rewire in P3.

- **Q3_K and Q2_K are out-of-v1-scope.** ADR-033 §"P0 What" explicitly lists the v1 set as `{q4_0, q4_1, q5_0, q5_1, q4_k, q5_k, q6_k, q8_0, iq4_nl}` (9 files). Q3_K and Q2_K are NOT in v1. However, their kernels in `k_quant.rs` (Q3_K: ~330 LOC kernels + ~110 LOC dequant; Q2_K: ~340 LOC kernels + ~90 LOC dequant) are referenced externally from `src/quality/mod.rs:612` (codec-direct dequant arm) and `src/backends/gguf.rs:2566,2819,3085` (size estimator + writer). **Flag for synthesizer:** either (a) ADR-033 v1 scope expands to include `q3_k.rs` + `q2_k.rs` (10 + 11 = 21 .rs files instead of 9) to preserve the existing external dequant path, or (b) P3 also rewires those external call sites to a typed `QuantizeError::NoQuantizerForType` and accepts that Q3_K/Q2_K GGUFs round-tripped via hf2q's quality smoke break until v2. Recommend (a) — these are well-trodden llama.cpp ports and adding two more files to P0 is cheap; (b) breaks the codec-direct dequant invariant. Marked MODIFY here assuming (a); flip to DELETE if synthesizer decides (b).

- **No `KEEP` rows.** Every external caller of this file is either (i) a fn that itself MODIFY-ports to `src/quantize/ggml_quants/<type>.rs` (the `dequantize_row_q*_k_bytes` family — caller-side rewrite is P3 IR-collapse work, not a "keep the symbol in `k_quant.rs`" decision), or (ii) the `BLOCK_*_K_SIZE` constants (also MODIFY-port). Nothing in `k_quant.rs` is structural utility that doesn't fit the `{Quantizer, QuantPolicy, QuantizedTensor, GGUF writer}` shape.

- **Test-module classification rationale.** All 96 test fns live inside `#[cfg(test)] mod tests` (L3067-3068 onwards, terminating at EOF L5541). They are uniformly DELETE because (a) ADR §"P0 Acceptance criteria" bullet 1 commits to per-`GgmlType` byte-cmp against fixed C-harness output at `tests/fixtures/ggml_quants/<type>_<n>.bin` — this is structurally stronger than the existing inline RMSE/round-trip tests, and (b) several test bodies depend on `k_quant_codec`, `k_quant_codec_quantizer`, `dwq_k_quantizer` (confirmed via grep at L3085, L4481-4482, L4597-4598, L4723-4724, L4843-4844, L4952, L5086, L5201-5202, L5327-5328, L5439-5440, L5934), all of which are on ADR-033 §P6's delete-list. Net: deleting the test module is no test-coverage loss once P0's fixture tests land; keeping it through P0/P1 forces deletion of all interior `use crate::quantize::{k_quant_codec, k_quant_codec_quantizer, dwq_k_quantizer}::*` imports as those files get deleted in P6 — net-net easier to wholesale-delete with the kernel ports moved.

- **`make_qkx2_quants` NEON-order caveat.** The module-level doc at L9-L18 explicitly says the byte-identical gate targets the **NEON code path** on `aarch64-apple-darwin`, not the scalar reference, because horizontal-sum associativity in `make_qkx2_quants` is sensitive to reduction order. P0's byte-cmp gate against `ggml_quantize_chunk` will hit this — the C harness must build with the same arch flags hf2q's ref build uses. Flag for synthesizer: the P0 acceptance-test C harness needs to be `aarch64-apple-darwin`-built with NEON path enabled, OR the new pure-Rust port targets the **scalar** reference (in which case existing NEON-ordered RMSE tests in `mod tests` would have failed already — they don't, suggesting we already match scalar). Worth a 5-minute confirmation before P0 starts.

- **Q4_K imatrix bytes-out fixture-test gap.** `quantize_row_q4_K_impl` in llama.cpp is a `static` function (file-scope); the public surface is `quantize_q4_K(src, dst, nrow, n_per_row, quant_weights)` at L1564. hf2q's `quantize_row_q4_k_imatrix_to_bytes` (L2388) emulates the dispatcher path. The P0 byte-cmp C harness must call `quantize_q4_K` with a non-null `quant_weights`, not the `_ref` or `_impl` directly. Same pattern for q5_K, q6_K, q3_K, q2_K imatrix variants. Generic note — the synthesizer can cross-reference against the siblings' audits if they noticed the same.
