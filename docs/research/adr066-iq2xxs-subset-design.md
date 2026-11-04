# ADR-066 IQ2_XXS subset (N-IQ2XXS) and ADR-063 Q4-DS4-RACE (#667): serving ds4's DeepSeek-V4 file and racing it

Status: design for approval, revision 2 (2026-11-03). Revision 2 applies the
independent review (ACCEPT-WITH-CONDITIONS, findings 1–14), the operator
decisions D1 and D2 (§9) and the Q4-DS4 Fit result (§7.1). No code, build or
GPU run backs this document. The header comparison and sample tensor parity
in §2 were measured on the CPU from two small reads (headers and a few MB of
tensor bytes). Every other number is quoted from source with its
`file:line`, from a cited receipt, or marked `[INFERENCE]`.

Stories:
- **N-IQ2XXS** (mlx-native) is the IQ2_XXS subset of ADR-066 N1 (#580),
  N3 (#582) and N4 (#583). It was pulled ahead of ADR-066 B2 by the operator
  decision of 2026-11-03. The companion ADR-066 amendment adds it as its own
  story and narrows N1/N3/N4 to build on it.
- **Q4-DS4-RACE** (#667, hf2q) consumes N-IQ2XXS. It adds the hf2q loader for
  ds4's file, the quality checks and the harness changes, and runs the race.
  Its predecessors are Q4-DS4 (#661, Fit), N-IQ2XXS and Q5 (#371); ADR-063
  records this (commit `64c6f902`). ADR-066 N1/N3/N4 are not predecessors.

Sources:
- ADR-063: Q4, Q4-DS4, Q4-DS4-RACE, the native-engine section (ds4 arm) and
  "Speed program, Q1 successors and release mapping".
- ADR-066: Context items 2 and 3, "Numerical contract", "Repository
  ownership", N1, N3, N4 and N5.
- `docs/research/adr063-q3-campaign.md` and
  `docs/research/adr063-s3-ds4-joint-decode.md`.
- mlx-native 0.15.16, the current hf2q pin (`Cargo.toml:176`), read from
  `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/mlx-native-0.15.16`
  (written `R/` below). `/opt/mlx-native` is 0.15.15 and not the pin.
- hf2q at `d817dffa` (this worktree's base).
- The peer reference checkout
  `/opt/hf2q-worktrees/adr063-peer-reference-e86f5fd1`, read-only (written
  `P/`). Its `ggml/src/ggml-common.h` has sha256 `0061131b…595c`.
- ds4 c97964b at `~/.local/share/hf2q/peers/ds4-c97964b`, read-only.

Evidence directory: `/Users/robert/.local/state/hf2q/v0130/q4-ds4-race-design/`
(`$D` below).
- `gguf_head.py`: a stdlib GGUF header reader that reads no tensor data.
- `ds4-head.json` and `hf2q-head.json`: full headers, including per-tensor
  offset and byte extent.
- `header-compare.json`: the comparison summary.
- `sample_parity.py` and `sample-parity.json`: the CPU sample parity in §2.4.

## 1. Decision summary

1. **One mlx-native story, no second kernel.** mlx-native adds
   `GgmlType::IQ2_XXS` (type 16, 256 values, 66 bytes). It adds a CPU dequant
   that is bit-exact against the peer's `dequantize_row_iq2_xxs`, and the
   `iq2xxs_grid`, `ksigns_iq2xs` and `kmask_iq2xs` tables, generated from one
   Rust source. It adds IQ2_XXS `mv_id`, simdgroup `mm_id` and tensor-API
   `mm_id` kernels. Parity runs at DeepSeek-V4-Flash expert shapes: 256
   experts, top-6, gate/up K=4096→N=2048, and down K=2048→N=4096 for the
   reused Q2_K kernels. ADR-066 N1/N3/N4 later extend the same type, tables
   and kernels to V4.1 shapes and never add a second IQ2_XXS path (§3.9).
2. **Q2_K, Q8_0 and F16 need no new kernel.** ds4's down experts are Q2_K,
   its shared experts, attention and head are Q8_0, and its small projections
   are F16. Each already has an mlx-native kernel. Q2_K gains parity tests at
   the down shape and a row-bitwise test (§3.5). hf2q must route one-row F16
   products to the existing F32-input matvec instead of the tiled kernel that
   rounds the input to half (L9, §4).
3. **The hf2q loader needs nine small changes (L1–L9, §4).** The tensor names
   and shapes equal hf2q's own artifact exactly (1,328 of 1,328). The changes
   are:
   - a pool identity for `general.file_type` 19;
   - the `joyai-llm` pre-tokenizer as an alias of `deepseek-v3` (same three
     regexes);
   - defaults for two absent config keys;
   - acceptance of the 44-entry `compress_ratios`;
   - an F16 token-embedding gather;
   - a per-row route for a float `output_hc_fn`;
   - IQ2_XXS in the joint-decode proven set, after the row-bitwise proof;
   - case files and identity for the file;
   - F16 one-row products through `dispatch_dense_matvec_f16w_f32io`.

   No hf2q kernel is added.
4. **Quality checks.** Five tensor-load parity checks (T1–T5, §6.1),
   real-weight kernel parity (§6.2), the S3 bitwise joint == serial check on
   this file (§6.3), 0731 non-regression on hf2q's artifact (§6.4) and the
   paired task-quality gates (§6.5).
5. **Race (§7), under operator decisions D1 and D2 (§9).** Q4-DS4 reported
   Fit. After a fresh Q5 calibration of hf2q on this file, three campaigns
   run with the same candidate binary:
   - a same-artifact candidate-vs-v0.1.28 campaign on hf2q's own artifact,
     which supplies the vs-stock contrast, fairness band and stock quality
     gate;
   - the ds4-file campaign of record, experiment against the peer with no
     stock arm, bound to that stock record;
   - a same-file native comparison against ds4, which never enters the
     verdict.

   Per-stage attribution uses the Q4-TRACE hooks.
6. **Harness changes (§8) are part of Q4-DS4-RACE:** a no-stock-arm campaign
   mode with a bound stock record, the matching `required_effects`,
   `required_references`, `family_verdict` and quality-gate binding changes,
   and a declared same-artifact mode for the ds4 adapter.

## 2. The file and how it differs from hf2q's artifact

### 2.1 Identity

| | ds4 file | hf2q artifact |
|---|---|---|
| Path | `~/.local/share/hf2q/peers/ds4-c97964b/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` | `~/.local/share/hf2q/models/deepseek4/DeepSeek-V4-Flash-0731-hf2q-deepseek4-agentic-q2.gguf` |
| Bytes | 86,720,111,488 | 107,431,343,264 |
| sha256 | `ca22ae2f…61c0` (ADR-063 Q4-DS4, re-hashed 2026-11-02; not re-hashed here) | from its `.receipt.json` |
| GGUF version / tensors / KV pairs | 3 / 1,328 / 62 | 3 / 1,328 / 62 |
| Data offset (alignment 32) | 5,333,824 | 5,336,608 |
| Tensor bytes | 86.715 GB | 107.426 GB |

Every tensor's byte extent, measured as the distance to the next offset,
equals its block-geometry size with less than 32 bytes of padding in both
files. The tensor order differs, which nothing in hf2q depends on (§4).

### 2.2 Metadata differences (`$D/header-compare.json`)

| Key | ds4 | hf2q | hf2q today | Change |
|---|---|---|---|---|
| `general.file_type` | 19 (MOSTLY_IQ2_XXS) | 10 (MOSTLY_Q2_K) | startup abort: `QuantType::from_gguf_file_type` → None (`src/serve/quant_select.rs:112-124`, called at `src/serve/mod.rs:5192-5198`) | L1 |
| `tokenizer.ggml.pre` | `joyai-llm` | `deepseek-v3` | refused (`src/inference/models/deepseek4/tokenizer.rs:37-42`) | L2 |
| `deepseek4.embedding_length_out` | absent | 16384 | required (`deepseek4/config.rs:163`, checked `:247-249`) | L3 |
| `deepseek4.swiglu_clamp_shexp` | absent | 43 × 10.0 | required (`config.rs:204`, `:287-293`) | L3 |
| `deepseek4.attention.compress_ratios` | 44 entries (43 layers + a trailing 0) | 43 entries | `len == block_count` required (`config.rs:271-278`) | L4 |
| `deepseek4.nextn_predict_layers` | 1 | absent | not read under `deepseek4/`; no MTP tensors exist in the file | none |
| `deepseek4.vocab_size` | 129280 | absent | accepted when present (`config.rs:53-68`) | none |
| `general.name` | `DeepSeek V4 Flash` | `DeepSeek V4 Flash 0731` | becomes the served model id (`src/serve/api/engine_deepseek4.rs:776-782`) | case file `MODEL_ID` |
| `hf2q.*` (producer, profile, source sha) | absent | present | `Provenance::External`; allowed for a direct `--model` path (`src/core/provenance/mod.rs:142-150`) | none |
| `tokenizer.chat_template` | 4,988 chars, sha256 `872492…af27` | sha256 `aec3d906…cd6d` (7,646 bytes, `chat_templates.rs:58`) | never executed for DeepSeek; the native encoder serves the compiled-in 0731 template (`engine_deepseek4.rs:605-607`) | none |
| `general.size_label`, `quantize.imatrix.*` | present | absent | not read | none |

The tokenizer is identical. `tokenizer.ggml.tokens` (sha `81d7baa7…`),
`token_type` (`9a44a0c2…`), `merges` (`f8cf9567…`), BOS/EOS/padding ids and
the add-BOS/EOS flags are equal in both headers. Every model hyperparameter is
equal: block count 43, 256 experts, top-6, intermediate 2048, 3 hash layers,
the HC constants, the indexer constants, YaRN and `swiglu_clamp_exp`.

### 2.3 Tensor types by class

Names and shapes are equal for all 1,328 tensors. Only the storage types
differ:

| Class (count) | Shape | ds4 | hf2q |
|---|---|---|---|
| `ffn_gate_exps`, `ffn_up_exps` (43 each) | [4096, 2048, 256] | **IQ2_XXS** | Q2_K |
| `ffn_down_exps` (43) | [2048, 4096, 256] | **Q2_K** | Q3_K |
| `ffn_{gate,up}_shexp` (43 each) | [4096, 2048] | Q8_0 | Q2_K |
| `ffn_down_shexp` (43) | [2048, 4096] | Q8_0 | Q3_K |
| `ffn_gate_inp` (43) | [4096, 256] | F16 | F32 |
| `hc_{attn,ffn}_fn` (43 each), `output_hc_fn` | [16384, 24], [16384, 4] | F16 | Q8_0 |
| `attn_compressor_{ape,gate,kv}` (41 each) | | F16 | Q8_0 |
| `indexer.attn_q_b`, `indexer.proj`, `indexer_compressor_{ape,gate,kv}` (21 each) | | F16 | Q8_0 |
| `token_embd` | [4096, 129280] | F16 | Q8_0 |
| `attn_q_a`, `attn_q_b`, `attn_kv`, `attn_output_{a,b}` (43 each), `output.weight` | | Q8_0 | Q8_0 |
| norms, sinks, `exp_probs_b`, HC base/scale | | F32 | F32 |
| `ffn_gate_tid2eid` (3) | [6, 129280] | I32 | I32 |

Type totals: ds4 has F32 492, F16 359, Q8_0 345, IQ2_XXS 86, Q2_K 43 and
I32 3. hf2q has F32 535, Q8_0 532, Q2_K 172, Q3_K 86 and I32 3.

### 2.4 Same source weights (sample parity, `$D/sample-parity.json`)

This was measured on the CPU from small reads of both files:

- All 492 tensors that are F32 in both files are byte-identical.
- 18 of 18 sampled tensors that are Q8_0 in both are byte-identical
  (`attn_q_a`, `attn_kv` and `attn_output_b` at layers 0, 1, 2, 3, 21 and 42).
- The 3 hash-routing tables are byte-identical.
- F16 (ds4) against Q8_0 (hf2q) has relative L2 error 0.54–0.78%, which is
  Q8_0 rounding:
  - `hc_attn_fn`/`hc_ffn_fn`/`output_hc_fn`: 0.656%, 0.648%, 0.778%;
  - compressor and indexer: 0.536%, 0.538%, 0.589%.
- The F16 router against hf2q's F32 router: 7.1e-9 and 5.5e-9.
- blk.5 `ffn_gate_exps`, expert 0, rows 0–63: IQ2_XXS (ds4, decoded with the
  peer's algorithm) against Q2_K (hf2q) has correlation 0.899, relative error
  0.450 and equal standard deviation (0.02546 vs 0.02550). Both are ~2-bit
  encodings of the same row in the same orientation.

Conclusion: the two files encode the same official weights in the same layout
and orientation. The tensor mapping is the identity on names. Only the
storage types change.

### 2.5 Bytes read per decode step `[INFERENCE]`

These figures come from the header sizes, assuming each step reads every
non-routed matrix once:

| | ds4 file | hf2q artifact |
|---|---:|---:|
| Routed experts, 6 of 256 (one slot) | 1.83 GB | 2.35 GB |
| Shared experts | 1.15 GB | 0.39 GB |
| Attention matrices | 5.27 GB | 5.09 GB |
| Output head | 0.56 GB | 0.56 GB |

At one slot the ds4 file reads about 8% more bytes (Q8_0 shared experts). At
B=4 joint decode with 24 distinct experts it reads about 6% fewer. The file
alone does not hand hf2q a speed win. The race compares engines on the same
bytes.

## 3. mlx-native: N-IQ2XXS

### 3.1 Storage type registration

`GgmlType::IQ2_XXS` gets an explicit `#[serde(rename = "iq2_xxs")]`. The
enum's `rename_all = "snake_case"` does not yield ggml names for `IQ…`
variants, and ADR-066 N1 requires frozen serde names. Seams, all exhaustive
matches in `R/`:

1. `src/gguf/mod.rs:78-88`: `GGML_TYPE_IQ2_XXS = 16`. Add the parse arm in
   `ggml_type_from_u32` (`:395-416`); type 16 currently reaches
   `"unsupported GGML type ID"` at `:412-414`.
2. `src/ops/quantized_matmul_ggml.rs`:
   - `BLOCK_IQ2_XXS_BYTES = 66` beside `:180-200`;
   - the enum variant (`:205-255`);
   - `block_values` 256 (`:263-281`) and `block_bytes` 66 (`:285-303`);
   - `kernel_name`, `mm_kernel_name`, `mm_tensor_kernel_name` and
     `mm_tensor_v2_kernel_name` (`:307-407`) all return `"unsupported"`,
     because no dense IQ2_XXS kernel ships.
   - The dense mv geometry match (`:1716-1735`) ends in `unreachable!()`. It
     MUST return a typed error for IQ2_XXS instead of panicking.
3. `compute_byte_len` (`gguf/mod.rs:424-455`) is correct once the block
   tables return 256/66. No other check is needed, because type 16 has one
   layout.
4. `load_tensor` (`:1688-1735`) and `raw_tensor_dtype` (`:1946-1962`) store it
   as U8, like the other block types.
5. `src/ggml_capability.rs:385-400` (`quantized_matmul_type`):
   - IQ2_XXS is a native route type for expert `mv_id`/`mm_id` only.
   - Dense mv/mm report "no kernel", using the existing IQ4_XS-style dense
     exclusions at `:661`, `:806` and `:855`.
   - The fused gate/up dense set (`:934-945`) excludes it.
   - Expert route planning (`:987-1000`) includes it.
6. `src/ggml_dispatch_trace.rs:215-230` needs no arm, because it has an error
   fallback and no dense IQ2_XXS mv exists.

### 3.2 Block geometry and CPU dequant

Block layout (`P/ggml/src/ggml-common.h:381-385`): `{ f16 d; u16 qs[32]; }`,
66 bytes per 256 values. Each 32-value sub-block `ib32` reads two `u32`, `a0`
and `a1`, from `qs[4*ib32 .. 4*ib32+4]` in little-endian order:

- `db = f32(d) * (0.5f + (a1 >> 28)) * 0.25f`
- For `l` in 0..4:
  - `grid = iq2xxs_grid[(a0 >> 8l) & 255]` as 8 bytes;
  - `signs = ksigns_iq2xs[(a1 >> 7l) & 127]`;
  - `y[8l + j] = db * grid[j] * ((signs & kmask_iq2xs[j]) ? -1 : 1)`.

This is `P/ggml/src/ggml-quants.c:2489-2513` exactly, including the
evaluation order. mlx-native adds `fn dequantize_iq2_xxs(data: &[u8], output:
&mut [f32]) -> Result<()>` in `src/gguf/mod.rs`:

- It uses the shape and errors of `dequantize_q2_k` (`:614-660`): a length
  that is not a multiple of 66, or a short output, is a `GgufParseError`.
- It gets an arm in `dequantize_to_f32` (`:1281-1299`) and a
  `test_only_dequantize` path (`:1241-1266`).
- The expressions are written in the same order as the C code, so an
  optimizing compiler cannot reassociate them. There are no adds to contract
  into FMA.

**Bit-exact test (ADR-066 N1 acceptance), in `tests/iq2_xxs_dequant.rs`:**
- **Fixture.** `tests/fixtures/iq2_xxs_dequant_v1.bin` holds input blocks and
  the expected F32 bit patterns. A developer generator
  (`tests/fixtures/gen_iq2_xxs_dequant.c`) links only the pinned peer
  checkout's `dequantize_row_iq2_xxs` and `ggml-common.h` tables. It records
  the peer commit and both source sha256 values in the fixture header.
- **Blocks.** 4,096 seeded random blocks, plus edge blocks:
  - all 16 scale nibbles;
  - every grid index 0–255 and sign index 0–127 at every `l`;
  - `d` = +0, −0, the smallest subnormal, 1.0, the largest finite f16, and
    ±inf and NaN. NaN outputs are compared by class (`is_nan()` on both
    sides), not by bit pattern: whether a compiler lowers the ±1 factor to a
    negation or a multiply changes the NaN sign bit. No real file carries a
    NaN scale.
- **Assertion.** `assert_eq!` on `f32::to_bits` for every non-NaN value, plus
  a second hand-ported Rust loop of the C code, as in the Q2_K test
  (`R/tests/deepseek_q2_k.rs:36-83`).
- The generator runs only on a developer host; the test reads the checked-in
  fixture. This keeps Python and peer code out of the runtime (AGENTS.md).

### 3.3 Grids: one checked source

The source is a new module, `src/quant/iq_tables.rs`. It holds `IQ2XXS_GRID:
[u64; 256]` and `KMASK_IQ2XS: [u8; 8]`. `KSIGNS_IQ2XS: [u8; 128]` is a
`const fn` table equal to `i | (popcount(i) & 1) << 7`; all 128 entries were
verified against the peer table. The module is written so ADR-066 N3 can add
`iq2xs`, `iq2s`, `iq3xxs`, `iq3s` and `iq1s` beside it.

Metal side:
- mlx-native shaders are standalone files, each carrying its own structs and
  tables.
- The IQ2_XXS tables are emitted between `// BEGIN GENERATED iq2xxs` and
  `// END GENERATED iq2xxs` markers in each shader that needs them (§3.4).
- A bless test (`tests/iq_tables.rs`, `MLX_BLESS_IQ_TABLES=1`) rewrites those
  blocks from the Rust source. The normal run asserts that each shader block
  is byte-identical to the rendered Rust table.

Verified against ggml:
- `tests/iq_tables.rs` asserts these sha256 values of the little-endian table
  bytes, measured from the peer checkout (`$D/sample-parity.json`; `kmask`
  computed from its eight bytes `1, 2, 4, …, 128`):
  - `iq2xxs_grid`: `05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54`;
  - `ksigns_iq2xs`: `a76742a603f8beca5212ecce0f1f02f11a4887fd2f7c8b15aca0ea3eb3380c31`;
  - `kmask_iq2xs`: `5ac9831b2e30eb285ef34f8501620f878432d5c04331ad1ae47f977a83ba41a5`.
- ds4's own Metal copy (`metal/moe.metal:307`) equals the grid and sign
  tables.

### 3.4 Expert kernels

Every kernel keeps mlx-native's existing expert ABI:

- buffers: `src0` = weights (0), `src1` = 1, `dst` = 2, `ids` = 3, params = 4;
- the weight pointer is `src0 + expert_id * p.expert_stride`;
- the input row is `src1 + (output_row / top_k) * ne10`;
- invalid expert ids go through `poison_invalid_expert_id`.

The ABI and these helpers are at `R/src/shaders/quantized_matmul_id_ggml.metal:37-78`.

**`kernel_mul_mv_id_iq2_xxs_f32`** (decode, and the forced per-token route of
joint decode):
- **File.** It goes in `quantized_matmul_id_ggml.metal`, ported from the
  peer's `kernel_mul_mv_iq2_xxs_f32_impl` (`P/ggml/src/ggml-metal/kernels/mul_mv.metal:1872-1972`).
- **Layout.** Each simdgroup produces `N_R0 = 4` rows (the peer's
  `N_R0_IQ2_XXS`). There are 2 simdgroups (the peer's `N_SG_IQ2_XXS`), so 8
  rows per threadgroup.
- **Shared memory.** The 256 grid entries (2,048 B) and 128 sign bytes go in
  threadgroup memory, 2,176 B in total. They are loaded cooperatively once
  per threadgroup, with one barrier.
- **Inner loop.** Each lane owns 32-value sub-blocks with stride 32. It sums
  `yl * grid * sign` in F32, multiplies by `d * (0.5 + s)` and applies the
  final `× 0.25` after `simd_sum`.
- **Host.** In `dispatch_id_mv` (`R/src/ops/quantized_matmul_id_ggml.rs:1162-1200`)
  the geometry arm is `(nth0, nth1, align) = (32, 2, 8)`, with 2,176 B of
  threadgroup memory, set the way the Q8_0 nr2 path does (`:1210-1229`).
- **Grid.** `(ceil(N / 8), n_tokens * top_k, 1)`.
- **Row-bitwise property.** One threadgroup reduces each output row of each
  routed row. No value crosses routed rows, and the geometry does not depend
  on `n_tokens`. So an `m = B` call equals B one-row calls bit for bit, which
  joint decode requires (§6.3).
- The peer's `FC_mul_mv_split` variant (`ntx = nb32` for short rows) is
  omitted. At K = 4096, `nb32 = 128 > 32`, and the split path never applies.

**`kernel_mul_mm_id_iq2_xxs_f32`** (prefill, more than 32 routed rows,
`MM_ID_ROUTING_THRESHOLD`, `quantized_matmul_id_ggml.rs:1084`):
- A `dequantize_iq2_xxs(device const block_iq2_xxs*, short il, thread
  half4x4&)` plug-in for the existing `hf2q_mul_mm_id_impl<block_q, nl,
  dequantize_func>` template (`R/src/shaders/quantized_matmul_id_mm.metal:613-848`),
  with `nl = QK_NL = 16`. It is ported from
  `P/ggml/src/ggml-metal/kernels/dequantize.h:509-532`: 16 values per call, a
  half of a 32-value sub-block.
- The instantiation follows `:849-928`.
- The existing map0 already has a `ne20 = 6` instantiation (`:567-583`). At
  256 experts it uses 256 threads and 6 KiB of shared memory (`:2276-2302`).

**`kernel_mul_mm_id_iq2_xxs_f32_tensor`** (tensor-API `mm_id`):
- A `dq_iq2_xxs_id` plug-in for `hf2q_mul_mm_id_tensor_impl`
  (`R/src/shaders/quantized_matmul_id_mm_tensor.metal:384-583`), instantiated
  as at `:584-650`.
- The `id_mm_tensor_kernel_name` selector already falls back to simdgroup
  when a name is `"unsupported"` (`quantized_matmul_id_ggml.rs:2350-2356`).
  The tensor kernel is still required, by ADR-066 N4.

**Paired gate/up.** `quantized_matmul_id_ggml_pooled_pair[_with_policy]`
(`quantized_matmul_id_ggml.rs:589-859`) takes one `ggml_type` and runs map0
once for both projections. ds4's gate and up are both IQ2_XXS, so the pair
entry needs only the new `mm_id` name. hf2q already enforces equal types
(`src/inference/models/deepseek4/forward_support.rs:671-677`).

**Registration.** The three names go into `src/kernel_registry.rs`:
- mv_id beside `:541-561`;
- mm_id beside `:575-591`;
- tensor mm_id beside `:610-654`.

`build.rs:83-124` picks the instantiations up for the metallib. No name may
resolve only through the precompiled metallib (the IQ4_XS gap, ADR-066
Context item 2). The general IQ4_XS fix stays in N1.

Not in this story:
- dense IQ2_XXS mv/mm, which ds4's file never uses;
- a fused IQ2_XXS pair + SwiGLU `mv_id`, as ds4 has
  (`metal/moe.metal:3582`, `kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32`);
- a fused down + six-way weighted sum (`:5801`, `:5901`).

If the §7.5 attribution puts the gap in routed-expert decode, those are
mlx-native kernels on the same IQ2_XXS tables. They extend this story's
kernels, never duplicate them, and need their own design under ADR-066 N7 or
a Q4-DS4-RACE follow-up.

### 3.5 Reused kernels: Q2_K, Q8_0 and F16

- **Q2_K down.** `kernel_mul_mv_id_q2_K_f32`, `mm_id`, tensor `mm_id` and the
  slotted down path exist. The tests cover the paired and slotted entry points
  (`R/tests/deepseek_q2_k.rs:514-560`), but only at K=256, N=8 and 8 experts
  (`:390-420`). No Q2_K test or bench runs the down shape K=2048, N=4096 at
  256 experts. N-IQ2XXS adds that parity (§3.7). hf2q's joint-route proof for
  Q2_K (`joint_decode_routes.rs:1-9`) was made on the gate/up shape, so the
  down shape gets its own row-bitwise test.
- **Q8_0** shared experts, attention and head use existing routes. Q8_0 is
  already in the joint proven set.
- **F16 projections.** mlx-native already has both kernels hf2q needs, but
  hf2q today picks the wrong one for one-row products. `raw_matmul` sends
  every F16 product, including `m = 1`, to `dense_matmul_f16_f32_tensor`
  (`forward_support.rs:333-347`). That kernel stages the F32 input as half
  (`R/src/shaders/dense_mm_f16_tensor.metal:193-215`) and runs a 64×32 tile
  for a single row. On this file that path carries, every decode token:
  - the router `ffn_gate_inp` (43 layers), whose output feeds top-6
    selection;
  - `hc_attn_fn`/`hc_ffn_fn` (86) and `output_hc_fn`;
  - the compressor and indexer projections.

  hf2q's own artifact never takes this path, because it stores these as F32
  or Q8_0. L9 (§4) routes F16 at `m = 1` to the existing
  `dispatch_dense_matvec_f16w_f32io` (`R/src/ops/dense_gemm.rs:189-232`,
  source-registered at `R/src/kernel_registry.rs:1308`). It reads the input
  as F32, widens the half weights to F32 and accumulates in F32
  (`R/src/shaders/dense_gemm.metal:148-223`). Its vector loop needs
  `K % 4 == 0`; every F16 matrix in the file has K ∈ {1024, 4096, 16384}.
  Products with `m > 1` (matrix prefill) keep the tiled kernel. Joint decode
  already splits float storage per row (`row_matmul.rs:30-31`), so serial and
  joint both issue the same `m = 1` matvec and stay bitwise equal.
- **F16 token embeddings** use `embedding_gather_dense`
  (`R/src/ops/embedding_dense.rs:44-66`). No new kernel.

### 3.6 Capability rows

The IQ2_XXS capability row reports:
- `mv_id`, `mm_id` and tensor `mm_id`: native;
- dense mv/mm: "no kernel".

ADR-066 N1's other new types (IQ2_XS, IQ2_S, IQ3_*, IQ1_S, MXFP4, MXFP8,
F8_E4M3) are untouched by this story.

### 3.7 Parity tests and microbenchmarks at DeepSeek-V4-Flash shapes

Each new test gets a `[[test]]` entry (`autotests = false`,
`R/Cargo.toml:36`). All tests run with the precompiled metallib on, and again
with `MLX_PRECOMPILED_METALLIB=0`.

| Test | Shape | Rows | Oracle and bound |
|---|---|---|---|
| `iq2_xxs_dequant` | random and edge blocks | | bit-exact (§3.2) |
| `iq_tables` | | | sha256 pin and shader-block equality (§3.3) |
| `iq2_xxs_mv_id_deepseek` | 256 experts, top-6, K=4096, N=2048 | 1, 2, 4, 6, 8, 24, 32 routed rows | CPU dequant (bit-exact) then F64 dot. `|got − want| ≤ 2.5e-4 · Σ|wᵢxᵢ|`, the worst-case F32 summation bound K·2⁻²⁴ at K = 4096 |
| `iq2_xxs_mv_id_row_bitwise` | gate/up as hf2q's joint decode issues it: `n_tokens = m`, `top_k = 6`, through the forced `mv_id` entry | m = 1..8 against m one-row calls | bitwise equal (joint-decode proof, §6.3) |
| `iq2_xxs_mm_id_deepseek`, `iq2_xxs_mm_id_tensor_deepseek` | same | 33, 64, 512, 4096 tokens × top-6 | F64 oracle, `≤ 2e-3 · Σ|wᵢxᵢ|` (half staging of the weights in both templates and of the input in the tensor path, `_tensor.metal:517-518`; 2·2⁻¹¹). Valid only on the value ranges below |
| `iq2_xxs_pair_deepseek` | gate+up, one map | 33, 512 tokens | each output equal to the unpaired `mm_id` output bit for bit |
| `q2_k_down_deepseek` | 256 experts, top-6, K=2048, N=4096 | the same row sets, mv/mm/tensor/slotted | same bounds and value ranges; slotted equal to flattened bit for bit (`R/tests/deepseek_q2_k.rs:524` pattern) |
| `q2_k_down_mv_id_row_bitwise` | down as hf2q's joint decode issues it (`ffn_forward.rs:562-578`): `n_tokens = 6·m`, `top_k = 1`, through the forced `mv_id` entry, not the auto-routing one (at m = 8 there are 48 routed rows, above the 32-row `mm_id` threshold) | m = 1..8 | bitwise |
| `iq2_kernels_resolve_from_source` | | | every IQ2_XXS kernel name builds with `MLX_PRECOMPILED_METALLIB=0` |

Test conventions:
- Weights are seeded random blocks; scale nibbles are drawn uniformly and
  every grid index is reachable.
- Value ranges for every Metal parity test, so that half staging neither
  overflows nor goes subnormal:
  - IQ2_XXS `d` is drawn from ±[2⁻¹², 2⁻⁴]. Every non-zero dequantized
    weight then has magnitude in [2⁻¹², 2⁻⁴ · 166.6], inside the normal half
    range [2⁻¹⁴, 65504]. The grid bytes are 8, 25 and 43, and the scale
    factor is at most 15.5 · 0.25.
  - Q2_K `d` and `dmin` are drawn from [2⁻¹², 2⁻⁴].
  - Inputs are drawn uniformly from [−4, 4], and values with magnitude below
    2⁻¹⁴ are set to 0.
  - Edge scales (subnormal, ±inf, NaN) belong to the CPU dequant test only.
- Routes are a seeded top-6 over 256 experts, plus one adversarial route in
  which all tokens share one expert.
- Bounds are fixed here, before any Metal run (ADR-066 Numerical contract).

`benches/bench_deepseek_iq2_xxs.rs` (`harness = false`) records medians of at
least 5 runs:
- GB/s and µs for `mv_id` at 6, 12 and 24 routed rows (one to four slots);
- the pair `mm_id` at 512 and 4096 tokens;
- Q2_K `mv_id` at the same gate/up shape, as the byte-normalized baseline
  from hf2q's current artifact;
- Q2_K down `mv_id` at K=2048.

Microbenchmarks are recorded, not gated. Performance is judged only by the
four-agent race.

### 3.8 Release and pin

The work is one mlx-native PR and one release, the next patch after 0.15.16:
- crates.io bytes verified;
- `mlx-native = "=<version>"` pinned in hf2q from a clean registry build;
- the native-primitive closure of ADR-066's table.

No other ADR-066 story ships in that release, so ADR-066 batch 1 (N1, N2,
S1) does not gate v0.1.30. The same mlx-native release MAY also carry
ADR-063's v0.1.30 mlx-native work, such as Q4-GEMMA-KERNEL; batch 0 of the
ADR-066 amendment means exactly this.

### 3.9 Relationship to N1, N3 and N4

- **N1** later adds the remaining types to the same enum and table pattern. It
  does not touch IQ2_XXS's parse, dequant or serde name.
- **N3** adds the other IQ grids to `src/quant/iq_tables.rs` and the generated
  shader blocks.
- **N4** adds V4.1 expert-shape parity and microbenchmarks for the existing
  IQ2_XXS kernels: K = 5120/2304, 384 experts top-6, 128 experts top-3. If a
  V4.1 shape needs a geometry change, N4 makes it in these kernels.

The ADR-066 amendment (`ADR-066 N-IQ2XXS`) records this.

## 4. hf2q loader changes

There is one implementation PR after the pin. Each change is exhaustive and
fails closed for anything else.

| # | Change | Where | 0731 artifact behaviour |
|---|---|---|---|
| L1 | `QuantType::IQ2_XXS` ↔ `general.file_type` 19, canonical `"IQ2_XXS"`, added to `ALL`, `as_str`, `from_canonical_str`, `from_gguf_file_type` and `gguf_file_type` | `src/serve/quant_select.rs:47-130,172-184` | unchanged (file_type 10) |
| L2 | Accept `tokenizer.ggml.pre` ∈ {`deepseek-v3`, `joyai-llm`} | `deepseek4/tokenizer.rs:37-42`; also exercised by `src/serve/info.rs:245` | unchanged |
| L3 | `embedding_length_out` optional, default `embedding_length × hyper_connection.count`, equality check kept. `swiglu_clamp_shexp` optional, default `swiglu_clamp_exp` | `deepseek4/config.rs:163,247-249,204,287-293` | unchanged (both present and equal to the defaults) |
| L4 | Accept `compress_ratios.len() ≥ block_count` and use the first `block_count` entries. Every entry past `block_count` MUST be 0. Shorter arrays fail | `config.rs:271-278` | unchanged (43 = 43) |
| L5 | F16/BF16/F32 token embedding through `embedding_gather_dense` | `deepseek4/forward.rs:109-141` (now `bail!` for anything but Q2_K/Q8_0), reached from `forward.rs:41`, `attention_forward.rs:374-386` and `joint_decode.rs:410-423` | unchanged (Q8_0 arm) |
| L6 | Float `output_hc_fn` is issued per row through `row_matmul(IndependentRows)` and classified `RowMatmul`, not `OutputHead` | `deepseek4/output_forward.rs:76`, `joint_decode_routes.rs:33-40` | unchanged (Q8_0 keeps `raw_matmul`) |
| L7 | Add IQ2_XXS to `proven_block_type` only for `JointProduct::RoutedExperts`, citing the mlx-native row-bitwise tests (§3.7) and the hardware check (§6.3) | `joint_decode_routes.rs:29-31` | unchanged |
| L8 | Case files and identity for the file (§7.2) | `scripts/four_agent/q5_cases/` | n/a |
| L9 | `raw_matmul` sends `GgmlType::F16 if m == 1` to `mlx_native::ops::dense_gemm::dispatch_dense_matvec_f16w_f32io` (F32 input, F32 accumulation, §3.5); `m > 1` keeps `dense_matmul_f16_f32_tensor` | `deepseek4/forward_support.rs:304-347`, beside the existing `F32 if m == 1` arm | unchanged: the artifact has no F16 raw matrix (§2.3) |

Notes on the table:

- **L1.** ADR-066 L5 later replaces file-type identity for V4.1. This variant
  only gives the 0731 ds4 file a pool and cache namespace. It also fixes the
  `/v1/models` quant label: today the tensor histogram would call the file
  "Q8_0" (`src/serve/load_info.rs:959-1010`).
- **L2.** In the peer, both names select one case with the same three regexes
  (the peer's vocabulary source, `P/src/<peer>-vocab.cpp:327-336`) and `clean_spaces = false`
  (`:2500-2502`, `:2712-2714`). hf2q's `NUMBER/CJK/MAIN_REGEX`
  (`tokenizer.rs:19-21`) match them character for character. The vocabulary
  and merges are byte-identical (§2.2).
- **L3.** This is the peer's behaviour: `swiglu_clamp_shexp` falls back to
  `swiglu_clamp_exp` (`P/src/models/deepseek4.cpp:29-31`), and the output
  width is derived as `dsv4_hc_mult × n_embd` (`:45`) without reading
  `embedding_length_out`.
- **L4.** The peer rejects only shorter arrays (`deepseek4.cpp:47-53`). The
  trailing entry belongs to the next-token-prediction layer
  (`nextn_predict_layers = 1`; no such tensors ship) `[INFERENCE]`. hf2q indexes ratios only
  for `layer < block_count` (`weights.rs:166`).
- **L6.** Today joint decode fails closed for this file, because `OutputHead`
  accepts only Q8_0/Q2_K/Q3_K (`joint_decode_routes.rs:39`). The F16
  tensor-API matmul at `m = B` is not proven row-bitwise. With L9 each split
  row is one `m = 1` matvec, the same kernel serial decode issues.
- **L7.** Without L6 and L7 this file serves through serial decode, which
  forfeits the S3 joint path (`adr063-s3-ds4-joint-decode.md`).
- **L9.** Without it, every one-row F16 product rounds its F32 input to half
  before the router's top-6 selection and the HC mixes, a rounding hf2q's own
  artifact never sees. It also runs a 64×32 tile for one row. Matrix prefill
  (`m > 1`) still stages the input as half on this file. That is a disclosed
  numeric difference between ds4-file prefill and hf2q-artifact prefill, and
  it is covered by the task gates (§6.5). Matrix prefill is not bitwise equal
  to serial decode on either artifact.

Not changed, because the ds4 file already passes:
- the tensor catalog, which checks names and shapes only
  (`weights.rs:66-264`);
- residency and role validation (`residency.rs:279-298`: RawMatrix accepts
  any non-integer type; F16 `*_ape` tensors decode through `load_tensor_f32`);
- joint decode's `RowMatmul`, which already splits float storage per row
  (`row_matmul.rs:30-31`);
- Q8_0 shared experts and head;
- the expert wrappers, which take `ggml_type` and stride from the tensor
  (`forward_support.rs:645-816`);
- provenance (`External`) and receipts (a direct `--model` never reads one,
  `local_artifacts.rs:322-355`);
- `gguf_contract.rs:29-33`, where deepseek4 has no hosted admission.

Unit tests:
- config defaults, the 44-entry ratios, a 44-entry array with a non-zero tail
  (rejected), and a 42-entry array (rejected);
- tokenizer alias encode parity on the existing DeepSeek fixtures;
- `joint_decode_routes` with the ds4 type set;
- F16 embedding gather against `token_embd` rows decoded on the CPU;
- L9: a `raw_matmul` F16 `m = 1` product equals an F64 CPU dot within the
  §3.7 matvec bound. A two-row `row_matmul(IndependentRows)` F16 product
  equals two one-row calls bitwise.

## 5. Tensor mapping

The mapping is the identity on all 1,328 names (§2.3). Each class is consumed
as follows:

| Class | ds4 type | Serial route | Joint-decode route (after L6/L7/L9) |
|---|---|---|---|
| Routed gate/up | IQ2_XXS | `mv_id` (≤32 rows), pooled pair `mm_id` (>32) | `RoutedExperts`, forced `mv_id`, proven by §3.7 |
| Routed down | Q2_K | `mv_id`, `SlottedMm` (>32, `ffn_forward.rs:543-560`) | `RoutedExperts`, forced `mv_id`, proven (existing); down shape re-proved (§3.7) |
| Shared experts | Q8_0 | `quantized_matmul_ggml` | `RowMatmul`, proven |
| Router `ffn_gate_inp` | F16 | decode: `dense_matvec_f16w_f32io` (L9); prefill: `dense_matmul_f16_f32_tensor` | `RowMatmul`, per-row matvec |
| `hc_{attn,ffn}_fn` | F16 | same | `RowMatmul`, per-row matvec |
| `output_hc_fn` | F16 | same | `RowMatmul`, per-row matvec (L6) |
| Compressor / indexer | F16 | same; `*_ape` decoded to F32 at load | per row (`m = 1`) |
| Attention Q8_0, `attn_output_a` | Q8_0 | existing | `RowMatmul` / `GroupedOutputA`, proven |
| `output.weight` | Q8_0 | existing | `OutputHead`, proven |
| `token_embd` | F16 | `embedding_gather_dense` (L5) | same |
| Hash tables | I32 | existing | existing |

## 6. Quality checks

### 6.1 Tensor-load parity (CPU)

The check is an `#[ignore]` hf2q integration test,
`tests/deepseek4_ds4_file_parity.rs`. It needs `HF2Q_DS4_FILE`, and
`HF2Q_DS4_HF2Q_ARTIFACT` for T5. It opens the file through
`mlx_native::gguf::GgufFile` and `Deepseek4Weights`, exactly as serve does,
and allocates no Metal model runtime. It runs once in a reserved window; it
reads the whole 87 GB file, so it is a heavy-disk job.

| # | Check | Pass rule |
|---|---|---|
| T1 | Header and catalog | 1,328 tensors. The type of each class equals §2.3. Every byte extent equals its block-geometry size. Every loader gate (L1–L4) passes |
| T2 | Bytes as loaded, and file identity | For every tensor, the sha256 of the bytes hf2q's loader exposes equals the sha256 from an independent reader. The reader is `$D/gguf_head.py` extended to per-tensor digests and a full-file sha256, a developer harness under `scripts/four_agent/`. The full-file digest MUST equal `ca22ae2f…61c0`. This receipt is the verification of the `MODEL_SHA256` every case file declares (§7.2) |
| T3 | Decode parity on real blocks | mlx-native's CPU dequant of a deterministic sample matches the bit-exact Rust oracle from §3.2 (`to_bits` equal). The sample is rows 0, 1023 and 2047 of every expert of `ffn_{gate,up}_exps`, and rows 0, 2047 and 4095 of `ffn_down_exps`, at layers 0, 2, 3, 21 and 42. That covers hash, compressed and last layers, IQ2_XXS and Q2_K |
| T4 | F16 and F32 classes | Every F16 tensor decodes to finite values. F16 `*_ape` tensors loaded through `load_tensor_f32` equal an elementwise f16→f32 conversion |
| T5 | Same-source cross-check against hf2q's artifact | Exact or absolute rules: every tensor that is F32 or Q8_0 in both files is byte-identical (§2.4 sampled 492/492 F32 and 18/18 Q8_0). F16 against Q8_0 has relative L2 ≤ 1%. F16 router against F32 router ≤ 1e-6. Relative rule for the 129 expert tensors (IQ2_XXS/Q2_K against Q2_K/Q3_K), next paragraph |

T5 expert rule. For each expert tensor, on the T3 sample rows of experts 0,
85 and 170, the test computes the Pearson correlation of ds4's decoded rows
with five candidates from hf2q's artifact:
- **correct:** the same layer, expert and row;
- **wrong expert:** expert e+1 mod 256, same layer and row;
- **wrong layer:** layer l+1 mod 43, same expert and row;
- **transposed:** the same values read with the other axis as rows;
- **shifted:** the row offset by one block (256 values).

The tensor passes when `correct` is the largest of the five and exceeds the
largest control by at least 0.3. No absolute floor is used. The single §2.4
sample (0.899) cannot calibrate one across 129 tensors; the relative rule
catches layout errors whatever the per-layer level. Every value goes into
the receipt. These are layout detectors, not quality claims.

### 6.2 Kernel parity on real weights (GPU, granted lane)

`iq2_xxs_real_weights` is an `#[ignore]` mlx-native test that takes a GGUF
path. For blk.3 and blk.21 gate/up/down it loads the whole tensor and runs
`mv_id` at 6/24 routed rows and `mm_id` (simdgroup and tensor) at 512 tokens
with seeded routes. It uses the §3.7 oracles and bounds, with inputs in the
§3.7 range; the weights are the file's own. It needs the GPU lane but only
about 2 GB of buffers, not the model runtime.

### 6.3 Joint == serial bitwise on this file (S3 check)

The existing DeepSeek official-artifact joint-decode bitwise test
(`adr063-s3-ds4-joint-decode.md`, widths 2–8) runs with the ds4 file. Logits
and sampled tokens at every width MUST be bit-identical to serial decode.
After L6/L7/L9, `unproven_joint_route` MUST return `None` for this file; that
is asserted from the startup log. ADR-063 Q4 names this check as half of
DeepSeek-V4's Q4 quality (`ADR-063…md:539`).

### 6.4 0731 non-regression

L2–L7 and L9 touch shared DeepSeek code. On hf2q's own artifact the candidate
MUST show `ExactTranscripts` against the pre-change main binary on the
canonical four-agent client, plus the existing DeepSeek unit and route tests
and the joint bitwise test. ADR-066's B1 golden does not exist yet; this is
ADR-063's current DeepSeek gate.

### 6.5 Paired task-quality gates

These use `quality_campaign.py`: 24 instances, Q1 numeric policy §3.5
`paired`, ABBA order, every run a Q5-supervised runtime. The subject is
always the exact candidate binary.

| Gate | Subject | Reference | Role |
|---|---|---|---|
| QG-peer | candidate on the ds4 file | the peer, best valid configuration, on the ds4 file, `--chat-template-file` the reviewed `aec3d906…` template | required (`peer`), bound to the ds4-file campaign of record |
| QG-stock | candidate on hf2q's artifact | v0.1.28 on hf2q's artifact | required (`stock-hf2q`), bound to the stock record (§7.4, D1) |
| QG-ds4 | candidate on the ds4 file | ds4 c97964b on the same file (`ds4-native`) | descriptive. Each ds4 Pass is checked by the trace audit (ADR-063 native section) |

Patches from passing runs still need independent review, as for every Q1
quality campaign.

## 7. Race plan

### 7.1 Preconditions

- **Q4-DS4 (#661) reported Fit.** The peer had 7 of 7 valid runs on this
  file. Its best valid configuration is load mode none with ubatch 1024,
  makespan median 65.2 s. Evidence: `/Users/robert/.local/state/hf2q/v0130/q4-ds4/`
  (`runs/peer-*`, `stage-profile-comparison.{json,txt}`).
- Had Q4-DS4 reported no-fit, only §3 (N-IQ2XXS, ADR-066 scope) could have
  landed. §4 and every later section belong to Q4-DS4-RACE, which ADR-063
  does not start on a no-fit, and ADR-066 serves no ds4 file.
- N-IQ2XXS is released and pinned. L1–L9 and the §8 harness changes are
  merged. §6.1 and §6.2 pass.
- The root grants the GPU lane (README-agents) for each hardware step.

### 7.2 Case files

- `q5_cases/deepseek4-ds4file.json` is a copy of `deepseek4.json` with:
  - `MODEL` set to the ds4 path;
  - `MODEL_SHA256 = ca22ae2f…61c0`;
  - `MODEL_ID = "DeepSeek V4 Flash"`;
  - a new `workload.id`;
  - a note that admission is a fresh calibration, not the 2026-10-30
    receipt.
- There is no sidecar receipt: model files are never modified or
  accompanied (README-agents). `q5_artifacts.py:58-89` records an
  operator-provided sha256 as not verified by the supervisor, and the ds4
  adapter does not hash inline. The verification is §6.1 T2's full-file
  digest. Its receipt path goes into each case's notes, and the campaign
  reports cite it.
- `q5_cases/peer-deepseek4-ds4file.json` takes its flags from Q4-DS4's best
  valid configuration (load mode none, ubatch 1024). It keeps
  `--chat-template-file` at `aec3d906…`, which the client enforces
  (`family_peer.py:45-46,153,176-177`).
- `q5_cases/native-ds4-deepseek4-samefile.json` is a copy of
  `native-ds4-deepseek4.json` (which already names this file) with:
  - `MODEL_SHA256` filled in;
  - the same-artifact declaration (§8.2);
  - the label `ds4 c97964b (same ds4 file)`.

  The existing different-artifact case stays unchanged for its own
  campaigns.

### 7.3 Fresh Q5 calibration

Run `q5_calibration.py` on `deepseek4-ds4file.json`. The hf2q weights are
86.7 GB instead of 107.4 GB, so the earlier DeepSeek admission does not
transfer. Then run the case's `CalibrationPass` on the release-candidate
binary (ADR-063 Q4).

### 7.4 Campaigns

All three campaigns run with one candidate binary and one runtime at a time.

1. **Stock record, on hf2q's own artifact** (existing harness):
   `q3_campaign.py --family deepseek4 --stock-bin <v0.1.28> --experiment-bin
   <candidate> --peer-unavailable-evidence <the 2026-10-30 peer OOM receipts,
   $E/peer-arms-v1/peer-deepseek4-*>`.
   - Arms: stock and experiment, Williams blocks 5–20, n4 primary.
   - It supplies the vs-stock makespan ratio, its precision and the fairness
     effects. Its own `family_verdict` is `Unavailable` (no peer) and is not
     the verdict of record.
   - QG-stock runs on this artifact.
   - Freeze its `verdict.json` and `campaign.json` read-only before step 2
     binds them, as for `--controls-of`.
2. **ds4-file campaign of record** (§8.1 mode):
   `q3_campaign.py --family deepseek4 --hf2q-case
   q5_cases/deepseek4-ds4file.json --peer-case
   q5_cases/peer-deepseek4-ds4file.json --no-stock-arm --stock-record
   <step 1 verdict.json> --experiment-bin <candidate>`.
   - Arms: experiment and peer, n4 primary, then the n1 and 3+1/2+2/1+3
     controls (Q3 design).
   - It produces the Q4 `family_verdict`, with the vs-stock terms read from
     the bound record.
3. **Same-file native comparison** (§8.2 mode):
   `q3_campaign.py --family deepseek4 --hf2q-case
   q5_cases/deepseek4-ds4file.json --peer-case
   q5_cases/native-ds4-deepseek4-samefile.json --native-comparison
   --no-stock-arm --experiment-bin <candidate>`.
   - Arms: the experiment and ds4 with `--ctx 393216` (the Think Max rule).
   - It is reported in `verdict.json` `native_peer`, task makespan only.
     Prompts are not byte-identical (ds4 renders tool schemas first, and its
     Think Max prefix differs), even though the weights are the same bytes.

Then `q3_report.py --rows <step 2> --of-record --quality-gate
stock-hf2q=<QG-stock manifest> --quality-gate peer=<QG-peer manifest>`.

### 7.5 Per-stage attribution

The hf2q and peer arms are measured on the same file in step 2's window. The
stages are:
- prefill;
- decode;
- routed MoE against attention GPU time;
- dispatch and encode;
- scheduler wait;
- SSE cadence.

hf2q uses the Q4-TRACE hooks (`docs/research/adr063-q4-trace-design.md`).
The peer uses Q4-DS4's per-stage method (`$Q4DS4/profile_extract.py`, where
`$Q4DS4` is `/Users/robert/.local/state/hf2q/v0130/q4-ds4`). Distortion is
stated from an A/A pair, trace on against trace off: the makespan ratio with
its interval. The microbenchmarks in §3.7 are cited only as supporting
evidence.

### 7.6 Verdict

- Win requires all of:
  - `q3_stats.family_verdict == Win` on step 2: the peer ratio's 95% CI
    upper bound ≤ 0.95 and its precision met there. The vs-stock gain,
    precision and fairness band are met in the bound step 1 record;
  - QG-peer and QG-stock passing;
  - §6.3 bitwise;
  - every deterministic task, tool and isolation gate in steps 1 and 2.
- The released claim is: "hf2q finishes the four-agent task on ds4's
  DeepSeek-V4 file X% sooner than the peer on the same file." It always
  carries the artifact identity. The vs-stock figure is reported as
  "candidate against v0.1.28 on hf2q's DeepSeek-V4 artifact".
- The file is a benchmark artifact, not a substitute for hf2q conversion
  (ADR-063 Q4-DS4).

## 8. Harness changes (Q4-DS4-RACE, hf2q `scripts/four_agent/`)

Each change is a precondition for §7.4. Each fails closed, and none changes
the behaviour of an existing campaign. Line numbers refer to `d817dffa`.

### 8.1 No-stock-arm campaign with a bound stock record (D1)

1. **`q3_campaign.py` arguments.**
   - New `--no-stock-arm`. It is allowed only together with exactly one of:
     - `--stock-record VERDICT_JSON` (the verdict-of-record mode);
     - `--native-comparison` (§8.2), where no vs-stock row exists anyway
       (`q3_report.py:522-524`).
   - `--stock-record` without `--no-stock-arm` is refused.
   - With `--no-stock-arm`, `--stock-bin`, `--stock-commit`, `--stock-label`
     and `--stock-var` are refused.
2. **`q3_campaign.arm_cases`** (`:127-134`) omits `stock` under
   `--no-stock-arm`. `build_plan` (`:136-206`) is changed in three places:
   - `explicit`, `extra` and the `HF2Q_BIN` defaulting loop (`:143-147`,
     `:177-178`) skip the absent stock arm;
   - `declared.stock` becomes `null`;
   - `plan["stock_record"]` is bound.
3. **Binding the stock record.** It reuses `primary_of_record`'s pattern
   (`:98-125`). The plan stores:
   - the verdict and campaign paths and their sha256 values;
   - family, primary status, the stock and experiment binary sha256 values
     and the artifact sha256.

   The refusals are:
   - the record is not a complete `deepseek4` primary;
   - the record is a no-stock or native campaign itself;
   - its experiment binary sha256 differs from this campaign's
     `--experiment-bin` (checked at launch in `q3_identity.launch_problems`
     and re-checked before every run);
   - either file changed since binding, as with `--controls-of`
     (`adr063-q3-campaign.md:173`).
4. **`q3_identity.launch_problems`** (`:135-160`):
   - require `--stock-bin`/`--stock-commit` only when a stock arm exists;
   - skip the "stock and experiment binaries are identical" check without
     one;
   - add the stock-record binary check.

   `plan["stock_record"]` is part of the plan that a resume must reproduce
   exactly (`build_plan`, `:136-137`), and the bound files' sha256 values are
   re-checked before every run, so a changed record halts the campaign.
5. **`q3_report.required_effects`** (`:73-79`) waits on `vs_stock` only when
   the plan has a stock arm. In a stock-record campaign, the stopping rule
   waits on `vs_peer`.
6. **`q3_report.compute_verdict` state** (`:199-260`). In a stock-record campaign:
   - `vs_stock` and `fairness` are read from the bound record's
     `verdict.json` primary effects;
   - `stock_failures` and `experiment_failures` are the union of both
     campaigns;
   - `thermal_valid` and `memory_valid` are the conjunction of both;
   - `state["stock_source"]` names the record (path and sha256).

   Rows emitted with `--of-record` carry the vs-stock effect with the note
   "from the same-artifact record <id> on hf2q's artifact". `--of-record` is
   accepted for a stock-record campaign (`:459-461` refuses only
   controls-only, native-comparison and native-peer campaigns).
7. **`q3_stats.family_verdict`** (`:89-125`). The decision logic is
   unchanged; it consumes the record-sourced `vs_stock`/`fairness`.
   - New input `stock_source`; when present, every vs-stock reason names it.
   - A missing or non-Complete bound record yields `Pending` with a reason,
     never a silent skip.
8. **`f2_q3_tier.required_references`** (`:146-151`) keeps
   `[stock-hf2q, peer]` for a stock-record campaign. The quality-gate
   validation (`:110-140`) binds:
   - the `stock-hf2q` gate to the bound record's stock binary and hf2q
     artifact (subject: the record's experiment binary, equal to this
     campaign's);
   - the `peer` gate to this campaign's peer binary and the ds4 file.

   `f2_q3_view.py` labels the fairness cell "vs stock (hf2q artifact
   record)".
9. **Docs:** `docs/research/adr063-q3-campaign.md` and the four-agent
   README gain the mode, its refusals and the root command.

### 8.2 Same-artifact ds4 native comparison (D2)

1. **Declaration.** A closed field `artifact_relation` ∈ {`different`,
   `same_as_hf2q`} in the native case's approval `native_ds4` block
   (`native_peer_ds4.native_spec`, `:308-312`; the key set is extended). The
   default is `different` for the existing case. It is not a case variable,
   so `--peer-var` cannot change it. It is bound through `case_sha256` and
   added to `native_peer_ds4.campaign_identity` (`:292-295`), so drift halts
   the campaign.
2. **`native_peer_ds4.NATIVE_COMPARISON`** (`:118`) becomes a string, for
   example "same-file native comparison on ds4's GGUF; not the verdict of
   record". `q3_campaign` (`:172-176`) then accepts `--native-comparison`.
   - A new adapter hook `mode_problems(case, evidence, comparison)` (added
     to the `native_peers.INTERFACE`, `native_peers.py:30-44`) refuses:
     - `same_as_hf2q` with `--peer-unavailable-evidence`, because the
       same-file peer runs (Q4-DS4 Fit);
     - `different` with `--native-comparison`, because the different-artifact
       ds4 arm stays the unavailable-peer stand-in.
   - TensorFold and oMLX implement the hook as a no-op, and their behaviour
     is unchanged.
   - `PEER_CASE_REFUSAL` (`:127-128`) states both modes.
3. **`launch_problems`** (`native_peer_ds4.py:298-305`) gets a case
   argument; the `native_peers.INTERFACE` text at `:41` and
   `q3_identity.launch_problems` (`:147-149`) pass it through.
   - For `same_as_hf2q` it requires both the native and the hf2q artifact
     sha256 to be present and equal. The `MODEL_FILE` name check stays.
   - For `different` it keeps today's refusal.
   - The TensorFold and oMLX `launch_problems` (`native_peer_tensorfold.py:375`,
     `native_peer_omlx.py:354`) refuse a `same_as_hf2q` case.
4. **Labels** become relation-aware. `label(case)` returns
   `ds4 c97964b (same ds4 file)` for `same_as_hf2q`, and `q3_role(case)`,
   `BASELINES` and `baseline_of` use the same relation (`ds4_samefile` as a
   new F2 baseline key). Every hard-coded "different artifact" text becomes
   conditional:
   - `native_peer_ds4.py:4-5,42,114-117`;
   - `native_peer_ds4_audit.py:137-138`;
   - `q3_report.py:451`;
   - `f2_q3_view.py:160`;
   - `f2_q3_tier.py:60`;
   - `f2_scoreboard_render.py:365`, a "Native-engine comparisons" section
     that renders "same file" rows separately;
   - `quality_campaign.py:45`;
   - the four-agent README's ds4 Q3/Q1 bullet (`README.md:248`).

   The statement stays task makespan only, never a win.
5. **Tests:**
   - `test_native_peer_tensorfold.py:335` asserts `NATIVE_COMPARISON is
     None` and becomes a non-`None` assertion for ds4.
   - New cases in `test_native_peer_ds4.py` and
     `test_native_peer_ds4_campaign.py`: same-file launch requires equal
     sha256 values; evidence mode is refused for `same_as_hf2q`; comparison
     mode is refused for `different`; labels and baselines follow the
     relation.
   - New cases in `test_q3_campaign.py`: `--no-stock-arm` refusal matrix,
     stock-record binding, experiment-binary mismatch, a changed record
     file.
   - New cases in `test_f2_q3_tier.py` and `test_f2_q3_record.py`:
     stock-hf2q gate bound to the record's binaries; vs-stock row
     provenance.
   - New cases in `test_f2_scoreboard_render.py` for same-file rows.

## 9. Operator decisions (recorded 2026-11-03)

Both are recorded in ADR-063 (commit `64c6f902`).

- **D1, the stock arm: option (b), as the review recommended.**
  - **Why there is no stock arm on this file.** v0.1.28 and v0.1.29 cannot
    open it (`unsupported GGML type ID 16`, `R/src/gguf/mod.rs:412-414`).
    Running v0.1.28 on a different artifact would mix the artifact change
    with the engine change (AGENTS.md: compare identical artifacts) and
    loosen `q3_identity.same_model`.
  - **What replaces it.** The ds4-file campaign of record has no stock arm
    and is judged against the peer. The vs-stock makespan gain, its
    precision, the fairness band and QG-stock come from the same-artifact
    candidate-vs-v0.1.28 record on hf2q's own artifact, with the same
    candidate binary (§7.4 step 1, §8.1).
  - **Consequence.** The Q4 vs-stock rule (95% CI of the ratio below 1)
    still applies, so the candidate must also be measurably faster than
    v0.1.28 on hf2q's artifact for a Win.
- **D2, ds4 on the same file: the widened declaration (§8.2).**
  - The declaration is bound into the native case and the campaign
    identity.
  - Both sha256 values must be present and equal.
  - It runs only under `--native-comparison`; `--peer-unavailable-evidence`
    is refused.
  - Labels and tests are fixed.

## 10. Risks

- **The gap is large** `[INFERENCE from one run each, different
  artifacts]`. Q4-DS4's stage comparison
  (`$Q4DS4/stage-profile-comparison.txt`) shows a single peer run on this
  file and a single hf2q v0.1.29 run on hf2q's own artifact:

  | | peer (ds4 file) | hf2q (own artifact) |
  |---|---:|---:|
  | Makespan | 64.8 s | 103.0 s |
  | Prefill, median tok/s | 125 | 74 |
  | Decode, median tok/s per stream | 16.8 | 10.8 |
  | Semantic TTFT, median | 1.7 s | 8.1 s |

  Winning by 5% therefore needs roughly a 40% makespan cut, not a kernel
  swap. §7.5 attributes it on the same file.
- **IQ2_XXS decode is lookup-bound.** The grid fetch and sign decode cost
  ALU and threadgroup bandwidth that Q2_K does not. The 21% byte saving per
  routed weight (66 against 84 B per 256) may not turn into time
  `[INFERENCE]`. §3.7 measures it. ds4's fused pair + SwiGLU kernel is the
  known faster shape (§3.4).
- **Shared experts are heavier.** Q8_0 shared experts make one-slot decode
  read about 8% more than hf2q's artifact (§2.5). The peer reads the same
  bytes.
- **Per-row splitting costs dispatches.** The F16 `RowMatmul` products
  (router, HC, `output_hc_fn`) issue B one-row matvecs per step in joint
  decode: (3 × 43 + 1) × (B − 1) extra small dispatches. If §7.5 shows
  it matters, a proven row-bitwise multi-row F16 matvec is the follow-up.
- **D1 couples the verdict to a second campaign.** A failure or quality loss
  in the stock record blocks the Win.

## 11. Work order

1. **mlx-native N-IQ2XXS (§3).** One PR, then release, then crates.io
   verification. Local checks: `cargo test` for the §3.7 tests twice, once
   with `MLX_PRECOMPILED_METALLIB=0`, and the bench recorded.
2. **hf2q pin and L1–L7, L9 (§4),** with unit tests and §6.4 on hf2q's
   artifact.
3. **Harness §8 and L8 case files (§7.2),** with the harness tests.
4. **GPU lane:**
   1. §6.1 (including T2's file digest) and §6.2;
   2. §6.3;
   3. the Q5 calibration (§7.3);
   4. §7.4 step 1, then QG-stock;
   5. §7.4 steps 2 and 3, then QG-peer and QG-ds4 (§6.5);
   6. attribution (§7.5);
   7. the verdict (§7.6).

## 12. Out of scope

- Other IQ types, MXFP4, type 43, shard sets and V4.1 shapes (ADR-066 N1–N5
  proper).
- Converting to IQ2_XXS in `hf2q convert` (ADR-066 C6).
- Dense IQ2_XXS kernels.
- Executing the file's embedded Jinja template.
- ds4's private layouts and any ds4 file other than this one (ADR-066
  Global acceptance and exclusions, as narrowed by `24885e63`).
