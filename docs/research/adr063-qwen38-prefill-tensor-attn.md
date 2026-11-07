# ADR-063 gap program F3: tensor attention for resumed Qwen prefill

`$Q` = `~/.local/state/hf2q/v0130/q38-opparity`. Design: `$Q/F3-design.md`. Deficit: `$Q/DEFICIT.md` §5. Inventory: `$Q/hf2q-inventory.md` §4. Measurements: `$Q/hf2q-prefill-20261107T042411Z/hf2q-ops.jsonl`.

Status: implemented behind a default-off switch. No hardware gate, Q1 run or serving A/B has been run yet.

## Scope

Qwen3.5/3.6/3.8 full attention for a resumed prefill chunk on a TQ-only slot, with a cached prefix (`c > 0`) and more than 32 queries (inventory route (c)). Route (c) dequantizes the prefix to F32, un-rotates it, casts Q/K/V to BF16 and runs the BF16 resume kernel. That kernel is 92–94 % of the route and runs at about 10.7 TFLOPS; the peer's F16 kernel runs at 17–19 TFLOPS. The measured deficit per chunk at n = 512 is +18.1 ms at c = 2048 and +79.8 ms at c = 8192.

Unchanged: fresh chunks (`c = 0`, route (a)), chunks of 32 queries or fewer (route (b), direct TQ), the F32 slot route (`HF2Q_TQ_KV=0`), head dims other than 256, and callers outside `forward_gpu_impl` (MTP drafts, the greedy decode forward), which keep route (c).

## Route (c′)

`src/inference/models/qwen35/prefill_tensor_attn.rs`, selected in `apply_sdpa_with_kv_cache`'s TQ resume branch. One command buffer per FA layer, committed without a host wait:
1. `tq_dequantize_hb_kv_seq_f16` writes K and V over `[0, L)` into F16 staging, in the FWHT-rotated domain (no undo). `fwht_sign_premult_f32` rotates `q_rope` in place (it has no reader after attention). The rotation is orthonormal, so attention in the rotated domain equals attention in the original domain.
2. In the first FA layer of the chunk only: the causal F16 mask (`q_abs_offset = c`, `seq_len_k = kv_pad`) and its blk at `(bq, bk) = (32, 64)`. Both depend only on `(c, n)`, so the other 15 FA layers reuse them.
3. One `flash_attn_prefill_tensor_f16_d256` dispatch. Scale 1/16 is folded into Q. Q and O are seq-major F32 `[n, 24, 256]`. K and V rows have stride 256 and heads stride `L·256`.
4. `fwht_sign_undo_f32` on O, which is then the layout the gate multiply and output projection already consume.

Staging, mask and blk are allocated once per chunk and owned by a thread-local scope that `forward_gpu_impl` opens around its layer loop. They are dropped when the forward returns. They are not taken from the decode pool. Each staging buffer holds `(n_kv_heads − 1)·L + kv_pad` rows. Rows past `n_kv_heads·L` are zeroed at allocation and never written. The kernel reads them, masked to −inf, so they must stay finite.

Every later FA layer must see the chunk's `(c, n)`; otherwise it errors. If one chunk's mask would exceed `QWEN_PREFILL_TENSOR_MASK_CAP_BYTES` (256 MiB, n·kv_pad·2 B), the queries are split into blocks of a multiple of 32 rows. Each block has its own mask and blk, rebuilt per layer. Every block except the last drains before the next block allocates its mask, so only one block's mask is live at a time.

Per FA layer this is 5 dispatches, plus 2 per chunk, against 10 for route (c). It removes the bridge's host wait (16 per chunk).

## Switch

`HF2Q_QWEN_PREFILL_TENSOR_ATTN`:
- Unset or `0`/`off`/`false` (default): route (c).
- `1`/`on`/`true`: route (c′) if `flash_attn_prefill_tensor_available(256)` holds; otherwise route (c), labelled `bridge (tensor unavailable)`.
- It is read once per model load. An invalid value warns and keeps the default, and any other probe error fails the load.

The resolved route (`bridge`, `tensor` or `bridge (tensor unavailable)`) appears in three places:
- the load banner (`hf2q load: qwen_prefill_attention = …`);
- `active_controls.qwen_prefill_attention` in `/hf2q/v1/runtime`;
- the Q1 Qwen actual-trace `actual_route`, which is suffixed whenever the switch was requested.

## Numerics

Route (c′) is Q1 Class B against route (c), never bitwise. Q, K, V and P are half, with an 11-bit mantissa, against BF16 with an 8-bit mantissa. The softmax uses lazy rescaling.

Model-free gates (in `prefill_tensor_attn::tests`):
- **F64 reference** at Qwen3.8 shapes (24/4/256). Cases are `(c, n)` ∈ {64, 2048} × {64, 512}, plus `(100, 64)` and `(64, 33)` with `L % 64 ≠ 0`. The TQ cache is written by the production `hadamard_quantize` encoder. The reference is CPU F64 attention over the un-rotated F32 dequant of the same cache. Bound: `max|Δ| ≤ 8·2⁻¹¹·max|V|`.
- **Repeatability:** bitwise across runs, with fresh and with reused mask/blk.
- **Chunking invariance:** bitwise for a 1024-row suffix split as 1×1024, 2×512 and 512+33+479, and for 64-row query blocks under the mask cap. Key tiles start at absolute 0, and the kernel's rows are independent.
- **Bridge agreement:** route (c′) against route (c) within `8·(2⁻¹¹ + 2⁻⁸)·max|V|`. The distance of each route to the F64 reference is printed as a diagnostic.
- **Micro-bench** (ignored): µs per layer for both routes at c ∈ {2048, 8192}, n = 512.

Q1 coverage of the resumed route is T1Resume64 / T3Chunk64 (a 64-token chunk at c = 64), run as a bridge arm and a tensor arm in separate processes (design §6).

## Deviations from the design

- The switch is read and probed at model load rather than at worker start. That way the load banner and `active_controls`, which are built at load, carry the route. An invalid value warns and keeps the bridge, the pattern the other Qwen switches use, instead of failing startup. A probe error other than "no tensor API" still fails the load.
- Under the mask cap, a block other than the last ends with a host wait. Without that wait, every block's builder-allocated mask stays live until the layer completes, and the cap would not bound memory. The cap engages only for n·kv_pad > 128 Mi (for example n = 2048 above 64 Ki context).
