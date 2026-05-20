# ADR-036: Parallelize convert pipeline — close the 3.4× wall-clock gap

**Status**: **SHIPPED 2026-05-19** (commit `3b24daea`) — all 11 kernels parallelized via `par_chunks_exact_mut`; Q4_K_M 10.1× speedup (12:05→1:12), 0/658 byte-cmp mismatches across Q4_K_M / Q5_K_M / IQ4_NL / Q4_0 validation runs. hf2q convert is now 3.0–3.3× FASTER than canonical pipeline.
**Date**: 2026-05-19
**Related**: ADR-033 §P1 (byte-equivalence), ADR-035 (codec ownership)

## Context

ADR-033 §P1 benchmark on Gemma 4 26B-A4B-IT Q4_K_M (M5 Max):

| Pipeline                                                  | Wall clock | CPU user | Cores  |
|-----------------------------------------------------------|-----------:|---------:|-------:|
| canonical (`convert_hf_to_gguf.py` + `llama-quantize`)    | **3:35**   | ~746s    | 4-5 (443%) |
| hf2q (`hf2q convert --quant q4_k_m`)                      | 12:05      | 662s     | 1 (~93%)   |

CPU work is comparable. hf2q is 3.4× slower wall-clock because the convert orchestrator (`src/convert/cli_driver.rs:539-544`) processes tensors sequentially:

```rust
for (idx, step) in plan.steps.iter().enumerate() {
    let data: Vec<f32> = step.materialize(&src, &synthesized)?;
    sw.stream_tensor(idx, &data)?;
}
```

Three independent units of work per iteration: (1) `step.materialize` reads safetensors → F32 (I/O + cast), (2) `sw.stream_tensor` quantizes F32 → ggml-quant bytes (CPU-bound), (3) writer emits the payload to disk (I/O). Wall-clock is dominated by (2) — the quantize kernels — which is 100% parallelizable across tensors AND across rows within a tensor.

`rayon = "1.10"` is already a project dependency.

## Decision

After ADR-033 §P1 closes, parallelize the convert pipeline in two layers:

### Layer A: per-row parallelism inside each `quantize()`

Every kernel's `quantize(src, n_per_row, imatrix)` walks rows sequentially with `for row in 0..n_rows { quantize_row(...) }`. Each row's quantization is independent (it owns its own per-row state — `weights`, `l_arr`, `mins`, `scales`). The row output bytes have known fixed size (`row_blocks * BLOCK_BYTES`), so the output buffer can be pre-allocated and filled in place.

Refactor pattern:

```rust
let row_bytes = row_blocks * BLOCK_BYTES;
let mut out = vec![0u8; n_rows * row_bytes];
out.par_chunks_exact_mut(row_bytes).enumerate().for_each(|(row, dst)| {
    let row_x = &src[row * n_per_row..(row + 1) * n_per_row];
    let mut tmp = Vec::with_capacity(row_bytes);
    match imatrix {
        None => quantize_row_ref(row_x, &mut tmp),
        Some(qw) => quantize_row_impl(row_x, qw, &mut tmp),
    }
    debug_assert_eq!(tmp.len(), row_bytes);
    dst.copy_from_slice(&tmp);
});
```

Apply to all 8 quants currently in `src/quantize/ggml_quants/` (Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, IQ4_NL, plus the imatrix variants). Each kernel's per-row function is unchanged — only the outer driver in `quantize()` is parallelized.

### Layer B: per-tensor parallelism in the convert driver

The convert driver in `cli_driver.rs:539-544` can fan out `step.materialize` across tensors (bounded parallelism to keep memory in check — only the N largest in-flight at any time) while preserving in-order writes via a channel-based gather:

```rust
let (tx, rx) = bounded_channel(in_flight_cap);
rayon::scope(|s| {
    s.spawn(move |_| {
        for (idx, step) in plan.steps.iter().enumerate() {
            let data = step.materialize(...)?;
            tx.send((idx, data))?;
        }
    });
    // Consumer: receives in submission order, writes via sw.stream_tensor
    while let Ok((idx, data)) = rx.recv() {
        sw.stream_tensor(idx, &data)?;
    }
});
```

The producer side runs `par_iter` over plan.steps with a semaphore bounding in-flight materializations. The consumer drains in idx order (the channel preserves order since the producer sends in idx order).

The two layers compose: Layer A parallelizes within a tensor, Layer B parallelizes across tensors. Layer A alone (simpler) captures most of the wall-clock win since the biggest tensors (MoE expert weights at 285MB) dominate. Ship Layer A first; revisit Layer B if it's needed.

## Validation

**Byte-equivalence regression must hold**: per-row parallelism shouldn't change quantized output (each row is deterministic and independent of others). After Layer A, re-run `scripts/byte_cmp_gguf.py` against all 8 canonical references on Gemma 4 26B-A4B-IT. Required result: 0/658 mismatches for every quant.

**Benchmark target**: Layer A alone should bring hf2q from 12:05 → ~3:30 (canonical-equivalent) on M5 Max, since CPU work parallelizes onto 8+ cores while I/O is unchanged.

## Why "after §P1 closes"

Same rationale as ADR-035: don't mix correctness-critical changes (FMA / F16 round-trip) with optimization (parallelism). §P1's 5,264 per-tensor verifications form the regression gate that catches any byte-divergence introduced by parallelization. With §P1 locked down first, ADR-036 is a pure performance change verified by the same byte-cmp test.

## Consequences

**Positive**:
- ~3.4× wall-clock speedup on the convert critical path, with no behavioral change.
- Better hardware utilization (currently 1 of 8+ cores busy).
- Sets up Layer B / pipelined I/O if needed later.

**Negative**:
- Per-row `Vec<u8>` allocation overhead (mitigatable via thread-local buffers).
- Larger peak memory (each parallel worker holds its own row buffer plus the imatrix slice). For QK_K-sized rows this is ~144 bytes × workers, negligible.
- Touches every kernel file in `src/quantize/ggml_quants/`.

## Acceptance Criteria

- All 8 quants × 658 tensors continue to produce byte-identical output to canonical.
- Wall-clock for `hf2q convert --quant q4_k_m` on Gemma 4 26B-A4B-IT drops to ≤ canonical's `convert_hf_to_gguf.py + llama-quantize` (~3:35 on M5 Max).
- `cargo test --release` continues to pass all 2807+ tests.
- No new clippy warnings introduced.
