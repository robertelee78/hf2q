# ADR-040 §21 — N=8 decode speed: current state + codex-reviewed roadmap (2026-06-27)

## Measured state (mvN default-on, gemma4 Q5_K_M, M5 Max)

| metric | value |
|---|---|
| N=8 decode (pre-mvN baseline) | 197 t/s |
| N=8 decode (mvN default-on) | **220 t/s** (+11.7%) |
| N=8 decode (mvN + KVENC default-on) | **228 t/s** (+15.7% vs baseline) |
| llama.cpp N=8 decode | 291 t/s |
| gap to llama | 1.47× → **1.28×** |
| per-step(N=8) GPU-busy | 28.75ms |
| per-step(N=8) wall | 36.2ms (79% GPU-busy) |
| dispatches/step | 2147 → **1731** (KVENC) |

**Key arithmetic:** hf2q GPU work (28.75ms/step) ≈ llama's *entire* step (27.5ms for 291 t/s).
The kernels are now near-competitive (mvN closed the F3 weight-reload gap). The residual 1.28×
is dominated by the **~7.5ms/step (21%) CPU-encode/non-overlap** (≈ 2147 dispatches ×
~3.5µs/dispatch encode), plus a ~5% residual GPU-work tail (Q5_K dense projections + Q8_0
MoE-down are not mvN-covered — mvN is Q6_K-only).

## Shipped this session

- **mvN default-on confirmed live** (`kernel_mul_mv_q6_K_f32_mN`, mlx-native 0.9.4): the F3
  weight-reload-per-row lever, bit-exact, +11.7%. **Gate 1 (precompiled-metallib byte-parity)
  CLEARED** — `xcrun metal` is now available on this host (Metal Toolchain v17.6), build.rs
  emits a real 3.1MB metallib, and `adr_040_q6k_mv_mN_byte_parity` passes under it. Gate 2
  (mvN-on release timing flake) was already fixed (encoder-retain 80a58de).
- **Batched KV-encode flipped default-ON** (`HF2Q_BATCHED_KVENC`, batched_body.rs:657):
  fuses N per-slot KV-encode into 2 grid-dim-N dispatches. **+3.5%, −19% dispatches/step,
  byte-parity GREEN.** Opt out: `HF2Q_BATCHED_KVENC=0`.

## Codex-reviewed roadmap for the remaining 21% (in priority order)

1. **Dispatch-record reuse for the batched-body decode loop** (the real lever for the
   encoding-bound 7.5ms). The single-seq path pre-bakes `decode_record_*` (m=1) OnceLocks;
   the batched path needs new `rows=N` / `n_tokens=N` / `n_tokens=N*top_k` records, preserving
   the forced `mv_id` MoE-down route. Byte-identity risks (codex): scalar-m=1 geometry leaking
   into batched-N, row-view/stride binding offsets, missing conflict-tracking around raw
   `dispatch_record`. Must stay byte-identical (gate: `slot_aware_n8_per_slot_parity_vs_serial`).
2. **More dispatch fusion** (the KVENC pattern, extended): collapse remaining per-slot loops
   into grid-dim-N dispatches where bit-identical (e.g. embed-gather).
3. **Persistent batched scratch buffers**: hoist the per-step `BatchedDecodeBuffers` +
   `positions_buf` + `slot_id_buf` allocations (batched_body.rs:1285/1289/1302). Codex's
   lowest-risk lever, but lower payoff than (1)/(2) since the 7.5ms is encoding-dominated, not
   allocation-dominated — measure its share before investing.
4. **Extend mvN to Q5_K (dense q/k/v/o) + Q8_0 (MoE-down)** for the residual ~5% GPU-work tail.

**Not viable:** parallel-encode "encode step k+1 while GPU runs step k" — blocked by the
autoregressive dependency (token k+1 unknown until step-k logits are sampled). Confirmed by codex.

## Update: intra-step CB pipelining BUILT + MEASURED — hypothesis refuted

Implemented the codex-blessed first lever: gated single-threaded intra-step command-buffer
pipelining (`HF2Q_DECODE_CB_CHUNKS=K`, batched_body.rs) — split the 30-layer decode loop into
K command buffers, async-`commit()` each in order (GPU runs chunk c while CPU encodes chunk
c+1), only the last waits. Cross-CB ordering = same-queue commit order (no fences). **Parity
GREEN** (`slot_aware_n8_per_slot_parity_vs_serial` byte-identical at K=3; default serial path
also re-verified green).

**Measured (N=8, gemma4 Q5_K_M): only +1.5–2.7%** (K=3–10, near the run-to-run noise floor) —
**NOT the ~20% predicted.** This **refutes** the "7.5ms/step recoverable CPU-encode" model: if
the wall-vs-GPU-busy gap were serialized decode-body encode, chunking would have recovered most
of it. It did not. So the 21% is dominated by **per-token engine overhead** (host
sampling/argmax, admission/scheduling) and the process-global gpu_busy spanning prefill+lm_head
— not the decode-body encode. Kept **gated, default OFF** as a small byte-exact lever + a
documented negative result.

**Revised gap attribution:** the remaining 1.27× (228 vs 291) is NOT decode-body encode. Next
investigation targets: (a) per-token host overhead (sampling, the scheduler/admission loop) vs
llama's; (b) residual GPU work (extend mvN to Q5_K dense + Q8_0 MoE-down — the ~5% still
reloading per-row). Record-reuse (cheaper per-dispatch encode) would also only help the
encode fraction, which this experiment shows is small — deprioritized.
