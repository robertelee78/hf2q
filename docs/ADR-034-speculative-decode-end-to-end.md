# ADR-034: Speculative-decode end-to-end — Native MTP (Qwen 3.6) + DFlash drafter (Qwen 3.6 + Gemma 4), correct and benchmarked

- **Status**: proposed (2026-05-19)
- **Date**: 2026-05-19
- **Deciders**: operator (robert@loveathome.us); claude (deep-research + audit + draft); codex (independent spot-check)
- **Tags**: `spec-decode`, `mtp`, `dflash`, `apple-metal`, `byte-parity`, `coherence-gated`, `multi-arch`
- **Supersedes**:
  - ADR-013 §15 "MTP tensors and speculative draft execution" + ADR-013 P10 + ADR-013 P14 "MTP speculative-decoding execution (COMPLETE)" — the "COMPLETE" status is **inaccurate at HEAD `eab0220b`**; convert side does not exist, loader/forward have never been validated against a known-good reference, and no acceptance-rate or throughput-improvement number has ever been measured. This ADR documents the actual state and the path to genuine completion.
  - ADR-012 §11 / §15 "MTP tensor round-trip integrity gate" — documented `model.mtp.layers.0.* → blk.{n_layer}.nextn.*` mapping as "shipped 2026-04-24" but current code in `src/convert/arch/qwen35moe.rs::map_tensor_name` has zero MTP arms and `src/convert/arch/qwen35.rs` does not exist. Either the convert code was never landed or was removed in a subsequent refactor. This ADR re-derives the mapping from canonical sources and re-lands it under byte-cmp gates.
  - ADR-030 DFlash block-diffusion spec-decode (status: proposed) — partially absorbed. ADR-030's Phase-1.5 sweep, drafter shape locking, and `/opt/dflash` reference are carried forward as inputs to Workstream B (DFlash). ADR-030's status field is reclassified as "subsumed by ADR-034 Workstream B" — no separate ADR-030 follow-up; all DFlash work tracks here.
- **External pins** (load-bearing — correctness gates assume these exact references; SHAs captured 2026-05-19 during codex review):
  - `/opt/llama.cpp` @ `e15384a5cb092b080c2a01c0b9e3f8635079d6df` — includes PR #22673 (merged 2026-05-16). This is the pinned converter the byte-cmp gate (G1) assumes. Operator's local checkout already has this; no re-fetch unless drift detected during P-1.
  - `/opt/dflash` @ `94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756` — `dflash/dflash/model_mlx.py` is the Python reference for Workstream B (582 LOC).
  - `/opt/MTPLX` @ `0ad700ca3a354c58c40217eebeff7f7384b6d99e` — Apple-Silicon native MTP reference. Perf-bar reference: published `63.056 / 62.886 tok/s` D3 on M5 Max (`/opt/MTPLX/README.md:9,59,154`).
  - z-lab DFlash drafters (HuggingFace): `z-lab/Qwen3.6-27B-DFlash`, `z-lab/Qwen3.6-35B-A3B-DFlash`, `z-lab/gemma-4-26B-A4B-it-DFlash`, `z-lab/gemma-4-31B-it-DFlash`.
  - DFlash paper: arxiv **2602.06036** (Chen / Liang / Liu, 2026).
  - Reference Qwen MTP GGUFs: `froggeric/Qwen3.6-27B-MTP-GGUF`, `unsloth/Qwen3.6-27B-MTP-GGUF`, `RDson/Qwen3.6-27B-MTP-Q4_K_M-GGUF`, `havenoammo/Qwen3.6-35B-A3B-MTP-GGUF`.
  - Reference Gemma 4 MTP-assistant: `google/gemma-4-26B-A4B-it-assistant`, `google/gemma-4-31B-it-assistant`, `google/gemma-4-E4B-it-assistant`, `mlx-community/gemma-4-26B-A4B-it-assistant-bf16` (Phase 7 follow-up; not in v1 scope).
  - Official HF target weights: `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.6-35B-A3B`, `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B-it`.
  - rustc: pinned via `rust-toolchain.toml` (mirrors ADR-033's pin).
  - `mlx-native`: pinned via `Cargo.toml:46` path-override (per ADR-008 sovereignty rule). This ADR does NOT modify mlx-native kernels — the `+1` bake is convert-side, not runtime. If a future phase needs a new mlx-native kernel variant, it goes through ADR-008's sovereignty review separately.

---

## 🎯 START HERE — Current state at HEAD `21be1efd` (2026-05-21)

> **READ THIS FIRST.** §1 and §2 below describe the audit baseline at HEAD `eab0220b`
> (2026-05-19) — substantial work has landed since. This section is the authoritative
> hand-off summary for an engineer picking up ADR-034 today.

### Per-cell empirical state

| Cell | Status | Evidence | Remaining work |
|---|---|---|---|
| **A — Qwen 3.6 27B dense MTP** | ✅ **1.16× SPEEDUP SHIPPED** | Auto-default K=1 BATCHED + Metropolis-Hastings stochastic acceptance (task #87 + #91 SHIPPED). Empirical at HEAD: temp=0 greedy 23.4 t/s @ 54.7% accept (1.09×); **temp=0.6 MH 25.1 t/s @ 69.5% accept (1.16×)** vs base 21.4 t/s. Determinism PASS 3/3 byte-identical (greedy AND seeded MH). Coherent essay output at both temperatures. | P6 perf gate vs MTPLX 63 t/s ref (gap remains; #89/#90 needed). |
| **A — Qwen 3.5/3.6 35B-A3B MoE-MTP** | ⚠️ **CORRECT but spec_decode is NET-NEGATIVE on MoE** | Loader works on 8/8 quants (Q4_0..Q8_0..IQ4_NL — all produce identical "**Paris**." at 85.7-100% MTP accept). BUT empirical paired bench at HEAD `0d19f4b0` (Q4_K_M, 200 tok): base 135.8 t/s, K=0 spec 111 t/s (0.82×), K=1 BATCHED 88 t/s (0.65×), K=2 cap=0 75.7 t/s @ 42.1% accept (0.56×). All spec modes SLOWER than base. T_v(N)/T_v(1) ratio = 2.4× (vs 1.26× on dense) — MoE batched-verifier overhead doesn't amortize. Loader fix shipped at afbf5684+66f2008f. | **Task #89** (forward_gpu_batched_decode) required to make spec_decode profitable on MoE. Until then, base K=0 is default (HF2Q_SPEC_DECODE unset). |
| **B — Qwen 3.6 DFlash** | 🚨 **ARCH INTEGRATION** | DFlash drafter loads + safetensors validated (test passes), but `try_dispatch_dflash_spec_decode` requires `target: &mut MlxModelWeights` and calls `target.install_dflash_capture`, `target.rollback_kv`, `target.forward_decode_verify_batched`. `Qwen35Model` does NOT implement any of these (separate forward stack with `HybridKvCache`). Empirically verified at HEAD `afbf5684`: no `install_dflash_capture` / `rollback_kv` / `forward_decode_verify_batched` exists anywhere under `src/inference/models/qwen35/`. | **500-1500 LOC integration** (originally estimated at 50-100 LOC — corrected 2026-05-21 after reading `dispatch_dflash_generate` body). Options: (a) implement DFlash capture session on `Qwen35Model` + `HybridKvCache.rollback_to(pos)` + `forward_decode_verify_batched` for the Qwen35 hybrid stack, or (b) introduce a `DFlashTarget` trait that both `MlxModelWeights` and `Qwen35Model` implement (more refactor up-front but cleaner long-term). |
| **C — Gemma 4 26B DFlash** | ⚠️ **COHERENT but 4.8× SLOWER than base** | Q8_0 paired bench 3 reps at HEAD `6d80e6be` (2026-05-21): base 92.9 t/s, DFlash 19.2 t/s = 0.21× (4.8× slowdown). Coherence is byte-identical to single-token decode at temp=0 (proven by e2e test). The "WORKING" claim in earlier ADR revisions was correctness-only — never measured against base. Source code at `src/serve/spec_decode_cli.rs:26-32` is honest: "Option C re-prefills full prefix from start_pos=0 each round; Option A (cross-length SDPA in flash_attn_prefill) deferred." | **Option A** = cross-length SDPA path (same scope as task #89 `forward_gpu_batched_decode`). Required for any perf gate. ~1500 LOC. |
| **D — Gemma 4 -assistant** | Phase 7, deferred | Not in v1 scope | — |

### Data assets on disk

| Asset | Location | Size | Use |
|---|---|---:|---|
| Qwen 3.6 27B MTP GGUF (Q4_K_M + Q8_0) | `/opt/hf2q/models/Qwen3.6-27B-MTP-GGUF/` | 46 GB | Cell A target |
| Qwen 3.5 35B-A3B safetensors (MoE + MTP heads) | `/opt/hf2q/models/Qwen-Qwen3.5-35B-A3B/` | 67 GB | Cell A 35B-A3B target |
| Qwen 3.5 35B-A3B Q4_K_M canonical ref GGUF (byte-identical to hf2q convert) | `/opt/hf2q/cache/byte_cmp/Qwen-Qwen3.5-35B-A3B_canonical_q4_k_m.gguf` | 20 GB | Test target for MoE MTP loader fix |
| Gemma 4 26B-A4B safetensors + canonical Q8_0 GGUF | `/opt/hf2q/models/google-gemma-4-26b-a4b-it/`, `/opt/hf2q/cache/byte_cmp/` | 50 + 24 GB | Cell C target |
| z-lab/Qwen3.6-27B-DFlash drafter | `/opt/hf2q/models/dflash-drafters/z-lab__Qwen3.6-27B-DFlash/` | 3.3 GB | Cell B drafter |
| z-lab/Qwen3.6-35B-A3B-DFlash drafter | `/opt/hf2q/models/dflash-drafters/z-lab__Qwen3.6-35B-A3B-DFlash/` | 948 MB | Cell B 35B drafter |
| z-lab/gemma-4-26B-A4B-it-DFlash drafter | `/opt/hf2q/models/dflash-drafters/z-lab__gemma-4-26B-A4B-it-DFlash/` | 858 MB | Cell C drafter (✅ tested) |

**External-drive disk (`/Volumes/Extreme Pro`) is at 100% capacity** — additional Qwen 3.6 27B / 35B BF16 safetensors downloads need disk freed first.

### What's already landed (DO NOT redo)

- ✅ MTP convert arms (`qwen35moe_full.rs:687-705`) — byte-identical to canonical
- ✅ `+1` bake on norm.weight tensors (`BakeOp::AddOne` applied at 10 sites)
- ✅ MTP runtime DENSE forward path (`mtp.rs::forward_draft`) — empirically working (Qwen 3.6 27B)
- ✅ MTP runtime **MoE forward path** (`mtp.rs::forward_ffn_residual` MoE branch) — empirically working (Qwen 3.5 35B-A3B Q4_K_M, this session)
- ✅ DFlash scaffold (7011 LOC) — config + weights loaders validated against real files via 6 new tests at HEAD `21be1efd`
- ✅ Parity harness scaffold (`scripts/parity/` + `tests/parity_*.rs`) — Python ref scripts SKELETONS only; Rust tests skip-clean
- ✅ Determinism (Phase -2 prereq) — PASS at HEAD `21be1efd` AND across MoE-MTP path (this session)
- ✅ HF2Q_SPEC_DFLASH env wiring (works for non-qwen35 archs)
- ✅ HF2Q_SPEC_DECODE env wiring (works for qwen35 MTP path)
- ✅ B-W-1 heisenbug closed per ADR-015 iter61a-2 receipts
- ✅ External pins all intact and verified

### Recommended execution sequence (revised after empirical prep)

1. **P-1 docs/supersession** (~30 min, 50 LOC) — mark ADR-013 P14 as superseded
2. ~~**P3.1 MoE MTP loader fix**~~ **LANDED 2026-05-21** at HEAD `afbf5684` + `66f2008f` — enum refactor `MtpFfnWeightsGpu::{Dense, Moe}` in `mtp.rs` + detection/dispatch in `mtp_weights_load.rs::load_mtp_ffn` + IQ4_NL/Q5_1 allowlist extension in `load_moe_ffn_quantized`. Total ~150 LOC across three files (significantly less than the original 300-500 LOC estimate because the production `load_moe_ffn_quantized` + `MoeFfnWeightsGpuQ::from_quantized` were perfectly reusable). Test gate passed across **8/8 quants** of canonical Qwen 3.5 35B-A3B (Q4_0/Q4_K_S/Q4_K_M/Q5_K_S/Q5_K_M/Q6_K/Q8_0/IQ4_NL) — all produce coherent identical text at 85.7-100% MTP acceptance. 20 MTP-related unit tests still pass (dense path preserved).
3. **P1.1 mtp_parity.py impl** (~2-4 hrs, 200-400 LOC) — unblocks G2 gate
   - Build on `scripts/parity/mtp_parity.py` scaffold
   - Use Qwen 3.6 27B MTP GGUF (on disk) + HF transformers OR /opt/MTPLX/mtplx as reference
   - Dump intermediates: enorm/hnorm outputs, eh_proj sum, attn output, ffn output, final logits
   - Compare via `tests/parity_mtp_python_ref.rs` (scaffold landed, skip-clean)
4. **P1.2 dflash_parity.py impl** (~2-3 hrs, 150-250 LOC) — unblocks G2 DFlash gate
   - Build on `scripts/parity/dflash_parity.py` scaffold
   - Use `/opt/dflash/dflash/model_mlx.py` (582 LOC reference) + Gemma 4 26B target (best-ready cell)
   - Dump intermediates: per-layer drafter outputs + final logits
   - Compare via `tests/parity_dflash_python_ref.rs`
5. **P5.1 Qwen DFlash arch integration** (~500-1500 LOC, **NOT a simple wire-up** — corrected 2026-05-21)
   - The original "~50-100 LOC" estimate was wrong: `try_dispatch_dflash_spec_decode` requires `target: &mut MlxModelWeights` and calls `install_dflash_capture` / `rollback_kv` / `forward_decode_verify_batched`. `Qwen35Model` does not implement any of these.
   - Path (a): port `install_dflash_capture` to `Qwen35Model` + add `HybridKvCache.rollback_to(pos)` + implement `forward_decode_verify_batched` for the Qwen35 hybrid attention stack. Reuse existing `apply_sdpa_with_kv_cache` infra. Expected: ~800-1200 LOC.
   - Path (b): introduce a `DFlashTarget` trait in `src/inference/spec_decode/dflash/target.rs`, refactor `MlxModelWeights` methods behind it, then add a `Qwen35Model` impl. Cleaner separation but more up-front refactor. Expected: ~600-1000 LOC.
   - Test gate: `HF2Q_SPEC_DFLASH=1 HF2Q_DFLASH_DRAFTER_PATH=... hf2q generate ...` produces `[HF2Q_SPEC_DFLASH=1]` banner + coherent text on Qwen 3.6 27B target
6. **Run parity gates** against all 3 working cells (A-27B, B-27B, C-Gemma-26B)
7. **P4 DFlash audit via parity diff** (~300 LOC remaining of original 600) — close gaps surfaced by parity harness
8. **P5.2 Leviathan-2023 rejection sampler** validation
9. **P6 perf gates** (~200 LOC scripting) — F1/F2/F3 measurements

**Total remaining: ~2100 LOC** (originally estimated ~2500; -16% net). The savings from ADR-033 §P1 + the MoE MTP loader fix (~750 LOC saved) are partially offset by the corrected P5.1 scope (~600-1400 LOC actual vs 50-100 LOC originally estimated). P5.1 was MIS-scoped in the 2026-05-21 readiness assessment; the corrected number is grounded in reading `dispatch_dflash_generate`'s body.

### Iteration 2026-05-21 — K=N hard limit on hybrid Qwen, K=1 BATCHED bench (HEAD `5505cdfc`)

This iteration corrected two false alarms and characterized a real architectural limit:

1. **False alarm — "pre-existing prefill non-determinism at max_seq>180"** was a `tail -4 | head -2` filter artifact. With a positional filter (`sed -n '/^prefill:/,$p' | tail -n +2 | tr -d '\n' | head -c 80`) the first 80 chars of the generated text are byte-identical across `--max-tokens` 12 / 50 / 100 / 128 (Qwen 3.6 27B Q8_0 MTP). Greedy IS deterministic across `max_seq_len` capacities. Memory entry updated.

2. **Kernel-divergence root cause for K=N divergence from greedy** — confirmed empirically:
   - Greedy (`HF2Q_SPEC_DECODE=0`, single-token decode kernel) byte-identical to `K=1 TWO_CALLS` (`HF2Q_SPEC_DECODE_K1_TWO_CALLS=1`, two single-token decode forwards).
   - `K=1 BATCHED` (single 2-token forward via `flash_attn_prefill_resume`) diverges from greedy at token ~22 — DIFFERENT correct sequence, not degenerate text.
   - `flash_attn_vec` (decode, F32) and `flash_attn_prefill_resume` (prefill, BF16 internal) produce slightly different logits → different argmax on close calls. This is rounding, not a bug.

3. **K=2 BATCHED degeneracy root cause** — DeltaNet recurrent state cannot be rolled back on partial reject. Qwen 3.5/3.6 are hybrid: 4 full-attn layers + 30 linear-attention (Gated DeltaNet) layers. Full-attn slots are rolled back by `truncate_full_attn_to(prior+accepted+1)` (just resets the `current_len` cursor). DeltaNet state is RECURRENT — it advances by N+1 token-steps inside one batched forward. On reject of any of those drafts, the recurrent state is "ahead" with no rollback mechanism. Over ~70 iterations the drift accumulates and the model collapses to the "the the the…" attractor. `LinearAttnStateSlot` has a `conv_state` / `conv_state_scratch` ping-pong but it only supports {all accept, all reject} rollback (the scratch holds state-after-N, the active holds state-before).

4. **K=1 BATCHED PROVEN COHERENT + FASTER** at 200 tokens (3 reps, deterministic):
   - spec mean: 23.93 tok/s @ 60% accept
   - base mean: 21.17 tok/s
   - **speedup: 1.13×**
   - Quality: full coherent essay through to 200 tokens, no degeneracy.
   - K=1 emits a different valid continuation from greedy at temp=0 (due to kernel rounding noted above), but the continuation itself is high-quality.

**Path forward for K≥2 on hybrid Qwen (multi-week scope):**
- **(a)** Per-token DN state snapshot — before each spec iter, copy `conv_state` and `recurrent_state` for all 30 LA layers. On partial reject, recompute prefix from snapshot. Extra: 30 × (conv_state + recurrent_state) bytes per iter + copy time. For 27B: ~5 MB per snapshot, microseconds.
- **(b)** Per-step DN forward — run the verifier as N+1 sequential single-token forwards, swapping `conv_state` after each. Loses batched-prefill speedup; equivalent to K=1 cost × (N+1).
- **(c)** Recompute prefix on reject — call DN forward from scratch on accepted-prefix when partial-reject happens. Costs (avg-accept) × per-layer-recompute per reject.

For pure-attention models (Llama 3, Gemma 4 standard layers without DN), K≥2 should work natively because all attention slots are roll-backable. The hybrid case is the bottleneck.

**Recommendation**: ship K=1 BATCHED as the default `HF2Q_SPEC_DECODE=1` mode (set `HF2Q_SPEC_DECODE_K1=1` automatically when MTP weights are present) and document K=N as experimental until per-layer DN snapshot lands.

### Iteration 2026-05-21 (cont.) — Per-target K=1 BATCHED bench + MoE 2-token batched penalty

Re-bench across BOTH Cell A targets at HEAD `8eca2387`, 200-token essay:

| Model | base | K=0 (single verify) | K=1 BATCHED | K=1 TWO_CALLS | Best |
|---|---:|---:|---:|---:|---|
| **27B Q8_0 (dense MTP)** | 21.1 t/s | 19.3 (0.91×, 76.4% acc) | **22.8 (1.08×, 60.0% acc)** | 19.9 (0.94×, 77.0% acc) | k1batched |
| **35B-A3B Q4_K_M (MoE MTP)** | **135.5 t/s** | 111.4 (0.82×, 73.7% acc) | 88.7 (0.65×, 73.7% acc) | 116.8 (0.86×, 72.4% acc) | base |

**Critical: no spec mode beats base on 35B-A3B MoE.** `HF2Q_MTP_PROFILE=1` shows:

```
Cell A 27B Q8_0 K=1 BATCHED:  mtp=4.2ms,  ver_2tok=59.7ms, T_v(2)/T_v(1)=1.26×
Cell A 35B-A3B K=1 BATCHED:   mtp=2.0ms,  ver_2tok=17.7ms, T_v(2)/T_v(1)=2.40×
```

**Root cause**: `T_v_2tok / T_v_1tok` ratio is broken on MoE. 27B (dense) gives the
normal 1.26× batched penalty; 35B-A3B (MoE, 256 experts top-8) gives a 2.40×
penalty. Per-token batched-forward overhead nearly doubles when adding the 2nd
token — likely because the MoE FFN dispatches a near-disjoint UNION of expert
kernels for two tokens (worst case 16 vs 8 experts), and the per-expert dispatch
overhead doesn't amortize across the 2 tokens efficiently.

**Modified recommendation**:
- **27B dense MTP**: enable K=1 BATCHED by default (1.08× win, deterministic).
- **35B-A3B MoE MTP**: leave as base (no spec). Need MoE-FFN batched-dispatch
  optimization OR a cheaper drafter before spec decode pays back on MoE Cell A.
- **K=N for hybrid Qwen**: still blocked on DN-snapshot rollback (separate work).

**Investigation:**
1. ✅ Tested `HF2Q_MM_ID_ROUTING_THRESHOLD=1` (force mm_id at seq_len=2): 70 tok/s
   (WORSE than mv_id at 91 tok/s; accept also drops to 63.9%). The MoE FFN
   dispatch is already optimal — mv_id route is correctly fastest for tg2.
2. ✅ Located CHUNK_THRESHOLD=64 in `gpu_delta_net.rs` — for seq_len ≤ 64 the
   Gated DeltaNet uses the `dispatch_gated_delta_net_decode` recurrent kernel
   (single dispatch, internal loop over tokens). Geometry `(d_v/nsg, n_v_heads,
   n_seqs)` doesn't grow with n_tokens. T(2) ≈ a + 2b should scale ≤ 2× over T(1).
3. **Conclusion**: The 2.4× ratio on 35B-A3B is NOT from MoE FFN dispatch
   inefficiency. It's from the entire `forward_gpu_impl` prefill-path setup
   (`FaPrefillArena` alloc, BF16 cast Q/K/V per layer, permute seq→head→seq,
   `apply_flash_attn_prefill_seq_major_into` BF16 attention, etc.) being
   invoked at seq_len=2 — designed for batched prefill (seq_len >> 1), not
   small batched-verify. For 27B (dense, slow base) the absolute overhead is
   the same but relative cost is lower because T_v(1) is already 47ms.

**Real fix path** (architectural, multi-week):
- Add a `forward_gpu_batched_decode` path for seq_len ∈ {2, 3, …, 16} that
  reuses the single-token decode kernels (F32 `flash_attn_vec`, mv_id MoE)
  WITHOUT the prefill staging overhead. Loops attention per-token internally
  with shared QKV projection arena, single FFN dispatch with n_tokens > 1.
- Expected: T_v(N) ≈ a + N·b where a (overhead) drops from 10ms to <2ms.
- For 35B-A3B at K=1: T_v(2) drops from 17.7ms to ~10ms → speedup recovers.

**Modified recommendation (final this iteration)**:
- **27B dense MTP**: K=1 BATCHED gives 1.08-1.13× (small win, can ship).
- **35B-A3B MoE MTP**: keep base (no spec) as default. Spec doesn't pay back
  until `batched_decode` path lands. K=1 TWO_CALLS is mathematically incapable
  of beating base on this model (proof: 1.737 verify-forwards + 1 MTP per cycle
  produces 1.737 tokens = same verify-per-token ratio as base, plus MTP overhead).
- **Multi-week scope**: implement `forward_gpu_batched_decode` for seq_len ∈
  [2, 8] to unlock K=N speedup on MoE A3B.

### Iteration 2026-05-21 (cont. 3) — Task #89 batched_decode concrete design

Code-reading deep dive at HEAD `4e3b8ed7` to scope the cross-length SDPA
work (= Option A from `spec_decode_cli.rs:26-32` = task #89 = critical
foundation for Routes A and B).

**Root cause located**: `gpu_full_attn.rs:1865+` (`apply_flash_attn_prefill_seq_major_resume`)
allocates BF16 mirror buffers sized for the FULL slot capacity:

```rust
let k_bf16_slot = device.alloc_buffer(
    kv_slot_elems * 2,  // n_kv_heads * kv_capacity * head_dim
    DType::BF16, vec![1, nkv, cap, d]).map_err(...)?;
```

Then casts F32 → BF16 over the entire `[0..max_seq_len)` even though the
kernel reads only `[0..kv_seq_len)`. For 35B-A3B with `max_seq=2200` and
`kv_seq_len=27` (just after prefill), that's 99% wasted cast work — but
99% of a small buffer. The real cost is the 7+ kernel dispatches per FA
layer (BF16 casts, permutes, FA resume, output permute, cast back to F32),
each with ~5-20μs launch overhead. With 4 FA layers on 35B-A3B that's
~200-300μs absolute. The other ~9-10ms comes from the 36 LA layers each
allocating their `DnPrefillArena` (~22 buffers each).

**F32 sdpa kernel already supports seq_len > 1**:
`/opt/mlx-native/src/shaders/sdpa.metal` handles batched Q via `abs_pos =
kv_seq_len - seq_len + q_pos` with `do_causal` + `kv_capacity` stride. No
new kernel needed.

**Concrete first-step plan (Step 1)**: add a small-batched-decode F32 fast
path in `apply_sdpa_with_kv_cache` (gpu_full_attn.rs:2209) BEFORE the BF16
resume branch:

```
if head_dim == 256
   && cur_len > 0
   && (1..16).contains(&seq_len)
   && slot.k.is_some()
   && slot.tq.is_none() {
    write_kv_with_optional_tq_encode(...)?;
    let q_hm = permute_seq_head_dim_to_head_seq_dim(q_seq_major, ...);
    apply_sdpa_causal_with_capacity(...)?;  // F32 sdpa, kv_capacity=max_seq_len
    return permute_021(out_hm, ...);
}
```

Test gates: (a) byte-identical determinism 3/3 runs on 27B K=1 BATCHED;
(b) coherent output; (c) T_v(2) on 35B-A3B drops ≥ 20% from 17.7ms;
(d) 27B K=1 BATCHED speedup unchanged or better.

Step 2 (next iter): lift FA prefill arenas off when seq_len < 16. Step 3:
LA layer batched_decode path. Full task #89 scope ~1500 LOC across all 3
steps.

### Iteration 2026-05-21 (cont. 4) — Step 1 design PIVOT after empirical profile

**Two mantra catches this iteration before writing wrong code:**

**Issue 1 — Chesterton's fence on F32 sdpa**: gpu_full_attn.rs:2438-2447
explicitly documents "Legacy SDPA: all-NaN at qL ≤ 15 on Qwen3.6 ...
qL=15 NaN, qL=17 coherent." Original Step 1 design (F32 sdpa bypass of
BF16 resume) targeted a kernel that DOESN'T WORK at qL=2 in production.

**Issue 2 — Per-kernel profile shows wrong bottleneck**:
`HF2Q_PROFILE_W5B8=1` on 35B-A3B K=1 BATCHED steady-state decode:

| Section | Time | % of T_v(2) |
|---|---:|---:|
| **fa.ops1_4** (QKV+norm+RoPE × 4 FA layers) | **9.33 ms** | **53%** |
| fa.sdpa_total (4 FA layers) | 1.87 ms | 11% |
| fa.ops6_7 (output projection) | 1.88 ms | 11% |
| layer.linear_total (30 LA aggregated) | 1.03 ms | 6% |
| layer.ffn_dispatch (40 FFN dispatches) | 2.00 ms | 11% |

fa.ops1_4 = 2.33 ms per FA layer at seq_len=2 vs ~0.18 ms/layer expected
at decode-equivalent compute. The bottleneck is NOT sdpa; it's the
QKV-projection-arena + LayerEncoderSession orchestration designed for
big batched prefill, proportionally ruinous at seq_len=2.

**Revised Step 1 design**: bypass `FaProjectionsArena` + `LayerEncoderSession`
for seq_len < 16. Use decode-mode `pooled_alloc_buffer` for QKV/Gate
projections + per-stage `LayerEncoder::plain()`. Same kernels (they handle
seq_len natively); just lift off the prefill-only orchestration. ~300-500 LOC.

Expected: fa.ops1_4 drops 9.33 → ~3 ms. T_v(2) drops 17.7 → ~11 ms.
Cycle = 11 + 2 (MTP) = 13 ms per 1.737 tokens = 7.5 ms/tok vs base 7.4 —
break-even on 35B-A3B, then small win above.

### Iteration 2026-05-21 (cont. 6) — Task #87 SHIPPED + task #91 design

**Task #87 SHIPPED at HEAD `3be36936`**: K=1 BATCHED auto-default for dense
MTP variant. 14 LOC change in spec_decode.rs gating on
`mtp.ffn_kind() == MtpFfnKind::Dense`. Empirical:
- Dense 27B Q8_0: 23.83 vs 21.40 base = **1.11× speedup** (was silent
  0.91× regression — closed).
- MoE 35B-A3B Q4_K_M: 112.35 tok/s unchanged (auto K=0 preserved).
- Determinism 3/3 byte-identical; 115/0 tests pass.

**Task #91 design (Metropolis-Hastings)**: deep scoping revealed the
`leviathan_step` primitive ALREADY EXISTS at
`src/inference/spec_decode/dflash/rejection_sampler.rs:94-159` (Leviathan-2023
§2.3, identical to MTPLX's sampling.py math). Task is wiring, not
reimplementation. Scope: ~330 LOC across 5 steps (temperature plumbing →
K=1 BATCHED MH branch → K=0 MH → K=N MH → tests). Expected at temp=0.6 on
27B: K=1 BATCHED 1.11× → ~1.30-1.40× via accept rate boost from 60% →
80-90%.

### Iteration 2026-05-21 (cont. 5) — Arena-lift hypothesis FALSIFIED

Implemented + A/B tested the Step 1 design from the previous iteration.
Result: **zero measurable effect**. Hypothesis falsified.

A/B at 35B-A3B Q4_K_M K=1 BATCHED, 100 tok, 3 reps each:

| `HF2Q_SMALL_BATCH_NO_ARENA_THRESHOLD` | Mean tok/s | Mean accept |
|---|---:|---:|
| 1 (legacy `seq_len > 1` arena alloc) | 85.5 | 63.9% |
| 8 (new no-arena for seq ≤ 8) | 84.8 | 65.2% |

Within noise. W5B8 profile confirms fa.ops1_4 at seq_len=2 unchanged at
9.39-9.62 ms (legacy 9.33 ms).

**Falsified premise**: arena setup was already amortized perfectly across
the 4 FA layers. The 9.5 ms is in the kernel-call work, NOT orchestration.

**Real bottleneck (revised)**: Metal dispatch overhead. ~9 kernel dispatches
per FA layer (norm + 4 projections + 2 per-head norms + 2 RoPE) × 4 FA
layers = ~36 dispatches × ~250 μs launch = ~9 ms. Matches empirical.

**Real fix candidates**:
1. Kernel FUSION: combine norm + QKV projection + RoPE into ONE kernel
   per FA layer (~3× fewer dispatches → ~3 ms reduction).
2. Combine Q/K/V/Gate into a single `qkvg` mega-projection (~4× fewer
   matmul dispatches → ~2 ms reduction).
3. Extend `use_fused_stage_ab` (gpu_full_attn.rs:2818) to support
   cur_len > 0 — currently restricted to cur_len == 0 prefill case,
   excludes K=1 BATCHED.

All three are bigger refactors than the original arena-lift. Real Step 1
budget: ~800-1500 LOC across mlx-native shader work + Rust dispatch.

Per mantra "code + test == truth": hypothesis falsified by empirical A/B
before shipping. Pivot to fusion-based approach.

Operator asked: "obviously our peers are doing something better than us? or wtf?
why can they be faster and we can not." Direct answer after reading
`/opt/MTPLX/mtplx/gdn_capture.py`:

**MTPLX's structural advantages over our K=1 BATCHED:**

1. **Per-position GDN state capture** (`gdn_capture.py:128-201` 80-line Metal
   kernel, plus Rust dispatch + cache wiring). MTPLX runs Gated DeltaNet in
   "capture mode" that writes `states[B, T, Hv, Dv, Dk]` — the recurrent state
   at EVERY position in the batch. On partial-reject of K drafts, they pick
   `states[accepted]` as the next-iter active state. This **unlocks K=3 (D3)
   batched spec-decode on hybrid Qwen** — which we cannot currently do because
   `LinearAttnStateSlot` only ping-pongs between {before, after-N-tokens} with
   no intermediate states.

2. **D3 vs D1**: at 60% accept E[tokens/cycle] = 1 + 0.6 + 0.36 + 0.216 = **2.18**
   tokens for D3 vs **1.6** tokens for D1. ~36% more tokens per verifier call.

3. **temp=0.6 stochastic sampling** (not greedy temp=0). MTPLX's recorded
   `63.056 tok/s` is at temp=0.6; their greedy `60.108 tok/s` is still 2.7× ours.

4. **Custom drafter** (`Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`) fine-tuned
   for spec-decode acceptance.

5. **MLX framework batched-prefill kernels** vs our hand-ported mlx-native FA
   prefill. The T_v(N)/T_v(1) ratio likely flatter on mature MLX.

6. **`performance-cold` profile + `--max` fans** = full thermal headroom.

**The single biggest structural lever**: GDN capture. Without it, K≥2 degenerates
(empirically reproduced this session at HEAD `5505cdfc`). With it + a fast
batched-decode forward path, K=3 should give ~2× on 27B and likely 1.5-1.8× on
35B-A3B once the prefill-path overhead is eliminated (task #89).

**Port plan for GDN capture (task #90)**:
- ~80-line Metal kernel `gated_delta_net_capture` mirroring MTPLX's source at
  `gdn_capture.py:132-194`. State buffer shape `[B, T, Hv, Dv, Dk]`. Step loop
  inside kernel reads `parent_state` from `states[t-1]` (or `state_in` for t=0)
  and writes `state_t = states + ((b·T+t)·Hv+hv)·Dv+dv)·Dk`.
- Rust dispatch wrapper in mlx-native: `dispatch_gated_delta_net_with_capture`.
- Extend `LinearAttnStateSlot` with optional `capture_states: Option<MlxBuffer>`
  shaped `[B, T_max, Hv, Dv, Dk]`. T_max = max spec depth + 1.
- Modify `build_delta_net_layer` (and prefill resume variants) to use capture
  variant when `kv_cache.in_speculative_decode == true`.
- Modify `HybridKvCache::rollback_to(absolute_pos)` to select `capture_states
  [pos - prior_len]` as the new active recurrent state.
- Scope: ~600-1200 LOC across mlx-native + hf2q.

Total path-to-D3 budget: GDN capture (~1000 LOC) + batched_decode (~1500 LOC)
+ K=N spec_decode integration (~300 LOC) ≈ **3000 LOC** for the 2× target.

### What an engineer needs (besides this section)

- Read [[project_adr034_readiness_final_2026_05_21]] memory entry for cross-cutting findings
- Read [[project_adr034_mtp_loader_moe_bug_2026_05_21]] for the concrete P3.1 bug detail
- Read [[project_adr034_qwen36_27b_mtp_working_2026_05_21]] for the empirical evidence chain
- Read [[project_adr034_prep_session_summary_2026_05_21]] for the data inventory
- Run `bash scripts/coherence-harness/determinism_check.sh ...` to re-verify Phase -2 at their HEAD

### Sections below — note staleness

- §1.2 has a 2026-05-21 REVISION block but the main text describes HEAD `eab0220b`
- §2.2 "What's missing" table — **partially stale**: convert MTP arms ARE landed; DFlash config+weights ARE validated
- §3.5 corrected 2026-05-21 (MoE row added; original "dense even for MoE" claim falsified)
- §4 phase LOC estimates — **stale by ~40%**, see "Recommended execution sequence" above for revised
- §5 risks — still relevant but missing the MoE MTP loader bug + Qwen DFlash dispatch wiring

---

## 0. Mantra alignment

> "DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it."

This ADR commits, with explicit teeth:

- **No stubs.** Every phase ships compiled, gated, tested code at landing. No `todo!()`, `unimplemented!()`, or "we'll wire this up next phase" placeholders in the production path. Phase 1's harness is the only intentional exception — it is itself the test infrastructure, ungated.
- **No fallback (strict).** When the user explicitly opts in (`HF2Q_SPEC_DECODE=1` env or per-request `spec_decode: on`), the speculative path is the path; missing MTP/DFlash assets, missing draft kernels, or runtime draft failures surface as **typed errors that abort the request**. There is no warn-and-fall-through under explicit opt-in. Warn-and-fall-through to greedy is reserved exclusively for the **auto** mode (no env, no per-request flag, default chosen by `model.mtp.is_some() && !sample_logits` heuristic). The current `serve/mod.rs:2820-2824` code is **wrong** in this respect (it warns-and-falls-through even when the operator set `HF2Q_SPEC_DECODE=1`); P5 fixes the routing to distinguish explicit vs auto mode.
- **No assumption.** Every claim of "this matches the reference" is gated by an actual diff (bytes or numerics) against the locked external pin. Documentation that says "shipped" without a green gate is not shipped.
- **Measure 3×.** Every perf gate runs a thermal-fair alt-pair (≥3 cycles, σ<1 %, 60-90 s cool-downs). Single-run numbers are diagnostic, not load-bearing.
- **Chesterton's fence.** §2 below catalogs every existing piece of MTP/spec-decode scaffolding. We touch only what's genuinely missing or genuinely wrong.
- **Code is truth.** Per `[[project_hf2q_convert_gemma4_f16_dispatch_2026_05_17]]`'s methodology rule: comments and ADRs are starting points; code is truth. This ADR will be the starting point; the diffs and CI logs that close each phase are the truth.

This ADR was authored after a deep audit that found three "shipped" status claims in prior ADRs (ADR-012 P4, ADR-013 P10, ADR-013 P14) that **do not match current code** at HEAD `eab0220b`. The audit was independently spot-checked by `codex exec -s read-only` (2026-05-19 21:36 UTC, exit 0, verdict "partial — central claim supported"). See §1.2.

---

## 1. Context

### 1.1 What native MTP and DFlash mean concretely

Speculative decoding cuts decode latency by having a cheap drafter propose K candidate next tokens and the target model verify them in one batched forward. At `temp=0` (greedy), the accept criterion is exact-match; at `temp>0` the Leviathan-2023 rejection sampler preserves the target distribution. Either way, output is **distribution-identical** to single-token decode — speedup comes only from amortizing per-token kernel-dispatch overhead over multiple emitted tokens, not from quality loss.

Two mechanisms in scope here:

**(a) Native MTP** ("Multi-Token Prediction" heads baked into the target checkpoint). The drafter is one extra transformer block trained jointly with the main model and appended at `blk.{num_hidden_layers}` of the same GGUF. Per the DeepSeek-V3 paper (arxiv 2412.19437 §2.2) the MTP block consists of:

```
input:   prev_hidden [1, H] from verifier   token_embed [1, H] of just-accepted token t+1
         │                                  │
         ▼                                  ▼
       hnorm = (1 + w_h) * rmsnorm(prev)   enorm = (1 + w_e) * rmsnorm(embed)
         │                                  │
         └──────────────┬───────────────────┘
                        ▼
                  concat → [1, 2H]
                        │
                        ▼
                eh_proj @ [2H, H]
                        │
                        ▼
              transformer block (attn + SwiGLU FFN)
                        │
                        ▼
              shared_head_norm + shared_head_head → logits [1, V]
```

The `+1` offset on RMSNorm weights — semantically `(1 + w) * rmsnorm(x)`, **not** the standard `w * rmsnorm(x)` — is a real load-bearing detail and applies **broadly across the Qwen 3.5/3.6 architecture**, not just MTP. The pinned llama.cpp converter (`/opt/llama.cpp/conversion/qwen.py:303-304`) implements the rule in `_LinearAttentionVReorderBase.modify_tensors`:

```python
elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
    data_torch = data_torch + 1
```

This bake applies to **every** Qwen 3.5/3.6 RMSNorm `weight` tensor EXCEPT `linear_attn.norm.weight` (which is part of the SSM/state-space sub-layer with its own convention). Concretely the baked set includes: `model.norm.weight` (output norm); per-block `input_layernorm`, `post_attention_layernorm`, `self_attn.q_norm`, `self_attn.k_norm`; AND the MTP block's `enorm`, `hnorm`, `shared_head.norm` (after the `mtp.layers.{bid}` → `model.layers.{bid + n_layer}` remap at `qwen.py:597-618`).

The runtime then uses **standard** `w_baked * rmsnorm(x)` and the math is correct because `w_baked = 1 + w_original`. **For hf2q to byte-match stock Qwen 3.5/3.6 GGUFs (with or without MTP), the `+1` MUST be applied convert-side to all in-scope tensors, not runtime-side.** Applying both would double-apply and break logits. The current bug at HEAD `eab0220b` is that hf2q convert has no Qwen 3.5/3.6 path at all (dense doesn't exist; MoE mapper has zero MTP arms and zero `+1` bake step) — so today hf2q runtime only works on **externally-converted** Qwen 3.5/3.6 GGUFs (all on-disk apex/dwq/27b-mtp/35b-a3b-mtp files were produced by external tools, not hf2q). **See §1.3 for the bug analysis.**

Native MTP coverage at the time of writing:
- DeepSeek-V3 / V4 (in-checkpoint heads; not currently in hf2q scope)
- Qwen 3.5 / 3.6 27B + 35B-A3B (in-checkpoint heads; Workstream A)
- Gemma 4 (Google trained MTP heads but **stripped them from the public HF release**, retaining them only in LiteRT on-device format; community extractions exist — see Workstream C / Phase 7)

**(b) DFlash** (block-diffusion drafter, external model). The drafter is a separate 2 B BF16 transformer (1-5 layers depending on the target) conditioned on the target's hidden states at a set of `target_layer_ids`. Per `/opt/dflash/dflash/model_mlx.py:DFlashDraftModel` and arxiv 2602.06036, the drafter:

1. Receives input `[last_verified_token, MASK_ID, MASK_ID, …, MASK_ID]` (1 + `block_size-1` mask tokens).
2. Concatenates target hidden states at `target_layer_ids` (e.g. `[1, 6, 11, 17, 22, 27]` for the gemma-4-26B drafter — 6 layers × 2816 hidden → 16896), projects through `fc: Linear(N_targets × hidden_size → hidden_size)` + RMSNorm.
3. Runs its own attention + FFN stack (own KV cache).
4. Argmaxes per mask position → `block_size - 1` candidate tokens.
5. Target then verifies `[last_verified_token, draft_1, …, draft_{K}]` (K+1 tokens) in one batched forward, returning per-position argmax AND target hidden states at `target_layer_ids` for the next iteration.

DFlash coverage: Gemma 4 26B/31B (Workstream B Phase 5), Qwen 3.5/3.6 27B/35B-A3B (Workstream B Phase 4), MiniMax-M2.x, Kimi K2.5/K2.6, gpt-oss-20B/120B, Llama-3.1-8B (out of v1 scope here but trivially extensible).

### 1.2 The audit: what was claimed vs what is true at HEAD `eab0220b`

Two-pronged deep audit (2026-05-19) — claude as primary, codex as independent spot-checker. Codex was given a fixed JSON-schema review prompt and read the source independently; its verdict is in `/tmp/cfa-mtp-audit/codex-last.txt` (preserved for the record).

**REVISION 2026-05-21 at HEAD `193802f3`** — empirical re-verification during ADR-034 prep deep-research:
- Convert side MTP arms: **LANDED** in `qwen35moe_full.rs:687-705` via ADR-033 §P1 closure work. SHA256-byte-identical to canonical's PR #22673 output on Qwen 3.5 35B-A3B (4 nextn.* tensors at blk.40 in `/opt/hf2q/cache/byte_cmp/Qwen-Qwen3.5-35B-A3B_canonical_q4_k_m.gguf`).
- Runtime DENSE MTP: **EMPIRICALLY WORKING**. Tested at HEAD `193802f3` on `froggeric/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q8_0-mtp.gguf` (downloaded during this prep session) — coherent haiku output + **63.6% MTP acceptance rate** at 20.1 tok/s. See [[project_adr034_qwen36_27b_mtp_working_2026_05_21]].
- Runtime MoE MTP: **BROKEN** by a hardcoded-dense FFN loader (`mtp_weights_load.rs:268-291`). Fails on canonical-converted Qwen 3.5 35B-A3B MTP GGUF with typed error. See [[project_adr034_mtp_loader_moe_bug_2026_05_21]]. Fix scope: 300-500 LOC enum refactor + MoE forward dispatch.
- DFlash scaffold: **STILL UNVALIDATED**. 7011 LOC + zero numerical-parity coverage against `/opt/dflash/dflash/model_mlx.py`. Parity harness scaffold landed in this prep session at `scripts/parity/` + `tests/parity_*.rs`; awaits Python-side implementation.
- B-W-1 heisenbug (Phase -2 prereq): per ADR-015 §iter61a-2 receipts at line 114, closed at commits `aa5b410` (mlx-native) + `c8809fc` (hf2q). Determinism check not re-verified in this prep session against HEAD `193802f3` — recommended action before P-1 starts.

**Claimed status (per prior ADRs and `model_card.md` style docs)**:
- ADR-013 P10 ("MTP load path") — "Pending (simple, MTP load-only)" then later marked complete
- ADR-013 P14 ("MTP speculative-decoding execution (COMPLETE)") — claims merged on main at `79140ec` 2026-04-25 with 4 passing tests
- ADR-012 P4 ("GGUF metadata + tensor naming") — claims `model.mtp.layers.0.* → blk.{n_layer}.nextn.*` mapping shipped 2026-04-24
- README.md and `src/inference/models/qwen35/mod.rs` references treat MTP as a working production feature

**Actual state at HEAD `eab0220b`**:

| Layer | File | Status |
|---|---|---|
| Arch catalog declarations | `src/arch/entries/qwen35.rs:165,171,177,183,201`, `qwen35moe.rs:188-208,224` | ✅ exists. `has_mtp: true`; 4 nextn template entries |
| Convert dispatch table | `src/convert/cli_driver.rs:901,924` | ❌ `ArchName` enum has `Qwen35Moe` but no dense `Qwen35` variant; dense Qwen 3.5/3.6 convert has no path at all |
| Convert MTP mapper (MoE) | `src/convert/arch/qwen35moe.rs::map_tensor_name` (195-272) | ❌ **zero MTP arms**. Returns `None` for any MTP HF name. `MapOutcome::Unmapped` → hard error per `cli_driver.rs:1274` |
| Convert MTP mapper (dense) | `src/convert/arch/qwen35.rs` | ❌ **file does not exist** |
| Runtime loader | `src/inference/models/qwen35/mtp_weights_load.rs` (345 LOC) | ⚠️ exists structurally. **Never exercised against any external known-good GGUF in the test suite.** Tests load only synthetic GGUFs with zeros/ones data |
| Runtime forward | `src/inference/models/qwen35/mtp.rs` (524 LOC) | ⚠️ exists structurally. **No numerical-parity test against any reference** (HF transformers, llama.cpp, MTPLX). Runtime kernel is **correct** for stock-baked GGUFs (§1.3) — bug is on the convert side, not here. Other numerical bugs (rope, head_dim mismatch, etc.) remain unknown until G2 lands. |
| Runtime spec-decode loop | `src/inference/models/qwen35/spec_decode.rs` (824 LOC) | ⚠️ exists structurally. **No acceptance-rate measurement** against a real MTP-bearing GGUF; P14 closing memo says throughput bench BLOCKED on B-W-1 (greedy-decode heisenbug) — status from 2026-04-30 unverified since |
| Serve dispatch | `src/serve/mod.rs:2810-2837` | ✅ wired. `HF2Q_SPEC_DECODE` env + auto-on when `model.mtp.is_some() && !sample_logits` |
| Stale guards | `src/arch/smoke.rs:479` sets `HF2Q_QWEN35_DROP_MTP=1` | ❌ no convert-side reader for that env var anywhere in current code; dead/superseded guard |
| DFlash | `src/inference/spec_decode/dflash/` | ⚠️ ADR-030 Phase 1-3 partial code exists; Phase 4 (orchestrator), Phase 5 (async), Phase 6 (rejection sampler) deferred or partial |
| Qwen3-VL MTP | `src/convert/arch/qwen3vl_text.rs:120` | ❌ explicit `mtp.*` drop ("out of v1"); not in this ADR's scope either |
| On-disk "MTP" GGUFs | `/opt/hf2q/models/qwen3.6-{27b,35b-a3b-abliterated}-mtp-q4_0/*.gguf` | ⚠️ produced by external converter, not hf2q. Not trusted as ground truth by operator |

**Codex verdict** (2026-05-19 21:36 UTC, verbatim from `/tmp/cfa-mtp-audit/codex-last.txt`): *"The central claim is supported for today's code: current hf2q convert has no dense qwen35 convert path and Qwen35MoE mapping cannot recognize or emit MTP tensors, so an HF checkpoint with MTP tensors will fail as unmapped rather than produce an MTP-bearing GGUF."*

### 1.3 Worked example — the `+1` offset disk/runtime contract

A representative example of "we are silent on a load-bearing convention; tests cannot tell":

```rust
// src/inference/models/qwen35/mtp.rs:181-189
let embed_norm = rms_norm_with_weight(
    &mut enc, registry, device,
    embed_t,            // [1, H], the just-accepted token embedding
    &self.enorm,        // the weight tensor loaded from blk.N.nextn.enorm.weight
    1, h, 1e-6,
)?;
```

`rms_norm_with_weight` (mtp.rs:472) calls `mlx_native::ops::rms_norm::dispatch_rms_norm`, which computes `out = w * rmsnorm(x)`. Stock-llama.cpp Qwen GGUFs have `+1` **baked into the weight at convert time** (`/opt/llama.cpp/conversion/qwen.py:303-304,597-618`: after MTP tensor remapping, any tensor whose name ends with `norm.weight` gets `data_torch = data_torch + 1`). So on a stock GGUF, the on-disk `enorm.weight` already equals `w_original + 1`, and the runtime's standard `w_baked * rmsnorm(x)` produces the mathematically-correct `(1 + w_original) * rmsnorm(x)`.

The disk/runtime contract is therefore: **convert-side bake; runtime-side standard kernel; no `+1` applied at runtime.**

The bug at HEAD `eab0220b` is two-fold:

1. **Convert side**: hf2q has no MTP convert path at all, so it cannot bake `+1`. (P2 fixes by adding the bake to the new convert mappers.)
2. **Runtime side**: hf2q's `forward_draft` correctly uses standard `dispatch_rms_norm` (so on a stock-baked GGUF, the math would be right). **No runtime change is needed.** The earlier draft of this ADR proposed a runtime `rms_norm_one_plus_w` kernel; codex's review (2026-05-19) caught that this would either (a) fail G1 byte-cmp if convert did NOT bake, or (b) double-apply if convert DID bake. The correct resolution per the pinned reference is convert-bake + standard runtime kernel.

The synthetic test `mtp_forward_draft_returns_logits` (`mtp_tests.rs:232-252`) cannot detect either form of disk/runtime mismatch because its zeros-input case produces zero output regardless. P1's numerical-parity harness, running against a real stock GGUF with non-zero calibration inputs, is what will catch any divergence.

**This is the canonical example of the bug class this ADR targets**: a load-bearing on-disk convention that is undocumented in our code, untested by our synthetic fixtures, and silently dependent on the upstream converter's behavior. P1's numerical-parity harness is the only systematic defense.

### 1.4 Target hardware + baselines

**Hardware**: M5 Max (operator's primary dev box). Apple Metal via mlx-native (path-pinned `Cargo.toml:46`).

**Today's no-spec-decode baseline** (`scripts/qwen35_bench.sh` 3-run median, HEAD `eab0220b`, `HF2Q_NO_FA=1` default):

| Model | `tg200` | `tg1500` / `tg2000` | Source |
|---|---|---|---|
| Qwen 3.6 35B-A3B APEX-Q5_K_M | 130.6 t/s | 129.1 t/s (tg1500) | README.md re-bench 2026-05-17 |
| Qwen 3.6 27B (no MTP, Q4_0 reference) | TBD | TBD | P0 will measure |
| Gemma 4 26B-A4B Q6_K | 105.2 t/s | 93.5 t/s (tg2000) | README.md re-bench 2026-05-17 |

**External reference points** (the perf bars to beat):

| Engine | Model | Hardware | Result | Source |
|---|---|---|---|---|
| MTPLX D3 native MTP | Qwen 3.6 27B-MTPLX-Optimized-Speed | M5 Max --max | **63.056 / 62.886 tok/s** vs **28.156 tok/s** no-MTP AR (2.24×) | /opt/MTPLX/README.md |
| llama.cpp MTP | Qwen 3.6 27B | RTX 3090 | 38 → 65 t/s (1.71×) | dredyson.com tutorial |
| vLLM MTP | Qwen 3.6 | various | ~80% acceptance, ~3× single-batch | vLLM docs |
| DFlash (Python MLX) | Gemma 4 26B-A4B | M5 Pro | TBD; ADR-030 §1.2 cited paper claim | /opt/dflash/README.md |
| SGLang MTP | DeepSeek V3 | MI300X | 1.7-2× | ROCm tutorial |

**MTPLX's 2.24× on M5 Max is the single Apple-Silicon native-MTP first-party number that exists.** Operator's perf bar: "as fast (or faster) than mtplx and llama.cpp, and obviously faster than non-MTP of hf2q". We translate that into P6's three hard gates.

---

## 2. Current state (Chesterton's fence)

Before scoping what to *add*, this section catalogues what hf2q already has so we touch only what's missing or wrong.

### 2.1 Existing scaffolding (keep, validate, don't rewrite)

| Component | File | LOC | Treatment |
|---|---|---|---|
| Arch catalogs (Qwen 3.5 dense + MoE; declare MTP templates) | `src/arch/entries/qwen35.rs`, `qwen35moe.rs` | — | KEEP. Templates are correct per the canonical `LLM_TENSOR_NEXTN_*` names. |
| `MtpWeights` GPU struct + sub-buffers | `src/inference/models/qwen35/mtp.rs:27-69` | — | KEEP shape. Re-validate via P3 numerical-parity harness. |
| GGUF→GPU loader (dedicated vs shared embed/head logic) | `src/inference/models/qwen35/mtp_weights_load.rs` | 345 | KEEP logic. P2 covers the case where convert emits both Qwen 3.5 (dedicated) and Qwen 3.6 (shared) GGUFs and the loader handles both. |
| `forward_draft` GPU dispatch (4 sub-steps) | `src/inference/models/qwen35/mtp.rs:95-169` | — | KEEP structure including standard `rms_norm` calls. P3 surfaces any non-`+1` numerical drift via the G2 parity harness; fix in-place, do not rewrite. |
| `HybridKvCache::mtp_slot` | `src/inference/models/qwen35/kv_cache.rs` | — | KEEP. Per-request alloc. ADR-017's persistent-cache extension lives outside this ADR. |
| Spec-decode greedy loop | `src/inference/models/qwen35/spec_decode.rs` | 824 | KEEP K=1 logic. P6 extends to K=2/3 sweep. |
| Generic spec-decode verifier (`accept_prefix`, `rollback_kv_state`) | `src/inference/spec_decode/verifier.rs` | 783 | KEEP. Already production-quality per ADR-029 Phase 1. |
| n-gram proposer (vLLM KMP port) | `src/inference/spec_decode/ngram_proposer.rs` | 323 | KEEP. Not used here but no reason to touch it. |
| DFlash scaffold (substantial; 7011 LOC) | `src/inference/spec_decode/dflash/` | 9 files: `config.rs` (370), `forward.rs` (2158), `hidden_capture.rs` (1085), `kv_cache.rs` (549), `mod.rs` (20), `orchestrator.rs` (1699), `rejection_sampler.rs` (372), `tensors.rs` (453), `weights.rs` (305) | KEEP. P4 is **audit + close gaps**, NOT write-from-scratch. Specifically: validate against /opt/dflash Python parity (G2); fix any kernel-precision bugs (Bug A/B class); confirm `rejection_sampler.rs` implements Leviathan-2023 correctly (used at P5 for sampled-path); confirm `hidden_capture.rs` captures at the right `target_layer_ids` for the live Qwen 3.6 27B/35B-A3B targets. |
| Serve dispatch (`HF2Q_SPEC_DECODE` env + auto-on rule) | `src/serve/mod.rs:2810-2837` | — | KEEP env + auto-on. P5 (sampled-path) drops the `!sample_logits` guard. |

### 2.2 What's missing or wrong (the actual surface for this ADR)

| Gap | Fix location | Phase |
|---|---|---|
| `ArchName::Qwen35` (dense/hybrid) enum variant + dispatch | `src/quantize/ggml_quants/tensor_ref.rs` + `src/convert/cli_driver.rs` | P2 |
| `src/convert/arch/qwen35.rs` — new file, dense convert mapper | new file | P2 |
| MTP arms in `qwen35moe::map_tensor_name` (HF `mtp.layers.{bid}.* → blk.N.nextn.*`) | `src/convert/arch/qwen35moe.rs` | P2 |
| `+1` offset on **all Qwen 3.5/3.6 `norm.weight` tensors except `ssm_norm.weight`** — **convert-time bake** (matches `/opt/llama.cpp/conversion/qwen.py:303-304`, applied arch-wide post-remap; see §3.5 full baked-tensor table); runtime kernel unchanged | new convert mappers + post-map data-transform hook in P2 | P2 |
| GGUF metadata: `qwen35.nextn_predict_layers` (dense) / `qwen35moe.nextn_predict_layers` (MoE) | `src/convert/arch/qwen35*.rs` metadata builders | P2 |
| `mtp_use_dedicated_embeddings` source — must come from HF config not be hardcoded | `Qwen35Config::from_gguf` + metadata round-trip | P2 |
| eh_proj split convention (HF `[2H, H]` interleaved → GGUF same shape; loader splits to `[H,H]` + `[H,H]`) — gate against llama.cpp's exact byte order | `mtp_weights_load.rs:load_split_eh_proj` + convert | P2 |
| Numerical-parity test harness (Python ref → tensor dump → hf2q forward → max-abs-diff) | new `tests/parity/mtp_python_ref.rs` + `scripts/mtp_parity.py` | P1 |
| Acceptance-rate telemetry + first-party measurement on M5 Max | `spec_decode.rs:SpecDecodeStats` already has fields; needs API surface + bench harness | P1 + P6 |
| Sampled-path support (Leviathan-2023 rejection sampler) | new module under `src/inference/spec_decode/` | P5 |
| Per-request HTTP knob (`spec_decode: on|off|auto`, `spec_decode_k: N`) | `src/serve/api/...` | P5 |
| DFlash forward parity-validate against /opt/dflash Python | EXTEND existing `src/inference/spec_decode/dflash/forward.rs` (2158 LOC) — audit, close gaps surfaced by parity diff | P4 |
| DFlash drafter loader for Gemma 4 (Qwen 3.6 loader already exists at `weights.rs`) | EXTEND existing `src/inference/spec_decode/dflash/weights.rs` (305 LOC) — verify target-parametric; add Gemma 4 drafter shape | P5 |
| DFlash orchestrator parity-validate (re-prefill, capture, accept-prefix, rollback) | EXTEND existing `src/inference/spec_decode/dflash/orchestrator.rs` (1699 LOC) — audit + close ADR-030 deferred Phase-4..6 gaps | P4 |
| SPEC-BENCH-style harness (uniform tok/s and acceptance reporting across hf2q-native / hf2q-MTP / hf2q-DFlash / llama.cpp-MTP / MTPLX / dflash-Python) | new `scripts/spec_bench.sh` | P1 |
| Stale `HF2Q_QWEN35_DROP_MTP` guard in `src/arch/smoke.rs:479` | delete (no convert-side reader) | P2 cleanup |

### 2.3 Doc cleanups required for ADR consistency

ADR-013 P14 status will change from "COMPLETE" to "scaffold-only, superseded by ADR-034". ADR-012 P11 status will change from "shipped 2026-04-24" to "intent documented; current code does not implement; superseded by ADR-034 P2". ADR-030 status will change from "proposed" to "subsumed by ADR-034 Workstream B".

These edits are part of P-1 and are non-negotiable: stale claims in prior ADRs caused the audit-blindspot that produced this ADR.

---

## 3. Decision

### 3.1 Scope — the 2×2 matrix

|  | Qwen 3.6 | Gemma 4 |
|---|---|---|
| **Native MTP** | **A** — in-checkpoint heads, byte-cmp gate vs llama.cpp PR #22673 converter. Both 27B dense-hybrid and 35B-A3B MoE in lockstep (single PR adds both convert mappers). | **D** — Google's `-assistant` drafter (separate model with built-in MTP-like structure). **Deferred to Phase 7** (follow-up), not v1. |
| **DFlash** | **B** — z-lab drafter on top of vanilla Qwen target. Port from `/opt/dflash/dflash/model_mlx.py` Python → mlx-native Rust. | **C** — same DFlash architecture, Gemma 4 target loader. Substantial mlx-native primitives reused from Workstream A's Qwen target serve path. |

Cells A, B, C are v1; cell D is Phase 7.

### 3.2 Correctness contracts (four layered gates)

Every shipped feature clears **all four** of the following before merge:

| Gate | What it proves | Harness | Where it applies |
|---|---|---|---|
| **G1 — Byte-cmp convert** | Our convert output is byte-equivalent (structurally) to the canonical reference for the same input | `tests/util/gguf_structural_diff` (defined in P2) of two GGUFs: `convert_hf_to_gguf.py` (stock llama.cpp post-#22673) vs `hf2q convert` on the same HF safetensors. The diff tool's tolerated-delta classes (4 enumerated in R1 §5) are the **contract**, not a judgment call. Method mirrors ADR-033 §P1 but with explicit structural classifier. | Workstream A (Qwen MTP). For DFlash there is no stock-converter equivalent; G1 is replaced by structural-shape gates against the downloaded z-lab safetensors (tensor count, shape, dtype, name set). |
| **G2 — Numerical parity (forward)** | Our `forward_draft` (or DFlash drafter forward) produces logits within ε of the Python reference on a calibration prompt | Python script consumes the SAME GGUF (or safetensors), runs HF transformers `model.mtp.forward(...)` (or `/opt/dflash/dflash/model_mlx.py` DFlash forward) at temp=0, dumps intermediate tensors at known checkpoints. Rust test loads same inputs, runs hf2q forward, asserts `max_abs_diff < ε` per sub-step. ε per-checkpoint ladder: enorm/hnorm/rms-norm sub-steps ε=1e-5 (BF16 noise floor); attn/ffn sub-steps ε=1e-3 (BF16 cumulative); final logits ε=1e-2 (full draft block). Tolerance ladder is defined in `mtp_parity.py` and committed to the repo (not a runtime knob). | All cells. |
| **G3 — Greedy byte-identical** | hf2q decode with spec-decode-on, greedy `temp=0`, produces byte-identical output to hf2q decode with spec-decode-off, greedy `temp=0`, on N≥10 deterministic prompts of length 2k tokens | New `tests/spec_decode_byte_identity_qwen36.sh` + same for gemma4. Operator-runs (real models). The CI proxy is `tests/spec_decode_byte_identity_synthetic.rs` (tiny shapes). G3 is **unconditional** — see §5 R5; if base decode is non-deterministic, Phase -2 closes that first. | All cells. |
| **G4 — Acceptance rate ≥ external reference** | Measured K=1 acceptance rate ≥ `max(public-reference floor, measured-vLLM-or-llama.cpp-floor-on-same-model-and-prompt-set)`. Qwen MTP public floor: 70 % per vLLM. DFlash public floor: 60 % per paper. The measured floor (from P0) is the gate when higher than the public number. | `scripts/spec_bench.sh` acceptance mode; same script that produces P6 perf numbers. Operator-runs. | All cells. |

If any single gate fails at merge time, the offending phase does not ship; it gets a P*x* designator (e.g. P3a for a first attempt that failed G2) and a new attempt under the same P-number.

**Sampled-path correctness (G5, separate)**: when `temp>0`, byte-identity is impossible; instead distribution-preservation per Leviathan-2023 is the gate, validated by KL ≤ 0.01 + log-prob ratio ∈ [0.98, 1.02] + top-50 5-gram Jaccard ≥ 0.95 (mirrors ADR-030 §3.6 row 3). This applies only to P5 (sampled-path) deliverables.

### 3.3 Performance contracts (three must-beats)

P6 ships if **and only if** all three are simultaneously green on M5 Max. (Gates renamed F1/F2/F3 to avoid collision with phase identifier P-1.)

| Gate | Reference | Target |
|---|---|---|
| **F1 — Beat MTPLX** | MTPLX D3 on `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`: 63.056 / 62.886 tok/s at `temp=0.6 top_p=0.95 top_k=20` | hf2q-MTP best-K decode `≥ 63.000 tok/s` on the same model at the same sampler params on the same machine, paired-run thermal-fair |
| **F2 — Match or beat llama.cpp Apple-Metal MTP** | llama.cpp HEAD `-fa 1 --spec-type draft-mtp --spec-draft-n-max 2` on Qwen 3.6 27B MTP GGUF, M5 Max; this is a **first-party number we produce in P0** because no public reference exists | hf2q-MTP best-K decode `≥` our locally-measured llama.cpp number, paired-run |
| **F3 — Beat hf2q non-MTP** | hf2q-native decode on the same target (Qwen 3.6 27B Q5_K_M-or-equivalent quant), `tg2000` floor from P0 | hf2q-MTP best-K decode `>` non-MTP baseline at `tg2000` |

For Gemma 4 DFlash (Workstream B / cell C), references are correspondingly: `/opt/dflash/dflash/model_mlx.py` Python on M5 Max (P0 measures); llama.cpp DFlash (does not exist upstream, skipped); hf2q-native gemma 4 decode (README.md baseline 105.2 / 93.5 t/s tg200/tg2000).

### 3.4 Module layout (proposed)

```
src/
├── arch/entries/
│   ├── qwen35.rs               (KEEP; declares has_mtp: true + nextn templates)
│   └── qwen35moe.rs            (KEEP)
├── convert/arch/
│   ├── qwen35.rs               (NEW — dense Qwen 3.5/3.6 convert mapper, includes MTP arms)
│   └── qwen35moe.rs            (EDIT — add MTP arms to map_tensor_name)
├── convert/cli_driver.rs       (EDIT — ArchName::Qwen35 enum variant + dispatch arm)
├── inference/models/qwen35/
│   ├── mtp.rs                  (UNCHANGED runtime — +1 offset is baked at convert; kernel stays standard)
│   ├── mtp_weights_load.rs     (KEEP; ensure shared-vs-dedicated logic still right after P2)
│   ├── mtp_tests.rs            (EDIT — replace zeros-input tests with real-fixture numerical-parity)
│   └── spec_decode.rs          (EDIT — extend to K=2/3; new sampled-path entry point in P5)
├── inference/spec_decode/
│   ├── dflash/                 (existing scaffold; 7011 LOC; see §2.1 inventory)
│   │   ├── config.rs           (KEEP, 370 LOC; verify per-target dispatch in P5)
│   │   ├── forward.rs          (EXTEND, 2158 LOC; P4a audits, P4b closes parity gaps)
│   │   ├── hidden_capture.rs   (EXTEND, 1085 LOC; P4d audits target_layer_ids; P5 adds Gemma 4 target hook — may refactor to per-target sub-modules, +200 LOC budget)
│   │   ├── kv_cache.rs         (KEEP, 549 LOC; verify rollback semantics in P4a)
│   │   ├── orchestrator.rs     (EXTEND, 1699 LOC; P4 closes ADR-030 deferred Phase 4-6 gaps)
│   │   ├── rejection_sampler.rs (KEEP, 372 LOC; P4c validates against Leviathan-2023 §3.5b spec)
│   │   ├── tensors.rs          (KEEP, 453 LOC)
│   │   └── weights.rs          (EXTEND, 305 LOC; P5 adds Gemma 4 drafter shape if not already parametric)
│   (note: parent-level `src/inference/spec_decode/rejection_sampler.rs` would be redundant — sampler lives inside `dflash/` and is shared with the MTP path via direct import; no new file)
├── serve/
│   ├── mod.rs                  (EDIT — drop !sample_logits guard at P5; add per-request knobs)
│   ├── api/                    (EDIT — chat-completions body schema: spec_decode, spec_decode_k, mtp_acceptance in usage)
│   └── spec_decode_cli.rs      (KEEP; extend DFlash dispatch to match orchestrator changes)
scripts/
├── spec_bench.sh               (NEW — SPEC-BENCH-style cross-engine harness)
├── mtp_parity.py               (NEW — Python reference forward, dumps intermediates)
└── adr034_gate.sh              (NEW — runs G1+G2+G3+G4 in sequence; CI hook)
tests/
├── parity/
│   ├── mtp_python_ref.rs       (NEW — consumes mtp_parity.py output; asserts max-abs-diff)
│   └── dflash_python_ref.rs    (NEW — same for /opt/dflash)
└── spec_decode_byte_identity_*.sh (NEW — G3 gates)
docs/
└── ADR-034-real-model-findings/(NEW — like adr-033-real-model-findings/, lands per-phase findings)
```

### 3.5 Tensor naming + the `+1` offset contract (resolution)

**HF source name convention** (verified against `/opt/llama.cpp/conversion/qwen.py:560-618` and `/opt/llama.cpp/gguf-py/gguf/tensor_mapping.py:2248-2269`):

Qwen 3.5/3.6 MTP safetensors carry the MTP block under the `mtp.layers.{bid}.*` prefix (NOT `model.mtp.layers.0.*`). The stock converter:

1. Sees `mtp.*` tensors during the streaming load and remaps `mtp.layers.{bid}` → `model.layers.{bid + num_hidden_layers}` (`qwen.py:597-618`) — i.e., the MTP block is renumbered to live at virtual block index `N = num_hidden_layers` of the main stack.
2. The general `TensorNameMap` (`tensor_mapping.py:2248-2269`) then translates `model.layers.{N}.enorm.weight` → `blk.{N}.nextn.enorm.weight`, etc.
3. Norm-baking (`qwen.py:303-304`) runs on the post-remap names: any tensor whose source name ends with `norm.weight` gets `data_torch = data_torch + 1` before quantization.

The net HF → GGUF table (citing both the converter remap step and the tensor_mapping step):

| HF source tensor (in safetensors) | GGUF tensor | Notes |
|---|---|---|
| `mtp.layers.0.embed_tokens.weight` | `blk.{N}.nextn.embed_tokens.weight` | N = `num_hidden_layers` (64 for Qwen 3.6 27B dense; 40 for Qwen 3.6 35B-A3B MoE). Present only when `mtp_use_dedicated_embeddings == True` (Qwen 3.5 convention); skipped otherwise (Qwen 3.6 convention — share main `token_embd`). |
| `mtp.layers.0.enorm.weight` | `blk.{N}.nextn.enorm.weight` | RMSNorm weight; **convert-side `+1` bake applied** per `qwen.py:303-304`. |
| `mtp.layers.0.hnorm.weight` | `blk.{N}.nextn.hnorm.weight` | RMSNorm weight; **convert-side `+1` bake applied**. |
| `mtp.layers.0.eh_proj.weight` | `blk.{N}.nextn.eh_proj.weight` | Shape `[2H, H]` interleaved (embed first H rows, hidden second H rows; loader splits into `[H,H]` halves). NO `+1` (not a norm). |
| `mtp.layers.0.shared_head.norm.weight` | `blk.{N}.nextn.shared_head_norm.weight` | RMSNorm weight; **convert-side `+1` bake applied** (`shared_head.norm.weight` matches `endswith('norm.weight')`). |
| `mtp.layers.0.shared_head.head.weight` | `blk.{N}.nextn.shared_head_head.weight` | LM-head projection; present only when `mtp_use_dedicated_embeddings == True`; skipped otherwise (share main `output.weight`). NO `+1`. |
| `mtp.layers.0.input_layernorm.weight` | `blk.{N}.attn_norm.weight` | Inner transformer block — uses normal `blk.{N}.*` names. **✅ `+1` baked**: after the `mtp.layers.0` → `model.layers.{N}` remap (`qwen.py:597-618`), the post-remap source name `model.layers.{N}.input_layernorm.weight` ends with `norm.weight` and does NOT end with `linear_attn.norm.weight`, so `qwen.py:303-304` applies. |
| `mtp.layers.0.post_attention_layernorm.weight` | `blk.{N}.post_attention_norm.weight` | **✅ `+1` baked** (same reason as above; post-remap name matches the rule). |
| `mtp.layers.0.self_attn.q_proj.weight` | `blk.{N}.attn_q.weight` | — |
| `mtp.layers.0.self_attn.k_proj.weight` | `blk.{N}.attn_k.weight` | — |
| `mtp.layers.0.self_attn.v_proj.weight` | `blk.{N}.attn_v.weight` | — |
| `mtp.layers.0.self_attn.o_proj.weight` | `blk.{N}.attn_output.weight` | — |
| `mtp.layers.0.self_attn.q_norm.weight` | `blk.{N}.attn_q_norm.weight` | Per-head Q norm (Qwen3 quirk). **✅ `+1` baked** (post-remap matches rule). |
| `mtp.layers.0.self_attn.k_norm.weight` | `blk.{N}.attn_k_norm.weight` | Per-head K norm. **✅ `+1` baked** (post-remap matches rule). |
| `mtp.layers.0.mlp.gate_proj.weight` (dense MTP variant) | `blk.{N}.ffn_gate.weight` | Dense path. **CORRECTION 2026-05-21**: ADR-034's original claim "Inner FFN is dense even for MoE-A3B targets" was **falsified empirically** during prep deep-research — see MoE row below. |
| `mtp.layers.0.mlp.experts.{E}.{gate,up,down}_proj.weight` (MoE MTP — Qwen 3.5/3.6 35B-A3B) | `blk.{N}.{ffn_gate_exps,ffn_up_exps,ffn_down_exps}.weight` + `blk.{N}.ffn_gate_inp.weight` + `blk.{N}.ffn_{gate,up,down}_shexp.weight` + `blk.{N}.ffn_gate_inp_shexp.weight` | **Inner FFN matches main-stack FFN topology** — MoE for MoE-A3B targets, dense for dense targets. Verified: `/opt/hf2q/models/Qwen-Qwen3.5-35B-A3B/model.safetensors.index.json` has 773 `mtp.layers.0.mlp.experts.*` tensors; canonical Q4_K_M output emits 16 MoE-style tensors at `blk.40.*`. The current hf2q MTP loader (`src/inference/models/qwen35/mtp_weights_load.rs:268-291`) hardcodes the dense path → **fails to load canonical MoE-MTP GGUFs** (typed error: `tensor 'blk.40.ffn_gate.weight' not found`). P3 must fix this. See [[project_adr034_mtp_loader_moe_bug_2026_05_21]]. |
| `mtp.layers.0.mlp.up_proj.weight` (dense path only) | `blk.{N}.ffn_up.weight` | — |
| `mtp.layers.0.mlp.down_proj.weight` (dense path only) | `blk.{N}.ffn_down.weight` | — |
| (Qwen 3.5 `attn_output_gate=true` variant) `mtp.layers.0.self_attn.gate_proj.weight` | `blk.{N}.attn_gate.weight` | Output-gate path; matched per `attn_output_gate` flag in config. |

**GGUF metadata keys** (re-derived from `/opt/llama.cpp/src/llama-arch.cpp:194,448-453`):

- `qwen35.nextn_predict_layers = 1` (Qwen 3.5 dense)
- `qwen35moe.nextn_predict_layers = 1` (Qwen 3.5/3.6 MoE)
- `qwen35[moe].block_count = num_hidden_layers + 1` (block count INCLUDES the appended MTP block — load-bearing for layer enumeration; smoke tests verify)

**The `+1` offset resolution (LOCKED — corrected per codex round-2 review 2026-05-19; broadened to mirror upstream exactly)**: applied **at convert time**, baked into the weight before quantization, matching `/opt/llama.cpp/conversion/qwen.py:303-304`. Runtime uses the **standard** `dispatch_rms_norm` kernel — no kernel variant needed.

**The bake rule** (verbatim from `qwen.py:303-304`, ported to Rust): for every Qwen 3.5/3.6 tensor whose **post-remap GGUF name ends with `norm.weight`** AND does **NOT** end with `linear_attn.norm.weight`, add `+1` to the F32 data before quantization. The post-remap names that satisfy this rule (and therefore get baked) are:

| GGUF tensor name (post-remap) | Source HF name | Baked? |
|---|---|---|
| `output_norm.weight` | `model.norm.weight` | ✅ |
| `blk.{L}.attn_norm.weight` (all L incl. MTP block) | `model.layers.{L}.input_layernorm.weight` OR `mtp.layers.0.input_layernorm.weight` | ✅ |
| `blk.{L}.post_attention_norm.weight` (all L incl. MTP block) | `model.layers.{L}.post_attention_layernorm.weight` OR equivalent under `mtp.*` | ✅ |
| `blk.{L}.attn_q_norm.weight` (all L incl. MTP block) | `model.layers.{L}.self_attn.q_norm.weight` OR equivalent under `mtp.*` | ✅ |
| `blk.{L}.attn_k_norm.weight` (all L incl. MTP block) | `model.layers.{L}.self_attn.k_norm.weight` OR equivalent under `mtp.*` | ✅ |
| `blk.{N}.nextn.enorm.weight` | `mtp.layers.0.enorm.weight` → remapped → matches `norm.weight` suffix | ✅ |
| `blk.{N}.nextn.hnorm.weight` | `mtp.layers.0.hnorm.weight` → remapped → matches | ✅ |
| `blk.{N}.nextn.shared_head_norm.weight` | `mtp.layers.0.shared_head.norm.weight` → remapped → matches | ✅ |
| `blk.{L}.ssm_norm.weight` (linear-attn layers, hybrid variant) | `model.layers.{L}.linear_attn.norm.weight` | ❌ excluded (`linear_attn.norm.weight` exclusion) |

Note: in stock convert, the MTP remap (`qwen.py:597-618`) renames `mtp.layers.{bid}` → `model.layers.{bid + n_layer}` BEFORE the bake check runs, so the MTP block's inner norms (`attn_norm`, `post_attention_norm`, `attn_q_norm`, `attn_k_norm`) ALL go through the same baked path as the main stack — same `+1` applied. This is broader than the round-1 draft of this ADR claimed; round-2 codex review caught the under-narrowing.

P2 implementation: extend the convert orchestrator's tensor-staging step with a per-arch `post_map_data_transform(arch, gguf_name) -> Option<fn(&mut Vec<f32>)>` hook. For `ArchName::Qwen35` and `ArchName::Qwen35Moe`, the hook returns `Some(add_one)` when `gguf_name.ends_with("norm.weight") && !gguf_name.ends_with("ssm_norm.weight")` (note: hf2q's GGUF name for `linear_attn.norm.weight` is `ssm_norm.weight` per `src/arch/entries/qwen35moe.rs:140-145`; the exclusion key is the GGUF name, not the HF name). The transform runs on the F32 tensor data after dequantization-from-source-dtype and before passing to the quantizer. Concrete plug-in point: `src/convert/cli_driver.rs:~1240` where `plan_steps.push(PlanStep { ... })` is called, attach the transform to the plan step. `mtp.rs:472 rms_norm_with_weight` and the mlx-native kernels are unchanged.

Rationale:
1. The pinned llama.cpp converter bakes `+1` at convert; not baking would break G1 byte-cmp at the very first phase that runs the gate.
2. The runtime `dispatch_rms_norm` kernel applied to `w_baked = w_original + 1` correctly produces `(1 + w_original) * rmsnorm(x)`. No kernel change required.
3. Applying `+1` BOTH at convert AND at runtime would double-apply and break logits. The contract is "disk has baked weight; runtime treats baked weight as ordinary".

**Scope note (implications beyond MTP)**: because the bake rule is shared across all Qwen 3.5/3.6 norm tensors, P2 effectively validates and ships the **first byte-cmp-gated Qwen 3.5/3.6 convert path in hf2q's history**, not just MTP. Existing on-disk apex/dwq Qwen GGUFs (used by hf2q runtime today) are all externally produced; their runtime correctness has been an unverified assumption. P2's G1 gate closes that gap for any future Qwen 3.5/3.6 convert work too.

### 3.5b Leviathan-2023 rejection sampler (the G5 math)

For `temp > 0`, distribution-preserving speculative decoding requires Leviathan-2023 exact rejection sampling. The math, in 7 lines (full proof: arxiv 2211.17192 §2):

Given:
- `p_target(t | context)`: target-model probability of token `t` at the current position (post-temperature, post-top_k/top_p)
- `p_draft(t | context)`: drafter probability of token `t` at the same position (independent sampler params allowed)
- `draft_sample t_d`: the drafter's sampled token at this position

```
1. Sample u ~ Uniform(0, 1)
2. If u ≤ min(1, p_target(t_d) / p_draft(t_d)):  ACCEPT t_d
3. Else:
4.   Compute residual distribution: q(t) = max(0, p_target(t) - p_draft(t))
5.   Normalize: q(t) /= sum(q)
6.   Sample t_r ~ q(t):  REJECT t_d and EMIT t_r
7. After accept-prefix break, the verifier's own argmax at the first reject position is the residual-sample t_r (Step 4-6 collapses to a single tensor op given full q-distribution).
```

Two invariants must hold per the proof:
- **Distribution preservation**: `Pr(emit t) == p_target(t)` exactly, regardless of drafter quality (drafter only affects expected speedup, not output distribution).
- **Greedy reduction**: at `temp == 0` with `top_k == 1` and `top_p == 1.0`, both distributions collapse to point-mass on argmax; the accept rule reduces to "argmax matches → accept". G3 (byte-identical greedy) is then mathematically forced.

Existing implementation at `src/inference/spec_decode/dflash/rejection_sampler.rs` (372 LOC) is validated against this spec in P4c. The G5 gate (§3.2) is: 100-prompt sample at `temp ∈ {0.6, 0.95}` × `top_p ∈ {0.95, 1.0}` × `top_k ∈ {20, 50}`, comparing the empirical output-token distribution (spec-decode on vs off) via KL ≤ 0.01, log-prob ratio ∈ [0.98, 1.02], top-50 5-gram Jaccard ≥ 0.95.

### 3.6 Sampling-regime contracts

| Mode | Contract | Validation |
|---|---|---|
| `temp == 0` (greedy) | Byte-identical to spec-decode-off greedy on the same prompt and seed | G3 gate per cell |
| `0 < temp ≤ 0.99` (sampled) | Distribution-preserved via Leviathan-2023 exact rejection sampling | G5 gate (P5 only) |
| `temp == 0` with `top_k > 1` or `top_p < 1` | This is a contradictory request (greedy + multi-candidate constraint). **Explicit opt-in** (`HF2Q_SPEC_DECODE=1` or `spec_decode: on`) → typed `InvalidSamplerForSpecDecode` error, request aborts. **Auto mode** → route through the sampler-mode path (rejection sampler), no fallback warning needed. | New `sampler_mode_test.rs` covers both branches |

EOS handling: spec-decode must stop at the first EOS in the accepted prefix, not after K tokens. New `eos_handling_test.rs` regression-pins.

Sliding-window: KV rollback must correctly trim sliding-window AND full-attention layers. Reuses ADR-029 `rollback_kv_state` math (verifier.rs:783).

### 3.7 ADR supersession (canonical authority)

- ADR-012 §11/§15 — MTP intent documented; this ADR re-derives + lands the actual code under byte-cmp gates. Prior status flags ("shipped 2026-04-24") are reclassified as "documented intent; not in current code".
- ADR-013 §15 + P10 + P14 — load-only + execution status claims are reclassified as "scaffold-only; never validated against external reference; never benchmarked". ADR-013 acceptance criteria are absorbed into this ADR's G1-G4 gates with tighter teeth.
- ADR-028 §iter-152 — MTP K=3 audit; promoted from "OUT-OF-SCOPE for ADR-028" to P6 sweep target.
- ADR-029 Phase 1 (n-gram proposer) — untouched. Lives alongside this ADR's MTP/DFlash work as a third spec-decode mechanism, unchanged.
- ADR-030 (DFlash, status: proposed) — fully absorbed as Workstream B. ADR-030's downloaded drafter shapes (3.0), algorithm spec (3.1), and Python reference path (`/opt/dflash`) are inputs to P4/P5 here. ADR-030 will be marked "subsumed by ADR-034".

---

## 4. Plan — 8 phases (Phase -2 prereq + P-1..P6 + Phase 7 follow-up for cell D)

Each phase ships compiled, tested, gated. Phases run sequentially; no phase begins until the prior phase's gates all green. Estimated LOC are upper bounds.

### Phase -2 — Close B-W-1 greedy-decode heisenbug (mandatory prereq) (~variable; ADR-015 iter61a-4 scope)

**Scope**: This ADR's G3 contract (greedy byte-identical) is unconditional and requires deterministic base decode. If the B-W-1 heisenbug (ADR-015 iter61a-3 localized to FullAttn layer 3 prefill, status 2026-04-30: open) is still open at the time ADR-034 starts, closing it is the **first action**. No ADR-034 P-1 work begins until determinism is restored.

**Acceptance**: `scripts/coherence-harness/determinism_check.sh` passes on the Qwen 3.6 35B-A3B APEX-Q5_K_M and Gemma 4 26B-A4B Q6_K targets for N≥10 runs on the same deterministic prompt; byte-identical output across all runs.

**Skip condition**: if `determinism_check.sh` passes today (operator-verified before P-1 starts), Phase -2 closes immediately as already-satisfied; ADR-015's iter61a-3 memo gets a closing addendum.

### P-1 — Preflight, supersession, external pins (~50 LOC, mostly docs)

**Scope**:
- Update ADR-012, ADR-013, ADR-030 status fields per §3.7 (3 file edits, status-line changes only).
- Re-verify the external SHAs in the ADR-034 header `External pins` block (captured at draft time 2026-05-19) against the operator's live `/opt/llama.cpp`, `/opt/dflash`, `/opt/MTPLX` HEADs; refresh if drift detected. Commit a vendored snapshot of `convert_hf_to_gguf.py` from llama.cpp HEAD post-#22673 under `vendor/llama.cpp/convert_hf_to_gguf.py` for byte-cmp reproducibility.
- Delete the stale `HF2Q_QWEN35_DROP_MTP=1` guard at `src/arch/smoke.rs:479` (no convert-side reader anywhere).
- Add a one-line entry to README's Status row noting MTP as "in progress under ADR-034".

**Acceptance**: prior ADRs' status fields match current code; vendored converter present; smoke tests still pass after DROP_MTP removal; README accurate.

### P0 — Reference acquisition (~0 LOC; data + measurement)

**Scope**:
- Download all reference artifacts to `/Volumes/Extreme Pro/hf2q-models/`:
  - `Qwen/Qwen3.6-27B` (official HF safetensors, target)
  - `Qwen/Qwen3.6-35B-A3B` (official HF safetensors, target)
  - `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B-it` (official HF safetensors, target)
  - `froggeric/Qwen3.6-27B-MTP-GGUF`, `unsloth/Qwen3.6-27B-MTP-GGUF`, `RDson/Qwen3.6-27B-MTP-Q4_K_M-GGUF` (reference Qwen MTP GGUFs)
  - `havenoammo/Qwen3.6-35B-A3B-MTP-GGUF` (reference Qwen MoE MTP GGUF)
  - `z-lab/Qwen3.6-27B-DFlash`, `z-lab/Qwen3.6-35B-A3B-DFlash`, `z-lab/gemma-4-26B-A4B-it-DFlash`, `z-lab/gemma-4-31B-it-DFlash` (DFlash drafters)
  - `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed` (MTPLX perf reference target)
- Build llama.cpp HEAD locally (`cmake -B build -DGGML_METAL=ON && cmake --build build`).
- Run stock llama.cpp converter on `Qwen/Qwen3.6-27B` and `Qwen/Qwen3.6-35B-A3B` → produce reference GGUFs.
- Run `cmp` against froggeric/unsloth/havenoammo published GGUFs. **If they don't byte-match**, file upstream issue and proceed with our locally-converted reference (the upstream-publisher gate is a free correctness signal; we don't chase a phantom bug if their files have known drift).
- Run `/opt/MTPLX` `mtplx bench tune` against `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed` on M5 Max; record D0/D1/D2/D3 numbers. This is our **F1** reference baseline (§3.3).
- Run `/opt/dflash/dflash/model_mlx.py` benchmark on Gemma 4 26B-A4B with `z-lab/gemma-4-26B-A4B-it-DFlash`; record t/s + acceptance. This is our Workstream B (cell C) DFlash reference.
- Build llama.cpp `--spec-type draft-mtp` invocation; run on `froggeric/Qwen3.6-27B-MTP-GGUF` for `tg200`, `tg2000` on M5 Max; record. This is the no-public-Apple-Metal-number-exists baseline we produce ourselves.

**Acceptance**: all references downloaded; locally-converted GGUFs byte-equal to at least one published reference per family; all four reference perf numbers logged to `docs/ADR-034-real-model-findings/2026-XX-XX-p0-references.md`.

### P1 — Shared parity + bench harness (~600 LOC across Rust + scripts)

**Scope**:
- `scripts/mtp_parity.py` — Python reference forward. **HF transformers does NOT expose the MTP forward natively** for `Qwen3_5ForConditionalGeneration` (the MTP block is a custom HF extension in the model card's `modeling_qwen3_5.py`). Approach: import the model's `trust_remote_code=True` custom module from `Qwen/Qwen3.6-27B`'s repo, then call `model.model.mtp.layers[0](hidden, embed, ...)` directly. Wrapper script at `scripts/mtp_python_ref_wrapper.py` does this; it loads any of (target HF safetensors + reference MTP GGUF — via `gguf` Python package — + DFlash drafter), runs forward through the MTP block (or DFlash drafter via `scripts/dflash_dump_wrapper.py`) at temp=0 on a calibration prompt (`docs/data/calibration_cdv3.txt` excerpt — same one ADR-033 §Pi uses). Dumps intermediate tensors at named checkpoints (`enorm_out`, `hnorm_out`, `eh_proj_out`, `attn_out`, `ffn_out`, `shared_head_out`) to a side-cart `.npz`. **Read-only against /opt/dflash and /opt/llama.cpp**: never edits them. If dump hooks are needed inside /opt/dflash's `model_mlx.py`, they live in a wrapper that imports and monkey-patches at runtime, not as a patch to /opt/dflash itself.
- `tests/parity/mtp_python_ref.rs` — Rust test. Loads same GGUF + same calibration prompt; runs hf2q `forward_draft`; asserts max-abs-diff at each named checkpoint < ε per the **committed** tolerance ladder in `mtp_parity.py` (§3.2 G2 row). ε is NOT a runtime knob; changes to the ladder require a committed update to `mtp_parity.py` with rationale in `docs/ADR-034-real-model-findings/`. Default ladder: enorm/hnorm/rms-norm sub-steps ε=1e-5 (BF16 noise floor); attn/ffn sub-steps ε=1e-3 (BF16 cumulative); final logits ε=1e-2 (full draft block).
- `tests/parity/dflash_python_ref.rs` — same structure for DFlash; consumes `scripts/dflash_dump_wrapper.py` output.
- `scripts/spec_bench.sh` — SPEC-BENCH-style harness. Args: `--engine {hf2q-native, hf2q-mtp, hf2q-dflash, llama-mtp, mtplx, dflash-py}`, `--model <path>`, `--target-tokens N`, `--temp T`, `--paired-baseline <engine>`. Output: median t/s, σ, acceptance rate, wall-clock, thermal-fair flag. 3-run minimum.

  **Example invocations** (each engine wraps a different external CLI; the harness translates the uniform args into per-engine flags):

  ```bash
  # F1: hf2q-MTP vs MTPLX D3 paired on M5 Max
  scripts/spec_bench.sh --engine hf2q-mtp \
    --model /Volumes/Extreme\ Pro/hf2q-models/Qwen3.6-27B-MTPLX-Optimized-Speed \
    --target-tokens 2000 --temp 0.6 --top-p 0.95 --top-k 20 \
    --paired-baseline mtplx --paired-args '--depth=D3'

  # F2: hf2q-MTP vs llama.cpp on froggeric GGUF
  scripts/spec_bench.sh --engine hf2q-mtp \
    --model /Volumes/Extreme\ Pro/hf2q-models/froggeric_Qwen3.6-27B-MTP-GGUF/q4_k_m.gguf \
    --target-tokens 2000 --temp 0 \
    --paired-baseline llama-mtp --paired-args '-fa 1 --spec-type draft-mtp --spec-draft-n-max 2 -ngl 999'

  # F3: hf2q-MTP vs hf2q-native (non-spec) on same GGUF
  scripts/spec_bench.sh --engine hf2q-mtp \
    --model <path> --target-tokens 2000 --temp 0 \
    --paired-baseline hf2q-native

  # DFlash branch: hf2q-DFlash vs /opt/dflash Python on Gemma 4
  scripts/spec_bench.sh --engine hf2q-dflash \
    --model /Volumes/Extreme\ Pro/hf2q-models/google_gemma-4-26B-A4B-it/ \
    --draft-model /Volumes/Extreme\ Pro/hf2q-models/z-lab_gemma-4-26B-A4B-it-DFlash/ \
    --target-tokens 2000 --temp 0 \
    --paired-baseline dflash-py
  ```

  Per-engine adapter shims live in `scripts/spec_bench/adapters/{hf2q,llama,mtplx,dflash_py}.sh` — each takes the uniform args, translates to that engine's CLI, runs, parses output for t/s + acceptance, writes back a uniform JSON line.
- `scripts/adr034_gate.sh` — runs G1+G2+G3+G4 in sequence on a specified target; CI hook for the **synthetic** subset; operator-runs the **real-model** subset.
- `tests/spec_decode_byte_identity_qwen36.sh` — G3 gate for Qwen; operator-only (real models).
- `tests/spec_decode_byte_identity_gemma4.sh` — G3 gate for Gemma 4 (DFlash path); operator-only.

**Acceptance**:
- **Synthetic path (CI-runnable)**: harness end-to-end on a tiny synthetic Qwen 3.6 27B-MTP-shaped GGUF (hidden=64, vocab=64, 2 layers + 1 MTP block, weights with known `enorm = [...]` so the `+1` bake invariant is testable). All harness scripts return exit 0; the Rust parity tests pass with `ε ≤ 1e-6` on this synthetic where BF16 envelope doesn't apply.
- **Real-model path (operator-only)**: same harness against `froggeric/Qwen3.6-27B-MTP-GGUF`. Produces diagnostic dumps and tolerance numbers — **diagnostic only, NOT gating** at P1. The parity-pass gate is at P3 (which assumes P2 has landed convert + `+1` bake). P1's job is to prove the harness itself works; P3's job is to assert parity-passes-with-ε on real GGUFs.

### P2 — Qwen MTP convert (both 27B dense + 35B-A3B MoE in lockstep) (~700 LOC; smaller than original estimate because runtime kernel edit removed per codex review)

**Reference-acquisition update 2026-05-19** (post-audit of locally-available safetensors at HEAD `bc04f3b8`): the operator's locally-downloaded Qwen 3.5/3.6 safetensors corpus is more constrained than the original P0 scope assumed:
- `/opt/hf2q/models/Qwen-Qwen3.5-35B-A3B` (the only safetensors download) is the **MULTIMODAL VLM** variant: 1,811 tensor patterns = 785 `mtp.*` + ~1,000 `model.language_model.*` + ~26 `model.visual.blocks.*`. Architecture: `Qwen3_5MoeForConditionalGeneration`. Tensor prefix is `model.language_model.*` (multimodal wrapping), NOT `model.*`. Experts are **already FUSED** in the safetensors (`mlp.experts.down_proj` and `mlp.experts.gate_up_proj` are single tensors per layer, NOT per-expert) — this is a different on-disk layout vs the Qwen 3.6 27B/35B-A3B text-only variants the original P2 scope assumed. Has `shared_expert.*` tensors (Qwen3MoE dropped shared experts; Qwen3.5MoE re-adds them).
- Other locally-available Qwen 3.5/3.6 models (`qwen3.6-35b-a3b-4bit-DWQ`, `qwen3.6-35b-a3b-abliterix-ega-abliterated-apex`, `qwen3.6-35b-a3b-abliterix-ega-abliterated-q4_0-flat`) are **GGUF-only** (no safetensors) — they can be runtime targets but cannot be P2 byte-cmp gate sources.

**Implication**: P2 byte-cmp G1 needs EITHER (a) operator downloads a text-only Qwen 3.6 27B or 35B-A3B safetensors (e.g. `Qwen/Qwen3.6-27B` or `Qwen/Qwen3.6-35B-A3B` non-VLM HF releases) to match the original P2 scope, OR (b) P2 scope expands to cover the multimodal `Qwen3_5MoeForConditionalGeneration` variant with the `model.language_model.*` prefix and pre-fused expert layout. Option (b) adds ~200 LOC for the prefix-stripping path, ~100 LOC for the pre-fused expert handling (no fusion-at-convert needed; just one tensor → one GGUF tensor), plus the vision-encoder filter (drop `model.visual.*` or wire to a separate `mmproj` GGUF — text-only convert can simply drop). Operator decision required at P0 boundary.

**Major audit-correction 2026-05-19 (commit `a69cd116`)**: the original ADR-034 audit at HEAD `eab0220b` found "no `src/convert/arch/qwen35.rs`; `qwen35moe::map_tensor_name` has zero MTP arms" and concluded P2 was net-new ~700 LOC. **That audit missed `/opt/hf2q/src/models/qwen35/`** — the ADR-012 conversion-side module (`mod.rs` 3,425 LOC + `moe.rs` 2,194 LOC + `dense.rs` 516 LOC = **6,135 LOC**) that **already fully implements** every transform P2 was scoping: MTP rename (`rename_mtp_tensors_to_layer_form` + lazy variant), norm+1 bake (`apply_rms_norm_plus_one_in_lazy_map`), V-head grouped→tiled reorder (`reorder_v_heads` + inverse), 6-case linear_attn dispatch (`transform_linear_attn_tensor`), QKVZ in_proj split (`transform_in_proj_qkvz`), per-expert merge (`merge_expert_tensors` + lazy/in-place variants), pre-fused gate_up_proj split (`split_and_rename_fused_gate_up_in_lazy_map`), HF→GGUF MoE name map (`hf_tensor_name_to_gguf_moe`), shared/full/linear-attn suffix mappers, and `emit_metadata_moe`. **All transforms pass unit tests** (visible in the full-suite 3,226 / 0 / 49 count) including dedicated byte-identical lazy-vs-eager equivalence tests. The orphan is disconnected from the convert orchestrator because ADR-033 §P0 chose a new `src/convert/arch/` directory with a different IR (`MetaValue` + `MappedTensor`/`ExpertGroup`) vs the orphan's `crate::ir::{TensorRef, LazyTensorMap}` types. **P2 revised plan**: mine the orphan as source-of-truth for transform semantics (saving the deep-research time of re-deriving from canonical Python at `qwen.py:296-360, 522-628`), port the algorithms into `src/convert/arch/qwen35moe.rs` (or a new `qwen35moe_vlm.rs` sibling) using the new IR, delete the orphan per [[feedback-no-backwards-compat-2026-05-18]] in the same commit-series. Net new LOC drops from ~700 to ~200-300 (mostly IR-adapter glue + the multimodal-VLM-specific paths). Tracked in [[project-adr012-orphaned-convert-code-2026-05-19]].

**Scope**:
- Add `ArchName::Qwen35` variant + dispatch in `src/quantize/ggml_quants/tensor_ref.rs:38` and `src/convert/cli_driver.rs`.
- New file `src/convert/arch/qwen35.rs` — dense Qwen 3.5/3.6 convert mapper. Copy-edit the existing pattern from `qwen35moe.rs`; remove MoE expert-fusion arms; add MTP arms per §3.5 table.
- Edit `src/convert/arch/qwen35moe.rs::map_tensor_name` to add MTP arms per §3.5 table (under a new `// MTP block — see ADR-034 §3.5` comment block). **Source name prefix is `mtp.layers.{bid}.*`**, NOT `model.mtp.layers.0.*`. Mapper remaps `bid=0` → GGUF block index `N = num_hidden_layers`.
- **Convert-side `+1` bake** (replaces the deleted runtime kernel edit; **arch-wide, NOT MTP-only** per §3.5 corrected table): for any Qwen 3.5/3.6 tensor whose final post-remap **GGUF name ends with `norm.weight`** AND does NOT end with `ssm_norm.weight` (hf2q's GGUF name for `linear_attn.norm.weight` — see `src/arch/entries/qwen35moe.rs:140-145`), apply `data_torch = data_torch + 1` to the F32 weights *before* passing to the quantizer. This applies to ALL such tensors regardless of whether the HF source was `mtp.*` or `model.layers.{L}.*` — both go through the same `_LinearAttentionVReorderBase.modify_tensors` bake step in `/opt/llama.cpp/conversion/qwen.py:303-304` post-MTP-remap. The mlx-native rms_norm kernel is **unchanged** (runtime uses standard `dispatch_rms_norm` on the already-baked weight).
- Edit `Qwen35Config::from_gguf` at `src/inference/models/qwen35/mod.rs:408-442` (the existing `mtp_num_hidden_layers` + `mtp_use_dedicated_embeddings` reader block):
  - Set `mtp_use_dedicated_embeddings` from presence of `blk.{N}.nextn.embed_tokens` tensor (derived flag — no separate metadata key needed; mirrors the existing loader contract at `mtp_weights_load.rs:53-87`).
  - Set `mtp_num_hidden_layers` from `qwen35[moe].nextn_predict_layers` metadata key (already at `mod.rs:408-411`; just verify it stays correct after MTP-bearing GGUF emission).
- Edit `mtp_tests.rs`:
  - Delete `mtp_forward_draft_returns_logits` (the all-zeros test that cannot distinguish baked vs non-baked weights).
  - Add `mtp_loads_real_gguf_27b_froggeric` — loads downloaded `froggeric/Qwen3.6-27B-MTP-GGUF`, asserts all expected tensors present with correct shapes. **Operator-only test** (gated behind `HF2Q_REAL_MODEL_TESTS=1` env; not in default CI due to 17 GB asset).
  - Add `mtp_byte_cmp_convert_against_llama_cpp_qwen36_27b` — runs `hf2q convert` + stock llama.cpp `convert_hf_to_gguf.py` on same official safetensors; asserts byte-identical output OR differs only by structural-equivalent header KV reordering (defined: same set of KV keys, same values, possibly different emission order per `gguf-py` v3 unordered-map property). **Operator-only test** (gated; 48 GB intermediate). Adds a new helper `tests/util/gguf_structural_diff.rs` (~100 LOC) for the tolerated-delta classifier.
  - Add `mtp_byte_cmp_convert_against_llama_cpp_qwen36_35b_a3b` — same for 35B MoE. Operator-only.
  - Add `mtp_one_plus_w_bake_synthetic` — **CI-runnable** synthetic test: convert a tiny fixture with known `enorm.weight = [0.5, -0.3, 1.0]`, assert the emitted GGUF tensor data equals `[1.5, 0.7, 2.0]`. Catches the `+1` bake regression without needing a real model.
- Delete `src/arch/smoke.rs:479`'s `HF2Q_QWEN35_DROP_MTP=1` guard (per P-1).

**Acceptance**:
- G1 (byte-cmp via `gguf_structural_diff`) passes on both 27B and 35B-A3B against stock llama.cpp converter. **Operator-run only**; CI runs `mtp_one_plus_w_bake_synthetic` + the unit tests, marked as the proxy gate.
- All new synthetic tests pass in default `cargo test --release qwen35::mtp` (count: ≥ 5 passing, 0 ignored, 0 failed).
- Real-model byte-cmp evidence committed to `docs/ADR-034-real-model-findings/P2-byte-cmp-qwen36-{27b,35b}.md` with `gguf_structural_diff` output and any tolerated-delta classification.

### P3 — Qwen MTP serve correctness (~400 LOC)

**Scope**:
- Run P1's numerical-parity harness (now G2-gating) against `froggeric/Qwen3.6-27B-MTP-GGUF` AND against our hf2q-converted Qwen 3.6 27B MTP GGUF (from P2). Fix any kernel-precision bugs that surface (Bug A / Bug B class — likely candidates: rope theta typo, head_dim mismatch in inner attn, attn_output_gate path on Qwen 3.5 vs 3.6).
- Run G3 (greedy byte-identical) on the N=18 existing golden fixtures at `tests/coherence_golden/` (per ADR-030 §2.3 inventory: 18 fixtures across `hello-my-name-is`, `the-quick-brown-fox`, `what-is-22` prompts and gemma-4/qwen35/dwq46/apex/apex-q5km model variants), each extended to 2k decode tokens. Fix any divergence. **G3 is unconditional**: if base decode is non-deterministic (Phase -2's B-W-1 closure failed), this ADR halts. No softening.
- Run G4 (acceptance rate). Acceptance rate measurement requires running `spec_bench.sh` on N=100 prompts of varying length (256/512/1024/2048) at greedy. Floor: max(70 %, measured-vLLM-floor-on-same-model). The "70 %" is a *public* reference number; the gate uses whichever is higher.
- Add `tests/spec_decode_real_qwen36_27b.rs` and `..._35b_a3b.rs` — full end-to-end byte-identity + acceptance regression-pin. **Operator-only tests** (gated behind `HF2Q_REAL_MODEL_TESTS=1`); a synthetic-shape sibling lives in CI.
- README.md Status row update: Qwen MTP marked **"correctness-shipped; perf TBD under P6"** — explicitly distinguishes correctness (P3) from full ship (P6). The full-ship label arrives only after P6's F1/F2/F3 all green.

**Acceptance**: G2 + G3 + G4 all green on both Qwen targets. Acceptance ≥ floor. README status reflects partial ship only (no "MTP shipped" claim until P6 lands).

### P4 — Qwen DFlash audit + correctness gates (~600 LOC delta over the existing 7011-LOC scaffold)

**Important framing**: substantial DFlash code already exists at `src/inference/spec_decode/dflash/` (9 files, 7011 LOC; per §2.1 inventory). This was written under ADR-030 (status: proposed) but never validated against the Python reference. P4 is therefore **audit + close gaps**, not write-from-scratch. The 600-LOC budget is for parity hooks + gap closures, not a re-port.

**Scope**:
- **P4a — Audit the existing scaffold against /opt/dflash Python** (no code yet, just produce findings):
  - Run /opt/dflash Python on a known prompt + Qwen 3.6 27B target; capture intermediate tensors at each layer transition AND at the drafter's fc/hidden_norm/per-layer-attn/per-layer-ffn outputs (read-only wrapper per P1's `scripts/dflash_dump_wrapper.py`).
  - Run the existing `src/inference/spec_decode/dflash/forward.rs` (2158 LOC) on the same prompt with the same drafter weights; capture the same intermediate tensors.
  - Diff. Findings go to `docs/ADR-034-real-model-findings/P4a-dflash-audit.md`. Expected: 1-3 kernel-precision or shape bugs to fix.
- **P4b — Close gaps surfaced by P4a**. Each gap is a separate commit with a regression test. No batch fixes; one bug → one commit → one test, mirroring [[project_bug_softmax_partial_simdgroup_FIXED_2026_05_17]] discipline.
- **P4c — Validate `rejection_sampler.rs` (372 LOC)** against the Leviathan-2023 spec (§3.5b above). If it already implements correctly, this is documentation work + a fixture test; if not, it's the Phase-6-of-ADR-030 deferred work that lands here.
- **P4d — Validate `hidden_capture.rs` (1085 LOC)** captures at the correct `target_layer_ids` for the live Qwen 3.6 27B (config: 5 layers, `target_layer_ids` per drafter HF config — read at P0 from downloaded drafter) and 35B-A3B targets. The MoE target's layer count differs from dense; the capture hooks must dispatch per target arch.
- **P4e — Wire into `serve/mod.rs`** (env `HF2Q_SPEC_DFLASH=1` already exists per ADR-030 iter-66 at `serve/mod.rs:1280-1303`; just complete the dispatch path with the post-P4b fixes).

**Acceptance**: G2 (numerical parity vs /opt/dflash Python on Qwen 3.6 27B + 35B-A3B), G3 (byte-identical), G4 (acceptance ≥ max(60 % paper floor, locally-measured /opt/dflash-Python floor on same model)) all green. New regression-pin tests, one per fixed gap.

**Default-flip rule (not a fallback)**: P4 ships behind `HF2Q_SPEC_DFLASH=1` opt-in by default. The default flip to "on" for DFlash-bearing-drafter-present requests happens at **P5**, gated on P5's per-request HTTP knob landing. There is no "P4 shipped independent of P5" — both ship together or neither does. This avoids the feature-flag-only-shipping pattern codex flagged.

### P5 — Gemma 4 DFlash audit + sampled-path enablement (~800 LOC delta)

**Important framing**: like P4, this is audit + close gaps over the existing DFlash scaffold; the rejection sampler already exists at `src/inference/spec_decode/dflash/rejection_sampler.rs` (372 LOC) and is validated in P4c. P5 extends to Gemma 4 + the serve-side sampled-path wire-up.

**Scope**:
- Add Gemma 4 hidden-state capture hooks. The existing `hidden_capture.rs` (1085 LOC) is Qwen-only today; add per-target dispatch. If the existing code is structured as a trait, add a Gemma 4 impl; if monolithic, refactor to per-target sub-modules (one Qwen file + one Gemma file). Refactor budget capped at +200 LOC; if structure resists, file follow-up and ship a Gemma-specific copy.
- Wire DFlash drafter loader (`weights.rs`, 305 LOC) to accept Gemma 4 drafter checkpoints from z-lab. The drafter shape differs from Qwen (per ADR-030 §3.0 locked iter-2 sizes: gemma drafter is 5-layer 2816-hidden vs Qwen drafter shape from `z-lab/Qwen3.6-27B-DFlash` config). `weights.rs` may already be target-parametric; verify before changes.
- Drop the `!sample_logits` guard at `serve/mod.rs:2810`; route sampled requests through the existing `rejection_sampler.rs` (validated in P4c).
- **Fix explicit-vs-auto routing at `serve/mod.rs:2810-2837`** (per §0 commitment — covers ALL four explicit-mode failure classes, not just missing MTP assets): split each failure case into explicit vs auto branches:

  | Failure class | Explicit opt-in (`HF2Q_SPEC_DECODE=1` env OR per-request `spec_decode: on/dflash/mtp`) | Auto mode (no flag, heuristic-chosen) |
  |---|---|---|
  | `model.mtp.is_none()` (Qwen MTP requested, no MTP weights in GGUF) | Typed error `SpecDecodeAssetsMissing { mechanism: "mtp", missing: ["blk.N.nextn.*"] }`; 4xx HTTP | Warn + fall through to greedy |
  | DFlash drafter weights not loaded / not provided (`spec_decode: dflash` with no `--draft-model` flag) | Typed error `SpecDecodeAssetsMissing { mechanism: "dflash", missing: ["draft_model_path"] }`; 4xx | Warn + fall through to greedy |
  | Required draft kernel missing (e.g. mlx-native dispatch returns `DispatchUnsupported` for the draft block on the live device) | Typed error `SpecDecodeKernelUnavailable { mechanism, kernel: "<name>" }`; 5xx | Warn + fall through to greedy |
  | Runtime draft failure (draft forward returns error mid-decode after accepting >0 tokens) | Typed error `SpecDecodeRuntimeFailure { mechanism, position, cause: <source error> }`; abort request mid-stream | **Same** — there is no defensible "auto fall through" mid-stream; in auto mode, abort with a less-strict error message but still abort (no token can be re-emitted with different mechanism mid-stream without breaking determinism) |

  New unit tests under `tests/serve_explicit_vs_auto/` (one file per failure class): `mtp_assets_missing.rs`, `dflash_assets_missing.rs`, `draft_kernel_unavailable.rs`, `draft_runtime_failure.rs`. Each test exercises both explicit and auto branches.
- Add per-request HTTP knobs (`spec_decode: on|off|auto|mtp|dflash`, `spec_decode_k: N`, `draft_model: <path>` for DFlash) to `src/serve/api/...` chat-completions body schema.
- Surface `mtp_acceptance` / `dflash_acceptance` in `usage` block (streaming + non-streaming).
- New `tests/sampled_path_qwen36.rs` + `tests/sampled_path_gemma4_dflash.rs` — G5 (distribution preservation) tests via KL/log-prob/Jaccard.

**Acceptance**: Gemma 4 DFlash: G2+G3+G4 green. Sampled path: G5 green on both Qwen MTP and Gemma DFlash. HTTP knobs documented in OpenAPI spec.

### P6 — Performance gate (~200 LOC scripting + bench logs)

**Scope**:
- K=1 vs K=2 vs K=3 sweep on Qwen 3.6 27B MTP, Qwen 3.6 35B-A3B MTP, Gemma 4 26B DFlash, Gemma 4 31B DFlash. 3-run paired-baseline alt-pair per `scripts/spec_bench.sh`.
- Tune the winning K per target into a config (`HF2Q_SPEC_DECODE_K=<K>`); document defaults in README.
- Run F1, F2, F3 perf gates per §3.3:
  - F1 against MTPLX D3 on `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed` at `temp=0.6 top_p=0.95 top_k=20` (paired same-machine).
  - F2 against llama.cpp `--spec-type draft-mtp` on `froggeric/Qwen3.6-27B-MTP-GGUF` (paired).
  - F3 against hf2q-native baseline (paired).
- If any of the three fails: surface the bottleneck (typically would be a kernel-dispatch site that mlx-native can fuse), open follow-up issue, do NOT mark P6 shipped.
- Write `docs/ADR-034-real-model-findings/P6-perf-bench.md` with all numbers, σ, paired-baseline matchup, thermal log.

**Acceptance**: F1 ≥ 63.000 tok/s on the MTPLX target; F2 ≥ our local llama.cpp number; F3 > non-MTP baseline. All three green; new perf regression-pin in `scripts/spec_bench.sh --gate` (operator-runs; not in CI).

### P7 (follow-up) — Gemma 4 Google `-assistant` MTP path

**Out of v1 scope; tracked separately**. Scope when picked up:
- Add Gemma 4 MTP-assistant arch entry + convert mapper (the `-assistant` checkpoints are separate HF models, not in-checkpoint heads; structurally closer to DFlash than to Qwen native MTP).
- Same gates G1-G5.
- Reuses Workstream B DFlash orchestrator if structurally compatible.

---

## 5. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **R1** — Byte-cmp gate at P2 fails because stock llama.cpp's GGUF KV ordering differs in a way our `gguf_structural_diff` helper doesn't classify | Medium | Phase 2 stalled | P2 adds a defined-set `gguf_structural_diff` tool (`tests/util/gguf_structural_diff.rs`) that whitelists exactly four tolerated deltas: (a) KV-key emission order, (b) tensor metadata order within `tensor_infos[]` provided offsets resolve to identical bytes, (c) `general.quantization_version` value differences when both are valid v2/v3 markers, (d) tokenizer-metadata KV reordering when value-bytes identical. Anything else fails G1. No subjective "envelope" decisions; the diff tool's source code IS the contract. |
| **R2** — `+1` offset is one of *multiple* silent kernel bugs in current MTP forward; P3 surfaces N>1 bugs | High | Phase 3 schedule blown | Plan for it. P3's deliverable is "all surfaced bugs fixed", not "the +1 bug fixed". Phase-3 budget is open-ended; cap at 2 weeks before escalating to operator. |
| **R3** — MTPLX's `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed` is a TUNED checkpoint, not a stock Qwen 3.6 27B MTP | Confirmed (true) | F1 comparison may be apples-vs-oranges | Run F1 *both* against the tuned MTPLX target AND against stock `Qwen/Qwen3.6-27B` with `froggeric` MTP GGUF. Ship the gate that's tighter. Cite both numbers in P6 final report. |
| **R4** — `/opt/dflash` Python is mlx_lm-flavored MLX, not mlx-native Rust; port loses some performance to per-token Python overhead in the reference, giving us an artificially soft G4 floor | Medium | Acceptance comparison is unfair to us | Acquire DFlash's published GPU numbers (RTX 6000, 40 t/s gsm8k); apply a hardware-class adjustment factor when comparing across CUDA→Apple Metal. Cite both reference numbers in P6. |
| **R5** — B-W-1 greedy-decode heisenbug from ADR-015 iter61a-3 may still be open; G3 (byte-identical) requires deterministic base decode | Medium | Hard-blocks all ADR-034 phases that depend on G3 | **Phase -2 is mandatory** (see §4): the first action of this ADR is verifying B-W-1 status; if open, iter61a-4 closure ships **before** ADR-034 P-1 begins. G3 is never softened or made optional — if B-W-1 cannot be closed, ADR-034 itself does not proceed. Codex review 2026-05-19 flagged the original "G3 conditionally optional" wording as a mantra violation; removed. |
| **R6** — Operator's "all Qwen 3.6 and Gemma 4" might mean architectures we haven't enumerated (e.g. Qwen 3.5 was also MTP-trained, Qwen 3-VL has MTP that's currently dropped) | Low | Scope creep | This ADR explicitly enumerates the 2×2 matrix in §3.1. Anything outside the matrix is out of v1 scope (see §6); operator confirms in P-1 review. |
| **R7** — Workstream B (DFlash) competes with Workstream A (MTP) for the same decode-time budget; shipping both may yield only marginal improvement over shipping one (per-mechanism overhead doesn't compose linearly) | Medium | P6 perf gate underwhelms | Treat MTP and DFlash as **mutually-exclusive runtime options** in v1 — per-request choice via `spec_decode: mtp|dflash|off|auto`. The `auto` rule defaults to MTP for Qwen MTP-bearing GGUFs and DFlash for Gemma 4 + Qwen non-MTP GGUFs. No combined-mechanism path in v1. |
| **R8** — Numerical-parity ε of 1e-3 may be too tight for BF16-accumulated 5-layer drafter; could spuriously fail G2 | Medium | False-positive failures | The 1e-3 floor came from ADR-028 iter-156's BF16 envelope for D=512 attention. For drafter depths < D=512 the envelope is tighter; for DFlash 5-layer it's looser. **Mitigation**: the committed tolerance ladder in `mtp_parity.py` is the single source of truth — if the default ladder spuriously fails on the DFlash 5-layer drafter, an update to the *committed* ladder (with rationale in `docs/ADR-034-real-model-findings/`) is the right response, not a runtime override. This is explicitly NOT a "measure-later" escape hatch. |

---

## 6. Explicitly NOT doing (v1 scope discipline)

- **Training our own MTP heads** for any arch. ADR-030 §3.8 noted this as out-of-scope for DFlash; it stays out-of-scope here too. We consume external trained drafters/heads only.
- **DeepSeek-V3 / V4 native MTP** support. Different arch family; would require porting the DeepSeek model graph to mlx-native — multi-week project unrelated to this ADR.
- **GLM-4.6 / MiniMax-M2 / Kimi-K2.5 / gpt-oss / Llama-3.1 / Qwen3-Coder DFlash** — z-lab publishes drafters for all of these but each requires a target arch port. Out of v1; trivially extensible via `dflash/targets/<arch>.rs` once a target arch is supported.
- **Qwen3-VL MTP** — `qwen3vl_text.rs:120` drops `mtp.*` explicitly. Lifting that drop is its own ADR.
- **K > 3** — public references stop at K=3 (MTPLX D3, llama.cpp `--spec-draft-n-max 3`). Higher K explores tail of acceptance distribution; not v1.
- **MTP slot residency in the ADR-017 persistent block-prefix cache** — separate optimization. v1 is per-request alloc.
- **Combined MTP+DFlash runtime path** (per R7).
- **Multi-batch spec-decode** — single-request only in v1.
- **MTP-aware quantization** (e.g. higher-precision MTP heads at convert time). Stock policy applies. Per-tier MTP-quant overrides are a follow-up ADR.
- **The `havenoammo/Qwen3.6-35B-A3B-MTP-GGUF` as authoritative ground truth** — used only as cross-check at P0. Authority is stock llama.cpp converter on official safetensors.

---

## 7. Open issues / interview takeaways

Per the operator interview (2026-05-19):

1. **Q**: Within Workstream A, which Qwen target ships first? **A**: Both in lockstep (single PR; §P2).
2. **Q**: Gemma 4 path — assistant MTP or DFlash? **A**: DFlash first (this ADR's P5); assistant MTP follow-up (Phase 7).
3. **Q**: ADR shape? **A**: One unified ADR-034 (this doc).
4. **Q**: Scope ceiling? **A**: Long "do it all correctly" — full 2×2 matrix (this ADR's P2-P5).
5. **Q**: Correctness gate? **A**: Byte-cmp vs llama.cpp PR #22673 converter (this ADR's G1).
6. **Q**: References? **A**: All four sources (stock llama.cpp converter + froggeric + unsloth/RDson + HF transformers Python) (this ADR's P0).
7. **Q**: Perf bar? **A**: "as fast (or faster) than mtplx and llama.cpp, and obviously faster than non-MTP of hf2q" (this ADR's F1/F2/F3 in §3.3).

**Open for follow-up (post-merge of this ADR)**:
- R5: B-W-1 heisenbug closure is now mandatory Phase -2 (per codex review 2026-05-19). Operator-verified `determinism_check.sh` result is the green-light for ADR-034 P-1 to start.
- R7: validate the mutually-exclusive MTP-vs-DFlash runtime policy in P6's actual measurements; if both can compose with > sum-of-parts speedup on some workloads, consider a v2 follow-up.

**Codex review history**:
- **Round 1** (2026-05-19, `/tmp/cfa-adr034-review/codex-last.txt`): verdict `request_changes`, severity `high`. 5 must-fix items + 4 nice-to-haves applied; see round-2 below for status verification.
- **Round 2** (2026-05-19, `/tmp/cfa-adr034-revisit/codex-last.txt`): verdict `request_changes` (still), partial-credit on 3 of the 5 must-fix items. Remaining blockers applied in this revision:
  1. ✅ `+1` offset contract: **broadened correctly** to match `qwen.py:303-304` exactly — applies to ALL `name.endswith("norm.weight") AND NOT name.endswith("linear_attn.norm.weight")` (i.e., main-stack norms + MTP block norms after remap, NOT just MTP-specific norms). §1.1 + §1.3 + §3.5 + P2 all rewritten with the full baked-tensor table.
  2. ✅ Stale `model.mtp.layers.0.*` at §2.2 line 196 corrected to `mtp.layers.{bid}.*`.
  3. ✅ R5 loophole — was already addressed in round-2 review; verified clean.
  4. ✅ P5 explicit-routing broadened to cover ALL four explicit-mode failure classes (missing MTP assets, missing DFlash assets, missing draft kernels, runtime draft failures) with per-class typed errors and per-class unit tests.
  5. ✅ CI vs operator gate split — already addressed in round-2 review; verified clean.
  - Stale F1/F2/F3 perf-bar refs at lines 432-433, 544, 578 corrected (`P-1 reference baseline` → `F1 reference baseline`, etc.).
- **Engineer-readiness additions applied at the same revision** (operator's "complete enough for an engineer to pick up tomorrow" bar):
  - ✅ Concrete file:line for `Qwen35Config::from_gguf` at `src/inference/models/qwen35/mod.rs:408-442` in P2.
  - ✅ Existing DFlash scaffold inventory (7011 LOC, 9 files) added to §2.1; P4/P5 rescoped from "write from scratch" to "audit + close gaps".
  - ✅ Convert-pipeline data-transform plug-in mechanism specified: `post_map_data_transform(arch, gguf_name) -> Option<fn(&mut Vec<f32>)>` hook at `cli_driver.rs:~1240`.
  - ✅ P1 Python reference forward strategy specified (HF transformers `trust_remote_code=True` import of `model.model.mtp.layers[0]` since MTP is not in the upstream HF transformers code).
  - ✅ Test-fixture pointer: G3 uses the existing 18 fixtures at `tests/coherence_golden/`.
  - ✅ Leviathan-2023 inline math added as §3.5b (7-line algorithm + invariants); G5's measurement protocol made concrete.
  - ✅ `scripts/spec_bench.sh` example invocations for all 4 paired-baseline matchups (F1, F2, F3, DFlash branch).
- **Round 4** (2026-05-19, `/tmp/cfa-adr034-r4/codex-last.txt`): verdict **`approve_with_minor`** — all 5 round-3 items addressed; 2 doc-cleanups applied: P-1 scope wording "External pins (currently TBD)" reworded to "re-verify against live HEADs; refresh if drift detected" (since SHAs are already locked in the header block); §3.4 redundant parent-level `rejection_sampler.rs` row removed (sampler lives inside `dflash/` and is shared via direct import). ADR cleared for engineer pickup.
- **Round 3** (2026-05-19, `/tmp/cfa-adr034-r3/codex-last.txt`): verdict `request_changes`, 2 narrow blockers + 3 polish items. All applied:
  1. ✅ §3.5 HF→GGUF table rows for `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm` corrected from "NO `+1`" to "✅ `+1` baked" — they DO get baked after the `mtp.layers.0` → `model.layers.{N}` remap matches the `endswith("norm.weight")` rule.
  2. ✅ P2 "+1 bake" wording at line 514 broadened: dropped the "AND the original source had the `mtp.` prefix" qualifier; now applies arch-wide to all post-remap GGUF names ending `norm.weight` except `ssm_norm.weight`.
  3. ✅ §2.2 gap-table row corrected: "+1 offset on enorm/hnorm" → "+1 offset on all Qwen 3.5/3.6 `norm.weight` tensors except `ssm_norm.weight`".
  4. ✅ §3.4 module layout updated: DFlash files marked as EXTEND of existing 7011-LOC scaffold, not NEW; per-file LOC counts inline.
  5. ✅ §2.2 gap-table rows for DFlash forward/loader/orchestrator changed from "new" to "EXTEND existing".
  6. ✅ Stale §3.6+8 reference at P4c corrected to §3.5b (the Leviathan-2023 inline math section).
  7. ✅ ε tolerance ladder contradiction resolved: G2 row + R8 + P1 wording now all say the ladder is committed in `mtp_parity.py`; changes require committed updates with rationale, not runtime overrides.

---

## 8. References

### Internal
- `[[project_adr033_p4b_transitive_proof_2026_05_19]]` — ADR-033's byte-cmp methodology (carried forward as G1).
- `[[feedback_codex_review_loop_2026_05_17]]` — standing rule: codex stays in the review loop for non-trivial changes (applied here at draft completion).
- `[[feedback_no_loop_suppression_2026_05_17]]` — no fallback / no suppression (encoded in G1-G4 hard-gate contract).
- `[[feedback_test_both_families_2026_05_17]]` — test both Gemma + Qwen before "done" (encoded in P2 lockstep + P5 Gemma DFlash).
- `[[feedback_no_backwards_compat_2026_05_18]]` — delete the stale `HF2Q_QWEN35_DROP_MTP` guard (P-1).
- `docs/ADR-012-qwen35moe-conversion.md:747-755` — MTP convert intent (superseded by this ADR's P2).
- `docs/ADR-013-qwen35-inference.md:780-801` — P14 "complete" claim (superseded by §1.2 audit).
- `docs/ADR-015-mlx-native-single-cb-decode.md` (iter61a-3) — B-W-1 heisenbug origin (R5).
- `docs/ADR-028-peer-parity-coherence-and-speed.md:1143,3354-3389` — MTP K=3 / Qwen3.6-27B-MTP GGUF inspection.
- `docs/ADR-029-gemma4-moe-pipeline-is-the-gap.md` — n-gram proposer (untouched).
- `docs/ADR-030-dflash-block-diffusion-spec-decode.md` — DFlash design (absorbed as Workstream B).
- `docs/ADR-033-unified-quant-convert-pipeline.md` — convert-pipeline correctness methodology (G1 inherits its byte-cmp gate style).

### External
- DeepSeek-V3 Technical Report — [arxiv 2412.19437](https://arxiv.org/html/2412.19437v1) — native-MTP origin
- DFlash paper — [arxiv 2602.06036](https://arxiv.org/abs/2602.06036) — block-diffusion drafter
- llama.cpp MTP PR — [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) — merged 2026-05-16
- Google Gemma 4 MTP blog — [blog.google](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/)
- Gemma 4 MTP HF docs — [ai.google.dev/gemma/docs/mtp/mtp](https://ai.google.dev/gemma/docs/mtp/mtp)
- LiteRT-LM Gemma 4 MTP extraction context — [groundy.com](https://groundy.com/articles/litert-lm-v0101-ships-gemma-4-mtp-heads-that-llamacpp-cant-access/)
- MTPLX — [github.com/youssofal/MTPLX](https://github.com/youssofal/MTPLX); local clone at `/opt/MTPLX`
- z-lab DFlash GitHub — [github.com/z-lab/dflash](https://github.com/z-lab/dflash); local clone at `/opt/dflash`
- vLLM MTP docs — [docs.vllm.ai/.../mtp/](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/)
- NodeNestor Qwen3.5-27B MTP llama.cpp — [github.com/NodeNestor/qwen3.5-27b-mtp-llamacpp](https://github.com/NodeNestor/qwen3.5-27b-mtp-llamacpp)
- Leviathan-2023 rejection sampling — *Fast Inference from Transformers via Speculative Decoding*, Leviathan / Kalman / Matias, [arxiv 2211.17192](https://arxiv.org/abs/2211.17192)
- SPEC-BENCH harness — [github.com/hemingkx/Spec-Bench](https://github.com/hemingkx/Spec-Bench) — uniform-harness reference for §3.3 perf gates

### Reference checkpoints (HuggingFace)
- `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.6-35B-A3B` — official target safetensors
- `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B-it` — official target safetensors
- `froggeric/Qwen3.6-27B-MTP-GGUF`, `unsloth/Qwen3.6-27B-MTP-GGUF`, `RDson/Qwen3.6-27B-MTP-Q4_K_M-GGUF`, `havenoammo/Qwen3.6-{27B,35B-A3B}-MTP-GGUF` — reference MTP GGUFs (G1 cross-check)
- `z-lab/Qwen3.6-27B-DFlash`, `z-lab/Qwen3.6-35B-A3B-DFlash`, `z-lab/gemma-4-26B-A4B-it-DFlash`, `z-lab/gemma-4-31B-it-DFlash` — DFlash drafters
- `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed` — MTPLX perf-reference target
- `google/gemma-4-{26B-A4B,31B,E4B}-it-assistant`, `mlx-community/gemma-4-26B-A4B-it-assistant-bf16` — Gemma 4 assistant MTP (Phase 7 follow-up)
