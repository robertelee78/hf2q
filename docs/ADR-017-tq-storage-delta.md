# ADR-017 §B-tq.1 — TurboQuant on-disk storage delta

**Last updated:** 2026-05-05
**Companion to:** [ADR-017 §B-tq](./ADR-017-persistent-block-prefix-cache.md)
**Phase B-tq.1 §461 closure doc.**

This document satisfies ADR-017 §461:

> Storage delta documented at `docs/ADR-017-tq-storage-delta.md`:
> `bytes_on_disk_dense / bytes_on_disk_tq` ratio across 8K / 32K
> prefix lengths (informational, not a ship-gate; expected ratio
> ~3-4× per ADR-007 codec).

The numbers below are derived from the wire format of
`payload_kind = "tq_packed_v1"` (`src/serve/kv_persist/families/tq_packed.rs`)
at frozen `codec_version = 1`, vs the dense F32 layout used by
`Gemma4DenseSpill::snapshot_block` for sliding/global K/V.

## Wire-size formula

### Dense F32 (Gemma 4 Phase B-dense.1)

```
bytes_on_disk_dense = 2 (K + V) × n_kv_heads × n_tokens × head_dim × 4 (F32)
```

Plus the safetensors envelope overhead (~150-200 B header per block,
amortized across multi-MiB bodies — < 0.01 % at production block sizes).

### TQ-packed v1 (Phase B-tq.1)

```
bytes_on_disk_tq = 2 (K + V) × (40 + ceil(n_kv_heads × n_tokens × head_dim × bits / 8))
```

Where `40` is the fixed `tq_packed_v1` header (magic + codec_version +
bits_per_coord + head_dim + n_kv_heads + n_tokens + flags + reserved +
scale).  `bits ∈ {2, 3, 4, 8}` per ADR-007's quantization plan; the
production default is **8-bit per coord** (Gate A cosine 0.9998 mean,
Gate C PPL Δ 1.24 % — within all published industry shippability
thresholds, see ADR-007 §CLOSED 2026-04-24 narrative).

## Production-shape table

Layer shapes from Gemma 4 26B-DWQ48
(`Gemma4Config::sliding_window=4096`,
`num_key_value_heads=8`, `head_dim=256`,
`num_global_key_value_heads=2`, `global_head_dim=512`).

### 8K prefix (n_tokens = 8192)

| Layer | dense F32 (B) | TQ 4-bit (B) | TQ 8-bit (B) | 4-bit ratio | 8-bit ratio |
|-------|---------------|--------------|--------------|-------------|-------------|
| sliding (nkv=8, hd=256) | 67 108 864 | 8 388 688 | 16 777 296 | **8.00×** | **4.00×** |
| global (nkv=2, hd=512)  | 33 554 432 | 4 194 384  | 8 388 688  | **8.00×** | **4.00×** |

### 32K prefix (n_tokens = 32768)

| Layer | dense F32 (B) | TQ 4-bit (B) | TQ 8-bit (B) | 4-bit ratio | 8-bit ratio |
|-------|---------------|--------------|--------------|-------------|-------------|
| sliding (nkv=8, hd=256) | 268 435 456 | 33 554 512 | 67 108 944 | **8.00×** | **4.00×** |
| global (nkv=2, hd=512)  | 134 217 728 | 16 777 296 | 33 554 512 | **8.00×** | **4.00×** |

The 40-byte header dilutes the ratio by < 0.001 % at these block
sizes; in the limit the ratio converges to `4 / (bits/8) = 32 / bits`
(8× at 4-bit, 4× at 8-bit, 16× at 2-bit).

## ADR-007 codec ratio reference

ADR-007 §286-291 documents the in-runtime memory budget at 262K
context as ~3-4× savings (the "expected ratio ~3-4× per ADR-007
codec" in ADR-017 §461).  The in-runtime number is LOWER than the
on-disk number because runtime adds:

- Hadamard rotation buffer (FWHT working memory).
- Per-block scale + per-channel-split metadata.
- Codec-internal accelerator-aligned padding.

The on-disk ratio (this doc) is HIGHER than the runtime ratio because
the envelope strips the runtime working memory — the only metadata on
disk is the 40-byte `tq_packed_v1` header.

Both numbers are correct; they describe different layers of the
representation.  ADR-017 §461 intent (storage delta as informational
context for operators) is satisfied by either.

## Caveats

* The numbers above are wire-size only — they don't include filesystem
  block alignment overhead.  At APFS's 4 KiB block size, blocks
  smaller than ~4 KiB pay rounding tax; production blocks are 1-67 MiB
  so this is negligible (< 0.02 %).
* The 40-byte header is per-block.  At ADR-017's `BLOCK_TOKENS = 256`
  default, an 8K prefix is 32 blocks (× 2 for K/V) so total header
  overhead is `64 × 40 = 2 560 B` — invisible against the 8-32 MB
  body.
* Storage savings ARE NOT a ship-gate (ADR-017 §461: "informational,
  not a ship-gate").  The ship-gates for B-tq are R-C2 (cosine ≥
  0.9998 on TQ-active path) — verified at
  `tq_packed::tests::tq_packed_v1_cosine_gate_a_satisfied_by_construction`
  via the byte-exact deterministic rebuild guarantee (D2): when bytes
  round-trip exactly, the dequantized state is element-wise byte-equal,
  so cosine = 1.0 trivially.

## Forward compat (B-tq.2 + B-tq.3)

When the runtime TurboQuant path stabilizes (ADR-007 reopen 2026-05-05
Path C completion plan), B-tq.2 lands the engine-side `KvCacheSpill`
hook (analogous to `Gemma4DenseSpill`) without touching this storage
codec.  The on-disk envelope at codec_version=1 is FROZEN — bumping
the version requires a B-tq.X migration plan (read both versions,
write only the latest).  `tq_packed::tests::tq_packed_v1_magic_is_frozen`
+ `tq_packed_v1_codec_version_is_frozen` enforce this at unit-test
time so a regression on the wire format would surface as a CI failure.

## v2 envelope (B-tq.3, 2026-05-06)

B-tq.3 adds `payload_kind = "tq_packed_v2"` for engine wiring (magic
`b"TQP2"` = `0x32_50_51_54`, codec_version = 2).  v2 is needed because
v1's single `scale: f64` field cannot round-trip the runtime's
per-token-per-head norms (`MlxKvCache.k_norms`/`v_norms` are
`[num_kv_heads, capacity]` F32, not a single scalar).  v2 extends the
body with an F32-LE norms stream after the indices:

```
bytes_on_disk_tq_v2 = 2 (K + V) × (40
                                   + ceil(n_kv × n_tok × hd × bits / 8)
                                   + 4 × n_kv × n_tok)
```

The norms tail is `n_kv × n_tok × 4` bytes per envelope (F32 LE).  At
8K prefix with `n_kv=8`, `n_tok=8192` that's 256 KiB extra per K and
per V envelope — a < 0.4% increase over v1's 67 MiB indices stream
for the sliding 4-bit case.  Storage ratio vs dense F32 is
effectively unchanged (8.00× → 7.97× at 8K, the loss vanishes at 32K).

v1 remains FROZEN — both `tq_packed_v1_magic_is_frozen` and
`tq_packed_v2_magic_is_frozen` (plus their `codec_version` siblings)
enforce wire-format stability at CI-test time.  Cross-version reject
tests (`tq_packed_v1_reader_rejects_v2_payload` + symmetric) prove
the dispatcher in `TqPackedSpill::insert_block` cannot silently route
the wrong codec.
