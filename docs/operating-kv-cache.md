# ADR-017 Operator Runbook — KV Cache

**Date:** 2026-05-01
**Authored against commit:** `7c6c16005250ffb3ce993a9227d01456adf403b2`
**ADR:** `docs/adr/diary/ADR-017-persistent-block-prefix-cache.md` (R-O1)

> **⚠ 2026-08-08 addendum (supersedes qwen35-family claims below).**
> This runbook predates the qwen35/3.6 persist family. What it calls
> `[NOT YET IMPLEMENTED]` items §11.9–10 ("Hybrid (Qwen 3.5) family",
> "TQ-active codec family") **landed** (ADR-027 iters 1..54:
> `src/serve/kv_persist/families/qwen35_{hybrid,disk}_persistor.rs`) and
> were made production-correct end-to-end by **ADR-027 sub-iter 23d-γ
> (2026-08-03)**: TQ-only LCP resume restores all four TQ buffers per
> slot (was silent zero-prefix corruption), `cfg_from_cache` derives
> shape from TQ buffers (was an engine panic → HTTP 500), QH35 codec is
> at **v5** (compact sequence-axis snapshots; readers retain v1--v4
> compatibility), and
> the on-disk fingerprint is **substrate-namespaced** (F32Only/TqOnly/
> Both never cross-hydrate). Qwen autoregressive serve is now the default;
> no `HF2Q_QWEN36_AUTOREG` activation is required. Disk restart hydration is
> proven only for SerialFifo; the canonical SlotAware launcher provides
> same-process retained-prefix affinity and does not enable this disk path.
> Current SerialFifo operator shape:
> `HF2Q_KV_LCP_RESUME=1 hf2q serve --scheduler fifo-serial
> --model <qwen3.6.gguf> --kv-persist <dir>
> --kv-persist-budget 32GiB --overflow-policy reject`.
> The typed path and budget now feed both the qwen35 disk persistor and the
> generic hot-swap spiller substrate. The rest of this runbook remains accurate
> for the Gemma-4 spiller substrate it was written against.

> **State of truth (2026-05-01, preserved for provenance).** This runbook documents only what is implemented in the
> tree at the commit above. Phase D is ship-gated but the substrate is fully
> wired (cmd_serve flag + recovery scan + spiller + writer + factory). The
> operator-facing behavior on the loaded path is observable but, for the
> default Gemma 4 family at this commit, _functionally inert_ until the
> `Gemma4DenseSpillFactory` placeholder shape is replaced with real GGUF
> metadata extraction. Sections that depend on absent code are marked
> `[NOT YET IMPLEMENTED — ADR-017 §X]` and listed in §11.

---

## 1. Quick start

```bash
hf2q serve --model <gguf-or-repo> --kv-persist /var/cache/hf2q/kv-persist
```

`--kv-persist PATH` is the typed serve authority. Without it, serving does not
construct a persistent store, and no environment variable overrides an
explicit path. The disk ceiling is the typed
`--kv-persist-budget` / `[serve] kv_persist_budget` control. The former
`HF2Q_KV_PERSIST_BUDGET_BYTES` production readers are removed.

---

## 2. What KV persistence does (and doesn't)

When `--kv-persist=PATH` is set, `cmd_serve`:

- Creates `PATH` if missing (`src/serve/mod.rs:1733`).
- Runs a synchronous `recover_from_disk(...)` scan
  (`src/serve/mod.rs:1743`; impl `src/serve/kv_persist/recovery.rs:110`)
  rebuilding the in-memory `BlockIndex`.
- Builds `DiskBlockStore` with the resolved `--kv-persist-budget` /
  `[serve] kv_persist_budget` value (`0` means unlimited)
  (`src/serve/mod.rs:1764`, `block_store.rs:90-93`).
- Spawns `AsyncWriterHandle` with `DEFAULT_CHANNEL_CAPACITY = 8`
  (`writer.rs:218`, `mod.rs:1779`).
- Wires `BlockPrefixCacheSpiller` into `HotSwapManager`
  (`mod.rs:1905-1912`).

**What gets persisted at this commit:** for the operator's `--model`,
the substrate registers `StubGemma4Spill` (always
`None`/`Err(CodecErr)` — `spiller.rs:506-528`) PLUS a
`Gemma4DenseSpillFactory` with **placeholder shape config**
(`mod.rs:1848-1869`: `layer_types: [Sliding, Full]`,
`nkv_heads: [8, 2]`, `head_dim: [256, 512]`,
`max_decode_tokens: 8192`). The factory substitutes the real
`Gemma4DenseSpill` at first engine load — gated on `Arc::downcast`
matching `Arc<EngineHandle>`. Real Gemma 4 26B has 64 layers
(48 sliding + 16 full), so the placeholder is correct shape but wrong
length; a real load may legitimately no-op via downcast mismatch.

**What survives evict-readmit:** dense F32/F16 K/V tiles for Gemma 4 IF
the factory substitutes successfully. See
`families/gemma4_dense.rs:14-65` for activation conditions and the
post-admit-before-prefill allocation contract.

**What does NOT survive:** hybrid/recurrent/conv state (no Qwen 3.5
hybrid family in `src/serve/kv_persist/families/` at this commit);
TQ-active codec state (no B-tq family); Gemma 4 cache when the
placeholder config does not match the loaded weights (silent no-op via
downcast mismatch per `registry.rs:120-126`).

---

## 3. Cache directory layout

Path: operator-supplied via `--kv-persist=PATH`. No default — flag is
`Option<PathBuf>` (`src/cli.rs:566`); absent ⇒ no cache. Doc-string
examples: `/tmp/hf2q-kv-persist`, `$HOME/.cache/hf2q/kv-persist`
(`src/cli.rs:548`).

`DiskBlockStore::new(...)` creates these subdirs
(`block_store.rs:117-121`):

```
<PATH>/
  locks/                              # advisory flock files
  models/                             # per-model namespace
```

Per-family namespace: a `ModelFingerprint` short hex (first 16 hex chars
of `sha256(repo \\0 quant \\0 producer_version \\0 source_sha256 \\0
tokenizer_chat_template)` — `format.rs:11-17`, `format.rs:170-176`,
`compute_model_fingerprint`). At this commit the spiller passes empty
strings for `producer_version`, `source_sha256`,
`tokenizer_chat_template` (`spiller.rs:242-244`); only `repo` + `quant`
contribute, both sourced from `LoadedHandle`.

Block file naming (`block_store.rs:175-184`):

```
<PATH>/models/<fp_short_hex>/kv/<hex0>/<full_hex>.safetensors
```

`<hex0>` = first hex char of the block hash (256-bucket fanout);
`<full_hex>` = lowercase 64-hex sha256(body). Envelope shape:
`format.rs:36-40`.

Lock files, per-`(model_short, hash[..2])` (256 buckets per model,
`block_store.rs:197-205`):

```
<PATH>/locks/<fp_short_hex>__<hash_prefix2>.lock
```

Quarantine subdir, lazy on first event (`recovery.rs:258-261`,
`block_store.rs:188-193`):

```
<PATH>/models/<fp_short_hex>/kv-quarantine/<reason>__<original_name>.safetensors
```

`<reason>` ∈ `trunc | verbump | bodyhash | parity` (`recovery.rs:80-87`).

`*.tmp.<pid>` scratch files appear during writes; recovery ignores them
(`recovery.rs:167-171`).

---

## 4. Operator commands

### Enable

```bash
hf2q serve --model <model> --kv-persist /var/cache/hf2q/kv-persist
```

Flag definition: `src/cli.rs:565`. Wiring: `src/serve/mod.rs:1719-1922`.

### Disable

Run `hf2q serve` without `--kv-persist`. The flag is Option-typed
(`src/cli.rs:566`); absent flag means `None` and the optional substrate
block at `src/serve/mod.rs:1719-1922` is skipped entirely. The default
`NoopKvSpiller` remains wired in `AppState::new_for_serve`. There is no
mid-flight toggle — `--kv-persist` is a startup-time decision.

### Clear

`[NOT YET IMPLEMENTED — ADR-017 §R-F6]` There is no native subcommand for
clearing KV namespaces. The CLI's `cache` subcommand
(`src/cli.rs:140-173`) supports only `List`, `Size`, and
`Clear { model, quant, all, yes }` — none of these touch the
`<kv-persist>/models/<fp_short>/` subtree. The ADR-017 R-F6 spec
(`docs/adr/diary/ADR-017-persistent-block-prefix-cache.md:336`) calls for
`hf2q cache clear --kv-namespace --model <repo>`; it has not landed.

Operator fallback (run while `cmd_serve` is stopped — there is no
write-quiesce signal):

```bash
# Wipe everything for one model:
rm -rf <PATH>/models/<fp_short_hex>

# Nuke the entire cache:
rm -rf <PATH>/models <PATH>/locks
```

The recovery scan tolerates missing/empty `models/` (`src/serve/kv_persist/recovery.rs:115-119`).

### Inspect

```bash
# Block file count for one model:
find <PATH>/models/<fp_short>/kv -name '*.safetensors' | wc -l

# Bytes on disk for one model:
du -sh <PATH>/models/<fp_short>/kv

# Quarantined block list with reason prefix:
ls -la <PATH>/models/<fp_short>/kv-quarantine/ 2>/dev/null
```

To inspect a single envelope: read the leading 8 bytes (LE u64
`header_len`), then the next `header_len` bytes are JSON header padded
with ASCII space to 8-byte alignment (`src/serve/kv_persist/format.rs:36-40`).
`hexdump -C <file> | head` is enough to surface the JSON header.

---

## 5. Quarantine inspection

Corrupted blocks are MOVED (not deleted) to
`<PATH>/models/<fp_short>/kv-quarantine/` per ADR-017 §R-F9, with a
filename prefix recording the cause (`src/serve/kv_persist/recovery.rs:78-87`):

| Prefix | Cause | Detection point |
|--------|-------|----------------|
| `trunc__` | `TruncatedHeader` — file shorter than declared `header_len` or unparseable | recovery scan (`recovery.rs:180-187`) |
| `verbump__` | `VersionMismatch` — `format_version` ≠ `CURRENT_FORMAT_VERSION` (= `1` at this commit, `format.rs:75`) | recovery scan (`recovery.rs:190-194`) |
| `bodyhash__` | `BodyHashMismatch` — body sha256 didn't match `header.block_hash` | lazy on read path (`recovery.rs:11-14`) |
| `parity__` | `ParityFail` — reserved for forward-compatible envelope parity check | not used at this commit |

Recovery-time quarantine drains during `cmd_serve` startup and is
reflected in the `tracing::info!` line emitted at
`src/serve/mod.rs:1750-1757` (fields: `cache_dir`, `blocks_indexed`,
`blocks_quarantined`, `bytes_indexed`, `elapsed_ms`).

**When to investigate vs delete:**

- One-off `trunc__` / `verbump__` after `kill -9`, disk-full, or format
  bump: safe to delete after capturing a hex dump. Recovery tolerates.
- Repeated `bodyhash__` on healthy hardware: silent corruption OR a
  per-family codec bug. Capture, file an ADR-017 defect, delete.
- Any quarantine after a clean shutdown: investigate. The atomic-rename
  + flock pattern (`block_store.rs:213-253`) should preclude this.

`/metrics` counter for quarantine: `[NOT YET IMPLEMENTED — ADR-017 §R-F7]`
— no `hf2q_kv_quarantined_total{reason}` emit site exists. Operators
must read the recovery `tracing::info!` line or walk `kv-quarantine/`.

---

## 6. Telemetry

### `/metrics` counters that ARE emitted

`hf2q_pool_kv_spills_total{repo,quant,outcome}` and
`hf2q_pool_kv_restores_total{repo,quant,outcome}` — emitted by the
shared ADR-005 Phase 4 `KvSpillCounters`
(`src/serve/api/state.rs:206-246`). Call sites:
`src/serve/multi_model.rs:1090-1095` (evict),
`src/serve/multi_model.rs:1188-1204` (admit).

Outcome cardinality (`src/serve/api/state.rs:200-204`): `success`,
`codec_err`, `io_err`, `parity_fail`. `Skipped` does NOT increment
(`state.rs:259-267`, `state.rs:299-301`) — a stub-registered family
leaves counters at 0 even though triggers fired.

### `/metrics` counters that are NOT emitted

`[NOT YET IMPLEMENTED — ADR-017 §R-F7]` Specified but absent (zero
emit sites in `src/`): `hf2q_kv_cache_bytes_on_disk{model_fingerprint}`,
`hf2q_kv_cache_blocks_total{model_fingerprint}`,
`hf2q_kv_quarantined_total{reason}`, `hf2q_kv_cache_evictions_total`,
`Server-Timing: kv_spill=NNNms, kv_restore=NNNms` response header.

### tracing fields the operator should grep stderr for

| target | event | source line | what it tells you |
|--------|-------|-------------|-------------------|
| (default) | `ADR-017 C.1: kv-persist recovery scan complete` | `src/serve/mod.rs:1756` | startup recovery results: `blocks_indexed`, `blocks_quarantined`, `bytes_indexed`, `elapsed_ms` |
| (default) | `ADR-017 C.1: registered StubGemma4Spill for operator --model` | `src/serve/mod.rs:1820-1824` | the C.1 stub is registered (functionally inert until factory substitutes) |
| (default) | `ADR-017 B-dense.2: registered Gemma4DenseSpillFactory ...` | `src/serve/mod.rs:1870-1875` | the lazy factory is registered; substitution happens at first load |
| (default) | `ADR-017 C.1: kv-persist spiller substrate wired into HotSwapManager` | `src/serve/mod.rs:1914-1917` | substrate replaced AppState's NoopKvSpiller-backed manager |
| `hf2q::kv_persist::writer` | `kv_persist writer: write_block_sync failed; continuing` | `src/serve/kv_persist/writer.rs:187-192` | a single block write failed; fields: `error`, `block_hash`. Writer continues with next job. |

There are no `tracing` emissions from `loader_wrapper.rs`, `registry.rs`,
or `families/gemma4_dense.rs` at this commit (verified by grep).

### Interpreting counters

Steady-state with cache hits: `kv_restores_total{outcome="success"}`
rises every readmit cycle; `kv_spills_total{outcome="success"}` rises
every LRU eviction.

Cache-poisoning signals:
- `parity_fail` on `kv_restores` ⇒ body-hash mismatch on read
  (`spiller.rs:437-442` maps any `io::Error` from `read_block` to
  `ParityFail`). Check `kv-quarantine/` for `bodyhash__` files.
- `codec_err` on `kv_restores` ⇒ per-family codec rejection (e.g.
  payload magic mismatch; `families/gemma4_dense.rs:122-125`,
  `PAYLOAD_MAGIC = b"G4D1"`).
- `io_err` on `kv_spills` ⇒ writer back-pressure short-circuit
  (`spiller.rs:370-375`); channel filled up
  (`DEFAULT_CHANNEL_CAPACITY = 8`, `writer.rs:218`).

---

## 7. Common operational scenarios

### a. Long-running serve restart (cache survives across restart)

Recovery scan reads JSON headers only (no body decode) at startup
(`src/serve/kv_persist/recovery.rs:180-204`); body integrity is verified
lazily on read (`recovery.rs:11-14`). Result logged at
`src/serve/mod.rs:1750-1757`. ADR §R-F8 SLA: <5s for ~50K blocks /
12.8 GiB. Single-threaded walk over `models/<slug>/kv/<fanout>/*.safetensors`
(`recovery.rs:121-146`).

### b. Hot-swap evict-then-readmit

`pre_evict` fires BEFORE the engine Arc drops
(`src/serve/multi_model.rs:1083-1097`); `post_admit` fires AFTER engine
load but BEFORE Arc publication (`multi_model.rs:1182-1209`). Restore
happens synchronously and completes before the first post-readmit
request sees a populated cache (`src/serve/kv_persist/spiller.rs:398-477`).

With the C.1 stub registered, the readmit cycle increments NEITHER
counter (stub returns `None`/`Skipped`). The factory-substituted real
`Gemma4DenseSpill` is gated on `Arc::downcast` against
`Arc<EngineHandle>` per `registry.rs:118-141`. Verify via counters AND
the Phase D coherence test (`scripts/adr017_phase_d.sh`).

### c. Disk pressure

The writer thread uses a `mpsc::sync_channel(8)` (`writer.rs:60-61`,
`writer.rs:218`). Inference-thread enqueue uses non-blocking `try_send`
(`writer.rs:84-89`); on `TrySendError::Full`, the spiller short-circuits
and returns `SpillOutcome::Error(IoErr)` (`spiller.rs:370-375`). The
inference path is never blocked by disk back-pressure.

The original 2026-05-01 implementation had no on-disk byte budget. That gap
has since landed and been promoted: `--kv-persist-budget <SIZE>` (or
`[serve] kv_persist_budget`) is wired to both persistent stores and enforced
with post-write LRU eviction. Unset or `0` means unlimited. A malformed value
fails before serving; setup also exposes `--serve-kv-persist-budget`. The CLI
budget requires `--kv-persist PATH`; a configured budget remains dormant on
serve invocations that do not enable a persistent store.

This disk limit is unrelated to `serve --kv-cache-budget <SIZE>`. The latter
governs the shared in-memory KV residency of active slots; it does not cap the
persistent-cache directory.

### d. System crash mid-write (`kill -9` proof)

The atomicity proof is `tests/kv_persist_writer_kill_minus_9.rs:46-89`:
- `format::write_envelope` writes to `<path>.tmp.<pid>` then atomically
  `fs::rename`s into place (per the contract in
  `src/serve/kv_persist/format.rs:25-30`).
- A `SIGKILL` mid-write leaves only `*.tmp.<pid>` orphans plus
  fully-committed `*.safetensors` files; never a partially-named final
  file (test asserts this at lines 81-87).
- The recovery scan ignores `*.tmp.*` files but counts them in
  `RecoveryReport::partial_tmp_files_ignored`
  (`src/serve/kv_persist/recovery.rs:50-52`, `recovery.rs:167-171`).

Operator-visible behavior after a crash:
1. `cmd_serve` restarts cleanly. The recovery scan log line
   (`src/serve/mod.rs:1750-1757`) shows `partial_tmp_files_ignored > 0`
   if there were in-flight writes when the crash hit.
2. No data loss for committed blocks; in-flight blocks are lost
   (best-effort spill, per `multi_model.rs:707` "spill is best-effort,
   not a precondition").
3. `*.tmp.<pid>` orphans are NOT cleaned up by recovery — they accumulate
   across crashes. Manual cleanup:
   `find <PATH>/models -name '*.tmp.*' -delete`.

---

## 8. Sizing guidance

The persistent-store budget is `--kv-persist-budget <SIZE>` or
`[serve] kv_persist_budget` (unset or `0` is uncapped). It is enforced after
successful writes through LRU eviction. It must not be confused with
`serve --kv-cache-budget`, which sizes shared in-memory KV residency for active
conversations.

**Per-block size ceiling:** `MAX_BLOCK_BYTES = 256 * 1024 * 1024`
(256 MiB) — `src/serve/kv_persist/block_store.rs:62`. Writes whose body
exceeds this are refused with `ErrorKind::InvalidInput`
(`block_store.rs:218-225`).

**Block tokens:** `BLOCK_TOKENS = 256` —
`src/serve/kv_persist/format.rs:65`. Per ADR-017 §D3 (oMLX
`scheduler.py:321-331`).

**Per-block size estimate (Gemma 4 dense, F32):**
- `MAX_BLOCK_BYTES` justification comment, `block_store.rs:54-62`:
  256 tokens × 64 layers × 8 KV heads × 128 dim × 2 (K + V) × 4 bytes ≈
  128 MiB ceiling for dense F32. The 256 MiB compile-time ceiling leaves
  headroom.
- For F16: halve all the above → ~64 MiB per block per layer-set.

The placeholder shape config at `src/serve/mod.rs:1857-1866`
(`nkv_heads: [8, 2]`, `head_dim: [256, 512]`, two layers) is NOT
representative of the real Gemma 4 26B (64 layers).

**Default budget recommendation:** ADR-017 §R-F5 says 10% of unified RAM
(12.8 GiB on 128 GiB M5 Max,
`docs/adr/diary/ADR-017-persistent-block-prefix-cache.md:334`). Operator must
manually run `du -sh <PATH>/models` and `rm -rf` periodically until the
budget knob lands.

---

## 9. Troubleshooting

### a. `/readyz` timeout on startup

Suspect a slow recovery scan over a large cache directory. Check the
`tracing::info!` line at `src/serve/mod.rs:1750-1757` — `elapsed_ms`
field. If > 5000 ms on warm SSD, the cache is over-large or under disk
pressure (competing process I/O). Reproduce with a clean process audit
per `feedback_bench_process_audit`.

The recovery scan is single-threaded
(`src/serve/kv_persist/recovery.rs:121-146`); a 50K-block directory on a
hot SSD takes single-digit seconds (per ADR §R-F8 SLA).

### b. Restore latency higher than expected

ADR §R2 risk register
(`docs/adr/diary/ADR-017-persistent-block-prefix-cache.md:528`): SSD restore
latency dominates at 32K context. Computed budget: 5 GB/s NVMe × 670 MB
≈ 130 ms. If observed restore wall is far higher:

1. Process audit. `mcp-brain-server`, `llama-server`, `ollama` all
   contend on M5 Max — see `feedback_bench_process_audit`.
2. Run the Phase D R-P4 ship-gate via `scripts/adr017_phase_d.sh
   --prefill 32768`. The harness fails fast on contamination
   (`scripts/adr017_phase_d.sh:106-128`).
3. Cross-check disk read throughput out-of-band:
   `dd if=<PATH>/models/<fp>/kv/<hex0>/<one-block> of=/dev/null bs=1M`.

### c. Decode tokens diverge after eviction (R-C4 internal failure)

The Phase D test
`tests/kv_persist_gemma4_roundtrip.rs::kv_persist_phase_d_coherence_e2e`
(driven by `scripts/adr017_phase_d.sh`) asserts byte-exact equality
between never-evicted and evict-then-restore decoded tokens. If
divergence appears in production:

1. Capture baseline decode (no `--kv-persist`) and post-restore decode
   (with `--kv-persist`, forced evict-readmit cycle) for the canonical
   sourdough fixture; diff. Per §R-C4
   (`docs/adr/diary/ADR-017-persistent-block-prefix-cache.md:354`), bytes MUST
   match under `HF2Q_USE_DENSE=1`.
2. Check `kv-quarantine/`. A `parity_fail` bump on `kv_restores_total`
   means the spiller already caught it (`spiller.rs:437-442`); past
   that, divergence is a per-family codec bug
   (`families/gemma4_dense.rs`).
3. Counters say `success` but tokens differ → per-family codec is
   silently wrong. File an ADR-017 §R-C6 defect (line 358 — silent
   wrong-output is a hard fail).

### d. Writer thread crashed

The worker is engineered NOT to panic on I/O errors
(`src/serve/kv_persist/writer.rs:177-202`); `tracing::warn!` at
`writer.rs:187-192` reports each failure and the worker continues. A
real panic surfaces only via `AsyncWriterHandle::shutdown`'s
`io::Error::other(...)` (`writer.rs:111-132`); cmd_serve never calls
`shutdown()` (handle drops at process exit, `writer.rs:135-146`). Silent
panic signal: enqueues return `TrySendError::Disconnected` (mapped to
`SpillOutcome::Error(IoErr)` per `spiller.rs:370-375`). Watch
`hf2q_pool_kv_spills_total{outcome="io_err"}` ramping while
`outcome="success"` flatlines.

---

## 10. Disabling / removing

### Startup disable

Omit `--kv-persist PATH` from the serve invocation. The explicit flag is the
single authority: shell state cannot silently disable a path shown in the
command line. To change a templated systemd or Kubernetes launch, change its
typed argument and restart the process.

### Mid-flight disable

Not supported as a live toggle. To disable an already-running serve, stop
`cmd_serve`, drop `--kv-persist=PATH`, and restart. True mid-flight disable
(without restart) remains intentionally out of scope.

### Migration off the persistence path

1. Stop `cmd_serve`.
2. (Optional) Archive `<PATH>` for forensics.
3. Remove `--kv-persist=PATH` from the launch command.
4. (Optional) `rm -rf <PATH>` to reclaim disk.

The off-path is byte-identical to the pre-ADR-017 path — `NoopKvSpiller`
default in `AppState::new_for_serve` is restored on next start
(`src/cli.rs:553-557`).

---

## 11. Future operator work — `[NOT YET IMPLEMENTED]` summary

Each entry cites the ADR section. None are wired in source at commit
`7c6c160`.

1. **Typed enable / disable** (§R-F1). **LANDED** — `--kv-persist PATH`
   enables the store and omission disables it. Production serving has no
   environment override. True mid-flight (no-restart) disable remains
   out of scope.
2. **On-disk byte budget** (§R-F5). **LANDED** —
   `--kv-persist-budget` / `[serve] kv_persist_budget` reaches
   `DiskBlockStore` and the Qwen-family disk persistor, and successful writes
   enforce the limit with LRU eviction. `0` / unset remains unlimited and
   malformed explicit values fail. This is not the in-memory
   `--kv-cache-budget` control.
3. **Writer-capacity knob** (`HF2Q_KV_WRITER_CAPACITY`, doc-comment at
   `writer.rs:217`). cmd_serve hard-codes `DEFAULT_CHANNEL_CAPACITY = 8`
   at `src/serve/mod.rs:1779`.
4. **`hf2q cache --kv-namespace ...`** (§R-F6). `src/cli.rs:140-173`
   supports List/Size/Clear over the model-weights cache only; never
   touches `<kv-persist>/models/`.
5. **Cache-side `/metrics` counters** (§R-F7):
   `hf2q_kv_cache_bytes_on_disk`, `hf2q_kv_cache_blocks_total`,
   `hf2q_kv_quarantined_total{reason}`, `hf2q_kv_cache_evictions_total`.
   Zero emit sites in `src/`. Only the shared ADR-005 Phase 4 counters
   are wired.
6. **`Server-Timing` response header** (§R-F7). Toggle plumbed at
   `src/serve/api/state.rs:241-245`; per-request emit not wired.
7. **Real Gemma 4 shape config** at `src/serve/mod.rs:1848-1869` —
   placeholder shape until GGUF-metadata extraction is plumbed through
   the post-load path. Factory substitution downcasts against this
   placeholder.
8. **Phase D ship-gate sign-off** (§R-P4/R-P5/R-P6/R-C4). Harness
   exists (`scripts/adr017_phase_d.sh`,
   `tests/kv_persist_gemma4_roundtrip.rs`); cold-M5-Max sign-off pending.
9. **Hybrid (Qwen 3.5) family** (Phase B-hybrid). Not present in
   `src/serve/kv_persist/families/`.
10. **TQ-active codec family** (Phase B-tq). Not present.
11. **Quarantine bound** (§R-F9). Quarantine dir grows unbounded; the
    "delete oldest when budget exceeded" rule is not enforced.
12. **Mid-flight persistence disable** (§R-O1). Intentionally out of scope;
    change the typed launch arguments and restart. See §10.
