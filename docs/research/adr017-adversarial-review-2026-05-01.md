# ADR-017 kv_persist Substrate — Adversarial Code Review

**Date**: 2026-05-01
**Reviewer**: code-review (adversarial pass)
**Scope**: `/opt/hf2q/src/serve/kv_persist/**`, `/opt/hf2q/src/serve/api/{engine.rs,kv_spill_descriptor.rs}`, `cmd_serve --kv-persist` block in `/opt/hf2q/src/serve/mod.rs`, `--kv-persist` flag in `/opt/hf2q/src/cli.rs`, three `tests/kv_persist_*.rs` integration tests.
**Read-only**: no `cargo`, no test execution. Findings cite file:line + 3-10 line evidence snippet.

---

## §1 Methodology + Scope

For each of the 11 review categories the reviewer:

1. Read every file in scope at full length (eight `kv_persist/*.rs` modules, the `kv_spill_descriptor` shim, the relevant slice of `engine.rs` worker arms, the `cmd_serve --kv-persist` block, and the three integration tests).
2. Hunted *actively* for evidence — did not certify code as "fine" without inspecting it.
3. Reported **only** confirmed defects with file:line + minimal evidence quote. Speculation marked clearly when present.

Foreign concurrent sessions own SoC (ADR-013/015 iter61c bisection, qwen35, iter219, coherence_smoke), so tests were not executed; conclusions are static-analysis-only. Anywhere a finding requires runtime confirmation that note is explicit.

Categories searched: race conditions, error handling, fsync/durability, advisory file lock (cross-process), fingerprint collision, recovery scan completeness, panic safety, memory safety / Arc cycles, surface-area abuse, snapshot/restore byte-exactness, test gaps.

---

## §2 P0 — Ship-blockers

### P0-1 — `write_envelope` does NOT fsync the parent directory after rename

**File**: `/opt/hf2q/src/serve/kv_persist/format.rs:357-366`
**Category**: fsync / durability

```rust
    {
        let mut f = File::create(&tmp_path)?;
        let header_len = header_bytes.len() as u64;
        f.write_all(&header_len.to_le_bytes())?;
        f.write_all(&header_bytes)?;
        f.write_all(body)?;
        f.sync_all()?;
    }

    std::fs::rename(&tmp_path, path)?;
    let total = 8u64 + header_bytes.len() as u64 + body.len() as u64;
    Ok(total)
```

The atomic-publication recipe is `tempfile → fsync(file) → rename → fsync(parent_dir)`. Step 4 is missing. Without an `fsync` on the parent directory's fd after `fs::rename`, a power loss between the `rename` syscall returning and the directory entry being persisted can leave the on-disk state without the renamed file. APFS, ext4 (default `data=ordered`), XFS — all can lose the rename if the parent dir is not synced. The kill-9 test (`tests/kv_persist_writer_kill_minus_9.rs`) only proves SIGKILL atomicity, NOT power-loss durability.

**Reproducer test idea**: subprocess writes N envelopes, pkill -9 the child, then on first observation the parent uses `dd if=/dev/null` style techniques to drop the page cache (`echo 3 > /proc/sys/vm/drop_caches` Linux only — on macOS this is a known limitation; the test is a Linux-CI-only proof). Assert all `final_files` exist after a forced unmount/remount.

**Suggested fix**: after `fs::rename`, open the parent dir with `OpenOptions::new().read(true)` and call `sync_all()` on it (no-op on platforms where it isn't supported).

```rust
std::fs::rename(&tmp_path, path)?;
// Durability: persist the rename's directory entry.
if let Some(parent) = path.parent() {
    if let Ok(dir) = std::fs::File::open(parent) {
        let _ = dir.sync_all(); // best-effort; ignored on platforms that don't support
    }
}
```

---

### P0-2 — No cross-process advisory lock on `cache_dir`; concurrent `cmd_serve --kv-persist=SAME_DIR` corrupts state

**File**: `/opt/hf2q/src/serve/mod.rs:1719-1771` (cmd_serve setup); `/opt/hf2q/src/serve/kv_persist/block_store.rs:100-128` (`DiskBlockStore::new_with_index`)
**Category**: advisory file lock (cross-process)

The per-block `flock(LOCK_EX)` on `<cache_root>/locks/<short>__<hash_prefix>.lock` (`block_store.rs:382-405`) protects two threads writing the SAME block. It does NOT protect two `cmd_serve` instances opening the same cache_dir. Each instance:

```rust
        let store = Arc::new(
            DiskBlockStore::new_with_index(cache_dir.clone(), recovered_index, 0)
                .with_context(|| {
                    format!(
                        "ADR-017 C.1: DiskBlockStore::new_with_index({})",
                        cache_dir.display()
                    )
                })?,
        );
        let writer = Arc::new(AsyncWriterHandle::spawn(
            Arc::clone(&store),
            DEFAULT_CHANNEL_CAPACITY,
        ));
```

— races to scan + write into the SAME `models/<short>/kv/<hex0>/<hash>.safetensors` tree, with two distinct `BlockIndex` in-memory views. Result: one process's `evict_lru_until_under_budget` will `fs::remove_file` blocks the other process believes are still in its index; budget accounting becomes inconsistent across processes; partially-written `.tmp.<pid>` files multiply.

oMLX `paged_ssd_cache.py` (the cited reference for the envelope format) holds a `flock(LOCK_EX)` on the cache_root dir for the lifetime of the process; this is a documented oMLX invariant the hf2q port silently dropped.

**Reproducer test idea**: spawn two `cmd_serve --kv-persist=SAME_DIR` subprocesses; assert one fails fast with "another instance owns this cache_dir" rather than both starting silently.

**Suggested fix**: in `DiskBlockStore::new_with_index`, acquire `flock(LOCK_EX | LOCK_NB)` on `<cache_root>/.lock` for the lifetime of the store; refuse with a structured error on contention. Lifetime-bound the fd inside `DiskBlockStore` so dropping the store releases the lock.

---

### P0-3 — Recovery scan does NOT garbage-collect orphan `*.tmp.<pid>` files

**File**: `/opt/hf2q/src/serve/kv_persist/recovery.rs:155-205` (scan_one); `/opt/hf2q/src/serve/kv_persist/index.rs:243-246` (also has the issue)
**Category**: recovery scan completeness

```rust
    // Atomic-rename leftovers from a prior crashed write (§D5 + §D8).
    if name.contains(".tmp.") {
        report.partial_tmp_files_ignored += 1;
        return Ok(());
    }
```

`.tmp.<pid>` files are *counted* but never *removed*. After many SIGKILLs the cache directory accumulates partial garbage (each block body up to 256 MiB per `MAX_BLOCK_BYTES`). Combined with `budget_bytes = 0` hardcoded in `serve/mod.rs:1764` (P1-3 below), the disk fills indefinitely. On a long-running operator deployment this is a slow-burn outage.

The kill-9 test (`tests/kv_persist_writer_kill_minus_9.rs:81-87`) explicitly *tolerates* tmp files surviving:

```rust
    if final_files.is_empty() {
        eprintln!("[kv-kill9] kill before any rename committed ({} tmp); skip", tmp_files.len());
        let _ = std::fs::remove_dir_all(&cache_root); return;
    }
```

so the issue is *known and currently unhandled*. Spec §D5+§D8 say "ignore on scan" — but ignoring means orphaning forever absent a janitor.

**Reproducer test idea**: write 5 valid blocks + drop 3 `*.tmp.<pid>` orphans whose pids are NOT alive (use `pid 0xDEADBEEF` style sentinel). Run recovery scan twice. Assert: after the second scan the orphans whose pids do not exist on the system are removed.

**Suggested fix**: in `scan_one`, when a `.tmp.<pid>` file is encountered, parse the `<pid>`. If the pid is not a live process AND the file's `mtime` is older than e.g. 5 minutes, `fs::remove_file` it. Increment a separate `report.orphan_tmp_files_collected` counter. Conservative: only remove if BOTH conditions hold so we never disturb a live writer.

---

## §3 P1 — Should-fix-this-iter

### P1-1 — Race window: `try_substitute_on_load` updates registry, then `update_spiller_registration` updates spiller — NOT atomic

**File**: `/opt/hf2q/src/serve/kv_persist/loader_wrapper.rs:309-324`
**Category**: race conditions

```rust
        let dyn_view_for_substitute: Arc<dyn Any + Send + Sync> = Arc::clone(&dyn_view);
        if let Some(kv_hook) =
            self.registry
                .try_substitute_on_load(&repo, quant, dyn_view_for_substitute)
        {
            self.update_spiller_registration(&repo, quant, kv_hook);
        }

        // 3b. The C.1 bind_for flow always fires after the optional
        //     B-dense.2 substitution — for the auto-Arc<E> path, the
        //     stub's bind_engine is a silent no-op; for the explicit
        //     Arc<EngineHandle> path (which cmd_serve drives via a
        //     SEPARATE post-load bind, not this auto-call),
```

Between the registry's `try_substitute_on_load` returning `Some(kv_hook)` (registry is now LIVE-with-real-hook) and `update_spiller_registration` running (spiller still has the stub), a concurrent `pre_evict` from another tokio task can read the spiller's STUB hook while the registry already returns the REAL hook. Since `BlockPrefixCacheSpiller::pre_evict` reads only the spiller's own registration table (`spiller.rs:289-385`), the stale stub's `Skipped` semantic fires: a snapshot opportunity is silently lost.

The module docs at `registry.rs:84-105` claim "atomic" substitution but only the registry write is atomic; the spiller-side write happens in a separate critical section. Production cmd_serve startup is single-threaded so the window doesn't fire there, but the C.1 wiring permits multiple `LoaderWrapper::load` calls (HotSwap evict→readmit) which can run from concurrent tokio tasks (per `multi_model.rs:887-892` cited in the registry module docs).

**Reproducer test idea**: synthesize two parallel `LoaderWrapper::load` calls against the same `(repo, quant)` factory. Inject a spy `KvCacheSpill` mock on the spiller side. Assert: never observe a load whose registry side is real but spiller side is stub.

**Suggested fix**: extend `try_substitute_on_load` to take an `Option<&BlockPrefixCacheSpiller<E>>` and update both maps under a single critical section, OR add a global `RwLock` around the (registry, spiller) pair so the substitute is atomic across both.

---

### P1-2 — `family_model_fp` uses empty strings for provenance; **collision risk across hf2q versions**

**File**: `/opt/hf2q/src/serve/kv_persist/spiller.rs:242-244`
**Category**: fingerprint collision risk

```rust
    fn family_model_fp(repo: &str, quant: QuantType) -> ModelFingerprint {
        compute_model_fingerprint(repo, quant.as_str(), "", "", "")
    }
```

The `format::compute_model_fingerprint` API takes `(repo_id, quant, producer_version, source_sha256, tokenizer_chat_template)`. The spiller passes `""` for the latter three. So `repo="acme/m1", quant=Q4_K_M` always hashes to the same fingerprint regardless of hf2q producer_version, GGUF source_sha256, or tokenizer template.

Effect: an operator who upgrades hf2q (which changes the `producer_version` baked into freshly-quantized GGUFs) and points `--kv-persist` at the SAME cache_dir will have the new hf2q's KV blocks land in the SAME `models/<short>/` namespace as the old one. A `post_admit` will happily `restore_block` blocks produced by the old codec into a buffer the new codec allocated — codec-version mismatch is checked (per `gemma4_dense.rs:550-575`) but a producer_version change without a codec_version bump is invisible.

The format-layer envelope DOES record `header.format_version` and `header.codec_version`, so a true binary mismatch is caught; but a *semantically* changed payload at the same `codec_version` (e.g. a bugfix in the dense_q codec that flips byte order without bumping the version) silently corrupts.

**Reproducer test idea**: register two distinct `producer_version` strings under the same `(repo, quant)`. Assert: two distinct fingerprints, two distinct on-disk subtrees, no cross-contamination.

**Suggested fix**: thread the live model's GGUF metadata (`hf2q.producer_version`, `hf2q.source_sha256`, `tokenizer.chat_template`) through to `family_model_fp`. The spiller already has the `LoadedHandle` in `pre_evict`; extend `LoadedHandle` (or add a sibling type) to carry these three strings. The B-dense.1 task list already calls this out (spiller.rs:236-244 documents the gap as "B-dense.1 wires the real GGUF metadata path"); document the **interim risk** in operator-facing docs and SHIP-GATE the lift.

---

### P1-3 — Hardcoded `budget_bytes = 0` ⇒ unbounded disk growth

**File**: `/opt/hf2q/src/serve/mod.rs:1759-1771`
**Category**: surface-area abuse

```rust
        // 3. DiskBlockStore — owns the read-path file I/O. The
        //    `budget_bytes = 0` argument disables the on-disk LRU
        //    budget (unbounded growth — Phase D wires the budget
        //    against `HF2Q_KV_PERSIST_BUDGET_BYTES`).
        let store = Arc::new(
            DiskBlockStore::new_with_index(cache_dir.clone(), recovered_index, 0)
```

The `evict_lru_until_under_budget` call site in `block_store.rs:313-360` short-circuits when `budget == 0`. Production `cmd_serve --kv-persist=PATH` therefore never evicts. Combined with P0-3 (orphan tmp accumulation), the cache_dir is a one-way write target until the operator manually `rm -rf`s it.

The comment promises "Phase D wires the budget" but Phase D scaffolding has landed (per recent commits) without actually plumbing this. The `HF2Q_KV_PERSIST_BUDGET_BYTES` env var is not read anywhere in the serve startup path (`grep -rn HF2Q_KV_PERSIST_BUDGET /opt/hf2q/src` returns 1 hit — the comment).

**Reproducer test idea**: end-to-end test that writes 1 GiB of blocks under `--kv-persist=PATH` with `HF2Q_KV_PERSIST_BUDGET_BYTES=128MiB`, asserts on-disk size <= budget.

**Suggested fix**: read `HF2Q_KV_PERSIST_BUDGET_BYTES` env in `serve/mod.rs:1759`, default to `0` (current behavior) but log a `WARN` when 0. Alternatively gate kv-persist behind a non-zero budget for production deployments.

---

### P1-4 — `--kv-persist=` accepts arbitrary paths including symlinks; no validation

**File**: `/opt/hf2q/src/cli.rs:565-566`; `/opt/hf2q/src/serve/mod.rs:1733-1738`
**Category**: surface-area abuse

```rust
    #[arg(long = "kv-persist", value_name = "PATH")]
    pub kv_persist_path: Option<PathBuf>,
```

Setup code:

```rust
        std::fs::create_dir_all(cache_dir).with_context(|| {
            format!(
                "ADR-017 C.1: create kv-persist cache dir at {}",
                cache_dir.display()
            )
        })?;
```

Concrete attack/footgun surfaces:

- `--kv-persist=/etc/passwd/..` — `create_dir_all` would surface a permission error (good), but a symlink-pointing-to-`/etc` would silently let writes proceed. There's no `is_symlink` check on the cache_dir.
- `--kv-persist=` (empty string) — `clap` will accept `""`; `Path::new("")` and `create_dir_all("")` returns an io::Error on most platforms — verifiable error path but not friendly.
- `--kv-persist=/tmp/dir-with-symlinks-inside` — once the directory is the cache_root, the recovery scan walks `models/*/kv/*/*` (`recovery.rs:121-145`) using `is_dir()` and `is_file()` which DO follow symlinks. A malicious operator-adjacent process that creates `<cache_root>/models/foo/kv/0/symlink-to-/etc/shadow.safetensors` would cause the recovery scan to attempt `read_envelope_header` against a privileged file, surfacing IO errors at minimum and quarantine-move attempts at worst (which would invoke `fs::rename` on the symlink).

**Reproducer test idea**: drop a symlink under `<cache_root>/models/<slug>/kv/0/poison.safetensors` pointing to `/dev/null`; run recovery; assert no quarantine attempt was made (or a structured error was returned).

**Suggested fix**: in `serve/mod.rs:1733`, validate the cache_dir path:
1. `cache_dir.canonicalize()` to resolve symlinks (refuse if canonical path differs from input + warn).
2. Refuse if the canonical path starts with `/etc`, `/proc`, `/sys`, `/dev`, `/`.
3. Refuse if the path is empty.
4. In `recovery.rs:scan_one`, use `fs::symlink_metadata` (does not follow) and skip files that are symlinks.

---

### P1-5 — `RwLock`/`Mutex` poison panic in production paths

**File**: multiple — `/opt/hf2q/src/serve/kv_persist/index.rs:125,134,140,148,159,165,173`; `/opt/hf2q/src/serve/kv_persist/registry.rs:188,201,212,239,253,281,291,300,341,352,365`; `/opt/hf2q/src/serve/kv_persist/loader_wrapper.rs:167,182,194,204,243,275`
**Category**: error handling

Pervasive use of `.expect("... RwLock poisoned")` and `.expect("... Mutex poisoned")` in production paths (NOT under `#[cfg(test)]`). Example:

```rust
    pub fn insert(&self, meta: BlockMeta) {
        let mut guard = self.inner.write().expect("BlockIndex inner RwLock poisoned");
        guard.insert(meta.hash, meta);
    }
```

```rust
    pub fn register(&self, repo: String, quant: QuantType, hook: Arc<dyn EngineBindable>) {
        let mut g = self
            .hooks
            .write()
            .expect("KvPersistRegistry::hooks RwLock poisoned");
        g.insert((repo, quant.as_str()), hook);
    }
```

A panic anywhere inside `KvCacheSpill::snapshot_block` / `restore_block` while holding any of these locks (e.g. an MlxBuffer alloc OOM that itself panics rather than returning Err) poisons the lock and brings down EVERY subsequent call to that lock — even though the spiller is supposed to be best-effort and not affect the inference-path. Cascading panic of the entire `cmd_serve` process.

`spiller.rs:309-316` does match the lock result correctly:

```rust
        let alignment = {
            let g = match hook_arc.lock() {
                Ok(g) => g,
                Err(_) => return SpillOutcome::Error(SpillErrorKind::CodecErr),
            };
            g.block_alignment()
        };
```

— so the *spiller* recovers. The registry, index, and loader_wrapper do not. Inconsistent discipline.

**Reproducer test idea**: in a test, inject a panic inside `BlockIndex::insert` (e.g. via a `Drop` panic in a test-only `BlockMeta` field). Then call any other `BlockIndex` method. Assert: the second call returns an error, NOT panics.

**Suggested fix**: replace every `.expect("... poisoned")` with `match guard { Ok(g) => ..., Err(poisoned) => poisoned.into_inner() }` or with a structured error return. Defensible policy: poison-recovery is preferred over poison-cascade at the registry / index layer because the registry's invariant (a `HashMap`) survives panic.

---

### P1-6 — Writer thread panic kills the cache silently; `Drop` `join()` is fire-and-forget

**File**: `/opt/hf2q/src/serve/kv_persist/writer.rs:135-146`
**Category**: panic safety

```rust
impl Drop for AsyncWriterHandle {
    fn drop(&mut self) {
        // Best-effort drop: close the channel; let the worker drain.
        // If the worker is still running, we don't block in Drop.
        // Production callers should call `shutdown()` explicitly.
        self.tx.take();
        if let Some(jh) = self.join_handle.take() {
            // Don't block forever in Drop; spawn a detacher.
            let _ = jh.join();
        }
    }
}
```

The Drop's comment says "production callers should call `shutdown()` explicitly". In `serve/mod.rs:1717-1922`, the `kv_persist_loader_wrapper` is created and its inner `Arc<AsyncWriterHandle>` is captured inside the `BlockPrefixCacheSpiller` and the `HotSwapManager`. **Nowhere in `cmd_serve` does anything call `writer.shutdown()`**. The writer is only dropped when the manager is dropped (i.e. process exit), at which point the `Drop` impl swallows the join error. A panic in the writer thread (e.g. a sha256-related `index OOB`) is invisible to the operator AND any pending `WriteJob`s in flight at the moment of the panic are silently dropped (the completion_tx fires `Err(...)` but only if the worker reaches the per-job error path).

Worse: the worker_loop has no `catch_unwind` (`writer.rs:151-175`):

```rust
fn worker_loop(store: Arc<DiskBlockStore>, rx: mpsc::Receiver<WriteJob>) {
    loop {
        let job = match rx.recv() {
            Ok(j) => j,
            Err(RecvError) => break,
        };
        process_job(&store, job);
        ...
```

If `process_job` panics (e.g. a poisoned RwLock inside `BlockIndex::insert` caused by P1-5), the worker thread dies. Subsequent `enqueue` calls succeed (the channel is still open) until the channel fills, at which point every `try_send` returns `Full` forever (since no consumer drains it). The spiller will then short-circuit to `IoErr` on every `pre_evict`. The cache silently de-grades.

**Reproducer test idea**: inject a job whose `header.format_version != CURRENT_FORMAT_VERSION.0`; OR use a test-only flag that makes `index.insert` panic. Submit 4 more valid jobs after. Assert: `enqueue` for jobs 2-5 either fails fast OR the writer is restarted. Today: jobs 2-5 silently disappear.

**Suggested fix**: wrap the worker loop body in `std::panic::catch_unwind`. On unwind, log + restart the worker (re-spawn with the same `Arc<DiskBlockStore>` + new channel). Alternatively, store the panic payload in an `Arc<Mutex<Option<Box<dyn Any+Send>>>>` slot on `AsyncWriterHandle` and surface it via a `health_check()` method that `cmd_serve` polls.

---

### P1-7 — `BlockHash::short_hex` (used as directory short-name) is 16 hex chars = 64 bits = real collision space

**File**: `/opt/hf2q/src/serve/kv_persist/format.rs:170-176`
**Category**: fingerprint collision risk

```rust
impl ModelFingerprint {
    /// First 16 hex characters — used as the directory short-name in the
    /// hex-fanout layout (`<root>/models/<short>/kv/...`).
    pub fn short_hex(&self) -> String {
        hex::encode(&self.0[..8])
    }
}
```

64-bit truncation. Birthday-paradox collision at √(2^64) ≈ 4.3 billion fingerprints. In practice this is fine for any one operator (no one has 4 billion models), BUT the short_hex is also used as the **lock-file bucket name** in `block_store.rs:200-204`:

```rust
        self.cache_root.join("locks").join(format!(
            "{}__{}.lock",
            model_fp.short_hex(),
            prefix
        ))
```

So two distinct ModelFingerprints with colliding short_hex would share a flock bucket — minor (over-serialization, not corruption) — AND share a `models/<short>/kv/...` subtree, which IS corruption: blocks for two different models cohabit one directory. Since `BlockMeta.model_fp` is the full 32-byte fingerprint, the in-memory index disambiguates correctly; but the on-disk filesystem layout cannot.

This is a *correctness* concern only at scale. Document the 64-bit short-name truncation in the spec OR widen to e.g. 24 hex chars (96 bits, room for 2^48 models).

**Reproducer test idea**: mock `compute_model_fingerprint` to return two ModelFingerprints that share their first 8 bytes. Write a block from each. Recover. Assert: index correctly attributes blocks to the right model (currently it would because lookup is by full BlockHash, which is unique by construction; but if BlockHashes collide, the disambiguation breaks).

**Suggested fix**: bump `short_hex` to 24 hex chars OR document the 64-bit ceiling explicitly in the format-version invariants (so a future format version can extend without surprise).

---

### P1-8 — `--kv-persist` startup proceeds even if `recover_from_disk` produces zero blocks AND every block was quarantined (silent disaster recovery)

**File**: `/opt/hf2q/src/serve/mod.rs:1743-1757`
**Category**: surface-area abuse / observability

```rust
        let (recovered_index, recovery_report) =
            recover_from_disk(cache_dir).with_context(|| {
                format!(
                    "ADR-017 C.1: recover_from_disk({})",
                    cache_dir.display()
                )
            })?;
        tracing::info!(
            cache_dir = %cache_dir.display(),
            blocks_indexed = recovery_report.blocks_indexed,
            blocks_quarantined = recovery_report.blocks_quarantined,
            bytes_indexed = recovery_report.bytes_indexed,
            elapsed_ms = recovery_report.elapsed_ms,
            "ADR-017 C.1: kv-persist recovery scan complete"
        );
```

Recovery report is logged at INFO level only. If 100% of blocks quarantine (e.g. a botched format-version bump that flipped every existing envelope to an unsupported version), the operator gets a one-line INFO that 0 indexed / N quarantined — easily missed in a busy log. Service starts up fine.

**Suggested fix**: emit `tracing::warn!` when `blocks_quarantined > 0`. Refuse startup (return Err) when `blocks_quarantined > 0 AND blocks_indexed == 0` UNLESS an explicit `--kv-persist-allow-empty-recovery` flag is set.

---

### P1-9 — `recovery.rs:bytes_quarantined` is read AFTER the rename can fail mid-recovery (not BEFORE the move)

**File**: `/opt/hf2q/src/serve/kv_persist/recovery.rs:174-194`
**Category**: error handling

```rust
    // Capture the file size BEFORE any potential move so the quarantine
    // bytes accounting is correct even if the rename succeeds.
    let file_bytes = match fs::metadata(blk_path) {
        Ok(m) => m.len(),
        Err(_) => 0,
    };

    let header = match format::read_envelope_header(blk_path) {
        Ok(h) => h,
        Err(_) => {
            quarantine_with_prefix(slug_path, blk_path, QuarantineReason::TruncatedHeader)?;
            report.blocks_quarantined += 1;
            report.bytes_quarantined = report.bytes_quarantined.saturating_add(file_bytes);
            return Ok(());
        }
    };
```

The comment promises capture-before-move. But if `quarantine_with_prefix` fails (e.g. permission denied on creating `kv-quarantine/`), the `?` propagates the error and the per-file scan errors out. That's fine for the file. But the OUTER scan (`recover_from_disk`) treats this as an unrecoverable I/O on the cache root and fails the entire startup. A single permission anomaly inside `kv-quarantine/` aborts cmd_serve.

**Suggested fix**: wrap `quarantine_with_prefix` errors at the per-file boundary; log and skip the quarantine attempt rather than abort the whole scan.

---

### P1-10 — `engine.rs:request_kv_snapshot` does not bound the `try_send → blocking_send` fallback, can deadlock spiller

**File**: `/opt/hf2q/src/serve/api/engine.rs:1712-1727`
**Category**: race conditions / deadlock

```rust
        match self.inner.tx.try_send(req) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(req)) => {
                // Queue full — do a blocking send so the request is
                // eventually delivered. Acceptable because KV snapshot
                // is rare (eviction-time only) and the FIFO already
                // forces serialization.
                self.inner
                    .tx
                    .blocking_send(req)
                    .context("engine worker is gone (kv_snapshot full→blocking)")?;
            }
```

`request_kv_snapshot` is called from `Gemma4DenseSpill::snapshot_block` (`gemma4_dense.rs:830-891`) which is called from `BlockPrefixCacheSpiller::pre_evict` which is called from inside `HotSwapManager::evict`. The manager holds a write lock while evicting (per `multi_model.rs:887-892` — cited in the registry module docs). If the engine's inbound channel is saturated by inference-path requests and `try_send` returns `Full`, the spiller blocks indefinitely with the manager's evict lock held — every other inference request blocks behind the evict.

The `eventually delivered` claim assumes the inference workload drains the engine's inbound; under sustained load that may never be true.

**Suggested fix**: replace `blocking_send` with `tx.send_timeout(req, Duration::from_secs(N))` and surface a timeout as `Ok(None)` (Skipped semantic) so the eviction proceeds without spilling. Document the SLA in `request_kv_snapshot`'s rustdoc.

---

### P1-11 — `KvSpillDescriptor::max_decode_tokens` is hardcoded `512usize` at engine spawn, no env binding

**File**: `/opt/hf2q/src/serve/api/engine.rs:1516-1527`
**Category**: configuration drift / snapshot/restore byte-exactness

```rust
                    // Static spill-side budget. Mirrors `SamplingParams`
                    // default (`max_tokens: 512`) — the hook's
                    // restore-before-prefill path uses this to seed
                    // full-attention layer linear capacity, then
                    // `forward_prefill.rs:274-285` reallocates to
                    // `seq_len + max_tokens` per request.
                    let max_decode_tokens = 512usize;
                    Some(super::kv_spill_descriptor::KvSpillDescriptor::from_gemma_loaded_model(
                        &g.weights,
                        max_decode_tokens,
                        kv_dtype,
                    ))
```

If a request comes in with `max_tokens > 512`, the `forward_prefill` allocator will use the request's `max_tokens` (per its own logic), but the spilled snapshot recorded `capacity = seq_len + 512`. On restore (different process / different request), `validate_header` (`gemma4_dense.rs:550-575`) accepts capacity drift only for non-sliding layers (`gemma4_dense.rs:1175-1180`):

```rust
        if hdr.is_sliding {
            if layer.capacity != hdr.capacity as usize {
                return Err(SpillErrorKind::CodecErr);
            }
        } else if (layer.capacity as u32) < hdr.range_end {
            return Err(SpillErrorKind::CodecErr);
        }
```

The sliding-layer arm is strict-equal; if the next request has a different `max_tokens` budget, the *sliding capacity* (= `sliding_window`) is invariant so this is fine. But for full-attn, the new request's `seq_len + max_tokens` could be < the spilled `capacity`, causing CodecErr. Operator-visible failure mode that "evict + readmit + serve a smaller-prefill request" rejects the cached state.

**Suggested fix**: read `HF2Q_KV_PERSIST_MAX_DECODE_TOKENS` env, default to a higher ceiling (e.g. 8192). Document that the descriptor's `max_decode_tokens` is a UPPER BOUND, not a per-request value. Or: treat the on-disk capacity field as a HINT and re-allocate buffers to fit the live request.

---

## §4 P2 — Nits / Future

- **P2-1** `format.rs:441-466`: `read_envelope_body` documents that it asserts `sha256(body) == header.block_hash`, but this is enforced ONLY at read time. The writer (`write_envelope`, `format.rs:326-369`) does NOT verify the body sha matches the header's recorded `block_hash` BEFORE writing. A spiller bug that produces a body whose sha mismatches the chain-hash would land on disk and only be caught on a future read.
- **P2-2** `block_store.rs:175-184` (`block_path`): hex-fanout uses `&hex[..1]` — single hex char → 16 fanout buckets per model. Plenty for thousands of blocks per model, but at ~1M blocks per model the per-bucket directory entry count becomes a perf concern (most filesystems degrade past 10k entries). Note for future scale.
- **P2-3** `format.rs:386-390`: `read_envelope_header_from_file` rejects header_len > 64 MiB — sensible ceiling, but the constant is inline-magic rather than a named const.
- **P2-4** `index.rs:140-155`: `iter_by_model` walks the entire HashMap every call. O(N) per restore. For a model with 100k blocks, this is a measurable hit on `post_admit` startup. Recommend: add a per-model index (`HashMap<ModelFingerprint, Vec<BlockHash>>`) updated on insert/remove.
- **P2-5** `recovery.rs:281-294` (`quarantine`): cross-fs fallback `fs::copy + fs::remove_file` is non-atomic; if `remove_file` fails the source is duplicated. Tolerated per spec but worth a TODO.
- **P2-6** `loader_wrapper.rs:333-340`: `Arc::try_unwrap` failure surfaces as `anyhow!("...retained a clone...")`. Bare anyhow loses structured error info — a downstream caller can't programmatically distinguish a "retained" failure from a real loader failure.
- **P2-7** `spiller.rs:485-492` (`parse_layer_rank`): silent `.unwrap_or(0)` masks malformed payload_kind strings. A future audit hunting "every block restored to layer 0" will spend time chasing this.
- **P2-8** `format.rs:381-398` (`read_envelope_header_from_file`): does NOT reject `hlen == 0` distinctly from `hlen > 64 MiB` — the error message "out of range (0, 64 MiB]" is fine, but a 0-byte header is corruption-of-a-different-flavor and could route to a more specific quarantine reason.
- **P2-9** `gemma4_dense.rs:1273-1304` (`bind_engine`): the impl tries `Arc::downcast::<Engine>` first, then `Arc::downcast::<EngineHandle>`. If a hypothetical future caller passes `Arc<dyn Any>` carrying neither, the impl silently no-ops. The C.1 contract is "silent no-op on mismatch" so this is correct, but a `tracing::debug!` on the no-op path would help debugging.
- **P2-10** `index.rs:124-127` (`insert`): overwrite semantics document that "fresh mtime/path wins" but the implementation uses `HashMap::insert` which doesn't compare mtimes — just blindly overwrites. If two writers race to write the same block_hash with different bodies (e.g. same chain-hash but distinct codec_versions), the LAST writer wins, not the FRESHEST mtime. Probably fine since chain-hash collision implies content equality; still worth a defensive check in production.
- **P2-11** `serve/mod.rs:1819-1819`: `pool_quant = quant_select::QuantType::Q4_K_M` is hardcoded for the registry registration; the actual loaded GGUF's quant may be Q4_0, Q6_K, etc. Spiller's `pre_evict` looks up by `(repo, parsed_quant_from_handle)` — if the parsed quant differs from the hardcoded `Q4_K_M`, the lookup misses and the spiller silently `Skipped`s. The comment at `mod.rs:1814-1815` admits this gap.
- **P2-12** `recovery.rs` test `recover_from_disk_with_corrupted_blocks_reports_quarantined_count` mixes truncated-header and version-mismatch quarantines but does NOT test what happens when `quarantine_with_prefix` collides on dest filename (e.g. two blocks both quarantined as `trunc__<full>.safetensors` with same `<full>`). `fs::rename` will overwrite atomically — losing the second post-mortem.
- **P2-13** `block_store.rs:382-405` (`AdvisoryLock::acquire`): `flock(LOCK_EX)` is Linux-and-macOS but NOT Windows (the cfg gating is implicit via `unix` cfg flags upstream — not visible here). A `cfg(unix)` guard in this module would surface the platform constraint.

---

## §5 Test gap inventory (production code paths with no test coverage)

| Path | Gap |
|------|-----|
| `format.rs:355-356` — `path.parent().ok_or_else` | No test exercises `write_envelope` with a path lacking a parent (e.g. `Path::new("/")`). |
| `format.rs:366` — `fs::rename` failure | No test for rename failure (e.g. permission denied on parent dir post-tmp-write). |
| `format.rs:386` — `hlen == 0` | No test specifically for `header_len = 0` envelope (covered conflated with version-mismatch). |
| `block_store.rs:286-296` — `remove_block` non-NotFound errors | The test `remove_block_decrements_index_and_deletes_file` covers NotFound tolerance only, not e.g. `EBUSY`. |
| `block_store.rs:386-390` — `AdvisoryLock` create-parent fallthrough | No test for missing `locks/` subdir scenario (covered only by happy-path init). |
| `index.rs:194-219` — `BlockIndex::rebuild_from_disk` non-fanout-dir entries | Test coverage doesn't include garbage files at the `models/<slug>/kv/<not-a-fanout-dir>/` level. |
| `recovery.rs:131` — outer `fs::read_dir` propagation | No test for permission-denied at `<cache_root>/models/`. |
| `recovery.rs:269-279` — quarantine cross-fs fallback | No test for `fs::rename` returning EXDEV (cross-fs); only the happy intra-fs path. |
| `registry.rs:230-244` — `bind_for` race with concurrent `register` | No test where register/bind interleave under contention. The multi-thread test only exercises bind/bind concurrency. |
| `loader_wrapper.rs:333-340` — `Arc::try_unwrap` failure path | Covered by `loader_wrapper_contract_violation_surfaces_error`, but ONLY for a deterministic mock-retains-arc path. No test for a real `Gemma4DenseSpill::bind_engine` keeping the wrong inner Arc by accident. |
| `spiller.rs:368-374` — IoErr branch on writer Full | The `pre_evict_with_writer_full_returns_error_io_err` test acknowledges (in comments lines 802-916) that it cannot exercise the Full branch with the public API. The branch is effectively untested. |
| `spiller.rs:436-442` — ParityFail on read failure | Covered by `post_admit_with_corrupted_block_returns_error_parity_fail`. |
| `spiller.rs:460-468` — IoErr/CodecErr/ParityFail mapping in `post_admit` | Only CodecErr is exercised; IoErr / ParityFail mapping from `restore_block` is not. |
| `writer.rs:179-202` — `process_job` panic path | `worker_does_not_panic_on_io_error` exercises an IO error, NOT a panic. A panic INSIDE `process_job` is uncovered (and per P1-6 would kill the worker). |
| `writer.rs:111-132` — `shutdown` panic propagation | No test for a worker that genuinely panics → `shutdown` returns Err. |
| `gemma4_dense.rs:1453-1508` (`Gemma4DenseSpillFactory::try_construct`) — defensive None branch | The `return None` at line 1503 (neither slot populated) has no test. |
| `serve/mod.rs:1719-1922` — entire `--kv-persist=PATH` block | The kv_persist_harness.rs test is `HF2Q_KV_PERSIST_E2E=1` gated; default test runs do NOT exercise this code path AT ALL. The `cmd_serve_constructs_spiller_when_flag_on_smoke` test in `loader_wrapper.rs` exercises only the wrapper, not the full `cmd_serve` setup including DiskBlockStore + AsyncWriterHandle + recovery. |
| `serve/mod.rs:1983-1985` — `set_pending_bind` arming before `load_or_get` | No test that `set_pending_bind` is actually called before each `load_or_get` (regression risk if a future refactor reorders these). |
| `engine.rs:1693-1731` — `request_kv_snapshot` blocking_send fallback | Default-channel-cap tests don't exercise the saturated path. |
| `engine.rs:2238-2372` — `kv_snapshot_gemma` worker arm full coverage | Tested via `kv_persist_gemma4_roundtrip.rs` E2E only (gated). The synthetic test fixture in `engine.rs:464-667` substitutes a different worker — coverage of the REAL `kv_snapshot_gemma` path is gated behind `HF2Q_KV_PERSIST_E2E=1`. |

**Overall**: The 160/160 PASS claim is technically true but rests heavily on default-off integration tests and on synthetic mocks. Real-GGUF coverage is gate-gated; production wire-up (`cmd_serve --kv-persist=...`) is not exercised by the default test runner.

---

## §6 Verdict

**hold-pending-P0-fix**

The three P0 findings (P0-1 missing parent-dir fsync, P0-2 missing cross-process advisory lock, P0-3 orphan tmp accumulation) are all real durability/safety issues that an operator deploying `cmd_serve --kv-persist=PATH` to production WILL hit:

- P0-1 fires on first power loss after a kernel cache flush window.
- P0-2 fires on operator double-launch (zero-warning silent corruption — most likely failure mode).
- P0-3 fires on slow accumulation; weeks-to-months timescale on a busy serve.

P0-1 and P0-2 are both ~10-line fixes on top of the existing code. P0-3 is a 30-line janitor function added to `recovery.rs`. None require API churn.

The P1 findings (especially P1-3 unbounded budget — also corruption-adjacent — and P1-5 pervasive `expect`-on-poison) compound the P0 risks but do not independently block ship. P1-1 (registry/spiller substitution race) is theoretical-but-real and worth fixing in this iter.

**Recommendation**: gate `cmd_serve --kv-persist=PATH` behind a `--kv-persist-i-know-this-is-experimental` flag until P0-1, P0-2, P0-3 land. Or fix them now — none are deep.

The Phase D scaffolding (operator recipe + R-C4 coherence test + R-P4 perf test) is correct for what it is, but the production-substrate underneath has the durability holes documented above. Phase D's "ship gate" cannot be coherently met until P0-1+P0-2 land.

---

*End of report.*
